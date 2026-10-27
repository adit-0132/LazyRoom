"""ImageToolbox "Geometry" group filters."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ...core.errors import invalid
from ...core.params import Bool, Choice, Color, File, Float, FloatList, Int, Point
from .._util import F32
from ._common import gmic_cmd, gmic_single_thread, itb_op
from . import _geometry_impl as G

GROUP = "geometry"

_BOUNDARY = ["transparent", "nearest", "periodic", "mirror"]
_BOUNDARY_LABELS = {"transparent": "Transparent", "nearest": "Nearest", "periodic": "Periodic", "mirror": "Mirror"}
_INTERP = ["nearest_neighbor", "linear"]
_MIRROR4 = ["none", "x_axis", "y_axis", "both_axes"]


def _boundary(default: str) -> Choice:
    return Choice("boundary", default, choices=_BOUNDARY, labels=_BOUNDARY_LABELS, title="Boundary")


def _interp(default: str = "linear") -> Choice:
    return Choice("interpolation", default, choices=_INTERP, title="Interpolation",
                  labels={"nearest_neighbor": "Nearest neighbor", "linear": "Linear"})


def _transform_params() -> list:
    """Shared field transform of ImageToolbox's procedural shaders."""
    return [
        Float("offset_x", 0.0, min=-1, max=1, step=0.001, title="Offset X"),
        Float("offset_y", 0.0, min=-1, max=1, step=0.001, title="Offset Y"),
        Float("scale", 1.0, min=0.05, max=8, step=0.001, title="Scale"),
        Float("rotation", 0.0, min=-180, max=180, step=0.001, title="Rotation", unit="deg"),
    ]


# ---------------------------------------------------------------------------
# jhlabs transform filters (Rust port in ImageToolboxLibs)
# ---------------------------------------------------------------------------

@itb_op("Arc", "Arc", GROUP, [
    Float("radius", 0.2, min=0, max=1, step=0.01, title="Radius", description="Inner radius (fraction of half the long side)"),
    Float("height", 0.3, min=0, max=1, step=0.01, title="Size", description="Arc thickness (fraction of half the long side)"),
    Float("angle", 0.0, min=0, max=360, step=0.01, title="Angle", unit="deg"),
    Float("spread_angle", 360.0, min=0, max=360, step=0.01, title="Spread angle", unit="deg"),
    Float("center_x", 0.5, min=0, max=1, step=0.01, title="Center X"),
    Float("center_y", 0.5, min=0, max=1, step=0.01, title="Center Y"),
], upstream="jhlabs", alpha="full", tags=["arc", "bend", "polar", "distort"])
def arc(img, ctx, radius, height, angle, spread_angle, center_x, center_y):
    """jhlabs ArcFilter: bends the image into an arc (outside = transparent)."""
    h, w = img.shape[:2]
    half = max(w, h) / 2.0
    r, ah = radius * half, height * half
    ang, spread = math.radians(angle), math.radians(spread_angle)
    cx, cy = center_x * w, center_y * h

    def mapping(X, Y):
        dx, dy = X - cx, Y - cy
        theta = G.gmod(np.arctan2(-dy, -dx) + ang, G.TAU)
        return (w - 1.0) * theta / (spread + 0.00001), h * (1.0 - (np.hypot(dx, dy) - r) / (ah + 0.00001))

    return G.jh_transform(img, mapping, G.EDGE_ZERO)


@itb_op("DistortPerspective", "Distort Perspective", GROUP, [
    Point("top_left", (0.2, 0.0), min=0, max=1, title="Top Left"),
    Point("top_right", (0.8, 0.0), min=0, max=1, title="Top Right"),
    Point("bottom_left", (0.0, 1.0), min=0, max=1, title="Bottom Left"),
    Point("bottom_right", (1.0, 1.0), min=0, max=1, title="Bottom Right"),
    Bool("clip", False, title="Clip", description="Kept for parity; ImageToolbox's native port ignores it for corner mode"),
], upstream="jhlabs", alpha="full", tags=["perspective", "distort", "keystone"])
def distort_perspective(img, ctx, top_left, top_right, bottom_left, bottom_right, clip):
    """jhlabs PerspectiveFilter.setCorners: maps the image onto the given quad (same canvas size)."""
    h, w = img.shape[:2]
    xs = [top_left[0] * w, top_right[0] * w, bottom_right[0] * w, bottom_left[0] * w]
    ys = [top_left[1] * h, top_right[1] * h, bottom_right[1] * h, bottom_left[1] * h]
    a11, a12, a13, a21, a22, a23, a31, a32, a33 = G.perspective_coeffs(xs[0], ys[0], xs[1], ys[1], xs[2], ys[2],
                                                                       xs[3], ys[3])
    A = a22 * a33 - a32 * a23
    B = a31 * a23 - a21 * a33
    C = a21 * a32 - a31 * a22
    D = a32 * a13 - a12 * a33
    E = a11 * a33 - a31 * a13
    Fv = a31 * a12 - a11 * a32
    Gv = a12 * a23 - a22 * a13
    Hv = a21 * a13 - a11 * a23
    Iv = a11 * a22 - a21 * a12
    off_x = float(int(np.float32(min(xs))))
    off_y = float(int(np.float32(min(ys))))

    def mapping(X, Y):
        ox, oy = X + off_x, Y + off_y
        den = Gv * ox + Hv * oy + Iv
        return w * (A * ox + B * oy + C) / den, h * (D * ox + E * oy + Fv) / den

    return G.jh_transform(img, mapping, G.EDGE_RGB_CLAMP)


@itb_op("Feedback", "Feedback", GROUP, [
    Int("distance", 10, min=-100, max=100, title="Distance", unit="px"),
    Int("angle", 0, min=-180, max=180, title="Angle", unit="deg"),
    Int("rotation", 3, min=-180, max=180, title="Rotation", unit="deg"),
    Float("zoom", 0.05, min=-1, max=1, step=0.01, title="Image zoom"),
], upstream="jhlabs", alpha="full", tags=["feedback", "echo", "trail", "repeat"])
def feedback(img, ctx, distance, angle, rotation, zoom):
    """jhlabs FeedbackFilter (3 iterations): transformed copies composited with decreasing opacity."""
    h, w = img.shape[:2]
    had = G.has_alpha(img)
    src = G.ensure_rgba(img)
    out = np.empty((h, w, 4), F32)
    cx, cy = 0.5 * w, 0.5 * h
    dist = ctx.px(distance)
    ang, rot = math.radians(angle), math.radians(rotation)
    dx, dy = dist * math.cos(ang), -dist * math.sin(ang)
    X = np.arange(w, dtype=np.float64)[None, :]
    for y0, y1 in G.bands(h, w):
        Y = np.arange(y0, y1, dtype=np.float64)[:, None]
        band = src[y0:y1].copy()
        for it in range(1, 4):
            scale = 1.0 + zoom * it
            s, c = math.sin(rot * it), math.cos(rot * it)
            with np.errstate(all="ignore"):  # zoom = -1 -> scale 0 (inf/nan coordinates -> outside)
                px = (X - cx - dx * it) / scale
                py = (Y - cy - dy * it) / scale
                mx = G._coords(cx + c * px + s * py, -2.0, w + 1.0, (y1 - y0, w))
                my = G._coords(cy - s * px + c * py, -2.0, h + 1.0, (y1 - y0, w))
            refl = G._remap(src, mx, my, cv2.BORDER_CONSTANT, (0.0, 0.0, 0.0, 0.0))
            sa = refl[..., 3:4] * F32((255 // (it + 1)) / 255.0)
            rem = (F32(1.0) - sa) * band[..., 3:4]
            band[..., :3] = refl[..., :3] * sa + band[..., :3] * rem
            band[..., 3:4] = sa + rem
        out[y0:y1] = band
    return G.finish(out, had)


@itb_op("Kaleidoscope", "Kaleidoscope", GROUP, [
    Int("angle", 0, min=0, max=360, title="Angle",
        description="Passed to jhlabs unconverted (ImageToolbox uses the slider value as radians)"),
    Int("secondary_angle", 0, min=0, max=360, title="Secondary angle",
        description="Passed to jhlabs unconverted (radians), like ImageToolbox"),
    Float("center_x", 0.5, min=0, max=1, step=0.01, title="Center X"),
    Float("center_y", 0.5, min=0, max=1, step=0.01, title="Center Y"),
    Int("sides", 5, min=2, max=12, title="Sides"),
    Float("radius", 0.0, min=0, max=100, step=0.01, title="Radius", unit="px",
          description="0 = unbounded; otherwise the pattern repeats with this radius"),
], upstream="jhlabs", alpha="full", tags=["kaleidoscope", "mirror", "symmetry"])
def kaleidoscope(img, ctx, angle, secondary_angle, center_x, center_y, sides, radius):
    """jhlabs KaleidoscopeFilter (CLAMP edges)."""
    h, w = img.shape[:2]
    cx, cy = center_x * w, center_y * h
    a1, a2 = float(angle), float(secondary_angle)
    er = ctx.px(radius)

    def mapping(X, Y):
        dx, dy = X - cx, Y - cy
        r = np.hypot(dx, dy)
        t = ((np.arctan2(dy, dx) - a1 - a2) / math.pi * sides * 0.5).astype(np.float32)
        theta = G.jh_triangle(t).astype(np.float64)
        if er != 0:
            rc = er / np.cos(theta)
            r = rc * G.jh_triangle((r / rc).astype(np.float32))
        theta = theta + a1
        return cx + r * np.cos(theta), cy + r * np.sin(theta)

    return G.jh_transform(img, mapping, G.EDGE_CLAMP)


@itb_op("Offset", "Offset", GROUP, [
    Float("offset_x", 0.25, min=0, max=1, step=0.01, title="Offset X"),
    Float("offset_y", 0.25, min=0, max=1, step=0.01, title="Offset Y"),
], upstream="jhlabs", alpha="full", tags=["offset", "wrap", "shift", "tile"])
def offset(img, ctx, offset_x, offset_y):
    """jhlabs OffsetFilter with wrap: circular shift by a fraction of the size."""
    h, w = img.shape[:2]
    dx, dy = G.kround(offset_x * w), G.kround(offset_y * h)
    return np.roll(img, (dy % h, dx % w), axis=(0, 1)).astype(F32, copy=False)


_POLAR = ["rect_to_polar", "polar_to_rect", "invert_in_circle"]


@itb_op("PolarCoordinates", "Polar Coordinates", GROUP, [
    Choice("type", "rect_to_polar", choices=_POLAR, title="Type",
           labels={"rect_to_polar": "Rect to polar", "polar_to_rect": "Polar to rect",
                   "invert_in_circle": "Invert in circle"}),
], upstream="jhlabs", alpha="full", tags=["polar", "coordinates", "distort"])
def polar_coordinates(img, ctx, type):
    """jhlabs PolarFilter (CLAMP edges)."""
    h, w = img.shape[:2]
    radius = max(w, h) / 2.0
    kind = _POLAR.index(type)

    def mapping(X, Y):
        dx, dy = X - w / 2.0, Y - h / 2.0
        if kind == 0:
            theta = G.gmod(np.arctan2(dx, -dy), G.TAU)
            return w - 1.0 - (w - 1.0) * theta / G.TAU, h * np.hypot(dx, dy) / radius
        if kind == 1:
            theta = X / w * G.TAU
            d = radius * Y / h
            return w / 2.0 - d * np.sin(theta), h / 2.0 - d * np.cos(theta)
        d2 = np.maximum(dx * dx + dy * dy, 0.0001)
        return w / 2.0 + w * w / 4.0 * dx / d2, h / 2.0 + h * h / 4.0 * dy / d2

    return G.jh_transform(img, mapping, G.EDGE_CLAMP)


@itb_op("Shear", "Shear", GROUP, [
    Int("x_angle", 15, min=-89, max=89, title="X angle", unit="deg"),
    Int("y_angle", 0, min=-89, max=89, title="Y angle", unit="deg"),
    Bool("resize", True, title="Resize output",
         description="Kept for parity; ImageToolbox's native port keeps the canvas size"),
], upstream="jhlabs", alpha="full", tags=["shear", "skew", "slant"])
def shear(img, ctx, x_angle, y_angle, resize):
    """jhlabs ShearFilter (same canvas size, outside = transparent)."""
    h, w = img.shape[:2]
    xa, ya = math.radians(x_angle), math.radians(y_angle)
    x_off = -h * math.tan(xa)
    tw = h * abs(math.tan(xa)) + w
    y_off = -tw * math.tan(ya)
    sxa, sya = math.sin(xa), math.sin(ya)
    return G.jh_transform(img, lambda X, Y: (X + x_off + Y * sxa, Y + y_off + X * sya), G.EDGE_RGB_CLAMP)


# ---------------------------------------------------------------------------
# Procedural GPU shaders (ImageToolbox ProceduralShaders.kt)
# ---------------------------------------------------------------------------

@itb_op("CircularMirror", "Circular Mirror", GROUP, [
    Float("intensity", 1.0, min=0, max=1, step=0.001, title="Intensity"),
    Float("radius", 0.3, min=0.01, max=1, step=0.001, title="Radius"),
    Int("mode", 0, min=0, max=1, title="Mode", description="0 = mirror inside out, 1 = mirror outside in"),
    *_transform_params(),
], upstream="ImageToolbox", alpha="full", tags=["mirror", "circle", "reflection", "procedural"])
def circular_mirror(img, ctx, intensity, radius, mode, offset_x, offset_y, scale, rotation):
    """Procedural shader: mirrors the field around a circle of the given radius."""
    T = G.proc_transform(img, offset_x, offset_y, scale, rotation)
    return G.proc_single(img, T, G.k_circular_mirror, float(intensity), float(radius), float(mode))


@itb_op("ProgressiveScaling", "Progressive Scaling", GROUP, [
    Float("intensity", 1.0, min=0, max=1, step=0.001, title="Intensity"),
    Float("ratio", 1.5, min=1.05, max=4, step=0.001, title="Scale ratio"),
    Float("radius", 0.3, min=0.01, max=1, step=0.001, title="Radius"),
    *_transform_params(),
], upstream="ImageToolbox", alpha="full", tags=["droste", "recursive", "zoom", "procedural"])
def progressive_scaling(img, ctx, intensity, ratio, radius, offset_x, offset_y, scale, rotation):
    """Procedural shader: rings scaled by powers of ``ratio`` (recursive zoom)."""
    T = G.proc_transform(img, offset_x, offset_y, scale, rotation)
    return G.proc_single(img, T, G.k_progressive_scaling, float(intensity), float(ratio), float(radius))


@itb_op("HexKaleidoscope", "Hex Kaleidoscope", GROUP, [
    Int("count", 6, min=1, max=32, title="Count"),
    Float("frequency", 4.0, min=1, max=16, step=0.001, title="Frequency"),
    Float("offset", 0.0, min=-1, max=1, step=0.001, title="Offset"),
    Int("mode", 0, min=0, max=2, title="Mode"),
    *_transform_params(),
], upstream="ImageToolbox", alpha="full", tags=["kaleidoscope", "hexagon", "tiles", "procedural"])
def hex_kaleidoscope(img, ctx, count, frequency, offset, mode, offset_x, offset_y, scale, rotation):
    """Procedural shader: kaleidoscope sectors inside a hexagonal grid."""
    T = G.proc_transform(img, offset_x, offset_y, scale, rotation)
    return G.proc_single(img, T, G.k_hex_kaleidoscope, float(count), float(frequency), float(offset), float(mode))


@itb_op("SmoothKaleidoscope", "Smooth Kaleidoscope", GROUP, [
    Float("blend", 0.25, min=0, max=1, step=0.001, title="Blend"),
    Float("offset", 0.0, min=-1, max=1, step=0.001, title="Offset"),
    Float("frequency", 2.0, min=1, max=16, step=0.001, title="Frequency"),
    *_transform_params(),
], upstream="ImageToolbox", alpha="full", tags=["kaleidoscope", "tiles", "smooth", "procedural"])
def smooth_kaleidoscope(img, ctx, blend, offset, frequency, offset_x, offset_y, scale, rotation):
    """Procedural shader: rotated mirrored tiles blended smoothly across cell corners."""
    T = G.proc_transform(img, offset_x, offset_y, scale, rotation)
    return G.proc_smooth_kaleidoscope(img, T, blend, offset, frequency)


@itb_op("RandomKaleidoscopeGrid", "Random Kaleidoscope Grid", GROUP, [
    Int("count", 6, min=1, max=32, title="Count"),
    Float("frequency", 3.0, min=1, max=16, step=0.001, title="Frequency"),
    Int("seed", 0, min=0, max=1000, title="Random seed"),
    Float("variability", 0.5, min=0, max=1, step=0.001, title="Variability"),
    Float("blend", 0.25, min=0, max=1, step=0.001, title="Blend"),
    *_transform_params(),
], upstream="ImageToolbox", alpha="full", tags=["kaleidoscope", "grid", "random", "procedural"])
def random_kaleidoscope_grid(img, ctx, count, frequency, seed, variability, blend, offset_x, offset_y, scale,
                             rotation):
    """Procedural shader: grid of kaleidoscope cells with hashed spike count/rotation."""
    T = G.proc_transform(img, offset_x, offset_y, scale, rotation)
    return G.proc_random_kaleidoscope_grid(img, T, float(count), float(frequency), float(seed), float(variability),
                                           float(blend))


@itb_op("InversionFractal", "Inversion Fractal", GROUP, [
    Int("mode", 0, min=0, max=2, title="Mode"),
    Int("iterations", 6, min=1, max=16, title="Iterations"),
    Float("intensity", 0.8, min=0.1, max=2, step=0.001, title="Intensity"),
    *_transform_params(),
], upstream="ImageToolbox", alpha="full", tags=["fractal", "inversion", "kaleidoscope", "procedural"])
def inversion_fractal(img, ctx, mode, iterations, intensity, offset_x, offset_y, scale, rotation):
    """Procedural shader: iterated folding + circle inversion of the sampling field."""
    T = G.proc_transform(img, offset_x, offset_y, scale, rotation)
    return G.proc_single(img, T, G.k_inversion_fractal, float(mode), int(iterations), float(intensity))


# ---------------------------------------------------------------------------
# OpenCV / Kotlin geometry tools (opencv-tools, Trickle)
# ---------------------------------------------------------------------------

@itb_op("AutoCrop", "Auto Crop", GROUP, [
    Int("tolerance", 5, min=0, max=10, title="Tolerance", description="Crop sensitivity (higher = more edge rows accepted)"),
], upstream="OpenCV", alpha="full", geometry=True, tags=["crop", "auto", "trim", "borders"])
def auto_crop(img, ctx, tolerance):
    """opencv-tools AutoCropper: picks the horizontal band between strong Canny edge rows."""
    return G.auto_crop(img, tolerance)


@itb_op("CropOrPerspective", "Crop or Perspective", GROUP, [
    Point("top_left", (0.1, 0.0), title="Top Left"),
    Point("top_right", (1.0, 0.0), title="Top Right"),
    Point("bottom_left", (0.0, 1.0), title="Bottom Left"),
    Point("bottom_right", (0.9, 1.0), title="Bottom Right"),
    Bool("absolute", False, title="Absolute", description="Corners are full-resolution pixels instead of 0..1 fractions"),
], upstream="OpenCV", alpha="full", geometry=True, tags=["crop", "perspective", "quad", "warp", "document"])
def crop_or_perspective(img, ctx, top_left, top_right, bottom_left, bottom_right, absolute):
    """opencv-tools AutoStraighten Manual mode (FreeCrop): rectify the quad to a rectangle."""
    h, w = img.shape[:2]
    pts = [top_left, top_right, bottom_right, bottom_left]
    if absolute:
        pts = [(ctx.px(x), ctx.px(y)) for x, y in pts]
    else:
        pts = [(x * w, y * h) for x, y in pts]
    return G.free_crop(img, pts)


@itb_op("CropToContent", "Crop to Content", GROUP, [
    Float("tolerance", 0.0, min=0, max=1, step=0.01, title="Tolerance"),
    Color("color_to_ignore", "#000000", title="Color To Ignore"),
], upstream="Trickle", alpha="full", geometry=True, tags=["crop", "trim", "content", "borders"])
def crop_to_content(img, ctx, tolerance, color_to_ignore):
    """Trickle cropToContent: bounding box of pixels differing from the ignored colour."""
    return G.crop_to_content(img, color_to_ignore, tolerance)


@itb_op("Deskew", "Deskew", GROUP, [
    Int("max", 15, min=0, max=89, title="Max", unit="deg", description="Maximum skew angle to correct"),
    Bool("allow_crop", True, title="Allow crop"),
], upstream="OpenCV", alpha="full", geometry=True, tags=["deskew", "straighten", "rotate", "document"])
def deskew(img, ctx, max, allow_crop):
    """opencv-tools AutoStraighten Deskew: Hough-line median angle, rotate (and crop)."""
    return G.deskew(img, int(max), bool(allow_crop))


@itb_op("ExpandImage", "Expand Image", GROUP, [
    Int("top", 64, min=0, max=1024, title="Top", unit="px"),
    Int("start", 64, min=0, max=1024, title="Start", unit="px"),
    Int("end", 64, min=0, max=1024, title="End", unit="px"),
    Int("bottom", 64, min=0, max=1024, title="Bottom", unit="px"),
], upstream="OpenCV", alpha="full", geometry=True, tags=["expand", "outpaint", "extend", "canvas", "fill"])
def expand_image(img, ctx, top, start, end, bottom):
    """opencv-tools ImageExpander: patch-based texture synthesis outpainting."""
    px = [ctx.ipx(v, minimum=0) for v in (top, start, end, bottom)]
    return G.expand_image(img, top=px[0], left=px[1], right=px[2], bottom=px[3])


@itb_op("LensCorrection", "Lens Correction", GROUP, [
    Float("strength", 1.0, min=-1, max=3, step=0.01, title="Strength"),
    File("lens_profile", "", file_kind="any", title="Target lens profile file in JSON format",
         description="Gyroflow-style JSON: fisheye_params.camera_matrix (3x3), fisheye_params.distortion_coeffs (4), "
                     "calib_dimension {w, h}. Not bundled: supply your own."),
    FloatList("camera_matrix", [], min_length=0, max_length=9, title="Camera matrix",
              description="Alternative to a profile: 9 row-major values (fx 0 cx 0 fy cy 0 0 1) at calibration size"),
    FloatList("distortion_coeffs", [], min_length=0, max_length=4, title="Distortion coefficients",
              description="Fisheye k1..k4 (used with camera_matrix)"),
    Int("calib_width", 0, min=0, max=100000, title="Calibration width", unit="px",
        description="Width the camera matrix refers to (0 = image width)"),
    Int("calib_height", 0, min=0, max=100000, title="Calibration height", unit="px",
        description="Height the camera matrix refers to (0 = image height)"),
], upstream="OpenCV", alpha="full", tags=["lens", "distortion", "fisheye", "undistort", "correction"])
def lens_correction(img, ctx, strength, lens_profile, camera_matrix, distortion_coeffs, calib_width, calib_height):
    """opencv-tools LensCorrection: cv2.fisheye.undistortImage with a (scaled) lens profile.

    ImageToolbox falls back to a bundled sample profile; LazyRoom does not
    bundle profiles, so without a profile/matrix the image is returned unchanged.
    """
    h, w = img.shape[:2]
    if lens_profile:
        text = ctx.resolve_file(lens_profile, "lens profile").read_text(encoding="utf-8", errors="replace")
        k, d, cw, ch = G.parse_lens_profile(text)
    elif camera_matrix:
        if len(camera_matrix) != 9:
            raise invalid("camera_matrix needs 9 values (3x3, row-major)", param="camera_matrix")
        if len(distortion_coeffs) != 4:
            raise invalid("distortion_coeffs needs 4 values (k1..k4)", param="distortion_coeffs")
        k = np.asarray(camera_matrix, np.float64).reshape(3, 3)
        d = np.asarray(distortion_coeffs, np.float64)
        cw, ch = calib_width or w, calib_height or h
    else:
        return img
    return G.lens_undistort(img, k, d, cw, ch, strength)


_MIRROR_SIDES = ["left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"]


@itb_op("Mirror", "Mirror", GROUP, [
    Float("center", 0.5, min=0, max=1, step=0.01, title="Center"),
    Choice("side", "left_to_right", choices=_MIRROR_SIDES, title="Side",
           labels={"left_to_right": "Left to Right", "right_to_left": "Right to Left",
                   "top_to_bottom": "Top to Bottom", "bottom_to_top": "Bottom to Top"}),
], upstream="ImageToolbox", alpha="full", geometry=True, tags=["mirror", "symmetry", "reflect"])
def mirror(img, ctx, center, side):
    """Keeps one side of the mirror line and reflects it (output = 2 x the shorter half)."""
    if center <= 0 or center >= 1:
        return img
    h, w = img.shape[:2]
    if side in ("left_to_right", "right_to_left"):
        c = min(max(int(np.float32(w) * np.float32(center)), 1), w - 1)
        half = min(c, w - c)
        if side == "left_to_right":
            part = img[:, c - half:c]
            return np.ascontiguousarray(np.concatenate([part, part[:, ::-1]], axis=1))
        part = img[:, c:c + half]
        return np.ascontiguousarray(np.concatenate([part[:, ::-1], part], axis=1))
    c = min(max(int(np.float32(h) * np.float32(center)), 1), h - 1)
    half = min(c, h - c)
    if side == "top_to_bottom":
        part = img[c - half:c]
        return np.ascontiguousarray(np.concatenate([part, part[::-1]], axis=0))
    part = img[c:c + half]
    return np.ascontiguousarray(np.concatenate([part[::-1], part], axis=0))


@itb_op("SeamCarving", "Seam Carving", GROUP, [
    Int("width", 0, min=0, max=16384, title="Width", unit="px", description="Target width (0 with height 0 = unchanged)"),
    Int("height", 0, min=0, max=16384, title="Height", unit="px", description="Target height"),
    Bool("backward_energy", False, title="Backward energy",
         description="Use simple gradient magnitude energy map instead of forward energy algorithm"),
    File("mask", "", file_kind="image", title="Mask", advanced=True,
         description="Optional protection mask (non-black = protect)"),
    Bool("use_mask_as_removal", False, title="Use mask as removal", advanced=True,
         description="Remove the masked area with seams instead of protecting it"),
], upstream="OpenCV", alpha="full", geometry=True, tags=["seam carving", "content aware", "resize", "retarget"])
def seam_carving(img, ctx, width, height, backward_energy, mask, use_mask_as_removal):
    """opencv-tools SeamCarver: content-aware resize by removing/inserting low-energy seams."""
    if width == 0 or height == 0:
        return img
    m = None
    if mask:
        from ...core.io import load_image

        m, _ = load_image(ctx.resolve_file(mask, "mask"))
    return G.seam_carve(img, ctx.ipx(width), ctx.ipx(height), m, backward_energy, use_mask_as_removal, ctx)


# ---------------------------------------------------------------------------
# G'MIC filters (ImageToolboxLibs gmic builders; exact command strings)
# ---------------------------------------------------------------------------

@itb_op("CircleTransform", "Circle Transform", GROUP, [
    Float("center_x", 50.0, min=0, max=100, step=0.01, title="Center X", unit="%"),
    Float("center_y", 50.0, min=0, max=100, step=0.01, title="Center Y", unit="%"),
    Float("radius_point_x", 75.0, min=0, max=100, step=0.01, title="Radius Point X", unit="%"),
    Float("radius_point_y", 50.0, min=0, max=100, step=0.01, title="Radius Point Y", unit="%"),
    Float("x_scale", -2.0, min=-16, max=16, step=0.01, title="X Scale"),
    Float("y_scale", -2.0, min=-16, max=16, step=0.01, title="Y Scale"),
    Choice("symmetry", "none", choices=["none", "inside", "outside"], title="Symmetry"),
    _interp("linear"),
    _boundary("mirror"),
], upstream="G'MIC", requires=["gmic"], alpha="full", tags=["circle", "inversion", "distort", "gmic"])
def circle_transform(img, ctx, center_x, center_y, radius_point_x, radius_point_y, x_scale, y_scale, symmetry,
                     interpolation, boundary):
    cmd = gmic_cmd("fx_circle_transform", center_x, center_y, radius_point_x, radius_point_y, x_scale, y_scale,
                   ["none", "inside", "outside"].index(symmetry), _INTERP.index(interpolation),
                   _BOUNDARY.index(boundary))
    return G.gmic_apply(img, cmd, ctx, process_alpha=True)


_CONFORMAL = ["identity", "mobius", "cosine", "sine", "tangent", "exponential", "logarithm", "dipole", "star"]


@itb_op("ConformalMap", "Conformal Map", GROUP, [
    Choice("mapping", "dipole", choices=_CONFORMAL, title="Mapping"),
    Float("exponent_real", 1.0, min=-16, max=16, step=0.01, title="Exponent Real"),
    Float("exponent_imaginary", 0.0, min=-16, max=16, step=0.01, title="Exponent Imaginary"),
    Float("zoom", 0.0, min=-4, max=4, step=0.01, title="Image zoom"),
    Float("angle", 0.0, min=-180, max=180, step=0.01, title="Angle", unit="deg"),
    Float("aspect_ratio", 0.0, min=-1, max=1, step=0.01, title="Aspect ratio"),
    Float("x_shift", 0.0, min=-5, max=5, step=0.01, title="X Shift"),
    Float("y_shift", 0.0, min=-5, max=5, step=0.01, title="Y Shift"),
    _boundary("mirror"),
    Int("antialias", 0, min=0, max=3, title="Antialias"),
], upstream="G'MIC", requires=["gmic"], alpha="full", tags=["conformal", "complex", "mobius", "distort", "gmic"])
def conformal_map(img, ctx, mapping, exponent_real, exponent_imaginary, zoom, angle, aspect_ratio, x_shift, y_shift,
                  boundary, antialias):
    cmd = gmic_cmd("fx_conformal_maps", _CONFORMAL.index(mapping) + 1, exponent_real, exponent_imaginary, '"z"',
                   zoom, angle, aspect_ratio, x_shift, y_shift, _BOUNDARY.index(boundary), antialias, False, 0, 0)
    return G.gmic_apply(img, cmd, ctx, process_alpha=True)


_DROSTE_MODES = ["replace", "replace_sharpest", "behind", "below"]


@itb_op("Droste", "Droste", GROUP, [
    Float("upper_left_x", 20.0, min=0, max=100, step=0.01, title="Upper Left X", unit="%"),
    Float("upper_left_y", 20.0, min=0, max=100, step=0.01, title="Upper Left Y", unit="%"),
    Float("upper_right_x", 80.0, min=0, max=100, step=0.01, title="Upper Right X", unit="%"),
    Float("upper_right_y", 20.0, min=0, max=100, step=0.01, title="Upper Right Y", unit="%"),
    Float("lower_right_x", 80.0, min=0, max=100, step=0.01, title="Lower Right X", unit="%"),
    Float("lower_right_y", 80.0, min=0, max=100, step=0.01, title="Lower Right Y", unit="%"),
    Float("lower_left_x", 20.0, min=0, max=100, step=0.01, title="Lower Left X", unit="%"),
    Float("lower_left_y", 80.0, min=0, max=100, step=0.01, title="Lower Left Y", unit="%"),
    Int("iterations", 1, min=1, max=10, title="Iterations"),
    Float("x_shift", 0.0, min=-100, max=100, step=0.01, title="X Shift"),
    Float("y_shift", 0.0, min=-100, max=100, step=0.01, title="Y Shift"),
    Float("angle", 0.0, min=0, max=360, step=0.01, title="Angle", unit="deg"),
    Float("zoom", 1.0, min=0.1, max=5, step=0.01, title="Image zoom"),
    Choice("mirror", "none", choices=_MIRROR4, title="Mirror"),
    _boundary("nearest"),
    Choice("draw_mode", "replace", choices=_DROSTE_MODES, title="Draw mode"),
], upstream="G'MIC", requires=["gmic"], alpha="full", tags=["droste", "recursive", "infinite", "gmic"])
def droste(img, ctx, upper_left_x, upper_left_y, upper_right_x, upper_right_y, lower_right_x, lower_right_y,
           lower_left_x, lower_left_y, iterations, x_shift, y_shift, angle, zoom, mirror, boundary, draw_mode):
    cmd = gmic_cmd("fx_droste", upper_left_x, upper_left_y, upper_right_x, upper_right_y, lower_right_x,
                   lower_right_y, lower_left_x, lower_left_y, iterations, x_shift, y_shift, angle, zoom,
                   _MIRROR4.index(mirror), _BOUNDARY.index(boundary), _DROSTE_MODES.index(draw_mode))
    return G.gmic_apply(img, cmd, ctx, process_alpha=True)


@itb_op("EquirectangularProjection", "Equirectangular Projection", GROUP, [
    Choice("mode", "nadir_zenith", choices=["nadir_zenith", "equirectangular"], title="Mode",
           labels={"nadir_zenith": "Equirectangular to nadir/zenith", "equirectangular": "Nadir/zenith to equirectangular"}),
], upstream="G'MIC", requires=["gmic"], alpha="full", geometry=True, tags=["panorama", "equirectangular", "360", "gmic"])
def equirectangular_projection(img, ctx, mode):
    cmd = gmic_cmd("fx_equirectangular2nadirzenith", ["nadir_zenith", "equirectangular"].index(mode))
    return G.gmic_apply(img, cmd, ctx, process_alpha=True)


@itb_op("PoincareDisk", "Poincare Disk", GROUP, [
    Int("p_value", 5, min=3, max=16, title="P Value"),
    Int("q_value", 6, min=3, max=32, title="Q Value"),
    Float("size", 100.0, min=0, max=200, step=0.01, title="Size"),
    Int("iterations", 20, min=0, max=40, title="Iterations"),
    Float("angle", 0.0, min=-180, max=180, step=0.01, title="Angle", unit="deg"),
    Choice("tiling", "polygonal", choices=["triangular", "polygonal"], title="Tiling"),
    Bool("antialias", True, title="Antialias"),
    Choice("fill", "image", choices=["binary", "image"], title="Fill"),
    Float("x_shift", 50.0, min=0, max=100, step=0.01, title="X Shift"),
    Float("y_shift", 50.0, min=0, max=100, step=0.01, title="Y Shift"),
    Float("zoom", 0.0, min=-10, max=10, step=0.01, title="Image zoom"),
    Float("image_angle", 0.0, min=-180, max=180, step=0.01, title="Image Angle", unit="deg"),
    Int("outline", 2, min=0, max=8, title="Outline"),
    Color("outline_color", "#000000", title="Outline Color"),
    Color("first_color", "#000000", title="First color"),
    Color("second_color", "#FFFFFF", title="Second color"),
], upstream="G'MIC", requires=["gmic"], alpha="full", geometry=True, tags=["poincare", "hyperbolic", "tiling", "gmic"])
def poincare_disk(img, ctx, p_value, q_value, size, iterations, angle, tiling, antialias, fill, x_shift, y_shift, zoom,
                  image_angle, outline, outline_color, first_color, second_color):
    cmd = gmic_cmd("fx_poincare_disk", p_value, q_value, size, iterations, angle,
                   ["triangular", "polygonal"].index(tiling), bool(antialias), ["binary", "image"].index(fill),
                   x_shift, y_shift, zoom, image_angle, outline, *G.gmic_color(outline_color),
                   *G.gmic_color(first_color), *G.gmic_color(second_color))
    with gmic_single_thread():  # its threaded min/max reduction was not reproducible (seen on Windows CI)
        return G.gmic_apply(img, cmd, ctx, process_alpha=True)


_POLAR_PRESETS = ["custom_transform", "inverse_radius", "swap_radius_angle"]
_RADIUS_EXPR = {"wavy": "r + R/10*cos(a*5)", "identity": "r", "inverse": "R-r", "angle": "a*R/(2*pi)"}
_ANGLE_EXPR = {"identity": "a", "mirror": "-a", "radius": "r*2*pi/R", "spiral": "a + r/R"}


@itb_op("PolarTransform", "Polar Transform", GROUP, [
    Choice("preset", "custom_transform", choices=_POLAR_PRESETS, title="Preset"),
    Float("center_x", 50.0, min=0, max=100, step=0.01, title="Center X", unit="%"),
    Float("center_y", 50.0, min=0, max=100, step=0.01, title="Center Y", unit="%"),
    Choice("radius_expression", "wavy", choices=list(_RADIUS_EXPR), labels=dict(_RADIUS_EXPR),
           title="Radius Expression", description="Used by the custom transform preset"),
    Choice("angle_expression", "identity", choices=list(_ANGLE_EXPR), labels=dict(_ANGLE_EXPR),
           title="Angle Expression", description="Used by the custom transform preset"),
    _boundary("mirror"),
], upstream="G'MIC", requires=["gmic"], alpha="full", tags=["polar", "transform", "distort", "gmic"])
def polar_transform(img, ctx, preset, center_x, center_y, radius_expression, angle_expression, boundary):
    # The CLI wraps argv items containing spaces in quotes, so spaces are removed from the
    # (whitespace-insensitive) math expressions.
    rexp = _RADIUS_EXPR[radius_expression].replace(" ", "")
    aexp = _ANGLE_EXPR[angle_expression].replace(" ", "")
    cmd = gmic_cmd("fx_transform_polar", _POLAR_PRESETS.index(preset), center_x, center_y, f'"{rexp}"', f'"{aexp}"',
                   _BOUNDARY.index(boundary))
    return G.gmic_apply(img, cmd, ctx, process_alpha=True)


@itb_op("SquareToCircle", "Square To Circle", GROUP, [
    Choice("mode", "square_to_circle", choices=["square_to_circle", "circle_to_square"], title="Mode"),
    Float("center_x", 50.0, min=0, max=100, step=0.01, title="Center X", unit="%"),
    Float("center_y", 50.0, min=0, max=100, step=0.01, title="Center Y", unit="%"),
    Float("strength", 100.0, min=0, max=100, step=0.01, title="Strength"),
    Float("zoom", 0.0, min=-100, max=100, step=0.01, title="Image zoom"),
    Float("angle", 0.0, min=-180, max=180, step=0.01, title="Angle", unit="deg"),
    _interp("linear"),
    _boundary("transparent"),
    Bool("adapt_to_image_ratio", True, title="Adapt To Image Ratio"),
], upstream="G'MIC", requires=["gmic"], alpha="full", tags=["square", "circle", "distort", "gmic"])
def square_to_circle(img, ctx, mode, center_x, center_y, strength, zoom, angle, interpolation, boundary,
                     adapt_to_image_ratio):
    cmd = gmic_cmd("fx_square_circle_alt", ["square_to_circle", "circle_to_square"].index(mode), center_x, center_y,
                   strength, zoom, angle, _INTERP.index(interpolation), _BOUNDARY.index(boundary),
                   bool(adapt_to_image_ratio))
    return G.gmic_apply(img, cmd, ctx, process_alpha=True)


@itb_op("StereographicProjection", "Stereographic Projection", GROUP, [
    Choice("transform", "direct", choices=["direct", "inverse"], title="Transform"),
    Float("center_x", 50.0, min=0, max=100, step=0.01, title="Center X", unit="%"),
    Float("center_y", 50.0, min=0, max=100, step=0.01, title="Center Y", unit="%"),
    Float("radius_point_x", 50.0, min=0, max=100, step=0.01, title="Radius Point X", unit="%"),
    Float("radius_point_y", 75.0, min=0, max=100, step=0.01, title="Radius Point Y", unit="%"),
    Float("horizon_leveling", 0.0, min=-10, max=10, step=0.01, title="Horizon Leveling"),
    Float("side_blur", 0.0, min=0, max=20, step=0.01, title="Side Blur"),
    Float("dilation", 0.0, min=-2, max=2, step=0.01, title="Dilation"),
    Choice("mirror", "none", choices=_MIRROR4, title="Mirror"),
    _boundary("transparent"),
], upstream="G'MIC", requires=["gmic"], alpha="full", geometry=True,
        tags=["stereographic", "little planet", "projection", "gmic"])
def stereographic_projection(img, ctx, transform, center_x, center_y, radius_point_x, radius_point_y,
                             horizon_leveling, side_blur, dilation, mirror, boundary):
    if center_x == radius_point_x and center_y == radius_point_y:
        return img  # zero radius: G'MIC's projection is undefined (it errors out)
    cmd = gmic_cmd("fx_project_stereographic", ["direct", "inverse"].index(transform), center_x, center_y,
                   radius_point_x, radius_point_y, horizon_leveling, side_blur, dilation, _MIRROR4.index(mirror),
                   _BOUNDARY.index(boundary), f'"{float(center_x)},{float(center_y)}"')
    return G.gmic_apply(img, cmd, ctx, process_alpha=True)


@itb_op("Tunnel", "Tunnel", GROUP, [
    Int("depth", 4, min=1, max=100, title="Depth"),
    Float("factor", 80.0, min=1, max=99, step=0.01, title="Factor"),
    Float("center_x", 50.0, min=0, max=100, step=0.01, title="Center X", unit="%"),
    Float("center_y", 50.0, min=0, max=100, step=0.01, title="Center Y", unit="%"),
    Float("opacity", 0.2, min=0, max=1, step=0.01, title="Opacity"),
    Float("angle", 0.0, min=-90, max=90, step=0.01, title="Angle", unit="deg"),
], upstream="G'MIC", requires=["gmic"], alpha="full", tags=["tunnel", "recursive", "zoom", "gmic"])
def tunnel(img, ctx, depth, factor, center_x, center_y, opacity, angle):
    cmd = gmic_cmd("fx_tunnel", depth, factor, center_x, center_y, opacity, angle)
    return G.gmic_apply(img, cmd, ctx, process_alpha=False)
