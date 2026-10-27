"""Geometric operations (category ``geometry``). These change pixel positions or
image size and therefore cannot be masked."""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..core.errors import invalid
from ..core.image import ensure_rgba, has_alpha
from ..core.params import Bool, Choice, Color, File, Float, Int, Point, Points, Text
from ..core.registry import op
from ._util import F32, remap

G = dict(geometry=True, alpha="full")

INTERP = {"nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC,
          "lanczos": cv2.INTER_LANCZOS4, "area": cv2.INTER_AREA}


def _fill(img: np.ndarray, color) -> tuple[np.ndarray, tuple]:
    """Return (img, border value) — promotes to RGBA if the fill is transparent."""
    if color[3] < 1.0:
        img = ensure_rgba(img)
    val = tuple(float(c) for c in color[: img.shape[2]])
    return img, val


@op("geometry.crop", "Crop", params=[
    Float("x", 0.0, min=0, description="Left edge"), Float("y", 0.0, min=0, description="Top edge"),
    Float("width", 1.0, min=0, description="Crop width"), Float("height", 1.0, min=0, description="Crop height"),
    Choice("units", "normalized", choices=["normalized", "pixels"],
           description="normalized: fractions of the image (0..1); pixels: full-resolution pixels"),
], tags=["crop", "cut", "trim", "region"], examples=[{"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8}], **G)
def crop(img, ctx, x, y, width, height, units):
    """Crop to a rectangle."""
    h, w = img.shape[:2]
    if units == "pixels":
        x0, y0, cw, ch = (ctx.px(v) for v in (x, y, width, height))
    else:
        if x > 1 or y > 1 or width > 1 or height > 1:
            raise invalid("Normalized crop values must be within 0..1", hint="Use units='pixels' for pixel values")
        x0, y0, cw, ch = x * w, y * h, width * w, height * h
    x0, y0 = int(round(x0)), int(round(y0))
    x1, y1 = min(w, int(round(x0 + cw))), min(h, int(round(y0 + ch)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise invalid("Crop rectangle is empty or outside the image")
    return img[y0:y1, x0:x1].copy()


def parse_ratio(s: str) -> float:
    s = s.strip().lower()
    for sep in (":", "/", "x"):
        if sep in s:
            a, b = s.split(sep, 1)
            return float(a) / float(b)
    return float(s)


@op("geometry.crop_aspect", "Crop to aspect ratio", params=[
    Text("ratio", "1:1", max_length=32, description="Aspect ratio like '16:9', '4:5', '3:2', '1.91' or 'original'"),
    Point("center", (0.5, 0.5), min=0, max=1, description="Centre of the crop (normalized)"),
    Float("zoom", 1.0, min=1, max=20, step=0.01, description="1 = largest possible crop"),
], tags=["aspect ratio", "instagram", "square", "16:9", "4:5", "crop"], **G)
def crop_aspect(img, ctx, ratio, center, zoom):
    """Largest crop of the given aspect ratio around a centre point."""
    h, w = img.shape[:2]
    try:
        r = w / h if ratio.strip().lower() == "original" else parse_ratio(ratio)
    except (ValueError, ZeroDivisionError):
        raise invalid(f"Invalid aspect ratio {ratio!r}", hint="Use e.g. '16:9' or '1.5'") from None
    if r <= 0:
        raise invalid("Aspect ratio must be positive")
    cw, ch = (w, w / r) if w / h < r else (h * r, h)
    cw, ch = cw / zoom, ch / zoom
    cx = min(max(center[0] * w, cw / 2), w - cw / 2)
    cy = min(max(center[1] * h, ch / 2), h - ch / 2)
    x0, y0 = int(round(cx - cw / 2)), int(round(cy - ch / 2))
    return img[max(0, y0):y0 + max(1, int(round(ch))), max(0, x0):x0 + max(1, int(round(cw)))].copy()


def resize_buffer(img: np.ndarray, nw: int, nh: int, method: str = "auto", colorspace: str = "srgb") -> np.ndarray:
    nw, nh = max(1, int(nw)), max(1, int(nh))
    if (nw, nh) == (img.shape[1], img.shape[0]):
        return img
    if method == "auto":
        method = "area" if nw * nh < img.shape[0] * img.shape[1] else "lanczos"
    if method in INTERP and colorspace != "srgb":
        from .resample import LINEAR_LIGHT_SPACES, decode_space, encode_space

        if colorspace in LINEAR_LIGHT_SPACES:  # OpenCV kernel applied to linear light
            out = resize_buffer(encode_space(img, colorspace), nw, nh, method)
            return decode_space(out, colorspace)
    elif method in INTERP:
        src = img
        alpha = has_alpha(img)
        if alpha and method not in ("nearest",):
            from ..core.image import premultiply, unpremultiply

            src = premultiply(img)
        out = cv2.resize(src, (nw, nh), interpolation=INTERP[method])
        if alpha and method not in ("nearest",):
            from ..core.image import unpremultiply

            out = unpremultiply(np.clip(out, 0, None))
        return out.astype(F32)
    from .resample import resample  # extended kernel family (and OpenCV names in other colour spaces)

    return resample(img, nw, nh, method, colorspace)


def resize_methods() -> list[str]:
    base = ["auto", "nearest", "bilinear", "bicubic", "lanczos", "area"]
    try:
        from .resample import KERNELS

        return base + [k for k in KERNELS if k not in base]
    except ImportError:  # pragma: no cover
        return base


def resize_colorspaces() -> list[str]:
    try:
        from .resample import SCALE_COLOR_SPACES

        return list(SCALE_COLOR_SPACES)
    except ImportError:  # pragma: no cover
        return ["srgb"]


@op("geometry.resize", "Resize", params=[
    Int("width", 0, min=0, max=65535, unit="px", description="Target width (0 = derive from height/aspect)"),
    Int("height", 0, min=0, max=65535, unit="px", description="Target height (0 = derive from width/aspect)"),
    Float("scale", 100, min=1, max=1600, unit="%", description="Used when width and height are both 0"),
    Choice("mode", "fit", choices=["fit", "fill", "stretch", "long_edge", "short_edge"],
           description="fit: inside the box; fill: cover the box and centre-crop; stretch: exact size; "
                       "long_edge/short_edge: set that edge to 'width'"),
    Choice("method", "auto", choices=resize_methods(), description="Resampling filter"),
    Bool("allow_upscale", True),
    Choice("colorspace", "srgb", choices=resize_colorspaces(),
           description="Colour space the filter works in (srgb: encoded values; linear*: linear light; "
                       "lab/luv/lch/oklab*/jzazbz*: perceptual spaces)"),
], tags=["resize", "scale", "downscale", "upscale", "dimensions", "thumbnail"],
    examples=[{"width": 2048, "mode": "long_edge"}, {"scale": 50}], **G)
def resize(img, ctx, width, height, scale, mode, method, allow_upscale, colorspace="srgb"):
    """Resize with aspect-ratio handling and a choice of 50+ resampling filters."""
    h, w = img.shape[:2]
    # target sizes are expressed at full resolution; convert to the current buffer scale
    fw, fh = w / ctx.scale, h / ctx.scale
    if width == 0 and height == 0:
        tw, th = fw * scale / 100, fh * scale / 100
    elif mode in ("long_edge", "short_edge"):
        edge = width or height
        cur = max(fw, fh) if mode == "long_edge" else min(fw, fh)
        s = edge / cur
        tw, th = fw * s, fh * s
    elif width == 0:
        tw, th = fw * height / fh, height
    elif height == 0:
        tw, th = width, fh * width / fw
    elif mode == "stretch":
        tw, th = width, height
    elif mode == "fill":
        s = max(width / fw, height / fh)
        tw, th = fw * s, fh * s
    else:
        s = min(width / fw, height / fh)
        tw, th = fw * s, fh * s
    if not allow_upscale and (tw > fw or th > fh):
        s = min(fw / tw, fh / th)
        tw, th = tw * s, th * s
    out = resize_buffer(img, round(tw * ctx.scale), round(th * ctx.scale), method, colorspace)
    if mode == "fill" and width and height:
        cw, ch = round(width * ctx.scale), round(height * ctx.scale)
        y0 = max(0, (out.shape[0] - ch) // 2)
        x0 = max(0, (out.shape[1] - cw) // 2)
        out = out[y0:y0 + ch, x0:x0 + cw]
    return out


@op("geometry.rotate", "Rotate", params=[
    Float("angle", 0, min=-360, max=360, step=0.1, unit="deg", description="Counter-clockwise degrees"),
    Bool("expand", True, description="Grow the canvas to fit the rotated image"),
    Color("fill", "#00000000", description="Background for uncovered areas (transparent by default)"),
    Choice("interpolation", "bicubic", choices=["nearest", "bilinear", "bicubic", "lanczos"]),
], tags=["rotate", "turn", "angle"], **G)
def rotate(img, ctx, angle, expand, fill, interpolation):
    """Rotate by an arbitrary angle."""
    a = angle % 360
    if a in (0.0,):
        return img
    if a in (90.0, 180.0, 270.0) and expand:
        return np.ascontiguousarray(np.rot90(img, int(a // 90)))
    img, val = _fill(img, fill)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    if expand:
        c, s = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(math.ceil(h * s + w * c)), int(math.ceil(h * c + w * s))
        M[0, 2] += nw / 2 - w / 2
        M[1, 2] += nh / 2 - h / 2
    else:
        nw, nh = w, h
    return cv2.warpAffine(img, M, (nw, nh), flags=INTERP[interpolation], borderMode=cv2.BORDER_CONSTANT,
                          borderValue=val).astype(F32)


@op("geometry.rotate90", "Rotate 90°", params=[
    Choice("direction", "cw", choices=["cw", "ccw", "180"]),
], tags=["rotate", "orientation", "portrait", "landscape"], **G)
def rotate90(img, ctx, direction):
    """Lossless quarter-turn rotation."""
    k = {"cw": -1, "ccw": 1, "180": 2}[direction]
    return np.ascontiguousarray(np.rot90(img, k))


@op("geometry.flip", "Flip / mirror", params=[
    Choice("axis", "horizontal", choices=["horizontal", "vertical", "both"]),
], tags=["flip", "mirror", "reflect"], **G)
def flip(img, ctx, axis):
    """Mirror horizontally and/or vertically."""
    if axis == "horizontal":
        return np.ascontiguousarray(img[:, ::-1])
    if axis == "vertical":
        return np.ascontiguousarray(img[::-1])
    return np.ascontiguousarray(img[::-1, ::-1])


def _inscribed(w: float, h: float, angle_rad: float) -> tuple[float, float]:
    """Largest axis-aligned rectangle within a w x h rectangle rotated by angle."""
    if w <= 0 or h <= 0:
        return 0, 0
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)
    sin_a, cos_a = abs(math.sin(angle_rad)), abs(math.cos(angle_rad))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
    return wr, hr


@op("geometry.straighten", "Straighten", params=[
    Float("angle", 0, min=-45, max=45, step=0.05, unit="deg", description="Counter-clockwise correction"),
    Bool("auto_crop", True, description="Crop to the largest rectangle without empty corners"),
    Bool("auto_detect", False, description="Detect the horizon/verticals automatically (ignores angle)"),
], tags=["straighten", "horizon", "level", "tilt"], **G)
def straighten(img, ctx, angle, auto_crop, auto_detect):
    """Level a tilted horizon; optionally detect the tilt automatically (Hough lines)."""
    if auto_detect:
        angle = detect_tilt(img)
    if abs(angle) < 1e-3:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101).astype(F32)
    if not auto_crop:
        return rot
    cw, ch = _inscribed(w, h, math.radians(angle))
    x0, y0 = int(math.ceil((w - cw) / 2)), int(math.ceil((h - ch) / 2))
    return rot[y0:h - y0, x0:w - x0].copy()


def detect_tilt(img: np.ndarray) -> float:
    """Estimate the tilt (degrees) from dominant near-horizontal/vertical lines."""
    from ._util import luma

    g = (np.clip(luma(img), 0, 1) * 255).astype(np.uint8)
    s = 1000 / max(g.shape)
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(g, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=60, minLineLength=max(g.shape) // 8, maxLineGap=8)
    if lines is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        a = (a + 90) % 180 - 90
        if abs(a) < 30:
            d = a
        elif abs(abs(a) - 90) < 30:
            d = a - 90 if a > 0 else a + 90
        else:
            continue
        angles.append(d)
        weights.append(math.hypot(x2 - x1, y2 - y1))
    if not angles:
        return 0.0
    order = np.argsort(angles)
    a = np.array(angles)[order]
    wts = np.array(weights)[order]
    cum = np.cumsum(wts)
    return float(a[np.searchsorted(cum, cum[-1] / 2)])


@op("geometry.perspective", "Perspective (corner pin)", params=[
    Points("corners", [[0, 0], [1, 0], [1, 1], [0, 1]], min_items=4, max_items=4,
           description="Source quadrilateral TL, TR, BR, BL (normalized) mapped to the output rectangle"),
    Choice("output_size", "auto", choices=["auto", "same"],
           description="auto: size from the quad's edge lengths; same: keep image size"),
], tags=["perspective", "keystone", "document", "warp", "corner pin", "rectify"], **G)
def perspective(img, ctx, corners, output_size):
    """Rectify a quadrilateral region to a rectangle (document / building correction)."""
    h, w = img.shape[:2]
    src = np.float32([[x * w, y * h] for x, y in corners])
    if output_size == "same":
        ow, oh = w, h
    else:
        ow = int(round(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3]))))
        oh = int(round(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1]))))
    dst = np.float32([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (max(1, ow), max(1, oh)), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE).astype(F32)


@op("geometry.keystone", "Transform (keystone)", params=[
    Float("vertical", 0, min=-100, max=100, description="Correct converging verticals"),
    Float("horizontal", 0, min=-100, max=100),
    Float("rotate", 0, min=-45, max=45, unit="deg"),
    Float("scale", 100, min=50, max=150, unit="%"),
    Float("aspect", 0, min=-100, max=100),
    Float("offset_x", 0, min=-100, max=100), Float("offset_y", 0, min=-100, max=100),
    Bool("crop", False, description="Crop away the empty borders"),
], tags=["keystone", "perspective", "upright", "architecture", "transform"], **G)
def keystone(img, ctx, vertical, horizontal, rotate, scale, aspect, offset_x, offset_y, crop):
    """Lightroom-style manual transform (vertical/horizontal perspective, rotate, scale, aspect)."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    v, hz = vertical / 100 * 0.25, horizontal / 100 * 0.25
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = corners.copy()
    dst[0, 0] += v * w; dst[1, 0] -= v * w
    dst[0, 1] += hz * h; dst[3, 1] -= hz * h
    M = cv2.getPerspectiveTransform(corners, dst)
    a = 1 + aspect / 200.0
    S = np.array([[scale / 100 * a, 0, 0], [0, scale / 100 / a, 0], [0, 0, 1]])
    R = np.vstack([cv2.getRotationMatrix2D((0, 0), rotate, 1.0), [0, 0, 1]])
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    T2 = np.array([[1, 0, cx + offset_x / 100 * w], [0, 1, cy + offset_y / 100 * h], [0, 0, 1]])
    H = T2 @ R @ S @ T1 @ M
    out = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0)).astype(F32)
    if crop:
        ones = cv2.warpPerspective(np.ones((h, w), F32), H, (w, h), flags=cv2.INTER_NEAREST)
        rows = np.where(ones.min(1) > 0.5)[0]
        cols = np.where(ones.min(0) > 0.5)[0]
        if len(rows) and len(cols):
            return out[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].copy()
    elif out.shape[2] == 3:
        ones = cv2.warpPerspective(np.ones((h, w), F32), H, (w, h), flags=cv2.INTER_LINEAR)
        if ones.min() < 0.999:
            out = np.dstack([out, ones])
    return out


@op("geometry.canvas", "Canvas size / padding", params=[
    Int("top", 0, min=0, max=20000, unit="px"), Int("right", 0, min=0, max=20000, unit="px"),
    Int("bottom", 0, min=0, max=20000, unit="px"), Int("left", 0, min=0, max=20000, unit="px"),
    Text("aspect", "", max_length=32, description="Instead of margins: pad to this aspect ratio (e.g. '1:1')"),
    Point("anchor", (0.5, 0.5), min=0, max=1, description="Where the image sits when padding to an aspect"),
    Choice("fill", "color", choices=["color", "blur", "edge", "mirror"],
           description="color: solid; blur: blurred enlarged copy; edge: repeat edge pixels; mirror: reflect"),
    Color("color", "#FFFFFF"),
], tags=["canvas", "padding", "border", "expand", "letterbox", "square fit", "background blur"], **G)
def canvas(img, ctx, top, right, bottom, left, aspect, anchor, fill, color):
    """Expand the canvas with margins or to a target aspect ratio."""
    h, w = img.shape[:2]
    if aspect.strip():
        r = parse_ratio(aspect)
        tw, th = (w, w / r) if w / h > r else (h * r, h)
        tw, th = int(round(tw)), int(round(th))
        left = int(round((tw - w) * anchor[0]))
        top = int(round((th - h) * anchor[1]))
        right, bottom = tw - w - left, th - h - top
    else:
        top, right, bottom, left = (ctx.ipx(v, 0) for v in (top, right, bottom, left))
    if not (top or right or bottom or left):
        return img
    nh, nw = h + top + bottom, w + left + right
    if fill == "edge":
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE).astype(F32)
    if fill == "mirror":
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT101).astype(F32)
    if fill == "blur":
        s = max(nw / w, nh / h)
        big = cv2.resize(img, (int(math.ceil(w * s)), int(math.ceil(h * s))), interpolation=cv2.INTER_LINEAR)
        y0, x0 = (big.shape[0] - nh) // 2, (big.shape[1] - nw) // 2
        bg = big[y0:y0 + nh, x0:x0 + nw]
        bg = cv2.GaussianBlur(bg, (0, 0), max(nw, nh) / 40).astype(F32)
        bg = bg * 0.85
        out = bg.copy()
        if out.shape[2] == 4:
            out[..., 3] = 1
        out[top:top + h, left:left + w] = img if img.shape[2] == out.shape[2] else img[..., :out.shape[2]]
        return out
    base, val = _fill(img, color)
    out = np.empty((nh, nw, base.shape[2]), F32)
    out[:] = np.array(val, F32)
    out[top:top + h, left:left + w] = base
    return out


@op("geometry.trim", "Trim borders", params=[
    Float("tolerance", 0.05, min=0, max=1, step=0.01, description="Colour difference treated as border"),
    Int("margin", 0, min=0, max=2000, unit="px"),
], tags=["trim", "auto crop", "remove borders", "whitespace"], **G)
def trim(img, ctx, tolerance, margin):
    """Remove uniform borders (colour taken from the corners, or transparency)."""
    if has_alpha(img) and img[..., 3].min() < 0.99:
        content = img[..., 3] > tolerance
    else:
        corners = np.stack([img[0, 0, :3], img[0, -1, :3], img[-1, 0, :3], img[-1, -1, :3]])
        bg = np.median(corners, axis=0)
        content = np.abs(img[..., :3] - bg).max(-1) > tolerance
    rows, cols = np.where(content.any(1))[0], np.where(content.any(0))[0]
    if not len(rows) or not len(cols):
        return img
    m = ctx.ipx(margin, 0)
    h, w = img.shape[:2]
    return img[max(0, rows[0] - m):min(h, rows[-1] + 1 + m), max(0, cols[0] - m):min(w, cols[-1] + 1 + m)].copy()


@op("geometry.lens_distortion", "Lens distortion", params=[
    Float("k1", 0, min=-1, max=1, step=0.01, description="Barrel (+) / pincushion (-) correction"),
    Float("k2", 0, min=-1, max=1, step=0.01, advanced=True),
    Float("zoom", 1.0, min=0.5, max=2, step=0.01),
    Choice("border", "clamp", choices=["clamp", "reflect", "transparent", "constant"]),
], tags=["lens", "barrel", "pincushion", "fisheye", "distortion", "correction"], **G)
def lens_distortion(img, ctx, k1, k2, zoom, border):
    """Brown–Conrady radial distortion / correction."""
    h, w = img.shape[:2]
    X, Y = np.meshgrid(np.arange(w, dtype=F32), np.arange(h, dtype=F32))
    cx, cy = (w - 1) / 2, (h - 1) / 2
    norm = math.hypot(cx, cy)
    nx, ny = (X - cx) / norm / zoom, (Y - cy) / norm / zoom
    r2 = nx * nx + ny * ny
    f = 1 + k1 * r2 + k2 * r2 * r2
    return remap(img, nx * f * norm + cx, ny * f * norm + cy, interpolation=cv2.INTER_CUBIC, border=border)


SHAPES = ["circle", "oval", "rounded", "squircle", "triangle", "diamond", "pentagon", "hexagon", "octagon",
          "star", "heart", "cut_corners", "image"]


def shape_mask(shape: str, h: int, w: int, roundness: float = 0.15, points: int = 5,
               image_mask: np.ndarray | None = None) -> np.ndarray:
    S = 4  # supersampling for anti-aliasing
    H, W = h * S, w * S
    m = np.zeros((H, W), np.uint8)
    cx, cy = W / 2, H / 2
    if shape in ("circle", "oval"):
        r = min(W, H) / 2
        axes = (int(r), int(r)) if shape == "circle" else (W // 2, H // 2)
        cv2.ellipse(m, (int(cx), int(cy)), axes, 0, 0, 360, 255, -1)
    elif shape in ("rounded", "squircle"):
        if shape == "squircle":
            yy, xx = np.mgrid[0:H, 0:W]
            u, v = (xx - cx) / (W / 2), (yy - cy) / (H / 2)
            m = ((np.abs(u) ** 4 + np.abs(v) ** 4) <= 1).astype(np.uint8) * 255
        else:
            r = int(roundness * min(W, H) / 2)
            cv2.rectangle(m, (r, 0), (W - r, H), 255, -1)
            cv2.rectangle(m, (0, r), (W, H - r), 255, -1)
            for px, py in ((r, r), (W - r, r), (r, H - r), (W - r, H - r)):
                cv2.circle(m, (px, py), r, 255, -1)
    elif shape == "cut_corners":
        c = int(roundness * min(W, H))
        poly = np.array([[c, 0], [W - c, 0], [W, c], [W, H - c], [W - c, H], [c, H], [0, H - c], [0, c]])
        cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    elif shape in ("triangle", "diamond", "pentagon", "hexagon", "octagon"):
        n = {"triangle": 3, "diamond": 4, "pentagon": 5, "hexagon": 6, "octagon": 8}[shape]
        off = -math.pi / 2 if shape != "octagon" else -math.pi / 2 + math.pi / 8
        pts = [[cx + W / 2 * math.cos(off + 2 * math.pi * i / n), cy + H / 2 * math.sin(off + 2 * math.pi * i / n)]
               for i in range(n)]
        cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
    elif shape == "star":
        pts = []
        for i in range(points * 2):
            r = 1.0 if i % 2 == 0 else 0.45
            a = -math.pi / 2 + math.pi * i / points
            pts.append([cx + W / 2 * r * math.cos(a), cy + H / 2 * r * math.sin(a)])
        cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
    elif shape == "heart":
        t = np.linspace(0, 2 * math.pi, 400)
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        x = (x - x.min()) / (x.max() - x.min()) * (W - 1)
        y = (y - y.min()) / (y.max() - y.min()) * (H - 1)
        cv2.fillPoly(m, [np.stack([x, y], -1).astype(np.int32)], 255)
    elif shape == "image" and image_mask is not None:
        return cv2.resize(image_mask, (w, h), interpolation=cv2.INTER_AREA).astype(F32)
    return cv2.resize(m.astype(F32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)


@op("geometry.shape_crop", "Crop to shape", params=[
    Choice("shape", "circle", choices=SHAPES),
    Float("roundness", 0.15, min=0, max=1, step=0.01, description="Corner size for rounded/cut corners"),
    Int("points", 5, min=3, max=24, description="Star points"),
    File("mask_image", "", file_kind="image", description="Mask image when shape='image'"),
    Bool("crop_to_shape", True, description="For circle: crop to a square first"),
], tags=["shape crop", "circle crop", "avatar", "rounded corners", "mask", "heart", "star"], **G)
def shape_crop(img, ctx, shape, roundness, points, mask_image, crop_to_shape):
    """Cut the image to a shape (transparent outside)."""
    h, w = img.shape[:2]
    if shape == "circle" and crop_to_shape and w != h:
        s = min(w, h)
        y0, x0 = (h - s) // 2, (w - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]
        h, w = s, s
    im_mask = None
    if shape == "image":
        from ..core.io import load_image
        from ..core.image import gray

        mi, _ = load_image(ctx.resolve_file(mask_image, "mask image"))
        im_mask = mi[..., 3] if mi.shape[2] == 4 else gray(mi)
    m = shape_mask(shape, h, w, roundness, points, im_mask)
    out = ensure_rgba(img).copy()
    out[..., 3] *= m
    return out
