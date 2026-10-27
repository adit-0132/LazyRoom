"""Path construction for :mod:`lazyroom.ops.draw_modes` (private).

Ports ImageToolbox ``feature/draw`` geometry (Apache-2.0):

* ``BitmapDrawer`` pointer handling: free paths are ``moveTo(p0)``, then for every move
  event ``quadraticTo(previous, midpoint(previous, current))``, and ``lineTo(up)``;
* ``PathHelper``: line, arrows (``drawEndArrow`` / ``drawStartArrow``), rect, oval,
  triangle, polygon, star, spray;
* ``FloodFill``: scanline fill of the 4-connected region whose ARGB channels are all
  within ``round(tolerance * 255)`` of the seed pixel;
* ``AndroidImageDrawApplier.asPathEffect``: dashed / dot-dashed / zigzag / stamped line
  styles, using the Skia path-effect helpers of :mod:`lazyroom.ops.shapes`.

A path is a list of ``(points (n, 2) float64, closed)`` sub-paths in pixel coordinates.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from numba import njit, prange

from . import shapes as S
from ._draw_raster import dedupe

SubPath = tuple[np.ndarray, bool]
FLAT_TOL = 0.2  # px


# ---------------------------------------------------------------------------
# Free paths (quadratic midpoint smoothing)
# ---------------------------------------------------------------------------

def _flatten_quads(s: np.ndarray, c: np.ndarray, e: np.ndarray, tol: float = FLAT_TOL) -> np.ndarray:
    """Flatten quadratic Beziers (arrays of start/control/end) into one polyline (start excluded)."""
    dd = np.hypot(*(s - 2 * c + e).T)
    n = np.clip(np.ceil(np.sqrt(dd / (8 * tol))), 1, 256).astype(int)
    out = []
    for k in range(len(s)):
        t = (np.arange(1, n[k] + 1) / n[k])[:, None]
        u = 1 - t
        out.append(u * u * s[k] + 2 * u * t * c[k] + t * t * e[k])
    return np.concatenate(out) if out else np.zeros((0, 2))


def free_path(pts: np.ndarray) -> np.ndarray:
    """``M p0 (Q p[i-1], mid(p[i-1], p[i]))* L p[n-1]`` flattened (BitmapDrawer onBaseDraw / onUp)."""
    pts = np.asarray(pts, np.float64)
    if len(pts) == 1:
        return np.vstack([pts, pts])
    prev = pts[:-1]
    mids = (pts[:-1] + pts[1:]) / 2
    starts = np.vstack([pts[:1], mids[:-1]])
    body = _flatten_quads(starts, prev, mids)
    return np.vstack([pts[:1], body, pts[-1:]])


# ---------------------------------------------------------------------------
# Shapes (PathHelper)
# ---------------------------------------------------------------------------

def _rotate_about(pts: np.ndarray, deg: float, cx: float, cy: float) -> np.ndarray:
    """``Matrix.setRotate(deg, cx, cy)`` (positive = clockwise on screen, y down)."""
    if not deg:
        return pts
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    x, y = pts[:, 0] - cx, pts[:, 1] - cy
    return np.stack([cx + x * c - y * s, cy + x * s + y * c], 1)


def _ellipse_arc(cx, cy, rx, ry, a0, a1, tol=0.05) -> np.ndarray:
    r = max(abs(rx), abs(ry), 1e-9)
    n = max(2, int(math.ceil(abs(a1 - a0) / (2 * math.acos(max(-1.0, 1 - tol / r)) if r > tol else 1.0))))
    t = np.linspace(a0, a1, n + 1)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def rect_path(p0, p1, rotation: float, corner_radius: float) -> list[SubPath]:
    left, right = min(p0[0], p1[0]), max(p0[0], p1[0])
    top, bottom = min(p0[1], p1[1]), max(p0[1], p1[1])
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        return []
    r = min(w, h) * min(max(corner_radius, 0.0), 0.5)
    if r <= 0:  # SkPath::addRect, CCW from the top-left corner
        pts = np.array([[left, top], [left, bottom], [right, bottom], [right, top]], np.float64)
    else:  # SkPath::addRRect, CCW from (left, top + r): left edge down, then the corners
        hp = math.pi / 2
        pts = np.vstack([
            [[left, top + r]],
            _ellipse_arc(left + r, bottom - r, r, r, 2 * hp, hp),
            _ellipse_arc(right - r, bottom - r, r, r, hp, 0.0),
            _ellipse_arc(right - r, top + r, r, r, 0.0, -hp),
            _ellipse_arc(left + r, top + r, r, r, -hp, -2 * hp)[:-1],
        ])
    return [(_rotate_about(pts, rotation, (left + right) / 2, (top + bottom) / 2), True)]


def oval_path(p0, p1) -> list[SubPath]:
    """``addOval(Rect(top = max y, bottom = min y, ...))``: Skia starts at (right, cy)."""
    left, right = min(p0[0], p1[0]), max(p0[0], p1[0])
    top, bottom = max(p0[1], p1[1]), min(p0[1], p1[1])
    cx, cy = (left + right) / 2, (top + bottom) / 2
    rx, ry = (right - left) / 2, (top - bottom) / 2
    if rx <= 0 and ry <= 0:
        return []
    pts = _ellipse_arc(cx, cy, rx, ry, 0.0, 2 * math.pi)[:-1]
    return [(pts, True)]


def triangle_path(p0, p1) -> list[SubPath]:
    pts = np.array([[p0[0], p0[1]], [p1[0], p0[1]], [(p0[0] + p1[0]) / 2, p1[1]], [p0[0], p0[1]]], np.float64)
    return [(pts, True)]


def _box(p0, p1):
    # PathHelper.drawPolygon / drawStar: note top = max(y), bottom = min(y) -> height <= 0
    top, bottom = max(p0[1], p1[1]), min(p0[1], p1[1])
    left, right = min(p0[0], p1[0]), max(p0[0], p1[0])
    return (left + right) / 2, (top + bottom) / 2, right - left, bottom - top


def polygon_path(p0, p1, vertices: int, rotation: float, regular: bool) -> list[SubPath]:
    cx, cy, w, h = _box(p0, p1)
    k = np.arange(vertices)
    ang = np.radians(rotation - 270.0 + k * (360.0 / vertices))
    if regular:
        # upstream: radius = min(width, height) / 2 with height <= 0, i.e. always -|height| / 2
        # (a box-width bug); the magnitude is fixed to min(|w|, |h|) / 2, the sign kept.
        r = -min(abs(w), abs(h)) / 2
        pts = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], 1)
    else:
        pts = np.stack([cx + w / 2 * np.cos(ang), cy + h / 2 * np.sin(ang)], 1)
    return [(pts, True)]


def star_path(p0, p1, vertices: int, inner: float, rotation: float, regular: bool) -> list[SubPath]:
    cx, cy, w, h = _box(p0, p1)
    k = np.arange(2 * vertices)
    ang = np.radians(rotation - 270.0 + k * (360.0 / (2 * vertices)))
    outer_pt = (k % 2 == 0)
    if regular:
        ro = -min(abs(w), abs(h)) / 2  # same fix as polygon_path
        r = np.where(outer_pt, ro, ro * inner)
        pts = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], 1)
    else:
        rx = np.where(outer_pt, w, w * inner) / 2
        ry = np.where(outer_pt, h, h * inner) / 2
        pts = np.stack([cx + rx * np.cos(ang), cy + ry * np.sin(ang)], 1)
    return [(pts, True)]


# ---------------------------------------------------------------------------
# Arrow heads (PathHelper.drawEndArrow / drawStartArrow)
# ---------------------------------------------------------------------------

def _rot(v: np.ndarray, deg: float) -> np.ndarray:
    """``Offset.rotate``: standard rotation matrix."""
    t = math.radians(deg)
    return np.array([v[0] * math.cos(t) - v[1] * math.sin(t), v[0] * math.sin(t) + v[1] * math.cos(t)])


def _measure(pts: np.ndarray):
    pts = dedupe(pts)
    if len(pts) < 2:
        return None
    return S._Measure(pts)


def add_arrows(main: np.ndarray, stroke: float, size: float, angle: float, start: bool,
               end: bool) -> list[SubPath]:
    """Main open polyline plus arrow-head lines; the first end-head line continues the main contour."""
    extra: list[SubPath] = []
    meas = _measure(main)
    if end and meas is not None:
        (pre, last), _ = meas.pos_tan(np.array([meas.length - stroke * size, meas.length]))
        vec = last - pre
        # upstream also requires |vec.x| < size * stroke and |vec.y| < size * stroke (strict), which drops
        # the head of exactly horizontal / vertical arrows; only degenerate vectors are skipped here.
        if np.hypot(*vec) > 1e-9:
            main = np.vstack([main, last + _rot(vec, angle)])
            extra.append((np.array([last, last + _rot(vec, 360 - angle)]), False))
    meas = _measure(main)
    if start and meas is not None:
        (first, second), _ = meas.pos_tan(np.array([0.0, stroke * size]))
        vec = first - second
        if np.hypot(*vec) > 1e-9:
            extra.append((np.array([first, first + _rot(vec, angle)]), False))
            extra.append((np.array([first, first + _rot(vec, 360 - angle)]), False))
    return [(main, False)] + extra


# ---------------------------------------------------------------------------
# Spray (PathHelper Spray branch)
# ---------------------------------------------------------------------------

def spray_events(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Pointer positions: the stroke points, with gaps longer than ``spacing`` subdivided."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2 or spacing <= 0:
        return pts
    out = [pts[:1]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil(math.hypot(*(b - a)) / spacing)))
        t = (np.arange(1, n + 1) / n)[:, None]
        out.append(a + (b - a) * t)
    return np.concatenate(out)


def spray_particles(events: np.ndarray, density: int, stroke: float, rng: np.random.Generator):
    """``density`` particles per event, uniform in a disc of radius = stroke width."""
    u = rng.random((len(events), density, 2))
    ang = u[..., 0] * (2 * math.pi)
    rad = np.sqrt(u[..., 1]) * stroke
    xs = events[:, None, 0] + rad * np.cos(ang)
    ys = events[:, None, 1] + rad * np.sin(ang)
    return xs.ravel(), ys.ravel()


# ---------------------------------------------------------------------------
# Flood fill (presentation/components/utils/FloodFill.kt)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _argb8(base, layer, has_layer, y, x, out):
    """Composite (layer over premultiplied base) -> straight 8-bit A, R, G, B (Bitmap.getPixels)."""
    C = base.shape[2]
    ba = base[y, x, 3] if C == 4 else 1.0
    r, g, b, a = base[y, x, 0], base[y, x, 1], base[y, x, 2], ba
    if has_layer:
        k = 1.0 - layer[y, x, 3]
        r = layer[y, x, 0] + r * k
        g = layer[y, x, 1] + g * k
        b = layer[y, x, 2] + b * k
        a = layer[y, x, 3] + a * k
    if a > 1e-7:
        r, g, b = r / a, g / a, b / a
    else:
        r, g, b = 0.0, 0.0, 0.0
    out[0] = int(min(max(a, 0.0), 1.0) * 255.0 + 0.5)
    out[1] = int(min(max(r, 0.0), 1.0) * 255.0 + 0.5)
    out[2] = int(min(max(g, 0.0), 1.0) * 255.0 + 0.5)
    out[3] = int(min(max(b, 0.0), 1.0) * 255.0 + 0.5)


@njit(cache=True, parallel=True)
def _within(base, layer, has_layer, sx, sy, tol, out):
    seed = np.empty(4, np.int64)
    _argb8(base, layer, has_layer, sy, sx, seed)
    H, W = out.shape
    for y in prange(H):
        px = np.empty(4, np.int64)
        for x in range(W):
            _argb8(base, layer, has_layer, y, x, px)
            ok = 1
            for c in range(4):
                if px[c] < seed[c] - tol or px[c] > seed[c] + tol:
                    ok = 0
                    break
            out[y, x] = ok


def flood_fill_region(base: np.ndarray, layer: np.ndarray | None, x: int, y: int, tolerance: float):
    """FloodFill.performFloodFill on ``layer`` over ``base`` (premultiplied): the 4-connected region around
    (x, y) whose A, R, G, B (8-bit) are all within ``round(tolerance * 255)`` of the seed pixel.
    Returns ``(mask uint8 0/1 on box, box)``."""
    h, w = base.shape[:2]
    x = min(max(int(x), 0), w - 1)
    y = min(max(int(y), 0), h - 1)
    tol = min(max(int(round(tolerance * 255)), 0), 255)
    within = np.empty((h, w), np.uint8)
    has = layer is not None
    _within(base, layer if has else np.zeros((1, 1, 4), np.float32), has, x, y, tol, within)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    _, _, _, (bx, by, bw, bh) = cv2.floodFill(within, mask, (x, y), 2, 0, 0,
                                              4 | cv2.FLOODFILL_MASK_ONLY | (1 << 8))
    if bw <= 0 or bh <= 0:
        return None
    return mask[1 + by:1 + by + bh, 1 + bx:1 + bx + bw], (by, bx, by + bh, bx + bw)


# ---------------------------------------------------------------------------
# Line styles (AndroidImageDrawApplier.asPathEffect)
# ---------------------------------------------------------------------------

def _open(pts: np.ndarray, closed: bool) -> np.ndarray:
    return np.vstack([pts, pts[:1]]) if closed and len(pts) > 1 else pts


def dash_pieces(path: list[SubPath], intervals: list[float]) -> list[SubPath]:
    """``PathEffect.dashPathEffect(intervals, phase = 0)`` per contour."""
    out: list[SubPath] = []
    for pts, closed in path:
        meas = _measure(_open(pts, closed))
        if meas is None:
            out.append((pts, closed))
            continue
        out.extend((p, False) for p in S._dash_pieces(meas, intervals))
    return out


def zigzag_stamp(stroke: float, ratio: float):
    z = stroke / ratio
    o = stroke / 4
    d = (f"M{-o} {-o}L{stroke / 2 - o} {stroke / 2 - o}L{stroke - o} {-o}L{stroke - o} {z - o}"
         f"L{stroke / 2 - o} {stroke / 2 + z - o}L{-o} {z - o}Z")
    return S.parse_svg_path(d)


def stamp_shape(shape: str, stroke: float):
    name, corner = S.STAMP_SHAPES[shape]
    return S.shape_contours(name, stroke, stroke, corner)


def stamped_polys(path: list[SubPath], stamp, advance: float) -> list[np.ndarray]:
    """``PathEffect.stampedPathEffect(stamp, advance, 0, Morph)``: stamps bent along every contour."""
    polys: list[np.ndarray] = []
    for pts, closed in path:
        meas = _measure(_open(pts, closed))
        if meas is not None:
            polys.extend(S._stamp(meas, stamp, advance))
    return polys


def main_contour(path: list[SubPath]) -> np.ndarray | None:
    """First contour with a non-zero length (what ``SkPathMeasure`` / ``drawTextOnPath`` use)."""
    for pts, closed in path:
        p = dedupe(_open(pts, closed))
        if len(p) >= 2:
            return p
    return None


# ---------------------------------------------------------------------------
# Markup layers: ShapeLayerRenderer.buildShapePath (feature/markup-layers)
# ---------------------------------------------------------------------------

def markup_polygon(left, top, right, bottom, vertices: int, rotation: float, regular: bool) -> np.ndarray:
    """buildPolygonPath: start angle ``rotation - 90`` (first vertex on top), regular radius min(w, h) / 2."""
    cx, cy, w, h = (left + right) / 2, (top + bottom) / 2, right - left, bottom - top
    n = max(int(vertices), 3)
    ang = np.radians(rotation - 90.0 + np.arange(n) * (360.0 / n))
    rx, ry = (min(w, h) / 2, min(w, h) / 2) if regular else (w / 2, h / 2)
    return np.stack([cx + rx * np.cos(ang), cy + ry * np.sin(ang)], 1)


def markup_star(left, top, right, bottom, vertices: int, rotation: float, inner: float, regular: bool) -> np.ndarray:
    """buildStarPath."""
    cx, cy, w, h = (left + right) / 2, (top + bottom) / 2, right - left, bottom - top
    n = max(int(vertices), 3)
    inner = min(max(inner, 0.0), 1.0)
    k = np.arange(2 * n)
    ang = np.radians(rotation - 90.0 + k * (360.0 / (2 * n)))
    f = np.where(k % 2 == 0, 1.0, inner)
    rx, ry = (min(w, h) / 2, min(w, h) / 2) if regular else (w / 2, h / 2)
    return np.stack([cx + rx * f * np.cos(ang), cy + ry * f * np.sin(ang)], 1)


def markup_line_arrow(left, top, right, bottom, start: bool, end: bool, size: float, angle: float) -> list[SubPath]:
    """buildLineArrowPath: horizontal line through the box centre with V heads of length
    min(length / (3 if double else 2), max(box height, 1) * max(size, 0.5))."""
    cy = (top + bottom) / 2
    length = right - left
    head = max(min(length / (3.0 if (start and end) else 2.0), max(bottom - top, 1.0) * max(size, 0.5)), 1.0)
    out: list[SubPath] = [(np.array([[left, cy], [right, cy]], np.float64), False)]

    def head_at(tx, direction):
        for a in (direction + angle, direction - angle):
            t = math.radians(a)
            out.append((np.array([[tx, cy], [tx + math.cos(t) * head, cy + math.sin(t) * head]]), False))

    if end:
        head_at(right, 0.0)
    if start:
        head_at(left, 180.0)
    return out


def markup_filled_arrow(left, top, right, bottom, double: bool, size: float, angle: float) -> np.ndarray:
    """buildFilledArrowPath (block arrow)."""
    width, height = right - left, bottom - top
    cy = (top + bottom) / 2
    ar = math.radians(min(max(angle, 100.0), 175.0))
    ns = (min(max(size, 0.5), 8.0) - 0.5) / 7.5
    head_len = width * (0.22 + ns * 0.28)
    back = abs(math.cos(ar)) * head_len
    tip_half = min(height / 2, abs(math.sin(ar)) * head_len)
    shaft_half = max(min(height * 0.24, tip_half * 0.55), height * 0.12)
    if double:
        off = min(back, width / 2.5)
        lb, rb = left + off, right - off
        pts = [[left, cy], [lb, cy - tip_half], [lb, cy - shaft_half], [rb, cy - shaft_half], [rb, cy - tip_half],
               [right, cy], [rb, cy + tip_half], [rb, cy + shaft_half], [lb, cy + shaft_half], [lb, cy + tip_half]]
    else:
        off = min(back, width * 0.6)
        bx = right - off
        pts = [[left, cy - shaft_half], [bx, cy - shaft_half], [bx, cy - tip_half], [right, cy], [bx, cy + tip_half],
               [bx, cy + shaft_half], [left, cy + shaft_half]]
    return np.array(pts, np.float64)
