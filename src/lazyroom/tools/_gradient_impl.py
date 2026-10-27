"""ImageToolbox Gradient Maker / mesh gradient port (feature/gradient-maker, Apache-2.0).

* ``UiGradientState.getBrush``: stop clean-up (sort, de-duplicate positions except for sweep, single
  stop duplicated), the linear start/end geometry derived from the angle, radial centre/radius
  (``max(w, h) / 2 * radiusFriction``) and sweep rotation (``setLocalMatrix(setRotate(angle))``).
* Android/Skia gradient evaluation: implicit stops at 0/1, monotonic pinning, tile modes clamp /
  repeated / mirror / decal, colours interpolated premultiplied in sRGB-encoded space (Android
  creates gradient shaders with ``kInterpolateColorsInPremul_Flag``), sampled at pixel centres.
* ``AndroidGradientMaker.drawMeshGradient`` / ``PointData``: control-point grid subdivided with
  cubic paths sampled by arc length (``PathMeasure`` -> Skia ``SkContourMeasure``), colours
  interpolated with Compose ``lerp(Color, Color)`` (Oklab, half-float storage, 8-bit sRGB result),
  drawn with ``Canvas.drawVertices(TRIANGLES)`` (Gouraud shading, premultiplied, source-over).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

TILE_MODES = ["clamp", "repeated", "mirror", "decal"]


# ------------------------------------------------------------------ colour stops (Skia preprocessing)

def prepare_stops(stops, sweep: bool) -> tuple[np.ndarray, np.ndarray]:
    """ITB ``getBrush`` stop clean-up + Skia ``SkGradientShaderBase`` dummy/pinned stops.

    ``stops``: [(pos, (r, g, b, a)), ...] -> (positions (N,), premultiplied colours (N, 4)).
    """
    st = sorted(stops, key=lambda s: s[0])  # stable, like sortedBy
    if not sweep:
        seen, uniq = set(), []
        for p, c in st:
            if p not in seen:
                seen.add(p)
                uniq.append((p, c))
        st = uniq
    if len(st) == 1:
        st = [st[0], st[0]]
    pos = [float(p) for p, _ in st]
    cols = [tuple(float(v) for v in c) for _, c in st]
    if pos[0] != 0.0:
        pos.insert(0, 0.0)
        cols.insert(0, cols[0])
    if pos[-1] != 1.0:
        pos.append(1.0)
        cols.append(cols[-1])
    prev = 0.0
    pinned = [0.0]
    for p in pos[1:-1]:
        prev = min(max(p, prev), 1.0)
        pinned.append(prev)
    pinned.append(1.0)
    c = np.array(cols, np.float64)
    c[:, :3] *= c[:, 3:4]
    return np.array(pinned, np.float64), c


def tile(t: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray | None]:
    if mode == "repeated":
        return t - np.floor(t), None
    if mode == "mirror":
        x = t - 1.0
        return np.abs(x - 2.0 * np.floor(x * 0.5) - 1.0), None
    if mode == "decal":
        return np.clip(t, 0.0, 1.0), (t < 0.0) | (t > 1.0)
    return np.clip(t, 0.0, 1.0), None


def eval_gradient(t: np.ndarray, pos: np.ndarray, premul: np.ndarray, mode: str) -> np.ndarray:
    """Evaluate a Skia-style gradient; returns unpremultiplied RGBA float32."""
    tt, outside = tile(t, mode)
    n = len(pos)
    idx = np.clip(np.searchsorted(pos[1:], tt, side="right"), 0, n - 2)
    lo, hi = pos[idx], pos[idx + 1]
    span = hi - lo
    f = np.where(span > 0, (tt - lo) / np.where(span > 0, span, 1.0), 0.0)[..., None]
    col = premul[idx] * (1.0 - f) + premul[idx + 1] * f
    if outside is not None:
        col[outside] = 0.0
    a = col[..., 3:4]
    rgb = np.where(a > 0, col[..., :3] / np.where(a > 0, a, 1.0), 0.0)
    return np.concatenate([np.clip(rgb, 0, 1), np.clip(a, 0, 1)], -1).astype(np.float32)


def _centres(width: int, height: int):
    return np.meshgrid(np.arange(width, dtype=np.float64) + 0.5, np.arange(height, dtype=np.float64) + 0.5)


def linear_t(width: int, height: int, angle: float) -> np.ndarray:
    """``GradientType.Linear``: start/end points exactly as UiGradientState computes them."""
    rad = angle / 180.0 * math.pi
    x, y = math.cos(rad), math.sin(rad)
    radius = math.sqrt(width ** 2 + height ** 2) / 2.0
    ox, oy = width / 2.0 + x * radius, height / 2.0 + y * radius
    ex = min(max(ox, 0.0), width)
    ey = height - min(max(oy, 0.0), height)
    sx, sy = width - ex, height - ey
    dx, dy = ex - sx, ey - sy
    d2 = dx * dx + dy * dy
    X, Y = _centres(width, height)
    if d2 == 0.0:  # Skia degenerate gradient (clamp -> last colour); cannot occur for valid angles
        return np.ones((height, width))
    return ((X - sx) * dx + (Y - sy) * dy) / d2


def radial_t(width: int, height: int, center, radius_friction: float) -> np.ndarray:
    cx, cy = width * center[0], height * center[1]
    r = max(max(width, height) / 2.0 * radius_friction, 0.01)
    X, Y = _centres(width, height)
    return np.hypot(X - cx, Y - cy) / r


def sweep_t(width: int, height: int, center, angle: float) -> np.ndarray:
    cx, cy = width * center[0], height * center[1]
    X, Y = _centres(width, height)
    a = np.arctan2(Y - cy, X - cx) - math.radians(angle)
    return np.mod(a / (2.0 * math.pi), 1.0)


# ------------------------------------------------------------------ Compose Color lerp (Oklab)

def _srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0.0), 1.0 / 2.4) - 0.055)


def _to_oklab(rgb):
    r, g, b = _srgb_to_linear(rgb[..., 0]), _srgb_to_linear(rgb[..., 1]), _srgb_to_linear(rgb[..., 2])
    l_ = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack([0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
                     1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
                     0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_], -1)


def _from_oklab(lab):
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3
    m_ = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3
    s_ = (L - 0.0894841775 * A - 1.2914855480 * B) ** 3
    r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    b = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076547010 * s_
    return np.clip(_linear_to_srgb(np.stack([r, g, b], -1)), 0.0, 1.0)


def _q8(v):
    return np.floor(np.clip(v, 0.0, 1.0) * 255.0 + 0.5) / 255.0


def compose_lerp(c0: np.ndarray, c1: np.ndarray, fraction) -> np.ndarray:
    """``androidx.compose.ui.graphics.lerp(Color, Color, Float)`` for sRGB colours (RGBA 0..1).

    Interpolates in Oklab; the intermediate Oklab Color stores L/a/b as half floats and alpha with
    10 bits, the result is converted back to an 8-bit sRGB Color.
    """
    fraction = np.asarray(fraction, np.float64)[..., None]
    a0, a1 = _to_oklab(c0[..., :3]), _to_oklab(c1[..., :3])
    lab = ((1.0 - fraction) * a0 + fraction * a1).astype(np.float16).astype(np.float64)
    alpha = (1.0 - fraction[..., 0]) * c0[..., 3] + fraction[..., 0] * c1[..., 3]
    alpha = np.floor(np.clip(alpha, 0.0, 1.0) * 1023.0 + 0.5) / 1023.0
    rgb = _from_oklab(lab)
    return np.concatenate([_q8(rgb), _q8(alpha)[..., None]], -1)


# ------------------------------------------------------------------ PathMeasure (SkContourMeasure)

_K_MAX_T = 0x3FFFFFFF
_TOLERANCE = 0.5  # CHEAP_DIST_LIMIT with resScale 1


def _chop_half(p):
    p0, p1, p2, p3 = p
    ab = (p0 + p1) / 2
    bc = (p1 + p2) / 2
    cd = (p2 + p3) / 2
    abc = (ab + bc) / 2
    bcd = (bc + cd) / 2
    abcd = (abc + bcd) / 2
    return [p0, ab, abc, abcd], [abcd, bcd, cd, p3]


def _too_curvy(p):
    def exceeds(pt, x, y):
        return max(abs(x - pt[0]), abs(y - pt[1])) > _TOLERANCE

    return (exceeds(p[1], p[0][0] + (p[3][0] - p[0][0]) / 3, p[0][1] + (p[3][1] - p[0][1]) / 3)
            or exceeds(p[2], p[0][0] + (p[3][0] - p[0][0]) * 2 / 3, p[0][1] + (p[3][1] - p[0][1]) * 2 / 3))


def _cubic_segments(p, distance, mint, maxt, segs):
    if ((maxt - mint) >> 10) and _too_curvy(p):
        half = (mint + maxt) >> 1
        a, b = _chop_half(p)
        distance = _cubic_segments(a, distance, mint, half, segs)
        return _cubic_segments(b, distance, half, maxt, segs)
    d = math.hypot(p[3][0] - p[0][0], p[3][1] - p[0][1])
    prev = distance
    distance += d
    if distance > prev:
        segs.append((distance, maxt))
    return distance


def _eval_cubic(p, t):
    p0, p1, p2, p3 = p
    a = p3 + 3.0 * (p1 - p2) - p0
    b = 3.0 * (p2 - 2.0 * p1 + p0)
    c = 3.0 * (p1 - p0)
    return ((a * t + b) * t + c) * t + p0


def sample_cubic_by_length(p, fractions) -> list[np.ndarray]:
    """``PathMeasure.getPosition(f * length)`` on ``moveTo(p0); cubicTo(p1, p2, p3); lineTo(p3)``."""
    p = [np.asarray(v, np.float64) for v in p]
    segs: list[tuple[float, int]] = []
    length = _cubic_segments(p, 0.0, 0, _K_MAX_T, segs)
    out = []
    for f in fractions:
        if not segs:  # zero-length contour: getPosTan fails -> Offset.Unspecified
            out.append(np.array([np.nan, np.nan]))
            continue
        dist = min(max(f * length, 0.0), length)
        k = 0
        while k < len(segs) - 1 and segs[k][0] < dist:
            k += 1
        start_d = segs[k - 1][0] if k > 0 else 0.0
        start_t = segs[k - 1][1] / _K_MAX_T if k > 0 else 0.0
        seg_t = segs[k][1] / _K_MAX_T
        t = start_t + (seg_t - start_t) * (dist - start_d) / (segs[k][0] - start_d)
        out.append(_eval_cubic(p, t))
    return out


def _cubic_x(p1, p2, position):
    delta = (p2[0] - p1[0]) * 0.5
    if position == 0:
        c1, c2 = (p1[0], p1[1]), (p2[0] - delta, p2[1])
    elif position == 2:
        c1, c2 = (p1[0] + delta, p1[1]), (p2[0], p2[1])
    else:
        c1, c2 = (p1[0] + delta, p1[1]), (p2[0] - delta, p2[1])
    return [p1, c1, c2, p2]


def _cubic_y(p1, p2, position):
    delta = (p2[1] - p1[1]) * 0.5
    if position == 0:
        c1, c2 = (p1[0], p1[1]), (p2[0], p2[1] - delta)
    elif position == 2:
        c1, c2 = (p1[0], p1[1] + delta), (p2[0], p2[1])
    else:
        c1, c2 = (p1[0], p1[1] + delta), (p2[0], p2[1] - delta)
    return [p1, c1, c2, p2]


def mesh_vertices(points: np.ndarray, colors: np.ndarray, steps_x: int, steps_y: int):
    """``PointData``: points (rows, cols, 2) normalised, colors (rows, cols, 4) -> (xy, rgba, triangles)."""
    rows, cols = points.shape[:2]
    xl = cols * steps_x - (steps_x - 1)
    yl = rows * steps_y - (steps_y - 1)
    off = np.zeros((yl, xl, 2), np.float64)
    col = np.zeros((yl, xl, 4), np.float64)
    for y in range(rows):
        for x in range(cols):
            off[y * steps_y, x * steps_x] = points[y, x]
            col[y * steps_y, x * steps_x] = colors[y, x]
            if x != cols - 1:
                position = 0 if x == 0 else (2 if x == cols - 2 else 1)
                path = _cubic_x(tuple(points[y, x]), tuple(points[y, x + 1]), position)
                fr = [i / float(steps_x) for i in range(1, steps_x)]
                pos = sample_cubic_by_length(path, fr)
                lerped = compose_lerp(colors[y, x][None], colors[y, x + 1][None], np.array(fr))
                for i in range(1, steps_x):
                    off[y * steps_y, x * steps_x + i] = pos[i - 1]
                    col[y * steps_y, x * steps_x + i] = lerped[i - 1]
    for y in range(rows - 1):
        position = 0 if y == 0 else (2 if y == cols - 2 else 1)  # upstream uses points[y].lastIndex
        fr = [i / float(steps_y) for i in range(1, steps_y)]
        for x in range(xl):
            path = _cubic_y(tuple(off[y * steps_y, x]), tuple(off[(y + 1) * steps_y, x]), position)
            pos = sample_cubic_by_length(path, fr)
            lerped = compose_lerp(col[y * steps_y, x][None], col[(y + 1) * steps_y, x][None], np.array(fr))
            for i in range(1, steps_y):
                off[y * steps_y + i, x] = pos[i - 1]
                col[y * steps_y + i, x] = lerped[i - 1]
    tris = []
    for y in range(yl - 1):
        for x in range(xl - 1):
            a = y * xl + x
            b = a + 1
            c = (y + 1) * xl + x
            d = c + 1
            tris.append((a, c, d))
            tris.append((a, b, d))
    return off.reshape(-1, 2), col.reshape(-1, 4), np.array(tris, np.int64)


@njit(cache=True)
def raster_triangles(out, vx, vy, cols, tris):
    """Gouraud-shaded triangles (premultiplied colours, source-over, pixel-centre coverage)."""
    h, w = out.shape[0], out.shape[1]
    for k in range(tris.shape[0]):
        i0, i1, i2 = tris[k, 0], tris[k, 1], tris[k, 2]
        x0, y0, x1, y1, x2, y2 = vx[i0], vy[i0], vx[i1], vy[i1], vx[i2], vy[i2]
        if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1)
                and math.isfinite(x2) and math.isfinite(y2)):
            continue
        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if area == 0.0:
            continue
        minx = max(int(math.floor(min(x0, min(x1, x2)) - 0.5)), 0)
        maxx = min(int(math.ceil(max(x0, max(x1, x2)) - 0.5)), w - 1)
        miny = max(int(math.floor(min(y0, min(y1, y2)) - 0.5)), 0)
        maxy = min(int(math.ceil(max(y0, max(y1, y2)) - 0.5)), h - 1)
        for py in range(miny, maxy + 1):
            cy = py + 0.5
            for px in range(minx, maxx + 1):
                cx = px + 0.5
                w0 = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
                w1 = (x0 - x2) * (cy - y2) - (y0 - y2) * (cx - x2)
                w2 = (x1 - x0) * (cy - y0) - (y1 - y0) * (cx - x0)
                if area > 0.0:
                    if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                        continue
                elif w0 > 0.0 or w1 > 0.0 or w2 > 0.0:
                    continue
                b0, b1, b2 = w0 / area, w1 / area, w2 / area
                sa = b0 * cols[i0, 3] + b1 * cols[i1, 3] + b2 * cols[i2, 3]
                inv = 1.0 - sa
                for c in range(3):
                    s = b0 * cols[i0, c] + b1 * cols[i1, c] + b2 * cols[i2, c]
                    out[py, px, c] = s + out[py, px, c] * inv
                out[py, px, 3] = sa + out[py, px, 3] * inv


def render_mesh(width: int, height: int, points: np.ndarray, colors: np.ndarray, resolution: int) -> np.ndarray:
    """Mesh gradient on a transparent canvas -> unpremultiplied RGBA float32."""
    xy, rgba, tris = mesh_vertices(points, colors, resolution, resolution)
    pre = rgba.copy()
    pre[:, :3] *= pre[:, 3:4]
    out = np.zeros((height, width, 4), np.float64)
    raster_triangles(out, xy[:, 0] * width, xy[:, 1] * height, pre, tris)
    a = out[..., 3:4]
    rgb = np.where(a > 0, out[..., :3] / np.where(a > 0, a, 1.0), 0.0)
    return np.concatenate([np.clip(rgb, 0, 1), np.clip(a, 0, 1)], -1).astype(np.float32)


def max_mesh_resolution(grid_size: int) -> int:
    """``GradientMakerComponent.meshResolutionMax``."""
    return {6: 24, 5: 32, 4: 40}.get(grid_size, 64)
