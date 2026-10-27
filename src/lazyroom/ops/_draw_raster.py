"""Vector rasterisation for :mod:`lazyroom.ops.draw_modes` (private).

* :func:`stroke_outline` builds the outline of a stroked polyline the way Skia's
  ``SkStroke`` does (left offset + joins, end cap, reversed right offset, start cap;
  inner joins routed through the pivot point; miter limit 4 = Android's default), so
  that filling the outline with the *nonzero* rule yields exactly the stroke.
* :func:`rasterize` fills polygons with the nonzero winding rule using 16 sub-scanlines
  per pixel row and exact horizontal span coverage.  Unlike signed-area accumulation
  (``shapes.fill_polylines``), the winding is resolved per sub-scanline, so overlapping
  pieces of one stroke (inner joins, self-crossing paths, overlapping stamps) union
  exactly instead of adding up their anti-aliasing.
* :func:`stamp_particles` rasterises spray particles (union, anti-aliased).

Everything works on a crop ("box") of the image: callers get ``(coverage, box)``
where ``box = (y0, x0, y1, x1)`` in pixel indices.  Coordinates are continuous pixel
coordinates (pixel ``i`` covers ``[i, i + 1)``).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from numba import njit, prange

F32 = np.float32
SUBSAMPLES = 16
MITER_LIMIT = 4.0  # android.graphics.Paint default
Box = tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------

def poly_box(polys: list[np.ndarray], h: int, w: int, pad: float = 0.0) -> Box | None:
    """Integer bounding box (clipped to the image) of polygons, grown by ``pad`` pixels."""
    pts = [p for p in polys if len(p)]
    if not pts:
        return None
    a = np.concatenate(pts)
    a = a[np.isfinite(a).all(axis=1)]
    if not len(a):
        return None
    x0 = max(0, int(math.floor(a[:, 0].min() - pad - 1)))
    y0 = max(0, int(math.floor(a[:, 1].min() - pad - 1)))
    x1 = min(w, int(math.ceil(a[:, 0].max() + pad + 1)))
    y1 = min(h, int(math.ceil(a[:, 1].max() + pad + 1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return y0, x0, y1, x1


def grow_box(box: Box, pad: float, h: int, w: int) -> Box:
    p = int(math.ceil(pad))
    return max(0, box[0] - p), max(0, box[1] - p), min(h, box[2] + p), min(w, box[3] + p)


def union_box(a: Box | None, b: Box | None) -> Box | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def sub(box: Box, inner: Box) -> tuple[slice, slice]:
    """Slices of ``inner`` relative to ``box`` (``inner`` must lie inside ``box``)."""
    return slice(inner[0] - box[0], inner[2] - box[0]), slice(inner[1] - box[1], inner[3] - box[1])


def place(cov: np.ndarray, box: Box, target: Box) -> np.ndarray:
    """Coverage ``cov`` (on ``box``) re-expressed on the larger ``target`` box."""
    out = np.zeros((target[2] - target[0], target[3] - target[1]), F32)
    ys, xs = sub(target, box)
    out[ys, xs] = cov
    return out


# ---------------------------------------------------------------------------
# Nonzero scanline rasteriser
# ---------------------------------------------------------------------------

@njit(cache=True)
def _add_span(cov, diff, row, xa, xb, wgt, W):
    if xb <= 0.0 or xa >= W or xb <= xa:
        return
    if xa < 0.0:
        xa = 0.0
    if xb > W:
        xb = float(W)
    ia = int(xa)
    ib = int(xb)
    if ia == ib:
        cov[row, ia] += (xb - xa) * wgt
        return
    cov[row, ia] += (ia + 1.0 - xa) * wgt
    if ib - ia > 1:
        diff[row, ia + 1] += wgt
        diff[row, ib] -= wgt
    if ib < W:
        cov[row, ib] += (xb - ib) * wgt


@njit(cache=True)
def _raster(ex0, ey0, ex1, ey1, H, W, ss):
    nsub = H * ss
    E = ex0.shape[0]
    counts = np.zeros(nsub + 1, np.int64)
    for e in range(E):
        y0 = ey0[e]
        y1 = ey1[e]
        if y0 == y1:
            continue
        lo = min(y0, y1)
        hi = max(y0, y1)
        j0 = int(math.ceil(lo * ss - 0.5))
        j1 = int(math.ceil(hi * ss - 0.5))
        j0 = max(j0, 0)
        j1 = min(j1, nsub)
        if j1 > j0:
            counts[j0] += 1
            counts[j1] -= 1
    per = np.zeros(nsub, np.int64)
    run = 0
    for j in range(nsub):
        run += counts[j]
        per[j] = run
    start = np.zeros(nsub + 1, np.int64)
    for j in range(nsub):
        start[j + 1] = start[j] + per[j]
    total = start[nsub]
    xs = np.empty(total, np.float64)
    ds = np.empty(total, np.int64)
    cur = start[:nsub].copy()
    for e in range(E):
        y0 = ey0[e]
        y1 = ey1[e]
        if y0 == y1:
            continue
        x0 = ex0[e]
        x1 = ex1[e]
        d = 1 if y1 > y0 else -1
        lo = min(y0, y1)
        hi = max(y0, y1)
        j0 = max(int(math.ceil(lo * ss - 0.5)), 0)
        j1 = min(int(math.ceil(hi * ss - 0.5)), nsub)
        k = (x1 - x0) / (y1 - y0)
        for j in range(j0, j1):
            yc = (j + 0.5) / ss
            xs[cur[j]] = x0 + (yc - y0) * k
            ds[cur[j]] = d
            cur[j] += 1
    cov = np.zeros((H, W), np.float32)
    diff = np.zeros((H, W + 1), np.float32)
    wgt = 1.0 / ss
    for j in range(nsub):
        a = start[j]
        b = start[j + 1]
        if b - a < 2:
            continue
        seg_x = xs[a:b]
        seg_d = ds[a:b]
        order = np.argsort(seg_x)
        row = j // ss
        wind = 0
        span0 = 0.0
        for t in range(b - a):
            i = order[t]
            prev = wind
            wind += seg_d[i]
            if prev == 0 and wind != 0:
                span0 = seg_x[i]
            elif prev != 0 and wind == 0:
                _add_span(cov, diff, row, span0, seg_x[i], wgt, W)
    for y in range(H):
        acc = 0.0
        for x in range(W):
            acc += diff[y, x]
            v = cov[y, x] + acc
            cov[y, x] = 1.0 if v > 1.0 else (0.0 if v < 1e-6 else v)
    return cov


def rasterize(polys: list[np.ndarray], h: int, w: int, pad: float = 0.0,
              box: Box | None = None) -> tuple[np.ndarray, Box] | None:
    """Nonzero-winding anti-aliased coverage of closed polygons (implicitly closed).

    Returns ``(coverage float32, box)`` on the polygons' bounding box grown by ``pad``
    (or on ``box`` when given), or ``None`` when nothing lies inside the image.
    """
    polys = [np.asarray(p, np.float64) for p in polys if len(p) >= 2]
    if box is None:
        box = poly_box(polys, h, w, pad)
    if box is None or not polys:
        return None
    y0, x0, y1, x1 = box
    segs = []
    for p in polys:
        q = p - (x0, y0)
        segs.append(np.concatenate([q, np.roll(q, -1, axis=0)], axis=1))
    s = np.concatenate(segs)
    s = s[np.isfinite(s).all(axis=1)]
    cov = _raster(np.ascontiguousarray(s[:, 0]), np.ascontiguousarray(s[:, 1]), np.ascontiguousarray(s[:, 2]),
                  np.ascontiguousarray(s[:, 3]), y1 - y0, x1 - x0, SUBSAMPLES)
    return cov, box


# ---------------------------------------------------------------------------
# Stroker (Skia SkStroke semantics for flattened paths)
# ---------------------------------------------------------------------------

def _arc_steps(r: float, sweep: float, tol: float = 0.05) -> int:
    if r <= tol:
        return max(1, int(math.ceil(abs(sweep) / (math.pi / 2))))
    step = 2 * math.acos(max(-1.0, 1 - tol / r))
    return max(1, int(math.ceil(abs(sweep) / max(step, 1e-3))))


def _arc(c, a, sweep: float, r: float) -> list:
    """Points of the arc around ``c`` starting at offset ``a`` (excluded) through ``sweep`` radians."""
    n = _arc_steps(r, sweep)
    ang0 = math.atan2(a[1], a[0])
    return [(c[0] + r * math.cos(ang0 + sweep * k / n), c[1] + r * math.sin(ang0 + sweep * k / n))
            for k in range(1, n + 1)]


def dedupe(pts: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > eps]
    return pts[keep]


def _join(out: list, p, a, b, r: float, outer: bool, join: str) -> None:
    """Append the join between offsets ``a`` (previous segment) and ``b`` (next) at ``p``."""
    dot = (a[0] * b[0] + a[1] * b[1]) / (r * r)
    if dot > 1 - 1e-9:  # Skia kNearlyLine: no join
        out.append((p[0] + b[0], p[1] + b[1]))
        return
    out.append((p[0] + a[0], p[1] + a[1]))
    if not outer:  # SkStrokerPriv HandleInnerJoin: go through the pivot
        out.append((p[0], p[1]))
    elif join == "round":
        cross = a[0] * b[1] - a[1] * b[0]
        sweep = math.atan2(cross, dot * r * r)
        out.extend(_arc(p, a, sweep, r)[:-1])
    elif join == "miter":
        cos_half = math.sqrt(max(0.0, (1 + dot) / 2))  # Skia's "sinHalfAngle"
        if cos_half > 1e-9 and 1.0 / cos_half <= MITER_LIMIT:
            mx, my = a[0] + b[0], a[1] + b[1]
            ml = math.hypot(mx, my)
            L = r / cos_half
            out.append((p[0] + mx / ml * L, p[1] + my / ml * L))
    out.append((p[0] + b[0], p[1] + b[1]))


def _cap(out: list, p, d, n, r: float, cap: str) -> None:
    """Cap at ``p`` facing direction ``d`` (unit), going from ``p + n`` to ``p - n``."""
    if cap == "round":
        # sweep from +n through +d to -n
        sweep = math.pi if (n[0] * d[1] - n[1] * d[0]) > 0 else -math.pi
        out.extend(_arc(p, n, sweep, r)[:-1])
    elif cap == "square":
        out.append((p[0] + n[0] + d[0] * r, p[1] + n[1] + d[1] * r))
        out.append((p[0] - n[0] + d[0] * r, p[1] - n[1] + d[1] * r))


def stroke_outline(pts: np.ndarray, closed: bool, width: float, cap: str = "round",
                   join: str = "round") -> list[np.ndarray]:
    """Polygons whose nonzero fill is the stroke of a polyline (``cap``: round|square|butt,
    ``join``: round|miter|bevel)."""
    r = width / 2.0
    if r <= 0:
        return []
    pts = dedupe(pts)
    if closed and len(pts) > 2 and np.hypot(*(pts[0] - pts[-1])) <= 1e-9:
        pts = pts[:-1]
    n = len(pts)
    if n == 0:
        return []
    if n == 1:  # zero-length subpath: Skia draws the cap shape only
        x, y = pts[0]
        if cap == "round":
            k = _arc_steps(r, 2 * math.pi)
            t = np.arange(k) * (2 * math.pi / k)
            return [np.stack([x + r * np.cos(t), y + r * np.sin(t)], 1)]
        if cap == "square":
            return [np.array([[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r]], np.float64)]
        return []
    seg_pts = np.vstack([pts, pts[:1]]) if closed else pts
    d = np.diff(seg_pts, axis=0)
    d = d / np.hypot(d[:, 0], d[:, 1])[:, None]
    nrm = np.stack([-d[:, 1], d[:, 0]], 1) * r
    m = len(d)

    def side(sign: float) -> list:
        out: list = []
        if closed:
            for k in range(n):
                a, b = nrm[k - 1] * sign, nrm[k] * sign
                turn = d[k - 1, 0] * d[k, 1] - d[k - 1, 1] * d[k, 0]
                _join(out, pts[k], a, b, r, (turn < 0) == (sign > 0), join)
            return out
        out.append(tuple(pts[0] + nrm[0] * sign))
        for k in range(1, m):
            a, b = nrm[k - 1] * sign, nrm[k] * sign
            turn = d[k - 1, 0] * d[k, 1] - d[k - 1, 1] * d[k, 0]
            _join(out, pts[k], a, b, r, (turn < 0) == (sign > 0), join)
        out.append(tuple(pts[-1] + nrm[-1] * sign))
        return out

    left = side(1.0)
    right = side(-1.0)
    if closed:
        return [np.array(left, np.float64), np.array(right[::-1], np.float64)]
    outline = list(left)
    _cap(outline, pts[-1], d[-1], nrm[-1], r, cap)
    outline.extend(right[::-1])
    _cap(outline, pts[0], -d[0], -nrm[0], r, cap)
    return [np.array(outline, np.float64)]


# ---------------------------------------------------------------------------
# Spray particles
# ---------------------------------------------------------------------------

@njit(cache=True)
def _stamp_particles(cov, xs, ys, size, square, ox, oy):
    H, W = cov.shape
    for k in range(xs.shape[0]):
        # particle = oval/rect (x, y, x + size, y + size) (PathHelper Spray)
        x0 = xs[k] - ox
        y0 = ys[k] - oy
        x1 = x0 + size
        y1 = y0 + size
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        rad = size * 0.5
        ia = max(0, int(math.floor(x0)) - 1)
        ib = min(W, int(math.ceil(x1)) + 1)
        ja = max(0, int(math.floor(y0)) - 1)
        jb = min(H, int(math.ceil(y1)) + 1)
        for j in range(ja, jb):
            for i in range(ia, ib):
                if square:
                    ow = min(i + 1.0, x1) - max(float(i), x0)
                    oh = min(j + 1.0, y1) - max(float(j), y0)
                    if ow <= 0.0 or oh <= 0.0:
                        continue
                    c = ow * oh
                else:
                    dx = i + 0.5 - cx
                    dy = j + 0.5 - cy
                    c = rad - math.sqrt(dx * dx + dy * dy) + 0.5
                    if c <= 0.0:
                        continue
                    c = min(c, 1.0, math.pi * rad * rad)
                if c > cov[j, i]:
                    cov[j, i] = min(c, 1.0)


def stamp_particles(xs: np.ndarray, ys: np.ndarray, size: float, square: bool, h: int, w: int,
                    pad: float = 0.0) -> tuple[np.ndarray, Box] | None:
    if not len(xs):
        return None
    pts = np.stack([xs, ys], 1)
    box = poly_box([pts, pts + size], h, w, pad)
    if box is None:
        return None
    cov = np.zeros((box[2] - box[0], box[3] - box[1]), F32)
    _stamp_particles(cov, np.ascontiguousarray(xs, np.float64), np.ascontiguousarray(ys, np.float64),
                     float(size), bool(square), float(box[1]), float(box[0]))
    return cov, box


# ---------------------------------------------------------------------------
# Blur (Android BlurMaskFilter / shadow layer radius -> sigma)
# ---------------------------------------------------------------------------

def radius_to_sigma(radius: float) -> float:
    """``SkBlurMask::ConvertRadiusToSigma`` (used by ``BlurMaskFilter`` and ``setShadowLayer``)."""
    return 0.57735 * radius + 0.5 if radius > 0 else 0.0


def blur_margin(radius: float) -> float:
    s = radius_to_sigma(radius)
    return math.ceil(3 * s) + 2 if s > 0 else 0.0


def blur(cov: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian blur of a coverage mask with ``sigma = radius_to_sigma(radius)``, transparent outside.

    Like Skia's raster mask blur, sigmas >= 2 use three box blurs (window
    ``d = floor(sigma * 3 * sqrt(2 * pi) / 4 + 0.5)``, the SVG/W3C approximation), which keeps
    large glows (neon defaults to sigma ~ 0.04 x the shorter side) fast."""
    s = radius_to_sigma(radius)
    if s <= 0:
        return cov
    if s < 2.0:
        return cv2.GaussianBlur(cov, (0, 0), s, borderType=cv2.BORDER_CONSTANT).astype(F32, copy=False)
    d = int(math.floor(s * 3 * math.sqrt(2 * math.pi) / 4 + 0.5))
    out = np.ascontiguousarray(cov, F32)
    if d % 2:
        for _ in range(3):
            out = cv2.blur(out, (d, d), borderType=cv2.BORDER_CONSTANT)
    else:
        out = cv2.blur(out, (d, d), anchor=(d // 2, d // 2), borderType=cv2.BORDER_CONSTANT)
        out = cv2.blur(out, (d, d), anchor=(d // 2 - 1, d // 2 - 1), borderType=cv2.BORDER_CONSTANT)
        out = cv2.blur(out, (d + 1, d + 1), borderType=cv2.BORDER_CONSTANT)
    return out


# ---------------------------------------------------------------------------
# Source-over compositing into premultiplied buffers (in place)
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def blend_solid(tgt, cov, r, g, b, alpha):
    """tgt = colour * (cov * alpha) over tgt (premultiplied RGB or RGBA view)."""
    H, W = cov.shape
    four = tgt.shape[2] == 4
    for y in prange(H):
        for x in range(W):
            a = cov[y, x] * alpha
            if a <= 0.0:
                continue
            inv = 1.0 - a
            tgt[y, x, 0] = tgt[y, x, 0] * inv + r * a
            tgt[y, x, 1] = tgt[y, x, 1] * inv + g * a
            tgt[y, x, 2] = tgt[y, x, 2] * inv + b * a
            if four:
                tgt[y, x, 3] = tgt[y, x, 3] * inv + a


@njit(cache=True, parallel=True)
def blend_src(tgt, src, cov, alpha):
    """tgt = src (premultiplied RGBA) * cov * alpha over tgt."""
    H, W = cov.shape
    four = tgt.shape[2] == 4
    for y in prange(H):
        for x in range(W):
            k = cov[y, x] * alpha
            sa = src[y, x, 3] * k
            if sa <= 0.0:
                continue
            inv = 1.0 - sa
            tgt[y, x, 0] = tgt[y, x, 0] * inv + src[y, x, 0] * k
            tgt[y, x, 1] = tgt[y, x, 1] * inv + src[y, x, 1] * k
            tgt[y, x, 2] = tgt[y, x, 2] * inv + src[y, x, 2] * k
            if four:
                tgt[y, x, 3] = tgt[y, x, 3] * inv + sa


@njit(cache=True, parallel=True)
def blend_straight(tgt, src, cov):
    """tgt = straight-alpha RGB(A) ``src`` through ``cov`` over tgt (premultiplied)."""
    H, W = cov.shape
    four = tgt.shape[2] == 4
    salpha = src.shape[2] == 4
    for y in prange(H):
        for x in range(W):
            sa = cov[y, x] * (src[y, x, 3] if salpha else 1.0)
            if sa <= 0.0:
                continue
            inv = 1.0 - sa
            tgt[y, x, 0] = tgt[y, x, 0] * inv + src[y, x, 0] * sa
            tgt[y, x, 1] = tgt[y, x, 1] * inv + src[y, x, 1] * sa
            tgt[y, x, 2] = tgt[y, x, 2] * inv + src[y, x, 2] * sa
            if four:
                tgt[y, x, 3] = tgt[y, x, 3] * inv + sa


@njit(cache=True, parallel=True)
def erase(tgt, cov):
    H, W = cov.shape
    for y in prange(H):
        for x in range(W):
            k = 1.0 - cov[y, x]
            if k < 1.0:
                for c in range(tgt.shape[2]):
                    tgt[y, x, c] *= k


@njit(cache=True, parallel=True)
def finish(base, layer, has_layer, out):
    """out = straight-alpha (layer over premultiplied base); RGB ``out`` for opaque bases."""
    H, W = out.shape[0], out.shape[1]
    four = base.shape[2] == 4
    for y in prange(H):
        for x in range(W):
            r, g, b = base[y, x, 0], base[y, x, 1], base[y, x, 2]
            a = base[y, x, 3] if four else 1.0
            if has_layer:
                k = 1.0 - layer[y, x, 3]
                r = layer[y, x, 0] + r * k
                g = layer[y, x, 1] + g * k
                b = layer[y, x, 2] + b * k
                a = layer[y, x, 3] + a * k
            if four:
                if a > 1e-7:
                    r, g, b = r / a, g / a, b / a
                else:
                    r, g, b = 0.0, 0.0, 0.0
                out[y, x, 3] = min(max(a, 0.0), 1.0)
            out[y, x, 0] = r
            out[y, x, 1] = g
            out[y, x, 2] = b
