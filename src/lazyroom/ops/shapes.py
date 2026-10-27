"""ImageToolbox shape catalogue as anti-aliased vector shapes, plus the markup line styles.

* ``SHAPES``: shape id -> ``render(h, w, **params) -> float32 (h, w) coverage mask``, in ImageToolbox's
  crop-mask order (``feature/crop/.../DefaultOutlineProperties.kt``) followed by the shapes that only
  appear in the icon-shape catalogue (``core/settings/.../IconShape.kt``).
* ``geometry.crop_to_shape``: ImageToolbox "Crop with shape mask" (``lib/cropper`` ``CropAgent``).
* ``draw.styled_line``: ImageToolbox draw line styles (``feature/draw`` ``DrawLineStyle`` rendered by
  ``AndroidImageDrawApplier.asPathEffect``: dashed, dot-dashed, zigzag, stamped).

Shape geometry sources (all Apache-2.0):
  * ImageToolbox ``core/resources/.../shapes/*.kt`` (path data and base sizes), ``lib/cropper``
    (``Paths.Favorite`` / ``Paths.Star`` and how ``CropAgent`` scales paths vs. shapes);
  * AOSP ``compose/material3/MaterialShapes.kt`` and ``graphics/graphics-shapes`` (``RoundedPolygon``,
    ``CornerRounding``, ``Cubic``, ``Shapes.kt``), ported below so the 30 Material 3 expressive shapes
    ImageToolbox offers are built with the same rounding/smoothing algorithm;
  * ImageToolbox ``RoundedPolygonUtils.toShape`` (polygon path scaled by the crop size).

Paths are kept as cubic Beziers (lines flagged) and filled with an exact-area scanline rasteriser
(font-rs style signed-area accumulation), i.e. analytic anti-aliasing at any size.
"""
from __future__ import annotations

import math
import re
from typing import Callable

import cv2
import numpy as np
from numba import njit

from ..core.errors import invalid
from ..core.image import ensure_rgba
from ..core.params import Bool, Choice, Color, File, Float, Int, Points
from ..core.registry import op

F32 = np.float32

# =============================================================================
# Paths: list of contours; a contour is an (n, 8) float64 array of cubics
# (x0, y0, c0x, c0y, c1x, c1y, x1, y1) plus an (n,) bool array "is straight line".
# =============================================================================

Contour = tuple[np.ndarray, np.ndarray]

_TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _line(x0, y0, x1, y1):
    return (x0, y0, x0 + (x1 - x0) / 3, y0 + (y1 - y0) / 3, x0 + 2 * (x1 - x0) / 3, y0 + 2 * (y1 - y0) / 3, x1, y1)


def parse_svg_path(d: str) -> list[Contour]:
    """SVG path data (M L H V C S Q T Z, absolute + relative) like Compose's ``PathParser``."""
    toks = _TOKEN.findall(d)
    contours: list[Contour] = []
    segs: list[tuple] = []
    lines: list[bool] = []
    i, cmd = 0, ""
    cx = cy = sx = sy = 0.0
    lcx = lcy = None  # last cubic / quad control point (for S / T)
    last = ""

    def flush():
        nonlocal segs, lines
        if segs:
            contours.append((np.array(segs, np.float64), np.array(lines, bool)))
        segs, lines = [], []

    def num():
        nonlocal i
        v = float(toks[i])
        i += 1
        return v

    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in "Zz":
                if segs and (cx, cy) != (sx, sy):
                    segs.append(_line(cx, cy, sx, sy))
                    lines.append(True)
                flush()
                cx, cy = sx, sy
                last = "Z"
                continue
        elif not cmd:
            raise ValueError(f"path data must start with a command: {d[:20]!r}")
        rel = cmd.islower()
        c = cmd.upper()
        ox, oy = (cx, cy) if rel else (0.0, 0.0)
        if c == "M":
            flush()
            cx, cy = num() + ox, num() + oy
            sx, sy = cx, cy
            cmd = "l" if rel else "L"  # implicit lineto after moveto
            last = "M"
            continue
        if c in "LHV":
            if c == "L":
                nx, ny = num() + ox, num() + oy
            elif c == "H":
                nx, ny = num() + ox, cy
            else:
                nx, ny = cx, num() + oy
            segs.append(_line(cx, cy, nx, ny))
            lines.append(True)
            cx, cy, last = nx, ny, "L"
        elif c in "CS":
            if c == "C":
                x1, y1 = num() + ox, num() + oy
            else:
                x1, y1 = (2 * cx - lcx, 2 * cy - lcy) if last == "C" else (cx, cy)
            x2, y2, nx, ny = num() + ox, num() + oy, num() + ox, num() + oy
            segs.append((cx, cy, x1, y1, x2, y2, nx, ny))
            lines.append(False)
            lcx, lcy, cx, cy, last = x2, y2, nx, ny, "C"
        elif c in "QT":
            if c == "Q":
                qx, qy = num() + ox, num() + oy
            else:
                qx, qy = (2 * cx - lcx, 2 * cy - lcy) if last == "Q" else (cx, cy)
            nx, ny = num() + ox, num() + oy
            segs.append((cx, cy, cx + 2 / 3 * (qx - cx), cy + 2 / 3 * (qy - cy),
                         nx + 2 / 3 * (qx - nx), ny + 2 / 3 * (qy - ny), nx, ny))
            lines.append(False)
            lcx, lcy, cx, cy, last = qx, qy, nx, ny, "Q"
        else:
            raise ValueError(f"unsupported path command {cmd!r}")
    flush()
    return contours


def path_bounds(contours: list[Contour]) -> tuple[float, float, float, float]:
    """Control-point bounds, like Skia ``SkPath::getBounds`` (Compose ``Path.getBounds``)."""
    pts = np.concatenate([c[:, :].reshape(-1, 2) for c, _ in contours])
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def transform(contours: list[Contour], sx: float, sy: float, tx: float = 0.0, ty: float = 0.0) -> list[Contour]:
    m = np.array([sx, sy] * 4)
    t = np.array([tx, ty] * 4)
    return [(c * m + t, ln) for c, ln in contours]


def flatten(contours: list[Contour], tol: float = 0.1) -> list[np.ndarray]:
    """Polylines (closed implicitly) — cubics subdivided with Wang's formula for ``tol`` px."""
    out = []
    for cub, ln in contours:
        p = cub.reshape(-1, 4, 2)
        dd = np.maximum(np.hypot(*(p[:, 0] - 2 * p[:, 1] + p[:, 2]).T), np.hypot(*(p[:, 1] - 2 * p[:, 2] + p[:, 3]).T))
        n = np.where(ln, 1, np.clip(np.ceil(np.sqrt(0.75 * dd / tol)), 1, 512)).astype(int)
        pts = [p[0, 0][None]]
        for k in range(len(p)):
            if n[k] == 1:
                pts.append(p[k, 3][None])
                continue
            t = (np.arange(1, n[k] + 1) / n[k])[:, None]
            u = 1 - t
            pts.append(u ** 3 * p[k, 0] + 3 * u * u * t * p[k, 1] + 3 * u * t * t * p[k, 2] + t ** 3 * p[k, 3])
        out.append(np.concatenate(pts))
    return out


# -----------------------------------------------------------------------------
# Exact-area rasteriser (signed area accumulation, after Raph Levien's font-rs)
# -----------------------------------------------------------------------------

@njit(cache=True)
def _acc_line(a, w, h, px0, py0, px1, py1):
    if abs(py0 - py1) <= 1e-9:
        return
    if py0 < py1:
        d0, x0_, y0_, y1_ = 1.0, px0, py0, py1
        dxdy = (px1 - px0) / (py1 - py0)
    else:
        d0, x0_, y0_, y1_ = -1.0, px1, py1, py0
        dxdy = (px0 - px1) / (py0 - py1)
    x = x0_
    if y0_ < 0.0:
        x -= y0_ * dxdy
    ys = max(0, int(math.floor(y0_)))
    ye = min(h, int(math.ceil(y1_)))
    for y in range(ys, ye):
        dy = min(y + 1.0, y1_) - max(float(y), y0_)
        xnext = x + dxdy * dy
        d = dy * d0
        xa, xb = (x, xnext) if x < xnext else (xnext, x)
        xa = min(max(xa, 0.0), float(w))
        xb = min(max(xb, 0.0), float(w))
        x0floor = math.floor(xa)
        x0i = int(x0floor)
        x1ceil = math.ceil(xb)
        x1i = int(x1ceil)
        if x1i <= x0i + 1:
            xmf = 0.5 * (xa + xb) - x0floor
            a[y, x0i] += d - d * xmf
            a[y, x0i + 1] += d * xmf
        else:
            s = 1.0 / (xb - xa)
            x0f = xa - x0floor
            a0 = 0.5 * s * (1.0 - x0f) * (1.0 - x0f)
            x1f = xb - x1ceil + 1.0
            am = 0.5 * s * x1f * x1f
            a[y, x0i] += d * a0
            if x1i == x0i + 2:
                a[y, x0i + 1] += d * (1.0 - a0 - am)
            else:
                a1 = s * (1.5 - x0f)
                a[y, x0i + 1] += d * (a1 - a0)
                for xi in range(x0i + 2, x1i - 1):
                    a[y, xi] += d * s
                a2 = a1 + (x1i - x0i - 3) * s
                a[y, x1i - 1] += d * (1.0 - a2 - am)
            a[y, x1i] += d * am
        x = xnext


@njit(cache=True)
def _acc_segments(segs, w, h):
    a = np.zeros((h, w + 2), np.float64)
    for k in range(segs.shape[0]):
        x0, y0, x1, y1 = segs[k, 0], segs[k, 1], segs[k, 2], segs[k, 3]
        # split at x = 0 and x = w so that clamping x to [0, w] is exact
        t1, t2 = 2.0, 2.0
        if (x0 < 0.0) != (x1 < 0.0):
            t1 = (0.0 - x0) / (x1 - x0)
        if (x0 < w) != (x1 < w):
            t2 = (w - x0) / (x1 - x0)
        if t1 > t2:
            t1, t2 = t2, t1
        px, py = x0, y0
        for t in (t1, t2, 1.0):
            if t > 1.0:
                continue
            qx, qy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            _acc_line(a, w, h, px, py, qx, qy)
            px, py = qx, qy
    return a


@njit(cache=True)
def _resolve(a, w):
    """Row prefix sums of the signed-area buffer -> coverage min(|sum|, 1)."""
    h = a.shape[0]
    out = np.empty((h, w), np.float32)
    for y in range(h):
        s = 0.0
        for x in range(w):
            s += a[y, x]
            v = abs(s)
            out[y, x] = 0.0 if v < 1e-6 else (1.0 if v > 1.0 - 1e-6 else v)  # snap float residue
    return out


def fill_polylines(polys: list[np.ndarray], h: int, w: int) -> np.ndarray:
    """Anti-aliased coverage (0..1, float32) of closed polylines in pixel coordinates.

    Exact area coverage; overlapping contours combine like the nonzero rule (|winding| clamped to 1).
    """
    segs = [np.concatenate([p, np.roll(p, -1, axis=0)], axis=1) for p in polys if len(p) >= 2]
    if not segs or h < 1 or w < 1:
        return np.zeros((max(h, 0), max(w, 0)), F32)
    return _resolve(_acc_segments(np.ascontiguousarray(np.concatenate(segs), np.float64), int(w), int(h)), int(w))


def fill_path(contours: list[Contour], h: int, w: int) -> np.ndarray:
    return fill_polylines(flatten(contours), h, w)


# =============================================================================
# androidx.graphics.shapes RoundedPolygon (graphics-shapes, AOSP, Apache-2.0)
# =============================================================================

_EPS = 1e-4  # DistanceEpsilon


def _dist(x, y):
    return math.sqrt(x * x + y * y)


def _dir(x, y):
    d = _dist(x, y)
    if d <= 0:
        raise ValueError("zero-length direction")
    return (x / d, y / d)


def _straight(x0, y0, x1, y1):
    return [x0, y0, x0 + (x1 - x0) / 3, y0 + (y1 - y0) / 3, x0 + (x1 - x0) * 2 / 3, y0 + (y1 - y0) * 2 / 3, x1, y1]


def _circular_arc(cx, cy, x0, y0, x1, y1):
    p0d = _dir(x0 - cx, y0 - cy)
    p1d = _dir(x1 - cx, y1 - cy)
    r0 = (-p0d[1], p0d[0])
    r1 = (-p1d[1], p1d[0])
    clockwise = r0[0] * (x1 - cx) + r0[1] * (y1 - cy) >= 0
    cosa = p0d[0] * p1d[0] + p0d[1] * p1d[1]
    if cosa > 0.999:
        return _straight(x0, y0, x1, y1)
    k = (_dist(x0 - cx, y0 - cy) * 4 / 3 * (math.sqrt(2 * (1 - cosa)) - math.sqrt(1 - cosa * cosa)) / (1 - cosa)
         * (1 if clockwise else -1))
    return [x0, y0, x0 + r0[0] * k, y0 + r0[1] * k, x1 - r1[0] * k, y1 - r1[1] * k, x1, y1]


def _split(c, t):
    u = 1 - t
    a0x, a0y, c0x, c0y, c1x, c1y, a1x, a1y = c
    px = a0x * u ** 3 + c0x * 3 * t * u * u + c1x * 3 * t * t * u + a1x * t ** 3
    py = a0y * u ** 3 + c0y * 3 * t * u * u + c1y * 3 * t * t * u + a1y * t ** 3
    first = [a0x, a0y, a0x * u + c0x * t, a0y * u + c0y * t,
             a0x * u * u + c0x * 2 * u * t + c1x * t * t, a0y * u * u + c0y * 2 * u * t + c1y * t * t, px, py]
    second = [px, py, c0x * u * u + c1x * 2 * u * t + a1x * t * t, c0y * u * u + c1y * 2 * u * t + a1y * t * t,
              c1x * u + a1x * t, c1y * u + a1y * t, a1x, a1y]
    return first, second


def _zero_length(c):
    return abs(c[0] - c[6]) < _EPS and abs(c[1] - c[7]) < _EPS


class _RoundedCorner:
    def __init__(self, p0, p1, p2, radius, smoothing):
        self.p0, self.p1, self.p2 = p0, p1, p2
        v01 = (p0[0] - p1[0], p0[1] - p1[1])
        v21 = (p2[0] - p1[0], p2[1] - p1[1])
        d01, d21 = _dist(*v01), _dist(*v21)
        if d01 > 0 and d21 > 0:
            self.d1 = (v01[0] / d01, v01[1] / d01)
            self.d2 = (v21[0] / d21, v21[1] / d21)
            self.radius, self.smoothing = radius, smoothing
            cos_a = self.d1[0] * self.d2[0] + self.d1[1] * self.d2[1]
            sin_a = math.sqrt(max(0.0, 1 - cos_a * cos_a))
            self.expected_round_cut = radius * (cos_a + 1) / sin_a if sin_a > 1e-3 else 0.0
        else:
            self.d1 = self.d2 = (0.0, 0.0)
            self.radius = self.smoothing = 0.0
            self.expected_round_cut = 0.0
        self.expected_cut = (1 + self.smoothing) * self.expected_round_cut

    def _actual_smoothing(self, allowed):
        if allowed > self.expected_cut:
            return self.smoothing
        if allowed > self.expected_round_cut:
            return self.smoothing * (allowed - self.expected_round_cut) / (self.expected_cut - self.expected_round_cut)
        return 0.0

    def cubics(self, cut0, cut1):
        p1 = self.p1
        allowed = min(cut0, cut1)
        if self.expected_round_cut < _EPS or allowed < _EPS or self.radius < _EPS:
            return [_straight(p1[0], p1[1], p1[0], p1[1])]
        round_cut = min(allowed, self.expected_round_cut)
        s0, s1 = self._actual_smoothing(cut0), self._actual_smoothing(cut1)
        r = self.radius * round_cut / self.expected_round_cut
        cdist = math.sqrt(r * r + round_cut * round_cut)
        dm = _dir((self.d1[0] + self.d2[0]) / 2, (self.d1[1] + self.d2[1]) / 2)
        center = (p1[0] + dm[0] * cdist, p1[1] + dm[1] * cdist)
        ci0 = (p1[0] + self.d1[0] * round_cut, p1[1] + self.d1[1] * round_cut)
        ci2 = (p1[0] + self.d2[0] * round_cut, p1[1] + self.d2[1] * round_cut)
        f0 = self._flanking(round_cut, s0, p1, self.p0, ci0, ci2, center, r)
        f2 = self._flanking(round_cut, s1, p1, self.p2, ci2, ci0, center, r)
        f2 = f2[6:8] + f2[4:6] + f2[2:4] + f2[0:2]  # reverse
        arc = _circular_arc(center[0], center[1], f0[6], f0[7], f2[0], f2[1])
        return [f0, arc, f2]

    @staticmethod
    def _flanking(round_cut, smooth, corner, side_start, csi, other_csi, cc, r):
        sd = _dir(side_start[0] - corner[0], side_start[1] - corner[1])
        cs = (corner[0] + sd[0] * round_cut * (1 + smooth), corner[1] + sd[1] * round_cut * (1 + smooth))
        mid = ((csi[0] + other_csi[0]) / 2, (csi[1] + other_csi[1]) / 2)
        p = ((1 - smooth) * csi[0] + smooth * mid[0], (1 - smooth) * csi[1] + smooth * mid[1])
        dv = _dir(p[0] - cc[0], p[1] - cc[1])
        ce = (cc[0] + dv[0] * r, cc[1] + dv[1] * r)
        tangent = (-(ce[1] - cc[1]), ce[0] - cc[0])
        # lineIntersection(side_start, sd, ce, tangent)
        rd1 = (-tangent[1], tangent[0])
        den = sd[0] * rd1[0] + sd[1] * rd1[1]
        num = (ce[0] - side_start[0]) * rd1[0] + (ce[1] - side_start[1]) * rd1[1]
        if abs(den) < _EPS or abs(den) < _EPS * abs(num):
            ae = csi
        else:
            k = num / den
            ae = (side_start[0] + sd[0] * k, side_start[1] + sd[1] * k)
        a_s = ((cs[0] + ae[0] * 2) / 3, (cs[1] + ae[1] * 2) / 3)
        return [cs[0], cs[1], a_s[0], a_s[1], ae[0], ae[1], ce[0], ce[1]]


class RoundedPolygon:
    """``features`` = list of cubic lists (corner, edge, corner, edge, ...); ``center`` (x, y)."""

    def __init__(self, features: list[list[list[float]]], center: tuple[float, float]):
        self.features, self.center = features, center
        self.cubics = self._cubics()

    def _cubics(self):
        f = self.features
        out: list[list[float]] = []
        first = last = None
        split_start = split_end = None
        if f and len(f[0]) == 3:
            s, e = _split(f[0][1], 0.5)
            split_start, split_end = [f[0][0], s], [e, f[0][2]]
        for i in range(len(f) + 1):
            if i == 0 and split_end is not None:
                fc = split_end
            elif i == len(f):
                if split_start is None:
                    break
                fc = split_start
            else:
                fc = f[i]
            for c in fc:
                if not _zero_length(c):
                    if last is not None:
                        out.append(last)
                    last = c
                    if first is None:
                        first = c
                elif last is not None:
                    last = list(last)
                    last[6], last[7] = c[0], c[1]
        if last is not None and first is not None:
            out.append(last[:6] + [first[0], first[1]])
        else:
            cx, cy = self.center
            out.append([cx, cy] * 4)
        return out

    def transformed(self, fn: Callable[[float, float], tuple[float, float]]) -> "RoundedPolygon":
        def tc(c):
            r = []
            for k in range(0, 8, 2):
                r.extend(fn(c[k], c[k + 1]))
            return r
        return RoundedPolygon([[tc(c) for c in feat] for feat in self.features], fn(*self.center))

    def bounds(self):  # calculateBounds(approximate = true)
        xs, ys = [], []
        for c in self.cubics:
            if _zero_length(c):
                xs.append(c[0])
                ys.append(c[1])
            else:
                xs.extend(c[0::2])
                ys.extend(c[1::2])
        return min(xs), min(ys), max(xs), max(ys)

    def normalized(self) -> "RoundedPolygon":
        l, t, r, b = self.bounds()
        w, h = r - l, b - t
        side = max(w, h)
        ox, oy = (side - w) / 2 - l, (side - h) / 2 - t
        return self.transformed(lambda x, y: ((x + ox) / side, (y + oy) / side))

    def contours(self) -> list[Contour]:
        return [(np.array(self.cubics, np.float64), np.zeros(len(self.cubics), bool))]


def rounded_polygon(vertices: list[tuple[float, float]], roundings: list[tuple[float, float]],
                    center: tuple[float, float] | None = None) -> RoundedPolygon:
    """``RoundedPolygon(vertices, perVertexRounding, centerX, centerY)``; rounding = (radius, smoothing)."""
    n = len(vertices)
    corners = [_RoundedCorner(vertices[(i - 1) % n], vertices[i], vertices[(i + 1) % n], *roundings[i])
               for i in range(n)]
    adjusts = []
    for i in range(n):
        erc = corners[i].expected_round_cut + corners[(i + 1) % n].expected_round_cut
        ec = corners[i].expected_cut + corners[(i + 1) % n].expected_cut
        side = _dist(vertices[i][0] - vertices[(i + 1) % n][0], vertices[i][1] - vertices[(i + 1) % n][1])
        if erc > side:
            adjusts.append((side / erc, 0.0))
        elif ec > side:
            adjusts.append((1.0, (side - erc) / (ec - erc)))
        else:
            adjusts.append((1.0, 1.0))
    corner_cubics = []
    for i in range(n):
        cuts = []
        for delta in (0, 1):
            rr, cr = adjusts[(i + n - 1 + delta) % n]
            c = corners[i]
            cuts.append(c.expected_round_cut * rr + (c.expected_cut - c.expected_round_cut) * cr)
        corner_cubics.append(corners[i].cubics(cuts[0], cuts[1]))
    feats = []
    for i in range(n):
        feats.append(corner_cubics[i])
        a, b = corner_cubics[i][-1], corner_cubics[(i + 1) % n][0]
        feats.append([_straight(a[6], a[7], b[0], b[1])])
    if center is None:
        center = (sum(v[0] for v in vertices) / n, sum(v[1] for v in vertices) / n)
    return RoundedPolygon(feats, center)


def _ngon_vertices(n, radius=1.0, cx=0.0, cy=0.0):
    return [(radius * math.cos(math.pi / n * 2 * i) + cx, radius * math.sin(math.pi / n * 2 * i) + cy)
            for i in range(n)]


def _rp_ngon(n, rounding=(0.0, 0.0), per_vertex=None, radius=1.0):
    return rounded_polygon(_ngon_vertices(n, radius), per_vertex or [rounding] * n, (0.0, 0.0))


def _rp_circle(n=8, radius=1.0):
    return _rp_ngon(n, (radius, 0.0), radius=radius / math.cos(math.pi / n))


def _rp_rectangle(width=2.0, height=2.0, rounding=(0.0, 0.0), per_vertex=None):
    l, t, r, b = -width / 2, -height / 2, width / 2, height / 2
    return rounded_polygon([(r, b), (l, b), (l, t), (r, t)], per_vertex or [rounding] * 4, (0.0, 0.0))


def _rp_star(npr, radius=1.0, inner=0.5, rounding=(0.0, 0.0)):
    v = []
    for i in range(npr):
        a = math.pi / npr * 2 * i
        v.append((radius * math.cos(a), radius * math.sin(a)))
        a = math.pi / npr * (2 * i + 1)
        v.append((inner * math.cos(a), inner * math.sin(a)))
    return rounded_polygon(v, [rounding] * (2 * npr), (0.0, 0.0))


def _rot(deg):  # Compose Matrix().rotateZ(deg) applied with Matrix.map
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return lambda x, y: (c * x - s * y, s * x + c * y)


def _custom(pnr, reps, center=(0.5, 0.5), mirroring=False):
    """MaterialShapes.customPolygon / doRepeat."""
    cx, cy = center
    pts: list[tuple[float, float, tuple[float, float]]] = []
    if mirroring:
        ang = [math.degrees(math.atan2(p[1] - cy, p[0] - cx)) for p, _ in pnr]
        dist = [_dist(p[0] - cx, p[1] - cy) for p, _ in pnr]
        actual = reps * 2
        sec = 360.0 / actual
        for it in range(actual):
            for index in range(len(pnr)):
                i = index if it % 2 == 0 else len(pnr) - 1 - index
                if i > 0 or it % 2 == 0:
                    a = math.radians(sec * it + (ang[i] if it % 2 == 0 else sec - ang[i] + 2 * ang[0]))
                    pts.append((math.cos(a) * dist[i] + cx, math.sin(a) * dist[i] + cy, pnr[i][1]))
    else:
        npnt = len(pnr)
        for it in range(npnt * reps):
            (px, py), r = pnr[it % npnt]
            a = math.radians((it // npnt) * 360.0 / reps)
            ox, oy = px - cx, py - cy
            pts.append((ox * math.cos(a) - oy * math.sin(a) + cx, ox * math.sin(a) + oy * math.cos(a) + cy, r))
    return rounded_polygon([(x, y) for x, y, _ in pts], [r for _, _, r in pts], center)


_U = (0.0, 0.0)  # CornerRounding.Unrounded
_R15, _R20, _R30, _R50, _R100 = (0.15, 0), (0.2, 0), (0.3, 0), (0.5, 0), (1.0, 0)


def _m3(name: str) -> RoundedPolygon:
    """The MaterialShapes polygons ImageToolbox uses (androidx-main ``MaterialShapes.kt``)."""
    P = _custom
    defs: dict[str, Callable[[], RoundedPolygon]] = {
        "slanted": lambda: P([((0.926, 0.970), (0.189, 0.811)), ((-0.021, 0.967), (0.187, 0.057))], 2),
        "arch": lambda: _rp_ngon(4, per_vertex=[_R100, _R100, _R20, _R20]).transformed(_rot(-135)),
        "fan": lambda: P([((1.004, 1.000), (0.148, 0.417)), ((0.000, 1.000), (0.151, 0)),
                          ((0.000, -0.003), (0.148, 0)), ((0.978, 0.020), (0.803, 0))], 1),
        "arrow": lambda: P([((0.500, 0.892), (0.313, 0)), ((-0.216, 1.050), (0.207, 0)),
                            ((0.499, -0.160), (0.215, 1.000)), ((1.225, 1.060), (0.211, 0))], 1),
        "semi_circle": lambda: _rp_rectangle(1.6, 1.0, per_vertex=[_R20, _R20, _R100, _R100]),
        "oval": lambda: _rp_circle().transformed(lambda x, y: (x, 0.64 * y)).transformed(_rot(-45)),
        "triangle": lambda: _rp_ngon(3, _R20).transformed(_rot(-90)),
        "diamond": lambda: P([((0.500, 1.096), (0.151, 0.524)), ((0.040, 0.500), (0.159, 0))], 2),
        "clam_shell": lambda: P([((0.171, 0.841), (0.159, 0)), ((-0.020, 0.500), (0.140, 0)),
                                 ((0.170, 0.159), (0.159, 0))], 2),
        "gem": lambda: P([((0.499, 1.023), (0.241, 0.778)), ((-0.005, 0.792), (0.208, 0)),
                          ((0.073, 0.258), (0.228, 0)), ((0.433, -0.000), (0.491, 0))], 1, mirroring=True),
        "sunny": lambda: _rp_star(8, inner=0.8, rounding=_R15),
        "very_sunny": lambda: P([((0.500, 1.080), (0.085, 0)), ((0.358, 0.843), (0.085, 0))], 8),
        "cookie_4": lambda: P([((1.237, 1.236), (0.258, 0)), ((0.500, 0.918), (0.233, 0))], 4),
        "cookie_6": lambda: P([((0.723, 0.884), (0.394, 0)), ((0.500, 1.099), (0.398, 0))], 6),
        "cookie_9": lambda: _rp_star(9, inner=0.8, rounding=_R50).transformed(_rot(-90)),
        "cookie_12": lambda: _rp_star(12, inner=0.8, rounding=_R50).transformed(_rot(-90)),
        "ghostish": lambda: P([((0.500, 0.0), (1.0, 0)), ((1.0, 0.0), (1.0, 0)), ((1.0, 1.140), (0.254, 0.106)),
                               ((0.575, 0.906), (0.253, 0))], 1, mirroring=True),
        "clover_4": lambda: P([((0.500, 0.074), _U), ((0.725, -0.099), (0.476, 0))], 4, mirroring=True),
        "clover_8": lambda: P([((0.500, 0.036), _U), ((0.758, -0.101), (0.209, 0))], 8),
        "burst": lambda: P([((0.500, -0.006), (0.006, 0)), ((0.592, 0.158), (0.006, 0))], 12),
        "soft_burst": lambda: P([((0.193, 0.277), (0.053, 0)), ((0.176, 0.055), (0.053, 0))], 10),
        "boom": lambda: P([((0.457, 0.296), (0.007, 0)), ((0.500, -0.051), (0.007, 0))], 15),
        "soft_boom": lambda: P([((0.733, 0.454), _U), ((0.839, 0.437), (0.532, 0)),
                                ((0.949, 0.449), (0.439, 1.000)), ((0.998, 0.478), (0.174, 0))], 16,
                               mirroring=True),
        "flower": lambda: P([((0.370, 0.187), _U), ((0.416, 0.049), (0.381, 0)), ((0.479, 0.001), (0.095, 0))],
                            8, mirroring=True),
        "puffy": lambda: P([((0.500, 0.053), _U), ((0.545, -0.040), (0.405, 0)), ((0.670, -0.035), (0.426, 0)),
                            ((0.717, 0.066), (0.574, 0)), ((0.722, 0.128), _U), ((0.777, 0.002), (0.360, 0)),
                            ((0.914, 0.149), (0.660, 0)), ((0.926, 0.289), (0.660, 0)), ((0.881, 0.346), _U),
                            ((0.940, 0.344), (0.126, 0)), ((1.003, 0.437), (0.255, 0))], 2,
                           mirroring=True).transformed(lambda x, y: (x, 0.742 * y)),
        "puffy_diamond": lambda: P([((0.870, 0.130), (0.146, 0)), ((0.818, 0.357), _U),
                                    ((1.000, 0.332), (0.853, 0))], 4, mirroring=True),
        "pixel_circle": lambda: P([((0.500, 0.000), _U), ((0.704, 0.000), _U), ((0.704, 0.065), _U),
                                   ((0.843, 0.065), _U), ((0.843, 0.148), _U), ((0.926, 0.148), _U),
                                   ((0.926, 0.296), _U), ((1.000, 0.296), _U)], 2, mirroring=True),
        "pixel_triangle": lambda: P([((0.110, 0.500), _U), ((0.113, 0.000), _U), ((0.287, 0.000), _U),
                                     ((0.287, 0.087), _U), ((0.421, 0.087), _U), ((0.421, 0.170), _U),
                                     ((0.560, 0.170), _U), ((0.560, 0.265), _U), ((0.674, 0.265), _U),
                                     ((0.675, 0.344), _U), ((0.789, 0.344), _U), ((0.789, 0.439), _U),
                                     ((0.888, 0.439), _U)], 1, mirroring=True),
        "bun": lambda: P([((0.796, 0.500), _U), ((0.853, 0.518), (1.0, 0)), ((0.992, 0.631), (1.0, 0)),
                          ((0.968, 1.000), (1.0, 0))], 2, mirroring=True),
        "heart": lambda: P([((0.500, 0.268), (0.016, 0)), ((0.792, -0.066), (0.958, 0)),
                            ((1.064, 0.276), (1.000, 0)), ((0.501, 0.946), (0.129, 0))], 1, mirroring=True),
    }
    return defs[name]().normalized()


M3_SHAPES = ["slanted", "arch", "fan", "arrow", "semi_circle", "oval", "triangle", "diamond", "clam_shell", "gem",
             "sunny", "very_sunny", "cookie_4", "cookie_6", "cookie_9", "cookie_12", "ghostish", "clover_4",
             "clover_8", "burst", "soft_burst", "boom", "soft_boom", "flower", "puffy", "puffy_diamond",
             "pixel_circle", "pixel_triangle", "bun", "heart"]

# =============================================================================
# ImageToolbox shape paths
# =============================================================================

# Kotlin ``Path().apply { moveTo/lineTo/cubicTo }`` shapes (core/resources/.../shapes/*.kt), scaled by
# size / (baseWidth, baseHeight); HeartShape uses fractions of the size (base 1 x 1).
# Generated from the Kotlin sources (calls converted 1:1 to SVG path commands).
_BASE_PATHS: dict[str, tuple[float, float, str]] = {
    "oval": (1000.0, 1000.0,
        "M1000 500C1000 776.1424 776.1424 1000 500 1000C223.8576 1000 0 776.1424 0 500"
        "C0 223.8576 223.8576 0 500 0C776.1424 0 1000 223.8576 1000 500Z"),
    "squircle": (1000.0, 1000.0,
        "M0 500C0 88.25 88.25 0 500 0C911.75 0 1000 88.25 1000 500C1000 911.75 911.75 1000 500 1000"
        "C88.25 1000 0 911.75 0 500Z"),
    "octagon": (1000.0, 1000.0,
        "M500 0L853.5534 146.4466L1000 500L853.5534 853.5534L500 1000L146.4466 853.5534L0 500"
        "L146.4466 146.4466L500 0Z"),
    "pentagon": (1224.1858, 1137.3882,
        "M521.133 28.588C575.7829 -9.5293 648.4028 -9.5293 703.0528 28.588L1156.1332 344.6029"
        "C1214.148 385.0671 1238.4572 458.9902 1215.7808 525.9892L1045.41 1029.3625"
        "C1023.5555 1093.9333 962.9716 1137.3882 894.8026 1137.3882L329.3832 1137.3882"
        "C261.2142 1137.3882 200.6302 1093.9333 178.7757 1029.3625L8.405 525.9893"
        "C-14.2714 458.9903 10.0377 385.0671 68.0526 344.6029L521.133 28.588Z"),
    "clover": (200.0, 200.0,
        "M12 100C12 76 0 77.6142 0 50C0 22.3858 22.3858 0 50 0C77.6142 0 76 12 100 12"
        "C124 12 122.3858 0 150 0C177.6142 0 200 22.3858 200 50C200 77.6142 188 76 188 100"
        "C188 124 200 122.3858 200 150C200 177.6142 177.6142 200 150 200C122.3858 200 124 188 100 188"
        "C76 188 77.6142 200 50 200C22.3858 200 0 177.6142 0 150C0 122.3858 12 124 12 100Z"),
    "kotlin": (1000.0, 1000.0,
        "M0 0L1000 0L473.5 500L1000 1000L0 1000L0 0Z"),
    "material_star": (865.0807, 865.0807,
        "M403.3913 8.7356C421.0787 -2.9119 444.002 -2.9119 461.6894 8.7356L518.743 46.3066"
        "C528.2839 52.5895 539.5995 55.6215 551.0036 54.9508L619.1989 50.9402"
        "C640.3404 49.6968 660.1926 61.1585 669.6865 80.0892L700.3109 141.1534"
        "C705.4321 151.365 713.7157 159.6486 723.9273 164.7699L784.9915 195.3942"
        "C803.9222 204.8881 815.3839 224.7403 814.1406 245.8818L810.1299 314.0771"
        "C809.4593 325.4812 812.4913 336.7969 818.7742 346.3378L856.3451 403.3913"
        "C867.9926 421.0787 867.9927 444.002 856.3452 461.6894L818.7742 518.743"
        "C812.4913 528.2839 809.4593 539.5995 810.1299 551.0036L814.1406 619.1989"
        "C815.3839 640.3404 803.9223 660.1926 784.9916 669.6865L723.9274 700.3109"
        "C713.7158 705.4321 705.4321 713.7157 700.3109 723.9273L669.6866 784.9915"
        "C660.1926 803.9222 640.3404 815.3839 619.1989 814.1406L551.0036 810.1299"
        "C539.5995 809.4593 528.2839 812.4913 518.743 818.7742L461.6894 856.3451"
        "C444.0021 867.9926 421.0787 867.9927 403.3914 856.3452L346.3378 818.7742"
        "C336.7969 812.4913 325.4812 809.4593 314.0771 810.1299L245.8818 814.1406"
        "C224.7404 815.3839 204.8882 803.9223 195.3942 784.9916L164.7699 723.9274"
        "C159.6486 713.7158 151.365 705.4321 141.1534 700.3109L80.0892 669.6866"
        "C61.1585 660.1926 49.6968 640.3404 50.9402 619.199L54.9508 551.0036"
        "C55.6215 539.5995 52.5895 528.2839 46.3066 518.743L8.7356 461.6894"
        "C-2.9119 444.0021 -2.9119 421.0787 8.7356 403.3914L46.3066 346.3378"
        "C52.5895 336.7969 55.6215 325.4813 54.9508 314.0771L50.9402 245.8818"
        "C49.6968 224.7404 61.1585 204.8882 80.0892 195.3942L141.1534 164.7699"
        "C151.365 159.6486 159.6486 151.365 164.7699 141.1534L195.3942 80.0892"
        "C204.8882 61.1585 224.7403 49.6968 245.8818 50.9402L314.0771 54.9508"
        "C325.4813 55.6215 336.7969 52.5895 346.3378 46.3066L403.3913 8.7356Z"),
    "small_material_star": (2695.79, 2680.65,
        "M2537.8 975.157C2623.02 1101.61 2665.64 1164.84 2683.13 1233.15"
        "C2699.69 1297.82 2700.13 1365.57 2684.42 1430.45C2667.81 1498.98 2626.03 1562.76 2542.46 1690.32"
        "C2517.24 1728.81 2504.64 1748.05 2494.33 1768.43C2484.53 1787.8 2476.3 1807.93 2469.7 1828.61"
        "C2462.77 1850.37 2458.26 1872.96 2449.25 1918.15C2419.3 2068.37 2404.33 2143.48 2367.9 2204.19"
        "C2334.18 2260.38 2287.43 2307.62 2231.59 2341.92C2171.26 2378.98 2095.52 2394.9 1944.04 2426.74"
        "C1897.48 2436.53 1874.19 2441.42 1852.04 2448.8C1832.55 2455.29 1813.57 2463.25 1795.28 2472.6"
        "C1774.49 2483.22 1754.88 2496.27 1715.65 2522.35C1584.84 2609.34 1519.44 2652.83 1448.72 2669.7"
        "C1388.23 2684.12 1325.25 2684.53 1264.59 2670.89C1193.65 2654.95 1127.69 2612.31 995.759 2527.04"
        "C956.196 2501.47 936.414 2488.68 915.487 2478.33C897.073 2469.21 877.998 2461.51 858.423 2455.27"
        "C836.176 2448.18 812.831 2443.59 766.142 2434.41C614.258 2404.54 538.317 2389.61 477.51 2353.34"
        "C421.231 2319.77 373.865 2273.14 339.421 2217.4C302.205 2157.16 286.253 2082.26 254.349 1932.44"
        "C244.752 1887.38 239.953 1864.84 232.735 1843.18C225.871 1822.58 217.375 1802.56 207.326 1783.32"
        "C196.758 1763.08 183.9 1744 158.185 1705.84C72.9637 1579.38 30.3531 1516.16 12.8577 1447.85"
        "C-3.7056 1383.18 -4.1467 1315.43 11.5731 1250.55C28.1775 1182.01 69.9612 1118.24 153.529 990.681"
        "C178.745 952.192 191.353 932.947 201.657 912.571C211.454 893.196 219.689 873.069 226.284 852.384"
        "C233.219 830.629 237.724 808.034 246.734 762.845C276.684 612.629 291.659 537.521 328.088 476.81"
        "C361.804 420.62 408.558 373.377 464.395 339.08C524.724 302.023 600.465 286.102 751.947 254.26"
        "C798.513 244.472 821.796 239.578 843.949 232.199C863.44 225.707 882.414 217.752 900.708 208.401"
        "C921.498 197.775 941.112 184.732 980.339 158.647C1111.14 71.6626 1176.55 28.1706 1247.27 11.3026"
        "C1307.75 -3.122 1370.73 -3.5321 1431.4 10.1038C1502.34 26.0493 1568.3 68.686 1700.23 153.959"
        "C1739.79 179.532 1759.57 192.318 1780.5 202.673C1798.91 211.784 1817.99 219.492 1837.57 225.73"
        "C1859.81 232.819 1883.16 237.41 1929.85 246.591C2081.73 276.457 2157.67 291.391 2218.48 327.659"
        "C2274.76 361.227 2322.12 407.856 2356.57 463.603C2393.78 523.834 2409.74 598.741 2441.64 748.554"
        "C2451.24 793.622 2456.04 816.156 2463.25 837.819C2470.12 858.417 2478.61 878.434 2488.66 897.68"
        "C2499.23 917.921 2512.09 937 2537.8 975.157Z"),
    "heart_cubic": (1.0, 1.0,
        "M0.5 0.25C0.5 0.225 0.45833334 0.125 0.29166666 0.125"
        "C0.041666668 0.125 0.041666668 0.4 0.041666668 0.4"
        "C0.041666668 0.5833333 0.20833333 0.76666665 0.5 0.9166667"
        "C0.7916667 0.76666665 0.9583333 0.5833333 0.9583333 0.4"
        "C0.9583333 0.4 0.9583333 0.125 0.7083333 0.125C0.5833333 0.125 0.5 0.225 0.5 0.25Z"),
    "simple_heart": (24.0, 24.0,
        "M12 21.35l-1.45 -1.32C5.4 15.36 2 12.28 2 8.5C2 5.42 4.42 3 7.5 3c1.74 0 3.41 0.81 4.5 2.09"
        "C13.09 3.81 14.76 3 16.5 3C19.58 3 22 5.42 22 8.5c0 3.78 -3.4 6.86 -8.55 11.54L12 21.35Z"),
}

# ``PathShape(pathData)`` shapes: scaled by size / control-point-bounds size about the origin
_PATH_SHAPES: dict[str, str] = {
    "pill": ("M48 20.4342C48 9.14772 38.8281 0 27.5193 0C21.8632 0 16.743 2.28863 13.036 5.98382"
        "L5.99744 13.1154C2.29384 16.814 0 21.9225 0 27.5658C0 38.8523 9.16854 48 20.4807 48"
        "C26.1368 48 31.257 45.7114 34.964 42.0162L42.0026 34.8846C45.7096 31.186 48 26.0775 48 20.4342Z"),
    "bookmark": ("M10 0C4.47715 0 0 4.47715 0 10V44.3522C0 46.1924 0.832489 48.0327 4 47.9996"
        "C5.04582 47.9996 11.1957 46.5617 16.3492 45.3564C18.5872 44.833 20.6375 44.3535 22 44.0549"
        "C22.5203 43.9314 23 44.0549 23 44.0549C26.0208 44.6465 29.2917 45.4016 32.2846 46.0925"
        "C36.4967 47.0648 40.158 47.91 41.796 47.9935L42.0617 47.9925"
        "C45.2368 47.8668 46 46.0449 46 43.8424V10C46 4.47715 41.5228 0 36 0H10Z"),
    "burger": ("M0.369565 7.44186C0.369565 3.33212 3.67833 0 7.76087 0H43.6087C47.6912 0 51 3.33212 51 7.44186"
        "V8.55814C51 12.6679 47.6912 16 43.6087 16C47.6912 16 51 19.3321 51 23.4419V24.5581"
        "C51 28.6679 47.6912 32 43.6087 32H43.2391C47.3217 32 50.6304 35.3321 50.6304 39.4419V40.5581"
        "C50.6304 44.6679 47.3217 48 43.2391 48H7.3913C3.30876 48 0 44.6679 0 40.5581V39.4419"
        "C0 35.3321 3.30876 32 7.3913 32H7.76087C3.67833 32 0.369565 28.6679 0.369565 24.5581V23.4419"
        "C0.369565 19.3321 3.67833 16 7.76087 16C3.67833 16 0.369565 12.6679 0.369565 8.55814V7.44186Z"),
    "shuriken": ("M46.1183 46.1349"
        "C46.1183 46.1349 44.5771 44.149 46.1183 41.3152 55.7085 25.337 50.6851 6.5989 50.6851 6.5989"
        "L46.1183 9.8651"
        "C46.1183 9.8651 44.0233 11.2763 41.4317 9.8651 25.4025.2998 6.599 5.3129 6.599 5.3129"
        "L9.8756 9.8651"
        "C9.8756 9.8651 11.5539 11.5722 9.8756 14.6905 9.8185 14.7929 9.7729 14.8783 9.7215 14.9579.3483 30.8736 5.3146 49.4011 5.3146 49.4011"
        "L9.8813 46.1349"
        "C9.8813 46.1349 11.8621 44.6668 14.5622 46.1349 30.5971 55.7002 49.395 50.6871 49.395 50.6871"
        "L46.1183 46.1349Z"),
    "explosion": ("M24.3629 0.130582C24.0071 -0.0526879 23.5886 -0.0422153 23.2434 0.156763"
        "C21.2763 1.29304 14.6586 5.04745 13.1833 5.04745"
        "C11.9748 5.04745 6.27256 2.52356 2.91398 0.984095"
        "C1.97755 0.55472 0.988809 1.45012 1.31839 2.42407C2.57394 6.1523 4.68221 12.6505 4.30554 13.2684"
        "C3.84518 14.0224 0.873717 21.6936 0.0890009 23.7357"
        "C-0.0313222 24.0499 -0.0103964 24.3955 0.146547 24.6992"
        "C1.13529 26.579 4.72929 33.5695 4.44156 35.1299C4.20614 36.4023 2.04556 42.0941 0.790014 45.3406"
        "C0.434276 46.2622 1.3027 47.1785 2.23912 46.8801C5.82789 45.7386 12.3463 43.686 12.9217 43.686"
        "C13.6227 43.686 21.925 47.0791 23.9601 47.9116C24.2478 48.0321 24.5669 48.0268 24.8546 47.9116"
        "C26.9211 47.0476 35.4745 43.487 35.8302 43.487C36.1179 43.487 41.7417 45.6286 44.9381 46.8591"
        "C45.8379 47.2047 46.743 46.3774 46.4866 45.4506C45.5554 42.0994 43.9285 36.1143 43.9756 35.3969"
        "C44.0331 34.5225 47.1458 26.223 47.9358 24.1284C48.0508 23.8247 48.0352 23.4844 47.8887 23.1912"
        "C46.9784 21.3689 43.6512 14.5984 43.7088 12.8704C43.7558 11.5038 45.6653 5.93238 46.8738 2.53403"
        "C47.2243 1.54961 46.2094 0.633265 45.2625 1.08358"
        "C41.7731 2.75396 35.9348 5.51871 35.4221 5.51871"
        "C34.7891 5.51871 26.6072 1.29827 24.3629 0.130582Z"),
    "map": ("M25.0702 4.38833C28.2859 2.26692 31.4412 0.185367 35.9294 0.00690002"
        "C41.3166 -0.204493 48 4.51158 48 4.51158L47.914 43.5389"
        "C41.8695 39.5777 37.3361 38.7019 34.1848 38.9888C29.4365 39.4182 26.0316 41.6001 22.7395 43.7097"
        "C19.4728 45.803 16.3173 47.8251 12.0706 47.9931C6.68339 48.2045 0 43.4884 0 43.4884V4.80351"
        "C5.84797 8.59348 10.4858 9.24276 13.8152 9.01123C18.563 8.68117 21.847 6.51473 25.0702 4.38833Z"),
    "shield": ("M22.8468 0.551384C24.8376 2.58461 34.1326 7.39938 41.7392 9.94215"
        "C42.1061 10.0645 42.4253 10.2994 42.6515 10.6135C42.8778 10.9276 42.9997 11.305 43 11.6923"
        "C43 32.5957 39.9697 42.7864 22.0137 47.9286C21.6828 48.0238 21.3319 48.0238 21.0011 47.9286"
        "C3.14341 42.8184 1.89533e-09 32.2757 1.89533e-09 12.0123C-1.82005"
        "e-05 11.6104 0.131069 11.2195 0.373309 10.8992C0.615549 10.5788 0.955673 10.3465 1.34191 10.2375"
        "C8.26315 8.39719 14.7081 5.08702 20.2392 0.531692C20.5968 0.202432 21.0611 0.0136179 21.5467 0"
        "C21.789 0.00155783 22.0286 0.051074 22.2517 0.145701"
        "C22.4748 0.240328 22.6771 0.378198 22.8468 0.551384Z"),
    "egg": ("M37 29.0542C36.8496 35.2815 34.6566 40.4326 29.781 44.0763"
        "C24.0677 48.3452 17.7493 49.1448 11.1127 46.4063C6.52734 44.5131 3.32181 41.1353 1.35268 36.5987"
        "C0.595451 34.8535 0.132022 32.9832 0.0428338 31.0619"
        "C-0.236972 25.0372 0.852523 19.2731 3.35679 13.7856"
        "C5.28745 9.55367 7.91064 5.81135 11.3033 2.65723"
        "C13.3004 0.799282 15.7172 0.0455345 18.4891 0.00150717"
        "C22.8523 -0.0689365 25.743 2.33848 28.263 5.43624C30.0888 7.67987 31.7082 10.0732 32.997 12.6972"
        "C35.0484 16.871 36.346 21.2403 36.7639 25.8755C36.8636 26.9849 36.93 28.0962 37 29.056V29.0542Z"),
    "droplet": ("M24 0C10.7452 0 0 10.7452 0 24C0 24.2317 0.00328295 24.4626 0.00980488 24.6927"
        "C0.00329738 24.8032 0 24.9144 0 25.0264V42.3896C0 45.4881 2.51186 48 5.61039 48H23.2847"
        "C23.3623 48 23.4396 47.9984 23.5167 47.9952C23.6774 47.9984 23.8385 48 24 48"
        "C37.2548 48 48 37.2548 48 24C48 10.7452 37.2548 0 24 0Z"),
    "arrow": ("M0.00127574 36.4191C0.00127574 42.8145 5.14768 48 11.4949 48"
        "C13.2236 48 14.863 47.6147 16.3341 46.9262C16.3341 46.9262 16.343 46.9223 16.3583 46.9146"
        "C16.496 46.8491 16.6337 46.7823 16.7676 46.7117C18.1252 46.0476 22.3143 44.0656 24.2634 43.9205"
        "C26.6065 43.7458 30.7293 45.0136 30.7293 45.0136L34.1891 46.6885"
        "C34.4147 46.808 34.6455 46.9185 34.88 47.0225L34.8915 47.0289H34.8953"
        "C35.8655 47.4579 36.9057 47.7559 37.9944 47.9011H37.997L38.0033 47.9024"
        "C38.2914 47.9409 38.5834 47.9679 38.8778 47.9833H38.8855C39.0907 47.9949 39.2973 48 39.5063 48"
        "C45.8536 48 51 42.8145 51 36.4191V36.3703C51 34.1995 50.4059 32.1687 49.3746 30.4321"
        "L35.905 6.40702C34.0195 2.6088 30.1225 1.30422e-06 25.6198 1.30422e-06C21.1172 1.30422"
        "e-06 17.2214 2.6088 15.3347 6.40702L1.88672 30.0133"
        "C1.63175 30.4051 1.39846 30.8123 1.19067 31.2349L1.18557 31.2439C0.427059 32.7904 -3.11934"
        "e-07 34.5296 -3.11934e-07 36.3703V36.4191H0.00127574Z"),
}

# lib/cropper ``Paths`` used as CustomPathOutline: scaled to the bounds, then moved to (0, 0)
_CROP_PATHS: dict[str, str] = {
    "heart": ("M12 21.35l-1.45 -1.32C5.4 15.36 2 12.28 2 8.5C2 5.42 4.42 3 7.5 3c1.74 0 3.41 0.81 4.5 2.09"
        "C13.09 3.81 14.76 3 16.5 3C19.58 3 22 5.42 22 8.5c0 3.78 -3.4 6.86 -8.55 11.54L12 21.35Z"),
    "star": ("M12 17.27L18.18 21l-1.64 -7.03L22 9.24l-7.19 -0.61L12 2L9.19 8.63L2 9.24l5.46 4.73L5.82 21Z"),
}

# (id, title) in ImageToolbox crop-mask order, then icon-shape-only extras
SHAPE_TITLES: dict[str, str] = {
    "rect": "Rect", "rounded_corners": "Rounded Corners", "cut_corners": "Cut Corners", "oval": "Oval",
    "squircle": "Squircle", "octagon": "Octagon", "pentagon": "Rounded Pentagon", "clover": "Clover",
    "pill": "Pill", "bookmark": "Bookmark", "burger": "Burger", "shuriken": "Shuriken", "explosion": "Explosion",
    "map": "Map", "material_star": "Material Star", "small_material_star": "Small Material Star",
    "shield": "Shield", "egg": "Egg", "droplet": "Droplet", "arrow": "Arrow", "kotlin": "Kotlin Logo",
    "heart": "Heart", "star": "Star",
    **{f"m3_{n}": "Material " + n.replace("_", " ").title() for n in M3_SHAPES},
    "image_mask": "Image Mask",
    "heart_cubic": "Heart (icon shape)", "simple_heart": "Simple Heart (icon shape)",
}


def shape_contours(shape: str, w: float, h: float, corner_radius: float = 20) -> list[Contour]:
    """Outline of ``shape`` for a ``w`` x ``h`` crop rectangle, in pixel coordinates (origin top-left)."""
    if shape in ("rect", "image_mask"):
        return parse_svg_path(f"M0 0H{w}V{h}H0Z")
    if shape in ("rounded_corners", "cut_corners"):
        c = max(0.0, min(50.0, corner_radius)) / 100 * min(w, h)  # Compose corner percent of the min side
        if c <= 0:
            return parse_svg_path(f"M0 0H{w}V{h}H0Z")
        if shape == "cut_corners":
            return parse_svg_path(f"M{c} 0H{w - c}L{w} {c}V{h - c}L{w - c} {h}H{c}L0 {h - c}V{c}Z")
        k = 0.5522847498 * c  # quarter-circle cubic
        return parse_svg_path(
            f"M{c} 0H{w - c}C{w - c + k} 0 {w} {c - k} {w} {c}V{h - c}C{w} {h - c + k} {w - c + k} {h} {w - c} {h}"
            f"H{c}C{c - k} {h} 0 {h - c + k} 0 {h - c}V{c}C0 {c - k} {c - k} 0 {c} 0Z")
    if shape in _BASE_PATHS:
        bw, bh, d = _BASE_PATHS[shape]
        return transform(parse_svg_path(d), w / bw, h / bh)
    if shape in _PATH_SHAPES:
        cs = parse_svg_path(_PATH_SHAPES[shape])
        l, t, r, b = path_bounds(cs)
        return transform(cs, w / (r - l), h / (b - t))
    if shape in _CROP_PATHS:
        cs = parse_svg_path(_CROP_PATHS[shape])
        l, t, r, b = path_bounds(cs)
        sx, sy = w / (r - l), h / (b - t)
        return transform(cs, sx, sy, -l * sx, -t * sy)
    if shape.startswith("m3_") and shape[3:] in M3_SHAPES:
        return transform(_m3_cached(shape[3:]).contours(), w, h)
    raise invalid(f"Unknown shape {shape!r}", valid=list(SHAPE_TITLES))


_M3_CACHE: dict[str, RoundedPolygon] = {}


def _m3_cached(name: str) -> RoundedPolygon:
    if name not in _M3_CACHE:
        _M3_CACHE[name] = _m3(name)
    return _M3_CACHE[name]


def _rotate_fit(polys: list[np.ndarray], w: int, h: int, rotation: float) -> list[np.ndarray]:
    """Rotate about the canvas centre; shrink uniformly about the centre if the result leaves the canvas."""
    if not rotation % 360:
        return polys
    a = math.radians(rotation)
    c, s = math.cos(a), math.sin(a)
    cx, cy = w / 2, h / 2
    out = []
    for p in polys:
        dx, dy = p[:, 0] - cx, p[:, 1] - cy
        out.append(np.stack([c * dx - s * dy, s * dx + c * dy], 1))
    ex = max(float(np.abs(p[:, 0]).max()) for p in out)
    ey = max(float(np.abs(p[:, 1]).max()) for p in out)
    k = min(1.0, cx / ex if ex > 0 else 1.0, cy / ey if ey > 0 else 1.0)
    return [p * k + (cx, cy) for p in out]


def render_shape(shape: str, h: int, w: int, rotation: float = 0.0, corner_radius: float = 20,
                 mask: np.ndarray | None = None) -> np.ndarray:
    """Coverage mask (float32 0..1) of ``shape`` stretched over an ``h`` x ``w`` canvas."""
    if shape == "image_mask":
        if mask is None:
            return np.ones((h, w), F32)
        m = cv2.resize(np.asarray(mask, F32), (w, h), interpolation=cv2.INTER_LINEAR)  # Bitmap.scale(filter=true)
        if rotation % 360:
            M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), -rotation, 1.0)
            m = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        return np.clip(m, 0, 1).astype(F32)
    polys = flatten(shape_contours(shape, float(w), float(h), corner_radius))
    return fill_polylines(_rotate_fit(polys, w, h, rotation), h, w)


def _make_render(shape: str) -> Callable[..., np.ndarray]:
    def render(h: int, w: int, **params) -> np.ndarray:
        return render_shape(shape, h, w, **params)

    render.__name__ = f"render_{shape}"
    render.__doc__ = f"{SHAPE_TITLES[shape]}: render(h, w, rotation=0, corner_radius=20, mask=None) -> mask"
    return render


SHAPES: dict[str, Callable[..., np.ndarray]] = {k: _make_render(k) for k in SHAPE_TITLES}

# Crop aspect-ratio presets (core/domain DomainAspectRatio.defaultList; "custom" -> use geometry.crop_aspect)
ASPECT_RATIOS: dict[str, float | None] = {
    "free": None, "original": None, "1:1": 1.0, "10:16": 10 / 16, "9:16": 9 / 16, "9:18.5": 9 / 18.5,
    "9:20": 9 / 20, "9:21": 9 / 21, "1:1.91": 1 / 1.91, "2:3": 2 / 3, "1:2": 1 / 2, "5:3": 5 / 3, "5:4": 5 / 4,
    "4:3": 4 / 3, "21:9": 21 / 9, "20:9": 20 / 9, "18.5:9": 18.5 / 9, "16:9": 16 / 9, "16:10": 16 / 10,
    "1.91:1": 1.91, "3:2": 3 / 2, "3:4": 3 / 4, "4:5": 4 / 5, "3:5": 3 / 5, "2:1": 2.0,
}


@op("geometry.crop_to_shape", "Crop to shape (ImageToolbox shapes)", params=[
    Choice("shape", "rounded_corners", choices=list(SHAPE_TITLES), labels=SHAPE_TITLES,
           description="ImageToolbox crop mask (23 shapes + 30 Material 3 shapes + image mask)"),
    Int("corner_radius", 20, min=0, max=50, title="Radius",
        description="Corner size in % of the shorter side (rounded_corners / cut_corners)"),
    Choice("aspect_ratio", "free", choices=list(ASPECT_RATIOS),
           description="Centre-crop to this aspect ratio first (ImageToolbox crop frame presets); "
                       "free/original keep the whole image"),
    Float("rotation", 0.0, min=-180, max=180, unit="deg",
          description="Rotate the shape about the centre (shrunk to stay inside the frame). LazyRoom extension"),
    File("mask_image", "", file_kind="image",
         description="shape=image_mask: mask image stretched over the crop; its alpha channel is the mask "
                     "(images without alpha use their luminance)"),
], tags=["shape crop", "crop mask", "squircle", "heart", "star", "material shapes", "cookie", "clover",
         "avatar", "mask", "rounded corners"], geometry=True, alpha="full")
def crop_to_shape(img, ctx, shape, corner_radius, aspect_ratio, rotation, mask_image):
    """Cut the image to one of ImageToolbox's crop shapes; outside becomes transparent (anti-aliased edge)."""
    h, w = img.shape[:2]
    r = ASPECT_RATIOS[aspect_ratio]
    if r is not None:
        cw, ch = (w, w / r) if w / h < r else (h * r, h)
        cw, ch = max(1, int(round(cw))), max(1, int(round(ch)))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        img = img[y0:y0 + ch, x0:x0 + cw]
        h, w = ch, cw
    mask = None
    if shape == "image_mask":
        if not mask_image:
            raise invalid("shape='image_mask' needs mask_image", param="mask_image")
        from ..core.image import gray
        from ..core.io import load_image

        mi, _ = load_image(ctx.resolve_file(mask_image, "mask image"))
        mask = mi[..., 3] if mi.shape[2] == 4 else gray(mi)
    m = render_shape(shape, h, w, rotation=rotation, corner_radius=corner_radius, mask=mask)
    out = ensure_rgba(img).copy()
    out[..., 3] *= m
    return out


# =============================================================================
# Markup line styles (feature/draw DrawLineStyle -> AndroidImageDrawApplier.asPathEffect)
# =============================================================================

LINE_STYLES = {"none": "Default", "dashed": "Dashed", "dot_dashed": "Dot Dashed", "zigzag": "Zigzag",
               "stamped": "Stamped"}

# Stamp shapes = IconShape.entriesNoRandom (core/settings/.../IconShape.kt), in order: (shape, corner %)
STAMP_SHAPES: dict[str, tuple[str, float]] = {
    "squircle": ("squircle", 0), "rounded_15": ("rounded_corners", 15), "rounded_25": ("rounded_corners", 25),
    "rounded_35": ("rounded_corners", 35), "rounded_45": ("rounded_corners", 45), "cut_25": ("cut_corners", 25),
    "cut_35": ("cut_corners", 35), "cut_50": ("cut_corners", 50),
    **{k: (k, 0) for k in ("clover", "material_star", "small_material_star", "bookmark", "pill", "burger", "oval",
                           "shield", "egg", "droplet", "arrow", "pentagon", "octagon", "shuriken", "explosion", "map",
                           "heart_cubic", "simple_heart")},
    **{f"m3_{k}": (f"m3_{k}", 0) for k in ("slanted", "arch", "semi_circle", "oval", "diamond", "clam_shell", "gem",
                                           "sunny", "very_sunny", "cookie_4", "cookie_6", "cookie_9", "cookie_12",
                                           "ghostish", "clover_4", "clover_8", "burst", "soft_burst", "boom",
                                           "soft_boom", "flower", "puffy", "puffy_diamond", "pixel_circle", "bun")},
}


class _Measure:
    """``SkPathMeasure`` for a polyline: position / tangent at a distance (pinned to [0, length])."""

    def __init__(self, pts: np.ndarray):
        keep = np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]
        self.p = pts[keep]
        seg = np.diff(self.p, axis=0)
        self.len = np.hypot(seg[:, 0], seg[:, 1])
        self.dir = seg / np.maximum(self.len, 1e-12)[:, None]
        self.cum = np.r_[0.0, np.cumsum(self.len)]
        self.length = float(self.cum[-1])

    def pos_tan(self, d: np.ndarray):
        d = np.clip(d, 0.0, self.length)
        i = np.clip(np.searchsorted(self.cum, d, side="right") - 1, 0, len(self.len) - 1)
        pos = self.p[i] + (d - self.cum[i])[..., None] * self.dir[i]
        return pos, self.dir[i]

    def segment(self, d0: float, d1: float) -> np.ndarray:
        """Sub-polyline between two distances (``SkPathMeasure::getSegment``)."""
        inner = self.p[1:-1][(self.cum[1:-1] > d0) & (self.cum[1:-1] < d1)]
        a, _ = self.pos_tan(np.array([d0, d1]))
        return np.vstack([a[:1], inner, a[1:]])


def _dash_pieces(meas: _Measure, intervals: list[float]) -> list[np.ndarray]:
    """``SkDashPathEffect`` with phase 0: the 'on' pieces of the path."""
    pieces, d, k, total = [], 0.0, 0, sum(intervals)
    if total <= 0:
        return [meas.p]
    while d < meas.length:
        on = intervals[k % len(intervals)]
        if k % 2 == 0:
            pieces.append(meas.segment(d, min(d + on, meas.length)))
        d += on
        k += 1
    return pieces


def _stamp(meas: _Measure, stamp: list[Contour], advance: float) -> list[np.ndarray]:
    """``SkPath1DPathEffect`` (kMorph, phase 0): bend copies of ``stamp`` along the path every ``advance``."""
    polys = []
    d = 0.0
    while d < meas.length and advance > 0 and len(polys) < 200_000:
        morphed = []
        for cub, ln in stamp:
            pts = cub.reshape(-1, 2)
            pos, tan = meas.pos_tan(d + pts[:, 0])
            y = pts[:, 1:2]
            q = pos + np.concatenate([-tan[:, 1:2] * y, tan[:, 0:1] * y], axis=1)
            morphed.append((q.reshape(-1, 8), ln))
        polys.extend(flatten(morphed))
        d += advance
    return polys


@njit(cache=True)
def _stroke_capsules(segs, r, h, w):
    """Union of round-capped segments (round caps + round joins), anti-aliased by distance."""
    out = np.zeros((h, w), np.float32)
    for k in range(segs.shape[0]):
        x0, y0, x1, y1 = segs[k, 0], segs[k, 1], segs[k, 2], segs[k, 3]
        xa = max(0, int(math.floor(min(x0, x1) - r - 1)))
        xb = min(w, int(math.ceil(max(x0, x1) + r + 1)))
        ya = max(0, int(math.floor(min(y0, y1) - r - 1)))
        yb = min(h, int(math.ceil(max(y0, y1) + r + 1)))
        dx, dy = x1 - x0, y1 - y0
        l2 = dx * dx + dy * dy
        for py in range(ya, yb):
            cy = py + 0.5
            for px in range(xa, xb):
                cx = px + 0.5
                t = 0.0
                if l2 > 0:
                    t = min(1.0, max(0.0, ((cx - x0) * dx + (cy - y0) * dy) / l2))
                ex, ey = x0 + t * dx - cx, y0 + t * dy - cy
                c = r - math.sqrt(ex * ex + ey * ey) + 0.5
                if c > out[py, px]:
                    out[py, px] = min(c, 1.0)
    return out


def _stroke(pieces: list[np.ndarray], width: float, h: int, w: int) -> np.ndarray:
    segs = [np.concatenate([p[:-1], p[1:]], axis=1) if len(p) > 1 else np.concatenate([p, p], axis=1)
            for p in pieces if len(p)]
    if not segs:
        return np.zeros((h, w), F32)
    return _stroke_capsules(np.ascontiguousarray(np.concatenate(segs), np.float64), width / 2, h, w)


def line_style_mask(points: np.ndarray, h: int, w: int, line_style: str = "none", line_width: float = 20,
                    dash_size: float = 10, gap_size: float = 20, zigzag_ratio: float = 4,
                    stamp_shape: str = "material_star", stamp_spacing: float = 20) -> np.ndarray:
    """Coverage of a polyline (pixel coordinates) drawn with an ImageToolbox line style; sizes in Pt."""
    pt = min(w, h) / 500  # Pt.toPx: min(width, height) * value / 500
    sw = line_width * pt
    meas = _Measure(np.asarray(points, np.float64))
    if meas.length <= 0 or len(meas.len) == 0:
        return _stroke([meas.p[:1]], sw, h, w) if line_style in ("none", "dashed", "dot_dashed") else \
            np.zeros((h, w), F32)
    if line_style == "none":
        return _stroke([meas.p], sw, h, w)
    if line_style == "dashed":
        return _stroke(_dash_pieces(meas, [dash_size * pt, gap_size * pt + sw]), sw, h, w)
    if line_style == "dot_dashed":
        return _stroke(_dash_pieces(meas, [sw * 4, sw * 2, sw / 4, sw * 2]), sw, h, w)
    if line_style == "zigzag":
        z = sw / zigzag_ratio
        o = sw / 4
        d = (f"M{-o} {-o}L{sw / 2 - o} {sw / 2 - o}L{sw - o} {-o}L{sw - o} {z - o}"
             f"L{sw / 2 - o} {sw / 2 + z - o}L{-o} {z - o}Z")
        return fill_polylines(_stamp(meas, parse_svg_path(d), sw), h, w)
    if line_style == "stamped":
        shape, corner = STAMP_SHAPES[stamp_shape]
        return fill_polylines(_stamp(meas, shape_contours(shape, sw, sw, corner), stamp_spacing * pt + sw), h, w)
    raise invalid(f"Unknown line style {line_style!r}", valid=list(LINE_STYLES))


@op("draw.styled_line", "Styled line (ImageToolbox line styles)", params=[
    Points("points", [[0.1, 0.5], [0.9, 0.5]], min_items=2, max_items=5000, min=-1, max=2,
           description="Polyline vertices, normalized image coordinates"),
    Bool("closed", False, description="Close the polyline back to its first point"),
    Choice("line_style", "none", choices=list(LINE_STYLES), labels=LINE_STYLES, title="Line Style"),
    Float("line_width", 20, min=1, max=100, step=0.1, unit="pt", title="Line Width",
          description="ImageToolbox Pt: 1 pt = 1/500 of the shorter image side"),
    Color("color", "#000000", title="Color"),
    Float("dash_size", 10, min=0, max=100, step=0.1, unit="pt", title="Dash Size", description="dashed"),
    Float("gap_size", 20, min=0, max=100, step=0.1, unit="pt", title="Gap Size",
          description="dashed (the line width is added, compensating the round caps)"),
    Float("zigzag_ratio", 4, min=0.5, max=20, step=0.01, title="Zigzag ratio",
          description="zigzag: line width / zigzag stroke thickness"),
    Choice("stamp_shape", "material_star", choices=list(STAMP_SHAPES),
           description="stamped: shape drawn along the path (ImageToolbox icon shapes)"),
    Float("stamp_spacing", 20, min=0, max=100, step=0.1, unit="pt", title="Spacing", description="stamped"),
], alpha="full", tags=["line", "dashed", "dotted", "dot dashed", "zigzag", "stamped", "markup", "annotate",
                       "path", "stroke"])
def styled_line(img, ctx, points, closed, line_style, line_width, color, dash_size, gap_size, zigzag_ratio,
                stamp_shape, stamp_spacing):
    """Draw a polyline with ImageToolbox's draw line styles: default (round caps), dashed, dot-dashed,
    zigzag or stamped shapes, bent along the path like Skia's path effects."""
    h, w = img.shape[:2]
    pts = np.array(points, np.float64) * (w, h)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    m = line_style_mask(pts, h, w, line_style, line_width, dash_size, gap_size, zigzag_ratio, stamp_shape,
                        stamp_spacing) * float(color[3])
    c = np.array(color[:3], F32)
    if img.shape[2] == 3:
        return img * (1 - m[..., None]) + c * m[..., None]
    a = img[..., 3:4]
    out_a = m[..., None] + a * (1 - m[..., None])
    rgb = (c * m[..., None] + img[..., :3] * a * (1 - m[..., None])) / np.maximum(out_a, 1e-6)
    return np.concatenate([rgb, out_a], axis=2)
