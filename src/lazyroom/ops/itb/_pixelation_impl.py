"""Private helpers for the ImageToolbox "Pixelation" group port.

Sources (all Apache-2.0 unless noted):
* ImageToolbox ``feature/filters/.../data/utils/pixelation/**`` (PixelationTool, PixelationCommands)
* ImageToolbox ``lib/ascii`` (ASCIIConverter)
* aire (MIT, awxkee): ``cpp/effect/CrystallizeEffect.cpp`` + ``jc_voronoi.h``, ``aire-filters/src/equalize_hist.rs``
* Trickle (Apache-2.0): ``polkaDotImpl`` / ``drawColorBehindImpl``
* ImageToolboxLibs jhlabs (Rust port of JH Labs filters): Smear, Crystallize (Cellular), Weave

Inherently sequential loops (Android canvas painting order, java.util.Random
sequences, Voronoi cell rasterisation) are ``numba.njit(cache=True)``.
"""
from __future__ import annotations

import functools
import math
import shutil
import subprocess
import sys

import cv2
import numpy as np
from numba import njit, prange

from ...core.errors import ErrorCode, LazyRoomError
from ._common import run_gmic

F32 = np.float32

# ---------------------------------------------------------------------------
# Alpha helpers
# ---------------------------------------------------------------------------


def to_rgba(img: np.ndarray) -> np.ndarray:
    """(H, W, 3|4) -> contiguous float32 (H, W, 4) straight-alpha RGBA."""
    if img.shape[2] == 4:
        return np.ascontiguousarray(img, dtype=F32)
    out = np.empty(img.shape[:2] + (4,), F32)
    out[..., :3] = img
    out[..., 3] = 1.0
    return out


def finish_rgba(out: np.ndarray, input_had_alpha: bool, min_alpha: float | None = None) -> np.ndarray:
    """Drop a fully opaque alpha channel when the input had none (ImageToolbox bitmaps are always ARGB)."""
    if out.shape[2] == 4 and not input_had_alpha:
        m = float(out[..., 3].min()) if min_alpha is None else min_alpha
        if m >= 1.0 - 1e-6:
            return out[..., :3]
    return out


@njit(cache=True, parallel=True)
def _prem_finish(prem, bg, out, row_min):
    """Optional colour behind (Trickle drawColorBehind) + unpremultiply into straight RGBA."""
    H = prem.shape[0]
    W = prem.shape[1]
    ba = bg[3]
    for y in prange(H):
        m = 1.0
        for x in range(W):
            r = prem[y, x, 0]
            g = prem[y, x, 1]
            b = prem[y, x, 2]
            a = prem[y, x, 3]
            if ba > 0.0:
                k = ba * (1.0 - a)
                r += bg[0] * k
                g += bg[1] * k
                b += bg[2] * k
                a += k
            if a > 1e-8:
                inv = 1.0 / a
                out[y, x, 0] = r * inv
                out[y, x, 1] = g * inv
                out[y, x, 2] = b * inv
            else:
                out[y, x, 0] = 0.0
                out[y, x, 1] = 0.0
                out[y, x, 2] = 0.0
            out[y, x, 3] = a
            m = min(m, a)
        row_min[y] = m


def prem_to_image(prem: np.ndarray, input_had_alpha: bool, behind=(0.0, 0.0, 0.0, 0.0)) -> np.ndarray:
    out = np.empty(prem.shape, F32)
    row_min = np.empty(prem.shape[0], F32)
    _prem_finish(prem, np.asarray(behind, F32), out, row_min)
    return finish_rgba(out, input_had_alpha, float(row_min.min()))


# ---------------------------------------------------------------------------
# ImageToolbox PixelationTool (Android Canvas, ANTI_ALIAS paint)
# ---------------------------------------------------------------------------

SQUARE, CIRCLE, DIAMOND = 0, 1, 2


def _layer(shape, res, size=None, alpha=1.0, off=0.0):
    # PixelationLayer: size defaults to the resolution, offset is applied to x and y.
    return (shape, float(res), float(res if size is None else size), float(alpha), float(off))


def pixelation_layers(kind: str, v: float) -> list[tuple]:
    """``PixelationCommands`` layer stacks (shape, resolution, size, alpha, offset)."""
    S, C, D = SQUARE, CIRCLE, DIAMOND
    L = _layer
    table = {
        "enhanced_diamond": [L(S, v), L(D, v, alpha=0.5, off=v / 4), L(D, v, alpha=0.5, off=v),
                             L(C, v / 3, size=v / 6, off=v / 12)],
        "circle": [L(C, v), L(C, v, size=v / 3, off=v / 2)],
        "diamond": [L(D, v, size=v + 1), L(D, v, off=v / 2), L(S, v, alpha=0.6)],
        "enhanced_circle": [L(S, v), L(C, v, off=v / 2), L(C, v, size=v / 1.2, off=v / 2.5),
                            L(C, v, size=v / 1.8, off=v / 3), L(C, v, size=v / 2.7, off=v / 4)],
        "enhanced_square": [L(S, v), L(D, v / 4, size=v / 6), L(D, v / 4, size=v / 6, off=v / 8)],
        "square": [L(S, v - 4, size=v)],
        "stroke": [L(C, v, size=v / 5, off=v / 4), L(C, v, size=v / 4, off=v / 2),
                   L(C, v, size=v / 3, off=v / 1.3), L(C, v, size=v / 4, off=0.0)],
        "simple_weave": [L(S, v), L(S, v, size=v / 2, off=v / 2, alpha=0.5),
                         L(D, v, size=v / 3, off=v / 4, alpha=0.4)],
        "staggered": [L(S, v), L(S, v, off=v / 2, size=v * 0.85), L(S, v * 2, alpha=0.35)],
        "cross": [L(S, v), L(D, v, size=v * 0.7), L(S, v / 2, size=v / 3, alpha=0.45)],
        "micro_macro": [L(S, v * 2), L(S, v / 2, size=v / 3)],
        "orbital": [L(S, v), L(C, v, size=v * 0.9, off=v / 3), L(C, v, size=v * 0.6, off=v / 1.5)],
        "vortex": [L(S, v), L(C, v, size=v * 0.85, off=v / 3), L(C, v * 1.4, size=v * 0.6, off=v / 1.1),
                   L(C, v / 2, size=v / 3)],
        "pulse_grid": [L(S, v * 1.5), L(C, v, size=v * 0.7), L(C, v / 1.8, size=v / 2.8, off=v / 2),
                       L(S, v / 2.5, alpha=0.35)],
        "nucleus": [L(C, v, size=v), L(C, v, size=v * 0.6, off=v / 2), L(C, v / 2, size=v / 3),
                    L(S, v * 2, alpha=0.25)],
        "radial_weave": [L(S, v), L(C, v, size=v * 0.9, off=v / 3), L(C, v, size=v * 0.6, off=v / 1.5),
                         L(C, v / 2, size=v / 3, off=v / 4), L(D, v / 3, size=v / 5, off=v / 6, alpha=0.35),
                         L(S, v / 4, alpha=0.25)],
    }
    return table[kind]


@njit(cache=True)
def _pixelate_layer(src, prem, shape, res, size, layer_alpha, offx, offy):
    """Draw one PixelationLayer onto the premultiplied canvas ``prem`` (row-major draw order).

    Shape centres are ``((i - 0.5) * res + offset)``; the colour is the source
    pixel at the clamped centre (alpha = int(layer_alpha * A8)).  Coverage is
    the anti-aliased area (exact for squares, distance-based for circles and
    the 45-degree rotated squares ("diamonds")).
    """
    H = src.shape[0]
    W = src.shape[1]
    cols = int(W / res + 1.0)
    rows = int(H / res + 1.0)
    half = size * 0.5
    sqrt2 = math.sqrt(2.0)
    cap_circle = math.pi * half * half
    cap_diamond = 2.0 * half * half
    for row in range(rows + 1):
        y = (row - 0.5) * res + offy
        iy = int(min(max(y, 0.0), H - 1.0))
        y0 = max(int(math.floor(y - half - 1.0)), 0)
        y1 = min(int(math.ceil(y + half + 1.0)), H)
        if y0 >= y1:
            continue
        for col in range(cols + 1):
            x = (col - 0.5) * res + offx
            ix = int(min(max(x, 0.0), W - 1.0))
            a8 = int(src[iy, ix, 3] * 255.0 + 0.5) if src.shape[2] == 4 else 255
            a = int(layer_alpha * a8) / 255.0
            if a <= 0.0:
                continue
            r = src[iy, ix, 0] * a
            g = src[iy, ix, 1] * a
            b = src[iy, ix, 2] * a
            x0 = max(int(math.floor(x - half - 1.0)), 0)
            x1 = min(int(math.ceil(x + half + 1.0)), W)
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    if shape == 0:
                        ox = min(xx + 1.0, x + half) - max(float(xx), x - half)
                        oy = min(yy + 1.0, y + half) - max(float(yy), y - half)
                        if ox <= 0.0 or oy <= 0.0:
                            continue
                        cov = min(ox, 1.0) * min(oy, 1.0)
                    elif shape == 1:
                        dx = xx + 0.5 - x
                        dy = yy + 0.5 - y
                        cov = half - math.sqrt(dx * dx + dy * dy) + 0.5
                        if cov <= 0.0:
                            continue
                        cov = min(cov, 1.0, cap_circle)
                    else:
                        dx = abs(xx + 0.5 - x)
                        dy = abs(yy + 0.5 - y)
                        cov = (half - (dx + dy)) / sqrt2 + 0.5
                        if cov <= 0.0:
                            continue
                        cov = min(cov, 1.0, cap_diamond)
                    sa = a * cov
                    inv = 1.0 - sa
                    prem[yy, xx, 0] = r * cov + prem[yy, xx, 0] * inv
                    prem[yy, xx, 1] = g * cov + prem[yy, xx, 1] * inv
                    prem[yy, xx, 2] = b * cov + prem[yy, xx, 2] * inv
                    prem[yy, xx, 3] = sa + prem[yy, xx, 3] * inv


def pixelate(img: np.ndarray, kind: str, value: float, scale: float = 1.0) -> np.ndarray:
    """ImageToolbox ``PixelationTool.pixelate`` -> premultiplied RGBA canvas (H, W, 4)."""
    src = np.ascontiguousarray(img, dtype=F32)
    prem = np.zeros(src.shape[:2] + (4,), F32)
    for shape, res, size, alpha, off in pixelation_layers(kind, float(value)):
        # Geometry is defined in full-resolution pixels; scale it for preview proxies.
        res_s = max(res * scale, 0.5)
        _pixelate_layer(src, prem, shape, res_s, max(size * scale, 0.0), alpha, off * scale, off * scale)
    return prem


# ---------------------------------------------------------------------------
# Trickle polkaDot
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _polka_dot(src, prem, radius, D):
    H = src.shape[0]
    W = src.shape[1]
    has_a = src.shape[2] == 4
    rr = radius * radius
    for y in prange(H):
        cy = (y // D) * D + radius
        dy = y - cy
        for x in range(W):
            cx = (x // D) * D + radius
            dx = x - cx
            if cy < H and cx < W and dx * dx + dy * dy <= rr:
                a = src[cy, cx, 3] if has_a else 1.0
                prem[y, x, 0] = src[cy, cx, 0] * a
                prem[y, x, 1] = src[cy, cx, 1] * a
                prem[y, x, 2] = src[cy, cx, 2] * a
                prem[y, x, 3] = a


def polka_dot(img: np.ndarray, radius: int, spacing: int) -> np.ndarray:
    """Dots of ``radius`` on a ``2*radius + spacing`` grid, colour of the dot centre; rest transparent.
    Returns a premultiplied RGBA canvas."""
    src = np.ascontiguousarray(img, dtype=F32)
    prem = np.zeros(src.shape[:2] + (4,), F32)
    _polka_dot(src, prem, int(radius), int(2 * radius + spacing))
    return prem


# ---------------------------------------------------------------------------
# aire crystallize (jc_voronoi cells rasterised as triangle bounding boxes)
# ---------------------------------------------------------------------------

def _voronoi_edges(pts: np.ndarray, rect: tuple[float, float, float, float]):
    """Voronoi cells of ``pts`` clipped to ``rect`` (as jc_voronoi with box clipping).

    Uses the mirror trick (reflect the sites across the rectangle edges) so
    every original cell is bounded by the rectangle.  Only sites near the
    border are mirrored; the result is validated and the band widened if any
    cell is still unbounded or leaks out of the rectangle.
    Returns (site index per edge, endpoint A (E, 2), endpoint B (E, 2)).
    """
    from scipy.spatial import Voronoi

    xmin, ymin, xmax, ymax = rect
    n = len(pts)
    span = max(xmax - xmin, ymax - ymin)
    band = min(span, 6.0 * math.sqrt((xmax - xmin) * (ymax - ymin) / max(n, 1)) + 2.0)
    tol = 1e-6 * span + 1e-6
    while True:
        full = band >= span
        mir = [pts]
        for axis, edge in ((0, xmin), (0, xmax), (1, ymin), (1, ymax)):
            sel = pts if full else pts[np.abs(pts[:, axis] - edge) <= band]
            if len(sel):
                m = sel.copy()
                m[:, axis] = 2.0 * edge - m[:, axis]
                mir.append(m)
        allp = np.concatenate(mir, axis=0)
        try:
            vor = Voronoi(allp)
        except Exception:  # degenerate (e.g. collinear) input: widen the mirror band
            if full:
                raise LazyRoomError(ErrorCode.OPERATION_FAILED, "Voronoi diagram construction failed")
            band = min(span, band * 2.0)
            continue
        rp = vor.ridge_points
        rv = np.asarray(vor.ridge_vertices, dtype=np.int64)
        ma = rp[:, 0] < n
        mb = rp[:, 1] < n
        sites = np.concatenate([rp[ma, 0], rp[mb, 1]])
        verts = np.concatenate([rv[ma], rv[mb]])
        ok = verts.size > 0 and (verts >= 0).all()
        if ok:
            va = vor.vertices[verts[:, 0]]
            vb = vor.vertices[verts[:, 1]]
            allv = np.concatenate([va, vb])
            ok = bool((allv[:, 0] >= xmin - tol).all() and (allv[:, 0] <= xmax + tol).all()
                      and (allv[:, 1] >= ymin - tol).all() and (allv[:, 1] <= ymax + tol).all())
            ok = ok and len(np.unique(sites)) == n
        if ok:
            va = np.clip(va, (xmin, ymin), (xmax, ymax))
            vb = np.clip(vb, (xmin, ymin), (xmax, ymax))
            return sites, va, vb
        if full:  # pragma: no cover - full mirroring always bounds every cell
            raise LazyRoomError(ErrorCode.OPERATION_FAILED, "Voronoi diagram construction failed")
        band = min(span, band * 2.0)


@njit(cache=True)
def _aire_line(out, color, x1, y1, x2, y2):
    """aire ``drawLine``: DDA with straight-alpha OVER blending."""
    H = out.shape[0]
    W = out.shape[1]
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        xi = 0.0
        yi = 0.0
    else:
        xi = dx / steps
        yi = dy / steps
    x = float(x1)
    y = float(y1)
    a1 = color[3]
    for _ in range(steps + 1):
        if x >= 0 and x < W and y >= 0 and y < H:
            px = int(x)
            py = int(y)
            for c in range(3):
                out[py, px, c] = color[c] * a1 + out[py, px, c] * (1.0 - a1)
            out[py, px, 3] = a1 + out[py, px, 3] * (1.0 - a1)
        x += xi
        y += yi


@njit(cache=True)
def _aire_crystallize_fill(src, out, site_xy, edge_site, ea, eb, stroke, has_stroke):
    H = src.shape[0]
    W = src.shape[1]
    cur = -1
    v0x = 0.0
    v0y = 0.0
    for e in range(edge_site.shape[0]):
        s = edge_site[e]
        if s != cur:
            cur = s
            v0x = site_xy[s, 0]
            v0y = site_xy[s, 1]
        v1x = ea[e, 0]
        v1y = ea[e, 1]
        v2x = eb[e, 0]
        v2y = eb[e, 1]
        minx = min(int(v0x), int(v1x), int(v2x))
        miny = min(int(v0y), int(v1y), int(v2y))
        maxx = max(int(v0x), int(v1x), int(v2x))
        maxy = max(int(v0y), int(v1y), int(v2y))
        minx = min(max(minx, 0), W - 1)
        miny = min(max(miny, 0), H - 1)
        maxx = min(max(maxx, 0), W - 1)
        maxy = min(max(maxy, 0), H - 1)
        # aire overwrites v0 with its ceil-clamped value (used by the following edges too)
        v0x = min(max(math.ceil(v0x), 0.0), W - 1.0)
        v0y = min(max(math.ceil(v0y), 0.0), H - 1.0)
        cy = int(v0y)
        cx = int(v0x)
        ca = src[cy, cx, 3] if src.shape[2] == 4 else 1.0
        for yy in range(miny, maxy + 1):
            for xx in range(minx, maxx + 1):
                for c in range(3):
                    out[yy, xx, c] = src[cy, cx, c]
                out[yy, xx, 3] = ca
        if has_stroke:
            _aire_line(out, stroke, minx, miny, minx, maxy)
            _aire_line(out, stroke, minx, maxy, maxx, maxy)
            _aire_line(out, stroke, maxx, maxy, maxx, miny)
            _aire_line(out, stroke, maxx, miny, minx, miny)


def aire_crystallize(img: np.ndarray, num_clusters: int, stroke, rng: np.random.Generator) -> np.ndarray:
    """aire ``crystallize``: random integer sites, jc_voronoi diagram remapped to the image,
    every (site, edge) triangle's bounding box filled with the site colour (+ optional stroke)."""
    src = np.ascontiguousarray(img, dtype=F32)
    H, W = src.shape[:2]
    if W < 2 or H < 2:
        return to_rgba(src)
    n = max(1, int(num_clusters))
    xs = rng.integers(1, W, size=n)  # uniform_int_distribution(1, width - 1)
    ys = rng.integers(1, H, size=n)
    key = np.unique(ys.astype(np.int64) * W + xs)  # jcv sorts sites by (y, x) and prunes duplicates
    pts = np.stack([key % W, key // W], axis=1).astype(np.float64)
    # jcv: bounding box of the sites, rounded and inflated by 10
    xmin, ymin = np.floor(pts.min(axis=0)) - 10.0
    xmax, ymax = np.ceil(pts.max(axis=0)) + 10.0
    sites, va, vb = _voronoi_edges(pts, (xmin, ymin, xmax, ymax))
    # graph edges are visited per site in order of the angle of their midpoint (jcv_calc_sort_metric)
    mid = 0.5 * (va + vb)
    dy = mid[:, 1] - pts[sites, 1]
    ang = np.arctan2(dy, mid[:, 0] - pts[sites, 0])
    ang = np.where(dy < 0, ang + 2.0 * np.pi, ang)
    order = np.lexsort((ang, sites))
    sites, va, vb = sites[order], va[order], vb[order]
    def remap(p):  # jcv remap(): (p - min) / (max - min) * dimensions, in float
        q = np.empty(p.shape, F32)
        q[:, 0] = (p[:, 0].astype(F32) - F32(xmin)) / F32(xmax - xmin) * F32(W)
        q[:, 1] = (p[:, 1].astype(F32) - F32(ymin)) / F32(ymax - ymin) * F32(H)
        return q

    out = np.zeros(src.shape[:2] + (4,), F32)
    stroke = np.asarray(stroke, F32)
    _aire_crystallize_fill(src, out, remap(pts), sites.astype(np.int64), remap(va), remap(vb), stroke,
                           bool(stroke[3] > 0.0))
    return out


# ---------------------------------------------------------------------------
# aire equalizeHistSquares (per-tile HSV value histogram equalisation, 101 bins)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _vbin(r, g, b):
    v = max(r, g, b)
    k = int(math.floor(v * 100.0 + 0.5))
    return min(max(k, 0), 100), v


@njit(cache=True)
def _eq_hist_lut(rgb, tw, th, ntx, nty):
    H = rgb.shape[0]
    W = rgb.shape[1]
    hist = np.zeros((ntx * nty, 101), np.int64)
    for y in range(H):
        ty = min(y // th, nty - 1)
        for x in range(W):
            tx = min(x // tw, ntx - 1)
            k, _ = _vbin(rgb[y, x, 0], rgb[y, x, 1], rgb[y, x, 2])
            hist[ty * ntx + tx, k] += 1
    lut = np.empty((ntx * nty, 101), np.float32)
    for t in range(ntx * nty):
        total = 0
        for k in range(101):
            total += hist[t, k]
        c0 = hist[t, 0]
        denom = total - c0
        cdf = 0
        for k in range(101):
            cdf += hist[t, k]
            if denom > 0:
                lut[t, k] = min(math.floor(100.0 * (cdf - c0) / denom + 0.5), 100.0) / 100.0
            else:  # Rust: NaN.min(100) == 100
                lut[t, k] = 1.0
    return lut


@njit(cache=True, parallel=True)
def _eq_hist_apply(rgb, out, lut, tw, th, ntx, nty):
    H = rgb.shape[0]
    W = rgb.shape[1]
    for y in prange(H):
        ty = min(y // th, nty - 1)
        for x in range(W):
            tx = min(x // tw, ntx - 1)
            k, v = _vbin(rgb[y, x, 0], rgb[y, x, 1], rgb[y, x, 2])
            nv = lut[ty * ntx + tx, k]
            if v > 0.0:  # HSV round trip with a new value: RGB scales by V'/V
                f = nv / v
                out[y, x, 0] = rgb[y, x, 0] * f
                out[y, x, 1] = rgb[y, x, 1] * f
                out[y, x, 2] = rgb[y, x, 2] * f
            else:
                out[y, x, 0] = nv
                out[y, x, 1] = nv
                out[y, x, 2] = nv


def equalize_hist_squares(rgb: np.ndarray, grid_w: int, grid_h: int) -> np.ndarray:
    """aire ``equalize_histogram_squares``: tiles of ``W // grid_w`` x ``H // grid_h`` (the last
    row/column extends to the border), HSV value equalised through a 101-bin CDF per tile."""
    rgb = np.ascontiguousarray(rgb, dtype=F32)
    H, W = rgb.shape[:2]
    tw = max(1, W // grid_w)
    th = max(1, H // grid_h)
    ntx = W // tw
    nty = H // th
    lut = _eq_hist_lut(rgb, tw, th, ntx, nty)
    out = np.empty(rgb.shape[:2] + (3,), F32)
    _eq_hist_apply(rgb, out, lut, tw, th, ntx, nty)
    return out


# ---------------------------------------------------------------------------
# java.util.Random (as ported in the jhlabs Rust library)
# ---------------------------------------------------------------------------

_JMUL = np.uint64(0x5DEECE66D)
_JADD = np.uint64(0xB)
_JMASK = np.uint64((1 << 48) - 1)


@njit(cache=True)
def _jr_init(seed):
    return (np.uint64(seed) ^ _JMUL) & _JMASK


@njit(cache=True)
def _jr_next_int(state):
    state = (state * _JMUL + _JADD) & _JMASK
    v = np.int64(state >> np.uint64(16))
    if v >= 2147483648:
        v -= 4294967296
    return state, v


@njit(cache=True)
def _jr_next_float(state):
    state = (state * _JMUL + _JADD) & _JMASK
    return state, np.int64(state >> np.uint64(24)) / 16777216.0


@njit(cache=True)
def _crem(a, b):
    """Rust/C ``%`` on signed integers (sign of the dividend)."""
    r = abs(a) % abs(b)
    return -r if a < 0 else r


# ---------------------------------------------------------------------------
# jhlabs Smear
# ---------------------------------------------------------------------------

@njit(cache=True)
def _smear_paint(img, x, y, color, mix):
    H = img.shape[0]
    W = img.shape[1]
    if x >= 0 and x < W and y >= 0 and y < H:
        for c in range(img.shape[2]):
            img[y, x, c] = img[y, x, c] * (1.0 - mix) + color[c] * mix


@njit(cache=True)
def _smear_crosses_lines(img, src, shape, distance, density, angle, mix):
    H = img.shape[0]
    W = img.shape[1]
    st = _jr_init(567)
    color = np.empty(img.shape[2], np.float32)
    if shape == 0:
        count = int(np.float32(2.0 * density * W * H / (distance + 1)))
        for _ in range(count):
            st, r1 = _jr_next_int(st)
            st, r2 = _jr_next_int(st)
            st, r3 = _jr_next_int(st)
            x = (r1 & 0x7FFFFFFF) % W
            y = (r2 & 0x7FFFFFFF) % H
            length = _crem(r3, distance) + 1
            for c in range(img.shape[2]):
                color[c] = src[y, x, c]
            for tx in range(x - length, x + length + 1):
                _smear_paint(img, tx, y, color, mix)
            for ty in range(y - length, y + length + 1):
                _smear_paint(img, x, ty, color, mix)
    else:
        count = int(np.float32(density * W * H))
        ca = math.cos(angle)
        sa = math.sin(angle)
        for _ in range(count):
            st, r1 = _jr_next_int(st)
            st, r2 = _jr_next_int(st)
            st, r3 = _jr_next_int(st)
            sx = (r1 & 0x7FFFFFFF) % W
            sy = (r2 & 0x7FFFFFFF) % H
            for c in range(img.shape[2]):
                color[c] = src[sy, sx, c]
            length = (r3 & 0x7FFFFFFF) % distance
            dx = int(length * ca)
            dy = int(length * sa)
            x0 = sx - dx
            y0 = sy - dy
            x1 = sx + dx
            y1 = sy + dy
            ddx = abs(x1 - x0)
            stx = 1 if x0 < x1 else -1
            ddy = -abs(y1 - y0)
            sty = 1 if y0 < y1 else -1
            err = ddx + ddy
            while True:
                _smear_paint(img, x0, y0, color, mix)
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= ddy:
                    err += ddy
                    x0 += stx
                if e2 <= ddx:
                    err += ddx
                    y0 += sty


@njit(cache=True)
def _dsu_find(parent, row, x):
    root = x
    while parent[row, root] != root:
        root = parent[row, root]
    while parent[row, x] != root:
        nxt = parent[row, x]
        parent[row, x] = root
        x = nxt
    return root


@njit(cache=True)
def _smear_blobs(img, src, shape, distance, density, mix):
    """Circles / squares.  Painting order is replayed backwards with per-pixel remaining weight
    ``(1-mix)^k``; pixels whose weight drops below 1/2048 are retired (union-find skip lists), so
    the result equals the forward painting to well below 8-bit precision while each pixel is only
    touched a bounded number of times."""
    H = img.shape[0]
    W = img.shape[1]
    C = img.shape[2]
    radius = distance + 1
    count = int(np.float32(2.0 * density * W * H / radius))
    if count <= 0:
        return
    xs = np.empty(count, np.int64)
    ys = np.empty(count, np.int64)
    st = _jr_init(567)
    for i in range(count):
        st, r1 = _jr_next_int(st)
        st, r2 = _jr_next_int(st)
        xs[i] = (r1 & 0x7FFFFFFF) % W
        ys[i] = (r2 & 0x7FFFFFFF) % H
    rem = np.ones((H, W), np.float32)
    acc = np.zeros((H, W, C), np.float32)
    parent = np.empty((H, W + 1), np.int32)
    for yy in range(H):
        for xx in range(W + 1):
            parent[yy, xx] = xx
    keep = 1.0 - mix
    eps = 1.0 / 2048.0
    for i in range(count - 1, -1, -1):
        sx = xs[i]
        sy = ys[i]
        for yy in range(max(sy - radius, 0), min(sy + radius, H - 1) + 1):
            dyy = yy - sy
            if shape == 3:
                half = radius
            else:
                rr = radius * radius - dyy * dyy
                if rr < 0:
                    continue
                half = int(math.sqrt(rr))
                while (half + 1) * (half + 1) <= rr:
                    half += 1
                while half * half > rr:
                    half -= 1
            xa = max(sx - half, 0)
            xb = min(sx + half, W - 1)
            x = _dsu_find(parent, yy, xa)
            while x <= xb:
                w = rem[yy, x] * mix
                for c in range(C):
                    acc[yy, x, c] += w * src[sy, sx, c]
                rem[yy, x] *= keep
                if rem[yy, x] < eps:
                    parent[yy, x] = x + 1
                x = _dsu_find(parent, yy, x + 1)
    for yy in range(H):
        for xx in range(W):
            for c in range(C):
                img[yy, xx, c] = acc[yy, xx, c] + rem[yy, xx] * img[yy, xx, c]


def smear(img: np.ndarray, angle_rad: float, density: float, mix: float, distance: int, shape: int) -> np.ndarray:
    src = np.ascontiguousarray(img, dtype=F32)
    out = src.copy()
    distance = max(1, int(distance))
    if shape in (0, 1):
        _smear_crosses_lines(out, src, int(shape), distance, float(density), float(angle_rad), float(mix))
    else:
        _smear_blobs(out, src, int(shape), distance, float(density), float(mix))
    return out


# ---------------------------------------------------------------------------
# jhlabs Perlin noise + Cellular / Crystallize
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _perlin_tables() -> tuple[np.ndarray, np.ndarray]:
    """PerlinNoise::new(0x4a484c616273): permutation (514,) and 2D gradients (514, 2)."""
    mask = (1 << 48) - 1
    state = (0x4A484C616273 ^ 0x5DEECE66D) & mask

    def next_pos():
        nonlocal state
        state = (state * 0x5DEECE66D + 0xB) & mask
        v = state >> 16
        if v >= 1 << 31:
            v -= 1 << 32
        return v & 0x7FFFFFFF

    perm = np.zeros(514, np.int64)
    g2 = np.zeros((514, 2), np.float32)
    for i in range(256):
        perm[i] = i
        next_pos()  # gradient1
        a = np.float32((next_pos() % 512 - 256) / 256.0)
        b = np.float32((next_pos() % 512 - 256) / 256.0)
        ln = np.float32(np.sqrt(np.float32(a * a + b * b)))
        g2[i] = (a / ln, b / ln) if ln != 0 else (np.nan, np.nan)
        for _ in range(3):  # gradient3
            next_pos()
    for i in range(255, -1, -1):
        t = next_pos() % 256
        perm[i], perm[t] = perm[t], perm[i]
    for i in range(258):
        perm[256 + i] = perm[i]
        g2[256 + i] = g2[i]
    return perm, g2


@functools.lru_cache(maxsize=1)
def _cellular_probabilities() -> np.ndarray:
    values = np.zeros(8192, np.int64)
    factorial = np.float32(1.0)
    total = np.float32(0.0)
    e = np.float32(np.exp(np.float32(-2.5)))
    for i in range(10):
        if i > 1:
            factorial = np.float32(factorial * np.float32(i))
        p = np.float32(np.float32(np.float32(2.5) ** i) * e / factorial)
        start = int(np.float32(total * np.float32(8192.0)))
        total = np.float32(total + p)
        end = int(np.float32(total * np.float32(8192.0)))
        values[min(start, 8192):min(end, 8192)] = i
    return values


@njit(cache=True)
def _noise2(perm, g2, x, y):
    sx = x + 4096.0
    ix = int(sx)
    x0 = ix & 255
    x1 = (x0 + 1) & 255
    rx = sx - ix
    nx = rx - 1.0
    sy = y + 4096.0
    iy = int(sy)
    y0 = iy & 255
    y1 = (y0 + 1) & 255
    ry = sy - iy
    ny = ry - 1.0
    i = perm[x0]
    j = perm[x1]
    b00 = perm[i + y0]
    b10 = perm[j + y0]
    b01 = perm[i + y1]
    b11 = perm[j + y1]
    cx = rx * rx * (3.0 - 2.0 * rx)
    cy = ry * ry * (3.0 - 2.0 * ry)
    tl = rx * g2[b00, 0] + ry * g2[b00, 1]
    tr = nx * g2[b10, 0] + ry * g2[b10, 1]
    bl = rx * g2[b01, 0] + ny * g2[b01, 1]
    br = nx * g2[b11, 0] + ny * g2[b11, 1]
    top = tl + cx * (tr - tl)
    bot = bl + cx * (br - bl)
    return 1.5 * (top + cy * (bot - top))


@njit(cache=True)
def _check_cube(x, y, cube_x, cube_y, grid_type, randomness, perm, g2, probs,
                d0, ax0, ay0, d1, ax1, ay1, d2, ax2, ay2):
    st = _jr_init(571 * cube_x + 23 * cube_y)
    if grid_type == 1 or grid_type == 2:
        npts = 1
    elif grid_type == 3 or grid_type == 4:
        npts = 2
    else:
        st, r = _jr_next_int(st)
        npts = probs[r & 0x1FFF]
    for index in range(npts):
        weight = 1.0
        if grid_type == 1:
            px = 0.5
            py = 0.5
        elif grid_type == 2:
            px = 0.75
            py = 0.0 if (cube_x & 1) == 0 else 0.5
        elif grid_type == 3:
            if index == 0:
                px = 0.207
                py = 0.207
            else:
                px = 0.707
                py = 0.707
                weight = 1.6
        elif grid_type == 4:
            if (cube_y & 1) == 0:
                if index == 0:
                    px = 0.25
                    py = 0.35
                else:
                    px = 0.75
                    py = 0.65
            else:
                if index == 0:
                    px = 0.75
                    py = 0.35
                else:
                    px = 0.25
                    py = 0.65
        else:
            st, px = _jr_next_float(st)
            st, py = _jr_next_float(st)
        if randomness != 0.0:
            if grid_type == 1:
                st, f1 = _jr_next_float(st)
                st, f2 = _jr_next_float(st)
                px += randomness * (f1 - 0.5)
                py += randomness * (f2 - 0.5)
            elif grid_type != 0:
                px += randomness * _noise2(perm, g2, 271.0 * (cube_x + px), 271.0 * (cube_y + py))
                py += randomness * _noise2(perm, g2, 271.0 * (cube_x + px) + 89.0, 271.0 * (cube_y + py) + 137.0)
        dx = abs(x - px) * weight
        dy = abs(y - py) * weight
        dist = math.sqrt(dx * dx + dy * dy)
        qx = cube_x + px
        qy = cube_y + py
        if dist < d0:
            d2, ax2, ay2 = d1, ax1, ay1
            d1, ax1, ay1 = d0, ax0, ay0
            d0, ax0, ay0 = dist, qx, qy
        elif dist < d1:
            d2, ax2, ay2 = d1, ax1, ay1
            d1, ax1, ay1 = dist, qx, qy
        elif dist < d2:
            d2, ax2, ay2 = dist, qx, qy
    return d0, ax0, ay0, d1, ax1, ay1, d2, ax2, ay2


@njit(cache=True)
def _cellular_evaluate(x, y, grid_type, randomness, perm, g2, probs):
    inf = np.inf
    r = (inf, 0.0, 0.0, inf, 0.0, 0.0, inf, 0.0, 0.0)
    ix = int(x)
    iy = int(y)
    fx = x - ix
    fy = y - iy
    r = _check_cube(fx, fy, ix, iy, grid_type, randomness, perm, g2, probs, *r)
    distance = r[6]
    if distance > fy:
        r = _check_cube(fx, fy + 1.0, ix, iy - 1, grid_type, randomness, perm, g2, probs, *r)
        distance = r[6]
    if distance > 1.0 - fy:
        r = _check_cube(fx, fy - 1.0, ix, iy + 1, grid_type, randomness, perm, g2, probs, *r)
        distance = r[6]
    if distance > fx:
        r = _check_cube(fx + 1.0, fy, ix - 1, iy, grid_type, randomness, perm, g2, probs, *r)
        if distance > fy:
            r = _check_cube(fx + 1.0, fy + 1.0, ix - 1, iy - 1, grid_type, randomness, perm, g2, probs, *r)
            distance = r[6]
        if distance > 1.0 - fy:
            r = _check_cube(fx + 1.0, fy - 1.0, ix - 1, iy + 1, grid_type, randomness, perm, g2, probs, *r)
            distance = r[6]
    if distance > 1.0 - fx:
        r = _check_cube(fx - 1.0, fy, ix + 1, iy, grid_type, randomness, perm, g2, probs, *r)
        distance = r[6]
        if distance > fy:
            r = _check_cube(fx - 1.0, fy + 1.0, ix + 1, iy - 1, grid_type, randomness, perm, g2, probs, *r)
            distance = r[6]
        if distance > 1.0 - fy:
            r = _check_cube(fx - 1.0, fy - 1.0, ix + 1, iy + 1, grid_type, randomness, perm, g2, probs, *r)
    return r


@njit(cache=True, parallel=True)
def _jh_crystallize(src, out, scale, stretch, angle, turbulence, grid_type, randomness, edge, edge_color,
                    perm, g2, probs):
    H = src.shape[0]
    W = src.shape[1]
    C = src.shape[2]
    ca = math.cos(angle)
    sa = math.sin(angle)
    for y in prange(H):
        for x in range(W):
            nx = (ca * x + sa * y) / scale + 1000.0
            ny = (-sa * x + ca * y) / (scale * stretch) + 1000.0
            r = _cellular_evaluate(nx, ny, grid_type, randomness, perm, g2, probs)
            if turbulence != 1.0:
                f = 1.0
                while f <= turbulence:
                    r = _cellular_evaluate(f * nx, f * ny, grid_type, randomness, perm, g2, probs)
                    f *= 2.0
            first = r[0]
            second = r[3]
            sx = min(max(int((r[1] - 1000.0) * scale), 0), W - 1)
            sy = min(max(int((r[2] - 1000.0) * scale), 0), H - 1)
            if edge > 0.0:
                v = (second - first) / edge
                if v < 0.0:
                    t = 0.0
                elif v >= edge:
                    t = 1.0
                else:
                    t = v / edge
                    t = t * t * (3.0 - 2.0 * t)
            else:
                t = 1.0
            for c in range(3):
                out[y, x, c] = edge_color[c] * (1.0 - t) + src[sy, sx, c] * t
            src_a = src[sy, sx, 3] if C == 4 else 1.0
            out[y, x, 3] = edge_color[3] * (1.0 - t) + src_a * t


def jh_crystallize(img: np.ndarray, *, scale: float, stretch: float, angle_rad: float, turbulence: float,
                   grid_type: int, randomness: float, edge: float, edge_color) -> np.ndarray:
    src = np.ascontiguousarray(img, dtype=F32)
    out = np.empty(src.shape[:2] + (4,), F32)
    perm, g2 = _perlin_tables()
    _jh_crystallize(src, out, float(scale), float(stretch), float(angle_rad), float(turbulence), int(grid_type),
                    float(randomness), float(edge), np.asarray(edge_color, F32), perm, g2,
                    _cellular_probabilities())
    return out


# ---------------------------------------------------------------------------
# jhlabs Weave
# ---------------------------------------------------------------------------

_WEAVE_CROSSINGS = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], np.int64)


@njit(cache=True)
def _smooth_step(a, b, v):
    if v < a:
        return 0.0
    if v >= b:
        return 1.0
    t = (v - a) / (b - a)
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, parallel=True)
def _weave(src, out, xw, yw, xg, yg, M):
    H = src.shape[0]
    W = src.shape[1]
    has_a = src.shape[2] == 4
    rows = M.shape[0]
    cols = M.shape[1]
    px = np.float32(xw + xg)
    py = np.float32(yw + yg)
    offx = int(xw + xg / 2.0)
    offy = int(yw + yg / 2.0)
    for y in prange(H):
        shy = y + offy
        ny = np.float32(shy) % py
        cy = int(np.float32(shy) / py)
        in_y = ny < yw
        cross_y = _smooth_step(yw / 2.0, yw / 2.0 + yg, abs(yw / 2.0 - ny))
        for x in range(W):
            shx = x + offx
            nx = np.float32(shx) % px
            cx = int(np.float32(shx) / px)
            in_x = nx < xw
            crossing = M[cy % rows, cx % cols]
            t = 0.0
            if in_x and not in_y:
                if crossing != M[(cy + 1) % rows, cx % cols]:
                    t = (1.0 - cross_y if crossing == 0 else cross_y) * 0.5
                elif crossing == 0:
                    t = 0.5
            elif in_y and not in_x:
                cross_x = _smooth_step(xw / 2.0, xw / 2.0 + xg, abs(xw / 2.0 - nx))
                if crossing != M[cy % rows, (cx + 1) % cols]:
                    t = (1.0 - cross_x if crossing == 1 else cross_x) * 0.5
                elif crossing == 1:
                    t = 0.5
            if in_x or in_y:  # mix_colors(t, colour, 0xff000000)
                a = src[y, x, 3] if has_a else 1.0
                out[y, x, 0] = src[y, x, 0] * (1.0 - t)
                out[y, x, 1] = src[y, x, 1] * (1.0 - t)
                out[y, x, 2] = src[y, x, 2] * (1.0 - t)
                out[y, x, 3] = a * (1.0 - t) + t
            else:
                out[y, x, 0] = 0.0
                out[y, x, 1] = 0.0
                out[y, x, 2] = 0.0
                out[y, x, 3] = 0.0


def weave(img: np.ndarray, xw: float, yw: float, xg: float, yg: float) -> np.ndarray:
    """jhlabs WeaveFilter with image colours, square threads and shaded crossings (gaps transparent)."""
    src = np.ascontiguousarray(img, dtype=F32)
    out = np.empty(src.shape[:2] + (4,), F32)
    _weave(src, out, float(max(xw, 1.0)), float(max(yw, 1.0)), float(max(xg, 0.0)), float(max(yg, 0.0)),
           _WEAVE_CROSSINGS)
    return out


# ---------------------------------------------------------------------------
# ImageToolbox ASCIIConverter
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=256)
def _fc_font_for_char(family: str, codepoint: int) -> str | None:
    if sys.platform == "win32" or not shutil.which("fc-match"):
        return None
    try:
        r = subprocess.run(["fc-match", "-f", "%{file}", f"{family or 'sans-serif'}:charset={codepoint:x}"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout:
            return r.stdout
    except Exception:  # pragma: no cover
        return None
    return None


def _glyph_font(font: str, ch: str, size: float):
    from PIL import ImageFont

    from ..draw import find_font

    path = None
    if font and (font.lower().endswith((".ttf", ".otf", ".ttc")) or "/" in font or "\\" in font):
        path = find_font(font)
    if path is None:
        path = _fc_font_for_char(font or "sans-serif", ord(ch)) or find_font(font or "sans-serif")
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _glyph_masks(chars: list[str], font: str, size: float):
    """Render each character once: masks (n, h, w) in a common box + box offset (left, top) to the
    baseline origin (Android ``drawText(x, baseline_y)``)."""
    from PIL import Image, ImageDraw

    rendered = []
    for ch in chars:
        f = _glyph_font(font, ch, size)
        l, t, r, b = f.getbbox(ch, anchor="ls")
        l, t, r, b = int(math.floor(l)), int(math.floor(t)), int(math.ceil(r)), int(math.ceil(b))
        w, h = max(1, r - l), max(1, b - t)
        im = Image.new("L", (w, h), 0)
        ImageDraw.Draw(im).text((-l, -t), ch, font=f, fill=255, anchor="ls")
        rendered.append((l, t, np.asarray(im, F32) / 255.0))
    L = min(x[0] for x in rendered)
    T = min(x[1] for x in rendered)
    R = max(x[0] + x[2].shape[1] for x in rendered)
    B = max(x[1] + x[2].shape[0] for x in rendered)
    masks = np.zeros((len(chars), B - T, R - L), F32)
    for i, (l, t, m) in enumerate(rendered):
        masks[i, t - T:t - T + m.shape[0], l - L:l - L + m.shape[1]] = m
    return masks, L, T


@njit(cache=True)
def _blit_glyphs(prem, masks, glyph, colors, ox, oy, fs, left, top):
    """Canvas.drawText per grid cell in ASCIIConverter order (grid x outer, grid y inner)."""
    H = prem.shape[0]
    W = prem.shape[1]
    gw = glyph.shape[0]
    gh = glyph.shape[1]
    mh = masks.shape[1]
    mw = masks.shape[2]
    for gx in range(gw):
        for gy in range(gh):
            g = glyph[gx, gy]
            if g < 0:
                continue
            a = colors[gx, gy, 3]
            if a <= 0.0:
                continue
            bx = int(math.floor(gx * fs + 0.5)) + left
            by = int(math.floor(gy * fs + 0.5)) + top
            for my in range(mh):
                yy = by + my
                if yy < 0 or yy >= H:
                    continue
                for mx in range(mw):
                    xx = bx + mx
                    if xx < 0 or xx >= W:
                        continue
                    cov = masks[g, my, mx]
                    if cov <= 0.0:
                        continue
                    sa = a * cov
                    inv = 1.0 - sa
                    for c in range(3):
                        prem[yy, xx, c] = colors[gx, gy, c] * sa + prem[yy, xx, c] * inv
                    prem[yy, xx, 3] = sa + prem[yy, xx, 3] * inv


def ascii_art(img: np.ndarray, *, gradient: str, font_size: float, font: str, background, grayscale: bool):
    """Returns a premultiplied RGBA canvas."""
    H, W = img.shape[:2]
    fs = float(font_size)
    if math.floor(W / fs + 0.5) < 5:  # ImageToolbox requires >= 5 columns ("Font size needs to be reduced")
        fs = W / 5.0
    columns = max(1, int(math.floor(W / fs + 0.5)))
    sf = columns if columns <= min(H, W) else min(H, W)
    gw = sf
    gh = max(1, int(sf / W * H))
    small = cv2.resize(np.ascontiguousarray(img, dtype=F32), (gw, gh), interpolation=cv2.INTER_LINEAR)
    small = np.clip(to_rgba(small.reshape(gh, gw, img.shape[2])), 0, 1)
    # Gradient("...").toMap(): char -> 1 - index / max(len - 1, 1); later duplicates win
    thr: dict[str, float] = {}
    n = len(gradient)
    for i, ch in enumerate(gradient):
        thr[ch] = 1.0 - i / max(n - 1, 1)
    metrics = sorted(thr.items(), key=lambda kv: -kv[1])
    bg = np.asarray(background, F32)
    prem = np.empty((H, W, 4), F32)
    # backgroundColor != Color.TRANSPARENT -> drawColor (a transparent colour leaves the canvas empty)
    prem[...] = np.concatenate([bg[:3] * bg[3], bg[3:4]])
    if not metrics:
        return prem
    a8 = np.floor(small[..., 3] * 255.0 + 0.5)
    luma = 0.2126 * small[..., 0] + 0.7152 * small[..., 1] + 0.0722 * small[..., 2]
    chars = [m[0] for m in metrics]
    t = np.array([m[1] for m in metrics], np.float64)
    # first (highest) threshold <= luma; fall back to the first entry
    idx = np.argmax(luma[..., None] >= t[None, None, :] - 1e-7, axis=2)
    none = ~(luma[..., None] >= t[None, None, :] - 1e-7).any(axis=2)
    idx = np.where(none, 0, idx)
    glyph = np.where(a8 > 0, idx, -1).astype(np.int64)
    colors = small.copy()
    if grayscale:
        colors[..., :3] = 0x88 / 255.0
        colors[..., 3] = np.floor(luma * 255.0) / 255.0
    else:
        colors[..., 3] = a8 / 255.0
    masks, left, top = _glyph_masks(chars, font, fs)
    # grids are indexed [x][y] like ASCIIConverter.toGrid()
    _blit_glyphs(prem, masks, np.ascontiguousarray(glyph.T), np.ascontiguousarray(colors.transpose(1, 0, 2)),
                 0, 0, fs, left, top)
    return prem


# ---------------------------------------------------------------------------
# G'MIC bridge with ImageToolboxLibs alpha semantics
# ---------------------------------------------------------------------------

def gmic_seeded(cmd: str, ctx) -> str:
    """Prefix a deterministic ``srand`` (G'MIC seeds its RNG from the clock otherwise)."""
    seed = int(ctx.rng("gmic").integers(0, 2**31 - 1)) if ctx is not None else 0
    return f"srand {seed} {cmd}"


def gmic_preserve(img: np.ndarray, cmd: str, ctx) -> np.ndarray:
    """``GmicAlphaMode.Preserve``: only RGB goes to G'MIC; the source alpha is kept
    (bilinearly resampled if the geometry changed) and G'MIC's own alpha output is ignored."""
    out = run_gmic(np.ascontiguousarray(img[..., :3]), gmic_seeded(cmd, ctx), ctx=ctx)[..., :3]
    if img.shape[2] == 4:
        a = img[..., 3]
        if a.shape != out.shape[:2]:
            a = cv2.resize(a, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR)
        return np.dstack([out, a]).astype(F32)
    return np.ascontiguousarray(out, dtype=F32)


def gmic_process(img: np.ndarray, cmd: str, ctx) -> np.ndarray:
    """``GmicAlphaMode.Process``: RGBA goes to G'MIC (opaque alpha added for RGB input)."""
    had_alpha = img.shape[2] == 4
    out = run_gmic(to_rgba(img), gmic_seeded(cmd, ctx), ctx=ctx, preserve_alpha=False)
    return finish_rgba(out, had_alpha)


def argb_ints(color) -> tuple[int, int, int, int]:
    """Colour param (r, g, b, a floats) -> (R, G, B, A) 0..255 as passed by ImageToolboxLibs."""
    return tuple(int(round(min(max(float(c), 0.0), 1.0) * 255)) for c in color)  # type: ignore[return-value]
