"""Private helpers for the ImageToolbox "Dithering" group.

Ports of
  * Trickle ``Dithering.cpp`` (Apache-2.0, T8RIN): ordered / clustered /
    threshold / random / error-diffusion dithering on 8-bit channels,
  * jhlabs ``ColorHalftoneFilter`` as implemented by ImageToolboxLibs' Rust
    ``libjhlabs`` (Apache-2.0),
  * aire ``WuQuantizer`` + ``RemapPalette`` (MIT, awxkee) for "Quantizier".

All sequential loops are ``numba.njit(cache=True)``.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

F32 = np.float32


@njit(cache=True, inline="always")
def _u8(f):
    """One float channel -> 0..255 int (round half up, clamped)."""
    if f <= 0.0:
        return 0
    if f >= 1.0:
        return 255
    return int(f * 255.0 + 0.5)


@njit(cache=True, inline="always")
def _put(out, y, x, ch, gray, v):
    """Write one dithered value; Trickle's grayscale mode writes the red result to R, G and B."""
    if gray:
        out[y, x, 0] = v
        out[y, x, 1] = v
        out[y, x, 2] = v
    else:
        out[y, x, ch] = v


# ---------------------------------------------------------------------------
# Ordered / threshold dithering
# ---------------------------------------------------------------------------

# Trickle ordered{2,3,4,8}By{..}BayerImpl: matrix[x % f][y % f], value + value * m / div
BAYER = {
    2: (np.array([[1, 3], [4, 2]]), 5),
    3: (np.array([[3, 7, 4], [6, 1, 9], [2, 8, 5]]), 10),
    4: (np.array([[1, 9, 3, 11], [13, 5, 15, 7], [4, 12, 2, 10], [16, 8, 14, 6]]), 17),
    8: (np.array([[1, 49, 13, 61, 4, 52, 16, 64],
                  [33, 17, 45, 29, 36, 20, 48, 32],
                  [9, 57, 5, 53, 12, 60, 8, 56],
                  [41, 25, 37, 21, 44, 28, 40, 24],
                  [3, 51, 15, 63, 2, 50, 14, 62],
                  [35, 19, 47, 31, 34, 18, 46, 30],
                  [11, 59, 7, 55, 10, 58, 6, 54],
                  [43, 27, 39, 23, 42, 26, 38, 22]]), 65),
}

# Trickle clustered{2x2,4x4,8x8}DitheringImpl: threshold = m[y % n][x % n] * k, value > threshold
CLUSTERED = {
    2: (np.array([[0, 2], [3, 1]]), 64),
    4: (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]), 16),
    8: (np.array([[0, 48, 12, 60, 3, 51, 15, 63],
                  [32, 16, 44, 28, 35, 19, 47, 31],
                  [8, 56, 4, 52, 11, 59, 7, 55],
                  [40, 24, 36, 20, 43, 27, 39, 23],
                  [2, 50, 14, 62, 1, 49, 13, 61],
                  [34, 18, 46, 30, 33, 17, 45, 29],
                  [10, 58, 6, 54, 9, 57, 5, 53],
                  [42, 26, 38, 22, 41, 25, 37, 21]]), 4),
}

# Trickle ylilomaDitheringImpl: threshold = m[y % 4][x % 4] * 16, value > threshold
YLILOMA = (np.array([[1, 9, 3, 11], [13, 5, 15, 7], [4, 12, 2, 10], [16, 8, 14, 6]]), 16)

# modes of _pointwise
_BAYER, _ABOVE, _BELOW = 0, 1, 2


@njit(cache=True, parallel=True)
def _pointwise(img, gray, mode, m, div, thr, tmap):
    """Per-pixel threshold dithering.

    mode _BAYER: v + v * m[x % n][y % n] / div < thr -> 0 else 255 (ordered Bayer)
    mode _ABOVE: v > m[y % n][x % n] * div -> 255 else 0 (clustered / Yililoma)
    mode _BELOW: v < (tmap[y, x] if tmap.size else thr) -> 0 else 255 (threshold / random)
    """
    h, w = img.shape[:2]
    n = m.shape[0]
    nc = 1 if gray else 3
    use_map = tmap.shape[0] > 0
    out = np.empty((h, w, 3), np.float32)
    for y in prange(h):
        for x in range(w):
            for ch in range(nc):
                v = _u8(img[y, x, ch])
                if mode == 0:
                    on = v + v * m[x % n, y % n] // div >= thr
                elif mode == 1:
                    on = v > m[y % n, x % n] * div
                else:
                    t = tmap[y, x] if use_map else thr
                    on = v >= t
                _put(out, y, x, ch, gray, 1.0 if on else 0.0)
    return out


_NO_MAP = np.zeros((0, 0), np.int32)
_NO_M = np.zeros((1, 1), np.int64)


def bayer(img: np.ndarray, size: int, threshold: int, gray: bool) -> np.ndarray:
    m, div = BAYER[size]
    return _pointwise(img, gray, _BAYER, m.astype(np.int64), div, int(threshold), _NO_MAP)


def clustered(img: np.ndarray, size: int, gray: bool) -> np.ndarray:
    m, k = CLUSTERED[size]
    return _pointwise(img, gray, _ABOVE, m.astype(np.int64), k, 0, _NO_MAP)


def yliloma(img: np.ndarray, gray: bool) -> np.ndarray:
    m, k = YLILOMA
    return _pointwise(img, gray, _ABOVE, m.astype(np.int64), k, 0, _NO_MAP)


def simple_threshold(img: np.ndarray, threshold: int, gray: bool) -> np.ndarray:
    return _pointwise(img, gray, _BELOW, _NO_M, 1, int(threshold), _NO_MAP)


def random_threshold(img: np.ndarray, rng: np.random.Generator, gray: bool) -> np.ndarray:
    """``std::rand() % 256`` per pixel (shared by the three channels)."""
    t = rng.integers(0, 256, size=img.shape[:2], dtype=np.int32)
    return _pointwise(img, gray, _BELOW, _NO_M, 1, 0, t)


# ---------------------------------------------------------------------------
# Error diffusion (Trickle kernels: (dx, dy, numerator), divisor)
# ---------------------------------------------------------------------------

KERNELS: dict[str, tuple[list[tuple[int, int, int]], int]] = {
    "floyd_steinberg": ([(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)], 16),
    "jarvis_judice_ninke": ([(1, 0, 7), (2, 0, 5),
                             (-2, 1, 3), (-1, 1, 5), (0, 1, 7), (1, 1, 5), (2, 1, 3),
                             (-2, 2, 1), (-1, 2, 3), (0, 2, 5), (1, 2, 3), (2, 2, 1)], 48),
    "sierra": ([(1, 0, 5), (2, 0, 3),
                (-2, 1, 2), (-1, 1, 4), (0, 1, 5), (1, 1, 4), (2, 1, 2),
                (-1, 2, 2), (0, 2, 3), (1, 2, 2)], 32),
    "two_row_sierra": ([(1, 0, 4), (2, 0, 3),
                        (-2, 1, 1), (-1, 1, 2), (0, 1, 3), (1, 1, 2), (2, 1, 1)], 16),
    # Trickle's sierraLiteImpl uses exactly the Two-Row Sierra kernel (not the classic 2/4,1/4,1/4).
    "sierra_lite": ([(1, 0, 4), (2, 0, 3),
                     (-2, 1, 1), (-1, 1, 2), (0, 1, 3), (1, 1, 2), (2, 1, 1)], 16),
    "atkinson": ([(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)], 8),
    "stucki": ([(1, 0, 8), (2, 0, 4),
                (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2),
                (-2, 2, 1), (-1, 2, 2), (0, 2, 4), (1, 2, 2), (2, 2, 1)], 42),
    "burkes": ([(1, 0, 8), (2, 0, 4),
                (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2)], 32),
    "false_floyd_steinberg": ([(1, 0, 3), (0, 1, 3), (1, 1, 2)], 8),
}


def _kernel_tables(name: str):
    """Taps plus a table of the C++ ``num * error / div`` (truncating) for every error in -255..255."""
    taps, div = KERNELS[name]
    dx = np.array([t[0] for t in taps], np.int64)
    dy = np.array([t[1] for t in taps], np.int64)
    num = np.array([t[2] for t in taps], np.int64)
    e = np.arange(-255, 256, dtype=np.int64)
    prod = num[:, None] * e[None, :]
    table = (np.sign(prod) * (np.abs(prod) // div)).astype(np.int32)
    return dx, dy, num, div, table


@njit(cache=True, parallel=True)
def _diffuse(img, gray, dx, dy, num, div, table, threshold):
    """Raster-order (left->right, top->bottom, no serpentine) error diffusion.

    Per channel: ``v = pixel + err``; ``v < threshold`` -> 0 (error v) else 255
    (error v - 255); every tap receives ``num * error / div`` (int, truncated
    toward zero).  Errors leaving the frame are dropped.  Channels are
    independent and processed in parallel.  |error| <= 255 for 8-bit input,
    so the per-tap shares come from ``table``.
    """
    h, w = img.shape[:2]
    nc = 1 if gray else 3
    ntap = dx.shape[0]
    out = np.empty((h, w, 3), np.float32)
    for ch in prange(nc):
        err = np.zeros((3, w + 4), np.int32)  # ring of 3 rows, 2 px padding on both sides
        ring = np.empty(ntap, np.int64)
        for y in range(h):
            for k in range(ntap):
                ring[k] = (y + dy[k]) % 3
            row = y % 3
            for x in range(w):
                v = _u8(img[y, x, ch]) + err[row, x + 2]
                if v < threshold:
                    e = v
                    _put(out, y, x, ch, gray, 0.0)
                else:
                    e = v - 255
                    _put(out, y, x, ch, gray, 1.0)
                if e == 0:
                    continue
                if -255 <= e <= 255:
                    ei = e + 255
                    for k in range(ntap):
                        err[ring[k], x + dx[k] + 2] += table[k, ei]
                else:  # pragma: no cover - unreachable for 8-bit input
                    for k in range(ntap):
                        p = num[k] * e
                        err[ring[k], x + dx[k] + 2] += p // div if p >= 0 else -((-p) // div)
            err[row, :] = 0
    return out


def error_diffusion(img: np.ndarray, kernel: str, threshold: int, gray: bool) -> np.ndarray:
    dx, dy, num, div, table = _kernel_tables(kernel)
    return _diffuse(img, gray, dx, dy, num, div, table, int(threshold))


@njit(cache=True, parallel=True)
def _left_to_right(img, gray, threshold):
    """Trickle simpleLeftToRightErrorDiffusionImpl: 1-D error carried along each row."""
    h, w = img.shape[:2]
    nc = 1 if gray else 3
    out = np.empty((h, w, 3), np.float32)
    for y in prange(h):
        for ch in range(nc):
            error = 0
            for x in range(w):
                v = _u8(img[y, x, ch])
                if v + error < threshold:
                    delta = v
                    _put(out, y, x, ch, gray, 0.0)
                else:
                    delta = v - 255
                    _put(out, y, x, ch, gray, 1.0)
                if abs(delta) < 10:
                    delta = 0
                error += delta
    return out


def left_to_right(img: np.ndarray, threshold: int, gray: bool) -> np.ndarray:
    return _left_to_right(img, gray, int(threshold))


# ---------------------------------------------------------------------------
# jhlabs Color Halftone (ImageToolboxLibs libjhlabs Rust port)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _smooth_step(a, b, v):
    if v < a:
        return 0.0
    if v >= b:
        return 1.0
    t = (v - a) / (b - a)
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True)
def _sat_i32(v):
    """Rust ``f32 as i32``: truncation toward zero, saturating."""
    if v != v:
        return 0
    if v >= 2147483647.0:
        return 2147483647
    if v <= -2147483648.0:
        return -2147483648
    return int(v)


@njit(cache=True, parallel=True)
def _color_halftone(src, dot_radius, angles):
    h, w, _ = src.shape
    out = np.empty((h, w, 3), np.float32)
    grid = 2.0 * dot_radius * 1.414
    half = grid / 2.0
    ox = np.array([0.0, -1.0, 1.0, 0.0, 0.0])
    oy = np.array([0.0, 0.0, 0.0, -1.0, 1.0])
    sines = np.sin(angles)
    cosines = np.cos(angles)
    for y in prange(h):
        fy = float(y)
        for x in range(w):
            fx = float(x)
            for ch in range(3):
                s = sines[ch]
                co = cosines[ch]
                tx = fx * co + fy * s
                ty = -fx * s + fy * co
                gx = tx - ((tx - half) % grid) + half  # rem_euclid (grid > 0)
                gy = ty - ((ty - half) % grid) + half
                factor = 1.0
                for k in range(5):
                    nx = gx + ox[k] * grid
                    ny = gy + oy[k] * grid
                    ix = nx * co - ny * s
                    iy = nx * s + ny * co
                    sx = min(max(_sat_i32(ix), 0), w - 1)
                    sy = min(max(_sat_i32(iy), 0), h - 1)
                    val = _u8(src[sy, sx, ch]) / 255.0
                    dot = (1.0 - val * val) * half * 1.414
                    dist = np.sqrt((fx - ix) ** 2 + (fy - iy) ** 2)
                    factor = min(factor, 1.0 - _smooth_step(dist, dist + 1.0, dot))
                out[y, x, ch] = int(255.0 * factor) / 255.0
    return out


def color_halftone(img: np.ndarray, dot_radius: float, angles_deg: tuple[float, float, float]) -> np.ndarray:
    if 2.0 * dot_radius * 1.414 <= 0.0:
        # grid size 0: every rem_euclid is NaN upstream and f32::min ignores NaN -> white
        return np.ones(img.shape[:2] + (3,), F32)
    angles = np.radians(np.asarray(angles_deg, np.float64))
    return _color_halftone(img, float(dot_radius), angles)


# ---------------------------------------------------------------------------
# GPUImage halftone / crosshatch fragment shaders (float32, pixel-centre texture coordinates)
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _glsl_mod(x, y):
    return x - y * np.floor(x / y)


@njit(cache=True, parallel=True)
def _halftone(h, w, f, aspect, sx, tx, syu, tyu, ix, iy, cell_lum):
    """``1 - step(distance, f/2 * (1 - lum(cell centre)))`` with the aspect-corrected distance."""
    out = np.empty((h, w, 3), np.float32)
    half = np.float32(0.5) * f
    for y in prange(h):
        dyv = syu[y] - tyu[y]
        dy2 = dyv * dyv
        for x in range(w):
            dxv = sx[x] - tx[x]
            dist = np.sqrt(dxv * dxv + dy2)
            radius = half * (np.float32(1.0) - cell_lum[iy[y], ix[x]])
            v = np.float32(1.0) if radius < dist else np.float32(0.0)
            out[y, x, 0] = v
            out[y, x, 1] = v
            out[y, x, 2] = v
    return out


@njit(cache=True, parallel=True)
def _crosshatch(img, tx, ty, spacing, line_width):
    h, w = img.shape[:2]
    out = np.empty((h, w, 3), np.float32)
    half = spacing / np.float32(2.0)
    wr = np.float64(np.float32(0.2125))
    wg = np.float64(np.float32(0.7154))
    wb = np.float64(np.float32(0.0721))
    for y in prange(h):
        for x in range(w):
            # dot(rgb, W) with a single rounding (as a fused GPU dot product does): white stays exactly 1.0
            lum = np.float32(img[y, x, 0] * wr + img[y, x, 1] * wg + img[y, x, 2] * wb)
            s = tx[x] + ty[y]
            d = tx[x] - ty[y]
            black = False
            if lum < 1.0 and _glsl_mod(s, spacing) <= line_width:
                black = True
            if lum < 0.75 and _glsl_mod(d, spacing) <= line_width:
                black = True
            if lum < 0.5 and _glsl_mod(s - half, spacing) <= line_width:
                black = True
            if lum < 0.3 and _glsl_mod(d - half, spacing) <= line_width:
                black = True
            v = np.float32(0.0) if black else np.float32(1.0)
            out[y, x, 0] = v
            out[y, x, 1] = v
            out[y, x, 2] = v
    return out


# ---------------------------------------------------------------------------
# aire "palette" (Xiaolin Wu quantizer, 4-D ARGB histogram, 33 bins per axis)
# ---------------------------------------------------------------------------

_SIDE = 33
_MAXSIDE = 32


@njit(cache=True)
def _idx4(a, r, g, b):
    return a + r * 33 + g * 1089 + b * 35937


@njit(cache=True)
def _wrap32(v):
    """Two's-complement int32 wrap-around of an int64."""
    v = v & 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


@njit(cache=True)
def _f2i(f):
    """ARM64 float->int32 conversion (fcvtzs): truncation, saturating, NaN -> 0."""
    if f != f:
        return 0
    if f >= 2147483647.0:
        return 2147483647
    if f <= -2147483648.0:
        return -2147483648
    return int(f)


@njit(cache=True)
def _sqr_sum(a, r, g, b):
    """``sqr(int) + sqr(int) + ...`` on float arguments with 32-bit int overflow."""
    s = 0
    for f in (a, r, g, b):
        i = _f2i(f)
        s = _wrap32(s + _wrap32(i * i))
    return np.float32(s)


@njit(cache=True)
def _f2u32(f):
    """float -> uint32 argument conversion (fcvtzu: truncation, saturating)."""
    if f != f or f <= 0.0:
        return np.float32(0.0)
    if f >= 4294967295.0:
        return np.float32(4294967295.0)
    return np.float32(np.floor(f))


@njit(cache=True)
def _vol_l(box, m):
    a0, a1, r0, r1, g0, g1, b0, b1 = box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]
    v = ((m[_idx4(a1, r1, g1, b1)] - m[_idx4(a1, r1, g0, b1)] - m[_idx4(a1, r0, g1, b1)] + m[_idx4(a1, r0, g0, b1)]
          - m[_idx4(a0, r1, g1, b1)] + m[_idx4(a0, r1, g0, b1)] + m[_idx4(a0, r0, g1, b1)] - m[_idx4(a0, r0, g0, b1)])
         - (m[_idx4(a1, r1, g1, b0)] - m[_idx4(a0, r1, g1, b0)] - m[_idx4(a1, r1, g0, b0)] + m[_idx4(a0, r1, g0, b0)]
            - m[_idx4(a1, r0, g1, b0)] + m[_idx4(a0, r0, g1, b0)] + m[_idx4(a1, r0, g0, b0)] - m[_idx4(a0, r0, g0, b0)]))
    return np.float32(v)


@njit(cache=True)
def _vol_f(box, m):
    """Volume over the float32 moment table, evaluated left to right in float32."""
    a0, a1, r0, r1, g0, g1, b0, b1 = box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]
    p = m[_idx4(a1, r1, g1, b1)]
    p = np.float32(p - m[_idx4(a1, r1, g0, b1)])
    p = np.float32(p - m[_idx4(a1, r0, g1, b1)])
    p = np.float32(p + m[_idx4(a1, r0, g0, b1)])
    p = np.float32(p - m[_idx4(a0, r1, g1, b1)])
    p = np.float32(p + m[_idx4(a0, r1, g0, b1)])
    p = np.float32(p + m[_idx4(a0, r0, g1, b1)])
    p = np.float32(p - m[_idx4(a0, r0, g0, b1)])
    q = m[_idx4(a1, r1, g1, b0)]
    q = np.float32(q - m[_idx4(a0, r1, g1, b0)])
    q = np.float32(q - m[_idx4(a1, r1, g0, b0)])
    q = np.float32(q + m[_idx4(a0, r1, g0, b0)])
    q = np.float32(q - m[_idx4(a1, r0, g1, b0)])
    q = np.float32(q + m[_idx4(a0, r0, g1, b0)])
    q = np.float32(q + m[_idx4(a1, r0, g0, b0)])
    q = np.float32(q - m[_idx4(a0, r0, g0, b0)])
    return np.float32(p - q)


@njit(cache=True)
def _top(box, d, pos, m):
    a0, a1, r0, r1, g0, g1, b0, b1 = box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]
    if d == 3:  # Alpha
        v = ((m[_idx4(pos, r1, g1, b1)] - m[_idx4(pos, r1, g0, b1)] - m[_idx4(pos, r0, g1, b1)] + m[_idx4(pos, r0, g0, b1)])
             - (m[_idx4(pos, r1, g1, b0)] - m[_idx4(pos, r1, g0, b0)] - m[_idx4(pos, r0, g1, b0)] + m[_idx4(pos, r0, g0, b0)]))
    elif d == 2:  # Red
        v = ((m[_idx4(a1, pos, g1, b1)] - m[_idx4(a1, pos, g0, b1)] - m[_idx4(a0, pos, g1, b1)] + m[_idx4(a0, pos, g0, b1)])
             - (m[_idx4(a1, pos, g1, b0)] - m[_idx4(a1, pos, g0, b0)] - m[_idx4(a0, pos, g1, b0)] + m[_idx4(a0, pos, g0, b0)]))
    elif d == 1:  # Green
        v = ((m[_idx4(a1, r1, pos, b1)] - m[_idx4(a1, r0, pos, b1)] - m[_idx4(a0, r1, pos, b1)] + m[_idx4(a0, r0, pos, b1)])
             - (m[_idx4(a1, r1, pos, b0)] - m[_idx4(a1, r0, pos, b0)] - m[_idx4(a0, r1, pos, b0)] + m[_idx4(a0, r0, pos, b0)]))
    else:  # Blue
        v = ((m[_idx4(a1, r1, g1, pos)] - m[_idx4(a1, r1, g0, pos)] - m[_idx4(a1, r0, g1, pos)] + m[_idx4(a1, r0, g0, pos)])
             - (m[_idx4(a0, r1, g1, pos)] - m[_idx4(a0, r1, g0, pos)] - m[_idx4(a0, r0, g1, pos)] + m[_idx4(a0, r0, g0, pos)]))
    return np.float32(v)


@njit(cache=True)
def _bottom(box, d, m):
    a0, a1, r0, r1, g0, g1, b0, b1 = box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]
    if d == 3:
        v = ((-m[_idx4(a0, r1, g1, b1)] + m[_idx4(a0, r1, g0, b1)] + m[_idx4(a0, r0, g1, b1)] - m[_idx4(a0, r0, g0, b1)])
             - (-m[_idx4(a0, r1, g1, b0)] + m[_idx4(a0, r1, g0, b0)] + m[_idx4(a0, r0, g1, b0)] - m[_idx4(a0, r0, g0, b0)]))
    elif d == 2:
        v = ((-m[_idx4(a1, r0, g1, b1)] + m[_idx4(a1, r0, g0, b1)] + m[_idx4(a0, r0, g1, b1)] - m[_idx4(a0, r0, g0, b1)])
             - (-m[_idx4(a1, r0, g1, b0)] + m[_idx4(a1, r0, g0, b0)] + m[_idx4(a0, r0, g1, b0)] - m[_idx4(a0, r0, g0, b0)]))
    elif d == 1:
        v = ((-m[_idx4(a1, r1, g0, b1)] + m[_idx4(a1, r0, g0, b1)] + m[_idx4(a0, r1, g0, b1)] - m[_idx4(a0, r0, g0, b1)])
             - (-m[_idx4(a1, r1, g0, b0)] + m[_idx4(a1, r0, g0, b0)] + m[_idx4(a0, r1, g0, b0)] - m[_idx4(a0, r0, g0, b0)]))
    else:
        v = ((-m[_idx4(a1, r1, g1, b0)] + m[_idx4(a1, r1, g0, b0)] + m[_idx4(a1, r0, g1, b0)] - m[_idx4(a1, r0, g0, b0)])
             - (-m[_idx4(a0, r1, g1, b0)] + m[_idx4(a0, r1, g0, b0)] + m[_idx4(a0, r0, g1, b0)] - m[_idx4(a0, r0, g0, b0)]))
    return np.float32(v)


@njit(cache=True)
def _maximize(wt, ma, mr, mg, mb, box, d, first, last, w_a, w_r, w_g, w_b, w_w):
    b_a = _bottom(box, d, ma)
    b_r = _bottom(box, d, mr)
    b_g = _bottom(box, d, mg)
    b_b = _bottom(box, d, mb)
    b_w = _bottom(box, d, wt)
    valid = False
    result = np.float32(0.0)
    cut = 0
    for pos in range(first, last):
        h_a = np.float32(b_a + _top(box, d, pos, ma))
        h_r = np.float32(b_r + _top(box, d, pos, mr))
        h_g = np.float32(b_g + _top(box, d, pos, mg))
        h_b = np.float32(b_b + _top(box, d, pos, mb))
        h_w = np.float32(b_w + _top(box, d, pos, wt))
        if h_w == 0:
            continue
        temp = np.float32(_sqr_sum(h_a, h_r, h_g, h_b) / h_w)
        h_a = np.float32(w_a - h_a)
        h_r = np.float32(w_r - h_r)
        h_g = np.float32(w_g - h_g)
        h_b = np.float32(w_b - h_b)
        h_w = np.float32(w_w - h_w)
        if h_w != 0:
            temp = np.float32(temp + np.float32(_sqr_sum(h_a, h_r, h_g, h_b) / h_w))
            if temp > result:
                valid = True
                result = temp
                cut = pos
    return valid, cut, result


@njit(cache=True)
def _box_size(b):
    return (b[1] - b[0]) * (b[3] - b[2]) * (b[5] - b[4]) * (b[7] - b[6])


@njit(cache=True)
def _cut(wt, ma, mr, mg, mb, first, second):
    w_a = _f2u32(_vol_l(first, ma))
    w_r = _f2u32(_vol_l(first, mr))
    w_g = _f2u32(_vol_l(first, mg))
    w_b = _f2u32(_vol_l(first, mb))
    w_w = _f2u32(_vol_l(first, wt))
    # box layout: [a0, a1, r0, r1, g0, g1, b0, b1]; direction ids: 3=A, 2=R, 1=G, 0=B
    va, pa, xa = _maximize(wt, ma, mr, mg, mb, first, 3, (first[0] + 1) & 0xFF, first[1], w_a, w_r, w_g, w_b, w_w)
    vr, pr, xr = _maximize(wt, ma, mr, mg, mb, first, 2, (first[2] + 1) & 0xFF, first[3], w_a, w_r, w_g, w_b, w_w)
    vg, pg, xg = _maximize(wt, ma, mr, mg, mb, first, 1, (first[4] + 1) & 0xFF, first[5], w_a, w_r, w_g, w_b, w_w)
    vb, pb, xb = _maximize(wt, ma, mr, mg, mb, first, 0, (first[6] + 1) & 0xFF, first[7], w_a, w_r, w_g, w_b, w_w)
    d = 0
    if xa >= xr and xa >= xg and xa >= xb:
        if not va:
            return False
        d = 3
    elif xr >= xa and xr >= xg and xr >= xb:
        d = 2
    elif xg >= xa and xg >= xr and xg >= xb:
        d = 1
    second[1] = first[1]
    second[3] = first[3]
    second[5] = first[5]
    second[7] = first[7]
    if d == 3:
        first[1] = pa
        second[0] = pa
        second[2] = first[2]
        second[4] = first[4]
        second[6] = first[6]
    elif d == 2:
        first[3] = pr
        second[2] = pr
        second[0] = first[0]
        second[4] = first[4]
        second[6] = first[6]
    elif d == 1:
        first[5] = pg
        second[4] = pg
        second[0] = first[0]
        second[2] = first[2]
        second[6] = first[6]
    else:
        first[7] = pb
        second[6] = pb
        second[0] = first[0]
        second[2] = first[2]
        second[4] = first[4]
    return True


@njit(cache=True)
def _variance(wt, ma, mr, mg, mb, m2, box):
    v_a = _vol_l(box, ma)
    v_r = _vol_l(box, mr)
    v_g = _vol_l(box, mg)
    v_b = _vol_l(box, mb)
    v_m = _vol_f(box, m2)
    v_w = _vol_l(box, wt)
    dist = _sqr_sum(v_a, v_r, v_g, v_b)
    if v_w != 0:
        return np.float32(v_m - np.float32(dist / v_w))
    return np.float32(0.0)


@njit(cache=True, parallel=True)
def _hist_prep(img):
    """Per pixel: 4-D bin index (alpha = 255 -> alpha bin 32), channel bytes and a^2+r^2+g^2+b^2."""
    h, w = img.shape[:2]
    idx = np.empty(h * w, np.int32)
    rgb = np.empty((3, h * w), np.uint8)
    sq = np.empty(h * w, np.int32)
    for y in prange(h):
        for x in range(w):
            i = y * w + x
            r = _u8(img[y, x, 0])
            g = _u8(img[y, x, 1])
            b = _u8(img[y, x, 2])
            idx[i] = _idx4(32, (r >> 3) + 1, (g >> 3) + 1, (b >> 3) + 1)
            rgb[0, i] = r
            rgb[1, i] = g
            rgb[2, i] = b
            sq[i] = 65025 + r * r + g * g + b * b
    return idx, rgb, sq


@njit(cache=True)
def _int_hist(idx, rgb, total):
    """Bin counts and R/G/B sums; one pass per table keeps the random writes cache resident."""
    wt = np.zeros(total, np.int64)
    for i in range(idx.shape[0]):
        wt[idx[i]] += 1
    out = np.zeros((3, total), np.int64)
    for c in range(3):
        m = out[c]
        v = rgb[c]
        for i in range(idx.shape[0]):
            m[idx[i]] += v[i]
    return wt, out[0], out[1], out[2]


@njit(cache=True)
def _float_hist(idx, sq, total):
    """``moments[index] += sqr(a)+sqr(r)+sqr(g)+sqr(b)`` accumulated in float32, raster order."""
    m2 = np.zeros(total, np.float32)
    for i in range(idx.shape[0]):
        k = idx[i]
        m2[k] = np.float32(m2[k] + np.float32(sq[i]))
    return m2


@njit(cache=True)
def _split_lookup(wt, ma, mr, mg, mb, m2, max_colors):
    """WuQuantizer SplitData + BuildLookups -> (N, 4) RGBA palette."""
    color_count = max_colors
    boxes = np.zeros((color_count, 8), np.int64)
    boxes[0, 1] = _MAXSIDE
    boxes[0, 3] = _MAXSIDE
    boxes[0, 5] = _MAXSIDE
    boxes[0, 7] = _MAXSIDE
    vv = np.zeros(color_count, np.float32)
    nxt = 0
    cube = 1
    while cube < color_count:
        if _cut(wt, ma, mr, mg, mb, boxes[nxt], boxes[cube]):
            vv[nxt] = _variance(wt, ma, mr, mg, mb, m2, boxes[nxt]) if _box_size(boxes[nxt]) > 1 else np.float32(0.0)
            vv[cube] = _variance(wt, ma, mr, mg, mb, m2, boxes[cube]) if _box_size(boxes[cube]) > 1 else np.float32(0.0)
        else:
            vv[nxt] = np.float32(0.0)
            cube -= 1
        nxt = 0
        temp = vv[0]
        for k in range(1, cube + 1):
            if vv[k] <= temp:
                continue
            temp = vv[k]
            nxt = k
        if temp > 0.0:
            cube += 1
            continue
        color_count = cube + 1
        break
    pal = np.zeros((color_count + 1, 4), np.int64)
    n = 0
    for k in range(color_count):
        box = boxes[k]
        weight = _vol_l(box, wt)
        if weight <= 0:
            continue
        pal[n, 3] = min(max(int(np.float32(_vol_l(box, ma) / weight)), 0), 255)
        pal[n, 0] = min(max(int(np.float32(_vol_l(box, mr) / weight)), 0), 255)
        pal[n, 1] = min(max(int(np.float32(_vol_l(box, mg) / weight)), 0), 255)
        pal[n, 2] = min(max(int(np.float32(_vol_l(box, mb) / weight)), 0), 255)
        n += 1
    return pal[:min(n, max_colors)]


def _prefix_int(m: np.ndarray) -> np.ndarray:
    """CalculateMoments for an integer table: uint32 running sums over (b, g, r), then long sum over alpha."""
    v = m.reshape(_SIDE, _SIDE, _SIDE, _SIDE)  # [b, g, r, a]
    v = np.cumsum(np.cumsum(np.cumsum(v, axis=0), axis=1), axis=2) & 0xFFFFFFFF
    return np.cumsum(v, axis=3).reshape(-1)


def _prefix_float(m: np.ndarray) -> np.ndarray:
    """CalculateMoments for the float32 table: sequential float32 sums along b, g, r, a (same order)."""
    v = m.reshape(_SIDE, _SIDE, _SIDE, _SIDE)
    for ax in range(4):
        v = np.cumsum(v, axis=ax, dtype=np.float32)
    return np.ascontiguousarray(v.reshape(-1))


def wu_palette(img: np.ndarray, max_colors: int) -> np.ndarray:
    """aire ``WuQuantizer::quantizeImage(maxColors, alphaThreshold=15, alphaFader=15)`` -> (N, 4) RGBA palette.

    The input is opaque (LazyRoom preserves alpha separately), so every pixel
    lands in alpha bin 32 with alpha moment 255.
    """
    total = _SIDE ** 4
    idx, rgb, sq = _hist_prep(np.ascontiguousarray(img[..., :3], dtype=F32))
    # BuildHistogram (integer moments are order independent)
    wt, mr, mg, mb = _int_hist(idx, rgb, total)
    ma = wt * 255
    m2 = _float_hist(idx, sq, total)
    del idx, rgb, sq
    # AdjustMoments: skipped when maxColors^2 / nonEmptyBins (integer division) == 0
    nz = np.flatnonzero(wt)
    if len(nz) and (max_colors * max_colors) // len(nz) != 0:
        d = wt[nz].astype(np.float64)
        sw = np.sqrt(d).astype(np.int64)
        ratio = sw / d
        wt[nz] = sw
        for arr in (mr, mg, mb, ma):
            arr[nz] = (arr[nz] * ratio).astype(np.int64)
        m2[nz] = (m2[nz].astype(np.float64) * ratio).astype(np.float32)
    # CalculateMoments
    wt, ma, mr, mg, mb = (_prefix_int(a) for a in (wt, ma, mr, mg, mb))
    m2 = _prefix_float(m2)
    return _split_lookup(wt, ma, mr, mg, mb, m2, int(max_colors))


@njit(cache=True, parallel=True)
def _pack(img):
    h, w = img.shape[:2]
    out = np.empty((h, w), np.int32)
    for y in prange(h):
        for x in range(w):
            out[y, x] = (_u8(img[y, x, 0]) << 16) | (_u8(img[y, x, 1]) << 8) | _u8(img[y, x, 2])
    return out


@njit(cache=True, parallel=True)
def _nearest(colors, pal):
    """Exact nearest palette entry for packed opaque colours under aire's CoverTreePoint metric.

    Palette rows are sorted by green (the heaviest weight); the scan walks away
    from the query's green in both directions and stops once 4*dG^2 alone
    exceeds the best distance found.  Ties keep the first entry in that order.
    """
    m = pal.shape[0]
    n = colors.shape[0]
    res = np.empty(n, np.int64)
    gcol = pal[:, 1].copy()
    for i in prange(n):
        c = colors[i]
        r = (c >> 16) & 255
        g = (c >> 8) & 255
        b = c & 255
        wb = 3 if r < 128 else 2
        start = np.searchsorted(gcol, g)
        best = 1 << 62
        bi = 0
        j = start
        while j < m:
            dg = pal[j, 1] - g
            if 4 * dg * dg > best:
                break
            dr = pal[j, 0] - r
            db = pal[j, 2] - b
            da = pal[j, 3] - 255
            d = 2 * dr * dr + 4 * dg * dg + wb * db * db + da * da
            if d < best:
                best = d
                bi = j
            j += 1
        j = start - 1
        while j >= 0:
            dg = pal[j, 1] - g
            if 4 * dg * dg > best:
                break
            dr = pal[j, 0] - r
            db = pal[j, 2] - b
            da = pal[j, 3] - 255
            d = 2 * dr * dr + 4 * dg * dg + wb * db * db + da * da
            if d < best:
                best = d
                bi = j
            j -= 1
        res[i] = bi
    return res


@njit(cache=True, parallel=True)
def _apply_palette(packed, lut, pal):
    h, w = packed.shape
    out = np.empty((h, w, 3), np.float32)
    for y in prange(h):
        for x in range(w):
            k = lut[packed[y, x]]
            out[y, x, 0] = pal[k, 0] / np.float32(255.0)
            out[y, x, 1] = pal[k, 1] / np.float32(255.0)
            out[y, x, 2] = pal[k, 2] / np.float32(255.0)
    return out


def remap_nearest(img: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """aire RemapPalette::remap (Cover-tree strategy): every pixel -> nearest palette colour.

    distance = 2*dR^2 + 4*dG^2 + w_b*dB^2 + dA^2 with w_b = 3 if the *query* red
    < 128 else 2 (CoverTreePoint::distance).  Solved exactly over the distinct
    colours of the image.  The selected JJN "dithering" of RemapPalette has no
    effect upstream (diffused errors are written into destination pixels that
    are overwritten afterwards), so none is applied here.
    """
    h, w = img.shape[:2]
    if len(palette) == 0:
        return np.zeros((h, w, 3), F32)
    pal = palette[np.argsort(palette[:, 1], kind="stable")].astype(np.int64)
    packed = _pack(img)
    present = np.zeros(1 << 24, np.bool_)
    present[packed.ravel()] = True
    uniq = np.flatnonzero(present).astype(np.int64)
    lut = np.zeros(1 << 24, np.int32)
    lut[uniq] = _nearest(uniq, pal)
    return _apply_palette(packed, lut, pal.astype(F32))
