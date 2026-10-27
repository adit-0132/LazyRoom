"""numba kernels for the JH Labs texture presets.

Port of ImageToolboxLibs' Rust re-implementation of the JH Labs filters
(``libs/jhlabs/src/main/rust/jhlabs/src/lib.rs``, Apache-2.0, T8RIN; original
JH Labs filters by Jerry Huxtable, Apache-2.0) - which is what ImageToolbox
actually runs - plus the Java colormaps it calls back into (``LinearColormap``,
``GrayscaleColormap``, ``ImageMath.mixColors``).

Arithmetic is carried out in float32 like the Rust/Java code (``f32``/``float``)
and every float->int conversion follows Rust ``as`` / Java ``(int)`` semantics
(truncation, saturation, NaN -> 0).  Pixels are ARGB ints held in int64 arrays.
"""
from __future__ import annotations

import functools

import numpy as np

from ._nb import njit, prange

F32 = np.float32
_MULT = 0x5DEECE66D
_MASK48 = (1 << 48) - 1
_PI32 = np.float32(np.pi)
PERLIN_SEED = 0x4A484C616273


# ---------------------------------------------------------------------------
# Scalar helpers (Rust / Java semantics)
# ---------------------------------------------------------------------------

@njit(cache=True, error_model="numpy", inline="always")
def _as_i32(v):
    """Rust ``f32 as i32`` / Java ``(int) f``: truncation, saturation, NaN -> 0."""
    if v != v:
        return 0
    if v >= 2147483647.0:
        return 2147483647
    if v <= -2147483648.0:
        return -2147483648
    return int(v)


@njit(cache=True, error_model="numpy", inline="always")
def _clamp_u8(v):
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v


@njit(cache=True, error_model="numpy", inline="always")
def _argb(a, r, g, b):
    return (np.int64(a) << 24) | (np.int64(r) << 16) | (np.int64(g) << 8) | np.int64(b)


@njit(cache=True, error_model="numpy", inline="always")
def _mix_colors(amount, first, second):
    """Rust ``mix_colors``: f32 lerp per channel, ``as i32`` truncation, clamp 0..255."""
    t = np.float32(amount)
    if t < np.float32(0.0):
        t = np.float32(0.0)
    elif t > np.float32(1.0):
        t = np.float32(1.0)
    inv = np.float32(1.0) - t
    a = _clamp_u8(_as_i32(np.float32((first >> 24) & 0xFF) * inv + np.float32((second >> 24) & 0xFF) * t))
    r = _clamp_u8(_as_i32(np.float32((first >> 16) & 0xFF) * inv + np.float32((second >> 16) & 0xFF) * t))
    g = _clamp_u8(_as_i32(np.float32((first >> 8) & 0xFF) * inv + np.float32((second >> 8) & 0xFF) * t))
    b = _clamp_u8(_as_i32(np.float32(first & 0xFF) * inv + np.float32(second & 0xFF) * t))
    return _argb(a, r, g, b)


@njit(cache=True, error_model="numpy", inline="always")
def _java_lerp(t, a, b):
    """Java ``ImageMath.lerp(float t, int a, int b)`` = ``(int) (a + t * (b - a))``."""
    return _as_i32(np.float32(a) + t * np.float32(b - a))


@njit(cache=True, error_model="numpy", inline="always")
def _linear_colormap(v, c1, c2):
    """Java ``LinearColormap.getColor`` -> ``ImageMath.mixColors(clamp(v, 0, 1), c1, c2)``."""
    t = np.float32(v)
    if t < np.float32(0.0):
        t = np.float32(0.0)
    elif t > np.float32(1.0):
        t = np.float32(1.0)
    a = _java_lerp(t, (c1 >> 24) & 0xFF, (c2 >> 24) & 0xFF)
    r = _java_lerp(t, (c1 >> 16) & 0xFF, (c2 >> 16) & 0xFF)
    g = _java_lerp(t, (c1 >> 8) & 0xFF, (c2 >> 8) & 0xFF)
    b = _java_lerp(t, c1 & 0xFF, c2 & 0xFF)
    return ((np.int64(a) << 24) | (np.int64(r) << 16) | (np.int64(g) << 8) | np.int64(b)) & 0xFFFFFFFF


@njit(cache=True, error_model="numpy", inline="always")
def _grayscale_colormap(v):
    """Java ``GrayscaleColormap.getColor``."""
    n = _as_i32(np.float32(v) * np.float32(255.0))
    if n < 0:
        n = 0
    elif n > 255:
        n = 255
    return _argb(255, n, n, n)


@njit(cache=True, error_model="numpy", inline="always")
def _smooth_step(first, second, value):
    if value < first:
        return np.float32(0.0)
    if value >= second:
        return np.float32(1.0)
    v = (value - first) / (second - first)
    return v * v * (np.float32(3.0) - np.float32(2.0) * v)


@njit(cache=True, error_model="numpy", inline="always")
def _triangle(value):
    r = value % np.float32(1.0)  # Rust rem_euclid
    if r < np.float32(0.0):
        r = r + np.float32(1.0)
    if r < np.float32(0.5):
        return np.float32(2.0) * r
    return np.float32(2.0) * (np.float32(1.0) - r)


@njit(cache=True, error_model="numpy", inline="always")
def _rust_fract(v):
    return v - np.float32(np.trunc(v))


# Java Random (48-bit LCG) without int64 overflow: state < 2**48, multiplier < 2**35
@njit(cache=True, error_model="numpy", inline="always")
def _jr_init(seed):
    return (np.int64(seed) ^ 0x5DEECE66D) & 0xFFFFFFFFFFFF


@njit(cache=True, error_model="numpy", inline="always")
def _jr_step(state):
    lo = state & 0xFFFFFF
    hi = state >> 24
    prod = lo * 0x5DEECE66D + (((hi * 0x5DEECE66D) & 0xFFFFFF) << 24)
    return (prod + 0xB) & 0xFFFFFFFFFFFF


@njit(cache=True, error_model="numpy", inline="always")
def _jr_bits(state, bits):
    v = (state >> (48 - bits)) & 0xFFFFFFFF
    if v >= 0x80000000:
        v -= 0x100000000
    return v


@njit(cache=True, error_model="numpy", inline="always")
def _jr_float(state):
    """Returns (new_state, next_float())."""
    state = _jr_step(state)
    return state, np.float32(_jr_bits(state, 24)) / np.float32(16777216.0)


@njit(cache=True, error_model="numpy", inline="always")
def _jr_int(state):
    state = _jr_step(state)
    return state, _jr_bits(state, 32)


# ---------------------------------------------------------------------------
# Rust PerlinNoise (JH Labs Noise port), tables built in Python with f32 math
# ---------------------------------------------------------------------------

class _JavaRandom:
    def __init__(self, seed: int) -> None:
        self.seed = (seed ^ _MULT) & _MASK48

    def next(self, bits: int) -> int:
        self.seed = (self.seed * _MULT + 0xB) & _MASK48
        v = (self.seed >> (48 - bits)) & 0xFFFFFFFF
        return v - (1 << 32) if v >= (1 << 31) else v


@functools.lru_cache(maxsize=16)
def perlin_tables(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``PerlinNoise::new(seed)`` -> (permutation int64[514], gradient2 f32[514,2], gradient3 f32[514,3])."""
    rnd = _JavaRandom(int(seed))

    def positive() -> int:
        return rnd.next(32) & 0x7FFFFFFF

    perm = list(range(256)) + [0] * 258
    g1 = [F32(0)] * 514
    g2 = [[F32(0), F32(0)] for _ in range(514)]
    g3 = [[F32(0), F32(0), F32(0)] for _ in range(514)]
    with np.errstate(all="ignore"):
        for i in range(256):
            g1[i] = F32((positive() % 512) - 256) / F32(256.0)
            for c in range(2):
                g2[i][c] = F32((positive() % 512) - 256) / F32(256.0)
            ln = np.sqrt(g2[i][0] * g2[i][0] + g2[i][1] * g2[i][1])
            g2[i][0] = g2[i][0] / ln
            g2[i][1] = g2[i][1] / ln
            for c in range(3):
                g3[i][c] = F32((positive() % 512) - 256) / F32(256.0)
            ln = np.sqrt(g3[i][0] * g3[i][0] + g3[i][1] * g3[i][1] + g3[i][2] * g3[i][2])
            for c in range(3):
                g3[i][c] = g3[i][c] / ln
    for i in range(255, -1, -1):
        target = positive() % 256
        perm[i], perm[target] = perm[target], perm[i]
    for i in range(258):
        perm[256 + i] = perm[i]
        g1[256 + i] = g1[i]
        g2[256 + i] = list(g2[i])
        g3[256 + i] = list(g3[i])
    return (np.array(perm, np.int64), np.array(g2, np.float32).reshape(514, 2),
            np.array(g3, np.float32).reshape(514, 3))


@njit(cache=True, error_model="numpy", inline="always")
def _curve(v):
    return v * v * (np.float32(3.0) - np.float32(2.0) * v)


@njit(cache=True, error_model="numpy")
def _noise2(perm, g2, x, y):
    sx = x + np.float32(4096.0)
    ix = _as_i32(sx)
    x0 = ix & 255
    x1 = (x0 + 1) & 255
    rx = sx - np.float32(ix)
    nx = rx - np.float32(1.0)
    sy = y + np.float32(4096.0)
    iy = _as_i32(sy)
    y0 = iy & 255
    y1 = (y0 + 1) & 255
    ry = sy - np.float32(iy)
    ny = ry - np.float32(1.0)
    first = perm[x0]
    second = perm[x1]
    c0 = perm[first + y0]
    c1 = perm[second + y0]
    c2 = perm[first + y1]
    c3 = perm[second + y1]
    cx = _curve(rx)
    cy = _curve(ry)
    tl = rx * g2[c0, 0] + ry * g2[c0, 1]
    tr = nx * g2[c1, 0] + ry * g2[c1, 1]
    bl = rx * g2[c2, 0] + ny * g2[c2, 1]
    br = nx * g2[c3, 0] + ny * g2[c3, 1]
    top = tl + cx * (tr - tl)
    bottom = bl + cx * (br - bl)
    return np.float32(1.5) * (top + cy * (bottom - top))


@njit(cache=True, error_model="numpy")
def _noise3(perm, g3, x, y, z):
    sx = x + np.float32(4096.0)
    ix = _as_i32(sx)
    x0 = ix & 255
    x1 = (x0 + 1) & 255
    rx0 = sx - np.float32(ix)
    rx1 = rx0 - np.float32(1.0)
    sy = y + np.float32(4096.0)
    iy = _as_i32(sy)
    y0 = iy & 255
    y1 = (y0 + 1) & 255
    ry0 = sy - np.float32(iy)
    ry1 = ry0 - np.float32(1.0)
    sz = z + np.float32(4096.0)
    iz = _as_i32(sz)
    z0 = iz & 255
    z1 = (iz & 255) + 1
    z1 = z1 & 255
    rz0 = sz - np.float32(iz)
    rz1 = rz0 - np.float32(1.0)
    first = perm[x0]
    second = perm[x1]
    b0 = perm[first + y0]
    b1 = perm[second + y0]
    b2 = perm[first + y1]
    b3 = perm[second + y1]
    cx = _curve(rx0)
    cy = _curve(ry0)
    cz = _curve(rz0)
    # front plane (z0, dz = rz0)
    i0 = b0 + z0
    i1 = b1 + z0
    i2 = b2 + z0
    i3 = b3 + z0
    tl = rx0 * g3[i0, 0] + ry0 * g3[i0, 1] + rz0 * g3[i0, 2]
    tr = rx1 * g3[i1, 0] + ry0 * g3[i1, 1] + rz0 * g3[i1, 2]
    bl = rx0 * g3[i2, 0] + ry1 * g3[i2, 1] + rz0 * g3[i2, 2]
    br = rx1 * g3[i3, 0] + ry1 * g3[i3, 1] + rz0 * g3[i3, 2]
    top = tl + cx * (tr - tl)
    bottom = bl + cx * (br - bl)
    front = top + cy * (bottom - top)
    # back plane (z1, dz = rz1)
    i0 = b0 + z1
    i1 = b1 + z1
    i2 = b2 + z1
    i3 = b3 + z1
    tl = rx0 * g3[i0, 0] + ry0 * g3[i0, 1] + rz1 * g3[i0, 2]
    tr = rx1 * g3[i1, 0] + ry0 * g3[i1, 1] + rz1 * g3[i1, 2]
    bl = rx0 * g3[i2, 0] + ry1 * g3[i2, 1] + rz1 * g3[i2, 2]
    br = rx1 * g3[i3, 0] + ry1 * g3[i3, 1] + rz1 * g3[i3, 2]
    top = tl + cx * (tr - tl)
    bottom = bl + cx * (br - bl)
    back = top + cy * (bottom - top)
    return np.float32(1.5) * (front + cz * (back - front))


@njit(cache=True, error_model="numpy")
def _turbulence3(perm, g3, x, y, z, octaves):
    result = np.float32(0.0)
    freq = np.float32(1.0)
    while freq <= octaves:
        result += np.float32(abs(_noise3(perm, g3, freq * x, freq * y, freq * z))) / freq
        freq *= np.float32(2.0)
    return result


# ---------------------------------------------------------------------------
# CheckFilter
# ---------------------------------------------------------------------------

@njit(cache=True, error_model="numpy", parallel=True)
def check_kernel(w, h, fg, bg, x_scale_i, y_scale_i, fuzz_i, angle):
    out = np.empty((h, w), np.int64)
    xs = np.float32(max(x_scale_i, 1))
    ys = np.float32(max(y_scale_i, 1))
    fuzz = np.float32(fuzz_i) / np.float32(100.0)
    a = np.float32(angle)
    sine = np.float32(np.sin(a))
    cosine = np.float32(np.cos(a))
    half = fuzz * np.float32(0.5)
    for y in prange(h):
        yf = np.float32(y)
        for x in range(w):
            xf = np.float32(x)
            nx = (cosine * xf + sine * yf) / xs
            ny = (-sine * xf + cosine * yf) / ys
            parity = (_as_i32(np.floor(nx)) + _as_i32(np.floor(ny))) & 1
            fx = abs(_rust_fract(nx))
            fy = abs(_rust_fract(ny))
            edge = min(min(min(fx, fy), np.float32(1.0) - fx), np.float32(1.0) - fy)
            if fuzz == np.float32(0.0):
                amount = np.float32(parity)
            else:
                s = _smooth_step(np.float32(0.0), half, edge)
                amount = s * np.float32(parity) + (np.float32(1.0) - s) * np.float32(0.5)
            out[y, x] = _mix_colors(amount, bg, fg)
    return out


# ---------------------------------------------------------------------------
# BrushedMetalFilter (+ box_blur_exact with its edge quirks)
# ---------------------------------------------------------------------------

@njit(cache=True, error_model="numpy")
def _box_blur_transpose(inp, width, height, radius):
    r = int(radius) if radius > 0 else 0
    ts = 2 * r + 1
    out = np.empty(inp.shape[0], np.int64)
    in_idx = 0
    for y in range(height):
        out_idx = y
        ta = 0
        tr = 0
        tg = 0
        tb = 0
        for off in range(-r, r + 1):
            o = min(max(off, 0), width - 1)
            c = inp[in_idx + o]
            ta += (c >> 24) & 0xFF
            tr += (c >> 16) & 0xFF
            tg += (c >> 8) & 0xFF
            tb += c & 0xFF
        for x in range(width):
            out[out_idx] = _argb(ta // ts, tr // ts, tg // ts, tb // ts)
            add = min(x + r + 1, width - 1)
            rem = x - r if x > r else 0
            ca = inp[in_idx + add]
            cr = inp[in_idx + rem]
            ta += ((ca >> 24) & 0xFF) - ((cr >> 24) & 0xFF)
            tr += ((ca >> 16) & 0xFF) - ((cr >> 16) & 0xFF)
            tg += ((ca >> 8) & 0xFF) - ((cr >> 8) & 0xFF)
            tb += (ca & 0xFF) - (cr & 0xFF)
            out_idx += height
        in_idx += width
    return out


@njit(cache=True, error_model="numpy", inline="always")
def _frac_channel(first, second, third, shift, fraction, factor):
    v = ((second >> shift) & 0xFF) + _as_i32(np.float32(((first >> shift) & 0xFF) + ((third >> shift) & 0xFF))
                                              * fraction)
    return _clamp_u8(_as_i32(np.float32(v) * factor))


@njit(cache=True, error_model="numpy")
def _box_blur_fractional_transpose(inp, width, height, radius):
    fraction = radius - np.float32(_as_i32(radius))
    factor = np.float32(1.0) / (np.float32(1.0) + np.float32(2.0) * fraction)
    out = np.zeros(inp.shape[0], np.int64)
    in_idx = 0
    for y in range(height):
        out_idx = y
        out[out_idx] = inp[0]  # upstream quirk: first pixel of the whole buffer
        out_idx += height
        for x in range(1, max(width - 1, 0) if width >= 1 else 0):
            a = inp[in_idx + x - 1]
            b = inp[in_idx + x]
            c = inp[in_idx + x + 1]
            out[out_idx] = _argb(_frac_channel(a, b, c, 24, fraction, factor),
                                 _frac_channel(a, b, c, 16, fraction, factor),
                                 _frac_channel(a, b, c, 8, fraction, factor),
                                 _frac_channel(a, b, c, 0, fraction, factor))
            out_idx += height
        if width > 1:
            out[out_idx] = inp[width - 1]  # upstream quirk: last pixel of the first row
        in_idx += width
    return out


@njit(cache=True, error_model="numpy")
def _box_blur_exact(src, width, height, hr, vr):
    inp = src.copy()
    o = _box_blur_transpose(inp, width, height, hr)
    inp = _box_blur_transpose(o, height, width, vr)
    o = _box_blur_fractional_transpose(inp, width, height, hr)
    return _box_blur_fractional_transpose(o, height, width, vr)


@njit(cache=True, error_model="numpy")
def brushed_metal_kernel(w, h, base, radius, amount, shine, monochrome, seed):
    pix = np.empty(w * h, np.int64)
    amt = np.float32(amount) * np.float32(255.0)
    sh = np.float32(shine)
    st = _jr_init(seed)
    two_pi = np.float32(2.0) * _PI32
    br = np.float32((base >> 16) & 0xFF)
    bgc = np.float32((base >> 8) & 0xFF)
    bb = np.float32(base & 0xFF)
    wf = np.float32(w)
    for y in range(h):
        for x in range(w):
            hl = np.float32(255.0) * sh * np.float32(np.sin(two_pi * np.float32(x) / wf))
            st, f = _jr_float(st)
            n1 = (f * np.float32(2.0) - np.float32(1.0)) * amt
            if monochrome:
                n2 = n1
                n3 = n1
            else:
                st, f = _jr_float(st)
                n2 = (f * np.float32(2.0) - np.float32(1.0)) * amt
                st, f = _jr_float(st)
                n3 = (f * np.float32(2.0) - np.float32(1.0)) * amt
            pix[y * w + x] = _argb(255, _clamp_u8(_as_i32(br + hl + n1)), _clamp_u8(_as_i32(bgc + hl + n2)),
                                   _clamp_u8(_as_i32(bb + hl + n3)))
    if radius > 0:
        pix = _box_blur_exact(pix, w, h, np.float32(radius), np.float32(0.0))
    return pix.reshape(h, w)


# ---------------------------------------------------------------------------
# CausticsFilter / FBMFilter / MarbleTexFilter / WoodFilter / PlasmaFilter
# ---------------------------------------------------------------------------

@njit(cache=True, error_model="numpy", parallel=True)
def caustics_kernel(w, h, perm, g3, scale, amount, turbulence, time, brightness_i, bg):
    out = np.empty((h, w), np.int64)
    sc = max(np.float32(scale), np.float32(0.01))
    amt = np.float32(amount)
    turb = max(np.float32(turbulence), np.float32(1.0))
    t = np.float32(time)
    bright = np.float32(brightness_i) / np.float32(10.0)
    for y in prange(h):
        ny = np.float32(y) / sc
        for x in range(w):
            nx = np.float32(x) / sc
            n = _turbulence3(perm, g3, nx, ny, t, turb)
            r = np.float32(1.0) - abs(n)
            r2 = r * r
            r4 = r2 * r2
            ridge = r4 * r4 * amt * bright
            out[y, x] = _mix_colors(ridge, bg, 0xFFFFFFFF)
    return out


@njit(cache=True, error_model="numpy", parallel=True)
def fbm_family_kernel(w, h, perm, g3, mode, scale, stretch, angle, turbulence, factor, amount, c1, c2):
    """mode 0 = FBMFilter (GrayscaleColormap, mixed with the transparent source by ``amount``),
    1 = MarbleTexFilter (LinearColormap c1->c2, ``factor`` = turbulenceFactor),
    2 = WoodFilter (LinearColormap c1->c2, ``factor`` = rings)."""
    out = np.empty((h, w), np.int64)
    sc = max(np.float32(scale), np.float32(0.01))
    stv = max(np.float32(stretch), np.float32(0.01))
    a = np.float32(angle)
    sine = np.float32(np.sin(a))
    cosine = np.float32(np.cos(a))
    turb = max(np.float32(turbulence), np.float32(1.0))
    fac = np.float32(factor)
    amt = np.float32(amount)
    for y in prange(h):
        yf = np.float32(y)
        for x in range(w):
            xf = np.float32(x)
            nx = (cosine * xf + sine * yf) / sc
            ny = (-sine * xf + cosine * yf) / (sc * stv)
            noise = _turbulence3(perm, g3, nx, ny, np.float32(0.0), turb)
            if mode == 1:
                v = _triangle(nx + fac * noise)
            elif mode == 2:
                v = _triangle(np.float32(np.sqrt(nx * nx + ny * ny)) + fac * noise)
            else:
                v = (noise + np.float32(1.0)) * np.float32(0.5)
            if v < np.float32(0.0):
                v = np.float32(0.0)
            elif v > np.float32(1.0):
                v = np.float32(1.0)
            if mode == 0:
                out[y, x] = _mix_colors(amt, 0, _grayscale_colormap(v))
            else:
                out[y, x] = _linear_colormap(v, c1, c2)
    return out


@njit(cache=True, error_model="numpy", parallel=True)
def plasma_kernel(w, h, perm, g3, turbulence, scaling, phase_seed, source):
    out = np.empty((h, w), np.int64)
    turb = max(np.float32(turbulence), np.float32(1.0))
    scl = np.float32(scaling)
    phase = np.float32(phase_seed) * np.float32(0.0001)
    wf = np.float32(w)
    hf = np.float32(h)
    for y in prange(h):
        ny = np.float32(y) / hf * np.float32(4.0)
        for x in range(w):
            nx = np.float32(x) / wf * np.float32(4.0)
            v = (_turbulence3(perm, g3, nx + phase, ny, phase, turb) + np.float32(1.0)) * np.float32(0.5) + scl
            if v < np.float32(0.0):
                v = np.float32(0.0)
            elif v > np.float32(1.0):
                v = np.float32(1.0)
            col = _linear_colormap(v, 0xFF000000, 0xFFFFFFFF)
            out[y, x] = _mix_colors(v, source, col)
    return out


# ---------------------------------------------------------------------------
# CellularFilter (useColor = true, solid source colour)
# ---------------------------------------------------------------------------

@njit(cache=True, error_model="numpy", inline="always")
def _insert(d, d0, d1, d2):
    if d < d0:
        return d, d0, d1
    if d < d1:
        return d0, d, d1
    if d < d2:
        return d0, d1, d
    return d0, d1, d2


@njit(cache=True, error_model="numpy")
def _check_cube(x, y, cube_x, cube_y, randomness, grid, power, probs, perm, g2, d0, d1, d2):
    st = _jr_init(np.int64(571) * cube_x + np.int64(23) * cube_y)
    if grid == 1 or grid == 2:
        npts = 1
    elif grid == 3 or grid == 4:
        npts = 2
    else:
        st, v = _jr_int(st)
        npts = probs[v & 0x1FFF]
    for index in range(npts):
        weight = np.float32(1.0)
        if grid == 1:
            px = np.float32(0.5)
            py = np.float32(0.5)
        elif grid == 2:
            px = np.float32(0.75)
            py = np.float32(0.0) if (cube_x & 1) == 0 else np.float32(0.5)
        elif grid == 3:
            if index == 0:
                px = np.float32(0.207)
                py = np.float32(0.207)
            else:
                px = np.float32(0.707)
                py = np.float32(0.707)
                weight = np.float32(1.6)
        elif grid == 4:
            if (cube_y & 1) == 0:
                if index == 0:
                    px = np.float32(0.25)
                    py = np.float32(0.35)
                else:
                    px = np.float32(0.75)
                    py = np.float32(0.65)
            else:
                if index == 0:
                    px = np.float32(0.75)
                    py = np.float32(0.35)
                else:
                    px = np.float32(0.25)
                    py = np.float32(0.65)
        else:
            st, px = _jr_float(st)
            st, py = _jr_float(st)
        if randomness != np.float32(0.0):
            if grid == 1:
                st, f = _jr_float(st)
                px += randomness * (f - np.float32(0.5))
                st, f = _jr_float(st)
                py += randomness * (f - np.float32(0.5))
            elif grid != 0:
                px += randomness * _noise2(perm, g2, np.float32(271.0) * (np.float32(cube_x) + px),
                                           np.float32(271.0) * (np.float32(cube_y) + py))
                py += randomness * _noise2(perm, g2, np.float32(271.0) * (np.float32(cube_x) + px) + np.float32(89.0),
                                           np.float32(271.0) * (np.float32(cube_y) + py) + np.float32(137.0))
        dx = abs(x - px) * weight
        dy = abs(y - py) * weight
        if power == np.float32(1.0):
            dist = dx + dy
        elif power == np.float32(2.0):
            dist = np.float32(np.sqrt(dx * dx + dy * dy))
        else:
            dist = np.float32(np.power(np.float32(np.power(dx, power)) + np.float32(np.power(dy, power)),
                                       np.float32(1.0) / power))
        d0, d1, d2 = _insert(dist, d0, d1, d2)
    return d0, d1, d2


@njit(cache=True, error_model="numpy")
def _cellular_evaluate(x, y, randomness, grid, power, probs, perm, g2):
    inf = np.float32(np.inf)
    d0 = inf
    d1 = inf
    d2 = inf
    ix = _as_i32(x)
    iy = _as_i32(y)
    fx = x - np.float32(ix)
    fy = y - np.float32(iy)
    one = np.float32(1.0)
    d0, d1, d2 = _check_cube(fx, fy, ix, iy, randomness, grid, power, probs, perm, g2, d0, d1, d2)
    distance = d2
    if distance > fy:
        d0, d1, d2 = _check_cube(fx, fy + one, ix, iy - 1, randomness, grid, power, probs, perm, g2, d0, d1, d2)
        distance = d2
    if distance > one - fy:
        d0, d1, d2 = _check_cube(fx, fy - one, ix, iy + 1, randomness, grid, power, probs, perm, g2, d0, d1, d2)
        distance = d2
    if distance > fx:
        d0, d1, d2 = _check_cube(fx + one, fy, ix - 1, iy, randomness, grid, power, probs, perm, g2, d0, d1, d2)
        if distance > fy:
            d0, d1, d2 = _check_cube(fx + one, fy + one, ix - 1, iy - 1, randomness, grid, power, probs, perm, g2,
                                     d0, d1, d2)
            distance = d2
        if distance > one - fy:
            d0, d1, d2 = _check_cube(fx + one, fy - one, ix - 1, iy + 1, randomness, grid, power, probs, perm, g2,
                                     d0, d1, d2)
            distance = d2
    if distance > one - fx:
        d0, d1, d2 = _check_cube(fx - one, fy, ix + 1, iy, randomness, grid, power, probs, perm, g2, d0, d1, d2)
        distance = d2
        if distance > fy:
            d0, d1, d2 = _check_cube(fx - one, fy + one, ix + 1, iy - 1, randomness, grid, power, probs, perm, g2,
                                     d0, d1, d2)
            distance = d2
        if distance > one - fy:
            d0, d1, d2 = _check_cube(fx - one, fy - one, ix + 1, iy + 1, randomness, grid, power, probs, perm, g2,
                                     d0, d1, d2)
    return d0, d1


@njit(cache=True, error_model="numpy", parallel=True)
def cellular_kernel(w, h, perm, g2, probs, color, scale, stretch, angle, turbulence, f1, f2, randomness, grid,
                    power):
    out = np.empty((h, w), np.int64)
    sc = np.float32(scale)
    stv = np.float32(stretch)
    a = np.float32(angle)
    ca = np.float32(np.cos(a))
    sa = np.float32(np.sin(a))
    turb = np.float32(turbulence)
    rnd = np.float32(randomness)
    pw = np.float32(power)
    c0 = np.float32(f1)
    c1 = np.float32(f2)
    for y in prange(h):
        yf = np.float32(y)
        for x in range(w):
            xf = np.float32(x)
            nx = (ca * xf + sa * yf) / sc + np.float32(1000.0)
            ny = (-sa * xf + ca * yf) / (sc * stv) + np.float32(1000.0)
            d0, d1 = _cellular_evaluate(nx, ny, rnd, grid, pw, probs, perm, g2)
            if turb != np.float32(1.0):
                freq = np.float32(1.0)
                while freq <= turb:
                    d0, d1 = _cellular_evaluate(freq * nx, freq * ny, rnd, grid, pw, probs, perm, g2)
                    freq *= np.float32(2.0)
            factor = (d1 - d0) / (d1 + d0)
            out[y, x] = _mix_colors(_smooth_step(c1, c0, factor), 0xFF000000, color)
    return out


@functools.lru_cache(maxsize=1)
def cellular_probabilities() -> np.ndarray:
    """Rust ``cellular_probabilities`` (Poisson(2.5) lookup table) evaluated in f32."""
    values = np.zeros(8192, np.int64)
    factorial = F32(1.0)
    total = F32(0.0)
    e = F32(np.exp(F32(-2.5)))
    for index in range(10):
        if index > 1:
            factorial = F32(factorial * F32(index))
        # llvm.powi (compiler-rt __powisf2): binary exponentiation in f32; all powers of 2.5 <= 9 are exact
        r, a, b = F32(1.0), F32(2.5), index
        while True:
            if b & 1:
                r = F32(r * a)
            b //= 2
            if b == 0:
                break
            a = F32(a * a)
        prob = F32(F32(r * e) / factorial)
        start = int(F32(total * F32(8192.0)))
        total = F32(total + prob)
        end = int(F32(total * F32(8192.0)))
        values[min(start, 8192):min(end, 8192)] = index
    return values


# ---------------------------------------------------------------------------
# QuiltFilter
# ---------------------------------------------------------------------------

@njit(cache=True, error_model="numpy", inline="always")
def _wrap(v):
    if v >= np.float32(0.0):
        return v - np.float32(np.trunc(v))
    return v - np.float32(np.trunc(v)) + np.float32(1.0)


@njit(cache=True, error_model="numpy")
def quilt_kernel(w, h, iterations, a_in, b_in, c_in, d_in, k_i):
    pi = _PI32
    a = np.float32(a_in)  # distinct names keep these float32 (numba unifies re-bound variables)
    b = np.float32(b_in)
    c = np.float32(c_in)
    d = np.float32(d_in)
    k = np.float32(k_i)
    two = np.float32(2.0)
    four = np.float32(4.0)
    six = np.float32(6.0)
    x = np.float32(0.1)
    y = np.float32(0.3)
    counts = np.zeros(w * h, np.int64)
    maximum = 0
    for it in range(20 + max(iterations, 0)):
        mx = pi * x
        my = pi * y
        nx = (a * np.sin(two * mx) + b * np.sin(two * mx) * np.cos(two * my) + c * np.sin(four * mx)
              + d * np.sin(six * mx) * np.cos(four * my) + k * x)
        ny = (a * np.sin(two * my) + b * np.sin(two * my) * np.cos(two * mx) + c * np.sin(four * my)
              + d * np.sin(six * my) * np.cos(four * mx) + k * y)
        x = _wrap(np.float32(nx))
        y = _wrap(np.float32(ny))
        if it < 20:
            continue
        ix = _as_i32(np.float32(w) * x)
        iy = _as_i32(np.float32(h) * y)
        if ix >= 0 and ix < w and iy >= 0 and iy < h:
            idx = iy * w + ix
            prev = counts[idx]
            counts[idx] = prev + 1
            if prev > maximum:
                maximum = prev
    out = np.empty((h, w), np.int64)
    mf = np.float32(maximum)
    for i in range(w * h):
        out[i // w, i % w] = _linear_colormap(np.float32(counts[i]) / mf, 0xFF000000, 0xFFFFFFFF)
    return out
