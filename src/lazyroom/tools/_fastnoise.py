# MIT License
#
# Copyright(c) 2023 Jordan Peck (jordan.me2@gmail.com)
# Copyright(c) 2023 Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# VERSION: 1.1.1
# https://github.com/Auburn/FastNoiseLite
"""Numba port of FastNoiseLite 1.1.1 as vendored by ImageToolboxLibs (``libs/fast-noise``).

Complete port: all noise types in 2D and 3D, fractal types (FBm, Ridged, PingPong, and the
DomainWarp fractals), all cellular distance functions / return types / jitter, 3D rotation types,
and DomainWarp (OpenSimplex2, OpenSimplex2Reduced, BasicGrid; single / progressive / independent).
Arithmetic is carried out in float32 like ``GetNoise(float, float)`` in ImageToolbox, integer hashing
reproduces C++ 32-bit wrap-around.  Note the vendored copy's 2D FBm weighting uses
``FastMin(noise + 1, 2)`` (kept).

Also ports ImageToolbox's colour mapping (``fast-noise.cpp`` / ``ColorUtils.cpp``): noise -> LAB
interpolation between two colours (black -> white in ImageToolbox) -> sRGB.

Config arrays used by the kernels:
  icfg (int64):  seed, noise_type, transform3d, fractal_type, octaves, cell_dist, cell_ret, warp_type,
                 warp_transform3d
  fcfg (float32): frequency, lacunarity, gain, weighted_strength, ping_pong_strength, fractal_bounding,
                 cellular_jitter, domain_warp_amp
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ._fastnoise_tables import GRADIENTS2D as G2D, GRADIENTS3D as G3D, RANDVECS2D as RV2, RANDVECS3D as RV3

f32 = np.float32

NOISE_TYPES = ["open_simplex2", "open_simplex2s", "cellular", "perlin", "value_cubic", "value"]
ROTATION_TYPES = ["none", "improve_xy_planes", "improve_xz_planes"]
FRACTAL_TYPES = ["none", "fbm", "ridged", "ping_pong", "domain_warp_progressive", "domain_warp_independent"]
CELLULAR_DISTANCE = ["euclidean", "euclidean_sq", "manhattan", "hybrid"]
CELLULAR_RETURN = ["cell_value", "distance", "distance2", "distance2_add", "distance2_sub", "distance2_mul",
                   "distance2_div"]
DOMAIN_WARP_TYPES = ["open_simplex2", "open_simplex2_reduced", "basic_grid"]

# TransformType3D
TT_NONE, TT_XY, TT_XZ, TT_OS2 = 0, 1, 2, 3

PRIME_X = 501125321
PRIME_Y = 1136930381
PRIME_Z = 1720413743
PX2 = 1002250642            # PrimeX << 1 (fits in int32)
PY2 = -2021106534           # PrimeY << 1 wrapped to int32
PZ2 = -854139810            # PrimeZ << 1 wrapped to int32

F0, F1, F2, F3, F6 = f32(0), f32(1), f32(2), f32(3), f32(6)
F05 = f32(0.5)
F10, F15 = f32(10), f32(15)
SQRT3 = f32(1.7320508075688772935274463415059)
G2 = (F3 - SQRT3) / F6
SKEW_F2 = F05 * (SQRT3 - F1)
C1 = F2 * (F1 - F2 * G2) * (F1 / G2 - F2)
C2 = f32(-2) * (F1 - F2 * G2) * (F1 - F2 * G2)
G2x2m1 = F2 * G2 - F1
G2m1 = G2 - F1
G2x3m2 = F3 * G2 - F2
G2x3m1 = F3 * G2 - F1
OneM2G2 = F1 - F2 * G2
OneMG2 = F1 - G2
TWO_THIRDS = F2 / F3
INV_2_31 = F1 / f32(2147483648.0)
INV_CUBIC2 = F1 / (f32(1.5) * f32(1.5))
INV_CUBIC3 = F1 / (f32(1.5) * f32(1.5) * f32(1.5))
R3 = f32(2.0 / 3.0)
S2K = -f32(0.211324865405187)
S3K = f32(0.577350269189626)


# ------------------------------------------------------------------ helpers

@njit(cache=True)
def i32(v):
    return ((v + 2147483648) & 0xFFFFFFFF) - 2147483648


@njit(cache=True)
def fast_floor(f):
    return int(f) if f >= F0 else int(f) - 1


@njit(cache=True)
def fast_round(f):
    return int(f + F05) if f >= F0 else int(f - F05)


@njit(cache=True)
def lerp(a, b, t):
    return a + t * (b - a)


@njit(cache=True)
def interp_hermite(t):
    return t * t * (F3 - F2 * t)


@njit(cache=True)
def interp_quintic(t):
    return t * t * t * (t * (t * F6 - F15) + F10)


@njit(cache=True)
def cubic_lerp(a, b, c, d, t):
    p = (d - c) - (a - b)
    return t * t * t * p + t * t * ((a - b) - p) + t * (c - a) + b


@njit(cache=True)
def ping_pong(t):
    t -= f32(int(t * F05) * 2)
    return t if t < F1 else F2 - t


@njit(cache=True)
def fmin32(a, b):
    return a if a < b else b


@njit(cache=True)
def fmax32(a, b):
    return a if a > b else b


@njit(cache=True)
def hash2(seed, xp, yp):
    return i32((seed ^ xp ^ yp) * 0x27D4EB2D)


@njit(cache=True)
def hash3(seed, xp, yp, zp):
    return i32((seed ^ xp ^ yp ^ zp) * 0x27D4EB2D)


@njit(cache=True)
def val_coord2(seed, xp, yp):
    h = hash2(seed, xp, yp)
    h = i32(h * h)
    h ^= i32(h << 19)
    return f32(h) * INV_2_31


@njit(cache=True)
def val_coord3(seed, xp, yp, zp):
    h = hash3(seed, xp, yp, zp)
    h = i32(h * h)
    h ^= i32(h << 19)
    return f32(h) * INV_2_31


@njit(cache=True)
def grad_coord2(seed, xp, yp, xd, yd):
    h = hash2(seed, xp, yp)
    h ^= h >> 15
    h &= 127 << 1
    return xd * G2D[h] + yd * G2D[h | 1]


@njit(cache=True)
def grad_coord3(seed, xp, yp, zp, xd, yd, zd):
    h = hash3(seed, xp, yp, zp)
    h ^= h >> 15
    h &= 63 << 2
    return xd * G3D[h] + yd * G3D[h | 1] + zd * G3D[h | 2]


@njit(cache=True)
def grad_out2(seed, xp, yp):
    h = hash2(seed, xp, yp) & (255 << 1)
    return RV2[h], RV2[h | 1]


@njit(cache=True)
def grad_out3(seed, xp, yp, zp):
    h = hash3(seed, xp, yp, zp) & (255 << 2)
    return RV3[h], RV3[h | 1], RV3[h | 2]


@njit(cache=True)
def grad_dual2(seed, xp, yp, xd, yd):
    h = hash2(seed, xp, yp)
    i1 = h & (127 << 1)
    i2 = (h >> 7) & (255 << 1)
    value = xd * G2D[i1] + yd * G2D[i1 | 1]
    return value * RV2[i2], value * RV2[i2 | 1]


@njit(cache=True)
def grad_dual3(seed, xp, yp, zp, xd, yd, zd):
    h = hash3(seed, xp, yp, zp)
    i1 = h & (63 << 2)
    i2 = (h >> 6) & (255 << 2)
    value = xd * G3D[i1] + yd * G3D[i1 | 1] + zd * G3D[i1 | 2]
    return value * RV3[i2], value * RV3[i2 | 1], value * RV3[i2 | 2]


def fractal_bounding(octaves: int, gain: float) -> np.float32:
    """CalculateFractalBounding()."""
    g = abs(f32(gain))
    amp = g
    af = F1
    for _ in range(1, octaves):
        af = f32(af + amp)
        amp = f32(amp * g)
    return f32(F1 / af)


# ------------------------------------------------------------------ 2D noise

@njit(cache=True)
def single_simplex2(seed, x, y):
    i = fast_floor(x)
    j = fast_floor(y)
    xi = x - f32(i)
    yi = y - f32(j)
    t = (xi + yi) * G2
    x0 = xi - t
    y0 = yi - t
    i = i32(i * PRIME_X)
    j = i32(j * PRIME_Y)
    a = F05 - x0 * x0 - y0 * y0
    n0 = F0
    if a > F0:
        n0 = (a * a) * (a * a) * grad_coord2(seed, i, j, x0, y0)
    c = C1 * t + (C2 + a)
    n2 = F0
    if c > F0:
        x2 = x0 + G2x2m1
        y2 = y0 + G2x2m1
        n2 = (c * c) * (c * c) * grad_coord2(seed, i32(i + PRIME_X), i32(j + PRIME_Y), x2, y2)
    n1 = F0
    if y0 > x0:
        x1 = x0 + G2
        y1 = y0 + G2m1
        b = F05 - x1 * x1 - y1 * y1
        if b > F0:
            n1 = (b * b) * (b * b) * grad_coord2(seed, i, i32(j + PRIME_Y), x1, y1)
    else:
        x1 = x0 + G2m1
        y1 = y0 + G2
        b = F05 - x1 * x1 - y1 * y1
        if b > F0:
            n1 = (b * b) * (b * b) * grad_coord2(seed, i32(i + PRIME_X), j, x1, y1)
    return (n0 + n1 + n2) * f32(99.83685446303647)


@njit(cache=True)
def single_os2s2(seed, x, y):
    i = fast_floor(x)
    j = fast_floor(y)
    xi = x - f32(i)
    yi = y - f32(j)
    i = i32(i * PRIME_X)
    j = i32(j * PRIME_Y)
    i1 = i32(i + PRIME_X)
    j1 = i32(j + PRIME_Y)
    t = (xi + yi) * G2
    x0 = xi - t
    y0 = yi - t
    a0 = TWO_THIRDS - x0 * x0 - y0 * y0
    value = (a0 * a0) * (a0 * a0) * grad_coord2(seed, i, j, x0, y0)
    a1 = C1 * t + (C2 + a0)
    x1 = x0 - OneM2G2
    y1 = y0 - OneM2G2
    value += (a1 * a1) * (a1 * a1) * grad_coord2(seed, i1, j1, x1, y1)
    xmyi = xi - yi
    if t > G2:
        if xi + xmyi > F1:
            x2 = x0 + G2x3m2
            y2 = y0 + G2x3m1
            a2 = TWO_THIRDS - x2 * x2 - y2 * y2
            if a2 > F0:
                value += (a2 * a2) * (a2 * a2) * grad_coord2(seed, i32(i + PX2), i32(j + PRIME_Y), x2, y2)
        else:
            x2 = x0 + G2
            y2 = y0 + G2m1
            a2 = TWO_THIRDS - x2 * x2 - y2 * y2
            if a2 > F0:
                value += (a2 * a2) * (a2 * a2) * grad_coord2(seed, i, i32(j + PRIME_Y), x2, y2)
        if yi - xmyi > F1:
            x3 = x0 + G2x3m1
            y3 = y0 + G2x3m2
            a3 = TWO_THIRDS - x3 * x3 - y3 * y3
            if a3 > F0:
                value += (a3 * a3) * (a3 * a3) * grad_coord2(seed, i32(i + PRIME_X), i32(j + PY2), x3, y3)
        else:
            x3 = x0 + G2m1
            y3 = y0 + G2
            a3 = TWO_THIRDS - x3 * x3 - y3 * y3
            if a3 > F0:
                value += (a3 * a3) * (a3 * a3) * grad_coord2(seed, i32(i + PRIME_X), j, x3, y3)
    else:
        if xi + xmyi < F0:
            x2 = x0 + OneMG2
            y2 = y0 - G2
            a2 = TWO_THIRDS - x2 * x2 - y2 * y2
            if a2 > F0:
                value += (a2 * a2) * (a2 * a2) * grad_coord2(seed, i32(i - PRIME_X), j, x2, y2)
        else:
            x2 = x0 + G2m1
            y2 = y0 + G2
            a2 = TWO_THIRDS - x2 * x2 - y2 * y2
            if a2 > F0:
                value += (a2 * a2) * (a2 * a2) * grad_coord2(seed, i32(i + PRIME_X), j, x2, y2)
        if yi < xmyi:
            x2 = x0 - G2
            y2 = y0 - G2m1
            a2 = TWO_THIRDS - x2 * x2 - y2 * y2
            if a2 > F0:
                value += (a2 * a2) * (a2 * a2) * grad_coord2(seed, i, i32(j - PRIME_Y), x2, y2)
        else:
            x2 = x0 + G2
            y2 = y0 + G2m1
            a2 = TWO_THIRDS - x2 * x2 - y2 * y2
            if a2 > F0:
                value += (a2 * a2) * (a2 * a2) * grad_coord2(seed, i, i32(j + PRIME_Y), x2, y2)
    return value * f32(18.24196194486065)


@njit(cache=True)
def cellular_return(d0, d1, closest, dist_fn, ret):
    if dist_fn == 0 and ret >= 1:
        d0 = np.sqrt(d0)
        if ret >= 2:
            d1 = np.sqrt(d1)
    if ret == 0:
        return f32(closest) * INV_2_31
    if ret == 1:
        return d0 - F1
    if ret == 2:
        return d1 - F1
    if ret == 3:
        return (d1 + d0) * F05 - F1
    if ret == 4:
        return d1 - d0 - F1
    if ret == 5:
        return d1 * d0 * F05 - F1
    if ret == 6:
        return d0 / d1 - F1
    return F0


@njit(cache=True)
def single_cellular2(seed, x, y, dist_fn, ret, jitter_mod):
    xr = fast_round(x)
    yr = fast_round(y)
    d0 = f32(1e10)
    d1 = f32(1e10)
    closest = 0
    jitter = f32(0.43701595) * jitter_mod
    xp = i32((xr - 1) * PRIME_X)
    ypb = i32((yr - 1) * PRIME_Y)
    for xi in range(xr - 1, xr + 2):
        yp = ypb
        for yi in range(yr - 1, yr + 2):
            h = hash2(seed, xp, yp)
            idx = h & (255 << 1)
            vx = (f32(xi) - x) + RV2[idx] * jitter
            vy = (f32(yi) - y) + RV2[idx | 1] * jitter
            if dist_fn == 2:
                nd = abs(vx) + abs(vy)
            elif dist_fn == 3:
                nd = (abs(vx) + abs(vy)) + (vx * vx + vy * vy)
            else:
                nd = vx * vx + vy * vy
            d1 = fmax32(fmin32(d1, nd), d0)
            if nd < d0:
                d0 = nd
                closest = h
            yp = i32(yp + PRIME_Y)
        xp = i32(xp + PRIME_X)
    return cellular_return(d0, d1, closest, dist_fn, ret)


@njit(cache=True)
def single_perlin2(seed, x, y):
    x0 = fast_floor(x)
    y0 = fast_floor(y)
    xd0 = x - f32(x0)
    yd0 = y - f32(y0)
    xd1 = xd0 - F1
    yd1 = yd0 - F1
    xs = interp_quintic(xd0)
    ys = interp_quintic(yd0)
    x0 = i32(x0 * PRIME_X)
    y0 = i32(y0 * PRIME_Y)
    x1 = i32(x0 + PRIME_X)
    y1 = i32(y0 + PRIME_Y)
    xf0 = lerp(grad_coord2(seed, x0, y0, xd0, yd0), grad_coord2(seed, x1, y0, xd1, yd0), xs)
    xf1 = lerp(grad_coord2(seed, x0, y1, xd0, yd1), grad_coord2(seed, x1, y1, xd1, yd1), xs)
    return lerp(xf0, xf1, ys) * f32(1.4247691104677813)


@njit(cache=True)
def single_value_cubic2(seed, x, y):
    x1 = fast_floor(x)
    y1 = fast_floor(y)
    xs = x - f32(x1)
    ys = y - f32(y1)
    x1 = i32(x1 * PRIME_X)
    y1 = i32(y1 * PRIME_Y)
    x0 = i32(x1 - PRIME_X)
    y0 = i32(y1 - PRIME_Y)
    x2 = i32(x1 + PRIME_X)
    y2 = i32(y1 + PRIME_Y)
    x3 = i32(x1 + PX2)
    y3 = i32(y1 + PY2)
    return cubic_lerp(
        cubic_lerp(val_coord2(seed, x0, y0), val_coord2(seed, x1, y0), val_coord2(seed, x2, y0),
                   val_coord2(seed, x3, y0), xs),
        cubic_lerp(val_coord2(seed, x0, y1), val_coord2(seed, x1, y1), val_coord2(seed, x2, y1),
                   val_coord2(seed, x3, y1), xs),
        cubic_lerp(val_coord2(seed, x0, y2), val_coord2(seed, x1, y2), val_coord2(seed, x2, y2),
                   val_coord2(seed, x3, y2), xs),
        cubic_lerp(val_coord2(seed, x0, y3), val_coord2(seed, x1, y3), val_coord2(seed, x2, y3),
                   val_coord2(seed, x3, y3), xs),
        ys) * INV_CUBIC2


@njit(cache=True)
def single_value2(seed, x, y):
    x0 = fast_floor(x)
    y0 = fast_floor(y)
    xs = interp_hermite(x - f32(x0))
    ys = interp_hermite(y - f32(y0))
    x0 = i32(x0 * PRIME_X)
    y0 = i32(y0 * PRIME_Y)
    x1 = i32(x0 + PRIME_X)
    y1 = i32(y0 + PRIME_Y)
    xf0 = lerp(val_coord2(seed, x0, y0), val_coord2(seed, x1, y0), xs)
    xf1 = lerp(val_coord2(seed, x0, y1), val_coord2(seed, x1, y1), xs)
    return lerp(xf0, xf1, ys)


@njit(cache=True)
def gen_single2(seed, x, y, ic, fc):
    nt = ic[1]
    if nt == 0:
        return single_simplex2(seed, x, y)
    if nt == 1:
        return single_os2s2(seed, x, y)
    if nt == 2:
        return single_cellular2(seed, x, y, ic[5], ic[6], fc[6])
    if nt == 3:
        return single_perlin2(seed, x, y)
    if nt == 4:
        return single_value_cubic2(seed, x, y)
    if nt == 5:
        return single_value2(seed, x, y)
    return F0


# ------------------------------------------------------------------ 3D noise

@njit(cache=True)
def single_os2_3(seed, x, y, z):
    i = fast_round(x)
    j = fast_round(y)
    k = fast_round(z)
    x0 = x - f32(i)
    y0 = y - f32(j)
    z0 = z - f32(k)
    xs = int(f32(-1.0) - x0) | 1
    ys = int(f32(-1.0) - y0) | 1
    zs = int(f32(-1.0) - z0) | 1
    ax0 = f32(xs) * -x0
    ay0 = f32(ys) * -y0
    az0 = f32(zs) * -z0
    i = i32(i * PRIME_X)
    j = i32(j * PRIME_Y)
    k = i32(k * PRIME_Z)
    value = F0
    a = (f32(0.6) - x0 * x0) - (y0 * y0 + z0 * z0)
    for lp in range(2):
        if a > F0:
            value += (a * a) * (a * a) * grad_coord3(seed, i, j, k, x0, y0, z0)
        b = a + F1
        i1, j1, k1 = i, j, k
        x1, y1, z1 = x0, y0, z0
        if ax0 >= ay0 and ax0 >= az0:
            x1 += f32(xs)
            b -= f32(xs * 2) * x1
            i1 = i32(i1 - xs * PRIME_X)
        elif ay0 > ax0 and ay0 >= az0:
            y1 += f32(ys)
            b -= f32(ys * 2) * y1
            j1 = i32(j1 - ys * PRIME_Y)
        else:
            z1 += f32(zs)
            b -= f32(zs * 2) * z1
            k1 = i32(k1 - zs * PRIME_Z)
        if b > F0:
            value += (b * b) * (b * b) * grad_coord3(seed, i1, j1, k1, x1, y1, z1)
        if lp == 1:
            break
        ax0 = F05 - ax0
        ay0 = F05 - ay0
        az0 = F05 - az0
        x0 = f32(xs) * ax0
        y0 = f32(ys) * ay0
        z0 = f32(zs) * az0
        a += (f32(0.75) - ax0) - (ay0 + az0)
        i = i32(i + ((xs >> 1) & PRIME_X))
        j = i32(j + ((ys >> 1) & PRIME_Y))
        k = i32(k + ((zs >> 1) & PRIME_Z))
        xs, ys, zs = -xs, -ys, -zs
        seed = ~seed
    return value * f32(32.69428253173828125)


@njit(cache=True)
def single_os2s_3(seed, x, y, z):
    i = fast_floor(x)
    j = fast_floor(y)
    k = fast_floor(z)
    xi = x - f32(i)
    yi = y - f32(j)
    zi = z - f32(k)
    i = i32(i * PRIME_X)
    j = i32(j * PRIME_Y)
    k = i32(k * PRIME_Z)
    seed2 = i32(seed + 1293373)
    xm = int(-F05 - xi)
    ym = int(-F05 - yi)
    zm = int(-F05 - zi)
    x0 = xi + f32(xm)
    y0 = yi + f32(ym)
    z0 = zi + f32(zm)
    a0 = f32(0.75) - x0 * x0 - y0 * y0 - z0 * z0
    value = (a0 * a0) * (a0 * a0) * grad_coord3(seed, i32(i + (xm & PRIME_X)), i32(j + (ym & PRIME_Y)),
                                                 i32(k + (zm & PRIME_Z)), x0, y0, z0)
    x1 = xi - F05
    y1 = yi - F05
    z1 = zi - F05
    a1 = f32(0.75) - x1 * x1 - y1 * y1 - z1 * z1
    value += (a1 * a1) * (a1 * a1) * grad_coord3(seed2, i32(i + PRIME_X), i32(j + PRIME_Y), i32(k + PRIME_Z),
                                                 x1, y1, z1)
    xaf0 = f32((xm | 1) << 1) * x1
    yaf0 = f32((ym | 1) << 1) * y1
    zaf0 = f32((zm | 1) << 1) * z1
    xaf1 = f32(-2 - (xm << 2)) * x1 - F1
    yaf1 = f32(-2 - (ym << 2)) * y1 - F1
    zaf1 = f32(-2 - (zm << 2)) * z1 - F1
    skip5 = False
    a2 = xaf0 + a0
    if a2 > F0:
        value += (a2 * a2) * (a2 * a2) * grad_coord3(seed, i32(i + (~xm & PRIME_X)), i32(j + (ym & PRIME_Y)),
                                                     i32(k + (zm & PRIME_Z)), x0 - f32(xm | 1), y0, z0)
    else:
        a3 = yaf0 + zaf0 + a0
        if a3 > F0:
            value += (a3 * a3) * (a3 * a3) * grad_coord3(seed, i32(i + (xm & PRIME_X)), i32(j + (~ym & PRIME_Y)),
                                                         i32(k + (~zm & PRIME_Z)), x0, y0 - f32(ym | 1),
                                                         z0 - f32(zm | 1))
        a4 = xaf1 + a1
        if a4 > F0:
            value += (a4 * a4) * (a4 * a4) * grad_coord3(seed2, i32(i + (xm & PX2)), i32(j + PRIME_Y),
                                                         i32(k + PRIME_Z), f32(xm | 1) + x1, y1, z1)
            skip5 = True
    skip9 = False
    a6 = yaf0 + a0
    if a6 > F0:
        value += (a6 * a6) * (a6 * a6) * grad_coord3(seed, i32(i + (xm & PRIME_X)), i32(j + (~ym & PRIME_Y)),
                                                     i32(k + (zm & PRIME_Z)), x0, y0 - f32(ym | 1), z0)
    else:
        a7 = xaf0 + zaf0 + a0
        if a7 > F0:
            value += (a7 * a7) * (a7 * a7) * grad_coord3(seed, i32(i + (~xm & PRIME_X)), i32(j + (ym & PRIME_Y)),
                                                         i32(k + (~zm & PRIME_Z)), x0 - f32(xm | 1), y0,
                                                         z0 - f32(zm | 1))
        a8 = yaf1 + a1
        if a8 > F0:
            value += (a8 * a8) * (a8 * a8) * grad_coord3(seed2, i32(i + PRIME_X), i32(j + (ym & PY2)),
                                                         i32(k + PRIME_Z), x1, f32(ym | 1) + y1, z1)
            skip9 = True
    skipd = False
    aa = zaf0 + a0
    if aa > F0:
        value += (aa * aa) * (aa * aa) * grad_coord3(seed, i32(i + (xm & PRIME_X)), i32(j + (ym & PRIME_Y)),
                                                     i32(k + (~zm & PRIME_Z)), x0, y0, z0 - f32(zm | 1))
    else:
        ab = xaf0 + yaf0 + a0
        if ab > F0:
            value += (ab * ab) * (ab * ab) * grad_coord3(seed, i32(i + (~xm & PRIME_X)), i32(j + (~ym & PRIME_Y)),
                                                         i32(k + (zm & PRIME_Z)), x0 - f32(xm | 1),
                                                         y0 - f32(ym | 1), z0)
        ac = zaf1 + a1
        if ac > F0:
            value += (ac * ac) * (ac * ac) * grad_coord3(seed2, i32(i + PRIME_X), i32(j + PRIME_Y),
                                                         i32(k + (zm & PZ2)), x1, y1, f32(zm | 1) + z1)
            skipd = True
    if not skip5:
        a5 = yaf1 + zaf1 + a1
        if a5 > F0:
            value += (a5 * a5) * (a5 * a5) * grad_coord3(seed2, i32(i + PRIME_X), i32(j + (ym & PY2)),
                                                         i32(k + (zm & PZ2)), x1, f32(ym | 1) + y1,
                                                         f32(zm | 1) + z1)
    if not skip9:
        a9 = xaf1 + zaf1 + a1
        if a9 > F0:
            value += (a9 * a9) * (a9 * a9) * grad_coord3(seed2, i32(i + (xm & PX2)), i32(j + PRIME_Y),
                                                         i32(k + (zm & PZ2)), f32(xm | 1) + x1, y1,
                                                         f32(zm | 1) + z1)
    if not skipd:
        ad = xaf1 + yaf1 + a1
        if ad > F0:
            value += (ad * ad) * (ad * ad) * grad_coord3(seed2, i32(i + (xm & PX2)), i32(j + (ym & PY2)),
                                                         i32(k + PRIME_Z), f32(xm | 1) + x1, f32(ym | 1) + y1, z1)
    return value * f32(9.046026385208288)


@njit(cache=True)
def single_cellular3(seed, x, y, z, dist_fn, ret, jitter_mod):
    xr = fast_round(x)
    yr = fast_round(y)
    zr = fast_round(z)
    d0 = f32(1e10)
    d1 = f32(1e10)
    closest = 0
    jitter = f32(0.39614353) * jitter_mod
    xp = i32((xr - 1) * PRIME_X)
    ypb = i32((yr - 1) * PRIME_Y)
    zpb = i32((zr - 1) * PRIME_Z)
    if 0 <= dist_fn <= 3:
        for xi in range(xr - 1, xr + 2):
            yp = ypb
            for yi in range(yr - 1, yr + 2):
                zp = zpb
                for zi in range(zr - 1, zr + 2):
                    h = hash3(seed, xp, yp, zp)
                    idx = h & (255 << 2)
                    vx = (f32(xi) - x) + RV3[idx] * jitter
                    vy = (f32(yi) - y) + RV3[idx | 1] * jitter
                    vz = (f32(zi) - z) + RV3[idx | 2] * jitter
                    if dist_fn == 2:
                        nd = abs(vx) + abs(vy) + abs(vz)
                    elif dist_fn == 3:
                        nd = (abs(vx) + abs(vy) + abs(vz)) + (vx * vx + vy * vy + vz * vz)
                    else:
                        nd = vx * vx + vy * vy + vz * vz
                    d1 = fmax32(fmin32(d1, nd), d0)
                    if nd < d0:
                        d0 = nd
                        closest = h
                    zp = i32(zp + PRIME_Z)
                yp = i32(yp + PRIME_Y)
            xp = i32(xp + PRIME_X)
    return cellular_return(d0, d1, closest, dist_fn, ret)


@njit(cache=True)
def single_perlin3(seed, x, y, z):
    x0 = fast_floor(x)
    y0 = fast_floor(y)
    z0 = fast_floor(z)
    xd0 = x - f32(x0)
    yd0 = y - f32(y0)
    zd0 = z - f32(z0)
    xd1 = xd0 - F1
    yd1 = yd0 - F1
    zd1 = zd0 - F1
    xs = interp_quintic(xd0)
    ys = interp_quintic(yd0)
    zs = interp_quintic(zd0)
    x0 = i32(x0 * PRIME_X)
    y0 = i32(y0 * PRIME_Y)
    z0 = i32(z0 * PRIME_Z)
    x1 = i32(x0 + PRIME_X)
    y1 = i32(y0 + PRIME_Y)
    z1 = i32(z0 + PRIME_Z)
    xf00 = lerp(grad_coord3(seed, x0, y0, z0, xd0, yd0, zd0), grad_coord3(seed, x1, y0, z0, xd1, yd0, zd0), xs)
    xf10 = lerp(grad_coord3(seed, x0, y1, z0, xd0, yd1, zd0), grad_coord3(seed, x1, y1, z0, xd1, yd1, zd0), xs)
    xf01 = lerp(grad_coord3(seed, x0, y0, z1, xd0, yd0, zd1), grad_coord3(seed, x1, y0, z1, xd1, yd0, zd1), xs)
    xf11 = lerp(grad_coord3(seed, x0, y1, z1, xd0, yd1, zd1), grad_coord3(seed, x1, y1, z1, xd1, yd1, zd1), xs)
    yf0 = lerp(xf00, xf10, ys)
    yf1 = lerp(xf01, xf11, ys)
    return lerp(yf0, yf1, zs) * f32(0.964921414852142333984375)


@njit(cache=True)
def _vc_row(seed, x0, x1, x2, x3, y, z, xs):
    return cubic_lerp(val_coord3(seed, x0, y, z), val_coord3(seed, x1, y, z), val_coord3(seed, x2, y, z),
                      val_coord3(seed, x3, y, z), xs)


@njit(cache=True)
def single_value_cubic3(seed, x, y, z):
    x1 = fast_floor(x)
    y1 = fast_floor(y)
    z1 = fast_floor(z)
    xs = x - f32(x1)
    ys = y - f32(y1)
    zs = z - f32(z1)
    x1 = i32(x1 * PRIME_X)
    y1 = i32(y1 * PRIME_Y)
    z1 = i32(z1 * PRIME_Z)
    x0 = i32(x1 - PRIME_X)
    y0 = i32(y1 - PRIME_Y)
    z0 = i32(z1 - PRIME_Z)
    x2 = i32(x1 + PRIME_X)
    y2 = i32(y1 + PRIME_Y)
    z2 = i32(z1 + PRIME_Z)
    x3 = i32(x1 + PX2)
    y3 = i32(y1 + PY2)
    z3 = i32(z1 + PZ2)
    zs_ = (z0, z1, z2, z3)
    vals = np.empty(4, np.float32)
    for n in range(4):
        zz = zs_[n]
        vals[n] = cubic_lerp(_vc_row(seed, x0, x1, x2, x3, y0, zz, xs), _vc_row(seed, x0, x1, x2, x3, y1, zz, xs),
                             _vc_row(seed, x0, x1, x2, x3, y2, zz, xs), _vc_row(seed, x0, x1, x2, x3, y3, zz, xs),
                             ys)
    return cubic_lerp(vals[0], vals[1], vals[2], vals[3], zs) * INV_CUBIC3


@njit(cache=True)
def single_value3(seed, x, y, z):
    x0 = fast_floor(x)
    y0 = fast_floor(y)
    z0 = fast_floor(z)
    xs = interp_hermite(x - f32(x0))
    ys = interp_hermite(y - f32(y0))
    zs = interp_hermite(z - f32(z0))
    x0 = i32(x0 * PRIME_X)
    y0 = i32(y0 * PRIME_Y)
    z0 = i32(z0 * PRIME_Z)
    x1 = i32(x0 + PRIME_X)
    y1 = i32(y0 + PRIME_Y)
    z1 = i32(z0 + PRIME_Z)
    xf00 = lerp(val_coord3(seed, x0, y0, z0), val_coord3(seed, x1, y0, z0), xs)
    xf10 = lerp(val_coord3(seed, x0, y1, z0), val_coord3(seed, x1, y1, z0), xs)
    xf01 = lerp(val_coord3(seed, x0, y0, z1), val_coord3(seed, x1, y0, z1), xs)
    xf11 = lerp(val_coord3(seed, x0, y1, z1), val_coord3(seed, x1, y1, z1), xs)
    yf0 = lerp(xf00, xf10, ys)
    yf1 = lerp(xf01, xf11, ys)
    return lerp(yf0, yf1, zs)


@njit(cache=True)
def gen_single3(seed, x, y, z, ic, fc):
    nt = ic[1]
    if nt == 0:
        return single_os2_3(seed, x, y, z)
    if nt == 1:
        return single_os2s_3(seed, x, y, z)
    if nt == 2:
        return single_cellular3(seed, x, y, z, ic[5], ic[6], fc[6])
    if nt == 3:
        return single_perlin3(seed, x, y, z)
    if nt == 4:
        return single_value_cubic3(seed, x, y, z)
    if nt == 5:
        return single_value3(seed, x, y, z)
    return F0


# ------------------------------------------------------------------ coordinate transforms

@njit(cache=True)
def transform_noise2(x, y, ic, fc):
    x *= fc[0]
    y *= fc[0]
    if ic[1] == 0 or ic[1] == 1:
        t = (x + y) * SKEW_F2
        x += t
        y += t
    return x, y


@njit(cache=True)
def transform3(x, y, z, tt):
    if tt == TT_XY:
        xy = x + y
        s2 = xy * S2K
        z *= S3K
        x += s2 - z
        y = y + s2 - z
        z += xy * S3K
    elif tt == TT_XZ:
        xz = x + z
        s2 = xz * S2K
        y *= S3K
        x += s2 - y
        z += s2 - y
        y += xz * S3K
    elif tt == TT_OS2:
        r = (x + y + z) * R3
        x = r - x
        y = r - y
        z = r - z
    return x, y, z


@njit(cache=True)
def transform_noise3(x, y, z, ic, fc):
    return transform3(x * fc[0], y * fc[0], z * fc[0], ic[2])


@njit(cache=True)
def transform_warp2(x, y, ic):
    if ic[7] == 0 or ic[7] == 1:
        t = (x + y) * SKEW_F2
        x += t
        y += t
    return x, y


# ------------------------------------------------------------------ fractals / GetNoise

@njit(cache=True)
def get_noise2(x, y, ic, fc):
    x, y = transform_noise2(x, y, ic, fc)
    ft = ic[3]
    if ft < 1 or ft > 3:
        return gen_single2(ic[0], x, y, ic, fc)
    seed = ic[0]
    total = F0
    amp = fc[5]
    for _ in range(ic[4]):
        n = gen_single2(seed, x, y, ic, fc)
        seed = i32(seed + 1)
        if ft == 1:
            total += n * amp
            amp *= lerp(F1, fmin32(n + F1, F2) * F05, fc[3])
        elif ft == 2:
            n = abs(n)
            total += (n * f32(-2) + F1) * amp
            amp *= lerp(F1, F1 - n, fc[3])
        else:
            n = ping_pong((n + F1) * fc[4])
            total += (n - F05) * F2 * amp
            amp *= lerp(F1, n, fc[3])
        x *= fc[1]
        y *= fc[1]
        amp *= fc[2]
    return total


@njit(cache=True)
def get_noise3(x, y, z, ic, fc):
    x, y, z = transform_noise3(x, y, z, ic, fc)
    ft = ic[3]
    if ft < 1 or ft > 3:
        return gen_single3(ic[0], x, y, z, ic, fc)
    seed = ic[0]
    total = F0
    amp = fc[5]
    for _ in range(ic[4]):
        n = gen_single3(seed, x, y, z, ic, fc)
        seed = i32(seed + 1)
        if ft == 1:
            total += n * amp
            amp *= lerp(F1, (n + F1) * F05, fc[3])
        elif ft == 2:
            n = abs(n)
            total += (n * f32(-2) + F1) * amp
            amp *= lerp(F1, F1 - n, fc[3])
        else:
            n = ping_pong((n + F1) * fc[4])
            total += (n - F05) * F2 * amp
            amp *= lerp(F1, n, fc[3])
        x *= fc[1]
        y *= fc[1]
        z *= fc[1]
        amp *= fc[2]
    return total


# ------------------------------------------------------------------ domain warp

@njit(cache=True)
def warp_simplex_gradient2(seed, warp_amp, freq, x, y, out_grad_only):
    x *= freq
    y *= freq
    i = fast_floor(x)
    j = fast_floor(y)
    xi = x - f32(i)
    yi = y - f32(j)
    t = (xi + yi) * G2
    x0 = xi - t
    y0 = yi - t
    i = i32(i * PRIME_X)
    j = i32(j * PRIME_Y)
    vx = F0
    vy = F0
    a = F05 - x0 * x0 - y0 * y0
    if a > F0:
        aaaa = (a * a) * (a * a)
        if out_grad_only:
            xo, yo = grad_out2(seed, i, j)
        else:
            xo, yo = grad_dual2(seed, i, j, x0, y0)
        vx += aaaa * xo
        vy += aaaa * yo
    c = C1 * t + (C2 + a)
    if c > F0:
        x2 = x0 + G2x2m1
        y2 = y0 + G2x2m1
        cccc = (c * c) * (c * c)
        if out_grad_only:
            xo, yo = grad_out2(seed, i32(i + PRIME_X), i32(j + PRIME_Y))
        else:
            xo, yo = grad_dual2(seed, i32(i + PRIME_X), i32(j + PRIME_Y), x2, y2)
        vx += cccc * xo
        vy += cccc * yo
    if y0 > x0:
        x1 = x0 + G2
        y1 = y0 + G2m1
        b = F05 - x1 * x1 - y1 * y1
        if b > F0:
            bbbb = (b * b) * (b * b)
            if out_grad_only:
                xo, yo = grad_out2(seed, i, i32(j + PRIME_Y))
            else:
                xo, yo = grad_dual2(seed, i, i32(j + PRIME_Y), x1, y1)
            vx += bbbb * xo
            vy += bbbb * yo
    else:
        x1 = x0 + G2m1
        y1 = y0 + G2
        b = F05 - x1 * x1 - y1 * y1
        if b > F0:
            bbbb = (b * b) * (b * b)
            if out_grad_only:
                xo, yo = grad_out2(seed, i32(i + PRIME_X), j)
            else:
                xo, yo = grad_dual2(seed, i32(i + PRIME_X), j, x1, y1)
            vx += bbbb * xo
            vy += bbbb * yo
    return vx * warp_amp, vy * warp_amp


@njit(cache=True)
def warp_basic_grid2(seed, warp_amp, freq, x, y):
    xf = x * freq
    yf = y * freq
    x0 = fast_floor(xf)
    y0 = fast_floor(yf)
    xs = interp_hermite(xf - f32(x0))
    ys = interp_hermite(yf - f32(y0))
    x0 = i32(x0 * PRIME_X)
    y0 = i32(y0 * PRIME_Y)
    x1 = i32(x0 + PRIME_X)
    y1 = i32(y0 + PRIME_Y)
    h0 = hash2(seed, x0, y0) & (255 << 1)
    h1 = hash2(seed, x1, y0) & (255 << 1)
    lx0x = lerp(RV2[h0], RV2[h1], xs)
    ly0x = lerp(RV2[h0 | 1], RV2[h1 | 1], xs)
    h0 = hash2(seed, x0, y1) & (255 << 1)
    h1 = hash2(seed, x1, y1) & (255 << 1)
    lx1x = lerp(RV2[h0], RV2[h1], xs)
    ly1x = lerp(RV2[h0 | 1], RV2[h1 | 1], xs)
    return lerp(lx0x, lx1x, ys) * warp_amp, lerp(ly0x, ly1x, ys) * warp_amp


@njit(cache=True)
def do_single_warp2(seed, amp, freq, x, y, ic):
    wt = ic[7]
    if wt == 0:
        return warp_simplex_gradient2(seed, amp * f32(38.283687591552734375), freq, x, y, False)
    if wt == 1:
        return warp_simplex_gradient2(seed, amp * f32(16.0), freq, x, y, True)
    return warp_basic_grid2(seed, amp, freq, x, y)


@njit(cache=True)
def domain_warp2(x, y, ic, fc):
    ft = ic[3]
    seed = ic[0]
    amp = fc[7] * fc[5]
    freq = fc[0]
    if ft == 5:  # independent
        xs, ys = transform_warp2(x, y, ic)
        for _ in range(ic[4]):
            dx, dy = do_single_warp2(seed, amp, freq, xs, ys, ic)
            x += dx
            y += dy
            seed = i32(seed + 1)
            amp *= fc[2]
            freq *= fc[1]
        return x, y
    octaves = ic[4] if ft == 4 else 1
    for _ in range(octaves):
        xs, ys = transform_warp2(x, y, ic)
        dx, dy = do_single_warp2(seed, amp, freq, xs, ys, ic)
        x += dx
        y += dy
        seed = i32(seed + 1)
        amp *= fc[2]
        freq *= fc[1]
    return x, y


@njit(cache=True)
def warp_os2_gradient3(seed, warp_amp, freq, x, y, z, out_grad_only):
    x *= freq
    y *= freq
    z *= freq
    i = fast_round(x)
    j = fast_round(y)
    k = fast_round(z)
    x0 = x - f32(i)
    y0 = y - f32(j)
    z0 = z - f32(k)
    xs = int(-x0 - F1) | 1
    ys = int(-y0 - F1) | 1
    zs = int(-z0 - F1) | 1
    ax0 = f32(xs) * -x0
    ay0 = f32(ys) * -y0
    az0 = f32(zs) * -z0
    i = i32(i * PRIME_X)
    j = i32(j * PRIME_Y)
    k = i32(k * PRIME_Z)
    vx = F0
    vy = F0
    vz = F0
    a = (f32(0.6) - x0 * x0) - (y0 * y0 + z0 * z0)
    for lp in range(2):
        if a > F0:
            aaaa = (a * a) * (a * a)
            if out_grad_only:
                xo, yo, zo = grad_out3(seed, i, j, k)
            else:
                xo, yo, zo = grad_dual3(seed, i, j, k, x0, y0, z0)
            vx += aaaa * xo
            vy += aaaa * yo
            vz += aaaa * zo
        b = a + F1
        i1, j1, k1 = i, j, k
        x1, y1, z1 = x0, y0, z0
        if ax0 >= ay0 and ax0 >= az0:
            x1 += f32(xs)
            b -= f32(xs * 2) * x1
            i1 = i32(i1 - xs * PRIME_X)
        elif ay0 > ax0 and ay0 >= az0:
            y1 += f32(ys)
            b -= f32(ys * 2) * y1
            j1 = i32(j1 - ys * PRIME_Y)
        else:
            z1 += f32(zs)
            b -= f32(zs * 2) * z1
            k1 = i32(k1 - zs * PRIME_Z)
        if b > F0:
            bbbb = (b * b) * (b * b)
            if out_grad_only:
                xo, yo, zo = grad_out3(seed, i1, j1, k1)
            else:
                xo, yo, zo = grad_dual3(seed, i1, j1, k1, x1, y1, z1)
            vx += bbbb * xo
            vy += bbbb * yo
            vz += bbbb * zo
        if lp == 1:
            break
        ax0 = F05 - ax0
        ay0 = F05 - ay0
        az0 = F05 - az0
        x0 = f32(xs) * ax0
        y0 = f32(ys) * ay0
        z0 = f32(zs) * az0
        a += (f32(0.75) - ax0) - (ay0 + az0)
        i = i32(i + ((xs >> 1) & PRIME_X))
        j = i32(j + ((ys >> 1) & PRIME_Y))
        k = i32(k + ((zs >> 1) & PRIME_Z))
        xs, ys, zs = -xs, -ys, -zs
        seed = i32(seed + 1293373)
    return vx * warp_amp, vy * warp_amp, vz * warp_amp


@njit(cache=True)
def warp_basic_grid3(seed, warp_amp, freq, x, y, z):
    xf = x * freq
    yf = y * freq
    zf = z * freq
    x0 = fast_floor(xf)
    y0 = fast_floor(yf)
    z0 = fast_floor(zf)
    xs = interp_hermite(xf - f32(x0))
    ys = interp_hermite(yf - f32(y0))
    zs = interp_hermite(zf - f32(z0))
    x0 = i32(x0 * PRIME_X)
    y0 = i32(y0 * PRIME_Y)
    z0 = i32(z0 * PRIME_Z)
    x1 = i32(x0 + PRIME_X)
    y1 = i32(y0 + PRIME_Y)
    z1 = i32(z0 + PRIME_Z)
    h0 = hash3(seed, x0, y0, z0) & (255 << 2)
    h1 = hash3(seed, x1, y0, z0) & (255 << 2)
    lx0x = lerp(RV3[h0], RV3[h1], xs)
    ly0x = lerp(RV3[h0 | 1], RV3[h1 | 1], xs)
    lz0x = lerp(RV3[h0 | 2], RV3[h1 | 2], xs)
    h0 = hash3(seed, x0, y1, z0) & (255 << 2)
    h1 = hash3(seed, x1, y1, z0) & (255 << 2)
    lx1x = lerp(RV3[h0], RV3[h1], xs)
    ly1x = lerp(RV3[h0 | 1], RV3[h1 | 1], xs)
    lz1x = lerp(RV3[h0 | 2], RV3[h1 | 2], xs)
    lx0y = lerp(lx0x, lx1x, ys)
    ly0y = lerp(ly0x, ly1x, ys)
    lz0y = lerp(lz0x, lz1x, ys)
    h0 = hash3(seed, x0, y0, z1) & (255 << 2)
    h1 = hash3(seed, x1, y0, z1) & (255 << 2)
    lx0x = lerp(RV3[h0], RV3[h1], xs)
    ly0x = lerp(RV3[h0 | 1], RV3[h1 | 1], xs)
    lz0x = lerp(RV3[h0 | 2], RV3[h1 | 2], xs)
    h0 = hash3(seed, x0, y1, z1) & (255 << 2)
    h1 = hash3(seed, x1, y1, z1) & (255 << 2)
    lx1x = lerp(RV3[h0], RV3[h1], xs)
    ly1x = lerp(RV3[h0 | 1], RV3[h1 | 1], xs)
    lz1x = lerp(RV3[h0 | 2], RV3[h1 | 2], xs)
    return (lerp(lx0y, lerp(lx0x, lx1x, ys), zs) * warp_amp, lerp(ly0y, lerp(ly0x, ly1x, ys), zs) * warp_amp,
            lerp(lz0y, lerp(lz0x, lz1x, ys), zs) * warp_amp)


@njit(cache=True)
def do_single_warp3(seed, amp, freq, x, y, z, ic):
    wt = ic[7]
    if wt == 0:
        return warp_os2_gradient3(seed, amp * f32(32.69428253173828125), freq, x, y, z, False)
    if wt == 1:
        return warp_os2_gradient3(seed, amp * f32(7.71604938271605), freq, x, y, z, True)
    return warp_basic_grid3(seed, amp, freq, x, y, z)


@njit(cache=True)
def domain_warp3(x, y, z, ic, fc):
    ft = ic[3]
    seed = ic[0]
    amp = fc[7] * fc[5]
    freq = fc[0]
    if ft == 5:
        xs, ys, zs = transform3(x, y, z, ic[8])
        for _ in range(ic[4]):
            dx, dy, dz = do_single_warp3(seed, amp, freq, xs, ys, zs, ic)
            x += dx
            y += dy
            z += dz
            seed = i32(seed + 1)
            amp *= fc[2]
            freq *= fc[1]
        return x, y, z
    octaves = ic[4] if ft == 4 else 1
    for _ in range(octaves):
        xs, ys, zs = transform3(x, y, z, ic[8])
        dx, dy, dz = do_single_warp3(seed, amp, freq, xs, ys, zs, ic)
        x += dx
        y += dy
        z += dz
        seed = i32(seed + 1)
        amp *= fc[2]
        freq *= fc[1]
    return x, y, z


# ------------------------------------------------------------------ image kernel + colour mapping

@njit(parallel=True, cache=True)
def noise_grid(out, ic, fc, use_3d, z, warp):
    """ImageToolbox loop: ``GetNoise((float) x, (float) y)`` for every pixel (optionally 3D / warped)."""
    h, w = out.shape
    for row in prange(h):
        for col in range(w):
            x = f32(col)
            y = f32(row)
            if use_3d:
                zz = z
                if warp:
                    x, y, zz = domain_warp3(x, y, zz, ic, fc)
                out[row, col] = get_noise3(x, y, zz, ic, fc)
            else:
                if warp:
                    x, y = domain_warp2(x, y, ic, fc)
                out[row, col] = get_noise2(x, y, ic, fc)


def make_config(seed, frequency, noise_type, rotation_type, fractal_type, octaves, lacunarity, gain,
                weighted_strength, ping_pong_strength, cellular_distance, cellular_return, cellular_jitter,
                domain_warp_type, domain_warp_amp):
    nt = NOISE_TYPES.index(noise_type)
    rt = ROTATION_TYPES.index(rotation_type)
    wt = DOMAIN_WARP_TYPES.index(domain_warp_type)
    if rt == 1:
        tt = wtt = TT_XY
    elif rt == 2:
        tt = wtt = TT_XZ
    else:
        tt = TT_OS2 if nt in (0, 1) else TT_NONE
        wtt = TT_OS2 if wt in (0, 1) else TT_NONE
    ic = np.array([int(seed), nt, tt, FRACTAL_TYPES.index(fractal_type), int(octaves),
                   CELLULAR_DISTANCE.index(cellular_distance), CELLULAR_RETURN.index(cellular_return), wt, wtt],
                  np.int64)
    fc = np.array([frequency, lacunarity, gain, weighted_strength, ping_pong_strength,
                   fractal_bounding(int(octaves), gain), cellular_jitter, domain_warp_amp], np.float32)
    return ic, fc


# ColorUtils.cpp (float internals) -----------------------------------------

def _srgb_to_linear(v: float) -> float:
    """``SRGBToLinear(float)`` (note the upstream 0.045 threshold) -> float value."""
    v = float(f32(v))
    return float(f32(v / 12.92)) if v <= 0.045 else float(f32(math.pow((v + 0.055) / 1.055, 2.4)))


def color_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    """``colorToLAB`` with the same float/double roundings as ColorUtils.cpp."""
    R = float(f32(_srgb_to_linear(r / 255.0) * 100.0))
    G = float(f32(_srgb_to_linear(g / 255.0) * 100.0))
    B = float(f32(_srgb_to_linear(b / 255.0) * 100.0))
    X = float(f32(0.4124564 * R + 0.3575761 * G + 0.1804375 * B))
    Y = float(f32(0.2126729 * R + 0.7151522 * G + 0.0721750 * B))
    Z = float(f32(0.0193339 * R + 0.1191920 * G + 0.9503041 * B))
    t0 = f32(6.0 / 29.0)
    t0_cubed = float(t0 * t0 * t0)
    t0_inv = float(f32(1.0) / t0)

    def fn(t: float) -> float:
        if t > t0_cubed:
            return float(np.cbrt(t))
        return (1.0 / 3.0) * t0_inv * t0_inv * t + (4.0 / 29.0)

    fx, fy, fz = f32(fn(X / 95.047)), f32(fn(Y / 100.0)), f32(fn(Z / 108.883))
    return 116.0 * float(fy) - 16.0, 500.0 * float(fx - fy), 200.0 * float(fy - fz)


@njit(cache=True)
def _f_inv(t):
    t0 = f32(6.0 / 29.0)
    if t > t0:
        return t * t * t
    return f32(3.0) * t0 * t0 * (t - f32(4.0) / f32(29.0))


@njit(cache=True)
def _lin_to_srgb(v):
    if v <= f32(0.0031308):
        return f32(12.92) * v
    return f32(float(f32(1.055)) * math.pow(float(v), float(f32(1.0) / f32(2.4))) - float(f32(0.055)))


@njit(cache=True)
def _round_half_away(x):
    t = math.trunc(x)
    if abs(x - t) >= 0.5:
        t += 1.0 if x > 0 else -1.0
    return t


@njit(parallel=True, cache=True)
def noise_to_rgba(noise, lab_a, lab_b, alpha_a, alpha_b, out):
    """fast-noise.cpp: fraction = (n + 1) / 2, LAB lerp (double), ``labToColor`` (float) -> 8-bit."""
    h, w = noise.shape
    for row in prange(h):
        for col in range(w):
            frac = (noise[row, col] + F1) / F2
            fd = float(frac)
            ll = (1.0 - fd) * lab_a[0] + fd * lab_b[0]
            la = (1.0 - fd) * lab_a[1] + fd * lab_b[1]
            lb = (1.0 - fd) * lab_a[2] + fd * lab_b[2]
            fy = f32((ll + 16.0) / 116.0)
            fx = f32(la / 500.0 + float(fy))
            fz = f32(float(fy) - lb / 200.0)
            X = f32(float(_f_inv(fx)) * 0.95047)
            Y = _f_inv(fy)
            Z = f32(float(_f_inv(fz)) * 1.08883)
            R = f32(3.2404542 * float(X) - 1.5371385 * float(Y) - 0.4985314 * float(Z))
            G = f32(-0.9692660 * float(X) + 1.8760108 * float(Y) + 0.0415560 * float(Z))
            B = f32(0.0556434 * float(X) - 0.2040259 * float(Y) + 1.0572252 * float(Z))
            for c, v in ((0, R), (1, G), (2, B)):
                s = _lin_to_srgb(v)
                s = f32(max(f32(0.0), min(f32(1.0), s)))
                out[row, col, c] = _round_half_away(float(s * f32(255.0)))
            out[row, col, 3] = _round_half_away((1.0 - fd) * alpha_a + fd * alpha_b)
