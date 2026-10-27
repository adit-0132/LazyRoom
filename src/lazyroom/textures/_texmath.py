"""Scalar helpers shared by the fast-noise texture kernels (numba).

Port of ImageToolboxLibs ``fast-noise/src/main/cpp/textures/TextureMath.h`` and
the small helpers of ``ProceduralTextures.cpp`` (Apache-2.0, T8RIN).  Colours are
ARGB ints carried as int64 (values 0..2**32-1); ``mix_color`` rounds every
channel to 8 bits exactly like the C++ ``mixColor`` (``std::lround``).
"""
from __future__ import annotations

import math

import numpy as np

from ._nb import njit

PI = 3.14159265358979323846


@njit(cache=True, inline="always")
def clamp01(value):
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@njit(cache=True, inline="always")
def clampf(value, lo, hi):
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


@njit(cache=True, inline="always")
def smoothstep(edge0, edge1, value):
    if edge0 == edge1:
        return 0.0 if value < edge0 else 1.0
    t = clamp01((value - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, inline="always")
def fract(value):
    return value - math.floor(value)


@njit(cache=True, inline="always")
def contrast(value, amount):
    return clamp01((value - 0.5) * max(amount, 0.0) + 0.5)


@njit(cache=True, inline="always")
def _mix_channel(a, b, shift, t):
    av = float((a >> shift) & 0xFF)
    bv = float((b >> shift) & 0xFF)
    v = av + (bv - av) * t
    # std::lround (half away from zero); channel values are never negative
    return np.int64(math.floor(v + 0.5)) << shift


@njit(cache=True)
def mix_color(src, dst, amount):
    """C++ ``mixColor(from, to, amount)`` on ARGB ints (int64)."""
    t = clamp01(amount)
    a = np.int64(src)
    b = np.int64(dst)
    return (_mix_channel(a, b, 24, t) | _mix_channel(a, b, 16, t) | _mix_channel(a, b, 8, t)
            | _mix_channel(a, b, 0, t))


@njit(cache=True)
def hash2(x, y, seed):
    """C++ ``textures::hash`` evaluated in float32 like upstream (sin * 43758.5453, fract)."""
    v = np.float32(np.sin(np.float32(x) * np.float32(127.1) + np.float32(y) * np.float32(311.7)
                          + np.float32(seed) * np.float32(0.0137))) * np.float32(43758.5453)
    return float(v - np.floor(v))


@njit(cache=True, inline="always")
def parameter(p, index, fallback):
    if index >= p.shape[0]:
        return fallback
    v = p[index]
    if not math.isfinite(v):
        return fallback
    return v


@njit(cache=True, inline="always")
def color(c, index, fallback):
    if index < c.shape[0]:
        return np.int64(c[index])
    return np.int64(fallback)


@njit(cache=True)
def palette4(c, value):
    t = clamp01(value) * 3.0
    if t < 1.0:
        return mix_color(color(c, 0, 0xff050611), color(c, 1, 0xff294a8a), t)
    if t < 2.0:
        return mix_color(color(c, 1, 0xff294a8a), color(c, 2, 0xffd94b91), t - 1.0)
    return mix_color(color(c, 2, 0xffd94b91), color(c, 3, 0xffffe8a3), t - 2.0)
