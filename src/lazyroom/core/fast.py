"""Numba-accelerated (multi-threaded, fused) colour kernels.

These back the public functions in :mod:`lazyroom.core.color` for float32
arrays; results match the numpy reference implementations to ~1e-6.
JIT-compiled code is cached on disk (``NUMBA_CACHE_DIR`` defaults to the
application cache) so only the very first run pays compilation time.
"""
from __future__ import annotations

import math
import os

import numpy as np

if "NUMBA_CACHE_DIR" not in os.environ:
    try:
        from .paths import cache_dir

        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir() / "numba")
    except Exception:  # pragma: no cover
        pass

import numba as nb  # noqa: E402

F32 = np.float32


@nb.njit(cache=True, fastmath=False, inline="always")
def _dec(v):
    a = abs(v)
    if a <= 0.04045:
        r = a / 12.92
    else:
        r = ((a + 0.055) / 1.055) ** 2.4
    return r if v >= 0 else -r


@nb.njit(cache=True, fastmath=False, inline="always")
def _enc(v):
    a = abs(v)
    if a <= 0.0031308:
        r = a * 12.92
    else:
        r = 1.055 * a ** (1.0 / 2.4) - 0.055
    return r if v >= 0 else -r


@nb.njit(cache=True, parallel=True)
def _srgb_to_linear(x):
    out = np.empty_like(x)
    for i in nb.prange(x.size):
        out[i] = _dec(x[i])
    return out


@nb.njit(cache=True, parallel=True)
def _linear_to_srgb(x):
    out = np.empty_like(x)
    for i in nb.prange(x.size):
        out[i] = _enc(x[i])
    return out


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=F32)
    return _srgb_to_linear(a.reshape(-1)).reshape(a.shape)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=F32)
    return _linear_to_srgb(a.reshape(-1)).reshape(a.shape)


@nb.njit(cache=True, parallel=True)
def _matrix_in_linear(px, m, offset, gain):
    """encode(M @ (decode(rgb) * gain) + offset) per pixel; px is (N, 3)."""
    n = px.shape[0]
    out = np.empty_like(px)
    for i in nb.prange(n):
        r = _dec(px[i, 0]) * gain
        g = _dec(px[i, 1]) * gain
        b = _dec(px[i, 2]) * gain
        out[i, 0] = _enc(m[0, 0] * r + m[0, 1] * g + m[0, 2] * b + offset)
        out[i, 1] = _enc(m[1, 0] * r + m[1, 1] * g + m[1, 2] * b + offset)
        out[i, 2] = _enc(m[2, 0] * r + m[2, 1] * g + m[2, 2] * b + offset)
    return out


def matrix_in_linear(rgb: np.ndarray, m: np.ndarray, offset: float = 0.0, gain: float = 1.0) -> np.ndarray:
    a = np.ascontiguousarray(rgb[..., :3], dtype=F32)
    out = _matrix_in_linear(a.reshape(-1, 3), np.ascontiguousarray(m, dtype=F32), F32(offset), F32(gain))
    return out.reshape(a.shape)


# OKLab ------------------------------------------------------------------------
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], F32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], F32)
_M1I = np.linalg.inv(_M1.astype(np.float64)).astype(F32)
_M2I = np.linalg.inv(_M2.astype(np.float64)).astype(F32)


@nb.njit(cache=True, inline="always")
def _cbrt(v):
    return math.copysign(abs(v) ** (1.0 / 3.0), v)


@nb.njit(cache=True, parallel=True)
def _rgb_to_oklab(px, m1, m2):
    n = px.shape[0]
    out = np.empty_like(px)
    for i in nb.prange(n):
        r = _dec(px[i, 0])
        g = _dec(px[i, 1])
        b = _dec(px[i, 2])
        l_ = _cbrt(m1[0, 0] * r + m1[0, 1] * g + m1[0, 2] * b)
        m_ = _cbrt(m1[1, 0] * r + m1[1, 1] * g + m1[1, 2] * b)
        s_ = _cbrt(m1[2, 0] * r + m1[2, 1] * g + m1[2, 2] * b)
        out[i, 0] = m2[0, 0] * l_ + m2[0, 1] * m_ + m2[0, 2] * s_
        out[i, 1] = m2[1, 0] * l_ + m2[1, 1] * m_ + m2[1, 2] * s_
        out[i, 2] = m2[2, 0] * l_ + m2[2, 1] * m_ + m2[2, 2] * s_
    return out


@nb.njit(cache=True, parallel=True)
def _oklab_to_rgb(lab, m1i, m2i):
    n = lab.shape[0]
    out = np.empty_like(lab)
    for i in nb.prange(n):
        L = lab[i, 0]
        a = lab[i, 1]
        b = lab[i, 2]
        l_ = m2i[0, 0] * L + m2i[0, 1] * a + m2i[0, 2] * b
        m_ = m2i[1, 0] * L + m2i[1, 1] * a + m2i[1, 2] * b
        s_ = m2i[2, 0] * L + m2i[2, 1] * a + m2i[2, 2] * b
        l3 = l_ * l_ * l_
        m3 = m_ * m_ * m_
        s3 = s_ * s_ * s_
        out[i, 0] = _enc(m1i[0, 0] * l3 + m1i[0, 1] * m3 + m1i[0, 2] * s3)
        out[i, 1] = _enc(m1i[1, 0] * l3 + m1i[1, 1] * m3 + m1i[1, 2] * s3)
        out[i, 2] = _enc(m1i[2, 0] * l3 + m1i[2, 1] * m3 + m1i[2, 2] * s3)
    return out


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(rgb[..., :3], dtype=F32)
    return _rgb_to_oklab(a.reshape(-1, 3), _M1, _M2).reshape(a.shape)


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(lab, dtype=F32)
    return _oklab_to_rgb(a.reshape(-1, 3), _M1I, _M2I).reshape(a.shape)


@nb.njit(cache=True, parallel=True)
def _rgb_to_hsv(px):
    n = px.shape[0]
    out = np.empty_like(px)
    for i in nb.prange(n):
        r = px[i, 0]
        g = px[i, 1]
        b = px[i, 2]
        mx = max(r, g, b)
        mn = min(r, g, b)
        d = mx - mn
        h = 0.0
        if d > 0:
            if mx == r:
                h = ((g - b) / d) % 6.0
            elif mx == g:
                h = (b - r) / d + 2.0
            else:
                h = (r - g) / d + 4.0
            h = h / 6.0
            h = h - math.floor(h)
        out[i, 0] = h
        out[i, 1] = d / mx if mx != 0 else 0.0
        out[i, 2] = mx
    return out


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(rgb[..., :3], dtype=F32)
    return _rgb_to_hsv(a.reshape(-1, 3)).reshape(a.shape)


@nb.njit(cache=True, parallel=True)
def _scale_chroma(lab, factor):
    n = lab.shape[0]
    for i in nb.prange(n):
        lab[i, 1] *= factor[i]
        lab[i, 2] *= factor[i]
    return lab


def warmup() -> None:
    """Compile/load every kernel once (called lazily; cheap after the first run)."""
    t = np.random.default_rng(0).random((8, 8, 3), dtype=np.float32)
    oklab_to_rgb(rgb_to_oklab(t))
    linear_to_srgb(srgb_to_linear(t))
    matrix_in_linear(t, np.eye(3, dtype=F32))
    rgb_to_hsv(t)
