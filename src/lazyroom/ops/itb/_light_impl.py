"""Private helpers for the ImageToolbox "Light" group port (see ``light.py``).

Ported from (all permissive licences):
  * aire (MIT, awxkee): ``cpp/color/ConvolveToneMapper.cpp`` + ``cpp/color/tone/*.hpp``
    (tone mappers), ``cpp/color/eotf-inl.h`` (sRGB transfer), ``cpp/color/Adjustments.cpp``.
  * histogram_equalization 0.2.4 (BSD-3-Clause OR Apache-2.0, awxkee) used by aire-filters for every
    CLAHE / adaptive-equalisation filter, with the colour conversions of colorutils-rs 0.7.5
    and the 8-bit YCgCo conversion of the ``yuv`` crate 0.8.6 (same licences).
  * Trickle (Apache-2.0): ``ProcessingFilters.cpp`` ``bloomImpl``.
  * ImageToolboxLibs jhlabs (Apache-2.0): the native Rust implementation
    ``libs/jhlabs/src/main/rust/jhlabs/src/lib.rs`` (Flare, Sparkle, Rays, Glow, Light, Levels).
  * ImageToolbox procedural GLSL shaders (``ProceduralShaders.kt``): BokehLights, Fireworks.
"""
from __future__ import annotations

import functools
import math

import cv2
import numpy as np
from numba import njit, prange

F32 = np.float32


def _nan0(x: np.ndarray) -> np.ndarray:
    """NaN -> 0 (what the 8-bit float->int conversion of the native code produces), inf kept for clamping."""
    return np.nan_to_num(x, nan=0.0, posinf=np.inf, neginf=-np.inf).astype(F32, copy=False)


# ===========================================================================
# aire tone mapping (ConvolveToneMapper: sRGB -> linear -> mapper -> sRGB)
# ===========================================================================

def aire_srgb_to_linear(v: np.ndarray) -> np.ndarray:
    """aire ``SRGBToLinear`` (note: threshold 0.045, not 0.04045)."""
    v = np.clip(v, 0.0, 1.0).astype(F32)
    return np.where(v <= 0.045, v / F32(12.92), np.power((v + F32(0.055)) / F32(1.055), F32(2.4))).astype(F32)


def aire_linear_to_srgb(v: np.ndarray) -> np.ndarray:
    """aire ``LinearSRGBTosRGB``; values <= 0.0031308 (incl. negatives) use the linear segment."""
    with np.errstate(invalid="ignore", over="ignore"):
        hi = F32(1.055) * np.power(np.maximum(v, 0), F32(1 / 2.4)) - F32(0.055)
        out = np.where(v <= F32(0.0031308), F32(12.92) * v, hi)
        out = np.where(np.isnan(v), np.nan, out)
    return out.astype(F32)


_LUT_N = 65536
LUT_GRID = np.linspace(0.0, 1.0, _LUT_N, dtype=F32)


@njit(cache=True)
def _lut_at(lut, v):
    """Nearest entry of a table sampled on [0, 1] (input clamped, NaN -> 0)."""
    if not (v > 0.0):
        return lut[0]
    return lut[int(min(v, 1.0) * (lut.shape[0] - 1) + 0.5)]


@njit(cache=True, parallel=True)
def apply_lut(img, lut):  # pragma: no cover - numba
    """Per-channel 1D table lookup on an (H, W, C) float32 image."""
    h, w, c = img.shape
    out = np.empty((h, w, c), np.float32)
    for y in prange(h):
        for x in range(w):
            for k in range(c):
                out[y, x, k] = _lut_at(lut, img[y, x, k])
    return out


@njit(cache=True, parallel=True)
def _apply_luma_scale(img, dec, scale, enc):  # pragma: no cover - numba
    h, w = img.shape[0], img.shape[1]
    out = np.empty((h, w, 3), np.float32)
    for y in prange(h):
        for x in range(w):
            r = _lut_at(dec, img[y, x, 0])
            g = _lut_at(dec, img[y, x, 1])
            b = _lut_at(dec, img[y, x, 2])
            s = _lut_at(scale, np.float32(0.299) * r + np.float32(0.587) * g + np.float32(0.114) * b)
            out[y, x, 0] = _lut_at(enc, r * s)
            out[y, x, 1] = _lut_at(enc, g * s)
            out[y, x, 2] = _lut_at(enc, b * s)
    return out


@functools.lru_cache(maxsize=2)
def _aire_codec_luts() -> tuple[np.ndarray, np.ndarray]:
    return aire_srgb_to_linear(LUT_GRID), np.clip(_nan0(aire_linear_to_srgb(LUT_GRID)), 0, 1)


def aire_tone_map(img: np.ndarray, mapper) -> np.ndarray:
    """Per-channel mappers: the whole decode -> curve -> encode chain is pointwise, so it is
    evaluated once on a 65536-entry table (upstream works on 8-bit values)."""
    dec, _ = _aire_codec_luts()
    with np.errstate(all="ignore"):
        lut = np.clip(_nan0(aire_linear_to_srgb(mapper(dec).astype(F32))), 0.0, 1.0)
    return apply_lut(np.ascontiguousarray(img[..., :3], dtype=F32), lut)


def aire_tone_map_luma(img: np.ndarray, scale_fn) -> np.ndarray:
    """Luma-driven mappers (``rgb * s(luma)``): ``scale_fn`` maps the Rec.601 luma of the
    linear input to the per-pixel factor; NaN factors give black like the native float->u8 cast."""
    dec, enc = _aire_codec_luts()
    with np.errstate(all="ignore"):
        s = scale_fn(LUT_GRID).astype(np.float64)
    s = np.nan_to_num(s, nan=0.0, posinf=1e30, neginf=-1e30).astype(F32)
    return _apply_luma_scale(np.ascontiguousarray(img[..., :3], dtype=F32), dec, s, enc)


def tm_aces_filmic(exposure: float):
    e = F32(exposure)

    def f(x):
        x = x * e
        return np.clip((x * (F32(2.51) * x + F32(0.03))) / (x * (F32(2.43) * x + F32(0.59)) + F32(0.14)), 0, 1)
    return f


def tm_hable(exposure: float):
    e = F32(exposure)
    A, B, C, D, E, Fh = (F32(v) for v in (0.15, 0.50, 0.10, 0.20, 0.02, 0.30))

    def f(x):
        x = x * e
        return (x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * Fh) - E / Fh
    return f


def tm_hejl_burgess(exposure: float):
    """SIMD path of aire's HejlBurgessToneMapper (offset 0.0004; scalar tail uses 0.004)."""
    e = F32(exposure)

    def f(v):
        x = np.maximum(F32(0), v * e - F32(0.0004))
        cout = (x * (F32(6.2) * x + F32(0.5))) / (x * (F32(6.2) * x + F32(1.7)) + F32(0.06))
        return np.power(cout, F32(2.4))
    return f


def tm_aldridge(exposure: float, cutoff: float):
    e, c = F32(exposure), F32(cutoff)

    def f(v):
        cin = v * e
        tmp = F32(2) * c
        x = (tmp - cin) * np.clip(tmp - cin, 0, 1) * (F32(0.25) / c) + cin - c
        cout = (x * (F32(6.2) * x + F32(0.5))) / (x * (F32(6.2) * x + F32(1.7)) + F32(0.06))
        return np.power(cout, F32(2.4))
    return f


def tm_uchimura(exposure: float):
    e = F32(exposure)
    P, a, m, l, c, b = F32(1), F32(1), F32(0.22), F32(0.4), F32(1.33), F32(0)
    l0 = (P - m) * l / a
    S0 = m + l0
    S1 = m + a * l0
    C2 = a * P / (P - S1)
    CP = -C2 / P

    def f(v):
        cin = v * e
        t = np.clip(cin / m, 0, 1)                       # smoothstep(0, m, cin)
        w0 = F32(1) - t * t * (F32(3) - F32(2) * t)
        w2 = (cin >= S0).astype(F32)                     # step(m + l0, cin)
        w1 = F32(1) - w0 - w2
        T = m * np.power(cin / m, c) + b
        L = (cin - m) * a + m
        S = P - (P - S1) * np.exp(CP * (cin - S0))
        return T * w0 + L * w1 + S * w2
    return f


def tm_mobius(exposure: float, transition: float, peak: float):
    """SIMD path of aire's MobiusToneMapper (its ``twos`` constant is 1.0, unlike the scalar tail)."""
    e, j, pk = F32(exposure), F32(transition), F32(peak)
    one = F32(1)
    with np.errstate(all="ignore"):
        a = (-j * j * (pk - one)) / (j * j - one * j + pk)
        b = (j * j - one * j * pk + pk) / max(pk - one, F32(1e-6))
        k = (j * j + (b * b + one * b * j)) / (b - a)

    def f(v):
        x = v * e
        return np.where(x <= j, x, k * (x + a) / (x + b))
    return f


def tm_logarithmic(exposure: float):
    e = F32(exposure)
    with np.errstate(divide="ignore"):
        den = np.float32(1.0) / np.log(np.float32(1.0 + 1.0 * exposure))

    def f(luma):  # per-pixel factor applied to the (unexposed) linear rgb
        lin = luma * e
        lout = np.log(np.abs(lin + F32(1))) * den
        lin = np.where(lin == 0, F32(1), lin)
        return e * (lout / lin)
    return f


def tm_drago(exposure: float, sdr_white_point: float):
    """SIMD path of aire's DragoToneMapper (Lwa = 1, bias = 0.85)."""
    e = F32(exposure)
    max_ld = F32(sdr_white_point)
    bias = F32(0.85)
    lwa_p = F32(1.0) / F32(math.pow(1.0 + bias - 0.85, 5.0))
    lmax_p = max_ld / lwa_p
    exponent = F32(math.log(bias) / math.log(0.5))
    with np.errstate(divide="ignore", invalid="ignore"):
        c1 = (F32(0.01) * max_ld) / np.log10(F32(1) + lmax_p)

    def f(luma):  # per-pixel factor applied to the (unexposed) linear rgb
        lin = luma * e
        lin_p = lin / lwa_p
        c2 = np.log(F32(1) + lin_p) / np.log(F32(8) * np.power(lin_p / lmax_p, exponent) + F32(2))
        lin = np.where(lin == 0, F32(1), lin)
        return e * (c1 * c2 / lin)
    return f


# ===========================================================================
# colorutils-rs 0.7.5 colour conversions (per pixel, numba) + histogram_equalization 0.2.4
# ===========================================================================

_CU_A = 0.0550107189475866
_CU_LIN_T = 0.0030412825601275209
_D65 = (95.047, 100.0, 108.883)
_LUV_U = 4.0 * _D65[1] / (_D65[0] + 15.0 * _D65[1] + 3.0 * _D65[2])
_LUV_V = 9.0 * _D65[1] / (_D65[0] + 15.0 * _D65[1] + 3.0 * _D65[2])
_JZ_D0 = 1.6295499532821566e-11
_JZ_NITS = 200.0

SPACES = {"ycgco": 0, "hsv": 1, "hsl": 2, "lab": 3, "luv": 4, "oklab": 5, "oklch": 6, "jzazbz": 7}


@njit(cache=True)
def cu_s2l(g):
    """colorutils ``srgb_to_linear``."""
    if g < 0.0:
        return 0.0
    if g < 12.92 * _CU_LIN_T:
        return g / 12.92
    if g < 1.0:
        return ((g + _CU_A) / (1.0 + _CU_A)) ** 2.4
    return 1.0


@njit(cache=True)
def cu_l2s(v):
    """colorutils ``srgb_from_linear``."""
    if not (v >= 0.0):  # also NaN
        return 0.0
    if v < _CU_LIN_T:
        return v * 12.92
    if v < 1.0:
        return (1.0 + _CU_A) * v ** (1.0 / 2.4) - _CU_A
    return 1.0


@njit(cache=True)
def cu_rgb_to_xyz(r, g, b):
    r, g, b = cu_s2l(r), cu_s2l(g), cu_s2l(b)
    return (0.4124564 * r + 0.3575761 * g + 0.1804375 * b,
            0.2126729 * r + 0.7151522 * g + 0.0721750 * b,
            0.0193339 * r + 0.1191920 * g + 0.9503041 * b)


@njit(cache=True)
def cu_xyz_to_rgb(x, y, z):
    return (cu_l2s(3.2404542 * x - 1.5371385 * y - 0.4985314 * z),
            cu_l2s(-0.9692660 * x + 1.8760108 * y + 0.0415560 * z),
            cu_l2s(0.0556434 * x - 0.2040259 * y + 1.0572252 * z))


@njit(cache=True)
def _lab_f(t):
    return np.cbrt(t) if t > 0.008856 else 7.787 * t + 16.0 / 116.0


@njit(cache=True)
def cu_xyz_to_lab(x, y, z):
    fx, fy, fz = _lab_f(x * 100.0 / 95.047), _lab_f(y), _lab_f(z * 100.0 / 108.883)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


@njit(cache=True)
def _lab_finv(t):
    t3 = t * t * t
    return t3 if t3 > 0.008856 else (t - 16.0 / 116.0) / 7.787


@njit(cache=True)
def cu_lab_to_xyz(l, a, b):
    y = (l + 16.0) / 116.0
    return 0.95047 * _lab_finv(a / 500.0 + y), _lab_finv(y), 1.08883 * _lab_finv(y - b / 200.0)


@njit(cache=True)
def cu_xyz_to_luv(x, y, z):
    den = x + 15.0 * y + 3.0 * z
    l = (29.0 / 3.0) ** 3 * y if y < (6.0 / 29.0) ** 3 else 116.0 * np.cbrt(y) - 16.0
    l = min(max(l, 0.0), 100.0)
    if den != 0.0:
        return l, 13.0 * l * (4.0 * x / den - _LUV_U), 13.0 * l * (9.0 * y / den - _LUV_V)
    return l, 0.0, 0.0


@njit(cache=True)
def cu_luv_to_xyz(l, u, v):
    if l <= 0.0:
        return 0.0, 0.0, 0.0
    up = u / (13.0 * l) + _LUV_U
    vp = v / (13.0 * l) + _LUV_V
    y = ((l + 16.0) / 116.0) ** 3 if l > 8.0 else l * (3.0 / 29.0) ** 3
    if vp != 0.0:
        den = 1.0 / (4.0 * vp)
        return y * 9.0 * up * den, y, y * (12.0 - 3.0 * up - 20.0 * vp) * den
    return 0.0, y, 0.0


@njit(cache=True)
def cu_lin_to_oklab(r, g, b):
    l_ = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


@njit(cache=True)
def cu_oklab_to_lin(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_
    return (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)


@njit(cache=True)
def _pq(x):
    if x <= 0.0:
        return 0.0
    xx = (x * 1e-4) ** 0.1593017578125
    rs = ((0.8359375 + 18.8515625 * xx) / (1.0 + 18.6875 * xx)) ** 134.034375
    return 0.0 if rs != rs else rs


@njit(cache=True)
def _pq_inv(x):
    if x <= 0.0:
        return 0.0
    xx = x ** 7.460772656268214e-03
    base = (0.8359375 - xx) / (18.6875 * xx - 18.8515625)
    if base < 0.0:  # f32::powf of a negative base -> NaN -> 0
        return 0.0
    return 1e4 * base ** 6.277394636015326


@njit(cache=True)
def cu_xyz_to_jzazbz(x, y, z):
    x, y, z = x * _JZ_NITS, y * _JZ_NITS, z * _JZ_NITS
    lp = _pq(0.674207838 * x + 0.382799340 * y - 0.047570458 * z)
    mp = _pq(0.149284160 * x + 0.739628340 * y + 0.083327300 * z)
    sp = _pq(0.070941080 * x + 0.174768000 * y + 0.670970020 * z)
    iz = 0.5 * (lp + mp)
    return ((0.44 * iz) / (1.0 - 0.56 * iz) - _JZ_D0,
            3.524000 * lp - 4.066708 * mp + 0.542708 * sp,
            0.199076 * lp + 1.096799 * mp - 1.295875 * sp)


@njit(cache=True)
def cu_jzazbz_to_xyz(jz, az, bz):
    jz = jz + _JZ_D0
    iz = jz / (0.44 + 0.56 * jz)
    l = _pq_inv(iz + 1.386050432715393e-1 * az + 5.804731615611869e-2 * bz)
    m = _pq_inv(iz - 1.386050432715393e-1 * az - 5.804731615611891e-2 * bz)
    s = _pq_inv(iz - 9.601924202631895e-2 * az - 8.118918960560390e-1 * bz)
    return ((1.661373055774069 * l - 0.9145230923250668 * m + 0.2313620767186147 * s) / _JZ_NITS,
            (-0.3250758740427037 * l + 1.571847038366936 * m - 0.2182538318672940 * s) / _JZ_NITS,
            (-0.09098281098284756 * l - 0.3127282905230740 * m + 1.522766561305260 * s) / _JZ_NITS)


@njit(cache=True)
def _hue(r, g, b, mx, d):
    if mx == r:
        h = 60.0 * np.fmod((g - b) / d, 6.0)
    elif mx == g:
        h = 60.0 * ((b - r) / d + 2.0)
    else:
        h = 60.0 * ((r - g) / d + 4.0)
    return h + 360.0 if h < 0.0 else h


@njit(cache=True)
def cu_rgb_to_hsv(r, g, b):
    """colorutils ``Hsv::from`` (h in degrees)."""
    mx = max(r, g, b)
    d = mx - min(r, g, b)
    if d > 0.0:
        return _hue(r, g, b, mx, d), (d / mx if mx > 0.0 else 0.0), mx
    return 0.0, 0.0, mx


@njit(cache=True)
def cu_rgb_to_hsl(r, g, b):
    """colorutils ``rgb2hsl``; the 0/0 saturation of greys (NaN upstream) is returned as 0 like the u16 cast."""
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    l = 0.5 * (mx + mn)
    den = 1.0 - abs(2.0 * l - 1.0)
    h = 0.0 if d == 0.0 else _hue(r, g, b, mx, d)
    return h, (d / den if den != 0.0 else 0.0), l


@njit(cache=True)
def cu_hsv_to_rgb(h, s, v):
    c = v * s
    hp = np.fmod(h / 60.0, 6.0)
    x = c * (1.0 - abs(np.fmod(hp, 2.0) - 1.0))
    m = v - c
    if 0.0 <= hp < 1.0:
        return c + m, x + m, m
    if 1.0 <= hp < 2.0:
        return x + m, c + m, m
    if 2.0 <= hp < 3.0:
        return m, c + m, x + m
    if 3.0 <= hp < 4.0:
        return m, x + m, c + m
    if 4.0 <= hp < 5.0:
        return x + m, m, c + m
    if 5.0 <= hp < 6.0:
        return c + m, m, x + m
    return m, m, m


@njit(cache=True)
def cu_hsl_to_rgb(h, s, l):
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - abs(np.fmod(h / 60.0, 2.0) - 1.0))
    m = l - c / 2.0
    if 0.0 <= h < 60.0:
        return c + m, x + m, m
    if 60.0 <= h < 120.0:
        return x + m, c + m, m
    if 120.0 <= h < 180.0:
        return m, c + m, x + m
    if 180.0 <= h < 240.0:
        return m, x + m, c + m
    if 240.0 <= h < 300.0:
        return x + m, m, c + m
    return c + m, m, x + m


@njit(cache=True)
def _as_u16(x):
    """Rust ``f32 as u16``: truncation toward zero, saturating, NaN -> 0."""
    if not (x > 0.0):
        return 0.0
    return min(math.floor(x), 65535.0)


@njit(cache=True)
def _round_u16(x):
    """Rust ``x.round() as u16`` (round half away from zero, saturating)."""
    if not (x > 0.0):
        return 0.0
    return min(math.floor(x + 0.5), 65535.0)


@njit(cache=True, parallel=True)
def encode_plane(rgb, space, scale):  # pragma: no cover - numba
    """Upstream ``destructuring`` step: quantised plane ``q`` (0..scale) + two side channels."""
    h, w = rgb.shape[0], rgb.shape[1]
    q = np.empty((h, w), np.float32)
    side = np.empty((h, w, 2), np.float32)
    for yy in prange(h):
        for xx in range(w):
            r = min(max(float(rgb[yy, xx, 0]), 0.0), 1.0)
            g = min(max(float(rgb[yy, xx, 1]), 0.0), 1.0)
            b = min(max(float(rgb[yy, xx, 2]), 0.0), 1.0)
            if space == 0:    # yuv 0.8.6 rgba_to_ycgco444, full range, 8 bit
                q[yy, xx] = min(max(math.floor((2.0 * g + r + b) * 255.0 / 4.0 + 4095.0 / 8192.0), 0.0), 255.0)
                side[yy, xx, 0] = (2.0 * g - r - b) * 255.0 / 4.0
                side[yy, xx, 1] = (r - b) * 255.0 / 2.0
            elif space == 1 or space == 2:
                if space == 1:
                    hh, ss, vv = cu_rgb_to_hsv(r, g, b)
                else:
                    hh, ss, vv = cu_rgb_to_hsl(r, g, b)
                q[yy, xx] = _round_u16(vv * scale)
                side[yy, xx, 0] = _as_u16(hh)            # hue stored as whole degrees
                side[yy, xx, 1] = _round_u16(ss * scale)
            elif space == 3 or space == 4:
                x, y, z = cu_rgb_to_xyz(r, g, b)
                if space == 3:
                    l, a, bb = cu_xyz_to_lab(x, y, z)
                else:
                    l, a, bb = cu_xyz_to_luv(x, y, z)
                q[yy, xx] = _as_u16(min(math.floor(l * scale / 100.0 + 0.5), scale))
                # Bug fix vs upstream: the crate stores ``(v * 100 + 100) as u16``, which clamps every
                # chroma value below -1 (greens/blues lose colour). Keep the exact values instead.
                side[yy, xx, 0] = a
                side[yy, xx, 1] = bb
            elif space == 5 or space == 6:
                L, a, bb = cu_lin_to_oklab(cu_s2l(r), cu_s2l(g), cu_s2l(b))
                q[yy, xx] = _as_u16(min(math.floor(L * scale + 0.5), scale))
                if space == 5:
                    side[yy, xx, 0] = a
                    side[yy, xx, 1] = bb
                else:
                    side[yy, xx, 0] = math.hypot(bb, a)
                    side[yy, xx, 1] = math.atan2(bb, a)
            else:
                x, y, z = cu_rgb_to_xyz(r, g, b)
                jz, az, bz = cu_xyz_to_jzazbz(x, y, z)
                q[yy, xx] = _as_u16(min(math.floor(jz * scale + 0.5), scale))
                side[yy, xx, 0] = az
                side[yy, xx, 1] = bz
    return q, side


@njit(cache=True, parallel=True)
def decode_plane(q, side, space, scale):  # pragma: no cover - numba
    """Upstream ``structuring`` step back to (unquantised) RGB 0..1."""
    h, w = q.shape
    out = np.empty((h, w, 3), np.float32)
    for yy in prange(h):
        for xx in range(w):
            v = float(q[yy, xx])
            s0 = float(side[yy, xx, 0])
            s1 = float(side[yy, xx, 1])
            if space == 0:
                t = v - s0
                r, g, b = (t + s1) / 255.0, (v + s0) / 255.0, (t - s1) / 255.0
            elif space == 1:
                r, g, b = cu_hsv_to_rgb(s0, s1 / scale, v / scale)
            elif space == 2:
                r, g, b = cu_hsl_to_rgb(s0, s1 / scale, v / scale)
            elif space == 3:
                x, y, z = cu_lab_to_xyz(v * 100.0 / scale, s0, s1)
                r, g, b = cu_xyz_to_rgb(x, y, z)
            elif space == 4:
                x, y, z = cu_luv_to_xyz(v * 100.0 / scale, s0, s1)
                r, g, b = cu_xyz_to_rgb(x, y, z)
            elif space == 5 or space == 6:
                a, bb = s0, s1
                if space == 6:
                    a, bb = s0 * math.cos(s1), s0 * math.sin(s1)
                lr, lg, lb = cu_oklab_to_lin(v / scale, a, bb)
                r, g, b = cu_l2s(lr), cu_l2s(lg), cu_l2s(lb)
            else:
                x, y, z = cu_jzazbz_to_xyz(v / scale, s0, s1)
                r, g, b = cu_xyz_to_rgb(x, y, z)
            out[yy, xx, 0] = min(max(r, 0.0), 1.0) if r == r else 0.0
            out[yy, xx, 1] = min(max(g, 0.0), 1.0) if g == g else 0.0
            out[yy, xx, 2] = min(max(b, 0.0), 1.0) if b == b else 0.0
    return out


@njit(cache=True)
def _tile_histograms(q, bins, tw, th, ntx, nty):  # pragma: no cover - numba
    h, w = q.shape
    hist = np.zeros((nty * ntx, bins), np.int64)
    for yy in range(h):
        row = min(yy // th, nty - 1)
        for xx in range(w):
            col = min(xx // tw, ntx - 1)
            hist[row * ntx + col, min(int(q[yy, xx]), bins - 1)] += 1
    return hist


@njit(cache=True, parallel=True)
def _interpolate_tiles(q, lut, maxb, c0, c1, x1, r0, r1, y1):  # pragma: no cover - numba
    h, w = q.shape
    out = np.empty((h, w), np.float32)
    for yy in prange(h):
        ty = y1[yy]
        for xx in range(w):
            v = min(int(q[yy, xx]), maxb)
            tx = x1[xx]
            b1 = lut[r0[yy], c0[xx], v]
            b2 = lut[r0[yy], c1[xx], v]
            b3 = lut[r1[yy], c0[xx], v]
            b4 = lut[r1[yy], c1[xx], v]
            top = b1 * (np.float32(1.0) - tx) + b2 * tx
            bot = b3 * (np.float32(1.0) - tx) + b4 * tx
            res = top * (np.float32(1.0) - ty) + bot * ty
            out[yy, xx] = math.floor(min(max(res, np.float32(0.0)), np.float32(maxb)))
    return out


def _cdf_lut(hist: np.ndarray, area: np.ndarray, bins: int) -> np.ndarray:
    """CDF -> ``round(max_bins * (cdf - min) / (area - min))`` with Rust's NaN-ignoring min/max."""
    maxb = bins - 1
    cdf = np.cumsum(hist, axis=-1)
    mn = cdf.min(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dr = 1.0 / (area - mn).astype(np.float64)
        val = np.floor(maxb * (cdf - mn[..., None]).astype(np.float64) * dr[..., None] + 0.5)
    return np.fmax(np.fmin(val, maxb), 0.0).astype(F32)


def ahe_map(q: np.ndarray, bins: int, grid_w: int, grid_h: int, threshold: float | None) -> np.ndarray:
    """``clahe_impl_u16`` / ``clahe_yuv_impl``: tile histograms (optionally clipped), CDF mapping and
    the crate's (extrapolating) bilinear interpolation.  ``q`` holds integer bin indices.
    Returns the equalised plane as integer-valued float32 in 0..bins-1."""
    h, w = q.shape
    tw = max(1, w // max(1, grid_w))   # upstream panics (division by zero) when grid > size
    th = max(1, h // max(1, grid_h))
    ntx, nty = w // tw, h // th
    hist = _tile_histograms(np.ascontiguousarray(q, dtype=F32), bins, tw, th, ntx, nty)
    rw = np.full(ntx, tw, np.int64)
    rw[-1] = w - (ntx - 1) * tw
    rh = np.full(nty, th, np.int64)
    rh[-1] = h - (nty - 1) * th
    area = (rh[:, None] * rw[None, :]).ravel()
    if threshold is not None:  # clip_hist_clahe (note: the whole value of a clipped bin counts as excess)
        tv = float(threshold) * area.astype(np.float64) / bins
        clip_limit = np.where(tv > 0, np.floor(tv), 0).astype(np.int64)[:, None]
        over = hist > clip_limit
        excess = np.where(over, hist, 0).sum(axis=1)
        hist = np.where(over, clip_limit, hist)
        hist = hist + np.floor(excess / bins).astype(np.int64)[:, None]
    lut = _cdf_lut(hist, area, bins).reshape(nty, ntx, bins)

    xs = np.arange(w, dtype=F32)
    ys = np.arange(h, dtype=F32)
    ftw, fth = F32(tw), F32(th)
    cxf = (xs - ftw / F32(2)) / ftw
    ryf = (ys - fth / F32(2)) / fth
    x1 = ((xs - (np.trunc(cxf) + F32(0.5)) * ftw) / ftw).astype(F32)
    y1 = ((ys - (np.trunc(ryf) + F32(0.5)) * fth) / fth).astype(F32)
    c0 = np.minimum(np.maximum(cxf, 0).astype(np.int64), ntx - 1)
    r0 = np.minimum(np.maximum(ryf, 0).astype(np.int64), nty - 1)
    return _interpolate_tiles(np.ascontiguousarray(q, dtype=F32), lut, bins - 1, c0, np.minimum(c0 + 1, ntx - 1),
                              x1, r0, np.minimum(r0 + 1, nty - 1), y1)


def he_map_global(q: np.ndarray, bins: int) -> np.ndarray:
    """``equalize_histogram_impl``: global CDF mapping of a quantised plane."""
    v = np.clip(q, 0, bins - 1).astype(np.int64)
    lut = _cdf_lut(np.bincount(v.ravel(), minlength=bins), np.int64(v.size), bins)
    return np.take(lut, v)


def equalize(img: np.ndarray, space: str, bins: int, grid: tuple[int, int] | None,
             threshold: float | None) -> np.ndarray:
    """Run (C)LAHE (``grid`` given) or global equalisation in a colour space.

    ``space``: ycgco | hsv | hsl | lab | luv | oklab | oklch | jzazbz (YCgCo always uses 256 bins)."""
    if space == "ycgco":
        bins = 256
    sid = SPACES[space]
    q, side = encode_plane(np.ascontiguousarray(img[..., :3], dtype=F32), sid, float(bins - 1))
    q2 = he_map_global(q, bins) if grid is None else ahe_map(q, bins, grid[0], grid[1], threshold)
    return decode_plane(q2, side, sid, float(bins - 1))


# ===========================================================================
# Trickle bloom
# ===========================================================================

@njit(cache=True, parallel=True)
def _bloom_bright(rgb, t255, knee, exposure):  # pragma: no cover - numba
    h, w = rgb.shape[0], rgb.shape[1]
    out = np.empty((h, w, 3), np.float32)
    for y in prange(h):
        for x in range(w):
            r = rgb[y, x, 0] * 255.0
            g = rgb[y, x, 1] * 255.0
            b = rgb[y, x, 2] * 255.0
            diff = 0.299 * r + 0.587 * g + 0.114 * b - t255
            weight = 0.0
            if diff > 0.0:
                weight = diff
            elif diff > -knee:
                weight = diff * diff / (4.0 * knee)
            glow = 1.0 - math.exp(-max(weight, 0.0) * exposure)
            out[y, x, 0] = r * glow
            out[y, x, 1] = g * glow
            out[y, x, 2] = b * glow
    return out


@njit(cache=True, parallel=True)
def _bloom_combine(rgb, blurred, intensity, gamma_lut):  # pragma: no cover - numba
    h, w = rgb.shape[0], rgb.shape[1]
    out = np.empty((h, w, 3), np.float32)
    for y in prange(h):
        for x in range(w):
            for k in range(3):
                out[y, x, k] = _lut_at(gamma_lut, rgb[y, x, k] + blurred[y, x, k] * intensity / 255.0)
    return out


def trickle_bloom(img, threshold, intensity, radius_px: int, soft_knee, exposure, gamma):
    """Trickle ``bloomImpl`` (raw RGBA bytes, no channel swap): soft-knee bright pass in 0..255
    units, truncated Gaussian (radius r, sigma r/2, clamped edges), additive, ``pow(x, 1/gamma)``."""
    rgb = np.ascontiguousarray(img[..., :3], dtype=F32)
    t255 = min(1.0, max(0.0, threshold)) * 255.0
    bright = _bloom_bright(rgb, t255, t255 * soft_knee, float(exposure))
    r = max(1, int(radius_px))
    blurred = cv2.GaussianBlur(bright, (2 * r + 1, 2 * r + 1), sigmaX=r * 0.5, sigmaY=r * 0.5,
                               borderType=cv2.BORDER_REPLICATE)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        inv_g = np.float32(1.0) / np.float32(gamma) if gamma != 0 else np.float32(np.inf)
        lut = np.clip(_nan0(np.power(LUT_GRID, inv_g)), 0, 1)  # values >= 1 clamp to 1 anyway
    return _bloom_combine(rgb, blurred.reshape(rgb.shape), float(intensity), lut)


# ===========================================================================
# jhlabs (native Rust port in ImageToolboxLibs)
# ===========================================================================

def jh_gaussian_kernel(radius: float) -> np.ndarray:
    half = int(math.ceil(radius))
    sigma = radius / 3.0
    rows = np.arange(-half, half + 1, dtype=np.float64)
    d = rows * rows
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        vals = np.exp(-d / (2 * sigma * sigma)) / math.sqrt(2 * math.pi * sigma)
    vals = np.where(d > radius * radius, 0.0, vals)
    vals = np.nan_to_num(vals, nan=0.0)
    if vals.sum() <= 0:
        vals = (rows == 0).astype(np.float64)
    return (vals / vals.sum()).astype(F32)


def jh_gaussian_blur(img: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return img
    k = jh_gaussian_kernel(radius)
    if len(k) == 1:
        return img
    out = cv2.sepFilter2D(img.astype(F32), -1, k, k, borderType=cv2.BORDER_REPLICATE)
    return out.reshape(img.shape).astype(F32)


class JavaRandom:
    """``java.util.Random`` (as re-implemented in the jhlabs Rust library)."""
    _MASK = (1 << 48) - 1

    def __init__(self, seed: int) -> None:
        self.seed = (seed ^ 0x5DEECE66D) & self._MASK
        self._next_gaussian: float | None = None

    def next(self, bits: int) -> int:
        self.seed = (self.seed * 0x5DEECE66D + 0xB) & self._MASK
        v = (self.seed >> (48 - bits)) & 0xFFFFFFFF
        return v - (1 << 32) if v >= (1 << 31) else v

    def next_int(self) -> int:
        return self.next(32)

    def next_double(self) -> float:
        return ((self.next(26) << 27) + self.next(27)) / float(1 << 53)

    def next_gaussian(self) -> float:
        if self._next_gaussian is not None:
            v, self._next_gaussian = self._next_gaussian, None
            return v
        while True:
            a = 2.0 * self.next_double() - 1.0
            b = 2.0 * self.next_double() - 1.0
            s = a * a + b * b
            if 0.0 < s < 1.0:
                mul = math.sqrt(-2.0 * math.log(s) / s)
                self._next_gaussian = b * mul
                return a * mul


@functools.lru_cache(maxsize=1)
def _perlin1_tables() -> tuple[np.ndarray, np.ndarray]:
    """``PerlinNoise::new(0x4a484c616273)``: permutation + 1D gradients (2D/3D draws consumed)."""
    rnd = JavaRandom(0x4A484C616273)
    perm = list(range(256)) + [0] * 258
    g1 = [0.0] * 514

    def pos() -> int:
        return rnd.next_int() & 0x7FFFFFFF
    for i in range(256):
        g1[i] = float(np.float32(((pos() % 512) - 256) / 256.0))
        for _ in range(5):  # gradient2 (2 values) and gradient3 (3 values) draws
            pos()
    for i in range(255, -1, -1):
        t = pos() % 256
        perm[i], perm[t] = perm[t], perm[i]
    for i in range(258):
        perm[256 + i] = perm[i]
        g1[256 + i] = g1[i]
    return np.array(perm, np.int64), np.array(g1, F32)


@njit(cache=True)
def _noise1(perm, g1, value):
    shifted = value + 4096.0
    ti = int(shifted)
    first = ti & 255
    second = (first + 1) & 255
    rem = shifted - ti
    curve = rem * rem * (3.0 - 2.0 * rem)
    fv = rem * g1[perm[first]]
    sv = (rem - 1.0) * g1[perm[second]]
    return 2.3 * (fv + curve * (sv - fv))


@njit(cache=True, parallel=True)
def _flare_kernel(img, inv_scale, cx, cy, radius, rw, base_amount, ring_amount, ray_amount, col,
                  perm, g1):  # pragma: no cover - numba
    h, w = img.shape[0], img.shape[1]
    out = np.empty((h, w, 3), np.float32)
    for y in prange(h):
        dy = y * inv_scale - cy
        for x in range(w):
            dx = x * inv_scale - cx
            dist = math.sqrt(dx * dx + dy * dy)
            amount = (math.exp(-dist * dist * 0.006) * 0.5 + math.exp(-dist * 0.03) * 0.5) * base_amount
            if dist > radius + rw:
                amount += min(max((dist - radius - rw) / 6.0, 0.0), 1.0) * (0.0 - amount)
            if not (dist < radius - rw or dist > radius + rw):
                v = abs(dist - radius) / rw
                amount += (1.0 - v * v * (3.0 - 2.0 * v)) * ring_amount
            angle = math.atan2(dx, dy) + math.pi
            t = angle / math.pi * 17.0 + 1.0 + _noise1(perm, g1, angle * 10.0)
            angle = ((t - math.floor(t)) - 0.5) * 2.0
            amount += ray_amount * abs(angle) ** 5 / (1.0 + dist * 0.1)
            a = min(max(amount, 0.0), 1.0)
            for k in range(3):
                out[y, x, k] = img[y, x, k] * (1.0 - a) + col[k] * a
    return out


def jh_flare(img, cx, cy, radius, ring_width, base_amount, ring_amount, ray_amount, color, inv_scale=1.0):
    """jhlabs Rust ``FlareFilter``; coordinates are mapped to full-resolution pixels with ``inv_scale``."""
    perm, g1 = _perlin1_tables()
    return _flare_kernel(np.ascontiguousarray(img[..., :3], dtype=F32), float(inv_scale), float(cx), float(cy),
                         float(radius), float(ring_width), float(base_amount), float(ring_amount), float(ray_amount),
                         np.asarray(color[:3], np.float64), perm, g1)


@njit(cache=True, parallel=True)
def _sparkle_kernel(img, inv_scale, cx, cy, radius, exponent, lengths, col):  # pragma: no cover - numba
    h, w = img.shape[0], img.shape[1]
    rays = lengths.shape[0]
    out = np.empty((h, w, 3), np.float32)
    for y in prange(h):
        dy = y * inv_scale - cy
        for x in range(w):
            dx = x * inv_scale - cx
            dist = dx * dx + dy * dy
            position = (math.atan2(dy, dx) + math.pi) / (2.0 * math.pi) * rays
            ray = int(position)
            factor = position - ray
            if radius != 0:
                l0 = lengths[ray % rays]
                length = l0 + factor * (lengths[(ray + 1) % rays] - l0)
                gain = (length * length / (dist + 0.0001)) ** exponent
                factor -= 0.5
                factor = (1.0 - factor * factor) * gain
            a = min(max(factor, 0.0), 1.0) if factor == factor else 0.0
            for k in range(3):
                out[y, x, k] = img[y, x, k] * (1.0 - a) + col[k] * a
    return out


def jh_sparkle(img, cx, cy, radius: int, amount: int, rays: int, randomness: int, color, inv_scale=1.0):
    """jhlabs Rust ``SparkleFilter`` (fixed ``java.util.Random`` seed 371 for the ray lengths)."""
    rays = max(1, int(rays))
    rnd = JavaRandom(371)
    lengths = np.array([np.float32(radius + randomness / 100.0 * radius * np.float32(rnd.next_gaussian()))
                        for _ in range(rays)], np.float64)
    return _sparkle_kernel(np.ascontiguousarray(img[..., :3], dtype=F32), float(inv_scale), float(cx), float(cy),
                           int(radius), (100 - amount) / 50.0, lengths, np.asarray(color[:3], np.float64))


@njit(cache=True, parallel=True)
def _rays_kernel(src, opacity, thr255, strength):  # pragma: no cover - numba
    h, w = src.shape[0], src.shape[1]
    cx, cy = w * 0.5, h * 0.5
    out = np.empty((h, w, 3), np.float32)
    for y in prange(h):
        for x in range(w):
            tr = 0.0
            tg = 0.0
            tb = 0.0
            count = 0
            for step in range(16):
                t = step / 15.0 * strength
                sx = x + (cx - x) * t
                sy = y + (cy - y) * t
                fx0 = math.floor(sx)
                fy0 = math.floor(sy)
                xw = sx - fx0
                yw = sy - fy0
                x0 = min(max(int(fx0), 0), w - 1)
                x1 = min(max(int(fx0) + 1, 0), w - 1)
                y0 = min(max(int(fy0), 0), h - 1)
                y1 = min(max(int(fy0) + 1, 0), h - 1)
                c = np.empty(3)
                for k in range(3):
                    north = src[y0, x0, k] * (1.0 - xw) + src[y0, x1, k] * xw
                    south = src[y1, x0, k] * (1.0 - xw) + src[y1, x1, k] * xw
                    c[k] = north * (1.0 - yw) + south * yw
                if (c[0] + c[1] + c[2]) / 3.0 * 255.0 >= thr255:
                    tr += c[0]
                    tg += c[1]
                    tb += c[2]
                    count += 1
            if count > 0:
                out[y, x, 0] = tr / count * opacity + src[y, x, 0] * (1.0 - opacity)
                out[y, x, 1] = tg / count * opacity + src[y, x, 1] * (1.0 - opacity)
                out[y, x, 2] = tb / count * opacity + src[y, x, 2] * (1.0 - opacity)
            else:
                out[y, x, 0] = src[y, x, 0]
                out[y, x, 1] = src[y, x, 1]
                out[y, x, 2] = src[y, x, 2]
    return out


def jh_rays(img, opacity, threshold, strength):
    """jhlabs Rust ``RaysFilter``: 16 bilinear samples towards the centre, averaged where bright,
    composited over the source with ``opacity``."""
    return _rays_kernel(np.ascontiguousarray(img[..., :3], dtype=F32), min(1.0, max(0.0, opacity)),
                        threshold * 255.0, float(strength))


@njit(cache=True, parallel=True)
def _light_kernel(src, blurred, bump_height, lx, ly, lz):  # pragma: no cover - numba
    h, w = src.shape[0], src.shape[1]
    out = np.empty((h, w, 3), np.float32)
    for y in prange(h):
        ym, yp = max(y - 1, 0), min(y + 1, h - 1)
        for x in range(w):
            xm, xp = max(x - 1, 0), min(x + 1, w - 1)
            gx = (blurred[y, xp, 0] + blurred[y, xp, 1] + blurred[y, xp, 2]
                  - blurred[y, xm, 0] - blurred[y, xm, 1] - blurred[y, xm, 2]) * (255.0 / 3.0)
            gy = (blurred[yp, x, 0] + blurred[yp, x, 1] + blurred[yp, x, 2]
                  - blurred[ym, x, 0] - blurred[ym, x, 1] - blurred[ym, x, 2]) * (255.0 / 3.0)
            nx = -gx * bump_height / 255.0
            ny = -gy * bump_height / 255.0
            shade = max((nx * lx + ny * ly + lz) / math.sqrt(nx * nx + ny * ny + 1.0), 0.0)
            for k in range(3):
                out[y, x, k] = src[y, x, k] * shade
    return out


def jh_light(img, bump_height, bump_softness_px):
    """jhlabs Rust ``LightFilter`` (simplified bump shading with the default distant light:
    azimuth 270 deg, elevation 30 deg, white, intensity 1)."""
    src = np.ascontiguousarray(img[..., :3], dtype=F32)
    blurred = np.ascontiguousarray(jh_gaussian_blur(src, bump_softness_px), dtype=F32)
    az, el = 4.712389, 0.523599
    return _light_kernel(src, blurred, float(bump_height), math.cos(az) * math.cos(el),
                         math.sin(az) * math.cos(el), math.sin(el))


# ===========================================================================
# GPUImage shaders with transcendental maths (per pixel)
# ===========================================================================

@njit(cache=True, parallel=True)
def highlight_shadow_wide_range(img, shadows, highlights):  # pragma: no cover - numba
    """ImageToolbox ``GPUImageHighlightShadowWideRangeFilter`` fragment shader."""
    h, w = img.shape[0], img.shape[1]
    out = np.empty((h, w, 3), np.float32)
    white_target = min(max(highlights, 1.0), 2.0) - 1.0
    black_target = 1.0 - min(max(shadows, 0.0), 1.0)
    for y in prange(h):
        for x in range(w):
            r, g, b = img[y, x, 0], img[y, x, 1], img[y, x, 2]
            lum = (r + g + b) * 0.3
            if shadows > 0.0:
                sh = lum ** (1.0 / shadows) + (-0.76) * lum ** (2.0 / shadows) - lum
            else:  # pow(lum, inf): 0 below 1
                sh = -lum
            sh = min(max(sh, 0.0), 1.0)
            omn = max(1.0 - lum, 0.0)
            e = 2.0 - highlights
            if e > 0.0:
                hl = 1.0 - (omn ** (1.0 / e) + (-0.8) * omn ** (2.0 / e)) - lum
            else:
                hl = 1.0 - lum
            hl = min(max(hl, -1.0), 0.0)
            ratio = (lum + sh + hl) / lum if lum > 0.0 else 0.0
            cl = (lum - 0.5) * 1.5 + 0.5
            wi = cl * cl * cl * white_target
            icl = 1.0 - cl
            bi = icl * icl * icl * black_target
            for k in range(3):
                v = img[y, x, k] * ratio
                v = v + (1.0 - v) * wi
                out[y, x, k] = v - v * bi
    return out


@njit(cache=True, parallel=True)
def gpuimage_vignette(img, start, end, col):  # pragma: no cover - numba
    """GPUImageVignetteFilter: mix(rgb, colour, smoothstep(start, end, distance(uv, (0.5, 0.5))))."""
    h, w = img.shape[0], img.shape[1]
    out = np.empty((h, w, 3), np.float32)
    for y in prange(h):
        v = (y + 0.5) / h - 0.5
        for x in range(w):
            u = (x + 0.5) / w - 0.5
            d = math.sqrt(u * u + v * v)
            if end != start:
                t = min(max((d - start) / (end - start), 0.0), 1.0)
            else:
                t = 1.0 if d >= start else 0.0
            pct = t * t * (3.0 - 2.0 * t)
            for k in range(3):
                out[y, x, k] = img[y, x, k] + (col[k] - img[y, x, k]) * pct
    return out


# GLSL ``mat3`` constructors are column-major: rows of these arrays are the GLSL columns.
_RGB2YIQ_COLS = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.212, -0.523, 0.311]])
_YIQ2RGB_COLS = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.105, 1.702]])


@njit(cache=True, parallel=True)
def _white_balance_kernel(img, m1, m2, temperature, tint):  # pragma: no cover - numba
    h, w = img.shape[0], img.shape[1]
    out = np.empty((h, w, 3), np.float32)
    warm = (0.93, 0.54, 0.0)
    for y in prange(h):
        for x in range(w):
            r, g, b = img[y, x, 0], img[y, x, 1], img[y, x, 2]
            yq0 = m1[0, 0] * r + m1[1, 0] * g + m1[2, 0] * b   # GLSL M * v with column-major M
            yq1 = m1[0, 1] * r + m1[1, 1] * g + m1[2, 1] * b
            yq2 = m1[0, 2] * r + m1[1, 2] * g + m1[2, 2] * b
            yq2 = min(max(yq2 + tint * 0.5226 * 0.1, -0.5226), 0.5226)
            for k in range(3):
                c = m2[0, k] * yq0 + m2[1, k] * yq1 + m2[2, k] * yq2
                if c < 0.5:
                    p = 2.0 * c * warm[k]
                else:
                    p = 1.0 - 2.0 * (1.0 - c) * (1.0 - warm[k])
                out[y, x, k] = c + (p - c) * temperature
    return out


def gpuimage_white_balance(img, temperature, tint):
    """GPUImageWhiteBalanceFilter (setTemperature/setTint mappings included)."""
    t = 0.0004 * (temperature - 5000.0) if temperature < 5000 else 0.00006 * (temperature - 5000.0)
    return _white_balance_kernel(np.ascontiguousarray(img[..., :3], dtype=F32), _RGB2YIQ_COLS, _YIQ2RGB_COLS,
                                 t, tint / 100.0)


# ===========================================================================
# ImageToolbox procedural GLSL shaders (BokehLights / Fireworks)
# ===========================================================================

def _fract(x):
    return x - np.floor(x)


def glsl_hash(px: float, py: float) -> float:
    p = np.array([px, py, px], F32)
    hv = _fract(p * np.array([0.1031, 0.1030, 0.0973], F32)).astype(F32)
    hv = (hv + F32(np.dot(hv, hv[[1, 2, 0]] + F32(33.33)))).astype(F32)
    return float(_fract(F32((hv[0] + hv[1]) * hv[2])))


def glsl_hash2(px: float, py: float) -> tuple[float, float]:
    return glsl_hash(px, py), glsl_hash(float(F32(px) + F32(17.17)), float(F32(py) + F32(47.23)))


def glsl_rgb_to_hsv(c):
    r, g, b = c
    high, low = max(r, g, b), min(r, g, b)
    d = high - low
    hue = 0.0
    if d > 0.00001:
        if high == r:
            t = (g - b) / d
            hue = t - 6.0 * math.floor(t / 6.0)  # GLSL mod
        elif high == g:
            hue = (b - r) / d + 2.0
        else:
            hue = (r - g) / d + 4.0
    return [hue / 6.0 - math.floor(hue / 6.0), d / high if high > 0.00001 else 0.0, high]


def glsl_hsv_to_rgb(hsv):
    h, s, v = hsv
    out = []
    for off in (0.0, 2.0 / 3.0, 1.0 / 3.0):
        t = h + off
        wave = min(1.0, max(0.0, abs((t - math.floor(t)) * 6.0 - 3.0) - 1.0))
        out.append(v * (1.0 + (wave - 1.0) * s))
    return out


def field_transform(w: int, h: int, offset_x: float, offset_y: float, scale: float, rotation: float):
    short = max(1, min(w, h))
    rad = math.radians(rotation)
    return w / short, h / short, offset_x, offset_y, math.cos(rad), math.sin(rad), scale


@njit(cache=True, parallel=True)
def _bokeh_kernel(h, w, ax, ay, offx, offy, cosr, sinr, scale, cx, cy, rad, colors, sector):  # pragma: no cover
    out = np.zeros((h, w, 3), np.float32)
    half = sector * 0.5
    cos_half = math.cos(half)
    n = cx.shape[0]
    for y in prange(h):
        v = (y + 0.5) / h
        qy = (v - 0.5 - offy) * ay
        for x in range(w):
            u = (x + 0.5) / w
            qx = (u - 0.5 - offx) * ax
            px = (cosr * qx + sinr * qy) / scale
            py = (-sinr * qx + cosr * qy) / scale
            lr = 0.0
            lg = 0.0
            lb = 0.0
            for i in range(n):
                dx = px - cx[i]
                dy = py - cy[i]
                d2 = dx * dx + dy * dy
                r = rad[i]
                if d2 >= r * r:
                    continue  # edge <= r, so the disc is exactly 0 here
                ang = 0.0 if d2 < 1e-8 else math.atan2(dy, dx)
                a = ang + half
                a = a - sector * math.floor(a / sector) - half
                edge = r * cos_half / max(math.cos(a), 0.001)
                e0 = edge * 0.8
                t = (math.sqrt(d2) - e0) / (edge - e0)
                t = min(1.0, max(0.0, t))
                disc = 1.0 - t * t * (3.0 - 2.0 * t)
                lr += colors[i, 0] * disc
                lg += colors[i, 1] * disc
                lb += colors[i, 2] * disc
            out[y, x, 0] = lr
            out[y, x, 1] = lg
            out[y, x, 2] = lb
    return out


@njit(cache=True, parallel=True)
def _fireworks_kernel(h, w, ax, ay, offx, offy, cosr, sinr, scale, ex, ey, rx, ry, colors, particles,
                      mode, time):  # pragma: no cover
    out = np.zeros((h, w, 3), np.float32)
    n = ex.shape[0]
    tau = 6.28318530718
    k = 80.0 if mode < 0.5 else 30.0
    for y in prange(h):
        v = (y + 0.5) / h
        qy = (v - 0.5 - offy) * ay
        for x in range(w):
            u = (x + 0.5) / w
            qx = (u - 0.5 - offx) * ax
            px = (cosr * qx + sinr * qy) / scale
            py = (-sinr * qx + cosr * qy) / scale
            lr = 0.0
            lg = 0.0
            lb = 0.0
            for i in range(n):
                dx = px - ex[i]
                dy = py - ey[i]
                d2 = dx * dx + dy * dy
                radius = math.sqrt(d2)
                dist = abs(radius - time * (0.1 + ry[i] * 0.4))
                trail = math.exp(-dist * k)
                if trail < 1e-7:  # contribution below float noise: skip the angular term
                    continue
                if mode > 1.5:
                    trail *= 0.5 + 0.5 * math.sin(radius * 150.0)
                ang = (0.0 if d2 < 1e-8 else math.atan2(dy, dx)) + rx[i] * tau
                c = max(0.0, math.cos(ang * particles))
                c2 = c * c
                c8 = c2 * c2 * c2 * c2
                f = c8 * c8 * c8 * trail  # pow(c, 24)
                lr += colors[i, 0] * f
                lg += colors[i, 1] * f
                lb += colors[i, 2] * f
            out[y, x, 0] = lr
            out[y, x, 1] = lg
            out[y, x, 2] = lb
    return out


def bokeh_lights(img, intensity, count, blades, radius, variability, color_variation, seed,
                 offset_x, offset_y, scale, rotation, color):
    h, w = img.shape[:2]
    ax, ay, ox, oy, cr, sr, sc = field_transform(w, h, offset_x, offset_y, scale, rotation)
    n = int(min(64, count))
    cx, cy, rad, cols = [], [], [], []
    base_hsv = glsl_rgb_to_hsv([float(color[0]), float(color[1]), float(color[2])])
    alpha = float(color[3]) if len(color) > 3 else 1.0
    for i in range(n):
        rnd = glsl_hash2(float(i), float(seed))
        cx.append((rnd[0] - 0.5) * ax * 1.5)
        cy.append((rnd[1] - 0.5) * ay * 1.5)
        rad.append(radius * (1.0 + (0.25 + rnd[0] - 1.0) * variability))
        hsv = list(base_hsv)
        t = hsv[0] + (rnd[1] - 0.5) * color_variation
        hsv[0] = t - math.floor(t)
        cols.append([c * alpha for c in glsl_hsv_to_rgb(hsv)])
    sector = 2 * math.pi / blades
    light = _bokeh_kernel(h, w, ax, ay, ox, oy, cr, sr, sc, np.array(cx, np.float64), np.array(cy, np.float64),
                          np.array(rad, np.float64), np.array(cols, np.float64).reshape(-1, 3), sector)
    return np.minimum(F32(1), img[..., :3] + light * F32(intensity)).astype(F32)


def fireworks(img, seed, explosions, particles, mode, intensity, time, offset_x, offset_y, scale, rotation):
    h, w = img.shape[:2]
    ax, ay, ox, oy, cr, sr, sc = field_transform(w, h, offset_x, offset_y, scale, rotation)
    n = int(min(12, explosions))
    ex, ey, rx, ry, cols = [], [], [], [], []
    for i in range(n):
        rnd = glsl_hash2(float(i), float(seed))
        ex.append((rnd[0] - 0.5) * ax)
        ey.append((rnd[1] - 0.5) * ay)
        rx.append(rnd[0])
        ry.append(rnd[1])
        cols.append(glsl_hsv_to_rgb([rnd[0], 0.8, 1.0]))
    light = _fireworks_kernel(h, w, ax, ay, ox, oy, cr, sr, sc, np.array(ex, np.float64), np.array(ey, np.float64),
                              np.array(rx, np.float64), np.array(ry, np.float64),
                              np.array(cols, np.float64).reshape(-1, 3), float(particles), float(mode), float(time))
    return np.minimum(F32(1), img[..., :3] + light * F32(intensity)).astype(F32)
