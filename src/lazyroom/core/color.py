"""Vectorised colour-science helpers (float32 numpy, channels-last).

Conventions
-----------
* "rgb" means sRGB-encoded (display-referred) values in 0..1 — the working
  encoding of the pipeline buffer.
* "lin" means linear-light sRGB/Rec.709 primaries, D65 white.
* Lab / LCh use D65 reference white; OKLab follows Björn Ottosson's definition.
* Hue values returned by HSV/HSL helpers are in 0..1 (multiply by 360 for degrees).
"""
from __future__ import annotations

import numpy as np

F32 = np.float32

# --------------------------------------------------------------------------
# Transfer functions
# --------------------------------------------------------------------------

def _np_srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=F32)
    a = np.abs(x)
    out = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return (np.sign(x) * out).astype(F32)


def _np_linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=F32)
    a = np.abs(x)
    out = np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(a, 1 / 2.4) - 0.055)
    return (np.sign(x) * out).astype(F32)


# Rec.709 / sRGB luminance weights (apply to linear values for true luminance,
# to encoded values for "luma").
LUMA_709 = np.array([0.2126, 0.7152, 0.0722], dtype=F32)
LUMA_601 = np.array([0.299, 0.587, 0.114], dtype=F32)


def luma(rgb: np.ndarray, weights: np.ndarray = LUMA_709) -> np.ndarray:
    return (rgb[..., :3] @ weights).astype(F32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Relative luminance Y (linear) from sRGB-encoded rgb."""
    return (srgb_to_linear(rgb[..., :3]) @ LUMA_709).astype(F32)


# --------------------------------------------------------------------------
# XYZ / Lab
# --------------------------------------------------------------------------
M_LIN_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=F32)
M_XYZ_TO_LIN = np.linalg.inv(M_LIN_TO_XYZ).astype(F32)
D65 = np.array([0.95047, 1.0, 1.08883], dtype=F32)


def lin_to_xyz(lin: np.ndarray) -> np.ndarray:
    return (lin @ M_LIN_TO_XYZ.T).astype(F32)


def xyz_to_lin(xyz: np.ndarray) -> np.ndarray:
    return (xyz @ M_XYZ_TO_LIN.T).astype(F32)


def _f_lab(t: np.ndarray) -> np.ndarray:
    d = 6 / 29
    return np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4 / 29)


def _f_lab_inv(t: np.ndarray) -> np.ndarray:
    d = 6 / 29
    return np.where(t > d, t ** 3, 3 * d * d * (t - 4 / 29))


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    f = _f_lab(xyz / D65)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(F32)


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    return (np.stack([_f_lab_inv(fx), _f_lab_inv(fy), _f_lab_inv(fz)], axis=-1) * D65).astype(F32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return xyz_to_lab(lin_to_xyz(srgb_to_linear(rgb[..., :3])))


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return linear_to_srgb(xyz_to_lin(lab_to_xyz(lab)))


# --------------------------------------------------------------------------
# OKLab (Ottosson 2020)
# --------------------------------------------------------------------------
_OK_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=F32)
_OK_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], dtype=F32)
_OK_M1_INV = np.linalg.inv(_OK_M1).astype(F32)
_OK_M2_INV = np.linalg.inv(_OK_M2).astype(F32)


def lin_to_oklab(lin: np.ndarray) -> np.ndarray:
    lms = lin @ _OK_M1.T
    return (np.cbrt(lms) @ _OK_M2.T).astype(F32)


def oklab_to_lin(lab: np.ndarray) -> np.ndarray:
    lms = (lab @ _OK_M2_INV.T) ** 3
    return (lms @ _OK_M1_INV.T).astype(F32)


def _np_rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    return lin_to_oklab(srgb_to_linear(rgb[..., :3]))


def _np_oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return linear_to_srgb(oklab_to_lin(lab))


def lab_to_lch(lab: np.ndarray) -> np.ndarray:
    C = np.hypot(lab[..., 1], lab[..., 2])
    h = np.arctan2(lab[..., 2], lab[..., 1])
    return np.stack([lab[..., 0], C, h], axis=-1).astype(F32)


def lch_to_lab(lch: np.ndarray) -> np.ndarray:
    return np.stack([lch[..., 0], lch[..., 1] * np.cos(lch[..., 2]), lch[..., 1] * np.sin(lch[..., 2])],
                    axis=-1).astype(F32)


# --------------------------------------------------------------------------
# HSV / HSL (hue 0..1)
# --------------------------------------------------------------------------

def _np_rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    safe = np.where(d == 0, 1, d)
    h = np.where(mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4))
    h = np.where(d == 0, 0, h) / 6.0
    s = np.where(mx == 0, 0, d / np.where(mx == 0, 1, mx))
    return np.stack([h % 1.0, s, mx], axis=-1).astype(F32)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0] % 1.0, hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6)
    f = h * 6 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    i = i.astype(np.int32) % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1).astype(F32)


def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    l = (mx + mn) / 2
    d = mx - mn
    denom = 1 - np.abs(2 * l - 1)
    s = np.where(d == 0, 0, d / np.where(denom == 0, 1, denom))
    hsv = rgb_to_hsv(rgb)
    return np.stack([hsv[..., 0], np.clip(s, 0, 1), l], axis=-1).astype(F32)


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    h, s, l = hsl[..., 0] % 1.0, hsl[..., 1], hsl[..., 2]
    c = (1 - np.abs(2 * l - 1)) * s
    hp = h * 6
    x = c * (1 - np.abs(hp % 2 - 1))
    m = l - c / 2
    i = np.floor(hp).astype(np.int32) % 6
    z = np.zeros_like(c)
    r = np.choose(i, [c, x, z, z, x, c])
    g = np.choose(i, [x, c, c, x, z, z])
    b = np.choose(i, [z, z, x, c, c, x])
    return (np.stack([r, g, b], axis=-1) + m[..., None]).astype(F32)


# --------------------------------------------------------------------------
# YCbCr (BT.601 full range)
# --------------------------------------------------------------------------

def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 + (b - y) * 0.564
    cr = 0.5 + (r - y) * 0.713
    return np.stack([y, cb, cr], axis=-1).astype(F32)


def ycbcr_to_rgb(ycc: np.ndarray) -> np.ndarray:
    y, cb, cr = ycc[..., 0], ycc[..., 1] - 0.5, ycc[..., 2] - 0.5
    r = y + 1.403 * cr
    g = y - 0.344 * cb - 0.714 * cr
    b = y + 1.773 * cb
    return np.stack([r, g, b], axis=-1).astype(F32)


# --------------------------------------------------------------------------
# Colour temperature / white balance
# --------------------------------------------------------------------------
M_BRADFORD = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
], dtype=np.float64)


def cct_to_xy(kelvin: float) -> tuple[float, float]:
    """CIE 1931 xy chromaticity of a CCT (Kim et al. cubic spline, 1667K..25000K),
    daylight locus above 4000K to match photographic white balance."""
    t = float(np.clip(kelvin, 1667.0, 25000.0))
    if t <= 4000:
        x = -0.2661239e9 / t ** 3 - 0.2343589e6 / t ** 2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t ** 3 + 2.1070379e6 / t ** 2 + 0.2226347e3 / t + 0.240390
    if t <= 2222:
        y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
    elif t <= 4000:
        y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483
    return x, y


def xy_to_xyz(x: float, y: float) -> np.ndarray:
    return np.array([x / y, 1.0, (1 - x - y) / y], dtype=np.float64)


def white_balance_matrix(src_kelvin: float, dst_kelvin: float = 6504.0, tint: float = 0.0) -> np.ndarray:
    """3x3 matrix (linear sRGB) adapting a scene lit at ``src_kelvin`` so it looks
    as if lit at ``dst_kelvin`` (Bradford CAT).  ``tint`` shifts green(-)/magenta(+)
    in units of roughly 0.01 Duv per 1.0."""
    sx, sy = cct_to_xy(src_kelvin)
    sy = sy - tint * 0.01  # positive tint -> source appears greener -> correct towards magenta
    src = xy_to_xyz(sx, sy)
    dst = xy_to_xyz(*cct_to_xy(dst_kelvin))
    s_lms = M_BRADFORD @ src
    d_lms = M_BRADFORD @ dst
    adapt = np.linalg.inv(M_BRADFORD) @ np.diag(d_lms / s_lms) @ M_BRADFORD
    m = np.linalg.inv(M_LIN_TO_XYZ.astype(np.float64)) @ adapt @ M_LIN_TO_XYZ.astype(np.float64)
    # normalise so that luminance of white is preserved
    white = m @ np.ones(3)
    m = m / (LUMA_709.astype(np.float64) @ white)
    return m.astype(F32)


def kelvin_to_rgb(kelvin: float) -> np.ndarray:
    """Approximate sRGB-encoded colour of a black-body light source (normalised)."""
    xyz = xy_to_xyz(*cct_to_xy(kelvin))
    lin = np.clip(M_XYZ_TO_LIN.astype(np.float64) @ xyz, 0, None)
    lin = lin / lin.max()
    return linear_to_srgb(lin.astype(F32))


def apply_matrix(rgb: np.ndarray, m: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
    out = rgb[..., :3] @ np.asarray(m, dtype=F32).T
    if offset is not None:
        out = out + np.asarray(offset, dtype=F32)
    return out.astype(F32)


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0 if e1 != e0 else 1e-6), 0, 1)
    return (t * t * (3 - 2 * t)).astype(F32)


def mix(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    return (a + (b - a) * t).astype(F32)


# ---------------------------------------------------------------------------
# Fast paths (numba, multi-threaded) for large float32 buffers; numpy otherwise.
# ---------------------------------------------------------------------------
_FAST_MIN = 4096
try:
    from . import fast as _fast
except Exception:  # pragma: no cover - numba unavailable
    _fast = None


def _use_fast(x) -> bool:
    return _fast is not None and isinstance(x, np.ndarray) and x.size >= _FAST_MIN


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return _fast.srgb_to_linear(x) if _use_fast(x) else _np_srgb_to_linear(x)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    return _fast.linear_to_srgb(x) if _use_fast(x) else _np_linear_to_srgb(x)


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    return _fast.rgb_to_oklab(rgb) if _use_fast(rgb) and rgb.shape[-1] >= 3 else _np_rgb_to_oklab(rgb)


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return _fast.oklab_to_rgb(lab) if _use_fast(lab) and lab.shape[-1] == 3 else _np_oklab_to_rgb(lab)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    return _fast.rgb_to_hsv(rgb) if _use_fast(rgb) and rgb.shape[-1] >= 3 else _np_rgb_to_hsv(rgb)


def matrix_in_linear(rgb: np.ndarray, m: np.ndarray, offset: float = 0.0, gain: float = 1.0) -> np.ndarray:
    """encode(M @ (decode(rgb) * gain) + offset) — fused white balance / exposure."""
    if _use_fast(rgb):
        return _fast.matrix_in_linear(rgb, m, offset, gain)
    lin = _np_srgb_to_linear(rgb[..., :3]) * gain
    return _np_linear_to_srgb(lin @ np.asarray(m, F32).T + offset)
