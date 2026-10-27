"""Private helpers for the ImageToolbox "Color" group port (see color.py).

Each helper mirrors a specific upstream implementation:
  * aire (MIT, awxkee) — C++ / Rust, and colorutils-rs / palette-transfer (awxkee) crates it uses,
  * Trickle (Apache-2.0, T8RIN) — C++,
  * GPUImage (BSD-3) — GLSL shaders (as forked in ImageToolboxLibs),
  * ImageToolbox (Apache-2.0) — procedural GLSL shaders, curves editor,
  * Skia / Android ``Color`` HSV conversions used by Trickle's palette generation.
"""
from __future__ import annotations

import math
from typing import Sequence

import numba
import numpy as np

F32 = np.float32


# ---------------------------------------------------------------------------
# sRGB transfer functions (each upstream library has slightly different constants)
# ---------------------------------------------------------------------------

def srgb_to_linear_045(x: np.ndarray) -> np.ndarray:
    """aire ``SRGBToLinear`` / Trickle ``SRGBToLinear`` (linear segment up to 0.045)."""
    x = np.asarray(x, F32)
    hi = np.power((np.maximum(x, 0.0) + 0.055) / 1.055, 2.4)
    return np.where(x <= 0.045, x / 12.92, hi).astype(F32)


def linear_to_srgb_std(x: np.ndarray) -> np.ndarray:
    """aire / Trickle ``LinearSRGBTosRGB``."""
    x = np.asarray(x, F32)
    hi = 1.055 * np.power(np.maximum(x, 0.0), 1.0 / 2.4) - 0.055
    return np.where(x <= 0.0031308, 12.92 * x, hi).astype(F32)


_CU_A = 0.0550107189475866
_CU_T = 0.0030412825601275209


def cu_srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """colorutils-rs ``srgb_to_linear`` (TransferFunction::Srgb)."""
    x = np.asarray(x, F32)
    hi = np.power((np.clip(x, 0.0, 1.0) + _CU_A) / (1.0 + _CU_A), 2.4)
    out = np.where(x < 12.92 * _CU_T, x / 12.92, hi)
    return np.clip(out, 0.0, 1.0).astype(F32)


def cu_linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """colorutils-rs ``srgb_from_linear`` (TransferFunction::Srgb)."""
    x = np.asarray(x, F32)
    hi = (1.0 + _CU_A) * np.power(np.clip(x, 0.0, 1.0), 1.0 / 2.4) - _CU_A
    out = np.where(x < _CU_T, x * 12.92, hi)
    return np.clip(out, 0.0, 1.0).astype(F32)


SRGB_TO_XYZ_D65 = np.array([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]], np.float64)
XYZ_TO_SRGB_D65 = np.array([[3.2404542, -1.5371385, -0.4985314],
                            [-0.9692660, 1.8760108, 0.0415560],
                            [0.0556434, -0.2040259, 1.0572252]], np.float64)


def _mat(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    return (img @ np.asarray(m, F32).T).astype(F32)


# ---------------------------------------------------------------------------
# Trickle CIELab (ColorUtils.cpp colorToLAB / labToColor)
# ---------------------------------------------------------------------------

_T0 = 6.0 / 29.0


def trickle_rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    xyz = _mat(srgb_to_linear_045(rgb) * 100.0, SRGB_TO_XYZ_D65)
    t = xyz / np.array([95.047, 100.0, 108.883], F32)
    f = np.where(t > _T0 ** 3, np.cbrt(t), (1.0 / 3.0) / (_T0 * _T0) * t + 4.0 / 29.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], -1).astype(F32)


def trickle_lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = lab[..., 1] / 500.0 + fy
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], -1)
    t = np.where(f > _T0, f * f * f, 3.0 * _T0 * _T0 * (f - 4.0 / 29.0))
    xyz = t * np.array([0.95047, 1.0, 1.08883], F32)
    return np.clip(linear_to_srgb_std(_mat(xyz, XYZ_TO_SRGB_D65)), 0.0, 1.0)


def trickle_transfer_palette(target: np.ndarray, reference: np.ndarray, intensity: float) -> np.ndarray:
    """Trickle ``transferPaletteImpl``: Reinhard mean/std transfer in CIELab.

    Upstream quirk kept: a channel whose target std is 0 is mapped to 0 (not left unchanged).
    """
    s_mean, s_std = channel_stats(chunked(trickle_rgb_to_lab, reference))
    lab = chunked(trickle_rgb_to_lab, target)
    t_mean, t_std = channel_stats(lab)

    def apply(px):
        out = np.empty_like(px)
        for c in range(3):
            if t_std[c] != 0.0:
                new = (px[..., c] - t_mean[c]) * (s_std[c] / t_std[c]) + s_mean[c]
            else:
                new = np.zeros_like(px[..., c])
            out[..., c] = px[..., c] + intensity * (new - px[..., c])
        return trickle_lab_to_rgb(out)

    return chunked(apply, lab)


# ---------------------------------------------------------------------------
# colorutils-rs perceptual spaces + palette-transfer crate (aire copyPalette)
# ---------------------------------------------------------------------------

_OK_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
_OK_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
_OK_M2_INV = np.array([[1.0, 0.3963377774, 0.2158037573],
                       [1.0, -0.1055613458, -0.0638541728],
                       [1.0, -0.0894841775, -1.2914855480]])
_OK_M1_INV = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                       [-1.2684380046, 2.6097574011, -0.3413193965],
                       [-0.0041960863, -0.7034186147, 1.7076147010]])

_D65 = (95.047, 100.0, 108.883)
_LUV_DEN = _D65[0] + 15.0 * _D65[1] + 3.0 * _D65[2]
_LUV_U_W = 4.0 * _D65[1] / _LUV_DEN
_LUV_V_W = 9.0 * _D65[1] / _LUV_DEN
_ISQ3, _ISQ6, _TSQ6, _ISQ2 = 0.57735026918962576, 0.40824829046386301, 0.816496580927726032, 0.7071067811865476


def lin_to_oklab(lin: np.ndarray) -> np.ndarray:
    return _mat(np.cbrt(_mat(lin, _OK_M1)), _OK_M2)


def oklab_to_lin(lab: np.ndarray) -> np.ndarray:
    lms = _mat(lab, _OK_M2_INV)
    return _mat(lms * lms * lms, _OK_M1_INV)


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    t = np.stack([xyz[..., 0] * 100.0 / 95.047, xyz[..., 1], xyz[..., 2] * 100.0 / 108.883], -1)
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], -1).astype(F32)


def _lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    y = (lab[..., 0] + 16.0) / 116.0
    x = lab[..., 1] / 500.0 + y
    z = y - lab[..., 2] / 200.0
    f = np.stack([x, y, z], -1)
    f3 = f * f * f
    t = np.where(f3 > 0.008856, f3, (f - 16.0 / 116.0) / 7.787)
    return (t * np.array(_D65, F32) / 100.0).astype(F32)


def _xyz_to_luv(xyz: np.ndarray) -> np.ndarray:
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    den = x + 15.0 * y + 3.0 * z
    L = np.where(y < (6.0 / 29.0) ** 3, (29.0 / 3.0) ** 3 * y, 116.0 * np.cbrt(y) - 16.0)
    L = np.clip(L, 0.0, 100.0)
    ok = den != 0
    sd = np.where(ok, den, 1.0)
    u = np.where(ok, 13.0 * L * (4.0 * x / sd - _LUV_U_W), 0.0)
    v = np.where(ok, 13.0 * L * (9.0 * y / sd - _LUV_V_W), 0.0)
    return np.stack([L, u, v], -1).astype(F32)


def _luv_to_xyz(luv: np.ndarray) -> np.ndarray:
    L, U, V = luv[..., 0], luv[..., 1], luv[..., 2]
    pos = L > 0
    l13 = 1.0 / (13.0 * np.where(pos, L, 1.0))
    u = U * l13 + _LUV_U_W
    v = V * l13 + _LUV_V_W
    y = np.where(L > 8.0, ((L + 16.0) / 116.0) ** 3, L * (3.0 / 29.0) ** 3)
    vok = v != 0
    den = 1.0 / (4.0 * np.where(vok, v, 1.0))
    x = np.where(vok, y * 9.0 * u * den, 0.0)
    z = np.where(vok, y * (12.0 - 3.0 * u - 20.0 * v) * den, 0.0)
    out = np.stack([x, y, z], -1)
    return np.where(pos[..., None], out, 0.0).astype(F32)


def _xyz_to_lalphabeta(xyz: np.ndarray) -> np.ndarray:
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lms = [0.3897 * x + 0.6890 * y - 0.0787 * z, -0.2298 * x + 1.1834 * y + 0.0464 * z, z]
    lp, mp, sp = (np.where(c > 0, np.log10(np.where(c > 0, c, 1.0)), 0.0) for c in lms)
    l = _ISQ3 * (lp + mp + sp)
    a = _ISQ6 * lp + _ISQ6 * mp - _TSQ6 * sp
    b = _ISQ2 * lp - _ISQ2 * mp
    return np.stack([l, a, b], -1).astype(F32)


def _lalphabeta_to_xyz(lab: np.ndarray) -> np.ndarray:
    la = lab[..., 0] * _ISQ3
    sa = _ISQ6 * lab[..., 1]
    pb = _ISQ2 * lab[..., 2]
    lp, mp, sp = la + sa + pb, la + sa - pb, la - _TSQ6 * lab[..., 1]
    l, m, s = (np.where(c == 0, 0.0, np.power(10.0, c)) for c in (lp, mp, sp))
    return np.stack([1.91024 * l - 1.11218 * m + 0.201941 * s,
                     0.370942 * l + 0.62905 * m + 5.13315e-6 * s, s], -1).astype(F32)


def to_perceptual(rgb: np.ndarray, space: str) -> np.ndarray:
    lin = cu_srgb_to_linear(rgb)
    if space == "oklab":
        return lin_to_oklab(lin)
    xyz = _mat(lin, SRGB_TO_XYZ_D65)
    return {"lab": _xyz_to_lab, "luv": _xyz_to_luv, "lalphabeta": _xyz_to_lalphabeta}[space](xyz)


def from_perceptual(p: np.ndarray, space: str) -> np.ndarray:
    if space == "oklab":
        lin = oklab_to_lin(p)
    else:
        xyz = {"lab": _lab_to_xyz, "luv": _luv_to_xyz, "lalphabeta": _lalphabeta_to_xyz}[space](p)
        lin = _mat(xyz, XYZ_TO_SRGB_D65)
    return cu_linear_to_srgb(np.clip(lin, 0.0, 1.0))


def aire_copy_palette(target: np.ndarray, reference: np.ndarray, intensity: float, space: str) -> np.ndarray:
    """palette-transfer 0.2 ``copy_palette_rgba`` (used by aire ``copyPalette``)."""
    s_mean, s_std = channel_stats(chunked(to_perceptual, reference, space))
    tgt = chunked(to_perceptual, target, space)
    t_mean, t_std = channel_stats(tgt)
    scales = []
    for c in range(3):
        scale = np.float32(s_std[c] / t_std[c]) if t_std[c] != 0 else np.float32(np.inf)
        scales.append(scale if (np.isfinite(scale) and scale != 0) else None)  # None: channel unchanged

    def apply(px):
        out = px.copy()
        for c, scale in enumerate(scales):
            if scale is not None:
                new = scale * (px[..., c] - np.float32(t_mean[c])) + np.float32(s_mean[c])
                out[..., c] = px[..., c] + intensity * (new - px[..., c])
        return from_perceptual(out, space)

    return chunked(apply, tgt)


def aces_hill(lin: np.ndarray) -> np.ndarray:
    """aire-filters ``aces_hill`` tone curve (on linear RGB)."""
    m_in = np.array([[0.59719, 0.35458, 0.04823], [0.07600, 0.90834, 0.01566], [0.02840, 0.13383, 0.83777]])
    m_out = np.array([[1.60475, -0.53108, -0.07367], [-0.10208, 1.10813, -0.00605], [-0.00327, -0.07276, 1.07602]])
    c = _mat(lin, m_in)
    a = c * (c + 0.0245786) - 0.000090537
    b = c * (0.983729 * c + 0.4329510) + 0.238081
    return _mat(a / b, m_out)


# ---------------------------------------------------------------------------
# GPUImageColorBalanceFilter shader (per-pixel kernel)
# ---------------------------------------------------------------------------

@numba.njit(cache=True, inline="always")
def _cb_shift(v, sh, mid, hi):
    a, b, scale = 0.25, 0.333, 0.7
    shadows = sh * min(max((v - b) / -a + 0.5, 0.0), 1.0) * scale
    midtones = mid * min(max((v - b) / a + 0.5, 0.0), 1.0) * min(max((v + b - 1.0) / -a + 0.5, 0.0), 1.0) * scale
    highlights = hi * min(max((v + b - 1.0) / a + 0.5, 0.0), 1.0) * scale
    return min(max(v + shadows + midtones + highlights, 0.0), 1.0)


@numba.njit(cache=True, inline="always")
def _hue_to_rgb(f1, f2, hue):
    if hue < 0.0:
        hue += 1.0
    elif hue > 1.0:
        hue -= 1.0
    if 6.0 * hue < 1.0:
        return f1 + (f2 - f1) * 6.0 * hue
    if 2.0 * hue < 1.0:
        return f2
    if 3.0 * hue < 2.0:
        return f1 + (f2 - f1) * (2.0 / 3.0 - hue) * 6.0
    return f1


@numba.njit(cache=True, parallel=True)
def gpuimage_color_balance(img, sh, mid, hi):
    """Shift by tonal range per channel, then RGBToHSL(new) with the old HSL lightness (preserveLuminosity)."""
    h, w = img.shape[0], img.shape[1]
    out = np.empty((h, w, 3), np.float32)
    for y in numba.prange(h):
        for x in range(w):
            r0, g0, b0 = img[y, x, 0], img[y, x, 1], img[y, x, 2]
            r = _cb_shift(r0, sh[0], mid[0], hi[0])
            g = _cb_shift(g0, sh[1], mid[1], hi[1])
            b = _cb_shift(b0, sh[2], mid[2], hi[2])
            fmin = min(r, g, b)
            fmax = max(r, g, b)
            delta = fmax - fmin
            hue = 0.0
            sat = 0.0
            if delta != 0.0:
                if (fmax + fmin) / 2.0 < 0.5:
                    sat = delta / (fmax + fmin)
                else:
                    sat = delta / (2.0 - fmax - fmin)
                dr = ((fmax - r) / 6.0 + delta / 2.0) / delta
                dg = ((fmax - g) / 6.0 + delta / 2.0) / delta
                db = ((fmax - b) / 6.0 + delta / 2.0) / delta
                if r == fmax:
                    hue = db - dg
                elif g == fmax:
                    hue = 1.0 / 3.0 + dr - db
                else:
                    hue = 2.0 / 3.0 + dg - dr
                if hue < 0.0:
                    hue += 1.0
                elif hue > 1.0:
                    hue -= 1.0
            lum = (max(r0, g0, b0) + min(r0, g0, b0)) / 2.0
            if sat == 0.0:
                out[y, x, 0] = lum
                out[y, x, 1] = lum
                out[y, x, 2] = lum
            else:
                f2 = lum * (1.0 + sat) if lum < 0.5 else (lum + sat) - sat * lum
                f1 = 2.0 * lum - f2
                out[y, x, 0] = _hue_to_rgb(f1, f2, hue + 1.0 / 3.0)
                out[y, x, 1] = _hue_to_rgb(f1, f2, hue)
                out[y, x, 2] = _hue_to_rgb(f1, f2, hue - 1.0 / 3.0)
    return out


# ---------------------------------------------------------------------------
# ImageToolbox procedural shader HSV (ProceduralShaders.kt COMMON_SHADER)
# ---------------------------------------------------------------------------

def shader_rgb_to_hsv(c: np.ndarray) -> np.ndarray:
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    high = np.maximum(np.maximum(r, g), b)
    low = np.minimum(np.minimum(r, g), b)
    delta = high - low
    big = delta > 0.00001
    sd = np.where(big, delta, 1.0)
    hue = np.where(high == r, np.mod((g - b) / sd, 6.0), np.where(high == g, (b - r) / sd + 2.0, (r - g) / sd + 4.0))
    hue = np.where(big, hue, 0.0) / 6.0
    hue = hue - np.floor(hue)
    sat = np.where(high > 0.00001, delta / np.where(high > 0.00001, high, 1.0), 0.0)
    return np.stack([hue, sat, high], -1).astype(F32)


def shader_hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h = hsv[..., 0:1] + np.array([0.0, 0.6666667, 0.3333333], F32)
    wave = np.clip(np.abs((h - np.floor(h)) * 6.0 - 3.0) - 1.0, 0.0, 1.0)
    return (hsv[..., 2:3] * (1.0 + (wave - 1.0) * hsv[..., 1:2])).astype(F32)


def procedural_field(h: int, w: int, offset_x: float, offset_y: float, scale: float,
                     rotation: float, rows: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``toField(textureCoordinate)`` of the ImageToolbox procedural shaders (y grows downwards).

    ``rows`` restricts the result to image rows [r0, r1).
    """
    short = max(1, min(w, h))
    ax, ay = w / short, h / short
    r0, r1 = rows or (0, h)
    u = ((np.arange(w, dtype=F32) + 0.5) / w - 0.5 - offset_x) * ax
    v = ((np.arange(r0, r1, dtype=F32) + 0.5) / h - 0.5 - offset_y) * ay
    U, V = np.meshgrid(u, v)
    ang = -math.radians(rotation)
    c, s = math.cos(ang), math.sin(ang)
    return ((c * U - s * V) / scale).astype(F32), ((s * U + c * V) / scale).astype(F32)


# ---------------------------------------------------------------------------
# GPUImage tone curve (GPUImageToneCurveFilter + ImageToolbox curves normalisation)
# ---------------------------------------------------------------------------

CURVE_MAX_POINTS = 16  # ImageToolbox CurvesValue.MaxPointCount


def itb_normalize_curve(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """ImageToolbox ``CurvesValue.replacePoints``: clamp, sort, unique x, pin ends to x=0 / x=1."""
    pts = sorted(((min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))) for x, y in points),
                 key=lambda p: p[0])
    uniq: list[list[float]] = []
    seen: set[float] = set()
    for x, y in pts:
        if x not in seen:
            seen.add(x)
            uniq.append([x, y])
    if len(uniq) < 2:
        return [(0.0, 0.0), (1.0, 1.0)]
    uniq[0][0] = 0.0
    uniq[-1][0] = 1.0
    if len(uniq) > CURVE_MAX_POINTS:
        uniq = uniq[:CURVE_MAX_POINTS - 1] + [uniq[-1]]
    return [(p[0], p[1]) for p in uniq]


def curve_is_default(points: Sequence[Sequence[float]]) -> bool:
    return all(abs(y - x) < 0.00001 for x, y in points)


def _second_derivative(pts: list[tuple[int, int]]) -> list[float]:
    n = len(pts)
    m = [[0.0, 0.0, 0.0] for _ in range(n)]
    res = [0.0] * n
    m[0][1] = 1.0
    for i in range(1, n - 1):
        (x1, y1), (x2, y2), (x3, y3) = pts[i - 1], pts[i], pts[i + 1]
        m[i][0] = (x2 - x1) / 6.0
        m[i][1] = (x3 - x1) / 3.0
        m[i][2] = (x3 - x2) / 6.0
        res[i] = (y3 - y2) / (x3 - x2) - (y2 - y1) / (x2 - x1)
    m[n - 1][1] = 1.0
    for i in range(1, n):
        k = m[i][0] / m[i - 1][1]
        m[i][1] -= k * m[i - 1][2]
        m[i][0] = 0.0
        res[i] -= k * res[i - 1]
    for i in range(n - 2, -1, -1):
        k = m[i][2] / m[i + 1][1]
        m[i][1] -= k * m[i + 1][0]
        m[i][2] = 0.0
        res[i] -= k * res[i + 1]
    return [res[i] / m[i][1] for i in range(n)]


def gpuimage_prepared_curve(points: Sequence[Sequence[float]]) -> np.ndarray:
    """``getPreparedSplineCurve``: 256 signed offsets (curve(x) - x) in 0..255 units."""
    conv: list[tuple[int, int]] = []
    for x, y in sorted(points, key=lambda p: p[0]):
        xi, yi = int(x * 255), int(y * 255)
        if conv and conv[-1][0] == xi:  # zero-width segment would make the spline NaN
            conv[-1] = (xi, yi)
        else:
            conv.append((xi, yi))
    if len(conv) < 2:
        return np.zeros(256, np.float64)
    sd = _second_derivative(conv)
    out: list[tuple[int, int]] = []
    for i in range(len(conv) - 1):
        (cx, cy), (nx, ny) = conv[i], conv[i + 1]
        h = nx - cx
        for x in range(cx, nx):
            t = (x - cx) / h
            a, b = 1.0 - t, t
            y = a * cy + b * ny + (h * h / 6.0) * ((a * a * a - a) * sd[i] + (b * b * b - b) * sd[i + 1])
            out.append((x, int(math.floor(min(255.0, max(0.0, y)) + 0.5))))
    if len(out) == 255:
        out.append(conv[-1])
    if out and out[0][0] > 0:
        out = [(i, 0) for i in range(0, out[0][0] + 1)] + out
    if out and out[-1][0] < 255:
        out = out + [(i, 255) for i in range(out[-1][0] + 1, 256)]
    prepared = np.array([y - x for x, y in out[:256]], np.float64)
    if len(prepared) < 256:
        prepared = np.concatenate([prepared, np.zeros(256 - len(prepared))])
    return prepared


def sample_curve_texture(c: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """``texture2D`` of a 256x1 GL_LINEAR / CLAMP_TO_EDGE byte texture at u=c, returned in 0..1."""
    t = np.clip(c * 256.0 - 0.5, 0.0, 255.0)
    i0 = np.minimum(np.floor(t).astype(np.int32), 254)
    f = (t - i0).astype(F32)
    L = (lut / 255.0).astype(F32)
    return (L[i0] * (1.0 - f) + L[i0 + 1] * f).astype(F32)


# ---------------------------------------------------------------------------
# Android / Skia HSV <-> colour int (Trickle TrickleUtils.generateShades)
# ---------------------------------------------------------------------------

def sk_rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    mn, mx = min(r, g, b), max(r, g, b)
    delta = mx - mn
    v = mx / 255.0
    if delta == 0:
        return 0.0, 0.0, v
    s = delta / mx
    if r == mx:
        h = (g - b) / delta
    elif g == mx:
        h = 2.0 + (b - r) / delta
    else:
        h = 4.0 + (r - g) / delta
    h *= 60.0
    if h < 0:
        h += 360.0
    return h, s, v


def sk_hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    def rnd(x: float) -> int:
        return int(math.floor(x + 0.5))

    s = min(1.0, max(0.0, s))
    v = min(1.0, max(0.0, v))
    vb = rnd(v * 255)
    if abs(s) <= 1.0 / 4096:
        return vb, vb, vb
    hx = 0.0 if (h < 0 or h >= 360) else h / 60.0
    w = math.floor(hx)
    f = hx - w
    p = rnd((1 - s) * v * 255)
    q = rnd((1 - s * f) * v * 255)
    t = rnd((1 - s * (1 - f)) * v * 255)
    return [(vb, t, p), (q, vb, p), (p, vb, t), (p, q, vb), (t, p, vb), (vb, p, q)][min(int(w), 5)]


@numba.njit(cache=True, parallel=True)
def nearest_palette(rgb255: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """Per-pixel nearest palette colour (Euclidean RGB, first minimum wins)."""
    n = rgb255.shape[0]
    k = pal.shape[0]
    out = np.empty_like(rgb255)
    for i in numba.prange(n):
        r, g, b = rgb255[i, 0], rgb255[i, 1], rgb255[i, 2]
        best = 1e30
        bi = 0
        for j in range(k):
            dr = r - pal[j, 0]
            dg = g - pal[j, 1]
            db = b - pal[j, 2]
            d = dr * dr + dg * dg + db * db
            if d < best:
                best = d
                bi = j
        out[i, 0] = pal[bi, 0]
        out[i, 1] = pal[bi, 1]
        out[i, 2] = pal[bi, 2]
    return out


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

CHUNK_PIXELS = 1 << 18


def chunked(fn, img: np.ndarray, *args, with_y: bool = False, **kw) -> np.ndarray:
    """Apply a per-pixel ``fn(rows, *args)`` over horizontal bands (keeps temporaries small).

    With ``with_y`` the band's first row index is passed as ``y0=``.
    """
    h, w = img.shape[:2]
    step = max(1, CHUNK_PIXELS // max(1, w))
    out = None
    for y0 in range(0, h, step):
        r = fn(img[y0:y0 + step], *args, **({"y0": y0} if with_y else {}), **kw)
        if out is None:
            out = np.empty((h, w) + r.shape[2:], F32)
        out[y0:y0 + step] = r
    return out


def channel_stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and population std (float64, two-pass, chunked)."""
    flat = arr.reshape(-1, arr.shape[-1])
    mean = flat.mean(0, dtype=np.float64)
    acc = np.zeros(flat.shape[1], np.float64)
    for i in range(0, flat.shape[0], CHUNK_PIXELS):
        d = flat[i:i + CHUNK_PIXELS].astype(np.float64) - mean
        acc += (d * d).sum(0)
    return mean, np.sqrt(acc / max(1, flat.shape[0]))


def to8(c: Sequence[float]) -> list[int]:
    """Colour param floats (0..1) -> Android 8-bit channel ints."""
    return [int(round(float(v) * 255.0)) for v in c]
