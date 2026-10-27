"""Photographic tone & colour adjustments (category ``adjust``).

These are the precise, composable controls an agent uses to realise a
creative intent ("warm, faded, muted greens...").  Slider conventions follow
mainstream editors: most amounts are -100..100 with 0 = no change.
"""
from __future__ import annotations

import numpy as np
import cv2

from ..core.color import (
    linear_to_srgb, srgb_to_linear, white_balance_matrix, rgb_to_oklab, oklab_to_rgb, rgb_to_hsv,
    LUMA_709, lin_to_oklab, oklab_to_lin, matrix_in_linear,
)
from ..core.params import Bool, Choice, Color, File, Float, FloatList, Int, Points
from ..core.registry import op
from ..core.errors import invalid
from ._util import (
    F32, apply_1d_lut, apply_lut3d, gaussian, guided_filter, hue_to_ab, load_cube, lut_from_hald_or_square,
    luma, monotone_curve, smooth_curve, zone_weights, box,
)

A = "adjust"


def _amt(name: str, title: str, desc: str = "", lo: float = -100, hi: float = 100, default: float = 0.0) -> Float:
    return Float(name, default, min=lo, max=hi, title=title, description=desc, step=1)


# ---------------------------------------------------------------------------
@op("adjust.exposure", "Exposure", params=[
    Float("ev", 0.0, min=-5, max=5, step=0.05, unit="EV", title="Exposure",
          description="Exposure compensation in stops (linear light, x2 per stop)"),
    Float("offset", 0.0, min=-0.2, max=0.2, step=0.005, title="Offset", description="Linear black offset"),
    Float("gamma", 1.0, min=0.2, max=5, step=0.01, title="Gamma correction", advanced=True),
], tags=["brightness", "stops", "ev", "light", "tone"], examples=[{"ev": 0.5}, {"ev": -1}])
def exposure(img, ctx, ev, offset, gamma):
    """Photographic exposure compensation performed in linear light."""
    out = matrix_in_linear(img, np.eye(3, dtype=F32), offset=offset, gain=2.0 ** ev)
    if gamma != 1.0:
        out = np.sign(out) * np.abs(out) ** (1.0 / gamma)
    return out


@op("adjust.white_balance", "White balance", params=[
    _amt("temperature", "Temperature", "Warm (+) / cool (-) shift; ±100 ≈ ±75 mired"),
    _amt("tint", "Tint", "Magenta (+) / green (-) shift"),
], tags=["temperature", "tint", "warm", "cool", "wb", "color cast"], examples=[{"temperature": 25, "tint": 5}])
def white_balance(img, ctx, temperature, tint):
    """Relative white balance using a Bradford chromatic adaptation in linear light."""
    if temperature == 0 and tint == 0:
        return img
    d65_mired = 1e6 / 6504.0
    src_k = 1e6 / max(20.0, d65_mired - temperature * 0.75)
    m = white_balance_matrix(src_k, 6504.0, tint=-tint * 0.035)
    return matrix_in_linear(img, m)


def neutral_white_balance(rgb) -> tuple[float, float]:
    """Temperature/tint (adjust.white_balance units) that make the sRGB colour ``rgb`` neutral grey."""
    from scipy.optimize import minimize

    px = np.asarray(rgb, F32).reshape(1, 1, 3)

    def chroma(v):
        t, ti = np.clip(v, -100, 100)
        lab = rgb_to_oklab(np.clip(white_balance(px, None, float(t), float(ti)), 0, 1))[0, 0]
        return float(lab[1] ** 2 + lab[2] ** 2)

    best = minimize(chroma, x0=[0.0, 0.0], method="Nelder-Mead",
                    options={"xatol": 0.05, "fatol": 1e-10, "initial_simplex": [[0, 0], [20, 0], [0, 20]]})
    t, ti = np.clip(best.x, -100, 100)
    return round(float(t), 1), round(float(ti), 1)


@op("adjust.white_balance_kelvin", "White balance (Kelvin)", params=[
    Float("source_kelvin", 6500, min=1700, max=20000, step=50, unit="K", title="Scene illuminant",
          description="Colour temperature of the light in the scene"),
    Float("target_kelvin", 6500, min=1700, max=20000, step=50, unit="K", title="Target",
          description="Temperature it should look like (6500 = neutral daylight)"),
    _amt("tint", "Tint"),
], tags=["temperature", "kelvin", "wb"])
def white_balance_kelvin(img, ctx, source_kelvin, target_kelvin, tint):
    """Absolute white balance: adapt from a scene illuminant to a target illuminant."""
    m = white_balance_matrix(source_kelvin, target_kelvin, tint=-tint * 0.035)
    return matrix_in_linear(img, m)


@op("adjust.auto_white_balance", "Auto white balance", params=[
    Choice("method", "gray_world", choices=["gray_world", "white_patch", "shades_of_gray"]),
    Float("strength", 1.0, min=0, max=1, step=0.01),
], tags=["auto", "wb", "neutralize", "cast removal"])
def auto_white_balance(img, ctx, method, strength):
    """Neutralise colour casts automatically (computed in linear light)."""
    lin = srgb_to_linear(np.clip(img, 0, 1))
    flat = lin.reshape(-1, 3)
    if method == "white_patch":
        ref = np.percentile(flat, 99.0, axis=0)
    elif method == "shades_of_gray":
        ref = np.power(np.mean(np.power(flat, 6), axis=0), 1 / 6)
    else:
        ref = flat.mean(0)
    ref = np.maximum(ref, 1e-4)
    gains = ref.mean() / ref
    gains = 1 + (gains - 1) * strength
    return linear_to_srgb(lin * gains.astype(F32))


def _sigmoid_curve(k: float, pivot: float, n: int = 4096) -> np.ndarray:
    x = np.linspace(0, 1, n, dtype=np.float64)
    p = float(np.clip(pivot, 0.05, 0.95))
    e = np.log(0.5) / np.log(p)
    xp = x ** e
    g = 2.0 ** (k * 1.6)
    y = xp ** g / (xp ** g + (1 - xp) ** g + 1e-12)
    return (y ** (1 / e)).astype(F32)


@op("adjust.contrast", "Contrast", params=[
    _amt("amount", "Contrast", "S-curve strength; negative values flatten (soft contrast)"),
    Float("pivot", 0.5, min=0.1, max=0.9, step=0.01, title="Pivot", description="Tone that stays fixed"),
    Bool("preserve_saturation", True, title="Preserve saturation",
         description="Apply on luminance only so contrast does not change saturation"),
], tags=["s-curve", "punch", "flat", "soft contrast", "tone"])
def contrast(img, ctx, amount, pivot, preserve_saturation):
    """Smooth S-curve contrast that keeps black and white fixed (no clipping)."""
    if amount == 0:
        return img
    curve = _sigmoid_curve(amount / 100.0, pivot)
    if preserve_saturation:
        y = luma(img)
        ny = apply_1d_lut(y, curve)
        return _scale_to_luma(img, y, ny)
    return apply_1d_lut(img, curve)


def _scale_to_luma(img: np.ndarray, y: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """Shift colours to a new luma in a hue/chroma-preserving way (additive in bright, ratio in dark)."""
    ratio = ny / np.maximum(y, 1e-4)
    scaled = img * ratio[..., None]
    shifted = img + (ny - y)[..., None]
    w = np.clip(y * 2, 0, 1)[..., None]  # ratio scaling in shadows, additive shift in highlights
    return (scaled * (1 - w) + shifted * w).astype(F32)


@op("adjust.brightness", "Brightness", params=[
    _amt("amount", "Brightness", "Midtone brightness (gamma); endpoints stay fixed"),
], tags=["midtones", "lighten", "darken", "gamma"])
def brightness(img, ctx, amount):
    """Midtone brightness via a gamma curve on luminance."""
    if amount == 0:
        return img
    g = 2.0 ** (-amount / 100.0 * 1.2)
    y = np.clip(luma(img), 0, None)
    ny = np.power(np.clip(y, 0, 1), g) + np.maximum(y - 1, 0)
    return _scale_to_luma(img, y, ny)


@op("adjust.gamma", "Gamma", params=[Float("gamma", 1.0, min=0.1, max=5.0, step=0.01)],
    tags=["gamma", "midtones"])
def gamma_op(img, ctx, gamma):
    """Per-channel power curve out = in^(1/gamma)."""
    return (np.sign(img) * np.abs(img) ** (1.0 / gamma)).astype(F32)


@op("adjust.tone", "Tone (highlights / shadows / whites / blacks)", params=[
    _amt("highlights", "Highlights", "Recover (-) or boost (+) bright areas"),
    _amt("shadows", "Shadows", "Lift (+) or deepen (-) dark areas"),
    _amt("whites", "Whites", "Move the white point"),
    _amt("blacks", "Blacks", "Move the black point"),
    Float("radius", 30, min=0, max=500, unit="px", title="Local radius",
          description="Edge-aware neighbourhood used to decide what is shadow/highlight (0 = global)",
          advanced=True),
], tags=["highlights", "shadows", "recovery", "whites", "blacks", "hdr", "tone"],
    examples=[{"highlights": -40, "shadows": 30}])
def tone(img, ctx, highlights, shadows, whites, blacks, radius):
    """Lightroom-style basic tone controls with edge-aware zone detection."""
    y = np.clip(luma(img), 0, 1)
    if radius > 0:
        r = ctx.ipx(radius)
        base = guided_filter(y, y, r, eps=0.01)
    else:
        base = y
    ny = y.copy()
    if shadows:
        k = shadows / 100.0
        w = np.clip(1 - base / 0.55, 0, 1) ** 2
        ny = ny + k * 0.5 * w * (np.sqrt(np.clip(ny, 0, 1)) - ny + 0.08 * (1 - ny)) * (2 if k > 0 else 1.5)
    if highlights:
        k = highlights / 100.0
        w = np.clip((base - 0.45) / 0.55, 0, 1) ** 2
        if k < 0:
            ny = ny + k * 0.6 * w * (ny - np.clip(ny, 0, None) ** 2.2 * 0.9)
        else:
            ny = ny + k * 0.5 * w * (1 - ny) * ny
    if whites:
        k = whites / 100.0
        ny = ny + k * 0.25 * np.clip(ny, 0, 1) ** 3
    if blacks:
        k = blacks / 100.0
        ny = ny + k * 0.12 * (1 - np.clip(ny, 0, 1)) ** 4
    return _scale_to_luma(img, y, ny)


@op("adjust.levels", "Levels", params=[
    Float("in_black", 0.0, min=0, max=1, step=0.005, title="Input black"),
    Float("in_white", 1.0, min=0, max=1, step=0.005, title="Input white"),
    Float("gamma", 1.0, min=0.1, max=10, step=0.01, title="Midtone gamma"),
    Float("out_black", 0.0, min=0, max=1, step=0.005, title="Output black"),
    Float("out_white", 1.0, min=0, max=1, step=0.005, title="Output white"),
    Choice("channel", "rgb", choices=["rgb", "red", "green", "blue"]),
], tags=["black point", "white point", "range", "fade"])
def levels(img, ctx, in_black, in_white, gamma, out_black, out_white, channel):
    """Classic levels: input range, midtone gamma, output range."""
    if in_white <= in_black:
        raise invalid("in_white must be greater than in_black")
    def f(x):
        t = np.clip((x - in_black) / (in_white - in_black), 0, 1)
        t = t ** (1.0 / gamma)
        return out_black + t * (out_white - out_black)
    out = img.copy()
    if channel == "rgb":
        return f(img).astype(F32)
    c = ["red", "green", "blue"].index(channel)
    out[..., c] = f(img[..., c])
    return out


@op("adjust.curves", "Curves", params=[
    Points("rgb", [[0, 0], [1, 1]], min_items=2, max_items=32, min=0, max=1, title="RGB (master)"),
    Points("red", [[0, 0], [1, 1]], min_items=2, max_items=32, min=0, max=1),
    Points("green", [[0, 0], [1, 1]], min_items=2, max_items=32, min=0, max=1),
    Points("blue", [[0, 0], [1, 1]], min_items=2, max_items=32, min=0, max=1),
    Points("luminance", [[0, 0], [1, 1]], min_items=2, max_items=32, min=0, max=1,
           description="Curve on luminance only (keeps colours)"),
    Choice("interpolation", "monotone", choices=["monotone", "smooth", "linear"]),
], tags=["tone curve", "fade", "matte", "s-curve", "contrast", "cross process"],
    examples=[{"rgb": [[0, 0.08], [0.25, 0.22], [0.75, 0.8], [1, 0.95]]},
              {"red": [[0, 0], [0.5, 0.55], [1, 1]], "blue": [[0, 0.1], [1, 0.9]]}])
def curves(img, ctx, rgb, red, green, blue, luminance, interpolation):
    """Tone curves per channel with monotone cubic interpolation (input/output 0..1)."""
    def mk(pts):
        if interpolation == "linear":
            xs, ys = zip(*sorted(pts))
            return np.interp(np.linspace(0, 1, 4096), xs, ys).astype(F32)
        return (smooth_curve if interpolation == "smooth" else monotone_curve)(pts)

    ident = [(0.0, 0.0), (1.0, 1.0)]
    out = img
    if list(map(tuple, rgb)) != ident:
        out = apply_1d_lut(out, mk(rgb))
    chans = [red, green, blue]
    if any(list(map(tuple, c)) != ident for c in chans):
        out = out.copy()
        for i, c in enumerate(chans):
            if list(map(tuple, c)) != ident:
                out[..., i] = apply_1d_lut(out[..., i], mk(c))
    if list(map(tuple, luminance)) != ident:
        y = luma(out)
        out = _scale_to_luma(out, y, apply_1d_lut(y, mk(luminance)))
    return out


# ---------------------------------------------------------------------------
# Saturation / HSL
# ---------------------------------------------------------------------------
def _chroma_scale(img: np.ndarray, factor: np.ndarray | float) -> np.ndarray:
    lab = rgb_to_oklab(img)
    lab[..., 1:] *= np.asarray(factor, F32)[..., None] if np.ndim(factor) else factor
    return oklab_to_rgb(lab)


@op("adjust.saturation", "Saturation & vibrance", params=[
    _amt("saturation", "Saturation", "Uniform chroma change (-100 = grayscale)"),
    _amt("vibrance", "Vibrance", "Boosts muted colours more than saturated ones; protects skin"),
], tags=["saturation", "vibrance", "muted", "desaturate", "colourful", "chroma"])
def saturation(img, ctx, saturation, vibrance):
    """Perceptual (OKLab) saturation and vibrance."""
    if saturation == 0 and vibrance == 0:
        return img
    lab = rgb_to_oklab(img)
    C = np.hypot(lab[..., 1], lab[..., 2])
    f = np.ones(C.shape, F32) * (1 + saturation / 100.0)
    if vibrance:
        v = vibrance / 100.0
        muted = np.clip(1 - C / 0.25, 0, 1)
        hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
        skin = np.clip(1 - np.abs(hue - 55) / 35, 0, 1) * 0.6  # OKLab hue of typical skin ~ 40..70°
        f = f * (1 + v * muted * (1 - skin * (v > 0)))
    lab[..., 1] *= f
    lab[..., 2] *= f
    return oklab_to_rgb(lab)


HSL_BANDS = {  # name: centre hue (HSV degrees)
    "red": 0, "orange": 30, "yellow": 60, "green": 120, "aqua": 180, "blue": 225, "purple": 270, "magenta": 315,
}


def _hsl_params():
    ps = []
    for band in HSL_BANDS:
        ps.append(_amt(f"{band}_hue", f"{band.capitalize()} hue", f"Shift hue of {band}s (±100 ≈ ±30°)"))
        ps.append(_amt(f"{band}_saturation", f"{band.capitalize()} saturation"))
        ps.append(_amt(f"{band}_luminance", f"{band.capitalize()} luminance"))
    return ps


@op("adjust.hsl", "HSL / colour mixer", params=_hsl_params(),
    tags=["hsl", "color mixer", "muted greens", "selective", "hue", "skin", "sky"],
    examples=[{"green_saturation": -40, "green_hue": -15, "orange_saturation": 10}])
def hsl(img, ctx, **kw):
    """Per-colour hue / saturation / luminance mixer (8 bands, smooth overlap, OKLab)."""
    active = {b: (kw[f"{b}_hue"], kw[f"{b}_saturation"], kw[f"{b}_luminance"]) for b in HSL_BANDS}
    active = {b: v for b, v in active.items() if any(v)}
    if not active:
        return img
    hsv = rgb_to_hsv(np.clip(img, 0, 1))
    # Evaluate the smooth band windows once on a 1-degree hue axis, then gather per pixel.
    axis = np.arange(3600, dtype=F32) / 10.0
    dh_t = np.zeros(3600, F32)
    sf_t = np.zeros(3600, F32)
    dl_t = np.zeros(3600, F32)
    centers = list(HSL_BANDS.values())
    names = list(HSL_BANDS)
    for name, (hh, ss, ll) in active.items():
        i = names.index(name)
        c = centers[i]
        prev_c = centers[i - 1] - (360 if i == 0 else 0)
        next_c = centers[(i + 1) % len(centers)] + (360 if i == len(centers) - 1 else 0)
        d = (axis - c + 180) % 360 - 180
        width = np.where(d < 0, c - prev_c, next_c - c)
        w = np.clip(1 - np.abs(d) / width, 0, 1)
        w = w * w * (3 - 2 * w)
        dh_t += w * np.radians(hh / 100.0 * 30.0)
        sf_t += w * (ss / 100.0)
        dl_t += w * (ll / 100.0) * 0.25
    idx = np.minimum((hsv[..., 0] * 3600).astype(np.int32), 3599)
    sat_w = np.clip(hsv[..., 1] * 4, 0, 1)  # protect neutrals
    lab = rgb_to_oklab(img)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    H = np.arctan2(b, a) + dh_t[idx] * sat_w
    C = C * np.clip(1 + sf_t[idx] * sat_w, 0, None)
    L = L + dl_t[idx] * sat_w * np.clip(C * 6, 0, 1)
    out = np.stack([L, C * np.cos(H), C * np.sin(H)], -1)
    return oklab_to_rgb(out)


@op("adjust.hue_shift", "Hue shift", params=[Float("degrees", 0, min=-180, max=180, step=1, unit="deg")],
    tags=["hue", "rotate", "color shift"])
def hue_shift(img, ctx, degrees):
    """Rotate all hues (OKLab hue rotation, luminance preserved)."""
    lab = rgb_to_oklab(img)
    th = np.radians(degrees)
    a = lab[..., 1] * np.cos(th) - lab[..., 2] * np.sin(th)
    b = lab[..., 1] * np.sin(th) + lab[..., 2] * np.cos(th)
    lab[..., 1], lab[..., 2] = a, b
    return oklab_to_rgb(lab)


# ---------------------------------------------------------------------------
# Colour balance / grading / split toning
# ---------------------------------------------------------------------------
def _cb_params():
    ps = []
    for zone in ("shadows", "midtones", "highlights"):
        ps += [_amt(f"{zone}_cyan_red", f"{zone.capitalize()} cyan↔red"),
               _amt(f"{zone}_magenta_green", f"{zone.capitalize()} magenta↔green"),
               _amt(f"{zone}_yellow_blue", f"{zone.capitalize()} yellow↔blue")]
    ps.append(Bool("preserve_luminosity", True))
    return ps


@op("adjust.color_balance", "Colour balance", params=_cb_params(),
    tags=["color balance", "shadows tint", "highlights tint", "cast", "grading"])
def color_balance(img, ctx, preserve_luminosity, **kw):
    """Photoshop-style colour balance per tonal zone."""
    y = np.clip(luma(img), 0, 1)
    sh, mid, hi = zone_weights(y)
    out = img.copy()
    for zone, w in (("shadows", sh), ("midtones", mid), ("highlights", hi)):
        cr, mg, yb = kw[f"{zone}_cyan_red"], kw[f"{zone}_magenta_green"], kw[f"{zone}_yellow_blue"]
        if not (cr or mg or yb):
            continue
        delta = np.array([cr, mg, yb], F32) / 100.0 * 0.25
        out = out + w[..., None] * delta
    if preserve_luminosity:
        out = _scale_to_luma(out, luma(out), y)
    return out.astype(F32)


def _wheel(prefix: str, title: str):
    return [Float(f"{prefix}_hue", 0, min=0, max=360, step=1, unit="deg", title=f"{title} hue"),
            Float(f"{prefix}_saturation", 0, min=0, max=100, step=1, title=f"{title} saturation"),
            _amt(f"{prefix}_luminance", f"{title} luminance")]


@op("adjust.color_grading", "Colour grading wheels", params=[
    *_wheel("shadows", "Shadows"), *_wheel("midtones", "Midtones"), *_wheel("highlights", "Highlights"),
    *_wheel("global", "Global"),
    Float("blending", 50, min=0, max=100, step=1, title="Blending", description="Overlap between zones"),
    _amt("balance", "Balance", "Shift the shadows/highlights split point"),
], tags=["color grading", "teal and orange", "cinematic", "split tone", "wheels", "look"],
    examples=[{"shadows_hue": 200, "shadows_saturation": 35, "highlights_hue": 35, "highlights_saturation": 30}])
def color_grading(img, ctx, blending, balance, **kw):
    """Three-way (+global) colour wheels in OKLab: hue/saturation tint and luminance per zone."""
    lab = rgb_to_oklab(img)
    L = np.clip(lab[..., 0], 0, 1)
    sh, mid, hi = zone_weights(L, balance / 100.0, blending / 100.0)
    for zone, w in (("shadows", sh), ("midtones", mid), ("highlights", hi), ("global", None)):
        hue, sat, lum = kw[f"{zone}_hue"], kw[f"{zone}_saturation"], kw[f"{zone}_luminance"]
        if not (sat or lum):
            continue
        da, db = hue_to_ab(hue, sat / 100.0 * 0.08)
        ww = 1.0 if w is None else w
        lab[..., 1] += ww * da
        lab[..., 2] += ww * db
        lab[..., 0] += ww * lum / 100.0 * 0.15
    return oklab_to_rgb(lab)


@op("adjust.split_toning", "Split toning", params=[
    Float("highlights_hue", 45, min=0, max=360, unit="deg"), Float("highlights_saturation", 0, min=0, max=100),
    Float("shadows_hue", 220, min=0, max=360, unit="deg"), Float("shadows_saturation", 0, min=0, max=100),
    _amt("balance", "Balance"),
], tags=["split tone", "toning", "vintage", "warm highlights", "cool shadows"])
def split_toning(img, ctx, highlights_hue, highlights_saturation, shadows_hue, shadows_saturation, balance):
    """Classic split toning: tint highlights and shadows with different hues."""
    return color_grading(img, ctx, blending=60, balance=balance,
                         shadows_hue=shadows_hue, shadows_saturation=shadows_saturation, shadows_luminance=0,
                         midtones_hue=0, midtones_saturation=0, midtones_luminance=0,
                         highlights_hue=highlights_hue, highlights_saturation=highlights_saturation,
                         highlights_luminance=0, global_hue=0, global_saturation=0, global_luminance=0)


@op("adjust.fade", "Fade / matte", params=[
    Float("amount", 0, min=0, max=100, title="Fade", description="Lift blacks (matte / faded film look)"),
    Float("highlight_rolloff", 0, min=0, max=100, title="Highlight roll-off",
          description="Lower and soften the whites"),
    Color("shadow_tint", "#000000", title="Shadow tint", description="Colour of lifted blacks (black = neutral)"),
], tags=["fade", "matte", "faded blacks", "vintage", "film", "retro", "washed"])
def fade(img, ctx, amount, highlight_rolloff, shadow_tint):
    """Lift the black point (with optional tint) and soften highlights for a faded film look."""
    lift = amount / 100.0 * 0.25
    roll = highlight_rolloff / 100.0 * 0.15
    tint = np.array(shadow_tint[:3], F32)
    floor = lift * (0.5 + tint) if tint.any() else np.full(3, lift, F32)
    x = np.clip(img, 0, None)
    # soft knee: compress the top end smoothly
    top = 1 - roll
    out = floor + x * (top - floor)
    if roll:
        out = out - roll * 0.5 * np.clip(x - 0.6, 0, None) ** 2
    return out.astype(F32)


@op("adjust.channel_mixer", "Channel mixer", params=[
    FloatList("red", [1, 0, 0], length=3, min=-2, max=2, description="Output red = r*R + g*G + b*B"),
    FloatList("green", [0, 1, 0], length=3, min=-2, max=2),
    FloatList("blue", [0, 0, 1], length=3, min=-2, max=2),
    FloatList("offset", [0, 0, 0], length=3, min=-1, max=1),
    Bool("monochrome", False, description="Use the red row for all channels"),
], tags=["channel mixer", "matrix", "infrared", "monochrome"])
def channel_mixer(img, ctx, red, green, blue, offset, monochrome):
    """Mix RGB channels with a 3x3 matrix plus offset."""
    m = np.array([red, green, blue], F32)
    if monochrome:
        m = np.array([red, red, red], F32)
    return (img @ m.T + np.array(offset, F32)).astype(F32)


@op("adjust.black_white", "Black & white", params=[
    Float("reds", 40, min=-200, max=300), Float("yellows", 60, min=-200, max=300),
    Float("greens", 40, min=-200, max=300), Float("cyans", 60, min=-200, max=300),
    Float("blues", 20, min=-200, max=300), Float("magentas", 80, min=-200, max=300),
    Float("tint_hue", 35, min=0, max=360, unit="deg"), Float("tint_saturation", 0, min=0, max=100),
], tags=["black and white", "monochrome", "grayscale", "b&w", "sepia"])
def black_white(img, ctx, reds, yellows, greens, cyans, blues, magentas, tint_hue, tint_saturation):
    """Monochrome conversion with per-colour brightness (Photoshop B&W) and optional tint."""
    hsv = rgb_to_hsv(np.clip(img, 0, 1))
    h = hsv[..., 0] * 360
    base = luma(img)
    weights = [(0, reds), (60, yellows), (120, greens), (180, cyans), (240, blues), (300, magentas)]
    adj = np.zeros_like(base)
    for c, v in weights:
        d = np.abs((h - c + 180) % 360 - 180)
        w = np.clip(1 - d / 60, 0, 1)
        adj += w * (v - 50) / 100.0
    chroma = hsv[..., 1] * hsv[..., 2]
    y = base + adj * chroma * 0.6
    out = np.repeat(y[..., None], 3, axis=2)
    if tint_saturation:
        lab = rgb_to_oklab(out)
        da, db = hue_to_ab(tint_hue, tint_saturation / 100.0 * 0.1)
        lab[..., 1] += da
        lab[..., 2] += db
        out = oklab_to_rgb(lab)
    return out.astype(F32)


@op("adjust.selective_color", "Selective colour", params=[
    *[Float(f"{c}_{k}", 0, min=-100, max=100, step=1, title=f"{c.capitalize()} {k}")
      for c in ("reds", "yellows", "greens", "cyans", "blues", "magentas", "whites", "neutrals", "blacks")
      for k in ("cyan", "magenta", "yellow", "black")],
    Choice("method", "relative", choices=["relative", "absolute"]),
], tags=["selective color", "cmyk", "color correction"])
def selective_color(img, ctx, method, **kw):
    """Photoshop-style selective colour: adjust CMYK amounts within colour ranges."""
    x = np.clip(img, 0, 1)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(-1)
    mn = x.min(-1)
    mid = np.median(x, axis=-1)
    out = x.copy()
    # range masks (approximate Photoshop's definitions)
    masks = {
        "reds": np.where((r == mx) & (r > mn), mx - mid, 0),
        "yellows": np.where((b == mn) & (b < mx), mid - mn, 0),
        "greens": np.where((g == mx) & (g > mn), mx - mid, 0),
        "cyans": np.where((r == mn) & (r < mx), mid - mn, 0),
        "blues": np.where((b == mx) & (b > mn), mx - mid, 0),
        "magentas": np.where((g == mn) & (g < mx), mid - mn, 0),
        "whites": np.clip((mn - 0.5) * 2, 0, 1),
        "neutrals": 1 - (np.abs(mx - 0.5) + np.abs(mn - 0.5)),
        "blacks": np.clip((0.5 - mx) * 2, 0, 1),
    }
    for cname, m in masks.items():
        vals = [kw[f"{cname}_{k}"] / 100.0 for k in ("cyan", "magenta", "yellow", "black")]
        if not any(vals):
            continue
        m = np.clip(m, 0, 1).astype(F32)
        c_, m_, y_, k_ = vals
        for ch, v in ((0, c_), (1, m_), (2, y_)):
            if not v and not k_:
                continue
            amount = v + k_  # adding black darkens all channels
            if method == "relative":
                delta = -amount * out[..., ch] if amount > 0 else -amount * (1 - out[..., ch])
            else:
                delta = -amount * np.ones_like(m)
            out[..., ch] = out[..., ch] + m * delta * (1 if amount > 0 else 1)
    return out.astype(F32)


# ---------------------------------------------------------------------------
@op("adjust.dehaze", "Dehaze", params=[
    _amt("amount", "Dehaze", "Remove (+) or add (-) atmospheric haze"),
    Float("radius", 15, min=1, max=200, unit="px", advanced=True),
], tags=["haze", "fog", "clarity", "atmosphere"])
def dehaze(img, ctx, amount, radius):
    """Dark-channel-prior dehazing (He et al.) with guided-filter refinement; negative adds haze."""
    if amount == 0:
        return img
    x = np.clip(img, 0, 1)
    k = amount / 100.0
    if k < 0:
        air = np.percentile(x.reshape(-1, 3), 99.5, axis=0).astype(F32)
        return (x + (air - x) * (-k) * 0.6).astype(F32)
    r = ctx.ipx(radius)
    dark = cv2.erode(x.min(-1), cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1)))
    flat = dark.ravel()
    idx = np.argsort(flat)[-max(1, flat.size // 1000):]
    air = x.reshape(-1, 3)[idx].mean(0).astype(F32)
    air = np.maximum(air, 0.3)
    norm = x / air
    dark_n = cv2.erode(norm.min(-1), cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1)))
    t = 1 - 0.95 * k * dark_n
    t = guided_filter(luma(x), t.astype(F32), max(2, r * 2), eps=1e-3)
    t = np.clip(t, 0.15, 1)[..., None]
    return ((x - air) / t + air).astype(F32)


@op("adjust.clarity", "Clarity / texture", params=[
    _amt("clarity", "Clarity", "Mid-frequency local contrast"),
    _amt("texture", "Texture", "Fine detail contrast"),
    Float("radius", 40, min=2, max=400, unit="px", advanced=True),
], tags=["clarity", "local contrast", "texture", "detail", "soft", "dreamy"])
def clarity(img, ctx, clarity, texture, radius):
    """Local contrast (edge-aware) — positive adds punch, negative softens (dreamy)."""
    y = luma(img)
    ny = y.copy()
    if clarity:
        base = guided_filter(y, y, ctx.ipx(radius), eps=0.005)
        detail = y - base
        mid = 1 - np.abs(np.clip(y, 0, 1) * 2 - 1) ** 2
        ny = ny + clarity / 100.0 * 0.9 * detail * mid
    if texture:
        base2 = gaussian(y, max(0.8, ctx.px(2.5)))
        ny = ny + texture / 100.0 * 0.8 * (y - base2)
    return _scale_to_luma(img, y, ny)


@op("adjust.auto_tone", "Auto tone", params=[
    Float("clip", 0.2, min=0, max=5, step=0.05, unit="%", description="Percent of pixels clipped at each end"),
    Bool("per_channel", False, description="Stretch channels independently (also fixes casts)"),
    Float("strength", 1.0, min=0, max=1, step=0.01),
], tags=["auto", "levels", "stretch", "normalize", "auto contrast"])
def auto_tone(img, ctx, clip, per_channel, strength):
    """Automatic levels stretch based on percentiles."""
    x = img.reshape(-1, 3)
    if per_channel:
        lo = np.percentile(x, clip, axis=0)
        hi = np.percentile(x, 100 - clip, axis=0)
    else:
        y = luma(img).ravel()
        lo = np.full(3, np.percentile(y, clip))
        hi = np.full(3, np.percentile(y, 100 - clip))
    hi = np.maximum(hi, lo + 1e-3)
    out = (img - lo.astype(F32)) / (hi - lo).astype(F32)
    return (img + (out - img) * strength).astype(F32)


# ---------------------------------------------------------------------------
# LUTs and colour transfer
# ---------------------------------------------------------------------------
_LUT_CACHE: dict[tuple[str, float], np.ndarray] = {}


def load_lut_any(path: str) -> tuple[np.ndarray, tuple, tuple]:
    p = path.lower()
    if p.endswith(".cube"):
        lut, dmin, dmax, lut1 = load_cube(path)
        if lut1 is not None:
            n = len(lut1)
            ident = np.linspace(0, 1, 17, dtype=F32)
            r, g, b = np.meshgrid(ident, ident, ident, indexing="ij")
            src = np.stack([r, g, b], -1)
            t = np.clip((src - np.array(dmin)) / (np.array(dmax) - np.array(dmin)), 0, 1) * (n - 1)
            out = np.stack([np.interp(t[..., c].ravel(), np.arange(n), lut1[:, c]).reshape(r.shape)
                            for c in range(3)], -1)
            return out.astype(F32), (0, 0, 0), (1, 1, 1)
        return lut, dmin, dmax
    from ..core.io import load_image

    im, _ = load_image(path)
    return lut_from_hald_or_square(im), (0, 0, 0), (1, 1, 1)


@op("adjust.lut", "Apply LUT", params=[
    File("path", "", file_kind="lut", required=True, title="LUT file",
         description=".cube (3D/1D), Hald CLUT image, or 512x512 lookup image"),
    Float("strength", 1.0, min=0, max=1, step=0.01),
], tags=["lut", "cube", "3d lut", "look", "film emulation", "hald"])
def lut(img, ctx, path, strength):
    """Apply a 3D LUT from a .cube file or LUT image with trilinear interpolation."""
    p = ctx.resolve_file(path, "LUT")
    table, dmin, dmax = load_lut_any(str(p))
    out = apply_lut3d(img, table, dmin, dmax)
    return (img + (out - img) * strength).astype(F32)


@op("adjust.color_transfer", "Colour transfer from reference", params=[
    File("reference", "", file_kind="image", required=True, title="Reference image"),
    Choice("method", "reinhard", choices=["reinhard", "mkl", "histogram"],
           description="reinhard: mean/std in OKLab; mkl: Monge-Kantorovich linear map; histogram: per-channel CDF match"),
    Float("strength", 1.0, min=0, max=1, step=0.01),
    Bool("preserve_luminance", False, description="Transfer only colour, keep source lightness"),
], tags=["color transfer", "match", "reference", "style", "histogram matching"])
def color_transfer(img, ctx, reference, method, strength, preserve_luminance):
    """Non-parametric colour statistics transfer from a reference image (statistical, not learned)."""
    from ..core.io import load_image
    from ..core.image import fit_within
    from ..style.transfer import transfer_colors

    ref, _ = load_image(ctx.resolve_file(reference, "reference image"))
    ref, _ = fit_within(ref[..., :3], 512)
    out = transfer_colors(img, ref, method=method, preserve_luminance=preserve_luminance)
    return (img + (out - img) * strength).astype(F32)


@op("adjust.invert", "Invert", params=[], tags=["negative", "invert"])
def invert(img, ctx):
    """Invert colours (negative)."""
    return (1.0 - img).astype(F32)


@op("adjust.posterize", "Posterize", params=[Int("levels", 6, min=2, max=64)], tags=["posterize", "levels"])
def posterize(img, ctx, levels):
    """Reduce each channel to N levels."""
    n = levels - 1
    return (np.round(np.clip(img, 0, 1) * n) / n).astype(F32)


@op("adjust.threshold", "Threshold", params=[Float("level", 0.5, min=0, max=1, step=0.01)],
    tags=["threshold", "binary", "black and white"])
def threshold(img, ctx, level):
    """Binary black/white threshold on luminance."""
    y = (luma(img) >= level).astype(F32)
    return np.repeat(y[..., None], 3, axis=2)
