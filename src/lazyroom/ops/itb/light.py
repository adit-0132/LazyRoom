"""ImageToolbox "Light" group filters."""
from __future__ import annotations

import numpy as np

from ...core.params import Bool, Choice, Color, Float, Int
from . import _light_impl as L
from ._common import gmic_cmd, gmic_pipeline, itb_op, run_gmic

GROUP = "light"
F32 = np.float32


@itb_op("Brightness", "Brightness", GROUP, [
    Float("bias", 0.5, min=-1, max=1, step=0.01, title="Brightness", description="Value added to every channel"),
], upstream="aire", tags=["brightness", "lighten", "darken"])
def brightness(img, ctx, bias):
    """aire ``adjustment(gain=1, bias)``: rgb + bias, clamped."""
    return img + bias


@itb_op("CinematicDream", "Cinematic Dream", GROUP, [
    Float("details_strength", 75, min=0, max=400, title="Details strength"),
    Float("details_scale", 45, min=0, max=255, title="Details scale"),
    Float("softness", 55, min=0, max=255, title="Softness"),
    Int("highlights", 28, min=0, max=100, title="Highlights"),
    Float("contrast", 18, min=0, max=100, title="Contrast"),
    Float("shadow_strength", 0.55, min=0, max=1, step=0.01, title="Shadow strength"),
    Float("shadow_scale", 6, min=0, max=20, title="Shadow scale"),
    Bool("normalize", True, title="Normalize"),
], upstream="G'MIC", requires=["gmic"], tags=["bloom", "cinematic", "dreamy", "glow", "shadows"])
def cinematic_dream(img, ctx, details_strength, details_scale, softness, highlights, contrast,
                    shadow_strength, shadow_scale, normalize):
    """Highlight bloom followed by pop shadows (G'MIC fx_highlight_bloom + fx_pop_shadows)."""
    cmd = gmic_pipeline(
        gmic_cmd("fx_highlight_bloom", details_strength, details_scale, softness, highlights, contrast),
        gmic_cmd("fx_pop_shadows", shadow_strength, shadow_scale, normalize),
    )
    return run_gmic(img, cmd, ctx=ctx)


# ---------------------------------------------------------------------------
# aire tone mapping (sRGB -> linear -> curve -> sRGB)
# ---------------------------------------------------------------------------

def _exposure(default=1.0, lo=-4.0, hi=4.0):
    return Float("exposure", default, min=lo, max=hi, step=0.01, title="Exposure",
                 description="Linear-light multiplier applied before the curve")


@itb_op("AcesFilmicToneMapping", "ACES Filmic Tone Mapping", GROUP, [_exposure()], upstream="aire",
        tags=["tone mapping", "aces", "filmic", "hdr"])
def aces_filmic_tone_mapping(img, ctx, exposure):
    """aire ``AcesFilmicToneMapper`` (Narkowicz ACES fit, clamped) in linear light."""
    return L.aire_tone_map(img, L.tm_aces_filmic(exposure))


@itb_op("AcesHillToneMapping", "ACES Hill Tone Mapping", GROUP, [_exposure()], upstream="aire",
        tags=["tone mapping", "aces", "filmic", "hdr"])
def aces_hill_tone_mapping(img, ctx, exposure):
    """aire ``acesHill``: upstream instantiates ``AcesFilmicToneMapper`` (not ``AcesHillToneMapper``),
    so the result equals ACES Filmic; reproduced as-is."""
    return L.aire_tone_map(img, L.tm_aces_filmic(exposure))


@itb_op("Aldridge", "Aldridge", GROUP, [
    _exposure(1.0, 0.0, 2.0),
    Float("cutoff", 0.025, min=-1, max=1, step=0.001, title="Cutoff"),
], upstream="aire", tags=["tone mapping", "hdr"])
def aldridge(img, ctx, exposure, cutoff):
    """aire ``AldridgeToneMapper``."""
    return L.aire_tone_map(img, L.tm_aldridge(exposure, cutoff))


@itb_op("Drago", "Drago", GROUP, [
    _exposure(1.0, 0.0, 2.0),
    Float("threshold", 250, min=0, max=500, title="Threshold", description="SDR white point (Lmax)"),
], upstream="aire", tags=["tone mapping", "hdr", "logarithmic"])
def drago(img, ctx, exposure, threshold):
    """aire ``DragoToneMapper`` (adaptive logarithmic mapping on Rec.601 luma)."""
    return L.aire_tone_map_luma(img, L.tm_drago(exposure, threshold))


@itb_op("HableFilmicToneMapping", "Hable Filmic Tone Mapping", GROUP, [_exposure()], upstream="aire",
        tags=["tone mapping", "hable", "uncharted", "filmic", "hdr"])
def hable_filmic_tone_mapping(img, ctx, exposure):
    """aire ``HableFilmicToneMapper`` (Uncharted 2 curve, no white-point normalisation)."""
    return L.aire_tone_map(img, L.tm_hable(exposure))


@itb_op("HejlBurgessToneMapping", "Heji-Burgess Tone mapping", GROUP, [_exposure(1.0, 0.0, 4.0)],
        upstream="aire", tags=["tone mapping", "hejl", "filmic", "hdr"])
def hejl_burgess_tone_mapping(img, ctx, exposure):
    """aire ``HejlBurgessToneMapper``."""
    return L.aire_tone_map(img, L.tm_hejl_burgess(exposure))


@itb_op("LogarithmicToneMapping", "Logarithmic Tone Mapping", GROUP, [_exposure(1.0, 0.0, 4.0)],
        upstream="aire", tags=["tone mapping", "logarithmic", "hdr"])
def logarithmic_tone_mapping(img, ctx, exposure):
    """aire ``LogarithmicToneMapper`` (log(1 + L) / log(1 + exposure) on Rec.601 luma)."""
    return L.aire_tone_map_luma(img, L.tm_logarithmic(exposure))


@itb_op("Mobius", "Mobius", GROUP, [
    _exposure(1.0, 0.0, 2.0),
    Float("transition", 0.9, min=-2, max=2, step=0.01, title="Transition"),
    Float("peak", 1.0, min=-2, max=2, step=0.01, title="Peak"),
], upstream="aire", tags=["tone mapping", "mobius", "hdr"])
def mobius(img, ctx, exposure, transition, peak):
    """aire ``MobiusToneMapper`` (vectorised path)."""
    return L.aire_tone_map(img, L.tm_mobius(exposure, transition, peak))


@itb_op("Uchimura", "Uchimura", GROUP, [_exposure(1.0, 0.0, 2.0)], upstream="aire",
        tags=["tone mapping", "uchimura", "gran turismo", "hdr"])
def uchimura(img, ctx, exposure):
    """aire ``UchimuraToneMapper`` (Gran Turismo curve)."""
    return L.aire_tone_map(img, L.tm_uchimura(exposure))


# ---------------------------------------------------------------------------
# aire basic adjustments
# ---------------------------------------------------------------------------

@itb_op("Contrast", "Contrast", GROUP, [
    Float("gain", 2.0, min=0, max=2, step=0.01, title="Contrast", description="Gain around mid-grey 0.5"),
], upstream="aire", tags=["contrast"])
def contrast(img, ctx, gain):
    """aire ``adjustment(gain, bias=0)``: gain * (rgb - 0.5) + 0.5."""
    return gain * (img - F32(0.5)) + F32(0.5)


@itb_op("Gamma", "Gamma", GROUP, [
    Float("gamma", 1.0, min=0, max=2, step=0.01, title="Gamma",
          description="Exponent applied to normalised values: out = in ** gamma (>1 darkens midtones)"),
], upstream="aire", tags=["gamma", "levels"])
def gamma(img, ctx, gamma):
    """Gamma curve ``out = in ** gamma`` on normalised values.

    Deviation (bug fix): aire raises the raw 0..255 code value to ``gamma``,
    which turns any gamma other than 1.0 into near-white/near-black; the
    intended normalised curve is used instead.
    """
    with np.errstate(divide="ignore"):
        lut = np.clip(np.power(L.LUT_GRID, F32(gamma)), 0, 1)
    return L.apply_lut(np.ascontiguousarray(img[..., :3], dtype=F32), lut.astype(F32))


# ---------------------------------------------------------------------------
# CLAHE / histogram equalisation (aire-filters -> histogram_equalization crate)
# ---------------------------------------------------------------------------

def _thr():
    return Float("threshold", 0.5, min=-10, max=10, step=0.01, title="Threshold", description="Histogram clip level")


def _grid(default):
    return [Int("grid_size_x", default, min=1, max=100, title="Grid Size X", description="Number of tiles horizontally"),
            Int("grid_size_y", default, min=1, max=100, title="Grid Size Y", description="Number of tiles vertically")]


def _bins():
    return Int("bins_count", 128, min=2, max=256, title="Bins count")


@itb_op("Clahe", "CLAHE", GROUP, [_thr(), *_grid(8)], upstream="aire",
        tags=["clahe", "contrast", "equalize", "histogram", "local contrast"])
def clahe(img, ctx, threshold, grid_size_x, grid_size_y):
    """``clahe_yuv_rgba``: CLAHE on the Y plane of 8-bit full-range YCgCo (256 bins)."""
    return L.equalize(img, "ycgco", 256, (grid_size_x, grid_size_y), threshold)


def _register_clahe(name, title, space, doc):
    @itb_op(name, title, GROUP, [_thr(), *_grid(8), _bins()], upstream="aire", description=doc,
            tags=["clahe", "contrast", "equalize", "histogram", "local contrast", space])
    def fn(img, ctx, threshold, grid_size_x, grid_size_y, bins_count):
        return L.equalize(img, space, bins_count, (grid_size_x, grid_size_y), threshold)
    return fn


_register_clahe("ClaheHSL", "Clahe HSL", "hsl", "``clahe_hsl_rgba``: CLAHE on HSL lightness.")
_register_clahe("ClaheHSV", "Clahe HSV", "hsv", "``clahe_hsv_rgba``: CLAHE on HSV value.")
_register_clahe("ClaheJzazbz", "Clahe Jzazbz", "jzazbz", "``clahe_jzazbz_rgba``: CLAHE on Jz (200 nits).")
_register_clahe("ClaheLAB", "CLAHE LAB", "lab",
                "``clahe_lab_rgba``: CLAHE on CIE L*; a*/b* round-trip through upstream's u16 storage.")
_register_clahe("ClaheLUV", "CLAHE LUV", "luv",
                "``clahe_luv_rgba``: CLAHE on CIE L*; u*/v* round-trip through upstream's u16 storage.")
_register_clahe("ClaheOklab", "Clahe Oklab", "oklab", "``clahe_oklab_rgba``: CLAHE on Oklab L.")
_register_clahe("ClaheOklch", "Clahe Oklch", "oklch", "``clahe_oklch_rgba``: CLAHE on Oklch L.")


@itb_op("EqualizeHistogramAdaptive", "Equalize Histogram Adaptive", GROUP, [*_grid(3)], upstream="aire",
        tags=["equalize", "histogram", "adaptive", "ahe", "contrast"])
def equalize_histogram_adaptive(img, ctx, grid_size_x, grid_size_y):
    """``ahe_yuv_rgba``: adaptive (unclipped) equalisation of 8-bit YCgCo Y."""
    return L.equalize(img, "ycgco", 256, (grid_size_x, grid_size_y), None)


def _register_ahe(name, title, space):
    doc = f"``ahe_{space}_rgba``: adaptive (unclipped) histogram equalisation in {space.upper()}."
    @itb_op(name, title, GROUP, [*_grid(3), _bins()], upstream="aire", description=doc,
            tags=["equalize", "histogram", "adaptive", "ahe", "contrast", space])
    def fn(img, ctx, grid_size_x, grid_size_y, bins_count):
        return L.equalize(img, space, bins_count, (grid_size_x, grid_size_y), None)
    return fn


_register_ahe("EqualizeHistogramAdaptiveHSL", "Equalize Histogram Adaptive HSL", "hsl")
_register_ahe("EqualizeHistogramAdaptiveHSV", "Equalize Histogram Adaptive HSV", "hsv")
_register_ahe("EqualizeHistogramAdaptiveLAB", "Equalize Histogram Adaptive LAB", "lab")
_register_ahe("EqualizeHistogramAdaptiveLUV", "Equalize Histogram Adaptive LUV", "luv")


@itb_op("EqualizeHistogramHSV", "Equalize Histogram HSV", GROUP, [_bins()], upstream="aire",
        tags=["equalize", "histogram", "contrast", "hsv"])
def equalize_histogram_hsv(img, ctx, bins_count):
    """``hist_equal_hsv_rgba``: global histogram equalisation of HSV value."""
    return L.equalize(img, "hsv", bins_count, None, None)


# ---------------------------------------------------------------------------
# Trickle / jhlabs / GPUImage / procedural
# ---------------------------------------------------------------------------

@itb_op("Bloom", "Bloom", GROUP, [
    Float("threshold", 0.6, min=0, max=1, step=0.01, title="Threshold"),
    Float("strength", 1.5, min=0, max=3, step=0.01, title="Strength"),
    Int("radius", 25, min=1, max=100, title="Radius", unit="px"),
    Float("soft_knee", 0.5, min=0, max=1, step=0.01, title="Soft Knee"),
    Float("exposure", 0.02, min=0, max=1, step=0.01, title="Exposure"),
    Float("gamma", 1.1, min=0, max=2, step=0.01, title="Gamma"),
], upstream="Trickle", tags=["bloom", "glow", "highlights", "dreamy"])
def bloom(img, ctx, threshold, strength, radius, soft_knee, exposure, gamma):
    """Trickle ``bloom``: soft-knee bright pass, Gaussian blur (sigma = radius/2), additive, gamma."""
    return L.trickle_bloom(img, threshold, strength, ctx.ipx(radius), soft_knee, exposure, gamma)


@itb_op("Glow", "Glow", GROUP, [Float("amount", 0.5, min=0, max=1, step=0.01, title="Amount")],
        upstream="jhlabs", tags=["glow", "soft", "bloom"])
def glow(img, ctx, amount):
    """jhlabs ``GlowFilter``: rgb + 4 * amount * gaussian(rgb, radius 2 px)."""
    blurred = L.jh_gaussian_blur(img[..., :3], ctx.px(2.0))
    return img[..., :3] + blurred * F32(4.0 * amount)


@itb_op("Levels", "Levels", GROUP, [
    Float("low_input", 0.0, min=0, max=1, step=0.01, title="Low input"),
    Float("high_input", 1.0, min=0, max=1, step=0.01, title="High input"),
    Float("low_output", 0.0, min=0, max=1, step=0.01, title="Low output"),
    Float("high_output", 1.0, min=0, max=1, step=0.01, title="High output"),
], upstream="jhlabs", tags=["levels", "black point", "white point", "contrast"])
def levels(img, ctx, low_input, high_input, low_output, high_output):
    """jhlabs ``LevelsFilter`` transfer: lo_out + (hi_out - lo_out) * (v - lo_in) / (hi_in - lo_in)."""
    if low_input == high_input:
        high_input = high_input + 1.0 / 255.0
    return F32(low_output) + F32(high_output - low_output) * (img - F32(low_input)) / F32(high_input - low_input)


@itb_op("LightEffects", "Light Effects", GROUP, [
    Float("bump_height", 1.0, min=0, max=10, step=0.01, title="Bump height"),
    Float("bump_softness", 5.0, min=0, max=50, step=0.01, title="Bump softness", unit="px"),
], upstream="jhlabs", tags=["light", "emboss", "relief", "bump"])
def light_effects(img, ctx, bump_height, bump_softness):
    """jhlabs ``LightFilter`` (native port): bump shading from the blurred brightness, default distant light."""
    return L.jh_light(img, bump_height, ctx.px(bump_softness))


@itb_op("Rays", "Rays", GROUP, [
    Float("opacity", 0.75, min=0, max=1, step=0.01, title="Opacity"),
    Float("threshold", 0.25, min=0, max=1, step=0.01, title="Threshold"),
    Float("strength", 1.0, min=0, max=5, step=0.01, title="Strength"),
], upstream="jhlabs", tags=["rays", "light rays", "god rays", "zoom"])
def rays(img, ctx, opacity, threshold, strength):
    """jhlabs ``RaysFilter`` (native port): radial samples towards the centre above a brightness threshold."""
    return L.jh_rays(img, opacity, threshold, strength)


def _full_res(img, ctx):
    h, w = img.shape[:2]
    s = ctx.scale if ctx.scale > 0 else 1.0
    return max(1, round(w / s)), max(1, round(h / s)), 1.0 / s


@itb_op("Flare", "Flare", GROUP, [
    Float("radius", 0.3, min=0.01, max=1, step=0.01, title="Radius", description="Fraction of half the longer side"),
    Float("base_amount", 1.0, min=0, max=2, step=0.01, title="Base amount"),
    Float("ring_amount", 0.2, min=0, max=2, step=0.01, title="Ring amount"),
    Float("ray_amount", 0.1, min=0, max=2, step=0.01, title="Ray amount"),
    Float("ring_width", 1.6, min=0.01, max=5, step=0.01, title="Ring width"),
    Float("center_x", 0.5, min=0, max=1, step=0.01, title="Center X"),
    Float("center_y", 0.5, min=0, max=1, step=0.01, title="Center Y"),
    Color("color", "#FFFFFF", title="Color"),
], upstream="jhlabs", tags=["flare", "lens flare", "light", "sun"])
def flare(img, ctx, radius, base_amount, ring_amount, ray_amount, ring_width, center_x, center_y, color):
    """jhlabs ``FlareFilter`` (native port); distances are evaluated in full-resolution pixels."""
    wf, hf, inv = _full_res(img, ctx)
    cx, cy = round(center_x * wf), round(center_y * hf)
    r = round(radius * (max(wf, hf) / 2.0))
    return L.jh_flare(img, cx, cy, r, ring_width, base_amount, ring_amount, ray_amount, color, inv)


@itb_op("Sparkle", "Sparkle", GROUP, [
    Int("amount", 50, min=0, max=200, title="Amount"),
    Int("rays", 50, min=0, max=200, title="Rays"),
    Float("radius", 0.3, min=0, max=1, step=0.01, title="Radius", description="Fraction of half the longer side"),
    Int("randomness", 25, min=0, max=100, title="Randomness"),
    Float("center_x", 0.5, min=0, max=1, step=0.01, title="Center X"),
    Float("center_y", 0.5, min=0, max=1, step=0.01, title="Center Y"),
    Color("color", "#FFFFFF", title="Color"),
], upstream="jhlabs", tags=["sparkle", "star", "glint", "light"])
def sparkle(img, ctx, amount, rays, radius, randomness, center_x, center_y, color):
    """jhlabs ``SparkleFilter`` (native port, fixed ray-length seed 371)."""
    wf, hf, inv = _full_res(img, ctx)
    cx, cy = round(center_x * wf), round(center_y * hf)
    r = round(radius * (max(wf, hf) / 2.0))
    return L.jh_sparkle(img, cx, cy, r, amount, rays, randomness, color, inv)


@itb_op("Exposure", "Exposure", GROUP, [
    Float("exposure", 1.0, min=-4, max=4, step=0.01, title="Exposure", description="Stops: rgb * 2^exposure"),
], upstream="GPUImage", tags=["exposure", "brightness", "stops"])
def exposure(img, ctx, exposure):
    """GPUImageExposureFilter: rgb * 2^exposure."""
    return img * F32(2.0 ** exposure)


@itb_op("Haze", "Haze", GROUP, [
    Float("distance", 0.2, min=-0.3, max=0.3, step=0.01, title="Distance"),
    Float("slope", 0.0, min=-0.3, max=0.3, step=0.01, title="Slope"),
], upstream="GPUImage", tags=["haze", "dehaze", "fog"])
def haze(img, ctx, distance, slope):
    """GPUImageHazeFilter: (c - d) / (1 - d) with d = y * slope + distance (y = 0 at the top)."""
    h = img.shape[0]
    y = (np.arange(h, dtype=F32) + F32(0.5)) / F32(h)
    d = (y * F32(slope) + F32(distance))[:, None, None]
    return (img - d) / (F32(1) - d)


@itb_op("HighlightsAndShadows", "Highlights and shadows", GROUP, [
    Float("shadows", 0.25, min=0, max=2, step=0.01, title="Highlights and shadows",
          description="GPUImage shadows parameter (highlights fixed at 1)"),
], upstream="GPUImage", tags=["shadows", "highlights", "recovery"])
def highlights_and_shadows(img, ctx, shadows):
    """ImageToolbox ``GPUImageHighlightShadowWideRangeFilter(shadows, highlights=1)``."""
    return L.highlight_shadow_wide_range(np.ascontiguousarray(img[..., :3], dtype=F32), float(shadows), 1.0)


@itb_op("Vignette", "Vignette", GROUP, [
    Float("start", 0.3, min=-2, max=2, step=0.01, title="Start"),
    Float("end", 0.75, min=-2, max=2, step=0.01, title="End"),
    Color("color", "#000000", title="Color"),
], upstream="GPUImage", tags=["vignette", "darken edges", "frame"])
def vignette(img, ctx, start, end, color):
    """GPUImageVignetteFilter: mix(rgb, color, smoothstep(start, end, |uv - (0.5, 0.5)|))."""
    return L.gpuimage_vignette(np.ascontiguousarray(img[..., :3], dtype=F32), float(start), float(end),
                               np.asarray(color[:3], np.float64))


@itb_op("WhiteBalance", "White balance", GROUP, [
    Int("temperature", 7000, min=1000, max=10000, title="Temperature", description="Kelvin-like slider (5000 = neutral)"),
    Float("tint", 100.0, min=-100, max=100, step=0.01, title="Tint"),
], upstream="GPUImage", tags=["white balance", "temperature", "tint", "warm", "cool"])
def white_balance(img, ctx, temperature, tint):
    """GPUImageWhiteBalanceFilter (YIQ tint shift + overlay with a warm filter colour; GLSL column-major mats)."""
    return L.gpuimage_white_balance(img, temperature, tint)


def _transform_params():
    return [Float("offset_x", 0.0, min=-1, max=1, step=0.001, title="Offset X"),
            Float("offset_y", 0.0, min=-1, max=1, step=0.001, title="Offset Y"),
            Float("scale", 1.0, min=0.05, max=8, step=0.001, title="Scale"),
            Float("rotation", 0.0, min=-180, max=180, step=0.001, title="Rotation")]


@itb_op("BokehLights", "Bokeh Lights", GROUP, [
    Color("color", "#F5B05B", title="Color", description="Light colour (alpha scales the lights)"),
    Float("intensity", 0.4, min=0, max=2, step=0.001, title="Intensity"),
    Int("count", 24, min=1, max=64, title="Count"),
    Int("blades", 6, min=3, max=12, title="Blades"),
    Float("radius", 0.07, min=0.005, max=0.3, step=0.001, title="Radius"),
    Float("variability", 0.5, min=0, max=1, step=0.001, title="Variability"),
    Float("color_variation", 0.3, min=0, max=1, step=0.001, title="Color variation"),
    Int("seed", 0, min=0, max=1000, title="Random seed"),
    *_transform_params(),
], upstream="ImageToolbox", tags=["bokeh", "lights", "overlay", "procedural"])
def bokeh_lights(img, ctx, color, intensity, count, blades, radius, variability, color_variation, seed,
                 offset_x, offset_y, scale, rotation):
    """ImageToolbox procedural shader ``BokehLights``: additive polygonal discs (hash-placed)."""
    return L.bokeh_lights(img, intensity, count, blades, radius, variability, color_variation, seed,
                          offset_x, offset_y, scale, rotation, color)


@itb_op("Fireworks", "Fireworks", GROUP, [
    Int("seed", 0, min=0, max=1000, title="Random seed"),
    Int("explosions", 4, min=1, max=12, title="Explosions"),
    Int("particles", 48, min=8, max=128, title="Particles"),
    Int("mode", 0, min=0, max=2, title="Mode", description="0 = sharp, 1 = soft, 2 = soft with sparkle bands"),
    Float("intensity", 0.6, min=0, max=2, step=0.001, title="Intensity"),
    Float("time", 0.5, min=0.05, max=1, step=0.001, title="Time"),
    *_transform_params(),
], upstream="ImageToolbox", tags=["fireworks", "overlay", "procedural", "light"])
def fireworks(img, ctx, seed, explosions, particles, mode, intensity, time, offset_x, offset_y, scale, rotation):
    """ImageToolbox procedural shader ``Fireworks``: additive radial spokes on expanding rings."""
    return L.fireworks(img, seed, explosions, particles, mode, intensity, time, offset_x, offset_y, scale, rotation)


# ---------------------------------------------------------------------------
# G'MIC filters (exact command strings of ImageToolboxLibs' gmic builders)
# ---------------------------------------------------------------------------

# GmicChannel enum (value = index)
_CHANNELS = ["All", "Rgba", "Rgb", "Red", "Green", "Blue", "Alpha", "LinearRgb", "LinearRed", "LinearGreen",
             "LinearBlue", "YCbCr", "Luminance", "Chrominance", "BlueChrominance", "RedChrominance",
             "GreenChrominance", "Lab", "LabLightness", "LabChrominance", "LabA", "LabB", "Lch", "LchChrominance",
             "LchChroma", "LchHue", "Hsv", "HsvHue", "HsvSaturation", "HsvValue", "Hsi", "HsiIntensity", "Hsl",
             "HslLightness", "Cmyk", "Cmy", "Cyan", "Magenta", "Yellow", "Key", "Yiq", "YiqLuma", "YiqChrominance",
             "Ryb", "RybRed", "RybYellow", "RybBlue"]
_LIGHTGLOW_MODES = ["Burn", "Dodge", "Freeze", "GrainMerge", "HardLight", "Interpolation", "Lighten", "Multiply",
                    "Overlay", "Reflect", "SoftLight", "Stamp", "Value"]
_COLOR_SPACES6 = ["Hsi", "Hsv", "Lab", "LinearRgb", "Rgb", "YCbCr"]  # RetinexColorSpace / ToneEnhanceChannel


def _snake_choice(name: str) -> str:
    special = {"YCbCr": "ycbcr", "LinearRgb": "linear_rgb", "GrainMerge": "grain_merge", "HardLight": "hard_light",
               "SoftLight": "soft_light", "PreserveRange": "preserve_range",
               "PreserveCovariance": "preserve_covariance"}
    if name in special:
        return special[name]
    out = ""
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out += "_"
        out += ch.lower()
    return out


def _enum_choice(name, title, values, default, description=""):
    choices = [_snake_choice(v) for v in values]
    return Choice(name, _snake_choice(default), choices=choices, title=title, description=description,
                  labels=dict(zip(choices, values)))


def _enum_index(values, choice):
    return [_snake_choice(v) for v in values].index(choice)


def _channel(default):
    return _enum_choice("channel", "Channel", _CHANNELS, default, "Channels / colour component the filter acts on")


def _srand(ctx) -> str:
    """Seed G'MIC's RNG from the render context (upstream is non-deterministic)."""
    return gmic_cmd("srand", int(ctx.rng("gmic").integers(0, 2**31 - 1)))


@itb_op("DynamicContrast", "Dynamic Contrast", GROUP, [
    Float("amount", 12, min=-64, max=64, title="Amount"),
    Float("threshold", 128, min=0, max=255, title="Threshold"),
    Float("intensity", 220, min=0, max=255, title="Intensity"),
], upstream="G'MIC", requires=["gmic"], tags=["contrast", "dynamic contrast"])
def dynamic_contrast(img, ctx, amount, threshold, intensity):
    """G'MIC ``rep_dynamic_contrast``."""
    return run_gmic(img, gmic_cmd("rep_dynamic_contrast", amount, threshold, intensity), ctx=ctx)


@itb_op("EqualizeLight", "Equalize Light", GROUP, [
    Float("amount", 88, min=0, max=100, title="Amount"),
    _enum_choice("mode", "Mode", ["PreserveRange", "PreserveCovariance"], "PreserveCovariance"),
    _channel("Luminance"),
], upstream="G'MIC", requires=["gmic"], tags=["equalize", "light", "illumination"])
def equalize_light(img, ctx, amount, mode, channel):
    """G'MIC ``fx_equalize_light``."""
    cmd = gmic_cmd("fx_equalize_light", amount, _enum_index(["PreserveRange", "PreserveCovariance"], mode),
                   _enum_index(_CHANNELS, channel))
    return run_gmic(img, cmd, ctx=ctx)


@itb_op("EqualizeShadow", "Equalize Shadow", GROUP, [
    Float("amplitude", 0.8, min=0, max=1, step=0.01, title="Amplitude"),
], upstream="G'MIC", requires=["gmic"], tags=["shadows", "equalize", "soft light"])
def equalize_shadow(img, ctx, amplitude):
    """G'MIC ``fx_equalize_shadow`` (soft-light blend with the negative)."""
    return run_gmic(img, gmic_cmd("fx_equalize_shadow", amplitude), ctx=ctx)


@itb_op("HighlightSynthesis", "Highlight Synthesis", GROUP, [
    Float("red", 1.1, min=0, max=2, step=0.01, title="Red"),
    Float("green", 0.7, min=0, max=2, step=0.01, title="Green"),
    Float("blue", 1.15, min=0, max=2, step=0.01, title="Blue"),
    Bool("rgb_to_srgb", True, title="RGB to sRGB"),
], upstream="G'MIC", requires=["gmic"], tags=["highlights", "recovery", "clipping"])
def highlight_synthesis(img, ctx, red, green, blue, rgb_to_srgb):
    """G'MIC ``iain_highlight_synthesis``."""
    return run_gmic(img, gmic_cmd("iain_highlight_synthesis", red, green, blue, rgb_to_srgb), ctx=ctx)


@itb_op("LightGlow", "Light Glow", GROUP, [
    Float("density", 30, min=0, max=100, title="Density"),
    Float("amplitude", 0.5, min=0, max=2, step=0.01, title="Amplitude"),
    _enum_choice("mode", "Mode", _LIGHTGLOW_MODES, "Overlay", "Blend mode"),
    Float("opacity", 0.8, min=0, max=1, step=0.01, title="Opacity"),
    _channel("Luminance"),
], upstream="G'MIC", requires=["gmic"], tags=["glow", "light", "edges"])
def light_glow(img, ctx, density, amplitude, mode, opacity, channel):
    """G'MIC ``fx_lightglow``."""
    cmd = gmic_cmd("fx_lightglow", density, amplitude, _enum_index(_LIGHTGLOW_MODES, mode), opacity,
                   _enum_index(_CHANNELS, channel))
    return run_gmic(img, cmd, ctx=ctx)


@itb_op("LightPatch", "Light Patch", GROUP, [
    Int("density", 5, min=2, max=30, title="Density"),
    Float("darkness", 0.7, min=0, max=1, step=0.01, title="Darkness"),
    Float("lightness", 2.5, min=1, max=4, step=0.01, title="Lightness"),
    _channel("Luminance"),
], upstream="G'MIC", requires=["gmic"], tags=["light", "patches", "texture"])
def light_patch(img, ctx, density, darkness, lightness, channel):
    """G'MIC ``fx_light_patch`` (random noise; seeded from the render context)."""
    cmd = gmic_pipeline(_srand(ctx), gmic_cmd("fx_light_patch", density, darkness, lightness,
                                              _enum_index(_CHANNELS, channel)))
    return run_gmic(img, cmd, ctx=ctx)


@itb_op("LocalContrastEnhancement", "Local Contrast Enhancement", GROUP, [
    Float("spatial_radius", 80, min=30, max=200, title="Spatial Radius", unit="px"),
    Float("amount", 1.2, min=0, max=5, step=0.01, title="Amount"),
    Float("darkness_level", 1.2, min=0, max=4, step=0.01, title="Darkness Level"),
    Float("lightness_level", 1.2, min=0, max=4, step=0.01, title="Lightness Level"),
    _channel("Luminance"),
], upstream="G'MIC", requires=["gmic"], tags=["local contrast", "clarity", "lce"])
def local_contrast_enhancement(img, ctx, spatial_radius, amount, darkness_level, lightness_level, channel):
    """G'MIC ``fx_LCE`` (spatial radius scaled for preview proxies)."""
    cmd = gmic_cmd("fx_LCE", ctx.px(spatial_radius), amount, darkness_level, lightness_level,
                   _enum_index(_CHANNELS, channel))
    return run_gmic(img, cmd, ctx=ctx)


@itb_op("PopShadows", "Pop Shadows", GROUP, [
    Float("strength", 0.75, min=0, max=1, step=0.01, title="Strength"),
    Float("scale", 5, min=0, max=20, step=0.01, title="Scale"),
    Bool("post_normalize", True, title="Post Normalize"),
], upstream="G'MIC", requires=["gmic"], tags=["shadows", "contrast", "pop"])
def pop_shadows(img, ctx, strength, scale, post_normalize):
    """G'MIC ``fx_pop_shadows``."""
    return run_gmic(img, gmic_cmd("fx_pop_shadows", strength, scale, post_normalize), ctx=ctx)


@itb_op("ReliefLight", "Relief Light", GROUP, [
    Float("ambient_lightness", 0.3, min=0, max=5, step=0.01, title="Ambient Lightness"),
    Float("specular_lightness", 0.2, min=0, max=2, step=0.01, title="Specular Lightness"),
    Float("specular_size", 0.2, min=0, max=1, step=0.01, title="Specular Size"),
    Float("darkness", 0.0, min=0, max=1, step=0.01, title="Darkness"),
    Float("light_smoothness", 1.0, min=0, max=5, step=0.01, title="Light Smoothness"),
    Float("light_x", 50, min=0, max=100, title="Light X"),
    Float("light_y", 50, min=0, max=100, title="Light Y"),
    Float("light_z", 5, min=0, max=20, step=0.01, title="Light Z"),
    Float("scale", 0.5, min=0, max=3, step=0.01, title="Scale"),
    Bool("opacity_as_heightmap", False, title="Opacity As Heightmap"),
    Float("image_smoothness", 0.0, min=0, max=10, step=0.01, title="Image Smoothness"),
], upstream="G'MIC", requires=["gmic"], tags=["relief", "light", "emboss", "3d"])
def relief_light(img, ctx, ambient_lightness, specular_lightness, specular_size, darkness, light_smoothness,
                 light_x, light_y, light_z, scale, opacity_as_heightmap, image_smoothness):
    """G'MIC ``fx_light_relief``."""
    cmd = gmic_cmd("fx_light_relief", ambient_lightness, specular_lightness, specular_size, darkness,
                   light_smoothness, light_x, light_y, light_z, scale, opacity_as_heightmap, image_smoothness)
    return run_gmic(img, cmd, ctx=ctx)


@itb_op("Retinex", "Retinex", GROUP, [
    Float("strength", 75, min=0, max=100, title="Strength"),
    Float("offset", 16, min=1, max=256, title="Offset"),
    _enum_choice("color_space", "Color Space", _COLOR_SPACES6, "Hsv"),
    Float("min_cut", 1, min=0, max=100, step=0.01, title="Min Cut"),
    Float("max_cut", 1, min=0, max=100, step=0.01, title="Max Cut"),
    Float("regularization", 5, min=0, max=32, step=0.01, title="Regularization"),
    Float("low_scale", 15, min=1, max=512, title="Low Scale", unit="px"),
    Float("middle_scale", 80, min=1, max=512, title="Middle Scale", unit="px"),
    Float("high_scale", 250, min=1, max=512, title="High Scale", unit="px"),
], upstream="G'MIC", requires=["gmic"], tags=["retinex", "dynamic range", "shadows", "local contrast"])
def retinex(img, ctx, strength, offset, color_space, min_cut, max_cut, regularization, low_scale, middle_scale,
            high_scale):
    """G'MIC ``fx_retinex`` (Gaussian scales scaled for preview proxies)."""
    cmd = gmic_cmd("fx_retinex", strength, offset, _enum_index(_COLOR_SPACES6, color_space), min_cut, max_cut,
                   regularization, ctx.px(low_scale), ctx.px(middle_scale), ctx.px(high_scale))
    return run_gmic(img, cmd, ctx=ctx)


@itb_op("ShadowPatch", "Shadow Patch", GROUP, [
    Float("opacity", 0.7, min=0, max=1, step=0.01, title="Opacity"),
    _channel("All"),
], upstream="G'MIC", requires=["gmic"], tags=["shadows", "patches", "texture"])
def shadow_patch(img, ctx, opacity, channel):
    """G'MIC ``fx_shadow_patch`` (plasma noise; seeded from the render context)."""
    cmd = gmic_pipeline(_srand(ctx), gmic_cmd("fx_shadow_patch", opacity, _enum_index(_CHANNELS, channel)))
    return run_gmic(img, cmd, ctx=ctx)


@itb_op("ToneEnhance", "Tone Enhance", GROUP, [
    Float("shadows_detail", 0.6, min=0, max=2, step=0.01, title="Shadows Detail"),
    Float("shadows_gamma", 0.9, min=0.2, max=1.8, step=0.01, title="Shadows Gamma"),
    Float("highlights_detail", 0.5, min=0, max=2, step=0.01, title="Highlights Detail"),
    Float("highlights_gamma", 1.1, min=0.2, max=1.8, step=0.01, title="Highlights Gamma"),
    Int("center", 128, min=0, max=255, title="Center"),
    Float("midpoint_detail", 0.4, min=0, max=2, step=0.01, title="Midpoint Detail"),
    Float("midpoint_gamma", 1.0, min=0.2, max=1.8, step=0.01, title="Midpoint Gamma"),
    Float("recovery_boost", 0.6, min=0, max=1, step=0.01, title="Recovery Boost"),
    Float("recovery_smoothness", 0.2, min=0, max=2, step=0.01, title="Recovery Smoothness"),
    Int("detail_mask_smoothness", 3, min=0, max=20, title="Detail Mask Smoothness"),
    _enum_choice("channel", "Channel", _COLOR_SPACES6, "YCbCr"),
    _enum_choice("values", "Values", ["Cut", "Normalize"], "Cut"),
    Bool("color_median", False, title="Color Median"),
], upstream="G'MIC", requires=["gmic"], tags=["tone", "shadows", "highlights", "detail", "hdr"])
def tone_enhance(img, ctx, shadows_detail, shadows_gamma, highlights_detail, highlights_gamma, center,
                 midpoint_detail, midpoint_gamma, recovery_boost, recovery_smoothness, detail_mask_smoothness,
                 channel, values, color_median):
    """G'MIC ``gcd_tone_enhance``."""
    cmd = gmic_cmd("gcd_tone_enhance", shadows_detail, shadows_gamma, highlights_detail, highlights_gamma, center,
                   midpoint_detail, midpoint_gamma, recovery_boost, recovery_smoothness, detail_mask_smoothness,
                   _enum_index(_COLOR_SPACES6, channel), _enum_index(["Cut", "Normalize"], values), color_median)
    return run_gmic(img, cmd, ctx=ctx)


@itb_op("ZoneSystem", "Zone System", GROUP, [
    Int("shadows_zone", 2, min=1, max=5, title="Shadows Zone"),
    Int("highlights_zone", 9, min=6, max=10, title="Highlights Zone"),
    Float("gamma", 1.1, min=0, max=5, step=0.01, title="Gamma"),
    Float("contrast", 1.15, min=0, max=4, step=0.01, title="Contrast"),
    Int("black_point", 8, min=0, max=255, title="Black Point"),
    Int("white_point", 245, min=0, max=255, title="White Point"),
], upstream="G'MIC", requires=["gmic"], tags=["zone system", "tone", "ansel adams", "contrast"])
def zone_system(img, ctx, shadows_zone, highlights_zone, gamma, contrast, black_point, white_point):
    """G'MIC ``fx_zonesystem`` (gamma is clamped to >= 0.01: the command takes ``log(gamma)``)."""
    cmd = gmic_cmd("fx_zonesystem", shadows_zone, highlights_zone, max(gamma, 0.01), contrast, black_point,
                   white_point)
    return run_gmic(img, cmd, ctx=ctx)
