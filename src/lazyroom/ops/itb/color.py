"""ImageToolbox "Color" group filters.

Ported from ImageToolbox (Apache-2.0) and the libraries it calls: GPUImage (BSD-3)
shaders, aire (MIT) C++/Rust incl. the palette-transfer / colorutils-rs crates,
Trickle (Apache-2.0) C++, the jhlabs filters (Apache-2.0) Rust port in
ImageToolboxLibs, and G'MIC (CeCILL, external CLI) command builders.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ...core.params import Bool, Choice, Color, File, Float, FloatList, Gradient, Int, Points
from . import _color_impl as ci
from ._common import color_matrix_3x3, gmic_cmd, itb_op, run_gmic

GROUP = "color"
F32 = np.float32
GMIC = dict(upstream="G'MIC", requires=["gmic"])


def _has_alpha(img: np.ndarray) -> bool:
    return img.ndim == 3 and img.shape[2] == 4


def _alpha(img: np.ndarray) -> np.ndarray:
    return img[..., 3] if _has_alpha(img) else np.ones(img.shape[:2], F32)


def _rgb(c) -> np.ndarray:
    return np.asarray(c[:3], F32)


def _gmic(img, ctx, name, *args, srand: bool = False):
    cmd = gmic_cmd(name, *args)
    if srand:  # G'MIC's RNG is time-seeded: pin it for deterministic renders
        cmd = f"srand {int(ctx.rng('gmic').integers(0, 2 ** 31 - 1))} " + cmd
    return run_gmic(img, cmd, ctx=ctx)


def _idx(choices, value) -> int:
    return list(choices).index(value)


# ===========================================================================
# G'MIC filters (exact ImageToolboxLibs command builders)
# ===========================================================================

BW_FILM_TYPES = ["manual", "agfa_200x", "agfapan_25", "agfapan_100", "agfapan_400", "ilford_delta_100",
                 "ilford_delta_400", "ilford_delta_3200", "ilford_fp4", "ilford_hp4", "ilford_pan_f", "ilford_sfx",
                 "ilford_xp2_super", "kodak_tmax_100", "kodak_tmax_400", "kodak_tri_x"]
BW_GRAIN_TYPES = ["gaussian", "uniform", "salt_and_pepper", "poisson"]


@itb_op("BlackAndWhiteFilm", "Black and White Film", GROUP, [
    Choice("film_type", "kodak_tri_x", choices=BW_FILM_TYPES, title="Film Type",
           description="Film preset (overrides the red/green/blue levels unless 'manual')"),
    Float("red_level", 0.299, min=0, max=1, step=0.001, title="Red Level"),
    Float("red_smoothness", 0, min=0, max=10, step=0.01, title="Red Smoothness"),
    Float("green_level", 0.587, min=0, max=1, step=0.001, title="Green Level"),
    Float("green_smoothness", 0, min=0, max=10, step=0.01, title="Green Smoothness"),
    Float("blue_level", 0.114, min=0, max=1, step=0.001, title="Blue Level"),
    Float("blue_smoothness", 0, min=0, max=10, step=0.01, title="Blue Smoothness"),
    Float("gamma", 1.05, min=0.01, max=5, step=0.01, title="Gamma"),
    Float("contrast", 1.2, min=0, max=4, step=0.01, title="Contrast"),
    Float("brightness", 0, min=-255, max=255, title="Brightness"),
    Float("hue", 30, min=0, max=360, title="Hue", description="Toning hue (degrees)"),
    Float("saturation", 0.08, min=0, max=1, step=0.01, title="Saturation", description="Toning saturation"),
    Float("grain_shadows", 18, min=0, max=200, title="Grain Shadows"),
    Float("grain_midtones", 10, min=0, max=200, title="Grain Midtones"),
    Float("grain_highlights", 5, min=0, max=200, title="Grain Highlights"),
    Float("grain_tone_fading", 2, min=0, max=10, step=0.01, title="Grain Tone Fading"),
    Float("grain_scale", 0.8, min=0, max=3, step=0.01, title="Grain Scale"),
    Choice("grain_type", "gaussian", choices=BW_GRAIN_TYPES, title="Grain Type"),
    Float("local_contrast", 8, min=0, max=60, step=0.1, title="Local Contrast"),
    Int("radius", 16, min=1, max=512, unit="px", title="Radius", description="Local contrast radius"),
    Float("contrast_smoothness", 4, min=0, max=10, step=0.01, title="Contrast Smoothness"),
], **GMIC, tags=["black and white", "monochrome", "film", "grain", "analog"])
def black_and_white_film(img, ctx, film_type, red_level, red_smoothness, green_level, green_smoothness, blue_level,
                         blue_smoothness, gamma, contrast, brightness, hue, saturation, grain_shadows, grain_midtones,
                         grain_highlights, grain_tone_fading, grain_scale, grain_type, local_contrast, radius,
                         contrast_smoothness):
    """G'MIC ``fx_bwfilmsimulate`` (film-stock B&W conversion, toning, grain, local contrast)."""
    return _gmic(img, ctx, "fx_bwfilmsimulate", _idx(BW_FILM_TYPES, film_type), red_level, red_smoothness,
                 green_level, green_smoothness, blue_level, blue_smoothness, gamma, contrast, brightness, hue,
                 saturation, grain_shadows, grain_midtones, grain_highlights, grain_tone_fading, grain_scale,
                 _idx(BW_GRAIN_TYPES, grain_type), local_contrast, min(512, ctx.ipx(radius)), contrast_smoothness,
                 srand=True)


BOOST_SPACES = ["ycbcr_distinct", "ycbcr_mixed", "lab_distinct", "lab_mixed"]


@itb_op("BoostChromaticity", "Boost Chromaticity", GROUP, [
    Float("amplitude", 75, min=0, max=100, title="Amplitude", description="Amount (%)"),
    Choice("color_space", "lab_mixed", choices=BOOST_SPACES, title="Color Space"),
], **GMIC, tags=["chroma", "saturation", "equalize", "vivid"])
def boost_chromaticity(img, ctx, amplitude, color_space):
    """G'MIC ``fx_boost_chroma``: equalise the chroma channels and blend by amplitude."""
    return _gmic(img, ctx, "fx_boost_chroma", amplitude, _idx(BOOST_SPACES, color_space))


CP_VALUE_ACTIONS = ["none", "cut", "cut_and_normalize", "normalize", "threshold"]
CP_TONES = ["all_tones", "shadows", "midtones", "highlights"]
GMIC_CHANNELS = ["all", "rgba", "rgb", "red", "green", "blue", "alpha", "linear_rgb", "linear_red", "linear_green",
                 "linear_blue", "ycbcr", "luminance", "chrominance", "blue_chrominance", "red_chrominance",
                 "green_chrominance", "lab", "lab_lightness", "lab_chrominance", "lab_a", "lab_b", "lch",
                 "lch_chrominance", "lch_chroma", "lch_hue", "hsv", "hsv_hue", "hsv_saturation", "hsv_value", "hsi",
                 "hsi_intensity", "hsl", "hsl_lightness", "cmyk", "cmy", "cyan", "magenta", "yellow", "key", "yiq",
                 "yiq_luma", "yiq_chrominance", "ryb", "ryb_red", "ryb_yellow", "ryb_blue"]


@itb_op("ChannelProcessing", "Channel Processing", GROUP, [
    Float("brightness", 5, min=-100, max=100, title="Brightness"),
    Float("contrast", 12, min=-100, max=100, title="Contrast"),
    Float("gamma", 0, min=-100, max=100, title="Gamma"),
    Float("smoothness", 0, min=0, max=10, step=0.01, title="Smoothness"),
    Choice("value_action", "none", choices=CP_VALUE_ACTIONS, title="Value Action"),
    Float("low_value", 0, min=0, max=100, title="Low Value"),
    Float("high_value", 100, min=0, max=100, title="High Value"),
    Int("quantization", 256, min=1, max=256, title="Quantization"),
    Bool("equalization", False, title="Equalization"),
    Bool("negation", False, title="Negation"),
    Choice("tones_range", "midtones", choices=CP_TONES, title="Tones Range"),
    Float("tones_smoothness", 2, min=0, max=10, step=0.01, title="Tones Smoothness"),
    Choice("channel", "luminance", choices=GMIC_CHANNELS, title="Channel"),
], **GMIC, tags=["channel", "levels", "brightness", "contrast", "gamma", "quantize"])
def channel_processing(img, ctx, brightness, contrast, gamma, smoothness, value_action, low_value, high_value,
                       quantization, equalization, negation, tones_range, tones_smoothness, channel):
    """G'MIC ``fx_channel_processing`` on one colour channel / tonal range."""
    return _gmic(img, ctx, "fx_channel_processing", brightness, contrast, gamma, smoothness,
                 _idx(CP_VALUE_ACTIONS, value_action), low_value, high_value, quantization, equalization, negation,
                 _idx(CP_TONES, tones_range), tones_smoothness, _idx(GMIC_CHANNELS, channel))


@itb_op("ColorTemperature", "Color Temperature", GROUP, [
    Float("color_temperature", 8, min=-20, max=20, step=0.1, title="Color Temperature",
          description="Negative = cooler, positive = warmer"),
    Bool("automatic_color_balance", False, title="Automatic Color Balance"),
], **GMIC, tags=["temperature", "warm", "cool", "white balance"])
def color_temperature(img, ctx, color_temperature, automatic_color_balance):
    """G'MIC ``fx_tk_colortemp``."""
    return _gmic(img, ctx, "fx_tk_colortemp", color_temperature, automatic_color_balance)


@itb_op("DesaturateNorm", "Desaturate Norm", GROUP, [
    Float("red", 0.6, min=0, max=1, step=0.01, title="Red"),
    Float("blue", 0.4, min=0, max=1, step=0.01, title="Blue"),
    Float("exponent", 2.2, min=1, max=3, step=0.01, title="Exponent"),
], **GMIC, tags=["desaturate", "black and white", "norm"])
def desaturate_norm(img, ctx, red, blue, exponent):
    """G'MIC ``fx_gcd_norm_eq`` (norm-based desaturation)."""
    return _gmic(img, ctx, "fx_gcd_norm_eq", red, blue, exponent)


GCA_MAX = ["from_input", "from_reference_color", "maximum_allowed"]
GCA_CHANNELS = ["saturation", "lightness"]


@itb_op("GraduatedColorAccent", "Graduated Color Accent", GROUP, [
    Color("reference_color", "#FF9800", title="Reference Color"),
    Float("amplitude", 2.5, min=0, max=5, step=0.01, title="Amplitude"),
    Choice("maximum_value", "from_input", choices=GCA_MAX, title="Maximum Value"),
    Choice("channel", "saturation", choices=GCA_CHANNELS, title="Channel"),
], **GMIC, tags=["color accent", "selective color", "desaturate"])
def graduated_color_accent(img, ctx, reference_color, amplitude, maximum_value, channel):
    """G'MIC ``fx_gca``: keep saturation/lightness near a reference colour, fade elsewhere."""
    r, g, b = ci.to8(reference_color)[:3]
    return _gmic(img, ctx, "fx_gca", r, g, b, amplitude, _idx(GCA_MAX, maximum_value), _idx(GCA_CHANNELS, channel))


@itb_op("HslAdjustment", "HSL Adjustment", GROUP, [
    Float("contrast", 1.1, min=0, max=2, step=0.01, title="Contrast"),
    Float("level", 0.05, min=-1, max=1, step=0.01, title="Level"),
    Bool("auto_reduce_level", False, title="Auto Reduce Level"),
    Float("hue", 35, min=0, max=360, title="Hue"),
    Float("amount", 0.35, min=0.2, max=1, step=0.01, title="Amount"),
    Bool("auto_set_inverse_hue", False, title="Auto Set Inverse Hue"),
    Float("gamma", 1, min=0, max=2, step=0.01, title="Gamma"),
    Float("output_contrast", 1.05, min=0, max=2, step=0.01, title="Output Contrast"),
], **GMIC, tags=["hsl", "hue", "saturation", "lightness"])
def hsl_adjustment(img, ctx, contrast, level, auto_reduce_level, hue, amount, auto_set_inverse_hue, gamma,
                   output_contrast):
    """G'MIC ``gcd_hsl``."""
    return _gmic(img, ctx, "gcd_hsl", contrast, level, auto_reduce_level, hue, amount, auto_set_inverse_hue, gamma,
                 output_contrast)


def _hsv_band_params(prefix: str, title: str, band: float, width: float, shift: float, sat: float, val: float):
    return [
        Float(f"{prefix}_hue_band", band, min=0, max=360, title=f"{title} Hue Band"),
        Float(f"{prefix}_band_width", width, min=1, max=360, title=f"{title} Band Width"),
        Float(f"{prefix}_hue_shift", shift, min=-180, max=180, title=f"{title} Hue Shift"),
        Float(f"{prefix}_saturation_correction", sat, min=-0.99, max=0.99, step=0.01,
              title=f"{title} Saturation Correction"),
        Float(f"{prefix}_value_correction", val, min=-0.99, max=0.99, step=0.01, title=f"{title} Value Correction"),
    ]


@itb_op("HsvEqualizer", "HSV Equalizer", GROUP,
        _hsv_band_params("first", "First", 25, 55, 8, 0.2, 0.04)
        + _hsv_band_params("second", "Second", 200, 70, -6, 0.14, -0.03)
        + _hsv_band_params("third", "Third", 330, 45, 4, 0.1, 0.06),
        **GMIC, tags=["hsv", "hue", "equalizer", "selective color"])
def hsv_equalizer(img, ctx, **v):
    """G'MIC ``fx_hsv_equalizer`` with three hue bands."""
    args = [v[f"{p}_{k}"] for p in ("first", "second", "third")
            for k in ("hue_band", "band_width", "hue_shift", "saturation_correction", "value_correction")]
    return _gmic(img, ctx, "fx_hsv_equalizer", False, *args)


@itb_op("Lomo", "Lomo", GROUP, [
    Float("vignette_size", 20, min=0, max=100, title="Vignette Size"),
], **GMIC, tags=["lomo", "vignette", "retro", "toy camera"])
def lomo(img, ctx, vignette_size):
    """G'MIC ``fx_lomo``."""
    return _gmic(img, ctx, "fx_lomo", vignette_size)


SEQ_NAMES = [("black", "Black", -12), ("near_black", "Near Black", -6), ("dark_gray", "Dark Gray", 4),
             ("mid_dark_gray", "Mid-Dark Gray", 8), ("middle_gray", "Middle Gray", 12),
             ("mid_light_gray", "Mid-Light Gray", 8), ("light_gray", "Light Gray", 4),
             ("highlights", "Highlights", -2), ("white", "White", -8), ("hue_0", "Hue 0", 8),
             ("hue_45", "Hue 45", 14), ("hue_90", "Hue 90", 6), ("hue_135", "Hue 135", -4),
             ("hue_180", "Hue 180", -8), ("hue_225", "Hue 225", 4), ("hue_270", "Hue 270", 10),
             ("hue_315", "Hue 315", 12), ("hue_360", "Hue 360", 8)]


@itb_op("SaturationEqualizer", "Saturation Equalizer", GROUP,
        [Float(n, d, min=-128, max=128, title=t) for n, t, d in SEQ_NAMES]
        + [Float("rotate_hue_bands", 0, min=-45, max=45, title="Rotate Hue Bands")],
        **GMIC, tags=["saturation", "equalizer", "selective color"])
def saturation_equalizer(img, ctx, **v):
    """G'MIC ``Saturation_EQ_p`` (saturation per lightness band and per hue band)."""
    return _gmic(img, ctx, "Saturation_EQ_p", *[v[n] for n, _, _ in SEQ_NAMES], v["rotate_hue_bands"])


SD_MODES = ["reference_color", "all_but_reference_color"]
SD_MAX = ["from_input", "from_reference_color", "maximum_value"]


@itb_op("SelectiveDesaturation", "Selective Desaturation", GROUP, [
    Color("reference_color", "#E53935", title="Reference Color"),
    Choice("desaturate", "all_but_reference_color", choices=SD_MODES, title="Desaturate"),
    Float("strength", 5, min=0, max=10, step=0.01, title="Strength"),
    Int("regularization", 4, min=0, max=20, title="Regularization"),
    Choice("maximum_saturation", "from_input", choices=SD_MAX, title="Maximum Saturation"),
], **GMIC, tags=["selective color", "color splash", "desaturate"])
def selective_desaturation(img, ctx, reference_color, desaturate, strength, regularization, maximum_saturation):
    """G'MIC ``fx_selective_desaturation``."""
    r, g, b = ci.to8(reference_color)[:3]
    return _gmic(img, ctx, "fx_selective_desaturation", r, g, b, _idx(SD_MODES, desaturate), strength, regularization,
                 _idx(SD_MAX, maximum_saturation))


@itb_op("SpecificSaturation", "Specific Saturation", GROUP, [
    Float("input", 1.2, min=0, max=2, step=0.01, title="Input"),
    Float("opposing", 0.9, min=0, max=2, step=0.01, title="Opposing"),
    Int("hue", 35, min=0, max=360, title="Hue"),
    Float("level", 0.08, min=-0.5, max=0.5, step=0.01, title="Level"),
    Float("gamma", 1, min=0.2, max=1.8, step=0.01, title="Gamma"),
    Float("contrast", 1.1, min=0.2, max=1.8, step=0.01, title="Contrast"),
    Bool("srgb_conversion", True, title="sRGB Conversion"),
    Bool("process_top_layer_only", False, title="Process Top Layer Only"),
], **GMIC, tags=["saturation", "hue", "selective color"])
def specific_saturation(img, ctx, input, opposing, hue, level, gamma, contrast, srgb_conversion,
                        process_top_layer_only):
    """G'MIC ``gcd_hio_levels``."""
    return _gmic(img, ctx, "gcd_hio_levels", input, opposing, hue, level, gamma, contrast, srgb_conversion,
                 process_top_layer_only)


TCS_RANGES = ["global", "wide", "narrow"]


@itb_op("TargetColorSpot", "Target Color Spot", GROUP, [
    Float("x_position", 50, min=0, max=100, title="X Position", description="Sampled spot X (% of width)"),
    Float("y_position", 50, min=0, max=100, title="Y Position", description="Sampled spot Y (% of height)"),
    Color("target_color", "#FFA726", title="Target Color"),
    Choice("chromaticity_range", "wide", choices=TCS_RANGES, title="Chromaticity Range"),
], **GMIC, tags=["color match", "white balance", "curves"])
def target_color_spot(img, ctx, x_position, y_position, target_color, chromaticity_range):
    """G'MIC ``fx_gcd_color_target``: curves that move the colour at a spot to a target colour."""
    r, g, b = ci.to8(target_color)[:3]
    return _gmic(img, ctx, "fx_gcd_color_target", x_position, y_position, r, g, b,
                 _idx(TCS_RANGES, chromaticity_range))


TB_LIGHTING = ["automatic", "automatic_scan_all_hues", "standard_no_scan"]


@itb_op("TemperatureBalance", "Temperature Balance", GROUP, [
    Float("saturation", 0.2, min=-1, max=1, step=0.01, title="Saturation"),
    Float("level", 0.1, min=-1, max=1, step=0.01, title="Level"),
    Float("color", 1.15, min=0, max=2, step=0.01, title="Color"),
    Choice("lighting", "automatic_scan_all_hues", choices=TB_LIGHTING, title="Lighting"),
], **GMIC, tags=["temperature", "white balance", "warm", "cool"])
def temperature_balance(img, ctx, saturation, level, color, lighting):
    """G'MIC ``gcd_temp_balance``."""
    return _gmic(img, ctx, "gcd_temp_balance", saturation, level, color, _idx(TB_LIGHTING, lighting))


@itb_op("Unpurple", "Unpurple", GROUP, [
    Float("intensity_of_purple_fringe", 1, min=0, max=1, step=0.01, title="Intensity of purple fringe"),
    Float("minimum_brightness", 0, min=0, max=1, step=0.01, title="Minimum brightness"),
    Float("minimum_red_blue_ratio", 0, min=0, max=1, step=0.01, title="Minimum red:blue ratio in the fringe"),
    Float("maximum_red_blue_ratio", 0.33, min=0, max=1, step=0.01, title="Maximum red:blue ratio in the fringe"),
    Int("blur_standard_deviation", 5, min=1, max=10, unit="px", title="Blur standard deviation"),
    Bool("gentle_mode", False, title="Gentle mode",
         description="Overrides minimum brightness and minimum red:blue ratio"),
    Int("bit_depth", 0, min=0, max=32, title="Bit depth", description="0 = use the image maximum"),
], **GMIC, tags=["purple fringe", "chromatic aberration", "defringe", "repair"])
def unpurple(img, ctx, intensity_of_purple_fringe, minimum_brightness, minimum_red_blue_ratio,
             maximum_red_blue_ratio, blur_standard_deviation, gentle_mode, bit_depth):
    """G'MIC ``unpurple`` (purple-fringe removal)."""
    return _gmic(img, ctx, "unpurple", intensity_of_purple_fringe, minimum_brightness, minimum_red_blue_ratio,
                 maximum_red_blue_ratio, min(10, ctx.ipx(blur_standard_deviation)), gentle_mode, bit_depth)


# ===========================================================================
# GPUImage shaders
# ===========================================================================

@itb_op("ColorBalance", "Color balance", GROUP, [
    FloatList("values", [0.0] * 9, length=9, title="Values",
              description="[highlights r,g,b, midtones r,g,b, shadows r,g,b] shifts. As in ImageToolbox, the "
                          "midtones blue shift is read from the 7th value (shadows red)."),
], upstream="GPUImage", tags=["color balance", "shadows", "midtones", "highlights", "tint"])
def color_balance(img, ctx, values):
    """GPUImageColorBalanceFilter (preserveLuminosity = true)."""
    v = [float(x) for x in values]
    hi = np.array(v[0:3], np.float64)
    mid = np.array([v[3], v[4], v[6]], np.float64)  # upstream ColorBalanceFilter indexing
    sh = np.array(v[6:9], np.float64)
    return ci.gpuimage_color_balance(np.ascontiguousarray(img, F32), sh, mid, hi)


@itb_op("ColorMatrix4x4", "Color Matrix 4x4", GROUP, [
    FloatList("matrix", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], length=16, title="Values",
              description="Row-major 4x4 matrix applied to (r, g, b, a): out_row = row . (r, g, b, a)"),
], upstream="GPUImage", alpha="full", tags=["color matrix", "channel mixer"])
def color_matrix_4x4(img, ctx, matrix):
    """GPUImageColorMatrixFilter (intensity 1): ``textureColor * colorMatrix`` with the array uploaded column-major."""
    m = np.asarray(matrix, F32).reshape(4, 4)
    if _has_alpha(img):
        return ci.chunked(lambda p: p @ m.T, img)
    out = ci.chunked(lambda p: p @ m[:, :3].T + m[:, 3], img)  # implicit alpha = 1
    return out[..., :3] if np.array_equal(m[3], [0, 0, 0, 1]) else out


@itb_op("FalseColor", "False color", GROUP, [
    Color("first_color", "#FFFF00", title="First color", description="Colour for dark tones"),
    Color("second_color", "#FF00FF", title="Second color", description="Colour for bright tones"),
], upstream="GPUImage", tags=["false color", "duotone", "gradient map"])
def false_color(img, ctx, first_color, second_color):
    """GPUImageFalseColorFilter: mix(first, second, luminance)."""
    c1, c2 = _rgb(first_color), _rgb(second_color)
    w = np.array([0.2125, 0.7154, 0.0721], F32)
    return ci.chunked(lambda p: c1 + (c2 - c1) * (p @ w)[..., None], img)


@itb_op("Hue", "Hue", GROUP, [
    Float("hue", 90, min=0, max=255, title="Hue", description="Hue rotation in degrees (ImageToolbox range)"),
], upstream="GPUImage", tags=["hue", "rotate", "color shift"])
def hue(img, ctx, hue):
    """GPUImageHueFilter: rotate the (I, Q) chroma of YIQ by -hue."""
    a = (math.fmod(hue, 360.0)) * math.pi / 180.0
    to_yiq = np.array([[0.299, 0.587, 0.114], [0.595716, -0.274453, -0.321263], [0.211456, -0.522591, 0.31135]])
    to_rgb = np.array([[1.0, 0.9563, 0.6210], [1.0, -0.2721, -0.6474], [1.0, -1.1070, 1.7046]])
    rot = np.array([[1, 0, 0], [0, math.cos(a), math.sin(a)], [0, -math.sin(a), math.cos(a)]])
    return color_matrix_3x3(img, (to_rgb @ rot @ to_yiq).ravel())


@itb_op("Opacity", "Opacity", GROUP, [
    Float("opacity", 0.5, min=0, max=1, step=0.01, title="Opacity"),
], upstream="GPUImage", alpha="full", tags=["opacity", "transparency", "alpha"])
def opacity(img, ctx, opacity):
    """GPUImageOpacityFilter: alpha *= opacity."""
    return np.dstack([img[..., :3], _alpha(img) * opacity]).astype(F32)


@itb_op("Posterize", "Posterize", GROUP, [
    Int("levels", 5, min=1, max=40, title="Levels", description="Colour levels per channel"),
], upstream="GPUImage", alpha="full", tags=["posterize", "levels", "quantize"])
def posterize(img, ctx, levels):
    """GPUImagePosterizeFilter: floor(c * levels + 0.5) / levels on every channel (incl. alpha)."""
    lv = F32(levels)
    return ci.chunked(lambda p: np.floor(p * lv + F32(0.5)) / lv, img)


@itb_op("RGB", "RGB filter", GROUP, [
    Color("color", "#00FF00", title="Color", description="Per-channel multipliers"),
], upstream="GPUImage", tags=["rgb", "channel", "tint", "multiply"])
def rgb_filter(img, ctx, color):
    """GPUImageRGBFilter: multiply each channel."""
    return (img * _rgb(color)).astype(F32)


@itb_op("Solarize", "Solarize", GROUP, [
    Float("threshold", 0.5, min=0, max=1, step=0.01, title="Threshold"),
], upstream="GPUImage", tags=["solarize", "invert", "sabattier"])
def solarize(img, ctx, threshold):
    """GPUImageSolarizeFilter: invert pixels whose luminance is <= threshold."""
    w = np.array([0.2125, 0.7154, 0.0721], F32)
    return ci.chunked(lambda p: np.abs((p @ w <= threshold).astype(F32)[..., None] - p), img)


_DIAG = [[0.0, 0.0], [1.0, 1.0]]


@itb_op("ToneCurves", "Tone Curves", GROUP, [
    Points("rgb", _DIAG, min_items=2, max_items=ci.CURVE_MAX_POINTS, min=0, max=1, title="RGB",
           description="Composite (luma) curve control points [[x, y], ...]"),
    Points("red", _DIAG, min_items=2, max_items=ci.CURVE_MAX_POINTS, min=0, max=1, title="Red"),
    Points("green", _DIAG, min_items=2, max_items=ci.CURVE_MAX_POINTS, min=0, max=1, title="Green"),
    Points("blue", _DIAG, min_items=2, max_items=ci.CURVE_MAX_POINTS, min=0, max=1, title="Blue"),
], upstream="GPUImage", tags=["curves", "tone curve", "contrast"])
def tone_curves(img, ctx, rgb, red, green, blue):
    """GPUImageToneCurveFilter (natural cubic spline, composite + per-channel offset LUT)."""
    curves = [ci.itb_normalize_curve(p) for p in (rgb, red, green, blue)]
    if all(ci.curve_is_default(c) for c in curves):
        return img
    comp = ci.gpuimage_prepared_curve(curves[0])
    idx = np.arange(256, dtype=np.float64)
    luts = [np.clip(idx + ci.gpuimage_prepared_curve(c) + comp, 0, 255).astype(np.int32).astype(F32)
            for c in curves[1:]]
    return ci.chunked(lambda p: np.stack([ci.sample_curve_texture(p[..., c], luts[c]) for c in range(3)], -1), img)


# ===========================================================================
# aire
# ===========================================================================

@itb_op("ColorMatrix3x3", "Color Matrix 3x3", GROUP, [
    FloatList("matrix", [1, 0, 0, 0, 1, 0, 0, 0, 1], length=9, title="Values",
              description="Row-major 3x3 matrix: out = M . (r, g, b)"),
], upstream="aire", tags=["color matrix", "channel mixer"])
def color_matrix_3x3_op(img, ctx, matrix):
    """aire ``colorMatrix``."""
    return color_matrix_3x3(img, matrix)


@itb_op("Grayscale", "Gray Scale", GROUP, [
    Float("red", 0.299, min=0, max=1, step=0.001, title="Red"),
    Float("green", 0.587, min=0, max=1, step=0.001, title="Green"),
    Float("blue", 0.114, min=0, max=1, step=0.001, title="Blue"),
], upstream="aire", tags=["grayscale", "black and white", "desaturate"])
def grayscale(img, ctx, red, green, blue):
    """aire ``grayscale``: weighted sum of linearised channels, written back without re-encoding."""
    w = np.array([red, green, blue], F32)

    def px(p):
        g = np.clip(ci.srgb_to_linear_045(p) @ w, 0, 1)[..., None]
        return np.repeat(g, 3, axis=2)

    return ci.chunked(px, img)


@itb_op("Monochrome", "Monochrome", GROUP, [
    Float("strength", 1, min=0, max=1, step=0.01, title="Strength"),
    Color("color", [0.6, 0.45, 0.3], title="Color"),
], upstream="aire", tags=["monochrome", "tint", "duotone", "sepia"])
def monochrome(img, ctx, strength, color):
    """aire ``monochrome`` (MonochromeToneMapper: overlay of the tint colour on linear luma)."""
    c = _rgb(color)
    s = F32(strength)
    w = np.array([0.299, 0.587, 0.114], F32)

    def px(p):
        lin = ci.srgb_to_linear_045(p)
        lum = (lin @ w)[..., None]
        new = np.where(lum < 0.5, 2.0 * lum * c, 1.0 - 2.0 * (1.0 - lum) * (1.0 - c))
        return ci.linear_to_srgb_std(np.clip(new * s + lin * (1.0 - s), 0, 1))

    return ci.chunked(px, img)


PT_SPACES = ["oklab", "lab", "luv", "lalphabeta"]


def _load_reference(ctx, path):
    from ...core.image import fit_within
    from ...core.io import load_image

    ref, _ = load_image(ctx.resolve_file(path, "reference image"))
    ref, _ = fit_within(np.ascontiguousarray(ref[..., :3]), 1000)  # ImageToolbox loads it at size 1000
    return np.clip(ref, 0, 1).astype(F32)


@itb_op("PaletteTransferVariant", "Palette Transfer Variant", GROUP, [
    Float("strength", 1, min=0, max=1, step=0.01, title="Strength"),
    Choice("color_space", "oklab", choices=PT_SPACES, title="Color Space"),
    File("target_image", "", file_kind="image", required=True, title="Target Image",
         description="Image whose colour palette is copied"),
], upstream="aire", tags=["palette transfer", "color transfer", "reference", "match"])
def palette_transfer_variant(img, ctx, strength, color_space, target_image):
    """aire ``copyPalette`` (palette-transfer crate: Reinhard mean/std transfer per channel)."""
    return ci.aire_copy_palette(img, _load_reference(ctx, target_image), strength, color_space)


@itb_op("Saturation", "Saturation", GROUP, [
    Float("strength", 2, min=0, max=2, step=0.01, title="Strength"),
    Bool("enable_tonemapping", True, title="Enable Tonemapping",
         description="Apply the ACES (Hill) tone curve after scaling chroma"),
], upstream="aire", tags=["saturation", "vivid", "oklab"])
def saturation(img, ctx, strength, enable_tonemapping):
    """aire ``saturation``: scale OKLab a/b, optional ACES Hill tone map in linear light."""
    def px(p):
        lab = ci.lin_to_oklab(ci.cu_srgb_to_linear(p))
        lab[..., 1:] *= F32(strength)
        lin = ci.oklab_to_lin(lab)
        if enable_tonemapping:
            lin = ci.aces_hill(lin)
        return ci.cu_linear_to_srgb(np.clip(lin, 0, 1))

    return ci.chunked(px, img)


@itb_op("Vibrance", "Vibrance", GROUP, [
    Float("vibrance", 3, min=-5, max=5, step=0.01, title="Vibrance"),
], upstream="aire", tags=["vibrance", "saturation"])
def vibrance(img, ctx, vibrance):
    """aire ``vibrance``: add (max - mean) * vibrance to every channel."""
    v = F32(vibrance)
    return ci.chunked(lambda p: p + np.clip((p.max(-1) - p.sum(-1) / F32(3)) * v, -1, 1)[..., None], img)


# ===========================================================================
# Trickle
# ===========================================================================

@itb_op("ColorOverlay", "Color filter", GROUP, [
    Color("color", [1.0, 1.0, 0.0, 0.3], title="Color", description="Colour drawn over the image (alpha = opacity)"),
], upstream="Trickle", alpha="full", tags=["overlay", "tint", "color fill"])
def color_overlay(img, ctx, color):
    """Trickle ``drawColorAbove``: fill the canvas with the colour using cairo OVER."""
    c, ca = _rgb(color), F32(color[3])
    if not _has_alpha(img):
        return ci.chunked(lambda p: c * ca + p * (1 - ca), img)

    def px(p):
        a = p[..., 3]
        out_a = ca + a * (1 - ca)
        prem = c * ca + p[..., :3] * (a * (1 - ca))[..., None]
        rgb = np.where(out_a[..., None] > 0, prem / np.maximum(out_a, 1e-12)[..., None], 0.0)
        return np.dstack([rgb, out_a])

    return ci.chunked(px, img)


@itb_op("ColorPoster", "Color Poster", GROUP, [
    Float("strength", 0.5, min=0, max=1, step=0.01, title="Strength", description="Larger = fewer shades"),
    Color("color", "#4DFFE4", title="Color"),
], upstream="Trickle", tags=["posterize", "palette", "monochrome", "shades"])
def color_poster(img, ctx, strength, color):
    """Trickle ``colorPosterize`` with ``TrickleUtils.generateShades`` (value shades of one colour)."""
    h, s, _ = ci.sk_rgb_to_hsv(*ci.to8(color)[:3])
    step = int(math.floor(max(50.0 * strength, 1.0) + 0.5))
    pal = np.array([ci.sk_hsv_to_rgb(h, s, i / 100.0) for i in range(2, 99, step)], F32)
    flat = np.ascontiguousarray(np.clip(img, 0, 1).reshape(-1, 3) * 255.0, dtype=F32)
    return (ci.nearest_palette(flat, pal).reshape(img.shape) / 255.0).astype(F32)


@itb_op("PaletteTransfer", "Palette Transfer", GROUP, [
    Float("strength", 1, min=0, max=1, step=0.01, title="Strength"),
    File("target_image", "", file_kind="image", required=True, title="Target Image",
         description="Image whose colour palette is copied"),
], upstream="Trickle", tags=["palette transfer", "color transfer", "reference", "match"])
def palette_transfer(img, ctx, strength, target_image):
    """Trickle ``transferPalette``: Reinhard mean/std transfer in CIELab."""
    return ci.trickle_transfer_palette(img, _load_reference(ctx, target_image), strength)


def _replace_color(img, tolerance, source, target):
    src8 = np.array(ci.to8(source)[:3], F32)
    limit = F32(np.cbrt(3.0) * tolerance)
    tc, ta = _rgb(target), F32(target[3])
    with_alpha = _has_alpha(img) or ci.to8(target)[3] != 255

    def px(p):
        rgb8 = np.round(np.clip(p[..., :3], 0, 1) * F32(255))
        mask = np.sqrt(((rgb8 - src8) ** 2).sum(-1)) / F32(255) <= limit
        rgb = np.where(mask[..., None], tc, p[..., :3])
        if not with_alpha:
            return rgb
        return np.dstack([rgb, np.where(mask, ta, _alpha(p))])

    return ci.chunked(px, img)


@itb_op("RemoveColor", "Remove Color", GROUP, [
    Float("tolerance", 0, min=0, max=1, step=0.01, title="Tolerance"),
    Color("color_to_remove", "#000000", title="Color to Remove"),
], upstream="Trickle", alpha="full", tags=["remove color", "transparency", "chroma key"])
def remove_color(img, ctx, tolerance, color_to_remove):
    """ReplaceColor with a transparent target (Trickle ``replaceColor``)."""
    return _replace_color(img, tolerance, color_to_remove, (0.0, 0.0, 0.0, 0.0))


@itb_op("ReplaceColor", "Replace Color", GROUP, [
    Float("tolerance", 0, min=0, max=1, step=0.01, title="Tolerance"),
    Color("color_to_replace", "#000000", title="Color to Replace"),
    Color("target_color", "#FFFFFF", title="Target Color"),
], upstream="Trickle", alpha="full", tags=["replace color", "recolor"])
def replace_color(img, ctx, tolerance, color_to_replace, target_color):
    """Trickle ``replaceColor``: pixels within ``cbrt(3) * tolerance`` (RGB distance / 255) get the target ARGB."""
    return _replace_color(img, tolerance, color_to_replace, target_color)


@itb_op("TriTone", "Tri Tone", GROUP, [
    Color("shadows_color", "#FF003B", title="Shadows"),
    Color("midtones_color", "#831111", title="Midtones"),
    Color("highlights_color", "#FF0099", title="Highlights"),
], upstream="Trickle", tags=["tritone", "duotone", "gradient map"])
def tri_tone(img, ctx, shadows_color, midtones_color, highlights_color):
    """Trickle ``tritone``: luma (BT.601) mapped through shadow -> mid -> highlight colour ramps."""
    opaque = min(ci.to8(c)[3] for c in (shadows_color, midtones_color, highlights_color)) == 255
    n = 3 if opaque else 4  # 4th channel: ramp alpha (multiplied with the image alpha)
    sh, mid, hi = (np.asarray(c[:n], F32) for c in (shadows_color, midtones_color, highlights_color))
    w = np.array([0.299, 0.587, 0.114], F32) * 255

    def px(p):
        g = p @ w
        t1 = np.clip(g / 127.0, 0, 1)[..., None]
        t2 = np.clip((g - 127.0) / 128.0, 0, 1)[..., None]
        return np.where((g <= 127.0)[..., None], sh + (mid - sh) * t1, mid + (hi - mid) * t2)

    return ci.chunked(px, img)


# ===========================================================================
# jhlabs (ImageToolboxLibs Rust port)
# ===========================================================================

@itb_op("ChannelMix", "Channel Mix", GROUP, [
    Int("blue_green", 255, min=0, max=255, title="Blue green", description="Mix of green vs blue fed into red"),
    Int("red_blue", 255, min=0, max=255, title="Red blue", description="Mix of blue vs red fed into green"),
    Int("green_red", 255, min=0, max=255, title="Green red", description="Mix of red vs green fed into blue"),
    Int("into_red", 255, min=0, max=255, title="Into red"),
    Int("into_green", 255, min=0, max=255, title="Into green"),
    Int("into_blue", 127, min=0, max=255, title="Into blue"),
], upstream="jhlabs", tags=["channel mixer", "channels"])
def channel_mix(img, ctx, blue_green, red_blue, green_red, into_red, into_green, into_blue):
    """jhlabs ChannelMixFilter."""
    # The mix is linear in (r, g, b): express it as a 3x3 matrix (rows = outputs).
    ir, ig, ib = into_red / 255, into_green / 255, into_blue / 255
    bg, rb, gr = blue_green / 255, red_blue / 255, green_red / 255
    m = [1 - ir, ir * bg, ir * (1 - bg),
         ig * (1 - rb), 1 - ig, ig * rb,
         ib * gr, ib * (1 - gr), 1 - ib]
    return color_matrix_3x3(img, m)


@itb_op("ColorMask", "Color Mask", GROUP, [
    Color("color", "#00FFFF", title="Color", description="ARGB bit mask ANDed with every pixel"),
], upstream="jhlabs", alpha="full", tags=["mask", "channels", "bitwise"])
def color_mask(img, ctx, color):
    """jhlabs MaskFilter: ``pixel &= mask`` on the 8-bit ARGB value."""
    m = ci.to8(color)
    chans = [img[..., c] for c in range(img.shape[2])]
    if len(chans) == 3 and m[3] != 255:
        chans.append(np.ones(img.shape[:2], F32))
    out = []
    for c, ch in enumerate(chans):
        if m[c] == 255:
            out.append(ch)
        elif m[c] == 0:
            out.append(np.zeros_like(ch))
        else:
            q = np.round(np.clip(ch, 0, 1) * 255.0).astype(np.uint8)
            out.append((q & np.uint8(m[c])).astype(F32) / 255.0)
    return np.stack(out, -1).astype(F32)


# ===========================================================================
# ImageToolbox own implementations
# ===========================================================================

SPECTRAL = ["#9E0142", "#D53E4F", "#F46D43", "#FEE08B", "#E6F598", "#66C2A5", "#3288BD", "#5E4FA2"]


@itb_op("GradientMap", "Gradient map", GROUP, [
    Gradient("gradient", [[i / (len(SPECTRAL) - 1), c] for i, c in enumerate(SPECTRAL)], title="Gradient",
             description="Colours for dark -> bright (default: ImageToolbox 'Spectral' palette)"),
], upstream="ImageToolbox", tags=["gradient map", "false color", "colorize"])
def gradient_map(img, ctx, gradient):
    """ImageToolbox GradientMapFilter: luma (77R+150G+29B)/256 looks up the gradient."""
    w = np.array([77.0, 150.0, 29.0], F32) / 256
    pos = np.array([p for p, _ in gradient], np.float64)
    cols = np.array([c[:3] for _, c in gradient], np.float64)

    def px(p):
        lum = p @ w
        return np.stack([np.interp(lum, pos, cols[:, c]) for c in range(3)], -1)

    return ci.chunked(px, img)


def _hue_shift_params():
    return [
        Float("intensity", 2, min=-10, max=10, step=0.001, title="Intensity"),
        Float("saturation", 1, min=0, max=2, step=0.001, title="Saturation"),
        Float("hue", 0, min=-1, max=1, step=0.001, title="Hue", description="Hue offset (turns)"),
        Float("offset_x", 0, min=-1, max=1, step=0.001, title="Offset X"),
        Float("offset_y", 0, min=-1, max=1, step=0.001, title="Offset Y"),
        Float("scale", 1, min=0.05, max=8, step=0.001, title="Scale"),
        Float("rotation", 0, min=-180, max=180, step=0.001, title="Rotation"),
    ]


def _hue_shift(img, mode, intensity, saturation, hue, offset_x, offset_y, scale, rotation):
    H, W = img.shape[:2]

    def px(p, y0):
        x, y = ci.procedural_field(H, W, offset_x, offset_y, scale, rotation, rows=(y0, y0 + p.shape[0]))
        if mode == "bands":
            d = y
        elif mode == "tunnel":
            d = np.sqrt(x * x + y * y)
        else:
            d = np.where(x * x + y * y < 0.00000001, 0.0, np.arctan2(y, x)) / F32(2.0 * math.pi)
        hsv = ci.shader_rgb_to_hsv(p)
        h = hsv[..., 0] + d * F32(intensity) + F32(hue)
        hsv[..., 0] = h - np.floor(h)
        hsv[..., 1] = np.clip(hsv[..., 1] * F32(saturation), 0, 1)
        return ci.shader_hsv_to_rgb(hsv)

    return ci.chunked(px, img, with_y=True)


@itb_op("HueShiftBands", "Hue Shift Bands", GROUP, _hue_shift_params(), upstream="ImageToolbox",
        tags=["hue", "rainbow", "bands", "procedural"])
def hue_shift_bands(img, ctx, **p):
    """ImageToolbox procedural shader: hue shifted by the vertical field coordinate."""
    return _hue_shift(img, "bands", **p)


@itb_op("HueShiftFan", "Hue Shift Fan", GROUP, _hue_shift_params(), upstream="ImageToolbox",
        tags=["hue", "rainbow", "fan", "angular", "procedural"])
def hue_shift_fan(img, ctx, **p):
    """ImageToolbox procedural shader: hue shifted by the angle around the field centre."""
    return _hue_shift(img, "fan", **p)


@itb_op("HueShiftTunnel", "Hue Shift Tunnel", GROUP, _hue_shift_params(), upstream="ImageToolbox",
        tags=["hue", "rainbow", "radial", "tunnel", "procedural"])
def hue_shift_tunnel(img, ctx, **p):
    """ImageToolbox procedural shader: hue shifted by the distance to the field centre."""
    return _hue_shift(img, "tunnel", **p)


@itb_op("Neon", "Neon", GROUP, [
    Int("amount", 1, min=1, max=25, unit="px", title="Amount", description="Sobel line width"),
    Float("strength", 0.26, min=0, max=1, step=0.01, title="Strength", description="Pre-sharpening"),
    Color("color", "#FF00FF", title="Color"),
], upstream="ImageToolbox", tags=["neon", "edges", "glow", "sobel"])
def neon(img, ctx, amount, strength, color):
    """ImageToolbox Neon = GPUImage Sharpen(strength) -> Sobel edge detection(amount) -> RGB filter(color)."""
    s = float(strength)
    # 1) GPUImageSharpenFilter: 4-neighbour unsharp with clamp-to-edge sampling (8-bit result -> clamp)
    kern = np.array([[0, -s, 0], [-s, 1 + 4 * s, -s], [0, -s, 0]], F32)
    sharp = cv2.filter2D(np.ascontiguousarray(img), cv2.CV_32F, kern, borderType=cv2.BORDER_REPLICATE)
    np.clip(sharp, 0, 1, out=sharp)
    # 2) GPUImageSobelEdgeDetectionFilter: grayscale, 3x3 Sobel sampled `amount` texels apart
    gray = ci.chunked(lambda p: p @ np.array([0.2125, 0.7154, 0.0721], F32), sharp)
    k = ctx.ipx(amount)
    taps = np.zeros(2 * k + 1, F32)
    smooth, deriv = taps.copy(), taps.copy()
    smooth[[0, k, 2 * k]] = (1, 2, 1)
    deriv[[0, 2 * k]] = (-1, 1)
    h = cv2.sepFilter2D(gray, cv2.CV_32F, smooth, deriv, borderType=cv2.BORDER_REPLICATE)  # bottom - top
    v = cv2.sepFilter2D(gray, cv2.CV_32F, deriv, smooth, borderType=cv2.BORDER_REPLICATE)  # right - left
    mag = np.clip(cv2.magnitude(h, v), 0, 1)
    # 3) RGBFilter(color)
    c = _rgb(color)
    return ci.chunked(lambda m: m[..., None] * c, mag)
