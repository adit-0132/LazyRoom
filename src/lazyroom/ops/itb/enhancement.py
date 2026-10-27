"""ImageToolbox "Enhancement" group filters.

G'MIC filters build the exact command strings of ImageToolboxLibs' gmic command
builders (``com.t8rin.gmic.filters``); enum arguments are passed as their
integer ``value`` and Booleans as 0/1.  Parameters measured in pixels are
scaled with the render context so previews match exports (at full resolution
the command is identical to ImageToolbox's).
"""
from __future__ import annotations

import functools
import re

from ...core.errors import ErrorCode, LazyRoomError
from ...core.params import Bool, Choice, Float, Int
from . import _enhancement_impl as impl
from ._common import gmic_cmd, itb_op, run_gmic, snake

GROUP = "enhancement"

# ---------------------------------------------------------------------------
# Upstream enums (order == G'MIC integer value)
# ---------------------------------------------------------------------------

GMIC_CHANNELS = [
    "All", "Rgba", "Rgb", "Red", "Green", "Blue", "Alpha", "LinearRgb", "LinearRed", "LinearGreen",
    "LinearBlue", "YCbCr", "Luminance", "Chrominance", "BlueChrominance", "RedChrominance",
    "GreenChrominance", "Lab", "LabLightness", "LabChrominance", "LabA", "LabB", "Lch", "LchChrominance",
    "LchChroma", "LchHue", "Hsv", "HsvHue", "HsvSaturation", "HsvValue", "Hsi", "HsiIntensity", "Hsl",
    "HslLightness", "Cmyk", "Cmy", "Cyan", "Magenta", "Yellow", "Key", "Yiq", "YiqLuma", "YiqChrominance",
    "Ryb", "RybRed", "RybYellow", "RybBlue",
]
CONSTRAINED_SHARPEN_CHANNELS = [
    "All", "Rgba", "Rgb", "Red", "Green", "Blue", "Alpha", "LinearRgb", "LinearRed", "LinearGreen",
    "LinearBlue", "Luminance", "Chrominance", "BlueChrominance", "RedChrominance", "GreenChrominance",
    "LabLightness", "LabChrominance", "LabA", "LabB", "LchChrominance", "LchChroma", "LchHue", "HsvHue",
    "HsvSaturation", "HsvValue", "HsiIntensity", "HslLightness", "Cyan", "Magenta", "Yellow", "Key",
]
PARALLELISM = ["Auto", "OneThread", "TwoThreads", "FourThreads", "EightThreads", "SixteenThreads"]
VALUE_ACTION = ["None", "Cut", "Normalize"]                       # GmicValueAction
CONSTRAINED_VALUE_ACTION = ["NoneOption", "Cut", "Normalize"]     # ConstrainedSharpenValueAction
SHARPEN_VALUE_ACTION = ["Cut", "NormalizeLuma"]                   # SharpenValueAction
DETAIL_SMOOTHING = ["Gaussian", "Bilateral", "Diffusion"]
DECONVOLUTION_BLUR = ["Exponential", "Gaussian"]
DEBLUR_REGULARIZATION = ["Tikhonov", "MeanCurvature", "TotalVariation"]
BANDING_TILES = ["NoTiling", "Px64", "Px128", "Px256", "Px512", "Px1024", "Px2048"]
SKIN_ESTIMATION = ["None", "Manual", "Automatic"]
SKIN_SMOOTHNESS_TYPE = ["Gaussian", "Bilateral"]


def _key(upstream: str) -> str:
    return {"NoneOption": "none", "YCbCr": "ycbcr"}.get(upstream, snake(upstream))


def _label(upstream: str) -> str:
    if upstream in ("YCbCr", "NoneOption"):
        return {"YCbCr": "YCbCr", "NoneOption": "None"}[upstream]
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", upstream)


def _choice(name: str, options: list[str], default: str, title: str, **kw) -> Choice:
    keys = [_key(o) for o in options]
    return Choice(name, _key(default), title=title, choices=keys, labels={k: _label(o) for k, o in zip(keys, options)},
                  **kw)


@functools.lru_cache(maxsize=None)
def _keys(options: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_key(o) for o in options)


def _ev(options: list[str], key: str) -> int:
    """Map a snake_case choice back to the upstream enum ``value`` (== its index)."""
    return _keys(tuple(options)).index(key)


def _channel(default: str = "All", options: list[str] = GMIC_CHANNELS) -> Choice:
    return _choice("channel", options, default, "Channel", description="Colour channels the filter is applied to")


def _parallelism() -> Choice:
    return _choice("parallelism", PARALLELISM, "Auto", "Parallelism", advanced=True,
                   description="G'MIC tiling/parallelism (performance only)")


def _overlap(default: int) -> Int:
    return Int("spatial_overlap", default, min=0, max=256, title="Spatial Overlap", advanced=True,
               description="Tile overlap used by the parallel G'MIC run")


def _gmic(img, ctx, name, *args):
    return run_gmic(img, gmic_cmd(name, *args), ctx=ctx)


_GMIC = dict(upstream="G'MIC", requires=["gmic"])


# ---------------------------------------------------------------------------
# G'MIC filters
# ---------------------------------------------------------------------------

@itb_op("AlphaSharpen", "Alpha Sharpen", GROUP, [
    Float("amplitude", 100, min=0, max=400, title="Amplitude"),
    Int("scale", 5, min=1, max=10, title="Scale", description="Number of scales"),
    Float("anisotropy", 50, min=0, max=100, title="Anisotropy"),
    Float("minimize_alpha", 100, min=0, max=100, title="Minimize Alpha"),
    _channel(),
], tags=["sharpen", "details", "multiscale"], **_GMIC)
def alpha_sharpen(img, ctx, amplitude, scale, anisotropy, minimize_alpha, channel):
    """G'MIC fx_sharpen_alpha (multi-scale sharpening with alpha boosting)."""
    return _gmic(img, ctx, "fx_sharpen_alpha", amplitude, scale, anisotropy, minimize_alpha,
                 _ev(GMIC_CHANNELS, channel))


@itb_op("BandingDenoise", "Banding Denoise", GROUP, [
    Float("v_cutoff", 5, min=0, max=50, title="V Cutoff"),
    Float("h_cutoff", 5, min=0, max=50, title="H Cutoff"),
    Float("space", 5, min=0, max=20, title="Space", unit="px", description="Bilateral spatial scale"),
    Float("value", 10, min=0, max=100, title="Value", description="Bilateral value scale"),
    _choice("size_of_tiles", BANDING_TILES, "NoTiling", "Size of Tiles"),
], tags=["denoise", "banding", "repair"], **_GMIC)
def banding_denoise(img, ctx, v_cutoff, h_cutoff, space, value, size_of_tiles):
    """G'MIC banding_denoise_v2 (remove horizontal/vertical banding in the Fourier domain)."""
    return _gmic(img, ctx, "banding_denoise_v2", v_cutoff, h_cutoff, ctx.px(space), value,
                 _ev(BANDING_TILES, size_of_tiles))


@itb_op("CleanText", "Clean Text", GROUP, [
    Int("clean", 8, min=0, max=10, title="Clean"),
    Float("range", 1, min=0.2, max=1, step=0.01, title="Range"),
    Int("black", 80, min=0, max=100, title="Black"),
    Int("white", 95, min=0, max=100, title="White"),
], tags=["text", "scan", "document", "clean"], **_GMIC)
def clean_text(img, ctx, clean, range, black, white):  # noqa: A002 - upstream parameter name
    """G'MIC afre_cleantext (clean scanned text)."""
    return _gmic(img, ctx, "afre_cleantext", clean, range, black, white)


@itb_op("ConstrainedSharpen", "Constrained Sharpen", GROUP, [
    Float("sharpen_radius", 0.75, min=0, max=10, step=0.01, title="Sharpen Radius", unit="px"),
    Float("amount", 2.5, min=0, max=10, step=0.01, title="Amount"),
    Float("threshold", 1, min=0, max=50, title="Threshold"),
    Int("constraint_radius", 5, min=0, max=10, title="Constraint Radius", unit="px"),
    Float("overshoot", 0, min=0, max=50, title="Overshoot"),
    _channel("Luminance", CONSTRAINED_SHARPEN_CHANNELS),
    _choice("value_action", CONSTRAINED_VALUE_ACTION, "Cut", "Value Action"),
], tags=["sharpen", "unsharp", "halo"], **_GMIC)
def constrained_sharpen(img, ctx, sharpen_radius, amount, threshold, constraint_radius, overshoot, channel,
                        value_action):
    """G'MIC iain_constrained_sharpen (unsharp mask limited to the local min/max)."""
    return _gmic(img, ctx, "iain_constrained_sharpen", ctx.px(sharpen_radius), amount, threshold,
                 ctx.ipx(constraint_radius, 0), overshoot, _ev(CONSTRAINED_SHARPEN_CHANNELS, channel),
                 _ev(CONSTRAINED_VALUE_ACTION, value_action))


@itb_op("DcpDehaze", "DCP Dehaze", GROUP, [
    Int("scale", 5, min=1, max=20, title="Scale", unit="px", description="Dark channel window radius"),
    Float("strength", 1, min=0, max=2, step=0.01, title="Strength"),
    Float("min", 0.2, min=0, max=1, step=0.01, title="Min", description="Minimum transmission"),
    Float("max", 1, min=0, max=1, step=0.01, title="Max", description="Maximum transmission"),
    Float("brightness", 0, min=-100, max=100, title="Brightness"),
    Float("contrast", 0, min=-100, max=100, title="Contrast"),
    Float("gamma", 0, min=-100, max=100, title="Gamma"),
    Bool("transmittance_map", False, title="Transmittance Map", description="Output the transmission map instead"),
], tags=["dehaze", "haze", "fog", "dark channel"], **_GMIC)
def dcp_dehaze(img, ctx, **p):
    """G'MIC jeje_dehaze (dark channel prior)."""
    return _gmic(img, ctx, "jeje_dehaze", ctx.ipx(p["scale"]), p["strength"], p["min"], p["max"], p["brightness"],
                 p["contrast"], p["gamma"], p["transmittance_map"])


@itb_op("Deblur", "Deblur", GROUP, [
    Float("radius", 2, min=0, max=20, step=0.01, title="Radius", unit="px"),
    Int("iterations", 10, min=0, max=100, title="Iterations"),
    Float("time_step", 20, min=0, max=50, title="Time Step"),
    Float("smoothness", 0.1, min=0, max=10, step=0.01, title="Smoothness"),
    _choice("regularization", DEBLUR_REGULARIZATION, "MeanCurvature", "Regularization"),
    _channel(),
    _parallelism(),
    _overlap(24),
], tags=["deblur", "sharpen", "deconvolution"], **_GMIC)
def deblur(img, ctx, radius, iterations, time_step, smoothness, regularization, channel, parallelism,
           spatial_overlap):
    """G'MIC fx_deblur (regularized Jansson-Van Cittert deconvolution)."""
    return _gmic(img, ctx, "fx_deblur", ctx.px(radius), iterations, time_step, smoothness,
                 _ev(DEBLUR_REGULARIZATION, regularization), _ev(GMIC_CHANNELS, channel),
                 _ev(PARALLELISM, parallelism), spatial_overlap)


@itb_op("DenoiseSmooth", "Denoise Smooth", GROUP, [
    Int("radius", 3, min=1, max=10, title="Radius", unit="px"),
    Int("amount", 10, min=1, max=1000, title="Amount"),
], tags=["denoise", "smooth", "guided filter"], **_GMIC)
def denoise_smooth(img, ctx, radius, amount):
    """G'MIC afre_denoisesmooth (iterated guided filtering)."""
    return _gmic(img, ctx, "afre_denoisesmooth", ctx.ipx(radius), amount)


@itb_op("Descreen", "Descreen", GROUP, [
    Float("circle_radius_percent", 7, min=0, max=100, title="Circle Radius", unit="%"),
    Float("cross_length_percent", 40, min=0, max=100, title="Cross Length", unit="%"),
    Float("cross_width_percent", 1, min=0, max=100, title="Cross Width", unit="%"),
], tags=["descreen", "halftone", "moire", "scan"], **_GMIC)
def descreen(img, ctx, circle_radius_percent, cross_length_percent, cross_width_percent):
    """G'MIC fx_pahlsson_descreen (remove halftone screens in the Fourier domain)."""
    return _gmic(img, ctx, "fx_pahlsson_descreen", circle_radius_percent, cross_length_percent,
                 cross_width_percent)


def _band(prefix: str, title: str) -> list:
    return [
        Float(f"{prefix}_threshold", 0, min=0, max=10, step=0.01, title=f"{title} threshold"),
        Float(f"{prefix}_smoothness", 0, min=0, max=10, step=0.01, title=f"{title} smoothness"),
        _choice(f"{prefix}_smoothing", DETAIL_SMOOTHING, "Diffusion", f"{title} smoothing"),
        Float(f"{prefix}_gain", 0, min=-4, max=4, step=0.01, title=f"{title} gain"),
    ]


@itb_op("DetailsEqualizer", "Details Equalizer", GROUP, [
    Float("base_scale", 5, min=0, max=15, step=0.01, title="Base Scale"),
    Float("detail_scale", 0.5, min=0, max=5, step=0.01, title="Detail Scale"),
    *_band("coarse", "Coarse"), *_band("medium", "Medium"), *_band("small", "Small"), *_band("fine", "Fine"),
    _channel(),
    _choice("value_action", VALUE_ACTION, "None", "Value Action"),
    _parallelism(),
    _overlap(32),
], tags=["details", "equalizer", "multiscale", "local contrast"], **_GMIC)
def details_equalizer(img, ctx, base_scale, detail_scale, channel, value_action, parallelism, spatial_overlap, **b):
    """G'MIC fx_equalize_details (4-band detail equalizer)."""
    bands = []
    for n in ("coarse", "medium", "small", "fine"):
        bands += [b[f"{n}_threshold"], b[f"{n}_smoothness"], _ev(DETAIL_SMOOTHING, b[f"{n}_smoothing"]),
                  b[f"{n}_gain"]]
    head = ["fx_equalize_details", base_scale, detail_scale, *bands, _ev(GMIC_CHANNELS, channel),
            _ev(VALUE_ACTION, value_action)]
    try:
        return _gmic(img, ctx, *head, _ev(PARALLELISM, parallelism), spatial_overlap)
    except LazyRoomError as exc:
        # G'MIC's parallel tiling breaks on small images split into tiny tiles (small overlap, many
        # threads: "_ac: Invalid assignment request of shared instance").  Tiling only affects speed,
        # so retry untiled (parallelism OneThread); a genuine error simply fails again.
        if parallelism == "one_thread" or exc.code != ErrorCode.OPERATION_FAILED:
            raise
        return _gmic(img, ctx, *head, _ev(PARALLELISM, "one_thread"), spatial_overlap)


@itb_op("FreakyDetails", "Freaky Details", GROUP, [
    Int("amplitude", 2, min=1, max=5, title="Amplitude"),
    Float("scale", 10, min=0, max=100, title="Scale", unit="px", description="Bilateral spatial scale"),
    Int("iterations", 1, min=1, max=4, title="Iterations"),
    _channel("LinearRgb"),
], tags=["details", "hdr", "local contrast"], **_GMIC)
def freaky_details(img, ctx, amplitude, scale, iterations, channel):
    """G'MIC fx_freaky_details."""
    return _gmic(img, ctx, "fx_freaky_details", amplitude, ctx.px(scale), iterations, _ev(GMIC_CHANNELS, channel))


@itb_op("GoldMeinelSharpen", "Gold Meinel Sharpen", GROUP, [
    Float("sigma", 1, min=0.5, max=10, step=0.01, title="Sigma", unit="px"),
    Int("iterations", 5, min=1, max=15, title="Iterations"),
    Float("acceleration", 1, min=1, max=3, step=0.01, title="Acceleration"),
    _choice("blur", DECONVOLUTION_BLUR, "Gaussian", "Blur"),
    Bool("cut_values", True, title="Cut Values"),
    _channel(),
    _parallelism(),
    _overlap(24),
], tags=["sharpen", "deconvolution", "deblur"], **_GMIC)
def gold_meinel_sharpen(img, ctx, sigma, iterations, acceleration, blur, cut_values, channel, parallelism,
                        spatial_overlap):
    """G'MIC fx_unsharp_goldmeinel (Gold-Meinel deconvolution)."""
    return _gmic(img, ctx, "fx_unsharp_goldmeinel", ctx.px(sigma), iterations, acceleration,
                 _ev(DECONVOLUTION_BLUR, blur), cut_values, _ev(GMIC_CHANNELS, channel),
                 _ev(PARALLELISM, parallelism), spatial_overlap)


@itb_op("GradientSharpen", "Gradient Sharpen", GROUP, [
    Float("amount", 0.8, min=0, max=2, step=0.01, title="Amount"),
    Float("scale", 1.4, min=0.1, max=2, step=0.01, title="Scale"),
    _choice("value_action", SHARPEN_VALUE_ACTION, "NormalizeLuma", "Value Action"),
], tags=["sharpen", "gradient"], **_GMIC)
def gradient_sharpen(img, ctx, amount, scale, value_action):
    """G'MIC gcd_sharpen_gradient."""
    return _gmic(img, ctx, "gcd_sharpen_gradient", amount, scale, _ev(SHARPEN_VALUE_ACTION, value_action))


@itb_op("HessianSharpen", "Hessian Sharpen", GROUP, [
    Int("number_of_scales", 3, min=2, max=10, title="Number of Scales"),
    Float("strength", 1, min=0, max=10, step=0.01, title="Strength"),
    Float("repeat", 1, min=1, max=5, step=0.01, title="Repeat"),
    Bool("cut", False, title="Cut"),
], tags=["sharpen", "hessian"], **_GMIC)
def hessian_sharpen(img, ctx, number_of_scales, strength, repeat, cut):
    """G'MIC jeje_hessian_sharpen."""
    return _gmic(img, ctx, "jeje_hessian_sharpen", number_of_scales, strength, repeat, cut)


@itb_op("LocalVarianceNormalization", "Local Variance Normalization", GROUP, [
    Float("amplitude", 60, min=0, max=100, title="Amplitude"),
    Float("smoothness", 6, min=0, max=20, step=0.01, title="Smoothness", unit="px"),
    Float("threshold", 10, min=0, max=100, title="Threshold"),
    Int("repeat", 1, min=1, max=4, title="Repeat"),
    _channel(),
], tags=["local contrast", "normalize", "details"], **_GMIC)
def local_variance_normalization(img, ctx, amplitude, smoothness, threshold, repeat, channel):
    """G'MIC jeje_normalize_local_variance."""
    return _gmic(img, ctx, "jeje_normalize_local_variance", amplitude, ctx.px(smoothness), threshold, repeat,
                 _ev(GMIC_CHANNELS, channel))


@itb_op("MagicDetails", "Magic Details", GROUP, [
    Float("amplitude", 6, min=0, max=30, step=0.01, title="Amplitude"),
    Float("spatial_scale", 3, min=0, max=10, step=0.01, title="Spatial Scale", unit="px"),
    Float("value_scale", 15, min=0, max=20, step=0.01, title="Value Scale"),
    Float("edges", -0.5, min=-3, max=3, step=0.01, title="Edges"),
    Float("smoothness", 2, min=0, max=20, step=0.01, title="Smoothness", unit="px"),
    _channel("Luminance"),
], tags=["details", "local contrast"], **_GMIC)
def magic_details(img, ctx, amplitude, spatial_scale, value_scale, edges, smoothness, channel):
    """G'MIC fx_magic_details."""
    return _gmic(img, ctx, "fx_magic_details", amplitude, ctx.px(spatial_scale), value_scale, edges,
                 ctx.px(smoothness), _ev(GMIC_CHANNELS, channel))


@itb_op("MightyDetails", "Mighty Details", GROUP, [
    Float("amplitude", 25, min=0, max=100, title="Amplitude"),
    Float("amount", 1, min=0, max=2, step=0.01, title="Amount"),
    Float("scale", 25, min=1, max=100, title="Scale"),
    Int("smoothness", 1, min=0, max=10, title="Smoothness"),
    _channel("Luminance"),
], tags=["details", "local contrast"], **_GMIC)
def mighty_details(img, ctx, amplitude, amount, scale, smoothness, channel):
    """G'MIC fx_mighty_details."""
    return _gmic(img, ctx, "fx_mighty_details", amplitude, amount, scale, smoothness, _ev(GMIC_CHANNELS, channel))


@itb_op("MultiscaleSharpen", "Multiscale Sharpen", GROUP, [
    Float("strength", 15, min=0, max=100, title="Strength"),
    Float("regularity", 20, min=0, max=100, title="Regularity"),
    _channel(),
], tags=["sharpen", "multiscale"], **_GMIC)
def multiscale_sharpen(img, ctx, strength, regularity, channel):
    """G'MIC fx_sharpen_multiscale."""
    return _gmic(img, ctx, "fx_sharpen_multiscale", strength, regularity, _ev(GMIC_CHANNELS, channel))


@itb_op("RemoveHotPixels", "Remove Hot Pixels", GROUP, [
    Int("mask_size", 3, min=3, max=20, title="Mask Size", unit="px"),
    Float("threshold", 10, min=0, max=200, title="Threshold"),
], tags=["hot pixels", "repair", "denoise"], **_GMIC)
def remove_hot_pixels(img, ctx, mask_size, threshold):
    """G'MIC remove_hotpixels (median-based outlier replacement)."""
    return _gmic(img, ctx, "remove_hotpixels", ctx.ipx(mask_size, 3), threshold)


@itb_op("RemoveScratches", "Remove Scratches", GROUP, [
    Float("threshold", 72, min=0, max=100, title="Threshold"),
    Int("erosion", 2, min=0, max=5, title="Erosion", unit="px"),
    Int("dilation", 4, min=0, max=7, title="Dilation", unit="px"),
], tags=["scratches", "repair", "inpaint"], **_GMIC)
def remove_scratches(img, ctx, threshold, erosion, dilation):
    """G'MIC fx_remove_scratches (detect light scratches and inpaint them)."""
    return _gmic(img, ctx, "fx_remove_scratches", threshold, ctx.ipx(erosion, 0), ctx.ipx(dilation, 0))


@itb_op("RichardsonLucySharpen", "Richardson Lucy Sharpen", GROUP, [
    Float("sigma", 1, min=0.5, max=10, step=0.01, title="Sigma", unit="px"),
    Int("iterations", 10, min=1, max=100, title="Iterations"),
    _choice("blur", DECONVOLUTION_BLUR, "Gaussian", "Blur"),
    Bool("cut_values", True, title="Cut Values"),
], tags=["sharpen", "deconvolution", "deblur"], **_GMIC)
def richardson_lucy_sharpen(img, ctx, sigma, iterations, blur, cut_values):
    """G'MIC fx_unsharp_richardsonlucy (Richardson-Lucy deconvolution)."""
    return _gmic(img, ctx, "fx_unsharp_richardsonlucy", ctx.px(sigma), iterations, _ev(DECONVOLUTION_BLUR, blur),
                 cut_values)


@itb_op("SmoothSkin", "Smooth Skin", GROUP, [
    _choice("skin_estimation", SKIN_ESTIMATION, "Automatic", "Skin Estimation"),
    Float("tolerance", 0.5, min=0, max=1, step=0.01, title="Tolerance"),
    Float("mask_smoothness", 1, min=0, max=5, step=0.01, title="Mask Smoothness"),
    Float("threshold", 1, min=0, max=10, step=0.01, title="Threshold"),
    Bool("pre_normalize", True, title="Pre Normalize"),
    Float("manual_x", 50, min=0, max=100, title="Manual X", unit="%"),
    Float("manual_y", 50, min=0, max=100, title="Manual Y", unit="%"),
    Float("radius", 5, min=0, max=25, step=0.01, title="Radius", unit="%", description="Manual skin sample radius"),
    Float("base_scale", 2, min=0, max=10, step=0.01, title="Base Scale"),
    Float("fine_scale", 0.2, min=0, max=0.8, step=0.01, title="Fine Scale"),
    Float("detail_smoothness", 3, min=0, max=10, step=0.01, title="Detail Smoothness"),
    _choice("smoothness_type", SKIN_SMOOTHNESS_TYPE, "Bilateral", "Smoothness Type"),
    Float("gain", 0.05, min=0, max=0.5, step=0.01, title="Gain"),
], tags=["skin", "portrait", "retouch", "smooth"], **_GMIC)
def smooth_skin(img, ctx, skin_estimation, tolerance, mask_smoothness, threshold, pre_normalize, manual_x,
                manual_y, radius, base_scale, fine_scale, detail_smoothness, smoothness_type, gain):
    """G'MIC fx_smooth_skin."""
    return _gmic(img, ctx, "fx_smooth_skin", _ev(SKIN_ESTIMATION, skin_estimation), tolerance, mask_smoothness,
                 threshold, pre_normalize, manual_x, manual_y, radius, base_scale, fine_scale, detail_smoothness,
                 _ev(SKIN_SMOOTHNESS_TYPE, smoothness_type), gain)


@itb_op("TextureSharpen", "Texture Sharpen", GROUP, [
    Float("strength", 1, min=0, max=4, step=0.01, title="Strength"),
    Float("radius", 4, min=0, max=32, step=0.01, title="Radius", unit="px"),
    _channel(),
], tags=["sharpen", "texture", "rolling guidance"], **_GMIC)
def texture_sharpen(img, ctx, strength, radius, channel):
    """G'MIC fx_sharpen_texture (rolling-guidance detail boost)."""
    return _gmic(img, ctx, "fx_sharpen_texture", strength, ctx.px(radius), _ev(GMIC_CHANNELS, channel))


@itb_op("ToneSharpen", "Tone Sharpen", GROUP, [
    Float("amount", 1.2, min=0, max=4, step=0.01, title="Amount"),
    Int("center", 128, min=0, max=255, title="Center"),
    _choice("value_action", SHARPEN_VALUE_ACTION, "NormalizeLuma", "Value Action"),
], tags=["sharpen", "tones"], **_GMIC)
def tone_sharpen(img, ctx, amount, center, value_action):
    """G'MIC gcd_sharpen_tones."""
    return _gmic(img, ctx, "gcd_sharpen_tones", amount, center, _ev(SHARPEN_VALUE_ACTION, value_action))


@itb_op("Unstrip", "Unstrip", GROUP, [
    Float("smoothness", 1, min=0, max=10, step=0.01, title="Smoothness"),
    Float("size", 20, min=1, max=50, step=0.01, title="Size"),
    Float("sensitivity", 4, min=1, max=10, step=0.01, title="Sensitivity"),
    Bool("normalize", True, title="Normalize"),
    Bool("fft_preview", False, title="FFT Preview", description="Show the power spectrum instead"),
], tags=["stripes", "repair", "fourier"], **_GMIC)
def unstrip(img, ctx, smoothness, size, sensitivity, normalize, fft_preview):
    """G'MIC jeje_unstrip (remove periodic stripes in the Fourier domain)."""
    return _gmic(img, ctx, "jeje_unstrip", smoothness, size, sensitivity, normalize, fft_preview)


@itb_op("WhitenSharpen", "Whiten Sharpen", GROUP, [
    Float("alpha", 50, min=0, max=100, title="Alpha"),
    Bool("cut", False, title="Cut"),
], tags=["sharpen", "fourier", "whitening"], **_GMIC)
def whiten_sharpen(img, ctx, alpha, cut):
    """G'MIC jeje_whiten_frequency (spectral whitening)."""
    return _gmic(img, ctx, "jeje_whiten_frequency", alpha, cut)


# ---------------------------------------------------------------------------
# Native (aire / Trickle / jhlabs / GPUImage / OpenCV) filters
# ---------------------------------------------------------------------------

@itb_op("AnisotropicDiffusion", "Anisotropic Diffusion", GROUP, [
    Int("repeat_count", 20, min=1, max=100, title="Repeat Count"),
    Float("conduction", 0.6, min=0.1, max=1, step=0.01, title="Conduction"),
    Float("diffusion", 0.5, min=0.01, max=1, step=0.01, title="Diffusion"),
], upstream="aire", tags=["diffusion", "perona-malik", "smooth"])
def anisotropic_diffusion(img, ctx, repeat_count, conduction, diffusion):
    """aire anisotropicDiffusion: in-place 8-bit Perona-Malik style update over the four diagonal neighbours."""
    return impl.anisotropic_diffusion(img, repeat_count, conduction, diffusion)


@itb_op("AutoRemoveRedEyes", "Auto Remove Red Eyes", GROUP, [
    Float("threshold", 150, min=0, max=255, title="Threshold", description="Minimum red value (0..255) of a red-eye pixel"),
], upstream="OpenCV", tags=["red eye", "portrait", "face", "repair"])
def auto_remove_red_eyes(img, ctx, threshold):
    """ImageToolbox RedEyeRemover: YuNet face detection, eye boxes around the landmarks, red > threshold and
    red > blue + green -> replaced by (blue + green) / 2 grey."""
    return impl.remove_red_eyes(img, ctx.px(35.0), threshold)


@itb_op("AutoWhiteBalance", "Auto White Balance", GROUP, [
    Float("strength", 1, min=0, max=1, step=0.01, title="Strength"),
    Float("clipping", 0.05, min=0, max=2, step=0.01, title="Clipping", unit="%",
          description="Percentage of darkest/brightest pixels clipped per channel"),
], upstream="Trickle", alpha="full", tags=["white balance", "auto", "levels", "color"])
def auto_white_balance(img, ctx, strength, clipping):
    """Trickle autoWhiteBalance: per-channel histogram stretch with percentile clipping, blended by strength."""
    return impl.auto_white_balance(img, strength, clipping)


@itb_op("Dehaze", "Dehaze", GROUP, [
    Int("radius", 17, min=1, max=50, title="Radius", unit="px"),
    Float("omega", 0.45, min=0, max=1, step=0.01, title="Omega"),
], upstream="aire", tags=["dehaze", "haze", "fog", "dark channel"])
def dehaze(img, ctx, radius, omega):
    """aire dehaze: dark channel prior with atmospheric light from the brightest 1 % of the dark channel."""
    return impl.dehaze_dark_channel(img, ctx.ipx(radius), omega)


@itb_op("HighPass", "High Pass", GROUP, [
    Float("radius", 10, min=1, max=100, step=0.01, title="Radius", unit="px"),
], upstream="jhlabs", alpha="full", tags=["high pass", "details", "edges"])
def high_pass(img, ctx, radius):
    """jhlabs HighPassFilter: (color + 255 - gaussian(color)) / 2."""
    return impl.jh_high_pass(img, ctx.px(radius))


@itb_op("Kuwahara", "Kuwahara smoothing", GROUP, [
    Int("radius", 9, min=0, max=10, title="Radius",
        description="Samples per quadrant side; sample spacing is width/768 x height/1024 pixels"),
], upstream="GPUImage", tags=["kuwahara", "painting", "smooth", "edge preserving"])
def kuwahara(img, ctx, radius):
    """GPUImageKuwaharaFilter (four-quadrant mean with the lowest variance)."""
    return impl.gpu_kuwahara(img, radius)


@itb_op("Sharpen", "Sharpen", GROUP, [
    Float("sharpness", 1, min=-1, max=1, step=0.01, title="Sharpness"),
], upstream="GPUImage", tags=["sharpen", "laplacian"])
def sharpen(img, ctx, sharpness):
    """GPUImageSharpenFilter (3x3 Laplacian sharpening)."""
    return impl.gpu_sharpen(img, sharpness)


@itb_op("Unsharp", "Unsharp", GROUP, [
    Float("intensity", 0.5, min=0, max=1, step=0.01, title="Intensity"),
], upstream="aire", tags=["unsharp", "soften"])
def unsharp(img, ctx, intensity):
    """aire unsharp: color - (color - gaussian5x5(sigma 4)) * intensity (moves towards the blur)."""
    return impl.aire_unsharp(img, intensity, ctx.scale)
