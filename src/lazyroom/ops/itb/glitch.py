"""ImageToolbox "Glitch" group filters.

Backends: G'MIC command builders of ImageToolboxLibs (exact command strings),
Trickle native code (C++ glitches and the ntsc-rs based VHS NTSC emulation),
aire (MIT) and ImageToolbox's own JPEG-glitch / procedural GLSL shaders.
Implementation details live in ``_glitch_impl``.
"""
from __future__ import annotations

import math

from ...core.errors import invalid
from ...core.params import Bool, Choice, Color, Float, Int
from . import _glitch_impl as G
from ._common import gmic_cmd, gmic_pipeline, itb_op, snake

GROUP = "glitch"
_GMIC = dict(upstream="G'MIC", requires=["gmic"])

# ---------------------------------------------------------------------------
# G'MIC enums (ImageToolboxLibs GmicCommon.kt / filter builders, upstream order)
# ---------------------------------------------------------------------------

BOUNDARY = ["transparent", "nearest", "periodic", "mirror"]
_CHANNELS_UP = ["All", "Rgba", "Rgb", "Red", "Green", "Blue", "Alpha", "LinearRgb", "LinearRed", "LinearGreen",
                "LinearBlue", "YCbCr", "Luminance", "Chrominance", "BlueChrominance", "RedChrominance",
                "GreenChrominance", "Lab", "LabLightness", "LabChrominance", "LabA", "LabB", "Lch", "LchChrominance",
                "LchChroma", "LchHue", "Hsv", "HsvHue", "HsvSaturation", "HsvValue", "Hsi", "HsiIntensity", "Hsl",
                "HslLightness", "Cmyk", "Cmy", "Cyan", "Magenta", "Yellow", "Key", "Yiq", "YiqLuma",
                "YiqChrominance", "Ryb", "RybRed", "RybYellow", "RybBlue"]
CHANNELS = [snake(c) for c in _CHANNELS_UP]
ABERRATION_TYPES = ["shift", "radial", "angular", "random"]
HUFFMAN_SPLIT = ["none", "horizontal_blocks", "vertical_blocks", "patches"]
HUFFMAN_SPACES = [snake(c) for c in ["Rgb", "Cmyk", "Hcy", "Hsi", "Hsl", "Hsv", "Jzazbz", "Lab", "Lch", "OkLab",
                                      "YCbCr", "Yiq"]]
BIT_REVERSING = ["none", "reverse_bits", "reverse_bytes"]
SORT_ORDER = ["decreasing", "increasing"]
SORT_AXIS = ["x", "y", "x_then_y", "y_then_x"]
SORT_CRITERION = ["red", "green", "blue", "intensity", "luminance", "lightness", "hue", "saturation", "minimum",
                  "maximum", "random"]
SORT_MASK = {"criterion": 1, "contours": 2, "random": 3}
ROW_SHIFT_ORDER = ["x_then_y", "simultaneous", "y_then_x"]
ROW_SHIFT_INTERP = ["none", "nearest", "average", "bilinear", "grid", "bicubic", "lanczos"]
STREAK_PROPAGATION = ["backward", "forward", "bidirectional_sharp", "bidirectional_smooth"]


def _channel(name="channel", default="all"):
    return Choice(name, default, choices=CHANNELS, title="Channel", description="Channels/colour space to process")


def _boundary(default="mirror"):
    return Choice("boundary", default, choices=BOUNDARY, title="Boundary")


def _px(ctx, v: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, ctx.px(v)))


# ---------------------------------------------------------------------------
# G'MIC filters
# ---------------------------------------------------------------------------

@itb_op("AnalogDamage", "Analog Damage", GROUP, [
    Float("dirt", 18, min=0, max=100, title="Dirt", description="Amount of DCT-domain dirt noise"),
    Bool("monochrome_dirt", True, title="Monochrome Dirt"),
    Float("aberration_x", 3, min=-32, max=32, unit="px", title="Aberration X"),
    Float("aberration_y", 1, min=-32, max=32, unit="px", title="Aberration Y"),
    Float("softness", 0.6, min=0, max=10, unit="px", title="Softness"),
    Int("quality", 38, min=1, max=100, title="Quality", description="JPEG quality of the final damage pass"),
], **_GMIC, tags=["glitch", "analog", "dirt", "jpeg", "chromatic aberration", "damage", "vhs"])
def analog_damage(img, ctx, dirt, monochrome_dirt, aberration_x, aberration_y, softness, quality):
    """fx_dirty (power-of-two padded) + red/cyan fx_chromatic_aberrations + fx_jpeg_artefacts."""
    ax, ay = _px(ctx, aberration_x, -32, 32), _px(ctx, aberration_y, -32, 32)
    sm = _px(ctx, softness, 0, 10)
    cmd = gmic_pipeline(
        G.with_power_of_two_padding(gmic_cmd("fx_dirty", float(dirt), monochrome_dirt, 0, 1)),
        gmic_cmd("fx_chromatic_aberrations", 255, 0, 0, 0, ax, ay, sm, 20, 1,
                 0, 255, 255, 0, -ax, -ay, sm, 0, 1),
        gmic_cmd("fx_jpeg_artefacts", int(quality)),
    )
    return G.run_gmic_glitch(img, cmd, ctx, random=True, salt="analog_damage")


@itb_op("Breaks", "Breaks", GROUP, [
    Choice("type", "flat", choices=["flat", "relief"], title="Type"),
    Float("amplitude", 30, min=0, max=300, unit="px", title="Amplitude"),
    Float("frequency", 30, min=0, max=100, title="Frequency"),
    Float("smoothness", 0.5, min=0, max=10, unit="px", title="Smoothness"),
    _boundary(),
], **_GMIC, alpha="full", tags=["glitch", "breaks", "shatter", "displacement", "warp"])
def breaks(img, ctx, type, amplitude, frequency, smoothness, boundary):
    """G'MIC fx_breaks (alpha processed by G'MIC)."""
    cmd = gmic_cmd("fx_breaks", ["flat", "relief"].index(type), _px(ctx, amplitude, 0, 300), float(frequency),
                   _px(ctx, smoothness, 0, 10), BOUNDARY.index(boundary))
    return G.run_gmic_glitch(img, cmd, ctx, process_alpha=True, random=True, salt="breaks")


@itb_op("BufferError", "Buffer Error", GROUP, [
    Float("width", 50, min=0, max=200, title="Width", description="Width of the reinterpreted buffer (% of image)"),
    Float("height", 50, min=0, max=200, title="Height", description="Height of the reinterpreted buffer (% of image)"),
    Float("buffer_selection_start_percent", 0, min=0, max=100, title="Buffer Selection Start"),
    Float("buffer_selection_length_percent", 100, min=0, max=400, title="Buffer Selection Length"),
    Float("buffer_selection_shift_percent", 0, min=0, max=100, title="Buffer Selection Shift"),
    Bool("repeat_buffer", True, title="Repeat Buffer"),
], **_GMIC, alpha="full", geometry=True, tags=["glitch", "buffer", "data bending", "databending", "resize"])
def buffer_error(img, ctx, width, height, buffer_selection_start_percent, buffer_selection_length_percent,
                 buffer_selection_shift_percent, repeat_buffer):
    """G'MIC fx_buffer_error: the pixel buffer is re-read at a new width/height (changes the image size)."""
    cmd = gmic_cmd("fx_buffer_error", float(width), float(height), float(buffer_selection_start_percent),
                   float(buffer_selection_length_percent), float(buffer_selection_shift_percent), repeat_buffer)
    return G.run_gmic_glitch(img, cmd, ctx)


@itb_op("ChromaticAberrations", "Chromatic Aberrations", GROUP, [
    Color("primary_color", "#FF0000", title="Primary color"),
    Choice("primary_type", "shift", choices=ABERRATION_TYPES, title="Primary Type"),
    Float("primary_x_amplitude", 2, min=-32, max=32, unit="px", title="Primary X amplitude"),
    Float("primary_y_amplitude", 2, min=-32, max=32, unit="px", title="Primary Y amplitude"),
    Float("primary_smoothness", 0, min=0, max=10, unit="px", title="Primary Smoothness"),
    Float("primary_center_attenuation", 50, min=-100, max=100, title="Primary Center Attenuation"),
    Float("primary_attenuation_decay", 1, min=0, max=8, title="Primary Attenuation Decay"),
    Color("secondary_color", "#00FF00", title="Secondary color"),
    Choice("secondary_type", "shift", choices=ABERRATION_TYPES, title="Secondary Type"),
    Float("secondary_x_amplitude", 0, min=-32, max=32, unit="px", title="Secondary X amplitude"),
    Float("secondary_y_amplitude", 0, min=-32, max=32, unit="px", title="Secondary Y amplitude"),
    Float("secondary_smoothness", 0, min=0, max=10, unit="px", title="Secondary Smoothness"),
    Float("secondary_center_attenuation", 0, min=-100, max=100, title="Secondary Center Attenuation"),
    Float("secondary_attenuation_decay", 1, min=0, max=8, title="Secondary Attenuation Decay"),
], **_GMIC, tags=["glitch", "chromatic aberration", "color fringing", "rgb split", "lens"])
def chromatic_aberrations(img, ctx, primary_color, primary_type, primary_x_amplitude, primary_y_amplitude,
                          primary_smoothness, primary_center_attenuation, primary_attenuation_decay,
                          secondary_color, secondary_type, secondary_x_amplitude, secondary_y_amplitude,
                          secondary_smoothness, secondary_center_attenuation, secondary_attenuation_decay):
    """G'MIC fx_chromatic_aberrations (two colour axes, each shifted/warped independently)."""
    pr, pg, pb, _ = G.gmic_rgb(primary_color)
    sr, sg, sb, _ = G.gmic_rgb(secondary_color)
    cmd = gmic_cmd(
        "fx_chromatic_aberrations",
        pr, pg, pb, ABERRATION_TYPES.index(primary_type),
        _px(ctx, primary_x_amplitude, -32, 32), _px(ctx, primary_y_amplitude, -32, 32),
        _px(ctx, primary_smoothness, 0, 10), float(primary_center_attenuation), float(primary_attenuation_decay),
        sr, sg, sb, ABERRATION_TYPES.index(secondary_type),
        _px(ctx, secondary_x_amplitude, -32, 32), _px(ctx, secondary_y_amplitude, -32, 32),
        _px(ctx, secondary_smoothness, 0, 10), float(secondary_center_attenuation),
        float(secondary_attenuation_decay),
    )
    random = "random" in (primary_type, secondary_type)
    return G.run_gmic_glitch(img, cmd, ctx, random=random, salt="chromatic_aberrations")


@itb_op("HuffmanGlitch", "Huffman Glitch", GROUP, [
    Float("noise_level", 30, min=0, max=100, title="Noise Level", description="Amount of flipped bits in the Huffman stream"),
    Choice("split_mode", "none", choices=HUFFMAN_SPLIT, title="Split Mode"),
    Int("block_size", 25, min=0, max=100, title="Block Size", description="Split block size (% of the largest side)"),
    Float("patch_overlap", 0, min=0, max=50, title="Patch Overlap"),
    Choice("color_space", "rgb", choices=HUFFMAN_SPACES, title="Color Space"),
    Int("quantization", 0, min=0, max=64, title="Quantization", description="Palette size (0 = no quantization)"),
    Int("seed", 0, min=0, max=65536, title="Seed"),
], **_GMIC, tags=["glitch", "huffman", "compression", "data bending", "corruption"])
def huffman_glitch(img, ctx, noise_level, split_mode, block_size, patch_overlap, color_space, quantization, seed):
    """G'MIC fx_huffman_glitches: Huffman-compress, flip random bits (seeded), decompress."""
    cmd = gmic_cmd("fx_huffman_glitches", float(noise_level), HUFFMAN_SPLIT.index(split_mode), int(block_size),
                   float(patch_overlap), HUFFMAN_SPACES.index(color_space), int(quantization), int(seed))
    return G.run_gmic_glitch(img, cmd, ctx)


@itb_op("JpegArtifacts", "JPEG Artifacts", GROUP, [
    Int("quality", 50, min=1, max=100, title="Quality"),
], **_GMIC, tags=["glitch", "jpeg", "compression", "artifacts", "blocky"])
def jpeg_artifacts(img, ctx, quality):
    """G'MIC fx_jpeg_artefacts (8x8 DCT quantisation with the standard JPEG table)."""
    return G.run_gmic_glitch(img, gmic_cmd("fx_jpeg_artefacts", int(quality)), ctx)


@itb_op("MessWithBits", "Mess With Bits", GROUP, [
    Bool("pre_normalize", True, title="Pre Normalize"),
    Float("smoothness", 15, min=0, max=100, title="Smoothness"),
    Int("multiplier", 1, min=1, max=256, title="Multiplier"),
    Choice("reversing", "reverse_bits", choices=BIT_REVERSING, title="Reversing"),
    Int("mask_start", 0, min=0, max=15, title="Mask Start"),
    Int("mask_end", 15, min=0, max=15, title="Mask End"),
    Float("opacity", 100, min=0, max=100, title="Opacity"),
    _channel(),
], **_GMIC, tags=["glitch", "bits", "bit depth", "data bending", "posterize"])
def mess_with_bits(img, ctx, pre_normalize, smoothness, multiplier, reversing, mask_start, mask_end, opacity,
                   channel):
    """G'MIC fx_mess_with_bits: 16-bit integer bit reversal/masking."""
    cmd = gmic_cmd("fx_mess_with_bits", pre_normalize, float(smoothness), int(multiplier),
                   BIT_REVERSING.index(reversing), int(mask_start), int(mask_end), float(opacity),
                   CHANNELS.index(channel))
    return G.run_gmic_glitch(img, cmd, ctx)


@itb_op("OffsetStripes", "Offset Stripes", GROUP, [
    Int("iterations", 1, min=0, max=256, title="Iterations"),
    Int("seed", 0, min=0, max=65535, title="Seed"),
    Bool("channel_shift", False, title="Channel Shift", description="Shift each channel independently"),
    Float("horizontal_height", 10, min=0, max=100, title="Horizontal Height"),
    Float("horizontal_amplitude", 20, min=0, max=100, title="Horizontal Amplitude"),
    Float("vertical_width", 10, min=0, max=100, title="Vertical Width"),
    Float("vertical_amplitude", 20, min=0, max=100, title="Vertical Amplitude"),
    _boundary(),
    _channel(),
], **_GMIC, alpha="full", tags=["glitch", "stripes", "offset", "shift", "slices"])
def offset_stripes(img, ctx, iterations, seed, channel_shift, horizontal_height, horizontal_amplitude,
                   vertical_width, vertical_amplitude, boundary, channel):
    """G'MIC fx_offset_stripes (seeded; alpha processed by G'MIC)."""
    cmd = gmic_cmd("fx_offset_stripes", int(iterations), int(seed), channel_shift, float(horizontal_height),
                   float(horizontal_amplitude), float(vertical_width), float(vertical_amplitude),
                   BOUNDARY.index(boundary), CHANNELS.index(channel))
    return G.run_gmic_glitch(img, cmd, ctx, process_alpha=True)


@itb_op("PixelSort", "Pixel Sort", GROUP, [
    Choice("order", "increasing", choices=SORT_ORDER, title="Order"),
    Choice("axis", "x", choices=SORT_AXIS, title="Axis"),
    Choice("criterion", "red", choices=SORT_CRITERION, title="Criterion"),
    Choice("mask", "criterion", choices=list(SORT_MASK), title="Mask"),
    Float("lower_threshold", 0, min=0, max=100, title="Lower Threshold"),
    Float("upper_threshold", 100, min=0, max=100, title="Upper Threshold"),
    Float("softness", 0, min=0, max=5, title="Softness"),
    Bool("invert_mask", False, title="Invert Mask"),
], **_GMIC, tags=["glitch", "pixel sort", "sorting", "streaks"])
def pixel_sort(img, ctx, order, axis, criterion, mask, lower_threshold, upper_threshold, softness, invert_mask):
    """G'MIC fx_pixelsort (sort runs of pixels selected by a thresholded mask)."""
    if lower_threshold > upper_threshold:
        raise invalid("lower_threshold must not be greater than upper_threshold", param="lower_threshold",
                      hint="Lower the lower threshold or raise the upper threshold.")
    cmd = gmic_cmd("fx_pixelsort", SORT_ORDER.index(order), SORT_AXIS.index(axis), SORT_CRITERION.index(criterion),
                   SORT_MASK[mask], float(lower_threshold), float(upper_threshold), float(softness), invert_mask)
    random = criterion == "random" or mask == "random"
    return G.run_gmic_glitch(img, cmd, ctx, random=random, salt="pixel_sort")


@itb_op("RowShifter", "Row Shifter", GROUP, [
    Choice("order", "x_then_y", choices=ROW_SHIFT_ORDER, title="Order"),
    Float("x_shift", 0, min=-5, max=5, title="X Shift"),
    Float("x_centre", 0.5, min=0, max=1, title="X Centre"),
    Float("y_shift", 0, min=-5, max=5, title="Y Shift"),
    Float("y_centre", 0.5, min=0, max=1, title="Y Centre"),
    Choice("interpolation", "bilinear", choices=ROW_SHIFT_INTERP, title="Interpolation"),
    Bool("invert_shifts", False, title="Invert Shifts"),
], **_GMIC, alpha="full", geometry=True, tags=["glitch", "shear", "row shift", "skew"])
def row_shifter(img, ctx, order, x_shift, x_centre, y_shift, y_centre, interpolation, invert_shifts):
    """G'MIC fx_row_shift (alpha processed; the G'MIC command keeps its 2px periodic padding)."""
    cmd = gmic_cmd("fx_row_shift", ROW_SHIFT_ORDER.index(order), float(x_shift), float(x_centre), float(y_shift),
                   float(y_centre), ROW_SHIFT_INTERP.index(interpolation), invert_shifts)
    return G.run_gmic_glitch(img, cmd, ctx, process_alpha=True)


@itb_op("Streak", "Streak", GROUP, [
    Color("mask_color", "#FF0000", title="Mask Color", description="Pixels of exactly this colour are filled by streaking"),
    Float("step", 0, min=0, max=30, title="Step", description="Streak step (% of the smallest side)"),
    Float("angle", 0, min=0, max=360, title="Angle"),
    Choice("propagation", "bidirectional_smooth", choices=STREAK_PROPAGATION, title="Propagation"),
], **_GMIC, tags=["glitch", "streak", "smear", "inpaint", "mask"])
def streak(img, ctx, mask_color, step, angle, propagation):
    """G'MIC fx_streak: pixels matching the mask colour are replaced by streaked neighbours."""
    r, g, b, a = G.gmic_rgb(mask_color)
    cmd = gmic_cmd("fx_streak", r, g, b, a, float(step), float(angle), STREAK_PROPAGATION.index(propagation))
    return G.run_gmic_glitch(img, cmd, ctx)


# ---------------------------------------------------------------------------
# Trickle (native C++) glitches
# ---------------------------------------------------------------------------

@itb_op("BlockGlitch", "Block Glitch", GROUP, [
    Float("block_size", 0.02, min=0, max=1, step=0.001, title="Block Size",
          description="Block size as a fraction of the image width (at least 4 px)"),
    Float("strength", 0.5, min=0, max=1, step=0.001, title="Strength"),
], upstream="Trickle", alpha="full", tags=["glitch", "blocks", "displacement", "datamosh"])
def block_glitch(img, ctx, block_size, strength):
    """Trickle blockGlitch: random blocks copied from a randomly displaced source position."""
    return G.trickle_block_glitch(img, ctx.rng("block_glitch"), block_size, strength)


@itb_op("CrtCurvature", "CRT curvature", GROUP, [
    Float("curvature", 0.25, min=-1, max=1, step=0.001, title="Curvature"),
    Float("vignette", 0.65, min=0, max=1, step=0.001, title="Vignette"),
    Float("chroma", 0.015, min=0, max=1, step=0.001, title="Chroma",
          description="Red/blue channel offset as a fraction of the width"),
], upstream="Trickle", alpha="full", tags=["crt", "tv", "retro", "barrel", "vignette", "chromatic aberration"])
def crt_curvature(img, ctx, curvature, vignette, chroma):
    """Trickle crtCurvature: barrel warp (nearest sampling), R/B offset and radial vignette."""
    return G.trickle_crt_curvature(img, curvature, vignette, chroma)


@itb_op("GlitchVariant", "Glitch Variant", GROUP, [
    Int("repeat_count", 30, min=0, max=100, title="Repeat Count"),
    Float("max_offset", 0.25, min=0, max=1, step=0.001, title="Max Offset",
          description="Maximum line shift as a fraction of half the width"),
    Float("channel_shift", 0.3, min=0, max=1, step=0.001, title="Channel Shift",
          description="Red/blue offset as a fraction of a tenth of the width"),
], upstream="Trickle", alpha="full", tags=["glitch", "rgb split", "scanline", "shift"])
def glitch_variant(img, ctx, repeat_count, max_offset, channel_shift):
    """Trickle glitchVariant: random 1-5 px tall line bands shifted, plus a red/blue split."""
    return G.trickle_glitch_variant(img, ctx.rng("glitch_variant"), repeat_count, max_offset, channel_shift)


@itb_op("PixelMelt", "Pixel Melt", GROUP, [
    Int("max_drop", 20, min=0, max=250, unit="px", title="Max Drop"),
    Float("strength", 0.5, min=0, max=1, step=0.001, title="Strength"),
], upstream="Trickle", alpha="full", tags=["glitch", "melt", "drip", "pixel"])
def pixel_melt(img, ctx, max_drop, strength):
    """Trickle pixelMelt: per-column random downward drops that persist until re-rolled."""
    drop = max(0, int(round(max_drop * ctx.scale)))
    return G.trickle_pixel_melt(img, ctx.rng("pixel_melt"), drop, strength)


@itb_op("VHS", "VHS", GROUP, [
    Float("seed", 2.0, min=0, max=math.pi, step=0.001, title="Seed",
          description="Phase ('time') of the horizontal sine wobble"),
    Float("strength", 3.0, min=0, max=10, step=0.001, title="Strength"),
], upstream="Trickle", alpha="full", tags=["vhs", "retro", "tape", "noise", "scanlines", "rgb split"])
def vhs(img, ctx, seed, strength):
    """Trickle vhsGlitch: sine wobble, random line jitter, R/B split, noise and darkened scanlines."""
    return G.trickle_vhs(img, ctx.rng("vhs"), seed, strength)


# ---------------------------------------------------------------------------
# VHS NTSC (Trickle -> ntsc-rs)
# ---------------------------------------------------------------------------

_NTSC_DEFAULTS_SLIDERS = [
    # name, default, min, max, title, int?
    ("amount", 1.0, 0, 3, "Amount", False),
    ("tape_wear", 1.0, 0, 4, "Tape wear", False),
    ("chroma_bleed", 1.0, 0, 4, "Chroma bleed", False),
    ("tracking", 1.0, 0, 4, "Tracking", False),
    ("noise", 1.0, 0, 4, "Noise", False),
    ("snow", 1.0, 0, 4, "Snow", False),
    ("luma_smear", 0.5, 0, 8, "Luma smear", False),
    ("composite_sharpening", 1.0, 0, 8, "Sharpen", False),
    ("ringing", 4.0, 0, 16, "Ringing", False),
]


def _ntsc_params():
    adv = dict(advanced=True)
    p = [Float(n, d, min=lo, max=hi, title=t) for n, d, lo, hi, t, _ in _NTSC_DEFAULTS_SLIDERS]
    p += [
        Int("seed", 0, min=0, max=10000, title="Seed", **adv),
        Choice("use_field", "interleaved_upper", choices=G.USE_FIELD, title="Use field", **adv),
        Choice("filter_type", "butterworth", choices=G.FILTER_TYPE, title="Filter type", **adv),
        Choice("input_luma_filter", "notch", choices=G.LUMA_LOWPASS, title="Input luma filter", **adv),
        Choice("chroma_lowpass_in", "full", choices=G.CHROMA_LOWPASS, title="Input chroma lowpass", **adv),
        Choice("chroma_demodulation", "notch", choices=G.CHROMA_DEMOD, title="Chroma demodulation", **adv),
        Choice("video_scanline_phase_shift", "degrees_180", choices=G.PHASE_SHIFT, title="Phase shift", **adv),
        Choice("chroma_lowpass_out", "full", choices=G.CHROMA_LOWPASS, title="Output chroma lowpass", **adv),
        Int("video_scanline_phase_shift_offset", 0, min=-16, max=16, title="Phase offset", **adv),
        Bool("head_switching_enabled", True, title="Head switching", **adv),
        Int("head_switching_height", 8, min=0, max=160, title="Head switching height", **adv),
        Int("head_switching_offset", 3, min=-80, max=80, title="Head switching offset", **adv),
        Float("head_switching_horizontal_shift", 72, min=-800, max=800, title="Head switching shift", **adv),
        Float("head_switching_mid_line_position", 0.95, min=0, max=1, title="Mid-line position", **adv),
        Float("head_switching_mid_line_jitter", 0.03, min=0, max=2, title="Mid-line jitter", **adv),
        Bool("tracking_noise_enabled", True, title="Tracking noise", **adv),
        Int("tracking_noise_height", 12, min=0, max=160, title="Tracking noise height", **adv),
        Float("tracking_noise_wave_intensity", 15, min=0, max=160, title="Tracking wave", **adv),
        Float("tracking_noise_snow_intensity", 0.025, min=0, max=1, title="Tracking snow", **adv),
        Float("tracking_noise_snow_anisotropy", 0.25, min=0, max=1, title="Tracking snow anisotropy", **adv),
        Float("tracking_noise_noise_intensity", 0.25, min=0, max=2, title="Tracking noise intensity", **adv),
        Bool("composite_noise_enabled", True, title="Composite noise", **adv),
        Float("composite_noise_frequency", 0.5, min=0, max=8, title="Composite noise frequency", **adv),
        Float("composite_noise_intensity", 0.05, min=0, max=3, title="Composite noise intensity", **adv),
        Int("composite_noise_detail", 1, min=0, max=16, title="Composite noise detail", **adv),
        Bool("ringing_enabled", True, title="Ringing", **adv),
        Float("ringing_frequency", 0.45, min=0, max=8, title="Ringing frequency", **adv),
        Float("ringing_power", 4, min=0, max=20, title="Ringing power", **adv),
        Bool("luma_noise_enabled", True, title="Luma noise", **adv),
        Float("luma_noise_frequency", 0.5, min=0, max=8, title="Luma noise frequency", **adv),
        Float("luma_noise_intensity", 0.01, min=0, max=3, title="Luma noise intensity", **adv),
        Int("luma_noise_detail", 1, min=0, max=16, title="Luma noise detail", **adv),
        Bool("chroma_noise_enabled", True, title="Chroma noise", **adv),
        Float("chroma_noise_frequency", 0.05, min=0, max=8, title="Chroma noise frequency", **adv),
        Float("chroma_noise_intensity", 0.1, min=0, max=3, title="Chroma noise intensity", **adv),
        Int("chroma_noise_detail", 2, min=0, max=16, title="Chroma noise detail", **adv),
        Float("snow_intensity", 0.00025, min=0, max=0.1, title="Snow intensity", **adv),
        Float("snow_anisotropy", 0.5, min=0, max=1, title="Snow anisotropy", **adv),
        Float("chroma_phase_noise_intensity", 0.001, min=0, max=0.25, title="Chroma phase noise", **adv),
        Float("chroma_phase_error", 0, min=-1, max=1, title="Chroma phase error", **adv),
        Float("chroma_delay_horizontal", 0, min=-100, max=100, title="Chroma delay horizontal", **adv),
        Int("chroma_delay_vertical", 0, min=-40, max=40, title="Chroma delay vertical", **adv),
        Bool("vhs_enabled", True, title="VHS", **adv),
        Choice("vhs_tape_speed", "lp", choices=G.TAPE_SPEED, title="VHS tape speed", **adv),
        Float("vhs_chroma_loss", 0.000025, min=0, max=0.01, title="VHS chroma loss", **adv),
        Float("vhs_sharpen_intensity", 0.25, min=0, max=4, title="VHS sharpen intensity", **adv),
        Float("vhs_sharpen_frequency", 1, min=0, max=12, title="VHS sharpen frequency", **adv),
        Float("vhs_edge_wave_intensity", 0.5, min=0, max=6, title="VHS edge wave intensity", **adv),
        Float("vhs_edge_wave_speed", 4, min=0, max=24, title="VHS edge wave speed", **adv),
        Float("vhs_edge_wave_frequency", 0.05, min=0, max=1, title="VHS edge wave frequency", **adv),
        Int("vhs_edge_wave_detail", 2, min=0, max=16, title="VHS edge wave detail", **adv),
        Bool("scale_enabled", True, title="Scale", **adv),
        Float("scale_horizontal", 1, min=0.1, max=4, title="Scale horizontal", **adv),
        Float("scale_vertical", 1, min=0.1, max=4, title="Scale vertical", **adv),
        Float("scale_factor_x", 1, min=0.1, max=4, title="Scale factor X", **adv),
        Float("scale_factor_y", 1, min=0.1, max=4, title="Scale factor Y", **adv),
    ]
    return p


@itb_op("VHSNtsc", "VHS NTSC", GROUP, _ntsc_params(), upstream="Trickle",
        description="Analog NTSC composite video + VHS tape emulation (ntsc-rs port): chroma bleed, "
                    "ringing, head switching, tracking noise, snow and tape wear.",
        tags=["vhs", "ntsc", "analog", "composite", "tape", "retro", "crt", "noise"])
def vhs_ntsc(img, ctx, **p):
    """Trickle ``ntsc`` (Rust ntsc-rs ``NtscEffect::apply_effect_to_buffer``) with ImageToolbox's
    ``NtscParams.toNtscSettings()`` mapping.  Preview proxies scale the bandwidth like ntsc-rs's
    'scale with video size' so they match the export."""
    if p["amount"] <= 0:
        return img
    cfg = G.itb_ntsc_config(p)
    return G.ntsc_apply(img, cfg, max(p["scale_factor_x"], 0.001), max(p["scale_factor_y"], 0.001),
                        preview_scale=ctx.scale)


# ---------------------------------------------------------------------------
# aire
# ---------------------------------------------------------------------------

@itb_op("EnhancedGlitch", "Enhanced Glitch", GROUP, [
    Float("channel_shift_x", -0.08, min=-1, max=1, step=0.01, title="Channel Shift X"),
    Float("channel_shift_y", -0.08, min=-1, max=1, step=0.01, title="Channel Shift Y",
          description="No effect: ImageToolbox/aire use the X shift for both axes"),
    Float("corruption_size", 0.01, min=0, max=1, step=0.01, title="Corruption Size"),
    Int("amount", 60, min=1, max=100, title="Amount", description="Number of corrupted bands"),
    Float("corruption_shift_x", -0.05, min=-1, max=1, step=0.01, title="Corruption Shift X",
          description="No effect: ImageToolbox passes the channel shift X instead"),
    Float("corruption_shift_y", 0.0, min=-1, max=1, step=0.01, title="Corruption Shift Y"),
], upstream="aire", alpha="full", tags=["glitch", "rgb split", "corruption", "scanlines"])
def enhanced_glitch(img, ctx, channel_shift_x, channel_shift_y, corruption_size, amount, corruption_shift_x,
                    corruption_shift_y):
    """aire ``glitch``: G/B channel shift, shifted corruption bands, halftone scanlines; output opaque.

    Faithful to ImageToolbox's argument wiring: ``EnhancedGlitchFilter`` passes
    (channelsShiftX, corruptionShiftY, size, count, channelsShiftX,
    corruptionShiftY) and aire's ``ShiftPipelineImpl.glitch`` forwards
    channelsShiftX as the Y shift too.
    """
    return G.aire_glitch(img, ctx.rng("enhanced_glitch"), channel_shift_x, channel_shift_x, corruption_size,
                         amount, channel_shift_x, corruption_shift_y)


@itb_op("HorizontalWindStagger", "Horizontal Wind Stagger", GROUP, [
    Float("strength", 0.2, min=0, max=100, step=0.1, title="Strength",
          description="Maximum streak length as a multiple of the width"),
    Int("amount", 90, min=10, max=200, title="Amount", description="Number of horizontal streams"),
    Color("color", "#00000000", title="Color", description="Colour revealed behind the compressed rows"),
], upstream="aire", alpha="full", tags=["wind", "stagger", "streak", "smear", "horizontal"])
def horizontal_wind_stagger(img, ctx, strength, amount, color):
    """aire ``horizontalWindStagger``: rows squeezed left by gaussian-profiled random lengths per stream."""
    return G.aire_horizontal_wind_stagger(img, ctx.rng("horizontal_wind_stagger"), strength, amount, color)


@itb_op("ColorAnomaly", "Color Anomaly", GROUP, [
    Float("cutoff", 0.56, min=0.56, max=0.8, step=0.001, title="Cutoff",
          description="Aldridge tone-mapping cutoff (exposure fixed at 0.5)"),
], upstream="aire", tags=["glitch", "color", "tone mapping", "aldridge", "anomaly"])
def color_anomaly(img, ctx, cutoff):
    """ImageToolbox ColorAnomaly = aire Aldridge tone mapper (exposure 0.5, cutoff)."""
    return G.aire_aldridge(img, 0.5, cutoff)


# ---------------------------------------------------------------------------
# ImageToolbox native
# ---------------------------------------------------------------------------

@itb_op("Glitch", "Glitch", GROUP, [
    Int("amount", 20, min=0, max=100, title="Amount", description="Byte value written into the JPEG stream"),
    Int("seed", 15, min=0, max=100, title="Seed", description="Position of the corrupted byte within each slice"),
    Int("repeat_count", 9, min=0, max=100, title="Repeat Count", description="Number of corrupted bytes"),
], upstream="ImageToolbox", alpha="full", tags=["glitch", "jpeg", "corruption", "data bending", "datamosh"])
def glitch(img, ctx, amount, seed, repeat_count):
    """ImageToolbox JpegGlitch: encode as q100 JPEG, overwrite bytes in the entropy data, decode (opaque)."""
    return G.jpeg_glitch(img, amount, seed, repeat_count)


def _proc_params(spec):
    titles = {"intensity": "Intensity", "count": "Count", "seed": "Random seed", "frequency": "Frequency",
              "balance": "Balance", "variability": "Variability", "layers": "Layers", "distortion": "Distortion",
              "radius": "Radius", "dampening": "Dampening", "size": "Size"}
    out = []
    for name, default, lo, hi, is_int in spec:
        if is_int:
            out.append(Int(name, int(default), min=int(lo), max=int(hi), title=titles[name]))
        else:
            out.append(Float(name, float(default), min=lo, max=hi, step=0.001, title=titles[name]))
    out += [
        Float("offset_x", 0.0, min=-1, max=1, step=0.001, title="Offset X"),
        Float("offset_y", 0.0, min=-1, max=1, step=0.001, title="Offset Y"),
        Float("scale", 1.0, min=0.05, max=8, step=0.001, title="Scale"),
        Float("rotation", 0.0, min=-180, max=180, step=0.001, title="Rotation"),
    ]
    return out


# (ProceduralEffect parameters: name, default, min, max, integer)
_PROCEDURAL = [
    ("ColumnStreak", "Column Streak", [("intensity", 1.0, 0, 1, False), ("variability", 0.6, 0, 1, False),
                                       ("count", 16, 1, 128, True)],
     ["streak", "columns", "quantize"]),
    ("ConcentricCircleBreaks", "Concentric Circle Breaks",
     [("intensity", 0.5, -1, 1, False), ("dampening", 2.0, 0, 8, False), ("distortion", 0.2, -1, 1, False),
      ("frequency", 16.0, 1, 64, False)], ["breaks", "rings", "circles", "shatter"]),
    ("ConcentricSquareBreaks", "Concentric Square Breaks",
     [("intensity", 0.5, -1, 1, False), ("dampening", 2.0, 0, 8, False), ("distortion", 0.2, -1, 1, False),
      ("frequency", 16.0, 1, 64, False)], ["breaks", "squares", "shatter"]),
    ("InlineBreaks", "Inline Breaks", [("intensity", 0.3, -1, 1, False), ("distortion", 0.2, -1, 1, False),
                                       ("frequency", 32.0, 1, 128, False)], ["breaks", "lines", "shift"]),
    ("RadialStreak", "Radial Streak", [("intensity", 0.5, 0, 1, False), ("count", 8, 1, 64, True),
                                       ("radius", 0.3, 0.01, 1, False)], ["streak", "radial", "rays"]),
    ("RandomColorDispersion", "Random Color Dispersion", [("intensity", 0.15, -1, 1, False),
                                                          ("seed", 0, 0, 1000, True)],
     ["rgb split", "dispersion", "chromatic aberration", "blocks"]),
    ("SpiralBreaks", "Spiral Breaks", [("intensity", 0.5, -1, 1, False), ("distortion", 0.3, -1, 1, False),
                                       ("count", 16, 1, 64, True)], ["breaks", "spiral", "shatter"]),
    ("StreakCircles", "Streak Circles", [("intensity", 1.0, 0, 1, False), ("count", 20, 1, 64, True),
                                         ("radius", 0.15, 0.01, 0.5, False), ("layers", 8, 1, 32, True),
                                         ("seed", 0, 0, 1000, True)], ["streak", "circles", "rings"]),
    ("StreakExpand", "Streak Expand", [("intensity", 1.0, 0, 1, False), ("size", 0.3, 0.01, 0.9, False)],
     ["streak", "stretch", "expand"]),
    ("TiledStreak", "Tiled Streak", [("intensity", 1.0, 0, 1, False), ("balance", 0.5, 0, 1, False),
                                     ("variability", 0.5, 0, 1, False), ("frequency", 8.0, 1, 32, False),
                                     ("seed", 0, 0, 1000, True)], ["streak", "tiles", "grid"]),
]


def _make_procedural(name: str, title: str, spec, tags) -> None:
    def fn(img, ctx, **values):
        return G.procedural(name, img, values)

    fn.__name__ = snake(name)
    fn.__doc__ = f"ImageToolbox procedural GLSL shader '{name}' (ProceduralShaders.kt) evaluated on the CPU."
    itb_op(name, title, GROUP, _proc_params(spec), upstream="ImageToolbox", alpha="full",
           tags=["glitch", "procedural", "distortion", *tags])(fn)


for _args in _PROCEDURAL:
    _make_procedural(*_args)
