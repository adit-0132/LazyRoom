"""ImageToolbox "Effects" group filters."""
from __future__ import annotations

import cv2
import numpy as np

from ...core.params import Bool, Choice, Color, File, Float, FloatList, Int, Text
from .._util import F32, lut_from_hald_or_square
from . import _effects_impl as fx
from ._common import gmic_cmd, gmic_pipeline, itb_op, run_gmic

GROUP = "effects"
GMIC = dict(upstream="G'MIC", requires=["gmic"])

GMIC_CHANNELS = ["all", "rgba", "rgb", "red", "green", "blue", "alpha", "linear_rgb", "linear_red", "linear_green",
                 "linear_blue", "ycbcr", "luminance", "chrominance", "blue_chrominance", "red_chrominance",
                 "green_chrominance", "lab", "lab_lightness", "lab_chrominance", "lab_a", "lab_b", "lch",
                 "lch_chrominance", "lch_chroma", "lch_hue", "hsv", "hsv_hue", "hsv_saturation", "hsv_value", "hsi",
                 "hsi_intensity", "hsl", "hsl_lightness", "cmyk", "cmy", "cyan", "magenta", "yellow", "key", "yiq",
                 "yiq_luma", "yiq_chrominance", "ryb", "ryb_red", "ryb_yellow", "ryb_blue"]
FILL_HOLES_CHANNELS = ["all", "rgba_all", "rgb_all", "rgb_red", "rgb_green", "rgb_blue", "rgba_alpha",
                       "ycbcr_luminance", "ycbcr_blue_red_chrominances", "ycbcr_blue_chrominance",
                       "ycbcr_red_chrominance", "ycbcr_green_chrominance", "lab_lightness", "lab_ab_chrominances",
                       "lab_a_chrominance", "lab_b_chrominance", "lch_ch_chrominances", "lch_c_chrominance",
                       "lch_h_chrominance", "hsv_hue", "hsv_saturation", "hsv_value", "hsi_intensity",
                       "hsl_lightness", "cmyk_cyan", "cmyk_magenta", "cmyk_yellow", "cmyk_key"]
VALUE_ACTIONS = ["none", "cut", "normalize"]
CRT_TYPES = ["stripes", "triad", "slot_mask"]
CRT_PRECISIONS = ["low", "high"]
CRT_SHAPES = ["square", "triangular", "sine", "gaussian"]
FILAMENT_MODES = ["white_on_black", "black_on_white", "white_on_transparent", "black_on_transparent"]
ISOPHOTE_FILLS = ["transparent", "colors"]
METALS = ["silver", "gold", "copper", "bronze", "blue_steel"]
SKELETON_METHODS = ["distance", "thinning"]


def _num(v) -> str:
    """Number formatted like a gmic_cmd argument."""
    return gmic_cmd("_", v).split(" ", 1)[1]


def _idx(choices, value) -> int:
    return choices.index(value)


def _c8(c) -> list[int]:
    """Colour param -> Android ``Color.red/green/blue/alpha`` ints."""
    return [int(round(float(v) * 255.0)) for v in c[:4]]


def _resize_alpha(a: np.ndarray, shape) -> np.ndarray:
    if a.shape == tuple(shape[:2]):
        return a
    return cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR).astype(F32)


def _gmic(img, ctx, cmd: str, *, process_alpha: bool = False, single_thread: bool = False,
          timeout: float = 600.0) -> np.ndarray:
    """Run an ImageToolboxLibs G'MIC command with the Android bridge's alpha semantics.

    ``process_alpha=False`` is GmicAlphaMode.Preserve: only RGB goes to G'MIC,
    any alpha G'MIC produces is ignored and the source alpha is kept (resized
    when the geometry changed).  ``True`` is GmicAlphaMode.Process: RGBA is
    passed (opaque alpha added for RGB input) and G'MIC's alpha is returned.
    G'MIC's random generator is seeded from the render context for
    reproducible renders.
    """
    cmd = gmic_pipeline(gmic_cmd("srand", int(ctx.rng("gmic").integers(0, 2 ** 31 - 1))), cmd)

    def run(src, preserve):
        if single_thread:
            with fx.gmic_single_thread():
                return run_gmic(src, cmd, ctx=ctx, preserve_alpha=preserve, timeout=timeout)
        return run_gmic(src, cmd, ctx=ctx, preserve_alpha=preserve, timeout=timeout)

    if process_alpha:
        src = img if img.shape[2] == 4 else np.dstack([img, np.ones(img.shape[:2], F32)])
        out = run(src, False)
        return out if out.shape[2] == 4 else np.dstack([out[..., :3], np.ones(out.shape[:2], F32)])
    rgb = run(np.ascontiguousarray(img[..., :3]), True)[..., :3]
    if img.shape[2] == 4:
        return np.dstack([rgb, _resize_alpha(img[..., 3], rgb.shape)]).astype(F32)
    return rgb


# ---------------------------------------------------------------------------
# Morphology (aire)
# ---------------------------------------------------------------------------

def _morph_params():
    return [
        Int("size", 25, min=1, max=150, unit="px", title="Size", description="Structuring element size"),
        Bool("use_circle_kernel", True, title="Use Circle Kernel",
             description="Disk of radius size/3 instead of a size x size square"),
    ]


_MORPH_DOC = (" Structuring element: aire square kernel, or a disk of radius size/3 inside the size x size window"
              " (REFLECT_101 borders). Images with transparency are processed in RGBA mode like ImageToolbox.")


def _morph_op(name, title, op, summary, tags):
    @itb_op(name, title, GROUP, _morph_params(), upstream="aire", alpha="full", tags=["morphology", *tags],
            description=summary + _MORPH_DOC)
    def _fn(img, ctx, size, use_circle_kernel):
        return fx.morphology(img, op, ctx.ipx(size), use_circle_kernel)

    return _fn


black_hat = _morph_op("BlackHat", "Black Hat", "blackhat",
                      "Morphological black hat: closing minus the image (dark details).", ["black hat", "details"])
closing = _morph_op("Closing", "Closing", "closing",
                    "Morphological closing (dilation followed by erosion): fills small dark gaps.", ["closing"])
dilation = _morph_op("Dilation", "Dilation", "dilate",
                     "Morphological dilation (local maximum): grows bright areas.", ["dilate", "maximum"])
erode = _morph_op("Erode", "Erode", "erode",
                  "Morphological erosion (local minimum): grows dark areas.", ["erode", "minimum"])
morphological_gradient = _morph_op("MorphologicalGradient", "Morphological Gradient", "gradient",
                                   "Morphological gradient: dilation minus erosion (outlines).", ["gradient", "edges"])
opening = _morph_op("Opening", "Opening", "opening",
                    "Morphological opening (erosion followed by dilation): removes small bright spots.", ["opening"])
top_hat = _morph_op("TopHat", "Top Hat", "tophat",
                    "Morphological top hat: the image minus its opening (bright details).", ["top hat", "details"])


# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------

@itb_op("Canny", "Canny", GROUP, [
    Float("threshold_one", 100, min=0, max=1000, step=0.01, title="Threshold One",
          description="Lower hysteresis threshold (8-bit gradient units)"),
    Float("threshold_two", 200, min=0, max=1000, step=0.01, title="Threshold Two",
          description="Upper hysteresis threshold"),
], upstream="OpenCV", tags=["edges", "edge detection", "canny", "lines"])
def canny(img, ctx, threshold_one, threshold_two):
    """OpenCV Canny edge detector on the 3x3-median-filtered grayscale image (white edges on black)."""
    return fx.canny(img, threshold_one, threshold_two)


@itb_op("CopyMoveDetection", "Copy Move Detection", GROUP, [
    Float("retain", 4, min=1, max=40, step=0.01, title="Retain",
          description="Low-frequency DCT coefficients kept per side of each 16x16 block (rounded; values above "
                      "16 behave as 16)"),
    Float("coefficient", 0, min=0, max=1, step=0.01, title="Coefficient",
          description="DCT quantisation divisor (0 keeps only the coefficient signs)"),
], upstream="OpenCV", tags=["forensics", "copy move", "clone detection", "tamper"])
def copy_move_detection(img, ctx, retain, coefficient):
    """Copy-move forgery detection: identical quantised DCT signatures of 16x16 blocks that share a frequent
    shift vector (> 10 pairs, > 16 px apart) are painted in a colour derived from the shift length and blended
    80% over the image."""
    return fx.detect_copy_move(img, fx.round_half_up(retain), coefficient, ctx)


@itb_op("ErrorLevelAnalysis", "Error Level Analysis", GROUP, [
    Int("quality", 90, min=0, max=100, title="Quality", description="JPEG quality used for re-saving"),
], upstream="OpenCV", tags=["forensics", "ela", "jpeg", "tamper"])
def error_level_analysis(img, ctx, quality):
    """Error level analysis: absolute difference to a JPEG re-save at the given quality, min-max normalised."""
    return fx.error_level_analysis(img, quality)


# ---------------------------------------------------------------------------
# jhlabs
# ---------------------------------------------------------------------------

@itb_op("Chrome", "Chrome", GROUP, [
    Float("amount", 0.5, min=0, max=1, step=0.01, title="Amount", description="Bump strength"),
    Float("exposure", 1, min=0, max=5, step=0.01, title="Exposure"),
], upstream="jhlabs", tags=["chrome", "metal", "emboss", "monochrome"])
def chrome(img, ctx, amount, exposure):
    """Grey chrome look: brightness gradient bump shaded with an exponential exposure curve."""
    return fx.chrome(img, amount, exposure)


@itb_op("Contour", "Contour", GROUP, [
    Float("levels", 5, min=0, max=50, step=0.01, title="Levels", description="Number of brightness levels"),
    Int("scale", 1, min=0, max=1, title="Scale", description="Contour strength multiplier"),
    Int("offset", 0, min=0, max=255, title="Offset", description="Brightness offset of the level boundaries"),
    Color("color", "#000000", title="Color", description="Contour colour"),
], upstream="jhlabs", alpha="full", tags=["contour", "isolines", "lines"])
def contour(img, ctx, levels, scale, offset, color):
    """Draw contour lines where the quantised brightness level changes (jhlabs ContourFilter)."""
    return fx.contour(img, levels, scale, offset, color)


@itb_op("Dissolve", "Dissolve", GROUP, [
    Float("density", 0.75, min=0, max=1, step=0.01, title="Density", description="Fraction of pixels kept"),
    Float("softness", 0.1, min=0, max=1, step=0.01, title="Softness"),
], upstream="jhlabs", alpha="full", tags=["dissolve", "transparency", "noise", "speckle"])
def dissolve(img, ctx, density, softness):
    """Randomly makes pixels transparent (alpha thresholded against java.util.Random(0) noise)."""
    return fx.dissolve(img, density, softness)


@itb_op("DoG", "DoG", GROUP, [
    Float("radius", 1, min=0, max=100, step=0.01, unit="px", title="Radius"),
    Float("second_radius", 2, min=0, max=100, step=0.01, unit="px", title="Second radius"),
], upstream="jhlabs", alpha="full", tags=["edges", "difference of gaussians", "dog"])
def do_g(img, ctx, radius, second_radius):
    """Difference of (3-pass box) blurs: blur(second radius) - blur(radius), normalised (jhlabs DoGFilter)."""
    return fx.dog(img, ctx.px(radius), ctx.px(second_radius))


# ---------------------------------------------------------------------------
# aire / Trickle
# ---------------------------------------------------------------------------

@itb_op("Convolution3x3", "Convolution 3x3", GROUP, [
    FloatList("kernel", [0, 0, 0, 0, 1, 0, 0, 0, 0], length=9, title="Matrix",
              description="Row-major 3x3 kernel (weights are applied as Q15 fixed point, i.e. limited to [-1, 1])"),
], upstream="aire", alpha="full", tags=["convolution", "kernel", "custom filter"])
def convolution3x3(img, ctx, kernel):
    """Custom 3x3 convolution (aire convolve2D, RGBA, REFLECT_101)."""
    return fx.convolve3x3_q15(img, kernel, alpha=True)


@itb_op("Emboss", "Emboss", GROUP, [
    Float("intensity", 1, min=0, max=4, step=0.01, title="Intensity"),
], upstream="aire", tags=["emboss", "relief"])
def emboss(img, ctx, intensity):
    """Emboss kernel [-2i,-i,0; -i,1,i; 0,i,2i] (aire; Q15 fixed point saturates weights at +-1)."""
    return fx.convolve3x3_q15(img, fx.emboss_kernel(intensity), alpha=False)


@itb_op("Grain", "Grain", GROUP, [
    Float("intensity", 0.75, min=0, max=2, step=0.01, title="Intensity"),
], upstream="aire", tags=["grain", "film grain", "noise"])
def grain(img, ctx, intensity):
    """Monochrome gaussian grain (sigma = 127 * intensity, in 8-bit units)."""
    return fx.grain(img, intensity, ctx.rng("grain"))


@itb_op("Noise", "Noise", GROUP, [
    Int("threshold", 128, min=0, max=255, title="Noise", description="Noise range (rand() % threshold)"),
], upstream="Trickle", tags=["noise", "grain"])
def noise(img, ctx, threshold):
    """Bitwise noise: every 8-bit channel is OR-ed with a random value below the threshold (Trickle)."""
    return fx.trickle_noise(img, threshold, ctx.rng("noise"))


@itb_op("Threshold", "Luminance threshold", GROUP, [
    Int("level", 128, min=0, max=255, title="Threshold"),
], upstream="aire", tags=["threshold", "binarize", "black and white"])
def threshold(img, ctx, level):
    """Black/white threshold (aire thresholds the red channel: R > level -> white)."""
    return fx.threshold_red(img, level)


# ---------------------------------------------------------------------------
# GPUImage
# ---------------------------------------------------------------------------

@itb_op("SobelEdgeDetection", "Sobel edge", GROUP, [
    Int("line_width", 1, min=1, max=25, unit="px", title="Line width", description="Sobel sampling distance"),
], upstream="GPUImage", tags=["edges", "sobel", "edge detection"])
def sobel_edge_detection(img, ctx, line_width):
    """GPUImage Sobel edge detection: luminance, 3x3 Sobel at +-line width, gradient magnitude (grey)."""
    return fx.sobel_edges(img, ctx.px(line_width))


@itb_op("Lookup", "Lookup", GROUP, [
    Float("intensity", -1, min=-10, max=10, step=0.01, title="Intensity"),
    File("lut", "", file_kind="image", title="LUT image",
         description="Optional 512x512 GPUImage lookup image (or Hald CLUT). Without one the filter behaves like "
                     "ImageToolbox, whose shader samples an unbound (black) texture: color * (1 - intensity)."),
], upstream="GPUImage", tags=["lut", "lookup", "brightness"])
def lookup(img, ctx, intensity, lut):
    """GPUImageLookupFilter: mix(color, LUT(color), intensity)."""
    table = None
    if lut:
        from ...core.io import load_image

        lut_img, _ = load_image(ctx.resolve_file(lut, "LUT image"))
        table = lut_from_hald_or_square(lut_img)
    return fx.gpu_lookup(img, table, intensity)


@itb_op("Shader", "Shader", GROUP, [
    Text("shader", "", multiline=True, max_length=200_000, title="Shader",
         description="GPUImage GLSL ES fragment shader (varying vec2 textureCoordinate; uniform sampler2D "
                     "inputImageTexture; void main() writing gl_FragColor) or a complete ImageToolbox .itshader "
                     "JSON preset. Empty = no-op."),
    Text("values", "", multiline=True, max_length=20_000, title="Values",
         description='JSON object with uniform values, e.g. {"amount": 0.5, "tint": "#FF8800"}'),
], upstream="GPUImage", alpha="full", requires=["pyside6"], tags=["shader", "glsl", "opengl", "custom"])
def shader(img, ctx, shader, values):
    """Run a custom GPUImage fragment shader (ImageToolbox Shader filter) with headless OpenGL (PySide6).

    textureCoordinate (0,0) is the top-left pixel; bool uniforms are ints, colours vec4 (0..1).  Rendering
    happens in a separate process with Qt's offscreen platform; it needs PySide6 and an OpenGL-capable display.
    """
    if not shader.strip():
        return img.copy()
    glsl, params, vals = fx.parse_shader(shader, values)
    uniforms = {}
    for p in params:
        v = vals.get(p["name"], p["default"])
        if v is None:
            continue
        kind = fx.SHADER_UNIFORM_TYPES[p["type"]]
        if p["type"] == "color":
            v = [c / 255.0 for c in v]
        elif p["type"] == "bool":
            v = int(bool(v))
        uniforms[p["name"]] = [kind, v]
    src = img if img.shape[2] == 4 else np.dstack([img, np.ones(img.shape[:2], F32)])
    out = fx.run_shader(src, glsl, uniforms)
    if img.shape[2] == 3 and bool((out[..., 3] >= 1.0).all()):
        return out[..., :3]
    return out


# ---------------------------------------------------------------------------
# G'MIC
# ---------------------------------------------------------------------------

@itb_op("Cracks", "Cracks", GROUP, [
    Float("density", 30, min=0, max=100, step=0.01, title="Density"),
    Bool("relief", True, title="Relief"),
    Color("color", "#FFFFFF80", title="Color"),
    Choice("channel", "all", choices=GMIC_CHANNELS, title="Channel"),
], **GMIC, tags=["cracks", "texture", "old", "damage"])
def cracks(img, ctx, density, relief, color, channel):
    """Random cracks texture (G'MIC fx_cracks)."""
    r, g, b, a = _c8(color)
    return _gmic(img, ctx, gmic_cmd("fx_cracks", density, relief, r, g, b, a, _idx(GMIC_CHANNELS, channel)),
                 single_thread=True)


@itb_op("CrtPhosphors", "CRT Phosphors", GROUP, [
    Choice("type", "stripes", choices=CRT_TYPES, title="Type"),
    Int("upscale_factor", 4, min=1, max=8, title="Upscale Factor", description="Output is this many times larger"),
    Choice("precision", "low", choices=CRT_PRECISIONS, title="Precision"),
    Float("smoothness", 0, min=0, max=10, step=0.01, title="Smoothness"),
    Int("neighborhood_size", 4, min=1, max=16, title="Neighborhood Size"),
    Float("stride", 50, min=0, max=100, step=0.01, title="Stride"),
    Bool("adaptive_pattern", True, title="Adaptive Pattern"),
    Bool("use_luma", False, title="Use Luma"),
    Bool("transpose_pattern", False, title="Transpose Pattern"),
    Bool("average_over_pattern", False, title="Average Over Pattern"),
    Bool("normalize_image", True, title="Normalize Image"),
], **GMIC, alpha="full", geometry=True, tags=["crt", "phosphor", "retro", "screen", "upscale"])
def crt_phosphors(img, ctx, type, upscale_factor, precision, smoothness, neighborhood_size, stride,
                  adaptive_pattern, use_luma, transpose_pattern, average_over_pattern, normalize_image):
    """CRT phosphor mask simulation (G'MIC fx_crt_phosphors); enlarges the image by the upscale factor."""
    return _gmic(img, ctx, gmic_cmd("fx_crt_phosphors", _idx(CRT_TYPES, type), upscale_factor - 1,
                                    _idx(CRT_PRECISIONS, precision), smoothness, neighborhood_size, stride,
                                    adaptive_pattern, use_luma, transpose_pattern, average_over_pattern,
                                    normalize_image))


@itb_op("CrtScanlines", "CRT Scanlines", GROUP, [
    Int("upscale_factor", 4, min=1, max=16, title="Upscale Factor", description="Output is this many times larger"),
    Int("neighborhood_size", 16, min=8, max=64, title="Neighborhood Size"),
    Choice("bloom_shape", "triangular", choices=CRT_SHAPES, title="Bloom Shape"),
    Float("threshold", 0, min=-1, max=1, step=0.01, title="Threshold"),
    Bool("use_luma_for_bloom", True, title="Use Luma For Bloom"),
    Choice("fuzz_shape", "triangular", choices=CRT_SHAPES, title="Fuzz Shape"),
    Bool("normalize_image", False, title="Normalize Image"),
], **GMIC, alpha="full", geometry=True, tags=["crt", "scanlines", "retro", "screen", "upscale"])
def crt_scanlines(img, ctx, upscale_factor, neighborhood_size, bloom_shape, threshold, use_luma_for_bloom,
                  fuzz_shape, normalize_image):
    """CRT scanlines with bloom (G'MIC fx_crt_scanlines); enlarges the image by the upscale factor."""
    return _gmic(img, ctx, gmic_cmd("fx_crt_scanlines", upscale_factor, neighborhood_size,
                                    _idx(CRT_SHAPES, bloom_shape), threshold, use_luma_for_bloom,
                                    _idx(CRT_SHAPES, fuzz_shape), normalize_image))


@itb_op("Dirty", "Dirty", GROUP, [
    Float("amplitude", 30, min=0, max=100, step=0.01, title="Amplitude"),
    Bool("monochrome", True, title="Monochrome"),
    Choice("channel", "all", choices=GMIC_CHANNELS, title="Channel"),
    Choice("value_action", "none", choices=VALUE_ACTIONS, title="Value Action"),
], **GMIC, tags=["dirt", "grunge", "noise", "old"])
def dirty(img, ctx, amplitude, monochrome, channel, value_action):
    """Dirt / grunge overlay (G'MIC fx_dirty, padded to a power-of-two size like ImageToolbox)."""
    return _gmic(img, ctx, fx.with_power_of_two_padding(
        gmic_cmd("fx_dirty", amplitude, monochrome, _idx(GMIC_CHANNELS, channel), _idx(VALUE_ACTIONS, value_action))))


@itb_op("Filaments", "Filaments", GROUP, [
    Float("density", 50, min=0, max=100, step=0.01, title="Density"),
    Float("length", 50, min=0, max=100, step=0.01, title="Length"),
    Float("contour", 75, min=0, max=100, step=0.01, title="Contour"),
    Float("distortion", 30, min=0, max=100, step=0.01, title="Distortion"),
    Float("smoothness", 10, min=0, max=100, step=0.01, title="Smoothness"),
    Float("rotation", 0, min=-180, max=180, step=0.01, title="Rotation"),
    Bool("local_normalization", True, title="Local Normalization"),
    Bool("throw_from_left", True, title="Throw From Left"),
    Bool("throw_from_right", True, title="Throw From Right"),
    Bool("throw_from_above", True, title="Throw From Above"),
    Bool("throw_from_below", True, title="Throw From Below"),
    Float("opacity", 50, min=0, max=100, step=0.01, title="Opacity"),
    Choice("color_mode", "white_on_black", choices=FILAMENT_MODES, title="Color Mode"),
], **GMIC, alpha="full", tags=["filaments", "lines", "fibers", "abstract"])
def filaments(img, ctx, density, length, contour, distortion, smoothness, rotation, local_normalization,
              throw_from_left, throw_from_right, throw_from_above, throw_from_below, opacity, color_mode):
    """Filament strokes following the image structure (G'MIC fx_filaments, alpha processed)."""
    return _gmic(img, ctx, gmic_cmd("fx_filaments", density, length, contour, distortion, smoothness, rotation,
                                    local_normalization, throw_from_left, throw_from_right, throw_from_above,
                                    throw_from_below, opacity, _idx(FILAMENT_MODES, color_mode)),
                 process_alpha=True, single_thread=True)


@itb_op("FillHoles", "Fill Holes", GROUP, [
    Int("morph_radius", 11, min=3, max=50, title="Morph Radius"),
    Int("edge_radius", 21, min=0, max=50, title="Edge Radius"),
    Int("close_radius", 5, min=0, max=10, title="Close Radius"),
    Choice("channel", "all", choices=FILL_HOLES_CHANNELS, title="Channel"),
    Bool("fill_light_colours", False, title="Fill Light Colours"),
    Bool("fast", True, title="Fast"),
], **GMIC, tags=["fill holes", "repair", "morphology"])
def fill_holes(img, ctx, morph_radius, edge_radius, close_radius, channel, fill_light_colours, fast):
    """Fill dark (or light) holes (G'MIC fill_holes)."""
    return _gmic(img, ctx, gmic_cmd("fill_holes", morph_radius, edge_radius, close_radius,
                                    _idx(FILL_HOLES_CHANNELS, channel), fill_light_colours, fast))


@itb_op("Fractalize", "Fractalize", GROUP, [
    Float("detail_level", 0.8, min=0, max=1, step=0.01, title="Detail Level"),
], **GMIC, tags=["fractal", "abstract"])
def fractalize(img, ctx, detail_level):
    """Fractal self-similar recomposition (G'MIC fractalize)."""
    return _gmic(img, ctx, gmic_cmd("fractalize", detail_level))


@itb_op("FreakyBlackAndWhite", "Freaky B&W", GROUP, [
    Float("strength", 90, min=0, max=100, step=0.01, title="Strength"),
    Float("oddness", 20, min=0, max=100, step=0.01, title="Oddness"),
    Float("brightness", 0, min=-100, max=100, step=0.01, title="Brightness"),
    Float("contrast", 0, min=-100, max=100, step=0.01, title="Contrast"),
    Float("gamma", 0, min=-100, max=100, step=0.01, title="Gamma"),
], **GMIC, tags=["black and white", "monochrome", "creative"])
def freaky_black_and_white(img, ctx, strength, oddness, brightness, contrast, gamma):
    """Freaky black & white conversion (G'MIC fx_freaky_bw)."""
    return _gmic(img, ctx, gmic_cmd("fx_freaky_bw", strength, oddness, brightness, contrast, gamma))


@itb_op("Ghost", "Ghost", GROUP, [
    Float("amplitude", 200, min=0, max=1000, step=0.01, title="Amplitude"),
    Float("smoothness", 2, min=0, max=10, step=0.01, title="Smoothness"),
    Float("coherence", 2, min=0, max=10, step=0.01, title="Coherence"),
    Float("gamma", 1, min=-3, max=3, step=0.01, title="Gamma"),
    Float("normalization_amplitude", 3, min=0, max=10, step=0.01, title="Normalization Amplitude"),
    Float("radius", 16, min=1, max=64, step=0.01, title="Radius"),
    Bool("invert", False, title="Invert"),
], **GMIC, tags=["ghost", "glow", "streaks"])
def ghost(img, ctx, amplitude, smoothness, coherence, gamma, normalization_amplitude, radius, invert):
    """Ghostly streaks along the image structure (G'MIC fx_ghost)."""
    return _gmic(img, ctx, gmic_cmd("fx_ghost", amplitude, smoothness, coherence, gamma, normalization_amplitude,
                                    radius, invert), single_thread=True)


@itb_op("Grid", "Grid", GROUP, [
    Float("horizontal_spacing", 10, min=0, max=100, step=0.01, title="Horizontal spacing",
          description="Percent of the image width"),
    Float("vertical_spacing", 10, min=0, max=100, step=0.01, title="Vertical spacing",
          description="Percent of the image height"),
    Float("offset_x", 0, min=-100, max=100, step=0.01, title="Offset X", description="Percent"),
    Float("offset_y", 0, min=-100, max=100, step=0.01, title="Offset Y", description="Percent"),
    Float("opacity", 0.5, min=0, max=1, step=0.01, title="Opacity"),
    Color("grid_color", "#000000", title="Grid Color"),
], **GMIC, tags=["grid", "lines", "overlay"])
def grid(img, ctx, horizontal_spacing, vertical_spacing, offset_x, offset_y, opacity, grid_color):
    """Draw a grid (G'MIC grid; spacings and offsets in percent)."""
    r, g, b, a = _c8(grid_color)
    pct = [_num(v) + "%" for v in (horizontal_spacing, vertical_spacing, offset_x, offset_y)]
    cmd = "grid " + ",".join(pct + [_num(opacity), str(r), str(g), str(b), str(a)])
    return _gmic(img, ctx, cmd)


@itb_op("Isophotes", "Isophotes", GROUP, [
    Int("levels", 8, min=1, max=256, title="Levels"),
    Float("smoothness", 4, min=0, max=10, step=0.01, title="Smoothness"),
    Choice("fill", "colors", choices=ISOPHOTE_FILLS, title="Fill"),
], **GMIC, tags=["isophotes", "contours", "levels"])
def isophotes(img, ctx, levels, smoothness, fill):
    """Iso-luminance contour lines (G'MIC fx_isophotes)."""
    return _gmic(img, ctx, gmic_cmd("fx_isophotes", levels, smoothness, _idx(ISOPHOTE_FILLS, fill)))


@itb_op("LocalOrientation", "Local Orientation", GROUP, [
    Float("smoothness", 0, min=0, max=5, step=0.01, title="Smoothness"),
    Float("minimum_threshold", 0, min=0, max=100, step=0.01, title="Minimum Threshold"),
    Float("maximum_threshold", 100, min=0, max=100, step=0.01, title="Maximum Threshold"),
    Bool("negative_colors", False, title="Negative Colors"),
    Choice("channel", "all", choices=GMIC_CHANNELS, title="Channel"),
], **GMIC, tags=["orientation", "gradient", "analysis"])
def local_orientation(img, ctx, smoothness, minimum_threshold, maximum_threshold, negative_colors, channel):
    """Colour-coded local gradient orientation (G'MIC fx_local_orientation)."""
    return _gmic(img, ctx, gmic_cmd("fx_local_orientation", smoothness, minimum_threshold, maximum_threshold,
                                    negative_colors, _idx(GMIC_CHANNELS, channel)))


@itb_op("MetallicLook", "Metallic Look", GROUP, [
    Float("strength", 1, min=0, max=1, step=0.01, title="Strength"),
    Float("smoothness", 0, min=0, max=20, step=0.01, title="Smoothness"),
    Choice("metal", "silver", choices=METALS, title="Metal"),
], **GMIC, tags=["metal", "metallic", "chrome", "gold"])
def metallic_look(img, ctx, strength, smoothness, metal):
    """Metallic rendering (G'MIC fx_tk_metallic)."""
    return _gmic(img, ctx, gmic_cmd("fx_tk_metallic", strength, smoothness, _idx(METALS, metal)))


@itb_op("OldMovieStripes", "Old Movie Stripes", GROUP, [
    Float("frequency", 10, min=0, max=100, step=0.01, title="Frequency"),
    Choice("channel", "all", choices=GMIC_CHANNELS, title="Channel"),
    Choice("value_action", "none", choices=VALUE_ACTIONS, title="Value Action"),
], **GMIC, tags=["old movie", "stripes", "film", "vintage"])
def old_movie_stripes(img, ctx, frequency, channel, value_action):
    """Vertical film-scratch stripes (G'MIC fx_stripes_y)."""
    return _gmic(img, ctx, gmic_cmd("fx_stripes_y", frequency, _idx(GMIC_CHANNELS, channel),
                                    _idx(VALUE_ACTIONS, value_action)))


@itb_op("RainAndSnow", "Rain And Snow", GROUP, [
    Float("angle", 65, min=-180, max=180, step=0.01, title="Angle"),
    Float("speed", 10, min=0, max=50, step=0.01, title="Speed"),
    Float("density", 50, min=0, max=100, step=0.01, title="Density"),
    Float("radius", 0.1, min=0, max=3, step=0.01, title="Radius"),
    Float("gamma", 1, min=0, max=2, step=0.01, title="Gamma"),
    Float("opacity", 1, min=0, max=1, step=0.01, title="Opacity"),
], **GMIC, tags=["rain", "snow", "weather"])
def rain_and_snow(img, ctx, angle, speed, density, radius, gamma, opacity):
    """Rain / snow streaks screen-blended over the image (G'MIC fx_rain + blend screen)."""
    cmd = gmic_cmd("fx_rain", angle, speed, density, radius, gamma, opacity) + " rv blend screen," + _num(opacity)
    return _gmic(img, ctx, cmd)


@itb_op("RetroFade", "Retro Fade", GROUP, [
    Int("iterations", 20, min=1, max=64, title="Iterations"),
    Int("colors", 6, min=2, max=32, title="Colors"),
    Float("grain", 40, min=1, max=100, step=0.01, title="Grain"),
], **GMIC, tags=["retro", "fade", "vintage"])
def retro_fade(img, ctx, iterations, colors, grain):
    """Faded retro colours (G'MIC fx_retrofade)."""
    return _gmic(img, ctx, gmic_cmd("fx_retrofade", iterations, colors, grain))


@itb_op("SixtiesCinema", "60's Cinema", GROUP, [
    Float("grain", 10, min=0, max=25, step=0.01, title="Grain"),
    Bool("noon_for_midnight", False, title="Noon for Midnight"),
    Bool("gene_kelly", False, title="GeneKelly"),
    Bool("ray_harryhausen", False, title="RayHarryhausen"),
    Bool("cary_grant", False, title="CaryGrant"),
], **GMIC, tags=["cinema", "60s", "film", "vintage"])
def sixties_cinema(img, ctx, grain, noon_for_midnight, gene_kelly, ray_harryhausen, cary_grant):
    """1960s cinema colour looks (G'MIC fx_gb_cfx)."""
    return _gmic(img, ctx, gmic_cmd("fx_gb_cfx", grain, noon_for_midnight, gene_kelly, ray_harryhausen, cary_grant))


@itb_op("Skeleton", "Skeleton", GROUP, [
    Choice("method", "distance", choices=SKELETON_METHODS, title="Method"),
    Float("smoothness", 0, min=0, max=10, step=0.01, title="Smoothness"),
], **GMIC, tags=["skeleton", "thinning", "morphology"])
def skeleton(img, ctx, method, smoothness):
    """Morphological skeleton of the image shapes (G'MIC fx_skeleton)."""
    return _gmic(img, ctx, gmic_cmd("fx_skeleton", _idx(SKELETON_METHODS, method), smoothness))


@itb_op("RawGmic", "GMIC", GROUP, [
    Text("command", "blur 3", max_length=4000, title="Command",
         description="G'MIC pipeline applied to the image, e.g. 'blur 3' or 'fx_sharpen 50,0,0'"),
], **GMIC, alpha="full", geometry=True, tags=["gmic", "custom", "script"],
    description=("Run a raw G'MIC command pipeline on the image (G'MIC GUI 'fx_*' filters and CLI commands). "
                 "Safety policy: the pipeline may only process the current image. Commands that execute programs "
                 "(exec/x/shell), read or write files (input/i, output/o and variants, command/m, delete, ...), "
                 "open windows, or access the network are rejected, as are file paths, URLs, '$' variable/"
                 "environment substitutions, string substitutions and side-effect math functions (run(), ext(), "
                 "isfile(), ...); network loading is additionally disabled inside G'MIC ('network -1'). "
                 "The check is conservative and may reject some harmless pipelines."))
def raw_gmic(img, ctx, command):
    """Raw G'MIC pipeline (see description for the safety policy)."""
    cmd = fx.check_raw_gmic(command)
    return _gmic(img, ctx, gmic_pipeline("network -1", cmd), timeout=300.0)
