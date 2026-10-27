"""ImageToolbox "Simple effects" group filters."""
from __future__ import annotations

import numpy as np

from ...core.params import Choice
from . import _simple_impl as S
from ._common import color_matrix_3x3, itb_op, snake

GROUP = "simple"

# aire ColorMatrices (MIT, awxkee) — row-major 3x3, out = M @ rgb
AUTUMN_TONES = [0.9, 0.3, 0.1, 0.2, 0.8, 0.2, 0.1, 0.1, 0.8]


@itb_op("AutumnTones", "Autumn Tones", GROUP, [], upstream="aire", tags=["autumn", "warm", "color matrix"])
def autumn_tones(img, ctx):
    """Fixed 3x3 colour matrix (aire ColorMatrices.AUTUMN_TONES)."""
    return color_matrix_3x3(img, AUTUMN_TONES)


# ---------------------------------------------------------------------------
# Fixed colour matrices: ImageToolbox ColorMatrix3x3Filter -> Aire.colorMatrix
# (out = clamp(M @ rgb), alpha untouched).  Values from aire ColorMatrices.kt.
# ---------------------------------------------------------------------------

_MATRICES: list[tuple[str, str, list[float], list[str]]] = [
    ("Sepia", "Sepia", [0.393, 0.769, 0.189, 0.349, 0.686, 0.168, 0.272, 0.534, 0.131], ["sepia", "vintage"]),
    ("Polaroid", "Polaroid", [1.438, -0.062, 0.062, 0.122, 1.378, -0.122, -0.016, -0.016, 1.483],
     ["polaroid", "vintage"]),
    ("Cool", "Cool", [0.99, 0.0, 0.0, 0.0, 0.93, 0.0, 0.0, 0.0, 1.08], ["cool", "cold", "blue"]),
    ("Warm", "Warm", [1.06, 0.0, 0.0, 0.0, 1.01, 0.0, 0.0, 0.0, 0.93], ["warm", "orange"]),
    ("NightVision", "Night Vision", [0.1, 0.4, 0.0, 0.3, 1.0, 0.3, 0.0, 0.4, 0.1], ["night vision", "green"]),
    ("CodaChrome", "Coda Chrome", [1.1285583, -0.39673823, -0.03992559, -0.1640434, 1.0835252, -0.054988053,
                                   -0.1678601, -0.56034166, 1.6014851], ["film", "kodachrome"]),
    ("Browni", "Browni", [0.59970236, 0.34553242, -0.27082986, -0.03770325, 0.86095774, 0.15059553,
                          0.24113636, -0.07441038, 0.4497218], ["brown", "vintage"]),
    ("Vintage", "Vintage", [0.6279346, 0.32021835, -0.039654084, 0.025783977, 0.64411885, 0.032591276,
                            0.046605557, -0.0851233, 0.5241648], ["vintage", "retro"]),
    ("Pastel", "Pastel", [0.9, 0.3, 0.1, 0.2, 0.7, 0.2, 0.1, 0.2, 0.8], ["pastel", "soft"]),
    ("OrangeHaze", "Orange Haze", [0.6, 0.3, 0.1, 0.1, 0.4, 0.5, 0.1, 0.1, 0.7], ["orange", "haze"]),
    ("PinkDream", "Pink Dream", [1.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.2, 0.2, 1.1], ["pink", "dreamy"]),
    ("GoldenHour", "Golden Hour", [1.2, 0.5, 0.1, 0.1, 0.8, 0.2, 0.1, 0.1, 1.0], ["golden", "warm", "sunset"]),
    ("HotSummer", "Hot Summer", [1.2, 0.2, 0.1, 0.1, 1.1, 0.1, 0.1, 0.2, 1.0], ["summer", "warm"]),
    ("PurpleMist", "Purple Mist", [0.7, 0.3, 0.8, 0.3, 0.6, 0.2, 0.1, 0.1, 1.0], ["purple", "mist"]),
    ("Sunrise", "Sunrise", [1.0, 0.2, 0.1, 0.1, 0.8, 0.2, 0.1, 0.2, 1.1], ["sunrise", "warm"]),
    ("ColorfulSwirl", "Colorful Swirl", [0.6, 0.4, 0.1, 0.1, 0.7, 0.4, 0.2, 0.1, 0.8], ["colorful"]),
    ("SoftSpringLight", "Soft Spring Light", [1.1, 0.1, 0.1, 0.1, 1.1, 0.1, 0.1, 0.1, 1.1], ["spring", "bright"]),
    ("LavenderDream", "Lavender Dream", [0.8, 0.2, 0.2, 0.1, 0.7, 0.3, 0.1, 0.2, 0.9], ["lavender", "purple"]),
    ("Cyberpunk", "Cyberpunk", [0.8, 0.2, 0.5, 0.1, 0.9, 0.3, 0.1, 0.3, 0.9], ["cyberpunk", "neon"]),
    ("LemonadeLight", "Lemonade Light", [1.0, 0.3, 0.1, 0.1, 1.0, 0.1, 0.2, 0.2, 1.0], ["lemonade", "bright"]),
    ("SpectralFire", "Spectral Fire", [0.9, 0.2, 0.1, 0.1, 0.8, 0.2, 0.1, 0.2, 0.9], ["fire"]),
    ("NightMagic", "Night Magic", [0.7, 0.1, 0.2, 0.2, 0.7, 0.1, 0.1, 0.2, 0.8], ["night", "dark"]),
    ("FantasyLandscape", "Fantasy Landscape", [0.8, 0.3, 0.2, 0.1, 0.8, 0.2, 0.1, 0.2, 0.9], ["fantasy"]),
    ("ColorExplosion", "Color Explosion", [1.0, 0.5, 0.2, 0.2, 0.8, 0.3, 0.1, 0.2, 1.0], ["colorful", "vivid"]),
    ("ElectricGradient", "Electric Gradient", [1.0, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 1.0], ["electric"]),
    ("CaramelDarkness", "Caramel Darkness", [0.8, 0.3, 0.2, 0.1, 0.7, 0.1, 0.2, 0.1, 0.8], ["caramel", "dark"]),
    ("FuturisticGradient", "Futuristic Gradient", [0.9, 0.1, 0.3, 0.2, 0.8, 0.1, 0.1, 0.3, 1.0], ["futuristic"]),
    ("GreenSun", "Green Sun", [0.7, 0.2, 0.1, 0.1, 1.1, 0.1, 0.1, 0.3, 0.9], ["green"]),
    ("RainbowWorld", "Rainbow World", [1.0, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0, 0.0, 1.0], ["rainbow"]),
    ("DeepPurple", "Deep Purple", [0.9, 0.2, 0.5, 0.1, 0.8, 0.1, 0.3, 0.5, 1.0], ["purple"]),
    ("SpacePortal", "Space Portal", [0.2, 0.1, 0.8, 0.1, 1.0, 0.1, 0.8, 0.1, 0.2], ["space", "swap"]),
    ("RedSwirl", "Red Swirl", [1.0, 0.5, 0.2, 0.2, 0.3, 0.8, 0.1, 0.8, 0.5], ["red"]),
    ("DigitalCode", "Digital Code", [0.1, 0.8, 0.1, 0.5, 0.2, 0.5, 0.1, 0.8, 0.1], ["digital", "green"]),
    # ColorMatrices.Assistance (colour-vision deficiency simulation)
    ("Protonomaly", "Protanomaly", [0.817, 0.183, 0.0, 0.333, 0.667, 0.0, 0.0, 0.125, 0.875],
     ["color blindness", "protanomaly", "accessibility"]),
    ("Deutaromaly", "Deuteranomaly", [0.8, 0.2, 0.0, 0.258, 0.742, 0.0, 0.0, 0.142, 0.858],
     ["color blindness", "deuteranomaly", "accessibility"]),
    ("Tritonomaly", "Tritanomaly", [0.967, 0.033, 0.0, 0.0, 0.733, 0.267, 0.0, 0.183, 0.817],
     ["color blindness", "tritanomaly", "accessibility"]),
    ("Protanopia", "Protanopia", [0.567, 0.433, 0.0, 0.558, 0.442, 0.0, 0.0, 0.242, 0.758],
     ["color blindness", "protanopia", "accessibility"]),
    ("Deuteranopia", "Deuteranopia", [0.625, 0.375, 0.0, 0.7, 0.3, 0.0, 0.0, 0.3, 0.7],
     ["color blindness", "deuteranopia", "accessibility"]),
    ("Tritanopia", "Tritanopia", [0.95, 0.05, 0.0, 0.0, 0.433, 0.567, 0.0, 0.475, 0.525],
     ["color blindness", "tritanopia", "accessibility"]),
    ("Achromatopsia", "Achromatopsia", [0.299, 0.587, 0.114, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114],
     ["color blindness", "achromatopsia", "grayscale", "accessibility"]),
    ("Achromatomaly", "Achromatomaly", [0.618, 0.320, 0.062, 0.163, 0.775, 0.062, 0.163, 0.320, 0.516],
     ["color blindness", "achromatomaly", "accessibility"]),
]


def _register_matrix(name: str, title: str, matrix: list[float], tags: list[str]) -> None:
    def fn(img, ctx):
        return color_matrix_3x3(img, matrix)

    fn.__name__ = fn.__qualname__ = snake(name)
    fn.__doc__ = f"Fixed 3x3 colour matrix (aire ColorMatrices, ImageToolbox {name}Filter)."
    itb_op(name, title, GROUP, [], upstream="aire", tags=[*tags, "color matrix"])(fn)


for _n, _t, _m, _tags in _MATRICES:
    _register_matrix(_n, _t, _m, _tags)


# ---------------------------------------------------------------------------
# GPUImage
# ---------------------------------------------------------------------------

@itb_op("BlackAndWhite", "Black and White", GROUP, [], upstream="GPUImage",
        tags=["black and white", "grayscale", "monochrome", "luminance"])
def black_and_white(img, ctx):
    """GPUImageLuminanceFilter: luminance with weights (0.2125, 0.7154, 0.0721)."""
    return S.gray3(img @ S.GPU_LUMINANCE_W)


@itb_op("Negative", "Negative", GROUP, [], upstream="GPUImage", tags=["negative", "invert"])
def negative(img, ctx):
    """GPUImageColorInvertFilter: 1 - rgb, alpha kept."""
    return 1.0 - img


@itb_op("CGAColorSpace", "CGA colorspace", GROUP, [], upstream="GPUImage", alpha="full",
        tags=["cga", "retro", "palette", "8-bit", "pixel"])
def cga_color_space(img, ctx):
    """GPUImageCGAColorspaceFilter: 200x320 sample grid snapped to black/white/cyan/magenta (opaque)."""
    return S.opaque(img, S.gpu_cga_colorspace(S.bitmap_rgb(img)))


@itb_op("Laplacian", "Laplacian", GROUP, [], upstream="GPUImage", tags=["laplacian", "edges", "edge detection"])
def laplacian(img, ctx):
    """GPUImageLaplacianFilter: 3x3 Laplacian (0.5 1 0.5 / 1 -6 1 / 0.5 1 0.5) offset by +0.5."""
    return S.gpu_laplacian(img)


@itb_op("NonMaximumSuppression", "Non maximum suppression", GROUP, [], upstream="GPUImage", alpha="full",
        tags=["non maximum suppression", "edges", "thinning"])
def non_maximum_suppression(img, ctx):
    """GPUImageNonMaximumSuppressionFilter: keep pixels whose red value is a 3x3 local maximum (opaque)."""
    return S.opaque(img, S.gpu_non_maximum_suppression(S.bitmap_rgb(img)))


@itb_op("WeakPixel", "Weak pixel inclusion", GROUP, [], upstream="GPUImage", alpha="full",
        tags=["weak pixel", "edges", "threshold"])
def weak_pixel(img, ctx):
    """GPUImageWeakPixelInclusionFilter: white where the 3x3 red sum >= 1.5 and the centre >= 0.01 (opaque)."""
    return S.opaque(img, S.gpu_weak_pixel_inclusion(S.bitmap_rgb(img)))


# ---------------------------------------------------------------------------
# aire
# ---------------------------------------------------------------------------

@itb_op("EqualizeHistogram", "Equalize Histogram", GROUP, [], upstream="aire",
        tags=["equalize", "histogram", "contrast", "auto"])
def equalize_histogram(img, ctx):
    """Aire.equalizeHist: equalise the luma of full-range YCgCo (chroma kept)."""
    return S.aire_equalize_hist(img)


@itb_op("LaplacianSimple", "Laplacian Simple", GROUP, [], upstream="aire", alpha="full",
        tags=["laplacian", "edges", "edge detection"])
def laplacian_simple(img, ctx):
    """Aire.laplacian (REFLECT_101): 8-neighbour Laplacian of the 8-bit luma, negatives clipped (opaque)."""
    return S.opaque(img, S.aire_laplacian(S.bitmap_rgb(img)))


@itb_op("SobelSimple", "Sobel Simple", GROUP, [], upstream="aire", alpha="full",
        tags=["sobel", "edges", "edge detection"])
def sobel_simple(img, ctx):
    """Aire.sobel (REFLECT_101): horizontal Sobel of the 8-bit luma, each pass saturated (opaque)."""
    return S.opaque(img, S.aire_sobel(S.bitmap_rgb(img)))


# ---------------------------------------------------------------------------
# jhlabs
# ---------------------------------------------------------------------------

@itb_op("Despeckle", "Despeckle", GROUP, [], upstream="jhlabs", tags=["despeckle", "denoise", "noise"])
def despeckle(img, ctx):
    """jhlabs DespeckleFilter: pepper-and-salt +-1 level nudges against the 4 neighbour pairs."""
    return S.jh_despeckle(img)


@itb_op("ReduceNoise", "Reduce Noise", GROUP, [], upstream="jhlabs", tags=["denoise", "noise", "smooth"])
def reduce_noise(img, ctx):
    """jhlabs ReduceNoiseFilter: clamp every channel to the range of its 8 neighbours."""
    return S.jh_reduce_noise(img)


@itb_op("Equalize", "Equalize", GROUP, [], upstream="jhlabs", tags=["equalize", "histogram", "contrast", "auto"])
def equalize(img, ctx):
    """jhlabs EqualizeFilter: per-channel histogram equalisation."""
    return S.jh_equalize(img)


@itb_op("SimpleSolarize", "Simple Solarize", GROUP, [], upstream="jhlabs", tags=["solarize", "invert"])
def simple_solarize(img, ctx):
    """jhlabs SolarizeFilter: v > 0.5 ? 2(v - 0.5) : 2(0.5 - v)."""
    return np.abs(2.0 * img - 1.0)


@itb_op("JavaLookAndFeel", "Java Look and Feel", GROUP, [], upstream="jhlabs", alpha="full",
        tags=["java", "stipple", "checkerboard", "disabled"])
def java_look_and_feel(img, ctx):
    """jhlabs JavaLnFFilter: Swing 'disabled icon' stipple (75% #999999 on every other pixel)."""
    return S.jh_java_lnf(img)


# ---------------------------------------------------------------------------
# Trickle
# ---------------------------------------------------------------------------

@itb_op("Gotham", "Gotham", GROUP, [], upstream="Trickle", alpha="full", tags=["gotham", "blue", "dark", "moody"])
def gotham(img, ctx):
    """Trickle GothamFilter: darken, gamma 0.7, HSI blue re-tint, contrast 1.2 (opaque)."""
    return S.opaque(img, S.trickle_gotham(S.bitmap_rgb(img)))


@itb_op("HDR", "HDR", GROUP, [], upstream="Trickle", alpha="full", tags=["hdr", "contrast", "sharpen", "local contrast"])
def hdr(img, ctx):
    """Trickle HDRFilter: overlay with a small gaussian blur of itself, then Laplacian sharpen (opaque)."""
    return S.opaque(img, S.trickle_hdr(S.bitmap_rgb(img)))


@itb_op("SimpleOldTv", "Simple Old TV", GROUP, [], upstream="Trickle", alpha="full",
        tags=["tv", "scanlines", "retro", "crt"])
def simple_old_tv(img, ctx):
    """Trickle TvFilter: 4-row scanline pattern (blue average, green average, black, original)."""
    return S.trickle_tv(img)


@itb_op("SimpleSketch", "Simple Sketch", GROUP, [], upstream="Trickle", alpha="full",
        tags=["sketch", "pencil", "drawing", "edges"])
def simple_sketch(img, ctx):
    """Trickle SketchFilter: black where the grey differs from its bottom-right neighbour by >= 7 (opaque)."""
    return S.opaque(img, S.trickle_sketch(S.bitmap_rgb(img)))


# ---------------------------------------------------------------------------
# ImageToolbox chains / opencv-tools
# ---------------------------------------------------------------------------

@itb_op("OldTv", "Old Tv", GROUP, [], upstream="ImageToolbox", alpha="full",
        tags=["old tv", "retro", "grain", "green", "monochrome"])
def old_tv(img, ctx):
    """ImageToolbox OldTvFilter chain: aire grain(0.36) -> GPUImage haze(0, 0.3) -> aire monochrome(#1C3A00)."""
    return S.old_tv(img, ctx.rng("old_tv_grain"))


_COLOR_MAPS = ["autumn", "bone", "jet", "winter", "rainbow", "ocean", "summer", "spring", "cool", "hsv", "pink",
               "hot", "parula", "magma", "inferno", "plasma", "viridis", "cividis", "twilight", "twilight_shifted",
               "turbo", "deep_green"]  # == ColorMapType ordinals == cv::ColormapTypes
_COLOR_MAP_LABELS = {c: c.replace("_", " ").title() for c in _COLOR_MAPS}
_COLOR_MAP_LABELS.update({"cool": "Cool Variant", "hsv": "HSV"})


@itb_op("ColorMap", "Color Map", GROUP, [
    Choice("color_map", "autumn", choices=_COLOR_MAPS, labels=_COLOR_MAP_LABELS, title="Color map",
           description="OpenCV colour map applied to the grey levels (ImageToolbox lists each one as its own filter)"),
], upstream="OpenCV", alpha="full", tags=["color map", "false color", "heatmap", "jet", "viridis", "autumn"])
def color_map(img, ctx, color_map):
    """opencv-tools ColorMap.apply: grey -> cv::applyColorMap (opaque)."""
    return S.opaque(img, S.cv_color_map(S.bitmap_rgb(img), _COLOR_MAPS.index(color_map)))


@itb_op("AverageDistance", "Average Distance", GROUP, [], upstream="OpenCV", alpha="full",
        tags=["forensics", "analysis", "noise", "edges"])
def average_distance(img, ctx):
    """opencv-tools ImageForensics.averageDistance: |pixel - 4-neighbour mean|, normalised (opaque)."""
    return S.opaque(img, S.cv_average_distance(S.bitmap_rgb(img)))


@itb_op("LuminanceGradient", "Luminance Gradient", GROUP, [], upstream="OpenCV", alpha="full",
        tags=["forensics", "analysis", "gradient", "sobel"])
def luminance_gradient(img, ctx):
    """opencv-tools ImageForensics.luminanceGradient: gradient direction in R/G, magnitude in B (opaque)."""
    return S.opaque(img, S.cv_luminance_gradient(S.bitmap_rgb(img)))


@itb_op("Moire", "Moire", GROUP, [], upstream="OpenCV", tags=["moire", "descreen", "fft", "denoise"])
def moire(img, ctx):
    """opencv-tools Moire.remove (AUTO): FFT peak notching + Butterworth low-pass per channel."""
    return S.cv_moire(img, "auto", ctx.check)


@itb_op("AutoPerspective", "Perspective Auto", GROUP, [], upstream="OpenCV", alpha="full", geometry=True,
        tags=["perspective", "document", "scan", "straighten", "auto"])
def auto_perspective(img, ctx):
    """opencv-tools AutoStraighten (Perspective): detect the dominant quadrilateral and rectify it."""
    return S.auto_perspective(img, ctx.check)

