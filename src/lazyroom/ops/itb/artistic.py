"""ImageToolbox "Artistic" group filters.

30 of the 40 filters are G'MIC commands (exact command strings of the
ImageToolboxLibs builders, ``com.t8rin.gmic.filters``); they run through
:func:`._artistic_impl.gmic_itb`, which mirrors ImageToolboxLibs' native bridge
(alpha modes, disabled ``smooth`` built-in).  The other ten are ports of
Trickle, aire, jhlabs and GPUImage implementations (see ``_artistic_impl``).
"""
from __future__ import annotations

import math

from ...core.params import Bool, Choice, Color, Float, Int
from . import _artistic_impl as impl
from ._common import gmic_cmd, gmic_pipeline, itb_op

GROUP = "artistic"
GMIC = dict(upstream="G'MIC", requires=["gmic"])

# com.t8rin.gmic.filters.GmicChannel (value = index)
GMIC_CHANNELS = (
    "all", "rgba", "rgb", "red", "green", "blue", "alpha", "linear_rgb", "linear_red", "linear_green",
    "linear_blue", "ycbcr", "luminance", "chrominance", "blue_chrominance", "red_chrominance",
    "green_chrominance", "lab", "lab_lightness", "lab_chrominance", "lab_a", "lab_b", "lch",
    "lch_chrominance", "lch_chroma", "lch_hue", "hsv", "hsv_hue", "hsv_saturation", "hsv_value", "hsi",
    "hsi_intensity", "hsl", "hsl_lightness", "cmyk", "cmy", "cyan", "magenta", "yellow", "key", "yiq",
    "yiq_luma", "yiq_chrominance", "ryb", "ryb_red", "ryb_yellow", "ryb_blue",
)
BRUSH_SHAPES = ("rectangle", "diamond", "pentagon", "hexagon", "octagon", "ellipse", "gaussian", "star", "heart")


def _f(name, default, lo, hi, title, **kw):
    """G'MIC float slider (GmicParameterInfo.Number, roundTo = 2)."""
    return Float(name, default, min=lo, max=hi, step=0.01, title=title, **kw)


def _i(name, default, lo, hi, title, **kw):
    return Int(name, default, min=lo, max=hi, title=title, **kw)


def _rgb(c):
    return impl.color255(c)[:3]


# ---------------------------------------------------------------------------
# G'MIC filters
# ---------------------------------------------------------------------------

@itb_op("Brushify", "Brushify", GROUP, [
    Choice("shape", "ellipse", choices=BRUSH_SHAPES, title="Shape"),
    _f("ratio", 0.25, 0, 1, "Ratio"),
    _i("number_of_sizes", 4, 1, 16, "Number Of Sizes"),
    _i("maximal_size", 64, 1, 128, "Maximal Size"),
    _f("minimal_size", 25, 0, 100, "Minimal Size", description="Percent of the maximal size"),
    _i("orientation", 12, 1, 24, "Orientation", description="Number of brush orientations"),
    _f("fuzziness", 0, 0, 10, "Fuzziness"),
    _f("smoothness", 2, 0, 10, "Smoothness"),
    Choice("light", "full", choices=("none", "flat", "darken", "lighten", "full"), title="Light"),
    _f("strength", 0.2, 0, 1, "Strength", description="Light strength"),
    _f("opacity", 0.5, 0, 1, "Opacity"),
    _f("density", 30, 0, 100, "Density"),
    _f("contour_coherence", 1, 0, 1, "Contour Coherence"),
    _f("orientation_coherence", 1, 0, 1, "Orientation Coherence"),
    _f("gradient_smoothness", 1, 0, 10, "Gradient Smoothness"),
    _f("structure_smoothness", 5, 0, 10, "Structure Smoothness"),
    _f("primary_angle", 0, -180, 180, "Primary Angle"),
    _f("dispersion", 0.2, 0, 1, "Dispersion", description="Angle dispersion"),
], **GMIC, deterministic=False, tags=["brush", "painting", "strokes"])
def brushify(img, ctx, shape, ratio, number_of_sizes, maximal_size, minimal_size, orientation, fuzziness,
             smoothness, light, strength, opacity, density, contour_coherence, orientation_coherence,
             gradient_smoothness, structure_smoothness, primary_angle, dispersion):
    """G'MIC fx_brushify (random brush placement is multi-threaded in G'MIC, hence not reproducible)."""
    lights = ("none", "flat", "darken", "lighten", "full")
    return impl.gmic_itb(img, ctx, gmic_cmd(
        "fx_brushify", BRUSH_SHAPES.index(shape) + 2, ratio, number_of_sizes, maximal_size, minimal_size,
        orientation, fuzziness, smoothness, lights.index(light), strength, opacity, density, contour_coherence,
        orientation_coherence, gradient_smoothness, structure_smoothness, primary_angle, dispersion))


@itb_op("Charcoal", "Charcoal", GROUP, [
    _i("granularity", 65, 0, 800, "Granularity"),
    _i("lowlight_crossover", 70, 0, 255, "Lowlight Crossover"),
    _i("highlight_crossover", 170, 0, 255, "Highlight Crossover"),
    Bool("contrast", False, title="Contrast", description="Boost contrast"),
    Bool("optimize_size", True, title="Optimize Size", description="Resize the image for the optimum effect"),
    Bool("chalk_highlights", False, title="Chalk Highlights"),
    _i("minimum_highlight", 50, 0, 255, "Minimum Highlight"),
    _i("maximum_highlight", 70, 0, 255, "Maximum Highlight"),
    Color("background_color", "#FFFFFF", title="Background color"),
    Color("foreground_color", "#000000", title="Foreground Color"),
    Bool("invert_colors", False, title="Invert Colors"),
], **GMIC, tags=["charcoal", "drawing", "sketch", "black and white"])
def charcoal(img, ctx, granularity, lowlight_crossover, highlight_crossover, contrast, optimize_size,
             chalk_highlights, minimum_highlight, maximum_highlight, background_color, foreground_color,
             invert_colors):
    """G'MIC fx_charcoal."""
    return impl.gmic_itb(img, ctx, gmic_cmd(
        "fx_charcoal", granularity, lowlight_crossover, highlight_crossover, contrast, optimize_size,
        chalk_highlights, minimum_highlight, maximum_highlight, *_rgb(background_color), *_rgb(foreground_color),
        invert_colors))


@itb_op("ColorAbstraction", "Color Abstraction", GROUP, [
    _f("smoothness", 1, 0, 10, "Smoothness"),
    _i("levels", 10, 2, 100, "Levels"),
    _f("contrast", 0.2, 0.01, 1, "Contrast"),
], **GMIC, tags=["abstract", "posterize", "flat colors"])
def color_abstraction(img, ctx, smoothness, levels, contrast):
    """G'MIC fx_color_abstraction."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_color_abstraction", smoothness, levels, contrast))


@itb_op("Cubism", "Cubism", GROUP, [
    _i("iterations", 2, 0, 10, "Iterations"),
    _f("density", 50, 0, 200, "Density"),
    _f("thickness", 10, 0, 50, "Thickness"),
    _f("angle", 90, 0, 360, "Angle"),
    _f("opacity", 0.7, 0.01, 1, "Opacity"),
    _f("smoothness", 0, 0, 5, "Smoothness"),
], **GMIC, tags=["cubism", "painting", "abstract"])
def cubism(img, ctx, iterations, density, thickness, angle, opacity, smoothness):
    """G'MIC fx_cubism."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_cubism", iterations, density, thickness, angle, opacity, smoothness))


@itb_op("Cutout", "Cutout", GROUP, [
    _i("levels", 4, 2, 32, "Levels"),
    _f("edge_simplicity", 0.5, 0, 3, "Edge Simplicity"),
    _i("edge_fidelity", 4, 0, 10, "Edge Fidelity"),
    Bool("normalize", True, title="Normalize"),
], **GMIC, tags=["cutout", "paper", "posterize", "flat colors"])
def cutout(img, ctx, levels, edge_simplicity, edge_fidelity, normalize):
    """G'MIC fx_cutout."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_cutout", levels, edge_simplicity, edge_fidelity, normalize))


@itb_op("DodgeSketch", "Dodge Sketch", GROUP, [
    _i("power", 3, 0, 10, "Power"),
    _f("spatial_variance", 10, 0, 100, "Spatial Variance"),
    _f("value_variance", 7, 0, 100, "Value Variance"),
    _i("iterations", 2, 1, 10, "Iterations"),
    Bool("colored", False, title="Colored"),
], **GMIC, tags=["sketch", "pencil", "drawing", "dodge"])
def dodge_sketch(img, ctx, power, spatial_variance, value_variance, iterations, colored):
    """G'MIC fx_dodgesketch."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_dodgesketch", power, spatial_variance, value_variance,
                                            iterations, colored))


@itb_op("Doodle", "Doodle", GROUP, [
    _f("precision", 30, 0, 100, "Precision"),
    _f("smoothness", 2, 0, 10, "Smoothness"),
    _f("coherence", 2, 0, 10, "Coherence"),
    _f("threshold", 1.5, 0, 10, "Threshold", description="Contour threshold"),
    _i("spacing", 2, 0, 20, "Spacing"),
    _f("minimum_stroke_length", 70, 0, 255, "Minimum Stroke Length"),
], **GMIC, tags=["doodle", "drawing", "lines", "slow"])
def doodle(img, ctx, precision, smoothness, coherence, threshold, spacing, minimum_stroke_length):
    """G'MIC fx_doodle (very slow on large images)."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_doodle", precision, smoothness, coherence, threshold, spacing,
                                            minimum_stroke_length, False))


@itb_op("DreamPainting", "Dream Painting", GROUP, [
    _i("abstraction", 4, 1, 10, "Abstraction"),
    _f("details_scale", 2, 0, 5, "Details Scale"),
    _f("color_strength", 1.7, 0, 4, "Color Strength"),
    _f("paint_smoothness", 40, 0, 1000, "Paint Smoothness"),
    Bool("sharpen_shades", True, title="Sharpen Shades"),
    _f("bloom_strength", 45, 0, 400, "Bloom Strength"),
    _f("bloom_scale", 35, 0, 255, "Bloom Scale"),
    _f("bloom_smoothness", 45, 0, 255, "Bloom Smoothness"),
    _i("highlights", 35, 0, 100, "Highlights"),
    _f("contrast", 12, 0, 100, "Contrast"),
], **GMIC, tags=["painting", "dreamy", "bloom", "glow"])
def dream_painting(img, ctx, abstraction, details_scale, color_strength, paint_smoothness, sharpen_shades,
                   bloom_strength, bloom_scale, bloom_smoothness, highlights, contrast):
    """G'MIC fx_painting + fx_highlight_bloom."""
    return impl.gmic_itb(img, ctx, gmic_pipeline(
        gmic_cmd("fx_painting", abstraction, details_scale, color_strength, paint_smoothness, sharpen_shades),
        gmic_cmd("fx_highlight_bloom", bloom_strength, bloom_scale, bloom_smoothness, highlights, contrast),
    ), no_smooth=True)


@itb_op("Engrave", "Engrave", GROUP, [
    _f("radius", 0.5, 0, 2, "Radius"),
    _f("density", 50, 0, 200, "Density"),
    _f("edges", 0, 0, 10, "Edges"),
    _f("coherence", 8, 0, 40, "Coherence"),
    _f("threshold", 40, 0, 100, "Threshold"),
    _i("minimal_area", 0, -256, 256, "Minimal Area"),
    _f("flat_regions_removal", 0, 0, 10, "Flat Regions Removal"),
    Bool("add_color_background", True, title="Add Color Background"),
    _f("quantization", 10, 0, 40, "Quantization"),
    _i("shading", 1, 0, 5, "Shading"),
    _f("hue", 0, -180, 180, "Hue"),
    _f("saturation", 0, -100, 100, "Saturation"),
    _f("lightness", 0, -100, 100, "Lightness"),
    Choice("antialias", "x1_5", choices=("disabled", "x1_5", "x2", "x3"), title="Antialias"),
], **GMIC, alpha="full", tags=["engrave", "etching", "lines", "print"])
def engrave(img, ctx, radius, density, edges, coherence, threshold, minimal_area, flat_regions_removal,
            add_color_background, quantization, shading, hue, saturation, lightness, antialias):
    """G'MIC fx_engrave_preview (alpha processed by G'MIC, like ImageToolbox)."""
    aa = ("disabled", "x1_5", "x2", "x3").index(antialias)
    return impl.gmic_itb(img, ctx, gmic_cmd(
        "fx_engrave_preview", radius, density, edges, coherence, threshold, minimal_area, flat_regions_removal,
        add_color_background, quantization, shading, hue, saturation, lightness, aa), mode="process", no_smooth=True)


@itb_op("FeltPen", "Felt Pen", GROUP, [
    _f("amplitude", 300, 0, 4000, "Amplitude"),
    _f("density", 50, 0, 100, "Density"),
    _f("smoothness", 1, 0, 10, "Smoothness"),
    _f("opacity", 0.1, 0, 1, "Opacity"),
    _f("edges", 20, 0, 100, "Edges"),
    _i("thickness", 5, 2, 32, "Thickness"),
], **GMIC, deterministic=False, tags=["felt pen", "marker", "drawing", "strokes"])
def felt_pen(img, ctx, amplitude, density, smoothness, opacity, edges, thickness):
    """G'MIC fx_feltpen (multi-threaded random strokes are not bit-reproducible)."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_feltpen", amplitude, density, smoothness, opacity, edges, thickness))


@itb_op("GraphicNovel", "Graphic Novel", GROUP, [
    _f("smoothness", 18, 0, 100, "Smoothness"),
    _f("threshold", 55, 0, 100, "Threshold", description="Edge threshold"),
    _f("edge_shade", 7, 0, 30, "Edge Shade"),
    _f("edge_thickness", 1, 0, 5, "Edge Thickness"),
    _i("posterization", 7, 0, 15, "Posterization"),
    _f("strength", 35, 0, 100, "Strength", description="Retinex strength"),
    _f("retinex_regularization", 4, 0, 32, "Retinex Regularization"),
], **GMIC, tags=["comic", "graphic novel", "poster", "edges"])
def graphic_novel(img, ctx, smoothness, threshold, edge_shade, edge_thickness, posterization, strength,
                  retinex_regularization):
    """G'MIC fx_poster_edges + fx_retinex (Lab)."""
    return impl.gmic_itb(img, ctx, gmic_pipeline(
        gmic_cmd("fx_poster_edges", smoothness, threshold, edge_shade, edge_thickness, 12, posterization, 20),
        gmic_cmd("fx_retinex", strength, 16, 2, 1, 1, retinex_regularization, 15, 80, 250),
    ), no_smooth=True)


@itb_op("Hedcut", "Hedcut", GROUP, [
    _f("contrast", 0.5, 0, 1, "Contrast"),
    _f("luminance_level", 0.5, 0, 1, "Luminance Level"),
    _f("smoothing", 0.5, 0, 1, "Smoothing"),
    _f("size", 0.0, 0, 1, "Size"),
    _f("step", 0.5, 0, 1, "Step"),
    Bool("quality", False, title="Quality"),
    Bool("force_gray", True, title="Force Gray"),
], **GMIC, tags=["hedcut", "stipple", "engraving", "portrait"])
def hedcut(img, ctx, contrast, luminance_level, smoothing, size, step, quality, force_gray):
    """G'MIC hedcut."""
    return impl.gmic_itb(img, ctx, gmic_cmd("hedcut", contrast, luminance_level, smoothing, size, step, quality,
                                            force_gray))


@itb_op("HopePoster", "Hope Poster", GROUP, [
    _f("gamma", 0, -3, 3, "Gamma"),
    _f("smoothness", 3, 0, 20, "Smoothness"),
], **GMIC, tags=["poster", "hope", "obama", "stencil"])
def hope_poster(img, ctx, gamma, smoothness):
    """G'MIC fx_poster_hope (its smoothing step is skipped, as in ImageToolbox)."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_poster_hope", gamma, smoothness), no_smooth=True)


@itb_op("HoughSketch", "Hough Sketch", GROUP, [
    _f("smoothness", 1.25, 0, 10, "Smoothness"),
    _f("density", 18, 0, 100, "Density"),
    _i("radius", 5, 0, 30, "Radius"),
    _f("threshold", 70, 0, 100, "Threshold"),
    _f("opacity", 0.35, 0, 1, "Opacity"),
    Choice("color_model", "color_on_white", choices=(
        "black_on_white", "white_on_black", "black_on_transparent_white", "white_on_transparent_black",
        "color_on_white"), title="Color Model"),
], **GMIC, tags=["sketch", "hough", "lines", "drawing"])
def hough_sketch(img, ctx, smoothness, density, radius, threshold, opacity, color_model):
    """G'MIC fx_houghsketchbw (ImageToolbox keeps the source alpha, so 'transparent' models stay opaque).

    ``radius = 0`` makes G'MIC fail (``normalize_local`` rejects it) and is run as 1.
    """
    models = ("black_on_white", "white_on_black", "black_on_transparent_white", "white_on_transparent_black",
              "color_on_white")
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_houghsketchbw", smoothness, density, max(1, radius), threshold,
                                            opacity, models.index(color_model)))


@itb_op("InkWash", "Ink Wash", GROUP, [
    _f("size", 0.14, 0, 4, "Size"),
    _f("amplitude", 23, 0, 200, "Amplitude"),
    Bool("skip_other_steps", False, title="Skip Other Steps"),
    _f("smoother_sharpness", 0.5, 0, 2, "Smoother Sharpness"),
    _f("edge_protection", 0.54, 0, 1, "Edge Protection"),
    _f("softness", 2.25, 0, 10, "Softness"),
    Choice("contrast", "none", choices=("none", "automatic", "automatic_with_contrast_mask", "manual"),
           title="Contrast", description="Stretch contrast"),
    _f("local_normalization_amplitude", 2, 0, 60, "Local Normalization Amplitude"),
    _f("local_normalization_size", 6, 0, 64, "Local Normalization Size"),
    _f("neighborhood_smoothness", 5, 0, 40, "Neighborhood Smoothness"),
    _f("average_smoothness", 20, 0, 40, "Average Smoothness"),
], **GMIC, tags=["ink", "wash", "sumi-e", "black and white"])
def ink_wash(img, ctx, size, amplitude, skip_other_steps, smoother_sharpness, edge_protection, softness, contrast,
             local_normalization_amplitude, local_normalization_size, neighborhood_smoothness, average_smoothness):
    """G'MIC fx_ink_wash."""
    c = ("none", "automatic", "automatic_with_contrast_mask", "manual").index(contrast)
    return impl.gmic_itb(img, ctx, gmic_cmd(
        "fx_ink_wash", size, amplitude, skip_other_steps, smoother_sharpness, edge_protection, softness, c,
        local_normalization_amplitude, local_normalization_size, neighborhood_smoothness, average_smoothness),
        no_smooth=True)


@itb_op("Linify", "Linify", GROUP, [
    _f("density", 40, 0, 100, "Density"),
    _f("spread", 2, 0, 10, "Spread"),
    _f("resolution", 40, 0, 100, "Resolution"),
    _f("opacity", 10, 0, 30, "Opacity"),
    _i("line_precision", 24, 1, 128, "Line Precision"),
    Choice("color_mode", "subtractive", choices=("subtractive", "additive"), title="Color Mode"),
], **GMIC, tags=["lines", "string art", "linify", "slow"])
def linify(img, ctx, density, spread, resolution, opacity, line_precision, color_mode):
    """G'MIC fx_linify (very slow on large images).

    ``resolution = 0`` makes G'MIC fail (``linify`` requires > 0) and is run as 0.01.
    """
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_linify", density, spread, max(0.01, resolution), opacity,
                                            line_precision, ("subtractive", "additive").index(color_mode), False))


@itb_op("LylejkPainting", "Lylejk Painting", GROUP, [
    _i("iterations", 10, 1, 20, "Iterations"),
    _i("abstraction", 2, 1, 20, "Abstraction"),
    _i("radius", 4, 1, 30, "Radius"),
    _f("canvas", 10, 0, 100, "Canvas"),
], **GMIC, tags=["painting", "canvas", "kuwahara"])
def lylejk_painting(img, ctx, iterations, abstraction, radius, canvas):
    """G'MIC fx_lylejk_painting."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_lylejk_painting", iterations, abstraction, radius, canvas),
                         no_smooth=True)


@itb_op("MarkerDrawing", "Marker Drawing", GROUP, [
    _i("length", 70, 1, 256, "Length", unit="px"),
    _f("radius", 4, 1, 32, "Radius", unit="px"),
    _f("opacity", 10, 0, 100, "Opacity"),
    _f("inertia", 15, 0, 100, "Inertia"),
    _f("curviness", 100, 0, 200, "Curviness"),
    _f("anisotropy", 85, 0, 100, "Anisotropy"),
    _f("smoothness", 0.5, 0, 5, "Smoothness"),
    _f("coherence", 3, 0, 16, "Coherence"),
    _i("iterations", 3, 1, 64, "Iterations"),
    Choice("background", "color", choices=("blur", "color"), title="Background"),
    _f("background_blur", 20, 0, 128, "Background Blur"),
    Color("background_color", "#FFFFFFFF", title="Background color"),
], **GMIC, alpha="full", deterministic=False, tags=["marker", "drawing", "strokes"])
def marker_drawing(img, ctx, length, radius, opacity, inertia, curviness, anisotropy, smoothness, coherence,
                   iterations, background, background_blur, background_color):
    """G'MIC fx_marker_drawing (alpha processed by G'MIC; multi-threaded random strokes are not reproducible)."""
    return impl.gmic_itb(img, ctx, gmic_cmd(
        "fx_marker_drawing", min(256, ctx.ipx(length)), min(32.0, max(1.0, ctx.px(radius))), opacity, inertia,
        curviness, anisotropy, smoothness, coherence, iterations, ("blur", "color").index(background),
        background_blur, *impl.color255(background_color)), mode="process")


@itb_op("OldPhotograph", "Old Photograph", GROUP, [
    _f("vignette_strength", 200, 0, 255, "Vignette Strength"),
    _f("vignette_min_radius", 50, 0, 100, "Vignette Min Radius"),
    _f("vignette_max_radius", 85, 0, 100, "Vignette Max Radius"),
], **GMIC, deterministic=False, tags=["old photo", "vintage", "sepia", "vignette"])
def old_photograph(img, ctx, vignette_strength, vignette_min_radius, vignette_max_radius):
    """G'MIC fx_old_photo (multi-threaded random noise is not bit-reproducible)."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_old_photo", vignette_strength, vignette_min_radius,
                                            vignette_max_radius))


@itb_op("Painting", "Painting", GROUP, [
    _i("abstraction", 5, 1, 10, "Abstraction"),
    _f("scale", 2.5, 0, 5, "Scale", description="Details scale"),
    _f("color", 1.5, 0, 4, "Color"),
    _f("smoothness", 50, 0, 1000, "Smoothness"),
    Bool("sharpen_shades", True, title="Sharpen Shades"),
], **GMIC, tags=["painting", "paint", "abstract"])
def painting(img, ctx, abstraction, scale, color, smoothness, sharpen_shades):
    """G'MIC fx_painting."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_painting", abstraction, scale, color, smoothness, sharpen_shades),
                         no_smooth=True)


@itb_op("PenDrawing", "Pen Drawing", GROUP, [
    _f("amplitude", 10, 0, 30, "Amplitude"),
], **GMIC, tags=["pen", "drawing", "ink"])
def pen_drawing(img, ctx, amplitude):
    """G'MIC fx_pen_drawing (the amplitude only drives its skipped smoothing step, as in ImageToolbox)."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_pen_drawing", amplitude), no_smooth=True)


@itb_op("PencilPortrait", "Pencil Portrait", GROUP, [
    _f("stroke_length", 30, 0, 500, "Stroke Length"),
    _f("angle", 120, 0, 180, "Angle"),
    _f("threshold", 1, 0, 10, "Threshold", description="Contour threshold"),
    _f("opacity", 0.5, 0, 1, "Opacity"),
    Color("color", "#904F15", title="Color"),
], **GMIC, deterministic=False, tags=["pencil", "portrait", "sketch", "drawing"])
def pencil_portrait(img, ctx, stroke_length, angle, threshold, opacity, color):
    """G'MIC fx_pencil_portraitbw (multi-threaded random strokes are not bit-reproducible)."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_pencil_portraitbw", stroke_length, angle, threshold, opacity,
                                            *_rgb(color)), no_smooth=True)


@itb_op("PosterEdges", "Poster Edges", GROUP, [
    _f("softness", 20, 0, 100, "Softness", description="Image smoothness"),
    _f("threshold", 60, 0, 100, "Threshold", description="Edge threshold"),
    _f("edge_shade", 5, 0, 30, "Edge Shade"),
    _f("edge_thickness", 0, 0, 5, "Edge Thickness"),
    _f("edge_antialiasing", 10, 0, 100, "Edge Antialiasing"),
    _i("levels", 0, 0, 15, "Levels", description="Posterization level"),
    _f("posterization_antialiasing", 0, 0, 100, "Posterization Antialiasing"),
], **GMIC, tags=["poster", "edges", "cartoon"])
def poster_edges(img, ctx, softness, threshold, edge_shade, edge_thickness, edge_antialiasing, levels,
                 posterization_antialiasing):
    """G'MIC fx_poster_edges."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_poster_edges", softness, threshold, edge_shade, edge_thickness,
                                            edge_antialiasing, levels, posterization_antialiasing), no_smooth=True)


@itb_op("Rodilius", "Rodilius", GROUP, [
    _f("amplitude", 10, 0, 30, "Amplitude"),
    _f("thickness", 10, 0, 100, "Thickness"),
    _f("sharpness", 300, 0, 1000, "Sharpness"),
    _i("orientation", 5, 2, 36, "Orientation", description="Number of orientations"),
    _f("offset", 30, 0, 180, "Offset"),
    _i("smoothness", 0, 0, 5, "Smoothness"),
    Choice("color_mode", "lighter", choices=("darker", "lighter"), title="Color Mode"),
    Choice("channel", "all", choices=GMIC_CHANNELS, title="Channel"),
    Choice("value_action", "none", choices=("none", "cut", "normalize"), title="Value Action"),
], **GMIC, tags=["rodilius", "streaks", "light", "abstract"])
def rodilius(img, ctx, amplitude, thickness, sharpness, orientation, offset, smoothness, color_mode, channel,
             value_action):
    """G'MIC fx_rodilius."""
    return impl.gmic_itb(img, ctx, gmic_cmd(
        "fx_rodilius", amplitude, thickness, sharpness, orientation, offset, smoothness,
        ("darker", "lighter").index(color_mode), GMIC_CHANNELS.index(channel),
        ("none", "cut", "normalize").index(value_action)), no_smooth=True)


@itb_op("SharpAbstract", "Sharp Abstract", GROUP, [
    _f("spatial_scale", 4, 0, 32, "Spatial Scale"),
    _f("value_scale", 10, 0, 16, "Value Scale"),
    _f("precision", 0.5, 0, 2, "Precision"),
    Choice("channel", "all", choices=GMIC_CHANNELS, title="Channel"),
], **GMIC, tags=["abstract", "sharp", "flat colors"])
def sharp_abstract(img, ctx, spatial_scale, value_scale, precision, channel):
    """G'MIC fx_sharp_abstract."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_sharp_abstract", spatial_scale, value_scale, precision,
                                            GMIC_CHANNELS.index(channel)))


@itb_op("SmoothAbstract", "Smooth Abstract", GROUP, [
    _f("smoothness", 75, 0, 100, "Smoothness"),
    Choice("regularization", "isotropic", choices=("isotropic", "delaunay_guided", "edge_oriented"),
           title="Regularization"),
    _i("iterations", 20, 0, 100, "Iterations"),
    _f("geometry", 1, 0, 5, "Geometry"),
    _f("detail", 30, 0, 50, "Detail"),
], **GMIC, tags=["abstract", "smooth", "flat colors"])
def smooth_abstract(img, ctx, smoothness, regularization, iterations, geometry, detail):
    """G'MIC fx_smooth_abstract."""
    reg = ("isotropic", "delaunay_guided", "edge_oriented").index(regularization)
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_smooth_abstract", smoothness, reg, iterations, geometry, detail),
                         no_smooth=True)


@itb_op("StainedGlass", "Stained Glass", GROUP, [
    _f("edges", 40, 0, 100, "Edges"),
    _f("shading", 0.1, 0, 0.5, "Shading"),
    Bool("thin_separators", False, title="Thin Separators"),
    Bool("equalize", True, title="Equalize"),
    _f("color", 1, 0, 3, "Color"),
    _f("brightness", 0, -100, 100, "Brightness"),
    _f("contrast", 0, -100, 100, "Contrast"),
    _f("gamma", 0, -100, 100, "Gamma"),
], **GMIC, tags=["stained glass", "mosaic", "cells"])
def stained_glass(img, ctx, edges, shading, thin_separators, equalize, color, brightness, contrast, gamma):
    """G'MIC fx_stained_glass."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_stained_glass", edges, shading, thin_separators, equalize, color,
                                            brightness, contrast, gamma))


@itb_op("Stringify", "Stringify", GROUP, [
    _i("smoothness", 2, 0, 10, "Smoothness"),
    _i("levels", 32, 2, 64, "Levels"),
    _f("sampling_rate", 20, 0, 100, "Sampling Rate"),
    Choice("link", "three_points", choices=("two_points", "three_points", "four_points", "all_points"),
           title="Link"),
    _f("offset", 100, 0, 100, "Offset"),
    _i("threshold", 32, 0, 1024, "Threshold"),
    _f("opacity", 20, 0, 100, "Opacity"),
    _f("normalization", 25, 0, 100, "Normalization"),
    Bool("fill_holes", False, title="Fill Holes"),
    Color("background_color", "#000000", title="Background color"),
], **GMIC, deterministic=False, tags=["string art", "lines", "abstract"])
def stringify(img, ctx, smoothness, levels, sampling_rate, link, offset, threshold, opacity, normalization,
              fill_holes, background_color):
    """G'MIC fx_stringify (multi-threaded polygon drawing is not bit-reproducible)."""
    lk = ("two_points", "three_points", "four_points", "all_points").index(link)
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_stringify", smoothness, levels, sampling_rate, lk, offset,
                                            threshold, opacity, normalization, fill_holes, *_rgb(background_color)),
                         no_smooth=True)


@itb_op("VectorPainting", "Vector Painting", GROUP, [
    _f("detail", 9, 0, 10, "Detail"),
], **GMIC, tags=["vector", "painting", "flat colors"])
def vector_painting(img, ctx, detail):
    """G'MIC fx_vector_painting."""
    return impl.gmic_itb(img, ctx, gmic_cmd("fx_vector_painting", detail))


@itb_op("Warhol", "Warhol", GROUP, [
    _i("x_tiles", 2, 1, 10, "X Tiles"),
    _i("y_tiles", 2, 1, 10, "Y Tiles"),
    _f("smoothness", 2, 0, 10, "Smoothness"),
    _f("color", 45, 0, 60, "Color"),
], **GMIC, alpha="full", geometry=True, tags=["warhol", "pop art", "tiles"])
def warhol(img, ctx, x_tiles, y_tiles, smoothness, color):
    """G'MIC warhol (tile mosaic; the output size is rounded to a multiple of the tile grid)."""
    return impl.gmic_itb(img, ctx, gmic_cmd("warhol", x_tiles, y_tiles, smoothness, color), mode="geometry")


# ---------------------------------------------------------------------------
# Native (Trickle / aire / jhlabs / GPUImage) filters
# ---------------------------------------------------------------------------

@itb_op("EnhancedOil", "Enhanced Oil", GROUP, [
    Int("strength", 10, min=1, max=25, unit="px", title="Strength", description="Window radius"),
], upstream="Trickle", alpha="full", tags=["oil", "painting", "mode filter"])
def enhanced_oil(img, ctx, strength):
    """Trickle OilFilter: per-channel most frequent value in a 2r x 2r window (opaque result)."""
    return impl.trickle_oil(img, ctx.ipx(strength))


@itb_op("LowPoly", "Low Poly", GROUP, [
    Int("strength", 2000, min=50, max=30000, title="Strength", description="Number of sampled edge points"),
    Bool("fill", True, title="Fill", description="Fill triangles (otherwise thin outlines on transparency)"),
], upstream="Trickle", alpha="full", tags=["low poly", "triangles", "delaunay", "geometric"])
def low_poly(img, ctx, strength, fill):
    """Trickle lowPoly: Sobel edge points -> random subset -> Delaunay triangles coloured at their centroid."""
    out = impl.lowpoly(img, float(strength), bool(fill), ctx.rng("lowpoly"))
    return impl.finish_rgba(out, img.shape[2] == 4)


@itb_op("Oil", "Oil", GROUP, [
    Int("radius", 4, min=1, max=20, unit="px", title="Radius"),
    Float("strength", 1.0, min=0, max=4, step=0.1, title="Strength", description="Intensity levels multiplier"),
], upstream="aire", alpha="full", tags=["oil", "painting"])
def oil(img, ctx, radius, strength):
    """aire oil: brightest pixel along the (x+i, y+i) diagonal (upstream window bug reproduced)."""
    return impl.aire_oil(img, ctx.ipx(radius), strength)


@itb_op("Pointillize", "Pointillize", GROUP, [
    Float("border_thickness", 0.4, min=0, max=5, step=0.01, title="Border thickness"),
    Float("scale", 32, min=1, max=300, step=0.01, unit="px", title="Scale"),
    Float("randomness", 5, min=0, max=10, step=0.01, title="Randomness"),
    Int("shape", 0, min=0, max=4, title="Shape",
        description="Grid: 0 random, 1 square, 2 hexagonal, 3 octagonal, 4 triangular"),
    Float("turbulence", 1, min=0, max=1, step=0.01, title="Turbulence"),
    Int("angle", 0, min=0, max=360, title="Angle", description="Degrees"),
    Float("stretch", 1, min=1, max=6, step=0.01, title="Stretch"),
    Float("amount", 1, min=0, max=1, step=0.01, title="Amount"),
    Color("border_color", "#000000", title="Border color"),
], upstream="jhlabs", alpha="full", tags=["pointillism", "dots", "cells", "voronoi"])
def pointillize(img, ctx, border_thickness, scale, randomness, shape, turbulence, angle, stretch, amount,
                border_color):
    """jhlabs PointillizeFilter (ImageToolboxLibs Rust port): cellular noise, dot = nearest feature point colour.

    ``turbulence`` and ``amount`` have no effect on this filter upstream (the
    turbulence loop only runs for values > 1, ``amount`` is unused).
    """
    out = impl.pointillize(img, max(1e-3, ctx.px(scale)), stretch, math.radians(angle), shape, randomness,
                           border_thickness, border_color)
    return impl.finish_rgba(out, img.shape[2] == 4)


@itb_op("PopArt", "Pop Art", GROUP, [
    Float("strength", 1.0, min=0, max=1, step=0.01, title="Strength"),
    Color("color", "#FF0000", title="Color"),
    Choice("overlay_mode", "multiply", choices=("multiply", "color_burn", "soft_light", "hsl_color", "hsl_hue",
                                                "difference"), title="Overlay Mode"),
], upstream="Trickle", alpha="full", tags=["pop art", "warhol", "quadrants"])
def pop_art(img, ctx, strength, color, overlay_mode):
    """Trickle popArt: 2x2 half-size copies, each blended with a square-harmony colour (cairo operators)."""
    return impl.finish_rgba(impl.pop_art(img, color, strength, overlay_mode), img.shape[2] == 4)


@itb_op("RubberStamp", "Rubber Stamp", GROUP, [
    Float("threshold", 0.3, min=0, max=1, step=0.01, title="Threshold"),
    Float("brush_softness", 0.1, min=0, max=1, step=0.01, title="Brush softness"),
    Float("radius", 0.05, min=0, max=2, step=0.01, title="Radius", description="Relative to max(width, height)/64"),
    Color("first_color", "#3FA08F", title="First color"),
    Color("second_color", "#003027", title="Second color"),
], upstream="jhlabs", alpha="full", tags=["stamp", "threshold", "two tone"])
def rubber_stamp(img, ctx, threshold, brush_softness, radius, first_color, second_color):
    """jhlabs StampFilter: Gaussian blur, then a smooth threshold between the two colours."""
    h, w = img.shape[:2]
    out = impl.rubber_stamp(img, radius * (max(w, h) / 64.0), threshold, brush_softness, first_color, second_color)
    return impl.finish_rgba(out, img.shape[2] == 4)


@itb_op("SandPainting", "Sand Painting", GROUP, [
    Int("strength", 5000, min=50, max=75000, title="Strength", description="Number of sampled edge points"),
    Int("threshold", 50, min=30, max=90, title="Threshold", description="Sobel magnitude threshold (0..255 scale)"),
    Color("background_color", "#000000", title="Background color"),
], upstream="Trickle", alpha="full", tags=["sand", "dots", "edges"])
def sand_painting(img, ctx, strength, threshold, background_color):
    """Trickle sandPainting + drawColorBehind."""
    out = impl.sand_painting(img, float(strength), threshold, background_color, ctx.rng("sand"))
    return impl.finish_rgba(out, img.shape[2] == 4)


@itb_op("Sketch", "Sketch", GROUP, [
    Int("kernel_size", 5, min=3, max=9, unit="px", title="Kernel size", description="Dilation kernel size"),
], upstream="aire", tags=["sketch", "pencil", "remove shadows"])
def sketch(img, ctx, kernel_size):
    """aire removeShadows: dilate, 21-px median, 255-|median-src|, stretch (per channel)."""
    return impl.aire_remove_shadows(img, ctx.ipx(kernel_size), ctx.ipx(21))


@itb_op("SmoothToon", "Smooth toon", GROUP, [
    Float("blur_size", 0.5, min=0, max=100, step=0.01, unit="px", title="Blur size"),
    Float("threshold", 0.2, min=0, max=5, step=0.01, title="Threshold"),
    Float("quantization_levels", 10, min=0, max=100, step=0.01, title="Quantization levels"),
], upstream="GPUImage", tags=["toon", "cartoon", "posterize", "edges"])
def smooth_toon(img, ctx, blur_size, threshold, quantization_levels):
    """GPUImageSmoothToonFilter: 9-tap Gaussian blur followed by the Toon filter."""
    return impl.gpu_smooth_toon(img, ctx.px(blur_size), threshold, quantization_levels)


@itb_op("Toon", "Toon", GROUP, [
    Float("threshold", 0.2, min=0, max=5, step=0.01, title="Threshold"),
    Float("quantization_levels", 10, min=0, max=100, step=0.01, title="Quantization levels"),
], upstream="GPUImage", tags=["toon", "cartoon", "posterize", "edges"])
def toon(img, ctx, threshold, quantization_levels):
    """GPUImageToonFilter: Sobel edges (red channel) drawn black over a posterized image."""
    return impl.gpu_toon(img, threshold, quantization_levels)
