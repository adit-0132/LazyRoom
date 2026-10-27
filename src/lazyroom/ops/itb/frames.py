"""ImageToolbox "Frames" group filters."""
from __future__ import annotations

from ...core.params import Bool, Choice, Color, Float, Int
from ._common import gmic_cmd, itb_op
from . import _frames_impl as Fr
from ._geometry_impl import gmic_apply, gmic_color

GROUP = "frames"

_ORIENT = ["normal", "mirror_x", "mirror_y", "mirror_xy"]
_ORIENT_LABELS = {"normal": "Normal", "mirror_x": "Mirror X", "mirror_y": "Mirror Y", "mirror_xy": "Mirror XY"}


# ---------------------------------------------------------------------------
# ImageToolbox (Android Canvas) frames
# ---------------------------------------------------------------------------

@itb_op("BorderFrame", "Border Frame", GROUP, [
    Int("horizontal_border_thickness", 20, min=0, max=500, title="Horizontal Border Thickness", unit="px"),
    Int("vertical_border_thickness", 40, min=0, max=500, title="Vertical Border Thickness", unit="px"),
    Color("border_color", "#FFFFFF", title="Border color"),
], upstream="ImageToolbox", alpha="full", geometry=True, tags=["border", "frame", "padding", "canvas"])
def border_frame(img, ctx, horizontal_border_thickness, vertical_border_thickness, border_color):
    """Adds a solid border of the given thickness on each side (canvas grows by 2x thickness)."""
    return Fr.border_frame(img, ctx.ipx(horizontal_border_thickness, minimum=0),
                           ctx.ipx(vertical_border_thickness, minimum=0), border_color)


@itb_op("DropShadow", "Drop Shadow", GROUP, [
    Float("blur_radius", 18.0, min=0, max=200, step=0.01, title="Blur radius", unit="px"),
    Float("offset_x", 0.0, min=-150, max=150, step=0.01, title="Offset X", unit="px"),
    Float("offset_y", 8.0, min=-150, max=150, step=0.01, title="Offset Y", unit="px"),
    Color("shadow_color", "#000000", title="Shadow Color"),
], upstream="ImageToolbox", alpha="full", geometry=True, tags=["shadow", "drop shadow", "frame", "depth"])
def drop_shadow(img, ctx, blur_radius, offset_x, offset_y, shadow_color):
    """Blurred, tinted copy of the alpha drawn under the image; the canvas grows to fit it."""
    return Fr.drop_shadow(img, ctx.px(blur_radius), ctx.px(offset_x), ctx.px(offset_y), shadow_color)


_FADE_SIDES = ["start", "end", "top", "bottom"]


@itb_op("SideFade", "Side Fade", GROUP, [
    Float("strength", 0.5, min=0, max=1, step=0.01, title="Strength", description="Fade length as a fraction of the side"),
    Choice("side", "end", choices=_FADE_SIDES, title="Side",
           labels={"start": "Start", "end": "End", "top": "Top", "bottom": "Bottom"}),
], upstream="ImageToolbox", alpha="full", tags=["fade", "transparent", "gradient", "edge"])
def side_fade(img, ctx, strength, side):
    """Fades one side of the image to transparent with a linear alpha gradient."""
    return Fr.side_fade(img, side, strength)


@itb_op("TornEdge", "Torn Edge", GROUP, [
    Int("tooth_height", 16, min=1, max=96, title="Tooth Height", unit="px"),
    Int("horizontal_tooth_range", 22, min=2, max=128, title="Horizontal Tooth Range", unit="px"),
    Int("vertical_tooth_range", 22, min=2, max=128, title="Vertical Tooth Range", unit="px"),
    Bool("top_edge", True, title="Top Edge"),
    Bool("right_edge", True, title="Right Edge"),
    Bool("bottom_edge", True, title="Bottom Edge"),
    Bool("left_edge", True, title="Left Edge"),
], upstream="ImageToolbox", alpha="full", tags=["torn", "paper", "edge", "frame"])
def torn_edge(img, ctx, tooth_height, horizontal_tooth_range, vertical_tooth_range, top_edge, right_edge, bottom_edge,
              left_edge):
    """Clears random jagged teeth along the chosen edges (java.util.Random seeded by size and params)."""
    h, w = img.shape[:2]
    paths = Fr.torn_paths(w, h, ctx.ipx(tooth_height), ctx.ipx(horizontal_tooth_range, minimum=2),
                          ctx.ipx(vertical_tooth_range, minimum=2), top_edge, right_edge, bottom_edge, left_edge,
                          (tooth_height, horizontal_tooth_range, vertical_tooth_range, bool(top_edge),
                           bool(right_edge), bool(bottom_edge), bool(left_edge)))
    return Fr.clear_paths(img, paths)


# ---------------------------------------------------------------------------
# G'MIC frames (ImageToolboxLibs gmic FrameFilters.kt / CreativeFilters.kt)
# ---------------------------------------------------------------------------

@itb_op("BlurredFrame", "Blurred Frame", GROUP, [
    Float("horizontal_size", 30.0, min=0, max=100, step=0.01, title="Horizontal Size", unit="%"),
    Float("vertical_size", 30.0, min=0, max=100, step=0.01, title="Vertical Size", unit="%"),
    Float("crop", 0.0, min=0, max=100, step=0.01, title="Crop"),
    Float("blur", 5.0, min=0, max=10, step=0.01, title="Blur"),
    Float("roundness", 0.0, min=0, max=1, step=0.01, title="Roundness"),
    Bool("color_balance", False, title="Color balance"),
    Color("balance_color", "#808080", title="Balance Color"),
    Choice("normalization", "none", choices=["none", "stretch", "equalize"], title="Normalization"),
    Float("outline_size", 5.0, min=0, max=50, step=0.01, title="Outline Size"),
    Color("outline_color", "#FFFFFF", title="Outline Color"),
    Float("x_shadow", 2.0, min=-10, max=10, step=0.01, title="X Shadow"),
    Float("y_shadow", 2.0, min=-10, max=10, step=0.01, title="Y Shadow"),
    Float("softness", 1.0, min=0, max=5, step=0.01, title="Softness"),
    Float("contrast", 0.0, min=0, max=100, step=0.01, title="Contrast"),
    Float("x_centering", 0.5, min=0, max=1, step=0.01, title="X Centering"),
    Float("y_centering", 0.5, min=0, max=1, step=0.01, title="Y Centering"),
    Float("angle", 0.0, min=-180, max=180, step=0.01, title="Angle", unit="deg"),
], upstream="G'MIC", requires=["gmic"], alpha="full", geometry=True, tags=["frame", "blur", "border", "gmic"])
def blurred_frame(img, ctx, horizontal_size, vertical_size, crop, blur, roundness, color_balance, balance_color,
                  normalization, outline_size, outline_color, x_shadow, y_shadow, softness, contrast, x_centering,
                  y_centering, angle):
    cmd = gmic_cmd("fx_frame_blur", horizontal_size, vertical_size, crop, blur, roundness, bool(color_balance),
                   *gmic_color(balance_color), ["none", "stretch", "equalize"].index(normalization), outline_size,
                   *gmic_color(outline_color), x_shadow, y_shadow, softness, contrast, x_centering, y_centering, angle)
    return gmic_apply(img, cmd, ctx, process_alpha=True)


@itb_op("CubeFrame", "Cube Frame", GROUP, [
    Float("depth", 3.0, min=0, max=30, step=0.01, title="Depth"),
    Float("center_x", 50.0, min=0, max=100, step=0.01, title="Center X", unit="%"),
    Float("center_y", 50.0, min=0, max=100, step=0.01, title="Center Y", unit="%"),
    Choice("left_orientation", "normal", choices=_ORIENT, labels=_ORIENT_LABELS, title="Left Orientation"),
    Choice("right_orientation", "normal", choices=_ORIENT, labels=_ORIENT_LABELS, title="Right Orientation"),
    Choice("upper_orientation", "normal", choices=_ORIENT, labels=_ORIENT_LABELS, title="Upper Orientation"),
    Choice("lower_orientation", "normal", choices=_ORIENT, labels=_ORIENT_LABELS, title="Lower Orientation"),
], upstream="G'MIC", requires=["gmic"], alpha="full", tags=["frame", "cube", "3d", "gmic"])
def cube_frame(img, ctx, depth, center_x, center_y, left_orientation, right_orientation, upper_orientation,
               lower_orientation):
    cmd = gmic_cmd("fx_frame_cube", depth, center_x, center_y, _ORIENT.index(left_orientation),
                   _ORIENT.index(right_orientation), _ORIENT.index(upper_orientation), _ORIENT.index(lower_orientation))
    return gmic_apply(img, cmd, ctx, process_alpha=True)


@itb_op("FuzzyFrame", "Fuzzy Frame", GROUP, [
    Float("horizontal_size", 5.0, min=0, max=100, step=0.01, title="Horizontal Size", unit="%"),
    Float("vertical_size", 5.0, min=0, max=100, step=0.01, title="Vertical Size", unit="%"),
    Float("fuzziness", 10.0, min=0, max=40, step=0.01, title="Fuzziness"),
    Float("smoothness", 1.0, min=0, max=5, step=0.01, title="Smoothness"),
    Color("color", "#FFFFFFFF", title="Color"),
], upstream="G'MIC", requires=["gmic"], alpha="full", tags=["frame", "fuzzy", "border", "gmic"])
def fuzzy_frame(img, ctx, horizontal_size, vertical_size, fuzziness, smoothness, color):
    cmd = gmic_cmd("fx_frame_fuzzy", horizontal_size, vertical_size, fuzziness, smoothness,
                   *gmic_color(color, alpha=True))
    return gmic_apply(img, cmd, ctx, process_alpha=True)


@itb_op("MirrorFrame", "Mirror Frame", GROUP, [
    Float("horizontal_size", 10.0, min=0, max=100, step=0.01, title="Horizontal Size", unit="%"),
    Float("vertical_size", 10.0, min=0, max=100, step=0.01, title="Vertical Size", unit="%"),
    Float("horizontal_alignment", 50.0, min=0, max=100, step=0.01, title="Horizontal Alignment"),
    Float("vertical_alignment", 50.0, min=0, max=100, step=0.01, title="Vertical Alignment"),
    Float("left_dilation", 0.0, min=-5, max=5, step=0.01, title="Left Dilation"),
    Float("right_dilation", 0.0, min=-5, max=5, step=0.01, title="Right Dilation"),
    Float("upper_dilation", 0.0, min=-5, max=5, step=0.01, title="Upper Dilation"),
    Float("lower_dilation", 0.0, min=-5, max=5, step=0.01, title="Lower Dilation"),
], upstream="G'MIC", requires=["gmic"], alpha="full", geometry=True, tags=["frame", "mirror", "border", "gmic"])
def mirror_frame(img, ctx, horizontal_size, vertical_size, horizontal_alignment, vertical_alignment, left_dilation,
                 right_dilation, upper_dilation, lower_dilation):
    cmd = gmic_cmd("fx_frame_mirror", horizontal_size, vertical_size, horizontal_alignment, vertical_alignment,
                   left_dilation, right_dilation, upper_dilation, lower_dilation)
    return gmic_apply(img, cmd, ctx, process_alpha=True)


@itb_op("PaintingFrame", "Painting Frame", GROUP, [
    Float("size", 10.0, min=0, max=100, step=0.01, title="Size"),
    Float("contrast", 0.4, min=0, max=1, step=0.01, title="Contrast"),
    Float("smoothness", 6.0, min=0, max=30, step=0.01, title="Smoothness"),
    Color("color", "#E1C878", title="Color"),
    Float("vignette_size", 2.0, min=0, max=50, step=0.01, title="Vignette Size"),
    Float("vignette_contrast", 400.0, min=0, max=1000, step=0.01, title="Vignette Contrast"),
    Float("defects_contrast", 50.0, min=0, max=512, step=0.01, title="Defects Contrast"),
    Float("density", 10.0, min=0, max=100, step=0.01, title="Density"),
    Float("defects_size", 1.0, min=0, max=10, step=0.01, title="Defects Size"),
    Float("defects_smoothness", 0.5, min=0, max=20, step=0.01, title="Defects Smoothness"),
    Int("serial_number", 123456, min=0, max=1000000, title="Serial Number", description="Random seed of the frame"),
], upstream="G'MIC", requires=["gmic"], alpha="full", geometry=True, tags=["frame", "painting", "wood", "gmic"])
def painting_frame(img, ctx, size, contrast, smoothness, color, vignette_size, vignette_contrast, defects_contrast,
                   density, defects_size, defects_smoothness, serial_number):
    cmd = gmic_cmd("fx_frame_painting", size, contrast, smoothness, *gmic_color(color), vignette_size,
                   vignette_contrast, defects_contrast, density, defects_size, defects_smoothness, serial_number, False)
    return gmic_apply(img, cmd, ctx, process_alpha=True)


@itb_op("PolaroidFrame", "Polaroid Frame", GROUP, [
    Int("frame_size", 10, min=0, max=400, title="Frame Size"),
    Int("bottom_size", 20, min=0, max=400, title="Bottom Size"),
    Float("x_shadow", 2.0, min=-20, max=20, step=0.01, title="X Shadow"),
    Float("y_shadow", 3.0, min=-20, max=20, step=0.01, title="Y Shadow"),
    Float("smoothness", 3.0, min=0, max=5, step=0.01, title="Smoothness"),
    Float("x_curvature", 0.0, min=0, max=1, step=0.01, title="X Curvature"),
    Float("y_curvature", 0.0, min=0, max=1, step=0.01, title="Y Curvature"),
    Float("angle", 8.0, min=-180, max=180, step=0.01, title="Angle", unit="deg"),
    Float("vignette_strength", 50.0, min=0, max=255, step=0.01, title="Vignette Strength"),
    Float("vignette_min_radius", 70.0, min=0, max=100, step=0.01, title="Vignette Min Radius"),
    Float("vignette_max_radius", 95.0, min=0, max=100, step=0.01, title="Vignette Max Radius"),
], upstream="G'MIC", requires=["gmic"], alpha="full", geometry=True, tags=["frame", "polaroid", "photo", "gmic"])
def polaroid_frame(img, ctx, frame_size, bottom_size, x_shadow, y_shadow, smoothness, x_curvature, y_curvature, angle,
                   vignette_strength, vignette_min_radius, vignette_max_radius):
    cmd = gmic_cmd("fx_polaroid", frame_size, bottom_size, x_shadow, y_shadow, smoothness, x_curvature, y_curvature,
                   angle, vignette_strength, vignette_min_radius, vignette_max_radius)
    return gmic_apply(img, cmd, ctx, process_alpha=False)


@itb_op("RoundedFrame", "Rounded Frame", GROUP, [
    Float("x_size", 10.0, min=0, max=100, step=0.01, title="X Size", unit="%"),
    Float("y_size", 10.0, min=0, max=100, step=0.01, title="Y Size", unit="%"),
    Float("radius", 20.0, min=0, max=100, step=0.01, title="Radius", unit="%"),
    Float("smoothness", 0.1, min=0, max=15, step=0.01, title="Smoothness"),
    Color("color", "#000000FF", title="Color"),
    Bool("antialias", True, title="Antialias"),
], upstream="G'MIC", requires=["gmic"], alpha="full", tags=["frame", "rounded", "border", "gmic"])
def rounded_frame(img, ctx, x_size, y_size, radius, smoothness, color, antialias):
    cmd = gmic_cmd("fx_frame_round", x_size, y_size, radius, smoothness, *gmic_color(color, alpha=True),
                   bool(antialias))
    return gmic_apply(img, cmd, ctx, process_alpha=True)


@itb_op("SmoothFrame", "Smooth Frame", GROUP, [
    Int("width", 5, min=0, max=100, title="Width", unit="%"),
    Int("height", 5, min=0, max=100, title="Height", unit="%"),
    Float("roundness", 0.25, min=0, max=1, step=0.01, title="Roundness"),
], upstream="G'MIC", requires=["gmic"], alpha="full", geometry=True, tags=["frame", "smooth", "inpaint", "gmic"])
def smooth_frame(img, ctx, width, height, roundness):
    cmd = gmic_cmd("fx_frame_smooth", width, height, roundness)
    return gmic_apply(img, cmd, ctx, process_alpha=False)
