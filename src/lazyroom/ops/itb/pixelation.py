"""ImageToolbox "Pixelation" group filters.

The canvas-drawn pixelation family (Pixelation, Circle/Diamond/Cross/... Pixelization)
ports ImageToolbox's ``PixelationTool`` + ``PixelationCommands`` layer stacks.  The
G'MIC-based filters run the exact ImageToolboxLibs command strings through the G'MIC
CLI (with ImageToolboxLibs' Preserve/Process alpha semantics and a deterministic
``srand`` prefix).
"""
from __future__ import annotations

import math

import cv2

from ...core.errors import LazyRoomError
from ...core.params import Bool, Choice, Color, Float, Int, Text
from . import _pixelation_impl as impl
from ._common import gmic_cmd, itb_op, snake

GROUP = "pixelation"
GMIC = dict(upstream="G'MIC", requires=["gmic"])

GMIC_CHANNELS = [
    "All", "Rgba", "Rgb", "Red", "Green", "Blue", "Alpha", "LinearRgb", "LinearRed", "LinearGreen", "LinearBlue",
    "YCbCr", "Luminance", "Chrominance", "BlueChrominance", "RedChrominance", "GreenChrominance", "Lab",
    "LabLightness", "LabChrominance", "LabA", "LabB", "Lch", "LchChrominance", "LchChroma", "LchHue", "Hsv",
    "HsvHue", "HsvSaturation", "HsvValue", "Hsi", "HsiIntensity", "Hsl", "HslLightness", "Cmyk", "Cmy", "Cyan",
    "Magenta", "Yellow", "Key", "Yiq", "YiqLuma", "YiqChrominance", "Ryb", "RybRed", "RybYellow", "RybBlue",
]


def _enum(name: str, upstream: list[str], default: str, title: str, description: str = "") -> Choice:
    """Choice mirroring an ImageToolboxLibs ``GmicArgument`` enum (value == declaration index)."""
    choices = [snake(u) for u in upstream]
    return Choice(name, snake(default), choices=choices, title=title, description=description)


def _ev(param_value: str, upstream: list[str]) -> int:
    return [snake(u) for u in upstream].index(param_value)


def _ipx(ctx, v: float, lo: int, hi: int) -> int:
    """Integer pixel parameter scaled for preview proxies and kept in the G'MIC range."""
    return int(min(max(round(v * ctx.scale), lo), hi))


def _fpx(ctx, v: float, lo: float, hi: float) -> float:
    return float(min(max(v * ctx.scale, lo), hi))


def _pixel_size(default: float, lo: float, title: str = "Pixel Size") -> Float:
    return Float("pixel_size", default, min=lo, max=200, step=0.01, unit="px", title=title,
                 description="Cell size of the pixelation grid")


def _tool(img, ctx, kind: str, value: float):
    return impl.prem_to_image(impl.pixelate(img, kind, value, ctx.scale), img.shape[2] == 4)


# ---------------------------------------------------------------------------
# ImageToolbox PixelationTool family
# ---------------------------------------------------------------------------

_TOOL_FILTERS = [
    # name, title, kind, default, min, tags
    ("Pixelation", "Pixelation", "square", 25.0, 5.0, ["pixelate", "mosaic", "blocks"]),
    ("EnhancedPixelation", "Enhanced Pixelation", "enhanced_square", 48.0, 10.0, ["pixelate", "mosaic", "diamonds"]),
    ("CirclePixelation", "Circle Pixelation", "circle", 24.0, 5.0, ["pixelate", "circles", "dots"]),
    ("EnhancedCirclePixelation", "Enhanced Circle Pixelation", "enhanced_circle", 32.0, 15.0,
     ["pixelate", "circles", "rings"]),
    ("DiamondPixelation", "Diamond Pixelation", "diamond", 24.0, 10.0, ["pixelate", "diamonds"]),
    ("EnhancedDiamondPixelation", "Enhanced Diamond Pixelation", "enhanced_diamond", 48.0, 20.0,
     ["pixelate", "diamonds"]),
    ("SimpleWeavePixelation", "Simple Weave Pixelization", "simple_weave", 25.0, 5.0, ["pixelate", "weave"]),
    ("CrossPixelization", "Cross Pixelization", "cross", 25.0, 5.0, ["pixelate", "cross"]),
    ("MicroMacroPixelization", "Micro Macro Pixelization", "micro_macro", 25.0, 5.0, ["pixelate", "blocks"]),
    ("NucleusPixelization", "Nucleus Pixelization", "nucleus", 25.0, 5.0, ["pixelate", "circles"]),
    ("OrbitalPixelization", "Orbital Pixelization", "orbital", 25.0, 5.0, ["pixelate", "circles"]),
    ("PulseGridPixelization", "Pulse Grid Pixelization", "pulse_grid", 25.0, 5.0, ["pixelate", "grid"]),
    ("RadialWeavePixelization", "Radial Weave Pixelization", "radial_weave", 25.0, 5.0, ["pixelate", "weave"]),
    ("StaggeredPixelization", "Staggered Pixelization", "staggered", 25.0, 5.0, ["pixelate", "blocks"]),
    ("VortexPixelization", "Vortex Pixelization", "vortex", 25.0, 5.0, ["pixelate", "circles"]),
]


def _register_tool(name: str, title: str, kind: str, default: float, lo: float, tags: list[str]) -> None:
    def fn(img, ctx, pixel_size):
        return _tool(img, ctx, kind, pixel_size)

    fn.__name__ = snake(name)
    fn.__doc__ = (f"ImageToolbox PixelationTool with the ``PixelationCommands.{kind}`` layer stack "
                  "(anti-aliased squares / circles / diamonds sampled at the cell centres; gaps stay transparent).")
    itb_op(name, title, GROUP, [_pixel_size(default, lo)], upstream="ImageToolbox", alpha="full",
           tags=tags)(fn)


for _args in _TOOL_FILTERS:
    _register_tool(*_args)


@itb_op("StrokePixelation", "Stroke Pixelation", GROUP, [
    _pixel_size(20.0, 5.0),
    Color("background_color", "#000000", title="Background color", description="Colour drawn behind the dots"),
], upstream="ImageToolbox", alpha="full", tags=["pixelate", "dots", "stroke"])
def stroke_pixelation(img, ctx, pixel_size, background_color):
    """PixelationTool ``stroke`` layers, then Trickle ``drawColorBehind``."""
    prem = impl.pixelate(img, "stroke", pixel_size, ctx.scale)
    return impl.prem_to_image(prem, img.shape[2] == 4, behind=background_color)


# ---------------------------------------------------------------------------
# Native (aire / Trickle / jhlabs / ImageToolbox) filters
# ---------------------------------------------------------------------------

ASCII_GRADIENTS = {
    "normal": ".:-=+*#%@", "normal2": " `.,-~+<>o=*%X@", "arrows": "↖←↙↓↘→↗↑", "old": "░▒▓█",
    "extended_high": ".:-~=+*^><)(][}{#%@", "minimal": ".-+#", "math": "π√∞≈≠=÷×-+", "numerical": "7132546980",
}


@itb_op("Ascii", "ASCII", GROUP, [
    Text("gradient", "░▒▓█", max_length=256, title="Gradient",
         description="Characters ordered from bright to dark pixels (duplicates and whitespace are ignored). "
                     "Presets: " + ", ".join(f"{k} '{v}'" for k, v in ASCII_GRADIENTS.items())),
    Float("font_size", 10.0, min=1, max=100, step=0.01, unit="px", title="Font Size",
          description="Character cell size"),
    Text("font", "sans-serif", max_length=512, title="Font",
         description="Font family (system fonts) or path to a .ttf/.otf; missing glyphs fall back via fontconfig"),
    Color("background_color", "#000000", title="Background color"),
    Bool("gray_scale", False, title="Gray Scale", description="Grey characters whose opacity follows the luma"),
], upstream="ImageToolbox", alpha="full", tags=["ascii", "text", "characters", "art"])
def ascii(img, ctx, gradient, font_size, font, background_color, gray_scale):
    """ImageToolbox ``ASCIIConverter.convertToAsciiBitmap``: downscale to round(width / font size) columns,
    map each cell's luma to a gradient character and draw it in the cell colour."""
    seen: list[str] = []
    for ch in gradient:
        if not ch.isspace() and ch not in seen:
            seen.append(ch)
    prem = impl.ascii_art(img, gradient="".join(seen), font_size=ctx.px(font_size), font=font,
                          background=background_color, grayscale=gray_scale)
    return impl.prem_to_image(prem, img.shape[2] == 4)


@itb_op("Crystallize", "Crystallize", GROUP, [
    Float("amount", 1.0, min=0.01, max=2, step=0.01, title="Amount",
          description="Cell density (number of cells = 1% of the pixel count x amount)"),
    Color("stroke_color", "#00000000", title="Stroke Color", description="Outline of every cell fragment"),
], upstream="aire", alpha="full", tags=["crystallize", "voronoi", "cells", "mosaic"])
def crystallize(img, ctx, amount, stroke_color):
    """aire ``crystallize``: Voronoi (jc_voronoi) cells of random sites, each site/edge triangle's
    bounding box filled with the site colour."""
    h, w = img.shape[:2]
    # cell area is 100/amount full-resolution pixels -> keep it constant on preview proxies
    n = int(w * h * 0.01 * amount / max(ctx.scale, 1e-6) ** 2)
    out = impl.aire_crystallize(img, n, stroke_color, ctx.rng("crystallize"))
    return impl.finish_rgba(out, img.shape[2] == 4)


@itb_op("EqualizeHistogramPixelation", "Equalize Histogram Pixelation", GROUP, [
    Int("grid_size_x", 50, min=1, max=200, title="Grid Size X", description="Number of tiles horizontally"),
    Int("grid_size_y", 50, min=1, max=200, title="Grid Size Y", description="Number of tiles vertically"),
], upstream="aire", tags=["equalize", "histogram", "tiles", "pixelate"])
def equalize_histogram_pixelation(img, ctx, grid_size_x, grid_size_y):
    """aire ``equalizeHistSquares``: HSV value histogram equalisation (101 bins) inside each tile."""
    return impl.equalize_hist_squares(img, grid_size_x, grid_size_y)


@itb_op("PolkaDot", "Polka Dot", GROUP, [
    Int("radius", 10, min=1, max=40, unit="px", title="Radius"),
    Int("spacing", 8, min=1, max=40, unit="px", title="Spacing"),
    Color("background_color", "#000000", title="Background color"),
], upstream="Trickle", alpha="full", tags=["dots", "polka", "halftone", "pixelate"])
def polka_dot(img, ctx, radius, spacing, background_color):
    """Trickle ``polkaDot`` (+ ``drawColorBehind`` unless the background is fully transparent)."""
    prem = impl.polka_dot(img, ctx.ipx(radius), ctx.ipx(spacing))
    # a fully transparent background skips drawColorBehind, which is the same as compositing over nothing
    return impl.prem_to_image(prem, img.shape[2] == 4, behind=background_color)


@itb_op("Smear", "Smear", GROUP, [
    Int("angle", 45, min=0, max=360, title="Angle", description="Direction of the line smears (degrees)"),
    Float("density", 1.0, min=0, max=1, step=0.01, title="Density"),
    Float("mix", 1.0, min=0, max=1, step=0.01, title="Mix"),
    Int("distance", 100, min=0, max=200, unit="px", title="Distance"),
    Int("shape", 0, min=0, max=3, title="Shape", description="0 crosses, 1 lines, 2 circles, 3 squares"),
], upstream="jhlabs", alpha="full", tags=["smear", "paint", "strokes"])
def smear(img, ctx, angle, density, mix, distance, shape):
    """jhlabs SmearFilter (java.util.Random seeded with 567, as in ImageToolboxLibs)."""
    d = max(1, int(round(distance * ctx.scale)))
    return impl.smear(img, math.radians(angle), density, mix, d, shape)


@itb_op("VoronoiCrystallize", "Voronoi Crystallize", GROUP, [
    Float("border_thickness", 0.4, min=0, max=5, step=0.01, title="Border thickness"),
    Float("scale", 32.0, min=1, max=300, step=0.01, unit="px", title="Scale", description="Cell size"),
    Float("randomness", 5.0, min=0, max=10, step=0.01, title="Randomness"),
    Int("shape", 0, min=0, max=4, title="Shape",
        description="Grid type: 0 random, 1 square, 2 hexagonal, 3 octagonal, 4 triangular"),
    Float("turbulence", 1.0, min=0, max=1, step=0.01, title="Turbulence"),
    Int("angle", 0, min=0, max=360, title="Angle"),
    Float("stretch", 1.0, min=1, max=6, step=0.01, title="Stretch"),
    Float("amount", 1.0, min=0, max=1, step=0.01, title="Amount"),
    Color("border_color", "#000000", title="Border color"),
], upstream="jhlabs", alpha="full", tags=["crystallize", "voronoi", "cells", "stained glass"])
def voronoi_crystallize(img, ctx, border_thickness, scale, randomness, shape, turbulence, angle, stretch, amount,
                        border_color):
    """jhlabs CrystallizeFilter (CellularFilter F1/F2 distances, colour of the nearest feature point)."""
    out = impl.jh_crystallize(img, scale=scale * ctx.scale, stretch=stretch, angle_rad=math.radians(angle),
                              turbulence=turbulence, grid_type=shape, randomness=randomness,
                              edge=border_thickness, edge_color=border_color)
    return impl.finish_rgba(out, img.shape[2] == 4)


@itb_op("Weave", "Weave", GROUP, [
    Float("x_width", 16.0, min=0, max=100, step=0.01, unit="px", title="X Width"),
    Float("y_width", 16.0, min=0, max=100, step=0.01, unit="px", title="Y Width"),
    Float("x_gap", 6.0, min=0, max=100, step=0.01, unit="px", title="X Gap"),
    Float("y_gap", 6.0, min=0, max=100, step=0.01, unit="px", title="Y Gap"),
], upstream="jhlabs", alpha="full", tags=["weave", "threads", "fabric", "texture"])
def weave(img, ctx, x_width, y_width, x_gap, y_gap):
    """jhlabs WeaveFilter (image colours, shaded crossings; gaps are transparent)."""
    out = impl.weave(img, ctx.px(x_width), ctx.px(y_width), ctx.px(x_gap), ctx.px(y_gap))
    return impl.finish_rgba(out, img.shape[2] == 4)


# ---------------------------------------------------------------------------
# G'MIC filters (ImageToolboxLibs command builders)
# ---------------------------------------------------------------------------

def _rgba(color) -> tuple[int, int, int, int]:
    return impl.argb_ints(color)


def _with_power_of_two_padding(command: str) -> str:
    """ImageToolboxLibs ``withPowerOfTwoPadding``."""
    return ("_gmic_android_w={w} _gmic_android_h={h} "
            "resize {2^ceil(log2(max(1,w)))},{2^ceil(log2(max(1,h)))},100%,100%,0,3,0.5,0.5 "
            + command +
            " crop {int((w-$_gmic_android_w)/2)},{int((h-$_gmic_android_h)/2)},"
            "{int((w-$_gmic_android_w)/2+$_gmic_android_w-1)},{int((h-$_gmic_android_h)/2+$_gmic_android_h-1)}")


def _with_square_padding(command: str) -> str:
    """ImageToolboxLibs ``withSquarePadding``."""
    return ("_gmic_android_w={w} _gmic_android_h={h} _gmic_android_size={max(w,h)} "
            "resize $_gmic_android_size,$_gmic_android_size,100%,100%,0,3,0.5,0.5 "
            + command +
            " crop {int((w-$_gmic_android_w)/2)},{int((h-$_gmic_android_h)/2)},"
            "{int((w-$_gmic_android_w)/2+$_gmic_android_w-1)},{int((h-$_gmic_android_h)/2+$_gmic_android_h-1)}")


@itb_op("Blocks3D", "3D Blocks", GROUP, [
    Int("resolution", 32, min=1, max=128, title="Resolution", description="Number of blocks along the longest side"),
    Float("smoothness", 0.0, min=0, max=40, title="Smoothness"),
    Float("elevation", 4.0, min=-10, max=10, title="Elevation"),
    Float("size", 1.5, min=0, max=3, title="Size"),
    Float("angle", 30.0, min=0, max=360, title="Angle"),
    Float("tilt", 60.0, min=0, max=90, title="Tilt"),
    Float("fov", 45.0, min=1, max=90, title="Field of View"),
    Float("x_centering", 50.0, min=0, max=100, title="X Centering"),
    Float("y_centering", 50.0, min=0, max=100, title="Y Centering"),
    Float("x_light", 0.0, min=-100, max=100, title="X Light"),
    Float("y_light", -50.0, min=-100, max=100, title="Y Light"),
    Float("z_light", -100.0, min=-100, max=0, title="Z Light"),
    Float("specular_lightness", 0.5, min=0, max=1, title="Specular Lightness"),
    Float("specular_shininess", 0.7, min=0, max=3, title="Specular Shininess"),
    Bool("use_light", True, title="Use Light"),
    Bool("antialias", True, title="Antialias"),
    Color("outline_color", "#00000080", title="Outline Color"),
], **GMIC, alpha="full", geometry=True, tags=["3d", "blocks", "extrude", "pixelate"])
def blocks3_d(img, ctx, resolution, smoothness, elevation, size, angle, tilt, fov, x_centering, y_centering,
              x_light, y_light, z_light, specular_lightness, specular_shininess, use_light, antialias,
              outline_color):
    """G'MIC ``fx_blocks3d`` (square output of the longest side)."""
    cmd = gmic_cmd("fx_blocks3d", resolution, smoothness, elevation, size, angle, tilt, fov, x_centering,
                   y_centering, x_light, y_light, z_light, specular_lightness, specular_shininess, use_light,
                   antialias, *_rgba(outline_color))
    return impl.gmic_preserve(img, cmd, ctx)


@itb_op("ChessboardOverlay", "Chessboard Overlay", GROUP, [
    Int("first_size", 64, min=1, max=512, unit="px", title="First Size"),
    Int("second_size", 64, min=1, max=512, unit="px", title="Second Size"),
    Int("first_offset", 0, min=0, max=512, unit="px", title="First Offset"),
    Int("second_offset", 0, min=0, max=512, unit="px", title="Second Offset"),
    Float("angle", 0.0, min=0, max=180, title="Angle"),
    Float("opacity", 0.25, min=0, max=1, title="Opacity"),
    Color("first_color", "#000000", title="First color"),
    Color("second_color", "#FFFFFF", title="Second color"),
], **GMIC, alpha="full", tags=["chessboard", "checkerboard", "overlay", "pattern"])
def chessboard_overlay(img, ctx, first_size, second_size, first_offset, second_offset, angle, opacity,
                       first_color, second_color):
    """G'MIC ``fx_chessboard`` (alpha processed)."""
    cmd = gmic_cmd("fx_chessboard", _ipx(ctx, first_size, 1, 512), _ipx(ctx, second_size, 1, 512),
                   _ipx(ctx, first_offset, 0, 512), _ipx(ctx, second_offset, 0, 512), angle, opacity,
                   *_rgba(first_color), *_rgba(second_color))
    return impl.gmic_process(img, cmd, ctx)


@itb_op("CircleAbstraction", "Circle Abstraction", GROUP, [
    Int("number_of_colors", 8, min=2, max=16, title="Number Of Colors"),
    Int("density", 5, min=1, max=100, title="Density"),
    Float("opacity", 0.8, min=0, max=1, title="Opacity"),
    Float("smoothness", 0.0, min=0, max=4, title="Smoothness"),
    Bool("filled_circles", True, title="Filled Circles"),
    Bool("fill_transparent_holes", True, title="Fill Transparent Holes"),
    Bool("normalize_colors", True, title="Normalize Colors"),
], **GMIC, alpha="full", tags=["circles", "abstract", "skeleton"])
def circle_abstraction(img, ctx, number_of_colors, density, opacity, smoothness, filled_circles,
                       fill_transparent_holes, normalize_colors):
    """G'MIC ``fx_circle_abstraction`` (alpha processed).

    The G'MIC command itself fails when a colour layer's skeleton is tiny (it then tries to create an
    image of height < 1; same in ImageToolbox's G'MIC 4.0.2).  In that case the command is retried on a
    2x / 4x nearest-neighbour upscaled copy whose result is area-downscaled back.
    """
    cmd = gmic_cmd("fx_circle_abstraction", number_of_colors, density, opacity, smoothness, filled_circles,
                   fill_transparent_holes, normalize_colors)
    try:
        return impl.gmic_process(img, cmd, ctx)
    except LazyRoomError as exc:
        if "input options" not in str(exc) + str(getattr(exc, "details", "")):
            raise
        err = exc
    h, w = img.shape[:2]
    for k in (2, 4):
        up = cv2.resize(img, (w * k, h * k), interpolation=cv2.INTER_NEAREST).reshape(h * k, w * k, img.shape[2])
        try:
            out = impl.gmic_process(up, cmd, ctx)
        except LazyRoomError as exc:
            err = exc
            continue
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA).reshape(h, w, out.shape[2])
    raise err


DICE_MODELS = ["BlackDice", "WhiteDice", "ColoredNumbers", "ColoredSides"]


@itb_op("DiceMosaic", "Dice Mosaic", GROUP, [
    Float("resolution", 2.0, min=1, max=10, title="Resolution"),
    Int("size", 24, min=8, max=64, title="Size", description="Dice size"),
    _enum("color_model", DICE_MODELS, "WhiteDice", "Color Model"),
], **GMIC, alpha="full", geometry=True, tags=["dice", "mosaic", "pattern"])
def dice_mosaic(img, ctx, resolution, size, color_model):
    """G'MIC ``fx_dices`` (output size changes with the dice size)."""
    return impl.gmic_preserve(img, gmic_cmd("fx_dices", resolution, size, _ev(color_model, DICE_MODELS)), ctx)


TENSOR_MODES = ["Monochrome", "Grayscale", "Orientation", "Color"]


@itb_op("DiffusionTensors", "Diffusion Tensors", GROUP, [
    Float("resolution", 10.0, min=0, max=20, title="Resolution"),
    Float("size", 5.0, min=0, max=16, title="Size"),
    _enum("color_mode", TENSOR_MODES, "Color", "Color Mode"),
    Int("outline", 1, min=0, max=16, title="Outline"),
    Float("sharpness", 0.15, min=0, max=1, title="Sharpness"),
    Float("anisotropy", 1.0, min=0, max=1, title="Anisotropy"),
    Float("gradient_smoothness", 0.0, min=0, max=10, title="Gradient Smoothness"),
    Float("tensor_smoothness", 3.0, min=0, max=10, title="Tensor Smoothness"),
], **GMIC, tags=["tensors", "ellipses", "structure"])
def diffusion_tensors(img, ctx, resolution, size, color_mode, outline, sharpness, anisotropy, gradient_smoothness,
                      tensor_smoothness):
    """G'MIC ``fx_diffusiontensors``."""
    cmd = gmic_cmd("fx_diffusiontensors", resolution, size, _ev(color_mode, TENSOR_MODES), outline, sharpness,
                   anisotropy, gradient_smoothness, tensor_smoothness)
    return impl.gmic_preserve(img, cmd, ctx)


@itb_op("Ellipsionism", "Ellipsionism", GROUP, [
    Float("primary_radius", 20.0, min=1, max=100, unit="px", title="Primary Radius"),
    Float("second_radius", 10.0, min=1, max=100, unit="px", title="Second radius"),
    Float("smoothness", 0.5, min=0, max=10, title="Smoothness"),
    Float("opacity", 0.7, min=0, max=1, title="Opacity"),
    Float("outline", 3.0, min=1, max=3, title="Outline"),
    Float("density", 0.5, min=0.1, max=2, title="Density"),
], **GMIC, tags=["ellipses", "painting", "abstract"])
def ellipsionism(img, ctx, primary_radius, second_radius, smoothness, opacity, outline, density):
    """G'MIC ``fx_ellipsionism``."""
    cmd = gmic_cmd("fx_ellipsionism", _fpx(ctx, primary_radius, 1, 100), _fpx(ctx, second_radius, 1, 100),
                   smoothness, opacity, outline, density)
    return impl.gmic_preserve(img, cmd, ctx)


ARRAY_MIRROR = ["None", "XAxis", "YAxis", "BothAxes"]
ARRAY_SIZING = ["Shrink", "Expand", "Repeat"]


@itb_op("FadedArray", "Faded Array", GROUP, [
    Int("x_tiles", 2, min=1, max=10, title="X Tiles"),
    Int("y_tiles", 2, min=1, max=10, title="Y Tiles"),
    Float("x_offset", 0.0, min=0, max=100, title="X Offset"),
    Float("y_offset", 0.0, min=0, max=100, title="Y Offset"),
    Float("fade_start", 80.0, min=1, max=100, title="Fade Start"),
    Float("fade_end", 90.0, min=1, max=100, title="Fade End"),
    _enum("mirror", ARRAY_MIRROR, "None", "Mirror"),
    _enum("sizing", ARRAY_SIZING, "Shrink", "Sizing", "'repeat' enlarges the image"),
], **GMIC, alpha="full", geometry=True, tags=["array", "tiles", "repeat", "fade"])
def faded_array(img, ctx, x_tiles, y_tiles, x_offset, y_offset, fade_start, fade_end, mirror, sizing):
    """G'MIC ``fx_array_fade``."""
    cmd = gmic_cmd("fx_array_fade", x_tiles, y_tiles, x_offset, y_offset, fade_start, fade_end,
                   _ev(mirror, ARRAY_MIRROR), _ev(sizing, ARRAY_SIZING))
    return impl.gmic_preserve(img, cmd, ctx)


BLOCK_FLIP = ["None", "XAxis", "YAxis", "BothAxes"]
BLOCK_ROTATION = ["Minus90", "Zero", "Plus90"]


@itb_op("FlipAndRotateBlocks", "Flip And Rotate Blocks", GROUP, [
    Int("x_size", 4, min=1, max=128, unit="px", title="X Size"),
    Int("y_size", 4, min=1, max=128, unit="px", title="Y Size"),
    _enum("flip", BLOCK_FLIP, "BothAxes", "Flip"),
    _enum("rotation", BLOCK_ROTATION, "Zero", "Rotation"),
    _enum("channel", GMIC_CHANNELS, "All", "Channel"),
], **GMIC, tags=["blocks", "flip", "rotate", "glitch"])
def flip_and_rotate_blocks(img, ctx, x_size, y_size, flip, rotation, channel):
    """G'MIC ``fx_flip_blocks``."""
    cmd = gmic_cmd("fx_flip_blocks", _ipx(ctx, x_size, 1, 128), _ipx(ctx, y_size, 1, 128),
                   _ev(flip, BLOCK_FLIP), _ev(rotation, BLOCK_ROTATION), _ev(channel, GMIC_CHANNELS))
    return impl.gmic_preserve(img, cmd, ctx)


@itb_op("MakeSeamless", "Make Seamless", GROUP, [
    Float("equalize", 0.0, min=0, max=100, title="Equalize", description="Equalize light"),
], **GMIC, tags=["seamless", "tileable", "texture"])
def make_seamless(img, ctx, equalize):
    """G'MIC ``fx_make_seamless`` inside ImageToolboxLibs' power-of-two padding."""
    return impl.gmic_preserve(img, _with_power_of_two_padding(gmic_cmd("fx_make_seamless", equalize)), ctx)


@itb_op("MineralMosaic", "Mineral Mosaic", GROUP, [
    Float("density", 1.0, min=0, max=3, title="Density"),
    Float("area", 2.0, min=0, max=32, unit="px", title="Area"),
    Float("smoothness", 1.0, min=0, max=10, unit="px", title="Smoothness"),
    Float("strength", 100.0, min=0, max=255, title="Strength", description="Shade strength"),
    Float("angle", 0.0, min=0, max=360, title="Angle", description="Shade angle"),
], **GMIC, tags=["mosaic", "mineral", "stone"])
def mineral_mosaic(img, ctx, density, area, smoothness, strength, angle):
    """G'MIC ``fx_mineral_mosaic``."""
    cmd = gmic_cmd("fx_mineral_mosaic", density, _fpx(ctx, area, 0, 32), _fpx(ctx, smoothness, 0, 10),
                   strength, angle)
    return impl.gmic_preserve(img, cmd, ctx)


@itb_op("Ministeck", "Ministeck", GROUP, [
    Int("color", 8, min=2, max=24, title="Color", description="Number of colours"),
    Int("resolution", 64, min=16, max=256, title="Resolution", description="Pieces along the longest side"),
    Int("piece_size", 8, min=1, max=64, title="Piece Size", description="Output size of one piece (px)"),
    Int("piece_complexity", 2, min=1, max=10, title="Piece Complexity"),
    Float("amplitude", 100.0, min=0, max=256, title="Amplitude", description="Relief amplitude"),
    Float("relief_size", 0.3, min=0, max=1, title="Relief Size"),
    Bool("outline", False, title="Outline"),
], **GMIC, alpha="full", geometry=True, tags=["ministeck", "mosaic", "toy", "pieces"])
def ministeck(img, ctx, color, resolution, piece_size, piece_complexity, amplitude, relief_size, outline):
    """G'MIC ``fx_ministeck`` (output size = resolution x piece size)."""
    cmd = gmic_cmd("fx_ministeck", color, resolution, piece_size, piece_complexity, amplitude, relief_size, outline)
    return impl.gmic_preserve(img, cmd, ctx)


MIRROR_MODES = ["XAxis", "YAxis", "BothAxes", "DoubleBothAxes"]
ARRAY_INIT = ["Original", "MirrorX", "MirrorY", "Rotate90", "Rotate180", "Rotate270"]


@itb_op("MirroredArray", "Mirrored Array", GROUP, [
    Int("iterations", 1, min=1, max=10, title="Iterations"),
    Float("x_offset", 0.0, min=0, max=100, title="X Offset"),
    Float("y_offset", 0.0, min=0, max=100, title="Y Offset"),
    _enum("mode", MIRROR_MODES, "BothAxes", "Mode"),
    _enum("initialization", ARRAY_INIT, "Original", "Initialization"),
    Bool("expand_size", False, title="Expand Size"),
    Int("crop", 0, min=0, max=100, title="Crop"),
], **GMIC, alpha="full", geometry=True, tags=["array", "mirror", "kaleidoscope", "tiles"])
def mirrored_array(img, ctx, iterations, x_offset, y_offset, mode, initialization, expand_size, crop):
    """G'MIC ``fx_array_mirror``."""
    cmd = gmic_cmd("fx_array_mirror", iterations, x_offset, y_offset, _ev(mode, MIRROR_MODES),
                   _ev(initialization, ARRAY_INIT), expand_size, crop)
    return impl.gmic_preserve(img, cmd, ctx)


@itb_op("OldSchool8Bit", "Old School 8 Bit", GROUP, [
    Float("scale", 25.0, min=1, max=100, title="Scale"),
    Float("dithering", 800.0, min=0, max=10000, title="Dithering"),
    Int("levels", 16, min=2, max=256, title="Levels"),
], **GMIC, tags=["8bit", "retro", "pixel art", "dither"])
def old_school8_bit(img, ctx, scale, dithering, levels):
    """G'MIC ``fx_8bits``."""
    return impl.gmic_preserve(img, gmic_cmd("fx_8bits", scale, dithering, levels), ctx)


@itb_op("Puzzle", "Puzzle", GROUP, [
    Int("x_tiles", 5, min=2, max=32, title="X Tiles"),
    Int("y_tiles", 5, min=2, max=32, title="Y Tiles"),
    Float("curvature", 0.5, min=0, max=1.5, title="Curvature"),
    Float("connector_centering", 0.0, min=0, max=1, title="Connector Centering"),
    Float("connector_variability", 0.0, min=0, max=2, title="Connector Variability"),
    Float("relief_smoothness", 0.3, min=0, max=3, title="Relief Smoothness"),
    Float("relief_contrast", 100.0, min=0, max=255, title="Relief Contrast"),
    Float("outline_smoothness", 0.2, min=0, max=3, title="Outline Smoothness"),
    Float("outline_contrast", 255.0, min=0, max=255, title="Outline Contrast"),
    Float("scale", 100.0, min=0, max=150, title="Scale"),
    Float("scale_variation", 0.0, min=0, max=100, title="Scale Variation"),
    Float("angle", 0.0, min=-180, max=180, title="Angle"),
    Float("variation", 0.0, min=0, max=180, title="Variation", description="Angle variation"),
    Bool("shuffle_pieces", False, title="Shuffle Pieces"),
    Bool("add_outline", False, title="Add Outline"),
], **GMIC, alpha="full", tags=["puzzle", "jigsaw", "pieces"])
def puzzle(img, ctx, x_tiles, y_tiles, curvature, connector_centering, connector_variability, relief_smoothness,
           relief_contrast, outline_smoothness, outline_contrast, scale, scale_variation, angle, variation,
           shuffle_pieces, add_outline):
    """G'MIC ``fx_puzzle`` (alpha processed; trailing 0 as in ImageToolboxLibs)."""
    cmd = gmic_cmd("fx_puzzle", x_tiles, y_tiles, curvature, connector_centering, connector_variability,
                   relief_smoothness, relief_contrast, outline_smoothness, outline_contrast, scale, scale_variation,
                   angle, variation, shuffle_pieces, add_outline, False)
    return impl.gmic_process(img, cmd, ctx)


QUADTREE_MODES = ["Squares", "Sierpinski", "EllipsePainting"]


@itb_op("Quadtree", "Quadtree", GROUP, [
    _enum("mode", QUADTREE_MODES, "Squares", "Mode"),
    Int("precision", 1024, min=2, max=4096, title="Precision"),
    Float("homogeneity", 0.5, min=0, max=2, title="Homogeneity"),
    Int("outline", 0, min=0, max=4, title="Outline"),
    Float("primary_radius", 3.0, min=0, max=5, title="Primary Radius"),
    Float("second_radius", 1.5, min=0, max=5, title="Second radius"),
    Float("anisotropy", 1.0, min=0, max=4, title="Anisotropy"),
    Bool("only_leafs", True, title="Only Leafs"),
], **GMIC, tags=["quadtree", "blocks", "adaptive", "pixelate"])
def quadtree(img, ctx, mode, precision, homogeneity, outline, primary_radius, second_radius, anisotropy,
             only_leafs):
    """G'MIC ``fx_quadtree``."""
    cmd = gmic_cmd("fx_quadtree", _ev(mode, QUADTREE_MODES), precision, homogeneity, outline, primary_radius,
                   second_radius, anisotropy, only_leafs)
    return impl.gmic_preserve(img, cmd, ctx)


@itb_op("RandomColorArray", "Random Color Array", GROUP, [
    Int("x_tiles", 5, min=1, max=20, title="X Tiles"),
    Int("y_tiles", 5, min=1, max=20, title="Y Tiles"),
    Float("opacity", 0.5, min=0, max=1, title="Opacity"),
], **GMIC, tags=["array", "tiles", "random", "color"])
def random_color_array(img, ctx, x_tiles, y_tiles, opacity):
    """G'MIC ``fx_array_color``."""
    return impl.gmic_preserve(img, gmic_cmd("fx_array_color", x_tiles, y_tiles, opacity), ctx)


@itb_op("RandomPatchArray", "Random Patch Array", GROUP, [
    Int("source_x_tiles", 5, min=1, max=128, title="Source X tiles"),
    Int("source_y_tiles", 5, min=1, max=128, title="Source Y tiles"),
    Int("destination_x_tiles", 7, min=1, max=128, title="Destination X tiles"),
    Int("destination_y_tiles", 7, min=1, max=128, title="Destination Y tiles"),
    Int("seed", 0, min=0, max=65535, title="Seed"),
], **GMIC, alpha="full", geometry=True, tags=["array", "tiles", "random", "patches"])
def random_patch_array(img, ctx, source_x_tiles, source_y_tiles, destination_x_tiles, destination_y_tiles, seed):
    """G'MIC ``fx_array_random`` (output size = destination / source tiles x input size).

    G'MIC fails when there are more source tiles than pixels; the source tile counts are capped at
    the image size in that case."""
    h, w = img.shape[:2]
    cmd = gmic_cmd("fx_array_random", min(source_x_tiles, w), min(source_y_tiles, h), destination_x_tiles,
                   destination_y_tiles, seed)
    return impl.gmic_preserve(img, cmd, ctx)


STRIPE_ORIENTATION = ["Horizontal", "Vertical"]
VALUE_ACTIONS = ["None", "Cut", "Normalize"]


@itb_op("RandomShadeStripes", "Random Shade Stripes", GROUP, [
    Float("frequency", 30.0, min=1, max=100, title="Frequency"),
    _enum("orientation", STRIPE_ORIENTATION, "Vertical", "Orientation"),
    Float("darkness", 0.8, min=0, max=3, title="Darkness"),
    Float("lightness", 1.3, min=0, max=3, title="Lightness"),
    _enum("channel", GMIC_CHANNELS, "All", "Channel"),
    _enum("value_action", VALUE_ACTIONS, "None", "Value Action"),
], **GMIC, tags=["stripes", "random", "shade"])
def random_shade_stripes(img, ctx, frequency, orientation, darkness, lightness, channel, value_action):
    """G'MIC ``fx_shade_stripes``."""
    cmd = gmic_cmd("fx_shade_stripes", frequency, _ev(orientation, STRIPE_ORIENTATION), darkness, lightness,
                   _ev(channel, GMIC_CHANNELS), _ev(value_action, VALUE_ACTIONS))
    return impl.gmic_preserve(img, cmd, ctx)


BLOCK_NORMS = ["L1", "L2"]


@itb_op("RebuildFromSimilarBlocks", "Rebuild From Similar Blocks", GROUP, [
    Float("block_size", 5.0, min=2, max=50, title="Block Size"),
    Float("regularization", 10.0, min=0, max=20, title="Regularization"),
    Float("luminance", 0.75, min=0, max=3, title="Luminance"),
    _enum("norm", BLOCK_NORMS, "L2", "Norm"),
], **GMIC, deterministic=False, tags=["blocks", "patches", "rebuild"])
def rebuild_from_similar_blocks(img, ctx, block_size, regularization, luminance, norm):
    """G'MIC ``fx_rebuild_from_similar_blocks``.

    Not deterministic: the command's block search is a forced-parallel G'MIC fill whose result depends
    on thread scheduling (stable only with ``OMP_NUM_THREADS=1``)."""
    cmd = gmic_cmd("fx_rebuild_from_similar_blocks", block_size, regularization, luminance, _ev(norm, BLOCK_NORMS))
    return impl.gmic_preserve(img, cmd, ctx)


SHAPEISM_SHAPES = ["Square", "Triangle", "Circle", "Diamond", "Hexagon", "Octagon", "Star", "Heart", "Custom"]
SHAPEISM_ANGLES = ["Zero", "ZeroOr180", "RightAngles", "Any"]


@itb_op("Shapeism", "Shapeism", GROUP, [
    _enum("shape", SHAPEISM_SHAPES, "Circle", "Shape"),
    Int("branches", 7, min=3, max=16, title="Branches"),
    Float("thickness", 0.38, min=0, max=1, title="Thickness"),
    Float("angle", 0.0, min=0, max=360, title="Angle"),
    Bool("antialias", True, title="Antialias"),
    Int("scale", 5, min=1, max=16, title="Scale", description="Number of scales"),
    Int("maximal_size", 32, min=1, max=256, unit="px", title="Maximal Size"),
    Int("minimal_size", 8, min=1, max=256, unit="px", title="Minimal Size"),
    _enum("allowed_angles", SHAPEISM_ANGLES, "Any", "Allowed Angles"),
    Int("spacing", 0, min=-5, max=5, title="Spacing"),
    Int("precision", 5, min=1, max=10, title="Precision"),
    Float("edges", 0.5, min=0, max=2, title="Edges"),
    Float("smoothness", 1.0, min=0, max=10, title="Smoothness"),
    Color("background_color", "#000000", title="Background color"),
], **GMIC, alpha="full", tags=["shapes", "abstract", "mosaic"])
def shapeism(img, ctx, shape, branches, thickness, angle, antialias, scale, maximal_size, minimal_size,
             allowed_angles, spacing, precision, edges, smoothness, background_color):
    """G'MIC ``fx_shapeism`` (alpha processed)."""
    cmd = gmic_cmd("fx_shapeism", _ev(shape, SHAPEISM_SHAPES), branches, thickness, angle, antialias, scale,
                   _ipx(ctx, maximal_size, 1, 256), _ipx(ctx, minimal_size, 1, 256),
                   _ev(allowed_angles, SHAPEISM_ANGLES), spacing, precision, edges, smoothness,
                   *_rgba(background_color))
    return impl.gmic_process(img, cmd, ctx)


SHUFFLE_MODES = ["Shuffle", "Rotate", "ShuffleAndRotate"]
RECONSTRUCTION = ["Colors", "Gradients", "Laplacians"]


@itb_op("ShufflePatches", "Shuffle Patches", GROUP, [
    _enum("mode", SHUFFLE_MODES, "Shuffle", "Mode"),
    _enum("reconstruct", RECONSTRUCTION, "Colors", "Reconstruct"),
    Int("patch_size", 64, min=4, max=512, unit="px", title="Patch Size"),
    Float("overlap", 0.0, min=0, max=50, title="Overlap"),
    Float("overlap_deviation", 20.0, min=0, max=100, title="Overlap Deviation"),
    Int("seed", 0, min=0, max=65535, title="Seed"),
], **GMIC, tags=["patches", "shuffle", "glitch"])
def shuffle_patches(img, ctx, mode, reconstruct, patch_size, overlap, overlap_deviation, seed):
    """G'MIC ``fx_shuffle_patches`` (power-of-two padded for gradient / laplacian reconstruction)."""
    rec = _ev(reconstruct, RECONSTRUCTION)
    cmd = gmic_cmd("fx_shuffle_patches", _ev(mode, SHUFFLE_MODES), rec, _ipx(ctx, patch_size, 4, 512), overlap,
                   overlap_deviation, seed)
    if rec != 0:
        cmd = _with_power_of_two_padding(cmd)
    return impl.gmic_preserve(img, cmd, ctx)


SUPERPIXEL_COLORS = ["Random", "Average"]


@itb_op("SuperPixels", "Super Pixels", GROUP, [
    Int("size", 16, min=4, max=64, unit="px", title="Size"),
    Float("regularity", 10.0, min=0, max=128, title="Regularity"),
    Int("iterations", 5, min=1, max=16, title="Iterations"),
    _enum("color", SUPERPIXEL_COLORS, "Average", "Color"),
    Float("border_opacity", 1.0, min=0, max=1, title="Border Opacity"),
    Int("border_thickness", 1, min=1, max=16, unit="px", title="Border thickness"),
    Color("border_color", "#000000", title="Border color"),
], **GMIC, alpha="full", tags=["superpixels", "slic", "segmentation", "cells"])
def super_pixels(img, ctx, size, regularity, iterations, color, border_opacity, border_thickness, border_color):
    """G'MIC ``fx_superpixels`` (alpha processed)."""
    cmd = gmic_cmd("fx_superpixels", _ipx(ctx, size, 4, 64), regularity, iterations, _ev(color, SUPERPIXEL_COLORS),
                   border_opacity, _ipx(ctx, border_thickness, 1, 16), *_rgba(border_color))
    return impl.gmic_process(img, cmd, ctx)


TAQUIN_REMOVED = ["None", "First", "Last", "Random"]


@itb_op("Taquin", "Taquin", GROUP, [
    Int("x_tiles", 7, min=1, max=20, title="X Tiles"),
    Int("y_tiles", 7, min=1, max=20, title="Y Tiles"),
    _enum("removed_tile", TAQUIN_REMOVED, "None", "Removed Tile"),
    Float("relief", 50.0, min=0, max=255, title="Relief"),
    Float("border_thickness", 5.0, min=0, max=100, title="Border thickness", description="Percent of a tile"),
    Int("border_outline", 0, min=0, max=16, title="Border Outline"),
    Color("outline_color", "#000000", title="Outline Color"),
    Int("seed", 0, min=0, max=65535, title="Seed"),
], **GMIC, alpha="full", geometry=True, tags=["taquin", "sliding puzzle", "tiles", "shuffle"])
def taquin(img, ctx, x_tiles, y_tiles, removed_tile, relief, border_thickness, border_outline, outline_color, seed):
    """G'MIC ``fx_taquin`` (alpha processed; crops to a multiple of the tile count)."""
    cmd = gmic_cmd("fx_taquin", x_tiles, y_tiles, _ev(removed_tile, TAQUIN_REMOVED), relief, border_thickness,
                   border_outline, *_rgba(outline_color), seed)
    return impl.gmic_process(img, cmd, ctx)


TILEABLE_ARRAY = ["None", "XAxis", "YAxis", "BothAxes", "DoubleBothAxes"]


@itb_op("TileableRotation", "Tileable Rotation", GROUP, [
    Float("angle", 45.0, min=0, max=360, title="Angle"),
    Int("maximum_size_factor", 8, min=0, max=20, title="Maximum Size Factor"),
    _enum("array_mode", TILEABLE_ARRAY, "None", "Array Mode"),
], **GMIC, tags=["tileable", "rotate", "seamless", "pattern"])
def tileable_rotation(img, ctx, angle, maximum_size_factor, array_mode):
    """G'MIC ``fx_rotate_tileable`` inside ImageToolboxLibs' square padding (cropped back to the input size)."""
    cmd = _with_square_padding(gmic_cmd("fx_rotate_tileable", angle, maximum_size_factor,
                                        _ev(array_mode, TILEABLE_ARRAY)))
    return impl.gmic_preserve(img, cmd, ctx)
