"""ImageToolbox "Distortion" group filters.

Coordinate-mapping filters are vectorised inverse maps sampled with
``cv2.remap`` (see ``_distortion_impl``); G'MIC filters run the exact
ImageToolboxLibs command strings through the G'MIC CLI.
"""
from __future__ import annotations

import math

import numpy as np

from ...core.params import Bool, Choice, Color, File, Float, Int
from . import _distortion_impl as impl
from ._common import gmic_cmd, itb_op, run_gmic, snake

GROUP = "distortion"


# ---------------------------------------------------------------------------
# Procedural shaders (ImageToolbox ProceduralShaders.kt, rendered by GPUImage)
# ---------------------------------------------------------------------------

def _transform_params():
    return [
        Float("offset_x", 0.0, min=-1, max=1, title="Offset X", description="Horizontal offset of the effect field"),
        Float("offset_y", 0.0, min=-1, max=1, title="Offset Y", description="Vertical offset of the effect field"),
        Float("scale", 1.0, min=0.05, max=8, title="Scale", description="Zoom of the effect field"),
        Float("rotation", 0.0, min=-180, max=180, title="Rotation",
              description="Rotation of the effect field (degrees)"),
    ]


def _intensity(default, lo=-1.0, hi=1.0):
    return Float("intensity", default, min=lo, max=hi, title="Intensity")


def _frequency(default=8.0, lo=1.0, hi=40.0):
    return Float("frequency", default, min=lo, max=hi, title="Frequency")


def _distortion(default):
    return Float("distortion", default, min=-1, max=1, title="Distortion")


def _count(default, hi=32):
    return Int("count", default, min=1, max=hi, title="Count")


def _seed():
    return Int("seed", 0, min=0, max=1000, title="Random seed", description="Pattern variation (hash seed)")


def _procedural(name: str, title: str, params: list, tags: list[str], doc: str) -> None:
    def fn(img, ctx, **values):
        return impl.procedural(img, name, values)

    fn.__name__ = snake(name)
    fn.__doc__ = doc
    itb_op(name, title, GROUP, params + _transform_params(), upstream="ImageToolbox", alpha="full",
           tags=["distortion", *tags])(fn)


_procedural("BrokenCells", "Broken Cells", [
    _intensity(0.5, 0.0, 1.0), _distortion(0.2),
    Float("variability", 0.5, min=0, max=1, title="Variability"), _frequency(), _seed(),
], ["voronoi", "cells", "shatter"], "Voronoi cells, each shifted by a random offset (procedural shader).")

_procedural("BrokenGlass", "Broken Glass", [
    _intensity(0.4), _frequency(), _seed(),
], ["voronoi", "glass", "shatter"], "Voronoi shards rotated and shifted randomly (procedural shader).")

_procedural("Gallium", "Gallium", [
    Int("iterations", 5, min=1, max=16, title="Iterations"), _intensity(0.2),
], ["liquid", "metal", "luminance"], "Iteratively pushes sampling along the luminance gradient (procedural shader).")

_procedural("GlassHexTiles", "Glass Hex Tiles", [
    _intensity(0.2), _distortion(0.25), _frequency(),
], ["glass", "tiles", "hexagon"], "Hexagonal glass-block refraction (procedural shader).")

_procedural("GlassRectTiles", "Glass Rect Tiles", [
    _intensity(0.2), _distortion(0.25), _frequency(),
], ["glass", "tiles", "square"], "Square glass-block refraction (procedural shader).")

_procedural("GlassTriangleTiles", "Glass Triangle Tiles", [
    _intensity(0.2), _distortion(0.25), _frequency(),
], ["glass", "tiles", "triangle"], "Triangular glass-block refraction (procedural shader).")

_procedural("GridSineDistortion2", "Grid Sine Distortion 2", [
    _intensity(0.2), _frequency(),
], ["wave", "sine", "grid"], "Crossed sine waves displacing a grid (procedural shader).")

_procedural("HelixWaves", "Helix Waves", [
    _frequency(), _intensity(0.2),
    Int("mode", 0, min=0, max=2, title="Mode", description="0 = radial scale, 1 = rotation, 2 = offset"),
    Float("lighting", 0.0, min=0, max=1, title="Lighting"),
], ["wave", "spiral", "helix"], "Spiral helix waves with optional shading (procedural shader).")

_procedural("LogarithmicSpiral", "Logarithmic Spiral", [
    _count(2), _intensity(1.0, -4.0, 4.0),
], ["spiral", "log polar"], "Log-polar spiral tiling of the image (procedural shader).")

_procedural("Quicksilver", "Quicksilver", [
    Float("displacement", 0.1, min=-0.5, max=0.5, title="Displacement"), _intensity(0.2), _frequency(),
], ["liquid", "metal", "luminance"], "Luminance-gradient displacement with sine ripples (procedural shader).")

_procedural("RadialInterpolate", "Radial Interpolate", [
    _intensity(1.0, 0.0, 1.0), _count(12, 64),
    Float("radius", 0.3, min=0.01, max=1, title="Radius"), Float("width", 0.2, min=0.01, max=0.8, title="Width"),
], ["radial", "rings"], "Quantises the radius inside a ring band (procedural shader).")

_procedural("RadialShimmer", "Radial Shimmer", [
    _count(8), _intensity(0.2), Float("spacing", 1.0, min=0.1, max=4, title="Spacing"),
], ["radial", "rings", "shimmer"], "Concentric rings rotating back and forth (procedural shader).")

_procedural("Rubidium", "Rubidium", [
    Float("displacement", 0.1, min=-0.5, max=0.5, title="Displacement"), _intensity(0.2),
    Float("balance", 0.5, min=0, max=1, title="Balance"),
], ["liquid", "luminance", "chroma"], "Gradient / tangent flow plus chroma-driven displacement (procedural shader).")

_procedural("SpiralArms", "Spiral Arms", [
    _count(6), _intensity(1.0, -4.0, 4.0),
], ["spiral", "polar"], "Polar spiral-arm tiling of the image (procedural shader).")

_procedural("SpiralDroste", "Spiral Droste", [
    Float("intensity", 2.0, min=1.1, max=8, title="Intensity"), _distortion(0.2),
], ["droste", "spiral", "recursive"], "Escher-like spiral Droste effect (procedural shader).")

_procedural("SquareFresnel", "Square Fresnel", [
    Float("intensity", 6.0, min=1, max=32, title="Intensity"),
], ["fresnel", "square", "rings"], "Square Fresnel-lens rings (procedural shader).")

_procedural("SquareSpiralDroste", "Square Spiral Droste", [
    Float("intensity", 2.0, min=1.1, max=8, title="Intensity"), _distortion(0.15),
    Float("shape_aspect", 1.0, min=0.25, max=4, title="Shape aspect ratio"),
], ["droste", "spiral", "square", "recursive"], "Square spiral Droste effect (procedural shader).")


# ---------------------------------------------------------------------------
# GPUImage
# ---------------------------------------------------------------------------

@itb_op("BulgeDistortion", "Bulge", GROUP, [
    Float("radius", 0.25, min=0, max=1, title="Radius"),
    Float("scale", 0.5, min=-1, max=1, title="Scale"),
], upstream="GPUImage", alpha="full", tags=["distortion", "bulge", "pinch", "lens"])
def bulge_distortion(img, ctx, radius, scale):
    """GPUImageBulgeDistortionFilter around the image centre."""
    return impl.gpu_bulge(img, radius, scale)


@itb_op("GlassSphereRefraction", "Glass sphere refraction", GROUP, [
    Float("radius", 0.25, min=0, max=1, title="Radius"),
    Float("refractive_index", 0.71, min=0, max=1, title="Refractive index"),
], upstream="GPUImage", alpha="full", tags=["distortion", "sphere", "glass", "refraction"])
def glass_sphere_refraction(img, ctx, radius, refractive_index):
    """GPUImageGlassSphereFilter: refracted, lit glass ball; outside the sphere is transparent."""
    return impl.gpu_sphere(img, radius, refractive_index, glass=True)


@itb_op("SphereRefraction", "Sphere refraction", GROUP, [
    Float("radius", 0.25, min=0, max=1, title="Radius"),
    Float("refractive_index", 0.71, min=0, max=1, title="Refractive index"),
], upstream="GPUImage", alpha="full", tags=["distortion", "sphere", "refraction"])
def sphere_refraction(img, ctx, radius, refractive_index):
    """GPUImageSphereRefractionFilter: refracted ball; outside the sphere is transparent."""
    return impl.gpu_sphere(img, radius, refractive_index, glass=False)


@itb_op("SwirlDistortion", "Swirl", GROUP, [
    Float("radius", 0.5, min=0, max=1, title="Radius"),
    Float("angle", 1.0, min=-1, max=1, title="Angle"),
    Float("center_x", 0.5, min=0.01, max=1, title="Center X"),
    Float("center_y", 0.5, min=0.01, max=1, title="Center Y"),
], upstream="GPUImage", alpha="full", tags=["distortion", "swirl", "twist"])
def swirl_distortion(img, ctx, radius, angle, center_x, center_y):
    """GPUImageSwirlFilter (texture-space radius, no aspect correction, like upstream)."""
    return impl.gpu_swirl(img, radius, angle, center_x, center_y)


# ---------------------------------------------------------------------------
# jhlabs (ImageToolboxLibs Rust port)
# ---------------------------------------------------------------------------

def _center():
    return [Float("center_x", 0.5, min=0, max=1, title="Center X"),
            Float("center_y", 0.5, min=0, max=1, title="Center Y")]


@itb_op("Pinch", "Whirl and Pinch", GROUP, [
    Int("angle", 15, min=0, max=360, title="Angle", description="Whirl angle in degrees"),
    *_center(),
    Float("radius", 1.0, min=0, max=2, title="Radius", description="Relative to half the longest side"),
    Float("amount", 0.5, min=-1, max=1, title="Amount"),
], upstream="jhlabs", alpha="full", tags=["distortion", "pinch", "whirl"])
def pinch(img, ctx, angle, center_x, center_y, radius, amount):
    """jhlabs PinchFilter (bilinear, RGB_CLAMP edges: transparent outside the source)."""
    h, w = img.shape[:2]
    return impl.jh_pinch(img, math.radians(angle), center_x, center_y, radius * (max(w, h) / 2.0), amount)


@itb_op("Ripple", "Ripple", GROUP, [
    Float("x_amplitude", 5.0, min=0, max=100, step=0.1, title="X amplitude", unit="px"),
    Float("x_wavelength", 16.0, min=1, max=200, step=0.1, title="X wavelength", unit="px"),
    Float("y_amplitude", 5.0, min=0, max=100, step=0.1, title="Y amplitude", unit="px"),
    Float("y_wavelength", 16.0, min=1, max=200, step=0.1, title="Y wavelength", unit="px"),
], upstream="jhlabs", alpha="full", tags=["distortion", "ripple", "wave"])
def ripple(img, ctx, x_amplitude, x_wavelength, y_amplitude, y_wavelength):
    """jhlabs RippleFilter, sine wave (bilinear, RGB_CLAMP edges)."""
    return impl.jh_ripple(img, ctx.px(x_amplitude), ctx.px(x_wavelength), ctx.px(y_amplitude), ctx.px(y_wavelength))


@itb_op("SphereLensDistortion", "Sphere Lens Distortion", GROUP, [
    Float("refraction_index", 2.5, min=1, max=10, title="Refraction index"),
    Float("radius", 0.5, min=0, max=1, title="Radius", description="Relative to half the longest side"),
    *_center(),
], upstream="jhlabs", alpha="full", tags=["distortion", "sphere", "lens", "refraction"])
def sphere_lens_distortion(img, ctx, refraction_index, radius, center_x, center_y):
    """jhlabs SphereFilter (bilinear, CLAMP edges)."""
    h, w = img.shape[:2]
    return impl.jh_sphere_lens(img, refraction_index, radius * (max(w, h) / 2.0), center_x, center_y)


@itb_op("Twirl", "Twirl", GROUP, [
    Float("angle", 45.0, min=-360, max=360, title="Angle", description="Twirl angle in degrees"),
    *_center(),
    Float("radius", 0.5, min=0, max=1, title="Radius", description="Relative to half the longest side"),
], upstream="jhlabs", alpha="full", tags=["distortion", "twirl", "swirl"])
def twirl(img, ctx, angle, center_x, center_y, radius):
    """jhlabs TwirlFilter (bilinear, CLAMP edges)."""
    h, w = img.shape[:2]
    return impl.jh_twirl(img, math.radians(angle), center_x, center_y, radius * (max(w, h) / 2.0))


@itb_op("WaterDrop", "Water Drop", GROUP, [
    Float("wavelength", 0.1, min=0.01, max=1, title="Wavelength", description="Relative to half the longest side"),
    Float("amplitude", 0.05, min=0, max=0.5, title="Amplitude", description="Relative to half the longest side"),
    Int("phase", 0, min=0, max=360, title="Phase", description="Degrees"),
    *_center(),
    Float("radius", 0.5, min=0.01, max=1, title="Radius", description="Relative to half the longest side"),
], upstream="jhlabs", alpha="full", tags=["distortion", "water", "ripple", "drop"])
def water_drop(img, ctx, wavelength, amplitude, phase, center_x, center_y, radius):
    """jhlabs WaterFilter: concentric ripples (bilinear, CLAMP edges)."""
    h, w = img.shape[:2]
    r = max(w, h) / 2.0
    return impl.jh_water(img, wavelength * r, amplitude * r, math.radians(phase), center_x, center_y, radius * r)


# ---------------------------------------------------------------------------
# aire
# ---------------------------------------------------------------------------

@itb_op("Convex", "Spline", GROUP, [
    Float("strength", 3.0, min=0, max=30, title="Strength"),
], upstream="aire", alpha="full", tags=["distortion", "convex", "lens", "vignette blur"])
def convex(img, ctx, strength):
    """aire ``ConvexEffect``: radial nearest-neighbour squeeze blended with a stack blur towards the edges."""
    return impl.aire_convex(img, strength, ctx.scale)


@itb_op("FractalGlass", "Fractal Glass", GROUP, [
    Float("strength", 0.02, min=0, max=1, title="Strength",
          description="Glass stripe width relative to the image width"),
    Float("amplitude", 0.02, min=0, max=1, title="Amplitude", description="Shift relative to the image width"),
], upstream="aire", alpha="full", tags=["distortion", "glass", "fluted", "reeded"])
def fractal_glass(img, ctx, strength, amplitude):
    """aire ``fractalGlassEffect``: vertical sawtooth displacement (reeded glass)."""
    return impl.aire_fractal_glass(img, strength, amplitude)


@itb_op("Marble", "Marble", GROUP, [
    Float("strength", 0.02, min=0, max=1, title="Strength"),
    Float("turbulence", 1.0, min=0, max=1, title="Turbulence"),
    Float("amplitude", 1.0, min=0, max=1, title="Amplitude"),
], upstream="aire", alpha="full", tags=["distortion", "marble", "noise", "perlin"])
def marble(img, ctx, strength, turbulence, amplitude):
    """aire ``marbleEffect``: Perlin-noise driven displacement (noise seeded from the render seed)."""
    return impl.aire_noise_warp(img, strength, turbulence, amplitude, ctx.rng("marble"), ctx.scale, colorize=False)


@itb_op("PerlinDistortion", "Perlin Distortion", GROUP, [
    Float("strength", 0.02, min=0, max=1, title="Strength"),
    Float("turbulence", 1.0, min=0, max=1, title="Turbulence"),
    Float("amplitude", 1.0, min=0, max=1, title="Amplitude"),
], upstream="aire", alpha="full", tags=["distortion", "noise", "perlin"])
def perlin_distortion(img, ctx, strength, turbulence, amplitude):
    """aire ``perlinDistortion``: marble displacement plus a noise brightness term."""
    return impl.aire_noise_warp(img, strength, turbulence, amplitude, ctx.rng("perlin"), ctx.scale, colorize=True)


@itb_op("WaterEffect", "Water Effect", GROUP, [
    Float("size", 0.2, min=0, max=1, step=0.01, title="Size"),
    Float("frequency_x", 2.0, min=-4, max=4, step=0.01, title="Frequency X"),
    Float("frequency_y", 2.0, min=-4, max=4, step=0.01, title="Frequency Y"),
    Float("amplitude_x", 0.5, min=-4, max=4, step=0.01, title="Amplitude X"),
    Float("amplitude_y", 0.5, min=-4, max=4, step=0.01, title="Amplitude Y"),
], upstream="aire", alpha="full", tags=["distortion", "water", "wave"])
def water_effect(img, ctx, size, frequency_x, frequency_y, amplitude_x, amplitude_y):
    """aire ``waterEffect`` (ImageToolbox passes fractionSize = 0.2 * size)."""
    return impl.aire_water(img, 0.2 * size, frequency_x, amplitude_x, frequency_y, amplitude_y)


# ---------------------------------------------------------------------------
# ImageToolbox / OpenCV
# ---------------------------------------------------------------------------

@itb_op("Anaglyph", "Anaglyph", GROUP, [
    Int("shift", 20, min=0, max=100, title="Shift", unit="px",
        description="Horizontal offset of the red and cyan layers"),
], upstream="ImageToolbox", alpha="full", tags=["distortion", "anaglyph", "3d", "stereo", "red cyan"])
def anaglyph(img, ctx, shift):
    """ImageToolbox GlitchTool.anaglyph: red layer shifted left, cyan layer shifted right, PorterDuff ADD."""
    return impl.anaglyph(img, int(round(ctx.px(shift))))


@itb_op("Distortion", "Content Aware Distortion", GROUP, [
    Int("strength", 50, min=0, max=95, title="Strength", description="Percentage of seams removed per axis"),
], upstream="OpenCV", alpha="full", tags=["distortion", "seam carving", "content aware", "liquid rescale"])
def distortion(img, ctx, strength):
    """opencv-tools SeamCarver.distort: forward-energy seam carving at <=512 px, then bicubic upscale."""
    return impl.seam_distort(img, float(strength))


# ---------------------------------------------------------------------------
# G'MIC
# ---------------------------------------------------------------------------

BOUNDARY = ["transparent", "nearest", "periodic", "mirror"]
_CHANNELS = ["All", "Rgba", "Rgb", "Red", "Green", "Blue", "Alpha", "LinearRgb", "LinearRed", "LinearGreen",
             "LinearBlue", "YCbCr", "Luminance", "Chrominance", "BlueChrominance", "RedChrominance",
             "GreenChrominance", "Lab", "LabLightness", "LabChrominance", "LabA", "LabB", "Lch", "LchChrominance",
             "LchChroma", "LchHue", "Hsv", "HsvHue", "HsvSaturation", "HsvValue", "Hsi", "HsiIntensity", "Hsl",
             "HslLightness", "Cmyk", "Cmy", "Cyan", "Magenta", "Yellow", "Key", "Yiq", "YiqLuma", "YiqChrominance",
             "Ryb", "RybRed", "RybYellow", "RybBlue"]
CHANNELS = [snake(c) for c in _CHANNELS]


def _boundary(default="mirror"):
    return Choice("boundary", default, choices=BOUNDARY, title="Boundary",
                  labels={c: c.capitalize() for c in BOUNDARY})


def _channel():
    return Choice("channel", "all", choices=CHANNELS, title="Channel", labels=dict(zip(CHANNELS, _CHANNELS)))


def _gmic(img, ctx, cmd, *, seed_salt=None, aux=()):
    """Run with GmicAlphaMode.Process semantics (alpha is a 4th channel, as for every Android bitmap)."""
    had = impl.has_alpha(img)
    rgba = impl.to_rgba(img)
    if seed_salt is not None:  # G'MIC's RNG is time-seeded upstream; seed it from the render seed instead
        cmd = f"srand {int(ctx.rng(seed_salt).integers(0, 2**31 - 1))} {cmd}"
    out = run_gmic(rgba, cmd, ctx=ctx, preserve_alpha=False, aux=aux)
    if out.shape[2] == 3:
        out = np.dstack([out, np.ones(out.shape[:2], np.float32)])
    return impl.finish(out, had)


_GMIC = dict(upstream="G'MIC", requires=["gmic"], alpha="full")


@itb_op("Crease", "Crease", GROUP, [
    Float("amplitude", 30.0, min=0, max=300, title="Amplitude", unit="px"),
    Float("frequency", 10.0, min=0, max=100, title="Frequency"),
    _boundary(),
], tags=["distortion", "crease", "crumple", "noise"], **_GMIC)
def crease(img, ctx, amplitude, frequency, boundary):
    """G'MIC ``fx_crease`` (random field seeded from the render seed)."""
    cmd = gmic_cmd("fx_crease", ctx.px(amplitude), frequency, BOUNDARY.index(boundary))
    return _gmic(img, ctx, cmd, seed_salt="crease")


@itb_op("Displacement", "Displacement", GROUP, [
    File("horizontal_map", "", file_kind="image", title="Horizontal displacement map",
         description="Luminance 128 = no shift; required for any effect"),
    File("vertical_map", "", file_kind="image", title="Vertical displacement map",
         description="Optional; defaults to the horizontal map"),
    Float("strength_x", 0.1, min=-1, max=1, step=0.001, title="Horizontal strength"),
    Float("strength_y", 0.1, min=-1, max=1, step=0.001, title="Vertical strength"),
    Choice("interpolation", "linear", choices=["nearest", "linear", "cubic"], title="Interpolation"),
    Choice("boundary", "clamp", choices=["transparent", "clamp", "wrap", "mirror"], title="Boundary"),
], tags=["distortion", "displacement map", "warp"], **_GMIC)
def displacement(img, ctx, horizontal_map, vertical_map, strength_x, strength_y, interpolation, boundary):
    """G'MIC warp by the luminance of displacement map(s) (ImageToolboxLibs ``Displacement``)."""
    if not horizontal_map:
        return img  # ImageToolbox returns the input when no map can be loaded
    from ...core.io import load_image

    hmap, _ = load_image(ctx.resolve_file(horizontal_map, "displacement map"))
    vmap = load_image(ctx.resolve_file(vertical_map, "displacement map"))[0] if vertical_map else hmap

    def f(v):  # number formatting of gmic_cmd
        return gmic_cmd("x", v)[2:]

    interp = ["nearest", "linear", "cubic"].index(interpolation)
    bnd = ["transparent", "clamp", "wrap", "mirror"].index(boundary)
    cmd = ("channels[1,2] 0,2 luminance[1,2] resize[1,2] {w#0},{h#0},1,1,3 "
           f"sub[1] 127.5 div[1] 127.5 mul[1] {{w#0*{f(strength_x)}}} "
           f"sub[2] 127.5 div[2] 127.5 mul[2] {{h#0*{f(strength_y)}}} "
           f"append[1,2] c warp[0] [1],1,{interp},{bnd} keep[0]")
    return _gmic(img, ctx, cmd, aux=[hmap[..., :3], vmap[..., :3]])


@itb_op("FishEye", "Fish-Eye", GROUP, [
    Float("radius", 70.0, min=0, max=100, title="Radius", description="Lens radius (% of the image)"),
    Float("amplitude", 1.0, min=0, max=2, title="Amplitude", description="Strength of the bulge"),
], tags=["distortion", "fisheye", "lens"], **_GMIC)
def fish_eye(img, ctx, radius, amplitude):
    """G'MIC ``fisheye 50,50,radius,amplitude`` (centred).

    Deviation (bug fix): ImageToolboxLibs calls ``fisheye radius,amplitude``, which G'MIC reads as the
    centre position, so its Radius/Amplitude sliders only moved the effect.  The intended call is used.
    """
    return _gmic(img, ctx, gmic_cmd("fisheye", 50, 50, radius, amplitude))


@itb_op("Flower", "Flower", GROUP, [
    Float("center_x", 50.0, min=0, max=100, title="Center X"),
    Float("center_y", 50.0, min=0, max=100, title="Center Y"),
    Float("amplitude_point_x", 75.0, min=0, max=100, title="Amplitude Point X"),
    Float("amplitude_point_y", 50.0, min=0, max=100, title="Amplitude Point Y"),
    Int("petals", 6, min=2, max=20, title="Petals"),
    Float("offset", 0.0, min=0, max=100, title="Offset"),
    _boundary(),
], tags=["distortion", "flower", "polar"], **_GMIC)
def flower(img, ctx, center_x, center_y, amplitude_point_x, amplitude_point_y, petals, offset, boundary):
    """G'MIC ``fx_flower``."""
    cmd = gmic_cmd("fx_flower", center_x, center_y, amplitude_point_x, amplitude_point_y, petals, offset,
                   BOUNDARY.index(boundary))
    return _gmic(img, ctx, cmd)


@itb_op("LensDistortion", "Distort Lens", GROUP, [
    Float("amplitude", 0.1, min=-1, max=1, title="Amplitude"),
    Float("aspect_ratio", 0.0, min=-2, max=2, title="Aspect ratio"),
    Float("zoom", 0.0, min=-4, max=4, title="Zoom"),
    Float("center_x", 50.0, min=0, max=100, title="Center X"),
    Float("center_y", 50.0, min=0, max=100, title="Center Y"),
    _boundary("transparent"),
], tags=["distortion", "lens", "barrel", "pincushion"], **_GMIC)
def lens_distortion(img, ctx, amplitude, aspect_ratio, zoom, center_x, center_y, boundary):
    """G'MIC ``fx_distort_lens`` (barrel / pincushion)."""
    cmd = gmic_cmd("fx_distort_lens", amplitude, aspect_ratio, zoom, center_x, center_y, BOUNDARY.index(boundary))
    return _gmic(img, ctx, cmd)


@itb_op("PowerTwirl", "Power Twirl", GROUP, [
    Float("amplitude", 1.0, min=-20, max=20, title="Amplitude"),
    Float("offset", 0.0, min=-180, max=180, title="Offset"),
    Float("center_x", 50.0, min=0, max=100, title="Center X"),
    Float("center_y", 50.0, min=0, max=100, title="Center Y"),
    Float("power", 1.0, min=-20, max=20, title="Power"),
    _boundary(),
    Choice("mode", "polar", choices=["polar", "cartesian"], title="Mode"),
], tags=["distortion", "twirl", "swirl"], **_GMIC)
def power_twirl(img, ctx, amplitude, offset, center_x, center_y, power, boundary, mode):
    """G'MIC ``fx_powertwirl``."""
    cmd = gmic_cmd("fx_powertwirl", amplitude, offset, center_x, center_y, power, BOUNDARY.index(boundary),
                   ["polar", "cartesian"].index(mode))
    return _gmic(img, ctx, cmd)


@itb_op("RandomDeformations", "Random Deformations", GROUP, [
    Bool("recompute", False, title="Recompute"),
    Float("amplitude", 5.0, min=0, max=50, title="Amplitude", unit="px"),
    Choice("interpolation", "linear", choices=["none", "linear", "bicubic"], title="Interpolation"),
    Float("matrix_density", 10.0, min=1, max=100, title="Matrix Density"),
    Choice("matrix_interpolation", "bicubic", choices=["linear", "bicubic"], title="Matrix Interpolation"),
    Choice("mode", "noise", choices=["noise", "spread_noise"], title="Mode"),
    Float("character", 0.0, min=-100, max=100, title="Character"),
    Choice("boundary", "neumann", choices=["dirichlet", "neumann", "periodic", "mirror"], title="Boundary"),
], tags=["distortion", "random", "deform", "noise"], **_GMIC)
def random_deformations(img, ctx, recompute, amplitude, interpolation, matrix_density, matrix_interpolation,
                        mode, character, boundary):
    """G'MIC ``fx_jr_deform`` (random field seeded from the render seed)."""
    cmd = gmic_cmd("fx_jr_deform", recompute, ctx.px(amplitude), ["none", "linear", "bicubic"].index(interpolation),
                   matrix_density, ["linear", "bicubic"].index(matrix_interpolation),
                   ["noise", "spread_noise"].index(mode), character,
                   ["dirichlet", "neumann", "periodic", "mirror"].index(boundary))
    return _gmic(img, ctx, cmd, seed_salt="jr_deform")


@itb_op("ShockWaves", "Shock Waves", GROUP, [
    Float("amplitude", 10.0, min=0, max=100, title="Amplitude"),
    Float("low_frequency", 10.0, min=0, max=100, title="Low Frequency"),
    Float("frequency_range", 20.0, min=0, max=100, title="Frequency Range"),
    _channel(),
], tags=["distortion", "shock waves", "dct", "ripple"], **_GMIC)
def shock_waves(img, ctx, amplitude, low_frequency, frequency_range, channel):
    """G'MIC ``fx_shockwaves`` (DCT band amplification)."""
    cmd = gmic_cmd("fx_shockwaves", amplitude, low_frequency, frequency_range, CHANNELS.index(channel))
    return _gmic(img, ctx, cmd)


@itb_op("WarpByIntensity", "Warp By Intensity", GROUP, [
    Float("x_factor", 0.04, min=-6, max=6, title="X Factor", description="Pixels of shift per intensity level"),
    Float("y_factor", 0.04, min=-6, max=6, title="Y Factor", description="Pixels of shift per intensity level"),
    Float("x_offset", 128.0, min=0, max=255, title="X Offset"),
    Float("y_offset", 128.0, min=0, max=255, title="Y Offset"),
    Bool("correlated_channels", False, title="Correlated Channels"),
    Choice("interpolation", "linear", choices=["nearest_neighbor", "linear"], title="Interpolation"),
    _boundary(),
    _channel(),
], tags=["distortion", "warp", "intensity"], **_GMIC)
def warp_by_intensity(img, ctx, x_factor, y_factor, x_offset, y_offset, correlated_channels, interpolation,
                      boundary, channel):
    """G'MIC ``fx_warp_by_intensity`` (factors scaled with the preview scale)."""
    cmd = gmic_cmd("fx_warp_by_intensity", ctx.px(x_factor), ctx.px(y_factor), x_offset, y_offset,
                   correlated_channels, ["nearest_neighbor", "linear"].index(interpolation),
                   BOUNDARY.index(boundary), CHANNELS.index(channel))
    return _gmic(img, ctx, cmd)


@itb_op("WaterReflection", "Water Reflection", GROUP, [
    Float("height", 50.0, min=0, max=100, title="Height"),
    Float("attenuation", 1.0, min=0.1, max=4, title="Attenuation"),
    Color("color", "#6EA0BE40", title="Color"),
    Float("amplitude", 0.0, min=0, max=100, title="Amplitude", unit="px"),
    Float("softness", 1.5, min=0, max=4, title="Softness", unit="px"),
    Float("x_angle", 0.0, min=-10, max=10, title="X angle"),
    Float("y_angle", -3.3, min=-10, max=10, title="Y angle"),
    Float("focal_length", 7.0, min=0, max=10, title="Focal Length"),
    Float("zoom", 1.5, min=1, max=5, title="Image zoom"),
], geometry=True, tags=["distortion", "reflection", "water", "mirror"], **_GMIC)
def water_reflection(img, ctx, height, attenuation, color, amplitude, softness, x_angle, y_angle, focal_length, zoom):
    """G'MIC ``fx_reflect``: mirrored, tinted water reflection in perspective (output is auto-cropped)."""
    r, g, b, a = (int(round(c * 255)) for c in color)
    cmd = gmic_cmd("fx_reflect", height, attenuation, r, g, b, a, ctx.px(amplitude), ctx.px(softness),
                   x_angle, y_angle, focal_length, zoom)
    return _gmic(img, ctx, cmd, seed_salt="reflect")
