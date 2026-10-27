"""ImageToolbox "Dithering" group filters.

Trickle (Apache-2.0) ``Dithering.cpp`` ports work on 8-bit channels exactly like
upstream: every channel is thresholded to 0/255 independently, and the "Gray
Scale" switch dithers the *red* channel only and writes it to R, G and B (that
is what Trickle does: ``int gray = r``).
"""
from __future__ import annotations

import cv2
import numpy as np

from ...core.params import Bool, Choice, Float, Int
from . import _dithering_impl as D
from ._common import gmic_cmd, itb_op, run_gmic

GROUP = "dithering"
F32 = np.float32


def _threshold():
    return Int("threshold", 200, min=1, max=255, title="Threshold",
               description="0..255 level a channel (plus diffused error) must reach to become white")


def _gray():
    return Bool("gray_scale", False, title="Gray Scale",
                description="Dither the red channel only and output black/white (Trickle behaviour)")


# ---------------------------------------------------------------------------
# Trickle error diffusion
# ---------------------------------------------------------------------------

_DIFFUSION = [
    ("FloydSteinbergDithering", "Floyd Steinberg Dithering", "floyd_steinberg", "7/16, 3/16, 5/16, 1/16"),
    ("FalseFloydSteinbergDithering", "False Floyd Steinberg Dithering", "false_floyd_steinberg", "3/8, 3/8, 2/8"),
    ("JarvisJudiceNinkeDithering", "Jarvis Judice Ninke Dithering", "jarvis_judice_ninke", "12 taps /48"),
    ("StuckiDithering", "Stucki Dithering", "stucki", "12 taps /42"),
    ("BurkesDithering", "Burkes Dithering", "burkes", "7 taps /32"),
    ("SierraDithering", "Sierra Dithering", "sierra", "10 taps /32"),
    ("TwoRowSierraDithering", "Two Row Sierra Dithering", "two_row_sierra", "7 taps /16"),
    ("SierraLiteDithering", "Sierra Lite Dithering", "sierra_lite",
     "Trickle uses the Two-Row Sierra kernel here"),
    ("AtkinsonDithering", "Atkinson Dithering", "atkinson", "6 taps of 1/8 (75% of the error)"),
]


def _make_diffusion(name: str, title: str, kernel: str, note: str) -> None:
    def fn(img, ctx, threshold, gray_scale):
        return D.error_diffusion(img, kernel, threshold, gray_scale)

    fn.__name__ = kernel + "_dithering"
    itb_op(name, title, GROUP, [_threshold(), _gray()], upstream="Trickle",
           description=f"{title}: raster-order error diffusion to pure 0/255 channels ({note}).",
           tags=["dithering", "error diffusion", "1-bit", "retro", kernel.replace("_", " ")])(fn)


for _args in _DIFFUSION:
    _make_diffusion(*_args)


@itb_op("LeftToRightDithering", "Left To Right Dithering", GROUP, [_threshold(), _gray()], upstream="Trickle",
        tags=["dithering", "error diffusion", "1-bit", "retro"])
def left_to_right_dithering(img, ctx, threshold, gray_scale):
    """Simple left-to-right error diffusion: each row carries a running error (|delta| < 10 ignored)."""
    return D.left_to_right(img, threshold, gray_scale)


# ---------------------------------------------------------------------------
# Trickle ordered / threshold dithering
# ---------------------------------------------------------------------------

def _make_bayer(name: str, title: str, size: int) -> None:
    def fn(img, ctx, threshold, gray_scale):
        return D.bayer(img, size, threshold, gray_scale)

    fn.__name__ = f"bayer_{size}_dithering"
    itb_op(name, title, GROUP, [_threshold(), _gray()], upstream="Trickle",
           description=f"Ordered dithering with Trickle's {size}x{size} Bayer matrix: "
                       f"v + v*M/{D.BAYER[size][1]} compared with the threshold.",
           tags=["dithering", "ordered", "bayer", "1-bit", "retro"])(fn)


for _n, _t, _s in [("BayerTwoDithering", "Bayer Two By Two Dithering", 2),
                   ("BayerThreeDithering", "Bayer Three By Three Dithering", 3),
                   ("BayerFourDithering", "Bayer Four By Four Dithering", 4),
                   ("BayerEightDithering", "Bayer Eight By Eight Dithering", 8)]:
    _make_bayer(_n, _t, _s)


def _make_clustered(name: str, title: str, size: int) -> None:
    def fn(img, ctx, gray_scale):
        return D.clustered(img, size, gray_scale)

    fn.__name__ = f"clustered_{size}_dithering"
    itb_op(name, title, GROUP, [_gray()], upstream="Trickle",
           description=f"Clustered-dot ordered dithering with a {size}x{size} threshold matrix "
                       f"(threshold = M * {D.CLUSTERED[size][1]}).",
           tags=["dithering", "ordered", "clustered", "halftone", "1-bit"])(fn)


for _n, _t, _s in [("Clustered2x2Dithering", "Clustered 2x2 Dithering", 2),
                   ("Clustered4x4Dithering", "Clustered 4x4 Dithering", 4),
                   ("Clustered8x8Dithering", "Clustered 8x8 Dithering", 8)]:
    _make_clustered(_n, _t, _s)


@itb_op("YililomaDithering", "Yililoma Dithering", GROUP, [_gray()], upstream="Trickle",
        tags=["dithering", "ordered", "yliluoma", "1-bit"])
def yililoma_dithering(img, ctx, gray_scale):
    """Trickle 'Yililoma' dithering: 4x4 ordered matrix (1..16) * 16 as per-pixel thresholds."""
    return D.yliloma(img, gray_scale)


@itb_op("SimpleThresholdDithering", "Simple Threshold Dithering", GROUP, [_threshold(), _gray()],
        upstream="Trickle", tags=["dithering", "threshold", "1-bit", "posterize"])
def simple_threshold_dithering(img, ctx, threshold, gray_scale):
    """Per-channel hard threshold to 0/255."""
    return D.simple_threshold(img, threshold, gray_scale)


@itb_op("RandomDithering", "Random Dithering", GROUP, [
    Int("threshold", 200, min=1, max=255, title="Threshold",
        description="Shown by ImageToolbox but ignored by the upstream implementation"),
    _gray(),
], upstream="Trickle", tags=["dithering", "random", "noise", "1-bit"])
def random_dithering(img, ctx, threshold, gray_scale):
    """Random-threshold dithering: one random 0..255 threshold per pixel (seeded, shared by R, G, B)."""
    return D.random_threshold(img, ctx.rng("random_dithering"), gray_scale)


# ---------------------------------------------------------------------------
# jhlabs / GPUImage halftones
# ---------------------------------------------------------------------------

@itb_op("ColorHalftone", "Color Halftone", GROUP, [
    Float("radius", 2.0, min=0, max=50, step=0.01, unit="px", title="Radius", description="Maximum dot radius"),
    Int("cyan", 108, min=0, max=360, title="Cyan", description="Screen angle of the cyan (red channel) dots, degrees"),
    Int("magenta", 162, min=0, max=360, title="Magenta",
        description="Screen angle of the magenta (green channel) dots, degrees"),
    Int("yellow", 90, min=0, max=360, title="Yellow",
        description="Screen angle of the yellow (blue channel) dots, degrees"),
], upstream="jhlabs", tags=["halftone", "cmyk", "print", "dots", "comic"])
def color_halftone(img, ctx, radius, cyan, magenta, yellow):
    """jhlabs Color Halftone: per-channel rotated dot screens whose dot size follows 1 - v^2."""
    return D.color_halftone(img, ctx.px(radius), (cyan, magenta, yellow))


_W_LUM = np.array([0.2125, 0.7154, 0.0721], F32)


def _tex_coords(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """GPUImage textureCoordinate at pixel centres (row 0 = t 0)."""
    return (np.arange(w, dtype=F32) + F32(0.5)) / F32(w), (np.arange(h, dtype=F32) + F32(0.5)) / F32(h)


def _glsl_mod(x, y):
    return x - y * np.floor(x / y)


@itb_op("Halftone", "Halftone", GROUP, [
    Float("size", 0.005, min=0.001, max=0.02, step=0.0001, title="Size",
          description="Dot cell size as a fraction of the image width"),
], upstream="GPUImage", tags=["halftone", "dots", "print", "black and white", "comic"])
def halftone(img, ctx, size):
    """GPUImage halftone: black dots on white, dot radius = cell/2 * (1 - luminance of the cell centre)."""
    h, w = img.shape[:2]
    f = F32(size)
    aspect = F32(h) / F32(w)
    tx, ty = _tex_coords(h, w)
    sx = (tx - _glsl_mod(tx, f) + F32(0.5) * f).astype(F32)
    dyf = F32(f / aspect)
    sy = (ty - _glsl_mod(ty, dyf) + F32(0.5) * dyf).astype(F32)
    # texture2D(samplePos): GL_LINEAR + CLAMP_TO_EDGE, evaluated once per distinct cell centre
    ux, ix = np.unique(sx, return_inverse=True)
    uy, iy = np.unique(sy, return_inverse=True)
    mx, my = np.meshgrid(ux * w - 0.5, uy * h - 0.5)
    cells = cv2.remap(np.ascontiguousarray(img[..., :3], dtype=F32), mx.astype(F32), my.astype(F32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(len(uy), len(ux), 3)
    lum = np.ascontiguousarray((cells.astype(np.float64) @ _W_LUM.astype(np.float64)).astype(F32))
    tyu = (ty * aspect + F32(0.5) - F32(0.5) * aspect).astype(F32)
    syu = (sy * aspect + F32(0.5) - F32(0.5) * aspect).astype(F32)
    return D._halftone(h, w, f, aspect, sx, tx, syu, tyu, ix.ravel().astype(np.int64),
                       iy.ravel().astype(np.int64), lum)


@itb_op("Crosshatch", "Crosshatch", GROUP, [
    Float("spacing", 0.01, min=0.001, max=0.05, step=0.0001, title="Spacing",
          description="Hatch spacing as a fraction of the image size"),
    Float("line_width", 0.003, min=0.001, max=0.02, step=0.0001, title="Line width",
          description="Hatch line width as a fraction of the image size"),
], upstream="GPUImage", tags=["crosshatch", "hatching", "sketch", "pen", "black and white"])
def crosshatch(img, ctx, spacing, line_width):
    """GPUImage crosshatch: up to four diagonal hatch layers below luminance 1.0 / 0.75 / 0.5 / 0.3."""
    h, w = img.shape[:2]
    tx, ty = _tex_coords(h, w)
    return D._crosshatch(img, tx, ty, F32(spacing), F32(line_width))


# ---------------------------------------------------------------------------
# G'MIC / aire
# ---------------------------------------------------------------------------

_MIXER = ["color_doping", "darken", "soft_light", "grain_merge", "multiply", "value"]


@itb_op("PosterizedDithering", "Posterized Dithering", GROUP, [
    Float("gamma", 1.0, min=0.01, max=5, step=0.01, title="Gamma"),
    Float("contrast", 1.0, min=0, max=4, step=0.01, title="Contrast"),
    Float("brightness", 0.0, min=-255, max=255, step=0.1, title="Brightness"),
    Float("smoothness", 0.0, min=0, max=10, step=0.01, unit="px", title="Smoothness",
          description="Blur applied to the luminance before black/white dithering"),
    Int("quantize_colors", 20, min=2, max=255, title="Quantize Colors"),
    Float("smooth_colors", 1.0, min=0, max=30, step=0.01, unit="px", title="Smooth Colors",
          description="Blur applied to the quantized colour layer"),
    Choice("mixer_mode", "color_doping", choices=_MIXER, title="Mixer Mode",
           labels={"color_doping": "Color Doping", "darken": "Darken", "soft_light": "Soft Light",
                   "grain_merge": "Grain Merge", "multiply": "Multiply", "value": "Value"}),
    Float("color_intensity", 1.0, min=0, max=1, step=0.01, title="Color Intensity"),
], upstream="G'MIC", requires=["gmic"], tags=["dithering", "posterize", "comic", "print"])
def posterized_dithering(img, ctx, gamma, contrast, brightness, smoothness, quantize_colors, smooth_colors,
                         mixer_mode, color_intensity):
    """G'MIC fx_pdithered: dithered B&W luminance composited over a quantized, blurred colour layer."""
    cmd = gmic_cmd("fx_pdithered", float(gamma), float(contrast), float(brightness), float(ctx.px(smoothness)),
                   int(quantize_colors), float(ctx.px(smooth_colors)), _MIXER.index(mixer_mode),
                   float(color_intensity))
    return run_gmic(img, cmd, ctx=ctx)


@itb_op("Quantizier", "Quantizier", GROUP, [
    Int("max_colors", 256, min=2, max=4096, title="Max color count",
        description="Maximum palette size of the Xiaolin Wu quantizer"),
], upstream="aire", tags=["quantize", "palette", "posterize", "reduce colors", "gif"])
def quantizier(img, ctx, max_colors):
    """aire palette(): Xiaolin Wu colour quantization, then nearest-colour remapping (no visible dithering)."""
    rgb = np.ascontiguousarray(img[..., :3], dtype=F32)
    return D.remap_nearest(rgb, D.wu_palette(rgb, max_colors))
