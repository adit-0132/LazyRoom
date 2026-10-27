"""JH Labs texture presets (ImageToolbox ``BrushedMetal``, ``Caustics``, ``Cellular``, ``Check``,
``FBM``, ``Marble``, ``Plasma``, ``Quilt``, ``Wood``).

ImageToolbox builds the ``com.jhlabs`` filter objects (``AndroidTextureGenerator.createFilter``)
and runs them through ImageToolboxLibs' Rust re-implementation (``libs/jhlabs``,
Apache-2.0) on a blank bitmap (filled with ``color`` for Cellular and Plasma,
transparent otherwise).  The kernels in :mod:`._jhlabs_impl` port that Rust code.
Parameters, ranges and defaults follow the ImageToolbox UI (``*Params.kt``) and
``TextureParams.withDefaultsFor``.

Seed: ``seed == 0`` reproduces ImageToolbox.  Otherwise the Perlin permutation seed
(``0x4a484c616273 + seed``), the brushed-metal ``java.util.Random`` seed (``seed``)
and the plasma ``seed`` field (``567 + seed``) are offset; Check and Quilt have no
random component and ignore it.
"""
from __future__ import annotations

import numpy as np

from ..core.params import Bool, Choice, Color, Float, Int
from . import _jhlabs_impl as K
from ._registry import TexturePreset, argb_hex, argb_to_rgba, color_argb, register

_UP = "ImageToolboxLibs jhlabs (Rust)"


def _out(packed: np.ndarray) -> np.ndarray:
    return argb_to_rgba(packed.astype(np.uint32))


def _perlin(seed: int):
    perm, g2, g3 = K.perlin_tables(K.PERLIN_SEED + int(seed))
    return perm, g2, g3


# ---------------------------------------------------------------------------

def _brushed_metal(w, h, seed, v):
    return _out(K.brushed_metal_kernel(w, h, color_argb(v["color"]), int(v["radius"]), float(v["amount"]),
                                       float(v["shine"]), bool(v["monochrome"]), int(seed)))


def _caustics(w, h, seed, v):
    perm, _, g3 = _perlin(seed)
    return _out(K.caustics_kernel(w, h, perm, g3, float(v["scale"]), float(v["amount"]), float(v["turbulence"]),
                                  float(v["time"]), int(v["brightness"]), color_argb(v["color"])))


_GRID = ["random", "square", "hexagonal", "octagonal", "triangular"]


def _cellular(w, h, seed, v):
    perm, g2, _ = _perlin(seed)
    return _out(K.cellular_kernel(w, h, perm, g2, K.cellular_probabilities(), color_argb(v["color"]),
                                  float(v["scale"]), float(v["stretch"]), float(v["angle"]), float(v["turbulence"]),
                                  float(v["f1"]), float(v["f2"]), float(v["randomness"]), _GRID.index(v["grid_type"]),
                                  float(v["distance_power"])))


def _check(w, h, seed, v):
    return _out(K.check_kernel(w, h, color_argb(v["light_color"]), color_argb(v["dark_color"]), int(v["scale_x"]),
                               int(v["scale_y"]), int(v["fuzziness"]), float(v["angle"])))


def _fbm(w, h, seed, v):
    perm, _, g3 = _perlin(seed)
    # FBMFilter has no getTurbulence(): the Rust port falls back to 4.0
    return _out(K.fbm_family_kernel(w, h, perm, g3, 0, float(v["scale"]), float(v["stretch"]), float(v["angle"]),
                                    4.0, 0.0, float(v["amount"]), 0, 0))


def _marble(w, h, seed, v):
    perm, _, g3 = _perlin(seed)
    return _out(K.fbm_family_kernel(w, h, perm, g3, 1, float(v["scale"]), float(v["stretch"]), float(v["angle"]),
                                    float(v["turbulence"]), float(v["turbulence_factor"]), 1.0,
                                    0xFF000000, 0xFFFFFFFF))


def _wood(w, h, seed, v):
    perm, _, g3 = _perlin(seed)
    return _out(K.fbm_family_kernel(w, h, perm, g3, 2, float(v["scale"]), float(v["stretch"]), float(v["angle"]),
                                    float(v["turbulence"]), float(v["rings"]), 1.0, 0xFFE5C494, 0xFF987B51))


def _plasma(w, h, seed, v):
    perm, _, g3 = _perlin(seed)
    return _out(K.plasma_kernel(w, h, perm, g3, float(v["turbulence"]), float(v["scaling"]), 567 + int(seed),
                                color_argb(v["color"])))


def _quilt(w, h, seed, v):
    return _out(K.quilt_kernel(w, h, int(v["iterations"]), float(v["a"]), float(v["b"]), float(v["c"]),
                               float(v["d"]), int(v["k"])))


# ---------------------------------------------------------------------------

def _angle(default: float = 0.0) -> Float:
    return Float("angle", default, min=-360, max=360, title="Angle",
                 description="Rotation; ImageToolbox's Rust port uses this value as radians")


def _scale() -> Float:
    return Float("scale", 32.0, min=1, max=512, title="Scale")


def _stretch() -> Float:
    return Float("stretch", 1.0, min=0.1, max=20, title="Stretch")


_SPECS = [
    ("brushed_metal", "Brushed Metal", "materials", "BrushedMetal", _brushed_metal, [
        Color("color", argb_hex(-7829368), title="Color"),
        Int("radius", 10, min=0, max=50, title="Radius", description="Horizontal box-blur radius in pixels"),
        Float("amount", 0.1, min=0, max=1, title="Amount", description="Noise amount"),
        Float("shine", 0.1, min=0, max=1, title="Shine"),
        Bool("monochrome", True, title="Monochrome"),
    ], "faithful",
        "Rust BrushedMetalFilter: java.util.Random(seed) noise + sine highlight, then box_blur_exact(radius, 0) "
        "including its fractional-pass quirk (first/last rows and columns are overwritten with the first row's "
        "corner pixels when radius > 0)."),
    ("caustics", "Caustics", "abstract", "Caustics", _caustics, [
        Color("color", argb_hex(-8806401), title="Color", description="Background colour"),
        _scale(),
        Int("brightness", 10, min=0, max=100, title="Brightness"),
        Float("amount", 1.0, min=0, max=10, title="Amount"),
        Float("turbulence", 1.0, min=0, max=10, title="Turbulence"),
        Float("dispersion", 0.0, min=0, max=10, title="Dispersion",
              description="No effect: ignored by ImageToolbox's Rust port"),
        Float("time", 0.0, min=0, max=10, title="Time"),
    ], "faithful",
        "Rust CausticsFilter (ridged 3D Perlin turbulence, not the original JH Labs ray-traced caustics). "
        "'dispersion' (and the hidden 'samples') have no effect upstream either.  Like upstream, the f32 mix "
        "truncates alpha to 254 on some pixels."),
    ("cellular", "Cellular", "abstract", "Cellular", _cellular, [
        Color("color", argb_hex(-1), title="Color"),
        _scale(), _stretch(), _angle(),
        Float("angle_coefficient", 0.0, min=-10, max=10, title="Angle coefficient",
              description="No visible effect (only feeds the colormap value, overridden by useColor)"),
        Float("gradient_coefficient", 0.0, min=-10, max=10, title="Gradient coefficient",
              description="No visible effect (only feeds the colormap value, overridden by useColor)"),
        Float("f1", 1.0, min=-5, max=5, title="F1"),
        Float("f2", 0.0, min=-5, max=5, title="F2"),
        Float("randomness", 0.0, min=0, max=1, title="Randomness"),
        Choice("grid_type", "hexagonal", choices=_GRID,
               labels={"random": "Random", "square": "Square", "hexagonal": "Hexagonal", "octagonal": "Octagonal",
                       "triangular": "Triangular"}, title="Grid type"),
        Float("distance_power", 2.0, min=0.1, max=10, title="Distance power"),
        Float("turbulence", 1.0, min=0, max=10, title="Turbulence"),
        Float("amount", 1.0, min=-10, max=10, title="Amount",
              description="No visible effect (only feeds the colormap value, overridden by useColor)"),
        Float("gain", 0.5, min=0, max=1, title="Gain", description="No effect: ignored by ImageToolbox's Rust port"),
        Float("bias", 0.5, min=0, max=1, title="Bias", description="No effect: ignored by ImageToolbox's Rust port"),
    ], "faithful",
        "Rust CellularFilter with useColor=true on a bitmap filled with 'color': output = mix(black, color, "
        "smooth_step(F2, F1, (d2-d1)/(d2+d1))).  amount, angle/gradient coefficients, gain and bias have no "
        "visible effect upstream (colormap branch is overridden); seed only matters with randomness > 0 on "
        "hexagonal/octagonal/triangular grids.  Like upstream, the f32 mix truncates alpha to 254 on some pixels."),
    ("check", "Checkerboard", "patterns", "Check", _check, [
        Color("light_color", argb_hex(-1), title="Light color"),
        Color("dark_color", argb_hex(-16777216), title="Dark color"),
        Int("scale_x", 8, min=1, max=128, title="Scale X"),
        Int("scale_y", 8, min=1, max=128, title="Scale Y"),
        Int("fuzziness", 0, min=0, max=128, title="Fuzziness"),
        _angle(),
    ], "faithful", "Rust CheckFilter; no randomness (seed ignored)."),
    ("fbm", "fBm", "abstract", "FBM", _fbm, [
        Float("amount", 1.0, min=0.01, max=10, title="Amount"),
        _scale(), _stretch(), _angle(),
    ], "faithful",
        "Rust FBMFilter: |3D Perlin| turbulence with 4 octaves (FBMFilter has no getTurbulence -> Rust default "
        "4.0) through GrayscaleColormap, mixed with the transparent source by 'amount' (alpha < 1 when "
        "amount < 1).  Hidden octaves/H/lacunarity/gain/bias are ignored upstream."),
    ("marble", "Marble", "abstract", "Marble", _marble, [
        _scale(), _stretch(), _angle(),
        Float("turbulence", 1.0, min=0, max=10, title="Turbulence"),
        Float("turbulence_factor", 0.4, min=0, max=5, title="Turbulence factor"),
    ], "faithful", "Rust MarbleTexFilter: triangle(nx + factor * turbulence3) through LinearColormap(black, white)."),
    ("plasma", "Plasma", "abstract", "Plasma", _plasma, [
        Color("color", argb_hex(-16777216), title="Color"),
        Float("turbulence", 1.0, min=0, max=10, title="Turbulence"),
        Float("scaling", 0.0, min=0, max=1, title="Scaling"),
    ], "faithful",
        "Rust PlasmaFilter (Perlin turbulence, not the original diamond-square plasma) with useImageColors: "
        "mix(color, grey(value), value)."),
    ("quilt", "Quilt", "patterns", "Quilt", _quilt, [
        Int("iterations", 25000, min=1, max=50000, title="Iterations"),
        Float("a", -0.59, min=-2, max=2, title="A"),
        Float("b", 0.2, min=-2, max=2, title="B"),
        Float("c", 0.1, min=-2, max=2, title="C"),
        Float("d", 0.0, min=-2, max=2, title="D"),
        Int("k", 0, min=0, max=20, title="K"),
    ], "faithful",
        "Rust QuiltFilter (chaotic quilt attractor hit counts through LinearColormap); no randomness "
        "(seed ignored).  The map is chaotic, so exact hit positions depend on the platform's sinf rounding "
        "(same attractor/density, not bit-identical to a given device)."),
    ("wood", "Wood", "abstract", "Wood", _wood, [
        Float("rings", 2.0, min=0, max=5, title="Rings"),
        Float("turbulence", 1.0, min=0, max=10, title="Turbulence"),
        Float("gain", 0.5, min=0, max=1, title="Gain", description="No effect: ignored by ImageToolbox's Rust port"),
        Float("bias", 0.5, min=0, max=1, title="Bias", description="No effect: ignored by ImageToolbox's Rust port"),
        _scale(), _stretch(), _angle(),
    ], "faithful",
        "Rust WoodFilter: triangle(sqrt(nx^2+ny^2) + rings * turbulence3) through LinearColormap(#E5C494, "
        "#987B51); gain/bias have no effect upstream."),
]

for _id, _title, _cat, _up, _fn, _params, _status, _notes in _SPECS:
    register(TexturePreset(id=_id, title=_title, category=_cat, backend="jhlabs", upstream=_up, params=_params,
                           render_fn=_fn, status=_status, notes=_notes,
                           description=f"ImageToolbox texture '{_title}' ({_UP} {_up}Filter)."))
