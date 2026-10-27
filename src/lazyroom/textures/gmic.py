"""G'MIC texture presets (ImageToolbox ``OrganicFibers``, ``GmicReactionDiffusion``, ``Truchet``).

ImageToolbox runs these G'MIC commands (ImageToolboxLibs ``gmic/filters/TextureFilters.kt``,
Apache-2.0) on a black, opaque canvas of the requested size
(``AndroidTextureGenerator``: ``createBitmap(w, h).eraseColor(BLACK)`` ->
``Gmic.runCancellable``).  The command strings below are identical (same argument
order, enum ``.value`` integers, Kotlin ``Float.toString`` values); they run
through the external G'MIC CLI (CeCILL).

Alpha handling follows the upstream ``GmicOptions``: OrganicFibers uses the default
``Preserve`` mode (RGB processed, canvas alpha kept), ReactionDiffusion and
Truchet use ``gmicProcessAlpha`` (RGBA passed to G'MIC, its output alpha kept).

Seed: ImageToolbox does not seed G'MIC's random generator (every run differs); here
the command is prefixed with ``srand <seed>`` and evaluated single-threaded, so a
given seed always yields the same texture.
"""
from __future__ import annotations

import re

import numpy as np

from ..core.params import Choice, Float, Int
from ._registry import TexturePreset, register

_ORGANIC_PALETTES = [
    "Default", "Hsv", "Lines", "Hot", "Cool", "Jet", "Flag", "Cube", "Rainbow", "Parula", "Spring", "Summer",
    "Autumn", "Winter", "Bone", "Copper", "Pink", "Vga", "Algae", "Amp", "Balance", "Curl", "Deep", "Delta",
    "Dense", "Diff", "Gray", "Haline", "Ice", "Matter", "Oxy", "Phase", "Rain", "Solar", "Speed", "Tarn", "Tempo",
    "Thermal", "Topo", "Turbid", "Aurora", "Hocuspocus", "Srb2", "Uzebox", "Amiga7800", "Amiga7800Mess",
    "FornaxVoid1",
]
_RD_MODES = ["Monochrome", "MonochromeWithAlpha", "Rgb", "RgbWithAlpha"]
_TRUCHET_TYPES = ["Straight", "Curved"]
_TRUCHET_COLOR_MODES = ["WhiteOnBlack", "BlackOnWhite", "WhiteOnTransparent", "BlackOnTransparent",
                        "TransparentOnWhite", "TransparentOnBlack", "Random"]


def _snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _enum_choice(name: str, entries: list[str], default: str, title: str) -> Choice:
    """Upstream enum -> Choice (snake_case values in upstream order; value index == enum ``.value``)."""
    choices = [_snake(e) for e in entries]
    labels = {c: re.sub(r"([a-z])([A-Z])", r"\1 \2", e) for c, e in zip(choices, entries)}
    return Choice(name, _snake(default), choices=choices, labels=labels, title=title)


def _kfloat(v: float) -> str:
    """Kotlin ``Float.toString`` value (shortest float32 repr), written positionally."""
    return np.format_float_positional(np.float32(v), unique=True, trim="-")


def _run(w: int, h: int, seed: int, command: str, preserve_alpha: bool) -> np.ndarray:
    from ..ops.itb._common import gmic_single_thread, run_gmic  # lazy: avoids importing the ops package early

    canvas = np.zeros((h, w, 4), np.float32)
    canvas[..., 3] = 1.0  # Bitmap.eraseColor(Color.BLACK): opaque black
    with gmic_single_thread():
        out = run_gmic(canvas, f"srand {int(seed)} {command}", preserve_alpha=preserve_alpha)
    if out.shape[:2] != (h, w):
        import cv2

        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def _cmd(name: str, *args) -> str:
    from ..ops.itb._common import gmic_cmd

    return gmic_cmd(name, *[_kfloat(a) if isinstance(a, float) else a for a in args])


def _organic_fibers(w, h, seed, v):
    cmd = _cmd("fx_organic_fibers", float(v["agent_density"]), int(v["iterations"]), int(v["orientations"]),
               int(v["sensor_distance"]), float(v["sensor_angle"]), int(v["motion_distance"]),
               float(v["motion_angle"]), float(v["motion_moment"]), float(v["trail_blur"]), int(v["particle_size"]),
               float(v["particle_thickness"]), int(v["quantized_orientations"]), float(v["opacity"]),
               float(v["sharpening"]), [_snake(e) for e in _ORGANIC_PALETTES].index(v["palette"]))
    return _run(w, h, seed, cmd, preserve_alpha=True)


def _reaction_diffusion(w, h, seed, v):
    cmd = _cmd("fx_reaction_diffusion", int(v["iterations"]), float(v["size"]),
               [_snake(e) for e in _RD_MODES].index(v["mode"]))
    return _run(w, h, seed, cmd, preserve_alpha=False)


def _truchet(w, h, seed, v):
    cmd = _cmd("fx_truchet", int(v["scale"]), int(v["radius"]), float(v["smoothness"]),
               [_snake(e) for e in _TRUCHET_TYPES].index(v["type"]),
               [_snake(e) for e in _TRUCHET_COLOR_MODES].index(v["color_mode"]))
    return _run(w, h, seed, cmd, preserve_alpha=False)


_SPECS = [
    ("organic_fibers", "Organic Fibers", "abstract", "OrganicFibers", _organic_fibers, [
        Float("agent_density", 20.0, min=0, max=50, title="Agent Density"),
        Int("iterations", 20, min=1, max=256, title="Iterations"),
        Int("orientations", 3, min=2, max=16, title="Orientation"),
        Int("sensor_distance", 15, min=0, max=400, title="Sensor Distance"),
        Float("sensor_angle", 15.0, min=0, max=180, title="Sensor Angle"),
        Int("motion_distance", 3, min=0, max=400, title="Motion Distance"),
        Float("motion_angle", 15.0, min=0, max=180, title="Motion Angle"),
        Float("motion_moment", 30.0, min=0, max=100, title="Motion Moment"),
        Float("trail_blur", 1.0, min=0, max=20, title="Trail Blur"),
        Int("particle_size", 19, min=1, max=128, title="Particle Size"),
        Float("particle_thickness", 10.0, min=0, max=100, title="Particle Thickness"),
        Int("quantized_orientations", 24, min=1, max=32, title="Quantized Orientations"),
        Float("opacity", 50.0, min=0, max=100, title="Opacity"),
        Float("sharpening", 0.0, min=0, max=100, title="Sharpen"),
        _enum_choice("palette", _ORGANIC_PALETTES, "Solar", "Palette"),
    ], "fx_organic_fibers"),
    ("gmic_reaction_diffusion", "Reaction Diffusion (G'MIC)", "abstract", "GmicReactionDiffusion",
     _reaction_diffusion, [
        Int("iterations", 5, min=1, max=10, title="Iterations"),
        Float("size", 5.0, min=0, max=20, title="Size"),
        _enum_choice("mode", _RD_MODES, "Monochrome", "Mode"),
    ], "fx_reaction_diffusion"),
    ("truchet", "Truchet", "patterns", "Truchet", _truchet, [
        Int("scale", 32, min=1, max=256, title="Scale"),
        Int("radius", 5, min=1, max=64, title="Radius"),
        Float("smoothness", 1.0, min=0, max=10, title="Smoothness"),
        _enum_choice("type", _TRUCHET_TYPES, "Curved", "Type"),
        _enum_choice("color_mode", _TRUCHET_COLOR_MODES, "WhiteOnBlack", "Color Mode"),
    ], "fx_truchet"),
]

_NOTES = {
    "organic_fibers": "Exact ImageToolboxLibs command on a black canvas (Preserve alpha mode -> opaque). "
                      "Seeded with 'srand <seed>' + single-threaded G'MIC (upstream is unseeded).",
    "gmic_reaction_diffusion": "Exact command on a black RGBA canvas, G'MIC alpha kept (gmicProcessAlpha). "
                               "Title suffixed '(G'MIC)' to disambiguate from the fast-noise 'Reaction Diffusion' "
                               "(upstream shows both as 'Reaction Diffusion'). Seeded with 'srand <seed>'.",
    "truchet": "Exact command on a black RGBA canvas, G'MIC alpha kept (gmicProcessAlpha; transparent colour "
               "modes return alpha). Seeded with 'srand <seed>' (upstream is unseeded).",
}

for _id, _title, _cat, _up, _fn, _params, _gmic in _SPECS:
    register(TexturePreset(id=_id, title=_title, category=_cat, backend="gmic", upstream=_up, params=_params,
                           render_fn=_fn, requires=["gmic"], status="faithful", notes=_NOTES[_id],
                           description=f"ImageToolbox texture '{_title}' (G'MIC {_gmic})."))
