"""ImageToolbox GLSL pattern textures (28 presets, backend ``pattern``).

Port of ImageToolbox ``PatternTextureType.kt`` (parameters, ranges, defaults,
default colours), ``PatternParams.kt`` (UI titles) and ``PatternTextureShaders.kt``
(GLSL bodies, see :mod:`._patterns_impl`), Apache-2.0, T8RIN.

Coordinates follow the GPUImage pipeline: pixel centres, texture y growing
downwards in the image, ``p = rotate((tc - .5) * aspect - offset) * frequency``.
Seed: presets with an upstream "seed" parameter use ``seed % 10001`` from the
render seed (0 = ImageToolbox default); the others ignore it.
"""
from __future__ import annotations

import numpy as np

from ..core.params import Color, Float, Int
from ._registry import TexturePreset, argb_hex, register

_DEFAULT_COLORS = [-16777216, -1, -8355712, -16711681]
_COLORS = {
    "PlaidPattern": [-256, -65536, -65281, -16711681],
    "GeneratedWaves": [-256, -65536, -65281, -16711681],
    "CrossStitchPattern": [-16777216, -2767701, -13338717, -1],
    "ScintillatingGrid": [-16777216, -8355712, -1, -1],
    "RandomTriangles": [-65536, -256, -16711681, -65281],
    "VoronoiGems": [-65536, -256, -16711681, -65281],
    "HexCubePattern": [-65536, -256, -16711681, -16777216],
    "LightArray": [-16777216, -256, -16711681, -65281],
    "CitySkyline": [-16117736, -9424331, -32768, -256],
    "WinterLandscape": [-16117736, -10053172, -14587077, -1],
    "CircleRipplePattern": [-256, -16777216, -1, -16776961],
    "SquareRipplePattern": [-16711681, -65281, -1, -16777216],
}

_TITLES = {  # PatternParams.kt string resources (English)
    "hardness": "Hardness", "shapeAspect": "Shape aspect ratio", "variability": "Variability",
    "thickness": "Thickness", "balance": "Balance", "intensity": "Intensity", "count": "Count",
    "shadows": "Shadows", "radius": "Radius", "offset": "Offset", "layers": "Layers", "time": "Time",
}

S = ("seed", 0.0, 0.0, 10000.0, True)  # handled by the render seed

# enum name, id, title, colorCount, defaultFrequency, [(name, default, min, max, integer)]
_SPECS = [
    ("RandomTriangles", "random_triangles", "Random triangles", 4, 1.0, [("count", 40, 1, 100, True), S]),
    ("HexCubePattern", "hex_cube_pattern", "Hex cubes", 4, 8.0, [("thickness", 0.03, 0.0, 0.3, False)]),
    ("SquareLooper", "square_looper", "Square looper", 2, 8.0,
     [("time", 0.0, 0.0, 10.0, False), ("thickness", 0.15, 0.01, 0.8, False)]),
    ("MoireInterference", "moire_interference", "Moire interference", 2, 1.0,
     [("intensity", 30.0, 1.0, 100.0, False), ("thickness", 0.5, 0.05, 0.95, False)]),
    ("TruchetArcs", "truchet_arcs", "Truchet arcs", 2, 8.0, [S, ("thickness", 0.15, 0.02, 0.45, False)]),
    ("CircuitPattern", "circuit_pattern", "Circuit pattern", 2, 8.0,
     [S, ("thickness", 0.05, 0.01, 0.25, False), ("radius", 0.13, 0.03, 0.3, False)]),
    ("RosacePattern", "rosace_pattern", "Rosaces", 2, 1.0,
     [("count", 24, 3, 64, True), ("radius", 0.4, 0.05, 0.8, False), ("thickness", 0.02, 0.005, 0.15, False)]),
    ("RosettePattern", "rosette_pattern", "Rosette", 2, 1.0,
     [("count", 24, 3, 64, True), ("radius", 0.3, 0.05, 0.8, False), ("thickness", 0.01, 0.005, 0.15, False)]),
    ("LightArray", "light_array", "Light array", 4, 4.0,
     [("intensity", 0.7, 0.0, 2.0, False), ("layers", 3, 1, 6, True), S]),
    ("VoronoiGems", "voronoi_gems", "Voronoi gems", 4, 8.0,
     [S, ("shadows", 0.5, 0.0, 1.0, False), ("variability", 1.0, 0.0, 1.0, False)]),
    ("CitySkyline", "city_skyline", "City skyline", 4, 1.0, [("layers", 4, 1, 8, True), S]),
    ("WinterLandscape", "winter_landscape", "Winter landscape", 4, 1.0, [("time", 0.0, 0.0, 10.0, False), S]),
    ("CircleRipplePattern", "circle_ripple_pattern", "Circle ripple illusion", 4, 8.0,
     [("radius", 0.4, 0.1, 0.49, False), ("shapeAspect", 0.7, 0.25, 2.0, False)]),
    ("SquareRipplePattern", "square_ripple_pattern", "Square ripple illusion", 4, 8.0,
     [("thickness", 0.08, 0.01, 0.3, False), ("offset", 0.08, 0.0, 0.3, False)]),
    ("HatchPattern", "hatch_pattern", "Hatch", 2, 12.0, [("thickness", 0.5, 0.02, 0.98, False)]),
    ("SmoothHatch", "smooth_hatch", "Smooth hatch", 2, 12.0,
     [("hardness", 1.0, 0.1, 8.0, False), ("balance", 0.5, 0.01, 0.99, False)]),
    ("XorPattern", "xor_pattern", "XOR pattern", 2, 12.0, [("intensity", 4, 1, 16, True)]),
    ("SquareSpiralPattern", "square_spiral_pattern", "Square spiral", 2, 5.0,
     [("count", 4, 1, 12, True), ("thickness", 0.4, 0.05, 0.95, False)]),
    ("CrossStitchPattern", "cross_stitch_pattern", "Cross stitch", 3, 12.0,
     [("thickness", 0.65, 0.1, 0.95, False), ("shadows", 0.6, 0.0, 1.0, False)]),
    ("PlaidPattern", "plaid_pattern", "Plaid", 4, 12.0, [S, ("thickness", 0.45, 0.05, 0.95, False)]),
    ("GeneratedWaves", "generated_waves", "Generated waves", 4, 12.0, [("variability", 0.5, 0.0, 1.0, False), S]),
    ("VoronoiHatch", "voronoi_hatch", "Voronoi hatch", 2, 12.0,
     [S, ("thickness", 0.45, 0.05, 0.95, False), ("variability", 1.0, 0.0, 1.0, False)]),
    ("LoopPattern", "loop_pattern", "Loops", 2, 1.0,
     [("count", 8, 1, 64, True), ("radius", 0.15, 0.01, 0.5, False), ("thickness", 0.1, 0.01, 0.5, False), S]),
    ("ScintillatingGrid", "scintillating_grid", "Scintillating grid", 3, 12.0,
     [("thickness", 0.08, 0.01, 0.4, False), ("radius", 0.09, 0.01, 0.4, False)]),
    ("CircleSpiralPattern", "circle_spiral_pattern", "Circle spiral", 2, 1.0, [("count", 5, 1, 24, True)]),
    ("TestChart", "test_chart", "Test chart", 2, 1.0, []),
    ("BentRows", "bent_rows", "Bent rows", 3, 12.0,
     [("offset", 0.2, 0.0, 0.5, False), ("thickness", 0.04, 0.005, 0.15, False)]),
    ("DiamondsIllusion", "diamonds_illusion", "Diamonds illusion", 2, 4.0,
     [("count", 5, 1, 16, True), ("thickness", 0.5, 0.05, 0.95, False)]),
]

_NOTES = {
    "TestChart": "colour parameters only (no shape parameters upstream)",
}
_SEED_NOTE = "upstream 'Random seed' parameter is driven by the render seed (seed % 10001)"
_HASH_NOTE = ("hash() = fract(sin(dot)*43758.5453) evaluated in float64; GPU float32 sin is implementation-"
              "defined, so the random layout cannot match a given GPU bit-exactly")
_RANDOM = {"RandomTriangles", "TruchetArcs", "CircuitPattern", "LightArray", "VoronoiGems", "CitySkyline",
           "WinterLandscape", "PlaidPattern", "GeneratedWaves", "VoronoiHatch", "LoopPattern"}


def _snake(name: str) -> str:
    return "".join("_" + c.lower() if c.isupper() else c for c in name)


def _rgba01(argb: int) -> list[float]:
    v = argb & 0xFFFFFFFF
    return [((v >> 16) & 0xFF) / 255.0, ((v >> 8) & 0xFF) / 255.0, (v & 0xFF) / 255.0, (v >> 24) / 255.0]


def _make(kind: int, enum: str, pid: str, title: str, ncolors: int, freq: float, spec: list) -> TexturePreset:
    defaults = _COLORS.get(enum, _DEFAULT_COLORS)
    shader_params = [s for s in spec if s[0] != "seed"]
    has_seed = any(s[0] == "seed" for s in spec)
    params = [Color(f"color{i + 1}", argb_hex(defaults[i]), title=f"Color {i + 1}") for i in range(ncolors)]
    for name, default, lo, hi, integer in shader_params:
        if integer:
            params.append(Int(_snake(name), int(default), title=_TITLES[name], min=int(lo), max=int(hi)))
        else:
            params.append(Float(_snake(name), float(default), title=_TITLES[name], min=float(lo), max=float(hi)))
    params += [
        Float("frequency", freq, title="Frequency", min=0.1, max=64.0),
        Float("offset_x", 0.0, title="Offset X", min=-1.0, max=1.0),
        Float("offset_y", 0.0, title="Offset Y", min=-1.0, max=1.0),
        Float("rotation", 0.0, title="Rotation", min=-180.0, max=180.0, unit="deg"),
    ]
    hidden = [_rgba01(defaults[i] if i < len(defaults) else -1) for i in range(4)]

    def render_fn(width: int, height: int, seed: int, v: dict) -> np.ndarray:
        from ._patterns_impl import prepare_aux, render_pattern

        m = float(min(width, height))
        u = [width / m, height / m, v["offset_x"], v["offset_y"], v["rotation"], v["frequency"],
             float(seed % 10001) if has_seed else 0.0]
        u += [float(v[_snake(s[0])]) for s in shader_params]
        u_arr = np.asarray(u, np.float64)
        col = np.array([list(v[f"color{i + 1}"]) if i < ncolors else hidden[i] for i in range(4)], np.float64)
        aux = prepare_aux(kind, u_arr)
        return render_pattern(kind, int(width), int(height), u_arr, col, aux)

    notes = [n for n in (_NOTES.get(enum), _SEED_NOTE if has_seed else None, _HASH_NOTE if enum in _RANDOM else None)
             if n]
    return TexturePreset(id=pid, title=title, category="patterns", backend="pattern", upstream=enum,
                         params=params, render_fn=render_fn, status="faithful", notes="; ".join(notes),
                         description=f"ImageToolbox GLSL pattern '{title}'")


def _register_all() -> None:
    # kind index = position in _SPECS (same order as _patterns_impl.KINDS)
    for kind, (enum, pid, title, ncolors, freq, spec) in enumerate(_SPECS):
        register(_make(kind, enum, pid, title, ncolors, freq, spec))


_register_all()
