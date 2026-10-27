"""Fractal type catalogue: ImageToolbox ``FractalFormula`` defaults + fractal-engine ``FractalType`` ids.

Sources (Apache-2.0):
  * ImageToolbox ``feature/fractal-generation/.../domain/model/FractalFormula.kt`` (UI defaults,
    capabilities, coefficient specs), ``FractalParams.kt``, ``FractalCamera.kt``,
    ``FractalQuaternion.kt`` (ranges / normalisation).
  * ImageToolboxLibs ``libs/fractal-engine`` ``FractalType.kt`` / ``FractalEngineBridge.kt``
    (stable ids, engine-side validation).

``HTree`` and ``HilbertCurve`` exist in the native engine but are not exposed by the ImageToolbox
UI (``itb_ui=False``); their defaults come from the engine's ``FractalType`` enum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Capabilities (FractalFormulaCapability)
POWER, ITER, BAILOUT, JULIA, PHOENIX, NOVA, DEEP, CAMERA, QUAT, GEOM, DENSITY = (
    "power", "iteration_controls", "bailout", "julia_constant", "phoenix_constant", "nova_relaxation",
    "deep_zoom", "camera_3d", "quaternion_constant", "geometric", "density_visualization")

# Engine families (how fractal_engine renders a type)
ESCAPE, DENSITY2D, RAYMARCH, ATTRACTOR3D = "escape_time", "density_2d", "ray_marched_3d", "attractor_3d"

DEFAULT_ITERATIONS = 320
DEFAULT_BAILOUT = 4.0
DEFAULT_YAW, DEFAULT_PITCH, DEFAULT_DISTANCE = -35.0, 20.0, 3.5


@dataclass(frozen=True)
class Coefficient:
    slot: str          # a | b | c | d  (engine wire: a->julia re, b->julia im, c->power, d->phoenix re)
    label: str         # option key (ImageToolbox FractalCoefficientLabel, snake_case)
    title: str
    default: float
    min: float
    max: float


@dataclass(frozen=True)
class FractalSpec:
    key: str
    kotlin: str
    stable_id: int
    title: str
    family: str
    center: tuple[str, str] = ("-0.5", "0")
    span: str = "3"
    power: float = 2.0
    iterations: int = DEFAULT_ITERATIONS
    bailout: float = DEFAULT_BAILOUT
    camera: tuple[float, float, float] = (DEFAULT_YAW, DEFAULT_PITCH, DEFAULT_DISTANCE)
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    julia: tuple[float, float] = (0.0, 0.0)
    phoenix: tuple[float, float] = (0.0, 0.0)
    coefficients: tuple[Coefficient, ...] = ()
    caps: frozenset[str] = field(default_factory=frozenset)
    itb_ui: bool = True

    @property
    def three_d(self) -> bool:
        return self.family in (RAYMARCH, ATTRACTOR3D)

    @property
    def is_density(self) -> bool:
        return DENSITY in self.caps

    @property
    def supports_floor(self) -> bool:
        return self.three_d and not self.is_density

    @property
    def default_span(self) -> float:
        return float(self.span)


def _c(slot, label, default, lo, hi, title=None):
    return Coefficient(slot, label, title or label.upper() if len(label) == 1 else (title or label), default, lo, hi)


def _abc(a, b, c, lo=-100.0, hi=100.0):  # martinVariantCoefficients
    return (_c("a", "a", a, lo, hi), _c("b", "b", b, lo, hi), _c("c", "c", c, lo, hi))


def _abcd(a, b, c, d, lo, hi):  # fourCoefficientAttractor
    return (_c("a", "a", a, lo, hi), _c("b", "b", b, lo, hi), _c("c", "c", c, lo, hi), _c("d", "d", d, lo, hi))


_ESC = frozenset({ITER, BAILOUT})            # escapeTimeCapabilities()
_ESC_J = frozenset({ITER, BAILOUT, JULIA})   # escapeTimeCapabilities(julia = true)
_PIB = frozenset({POWER, ITER, BAILOUT})
_DENS = frozenset({ITER, DENSITY})           # densityCapabilities()
_DENS_G = frozenset({ITER, DENSITY, GEOM})   # densityCapabilities(geometric = true)
_J0 = (-0.8, 0.156)
_SCALE, _FOLD, _MINR = "scale", "fold", "minimum_radius"


def _specs() -> list[FractalSpec]:
    S = FractalSpec
    E, D, R, A = ESCAPE, DENSITY2D, RAYMARCH, ATTRACTOR3D
    return [
        S("mandelbrot", "Mandelbrot", 1, "Mandelbrot", E, ("-0.5", "0"), "3", caps=_PIB | {DEEP}),
        S("julia", "Julia", 2, "Julia", E, ("0", "0"), "3", julia=_J0, caps=_PIB | {JULIA, DEEP}),
        S("burning_ship", "BurningShip", 3, "Burning ship", E, ("-0.45", "-0.5"), "2.4", caps=_PIB),
        S("tricorn", "Tricorn", 4, "Tricorn", E, ("0", "0"), "3.2", caps=_PIB),
        S("multibrot", "Multibrot", 5, "Multibrot", E, ("0", "0"), "3", power=3.0, caps=_PIB),
        S("multicorn", "Multicorn", 6, "Multicorn", E, ("0", "0"), "3.2", power=3.0, caps=_PIB),
        S("celtic", "Celtic", 7, "Celtic", E, ("-0.4", "0"), "3", caps=_PIB),
        S("buffalo", "Buffalo", 8, "Buffalo", E, ("-0.5", "0"), "3", caps=_PIB),
        S("perpendicular_burning_ship", "PerpendicularBurningShip", 9, "Perpendicular burning ship", E,
          ("-0.5", "-0.25"), "3", caps=_PIB),
        S("phoenix", "Phoenix", 10, "Phoenix", E, ("0", "0"), "3.2", phoenix=(-0.5, 0.0), caps=_PIB | {PHOENIX}),
        S("nova", "Nova", 11, "Nova", E, ("0", "0"), "4", power=3.0, caps=_PIB | {NOVA}),
        S("newton", "Newton", 12, "Newton", E, ("0", "0"), "4", power=3.0, caps=_PIB),
        S("magnet_i", "MagnetI", 101, "Magnet I", E, ("1.5", "0"), "4", caps=_ESC),
        S("magnet_ii", "MagnetII", 102, "Magnet II", E, ("1.0", "0"), "4", caps=_ESC),
        S("lyapunov", "Lyapunov", 103, "Lyapunov", E, ("3.0", "3.0"), "2", caps=frozenset({ITER})),
        S("sierpinski_carpet", "SierpinskiCarpet", 104, "Sierpinski carpet", E, ("0", "0"), "2.2",
          caps=frozenset({GEOM})),
        S("sierpinski_triangle", "SierpinskiTriangle", 105, "Sierpinski triangle", E, ("0", "0"), "2.2",
          caps=frozenset({GEOM})),
        S("burning_ship_julia", "BurningShipJulia", 106, "Burning Ship Julia", E, ("0", "0"), "3", julia=_J0,
          caps=_PIB | {JULIA}),
        S("celtic_julia", "CelticJulia", 107, "Celtic Julia", E, ("0", "0"), "3", julia=_J0, caps=_PIB | {JULIA}),
        S("collatz", "Collatz", 108, "Collatz", E, ("0", "0"), "4", iterations=160, bailout=100.0, caps=_ESC),
        S("buddhabrot", "Buddhabrot", 109, "Buddhabrot", D, ("0.4", "0"), "4.444444444444444", iterations=800,
          bailout=2.0, caps=_DENS),
        S("hopalong", "Hopalong", 110, "Hopalong", D, ("-0.1", "0.5"), "4.2", iterations=1000,
          coefficients=(_c("a", "a", 0.4, -10, 10), _c("b", "b", 1.0, -10, 10), _c("c", "c", 0.0, -10, 10)),
          caps=_DENS),
        S("martin", "Martin", 111, "Martin", D, ("1.5707963267948966", "1.5707963267948966"), "145",
          iterations=1000, coefficients=(_c("a", "a", math.pi, -10, 10),), caps=_DENS),
        S("gingerbreadman", "Gingerbreadman", 112, "Gingerbread man", D, ("2.5", "2.5"), "12", iterations=1000,
          caps=_DENS),
        S("chip", "Chip", 113, "Chip", D, ("-7.2", "-7.9"), "720", iterations=1000,
          coefficients=_abc(-15.0, -19.0, 1.0), caps=_DENS),
        S("quadruptwo", "Quadruptwo", 114, "Quadruptwo", D, ("16.5", "17.5"), "200", iterations=1000,
          coefficients=_abc(34.0, 1.0, 5.0), caps=_DENS),
        S("threeply", "Threeply", 115, "Threeply", D, ("-34", "-21"), "5500", iterations=1000,
          coefficients=_abc(-55.0, -1.0, -42.0), caps=_DENS),
        S("clifford", "Clifford", 116, "Clifford attractor", D, ("0", "0"), "4", power=1.0, iterations=1000,
          coefficients=_abcd(-1.4, 1.6, 1.0, 0.7, -3.0, 3.0), caps=_DENS),
        S("de_jong", "DeJong", 117, "De Jong attractor", D, ("0", "0"), "4.5", power=-0.65, iterations=1000,
          coefficients=_abcd(-2.24, 0.43, -0.65, -2.43, -3.0, 3.0), caps=_DENS),
        S("ikeda", "Ikeda", 118, "Ikeda attractor", D, ("1", "0"), "3", power=0.4, iterations=1000,
          coefficients=(_c("a", "a", 0.85, 0, 2), _c("b", "b", 0.9, 0, 1), _c("c", "c", 0.4, -10, 10),
                        _c("d", "d", 7.7, 0, 20)), caps=_DENS),
        S("tinkerbell", "Tinkerbell", 119, "Tinkerbell map", D, ("-0.5", "-0.25"), "3.5", iterations=1000,
          coefficients=_abcd(0.9, -0.6013, 2.0, 0.5, -5.0, 5.0), caps=_DENS),
        S("gumowski_mira", "GumowskiMira", 120, "Gumowski–Mira", D, ("0", "0"), "28", iterations=1000,
          coefficients=(_c("a", "a", 0.008, -1, 1), _c("b", "b", 0.05, -1, 1)), caps=_DENS),
        S("barnsley_fern", "BarnsleyFern", 121, "Barnsley fern", D, ("0", "5"), "11", iterations=1000, caps=_DENS_G),
        S("ifs_dragon", "IFSDragon", 122, "IFS dragon", D, ("0", "4"), "12", iterations=1000, caps=_DENS_G),
        S("ifs_twig", "IFSTwig", 123, "IFS twig", D, ("0.5", "0.5"), "1.2", iterations=1000, caps=_DENS_G),
        S("christmas_tree", "ChristmasTree", 124, "Christmas tree", D, ("0.5", "0.5"), "1.2", iterations=1000,
          caps=_DENS_G),
        S("vicsek_cross", "VicsekCross", 125, "Vicsek cross", D, ("0", "0"), "2.2", iterations=1000, caps=_DENS_G),
        S("pythagoras_tree", "PythagorasTree", 126, "Pythagoras tree", D, ("0", "1.25"), "5", iterations=1000,
          caps=_DENS_G),
        S("h_tree", "HTree", 127, "H-tree", D, ("0", "0"), "2.2", iterations=1000, caps=_DENS_G, itb_ui=False),
        S("heighway_dragon", "HeighwayDragon", 128, "Heighway dragon", D, ("0", "0"), "2.2", iterations=1000,
          caps=_DENS_G),
        S("koch_snowflake", "KochSnowflake", 129, "Koch snowflake", D, ("0", "0"), "2.2", iterations=1000,
          caps=_DENS_G),
        S("hilbert_curve", "HilbertCurve", 130, "Hilbert curve", D, ("0", "0"), "2.2", iterations=1000,
          caps=_DENS_G, itb_ui=False),
        S("barnsley_mandelbrot", "BarnsleyMandelbrot", 131, "Barnsley Mandelbrot", E, ("0", "0"), "4", caps=_ESC),
        S("barnsley_julia", "BarnsleyJulia", 132, "Barnsley Julia", E, ("0", "0"), "6", julia=(1.1, 0.6),
          caps=_ESC_J),
        S("alpha_mandelbrot", "AlphaMandelbrot", 133, "Alpha Mandelbrot", E, ("-0.5", "0"), "3", caps=_ESC),
        S("alpha_mandelbrot_julia", "AlphaMandelbrotJulia", 134, "Alpha Mandelbrot Julia", E, ("0", "0"), "3",
          julia=(0.36228, -0.0777), caps=_ESC_J),
        S("mandelbrot_sine", "MandelbrotSine", 135, "Mandelbrot sine", E, ("0", "0"), "4", caps=_ESC),
        S("julia_sine", "JuliaSine", 136, "Julia sine", E, ("0", "0"), "4", julia=(1.0, 0.1), caps=_ESC_J),
        S("spider", "Spider", 137, "Spider", E, ("-0.5", "0"), "3", caps=_ESC),
        S("man_o_war", "ManOWar", 138, "Man-o’-War", E, ("-0.5", "0"), "3", caps=_ESC),
        S("lambda", "Lambda", 139, "Lambda", E, ("0", "0"), "4", caps=_ESC),
        S("thorn", "Thorn", 140, "Thorn", E, ("0", "0"), "3.5", iterations=500, bailout=10_000.0,
          coefficients=(_c("a", "a", 0.1, -2, 2), _c("b", "b", -0.1, -2, 2)), caps=_ESC),
        S("barnsley_ii", "BarnsleyII", 141, "Barnsley II", E, ("0", "0"), "4", julia=(1.1, 0.6), caps=_ESC_J),
        S("barnsley_iii", "BarnsleyIII", 142, "Barnsley III", E, ("0", "0"), "2.4", julia=(-0.75, -0.2),
          caps=_ESC_J),
        S("mandelbrot_cosine", "MandelbrotCosine", 143, "Mandelbrot cosine", E, ("0", "0"), "6", bailout=50.0,
          caps=_ESC),
        S("julia_cosine", "JuliaCosine", 144, "Julia cosine", E, ("0", "0"), "6", bailout=50.0, julia=(-0.2, 0.7),
          caps=_ESC_J),
        S("mandelbrot_sinh", "MandelbrotSinh", 145, "Mandelbrot sinh", E, ("0", "0"), "4", bailout=50.0, caps=_ESC),
        S("julia_sinh", "JuliaSinh", 146, "Julia sinh", E, ("0", "0"), "4", bailout=50.0, julia=(0.065, 0.122),
          caps=_ESC_J),
        S("feather", "Feather", 147, "Feather", E, ("0", "0"), "4", bailout=100.0, caps=_ESC),
        S("cactus", "Cactus", 148, "Cactus", E, ("0", "0"), "4", bailout=100.0, caps=_ESC),
        S("zubieta", "Zubieta", 149, "Zubieta", E, ("0", "0"), "1.8", bailout=100.0, caps=_ESC),
        S("tetration", "Tetration", 150, "Tetration", E, ("0", "0"), "4", bailout=100.0, caps=_ESC),
        S("mandelbulb", "Mandelbulb", 1001, "Mandelbulb", R, power=8.0, camera=(DEFAULT_YAW, DEFAULT_PITCH, 3.2),
          caps=frozenset({POWER, CAMERA})),
        S("mandelbox", "Mandelbox", 1002, "Mandelbox", R, camera=(DEFAULT_YAW, DEFAULT_PITCH, 5.0),
          caps=frozenset({CAMERA})),
        S("menger_sponge", "MengerSponge", 1003, "Menger sponge", R, camera=(DEFAULT_YAW, 25.0, 4.0),
          caps=frozenset({CAMERA, GEOM})),
        S("sierpinski_tetrahedron", "SierpinskiTetrahedron", 1004, "Sierpinski pyramid", R,
          camera=(DEFAULT_YAW, 25.0, 4.0), caps=frozenset({CAMERA, GEOM})),
        S("quaternion_julia", "QuaternionJulia", 1005, "Julia set 3D", R, camera=(DEFAULT_YAW, DEFAULT_PITCH, 3.2),
          quaternion=(-0.2, 0.8, 0.0, 0.0), caps=frozenset({CAMERA, QUAT})),
        S("octahedral_ifs", "OctahedralIFS", 1006, "Octahedral IFS", R, iterations=160,
          camera=(DEFAULT_YAW, DEFAULT_PITCH, 9.0),
          coefficients=(_c("a", _SCALE, 2.0, 0.5, 5.0, "Scale"), _c("b", _FOLD, 1.2, 0.5, 3.0, "Fold")),
          caps=frozenset({CAMERA, GEOM})),
        S("icosahedral_ifs", "IcosahedralIFS", 1007, "Icosahedral IFS", R, iterations=160,
          camera=(DEFAULT_YAW, DEFAULT_PITCH, 9.0),
          coefficients=(_c("a", _SCALE, 1.7, 0.5, 5.0, "Scale"), _c("b", _FOLD, 1.5, 0.5, 3.0, "Fold")),
          caps=frozenset({CAMERA, GEOM})),
        S("apollonian_gasket", "ApollonianGasket", 1008, "Apollonian gasket", R, iterations=160,
          camera=(DEFAULT_YAW, DEFAULT_PITCH, 8.0),
          coefficients=(_c("a", _SCALE, 1.3, 0.5, 3.0, "Scale"), _c("b", _FOLD, 1.35, 0.8, 2.0, "Fold"),
                        _c("c", _MINR, 1.12, 0.1, 2.0, "Minimum radius")),
          caps=frozenset({CAMERA})),
        S("kleinian", "Kleinian", 1009, "Kleinian", R, iterations=192,
          camera=(DEFAULT_YAW, DEFAULT_PITCH, 5.0),
          coefficients=(_c("a", _SCALE, 1.5, 0.5, 3.0, "Scale"), _c("b", _FOLD, 1.0, 0.5, 2.0, "Fold"),
                        _c("c", _MINR, 0.5, 0.5, 3.0, "Minimum radius")),
          caps=frozenset({CAMERA})),
        S("hybrid_mandelbulb_julia", "HybridMandelbulbJulia", 1010, "Hybrid Mandelbulb–Julia", R, power=8.0,
          iterations=192, camera=(DEFAULT_YAW, DEFAULT_PITCH, 4.0), julia=(-0.2, 0.8),
          caps=frozenset({POWER, CAMERA, JULIA})),
        S("quaternion_cubic", "QuaternionCubic", 1011, "Quaternion cubic", R, iterations=256,
          camera=(DEFAULT_YAW, DEFAULT_PITCH, 3.5), quaternion=(-0.2, 0.6, 0.3, 0.0),
          caps=frozenset({CAMERA, QUAT})),
        S("sierpinski_gasket", "SierpinskiGasket", 1012, "Sierpinski gasket", R, iterations=128,
          camera=(DEFAULT_YAW, 25.0, 5.0),
          coefficients=(_c("a", _SCALE, 1.5, 0.5, 5.0, "Scale"), _c("b", _FOLD, 1.0, 0.0, 2.0, "Fold"),
                        _c("c", _MINR, 0.5, 0.1, 2.0, "Minimum radius")),
          caps=frozenset({CAMERA, GEOM})),
        S("pickover", "Pickover", 1013, "Pickover", A, iterations=1000,
          camera=(DEFAULT_YAW, DEFAULT_PITCH, 3.0),
          coefficients=_abcd(2.24, 0.43, -0.65, -2.43, -10.0, 10.0), caps=frozenset({CAMERA, ITER, DENSITY})),
        S("lorenz", "Lorenz", 1014, "Lorenz", A, iterations=1000,
          camera=(DEFAULT_YAW, DEFAULT_PITCH, 5.0),
          coefficients=(_c("a", "sigma", 10.0, 0.1, 30.0, "σ"), _c("b", "rho", 28.0, 0.1, 100.0, "ρ"),
                        _c("c", "beta", 8.0 / 3.0, 0.1, 10.0, "β")),
          caps=frozenset({CAMERA, ITER, DENSITY})),
        S("rossler", "Rossler", 1015, "Rössler", A, iterations=1000,
          camera=(DEFAULT_YAW, DEFAULT_PITCH, 5.0),
          coefficients=(_c("a", "a", 0.2, -1, 1), _c("b", "b", 0.2, 0.01, 2), _c("c", "c", 5.7, 0.1, 20)),
          caps=frozenset({CAMERA, ITER, DENSITY})),
    ]


SPECS: dict[str, FractalSpec] = {s.key: s for s in _specs()}
TYPE_NAMES: list[str] = list(SPECS)

# FractalParams / FractalCamera / FractalQuaternion limits (ImageToolbox normalisation ranges)
MIN_POWER, MAX_POWER = 2.0, 12.0
MIN_BAILOUT, MAX_BAILOUT = 2.0, 1.0e12
MIN_PALETTE_CYCLES, MAX_PALETTE_CYCLES = 0.05, 64.0
MIN_NOVA, MAX_NOVA = 0.05, 2.0
MIN_SS, MAX_SS = 1, 4
MIN_ITERATIONS, MAX_ITERATIONS, MAX_EFFECTIVE_ITERATIONS = 16, 8192, 12_000
MIN_PITCH, MAX_PITCH = -89.0, 89.0
MIN_DISTANCE, MAX_DISTANCE = 0.5, 20.0
MIN_SPAN, MIN_DIRECT_RENDER_SPAN, MAX_SPAN = 1e-300, 1e-12, 1e6
MAX_OUTPUT_DIMENSION = 8192
COLORINGS = ["smooth", "banded", "orbit_trap", "angle"]
ITERATION_POLICIES = ["fixed", "scale_with_zoom", "adaptive"]


def option_schema(spec: FractalSpec) -> dict[str, dict]:
    """JSON-ish schema of the ``options`` accepted for a fractal type (key -> description)."""
    o: dict[str, dict] = {
        "coloring": {"type": "choice", "choices": COLORINGS, "default": "smooth",
                     "description": "Escape-time colouring (ignored by density and 3D types)"},
        "palette_cycles": {"type": "float", "min": MIN_PALETTE_CYCLES, "max": MAX_PALETTE_CYCLES, "default": 1.0},
        "palette_offset": {"type": "float", "default": 0.0, "description": "Wrapped into 0..1"},
        "palette_colors": {"type": "color_list", "min_items": 2, "max_items": 64,
                           "description": "Custom palette (overrides the palette parameter)"},
        "inside_color": {"type": "color", "default": "#000000",
                         "description": "Interior colour (background for density / 3D types)"},
        "supersampling": {"type": "int", "min": MIN_SS, "max": MAX_SS, "default": 1,
                          "description": "Samples per axis (density types: sampling density)"},
        "iteration_policy": {"type": "choice", "choices": ITERATION_POLICIES, "default": "adaptive"},
    }
    if not spec.three_d:
        o["center_x"] = {"type": "decimal", "default": spec.center[0],
                         "description": "Exact centre (number or decimal string; overrides center)"}
        o["center_y"] = {"type": "decimal", "default": spec.center[1]}
        o["span"] = {"type": "decimal", "default": spec.span, "min": MIN_SPAN, "max": MAX_SPAN,
                     "description": "Visible vertical extent (overrides zoom)"}
    if POWER in spec.caps:
        o["power"] = {"type": "float", "min": MIN_POWER, "max": MAX_POWER, "default": spec.power}
    if BAILOUT in spec.caps:
        o["bailout"] = {"type": "float", "min": MIN_BAILOUT, "max": MAX_BAILOUT, "default": spec.bailout,
                        "title": "Escape radius"}
    if JULIA in spec.caps:
        o["julia_real"] = {"type": "float", "min": -2.0, "max": 2.0, "default": spec.julia[0]}
        o["julia_imaginary"] = {"type": "float", "min": -2.0, "max": 2.0, "default": spec.julia[1]}
    if PHOENIX in spec.caps:
        o["phoenix_real"] = {"type": "float", "min": -2.0, "max": 2.0, "default": spec.phoenix[0]}
        o["phoenix_imaginary"] = {"type": "float", "min": -2.0, "max": 2.0, "default": spec.phoenix[1]}
    if NOVA in spec.caps:
        o["nova_relaxation"] = {"type": "float", "min": MIN_NOVA, "max": MAX_NOVA, "default": 1.0}
    for c in spec.coefficients:
        o[c.label] = {"type": "float", "min": c.min, "max": c.max, "default": c.default, "title": c.title,
                      "alias": c.slot}
    if QUAT in spec.caps:
        for i, k in enumerate("xyzw"):
            o[f"quaternion_{k}"] = {"type": "float", "min": -2.0, "max": 2.0, "default": spec.quaternion[i]}
    if CAMERA in spec.caps:
        o["camera_yaw"] = {"type": "float", "min": -180.0, "max": 180.0, "default": spec.camera[0], "unit": "deg"}
        o["camera_pitch"] = {"type": "float", "min": MIN_PITCH, "max": MAX_PITCH, "default": spec.camera[1],
                             "unit": "deg"}
        o["camera_distance"] = {"type": "float", "min": MIN_DISTANCE, "max": MAX_DISTANCE,
                                "default": spec.camera[2], "description": "Overrides zoom"}
    if spec.supports_floor:
        o["show_floor"] = {"type": "bool", "default": True}
        o["floor_primary_color"] = {"type": "color", "default": "#FFFFFF"}
        o["floor_secondary_color"] = {"type": "color", "default": "#000000"}
    if spec.key == "lyapunov":
        o["sequence"] = {"type": "text", "default": "AB", "pattern": "^[AB]{1,64}$",
                         "description": "Lyapunov rate sequence (engine parameter; ImageToolbox UI fixes 'AB')"}
    return o


def describe_type(spec: FractalSpec) -> dict:
    d = {"name": spec.key, "title": spec.title, "upstream": spec.kotlin, "stable_id": spec.stable_id,
         "family": spec.family, "dimension": "3d" if spec.three_d else "2d",
         "capabilities": sorted(spec.caps), "default_iterations": spec.iterations,
         "itb_ui": spec.itb_ui, "options": option_schema(spec)}
    if spec.three_d:
        d["default_camera"] = {"yaw": spec.camera[0], "pitch": spec.camera[1], "distance": spec.camera[2]}
    else:
        d["default_center"] = [float(spec.center[0]), float(spec.center[1])]
        d["default_span"] = spec.default_span
    return d
