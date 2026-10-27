"""Fractal rendering front-end: ImageToolbox parameter normalisation + render planning + kernel dispatch.

Mirrors (Apache-2.0):
  * ImageToolbox ``FractalParams.normalized()``, ``FractalIterationPolicy.resolve()``,
    ``FractalRenderRequest.resolvePlan()`` and ``NativeFractalRendererBackend.toNativeRequest()``
    (coefficient -> engine wire mapping, degrees -> radians, FOV 45, Lyapunov "AB").
  * fractal_engine ``RenderSettings::work_plan`` / ``ray_march_plan`` / ``Camera::new`` and the
    arbitrary-precision ``PerturbationReference`` (Python big integers instead of num-bigint).

LazyRoom extensions (no upstream equivalent, default = upstream behaviour): ``rotation`` (view
rotation about the image centre) and the ``zoom`` convenience (span = default span / zoom; for 3D
types camera distance = default distance / zoom).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

import numpy as np

from ..core.errors import invalid
from ..core.params import Bool, Choice, Float, Int, parse_color
from . import _density, _escape, _raymarch
from .catalog import (ATTRACTOR3D, COLORINGS, DENSITY2D, ESCAPE, MAX_BAILOUT, MAX_DISTANCE,
                      MAX_EFFECTIVE_ITERATIONS, MAX_ITERATIONS, MAX_NOVA, MAX_OUTPUT_DIMENSION, MAX_PALETTE_CYCLES,
                      MAX_PITCH, MAX_POWER, MIN_BAILOUT, MIN_DIRECT_RENDER_SPAN, MIN_DISTANCE,
                      MIN_ITERATIONS, MIN_NOVA, MIN_PALETTE_CYCLES, MIN_PITCH, MIN_POWER, MIN_SS, DEEP,
                      RAYMARCH, SPECS, FractalSpec, option_schema)
from .palettes import PALETTES, rgba_to_argb

MAX_RENDER_WORK_UNITS = 500_000_000
MAX_ESCAPE_TIME_WORK_UNITS = 10_000_000_000
REFERENCE_PIXELS = 640 * 480
FIELD_OF_VIEW = 45.0
MAX_CUSTOM_PALETTE = 64

IFS_MAPS = {
    121: ([(0.0, 0.0, 0.0, 0.16, 0.0, 0.0), (0.85, 0.04, -0.04, 0.85, 0.0, 1.6),
           (0.2, -0.26, 0.23, 0.22, 0.0, 1.6), (-0.15, 0.28, 0.26, 0.24, 0.0, 0.44)], (-2.19, 2.66, 0.0, 10.0)),
    122: ([(0.824074, 0.281428, -0.212346, 0.864198, -1.88229, -0.110607),
           (0.088272, 0.520988, -0.463889, -0.377778, 0.78536, 8.095795)], (-3.1, 3.1, -0.7, 9.2)),
    123: ([(0.387, 0.43, 0.43, -0.387, 0.256, 0.522), (0.441, -0.091, -0.009, -0.322, 0.4219, 0.5059),
           (-0.468, 0.02, -0.113, 0.015, 0.4, 0.4)], (-0.05, 1.05, -0.05, 1.05)),
    124: ([(0.0, -0.5, 0.5, 0.0, 0.5, 0.0), (0.0, 0.5, -0.5, 0.0, 0.5, 0.5), (0.5, 0.0, 0.0, 0.5, 0.25, 0.5)],
          (-0.05, 1.05, -0.05, 1.05)),
    125: ([(1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0), (1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0),
           (1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0, 0.0, 0.0), (1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0),
           (1.0 / 3.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0)], (-1.05, 1.05, -1.05, 1.05)),
}


# ------------------------------------------------------------------ small numeric helpers

def _kround(v: float) -> int:  # Kotlin roundToInt (half up)
    return int(math.floor(v + 0.5))


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _to_int32(v: int) -> int:  # Kotlin Long.toInt()
    return ((v + 2 ** 31) % 2 ** 32) - 2 ** 31


def _ceil_div(a: int, b: int) -> int:
    return a // b + (1 if a % b else 0)


def _ceil_sqrt(v: int) -> int:
    lo, hi = 1, max(v, 1)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if mid >= _ceil_div(v, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _density_multiplier(pixels: int, ss2: int) -> int:
    return max(_ceil_sqrt(_ceil_div(pixels * ss2, REFERENCE_PIXELS)), 1)


def _dec(value: Any, name: str) -> Decimal:
    try:
        d = Decimal(value if isinstance(value, str) else repr(float(value)))
    except (InvalidOperation, ValueError, TypeError):
        raise invalid(f"Option '{name}' must be a finite decimal number, got {value!r}", param=name) from None
    if not d.is_finite():
        raise invalid(f"Option '{name}' must be finite", param=name)
    if isinstance(value, str) and len(value) > 512:
        raise invalid(f"Option '{name}' is longer than 512 characters", param=name)
    return d


def _log10_magnitude(d: Decimal) -> float:
    d = abs(d)
    if d == 0:
        return -math.inf
    e = d.adjusted()
    return e + math.log10(float(d.scaleb(-e)))


# ------------------------------------------------------------------ option parsing

def parse_options(spec: FractalSpec, options: dict[str, Any] | None) -> dict[str, Any]:
    """Validate the per-type ``options`` object (unknown keys and out-of-range values are rejected)."""
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise invalid("'options' must be a JSON object", param="options")
    schema = option_schema(spec)
    aliases = {c.slot: c.label for c in spec.coefficients if c.slot != c.label}
    out: dict[str, Any] = {}
    for raw_key, value in options.items():
        key = aliases.get(raw_key, raw_key)
        if key not in schema:
            raise invalid(f"Unknown option '{raw_key}' for fractal type '{spec.key}'", param="options",
                          valid=sorted(schema), hint="Call fractal_types to see the options of each type.")
        s = schema[key]
        t = s["type"]
        if value is None:
            continue
        if t == "float":
            v = Float(key, min=s.get("min"), max=s.get("max")).coerce(value)
        elif t == "int":
            v = Int(key, min=s.get("min"), max=s.get("max")).coerce(value)
        elif t == "bool":
            v = Bool(key).coerce(value)
        elif t == "choice":
            v = Choice(key, choices=s["choices"]).coerce(value)
        elif t == "color":
            v = rgba_to_argb(parse_color(value, key))
        elif t == "color_list":
            if not isinstance(value, (list, tuple)) or not 2 <= len(value) <= MAX_CUSTOM_PALETTE:
                raise invalid(f"Option '{key}' must be a list of 2..{MAX_CUSTOM_PALETTE} colours", param=key)
            v = [rgba_to_argb(parse_color(c, key)) for c in value]
        elif t == "decimal":
            v = _dec(value, key)
            if key == "span" and not (Decimal("1E-300") <= v <= Decimal("1E6")):
                raise invalid(f"Option 'span' must be in [1e-300, 1e6], got {value!r}", param=key)
        else:  # text: Lyapunov sequence
            v = str(value).strip().upper()
            if not 1 <= len(v) <= 64 or any(ch not in "AB" for ch in v):
                raise invalid("Option 'sequence' must be 1..64 characters of 'A'/'B'", param=key)
        out[key] = v
    return out


# ------------------------------------------------------------------ planning (ImageToolbox + engine)

def _iterations_for(policy: str, base: int, span: Decimal, width: int, height: int) -> int:
    base = _clamp(base, MIN_ITERATIONS, MAX_ITERATIONS)
    if policy == "fixed":
        return base
    depth = max(_log10_magnitude(Decimal(3)) - _log10_magnitude(span), 0.0)
    ratio = max(max(width, 1) * max(height, 1) / float(REFERENCE_PIXELS), 1.0)
    res = math.log(ratio) * 48.0 if policy == "adaptive" else 0.0
    return _clamp(_kround(base + depth * 24.0 + res), MIN_ITERATIONS, MAX_EFFECTIVE_ITERATIONS)


def _rm_distance_iterations(kind: int, it: int) -> int:
    if kind == 1003:
        return _clamp(it // 48, 3, 7)
    if kind == 1004:
        return _clamp(it // 32, 6, 16)
    if kind in (1006, 1007, 1008):
        return _clamp(it // 16, 4, 18)
    if kind == 1009:
        return _clamp(it // 16, 10, 16)
    if kind in (1010, 1011):
        return _clamp(it // 16, 6, 24)
    if kind == 1012:
        return _clamp(it // 16, 5, 16)
    return _clamp(it // 16, 8, 28)  # Mandelbulb, Mandelbox, QuaternionJulia


_RM_MIN_ITERS = {1003: 3, 1004: 6, 1006: 4, 1007: 4, 1008: 4, 1009: 10, 1010: 6, 1011: 6, 1012: 5}


def _rm_work(kind: int, it: int, sampled: int) -> int:
    return sampled * (_clamp(it, 24, 192) + 6) * _rm_distance_iterations(kind, it)


def _max_rm_iterations(kind: int, requested: int, sampled: int) -> int | None:
    for it in range(requested, MIN_ITERATIONS - 1, -1):
        if _rm_work(kind, it, sampled) <= MAX_RENDER_WORK_UNITS:
            return it
    return None


def max_output_pixels(spec: FractalSpec) -> int:
    """ImageToolbox ``FractalRenderRequest.maxOutputPixelsFor``."""
    if spec.family == RAYMARCH:
        per_pixel = (24 + 6) * _RM_MIN_ITERS.get(spec.stable_id, 8)
        return min(MAX_RENDER_WORK_UNITS // per_pixel, MAX_OUTPUT_DIMENSION * MAX_OUTPUT_DIMENSION)
    return MAX_OUTPUT_DIMENSION * MAX_OUTPUT_DIMENSION


def resolve_plan(spec: FractalSpec, base_iterations: int, policy: str, span: Decimal, width: int, height: int,
                 supersampling: int) -> tuple[int, int]:
    """``FractalRenderRequest.resolvePlan`` -> (iterations, supersampling)."""
    density = spec.is_density
    w_eff = width if density else width * supersampling
    h_eff = height if density else height * supersampling
    requested = _iterations_for(policy, base_iterations, span, w_eff, h_eff)
    pixels = width * height
    if density:
        return requested, supersampling
    preferred = min(requested, 64)
    ss = supersampling
    if spec.family == RAYMARCH:
        it = _max_rm_iterations(spec.stable_id, requested, pixels * ss * ss)
        while ss > MIN_SS and (it is None or it < preferred):
            ss -= 1
            it = _max_rm_iterations(spec.stable_id, requested, pixels * ss * ss)
        if it is None:
            raise invalid(f"{width}x{height} is too large for the ray-marched type '{spec.key}' "
                          f"(ImageToolbox work budget allows at most {max_output_pixels(spec)} pixels)",
                          param="width", max_pixels=max_output_pixels(spec))
        return it, ss

    def max_it_for(s: int) -> int:
        return max(_to_int32(MAX_ESCAPE_TIME_WORK_UNITS // pixels // s // s), 1)

    while ss > MIN_SS and max_it_for(ss) < preferred:
        ss -= 1
    return min(requested, max_it_for(ss)), ss


def engine_ray_march_plan(kind: int, requested: int, sampled: int) -> tuple[int, int] | None:
    """fractal_engine ``ray_march_plan`` -> (maximum_steps, distance_iterations)."""
    desired_steps = _clamp(requested, 24, 192)
    desired_it = _rm_distance_iterations(kind, requested)
    min_it = _RM_MIN_ITERS.get(kind, 8)
    allowance = MAX_RENDER_WORK_UNITS // sampled
    if allowance < (24 + 6) * min_it:
        return None
    steps, iters = desired_steps, desired_it
    if (desired_steps + 6) * desired_it > allowance:
        iters = max(min(allowance // (desired_steps + 6), desired_it), min_it)
        if (desired_steps + 6) * iters > allowance:
            steps = max(min(max(allowance // iters - 6, 0), desired_steps), 24)
    return steps, iters


# ------------------------------------------------------------------ perturbation reference (deep zoom)

def _tdiv(a: int, b: int) -> int:  # BigInt division truncating toward zero
    q = abs(a) // b
    return -q if a < 0 else q


def _to_fixed(d: Decimal, digits: int) -> int:
    sign, ds, exp = d.as_tuple()
    coef = int("".join(map(str, ds)) or "0") * (-1 if sign else 1)
    shift = digits + exp
    return coef * 10 ** shift if shift >= 0 else _tdiv(coef, 10 ** (-shift))


def _fixed_to_float(v: int, scale: int) -> float:
    if v == 0:
        return 0.0
    try:
        return v / scale
    except OverflowError:
        return math.inf if v > 0 else -math.inf


def perturbation_reference(julia: bool, cx: Decimal, cy: Decimal, span: Decimal, jr: float, ji: float,
                           max_it: int) -> tuple[np.ndarray, np.ndarray]:
    digits = max(-span.adjusted(), 0) + 20
    scale = 10 ** digits
    if julia:
        zr, zi = _to_fixed(cx, digits), _to_fixed(cy, digits)
        cr, ci = _to_fixed(Decimal(repr(jr)), digits), _to_fixed(Decimal(repr(ji)), digits)
    else:
        zr = zi = 0
        cr, ci = _to_fixed(cx, digits), _to_fixed(cy, digits)
    re = np.full(max_it + 1, np.nan)
    im = np.full(max_it + 1, np.nan)
    re[0], im[0] = _fixed_to_float(zr, scale), _fixed_to_float(zi, scale)
    for i in range(max_it):
        zr, zi = _tdiv(zr * zr - zi * zi, scale) + cr, _tdiv((zr * zi) << 1, scale) + ci
        a, b = _fixed_to_float(zr, scale), _fixed_to_float(zi, scale)
        re[i + 1], im[i + 1] = a, b
        if not (math.isfinite(a) and math.isfinite(b)) or abs(a) > 1e100 or abs(b) > 1e100:
            break
    return re, im


# ------------------------------------------------------------------ camera

def _wrap_degrees(v: float) -> float:
    w = math.fmod(math.fmod(v + 180.0, 360.0) + 360.0, 360.0) - 180.0
    return 180.0 if (w == -180.0 and v > 0.0) else w


def _norm3(x, y, z):
    ln = math.sqrt(x * x + y * y + z * z)
    return (x / ln, y / ln, z / ln) if ln > 1e-15 else (x, y, z)


def build_camera(yaw_rad: float, pitch_rad: float, distance: float, fov: float = FIELD_OF_VIEW) -> np.ndarray:
    pitch = _clamp(pitch_rad, -math.pi * 0.49, math.pi * 0.49)
    ox = math.cos(pitch) * math.cos(yaw_rad) * distance
    oy = math.sin(pitch) * distance
    oz = math.cos(pitch) * math.sin(yaw_rad) * distance
    fx, fy, fz = _norm3(0.0 - ox, 0.0 - oy, 0.0 - oz)
    rx, ry, rz = _norm3(fy * 0.0 - fz * 1.0, fz * 0.0 - fx * 0.0, fx * 1.0 - fy * 0.0)
    if math.sqrt(rx * rx + ry * ry + rz * rz) < 1e-10:
        rx, ry, rz = 1.0, 0.0, 0.0
    ux, uy, uz = _norm3(ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx)
    return np.array([0.0 + ox, 0.0 + oy, 0.0 + oz, fx, fy, fz, rx, ry, rz, ux, uy, uz,
                     math.tan(fov * (math.pi / 180.0) * 0.5)], np.float64)


# ------------------------------------------------------------------ main entry

def render_fractal(type_key: str, width: int, height: int, *, iterations: int = 0, center=None, zoom: float = 1.0,
                   rotation: float = 0.0, palette: str = "classic",
                   options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Render a fractal; returns (RGBA uint8 (H, W, 4), info)."""
    spec = SPECS.get(type_key)
    if spec is None:
        raise invalid(f"Unknown fractal type '{type_key}'", param="type", valid=list(SPECS))
    if not (1 <= width <= MAX_OUTPUT_DIMENSION and 1 <= height <= MAX_OUTPUT_DIMENSION):
        raise invalid(f"Width and height must be in 1..{MAX_OUTPUT_DIMENSION}", param="width")
    opts = parse_options(spec, options)
    notes: list[str] = []
    kind = spec.stable_id

    # ---- FractalParams (normalised like FractalParams.normalized())
    power = _clamp(float(opts.get("power", spec.power)), MIN_POWER, MAX_POWER)
    if abs(power - 2.0) < 1e-10:
        power = 2.0
    deep_ok = DEEP in spec.caps and abs(power - 2.0) < 1e-10
    if spec.three_d:
        if center is not None and len(center):
            notes.append("center is ignored for 3D types (camera orbits the origin)")
        span = Decimal(3)
        cx_d = cy_d = Decimal(0)
    else:
        if center is not None and len(center):
            if len(center) != 2:
                raise invalid("'center' must be [x, y] or empty", param="center")
            cx_d, cy_d = _dec(center[0], "center"), _dec(center[1], "center")
        else:
            cx_d, cy_d = Decimal(spec.center[0]), Decimal(spec.center[1])
        cx_d = opts.get("center_x", cx_d)
        cy_d = opts.get("center_y", cy_d)
        if "span" in opts:
            span = opts["span"]
        else:
            with localcontext() as ctx:
                ctx.prec = 60
                span = Decimal(spec.span) / Decimal(repr(float(zoom)))
        span = _clamp(span, Decimal("1E-300"), Decimal("1E6"))
        if not deep_ok and span < Decimal(repr(MIN_DIRECT_RENDER_SPAN)):
            span = Decimal("1E-12")
            notes.append("span clamped to 1e-12: deep zoom needs mandelbrot/julia with power 2 (as in ImageToolbox)")
    bailout = _clamp(float(opts.get("bailout", spec.bailout)), MIN_BAILOUT, MAX_BAILOUT)
    cycles = _clamp(float(opts.get("palette_cycles", 1.0)), MIN_PALETTE_CYCLES, MAX_PALETTE_CYCLES)
    offset = float(opts.get("palette_offset", 0.0))
    offset = offset - math.floor(offset)
    jr = float(opts.get("julia_real", spec.julia[0]))
    ji = float(opts.get("julia_imaginary", spec.julia[1]))
    phr = float(opts.get("phoenix_real", spec.phoenix[0]))
    phi = float(opts.get("phoenix_imaginary", spec.phoenix[1]))
    nova = _clamp(float(opts.get("nova_relaxation", 1.0)), MIN_NOVA, MAX_NOVA)
    coeff = {c.slot: _clamp(float(opts.get(c.label, c.default)), c.min, c.max) for c in spec.coefficients}
    quat = tuple(_clamp(float(opts.get(f"quaternion_{k}", spec.quaternion[i])), -2.0, 2.0)
                 for i, k in enumerate("xyzw"))
    yaw = _wrap_degrees(float(opts.get("camera_yaw", spec.camera[0])))
    pitch = _clamp(float(opts.get("camera_pitch", spec.camera[1])), MIN_PITCH, MAX_PITCH)
    if "camera_distance" in opts:
        distance = float(opts["camera_distance"])
    else:
        distance = spec.camera[2] / float(zoom)
    distance = _clamp(distance, MIN_DISTANCE, MAX_DISTANCE)
    show_floor = bool(opts.get("show_floor", spec.supports_floor)) and spec.supports_floor
    floor1 = int(opts.get("floor_primary_color", 0xFFFFFFFF)) | 0xFF000000
    floor2 = int(opts.get("floor_secondary_color", 0xFF000000)) | 0xFF000000
    supersampling = int(opts.get("supersampling", 1))
    policy = opts.get("iteration_policy", "adaptive")
    coloring = COLORINGS.index(opts.get("coloring", "smooth"))
    inside = int(opts.get("inside_color", 0xFF000000))
    if "palette_colors" in opts:
        colors = opts["palette_colors"]
    else:
        if palette not in PALETTES:
            raise invalid(f"Unknown palette '{palette}'", param="palette", valid=list(PALETTES))
        colors = PALETTES[palette][1]
    base_it = int(iterations) if iterations else spec.iterations
    base_it = _clamp(base_it, MIN_ITERATIONS, MAX_ITERATIONS)

    # ---- plan (FractalRenderRequest.resolvePlan)
    max_it, ss = resolve_plan(spec, base_it, policy, span, width, height, supersampling)
    if ss != supersampling:
        notes.append(f"supersampling reduced to {ss} to fit the ImageToolbox work budget")

    # ---- native wire mapping (toNativeRequest)
    ep = EngineParams(
        kind=kind, max_iterations=max_it, center_x=cx_d, center_y=cy_d, span=span, aspect=width / height,
        power=coeff.get("c", power), bailout=bailout, supersampling=ss, coloring=coloring, palette_cycles=cycles,
        palette_offset=offset, inside_color=inside, julia_real=coeff.get("a", jr), julia_imaginary=coeff.get("b", ji),
        phoenix_real=coeff.get("d", phr), phoenix_imaginary=phi, nova_relaxation=nova,
        sequence=opts.get("sequence", "AB"), camera_yaw=math.radians(yaw), camera_pitch=math.radians(pitch),
        camera_distance=distance, quaternion=quat, show_floor=show_floor, floor_primary=floor1, floor_secondary=floor2,
        palette=list(colors), rotation=float(rotation))
    out, info = render_engine(ep, width, height)
    info = {"type": spec.key, "iterations": max_it, "supersampling": ss, **info}
    if spec.three_d:
        info["camera"] = {"yaw": yaw, "pitch": pitch, "distance": distance}
    else:
        info.update(center=[str(cx_d), str(cy_d)], span=str(span), zoom=float(Decimal(spec.span) / span))
    if notes:
        info["notes"] = notes
    return out, info


@dataclass
class EngineParams:
    """fractal_engine wire parameters (``FractalRenderRequest.toNativeParameters`` + palette/sequence)."""
    kind: int
    max_iterations: int
    center_x: Decimal = Decimal("-0.5")
    center_y: Decimal = Decimal(0)
    span: Decimal = Decimal(3)
    aspect: float = 1.0
    power: float = 2.0
    bailout: float = 4.0
    supersampling: int = 1
    coloring: int = 0
    palette_cycles: float = 1.0
    palette_offset: float = 0.0
    inside_color: int = 0xFF000000
    julia_real: float = -0.8
    julia_imaginary: float = 0.156
    phoenix_real: float = -0.5
    phoenix_imaginary: float = 0.0
    nova_relaxation: float = 1.0
    sequence: str = "AB"
    camera_yaw: float = 0.65
    camera_pitch: float = 0.3
    camera_distance: float = 3.5
    quaternion: tuple = (-0.2, 0.8, 0.0, 0.0)
    field_of_view: float = FIELD_OF_VIEW
    show_floor: bool = False
    floor_primary: int = 0xFFFFFFFF
    floor_secondary: int = 0xFF000000
    palette: list = field(default_factory=lambda: list(PALETTES["classic"][1]))
    rotation: float = 0.0  # LazyRoom extension


def engine_work_plan(ep: EngineParams, width: int, height: int) -> dict[str, int] | None:
    """``RenderSettings::work_plan``: None when the engine would refuse the render (work limit)."""
    kind = ep.kind
    spec = _BY_ID[kind]
    pixels = width * height
    ss2 = ep.supersampling * ep.supersampling
    sampled = pixels * ss2
    if kind == 109:
        orbit = max(ep.max_iterations, 32)
        samples = min(max(sampled // 2, 8192), (MAX_RENDER_WORK_UNITS - pixels) // orbit)
        return {"work": pixels + samples * orbit, "samples": samples}
    if spec.family == DENSITY2D:
        base = _clamp(ep.max_iterations * 128, 24_000, 1_000_000)
        steps = min(base * _density_multiplier(pixels, ss2), (MAX_RENDER_WORK_UNITS - pixels) // 25)
        return {"work": pixels + steps * 25, "steps": steps}
    if spec.family == ATTRACTOR3D:
        base = _clamp(ep.max_iterations * 32, 12_000, 240_000)
        steps = min(base * _density_multiplier(pixels, ss2), (MAX_RENDER_WORK_UNITS - pixels) // 9)
        return {"work": pixels + steps * 9, "steps": steps}
    if spec.family == RAYMARCH:
        plan = engine_ray_march_plan(kind, ep.max_iterations, sampled)
        if plan is None:
            return None
        work = sampled * (plan[0] + 6) * plan[1]
        return {"work": work, "steps": plan[0], "distance_iterations": plan[1]} if work <= MAX_RENDER_WORK_UNITS \
            else None
    work = sampled * ep.max_iterations
    return {"work": work} if work <= MAX_ESCAPE_TIME_WORK_UNITS else None


def render_engine(ep: EngineParams, width: int, height: int) -> tuple[np.ndarray, dict[str, Any]]:
    """fractal_engine ``render_into`` (unpremultiplied RGBA uint8)."""
    plan = engine_work_plan(ep, width, height)
    if plan is None:
        raise invalid(f"{width}x{height} exceeds the fractal engine work budget for this type and quality",
                      param="width")
    kind = ep.kind
    spec = _BY_ID[kind]
    ss = ep.supersampling
    pal = np.array(ep.palette, np.int64)
    seq = np.array([ch == "B" for ch in ep.sequence], np.bool_)
    rot_deg = math.fmod(ep.rotation, 360.0)
    rot = rot_deg != 0.0
    rc, rs = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    pixels = width * height
    cx, cy, span_f = float(ep.center_x), float(ep.center_y), float(ep.span)
    aspect = ep.aspect
    inside = int(ep.inside_color)
    out = np.zeros((height, width, 4), np.uint8)
    info: dict[str, Any] = {}
    if spec.family == ESCAPE:
        use_pert = kind in (1, 2) and ep.power == 2.0 and ep.span <= Decimal("1E-12")
        if use_pert:
            ref_r, ref_i = perturbation_reference(kind == 2, ep.center_x, ep.center_y, ep.span, ep.julia_real,
                                                  ep.julia_imaginary, ep.max_iterations)
            info["perturbation"] = True
        else:
            ref_r, ref_i = _escape.empty_ref()
        _escape.render_2d(out, kind, ss, cx, cy, span_f, aspect, rot, rc, rs, ep.max_iterations, float(ep.power),
                          float(ep.bailout), float(ep.julia_real), float(ep.julia_imaginary), float(ep.phoenix_real),
                          float(ep.phoenix_imaginary), float(ep.nova_relaxation), int(ep.coloring),
                          float(ep.palette_cycles), float(ep.palette_offset), inside, pal, seq, use_pert, kind == 2,
                          ref_r, ref_i)
    elif spec.family == DENSITY2D:
        hspan = span_f * aspect
        if rot:
            ex = abs(rc) * hspan * 0.5 + abs(rs) * span_f * 0.5
            ey = abs(rs) * hspan * 0.5 + abs(rc) * span_f * 0.5
            bounds = (cx - ex, cx + ex, cy - ey, cy + ey)
        else:
            bounds = (cx - hspan * 0.5, cx + hspan * 0.5, cy - span_f * 0.5, cy + span_f * 0.5)
        vp = np.array([cx, cy, span_f, aspect, width, height, 1.0 if rot else 0.0, rc, rs, *bounds], np.float64)
        dens = np.zeros(pixels, np.int64)
        a, b, c, d = float(ep.julia_real), float(ep.julia_imaginary), float(ep.power), float(ep.phoenix_real)
        if kind == 109:
            samples, orbit = plan["samples"], max(ep.max_iterations, 32)
            nchunks = max(1, min(_threads(), (256 << 20) // max(pixels * 4, 1), samples))
            chunks = np.zeros((nchunks, pixels), np.int32)
            _density.buddhabrot(chunks, vp, samples, orbit)
            dens = chunks.sum(axis=0, dtype=np.int64)
            info["samples"] = samples
        elif 110 <= kind <= 120:
            _density.attractor_2d(dens, vp, kind, a, b, c, d, plan["steps"])
            info["steps"] = plan["steps"]
        elif kind in IFS_MAPS:
            maps, bnds = IFS_MAPS[kind]
            _density.affine_ifs(dens, vp, np.array(maps, np.float64), np.array(bnds, np.float64), _threads())
        elif kind == 126:
            _density.pythagoras_tree(dens, vp)
        elif kind == 127:
            _density.h_tree(dens, vp)
        elif kind == 128:
            _density.heighway_dragon(dens, vp)
        elif kind == 129:
            _density.koch_snowflake(dens, vp)
        else:
            _density.hilbert_curve(dens, vp)
        _density.write_density(dens, out, float(ep.palette_cycles), float(ep.palette_offset), inside, pal)
    else:
        cam = build_camera(ep.camera_yaw, ep.camera_pitch, ep.camera_distance, ep.field_of_view)
        if spec.family == ATTRACTOR3D:
            dens = np.zeros(pixels, np.int64)
            _density.attractor_3d(dens, cam, width, height, aspect, kind, float(ep.julia_real),
                                  float(ep.julia_imaginary), float(ep.power), float(ep.phoenix_real), plan["steps"],
                                  rot, rc, rs)
            _density.write_density(dens, out, float(ep.palette_cycles), float(ep.palette_offset), inside, pal)
            info["steps"] = plan["steps"]
        else:
            _raymarch.render_3d(out, kind, cam, ss, aspect, rot, rc, rs, plan["steps"], plan["distance_iterations"],
                                float(ep.power), float(ep.julia_real), float(ep.julia_imaginary), float(ep.power),
                                np.array(ep.quaternion, np.float64), _ICO_NORMALS, float(ep.camera_distance),
                                bool(ep.show_floor), 0.0, int(ep.floor_primary) | 0xFF000000,
                                int(ep.floor_secondary) | 0xFF000000, float(ep.palette_cycles),
                                float(ep.palette_offset), inside, pal)
            info.update(steps=plan["steps"], distance_iterations=plan["distance_iterations"],
                        show_floor=bool(ep.show_floor))
    return out, info


_BY_ID = {s.stable_id: s for s in SPECS.values()}
_ICO_NORMALS = _raymarch.icosahedral_normals()


def _threads() -> int:
    try:
        import numba

        return int(numba.get_num_threads())
    except Exception:  # pragma: no cover
        return 1
