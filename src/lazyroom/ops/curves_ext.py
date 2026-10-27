"""ImageToolbox colour curves (``feature/curves`` + ``lib/curves``) as ``adjust.color_curves``.

Port of ImageToolbox's curves editor state and its GPU filter chain:

* ``view/PhotoFilterCurvesControl.kt`` – ``CurvesValue`` (control points, ``replacePoints``
  normalisation, Catmull-Rom / centred Hermite / periodic Hermite interpolation, ``toLut``) and
  ``CurvesToolValue`` (17 curves in a fixed order + colour wheels).
* ``GPUImageColorCurvesFilter.kt`` – ``buildCurvesFilter``: RGB tone curve (CyberAgent
  ``GPUImageToneCurveFilter``: natural cubic spline on 0..255 integer points, additive master
  offsets), CMYK and Lab LUT shaders, HSV relation curves (Hue vs Sat/Hue/Luma, Luma vs
  Sat/Hue, Sat vs Sat) and the OKLab colour-wheels shader.  GLSL maths is ported to
  vectorised float32 numpy; 256-entry byte LUT textures are sampled with OpenGL ``GL_LINEAR`` /
  ``CLAMP_TO_EDGE`` semantics (texel centres at (i + 0.5) / 256).
* ``ImageCurvesEditorState.kt`` / ``ColorWheelsValue.kt`` – serialised control points
  (17 flattened x/y curves + 3 wheel points + edges) shared with ``CurvesPresetCodec``.

Deviations (documented in docs/dev/port_reports/color_tools.md): processing stays in float
(no 8-bit quantisation of the input or of the intermediate HSV framebuffer), and control-point
lists without the fixed x=0 / x=1 endpoints get neutral endpoints added (the ImageToolbox UI
always stores them).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.errors import invalid
from ..core.params import File, Float, Point, Points, Text
from ..core.registry import op

F32 = np.float32

# Curve order of CurvesToolValue.allCurves (ImageCurvesEditorType.curveOffset + channel).
CURVE_NAMES: tuple[str, ...] = (
    "rgb", "red", "green", "blue",                        # RGB  (offset 0): Luma, R, G, B
    "cyan", "magenta", "yellow", "black",                 # CMYK (offset 4)
    "lab_l", "lab_a", "lab_b",                            # Lab  (offset 8)
    "hue_vs_sat", "hue_vs_hue", "hue_vs_luma",            # offsets 11-13 (circular input)
    "luma_vs_sat", "luma_vs_hue", "sat_vs_sat",           # offsets 14-16 (centred, linear input)
)
CURVE_COUNT = 17
CENTERED = {n: i for i, n in enumerate(CURVE_NAMES) if i >= 11}
CIRCULAR = {"hue_vs_sat", "hue_vs_hue", "hue_vs_luma"}
EDITOR_TYPES = {  # ImageCurvesEditorType -> curve names
    "RGB": CURVE_NAMES[0:4], "CMYK": CURVE_NAMES[4:8], "Lab": CURVE_NAMES[8:11],
    "HueVsSat": ("hue_vs_sat",), "HueVsHue": ("hue_vs_hue",), "HueVsLuma": ("hue_vs_luma",),
    "LumaVsSat": ("luma_vs_sat",), "LumaVsHue": ("luma_vs_hue",), "SatVsSat": ("sat_vs_sat",),
    "ColorWheels": (),
}
WHEEL_NAMES = ("shadows", "midtones", "highlights")

CURVE_GRANULARITY = 100      # PhotoFilterCurvesControl.curveGranularity
MAX_POINT_COUNT = 16         # PhotoFilterCurvesControl.MaxPointCount
CURVE_SAMPLE_COUNT = 256     # CurveSampleCount (= CircularCurveSampleCount)
LUT_SIZE = 256               # GPUImageColorCurvesFilter.LutSize
LEGACY_POINT_COUNT = 5       # ImageCurvesEditorState.LegacyPointCount
DEFAULT_EDGES = 0.5          # ColorWheelsValue.DefaultEdges
EPS_DEFAULT = 0.00001


def _f32(v: float) -> float:
    """Round a Python float to the nearest float32 (Kotlin ``Float`` arithmetic)."""
    return float(np.float32(v))


# =============================================================================
# CurvesValue (PhotoFilterCurvesControl.CurvesValue)
# =============================================================================

@dataclass
class CurvesValue:
    neutral: float | None = None          # neutralValue (0.5 for centred curves)
    circular: bool = False                # circularInput (hue curves)
    points: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.points:
            self.reset()

    def reset(self) -> None:
        n = self.neutral
        self.points = [[0.0, n if n is not None else 0.0], [1.0, n if n is not None else 1.0]]

    def copy(self) -> "CurvesValue":
        c = CurvesValue(self.neutral, self.circular)
        c.replace_points(self.points)
        return c

    @property
    def is_default(self) -> bool:
        return all(abs(p[1] - (self.neutral if self.neutral is not None else p[0])) < EPS_DEFAULT
                   for p in self.points)

    def replace_points(self, new_points: Sequence[Sequence[float]]) -> None:
        """``replacePoints``: clamp, sort by x, (dedupe x for plain curves), pin x of the endpoints to
        0 and 1, keep at most 16 points, and mirror neighbour y into the endpoints of centred curves."""
        pts = [[_f32(min(max(float(p[0]), 0.0), 1.0)), _f32(min(max(float(p[1]), 0.0), 1.0))]
               for p in new_points]
        pts.sort(key=lambda p: p[0])  # stable, like Kotlin sortedBy
        if self.neutral is None:
            seen, uniq = set(), []
            for p in pts:  # distinctBy { it.x } keeps the first occurrence
                if p[0] not in seen:
                    seen.add(p[0])
                    uniq.append(p)
            pts = uniq
        if len(pts) < 2:
            return
        pts[0][0] = 0.0
        pts[-1][0] = 1.0
        if len(pts) > MAX_POINT_COUNT:
            pts = pts[:MAX_POINT_COUNT - 1] + [pts[-1]]
        if self.neutral is not None and not self.circular and len(pts) > 2:
            pts[0][1] = pts[1][1]
            pts[-1][1] = pts[-2][1]
        self.points = pts

    # -- interpolation ------------------------------------------------------
    def interpolate(self) -> np.ndarray:
        """``interpolateCurve`` -> (N, 2) array of (x, y) samples."""
        if self.circular:
            return self._interpolate_circular()
        if self.neutral is not None:
            return self._interpolate_centered()
        # Catmull-Rom, 100 samples per segment, evaluated in float32 like the Kotlin code.
        pts = self.points
        f = np.float32
        t = np.arange(1, CURVE_GRANULARITY + 1, dtype=f) * (f(1.0) / f(CURVE_GRANULARITY))
        tt = t * t
        ttt = tt * t
        xs_out = [np.array([pts[0][0]], f)]
        ys_out = [np.array([pts[0][1]], f)]
        for index in range(len(pts) - 1):
            p0 = pts[index - 1] if index >= 1 else pts[index]
            p1, p2 = pts[index], pts[index + 1]
            p3 = pts[index + 2] if index + 2 < len(pts) else p2

            def cr(a0, a1, a2, a3):
                a0, a1, a2, a3 = f(a0), f(a1), f(a2), f(a3)
                return f(0.5) * (f(2) * a1 + (a2 - a0) * t + (f(2) * a0 - f(5) * a1 + f(4) * a2 - a3) * tt
                                 + (f(3) * a1 - a0 - f(3) * a2 + a3) * ttt)

            pix = cr(p0[0], p1[0], p2[0], p3[0])
            piy = cr(p0[1], p1[1], p2[1], p3[1])
            xs_out.append(np.clip(pix, f(p1[0]), f(p2[0])))
            ys_out.append(np.clip(piy, f(0), f(1)))
        return np.stack([np.concatenate(xs_out), np.concatenate(ys_out)], 1).astype(np.float64)

    def _interpolate_circular(self) -> np.ndarray:
        ctrl = self.points[1:-1]
        neutral = self.neutral if self.neutral is not None else 0.5
        if not ctrl:
            return np.array([[0.0, neutral], [1.0, neutral]])
        if len(ctrl) == 1:
            v = ctrl[0][1]
            return np.array([[0.0, v], [1.0, v]])
        n = len(ctrl)
        cx = np.array([p[0] for p in ctrl], np.float64)
        cy = np.array([p[1] for p in ctrl], np.float64)

        def periodic(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            cycle = np.floor_divide(idx, n)
            k = np.mod(idx, n)
            return cx[k] + cycle, cy[k]

        x = np.arange(CURVE_SAMPLE_COUNT + 1, dtype=np.float64) / CURVE_SAMPLE_COUNT
        # indexOfFirst { point.x >= x }, or size when none
        ge = cx[None, :] >= x[:, None]
        nxt = np.where(ge.any(1), ge.argmax(1), n)
        p0x, p0y = periodic(nxt - 2)
        p1x, p1y = periodic(nxt - 1)
        p2x, p2y = periodic(nxt)
        p3x, p3y = periodic(nxt + 1)
        seg = np.maximum(p2x - p1x, 0.00001)
        t = np.clip((x - p1x) / seg, 0, 1)
        tt, ttt = t * t, t * t * t
        s1 = (p2y - p0y) / np.maximum(p2x - p0x, 0.00001)
        s2 = (p3y - p1y) / np.maximum(p3x - p1x, 0.00001)
        y = ((2 * ttt - 3 * tt + 1) * p1y + (ttt - 2 * tt + t) * seg * s1
             + (-2 * ttt + 3 * tt) * p2y + (ttt - tt) * seg * s2)
        return np.stack([x, np.clip(y, 0, 1)], 1)

    def _interpolate_centered(self) -> np.ndarray:
        ctrl: list[list[float]] = []
        for p in self.points[1:-1]:  # fold: a point with the same x replaces the previous one
            if ctrl and ctrl[-1][0] == p[0]:
                ctrl[-1] = p
            else:
                ctrl.append(p)
        neutral = self.neutral if self.neutral is not None else 0.5
        if not ctrl:
            return np.array([[0.0, neutral], [1.0, neutral]])
        if len(ctrl) == 1:
            v = ctrl[0][1]
            return np.array([[0.0, v], [1.0, v]])
        last = len(ctrl) - 1

        def slope(i: int) -> float:
            if i == 0 or i == last:
                return 0.0
            prev, nxt = ctrl[i - 1], ctrl[i + 1]
            return (nxt[1] - prev[1]) / max(nxt[0] - prev[0], 0.00001)

        out = []
        seg_i = 0
        for sample in range(CURVE_SAMPLE_COUNT + 1):
            x = sample / CURVE_SAMPLE_COUNT
            if x <= ctrl[0][0]:
                out.append((x, ctrl[0][1]))
            elif x >= ctrl[-1][0]:
                out.append((x, ctrl[-1][1]))
            else:
                while seg_i < last - 1 and ctrl[seg_i + 1][0] < x:
                    seg_i += 1
                p1, p2 = ctrl[seg_i], ctrl[seg_i + 1]
                w = max(p2[0] - p1[0], 0.00001)
                t = min(max((x - p1[0]) / w, 0.0), 1.0)
                tt, ttt = t * t, t * t * t
                y = ((2 * ttt - 3 * tt + 1) * p1[1] + (ttt - 2 * tt + t) * w * slope(seg_i)
                     + (-2 * ttt + 3 * tt) * p2[1] + (ttt - tt) * w * slope(seg_i + 1))
                out.append((x, min(max(y, 0.0), 1.0)))
        return np.array(out, np.float64)

    def to_lut(self, size: int = LUT_SIZE) -> np.ndarray:
        """``toLut``: piecewise-linear resampling of the interpolated curve at x = i / (size - 1)."""
        interp = self.interpolate()
        if len(interp) < 2:
            return np.arange(size, dtype=np.float64) / (size - 1)
        xs, ys = interp[:, 0], interp[:, 1]
        npts = len(interp)
        out = np.empty(size, np.float64)
        seg = 0
        for i in range(size):
            x = i / (size - 1)
            while seg < npts - 2 and xs[seg + 1] < x:
                seg += 1
            x0, y0, x1, y1 = xs[seg], ys[seg], xs[seg + 1], ys[seg + 1]
            frac = min(max((x - x0) / (x1 - x0), 0.0), 1.0) if x1 > x0 else 0.0
            out[i] = y0 + (y1 - y0) * frac
        return out


def lut_bytes(values: np.ndarray) -> np.ndarray:
    """Upload as GL_UNSIGNED_BYTE: ``(value.coerceIn(0, 1) * 255).roundToInt()`` -> 0..1 floats."""
    v = np.clip(np.asarray(values, np.float64), 0, 1) * 255.0
    return (np.floor(v + 0.5) / 255.0).astype(F32)  # Kotlin roundToInt: half up


def sample_texture(lut: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``texture2D`` on a 256x1 GL_LINEAR / CLAMP_TO_EDGE texture: texel i centred at (i + 0.5) / 256."""
    n = len(lut)
    pos = np.asarray(u, F32) * F32(n) - F32(0.5)
    return np.interp(pos, np.arange(n, dtype=F32), lut).astype(F32)


# =============================================================================
# Editor state (CurvesToolValue + ColorWheelsValue + ImageCurvesEditorState)
# =============================================================================

def _normalize_wheel(x: float, y: float) -> tuple[float, float]:
    """``ColorWheelPoint.normalized``: clamp inside the unit disc, project points outside onto it."""
    length = math.sqrt(x * x + y * y)
    if length <= 1.0:
        return (_f32(min(max(x, -1.0), 1.0)), _f32(min(max(y, -1.0), 1.0)))
    return (_f32(x / length), _f32(y / length))


@dataclass
class CurvesState:
    curves: dict[str, CurvesValue] = field(default_factory=dict)
    wheels: dict[str, tuple[float, float]] = field(default_factory=dict)
    edges: float = DEFAULT_EDGES

    def __post_init__(self) -> None:
        for i, name in enumerate(CURVE_NAMES):
            if name not in self.curves:
                self.curves[name] = CurvesValue(0.5 if i >= 11 else None, name in CIRCULAR)
        for w in WHEEL_NAMES:
            self.wheels.setdefault(w, (0.0, 0.0))

    def copy(self) -> "CurvesState":
        return CurvesState({k: v.copy() for k, v in self.curves.items()}, dict(self.wheels), self.edges)

    @property
    def wheels_default(self) -> bool:
        return (all(x * x + y * y < 0.000001 for x, y in self.wheels.values())
                and abs(self.edges - DEFAULT_EDGES) < 0.000001)

    def is_default(self) -> bool:
        return all(c.is_default for c in self.curves.values()) and self.wheels_default

    # -- serialised form ----------------------------------------------------
    def control_points(self) -> list[list[float]]:
        """``ImageCurvesEditorState.controlPoints``: 17 flattened x/y curves + wheels (x,y)x3 + [edges]."""
        cps = [[float(v) for p in self.curves[n].points for v in p] for n in CURVE_NAMES]
        cps += [list(map(float, self.wheels[w])) for w in WHEEL_NAMES]
        cps.append([float(self.edges)])
        return cps

    @classmethod
    def from_control_points(cls, cps: Sequence[Sequence[float]]) -> "CurvesState":
        """``ImageCurvesEditorState(controlPoints)`` (``initControlPoints``)."""
        st = cls()
        for i, name in enumerate(CURVE_NAMES):
            if i < len(cps) and cps[i] is not None:
                vals = [float(v) for v in cps[i]]
                if len(vals) == LEGACY_POINT_COUNT:  # legacy Y-only curves
                    pts = [[k / (LEGACY_POINT_COUNT - 1.0), y] for k, y in enumerate(vals)]
                else:
                    pts = [[vals[k], vals[k + 1]] for k in range(0, len(vals) - 1, 2)]
                st.curves[name].replace_points(pts)

        def point_at(idx: int) -> tuple[float, float]:
            if idx < len(cps) and cps[idx] is not None:
                v = list(cps[idx])
                return (float(v[0]) if len(v) > 0 else 0.0, float(v[1]) if len(v) > 1 else 0.0)
            return (0.0, 0.0)

        for k, w in enumerate(WHEEL_NAMES):
            st.wheels[w] = _normalize_wheel(*point_at(CURVE_COUNT + k))
        edges_list = cps[CURVE_COUNT + 3] if len(cps) > CURVE_COUNT + 3 else None
        edges = float(edges_list[0]) if edges_list else 1.0  # ImageToolbox default when missing: 1f
        st.edges = _f32(min(max(edges, 0.0), 1.0))
        return st

    # -- op parameter form --------------------------------------------------
    def to_params(self) -> dict[str, Any]:
        d: dict[str, Any] = {n: [[float(x), float(y)] for x, y in self.curves[n].points] for n in CURVE_NAMES}
        for w in WHEEL_NAMES:
            d[f"wheel_{w}"] = [float(v) for v in self.wheels[w]]
        d["wheel_edges"] = float(self.edges)
        return d


def default_params() -> dict[str, Any]:
    return CurvesState().to_params()


def _with_endpoints(name: str, pts: Sequence[Sequence[float]]) -> list[list[float]]:
    """Add the fixed x=0 / x=1 endpoints when a user-supplied list lacks them (LazyRoom extension;
    ImageToolbox's editor always stores them)."""
    pts = sorted(([float(p[0]), float(p[1])] for p in pts), key=lambda p: p[0])
    if not pts:
        return pts
    centered = name in CENTERED
    if pts[0][0] > 0.0:
        pts.insert(0, [0.0, 0.5 if centered else 0.0])
    if pts[-1][0] < 1.0:
        pts.append([1.0, 0.5 if centered else 1.0])
    return pts


def state_from_params(values: dict[str, Any], base: "CurvesState | None" = None) -> CurvesState:
    """Build a state from op-style params.  Curves equal to their default are left as in ``base``."""
    st = base.copy() if base is not None else CurvesState()
    defaults = default_params()
    for name in CURVE_NAMES:
        if name not in values or values[name] is None:
            continue
        pts = [[float(p[0]), float(p[1])] for p in values[name]]
        if base is not None and pts == defaults[name]:
            continue
        st.curves[name].replace_points(_with_endpoints(name, pts))
    for w in WHEEL_NAMES:
        v = values.get(f"wheel_{w}")
        if v is not None and not (base is not None and [float(v[0]), float(v[1])] == defaults[f"wheel_{w}"]):
            st.wheels[w] = _normalize_wheel(float(v[0]), float(v[1]))
    e = values.get("wheel_edges")
    if e is not None and not (base is not None and float(e) == DEFAULT_EDGES):
        st.edges = _f32(min(max(float(e), 0.0), 1.0))
    return st


# =============================================================================
# GPU filter chain (buildCurvesFilter)
# =============================================================================

# --- GPUImageToneCurveFilter (CyberAgent GPUImage) ----------------------------

def _second_derivative(pts: list[tuple[int, int]]) -> list[float] | None:
    n = len(pts)
    if n <= 1:
        return None
    m = [[0.0, 0.0, 0.0] for _ in range(n)]
    r = [0.0] * n
    m[0][1] = 1.0
    for i in range(1, n - 1):
        p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1]
        m[i][0] = (p2[0] - p1[0]) / 6
        m[i][1] = (p3[0] - p1[0]) / 3
        m[i][2] = (p3[0] - p2[0]) / 6
        r[i] = (p3[1] - p2[1]) / (p3[0] - p2[0]) - (p2[1] - p1[1]) / (p2[0] - p1[0])
    m[n - 1][1] = 1.0
    for i in range(1, n):
        k = m[i][0] / m[i - 1][1]
        m[i][1] -= k * m[i - 1][2]
        m[i][0] = 0.0
        r[i] -= k * r[i - 1]
    for i in range(n - 2, -1, -1):
        k = m[i][2] / m[i + 1][1]
        m[i][1] -= k * m[i + 1][0]
        m[i][2] = 0.0
        r[i] -= k * r[i + 1]
    return [r[i] / m[i][1] for i in range(n)]


def _java_round(v: float) -> int:
    return int(math.floor(v + 0.5))  # Math.round(double)


def gpuimage_spline_offsets(points: Sequence[Sequence[float]]) -> np.ndarray:
    """``getPreparedSplineCurve``: natural cubic spline through (int)(x*255),(int)(y*255) points,
    returned as per-index offsets ``y - x`` (GPUImage's 'distance' with sign)."""
    srt = sorted(([float(p[0]), float(p[1])] for p in points), key=lambda p: p[0])
    conv: list[tuple[int, int]] = []
    for x, y in srt:
        px, py = int(np.float32(x) * np.float32(255)), int(np.float32(y) * np.float32(255))
        if conv and conv[-1][0] == px:  # duplicate integer x would divide by zero in Java: keep the last
            conv[-1] = (px, py)
        else:
            conv.append((px, py))
    sd = _second_derivative(conv)
    spline: list[tuple[int, int]] = []
    if sd:
        for i in range(len(sd) - 1):
            cur, nxt = conv[i], conv[i + 1]
            h = nxt[0] - cur[0]
            for x in range(cur[0], nxt[0]):
                t = (x - cur[0]) / h
                a, b = 1 - t, t
                y = a * cur[1] + b * nxt[1] + (h * h / 6) * ((a * a * a - a) * sd[i] + (b * b * b - b) * sd[i + 1])
                y = min(max(y, 0.0), 255.0)
                spline.append((x, _java_round(y)))
        if len(spline) == 255:
            spline.append(conv[-1])
    if spline and spline[0][0] > 0:  # GPUImage pads (and, quirk, duplicates the first x)
        for i in range(spline[0][0], -1, -1):
            spline.insert(0, (i, 0))
    if spline and spline[-1][0] < 255:
        for i in range(spline[-1][0] + 1, 256):
            spline.append((i, 255))
    offs = np.array([y - x for x, y in spline], np.float64)
    if len(offs) < 256:  # degenerate input: GPUImage would skip the texture upload
        offs = np.concatenate([offs, np.zeros(256 - len(offs))])
    return offs[:256]


def tone_curve_luts(rgb: CurvesValue, red: CurvesValue, green: CurvesValue, blue: CurvesValue) -> np.ndarray:
    """``updateToneCurveTexture``: per-channel byte LUTs ``clamp(i + channel + master)`` (3, 256) in 0..1."""
    master = gpuimage_spline_offsets(rgb.points)
    i = np.arange(256, dtype=np.float64)
    out = []
    for c in (red, green, blue):
        v = np.clip(i + gpuimage_spline_offsets(c.points) + master, 0, 255)
        out.append(v.astype(np.int64).astype(F32) / 255.0)  # (int) cast of a non-negative float
    return np.stack(out).astype(F32)


# --- shared GLSL helpers -----------------------------------------------------

def _fract(x: np.ndarray) -> np.ndarray:
    return x - np.floor(x)


def rgb_to_hsv_glsl(rgb: np.ndarray) -> np.ndarray:
    """GPUImageRgbToHsvFilter / ``rgbToHsv`` (Sam Hocevar's branchless form, epsilon 1e-7)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    s1 = (g >= b)
    fx = np.where(s1, g, b)
    fy = np.where(s1, b, g)
    fz = np.where(s1, F32(0.0), F32(-1.0))
    fw = np.where(s1, F32(-1.0 / 3.0), F32(2.0 / 3.0))
    s2 = (r >= fx)
    sx = np.where(s2, r, fx)
    sy = fy
    sz = np.where(s2, fz, fw)
    sw = np.where(s2, fx, r)
    delta = sx - np.minimum(sw, sy)
    eps = F32(0.0000001)
    h = np.abs(sz + (sw - sy) / (F32(6.0) * delta + eps))
    s = delta / (sx + eps)
    return np.stack([h, s, sx], -1).astype(F32)


def hsv_to_rgb_glsl(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0:1], hsv[..., 1:2], hsv[..., 2:3]
    k = np.array([0.0, 2.0 / 3.0, 1.0 / 3.0], F32)
    p = np.abs(_fract(h + k) * F32(6.0) - F32(3.0))
    c = np.clip(p - F32(1.0), 0, 1)
    return (v * (F32(1.0) + (c - F32(1.0)) * s)).astype(F32)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - F32(e0)) / F32(e1 - e0), 0, 1)
    return t * t * (F32(3.0) - F32(2.0) * t)


def _srgb_to_linear_glsl(c: np.ndarray) -> np.ndarray:
    lower = c / F32(12.92)
    upper = np.power(np.maximum((c + F32(0.055)) / F32(1.055), 0), F32(2.4))
    return np.where(c >= F32(0.04045), upper, lower).astype(F32)


def _linear_to_srgb_glsl(c: np.ndarray) -> np.ndarray:
    safe = np.maximum(c, 0)
    lower = safe * F32(12.92)
    upper = F32(1.055) * np.power(safe, F32(1.0 / 2.4)) - F32(0.055)
    return np.where(safe >= F32(0.0031308), upper, lower).astype(F32)


_M_LIN_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                       [0.2119034982, 0.6806995451, 0.1073969566],
                       [0.0883024619, 0.2817188376, 0.6299787005]], F32)
_M_LMS_LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                       [1.9779984951, -2.4285922050, 0.4505937099],
                       [0.0259040371, 0.7827717662, -0.8086757660]], F32)
_M_LAB_LMS = np.array([[1.0, 0.3963377774, 0.2158037573],
                       [1.0, -0.1055613458, -0.0638541728],
                       [1.0, -0.0894841775, -1.2914855480]], F32)
_M_LMS_LIN = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                       [-1.2684380046, 2.6097574011, -0.3413193965],
                       [-0.0041960863, -0.7034186147, 1.7076147010]], F32)


def _linear_to_oklab(c: np.ndarray) -> np.ndarray:
    lms = c @ _M_LIN_LMS.T
    lms = np.sign(lms) * np.cbrt(np.abs(lms))
    return (lms @ _M_LMS_LAB.T).astype(F32)


def _oklab_to_linear(c: np.ndarray) -> np.ndarray:
    lms = c @ _M_LAB_LMS.T
    return ((lms * lms * lms) @ _M_LMS_LIN.T).astype(F32)


# --- CMYK / Lab LUT shaders --------------------------------------------------

def apply_cmyk(rgb: np.ndarray, luts: Sequence[np.ndarray]) -> np.ndarray:
    k = F32(1.0) - rgb.max(-1)
    den = np.maximum(F32(1.0) - k, F32(0.00001))
    cmy = (F32(1.0) - rgb - k[..., None]) / den[..., None]
    c = sample_texture(luts[0], cmy[..., 0])
    m = sample_texture(luts[1], cmy[..., 1])
    y = sample_texture(luts[2], cmy[..., 2])
    k2 = sample_texture(luts[3], k)
    out = (F32(1.0) - np.stack([c, m, y], -1)) * (F32(1.0) - k2)[..., None]
    return np.clip(out, 0, 1).astype(F32)


def _lab_forward(v: np.ndarray) -> np.ndarray:
    return np.where(v > F32(0.008856), np.cbrt(v), F32(7.787) * v + F32(16.0 / 116.0)).astype(F32)


def _lab_inverse(v: np.ndarray) -> np.ndarray:
    cube = v * v * v
    return np.where(cube > F32(0.008856), cube, (v - F32(16.0 / 116.0)) / F32(7.787)).astype(F32)


def _to_linear_lab(v: np.ndarray) -> np.ndarray:
    return np.where(v <= F32(0.04045), v / F32(12.92),
                    np.power(np.maximum((v + F32(0.055)) / F32(1.055), 0), F32(2.4))).astype(F32)


def _to_srgb_lab(v: np.ndarray) -> np.ndarray:
    return np.where(v <= F32(0.0031308), v * F32(12.92),
                    F32(1.055) * np.power(np.maximum(v, 0), F32(1.0 / 2.4)) - F32(0.055)).astype(F32)


_M_RGB_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]], F32)
_M_XYZ_RGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                       [-0.9692660, 1.8760108, 0.0415560],
                       [0.0556434, -0.2040259, 1.0572252]], F32)


def apply_lab(rgb: np.ndarray, luts: Sequence[np.ndarray]) -> np.ndarray:
    xyz = _to_linear_lab(rgb) @ _M_RGB_XYZ.T
    fx = _lab_forward(xyz[..., 0] / F32(0.95047))
    fy = _lab_forward(xyz[..., 1])
    fz = _lab_forward(xyz[..., 2] / F32(1.08883))
    lab_l = (F32(116.0) * fy - F32(16.0)) / F32(100.0)
    lab_a = (F32(500.0) * (fx - fy) + F32(128.0)) / F32(255.0)
    lab_b = (F32(200.0) * (fy - fz) + F32(128.0)) / F32(255.0)
    lv = sample_texture(luts[0], np.clip(lab_l, 0, 1)) * F32(100.0)
    av = sample_texture(luts[1], np.clip(lab_a, 0, 1)) * F32(255.0) - F32(128.0)
    bv = sample_texture(luts[2], np.clip(lab_b, 0, 1)) * F32(255.0) - F32(128.0)
    fy = (lv + F32(16.0)) / F32(116.0)
    fx = fy + av / F32(500.0)
    fz = fy - bv / F32(200.0)
    xyz = np.stack([F32(0.95047) * _lab_inverse(fx), _lab_inverse(fy), F32(1.08883) * _lab_inverse(fz)], -1)
    lin = xyz @ _M_XYZ_RGB.T
    return np.clip(_to_srgb_lab(lin), 0, 1).astype(F32)


# --- HSV relation curves -------------------------------------------------------

def apply_relation(hsv: np.ndarray, name: str, lut: np.ndarray) -> np.ndarray:
    """GPUImageRelationCurveFilter for one centred curve; ``hsv`` is modified and clamped."""
    inp = {"hue_vs_sat": 0, "hue_vs_hue": 0, "hue_vs_luma": 0, "luma_vs_sat": 2, "luma_vs_hue": 2,
           "sat_vs_sat": 1}[name]
    curve = np.clip(sample_texture(lut, np.clip(hsv[..., inp], 0, 1)), 0, 1)
    delta = curve - F32(0.5)
    hsv = hsv.copy()
    if name in ("hue_vs_sat", "luma_vs_sat", "sat_vs_sat"):
        sat_delta = np.where(delta >= 0, delta * F32(2.0 / 3.0), delta * F32(2.0))
        hsv[..., 1] = np.clip(hsv[..., 1] + sat_delta, 0, 1)
    elif name == "hue_vs_luma":
        rgb = hsv_to_rgb_glsl(hsv)
        luma = rgb @ np.array([0.2126, 0.7152, 0.0722], F32)
        chroma = hsv[..., 1] * hsv[..., 2]
        w = _smoothstep(0.04, 0.30, chroma)
        target = np.clip(luma + delta * F32(2.0) * w, 0, 1)
        darker = target < luma
        scale = (target / np.maximum(luma, F32(0.00001)))[..., None]
        t = ((target - luma) / np.maximum(F32(1.0) - luma, F32(0.00001)))[..., None]
        rgb = np.where(darker[..., None], rgb * scale, rgb + (F32(1.0) - rgb) * t)
        hsv = rgb_to_hsv_glsl(np.clip(rgb, 0, 1))
    else:  # hue_vs_hue, luma_vs_hue
        hsv[..., 0] = np.mod(hsv[..., 0] + delta + F32(1.0), F32(1.0))
    return np.clip(hsv, 0, 1).astype(F32)


# --- colour wheels -------------------------------------------------------------

def _wheel_balance(x: float, y: float) -> np.ndarray:
    strength = min(math.hypot(x, y), 1.0)
    angle = math.atan2(-y, x)
    hue = ((angle - 1.5707963268) / 6.2831853072) % 1.0
    color = hsv_to_rgb_glsl(np.array([[hue, 1.0, 1.0]], F32))
    lab = _linear_to_oklab(_srgb_to_linear_glsl(color))[0]
    chroma = lab[1:3].astype(np.float64)
    return (chroma / max(float(np.hypot(*chroma)), 0.0001) * strength).astype(F32)


def apply_color_wheels(rgb: np.ndarray, wheels: dict[str, tuple[float, float]], edges: float) -> np.ndarray:
    lab = _linear_to_oklab(_srgb_to_linear_glsl(rgb))
    tonal = rgb @ np.array([0.2126, 0.7152, 0.0722], F32)
    e = min(max(edges, 0.0), 1.0)
    sharp = F32(0.8 + (16.0 - 0.8) * (e * e * (3 - 2 * e)))
    d = np.stack([tonal, np.abs(tonal - F32(0.5)), F32(1.0) - tonal], -1)
    w = np.exp(-sharp * d * d)
    w /= np.maximum(w.sum(-1, keepdims=True), F32(0.0001))
    adj = (w[..., 0:1] * _wheel_balance(*wheels["shadows"]) + w[..., 1:2] * _wheel_balance(*wheels["midtones"])
           + w[..., 2:3] * _wheel_balance(*wheels["highlights"]))
    lab[..., 1:3] += adj * F32(0.115)
    return np.clip(_linear_to_srgb_glsl(_oklab_to_linear(lab)), 0, 1).astype(F32)


# --- fused numba kernel (same maths, one pass per pixel) ------------------------

_REL_CODES = {"hue_vs_sat": 0, "hue_vs_hue": 1, "hue_vs_luma": 2, "luma_vs_sat": 3, "luma_vs_hue": 4, "sat_vs_sat": 5}
_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        from ._curves_kernel import kern

        _kernel = kern
    return _kernel


def _chain_inputs(st: CurvesState) -> dict[str, Any] | None:
    rgb_names = EDITOR_TYPES["RGB"]
    active_rgb = any(not st.curves[n].is_default for n in rgb_names)
    active_cmyk = any(not st.curves[n].is_default for n in EDITOR_TYPES["CMYK"])
    active_lab = any(not st.curves[n].is_default for n in EDITOR_TYPES["Lab"])
    relations = [n for n in CENTERED if not st.curves[n].is_default]
    wheels_active = not all(x * x + y * y < 0.000001 for x, y in st.wheels.values())
    if not (active_rgb or active_cmyk or active_lab or relations or wheels_active):
        return None
    e = min(max(st.edges, 0.0), 1.0)
    z = np.zeros((1, 256), F32)
    return {
        "do_rgb": active_rgb,
        "tone": tone_curve_luts(*(st.curves[n] for n in rgb_names)) if active_rgb else z,
        "do_cmyk": active_cmyk,
        "cmyk": np.stack([lut_bytes(st.curves[n].to_lut()) for n in EDITOR_TYPES["CMYK"]]) if active_cmyk else z,
        "do_lab": active_lab,
        "lab": np.stack([lut_bytes(st.curves[n].to_lut()) for n in EDITOR_TYPES["Lab"]]) if active_lab else z,
        "relations": relations,
        "rels": np.array([_REL_CODES[n] for n in relations], np.int64),
        "rel_luts": (np.stack([lut_bytes(st.curves[n].to_lut()) for n in relations]) if relations else z),
        "do_wheels": wheels_active,
        "bal": np.stack([_wheel_balance(*st.wheels[w]) for w in WHEEL_NAMES]).astype(np.float64),
        "sharp": 0.8 + (16.0 - 0.8) * (e * e * (3 - 2 * e)),
    }


def apply_state(img: np.ndarray, st: CurvesState) -> np.ndarray:
    """Fused numba version of :func:`apply_state_numpy` (same chain, float64 per-pixel maths)."""
    c = _chain_inputs(st)
    if c is None:
        return img
    h, w = img.shape[:2]
    px = np.ascontiguousarray(img[..., :3], dtype=F32).reshape(-1, 3)
    kern = _get_kernel()
    out = kern(px, c["do_rgb"], c["tone"].astype(F32), c["do_cmyk"], c["cmyk"].astype(F32), c["do_lab"],
               c["lab"].astype(F32), c["rels"], c["rel_luts"].astype(F32), c["do_wheels"], c["bal"], float(c["sharp"]))
    return out.reshape(h, w, 3)


# --- the whole chain (numpy reference) ---------------------------------------------

def apply_state_numpy(img: np.ndarray, st: CurvesState) -> np.ndarray:
    """``buildCurvesFilter`` order: RGB tone curve -> CMYK -> Lab -> HSV relation group -> colour wheels."""
    rgb_names = EDITOR_TYPES["RGB"]
    active_rgb = any(not st.curves[n].is_default for n in rgb_names)
    active_cmyk = any(not st.curves[n].is_default for n in EDITOR_TYPES["CMYK"])
    active_lab = any(not st.curves[n].is_default for n in EDITOR_TYPES["Lab"])
    relations = [n for n in CENTERED if not st.curves[n].is_default]
    wheels_active = not all(x * x + y * y < 0.000001 for x, y in st.wheels.values())
    if not (active_rgb or active_cmyk or active_lab or relations or wheels_active):
        return img
    out = np.clip(img[..., :3], 0, 1).astype(F32)
    if active_rgb:
        luts = tone_curve_luts(*(st.curves[n] for n in rgb_names))
        out = np.stack([sample_texture(luts[c], out[..., c]) for c in range(3)], -1)
    if active_cmyk:
        out = apply_cmyk(out, [lut_bytes(st.curves[n].to_lut()) for n in EDITOR_TYPES["CMYK"]])
    if active_lab:
        out = apply_lab(out, [lut_bytes(st.curves[n].to_lut()) for n in EDITOR_TYPES["Lab"]])
    if relations:
        hsv = np.clip(rgb_to_hsv_glsl(out), 0, 1)
        for n in relations:
            hsv = apply_relation(hsv, n, lut_bytes(st.curves[n].to_lut()))
        out = hsv_to_rgb_glsl(hsv)
    if wheels_active:
        out = apply_color_wheels(out, st.wheels, st.edges)
    return out.astype(F32)


# =============================================================================
# Operation
# =============================================================================

_PTS_DESC = ("ImageToolbox control points [[x, y], ...] in 0..1 (max 16 used). The first/last points are the "
             "fixed endpoints at x=0 and x=1 (added automatically when missing).")
_CENTER_DESC = ("Centred relation curve: y=0.5 means no change. Endpoints only mirror their neighbours, so put "
                "control points strictly inside 0<x<1.")


def _curve_params() -> list:
    titles = {"rgb": "RGB (luma master)", "red": "Red", "green": "Green", "blue": "Blue", "cyan": "Cyan",
              "magenta": "Magenta", "yellow": "Yellow", "black": "Black (K)", "lab_l": "Lab L", "lab_a": "Lab a",
              "lab_b": "Lab b", "hue_vs_sat": "Hue vs Sat", "hue_vs_hue": "Hue vs Hue",
              "hue_vs_luma": "Hue vs Luma", "luma_vs_sat": "Luma vs Sat", "luma_vs_hue": "Luma vs Hue",
              "sat_vs_sat": "Sat vs Sat"}
    out = []
    for name in CURVE_NAMES:
        centered = name in CENTERED
        out.append(Points(name, [[0, 0.5], [1, 0.5]] if centered else [[0, 0], [1, 1]], min_items=1,
                          max_items=256, min=0, max=1, title=titles[name],
                          description=_CENTER_DESC if centered else _PTS_DESC))
    return out


def _resolve_preset(ctx, preset: str, preset_file: str) -> CurvesState | None:
    if not preset and not preset_file:
        return None
    from ..tools import _curves_presets as codec

    try:
        if preset_file:
            p = ctx.resolve_file(preset_file, "curves preset")
            data = p.read_bytes()
            if len(data) > codec.MAX_PRESET_SIZE_BYTES:
                raise codec.CurvesPresetError("The curve preset is too large")
            return codec.decode(data)
        return codec.decode(preset)
    except codec.CurvesPresetError as e:
        raise invalid(f"Could not read the curves preset: {e}",
                      param="preset_file" if preset_file else "preset") from None


@op("adjust.color_curves", "Color curves (ImageToolbox)", params=[
    *_curve_params(),
    Point("wheel_shadows", (0.0, 0.0), min=-1, max=1, title="Shadows wheel",
          description="Colour-wheel offset (x right, y down; unit disc). Hue = direction, strength = distance"),
    Point("wheel_midtones", (0.0, 0.0), min=-1, max=1, title="Midtones wheel"),
    Point("wheel_highlights", (0.0, 0.0), min=-1, max=1, title="Highlights wheel"),
    Float("wheel_edges", DEFAULT_EDGES, min=0, max=1, step=1 / 15, title="Wheel edges",
          description="Tonal range separation of the three wheels (0 = broad overlap, 1 = isolated)"),
    Text("preset", "", max_length=4_000_000, multiline=True, title="Preset text",
         description="ImageToolbox .itcurve JSON, Lightroom XMP or .lrtemplate text; explicit non-default "
                     "curve/wheel params override the preset's values"),
    File("preset_file", "", file_kind="any", title="Preset file",
         description=".itcurve, .xmp, .lrtemplate, Photoshop .acv or a .zip containing one"),
], tags=["curves", "tone curve", "cmyk", "lab", "hue vs sat", "hue vs hue", "hue vs luma", "luma vs sat",
         "sat vs sat", "color wheels", "grading", "imagetoolbox", "acv", "xmp"],
    source={"project": "ImageToolbox", "feature": "Curves"},
    examples=[{"rgb": [[0, 0], [0.25, 0.2], [0.75, 0.82], [1, 1]]},
              {"hue_vs_sat": [[0, 0.5], [0.33, 0.8], [1, 0.5]]},
              {"wheel_shadows": [0.0, -0.4], "wheel_highlights": [0.0, 0.3]}])
def color_curves(img, ctx, preset, preset_file, **values):
    """ImageToolbox Curves: RGB master/R/G/B, CMYK, Lab, Hue vs Sat/Hue/Luma, Luma vs Sat/Hue,
    Sat vs Sat and shadows/midtones/highlights colour wheels, with ImageToolbox's interpolation
    (GPUImage natural spline for RGB, Catmull-Rom / Hermite LUTs for the others) and shader maths."""
    base = _resolve_preset(ctx, preset, preset_file)
    st = state_from_params(values, base)
    return apply_state(img, st)
