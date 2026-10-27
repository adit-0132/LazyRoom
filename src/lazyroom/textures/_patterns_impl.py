"""numba port of ImageToolbox's GLSL pattern texture shaders.

Source: ImageToolbox ``feature/texture-generation/.../data/PatternTextureShaders.kt``
(Apache-2.0, T8RIN).  Every shader body is translated statement by statement into
a scalar per-pixel function; GLSL vector maths is spelled out on scalars/tuples.

Uniform vector ``u`` layout (float64):
    0,1 aspect.xy | 2,3 offsetX/Y | 4 rotation (deg) | 5 frequency | 6 seed | 7.. pattern params
``col`` is (4, 4) float64: color1..color4 as RGBA 0..1.  ``aux`` holds per-image
constants that the GLSL recomputes per pixel from uniforms only (triangle vertices,
loop centres, tree positions); precomputing them gives identical values.
"""
from __future__ import annotations

import math

import numpy as np

from ._nb import njit, prange

TAU = 6.28318530718

# kind indices (order matters: kept in sync with patterns.py)
KINDS = (
    "RandomTriangles", "HexCubePattern", "SquareLooper", "MoireInterference", "TruchetArcs", "CircuitPattern",
    "RosacePattern", "RosettePattern", "LightArray", "VoronoiGems", "CitySkyline", "WinterLandscape",
    "CircleRipplePattern", "SquareRipplePattern", "HatchPattern", "SmoothHatch", "XorPattern",
    "SquareSpiralPattern", "CrossStitchPattern", "PlaidPattern", "GeneratedWaves", "VoronoiHatch", "LoopPattern",
    "ScintillatingGrid", "CircleSpiralPattern", "TestChart", "BentRows", "DiamondsIllusion",
)


# ---------------------------------------------------------------------------
# GLSL helpers / shared prelude
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _fract(v):
    return v - math.floor(v)


@njit(cache=True, inline="always")
def _mod(x, y):
    return x - y * math.floor(x / y)


@njit(cache=True, inline="always")
def _step(edge, x):
    return 0.0 if x < edge else 1.0


@njit(cache=True, inline="always")
def _clamp(v, lo, hi):
    return min(max(v, lo), hi)


@njit(cache=True, inline="always")
def _smoothstep(e0, e1, x):
    t = _clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, inline="always")
def _hash(px, py, seed):
    # fract(sin(dot(p + seed, vec2(127.1, 311.7))) * 43758.5453)
    return _fract(math.sin((px + seed) * 127.1 + (py + seed) * 311.7) * 43758.5453)


@njit(cache=True, inline="always")
def _safe_angle(x, y):
    return 0.0 if x * x + y * y < 0.0000001 else math.atan2(y, x)


@njit(cache=True, inline="always")
def _segment(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    pax = px - ax
    pay = py - ay
    h = _clamp((pax * abx + pay * aby) / max(abx * abx + aby * aby, 0.00001), 0.0, 1.0)
    dx = pax - abx * h
    dy = pay - aby * h
    return math.sqrt(dx * dx + dy * dy)


@njit(cache=True, inline="always")
def _col(col, i):
    return col[i, 0], col[i, 1], col[i, 2], col[i, 3]


@njit(cache=True, inline="always")
def _mix(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t)


@njit(cache=True)
def _palette(col, t):
    t = _fract(t) * 4.0
    if t < 1.0:
        return _mix(_col(col, 0), _col(col, 1), t)
    if t < 2.0:
        return _mix(_col(col, 1), _col(col, 2), t - 1.0)
    if t < 3.0:
        return _mix(_col(col, 2), _col(col, 3), t - 2.0)
    return _mix(_col(col, 3), _col(col, 0), t - 3.0)


@njit(cache=True, inline="always")
def _hsv_ch(h, off, s, v):
    k = _clamp(abs(_fract(h + off) * 6.0 - 3.0) - 1.0, 0.0, 1.0)
    return v * (1.0 + (k - 1.0) * s)


# ---------------------------------------------------------------------------
# Shader bodies (p = transformed coordinate)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _random_triangles(px, py, u, col, aux):
    result = _col(col, 0)
    for i in range(aux.shape[0]):  # aux rows = min(count, 100) triangles, in order
        ax, ay, bx, by, cx, cy, tcol = aux[i, 0], aux[i, 1], aux[i, 2], aux[i, 3], aux[i, 4], aux[i, 5], aux[i, 6]
        vx, vy = px - ax, py - ay
        ux, uy = bx - ax, by - ay
        wx, wy = cx - ax, cy - ay
        cross = ux * wy - uy * wx
        den = 0.00001 if abs(cross) < 0.00001 else cross
        s = (vx * wy - vy * wx) / den
        t = (ux * vy - uy * vx) / den
        if s >= 0.0 and t >= 0.0 and s + t <= 1.0:
            result = _palette(col, tcol)
    return result


@njit(cache=True)
def _hex_cube(px, py, u, col):
    thickness = u[7]
    cellx = math.floor(px / 1.73205)
    celly = math.floor(py / 1.5)
    lx = 0.0
    ly = 0.0
    nearest = 100.0
    for y in range(-1, 2):
        for x in range(-1, 2):
            cx = cellx + x
            cy = celly + y
            centerx = (cx + _mod(cy, 2.0) * 0.5) * 1.73205
            centery = cy * 1.5
            dx = px - centerx
            dy = py - centery
            dd = dx * dx + dy * dy
            if dd < nearest:
                nearest = dd
                lx = dx
                ly = dy
    angle = _fract((_safe_angle(lx, ly) + TAU * 0.25) / TAU)
    if angle < 0.333333:
        result = _col(col, 0)
    elif angle < 0.666667:
        result = _col(col, 1)
    else:
        result = _col(col, 2)
    vx = abs(lx)
    vy = abs(ly)
    edge = max(vx * 0.866025 + vy * 0.5, vy)
    if edge > 1.0 - thickness:
        result = _col(col, 3)
    return result


@njit(cache=True)
def _square_looper(px, py, u, col):
    time, thickness = u[7], u[8]
    qx = 0.7071 * px + 0.7071 * py
    qy = -0.7071 * px + 0.7071 * py
    d = max(abs(qx), abs(qy))
    wave = abs(_fract(d - time * 0.1) - 0.5)
    return _mix(_col(col, 0), _col(col, 1), 1.0 - _step(thickness * 0.5, wave))


@njit(cache=True)
def _moire(px, py, u, col):
    intensity, thickness = u[7], u[8]
    a = math.sqrt((px - 0.4) ** 2 + py * py)
    b = math.sqrt((px + 0.4) ** 2 + py * py)
    v = math.sin(a * intensity) * math.sin(b * intensity)
    return _mix(_col(col, 0), _col(col, 1), _smoothstep(thickness - 0.6, thickness - 0.4, v))


@njit(cache=True)
def _truchet_arcs(px, py, u, col):
    seed, thickness = u[6], u[7]
    cx = math.floor(px)
    cy = math.floor(py)
    qx = _fract(px)
    qy = _fract(py)
    if _hash(cx, cy, seed) > 0.5:
        qx = 1.0 - qx
    d = min(abs(math.sqrt(qx * qx + qy * qy) - 0.5), abs(math.sqrt((qx - 1.0) ** 2 + (qy - 1.0) ** 2) - 0.5))
    return _mix(_col(col, 0), _col(col, 1), 1.0 - _smoothstep(thickness * 0.5, thickness * 0.5 + 0.015, d))


@njit(cache=True)
def _circuit(px, py, u, col):
    seed, thickness, radius = u[6], u[7], u[8]
    cx = math.floor(px)
    cy = math.floor(py)
    qx = _fract(px) - 0.5
    qy = _fract(py) - 0.5
    choice = _hash(cx, cy, seed)
    if choice > 0.5:
        qx, qy = qy, qx
    if choice > 0.75:
        qx = -qx
    wire = min(_segment(qx, qy, -0.5, 0.0, 0.0, 0.0), _segment(qx, qy, 0.0, 0.0, 0.0, 0.5))
    terminal = abs(math.sqrt((qx - 0.3) ** 2 + (qy + 0.3) ** 2) - radius)
    d = min(wire, terminal)
    return _mix(_col(col, 0), _col(col, 1), 1.0 - _smoothstep(thickness, thickness + 0.01, d))


@njit(cache=True)
def _rosace(px, py, u, col, ring):
    count, radius, thickness = u[7], u[8], u[9]
    d = 10.0
    for i in range(64):
        if float(i) >= count:
            break
        angle = float(i) * TAU / count
        cr = radius if ring == 0 else 0.4
        cx = math.cos(angle) * cr
        cy = math.sin(angle) * cr
        d = min(d, abs(math.sqrt((px - cx) ** 2 + (py - cy) ** 2) - radius))
    return _mix(_col(col, 0), _col(col, 1), 1.0 - _smoothstep(thickness, thickness + 0.005, d))


@njit(cache=True)
def _light_array(px, py, u, col):
    seed, intensity, layers = u[6], u[7], u[8]
    lr = 0.0
    lg = 0.0
    lb = 0.0
    scale = 1.0
    for i in range(6):
        if float(i) >= layers:
            break
        qx = _fract(px * scale) - 0.5
        qy = _fract(py * scale) - 0.5
        cx = math.floor(px * scale)
        cy = math.floor(py * scale)
        rays = (math.exp(-abs(qx) * 120.0) * math.exp(-abs(qy) * 5.0)
                + math.exp(-abs(qy) * 120.0) * math.exp(-abs(qx) * 5.0))
        glow = math.exp(-(qx * qx + qy * qy) * 60.0)
        c = _palette(col, _hash(cx + float(i), cy + float(i), seed))
        k = (rays + glow) * intensity / scale
        lr += c[0] * k
        lg += c[1] * k
        lb += c[2] * k
        scale *= 2.0
    return (_clamp(col[0, 0] + lr, 0.0, 1.0), _clamp(col[0, 1] + lg, 0.0, 1.0),
            _clamp(col[0, 2] + lb, 0.0, 1.0), col[0, 3])


@njit(cache=True)
def _voronoi_gems(px, py, u, col):
    seed, shadows, variability = u[6], u[7], u[8]
    cellx = math.floor(px)
    celly = math.floor(py)
    chx, chy = cellx, celly
    lx = 0.0
    ly = 0.0
    nearest = 100.0
    for y in range(-1, 2):
        for x in range(-1, 2):
            cx = cellx + x
            cy = celly + y
            vx = px - cx - 0.5 - (_hash(cx, cy, seed) - 0.5) * variability
            vy = py - cy - 0.5 - (_hash(cx + 19.3, cy + 19.3, seed) - 0.5) * variability
            d = vx * vx + vy * vy
            if d < nearest:
                nearest = d
                chx, chy = cx, cy
                lx, ly = vx, vy
    facet = math.floor((_safe_angle(lx, ly) + TAU * 0.5) / TAU * 6.0)
    r = _palette(col, _hash(chx, chy, seed))
    k = 1.0 - shadows * (0.25 + 0.6 * _hash(chx + facet, chy + facet, seed))
    return (r[0] * k, r[1] * k, r[2] * k, r[3])


@njit(cache=True)
def _city_skyline(px, py, u, col):
    seed, layers = u[6], u[7]
    qx, qy = px, py
    result = _mix(_col(col, 0), _col(col, 1), _smoothstep(-0.5, 0.5, qy))
    sun = 1.0 - _smoothstep(0.13, 0.135, math.sqrt((qx - 0.18) ** 2 + (qy + 0.18) ** 2))
    result = _mix(result, _col(col, 2), sun)
    for i in range(8):
        if float(i) >= layers:
            break
        layer = float(i)
        scale = 20.0 + layer * 9.0
        x = math.floor(qx * scale)
        top = 0.02 + layer * 0.055 - _hash(x, layer, seed) * 0.17
        if qy > top:
            building = _mix(_col(col, 1), _col(col, 0), (layer + 1.0) / layers)
            wx = _fract(qx * scale * 3.0)
            wy = _fract(qy * 90.0)
            lit = _step(0.94, _hash(math.floor(qx * scale * 3.0) + layer, math.floor(qy * 90.0) + layer, seed))
            building = _mix(building, _col(col, 3), lit * _step(0.3, wx) * _step(0.3, wy))
            result = building
    return result


@njit(cache=True)
def _winter_landscape(px, py, u, col, aux):
    seed, time = u[6], u[7]
    qx, qy = px, py
    result = _mix(_col(col, 0), _col(col, 1), _clamp(qy + 0.5, 0.0, 1.0))
    moon = 1.0 - _smoothstep(0.05, 0.055, math.sqrt((qx - 0.22) ** 2 + (qy + 0.3) ** 2))
    result = _mix(result, _col(col, 3), moon)
    for i in range(4):
        layer = float(i)
        hill = 0.03 + layer * 0.12 + 0.09 * math.sin(qx * 5.0 + layer * 2.0)
        if qy > hill:
            result = _mix(_col(col, 1), _col(col, 3), layer * 0.22 + 0.2)
        for j in range(16):
            rx = aux[i * 16 + j, 0]  # hash2(vec2(j, layer))
            ry = aux[i * 16 + j, 1]
            x = rx * 1.3 - 0.65
            base = 0.03 + layer * 0.12 + 0.09 * math.sin(x * 5.0 + layer * 2.0)
            height = 0.06 + ry * 0.1
            y = qy - base
            if y < 0.0 and y > -height and abs(qx - x) < (y + height) * 0.25:
                result = _mix(_col(col, 0), _col(col, 2), 0.35 + layer * 0.15)
    sx = qx * 60.0
    sy = (qy + time * 0.05) * 60.0
    cx = math.floor(sx)
    cy = math.floor(sy)
    vx = _fract(sx) - _hash(cx, cy, seed)
    vy = _fract(sy) - _hash(cx + 19.3, cy + 19.3, seed)
    flake = (1.0 - _smoothstep(0.03, 0.08, math.sqrt(vx * vx + vy * vy))) * _step(0.8, _hash(cx, cy, seed))
    return _mix(result, _col(col, 3), flake)


@njit(cache=True)
def _circle_ripple(px, py, u, col):
    radius, aspect = u[7], u[8]
    celly = math.floor(py)
    qx = _fract(px) - 0.5
    qy = _fract(py) - 0.5
    qx += _mod(celly, 2.0) * 0.5
    qx = _mod(qx + 0.5, 1.0) - 0.5
    qy /= aspect
    d = math.sqrt(qx * qx + qy * qy)
    result = _col(col, 0)
    if d < radius:
        a = _fract(_safe_angle(qx, qy) / TAU)
        result = _col(col, 1) if a < 0.5 else _col(col, 2)
        if d < radius * 0.75:
            result = _col(col, 3)
    return result


@njit(cache=True)
def _square_ripple(px, py, u, col):
    thickness, offset = u[7], u[8]
    qx = px + math.sin(py * 1.2) * offset
    qy = py + math.sin(px * 1.2) * offset
    cx = math.floor(qx)
    cy = math.floor(qy)
    vx = _fract(qx)
    vy = _fract(qy)
    even = _mod(cx + cy, 2.0) < 1.0
    result = _col(col, 0) if even else _col(col, 1)
    if min(min(vx, 1.0 - vx), min(vy, 1.0 - vy)) < thickness:
        result = _col(col, 2) if even else _col(col, 3)
    return result


@njit(cache=True)
def _hatch(px, py, u, col):
    return _mix(_col(col, 0), _col(col, 1), _step(u[7], _fract(px)))


@njit(cache=True)
def _smooth_hatch(px, py, u, col):
    hardness, balance = u[7], u[8]
    v = 0.5 + 0.5 * math.cos(px * TAU)
    return _mix(_col(col, 0), _col(col, 1), _smoothstep(balance - 0.5 / hardness, balance + 0.5 / hardness, v))


@njit(cache=True)
def _xor(px, py, u, col):
    intensity = u[7]
    cx = math.floor(abs(px) * 8.0)
    cy = math.floor(abs(py) * 8.0)
    value = 0.0
    weight = 1.0
    for i in range(12):
        if float(i) >= intensity:
            break
        value += _mod(_mod(cx, 2.0) + _mod(cy, 2.0), 2.0) * weight
        cx = math.floor(cx * 0.5)
        cy = math.floor(cy * 0.5)
        weight *= 2.0
    return _mix(_col(col, 0), _col(col, 1), value / max(weight - 1.0, 1.0))


@njit(cache=True)
def _square_spiral(px, py, u, col):
    count, thickness = u[7], u[8]
    vx = _fract(px) - 0.5
    vy = _fract(py) - 0.5
    d = 10.0
    ax = 0.47
    ay = 0.47
    step = 0.94 / (count * 2.0)
    for i in range(48):
        if float(i) >= count * 4.0:
            break
        j = float(i)
        side = _mod(j, 4.0)
        r = 0.47 - math.floor(j / 4.0) * step
        if side < 0.5:
            bx, by = -r, r
        elif side < 1.5:
            bx, by = -r, -r
        elif side < 2.5:
            bx, by = r - step, -r
        else:
            bx, by = r - step, r - step
        d = min(d, _segment(vx, vy, ax, ay, bx, by))
        ax, ay = bx, by
    e0 = thickness * 0.235 / count
    return _mix(_col(col, 0), _col(col, 1), 1.0 - _smoothstep(e0, e0 + 0.005, d))


@njit(cache=True)
def _cross_stitch(px, py, u, col):
    thickness, shadows = u[7], u[8]
    vx = _fract(px) - 0.5
    vy = _fract(py) - 0.5
    parity = _mod(math.floor(px) + math.floor(py), 2.0)
    horizontal = abs(vy) < thickness * 0.5
    vertical = abs(vx) < thickness * 0.5
    result = _col(col, 0)
    if horizontal or vertical:
        top = horizontal and ((not vertical) or parity < 0.5)
        across = vy if top else vx
        result = _col(col, 1) if top else _col(col, 2)
        k = (1.0 - shadows * (abs(across) * 2.0 / thickness) ** 2 * 0.6) * (0.9 + 0.1 * math.cos(across * TAU * 40.0))
        result = (result[0] * k, result[1] * k, result[2] * k, result[3])
    return result


@njit(cache=True)
def _plaid(px, py, u, col):
    seed, thickness = u[6], u[7]
    x = math.floor(px)
    y = math.floor(py)
    a = _palette(col, _hash(x, 0.0, seed))
    b = _palette(col, _hash(y, 0.0, seed))
    over = _step(0.5, _fract((px + py) * 12.0))
    result = _mix(a, b, over)
    if _fract(px) < thickness * 0.2 or _fract(py) < thickness * 0.2:
        result = _mix(result, _col(col, 0), 0.6)
    return result


@njit(cache=True)
def _generated_waves(px, py, u, col):
    seed, variability = u[6], u[7]
    band = math.floor(py + math.sin(px * TAU * 0.3) * variability)
    return _palette(col, _hash(band, 1.0, seed))


@njit(cache=True)
def _voronoi_hatch(px, py, u, col):
    seed, thickness, variability = u[6], u[7], u[8]
    cellx = math.floor(px)
    celly = math.floor(py)
    chx, chy = cellx, celly
    nearest = 100.0
    for y in range(-1, 2):
        for x in range(-1, 2):
            cx = cellx + x
            cy = celly + y
            vx = px - cx - 0.5 - (_hash(cx, cy, seed) - 0.5) * variability
            vy = py - cy - 0.5 - (_hash(cx + 19.3, cy + 19.3, seed) - 0.5) * variability
            d = vx * vx + vy * vy
            if d < nearest:
                nearest = d
                chx, chy = cx, cy
    angle = _hash(chx, chy, seed) * TAU
    stripe = _fract((px * math.cos(angle) + py * math.sin(angle)) * 8.0)
    return _mix(_col(col, 0), _col(col, 1), _step(thickness, stripe))


@njit(cache=True)
def _loops(px, py, u, col, aux):
    thickness = u[9]
    d = 10.0
    for i in range(aux.shape[0]):  # rows: centre.x, centre.y, ring radius (first min(count, 64) loops)
        d = min(d, abs(math.sqrt((px - aux[i, 0]) ** 2 + (py - aux[i, 1]) ** 2) - aux[i, 2]))
    return _mix(_col(col, 0), _col(col, 1), 1.0 - _smoothstep(thickness * 0.2, thickness * 0.2 + 0.005, d))


@njit(cache=True)
def _scintillating(px, py, u, col):
    thickness, radius = u[7], u[8]
    qx = abs(_fract(px + 0.5) - 0.5)
    qy = abs(_fract(py + 0.5) - 0.5)
    line = min(qx, qy)
    result = _mix(_col(col, 0), _col(col, 1), 1.0 - _step(thickness, line))
    return _mix(result, _col(col, 2), 1.0 - _step(radius, math.sqrt(qx * qx + qy * qy)))


@njit(cache=True)
def _circle_spiral(px, py, u, col):
    count = u[7]
    phase = math.sqrt(px * px + py * py) - _safe_angle(px, py) / TAU
    return _mix(_col(col, 0), _col(col, 1), _step(0.5, _fract(phase * count)))


@njit(cache=True)
def _test_chart(px, py, u, col):
    qx = _fract(px + 0.5) * 2.0
    qy = _fract(py + 0.5) * 2.0
    if qy < 1.0 and qx < 1.0:
        v = _step(0.5, _fract(math.sqrt((qx - 0.5) ** 2 + (qy - 0.5) ** 2) * 8.0))
    elif qy < 1.0:
        v = _step(0.5, _fract((qx + qy) * 8.0))
    elif qx < 1.0:
        # early return of hsv(vec3(q.x, 1.0, 2.0 - q.y)) with alpha 1 (written un-premultiplied)
        s = 1.0
        val = 2.0 - qy
        return (_hsv_ch(qx, 0.0, s, val), _hsv_ch(qx, 0.666667, s, val), _hsv_ch(qx, 0.333333, s, val), 1.0)
    else:
        v = _mod(math.floor(qx * 8.0) + math.floor(qy * 8.0), 2.0)
    return _mix(_col(col, 0), _col(col, 1), v)


@njit(cache=True)
def _bent_rows(px, py, u, col):
    offset, thickness = u[7], u[8]
    row = math.floor(py)
    x = px + math.sin(row * 1.7) * offset
    v = _mod(math.floor(x) + row, 2.0)
    result = _mix(_col(col, 0), _col(col, 1), v)
    if _fract(py) < thickness:
        result = _col(col, 2)
    return result


@njit(cache=True)
def _diamonds(px, py, u, col):
    count, thickness = u[7], u[8]
    qx = _fract(px) - 0.5
    qy = _fract(py) - 0.5
    d = max(abs(qx), abs(qy))
    phase = d * count * 4.0
    if qx + qy > 0.0:
        phase += 0.5
    if qx - qy > 0.0:
        phase += 0.5
    return _mix(_col(col, 0), _col(col, 1), _step(thickness, _fract(phase)))


# ---------------------------------------------------------------------------

@njit(cache=True)
def _shade(kind, px, py, u, col, aux):
    if kind == 0:
        return _random_triangles(px, py, u, col, aux)
    if kind == 1:
        return _hex_cube(px, py, u, col)
    if kind == 2:
        return _square_looper(px, py, u, col)
    if kind == 3:
        return _moire(px, py, u, col)
    if kind == 4:
        return _truchet_arcs(px, py, u, col)
    if kind == 5:
        return _circuit(px, py, u, col)
    if kind == 6:
        return _rosace(px, py, u, col, 0)
    if kind == 7:
        return _rosace(px, py, u, col, 1)
    if kind == 8:
        return _light_array(px, py, u, col)
    if kind == 9:
        return _voronoi_gems(px, py, u, col)
    if kind == 10:
        return _city_skyline(px, py, u, col)
    if kind == 11:
        return _winter_landscape(px, py, u, col, aux)
    if kind == 12:
        return _circle_ripple(px, py, u, col)
    if kind == 13:
        return _square_ripple(px, py, u, col)
    if kind == 14:
        return _hatch(px, py, u, col)
    if kind == 15:
        return _smooth_hatch(px, py, u, col)
    if kind == 16:
        return _xor(px, py, u, col)
    if kind == 17:
        return _square_spiral(px, py, u, col)
    if kind == 18:
        return _cross_stitch(px, py, u, col)
    if kind == 19:
        return _plaid(px, py, u, col)
    if kind == 20:
        return _generated_waves(px, py, u, col)
    if kind == 21:
        return _voronoi_hatch(px, py, u, col)
    if kind == 22:
        return _loops(px, py, u, col, aux)
    if kind == 23:
        return _scintillating(px, py, u, col)
    if kind == 24:
        return _circle_spiral(px, py, u, col)
    if kind == 25:
        return _test_chart(px, py, u, col)
    if kind == 26:
        return _bent_rows(px, py, u, col)
    return _diamonds(px, py, u, col)


@njit(cache=True)
def prepare_aux(kind, u):
    """Per-image constants computed exactly as the GLSL does per pixel (uniform-only inputs)."""
    seed = u[6]
    if kind == 0:  # RandomTriangles: vertices a, b, c and palette position, for i < count (<= 100)
        n_tri = 0
        for i in range(100):
            if float(i) >= u[7]:
                break
            n_tri += 1
        aux = np.empty((n_tri, 7))
        for i in range(n_tri):
            n = float(i)
            aux[i, 0] = _hash(n, 1.0, seed) - 0.5
            aux[i, 1] = _hash(n + 19.3, 1.0 + 19.3, seed) - 0.5
            aux[i, 2] = _hash(n, 2.0, seed) - 0.5
            aux[i, 3] = _hash(n + 19.3, 2.0 + 19.3, seed) - 0.5
            aux[i, 4] = _hash(n, 3.0, seed) - 0.5
            aux[i, 5] = _hash(n + 19.3, 3.0 + 19.3, seed) - 0.5
            aux[i, 6] = _hash(n, 4.0, seed)
        return aux
    if kind == 11:  # WinterLandscape: r = hash2(vec2(j, layer))
        aux = np.empty((64, 2))
        for i in range(4):
            for j in range(16):
                aux[i * 16 + j, 0] = _hash(float(j), float(i), seed)
                aux[i * 16 + j, 1] = _hash(float(j) + 19.3, float(i) + 19.3, seed)
        return aux
    if kind == 22:  # LoopPattern: centre = hash2(vec2(i, 1)) - .5, radius*(.5 + hash(vec2(i, 3)))
        n_loop = 0
        for i in range(64):
            if float(i) >= u[7]:
                break
            n_loop += 1
        aux = np.empty((n_loop, 3))
        for i in range(n_loop):
            n = float(i)
            aux[i, 0] = (_hash(n, 1.0, seed) - 0.5) * 1.0
            aux[i, 1] = (_hash(n + 19.3, 1.0 + 19.3, seed) - 0.5) * 1.0
            aux[i, 2] = u[8] * (0.5 + _hash(n, 3.0, seed))
        return aux
    return np.zeros((0, 1))


@njit(cache=True, parallel=True)
def render_pattern(kind, width, height, u, col, aux):
    out = np.empty((height, width, 4), np.float32)
    ax = u[0]
    ay = u[1]
    a = math.radians(u[4])
    ca = math.cos(a)
    sa = math.sin(a)
    freq = u[5]
    for y in prange(height):
        ty = (y + 0.5) / height
        for x in range(width):
            tx = (x + 0.5) / width
            p0 = (tx - 0.5) * ax - u[2]
            p1 = (ty - 0.5) * ay - u[3]
            # mat2(cos,-sin,sin,cos) is column-major: (cos*x + sin*y, -sin*x + cos*y)
            px = (ca * p0 + sa * p1) * freq
            py = (-sa * p0 + ca * p1) * freq
            r = _shade(kind, px, py, u, col, aux)
            alpha = _clamp(r[3], 0.0, 1.0)
            if alpha > 0.0:
                out[y, x, 0] = _clamp(r[0] * alpha, 0.0, 1.0) / alpha
                out[y, x, 1] = _clamp(r[1] * alpha, 0.0, 1.0) / alpha
                out[y, x, 2] = _clamp(r[2] * alpha, 0.0, 1.0) / alpha
            else:
                out[y, x, 0] = 0.0
                out[y, x, 1] = 0.0
                out[y, x, 2] = 0.0
            out[y, x, 3] = alpha
    return out
