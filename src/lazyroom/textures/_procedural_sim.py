"""Fast-noise "simulation" textures (native types 71..121) - numba port.

Line-by-line port of ``simulationField()`` and ``shadeSimulation()`` in
ImageToolboxLibs ``fast-noise/src/main/cpp/textures/ProceduralTextures.cpp``
(Apache-2.0, T8RIN).  Like the JNI entry point, a scalar field is computed for
every pixel first (stored as float32, as the C++ ``std::vector<float>``), then
each pixel is shaded from its clamped 4-neighbourhood (gradient, Laplacian,
palette, lighting).

Arithmetic runs in float64 (upstream float32) except ``hash2`` (float32-emulated
``textures::hash``) and the parameter products that feed ``floor``/``int``
casts (``_m32``), which are rounded to float32 like upstream so integer counts
(branches, iterations, symmetry...) match exactly.
"""
from __future__ import annotations

import math

import numpy as np

from ._nb import njit, prange
from ._noisebank import fbm, fine, warp
from ._texmath import PI, clamp01, color, contrast, fract, hash2, mix_color, palette4, parameter, smoothstep

_SQRT3_F = 1.7320508


@njit(cache=True, inline="always")
def _m32(a, b):
    """float32 product (operands are float32-representable), for values that are floored / truncated."""
    return float(np.float32(a) * np.float32(b))


@njit(cache=True, inline="always")
def _div32(a, b):
    """float32 quotient (lattice coordinates that are floored, like upstream float math)."""
    return float(np.float32(a) / np.float32(b))


@njit(cache=True, inline="always")
def _lin32(a, x, k):
    """float32 ``a + x * k``."""
    return float(np.float32(a) + np.float32(x) * np.float32(k))


@njit(cache=True, inline="always")
def _inv_lin32(a, x, k):
    """float32 ``a + (1 - x) * k``."""
    return float(np.float32(a) + (np.float32(1.0) - np.float32(x)) * np.float32(k))


@njit(cache=True, inline="always")
def _shift_mul32(u, grid):
    """float32 ``(u + 0.5) * grid``."""
    return float((np.float32(u) + np.float32(0.5)) * np.float32(grid))


@njit(cache=True)
def _segment_distance(u, v, ax, ay, bx, by):
    vx = bx - ax
    vy = by - ay
    length_squared = max(vx * vx + vy * vy, 0.000001)
    t = clamp01(((u - ax) * vx + (v - ay) * vy) / length_squared)
    dx = u - ax - vx * t
    dy = v - ay - vy * t
    return math.sqrt(dx * dx + dy * dy)


@njit(cache=True)
def _jittered_point(px, py, spacing, seed, seed_y, amount):
    jitter_x = (hash2(float(px), float(py), seed) - 0.5) * spacing * amount
    jitter_y = (hash2(float(py), float(px), seed_y) - 0.5) * spacing * amount
    return (px + 0.5) * spacing + jitter_x, (py + 0.5) * spacing + jitter_y


@njit(cache=True)
def _droplet_layer(u, v, seed, distortion, grid, layer_seed, stretch):
    gx = float(np.float32(u) * np.float32(grid))
    gy = float(np.float32(v) * np.float32(grid))
    ix = int(math.floor(gx))
    iy = int(math.floor(gy))
    local_x = fract(gx) - 0.5
    local_y = fract(gy) - 0.5
    local_x += (hash2(float(ix), float(iy), seed + layer_seed) - 0.5) * 0.56
    local_y += (hash2(float(iy), float(ix), seed + layer_seed + 19) - 0.5) * 0.5
    drop_seed = hash2(float(ix * 7), float(iy * 11), seed + layer_seed + 41)
    radius_x = 0.16 + drop_seed * 0.11
    radius_y = radius_x * stretch
    teardrop_x = local_x / radius_x
    teardrop_y = (local_y + max(local_y, 0.0) * 0.28) / radius_y
    distance = math.sqrt(teardrop_x * teardrop_x + teardrop_y * teardrop_y)
    dome = clamp01(1.0 - distance)
    rim = 1.0 - smoothstep(0.72, 1.0, distance)
    streak = (1.0 - smoothstep(0.0, 0.18, abs(teardrop_x))) * smoothstep(0.15, 0.48, local_y) * distortion
    return clamp01(dome * dome * 0.82 + rim * 0.16 + streak * 0.45)


@njit(cache=True, inline="always")
def _clamp32(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@njit(cache=True)
def _smoothstep32(e0, e1, v):
    if e0 == e1:
        return np.float32(0.0) if v < e0 else np.float32(1.0)
    t = _clamp32((v - e0) / (e1 - e0), np.float32(0.0), np.float32(1.0))
    return t * t * (np.float32(3.0) - np.float32(2.0) * t)


@njit(cache=True)
def _escape_time32(t, x, y, width, height, p):
    """Mandelbrot / Burning Ship / Julia (types 96..98) evaluated entirely in float32 like upstream:
    escape-time iteration is chaotic near the set boundary, so float64 would visibly change those pixels."""
    f = np.float32
    s = f(max(parameter(p, 0, 0.009), 0.00001))
    complexity = f(clamp01(parameter(p, 1, 0.62)))
    distortion = f(clamp01(parameter(p, 2, 0.48)))
    unit = f(min(width, height))
    ss = _clamp32(s / f(0.008), f(0.125), f(5.0))
    u = (f(x) - f(width) * f(0.5)) / unit * ss
    v = (f(y) - f(height) * f(0.5)) / unit * ss
    c01 = _clamp32((s - f(0.001)) / f(0.039), f(0.0), f(1.0))
    if t == 96:
        magnification = np.exp2(c01 * f(8.0))
        focus = _smoothstep32(f(1.6), f(8.0), magnification)
        center_x = f(-0.5) + (f(-0.7436439) + f(0.5)) * focus + (distortion - f(0.5)) * f(0.16) / magnification
        center_y = f(0.1318259) * focus + (distortion - f(0.5)) * f(0.1) / magnification
        cx = u / ss * f(3.1) / magnification + center_x
        cy = v / ss * f(3.1) / magnification + center_y
        zx = f(0.0)
        zy = f(0.0)
        max_iterations = 42 + int(complexity * f(104.0))
    elif t == 97:
        magnification = np.exp2(c01 * f(7.5))
        focus = _smoothstep32(f(1.7), f(7.0), magnification)
        center_x = f(-0.5) + (f(-1.7443359) + f(0.5)) * focus + (distortion - f(0.5)) * f(0.18) / magnification
        center_y = f(-0.48) + (f(-0.017451) + f(0.48)) * focus
        cx = u / ss * f(3.2) / magnification + center_x
        cy = v / ss * f(3.2) / magnification + center_y
        zx = f(0.0)
        zy = f(0.0)
        max_iterations = 42 + int(complexity * f(100.0))
    else:
        magnification = np.exp2(c01 * f(7.0))
        focus = _smoothstep32(f(1.6), f(6.0), magnification)
        zx = u / ss * f(2.7) / magnification - f(0.11) * focus
        zy = v / ss * f(2.7) / magnification + f(0.64) * focus
        phase = distortion * f(2.0) * f(PI)
        cx = f(-0.73) + np.cos(phase) * f(0.12)
        cy = np.sin(phase) * f(0.21)
        max_iterations = 42 + int(complexity * f(104.0))
    iteration = 0
    while iteration < max_iterations and zx * zx + zy * zy < f(16.0):
        if t == 97:
            zx = abs(zx)
            zy = abs(zy)
        next_x = zx * zx - zy * zy + cx
        zy = f(2.0) * zx * zy + cy
        zx = next_x
        iteration += 1
    return float(f(iteration) / f(max_iterations))


@njit(cache=True)
def _phyllotaxis32(u, v, complexity, distortion):
    """Phyllotaxis Bloom (type 112) in float32 like upstream: seed angles reach ~1000 rad, where float32
    cos/sin differ from float64 by ~1e-4 and the 0.004..0.012 seed-edge smoothstep magnifies that."""
    f = np.float32
    u32 = f(u)
    v32 = f(v)
    seed_count = 150 + int(f(complexity) * f(330.0))
    radius = np.sqrt(u32 * u32 + v32 * v32)
    normalized_radius = _clamp32(radius / f(0.48), f(0.0), f(1.0))
    estimated = int(normalized_radius * normalized_radius * f(seed_count))
    nearest = f(1.0)
    golden_angle = f(2.3999632) + (f(distortion) - f(0.5)) * f(0.24)
    for offset in range(-7, 8):
        index = estimated + offset
        if index < 0:
            index = 0
        elif index > seed_count - 1:
            index = seed_count - 1
        seed_radius = np.sqrt(f(index) / f(seed_count)) * f(0.47)
        seed_angle = f(index) * golden_angle
        px = np.cos(seed_angle) * seed_radius
        py = np.sin(seed_angle) * seed_radius
        d = np.sqrt((u32 - px) * (u32 - px) + (v32 - py) * (v32 - py))
        if d < nearest:
            nearest = d
    return clamp01(1.0 - _smoothstep32(f(0.004), f(0.012), nearest))


@njit(cache=True, inline="always")
def _needs_noise(t):
    return (t == 71 or t == 72 or t == 73 or t == 76 or t == 77 or t == 80 or t == 81 or t == 82 or t == 83
            or t == 85 or t == 87 or t == 88 or t == 90 or t == 94 or t == 103 or t == 104 or t == 107
            or t == 108 or t == 109 or t == 119)


@njit(cache=True)
def _field(t, seed, x, y, width, height, p):
    s = max(parameter(p, 0, 0.009), 0.00001)
    complexity = clamp01(parameter(p, 1, 0.62))
    distortion = clamp01(parameter(p, 2, 0.48))
    # u, v and the spatial scale are evaluated in float32 like upstream: they feed floor()/int() lattices
    # (cells, grids), where a float64 evaluation would shift exact lattice-aligned pixels
    unit32 = np.float32(min(width, height))
    ss32 = np.float32(s) / np.float32(0.008)
    if ss32 < np.float32(0.125):
        ss32 = np.float32(0.125)
    elif ss32 > np.float32(5.0):
        ss32 = np.float32(5.0)
    spatial_scale = float(ss32)
    u = float((np.float32(x) - np.float32(width) * np.float32(0.5)) / unit32 * ss32)
    v = float((np.float32(y) - np.float32(height) * np.float32(0.5)) / unit32 * ss32)
    nx = x
    ny = y
    a = 0.0
    b = 0.0
    # cases that never read nx/ny/a/b skip the (side-effect free) warp + noise evaluation
    if _needs_noise(t):
        nx, ny = warp(seed, x, y, s * 0.65, distortion * 62.0)
        a = fbm(seed, nx * s, ny * s)
        b = fine(seed, nx * s * 3.8, ny * s * 3.8)
    radius = math.sqrt(u * u + v * v)
    angle = math.atan2(v, u)

    if t == 96 or t == 97 or t == 98:
        return _escape_time32(t, x, y, width, height, p)
    elif t == 71:
        return clamp01(0.5 + 0.5 * math.sin((a * 1.7 + b * 0.8) * (10.0 + complexity * 18.0)))
    elif t == 72:
        grid_x = _div32(u, 0.68)
        grid_y = _div32(v, 0.68)
        cell_x = int(math.floor(grid_x))
        cell_y = int(math.floor(grid_y))
        local_x = fract(grid_x) - 0.5
        local_y = fract(grid_y) - 0.5
        colony_seed = hash2(float(cell_x), float(cell_y), seed)
        sway = math.sin(local_y * (7.0 + colony_seed * 5.0) + colony_seed * PI * 2.0) * distortion * 0.045
        trunk_width = 0.018 + colony_seed * 0.012
        coral = ((1.0 - smoothstep(trunk_width, trunk_width + 0.018, abs(local_x - sway)))
                 * smoothstep(0.48, 0.31, local_y) * smoothstep(-0.48, -0.38, local_y))
        limb_count = 4 + int(_m32(complexity, 6.0))
        for i in range(limb_count):
            origin_y = 0.3 - float(i) * 0.65 / float(limb_count)
            side = -1.0 if ((i + cell_x + cell_y) & 1) == 0 else 1.0
            limb_seed = hash2(float(cell_x * 13 + i), float(cell_y), seed + 23)
            reach = 0.13 + limb_seed * 0.18
            dy = local_y - origin_y
            progress = smoothstep(0.015, -0.22, dy)
            branch_x = sway + side * reach * progress + math.sin(dy * 21.0 + limb_seed * 5.0) * distortion * 0.018
            branch = 1.0 - smoothstep(0.012, 0.032, abs(local_x - branch_x))
            branch *= smoothstep(-0.27, -0.05, dy) * (1.0 - smoothstep(-0.015, 0.035, dy))
            polyp = 1.0 - smoothstep(0.025, 0.065, math.sqrt((local_x - branch_x) * (local_x - branch_x)
                                                             + (dy + 0.24) * (dy + 0.24)))
            coral = max(coral, branch)
            coral = max(coral, polyp)
        return clamp01(coral * (0.72 + b * 0.28))
    elif t == 73:
        flow = math.sin((u + math.sin(v * 8.0 + a * 3.0) * distortion * 0.18) * (28.0 + complexity * 62.0))
        split = math.sin((v - math.cos(u * 11.0) * distortion * 0.14) * (17.0 + complexity * 39.0))
        trails = min(abs(flow), abs(flow * 0.62 + split * 0.38))
        return clamp01(1.0 - trails * 4.0)
    elif t == 74:
        grid_x = float(np.float32(u) / np.float32(0.72) + np.float32(0.5))
        grid_y = float(np.float32(v) / np.float32(0.72) + np.float32(0.5))
        local_x = fract(grid_x) - 0.5
        local_y = fract(grid_y) - 0.5
        cell_x = int(math.floor(grid_x))
        cell_y = int(math.floor(grid_y))
        crystal_seed = hash2(float(cell_x), float(cell_y), seed)
        local_radius = math.sqrt(local_x * local_x + local_y * local_y)
        local_angle = math.atan2(local_y, local_x) + crystal_seed * 0.35
        arms = 5.0 + math.floor(_m32(complexity, 5.0))
        sector = 2.0 * PI / arms
        folded_angle = abs(fract((local_angle + PI) / sector) - 0.5) * sector
        branch_distance = local_radius * math.sin(folded_angle)
        trunk = 1.0 - smoothstep(0.008, 0.023, abs(branch_distance))
        side_branches = 1.0 - smoothstep(0.0, 0.065, abs(math.sin(local_radius * (65.0 + distortion * 55.0)
                                                                     - folded_angle * 9.0)))
        side_branches *= smoothstep(0.04, 0.31, local_radius) * (1.0 - smoothstep(0.3, 0.42, local_radius))
        crystal = max(trunk, side_branches * 0.72)
        return clamp01(crystal * (1.0 - smoothstep(0.34, 0.46, local_radius)))
    elif t == 75:
        grid_x = float(np.float32(u) / np.float32(0.58) + np.float32(0.5))
        grid_y = float(np.float32(v) / np.float32(0.92) + np.float32(0.5))
        cell_x = int(math.floor(grid_x))
        cell_y = int(math.floor(grid_y))
        local_x = fract(grid_x) - 0.5
        local_y = fract(grid_y) - 0.5
        bolt_seed = hash2(float(cell_x), float(cell_y), seed)
        main_path = (math.sin(local_y * (12.0 + bolt_seed * 9.0) + bolt_seed * 8.0) * (0.025 + distortion * 0.07)
                     + math.sin(local_y * 37.0 + bolt_seed * 17.0) * distortion * 0.018)
        arc = 1.0 - smoothstep(0.008, 0.022, abs(local_x - main_path))
        branches = 2 + int(_m32(complexity, 5.0))
        for i in range(branches):
            fork_y = 0.28 - float(i) * 0.56 / max(branches - 1, 1)
            dy = local_y - fork_y
            side = -1.0 if ((i + cell_x) & 1) == 0 else 1.0
            progress = smoothstep(-0.02, -0.26, dy)
            branch_path = main_path + side * progress * (0.09 + 0.045 * i)
            branch = 1.0 - smoothstep(0.006, 0.018, abs(local_x - branch_path))
            branch *= smoothstep(-0.3, -0.04, dy) * (1.0 - smoothstep(-0.015, 0.035, dy))
            arc = max(arc, branch * 0.82)
        return clamp01(arc * smoothstep(0.5, 0.42, abs(local_y)))
    elif t == 76:
        tracks = 0.0
        spacing = 0.09 + (1.0 - complexity) * 0.08
        family_slope = -0.72 + distortion * 1.44
        estimated_track = int(math.floor((v - family_slope * u) / spacing))
        for offset in range(-2, 3):
            track_index = estimated_track + offset
            track_seed = hash2(float(track_index), 3.7, seed)
            slope = family_slope + (track_seed - 0.5) * 0.65
            path_y = (track_index * spacing + slope * u
                      + math.sin(u * (2.5 + track_seed * 5.0) + track_seed * 9.0) * distortion * 0.045)
            track_width = 0.0025 + hash2(float(track_index), 8.1, seed + 17) * 0.0045
            track = 1.0 - smoothstep(track_width, track_width * 2.6, abs(v - path_y))
            droplets = smoothstep(0.82, 1.0, math.sin(u * (48.0 + track_seed * 42.0) + track_seed * 13.0) * 0.5 + 0.5)
            tracks = max(tracks, track)
            tracks = max(tracks, track * droplets)
        return clamp01(tracks * (0.78 + b * 0.22))
    elif t == 77:
        return clamp01(0.5 + 0.5 * math.sin((nx - ny * 0.35) * s * (9.0 + complexity * 22.0) + a * 12.0))
    elif t == 78:
        grid = _lin32(4.0, complexity, 8.0)
        sx = _shift_mul32(u, grid)
        sy = _shift_mul32(v, grid)
        gx = fract(sx) - 0.5
        gy = fract(sy) - 0.5
        ix = int(math.floor(sx))
        iy = int(math.floor(sy))
        tilt = (hash2(float(ix), float(iy), seed) - 0.5) * distortion
        embryo = math.sqrt((gx + gy * tilt) * (gx + gy * tilt) / 0.09 + gy * gy / 0.16)
        membrane = 1.0 - smoothstep(0.78, 1.0, embryo)
        nucleus = 1.0 - smoothstep(0.08, 0.18, math.sqrt((gx + 0.1) * (gx + 0.1) + (gy - 0.04) * (gy - 0.04)))
        return clamp01(membrane * 0.45 + nucleus * 0.75)
    elif t == 79:
        spacing = float(np.float32(0.31) - np.float32(complexity) * np.float32(0.1))
        base_x = int(math.floor(_div32(u, spacing)))
        base_y = int(math.floor(_div32(v, spacing)))
        network = 0.0
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                neuron_x = base_x + ox
                neuron_y = base_y + oy
                px, py = _jittered_point(neuron_x, neuron_y, spacing, seed, seed + 29, 0.5)
                soma_distance = math.sqrt((u - px) * (u - px) + (v - py) * (v - py))
                soma = 1.0 - smoothstep(spacing * 0.055, spacing * 0.16, soma_distance)
                network = max(network, soma)
                rx, ry = _jittered_point(neuron_x + 1, neuron_y, spacing, seed, seed + 29, 0.5)
                dxn = neuron_x + (1 if (neuron_y & 1) == 0 else -1)
                qx, qy = _jittered_point(dxn, neuron_y + 1, spacing, seed, seed + 29, 0.5)
                axon = 1.0 - smoothstep(spacing * 0.012, spacing * (0.035 + distortion * 0.025),
                                        min(_segment_distance(u, v, px, py, rx, ry),
                                            _segment_distance(u, v, px, py, qx, qy)))
                network = max(network, axon * 0.72)
        return clamp01(network)
    elif t == 80:
        left = math.atan2(v, u + 0.18)
        right = math.atan2(v, u - 0.18)
        return clamp01(0.5 + 0.5 * math.sin((left - right) * (8.0 + complexity * 18.0) + a * distortion * 4.0))
    elif t == 81:
        frequency = 4.0 + complexity * 12.0
        meander = u + math.sin(v * 3.7 + a * 2.0) * distortion * 0.16
        trunk = abs(math.sin(meander * frequency * PI))
        tributary_a = abs(math.sin((u * 0.62 + v * 0.78 + math.sin(v * 5.0) * distortion * 0.08) * frequency * 1.7 * PI))
        tributary_b = abs(math.sin((u * 0.58 - v * 0.82 + math.cos(v * 4.2) * distortion * 0.07) * frequency * 2.3 * PI))
        hierarchy = min(trunk, min(tributary_a * 1.25, tributary_b * 1.55))
        width_ = 0.045 + (0.5 + 0.5 * math.sin(v * frequency * 0.7)) * 0.028
        return clamp01(1.0 - smoothstep(width_ * 0.35, width_, hierarchy))
    elif t == 82:
        rosettes = (math.sin(u * (18.0 + complexity * 38.0) + math.sin(v * 13.0) * 2.0)
                    * math.sin(v * (21.0 + complexity * 31.0) + math.cos(u * 11.0) * 2.0))
        patches = smoothstep(0.38 - distortion * 0.3, 0.82, a)
        return clamp01((0.5 + rosettes * 0.5) * patches + b * 0.18)
    elif t == 83:
        spacing = _inv_lin32(0.15, complexity, 0.12)
        base_x = int(math.floor(_div32(u, spacing)))
        base_y = int(math.floor(_div32(v, spacing)))
        rods = 0.0
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                bacteria_x = base_x + ox
                bacteria_y = base_y + oy
                bacteria_seed = hash2(float(bacteria_x), float(bacteria_y), seed)
                if bacteria_seed < 0.18 + (1.0 - complexity) * 0.28:
                    continue
                cx = ((bacteria_x + 0.5) * spacing
                      + (hash2(float(bacteria_x), float(bacteria_y), seed + 13) - 0.5) * spacing * 0.58)
                cy = ((bacteria_y + 0.5) * spacing
                      + (hash2(float(bacteria_y), float(bacteria_x), seed + 31) - 0.5) * spacing * 0.58)
                orientation = bacteria_seed * PI
                dx = u - cx
                dy = v - cy
                along = dx * math.cos(orientation) + dy * math.sin(orientation)
                across = -dx * math.sin(orientation) + dy * math.cos(orientation)
                half_length = spacing * (0.22 + bacteria_seed * 0.16)
                body_x = max(abs(along) - half_length, 0.0)
                body_distance = math.sqrt(body_x * body_x + across * across)
                body = 1.0 - smoothstep(spacing * 0.055, spacing * 0.11, body_distance)
                nucleus = 1.0 - smoothstep(spacing * 0.02, spacing * 0.055, math.sqrt(along * along + across * across))
                rods = max(rods, body * 0.7 + nucleus * 0.42)
        return clamp01(a * 0.12 + rods)
    elif t == 84:
        vortices = 0.0
        for i in range(4):
            cx = (hash2(float(i), 3.7, seed) - 0.5) * 0.72
            cy = (hash2(8.4, float(i), seed + 13) - 0.5) * 0.72
            dx = u - cx
            dy = v - cy
            local_radius = math.sqrt(dx * dx + dy * dy)
            spiral = math.sin(math.atan2(dy, dx) * (2.0 + i) + local_radius * (42.0 + complexity * 58.0))
            vortices += spiral * math.exp(-local_radius * (7.0 - distortion * 3.0))
        return clamp01(0.5 + vortices * 0.34)
    elif t == 85:
        shards = 0.0
        shard_count = 7 + int(_m32(complexity, 13.0))
        for i in range(shard_count):
            direction = 2.0 * PI * float(i) / float(shard_count)
            projection = u * math.cos(direction) + v * math.sin(direction)
            lateral = abs(-u * math.sin(direction) + v * math.cos(direction))
            shard = smoothstep(-0.03, 0.05, projection) * (1.0 - smoothstep(0.018, 0.055 + projection * 0.16, lateral))
            shards = max(shards, shard)
        return clamp01(shards * (0.75 + b * 0.25))
    elif t == 86:
        spacing = _inv_lin32(0.27, complexity, 0.13)
        base_x = int(math.floor(_div32(u, spacing)))
        base_y = int(math.floor(_div32(v, spacing)))
        web = 0.0
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                node_x = base_x + ox
                node_y = base_y + oy
                px, py = _jittered_point(node_x, node_y, spacing, seed, seed + 37, 0.62)
                node_distance = math.sqrt((u - px) * (u - px) + (v - py) * (v - py))
                halo = math.exp(-node_distance * node_distance / (spacing * spacing * 0.025))
                web = max(web, halo)
                rx, ry = _jittered_point(node_x + 1, node_y, spacing, seed, seed + 37, 0.62)
                ux, uy = _jittered_point(node_x, node_y + 1, spacing, seed, seed + 37, 0.62)
                link_distance = min(_segment_distance(u, v, px, py, rx, ry), _segment_distance(u, v, px, py, ux, uy))
                if hash2(float(node_x), float(node_y), seed + 71) > 0.48:
                    qx, qy = _jittered_point(node_x + 1, node_y + 1, spacing, seed, seed + 37, 0.62)
                    link_distance = min(link_distance, _segment_distance(u, v, px, py, qx, qy))
                filament = 1.0 - smoothstep(spacing * 0.008, spacing * (0.025 + distortion * 0.025), link_distance)
                web = max(web, filament * 0.78)
        return clamp01(web)
    elif t == 87:
        main_vein = 1.0 - smoothstep(0.0, 0.035, abs(u))
        side = 1.0 - abs(math.sin((v + a * 0.09) * (22.0 + complexity * 38.0) + abs(u) * 46.0))
        return clamp01(main_vein + side * 0.6 + a * 0.22)
    elif t == 88:
        pores = (math.sin(u * (42.0 + complexity * 72.0))
                 + math.sin((u * 0.5 + v * 0.866) * (45.0 + complexity * 68.0))
                 + math.sin((-u * 0.5 + v * 0.866) * (39.0 + complexity * 76.0)))
        holes = smoothstep(1.25 - distortion * 0.55, 2.55, pores)
        return clamp01((1.0 - holes) * 0.72 + b * 0.28)
    elif t == 89:
        large_drops = _droplet_layer(u, v, seed, distortion, _lin32(5.0, complexity, 7.0), 0, 1.35)
        small_drops = _droplet_layer(u, v, seed, distortion, _lin32(9.0, complexity, 12.0), 97, 1.05)
        return clamp01(max(large_drops, small_drops * 0.72))
    elif t == 90:
        embers = smoothstep(0.985 - complexity * 0.025, 1.0,
                            hash2(math.floor(x * 0.45), math.floor(y * 0.45), seed))
        return clamp01(a * 0.25 + embers * (0.65 + b * 0.35))
    elif t == 91:
        foam = 0.0
        wave_count = 4 + int(_m32(complexity, 8.0))
        for i in range(wave_count):
            cx = hash2(float(i), 2.6, seed) - 0.5
            cy = hash2(7.1, float(i), seed + 37) - 0.5
            distance = math.sqrt((u - cx) * (u - cx) + (v - cy) * (v - cy))
            foam += math.sin(distance * (48.0 + distortion * 82.0) + i * 1.9)
        return clamp01(0.5 + foam / float(wave_count) * 0.5)
    elif t == 92:
        mode_a = 2.0 + math.floor(_m32(complexity, 6.0))
        mode_b = mode_a + 1.0 + math.floor(_m32(distortion, 4.0))
        plate_x = u + 0.5
        plate_y = v + 0.5
        vibration = (math.sin(mode_a * PI * plate_x) * math.sin(mode_b * PI * plate_y)
                     - math.sin(mode_b * PI * plate_x) * math.sin(mode_a * PI * plate_y))
        return clamp01(1.0 - abs(vibration) * 4.5)
    elif t == 93:
        petals = 5.0 + math.floor(_m32(complexity, 15.0))
        radial_wave = math.sin(radius * (72.0 + complexity * 120.0)
                               + math.sin(angle * petals) * (2.0 + distortion * 8.0))
        angular_wave = math.cos(angle * petals + radius * 18.0)
        return clamp01(1.0 - abs(radial_wave * 0.72 + angular_wave * 0.28) * 2.8)
    elif t == 94:
        branches = 7.0 + math.floor(_m32(complexity, 22.0))
        fork = math.sin(angle * branches + math.sin(angle * 3.0 + radius * 31.0) * (1.0 + distortion * 5.0))
        secondary = math.sin(angle * branches * 2.07 - radius * 44.0 + a * 4.0)
        return clamp01(1.0 - min(abs(fork), abs(secondary) * 1.2) * (5.0 + radius * 3.0))
    elif t == 95:
        symmetry = 5 + int(_m32(complexity, 8.0))
        total = 0.0
        for i in range(symmetry):
            direction = 2.0 * PI * float(i) / float(symmetry)
            total += math.cos((u * math.cos(direction) + v * math.sin(direction)) * (48.0 + distortion * 70.0))
        total /= float(symmetry)
        return clamp01(1.0 - abs(total) * 3.2)
    elif t == 96:
        magnification = 2.0 ** (clamp01((s - 0.001) / 0.039) * 8.0)
        focus = smoothstep(1.6, 8.0, magnification)
        center_x = -0.5 + (-0.7436439 + 0.5) * focus + (distortion - 0.5) * 0.16 / magnification
        center_y = 0.1318259 * focus + (distortion - 0.5) * 0.1 / magnification
        cx = u / spatial_scale * 3.1 / magnification + center_x
        cy = v / spatial_scale * 3.1 / magnification + center_y
        zx = 0.0
        zy = 0.0
        iteration = 0
        max_iterations = 42 + int(_m32(complexity, 104.0))
        while iteration < max_iterations and zx * zx + zy * zy < 16.0:
            next_x = zx * zx - zy * zy + cx
            zy = 2.0 * zx * zy + cy
            zx = next_x
            iteration += 1
        return float(iteration) / float(max_iterations)
    elif t == 97:
        magnification = 2.0 ** (clamp01((s - 0.001) / 0.039) * 7.5)
        focus = smoothstep(1.7, 7.0, magnification)
        center_x = -0.5 + (-1.7443359 + 0.5) * focus + (distortion - 0.5) * 0.18 / magnification
        center_y = -0.48 + (-0.017451 + 0.48) * focus
        cx = u / spatial_scale * 3.2 / magnification + center_x
        cy = v / spatial_scale * 3.2 / magnification + center_y
        zx = 0.0
        zy = 0.0
        iteration = 0
        max_iterations = 42 + int(_m32(complexity, 100.0))
        while iteration < max_iterations and zx * zx + zy * zy < 16.0:
            zx = abs(zx)
            zy = abs(zy)
            next_x = zx * zx - zy * zy + cx
            zy = 2.0 * zx * zy + cy
            zx = next_x
            iteration += 1
        return float(iteration) / float(max_iterations)
    elif t == 98:
        magnification = 2.0 ** (clamp01((s - 0.001) / 0.039) * 7.0)
        focus = smoothstep(1.6, 6.0, magnification)
        zx = u / spatial_scale * 2.7 / magnification - 0.11 * focus
        zy = v / spatial_scale * 2.7 / magnification + 0.64 * focus
        phase = distortion * 2.0 * PI
        cx = -0.73 + math.cos(phase) * 0.12
        cy = math.sin(phase) * 0.21
        iteration = 0
        max_iterations = 42 + int(_m32(complexity, 104.0))
        while iteration < max_iterations and zx * zx + zy * zy < 16.0:
            next_x = zx * zx - zy * zy + cx
            zy = 2.0 * zx * zy + cy
            zx = next_x
            iteration += 1
        return float(iteration) / float(max_iterations)
    elif t == 99:
        segments = 4.0 + math.floor(_m32(complexity, 12.0))
        sector = 2.0 * PI / segments
        folded = abs(fract((angle + PI) / sector) - 0.5) * sector
        crystal = math.sin(radius * (70.0 + distortion * 90.0) + folded * 44.0 + math.sin(folded * segments) * 3.0)
        return clamp01(1.0 - abs(crystal) * 2.9)
    elif t == 100:
        facet = abs(fract((angle / (2.0 * PI) + 0.5) * (4.0 + math.floor(_m32(complexity, 10.0)))) - 0.5)
        refraction = fract(radius * (5.0 + distortion * 11.0) + facet * 3.0 + angle / (2.0 * PI))
        return clamp01(refraction * 0.75 + (1.0 - facet * 2.0) * 0.25)
    elif t == 101:
        turns = 2.0 + math.floor(_m32(complexity, 6.0))
        twist = 2.0 + math.floor(_m32(distortion, 7.0))
        knot_radius = 0.27 + 0.09 * math.cos(angle * turns)
        tube = abs(radius - knot_radius - 0.045 * math.sin(angle * twist))
        crossing = 0.5 + 0.5 * math.sin(angle * turns * twist)
        return clamp01(1.0 - tube * (20.0 + crossing * 18.0))
    elif t == 102:
        grid_x = float(np.float32(u) / np.float32(0.62) + np.float32(0.5))
        grid_y = float(np.float32(v) / np.float32(0.94) + np.float32(0.5))
        plant_x = int(math.floor(grid_x))
        plant_y = int(math.floor(grid_y))
        local_x = fract(grid_x) - 0.5
        local_y = fract(grid_y) - 0.5
        plant_seed = hash2(float(plant_x), float(plant_y), seed)
        stem_path = math.sin(local_y * (5.0 + plant_seed * 4.0) + plant_seed * 6.0) * distortion * 0.045
        stem = ((1.0 - smoothstep(0.007, 0.022, abs(local_x - stem_path)))
                * smoothstep(0.49, 0.42, abs(local_y)))
        leaves = 0.0
        leaf_count = 5 + int(_m32(complexity, 6.0))
        for i in range(leaf_count):
            leaf_y = -0.36 + float(i) * 0.72 / max(leaf_count - 1, 1)
            side = -1.0 if ((i + plant_x + plant_y) & 1) == 0 else 1.0
            leaf_seed = hash2(float(plant_x * 17 + i), float(plant_y), seed + 43)
            leaf_x = stem_path + side * (0.13 + leaf_seed * 0.08)
            rotation = side * (0.28 + leaf_seed * 0.34)
            dx = local_x - leaf_x
            dy = local_y - leaf_y
            rx = dx * math.cos(rotation) + dy * math.sin(rotation)
            ry = -dx * math.sin(rotation) + dy * math.cos(rotation)
            leaf_distance = math.sqrt(rx * rx / 0.038 + ry * ry / 0.0042)
            leaf = 1.0 - smoothstep(0.72, 1.0, leaf_distance)
            main_vein = 1.0 - smoothstep(0.004, 0.018, abs(ry))
            side_veins = 1.0 - smoothstep(0.0, 0.16, abs(math.sin(rx * (48.0 + complexity * 35.0)) - ry * 18.0))
            leaf_structure = leaf * (0.22 + main_vein * 0.52 + side_veins * 0.32)
            leaves = max(leaves, leaf_structure)
        return clamp01(stem + leaves)
    elif t == 103:
        spots = (math.sin(u * (52.0 + complexity * 68.0) + math.sin(v * 17.0) * 3.0)
                 + math.sin(v * (46.0 + complexity * 74.0) + math.cos(u * 19.0) * 3.0))
        pulse = 0.5 + 0.5 * math.sin(spots * 2.4 + distortion * 2.0 * PI + a * 3.0)
        return clamp01(pulse)
    elif t == 104:
        ribs = abs(math.sin((u + math.sin(v * 9.0) * distortion * 0.08) * (48.0 + complexity * 90.0)))
        tendons = abs(math.sin((v - u * 0.28) * (22.0 + complexity * 36.0) + a * 5.0))
        return clamp01(1.0 - min(ribs, tendons * 1.4) * 2.7)
    elif t == 105:
        curls = math.sin(angle * (4.0 + complexity * 11.0) + radius * (34.0 + distortion * 55.0))
        scrolls = math.sin((u * u - v * v) * 95.0 + angle * 6.0)
        return clamp01(1.0 - min(abs(curls), abs(scrolls)) * 4.2)
    elif t == 106:
        grid = _lin32(5.0, complexity, 9.0)
        sx = _shift_mul32(u, grid)
        sy = _shift_mul32(v, grid)
        gx = fract(sx) - 0.5
        gy = fract(sy) - 0.5
        cell_x = int(math.floor(sx))
        cell_y = int(math.floor(sy))
        glyph = hash2(float(cell_x), float(cell_y), seed)
        diagonal = abs(gx - gy) if glyph < 0.5 else abs(gx + gy)
        vertical = abs(gx + (glyph - 0.5) * 0.32)
        crossbar = abs(gy - (glyph - 0.5) * 0.26)
        stroke = min(diagonal, min(vertical, crossbar))
        cell_mask = smoothstep(0.48, 0.36, max(abs(gx), abs(gy)))
        return clamp01((1.0 - smoothstep(0.018, 0.065, stroke)) * cell_mask)
    elif t == 107:
        convection = math.sin((a - 0.5) * (18.0 + complexity * 34.0) + b * 7.0)
        granules = 1.0 - abs(convection)
        return clamp01(granules * 0.78 + a * 0.3)
    elif t == 108:
        spacing = _inv_lin32(0.18, complexity, 0.14)
        base_x = int(math.floor(_div32(u, spacing)))
        base_y = int(math.floor(_div32(v, spacing)))
        crater_field = 0.0
        rays = 0.0
        depressions = 0.0
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                crater_x = base_x + ox
                crater_y = base_y + oy
                crater_seed = hash2(float(crater_x), float(crater_y), seed)
                if crater_seed < 0.22:
                    continue
                cx = ((crater_x + 0.5) * spacing
                      + (hash2(float(crater_x), float(crater_y), seed + 19) - 0.5) * spacing * 0.52)
                cy = ((crater_y + 0.5) * spacing
                      + (hash2(float(crater_y), float(crater_x), seed + 37) - 0.5) * spacing * 0.52)
                crater_radius = spacing * (0.12 + crater_seed * 0.2)
                dx = u - cx
                dy = v - cy
                distance = math.sqrt(dx * dx + dy * dy)
                rim_width = crater_radius * (0.12 + distortion * 0.12)
                rim = 1.0 - smoothstep(rim_width * 0.35, rim_width, abs(distance - crater_radius))
                depression = (1.0 - smoothstep(0.0, crater_radius, distance)) * (0.4 + crater_seed * 0.28)
                ray_pattern = 1.0 - smoothstep(0.0, 0.13, abs(math.sin(math.atan2(dy, dx)
                                                                      * (4.0 + math.floor(_m32(crater_seed, 7.0))))))
                ray_fade = smoothstep(crater_radius * (5.0 + distortion * 4.0), crater_radius * 1.2, distance)
                crater_field = max(crater_field, rim)
                depressions = max(depressions, depression)
                rays = max(rays, ray_pattern * ray_fade * 0.5)
        return clamp01(0.24 + (a - 0.5) * 0.22 - depressions * 0.24 + crater_field * 0.68 + rays * 0.42)
    elif t == 109:
        spacing = _inv_lin32(0.34, complexity, 0.18)
        base_x = int(math.floor(_div32(u, spacing)))
        base_y = int(math.floor(_div32(v, spacing)))
        velocity_phase = 0.0
        total_weight = 0.0
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                vortex_x = base_x + ox
                vortex_y = base_y + oy
                vortex_seed = hash2(float(vortex_x), float(vortex_y), seed)
                cx = (vortex_x + 0.5) * spacing + (vortex_seed - 0.5) * spacing * 0.48
                cy = ((vortex_y + 0.5) * spacing
                      + (hash2(float(vortex_y), float(vortex_x), seed + 53) - 0.5) * spacing * 0.48)
                dx = u - cx
                dy = v - cy
                local_radius = math.sqrt(dx * dx + dy * dy)
                weight = math.exp(-local_radius * local_radius / (spacing * spacing * 0.34))
                spin = -1.0 if vortex_seed < 0.5 else 1.0
                velocity_phase += math.sin(math.atan2(dy, dx) * spin * (1.0 + math.floor(_m32(vortex_seed, 3.0)))
                                           + local_radius / spacing * (13.0 + distortion * 18.0)) * weight
                total_weight += weight
        jet = math.sin((v + math.sin(u * 3.2 + a * 2.0) * distortion * 0.12) * (18.0 + complexity * 34.0))
        current = velocity_phase / max(total_weight, 0.001) * 0.72 + jet * 0.28
        return clamp01(1.0 - abs(current) * 2.35)
    elif t == 110:
        mountains = 0.0
        for layer in range(6):
            depth = float(layer) / 5.0
            ridge = (-0.34 + depth * 0.13
                     + math.sin(u * (7.0 + layer * 2.3) + layer * 1.7) * (0.06 + depth * 0.025)
                     + fbm(seed, (u + layer * 0.37) * (3.0 + complexity * 4.0), layer * 2.1) * 0.13)
            silhouette = smoothstep(ridge - 0.018, ridge + 0.018 + distortion * 0.025, v)
            mountains = max(mountains, silhouette * (1.0 - depth * 0.13))
        return clamp01(mountains)
    elif t == 111:
        # the window-column lattice (floored) is evaluated in float32 like upstream
        f = np.float32
        horizon32 = f(0.08) + np.sin(f(u) * f(13.0)) * f(0.025)
        perspective32 = _clamp32((f(v) - horizon32) / f(0.45), f(0.0), f(1.0))
        columns32 = ((f(u) + f(0.5)) * (f(12.0) + f(complexity) * f(24.0))
                     / max(f(0.28), f(1.0) - perspective32 * f(0.72)))
        columns32 = columns32 - np.floor(columns32)
        horizon = float(horizon32)
        buildings = smoothstep(horizon - 0.32 * hash2(float(np.floor(columns32 * f(17.0))),
                                                      float(np.floor(f(u) * f(31.0))), seed),
                               horizon, v)
        windows = math.pow(clamp01(0.5 + 0.5 * math.sin(u * 180.0) * math.sin(v * 145.0)), 7.0)
        road = smoothstep(horizon, 0.48, v) * (1.0 - smoothstep(0.02, 0.11, abs(u)))
        return clamp01(buildings * 0.25 + windows * buildings + road * (0.55 + distortion * 0.4))
    elif t == 112:
        return _phyllotaxis32(u, v, complexity, distortion)
    elif t == 113:
        depth = 4 + int(_m32(complexity, 7.0))
        size = 1 << depth
        rotation = (distortion - 0.5) * PI * 0.5
        rx = u * math.cos(rotation) - v * math.sin(rotation)
        ry = u * math.sin(rotation) + v * math.cos(rotation)
        ix = int((rx + 0.5) * size)
        iy = int((ry + 0.5) * size)
        ix = 0 if ix < 0 else (size - 1 if ix > size - 1 else ix)
        iy = 0 if iy < 0 else (size - 1 if iy > size - 1 else iy)
        triangle = 1.0 if (ix & iy) == 0 else 0.0
        triangle *= smoothstep(0.52, 0.45, max(abs(rx), abs(ry)))
        return triangle
    elif t == 114:
        rings = 0.0
        levels = 3 + int(_m32(complexity, 6.0))
        for level in range(levels):
            frequency = 2.0 ** float(level)
            qx = fract((u + 0.5) * frequency) - 0.5
            qy = fract((v + 0.5) * frequency) - 0.5
            local_radius = math.sqrt(qx * qx + qy * qy)
            target = 0.28 + math.sin(level * 1.7) * distortion * 0.035
            ring = 1.0 - smoothstep(0.008, 0.032, abs(local_radius - target))
            rings = max(rings, ring * (1.0 - level * 0.06))
        return clamp01(rings)
    elif t == 115:
        cell_size = 0.42
        row_step = cell_size * _SQRT3_F * 0.5
        base_row = int(math.floor(v / row_step))
        best_distance_squared = 1000.0
        local_x = 0.0
        local_y = 0.0
        selected_column = 0
        selected_row = 0
        for row in range(base_row - 1, base_row + 2):
            row_offset = cell_size * 0.5 if row % 2 != 0 else 0.0
            base_column = int(math.floor((u - row_offset) / cell_size))
            for column in range(base_column - 1, base_column + 2):
                center_x = (float(column) + 0.5) * cell_size + row_offset
                center_y = (float(row) + 0.5) * row_step
                dx = u - center_x
                dy = v - center_y
                distance_squared = dx * dx + dy * dy
                if distance_squared < best_distance_squared:
                    best_distance_squared = distance_squared
                    local_x = dx / cell_size
                    local_y = dy / cell_size
                    selected_column = column
                    selected_row = row
        hex_radius = max(abs(local_x) * 2.0, abs(local_x) + abs(local_y) * _SQRT3_F)
        hex_radius = min(hex_radius, 0.992)
        hyperbolic_radius = math.log((1.0 + hex_radius) / max(1.0 - hex_radius, 0.008))
        local_angle = math.atan2(local_y, local_x)
        sides = 4.0 + math.floor(_m32(complexity, 9.0))
        parity = 1.0 if (selected_column + selected_row) % 2 != 0 else 0.0
        orientation = parity * PI / sides + float(seed) * 0.0017
        curved_angle = local_angle + orientation + math.sin(hyperbolic_radius * 0.72) * distortion * 0.16
        radial_family = abs(math.sin(hyperbolic_radius * (2.15 + complexity * 1.35)
                                     + math.cos(curved_angle * sides) * (0.85 + distortion * 1.25)))
        diagonal_family = abs(math.sin(curved_angle * sides + hyperbolic_radius * (0.62 + distortion * 1.45)))
        counter_family = abs(math.sin(curved_angle * (sides - 2.0) - hyperbolic_radius * (1.18 + distortion * 0.82)))
        geodesic_distance = min(radial_family, min(diagonal_family, counter_family))
        geodesics = 1.0 - smoothstep(0.025, 0.13, geodesic_distance)
        boundary = 1.0 - smoothstep(0.018, 0.075, 1.0 - hex_radius)
        chambers = 0.5 + 0.5 * math.cos(hyperbolic_radius * 1.32 - curved_angle * (sides * 0.5) + parity * PI)
        return clamp01(geodesics * 0.78 + boundary * 0.58 + chambers * 0.14)
    elif t == 116:
        bands = 3.0 + math.floor(_m32(complexity, 9.0))
        twist = angle * (1.0 + distortion * 4.0) + radius * 24.0
        ribbon = math.sin(twist) * math.cos(angle * bands - radius * 8.0)
        front_back = 0.5 + 0.5 * math.sin(angle + radius * PI * 2.0)
        return clamp01((1.0 - abs(ribbon) * 2.9) * (0.55 + front_back * 0.45))
    elif t == 117:
        mirrored_x = abs(u)
        ink = fbm(seed, (mirrored_x + math.sin(v * 7.0) * distortion * 0.08) * (3.0 + complexity * 5.0),
                  v * (3.0 + complexity * 5.0))
        lobes = 0.5 + 0.5 * math.sin(v * (8.0 + complexity * 18.0) + mirrored_x * 11.0)
        silhouette = smoothstep(0.58 - distortion * 0.22, 0.76, ink * 0.72 + lobes * 0.28)
        return clamp01(silhouette * smoothstep(0.48, 0.03, mirrored_x))
    elif t == 118:
        waves = 0.0
        sources = 2 + int(_m32(complexity, 5.0))
        for i in range(sources):
            source_x = (float(i) / max(sources - 1, 1) - 0.5) * 0.72
            source_y = (hash2(float(i), 6.4, seed) - 0.5) * 0.42
            distance = math.sqrt((u - source_x) * (u - source_x) + (v - source_y) * (v - source_y))
            waves += math.sin(distance * (58.0 + distortion * 72.0) + i * 1.3)
        return clamp01(0.5 + waves / float(sources) * 0.5)
    elif t == 119:
        frequency = 24.0 + complexity * 52.0
        convection = (math.sin(u * frequency + math.sin(v * 9.0) * distortion * 5.0)
                      + math.sin((u * 0.5 + v * 0.866) * frequency)
                      + math.sin((-u * 0.5 + v * 0.866) * frequency))
        plumes = 0.5 + 0.5 * math.sin(v * 18.0 + a * 5.0)
        return clamp01(convection / 6.0 + 0.5 + plumes * 0.2)
    elif t == 120:
        fold_a = abs(fract((u + v * 0.62) * (5.0 + complexity * 13.0)) - 0.5)
        fold_b = abs(fract((v - u * 0.38) * (6.0 + complexity * 11.0) + distortion * 0.3) - 0.5)
        crease = min(fold_a, fold_b)
        face = fract(fold_a * 2.0 + fold_b * 3.0)
        return clamp01(face * 0.58 + (1.0 - smoothstep(0.0, 0.055, crease)) * 0.62)
    elif t == 121:
        grid = 8.0 + complexity * 19.0
        gx = fract((u + 0.5 + math.sin(v * 8.0) * distortion * 0.04) * grid) - 0.5
        gy = fract((v + 0.5 + math.cos(u * 7.0) * distortion * 0.04) * grid) - 0.5
        core = math.sqrt(gx * gx + gy * gy)
        ring = 1.0 - smoothstep(0.07, 0.18, core)
        cladding = 1.0 - smoothstep(0.16, 0.28, core)
        return clamp01(ring + cladding * 0.28)
    return a


@njit(cache=True, inline="always")
def _edge_family(t):
    return (t == 74 or t == 75 or t == 79 or t == 86 or t == 87 or t == 92 or t == 93 or t == 94 or t == 95
            or t == 99 or t == 101 or t == 102 or t == 104 or t == 105 or t == 106 or t == 109 or t == 112
            or t == 113 or t == 114 or t == 115 or t == 116 or t == 118 or t == 121)


@njit(cache=True)
def _shade(t, field, x, y, width, height, p, c):
    xl = x - 1 if x > 0 else 0
    xr = x + 1 if x < width - 1 else width - 1
    yt = y - 1 if y > 0 else 0
    yb = y + 1 if y < height - 1 else height - 1
    center = float(field[y, x])
    left = float(field[y, xl])
    right = float(field[y, xr])
    top = float(field[yt, x])
    bottom = float(field[yb, x])
    gx = right - left
    gy = bottom - top
    gradient = clamp01(math.sqrt(gx * gx + gy * gy) * 7.0)
    laplacian = clamp01(abs(left + right + top + bottom - center * 4.0) * 5.0)
    sharpness = clamp01(parameter(p, 3, 0.58))
    glow = clamp01(parameter(p, 4, 0.52))
    contrast_amount = max(parameter(p, 5, 1.25), 0.05)
    edge = smoothstep(0.02 + (1.0 - sharpness) * 0.08, 0.45, gradient + laplacian * 0.7)
    value = contrast(center * 0.72 + edge * 0.42, contrast_amount)

    if _edge_family(t):
        value = contrast(edge * 0.8 + center * 0.42, contrast_amount)
    elif t == 81 or t == 88 or t == 89:
        value = contrast(center * 0.62 + (1.0 - edge) * 0.28, contrast_amount)
    elif t == 84 or t == 91:
        value = fract(center + gradient * 0.8 + laplacian * 0.45)

    if t == 71:
        value = contrast(smoothstep(0.38, 0.62, center) * 0.72 + edge * 0.35, contrast_amount)
    elif t == 72:
        value = contrast(center * 0.74 + edge * 0.38 + laplacian * 0.16, contrast_amount)
    elif t == 73:
        value = contrast(edge * 0.62 + center * 0.28, contrast_amount)
    elif t == 76:
        value = clamp01(center * 0.48 + edge * 0.72 + laplacian * 0.38)
    elif t == 78:
        value = contrast(center * 0.72 + gradient * 0.5, contrast_amount)
    elif t == 74:
        value = contrast(center * 0.48 + edge * 0.76 + laplacian * 0.22, contrast_amount)
    elif t == 75:
        value = contrast(center * 0.72 + edge * 0.62, contrast_amount)
    elif t == 79:
        value = contrast(center * 0.58 + edge * 0.68 + laplacian * 0.18, contrast_amount)
    elif t == 81:
        value = contrast(center * 0.82 + edge * 0.26, contrast_amount)
    elif t == 83:
        value = contrast(center * 0.78 + edge * 0.28, contrast_amount)
    elif t == 86:
        value = contrast(center * 0.54 + edge * 0.72 + laplacian * 0.18, contrast_amount)
    elif t == 89:
        value = contrast(center * 0.22 + gradient * 0.86 + clamp01((-gx - gy) * 2.0 + 0.5) * 0.18, contrast_amount)
    elif t == 96 or t == 97 or t == 98:
        value = 0.0 if center > 0.995 else fract(center * 7.0 + edge * 0.42)
    elif t == 100:
        value = fract(center + gradient * 0.7)
    elif t == 102:
        value = contrast(center * 0.42 + edge * 0.76 + laplacian * 0.22, contrast_amount)
    elif t == 107 or t == 119:
        value = contrast(center * 0.76 + laplacian * 0.48, contrast_amount)
    elif t == 110:
        value = contrast(center * 0.82 + edge * 0.16, contrast_amount)
    elif t == 109:
        value = contrast(center * 0.36 + edge * 0.74 + fract(center * 3.0) * 0.18, contrast_amount)
    elif t == 115:
        value = contrast(center * 0.54 + edge * 0.68 + laplacian * 0.2, contrast_amount)
    elif t == 117:
        value = contrast(center * 0.9 + gradient * 0.16, contrast_amount)
    elif t == 120:
        value = contrast(center * 0.55 + (-gx - gy + 1.0) * 0.22 + edge * 0.24, contrast_amount)

    result = palette4(c, value)
    light = clamp01((-gx - gy) * 2.8 + gradient * 0.35)
    result = mix_color(result, color(c, 3, 0xffffe8a3), light * (0.2 + glow * 0.55))
    return mix_color(result, color(c, 3, 0xffffe8a3), edge * glow * 0.32)


@njit(cache=True, parallel=True)
def _field_pass(t, width, height, seed, p, field):
    for yy in prange(height):
        fy = float(yy)
        for xx in range(width):
            field[yy, xx] = _field(t, seed, float(xx), fy, width, height, p)


@njit(cache=True, parallel=True)
def _shade_pass(t, field, width, height, p, c, out):
    for yy in prange(height):
        for xx in range(width):
            out[yy, xx] = _shade(t, field, xx, yy, width, height, p, c)


def render_simulation(texture_type: int, width: int, height: int, seed: int, params, colors) -> np.ndarray:
    """Native types 71..121 -> uint32 (H, W) ARGB."""
    if not 71 <= texture_type <= 121:
        raise ValueError(f"render_simulation handles texture types 71..121, got {texture_type}")
    w, h, t = int(width), int(height), int(texture_type)
    p = np.ascontiguousarray(params, np.float64)
    c = np.ascontiguousarray(colors, np.uint32)
    field = np.empty((h, w), np.float32)
    _field_pass(t, w, h, int(seed), p, field)
    out = np.empty((h, w), np.uint32)
    _shade_pass(t, field, w, h, p, c, out)
    return out
