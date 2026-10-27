"""Fast-noise "showcase" textures (native types 51..70) - numba port.

Line-by-line port of ``showcaseTexture()`` in ImageToolboxLibs
``fast-noise/src/main/cpp/textures/ProceduralTextures.cpp`` (Apache-2.0, T8RIN):
Geode, Prismatic Light, Stained Glass, Kelp Forest, Frost Fern, Liquid Crystal,
Dragon Scales, Firefly Swarm, Mycelium, Kintsugi, Carbon Fiber, Circuit Board,
Soap Film, Moire Guilloche, Snake Skin, Terrazzo, Galaxy Filaments, Volcanic
Obsidian, Motherboard Heatmap, Microscopic Diatoms.

Arithmetic runs in float64 (upstream float32) except ``hash2`` which emulates the
float32 ``textures::hash``; colours are 8-bit ARGB ints mixed with ``mix_color``
exactly like the C++ ``mixColor``.
"""
from __future__ import annotations

import math

import numpy as np

from ._nb import njit, prange
from ._noisebank import cell_edge, cell_variation, fbm, fine, secondary_cell_edge, warp
from ._texmath import PI, clamp01, color, contrast, fract, hash2, mix_color, palette4, parameter, smoothstep


@njit(cache=True)
def _showcase_pixel(t, seed, x, y, width, height, p, c):
    s = max(parameter(p, 0, 0.009), 0.00001)
    complexity = clamp01(parameter(p, 1, 0.62))
    distortion = clamp01(parameter(p, 2, 0.48))
    sharpness = clamp01(parameter(p, 3, 0.58))
    glow = clamp01(parameter(p, 4, 0.52))
    contrast_amount = max(parameter(p, 5, 1.25), 0.05)
    # u, v evaluated in float32 like upstream (they feed fract()/floor() lattices)
    unit32 = np.float32(min(width, height))
    ss32 = np.float32(s) / np.float32(0.008)
    if ss32 < np.float32(0.125):
        ss32 = np.float32(0.125)
    elif ss32 > np.float32(5.0):
        ss32 = np.float32(5.0)
    u = float((np.float32(x) - np.float32(width) * np.float32(0.5)) / unit32 * ss32)
    v = float((np.float32(y) - np.float32(height) * np.float32(0.5)) / unit32 * ss32)
    nx = x
    ny = y
    base = 0.0
    fine_v = 0.0
    # types 52, 64 and 70 never read the warped coordinates / base / fine values (no side effects upstream)
    if t != 52 and t != 64 and t != 70:
        nx, ny = warp(seed, x, y, s * 0.7, distortion * 54.0)
        base = fbm(seed, nx * s, ny * s)
        fine_v = fine(seed, nx * s * 4.1, ny * s * 4.1)
    result = base
    accent = 0.0

    if t == 51:
        radius = math.sqrt(u * u + v * v) + (base - 0.5) * distortion * 0.16
        bands = 0.5 + 0.5 * math.sin(radius * (55.0 + complexity * 95.0) + fine_v * 8.0)
        cavity = 1.0 - smoothstep(0.05, 0.2, radius)
        crystals = math.pow(clamp01(0.5 + 0.5 * math.sin(math.atan2(v, u) * (10.0 + complexity * 24.0)
                                                         + radius * 180.0)), 5.0)
        result = contrast(bands * (1.0 - cavity * 0.55) + cavity * crystals, contrast_amount)
        accent = cavity * crystals * glow
    elif t == 52:
        angle = math.atan2(v, u)
        radius = math.sqrt(u * u + v * v)
        facets = abs(fract((angle / (2.0 * PI) + 0.5) * (5.0 + complexity * 13.0)) - 0.5)
        beam = abs(math.sin((u * 0.84 + v * 0.54) * (42.0 + complexity * 86.0) + facets * 9.0))
        spectrum = fract(angle / (2.0 * PI) + radius * (2.0 + distortion * 5.0) + facets)
        accent = (1.0 - smoothstep(0.01, 0.08 + (1.0 - sharpness) * 0.12, beam)) * glow
        return mix_color(palette4(c, spectrum), color(c, 3, 0xffffe47a), accent)
    elif t == 53:
        edge = cell_edge(seed, nx, ny, s * (1.3 + complexity * 3.2), 1.0)
        lead = 1.0 - smoothstep(0.035, 0.095 + (1.0 - sharpness) * 0.08, edge)
        glass = cell_variation(seed, nx, ny, s * (1.3 + complexity * 3.2), 1.0)
        pane = palette4(c, fract(glass * 2.8 + fine_v * 0.22))
        pane = mix_color(pane, color(c, 3, 0xffffd34e), fine_v * glow * 0.3)
        return mix_color(pane, color(c, 0, 0xff120f18), lead)
    elif t == 54:
        depth = clamp01(v + 0.5)
        stems = abs(math.sin((u + math.sin(v * 8.0 + base * 3.0) * distortion * 0.12) * (24.0 + complexity * 46.0)))
        blades = 1.0 - smoothstep(0.015, 0.07 + (1.0 - sharpness) * 0.12, stems)
        rays = math.pow(clamp01(0.5 + 0.5 * math.sin((u + v * 0.24) * 18.0 + fine_v * 2.0)), 7.0)
        result = clamp01(base * 0.28 + blades * (0.45 + depth * 0.35) + rays * (1.0 - depth) * glow)
        accent = rays * (1.0 - depth) * glow
    elif t == 55:
        angle = math.atan2(v, u)
        radius = math.sqrt(u * u + v * v)
        branches = abs(math.sin(angle * (5.0 + complexity * 13.0) + radius * 68.0 + base * distortion * 7.0))
        needles = abs(math.sin((u * 0.7 - v) * (170.0 + complexity * 260.0)))
        accent = (1.0 - smoothstep(0.02, 0.14, branches)) * (0.55 + 0.45 * (1.0 - needles))
        result = clamp01(base * 0.45 + accent * (0.7 + glow * 0.35))
    elif t == 56:
        angle = math.atan2(v, u)
        phase = base * 14.0 + angle * (4.0 + complexity * 10.0) + (u - v) * 28.0
        result = fract(phase / (2.0 * PI))
        accent = math.pow(0.5 + 0.5 * math.sin(phase * 1.73), 5.0) * glow
    elif t == 57:
        cells_x = float(np.float32(x) * np.float32(s) * (np.float32(7.0) + np.float32(complexity) * np.float32(12.0)))
        cells_y = float(np.float32(y) * np.float32(s) * (np.float32(5.0) + np.float32(complexity) * np.float32(9.0)))
        row = math.floor(cells_y)
        px = fract(cells_x + (int(row) & 1) * 0.5) - 0.5
        py = fract(cells_y) - 0.32
        scale_shape = math.sqrt(px * px + py * py * 0.65)
        rim = 1.0 - smoothstep(0.32, 0.42 + (1.0 - sharpness) * 0.09, scale_shape)
        result = clamp01(rim * (0.34 + 0.66 * (0.5 - py)) + fine_v * 0.2)
        accent = rim * smoothstep(0.19, -0.1, py) * glow
    elif t == 58:
        density32 = np.float32(7.0) + np.float32(complexity) * np.float32(15.0)
        gx = float((np.float32(u) + np.float32(0.5)) * density32)
        gy = float((np.float32(v) + np.float32(0.5)) * density32)
        nearest = 10.0
        trail = 10.0
        cell_x = int(math.floor(gx))
        cell_y = int(math.floor(gy))
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                cx = cell_x + ox
                cy = cell_y + oy
                px = cx + hash2(float(cx), float(cy), seed) * 0.76 + 0.12
                py = cy + hash2(float(cy), float(cx), seed + 31) * 0.76 + 0.12
                dx = gx - px
                dy = gy - py
                nearest = min(nearest, math.sqrt(dx * dx + dy * dy))
                tx = dx + dy * distortion * 0.8
                trail = min(trail, math.sqrt(tx * tx + dy * dy * 0.08))
        spark = 1.0 - smoothstep(0.015, 0.10 + (1.0 - sharpness) * 0.12, nearest)
        tail = 1.0 - smoothstep(0.03, 0.18, trail)
        result = clamp01(base * 0.18 + tail * 0.34 + spark)
        accent = spark * glow
    elif t == 59:
        edge1 = cell_edge(seed, nx, ny, s * (2.0 + complexity * 3.0), 1.2)
        edge2 = secondary_cell_edge(seed, nx + 37.0, ny - 19.0, s * (4.0 + complexity * 5.0), 1.4)
        accent = (1.0 - smoothstep(0.018, 0.11, edge1)) * 0.7 + (1.0 - smoothstep(0.012, 0.08, edge2)) * 0.5
        result = clamp01(base * 0.3 + accent * (0.65 + glow * 0.35))
    elif t == 60:
        edge = cell_edge(seed, nx, ny, s * (1.3 + complexity * 3.7), 1.3)
        crack = 1.0 - smoothstep(0.012, 0.065 + (1.0 - sharpness) * 0.07, edge)
        ceramic = mix_color(color(c, 0, 0xff10131a), color(c, 1, 0xff344158), base * 0.7)
        gold = mix_color(color(c, 2, 0xffd99b32), color(c, 3, 0xffffe7a3), fine_v * glow)
        return mix_color(ceramic, gold, crack)
    elif t == 61:
        frequency = 0.08 + s * 8.0 + complexity * 0.11
        ax = math.sin((x + y * 0.48) * frequency * PI)
        ay = math.sin((y - x * 0.48) * frequency * PI)
        weave = ax * ay
        result = clamp01(0.38 + weave * 0.3 + fine_v * 0.22)
        accent = math.pow(clamp01(weave), 4.0) * glow
    elif t == 62:
        grid32 = np.float32(12.0) + np.float32(complexity) * np.float32(32.0)
        gx = fract(float(np.float32(x) * np.float32(s) * grid32))
        gy = fract(float(np.float32(y) * np.float32(s) * grid32))
        trace = min(abs(gx - 0.5), abs(gy - 0.5))
        path = 1.0 - smoothstep(0.025, 0.08 + (1.0 - sharpness) * 0.06, trace)
        via = 1.0 - smoothstep(0.08, 0.17, math.sqrt((gx - 0.5) * (gx - 0.5) + (gy - 0.5) * (gy - 0.5)))
        result = clamp01(base * 0.3 + path * 0.48 + via * 0.5)
        accent = via * glow
    elif t == 63:
        film = base * (8.0 + complexity * 18.0) + fine_v * distortion * 5.0 + (u * u - v * v) * 18.0
        result = fract(film)
        accent = math.pow(0.5 + 0.5 * math.sin(film * PI * 2.0), 6.0) * glow
    elif t == 64:
        r1 = math.sqrt((u - 0.18) * (u - 0.18) + v * v)
        r2 = math.sqrt((u + 0.18) * (u + 0.18) + v * v)
        pattern = math.sin(r1 * (120.0 + complexity * 260.0)) * math.sin(r2 * (124.0 + complexity * 252.0))
        result = contrast(pattern * 0.5 + 0.5, contrast_amount)
        accent = math.pow(result, 7.0) * glow
    elif t == 65:
        edge = cell_edge(seed, nx, ny, s * (2.5 + complexity * 4.0), 0.78)
        inside = cell_variation(seed, nx, ny, s * (2.5 + complexity * 4.0), 0.78)
        rim = 1.0 - smoothstep(0.025, 0.11, edge)
        result = clamp01(inside * 0.78 + (1.0 - rim) * fine_v * 0.2)
        accent = rim * glow
    elif t == 66:
        cells = cell_variation(seed, nx, ny, s * (4.0 + complexity * 8.0), 1.7)
        edge = cell_edge(seed, nx, ny, s * (4.0 + complexity * 8.0), 1.7)
        chips = smoothstep(0.68 - complexity * 0.22, 0.9, cells) * smoothstep(0.02, 0.13, edge)
        if chips > 0.05:
            return palette4(c, fract(cells * 3.7 + fine_v * 0.2))
        return mix_color(color(c, 0, 0xffe5dccf), color(c, 3, 0xffe5a73d), fine_v * 0.08)
    elif t == 67:
        e1 = 1.0 - smoothstep(0.015, 0.13, cell_edge(seed, nx, ny, s * (1.2 + complexity * 2.0), 1.4))
        e2 = 1.0 - smoothstep(0.01, 0.08, secondary_cell_edge(seed, nx, ny, s * (3.2 + complexity * 4.0), 1.5))
        stars = smoothstep(0.998 - complexity * 0.006, 1.0, hash2(x, y, seed))
        result = clamp01(e1 * 0.72 + e2 * 0.38 + stars)
        accent = stars + e1 * e2 * glow
    elif t == 68:
        facets = cell_variation(seed, nx, ny, s * (1.8 + complexity * 4.0), 1.2)
        edge = 1.0 - smoothstep(0.02, 0.12, cell_edge(seed, nx, ny, s * (1.8 + complexity * 4.0), 1.2))
        result = clamp01(facets * 0.25 + edge * 0.22 + fine_v * 0.08)
        accent = edge * smoothstep(0.82, 1.0, base) * glow
    elif t == 69:
        board_scale32 = np.float32(2.2) + np.float32(complexity) * np.float32(4.8)
        board_x = float(np.float32(x) * np.float32(s) * board_scale32)
        board_y = float(np.float32(y) * np.float32(s) * board_scale32)
        cell_x = int(math.floor(board_x))
        cell_y = int(math.floor(board_y))
        local_x = fract(board_x) - 0.5
        local_y = fract(board_y) - 0.5
        cell_kind = hash2(float(cell_x), float(cell_y), seed)
        horizontal = 1.0 - smoothstep(0.018, 0.055, abs(local_y - (cell_kind - 0.5) * 0.42))
        vertical = 1.0 - smoothstep(0.018, 0.055,
                                    abs(local_x - (hash2(float(cell_y), float(cell_x), seed + 17) - 0.5) * 0.42))
        if cell_kind < 0.5:
            route = max(horizontal, vertical * smoothstep(0.05, 0.42, abs(local_y)))
        else:
            route = max(vertical, horizontal * smoothstep(0.05, 0.42, abs(local_x)))
        h31 = hash2(float(cell_y), float(cell_x), seed + 31)
        vdx = local_x - (cell_kind - 0.5) * 0.38
        vdy = local_y - (h31 - 0.5) * 0.38
        via_distance = math.sqrt(vdx * vdx + vdy * vdy)
        via = 1.0 - smoothstep(0.055, 0.115, via_distance)
        if cell_kind > 0.76:
            chip = 1.0 - smoothstep(0.22, 0.29, max(abs(local_x), abs(local_y) * 1.45))
        else:
            chip = 0.0
        pins = chip * (1.0 - smoothstep(0.02, 0.06, abs(abs(local_y) - 0.2)))
        heat = 0.0
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                hot_x = cell_x + ox
                hot_y = cell_y + oy
                hot_kind = hash2(float(hot_x), float(hot_y), seed)
                if hot_kind <= 0.76:
                    continue
                dx = board_x - (hot_x + 0.5)
                dy = board_y - (hot_y + 0.5)
                heat = max(heat, math.exp(-(dx * dx + dy * dy) * (2.5 + distortion * 7.0)))
        board = mix_color(color(c, 0, 0xff07120e), color(c, 1, 0xff08705b), 0.2 + base * 0.32)
        board = mix_color(board, np.int64(0xff020504), chip * 0.8)
        board = mix_color(board, color(c, 2, 0xffff7b22), clamp01(route * 0.62 + via * 0.78 + pins))
        board = mix_color(board, color(c, 3, 0xfffff1a8), heat * glow * (0.35 + via * 0.65))
        return board
    elif t == 70:
        grid32 = np.float32(4.0) + np.float32(complexity) * np.float32(9.0)
        gx = fract(float((np.float32(u) + np.float32(0.5)) * grid32)) - 0.5
        gy = fract(float((np.float32(v) + np.float32(0.5)) * grid32)) - 0.5
        radius = math.sqrt(gx * gx + gy * gy)
        angle = math.atan2(gy, gx)
        shell = 1.0 - smoothstep(0.31, 0.42, radius)
        pores = 0.5 + 0.5 * math.sin(angle * (8.0 + complexity * 14.0) + radius * 90.0)
        result = clamp01(shell * (0.38 + pores * 0.62))
        accent = shell * math.pow(pores, 7.0) * glow

    result = contrast(result, contrast_amount)
    return mix_color(palette4(c, result), color(c, 3, 0xffffe8a3), clamp01(accent))


@njit(cache=True, parallel=True)
def _render(t, width, height, seed, p, c, out):
    for yy in prange(height):
        fy = float(yy)
        for xx in range(width):
            out[yy, xx] = _showcase_pixel(t, seed, float(xx), fy, width, height, p, c)


def render_showcase(texture_type: int, width: int, height: int, seed: int, params, colors) -> np.ndarray:
    """Native types 51..70 -> uint32 (H, W) ARGB."""
    if not 51 <= texture_type <= 70:
        raise ValueError(f"render_showcase handles texture types 51..70, got {texture_type}")
    out = np.empty((int(height), int(width)), np.uint32)
    _render(int(texture_type), int(width), int(height), int(seed), np.ascontiguousarray(params, np.float64),
            np.ascontiguousarray(colors, np.uint32), out)
    return out
