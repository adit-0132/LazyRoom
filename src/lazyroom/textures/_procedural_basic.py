"""numba port of ImageToolboxLibs ``fast-noise/src/main/cpp/textures/ProceduralTextures.cpp``
(Apache-2.0, T8RIN) — native texture types 0..50 (``lava`` .. ``ringedPlanet``,
dispatched by ``generatePixel``).

Every function is a line-by-line port of its C++ counterpart: same constants, same
order of operations, same ``std::`` semantics.  Arithmetic is float64 (upstream
float32) except ``hash2`` which emulates the float32 ``textures::hash``.  Colours are
ARGB ints carried as int64; ``mix_color`` rounds to 8 bits per channel like the C++.
The noise seed passed in is the texture seed (``NoiseBank(seed)``).
"""
from __future__ import annotations

import math

import numpy as np

from ._nb import njit, prange
from ._noisebank import cell_edge, cell_variation, fbm, fine, secondary_cell_edge, warp
from ._texmath import PI, clamp01, clampf, color, contrast, fract, hash2, mix_color, parameter, smoothstep



@njit(cache=True, inline="always")
def _cfmod(a, b):
    """C ``std::fmod`` (result has the sign of ``a``); exact for the integer-valued uses here."""
    return a - b * math.trunc(a / b)


# ---------------------------------------------------------------------------
# 0..11
# ---------------------------------------------------------------------------


@njit(cache=True)
def _lava(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.008), 0.00001)
    x, y = warp(seed, x, y, scale * 0.7, parameter(p, 1, 32.0))
    flow = max(parameter(p, 2, 1.35), 0.05)
    detail = clamp01(parameter(p, 3, 0.55))
    crust = clamp01(parameter(p, 4, 0.52))
    glow = clamp01(parameter(p, 5, 0.72))
    field = fbm(seed, x * scale, y * scale)
    field = field * (1.0 - detail * 0.35) + fine(seed, x * scale * 3.7, y * scale * 3.7) * detail * 0.35
    channels = clamp01(1.0 - abs(field * 2.0 - 1.0)) ** flow
    molten = smoothstep(crust - 0.18, crust + 0.18, channels)
    core = smoothstep(0.66, 0.98, channels) * glow
    result = mix_color(color(c, 0, 0xff120b0a), color(c, 1, 0xffd52b0a), molten)
    return mix_color(result, color(c, 2, 0xffffd45a), core)


@njit(cache=True)
def _clouds(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.0045), 0.00001)
    coverage = clamp01(parameter(p, 1, 0.48))
    softness = max(parameter(p, 2, 0.22), 0.005)
    detail = clamp01(parameter(p, 3, 0.62))
    x, y = warp(seed, x, y, scale * 0.65, parameter(p, 4, 24.0))
    density = max(parameter(p, 5, 0.92), 0.0)
    field = fbm(seed, x * scale, y * scale) * (1.0 - detail * 0.3) + \
        fine(seed, x * scale * 3.2, y * scale * 3.2) * detail * 0.3
    mask = smoothstep(coverage - softness, coverage + softness, field) * density
    light = clamp01(field + (fbm(seed, (x - 7.0) * scale, (y - 11.0) * scale) - field) * 1.8)
    cloud = mix_color(color(c, 1, 0xff9eadbc), color(c, 2, 0xfff8fbff), light)
    return mix_color(color(c, 0, 0xff6fa9dc), cloud, mask)


@njit(cache=True)
def _smoke(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.006), 0.00001)
    x, y = warp(seed, x, y, scale * 0.7, parameter(p, 1, 34.0))
    density = clamp01(parameter(p, 2, 0.56))
    wisps = max(parameter(p, 3, 2.2), 0.0)
    amount = max(parameter(p, 4, 1.35), 0.0)
    detail = clamp01(parameter(p, 5, 0.5))
    base = fbm(seed, x * scale, y * scale)
    curls = 0.5 + 0.5 * math.sin((x + y * 0.35) * scale * wisps * 4.0 + base * 8.0)
    field = base * (0.8 - detail * 0.25) + fine(seed, x * scale * 4.0, y * scale * 4.0) * detail * 0.25 + curls * 0.2
    field = contrast(field, amount)
    mask = smoothstep(1.0 - density, 0.92, field)
    plume = mix_color(color(c, 1, 0xff444a52), color(c, 2, 0xffd5d9dd), smoothstep(0.42, 0.9, field))
    return mix_color(color(c, 0, 0xff111318), plume, mask)


@njit(cache=True)
def _stone(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.011), 0.00001)
    grain = clamp01(parameter(p, 1, 0.3))
    veins = clamp01(parameter(p, 2, 0.58))
    vein_scale = max(parameter(p, 3, 0.024), 0.00001)
    x, y = warp(seed, x, y, scale, parameter(p, 4, 18.0))
    amount = max(parameter(p, 5, 1.12), 0.0)
    body = fbm(seed, x * scale, y * scale)
    body = contrast(body + (fine(seed, x * scale * 7.0, y * scale * 7.0) - 0.5) * grain * 0.35, amount)
    vein_field = abs(math.sin((x + y * 0.42) * vein_scale + fbm(seed, x * scale * 0.7, y * scale * 0.7) * 9.0))
    vein = (1.0 - smoothstep(0.0, 0.16, vein_field)) * veins
    return mix_color(mix_color(color(c, 0, 0xff343536), color(c, 1, 0xffaaa79f), body),
                     color(c, 2, 0xffe2ded3), vein)


@njit(cache=True)
def _wood(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.0039), 0.00001)
    rings = max(parameter(p, 1, 11.4), 0.1)
    grain = clamp01(parameter(p, 2, 0.75))
    distortion = parameter(p, 3, 15.0)
    stretch = max(parameter(p, 4, 7.42), 0.1)
    amount = max(parameter(p, 5, 0.17), 0.0)
    low = fbm(seed, x * scale / stretch, y * scale)
    phase = (x * scale / stretch + low * distortion * scale) * rings * 2.0 * PI
    ring = contrast(0.5 + 0.5 * math.sin(phase), amount)
    fibers = fine(seed, x * scale * 7.0 / stretch, y * scale * 7.0)
    tone = clamp01(ring * (1.0 - grain * 0.35) + fibers * grain * 0.35)
    pore_noise = fine(seed, x * scale * 19.0 / stretch, y * scale * 12.0)
    elongated_pores = clamp01(1.0 - abs(pore_noise * 2.0 - 1.0)) ** 7.0
    pores = smoothstep(0.62, 0.94, elongated_pores) * grain * 0.85
    return mix_color(mix_color(color(c, 0, 0xff3a190b), color(c, 1, 0xffb96e32), tone),
                     color(c, 2, 0xff1d0d07), pores)


@njit(cache=True)
def _camouflage(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.007), 0.00001)
    x, y = warp(seed, x, y, scale * 0.75, parameter(p, 4, 28.0))
    field = fbm(seed, x * scale, y * scale)
    softness = max(parameter(p, 5, 0.035), 0.0001)
    first_mask = smoothstep(parameter(p, 1, 0.36) - softness, parameter(p, 1, 0.36) + softness, field)
    second_mask = smoothstep(parameter(p, 2, 0.58) - softness, parameter(p, 2, 0.58) + softness, field)
    third_mask = smoothstep(parameter(p, 3, 0.76) - softness, parameter(p, 3, 0.76) + softness, field)
    result = mix_color(color(c, 0, 0xff202719), color(c, 1, 0xff4f6134), first_mask)
    result = mix_color(result, color(c, 2, 0xff786344), second_mask)
    return mix_color(result, color(c, 3, 0xffb4a477), third_mask)


@njit(cache=True)
def _paper(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.012), 0.00001)
    density = max(parameter(p, 1, 72.0), 1.0)
    fiber_strength = clamp01(parameter(p, 2, 0.24))
    grain = clamp01(parameter(p, 3, 0.16))
    stains = clamp01(parameter(p, 4, 0.14))
    roughness = clamp01(parameter(p, 5, 0.35))
    body = fbm(seed, x * scale, y * scale)
    fibers = abs(math.sin((x + fine(seed, x * scale, y * scale) * 8.0) * scale * density)) ** 18.0
    fibers = max(fibers, abs(math.sin((y + body * 7.0) * scale * density * 0.83)) ** 22.0)
    speckle = hash2(x, y, seed)
    tone = clamp01(0.58 + (body - 0.5) * roughness + (speckle - 0.5) * grain)
    result = mix_color(color(c, 0, 0xffe4d4ae), color(c, 1, 0xfff5e9c9), tone)
    result = mix_color(result, color(c, 2, 0xffb59c70), fibers * fiber_strength)
    stain = smoothstep(0.72, 0.94, fbm(seed, x * scale * 0.35, y * scale * 0.35)) * stains
    return mix_color(result, color(c, 3, 0xff8a6841), stain)


@njit(cache=True)
def _rust(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.009), 0.00001)
    corrosion = clamp01(parameter(p, 1, 0.55))
    pitting = clamp01(parameter(p, 2, 0.34))
    flakes = clamp01(parameter(p, 3, 0.42))
    x, y = warp(seed, x, y, scale, parameter(p, 4, 16.0))
    amount = max(parameter(p, 5, 1.25), 0.0)
    field = contrast(fbm(seed, x * scale, y * scale), amount)
    rust_mask = smoothstep(0.72 - corrosion * 0.55, 0.82 - corrosion * 0.25, field)
    oxidation = fine(seed, x * scale * 4.2, y * scale * 4.2)
    rust_color = mix_color(color(c, 1, 0xff3a170b), color(c, 2, 0xff9b3e13), oxidation)
    rust_color = mix_color(rust_color, color(c, 3, 0xffe07822), smoothstep(0.68, 0.95, oxidation) * flakes)
    pits = (1.0 - smoothstep(0.0, 0.16, cell_edge(seed, x, y, scale * 2.4, 1.0))) * pitting
    return mix_color(mix_color(color(c, 0, 0xff596065), rust_color, rust_mask), color(c, 1, 0xff3a170b), pits)


@njit(cache=True)
def _fabric(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.018), 0.00001)
    horizontal = max(parameter(p, 1, 28.0), 1.0)
    vertical = max(parameter(p, 2, 28.0), 1.0)
    irregularity = clamp01(parameter(p, 3, 0.14))
    depth = clamp01(parameter(p, 4, 0.48))
    fuzz = clamp01(parameter(p, 5, 0.1))
    wobble = (fine(seed, x * scale * 2.0, y * scale * 2.0) - 0.5) * irregularity * 2.0
    warp_thread = 0.5 + 0.5 * math.cos((x * scale * vertical + wobble) * 2.0 * PI)
    weft = 0.5 + 0.5 * math.cos((y * scale * horizontal - wobble) * 2.0 * PI)
    # cell parity: evaluated in float32 like upstream, since x * scale * vertical lands exactly on
    # integers for whole columns/rows (e.g. x=125 at the defaults) and the rounding decides the cell
    fx = np.float32(np.float32(x) * np.float32(scale)) * np.float32(vertical)
    fy = np.float32(np.float32(y) * np.float32(scale)) * np.float32(horizontal)
    over = _cfmod(float(math.floor(fx) + math.floor(fy)), 2.0)
    thread = warp_thread if over < 1.0 else weft
    result = mix_color(color(c, 0, 0xff496b82), color(c, 1, 0xff7895a7), over)
    result = mix_color(result, color(c, 2, 0xff263c4b), (1.0 - thread) * depth)
    result = mix_color(result, color(c, 3, 0xffb6cad4), smoothstep(0.72, 1.0, thread) * depth)
    return mix_color(result, color(c, 3, 0xffb6cad4), smoothstep(1.0 - fuzz * 0.12, 1.0, hash2(x, y, seed)) * fuzz)


@njit(cache=True)
def _topography(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.0055), 0.00001)
    line_count = max(parameter(p, 1, 16.0), 1.0)
    thickness = clampf(parameter(p, 2, 0.11), 0.001, 0.49)
    shading = clamp01(parameter(p, 3, 0.42))
    x, y = warp(seed, x, y, scale * 0.75, parameter(p, 4, 12.0))
    height = contrast(fbm(seed, x * scale, y * scale), parameter(p, 5, 1.1))
    contour_position = fract(height * line_count)
    distance = min(contour_position, 1.0 - contour_position)
    line = 1.0 - smoothstep(thickness * 0.45, thickness, distance)
    fill = 0.5 + (height - 0.5) * shading * 2.0
    return mix_color(mix_color(color(c, 0, 0xff173c3b), color(c, 1, 0xffb5c98a), fill), color(c, 2, 0xfff2e6b6), line)


@njit(cache=True)
def _cells(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.018), 0.00001)
    jitter = parameter(p, 1, 0.92)
    border_width = clampf(parameter(p, 2, 0.12), 0.001, 0.9)
    glow = clamp01(parameter(p, 3, 0.28))
    x, y = warp(seed, x, y, scale, parameter(p, 4, 6.0))
    variation = clamp01(parameter(p, 5, 0.42))
    edge_distance = cell_edge(seed, x, y, scale, jitter)
    edge = 1.0 - smoothstep(border_width * 0.35, border_width, edge_distance)
    halo = (1.0 - smoothstep(border_width, border_width * 3.5, edge_distance)) * glow
    cell_tone = cell_variation(seed, x, y, scale, jitter) * variation
    result = mix_color(color(c, 0, 0xff101a24), color(c, 1, 0xff216e78), 0.55 + cell_tone)
    result = mix_color(result, color(c, 2, 0xff5ad1c8), max(edge, halo))
    return mix_color(result, color(c, 3, 0xffc9fff4), edge * glow)


@njit(cache=True)
def _cracks(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.019), 0.00001)
    width = clampf(parameter(p, 1, 0.065), 0.001, 0.5)
    density = clamp01(parameter(p, 2, 0.72))
    x, y = warp(seed, x, y, scale, parameter(p, 3, 9.0))
    depth = clamp01(parameter(p, 4, 0.72))
    branching = clamp01(parameter(p, 5, 0.45))
    primary_distance = cell_edge(seed, x, y, scale, 1.0)
    secondary_distance = secondary_cell_edge(seed, x + 37.0, y - 19.0, scale * 1.9, 0.82)
    primary = 1.0 - smoothstep(width * 0.45, width, primary_distance)
    secondary = (1.0 - smoothstep(width * 0.28, width * 0.72, secondary_distance)) * branching
    crack = max(primary, secondary) * density
    edge = (1.0 - smoothstep(width, width * 2.8, primary_distance)) * (1.0 - crack) * depth
    surface = fbm(seed, x * scale * 0.65, y * scale * 0.65)
    result = mix_color(color(c, 0, 0xff8d8374), color(c, 1, 0xffc0b6a5), surface)
    result = mix_color(result, color(c, 3, 0xff554e45), edge)
    return mix_color(result, color(c, 2, 0xff161412), crack * depth)


# ---------------------------------------------------------------------------
# 12..25
# ---------------------------------------------------------------------------


@njit(cache=True)
def _water_ripples(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.0075), 0.00001)
    frequency = max(parameter(p, 1, 22.0), 0.1)
    distortion = parameter(p, 2, 22.0)
    caustics = clamp01(parameter(p, 3, 0.58))
    depth = clamp01(parameter(p, 4, 0.55))
    highlights = clamp01(parameter(p, 5, 0.62))
    x, y = warp(seed, x, y, scale * 0.8, distortion)
    flow = fbm(seed, x * scale, y * scale)
    wave_a = math.sin((x * 0.78 + y * 0.22) * scale * frequency + flow * 5.0)
    wave_b = math.sin((x * -0.31 + y * 0.95) * scale * frequency * 0.73 - flow * 4.0)
    waves = clamp01(0.5 + (wave_a + wave_b) * 0.22)
    light_lines = clamp01(1.0 - abs(wave_a + wave_b) * 0.5) ** 7.0 * caustics
    tone = clamp01(waves * depth + flow * (1.0 - depth))
    return mix_color(mix_color(color(c, 0, 0xff073456), color(c, 1, 0xff168eae), tone),
                     color(c, 2, 0xffb9f7f2), light_lines * highlights)


@njit(cache=True)
def _foliage(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.022), 0.00001)
    density = clamp01(parameter(p, 1, 0.72))
    softness = clampf(parameter(p, 2, 0.16), 0.01, 0.6)
    veins = clamp01(parameter(p, 3, 0.38))
    lighting = clamp01(parameter(p, 4, 0.62))
    variation = clamp01(parameter(p, 5, 0.58))
    x, y = warp(seed, x, y, scale, 4.0 + variation * 10.0)
    edge_distance = cell_edge(seed, x, y, scale, 0.92)
    cell_tone = cell_variation(seed, x, y, scale, 0.92)
    leaf_mask = smoothstep(softness * 0.35, softness, edge_distance)
    overlap = fbm(seed, x * scale * 0.7, y * scale * 0.7)
    leaf_mask *= smoothstep(1.0 - density, 1.0, overlap + density * 0.55)
    vein = (1.0 - smoothstep(softness, softness * 2.1, edge_distance)) * veins
    detail = fine(seed, x * scale * 2.5, y * scale * 2.5)
    light = clamp01(cell_tone * variation + detail * lighting)
    leaf = mix_color(color(c, 1, 0xff245c25), color(c, 2, 0xff4e9a3d), light)
    leaf = mix_color(leaf, color(c, 3, 0xffa4ce65), smoothstep(0.68, 0.96, light) * lighting)
    leaf = mix_color(leaf, color(c, 1, 0xff245c25), vein)
    return mix_color(color(c, 0, 0xff102a13), leaf, leaf_mask)


@njit(cache=True)
def _bricks(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.012), 0.00001)
    aspect = max(parameter(p, 1, 2.15), 0.25)
    mortar_width = clampf(parameter(p, 2, 0.09), 0.005, 0.45)
    irregularity = clamp01(parameter(p, 3, 0.18))
    roughness = clamp01(parameter(p, 4, 0.42))
    bevel = clamp01(parameter(p, 5, 0.55))
    row = math.floor(y * scale)
    stagger = _cfmod(abs(row), 2.0) * 0.5
    wobble = (fine(seed, x * scale * 0.7, y * scale * 0.7) - 0.5) * irregularity
    ux = fract(x * scale / aspect + stagger + wobble * 0.12)
    uy = fract(y * scale + wobble * 0.1)
    dx = min(ux, 1.0 - ux) * aspect
    edge_distance = min(dx, min(uy, 1.0 - uy))
    brick_mask = smoothstep(mortar_width * 0.65, mortar_width, edge_distance)
    edge_light = smoothstep(mortar_width, mortar_width * 3.5, edge_distance) * bevel
    surface = fine(seed, x * scale * 5.0, y * scale * 5.0)
    brick_id = hash2(math.floor(x * scale / aspect + stagger), row, 1777)
    tone = clamp01(0.35 + brick_id * 0.38 + (surface - 0.5) * roughness * 0.7)
    brick = mix_color(color(c, 1, 0xff6f2418), color(c, 2, 0xffb64c32), tone)
    brick = mix_color(brick, color(c, 3, 0xffd77a55), edge_light * 0.32)
    return mix_color(color(c, 0, 0xffb8ad99), brick, brick_mask)


@njit(cache=True)
def _terrain(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.0045), 0.00001)
    water_level = clamp01(parameter(p, 1, 0.34))
    mountain_level = clamp01(parameter(p, 2, 0.7))
    erosion = clamp01(parameter(p, 3, 0.46))
    detail = clamp01(parameter(p, 4, 0.58))
    snow_level = clamp01(parameter(p, 5, 0.86))
    x, y = warp(seed, x, y, scale * 0.7, erosion * 38.0)
    broad = fbm(seed, x * scale, y * scale)
    fine_v = fine(seed, x * scale * 2.3, y * scale * 2.3)
    ridges = 1.0 - abs(fine_v * 2.0 - 1.0)
    height = clamp01(broad * (1.0 - detail * 0.3) + ridges * detail * 0.3)
    left = fbm(seed, (x - 2.0) * scale, y * scale)
    right = fbm(seed, (x + 2.0) * scale, y * scale)
    slope = abs(right - left) * 9.0
    shore = smoothstep(water_level - 0.025, water_level + 0.035, height)
    rock = smoothstep(mountain_level - 0.12, mountain_level + 0.1, height + slope)
    land = mix_color(color(c, 1, 0xff5f873f), color(c, 2, 0xff746c5d), rock)
    land = mix_color(land, color(c, 3, 0xffe9eee8), smoothstep(snow_level - 0.07, snow_level, height))
    water = mix_color(color(c, 0, 0xff194d73), color(c, 1, 0xff5f873f),
                      smoothstep(water_level - 0.12, water_level, height) * 0.2)
    return mix_color(water, land, shore)


@njit(cache=True)
def _ice(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.014), 0.00001)
    crack_width = clampf(parameter(p, 1, 0.075), 0.002, 0.4)
    frost = clamp01(parameter(p, 2, 0.48))
    depth = clamp01(parameter(p, 3, 0.64))
    x, y = warp(seed, x, y, scale, parameter(p, 4, 8.0))
    sparkle = clamp01(parameter(p, 5, 0.32))
    body = fbm(seed, x * scale * 0.65, y * scale * 0.65)
    edge_distance = cell_edge(seed, x, y, scale, 0.95)
    crack = 1.0 - smoothstep(crack_width * 0.45, crack_width, edge_distance)
    fine_v = fine(seed, x * scale * 3.2, y * scale * 3.2)
    frost_mask = smoothstep(0.58, 0.9, fine_v) * frost
    speck = smoothstep(1.0 - sparkle * 0.025, 1.0, hash2(x, y, seed)) * sparkle
    result = mix_color(color(c, 0, 0xff155a79), color(c, 1, 0xff79c8dc), clamp01(body + depth * 0.2))
    result = mix_color(result, color(c, 2, 0xffd8f4f5), max(frost_mask, speck))
    return mix_color(result, color(c, 3, 0xfff5ffff), crack * depth)


@njit(cache=True)
def _sand(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.005), 0.00001)
    frequency = max(parameter(p, 1, 12.0), 0.1)
    angle = parameter(p, 2, 0.32) * 2.0 * PI
    ripples = clamp01(parameter(p, 3, 0.62))
    grain = clamp01(parameter(p, 4, 0.22))
    amount = max(parameter(p, 5, 1.18), 0.0)
    dune = fbm(seed, x * scale, y * scale)
    axis = x * math.cos(angle) + y * math.sin(angle)
    wave = 0.5 + 0.5 * math.sin(axis * scale * frequency + dune * 5.0)
    grains = hash2(x, y, seed)
    tone = dune * (1.0 - ripples * 0.42) + wave * ripples * 0.42
    tone = contrast(tone + (grains - 0.5) * grain * 0.22, amount)
    result = mix_color(color(c, 0, 0xff9a632e), color(c, 1, 0xffd6a85a), tone)
    highlight = smoothstep(0.7, 0.96, wave) * ripples * 0.45
    return mix_color(result, color(c, 2, 0xfff2d58d), highlight)


@njit(cache=True)
def _nebula(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.004), 0.00001)
    x, y = warp(seed, x, y, scale * 0.65, parameter(p, 1, 44.0))
    density = clamp01(parameter(p, 2, 0.64))
    stars = clamp01(parameter(p, 3, 0.38))
    glow = clamp01(parameter(p, 4, 0.72))
    amount = max(parameter(p, 5, 1.45), 0.0)
    a = fbm(seed, x * scale, y * scale)
    b = fine(seed, x * scale * 2.1 + 11.0, y * scale * 2.1 - 7.0)
    cloud = contrast(a * 0.68 + b * 0.32, amount)
    mask = smoothstep(0.7 - density * 0.42, 0.88, cloud)
    gas = mix_color(color(c, 1, 0xff542577), color(c, 2, 0xff2367a2), b)
    gas = mix_color(gas, color(c, 3, 0xfff2a4df), smoothstep(0.7, 0.96, cloud) * glow)
    result = mix_color(color(c, 0, 0xff050713), gas, mask)
    star = smoothstep(1.0 - stars * 0.012, 1.0, hash2(x, y, seed))
    return mix_color(result, 0xffffffff, star)


@njit(cache=True)
def _honeycomb(seed, x, y, w, h, p, c):
    scale = max(parameter(p, 0, 0.018), 0.00001)
    border_width = clampf(parameter(p, 1, 0.095), 0.005, 0.4)
    bevel = clamp01(parameter(p, 2, 0.5))
    irregularity = clamp01(parameter(p, 3, 0.12))
    fill = clamp01(parameter(p, 4, 0.72))
    glow = clamp01(parameter(p, 5, 0.35))
    wobble = (fine(seed, x * scale, y * scale) - 0.5) * irregularity
    px = x * scale + wobble
    py = y * scale + wobble * 0.7
    ax = fract(px / 1.7320508) * 1.7320508 - 0.8660254
    ay = fract(py) - 0.5
    bx = fract((px - 0.8660254) / 1.7320508) * 1.7320508 - 0.8660254
    by = fract(py - 0.5) - 0.5
    use_a = ax * ax + ay * ay < bx * bx + by * by
    gx = ax if use_a else bx
    gy = ay if use_a else by
    hex_distance = max(abs(gx) * 0.8660254 + abs(gy) * 0.5, abs(gy))
    edge_distance = 0.5 - hex_distance
    interior = smoothstep(border_width * 0.55, border_width, edge_distance)
    bevel_light = smoothstep(border_width, border_width * 3.2, edge_distance) * bevel
    center = clamp01(1.0 - hex_distance * 1.7)
    cell = mix_color(color(c, 1, 0xff6f3a08), color(c, 2, 0xffe59b12), center * fill)
    cell = mix_color(cell, color(c, 3, 0xffffd45c), bevel_light * glow)
    return mix_color(color(c, 0, 0xff241305), cell, interior)


@njit(cache=True)
def _grass(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.012), 0.00001)
    density = clamp01(parameter(p, 1, 1.0))
    length = max(parameter(p, 2, 50.0), 1.0)
    wind = parameter(p, 3, 1.0)
    patchiness = clamp01(parameter(p, 4, 0.35))
    highlights = clamp01(parameter(p, 5, 0.45))
    patch = fbm(seed, x * s * 0.45, y * s * 0.45)
    bend = fbm(seed, x * s, y * s) * wind * 12.0
    blades = clamp01(0.5 + 0.5 * math.sin((x * s + bend) * length + y * s * 1.7)) ** 9.0
    cover = smoothstep(1.0 - density, 0.92, patch + density * 0.42)
    tone = clamp01(fine(seed, x * s * 3.0, y * s * 3.0) + patchiness * (patch - 0.5))
    blade = mix_color(color(c, 1, 0xff1d491f), color(c, 2, 0xff4d8a32), tone)
    blade = mix_color(blade, color(c, 3, 0xff9bc45a), blades * highlights)
    return mix_color(color(c, 0, 0xff3a2b18), blade, clamp01(cover * 0.75 + blades * 0.5))


@njit(cache=True)
def _dirt(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.009), 0.00001)
    clumps = clamp01(parameter(p, 1, 0.62))
    moisture = clamp01(parameter(p, 2, 0.35))
    pebbles = clamp01(parameter(p, 3, 0.22))
    roughness = clamp01(parameter(p, 4, 0.68))
    variation = clamp01(parameter(p, 5, 0.52))
    broad = fbm(seed, x * s, y * s)
    crumb = fine(seed, x * s * 5.0, y * s * 5.0)
    body = clamp01(broad * (1.0 - clumps * 0.3) + crumb * clumps * 0.3)
    body = clamp01(body + (hash2(x, y, seed) - 0.5) * roughness * 0.22)
    earth = mix_color(color(c, 0, 0xff2a190e), color(c, 1, 0xff694226), body + moisture * 0.1)
    earth = mix_color(earth, color(c, 2, 0xffa1764d), smoothstep(0.58, 0.9, broad) * variation * (1.0 - moisture))
    # x * 0.37f / y * 0.37f are float32 products upstream
    pebble = smoothstep(1.0 - pebbles * 0.018, 1.0,
                        hash2(np.float32(x) * np.float32(0.37), np.float32(y) * np.float32(0.37), seed + 91))
    return mix_color(earth, color(c, 3, 0xffb1a28c), pebble)


@njit(cache=True)
def _leather(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.011), 0.00001)
    wrinkles = clamp01(parameter(p, 1, 0.58))
    pores = clamp01(parameter(p, 2, 0.42))
    grain = clamp01(parameter(p, 3, 0.5))
    softness = clamp01(parameter(p, 4, 0.38))
    shine = clamp01(parameter(p, 5, 0.035))
    x, y = warp(seed, x, y, s, 10.0 * wrinkles)
    base = fbm(seed, x * s, y * s)
    fold = 1.0 - abs(math.sin((x * 0.63 + y) * s * 8.0 + base * 7.0))
    fine_v = fine(seed, x * s * 8.0, y * s * 8.0)
    pore = smoothstep(0.78, 0.96, fine_v) * pores
    tone = clamp01(0.42 + (base - 0.5) * grain + fold * wrinkles * 0.24)
    result = mix_color(color(c, 0, 0xff30160f), color(c, 1, 0xff7b3d25), tone + softness * 0.18)
    result = mix_color(result, color(c, 2, 0xffb36b43), smoothstep(0.72, 0.98, fold) * shine)
    return mix_color(result, color(c, 3, 0xff1c0c08), pore)


@njit(cache=True)
def _concrete(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.014), 0.00001)
    aggregate = clamp01(parameter(p, 1, 0.45))
    stains = clamp01(parameter(p, 2, 0.28))
    roughness = clamp01(parameter(p, 3, 0.65))
    cracks = clamp01(parameter(p, 4, 0.14))
    amount = parameter(p, 5, 1.08)
    x, y = warp(seed, x, y, s * 0.7, 16.0)
    body = contrast(fbm(seed, x * s, y * s), amount)
    grit = fine(seed, x * s * 6.0, y * s * 6.0)
    tone = clamp01(body * 0.65 + grit * roughness * 0.35)
    result = mix_color(color(c, 0, 0xff5a5955), color(c, 2, 0xffc7c3b9), tone)
    result = mix_color(result, color(c, 1, 0xff99968e), 0.35 + aggregate * 0.3)
    result = mix_color(result, color(c, 0, 0xff5a5955),
                       smoothstep(0.72, 0.94, fbm(seed, x * s * 0.3, y * s * 0.3)) * stains)
    line = abs(math.sin((x + y * 0.41) * s * 3.0 + body * 12.0))
    crack = (1.0 - smoothstep(0.0, 0.018 + cracks * 0.09, line)) * cracks
    speck = smoothstep(0.995, 1.0, hash2(x, y, seed)) * aggregate
    return mix_color(mix_color(result, color(c, 3, 0xff353532), crack), color(c, 2, 0xffc7c3b9), speck)


@njit(cache=True)
def _asphalt(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.02), 0.00001)
    aggregate = clamp01(parameter(p, 1, 0.72))
    tar = clamp01(parameter(p, 2, 0.44))
    wear = clamp01(parameter(p, 3, 0.28))
    speckles = clamp01(parameter(p, 4, 0.48))
    amount = parameter(p, 5, 1.3)
    coarse = contrast(fine(seed, x * s * 2.2, y * s * 2.2), amount)
    fine_v = hash2(x, y, seed)
    tone = clamp01(coarse * aggregate + fine_v * (1.0 - aggregate))
    result = mix_color(color(c, 0, 0xff17191a), color(c, 1, 0xff3e4242), tone + tar * 0.12)
    stone = smoothstep(0.72, 0.96, coarse) * aggregate
    result = mix_color(result, color(c, 2, 0xff898b85), stone)
    dust = smoothstep(1.0 - speckles * 0.035, 1.0, fine_v) * (0.4 + wear * 0.6)
    return mix_color(result, color(c, 3, 0xffb0a995), dust)


@njit(cache=True)
def _moss(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.018), 0.00001)
    density = clamp01(parameter(p, 1, 0.74))
    fibers = clamp01(parameter(p, 2, 0.58))
    moisture = clamp01(parameter(p, 3, 0.38))
    variation = clamp01(parameter(p, 4, 0.62))
    clumps = clamp01(parameter(p, 5, 0.46))
    patch = fbm(seed, x * s * 0.65, y * s * 0.65)
    fuzz = fine(seed, x * s * 4.0, y * s * 4.0)
    mask = smoothstep(1.0 - density, 0.9, patch + density * 0.45)
    tone = clamp01(fuzz * variation + patch * (1.0 - variation))
    result = mix_color(color(c, 1, 0xff29451c), color(c, 2, 0xff648044), tone)
    tips = smoothstep(0.75, 0.96, fuzz) * fibers
    tips += smoothstep(0.995, 1.0, hash2(x, y, seed)) * fibers
    result = mix_color(result, color(c, 3, 0xffb1bd69), clamp01(tips))
    result = mix_color(result, color(c, 1, 0xff29451c), moisture * (1.0 - tone) * 0.4)
    return mix_color(color(c, 0, 0xff292516), result, mask * (0.7 + clumps * 0.3))


# ---------------------------------------------------------------------------
# 26..39
# ---------------------------------------------------------------------------


@njit(cache=True)
def _fire(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.006), 0.00001)
    frequency = parameter(p, 1, 8.0)
    turbulence = parameter(p, 2, 36.0)
    intensity = clamp01(parameter(p, 3, 0.78))
    smoke = clamp01(parameter(p, 4, 0.18))
    detail = clamp01(parameter(p, 5, 0.62))
    x, y = warp(seed, x, y, s, turbulence)
    base = fbm(seed, x * s, y * s * 1.7)
    tongues = 0.5 + 0.5 * math.sin(x * s * frequency + base * 9.0 - y * s * 3.0)
    flame = clamp01(base * (1.0 - detail * 0.35) + tongues * detail * 0.35)
    flame = smoothstep(0.28 + smoke * 0.25, 0.9, flame) * intensity
    result = mix_color(color(c, 0, 0xff100807), color(c, 1, 0xffb51b08), flame)
    result = mix_color(result, color(c, 2, 0xffff7a0a), smoothstep(0.38, 0.75, flame))
    return mix_color(result, color(c, 3, 0xffffe98a), smoothstep(0.72, 0.98, flame))


@njit(cache=True)
def _aurora(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.004), 0.00001)
    ribbons = parameter(p, 1, 7.0)
    x, y = warp(seed, x, y, s * 0.7, parameter(p, 2, 34.0))
    glow = clamp01(parameter(p, 3, 0.72))
    stars = clamp01(parameter(p, 4, 0.2))
    amount = parameter(p, 5, 1.32)
    flow = fbm(seed, x * s, y * s)
    band = clamp01(0.5 + 0.5 * math.sin(y * s * ribbons + flow * 9.0)) ** 3.0
    band = contrast(band, amount)
    light = mix_color(color(c, 1, 0xff42d9a0), color(c, 2, 0xff65e4e8), flow)
    light = mix_color(light, color(c, 3, 0xffa56de2), smoothstep(0.62, 0.95, flow))
    result = mix_color(color(c, 0, 0xff061227), light, band * glow)
    star = smoothstep(1.0 - stars * 0.009, 1.0, hash2(x, y, seed))
    return mix_color(result, 0xffffffff, star)


@njit(cache=True)
def _oil_slick(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.008), 0.00001)
    bands = parameter(p, 1, 13.0)
    x, y = warp(seed, x, y, s, parameter(p, 2, 32.0))
    iridescence = clamp01(parameter(p, 3, 0.82))
    darkness = clamp01(parameter(p, 4, 0.3))
    amount = parameter(p, 5, 1.2)
    field = fbm(seed, x * s, y * s)
    phase = fract((field + x * s * 0.13 - y * s * 0.08) * bands)
    phase = contrast(phase, amount)
    spectrum = mix_color(color(c, 1, 0xffd5329b), color(c, 2, 0xff25d4d0), smoothstep(0.0, 0.5, phase))
    spectrum = mix_color(spectrum, color(c, 3, 0xfff0c33c), smoothstep(0.5, 1.0, phase))
    return mix_color(color(c, 0, 0xff10101a), spectrum, iridescence * (1.0 - darkness * 0.65))


@njit(cache=True)
def _watercolor(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.006), 0.00001)
    blooms = clamp01(parameter(p, 1, 0.72))
    pigment = clamp01(parameter(p, 2, 0.65))
    edges = clamp01(parameter(p, 3, 0.38))
    paper = clamp01(parameter(p, 4, 0.22))
    diffusion = clamp01(parameter(p, 5, 0.62))
    x, y = warp(seed, x, y, s, diffusion * 28.0)
    a = fbm(seed, x * s, y * s)
    b = fbm(seed, x * s * 0.82 + 9.0, y * s * 0.82 - 13.0)
    granulation = fine(seed, x * s * 5.5, y * s * 5.5)
    first_wash = smoothstep(0.38 - blooms * 0.16, 0.68, a)
    second_wash = smoothstep(0.48 - blooms * 0.12, 0.77, b)
    first_wash *= 0.58 + granulation * 0.42
    second_wash *= 0.62 + (1.0 - granulation) * 0.38

    first_boundary = 1.0 - smoothstep(0.018, 0.09, abs(a - 0.5))
    second_boundary = 1.0 - smoothstep(0.018, 0.085, abs(b - 0.56))
    pooling = max(first_boundary, second_boundary) * edges

    result = color(c, 0, 0xfff4ebd8)
    result = mix_color(result, color(c, 1, 0xff337fb3), first_wash * pigment * 0.72)
    result = mix_color(result, color(c, 2, 0xffd85d79), second_wash * pigment * 0.68)
    result = mix_color(result, color(c, 3, 0xff493f76), pooling * pigment * 0.7)

    horizontal_fiber = abs(math.sin((y + a * 3.0) * s * 85.0)) ** 24.0
    paper_grain = abs(hash2(x, y, seed) - 0.5) * 2.0
    fiber = clamp01(horizontal_fiber * 0.55 + paper_grain * 0.45) * paper
    paper_shade = np.int64(0xffffffff) if hash2(x, y, seed + 31) > 0.5 else np.int64(0xffc8bba4)
    return mix_color(result, paper_shade, fiber * 0.16)


@njit(cache=True)
def _flow(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.006), 0.00001)
    frequency = parameter(p, 1, 12.0)
    x, y = warp(seed, x, y, s, parameter(p, 2, 46.0))
    symmetry = clamp01(parameter(p, 3, 0.3))
    sharpness = max(parameter(p, 4, 1.25), 0.05)
    glow = clamp01(parameter(p, 5, 0.62))
    q = fbm(seed, x * s, y * s)
    phase = math.sin((x + y * symmetry) * s * frequency + q * 11.0)
    band = clamp01(0.5 + 0.5 * phase) ** sharpness
    result = mix_color(color(c, 0, 0xff111128), color(c, 1, 0xff5155d9), band)
    result = mix_color(result, color(c, 2, 0xffde4bb3), smoothstep(0.42, 0.82, q))
    line = clamp01(1.0 - abs(phase)) ** 8.0 * glow
    return mix_color(result, color(c, 3, 0xff7ef2e7), line)


@njit(cache=True)
def _opal(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.007), 0.00001)
    play = clamp01(parameter(p, 1, 0.82))
    milk = clamp01(parameter(p, 2, 0.48))
    bands = parameter(p, 3, 8.0)
    x, y = warp(seed, x, y, s, parameter(p, 4, 28.0))
    glow = clamp01(parameter(p, 5, 0.62))
    a = fbm(seed, x * s, y * s)
    b = fine(seed, x * s * 2.0, y * s * 2.0)
    phase = fract((a * 0.7 + b * 0.3) * bands)
    spectrum = mix_color(color(c, 1, 0xff4fd5d1), color(c, 2, 0xfff072b6), smoothstep(0.15, 0.58, phase))
    spectrum = mix_color(spectrum, color(c, 3, 0xffffc857), smoothstep(0.58, 0.95, phase))
    return mix_color(color(c, 0, 0xffdde8df), spectrum,
                     play * (1.0 - milk * 0.55) * smoothstep(0.22, 0.8, a + glow * 0.12))


@njit(cache=True)
def _damascus(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.006), 0.00001)
    layers = parameter(p, 1, 22.0)
    folding = clamp01(parameter(p, 2, 0.72))
    x, y = warp(seed, x, y, s, parameter(p, 3, 34.0))
    polish = clamp01(parameter(p, 4, 0.58))
    amount = parameter(p, 5, 1.4)
    q = fbm(seed, x * s, y * s)
    band = contrast(0.5 + 0.5 * math.sin((x + y * 0.22) * s * layers + q * folding * 13.0), amount)
    metal = mix_color(color(c, 0, 0xff171c20), color(c, 1, 0xff66727a), band)
    metal = mix_color(metal, color(c, 2, 0xffc8d0d2), smoothstep(0.76, 0.98, band) * polish)
    return mix_color(metal, color(c, 3, 0xff304c57), smoothstep(0.42, 0.58, q) * (1.0 - band) * 0.35)


@njit(cache=True)
def _lightning(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.006), 0.00001)
    branches = parameter(p, 1, 7.0)
    x, y = warp(seed, x, y, s, parameter(p, 2, 42.0))
    width = max(parameter(p, 3, 0.055), 0.002)
    glow = clamp01(parameter(p, 4, 0.82))
    intensity = clamp01(parameter(p, 5, 0.9))
    q = fbm(seed, x * s, y * s)
    path_a = abs(math.sin((x * 0.35 + y) * s * branches + q * 11.0))
    path_b = abs(math.sin((x - y * 0.48) * s * branches * 1.7 - q * 9.0))
    distance = min(path_a, path_b + 0.08)
    halo = 1.0 - smoothstep(width, width * 5.0, distance)
    bolt = 1.0 - smoothstep(width * 0.25, width, distance)
    result = mix_color(color(c, 0, 0xff030612), color(c, 1, 0xff2546a8), halo * glow)
    result = mix_color(result, color(c, 2, 0xff68c9ff), bolt * intensity)
    return mix_color(result, color(c, 3, 0xfff2fcff), bolt * bolt)


@njit(cache=True)
def _velvet(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.014), 0.00001)
    fibers = clamp01(parameter(p, 1, 0.78))
    angle = parameter(p, 2, 0.18) * 2.0 * PI
    softness = clamp01(parameter(p, 3, 0.7))
    sheen = clamp01(parameter(p, 4, 0.62))
    folds = clamp01(parameter(p, 5, 0.34))
    axis = x * math.cos(angle) + y * math.sin(angle)
    q = fbm(seed, x * s * 0.5, y * s * 0.5)
    nap = clamp01(0.5 + 0.5 * math.sin(axis * s * 70.0 + q * 4.0)) ** 5.0
    fold = 0.5 + 0.5 * math.sin((x - y * 0.3) * s * 4.0 + q * 8.0)
    result = mix_color(color(c, 0, 0xff17051e), color(c, 1, 0xff5e176a), softness + fold * folds * 0.3)
    result = mix_color(result, color(c, 2, 0xffc148b8), nap * fibers * sheen)
    return mix_color(result, color(c, 3, 0xfff0a6d8), nap ** 4.0 * sheen)


@njit(cache=True)
def _ink_marbling(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.006), 0.00001)
    ribbons = parameter(p, 1, 14.0)
    x, y = warp(seed, x, y, s, parameter(p, 2, 48.0))
    feather = clamp01(parameter(p, 3, 0.52))
    balance = clamp01(parameter(p, 4, 0.5))
    amount = parameter(p, 5, 1.25)
    q = fbm(seed, x * s, y * s)
    wave = contrast(0.5 + 0.5 * math.sin((x + y * 0.45) * s * ribbons + q * 12.0), amount)
    ink = smoothstep(0.22 - feather * 0.1, 0.78 + feather * 0.1, wave)
    result = mix_color(color(c, 0, 0xfff3e9d3), color(c, 1, 0xff164b78), ink * (1.0 - balance * 0.35))
    result = mix_color(result, color(c, 2, 0xff9d2949), smoothstep(0.58, 0.92, q) * balance)
    return mix_color(result, color(c, 3, 0xff1c1831), clamp01(1.0 - abs(wave * 2.0 - 1.0)) ** 8.0 * 0.55)


@njit(cache=True)
def _holographic(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.008), 0.00001)
    spectrum = parameter(p, 1, 12.0)
    crinkles = clamp01(parameter(p, 2, 0.7))
    diffraction = clamp01(parameter(p, 3, 0.82))
    angle = parameter(p, 4, 0.3) * 2.0 * PI
    shine = clamp01(parameter(p, 5, 0.72))
    q = fbm(seed, x * s, y * s)
    axis = x * math.cos(angle) + y * math.sin(angle)
    phase = fract(axis * s * spectrum + q * crinkles * 3.0)
    result = mix_color(color(c, 1, 0xff4de5e2), color(c, 2, 0xffe95ac8), smoothstep(0.1, 0.55, phase))
    result = mix_color(result, color(c, 3, 0xfff7ea72), smoothstep(0.55, 0.95, phase))
    result = mix_color(color(c, 0, 0xffcdd4de), result, diffraction)
    return mix_color(result, 0xffffffff, clamp01(1.0 - abs(phase - 0.5) * 2.0) ** 12.0 * shine)


@njit(cache=True)
def _bioluminescence(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.009), 0.00001)
    veins = clamp01(parameter(p, 1, 0.72))
    branching = clamp01(parameter(p, 2, 0.58))
    x, y = warp(seed, x, y, s, parameter(p, 3, 30.0))
    glow = clamp01(parameter(p, 4, 0.86))
    depth = clamp01(parameter(p, 5, 0.62))
    a = fbm(seed, x * s, y * s)
    b = fine(seed, x * s * 2.2, y * s * 2.2)
    ridge_a = 1.0 - abs(a * 2.0 - 1.0)
    ridge_b = 1.0 - abs(b * 2.0 - 1.0)
    network = max(ridge_a ** 9.0, ridge_b ** 12.0 * branching) * veins
    result = mix_color(color(c, 0, 0xff031a1a), color(c, 1, 0xff0b514d), a * depth)
    result = mix_color(result, color(c, 2, 0xff20e6c2), smoothstep(0.05, 0.55, network) * glow)
    return mix_color(result, color(c, 3, 0xffc4fff1), smoothstep(0.62, 0.98, network))


@njit(cache=True)
def _cosmic_vortex(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.008), 0.00001)
    arms = parameter(p, 1, 5.0)
    twist = parameter(p, 2, 12.0)
    turbulence = clamp01(parameter(p, 3, 0.52))
    stars = clamp01(parameter(p, 4, 0.28))
    core_glow = clamp01(parameter(p, 5, 0.82))
    dx = (x - w * 0.5) / max(w, 1)
    dy = (y - h * 0.5) / max(h, 1)
    radius = math.sqrt(dx * dx + dy * dy)
    angle = math.atan2(dy, dx)
    q = fbm(seed, x * s, y * s)
    spiral = 0.5 + 0.5 * math.sin(angle * arms + radius * twist * 20.0 + q * turbulence * 8.0)
    mask = smoothstep(0.52, 0.08, radius) * smoothstep(0.35, 0.92, spiral)
    gas = mix_color(color(c, 1, 0xff274aa8), color(c, 2, 0xff8d3ec2), q)
    result = mix_color(color(c, 0, 0xff02030b), gas, mask)
    result = mix_color(result, color(c, 3, 0xffffdda0), smoothstep(0.18, 0.0, radius) * core_glow)
    return mix_color(result, 0xffffffff, smoothstep(1.0 - stars * 0.009, 1.0, hash2(x, y, seed)))


@njit(cache=True)
def _lava_lamp(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.007), 0.00001)
    blobs = parameter(p, 1, 6.0)
    softness = clamp01(parameter(p, 2, 0.24))
    x, y = warp(seed, x, y, s, parameter(p, 3, 32.0))
    glow = clamp01(parameter(p, 4, 0.58))
    amount = parameter(p, 5, 1.3)
    a = math.sin(x * s * blobs + fbm(seed, x * s, y * s) * 7.0)
    b = math.sin(y * s * blobs * 0.83 - fine(seed, x * s, y * s) * 6.0)
    field = contrast(clamp01(0.5 + (a + b) * 0.24), amount)
    mask = smoothstep(0.52 - softness * 0.3, 0.52 + softness * 0.3, field)
    blob = mix_color(color(c, 1, 0xffff3d81), color(c, 2, 0xffff8a24), field)
    blob = mix_color(blob, color(c, 3, 0xffffd45c), smoothstep(0.74, 0.98, field) * glow)
    return mix_color(color(c, 0, 0xff1c0929), blob, mask)


# ---------------------------------------------------------------------------
# 40..50 (centred compositions)
# ---------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _unit(w, h):
    return float(max(min(w, h), 1))


@njit(cache=True)
def _event_horizon(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.008), 0.00001)
    tilt = clamp01(parameter(p, 1, 0.72))
    horizon = clampf(parameter(p, 2, 0.15), 0.04, 0.35)
    disk_width = max(parameter(p, 3, 0.065), 0.005)
    lensing = clamp01(parameter(p, 4, 0.82))
    stars = clamp01(parameter(p, 5, 0.22))
    unit = _unit(w, h)
    dx = (x - w * 0.5) / unit
    dy = (y - h * 0.5) / unit
    radius = math.sqrt(dx * dx + dy * dy)
    flatten = 0.42 - tilt * 0.34
    fl = max(flatten, 0.04)
    disk_radius = math.sqrt(dx * dx + (dy / fl) * (dy / fl))
    grain = fbm(seed, x * s, y * s)
    inner = horizon * 1.18
    outer = horizon * 2.75
    disk = smoothstep(inner, inner + disk_width, disk_radius) * (1.0 - smoothstep(outer - disk_width, outer, disk_radius))
    disk *= 0.62 + grain * 0.38
    hot = disk * smoothstep(outer, inner, disk_radius)
    photon_ring = (1.0 - smoothstep(horizon * 0.045, horizon * 0.18, abs(radius - horizon * 1.08))) * lensing
    upper_lens = photon_ring * smoothstep(-horizon * 0.25, -horizon, dy)
    result = mix_color(color(c, 0, 0xff010107), color(c, 3, 0xff7c48ff), upper_lens * 0.7)
    result = mix_color(result, color(c, 1, 0xffff5a18), disk)
    result = mix_color(result, color(c, 2, 0xffffe0a3), hot * (0.45 + 0.55 * smoothstep(0.0, outer, dx)))
    result = mix_color(result, color(c, 2, 0xffffe0a3), photon_ring)
    if radius < horizon:
        result = mix_color(color(c, 0, 0xff010107), 0xff000000, 0.94)
    star = smoothstep(1.0 - stars * 0.008, 1.0, hash2(x, y, seed))
    star *= smoothstep(horizon * 3.2, horizon * 4.0, radius)
    return mix_color(result, 0xffffffff, star)


@njit(cache=True)
def _fractal_bloom(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.01), 0.00001)
    petals = max(parameter(p, 1, 7.0), 2.0)
    layers = max(parameter(p, 2, 5.0), 1.0)
    curl = parameter(p, 3, 4.2)
    filigree = clamp01(parameter(p, 4, 0.7))
    glow = clamp01(parameter(p, 5, 0.72))
    unit = _unit(w, h)
    dx = (x - w * 0.5) / unit
    dy = (y - h * 0.5) / unit
    radius = math.sqrt(dx * dx + dy * dy)
    angle = math.atan2(dy, dx)
    organic = fbm(seed, x * s, y * s) - 0.5
    petal_wave = 0.5 + 0.5 * math.cos(angle * petals + radius * curl * 10.0 + organic * 1.2)
    boundary = 0.18 + 0.19 * petal_wave ** 0.62
    body = smoothstep(boundary + 0.018, boundary - 0.018, radius)
    layer_wave = 0.5 + 0.5 * math.cos(radius * layers * 42.0 - angle * 1.7 + petal_wave * 2.5)
    veins = clamp01(1.0 - abs(layer_wave * 2.0 - 1.0)) ** 9.0 * filigree * body
    edge = (1.0 - smoothstep(0.0, 0.026, abs(radius - boundary))) * body
    bloom = mix_color(color(c, 1, 0xff5940d6), color(c, 2, 0xffff3e93), smoothstep(boundary, 0.04, radius))
    result = mix_color(color(c, 0, 0xff09061b), bloom, body)
    result = mix_color(result, color(c, 3, 0xffffe88a), veins * glow)
    result = mix_color(result, color(c, 3, 0xffffe88a), edge * 0.55)
    return mix_color(result, color(c, 3, 0xffffe88a), smoothstep(0.075, 0.0, radius) * glow)


@njit(cache=True)
def _chromatic_tunnel(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.009), 0.00001)
    depth = max(parameter(p, 1, 18.0), 1.0)
    twist = parameter(p, 2, 5.5)
    facets = max(parameter(p, 3, 7.0), 2.0)
    curvature = clamp01(parameter(p, 4, 0.48))
    glow = clamp01(parameter(p, 5, 0.78))
    unit = _unit(w, h)
    bend = (fbm(seed, x * s * 0.35, y * s * 0.35) - 0.5) * curvature * 0.08
    dx = (x - w * 0.5) / unit + bend
    dy = (y - h * 0.5) / unit - bend * 0.55
    radius = math.sqrt(dx * dx + dy * dy)
    angle = math.atan2(dy, dx)
    perspective = 1.0 / (radius + 0.045)
    spiral = angle * twist + perspective * depth
    pulse = 0.5 + 0.5 * math.sin(spiral)
    facet = 0.5 + 0.5 * math.cos(angle * facets - math.log(radius + 0.025) * twist)
    ribbon = clamp01(1.0 - abs(pulse * 2.0 - 1.0)) ** 5.0
    spectrum = mix_color(color(c, 1, 0xff00d5ff), color(c, 2, 0xffff2c9c), facet)
    result = mix_color(color(c, 0, 0xff030516), spectrum, smoothstep(0.02, 0.48, radius) * (0.32 + ribbon * 0.68))
    result = mix_color(result, color(c, 3, 0xfffff1b8), ribbon * glow * smoothstep(0.5, 0.06, radius))
    return mix_color(result, color(c, 0, 0xff030516), smoothstep(0.05, 0.0, radius))


@njit(cache=True)
def _eclipse_corona(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.008), 0.00001)
    moon = clampf(parameter(p, 1, 0.23), 0.06, 0.4)
    corona_size = max(parameter(p, 2, 0.2), 0.02)
    rays = max(parameter(p, 3, 34.0), 2.0)
    turbulence = clamp01(parameter(p, 4, 0.62))
    diamond_ring = clamp01(parameter(p, 5, 0.8))
    unit = _unit(w, h)
    dx = (x - w * 0.5) / unit
    dy = (y - h * 0.5) / unit
    radius = math.sqrt(dx * dx + dy * dy)
    angle = math.atan2(dy, dx)
    noise = fbm(seed, x * s, y * s)
    ray_shape = 0.38 + 0.62 * clamp01(0.5 + 0.5 * math.sin(angle * rays + noise * turbulence * 7.0)) ** 3.0
    outer = moon + corona_size * ray_shape
    corona = smoothstep(outer, moon * 1.01, radius) * smoothstep(moon * 0.94, moon * 1.04, radius)
    halo = smoothstep(moon + corona_size * 1.8, moon, radius) * 0.32
    rim = 1.0 - smoothstep(moon * 0.015, moon * 0.06, abs(radius - moon))
    point_angle = -0.72
    px = math.cos(point_angle) * moon
    py = math.sin(point_angle) * moon
    point_distance = math.sqrt((dx - px) * (dx - px) + (dy - py) * (dy - py))
    diamond = (1.0 - smoothstep(0.004, 0.035, point_distance)) * diamond_ring
    result = mix_color(color(c, 0, 0xff02030a), color(c, 1, 0xff765bff), halo)
    result = mix_color(result, color(c, 2, 0xffff9b45), corona)
    result = mix_color(result, color(c, 3, 0xffffffff), rim * 0.82 + diamond)
    if radius < moon * 0.985:
        result = mix_color(color(c, 0, 0xff02030a), 0xff000000, 0.88)
    return result


@njit(cache=True)
def _attractor_tables(p):
    """Pixel-independent part of ``strangeAttractor``'s orbit (same expressions, evaluated once)."""
    lobes = max(parameter(p, 1, 3.0), 1.0)
    density = clampf(parameter(p, 2, 18.0), 4.0, 40.0)
    curvature = parameter(p, 3, 6.0)
    samples = int(64.0 + density * 3.0)
    tab = np.empty((4, samples))
    for i in range(samples):
        t = float(i) / float(samples - 1) * 2.0 * PI
        tab[0, i] = math.sin(t * lobes + math.sin(t * curvature) * 0.72) * 0.31
        tab[1, i] = math.sin(t * (lobes + 1.0) + math.cos(t * curvature * 0.73) * 0.58) * 0.31
        tab[2, i] = math.sin(t * 2.0)
        tab[3, i] = math.cos(t * 3.0)
    return tab


@njit(cache=True)
def _strange_attractor(seed, x, y, w, h, p, c, tab):
    s = max(parameter(p, 0, 0.01), 0.00001)
    thickness = max(parameter(p, 4, 0.035), 0.002)
    glow = clamp01(parameter(p, 5, 0.8))
    unit = _unit(w, h)
    dx = (x - w * 0.5) / unit
    dy = (y - h * 0.5) / unit
    phase = (fbm(seed, x * s * 0.2, y * s * 0.2) - 0.5) * 0.12
    # min over sqrt(d2) == sqrt(min d2) exactly (sqrt is monotonic and correctly rounded)
    minimum_sq = 100.0
    for i in range(tab.shape[1]):
        orbit_x = tab[0, i] + tab[2, i] * phase
        orbit_y = tab[1, i] + tab[3, i] * phase
        d2 = (dx - orbit_x) * (dx - orbit_x) + (dy - orbit_y) * (dy - orbit_y)
        if d2 < minimum_sq:
            minimum_sq = d2
    minimum_distance = math.sqrt(minimum_sq)
    line = 1.0 - smoothstep(thickness * 0.22, thickness, minimum_distance)
    halo = 1.0 - smoothstep(thickness, thickness * 4.5, minimum_distance)
    tone = clamp01(0.5 + dx * 1.8 + dy * 0.6)
    orbit = mix_color(color(c, 1, 0xff14d9c5), color(c, 2, 0xffff3a79), tone)
    result = mix_color(color(c, 0, 0xff030511), orbit, halo * glow * 0.62)
    result = mix_color(result, orbit, line)
    return mix_color(result, color(c, 3, 0xfffff1b5), line * line * glow)


@njit(cache=True)
def _ferrofluid_crown(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.012), 0.00001)
    spikes = max(parameter(p, 1, 19.0), 3.0)
    spike_length = clampf(parameter(p, 2, 0.14), 0.0, 0.3)
    body_size = clampf(parameter(p, 3, 0.22), 0.05, 0.4)
    metallic = clamp01(parameter(p, 4, 0.86))
    distortion = clamp01(parameter(p, 5, 0.42))
    unit = _unit(w, h)
    dx = (x - w * 0.5) / unit
    dy = (y - h * 0.52) / unit
    radius = math.sqrt(dx * dx + dy * dy)
    angle = math.atan2(dy, dx)
    noise = fbm(seed, x * s, y * s) - 0.5
    spike_wave = max(math.cos(angle * spikes + noise * distortion * 5.0), 0.0)
    boundary = body_size + spike_wave ** 9.0 * spike_length * (0.72 + noise * 0.35)
    body = smoothstep(boundary + 0.008, boundary - 0.008, radius)
    shadow_radius = math.sqrt((dx + 0.025) * (dx + 0.025) + (dy - 0.035) * (dy - 0.035))
    shadow = smoothstep(boundary + 0.09, boundary, shadow_radius) * (1.0 - body) * 0.34
    normal_light = clamp01(0.45 - dx * 1.15 - dy * 0.92 + noise * 0.18)
    specular = normal_light ** 9.0 * metallic
    result = mix_color(color(c, 0, 0xffe8e1d4), color(c, 1, 0xff07090c), shadow)
    metal = mix_color(color(c, 1, 0xff07090c), color(c, 2, 0xff353d43), normal_light)
    metal = mix_color(metal, color(c, 3, 0xffd9f7ff), specular)
    result = mix_color(result, metal, body)
    rim = (1.0 - smoothstep(0.0, 0.012, abs(radius - boundary))) * body
    return mix_color(result, color(c, 3, 0xffd9f7ff), rim * metallic * 0.55)


@njit(cache=True)
def _supernova(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.01), 0.00001)
    shock_radius = clampf(parameter(p, 1, 0.27), 0.08, 0.45)
    shell_width = max(parameter(p, 2, 0.075), 0.008)
    ejecta = clamp01(parameter(p, 3, 0.72))
    turbulence = clamp01(parameter(p, 4, 0.68))
    stars = clamp01(parameter(p, 5, 0.24))
    unit = _unit(w, h)
    dx = (x - w * 0.5) / unit
    dy = (y - h * 0.5) / unit
    radius = math.sqrt(dx * dx + dy * dy)
    angle = math.atan2(dy, dx)
    noise = fbm(seed, x * s, y * s)
    fine_v = fine(seed, x * s * 2.7, y * s * 2.7)
    distorted_radius = radius + (noise - 0.5) * shell_width * turbulence * 1.8
    shell = 1.0 - smoothstep(shell_width * 0.2, shell_width, abs(distorted_radius - shock_radius))
    rays = clamp01(0.5 + 0.5 * math.sin(angle * 37.0 + fine_v * 8.0)) ** 5.0
    radial_ejecta = rays * smoothstep(shock_radius + shell_width * 2.8, shock_radius * 0.35, radius) * ejecta
    cloud = smoothstep(shock_radius + shell_width * 2.1, shock_radius * 0.18, distorted_radius) * (0.28 + noise * 0.55)
    result = mix_color(color(c, 0, 0xff02030b), color(c, 1, 0xff6923b7), cloud)
    result = mix_color(result, color(c, 2, 0xffff4d22), shell * (0.62 + fine_v * 0.38) + radial_ejecta * 0.54)
    core = smoothstep(shock_radius * 0.22, 0.0, radius)
    result = mix_color(result, color(c, 3, 0xfffff2c5), core + shell * shell * 0.72)
    star = smoothstep(1.0 - stars * 0.008, 1.0, hash2(x, y, seed))
    star *= smoothstep(shock_radius * 1.7, shock_radius * 2.2, radius)
    return mix_color(result, 0xffffffff, star)


@njit(cache=True)
def _iris(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.014), 0.00001)
    pupil = clampf(parameter(p, 1, 0.12), 0.03, 0.28)
    iris_size = clampf(parameter(p, 2, 0.38), pupil + 0.04, 0.48)
    fibers = max(parameter(p, 3, 46.0), 6.0)
    variation = clamp01(parameter(p, 4, 0.72))
    catchlight = clamp01(parameter(p, 5, 0.82))
    unit = _unit(w, h)
    dx = (x - w * 0.5) / unit
    dy = (y - h * 0.5) / unit
    radius = math.sqrt(dx * dx + dy * dy)
    angle = math.atan2(dy, dx)
    noise = fbm(seed, x * s, y * s)
    angular = 0.5 + 0.5 * math.sin(angle * fibers + noise * 8.0 + radius * 65.0)
    radial_fibers = clamp01(angular) ** 3.0 * smoothstep(pupil, iris_size, radius)
    inner_gold = smoothstep(pupil + 0.11, pupil, radius) * smoothstep(pupil * 0.92, pupil * 1.08, radius)
    iris_mask = smoothstep(iris_size + 0.008, iris_size - 0.008, radius)
    iris_color = mix_color(color(c, 1, 0xff123646), color(c, 2, 0xff37b6a5), radial_fibers * variation + noise * 0.18)
    iris_color = mix_color(iris_color, color(c, 3, 0xffe6b650), inner_gold)
    result = mix_color(color(c, 0, 0xff080a0d), iris_color, iris_mask)
    limbal_ring = (1.0 - smoothstep(0.0, 0.026, abs(radius - iris_size))) * iris_mask
    result = mix_color(result, color(c, 0, 0xff080a0d), limbal_ring * 0.86)
    if radius < pupil:
        result = np.int64(0xff010103)
    light_dx = dx + iris_size * 0.34
    light_dy = dy + iris_size * 0.38
    reflection = (1.0 - smoothstep(iris_size * 0.035, iris_size * 0.13,
                                   math.sqrt(light_dx * light_dx + light_dy * light_dy))) * catchlight
    return mix_color(result, 0xffffffff, reflection)


@njit(cache=True)
def _peacock_feather(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.012), 0.00001)
    eye_size = clampf(parameter(p, 1, 0.24), 0.08, 0.4)
    density = max(parameter(p, 2, 54.0), 8.0)
    curvature = clamp01(parameter(p, 3, 0.58))
    iridescence = clamp01(parameter(p, 4, 0.8))
    softness = clamp01(parameter(p, 5, 0.42))
    unit = _unit(w, h)
    dx = (x - w * 0.5) / unit
    dy = (y - h * 0.43) / unit
    curved_x = dx + dy * dy * curvature * 0.48
    feather_radius = math.sqrt((curved_x / 0.34) * (curved_x / 0.34) + (dy / 0.46) * (dy / 0.46))
    feather_mask = smoothstep(1.05, 0.86 - softness * 0.08, feather_radius)
    eye_x = curved_x / eye_size
    eye_y = dy / (eye_size * 1.22)
    eye_radius = math.sqrt(eye_x * eye_x + eye_y * eye_y)
    noise = fbm(seed, x * s, y * s)
    barb_phase = abs(curved_x) * density + dy * density * 0.16 + noise * 3.0
    barbs = clamp01(0.5 + 0.5 * math.sin(barb_phase)) ** 5.0 * feather_mask
    result = mix_color(color(c, 0, 0xff06150f), color(c, 1, 0xff167d55), feather_mask * (0.58 + barbs * 0.42))
    gold_ring = (1.0 - smoothstep(0.08, 0.18, abs(eye_radius - 0.83))) * feather_mask
    blue_eye = smoothstep(0.72, 0.56, eye_radius) * feather_mask
    dark_eye = smoothstep(0.38, 0.18, eye_radius) * feather_mask
    result = mix_color(result, color(c, 3, 0xffe4bc45), gold_ring * iridescence)
    result = mix_color(result, color(c, 2, 0xff1248bc), blue_eye * iridescence)
    result = mix_color(result, color(c, 0, 0xff06150f), dark_eye * 0.88)
    shaft = (1.0 - smoothstep(0.003, 0.012, abs(curved_x))) * smoothstep(0.18, 0.55, dy + 0.36)
    return mix_color(result, color(c, 3, 0xffe4bc45), shaft * 0.62)


@njit(cache=True)
def _nautilus_shell(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.011), 0.00001)
    turns = max(parameter(p, 1, 3.4), 0.5)
    chambers = max(parameter(p, 2, 19.0), 3.0)
    opening = clampf(parameter(p, 3, 0.13), 0.03, 0.3)
    ridges = clamp01(parameter(p, 4, 0.68))
    pearl = clamp01(parameter(p, 5, 0.46))
    unit = _unit(w, h)
    dx = (x - w * 0.5) / unit
    dy = (y - h * 0.5) / unit
    shell_radius = math.sqrt((dx / 0.43) * (dx / 0.43) + (dy / 0.36) * (dy / 0.36))
    radius = math.sqrt(dx * dx + dy * dy)
    angle = math.atan2(dy, dx)
    mask = smoothstep(1.02, 0.96, shell_radius)
    noise = fbm(seed, x * s, y * s)
    spiral_phase = angle - math.log(radius + 0.018) * turns * 2.0
    spiral = clamp01(1.0 - abs(math.sin(spiral_phase))) ** 8.0
    chamber_line = clamp01(1.0 - abs(math.sin(angle * chambers * 0.5 + math.log(radius + 0.025) * 3.0))) ** 12.0
    relief = clamp01(spiral * 0.8 + chamber_line * ridges * 0.65)
    shell = mix_color(color(c, 1, 0xff4a2d22), color(c, 2, 0xffd9ad79),
                      clamp01(0.35 + noise * 0.42 + (0.5 - dy) * 0.28))
    shell = mix_color(shell, color(c, 3, 0xfffff0d1), relief * pearl)
    result = mix_color(color(c, 0, 0xff152228), shell, mask)
    mouth_x = dx - 0.25
    mouth_y = dy + 0.02
    mouth = 1.0 - smoothstep(opening * 0.45, opening, math.sqrt(mouth_x * mouth_x + mouth_y * mouth_y))
    result = mix_color(result, color(c, 1, 0xff4a2d22), mouth * mask * 0.72)
    rim = (1.0 - smoothstep(0.0, 0.025, abs(shell_radius - 0.98))) * mask
    return mix_color(result, color(c, 3, 0xfffff0d1), rim * 0.54)


@njit(cache=True)
def _ringed_planet(seed, x, y, w, h, p, c):
    s = max(parameter(p, 0, 0.009), 0.00001)
    planet_size = clampf(parameter(p, 1, 0.24), 0.08, 0.38)
    tilt = clamp01(parameter(p, 2, 0.72))
    ring_width = clampf(parameter(p, 3, 0.16), 0.03, 0.3)
    atmosphere = clamp01(parameter(p, 4, 0.62))
    stars = clamp01(parameter(p, 5, 0.25))
    unit = _unit(w, h)
    dx = (x - w * 0.5) / unit
    dy = (y - h * 0.5) / unit
    radius = math.sqrt(dx * dx + dy * dy)
    flatten = 0.5 - tilt * 0.38
    fl = max(flatten, 0.06)
    ring_radius = math.sqrt(dx * dx + (dy / fl) * (dy / fl))
    ring_inner = planet_size * 1.15
    ring_outer = ring_inner + ring_width
    ring = smoothstep(ring_inner, ring_inner + 0.012, ring_radius) * \
        (1.0 - smoothstep(ring_outer - 0.012, ring_outer, ring_radius))
    ring_bands = 0.5 + 0.5 * math.sin(ring_radius * 260.0 + fine(seed, x * s, y * s) * 2.0)
    star = smoothstep(1.0 - stars * 0.008, 1.0, hash2(x, y, seed))
    result = mix_color(color(c, 0, 0xff02040d), 0xffffffff, star)
    rings = mix_color(color(c, 1, 0xff15234d), color(c, 3, 0xffffe1a6), 0.46 + ring_bands * 0.45)
    result = mix_color(result, rings, ring * 0.68)
    planet_mask = smoothstep(planet_size + 0.006, planet_size - 0.006, radius)
    light = clamp01(0.52 - dx * 1.55 - dy * 0.75)
    bands = 0.5 + 0.5 * math.sin(dy / planet_size * 23.0 + fbm(seed, x * s, y * s) * 2.8)
    globe = mix_color(color(c, 1, 0xff15234d), color(c, 2, 0xffe58c62), light * (0.72 + bands * 0.28))
    result = mix_color(result, globe, planet_mask)
    result = mix_color(result, rings, ring * smoothstep(-0.01, 0.02, dy))
    atmosphere_rim = (1.0 - smoothstep(0.0, planet_size * 0.06, abs(radius - planet_size))) * atmosphere
    return mix_color(result, color(c, 3, 0xffffe1a6), atmosphere_rim * 0.48)


# ---------------------------------------------------------------------------
# drivers: one small parallel driver per texture (generatePixel's switch), so the
# first use of a preset only JIT-compiles that texture (numba caches each on disk)
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _r_lava(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _lava(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_clouds(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _clouds(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_smoke(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _smoke(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_stone(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _stone(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_wood(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _wood(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_camouflage(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _camouflage(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_paper(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _paper(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_rust(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _rust(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_fabric(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _fabric(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_topography(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _topography(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_cells(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _cells(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_cracks(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _cracks(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_water_ripples(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _water_ripples(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_foliage(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _foliage(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_bricks(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _bricks(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_terrain(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _terrain(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_ice(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _ice(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_sand(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _sand(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_nebula(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _nebula(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_honeycomb(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _honeycomb(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_grass(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _grass(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_dirt(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _dirt(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_leather(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _leather(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_concrete(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _concrete(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_asphalt(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _asphalt(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_moss(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _moss(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_fire(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _fire(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_aurora(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _aurora(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_oil_slick(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _oil_slick(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_watercolor(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _watercolor(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_flow(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _flow(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_opal(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _opal(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_damascus(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _damascus(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_lightning(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _lightning(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_velvet(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _velvet(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_ink_marbling(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _ink_marbling(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_holographic(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _holographic(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_bioluminescence(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _bioluminescence(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_cosmic_vortex(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _cosmic_vortex(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_lava_lamp(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _lava_lamp(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_event_horizon(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _event_horizon(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_fractal_bloom(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _fractal_bloom(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_chromatic_tunnel(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _chromatic_tunnel(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_eclipse_corona(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _eclipse_corona(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_strange_attractor(w, h, seed, p, c, out):
    tab = _attractor_tables(p)  # orbit samples tabulated once per image (identical values)
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _strange_attractor(seed, float(xi), float(yi), w, h, p, c, tab)


@njit(cache=True, parallel=True)
def _r_ferrofluid_crown(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _ferrofluid_crown(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_supernova(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _supernova(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_iris(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _iris(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_peacock_feather(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _peacock_feather(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_nautilus_shell(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _nautilus_shell(seed, float(xi), float(yi), w, h, p, c)


@njit(cache=True, parallel=True)
def _r_ringed_planet(w, h, seed, p, c, out):
    for yi in prange(h):
        for xi in range(w):
            out[yi, xi] = _ringed_planet(seed, float(xi), float(yi), w, h, p, c)


_DRIVERS = (
    _r_lava,
    _r_clouds,
    _r_smoke,
    _r_stone,
    _r_wood,
    _r_camouflage,
    _r_paper,
    _r_rust,
    _r_fabric,
    _r_topography,
    _r_cells,
    _r_cracks,
    _r_water_ripples,
    _r_foliage,
    _r_bricks,
    _r_terrain,
    _r_ice,
    _r_sand,
    _r_nebula,
    _r_honeycomb,
    _r_grass,
    _r_dirt,
    _r_leather,
    _r_concrete,
    _r_asphalt,
    _r_moss,
    _r_fire,
    _r_aurora,
    _r_oil_slick,
    _r_watercolor,
    _r_flow,
    _r_opal,
    _r_damascus,
    _r_lightning,
    _r_velvet,
    _r_ink_marbling,
    _r_holographic,
    _r_bioluminescence,
    _r_cosmic_vortex,
    _r_lava_lamp,
    _r_event_horizon,
    _r_fractal_bloom,
    _r_chromatic_tunnel,
    _r_eclipse_corona,
    _r_strange_attractor,
    _r_ferrofluid_crown,
    _r_supernova,
    _r_iris,
    _r_peacock_feather,
    _r_nautilus_shell,
    _r_ringed_planet,
)


def render_basic(texture_type: int, width: int, height: int, seed: int, params, colors) -> np.ndarray:
    """Render native fast-noise texture ``texture_type`` (0..50) -> uint32 (H, W) ARGB."""
    t = int(texture_type)
    if not 0 <= t < len(_DRIVERS):
        raise ValueError(f"render_basic handles texture types 0..{len(_DRIVERS) - 1}, got {t}")
    out = np.empty((int(height), int(width)), np.uint32)
    _DRIVERS[t](int(width), int(height), int(seed), np.ascontiguousarray(params, np.float64),
                np.ascontiguousarray(colors, np.uint32), out)
    return out
