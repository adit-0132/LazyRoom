"""``NoiseBank`` of ImageToolboxLibs ``ProceduralTextures.cpp`` (Apache-2.0, T8RIN) on the
FastNoiseLite port.

Seven FastNoiseLite instances derived from the texture seed (frequency 1.0):

* base      seed        OpenSimplex2S, FBm, 5 octaves, gain 0.52
* detail    seed+1013   Perlin,        FBm, 3 octaves, gain 0.55
* warpX/Y   seed+2027 / seed+4093  OpenSimplex2, FBm, 3 octaves, gain 0.5
* cells     seed+8191   Cellular Euclidean, Distance2Sub (jitter set per call, clamped 0..2)
* cells2    seed+12289  Cellular Euclidean, Distance2Sub
* cellValue seed+16381  Cellular Euclidean, CellValue
"""
from __future__ import annotations

from ._nb import njit
from .fastnoise_lite import (CELL_EUCLIDEAN, FRACTAL_FBM, FRACTAL_NONE, NOISE_CELLULAR, NOISE_OPENSIMPLEX2,
                             NOISE_OPENSIMPLEX2S, NOISE_PERLIN, RET_CELL_VALUE, RET_DISTANCE2_SUB, get_noise)


@njit(cache=True, inline="always")
def _clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


@njit(cache=True, inline="always")
def _jitter(j):
    return 0.0 if j < 0.0 else (2.0 if j > 2.0 else j)


@njit(cache=True)
def fbm(seed, x, y):
    """``NoiseBank::fbm`` -> 0..1."""
    return _clamp01(get_noise(x, y, seed, 1.0, NOISE_OPENSIMPLEX2S, FRACTAL_FBM, 5, 2.0, 0.52, 0.0, 2.0,
                              CELL_EUCLIDEAN, RET_DISTANCE2_SUB, 1.0) * 0.5 + 0.5)


@njit(cache=True)
def fine(seed, x, y):
    """``NoiseBank::fine`` -> 0..1."""
    return _clamp01(get_noise(x, y, seed + 1013, 1.0, NOISE_PERLIN, FRACTAL_FBM, 3, 2.0, 0.55, 0.0, 2.0,
                              CELL_EUCLIDEAN, RET_DISTANCE2_SUB, 1.0) * 0.5 + 0.5)


@njit(cache=True)
def warp(seed, x, y, scale, amount):
    """``NoiseBank::warp`` -> warped (x, y)."""
    if amount == 0.0:
        return x, y
    sx = x * scale
    sy = y * scale
    dx = get_noise(sx + 19.17, sy - 7.31, seed + 2027, 1.0, NOISE_OPENSIMPLEX2, FRACTAL_FBM, 3, 2.0, 0.5, 0.0, 2.0,
                   CELL_EUCLIDEAN, RET_DISTANCE2_SUB, 1.0)
    dy = get_noise(sx - 13.43, sy + 23.67, seed + 4093, 1.0, NOISE_OPENSIMPLEX2, FRACTAL_FBM, 3, 2.0, 0.5, 0.0, 2.0,
                   CELL_EUCLIDEAN, RET_DISTANCE2_SUB, 1.0)
    return x + dx * amount, y + dy * amount


@njit(cache=True)
def cell_edge(seed, x, y, scale, jitter):
    """``NoiseBank::cellEdge`` -> 0..1 (distance to the nearest cell border)."""
    return _clamp01(get_noise(x * scale, y * scale, seed + 8191, 1.0, NOISE_CELLULAR, FRACTAL_NONE, 3, 2.0, 0.5,
                              0.0, 2.0, CELL_EUCLIDEAN, RET_DISTANCE2_SUB, _jitter(jitter)) + 1.0)


@njit(cache=True)
def secondary_cell_edge(seed, x, y, scale, jitter):
    """``NoiseBank::secondaryCellEdge`` -> 0..1."""
    return _clamp01(get_noise(x * scale, y * scale, seed + 12289, 1.0, NOISE_CELLULAR, FRACTAL_NONE, 3, 2.0, 0.5,
                              0.0, 2.0, CELL_EUCLIDEAN, RET_DISTANCE2_SUB, _jitter(jitter)) + 1.0)


@njit(cache=True)
def cell_variation(seed, x, y, scale, jitter):
    """``NoiseBank::cellVariation`` -> 0..1 (per-cell random value)."""
    return _clamp01(get_noise(x * scale, y * scale, seed + 16381, 1.0, NOISE_CELLULAR, FRACTAL_NONE, 3, 2.0, 0.5,
                              0.0, 2.0, CELL_EUCLIDEAN, RET_CELL_VALUE, _jitter(jitter)) * 0.5 + 0.5)
