"""Private helpers for the ImageToolbox "Effects" group (see effects.py).

Ports of the native code ImageToolbox actually runs for these filters:
aire (MIT, awxkee) + fast_morphology / libblur (BSD-3, awxkee), the
ImageToolboxLibs jhlabs Rust port (Apache-2.0), Trickle (Apache-2.0),
GPUImage (BSD-3) shaders and ImageToolbox's OpenCV helpers (Apache-2.0).
All pixel maths that upstream does on 8-bit Android bitmaps is reproduced on
0..255 integer values where rounding matters.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading

import cv2
import numba
import numpy as np

from ...core.errors import ErrorCode, LazyRoomError
from .._util import F32, apply_lut3d

# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def to_u8(x: np.ndarray) -> np.ndarray:
    """0..1 floats -> 0..255 uint8 (Android 8-bit bitmap values, rounded)."""
    y = np.asarray(x, np.float32) * np.float32(255.0)
    np.rint(y, out=y)
    np.clip(y, 0, 255, out=y)
    return y.astype(np.uint8)


def to8(x: np.ndarray) -> np.ndarray:
    """0..1 floats -> 0..255 int64 (for integer arithmetic)."""
    return to_u8(x).astype(np.int64)


def from8(x: np.ndarray) -> np.ndarray:
    return np.divide(x, np.float32(255.0), dtype=np.float32)


def rs_i32(x) -> np.ndarray:
    """Rust ``f32 as i32``: NaN -> 0, saturating, truncation toward zero."""
    x = np.asarray(x, np.float64)
    with np.errstate(invalid="ignore"):
        out = np.where(np.isnan(x), 0.0, np.clip(np.trunc(x), -2147483648.0, 2147483647.0))
    return out.astype(np.int64)


def wrap_i32(x) -> np.ndarray:
    """Two's complement wrap of int64 values to the i32 range (Rust release-mode overflow)."""
    x = np.asarray(x, np.int64)
    return ((x + 2147483648) % 4294967296) - 2147483648


def round_half_up(x: float) -> int:
    """Kotlin ``Float.roundToInt()``."""
    return int(math.floor(float(x) + 0.5))


def split_rgba(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if img.shape[2] == 4:
        return img[..., :3], img[..., 3]
    return img, None


def join_rgba(rgb: np.ndarray, a: np.ndarray | None) -> np.ndarray:
    if a is None:
        return rgb.astype(F32, copy=False)
    return np.dstack([rgb, a]).astype(F32)


# ---------------------------------------------------------------------------
# Morphology (aire Aire.morphology -> fast_morphology 0.3)
# ---------------------------------------------------------------------------

def morph_kernel(size: int, circle: bool) -> np.ndarray:
    """aire ``MorphKernels.box`` / ``MorphKernels.circle`` (non-zero -> 1).

    ``circle`` marks pixels whose distance to (size/2, size/2) is <= size/3
    (aire compares a double distance with a float radius).
    """
    if not circle:
        return np.ones((size, size), np.uint8)
    c = size // 2
    radius = float(np.float32(size) / np.float32(3.0))
    yy, xx = np.mgrid[0:size, 0:size]
    return (np.sqrt((yy - c) ** 2.0 + (xx - c) ** 2.0) <= radius).astype(np.uint8)


def _dilate(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.dilate(img, mask, borderType=cv2.BORDER_REFLECT_101).reshape(img.shape)


def _erode(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.erode(img, mask, borderType=cv2.BORDER_REFLECT_101).reshape(img.shape)


def morphology(img: np.ndarray, op: str, size: int, circle: bool) -> np.ndarray:
    """``Aire.morphology`` as called by ImageToolbox (8-bit, REFLECT_101 border).

    Like ImageToolbox, images with any non-opaque pixel are processed in RGBA
    mode (alpha is morphed too, on premultiplied colours as stored in Android
    bitmaps); fully opaque images in RGB mode.  Differences saturate at 0.
    """
    mask = morph_kernel(size, circle)
    rgba = img.shape[2] == 4 and bool((to8(img[..., 3]) < 255).any())
    if rgba:
        a = img[..., 3:4]
        work = to_u8(np.concatenate([img[..., :3] * a, a], axis=2))
    else:
        work = to_u8(img[..., :3])

    def opening(x):
        return _dilate(_erode(x, mask), mask)

    def closing(x):
        return _erode(_dilate(x, mask), mask)

    if op == "dilate":
        res = _dilate(work, mask)
    elif op == "erode":
        res = _erode(work, mask)
    elif op == "opening":
        res = opening(work)
    elif op == "closing":
        res = closing(work)
    elif op == "gradient":
        res = cv2.subtract(_dilate(work, mask), _erode(work, mask))
    elif op == "tophat":
        res = cv2.subtract(work, opening(work))
    elif op == "blackhat":
        res = cv2.subtract(closing(work), work)
    else:  # pragma: no cover
        raise ValueError(op)
    res = res.reshape(work.shape)
    if not rgba:
        out = from8(res)
        return np.dstack([out, img[..., 3]]).astype(F32) if img.shape[2] == 4 else out
    a8 = res[..., 3:4].astype(np.float32)
    rgb = np.where(a8 > 0, np.minimum(res[..., :3] * np.float32(255.0) / np.maximum(a8, 1), 255), 0)
    return np.concatenate([rgb, a8], axis=2).astype(F32) / np.float32(255.0)


# ---------------------------------------------------------------------------
# jhlabs (ImageToolboxLibs Rust port)
# ---------------------------------------------------------------------------

def _brightness8(rgb8: np.ndarray) -> np.ndarray:
    """jhlabs ``brightness``: (r + g + b) / 3 with integer division."""
    return (rgb8[..., 0] + rgb8[..., 1] + rgb8[..., 2]) // 3


def chrome(img: np.ndarray, amount: float, exposure: float) -> np.ndarray:
    """``ChromeFilter``: grey bump from the 4-neighbour brightness gradient, exposure curve."""
    rgb, a = split_rgba(img)
    br = np.pad(_brightness8(to8(rgb)), 1, mode="edge")
    left, right = br[1:-1, :-2], br[1:-1, 2:]
    top, bottom = br[:-2, 1:-1], br[2:, 1:-1]
    f32 = np.float32
    bump = np.clip((left - right + top - bottom).astype(f32) / f32(255.0) * f32(amount) + f32(0.5), 0, 1).astype(f32)
    value = np.clip(f32(1.0) - np.exp(-(bump + np.sin(bump)) * f32(exposure)), 0, 1).astype(f32)
    ch = np.clip(rs_i32(value * f32(255.0)), 0, 255)
    g = from8(ch)
    return join_rgba(np.repeat(g[..., None], 3, axis=2), a)


@numba.njit(cache=True, parallel=True)
def _contour_kernel(px, lum, table, scale, col):  # pragma: no cover - compiled
    h, w, _ = px.shape
    out = px.copy()
    for y in numba.prange(1, h - 1):
        for x in range(1, w - 1):
            nw = lum[y - 1, x - 1]
            ne = lum[y - 1, x]
            sw = lum[y, x - 1]
            se = lum[y, x]
            if (table[nw] != table[ne] or table[nw] != table[sw] or table[ne] != table[se]
                    or table[sw] != table[se]):
                total = abs(nw - ne) + abs(nw - sw) + abs(ne - se) + abs(sw - se)
                strength = np.int64(scale * np.float32(total))
                if strength != 0:
                    st = min(strength, 255)
                    sa = np.int64(px[y, x, 3]) * st // 255
                    ra = (255 - sa) * col[3] // 255
                    for c in range(3):
                        out[y, x, c] = min(max((np.int64(px[y, x, c]) * sa + col[c] * ra) // 255, 0), 255)
                    out[y, x, 3] = min(max(sa + ra, 0), 255)
    return out


def contour(img: np.ndarray, levels: float, scale: float, offset: float, color) -> np.ndarray:
    """``ContourFilter`` (Rust ``contour_exact``): source composited over the contour colour with
    the local brightness-difference strength as extra alpha, where the level table changes."""
    f32 = np.float32
    rgb, a = split_rgba(img)
    h, w = rgb.shape[:2]
    px = np.empty((h, w, 4), np.int32)
    px[..., :3] = to_u8(rgb)
    px[..., 3] = to_u8(a) if a is not None else 255
    levels32 = f32(levels)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        offset_level = int(rs_i32(f32(offset) * f32(256.0) / levels32))
        v = wrap_i32(np.arange(256, dtype=np.int64) + offset_level).astype(f32)
        t = np.floor(levels32 * v / f32(256.0)).astype(f32)
        t = (f32(255.0) * t / (levels32 - f32(1.0)) - f32(offset_level)).astype(f32)
    table = np.clip(rs_i32(t), 0, 255).astype(np.int32)
    lum = _brightness8(px).astype(np.int32)
    col = np.array([round(float(c) * 255.0) for c in color[:4]], np.int64)
    out8 = _contour_kernel(px, lum, table, f32(scale), col)
    res = from8(out8)
    return res if a is not None or (out8[..., 3] < 255).any() else res[..., :3]


@numba.njit(cache=True)
def _java_random_floats(seed, n):  # pragma: no cover - compiled
    mult = np.uint64(0x5DEECE66D)
    mask = np.uint64((1 << 48) - 1)
    s = (np.uint64(seed) ^ mult) & mask
    out = np.empty(n, np.float32)
    for i in range(n):
        s = (s * mult + np.uint64(0xB)) & mask
        out[i] = np.float32(s >> np.uint64(24)) / np.float32(16777216.0)
    return out


def java_random_next_int(seed: int, bound: int, count: int) -> list[int]:
    """``java.util.Random(seed).nextInt(bound)`` x count (bound a power of two)."""
    mask = (1 << 48) - 1
    s = (seed ^ 0x5DEECE66D) & mask
    out = []
    for _ in range(count):
        s = (s * 0x5DEECE66D + 0xB) & mask
        out.append((bound * (s >> 17)) >> 31)
    return out


def dissolve(img: np.ndarray, density: float, softness: float) -> np.ndarray:
    """``DissolveFilter``: alpha *= smoothstep(min, max, rand) with java.util.Random(0)."""
    f32 = np.float32
    rgb, a = split_rgba(img)
    h, w = rgb.shape[:2]
    a8 = to8(a) if a is not None else np.full((h, w), 255, np.int64)
    maximum = (f32(1.0) - f32(density)) * (f32(1.0) + f32(softness))
    minimum = maximum - f32(softness)
    r = _java_random_floats(0, h * w).reshape(h, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((r - minimum) / (maximum - minimum)).astype(f32)
    factor = np.where(r < minimum, f32(0), np.where(r >= maximum, f32(1), t * t * (f32(3) - f32(2) * t)))
    na = np.clip(rs_i32(a8.astype(f32) * factor.astype(f32)), 0, 255)
    return np.dstack([rgb, from8(na)]).astype(F32)


def _premultiply8(px: np.ndarray) -> np.ndarray:
    """jhlabs ``premultiply`` on (H, W, 4) uint8 RGBA (truncating)."""
    a = px[..., 3:4]
    if (a == 255).all():
        return px
    pm = (px[..., :3] * (a.astype(np.float32) / np.float32(255.0))).astype(np.uint8)
    out = px.copy()
    out[..., :3] = np.where(a != 255, pm, px[..., :3])
    return out


def _unpremultiply8(px: np.ndarray) -> np.ndarray:
    """jhlabs ``unpremultiply`` on (H, W, 4) uint8 RGBA."""
    a = px[..., 3:4]
    sel = (a != 0) & (a != 255)
    if not sel.any():
        return px
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.clip(rs_i32(px[..., :3] * (np.float32(255.0) / a.astype(np.float32))), 0, 255)
    out = px.copy()
    out[..., :3] = np.where(sel, up, px[..., :3])
    return out


@numba.njit(cache=True, parallel=True)
def _box_h(src, r):  # pragma: no cover - compiled
    """jhlabs ``blur`` along rows: clamped (2r+1) window sum with integer division."""
    h, w, nc = src.shape
    out = np.empty_like(src)
    ts = 2 * r + 1
    for y in numba.prange(h):
        for c in range(nc):
            tot = 0
            for k in range(-r, r + 1):
                tot += np.int64(src[y, min(max(k, 0), w - 1), c])
            for x in range(w):
                out[y, x, c] = tot // ts
                tot += np.int64(src[y, min(x + r + 1, w - 1), c]) - np.int64(src[y, max(x - r, 0), c])
    return out


@numba.njit(cache=True, parallel=True)
def _box_v(src, r):  # pragma: no cover - compiled
    """Same pass along columns (processed in cache-friendly column blocks)."""
    h, w, nc = src.shape
    out = np.empty_like(src)
    ts = 2 * r + 1
    bw = 64
    nblk = (w + bw - 1) // bw
    for blk in numba.prange(nblk):
        x0 = blk * bw
        x1 = min(w, x0 + bw)
        tot = np.zeros((x1 - x0, nc), np.int64)
        for k in range(-r, r + 1):
            yy = min(max(k, 0), h - 1)
            for x in range(x0, x1):
                for c in range(nc):
                    tot[x - x0, c] += src[yy, x, c]
        for y in range(h):
            ya = min(y + r + 1, h - 1)
            yr = max(y - r, 0)
            for x in range(x0, x1):
                for c in range(nc):
                    out[y, x, c] = tot[x - x0, c] // ts
                    tot[x - x0, c] += np.int64(src[ya, x, c]) - np.int64(src[yr, x, c])
    return out


@numba.njit(cache=True)
def _frac_px(a, b, mid, fraction, factor):  # pragma: no cover - compiled
    nb = np.int64(np.float32(a + b) * fraction)
    v = np.float32(mid + nb) * factor
    return np.uint8(min(max(v, np.float32(0.0)), np.float32(255.0)))


@numba.njit(cache=True, parallel=True)
def _frac_h(src, fraction, factor):  # pragma: no cover - compiled
    """jhlabs ``blurFractional`` along rows incl. its edge quirk: upstream copies ``in[0]`` and
    ``in[width-1]`` (pixels of the *first* row) into the first/last column of every row."""
    h, w, nc = src.shape
    out = np.empty_like(src)
    for y in numba.prange(h):
        for c in range(nc):
            out[y, 0, c] = src[0, 0, c]
            if w > 1:
                out[y, w - 1, c] = src[0, w - 1, c]
            for x in range(1, w - 1):
                out[y, x, c] = _frac_px(np.int64(src[y, x - 1, c]), np.int64(src[y, x + 1, c]),
                                        np.int64(src[y, x, c]), fraction, factor)
    return out


@numba.njit(cache=True, parallel=True)
def _frac_v(src, fraction, factor):  # pragma: no cover - compiled
    """Same along columns (upstream runs it on the transposed image, so the quirk copies the
    top-left pixel to the first row and the *bottom-left* pixel to the last row)."""
    h, w, nc = src.shape
    out = np.empty_like(src)
    for x in numba.prange(w):
        for c in range(nc):
            out[0, x, c] = src[0, 0, c]
            if h > 1:
                out[h - 1, x, c] = src[h - 1, 0, c]
    for y in numba.prange(1, h - 1):
        for x in range(w):
            for c in range(nc):
                out[y, x, c] = _frac_px(np.int64(src[y - 1, x, c]), np.int64(src[y + 1, x, c]),
                                        np.int64(src[y, x, c]), fraction, factor)
    return out


def _box_blur_exact(px: np.ndarray, radius: float) -> np.ndarray:
    """jhlabs ``BoxBlurFilter`` (3 iterations, premultiplied) on (H, W, 4) uint8 RGBA."""
    f32 = np.float32
    r = int(radius)
    fraction = f32(radius) - f32(r)
    factor = f32(1.0) / (f32(1.0) + f32(2.0) * fraction)
    x = _premultiply8(px)
    if r > 0:
        for _ in range(3):
            x = _box_v(_box_h(x, r), r)
    x = _frac_v(_frac_h(x, fraction, factor), fraction, factor)
    return _unpremultiply8(x)


@numba.njit(cache=True, parallel=True)
def _dog_combine(first, second, normalize):  # pragma: no cover - compiled
    h, w, _ = first.shape
    ch = np.empty((h, w, 3), np.int32)
    na = np.empty((h, w), np.uint8)
    for y in numba.prange(h):
        for x in range(w):
            amount = np.float32(first[y, x, 3]) / np.float32(255.0)
            inverse = np.float32(1.0) - amount
            for c in range(3):
                d = np.int64(second[y, x, c])
                delta = max(d - np.int64(first[y, x, c]), 0)
                v = amount * np.float32(delta) + inverse * np.float32(d)
                ch[y, x, c] = np.int32(min(max(v, np.float32(0.0)), np.float32(255.0)))
            va = np.float32(first[y, x, 3]) + np.float32(second[y, x, 3]) * inverse
            na[y, x] = np.uint8(min(max(va, np.float32(0.0)), np.float32(255.0)))
    if normalize:
        mx = 0
        for y in range(h):
            for x in range(w):
                for c in range(3):
                    mx = max(mx, ch[y, x, c])
        if mx != 0:
            for y in numba.prange(h):
                for x in range(w):
                    for c in range(3):
                        ch[y, x, c] = ch[y, x, c] * 255 // mx
    return ch, na


def dog(img: np.ndarray, radius1: float, radius2: float) -> np.ndarray:
    """``DoGFilter``: box-blur(radius2) minus box-blur(radius1), normalised."""
    rgb, a = split_rgba(img)
    h, w = rgb.shape[:2]
    px = np.empty((h, w, 4), np.uint8)
    px[..., :3] = to_u8(rgb)
    px[..., 3] = to_u8(a) if a is not None else 255
    first = _box_blur_exact(px, radius1)
    second = _box_blur_exact(px, radius2)
    ch, na = _dog_combine(first, second, radius1 != radius2)
    if a is None:
        return from8(ch)
    return np.dstack([from8(ch), from8(na)]).astype(F32)


# ---------------------------------------------------------------------------
# aire convolution (libblur filter_2d, Q15 fixed point for u8)
# ---------------------------------------------------------------------------

def convolve3x3_q15(img: np.ndarray, kernel, *, alpha: bool) -> np.ndarray:
    """libblur ``filter_2d_rgb(a)`` for u8 images with a 3x3 f32 kernel.

    Weights are quantised to Q15 and saturated to the i16 range (so weights
    outside about [-1, 1] clip), the integer sum is shifted back with rounding
    and saturated to 0..255.  Correlation (no kernel flip), REFLECT_101 border.
    """
    k = np.asarray(kernel, np.float64).reshape(3, 3)
    kq = np.clip(np.round(k * 32767.0), -32768, 32767).astype(np.int64)
    ch = 4 if (alpha and img.shape[2] == 4) else 3
    px = to_u8(img[..., :ch])
    h, w = px.shape[:2]
    p = np.pad(px, ((1, 1), (1, 1), (0, 0)), mode="reflect").astype(np.int32)  # reflect == REFLECT_101
    acc = np.full((h, w, ch), 16384, np.int32)  # |sum| <= 9 * 32768 * 255 fits int32
    for dy in range(3):
        for dx in range(3):
            if kq[dy, dx]:
                acc += int(kq[dy, dx]) * p[dy:dy + h, dx:dx + w]
    out = from8(np.clip(acc >> 15, 0, 255))
    if ch == 3 and img.shape[2] == 4:
        out = np.dstack([out, img[..., 3]])
    return out.astype(F32)


def emboss_kernel(intensity: float) -> list[float]:
    """aire ``BasePipelinesImpl.emboss`` matrix."""
    i = float(np.float32(intensity))
    return [i * -2, -i, 0.0, -i, 1.0, i, 0.0, i, i * 2]


def grain(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """aire ``grain``: one N(0, 127*intensity) sample per pixel, truncated, added to R, G and B."""
    sigma = np.float32(127.0 * float(intensity))
    if sigma <= 0:
        return from8(to_u8(img))
    noise = np.clip(np.trunc(rng.standard_normal(img.shape[:2], dtype=np.float32) * sigma), -1024, 1024)
    noise = noise.astype(np.int16)
    return from8(np.clip(to_u8(img).astype(np.int16) + noise[..., None], 0, 255))


def threshold_red(img: np.ndarray, level: int) -> np.ndarray:
    """aire ``thresholdPipeline``: thresholds the *red* channel (> level) and writes it to R, G and B."""
    r = to8(img[..., 0])
    g = np.where(r > int(level), 1.0, 0.0).astype(F32)
    return np.repeat(g[..., None], 3, axis=2)


def trickle_noise(img: np.ndarray, threshold: int, rng: np.random.Generator) -> np.ndarray:
    """Trickle ``noise``: every channel ``value | (rand() % threshold)``."""
    if threshold <= 0:  # upstream divides by zero; treat as no-op
        return img.astype(F32, copy=True)
    rgb8 = to_u8(img)
    noise = rng.integers(0, int(threshold), size=rgb8.shape, dtype=np.uint8)
    return from8(rgb8 | noise)


# ---------------------------------------------------------------------------
# GPUImage
# ---------------------------------------------------------------------------

def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """out(x, y) = img(x + dx, y + dy), bilinear, clamp-to-edge (GL_LINEAR + CLAMP_TO_EDGE)."""
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def sobel_edges(img: np.ndarray, line: float) -> np.ndarray:
    """GPUImageSobelEdgeDetectionFilter: grayscale pass (RGBA8 FBO) + 3x3 Sobel magnitude at +-line px."""
    lum = img[..., :3] @ np.array([0.2125, 0.7154, 0.0721], np.float32)
    lum = from8(to8(lum))  # the grayscale pass is rendered into an 8-bit framebuffer
    s = float(line)
    tl, t, tr = _shift(lum, -s, -s), _shift(lum, 0, -s), _shift(lum, s, -s)
    lft, rgt = _shift(lum, -s, 0), _shift(lum, s, 0)
    bl, b, br = _shift(lum, -s, s), _shift(lum, 0, s), _shift(lum, s, s)
    hh = -tl - 2.0 * t - tr + bl + 2.0 * b + br
    vv = -bl - 2.0 * lft - tl + br + 2.0 * rgt + tr
    mag = np.clip(np.sqrt(hh * hh + vv * vv), 0, 1).astype(F32)
    return np.repeat(mag[..., None], 3, axis=2)


def gpu_lookup(img: np.ndarray, lut: np.ndarray | None, intensity: float) -> np.ndarray:
    """GPUImageLookupFilter: mix(color, lut(color), intensity).

    ImageToolbox never binds a lookup texture to this filter, so the shader
    samples an incomplete texture, which OpenGL ES defines as (0, 0, 0, 1):
    without a LUT file the result is ``color * (1 - intensity)``.
    """
    rgb = np.clip(img[..., :3], 0, 1).astype(F32)
    new = apply_lut3d(rgb, lut) if lut is not None else np.zeros_like(rgb)
    return (rgb + (new - rgb) * np.float32(intensity)).astype(F32)


# ---------------------------------------------------------------------------
# OpenCV (ImageToolbox lib/opencv-tools)
# ---------------------------------------------------------------------------

def canny(img: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """``ImageProcessing.canny``: RGBA2GRAY, 3x3 median, Canny(t1, t2)."""
    gray = cv2.cvtColor(to_u8(img[..., :3]), cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 3)
    edges = cv2.Canny(gray, float(t1), float(t2))
    return np.repeat(from8(edges)[..., None], 3, axis=2)


def error_level_analysis(img: np.ndarray, quality: int) -> np.ndarray:
    """``ImageForensics.errorLevelAnalysis``: |img - jpeg(img, q)| min-max normalised."""
    bgr = np.ascontiguousarray(to_u8(img[..., :3])[..., ::-1])
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:  # pragma: no cover
        raise LazyRoomError(ErrorCode.OPERATION_FAILED, "JPEG re-encoding failed")
    resaved = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(bgr, resaved)
    dst = cv2.normalize(diff, None, 0.0, 255.0, cv2.NORM_MINMAX)
    return from8(dst[..., ::-1])


def _dct_matrix(n: int = 16) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    c = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    c[0] = np.sqrt(1.0 / n)
    return c


COPY_MOVE_MEMORY_LIMIT = 768 * 1024 * 1024


def detect_copy_move(img: np.ndarray, retain: int, q: float, ctx=None) -> np.ndarray:
    """``ImageForensics.detectCopyMove`` (16x16 block DCT matching, shift-vector voting)."""
    bs = 16
    rgb8 = to_u8(img[..., :3])
    h, w = rgb8.shape[:2]
    bh, bw = h - bs + 1, w - bs + 1
    if bh <= 1 or bw <= 1:
        return img[..., :3].astype(F32, copy=True)
    n = bh * bw
    retain = max(1, min(int(retain), bs))
    if n * retain * retain > COPY_MOVE_MEMORY_LIMIT:
        raise LazyRoomError(ErrorCode.RESOURCE_LIMIT,
                              f"Copy-move detection would need {n * retain * retain / 2**20:.0f} MiB",
                              hint="Lower 'retain' or run on a smaller image.")
    # upstream converts the RGBA mat with COLOR_BGR2GRAY, i.e. with swapped R/B weights
    gray = cv2.cvtColor(rgb8, cv2.COLOR_BGR2GRAY).astype(np.float64)
    cm = _dct_matrix(bs)
    planes = np.empty((retain * retain, n), np.uint8)
    for v in range(retain):
        rowpass = cv2.filter2D(gray, cv2.CV_64F, cm[v][None, :], anchor=(0, 0),
                               borderType=cv2.BORDER_CONSTANT)
        for u in range(retain):
            coef = cv2.filter2D(rowpass, cv2.CV_64F, cm[u][:, None], anchor=(0, 0),
                                borderType=cv2.BORDER_CONSTANT)[:bh, :bw]
            coef[np.abs(coef) < 1e-6] = 0.0  # cv::dct yields exact zeros for flat/separable blocks
            if q > 0:  # convertTo(CV_8U): round to nearest, saturate
                val = np.clip(np.rint(coef / q), 0, 255)
            else:  # x/0 -> +-inf/NaN, saturated by ARM float->int conversion: 255 if > 0 else 0
                val = (coef > 0) * 255
            planes[u * retain + v] = val.reshape(-1)
        if ctx is not None:
            ctx.check()
    sig = planes.T
    # java's stable sort over signed bytes == lexicographic order of (byte ^ 0x80); pack the
    # signature into big-endian uint64 words and lexsort them (stable)
    nb = sig.shape[1]
    nwords = (nb + 7) // 8
    keyed = np.zeros((n, nwords * 8), np.uint8)
    keyed[:, :nb] = sig ^ np.uint8(0x80)
    del sig
    words = keyed.view(">u8").astype(np.uint64)
    del keyed
    order = np.argsort(words[:, nwords - 1], kind="stable")  # LSD: stable sorts, least significant first
    for j in range(nwords - 2, -1, -1):
        order = order[np.argsort(words[order, j], kind="stable")]
    sw = words[order]
    eq = np.all(sw[1:] == sw[:-1], axis=1)
    del words, sw
    a_idx = order[:-1][eq]
    b_idx = order[1:][eq]
    cx, cy = a_idx % bw, a_idx // bw
    nx, ny = b_idx % bw, b_idx // bw
    sx, sy = np.abs(cx - nx), np.abs(cy - ny)
    mag = np.hypot(sx.astype(np.float64), sy.astype(np.float64))
    key = sy * w + sx
    counts = np.bincount(key[mag > bs], minlength=int(key.max()) + 1 if key.size else 1)
    sel = counts[key] > 10 if key.size else np.zeros(0, bool)
    buf = rgb8.copy()
    if sel.any():
        # Upstream paints both 16x16 blocks of every selected pair in order; the final colour of a pixel
        # is the one of the last write covering it -> max "write time" over the covering block origins.
        uniq, inv = np.unique(np.trunc(mag[sel]).astype(np.int64), return_inverse=True)
        colors = np.array([java_random_next_int(int(m), 256, 3) for m in uniq.tolist()], np.uint8)[inv]
        t = np.arange(colors.shape[0], dtype=np.int64) * 2
        tmax = np.full(h * w, -1.0, np.float64)
        np.maximum.at(tmax, (cy[sel] * w + cx[sel]).astype(np.int64), t.astype(np.float64))
        np.maximum.at(tmax, (ny[sel] * w + nx[sel]).astype(np.int64), (t + 1).astype(np.float64))
        last = cv2.dilate(tmax.reshape(h, w), np.ones((bs, bs), np.uint8), anchor=(bs - 1, bs - 1),
                          borderType=cv2.BORDER_CONSTANT, borderValue=-1.0)
        hit = last >= 0
        buf[hit] = colors[(last[hit] // 2).astype(np.int64)]
    out = np.clip(np.rint(0.2 * rgb8.astype(np.float64) + 0.8 * buf.astype(np.float64)), 0, 255)
    return from8(out)


# ---------------------------------------------------------------------------
# G'MIC helpers
# ---------------------------------------------------------------------------
# Some G'MIC filters (fx_cracks, fx_filaments, fx_ghost) draw random numbers in
# multi-threaded math expressions, so their output depends on OpenMP
# scheduling even after ``srand``.  They are run with OMP_NUM_THREADS=1
# (reference counted; the variable is only read by the child at spawn time).
# NOTE: same mechanism as _glitch_impl.py -- a shared helper in _common would
# avoid two independent reference counts.



from ._common import gmic_single_thread  # noqa: E402  (shared thread-local switch)


def with_power_of_two_padding(command: str) -> str:
    """ImageToolboxLibs ``withPowerOfTwoPadding`` (GmicCommon.kt)."""
    return ("_gmic_android_w={w} _gmic_android_h={h} "
            "resize {2^ceil(log2(max(1,w)))},{2^ceil(log2(max(1,h)))},100%,100%,0,3,0.5,0.5 "
            + command +
            " crop {int((w-$_gmic_android_w)/2)},{int((h-$_gmic_android_h)/2)},"
            "{int((w-$_gmic_android_w)/2+$_gmic_android_w-1)},"
            "{int((h-$_gmic_android_h)/2+$_gmic_android_h-1)}")


# --- Raw G'MIC safety policy ------------------------------------------------

#: Built-in commands / shortcuts that execute programs, read or write files,
#: access the network, open windows or load command definitions.
GMIC_DENY_COMMANDS = frozenset("""
exec x xo exec_out shell run
input i in ib ig it input_565 input_bytes input_cached input_csv input_cube input_flo input_glob input_gpl
input_normalized input_obj input_text
output o on op ow ox ot output_565 output_csv output_cube output_flo output_ggr output_gmz output_obj
output_text outputn outputp outputw outputx
command m uncommand um download network update up camera cam delete createdir cursor screen select wait
window w w0 w1 w2 w3 display d d0 da dc dclut dfft dg dh dq drgba dt dv3d dw d3d display0 display3d
sample sp lorem echo_file files filename font2gmz tiff3d toff3d cimg2tiff demos xz
apply_camera apply_files apply_video average_files average_video fade_files fade_video files2video
median_files median_video morph_files morph_video video2files
parse_cli parse_gmd parse_gui gmd2html gmd2ascii
""".split())

#: Command-name prefixes rejected outright.
GMIC_DENY_PREFIXES = ("input", "output", "display", "exec", "x_", "file", "path_", "update", "upload",
                      "download", "camera", "window", "pink_input", "gmd2", "parse_", "reference")

#: G'MIC 4.0.2 stdlib commands that (directly or through the commands they
#: call) execute programs or write/delete files.  Derived by a call-graph scan
#: of stdlib.gmic seeded with exec/output*/delete/createdir/echo_file/camera/
#: screen (network access is additionally disabled with ``network -1``).
GMIC_DENY_STDLIB = frozenset("""
__pal_rename_file _apply_stream _average_stream _display _display_camera _display_voxels3d _exec_out
_fade_stream _gmd2html_formula _gmd2html_pipeline _gui_download_all_data _input_bytes _input_glob
_morph_stream _output_text _outputn _outputp _outputw _outputx _plot _rep_rde_validate_frequency _update
_update_header_html _wrbytepnk _x_paint _xpink _xpinks afre_compare animate blend_fft blend_sharpness
cli_noarg compress_cluts compress_palette compress_pdf font2cimgh fouriermix fx_SimpleNoiseCanvas
fx_animate_cartoon fx_animate_edges fx_animate_elevation3d fx_animate_extrude3d fx_animate_glow
fx_animate_imageobject3d fx_animate_lissajous fx_animate_pencilbw fx_animate_rodilius fx_animate_stencilbw
fx_apply_Bcurve_preview fx_apply_Cbcurve_preview fx_apply_Crcurve_preview fx_apply_Gcurve_preview
fx_apply_Labcurve_preview fx_apply_Lcurve_preview fx_apply_RGBcurve_preview fx_apply_Rcurve_preview
fx_apply_YCbCrcurve_preview fx_apply_Ycurve_preview fx_apply_acurve_preview fx_apply_bcurve_preview
fx_asciiart fx_asciiart_preview fx_clut_from_ab fx_clut_from_ab_preview fx_custom_gradient
fx_custom_gradient_preview fx_denoise fx_denoise_preview fx_extrude25d fx_extrude25d_preview
fx_generate_random_portrait fx_generate_random_portrait_preview fx_gradient_from_curve fx_intarsia
fx_karo_cimg_nlmeans fx_karo_cimg_skel fx_karo_pink_bianca fx_montage_preview fx_ncee fx_output_565 fx_pack
fx_pack_preview fx_rep_lavander_binary_map_preview fx_rep_photomosaic fx_rep_photomosaic_preview
fx_rep_rpgtiler_noniso fx_scale2x_cnn fx_scale2x_cnn_preview fx_send_feedback fx_tk_video3D
fx_tk_video3D_preview fx_whatsnew_preview gcd_browse gcd_fmean_local gcd_stereo_vid gui_download_all_data
gui_layer_info gui_layer_info_preview gui_rep_recc_preview gui_rep_regm_preview iain_savenoiseprint
iain_savenoiseprint_p img2ascii is_bwclut karo_nlmeans karo_skeleton karo_xcimg karo_zerleg kr_is_exec
lezoray3d mm_mozaic nn_save ovh2stats parse_cli_html parse_cli_images parse_gui_images
parse_gui_thumbnails pink pink_asflin pink_delaunay pink_grayskel pink_heightmaxima pink_heightminima
pink_htkern pink_karo_zerleg pink_lvkern pink_new pink_reg_maxima pink_reg_minima pink_setorigin
pink_skelcurv pink_skelend pink_skeleton pink_skelpar pink_wshed render_donations rep_ncee rep_output_gpl
rep_output_jasc rep_output_pdn_pal rep_permutation_list2index rep_rde_compsel_perm_list2index
rep_sum_combination_list2index rep_total_sum_permutation_list2index rep_txt2uints
samj_GUI_Test_Liste_Points_TSP samj_Layers_To_PNG samj_SVG_Symetries_A samj_Test_Liste_Points_TSP
samj_Test_Liste_Points_TSP_Bezier samj_Valeur_Moyenne_LCH samj_Valeur_Moyenne_LCH_CSV
samj_fx_spatial_transition_to_PNG samj_test_paint_gmic_gimp samj_transition_H_V_to_PNG shell_cols
update_color_presets_html update_deprecate update_download_html update_fonts update_gallery_html
update_gmicol update_instimg update_reference_html update_tutorial_html upload_filters x_jawbreaker x_life
x_minimal_path x_tixy
""".split())

#: Commands after which a bare ``x`` is an axis argument (``a x``, ``mirror x``...).
_GMIC_AXIS_COMMANDS = frozenset({"a", "append", "mirror", "s", "split"})
_GMIC_ITEM_RE = re.compile(r"^[+-]?([A-Za-z_][A-Za-z0-9_]*)(\[[^\]]*\]|\.+)?$")
_GMIC_MATH_FUNC_RE = re.compile(r"\b(run|ext|fsize|isfile|isdir|date|files|display|set)\s*\(")
_GMIC_PATH_RES = (
    re.compile(r"://"),                                   # URL
    re.compile(r"^(https?|ftp|file|www)\b", re.I),        # URL-ish
    re.compile(r"^[/\\~]"),                               # absolute / home path
    re.compile(r"^[A-Za-z]:"),                            # drive letter or [type:]filename prefix
    re.compile(r"\.\.[/\\]"),                             # parent directory
    re.compile(r"[/\\][A-Za-z0-9_.-]+[/\\]"),             # dir/sub/
    re.compile(r"^[A-Za-z_][\w-]*(\.[\w-]+)*\.[A-Za-z][A-Za-z0-9]*$"),  # name.ext
)
_GMIC_ALLOWED_DOLLAR = set("><!%^|")


def _gmic_reject(reason: str, token: str | None = None) -> LazyRoomError:
    msg = f"G'MIC command rejected by the safety policy: {reason}"
    if token:
        msg += f" ({token!r})"
    return LazyRoomError(
        ErrorCode.INVALID_PARAMS, msg, details={"param": "command", "token": token},
        hint="Raw G'MIC commands may only process the current image: commands that execute programs, "
             "read/write files, open windows, load command files or access the network are rejected, as are "
             "file paths, URLs, '$variable' substitutions and string/text substitutions. "
             "Use plain filter commands such as 'blur 3' or 'fx_sharpen 50,0,0'.")


def _gmic_command_name(token: str) -> str | None:
    m = _GMIC_ITEM_RE.match(token)
    return m.group(1) if m else None


def _gmic_is_denied(name: str) -> bool:
    return (name in GMIC_DENY_COMMANDS or name in GMIC_DENY_STDLIB
            or any(name.startswith(p) for p in GMIC_DENY_PREFIXES))


def check_raw_gmic(command: str) -> str:
    """Validate a user/agent supplied G'MIC pipeline; returns the stripped command.

    The check is deliberately conservative (it may reject some harmless
    pipelines): quoting is ignored, so words inside strings and sub-pipelines
    (``parallel "..."``, ``repeat ... { }``) are checked too.
    """
    cmd = (command or "").strip()
    if not cmd:
        raise _gmic_reject("empty command")
    bad = sorted({c for c in cmd if not (32 <= ord(c) <= 126 or c in "\t\r\n")})
    if bad:
        raise _gmic_reject("non-printable or non-ASCII characters", "".join(bad))
    for ch, why in (("\\", "backslash escapes"), ("'", "single-quoted strings"), ("`", "backquote substitution")):
        if ch in cmd:
            raise _gmic_reject(why, ch)
    for i, ch in enumerate(cmd):
        if ch == "$" and (i + 1 >= len(cmd) or cmd[i + 1] not in _GMIC_ALLOWED_DOLLAR):
            raise _gmic_reject("'$' substitutions (variables, environment, pipelines) are not allowed",
                               cmd[i:i + 12])
    m = _GMIC_MATH_FUNC_RE.search(cmd)
    if m:
        raise _gmic_reject("math function with side effects", m.group(0))
    # {...} substitutions: only numeric math is allowed (no strings, image text/name features, display)
    for m in re.finditer(r"\{([^{}]*)\}", cmd):
        inner = m.group(1).strip()
        if inner[:1] in ("*", "/", "^", "@"):
            raise _gmic_reject("string/display substitution", m.group(0))
        last = inner.rsplit(",", 1)[-1].split(":", 1)[0].strip()
        if "," in inner and (last in {"b", "f", "n", "t", "x", "^"} or last.startswith("@")):
            raise _gmic_reject("image name/text substitution", m.group(0))
    tokens = [t for t in re.split(r"[\s\"]+", cmd) if t]
    axis_args: set[int] = set()
    for k, tok in enumerate(tokens):
        name = _gmic_command_name(tok)
        if name is not None:
            if tok == "x" and k >= 1 and _gmic_command_name(tokens[k - 1]) in _GMIC_AXIS_COMMANDS:
                # ``a x`` / ``mirror x`` / ``split x``: only when the axis command itself is in command
                # position, i.e. not the argument of a preceding command (``nm a x ...`` would exec).
                if k == 1 or _gmic_command_name(tokens[k - 2]) is None or (k - 2) in axis_args:
                    axis_args.add(k)
                    continue
            if _gmic_is_denied(name):
                raise _gmic_reject("command not allowed", tok)
        for part in re.split(r"[,;(){}\[\]=]+", tok.lstrip("+-")):
            if part and any(rx.search(part) for rx in _GMIC_PATH_RES):
                raise _gmic_reject("file paths and URLs are not allowed", part)
    return cmd


# ---------------------------------------------------------------------------
# Custom GPUImage shader (ImageToolbox ShaderFilter) via headless Qt OpenGL
# ---------------------------------------------------------------------------

SHADER_SUPPORTED_VERSION = 1
SHADER_RESERVED = {"inputImageTexture", "textureCoordinate", "inputImageSize", "outputImageSize"}
SHADER_UNIFORM_TYPES = {"float": "float", "int": "int", "bool": "bool", "color": "vec4", "vec2": "vec2"}
_GLSL_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _strip_glsl_comments(src: str) -> str:
    src = re.sub(r"/\*.*?(\*/|$)", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def _glsl_has_ident(src: str, ident: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(ident) + r"(?![A-Za-z0-9_])", src) is not None


def _glsl_declarations(src: str, keyword: str) -> dict[str, str]:
    """``ShaderValidator.readDeclaration`` over ';'-separated statements."""
    out: dict[str, str] = {}
    for stmt in src.split(";"):
        toks = stmt.split()
        if not toks or toks[0] != keyword:
            continue
        i = 1
        if len(toks) > i and toks[i] in ("lowp", "mediump", "highp"):
            i += 1
        if len(toks) <= i:
            continue
        typ = toks[i]
        for part in " ".join(toks[i + 1:]).split(","):
            mm = re.match(r"^[A-Za-z_][A-Za-z0-9_]*", part.strip())
            if mm:
                out[mm.group(0)] = typ
    return out


def _parse_shader_value(typ: str, value, path: str, errors: list[str]):
    try:
        if typ == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError
            return float(value)
        if typ == "int":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) % 1:
                raise ValueError
            return int(value)
        if typ == "bool":
            if not isinstance(value, bool):
                raise ValueError
            return value
        if typ == "vec2":
            if isinstance(value, dict):
                value = [value.get("x"), value.get("y")]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError
            return [_parse_shader_value("float", v, path, errors) for v in value]
        if typ == "color":
            if isinstance(value, str):
                hx = value.lstrip("#")
                if len(hx) not in (6, 8):
                    raise ValueError
                ch = [int(hx[i:i + 2], 16) for i in range(0, len(hx), 2)]
            elif isinstance(value, dict):
                ch = [value.get("r", value.get("red")), value.get("g", value.get("green")),
                      value.get("b", value.get("blue")), value.get("a", value.get("alpha", 255))]
            elif isinstance(value, (list, tuple)) and len(value) in (3, 4):
                ch = list(value)
            else:
                raise ValueError
            ch = [int(v) for v in ch] + ([255] if len(ch) == 3 else [])
            if any(not 0 <= v <= 255 for v in ch):
                raise ValueError
            return ch
    except (TypeError, ValueError):
        pass
    errors.append(f"{path} is not a valid '{typ}' value")
    return None


def parse_shader(shader: str, values_json: str) -> tuple[str, list[dict], dict]:
    """Return (glsl, params, values) from GLSL source or an ``.itshader`` JSON preset.

    Validates like ImageToolbox's ShaderParser/ShaderValidator and raises
    INVALID_PARAMS listing every problem (ImageToolbox silently returns the
    input image instead).
    """
    errors: list[str] = []
    text = shader.strip()
    params: list[dict] = []
    if text.startswith("{") and text.endswith("}"):
        try:
            preset = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"Shader preset is not valid JSON: {exc}",
                                  details={"param": "shader"}) from None
        for f in ("version", "name", "shader"):
            if f not in preset:
                errors.append(f"Field '{f}' is required.")
        if preset.get("version", SHADER_SUPPORTED_VERSION) != SHADER_SUPPORTED_VERSION:
            errors.append(f"Unsupported shader version {preset.get('version')}. Supported version: 1.")
        glsl = str(preset.get("shader") or "")
        for idx, p in enumerate(preset.get("params") or []):
            name = str(p.get("name", "")).strip()
            typ = str(p.get("type", "")).strip().lower()
            if typ not in SHADER_UNIFORM_TYPES:
                errors.append(f"params[{idx}].type '{typ}' is not supported.")
                continue
            if not name or not _GLSL_IDENT.match(name):
                errors.append(f"Parameter '{name}' must be a valid GLSL identifier.")
            if name in SHADER_RESERVED:
                errors.append(f"Parameter '{name}' uses a reserved GPUImage name.")
            default = _parse_shader_value(typ, p.get("default"), f"params[{idx}].default", errors)
            params.append({"name": name, "type": typ, "default": default})
        names = [p["name"] for p in params]
        for dup in sorted({n for n in names if names.count(n) > 1}):
            errors.append(f"Parameter '{dup}' is declared more than once.")
    else:
        glsl = shader
    if not glsl.strip():
        errors.append("Shader source must not be blank.")
    bad = sorted({c for c in glsl if not (32 <= ord(c) <= 126 or c in "\n\r\t")})
    if bad:
        errors.append("Shader source contains unsupported characters. Use ASCII GLSL source only.")
    src = _strip_glsl_comments(glsl)
    uniforms = _glsl_declarations(src, "uniform")
    varyings = _glsl_declarations(src, "varying")
    if uniforms.get("inputImageTexture") != "sampler2D":
        errors.append("Shader must declare 'uniform sampler2D inputImageTexture;'.")
    if varyings.get("textureCoordinate") != "vec2":
        errors.append("Shader must declare 'varying vec2 textureCoordinate;'.")
    if not re.search(r"\bvoid\s+main\s*\(\s*\)", src):
        errors.append("Shader must define 'void main()'.")
    if not _glsl_has_ident(src, "gl_FragColor"):
        errors.append("Shader must write a color to gl_FragColor.")
    for ident, msg in (("mainImage", "ShaderToy mainImage shaders are not supported."),
                       ("iTime", "Animated ShaderToy uniform 'iTime' is not supported."),
                       ("iFrame", "Animated ShaderToy uniform 'iFrame' is not supported."),
                       ("iResolution", "ShaderToy uniform 'iResolution' is not supported."),
                       ("samplerExternalOES", "External textures are not supported."),
                       ("GL_OES_EGL_image_external", "External texture extensions are not supported.")):
        if _glsl_has_ident(src, ident):
            errors.append(msg)
    if re.search(r"(?<![A-Za-z0-9_])iChannel", src):
        errors.append("ShaderToy iChannel textures are not supported.")
    for name, typ in uniforms.items():
        if typ in ("sampler2D", "samplerCube", "samplerExternalOES") and name != "inputImageTexture":
            errors.append(f"Only one input texture is supported. Remove 'uniform {typ} {name};'.")
    if not params:  # plain GLSL: expose the declared scalar/vector uniforms as parameters
        rev = {v: k for k, v in SHADER_UNIFORM_TYPES.items()}
        for name, typ in uniforms.items():
            if typ in rev and name not in SHADER_RESERVED:
                params.append({"name": name, "type": rev[typ], "default": None})
    else:
        for p in params:
            ut = uniforms.get(p["name"])
            want = SHADER_UNIFORM_TYPES[p["type"]]
            if ut is None:
                errors.append(f"Parameter '{p['name']}' must be declared as 'uniform {want} {p['name']};'.")
            elif ut != want:
                errors.append(f"Parameter '{p['name']}' is declared as 'uniform {ut} {p['name']};', "
                              f"expected 'uniform {want} {p['name']};'.")
    values: dict = {}
    if values_json and values_json.strip():
        try:
            raw = json.loads(values_json)
        except json.JSONDecodeError as exc:
            raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"'values' is not valid JSON: {exc}",
                                  details={"param": "values"}) from None
        if not isinstance(raw, dict):
            errors.append("'values' must be a JSON object mapping uniform names to values.")
            raw = {}
        known = {p["name"]: p for p in params}
        for k, v in raw.items():
            if k not in known:
                errors.append(f"'values' sets unknown shader parameter '{k}'.")
                continue
            val = _parse_shader_value(known[k]["type"], v, f"values.{k}", errors)
            if val is not None:
                values[k] = val
    if errors:
        raise LazyRoomError(ErrorCode.INVALID_PARAMS, "Invalid shader: " + " ".join(errors),
                              details={"param": "shader", "errors": errors},
                              hint="Write a GLSL ES 1.0 fragment shader with GPUImage's contract: "
                                   "'varying vec2 textureCoordinate;', 'uniform sampler2D inputImageTexture;', "
                                   "'void main()' writing gl_FragColor.")
    return glsl, params, values


def shader_desktop_source(glsl: str) -> str:
    """Translate a GPUImage GLSL ES 1.00 fragment shader to desktop GLSL 1.20.

    Qt defines ``lowp``/``mediump``/``highp`` away on desktop GL, so default
    ``precision`` statements and ES-only ``#version``/``#extension`` lines are
    removed (derivatives are core on desktop).
    """
    src = re.sub(r"^\s*#\s*version[^\n]*", "", glsl, flags=re.M)
    src = re.sub(r"^\s*#\s*extension\s+GL_OES_standard_derivatives[^\n]*", "", src, flags=re.M)
    src = re.sub(r"\bprecision\s+(lowp|mediump|highp)\s+\w+\s*;", "", src)
    return "#version 120\n" + src


_SHADER_WORKER = r'''
import json, os, sys
import numpy as np

def main(d):
    spec = json.load(open(os.path.join(d, "spec.json"), encoding="utf-8"))
    img = np.ascontiguousarray(np.load(os.path.join(d, "in.npy")))
    h, w = img.shape[:2]
    try:
        from PySide6.QtGui import QGuiApplication, QImage, QOffscreenSurface, QOpenGLContext, QSurfaceFormat
        from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat,
                                      QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture)
    except ImportError:
        sys.exit(5)
    app = QGuiApplication.instance() or QGuiApplication(["lazyroom-shader"])
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    ctx = QOpenGLContext()
    ctx.setFormat(fmt)
    if not ctx.create():
        sys.exit(3)
    surf = QOffscreenSurface()
    surf.setFormat(ctx.format())
    surf.create()
    if not surf.isValid() or not ctx.makeCurrent(surf) or ctx.isOpenGLES():
        sys.exit(3)
    gl = ctx.functions()
    prog = QOpenGLShaderProgram()
    ok = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, spec["vertex"])
    ok = ok and prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, spec["fragment"])
    ok = ok and prog.link()
    if not ok:
        with open(os.path.join(d, "log.txt"), "w", encoding="utf-8") as f:
            f.write(prog.log())
        sys.exit(4)
    qimg = QImage(img.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    tex = QOpenGLTexture(qimg, QOpenGLTexture.MipMapGeneration.DontGenerateMipMaps)
    tex.setMinificationFilter(QOpenGLTexture.Filter.Linear)
    tex.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
    tex.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
    ffmt = QOpenGLFramebufferObjectFormat()
    ffmt.setInternalTextureFormat(0x8058)
    fbo = QOpenGLFramebufferObject(w, h, ffmt)
    if not fbo.isValid() or not fbo.bind():
        sys.exit(3)
    quad = np.array([-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1], np.float32)
    vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
    vbo.create()
    vbo.bind()
    vbo.allocate(quad.tobytes(), quad.nbytes)
    gl.glViewport(0, 0, w, h)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glClear(0x4000)
    prog.bind()
    pos = prog.attributeLocation("position")
    tc = prog.attributeLocation("inputTextureCoordinate")
    prog.enableAttributeArray(pos)
    prog.setAttributeBuffer(pos, 0x1406, 0, 2, 16)
    prog.enableAttributeArray(tc)
    prog.setAttributeBuffer(tc, 0x1406, 8, 2, 16)
    gl.glActiveTexture(0x84C0)
    tex.bind(0)
    prog.setUniformValue1i("inputImageTexture", 0)
    for name, (kind, value) in spec["uniforms"].items():
        if kind == "float":
            prog.setUniformValue1f(name, float(value))
        elif kind in ("int", "bool"):
            prog.setUniformValue1i(name, int(value))
        elif kind == "vec2":
            prog.setUniformValue(name, float(value[0]), float(value[1]))
        elif kind == "vec4":
            prog.setUniformValue(name, *(float(v) for v in value))
    gl.glDrawArrays(0x0005, 0, 4)
    gl.glFinish()
    out = fbo.toImage(False)
    raw = np.frombuffer(out.constBits(), np.uint8, count=out.sizeInBytes()).reshape(out.height(), out.bytesPerLine())
    arr = np.ascontiguousarray(raw[:, :4 * w].reshape(h, w, 4))
    if out.format() in (QImage.Format.Format_ARGB32, QImage.Format.Format_ARGB32_Premultiplied,
                        QImage.Format.Format_RGB32):
        arr = arr[..., [2, 1, 0, 3]] if sys.byteorder == "little" else arr[..., [1, 2, 3, 0]]
    np.save(os.path.join(d, "out.npy"), arr)
    fbo.release()
    tex.destroy()
    ctx.doneCurrent()

main(sys.argv[1])
'''

_SHADER_VERTEX = """#version 120
attribute vec4 position;
attribute vec4 inputTextureCoordinate;
varying vec2 textureCoordinate;
void main()
{
    gl_Position = position;
    textureCoordinate = inputTextureCoordinate.xy;
}
"""

SHADER_GL_HINT = ("The Shader filter renders with OpenGL through PySide6 (pip install 'lazyroom[gui]') and needs "
                  "an OpenGL-capable display (X11/Wayland session); headless servers without one are not supported.")


def run_shader(img: np.ndarray, glsl: str, uniforms: dict, timeout: float = 120.0) -> np.ndarray:
    """Render ``glsl`` over ``img`` (H, W, 4 float) in a separate process; returns (H, W, 4) floats."""
    rgba8 = np.ascontiguousarray(to_u8(img))
    frag = shader_desktop_source(glsl)
    platforms = ["offscreen"]
    if os.environ.get("WAYLAND_DISPLAY"):
        platforms.append("wayland")
    if os.environ.get("DISPLAY"):
        platforms.append("xcb")
    with tempfile.TemporaryDirectory(prefix="pm-shader-") as td:
        np.save(os.path.join(td, "in.npy"), rgba8)
        with open(os.path.join(td, "spec.json"), "w", encoding="utf-8") as f:
            json.dump({"vertex": _SHADER_VERTEX, "fragment": frag, "uniforms": uniforms}, f)
        last = None
        for plat in platforms:
            env = dict(os.environ, QT_QPA_PLATFORM=plat)
            try:
                proc = subprocess.run([sys.executable, "-I", "-c", _SHADER_WORKER, td], capture_output=True,
                                      text=True, timeout=timeout, env=env, cwd=td)
            except subprocess.TimeoutExpired:
                raise LazyRoomError(ErrorCode.TIMEOUT, "Shader rendering timed out") from None
            if proc.returncode == 0:
                return from8(np.load(os.path.join(td, "out.npy")))
            if proc.returncode == 4:
                log = open(os.path.join(td, "log.txt"), encoding="utf-8").read()
                lines = [ln for ln in log.splitlines() if "error" in ln.lower()] or log.splitlines()[-3:]
                raise LazyRoomError(ErrorCode.INVALID_PARAMS, "Shader failed to compile: " + " | ".join(lines[:5]),
                                      details={"param": "shader", "log": log[-4000:]})
            if proc.returncode == 5:
                raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, "PySide6 is not installed",
                                      details={"backend": "opengl"}, hint=SHADER_GL_HINT)
            last = (proc.stderr or "").strip().splitlines()[-1:] or [f"exit code {proc.returncode}"]
    raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, "No OpenGL context available for the Shader filter",
                          details={"backend": "opengl", "error": last[0] if last else None}, hint=SHADER_GL_HINT)
