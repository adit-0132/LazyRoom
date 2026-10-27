"""Private implementations for the ImageToolbox "Glitch" group (see ``glitch.py``).

Ports of:
  * Trickle (Apache-2.0) native glitches: ``ProcessingFilters.cpp``
    (glitchVariant, vhsGlitch, blockGlitch, crtCurvature, pixelMelt) and the
    Rust ``ntsc_jni`` bridge to ntsc-rs (MIT/ISC/Apache-2.0, valadaptive);
  * aire (MIT, awxkee): ``shift/Glitch.cpp``, ``shift/WindStagger.cpp``,
    ``color/tone/AldridgeToneMapper.hpp``;
  * ImageToolbox (Apache-2.0): ``utils/glitch/tools/JpegGlitch.kt`` and the
    procedural GLSL shaders of ``utils/gpu/ProceduralShaders.kt``.

Android bitmaps are premultiplied RGBA; the native code operates on those
bytes, so the ports premultiply where channels from different pixels are
mixed.  Unseeded ``rand()``/time-seeded engines upstream are replaced by
``ctx.rng`` draws (deterministic per render seed); explicit seed parameters
keep their upstream meaning.
"""
from __future__ import annotations

import contextlib
import math
import os
import threading

import cv2
import numpy as np
from numba import njit, prange

from .._util import F32
from ._common import gmic_pipeline, run_gmic

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _premul_kernel(src, out, inverse):
    h, w = src.shape[0], src.shape[1]
    for y in prange(h):
        for x in range(w):
            a = src[y, x, 3]
            if inverse:
                k = np.float32(1.0) / a if a > np.float32(1e-6) else np.float32(0.0)
            else:
                k = a
            out[y, x, 0] = src[y, x, 0] * k
            out[y, x, 1] = src[y, x, 1] * k
            out[y, x, 2] = src[y, x, 2] * k
            out[y, x, 3] = a


def premultiply(img: np.ndarray) -> np.ndarray:
    """Return a premultiplied float32 copy (a plain copy when there is no alpha)."""
    if img.shape[2] != 4:
        return np.array(img, dtype=F32, copy=True, order="C")
    src = np.ascontiguousarray(img, dtype=F32)
    out = np.empty_like(src)
    _premul_kernel(src, out, False)
    return out


def unpremultiply(img: np.ndarray) -> np.ndarray:
    """Undo :func:`premultiply` in place (the buffer is owned by the caller)."""
    if img.shape[2] != 4:
        return img
    img = np.ascontiguousarray(img, dtype=F32)
    _premul_kernel(img, img, True)
    return img


def _trunc_int(x) -> int:
    """C/C++ float -> int conversion (truncation toward zero)."""
    return int(math.trunc(float(x)))


def round_half_up(x: float) -> int:
    """Kotlin ``Float.roundToInt()`` (Math.round: half up)."""
    return int(math.floor(float(x) + 0.5))


def round_half_away(x: float) -> int:
    """Rust ``f32::round()`` (half away from zero)."""
    x = float(x)
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


# ---------------------------------------------------------------------------
# G'MIC: seeded / single-threaded runner
# ---------------------------------------------------------------------------
# G'MIC's ``noise``/``rand`` commands and ``u()``/``v()`` in parallel ``fill``
# expressions draw from per-thread generators, so even after ``srand`` the
# output depends on OpenMP scheduling.  For the filters that use randomness we
# prefix ``srand <seed>`` (seed from ctx.rng) and run the CLI with
# ``OMP_NUM_THREADS=1``.  The environment variable is reference counted so
# concurrent renders do not restore it under each other's feet.



from ._common import gmic_single_thread as _shared_single_thread
_gmic_single_thread = _shared_single_thread  # shared thread-local switch (no global env mutation)


def run_gmic_glitch(img: np.ndarray, command: str, ctx, *, process_alpha: bool = False,
                    random: bool = False, salt: str = "gmic") -> np.ndarray:
    """Run an ImageToolboxLibs G'MIC command with the Android bridge's alpha semantics.

    ``process_alpha=False`` is GmicAlphaMode.Preserve (RGB only, source alpha
    kept -- resized if the geometry changed; any alpha channel G'MIC creates is
    dropped); ``True`` is GmicAlphaMode.Process (alpha passed through G'MIC).
    ``random`` seeds G'MIC's RNG from ``ctx.rng`` and forces single-threaded
    evaluation so the result is reproducible.
    """
    if random:
        seed = int(ctx.rng(salt).integers(0, 2 ** 31 - 1))
        command = gmic_pipeline(f"srand {seed}", command)
        with _gmic_single_thread():
            out = run_gmic(img, command, ctx=ctx, preserve_alpha=not process_alpha)
    else:
        out = run_gmic(img, command, ctx=ctx, preserve_alpha=not process_alpha)
    if not process_alpha:
        if img.shape[2] == 4 and out.shape[2] == 4:
            return out  # run_gmic re-attached the source alpha
        return out[..., :3]
    return out


def gmic_rgb(c) -> tuple[int, int, int, int]:
    """Colour tuple (0..1 floats) -> Android ``Color.red/green/blue/alpha`` ints."""
    return tuple(int(round(float(v) * 255.0)) for v in c[:4])  # type: ignore[return-value]


#: Android ``Bitmap.withPowerOfTwoPadding`` wrapper of ImageToolboxLibs (RawGmicFilter.kt).
def with_power_of_two_padding(command: str) -> str:
    return ("_gmic_android_w={w} _gmic_android_h={h} "
            "resize {2^ceil(log2(max(1,w)))},{2^ceil(log2(max(1,h)))},100%,100%,0,3,0.5,0.5 "
            + command +
            " crop {int((w-$_gmic_android_w)/2)},{int((h-$_gmic_android_h)/2)},"
            "{int((w-$_gmic_android_w)/2+$_gmic_android_w-1)},"
            "{int((h-$_gmic_android_h)/2+$_gmic_android_h-1)}")


# ---------------------------------------------------------------------------
# Counter-based random numbers for the parallel kernels
# ---------------------------------------------------------------------------
# Upstream draws from libc ``rand()`` / time-seeded std engines in pixel order;
# here every draw is a SplitMix64 hash of (stream seed, pixel index), the
# stream seed coming from ``ctx.rng``.  Same distributions, deterministic,
# independent of thread scheduling.

@njit(cache=True, inline="always")
def _urand(seed, idx):
    z = seed + np.uint64(idx) * np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return np.float64(z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, inline="always")
def _irand(seed, idx, n):
    """Uniform integer in [0, n), like ``rand() % n``."""
    return np.int64(_urand(seed, idx) * n)


def _seed64(rng) -> np.uint64:
    return np.uint64(int(rng.integers(0, 2 ** 63)))


@njit(cache=True, inline="always")
def _clampi(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


@njit(cache=True, inline="always")
def _clamp01(v):
    return np.float32(0.0) if v < 0.0 else (np.float32(1.0) if v > 1.0 else v)


# ---------------------------------------------------------------------------
# Trickle (ProcessingFilters.cpp)
# ---------------------------------------------------------------------------
# Note on channel order: Android RGBA_8888 is stored R,G,B,A in memory, so the
# ``(px >> 16) & 0xFF`` the C++ calls "r" is really blue and ``px & 0xFF`` is red.
# The code writes them back to the same byte positions, so the net effect is
# out.B = src[x + shift].B and out.R = src[x - shift].R.

@njit(cache=True, parallel=True)
def _glitch_variant_kernel(src, out, row_off, cshift):
    h, w, c = src.shape
    for y in prange(h):
        off = row_off[y]
        for x in range(w):
            sx = _clampi(x + off, 0, w - 1)
            for ch in range(c):
                out[y, x, ch] = src[y, sx, ch]
            out[y, x, 2] = src[y, _clampi(x + cshift, 0, w - 1), 2]
            out[y, x, 0] = src[y, _clampi(x - cshift, 0, w - 1), 0]


def trickle_glitch_variant(img, rng, iterations, max_offset_fraction, channel_shift_fraction):
    src = premultiply(img)
    h, w = src.shape[:2]
    max_off = _trunc_int(np.float32(w / 2.0) * np.float32(max_offset_fraction))
    cshift = _trunc_int(np.float32(w / 10.0) * np.float32(channel_shift_fraction))
    row_off = np.zeros(h, np.int64)
    for _ in range(int(iterations)):
        y = int(rng.integers(0, h))
        height = 1 + int(rng.integers(0, 5))
        off = int(rng.integers(0, 2 * max_off + 1)) - max_off
        row_off[y:y + height] = off
    out = np.empty_like(src)
    _glitch_variant_kernel(src, out, row_off, cshift)
    return unpremultiply(out)


@njit(cache=True, parallel=True)
def _vhs_kernel(src, out, wave, jit, rgb_shift, noise_amp, seed):
    h, w, c = src.shape
    inv = np.float32(1.0 / 255.0)
    dark = np.float32(0.92)
    for y in prange(h):
        for x in range(w):
            base = _clampi(np.int64(np.float32(x) + wave[y] + jit[y]), 0, w - 1)
            n = np.float32(_irand(seed, np.int64(y) * w + x, 2 * noise_amp + 1) - noise_amp) * inv
            r = _clamp01(src[y, _clampi(base - rgb_shift, 0, w - 1), 0] + n)
            g = _clamp01(src[y, base, 1] + n)
            b = _clamp01(src[y, _clampi(base + rgb_shift, 0, w - 1), 2] + n)
            if y % 3 == 0:
                r *= dark
                g *= dark
                b *= dark
            out[y, x, 0] = r
            out[y, x, 1] = g
            out[y, x, 2] = b
            for ch in range(3, c):
                out[y, x, ch] = src[y, base, ch]


def trickle_vhs(img, rng, time, strength):
    src = premultiply(img)
    h, w = src.shape[:2]
    s = np.float32(strength)
    line_jitter = _trunc_int(np.float32(w) * np.float32(0.015) * s)
    rgb_shift = _trunc_int(np.float32(w) * np.float32(0.008) * s)
    wave_amp = np.float32(2.0) * s
    noise_amp = _trunc_int(np.float32(20.0) * s)
    ys = np.arange(h, dtype=F32)
    wave = (np.sin(ys * np.float32(0.06) + np.float32(time) * np.float32(4.0)) * wave_amp).astype(F32)
    hit = rng.random(h) < 0.08 * float(strength)
    jit = np.where(hit, rng.integers(-line_jitter, line_jitter + 1, size=h), 0).astype(F32)
    out = np.empty_like(src)
    _vhs_kernel(src, out, wave, jit, rgb_shift, noise_amp, _seed64(rng))
    return unpremultiply(out)


@njit(cache=True, parallel=True)
def _block_kernel(src, out, bs, active, offx, offy):
    h, w, c = src.shape
    for y in prange(h):
        by = y // bs
        for x in range(w):
            bx = x // bs
            sy, sx = np.int64(y), np.int64(x)
            if active[by, bx]:
                sy = _clampi(y + offy[by, bx], 0, h - 1)
                sx = _clampi(x + offx[by, bx], 0, w - 1)
            for ch in range(c):
                out[y, x, ch] = src[sy, sx, ch]


def trickle_block_glitch(img, rng, block_size_fraction, strength):
    src = np.ascontiguousarray(img, dtype=F32)
    h, w = src.shape[:2]
    bs = max(_trunc_int(np.float32(w) * np.float32(block_size_fraction)), 4)
    max_off = _trunc_int(np.float32(w) * np.float32(0.15) * np.float32(strength))
    nby, nbx = -(-h // bs), -(-w // bs)
    active = rng.random((nby, nbx)) < float(strength) * 0.4
    offx = rng.integers(0, 2 * max_off + 1, size=(nby, nbx)) - max_off
    offy = rng.integers(0, 2 * bs + 1, size=(nby, nbx)) - bs
    out = np.empty_like(src)
    _block_kernel(src, out, bs, active, offx.astype(np.int64), offy.astype(np.int64))
    return out


@njit(cache=True, parallel=True)
def _crt_kernel(src, out, curvature, vignette, chroma_px):
    h, w, c = src.shape
    cx = np.float32(w * 0.5)
    cy = np.float32(h * 0.5)
    max_r = np.float32(math.sqrt(cx * cx + cy * cy))
    curve = curvature * np.float32(0.45)
    sf = np.float32(1.0) - curve * np.float32(0.2)
    for y in prange(h):
        fy = np.float32(y)
        for x in range(w):
            fx = np.float32(x)
            nx = ((fx - cx) / cx) * sf
            ny = ((fy - cy) / cy) * sf
            k = np.float32(1.0) - (nx * nx + ny * ny) * curve
            sx = _clampi(np.int64(cx + nx * cx / k), 0, w - 1)
            sy = _clampi(np.int64(cy + ny * cy / k), 0, h - 1)
            dx = fx - cx
            dy = fy - cy
            dist = np.float32(math.sqrt(dx * dx + dy * dy)) / max_r
            vig = np.float32(1.0) - vignette * dist * dist
            out[y, x, 0] = _clamp01(src[sy, _clampi(sx - chroma_px, 0, w - 1), 0] * vig)
            out[y, x, 1] = _clamp01(src[sy, sx, 1] * vig)
            out[y, x, 2] = _clamp01(src[sy, _clampi(sx + chroma_px, 0, w - 1), 2] * vig)
            for ch in range(3, c):
                out[y, x, ch] = src[sy, sx, ch]


def trickle_crt_curvature(img, curvature, vignette, chroma):
    src = premultiply(img)
    w = src.shape[1]
    out = np.empty_like(src)
    _crt_kernel(src, out, np.float32(curvature), np.float32(vignette), _trunc_int(np.float32(w) * np.float32(chroma)))
    return unpremultiply(out)


@njit(cache=True, parallel=True)
def _pixel_melt_kernel(src, out, max_drop, strength, seed_a, seed_b):
    h, w, c = src.shape
    nblocks = (w + 63) // 64
    for blk in prange(nblocks):
        x0 = blk * 64
        x1 = min(w, x0 + 64)
        drops = np.zeros(x1 - x0, np.int64)
        for y in range(h):
            for x in range(x0, x1):
                idx = np.int64(x) * h + y
                if _urand(seed_a, idx) < strength:
                    drops[x - x0] = 1 + _irand(seed_b, idx, max_drop)
                ny = min(y + drops[x - x0], h - 1)
                for ch in range(c):
                    out[ny, x, ch] = src[y, x, ch]


def trickle_pixel_melt(img, rng, max_drop, strength):
    src = np.ascontiguousarray(img, dtype=F32)
    out = src.copy()
    if max_drop <= 0:  # upstream: rand() % 0 is undefined behaviour; treated as no-op
        return out
    _pixel_melt_kernel(src, out, int(max_drop), float(strength), _seed64(rng), _seed64(rng))
    return out


# ---------------------------------------------------------------------------
# aire
# ---------------------------------------------------------------------------

def aire_glitch(img, rng, channels_shift_x, channels_shift_y, corruption_size, corruption_count,
                corruption_shift_x, corruption_shift_y):
    """aire ``glitchEffect`` (shift/Glitch.cpp) on premultiplied RGBA; output is opaque."""
    h, w = img.shape[:2]
    data = premultiply(img)
    if data.shape[2] == 3:
        data = np.dstack([data, np.ones((h, w), F32)])
    transient = data.copy()
    transient[h - 1] = 0.0  # upstream copies only (height - 1) rows into the scratch buffer
    sx = _trunc_int(np.float32(channels_shift_x) * w)
    sy = _trunc_int(np.float32(channels_shift_y) * h)
    y0, y1 = max(0, -sy), min(h, h - sy)
    x0, x1 = max(0, -sx), min(w, w - sx)
    if y1 > y0 and x1 > x0:  # G and B of every source pixel move by (+sx, +sy)
        transient[y0 + sy:y1 + sy, x0 + sx:x1 + sx, 1:3] = data[y0:y1, x0:x1, 1:3]
    del data
    transient2 = transient.copy()
    rh = _trunc_int(np.float32(h) * np.float32(corruption_size))
    cx = _trunc_int(np.float32(w) * np.float32(corruption_shift_x))
    cy = _trunc_int(np.float32(h) * np.float32(corruption_shift_y))
    for _ in range(int(corruption_count)):
        ry = int(rng.integers(0, h))
        rx = int(rng.integers(0, w))
        rng.integers(1, max(2, w))  # "length" draw, unused upstream (loop runs to the row end)
        za, zb = max(ry, -cy), min(ry + rh, h, h - cy)
        ja, jb = max(rx, -cx), min(w, w - cx)
        if zb > za and jb > ja:
            transient2[za:zb, ja:jb, :3] = transient[za + cy:zb + cy, ja + cx:jb + cx, :3]
    del transient
    # halftone horizontal lines: 2px of black at alpha 0.15 every 4 rows, then premultiply by alpha
    line_a = np.float32(np.uint8(255 * 0.15)) / np.float32(255.0)
    factor = np.where((np.arange(h) % 4) < 2, np.float32(1.0) - line_a, np.float32(1.0)).astype(F32)
    rgb = transient2[..., :3]
    rgb *= transient2[..., 3:4]
    rgb *= factor[:, None, None]
    return np.clip(rgb, 0.0, 1.0)


def _aire_gaussian_kernel(width: int, sigma: float) -> np.ndarray:
    if width <= 0:
        return np.zeros(0, F32)
    mean = width // 2
    x = np.arange(width, dtype=F32)
    k = np.exp(np.float32(-0.5) * ((x - mean) / np.float32(sigma)) ** 2).astype(F32)
    k *= np.float32(1.0 / (math.sqrt(2 * math.pi) * sigma))
    return (k / k.sum()).astype(F32)


@njit(cache=True, parallel=True)
def _wind_kernel(src, out, row_len, last_y, clear):
    h, w, c = src.shape
    for y in prange(h):
        if y >= last_y:
            for x in range(w):
                for ch in range(c):
                    out[y, x, ch] = src[y, x, ch]
            continue
        length = row_len[y]
        scale = np.float32(w) / np.float32(w - length)
        lim = w - length
        for x in range(w):
            if x < lim:
                sxf = np.float32(x) * scale
                x1 = min(np.int64(sxf), w - 1)
                x2 = min(x1 + 1, w - 1)
                dx = max(sxf - np.float32(x1), np.float32(0.0))
                for ch in range(c):
                    a = src[y, x1, ch]
                    out[y, x, ch] = a + (src[y, x2, ch] - a) * dx
            else:
                for ch in range(c):
                    out[y, x, ch] = clear[ch]


def aire_horizontal_wind_stagger(img, rng, wind_strength, streams_count, clear_color):
    """aire ``horizontalWindStagger`` (shift/WindStagger.cpp), positive-strength branch."""
    h, w = img.shape[:2]
    src = premultiply(img)
    if src.shape[2] == 3:
        src = np.dstack([src, np.ones((h, w), F32)])
    cr, cg, cb, ca = (np.float32(v) for v in clear_color)
    clear = np.array([cr * ca, cg * ca, cb * ca, ca], F32)
    stagger = _trunc_int(np.float32(w) * np.float32(abs(wind_strength)))
    lo = _trunc_int(np.float32(w) * np.float32(0.0001))
    hi = max(lo, min(max(stagger, 0), w - 1))
    stream_size = _trunc_int(np.float32(h) / np.float32(streams_count))
    lengths = rng.integers(lo, hi + 1, size=int(streams_count))
    row_len = np.zeros(h, np.int64)
    last_y = 0
    if stream_size > 0:
        nrows = min(int(streams_count) * stream_size, h)
        last_y = nrows - 1  # the last processed row is overwritten by the final source copy
        rows = np.arange(nrows)
        kernel = _aire_gaussian_kernel(stream_size, 1.0)
        row_len[:nrows] = (kernel[rows % stream_size] * lengths[rows // stream_size].astype(F32)).astype(np.int64)
    out = np.empty_like(src)
    _wind_kernel(src, out, row_len, last_y, clear)
    res = unpremultiply(out)
    if img.shape[2] == 3 and ca >= 1.0:
        return res[..., :3]
    return res


@njit(cache=True, parallel=True, error_model="numpy")
def _aldridge_kernel(src, out, exposure, cutoff):
    tmp = np.float32(2.0) * cutoff
    k = np.float32(0.25) / cutoff
    n = src.shape[0]
    for p in prange(n):
        v = src[p]
        if v <= np.float32(0.045):
            lin = v / np.float32(12.92)
        else:
            lin = ((v + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4)
        cin = exposure * lin
        d = tmp - cin
        x = cin + d * min(max(d, np.float32(0.0)), np.float32(1.0)) * k - cutoff
        cout = (x * (np.float32(6.2) * x + np.float32(0.5))) / (x * (np.float32(6.2) * x + np.float32(1.7)) + np.float32(0.06))
        if not (cout > 0.0):  # pow(negative, 2.4) is NaN upstream -> written as 0
            out[p] = np.float32(0.0)
            continue
        o = cout ** np.float32(2.4)
        if o <= np.float32(0.0031308):
            s = o * np.float32(12.92)
        else:
            s = np.float32(1.055) * o ** np.float32(1.0 / 2.4) - np.float32(0.055)
        out[p] = min(max(s, np.float32(0.0)), np.float32(1.0))


def aire_aldridge(img, exposure, cutoff):
    """aire Aldridge tone mapper applied in linear sRGB (convolveToneMapper, 0.045 sRGB threshold)."""
    src = np.ascontiguousarray(img[..., :3], dtype=F32)
    out = np.empty_like(src)
    _aldridge_kernel(src.reshape(-1), out.reshape(-1), np.float32(exposure), np.float32(cutoff))
    return out


# ---------------------------------------------------------------------------
# ImageToolbox JpegGlitch
# ---------------------------------------------------------------------------

def jpeg_glitch(img, amount: int, seed: int, iterations: int) -> np.ndarray:
    """``GlitchTool.jpegGlitch``: overwrite bytes of a q100 JPEG then decode it.

    ``getJpegHeaderSize`` compares signed Kotlin bytes with 255/218, which never
    matches, so the header length is always the fallback 417.
    """
    rgb = premultiply(img)[..., :3]
    u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 100, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    if not ok:
        return rgb
    data = bytearray(enc.tobytes())
    header = 417
    value = int(math.floor(np.float32(amount) / np.float32(100.0) * np.float32(256.0))) & 0xFF
    f = np.float32
    for pos in range(int(iterations)):
        max_index = f(len(data) - header) - f(4.0)
        px_min = max_index / f(iterations) * f(pos)
        px_max = max_index / f(iterations) * f(pos + 1)
        delta = px_max - px_min
        px_index = px_min + delta * f(seed) / f(100.0)
        if px_index > max_index:
            px_index = max_index
        index = int(math.floor(float(f(header) + px_index)))
        if 0 <= index < len(data):
            data[index] = value
    dec = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
    if dec is None or dec.shape[:2] != u8.shape[:2]:
        return rgb
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(F32) / np.float32(255.0)


# ---------------------------------------------------------------------------
# ImageToolbox procedural GLSL shaders (ProceduralShaders.kt) on the CPU
# ---------------------------------------------------------------------------
# GPUImage renders with textureCoordinate = pixel centre / size (top row = 0),
# ``sampleInput`` clamps to the pixel centres and samples bilinearly
# (GL_LINEAR) from the premultiplied bitmap texture.

_TAU = np.float32(6.28318530718)
_PROC_IDS = {"StreakExpand": 0, "RadialStreak": 1, "ColumnStreak": 2, "TiledStreak": 3, "SpiralBreaks": 4,
             "ConcentricCircleBreaks": 5, "ConcentricSquareBreaks": 6, "InlineBreaks": 7,
             "RandomColorDispersion": 8, "StreakCircles": 9}
#: effect-specific uniforms in the order they are packed after the common ones
_PROC_ARGS = {"StreakExpand": ("size",), "RadialStreak": ("count", "radius"),
              "ColumnStreak": ("variability", "count"), "TiledStreak": ("balance", "variability", "frequency", "seed"),
              "SpiralBreaks": ("distortion", "count"),
              "ConcentricCircleBreaks": ("dampening", "distortion", "frequency"),
              "ConcentricSquareBreaks": ("dampening", "distortion", "frequency"),
              "InlineBreaks": ("distortion", "frequency"), "RandomColorDispersion": ("seed",),
              "StreakCircles": ("count", "radius", "layers", "seed")}


@njit(cache=True, inline="always")
def _fract(x):
    return x - np.float32(math.floor(x))


@njit(cache=True)
def _phash(px, py):
    """GLSL ``hash(vec2)`` of ProceduralShaders.kt (float32)."""
    hx = _fract(px * np.float32(0.1031))
    hy = _fract(py * np.float32(0.1030))
    hz = _fract(px * np.float32(0.0973))
    d = hx * (hy + np.float32(33.33)) + hy * (hz + np.float32(33.33)) + hz * (hx + np.float32(33.33))
    hx += d
    hy += d
    hz += d
    return _fract((hx + hy) * hz)


@njit(cache=True, inline="always")
def _angle_of(x, y):
    if x * x + y * y < np.float32(1e-8):
        return np.float32(0.0)
    return np.float32(math.atan2(y, x))


@njit(cache=True, inline="always")
def _rot(x, y, a):
    c = np.float32(math.cos(a))
    s = np.float32(math.sin(a))
    return c * x - s * y, s * x + c * y


@njit(cache=True, inline="always")
def _mirror(u):
    m = u - np.float32(2.0) * np.float32(math.floor(u / np.float32(2.0)))  # GLSL mod(u, 2)
    return np.float32(1.0) - abs(m - np.float32(1.0))


@njit(cache=True)
def _sample_input(src, u, v, dst):
    h, w, c = src.shape
    u = min(max(u, np.float32(0.5) / w), np.float32(1.0) - np.float32(0.5) / w)
    v = min(max(v, np.float32(0.5) / h), np.float32(1.0) - np.float32(0.5) / h)
    fx = u * w - np.float32(0.5)
    fy = v * h - np.float32(0.5)
    x0 = _clampi(np.int64(math.floor(fx)), 0, w - 1)
    y0 = _clampi(np.int64(math.floor(fy)), 0, h - 1)
    tx = fx - np.float32(x0)
    ty = fy - np.float32(y0)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    for ch in range(c):
        a = src[y0, x0, ch] + (src[y0, x1, ch] - src[y0, x0, ch]) * tx
        b = src[y1, x0, ch] + (src[y1, x1, ch] - src[y1, x0, ch]) * tx
        dst[ch] = a + (b - a) * ty


@njit(cache=True)
def _sample_field(src, px, py, prm, dst):
    xr, yr = _rot(px * prm[4], py * prm[4], prm[5])
    u = _mirror(xr / prm[0] + np.float32(0.5) + prm[2])
    v = _mirror(yr / prm[1] + np.float32(0.5) + prm[3])
    _sample_input(src, u, v, dst)


@njit(cache=True, parallel=True)
def _proc_render(effect, src, out, prm, circ):
    """main() of the generated fragment shader for one ProceduralEffect.

    prm = [aspect.x, aspect.y, offsetX, offsetY, scale, radians(rotation), intensity, p0, p1, p2, p3]
    """
    h, w, c = src.shape
    it = prm[6]
    one = np.float32(1.0)
    half = np.float32(0.5)
    for y in prange(h):
        t0 = np.empty(c, np.float32)
        t1 = np.empty(c, np.float32)
        t2 = np.empty(c, np.float32)
        v = (np.float32(y) + half) / np.float32(h)
        for x in range(w):
            u = (np.float32(x) + half) / np.float32(w)
            # p = toField(textureCoordinate)
            fx, fy = _rot((u - half - prm[2]) * prm[0], (v - half - prm[3]) * prm[1], -prm[5])
            px = fx / prm[4]
            py = fy / prm[4]
            if effect == 8:  # RandomColorDispersion
                seed = prm[7]
                gx = np.float32(math.floor(px * np.float32(24.0))) + seed
                gy = np.float32(math.floor(py * np.float32(24.0))) + seed
                k = it * np.float32(0.15)
                dx = (_phash(gx, gy) - half) * k
                dy = (_phash(gx + np.float32(17.17), gy + np.float32(47.23)) - half) * k
                _sample_field(src, px + dx, py + dy, prm, t0)
                _sample_field(src, px, py, prm, t1)
                _sample_field(src, px - dx, py - dy, prm, t2)
                out[y, x, 0] = t0[0]
                out[y, x, 1] = t1[1]
                out[y, x, 2] = t2[2]
                if c == 4:
                    out[y, x, 3] = max(t0[3], max(t1[3], t2[3]))
                continue
            if effect == 9:  # StreakCircles
                for ch in range(c):
                    t0[ch] = src[y, x, ch]  # sampleInput(textureCoordinate) == the pixel itself
                layers = prm[9]
                for i in range(circ.shape[0]):
                    ccx = circ[i, 0]
                    ccy = circ[i, 1]
                    lim = circ[i, 2]
                    ddx = px - ccx
                    ddy = py - ccy
                    r = np.float32(math.sqrt(ddx * ddx + ddy * ddy))
                    if r < lim:
                        ring = np.float32(math.floor(r / lim * layers)) / layers
                        _sample_field(src, ccx + ring * lim, ccy, prm, t1)
                        for ch in range(c):
                            t0[ch] = t0[ch] + (t1[ch] - t0[ch]) * it
                for ch in range(c):
                    out[y, x, ch] = t0[ch]
                continue
            qx, qy = px, py
            if effect == 0:  # StreakExpand
                ay = abs(py) - prm[7] * half
                mapped = np.float32(0.0)
                if ay > 0.0:
                    mapped = ay if py > 0 else -ay
                qy = py + (mapped - py) * it
            elif effect == 1:  # RadialStreak
                r = np.float32(math.sqrt(px * px + py * py))
                segment = np.float32(math.floor(_angle_of(px, py) / _TAU * prm[7]))
                target = prm[8] * (np.float32(0.7) + np.float32(0.3) * _phash(segment, np.float32(3.0)))
                m = one + (target / max(r, np.float32(1e-5)) - one) * it
                qx, qy = px * m, py * m
            elif effect == 2:  # ColumnStreak
                column = np.float32(math.floor(px * prm[8]))
                levels = np.float32(4.0) + np.float32(60.0) * (_phash(column, one) * prm[7])
                qy = py + ((np.float32(math.floor(py * levels)) + half) / levels - py) * it
            elif effect == 3:  # TiledStreak
                f = prm[9]
                cx = np.float32(math.floor(px * f))
                cy = np.float32(math.floor(py * f))
                lx = _fract(px * f) - half
                ly = _fract(py * f) - half
                ang = _angle_of(lx, ly)
                rad = np.float32(math.sqrt(lx * lx + ly * ly))
                mx = (cx + half) / f + rad / f
                my = (cy + half) / f + np.float32(math.sin(ang)) * (one - prm[7]) * np.float32(0.2) / f
                mx += (_phash(cx + prm[10], cy + prm[10]) - half) * prm[8] / f
                my += (_phash(cx + prm[10] + np.float32(17.17), cy + prm[10] + np.float32(47.23)) - half) * prm[8] / f
                qx = px + (mx - px) * it
                qy = py + (my - py) * it
            elif effect == 4:  # SpiralBreaks
                r = np.float32(math.sqrt(px * px + py * py))
                segment = np.float32(math.floor((_angle_of(px, py) + r * np.float32(8.0)) / _TAU * prm[8]))
                ox = _phash(segment, np.float32(3.0)) - half
                oy = _phash(segment + np.float32(17.17), np.float32(3.0) + np.float32(47.23)) - half
                rx, ry = _rot(px, py, ox * it * prm[7])
                qx = rx + ox * it * np.float32(0.15)
                qy = ry + oy * it * np.float32(0.15)
            elif effect == 5 or effect == 6:  # Concentric circle / square breaks
                if effect == 5:
                    r = np.float32(math.sqrt(px * px + py * py))
                else:
                    r = max(abs(px), abs(py))
                f = prm[9]
                band = np.float32(math.floor(r * f))
                ox = _phash(band, np.float32(17.0)) - half
                oy = _phash(band + np.float32(17.17), np.float32(17.0) + np.float32(47.23)) - half
                strength = it * np.float32(math.exp(-r * prm[7]))
                rx, ry = _rot(px, py, ox * prm[8] * strength)
                qx = rx + ox * strength / f
                qy = ry + oy * strength / f
            elif effect == 7:  # InlineBreaks
                f = prm[8]
                line = np.float32(math.floor(py * f))
                shift = _phash(line, np.float32(13.0)) - half
                qx = px + shift * it * np.float32(0.4)
                qy = py + np.float32(math.sin(px * np.float32(12.0) + line)) * it * prm[7] / f
            _sample_field(src, qx, qy, prm, t0)
            for ch in range(c):
                out[y, x, ch] = t0[ch]


def procedural(effect: str, img: np.ndarray, values: dict) -> np.ndarray:
    src = premultiply(img)  # GL textures hold premultiplied bitmaps
    h, w = src.shape[:2]
    short = float(max(1, min(w, h)))
    prm = np.zeros(11, np.float32)
    prm[:7] = [w / short, h / short, values.get("offset_x", 0.0), values.get("offset_y", 0.0),
               values.get("scale", 1.0), math.radians(values.get("rotation", 0.0)), values.get("intensity", 0.0)]
    for k, name in enumerate(_PROC_ARGS[effect]):
        prm[7 + k] = values[name]
    circ = np.zeros((0, 3), np.float32)
    if effect == "StreakCircles":
        seed, radius = np.float32(values["seed"]), np.float32(values["radius"])
        rows = []
        for i in range(min(int(values["count"]), 64)):
            fi = np.float32(i)
            ccx = (_phash(fi, seed) - np.float32(0.5)) * prm[0]
            ccy = (_phash(fi + np.float32(17.17), seed + np.float32(47.23)) - np.float32(0.5)) * prm[1]
            lim = radius * (np.float32(0.3) + _phash(fi, seed + np.float32(7.0)))
            rows.append((ccx, ccy, lim))
        circ = np.array(rows, np.float32).reshape(-1, 3)
    out = np.empty_like(src)
    _proc_render(_PROC_IDS[effect], src, out, prm, circ)
    return unpremultiply(out)


# ===========================================================================
# NTSC / VHS emulation -- port of ntsc-rs (github.com/ntsc-rs/ntsc-rs, crate
# ``ntscrs`` at rev add90f5, MIT/ISC/Apache-2.0) as driven by Trickle's
# ``ntsc_jni`` bridge.  Random streams are reproduced exactly: the Seeder is
# SipHash-2-4 (keys 0) over the Rust ``Hash`` byte stream, RNGs are
# Xoshiro256++ seeded through SplitMix64, floats follow rand 0.9, and the
# gradient noise is the Clatter-derived simplex noise used by ntsc-rs.
# ===========================================================================

_M32 = np.uint64(0xFFFFFFFF)
_PHI = np.uint64(0x9E3779B97F4A7C15)
_SM1 = np.uint64(0xBF58476D1CE4E5B9)
_SM2 = np.uint64(0x94D049BB133111EB)
_SIP0 = np.uint64(0x736F6D6570736575)
_SIP1 = np.uint64(0x646F72616E646F6D)
_SIP2 = np.uint64(0x6C7967656E657261)
_SIP3 = np.uint64(0x7465646279746573)
_XO_JUMP = np.array([0x180EC6D33CFD0ABA, 0xD5A61266F0C9392C, 0xA9582618E03FC9AA, 0x39ABDC4529B1661C],
                    dtype=np.uint64)
_H1 = np.uint64(2747636419)
_H2 = np.uint64(2654435769)
_PCG_A = np.uint64(1664525)
_PCG_C = np.uint64(1013904223)
_SKEW = np.float32((math.sqrt(3.0) - 1.0) / 2.0)
_UNSKEW = np.float32(-((1.0 / math.sqrt(3.0)) - 1.0) / 2.0)
_PI32 = np.float32(math.pi)
_INV_2_23 = np.float32(1.0 / 8388608.0)
_INV_2_24 = np.float32(1.0 / 16777216.0)


@njit(cache=True)
def _rotl(x, b):
    return (x << np.uint64(b)) | (x >> np.uint64(64 - b))


@njit(cache=True)
def _sip_rounds(v0, v1, v2, v3, n):
    for _ in range(n):
        v0 = v0 + v1
        v1 = _rotl(v1, 13)
        v1 = v1 ^ v0
        v0 = _rotl(v0, 32)
        v2 = v2 + v3
        v3 = _rotl(v3, 16)
        v3 = v3 ^ v2
        v0 = v0 + v3
        v3 = _rotl(v3, 21)
        v3 = v3 ^ v0
        v2 = v2 + v1
        v1 = _rotl(v1, 17)
        v1 = v1 ^ v2
        v2 = _rotl(v2, 32)
    return v0, v1, v2, v3


@njit(cache=True)
def _siphash(msg, extra, extra_len):
    """SipHash-2-4 with zero keys of ``msg`` followed by ``extra_len`` little-endian bytes of ``extra``."""
    nmsg = msg.shape[0]
    n = nmsg + extra_len
    buf = np.empty(n, np.uint8)
    for k in range(nmsg):
        buf[k] = msg[k]
    for k in range(extra_len):
        buf[nmsg + k] = np.uint8((extra >> np.uint64(8 * k)) & np.uint64(0xFF))
    v0, v1, v2, v3 = _SIP0, _SIP1, _SIP2, _SIP3
    nb = n // 8
    for b in range(nb):
        m = np.uint64(0)
        for k in range(8):
            m |= np.uint64(buf[b * 8 + k]) << np.uint64(8 * k)
        v3 ^= m
        v0, v1, v2, v3 = _sip_rounds(v0, v1, v2, v3, 2)
        v0 ^= m
    last = np.uint64(n & 0xFF) << np.uint64(56)
    for k in range(n - nb * 8):
        last |= np.uint64(buf[nb * 8 + k]) << np.uint64(8 * k)
    v3 ^= last
    v0, v1, v2, v3 = _sip_rounds(v0, v1, v2, v3, 2)
    v0 ^= last
    v2 ^= np.uint64(0xFF)
    v0, v1, v2, v3 = _sip_rounds(v0, v1, v2, v3, 4)
    return v0 ^ v1 ^ v2 ^ v3


def seeder(*parts: tuple[int, int]) -> np.ndarray:
    """``Seeder::new(a).mix(b)...`` byte stream: ``(value, nbytes)`` pairs, little endian.

    Rust hashes ``u64``/``usize`` as 8 bytes and an untyped integer literal
    (``i32``) as 4 bytes.
    """
    out = bytearray()
    for value, nbytes in parts:
        out += (int(value) & ((1 << (8 * nbytes)) - 1)).to_bytes(nbytes, "little")
    return np.frombuffer(bytes(out), np.uint8).copy()


def seeder_finalize(msg: np.ndarray, extra: int = 0, extra_len: int = 0) -> int:
    return int(_siphash(msg, np.uint64(extra), extra_len))


def seed_to_f32(h: int) -> np.float32:
    return np.float32(((h & 0xFFFFFFFF) >> 9) * (1.0 / 8388608.0))


def seed_to_i32(h: int) -> int:
    v = h & 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


@njit(cache=True)
def _xo_seed(seed, s):
    x = seed
    for k in range(4):
        x = x + _PHI
        z = x
        z = (z ^ (z >> np.uint64(30))) * _SM1
        z = (z ^ (z >> np.uint64(27))) * _SM2
        s[k] = z ^ (z >> np.uint64(31))


@njit(cache=True)
def _xo_next(s):
    s0, s1, s2, s3 = s[0], s[1], s[2], s[3]
    res = _rotl(s0 + s3, 23) + s0
    t = s1 << np.uint64(17)
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = _rotl(s3, 45)
    s[0], s[1], s[2], s[3] = s0, s1, s2, s3
    return res


@njit(cache=True)
def _xo_jump(s):
    a0 = np.uint64(0)
    a1 = np.uint64(0)
    a2 = np.uint64(0)
    a3 = np.uint64(0)
    for j in range(4):
        jj = _XO_JUMP[j]
        for b in range(64):
            if (jj >> np.uint64(b)) & np.uint64(1):
                a0 ^= s[0]
                a1 ^= s[1]
                a2 ^= s[2]
                a3 ^= s[3]
            _xo_next(s)
    s[0], s[1], s[2], s[3] = a0, a1, a2, a3


@njit(cache=True)
def _xo_u32(s):
    return _xo_next(s) >> np.uint64(32)


@njit(cache=True)
def _xo_f32(s):
    """``rng.random::<f32>()``: (next_u32 >> 8) * 2^-24."""
    return np.float32(_xo_next(s) >> np.uint64(40)) * _INV_2_24


@njit(cache=True)
def _xo_f64(s):
    return np.float64(_xo_next(s) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _xo_range_f32(s, lo, hi):
    """rand 0.9 ``UniformFloat::sample_single(_inclusive)`` for f32."""
    v = np.float32(_xo_next(s) >> np.uint64(41)) * _INV_2_23
    return v * (hi - lo) + lo


@njit(cache=True)
def _u32_to_i32(v):
    x = np.int64(v & _M32)
    if x >= 2147483648:
        x -= 4294967296
    return x


# -- simplex noise (Clatter port in ntscrs/src/noise) -------------------------

@njit(cache=True)
def _hash1d(v):
    v = ((v ^ _H1) * _H2) & _M32
    v = ((v ^ (v >> np.uint64(16))) * _H2) & _M32
    v = ((v ^ (v >> np.uint64(16))) * _H2) & _M32
    return v


@njit(cache=True)
def _grad1d(hv):
    h = hv >> np.uint64(28)
    v = np.float32((h & np.uint64(7)) + np.uint64(1))
    # ``v.copysign(((h & 8) == 0).bitcast())``: an all-ones mask has the sign bit set
    if (h & np.uint64(8)) == np.uint64(0):
        return -v
    return v


@njit(cache=True)
def _simplex1d(x, seed):
    fi = np.float32(math.floor(x))
    i0 = np.int64(fi)
    x0 = x - fi
    x1 = x0 - np.float32(1.0)
    g0 = _grad1d(_hash1d(np.uint64((i0 ^ seed) & 0xFFFFFFFF)))
    g1 = _grad1d(_hash1d(np.uint64(((i0 + 1) ^ seed) & 0xFFFFFFFF)))
    t0 = np.float32(1.0) - x0 * x0
    t0 = t0 * t0
    t0 = t0 * t0
    t1 = np.float32(1.0) - x1 * x1
    t1 = t1 * t1
    t1 = t1 * t1
    return t0 * g0 * x0 + t1 * g1 * x1


@njit(cache=True)
def _pcg3d_x(vx, vy, vz):
    vx = (vx * _PCG_A + _PCG_C) & _M32
    vy = (vy * _PCG_A + _PCG_C) & _M32
    vz = (vz * _PCG_A + _PCG_C) & _M32
    vx = (vx + vy * vz) & _M32
    vy = (vy + vz * vx) & _M32
    vz = (vz + vx * vy) & _M32
    vx ^= vx >> np.uint64(16)
    vy ^= vy >> np.uint64(16)
    vz ^= vz >> np.uint64(16)
    return (vx + vy * vz) & _M32


@njit(cache=True)
def _grad2d(hv):
    h = hv & np.uint64(7)
    lt4 = h < np.uint64(4)
    xm = np.float32(1.0) if lt4 else np.float32(2.0)
    ym = np.float32(2.0) if lt4 else np.float32(1.0)
    a1 = (h & np.uint64(1)) == np.uint64(0)
    a2 = (h & np.uint64(2)) == np.uint64(0)
    sx = a1 if lt4 else a2
    sy = a2 if lt4 else a1
    gx = xm if sx else -xm
    gy = ym if sy else -ym
    return gx, gy


@njit(cache=True)
def _simplex2d(x, y, seed):
    s = (x + y) * _SKEW
    ips = np.float32(math.floor(x + s))
    jps = np.float32(math.floor(y + s))
    i = np.int64(ips)
    j = np.int64(jps)
    t = np.float32(i + j) * _UNSKEW
    x0 = x - (ips - t)
    y0 = y - (jps - t)
    if x0 >= y0:
        i1 = -1
        j1 = 0
    else:
        i1 = 0
        j1 = -1
    x1 = x0 + np.float32(i1) + _UNSKEW
    y1 = y0 + np.float32(j1) + _UNSKEW
    x2 = x0 - np.float32(1.0) + np.float32(2.0) * _UNSKEW
    y2 = y0 - np.float32(1.0) + np.float32(2.0) * _UNSKEW
    su = np.uint64(seed & 0xFFFFFFFF)
    h0 = _pcg3d_x(np.uint64(i & 0xFFFFFFFF), np.uint64(j & 0xFFFFFFFF), su)
    h1 = _pcg3d_x(np.uint64((i - i1) & 0xFFFFFFFF), np.uint64((j - j1) & 0xFFFFFFFF), su)
    h2 = _pcg3d_x(np.uint64((i + 1) & 0xFFFFFFFF), np.uint64((j + 1) & 0xFFFFFFFF), su)
    t0 = max(np.float32(0.5) - x0 * x0 - y0 * y0, np.float32(0.0))
    t1 = max(np.float32(0.5) - x1 * x1 - y1 * y1, np.float32(0.0))
    t2 = max(np.float32(0.5) - x2 * x2 - y2 * y2, np.float32(0.0))
    t0 = t0 * t0
    t0 = t0 * t0
    t1 = t1 * t1
    t1 = t1 * t1
    t2 = t2 * t2
    t2 = t2 * t2
    gx0, gy0 = _grad2d(h0)
    gx1, gy1 = _grad2d(h1)
    gx2, gy2 = _grad2d(h2)
    return t0 * (gx0 * x0 + gy0 * y0) + t1 * (gx1 * x1 + gy1 * y1) + t2 * (gx2 * x2 + gy2 * y2)


@njit(cache=True)
def _fbm1d(x, seed, octaves, gain, lacunarity, frequency):
    x = x * frequency
    r = _simplex1d(x, seed)
    amp = gain
    for _ in range(1, octaves):
        x = x * lacunarity
        r = _simplex1d(x, seed) * amp + r
        amp = amp * gain
    return r


@njit(cache=True)
def _fbm2d(x, y, seed, octaves, gain, lacunarity, frequency):
    x = x * frequency
    y = y * frequency
    r = _simplex2d(x, y, seed)
    amp = gain
    for _ in range(1, octaves):
        x = x * lacunarity
        y = y * lacunarity
        r = _simplex2d(x, y, seed) * amp + r
        amp = amp * gain
    return r


# -- per-row helpers ----------------------------------------------------------

@njit(cache=True)
def _video_noise_line(row, prefix, index, frequency, intensity, octaves):
    """``video_noise_line``: fBm simplex noise along one scanline."""
    s = np.empty(4, np.uint64)
    _xo_seed(_siphash(prefix, np.uint64(index), 8), s)
    nseed = _u32_to_i32(_xo_u32(s))
    width = row.shape[0]
    offset = _xo_f32(s) * np.float32(width)
    k = intensity * np.float32(0.25)
    for i in range(width):
        row[i] += _fbm1d(offset + np.float32(i), nseed, octaves, np.float32(1.0), np.float32(2.0), frequency) * k


@njit(cache=True, parallel=True)
def _plane_noise(plane, prefix, frequency, intensity, octaves):
    for r in prange(plane.shape[0]):
        _video_noise_line(plane[r], prefix, r, frequency, intensity, octaves)


@njit(cache=True, error_model="numpy")
def _geometric(s, lam):
    v = math.log(_xo_f64(s)) / lam
    if v != v:
        return np.int64(0)
    if v >= 9.0e18:
        return np.int64(9000000000000000000)
    return np.int64(v)


@njit(cache=True, error_model="numpy")
def _row_speckles(row, s, intensity, anisotropy, horizontal_scale):
    """``row_speckles``: snow transients with geometric spacing."""
    width = row.shape[0]
    inten = np.float64(intensity)
    anis = np.float64(anisotropy)
    u = _xo_f64(s)
    logistic = math.exp((u - inten) / (inten * (1.0 - inten) * (1.0 - anis)))
    line = anis / (1.0 + logistic) + inten * (1.0 - anis)
    line *= 0.125
    line = min(max(line, 0.0), 1.0)
    if not (line > 0.0):
        return
    lam = math.log(1.0 - line)
    idx = np.int64(-64)
    while True:
        idx += _geometric(s, lam)
        if idx >= width:
            break
        tlen = _xo_range_f32(s, np.float32(8.0), np.float32(64.0)) * horizontal_scale
        tfreq = _xo_range_f32(s, tlen * np.float32(3.0), tlen * np.float32(5.0))
        end = idx + np.int64(math.ceil(tlen))
        _xo_jump(s)
        t = s.copy()
        a = min(max(idx, 0), width)
        b = min(max(end, 0), width)
        for i in range(a, b):
            x = np.float32(i - idx)
            c = np.float32(math.cos((x * _PI32) / tfreq))
            d = np.float32(1.0) - x / tlen
            row[i] += c * (d * d) * _xo_range_f32(t, np.float32(-1.0), np.float32(2.0))
        idx += 1


@njit(cache=True, parallel=True)
def _snow_rows(plane, prefix, intensity, anisotropy, horizontal_scale):
    for r in prange(plane.shape[0]):
        s = np.empty(4, np.uint64)
        _xo_seed(_siphash(prefix, np.uint64(r), 8), s)
        _row_speckles(plane[r], s, intensity, anisotropy, horizontal_scale)


@njit(cache=True)
def _shift_into(src, dst, shift, extend, const):
    """``shift_row``/``shift_row_to``: linear-interpolated horizontal shift."""
    w = src.shape[0]
    si = np.int64(shift)
    fr = shift - np.float32(si)
    if shift < 0:
        si -= 1
        fr = np.float32(1.0) - abs(fr)
    if extend:
        bv = src[0] if si >= 0 else src[w - 1]
    else:
        bv = const
    for i in range(w):
        li = i - si - 1
        ri = li + 1
        left = src[li] if (li >= 0 and li < w) else bv
        right = src[ri] if (ri >= 0 and ri < w) else bv
        dst[i] = left * fr + right * (np.float32(1.0) - fr)


@njit(cache=True)
def _shift_inplace(row, shift, extend, const):
    tmp = row.copy()
    _shift_into(tmp, row, shift, extend, const)


@njit(cache=True, parallel=True)
def _shift_rows(plane, shifts, extend):
    for r in prange(plane.shape[0]):
        _shift_inplace(plane[r], shifts[r], extend, np.float32(0.0))


@njit(cache=True, parallel=True)
def _iir_rows(plane, num, den, first_sample, delay):
    """``TransferFunction::filter_signal_in_place`` (scipy lfilter, direct form II transposed)."""
    h, w = plane.shape
    L = num.shape[0]
    for r in prange(h):
        row = plane[r]
        z = np.zeros(4, np.float32)
        init = row[0] if first_sample else np.float32(0.0)
        if init != 0.0:
            bsum = np.float32(0.0)
            dsum = np.float32(1.0)
            for i in range(1, L):
                bsum += num[i] - den[i - 1] * num[0]
                dsum += den[i - 1]
            z0 = bsum / dsum
            asum = np.float32(1.0)
            csum = np.float32(0.0)
            for i in range(1, L - 1):
                asum += den[i - 1]
                csum += num[i] - den[i - 1] * num[0]
                z[i] = (asum * z0 - csum) * init
            z[0] = z0 * init
        for i in range(w + delay):
            sample = row[min(i, w - 1)]
            filt = z[0] + num[0] * sample
            for k in range(L - 1):
                z[k] = z[k + 1] + num[k + 1] * sample - den[k] * filt
            if i >= delay:
                row[i - delay] = filt


_I_MULT_INV = np.array([-1.0, 0.0, 1.0, 0.0], np.float32)
_Q_MULT_INV = np.array([0.0, -1.0, 0.0, 1.0], np.float32)
_I_MULT = np.array([1.0, 0.0, -1.0, 0.0], np.float32)
_Q_MULT = np.array([0.0, 1.0, 0.0, -1.0], np.float32)


@njit(cache=True, parallel=True)
def _demodulate(y, i_out, q_out, modulated, xi_rows):
    """``demodulate_chroma_line`` for every row."""
    h, w = y.shape
    for r in prange(h):
        xi = xi_rows[r]
        for idx in range(w):
            oc = (idx + xi) & 3
            c = y[r, idx] - modulated[r, idx]
            iv = c * _I_MULT_INV[oc]
            qv = c * _Q_MULT_INV[oc]
            if idx < w - 1:
                o = (idx + 1 + xi) & 3
                cr = y[r, idx + 1] - modulated[r, idx + 1]
                iv += cr * _I_MULT_INV[o] * np.float32(0.5)
                qv += cr * _Q_MULT_INV[o] * np.float32(0.5)
            if idx > 0:
                o = (idx - 1 + xi) & 3
                cl = y[r, idx - 1] - modulated[r, idx - 1]
                iv += cl * _I_MULT_INV[o] * np.float32(0.5)
                qv += cl * _Q_MULT_INV[o] * np.float32(0.5)
            i_out[r, idx] = iv
            q_out[r, idx] = qv


@njit(cache=True, parallel=True)
def _chroma_into_luma(y, i, q, xi_rows):
    """``chroma_into_luma_line`` for every row (QAM-modulate I/Q onto the luma carrier)."""
    for r in prange(y.shape[0]):
        x3 = xi_rows[r] & 3
        for k in range(y.shape[1]):
            ph = (k + x3) & 3
            y[r, k] += i[r, k] * _I_MULT[ph] + q[r, k] * _Q_MULT[ph]


@njit(cache=True, parallel=True)
def _vert_blend(plane):
    """``chroma_vert_blend``: average each chroma row with the one above (row -1 is 0)."""
    h, w = plane.shape
    for k in prange(w):
        prev = np.float32(0.0)
        for r in range(h):
            cur = plane[r, k]
            plane[r, k] = (prev + cur) * np.float32(0.5)
            prev = cur


@njit(cache=True, parallel=True)
def _chroma_phase_noise_rows(i, q, prefix, intensity):
    for r in prange(i.shape[0]):
        hv = _siphash(prefix, np.uint64(r), 8)
        f = np.float32((hv & _M32) >> np.uint64(9)) * _INV_2_23
        phase = (f - np.float32(0.5)) * np.float32(2.0) * intensity * _PI32 * np.float32(2.0)
        sa = np.float32(math.sin(phase))
        ca = np.float32(math.cos(phase))
        for k in range(i.shape[1]):
            iv = i[r, k]
            qv = q[r, k]
            i[r, k] = iv * ca - qv * sa
            q[r, k] = iv * sa + qv * ca


# -- IIR transfer functions (ntscrs/src/filter.rs) ---------------------------

NTSC_RATE = np.float32((315000000.00 / 88.0) * 4.0)


class TransferFunction:
    def __init__(self, num, den):
        num = [np.float32(v) for v in num]
        den = [np.float32(v) for v in den]
        length = len(den) + 1
        if len(num) > 4 or len(den) > 3 or len(num) > length:  # pragma: no cover - programming error
            raise ValueError("Filter is too large")
        self.num = np.zeros(length, np.float32)
        self.num[:len(num)] = num
        self.den = np.array(den, np.float32)

    def __len__(self) -> int:
        return len(self.num)

    def with_scale(self, scale) -> "TransferFunction":
        s = np.float32(scale)
        one = np.float32(1.0)
        nn = [s * self.num[0] + (one - s)]
        for i in range(1, len(self)):
            nn.append(s * self.num[i] + (one - s) * self.den[i - 1])
        return TransferFunction(nn, self.den)

    def __mul__(self, other: "TransferFunction") -> "TransferFunction":
        num = np.trim_zeros(np.convolve(self.num, other.num).astype(np.float32), "b")
        dl = np.concatenate([[np.float32(1.0)], self.den]).astype(np.float32)
        dr = np.concatenate([[np.float32(1.0)], other.den]).astype(np.float32)
        den = np.trim_zeros(np.convolve(dl, dr).astype(np.float32), "b")[1:]
        return TransferFunction(num, den)

    def cascade_self(self, n: int) -> "TransferFunction":
        f = self
        for _ in range(1, n):
            f = f * self
        return f

    def apply(self, plane: np.ndarray, first_sample: bool = False, delay: int = 0) -> None:
        if plane.size:
            _iir_rows(plane, self.num, self.den, bool(first_sample), int(delay))


def make_lowpass(cutoff, rate) -> TransferFunction:
    f = np.float32
    ti = f(1.0) / f(rate)
    tau = f(1.0) / (f(cutoff) * f(2.0) * _PI32)
    alpha = ti / (tau + ti)
    return TransferFunction([alpha], [-(f(1.0) - alpha)])


def make_notch(freq, quality) -> TransferFunction:
    f = np.float32
    bandwidth = (f(freq) / f(quality)) * _PI32
    fr = f(freq) * _PI32
    beta = f(math.tan(bandwidth * f(0.5)))
    gain = f(1.0) / (f(1.0) + beta)
    c = f(math.cos(fr))
    return TransferFunction([gain, f(-2.0) * c * gain, gain], [f(-2.0) * c * gain, f(2.0) * gain - f(1.0)])


def make_butterworth(cutoff, rate) -> TransferFunction:
    f = np.float32
    freq = min(f(2.0) * f(cutoff), f(rate)) / f(rate) * _PI32
    os_, oc = f(math.sin(freq)), f(math.cos(freq))
    alpha = os_ / (f(2.0) * f(1.0 / math.sqrt(2.0)))
    gain = f(1.0) / (f(1.0) + alpha)
    return TransferFunction([(f(1.0) - oc) * f(0.5) * gain, (f(1.0) - oc) * gain, (f(1.0) - oc) * f(0.5) * gain],
                            [f(-2.0) * oc * gain, (f(1.0) - alpha) * gain])


def make_lowpass_for_type(cutoff, rate, filter_type: int) -> TransferFunction:
    if filter_type == 0:  # ConstantK
        return make_lowpass(cutoff, rate).cascade_self(3)
    return make_butterworth(cutoff, rate)


# -- the effect ---------------------------------------------------------------

# ntsc-rs enum orders (as mapped by the Trickle JNI bridge)
USE_FIELD = ["alternating", "upper", "lower", "both", "interleaved_upper", "interleaved_lower"]
FILTER_TYPE = ["constant_k", "butterworth"]
LUMA_LOWPASS = ["none", "box", "notch"]
CHROMA_LOWPASS = ["none", "light", "full"]
CHROMA_DEMOD = ["box", "notch", "one_line_comb", "two_line_comb"]
PHASE_SHIFT = ["degrees_0", "degrees_90", "degrees_180", "degrees_270"]
TAPE_SPEED = ["none", "sp", "lp", "ep"]
_TAPE_PARAMS = {1: (2400000.0, 320000.0, 4), 2: (1900000.0, 300000.0, 5), 3: (1400000.0, 280000.0, 6)}

_NS_VIDEO_COMPOSITE, _NS_HEAD_SWITCHING, _NS_TRACKING_NOISE, _NS_CHROMA_PHASE = 0, 2, 3, 4
_NS_EDGE_WAVE, _NS_SNOW, _NS_CHROMA_LOSS, _NS_MID_LINE_JITTER = 5, 6, 7, 8
_NS_CHROMA_I, _NS_CHROMA_Q, _NS_LUMA = 1, 9, 10

_UPPER, _LOWER, _BOTH = 1, 2, 3


def itb_ntsc_config(p: dict) -> dict:
    """ImageToolbox ``NtscParams.toNtscSettings()`` + the JNI bridge's sanitising."""
    amount = float(p["amount"])
    chroma_bleed = p["chroma_bleed"] * amount
    tape_wear = p["tape_wear"] * amount
    noise = p["noise"] * amount
    tracking = p["tracking"] * amount
    snow = p["snow"] * amount
    ringing = p["ringing"] * amount
    hs_amount = max(tracking, tape_wear * 0.7)
    cfg = {
        "seed": int(p["seed"]),
        "use_field": USE_FIELD.index(p["use_field"]),
        "filter_type": FILTER_TYPE.index(p["filter_type"]),
        "input_luma_filter": LUMA_LOWPASS.index(p["input_luma_filter"]),
        "chroma_lowpass_in": CHROMA_LOWPASS.index(p["chroma_lowpass_in"]),
        "chroma_demodulation": CHROMA_DEMOD.index(p["chroma_demodulation"]),
        "luma_smear": p["luma_smear"] * amount,
        "composite_sharpening": p["composite_sharpening"] * amount,
        "phase_shift": PHASE_SHIFT.index(p["video_scanline_phase_shift"]),
        "phase_shift_offset": int(p["video_scanline_phase_shift_offset"]),
        "head_switching": None,
        "tracking_noise": None,
        "composite_noise": None,
        "ringing": None,
        "luma_noise": None,
        "chroma_noise": None,
        "snow_intensity": max(0.0, p["snow_intensity"] * snow),
        "snow_anisotropy": p["snow_anisotropy"],
        "chroma_phase_noise": max(0.0, p["chroma_phase_noise_intensity"] * chroma_bleed),
        "chroma_phase_error": p["chroma_phase_error"] * chroma_bleed,
        "chroma_delay_h": p["chroma_delay_horizontal"] * chroma_bleed,
        "chroma_delay_v": round_half_up(p["chroma_delay_vertical"] * chroma_bleed),
        "vhs": None,
        "chroma_vert_blend": True,  # NtscSettings default, not exposed by ImageToolbox
        "chroma_lowpass_out": CHROMA_LOWPASS.index(p["chroma_lowpass_out"]),
        "scale": None,
    }
    if p["head_switching_enabled"]:
        cfg["head_switching"] = (round_half_up(p["head_switching_height"] * hs_amount),
                                 round_half_up(p["head_switching_offset"] * tracking),
                                 p["head_switching_horizontal_shift"] * hs_amount,
                                 (p["head_switching_mid_line_position"], p["head_switching_mid_line_jitter"] * tracking))
    if p["tracking_noise_enabled"]:
        cfg["tracking_noise"] = (round_half_up(p["tracking_noise_height"] * tracking),
                                 p["tracking_noise_wave_intensity"] * tracking,
                                 max(0.0, p["tracking_noise_snow_intensity"] * tracking),
                                 p["tracking_noise_snow_anisotropy"],
                                 max(0.0, p["tracking_noise_noise_intensity"] * tracking))
    if p["composite_noise_enabled"]:
        cfg["composite_noise"] = (p["composite_noise_frequency"], p["composite_noise_intensity"] * noise,
                                  int(p["composite_noise_detail"]))
    if p["ringing_enabled"]:
        cfg["ringing"] = (p["ringing_frequency"], p["ringing_power"], ringing)
    if p["luma_noise_enabled"]:
        cfg["luma_noise"] = (p["luma_noise_frequency"], p["luma_noise_intensity"] * noise, int(p["luma_noise_detail"]))
    if p["chroma_noise_enabled"]:
        cfg["chroma_noise"] = (p["chroma_noise_frequency"], p["chroma_noise_intensity"] * chroma_bleed,
                               int(p["chroma_noise_detail"]))
    if p["vhs_enabled"]:
        speed = TAPE_SPEED.index(p["vhs_tape_speed"]) if tape_wear > 0.03 else 0
        cfg["vhs"] = (speed,
                      min(max(p["vhs_chroma_loss"] * tape_wear, 0.0), 1.0),
                      (p["vhs_sharpen_intensity"] * amount, p["vhs_sharpen_frequency"]),
                      (p["vhs_edge_wave_intensity"] * tape_wear, p["vhs_edge_wave_speed"],
                       p["vhs_edge_wave_frequency"], int(p["vhs_edge_wave_detail"])))
    if p["scale_enabled"]:
        cfg["scale"] = (max(p["scale_horizontal"], 0.001), max(p["scale_vertical"], 0.001))
    return cfg


def _chroma_phase_shift(phase_shift: int, offset: int, frame_num: int, line_num: int) -> int:
    if phase_shift in (1, 3):  # 90 / 270
        return (frame_num + offset + (line_num >> 1)) & 3
    if phase_shift == 2:  # 180
        return (((frame_num + line_num) & 2) + offset) & 3
    return 0


def _xi_rows(cfg: dict, frame_num: int, rows: int) -> np.ndarray:
    return np.array([_chroma_phase_shift(cfg["phase_shift"], cfg["phase_shift_offset"], frame_num, r * 2)
                     for r in range(rows)], np.int64)


def _box4(src: np.ndarray) -> np.ndarray:
    """Mean of (x[k-1] or 16/255, x[k], x[k+1], x[k+2]) with the right edge clamped."""
    rows, w = src.shape
    pad = np.empty((rows, w + 3), np.float32)
    pad[:, 0] = np.float32(16.0 / 255.0)
    pad[:, 1:w + 1] = src
    pad[:, w + 1] = src[:, -1]
    pad[:, w + 2] = src[:, -1]
    return ((pad[:, 0:w] + pad[:, 1:w + 1] + pad[:, 2:w + 2] + pad[:, 3:w + 3]) * np.float32(0.25)).astype(np.float32)


def _composite_chroma_lowpass(i, q, mode: int, filter_type: int, hs) -> None:
    rate = NTSC_RATE * np.float32(hs)
    if mode == 2:
        make_lowpass_for_type(1300000.0, rate, filter_type).apply(i, False, 2)
        make_lowpass_for_type(600000.0, rate, filter_type).apply(q, False, 4)
    elif mode == 1:
        f = make_lowpass_for_type(2600000.0, rate, filter_type)
        f.apply(i, False, 1)
        f.apply(q, False, 1)


def _head_switching(y, scratch, seed, frame_num, cfg_hs, hs, vs) -> None:
    height_s, offset_s, horiz_shift, mid_line = cfg_hs
    num_rows = round_half_away(max(0, height_s) * vs)
    offset = round_half_away(max(0, offset_s) * vs)
    if offset > num_rows:
        return
    n_aff = num_rows - offset
    height, width = y.shape
    start = max(height, n_aff) - n_aff
    cut = max(n_aff - height, 0)
    prefix = seeder((seed, 8), (_NS_HEAD_SWITCHING, 8), (frame_num, 8))
    hs32 = np.float32(hs)
    for k, r in enumerate(range(start, height)):
        index = n_aff - (k + cut)
        row_shift = np.float32(horiz_shift) * np.float32(np.float32(index + offset) / np.float32(num_rows)) ** np.float32(1.5)
        noisy = (row_shift + (seed_to_f32(seeder_finalize(prefix, index, 8)) - np.float32(0.5))) * hs32
        if index == n_aff and mid_line is not None:
            _shift_into(y[r].copy(), scratch, np.float32(noisy), False, np.float32(0.0))
            p2 = seeder((seed, 8), (_NS_MID_LINE_JITTER, 8), (frame_num, 8))
            r0 = seed_to_f32(seeder_finalize(p2, 0, 4))
            r1 = seed_to_f32(seeder_finalize(p2, 1, 4))
            jitter = ((r0 + r1) * np.float32(0.5) - np.float32(0.5)) * np.float32(mid_line[1])
            start_f = np.float32(width) * (np.float32(mid_line[0]) + jitter)
            copy_start = max(0, int(start_f)) if np.isfinite(start_f) else 0
            if copy_start > width:
                continue
            y[r, copy_start:] = scratch[copy_start:]
            t_int = (r0 + np.float32(0.5)) * np.float32(0.5)
            t_len = np.float32(16.0) * hs32
            stop = min(copy_start + int(math.ceil(t_len)), width)
            if stop > copy_start:
                x = np.arange(stop - copy_start, dtype=np.float32)
                y[r, copy_start:stop] += (np.float32(1.0) - x / t_len) ** 3 * t_int
        else:
            _shift_inplace(y[r], np.float32(noisy), False, np.float32(0.0))


def _tracking_noise(y, seed, frame_num, cfg_tn, hs, vs) -> None:
    height_s, wave, snow_int, snow_anis, noise_int = cfg_tn
    num_rows = round_half_away(max(0, height_s) * vs)
    height, width = y.shape
    prefix = seeder((seed, 8), (_NS_TRACKING_NOISE, 8), (frame_num, 8))
    noise_seed = seed_to_i32(seeder_finalize(prefix, 0, 4))
    offset = seed_to_f32(seeder_finalize(prefix, 1, 4)) * np.float32(height)
    p2 = seeder((seed, 8), (_NS_TRACKING_NOISE, 8), (frame_num, 8), (2, 4))
    start = max(height, num_rows) - num_rows
    cut = max(num_rows - height, 0)
    n = min(num_rows, height)
    hs32 = np.float32(hs)
    s = np.empty(4, np.uint64)
    for k in range(n):
        row = y[start + k]
        index = k + cut
        scale = np.float32(index) / np.float32(num_rows)
        sn = _simplex1d((offset + np.float32(k)) * np.float32(0.5), noise_seed)
        _shift_inplace(row, np.float32(sn * scale * np.float32(wave) * np.float32(0.25) * hs32), False, np.float32(0.0))
        _video_noise_line(row, p2, index, np.float32(0.25) / hs32,
                          np.float32(scale * scale * np.float32(noise_int) * np.float32(4.0)), 1)
        _xo_seed(np.uint64(seeder_finalize(p2, index, 8)), s)
        _row_speckles(row, s, np.float32(np.float32(snow_int) * scale * scale), np.float32(snow_anis), hs32)


def _luma_into_chroma(y, i, q, cfg, frame_num) -> None:
    rows, width = y.shape
    modulated = y.copy()
    mode = cfg["chroma_demodulation"]
    if rows == 1 and mode in (2, 3):
        mode = 1
    if mode == 0:
        y[:] = _box4(modulated)
    elif mode == 1:
        make_notch(0.5, 2.0).apply(y, False, 0)
    elif mode == 2:
        top = np.concatenate([modulated[1:2], modulated[:-1]], axis=0)
        y[:] = (top + modulated) * np.float32(0.5)
    else:
        prev = np.concatenate([modulated[1:2], modulated[:-1]], axis=0)
        nxt = np.concatenate([modulated[1:], modulated[rows - 2:rows - 1]], axis=0)
        y[:] = modulated * np.float32(0.5) + prev * np.float32(0.25) + nxt * np.float32(0.25)
    _demodulate(y, i, q, modulated, _xi_rows(cfg, frame_num, rows))


def _vhs(y, i, q, cfg, info) -> None:
    speed, chroma_loss, sharpen, edge_wave = cfg["vhs"]
    rows, width = y.shape
    hs, vs, seed, frame_num = info["hs"], info["vs"], info["seed"], info["frame"]
    hs32 = np.float32(hs)
    if edge_wave is not None and edge_wave[0] > 0.0:
        e_int, e_speed, e_freq, e_detail = edge_wave
        prefix = seeder((seed, 8), (_NS_EDGE_WAVE, 8))
        nseed = seed_to_i32(seeder_finalize(prefix, 0, 4))
        offset = seed_to_f32(seeder_finalize(prefix, 1, 4)) * np.float32(rows)
        octaves = min(max(e_detail, 1), 5)
        ycoord = np.float32(frame_num) * np.float32(e_speed)
        freq = np.float32(e_freq) / np.float32(vs)
        noise = np.array([_fbm2d(offset + np.float32(k), ycoord, nseed, octaves, np.float32(1.0 / math.sqrt(2.0)),
                                 np.float32(2.0), freq) for k in range(rows)], np.float32)
        shifts = (noise / np.float32(0.022)) * np.float32(e_int) * np.float32(0.5) * hs32
        for plane in (y, i, q):
            _shift_rows(plane, shifts.astype(np.float32), True)
    tape = _TAPE_PARAMS.get(speed)
    ftype = cfg["filter_type"]
    rate = NTSC_RATE * hs32
    if tape is not None:
        luma_cut, chroma_cut, cdelay = tape
        cdelay = max(0, round_half_away(cdelay * hs))
        make_lowpass_for_type(luma_cut, rate, ftype).apply(y, False, 0)
        cf = make_lowpass_for_type(chroma_cut, rate, ftype)
        cf.apply(i, False, cdelay)
        cf.apply(q, False, cdelay)
        make_lowpass(luma_cut, rate).with_scale(-1.6).apply(y, False, 0)
    if chroma_loss > 0.0:
        s = np.empty(4, np.uint64)
        _xo_seed(np.uint64(seeder_finalize(seeder((seed, 8), (_NS_CHROMA_LOSS, 8), (frame_num, 8)))), s)
        lam = math.log(1.0 - chroma_loss) if chroma_loss < 1.0 else -math.inf
        r = 0
        while True:
            r += int(_geometric(s, lam))
            if r >= rows:
                break
            i[r] = 0.0
            q[r] = 0.0
            r += 1
    if sharpen is not None and tape is not None:
        mult = 4.0 if ftype == 0 else 1.0
        sh_int, sh_freq = sharpen
        make_lowpass_for_type(np.float32(tape[0]) * np.float32(mult) * np.float32(sh_freq), rate, ftype) \
            .with_scale(np.float32(-sh_int) * np.float32(2.0) * np.float32(sh_freq)).apply(y, False, 0)


def _ntsc_field(y, i, q, cfg, frame_num, hs, vs, full_height) -> None:
    """``NtscEffect::apply_effect_to_yiq_field`` on one (packed) field view."""
    rows, width = y.shape
    if rows == 0 or width == 0:
        return
    seed = cfg["seed"] & 0xFFFFFFFF
    hs32, vs32 = np.float32(hs), np.float32(vs)
    info = {"hs": hs, "vs": vs, "seed": seed, "frame": frame_num}
    scratch = np.zeros(width, np.float32)

    lf = cfg["input_luma_filter"]
    if lf == 1 and width >= 2:
        y[:] = _box4(y)
    elif lf == 2:
        make_notch(0.5, 2.0).apply(y, True, 0)
    _composite_chroma_lowpass(i, q, cfg["chroma_lowpass_in"], cfg["filter_type"], hs)

    _chroma_into_luma(y, i, q, _xi_rows(cfg, frame_num, rows))

    if cfg["composite_sharpening"] != 0.0:
        make_lowpass(np.float32(315000000.0 / 88.0 / 2.0) * hs32, NTSC_RATE * hs32) \
            .with_scale(-np.float32(cfg["composite_sharpening"])).apply(y, False, 0)
    if cfg["composite_noise"] is not None:
        freq, inten, detail = cfg["composite_noise"]
        _plane_noise(y, seeder((seed, 8), (_NS_VIDEO_COMPOSITE, 8), (frame_num, 8)),
                     np.float32(freq) / hs32, np.float32(inten), min(max(detail, 1), 5))
    if cfg["snow_intensity"] > 0.0 and hs > 0.0:
        _snow_rows(y, seeder((seed, 8), (_NS_SNOW, 8), (frame_num, 8)),
                   np.float32(np.float32(cfg["snow_intensity"]) * np.float32(0.01)),
                   np.float32(cfg["snow_anisotropy"]), hs32)
    if cfg["head_switching"] is not None:
        _head_switching(y, scratch, seed, frame_num, cfg["head_switching"], hs, vs)
    if cfg["tracking_noise"] is not None:
        _tracking_noise(y, seed, frame_num, cfg["tracking_noise"], hs, vs)

    _luma_into_chroma(y, i, q, cfg, frame_num)

    if cfg["luma_smear"] > 0.0:
        make_lowpass(np.float32(2.0) ** (np.float32(-4.0) * np.float32(cfg["luma_smear"])) * np.float32(0.25),
                     hs32).apply(y, False, 0)
    if cfg["ringing"] is not None:
        rf, rp, ri = cfg["ringing"]
        make_notch(min(max(np.float32(rf) / hs32, 0.0), 1.0), rp).with_scale(ri).apply(y, True, 1)
    if cfg["luma_noise"] is not None:
        freq, inten, detail = cfg["luma_noise"]
        _plane_noise(y, seeder((seed, 8), (_NS_LUMA, 8), (frame_num, 8)),
                     np.float32(freq) / hs32, np.float32(inten), min(max(detail, 1), 5))
    if cfg["chroma_noise"] is not None:
        freq, inten, detail = cfg["chroma_noise"]
        octaves = min(max(detail, 1), 5)
        _plane_noise(i, seeder((seed, 8), (_NS_CHROMA_I, 8), (frame_num, 8)),
                     np.float32(freq) / hs32, np.float32(inten), octaves)
        _plane_noise(q, seeder((seed, 8), (_NS_CHROMA_Q, 8), (frame_num, 8)),
                     np.float32(freq) / hs32, np.float32(inten), octaves)
    if cfg["chroma_phase_error"] > 0.0:
        phase = np.float32(cfg["chroma_phase_error"]) * _PI32 * np.float32(2.0)
        sa, ca = np.float32(math.sin(phase)), np.float32(math.cos(phase))
        i0 = i.copy()
        i[:] = i0 * ca - q * sa
        q[:] = i0 * sa + q * ca
    if cfg["chroma_phase_noise"] > 0.0:
        _chroma_phase_noise_rows(i, q, seeder((seed, 8), (_NS_CHROMA_PHASE, 8), (frame_num, 8)),
                                 np.float32(cfg["chroma_phase_noise"]))
    if cfg["chroma_delay_h"] != 0.0 or cfg["chroma_delay_v"] != 0:
        ho = np.float32(cfg["chroma_delay_h"]) * hs32
        vo = round_half_away(np.float32(cfg["chroma_delay_v"]) * vs32)
        for plane in (i, q):
            if vo != 0:
                moved = np.zeros_like(plane)
                if vo < 0 and -vo < rows:
                    moved[:rows + vo] = plane[-vo:]
                elif 0 < vo < rows:
                    moved[vo:] = plane[:rows - vo]
                plane[:] = moved
            if ho != 0.0:
                _shift_rows(plane, np.full(rows, ho, np.float32), False)
    if cfg["vhs"] is not None:
        _vhs(y, i, q, cfg, info)
    if cfg["chroma_vert_blend"] and full_height >= 2:
        _vert_blend(i)
        _vert_blend(q)
    _composite_chroma_lowpass(i, q, cfg["chroma_lowpass_out"], cfg["filter_type"], hs)


_RGB2YIQ = np.array([[0.299, 0.587, 0.114], [0.5959, -0.2746, -0.3213], [0.2115, -0.5227, 0.3112]], np.float32)
_YIQ2RGB = np.array([[1.0, 0.956, 0.619], [1.0, -0.272, -0.647], [1.0, -1.106, 1.703]], np.float32)


@njit(cache=True, parallel=True)
def _to_yiq_planes(rgb, r0, step, n, m):
    """Rows r0, r0+step, ... (n of them) of an RGB image -> Y, I, Q planes."""
    w = rgb.shape[1]
    y = np.empty((n, w), np.float32)
    i = np.empty((n, w), np.float32)
    q = np.empty((n, w), np.float32)
    for k in prange(n):
        r = r0 + k * step
        for x in range(w):
            cr, cg, cb = rgb[r, x, 0], rgb[r, x, 1], rgb[r, x, 2]
            y[k, x] = m[0, 0] * cr + m[0, 1] * cg + m[0, 2] * cb
            i[k, x] = m[1, 0] * cr + m[1, 1] * cg + m[1, 2] * cb
            q[k, x] = m[2, 0] * cr + m[2, 1] * cg + m[2, 2] * cb
    return y, i, q


@njit(cache=True, parallel=True)
def _from_yiq_rows(out, src_rows, y, i, q, src_rows_b, interp, m):
    """out[r] = YIQ->RGB of plane row src_rows[r] (averaged with src_rows_b[r] when interp[r])."""
    h, w = out.shape[0], out.shape[1]
    for r in prange(h):
        a = src_rows[r]
        b = src_rows_b[r]
        for x in range(w):
            yy, ii, qq = y[a, x], i[a, x], q[a, x]
            if interp[r]:
                yy = (yy + y[b, x]) * np.float32(0.5)
                ii = (ii + i[b, x]) * np.float32(0.5)
                qq = (qq + q[b, x]) * np.float32(0.5)
            out[r, x, 0] = m[0, 0] * yy + m[0, 1] * ii + m[0, 2] * qq
            out[r, x, 1] = m[1, 0] * yy + m[1, 1] * ii + m[1, 2] * qq
            out[r, x, 2] = m[2, 0] * yy + m[2, 1] * ii + m[2, 2] * qq


def ntsc_apply(rgb: np.ndarray, cfg: dict, scale_factor_x: float, scale_factor_y: float,
               frame_num: int = 0, preview_scale: float = 1.0) -> np.ndarray:
    """``NtscEffect::apply_effect_to_buffer`` (Rgbx/u8 in ntsc_jni); returns unclamped RGB.

    ``preview_scale`` (render scale of a preview proxy) multiplies ntsc-rs's
    bandwidth scales, like its "scale with video size" option, so that
    previews look like the full-resolution export.
    """
    h, w = rgb.shape[:2]
    src = np.ascontiguousarray(rgb[..., :3], dtype=np.float32)
    if cfg["scale"] is not None:
        hs = float(np.float32(cfg["scale"][0]) * np.float32(scale_factor_x))
        vs = float(np.float32(cfg["scale"][1]) * np.float32(scale_factor_y))
    else:
        hs = vs = 1.0
    hs, vs = hs * float(preview_scale), vs * float(preview_scale)
    uf = cfg["use_field"]
    field = (_LOWER if frame_num % 2 == 0 else _UPPER) if uf == 0 else uf

    def run(r0: int, step: int, n: int, frame: int):
        y, i, q = _to_yiq_planes(src, r0, step, n, _RGB2YIQ)
        _ntsc_field(y, i, q, cfg, frame, hs, vs, h)
        return y, i, q

    rows = np.arange(h, dtype=np.int64)
    interp = np.zeros(h, np.bool_)
    if field == _BOTH:
        y, i, q = run(0, 1, h, frame_num)
        src_a = rows
    elif field in (_UPPER, _LOWER):
        if field == _UPPER:
            n, skip = (h + 1) // 2, 1
            y, i, q = run(0, 2, n, frame_num)
        else:
            n, skip = max(h // 2, 1), 0
            y, i, q = run(1, 2, n, frame_num) if h >= 2 else run(0, 1, 1, frame_num)
        # DeinterlaceMode::Bob: missing field lines are the mean of their neighbours
        interp = ((rows & 1) == skip) & (rows != 0) & (rows != h - 1)
        src_a = np.where(interp, (rows - 1) >> 1, np.minimum(rows >> 1, n - 1))
    else:  # interleaved: both fields, each processed as its own frame
        nu, nl = (h + 1) // 2, h // 2
        fu, fl = (frame_num * 2, frame_num * 2 + 1) if field == 4 else (frame_num * 2 + 1, frame_num * 2)
        y, i, q = run(0, 2, nu, fu)
        if nl:
            yl, il, ql = run(1, 2, nl, fl)
            y, i, q = np.concatenate([y, yl]), np.concatenate([i, il]), np.concatenate([q, ql])
        src_a = np.where(rows & 1, nu + (rows >> 1), rows >> 1)
    src_b = np.where(interp, (rows + 1) >> 1, src_a)
    out = np.empty((h, w, 3), np.float32)
    _from_yiq_rows(out, src_a, y, i, q, src_b, interp, _YIQ2RGB)
    return out  # u8 output clamps to 0..1; the itb_op wrapper performs that clamp
