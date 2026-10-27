"""Private implementations for the ImageToolbox "Simple effects" group.

Faithful ports of the algorithms ImageToolbox calls for these filters:
GPUImage fragment shaders (BSD-3), aire (MIT, incl. its libblur /
histogram_equalization / yuv Rust dependencies), Trickle (Apache-2.0),
the jhlabs Rust port in ImageToolboxLibs (Apache-2.0) and ImageToolbox's
``opencv-tools`` (Apache-2.0).

Buffers are float32 0..1.  Android bitmaps are premultiplied, so filters
that force an opaque result operate on the premultiplied colour (see
:func:`bitmap_rgb`), exactly like upstream does.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from numba import njit, prange

from .._util import F32

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def to_u8(x: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def bitmap_rgb(img: np.ndarray) -> np.ndarray:
    """RGB as an Android ARGB_8888 bitmap stores it (premultiplied by alpha)."""
    if img.shape[2] == 4:
        return (img[..., :3] * img[..., 3:4]).astype(F32)
    return np.ascontiguousarray(img[..., :3], dtype=F32)


def opaque(img: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Upstream writes alpha = 255: attach an opaque alpha when the input had one."""
    if img.shape[2] == 4:
        return np.dstack([rgb[..., :3], np.ones(img.shape[:2], F32)]).astype(F32)
    return np.ascontiguousarray(rgb[..., :3], dtype=F32)


def gray3(g: np.ndarray) -> np.ndarray:
    return np.repeat(g[..., None], 3, axis=2).astype(F32)


def _neighbours(a: np.ndarray, border: int = cv2.BORDER_REPLICATE) -> dict[str, np.ndarray]:
    """3x3 neighbourhood views (image coordinates, row 0 = top)."""
    p = cv2.copyMakeBorder(a, 1, 1, 1, 1, border)
    h, w = a.shape[:2]
    sl = lambda dy, dx: p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]  # noqa: E731
    return {"tl": sl(-1, -1), "t": sl(-1, 0), "tr": sl(-1, 1), "l": sl(0, -1), "c": a, "r": sl(0, 1),
            "bl": sl(1, -1), "b": sl(1, 0), "br": sl(1, 1)}


# ---------------------------------------------------------------------------
# GPUImage shaders
# ---------------------------------------------------------------------------

GPU_LUMINANCE_W = np.array([0.2125, 0.7154, 0.0721], F32)


def gpu_cga_colorspace(rgb: np.ndarray) -> np.ndarray:
    """GPUImageCGAColorspaceFilter: sample a 200x320 grid (GL_LINEAR) and snap to the CGA palette."""
    h, w = rgb.shape[:2]
    # samplePos = tc - mod(tc, 1/200|1/320) with tc = (i + 0.5) / size -> grid cell index
    kx = ((2 * np.arange(w, dtype=np.int64) + 1) * 100) // w
    ky = ((2 * np.arange(h, dtype=np.int64) + 1) * 160) // h
    ux, inv_x = np.unique(kx, return_inverse=True)
    uy, inv_y = np.unique(ky, return_inverse=True)

    def lin_idx(k: np.ndarray, cells: int, size: int):
        t = k / cells * size - 0.5          # GL_LINEAR texel coordinate
        i0 = np.floor(t).astype(np.int64)
        f = (t - i0).astype(F32)
        return np.clip(i0, 0, size - 1), np.clip(i0 + 1, 0, size - 1), f

    x0, x1, fx = lin_idx(ux, 200, w)
    y0, y1, fy = lin_idx(uy, 320, h)
    rows = rgb[y0] * (1 - fy)[:, None, None] + rgb[y1] * fy[:, None, None]
    grid = rows[:, x0] * (1 - fx)[None, :, None] + rows[:, x1] * fx[None, :, None]
    # palette in the shader's tie-break order: black, white, cyan, magenta
    pal = np.array([[0, 0, 0], [1, 1, 1], [85 / 255, 1, 1], [1, 85 / 255, 1]], F32)
    d = ((grid[:, :, None, :] - pal[None, None]) ** 2).sum(-1)  # alpha term is equal for all four
    small = pal[np.argmin(d, axis=-1)]
    # kx / ky are non-decreasing: expand the cell grid with run lengths instead of fancy indexing
    return np.repeat(np.repeat(small, np.bincount(inv_y), axis=0), np.bincount(inv_x), axis=1)


def gpu_laplacian(rgb: np.ndarray) -> np.ndarray:
    """GPUImageLaplacianFilter: 3x3 kernel (0.5 1 0.5 / 1 -6 1 / 0.5 1 0.5) + 0.5."""
    k = np.array([[0.5, 1, 0.5], [1, -6, 1], [0.5, 1, 0.5]], F32)
    return cv2.filter2D(rgb, -1, k, borderType=cv2.BORDER_REPLICATE) + 0.5


def gpu_non_maximum_suppression(rgb: np.ndarray) -> np.ndarray:
    """GPUImageNonMaximumSuppressionFilter (red channel test, colour kept where it is a local max)."""
    n = _neighbours(np.ascontiguousarray(rgb[..., 0]))
    c = n["c"]
    # shader: multiplier = multiplier * 1.0 - step(c, x) (precedence => plain subtraction), so the
    # multiplier is 1 - (#of top/top-left/left/bottom-left >= c) and only == 1 gives a positive
    # (unclamped) result; out = rgb * step(maxValue, c) * multiplier.
    keep = (n["t"] < c) & (n["tl"] < c) & (n["l"] < c) & (n["bl"] < c)
    keep &= c >= np.maximum(np.maximum(n["b"], n["br"]), np.maximum(n["r"], n["tr"]))
    return np.where(keep[..., None], rgb, np.float32(0.0)).astype(F32)


def gpu_weak_pixel_inclusion(rgb: np.ndarray) -> np.ndarray:
    """GPUImageWeakPixelInclusionFilter: on if 3x3 red sum >= 1.5 and centre >= 0.01."""
    r = np.ascontiguousarray(rgb[..., 0])
    s = cv2.boxFilter(r, -1, (3, 3), normalize=False, borderType=cv2.BORDER_REPLICATE)
    return gray3(((s >= 1.5) & (r >= 0.01)).astype(F32))


# ---------------------------------------------------------------------------
# aire
# ---------------------------------------------------------------------------

def _aire_gray8(rgb: np.ndarray) -> np.ndarray:
    """aire's fixed-point Rec.709 luma: (218 R + 733 G + 74 B) >> 10."""
    q = to_u8(rgb).astype(np.int32)
    return np.minimum((q[..., 0] * 218 + q[..., 1] * 733 + q[..., 2] * 74) >> 10, 255).astype(F32)


def aire_laplacian(rgb: np.ndarray) -> np.ndarray:
    """aire laplacian: libblur 3x3 (-1.. 8 ..-1) on 8-bit luma, REFLECT_101, saturated."""
    k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], F32)
    out = cv2.filter2D(_aire_gray8(rgb), -1, k, borderType=cv2.BORDER_REFLECT_101)
    return gray3(np.clip(out, 0, 255) / 255.0)


def aire_sobel(rgb: np.ndarray) -> np.ndarray:
    """aire sobel: libblur separable [-1 0 1] (rows) then [1 2 1] (columns), each pass saturated to u8."""
    g = _aire_gray8(rgb)
    gx = cv2.filter2D(g, -1, np.array([[-1, 0, 1]], F32), borderType=cv2.BORDER_REFLECT_101)
    gx = np.clip(gx, 0, 255)
    gy = cv2.filter2D(gx, -1, np.array([[1], [2], [1]], F32), borderType=cv2.BORDER_REFLECT_101)
    return gray3(np.clip(gy, 0, 255) / 255.0)


def aire_equalize_hist(rgb: np.ndarray) -> np.ndarray:
    """histogram_equalization::hist_equal_yuv_rgba: equalise Y of full-range YCgCo (256 bins)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.5 * g + 0.25 * (r + b)
    cg = 0.5 * g - 0.25 * (r + b)
    co = 0.5 * (r - b)
    # yuv crate: (v * 2^13 + 2^12 - 1) >> 13 on 8-bit values
    y8 = np.floor(np.clip(y, 0, 1) * 255.0 + 4095 / 8192).astype(np.int64)
    y8 = np.clip(y8, 0, 255)
    bins = np.cumsum(np.bincount(y8.ravel(), minlength=256)).astype(np.float64)
    n = float(y8.size)
    lo = bins.min()
    if n - lo != 0:
        lut = np.minimum(np.floor(255.0 * (bins - lo) / (n - lo) + 0.5), 255.0)
    else:  # pragma: no cover - single-valued image
        lut = bins
    yn = (lut[y8] / 255.0).astype(F32)
    t = yn - cg
    return np.stack([t + co, yn + cg, t - co], axis=-1).astype(F32)


@njit(cache=True, inline="always")
def _clamp01(v):
    return min(max(v, 0.0), 1.0)


@njit(cache=True, inline="always")
def _aire_srgb_to_linear(v):
    """aire SRGBToLinear (note its 0.045 knee)."""
    if v <= 0.045:
        return v / 12.92
    return ((v + 0.055) / 1.055) ** 2.4


@njit(cache=True, inline="always")
def _aire_linear_to_srgb(v):
    if v <= 0.0031308:
        return 12.92 * v
    return 1.055 * v ** (1.0 / 2.4) - 0.055


@njit(cache=True, inline="always")
def _aire_mono(lin, L, col, intensity):
    """aire MonochromeToneMapper::applyMono + mix + clamp (linear light)."""
    new = 2.0 * col * L if L < 0.5 else 1.0 - 2.0 * (1.0 - L) * (1.0 - col)
    return _clamp01((1.0 - intensity) * lin + intensity * new)


@njit(cache=True, parallel=True)
def _old_tv_kernel(img, noise, out, distance, slope, cr, cg, cb, intensity):
    h, w, nc = img.shape
    for y in prange(h):
        d = (y + 0.5) / h * slope + distance          # GPUImageHaze: textureCoordinate.y * slope + distance
        inv = 1.0 / (1.0 - d)
        for x in range(w):
            n = noise[y, x]
            # aire grain: same truncated gaussian value added to R, G, B
            r = _clamp01(img[y, x, 0] + n)
            g = _clamp01(img[y, x, 1] + n)
            b = _clamp01(img[y, x, 2] + n)
            # GPUImage haze (all four channels, colour = white), framebuffer clamp
            r = _clamp01((r - d) * inv)
            g = _clamp01((g - d) * inv)
            b = _clamp01((b - d) * inv)
            if nc == 4:
                out[y, x, 3] = _clamp01((img[y, x, 3] - d) * inv)
            # aire monochrome (exposure 1)
            lr = _aire_srgb_to_linear(r)
            lg = _aire_srgb_to_linear(g)
            lb = _aire_srgb_to_linear(b)
            L = 0.299 * lr + 0.587 * lg + 0.114 * lb
            out[y, x, 0] = _clamp01(_aire_linear_to_srgb(_aire_mono(lr, L, cr, intensity)))
            out[y, x, 1] = _clamp01(_aire_linear_to_srgb(_aire_mono(lg, L, cg, intensity)))
            out[y, x, 2] = _clamp01(_aire_linear_to_srgb(_aire_mono(lb, L, cb, intensity)))


def old_tv(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """ImageToolbox OldTvFilter chain: aire grain(0.36) -> GPUImage haze(distance 0, slope 0.3)
    -> aire monochrome(intensity 1, colour #1C3A00).

    Grain: ``int(N(0, 127 * intensity))`` per pixel (aire seeds from the clock;
    here the render seed is used).
    """
    a = np.ascontiguousarray(img, dtype=F32)
    noise = rng.standard_normal(a.shape[:2], dtype=np.float32)
    noise *= np.float32(127.0 * 0.36)
    np.trunc(noise, out=noise)
    noise *= np.float32(1.0 / 255.0)
    out = np.empty_like(a)
    _old_tv_kernel(a, noise, out, 0.0, 0.3, 0x1C / 255.0, 0x3A / 255.0, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# jhlabs (ImageToolboxLibs Rust port)
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _pepper_and_salt(c, v1, v2):
    eps = 1e-3  # integer comparisons on 8-bit data, robust to float rounding
    if c < v1 - eps:
        c += 1.0
    if c < v2 - eps:
        c += 1.0
    if c > v1 + eps:
        c -= 1.0
    if c > v2 + eps:
        c -= 1.0
    return c


@njit(cache=True, parallel=True)
def _despeckle_kernel(a, out):
    h, w, nc = a.shape
    for y in prange(h):
        iy = 0 < y < h - 1
        for x in range(w):
            ix = 0 < x < w - 1
            for ch in range(nc):
                c = a[y, x, ch]
                if iy:
                    c = _pepper_and_salt(c, a[y - 1, x, ch], a[y + 1, x, ch])
                if ix:
                    c = _pepper_and_salt(c, a[y, x - 1, ch], a[y, x + 1, ch])
                if ix and iy:
                    c = _pepper_and_salt(c, a[y - 1, x - 1, ch], a[y + 1, x + 1, ch])
                    c = _pepper_and_salt(c, a[y + 1, x - 1, ch], a[y - 1, x + 1, ch])
                out[y, x, ch] = c


def jh_despeckle(rgb: np.ndarray) -> np.ndarray:
    """despeckle_exact: pepper-and-salt +-1 (8-bit units) nudges against 4 neighbour pairs."""
    a = np.ascontiguousarray(rgb, dtype=F32) * np.float32(255.0)
    out = np.empty_like(a)
    _despeckle_kernel(a, out)
    out *= np.float32(1.0 / 255.0)
    return out


def jh_reduce_noise(rgb: np.ndarray) -> np.ndarray:
    """reduce_noise_exact: clamp each channel to the min/max of its 8 neighbours.

    Out-of-image neighbours are replaced by the centre, so border pixels never change.
    """
    ring = np.ones((3, 3), np.uint8)
    ring[1, 1] = 0
    lo = cv2.erode(rgb, ring, borderType=cv2.BORDER_REPLICATE)
    hi = cv2.dilate(rgb, ring, borderType=cv2.BORDER_REPLICATE)
    out = np.clip(rgb, lo, hi)
    out[0], out[-1], out[:, 0], out[:, -1] = rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]
    return out.astype(F32)


def jh_equalize(rgb: np.ndarray) -> np.ndarray:
    """EqualizeFilter: independent per-channel histogram equalisation (8-bit histograms)."""
    q = to_u8(rgb)
    n = max(1, q.shape[0] * q.shape[1])
    out = np.empty(rgb.shape, F32)
    for ch in range(3):
        cdf = np.cumsum(np.bincount(q[..., ch].ravel(), minlength=256)).astype(np.float64)
        lut = np.floor(cdf * 255.0 / n + 0.5) / 255.0
        out[..., ch] = lut[q[..., ch]]
    return out


def jh_java_lnf(img: np.ndarray) -> np.ndarray:
    """JavaLnFFilter: mix 75% of opaque 0x999999 into every other pixel (checkerboard).

    The mix runs on Android's premultiplied pixels, alpha included.
    """
    h, w = img.shape[:2]
    yy, xx = np.ogrid[0:h, 0:w]
    sel = ((xx & 1) != (yy & 1))[..., None]
    grey = 153.0 / 255.0
    if img.shape[2] == 3:
        return np.where(sel, 0.75 * grey + 0.25 * img, img).astype(F32)
    a = img[..., 3:4]
    na = 0.75 + 0.25 * a
    rgb = (0.75 * grey + 0.25 * img[..., :3] * a) / na
    out = np.where(sel, np.concatenate([rgb, na], axis=-1), img)
    return out.astype(F32)


# ---------------------------------------------------------------------------
# Trickle
# ---------------------------------------------------------------------------
# Trickle reads Android's RGBA bytes as little-endian ints and then calls the
# blue byte "R()" and the red byte "B()"; the asymmetric parts below are
# written in real RGB order with that swap resolved.

def trickle_hdr(rgb: np.ndarray) -> np.ndarray:
    """HDRFilter: 3x3 gaussian (sigma 0.6, interior only), overlay(orig, blur), then Laplacian sharpen."""
    s = 0.6
    ax = np.arange(3, dtype=np.float64) - 1
    k = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * s * s))
    k = (k / k.sum()).astype(F32)
    inner = rgb.shape[0] > 2 and rgb.shape[1] > 2
    blur = cv2.filter2D(rgb, -1, k)
    if inner:  # only interior pixels are blurred
        blur[0], blur[-1], blur[:, 0], blur[:, -1] = rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]
    else:
        blur = rgb.copy()
    # overlay: s <= .5 ? 2 s o : 1 - 2 (1 - o)(1 - s)  ==  2 (o + s) - 1 - 2 s o
    a = blur * rgb
    a *= np.float32(2.0)
    over = np.where(blur <= np.float32(0.5), a, (rgb + blur) * np.float32(2.0) - np.float32(1.0) - a)
    if not inner:
        return over
    lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], F32)
    edge = cv2.filter2D(over, -1, lap)
    np.clip(edge, 0.0, 1.0, out=edge)
    edge[0] = edge[-1] = 0.0
    edge[:, 0] = edge[:, -1] = 0.0
    over += edge
    return over


@njit(cache=True, parallel=True)
def _gotham_kernel(rgb, out, gain_g, gain_b):
    h, w, _ = rgb.shape
    for y in prange(h):
        for x in range(w):
            # setBrightness(-50), then GammaCorrectionFilter(0.7): v^(1/0.7)
            r = _clamp01(rgb[y, x, 0] - 50.0 / 255.0) ** (1.0 / 0.7)
            g = _clamp01(rgb[y, x, 1] - 50.0 / 255.0) ** (1.0 / 0.7)
            b = _clamp01(rgb[y, x, 2] - 50.0 / 255.0) ** (1.0 / 0.7)
            # setHueSaturationIntesity: HSI2RGB(233.69, 0.2, I); written only if checkRGB passes
            i = (r + g + b) / 3.0
            tb = gain_b * i
            if tb * 255.0 < 256.0:
                r, g, b = 0.8 * i, gain_g * i, tb
            # setContrast(1.2)
            out[y, x, 0] = _clamp01(128.0 / 255.0 + (r - 128.0 / 255.0) * 1.2)
            out[y, x, 1] = _clamp01(128.0 / 255.0 + (g - 128.0 / 255.0) * 1.2)
            out[y, x, 2] = _clamp01(128.0 / 255.0 + (b - 128.0 / 255.0) * 1.2)


def trickle_gotham(rgb: np.ndarray) -> np.ndarray:
    """GothamFilter: brightness -50, gamma 0.7, HSI re-tint (hue 233.69, sat 0.2), contrast 1.2.

    setHueSaturationIntesity overwrites hue and saturation and (upstream bug)
    ignores its intensity argument, so every pixel becomes HSI2RGB(233.69, 0.2,
    I) with its own intensity I: the h in [120, 240) branch gives code-space
    r = 0.8 I, g = I (1 + s cos h' / cos(60 - h')), b = 3 I - r - g.  It is
    written as RGB2Color(b, g, r), which Trickle's byte-swapped pixel access
    turns back into real (R, G, B) = (r, g, b).  Pixels whose b would exceed
    255 fail checkRGB and keep their colour.
    """
    hh = math.radians(233.69 - 120.0)
    gain_g = 1.0 + (0.2 * math.cos(hh)) / math.cos(math.radians(60.0) - hh)
    gain_b = 3.0 - 0.8 - gain_g
    src = np.ascontiguousarray(rgb, dtype=F32)
    out = np.empty_like(src)
    _gotham_kernel(src, out, gain_g, gain_b)
    return out


def trickle_tv(img: np.ndarray) -> np.ndarray:
    """TvFilter (gap 4): per 4-row group -> blue-average row, green-average row, black row, original row."""
    h, w = img.shape[:2]
    groups = (h + 3) // 4
    pad = groups * 4 - h
    out = np.array(img, dtype=F32, copy=True)
    prem = bitmap_rgb(img)
    for ch in (2, 1):  # code "r" = blue byte (row 0), green (row 1); code "b" is forced to 0 (row 2)
        q = to_u8(prem[..., ch]) // 4                        # integer division per pixel
        q = np.pad(q, ((0, pad), (0, 0)))                    # rows past the end are skipped
        s = q.reshape(groups, 4, w).sum(axis=1, dtype=np.uint16)
        v = np.minimum(s, 255).astype(F32) / np.float32(255.0)
        row = out[0::4] if ch == 2 else out[1::4]
        row[..., :3] = 0.0
        row[..., ch] = v[:row.shape[0]]
    out[2::4, :, :3] = 0.0
    if img.shape[2] == 4:
        out[0::4, :, 3] = out[1::4, :, 3] = out[2::4, :, 3] = 1.0
    return out


def trickle_sketch(rgb: np.ndarray) -> np.ndarray:
    """SketchFilter: grey (0.3 B + 0.59 G + 0.11 R, truncated), black where |c - bottom-right| >= 7."""
    q = to_u8(rgb).astype(np.float64)
    gray = np.trunc(((q[..., 2] * 0.3 + q[..., 1] * 0.59) + q[..., 0] * 0.11).astype(np.float32)).astype(F32)
    out = gray.copy()
    if gray.shape[0] > 2 and gray.shape[1] > 2:
        diff = np.abs(gray[1:-1, 1:-1] - gray[2:, 2:])
        out[1:-1, 1:-1] = np.where(diff >= 7, 0.0, 255.0)
    return gray3(out / 255.0)


# ---------------------------------------------------------------------------
# ImageToolbox opencv-tools
# ---------------------------------------------------------------------------

def cv_color_map(rgb: np.ndarray, code: int) -> np.ndarray:
    """ColorMap.apply: RGB -> 8-bit grey -> cv::applyColorMap."""
    g = cv2.cvtColor(to_u8(rgb), cv2.COLOR_RGB2GRAY)
    bgr = cv2.applyColorMap(g, code)
    return (bgr[..., ::-1].astype(F32) / 255.0)


def cv_average_distance(rgb: np.ndarray) -> np.ndarray:
    """ImageForensics.averageDistance: |x - mean of 4-neighbours|, min-max normalised over all channels."""
    k = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], F32)
    f = cv2.filter2D(rgb, -1, k)
    d = cv2.absdiff(rgb, f)
    return cv2.normalize(d, None, 0.0, 1.0, cv2.NORM_MINMAX)


def cv_luminance_gradient(rgb: np.ndarray) -> np.ndarray:
    """ImageForensics.luminanceGradient: R/G encode the Sobel direction, B the normalised magnitude."""
    g = cv2.cvtColor(to_u8(rgb), cv2.COLOR_RGB2GRAY).astype(F32)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    ang = np.arctan2(sx, sy)
    gch = -np.sin(ang) / 2.0 + 0.5
    rch = -np.cos(ang) / 2.0 + 0.5
    mag = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return np.stack([rch, gch, mag], axis=-1).astype(F32)


# -- Moire (opencv-tools moire/Moire.kt) ------------------------------------

def _shift_dft(a: np.ndarray) -> None:
    """In-place quadrant swap exactly like Moire.shiftDFT (odd last row/column untouched)."""
    rows, cols = a.shape[:2]
    cx, cy = cols // 2, rows // 2
    q0 = a[:cy, :cx].copy()
    a[:cy, :cx] = a[cy:2 * cy, cx:2 * cx]
    a[cy:2 * cy, cx:2 * cx] = q0
    q1 = a[:cy, cx:2 * cx].copy()
    a[:cy, cx:2 * cx] = a[cy:2 * cy, :cx]
    a[cy:2 * cy, :cx] = q1


def _log_mag_u8(cplx: np.ndarray) -> np.ndarray:
    """log(1 + |F|), min-max normalised to 0..255 and rounded to 8 bits (normalize + convertTo(CV_8U))."""
    mag = cv2.magnitude(cplx[..., 0], cplx[..., 1])
    mag += np.float32(1.0)
    cv2.log(mag, mag)
    return cv2.normalize(mag, None, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def _moire_detect(mag8: np.ndarray) -> str | None:
    rows, cols = mag8.shape
    ys, xs = np.nonzero(mag8 > 180)
    keep = (np.abs(xs - cols / 2.0) > 10) | (np.abs(ys - rows / 2.0) > 10)
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 2:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    _, ev = cv2.PCACompute(pts, mean=np.empty((0)))
    angle = math.degrees(math.atan2(ev[0, 1], ev[0, 0]))
    if abs(angle) < 15:
        return "vertical"
    if 70 < abs(angle) < 110:
        return "horizontal"
    if 15.0 <= angle <= 60.0:
        return "right_diagonal"
    if -60.0 <= angle <= -30.0:
        return "left_diagonal"
    return None


_DISK8 = (np.add.outer(np.arange(-8, 9) ** 2, np.arange(-8, 9) ** 2) <= 64).astype(np.uint8)


def _moire_mask(cplx: np.ndarray, mag8: np.ndarray, kind: str) -> None:
    rows, cols = mag8.shape
    crow, ccol = rows // 2, cols // 2
    ys, xs = np.nonzero(mag8 > 180)
    dx, dy = xs - ccol, ys - crow
    keep = (np.abs(dx) > 10) | (np.abs(dy) > 10)
    xs, ys, dx, dy = xs[keep], ys[keep], dx[keep], dy[keep]
    ang = np.degrees(np.arctan2(dy, dx))
    if kind == "vertical":
        hit = (np.abs(ang) < 15.0) | (np.abs(ang) > 165.0)
    elif kind == "horizontal":
        hit = (np.abs(ang - 90) < 15) | (np.abs(ang + 90) < 15)
    elif kind == "right_diagonal":
        hit = (np.abs(ang - 45) < 15) | (np.abs(ang + 135) < 15)
    else:
        hit = (np.abs(ang + 45) < 15) | (np.abs(ang - 135) < 15)
    if not hit.any():
        return
    m = np.zeros((rows, cols), np.uint8)
    m[ys[hit], xs[hit]] = 1
    m = cv2.dilate(m, _DISK8, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    cplx[m.astype(bool)] = 0.0


def cv_moire(rgb: np.ndarray, kind: str = "auto", check=None) -> np.ndarray:
    """Moire.remove(type=AUTO): per channel notch the detected periodic peaks + Butterworth low-pass.

    Deviation from upstream (intentional bug fix): ImageToolbox builds the
    Butterworth filter as a complex (B, B) pair, i.e. B·(1+i), which rotates
    every frequency by 45° and makes DFT_REAL_OUTPUT garble ordinary photos.
    The intended real filter (B, 0) is used here, so photos survive while
    periodic moiré peaks are still notched out.
    """
    rows, cols = rgb.shape[:2]
    i2 = ((np.arange(rows, dtype=np.float32) - np.float32(rows / 2.0)) / np.float32(80.0)) ** 2
    j2 = ((np.arange(cols, dtype=np.float32) - np.float32(cols / 2.0)) / np.float32(80.0)) ** 2
    with np.errstate(over="ignore"):
        bw = 1.0 / (1.0 + np.power(i2[:, None] + j2[None, :], np.float32(10.0)))  # 1 / (1 + (d/d0)^2n)
    bw = bw.astype(F32)
    filt = cv2.merge([bw, np.zeros_like(bw)])
    del bw
    out = np.empty((rows, cols, 3), F32)

    def channel(ch: int) -> None:
        if check is not None:
            check()
        plane = np.ascontiguousarray(rgb[..., ch], dtype=F32) * np.float32(255.0)
        # == dft of the (plane, 0) complex matrix upstream builds, at half the cost
        cplx = cv2.dft(plane, flags=cv2.DFT_COMPLEX_OUTPUT)
        del plane
        _shift_dft(cplx)
        mag8 = _log_mag_u8(cplx)
        applied = kind if kind != "auto" else (_moire_detect(mag8) or "vertical")
        _moire_mask(cplx, mag8, applied)
        del mag8
        cplx = cv2.mulSpectrums(cplx, filt, 0)
        _shift_dft(cplx)
        inv = cv2.idft(cplx, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        del cplx
        out[..., ch] = inv * np.float32(1.0 / 255.0)

    # the three channels are independent; OpenCV/numpy release the GIL
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in [pool.submit(channel, ch) for ch in range(3)]:
            f.result()
    return out


# -- Auto perspective (opencv-tools auto_straight + document_detector) -------

def _angle_cos(p1, p2, o) -> float:
    dx1, dy1 = p1[0] - o[0], p1[1] - o[1]
    dx2, dy2 = p2[0] - o[0], p2[1] - o[1]
    return (dx1 * dx2 + dy1 * dy2) / math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10)


def _doc_max_cos(pts) -> float:
    m = 0.0
    for i in range(2, 6):
        m = max(m, abs(_angle_cos(pts[i % 4], pts[i - 2], pts[(i - 1) % 4])))
    return m


def _doc_collect(mask, iw, ih, cands, weight) -> None:
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lo, hi = iw * ih * 0.04, iw * ih * 0.92
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (lo <= area < hi):
            continue
        c2f = cnt.astype(np.float32)
        arc = cv2.arcLength(c2f, True)
        if arc < 100:
            continue
        approx = cv2.approxPolyDP(c2f, 0.02 * arc, True)
        if len(approx) != 4:
            continue
        quad = approx.reshape(-1, 2).astype(np.int32)
        if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
            continue
        pts = [(float(x), float(y)) for x, y in quad]
        mc = _doc_max_cos(pts)
        if mc >= 0.4:
            continue
        cands.append((area + weight * (1 - mc), pts, area, mc))


def _doc_find_candidates(ch, iw, ih, cands, weight) -> None:
    morph = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    _, t = cv2.threshold(ch, 160.0, 255.0, cv2.THRESH_BINARY)
    t = cv2.dilate(cv2.morphologyEx(t, cv2.MORPH_CLOSE, morph), dil)
    _doc_collect(t, iw, ih, cands, weight)
    _, o = cv2.threshold(ch, 0.0, 255.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    o = cv2.dilate(cv2.morphologyEx(o, cv2.MORPH_CLOSE, morph), dil)
    _doc_collect(o, iw, ih, cands, weight - 0.25)
    thr = 60
    while thr >= 10:
        e = cv2.dilate(cv2.Canny(ch, thr * 2.0, thr * 4.0), dil)
        _doc_collect(e, iw, ih, cands, weight - (60 - thr + 1))
        if cands:
            best = max(cands, key=lambda c: c[0])
            if best[3] < 0.3 and best[2] > iw * ih * 0.2:
                break
        thr -= 10


def _document_corners(rgb8: np.ndarray):
    """DocumentDetector.findDocumentCorners (TL, TR, BL, BR) or None."""
    h, w = rgb8.shape[:2]
    max_side = float(max(w, h))
    s = max_side / 500.0 if max_side > 500.0 else 1.0
    sw, sh = w / s, h / s
    small = cv2.resize(rgb8, (max(1, int(sw)), max(1, int(sh))))
    blurred = cv2.medianBlur(small, 9)
    luv = cv2.cvtColor(blurred, cv2.COLOR_RGB2Luv)
    cands: list = []
    weight = 3_000_000.0
    for ch in [blurred[..., i] for i in range(3)] + [luv[..., i] for i in range(3)]:
        _doc_find_candidates(np.ascontiguousarray(ch), sw, sh, cands, weight)
        weight -= 1.0
    if not cands:
        return None
    best = max(cands, key=lambda c: c[0])
    pts = [(x * s, y * s) for x, y in best[1]]
    by_y = sorted(pts, key=lambda p: p[1])
    return sorted(by_y[:2], key=lambda p: p[0]) + sorted(by_y[-2:], key=lambda p: p[0])


def _sort_corners(pts):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    ordered = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    tl = min(range(len(ordered)), key=lambda i: ordered[i][0] + ordered[i][1])
    return [ordered[(tl + k) % len(ordered)] for k in range(len(ordered))]


def _poly_area(pts) -> float:
    n = len(pts)
    return abs(sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n)) / 2.0)


def _max_corner_cos(pts) -> float:
    n = len(pts)
    best = -1.0
    for i in range(n):
        p, c, q = pts[(i + n - 1) % n], pts[i], pts[(i + 1) % n]
        fx, fy, sx, sy = p[0] - c[0], p[1] - c[1], q[0] - c[0], q[1] - c[1]
        den = math.sqrt((fx * fx + fy * fy) * (sx * sx + sy * sy))
        best = max(best, 1.0 if den == 0.0 else abs((fx * sx + fy * sy) / den))
    return best


def _add_candidate(pts, contour_area, iw, ih, scale, bonus, cands) -> None:
    srt = _sort_corners(pts)
    quad_area = _poly_area(srt)
    area_factor = quad_area / (iw * ih)
    if not (0.02 <= area_factor <= 1.02):
        return
    mc = _max_corner_cos(srt)
    if mc > 0.82:
        return
    sides = [math.dist(srt[i], srt[(i + 1) % len(srt)]) for i in range(len(srt))]
    if min(sides) < min(iw, ih) * 0.04:
        return
    fill = min(1.0, max(0.0, contour_area / quad_area))
    if fill < 0.45:
        return
    margin = min(iw, ih) * 0.012
    border = sum(1 for x, y in srt if x <= margin or y <= margin or x >= iw - margin or y >= ih - margin)
    if border == 4:
        return
    penalty = 0.7 if border == 3 else 0.0
    score = area_factor * 5.0 + (1.0 - mc) * 2.0 + fill + bonus - penalty
    cands.append((score, [(x * scale, y * scale) for x, y in srt]))


def _collect_quads(mask, iw, ih, scale, bonus, cands) -> None:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = iw * ih
    for cnt in contours:
        area = abs(cv2.contourArea(cnt))
        if area < img_area * 0.02:
            continue
        c2f = cnt.astype(np.float32)
        per = cv2.arcLength(c2f, True)
        if per <= 0.0:
            continue
        for f in (0.012, 0.02, 0.035, 0.05):
            approx = cv2.approxPolyDP(c2f, per * f, True)
            if len(approx) != 4:
                continue
            quad = approx.reshape(-1, 2)
            if not cv2.isContourConvex(quad.astype(np.int32).reshape(-1, 1, 2)):
                continue
            _add_candidate([(float(x), float(y)) for x, y in quad], area, iw, ih, scale, bonus, cands)


def find_perspective_corners(rgb8: np.ndarray, check=None):
    """PerspectiveDetector.findCorners -> [TL, TR, BR, BL] in full-resolution pixels, or None."""
    h, w = rgb8.shape[:2]
    cands: list = []
    doc = _document_corners(rgb8)
    if doc is not None:
        srt = _sort_corners(doc)
        _add_candidate(srt, _poly_area(srt), float(w), float(h), 1.0, 0.3, cands)
    if check is not None:
        check()
    scale = max(1.0, max(w, h) / 1200.0)
    iw, ih = w / scale, h / scale
    small = cv2.resize(rgb8, (max(1, int(iw)), max(1, int(ih))))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    masks = []
    otsu, m = cv2.threshold(blurred, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    masks.append((cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel), 0.1))
    _, m = cv2.threshold(blurred, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    masks.append((cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel), 0.1))
    lower = min(110.0, max(20.0, otsu * 0.45))
    upper = min(255.0, max(lower + 30.0, otsu * 1.35))
    for lo, hi, bonus in ((lower, upper, 0.25), (30.0, 100.0, 0.2)):
        e = cv2.Canny(blurred, lo, hi)
        e = cv2.dilate(cv2.morphologyEx(e, cv2.MORPH_CLOSE, kernel), kernel)
        masks.append((e, bonus))
    block = min(101, max(15, int(min(iw, ih) / 12)))
    if block % 2 == 0:
        block += 1
    for ttype in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        m = cv2.adaptiveThreshold(blurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, ttype, block, 7.0)
        masks.append((cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel), 0.15))
    for mask, bonus in masks:
        if check is not None:
            check()
        _collect_quads(mask, iw, ih, scale, bonus, cands)
    if not cands:
        return None
    return max(cands, key=lambda c: c[0])[1]


def auto_perspective(img: np.ndarray, check=None) -> np.ndarray:
    """AutoStraighten.process(mode = Perspective): detect a quad and rectify it."""
    prem = img.copy() if img.shape[2] == 3 else np.dstack([bitmap_rgb(img), img[..., 3]]).astype(F32)
    corners = find_perspective_corners(to_u8(prem[..., :3]), check)
    if corners is None:
        return img
    tl, tr, br, bl = corners
    max_w = int(max(math.dist(tl, tr), math.dist(br, bl)))
    max_h = int(max(math.dist(tl, bl), math.dist(tr, br)))
    if max_w < 2 or max_h < 2:
        return img
    src = np.array([tl, tr, br, bl], np.float32)
    dst = np.array([[0, 0], [max_w, 0], [max_w, max_h], [0, max_h]], np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(prem, m, (max_w, max_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if out.ndim == 2:  # pragma: no cover
        out = out[..., None]
    if img.shape[2] == 4:
        a = out[..., 3:4]
        rgb = np.where(a > 1e-6, out[..., :3] / np.maximum(a, 1e-6), 0.0)
        out = np.concatenate([rgb, a], axis=-1)
    return out.astype(F32)
