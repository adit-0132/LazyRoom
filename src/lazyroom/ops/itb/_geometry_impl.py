"""Private helpers for the ImageToolbox "Geometry" (and "Frames") group ports.

Ported from ImageToolbox / ImageToolboxLibs (Apache-2.0, T8RIN):
* jhlabs transform filters (ImageToolboxLibs ``libs/jhlabs/src/main/rust/jhlabs/src/lib.rs``),
* procedural GPU shaders (``feature/filters/.../utils/gpu/ProceduralShaders.kt``),
* opencv-tools (``lib/opencv-tools``: AutoCropper, AutoStraighten, FreeCrop,
  ImageExpander, LensCorrection, SeamCarver),
* Trickle ``cropToContent`` (``trickle/src/main/cpp/ProcessingFilters.cpp``),
* the G'MIC JNI bridge alpha handling (``libs/gmic/src/main/cpp/gmic_jni.cpp``).
"""
from __future__ import annotations

import json
import math
from typing import Sequence

import cv2
import numpy as np
from numba import njit, prange

from ...core.errors import ErrorCode, LazyRoomError
from .._util import F32
from ._common import run_gmic

TAU = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Generic buffer helpers
# ---------------------------------------------------------------------------

def has_alpha(img: np.ndarray) -> bool:
    return img.ndim == 3 and img.shape[2] == 4


def ensure_rgba(img: np.ndarray) -> np.ndarray:
    if img.shape[2] == 4:
        return img.astype(F32, copy=False)
    return np.dstack([img[..., :3], np.ones(img.shape[:2], F32)]).astype(F32)


def finish(out: np.ndarray, had_alpha: bool) -> np.ndarray:
    """RGBA output keeps its alpha when the input had one or when transparency was created.

    Alpha within 2/255 of opaque (8-bit rounding of upstream blends) is not treated as transparency.
    """
    out = out.astype(F32, copy=False)
    if out.shape[2] == 4 and not had_alpha and float(out[..., 3].min()) >= 1.0 - 2.5 / 255.0:
        return np.ascontiguousarray(out[..., :3])
    return out


def to_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] <= 4:
        return cv2.convertScaleAbs(np.maximum(img, 0, dtype=F32), alpha=255.0).reshape(img.shape)
    return np.clip(np.floor(img * 255.0 + 0.5), 0, 255).astype(np.uint8)


def from_u8(img: np.ndarray) -> np.ndarray:
    return np.multiply(img, F32(1.0 / 255.0), dtype=F32)


def kround(x: float) -> int:
    """Kotlin ``roundToInt`` (round half up)."""
    return int(math.floor(x + 0.5))


def over(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Straight-alpha source-over of two RGBA buffers of the same size."""
    sa = src[..., 3:4]
    if float(sa.min()) >= 1.0:
        return src.astype(F32, copy=True)
    da = dst[..., 3:4]
    oa = sa + da * (1.0 - sa)
    rgb = (src[..., :3] * sa + dst[..., :3] * da * (1.0 - sa)) / np.maximum(oa, 1e-8)
    rgb = np.where(oa > 1e-8, rgb, 0.0)
    return np.dstack([rgb, oa]).astype(F32)


def int32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


# ---------------------------------------------------------------------------
# G'MIC bridge with ImageToolbox's alpha modes (GmicAlphaMode.Preserve / Process)
# ---------------------------------------------------------------------------

def gmic_apply(img: np.ndarray, cmd: str, ctx, process_alpha: bool) -> np.ndarray:
    """Run ``cmd`` like ImageToolboxLibs' native bridge.

    * Process mode (``gmicProcessAlpha``): RGBA is always passed (alpha 255
      when opaque) and the output alpha (4th or 2nd channel) is used.
    * Preserve mode: only RGB is passed; the source alpha is re-attached
      (bilinearly resampled when the size changed) and any alpha produced by
      G'MIC is discarded.

    The pipeline is prefixed with ``srand`` (seeded from ``ctx.rng``) because
    G'MIC's random generator is clock-seeded otherwise.
    """
    had = has_alpha(img)
    cmd = gmic_seeded(ctx, cmd)
    if process_alpha:
        res = run_gmic(ensure_rgba(img), cmd, ctx=ctx, preserve_alpha=False)
        return finish(ensure_rgba(res), had)
    res = run_gmic(np.ascontiguousarray(img[..., :3]), cmd, ctx=ctx, preserve_alpha=False)
    rgb = np.ascontiguousarray(res[..., :3])
    if not had:
        return rgb
    a = img[..., 3]
    if a.shape != rgb.shape[:2]:
        a = cv2.resize(a, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.dstack([rgb, a]).astype(F32)


def gmic_seeded(ctx, cmd: str) -> str:
    """Prefix ``srand`` so G'MIC commands using randomness (clock-seeded by default) are deterministic."""
    return f"srand {int(ctx.rng('gmic').integers(0, 2 ** 31 - 1))} {cmd}"


def gmic_color(c: Sequence[float], alpha: bool = False) -> list[int]:
    vals = [int(round(float(x) * 255.0)) for x in c[:4]]
    return vals[:4] if alpha else vals[:3]


# ---------------------------------------------------------------------------
# jhlabs TransformFilter sampling (Rust port in ImageToolboxLibs)
# ---------------------------------------------------------------------------

EDGE_ZERO, EDGE_CLAMP, EDGE_WRAP, EDGE_RGB_CLAMP = 0, 1, 2, 3


BAND_PIXELS = 1 << 20


def bands(h: int, w: int):
    """Row bands of about one megapixel (bounded memory for full-frame coordinate maps)."""
    rows = max(1, BAND_PIXELS // max(w, 1))
    for y0 in range(0, h, rows):
        yield y0, min(h, y0 + rows)


def _remap(src: np.ndarray, mx: np.ndarray, my: np.ndarray, border: int, value=0.0) -> np.ndarray:
    out = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=border, borderValue=value)
    return out.reshape(mx.shape + src.shape[2:])


def _coords(v, lo: float, hi: float, shape) -> np.ndarray:
    """Source coordinate map as float32, clamped to [lo, hi]; NaN -> lo."""
    out = np.empty(shape, F32)
    np.clip(v, lo, hi, out=out)
    nan = np.isnan(out)
    if nan.any():
        out[nan] = lo
    return out


def jh_transform(img: np.ndarray, mapfn, edge: int) -> np.ndarray:
    """Bilinear inverse mapping with jhlabs edge actions.

    ``mapfn(X, Y)`` gets pixel coordinates as broadcastable float64 arrays
    (``X`` shape (1, w), ``Y`` shape (rows, 1)) and returns the source
    coordinates.  ZERO: outside = transparent black; CLAMP: nearest edge
    pixel; WRAP: periodic; RGB_CLAMP: edge colour with alpha 0.  Pixel ``x``
    samples the source at integer coordinate ``x`` (no half-pixel offset),
    like the Rust port.
    """
    h, w = img.shape[:2]
    had = has_alpha(img)
    src = np.ascontiguousarray(img, dtype=F32)
    if edge in (EDGE_ZERO, EDGE_RGB_CLAMP):
        src_rgb = np.ascontiguousarray(src[..., :3]) if had else src
        src_a = np.ascontiguousarray(src[..., 3]) if had else np.ones((h, w), F32)
        out = np.empty((h, w, 4), F32)
    else:
        out = np.empty(src.shape, F32)
    X = np.arange(w, dtype=np.float64)[None, :]
    for y0, y1 in bands(h, w):
        Y = np.arange(y0, y1, dtype=np.float64)[:, None]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sx, sy = mapfn(X, Y)
            shape = (y1 - y0, w)
            if edge == EDGE_WRAP:
                mx = _coords(gmod(np.asarray(sx, np.float64), w), 0.0, w, shape)
                my = _coords(gmod(np.asarray(sy, np.float64), h), 0.0, h, shape)
                out[y0:y1] = _remap(src, mx, my, cv2.BORDER_WRAP)
                continue
            mx = _coords(sx, -2.0, w + 1.0, shape)
            my = _coords(sy, -2.0, h + 1.0, shape)
        if edge == EDGE_CLAMP:
            out[y0:y1] = _remap(src, mx, my, cv2.BORDER_REPLICATE)
        else:
            rgb_border = cv2.BORDER_CONSTANT if edge == EDGE_ZERO else cv2.BORDER_REPLICATE
            out[y0:y1, :, :3] = _remap(src_rgb, mx, my, rgb_border, (0.0, 0.0, 0.0, 0.0))
            out[y0:y1, :, 3] = _remap(src_a, mx, my, cv2.BORDER_CONSTANT, 0.0)
    if edge in (EDGE_CLAMP, EDGE_WRAP):
        return out
    return finish(out, had)


def jh_triangle(v: np.ndarray) -> np.ndarray:
    r = v - np.floor(v)
    return 2.0 * np.where(r < 0.5, r, 1.0 - r)


def perspective_coeffs(x0, y0, x1, y1, x2, y2, x3, y3) -> tuple[float, ...]:
    """jhlabs ``PerspectiveFilter.unitSquareToQuad`` (float32 arithmetic)."""
    f = np.float32
    x0, y0, x1, y1, x2, y2, x3, y3 = (f(v) for v in (x0, y0, x1, y1, x2, y2, x3, y3))
    dx1, dy1 = x1 - x2, y1 - y2
    dx2, dy2 = x3 - x2, y3 - y2
    dx3, dy3 = x0 - x1 + x2 - x3, y0 - y1 + y2 - y3
    if dx3 == 0 and dy3 == 0:
        a11, a21, a31 = x1 - x0, x2 - x1, x0
        a12, a22, a32 = y1 - y0, y2 - y1, y0
        a13 = a23 = f(0)
    else:
        den = dx1 * dy2 - dy1 * dx2
        with np.errstate(divide="ignore", invalid="ignore"):
            a13 = (dx3 * dy2 - dx2 * dy3) / den
            a23 = (dx1 * dy3 - dy1 * dx3) / den
        a11 = x1 - x0 + a13 * x1
        a21 = x3 - x0 + a23 * x3
        a31 = x0
        a12 = y1 - y0 + a13 * y1
        a22 = y3 - y0 + a23 * y3
        a32 = y0
    return tuple(float(v) for v in (a11, a12, a13, a21, a22, a23, a31, a32, f(1)))


# ---------------------------------------------------------------------------
# Procedural GPU shaders (ProceduralShaders.kt) — numba kernels emitting remap coordinates
# ---------------------------------------------------------------------------
#
# GLSL conventions: textureCoordinate = ((x + 0.5) / w, (y + 0.5) / h);
# toField(uv) = rotate((uv - 0.5 - offset) * imageAspect, -rotation) / scale;
# sampleField(p): uv = rotate(p * scale, rotation) / imageAspect + 0.5 + offset,
# mirrored-repeat (1 - |mod(uv, 2) - 1|), clamped to half a texel, GL_LINEAR.

def gmod(x, y):
    """GLSL ``mod``: x - y * floor(x / y) (also jhlabs ``rem_euclid`` for y > 0)."""
    return x - y * np.floor(x / y)


def proc_transform(img: np.ndarray, offset_x: float, offset_y: float, scale: float, rotation: float) -> tuple:
    h, w = img.shape[:2]
    short = float(max(min(w, h), 1))
    rot = math.radians(rotation)
    return (w / short, h / short, float(offset_x), float(offset_y), float(scale), math.cos(-rot), math.sin(-rot),
            math.cos(rot), math.sin(rot), float(w), float(h))


@njit(cache=True)
def _to_field(x, y, T):
    ax, ay, ox, oy, scale, cn, sn, cp, sp, w, h = T
    fx = ((x + 0.5) / w - 0.5 - ox) * ax
    fy = ((y + 0.5) / h - 0.5 - oy) * ay
    return (cn * fx - sn * fy) / scale, (sn * fx + cn * fy) / scale


@njit(cache=True)
def _mirror(u):
    r = u - 2.0 * np.floor(u / 2.0)
    return 1.0 - abs(r - 1.0)


@njit(cache=True)
def _uv_px(u, v, w, h):
    u = min(max(u, 0.5 / w), 1.0 - 0.5 / w)
    v = min(max(v, 0.5 / h), 1.0 - 0.5 / h)
    return u * w - 0.5, v * h - 0.5


@njit(cache=True)
def _field_px(qx, qy, T):
    ax, ay, ox, oy, scale, cn, sn, cp, sp, w, h = T
    x = qx * scale
    y = qy * scale
    u = (cp * x - sp * y) / ax + 0.5 + ox
    v = (sp * x + cp * y) / ay + 0.5 + oy
    return _uv_px(_mirror(u), _mirror(v), w, h)


@njit(cache=True)
def _angle_of(x, y):
    if x * x + y * y < 1e-8:
        return 0.0
    return math.atan2(y, x)


@njit(cache=True)
def _gm(x, y):
    return x - y * np.floor(x / y)


@njit(cache=True)
def _fr32(x):
    return np.float32(x - np.floor(x))


@njit(cache=True)
def _hash(px, py):
    """Shader ``hash(vec2)`` evaluated in float32 like the GPU."""
    f = np.float32
    x = f(px)
    y = f(py)
    hx = _fr32(f(x * f(0.1031)))
    hy = _fr32(f(y * f(0.1030)))
    hz = _fr32(f(x * f(0.0973)))
    d = f(f(hx * f(hy + f(33.33))) + f(hy * f(hz + f(33.33)))) + f(hz * f(hx + f(33.33)))
    hx = f(hx + d)
    hy = f(hy + d)
    hz = f(hz + d)
    return float(_fr32(f(f(hx + hy) * hz)))


@njit(parallel=True, cache=True)
def k_circular_mirror(y0, mx, my, T, intensity, radius, mode):
    rows, w = mx.shape
    for j in prange(rows):
        for i in range(w):
            px, py = _to_field(i, y0 + j, T)
            r = math.sqrt(px * px + py * py)
            if mode < 0.5:
                target = radius + abs(r - radius)
            else:
                target = max(0.0, radius - abs(r - radius))
            k = 1.0 + (target / max(r, 0.00001) - 1.0) * intensity
            mx[j, i], my[j, i] = _field_px(px * k, py * k, T)


@njit(parallel=True, cache=True)
def k_progressive_scaling(y0, mx, my, T, intensity, ratio, radius):
    rows, w = mx.shape
    lr = math.log(ratio)
    for j in prange(rows):
        for i in range(w):
            px, py = _to_field(i, y0 + j, T)
            r = max(math.sqrt(px * px + py * py), 0.00001)
            level = np.floor(math.log(r / radius) / lr)
            k = math.pow(ratio, -level)
            mx[j, i], my[j, i] = _field_px(px + (px * k - px) * intensity, py + (py * k - py) * intensity, T)


@njit(parallel=True, cache=True)
def k_hex_kaleidoscope(y0, mx, my, T, count, frequency, offset, mode):
    rows, w = mx.shape
    sx, sy = 1.7320508, 3.0
    sector = 6.28318530718 / count
    for j in prange(rows):
        for i in range(w):
            px, py = _to_field(i, y0 + j, T)
            gx, gy = px * frequency, py * frequency
            ax_, ay_ = _gm(gx, sx) - sx * 0.5, _gm(gy, sy) - sy * 0.5
            bx_, by_ = _gm(gx - sx * 0.5, sx) - sx * 0.5, _gm(gy - sy * 0.5, sy) - sy * 0.5
            if ax_ * ax_ + ay_ * ay_ < bx_ * bx_ + by_ * by_:
                lx, ly = ax_, ay_
            else:
                lx, ly = bx_, by_
            ang = _gm(_angle_of(lx, ly) + sector * 0.5, sector)
            if mode < 0.5:
                ang = abs(ang - sector * 0.5)
            elif mode < 1.5:
                ang -= sector * 0.5
            else:
                ang = sector * 0.5 - abs(ang - sector * 0.5)
            ln = math.sqrt(lx * lx + ly * ly) * 0.5
            mx[j, i], my[j, i] = _field_px(math.cos(ang) * ln + offset, math.sin(ang) * ln, T)


@njit(parallel=True, cache=True)
def k_smooth_kaleidoscope(y0, mxs, mys, wts, T, blend, offset, frequency):
    _, rows, w = mxs.shape
    ww, hh = T[9], T[10]
    expo = 16.0 + (1.0 - 16.0) * blend
    for j in prange(rows):
        for i in range(w):
            px, py = _to_field(i, y0 + j, T)
            gx, gy = px * frequency, py * frequency
            cx, cy = np.floor(gx), np.floor(gy)
            fx, fy = gx - cx, gy - cy
            n = 0
            for yy in range(2):
                for xx in range(2):
                    lx, ly = fx - xx, fy - yy
                    ang = _gm(cx + xx + cy + yy, 4.0) * 6.28318530718 * 0.25
                    c, s = math.cos(ang), math.sin(ang)
                    u = (c * lx - s * ly) * 0.5 + 0.5 + offset
                    v = (s * lx + c * ly) * 0.5 + 0.5 + offset
                    mxs[n, j, i], mys[n, j, i] = _uv_px(_mirror(u), _mirror(v), ww, hh)
                    wx, wy = max(0.0, 1.0 - abs(lx)), max(0.0, 1.0 - abs(ly))
                    p = wx * wy
                    wts[n, j, i] = math.pow(p, expo) if p > 0.0 else 0.0
                    n += 1


@njit(parallel=True, cache=True)
def k_random_kaleidoscope_grid(y0, mx1, my1, mx2, my2, wgt, T, count, frequency, seed, variability, blend):
    rows, w = mx1.shape
    e0 = 1.0 - max(blend, 0.001)
    for j in prange(rows):
        for i in range(w):
            px, py = _to_field(i, y0 + j, T)
            gx, gy = px * frequency, py * frequency
            cx, cy = np.floor(gx), np.floor(gy)
            lx, ly = gx - cx - 0.5, gy - cy - 0.5
            r1 = _hash(cx + seed, cy + seed)
            r2 = _hash(cx + seed + 17.17, cy + seed + 47.23)
            spikes = max(1.0, np.floor(count + (r1 - 0.5) * variability * count))
            sector = 6.28318530718 / spikes
            ang = abs(_gm(_angle_of(lx, ly) + r2 * 6.28318530718, sector) - sector * 0.5)
            ln = math.sqrt(lx * lx + ly * ly)
            mx1[j, i], my1[j, i] = _field_px(math.cos(ang) * ln, math.sin(ang) * ln, T)
            mx2[j, i], my2[j, i] = _field_px(lx, ly, T)
            edge = max(abs(lx), abs(ly)) * 2.0
            t = min(max((edge - e0) / (1.0 - e0), 0.0), 1.0)
            wgt[j, i] = t * t * (3.0 - 2.0 * t)


@njit(parallel=True, cache=True)
def k_inversion_fractal(y0, mx, my, T, mode, iterations, intensity):
    rows, w = mx.shape
    for j in prange(rows):
        for i in range(w):
            qx, qy = _to_field(i, y0 + j, T)
            for _ in range(iterations):
                if mode < 0.5:
                    qx, qy = abs(qx) - 0.4, abs(qy) - 0.4
                elif mode < 1.5:
                    qx, qy = abs(qy) - 0.35, abs(qx) - 0.5
                else:
                    qx, qy = _gm(qx + 0.5, 1.0) - 0.5, _gm(qy + 0.5, 1.0) - 0.5
                k = intensity / max(qx * qx + qy * qy, 0.01)
                qx = min(max(qx * k, -100.0), 100.0)
                qy = min(max(qy * k, -100.0), 100.0)
            mx[j, i], my[j, i] = _field_px(qx, qy, T)


def proc_single(img: np.ndarray, T: tuple, kernel, *args) -> np.ndarray:
    """Run a one-sample shader kernel band by band and sample with GL_LINEAR/clamp."""
    img = img.astype(F32, copy=False)
    h, w = img.shape[:2]
    out = np.empty(img.shape, F32)
    for y0, y1 in bands(h, w):
        mx = np.empty((y1 - y0, w), F32)
        my = np.empty((y1 - y0, w), F32)
        kernel(y0, mx, my, T, *args)
        out[y0:y1] = _remap(img, mx, my, cv2.BORDER_REPLICATE)
    return out


def proc_smooth_kaleidoscope(img: np.ndarray, T: tuple, blend: float, offset: float, frequency: float) -> np.ndarray:
    img = img.astype(F32, copy=False)
    h, w = img.shape[:2]
    out = np.empty(img.shape, F32)
    for y0, y1 in bands(h, w // 2 + 1):
        rows = y1 - y0
        mxs = np.empty((4, rows, w), F32)
        mys = np.empty((4, rows, w), F32)
        wts = np.empty((4, rows, w), F32)
        k_smooth_kaleidoscope(y0, mxs, mys, wts, T, float(blend), float(offset), float(frequency))
        acc = np.zeros((rows, w, img.shape[2]), F32)
        for n in range(4):
            acc += _remap(img, mxs[n], mys[n], cv2.BORDER_REPLICATE) * wts[n][..., None]
        out[y0:y1] = acc / np.maximum(wts.sum(axis=0), F32(1e-20))[..., None]
    return out


def proc_random_kaleidoscope_grid(img: np.ndarray, T: tuple, *args) -> np.ndarray:
    img = img.astype(F32, copy=False)
    h, w = img.shape[:2]
    out = np.empty(img.shape, F32)
    for y0, y1 in bands(h, w):
        rows = y1 - y0
        m = [np.empty((rows, w), F32) for _ in range(5)]
        k_random_kaleidoscope_grid(y0, *m, T, *args)
        kal = _remap(img, m[0], m[1], cv2.BORDER_REPLICATE)
        orig = _remap(img, m[2], m[3], cv2.BORDER_REPLICATE)
        out[y0:y1] = kal + (orig - kal) * m[4][..., None]
    return out


# ---------------------------------------------------------------------------
# AutoCropper (opencv-tools)
# ---------------------------------------------------------------------------

def auto_crop(img: np.ndarray, sensitivity: int) -> np.ndarray:
    rgba = to_u8(ensure_rgba(img))
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    gray = cv2.medianBlur(gray, 3)
    canny = cv2.Canny(gray, 100.0, 200.0)
    threshold = int((10 - sensitivity) * 20.5) + 50
    rows = np.nonzero(canny.mean(axis=1) > threshold)[0].tolist()
    if not rows:
        return img
    cands = [0] + rows + [canny.shape[0]]
    sobel = cv2.Sobel(rgba, cv2.CV_16U, 2, 2, ksize=5).astype(np.float64)
    row_means = sobel.mean(axis=1)  # (H, 4)
    csum = np.concatenate([np.zeros((1, 4)), np.cumsum(row_means, axis=0)])
    best, best_edges = 0.0, None
    total = sobel.shape[0]
    for top, bottom in zip(cands[:-1], cands[1:]):
        n = bottom - top
        mean = float(((csum[bottom] - csum[top]) / n).mean()) if n > 0 else 0.0
        score = mean * (n / total)
        if score > best:
            best, best_edges = score, (top, bottom)
    if best_edges is None:
        return img
    top, bottom = best_edges
    if bottom - top - 1 <= 0:
        return img
    return np.ascontiguousarray(img[top + 1:bottom])


# ---------------------------------------------------------------------------
# AutoStraighten (opencv-tools): Deskew and Manual (FreeCrop) modes
# ---------------------------------------------------------------------------

def deskew(img: np.ndarray, max_skew: int, allow_crop: bool) -> np.ndarray:
    had = has_alpha(img)
    rgba = ensure_rgba(img)
    w0 = rgba.shape[1]
    gray = cv2.cvtColor(to_u8(rgba), cv2.COLOR_BGR2GRAY)  # upstream converts an RGBA mat with BGR2GRAY
    gray = cv2.fastNlMeansDenoising(gray, None, 3.0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    lines = cv2.HoughLinesP(binary, 1.0, math.pi / 180, 200, minLineLength=w0 / 12.0, maxLineGap=w0 / 150.0)
    if lines is None or len(lines) == 0:
        return img
    l = lines.reshape(-1, 4).astype(np.float64)
    angles = np.arctan2(l[:, 3] - l[:, 1], l[:, 2] - l[:, 0])
    landscape = int(np.count_nonzero(np.abs(angles) > math.pi / 4)) > len(angles) // 2
    deg = np.abs(np.degrees(angles))
    if landscape:
        filtered = angles[(deg > 90 - max_skew) & (deg < 90 + max_skew)]
    else:
        filtered = angles[deg < max_skew]
    if len(filtered) < 5:
        return img
    angle_deg = math.degrees(float(np.median(filtered)))
    if landscape:
        if angle_deg < 0:
            rotated = cv2.rotate(rgba, cv2.ROTATE_90_CLOCKWISE)
            angle_deg += 90
        else:
            rotated = cv2.rotate(rgba, cv2.ROTATE_90_COUNTERCLOCKWISE)
            angle_deg -= 90
    else:
        rotated = rgba
    rh, rw = rotated.shape[:2]
    center = (rw / 2.0, rh / 2.0)
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rad = math.radians(angle_deg)
    s, c = abs(math.sin(rad)), abs(math.cos(rad))
    nw = int(rh * s + rw * c)
    nh = int(rh * c + rw * s)
    m[0, 2] += nw / 2.0 - center[0]
    m[1, 2] += nh / 2.0 - center[1]
    full = cv2.warpAffine(rotated, m, (max(nw, 1), max(nh, 1)), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    if allow_crop:
        bw = max(rw * c - rh * s, 0.0)
        bh = max(rh * c - rw * s, 0.0)
        x = int((rh * s + rw * c - bw) / 2.0)
        y = int((rh * c + rw * s - bh) / 2.0)
        bw, bh = int(bw), int(bh)
        if bw > 0 and bh > 0 and x >= 0 and y >= 0 and x + bw <= full.shape[1] and y + bh <= full.shape[0]:
            full = full[y:y + bh, x:x + bw]
    return finish(np.ascontiguousarray(full), had)


def free_crop(img: np.ndarray, pts: Sequence[tuple[float, float]]) -> np.ndarray:
    """FreeCrop.crop: warp the quad (TL, TR, BR, BL) to a min-edge sized rectangle."""
    had = has_alpha(img)
    tl, tr, br, bl = (np.asarray(p, np.float64) for p in pts)
    width = min(np.hypot(*(tl - tr)), np.hypot(*(bl - br)))
    height = min(np.hypot(*(tl - bl)), np.hypot(*(tr - br)))
    ow, oh = max(int(width), 1), max(int(height), 1)
    src = np.array([tl, tr, br, bl], np.float32)
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    try:
        m = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return img
    if not np.isfinite(m).all():
        return img
    out = cv2.warpPerspective(ensure_rgba(img), m, (ow, oh), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return finish(out.reshape(oh, ow, 4), had)


# ---------------------------------------------------------------------------
# Trickle cropToContent
# ---------------------------------------------------------------------------

def crop_to_content(img: np.ndarray, color: Sequence[float], tolerance: float) -> np.ndarray:
    had = has_alpha(img)
    # native code reads premultiplied RGBA_8888 memory as a little-endian uint32 and compares
    # its bytes 2,1,0 (B,G,R) with the ARGB colour's R,G,B (upstream channel order kept).
    rgb = img[..., :3] * img[..., 3:4] if had else img[..., :3]
    u8 = to_u8(rgb)
    c = [int(round(float(v) * 255)) for v in color[:3]]
    d2 = np.zeros(img.shape[:2], np.int32)
    for ch, cv in ((2, c[0]), (1, c[1]), (0, c[2])):
        dd = u8[..., ch].astype(np.int32) - cv
        d2 += dd * dd
    keep = d2 > (tolerance * 255.0) ** 2
    rows = np.flatnonzero(keep.any(axis=1))
    if len(rows) == 0:
        return img
    cols = np.flatnonzero(keep.any(axis=0))
    top, bottom, left, right = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
    out = np.zeros((bottom - top + 1, right - left + 1, 4), F32)
    # upstream memcpy copies (right - left) pixels per row: the last column stays transparent
    out[:, :right - left, :3] = img[top:bottom + 1, left:right, :3]
    out[:, :right - left, 3] = img[top:bottom + 1, left:right, 3] if had else 1.0
    return out


# ---------------------------------------------------------------------------
# LensCorrection (opencv-tools)
# ---------------------------------------------------------------------------

def parse_lens_profile(text: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    def bad(msg: str) -> LazyRoomError:
        return LazyRoomError(ErrorCode.INVALID_PARAMS, f"Invalid lens profile: {msg}")

    try:
        data = json.loads(text)
    except ValueError as exc:
        raise bad(f"not JSON ({exc})") from exc
    fp = data.get("fisheye_params") if isinstance(data, dict) else None
    if not isinstance(fp, dict):
        raise bad("No fisheye_params in JSON")
    cm = fp.get("camera_matrix")
    if not (isinstance(cm, list) and len(cm) == 3 and all(isinstance(r, list) and len(r) == 3 for r in cm)):
        raise bad("Incorrect camera_matrix size (pass 3x3)")
    dc = fp.get("distortion_coeffs")
    if not (isinstance(dc, list) and len(dc) == 4):
        raise bad("Bad distortion coefficients (pass only 4)")
    dim = None
    for key in ("calib_dimension", "orig_dimension", "output_dimension"):
        if isinstance(data.get(key), dict):
            dim = data[key]
            break
    if dim is None:
        raise bad("Invalid calibration dimensions")
    cw, ch = int(dim.get("w", 0)), int(dim.get("h", 0))
    if cw <= 0 or ch <= 0:
        raise bad("Invalid calibration dimensions")
    return np.array(cm, np.float64), np.array(dc, np.float64), cw, ch


def lens_undistort(img: np.ndarray, k: np.ndarray, d: np.ndarray, calib_w: int, calib_h: int,
                   intensity: float) -> np.ndarray:
    h, w = img.shape[:2]
    K = k.astype(np.float64).copy()
    sx, sy = w / calib_w, h / calib_h
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    D = (-(d.astype(np.float64) * intensity)).reshape(4, 1)
    rgb = np.ascontiguousarray(img[..., :3])
    out = cv2.fisheye.undistortImage(rgb, K, D, Knew=K, new_size=(w, h))
    return out.reshape(h, w, 3).astype(F32)  # upstream converts RGBA->RGB->RGBA: output is opaque


# ---------------------------------------------------------------------------
# Kotlin XorWowRandom (kotlin.random.Random(seed)) for ImageExpander
# ---------------------------------------------------------------------------

@njit(cache=True)
def _i32(v):
    v &= 0xFFFFFFFF
    if v >= 0x80000000:
        v -= 0x100000000
    return v


@njit(cache=True)
def _xw_next(st):
    # st: int64[6] = x, y, z, w, v, addend (int32 semantics)
    t = st[0]
    t = _i32(t ^ ((t & 0xFFFFFFFF) >> 2))
    st[0] = st[1]
    st[1] = st[2]
    st[2] = st[3]
    v0 = st[4]
    st[3] = v0
    t = _i32((t ^ _i32(t << 1)) ^ v0 ^ _i32(v0 << 4))
    st[4] = t
    st[5] = _i32(st[5] + 362437)
    return _i32(t + st[5])


@njit(cache=True)
def _xw_init(seed):
    st = np.zeros(6, np.int64)
    s1 = _i32(seed)
    s2 = s1 >> 31
    st[0] = s1
    st[1] = s2
    st[2] = 0
    st[3] = 0
    st[4] = _i32(~s1)
    st[5] = _i32(_i32(s1 << 10) ^ ((s2 & 0xFFFFFFFF) >> 4))
    for _ in range(64):
        _xw_next(st)
    return st


@njit(cache=True)
def _xw_next_int_range(st, frm, until):
    n = until - frm  # always > 0 here
    if (n & -n) == n:
        bit_count = 0
        m = n
        while m > 1:
            m >>= 1
            bit_count += 1
        r = _xw_next(st)
        if bit_count == 0:
            rnd = 0
        else:
            rnd = (r & 0xFFFFFFFF) >> (32 - bit_count)
        return frm + rnd
    while True:
        bits = (_xw_next(st) & 0xFFFFFFFF) >> 1
        v = bits % n
        if _i32(bits - v + (n - 1)) >= 0:
            return frm + v


@njit(cache=True)
def _xw_inclusive(st, frm, to):
    if frm >= to:
        return frm
    return _xw_next_int_range(st, frm, to + 1)


# ---------------------------------------------------------------------------
# ImageExpander (opencv-tools): patch-based texture synthesis outpainting
# ---------------------------------------------------------------------------

@njit(cache=True)
def _smooth_step_i(position, length):
    den = length - 1
    if den < 1:
        den = 1
    x = np.float32(position) / np.float32(den)
    if x < 0:
        x = np.float32(0)
    elif x > 1:
        x = np.float32(1)
    return x * x * (np.float32(3) - np.float32(2) * x)


@njit(cache=True)
def _mix(res, ry, rx, src, sy, sx, amount):
    keep = np.float32(1) - amount
    for ch in range(4):
        res[ry, rx, ch] = np.uint8(np.int32(keep * np.float32(res[ry, rx, ch]) + amount * np.float32(src[sy, sx, ch])))


@njit(cache=True)
def _bounds(cx, cy, ps, W, H):
    r = ps // 2
    left = max(cx - r, 0)
    top = max(cy - r, 0)
    return left, top, min(left + ps, W), min(top + ps, H)


@njit(cache=True)
def _expand_synth(src, res, known, sl, st_, place_x, place_y, ps, stride, req_overlap, cand_count, band,
                  blend_w, max_passes, seed):
    sh, sw = src.shape[0], src.shape[1]
    H, W = res.shape[0], res.shape[1]
    rng = _xw_init(seed)
    remaining = 0
    for y in range(H):
        for x in range(W):
            if not known[y, x]:
                remaining += 1
    required = req_overlap
    for pss in range(max_passes):
        if remaining == 0:
            return
        filled_pass = 0
        relaxed = 1 if pss >= max_passes - 2 else required
        for pi in range(place_x.shape[0]):
            left, top, right, bottom = _bounds(place_x[pi], place_y[pi], ps, W, H)
            bw = right - left
            bh = bottom - top
            has_unknown = False
            for y in range(top, bottom):
                for x in range(left, right):
                    if not known[y, x]:
                        has_unknown = True
                        break
                if has_unknown:
                    break
            if not has_unknown:
                continue
            overlap = 0
            for y in range(top, bottom, stride):
                for x in range(left, right, stride):
                    if known[y, x]:
                        overlap += 1
            if overlap < relaxed:
                continue
            # --- SourceSearchArea.forTarget
            s_right = sl + sw - 1
            s_bottom = st_ + sh - 1
            max_pl = sw - bw
            max_pt = sh - bh
            tcx = left + bw // 2
            tcy = top + bh // 2
            if tcx < sl:
                min_l, max_l = 0, min(band, max_pl)
            elif tcx > s_right:
                min_l, max_l = max(max_pl - band, 0), max_pl
            else:
                min_l, max_l = 0, max_pl
            if tcy < st_:
                min_t, max_t = 0, min(band, max_pt)
            elif tcy > s_bottom:
                min_t, max_t = max(max_pt - band, 0), max_pt
            else:
                min_t, max_t = 0, max_pt
            # --- anchor patch
            ax = min(max(left + bw // 2 - sl, 0), sw - 1)
            ay = min(max(top + bh // 2 - st_, 0), sh - 1)
            best_l = min(max(ax - bw // 2, min_l), max_l)
            best_t = min(max(ay - bh // 2, min_t), max_t)
            best_err = np.int64(0)
            for k in range(cand_count + 1):
                if k == 0:
                    cl, ct = best_l, best_t
                else:
                    cl = _xw_inclusive(rng, min_l, max_l)
                    ct = _xw_inclusive(rng, min_t, max_t)
                err = np.int64(0)
                samples = 0
                for yy in range(0, bh, stride):
                    for xx in range(0, bw, stride):
                        if known[top + yy, left + xx]:
                            dr = np.int64(res[top + yy, left + xx, 1]) - np.int64(src[ct + yy, cl + xx, 1])
                            dg = np.int64(res[top + yy, left + xx, 2]) - np.int64(src[ct + yy, cl + xx, 2])
                            db = np.int64(res[top + yy, left + xx, 3]) - np.int64(src[ct + yy, cl + xx, 3])
                            da = np.int64(res[top + yy, left + xx, 0]) - np.int64(src[ct + yy, cl + xx, 0])
                            err += dr * dr * 3 + dg * dg * 4 + db * db * 2 + da * da * 2
                            samples += 1
                if samples == 0:
                    e = np.int64(9223372036854775807)
                else:
                    e = err // samples
                if k == 0:
                    best_err = e
                elif e < best_err:
                    best_err = e
                    best_l, best_t = cl, ct
            # --- copyUnknownPatch
            blend_left = False
            blend_right = False
            for y in range(top, bottom):
                if known[y, left]:
                    blend_left = True
                if known[y, left + bw - 1]:
                    blend_right = True
            blend_top = False
            blend_bottom = False
            for x in range(left, right):
                if known[top, x]:
                    blend_top = True
                if known[top + bh - 1, x]:
                    blend_bottom = True
            filled = 0
            for yy in range(bh):
                for xx in range(bw):
                    ry = top + yy
                    rx = left + xx
                    sy = best_t + yy
                    sx = best_l + xx
                    if not known[ry, rx]:
                        for ch in range(4):
                            res[ry, rx, ch] = src[sy, sx, ch]
                        known[ry, rx] = True
                        filled += 1
                    else:
                        blend = np.float32(1)
                        if blend_left and xx < blend_w:
                            blend = min(blend, _smooth_step_i(xx, blend_w))
                        if blend_top and yy < blend_w:
                            blend = min(blend, _smooth_step_i(yy, blend_w))
                        if blend_right and bw - xx - 1 < blend_w:
                            blend = min(blend, _smooth_step_i(bw - xx - 1, blend_w))
                        if blend_bottom and bh - yy - 1 < blend_w:
                            blend = min(blend, _smooth_step_i(bh - yy - 1, blend_w))
                        inside = sl <= rx < sl + sw and st_ <= ry < st_ + sh
                        if inside:
                            edge = np.float32(0)
                            if sl > 0:
                                d = rx - sl
                                if 0 <= d < blend_w:
                                    edge = max(edge, np.float32(1) - _smooth_step_i(d, blend_w))
                            if st_ > 0:
                                d = ry - st_
                                if 0 <= d < blend_w:
                                    edge = max(edge, np.float32(1) - _smooth_step_i(d, blend_w))
                            if s_right < W - 1:
                                d = s_right - rx
                                if 0 <= d < blend_w:
                                    edge = max(edge, np.float32(1) - _smooth_step_i(d, blend_w))
                            if s_bottom < H - 1:
                                d = s_bottom - ry
                                if 0 <= d < blend_w:
                                    edge = max(edge, np.float32(1) - _smooth_step_i(d, blend_w))
                            amount = min(blend, edge)
                            if amount > 0:
                                _mix(res, ry, rx, src, sy, sx, amount)
                        elif blend < 1:
                            _mix(res, ry, rx, src, sy, sx, blend)
            filled_pass += filled
            remaining -= filled
        if filled_pass == 0:
            required = max(required // 2, 1)


def _stepped(size: int, step: int) -> list[int]:
    return sorted(set(range(0, size, step)) | {size - 1})


def expand_image(img: np.ndarray, top: int, left: int, right: int, bottom: int) -> np.ndarray:
    had = has_alpha(img)
    if top == 0 and left == 0 and right == 0 and bottom == 0:
        return img
    rgba = to_u8(ensure_rgba(img))
    sh, sw = rgba.shape[:2]
    rh, rw = sh + top + bottom, sw + left + right
    src = np.ascontiguousarray(rgba[..., [3, 0, 1, 2]])  # ARGB order, uint8
    res = np.zeros((rh, rw, 4), np.uint8)
    known = np.zeros((rh, rw), np.bool_)
    res[top:top + sh, left:left + sw] = src
    known[top:top + sh, left:left + sw] = True
    if not (sw == 1 and sh == 1):
        min_side = min(sw, sh)
        ps = min(min_side, max(min_side // 8, 7))
        ps = min(ps, 120)
        if ps % 2 == 0:
            ps -= 1
        ps = max(ps, 1)
        step = max(ps // 2, 1)
        stride = 4 if ps >= 25 else (2 if ps >= 15 else 1)
        sampled = ((ps + stride - 1) // stride) ** 2
        max_exp = max(left, right, top, bottom)
        passes = min(max(max_exp // step + 6, 8), 48)
        cand = 32 if sw * sh < 128 * 128 else 64
        band = min(ps * 3, max(sw, sh))
        blend_w = max(ps - step, 1)
        req = max(sampled // 5, 1)
        xs, ys = _stepped(rw, step), _stepped(rh, step)
        px, py, pd = [], [], []
        s_right, s_bottom = left + sw - 1, top + sh - 1
        for y in ys:
            for x in xs:
                l0 = max(x - ps // 2, 0)
                t0 = max(y - ps // 2, 0)
                r0 = min(l0 + ps, rw)
                b0 = min(t0 + ps, rh)
                if l0 >= left and t0 >= top and r0 <= left + sw and b0 <= top + sh:
                    continue
                dx = left - x if x < left else (x - s_right if x > s_right else 0)
                dy = top - y if y < top else (y - s_bottom if y > s_bottom else 0)
                px.append(x)
                py.append(y)
                pd.append(dx * dx + dy * dy)
        order = np.argsort(np.asarray(pd, np.int64), kind="stable")
        place_x = np.asarray(px, np.int64)[order]
        place_y = np.asarray(py, np.int64)[order]
        seed = int32(sw * 73856093 + sh * 19349663 + rw * 83492791 + rh * 265443576)
        _expand_synth(src, res, known, left, top, place_x, place_y, ps, stride, req, cand, band, blend_w,
                      passes, seed)
    # fillUnknownWithProjection
    if not known.all():
        yy, xx = np.nonzero(~known)
        sy = np.clip(yy - top, 0, sh - 1)
        sx = np.clip(xx - left, 0, sw - 1)
        res[yy, xx] = src[sy, sx]
    out = from_u8(res[..., [1, 2, 3, 0]])
    return finish(out, had)


# ---------------------------------------------------------------------------
# SeamCarver (opencv-tools)
# ---------------------------------------------------------------------------

PROTECTED_MASK_ENERGY = 1_000_000_000.0
REMOVAL_MASK_ENERGY = -1_000_000_000.0


@njit(cache=True)
def _forward_seam(gray, mask_energy, use_energy, protected, forbid):
    rows, cols = gray.shape
    prev = np.empty(cols, np.float64)
    cur = np.empty(cols, np.float64)
    back = np.zeros((rows, cols), np.int8)  # column offset -1/0/+1
    for c in range(cols):
        if forbid and protected[0, c]:
            prev[c] = np.inf
        else:
            prev[c] = mask_energy[0, c] if use_energy else 0.0
    for r in range(1, rows):
        for c in range(cols):
            if forbid and protected[r, c]:
                cur[c] = np.inf
                back[r, c] = 0
                continue
            up = np.float64(gray[r - 1, c])
            lft = np.float64(gray[r, max(c - 1, 0)])
            rgt = np.float64(gray[r, min(c + 1, cols - 1)])
            cost_up = abs(rgt - lft)
            cost_left = abs(up - lft) + cost_up
            cost_right = abs(up - rgt) + cost_up
            best = prev[c] + cost_up
            bc = c
            if c > 0:
                t = prev[c - 1] + cost_left
                if t < best:
                    best = t
                    bc = c - 1
            if c < cols - 1:
                t = prev[c + 1] + cost_right
                if t < best:
                    best = t
                    bc = c + 1
            if np.isinf(best):
                cur[c] = np.inf
            else:
                cur[c] = best + (mask_energy[r, c] if use_energy else 0.0)
            back[r, c] = bc - c
        tmp = prev
        prev = cur
        cur = tmp
    min_col = -1
    min_val = np.inf
    for c in range(cols):
        if prev[c] < min_val:
            min_val = prev[c]
            min_col = c
    seam = np.zeros(rows, np.int64)
    if min_col < 0 or np.isinf(min_val):
        return seam, False
    col = min_col
    for r in range(rows - 1, -1, -1):
        seam[r] = col
        if r > 0:
            col += back[r, col]
    return seam, True


@njit(cache=True)
def _backward_seam(energy, protected, avoid):
    rows, cols = energy.shape
    prev = np.empty(cols, np.float64)
    cur = np.empty(cols, np.float64)
    back = np.zeros((rows, cols), np.int8)  # column offset -1/0/+1
    for c in range(cols):
        prev[c] = np.inf if (avoid and protected[0, c]) else energy[0, c]
    for r in range(1, rows):
        for c in range(cols):
            if avoid and protected[r, c]:
                cur[c] = np.inf
                back[r, c] = 0
                continue
            mp = prev[c]
            bc = c
            if c > 0 and prev[c - 1] < mp:
                mp = prev[c - 1]
                bc = c - 1
            if c < cols - 1 and prev[c + 1] < mp:
                mp = prev[c + 1]
                bc = c + 1
            if avoid and np.isinf(mp):
                cur[c] = np.inf
            else:
                cur[c] = energy[r, c] + mp
            back[r, c] = bc - c
        tmp = prev
        prev = cur
        cur = tmp
    seam = np.zeros(rows, np.int64)
    if avoid:
        min_col = -1
        min_val = np.inf
        for c in range(cols):
            if prev[c] < min_val:
                min_val = prev[c]
                min_col = c
        if min_col < 0 or np.isinf(min_val):
            return seam, False
    else:
        min_col = 0
        min_val = prev[0]
        for c in range(1, cols):
            if prev[c] < min_val:
                min_val = prev[c]
                min_col = c
    col = min_col
    for r in range(rows - 1, -1, -1):
        seam[r] = col
        if r > 0:
            col += back[r, col]
    return seam, True


@njit(cache=True)
def _remove_seam(arr, seam):
    rows, cols, ch = arr.shape
    out = np.empty((rows, cols - 1, ch), arr.dtype)
    for r in range(rows):
        s = seam[r]
        for c in range(s):
            for k in range(ch):
                out[r, c, k] = arr[r, c, k]
        for c in range(s + 1, cols):
            for k in range(ch):
                out[r, c - 1, k] = arr[r, c, k]
    return out


@njit(cache=True)
def _insert_seams(arr, seams):
    rows, cols, ch = arr.shape
    n = seams.shape[0]
    out = np.empty((rows, cols + n, ch), arr.dtype)
    for r in range(rows):
        rs = np.empty(n, np.int64)
        for i in range(n):
            rs[i] = min(max(seams[i, r], 0), cols - 1)
        rs.sort()
        d = 0
        si = 0
        for c in range(cols):
            for k in range(ch):
                out[r, d, k] = arr[r, c, k]
            d += 1
            while si < n and rs[si] == c:
                nb = c + 1 if c < cols - 1 else max(c - 1, 0)
                for k in range(ch):
                    out[r, d, k] = (np.int64(arr[r, c, k]) + np.int64(arr[r, nb, k])) // 2
                d += 1
                si += 1
    return out


class _Carver:
    """SeamCarver.carveMat on a packed uint8 array: R, G, B, A, gray (RGBA2GRAY) [, binary mask].

    Gray is a per-pixel conversion, so carrying it along (and removing seams from
    it) equals recomputing it from the carved image like upstream does.
    """

    def __init__(self, arr: np.ndarray, has_mask: bool, backward: bool, ctx):
        self.arr = arr
        self.has_mask = has_mask
        self.backward = backward
        self.ctx = ctx

    def find_seam(self, arr: np.ndarray, removal: bool, use_mask: bool = True) -> np.ndarray:
        rows, cols = arr.shape[:2]
        if cols <= 1:
            return np.zeros(rows, np.int64)
        masked = self.has_mask and use_mask
        prot = (arr[..., 5] > 0) if masked else np.zeros((1, 1), np.bool_)
        gray = arr[..., 4]  # uint8 view; integer values, so the DP / Sobel results equal upstream's CV_64F maths
        if self.backward:
            gray = np.ascontiguousarray(gray)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            energy = np.sqrt(gx * gx + gy * gy)
            if removal:
                if masked:
                    energy = np.where(prot, REMOVAL_MASK_ENERGY, energy)
                return _backward_seam(energy, prot, False)[0]
            if masked:
                seam, ok = _backward_seam(energy, prot, True)
                if ok:
                    return seam
                energy = energy + np.where(prot, PROTECTED_MASK_ENERGY, 0.0)
            return _backward_seam(energy, prot, False)[0]
        if masked:
            menergy = np.where(prot, REMOVAL_MASK_ENERGY if removal else PROTECTED_MASK_ENERGY, 0.0)
        else:
            menergy = np.zeros((1, 1), np.float64)
        if not removal and masked:
            seam, ok = _forward_seam(gray, menergy, True, prot, True)
            if ok:
                return seam
        seam, ok = _forward_seam(gray, menergy, masked, prot, False)
        return seam if ok else np.zeros(rows, np.int64)

    def resize_width(self, target: int, use_mask: bool) -> None:
        arr = self.arr
        while arr.shape[1] > target:
            self.ctx.check()
            arr = _remove_seam(arr, self.find_seam(arr, False, use_mask))
        while arr.shape[1] < target:
            cols = arr.shape[1]
            batch = 1 if cols <= 1 else min(max(kround(cols * 0.15), 1), cols - 1)
            arr = _insert_seams(arr, self.seams_for_insertion(arr, min(batch, target - cols), use_mask))
            arr[..., 4] = cv2.cvtColor(np.ascontiguousarray(arr[..., :4]), cv2.COLOR_RGBA2GRAY)
        self.arr = arr

    def seams_for_insertion(self, arr: np.ndarray, count: int, use_mask: bool) -> np.ndarray:
        rows, cols = arr.shape[:2]
        if cols <= 1:
            return np.zeros((count, rows), np.int64)
        idx = np.ascontiguousarray(np.broadcast_to(np.arange(cols, dtype=np.int32), (rows, cols))[..., None])
        work = arr
        seams = []
        for _ in range(min(count, cols - 1)):
            self.ctx.check()
            seam = self.find_seam(work, False, use_mask)
            seams.append(idx[np.arange(rows), seam, 0].astype(np.int64))
            idx = _remove_seam(idx, seam)
            work = _remove_seam(work, seam)
        return np.asarray(seams, np.int64).reshape(len(seams), rows)

    def remove_masked_area(self) -> None:
        arr = self.arr
        ys, xs = np.nonzero(arr[..., 5] > 0)
        if len(ys) == 0:
            return
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        vertical = bw <= bh or arr.shape[0] <= 1
        flip = not vertical and arr.shape[1] > 1
        if flip:
            arr = arr.transpose(1, 0, 2).copy()
        while arr.shape[1] > 1 and (arr[..., 5] > 0).any():
            self.ctx.check()
            seam = self.find_seam(arr, True)
            if not (arr[np.arange(arr.shape[0]), seam, 5] > 0).any():
                break
            arr = _remove_seam(arr, seam)
        self.arr = arr.transpose(1, 0, 2).copy() if flip else arr

    def carve(self, width: int, height: int, use_mask_as_removal: bool) -> np.ndarray:
        use_mask = self.has_mask
        if use_mask_as_removal and self.has_mask and (self.arr[..., 5] > 0).any():
            self.remove_masked_area()
            use_mask = False  # upstream drops the mask after removing the masked area
        self.resize_width(max(width, 1), use_mask)
        self.arr = self.arr.transpose(1, 0, 2).copy()
        self.resize_width(max(height, 1), use_mask)
        return self.arr.transpose(1, 0, 2)[..., :4].copy()


def seam_carve(img: np.ndarray, width: int, height: int, mask: np.ndarray | None, backward: bool,
               use_mask_as_removal: bool, ctx) -> np.ndarray:
    had = has_alpha(img)
    u8 = to_u8(ensure_rgba(img))
    chans = [u8, cv2.cvtColor(u8, cv2.COLOR_RGBA2GRAY)[..., None]]
    if mask is not None:
        mrgba = ensure_rgba(mask)
        if mrgba.shape[:2] != img.shape[:2]:
            mrgba = cv2.resize(mrgba, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        prem = to_u8(np.dstack([mrgba[..., :3] * mrgba[..., 3:4], mrgba[..., 3:4]]))
        chans.append(cv2.cvtColor(prem, cv2.COLOR_RGBA2GRAY)[..., None])
    arr = np.ascontiguousarray(np.concatenate(chans, axis=2))
    out = _Carver(arr, mask is not None, backward, ctx).carve(width, height, use_mask_as_removal)
    return finish(from_u8(out), had)


# ---------------------------------------------------------------------------
# java.util.Random (TornEdge) and Kotlin data-class hash
# ---------------------------------------------------------------------------

class JavaRandom:
    _MULT = 0x5DEECE66D
    _MASK = (1 << 48) - 1

    def __init__(self, seed: int) -> None:
        self.seed = (seed ^ self._MULT) & self._MASK

    def _next(self, bits: int) -> int:
        self.seed = (self.seed * self._MULT + 0xB) & self._MASK
        return int32(self.seed >> (48 - bits))

    def next_int(self, bound: int) -> int:
        if bound & -bound == bound:
            return (bound * self._next(31)) >> 31
        while True:
            bits = self._next(31)
            val = bits % bound
            if bits - val + (bound - 1) < (1 << 31):
                return val


def kotlin_hash(*fields) -> int:
    h = 0
    for i, f in enumerate(fields):
        v = (1231 if f else 1237) if isinstance(f, bool) else int(f)
        h = int32(v) if i == 0 else int32(h * 31 + v)
    return h
