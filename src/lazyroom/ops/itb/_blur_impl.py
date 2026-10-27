"""Private kernels for the ImageToolbox "Blur" group port (see blur.py).

Ported from the implementations ImageToolbox actually calls:

* aire (MIT, awxkee) -> libblur 0.19.3 (BSD-3/Apache-2.0, awxkee) for most
  blurs, aire C++ for poisson / zoom / tilt-shift / 4-degree gaussian,
* Trickle (Apache-2.0, T8RIN) C++ for fast / stack / shuffle blur,
* ImageToolboxLibs' native jhlabs port (Apache-2.0) for smart / lens blur
  and diffuse,
* GPUImage (BSD-3) for the zoom blur shader.

Buffers are float32 (H, W, C) in 0..1.  Android bitmaps are premultiplied and
the native code works on the raw premultiplied RGBA memory, so RGBA inputs
are premultiplied before blurring and un-premultiplied afterwards
(:func:`run_premul`).  Algorithms run in float precision instead of the
upstream 8/16-bit storage (no intermediate quantisation).
"""
from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np
from numba import njit, prange

F32 = np.float32

EDGE_MODES = ("clamp", "wrap", "reflect", "reflect_101")
EDGE_LABELS = {"clamp": "Clamp", "wrap": "Wrap", "reflect": "Mirror", "reflect_101": "Mirror 101"}
_EDGE_CODE = {"clamp": 0, "wrap": 1, "reflect": 2, "reflect_101": 3}
TRANSFER_FUNCS = ("srgb", "rec709", "gamma2p2", "gamma2p8")
TRANSFER_LABELS = {"srgb": "sRGB", "rec709": "Rec.709", "gamma2p2": "Gamma 2.2", "gamma2p8": "Gamma 2.8"}


# ---------------------------------------------------------------------------
# Small helpers (Kotlin rounding, premultiplied alpha)
# ---------------------------------------------------------------------------

def kround(x: float) -> int:
    """Kotlin ``Float.roundToInt()`` (round half up)."""
    return int(math.floor(float(x) + 0.5))


def nearest_odd(x: float) -> int:
    """ImageToolbox ``roundToNearestOdd``: round, then bump even values up by one."""
    n = kround(x)
    return n if n % 2 != 0 else n + 1


def premul(img: np.ndarray) -> np.ndarray:
    if img.shape[2] == 4:
        a = img[..., 3:4]
        return np.concatenate([img[..., :3] * a, a], axis=2).astype(F32)
    return np.ascontiguousarray(img, dtype=F32)


def unpremul(data: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    """Premultiplied RGB(A) -> straight RGBA.  ``alpha`` overrides channel 3."""
    a = (data[..., 3:4] if alpha is None else alpha[..., None]).astype(F32)
    rgb = np.zeros(data.shape[:2] + (3,), F32)
    np.divide(data[..., :3], a, out=rgb, where=a > 1e-6)
    return np.concatenate([rgb, a], axis=2)


def run_premul(img: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply ``fn`` to all channels of the premultiplied buffer (aire/libblur RGBA semantics)."""
    if img.shape[2] == 3:
        return fn(np.ascontiguousarray(img, dtype=F32))
    return unpremul(fn(premul(img)))


def run_premul_rgb(img: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply ``fn`` to the premultiplied RGB only, alpha untouched (filters that skip channel 3)."""
    if img.shape[2] == 3:
        return fn(np.ascontiguousarray(img, dtype=F32))
    a = img[..., 3]
    return unpremul(fn(np.ascontiguousarray(img[..., :3] * a[..., None], dtype=F32)), a)


def _c(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img, dtype=F32)


def _cv_out(out: np.ndarray, like: np.ndarray) -> np.ndarray:
    return out.reshape(like.shape[:2] + (-1,)).astype(F32, copy=False)


# ---------------------------------------------------------------------------
# Edge handling (libblur / fast_morphology index rules)
# ---------------------------------------------------------------------------

def edge_index(n: int, before: int, after: int, mode: str) -> np.ndarray:
    """Source indices for positions -before .. n+after-1 using libblur's edge rules."""
    i = np.arange(-before, n + after, dtype=np.int64)
    if mode == "clamp":
        return np.clip(i, 0, n - 1)
    if mode == "wrap":
        return np.mod(i, n)
    inside = (i >= 0) & (i < n)
    if mode == "reflect":
        r = n - np.mod(i, n) - 1
    else:  # reflect_101
        nr = n - 1
        r = np.zeros_like(i) if nr == 0 else nr - np.mod(i, nr)
    return np.where(inside, i, r)


_CV_BORDER = {"clamp": cv2.BORDER_REPLICATE, "wrap": cv2.BORDER_WRAP, "reflect": cv2.BORDER_REFLECT,
              "reflect_101": cv2.BORDER_REFLECT_101}


def pad(img: np.ndarray, top: int, bottom: int, left: int, right: int, mode: str) -> np.ndarray:
    h, w = img.shape[:2]
    if max(top, bottom) < h - 1 and max(left, right) < w - 1:
        # identical to libblur's rules while the pad is smaller than the image
        out = cv2.copyMakeBorder(img, top, bottom, left, right, _CV_BORDER[mode])
        return out.reshape((h + top + bottom, w + left + right) + img.shape[2:])
    iy = edge_index(h, top, bottom, mode)
    ix = edge_index(w, left, right, mode)
    return np.ascontiguousarray(np.take(np.take(img, iy, axis=0), ix, axis=1))


def trim_kernel(k: np.ndarray, anchor: int, rel: float = 1e-9) -> tuple[np.ndarray, int]:
    """Drop numerically irrelevant tail taps (weights below ``rel`` of the peak)."""
    k = np.asarray(k, np.float64)
    keep = np.nonzero(np.abs(k) > np.abs(k).max() * rel)[0]
    lo, hi = int(keep[0]), int(keep[-1])
    return k[lo:hi + 1], anchor - lo


def sep_correlate(img: np.ndarray, kx: np.ndarray, ky: np.ndarray, mode: str,
                  ax: int | None = None, ay: int | None = None, trim: bool = True) -> np.ndarray:
    """out[y, x] = sum_j ky[j] sum_i kx[i] src[y + j - ay, x + i - ax] with libblur edge handling."""
    ax = len(kx) // 2 if ax is None else ax
    ay = len(ky) // 2 if ay is None else ay
    if trim:
        kx, ax = trim_kernel(kx, ax)
        ky, ay = trim_kernel(ky, ay)
    h, w = img.shape[:2]
    kx32, ky32 = np.asarray(kx, F32), np.asarray(ky, F32)
    pads = (ay, len(ky) - 1 - ay, ax, len(kx) - 1 - ax)
    if _cv_native(mode, h, w, *pads):
        out = cv2.sepFilter2D(_c(img), cv2.CV_32F, kx32, ky32, anchor=(ax, ay), borderType=_CV_BORDER[mode])
        return out.reshape(img.shape)
    p = pad(_c(img), *pads, mode)
    out = cv2.sepFilter2D(p, cv2.CV_32F, kx32, ky32, anchor=(ax, ay), borderType=cv2.BORDER_CONSTANT)
    out = out.reshape(p.shape)
    return np.ascontiguousarray(out[ay:ay + h, ax:ax + w])


def _cv_native(mode: str, h: int, w: int, top: int, bottom: int, left: int, right: int) -> bool:
    """OpenCV's own border extrapolation equals libblur's rules (no wrap for filters, pad < size - 1)."""
    return mode != "wrap" and max(top, bottom) < h - 1 and max(left, right) < w - 1


def correlate2d(img: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
    """libblur ``filter_2d`` (correlation, anchor = size / 2) with libblur edge handling."""
    kh, kw = kernel.shape
    ay, ax = kh // 2, kw // 2
    h, w = img.shape[:2]
    pads = (ay, kh - 1 - ay, ax, kw - 1 - ax)
    ys, xs = np.nonzero(kernel)
    if len(ys) <= 64:  # sparse kernels (lines, arcs, crosses): direct tap sum is much cheaper than a DFT
        p = pad(_c(img), *pads, mode)
        d = p if p.ndim == 3 else p[..., None]
        out = _sparse_taps(d, ys.astype(np.int64), xs.astype(np.int64),
                           kernel[ys, xs].astype(np.float64), h, w)
        return out.reshape(img.shape)
    if _cv_native(mode, h, w, *pads):
        out = cv2.filter2D(_c(img), cv2.CV_32F, kernel.astype(F32), anchor=(ax, ay), borderType=_CV_BORDER[mode])
        return out.reshape(img.shape)
    p = pad(_c(img), *pads, mode)
    out = cv2.filter2D(p, cv2.CV_32F, kernel.astype(F32), anchor=(ax, ay), borderType=cv2.BORDER_CONSTANT)
    out = out.reshape(p.shape)
    return np.ascontiguousarray(out[ay:ay + h, ax:ax + w])


@njit(cache=True, parallel=True)
def _sparse_taps(p, ty, tx, tw, h, w):
    nc = p.shape[2]
    out = np.zeros((h, w, nc), np.float32)
    for y in prange(h):
        for t in range(ty.size):
            py = y + ty[t]
            ox = tx[t]
            wt = tw[t]
            for x in range(w):
                for c in range(nc):
                    out[y, x, c] += p[py, x + ox, c] * wt
    return out


# ---------------------------------------------------------------------------
# Transfer functions (libblur gamma_curves.rs)
# ---------------------------------------------------------------------------

_TF_CODE = {"srgb": 0, "rec709": 1, "gamma2p2": 2, "gamma2p8": 3}


@njit(cache=True, parallel=True)
def _transfer(x, code, forward):
    out = np.empty_like(x)
    for i in prange(x.size):
        v = x[i]
        if v <= 0.0:
            r = 0.0
        elif v >= 1.0:
            r = 1.0
        elif code == 0:
            if forward:
                r = v / 12.92 if v < 12.92 * 0.0030412825601275209 else \
                    ((v + 0.0550107189475866) / 1.0550107189475866) ** 2.4
            else:
                r = v * 12.92 if v < 0.0030412825601275209 else \
                    1.0550107189475866 * v ** (1.0 / 2.4) - 0.0550107189475866
        elif code == 1:
            if forward:
                r = v / 4.5 if v < 4.5 * 0.018053968510807 else \
                    ((v + 0.09929682680944) / 1.09929682680944) ** (1.0 / 0.45)
            else:
                r = v * 4.5 if v < 0.018053968510807 else 1.09929682680944 * v ** 0.45 - 0.09929682680944
        else:
            g = 2.2 if code == 2 else 2.8
            r = v ** g if forward else v ** (1.0 / g)
        out[i] = min(max(r, 0.0), 1.0)
    return out


_TF_LUTS: dict[tuple[str, bool], np.ndarray] = {}


def _tf_lut(tf: str, forward: bool) -> np.ndarray:
    """65536-entry table like libblur's 16-bit linearisation / gamma tables (sampled exactly)."""
    key = (tf, forward)
    if key not in _TF_LUTS:
        grid = np.linspace(0.0, 1.0, 65536)
        _TF_LUTS[key] = _transfer(grid, _TF_CODE[tf], forward)
    return _TF_LUTS[key]


@njit(cache=True, parallel=True)
def _apply_lut(x, lut):
    out = np.empty_like(x)
    n = lut.size - 1
    for i in prange(x.size):
        v = min(max(x[i], 0.0), 1.0) * n
        j = min(int(v), n - 1)
        f = v - j
        out[i] = lut[j] * (1.0 - f) + lut[j + 1] * f
    return out


def linearize(x: np.ndarray, tf: str) -> np.ndarray:
    """libblur ``TransferFunction::linearize`` (sRGB, Rec.709, pure gamma 2.2 / 2.8)."""
    x = np.ascontiguousarray(x, dtype=F32)
    return _apply_lut(x.ravel(), _tf_lut(tf, True)).reshape(x.shape)


def gamma(x: np.ndarray, tf: str) -> np.ndarray:
    """libblur ``TransferFunction::gamma`` (inverse of :func:`linearize`)."""
    x = np.ascontiguousarray(x, dtype=F32)
    return _apply_lut(x.ravel(), _tf_lut(tf, False)).reshape(x.shape)


def in_transfer(data: np.ndarray, tf: str, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """libblur ``linearize(tf, may_have_alpha=true)`` -> fn -> ``gamma8``: alpha stays linear."""
    lin = data.copy()
    lin[..., :3] = linearize(data[..., :3], tf)
    out = fn(lin)
    out[..., :3] = gamma(out[..., :3], tf)
    return out


# ---------------------------------------------------------------------------
# libblur box family: box / tent / gaussian box / stack blur
# ---------------------------------------------------------------------------

def lb_box(data: np.ndarray, kx: int, ky: int) -> np.ndarray:
    """libblur ``box_blur``: kernel K -> r = max(K/2, 1); running sum over [x-r+1, x+r] / 2r, clamp edges."""
    if kx == 1 and ky == 1:
        return data
    rx, ry = max(kx // 2, 1), max(ky // 2, 1)
    out = cv2.blur(_c(data), (2 * rx, 2 * ry), anchor=(rx - 1, ry - 1), borderType=cv2.BORDER_REPLICATE)
    return _cv_out(out, data)


def create_box_gauss(sigma: float, n: int) -> list[int]:
    """libblur ``create_box_gauss`` (f32 arithmetic)."""
    sigma = np.float32(sigma)
    nf = np.float32(n)
    w_ideal = np.sqrt(np.float32(12.0) * sigma * sigma / nf) + np.float32(1.0)
    wl = int(np.floor(w_ideal))
    if wl % 2 == 0:
        wl -= 1
    wu = wl + 2
    wlf = np.float32(wl)
    m_ideal = (np.float32(12.0) * sigma * sigma - nf * wlf * wlf - np.float32(4.0) * nf * wlf
               - np.float32(3.0) * nf) / (np.float32(-4.0) * wlf - np.float32(4.0))
    m = max(0, kround(float(m_ideal)))  # Rust f32 -> usize saturates at 0
    sizes = []
    for i in range(n):
        v = (wl if i < m else wu) // 2
        if v % 2 == 0:
            v += 1
        sizes.append(v)
    return sizes


def lb_clt(data: np.ndarray, sigma: float, passes: int) -> np.ndarray:
    """libblur ``tent_blur`` (passes=2) / ``gaussian_box_blur`` (passes=3)."""
    out = data
    for v in create_box_gauss(sigma, passes):
        out = lb_box(out, 2 * v + 1, 2 * v + 1)
    return out


def triangle_blur(data: np.ndarray, r: int) -> np.ndarray:
    """Stack blur kernel (r+1-|i|)/(r+1)^2 with clamped edges, as two exact box passes."""
    if r < 1:
        return data
    h, w = data.shape[:2]
    p = pad(_c(data), r, r, r, r, "clamp")
    a1 = r // 2
    a2 = r - a1
    p = cv2.blur(p, (r + 1, r + 1), anchor=(a1, a1), borderType=cv2.BORDER_REPLICATE)
    p = cv2.blur(p, (r + 1, r + 1), anchor=(a2, a2), borderType=cv2.BORDER_REPLICATE)
    p = p.reshape((h + 2 * r, w + 2 * r) + data.shape[2:])
    return np.ascontiguousarray(p[r:r + h, r:r + w]).reshape(data.shape)


def centered_box(data: np.ndarray, r: int) -> np.ndarray:
    """Symmetric (2r+1) box with clamped edges (Trickle fastBlur)."""
    if r < 1:
        return data
    return _cv_out(cv2.blur(_c(data), (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REPLICATE), data)


# ---------------------------------------------------------------------------
# libblur fast_gaussian / fast_gaussian_next and aire gaussianApproximation4D
# (recursive integrated-difference filters, ported literally incl. their
# in-place read-ahead behaviour at the image borders)
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _edge_i(i, n, mode):
    if i >= 0 and i < n:
        return i
    if mode == 1:
        return i % n
    if mode == 2:
        return n - (i % n) - 1
    if mode == 3:
        nr = n - 1
        if nr == 0:
            return 0
        return nr - (i % nr)
    return min(max(i, 0), n - 1)


@njit(cache=True, parallel=True)
def _fg2_lines(data, r, mode):
    nl, n, nc = data.shape
    w = 1.0 / (r * r)
    for li in prange(nl):
        buf = np.zeros(1024)
        for c in range(nc):
            dif = 0.0
            s = 0.0
            for y in range(-2 * r, n):
                if y >= 0:
                    data[li, y, c] = s * w
                    dif += buf[(y - r) & 1023] - 2.0 * buf[y & 1023]
                elif y + r >= 0:
                    dif -= 2.0 * buf[y & 1023]
                v = data[li, _edge_i(y + r, n, mode), c]
                dif += v
                s += dif
                buf[(y + r) & 1023] = v


@njit(cache=True, parallel=True)
def _fg3_lines(data, r, mode):
    nl, n, nc = data.shape
    w = 1.0 / (r * r * r)
    off = (3 * r) >> 1
    for li in prange(nl):
        buf = np.zeros(1024)
        for c in range(nc):
            dif = 0.0
            der = 0.0
            s = 0.0
            for y in range(-3 * r, n):
                if y >= 0:
                    data[li, y, c] = s * w
                    dif += 3.0 * (buf[y & 1023] - buf[(y + r) & 1023]) - buf[(y - r) & 1023]
                elif y + r >= 0:
                    dif += 3.0 * (buf[y & 1023] - buf[(y + r) & 1023])
                elif y + 2 * r >= 0:
                    dif -= 3.0 * buf[(y + r) & 1023]
                v = data[li, _edge_i(y + off, n, mode), c]
                dif += v
                der += dif
                s += der
                buf[(y + 2 * r) & 1023] = v


@njit(cache=True, parallel=True)
def _fg4_lines(data, r):
    nl, n, nc = data.shape
    w = 1.0 / float(r * r) / float(r * r)
    for li in prange(nl):
        buf = np.zeros(1024)
        for c in range(nc):
            dif = 0.0
            d2 = 0.0
            d1 = 0.0
            s = 0.0
            for y in range(-4 * r, n):
                if y >= 0:
                    dif += (-4.0 * (buf[(y - r) & 1023] + buf[(y + r) & 1023]) + 6.0 * buf[y & 1023]
                            + buf[(y - 2 * r) & 1023])
                    data[li, y, c] = s * w
                else:
                    if y + 3 * r >= 0:
                        dif -= 4.0 * buf[(y + r) & 1023]
                    if y + 2 * r >= 0:
                        dif += 6.0 * buf[y & 1023]
                    if y + r >= 0:
                        dif -= 4.0 * buf[(y - r) & 1023]
                v = data[li, min(max(y + 2 * r - 1, 0), n - 1), c]
                dif += v
                d2 += dif
                d1 += d2
                s += d1
                buf[(y + 2 * r) & 1023] = v


def _vertical_then_horizontal(data: np.ndarray, line_fn) -> np.ndarray:
    d = np.ascontiguousarray(data, dtype=F32)
    if d.ndim == 2:
        d = d[..., None]
    t = np.ascontiguousarray(d.transpose(1, 0, 2))  # columns as lines
    line_fn(t)
    d = np.ascontiguousarray(t.transpose(1, 0, 2))
    line_fn(d)
    return d.reshape(data.shape)


def fast_gaussian(data: np.ndarray, r: int, edge: str, degree: int) -> np.ndarray:
    """libblur ``fast_gaussian`` (degree 2) / ``fast_gaussian_next`` (degree 3), vertical then horizontal."""
    mode = _EDGE_CODE[edge]
    fn = _fg2_lines if degree == 2 else _fg3_lines
    return _vertical_then_horizontal(data, lambda a: fn(a, int(r), mode))


def fast_gaussian_4d(data: np.ndarray, r: int) -> np.ndarray:
    """aire ``gaussianApproximation4D`` (clamped edges)."""
    return _vertical_then_horizontal(data, lambda a: _fg4_lines(a, int(r)))


# ---------------------------------------------------------------------------
# Gaussian kernels
# ---------------------------------------------------------------------------

def gaussian_kernel_1d(width: int, sigma: float) -> np.ndarray:
    """libblur ``gaussian_kernel_1d`` / aire ``compute1DGaussianKernel`` (mean = width / 2)."""
    width = int(width)
    x = np.arange(width, dtype=np.float64) - (width // 2)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum()


def jh_gaussian_kernel(radius: float) -> np.ndarray:
    """ImageToolboxLibs jhlabs ``gaussian_kernel_exact``: sigma = r/3, taps +-ceil(r), zero beyond r."""
    half = int(math.ceil(radius))
    sigma = radius / 3.0
    x = np.arange(-half, half + 1, dtype=np.float64)
    k = np.where(x * x > radius * radius, 0.0, np.exp(-(x * x) / (2.0 * sigma * sigma)))
    return k / k.sum()


# ---------------------------------------------------------------------------
# Bilateral filters
# ---------------------------------------------------------------------------

def bilateral(data: np.ndarray, kernel: int, spatial_sigma: float, range_sigma: float, edge: str) -> np.ndarray:
    """libblur ``bilateral_filter``: square window, per-channel range weights on 0..1 values.

    Evaluated exactly (to ~1e-7) as a short power series: with
    w_r = exp(-(c-n)^2 / 2s^2) = exp(-c^2/2s^2) exp(-n^2/2s^2) exp(cn/s^2) the
    centre term cancels and every power of ``n`` is one separable spatial
    convolution, so the cost is independent of the window area.
    """
    pad_r = kernel // 2
    d = np.arange(-pad_r, pad_r + 1, dtype=np.float64)
    gs = np.exp(-(d * d) / (2.0 * spatial_sigma * spatial_sigma))
    s2 = range_sigma * range_sigma
    xmax = 1.0 / s2
    kmax = 0
    term = 1.0
    while term * math.e > 1e-6 and kmax < 40:
        kmax += 1
        term *= xmax / kmax
    n = _c(data)
    shape = n.shape
    nf = n.ravel()
    pw = np.exp(-(n * n) / F32(2.0 * s2)).astype(F32)   # exp(-n^2/2s^2) * n^k
    num = np.zeros(nf.size, F32)
    den = np.zeros(nf.size, F32)
    coef = np.ones(nf.size, F32)      # (n_centre / s^2)^k / k!
    prev = np.zeros(nf.size, F32)
    for k in range(kmax + 2):
        t = sep_correlate(pw, gs, gs, edge, trim=False).ravel()
        if k < kmax + 1:
            pw = _bil_step(t, nf, pw.ravel(), coef, prev, num, den, F32(1.0 / s2), k).reshape(shape)
        else:
            _bil_last(t, prev, num)
    return _bil_finish(num, den, nf).reshape(shape)


@njit(cache=True, parallel=True)
def _bil_step(t, n, pw, coef, prev, num, den, inv_s2, k):
    out = np.empty_like(pw)
    for i in prange(t.size):
        den[i] += coef[i] * t[i]
        if k > 0:
            num[i] += prev[i] * t[i]
        prev[i] = coef[i]
        coef[i] = coef[i] * n[i] * inv_s2 / (k + 1)
        out[i] = pw[i] * n[i]
    return out


@njit(cache=True, parallel=True)
def _bil_last(t, prev, num):
    for i in prange(t.size):
        num[i] += prev[i] * t[i]


@njit(cache=True, parallel=True)
def _bil_finish(num, den, n):
    out = np.empty_like(n)
    for i in prange(n.size):
        out[i] = num[i] / den[i] if den[i] > 0 else n[i]
    return out


def _fbl_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """Effective 1D weights of libblur's fast-bilateral grid convolution, offsets -half..+half.

    The upstream loop pairs tap ``i`` with offsets ``-half+i`` and ``half-1-i``,
    so the right half is shifted one cell towards the centre.
    """
    k = gaussian_kernel_1d(kernel_size, sigma).astype(np.float32).astype(np.float64)
    half = len(k) // 2
    w = np.zeros(2 * half + 1)
    w[half] += k[half]
    for i in range(half):
        w[i] += k[i]                      # offset -half + i
        w[half + half - 1 - i] += k[i]    # offset half - 1 - i
    return w


def fast_bilateral_plane(v: np.ndarray, kernel_size: int, spatial_sigma: float, range_sigma: float) -> np.ndarray:
    """libblur ``fast_bilateral_filter`` (bilateral grid) on one 0..1 plane, ported as written."""
    from scipy.ndimage import correlate1d

    h, w = v.shape
    v = v.astype(F32)
    base_min = float(v.min())
    delta = float(v.max()) - base_min
    ss, rs = np.float32(spatial_sigma), np.float32(range_sigma)
    if ss > 1:
        s_scale = np.float32(min(max(np.float32(1.0) / ss * np.float32(1.3 if ss > 1.3 else 1.0),
                                     np.float32(1 / 5)), np.float32(1 / 1.2)))
    else:
        s_scale = np.float32(1.0)
    r_scale = np.float32(min(max(np.float32(1.0) / rs, np.float32(1 / 5)), np.float32(1 / 1.1))) if rs > 1 \
        else np.float32(1.0)
    sw = int(np.float32(w - 1) * s_scale + np.float32(1.0) + np.float32(4.0))
    sh = int(np.float32(h - 1) * s_scale + np.float32(1.0) + np.float32(4.0))
    sd = int(np.float32(delta) * r_scale + np.float32(1.0) + np.float32(4.0))
    vals, cnt = _fbl_splat(v, np.float32(base_min), s_scale, r_scale, sw, sh, sd)
    sig = float(np.sqrt(ss * ss + rs * rs))
    wk = _fbl_kernel(int(kernel_size), sig)
    for axis in (0, 2, 1):
        vals = correlate1d(vals, wk, axis=axis, mode="nearest")
        cnt = correlate1d(cnt, wk, axis=axis, mode="nearest")
    return _fbl_slice(v, vals, cnt, np.float32(base_min), s_scale, r_scale)


@njit(cache=True)
def _fbl_splat(v, base_min, s_scale, r_scale, sw, sh, sd):
    h, w = v.shape
    vals = np.zeros((sw, sh, sd))
    cnt = np.zeros((sw, sh, sd))
    half = np.float32(0.5)
    two = np.float32(2.0)
    for y in range(h):
        gy = int(np.floor(np.float32(y) * s_scale + half + two))
        for x in range(w):
            gx = int(np.floor(np.float32(x) * s_scale + half + two))
            p = v[y, x]
            gz = int(np.floor((p - base_min) * r_scale + half + two))
            vals[gx, gy, gz] += p
            cnt[gx, gy, gz] += 1.0
    return vals, cnt


@njit(cache=True, parallel=True)
def _fbl_slice(v, vals, cnt, base_min, s_scale, r_scale):
    """Upstream ``trilinear_interpolation`` collapses to the floor cell (its fractions are always 0)."""
    h, w = v.shape
    out = np.empty((h, w), np.float32)
    two = np.float32(2.0)
    for y in prange(h):
        iy = int(np.floor(np.float32(y) * s_scale + two))
        for x in range(w):
            ix = int(np.floor(np.float32(x) * s_scale + two))
            iz = int(np.floor((v[y, x] - base_min) * r_scale + two))
            d = cnt[ix, iy, iz]
            out[y, x] = vals[ix, iy, iz] / d if d != 0.0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Convolution kernels (aire ConvolveKernels, libblur motion kernel, aire bokeh)
# ---------------------------------------------------------------------------

def kernel_circle(k: int) -> np.ndarray | None:
    if k < 5:
        return None  # aire throws InvalidParameterException
    c = k // 2
    r = np.float32(k) / np.float32(3.0)
    yy, xx = np.mgrid[0:k, 0:k]
    m = np.sqrt((yy - c) ** 2.0 + (xx - c) ** 2.0) <= r
    return (m / m.sum()).astype(F32)


def kernel_cross(k: int) -> np.ndarray:
    c = k // 2
    yy, xx = np.mgrid[0:k, 0:k]
    m = (yy == xx) & (np.sqrt((yy - c) ** 2.0 + (xx - c) ** 2.0) < c + 1)
    return (m / m.sum()).astype(F32)


def kernel_ring(k: int) -> np.ndarray:
    # NB: upstream measures the distance from (k, k), i.e. outside the kernel -> a quarter arc
    yy, xx = np.mgrid[0:k, 0:k]
    dist = np.sqrt((yy - k) ** 2.0 + (xx - k) ** 2.0)
    m = (dist > k - 0.5) & (dist < k + 0.5)
    return (m / max(m.sum(), 1)).astype(F32)


def kernel_star(k: int) -> np.ndarray:
    c = k // 2
    yy, xx = np.mgrid[0:k, 0:k]
    m = (yy == c) | (xx == c)
    return (m / m.sum()).astype(F32)


def _bresenham(x0: int, y0: int, x1: int, y1: int):
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    err = dx + dy
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    x, y = x0, y0
    pts = []
    while True:
        pts.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return pts


def motion_kernel(size: int, angle_deg: float) -> np.ndarray:
    """libblur ``generate_motion_kernel`` (f32 maths, saturating casts)."""
    a = np.float32(angle_deg) * np.float32(np.pi) / np.float32(180.0)
    start = np.float32(size // 2)
    px = np.ceil(start + np.float32(size) * np.cos(a))
    py = np.ceil(start + np.float32(size) * np.sin(a))
    px = 0 if not np.isfinite(px) or px < 0 else int(px)
    py = 0 if not np.isfinite(py) or py < 0 else int(py)
    ex, ey = size - px - 1, size - py - 1
    k = np.zeros((size, size), F32)
    for x, y in _bresenham(px, py, ex, ey):
        if 0 <= x < size and 0 <= y < size:
            k[y, x] = 1.0
    s = k.sum()
    return k / s if s > 0 else k


def _flood_fill(grid: np.ndarray, sx: int, sy: int) -> None:
    from collections import deque

    rows, cols = grid.shape
    if not (0 <= sx < rows and 0 <= sy < cols):
        return
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        if x < 0 or x >= rows or y < 0 or y >= cols:
            continue
        if grid[y, x] == 0:
            grid[y, x] = 1
            q.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))


def bokeh_kernel(size: int, sides: int) -> np.ndarray | None:
    """aire ``getBokehEffect``: polygon outline (Bresenham) + 4-connected flood fill from the centre."""
    if size < 3 or sides < 3:
        return None  # aire throws
    k = np.zeros((size, size), np.int32)
    c = np.float32(size / 2.0)
    radius = c * np.float32(0.85)
    two_pi = np.float32(2.0) * np.float32(np.pi)
    for i in range(sides):
        a1 = two_pi * np.float32(i) / np.float32(sides)
        a2 = two_pi * np.float32(i + 1) / np.float32(sides)
        x1, y1 = int(c + radius * np.cos(a1)), int(c + radius * np.sin(a1))
        x2, y2 = int(c + radius * np.cos(a2)), int(c + radius * np.sin(a2))
        for x, y in _bresenham(x1, y1, x2, y2):
            if 0 <= x < size and 0 <= y < size:
                k[y, x] = 1
    _flood_fill(k, size // 2, size // 2)
    return k.astype(bool)


def dilate(data: np.ndarray, kernel: np.ndarray, edge: str = "reflect_101") -> np.ndarray:
    """Flat dilation (fast_morphology semantics = OpenCV: anchor = size / 2, no kernel reflection)."""
    kh, kw = kernel.shape
    ay, ax = kh // 2, kw // 2
    h, w = data.shape[:2]
    k8 = kernel.astype(np.uint8)
    pads = (ay, kh - 1 - ay, ax, kw - 1 - ax)
    if _cv_native(edge, h, w, *pads):
        return cv2.dilate(_c(data), k8, anchor=(ax, ay), borderType=_CV_BORDER[edge]).reshape(data.shape)
    p = pad(_c(data), *pads, edge)
    out = cv2.dilate(p, k8, anchor=(ax, ay), borderType=cv2.BORDER_REPLICATE).reshape(p.shape)
    return np.ascontiguousarray(out[ay:ay + h, ax:ax + w])


# ---------------------------------------------------------------------------
# aire ZoomBlur (EnhancedZoomBlur), jhlabs SmartBlur
# ---------------------------------------------------------------------------

def aire_zoom_blur(src: np.ndarray, kern: np.ndarray, cx, cy, strength, cos_a, sin_a) -> np.ndarray:
    """aire ``ZoomBlur::apply``; taps whose weight is numerically irrelevant (< 1e-9 of the peak) are skipped."""
    half = kern.size // 2
    keep = np.nonzero(kern > kern.max() * 1e-9)[0]
    return _aire_zoom(_c(src), kern[keep].astype(np.float64), (keep - half).astype(np.float32),
                      cx, cy, strength, cos_a, sin_a)


@njit(cache=True, parallel=True)
def _aire_zoom(src, kern, offs, cx, cy, strength, cos_a, sin_a):
    h, w, nc = src.shape
    out = np.empty_like(src)
    for y in prange(h):
        acc = np.zeros(nc)
        for x in range(w):
            ox = np.float32(0.01) * (cx - np.float32(x)) * strength * cos_a
            oy = np.float32(0.01) * (cy - np.float32(y)) * strength * sin_a
            acc[:] = 0.0
            for j in range(kern.size):
                m = offs[j]
                sx = int(np.float32(x) + m * ox)
                sy = int(np.float32(y) + m * oy)
                sx = min(max(sx, 0), w - 1)
                sy = min(max(sy, 0), h - 1)
                wt = kern[j]
                for c in range(nc):
                    acc[c] += src[sy, sx, c] * wt
            for c in range(nc):
                out[y, x, c] = acc[c]
    return out


@njit(cache=True, parallel=True)
def _smart_pass_t(src, kern, thr):
    """One jhlabs threshold-blur pass along x; returns the transposed result (W, H, C)."""
    h, w, nc = src.shape
    k = kern.size
    half = k // 2
    out = np.empty((w, h, nc), np.float32)
    for y in prange(h):
        for x in range(w):
            for c in range(nc):
                cen = src[y, x, c]
                tot = 0.0
                ws = 0.0
                for j in range(k):
                    sx = x + j - half
                    v = cen if (sx < 0 or sx >= w) else src[y, sx, c]
                    if abs(cen - v) * 255.0 <= thr:
                        tot += kern[j] * v
                        ws += kern[j]
                out[x, y, c] = tot / ws if ws != 0.0 else cen
    return out


def smart_blur(data: np.ndarray, radius: int, threshold: int) -> np.ndarray:
    kern = jh_gaussian_kernel(float(radius))
    thr = float(threshold) + 1e-3  # 8-bit integer comparison on 0..1 values
    d = data if data.ndim == 3 else data[..., None]
    first = _smart_pass_t(_c(d), kern, thr)
    return _smart_pass_t(first, kern, thr).reshape(data.shape)


# ---------------------------------------------------------------------------
# GPUImage zoom blur, jhlabs diffuse
# ---------------------------------------------------------------------------

_ZOOM_TAPS = ((0, 0.18), (1, 0.15), (2, 0.12), (3, 0.09), (4, 0.05), (-1, 0.15), (-2, 0.12), (-3, 0.09),
              (-4, 0.05))


def gpu_zoom_blur(img: np.ndarray, cx: float, cy: float, size: float) -> np.ndarray:
    """GPUImageZoomBlurFilter: 9 bilinear (GL_LINEAR, CLAMP_TO_EDGE) taps at uv +- k * offset."""
    ks = np.array([k for k, _ in _ZOOM_TAPS], np.float64)
    ws = np.array([w for _, w in _ZOOM_TAPS], np.float64)
    return _gpu_zoom(_c(img), float(cx), float(cy), float(size), ks, ws)


@njit(cache=True, parallel=True)
def _gpu_zoom(src, cx, cy, size, ks, ws):
    h, w, nc = src.shape
    out = np.zeros_like(src)
    for y in prange(h):
        ty = (y + 0.5) / h
        oy = 0.01 * (cy - ty) * size
        for x in range(w):
            tx = (x + 0.5) / w
            ox = 0.01 * (cx - tx) * size
            for t in range(ks.size):
                u = (tx + ks[t] * ox) * w - 0.5
                v = (ty + ks[t] * oy) * h - 0.5
                x0 = int(np.floor(u))
                y0 = int(np.floor(v))
                fx = u - x0
                fy = v - y0
                xa = min(max(x0, 0), w - 1)
                xb = min(max(x0 + 1, 0), w - 1)
                ya = min(max(y0, 0), h - 1)
                yb = min(max(y0 + 1, 0), h - 1)
                wt = ws[t]
                for c in range(nc):
                    top = src[ya, xa, c] * (1.0 - fx) + src[ya, xb, c] * fx
                    bot = src[yb, xa, c] * (1.0 - fx) + src[yb, xb, c] * fx
                    out[y, x, c] += wt * (top * (1.0 - fy) + bot * fy)
    return out


def diffuse(data: np.ndarray, scale: float) -> np.ndarray:
    """jhlabs DiffuseFilter (native port): hash-based offset, bilinear sampling, clamped edges."""
    h, w = data.shape[:2]
    x = np.arange(w, dtype=np.uint64)
    y = np.arange(h, dtype=np.uint64)
    with np.errstate(over="ignore"):
        hx = (x * np.uint64(0x9E3779B1)) & np.uint64(0xFFFFFFFF)
        hy = (y * np.uint64(0x85EBCA77)) & np.uint64(0xFFFFFFFF)
    hsh = hx[None, :] ^ hy[:, None]
    ang = (hsh & np.uint64(255)).astype(F32) / F32(256.0) * F32(2 * np.pi)
    dist = ((hsh >> np.uint64(8)) & np.uint64(0xFFFF)).astype(F32) / F32(65535.0) * F32(scale)
    mx = (np.arange(w, dtype=F32)[None, :] + dist * np.sin(ang)).astype(F32)
    my = (np.arange(h, dtype=F32)[:, None] + dist * np.cos(ang)).astype(F32)
    out = cv2.remap(_c(data), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return _cv_out(out, data)


# ---------------------------------------------------------------------------
# Tilt-shift blends (aire TiltShift.cpp)
# ---------------------------------------------------------------------------

def aire_gauss_blur(data: np.ndarray, size: int, sigma: float) -> np.ndarray:
    """aire ``gaussBlurU8``: separable ``convolve1D`` with an even/odd ``size`` kernel, clamped edges."""
    if size <= 1:
        return data
    k = gaussian_kernel_1d(size, sigma)
    return sep_correlate(data, k, k, "clamp", ax=size // 2, ay=size // 2)


@njit(cache=True, parallel=True)
def tilt_blend(src, blurred, fraction):
    """mix(src, blurred, blurredA * fraction); alpha = source alpha (premultiplied buffers)."""
    h, w, nc = src.shape
    out = np.empty_like(src)
    for y in prange(h):
        for x in range(w):
            t = fraction[y, x]
            if nc == 4:
                t = t * blurred[y, x, 3]
                out[y, x, 3] = src[y, x, 3]
            for c in range(3):
                out[y, x, c] = src[y, x, c] + (blurred[y, x, c] - src[y, x, c]) * t
    return out


@njit(cache=True, inline="always")
def _cube(d, min_d):
    if min_d == 0.0:
        return np.float32(1.0)  # upstream divides by zero (inf/NaN -> fully blurred)
    fr = np.float32(d / min_d)
    return np.float32(min(max(fr * fr * fr, np.float32(0.0)), np.float32(1.0)))


@njit(cache=True, parallel=True)
def _radial_frac(h, w, cx, cy, min_d):
    out = np.empty((h, w), np.float32)
    for y in prange(h):
        dy = np.float32(y - cy)
        for x in range(w):
            dx = np.float32(x - cx)
            out[y, x] = _cube(np.sqrt(dx * dx + dy * dy), min_d)
    return out


@njit(cache=True, parallel=True)
def _linear_frac(h, w, fax, fay, ca, sa, horizontal, min_d):
    out = np.empty((h, w), np.float32)
    for y in prange(h):
        ys = np.float32(y - fay)
        for x in range(w):
            xs = np.float32(x - fax)
            if horizontal:
                d = abs(int(xs * ca - ys * sa + np.float32(fax)) - fax)
            else:
                d = abs(int(xs * sa + ys * ca + np.float32(fay)) - fay)
            out[y, x] = _cube(np.float32(d), min_d)
    return out


def radial_fraction(h: int, w: int, ax: float, ay: float, radius: float) -> np.ndarray:
    """aire ``tiltShift`` blend factor: clamp((dist / (diagonal * radius))^3)."""
    min_d = np.float32(np.sqrt(np.float32(h * h + w * w))) * np.float32(radius)
    cx, cy = int(np.float32(w) * np.float32(ax)), int(np.float32(h) * np.float32(ay))
    return _radial_frac(h, w, cx, cy, np.float32(min_d))


def linear_fraction(h: int, w: int, ax: float, ay: float, radius: float, angle: float) -> np.ndarray:
    """aire ``horizontalTiltShift`` distance field (angle in radians, f32 like upstream)."""
    angle = np.float32(angle)
    ca, sa = np.cos(angle), np.sin(angle)
    new_w = int(abs(np.float32(w) * ca) + abs(np.float32(h) * sa))
    new_h = int(abs(np.float32(w) * sa) + abs(np.float32(h) * ca))
    rot = np.float32(math.fmod(float(angle) + math.pi, 2 * math.pi) - math.pi)
    horizontal = bool(np.float32(-math.pi / 4.0) <= rot <= np.float32(math.pi / 4.0))
    min_d = np.float32(new_w if horizontal else new_h) * np.float32(radius)
    fax, fay = int(np.float32(w) * np.float32(ax)), int(np.float32(h) * np.float32(ay))
    return _linear_frac(h, w, fax, fay, np.float32(ca), np.float32(sa), horizontal, np.float32(min_d))


# ---------------------------------------------------------------------------
# Trickle
# ---------------------------------------------------------------------------

def cpp_round(x: float) -> int:
    """std::round (half away from zero)."""
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def trickle_scaled(data: np.ndarray, scale: float, blur: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """createScaledBitmap(scale, bilinear) -> blur -> scale back to the input size."""
    h, w = data.shape[:2]
    sw, sh = max(1, cpp_round(w * scale)), max(1, cpp_round(h * scale))
    nc = data.shape[2]
    if (sw, sh) == (w, h):
        return blur(_c(data))
    small = cv2.resize(_c(data), (sw, sh), interpolation=cv2.INTER_LINEAR).reshape(sh, sw, nc)
    small = blur(np.ascontiguousarray(small))
    return cv2.resize(_c(small), (w, h), interpolation=cv2.INTER_LINEAR).reshape(h, w, nc).astype(F32)


@njit(cache=True, inline="always")
def _rand31(seed, idx):
    """Counter-based stand-in for libc ``rand()`` (splitmix64, 31-bit result)."""
    z = seed + np.uint64(idx) * np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return np.int64(z >> np.uint64(33))


@njit(cache=True, inline="always")
def _trickle_lin(c):
    v = np.float32(np.float32(min(max(round(c * 255.0), 0.0), 255.0)) / np.float32(255.0))
    if v <= 0.045:
        return np.float32(v / 12.92)
    return np.float32(((v + 0.055) / 1.055) ** 2.4)


@njit(cache=True, parallel=True)
def _shuffle(data, threshold, strength, seed):
    h, w, nc = data.shape
    out = np.zeros((h, w, 4), np.float32)
    s = np.float32(strength)
    thr = np.float32(threshold)
    for y in prange(h):
        for x in range(w):
            r = _trickle_lin(data[y, x, 0])
            g = _trickle_lin(data[y, x, 1])
            b = _trickle_lin(data[y, x, 2])
            luma = np.float32(np.float64(np.float32(0.2125) * r + np.float32(0.7154) * g) + 0.0721 * np.float64(b))
            ok = luma <= thr if thr >= 0 else luma > abs(thr)
            if not ok:
                continue
            base = (y * w + x) * 6
            sy = int(np.float32(y) - np.float32(_rand31(seed, base) % (y + 1)) * s)
            ey = int(np.float32(y) + np.float32(_rand31(seed, base + 1) % (y + 1)) * s)
            sx = int(np.float32(x) - np.float32(_rand31(seed, base + 2) % (x + 1)) * s)
            ex = int(np.float32(x) + np.float32(_rand31(seed, base + 3) % (x + 1)) * s)
            ry = sy + _rand31(seed, base + 4) % (ey - sy + 1)
            rx = sx + _rand31(seed, base + 5) % (ex - sx + 1)
            ny = min(max(ry, 0), h - 1)
            nx = min(max(rx, 0), w - 1)
            for c in range(3):
                out[y, x, c] = data[ny, nx, c]
            out[y, x, 3] = data[y, x, 3] if nc == 4 else 1.0
    return out


def shuffle_blur(data: np.ndarray, threshold: float, strength: float, seed: int) -> np.ndarray:
    """Trickle ``shuffleBlurImpl`` on a premultiplied buffer -> new RGBA (unmatched pixels stay 0)."""
    return _shuffle(_c(data), float(threshold), float(strength), np.uint64(seed))


# ---------------------------------------------------------------------------
# aire Poisson blur, libblur median
# ---------------------------------------------------------------------------

def poisson_kernel(size: int, rng: np.random.Generator) -> np.ndarray:
    k = rng.poisson(lam=size, size=size).astype(np.float64)
    for _ in range(50):
        if k.sum() != 0:
            break
        k = rng.poisson(lam=size, size=size).astype(np.float64)
    return k / k.sum() if k.sum() != 0 else np.full(size, 1.0 / size)


@njit(cache=True, inline="always")
def _median_hist_col(src, hist, y, x, rad, sign):
    h, w, nc = src.shape
    if x < 0 or x >= w:
        return 0
    cnt = 0
    for i in range(max(y - rad, 0), min(y + rad, h - 1) + 1):
        for c in range(nc):
            hist[c, src[i, x, c]] += sign
        cnt += 1
    return cnt * sign


@njit(cache=True, parallel=True)
def _median_libblur(src, rad):
    h, w, nc = src.shape
    out = np.empty_like(src)
    for y in prange(h):
        hist = np.zeros((nc, 256), np.int32)
        n = 0
        for x in range(w):
            if x == 0:
                for j in range(min(rad, w)):  # NB: column `rad` itself is never added (upstream quirk)
                    n += _median_hist_col(src, hist, y, j, rad, 1)
            else:
                n += _median_hist_col(src, hist, y, x - rad, rad, -1)
                n += _median_hist_col(src, hist, y, x + rad, rad, 1)
            if n > 0:
                for c in range(nc):
                    t = n // 2
                    i = 0
                    while i < 256:
                        t -= hist[c, i]
                        if t > 0:
                            i += 1
                        else:
                            break
                    out[y, x, c] = min(i, 255)
            else:
                for c in range(nc):
                    out[y, x, c] = src[y, x, c]
    return out


def median_libblur(data: np.ndarray, radius: int) -> np.ndarray:
    """libblur ``median_blur`` (per-row sliding 8-bit histograms, shrinking window at the borders)."""
    q = np.ascontiguousarray(np.clip(np.round(data * 255.0), 0, 255).astype(np.uint8))
    return _median_libblur(q, int(radius)).astype(F32) / F32(255.0)
