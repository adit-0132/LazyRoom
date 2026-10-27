"""Private helpers for the ImageToolbox "Enhancement" group (non-G'MIC algorithms).

Ports of:
  * aire (MIT, awxkee): anisotropicDiffusion, dehaze (dark channel prior), unsharp
  * Trickle (Apache-2.0, T8RIN): autoWhiteBalance
  * ImageToolboxLibs jhlabs port (Apache-2.0): HighPassFilter (native Rust implementation)
  * GPUImage (BSD-3): Kuwahara and Sharpen fragment shaders
  * ImageToolbox opencv-tools (Apache-2.0): RedEyeRemover (YuNet face detector, MIT)
"""
from __future__ import annotations

import math
import threading
from pathlib import Path

import cv2
import numpy as np
from numba import njit, prange

from ...core.errors import ErrorCode, LazyRoomError
from ...core.paths import package_data_dir

F32 = np.float32


@njit(cache=True, parallel=True)
def _to_u8(img):
    h, w = img.shape[0], img.shape[1]
    out = np.empty((h, w, 3), np.uint8)
    for y in prange(h):
        for x in range(w):
            for c in range(3):
                v = np.rint(img[y, x, c] * np.float32(255.0))
                out[y, x, c] = np.uint8(min(max(v, 0.0), 255.0))
    return out


def to_u8(img: np.ndarray) -> np.ndarray:
    """0..1 float RGB -> uint8 like an Android ARGB_8888 bitmap."""
    return _to_u8(np.ascontiguousarray(img[..., :3], dtype=F32))


@njit(cache=True, parallel=True)
def _u8_to_f32(u8):
    h, w = u8.shape[0], u8.shape[1]
    out = np.empty((h, w, 3), np.float32)
    for y in prange(h):
        for x in range(w):
            for c in range(3):
                out[y, x, c] = np.float32(u8[y, x, c]) / np.float32(255.0)
    return out


# ---------------------------------------------------------------------------
# aire: anisotropic diffusion (blur/AnisotropicDiffusion.cpp)
# ---------------------------------------------------------------------------

def _perona_malik_lut(conduction: float) -> np.ndarray:
    """PerconaMalik(d, conduction) * d for every possible 8-bit difference d in -255..255 (float32 like C++)."""
    d = np.arange(-255, 256, dtype=F32)
    k = d / np.float32(conduction)
    g = (1.0 / (1.0 + (k * k).astype(np.float64))).astype(F32)  # double division stored in a float
    return (g * d).astype(F32)


@njit(cache=True)
def _aniso_row(buf, lut, diffusion, y):
    """Update row ``y`` in place (aire's per-row worker)."""
    h, w = buf.shape[0], buf.shape[1]
    py = max(y - 1, 0)
    ny = min(y + 1, h - 1)
    for x in range(w):  # sequential: rows 0 / h-1 read their own (already updated) neighbours
        px = max(x - 1, 0)
        nx = min(x + 1, w - 1)
        for c in range(3):
            loc = np.int32(buf[y, x, c])
            s = lut[np.int32(buf[py, px, c]) - loc + 255] + lut[np.int32(buf[py, nx, c]) - loc + 255] \
                + lut[np.int32(buf[ny, px, c]) - loc + 255] + lut[np.int32(buf[ny, nx, c]) - loc + 255]
            lf = np.float32(loc)
            v = math.ceil(lf + lf * (diffusion * s))
            buf[y, x, c] = np.uint8(min(max(v, 0.0), 255.0))


@njit(cache=True, parallel=True)
def _aniso_diffusion(buf, steps, lut, diffusion):
    # aire runs each iteration with concurrency::parallel_for(4, height): four contiguous row
    # segments updated in place by four threads at the same time.  The first row of a segment
    # sees the (not yet updated) last row of the previous segment, its last row sees the already
    # updated first row of the next segment.  Phase 1 updates the first rows of segments 1..3,
    # phase 2 the rest of every segment in parallel - identical to processing the segments
    # last-to-first sequentially, i.e. that (by far most likely) interleaving, deterministically.
    h = buf.shape[0]
    seg = h // 4
    for _ in range(steps):
        if seg < 2:
            for k in range(3, -1, -1):
                for y in range(k * seg, h if k == 3 else (k + 1) * seg):
                    _aniso_row(buf, lut, diffusion, y)
            continue
        for k in prange(1, 4):
            _aniso_row(buf, lut, diffusion, k * seg)
        for k in prange(4):
            y0 = 0 if k == 0 else k * seg + 1
            y1 = h if k == 3 else (k + 1) * seg
            for y in range(y0, y1):
                _aniso_row(buf, lut, diffusion, y)


def anisotropic_diffusion(img: np.ndarray, steps: int, conduction: float, diffusion: float) -> np.ndarray:
    buf = to_u8(img)
    _aniso_diffusion(buf, int(steps), _perona_malik_lut(conduction), np.float32(diffusion))
    return _u8_to_f32(buf)


# ---------------------------------------------------------------------------
# aire: dark channel prior dehaze (pipelines/DehazeDarkChannel.cpp)
# ---------------------------------------------------------------------------

_LANES = 16  # highway ScalableTag<uint8_t> on arm64 NEON (the Android build)


@njit(cache=True)
def _dark_rows(u8, rmin, radius, lanes):
    """Horizontal part of aire's getDarkChannel, per row.

    The SIMD branch takes ``MinOfLanes(r)`` three times (upstream bug), i.e. only the red
    channel of each 16-pixel block; the scalar tail takes min(r, g, b)."""
    h, w = u8.shape[0], u8.shape[1]
    out = np.empty((h, w), np.uint8)
    for y in range(h):
        for j in range(w):
            m = 255
            x = -radius
            while x + lanes <= radius and x + j + lanes < w:
                pos = min(max(x + j, 0), w - 1)
                v = rmin[y, pos]
                if v < m:
                    m = v
                x += lanes
            while x <= radius:
                pos = x + j
                if pos >= w:
                    break
                if pos < 0:
                    pos = 0
                v = min(u8[y, pos, 0], u8[y, pos, 1], u8[y, pos, 2])
                if v < m:
                    m = v
                x += 1
            out[y, j] = m
    return out


def dehaze_dark_channel(img: np.ndarray, radius: int, omega: float) -> np.ndarray:
    u8 = to_u8(img)
    r = max(0, int(radius))
    # red-channel minimum over [p, p + 15] (border: replicate; never reached for valid loads)
    rmin = cv2.erode(np.ascontiguousarray(u8[..., 0]), np.ones((1, _LANES), np.uint8), anchor=(0, 0),
                     borderType=cv2.BORDER_REPLICATE)
    rows = _dark_rows(u8, np.ascontiguousarray(rmin), r, _LANES)
    dark = cv2.erode(rows, np.ones((2 * r + 1, 1), np.uint8), borderType=cv2.BORDER_REPLICATE) if r > 0 else rows
    # atmospheric light: mean of the brightest 1 % of the dark channel
    flat = dark.reshape(-1)
    k = max(1, int(flat.size * 0.01))
    top = np.partition(flat, flat.size - k)[flat.size - k:]
    a = np.float32(top.astype(np.float64).sum() / k)
    a = max(a, np.float32(1e-6))
    return _transmission(u8, dark, a, np.float32(omega))


@njit(cache=True, parallel=True)
def _transmission(u8, dark, a, omega):
    """aire getTransmission: t = max(1 - w*dark/A, 1 - w*min(rgb)/A); (c - A) / t + A, clamped, truncated."""
    h, w = u8.shape[0], u8.shape[1]
    out = np.empty((h, w, 3), np.float32)
    one = np.float32(1.0)
    for y in prange(h):
        for x in range(w):
            r = np.float32(u8[y, x, 0])
            g = np.float32(u8[y, x, 1])
            b = np.float32(u8[y, x, 2])
            t0 = one - omega * (min(r, g, b) / a)
            t = max(one - omega * (np.float32(dark[y, x]) / a), t0)
            for c in range(3):
                cv = np.float32(u8[y, x, c])
                v = (cv - a) / t + a
                if v != v:      # NaN (0/0) -> cast<int> yields 0 on arm64
                    v = 0.0
                v = min(max(v, 0.0), 255.0)
                out[y, x, c] = np.float32(int(v)) / np.float32(255.0)
    return out


# ---------------------------------------------------------------------------
# aire: unsharp (jni/Base.cpp unsharpImpl + base/Sharpness.cpp applyUnsharp)
# ---------------------------------------------------------------------------

def aire_gauss_kernel(width: int, sigma: float) -> np.ndarray:
    """aire MathUtils compute1DGaussianKernel(width, sigma)."""
    mean = width // 2
    x = np.arange(width, dtype=np.float64)
    k = np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma)
    return (k / k.sum()).astype(F32)


def aire_unsharp(img: np.ndarray, intensity: float, scale: float = 1.0) -> np.ndarray:
    """color - (color - gaussBlur(size 5, sigma 4)) * intensity  (aire's 'unsharp' blends towards the blur)."""
    size, sigma = 5, 4.0
    if abs(scale - 1.0) > 1e-6:  # keep previews representative of the full-resolution result
        size = max(1, int(round(5 * scale)) | 1)
        sigma = max(0.05, 4.0 * scale)
    k = aire_gauss_kernel(size, sigma)
    anchor = size // 2
    blur = cv2.sepFilter2D(img, -1, k, k, anchor=(anchor, anchor), borderType=cv2.BORDER_REPLICATE)
    return (img - (img - blur) * np.float32(intensity)).astype(F32)


# ---------------------------------------------------------------------------
# Trickle: AutoWhiteBalanceFilter.cpp
# ---------------------------------------------------------------------------

@njit(cache=True)
def _awb_hist(u8, alpha, use_alpha):
    h, w = u8.shape[0], u8.shape[1]
    hist = np.zeros((3, 256), np.int64)
    n = 0
    for y in range(h):
        for x in range(w):
            if use_alpha and alpha[y, x] < np.float32(0.5 / 255.0):
                continue
            n += 1
            for c in range(3):
                hist[c, u8[y, x, c]] += 1
    return hist, n


@njit(cache=True, parallel=True)
def _awb_apply(img, alpha, use_alpha, low, high, active, strength):
    h, w = img.shape[0], img.shape[1]
    out = np.empty((h, w, 3), np.float32)
    for y in prange(h):
        for x in range(w):
            skip = use_alpha and alpha[y, x] < np.float32(0.5 / 255.0)
            for c in range(3):
                v = img[y, x, c]
                if skip or not active[c]:
                    out[y, x, c] = v
                else:
                    bal = (v * np.float32(255.0) - low[c]) / np.float32(high[c] - low[c])
                    bal = min(max(bal, np.float32(0.0)), np.float32(1.0))
                    out[y, x, c] = v + (bal - v) * strength
    return out


def auto_white_balance(img: np.ndarray, strength: float, clip_percent: float) -> np.ndarray:
    """Trickle AutoWhiteBalanceFilter.  The 8-bit LUT ((i - low) * 255 / (high - low), clamped) is applied as
    the continuous stretch it samples, so float precision is kept."""
    rgb = np.ascontiguousarray(img[..., :3], dtype=F32)
    use_alpha = img.shape[2] == 4
    alpha = np.ascontiguousarray(img[..., 3], dtype=F32) if use_alpha else np.ones((1, 1), F32)
    hist, n = _awb_hist(to_u8(rgb), alpha, use_alpha)
    if n == 0:
        return img
    clip = min(10.0, max(0.0, clip_percent))
    cut = int(np.float32(n) * np.float32(clip) / np.float32(100.0))
    low = np.zeros(3, np.int64)
    high = np.zeros(3, np.int64)
    active = np.zeros(3, np.bool_)
    for c in range(3):
        low[c] = np.searchsorted(np.cumsum(hist[c]), cut, side="right")  # first index whose count > cut
        high[c] = 255 - np.searchsorted(np.cumsum(hist[c][::-1]), cut, side="right")
        active[c] = high[c] > low[c]  # otherwise the identity LUT
    out = _awb_apply(rgb, alpha, use_alpha, low.astype(F32), high.astype(F32), active,
                     np.float32(min(1.0, max(0.0, strength))))
    return np.dstack([out, img[..., 3]]) if use_alpha else out


# ---------------------------------------------------------------------------
# jhlabs (ImageToolboxLibs native port): HighPassFilter
# ---------------------------------------------------------------------------

def jh_gaussian_kernel(radius: float) -> np.ndarray:
    """gaussian_kernel_exact(radius): sigma = radius / 3, taps beyond the radius are zero."""
    half = int(math.ceil(radius))
    sigma = radius / 3.0
    rows = np.arange(-half, half + 1, dtype=np.float64)
    d = rows * rows
    k = np.where(d > radius * radius, 0.0,
                 np.exp(-d / (2 * sigma * sigma)) / math.sqrt(2 * math.pi * sigma))
    return (k / k.sum()).astype(F32)


def jh_high_pass(img: np.ndarray, radius: float) -> np.ndarray:
    """(color + 255 - gaussian(color)) / 2, gaussian computed on premultiplied colours."""
    if radius <= 0:
        return img
    k = jh_gaussian_kernel(radius)
    half = len(k) // 2
    rgb = img[..., :3]

    def blur(x):
        return cv2.sepFilter2D(x, -1, k, k, anchor=(half, half), borderType=cv2.BORDER_REPLICATE)

    if img.shape[2] == 4:
        a = img[..., 3:4]
        num = blur(np.ascontiguousarray(rgb * a))
        den = blur(np.ascontiguousarray(img[..., 3]))[..., None]
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.where(den > 1e-6, num / den, num)
    else:
        b = blur(np.ascontiguousarray(rgb))
    out = (rgb + 1.0 - b) * 0.5
    if img.shape[2] == 4:
        return np.dstack([out, img[..., 3]]).astype(F32)
    return out.astype(F32)


# ---------------------------------------------------------------------------
# GPUImage: Kuwahara and Sharpen shaders
# ---------------------------------------------------------------------------

def _taps(d: np.ndarray, w0: np.ndarray, w1: np.ndarray | None, lo: int, hi: int) -> np.ndarray:
    """Dense 1D kernel over integer offsets lo..hi from sparse taps (w0 at d, w1 at d + 1)."""
    k = np.zeros(hi - lo + 1, np.float64)
    np.add.at(k, d - lo, w0)
    if w1 is not None:
        np.add.at(k, d + 1 - lo, w1)
    return k.astype(F32)


@njit(cache=True, parallel=True)
def _kuw_products(pp, sq, vn, hn2, dg):
    """Corner-pair products on a replicate-padded slab (last row/column are never sampled)."""
    hh, ww = pp.shape[0], pp.shape[1]
    for y in prange(hh):
        y1 = min(y + 1, hh - 1)
        for x in range(ww):
            x1 = min(x + 1, ww - 1)
            for c in range(3):
                v = pp[y, x, c]
                sq[y, x, c] = v * v                                     # P^2
                vn[y, x, c] = v * pp[y1, x, c]                          # vertical neighbours
                hn2[y, x, c] = np.float32(2.0) * v * pp[y, x1, c]       # horizontal neighbours (x2)
                dg[y, x, c] = v * pp[y1, x1, c] + pp[y, x1, c] * pp[y1, x, c]  # both diagonals


@njit(cache=True, parallel=True)
def _kuw_vertical_select(hl, ha, hb, dy, fy, ranges, quads, my, mx, rows, w, inv_n, out, y_out):
    """Vertical sparse filtering of the horizontal sums + the shader's quadrant selection.

    ``hl/ha/hb[k]`` (k = 0: i<=0, 1: i>=0) are (rows, (w + 2 mx) * 3) planes of the linear sum, the
    squared-weight sum and the cross-weight sum of the bilinear sample expansion."""
    one = np.float32(1.0)
    two = np.float32(2.0)
    n3 = w * 3
    o3 = mx * 3
    for y in prange(rows):
        m = np.zeros((4, n3), np.float32)
        s = np.zeros((4, n3), np.float32)
        for q in range(4):
            hk = quads[q, 0]
            j0, j1 = ranges[quads[q, 1], 0], ranges[quads[q, 1], 1]
            for jj in range(j0, j1):
                ya = y + my + dy[jj]
                f = fy[jj]
                wl0 = one - f
                ws0 = wl0 * wl0
                ws1 = f * f
                wc = two * f * wl0
                l0 = hl[hk, ya]
                l1 = hl[hk, ya + 1]
                a0 = ha[hk, ya]
                a1 = ha[hk, ya + 1]
                b0 = hb[hk, ya]
                for k in range(n3):
                    m[q, k] += wl0 * l0[o3 + k] + f * l1[o3 + k]
                    s[q, k] += ws0 * a0[o3 + k] + ws1 * a1[o3 + k] + wc * b0[o3 + k]
        for x in range(w):
            best = 0
            min_sigma = np.float32(1e2)
            for q in range(4):
                sig = np.float32(0.0)
                for c in range(3):
                    mm = m[q, 3 * x + c] * inv_n
                    sig += abs(s[q, 3 * x + c] * inv_n - mm * mm)
                if sig < min_sigma:
                    min_sigma = sig
                    best = q
            for c in range(3):
                out[y_out + y, x, c] = m[best, 3 * x + c] * inv_n


def gpu_kuwahara(img: np.ndarray, radius: int, strip: int = 256) -> np.ndarray:
    """GPUImageKuwaharaFilter.

    The shader uses a hard-coded texel size ``src_size = (1/768, 1/1024)``, so sample ``(i, j)``
    lies ``i * width / 768`` pixels away horizontally and ``j * height / 1024`` vertically, fetched
    with bilinear filtering (clamp-to-edge); mean and variance use the interpolated samples.

    Exact evaluation without per-sample loops: a bilinear sample is ``c = sum_k w_k P_k`` over 4
    corners, so ``c^2`` expands into corner-pair products (P^2, horizontal, vertical and diagonal
    neighbour products) whose weights factor into horizontal x vertical terms.  Every quadrant sum
    of ``c`` and ``c^2`` is therefore a sum of separable sparse 1D filters of a few product images.
    """
    r = int(radius)
    if r <= 0:
        return img
    rgb = np.ascontiguousarray(img[..., :3], dtype=F32)
    h, w = rgb.shape[:2]
    offs = np.arange(-r, r + 1, dtype=np.float64)
    ax, ay = offs * (w / 768.0), offs * (h / 1024.0)
    dx, dy = np.floor(ax).astype(np.int64), np.floor(ay).astype(np.int64)
    fx, fy = ax - dx, ay - dy
    mx, my = int(np.abs(dx).max()) + 2, int(np.abs(dy).max()) + 2
    lox, hix = int(dx.min()), int(dx.max()) + 1

    def hk(sel):  # horizontal kernels: linear, squared weights, cross weight f(1-f)
        d, f = dx[sel], fx[sel]
        return [_taps(d, 1 - f, f, lox, hix).reshape(1, -1), _taps(d, (1 - f) ** 2, f * f, lox, hix).reshape(1, -1),
                _taps(d, f * (1 - f), None, lox, hix).reshape(1, -1)]

    hks = [hk(offs <= 0), hk(offs >= 0)]                       # i <= 0, i >= 0
    ranges = np.array([[0, r + 1], [r, 2 * r + 1]], np.int64)   # j <= 0, j >= 0 (indices into dy/fy)
    # shader quadrant order: (j<=0,i<=0), (j<=0,i>=0), (j>=0,i>=0), (j>=0,i<=0) as (i-range, j-range)
    quads = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.int64)
    inv_n = np.float32(1.0 / ((r + 1) * (r + 1)))
    anchor = (-lox, 0)
    brd = cv2.BORDER_REPLICATE

    padded = cv2.copyMakeBorder(rgb, my, my, mx, mx, cv2.BORDER_REPLICATE)
    out = np.empty_like(rgb)
    slab_h = min(h, strip) + 2 * my
    wp = w + 2 * mx
    prods = np.empty((4, slab_h, wp, 3), F32)       # P^2, vertical, 2*horizontal, diagonal products
    tmp = np.empty((2, slab_h, wp, 3), F32)
    hl = np.empty((2, slab_h, wp, 3), F32)
    ha = np.empty((2, slab_h, wp, 3), F32)
    hb = np.empty((2, slab_h, wp, 3), F32)
    for y0 in range(0, h, strip):
        y1 = min(h, y0 + strip)
        n = y1 - y0 + 2 * my
        pp = padded[y0:y0 + n]
        sq, vn, hn2, dg = prods[0, :n], prods[1, :n], prods[2, :n], prods[3, :n]
        _kuw_products(pp, sq, vn, hn2, dg)
        for k, (kl, ksq, kx) in enumerate(hks):
            t0, t1 = tmp[0, :n], tmp[1, :n]
            cv2.filter2D(pp, -1, kl, dst=hl[k, :n], anchor=anchor, borderType=brd)
            cv2.filter2D(sq, -1, ksq, dst=t0, anchor=anchor, borderType=brd)
            cv2.filter2D(hn2, -1, kx, dst=t1, anchor=anchor, borderType=brd)
            cv2.add(t0, t1, dst=ha[k, :n])
            cv2.filter2D(vn, -1, ksq, dst=t0, anchor=anchor, borderType=brd)
            cv2.filter2D(dg, -1, kx, dst=t1, anchor=anchor, borderType=brd)
            cv2.add(t0, t1, dst=hb[k, :n])
        _kuw_vertical_select(hl[:, :n].reshape(2, n, wp * 3), ha[:, :n].reshape(2, n, wp * 3),
                             hb[:, :n].reshape(2, n, wp * 3), dy, fy.astype(F32), ranges, quads, my, mx,
                             y1 - y0, w, inv_n, out, y0)
    return out


def gpu_sharpen(img: np.ndarray, sharpness: float) -> np.ndarray:
    """GPUImageSharpenFilter: c * (1 + 4s) - s * (left + right + top + bottom), clamp-to-edge."""
    s = float(sharpness)
    k = np.array([[0, -s, 0], [-s, 1 + 4 * s, -s], [0, -s, 0]], F32)
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REPLICATE).astype(F32)


# ---------------------------------------------------------------------------
# opencv-tools RedEyeRemover (FaceDetectorYN / YuNet)
# ---------------------------------------------------------------------------

YUNET_MODEL = "face_detection_yunet_2026may.onnx"
_yunet_lock = threading.Lock()
_yunet = None


def _yunet_path() -> Path:
    return package_data_dir() / "models" / YUNET_MODEL


def detect_faces(bgr: np.ndarray) -> np.ndarray:
    """FaceDetectorYN.create(model, "", size, 0.7, 0.3, 5000).detect(bgr) -> (N, 15) rows."""
    global _yunet
    h, w = bgr.shape[:2]
    with _yunet_lock:
        if _yunet is None:
            path = _yunet_path()
            if not path.exists():
                raise LazyRoomError(ErrorCode.FILE_NOT_FOUND, f"Face detection model missing at {path}")
            try:
                _yunet = cv2.FaceDetectorYN.create(str(path), "", (w, h), 0.7, 0.3, 5000)
            except cv2.error as exc:  # pragma: no cover - depends on the OpenCV build
                raise LazyRoomError(ErrorCode.OPERATION_FAILED, f"Cannot load the YuNet face detector: {exc}") from exc
        _yunet.setInputSize((w, h))
        _, faces = _yunet.detect(bgr)
    return np.zeros((0, 15), F32) if faces is None else faces


def _round_half_up(v: float) -> int:  # Kotlin Double.roundToInt()
    return int(math.floor(v + 0.5))


def _eye_rect(w: int, h: int, cx: float, cy: float, size: float):
    left = min(max(_round_half_up(cx - size / 2.0), 0), w - 1)
    top = min(max(_round_half_up(cy - size / 2.0), 0), h - 1)
    right = min(max(_round_half_up(cx + size / 2.0), 0), w)
    bottom = min(max(_round_half_up(cy + size / 2.0), 0), h)
    if right - left <= 0 or bottom - top <= 0:
        return None
    return left, top, right - left, bottom - top


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    flood = mask.copy()
    ff = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    return cv2.bitwise_or(cv2.bitwise_not(flood), mask)


def remove_red_eyes(img: np.ndarray, min_eye_size: float, red_threshold: float) -> np.ndarray:
    rgb = img[..., :3]
    src = to_u8(rgb)                       # R, G, B
    bgr = np.ascontiguousarray(src[..., ::-1])
    h, w = src.shape[:2]
    faces = detect_faces(bgr)
    rects = []
    for f in faces:
        eye = max(min_eye_size, min(float(f[2]), float(f[3])) * 0.16)
        for cx, cy in ((f[4], f[5]), (f[6], f[7])):
            rc = _eye_rect(w, h, float(cx), float(cy), eye)
            if rc is not None:
                rects.append(rc)
    out = rgb.copy()
    for x, y, rw, rh in rects:
        roi = bgr[y:y + rh, x:x + rw]
        b, g, r = roi[..., 0], roi[..., 1], roi[..., 2]
        bg = cv2.add(b, g)                                     # saturating like Core.add on CV_8U
        mask_red = np.where(r > red_threshold, 255, 0).astype(np.uint8)  # THRESH_BINARY
        mask_cmp = np.where(r > bg, 255, 0).astype(np.uint8)
        mask = cv2.bitwise_and(mask_red, mask_cmp)
        mask = cv2.dilate(_fill_holes(mask), np.ones((3, 3), np.uint8), iterations=3)
        mean = np.rint(bg.astype(np.float64) / 2.0).astype(F32) / 255.0  # saturate_cast(cvRound)
        # each eye is computed from the original image and pasted (overlaps: the later one wins)
        patch = rgb[y:y + rh, x:x + rw].copy()
        sel = mask > 0
        patch[sel] = mean[sel][:, None]
        out[y:y + rh, x:x + rw] = patch
    return out.astype(F32)
