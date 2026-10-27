"""Extended resampling filters: a port of pic-scale 0.6.6 (Rust), which ImageToolbox
uses through aire (``ResizeFunction`` x ``ScaleColorSpace``).

* ``KERNELS`` - the 49 ``ResizeFunction`` entries (snake_case).
* ``SCALE_COLOR_SPACES`` - the 19 ``ScaleColorSpace`` entries (snake_case).
* ``resample(img, width, height, method, colorspace)`` - separable resize of a
  float32 (H, W, 3|4) buffer. Alpha is premultiplied in the working space
  before filtering and unpremultiplied afterwards.

Kernel functions are ported from ``pic-scale/src/math/*.rs`` (``sinc`` and ``jinc``
there are the *unnormalised* ``sin(x)/x`` and ``J1(x)/x``). Weight generation is
our own (symmetric windows of the kernel's real support, scaled by the downscale
factor). It differs from pic-scale's ``generate_weights`` in documented ways, and
so do a few upstream kernels that are fixed because they break normalisation.
See docs/dev/port_reports/resample.md for the full comparison.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core.errors import invalid

F32 = np.float32
_PI = math.pi
_SQRT2 = math.sqrt(2.0)

# ---------------------------------------------------------------------------
# Kernel functions (pic-scale math/*.rs). Arguments are distances >= 0 in
# kernel units (pic-scale passes |dx| * filter_scale); float64 arrays.
# ---------------------------------------------------------------------------


def _sinc(x):
    """pxfm ``f_sinc``: sin(x)/x, 1 at 0 (np.sinc is the normalised sin(pi x)/(pi x))."""
    return np.sinc(np.asarray(x) / _PI)


def _jinc(x):
    """bessel_order_one.rs ``jinc``: J1(x)/x.

    DEVIATION: upstream returns 0 at x == 0, which zeroes the centre tap of
    every jinc-based filter (EWA-Hanning then has an all-zero weight sum and
    produces black pixels). We use the limit J1(x)/x -> 1/2.
    """
    from scipy.special import j1

    x = np.asarray(x, dtype=np.float64)
    safe = np.where(x == 0, 1.0, x)
    return np.where(x == 0, 0.5, j1(safe) / safe)


def _bilinear(x):
    x = np.abs(x)
    return np.where(x < 1, 1 - x, 0.0)


def _bc_spline(x, b, c):
    x = np.abs(x)
    x2 = x * x
    x3 = x2 * x
    near = ((12 - 9 * b - 6 * c) * x3 + (-18 + 12 * b + 6 * c) * x2 + (6 - 2 * b)) / 6
    far = ((-b - 6 * c) * x3 + (6 * b + 30 * c) * x2 + (-12 * b - 48 * c) * x + (8 * b + 24 * c)) / 6
    return np.where(x < 1, near, np.where(x < 2, far, 0.0))


def _hermite(x):
    return _bc_spline(x, 0.0, 0.0)


def _b_spline(x):
    return _bc_spline(x, 1.0, 0.0)


def _mitchell(x):
    return _bc_spline(x, 1 / 3, 1 / 3)


def _catmull_rom(x):
    return _bc_spline(x, 0.0, 0.5)


def _robidoux(x):
    # Upstream C = 13/(58+216*sqrt2) ~ 0.0358; Robidoux's published value is
    # 113/(58+216*sqrt2) ~ 0.3109. Kept as upstream (BC-splines stay normalised).
    return _bc_spline(x, 12 / (19 + 9 * _SQRT2), 13 / (58 + 216 * _SQRT2))


def _robidoux_sharp(x):
    return _bc_spline(x, 6 / (13 + 7 * _SQRT2), 7 / (2 + 12 * _SQRT2))


def _cubic_spline(x):
    x = np.abs(x)
    return np.where(x < 1, (4 + x * x * (3 * x - 6)) / 6,
                    np.where(x < 2, (8 + x * (-12 + x * (6 - x))) / 6, 0.0))


def _bicubic(x):
    a = -0.5
    m = np.abs(x)
    m2, m3 = m * m, m * m * m
    return np.where(m >= 2, 0.0, np.where(m <= 1, (a + 2) * m3 - (a + 3) * m2 + 1,
                                          a * m3 - 5 * a * m2 + 8 * a * m - 4 * a))


def _hann(x):
    return np.where(np.abs(x) > 2, 0.0, np.cos(x * _PI / 4) ** 2 / 4)


def _hamming(x):
    x = np.abs(x)
    return np.where(x == 0, 1.0, np.where(x >= 1, 0.0, 0.54 + 0.46 * np.cos(x * _PI)))


def _hanning(x):
    x = np.abs(x)
    return np.where(x == 0, 1.0, np.where(x >= 1, 0.0, 0.5 + 0.5 * np.cos(x * _PI)))


def _blackman(x):
    # Upstream quirk kept: the Blackman window is evaluated at |x|/2 in [0, 1),
    # i.e. it is *not* centred (~0 at x = 0, peak at |x| = 1), and sinc has no pi.
    # All weights stay >= 0, so normalisation is unaffected.
    x = np.abs(x)
    t = x / 2
    win = 0.42 - 0.49656062 * np.cos(2 * _PI * t) + 0.07684867 * np.cos(4 * _PI * t)
    return np.where(x < 2, _sinc(x) * win, 0.0)


def _welch(x):
    return np.where(x == 0, 1.0, np.where(x >= 1, 0.0, 1 - x * x))


def _quadric(x):
    x = np.abs(x)
    return np.where(x < 0.5, 0.75 - x * x, np.where(x < 1.5, 0.5 * (x - 1.5) ** 2, 0.0))


def _gaussian(x):
    # Upstream quirk kept: exp(-x / (2 sigma^2)^2) - Laplacian-shaped, not exp(-x^2/(2 sigma^2)).
    sigma = 0.35
    den = (2 * sigma * sigma) ** 2
    return (1 / (math.sqrt(2 * _PI) * sigma)) * np.exp(-x / den)


def _sphinx(x):
    y = np.asarray(x, dtype=np.float64) * _PI
    safe = np.where(np.abs(y) < 1e-8 * _PI, 1.0, y)
    val = 3.0 * (np.sin(safe) - safe * np.cos(safe)) / (safe * safe * safe)
    return np.where(np.abs(x) < 1e-8, 1.0, val)


def _bartlett(x):
    """DEVIATION: upstream is ``2x`` for x in [0,1] and ``2-2x`` beyond, a Bartlett
    window on [0, 2] that is not centred (0 at the tap centre, jumps from 2 to 0 at
    x = 1, negative after). Weight sums are 0 (black pixels in pic-scale at 2x
    upscale). We centre the same window: triangle 1 - |x| (half-width 1)."""
    return np.maximum(0.0, 1 - np.abs(x))


def _kaiser(x):
    from numpy import i0

    inside = np.sqrt(np.maximum(1.0 - x * x, 0.0))
    return np.where(x > 1, 0.0, i0(6.33 * inside) / i0(6.33))


def _bartlett_hann(x):
    """DEVIATION: upstream uses fac = |x - 0.5| on |x| <= 2 (window not centred: 0 at
    the tap centre, -0.48 at |x| = 2). For output pixels aligned with an input pixel
    the weight sum is negative, so normalisation flips the kernel. We use the centred
    Bartlett-Hann window over the same support |x| <= 2 (fac = |x|/4) with the
    canonical 0.48 coefficient (upstream 0.4832), which is 1 at 0 and 0 at |x| = 2."""
    x = np.abs(x)
    fac = x / 4
    return np.where(x > 2, 0.0, 0.62 - 0.48 * fac + 0.38 * np.cos(2 * _PI * fac))


def _box(x):
    return np.ones_like(np.asarray(x, dtype=np.float64))


def _bohman(x):
    ax = np.abs(x)
    d = _PI * ax
    return np.where(ax > 1, 0.0, (1 - ax) * np.cos(d) + np.sin(d) / _PI)


def _lanczos_sinc(a):
    def k(x):
        # sinc(pi x) * sinc(pi x / a) with unnormalised sinc == np.sinc(x) * np.sinc(x / a)
        return np.where(np.abs(x) < a, np.sinc(x) * np.sinc(x / a), 0.0)
    return k


def _lanczos_jinc(a):
    def k(x):
        # DEVIATION: upstream returns 0 at x == 0 (centre-tap hole); _jinc gives the limit 1/4.
        d = _PI * np.asarray(x, dtype=np.float64)
        return np.where(np.abs(x) < a, _jinc(d) * _jinc(d / a), 0.0)
    return k


def _lagrange(support: int):
    def k(x):
        x = np.asarray(x, dtype=np.float64)
        n = np.floor(support + x)  # usize cast of (support + x), x >= 0
        val = np.ones_like(x)
        for i in range(2 * support):
            den = n - i
            val = np.where(n != i, val * (den - x) / np.where(den == 0, 1.0, den), val)
        return np.where(x > support, 0.0, val)
    return k


def _spline16(x):
    t = x - 1
    return np.where(x < 1, ((x - 9 / 5) * x - 1 / 5) * x + 1, ((-1 / 3 * t + 4 / 5) * t - 7 / 15) * t)


def _spline36(x):
    t1, t2 = x - 1, x - 2
    return np.where(x < 1, ((13 / 11 * x - 453 / 209) * x - 3 / 209) * x + 1,
                    np.where(x < 2, ((-6 / 11 * t1 + 270 / 209) * t1 - 156 / 209) * t1,
                             ((1 / 11 * t2 - 45 / 209) * t2 + 26 / 209) * t2))


def _spline64(x):
    t1, t2, t3 = x - 1, x - 2, x - 3
    return np.where(x < 1, ((49 / 41 * x - 6387 / 2911) * x - 3 / 2911) * x + 1,
                    np.where(x < 2, ((-24 / 41 * t1 + 4032 / 2911) * t1 - 2328 / 2911) * t1,
                             np.where(x < 3, ((6 / 41 * t2 - 1008 / 2911) * t2 + 582 / 2911) * t2,
                                      ((-1 / 41 * t3 + 168 / 2911) * t3 - 97 / 2911) * t3)))


# ---------------------------------------------------------------------------
# Filter table (sampler.rs get_resampling_filter)
# ---------------------------------------------------------------------------

_JINC_R3 = 3.2383154841662362
_JINC_R4 = 4.2410628637960699
_EWA_ALPHA = 0.7  # scaler.rs: smoothing of consecutive weights for is_ewa filters


@dataclass(frozen=True)
class Filter:
    """One resampling filter.

    ``radius`` is the support we use, in kernel units: the kernel's real support,
    or pic-scale's ``min_kernel_size`` taken as a radius for unbounded kernels.
    ``min_kernel_size`` is the upstream value, kept for reference: pic-scale uses
    it as the tap *count*, i.e. radius min_kernel_size / 2.
    """
    kernel: Callable | None
    radius: float
    min_kernel_size: float
    ewa: bool = False
    resizable: bool = True
    window: Callable | None = None
    window_size: float = 0.0
    blur: float = 0.0
    taper: float = 0.0
    kind: str = "conv"  # conv | nearest | area

    def weights(self, t: np.ndarray) -> np.ndarray:
        """Weights at scaled distances t = |dx| / max(scale, 1) (scaler.rs generate_weights)."""
        if self.window is None:
            return self.kernel(t)
        x = t / self.blur if self.blur > 0 else t
        x = np.where(x <= self.taper, 0.0, (x - self.taper) / (1 - self.taper))
        # Upstream compares the *unscaled* distance with window_size, which truncates
        # the window at large downscale factors. We compare in kernel units (scale-invariant).
        win = np.where(x < self.window_size, self.window(x * self.window_size), 0.0)
        return win * self.kernel(x)


def _f(kernel, radius, mks, **kw) -> Filter:
    return Filter(kernel, float(radius), float(mks), **kw)


def _win(kernel, window, window_size, blur, mks, radius, ewa) -> Filter:
    return Filter(kernel, float(radius), float(mks), ewa=ewa, window=window, window_size=window_size, blur=blur)


KERNELS: dict[str, Filter] = {
    "bilinear": _f(_bilinear, 1, 2),
    "nearest": Filter(None, 0.5, 2, kind="nearest"),
    "cubic": _f(_cubic_spline, 2, 2),
    "mitchell_netravalli": _f(_mitchell, 2, 2),
    "catmull_rom": _f(_catmull_rom, 2, 2),
    "hermite": _f(_hermite, 1, 2),
    "b_spline": _f(_b_spline, 2, 2),
    "hann": _f(_hann, 2, 3),
    "bicubic": _f(_bicubic, 2, 2),
    "hamming": _f(_hamming, 1, 2),
    "hanning": _f(_hanning, 1, 2),
    "blackman": _f(_blackman, 2, 2),
    "welch": _f(_welch, 1, 2),
    "quadric": _f(_quadric, 1.5, 2),
    "gaussian": _f(_gaussian, 2, 2),
    "sphinx": _f(_sphinx, 2, 2),
    "bartlett": _f(_bartlett, 1, 2),
    "robidoux": _f(_robidoux, 2, 2),
    "robidoux_sharp": _f(_robidoux_sharp, 2, 2),
    "spline16": _f(_spline16, 2, 2, resizable=False),
    "spline36": _f(_spline36, 3, 4, resizable=False),
    "spline64": _f(_spline64, 4, 6, resizable=False),
    "kaiser": _f(_kaiser, 1, 2),
    "bartlett_hann": _f(_bartlett_hann, 2, 2),
    "box": _f(_box, 0.5, 2),
    "bohman": _f(_bohman, 1, 2),
    "lanczos2": _f(_lanczos_sinc(2), 2, 2),
    "lanczos3": _f(_lanczos_sinc(3), 3, 3),
    "lanczos4": _f(_lanczos_sinc(4), 4, 4),
    "lanczos2_jinc": _f(_lanczos_jinc(2), 2, 2),
    "lanczos3_jinc": _f(_lanczos_jinc(3), 3, 3),
    "lanczos4_jinc": _f(_lanczos_jinc(4), 4, 4),
    "ewa_hanning": _win(_jinc, _hanning, 2, 0, 1, radius=0.5, ewa=True),
    "ewa_robidoux": _f(_robidoux, 2, 2, ewa=True),
    "ewa_blackman": _f(_blackman, 2, 2, ewa=True),
    "ewa_quadric": _f(_quadric, 1.5, 2, ewa=True),
    "ewa_robidoux_sharp": _f(_robidoux_sharp, 2, 2, ewa=True),
    "ewa_lanczos3_jinc": _f(_lanczos_jinc(3), 3, 3, ewa=True),
    "ginseng": _win(_sinc, _jinc, 3, 1, 3, radius=3, ewa=False),
    "ewa_ginseng": _win(_sinc, _jinc, _JINC_R3, 1, 3, radius=_JINC_R3, ewa=True),
    "ewa_lanczos_sharp": _win(_jinc, _jinc, _JINC_R3, 0.9812505837223707, 3,
                              radius=_JINC_R3 * 0.9812505837223707, ewa=True),
    "ewa_lanczos4_sharpest": _win(_jinc, _jinc, _JINC_R4, 0.8845120932605005, 4,
                                  radius=_JINC_R4 * 0.8845120932605005, ewa=True),
    "ewa_lanczos_soft": _win(_jinc, _jinc, _JINC_R3, 1.0164667662867047, 3,
                             radius=_JINC_R3 * 1.0164667662867047, ewa=True),
    # hanning(3 x) is 0 for x >= 1/3, x = |dx| / 1.11 -> support 1.11 / 3
    "haasn_soft": _win(_jinc, _hanning, 3, 1.11, 3, radius=1.11 / 3, ewa=False),
    "lagrange2": _f(_lagrange(2), 2, 2),
    "lagrange3": _f(_lagrange(3), 3, 3),
    "lanczos6": _f(_lanczos_sinc(6), 6, 6),
    "lanczos6_jinc": _f(_lanczos_jinc(6), 6, 6),
    "area": Filter(None, 0.5, 2, kind="area"),
}

#: OpenCV-style names accepted by resample() (geometry.resize routes them here for
#: non-sRGB, non-linear colour spaces).
ALIASES = {"lanczos": "lanczos4"}  # cv2.INTER_LANCZOS4 is Lanczos a=4

# ---------------------------------------------------------------------------
# Weight matrices
# ---------------------------------------------------------------------------


def _ewa_smooth(w: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Zero-phase version of scaler.rs' is_ewa smoothing.

    Upstream runs w'[j] = a*w[j] + (1-a)*w'[j-1] left to right over the taps
    (a = 0.7). That causal one-pole filter moves the kernel centroid by
    (1-a)/a ~ 0.43 taps, which shifts the image. We run the same recurrence
    left to right and right to left and average the two.
    """
    a = _EWA_ALPHA
    fwd, bwd = w.copy(), w.copy()
    n = w.shape[1]
    for t in range(1, n):
        fwd[:, t] = np.where(valid[:, t - 1], a * w[:, t] + (1 - a) * fwd[:, t - 1], w[:, t])
    for t in range(n - 2, -1, -1):
        bwd[:, t] = np.where(valid[:, t + 1], a * w[:, t] + (1 - a) * bwd[:, t + 1], w[:, t])
    return np.where(valid, 0.5 * (fwd + bwd), 0.0)


def weight_matrix(n_in: int, n_out: int, f: Filter):
    """Sparse (n_out x n_in) CSR matrix of normalised weights for one axis."""
    import scipy.sparse as sp

    scale = n_in / n_out
    i = np.arange(n_out, dtype=np.float64)
    center = (i + 0.5) * scale - 0.5
    if f.kind == "area" and scale >= 1:
        # downscale: exact pixel coverage, i.e. OpenCV INTER_AREA (pic-scale's comment says
        # it replicates INTER_AREA, but its code averages a flat box twice as wide)
        lo, hi = i * scale, (i + 1) * scale
        k = np.floor(lo).astype(np.int64)[:, None] + np.arange(int(math.ceil(scale)) + 1)
        w = np.clip(np.minimum(k + 1, hi[:, None]) - np.maximum(k, lo[:, None]), 0.0, None)
        valid = (w > 0) & (k < n_in)
    elif f.kind == "area":
        # upscale: scaler.rs is_area branch (OpenCV INTER_AREA upscale coefficients), verbatim
        sx = np.floor(i * scale)
        fx = (i + 1) - (sx + 1) / scale
        d = np.where(fx <= 0, 0.0, fx - np.floor(fx))
        k = sx.astype(np.int64)[:, None] + np.arange(2)
        w = np.stack([1 - d, d], axis=1)
        valid = k < n_in
    else:
        # Scaled distance t = |dx| / max(scale, 1) (fixed kernels: |dx|), from exact integers:
        # dx = ((2k+1) n_out - (2i+1) n_in) / (2 n_out). Taps at equal distances, e.g. mirror
        # images or ties on the support edge of discontinuous kernels, get identical weights.
        denom = 2 * (max(n_in, n_out) if f.resizable else n_out)
        support = f.radius * denom / (2 * n_out)  # in input pixels
        k = np.floor(center - support).astype(np.int64)[:, None] - 1 + np.arange(int(2 * support) + 4)
        num = (2 * k + 1) * n_out - (2 * np.arange(n_out, dtype=np.int64)[:, None] + 1) * n_in
        t = np.abs(num) / denom
        valid = (t <= f.radius) & (k >= 0) & (k < n_in)
        w = np.where(valid, f.weights(t), 0.0)
        if f.ewa:
            w = _ewa_smooth(w, valid)
    w = np.where(valid, w, 0.0)
    total = w.sum(axis=1)
    # A non-positive sum means no usable tap (kernel narrower than the tap spacing, e.g.
    # haasn_soft / ewa_hanning when upscaling). pic-scale leaves those rows at 0 (black
    # pixels); we use the limit of a shrinking kernel, the nearest input pixel.
    bad = total <= 1e-12
    w = w / np.where(bad, 1.0, total)[:, None]
    valid &= ~bad[:, None]
    rows = np.broadcast_to(np.arange(n_out)[:, None], k.shape)
    r, c, v = rows[valid], k[valid], w[valid]
    if bad.any():
        bi = np.nonzero(bad)[0]
        near = np.clip(np.floor(center[bi] + 0.5), 0, n_in - 1).astype(np.int64)
        r, c, v = np.concatenate([r, bi]), np.concatenate([c, near]), np.concatenate([v, np.ones(len(bi))])
    return sp.csr_matrix((v.astype(F32), (r, c)), shape=(n_out, n_in))


def _resize_rows(buf: np.ndarray, m) -> np.ndarray:
    h, w, c = buf.shape
    out = m @ np.ascontiguousarray(buf).reshape(h, w * c)
    return np.asarray(out, dtype=F32).reshape(m.shape[0], w, c)


def _resize_cols(buf: np.ndarray, m) -> np.ndarray:
    h, w, c = buf.shape
    t = np.ascontiguousarray(buf.transpose(1, 0, 2)).reshape(w, h * c)
    out = np.asarray(m @ t, dtype=F32).reshape(m.shape[0], h, c)
    return np.ascontiguousarray(out.transpose(1, 0, 2))


def _nearest_index(n_in: int, n_out: int) -> np.ndarray:
    # pic-scale truncates (x + 0.5) * scale - 0.5, a half-pixel bias; we take the
    # pixel whose cell contains the output centre.
    return np.minimum(np.floor((np.arange(n_out) + 0.5) * (n_in / n_out)).astype(np.int64), n_in - 1)


# ---------------------------------------------------------------------------
# Colour spaces (aire resizeImpl -> pic-scale colors/*.rs -> colorutils-rs 0.7)
# ---------------------------------------------------------------------------
# colorutils clamps transfer functions and L to [0, 1] / [0, 100]; for its 8-bit
# inputs that is a no-op. Here the clamps are dropped and powers are sign-mirrored
# so that float/HDR buffers and filter overshoot survive the round trip.

def _signed_pow(v, p):
    return np.copysign(np.abs(v) ** p, v)


def _srgb_to_lin(v):
    a = np.abs(v)
    out = np.where(a < 12.92 * 0.0030412825601275209, a / 12.92,
                   ((a + 0.0550107189475866) / 1.0550107189475866) ** 2.4)
    return np.copysign(out, v).astype(F32)


def _lin_to_srgb(v):
    a = np.abs(v)
    out = np.where(a < 0.0030412825601275209, a * 12.92,
                   1.0550107189475866 * a ** (1 / 2.4) - 0.0550107189475866)
    return np.copysign(out, v).astype(F32)


def _rec709_to_lin(v):
    a = np.abs(v)
    out = np.where(a < 4.5 * 0.018053968510807, a / 4.5,
                   ((a + 0.09929682680944) / 1.09929682680944) ** (1 / 0.45))
    return np.copysign(out, v).astype(F32)


def _lin_to_rec709(v):
    a = np.abs(v)
    out = np.where(a < 0.018053968510807, a * 4.5, 1.09929682680944 * a ** 0.45 - 0.09929682680944)
    return np.copysign(out, v).astype(F32)


_TRANSFER = {  # name -> (to linear, from linear)
    "srgb": (_srgb_to_lin, _lin_to_srgb),
    "rec709": (_rec709_to_lin, _lin_to_rec709),
    "gamma22": (lambda v: _signed_pow(v, 2.2).astype(F32), lambda v: _signed_pow(v, 1 / 2.2).astype(F32)),
    "gamma28": (lambda v: _signed_pow(v, 2.8).astype(F32), lambda v: _signed_pow(v, 1 / 2.8).astype(F32)),
}

_SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]], F32)
_XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                         [-0.9692660, 1.8760108, 0.0415560],
                         [0.0556434, -0.2040259, 1.0572252]], F32)
_WHITE = np.array([0.95047, 1.0, 1.08883], F32)
_U_WHITE = 4 * 100.0 / (95.047 + 15 * 100.0 + 3 * 108.883)
_V_WHITE = 9 * 100.0 / (95.047 + 15 * 100.0 + 3 * 108.883)


def _mat(v, m):
    return (v.reshape(-1, 3) @ m.T).reshape(v.shape).astype(F32)


def _lin_to_lab(lin):
    t = _mat(lin, _SRGB_TO_XYZ) / _WHITE
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
                    axis=-1).astype(F32)


def _lab_to_lin(lab):
    y = (lab[..., 0] + 16) / 116
    f = np.stack([lab[..., 1] / 500 + y, y, y - lab[..., 2] / 200], axis=-1)
    f3 = f * f * f
    xyz = np.where(f3 > 0.008856, f3, (f - 16 / 116) / 7.787) * _WHITE
    return _mat(xyz, _XYZ_TO_SRGB)


def _xyz_to_luv(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    den = x + 15 * y + 3 * z
    L = np.where(y < (6 / 29) ** 3, (29 / 3) ** 3 * y, 116 * np.cbrt(y) - 16)
    ok = den != 0
    sden = np.where(ok, den, 1.0)
    u = np.where(ok, 13 * L * (4 * x / sden - _U_WHITE), 0.0)
    v = np.where(ok, 13 * L * (9 * y / sden - _V_WHITE), 0.0)
    return np.stack([L, u, v], axis=-1).astype(F32)


def _luv_to_xyz(luv):
    L, u, v = luv[..., 0], luv[..., 1], luv[..., 2]
    pos = L > 0
    l13 = 1 / (13 * np.where(pos, L, 1.0))
    up = u * l13 + _U_WHITE
    vp = v * l13 + _V_WHITE
    y = np.where(L > 8, ((L + 16) / 116) ** 3, L * (3 / 29) ** 3)
    okv = vp != 0
    den = 1 / (4 * np.where(okv, vp, 1.0))
    x = np.where(okv, y * 9 * up * den, 0.0)
    z = np.where(okv, y * (12 - 3 * up - 20 * vp) * den, 0.0)
    return (np.stack([x, y, z], axis=-1) * pos[..., None]).astype(F32)


def _lin_to_luv(lin):
    return _xyz_to_luv(_mat(lin, _SRGB_TO_XYZ))


def _luv_to_lin(luv):
    return _mat(_luv_to_xyz(luv), _XYZ_TO_SRGB)


def _lin_to_lch(lin):
    luv = _lin_to_luv(lin)
    return np.stack([luv[..., 0], np.hypot(luv[..., 1], luv[..., 2]), np.arctan2(luv[..., 2], luv[..., 1])],
                    axis=-1).astype(F32)


def _lch_to_lin(lch):
    luv = np.stack([lch[..., 0], lch[..., 1] * np.cos(lch[..., 2]), lch[..., 1] * np.sin(lch[..., 2])], axis=-1)
    return _luv_to_lin(luv)


_OK_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]], F32)
_OK_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]], F32)
_OK_M2_INV = np.array([[1.0, 0.3963377774, 0.2158037573],
                       [1.0, -0.1055613458, -0.0638541728],
                       [1.0, -0.0894841775, -1.2914855480]], F32)
_OK_M1_INV = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                       [-1.2684380046, 2.6097574011, -0.3413193965],
                       [-0.0041960863, -0.7034186147, 1.7076147010]], F32)


def _lin_to_oklab(lin):
    return _mat(np.cbrt(_mat(lin, _OK_M1)), _OK_M2)


def _oklab_to_lin(lab):
    lms = _mat(lab, _OK_M2_INV)
    return _mat(lms * lms * lms, _OK_M1_INV)


_JZ_LUMINANCE = 200.0  # aire: JzazbzScaler::new(resampling, 200., transfer)
_JZ_D0 = 1.6295499532821566e-11
_JZ_LMS = np.array([[0.674207838, 0.382799340, -0.047570458],
                    [0.149284160, 0.739628340, 0.083327300],
                    [0.070941080, 0.174768000, 0.670970020]])
_JZ_IAB = np.array([[0.5, 0.5, 0.0],
                    [3.524000, -4.066708, 0.542708],
                    [0.199076, 1.096799, -1.295875]])
_JZ_IAB_INV = np.array([[1.0, 1.386050432715393e-1, 5.804731615611869e-2],
                        [1.0, -1.386050432715393e-1, -5.804731615611891e-2],
                        [1.0, -9.601924202631895e-2, -8.118918960560390e-1]])
_JZ_LMS_INV = np.array([[1.661373055774069e+00, -9.145230923250668e-01, 2.313620767186147e-01],
                        [-3.250758740427037e-01, 1.571847038366936e+00, -2.182538318672940e-01],
                        [-9.098281098284756e-02, -3.127282905230740e-01, 1.522766561305260e+00]])


def _pq(x):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        xx = np.power(np.maximum(x, 0.0) * 1e-4, 0.1593017578125)
        r = np.power((0.8359375 + 18.8515625 * xx) / (1 + 18.6875 * xx), 134.034375)
    return np.where((x <= 0) | ~np.isfinite(r), 0.0, r)


def _pq_inv(x):
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        xx = np.power(np.maximum(x, 0.0), 7.460772656268214e-03)
        r = 1e4 * np.power((0.8359375 - xx) / (18.6875 * xx - 18.8515625), 6.277394636015326)
    return np.where((x <= 0) | ~np.isfinite(r), 0.0, r)


def _lin_to_jzazbz(lin):
    # float64 here: PQ's large exponents lose too much precision in float32
    xyz = lin.reshape(-1, 3).astype(np.float64) @ _SRGB_TO_XYZ.T.astype(np.float64) * _JZ_LUMINANCE
    iab = _pq(xyz @ _JZ_LMS.T) @ _JZ_IAB.T
    iz = iab[:, 0]
    jz = 0.44 * iz / (1 - 0.56 * iz) - _JZ_D0
    return np.stack([jz, iab[:, 1], iab[:, 2]], axis=-1).reshape(lin.shape).astype(F32)


def _jzazbz_to_lin(jab):
    flat = jab.reshape(-1, 3).astype(np.float64)
    jz = flat[:, 0] + _JZ_D0
    iz = jz / (0.44 + 0.56 * jz)
    lms = _pq_inv(np.stack([iz, flat[:, 1], flat[:, 2]], axis=-1) @ _JZ_IAB_INV.T)
    xyz = (lms @ _JZ_LMS_INV.T) / _JZ_LUMINANCE
    return (xyz @ _XYZ_TO_SRGB.T.astype(np.float64)).reshape(jab.shape).astype(F32)


def _to_sigmoidal(v):
    return (1 / (1 + np.exp(-v))).astype(F32)


def _from_sigmoidal(v):
    den = 1 - v
    with np.errstate(divide="ignore", invalid="ignore"):
        k = v / np.where(den == 0, 1.0, den)
        out = np.log(np.where(k > 0, k, 1.0))
    return np.where((den == 0) | (k <= 0), 0.0, out).astype(F32)


_MODELS = {  # model -> (from linear RGB, to linear RGB); "linear" is the identity
    "linear": (lambda v: v, lambda v: v),
    "xyz": (lambda v: _mat(v, _SRGB_TO_XYZ), lambda v: _mat(v, _XYZ_TO_SRGB)),
    "lab": (_lin_to_lab, _lab_to_lin),
    "luv": (_lin_to_luv, _luv_to_lin),
    "lch": (_lin_to_lch, _lch_to_lin),
    "oklab": (_lin_to_oklab, _oklab_to_lin),
    "jzazbz": (_lin_to_jzazbz, _jzazbz_to_lin),
}

# ScaleColorSpace (aire) -> (model, transfer function); order = upstream enum values 0..18
_SPACES: dict[str, tuple[str, str] | None] = {
    "srgb": None,
    "lab": ("lab", "srgb"),
    "linear": ("linear", "srgb"),          # pic-scale LinearScaler (12-bit fixed point upstream)
    "luv": ("luv", "srgb"),
    "sigmoidal": ("sigmoidal", ""),        # logistic of the encoded values, no linearisation
    "xyz": ("xyz", "srgb"),
    "linear_f32_srgb": ("linear", "srgb"),
    "linear_f32_rec709": ("linear", "rec709"),
    "linear_f32_gamma_2_2": ("linear", "gamma22"),
    "linear_f32_gamma_2_8": ("linear", "gamma28"),
    "lch": ("lch", "srgb"),                # CIE LCh(uv), hue in radians filtered as a scalar
    "oklab_srgb": ("oklab", "srgb"),
    "oklab_rec709": ("oklab", "rec709"),
    "oklab_gamma_2_2": ("oklab", "gamma22"),
    "oklab_gamma_2_8": ("oklab", "gamma28"),
    "jzazbz_srgb": ("jzazbz", "srgb"),
    "jzazbz_rec709": ("jzazbz", "rec709"),
    "jzazbz_gamma_2_2": ("jzazbz", "gamma22"),
    "jzazbz_gamma_2_8": ("jzazbz", "gamma28"),
}
SCALE_COLOR_SPACES: list[str] = list(_SPACES)

#: spaces in which filtering is a linear-light operation on RGB (geometry.resize keeps
#: OpenCV's kernels for these by converting before/after)
LINEAR_LIGHT_SPACES = frozenset(k for k, v in _SPACES.items() if v and v[0] == "linear")


def _check_space(colorspace: str) -> None:
    if colorspace not in _SPACES:
        raise invalid(f"Unknown scale colour space {colorspace!r}", param="colorspace",
                      choices=SCALE_COLOR_SPACES)


def _encode(rgb: np.ndarray, colorspace: str) -> np.ndarray:
    spec = _SPACES[colorspace]
    if spec is None:
        return rgb.astype(F32, copy=False)
    model, tf = spec
    if model == "sigmoidal":
        return _to_sigmoidal(rgb)
    return _MODELS[model][0](_TRANSFER[tf][0](rgb))


def _decode(work: np.ndarray, colorspace: str) -> np.ndarray:
    spec = _SPACES[colorspace]
    if spec is None:
        return work.astype(F32, copy=False)
    model, tf = spec
    if model == "sigmoidal":
        return _from_sigmoidal(work)
    return _TRANSFER[tf][1](_MODELS[model][1](work))


def _pixelwise(fn: Callable[[np.ndarray], np.ndarray], img: np.ndarray) -> np.ndarray:
    """Apply a per-pixel function (same channel count in and out) over row blocks in
    a thread pool; numpy ufuncs release the GIL and blocks stay cache-sized."""
    h, w = img.shape[:2]
    step = max(1, (1 << 16) // max(1, w))
    if h <= step:
        return fn(img)
    out = np.empty(img.shape, F32)

    def run(y0):
        out[y0:y0 + step] = fn(img[y0:y0 + step])

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as ex:
        list(ex.map(run, range(0, h, step)))
    return out


def _with_alpha(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    return lambda b: np.concatenate([fn(b[..., :3]), b[..., 3:]], axis=2) if b.shape[2] > 3 else fn(b)


def encode_space(img: np.ndarray, colorspace: str) -> np.ndarray:
    """sRGB-encoded (H, W, 3|4) buffer -> working colour space (alpha passed through)."""
    _check_space(colorspace)
    return _pixelwise(_with_alpha(lambda b: _encode(b, colorspace)), img)


def decode_space(img: np.ndarray, colorspace: str) -> np.ndarray:
    """Inverse of :func:`encode_space`."""
    _check_space(colorspace)
    return _pixelwise(_with_alpha(lambda b: _decode(b, colorspace)), img)


def _encode_premul(b: np.ndarray, colorspace: str) -> np.ndarray:
    work = _encode(b[..., :3], colorspace)
    if b.shape[2] < 4:
        return work
    a = b[..., 3:4].astype(F32)
    return np.concatenate([work * a, a], axis=2)


def _decode_unpremul(b: np.ndarray, colorspace: str) -> np.ndarray:
    if b.shape[2] < 4:
        return _decode(b, colorspace)
    a = np.clip(b[..., 3:4], 0.0, 1.0)
    col = np.where(a > 1e-6, b[..., :3] / np.maximum(a, 1e-6), 0.0).astype(F32)
    return np.concatenate([_decode(col, colorspace), a], axis=2)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def resample(img: np.ndarray, width: int, height: int, method: str, colorspace: str = "srgb") -> np.ndarray:
    """Resize a float32 (H, W, 3|4) buffer with a pic-scale filter in a pic-scale colour space."""
    name = ALIASES.get(method, method)
    if name not in KERNELS:
        raise invalid(f"Unknown resampling method {method!r}", param="method", choices=list(KERNELS))
    _check_space(colorspace)
    f = KERNELS[name]
    nw, nh = max(1, int(width)), max(1, int(height))
    h, w = img.shape[:2]
    if (nw, nh) == (w, h):
        return img.astype(F32, copy=False)
    if f.kind == "nearest":  # point sampling commutes with any per-pixel colour transform
        return np.ascontiguousarray(img[_nearest_index(h, nh)][:, _nearest_index(w, nw)], dtype=F32)

    alpha = img.shape[2] == 4
    buf = img if colorspace == "srgb" and not alpha else _pixelwise(lambda b: _encode_premul(b, colorspace), img)
    # The two passes commute; run the column pass (which transposes) on the smaller buffer.
    passes = [(_resize_rows, h, nh), (_resize_cols, w, nw)]
    for apply, n_in, n_out in (passes[::-1] if nh > h else passes):
        if n_in != n_out:
            buf = apply(buf, weight_matrix(n_in, n_out, f))
    if colorspace == "srgb" and not alpha:
        return buf
    return _pixelwise(lambda b: _decode_unpremul(b, colorspace), buf)
