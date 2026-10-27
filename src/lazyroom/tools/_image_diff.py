"""Port of ImageToolbox ``ImageDiffTool`` (lib/opencv-tools, Apache-2.0, T8RIN).

Pixel-by-pixel comparison used by the Compare screen ("Pixel by pixel"), Compare PDF and the
Compression Lab.  Six metrics (``ComparisonType``): SSIM, AE, MAE, NCC, PSNR, RMSE.

Faithful to the Kotlin/OpenCV code:
* both images are 8-bit RGBA (Android ``Bitmap`` -> ``Utils.bitmapToMat``: premultiplied RGBA);
  every channel including alpha participates;
* the second image is ``resizeAndPad``-ed to the first one's size (aspect-preserving
  ``Imgproc.resize`` (bilinear) anchored top-left, zero padding = transparent black);
* values are float32 in 0..255; Gaussian window min(11, side) (odd), sigma 1.5;
* ``threshold`` (0..100) maps to a 0..255 pixel error for AE/MAE/RMSE/PSNR, to ``1 - t/100``
  minimum SSIM and to ``1 - 2t/100`` minimum NCC; masked pixels are painted ``highlight_color``
  on a copy of the first image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..core import fast as _fast  # noqa: F401  (sets NUMBA_CACHE_DIR before numba is imported)
from numba import njit, prange  # noqa: E402

COMPARISON_TYPES = ["SSIM", "AE", "MAE", "NCC", "PSNR", "RMSE"]  # ComparisonType.entries order

DEFAULT_WINDOW_SIZE = 11
GAUSSIAN_SIGMA = 1.5
SSIM_C1 = 6.5025  # (0.01 * 255)^2
SSIM_C2 = 58.5225  # (0.03 * 255)^2
EPSILON = 1e-6


@dataclass
class DiffResult:
    score: float
    mask: np.ndarray          # bool HxW, True = "different"
    highlighted: np.ndarray   # uint8 HxWx4 RGBA (premultiplied like the Android bitmap)


def to_rgba8(img: np.ndarray) -> np.ndarray:
    """Float 0..1 RGB(A) -> premultiplied 8-bit RGBA (what ``Utils.bitmapToMat`` returns)."""
    x = np.clip(img, 0, 1)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, 2)
    if x.shape[2] == 3:
        x = np.dstack([x, np.ones(x.shape[:2], x.dtype)])
    u = np.round(x * 255.0).astype(np.uint8)
    if (u[..., 3] < 255).any():  # Android bitmaps store premultiplied colour
        a = u[..., 3:4].astype(np.uint16)
        u[..., :3] = ((u[..., :3].astype(np.uint16) * a + 127) // 255).astype(np.uint8)
    return u


def resize_and_pad(src: np.ndarray, width: int, height: int) -> np.ndarray:
    """``Mat.resizeAndPad`` (OpenCVUtils.kt)."""
    aspect = src.shape[1] / src.shape[0]
    target_aspect = width / height
    if aspect > target_aspect:
        nw, nh = width, max(1, int(width / aspect))
    else:
        nw, nh = max(1, int(height * aspect)), height
    resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((height, width) + src.shape[2:], src.dtype)
    out[:nh, :nw] = resized[:height, :width]
    return out


def _window(h: int, w: int) -> tuple[int, int]:
    def odd(v: int) -> int:
        c = min(DEFAULT_WINDOW_SIZE, max(1, v))
        return max(1, c - 1) if c % 2 == 0 else c

    return odd(w), odd(h)


def _blur(x: np.ndarray, win: tuple[int, int]) -> np.ndarray:
    return cv2.GaussianBlur(x, win, GAUSSIAN_SIGMA)


# Fused per-pixel kernels (numba) - the maths of the Kotlin Core.* call chains without numpy temporaries.
@njit(cache=True, parallel=True)
def _diff_maps(a, b, out_max, out_mean, out_mse):  # pragma: no cover - jitted
    h, w, c = a.shape
    for y in prange(h):
        for x in range(w):
            mx = np.float32(0.0)
            sa = np.float32(0.0)
            sq = np.float32(0.0)
            for k in range(c):
                d = np.float32(a[y, x, k]) - np.float32(b[y, x, k])
                ad = abs(d)
                if ad > mx:
                    mx = ad
                sa += ad
                sq += d * d
            out_max[y, x] = mx
            out_mean[y, x] = sa / np.float32(c)
            out_mse[y, x] = sq / np.float32(c)


@njit(cache=True, parallel=True)
def _ssim_combine(mu_a, mu_b, saa, sbb, sab, out_min, row_sums):  # pragma: no cover - jitted
    h, w = mu_a.shape
    for y in prange(h):
        acc = 0.0
        for x in range(w):
            ma = mu_a[y, x]
            mb = mu_b[y, x]
            maa = ma * ma
            mbb = mb * mb
            mab = ma * mb
            va = saa[y, x] - maa
            if va < 0:
                va = np.float32(0.0)
            vb = sbb[y, x] - mbb
            if vb < 0:
                vb = np.float32(0.0)
            cov = sab[y, x] - mab
            num = (np.float32(2.0) * mab + np.float32(SSIM_C1)) * (np.float32(2.0) * cov + np.float32(SSIM_C2))
            den = (maa + mbb + np.float32(SSIM_C1)) * (va + vb + np.float32(SSIM_C2)) + np.float32(EPSILON)
            v = num / den
            if v < 0:
                v = np.float32(0.0)
            elif v > 1:
                v = np.float32(1.0)
            acc += v
            if v < out_min[y, x]:
                out_min[y, x] = v
        row_sums[y] = acc


@njit(cache=True, parallel=True)
def _ncc_combine(mu_a, mu_b, saa, sbb, sab, out_min):  # pragma: no cover - jitted
    h, w = mu_a.shape
    eps = np.float32(EPSILON)
    for y in prange(h):
        for x in range(w):
            ma = mu_a[y, x]
            mb = mu_b[y, x]
            va = saa[y, x] - ma * ma
            if va < 0:
                va = np.float32(0.0)
            vb = sbb[y, x] - mb * mb
            if vb < 0:
                vb = np.float32(0.0)
            cov = sab[y, x] - ma * mb
            d2 = va * vb
            if d2 < 0:
                d2 = np.float32(0.0)
            den = np.sqrt(d2)
            if den <= eps:  # flat windows: similarity of the local means
                v = np.float32(1.0) - abs(ma - mb) / np.float32(255.0)
                if v < 0:
                    v = np.float32(0.0)
                elif v > 1:
                    v = np.float32(1.0)
            else:
                v = cov / (den + eps)
            if v > 1:
                v = np.float32(1.0)
            elif v < -1:
                v = np.float32(-1.0)
            if v < out_min[y, x]:
                out_min[y, x] = v


@njit(cache=True, parallel=True)
def _centered_sums(a, b, ma, mb):  # pragma: no cover - jitted
    h, w = a.shape
    sab = 0.0
    saa = 0.0
    sbb = 0.0
    sdd = 0.0
    for y in prange(h):
        for x in range(w):
            da = np.float64(a[y, x]) - ma
            db = np.float64(b[y, x]) - mb
            sab += da * db
            saa += da * da
            sbb += db * db
            d = np.float64(a[y, x]) - np.float64(b[y, x])
            sdd += d * d
    return sab, saa, sbb, sdd


def _global_ncc(a: np.ndarray, b: np.ndarray) -> float:
    ma, mb = float(a.mean(dtype=np.float64)), float(b.mean(dtype=np.float64))
    num, ea, eb, sdd = _centered_sums(a, b, ma, mb)
    den = math.sqrt(max(ea, 0.0) * max(eb, 0.0))
    if den <= EPSILON:  # constant channel(s): identical -> 1, different -> 0
        return 1.0 if sdd / a.size <= EPSILON else 0.0
    return max(-1.0, min(1.0, num / den))


def _moments(a: np.ndarray, b: np.ndarray, win):
    return (_blur(a, win), _blur(b, win), _blur(cv2.multiply(a, a), win), _blur(cv2.multiply(b, b), win),
            _blur(cv2.multiply(a, b), win))


def _channels(x: np.ndarray) -> list[np.ndarray]:
    return [np.ascontiguousarray(x[..., c], dtype=np.float32) for c in range(x.shape[2])]


def compute(first: np.ndarray, second: np.ndarray, comparison_type: str = "AE") -> tuple[float, np.ndarray]:
    """Score and the per-pixel similarity/error map used for highlighting (both inputs: RGBA uint8)."""
    t = comparison_type.upper()
    if t not in COMPARISON_TYPES:
        raise ValueError(f"Unknown comparison type {comparison_type!r}; use one of {COMPARISON_TYPES}")
    if second.shape[:2] != first.shape[:2]:
        second = resize_and_pad(second, first.shape[1], first.shape[0])
    h, w = first.shape[:2]
    win = _window(h, w)
    if t in ("AE", "MAE", "RMSE", "PSNR"):
        mx, mean, mse = (np.empty((h, w), np.float32) for _ in range(3))
        _diff_maps(np.ascontiguousarray(first), np.ascontiguousarray(second), mx, mean, mse)
        if t == "AE":
            return float("nan"), mx
        if t == "MAE":
            return float(mean.mean(dtype=np.float64)), mean
        g = max(float(mse.mean(dtype=np.float64)), 0.0)
        if t == "RMSE":
            return math.sqrt(g), np.sqrt(mse)
        score = math.inf if g <= EPSILON else 10.0 * math.log10(255.0 * 255.0 / g)
        return score, np.sqrt(np.maximum(_blur(mse, win), 0))
    ca, cb = _channels(first), _channels(second)
    out = np.full((h, w), np.float32(np.inf), np.float32)
    if t == "SSIM":
        means = []
        rows = np.empty(h, np.float64)
        for a, b in zip(ca, cb):
            _ssim_combine(*_moments(a, b, win), out, rows)
            means.append(rows.sum() / (h * w))
        return min(1.0, max(0.0, float(np.mean(means)))), out
    score = float(np.mean([_global_ncc(a, b) for a, b in zip(ca, cb)]))
    for a, b in zip(ca, cb):
        _ncc_combine(*_moments(a, b, win), out)
    return max(-1.0, min(1.0, score)), out


def highlight_differences(first: np.ndarray, second: np.ndarray, comparison_type: str = "AE",
                          highlight_color: tuple[int, int, int, int] = (255, 0, 0, 255),
                          threshold: float = 4.0) -> DiffResult:
    """``ImageDiffTool.highlightDifferences`` on two RGBA uint8 images (second is resized/padded)."""
    t = comparison_type.upper()
    tol = min(100.0, max(0.0, float(threshold)))
    pixel_threshold = tol * 255.0 / 100.0
    if second.shape[:2] != first.shape[:2]:
        second = resize_and_pad(second, first.shape[1], first.shape[0])
    score, m = compute(first, second, t)
    if t == "AE":
        mask = m > pixel_threshold
        score = float(np.count_nonzero(mask))
    elif t in ("MAE", "RMSE", "PSNR"):
        mask = m > pixel_threshold
    elif t == "SSIM":
        mask = m < 1.0 - tol / 100.0
    else:
        mask = m < 1.0 - 2.0 * tol / 100.0
    out = first.copy()
    out[mask] = np.array(highlight_color, np.uint8)
    return DiffResult(score=score, mask=mask, highlighted=out)


def metric(first: np.ndarray, second: np.ndarray, comparison_type: str) -> float:
    """Just the score (what Compression Lab's ``metric(...)`` reads)."""
    t = comparison_type.upper()
    if t == "AE":
        return highlight_differences(first, second, "AE").score
    return compute(first, second, t)[0]


def _json_score(method: str, score: float) -> dict:
    d: dict = {"score": None if not math.isfinite(score) else round(float(score), 6)}
    if method == "psnr" and math.isinf(score):
        d["infinite"] = True  # identical images: PSNR = +inf (ImageToolbox shows Infinity)
    return d


def pixel_by_pixel(a: np.ndarray, b: np.ndarray, methods: list[str], *, threshold: float = 4.0,
                   highlight_color=(1.0, 0.0, 0.0, 1.0), highlight_method: str | None = None,
                   max_side: int = 4096) -> tuple[dict, np.ndarray | None]:
    """Compare-screen pixel-by-pixel mode on float images (both loaded at <= 4096 px like CompareComponent).

    Returns (JSON-serialisable result, highlighted float RGBA image or None).
    """
    from ..core.image import fit_within

    fa, _ = fit_within(a, max_side)
    fb, _ = fit_within(b, max_side)
    ra, rb = to_rgba8(fa), to_rgba8(fb)
    if rb.shape[:2] != ra.shape[:2]:
        rb = resize_and_pad(rb, ra.shape[1], ra.shape[0])
    hc = tuple(int(round(min(1.0, max(0.0, v)) * 255)) for v in (list(highlight_color) + [1.0])[:4])
    res: dict = {"resolution": [int(ra.shape[1]), int(ra.shape[0])], "threshold": float(threshold),
                 "resized_and_padded": bool(fb.shape[:2] != fa.shape[:2]), "methods": {}}
    hl = None
    for m in dict.fromkeys(methods + ([highlight_method] if highlight_method else [])):
        r = highlight_differences(ra, rb, m.upper(), hc, threshold)
        if m in methods:
            entry = _json_score(m, r.score)
            entry["highlighted_pixels"] = int(r.mask.sum())
            entry["highlighted_fraction"] = round(float(r.mask.mean()), 6)
            res["methods"][m] = entry
        if m == highlight_method:
            hl = r.highlighted.astype(np.float32) / 255.0
            if (r.highlighted[..., 3] == 255).all():
                hl = hl[..., :3]
    return res, hl
