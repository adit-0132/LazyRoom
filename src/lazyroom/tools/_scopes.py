"""ImageToolbox curves-screen scopes: waveform, vector scope and channel histogram preview.

Data side (exact port, vectorised with ``np.bincount``):

* ``ImageToolboxLibs/libs/histogram/src/main/cpp/histogram.cpp`` ``generateScope`` – waveform:
  R/G/B overlaid column bins (image x -> scope column, 8-bit level -> scope row) with per-channel
  ``log1p(count) / log1p(max)`` intensity; vector scope: BT.709 Cb/Cr bins (``(B-Y)/1.8556``,
  ``(R-Y)/1.5748``) with the bin's mean colour and ``log1p`` intensity.  Size clamp 64..512
  (``ImageScope.from``); the editor uses 256 (waveform) and 384 (vector scope).
* ``HistogramGenerator.generate`` – 256-bin R/G/B and 101-bin brightness (rounded CIE L*) histograms
  used by ``CurveChannelHistogram`` in ``ImageCurvesEditor.kt``.

Drawing side (``lib/curves/ImageScopeOverlay.kt`` + ``ImageCurvesEditor.kt``): density buckets
(6 levels, alpha 46..224), waveform colour masks (0.55 overlap threshold), 18 hue groups + neutral
for the vector scope with glow, graticule (radius 0.46, axes, six 60-degree reference marks), and the
filled/lined histogram with a 0.72 -> 0.08 vertical alpha gradient.  Theme colours are the
editor's base colours (dark theme, no blending with the app's primary colour).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.params import Bool, Choice, Color, File, Float, Int, Text
from .registry import read_image, save, tool

F32 = np.float32
ITB = "ImageToolbox"

WAVEFORM_RESOLUTION = 256
VECTORSCOPE_RESOLUTION = 384
DENSITY_LEVELS = 6
WAVEFORM_OVERLAP_THRESHOLD = 0.55
HUE_BUCKETS = 18
NEUTRAL_THRESHOLD = 16
MIN_POINT_ALPHA = 46
MAX_POINT_ALPHA = 224
VECTORSCOPE_RADIUS_FRACTION = 0.46

# ImageCurvesEditorDefaults base colours (before blending with the theme primary).
RED_CURVE = (0xED, 0x3D, 0x4C)       # Color(-0x12c2b4)
GREEN_CURVE = (0x10, 0xEE, 0x9D)     # Color(-0xef1163)
BLUE_CURVE = (0x33, 0x77, 0xFB)      # Color(-0xcc8805)
LUMA_CURVE = (0xE6, 0xE1, 0xE5)      # onSurface (Material 3 dark baseline)
HUE_COLORS = [(0xFF, 0x52, 0x52), (0xFF, 0xAB, 0x40), (0xFF, 0xE0, 0x33), (0x4C, 0xD9, 0x64), (0x38, 0xD6, 0xD2),
              (0x55, 0x96, 0xFF), (0xA6, 0x6B, 0xFF), (0xF5, 0x5C, 0xAA), (0xFF, 0x52, 0x52)]
GRID_ALPHA_NIGHT = 0.26


# =============================================================================
# Data
# =============================================================================

def _bitmap_bytes(img: np.ndarray) -> np.ndarray:
    """Float image -> the RGB bytes an Android ARGB_8888 bitmap exposes (premultiplied by alpha).
    An (H, W, 3) uint8 array is taken as already converted."""
    if img.dtype == np.uint8 and img.shape[-1] == 3:
        return img
    rgb = np.clip(img[..., :3], 0, 1)
    if img.shape[-1] == 4:
        rgb = rgb * np.clip(img[..., 3:4], 0, 1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def _clamp_size(size: int) -> int:
    return int(min(max(int(size), 64), 512))


def _scope_intensity(counts: np.ndarray, maximum: int) -> np.ndarray:
    """``scopeIntensity``: round(log1p(count) / log1p(max) * 255) (float maths), 0 for empty bins."""
    if maximum == 0:
        return np.zeros(counts.shape, np.uint8)
    norm = np.log1p(counts.astype(F32)) / np.log1p(F32(maximum))
    out = np.floor(norm * F32(255.0) + F32(0.5)).astype(np.int64)  # std::round (non-negative)
    out[counts == 0] = 0
    return out.astype(np.uint8)


def waveform_data(img: np.ndarray, size: int = WAVEFORM_RESOLUTION) -> dict[str, Any]:
    """``generateScope`` WAVEFORM: returns bins (counts) and the RGBA intensity buffer (S, S, 4)."""
    s = _clamp_size(size)
    px = _bitmap_bytes(img)
    h, w = px.shape[:2]
    cols = np.minimum(s - 1, (np.arange(w, dtype=np.int64) * s) // w)
    rows = (s - 1 - (np.arange(256, dtype=np.int64) * (s - 1)) // 255) * s
    counts, data = [], np.zeros((s * s, 4), np.uint8)
    for c in range(3):
        idx = rows[px[..., c].astype(np.int64)] + cols[None, :]
        cnt = np.bincount(idx.ravel(), minlength=s * s)
        counts.append(cnt)
        data[:, c] = _scope_intensity(cnt, int(cnt.max()))
    data[:, 3] = data[:, :3].max(1)
    return {"size": s, "counts": np.stack(counts).reshape(3, s, s), "data": data.reshape(s, s, 4),
            "pixels": int(h * w)}


def vectorscope_data(img: np.ndarray, size: int = VECTORSCOPE_RESOLUTION) -> dict[str, Any]:
    """``generateScope`` VECTORSCOPE: BT.709 Cb/Cr bins, mean colour per bin and log1p intensity."""
    s = _clamp_size(size)
    px = _bitmap_bytes(img).reshape(-1, 3)
    v = px.astype(F32) / F32(255.0)
    luma = F32(0.2126) * v[:, 0] + F32(0.7152) * v[:, 1] + F32(0.0722) * v[:, 2]
    cb = np.clip((v[:, 2] - luma) / F32(1.8556) + F32(0.5), 0, 1)
    cr = np.clip((v[:, 0] - luma) / F32(1.5748) + F32(0.5), 0, 1)
    sx = np.floor(cb * F32(s - 1) + F32(0.5)).astype(np.int64)
    sy = np.floor((F32(1.0) - cr) * F32(s - 1) + F32(0.5)).astype(np.int64)
    idx = sy * s + sx
    counts = np.bincount(idx, minlength=s * s)
    data = np.zeros((s * s, 4), np.uint8)
    nz = counts > 0
    for c in range(3):
        sums = np.bincount(idx, weights=px[:, c].astype(np.float64), minlength=s * s)
        mean = np.zeros(s * s, np.int64)
        mean[nz] = (sums[nz].astype(np.uint64) // counts[nz].astype(np.uint64)).astype(np.int64)
        data[:, c] = mean
    data[:, 3] = _scope_intensity(counts, int(counts.max()))
    return {"size": s, "counts": counts.reshape(s, s), "data": data.reshape(s, s, 4), "pixels": int(len(px))}


def _srgb_to_linear_hist(v: np.ndarray) -> np.ndarray:  # ColorUtils.cpp SRGBToLinear (threshold 0.045)
    return np.where(v <= 0.045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


_LIN_LUT = (_srgb_to_linear_hist(np.arange(256) / 255.0) * 100.0).astype(F32)


def histogram_data(img: np.ndarray) -> dict[str, np.ndarray]:
    """``HistogramGenerator.generate``: R/G/B (256 bins) and brightness = round(CIE L*) (101 bins)."""
    px = _bitmap_bytes(img).reshape(-1, 3)
    out = {name: np.bincount(px[:, c], minlength=256) for c, name in enumerate(("red", "green", "blue"))}
    lin = _LIN_LUT[px]
    y = lin @ np.array([0.2126729, 0.7151522, 0.0721750], F32)
    t = y.astype(np.float64) / 100.0
    t0 = 6.0 / 29.0
    fy = np.where(t > t0 ** 3, np.cbrt(t), (1.0 / 3.0) * (1 / t0) ** 2 * t + 4.0 / 29.0)
    lum = (116.0 * fy.astype(F32) - 16.0).astype(np.float64)
    li = np.clip(np.where(lum >= 0, np.floor(lum + 0.5), np.ceil(lum - 0.5)), 0, 100).astype(np.int64)
    out["brightness"] = np.bincount(li, minlength=101)[:101]
    return out


# =============================================================================
# Summaries (structured, JSON-serialisable)
# =============================================================================

def _percentile_levels(hist: np.ndarray, levels: np.ndarray, qs=(0.01, 0.5, 0.99)) -> list[float | None]:
    total = hist.sum()
    if total == 0:
        return [None for _ in qs]
    cdf = np.cumsum(hist) / total
    return [round(float(levels[min(int(np.searchsorted(cdf, q)), len(levels) - 1)]), 4) for q in qs]


def waveform(img: np.ndarray, *, size: int = WAVEFORM_RESOLUTION, column_groups: int = 8,
             include_data: bool = False, data: dict[str, Any] | None = None) -> dict[str, Any]:
    d = data if data is not None else waveform_data(img, size)
    s, counts = d["size"], d["counts"]
    levels = 1.0 - np.arange(s) / (s - 1)  # scope row -> level (row 0 = white)
    px = _bitmap_bytes(img)
    res: dict[str, Any] = {"type": "waveform", "size": s, "pixels": d["pixels"],
                           "max_bin_count": {n: int(counts[c].max()) for c, n in enumerate(("red", "green", "blue"))},
                           "clipping": {}, "columns": []}
    for c, n in enumerate(("red", "green", "blue")):
        ch = px[..., c]
        res["clipping"][n] = {"black": round(float((ch == 0).mean()), 6), "white": round(float((ch == 255).mean()), 6)}
    edges = np.linspace(0, s, column_groups + 1).round().astype(int)
    for g in range(column_groups):
        a, b = edges[g], edges[g + 1]
        col = {"scope_columns": [int(a), int(b - 1)], "image_x": [round(a / s, 4), round(b / s, 4)]}
        for c, n in enumerate(("red", "green", "blue")):
            p1, p50, p99 = _percentile_levels(counts[c][::-1, a:b].sum(1), levels[::-1])
            col[n] = {"p01": p1, "p50": p50, "p99": p99}
        res["columns"].append(col)
    if include_data:
        res["intensity"] = {n: d["data"][..., c].tolist() for c, n in enumerate(("red", "green", "blue"))}
        res["intensity_note"] = "row 0 = level 255 (top), column = image x; values 0..255 (log1p-scaled)"
    return res


def _vector_color_group(rgb: np.ndarray) -> np.ndarray:
    """``vectorColorGroup`` on integer bin colours -> 0..17 hue groups, 18 = neutral."""
    r, g, b = (rgb[..., i].astype(np.int64) for i in range(3))
    mx = np.maximum(r, np.maximum(g, b))
    mn = np.minimum(r, np.minimum(g, b))
    rng = mx - mn
    safe = np.maximum(rng, 1).astype(F32)
    hue = np.where(mx == r, (g - b).astype(F32) / safe,
                   np.where(mx == g, F32(2) + (b - r).astype(F32) / safe, F32(4) + (r - g).astype(F32) / safe))
    hue = hue / F32(6)
    hue = np.where(hue < 0, hue + F32(1), hue)
    grp = np.clip((hue * HUE_BUCKETS).astype(np.int64), 0, HUE_BUCKETS - 1)
    return np.where(rng < NEUTRAL_THRESHOLD, HUE_BUCKETS, grp)


def vectorscope(img: np.ndarray, *, size: int = VECTORSCOPE_RESOLUTION, include_data: bool = False,
                data: dict[str, Any] | None = None) -> dict[str, Any]:
    d = data if data is not None else vectorscope_data(img, size)
    s, counts, data = d["size"], d["counts"], d["data"]
    total = counts.sum()
    ys, xs = np.nonzero(counts)
    w = counts[ys, xs].astype(np.float64)
    cb = xs / (s - 1) - 0.5
    cr = 0.5 - ys / (s - 1)
    radius = np.hypot(cb, cr) / 0.5          # 1.0 = edge of the Cb/Cr square's inscribed circle
    angle = np.degrees(np.arctan2(cr, cb)) % 360.0
    groups = _vector_color_group(data[ys, xs, :3])
    hue_fracs = np.bincount(groups, weights=w, minlength=HUE_BUCKETS + 1) / max(total, 1)
    vx, vy = (w * cb).sum(), (w * cr).sum()
    order = np.argsort(radius)
    cdf = np.cumsum(w[order]) / max(w.sum(), 1)

    def rq(q):
        return round(float(radius[order][min(int(np.searchsorted(cdf, q)), len(order) - 1)]), 4) if len(order) else None

    res: dict[str, Any] = {
        "type": "vectorscope", "size": s, "pixels": d["pixels"], "max_bin_count": int(counts.max()),
        "occupied_bins": int(len(xs)),
        "mean_vector": {"cb": round(float(vx / max(total, 1)), 5), "cr": round(float(vy / max(total, 1)), 5),
                        "angle_deg": round(float(math.degrees(math.atan2(vy, vx)) % 360.0), 2),
                        "magnitude": round(float(math.hypot(vx, vy) / max(total, 1) / 0.5), 5)},
        "saturation": {"mean": round(float((w * radius).sum() / max(total, 1)), 5), "p50": rq(0.5), "p95": rq(0.95)},
        "hue_groups": [{"group": i, "hue_deg": [i * 360 // HUE_BUCKETS, (i + 1) * 360 // HUE_BUCKETS],
                        "fraction": round(float(hue_fracs[i]), 6)} for i in range(HUE_BUCKETS)],
        "neutral_fraction": round(float(hue_fracs[HUE_BUCKETS]), 6),
        "angle_note": "angle = atan2(Cr, Cb) in degrees (counter-clockwise from +Cb); pure red is ~103",
    }
    if len(xs):
        k = int(np.argmax(counts[ys, xs]))
        res["peak"] = {"x": int(xs[k]), "y": int(ys[k]), "angle_deg": round(float(angle[k]), 2),
                       "saturation": round(float(radius[k]), 4), "fraction": round(float(w[k] / total), 6)}
    if include_data:
        res["intensity"] = data[..., 3].tolist()
        res["mean_color"] = data[..., :3].tolist()
        res["intensity_note"] = "x = Cb (0..1 left->right), y = 1 - Cr (top = Cr 1); values 0..255 (log1p-scaled)"
    return res


def histogram(img: np.ndarray, *, include_data: bool = False,
              data: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    d = data if data is not None else histogram_data(img)
    res: dict[str, Any] = {"type": "histogram", "bins": {"red": 256, "green": 256, "blue": 256, "brightness": 101}}
    for n, hist in d.items():
        lv = np.arange(len(hist)) / (len(hist) - 1)
        p1, p50, p99 = _percentile_levels(hist, lv)
        total = max(int(hist.sum()), 1)
        res[n] = {"mean": round(float((hist * lv).sum() / total), 5), "p01": p1, "median": p50, "p99": p99,
                  "peak_bin": int(np.argmax(hist))}
        if include_data:
            res[n]["counts"] = hist.astype(int).tolist()
    res["brightness_note"] = "brightness = round(CIE L*) 0..100 (ImageToolbox Histogram.brightnessData)"
    return res


# =============================================================================
# Rendering (ImageScopeOverlay / CurveChannelHistogram)
# =============================================================================

def _bucket(density: np.ndarray) -> np.ndarray:
    return np.clip(((density.astype(np.int64) - 1) * DENSITY_LEVELS) // 255, 0, DENSITY_LEVELS - 1)


def _bucket_alpha(bucket: int) -> int:
    return MIN_POINT_ALPHA + bucket * (MAX_POINT_ALPHA - MIN_POINT_ALPHA) // (DENSITY_LEVELS - 1)


class _Canvas:
    """Premultiplied RGBA float canvas with anti-aliased round points composited SRC_OVER."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.rgb = np.zeros((h, w, 3), F32)
        self.a = np.zeros((h, w), F32)

    def layer(self, color: tuple[int, int, int], coverage: np.ndarray) -> None:
        c = np.array(color, F32) / 255.0
        self.rgb = c * coverage[..., None] + self.rgb * (1 - coverage[..., None])
        self.a = coverage + self.a * (1 - coverage)

    def points_coverage(self, pts: np.ndarray, width: float, alpha: float) -> np.ndarray:
        """Union coverage of N round points (stroke ``width``, ROUND cap) each with opacity ``alpha``."""
        acc = np.zeros(self.h * self.w, np.float64)
        if len(pts):
            r = width / 2.0
            k = int(math.ceil(r + 0.5))
            offs = np.arange(-k, k + 1)
            ox, oy = np.meshgrid(offs, offs)
            base_x = np.floor(pts[:, 0]).astype(np.int64)
            base_y = np.floor(pts[:, 1]).astype(np.int64)
            pxs = base_x[:, None] + ox.ravel()[None, :]
            pys = base_y[:, None] + oy.ravel()[None, :]
            d = np.hypot(pxs + 0.5 - pts[:, 0:1], pys + 0.5 - pts[:, 1:2])
            area = min(1.0, math.pi * r * r)  # tiny points cannot fully cover a pixel
            cov = np.clip(r + 0.5 - d, 0, 1) * area
            ok = (pxs >= 0) & (pxs < self.w) & (pys >= 0) & (pys < self.h) & (cov > 0)
            np.add.at(acc, (pys[ok] * self.w + pxs[ok]), np.log1p(-np.minimum(cov[ok] * alpha, 0.999999)))
        return (1 - np.exp(acc)).reshape(self.h, self.w).astype(F32)

    def finish(self, background: tuple[float, float, float, float]) -> np.ndarray:
        bg = np.array(background, F32)
        a_out = self.a + bg[3] * (1 - self.a)
        rgb = self.rgb + bg[:3] * bg[3] * (1 - self.a[..., None])
        with np.errstate(invalid="ignore", divide="ignore"):
            rgb = np.where(a_out[..., None] > 0, rgb / np.maximum(a_out[..., None], 1e-8), 0)
        if bg[3] >= 1:
            return np.clip(rgb, 0, 1).astype(F32)
        return np.dstack([np.clip(rgb, 0, 1), a_out]).astype(F32)


def render_waveform(d: dict[str, Any], width: int = 512, height: int = 512, density: float = 2.0,
                    background=(0.0, 0.0, 0.0, 1.0)) -> np.ndarray:
    s, data = d["size"], d["data"].reshape(-1, 4)
    r, g, b = (data[:, i].astype(np.int64) for i in range(3))
    dens = np.maximum(r, np.maximum(g, b))
    thr = dens * WAVEFORM_OVERLAP_THRESHOLD
    mask = (r >= thr) * 1 | (g >= thr) * 2 | (b >= thr) * 4
    mask = np.maximum(mask, 1)
    idx = np.arange(s * s)
    xs = (idx % s + 0.5) / s * width
    ys = (idx // s + 0.5) / s * height
    cols = [RED_CURVE, GREEN_CURVE, BLUE_CURVE]
    canvas = _Canvas(width, height)
    pw = max(density * 0.9, 1.0)
    bucket = _bucket(dens)
    for group in range(7 * DENSITY_LEVELS):
        m, bk = group // DENSITY_LEVELS + 1, group % DENSITY_LEVELS
        sel = (dens > 0) & (mask == m) & (bucket == bk)
        if not sel.any():
            continue
        color = tuple(max(cols[i][j] for i in range(3) if m & (1 << i)) for j in range(3))
        cov = canvas.points_coverage(np.stack([xs[sel], ys[sel]], 1), pw, _bucket_alpha(bk) / 255.0)
        canvas.layer(color, cov)
    return canvas.finish(background)


def _draw_guides(canvas: _Canvas, density: float) -> None:
    w, h = canvas.w, canvas.h
    m = np.zeros((h, w), np.uint8)
    sh = 4
    sc = 1 << sh
    cx, cy = w / 2.0, h / 2.0
    rad = min(w, h) * VECTORSCOPE_RADIUS_FRACTION
    th = max(1, int(round(0.75 * density)))

    def P(x, y):
        return (int(round(x * sc)), int(round(y * sc)))

    cv2.circle(m, P(cx, cy), int(round(rad * sc)), 255, th, cv2.LINE_AA, sh)
    cv2.line(m, P(cx - rad, cy), P(cx + rad, cy), 255, th, cv2.LINE_AA, sh)
    cv2.line(m, P(cx, cy - rad), P(cx, cy + rad), 255, th, cv2.LINE_AA, sh)
    for i in range(6):
        ang = math.pi * 2 * i / 6
        cv2.circle(m, P(cx + math.cos(ang) * rad, cy + math.sin(ang) * rad), int(round(2 * density * sc)), 255, th,
                   cv2.LINE_AA, sh)
    canvas.layer(LUMA_CURVE, m.astype(F32) / 255.0 * GRID_ALPHA_NIGHT)


def render_vectorscope(d: dict[str, Any], width: int = 512, height: int = 512, density: float = 2.0,
                       background=(0.0, 0.0, 0.0, 1.0), guides: bool = True) -> np.ndarray:
    s, data = d["size"], d["data"].reshape(-1, 4)
    dens = data[:, 3].astype(np.int64)
    groups = _vector_color_group(data[:, :3])
    idx = np.arange(s * s)
    diameter = min(width, height) * VECTORSCOPE_RADIUS_FRACTION * 2
    left, top = (width - diameter) / 2.0, (height - diameter) / 2.0
    xs = left + (idx % s) / (s - 1.0) * diameter
    ys = top + (idx // s) / (s - 1.0) * diameter
    canvas = _Canvas(width, height)
    pw = max(density * 0.9, 1.0)
    bucket = _bucket(dens)
    for group in range((HUE_BUCKETS + 1) * DENSITY_LEVELS):
        cg, bk = group // DENSITY_LEVELS, group % DENSITY_LEVELS
        sel = (dens > 0) & (groups == cg) & (bucket == bk)
        if not sel.any():
            continue
        color = LUMA_CURVE if cg == HUE_BUCKETS else HUE_COLORS[cg * len(HUE_COLORS) // HUE_BUCKETS]
        alpha = _bucket_alpha(bk)
        pts = np.stack([xs[sel], ys[sel]], 1)
        canvas.layer(color, canvas.points_coverage(pts, pw * 2.6, max(alpha // 4, 8) / 255.0))  # glow
        canvas.layer(color, canvas.points_coverage(pts, pw, alpha / 255.0))
    if guides:
        _draw_guides(canvas, density)
    return canvas.finish(background)


def render_histogram(hist: np.ndarray, color: tuple[int, int, int], width: int = 512, height: int = 512,
                     density: float = 2.0, background=(0.0, 0.0, 0.0, 1.0)) -> np.ndarray:
    """``CurveChannelHistogram``: normalised by the max bin, gradient fill (alpha 0.72 -> 0.08) + 1dp line."""
    hist = np.asarray(hist, np.float64)
    mx = hist.max() if len(hist) else 0
    vals = np.clip(hist / mx, 0, 1) if mx > 0 else np.zeros_like(hist)
    canvas = _Canvas(width, height)
    if len(vals) >= 2:
        sh, sc = 4, 16
        step = width / (len(vals) - 1)
        line = np.stack([np.arange(len(vals)) * step, height * (1 - vals)], 1)
        poly = np.concatenate([[[0, height]], line, [[width, height]]])
        fill = np.zeros((height, width), np.uint8)
        cv2.fillPoly(fill, [np.round(poly * sc).astype(np.int32)], 255, cv2.LINE_AA, sh)
        grad = (0.72 + (0.08 - 0.72) * (np.arange(height) + 0.5) / height).astype(F32)[:, None]
        canvas.layer(color, fill.astype(F32) / 255.0 * grad)
        lm = np.zeros((height, width), np.uint8)
        cv2.polylines(lm, [np.round(line * sc).astype(np.int32)], False, 255, max(1, int(round(density))),
                      cv2.LINE_AA, sh)
        canvas.layer(color, lm.astype(F32) / 255.0 * 0.9)
    return canvas.finish(background)


# =============================================================================
# Tool
# =============================================================================

_HIST_COLORS = {"luma": LUMA_CURVE, "red": RED_CURVE, "green": GREEN_CURVE, "blue": BLUE_CURVE}


@tool("image_scopes", "Waveform / vector scope / histogram", category="analysis", params=[
    File("input", "", file_kind="image", required=True),
    Choice("scope", "all", choices=["waveform", "vectorscope", "histogram", "all"]),
    Int("resolution", 0, min=0, max=512,
        description="Scope bin resolution; 0 = ImageToolbox default (waveform 256, vector scope 384); "
                    "other values are clamped to 64..512"),
    Choice("histogram_channel", "luma", choices=["luma", "red", "green", "blue"],
           description="Channel drawn in the histogram render (luma = 101-bin CIE L* brightness)"),
    Text("output_dir", "", description="Write <stem>_waveform.png / _vectorscope.png / _histogram.png here"),
    Text("waveform_output", "", description="Explicit output path for the waveform render"),
    Text("vectorscope_output", "", description="Explicit output path for the vector scope render"),
    Text("histogram_output", "", description="Explicit output path for the histogram render"),
    Int("render_width", 512, min=64, max=4096), Int("render_height", 512, min=64, max=4096),
    Float("density", 2.0, min=0.5, max=4.0, step=0.25,
          description="Display density used for point/line widths (Android dp -> px)"),
    Color("background", "#000000", description="Render background; '#00000000' = transparent PNG"),
    Bool("include_data", False, description="Include full bin/intensity arrays in the JSON (large)"),
    Bool("overwrite", False),
], source={"project": ITB, "feature": "Curves: ImageScopeOverlay + native generateScope / Histogram"})
def image_scopes(api, input, scope, resolution, histogram_channel, output_dir, waveform_output,
                 vectorscope_output, histogram_output, render_width, render_height, density, background,
                 include_data, overwrite):
    """ImageToolbox curves-screen scopes: RGB waveform, BT.709 Cb/Cr vector scope and channel histogram as
    structured data, with optional renders matching ImageToolbox's overlay drawing."""
    img, _ = read_image(api, input)
    h, w = img.shape[:2]
    img = _bitmap_bytes(img)  # convert once; every scope works on the 8-bit bitmap
    kinds = ["waveform", "vectorscope", "histogram"] if scope == "all" else [scope]
    explicit = {"waveform": waveform_output, "vectorscope": vectorscope_output, "histogram": histogram_output}
    od = None
    if output_dir:
        od = api.ws.policy.check_write_dir(output_dir)
        od.mkdir(parents=True, exist_ok=True)
    stem = Path(input).stem
    res: dict[str, Any] = {"input": input, "width": int(w), "height": int(h)}
    for k in kinds:
        path = explicit[k] or (str(od / f"{stem}_{k}.png") if od is not None else "")
        if k == "waveform":
            d = waveform_data(img, resolution or WAVEFORM_RESOLUTION)
            res[k] = waveform(img, include_data=include_data, data=d)
            if path:
                arr = render_waveform(d, render_width, render_height, density, background)
                res[k]["render"] = save(api, arr, path, overwrite=overwrite)
        elif k == "vectorscope":
            d = vectorscope_data(img, resolution or VECTORSCOPE_RESOLUTION)
            res[k] = vectorscope(img, include_data=include_data, data=d)
            if path:
                arr = render_vectorscope(d, render_width, render_height, density, background)
                res[k]["render"] = save(api, arr, path, overwrite=overwrite)
        else:
            hd = histogram_data(img)
            res[k] = histogram(img, include_data=include_data, data=hd)
            if path:
                hist = hd["brightness"] if histogram_channel == "luma" else hd[histogram_channel]
                arr = render_histogram(hist, _HIST_COLORS[histogram_channel], render_width, render_height, density,
                                       background)
                res[k]["render"] = save(api, arr, path, overwrite=overwrite)
                res[k]["render"]["channel"] = histogram_channel
    return res
