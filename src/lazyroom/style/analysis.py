"""Photographic style analysis — measurable properties of an image's "look".

All measurements are computed on a downscaled copy (≤1024 px) in perceptual
spaces: OKLab for colour/lightness, linear light for white balance.  Values are
plain JSON so agents can reason about them, e.g. ``tone.p50`` (median
lightness), ``tone.black_point`` (faded blacks > ~0.08), ``color.cast`` (hue and
strength of the overall cast), ``zones.shadows.tint`` (split toning),
``texture.grain`` (noise sigma), ``vignette.strength``.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from ..core.color import (
    LUMA_709, lin_to_xyz, rgb_to_hsv, rgb_to_oklab, srgb_to_linear, oklab_to_rgb,
)
from ..core.image import fit_within
from ..core.params import color_to_hex

F32 = np.float32
QUANTILES = (0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995)
BAND_HUES = {"red": 0, "orange": 30, "yellow": 60, "green": 120, "aqua": 180, "blue": 225, "purple": 270, "magenta": 315}


def _r(x: float, n: int = 4) -> float:
    return round(float(x), n)


def quick_stats(img: np.ndarray) -> dict[str, Any]:
    small, _ = fit_within(img[..., :3], 512)
    x = np.clip(small, 0, 1)
    lab = rgb_to_oklab(x)
    L = lab[..., 0].ravel()
    C = np.hypot(lab[..., 1], lab[..., 2]).ravel()
    return {
        "mean_rgb": [_r(v) for v in x.reshape(-1, 3).mean(0)],
        "lightness": {"p05": _r(np.quantile(L, 0.05)), "median": _r(np.median(L)), "p95": _r(np.quantile(L, 0.95)),
                      "std": _r(L.std())},
        "chroma_mean": _r(C.mean()),
        "clipped_highlights_pct": _r((x.max(-1) >= 0.999).mean() * 100, 2),
        "clipped_shadows_pct": _r((x.max(-1) <= 0.002).mean() * 100, 2),
    }


def _cct_from_xy(x: float, y: float) -> float:
    n = (x - 0.3320) / (0.1858 - y)  # McCamy
    return 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33


def _hsv_hue_of_oklab_angle(angle: float) -> float:
    """Inverse of _util.hue_to_ab: HSV-style hue (deg) whose OKLab direction is ``angle`` (rad)."""
    return float(_HUE_TABLE[int(round(math.degrees(angle) % 360)) % 360])


def _build_hue_table() -> np.ndarray:
    from ..core.color import hsv_to_rgb

    h = np.arange(3600, dtype=F32) / 3600
    rgb = hsv_to_rgb(np.stack([h, np.ones_like(h), np.ones_like(h)], -1)[None])[0]
    lab = rgb_to_oklab(rgb)
    ang = (np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360)
    table = np.zeros(360, F32)
    for a in range(360):
        d = np.abs((ang - a + 180) % 360 - 180)
        table[a] = h[int(np.argmin(d))] * 360
    return table


_HUE_TABLE = _build_hue_table()


def zone_masks(L: np.ndarray) -> dict[str, np.ndarray]:
    sh = np.clip((0.45 - L) / 0.2, 0, 1)
    hi = np.clip((L - 0.62) / 0.2, 0, 1)
    mid = np.clip(1 - sh - hi, 0, 1)
    return {"shadows": sh, "midtones": mid, "highlights": hi}


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkær's fast noise variance estimate (robust to image content), sRGB 0..1 units."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], F32)
    conv = cv2.filter2D(gray.astype(F32), -1, k, borderType=cv2.BORDER_REFLECT101)
    # ignore strong edges so texture/detail does not count as grain
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    flat = mag < np.quantile(mag, 0.6)
    v = np.abs(conv[flat]) if flat.any() else np.abs(conv)
    h, w = gray.shape
    return float(math.sqrt(math.pi / 2) * v.mean() / 6.0)


def grain_size(gray: np.ndarray) -> float:
    """Correlation length (px) of the fine-scale residual — rough grain size estimate."""
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 3)
    hp = hp - hp.mean()
    f = np.fft.rfft2(hp)
    ac = np.fft.irfft2(f * np.conj(f), s=hp.shape)
    ac = np.fft.fftshift(ac)
    cy, cx = np.array(ac.shape) // 2
    prof = ac[cy, cx: cx + 8] / max(ac[cy, cx], 1e-9)
    below = np.where(prof < 0.5)[0]
    return float(below[0]) if len(below) else 8.0


def vignette_strength(L: np.ndarray) -> float:
    """Edge-vs-centre lightness ratio minus 1 (negative = darker corners), robust to content via medians."""
    h, w = L.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / math.sqrt(2)
    centre = np.median(L[r < 0.35]) if (r < 0.35).any() else np.median(L)
    corner = np.median(L[r > 0.85]) if (r > 0.85).any() else centre
    return float(corner / max(centre, 1e-3) - 1.0)


def palette(lab: np.ndarray, k: int = 6, seed: int = 0) -> list[dict[str, Any]]:
    pts = lab.reshape(-1, 3).astype(F32)
    if len(pts) > 40000:
        idx = np.random.default_rng(seed).choice(len(pts), 40000, replace=False)
        pts = pts[idx]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    cv2.setRNGSeed(seed)
    k = max(1, min(k, len(pts)))
    _, labels, centers = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k).astype(float)
    order = np.argsort(-counts)
    rgb = np.clip(oklab_to_rgb(centers[None].astype(F32))[0], 0, 1)
    return [{"hex": color_to_hex(rgb[i]), "weight": _r(counts[i] / counts.sum(), 3),
             "oklab": [_r(v) for v in centers[i]]} for i in order]


def analyze(img: np.ndarray, full: bool = False) -> dict[str, Any]:
    small, _ = fit_within(img[..., :3], 1024)
    x = np.clip(small, 0, 1).astype(F32)
    lab = rgb_to_oklab(x)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    H = np.arctan2(b, a)
    Lf = L.ravel()
    q = np.quantile(Lf, QUANTILES)
    tone = {f"p{str(k * 100).rstrip('0').rstrip('.').replace('.', '_')}": _r(v) for k, v in zip(QUANTILES, q)}
    tone.update({
        "mean": _r(Lf.mean()), "std": _r(Lf.std()), "median": _r(q[5]),
        "black_point": _r(q[0]), "white_point": _r(q[-1]),
        "dynamic_range": _r(q[-1] - q[0]),
        "clipped_shadows_pct": _r((x.max(-1) <= 0.002).mean() * 100, 2),
        "clipped_highlights_pct": _r((x.min(-1) >= 0.998).mean() * 100, 2),
    })
    # white balance estimate: gray-world on mid-lightness, low-chroma pixels (linear light)
    lin = srgb_to_linear(x)
    neutralish = (C < np.quantile(C, 0.5)) & (L > 0.2) & (L < 0.9)
    ref = lin[neutralish].mean(0) if neutralish.sum() > 50 else lin.reshape(-1, 3).mean(0)
    xyz = lin_to_xyz(ref[None, None])[0, 0]
    s = float(xyz.sum()) or 1.0
    cx, cy = xyz[0] / s, xyz[1] / s
    cct = _cct_from_xy(cx, cy)
    cast_a, cast_b = float(a.mean()), float(b.mean())
    cast_strength = math.hypot(cast_a, cast_b)
    color = {
        "chroma_mean": _r(C.mean()), "chroma_p50": _r(np.median(C)), "chroma_p90": _r(np.quantile(C, 0.9)),
        "colorfulness": _r(_colorfulness(x), 2),
        "cast": {"a": _r(cast_a), "b": _r(cast_b), "strength": _r(cast_strength),
                 "hue_deg": _r(_hsv_hue_of_oklab_angle(math.atan2(cast_b, cast_a)), 1)},
        "estimated_cct_k": int(round(max(1500, min(25000, cct)))),
        "warmth": _r(cast_b, 4),
        "tint_green_magenta": _r(cast_a, 4),
        "is_monochrome": bool(np.quantile(C, 0.95) < 0.02),
    }
    zones = {}
    for name, m in zone_masks(L).items():
        wsum = float(m.sum())
        if wsum < 1:
            zones[name] = {"weight": 0.0}
            continue
        za, zb = float((a * m).sum() / wsum), float((b * m).sum() / wsum)
        zones[name] = {"weight": _r(wsum / m.size, 3), "a": _r(za), "b": _r(zb),
                       "tint_hue_deg": _r(_hsv_hue_of_oklab_angle(math.atan2(zb, za)), 1),
                       "tint_strength": _r(math.hypot(za, zb)), "chroma": _r(float((C * m).sum() / wsum)),
                       "lightness": _r(float((L * m).sum() / wsum))}
    # hue distribution (chroma weighted) in 8 editor bands
    hsv = rgb_to_hsv(x)
    hue = hsv[..., 0] * 360
    sat_w = C * (C > 0.02)
    bands = {}
    total_w = float(sat_w.sum()) or 1.0
    for name, centre in BAND_HUES.items():
        d = np.abs((hue - centre + 180) % 360 - 180)
        wmask = (d < 25).astype(F32) * sat_w
        ws = float(wmask.sum())
        bands[name] = {"share": _r(ws / total_w, 3),
                       "chroma": _r(float((C * (d < 25)).sum() / max((d < 25).sum(), 1))),
                       "lightness": _r(float((L * (d < 25)).sum() / max((d < 25).sum(), 1)))}
    gray = (x @ LUMA_709).astype(F32)
    texture = {"grain_sigma": _r(noise_sigma(gray), 5), "sharpness": _r(float(cv2.Laplacian(gray, cv2.CV_32F).var()), 5)}
    vig = vignette_strength(L)
    out: dict[str, Any] = {
        "size": [int(img.shape[1]), int(img.shape[0])],
        "tone": tone, "color": color, "zones": zones, "hue_bands": bands, "texture": texture,
        "vignette": {"strength": _r(vig, 3)},
        "palette": palette(lab, 6),
    }
    out["descriptors"] = describe(out)
    if full:
        hist, _ = np.histogram(Lf, bins=32, range=(0, 1))
        out["tone"]["histogram_32"] = [int(v) for v in hist]
        texture["grain_size_px"] = _r(grain_size(gray), 2)
    return out


def _colorfulness(x: np.ndarray) -> float:
    r, g, b = x[..., 0] * 255, x[..., 1] * 255, x[..., 2] * 255
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean()))


def describe(an: dict[str, Any]) -> list[str]:
    """Human-readable style descriptors derived from the measurements."""
    t, c, z = an["tone"], an["color"], an["zones"]
    d: list[str] = []
    med = t["median"]
    d.append("low-key / dark" if med < 0.42 else "high-key / bright" if med > 0.75 else "mid-key")
    std = t["std"]
    d.append("high contrast" if std > 0.24 else "low / soft contrast" if std < 0.13 else "moderate contrast")
    if t["black_point"] > 0.18:
        d.append("faded / lifted blacks (matte)")
    if t["white_point"] < 0.85:
        d.append("muted highlights (soft roll-off)")
    if c["is_monochrome"]:
        d.append("monochrome")
    else:
        cm = c["chroma_mean"]
        d.append("vibrant / saturated" if cm > 0.11 else "muted / desaturated" if cm < 0.045 else "natural saturation")
        cast = c["cast"]
        if cast["strength"] > 0.012:
            hue = cast["hue_deg"]
            name = ("red" if hue < 15 or hue >= 345 else "orange" if hue < 45 else "yellow" if hue < 75 else
                    "green" if hue < 150 else "cyan" if hue < 195 else "blue" if hue < 255 else
                    "purple" if hue < 290 else "magenta")
            warm = name in ("red", "orange", "yellow")
            d.append(f"{'warm' if warm else 'cool' if name in ('cyan', 'blue') else name} cast ({name})")
        sh, hi = z.get("shadows", {}), z.get("highlights", {})
        if sh.get("tint_strength", 0) > 0.015 and hi.get("tint_strength", 0) > 0.015 and \
                abs(((sh["tint_hue_deg"] - hi["tint_hue_deg"]) + 180) % 360 - 180) > 60:
            d.append(f"split-toned (shadows hue {sh['tint_hue_deg']:.0f}°, highlights hue {hi['tint_hue_deg']:.0f}°)")
    if an["texture"]["grain_sigma"] > 0.015:
        d.append("visible grain / noise")
    if an["vignette"]["strength"] < -0.15:
        d.append("vignetted (dark corners)")
    return d
