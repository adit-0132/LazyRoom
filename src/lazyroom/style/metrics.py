"""Image comparison / style distance metrics."""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from ..core.color import rgb_to_oklab, rgb_to_lab
from ..core.image import fit_within, resize

F32 = np.float32


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM on luma (Wang et al. 2004, Gaussian window σ=1.5)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    g = lambda x: cv2.GaussianBlur(x, (0, 0), 1.5)
    mu_a, mu_b = g(a), g(b)
    va = g(a * a) - mu_a ** 2
    vb = g(b * b) - mu_b ** 2
    cov = g(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(s.mean())


def delta_e76(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((rgb_to_lab(a) - rgb_to_lab(b)) ** 2).sum(-1))


def sliced_wasserstein(a: np.ndarray, b: np.ndarray, n_proj: int = 48, n_samples: int = 20000, seed: int = 0) -> float:
    """Sliced Wasserstein-1 distance between the OKLab colour distributions of two images."""
    rng = np.random.default_rng(seed)
    pa = rgb_to_oklab(np.clip(a[..., :3], 0, 1)).reshape(-1, 3)
    pb = rgb_to_oklab(np.clip(b[..., :3], 0, 1)).reshape(-1, 3)
    if len(pa) > n_samples:
        pa = pa[rng.choice(len(pa), n_samples, replace=False)]
    if len(pb) > n_samples:
        pb = pb[rng.choice(len(pb), n_samples, replace=False)]
    dirs = rng.normal(size=(n_proj, 3))
    dirs[:, 0] *= 1.0  # lightness and chroma axes weighted equally
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    qs = np.linspace(0.005, 0.995, 100)
    total = 0.0
    for d in dirs:
        total += float(np.abs(np.quantile(pa @ d, qs) - np.quantile(pb @ d, qs)).mean())
    return total / n_proj


def compare_images(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Metrics between two images (resized to the same size): pixel & perceptual & distribution distances."""
    a3, _ = fit_within(a[..., :3], 768)
    b3 = resize(b[..., :3], a3.shape[1], a3.shape[0]) if b.shape[:2] != a.shape[:2] else fit_within(b[..., :3], 768)[0]
    if b3.shape[:2] != a3.shape[:2]:
        b3 = resize(b3, a3.shape[1], a3.shape[0])
    a3, b3 = np.clip(a3, 0, 1), np.clip(b3, 0, 1)
    de = delta_e76(a3, b3)
    ya = a3 @ np.array([0.2126, 0.7152, 0.0722], F32)
    yb = b3 @ np.array([0.2126, 0.7152, 0.0722], F32)
    mse = float(((a3 - b3) ** 2).mean())
    return {
        "mean_delta_e": round(float(de.mean()), 3),
        "p95_delta_e": round(float(np.quantile(de, 0.95)), 3),
        "ssim_luma": round(ssim_gray(ya, yb), 4),
        "psnr_db": round(10 * math.log10(1.0 / mse), 2) if mse > 1e-12 else float("inf"),
        "color_distribution_distance": round(sliced_wasserstein(a3, b3), 5),
        "identical": bool(mse < 1e-10),
    }
