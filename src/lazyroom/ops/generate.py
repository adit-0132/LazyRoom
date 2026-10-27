"""Procedural content composited into the image (category ``generate``)."""
from __future__ import annotations

import numpy as np

from ..core.blend import BLEND_MODES, layer_over
from ..core.params import Choice, Float, Gradient, Int, Point
from ..core.registry import op
from ._util import F32, luma, value_noise


def eval_gradient(stops, t: np.ndarray) -> np.ndarray:
    pos = np.array([p for p, _ in stops], F32)
    cols = np.array([c for _, c in stops], F32)
    t = np.clip(t, 0, 1)
    out = np.empty(t.shape + (4,), F32)
    for c in range(4):
        out[..., c] = np.interp(t, pos, cols[:, c])
    return out


@op("generate.gradient", "Gradient overlay", params=[
    Gradient("gradient", [[0, "#FF7E5F"], [1, "#FEB47B00"]]),
    Choice("kind", "linear", choices=["linear", "radial", "angular", "diamond", "reflected"]),
    Point("start", (0.5, 0.0), min=-1, max=2), Point("end", (0.5, 1.0), min=-1, max=2),
    Float("opacity", 0.5, min=0, max=1, step=0.01),
    Choice("blend", "soft_light", choices=list(BLEND_MODES)),
], alpha="full", tags=["gradient", "overlay", "sky", "color wash", "graduated filter", "tint"])
def gradient(img, ctx, gradient, kind, start, end, opacity, blend):
    """Blend a colour gradient over the image (graduated filter, colour wash)."""
    h, w = img.shape[:2]
    X, Y = np.meshgrid((np.arange(w, dtype=F32) + 0.5) / w, (np.arange(h, dtype=F32) + 0.5) / h)
    dx, dy = (end[0] - start[0]) * w, (end[1] - start[1]) * h
    L = max(np.hypot(dx, dy), 1e-6)
    px, py = (X - start[0]) * w, (Y - start[1]) * h
    if kind == "linear":
        t = (px * dx + py * dy) / (L * L)
    elif kind == "reflected":
        t = np.abs((px * dx + py * dy) / (L * L))
    elif kind == "radial":
        t = np.hypot(px, py) / L
    elif kind == "diamond":
        t = (np.abs(px) + np.abs(py)) / L
    else:
        t = ((np.arctan2(py, px) - np.arctan2(dy, dx)) / (2 * np.pi)) % 1.0
    layer = eval_gradient(gradient, t)
    return layer_over(img, layer, blend, opacity)


@op("generate.gradient_map", "Gradient map", params=[
    Gradient("gradient", [[0, "#1B1035"], [0.5, "#C2577A"], [1, "#FFE8B0"]]),
    Float("opacity", 1.0, min=0, max=1, step=0.01),
    Choice("blend", "normal", choices=list(BLEND_MODES)),
], tags=["gradient map", "duotone", "toning", "false color", "tritone"])
def gradient_map(img, ctx, gradient, opacity, blend):
    """Map luminance to a colour gradient (duotone / tritone)."""
    layer = eval_gradient(gradient, luma(np.clip(img, 0, 1)))
    layer[..., 3] = 1.0
    return layer_over(img, layer, blend, opacity)


@op("generate.texture_overlay", "Procedural texture overlay", params=[
    Choice("texture", "paper", choices=["paper", "dust", "clouds", "canvas"]),
    Float("scale", 1.0, min=0.1, max=10, step=0.05),
    Float("opacity", 0.3, min=0, max=1, step=0.01),
    Choice("blend", "overlay", choices=list(BLEND_MODES)),
    Int("seed", 0, min=0, max=1_000_000),
], tags=["texture", "paper", "dust", "scratches", "vintage", "overlay", "film"])
def texture_overlay(img, ctx, texture, scale, opacity, blend, seed):
    """Overlay a procedural texture (paper fibres, dust specks, clouds, canvas weave)."""
    h, w = img.shape[:2]
    rng = ctx.rng(f"tex{seed}")
    base = min(h, w) * scale
    if texture == "clouds":
        t = value_noise(h, w, base / 4, rng, octaves=5)
    elif texture == "paper":
        t = 0.5 + 0.35 * (value_noise(h, w, base / 60, rng, octaves=4) - 0.5) + \
            0.15 * (value_noise(h, w, max(1, base / 400), rng, octaves=2) - 0.5)
    elif texture == "canvas":
        X, Y = np.meshgrid(np.arange(w, dtype=F32), np.arange(h, dtype=F32))
        p = max(2.0, base / 150)
        t = 0.5 + 0.25 * np.sin(X / p * np.pi) * np.sin(Y / p * np.pi) + 0.1 * (value_noise(h, w, p, rng) - 0.5)
    else:  # dust
        t = np.full((h, w), 0.5, F32)
        n = int(h * w / 4000 * scale) + 1
        ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
        vals = rng.choice([0.0, 1.0], n)
        t[ys, xs] = vals
        import cv2

        t = cv2.GaussianBlur(t, (0, 0), max(0.5, min(h, w) / 1500))
        t = 0.5 + (t - 0.5) * 3
    layer = np.dstack([np.repeat(np.clip(t, 0, 1)[..., None], 3, axis=2), np.ones((h, w), F32)])
    return layer_over(img, layer, blend, opacity)
