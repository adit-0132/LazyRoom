"""Layer blend modes (operate on sRGB-encoded 0..1 values, like common editors)."""
from __future__ import annotations

import numpy as np

from .color import hsl_to_rgb, rgb_to_hsl, luma

F32 = np.float32


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _soft_light(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # W3C formula
    d = np.where(a <= 0.25, ((16 * a - 12) * a + 4) * a, np.sqrt(np.maximum(a, 0)))
    return np.where(b <= 0.5, a - (1 - 2 * b) * a * (1 - a), a + (2 * b - 1) * (d - a))


def _set_lum(c: np.ndarray, l: np.ndarray) -> np.ndarray:
    d = l - luma(c)
    c = c + d[..., None]
    lum = luma(c)[..., None]
    n = c.min(axis=-1, keepdims=True)
    x = c.max(axis=-1, keepdims=True)
    c = np.where(n < 0, lum + (c - lum) * lum / np.maximum(lum - n, 1e-6), c)
    c = np.where(x > 1, lum + (c - lum) * (1 - lum) / np.maximum(x - lum, 1e-6), c)
    return c


BLEND_MODES = (
    "normal", "multiply", "screen", "overlay", "soft_light", "hard_light", "darken", "lighten",
    "color_dodge", "color_burn", "linear_dodge", "linear_burn", "linear_light", "vivid_light",
    "pin_light", "hard_mix", "difference", "exclusion", "subtract", "divide",
    "hue", "saturation", "color", "luminosity",
)


def blend_rgb(base: np.ndarray, top: np.ndarray, mode: str) -> np.ndarray:
    a, b = base[..., :3], top[..., :3]
    if mode == "normal":
        return b
    if mode == "multiply":
        return a * b
    if mode == "screen":
        return 1 - (1 - a) * (1 - b)
    if mode == "overlay":
        return np.where(a <= 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    if mode == "hard_light":
        return np.where(b <= 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    if mode == "soft_light":
        return _soft_light(_clip(a), _clip(b))
    if mode == "darken":
        return np.minimum(a, b)
    if mode == "lighten":
        return np.maximum(a, b)
    if mode == "color_dodge":
        return np.where(b >= 1, 1, np.minimum(1, a / np.maximum(1 - b, 1e-6)))
    if mode == "color_burn":
        return np.where(b <= 0, 0, 1 - np.minimum(1, (1 - a) / np.maximum(b, 1e-6)))
    if mode == "linear_dodge":
        return a + b
    if mode == "linear_burn":
        return a + b - 1
    if mode == "linear_light":
        return a + 2 * b - 1
    if mode == "vivid_light":
        burn = 1 - np.minimum(1, (1 - a) / np.maximum(2 * b, 1e-6))
        dodge = np.minimum(1, a / np.maximum(2 * (1 - b), 1e-6))
        return np.where(b <= 0.5, burn, dodge)
    if mode == "pin_light":
        return np.where(b <= 0.5, np.minimum(a, 2 * b), np.maximum(a, 2 * b - 1))
    if mode == "hard_mix":
        return (a + b >= 1).astype(F32)
    if mode == "difference":
        return np.abs(a - b)
    if mode == "exclusion":
        return a + b - 2 * a * b
    if mode == "subtract":
        return a - b
    if mode == "divide":
        return a / np.maximum(b, 1e-6)
    if mode in ("hue", "saturation", "color"):
        ha, hb = rgb_to_hsl(_clip(a)), rgb_to_hsl(_clip(b))
        if mode == "hue":
            out = np.stack([hb[..., 0], ha[..., 1], ha[..., 2]], -1)
        elif mode == "saturation":
            out = np.stack([ha[..., 0], hb[..., 1], ha[..., 2]], -1)
        else:
            out = _set_lum(b, luma(a))
            return out
        return _set_lum(hsl_to_rgb(out), luma(a))
    if mode == "luminosity":
        return _set_lum(a, luma(b))
    raise ValueError(f"Unknown blend mode {mode}")


def mix_step(base: np.ndarray, result: np.ndarray, mode: str = "normal", opacity: float = 1.0,
             mask: np.ndarray | None = None) -> np.ndarray:
    """Mix an operation's ``result`` back into its input ``base`` (adjustment-layer semantics).

    weight = opacity * mask; colour uses the blend mode, alpha is interpolated.
    """
    if result.shape[:2] != base.shape[:2]:
        raise ValueError("mix_step: size mismatch")
    if mode == "normal" and mask is None and opacity >= 1.0:
        return result
    w = np.full(base.shape[:2], float(opacity), F32) if mask is None else (mask.astype(F32) * float(opacity))
    blended = blend_rgb(base, result, mode)
    rgb = base[..., :3] + (blended - base[..., :3]) * w[..., None]
    if base.shape[2] == 4 or result.shape[2] == 4:
        a_base = base[..., 3] if base.shape[2] == 4 else np.ones(base.shape[:2], F32)
        a_res = result[..., 3] if result.shape[2] == 4 else np.ones(base.shape[:2], F32)
        alpha = a_base + (a_res - a_base) * w
        return np.dstack([rgb, np.clip(alpha, 0, 1)]).astype(F32)
    return rgb.astype(F32)


def layer_over(base: np.ndarray, layer: np.ndarray, mode: str = "normal", opacity: float = 1.0,
               mask: np.ndarray | None = None) -> np.ndarray:
    """Composite a layer with its own alpha over ``base`` (source-over with blend mode)."""
    if layer.shape[:2] != base.shape[:2]:
        raise ValueError("layer_over: size mismatch")
    la = layer[..., 3] if layer.shape[2] == 4 else np.ones(layer.shape[:2], F32)
    w = la * float(opacity)
    if mask is not None:
        w = w * mask
    blended = blend_rgb(base, layer, mode)
    rgb = base[..., :3] + (blended - base[..., :3]) * w[..., None]
    if base.shape[2] == 4:
        ab = base[..., 3]
        # blended colour where base is transparent should be the layer's own colour
        rgb = np.where(ab[..., None] > 1e-6,
                       (base[..., :3] * ab[..., None] * (1 - w[..., None]) + blended * w[..., None] * ab[..., None]
                        + layer[..., :3] * w[..., None] * (1 - ab[..., None])) / np.maximum(
                           ab + w - ab * w, 1e-6)[..., None],
                       layer[..., :3])
        alpha = ab + w - ab * w
        return np.dstack([rgb, alpha]).astype(F32)
    return rgb.astype(F32)
