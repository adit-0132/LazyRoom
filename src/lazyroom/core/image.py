"""Pipeline image buffer conventions and helpers.

The working buffer is a ``numpy.ndarray`` of dtype ``float32`` with shape
``(H, W, 3)`` (RGB) or ``(H, W, 4)`` (RGBA, *straight* / un-premultiplied
alpha).  Colour values are sRGB-encoded in nominal range 0..1; intermediate
results may exceed that range (no clipping between steps) and are clipped only
when encoding for display or export.  Operations that need linear light
convert internally (see :mod:`lazyroom.core.color`).
"""
from __future__ import annotations

import numpy as np
import cv2

F32 = np.float32


def to_float(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(F32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(F32) / 65535.0
    if img.dtype in (np.float16, np.float64):
        return img.astype(F32)
    if img.dtype == F32:
        return img
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        return img.astype(F32) / float(info.max)
    if img.dtype == np.bool_:
        return img.astype(F32)
    raise TypeError(f"Unsupported image dtype {img.dtype}")


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_uint16(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def normalize_channels(img: np.ndarray) -> np.ndarray:
    """Return a float32 (H, W, 3|4) buffer from any 2D/3D input."""
    img = to_float(img)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 2:  # LA
        img = np.concatenate([np.repeat(img[..., :1], 3, axis=2), img[..., 1:2]], axis=2)
    elif img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported image shape {img.shape}")
    return np.ascontiguousarray(img, dtype=F32)


def has_alpha(img: np.ndarray) -> bool:
    return img.ndim == 3 and img.shape[2] == 4


def split_alpha(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if has_alpha(img):
        return img[..., :3], img[..., 3]
    return img, None


def merge_alpha(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    rgb = rgb[..., :3]
    if alpha is None:
        return np.ascontiguousarray(rgb, dtype=F32)
    if alpha.shape[:2] != rgb.shape[:2]:
        alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(np.dstack([rgb, alpha]), dtype=F32)


def ensure_rgba(img: np.ndarray) -> np.ndarray:
    if has_alpha(img):
        return img
    return np.dstack([img, np.ones(img.shape[:2], F32)])


def opaque_if_full(img: np.ndarray) -> np.ndarray:
    """Drop an alpha channel that is fully opaque."""
    if has_alpha(img) and float(img[..., 3].min()) >= 0.9999:
        return np.ascontiguousarray(img[..., :3])
    return img


def size_of(img: np.ndarray) -> tuple[int, int]:
    """(width, height)"""
    return int(img.shape[1]), int(img.shape[0])


def resize(img: np.ndarray, width: int, height: int, interpolation: int | None = None) -> np.ndarray:
    width, height = max(1, int(width)), max(1, int(height))
    if (width, height) == size_of(img):
        return img
    if interpolation is None:
        interpolation = cv2.INTER_AREA if width < img.shape[1] else cv2.INTER_CUBIC
    out = cv2.resize(img, (width, height), interpolation=interpolation)
    if out.ndim == 2:
        out = out[..., None]
    return out.astype(F32, copy=False)


def fit_within(img: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Downscale so the long edge is <= max_dim. Returns (image, scale)."""
    w, h = size_of(img)
    long_edge = max(w, h)
    if max_dim <= 0 or long_edge <= max_dim:
        return img, 1.0
    s = max_dim / long_edge
    return resize(img, round(w * s), round(h * s), cv2.INTER_AREA), s


def composite_on(img: np.ndarray, background: tuple[float, float, float] | None = None,
                 checker: int = 0) -> np.ndarray:
    """Flatten RGBA over a solid colour or a checkerboard (for display)."""
    if not has_alpha(img):
        return img[..., :3]
    h, w = img.shape[:2]
    if checker > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        board = (((yy // checker) + (xx // checker)) % 2).astype(F32)
        bg = (0.8 + 0.15 * board)[..., None].repeat(3, axis=2)
    else:
        bg = np.empty((h, w, 3), F32)
        bg[:] = background if background is not None else (1.0, 1.0, 1.0)
    a = np.clip(img[..., 3:4], 0, 1)
    return (img[..., :3] * a + bg * (1 - a)).astype(F32)


def premultiply(img: np.ndarray) -> np.ndarray:
    if not has_alpha(img):
        return img
    a = img[..., 3:4]
    return np.dstack([img[..., :3] * a, a]).astype(F32)


def unpremultiply(img: np.ndarray) -> np.ndarray:
    if not has_alpha(img):
        return img
    a = img[..., 3:4]
    rgb = np.where(a > 1e-6, img[..., :3] / np.maximum(a, 1e-6), 0)
    return np.dstack([rgb, a]).astype(F32)


def gray(img: np.ndarray) -> np.ndarray:
    """Rec.709 luma of encoded values, shape (H, W)."""
    return (img[..., :3] @ np.array([0.2126, 0.7152, 0.0722], F32)).astype(F32)


def stats(img: np.ndarray) -> dict:
    rgb = img[..., :3]
    return {
        "mean": [round(float(v), 5) for v in rgb.reshape(-1, 3).mean(0)],
        "min": round(float(rgb.min()), 5),
        "max": round(float(rgb.max()), 5),
    }
