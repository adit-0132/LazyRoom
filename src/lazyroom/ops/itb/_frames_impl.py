"""Private helpers for the ImageToolbox "Frames" group ports (Android Canvas emulation)."""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from .._util import F32
from ._geometry_impl import JavaRandom, ensure_rgba, finish, has_alpha, kotlin_hash, kround, over


def border_frame(img: np.ndarray, horizontal: int, vertical: int, color: Sequence[float]) -> np.ndarray:
    """BorderFrameFilter: canvas filled with ``color``, image drawn centred (source-over)."""
    had = has_alpha(img)
    h, w = img.shape[:2]
    col = np.asarray(color, F32)[:4]
    if not had and col[3] >= 1.0:
        canvas = np.empty((h + 2 * vertical, w + 2 * horizontal, 3), F32)
        canvas[...] = col[:3]
        canvas[vertical:vertical + h, horizontal:horizontal + w] = img[..., :3]
        return canvas
    canvas = np.empty((h + 2 * vertical, w + 2 * horizontal, 4), F32)
    canvas[...] = col
    region = canvas[vertical:vertical + h, horizontal:horizontal + w]
    canvas[vertical:vertical + h, horizontal:horizontal + w] = over(region, ensure_rgba(img))
    return finish(canvas, had)


def drop_shadow(img: np.ndarray, blur_radius: float, offset_x: float, offset_y: float,
                color: Sequence[float]) -> np.ndarray:
    """DropShadowFilter: blurred, tinted alpha (Skia BlurMaskFilter NORMAL) under the image on a padded canvas."""
    rgba = ensure_rgba(img)
    h, w = rgba.shape[:2]
    r = max(blur_radius, 0.0)
    pl = math.ceil(max(r - offset_x, 0.0))
    pt = math.ceil(max(r - offset_y, 0.0))
    pr = math.ceil(max(r + offset_x, 0.0))
    pb = math.ceil(max(r + offset_y, 0.0))
    oh, ow = h + pt + pb, w + pl + pr
    # shadow mask placed at padding + offset (non-filtered bitmap draw -> integer position)
    sx, sy = pl + kround(offset_x), pt + kround(offset_y)
    sigma = 0.57735 * r + 0.5 if r > 0 else 0.0
    margin = int(math.ceil(3 * sigma)) if sigma > 0 else 0
    big = np.zeros((oh + 2 * margin, ow + 2 * margin), F32)
    y0, x0 = sy + margin, sx + margin
    ys0, xs0 = max(y0, 0), max(x0, 0)
    ys1, xs1 = min(y0 + h, big.shape[0]), min(x0 + w, big.shape[1])
    if ys1 > ys0 and xs1 > xs0:
        big[ys0:ys1, xs0:xs1] = rgba[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0, 3]
    if sigma > 0:
        big = cv2.GaussianBlur(big, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_CONSTANT)
    mask = big[margin:margin + oh, margin:margin + ow]
    col = np.asarray(color, F32)
    out = np.zeros((oh, ow, 4), F32)
    out[..., :3] = col[:3]
    out[..., 3] = mask * col[3]
    region = out[pt:pt + h, pl:pl + w]
    out[pt:pt + h, pl:pl + w] = over(region, rgba)
    return out


def side_fade(img: np.ndarray, side: str, scale: float) -> np.ndarray:
    """SideFadeFilter (Relative): DST_IN with a transparent->opaque black linear gradient."""
    rgba = ensure_rgba(img).copy()
    h, w = rgba.shape[:2]
    horizontal = side in ("start", "end")
    length = kround((w if horizontal else h) * scale)
    if length <= 0:  # degenerate gradient: Skia clamps to the last (opaque) colour
        return img
    n = w if horizontal else h
    pos = np.arange(n, dtype=np.float64) + 0.5
    if side in ("start", "top"):
        t = pos / length
    else:
        t = (n - pos) / length
    t = np.clip(t, 0.0, 1.0).astype(F32)
    if horizontal:
        rgba[..., 3] *= t[None, :]
    else:
        rgba[..., 3] *= t[:, None]
    return rgba


def torn_paths(width: int, height: int, tooth_height: int, h_range: int, v_range: int, top: bool, right: bool,
               bottom: bool, left: bool, hash_fields: Sequence) -> list[np.ndarray]:
    """TornEdgeFilter.createTornPaths with java.util.Random seeded like upstream."""
    rnd = JavaRandom((31 * width + height) * 31 + kotlin_hash(*hash_fields))
    th = min(max(tooth_height, 1), max(min(width, height), 1))
    h_range = max(h_range, 2)
    v_range = max(v_range, 2)
    hr = max(math.ceil(np.float32(width) / np.float32(h_range)), 2)
    vr = max(math.ceil(np.float32(height) / np.float32(v_range)), 2)

    def tooth() -> float:
        return float(rnd.next_int(th) + 1)

    paths = []
    W, H = float(width), float(height)
    if top:
        p = [(0.0, 0.0), (W, 0.0), (W, tooth())]
        for i in range(hr - 1, 0, -1):
            p.append((width * i / float(hr), tooth()))
        p.append((0.0, tooth()))
        paths.append(np.asarray(p))
    if right:
        p = [(W, 0.0), (W, H), (W - tooth(), H)]
        for i in range(vr - 1, 0, -1):
            p.append((W - tooth(), height * i / float(vr)))
        p.append((W - tooth(), 0.0))
        paths.append(np.asarray(p))
    if bottom:
        p = [(W, H), (0.0, H), (0.0, H - tooth())]
        for i in range(1, hr):
            p.append((width * i / float(hr), H - tooth()))
        p.append((W, H - tooth()))
        paths.append(np.asarray(p))
    if left:
        p = [(0.0, H), (0.0, 0.0), (tooth(), 0.0)]
        for i in range(1, vr):
            p.append((tooth(), height * i / float(vr)))
        p.append((tooth(), H))
        paths.append(np.asarray(p))
    return paths


def clear_paths(img: np.ndarray, paths: list[np.ndarray], ss: int = 4) -> np.ndarray:
    """Anti-aliased PorterDuff.CLEAR of each closed path (alpha *= 1 - coverage).

    Coverage is computed by ``ss`` x ``ss`` supersampling inside the path's bounding box.
    """
    rgba = ensure_rgba(img).copy()
    h, w = rgba.shape[:2]
    shift = 4
    for p in paths:
        x0 = max(int(np.floor(p[:, 0].min())), 0)
        y0 = max(int(np.floor(p[:, 1].min())), 0)
        x1 = min(int(np.ceil(p[:, 0].max())), w)
        y1 = min(int(np.ceil(p[:, 1].max())), h)
        if x1 <= x0 or y1 <= y0:
            continue
        hb, wb = y1 - y0, x1 - x0
        cov = np.zeros((hb * ss, wb * ss), np.uint8)
        pts = np.round(((p - (x0, y0)) * ss - 0.5) * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(cov, [pts], 1, lineType=cv2.LINE_8, shift=shift)
        frac = cov.reshape(hb, ss, wb, ss).mean(axis=(1, 3), dtype=np.float32)
        rgba[y0:y1, x0:x1, 3] *= 1.0 - frac
    return rgba
