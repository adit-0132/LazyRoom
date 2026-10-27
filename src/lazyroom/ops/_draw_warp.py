"""ImageToolbox draw "Warp" brush (private): port of Trickle ``WarpEngine.cpp`` (Apache-2.0).

One engine per gesture: a backward displacement map ``(dx, dy)`` starts at zero, every
pointer segment is applied as brush "dabs" spaced ``radius / 2`` apart, and the result is
``out(x, y) = src(x + dx, y + dy)`` sampled bilinearly.  ``mix`` blends pixels towards the
colour under the brush centre in place instead of displacing them.

Per dab (pixel centres at integer coordinates, brush centre ``to``)::

    t       = 1 - dist / radius                          (dist <= radius)
    falloff = smoothstep(hardness, 1, t) * strength * amp   amp: grow/shrink 0.03, swirl 0.2, else 1
    move:   d += (from - to) * falloff
    grow:   d -= unit(p - to) * falloff * radius         (shrink: +=)
    swirl:  d += R(+-0.8 * t * falloff) (p - to) - (p - to)   (cw: +, ccw: -)
    mix:    pix = lerp(pix, colour(to), falloff)
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from numba import njit

F32 = np.float32
MODES = ("move", "grow", "shrink", "swirl_cw", "swirl_ccw", "mix")
MOVE, GROW, SHRINK, SWIRL_CW, SWIRL_CCW, MIX = range(6)


@njit(cache=True)
def _smoothstep(e0, e1, x):
    if e1 <= e0:
        return 1.0 if x >= e1 else 0.0
    t = (x - e0) / (e1 - e0)
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True)
def _sample(pix, fx, fy):
    """Bilinear sample with clamped coordinates (WarpEngine.sampleBilinear clamps the base pixel)."""
    H, W, C = pix.shape
    fx = min(max(fx, 0.0), W - 1.0)
    fy = min(max(fy, 0.0), H - 1.0)
    x0 = int(fx)
    y0 = int(fy)
    x1 = min(W - 1, x0 + 1)
    y1 = min(H - 1, y0 + 1)
    ax = fx - x0
    ay = fy - y0
    out = np.empty(C, np.float32)
    for c in range(C):
        top = pix[y0, x0, c] + (pix[y0, x1, c] - pix[y0, x0, c]) * ax
        bot = pix[y1, x0, c] + (pix[y1, x1, c] - pix[y1, x0, c]) * ax
        out[c] = top + (bot - top) * ay
    return out


@njit(cache=True)
def _dab(dx, dy, pix, ox, oy, fx, fy, tx, ty, r, hardness, strength, mode):
    """WarpEngine::applyStrokeToPoint.  ``dx``/``dy`` cover the box starting at (ox, oy); ``pix`` is the
    whole premultiplied image (modified in place for ``mix``)."""
    H, W = pix.shape[0], pix.shape[1]
    r2 = r * r
    if mode == MIX:
        center = _sample(pix, tx, ty)
    else:
        center = np.zeros(pix.shape[2], np.float32)
    min_x = max(0, int(tx - r))
    max_x = min(W - 1, int(tx + r))
    min_y = max(0, int(ty - r))
    max_y = min(H - 1, int(ty + r))
    if mode == GROW or mode == SHRINK:
        amp = 0.03
    elif mode == SWIRL_CW or mode == SWIRL_CCW:
        amp = 0.2
    else:
        amp = 1.0
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            ddx = x - tx
            ddy = y - ty
            dist2 = ddx * ddx + ddy * ddy
            if dist2 > r2:
                continue
            dist = math.sqrt(dist2)
            t = 1.0 - dist / r
            falloff = _smoothstep(hardness, 1.0, t) * (strength * amp)
            if falloff == 0.0:
                continue
            iy = y - oy
            ix = x - ox
            if mode == MOVE:
                dx[iy, ix] += (fx - tx) * falloff
                dy[iy, ix] += (fy - ty) * falloff
            elif mode == GROW or mode == SHRINK:
                if dist > 0.0:
                    sign = -1.0 if mode == GROW else 1.0
                    dx[iy, ix] += sign * (ddx / dist) * falloff * r
                    dy[iy, ix] += sign * (ddy / dist) * falloff * r
            elif mode == SWIRL_CW or mode == SWIRL_CCW:
                ang = t * 0.8 * falloff * (1.0 if mode == SWIRL_CW else -1.0)
                s = math.sin(ang)
                c = math.cos(ang)
                rx = ddx * c - ddy * s
                ry = ddx * s + ddy * c
                dx[iy, ix] += rx - ddx
                dy[iy, ix] += ry - ddy
            else:
                for k in range(pix.shape[2]):
                    pix[y, x, k] = pix[y, x, k] + (center[k] - pix[y, x, k]) * falloff


@njit(cache=True)
def _segment(dx, dy, pix, ox, oy, fx, fy, tx, ty, r, hardness, strength, mode):
    """WarpEngine::interpolateAndApplyStroke (including its loop bounds: sub-steps i = 0..steps-1,
    so the last sub-step of a segment is applied as the first one of the next segment)."""
    ddx = tx - fx
    ddy = ty - fy
    distance = math.sqrt(ddx * ddx + ddy * ddy)
    step = r / 2.0
    if distance <= step:
        _dab(dx, dy, pix, ox, oy, fx, fy, tx, ty, r, hardness, strength, mode)
        return
    steps = int(math.ceil(distance / step))
    sx = ddx / steps
    sy = ddy / steps
    for i in range(steps):
        a = max(i - 1, 0)
        _dab(dx, dy, pix, ox, oy, fx + sx * a, fy + sy * a, fx + sx * i, fy + sy * i, r, hardness, strength,
             mode)


def warp_gesture(pix: np.ndarray, points: np.ndarray, radius: float, strength: float, hardness: float,
                 mode: str) -> None:
    """Apply one gesture (engine coordinates: pixel centres at integers) to the premultiplied float32
    image ``pix`` in place."""
    H, W = pix.shape[:2]
    pts = np.asarray(points, np.float64)
    if radius <= 0 or len(pts) == 0:
        return
    m = MODES.index(mode)
    x0 = max(0, int(math.floor(pts[:, 0].min() - radius - 2)))
    y0 = max(0, int(math.floor(pts[:, 1].min() - radius - 2)))
    x1 = min(W, int(math.ceil(pts[:, 0].max() + radius + 2)))
    y1 = min(H, int(math.ceil(pts[:, 1].max() + radius + 2)))
    if x1 <= x0 or y1 <= y0:
        return
    dx = np.zeros((y1 - y0, x1 - x0), F32)
    dy = np.zeros_like(dx)
    if len(pts) == 1:  # a single dab (upstream needs a drag; see the port report)
        _dab(dx, dy, pix, x0, y0, pts[0, 0], pts[0, 1], pts[0, 0], pts[0, 1], radius, hardness, strength, m)
    else:
        for a, b in zip(pts[:-1], pts[1:]):
            _segment(dx, dy, pix, x0, y0, a[0], a[1], b[0], b[1], radius, hardness, strength, m)
        # pointer-up: a final stroke from the path end to the up position (= the last point)
        _segment(dx, dy, pix, x0, y0, pts[-1, 0], pts[-1, 1], pts[-1, 0], pts[-1, 1], radius, hardness,
                 strength, m)
    if m == MIX:
        return
    gx, gy = np.meshgrid(np.arange(x0, x1, dtype=F32), np.arange(y0, y1, dtype=F32))
    out = cv2.remap(pix, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    pix[y0:y1, x0:x1] = out.reshape(y1 - y0, x1 - x0, -1)
