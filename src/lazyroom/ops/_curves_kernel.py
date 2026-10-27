"""Fused numba kernel of ``adjust.color_curves`` (ImageToolbox GPUImageColorCurvesFilter chain).

Module-level jitted functions so numba's on-disk cache (``NUMBA_CACHE_DIR``, set by ``core.fast``)
works across processes; imported lazily by :mod:`lazyroom.ops.curves_ext`.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import fast as _fast  # noqa: F401  (sets NUMBA_CACHE_DIR before numba is imported)
import numba as nb  # noqa: E402

@nb.njit(cache=True, inline="always")
def tex(lut, u):
    pos = u * 256.0 - 0.5
    if pos <= 0.0:
        return lut[0]
    if pos >= 255.0:
        return lut[255]
    i0 = int(math.floor(pos))
    f = pos - i0
    return lut[i0] + (lut[i0 + 1] - lut[i0]) * f

@nb.njit(cache=True, inline="always")
def clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

@nb.njit(cache=True, inline="always")
def fract(v):
    return v - math.floor(v)

@nb.njit(cache=True, inline="always")
def hsv2rgb_c(h, s, v, k):
    p = abs(fract(h + k) * 6.0 - 3.0)
    c = clamp01(p - 1.0)
    return v * (1.0 + (c - 1.0) * s)

@nb.njit(cache=True, inline="always")
def lab_fwd(v):
    return v ** (1.0 / 3.0) if v > 0.008856 else 7.787 * v + 16.0 / 116.0

@nb.njit(cache=True, inline="always")
def lab_inv(v):
    c = v * v * v
    return c if c > 0.008856 else (v - 16.0 / 116.0) / 7.787

@nb.njit(cache=True, inline="always")
def to_lin(v):
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

@nb.njit(cache=True, inline="always")
def to_srgb(v):
    if v <= 0.0031308:
        return v * 12.92
    return 1.055 * max(v, 0.0) ** (1.0 / 2.4) - 0.055

@nb.njit(cache=True, inline="always")
def to_srgb_w(v):
    s = max(v, 0.0)
    return s * 12.92 if s < 0.0031308 else 1.055 * s ** (1.0 / 2.4) - 0.055

@nb.njit(cache=True, inline="always")
def cbrt_s(v):
    return math.copysign(abs(v) ** (1.0 / 3.0), v)

@nb.njit(cache=True, parallel=True)
def kern(px, do_rgb, tone, do_cmyk, cmyk, do_lab, lab, rels, rel_luts, do_wheels, bal, sharp):
    n = px.shape[0]
    out = np.empty_like(px)
    for i in nb.prange(n):
        r = clamp01(px[i, 0])
        g = clamp01(px[i, 1])
        b = clamp01(px[i, 2])
        if do_rgb:
            r = tex(tone[0], r)
            g = tex(tone[1], g)
            b = tex(tone[2], b)
        if do_cmyk:
            k = 1.0 - max(r, g, b)
            den = max(1.0 - k, 0.00001)
            c_ = tex(cmyk[0], (1.0 - r - k) / den)
            m_ = tex(cmyk[1], (1.0 - g - k) / den)
            y_ = tex(cmyk[2], (1.0 - b - k) / den)
            k2 = tex(cmyk[3], k)
            r = clamp01((1.0 - c_) * (1.0 - k2))
            g = clamp01((1.0 - m_) * (1.0 - k2))
            b = clamp01((1.0 - y_) * (1.0 - k2))
        if do_lab:
            lr, lg, lb = to_lin(r), to_lin(g), to_lin(b)
            x = 0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb
            yy = 0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb
            z = 0.0193339 * lr + 0.1191920 * lg + 0.9503041 * lb
            fx, fy, fz = lab_fwd(x / 0.95047), lab_fwd(yy), lab_fwd(z / 1.08883)
            lv = tex(lab[0], clamp01((116.0 * fy - 16.0) / 100.0)) * 100.0
            av = tex(lab[1], clamp01((500.0 * (fx - fy) + 128.0) / 255.0)) * 255.0 - 128.0
            bv = tex(lab[2], clamp01((200.0 * (fy - fz) + 128.0) / 255.0)) * 255.0 - 128.0
            fy = (lv + 16.0) / 116.0
            fx = fy + av / 500.0
            fz = fy - bv / 200.0
            x, yy, z = 0.95047 * lab_inv(fx), lab_inv(fy), 1.08883 * lab_inv(fz)
            r = clamp01(to_srgb(3.2404542 * x - 1.5371385 * yy - 0.4985314 * z))
            g = clamp01(to_srgb(-0.9692660 * x + 1.8760108 * yy + 0.0415560 * z))
            b = clamp01(to_srgb(0.0556434 * x - 0.2040259 * yy + 1.0572252 * z))
        if rels.shape[0] > 0:
            # rgbToHsv (GPUImageRgbToHsvFilter), clamped
            if g >= b:
                fx_, fy_, fz_, fw_ = g, b, 0.0, -1.0 / 3.0
            else:
                fx_, fy_, fz_, fw_ = b, g, -1.0, 2.0 / 3.0
            if r >= fx_:
                sx, sy, sz, sw = r, fy_, fz_, fx_
            else:
                sx, sy, sz, sw = fx_, fy_, fw_, r
            d = sx - min(sw, sy)
            h = clamp01(abs(sz + (sw - sy) / (6.0 * d + 1e-7)))
            s = clamp01(d / (sx + 1e-7))
            v = clamp01(sx)
            for j in range(rels.shape[0]):
                code = rels[j]
                inp = h if code <= 2 else (v if code <= 4 else s)
                cv = clamp01(tex(rel_luts[j], clamp01(inp)))
                delta = cv - 0.5
                if code == 0 or code == 3 or code == 5:
                    sd = delta * (2.0 / 3.0) if delta >= 0.0 else delta * 2.0
                    s = clamp01(s + sd)
                elif code == 2:
                    rr = hsv2rgb_c(h, s, v, 0.0)
                    gg = hsv2rgb_c(h, s, v, 2.0 / 3.0)
                    bb = hsv2rgb_c(h, s, v, 1.0 / 3.0)
                    luma = 0.2126 * rr + 0.7152 * gg + 0.0722 * bb
                    chroma = s * v
                    t = clamp01((chroma - 0.04) / (0.30 - 0.04))
                    wgt = t * t * (3.0 - 2.0 * t)
                    target = clamp01(luma + delta * 2.0 * wgt)
                    if target < luma:
                        sc = target / max(luma, 0.00001)
                        rr, gg, bb = rr * sc, gg * sc, bb * sc
                    else:
                        tt = (target - luma) / max(1.0 - luma, 0.00001)
                        rr, gg, bb = rr + (1.0 - rr) * tt, gg + (1.0 - gg) * tt, bb + (1.0 - bb) * tt
                    rr, gg, bb = clamp01(rr), clamp01(gg), clamp01(bb)
                    if gg >= bb:
                        fx_, fy_, fz_, fw_ = gg, bb, 0.0, -1.0 / 3.0
                    else:
                        fx_, fy_, fz_, fw_ = bb, gg, -1.0, 2.0 / 3.0
                    if rr >= fx_:
                        sx, sy, sz, sw = rr, fy_, fz_, fx_
                    else:
                        sx, sy, sz, sw = fx_, fy_, fw_, rr
                    d = sx - min(sw, sy)
                    h = clamp01(abs(sz + (sw - sy) / (6.0 * d + 1e-7)))
                    s = clamp01(d / (sx + 1e-7))
                    v = clamp01(sx)
                else:
                    h = clamp01((h + delta + 1.0) % 1.0)
            r = hsv2rgb_c(h, s, v, 0.0)
            g = hsv2rgb_c(h, s, v, 2.0 / 3.0)
            b = hsv2rgb_c(h, s, v, 1.0 / 3.0)
        if do_wheels:
            lr = ((r + 0.055) / 1.055) ** 2.4 if r >= 0.04045 else r / 12.92
            lg = ((g + 0.055) / 1.055) ** 2.4 if g >= 0.04045 else g / 12.92
            lb = ((b + 0.055) / 1.055) ** 2.4 if b >= 0.04045 else b / 12.92
            l1 = cbrt_s(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb)
            m1 = cbrt_s(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb)
            s1 = cbrt_s(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb)
            L = 0.2104542553 * l1 + 0.7936177850 * m1 - 0.0040720468 * s1
            A = 1.9779984951 * l1 - 2.4285922050 * m1 + 0.4505937099 * s1
            B = 0.0259040371 * l1 + 0.7827717662 * m1 - 0.8086757660 * s1
            tl = 0.2126 * r + 0.7152 * g + 0.0722 * b
            d0, d1, d2 = tl, abs(tl - 0.5), 1.0 - tl
            w0 = math.exp(-sharp * d0 * d0)
            w1 = math.exp(-sharp * d1 * d1)
            w2 = math.exp(-sharp * d2 * d2)
            ws = max(w0 + w1 + w2, 0.0001)
            w0, w1, w2 = w0 / ws, w1 / ws, w2 / ws
            A += (bal[0, 0] * w0 + bal[1, 0] * w1 + bal[2, 0] * w2) * 0.115
            B += (bal[0, 1] * w0 + bal[1, 1] * w1 + bal[2, 1] * w2) * 0.115
            l2 = L + 0.3963377774 * A + 0.2158037573 * B
            m2 = L - 0.1055613458 * A - 0.0638541728 * B
            s2 = L - 0.0894841775 * A - 1.2914855480 * B
            l2, m2, s2 = l2 * l2 * l2, m2 * m2 * m2, s2 * s2 * s2
            r = clamp01(to_srgb_w(4.0767416621 * l2 - 3.3077115913 * m2 + 0.2309699292 * s2))
            g = clamp01(to_srgb_w(-1.2684380046 * l2 + 2.6097574011 * m2 - 0.3413193965 * s2))
            b = clamp01(to_srgb_w(-0.0041960863 * l2 - 0.7034186147 * m2 + 1.7076147010 * s2))
        out[i, 0] = r
        out[i, 1] = g
        out[i, 2] = b
    return out
