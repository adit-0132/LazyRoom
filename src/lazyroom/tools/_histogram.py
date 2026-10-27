"""ImageToolbox histogram (ImageToolboxLibs ``libs/histogram``, Apache-2.0, T8RIN).

* ``itb_histogram``: ``HistogramGenerator.generate`` (native histogram.cpp): 256-bin R, G, B counts of
  the 8-bit (premultiplied) bitmap and a 101-bin "Brightness" histogram of ``round(L*)`` where L* comes
  from the library's own ``colorToLAB`` (sRGB decode with a 0.045 threshold - sic - , D65 XYZ, CIE Lab).
  The chart loads the image with Coil ``.size(1024)``: we fit it within 1024 px (INTER_AREA).
* ``render_histogram``: raster rendering of ``ImageHistogram``/``LineChart``: RGB, Brightness and
  Camera types, cubic "curved edges" path (control points at the x mid-point), vertical gradient fill
  (top alpha 0.8 -> bottom 0.4; Camera: opaque pure R/G/B drawn with BlendMode.Lighten), 8 x 4 dashed grid,
  colours blended 25 % with the theme's harmonisation colour (optional here).
"""
from __future__ import annotations

import numpy as np

from ..core.image import fit_within
from ._image_diff import to_rgba8

F32 = np.float32


def _srgb_to_linear_itb(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def lab_l_round(rgb8: np.ndarray) -> np.ndarray:
    """``round(colorToLAB(RGB(r, g, b)).l)`` from histogram.cpp/ColorUtils.cpp (float32 maths)."""
    lin = _srgb_to_linear_itb(np.arange(256, dtype=np.float64) / 255.0).astype(F32) * F32(100.0)
    y = (F32(0.2126729) * lin[rgb8[..., 0]] + F32(0.7151522) * lin[rgb8[..., 1]]
         + F32(0.0721750) * lin[rgb8[..., 2]]).astype(F32)
    t = (y / F32(100.0)).astype(np.float64)
    t0 = 6.0 / 29.0
    fy = np.where(t > t0 ** 3, np.cbrt(t), (1.0 / 3.0) * (1 / t0) ** 2 * t + 4.0 / 29.0).astype(F32)
    L = (F32(116.0) * fy - F32(16.0)).astype(F32)
    return np.clip(np.floor(L.astype(np.float64) + 0.5), 0, 100).astype(np.int64)  # C++ round() for L >= 0


def itb_histogram(img: np.ndarray, max_side: int = 1024) -> dict[str, list[int]]:
    x, _ = fit_within(img, max_side)
    u = to_rgba8(x)
    r, g, b = u[..., 0].ravel(), u[..., 1].ravel(), u[..., 2].ravel()
    L = lab_l_round(u[..., :3]).ravel()
    return {"red": np.bincount(r, minlength=256).tolist(), "green": np.bincount(g, minlength=256).tolist(),
            "blue": np.bincount(b, minlength=256).tolist(), "brightness": np.bincount(L, minlength=101).tolist(),
            "width": int(u.shape[1]), "height": int(u.shape[0])}


def _blend(c1, c2, frac: float):
    from ._color_models import blend_argb

    return blend_argb(c1, c2, frac)


def _line_y(values: np.ndarray, maxv: float, h: int, w: int, ss: int) -> np.ndarray:
    """Top y (in supersampled px) of the curved-edges path for every supersampled column."""
    n = len(values)
    W = w * ss
    Hs = h * ss
    ys = Hs - (values / maxv * Hs if maxv > 0 else np.zeros_like(values))
    step = W / (n - 1)
    xs = np.arange(W, dtype=np.float64) + 0.5
    seg = np.clip((xs // step).astype(int), 0, n - 2)
    # cubicTo(cx, y0, cx, y1, x1, y1) with cx the x mid-point -> x(t) = x0 + step*(3t^2 - 2t^3) ... solve t
    u = (xs - seg * step) / step  # 0..1 position along the segment; invert the Bezier x(t) by bisection
    lo, hi = np.zeros_like(u), np.ones_like(u)
    for _ in range(22):
        t = (lo + hi) / 2
        bx = 3 * (1 - t) ** 2 * t * 0.5 + 3 * (1 - t) * t * t * 0.5 + t ** 3
        less = bx < u
        lo = np.where(less, t, lo)
        hi = np.where(less, hi, t)
    t = (lo + hi) / 2
    y0, y1 = ys[seg], ys[seg + 1]
    by = (1 - t) ** 3 * y0 + 3 * (1 - t) ** 2 * t * y0 + 3 * (1 - t) * t * t * y1 + t ** 3 * y1
    return by


def render_histogram(data: dict, kind: str = "rgb", width: int = 1000, height: int = 500,
                     background=(0x1B / 255, 0x1B / 255, 0x1F / 255, 1.0), harmonization=None,
                     line_px: float = 1.0, border=(0x93 / 255, 0x8F / 255, 0x99 / 255)) -> np.ndarray:
    """Render the ImageHistogram chart (float RGB(A) image)."""
    import cv2

    ss = 2
    W, H = width * ss, height * ss
    canvas = np.empty((H, W, 3), np.float64)
    canvas[:] = np.array(background[:3], np.float64)

    def harm(argb, frac=0.25):
        if harmonization is None:
            return argb
        return _blend(argb, harmonization, frac)

    camera = kind == "camera"
    if kind == "brightness":
        lines = [(np.array(data["brightness"], np.float64), harm((255, 255, 255, 255), 0.1))]
    else:
        cols = ([(255, 255, 0, 0), (255, 0, 255, 0), (255, 0, 0, 255)] if camera else
                [harm((255, 0xE5, 0x15, 0x15)), harm((255, 0x16, 0xD7, 0x2F)), harm((255, 0x00, 0x67, 0xFF))])
        lines = [(np.array(data[k], np.float64), c) for k, c in zip(("red", "green", "blue"), cols)]
    maxv = max(float(v.max()) for v, _ in lines) or 1.0
    alpha_bottom = 1.0 if camera else 0.4
    alpha_top = 1.0 if camera else 0.8
    # grid (drawn first, dashed 1:1 px, outline colour at 50 % alpha)
    gcol = np.array(border, np.float64)
    for i in range(9):
        x = min(W - 1, int(round(W * i / 8)))
        mask = (np.arange(H) // ss) % 2 == 0
        canvas[mask, x] = canvas[mask, x] * 0.5 + gcol * 0.5
    for i in range(5):
        y = min(H - 1, int(round(H * i / 4)))
        mask = (np.arange(W) // ss) % 2 == 0
        canvas[y, mask] = canvas[y, mask] * 0.5 + gcol * 0.5
    yy = np.arange(H, dtype=np.float64)[:, None] + 0.5
    grad_t = yy / H
    for values, argb in lines:
        col = np.array(argb[1:], np.float64) / 255.0
        top = _line_y(values, maxv, height, width, ss)[None, :]
        inside = yy >= top
        a = (alpha_top + (alpha_bottom - alpha_top) * grad_t) * inside
        if camera:  # BlendMode.Lighten: max(src, dst) with full alpha
            canvas = np.where(inside[..., None], np.maximum(canvas, col), canvas)
        else:
            canvas = canvas * (1 - a[..., None]) + col * a[..., None]
        # stroke along the path
        pts = np.stack([np.arange(W) + 0.5, top[0]], 1).astype(np.int32).reshape(-1, 1, 2)
        layer = np.zeros((H, W), np.uint8)
        cv2.polylines(layer, [pts], False, 255, max(1, int(round(line_px * ss))), cv2.LINE_AA)
        la = (layer.astype(np.float64) / 255.0)[..., None]
        canvas = np.maximum(canvas, col * la) if camera else canvas * (1 - la) + col * la
    # border
    bw = max(1, int(round(line_px * ss)))
    canvas[:bw], canvas[-bw:], canvas[:, :bw], canvas[:, -bw:] = gcol, gcol, gcol, gcol
    out = cv2.resize(canvas.astype(F32), (width, height), interpolation=cv2.INTER_AREA)
    return np.clip(out, 0, 1)
