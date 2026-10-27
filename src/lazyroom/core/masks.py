"""Resolution-independent masks / selections for local adjustments.

A mask spec is plain JSON so agents can author and inspect it::

    {"shapes": [
        {"type": "radial", "center": [0.5, 0.45], "radius": [0.3, 0.4], "feather": 0.5},
        {"type": "luminance", "min": 0.0, "max": 0.35, "softness": 0.1, "mode": "intersect"}
     ],
     "invert": false, "feather": 0.0, "density": 1.0}

Coordinates are normalised to the image (x: 0..1 of width, y: 0..1 of height);
lengths (``feather``, brush ``radius``, polygon ``feather``) are fractions of the
image's shorter side.  Shapes combine sequentially with ``mode`` =
add | subtract | intersect | replace (default add).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .color import rgb_to_oklab, luminance as rel_luminance, rgb_to_ycbcr
from .context import RenderContext
from .errors import invalid

F32 = np.float32

SHAPE_TYPES = {
    "all": "Whole image (useful as a base before subtracting)",
    "rect": "Rectangle: x, y, w, h (normalised), rotation (deg), roundness 0..1, feather",
    "ellipse": "Ellipse: center [x,y], radius [rx,ry] (normalised), rotation, feather (0..1 of radius)",
    "radial": "Alias of ellipse with soft falloff (feather defaults to 0.5)",
    "linear": "Linear gradient from start [x,y] (0) to end [x,y] (1)",
    "polygon": "Polygon: points [[x,y],...], feather",
    "brush": "Brush strokes: strokes [{points, radius, hardness, opacity, erase}]",
    "luminance": "Luminance range: min, max (0..1), softness",
    "color": "Colour range: color '#RRGGBB', tolerance (0..1), softness",
    "hue": "Hue range: hue (deg), width (deg), softness (deg), min_saturation",
    "skin": "Approximate skin-tone detection (YCbCr), softness",
    "image": "Grayscale mask image file: path, channel (luma|alpha|r|g|b)",
    "subject": "AI subject / salient-object mask (requires a segmentation model)",
}
MODES = ("add", "subtract", "intersect", "replace")


def _num(d: dict, k: str, default: float) -> float:
    v = d.get(k, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        raise invalid(f"Mask field '{k}' must be a number, got {v!r}") from None


def _pt(d: dict, k: str, default: tuple[float, float]) -> tuple[float, float]:
    v = d.get(k, default)
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise invalid(f"Mask field '{k}' must be [x, y]")
    return float(v[0]), float(v[1])


def validate_mask(spec: Any) -> dict[str, Any] | None:
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise invalid("Mask must be an object with a 'shapes' list")
    shapes = spec.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        raise invalid("Mask needs a non-empty 'shapes' list", hint=f"Shape types: {', '.join(SHAPE_TYPES)}")
    for s in shapes:
        if not isinstance(s, dict) or s.get("type") not in SHAPE_TYPES:
            raise invalid(f"Unknown mask shape {s!r}", valid_types=list(SHAPE_TYPES))
        if s.get("mode", "add") not in MODES:
            raise invalid(f"Unknown mask mode {s.get('mode')!r}", valid_modes=list(MODES))
    unknown = set(spec) - {"shapes", "invert", "feather", "density"}
    if unknown:
        raise invalid(f"Unknown mask fields {sorted(unknown)}")
    return spec


def _grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    ys = (np.arange(h, dtype=F32) + 0.5) / h
    xs = (np.arange(w, dtype=F32) + 0.5) / w
    return np.meshgrid(xs, ys)


def _feather(mask: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return mask
    h, w = mask.shape
    sigma = max(0.5, amount * min(h, w) / 2.0)
    return cv2.GaussianBlur(mask, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)


def _smooth_range(x: np.ndarray, lo: float, hi: float, soft: float) -> np.ndarray:
    soft = max(soft, 1e-4)
    a = np.clip((x - (lo - soft)) / soft, 0, 1)
    b = np.clip(((hi + soft) - x) / soft, 0, 1)
    m = np.minimum(a, b)
    return (m * m * (3 - 2 * m)).astype(F32)


def _shape(s: dict[str, Any], img: np.ndarray, ctx: RenderContext) -> np.ndarray:
    h, w = img.shape[:2]
    t = s["type"]
    if t == "all":
        return np.ones((h, w), F32)
    if t in ("ellipse", "radial"):
        cx, cy = _pt(s, "center", (0.5, 0.5))
        rx, ry = _pt(s, "radius", (0.3, 0.3))
        rot = np.deg2rad(_num(s, "rotation", 0.0))
        fea = np.clip(_num(s, "feather", 0.5 if t == "radial" else 0.1), 0, 1)
        X, Y = _grid(h, w)
        dx, dy = (X - cx) * w, (Y - cy) * h
        c, sn = np.cos(rot), np.sin(rot)
        u, v = dx * c + dy * sn, -dx * sn + dy * c
        d = np.sqrt((u / max(rx * w, 1e-6)) ** 2 + (v / max(ry * h, 1e-6)) ** 2)
        inner = 1.0 - fea
        m = np.clip((1.0 - d) / max(1.0 - inner, 1e-4), 0, 1) if fea > 0 else (d <= 1).astype(F32)
        return (m * m * (3 - 2 * m)).astype(F32) if fea > 0 else m
    if t == "rect":
        x, y = _num(s, "x", 0.25), _num(s, "y", 0.25)
        rw, rh = _num(s, "w", 0.5), _num(s, "h", 0.5)
        rot = _num(s, "rotation", 0.0)
        rnd = np.clip(_num(s, "roundness", 0.0), 0, 1)
        m = np.zeros((h, w), np.uint8)
        cx, cy = (x + rw / 2) * w, (y + rh / 2) * h
        if rnd <= 0:
            box = cv2.boxPoints(((cx, cy), (rw * w, rh * h), rot))
            cv2.fillPoly(m, [np.round(box * 16).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=4)
        else:
            r = rnd * min(rw * w, rh * h) / 2
            X, Y = np.meshgrid(np.arange(w, dtype=F32) + 0.5, np.arange(h, dtype=F32) + 0.5)
            th = np.deg2rad(rot)
            u = (X - cx) * np.cos(th) + (Y - cy) * np.sin(th)
            v = -(X - cx) * np.sin(th) + (Y - cy) * np.cos(th)
            qx = np.abs(u) - (rw * w / 2 - r)
            qy = np.abs(v) - (rh * h / 2 - r)
            sd = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
            return _feather(np.clip(0.5 - sd, 0, 1).astype(F32), _num(s, "feather", 0.0))
        return _feather(m.astype(F32) / 255.0, _num(s, "feather", 0.0))
    if t == "linear":
        x0, y0 = _pt(s, "start", (0.5, 0.0))
        x1, y1 = _pt(s, "end", (0.5, 1.0))
        X, Y = _grid(h, w)
        dx, dy = (x1 - x0) * w, (y1 - y0) * h
        L2 = max(dx * dx + dy * dy, 1e-6)
        tt = ((X - x0) * w * dx + (Y - y0) * h * dy) / L2
        m = np.clip(tt, 0, 1)
        return (m * m * (3 - 2 * m)).astype(F32)
    if t == "polygon":
        pts = s.get("points") or []
        if len(pts) < 3:
            raise invalid("polygon mask needs at least 3 points")
        arr = np.array([[float(p[0]) * w, float(p[1]) * h] for p in pts], dtype=np.float64)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(arr * 16).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=4)
        return _feather(m.astype(F32) / 255.0, _num(s, "feather", 0.0))
    if t == "brush":
        m = np.zeros((h, w), F32)
        short = min(h, w)
        for st in s.get("strokes") or []:
            pts = st.get("points") or []
            if not pts:
                continue
            radius = max(0.5, float(st.get("radius", 0.03)) * short)
            hard = float(np.clip(st.get("hardness", 0.5), 0, 1))
            opa = float(np.clip(st.get("opacity", 1.0), 0, 1))
            layer = np.zeros((h, w), np.uint8)
            arr = np.round(np.array([[float(p[0]) * w, float(p[1]) * h] for p in pts]) * 16).astype(np.int32)
            th = max(1, int(round(radius * 2 * (0.5 + 0.5 * hard))))
            if len(arr) == 1:
                cv2.circle(layer, tuple(arr[0]), int(round(radius * 16)), 255, -1, cv2.LINE_AA, shift=4)
            else:
                cv2.polylines(layer, [arr], False, 255, th, cv2.LINE_AA, shift=4)
                for p in (arr[0], arr[-1]):
                    cv2.circle(layer, tuple(p), int(round(th / 2 * 16)), 255, -1, cv2.LINE_AA, shift=4)
            lf = layer.astype(F32) / 255.0
            if hard < 1:
                lf = cv2.GaussianBlur(lf, (0, 0), max(0.5, radius * (1 - hard) * 0.6))
            lf *= opa
            m = m * (1 - lf) if st.get("erase") else np.maximum(m, lf)
        return m
    if t == "luminance":
        Y = rel_luminance(np.clip(img[..., :3], 0, 1)) ** (1 / 2.2)  # perceptual-ish lightness
        return _smooth_range(Y, _num(s, "min", 0.0), _num(s, "max", 1.0), _num(s, "softness", 0.1))
    if t == "color":
        from .params import parse_color

        target = np.array(parse_color(s.get("color", "#808080"))[:3], F32).reshape(1, 1, 3)
        lab = rgb_to_oklab(np.clip(img[..., :3], 0, 1))
        tl = rgb_to_oklab(target)[0, 0]
        d = np.sqrt(((lab - tl) ** 2).sum(-1)) / 0.4  # ~0.4 spans the sRGB gamut
        tol = _num(s, "tolerance", 0.15)
        soft = max(_num(s, "softness", 0.1), 1e-4)
        return np.clip(1 - (d - tol) / soft, 0, 1).astype(F32)
    if t == "hue":
        from .color import rgb_to_hsv

        hsv = rgb_to_hsv(np.clip(img[..., :3], 0, 1))
        hue = hsv[..., 0] * 360
        c = _num(s, "hue", 0.0) % 360
        width = _num(s, "width", 30.0)
        soft = max(_num(s, "softness", 20.0), 1e-3)
        dist = np.abs((hue - c + 180) % 360 - 180)
        m = np.clip(1 - (dist - width / 2) / soft, 0, 1)
        m *= np.clip((hsv[..., 1] - _num(s, "min_saturation", 0.08)) / 0.08, 0, 1)
        return m.astype(F32)
    if t == "skin":
        ycc = rgb_to_ycbcr(np.clip(img[..., :3], 0, 1)) * 255
        cb, cr = ycc[..., 1], ycc[..., 2]
        # elliptical skin cluster in CbCr (Hsu et al. inspired)
        d = ((cb - 110) / 22) ** 2 + ((cr - 152) / 18) ** 2
        soft = _num(s, "softness", 0.5)
        m = np.clip((1.6 + soft - d) / (soft + 1e-3), 0, 1)
        m = cv2.GaussianBlur(m.astype(F32), (0, 0), max(1.0, min(h, w) / 300))
        return m.astype(F32)
    if t == "image":
        from .io import load_image
        from .image import resize, gray

        p = ctx.resolve_file(str(s.get("path", "")), "mask image")
        mi, _ = load_image(p)
        ch = s.get("channel", "luma")
        if ch == "alpha" and mi.shape[2] == 4:
            m = mi[..., 3]
        elif ch in ("r", "g", "b"):
            m = mi[..., "rgb".index(ch)]
        else:
            m = gray(mi)
        return resize(m[..., None], w, h)[..., 0]
    if t == "subject":
        from ..ai.segmentation import subject_mask  # optional

        return subject_mask(img, ctx, model=s.get("model"))
    raise invalid(f"Unknown mask type {t}")


def render_mask(spec: dict[str, Any] | None, img: np.ndarray, ctx: RenderContext) -> np.ndarray | None:
    if not spec:
        return None
    h, w = img.shape[:2]
    acc: np.ndarray | None = None
    for s in spec["shapes"]:
        m = _shape(s, img, ctx)
        if s.get("invert"):
            m = 1 - m
        m = m * float(s.get("opacity", 1.0))
        mode = s.get("mode", "add")
        if acc is None or mode == "replace":
            acc = m if mode != "subtract" else np.clip(1 - m, 0, 1)
        elif mode == "add":
            acc = np.maximum(acc, m)
        elif mode == "subtract":
            acc = acc * (1 - m)
        elif mode == "intersect":
            acc = acc * m
    assert acc is not None
    acc = _feather(acc.astype(F32), float(spec.get("feather", 0.0)))
    if spec.get("invert"):
        acc = 1 - acc
    acc = np.clip(acc * float(spec.get("density", 1.0)), 0, 1)
    return acc.astype(F32)
