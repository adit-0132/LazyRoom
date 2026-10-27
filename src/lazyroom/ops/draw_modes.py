"""ImageToolbox "Draw" modes as pipeline operations (category ``draw``).

Port of ImageToolbox ``feature/draw`` (Apache-2.0, T8RIN): ``DrawMode``, ``DrawPathMode``,
``DrawLineStyle``, ``PathHelper``, ``BitmapDrawer``, ``FloodFill``, ``DrawUtils``,
``DrawRepeatedPath``, ``DrawOutlinedFill``, ``AndroidImageDrawApplier`` (the export renderer,
which is the reference for every formula here), Trickle ``WarpEngine.cpp`` and
``lib/opencv-tools`` ``SpotHealer``.  See ``docs/dev/port_reports/draw.md``.

Operations:

* ``draw.paint``      pen / highlighter / neon / eraser / privacy blur / pixelation / filter brush /
                      image brush / text-on-path strokes, with every path mode (free, line, arrows,
                      lasso, rect, oval, triangle, polygon, star, spray, flood fill) and line style
                      (dashed, dot-dashed, zigzag, stamped).
* ``draw.flood_fill`` paint-bucket fill (ImageToolbox's scanline flood fill + tolerance).
* ``draw.spot_heal``  inpaint brushed areas (OpenCV Telea / Navier-Stokes in XYZ, or LaMa).
* ``draw.warp``       liquify brush: move, grow, shrink, swirl, mix.

Like ImageToolbox, all strokes of one ``draw.paint`` step are drawn on a transparent drawing
layer that is composited over the image at the end; the eraser clears that layer (strokes of
the same step), revealing the image underneath.  Effects (blur, pixelation, filters) read the
image with the strokes drawn so far, exactly like the app's ``image.overlay(drawing)``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..core.blend import BLEND_MODES, layer_over
from ..core.errors import ErrorCode, LazyRoomError, invalid
from ..core.params import Bool, Choice, Color, File, Float, Int, Param, Point, Text
from ..core.registry import op
from . import _draw_paths as P
from . import _draw_raster as R
from ._draw_warp import MODES as WARP_MODES, warp_gesture
from .shapes import LINE_STYLES, STAMP_SHAPES

F32 = np.float32
SOURCE = {"project": "ImageToolbox", "feature": "Draw", "license": "Apache-2.0",
          "files": "feature/draw/**, Trickle WarpEngine.cpp, lib/opencv-tools/spot_heal"}

PT_PER_SIDE = 500.0  # ImageToolbox Pt: 1 pt = min(width, height) / 500 px

MODES = {
    "pen": "Pen", "highlighter": "Highlighter", "neon": "Neon", "eraser": "Eraser",
    "privacy_blur": "Privacy blur", "pixelation": "Pixelation paint", "filter_brush": "Filter brush",
    "image_brush": "Image brush", "text": "Text",
}
PATH_MODES = {
    "free": "Free drawing", "line": "Line", "arrow": "Arrow", "double_arrow": "Double arrow",
    "line_arrow": "Line arrow", "double_line_arrow": "Double line arrow", "lasso": "Lasso",
    "rect": "Rectangle", "oval": "Oval", "triangle": "Triangle", "polygon": "Polygon", "star": "Star",
    "spray": "Spray", "flood_fill": "Flood fill",
}
SHAPES = ("rect", "oval", "triangle", "polygon", "star")
ARROWS = ("arrow", "double_arrow", "line_arrow", "double_line_arrow")
EFFECT_MODES = ("privacy_blur", "pixelation", "filter_brush")


# =============================================================================
# Parameter types
# =============================================================================

def _is_point(v: Any) -> bool:
    if isinstance(v, dict):
        return "x" in v and "y" in v and "points" not in v
    return (isinstance(v, (list, tuple)) and len(v) == 2
            and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in v))


def _coerce_points(name: str, i: int, pts: Any) -> list[tuple[float, float]]:
    if not isinstance(pts, (list, tuple)) or not pts:
        raise invalid(f"Parameter '{name}': stroke {i} needs a non-empty 'points' list of [x, y]",
                      param=name, hint="Example: [[0.1, 0.5], [0.5, 0.4], [0.9, 0.5]] (normalised 0..1)")
    out = []
    for p in pts:
        if isinstance(p, dict) and "x" in p and "y" in p:
            p = (p["x"], p["y"])
        if not _is_point(p):
            raise invalid(f"Parameter '{name}': stroke {i} has an invalid point {p!r}", param=name,
                          hint="Points are [x, y] pairs in normalised image coordinates (0..1).")
        x, y = float(p[0]), float(p[1])
        if not (math.isfinite(x) and math.isfinite(y)) or not (-1 <= x <= 2 and -1 <= y <= 2):
            raise invalid(f"Parameter '{name}': stroke {i} point {p!r} is outside [-1, 2]", param=name)
        out.append((x, y))
    return out


@dataclass
class Strokes(Param):
    """List of strokes.  A stroke is ``[[x, y], ...]`` (normalised 0..1 image coordinates, in drawing
    order) or ``{"points": [[x, y], ...], <per-stroke overrides>}``.  A JSON string is accepted."""
    overrides: dict[str, Param] = field(default_factory=dict)
    max_strokes: int = 10_000

    def __post_init__(self) -> None:
        self.kind = "strokes"
        self.default = self.coerce([] if self.default is None else self.default)

    def coerce(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, str):
            if not value.strip():
                return []
            try:
                value = json.loads(value)
            except ValueError:
                raise invalid(f"Parameter '{self.name}' is not valid JSON", param=self.name) from None
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, (list, tuple)):
            raise invalid(f"Parameter '{self.name}' must be a list of strokes", param=self.name)
        if value and _is_point(value[0]):  # a single bare stroke
            value = [value]
        if len(value) > self.max_strokes:
            raise invalid(f"Parameter '{self.name}' has more than {self.max_strokes} strokes", param=self.name)
        out = []
        for i, item in enumerate(value):
            extra: dict[str, Any] = {}
            if isinstance(item, dict):
                extra = {k: v for k, v in item.items() if k != "points"}
                item = item.get("points")
            st: dict[str, Any] = {"points": _coerce_points(self.name, i, item)}
            for k, v in extra.items():
                p = self.overrides.get(k)
                if p is None:
                    raise invalid(f"Parameter '{self.name}': stroke {i} has an unknown field '{k}'", param=self.name,
                                  valid=["points", *self.overrides])
                if v is not None:
                    st[k] = p.coerce(v)
            out.append(st)
        return out

    def to_json(self, value: Any) -> list[Any]:
        res = []
        for st in value or []:
            pts = [[float(x), float(y)] for x, y in st["points"]]
            extra = {k: self.overrides[k].to_json(v) for k, v in st.items() if k != "points"}
            res.append({"points": pts, **extra} if extra else pts)
        return res

    def json_schema(self) -> dict[str, Any]:
        pts = {"type": "array", "minItems": 1,
               "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}
        props: dict[str, Any] = {"points": pts}
        keep = ("type", "enum", "minimum", "maximum", "pattern", "items", "minItems", "maxItems")
        for k, p in self.overrides.items():  # same meaning as the step parameter of that name
            props[k] = {kk: vv for kk, vv in p.json_schema().items() if kk in keep}
        obj = {"type": "object", "properties": props, "required": ["points"], "additionalProperties": False}
        return {"type": "array", "items": {"anyOf": [pts, obj]}, **self._schema_base()}

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["overrides"] = list(self.overrides)
        return d


@dataclass
class JsonObject(Param):
    """A JSON object (dict) or its JSON text."""

    def __post_init__(self) -> None:
        self.kind = "object"
        self.default = self.coerce({} if self.default is None else self.default)

    def coerce(self, value: Any) -> dict[str, Any]:
        if isinstance(value, str):
            if not value.strip():
                return {}
            try:
                value = json.loads(value)
            except ValueError:
                raise invalid(f"Parameter '{self.name}' is not valid JSON", param=self.name) from None
        if not isinstance(value, dict):
            raise invalid(f"Parameter '{self.name}' must be an object of parameter values", param=self.name)
        return dict(value)

    def json_schema(self) -> dict[str, Any]:
        return {"type": "object", **self._schema_base()}


# =============================================================================
# Drawing canvas (image + transparent drawing layer, premultiplied)
# =============================================================================

def _premul(img: np.ndarray) -> np.ndarray:
    if img.shape[2] == 3:
        return np.array(img, F32)
    a = img[..., 3:4]
    return np.concatenate([img[..., :3] * a, a], axis=2).astype(F32)


def _unpremul(prem: np.ndarray) -> np.ndarray:
    if prem.shape[2] == 3:
        return prem
    a = prem[..., 3:4]
    rgb = np.where(a > 1e-7, prem[..., :3] / np.maximum(a, 1e-7), 0.0)
    return np.concatenate([rgb, np.clip(a, 0, 1)], axis=2).astype(F32)


class _Canvas:
    """``image.overlay(drawing)``.  Without an eraser the strokes are drawn straight onto the image
    (source-over is associative, so this equals compositing a separate layer at the end)."""

    def __init__(self, img: np.ndarray, layer: bool):
        self.h, self.w = img.shape[:2]
        self.alpha = img.shape[2] == 4
        self.base = _premul(img)
        self.layer = np.zeros((self.h, self.w, 4), F32) if layer else None

    @property
    def full(self) -> R.Box:
        return 0, 0, self.h, self.w

    def composite(self, box: R.Box, copy: bool = True) -> np.ndarray:
        """Straight-alpha image + drawing so far, on ``box``."""
        y0, x0, y1, x1 = box
        b = self.base[y0:y1, x0:x1]
        if self.layer is not None:
            L = self.layer[y0:y1, x0:x1]
            la = L[..., 3:4]
            b = (L + b * (1 - la)) if self.alpha else (L[..., :3] + b * (1 - la))
        elif copy and not self.alpha:
            b = np.array(b, F32)
        return _unpremul(b)

    def _target(self, box: R.Box) -> np.ndarray:
        y0, x0, y1, x1 = box
        return (self.layer if self.layer is not None else self.base)[y0:y1, x0:x1]

    def paint_solid(self, box: R.Box, cov: np.ndarray, rgb, alpha: float) -> None:
        r, g, b = (float(c) for c in rgb[:3])
        R.blend_solid(self._target(box), np.ascontiguousarray(cov, F32), r, g, b, float(alpha))

    def paint(self, box: R.Box, src: np.ndarray, cov: np.ndarray | None = None, alpha: float = 1.0) -> None:
        """Source-over of a premultiplied RGBA ``src`` (times ``cov`` and ``alpha``)."""
        if cov is None:
            cov = np.ones(src.shape[:2], F32)
        R.blend_src(self._target(box), np.ascontiguousarray(src, F32), np.ascontiguousarray(cov, F32), float(alpha))

    def paint_straight(self, box: R.Box, img: np.ndarray, cov: np.ndarray) -> None:
        """Source-over of a straight-alpha RGB(A) image through ``cov``."""
        R.blend_straight(self._target(box), np.ascontiguousarray(img, F32), np.ascontiguousarray(cov, F32))

    def erase(self, box: R.Box, cov: np.ndarray) -> None:
        if self.layer is not None:
            R.erase(self._target(box), np.ascontiguousarray(cov, F32))

    def result(self) -> np.ndarray:
        if self.layer is None and not self.alpha:
            return self.base
        out = np.empty((self.h, self.w, 4 if self.alpha else 3), F32)
        R.finish(self.base, self.layer if self.layer is not None else np.zeros((1, 1, 4), F32),
                 self.layer is not None, out)
        return out


# =============================================================================
# Stroke settings
# =============================================================================

def _pt(h: int, w: int) -> float:
    return min(h, w) / PT_PER_SIDE


def _length(ctx, value: float, auto_pt: float, h: int, w: int) -> float:
    """``value`` px (full resolution) scaled to the buffer, or ImageToolbox's default in Pt when < 0."""
    return auto_pt * _pt(h, w) if value < 0 else ctx.px(value)


@dataclass
class _Stroke:
    s: dict[str, Any]
    pts: np.ndarray       # pixel coordinates
    width: float
    softness: float
    alpha: float

    @property
    def mode(self) -> str:
        return self.s["mode"]

    @property
    def path_mode(self) -> str:
        return "free" if self.mode == "eraser" else self.s["path_mode"]  # the eraser always draws free paths

    @property
    def filled(self) -> bool:
        pm = self.path_mode
        return pm in ("lasso", "spray", "flood_fill") or (pm in SHAPES and self.s.get("filled", False))

    @property
    def sharp(self) -> bool:  # DrawPathMode.isSharpEdge
        return self.path_mode in ("lasso", "spray", "flood_fill", "rect", "oval")


def _resolve(st: dict[str, Any], defaults: dict[str, Any], ctx, h: int, w: int) -> _Stroke:
    s = dict(defaults)
    s.update({k: v for k, v in st.items() if k != "points"})
    mode = s["mode"]
    pts = np.array(st["points"], np.float64) * (w, h)
    width = _length(ctx, s.get("width", -1.0), 20.0, h, w)
    softness = _length(ctx, s.get("softness", -1.0), 35.0 if mode == "neon" else 0.0, h, w)
    alpha = s.get("alpha", -1.0)
    alpha = (0.4 if mode == "highlighter" else 1.0) if alpha < 0 else alpha
    return _Stroke(s, pts, max(width, 1e-3), max(softness, 0.0), float(alpha))


# =============================================================================
# Geometry -> coverage
# =============================================================================

def _path(st: _Stroke) -> list[P.SubPath]:
    s, pts, pm = st.s, st.pts, st.path_mode
    if pm in ("free", "lasso"):
        return [(P.free_path(pts), pm == "lasso")]
    if pm == "line":
        return [(np.array([pts[0], pts[-1]]), False)]
    if pm in ARROWS:
        main = P.free_path(pts) if pm in ("arrow", "double_arrow") else np.array([pts[0], pts[-1]])
        return P.add_arrows(main, st.width, s["arrow_size"], s["arrow_angle"], start=pm.startswith("double"),
                            end=True)
    a, b = pts[0], pts[-1]
    if pm == "rect":
        return P.rect_path(a, b, s["rotation"], s["corner_radius"])
    if pm == "oval":
        return P.oval_path(a, b)
    if pm == "triangle":
        return P.triangle_path(a, b)
    if pm == "polygon":
        return P.polygon_path(a, b, s["vertices"], s["rotation"], s["regular"])
    if pm == "star":
        return P.star_path(a, b, s["vertices"], s["inner_radius"], s["rotation"], s["regular"])
    raise invalid(f"Path mode '{pm}' has no vector path")  # pragma: no cover - guarded by callers


def _line_style_polys(st: _Stroke, path: list[P.SubPath], h: int, w: int, ctx) -> list[np.ndarray] | None:
    """Zigzag / stamped (1D path effects, always filled) -> polygons; None for other styles."""
    ls = st.s.get("line_style", "none")
    if ls == "zigzag":
        return P.stamped_polys(path, P.zigzag_stamp(st.width, st.s["zigzag_ratio"]), st.width)
    if ls == "stamped":
        spacing = _length(ctx, st.s["stamp_spacing"], 20.0, h, w)
        return P.stamped_polys(path, P.stamp_shape(st.s["stamp_shape"], st.width), spacing + st.width)
    return None


def _coverage(st: _Stroke, path: list[P.SubPath], *, fill: bool, cap: str, join: str, line_style: bool,
              h: int, w: int, ctx, pad: float) -> tuple[np.ndarray, R.Box] | None:
    """``canvas.drawPath(path, paint)`` coverage for a solid paint (Skia stroke/fill + path effect)."""
    if line_style:
        polys = _line_style_polys(st, path, h, w, ctx)
        if polys is not None:
            return R.rasterize(polys, h, w, pad)
    if fill:  # dash effects do nothing on filled paths (SkDashPath skips fill styles)
        return R.rasterize([p for p, _ in path], h, w, pad)
    pieces = path
    ls = st.s.get("line_style", "none") if line_style else "none"
    if ls == "dashed":
        dash = _length(ctx, st.s["dash_size"], 10.0, h, w)
        gap = _length(ctx, st.s["gap_size"], 20.0, h, w)
        pieces = P.dash_pieces(path, [dash, gap + st.width])
    elif ls == "dot_dashed":
        sw = st.width
        pieces = P.dash_pieces(path, [sw * 4, sw * 2, sw / 4, sw * 2])
    polys: list[np.ndarray] = []
    for pts, closed in pieces:
        polys.extend(R.stroke_outline(pts, closed, st.width, cap, join))
    return R.rasterize(polys, h, w, pad)


def _paint_style(st: _Stroke, effect: bool = False) -> tuple[bool, str, str]:
    """(fill, cap, join): createDrawPaint / pathEffectPaint.  Android's default join is MITER."""
    if st.filled:
        return True, "butt", "miter"
    if st.sharp or (st.mode == "highlighter" and not effect):
        return False, "square", "miter"
    return False, "round", "round"


def _shape_coverage(st: _Stroke, canvas: _Canvas, ctx, idx: int, pad: float, *, effect: bool = False,
                    line_style: bool = True) -> tuple[np.ndarray, R.Box] | None:
    h, w = canvas.h, canvas.w
    pm = st.path_mode
    if pm == "spray":
        events = P.spray_events(st.pts, st.width / 2)
        xs, ys = P.spray_particles(events, int(st.s["density"]), st.width, ctx.rng(f"spray:{idx}"))
        return R.stamp_particles(xs, ys, max(ctx.px(st.s["particle_size"]), 0.05), st.s["square_particles"], h, w,
                                 pad)
    if pm == "flood_fill":
        x, y = st.pts[-1]
        res = P.flood_fill_region(canvas.base, canvas.layer, int(math.floor(x)), int(math.floor(y)),
                                  st.s["tolerance"])
        if res is None:
            return None
        mask, inner = res
        box = R.grow_box(inner, pad, h, w)
        return R.place(mask.astype(F32), inner, box), box
    path = _path(st)
    if not path:
        return None
    fill, cap, join = _paint_style(st, effect)
    return _coverage(st, path, fill=fill, cap=cap, join=join, line_style=line_style and not effect, h=h, w=w,
                     ctx=ctx, pad=pad)


# =============================================================================
# Mode renderers
# =============================================================================

def _draw_solid(canvas: _Canvas, cb, rgb, alpha: float, neon: bool, softness: float, neon_rgb=None) -> None:
    """One ``drawPath`` with a colour paint: BlurMaskFilter softness, or Neon's shadow layer + white core."""
    if cb is None:
        return
    cov, box = cb
    if neon:
        if softness > 0:  # setShadowLayer(softness, 0, 0, colour @ 0.8); radius 0 removes the shadow
            canvas.paint_solid(box, R.blur(cov, softness), neon_rgb if neon_rgb is not None else rgb, 0.8)
        canvas.paint_solid(box, cov, rgb, alpha)
        return
    canvas.paint_solid(box, R.blur(cov, softness), rgb, alpha)


def _render_color(canvas: _Canvas, st: _Stroke, ctx, idx: int) -> None:
    s = st.s
    color = s["color"]
    neon = st.mode == "neon"
    pad = R.blur_margin(st.softness)
    alpha = float(color[3]) * st.alpha
    # outlined shapes: DrawOutlinedFill first (paint copy without path effect; opaque fill colours take
    # the stroke alpha, translucent ones keep their own)
    if st.path_mode in SHAPES and not st.filled and s["fill_color"][3] > 0:
        path = _path(st)
        fc = s["fill_color"]
        if path:
            fa = float(fc[3]) if fc[3] < 1 else (1.0 if neon else alpha)
            cb = R.rasterize([p for p, _ in path], canvas.h, canvas.w, pad)
            _draw_solid(canvas, cb, fc[:3], fa, neon, st.softness, neon_rgb=color[:3])
    cb = _shape_coverage(st, canvas, ctx, idx, pad)
    if neon:  # paint.color = White (opaque), shadow = draw colour
        _draw_solid(canvas, cb, (1.0, 1.0, 1.0), 1.0, True, st.softness, neon_rgb=color[:3])
    else:
        _draw_solid(canvas, cb, color[:3], alpha, False, st.softness)


def _render_eraser(canvas: _Canvas, st: _Stroke, ctx) -> None:
    path = [(P.free_path(st.pts), False)]
    pad = R.blur_margin(st.softness)
    polys = R.stroke_outline(path[0][0], False, st.width, "round", "round")
    cb = R.rasterize(polys, canvas.h, canvas.w, pad)
    if cb is not None:
        canvas.erase(cb[1], R.blur(cb[0], st.softness))


def _stack_blur(img: np.ndarray, ctx, radius: int) -> np.ndarray:
    from .itb.blur import native_stack_blur  # libblur stack blur port (NativeStackBlurFilter)

    return native_stack_blur(img, ctx.derive(scale=1.0), radius=int(radius))


def _pixelate(img: np.ndarray, cell: float, oy: int, ox: int) -> np.ndarray:
    """PixelationFilter (PixelationTool square layer) on a crop whose origin is (oy, ox) in the image,
    keeping the full-image cell grid."""
    from .itb import _pixelation_impl as PX

    res = max(float(cell), 0.5)
    src = np.ascontiguousarray(img, dtype=F32)
    prem = np.zeros(src.shape[:2] + (4,), F32)
    offx = -(ox - math.floor(ox / res) * res)
    offy = -(oy - math.floor(oy / res) * res)
    PX._pixelate_layer(src, prem, PX.SQUARE, res, res, 1.0, offx, offy)
    return PX.prem_to_image(prem, img.shape[2] == 4)


def _filter_spec(fid: str, params: dict[str, Any]):
    from ..core.registry import get_op

    if not fid:
        raise invalid("filter_brush needs 'filter': the id of an operation to apply inside the stroke",
                      param="filter", hint="e.g. filter='filter.gaussian_blur' or 'adjust.saturation'; "
                                           "see list_operations")
    try:
        spec = get_op(fid)
    except LazyRoomError as e:
        raise invalid(f"filter_brush: unknown operation '{fid}'", param="filter",
                      suggestions=e.details.get("suggestions", [])) from None
    if spec.geometry:
        raise invalid(f"filter_brush: '{fid}' changes the image geometry and cannot be painted", param="filter")
    return spec, spec.validate(params)


def _render_effect(canvas: _Canvas, st: _Stroke, ctx, idx: int) -> None:
    """PathEffect modes: filter ``image.overlay(drawing)`` and draw it through the path
    (``clipBitmap`` + ``pathEffectPaint``: no colour, alpha, softness or line style)."""
    h, w = canvas.h, canvas.w
    cb = _shape_coverage(st, canvas, ctx, idx, 0.0, effect=True)
    if cb is None:
        return
    cov, box = cb
    mode, s = st.mode, st.s
    longest = max(h, w)
    if mode == "privacy_blur":
        r = max(1, int(s["blur_radius"] / 1000.0 * longest))
        region = R.grow_box(box, r + 2, h, w)
        filt = _stack_blur(canvas.composite(region, copy=False), ctx, r)
    elif mode == "pixelation":
        r = max(1, int(20.0 / 1000.0 * longest))
        cell = s["pixel_size"] / 1000.0 * longest
        region = R.grow_box(box, r + cell + 2, h, w)
        filt = _pixelate(_stack_blur(canvas.composite(region, copy=False), ctx, r), cell, region[0], region[1])
    else:
        spec, vals = _filter_spec(s["filter"], s["filter_params"])
        region = canvas.full
        filt = spec.run(canvas.composite(region), ctx, vals)
        if filt.shape[:2] != (h, w):
            raise LazyRoomError(ErrorCode.OPERATION_FAILED, f"filter_brush: '{s['filter']}' changed the image size")
    ys, xs = R.sub(region, box)
    canvas.paint_straight(box, filt[ys, xs], cov)


# -- text on path -----------------------------------------------------------------------------

def _glyph_layout(text: str, font) -> list[tuple[str, float, float]]:
    """(char, x, advance) like minikin ``Layout.getX`` / ``getCharAdvance``; pair kerning from
    ``getlength(a + b) - getlength(a) - getlength(b)``."""
    cache: dict[str, float] = {}

    def ln(t: str) -> float:
        v = cache.get(t)
        if v is None:
            v = cache[t] = float(font.getlength(t))
        return v

    out, x, prev = [], 0.0, None
    for ch in text:
        adv = ln(ch)
        if prev is not None:
            x += ln(prev + ch) - adv  # previous advance + kerning
        out.append((ch, x, adv))
        prev = ch
    return out


def _render_text(canvas: _Canvas, st: _Stroke, ctx) -> None:
    from .draw import load_font

    s = st.s
    text = s["text"]
    main = P.main_contour(_path(st))
    if not text or main is None:
        return
    size = max(1, int(round(st.width)))  # textSize = stroke width
    font = load_font(s["font"], size)
    meas = P._measure(main)
    L = meas.length
    runs: list[tuple[str, float]] = []
    if s["text_repeat"]:  # DrawRepeatedPath.drawRepeatedTextOnPath
        tw = font.getlength(text)
        unit = max(tw + ctx.px(s["interval"]), 1.0)
        full = int(L / unit)
        runs = [(text, k * unit) for k in range(full)]
        rem = L - full * unit
        if rem > 0:
            ratio = (unit - rem) / unit
            end = max(len(text) - int(len(text) * ratio), 0)
            if end:
                runs.append((text[:end], full * unit))
    else:
        runs = [(text, 0.0)]
    glyphs = []
    for run, h_off in runs:
        for ch, x, adv in _glyph_layout(run, font):
            l, t, r, b = font.getbbox(ch, anchor="ls")
            l, t, r, b = int(math.floor(l)), int(math.floor(t)), int(math.ceil(r)), int(math.ceil(b))
            if r <= l or b <= t:
                continue
            (pos,), (tan,) = meas.pos_tan(np.array([h_off + x + adv / 2]))  # pinned to [0, length]
            origin = pos - tan * (adv / 2)  # SkiaCanvas::drawLayoutOnPath RSXform
            glyphs.append((ch, origin, tan, (l, t, r, b)))
    if not glyphs:
        return
    corners = []
    for _, o, tn, (l, t, r, b) in glyphs:
        rot = np.array([[tn[0], -tn[1]], [tn[1], tn[0]]])
        loc = np.array([[l, t], [r, t], [r, b], [l, b]], np.float64)
        corners.append(o + loc @ rot.T)
    pad = R.blur_margin(st.softness) + 2
    box = R.poly_box(corners, canvas.h, canvas.w, pad)
    if box is None:
        return
    bh, bw = box[2] - box[0], box[3] - box[1]
    cov = np.zeros((bh, bw), F32)
    for ch, o, tn, (l, t, r, b) in glyphs:
        im = Image.new("L", (r - l, b - t), 0)
        ImageDraw.Draw(im).text((-l, -t), ch, font=font, fill=255, anchor="ls")
        g = np.asarray(im, F32) / 255.0
        c, sn = float(tn[0]), float(tn[1])
        # glyph pixel (i, j) centre = local (l + i + 0.5, t + j + 0.5); target index = world - 0.5 - box origin
        lx, ly = l + 0.5, t + 0.5
        M = np.array([[c, -sn, o[0] + c * lx - sn * ly - 0.5 - box[1]],
                      [sn, c, o[1] + sn * lx + c * ly - 0.5 - box[0]]], np.float64)
        warped = cv2.warpAffine(g, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        np.maximum(cov, warped, out=cov)
    color = s["color"]
    canvas.paint_solid(box, R.blur(cov, st.softness), color[:3], float(color[3]) * st.alpha)


# -- image brush ------------------------------------------------------------------------------

def _load_brush_image(ctx, path: str, size: int) -> np.ndarray:
    from ..core.io import load_image
    from ..core.image import ensure_rgba

    if not path:
        raise invalid("image_brush needs 'image': the path of the picture to repeat along the stroke",
                      param="image")
    im, _ = load_image(ctx.resolve_file(path, "image brush"))
    im = ensure_rgba(im)
    ih, iw = im.shape[:2]
    k = size / max(ih, iw)  # Coil .size(stroke): fit inside stroke x stroke
    nw, nh = max(1, int(round(iw * k))), max(1, int(round(ih * k)))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    return _premul(np.clip(im, 0, 1).reshape(nh, nw, 4))


def _render_image(canvas: _Canvas, st: _Stroke, ctx) -> None:
    """DrawRepeatedPath.drawRepeatedBitmapOnPath: copies every (bitmap width + interval) along the path."""
    main = P.main_contour(_path(st))
    if main is None:
        return
    stamp = _load_brush_image(ctx, st.s["image"], max(1, int(round(st.width))))
    sh, sw = stamp.shape[:2]
    meas = P._measure(main)
    adv = max(sw + ctx.px(st.s["interval"]), 1.0)
    alpha = st.alpha
    n = int(math.ceil(meas.length / adv)) if meas.length > 0 else 1
    dists = np.arange(max(n, 1)) * adv
    pos, tan = meas.pos_tan(dists)
    pad = R.blur_margin(st.softness) + 2
    for p, t in zip(pos, tan):
        c, sn = float(t[0]), float(t[1])
        rot = np.array([[c, -sn], [sn, c]])
        loc = np.array([[-sw / 2, -sh / 2], [sw / 2, -sh / 2], [sw / 2, sh / 2], [-sw / 2, sh / 2]])
        box = R.poly_box([p + loc @ rot.T], canvas.h, canvas.w, pad)
        if box is None:
            continue
        bh, bw = box[2] - box[0], box[3] - box[1]
        lx, ly = -sw / 2 + 0.5, -sh / 2 + 0.5
        M = np.array([[c, -sn, p[0] + c * lx - sn * ly - 0.5 - box[1]],
                      [sn, c, p[1] + sn * lx + c * ly - 0.5 - box[0]]], np.float64)
        if st.softness > 0:  # Skia: a mask filter on drawBitmap blurs the bitmap's rectangle, clamp shader
            col = cv2.warpAffine(stamp, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            rect = cv2.warpAffine(np.ones((sh, sw), F32), M, (bw, bh), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
            cov = R.blur(rect, st.softness)
        else:
            col = cv2.warpAffine(stamp, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cov = None
        canvas.paint(box, col.reshape(bh, bw, 4), cov, alpha)


# =============================================================================
# draw.paint
# =============================================================================

def _validate_stroke(st: _Stroke) -> None:
    mode, pm = st.mode, st.path_mode
    if mode in ("text", "image_brush") and pm in ("spray", "flood_fill"):
        raise invalid(f"Mode '{mode}' follows a path; path mode '{pm}' has none",
                      hint="Use free, line, an arrow or a shape path mode (ImageToolbox offers Free, Line and the "
                           "outlined shapes for text and images).")
    if mode == "filter_brush":
        _filter_spec(st.s["filter"], st.s["filter_params"])
    if mode == "image_brush" and not st.s["image"]:
        raise invalid("image_brush needs 'image': the path of the picture to repeat along the stroke",
                      param="image")


def _paint(img: np.ndarray, ctx, strokes: list[dict[str, Any]], defaults: dict[str, Any]) -> np.ndarray:
    h, w = img.shape[:2]
    resolved = [_resolve(st, defaults, ctx, h, w) for st in strokes]
    for st in resolved:
        _validate_stroke(st)
    if not resolved:
        return img
    erasing = any(st.mode == "eraser" for st in resolved[1:])
    canvas = _Canvas(img, layer=erasing)
    for i, st in enumerate(resolved):
        ctx.check()
        if st.mode == "eraser":
            _render_eraser(canvas, st, ctx)
        elif st.mode in EFFECT_MODES:
            _render_effect(canvas, st, ctx, i)
        elif st.mode == "text":
            _render_text(canvas, st, ctx)
        elif st.mode == "image_brush":
            _render_image(canvas, st, ctx)
        else:
            _render_color(canvas, st, ctx, i)
    return canvas.result()


def _paint_params() -> list[Param]:
    return [
        Choice("mode", "pen", choices=list(MODES), labels=MODES, title="Draw mode",
               description="pen; highlighter (square caps, default alpha 0.4); neon (white core + glow in the "
                           "colour); eraser (clears strokes drawn earlier in this step); privacy_blur, pixelation, "
                           "filter_brush (apply an effect inside the stroke); image_brush (repeat a picture "
                           "along the path); text (text along the path)"),
        Choice("path_mode", "free", choices=list(PATH_MODES), labels=PATH_MODES, title="Path mode",
               description="free: smoothed freehand path through the points; line / *arrow / shapes use the "
                           "first and last point (shapes: the box between them); lasso: filled freehand area; "
                           "spray: particles around every point; flood_fill: fill the region at the last point"),
        Bool("filled", False, title="Filled shape",
             description="rect/oval/triangle/polygon/star: fill the shape instead of stroking its outline"),
        Float("width", -1, min=-1, max=5000, step=0.5, unit="px", title="Line width",
              description="Stroke width in full-resolution pixels (text: font size; image brush: picture size; "
                          "spray: spray radius). -1 = ImageToolbox default 20 Pt (4 % of the shorter side)"),
        Color("color", "#000000", title="Colour", description="Draw colour (its alpha multiplies 'alpha')"),
        Float("alpha", -1, min=-1, max=1, step=0.01, title="Alpha",
              description="Stroke opacity. -1 = mode default (0.4 for highlighter, 1 otherwise); ignored by neon"),
        Float("softness", -1, min=-1, max=2000, step=0.5, unit="px", title="Brush softness",
              description="Edge blur radius (Android BlurMaskFilter); for neon the glow radius. -1 = mode "
                          "default (35 Pt = 7 % of the shorter side for neon, 0 otherwise)"),
        Color("fill_color", "#00000000", title="Fill colour",
              description="Outlined shapes: colour inside the outline (transparent = none)"),
        Choice("line_style", "none", choices=list(LINE_STYLES), labels=LINE_STYLES, title="Line style",
               description="Pen, highlighter and neon only: dashed, dot-dashed, zigzag or stamped shapes"),
        Float("dash_size", -1, min=-1, max=5000, step=0.5, unit="px", title="Dash size",
              description="dashed: dash length. -1 = 10 Pt"),
        Float("gap_size", -1, min=-1, max=5000, step=0.5, unit="px", title="Gap size",
              description="dashed: gap between dashes (the line width is added, compensating round caps). "
                          "-1 = 20 Pt"),
        Float("zigzag_ratio", 4, min=0.5, max=20, step=0.01, title="Zigzag ratio",
              description="zigzag: line width / zigzag thickness"),
        Choice("stamp_shape", "material_star", choices=list(STAMP_SHAPES), title="Stamp shape",
               description="stamped: shape repeated along the path (ImageToolbox icon shapes)"),
        Float("stamp_spacing", -1, min=-1, max=5000, step=0.5, unit="px", title="Stamp spacing",
              description="stamped: gap between stamps (added to the line width). -1 = 20 Pt"),
        Float("arrow_size", 3, min=0.5, max=8, step=0.1, title="Arrow head length",
              description="Arrow modes: head length as a multiple of the line width"),
        Float("arrow_angle", 150, min=90, max=180, step=0.5, unit="deg", title="Arrow angle",
              description="Arrow modes: angle between the shaft direction and each head line (ImageToolbox "
                          "shows this minus 90)"),
        Int("vertices", 5, min=3, max=24, title="Vertices", description="polygon / star: number of corners"),
        Float("rotation", 0, min=0, max=360, step=1, unit="deg", title="Angle",
              description="rect / polygon / star rotation"),
        Float("corner_radius", 0, min=0, max=0.5, step=0.01, title="Corner radius",
              description="rect: corner radius as a fraction of the shorter rectangle side"),
        Float("inner_radius", 0.5, min=0, max=1, step=0.01, title="Inner radius ratio",
              description="star: inner / outer radius"),
        Bool("regular", False, title="Regular shape",
             description="polygon / star: regular shape inscribed in the box instead of stretched to it"),
        Int("density", 50, min=1, max=500, title="Spray density", description="spray: particles per point"),
        Float("particle_size", 2, min=0.1, max=200, step=0.1, unit="px", title="Particle size",
              description="spray: particle diameter"),
        Bool("square_particles", False, title="Square particles", description="spray: squares instead of dots"),
        Float("tolerance", 0.25, min=0, max=1, step=0.01, title="Tolerance",
              description="flood_fill: per-channel colour tolerance (0..1 of 255)"),
        Int("blur_radius", 20, min=5, max=50, title="Blur radius",
            description="privacy_blur: stack-blur radius in 1/1000 of the longer image side"),
        Float("pixel_size", 35, min=10, max=75, step=0.5, title="Pixel size",
              description="pixelation: cell size in 1/1000 of the longer image side"),
        Text("filter", "", max_length=200, title="Filter operation",
             description="filter_brush: id of any non-geometry operation (e.g. filter.sepia, adjust.saturation)"),
        JsonObject("filter_params", {}, title="Filter parameters",
                   description="filter_brush: parameters of that operation, e.g. {\"amount\": 50}"),
        File("image", "", file_kind="image", title="Brush image",
             description="image_brush: picture repeated along the path (fitted inside width x width)"),
        Float("interval", 0, min=-5000, max=5000, step=0.5, unit="px", title="Repeat interval",
              description="image_brush / repeated text: extra spacing between copies (negative overlaps images)"),
        Text("text", "Text", max_length=5000, title="Text", description="text: the text drawn along the path"),
        Text("font", "sans-serif", max_length=512, title="Font",
             description="text: font family name (system fonts) or path to a .ttf/.otf"),
        Bool("text_repeat", False, title="Repeat text", description="text: repeat the text along the whole path"),
    ]


_PAINT_OVERRIDES = _paint_params()
_DEMO = [[[0.2, 0.6], [0.35, 0.4], [0.5, 0.6], [0.65, 0.4], [0.8, 0.6]]]

_STROKES_DOC = ("Strokes in drawing order. Each stroke is a list of [x, y] points in normalised image "
                "coordinates (0..1, like pointer events: the first is the 'down', the last the 'up' position) or "
                "an object {\"points\": [...], ...} whose other keys override this step's parameters for that "
                "stroke, e.g. {\"points\": [[0.1, 0.1], [0.4, 0.3]], \"path_mode\": \"rect\", \"color\": \"#FF0000\"}")


@op("draw.paint", "Draw / paint strokes (ImageToolbox Draw)", params=[
    Strokes("strokes", _DEMO, title="Strokes", description=_STROKES_DOC,
            overrides={p.name: p for p in _PAINT_OVERRIDES}),
    *_PAINT_OVERRIDES,
], alpha="full", source=SOURCE,
    tags=["draw", "paint", "brush", "pen", "highlighter", "neon", "marker", "eraser", "blur brush", "censor",
          "pixelate", "mosaic", "spray", "arrow", "rectangle", "oval", "triangle", "polygon", "star", "lasso",
          "dashed", "zigzag", "stamp", "text on path", "image brush", "flood fill", "markup", "annotate"],
    examples=[
        {"strokes": [[[0.1, 0.8], [0.3, 0.75], [0.5, 0.8]]], "mode": "highlighter", "color": "#FFEB3B",
         "width": 40},
        {"strokes": [{"points": [[0.2, 0.2], [0.6, 0.5]], "path_mode": "line_arrow"}], "color": "#FF3B30"},
        {"strokes": [[[0.3, 0.3], [0.5, 0.45]]], "mode": "privacy_blur", "path_mode": "rect", "filled": True},
        {"strokes": [[[0.2, 0.5], [0.8, 0.5]]], "line_style": "dashed", "mode": "neon", "color": "#00E5FF"},
    ])
def paint(img, ctx, strokes, **defaults):
    """ImageToolbox Draw: freehand and shape strokes with pen, highlighter, neon, eraser, privacy blur,
    pixelation, filter brush, image brush and text-on-path modes, line styles and spray / flood fill.
    Strokes of one step share a drawing layer (the eraser removes earlier strokes of the same step)."""
    return _paint(img, ctx, strokes, defaults)


# =============================================================================
# draw.flood_fill
# =============================================================================

@op("draw.flood_fill", "Flood fill (paint bucket)", params=[
    Point("point", (0.5, 0.5), min=0, max=1, title="Seed point",
          description="Where to start filling (normalised image coordinates)"),
    Float("tolerance", 0.25, min=0, max=1, step=0.01, title="Tolerance",
          description="Fill the 4-connected area whose A, R, G and B each differ from the seed pixel by at most "
                      "tolerance x 255"),
    Color("color", "#000000", title="Fill colour"),
    Float("alpha", 1, min=0, max=1, step=0.01, title="Alpha"),
    Float("softness", 0, min=0, max=2000, step=0.5, unit="px", title="Softness",
          description="Blur radius of the fill edge (Android BlurMaskFilter)"),
], alpha="full", source=SOURCE, tags=["flood fill", "paint bucket", "fill", "bucket", "region", "magic fill"],
    examples=[{"point": [0.5, 0.2], "tolerance": 0.1, "color": "#87CEEB"}])
def flood_fill(img, ctx, point, tolerance, color, alpha, softness):
    """ImageToolbox Draw > Flood fill: scanline fill of the region connected to the seed point.  Combine
    with other draw modes through draw.paint (path_mode='flood_fill', e.g. mode='privacy_blur')."""
    defaults = {p.name: p.default for p in _PAINT_OVERRIDES}
    defaults.update(mode="pen", path_mode="flood_fill", tolerance=tolerance, color=color, alpha=alpha,
                    softness=softness)
    return _paint(img, ctx, [{"points": [tuple(point)]}], defaults)


# =============================================================================
# draw.spot_heal
# =============================================================================

HEAL_METHODS = {"telea": "OpenCV Telea (ImageToolbox default)", "ns": "OpenCV Navier-Stokes",
                "lama": "LaMa AI inpainting (model download required)"}
HEAL_PATHS = {"free": "Brush", "line": "Line", "lasso": "Lasso (filled)", "rect": "Rectangle (filled)",
              "oval": "Oval (filled)"}


def _inpaint_cv(img: np.ndarray, mask8: np.ndarray, radius: float, method: str) -> np.ndarray:
    """SpotHealer.heal: 8-bit RGB -> XYZ, Photo.inpaint, XYZ -> RGB (alpha inpainted separately)."""
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    rgb8 = (np.clip(img[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)
    xyz = cv2.cvtColor(rgb8, cv2.COLOR_RGB2XYZ)
    out = cv2.cvtColor(cv2.inpaint(xyz, mask8, float(radius), flag), cv2.COLOR_XYZ2RGB).astype(F32) / 255.0
    if img.shape[2] == 4:
        a = cv2.inpaint(np.ascontiguousarray(img[..., 3], F32), mask8, float(radius), flag)
        out = np.dstack([out, np.clip(a, 0, 1)])
    return out


def _inpaint_lama(img: np.ndarray, mask: np.ndarray, ctx) -> np.ndarray:
    from ..ai import models, processors

    name = models.resolve_name("@inpaint")
    path = models.require_model(name)
    rgb = processors.lama_inpaint(name, path, img[..., :3], mask, ctx)
    return np.dstack([rgb, img[..., 3]]) if img.shape[2] == 4 else rgb


@op("draw.spot_heal", "Spot healing brush", params=[
    Strokes("strokes", [], title="Strokes",
            description="Areas to heal: lists of [x, y] points (normalised 0..1); objects {\"points\": ..., "
                        "\"width\": px, \"path_mode\": ...} override per stroke",
            overrides={"width": Float("width", -1, min=-1, max=5000, unit="px"),
                       "path_mode": Choice("path_mode", "free", choices=list(HEAL_PATHS))}),
    Float("width", -1, min=-1, max=5000, step=0.5, unit="px", title="Brush width",
          description="Brush width in full-resolution pixels; -1 = ImageToolbox default 20 Pt (4 % of the "
                      "shorter side)"),
    Choice("path_mode", "free", choices=list(HEAL_PATHS), labels=HEAL_PATHS, title="Path mode",
           description="free/line: brush stroke; lasso/rect/oval: heal the enclosed area"),
    Choice("method", "telea", choices=list(HEAL_METHODS), labels=HEAL_METHODS, title="Method",
           description="telea/ns: classical OpenCV inpainting; lama: generative inpainting (ai.inpaint model)"),
    Float("radius", 3, min=1, max=50, step=0.5, unit="px", title="Inpaint radius",
          description="OpenCV neighbourhood radius (ImageToolbox uses 3)"),
], alpha="full", source=SOURCE, tags=["heal", "spot heal", "healing brush", "remove blemish", "inpaint", "retouch",
                                      "remove object", "clone"],
    examples=[{"strokes": [[[0.42, 0.40], [0.45, 0.41]]], "width": 30}])
def spot_heal(img, ctx, strokes, width, path_mode, method, radius):
    """ImageToolbox Draw > Spot heal: paint over blemishes/objects; the painted area is inpainted from its
    surroundings (OpenCV Telea in XYZ like SpotHealer, Navier-Stokes, or LaMa) and drawn through the path."""
    h, w = img.shape[:2]
    if not strokes:
        return img
    canvas = _Canvas(img, layer=False)
    defaults = {p.name: p.default for p in _PAINT_OVERRIDES}
    defaults.update(mode="pen", path_mode=path_mode, width=width, softness=0.0, alpha=1.0, filled=True,
                    line_style="none")
    for i, raw in enumerate(strokes):
        ctx.check()
        st = _resolve(raw, defaults, ctx, h, w)
        cb = _shape_coverage(st, canvas, ctx, i, 0.0, effect=True, line_style=False)
        if cb is None:
            continue
        cov, box = cb
        if method == "lama":
            region = canvas.full
            full_mask = np.zeros((h, w), F32)
            full_mask[box[0]:box[2], box[1]:box[3]] = (cov > 0).astype(F32)
            healed = _inpaint_lama(canvas.composite(region), full_mask, ctx)
        else:
            rad = max(1.0, ctx.px(radius))
            region = R.grow_box(box, 2 * rad + 8, h, w)
            m8 = np.zeros((region[2] - region[0], region[3] - region[1]), np.uint8)
            ys, xs = R.sub(region, box)
            m8[ys, xs] = (cov > 0).astype(np.uint8) * 255
            healed = _inpaint_cv(canvas.composite(region), m8, rad, method)
        ys, xs = R.sub(region, box)
        canvas.paint_straight(box, healed[ys, xs], cov)
    return canvas.result()


# =============================================================================
# draw.warp
# =============================================================================

WARP_LABELS = {"move": "Move", "grow": "Grow", "shrink": "Shrink", "swirl_cw": "Swirl CW",
               "swirl_ccw": "Swirl CCW", "mix": "Mix"}


@op("draw.warp", "Warp / liquify brush", params=[
    Strokes("strokes", [], title="Strokes",
            description="Brush gestures: lists of [x, y] points (normalised 0..1) in drag order; every point is a "
                        "pointer event, so pass a dense path (or a back-and-forth / circular path for grow, shrink "
                        "and swirl). Objects {\"points\": ..., \"warp_mode\", \"radius\", \"strength\", "
                        "\"hardness\"} override per gesture",
            overrides={"warp_mode": Choice("warp_mode", "move", choices=list(WARP_MODES)),
                       "radius": Float("radius", -1, min=-1, max=5000, unit="px"),
                       "strength": Float("strength", 0.25, min=0, max=1),
                       "hardness": Float("hardness", 0.5, min=0, max=1)}),
    Choice("warp_mode", "move", choices=list(WARP_MODES), labels=WARP_LABELS, title="Warp mode",
           description="move: push pixels along the drag; grow / shrink: bloat / pinch under the brush; swirl_cw / "
                       "swirl_ccw: twist; mix: smudge colours towards the brush centre"),
    Float("radius", -1, min=-1, max=5000, step=0.5, unit="px", title="Brush radius",
          description="Brush radius in full-resolution pixels (ImageToolbox: the line width); -1 = 20 Pt (4 % of "
                      "the shorter side)"),
    Float("strength", 0.25, min=0, max=1, step=0.01, title="Strength"),
    Float("hardness", 0.5, min=0, max=1, step=0.01, title="Hardness",
          description="Falloff start: only distances below (1 - hardness) x radius are affected, smoothly"),
], alpha="full", source=SOURCE, tags=["warp", "liquify", "push", "bloat", "pinch", "swirl", "twirl", "smudge",
                                      "reshape", "distort brush"],
    examples=[{"strokes": [[[0.4, 0.5], [0.45, 0.5], [0.5, 0.5], [0.55, 0.5]]], "warp_mode": "move",
               "radius": 120, "strength": 0.6}])
def warp(img, ctx, strokes, warp_mode, radius, strength, hardness):
    """ImageToolbox Draw > Warp (Trickle WarpEngine): each gesture builds a displacement map from brush dabs
    spaced radius/2 apart and resamples the image (move / grow / shrink / swirl), or smudges colours (mix)."""
    h, w = img.shape[:2]
    if not strokes:
        return img
    pix = _premul(img)
    for raw in strokes:
        ctx.check()
        mode = raw.get("warp_mode", warp_mode)
        r = _length(ctx, raw.get("radius", radius), 20.0, h, w)
        pts = np.array(raw["points"], np.float64) * (w, h) - 0.5  # engine: pixel centres at integers
        warp_gesture(pix, pts, r, float(raw.get("strength", strength)), float(raw.get("hardness", hardness)), mode)
    out = _unpremul(pix)
    return out if img.shape[2] == 4 else out[..., :3]


# =============================================================================
# draw.shape_layer (ImageToolbox Markup layers: shape layer)
# =============================================================================

LAYER_SHAPES = {"line": "Line", "arrow": "Arrow (filled)", "double_arrow": "Double arrow (filled)",
                "line_arrow": "Line arrow", "double_line_arrow": "Double line arrow", "rect": "Rectangle",
                "oval": "Oval", "triangle": "Triangle", "polygon": "Polygon", "star": "Star"}


def _layer_shape_paths(shape: str, filled: bool, bw: float, bh: float, sw: float, s: dict) -> tuple[list, bool, bool]:
    """ShapeLayerRenderer.buildShapePath in the shape box (0..bw, 0..bh).  Returns (sub-paths, fill, stroke)."""
    closed5 = shape in SHAPES
    uses_stroke = shape in ("line", "line_arrow", "double_line_arrow") or (closed5 and not filled)
    inset = sw / 2 + 1 if uses_stroke else 1.0
    left, top = min(inset, bw / 2), min(inset, bh / 2)
    right, bottom = max(left + 1, bw - inset), max(top + 1, bh - inset)
    if shape == "rect":
        path = P.rect_path((left, top), (right, bottom), s["shape_rotation"], s["corner_radius"])
    elif shape == "oval":
        path = P.oval_path((left, top), (right, bottom))
    elif shape == "triangle":
        path = [(np.array([[(left + right) / 2, top], [right, bottom], [left, bottom]], np.float64), True)]
    elif shape == "polygon":
        path = [(P.markup_polygon(left, top, right, bottom, s["vertices"], s["shape_rotation"], s["regular"]), True)]
    elif shape == "star":
        path = [(P.markup_star(left, top, right, bottom, s["vertices"], s["shape_rotation"], s["inner_radius"],
                               s["regular"]), True)]
    elif shape in ("arrow", "double_arrow"):
        path = [(P.markup_filled_arrow(left, top, right, bottom, shape == "double_arrow", s["arrow_size"],
                                       s["arrow_angle"]), True)]
    else:
        path = P.markup_line_arrow(left, top, right, bottom, shape == "double_line_arrow",
                                   shape in ("line_arrow", "double_line_arrow"), s["arrow_size"], s["arrow_angle"])
    is_fill = shape in ("arrow", "double_arrow") or (closed5 and filled)
    return path, is_fill, uses_stroke


@op("draw.shape_layer", "Shape layer (ImageToolbox Markup)", params=[
    Choice("shape", "star", choices=list(LAYER_SHAPES), labels=LAYER_SHAPES, title="Shape",
           description="Markup shape; arrow / double_arrow are filled block arrows"),
    Bool("filled", True, title="Filled",
         description="rect/oval/triangle/polygon/star: filled shape instead of an outline"),
    Point("position", (0.5, 0.5), min=-1, max=2, title="Position", description="Layer centre (normalised)"),
    Float("width", 0.35, min=0.001, max=4, step=0.005, title="Width",
          description="Shape box width as a fraction of the shorter image side"),
    Float("height", 0.35, min=0.001, max=4, step=0.005, title="Height",
          description="Shape box height as a fraction of the shorter image side (line arrows: head size)"),
    Float("rotation", 0, min=-360, max=360, step=0.5, unit="deg", title="Layer rotation",
          description="Clockwise rotation of the whole layer"),
    Bool("flip_h", False, title="Flip horizontally"), Bool("flip_v", False, title="Flip vertically"),
    Float("stroke_width", 0.016, min=0, max=0.5, step=0.001, title="Line width",
          description="Outline / line width as a fraction of the shorter side (ImageToolbox: 16 px of the editor "
                      "canvas)"),
    Color("color", "#000000", title="Colour"),
    Color("fill_color", "#00000000", title="Fill colour", description="Outlined shapes: interior colour"),
    Float("arrow_size", 3, min=0.5, max=8, step=0.1, title="Arrow head scale"),
    Float("arrow_angle", 150, min=90, max=180, step=0.5, unit="deg", title="Arrow angle"),
    Int("vertices", 5, min=3, max=24, title="Vertices"),
    Float("inner_radius", 0.5, min=0, max=1, step=0.01, title="Inner radius ratio"),
    Bool("regular", False, title="Regular shape"),
    Float("shape_rotation", 0, min=0, max=360, step=1, unit="deg", title="Shape angle",
          description="rect / polygon / star rotation inside the box"),
    Float("corner_radius", 0, min=0, max=0.5, step=0.01, title="Corner radius",
          description="rect: corner radius as a fraction of the shorter box side"),
    Bool("shadow", False, title="Drop shadow"),
    Color("shadow_color", "#000000", title="Shadow colour"),
    Float("shadow_dx", 0, min=-0.2, max=0.2, step=0.001, title="Shadow offset X",
          description="Fraction of the shorter side"),
    Float("shadow_dy", 0.006, min=-0.2, max=0.2, step=0.001, title="Shadow offset Y",
          description="Fraction of the shorter side"),
    Float("shadow_blur", 0.012, min=0, max=0.2, step=0.001, title="Shadow blur",
          description="Blur radius as a fraction of the shorter side"),
    Float("opacity", 1, min=0, max=1, step=0.01, title="Opacity"),
    Choice("blend", "normal", choices=list(BLEND_MODES), title="Blend mode"),
], alpha="full", source={**SOURCE, "feature": "Markup layers (shape layer)",
                         "files": "feature/markup-layers/**/ShapeLayerRenderer.kt, LayersRenderer.kt"},
    tags=["shape", "markup", "layer", "arrow", "block arrow", "star", "polygon", "triangle", "drop shadow",
          "sticker", "annotate"],
    examples=[{"shape": "arrow", "position": [0.3, 0.7], "width": 0.3, "height": 0.12, "rotation": -30,
               "color": "#FF3B30", "shadow": True}])
def shape_layer(img, ctx, shape, filled, position, width, height, rotation, flip_h, flip_v, stroke_width, color,
                fill_color, shadow, shadow_color, shadow_dx, shadow_dy, shadow_blur, opacity, blend, **s):
    """ImageToolbox Markup shape layer: a shape in a box (block arrows, line arrows, rect, oval, triangle,
    polygon, star; filled or outlined with a fill colour) with an optional blurred drop shadow, rotated,
    flipped and blended over the image."""
    h, w = img.shape[:2]
    S = min(h, w)
    sw = max(stroke_width * S, 1.0)
    bw, bh = max(width * S, 1.0), max(height * S, 1.0)
    path, is_fill, uses_stroke = _layer_shape_paths(shape, filled, bw, bh, sw, s)
    if not path:
        return img
    allp = np.concatenate([p for p, _ in path])
    vc = (allp.min(0) + allp.max(0)) / 2
    # LayersRenderer: the (content + shadow padding) box is centred on the layer position
    blur_px, dx, dy = shadow_blur * S, shadow_dx * S, shadow_dy * S
    shift = np.zeros(2)
    if shadow:
        shift = np.array([(max(blur_px - dx, 0) - max(blur_px + dx, 0)) / 2,
                          (max(blur_px - dy, 0) - max(blur_px + dy, 0)) / 2])
    c = np.array([position[0] * w, position[1] * h])
    t = math.radians(rotation)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    flip = np.array([-1.0 if flip_h else 1.0, -1.0 if flip_v else 1.0])

    def world(pts, offset=(0.0, 0.0)):
        return c + ((pts - vc + shift + offset) * flip) @ rot.T

    pad = R.blur_margin(blur_px) + 2 if shadow else 2

    def layers(offset=(0.0, 0.0)):
        """drawShapeLayer: filled modes draw the fill only; the others an optional fill colour (outlined
        shapes), then the stroke (round caps and joins).  Returns [(coverage, box, rgba)]."""
        out = []
        if is_fill or (shape in SHAPES and fill_color[3] > 0):
            cb = R.rasterize([world(p, offset) for p, _ in path], h, w, pad)
            if cb is not None:
                out.append((*cb, color if is_fill else fill_color))
        if uses_stroke:
            polys = []
            for p, closed in path:
                polys.extend(R.stroke_outline(world(p, offset), closed, sw, "round", "round"))
            cb = R.rasterize(polys, h, w, pad)
            if cb is not None:
                out.append((*cb, color))
        return out

    content = layers()
    shade = layers((dx, dy)) if shadow else []
    box = None
    for cov, b, _ in content + shade:
        box = R.union_box(box, b)
    if box is None:
        return img
    lh, lw = box[2] - box[0], box[3] - box[1]
    layer = np.zeros((lh, lw, 4), F32)  # premultiplied layer bitmap
    if shade:
        a = np.zeros((lh, lw), F32)
        for cov, b, rgba in shade:  # extractAlpha of the rendered content
            k = R.place(cov, b, box) * float(rgba[3])
            a = a + k * (1 - a)
        a = R.blur(a, blur_px)  # BlurMaskFilter(blurRadius, NORMAL), tinted with SRC_IN
        R.blend_solid(layer, a, float(shadow_color[0]), float(shadow_color[1]), float(shadow_color[2]),
                      float(shadow_color[3]))
    for cov, b, rgba in content:
        R.blend_solid(layer, R.place(cov, b, box), float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
    straight = _unpremul(layer)
    out = np.array(img, F32)
    y0, x0, y1, x1 = box
    out[y0:y1, x0:x1] = layer_over(out[y0:y1, x0:x1], straight, blend, opacity)
    return out
