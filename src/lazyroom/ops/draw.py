"""Overlays and annotations (category ``draw``): text, watermarks, shapes, images.

All positions are normalised (0..1) and sizes are relative to the image, so a
step renders identically on previews and full-resolution exports.
"""
from __future__ import annotations

import functools
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..core.blend import BLEND_MODES, layer_over
from ..core.errors import ErrorCode, LazyRoomError
from ..core.params import Bool, Choice, Color, File, Float, Int, Point, Points, Text
from ..core.registry import op
from ._util import F32

ANCHORS = ["top_left", "top", "top_right", "left", "center", "right", "bottom_left", "bottom", "bottom_right"]


@functools.lru_cache(maxsize=128)
def find_font(name: str, bold: bool = False, italic: bool = False) -> str | None:
    """Resolve a font family name or path to a font file (fontconfig on Linux, Fonts dir on Windows)."""
    if name and os.path.exists(name):
        return name
    style = " ".join(s for s, on in (("Bold", bold), ("Italic", italic)) if on)
    if sys.platform != "win32" and shutil.which("fc-match"):
        try:
            q = f"{name or 'sans-serif'}:style={style or 'Regular'}"
            r = subprocess.run(["fc-match", "-f", "%{file}", q], capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout and os.path.exists(r.stdout):
                return r.stdout
        except Exception:
            pass
    if sys.platform == "win32":
        windir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        cand = {"": "segoeui", "sans-serif": "segoeui", "serif": "times", "monospace": "consola"}.get(
            (name or "").lower(), (name or "segoeui").lower().replace(" ", ""))
        suffix = ("bi" if bold and italic else "b" if bold else "i" if italic else "")
        for fn in (f"{cand}{suffix}.ttf", f"{cand}.ttf", f"{cand}.otf"):
            if (windir / fn).exists():
                return str(windir / fn)
    return None


def load_font(name: str, size_px: int, bold: bool = False, italic: bool = False) -> ImageFont.ImageFont:
    path = find_font(name, bold, italic)
    if path:
        try:
            return ImageFont.truetype(path, size_px)
        except OSError:
            pass
    return ImageFont.load_default(size=size_px)


def _anchor_frac(anchor: str) -> tuple[float, float]:
    """Fractions (fx, fy) of a box that sit on the anchor point."""
    fx = 0.0 if "left" in anchor else 1.0 if "right" in anchor else 0.5
    fy = 0.0 if anchor.startswith("top") else 1.0 if anchor.startswith("bottom") else 0.5
    return fx, fy


def _to_rgba_array(layer: Image.Image) -> np.ndarray:
    return np.asarray(layer, dtype=np.float32) / 255.0


def render_text_layer(w: int, h: int, *, text: str, font: str, size: float, color, position, anchor: str,
                      align: str, rotation: float, bold: bool, italic: bool, stroke_width: float, stroke_color,
                      shadow: float, shadow_color, background, padding: float, line_spacing: float,
                      max_width: float = 0.0) -> np.ndarray:
    px = max(4, int(round(size * h)))
    f = load_font(font, px, bold, italic)
    sw = int(round(stroke_width * px))
    if max_width > 0:
        text = _wrap(text, f, max_width * w)
    tmp = ImageDraw.Draw(Image.new("L", (1, 1)))
    bbox = tmp.multiline_textbbox((0, 0), text, font=f, align=align, stroke_width=sw,
                                  spacing=int(px * (line_spacing - 1) + px * 0.15))
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = int(round(padding * px))
    margin = int(px * 0.5 + abs(shadow) * px * 2) + pad
    lw, lh = tw + 2 * margin, th + 2 * margin
    layer = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    to8 = lambda c: tuple(int(round(v * 255)) for v in c)
    if background[3] > 0:
        d.rounded_rectangle([margin - pad, margin - pad, margin + tw + pad, margin + th + pad],
                            radius=int(pad * 0.6), fill=to8(background))
    origin = (margin - bbox[0], margin - bbox[1])
    spacing = int(px * (line_spacing - 1) + px * 0.15)
    if shadow:
        sh = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
        ImageDraw.Draw(sh).multiline_text((origin[0] + shadow * px * 0.6, origin[1] + shadow * px * 0.6), text,
                                          font=f, fill=to8(shadow_color), align=align, spacing=spacing,
                                          stroke_width=sw, stroke_fill=to8(shadow_color))
        sh = sh.filter(ImageFilter.GaussianBlur(radius=abs(shadow) * px * 0.3))
        layer = Image.alpha_composite(layer, sh)
        d = ImageDraw.Draw(layer)
    d.multiline_text(origin, text, font=f, fill=to8(color), align=align, spacing=spacing,
                     stroke_width=sw, stroke_fill=to8(stroke_color))
    if rotation:
        layer = layer.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
        box_w, box_h, inset = layer.size[0], layer.size[1], 0
    else:
        box_w, box_h, inset = tw + 2 * pad, th + 2 * pad, margin - pad
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    fx, fy = _anchor_frac(anchor)
    x = position[0] * w - fx * box_w - inset
    y = position[1] * h - fy * box_h - inset
    canvas.paste(layer, (int(round(x)), int(round(y))), layer)
    return _to_rgba_array(canvas)


def _wrap(text: str, font, max_px: float) -> str:
    lines = []
    for para in text.split("\n"):
        words, cur = para.split(" "), ""
        for wd in words:
            cand = (cur + " " + wd).strip()
            if font.getlength(cand) <= max_px or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = wd
        lines.append(cur)
    return "\n".join(lines)


TEXT_PARAMS = [
    Text("text", "Text", max_length=5000, multiline=True),
    Text("font", "sans-serif", max_length=512, description="Font family name (system fonts) or path to a .ttf/.otf"),
    Float("size", 0.06, min=0.005, max=1, step=0.005, description="Text height as a fraction of image height"),
    Color("color", "#FFFFFF"),
    Point("position", (0.5, 0.9), min=-1, max=2, description="Anchor point (normalized)"),
    Choice("anchor", "bottom", choices=ANCHORS, description="Which part of the text box sits on the position"),
    Choice("align", "center", choices=["left", "center", "right"]),
    Float("rotation", 0, min=-180, max=180, unit="deg"),
    Bool("bold", False), Bool("italic", False),
    Float("stroke_width", 0, min=0, max=0.3, step=0.005, description="Outline width relative to text size"),
    Color("stroke_color", "#000000"),
    Float("shadow", 0, min=-0.3, max=0.3, step=0.01, description="Drop shadow offset relative to text size"),
    Color("shadow_color", "#000000AA"),
    Color("background", "#00000000", description="Box behind the text (transparent = none)"),
    Float("padding", 0.3, min=0, max=3, step=0.05, description="Box padding relative to text size"),
    Float("line_spacing", 1.0, min=0.5, max=3, step=0.05),
    Float("max_width", 0, min=0, max=1, step=0.01, description="Wrap text to this fraction of image width (0 = no wrap)"),
    Float("opacity", 1.0, min=0, max=1, step=0.01),
    Choice("blend", "normal", choices=list(BLEND_MODES)),
]


@op("draw.text", "Text", params=TEXT_PARAMS, alpha="full",
    tags=["text", "caption", "title", "label", "typography", "overlay", "annotation"],
    examples=[{"text": "Summer 1974", "size": 0.05, "position": [0.95, 0.95], "anchor": "bottom_right",
               "color": "#FFB347", "font": "monospace"}])
def text(img, ctx, opacity, blend, **kw):
    """Render text with optional outline, shadow and background box."""
    h, w = img.shape[:2]
    layer = render_text_layer(w, h, **kw)
    return layer_over(img, layer, blend, opacity)


@op("draw.watermark", "Watermark", params=[
    Text("text", "© LazyRoom", max_length=500),
    File("image", "", file_kind="image", description="Logo image (used instead of text when set)"),
    Choice("layout", "single", choices=["single", "tiled"]),
    Choice("anchor", "bottom_right", choices=ANCHORS),
    Float("margin", 0.03, min=0, max=0.5, step=0.005),
    Float("size", 0.05, min=0.005, max=1, step=0.005, description="Text height or logo width (fraction)"),
    Float("rotation", 0, min=-180, max=180, unit="deg"),
    Float("spacing", 0.25, min=0.02, max=2, step=0.01, description="Tile spacing (tiled layout)"),
    Color("color", "#FFFFFF"), Float("opacity", 0.5, min=0, max=1, step=0.01),
    Text("font", "sans-serif", max_length=512),
], alpha="full", tags=["watermark", "copyright", "logo", "branding", "protect"])
def watermark(img, ctx, text, image, layout, anchor, margin, size, rotation, spacing, color, opacity, font):
    """Text or logo watermark, single or tiled across the image."""
    h, w = img.shape[:2]
    if image:
        from ..core.io import load_image
        from ..core.image import ensure_rgba

        logo, _ = load_image(ctx.resolve_file(image, "watermark image"))
        logo = ensure_rgba(logo)
        lw = max(2, int(size * w))
        lh = max(2, int(logo.shape[0] * lw / logo.shape[1]))
        tile = Image.fromarray((np.clip(cv2.resize(logo, (lw, lh), interpolation=cv2.INTER_AREA), 0, 1) * 255).astype(np.uint8), "RGBA")
    else:
        px = max(4, int(size * h))
        f = load_font(font, px)
        bb = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=f)
        tile = Image.new("RGBA", (bb[2] - bb[0] + 4, bb[3] - bb[1] + 4), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((2 - bb[0], 2 - bb[1]), text, font=f,
                                  fill=tuple(int(c * 255) for c in color[:3]) + (255,))
    if rotation:
        tile = tile.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    tw, th = tile.size
    if layout == "tiled":
        step_x = int(tw + spacing * w)
        step_y = int(th + spacing * h)
        for row, y in enumerate(range(-th, h + th, max(4, step_y))):
            off = (step_x // 2) * (row % 2)
            for x in range(-tw - off, w + tw, max(4, step_x)):
                canvas.paste(tile, (x, y), tile)
    else:
        fx, fy = _anchor_frac(anchor)
        mx, my = margin * w, margin * h
        px_ = mx + fx * (w - 2 * mx) - fx * tw
        py_ = my + fy * (h - 2 * my) - fy * th
        canvas.paste(tile, (int(round(px_)), int(round(py_))), tile)
    return layer_over(img, _to_rgba_array(canvas), "normal", opacity)


@op("draw.shape", "Shape / annotation", params=[
    Choice("shape", "rectangle", choices=["rectangle", "ellipse", "line", "arrow", "polygon", "polyline"]),
    Points("points", [[0.3, 0.3], [0.7, 0.7]], min_items=2, max_items=1000, min=-1, max=2,
           description="rectangle/ellipse/line/arrow: two corner/end points; polygon/polyline: vertices"),
    Color("stroke", "#FF3B30"), Float("stroke_width", 0.006, min=0, max=0.2, step=0.001,
                                      description="Relative to the shorter image side"),
    Color("fill", "#00000000"), Float("corner_radius", 0, min=0, max=0.5, step=0.005),
    Bool("dashed", False), Float("opacity", 1.0, min=0, max=1, step=0.01),
], alpha="full", tags=["annotate", "arrow", "rectangle", "circle", "highlight", "markup", "line", "draw"])
def shape(img, ctx, shape, points, stroke, stroke_width, fill, corner_radius, dashed, opacity):
    """Draw vector shapes for annotation (anti-aliased, 4x supersampled)."""
    h, w = img.shape[:2]
    S = 4 if w * h < 8_000_000 else 2
    W, H = w * S, h * S
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    to8 = lambda c: tuple(int(round(v * 255)) for v in c)
    pts = [(p[0] * W, p[1] * H) for p in points]
    sw = max(1, int(round(stroke_width * min(W, H)))) if stroke_width > 0 else 0
    st = to8(stroke) if sw else None
    fl = to8(fill) if fill[3] > 0 else None
    if shape in ("rectangle", "ellipse"):
        (x0, y0), (x1, y1) = pts[0], pts[1]
        box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        if shape == "ellipse":
            d.ellipse(box, fill=fl, outline=st, width=sw)
        else:
            d.rounded_rectangle(box, radius=corner_radius * min(W, H), fill=fl, outline=st, width=sw)
    elif shape in ("line", "arrow", "polyline"):
        seq = pts if shape == "polyline" else pts[:2]
        if dashed:
            for (a, b) in zip(seq, seq[1:]):
                L = math.dist(a, b)
                n = max(1, int(L / (sw * 4)))
                for i in range(0, n, 2):
                    t0, t1 = i / n, min(1, (i + 1) / n)
                    d.line([(a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                            (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)], fill=st, width=sw)
        else:
            d.line(seq, fill=st, width=sw, joint="curve")
        if shape == "arrow":
            (x0, y0), (x1, y1) = pts[0], pts[1]
            ang = math.atan2(y1 - y0, x1 - x0)
            L = max(sw * 4, math.dist(pts[0], pts[1]) * 0.12)
            head = [(x1, y1), (x1 - L * math.cos(ang - 0.45), y1 - L * math.sin(ang - 0.45)),
                    (x1 - L * math.cos(ang + 0.45), y1 - L * math.sin(ang + 0.45))]
            d.polygon(head, fill=st)
    elif shape == "polygon":
        d.polygon(pts, fill=fl, outline=st, width=sw)
    layer = layer.resize((w, h), Image.Resampling.LANCZOS)
    return layer_over(img, _to_rgba_array(layer), "normal", opacity)


@op("draw.image", "Image overlay / layer", params=[
    File("path", "", file_kind="image", required=True),
    Point("position", (0.5, 0.5), min=-1, max=2, description="Centre of the overlay (normalized)"),
    Float("scale", 0.3, min=0.01, max=4, step=0.01, description="Overlay width as a fraction of image width"),
    Float("rotation", 0, min=-180, max=180, unit="deg"),
    Float("opacity", 1.0, min=0, max=1, step=0.01),
    Choice("blend", "normal", choices=list(BLEND_MODES)),
    Choice("fit", "width", choices=["width", "cover", "contain", "stretch"],
           description="width: use 'scale'; cover/contain/stretch: fit the whole canvas"),
], alpha="full", tags=["overlay", "layer", "composite", "texture overlay", "sticker", "logo", "double exposure"])
def image_overlay(img, ctx, path, position, scale, rotation, opacity, blend, fit):
    """Composite another image (with its alpha) over the current one."""
    from ..core.io import load_image
    from ..core.image import ensure_rgba

    ov, _ = load_image(ctx.resolve_file(path, "overlay image"))
    ov = ensure_rgba(ov)
    h, w = img.shape[:2]
    oh, ow = ov.shape[:2]
    if fit == "stretch":
        nw, nh = w, h
    elif fit in ("cover", "contain"):
        s = (max if fit == "cover" else min)(w / ow, h / oh)
        nw, nh = int(round(ow * s)), int(round(oh * s))
    else:
        nw = max(1, int(round(scale * w)))
        nh = max(1, int(round(oh * nw / ow)))
    ov = cv2.resize(ov, (nw, nh), interpolation=cv2.INTER_AREA if nw < ow else cv2.INTER_CUBIC)
    if rotation:
        M = cv2.getRotationMatrix2D((nw / 2, nh / 2), rotation, 1.0)
        c, s_ = abs(M[0, 0]), abs(M[0, 1])
        bw, bh = int(nh * s_ + nw * c), int(nh * c + nw * s_)
        M[0, 2] += bw / 2 - nw / 2
        M[1, 2] += bh / 2 - nh / 2
        ov = cv2.warpAffine(ov, M, (bw, bh), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0))
        nw, nh = bw, bh
    layer = np.zeros((h, w, 4), F32)
    cx, cy = (w / 2, h / 2) if fit in ("cover", "contain", "stretch") else (position[0] * w, position[1] * h)
    x0, y0 = int(round(cx - nw / 2)), int(round(cy - nh / 2))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(w, x0 + nw), min(h, y0 + nh)
    if dx1 > dx0 and dy1 > dy0:
        layer[dy0:dy1, dx0:dx1] = ov[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    return layer_over(img, layer, blend, opacity)


@op("draw.border", "Border / frame", params=[
    Float("width", 0.03, min=0, max=0.5, step=0.001, description="Relative to the shorter side"),
    Color("color", "#FFFFFF"),
    Float("corner_radius", 0, min=0, max=0.5, step=0.005, description="Rounded outer corners (transparent outside)"),
    Bool("expand", True, description="Add the border outside the image instead of covering it"),
    Float("bottom_extra", 0, min=0, max=1, step=0.01, description="Extra bottom margin (Polaroid style), relative"),
], alpha="full", geometry=True, tags=["border", "frame", "polaroid", "matte", "passe-partout", "rounded corners"])
def border(img, ctx, width, color, corner_radius, expand, bottom_extra):
    """Solid border/frame with optional rounded corners and a Polaroid-style bottom margin."""
    h, w = img.shape[:2]
    b = int(round(width * min(w, h)))
    extra = int(round(bottom_extra * min(w, h)))
    col = np.array(color[:3], F32)
    if expand:
        nh, nw = h + 2 * b + extra, w + 2 * b
        out = np.empty((nh, nw, 4), F32)
        out[..., :3] = col
        out[..., 3] = color[3]
        src = img if img.shape[2] == 4 else np.dstack([img, np.ones((h, w), F32)])
        a = src[..., 3:4]
        region = out[b:b + h, b:b + w]
        region[..., :3] = src[..., :3] * a + region[..., :3] * (1 - a)
        region[..., 3:4] = a + region[..., 3:4] * (1 - a)
    else:
        out = img.copy() if img.shape[2] == 4 else np.dstack([img, np.ones((h, w), F32)])
        nh, nw = h, w
        m = np.ones((h, w), bool)
        m[b:h - b - extra, b:w - b] = False
        out[m, :3] = col
        out[m, 3] = color[3]
    if corner_radius > 0:
        from .geometry import shape_mask

        out[..., 3] *= shape_mask("rounded", nh, nw, roundness=corner_radius * 2)
    if out[..., 3].min() >= 0.9999:
        out = out[..., :3]
    return np.ascontiguousarray(out)
