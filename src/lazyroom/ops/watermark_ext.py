"""ImageToolbox watermarking ports (category ``draw``) that ``draw.watermark`` does not cover.

ImageToolbox (https://github.com/T8RIN/ImageToolbox, Apache-2.0) ``feature/watermarking`` offers Text and Image
watermarks (drawn by the AndroidWM library, ImageToolboxLibs ``libs/androidwm``, Apache-2.0), text Stamps,
time Stamps and digital (hidden) watermarks.  ``draw.watermark`` has a different tiled layout (spacing,
half-offset rows) and no blend modes, so the faithful layouts are ported here:

* ``draw.repeating_text``      - ImageToolbox "Text" watermark (repeated grid or single, rotation, blend mode)
* ``draw.repeating_image``     - ImageToolbox "Image" watermark
* ``draw.stamp_watermark``     - ImageToolbox "Stamp" (text at one of 9 positions with padding)
* ``draw.timestamp_watermark`` - ImageToolbox "Time stamp" (SimpleDateFormat pattern)

Hidden (LSB / frequency-domain) watermarks are file-level tools (``watermark_hidden_embed`` /
``watermark_hidden_extract`` in ``tools/itb_extras.py``) because they must be saved losslessly.
"""
from __future__ import annotations

from ..core.params import Bool, Choice, Color, File, Float, Point, Text
from ..core.registry import op
from . import _wm_render as R

_SRC = {"project": "ImageToolbox", "feature": "Watermarking", "upstream": "AndroidWM (ImageToolboxLibs)",
        "license": "Apache-2.0", "url": "https://github.com/T8RIN/ImageToolbox"}

_COMMON_TAIL = [
    Float("rotation", 45, min=0, max=360, step=1, unit="deg", title="Angle",
          description="Clockwise rotation of the watermark bitmap (ImageToolbox default 45)"),
    Float("opacity", 0.5, min=0, max=1, step=0.01, title="Alpha"),
    Bool("repeat", True, title="Repeat watermark",
         description="Tile the (rotated) watermark over the whole image from the top-left corner"),
    Point("position", (0.0, 0.0), min=0, max=1,
          description="Top-left corner of the watermark (fractions of the image) when repeat is off"),
    Choice("blend", "src_over", choices=R.BLEND_MODES, title="Overlay mode",
           description="ImageToolbox BlendingMode (Skia Porter-Duff / blend modes)"),
]


def _place(img, layer, rotation, repeat, position, blend):
    canvas = R.to_premul(img)
    layer = R.rotate_layer(layer, rotation)
    if repeat:
        out = R.tile_layer(canvas, layer, blend)
    else:
        h, w = img.shape[:2]
        out = R.draw_layer(canvas, layer, int(round(position[0] * w)), int(round(position[1] * h)), blend)
    return R.from_premul(out, img.shape[2] == 4)


@op("draw.repeating_text", "Text watermark (ImageToolbox)", params=[
    Text("text", "Watermark", max_length=2000, multiline=True),
    Float("size", 0.1, min=0.01, max=1, step=0.01, title="Watermark size",
          description="Width of the text block as a fraction of the image width"),
    Color("color", "#000000", title="Text color", description="Its alpha is replaced by the watermark alpha"),
    Color("background", "#00000000", title="Background color", description="Fills the text bitmap"),
    Text("font", "sans-serif", max_length=512, description="Font family or .ttf/.otf path"),
    *_COMMON_TAIL,
], alpha="full", source=_SRC, tags=["watermark", "text", "repeat", "tile", "copyright", "imagetoolbox"])
def repeating_text(img, ctx, text, size, color, background, font, rotation, opacity, repeat, position, blend):
    """ImageToolbox Text watermark (AndroidWM ``WatermarkText``): the text bitmap is ``size`` x image width wide,
    rotated about its centre, then tiled on a grid from the image origin (``BitmapShader`` REPEAT) or drawn once
    at ``position``.  As in AndroidWM the alpha is applied to the text paint *and* to the bitmap paint, so text
    opacity is alpha^2 while the background colour gets alpha once."""
    w = img.shape[1]
    layer = R.text_bitmap(text, size, w, color, background, font, opacity) * float(opacity)
    return _place(img, layer, rotation, repeat, position, blend)


@op("draw.repeating_image", "Image watermark (ImageToolbox)", params=[
    File("image", "", file_kind="image", required=True, description="Watermark image (logo)"),
    Float("size", 0.1, min=0.01, max=1, step=0.01, title="Watermark size",
          description="Watermark width as a fraction of the image width"),
    *_COMMON_TAIL,
], alpha="full", source=_SRC, tags=["watermark", "logo", "image", "repeat", "tile", "imagetoolbox"])
def repeating_image(img, ctx, image, size, rotation, opacity, repeat, position, blend):
    """ImageToolbox Image watermark (AndroidWM ``WatermarkImage``): the logo is scaled to ``size`` x image width
    (aspect kept), rotated, then tiled from the origin or drawn once; alpha applied once."""
    from ..core.io import load_image

    logo, _ = load_image(ctx.resolve_file(image, "watermark image"))
    h, w = img.shape[:2]
    lw = max(1, int(round(w * size)))
    lh = max(1, int(round(logo.shape[0] * lw / logo.shape[1])))
    layer = R.resize_layer(R.to_premul(logo), lw, lh) * float(opacity)
    return _place(img, layer, rotation, repeat, position, blend)


_STAMP_PARAMS = [
    Choice("position", "bottom_right", choices=R.POSITIONS),
    Float("padding", 20, min=0, max=50, step=1, unit="px",
          description="Horizontal padding; vertical padding is 0.7x (ImageToolbox: padding - 6*padding/20)"),
    Float("size", 0.2, min=0.01, max=1, step=0.01, title="Watermark size",
          description="Width of the stamp (incl. padding) as a fraction of the image width"),
    Color("color", "#000000", title="Text color"),
    Color("background", "#00000000", title="Background color",
          description="Rounded (12 px) box behind the text when not transparent"),
    Text("font", "sans-serif", max_length=512),
    Float("opacity", 0.5, min=0, max=1, step=0.01, title="Alpha", description="Text alpha (the box is not affected)"),
    Choice("blend", "src_over", choices=R.BLEND_MODES, title="Overlay mode"),
]


_POS = {"top_left": (0, 0), "top_center": (1, 0), "top_right": (2, 0), "center_left": (0, 1), "center": (1, 1),
        "center_right": (2, 1), "bottom_left": (0, 2), "bottom_center": (1, 2), "bottom_right": (2, 2)}


def _stamp(img, ctx, text, position, padding, size, color, background, font, opacity, blend):
    """AndroidWatermarkApplier.drawStamp."""
    h, w = img.shape[:2]
    hp = ctx.px(padding)
    vp = hp - 6.0 * hp / 20.0
    base = R.text_bitmap(text, size, w, color, background, font, opacity)
    ph, pw = base.shape[:2]
    tw, th = int(round(pw - 2 * hp)), int(round(ph - 2 * vp))
    tw, th = (tw if tw > 0 else pw), (th if th > 0 else ph)
    mx = max(tw, th)  # ImageScaler.scaleImage(ResizeType.Flexible) -> flexibleResize(ResizeAnchor.Default)
    if ph >= pw:
        nw, nh = int(mx * pw / ph), mx
    else:
        nw, nh = mx, int(mx * ph / pw)
    layer = R.resize_layer(base, nw, nh)
    if background[3] > 0:
        layer = R.round_corners(layer, ctx.px(12))
    lh, lw = layer.shape[:2]
    hpos, vpos = _POS[position]  # CanvasUtils.drawBitmap(position, horizontalPadding, verticalPadding)
    left = hp if hpos == 0 else (w - lw) / 2 if hpos == 1 else w - lw - hp
    top = vp if vpos == 0 else (h - lh) / 2 if vpos == 1 else h - lh - vp
    out = R.draw_layer(R.to_premul(img), layer, int(round(left)), int(round(top)), blend)
    return R.from_premul(out, img.shape[2] == 4)


@op("draw.stamp_watermark", "Stamp watermark (ImageToolbox)", params=[
    Text("text", "Stamp", max_length=2000, multiline=True), *_STAMP_PARAMS,
], alpha="full", source=_SRC, tags=["watermark", "stamp", "text", "corner", "imagetoolbox"])
def stamp_watermark(img, ctx, text, position, padding, size, color, background, font, opacity, blend):
    """ImageToolbox Stamp: the text bitmap (``size`` x width) is shrunk by twice the padding and placed at one of
    nine positions with ``padding`` horizontally and 0.7 x padding vertically."""
    return _stamp(img, ctx, text, position, padding, size, color, background, font, opacity, blend)


@op("draw.timestamp_watermark", "Time stamp watermark (ImageToolbox)", params=[
    Text("format", "dd/MM/yyyy HH:mm", max_length=200,
         description="java.text.SimpleDateFormat pattern (English names); an invalid pattern prints epoch millis "
                     "like ImageToolbox"),
    Text("date", "", max_length=64,
         description="Date to print: EXIF 'YYYY:MM:DD HH:MM:SS' (e.g. DateTimeOriginal from metadata_read), "
                     "ISO 8601 or epoch milliseconds. Empty = current time at render (ImageToolbox behaviour)"),
    *_STAMP_PARAMS,
], alpha="full", source=_SRC, deterministic=False,
    tags=["watermark", "timestamp", "date", "time", "stamp", "imagetoolbox"])
def timestamp_watermark(img, ctx, format, date, position, padding, size, color, background, font, opacity, blend):
    """ImageToolbox Time stamp: ImageToolbox prints the current time (``timestamp(format)``), not the EXIF date;
    pass ``date`` explicitly (EXIF DateTimeOriginal from ``metadata_read``) for capture dates or reproducible
    renders."""
    text = R.timestamp_text(format, date)
    return _stamp(img, ctx, text, position, padding, size, color, background, font, opacity, blend)


__all__ = ["repeating_text", "repeating_image", "stamp_watermark", "timestamp_watermark"]
