"""Rendering helpers for the ImageToolbox watermark ports (``ops/watermark_ext.py``).

Ports of ImageToolbox ``AndroidWatermarkApplier`` and the AndroidWM library it
uses (``com.watermark.androidwm``: ``Watermark``, ``BitmapUtils.textAsBitmap``),
both Apache-2.0.  Layers are float32 *premultiplied* RGBA (Android/Skia
bitmaps are premultiplied), composited with Skia's Porter-Duff / blend-mode
formulas.
"""
from __future__ import annotations

import calendar
import datetime as _dt
import math
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..core.blend import blend_rgb
from ..core.errors import invalid
from .draw import load_font

F32 = np.float32

# ImageToolbox BlendingMode (values 0..28, Skia/Compose order) -> snake_case choice
BLEND_MODES = ["clear", "src", "dst", "src_over", "dst_over", "src_in", "dst_in", "src_out", "dst_out", "src_atop",
               "dst_atop", "xor", "plus", "modulate", "screen", "overlay", "darken", "lighten", "color_dodge",
               "color_burn", "hard_light", "soft_light", "difference", "exclusion", "multiply", "hue", "saturation",
               "color", "luminosity"]
_ADVANCED = {"overlay", "darken", "lighten", "color_dodge", "color_burn", "hard_light", "soft_light", "difference",
             "exclusion", "multiply", "hue", "saturation", "color", "luminosity"}

# core.domain.model.Position (entriesSorted)
POSITIONS = ["top_left", "top_center", "top_right", "center_left", "center", "center_right", "bottom_left",
             "bottom_center", "bottom_right"]


# ---------------------------------------------------------------------------
# premultiplied buffers
# ---------------------------------------------------------------------------

def to_premul(img: np.ndarray) -> np.ndarray:
    if img.shape[2] == 4:
        a = np.clip(img[..., 3:4], 0, 1)
        return np.concatenate([np.clip(img[..., :3], 0, 1) * a, a], -1).astype(F32)
    return np.concatenate([np.clip(img, 0, 1), np.ones(img.shape[:2] + (1,), F32)], -1).astype(F32)


def from_premul(p: np.ndarray, had_alpha: bool) -> np.ndarray:
    a = p[..., 3:4]
    rgb = np.where(a > 1e-6, p[..., :3] / np.maximum(a, 1e-6), 0.0)
    out = np.concatenate([np.clip(rgb, 0, 1), np.clip(a, 0, 1)], -1).astype(F32)
    if not had_alpha and float(out[..., 3].min()) >= 1.0 - 1e-6:
        return np.ascontiguousarray(out[..., :3])
    return out


def composite(dst: np.ndarray, src: np.ndarray, mode: str) -> np.ndarray:
    """Skia ``SkBlendMode`` on premultiplied RGBA float arrays (same shape)."""
    S, Sa = src[..., :3], src[..., 3:4]
    D, Da = dst[..., :3], dst[..., 3:4]
    if mode == "clear":
        return np.zeros_like(dst)
    if mode == "src":
        return src.copy()
    if mode == "dst":
        return dst.copy()
    if mode == "src_over":
        c, a = S + D * (1 - Sa), Sa + Da * (1 - Sa)
    elif mode == "dst_over":
        c, a = D + S * (1 - Da), Da + Sa * (1 - Da)
    elif mode == "src_in":
        c, a = S * Da, Sa * Da
    elif mode == "dst_in":
        c, a = D * Sa, Da * Sa
    elif mode == "src_out":
        c, a = S * (1 - Da), Sa * (1 - Da)
    elif mode == "dst_out":
        c, a = D * (1 - Sa), Da * (1 - Sa)
    elif mode == "src_atop":
        c, a = S * Da + D * (1 - Sa), Da
    elif mode == "dst_atop":
        c, a = D * Sa + S * (1 - Da), Sa
    elif mode == "xor":
        c, a = S * (1 - Da) + D * (1 - Sa), Sa * (1 - Da) + Da * (1 - Sa)
    elif mode == "plus":
        c, a = np.minimum(S + D, 1), np.minimum(Sa + Da, 1)
    elif mode == "modulate":
        c, a = S * D, Sa * Da
    elif mode == "screen":
        c, a = S + D - S * D, Sa + Da - Sa * Da
    elif mode in _ADVANCED:
        sc = np.where(Sa > 1e-6, S / np.maximum(Sa, 1e-6), 0)
        dc = np.where(Da > 1e-6, D / np.maximum(Da, 1e-6), 0)
        b = np.clip(blend_rgb(dc, sc, mode), 0, 1)
        c = S * (1 - Da) + D * (1 - Sa) + Sa * Da * b
        a = Sa + Da - Sa * Da
    else:
        raise invalid(f"Unknown blend mode {mode!r}", valid=BLEND_MODES)
    return np.concatenate([np.clip(c, 0, 1), np.clip(a, 0, 1)], -1).astype(F32)


def draw_layer(canvas: np.ndarray, layer: np.ndarray, x: int, y: int, mode: str) -> np.ndarray:
    """``Canvas.drawBitmap(layer, x, y, paint)``: the blend mode applies inside the bitmap rectangle only."""
    h, w = canvas.shape[:2]
    lh, lw = layer.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + lw), min(h, y + lh)
    if x0 >= x1 or y0 >= y1:
        return canvas
    out = canvas.copy()
    out[y0:y1, x0:x1] = composite(canvas[y0:y1, x0:x1], layer[y0 - y:y1 - y, x0 - x:x1 - x], mode)
    return out


def tile_layer(canvas: np.ndarray, tile: np.ndarray, mode: str) -> np.ndarray:
    """``BitmapShader(REPEAT, REPEAT)`` + ``drawRect(clipBounds)``: the tile grid starts at the canvas origin and the
    blend mode applies to the whole canvas."""
    h, w = canvas.shape[:2]
    th, tw = tile.shape[:2]
    reps = (-(-h // th), -(-w // tw), 1)
    full = np.tile(tile, reps)[:h, :w]
    return composite(canvas, full, mode)


def rotate_layer(layer: np.ndarray, degrees: float) -> np.ndarray:
    """AndroidWM ``adjustPhotoRotation``: ``Matrix.setRotate(deg, w/2, h/2)`` + ``createBitmap(.., filter=true)``.
    The output is the rotated bounding box (clockwise for positive angles, y down)."""
    deg = float(degrees) % 360
    if deg == 0:
        return layer
    h, w = layer.shape[:2]
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    # rotation about the centre in y-down screen coordinates (positive = clockwise)
    cx, cy = w / 2, h / 2
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float64)
    rx = c * (corners[:, 0] - cx) - s * (corners[:, 1] - cy) + cx
    ry = s * (corners[:, 0] - cx) + c * (corners[:, 1] - cy) + cy
    left, top = rx.min(), ry.min()
    nw, nh = max(1, int(round(rx.max() - left))), max(1, int(round(ry.max() - top)))
    m = np.array([[c, -s, -c * cx + s * cy + cx - left],
                  [s, c, -s * cx - c * cy + cy - top]], np.float64)
    return cv2.warpAffine(layer, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0)).astype(F32)


def resize_layer(layer: np.ndarray, w: int, h: int) -> np.ndarray:
    w, h = max(1, int(w)), max(1, int(h))
    if (h, w) == layer.shape[:2]:
        return layer
    interp = cv2.INTER_AREA if w < layer.shape[1] and h < layer.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(layer, (w, h), interpolation=interp).astype(F32)


# ---------------------------------------------------------------------------
# AndroidWM BitmapUtils.textAsBitmap
# ---------------------------------------------------------------------------

def text_bitmap(text: str, size: float, bg_width: int, color, background, font: str,
                text_alpha: float) -> np.ndarray:
    """Premultiplied RGBA bitmap of ``text`` laid out like ``BitmapUtils.textAsBitmap``.

    ImageToolbox rasterises at ``size * 200`` px and then scales the bitmap so that its width is
    ``bg_width * size``; here the text is drawn directly at the final scale (same geometry, sharper glyphs).
    The text colour's own alpha is replaced by
    ``text_alpha`` (``Paint.setAlpha``); the background colour fills the whole bitmap.
    """
    ref = 1000.0
    f_ref = load_font(font, int(ref))
    lines = text.split("\n")
    flat = text.replace("\n", "")
    f0 = size * 200.0
    k0 = f0 / ref
    asc_r, desc_r = f_ref.getmetrics()
    adv0 = f_ref.getlength(flat) * k0 if flat else 0.0
    bb = f_ref.getbbox(flat) if flat else (0, 0, 0, 0)
    bound_w = int(round((bb[2] - bb[0]) * k0)) + 20
    measure = int(adv0)
    w0 = min(bound_w, measure)
    baseline0 = int(asc_r * k0 + 1.0)
    line_h0 = int(baseline0 + desc_r * k0 + 3)
    # StaticLayout(spacingMult = 2, spacingAdd = 2): line pitch below the first line.  ImageToolbox sizes the bitmap
    # for lineCount * line_h0 and so clips later lines; the full block is kept here (documented deviation).
    pitch0 = (asc_r + desc_r) * k0 * 2.0 + 2.0
    h0 = line_h0 + int(math.ceil(pitch0 * (len(lines) - 1))) if len(lines) > 1 else line_h0
    if w0 <= 0 or h0 <= 0:
        w0, h0 = 1, 1
    k = (bg_width * size) / w0
    W, H = max(1, int(round(w0 * k))), max(1, int(round(h0 * k)))
    fs = max(1.0, f0 * k)
    fnt = load_font(font, int(round(fs))) if fs < 4 else _float_font(font, fs)
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    for i, line in enumerate(lines):
        if line:
            d.text((0, (asc_r * k0 + i * pitch0) * k), line, font=fnt, fill=255, anchor="ls")
    cov = np.asarray(mask, F32)[..., None] / 255.0
    bg = np.array(background, F32)
    col = np.array(color, F32)
    ta = cov * float(text_alpha)
    bg_p = np.concatenate([bg[:3] * bg[3], bg[3:4]])
    txt_p = np.concatenate([col[:3], [1.0]]).astype(F32)
    return (txt_p * ta + bg_p * (1 - ta)).astype(F32)


def _float_font(name: str, size: float):
    from PIL import ImageFont

    from .draw import find_font

    path = find_font(name)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return load_font(name, int(round(size)))


def round_corners(layer: np.ndarray, radius: float) -> np.ndarray:
    """coil ``RoundedCornersTransformation(radius)`` (anti-aliased rounded-rectangle mask)."""
    h, w = layer.shape[:2]
    r = min(float(radius), w / 2, h / 2)
    if r <= 0:
        return layer
    ss = 4
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=r * ss, fill=255)
    mask = np.asarray(m.resize((w, h), Image.Resampling.BOX), F32)[..., None] / 255.0
    return (layer * mask).astype(F32)


# ---------------------------------------------------------------------------
# java.text.SimpleDateFormat (Locale.US names)
# ---------------------------------------------------------------------------

_MONTHS = list(calendar.month_name)[1:]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_PATTERN_LETTERS = set("GyYMLwWDdFEuaHkKhmsSzZX")


class DateFormatError(ValueError):
    pass


def _us_week_of_year(d: _dt.datetime) -> int:
    jan1 = _dt.date(d.year, 1, 1)
    offset = (jan1.weekday() + 1) % 7  # Sunday-first
    return (d.timetuple().tm_yday - 1 + offset) // 7 + 1


def java_date_format(pattern: str, d: _dt.datetime) -> str:
    """Format like ``SimpleDateFormat(pattern, Locale.US).format(d)``; raises DateFormatError for bad patterns."""
    out, i, n = [], 0, len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "'":
            j = i + 1
            if j < n and pattern[j] == "'":
                out.append("'")
                i += 2
                continue
            buf = []
            while j < n:
                if pattern[j] == "'":
                    if j + 1 < n and pattern[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    break
                buf.append(pattern[j])
                j += 1
            if j >= n:
                raise DateFormatError("Unterminated quote")
            out.append("".join(buf))
            i = j + 1
            continue
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            if ch not in _PATTERN_LETTERS:
                raise DateFormatError(f"Illegal pattern character '{ch}'")
            j = i
            while j < n and pattern[j] == ch:
                j += 1
            out.append(_field(ch, j - i, d))
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _num(v: int, count: int) -> str:
    return str(v).zfill(count)


def _field(ch: str, count: int, d: _dt.datetime) -> str:
    if ch == "G":
        return "AD" if d.year > 0 else "BC"
    if ch in "yY":
        return _num(d.year % 100, 2) if count == 2 else _num(d.year, count)
    if ch in "ML":
        if count >= 4:
            return _MONTHS[d.month - 1]
        if count == 3:
            return _MONTHS[d.month - 1][:3]
        return _num(d.month, count)
    if ch == "w":
        return _num(_us_week_of_year(d), count)
    if ch == "W":
        first = _dt.date(d.year, d.month, 1)
        return _num((d.day - 1 + (first.weekday() + 1) % 7) // 7 + 1, count)
    if ch == "D":
        return _num(d.timetuple().tm_yday, count)
    if ch == "d":
        return _num(d.day, count)
    if ch == "F":
        return _num((d.day - 1) // 7 + 1, count)
    if ch == "E":
        name = _DAYS[d.weekday()]
        return name if count >= 4 else name[:3]
    if ch == "u":
        return _num(d.isoweekday(), count)
    if ch == "a":
        return "AM" if d.hour < 12 else "PM"
    if ch == "H":
        return _num(d.hour, count)
    if ch == "k":
        return _num(d.hour or 24, count)
    if ch == "K":
        return _num(d.hour % 12, count)
    if ch == "h":
        return _num(d.hour % 12 or 12, count)
    if ch == "m":
        return _num(d.minute, count)
    if ch == "s":
        return _num(d.second, count)
    if ch == "S":
        return _num(d.microsecond // 1000, count)
    off = d.utcoffset() or _dt.timedelta(0)
    mins = int(off.total_seconds() // 60)
    sign = "+" if mins >= 0 else "-"
    hh, mm = divmod(abs(mins), 60)
    if ch == "z":
        name = d.tzname()
        return name if name and not name.startswith(("UTC+", "UTC-")) else f"GMT{sign}{hh:02d}:{mm:02d}"
    if ch == "Z":
        return f"{sign}{hh:02d}{mm:02d}"
    if ch == "X":
        if mins == 0:
            return "Z"
        return f"{sign}{hh:02d}" if count == 1 else (f"{sign}{hh:02d}{mm:02d}" if count == 2 else f"{sign}{hh:02d}:{mm:02d}")
    raise DateFormatError(ch)


_EXIF_RE = re.compile(r"^\s*(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?\s*$")


def parse_date(value: str) -> _dt.datetime:
    """'' -> now (local time, ImageToolbox behaviour); EXIF 'YYYY:MM:DD HH:MM:SS'; ISO 8601; epoch milliseconds."""
    v = (value or "").strip()
    if not v:
        return _dt.datetime.now().astimezone()
    m = _EXIF_RE.match(v)
    if m:
        frac = (m.group(7) or "0")[:6].ljust(6, "0")
        return _dt.datetime(*(int(g) for g in m.groups()[:6]), int(frac))
    if re.fullmatch(r"-?\d{9,}", v):
        return _dt.datetime.fromtimestamp(int(v) / 1000).astimezone()
    try:
        return _dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        raise invalid(f"Unrecognised date {value!r}", param="date",
                      hint="Use EXIF 'YYYY:MM:DD HH:MM:SS', ISO 8601 or epoch milliseconds; empty = now.") from None


def timestamp_text(fmt: str, date: str) -> str:
    """ImageToolbox ``timestamp(format)``: SimpleDateFormat with the device locale; an invalid pattern falls back to
    the epoch milliseconds (``Date().time.toString()``)."""
    d = parse_date(date)
    try:
        return java_date_format(fmt, d)
    except DateFormatError:
        return str(int(d.timestamp() * 1000))
