"""Compare-screen renders and the animated before/after GIF (ImageToolbox feature/compare, Apache-2.0).

* ``CompareAnimationParams`` (domain/CompareAnimationParams.kt): duration 2..8 s, 500 ms holds at both
  ends, 50 ms travel frames, smoothstep easing, ``outputSize`` (fit within maxSize, never upscale),
  sizes 720/1080/2048/4096.
* ``CompareAnimationRenderer`` (data/CompareAnimationRenderer.kt): white canvas, "before" fitted and
  centred, "after" drawn over it clipped to ``[0, edge]`` (left / top part), white divider
  ``max(1, 0.004 * min(w, h))`` px, labels in a rounded ``argb(170, 0, 0, 0)`` box with bold white text
  of size ``0.045 * min(w, h)`` ("After" top-start, "Before" bottom-end).
* ``writeGif`` (data/CompareAnimationEncoder.kt): repeat forever, dispose = 1 (do not dispose), the first
  frame is full size and every later frame only covers ``frameBounds`` (the strip between the previous
  and the current divider position, +- divider width).  We assemble the GIF ourselves so frame count,
  per-frame bounds and delays are exactly those of the app; pixels are quantised once to a shared
  256-colour palette (Pillow median cut + Floyd-Steinberg) instead of the app's per-frame encoder.
"""
from __future__ import annotations

import io
import math
import struct
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

F32 = np.float32
HOLD_MILLIS = 500
FRAME_MILLIS = 50
SIZES = [720, 1080, 2048, 4096]


@dataclass
class AnimationParams:
    duration_seconds: int = 4
    is_vertical: bool = False
    show_labels: bool = True
    max_size: int = 2048

    @property
    def duration_millis(self) -> int:
        return min(8, max(2, self.duration_seconds)) * 1000

    def position_at(self, time_millis: int) -> float:
        d = self.duration_millis
        t = time_millis % d
        travel = (d - 2 * HOLD_MILLIS) // 2
        if t < HOLD_MILLIS:
            p = 0.0
        elif t < HOLD_MILLIS + travel:
            p = float(np.float32(t - HOLD_MILLIS) / np.float32(travel))
        elif t < 2 * HOLD_MILLIS + travel:
            p = 1.0
        else:
            p = float(np.float32(1) - np.float32(t - 2 * HOLD_MILLIS - travel) / np.float32(travel))
        p32 = np.float32(p)
        return float(p32 * p32 * (np.float32(3) - np.float32(2) * p32))

    def frames(self) -> list[tuple[float, int]]:
        out, t = [], 0
        travel = (self.duration_millis - 2 * HOLD_MILLIS) // 2
        for _ in range(2):
            out.append((self.position_at(t), HOLD_MILLIS))
            t += HOLD_MILLIS
            for _ in range(travel // FRAME_MILLIS):
                out.append((self.position_at(t), FRAME_MILLIS))
                t += FRAME_MILLIS
        return out


def output_size(width: int, height: int, max_size: int = 2048) -> tuple[int, int]:
    scale = min(np.float32(max(1, max_size)) / np.float32(max(width, height)), np.float32(1))
    return max(1, int(math.floor(float(np.float32(width * scale)) + 0.5))), \
        max(1, int(math.floor(float(np.float32(height * scale)) + 0.5)))


def _fit_rect(iw: int, ih: int, w: int, h: int) -> tuple[float, float, float, float]:
    s = min(w / iw, h / ih)
    left, top = (w - iw * s) / 2, (h - ih * s) / 2
    return left, top, iw * s, ih * s


def draw_fitted(canvas: np.ndarray, img: np.ndarray) -> None:
    """``drawBitmap(img, null, fitRect)`` on an RGB float canvas (alpha composited)."""
    h, w = canvas.shape[:2]
    left, top, dw, dh = _fit_rect(img.shape[1], img.shape[0], w, h)
    x0, y0 = int(round(left)), int(round(top))
    x1, y1 = int(round(left + dw)), int(round(top + dh))
    if x1 <= x0 or y1 <= y0:
        return
    interp = cv2.INTER_AREA if (x1 - x0) < img.shape[1] else cv2.INTER_LINEAR
    r = cv2.resize(np.ascontiguousarray(img, dtype=F32), (x1 - x0, y1 - y0), interpolation=interp)
    if r.ndim == 2:
        r = r[..., None]
    if r.shape[2] == 4:
        a = r[..., 3:4]
        canvas[y0:y1, x0:x1] = r[..., :3] * a + canvas[y0:y1, x0:x1] * (1 - a)
    else:
        canvas[y0:y1, x0:x1] = r[..., :3]


def _font(size: int):
    from ..ops.draw import load_font

    return load_font("sans-serif", max(1, size), bold=True)


def draw_label(canvas: np.ndarray, text: str, at_start: bool, text_size: float | None = None) -> None:
    """``CompareAnimationRenderer.drawLabel`` on a float RGB canvas."""
    h, w = canvas.shape[:2]
    ts = text_size if text_size is not None else min(w, h) * 0.045
    font = _font(int(round(ts)))
    ascent, descent = font.getmetrics()
    pad = ts * 0.55
    tw = font.getlength(text)
    lw, lh = tw + 2 * pad, ascent + descent + 2 * pad
    left = pad if at_start else w - pad - lw
    top = pad if at_start else h - pad - lh
    ox, oy = int(math.floor(left)), int(math.floor(top))
    bw, bh = int(math.ceil(left + lw)) - ox + 1, int(math.ceil(top + lh)) - oy + 1
    layer = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle([left - ox, top - oy, left - ox + lw, top - oy + lh], radius=pad, fill=(0, 0, 0, 170))
    d.text((left - ox + pad, top - oy + pad), text, font=font, fill=(255, 255, 255, 255))
    arr = np.asarray(layer, np.float32) / 255.0
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(w, ox + bw), min(h, oy + bh)
    if x1 <= x0 or y1 <= y0:
        return
    sub = arr[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    a = sub[..., 3:4]
    canvas[y0:y1, x0:x1] = sub[..., :3] * a + canvas[y0:y1, x0:x1] * (1 - a)


def render_layers(before: np.ndarray, after: np.ndarray, width: int, height: int, show_labels: bool,
                  before_label: str, after_label: str) -> tuple[np.ndarray, np.ndarray]:
    """The two full-frame layers of the renderer: (before layer, after layer) as float RGB."""
    lb = np.ones((height, width, 3), F32)
    draw_fitted(lb, before)
    la = np.ones((height, width, 3), F32)
    draw_fitted(la, after)
    if show_labels:
        draw_label(lb, before_label, at_start=False)
        draw_label(la, after_label, at_start=True)
    return lb, la


def divider_width(width: int, height: int) -> float:
    return max(1.0, min(width, height) * 0.004)


def frame_bounds(prev: float | None, pos: float, width: int, height: int, vertical: bool) -> tuple[int, int, int, int]:
    """``frameBounds`` -> (left, top, right, bottom)."""
    if prev is None:
        return 0, 0, width, height
    side = height if vertical else width
    dw = np.float32(divider_width(width, height))
    pe = np.float32(side) * np.float32(min(1.0, max(0.0, prev)))
    e = np.float32(side) * np.float32(min(1.0, max(0.0, pos)))
    start = min(max(int(math.floor(min(pe, e) - dw)), 0), side - 1)
    end = min(max(int(math.ceil(max(pe, e) + dw)), start + 1), side)
    return (0, start, width, end) if vertical else (start, 0, end, height)


def compose_index_frame(ib: np.ndarray, ia: np.ndarray, pos: float, vertical: bool, white_index: int,
                        dw: float) -> np.ndarray:
    """Palette-index frame at divider position ``pos`` (after layer clipped to [0, edge] over before)."""
    h, w = ib.shape
    side = h if vertical else w
    p = min(1.0, max(0.0, pos))
    edge = side * p
    centres = np.arange(side) + 0.5
    use_after = centres < edge
    out = np.where(use_after[:, None] if vertical else use_after[None, :], ia, ib)
    if 0.0 < p < 1.0:
        line = np.abs(centres - edge) <= dw / 2
        if not line.any():  # a hairline always covers at least the nearest pixel row/column
            line[min(side - 1, int(edge))] = True
        if vertical:
            out[line, :] = white_index
        else:
            out[:, line] = white_index
    return out


def _lzw_block(index_img: np.ndarray, palette: bytes) -> tuple[bytes, int]:
    """Encode one frame with Pillow and return (image data starting at the LZW code size byte, packed flags)."""
    im = Image.fromarray(np.ascontiguousarray(index_img, dtype=np.uint8), "P")
    im.putpalette(palette)
    buf = io.BytesIO()
    im.save(buf, "GIF", interlace=False, optimize=False)
    data = buf.getvalue()
    # header(6) + logical screen descriptor(7) + global colour table
    packed = data[10]
    pos = 13 + (3 * (2 ** ((packed & 7) + 1)) if packed & 0x80 else 0)
    gct = data[13:pos]
    while data[pos] == 0x21:  # skip extensions
        pos += 2
        while data[pos]:
            pos += data[pos] + 1
        pos += 1
    assert data[pos] == 0x2C, "unexpected GIF structure"
    desc_flags = data[pos + 9]
    pos += 10
    if desc_flags & 0x80:
        pos += 3 * (2 ** ((desc_flags & 7) + 1))
    start = pos
    pos += 1  # LZW minimum code size
    while data[pos]:
        pos += data[pos] + 1
    pos += 1
    return data[start:pos], (gct, desc_flags)


def write_gif(frames: list[tuple[tuple[int, int, int, int], np.ndarray, int]], width: int, height: int,
              palette: bytes) -> bytes:
    """Assemble a looping GIF89a from (bounds, index sub-image, delay ms) with disposal 1."""
    pal = (palette + bytes(768))[:768]
    out = bytearray(b"GIF89a")
    out += struct.pack("<HHBBB", width, height, 0xF7, 0, 0)  # GCT present, 8 bits colour res, 256 entries
    out += pal
    out += b"\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00"  # repeat forever (setRepeat(0))
    for (l, t, r, b), sub, delay in frames:
        out += struct.pack("<BBBBHBB", 0x21, 0xF9, 4, (1 << 2), int(round(delay / 10)), 0, 0)
        lzw, (gct, flags) = _lzw_block(sub, pal)
        out += struct.pack("<BHHHH", 0x2C, l, t, r - l, b - t)
        if gct[:768] != pal[:len(gct)] or len(gct) < 768:
            n = len(gct) // 3
            bits = max(1, (n - 1).bit_length())
            lct = (gct + bytes(3 * (2 ** bits)))[:3 * (2 ** bits)]
            out += bytes([0x80 | (bits - 1)]) + lct
        else:
            out += b"\x00"
        out += lzw
    out += b"\x3B"
    return bytes(out)


def shared_palette(layers: list[np.ndarray]) -> tuple[Image.Image, int]:
    """256-entry palette (255 median-cut colours + pure white) shared by all frames."""
    stack = np.concatenate([cv2.resize(l, (max(1, l.shape[1] // 2), max(1, l.shape[0] // 2)),
                                       interpolation=cv2.INTER_AREA) for l in layers], 0)
    pil = Image.fromarray((np.clip(stack, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
    q = pil.quantize(255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = list(q.getpalette()[:255 * 3])
    pal += [0] * (255 * 3 - len(pal))
    pal += [255, 255, 255]
    pimg = Image.new("P", (1, 1))
    pimg.putpalette(pal)
    return pimg, 255


def quantize(layer: np.ndarray, pal_img: Image.Image) -> np.ndarray:
    pil = Image.fromarray((np.clip(layer, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
    return np.asarray(pil.quantize(palette=pal_img, dither=Image.Dither.FLOYDSTEINBERG))


def render_animation(before: np.ndarray, after: np.ndarray, params: AnimationParams, before_label: str = "Before",
                     after_label: str = "After") -> tuple[bytes, dict]:
    width, height = output_size(before.shape[1], before.shape[0], params.max_size)
    lb, la = render_layers(before, after, width, height, params.show_labels, before_label, after_label)
    pal_img, white = shared_palette([lb, la])
    ib, ia = quantize(lb, pal_img), quantize(la, pal_img)
    dw = divider_width(width, height)
    frames = []
    prev = None
    meta = []
    for pos, delay in params.frames():
        l, t, r, b = frame_bounds(prev, pos, width, height, params.is_vertical)
        full = compose_index_frame(ib, ia, pos, params.is_vertical, white, dw)
        frames.append(((l, t, r, b), full[t:b, l:r], delay))
        meta.append({"position": round(pos, 6), "delay_ms": delay, "bounds": [l, t, r, b]})
        prev = pos
    data = write_gif(frames, width, height, bytes(pal_img.getpalette()[:768]))
    return data, {"width": width, "height": height, "frames": meta}
