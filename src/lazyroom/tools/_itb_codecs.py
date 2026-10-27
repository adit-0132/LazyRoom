"""ImageToolbox image-format encoders (private helpers for ``format_convert`` and ``_itb_formats``).

Ports of ImageToolbox (Apache-2.0) ``core/data/.../image/utils/compressor``:

* ``PngLossyBackend`` -> :func:`png_lossy_encode`: aire ``toPNG`` (MIT) = Xiaolin Wu 4-D (ARGB)
  colour quantiser (nQuant variant, 3-bit cells, sqrt weighting, alpha fader) + nearest-colour
  remapping with aire's weighted RGBA metric.
* ``IcoBackend`` -> :func:`ico_encode`: 16/32/48/64/128/256 px 32-bit BMP entries, each image fitted
  and centred on a transparent square.
* Telegram sticker preset -> :func:`telegram_sticker_size` (ResizeType.Flexible, ResizeAnchor.Default, 512).
* ``OxiPngBackend`` / ``JpegliBackend`` -> optional ``pyoxipng`` (MIT) / ``pyjpegli`` (BSD-3).
* JXL tools animation encoding -> :func:`encode_jxl_animation` (imagecodecs/libjxl, BSD-3).

See docs/dev/port_reports/itb_extras.md for deviations.
"""
from __future__ import annotations

import io
import math
from functools import reduce
from typing import Any

import numpy as np
from numba import njit
from PIL import Image

from ..core.errors import ErrorCode, LazyRoomError, invalid
from ..core.image import has_alpha, to_uint8

OXIPNG_HINT = "pip install 'lazyroom[oxipng]' (installs pyoxipng, MIT)"
JPEGLI_HINT = "pip install 'lazyroom[jpegli]' (installs pyjpegli, BSD-3, bundles google/jpegli)"

# special format_convert choices -> written extension
SPECIAL_FORMATS = {"png_lossy": ".png", "oxipng": ".png", "jpegli": ".jpg", "ico_multi": ".ico"}


def _require(module: str, hint: str, what: str):
    import importlib

    try:
        return importlib.import_module(module)
    except Exception:
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, f"{what} requires the optional '{module}' package",
                              details={"backend": module}, hint=hint) from None


# ---------------------------------------------------------------------------
# Xiaolin Wu colour quantiser (aire algo/WuQuantizer.cpp)
# ---------------------------------------------------------------------------
_SIDE = 33  # (256 >> 3) + 1
_W, _A, _R, _G, _B, _M2 = range(6)  # moment planes; arrays are indexed [blue, green, red, alpha]


@njit(cache=True)
def _box_sum(m, box, d, pos):
    """Inclusion-exclusion sum over a 4-D box (d < 0: Volume) or the box's slice at ``pos`` along d (Top).

    box = [aMin, aMax, rMin, rMax, gMin, gMax, bMin, bMax]; d: 0 alpha, 1 red, 2 green, 3 blue."""
    s = 0.0
    for mask in range(16):
        if d >= 0 and (mask >> d) & 1:
            continue
        sign = 1.0
        a, r, g, b = box[1], box[3], box[5], box[7]
        if mask & 1:
            a = box[0]
            sign = -sign
        if mask & 2:
            r = box[2]
            sign = -sign
        if mask & 4:
            g = box[4]
            sign = -sign
        if mask & 8:
            b = box[6]
            sign = -sign
        if d == 0:
            a = pos
        elif d == 1:
            r = pos
        elif d == 2:
            g = pos
        elif d == 3:
            b = pos
        s += sign * m[b, g, r, a]
    return s


@njit(cache=True)
def _maximize(mom, box, d, first, last, whole):
    planes = (_A, _R, _G, _B, _W)
    bottom = np.empty(5)
    for k in range(5):
        bottom[k] = -_box_sum(mom[planes[k]], box, d, box[2 * d])
    valid = False
    result = 0.0
    cut = 0
    half = np.empty(5)
    for pos in range(first, last):
        for k in range(5):
            half[k] = bottom[k] + _box_sum(mom[planes[k]], box, d, pos)
        if half[4] == 0:
            continue
        temp = (half[0] ** 2 + half[1] ** 2 + half[2] ** 2 + half[3] ** 2) / half[4]
        for k in range(5):
            half[k] = whole[k] - half[k]
        if half[4] != 0:
            temp += (half[0] ** 2 + half[1] ** 2 + half[2] ** 2 + half[3] ** 2) / half[4]
            if temp > result:
                valid = True
                result = temp
                cut = pos
    return valid, cut, result


@njit(cache=True)
def _box_size(box):
    return (box[1] - box[0]) * (box[3] - box[2]) * (box[5] - box[4]) * (box[7] - box[6])


@njit(cache=True)
def _cut(mom, boxes, i1, i2):
    first = boxes[i1]
    whole = np.empty(5)
    for k, p in enumerate((_A, _R, _G, _B, _W)):
        whole[k] = _box_sum(mom[p], first, -1, 0)
    valid = np.zeros(4, np.bool_)
    cuts = np.zeros(4, np.int64)
    vals = np.zeros(4)
    for d in range(4):
        v, c, r = _maximize(mom, first, d, first[2 * d] + 1, first[2 * d + 1], whole)
        valid[d], cuts[d], vals[d] = v, c, r
    if vals[0] >= vals[1] and vals[0] >= vals[2] and vals[0] >= vals[3]:
        if not valid[0]:
            return False
        d = 0
    elif vals[1] >= vals[0] and vals[1] >= vals[2] and vals[1] >= vals[3]:
        d = 1
    elif vals[2] >= vals[0] and vals[2] >= vals[1] and vals[2] >= vals[3]:
        d = 2
    else:
        d = 3
    second = boxes[i2]
    for k in range(4):
        second[2 * k + 1] = first[2 * k + 1]
        second[2 * k] = first[2 * k]
    first[2 * d + 1] = cuts[d]
    second[2 * d] = cuts[d]
    return True


@njit(cache=True)
def _variance(mom, box):
    va = _box_sum(mom[_A], box, -1, 0)
    vr = _box_sum(mom[_R], box, -1, 0)
    vg = _box_sum(mom[_G], box, -1, 0)
    vb = _box_sum(mom[_B], box, -1, 0)
    vm = _box_sum(mom[_M2], box, -1, 0)
    vw = _box_sum(mom[_W], box, -1, 0)
    dist = va * va + vr * vr + vg * vg + vb * vb
    return vm - dist / vw if vw != 0 else 0.0


@njit(cache=True)
def _split(mom, color_count):
    boxes = np.zeros((color_count, 8), np.int64)
    boxes[0, 1] = boxes[0, 3] = boxes[0, 5] = boxes[0, 7] = _SIDE - 1
    variance = np.zeros(color_count)
    nxt = 0
    cube = 1
    while cube < color_count:
        if _cut(mom, boxes, nxt, cube):
            variance[nxt] = _variance(mom, boxes[nxt]) if _box_size(boxes[nxt]) > 1 else 0.0
            variance[cube] = _variance(mom, boxes[cube]) if _box_size(boxes[cube]) > 1 else 0.0
        else:
            variance[nxt] = 0.0
            cube -= 1
        nxt = 0
        temp = variance[0]
        for i in range(1, cube + 1):
            if variance[i] > temp:
                temp = variance[i]
                nxt = i
        if temp > 0.0:
            cube += 1
            continue
        color_count = cube + 1
        break
    return boxes[:color_count]


@njit(cache=True)
def _histogram(px, alpha_threshold, alpha_fader):
    """BuildHistogram/Histogram3d: 33^4 ARGB cells (index 1 + value >> 3), faded alpha, alpha <= threshold skipped."""
    s = _SIDE
    mom = np.zeros((6, s ** 4))
    transparent = False
    for i in range(px.shape[0]):
        r, g, b, a = np.int64(px[i, 0]), np.int64(px[i, 1]), np.int64(px[i, 2]), np.int64(px[i, 3])
        if a == 0:
            transparent = True
        if a <= alpha_threshold:
            continue
        if a < 255:
            a = min(255, a + a % alpha_fader)
        k = ((b >> 3) + 1) * s * s * s + ((g >> 3) + 1) * s * s + ((r >> 3) + 1) * s + ((a >> 3) + 1)
        mom[_W, k] += 1.0
        mom[_A, k] += a
        mom[_R, k] += r
        mom[_G, k] += g
        mom[_B, k] += b
        mom[_M2, k] += a * a + r * r + g * g + b * b
    return mom, transparent


def wu_palette(rgba: np.ndarray, max_colors: int, alpha_threshold: int = 15, alpha_fader: int = 15) -> np.ndarray:
    """aire ``WuQuantizer::quantizeImage(maxColors, 15, 15)`` on straight-alpha RGBA uint8 pixels (N, 4).

    Returns the palette as (P, 4) uint8 RGBA (P <= max_colors)."""
    mom, transparent = _histogram(np.ascontiguousarray(rgba.reshape(-1, 4)), alpha_threshold, alpha_fader)
    s = _SIDE
    nz = mom[_W] > 0
    count = int(nz.sum())
    # AdjustMoments: `sqr(nMaxColors) / indices.size() < .04` is an integer division -> skip when n^2 < cells
    if count and (max_colors * max_colors) // count != 0:
        w = mom[_W, nz]
        sw = np.floor(np.sqrt(w))
        f = sw / w
        mom[_W, nz] = sw
        for plane in (_R, _G, _B, _A):
            mom[plane, nz] = np.floor(mom[plane, nz] * f)  # long *= double truncates
        mom[_M2, nz] = mom[_M2, nz] * f
    mom = mom.reshape(6, s, s, s, s)
    for ax in range(1, 5):  # CalculateMoments: 4-D cumulative moments
        np.cumsum(mom, axis=ax, out=mom)
    boxes = _split(mom, int(max_colors))
    out = [(0, 0, 0, 0)] if transparent else []
    for box in boxes:
        wgt = _box_sum(mom[_W], box, -1, 0)
        if wgt <= 0:
            continue
        out.append(tuple(int(_box_sum(mom[p], box, -1, 0) / wgt) for p in (_R, _G, _B, _A)))
    return np.array(out[:max_colors], np.uint8).reshape(-1, 4)


def remap_nearest(rgba: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Nearest palette colour with aire's metric 2dR^2 + 4dG^2 + (R<128 ? 3 : 2)dB^2 + dA^2 (R of the pixel)."""
    from scipy.spatial import cKDTree

    h, w = rgba.shape[:2]
    if len(palette) == 0:
        return np.zeros_like(rgba)
    packed = rgba.reshape(-1, 4).astype(np.uint32)
    packed = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16) | (packed[:, 3] << 24)
    if bool((rgba[..., 3] == 255).all()):  # opaque: O(N) unique colours through a 2^24 lookup table
        rgb24 = packed & 0xFFFFFF
        present = np.zeros(1 << 24, np.bool_)
        present[rgb24] = True
        uniq = np.flatnonzero(present).astype(np.uint32) | np.uint32(0xFF000000)
        lut = np.zeros(1 << 24, np.int32)
        lut[uniq & 0xFFFFFF] = np.arange(len(uniq), dtype=np.int32)
        inv = lut[rgb24]
    else:
        uniq, inv = np.unique(packed, return_inverse=True)
    cols = np.stack([(uniq >> s) & 255 for s in (0, 8, 16, 24)], axis=1).astype(np.float64)
    idx = np.empty(len(uniq), np.int64)
    for low, wb in ((True, 3.0), (False, 2.0)):
        sel = (cols[:, 0] < 128) == low
        if sel.any():
            scale = np.sqrt(np.array([2.0, 4.0, wb, 1.0]))
            _, idx[sel] = cKDTree(palette.astype(np.float64) * scale).query(cols[sel] * scale, workers=-1)
    return palette[idx][inv.ravel()].reshape(h, w, 4)


def png_lossy_pixels(img: np.ndarray, max_colors: int = 512) -> np.ndarray:
    """ImageToolbox "PNG Lossy" pixels: Wu palette (<= max_colors) + nearest-colour remap (RGBA uint8)."""
    if not 2 <= max_colors <= 1024:
        raise invalid("max_colors must be within 2..1024 (ImageToolbox range)", param="max_colors")
    u8 = to_uint8(img if has_alpha(img) else np.dstack([img[..., :3], np.ones(img.shape[:2], np.float32)]))
    pal = wu_palette(u8, max_colors)
    if (u8[..., 3] == 255).all():
        # Deviation: aire truncates the sqrt-weighted alpha moments (long *= double), so fully opaque images can
        # get alpha-254 palette entries (a stray tRNS chunk).  Keep opaque images opaque.
        pal[:, 3] = 255
    return remap_nearest(u8, pal)


def _png_kwargs(meta, metadata: str) -> dict[str, Any]:
    from ..core.io import _SRGB_ICC_BYTES, _filter_exif

    kw: dict[str, Any] = {"icc_profile": _SRGB_ICC_BYTES}
    ex = _filter_exif(meta.exif, metadata) if meta is not None else None
    if ex:
        kw["exif"] = ex
    return kw


def png_lossy_encode(img: np.ndarray, max_colors: int = 512, compression_level: int = 7, *, meta=None,
                     metadata: str = "strip") -> bytes:
    """PNG with <= max_colors colours. Written as a palette PNG when <= 256 colours (ImageToolbox writes the
    same pixels as RGBA truecolour), otherwise as RGB(A)."""
    if not 0 <= compression_level <= 9:
        raise invalid("compression_level must be within 0..9", param="effort")
    q = png_lossy_pixels(img, max_colors)
    opaque = bool((q[..., 3] == 255).all())
    packed = q.reshape(-1, 4).astype(np.uint32)
    packed = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16) | (packed[:, 3] << 24)
    uniq, inv = np.unique(packed, return_inverse=True)
    kw = _png_kwargs(meta, metadata) | {"compress_level": int(compression_level), "optimize": False}
    if len(uniq) <= 256:
        pal = np.stack([(uniq >> s) & 255 for s in (0, 8, 16, 24)], axis=1).astype(np.uint8)
        pil = Image.fromarray(inv.reshape(q.shape[:2]).astype(np.uint8), "P")
        pil.putpalette(pal[:, :3].tobytes(), rawmode="RGB")
        if not opaque:
            kw["transparency"] = pal[:, 3].tobytes()
    else:
        pil = Image.fromarray(np.ascontiguousarray(q[..., :3] if opaque else q), "RGB" if opaque else "RGBA")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", **kw)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# ICO (IcoBackend)
# ---------------------------------------------------------------------------
ICO_SIZES = (16, 32, 48, 64, 128, 256)


def _fit_on_square(pil: Image.Image, size: int) -> Image.Image:
    """ResizeType.Fit(canvasColor = transparent): fit inside size x size (Kotlin float/Int truncation), centre."""
    w, h = pil.size
    if (w, h) == (size, size):
        return pil
    aspect = np.float32(w) / np.float32(h)
    if aspect > 1:
        fw, fh = size, int(np.float32(size) / aspect)
    else:
        fw, fh = int(np.float32(size) * aspect), size
    fw, fh = max(1, fw), max(1, fh)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(pil.resize((fw, fh), Image.Resampling.LANCZOS), ((size - fw) // 2, (size - fh) // 2))
    return canvas


def ico_encode(img: np.ndarray, sizes: tuple[int, ...] = ICO_SIZES) -> bytes:
    """Multi-size ICO exactly laid out like ImageToolbox's IcoBackend (BITMAPINFOHEADER, 32 bpp BGRA bottom-up,
    empty AND mask, size 256 stored as 0 in the directory)."""
    u8 = to_uint8(img if has_alpha(img) else np.dstack([img[..., :3], np.ones(img.shape[:2], np.float32)]))
    src = Image.fromarray(u8, "RGBA")
    entries = []
    for size in sizes:
        px = np.asarray(_fit_on_square(src, size).convert("RGBA"))
        and_size = ((size + 31) // 32) * 4 * size
        xor_size = size * size * 4
        header = (40).to_bytes(4, "little") + size.to_bytes(4, "little") + (2 * size).to_bytes(4, "little") + \
            (1).to_bytes(2, "little") + (32).to_bytes(2, "little") + (0).to_bytes(4, "little") + \
            (xor_size + and_size).to_bytes(4, "little") + bytes(16)
        bgra = px[::-1][..., [2, 1, 0, 3]].tobytes()
        entries.append((size, header + bgra + bytes(and_size)))
    out = bytearray(b"\x00\x00\x01\x00" + len(entries).to_bytes(2, "little"))
    offset = 6 + 16 * len(entries)
    for size, data in entries:
        dim = 0 if size == 256 else size
        out += bytes([dim, dim, 0, 0]) + (1).to_bytes(2, "little") + (32).to_bytes(2, "little") + \
            len(data).to_bytes(4, "little") + offset.to_bytes(4, "little")
        offset += len(data)
    for _, data in entries:
        out += data
    return bytes(out)


# ---------------------------------------------------------------------------
# Telegram sticker preset (Preset.Telegram -> 512 x 512, ResizeType.Flexible, PNG lossless)
# ---------------------------------------------------------------------------

def telegram_sticker_size(w: int, h: int, target: int = 512) -> tuple[int, int]:
    """AndroidImageScaler.flexibleResize with ResizeAnchor.Default and max(width, height) = 512:
    the longer side becomes 512, the other is truncated (Kotlin Float -> Int) and clamped to >= 1."""
    aspect = np.float32(w) / np.float32(h)
    if h >= w:
        return max(1, int(np.float32(target) * aspect)), target
    return target, max(1, int(np.float32(target) * (np.float32(1) / aspect)))


def telegram_sticker(img: np.ndarray) -> np.ndarray:
    tw, th = telegram_sticker_size(img.shape[1], img.shape[0])
    if (tw, th) == (img.shape[1], img.shape[0]):
        return img
    mode = "RGBA" if has_alpha(img) else "RGB"
    pil = Image.fromarray(to_uint8(img), mode).resize((tw, th), Image.Resampling.LANCZOS)  # ImageScaleMode.Lanczos3
    return np.asarray(pil, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# OxiPNG / Jpegli (optional packages)
# ---------------------------------------------------------------------------

def oxipng_encode(img: np.ndarray, level: int = 6, *, meta=None, metadata: str = "strip") -> bytes:
    """OxiPngBackend: PNG optimised by oxipng with preset ``level`` (0..6)."""
    oxipng = _require("oxipng", OXIPNG_HINT, "OxiPNG output")
    from ..core.io import encode_image

    if not 0 <= level <= 6:
        raise invalid("OxiPNG level (effort) must be within 0..6", param="effort")
    data = encode_image(img, ".png", meta=meta, metadata=metadata, extra={"optimize": False, "compress_level": 1})
    try:
        return bytes(oxipng.optimize_from_memory(data, level=int(level)))
    except Exception as exc:
        raise LazyRoomError(ErrorCode.ENCODE_FAILED, f"oxipng failed: {exc}") from exc


def _jpeg_insert_segments(jpeg: bytes, segments: list[bytes]) -> bytes:
    pos = 2
    if jpeg[2:4] == b"\xff\xe0":  # keep JFIF APP0 first
        pos += 2 + int.from_bytes(jpeg[4:6], "big")
    return jpeg[:pos] + b"".join(segments) + jpeg[pos:]


def jpegli_encode(img: np.ndarray, quality: int = 90, *, meta=None, metadata: str = "strip",
                  background: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> bytes:
    """JpegliBackend: google/jpegli via pyjpegli (baseline; pyjpegli exposes no progressive switch)."""
    pyjpegli = _require("pyjpegli", JPEGLI_HINT, "Jpegli output")
    from ..core.image import composite_on
    from ..core.io import _SRGB_ICC_BYTES, _filter_exif

    rgb = np.ascontiguousarray(to_uint8(composite_on(img, background) if has_alpha(img) else img[..., :3]))
    try:
        data = bytes(pyjpegli.encode(rgb, rgb.shape[1], rgb.shape[0], quality=int(quality)))
    except Exception as exc:
        raise LazyRoomError(ErrorCode.ENCODE_FAILED, f"jpegli failed: {exc}") from exc
    segs = []
    ex = _filter_exif(meta.exif, metadata) if meta is not None else None
    if ex:
        tiff = ex[6:] if ex.startswith(b"Exif\x00\x00") else ex
        if len(tiff) + 8 <= 0xFFFF:
            segs.append(b"\xff\xe1" + (len(tiff) + 8).to_bytes(2, "big") + b"Exif\x00\x00" + tiff)
    icc = b"ICC_PROFILE\x00\x01\x01" + _SRGB_ICC_BYTES
    segs.append(b"\xff\xe2" + (len(icc) + 2).to_bytes(2, "big") + icc)
    return _jpeg_insert_segments(data, segs)


def check_special(fmt: str, effort: int) -> str:
    """Validate a special ``format_convert`` choice up front (backend + effort range); returns its extension."""
    limit = {"png_lossy": 9, "oxipng": 6}.get(fmt)
    if limit is not None and effort > limit:
        raise invalid(f"effort for {fmt} must be within 0..{limit} (or -1 for the default)", param="effort")
    if fmt == "oxipng":
        _require("oxipng", OXIPNG_HINT, "OxiPNG output")
    elif fmt == "jpegli":
        _require("pyjpegli", JPEGLI_HINT, "Jpegli output")
    return SPECIAL_FORMATS[fmt]


def encode_special(fmt: str, img: np.ndarray, *, quality: int, effort: int, max_colors: int, meta=None,
                   metadata: str = "keep") -> bytes:
    """Encoders for the ImageToolbox-only ``format_convert`` choices (``effort`` < 0 = ImageToolbox default)."""
    if fmt == "png_lossy":
        return png_lossy_encode(img, max_colors, 7 if effort < 0 else effort, meta=meta, metadata=metadata)
    if fmt == "oxipng":
        return oxipng_encode(img, 6 if effort < 0 else effort, meta=meta, metadata=metadata)
    if fmt == "jpegli":
        return jpegli_encode(img, quality, meta=meta, metadata=metadata)
    if fmt == "ico_multi":
        return ico_encode(img)
    raise invalid(f"Unknown special format {fmt!r}")


# ---------------------------------------------------------------------------
# Animated JPEG XL (imagecodecs writes tps = 10/s and 1 tick per frame; we rewrite the tick)
# ---------------------------------------------------------------------------

def _put_bits(buf: bytearray, pos: int, n: int, v: int) -> None:
    for i in range(n):
        byte, sh = (pos + i) >> 3, (pos + i) & 7
        buf[byte] = (buf[byte] & ~(1 << sh) & 0xFF) | (((v >> i) & 1) << sh)


def set_jxl_tick(data: bytes, tick_ms: int) -> bytes:
    """Rewrite the AnimationHeader of an imagecodecs animated JXL codestream so one tick = ``tick_ms``.

    imagecodecs encodes tps_numerator=10 (U32 selector 2 + 10 bits), tps_denominator=1, num_loops=0,
    have_timecodes=0 = 17 bits.  tps 1000/tick_ms is written as numerator Val(1000) (selector 1),
    denominator BitsOffset(10, 1) (selector 3 + 10 bits), loops 0, no timecodes = the same 17 bits,
    so nothing after the header moves.  Only 1..1024 ms ticks are representable this way."""
    from ..core.io import jxl_header

    if not 1 <= tick_ms <= 1024:
        raise ValueError("tick must be within 1..1024 ms")
    info = jxl_header(data)
    if data[:2] != b"\xff\x0a" or not info.get("animated") or info["tps_numerator"] != 10 \
            or info["tps_denominator"] != 1 or info["num_loops"] != 0:
        raise LazyRoomError(ErrorCode.INTERNAL, "Unexpected animated JPEG XL header from imagecodecs",
                              details={"header": {k: v for k, v in info.items() if k != "anim_bitpos"}})
    buf = bytearray(data)
    pos = info["anim_bitpos"]
    _put_bits(buf, pos, 2, 1)             # tps_numerator selector 1 -> Val(1000)
    _put_bits(buf, pos + 2, 2, 3)         # tps_denominator selector 3 -> BitsOffset(10, 1)
    _put_bits(buf, pos + 4, 10, tick_ms - 1)
    _put_bits(buf, pos + 14, 2, 0)        # num_loops Val(0) (infinite)
    _put_bits(buf, pos + 16, 1, 0)        # have_timecodes
    out = bytes(buf)
    check = jxl_header(out)
    if (check["tps_numerator"], check["tps_denominator"]) != (1000, tick_ms):
        raise LazyRoomError(ErrorCode.INTERNAL, "JPEG XL tick rewrite failed")
    return out


def jxl_tick_plan(durations_ms: list[int]) -> tuple[int, list[int]]:
    """Common tick (<= 1024 ms, divides every duration) and per-frame repeat counts."""
    durs = [max(1, int(round(d))) for d in durations_ms]
    g = reduce(math.gcd, durs)
    tick = max(d for d in range(1, min(g, 1024) + 1) if g % d == 0)
    return tick, [d // tick for d in durs]


def encode_jxl_animation(frames: list[np.ndarray], durations_ms: list[int], *, quality: int = 50,
                         lossless: bool = False, effort: int = 2, decoding_speed: int = 0) -> tuple[bytes, dict]:
    """Encode equally sized float frames as an animated JXL (infinite loop) with exact per-frame durations
    (frames are repeated when durations are not equal multiples of one tick <= 1024 ms)."""
    from ..core.io import HAVE_JXL, JXL_HINT

    if not HAVE_JXL:
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, "JPEG XL output requires the optional 'imagecodecs' package",
                              details={"backend": "imagecodecs"}, hint=JXL_HINT)
    import imagecodecs

    alpha = any(has_alpha(f) for f in frames)
    stack = []
    tick, reps = jxl_tick_plan(durations_ms)
    if sum(reps) > max(1000, 4 * len(frames)):
        raise invalid(f"Frame durations {sorted(set(durations_ms))} ms need {sum(reps)} stored frames "
                      f"(common tick {tick} ms)", hint="Pass a uniform frame_ms to override the source durations.")
    for f, n in zip(frames, reps):
        f = f if not alpha or has_alpha(f) else np.dstack([f[..., :3], np.ones(f.shape[:2], np.float32)])
        u8 = to_uint8(f if alpha else f[..., :3])
        stack.extend([u8] * n)
    kw: dict[str, Any] = {"effort": int(effort), "decodingspeed": int(decoding_speed)}
    if lossless or quality >= 100:
        kw["lossless"] = True
    else:
        kw["level"] = int(quality)
    try:
        data = bytes(imagecodecs.jpegxl_encode(np.stack(stack) if len(stack) > 1 else stack[0], **kw))
    except Exception as exc:
        raise LazyRoomError(ErrorCode.ENCODE_FAILED, f"Failed to encode animated JPEG XL: {exc}") from exc
    if len(stack) == 1:  # one frame: a still JXL (no animation header)
        return data, {"tick_ms": None, "stored_frames": 1, "frames": len(frames)}
    return set_jxl_tick(data, tick), {"tick_ms": tick, "stored_frames": len(stack), "frames": len(frames)}
