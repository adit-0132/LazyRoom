"""Hidden ("digital") watermarks, bit-compatible with ImageToolbox.

Port of ImageToolboxLibs ``libs/androidwm`` (AndroidWM by huangyz0918, maintained in ImageToolboxLibs by
T8RIN, Apache-2.0): ``src/main/cpp/Watermark.cpp`` (embed / extract), ``task/*`` and ``BitmapUtils``.

Frame (both methods): 32-bit magic (MSB first), 1 bit "is image", 7 zero bits, 24-bit payload length in bytes,
then the payload bytes MSB first.  The payload is the text as Java *modified UTF-8* (``GetStringUTFChars``) or,
for image watermarks, the Base64 (no wrap) of a PNG (LSB) / JPEG (FD) of the watermark image.

* LSB (magic ``LSBW``): frame bit ``i`` replaces the least significant bit of pixel ``i // 3``, channel
  ``i % 3`` taken in the order of the ``uint32`` shifts {16, 8, 0} of an Android ``RGBA_8888`` pixel on a
  little-endian device = **blue, green, red**.
* FD (magic ``FDWM``): frame bit ``i`` lives in 8x8 block ``i`` (row-major, ``width // 8`` blocks per row),
  replicated in the B, G and R channels: the orthonormal 8x8 DCT coefficients (2,3) and (3,2) are pushed apart
  to a difference of +-80 (retried with 160 and 240 when rounding/clamping flips it); reading takes the majority
  of ``D[2][3] >= D[3][2]`` over the three channels.

Android bitmaps are premultiplied: ImageToolbox embeds into premultiplied channel values and the PNG encoder
un-premultiplies.  For pixels with alpha < 255 this module emulates that (premultiply -> embed -> un-premultiply),
so semi-transparent images interoperate too; fully transparent pixels cannot carry data (as in ImageToolbox).

LSB data does not survive any lossy re-encoding, resizing or colour conversion; FD tolerates mild JPEG
compression only.  Save results as PNG (or another lossless format).
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

LSB_MAGIC = 0x4C534257
FD_MAGIC = 0x4644574D
HEADER_BITS = 64
FD_REDUNDANCY = 3
FD_STRENGTH = 80.0
CHANNEL_ORDER = (2, 1, 0)  # uint32 shifts 16, 8, 0 of little-endian RGBA_8888 -> B, G, R
MAX_PAYLOAD = (1 << 24) - 1

# orthonormal 8x8 DCT-II basis: M[v, y] = s(v) cos((2y + 1) v pi / 16)
_s = np.array([np.sqrt(0.125)] + [0.5] * 7)
_M = _s[:, None] * np.cos((2 * np.arange(8)[None, :] + 1) * np.arange(8)[:, None] * np.pi / 16)


class StegoError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Java modified UTF-8 (JNI GetStringUTFChars / NewStringUTF)
# ---------------------------------------------------------------------------

def to_modified_utf8(text: str) -> bytes:
    out = bytearray()
    units = text.encode("utf-16-le", "surrogatepass")
    for i in range(0, len(units), 2):
        u = units[i] | (units[i + 1] << 8)
        if 0 < u < 0x80:
            out.append(u)
        elif u < 0x800:  # includes U+0000 -> C0 80
            out += bytes((0xC0 | (u >> 6), 0x80 | (u & 0x3F)))
        else:
            out += bytes((0xE0 | (u >> 12), 0x80 | ((u >> 6) & 0x3F), 0x80 | (u & 0x3F)))
    return bytes(out)


def from_modified_utf8(data: bytes) -> str:
    data = data.split(b"\x00", 1)[0]  # NewStringUTF(result.c_str()) stops at the first NUL
    units: list[int] = []
    i, n = 0, len(data)
    while i < n:
        b = data[i]
        if b < 0x80:
            units.append(b)
            i += 1
        elif b >> 5 == 0b110 and i + 1 < n and data[i + 1] >> 6 == 2:
            units.append(((b & 0x1F) << 6) | (data[i + 1] & 0x3F))
            i += 2
        elif b >> 4 == 0b1110 and i + 2 < n and data[i + 1] >> 6 == 2 and data[i + 2] >> 6 == 2:
            units.append(((b & 0x0F) << 12) | ((data[i + 1] & 0x3F) << 6) | (data[i + 2] & 0x3F))
            i += 3
        elif b >> 3 == 0b11110 and i + 3 < n:  # standard 4-byte UTF-8 (lenient)
            cp = ((b & 7) << 18) | ((data[i + 1] & 0x3F) << 12) | ((data[i + 2] & 0x3F) << 6) | (data[i + 3] & 0x3F)
            cp -= 0x10000
            units += [0xD800 | (cp >> 10), 0xDC00 | (cp & 0x3FF)]
            i += 4
        else:
            units.append(0xFFFD)
            i += 1
    raw = b"".join(u.to_bytes(2, "little") for u in units)
    return raw.decode("utf-16-le", "surrogatepass").encode("utf-16", "surrogatepass").decode("utf-16", "replace")


# ---------------------------------------------------------------------------
# premultiplied storage emulation (8-bit)
# ---------------------------------------------------------------------------

def _premul(rgb: np.ndarray, a: np.ndarray) -> np.ndarray:
    prod = rgb.astype(np.int32) * a[..., None].astype(np.int32) + 128  # SkMulDiv255Round
    return ((prod + (prod >> 8)) >> 8).astype(np.int32)


def _unpremul(prem: np.ndarray, a: np.ndarray) -> np.ndarray:
    a32 = a[..., None].astype(np.int32)
    out = np.where(a32 > 0, (prem * 255 + a32 // 2) // np.maximum(a32, 1), 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def to_storage(rgba: np.ndarray) -> np.ndarray:
    """uint8 RGB(A) -> int32 (H, W, 3) channel values as seen by ImageToolbox (premultiplied)."""
    if rgba.shape[2] == 3 or bool((rgba[..., 3] == 255).all()):
        return rgba[..., :3].astype(np.int32)
    return _premul(rgba[..., :3], rgba[..., 3])


def from_storage(values: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    out = rgba.copy()
    if rgba.shape[2] == 3 or bool((rgba[..., 3] == 255).all()):
        out[..., :3] = np.clip(values, 0, 255).astype(np.uint8)
    else:
        out[..., :3] = _unpremul(np.clip(values, 0, 255), rgba[..., 3])
    return out


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

def frame_bits(magic: int, payload: bytes, is_image: bool) -> np.ndarray:
    if not payload:
        raise StegoError("Empty watermark")
    if len(payload) > MAX_PAYLOAD:
        raise StegoError(f"Watermark too large ({len(payload)} bytes; the frame stores a 24-bit length)")
    head = np.zeros(HEADER_BITS, np.uint8)
    head[:32] = [(magic >> (31 - i)) & 1 for i in range(32)]
    head[32] = 1 if is_image else 0
    head[40:64] = [(len(payload) >> (23 - i)) & 1 for i in range(24)]
    body = np.unpackbits(np.frombuffer(payload, np.uint8))
    return np.concatenate([head, body])


def _parse_header(read) -> tuple[int, bool, int]:
    bits = read(0, HEADER_BITS)
    magic = int("".join(map(str, bits[:32])), 2)
    length = int("".join(map(str, bits[40:64])), 2)
    return magic, bool(bits[32]), length


# ---------------------------------------------------------------------------
# LSB
# ---------------------------------------------------------------------------

def _lsb_view(values: np.ndarray) -> np.ndarray:
    return values[..., list(CHANNEL_ORDER)].reshape(-1)


def lsb_capacity(h: int, w: int) -> int:
    return w * h * 3


def lsb_embed(rgba: np.ndarray, payload: bytes, is_image: bool) -> np.ndarray:
    bits = frame_bits(LSB_MAGIC, payload, is_image)
    h, w = rgba.shape[:2]
    if len(bits) > lsb_capacity(h, w):
        raise StegoError(f"The image is too small for this watermark: {len(bits)} bits needed, "
                         f"{lsb_capacity(h, w)} available (ImageToolbox: ERROR_PIXELS_NOT_ENOUGH)")
    vals = to_storage(rgba)
    flat = _lsb_view(vals).copy()
    n = len(bits)
    flat[:n] = (flat[:n] & ~1) | bits
    order = list(CHANNEL_ORDER)
    new = np.empty_like(vals)
    new[..., order] = flat.reshape(h, w, 3)
    return from_storage(new, rgba)


def lsb_extract(rgba: np.ndarray, is_image: bool) -> bytes | None:
    h, w = rgba.shape[:2]
    cap = lsb_capacity(h, w)
    if cap < HEADER_BITS:
        return None
    flat = _lsb_view(to_storage(rgba))

    def read(start, count):
        return (flat[start:start + count] & 1).astype(np.uint8)

    magic, img_bit, length = _parse_header(read)
    if magic != LSB_MAGIC or img_bit != is_image:
        return None
    if HEADER_BITS + length * 8 > cap:
        return None
    return np.packbits(read(HEADER_BITS, length * 8)).tobytes()


# ---------------------------------------------------------------------------
# FD (8x8 DCT)
# ---------------------------------------------------------------------------

def fd_capacity(h: int, w: int) -> int:
    return (w // 8) * (h // 8) * 3 // FD_REDUNDANCY


def fd_payload_capacity(h: int, w: int) -> int:
    """``FDWatermarkTask.getPayloadCapacity`` (bytes)."""
    return max(0, ((w // 8) * (h // 8) - 64) // 8)


def _blocks(idx: np.ndarray, cols: int) -> tuple[np.ndarray, np.ndarray]:
    by, bx = idx // cols, idx % cols
    ys = (by[:, None, None] * 8 + np.arange(8)[None, :, None])
    xs = (bx[:, None, None] * 8 + np.arange(8)[None, None, :])
    return ys, xs


def _dct_bits(block: np.ndarray) -> np.ndarray:
    spec = _M @ block @ _M.T
    return spec[:, 2, 3] >= spec[:, 3, 2]


def fd_embed(rgba: np.ndarray, payload: bytes, is_image: bool) -> np.ndarray:
    bits = frame_bits(FD_MAGIC, payload, is_image)
    h, w = rgba.shape[:2]
    if len(bits) > fd_capacity(h, w):
        raise StegoError(f"The image is too small for this watermark: {len(bits)} blocks needed, "
                         f"{fd_capacity(h, w)} available")
    cols = w // 8
    vals = to_storage(rgba).astype(np.float64)
    n = len(bits)
    idx = np.repeat(np.arange(n), FD_REDUNDANCY)
    rep = np.tile(np.arange(FD_REDUNDANCY), n)
    ch = np.array(CHANNEL_ORDER)[rep]  # physicalIndex % 3 -> shifts {16, 8, 0} -> B, G, R
    want = bits[idx].astype(bool)
    ys, xs = _blocks(idx, cols)
    cc = ch[:, None, None]
    ok = np.zeros(len(idx), bool)
    pending = np.arange(len(idx))
    for attempt in (1, 2, 3):
        if pending.size == 0:
            break
        y, x, c = ys[pending], xs[pending], cc[pending]
        block = vals[y, x, c]
        spec = _M @ block @ _M.T
        diff = spec[:, 2, 3] - spec[:, 3, 2]
        target = np.where(want[pending], FD_STRENGTH * attempt, -FD_STRENGTH * attempt)
        adj = (target - diff) / 2.0
        spec[:, 2, 3] += adj
        spec[:, 3, 2] -= adj
        out = _M.T @ spec @ _M
        vals[y, x, c] = np.floor(np.clip(out, 0.0, 255.0) + 0.5)  # clampColor: std::round of a value >= 0
        good = _dct_bits(vals[y, x, c]) == want[pending]
        ok[pending[good]] = True
        pending = pending[~good]
    per_bit = ok.reshape(n, FD_REDUNDANCY).sum(1)
    if (per_bit <= FD_REDUNDANCY // 2).any():
        raise StegoError("The frequency-domain watermark could not be written (blocks saturated); "
                         "ImageToolbox fails with ERROR_PIXELS_NOT_ENOUGH in this case")
    return from_storage(vals.astype(np.int32), rgba)


def fd_extract(rgba: np.ndarray, is_image: bool) -> bytes | None:
    h, w = rgba.shape[:2]
    cap = fd_capacity(h, w)
    if cap < HEADER_BITS:
        return None
    cols = w // 8
    vals = to_storage(rgba).astype(np.float64)

    def read(start, count):
        idx = np.repeat(np.arange(start, start + count), FD_REDUNDANCY)
        ch = np.array(CHANNEL_ORDER)[np.tile(np.arange(FD_REDUNDANCY), count)]
        ys, xs = _blocks(idx, cols)
        b = _dct_bits(vals[ys, xs, ch[:, None, None]]).reshape(count, FD_REDUNDANCY).sum(1)
        return (b > FD_REDUNDANCY // 2).astype(np.uint8)

    magic, img_bit, length = _parse_header(read)
    if magic != FD_MAGIC or img_bit != is_image:
        return None
    if HEADER_BITS + length * 8 > cap:
        return None
    return np.packbits(read(HEADER_BITS, length * 8)).tobytes()


# ---------------------------------------------------------------------------
# payloads (BitmapUtils)
# ---------------------------------------------------------------------------

def _pil(rgba: np.ndarray) -> Image.Image:
    return Image.fromarray(rgba, "RGBA" if rgba.shape[2] == 4 else "RGB")


def image_payload_lsb(wm: np.ndarray) -> str:
    """``BitmapUtils.bitmapToString``: PNG, Base64 NO_WRAP."""
    buf = io.BytesIO()
    _pil(wm).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_payload_fd(wm: np.ndarray, max_length: int) -> str | None:
    """``BitmapUtils.bitmapToFdString``: JPEG quality 85..35 (step 10), then 3/4 downscales until it fits."""
    cur = wm
    if cur.shape[2] == 4:  # Android's JPEG encoder drops alpha of the premultiplied pixels (black background)
        a = cur[..., 3:4].astype(np.float32) / 255
        cur = np.clip(cur[..., :3] * a + 0.5, 0, 255).astype(np.uint8)
    while True:
        for q in range(85, 34, -10):
            buf = io.BytesIO()
            _pil(cur).save(buf, format="JPEG", quality=q)
            enc = base64.b64encode(buf.getvalue()).decode("ascii")
            if len(enc) <= max_length:
                return enc
        h, w = cur.shape[:2]
        if w == 1 and h == 1:
            return None
        nw, nh = max(1, w * 3 // 4), max(1, h * 3 // 4)
        cur = np.asarray(_pil(cur).resize((nw, nh), Image.Resampling.BILINEAR))


def decode_image_payload(text: str) -> tuple[bytes, Image.Image] | None:
    """``BitmapUtils.stringToBitmap`` (Base64.DEFAULT ignores non-alphabet characters)."""
    try:
        raw = base64.b64decode(text, validate=False)
        im = Image.open(io.BytesIO(raw))
        im.load()
        return raw, im
    except Exception:
        return None


@dataclass
class Detection:
    method: str
    kind: str  # "text" | "image"
    text: str | None = None
    image_bytes: bytes | None = None
    image: Image.Image | None = None


def detect(rgba: np.ndarray, method: str = "auto") -> Detection | None:
    """``AutomaticDetectionTask`` order: LSB text, LSB image, FD text, FD image."""
    order = {"auto": ("lsb", "fd"), "lsb": ("lsb",), "fd": ("fd",)}[method]
    for m in order:
        ex = lsb_extract if m == "lsb" else fd_extract
        raw = ex(rgba, False)
        if raw is not None:
            return Detection(m, "text", text=from_modified_utf8(raw))
        raw = ex(rgba, True)
        if raw is not None:
            dec = decode_image_payload(from_modified_utf8(raw))
            if dec is not None:
                return Detection(m, "image", image_bytes=dec[0], image=dec[1])
    return None
