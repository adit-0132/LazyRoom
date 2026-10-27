"""Palette file formats: a pure-Python port of ImageToolbox's palette library.

Ported from ImageToolbox by T8RIN (https://github.com/T8RIN/ImageToolbox, Apache-2.0),
``lib/palette/src/main/java/com/t8rin/palette`` (PaletteFormat, PaletteColor, coders/*, utils/*), which
itself follows dagronf's ColorPaletteCodable. Byte layouts, endianness, string encodings, scaling,
truncation/rounding, defaults and quirks follow the Kotlin code; the androidx ``ColorUtils`` LAB
maths and Compose's 8-bit colour quantisation are reproduced as well.

Deliberate deviations (marked ``DEVIATION`` below):
  * ANDROID_XML decode: upstream applies ARGB byte order to ``#RGB``/``#RRGGBB`` values, so
    ``#FF0000`` decodes as blue; colours without alpha digits are read as RGB (as Android does).
  * ACO decode: LAB a/b are read as signed int16 (Adobe spec); upstream reads them unsigned.
  * VGA_24BIT/VGA_18BIT decode: the ``; Names:`` header is cut at the newline *byte*; upstream uses
    the UTF-16 char index as a byte offset, corrupting non-ASCII-named files.
Environment differences: IMAGE uses Pillow instead of android.graphics (same 32x32 swatch PNG plus
name trailer; PNG bytes differ, and Android's premultiplied-alpha rounding is not replicated); zip
formats use ``zipfile`` (same entries/order/compression methods, no data descriptors); XML uses
ElementTree replayed as SAX events, with DTD/entity declarations rejected and zip entries capped.
"""
from __future__ import annotations

import io
import json
import math
import re
import struct
import time
import uuid
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

from PIL import Image

__all__ = [
    "EXPORT_ORDER", "FORMATS", "ColorGroup", "Palette", "PaletteColor", "PaletteFormatError",
    "decode", "decode_order_for_filename", "encode", "formats_for_filename",
]


class PaletteFormatError(ValueError):
    """A palette could not be decoded or encoded."""


_err = PaletteFormatError
_INT_MIN, _INT_MAX = -(2 ** 31), 2 ** 31 - 1
_ZIP_LIMIT = 50 * 1024 * 1024
_U16 = {False: "utf-16-be", True: "utf-16-le"}

# ---------------------------------------------------------------------------
# Kotlin / Java numeric and text semantics
# ---------------------------------------------------------------------------


def _f32(x: float) -> float:
    """Round to IEEE float32 (Kotlin ``toFloat()``)."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:  # coerceIn (NaN passes through)
    return lo if x < lo else hi if x > hi else x


def _toint(x: float) -> int:  # Double.toInt(): truncate, NaN -> 0, saturate
    if x != x:
        return 0
    if not math.isfinite(x):
        return _INT_MAX if x > 0 else _INT_MIN
    return max(_INT_MIN, min(_INT_MAX, int(x)))


def _b255(x: float) -> int:  # (x * 255).toInt().coerceIn(0, 255)
    return min(max(_toint(x * 255), 0), 255)


def _jround(x: float) -> int:  # Math.round / roundToInt(): ties toward +infinity
    if not math.isfinite(x):
        return _toint(x)
    f = math.floor(x)
    return int(f) + (1 if x - f >= 0.5 else 0)


def _kround(x: float) -> int:  # kotlin.math.round(x).toInt(): ties to even
    return _toint(x) if not math.isfinite(x) else int(round(x))


def _jstr(x: float) -> str:
    """``Double.toString()`` (shortest repr digits, Java layout: ``1.0``, ``0.25``, ``1.0E-5``)."""
    x = float(x)
    if x != x:
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0:
        return "-0.0" if math.copysign(1.0, x) < 0 else "0.0"
    sign, d = ("-" if x < 0 else ""), Decimal(repr(abs(x)))
    if 1e-3 <= abs(x) < 1e7:
        ip, _, fp = format(d, "f").partition(".")
        return f"{sign}{ip}.{fp.rstrip('0') or '0'}"
    digits = "".join(map(str, d.as_tuple().digits)).rstrip("0") or "0"
    return f"{sign}{digits[0]}.{digits[1:] or '0'}E{d.adjusted()}"


def _javafmt(x: float, places: int) -> str:  # String.format(Locale.US, "%.Nf"): HALF_UP
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return format(q, "f")


def _decfmt(x: float, max_frac: int, min_frac: int = 0, min_int: int = 1) -> str:
    """java.text.DecimalFormat with Locale.US symbols (HALF_EVEN), e.g. "0.0000", "#.###"."""
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-max_frac), rounding=ROUND_HALF_EVEN)
    s = format(q, "f")
    sign = "-" if s.startswith("-") else ""
    ip, _, fp = s.lstrip("-").partition(".")
    ip, fp = ip.lstrip("0").rjust(min_int, "0"), fp.rstrip("0").ljust(min_frac, "0")
    return sign + (ip or ("" if fp else "0")) + ("." + fp if fp else "")


def _text(data: bytes) -> str:  # InputStream.readText() (UTF-8, malformed -> U+FFFD)
    return bytes(data).decode("utf-8", "replace")


def _lines(text: str) -> list[str]:  # CharSequence.lines()
    return re.split(r"\r\n|\n|\r", text)


_KINT = {10: re.compile(r"[+-]?[0-9]+"), 16: re.compile(r"[+-]?[0-9a-fA-F]+")}
_KDOUBLE = re.compile(
    r"[\x00-\x20]*[+-]?(?:NaN|Infinity|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?[fFdD]?)[\x00-\x20]*")


def _int(s: str | None, base: int = 10) -> int | None:  # String.toIntOrNull(radix)
    if s is None or not _KINT[base].fullmatch(s):
        return None
    v = int(s, base)
    return v if _INT_MIN <= v <= _INT_MAX else None


def _double(s: str | None) -> float | None:  # String.toDoubleOrNull()
    if s is None or not _KDOUBLE.fullmatch(s):
        return None
    return float(s.strip("".join(map(chr, range(33)))).rstrip("fFdD"))


def _xml_decoded(s: str) -> str:
    return s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace(
        "&quot;", '"').replace("&apos;", "'")


def _xml_escaped(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(
        '"', "&quot;").replace("'", "&apos;")


def _xd(v: str | None) -> str | None:
    return None if v is None else _xml_decoded(v)


def _sax(data: bytes, start, end=None) -> None:
    """Replay an XML document as SAX events using ElementTree.

    ``start(qname, attrs)`` / ``end(qname, chars)``; names keep their document prefixes like a
    SAX qName, ``chars`` is the text since the previous start/end event (what upstream handlers
    accumulate). DTD/entity declarations are refused before parsing (entity-expansion attacks).
    """
    data = bytes(data)
    for marker in ("<!DOCTYPE", "<!ENTITY"):
        if any(marker.encode(enc) in data for enc in ("ascii", "utf-16-le", "utf-16-be")):
            raise _err("XML DTD/entity declarations are not allowed")
    prefixes = {"http://www.w3.org/XML/1998/namespace": "xml"}

    def qname(tag: str) -> str:
        if tag[:1] != "{":
            return tag
        uri, local = tag[1:].split("}", 1)
        return f"{prefixes[uri]}:{local}" if prefixes.get(uri) else local

    for event, obj in ET.iterparse(io.BytesIO(data), events=("start-ns", "start", "end")):
        if event == "start-ns":
            prefixes[obj[1]] = obj[0]
        elif event == "start":
            start(qname(obj.tag), {qname(k): v for k, v in obj.attrib.items()})
        elif end is not None:
            end(qname(obj.tag), (obj[-1].tail if len(obj) else obj.text) or "")


def _zip_entry(data: bytes, match) -> bytes | None:
    """First zip entry whose name satisfies ``match`` (size-capped against zip bombs)."""
    with zipfile.ZipFile(io.BytesIO(bytes(data))) as zf:
        for info in zf.infolist():
            if match(info.filename):
                if info.file_size > _ZIP_LIMIT:
                    raise _err("Zip entry too large")
                with zf.open(info) as fh:
                    content = fh.read(_ZIP_LIMIT + 1)
                if len(content) > _ZIP_LIMIT:
                    raise _err("Zip entry too large")
                return content
    return None


def _zip(entries: list[tuple[str, bytes, bool]]) -> bytes:  # (name, data, stored)
    buf, now = io.BytesIO(), time.localtime()[:6]
    with zipfile.ZipFile(buf, "w") as zf:
        for name, payload, stored in entries:
            info = zipfile.ZipInfo(name, date_time=now)
            info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
            zf.writestr(info, payload)
    return buf.getvalue()


def _kjson(v, pretty: bool = False, level: int = 0) -> str:
    """kotlinx.serialization JSON layout (compact, or prettyPrint with 4-space indent)."""
    if isinstance(v, (dict, list)):
        is_dict = isinstance(v, dict)
        sep = ": " if pretty else ":"
        parts = [(json.dumps(k, ensure_ascii=False) + sep if is_dict else "") + _kjson(x, pretty, level + 1)
                 for k, x in (v.items() if is_dict else enumerate(v))]
        o, c = "{}" if is_dict else "[]"
        if not parts:
            return o + c
        if not pretty:
            return o + ",".join(parts) + c
        pad = "\n" + "    " * (level + 1)
        return o + pad + ("," + pad).join(parts) + "\n" + "    " * level + c
    if isinstance(v, float):
        return _jstr(v)
    return json.dumps(v, ensure_ascii=False)


def _json_load(data: bytes):
    def reject(token):
        raise ValueError(f"Unexpected JSON token {token}")
    return json.loads(_text(data), parse_constant=reject)


_REQUIRED = object()


def _jf(obj, key: str, typ, default=_REQUIRED, nullable: bool = False):
    """Typed JSON field access with kotlinx semantics (unknown keys ignored, types enforced)."""
    if not isinstance(obj, dict):
        raise _err("Expected a JSON object")
    if key not in obj:
        if default is _REQUIRED:
            raise _err(f"Missing JSON field '{key}'")
        return default
    v = obj[key]
    if v is None and nullable:
        return None
    if typ is float:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    elif typ is int:
        if isinstance(v, int) and not isinstance(v, bool) and _INT_MIN <= v <= _INT_MAX:
            return v
    elif isinstance(v, typ):
        return v
    raise _err(f"Invalid JSON field '{key}'")


def _jfloats(values: list) -> list[float]:
    return [_jf({"v": v}, "v", float) for v in values]


# ---------------------------------------------------------------------------
# Binary reader / writer helpers (utils/BytesReader.kt, utils/BytesWriter.kt)
# ---------------------------------------------------------------------------


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = bytes(data), 0

    def read(self, n: int) -> bytes:
        if n < 0:
            raise ValueError("Negative length")
        if self.pos + n > len(self.data):
            raise EOFError("Unexpected end of stream")
        self.pos += n
        return self.data[self.pos - n:self.pos]

    def num(self, fmt: str, le: bool = False):
        return struct.unpack(("<" if le else ">") + fmt, self.read(struct.calcsize(fmt)))[0]

    def u8(self) -> int:
        return self.read(1)[0]

    def u16(self, le=False): return self.num("H", le)
    def i16(self, le=False): return self.num("h", le)
    def u32(self, le=False): return self.num("I", le)
    def i32(self, le=False): return self.num("i", le)
    def f32(self, le=False): return self.num("f", le)

    def seek_set(self, pos: int) -> None:
        if not 0 <= pos <= len(self.data):
            raise EOFError("Position out of range")
        self.pos = pos

    def skip(self, n: int) -> None:  # BytesReader.seek(count)
        if self.pos + n > len(self.data):
            raise EOFError("Unexpected end of stream")
        self.pos += n

    def try_skip(self, n: int) -> None:
        if self.pos + n <= len(self.data):
            self.pos += n

    def find(self, pattern: bytes) -> int:  # findPattern (does not move)
        return self.data.find(pattern, self.pos)

    def seek_to(self, pattern: bytes) -> None:
        """seekToNextInstanceOfPattern/ASCII: upstream's naive matcher, which consumes the byte that
        breaks a partial match without retrying it as a match start (so "rrloC" misses "rloC").
        Bytes before a first-byte candidate are skipped with ``bytes.find`` (same result, faster)."""
        data, n, pos = self.data, len(pattern), self.pos
        while (pos := data.find(pattern[:1], pos)) >= 0:
            k, pos = 1, pos + 1
            while k < n and pos < len(data) and data[pos] == pattern[k]:
                k, pos = k + 1, pos + 1
            if k == n:
                self.pos = pos - n
                return
            pos += 1  # the mismatching byte is consumed
        raise EOFError("Pattern not found")

    def ascii(self, n): return self.read(n).decode("ascii", "replace")
    def latin1(self, n): return self.read(n).decode("latin-1")
    def utf8(self, n): return self.read(n).decode("utf-8", "replace")
    def utf16(self, n, le=False): return self.read(2 * n).decode(_U16[le], "replace")

    def utf16z(self, le: bool = False) -> str:  # null-terminated UTF-16
        out = bytearray()
        while self.pos + 1 < len(self.data):
            pair = self.data[self.pos:self.pos + 2]
            self.pos += 2
            if pair == b"\0\0":
                break
            out += pair
        return out.decode(_U16[le], "replace")

    def adobe(self) -> str:  # UInt32 length in UTF-16 units + UTF-16BE, trailing NULs trimmed
        n = self.i32()
        return "" if n <= 0 else self.read(2 * n).decode("utf-16-be", "replace").rstrip("\x00")

    def pascal16(self, le: bool) -> str:  # UInt16 length + UTF-16
        n = self.u16(le)
        return "" if n == 0 else self.read(2 * n).decode(_U16[le], "replace")


def _u16s(s: str, le: bool = False) -> bytes:
    return s.encode(_U16[le], "surrogatepass")


def _adobe(s: str) -> bytes:
    b = _u16s(s)
    return struct.pack(">I", len(b) // 2) + b


def _pascal16(s: str, le: bool = True) -> bytes:
    b = _u16s(s, le)
    if len(b) // 2 > 65535:
        raise _err("String too long for Pascal UTF-16 (max 65535 characters)")
    return struct.pack("<H" if le else ">H", len(b) // 2) + b


def _pf(order: str, v: float) -> bytes:
    return struct.pack(order + "f", _f32(v))


# ---------------------------------------------------------------------------
# Colour model (PaletteColor.kt, ColorGroup.kt, Palette.kt) + androidx ColorUtils LAB maths
# ---------------------------------------------------------------------------

_SPACE_SIZE = {"rgb": 3, "cmyk": 4, "lab": 3, "gray": 1}
_TYPES = ("global", "spot", "normal")


def _rgb8_to_lab(r: int, g: int, b: int) -> list[float]:  # ColorUtils.RGBToLAB
    def lin(c):
        c /= 255.0
        return c / 12.92 if c < 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    def pivot(c):
        return c ** (1 / 3.0) if c > 0.008856 else (903.3 * c + 16) / 116

    sr, sg, sb = lin(r), lin(g), lin(b)
    x = pivot(100 * (sr * 0.4124 + sg * 0.3576 + sb * 0.1805) / 95.047)
    y = pivot(100 * (sr * 0.2126 + sg * 0.7152 + sb * 0.0722) / 100.0)
    z = pivot(100 * (sr * 0.0193 + sg * 0.1192 + sb * 0.9505) / 108.883)
    return [max(0.0, 116 * y - 16), 500 * (x - y), 200 * (y - z)]


def _lab_to_rgb8(l: float, a: float, b: float) -> tuple[int, int, int]:  # ColorUtils.LABToColor
    fy = (l + 16) / 116
    fx, fz = a / 500 + fy, fy - b / 200
    xr = fx ** 3 if fx ** 3 > 0.008856 else (116 * fx - 16) / 903.3
    yr = fy ** 3 if l > 903.3 * 0.008856 else l / 903.3
    zr = fz ** 3 if fz ** 3 > 0.008856 else (116 * fz - 16) / 903.3
    x, y, z = xr * 95.047, yr * 100.0, zr * 108.883
    rgb = ((x * 3.2406 + y * -1.5372 + z * -0.4986) / 100,
           (x * -0.9689 + y * 1.8758 + z * 0.0415) / 100,
           (x * 0.0557 + y * -0.2040 + z * 1.0570) / 100)
    out = (1.055 * c ** (1 / 2.4) - 0.055 if c > 0.0031308 else 12.92 * c for c in rgb)
    return tuple(min(max(_jround(c * 255), 0), 255) for c in out)


def _q8(v: float) -> int:  # Compose sRGB Color(): (v * 255f + 0.5f).toInt() in float32
    return _toint(_f32(_f32(v * 255.0) + 0.5))


def _sector(h: float, c: float, x: float) -> tuple[float, float, float]:
    if h < 1.0 / 6:
        return c, x, 0.0
    if h < 2.0 / 6:
        return x, c, 0.0
    if h < 3.0 / 6:
        return 0.0, c, x
    if h < 4.0 / 6:
        return 0.0, x, c
    if h < 5.0 / 6:
        return x, 0.0, c
    return c, 0.0, x


@dataclass
class PaletteColor:
    """A palette colour. ``components`` use upstream units: RGB/CMYK/Gray 0..1, LAB L 0..100, a/b ±128."""

    name: str
    color_space: str  # 'rgb' | 'cmyk' | 'lab' | 'gray'
    components: list[float]
    alpha: float = 1.0
    color_type: str = "global"  # 'global' | 'spot' | 'normal'

    def __post_init__(self):
        self.components = [float(c) for c in self.components]
        self.alpha = float(self.alpha)
        if len(self.components) != _SPACE_SIZE.get(self.color_space, -1):
            raise _err("Invalid color component count for model type")
        if self.color_type not in _TYPES:
            raise _err(f"Unknown color type: {self.color_type}")

    @classmethod
    def rgb(cls, r: float, g: float, b: float, a: float = 1.0, name: str = "", color_type: str = "global"):
        return cls(name, "rgb", [_clamp(r), _clamp(g), _clamp(b)], _clamp(a), color_type)

    @classmethod
    def cmyk(cls, c, m, y, k, alpha=1.0, name="", color_type="global"):
        return cls(name, "cmyk", [_clamp(c), _clamp(m), _clamp(y), _clamp(k)], _clamp(alpha), color_type)

    @classmethod
    def gray(cls, white, alpha=1.0, name="", color_type="global"):
        return cls(name, "gray", [_clamp(white)], _clamp(alpha), color_type)

    @classmethod
    def lab(cls, l, a, b, alpha=1.0, name="", color_type="global"):
        return cls(name, "lab", [l, a, b], _clamp(alpha), color_type)

    @classmethod
    def hsb(cls, h, s, b, alpha=1.0, name="", color_type="global"):
        h, s, b = _clamp(h), _clamp(s), _clamp(b)
        c = b * s
        r, g, bl = _sector(h, c, c * (1 - abs(math.fmod(h * 6, 2) - 1)))
        m = b - c
        return cls.rgb(_clamp(r + m), _clamp(g + m), _clamp(bl + m), alpha, name, color_type)

    @classmethod
    def hsl(cls, h, s, l, alpha=1.0, name="", color_type="global"):
        h, s, l = _clamp(h), _clamp(s), _clamp(l)
        c = (1 - abs(2 * l - 1)) * s
        r, g, b = _sector(h, c, c * (1 - abs(math.fmod(h * 6, 2) - 1)))
        m = l - c / 2
        return cls.rgb(_clamp(r + m), _clamp(g + m), _clamp(b + m), alpha, name, color_type)

    def _rgb(self) -> tuple[float, float, float, float]:  # PaletteColor.toRgb() (unclamped)
        c = self.components
        if self.color_space == "rgb":
            return c[0], c[1], c[2], self.alpha
        if self.color_space == "cmyk":
            k = 1.0 - c[3]
            return (1.0 - c[0]) * k, (1.0 - c[1]) * k, (1.0 - c[2]) * k, self.alpha
        if self.color_space == "gray":
            return c[0], c[0], c[0], self.alpha
        r, g, b = _lab_to_rgb8(*c)  # Compose Color(LABToColor(..)).red etc. are float32 ch / 255f
        return _f32(r / 255), _f32(g / 255), _f32(b / 255), self.alpha

    def to_rgba(self) -> tuple[float, float, float, float]:
        """(r, g, b, a) in 0..1 via upstream's toRgb() (CMYK/LAB/Gray conversions as PaletteColor.kt)."""
        return tuple(_clamp(v) for v in self._rgb())

    def to_cmyk(self) -> tuple[float, float, float, float, float]:
        if self.color_space == "cmyk":
            return (*self.components, self.alpha)
        r, g, b, _ = self.to_rgba()
        k = 1.0 - max(r, g, b)
        f = (lambda v: (1.0 - v - k) / (1.0 - k)) if k < 1.0 else (lambda v: 0.0)
        return f(r), f(g), f(b), k, self.alpha

    def to_lab(self) -> tuple[float, float, float, float]:
        if self.color_space == "lab":
            return (*self.components, self.alpha)
        argb = self.to_argb()
        return (*_rgb8_to_lab((argb >> 16) & 255, (argb >> 8) & 255, argb & 255), self.alpha)

    def to_hsb(self) -> tuple[float, float, float, float]:
        r, g, b, _ = self.to_rgba()
        mx, mn = max(r, g, b), min(r, g, b)
        d = mx - mn
        if d == 0.0:
            h = 0.0
        elif mx == r:
            h = math.fmod((g - b) / d, 6.0) * 60.0
        elif mx == g:
            h = ((b - r) / d + 2.0) * 60.0
        else:
            h = ((r - g) / d + 4.0) * 60.0
        return (h + 360.0 if h < 0 else h) / 360.0, (0.0 if mx == 0.0 else d / mx), mx, self.alpha

    def to_argb(self) -> int:
        """Compose ``toComposeColor().toArgb()`` (8-bit quantised, signed 32-bit like Kotlin Int)."""
        c, a = self.components, _f32(_clamp(self.alpha))
        if self.color_space == "lab":
            r, g, b = _lab_to_rgb8(*c)
        else:
            if self.color_space == "rgb":
                fr, fg, fb = (_f32(_clamp(v)) for v in c)
            elif self.color_space == "gray":
                fr = fg = fb = _f32(_clamp(c[0]))
            else:
                cc, m, y, k = (_f32(_clamp(v)) for v in c)
                fr, fg, fb = (_f32(_f32(1.0 - v) * _f32(1.0 - k)) for v in (cc, m, y))
            r, g, b = _q8(fr), _q8(fg), _q8(fb)
        v = (_q8(a) << 24) | (r << 16) | (g << 8) | b
        return v - (1 << 32) if v & 0x80000000 else v

    def converted(self, space: str) -> PaletteColor:
        if space == self.color_space:
            return self
        if space == "cmyk":
            c, m, y, k, a = (_clamp(v) for v in self.to_cmyk())
            return PaletteColor.cmyk(c, m, y, k, a, self.name, self.color_type)
        if space == "lab":
            return PaletteColor.lab(*self.to_lab(), self.name, self.color_type)
        r, g, b, a = self.to_rgba()
        if space == "gray":
            return PaletteColor.gray(0.299 * r + 0.587 * g + 0.114 * b, a, self.name, self.color_type)
        return PaletteColor.rgb(r, g, b, a, self.name, self.color_type)

    def hex(self) -> str:
        """'#RRGGBB', or '#RRGGBBAA' when alpha < 1 (upstream hexString: channels truncated)."""
        r, g, b, a = self.to_rgba()
        return _hexrgb(r, g, b, a, "RGBA" if a < 1.0 else "RGB", True, True)


@dataclass
class ColorGroup:
    name: str = ""
    colors: list[PaletteColor] = field(default_factory=list)


@dataclass
class Palette:
    name: str = ""
    colors: list[PaletteColor] = field(default_factory=list)
    groups: list[ColorGroup] = field(default_factory=list)

    def all_colors(self) -> list[PaletteColor]:
        return [*self.colors, *(c for g in self.groups for c in g.colors)]

    @property
    def total_color_count(self) -> int:
        return len(self.colors) + sum(len(g.colors) for g in self.groups)

    @property
    def all_groups(self) -> list[ColorGroup]:  # global colours as a group called 'global'
        return [ColorGroup("global", list(self.colors)), *self.groups]


def _hexrgb(r, g, b, a, fmt: str, hashmark: bool = True, upper: bool = False) -> str:  # hexRGBString
    v = {"r": _b255(r), "g": _b255(g), "b": _b255(b), "a": _b255(a)}
    return ("#" if hashmark else "") + "".join(f"{v[ch]:02X}" if upper else f"{v[ch]:02x}" for ch in fmt.lower())


def _hexstr(c: PaletteColor, fmt: str, hashmark: bool = True, upper: bool = False) -> str:
    return _hexrgb(*c.to_rgba(), fmt, hashmark, upper)


def _extract_hex_rgba(s: str, fmt: str) -> tuple[float, float, float, float] | None:  # extractHexRGBA
    h = re.sub("[^0-9a-f]", "", s.lower())
    if h.startswith("0x"):
        h = h[2:]
    if not h or len(h) > 16:
        return None
    v, n = int(h, 16), len(h)
    if n == 3:
        q = ((v >> 8) & 0xF) * 17, ((v >> 4) & 0xF) * 17, (v & 0xF) * 17, 255
    elif n == 4:
        q = ((v >> 12) & 0xF) * 17, ((v >> 8) & 0xF) * 17, ((v >> 4) & 0xF) * 17, (v & 0xF) * 17
    elif n == 6:
        q = (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF, 255
    elif n == 8:
        q = (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF
    else:
        return None
    c0, c1, c2, c3 = (x / 255.0 for x in q)
    alpha = n in (4, 8)
    return {"RGB": (c0, c1, c2, 1.0), "BGR": (c2, c1, c0, 1.0),
            "RGBA": (c0, c1, c2, c3 if alpha else 1.0), "ARGB": (c1, c2, c3, c0 if alpha else 1.0),
            "BGRA": (c2, c1, c0, c3 if alpha else 1.0), "ABGR": (c3, c2, c1, c0 if alpha else 1.0)}[fmt]


def _hex_color(s: str, fmt: str = "RGBA", name: str = "", color_type: str = "normal") -> PaletteColor:
    rgba = _extract_hex_rgba(s, fmt)
    if rgba is None:
        raise _err(f"Invalid RGB hex string: {s}")
    r, g, b, a = (_clamp(v) for v in rgba)
    return PaletteColor(name, "rgb", [r, g, b], a, color_type)


def _as_space(colors: list[PaletteColor], space: str) -> list[PaletteColor]:
    return [c.converted(space) for c in colors]


# ---------------------------------------------------------------------------
# Coders (coders/*.kt), one encode/decode pair per format
# ---------------------------------------------------------------------------

# ACB: Adobe Color Book -------------------------------------------------------
_ACB_SPACES = {0: ("rgb", 3), 2: ("cmyk", 4), 7: ("lab", 3), 8: ("gray", 1)}


def _dec_acb(data: bytes) -> Palette:
    r = _Reader(data)
    if r.ascii(4) != "8BCB":
        raise _err("Invalid BOM")
    r.u16(), r.u16()  # version, identifier
    title = r.adobe()
    if title.startswith("$$$"):
        title = title.split("=")[-1]
    for _ in range(3):  # prefix, suffix, description
        r.adobe()
    count = r.u16()
    r.u16(), r.u16()  # page size, page selector offset
    space_id = r.u16()
    if space_id not in _ACB_SPACES:
        raise _err("Unsupported color space")
    space, n = _ACB_SPACES[space_id]
    colors = []
    for _ in range(count):
        name, code, ch = r.adobe(), r.ascii(6), r.read(n)
        if not name.strip() and not code.strip():
            continue
        if space == "cmyk":
            comps = [_clamp((255.0 - v) / 255.0) for v in ch]
        elif space == "lab":
            comps = [ch[0] / 2.55, ch[1] - 128.0, ch[2] - 128.0]
        else:
            comps = [_clamp(v / 255.0) for v in ch]
        colors.append(PaletteColor(name, space, comps))
    return Palette(title, colors)  # the optional trailing 8-byte spot identifier is ignored


def _enc_acb(p: Palette) -> bytes:
    cols = p.all_colors()
    if not cols:
        raise _err("Too few colors")
    target = cols[0].color_space
    if any(c.color_space != target for c in cols):
        target = "rgb"
    space_id = {"rgb": 0, "cmyk": 2, "lab": 7, "gray": 8}[target]
    out = bytearray(b"8BCB" + struct.pack(">HH", 1, 0) + _adobe(p.name or "Palette") + _adobe("") * 3)
    out += struct.pack(">HHHH", len(cols) & 0xFFFF, 0, 0, space_id)
    for c in cols:
        cc = c.converted(target)
        out += _adobe(cc.name or "Color")
        r, g, b, _ = cc.to_rgba()
        out += f"{_b255(r):02X}{_b255(g):02X}{_b255(b):02X}"[:6].encode("ascii")
        v = cc.components
        if target == "cmyk":
            out += bytes(_b255(1.0 - x) for x in v)
        elif target == "lab":
            out += bytes(min(max(_toint(x), 0), 255) for x in (v[0] * 2.55, v[1] + 128.0, v[2] + 128.0))
        else:
            out += bytes(_b255(x) for x in v)
    return bytes(out)


# ACO: Adobe Photoshop swatches (v1 + v2) ------------------------------------
def _dec_aco(data: bytes) -> Palette:
    r = _Reader(data)
    v1: list[PaletteColor] = []
    v2: list[PaletteColor] = []
    for typ in (1, 2):
        try:
            if r.u16() != typ:
                raise _err("Invalid version")
        except Exception:
            if typ == 2 and v1:
                return Palette(colors=v1)
            raise _err("Invalid format") from None
        for _ in range(r.u16()):
            space, c0, c1, c2, c3 = (r.u16() for _ in range(5))
            name = r.adobe() if typ == 2 else ""
            if space == 0:
                col = PaletteColor.rgb(c0 / 65535.0, c1 / 65535.0, c2 / 65535.0, name=name)
            elif space == 2:
                col = PaletteColor.cmyk(*((65535 - v) / 65535.0 for v in (c0, c1, c2, c3)), name=name)
            elif space == 8:
                col = PaletteColor.gray(c0 / 10000.0, name=name)
            elif space == 7:
                # DEVIATION: a/b are signed int16 per Adobe's spec (upstream reads them unsigned).
                sa, sb = (v - 65536 if v >= 32768 else v for v in (c1, c2))
                col = PaletteColor.lab(c0 / 100.0, sa / 100.0, sb / 100.0, name=name)
            elif space == 1:  # HSB, converted with upstream's unclamped hsbToRgb
                h, s, b = c0 / 65535.0, c1 / 65535.0, c2 / 65535.0
                cc = b * s
                rr, gg, bb = _sector(h, cc, cc * (1 - abs(math.fmod(h * 6, 2) - 1)))
                col = PaletteColor.rgb(rr + b - cc, gg + b - cc, bb + b - cc, name=name)
            else:
                col = PaletteColor.rgb(1.0, 0.0, 0.0, 0.5, name="Unsupported Colorspace")
            (v1 if typ == 1 else v2).append(col)
    p = Palette(colors=v2 or v1)
    if not p.total_color_count:
        raise _err("Invalid format")
    return p


def _enc_aco(p: Palette) -> bytes:
    cols = p.all_colors()
    out = bytearray()
    for typ in (1, 2):
        out += struct.pack(">HH", typ, len(cols) & 0xFFFF)
        for c in cols:
            if c.color_space == "cmyk":
                model, vals = 2, [65535 - _toint(v * 65535) for v in c.components]
            elif c.color_space == "gray":
                model, vals = 8, [_toint(c.components[0] * 10000), 0, 0, 0]
            else:  # RGB; LAB is converted to RGB
                r, g, b, _ = c.converted("rgb").to_rgba()
                model, vals = 0, [_toint(r * 65535), _toint(g * 65535), _toint(b * 65535), 0]
            out += struct.pack(">5H", model, *(v & 0xFFFF for v in vals))
            if typ == 2:
                out += _adobe(c.name)
    return bytes(out)


# ACT: Adobe Color Table --------------------------------------------------------
def _dec_act(data: bytes) -> Palette:
    r = _Reader(data)
    colors = []
    for _ in range(256):
        b = r.read(3)
        colors.append(PaletteColor.rgb(b[0] / 255.0, b[1] / 255.0, b[2] / 255.0))
    try:
        n = r.i16()
        if 1 <= n < 256:
            colors = colors[:n]
    except EOFError:
        pass
    try:
        transparent = r.i16()
        if 0 <= transparent < len(colors):
            colors[transparent].alpha = 0.0
    except EOFError:
        pass
    tag = "; ACT_NAMES: "  # non-standard names extension (read only, never written)
    line = next((ln for ln in _lines(_text(r.data[r.pos:])) if ln.startswith("; ACT_NAMES:")), None)
    if line is not None and len(line) >= len(tag):
        for i, name in enumerate(line[len(tag):].strip().split("|")):
            if i < len(colors) and name:
                colors[i].name = name
    return Palette(colors=colors)


def _enc_act(p: Palette) -> bytes:
    cols = [c.to_rgba() for c in p.all_colors()][:256]
    out = bytearray()
    for i in range(256):
        out += bytes(_b255(v) for v in cols[i][:3]) if i < len(cols) else b"\0\0\0"
    if len(cols) < 256:
        out += struct.pack(">HH", len(cols), 0xFFFF)
    return bytes(out)


# ANDROID_XML: <resources><color name="..">#AARRGGBB</color></resources> -------
def _dec_android_xml(data: bytes) -> Palette:
    colors: list[PaletteColor] = []
    inside, current, name = False, "", None

    def start(tag, attrs):
        nonlocal inside, current, name
        t = tag.split(":", 1)[-1].lower()
        if t == "resources":
            inside = True
        elif t == "color":
            current, name = "color", _xd(attrs.get("name"))

    def end(tag, chars):
        nonlocal inside, current, name
        t = tag.split(":", 1)[-1].lower()
        if t == "resources":
            inside = False
        elif t == "color":
            if inside and current == "color":
                s = chars.strip()
                # DEVIATION: upstream always uses ARGB order; without alpha digits that misreads
                # #RRGGBB as (GG, BB, FF). Android's Color.parseColor treats 3/6 digits as RGB.
                fmt = "ARGB" if len(re.sub("[^0-9a-f]", "", s.lower())) in (4, 8) else "RGB"
                try:
                    colors.append(_hex_color(s, fmt, name if name is not None else f"color_{len(colors)}"))
                except PaletteFormatError:
                    pass
            current, name = "", None

    _sax(data, start, end)
    if not colors:
        raise _err("Invalid format")
    return Palette(colors=colors)


def _enc_android_xml(p: Palette) -> bytes:
    s = '<?xml version="1.0" encoding="utf-8"?>\n<resources>\n'
    for i, c in enumerate(p.all_colors()):
        name = _xml_escaped((c.name or f"color_{i}").replace(" ", "_"))
        s += f'   <color name="{name}">{_hexstr(c, "ARGB", True, True)}</color>\n'
    return (s + "</resources>\n").encode()


# ASE: Adobe Swatch Exchange -------------------------------------------------
_ASE_MODELS = {"CMYK": "cmyk", "RGB ": "rgb", "LAB ": "lab", "Gray": "gray"}


def _dec_ase(data: bytes) -> Palette:
    r = _Reader(data)
    if r.read(4) != b"ASEF":
        raise _err("Invalid ASE header")
    r.u16(), r.u16()  # version (unknown versions accepted)
    colors: list[PaletteColor] = []
    groups: list[ColorGroup] = []
    group = None
    for _ in range(r.i32()):
        block = r.u16()
        r.u32()  # block length (ignored)
        if block == 0xC001:
            if group is not None:
                raise _err("Group already open")
            r.u16()
            group = ColorGroup(r.utf16z())
        elif block == 0xC002:
            if group is None:
                raise _err("Group not open")
            groups.append(group)
            group = None
        elif block == 0x0001:
            r.u16()
            name, mode = r.utf16z(), r.ascii(4)
            if mode not in _ASE_MODELS:
                raise _err(f"Unknown color mode: {mode}")
            space = _ASE_MODELS[mode]
            comps = [r.f32() for _ in range(_SPACE_SIZE[space])]
            if space == "lab":
                comps[0] *= 100.0
            ctype = r.u16()
            if ctype > 2:
                raise _err(f"Unknown color type: {ctype}")
            (group.colors if group is not None else colors).append(
                PaletteColor(name, space, comps, 1.0, _TYPES[ctype]))
        else:
            raise _err("Unknown block type")
    if group is not None:
        groups.append(group)
    return Palette(colors=colors, groups=groups)


def _ase_name(s: str) -> bytes:
    b = _u16s(s)
    return struct.pack(">H", (len(b) // 2 + 1) & 0xFFFF) + b + b"\0\0"


def _ase_color(c: PaletteColor) -> bytes:
    comps = list(c.components)
    if c.color_space == "lab":
        comps[0] /= 100.0
    body = _ase_name(c.name) + {"rgb": b"RGB ", "cmyk": b"CMYK", "lab": b"LAB ", "gray": b"Gray"}[c.color_space]
    body += b"".join(_pf(">", v) for v in comps) + struct.pack(">H", _TYPES.index(c.color_type))
    return struct.pack(">HI", 0x0001, len(body)) + body


def _enc_ase(p: Palette) -> bytes:
    blocks = len(p.colors) + 2 * len(p.groups) + sum(len(g.colors) for g in p.groups)
    out = bytearray(b"ASEF" + struct.pack(">HHI", 1, 0, blocks))
    for c in p.colors:
        out += _ase_color(c)
    for g in p.groups:
        head = _ase_name(g.name)
        out += struct.pack(">HI", 0xC001, len(head)) + head
        for c in g.colors:
            out += _ase_color(c)
        out += struct.pack(">HI", 0xC002, 0)
    return bytes(out)


# BASIC_XML: <palette name><color name hex r g b a/></palette> ----------------
def _dec_basic_xml(data: bytes) -> Palette:
    p = Palette()

    def start(tag, a):
        t = tag.split(":", 1)[-1].lower()
        if t == "palette" and a.get("name") is not None:
            p.name = _xml_decoded(a["name"])
        elif t == "color":
            name = _xd(a.get("name")) or ""
            r, g, b = (_int(a.get(k)) for k in "rgb")
            alpha = _int(a.get("a"))
            try:
                if a.get("hex") is not None:
                    p.colors.append(_hex_color(a["hex"], "RGBA", name))
                elif None not in (r, g, b):
                    p.colors.append(PaletteColor.rgb(r / 255.0, g / 255.0, b / 255.0,
                                                     (255 if alpha is None else alpha) / 255.0, name))
            except PaletteFormatError:
                pass

    _sax(data, start)
    if not p.total_color_count:
        raise _err("Invalid format")
    return p


def _enc_basic_xml(p: Palette) -> bytes:
    s = '<?xml version="1.0"?>\n<palette' + (f' name="{_xml_escaped(p.name)}"' if p.name else "") + ">\n"
    for c in p.all_colors():
        r, g, b, a = c.to_rgba()
        s += "<color" + (f' name="{_xml_escaped(c.name)}"' if c.name else "")
        s += f' hex="{_hexrgb(r, g, b, a, "RGBA", False)}"'
        s += f' r="{_toint(r * 255)}" g="{_toint(g * 255)}" b="{_toint(b * 255)}" a="{_toint(a * 255)}" />\n'
    return (s + "</palette>\n").encode()


# COREL_PAINTER: Corel Painter swatches (.txt) --------------------------------
_COREL_PAINTER_RE = re.compile(
    r"[ \t]*R[ \t]*:[ \t]*(\d*\.?\d+)[ \t,]*G[ \t]*:[ \t]*(\d*\.?\d+)[ \t,]*B[ \t]*:[ \t]*(\d*\.?\d+)[ \t,]*"
    r"(?:HV[ \t]*:[ \t]*(\d*\.?\d+)[ \t,]*)?(?:SV[ \t]*:[ \t]*(\d*\.?\d+)[ \t,]*)?"
    r"(?:VV[ \t]*:[ \t]*(\d*\.?\d+)[ \t,]*)?(.*)", re.A)


def _dec_corel_painter(data: bytes) -> Palette:
    text = _text(data)
    lines = _lines(text)
    keys = ("ROWS", "COLS", "WIDTH", "HEIGHT", "TEXTHEIGHT", "SPACING")
    if not text.startswith("ROWS ") or len(lines) < 7 or not all(lines[i].startswith(k) for i, k in enumerate(keys)):
        raise _err("Invalid format")
    colors = []
    for line in lines[6:]:
        m = _COREL_PAINTER_RE.search(line.strip())
        if m:
            r, g, b = (_int(m.group(i)) for i in (1, 2, 3))
            if None not in (r, g, b):
                colors.append(PaletteColor.rgb(_clamp(r / 255.0), _clamp(g / 255.0), _clamp(b / 255.0),
                                               name=m.group(7).strip()))
    return Palette(colors=colors)


def _enc_corel_painter(p: Palette) -> bytes:
    s = "ROWS 12\nCOLS 22\nWIDTH 16\nHEIGHT 16\nTEXTHEIGHT 0\nSPACING 1\n\n"
    for c in _as_space(p.all_colors(), "rgb"):
        r, g, b, _ = c.to_rgba()
        s += f"R: {_toint(r * 255)}, G:{_toint(g * 255)}, B:{_toint(b * 255)}  HV:0.00, SV:0.00, VV:0.00"
        s += (f"  {c.name}" if c.name else "") + "\n"
    return s.encode()


# COREL_DRAW: CorelDraw XML palette ------------------------------------------
def _dec_corel_draw(data: bytes) -> Palette:
    p = Palette()
    spaces: list[tuple[str, list[PaletteColor]]] = []
    group, in_spaces = None, False

    def start(tag, a):
        nonlocal group, in_spaces
        t = tag.strip().lower()
        if t == "palette":
            if _xd(a.get("name")):
                p.name = _xd(a["name"])
        elif t == "colorspaces":
            in_spaces = True
        elif t == "cs":
            spaces.append(((_xd(a.get("name")) or "").lower(), []))
        elif t == "page":
            group = ColorGroup(_xd(a.get("name")) or "")
        elif t == "color":
            cs = a["cs"].lower() if a.get("cs") is not None else None
            name = _xd(a.get("name")) or ""
            v = [x for x in (_double(s.strip()) for s in (a.get("tints") or "").split(",")) if x is not None]
            col = None
            if cs == "cmyk" and len(v) >= 4:
                col = PaletteColor.cmyk(*v[:4], name=name)
            elif cs == "rgb" and len(v) >= 3:
                col = PaletteColor.rgb(*v[:3], name=name)
            elif cs == "lab" and len(v) >= 3:
                col = PaletteColor.lab(v[0] * 100.0, v[1] * 256.0 - 128.0, v[2] * 256.0 - 128.0, name=name)
            elif cs in ("gray", "grey") and v:
                col = PaletteColor.gray(v[0], name=name)
            elif cs not in (None, "", "cmyk", "rgb", "lab", "gray", "grey"):  # named colorspace template
                tpl = next((s for s in spaces if s[0] == cs), None)
                if tpl and tpl[1]:
                    e = tpl[1][0]
                    col = PaletteColor(name, e.color_space, list(e.components), e.alpha, e.color_type)
            if col is not None:
                if in_spaces:
                    if spaces:
                        spaces[-1][1].append(col)
                else:
                    if group is None:
                        group = ColorGroup()
                    group.colors.append(col)

    def end(tag, chars):
        nonlocal group, in_spaces
        t = tag.strip().lower()
        if t == "page":
            if group is not None and group.colors:
                if not group.name and not p.colors and not p.groups:
                    p.colors.extend(group.colors)
                elif group.name:
                    p.groups.append(group)
                else:
                    p.colors.extend(group.colors)
            group = None
        elif t == "colorspaces":
            in_spaces = False

    _sax(data, start, end)
    if not p.colors and p.groups:
        p.colors = p.groups.pop(0).colors
    if not p.colors and not p.groups:
        raise _err("Invalid format")
    return p


def _corel_page(name: str, colors: list[PaletteColor]) -> str:
    s = "<page" + (f' name="{_xml_escaped(name)}"' if name else "") + ">"
    for c in colors:
        v = c.components
        if c.color_space == "lab":
            v = [v[0] / 100.0, (v[1] + 128.0) / 256.0, (v[2] + 128.0) / 256.0]
        tints = ",".join(_javafmt(x, 6) for x in v)
        cs = {"cmyk": "CMYK", "rgb": "RGB", "gray": "GRAY", "lab": "LAB"}[c.color_space]
        s += "<color" + (f' name="{_xml_escaped(c.name)}"' if c.name else "") + f' cs="{cs}"'
        s += (f' tints="{tints}"' if tints else "") + "/>\n"
    return s + "</page>\n"


def _enc_corel_draw(p: Palette) -> bytes:
    s = f'<?xml version="1.0"?>\n<palette guid="{uuid.uuid4()}"'
    s += (f' name="{_xml_escaped(p.name)}"' if p.name else "") + ">\n<colors>\n"
    if p.colors:
        s += _corel_page("", p.colors)
    for g in p.groups:
        s += _corel_page(g.name, g.colors)
    return (s + "</colors>\n</palette>\n").encode()


# SCRIBUS_XML: <SCRIBUSCOLORS><COLOR RGB|CMYK NAME/> ----------------------------
def _dec_scribus_xml(data: bytes) -> Palette:
    p = Palette()

    def first(a, *keys):
        return next((a[k] for k in keys if a.get(k) is not None), None)

    def start(tag, a):
        t = tag.lower()
        if t in ("scribuscolors", "scolors"):
            p.name = _xml_decoded(first(a, "Name", "name") or "")
        elif t == "color":
            name = _xml_decoded(first(a, "NAME", "name") or "")
            rgb, cmyk = first(a, "RGB", "rgb"), first(a, "CMYK", "cmyk")
            try:
                if rgb is not None:
                    p.colors.append(_hex_color(rgb, "RGB", name))
                elif cmyk is not None:
                    h = re.sub("[^0-9a-f]", "", cmyk.lower())
                    if len(h) != 8:
                        raise _err("Invalid format")
                    v = int(h, 16)
                    p.colors.append(PaletteColor(name, "cmyk", [((v >> s) & 0xFF) / 255.0 for s in (24, 16, 8, 0)],
                                                 1.0, "normal"))
            except PaletteFormatError:
                pass

    _sax(data, start)
    if not p.colors:
        raise _err("Invalid format")
    return p


def _enc_scribus_xml(p: Palette) -> bytes:
    s = '<?xml version="1.0"?>\n<SCRIBUSCOLORS' + (f' Name="{_xml_escaped(p.name)}"' if p.name else "") + " >\n"
    for c in p.all_colors():
        if c.color_space == "cmyk":
            s += '<COLOR CMYK="#' + "".join(f"{_b255(v):02x}" for v in c.to_cmyk()[:4]) + '"'
        else:
            s += f'<COLOR RGB="{_hexstr(c, "RGB")}"'
        s += (f' NAME="{_xml_escaped(c.name)}"' if c.name else "") + " />\n"
    return (s + "</SCRIBUSCOLORS>\n").encode()


# COREL_PALETTE: Corel .cpl ------------------------------------------------
_CPL_SPOT = {3, 8, 9, 10, 11, 16, 17, 18, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31, 32, 35, 36, 37}


def _cpl_model(r: _Reader, model: int) -> tuple[str | None, list[float] | None]:
    if model in (2, 3, 17, 4):  # CMYK percentages / fractions, CMY fractions
        r.skip(4)
        b = r.read(4)
        if model == 2:
            return "cmyk", [v / 100.0 for v in b]
        return "cmyk", [b[0] / 255.0, b[1] / 255.0, b[2] / 255.0, b[3] / 255.0 if model != 4 else 0.0]
    if model in (5, 21):  # BGR fractions
        r.skip(4)
        b = r.read(3)
        r.skip(1)
        return "rgb", [b[2] / 255.0, b[1] / 255.0, b[0] / 255.0]
    if model == 9:  # grayscale
        r.skip(4)
        k = r.u8()
        r.skip(3)
        return "gray", [(255 - k) / 255.0]
    r.skip(8)
    return None, None


def _dec_corel_palette(data: bytes) -> Palette:
    r = _Reader(data)
    name, spot, ptype = "", False, 0
    version = r.u16()
    if version == 0xDCDC:
        n = r.u8()
        if n > 0:
            name = r.ascii(n)
        count = r.u16(True)
    elif version in (0xCCBC, 0xCCDC):
        count = r.u16(True)
    else:
        headers = [(r.i32(True), r.i32(True)) for _ in range(r.i32(True))]
        if headers:
            r.seek_set(headers[0][1])
            n = r.u8()
            if n > 0:
                name = r.latin1(n) if version == 0xCDDC else r.utf16(n, le=True)
        if len(headers) > 1:
            r.seek_set(headers[1][1])
            ptype = r.u16(True)
        count = 0
        if len(headers) > 2:
            r.seek_set(headers[2][1])
            count = r.u16(True)
            spot = ptype in _CPL_SPOT
            r.seek_set(headers[2][1] + 2)
    long_ = version in (0xCDBC, 0xCDDC, 0xCDDD) and ptype < 38 and ptype not in (5, 16)
    ctype = "spot" if spot else "normal"
    colors = []
    for _ in range(count):
        if long_:
            r.u32(True)
        model = r.u16(True)
        r.skip(2)
        found = [_cpl_model(r, model)]
        if long_:
            found.append(_cpl_model(r, r.u16(True)))
        n = r.u8()
        cname = ""
        if n > 0:
            cname = r.latin1(n) if version in (0xCDDC, 0xDCDC, 0xCCDC) else r.utf16(n, le=True)
        colors += [PaletteColor(cname, s, c, 1.0, ctype) for s, c in found if s is not None]
        if version == 0xCDDD:  # row and column
            r.read(12)
    return Palette(name, colors)


def _enc_corel_palette(p: Palette) -> bytes:
    cols = p.all_colors()
    if not cols:
        raise _err("Too few colors")
    nb = (p.name or "Palette").encode("latin-1", "replace")[:255]
    out = bytearray(struct.pack(">H", 0xDCDC) + bytes([len(nb)]) + nb + struct.pack("<H", len(cols) & 0xFFFF))
    for c in cols:
        r, g, b, _ = c.to_rgba()
        cn = c.name.encode("latin-1", "replace")[:255]
        out += struct.pack("<H", 5) + bytes(6) + bytes((_b255(b), _b255(g), _b255(r), 0, len(cn))) + cn
    return bytes(out)


# CSV ------------------------------------------------------------------------
def _dec_csv(data: bytes) -> Palette:
    records = [[f.strip() for f in line.split(",")] for line in _lines(_text(data)) if line.strip()]
    if not records:
        raise _err("Invalid format")

    def color(s, name=""):
        try:
            return _hex_color(s.strip(), "RGBA", name)
        except PaletteFormatError:
            return None

    if len(records) == 1:
        colors = [color(f) for f in records[0]]
    else:
        colors = [color(r[0]) if len(r) == 1 else color(r[0], r[1].strip()) for r in records]
    colors = [c for c in colors if c is not None]
    if not colors:
        raise _err("Invalid format")
    return Palette(colors=colors)


def _enc_csv(p: Palette) -> bytes:
    return "".join(_hexstr(c, "RGB") + (f", {c.name}" if c.name else "") + "\n" for c in p.all_colors()).encode()


# DCP: ColorPaletteCodable binary ----------------------------------------------
_DCP_SPACES = {1: "cmyk", 2: "rgb", 3: "lab", 4: "gray"}


def _dcp_color(r: _Reader) -> PaletteColor:
    if r.u8() != 0xC0:
        raise _err("Invalid BOM")
    name = r.pascal16(True)
    space = _DCP_SPACES.get(r.u8())
    if space is None:
        raise _err("Invalid format")
    comps = [r.f32(True) for _ in range(_SPACE_SIZE[space])]
    alpha, ctype = r.f32(True), r.u8()
    if ctype not in (1, 2, 3):
        raise _err("Invalid format")
    return PaletteColor(name, space, comps, alpha, _TYPES[ctype - 1])


def _dec_dcp(data: bytes) -> Palette:
    r = _Reader(data)
    if r.u16(True) != 32156 or r.u16(True) != 1:
        raise _err("Invalid BOM")
    name = r.pascal16(True)
    groups = []
    for _ in range(r.u16(True)):
        if r.u8() != 0xEA:
            raise _err("Invalid BOM")
        gname = r.pascal16(True)
        groups.append(ColorGroup(gname, [_dcp_color(r) for _ in range(r.u16(True))]))
    if not groups:  # the first group always holds the global colours
        raise _err("Invalid format")
    return Palette(name, groups[0].colors, groups[1:])


def _enc_dcp(p: Palette) -> bytes:
    out = bytearray(struct.pack("<HH", 32156, 1) + _pascal16(p.name))
    out += struct.pack("<H", len(p.all_groups) & 0xFFFF)
    for g in p.all_groups:
        out += b"\xEA" + _pascal16(g.name) + struct.pack("<H", len(g.colors) & 0xFFFF)
        for c in g.colors:
            out += b"\xC0" + _pascal16(c.name) + bytes([{v: k for k, v in _DCP_SPACES.items()}[c.color_space]])
            out += b"".join(_pf("<", v) for v in c.components) + _pf("<", c.alpha)
            out += bytes([_TYPES.index(c.color_type) + 1])
    return bytes(out)


# GIMP .gpl --------------------------------------------------------------------
_GIMP_NAME = re.compile(r"^Name:\s*(.*)$", re.I | re.A)
_GIMP_COLOR = re.compile(r"^\s*(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})(?:\s+(.*))?$", re.A)


def _dec_gimp(data: bytes) -> Palette:
    lines = _lines(_text(data))
    if lines[0].removeprefix("﻿").strip().lower() != "gimp palette":
        raise _err("Invalid format")
    p = Palette()
    for line in lines[1:]:
        t = line.strip()
        if not t or t.startswith("#") or t[:8].lower() == "columns:":
            continue
        m = _GIMP_NAME.search(t)
        if m:
            p.name = m.group(1).strip()
            continue
        m = _GIMP_COLOR.search(t)
        if m:
            rgb = [_int(m.group(i)) for i in (1, 2, 3)]
            if all(v is not None and 0 <= v <= 255 for v in rgb):
                p.colors.append(PaletteColor.rgb(*(_clamp(v / 255.0) for v in rgb), name=(m.group(4) or "").strip()))
    return p


def _gimp_text(s: str) -> str:
    return re.sub(r"[\r\n\t]+", " ", s).strip()


def _enc_gimp(p: Palette) -> bytes:
    cols = p.all_colors()
    s = f"GIMP Palette\nName: {_gimp_text(p.name)}\nColumns: 0\n#Colors: {len(cols)}\n"
    for c in cols:
        r, g, b, _ = c.converted("rgb").to_rgba()
        s += "\t".join(str(min(max(_jround(v * 255.0), 0), 255)) for v in (r, g, b))
        s += (f"\t{_gimp_text(c.name)}" if _gimp_text(c.name) else "") + "\n"
    return s.encode()


# HEX_RGBA: hex codes, '; name' comment lines -----------------------------------
def _dec_hex_rgba(data: bytes) -> Palette:
    colors, name = [], ""
    for line in _lines(_text(data)):
        t = line.strip()
        if not t:
            continue
        if t[0] == ";":
            name = t[1:].strip() or name
            continue
        for token in re.findall(r"[#0-9a-fA-F]+", line):
            try:
                colors.append(_hex_color(token, "RGBA", name))
                name = ""
            except PaletteFormatError:
                pass
    if not colors:
        raise _err("Invalid format")
    return Palette(colors=colors)


def _enc_hex_rgba(p: Palette) -> bytes:
    s = ""
    for c in _as_space(p.all_colors(), "rgb"):
        r, g, b, a = c.to_rgba()
        s += (f"; {c.name}\n" if c.name else "") + _hexrgb(r, g, b, a, "RGBA" if a < 1.0 else "RGB") + "\n"
    return s.encode()


# IMAGE: one 32x32 swatch per colour, names appended after IEND -------------------
_IMAGE_FORMATS = ["PNG", "JPEG", "WEBP", "GIF", "BMP", "ICO"]  # what Android's BitmapFactory reads
_IEND = b"IEND\xaeB`\x82"


def _dec_image(data: bytes) -> Palette:
    data = bytes(data)
    try:
        with Image.open(io.BytesIO(data), formats=_IMAGE_FORMATS) as im:
            im = im.convert("RGBA")
    except Exception as exc:
        raise _err("Invalid format") from exc
    if im.width // 32 == 0:
        raise _err("Invalid format")
    pixels = [im.getpixel((i * 32 + 16, 0)) for i in range(im.width // 32) if i * 32 + 16 < im.width]
    names: list[str] = []
    end = data.find(_IEND)
    if end >= 0 and end + 8 < len(data):
        ext = _text(data[end + 8:])
        if ext.startswith("\n; IMAGE_NAMES: "):
            line = next((ln for ln in _lines(ext) if ln.startswith("; IMAGE_NAMES:")), None)
            if line is not None:
                names = line[len("; IMAGE_NAMES: "):].strip().split("|")
    return Palette(colors=[PaletteColor.rgb(r / 255.0, g / 255.0, b / 255.0, a / 255.0, names[i] if i < len(names) else "")
                           for i, (r, g, b, a) in enumerate(pixels)])


def _enc_image(p: Palette) -> bytes:
    cols = p.all_colors()
    if not cols:
        raise _err("Too few colors")
    im = Image.new("RGBA", (len(cols) * 32, 32), (0, 0, 0, 0))
    for i, c in enumerate(cols):
        r, g, b, a = c.to_rgba()
        im.paste((_b255(r), _b255(g), _b255(b), _b255(a)), (i * 32, 0, i * 32 + 32, 32))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue() + ("\n; IMAGE_NAMES: " + "|".join(c.name for c in cols) + "\n").encode()


# JSON: kotlinx serialisation of Palette ------------------------------------------
_JSPACE = {"rgb": "RGB", "cmyk": "CMYK", "lab": "LAB", "gray": "Gray"}
_JTYPE = {"global": "Global", "spot": "Spot", "normal": "Normal"}


def _json_color_out(c: PaletteColor) -> dict:
    d: dict = {"name": c.name} if c.name else {}
    if c.color_type != "global":
        d["colorType"] = _JTYPE[c.color_type]
    d["colorSpace"], d["colorComponents"] = _JSPACE[c.color_space], list(c.components)
    if c.alpha != 1.0:
        d["alpha"] = c.alpha
    d["id"] = str(uuid.uuid4())  # random-UUID defaults never equal their default: always encoded
    return d


def _json_color_in(o) -> PaletteColor:
    space = {v: k for k, v in _JSPACE.items()}.get(_jf(o, "colorSpace", str))
    ctype = {v: k for k, v in _JTYPE.items()}.get(_jf(o, "colorType", str, "Global"))
    if space is None or ctype is None:
        raise _err("Unknown enum value")
    return PaletteColor(_jf(o, "name", str, ""), space, _jfloats(_jf(o, "colorComponents", list)),
                        _jf(o, "alpha", float, 1.0), ctype)


def _dec_json(data: bytes) -> Palette:
    o = _json_load(data)
    colors = [_json_color_in(c) for c in _jf(o, "colors", list, [])]
    groups = [ColorGroup(_jf(g, "name", str, ""), [_json_color_in(c) for c in _jf(g, "colors", list, [])])
              for g in _jf(o, "groups", list, [])]
    return Palette(_jf(o, "name", str, ""), colors, groups)


def _enc_json(p: Palette) -> bytes:
    d: dict = {"name": p.name} if p.name else {}
    if p.colors:
        d["colors"] = [_json_color_out(c) for c in p.colors]
    if p.groups:
        d["groups"] = [{"id": str(uuid.uuid4()), **({"name": g.name} if g.name else {}),
                        **({"colors": [_json_color_out(c) for c in g.colors]} if g.colors else {})}
                       for g in p.groups]
    d["id"] = str(uuid.uuid4())
    return _kjson(d).encode()


# OPEN_OFFICE: .soc colour table -----------------------------------------------
_SOC_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<office:color-table xmlns:office="http://openoffice.org/2000/office" '
    'xmlns:style="http://openoffice.org/2000/style" xmlns:text="http://openoffice.org/2000/text" '
    'xmlns:table="http://openoffice.org/2000/table" xmlns:draw="http://openoffice.org/2000/drawing" '
    'xmlns:fo="http://www.w3.org/1999/XSL/Format" xmlns:xlink="http://www.w3.org/1999/xlink" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:meta="http://openoffice.org/2000/meta" '
    'xmlns:number="http://openoffice.org/2000/datastyle" xmlns:svg="http://www.w3.org/2000/svg" '
    'xmlns:chart="http://openoffice.org/2000/chart" xmlns:dr3d="http://openoffice.org/2000/dr3d" '
    'xmlns:math="http://www.w3.org/1998/Math/MathML" xmlns:form="http://openoffice.org/2000/form" '
    'xmlns:script="http://openoffice.org/2000/script">\n')


def _dec_open_office(data: bytes) -> Palette:
    p = Palette()

    def start(tag, a):
        if tag == "draw:color":
            try:
                p.colors.append(_hex_color(a.get("draw:color") or "", "RGB", _xd(a.get("draw:name")) or ""))
            except PaletteFormatError:
                pass

    _sax(data, start)
    if not p.total_color_count:
        raise _err("Invalid format")
    return p


def _enc_open_office(p: Palette) -> bytes:
    s = _SOC_HEAD + "".join(f'<draw:color draw:name="{_xml_escaped(c.name)}" draw:color="{_hexstr(c, "RGB", True, True)}"/>\n'
                            for c in p.all_colors())
    return (s + "</office:color-table>\n").encode()


# PAINT_NET: aarrggbb lines ----------------------------------------------------
def _dec_paint_net(data: bytes) -> Palette:
    data = bytes(data)
    if data[:3] == b"\xef\xbb\xbf":
        data = data[3:]
    p, name = Palette(), ""
    for line in _lines(_text(data)):
        t = line.strip()
        if not t:
            continue
        if t.startswith(";"):
            comment = t[1:].strip()
            if comment[:12].lower() == "palette name" or comment[:8].lower() == "palette:":
                value = comment.split(":", 1)[-1].strip()
                if value.strip():
                    p.name = value
                continue
            skip = ("paint.net", "Colors are written", "Downloaded", "Description", "Colors")
            if comment and not any(k in comment for k in skip):
                name = comment
            continue
        if len(t) != 8:
            raise _err("Invalid format")
        a, r, g, b = (_int(t[i:i + 2], 16) for i in (0, 2, 4, 6))
        if None in (a, r, g, b):
            continue
        p.colors.append(PaletteColor.rgb(r / 255.0, g / 255.0, b / 255.0, a / 255.0, name))
        name = ""
    if not p.total_color_count:
        raise _err("Invalid format")
    return p


def _enc_paint_net(p: Palette) -> bytes:
    s = ("; paint.net Palette File\n; Lines that start with a semicolon are comments\n"
         "; Colors are written as 8-digit hexadecimal numbers: aarrggbb\n")
    if p.name:
        s += f"; Palette Name: {p.name}\r\n"
    for c in _as_space(p.all_colors(), "rgb"):
        s += (f"; {c.name}\r\n" if c.name else "") + _hexstr(c, "ARGB", False, True) + "\r\n"
    return s.encode()


# PAINT_SHOP_PRO: JASC-PAL --------------------------------------------------------
_PSP_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*$", re.A)


def _dec_paint_shop_pro(data: bytes) -> Palette:
    lines = _lines(_text(data))
    if len(lines) < 3 or "JASC-PAL" not in lines[0] or lines[1] != "0100" or _int(lines[2]) is None:
        raise _err("Invalid format")
    colors, name = [], ""
    for line in lines[3:]:
        t = line.strip()
        if not t:
            continue
        if t.startswith(";"):
            name = t[1:].strip() or name
            continue
        m = _PSP_RE.search(line)
        if m:
            rgb = [_int(m.group(i)) for i in (1, 2, 3)]
            if None not in rgb:
                colors.append(PaletteColor.rgb(*(_clamp(v / 255.0) for v in rgb), name=name))
                name = ""
    return Palette(colors=colors)


def _rgb_ints(c: PaletteColor) -> list[int]:  # (x * 255).coerceIn(0.0, 255.0).toInt()
    return [_toint(_clamp(v * 255, 0.0, 255.0)) for v in c.to_rgba()[:3]]


def _enc_paint_shop_pro(p: Palette) -> bytes:
    cols = _as_space(p.all_colors(), "rgb")
    s = f"JASC-PAL\n0100\n{len(cols)}\n"
    for c in cols:
        s += (f"; {c.name}\n" if c.name else "") + " ".join(map(str, _rgb_ints(c))) + "\n"
    return s.encode()


# RGBA / RGB: '#hex name' lines ---------------------------------------------------
_HEX_LINE = re.compile(r"^#?\s*([a-f0-9]{3,8})\s*(.*)\s*", re.I | re.A)


def _dec_hex_lines(data: bytes, fmt: str) -> Palette:
    colors = []
    for line in _lines(_text(data)):
        t = line.strip()
        if not t:
            continue
        m = _HEX_LINE.match(t)
        if m is None:
            if fmt == "RGBA":
                raise _err(f"Invalid RGBA hex string: {t}")
            continue
        try:
            colors.append(_hex_color(m.group(1), fmt, m.group(2).strip()))
        except PaletteFormatError:
            if fmt == "RGBA":
                raise _err(f"Invalid RGBA hex string: {m.group(1)}") from None
    if not colors:
        raise _err("Invalid format")
    return Palette(colors=colors)


def _enc_hex_lines(p: Palette, fmt: str) -> bytes:
    return "\r\n".join(_hexstr(c, fmt) + (f" {c.name}" if c.name else "")
                       for c in _as_space(p.all_colors(), "rgb")).encode()


# RIFF: Microsoft 'PAL ' --------------------------------------------------------
def _dec_riff(data: bytes) -> Palette:
    r = _Reader(data)
    if r.ascii(4) != "RIFF":
        raise _err("Invalid format")
    r.i32(True)
    if r.ascii(4) != "PAL ":
        raise _err("Invalid format")
    chunk = r.ascii(4)
    r.i32(True)
    if chunk != "data":
        raise _err("Invalid format")
    r.i16(True)  # palVersion
    colors = []
    for _ in range(r.i16(True)):
        b = r.read(4)
        colors.append(PaletteColor.rgb(b[0] / 255.0, b[1] / 255.0, b[2] / 255.0))
    try:  # non-standard "name" chunk: UTF-8 names joined by '\n'
        if r.i32() == 0x6E616D65:
            names = r.read(r.i32(True)).decode("utf-8", "replace").rstrip("\x00").split("\n")
            for i, name in enumerate(names):
                if i < len(colors) and name:
                    colors[i].name = name
    except Exception:
        pass
    return Palette(colors=colors)


def _enc_riff(p: Palette) -> bytes:
    cols = p.all_colors()
    data_size = 4 + 4 * len(cols)
    has_names = any(c.name for c in cols)
    names = "\n".join(c.name for c in cols).encode() if has_names else b""
    pad = 1 if has_names and len(names) % 2 else 0
    names_total = 8 + len(names) + pad if has_names else 0
    out = bytearray(b"RIFF" + struct.pack("<I", (12 + data_size + names_total) & 0xFFFFFFFF) + b"PAL data")
    out += struct.pack("<IHH", data_size & 0xFFFFFFFF, 3, len(cols) & 0xFFFF)
    for c in cols:
        r, g, b, _ = c.to_rgba()
        out += bytes((_b255(r), _b255(g), _b255(b), 0))
    if has_names:
        out += b"name" + struct.pack("<I", len(names)) + names + b"\0" * pad
    return bytes(out)


# SKETCH: .sketchpalette JSON ---------------------------------------------------
def _dec_sketch(data: bytes) -> Palette:
    o = _json_load(data)

    def parse(named: bool) -> list[PaletteColor]:
        _jf(o, "compatibleVersion", str), _jf(o, "pluginVersion", str)
        out = []
        for c in _jf(o, "colors", list):
            name = _jf(c, "name", str, None, nullable=True) if named else None
            r, g, b, a = (_jf(c, k, float) for k in ("red", "green", "blue", "alpha"))
            out.append(PaletteColor.rgb(r, g, b, a, name or ""))
        return out

    try:
        return Palette(colors=parse(True))
    except PaletteFormatError:
        return Palette(colors=parse(False))


def _enc_sketch(p: Palette) -> bytes:
    colors = []
    for c in p.all_colors():
        r, g, b, a = c.to_rgba()
        colors.append({**({"name": c.name} if c.name else {}), "red": r, "green": g, "blue": b, "alpha": a})
    return _kjson({"compatibleVersion": "1.4", "pluginVersion": "1.4", "colors": colors}).encode()


# SKP: sK1 palette ----------------------------------------------------------------
_SKP_COLOR = re.compile(r"color\(\['RGB', \[(\d+(?:\.\d+)?), (\d+(?:\.\d+)?), (\d+(?:\.\d+)?)], "
                        r"(\d+(?:\.\d+)?),.*?'(.*?)'", re.A)
_SKP_NAME = re.compile(r"set_name\(.*?'(.*?)'\)", re.A)


def _dec_skp(data: bytes) -> Palette:
    lines = _lines(_text(data))
    if "##sK1 palette" not in lines[0]:
        raise _err("Invalid format")
    p = Palette()
    for line in lines[1:]:
        t = line.strip()
        if not t:
            continue
        m = _SKP_NAME.search(t)
        if m and m.group(1):
            p.name = m.group(1)
        m = _SKP_COLOR.search(t)
        if m:
            v = [_double(m.group(i)) for i in (1, 2, 3, 4)]
            if None not in v:
                p.colors.append(PaletteColor.rgb(*v, name=m.group(5)))
    if not p.all_colors():
        raise _err("Invalid format")
    return p


def _enc_skp(p: Palette) -> bytes:
    def q(s):
        return s.replace("'", "_").replace('"', "_")
    s = f"##sK1 palette\nPalette.Builder()\nset_name('{q(p.name)}')\nset_source('ColorPaletteCodable')\nset_columns(4)\n"
    for c in p.all_colors():
        r, g, b, a = c.to_rgba()
        s += f"color(['RGB', [{_jstr(r)}, {_jstr(g)}, {_jstr(b)}], {_jstr(a)}, '{q(c.name)}'])\n"
    return (s + "palette_end()\n").encode()


# SVG: <rect fill="#RRGGBB" fill-opacity id=name> swatches ---------------------------
_SVG_RGB = re.compile(r"rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)", re.A)


def _dec_svg(data: bytes) -> Palette:
    p = Palette()

    def start(tag, a):
        if tag.strip().lower() != "rect":
            return
        fill, name = a.get("fill"), a.get("id") or ""
        opacity = _double(a.get("fill-opacity"))
        opacity = 1.0 if opacity is None else opacity
        if not fill:
            return
        if fill.startswith("#"):
            rgb = _extract_hex_rgba(fill, "RGB")
            if rgb is not None:
                p.colors.append(PaletteColor.rgb(*(_clamp(v) for v in rgb[:3]), opacity, name))
        elif fill.startswith("rgb"):
            m = _SVG_RGB.search(fill)
            if m:
                r, g, b = (_int(m.group(i)) or 0 for i in (1, 2, 3))
                alpha = _double(m.group(4) or "")
                p.colors.append(PaletteColor.rgb(_clamp(r / 255.0), _clamp(g / 255.0), _clamp(b / 255.0),
                                                 _clamp(opacity if alpha is None else alpha), name))

    _sax(data, start)
    if not p.colors:
        raise _err("Invalid format")
    return p


def _enc_svg(p: Palette) -> bytes:
    size, inset, max_width = 40.0, 4.0, 600.0
    x, y = inset, inset

    def fmt(v):
        return _decfmt(v, 3, 0, 0)  # DecimalFormat("#.###")

    def grouping(colors):
        nonlocal x, y
        s = ""
        for c in colors:
            s += f'      <rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(size)}" height="{fmt(size)}" '
            s += (f'id="{_xml_escaped(c.name)}" ' if c.name else "")
            s += f'fill="{_hexstr(c, "RGB", True, True)}" fill-opacity="{fmt(c.alpha)}" />\n'
            x += size + 1
            if x + size + inset > max_width:
                y += size + 1
                x = inset
        return s

    body = grouping(p.colors)
    for g in p.groups:
        x = inset
        body += grouping(g.colors)
        if g.name:
            y += size + 10
            body += (f"      <text x='5' y='{fmt(y)}' font-size='8' alignment-baseline='middle'>"
                     f"{_xml_escaped(g.name)}</text>\n\n")
        y += 10
    y += inset
    head = ('<?xml version="1.0" encoding="utf-8"?>\n\t<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" '
            f'viewBox="0 0 {fmt(max_width)} {fmt(y + size)}" xml:space="preserve">\n\n')
    return (head + body + "</svg>\n").encode()


# SWIFT / KOTLIN source exports (upstream also implements best-effort decoders) -------
_SWIFT_COLOR = re.compile(
    r"#colorLiteral\s*\(\s*red:\s*([\d.]+)\s*,\s*green:\s*([\d.]+)\s*,\s*blue:\s*([\d.]+)\s*,\s*alpha:\s*([\d.]+)"
    r"\s*\)\s*,?\s*//\s*([^#\n]*?)(?=\s*#colorLiteral|$)", re.I | re.M | re.A)


def _dec_swift(data: bytes) -> Palette:
    text, p = _text(data), Palette()
    for m in _SWIFT_COLOR.finditer(text):
        r, g, b, a = (_double(m.group(i)) for i in (1, 2, 3, 4))
        p.colors.append(PaletteColor.rgb(r or 0.0, g or 0.0, b or 0.0, 1.0 if a is None else a,
                                         m.group(5).strip() or f"Color_{len(p.colors)}"))
    m = re.search(r"(?m)^\s*//\s*Palette:\s*(.+)\s*$", text, re.A)
    if m:
        p.name = m.group(1).strip()
    elif m := re.search(r"struct\s+(\w+)", text, re.A):
        p.name = m.group(1)
    if not p.colors:
        raise _err("Invalid format")
    return p


def _enc_swift(p: Palette) -> bytes:
    def fmt(v):
        return _decfmt(v, 4, 4)  # DecimalFormat("0.0000")
    s = (f"// Palette: {p.name}\n" if p.name else "") + "struct ExportedPalettes {\n"
    for gi, g in enumerate(p.all_groups):
        if not g.colors:
            continue
        s += f"   // Group ({g.name})\n   static let group{gi}: [CGColor] = ["
        for i, c in enumerate(g.colors):
            r, gg, b, a = c.converted("rgb").to_rgba()
            s += ("\n     " if i % 8 == 0 else "")
            s += f" #colorLiteral(red: {fmt(r)}, green: {fmt(gg)}, blue: {fmt(b)}, alpha: {fmt(a)}),"
            s += f" // {c.name}" if c.name else ""
        s += "\n   ]\n\n"
    return (s + "}\n").encode()


_KT_COLOR = re.compile(r"(?:val\s+(\w+)\s*:\s*Color\s*=\s*)?Color\s*\(\s*0x([0-9A-Fa-f]{6,8})\s*\)", re.I | re.A)


def _dec_kotlin(data: bytes) -> Palette:
    text, p, seen = _text(data), Palette(), set()
    for m in _KT_COLOR.finditer(text):
        hx = m.group(2)
        v = int(hx, 16)
        if len(hx) == 8:
            a, r, g, b = (((v >> s) & 0xFF) / 255.0 for s in (24, 16, 8, 0))
        else:
            (r, g, b), a = (((v >> s) & 0xFF) / 255.0 for s in (16, 8, 0)), 1.0
        key = (_toint(r * 255), _toint(g * 255), _toint(b * 255))
        if key not in seen:  # duplicates (by RGB) are skipped
            seen.add(key)
            p.colors.append(PaletteColor.rgb(r, g, b, a, m.group(1) or f"Color_{len(p.colors)}"))
    m = re.search(r"object\s+(\w+)", text, re.A)
    if m:
        p.name = m.group(1)
    elif m := re.search(r"Exported palette:\s*(.+)", text, re.A):
        p.name = m.group(1).strip()
    if not p.colors:
        raise _err("Invalid format")
    return p


def _kt_ident(name: str) -> str:
    return re.sub(r"^[0-9]", r"_\g<0>", re.sub(r"[^a-zA-Z0-9_]", "_", name)) or "color"


def _kt_color(c: PaletteColor) -> str:
    r, g, b, a = c.converted("rgb").to_rgba()
    return f"Color(0x{(_b255(a) << 24) | (_b255(r) << 16) | (_b255(g) << 8) | _b255(b):08X})"


def _enc_kotlin(p: Palette) -> bytes:
    pkg = re.sub("[^a-z0-9]", "", p.name.lower())
    pkg = pkg if pkg and pkg[0].isalpha() else "palette"
    cols = p.all_colors()
    s = (f"package {pkg}\n\nimport androidx.compose.ui.graphics.Color\n\n/**\n"
         f" * Exported palette: {p.name or 'Untitled'}\n * Total colors: {p.total_color_count}\n */\n"
         "object ExportedPalette {\n\n")
    if cols:
        s += "    // Individual color constants\n"
        s += "".join(f"    val {_kt_ident(c.name) if c.name else f'color{i}'}: Color = {_kt_color(c)}\n"
                     for i, c in enumerate(cols)) + "\n"
    for gi, g in enumerate(p.all_groups):
        if g.colors:
            gname = _kt_ident(g.name) if g.name and g.name != "global" else f"group{gi}"
            s += f"    // Group: {g.name}\n    val {gname}: List<Color> = listOf(\n"
            s += ",\n".join(f"        {_kt_color(c)}" for c in g.colors) + "\n    )\n\n"
    if cols:
        s += "    /**\n     * All colors from all groups\n     */\n    val allColors: List<Color> = listOf(\n"
        s += ",\n".join(f"        {_kt_color(c)}" for c in cols) + "\n    )\n"
    return (s + "}\n").encode()


# COREL_DRAW_V3: "name" c m y k (percent) ---------------------------------------
_CDR3_RE = re.compile(r'"(.*)"[ \t]*(\d*\.?\d+)[ \t]*(\d*\.?\d+)[ \t]*(\d*\.?\d+)[ \t]*(\d*\.?\d+)[ \t]*', re.A)


def _dec_corel_draw_v3(data: bytes) -> Palette:
    colors = []
    for line in _lines(_text(data)):
        m = _CDR3_RE.search(line.strip())
        if m:
            v = [_double(m.group(i)) for i in (2, 3, 4, 5)]
            if None not in v:
                colors.append(PaletteColor.cmyk(*(_clamp(x / 100.0) for x in v), name=m.group(1)))
    if not colors:
        raise _err("Invalid format")
    return Palette(colors=colors)


def _enc_corel_draw_v3(p: Palette) -> bytes:
    s = ""
    for c in _as_space(p.all_colors(), "cmyk"):
        v = [_kround(_clamp(x) * 100) for x in c.to_cmyk()[:4]]
        s += f'"{c.name}"    {v[0]}    {v[1]}    {v[2]}    {v[3]}\r\n'
    return s.encode()


# CLF: name<TAB>L<TAB>a<TAB>b ------------------------------------------------------
def _dec_clf(data: bytes) -> Palette:
    colors = []
    for line in _lines(_text(data)):
        parts = line.split("\t")
        if len(parts) == 4:
            v = [_double(x.replace(",", ".").strip()) for x in parts[1:]]
            if None not in v:
                colors.append(PaletteColor.lab(*v, name=parts[0]))
    if not colors:
        raise _err("Invalid format")
    return Palette(colors=colors)


def _enc_clf(p: Palette) -> bytes:
    return "\n".join(f"{c.name or 'Color'}\t" + "\t".join(_jstr(v) for v in c.converted("lab").components)
                     for c in p.all_colors()).encode()


# SWATCHES: Procreate (zip with Swatches.json, HSB) ------------------------------
def _procreate_palette(o) -> tuple[str, list[PaletteColor]]:
    name, colors = _jf(o, "name", str), []
    for s in _jf(o, "swatches", list):
        if s is None:
            continue
        for key, typ in (("colorSpace", int), ("origin", int), ("colorModel", int), ("colorProfile", str),
                         ("version", str), ("components", list)):
            _jf(s, key, typ, None, nullable=True)
        if s.get("components") is not None:
            _jfloats(s["components"])
        h, sat, b = (_jf(s, k, float) for k in ("hue", "saturation", "brightness"))
        colors.append(PaletteColor.hsb(h, sat, b, _jf(s, "alpha", float, 1.0),
                                       _jf(s, "name", str, None, nullable=True) or ""))
    return name, colors


def _dec_swatches(data: bytes) -> Palette:
    raw = _zip_entry(data, lambda n: n == "Swatches.json")
    if not raw:
        raise _err("Invalid format")
    try:
        o = _json_load(raw)
    except ValueError as exc:
        raise _err("Invalid format") from exc
    try:
        if not isinstance(o, list):
            raise _err("Expected a JSON array")
        palettes = [_procreate_palette(x) for x in o]
    except PaletteFormatError:
        try:
            palettes = [_procreate_palette(o)]
        except PaletteFormatError as exc:
            raise _err("Invalid format") from exc
    p = Palette()
    for name, colors in palettes:
        if colors:
            if not p.colors and not p.groups:
                p.colors, p.name = colors, name
            else:
                p.groups.append(ColorGroup(name, colors))
    if not p.total_color_count:
        raise _err("Invalid format")
    return p


def _enc_swatches(p: Palette) -> bytes:
    if not p.total_color_count:
        raise _err("Too few colors")
    out = []
    for g in [ColorGroup(p.name, p.colors), *p.groups]:
        if g.colors:
            swatches = []
            for c in g.colors:
                h, s, b, a = (_clamp(v) for v in c.to_hsb())
                d = {**({"name": c.name} if c.name else {}), "hue": h, "saturation": s, "brightness": b}
                if a != 1.0:
                    d["alpha"] = a
                swatches.append(d)
            out.append({"name": g.name, "swatches": swatches})
    return _zip([("Swatches.json", _kjson(out).encode(), False)])


# INFINITE_PAINTER: {"colors": [signed ARGB ints], "name"} ------------------------
def _dec_infinite_painter(data: bytes) -> Palette:
    o = _json_load(data)
    values = [_jf({"v": v}, "v", int) for v in _jf(o, "colors", list)]
    colors = [PaletteColor.rgb(((v >> 16) & 255) / 255.0, ((v >> 8) & 255) / 255.0, (v & 255) / 255.0,
                               ((v >> 24) & 255) / 255.0) for v in values]
    return Palette(_jf(o, "name", str, ""), colors)


def _enc_infinite_painter(p: Palette) -> bytes:
    values = []
    for c in p.all_colors():
        r, g, b, _ = c.to_rgba()
        v = 0xFF000000 | (_jround(r * 255) << 16) | (_jround(g * 255) << 8) | _jround(b * 255)
        values.append(v - (1 << 32))  # alpha is always written opaque, as a signed Int
    return _kjson({"colors": values, **({"name": p.name} if p.name else {})}, pretty=True).encode()


# AUTODESK_COLOR_BOOK: unencrypted XML colour book --------------------------------
def _dec_autodesk_color_book(data: bytes) -> Palette:
    p = Palette()
    group, cur = None, {"name": None, "r": None, "g": None, "b": None}

    def start(tag, a):
        nonlocal group
        t = tag.strip().lower()
        if t == "colorpage":
            group = ColorGroup(_xd(a.get("name")) or "")
        elif t in ("colorentry", "pagecolor"):
            cur.update(name=None, r=None, g=None, b=None)

    def end(tag, chars):
        nonlocal group
        t, content = tag.strip().lower(), chars.strip()
        if t == "bookname" and content:
            p.name = content
        elif t == "colorname" and content:
            cur["name"] = content
        elif t in ("red", "green", "blue"):
            v = _int(content)
            cur[t[0]] = None if v is None else min(max(v, 0), 255)
        elif t in ("colorentry", "pagecolor"):
            if None not in (cur["r"], cur["g"], cur["b"]):
                if group is None:
                    group = ColorGroup()
                group.colors.append(PaletteColor.rgb(cur["r"] / 255.0, cur["g"] / 255.0, cur["b"] / 255.0,
                                                     name=cur["name"] or ""))
            cur.update(name=None, r=None, g=None, b=None)
        elif t == "colorpage":
            if group is not None and group.colors:
                if not p.colors and not p.groups:
                    p.colors.extend(group.colors)
                    if group.name and not p.name:
                        p.name = group.name
                else:
                    p.groups.append(group if group.name else ColorGroup(f"Color Page {len(p.groups) + 1}", group.colors))
            group = None

    _sax(data, start, end)
    if not p.colors and p.groups:
        p.colors = p.groups.pop(0).colors
    if not p.colors and not p.groups:
        raise _err("Invalid format")
    return p


def _enc_autodesk_color_book(p: Palette) -> bytes:
    s = (f'<?xml version="1.0" encoding="UTF-8"?>\n<colorBook>\n   <bookName>{_xml_escaped(p.name or "Untitled")}'
         "</bookName>\n   <majorVersion>1</majorVersion>\n   <minorVersion>0</minorVersion>\n")
    for g in p.all_groups:
        if not g.colors:
            continue
        s += "   <colorPage>\n"
        for c in g.colors:
            r, gg, b, _ = c.to_rgba()
            s += (f"      <colorEntry>\n         <colorName>{_xml_escaped(c.name or 'Color')}</colorName>\n"
                  f"         <RGB8>\n            <red>{_toint(r * 255)}</red>\n"
                  f"            <green>{_toint(gg * 255)}</green>\n            <blue>{_toint(b * 255)}</blue>\n"
                  "         </RGB8>\n      </colorEntry>\n")
        s += "   </colorPage>\n"
    return (s + "</colorBook>\n").encode()


# SIMPLE_PALETTE: .color-palette JSON (linear sRGB, 4 decimals) --------------------
def _r4(v: float) -> float:  # roundToPlaces(4): kotlin.math.round = ties-to-even
    return v if not math.isfinite(v * 10000.0) else round(v * 10000.0) / 10000.0


def _srgb_to_linear(v: float) -> float:
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(v: float) -> float:
    return v * 12.92 if v <= 0.0031308 else 1.055 * v ** (1.0 / 2.4) - 0.055


def _dec_simple_palette(data: bytes) -> Palette:
    o = _json_load(data)
    name, colors = _jf(o, "name", str, None, nullable=True), []
    for c in _jf(o, "colors", list):
        comps = [_r4(v) for v in _jfloats(_jf(c, "components", list))]
        cname = _jf(c, "name", str, None, nullable=True)
        if 3 <= len(comps) <= 4:
            alpha = _clamp(comps[3]) if len(comps) == 4 else 1.0
            colors.append(PaletteColor.rgb(*(_linear_to_srgb(v) for v in comps[:3]), alpha, cname or ""))
    return Palette(name or "", colors)


def _enc_simple_palette(p: Palette) -> bytes:
    colors = []
    for c in p.all_colors():
        r, g, b, a = c.to_rgba()
        comps = [_r4(_srgb_to_linear(v)) for v in (r, g, b)] + ([_r4(a)] if _r4(a) != 1.0 else [])
        colors.append({**({"name": c.name} if c.name else {}), "components": comps})
    return _kjson({**({"name": p.name} if p.name else {}), "colors": colors}, pretty=True).encode()


# SWATCHBOOKER: .sbz (zip with swatchbook.xml) -------------------------------------
def _dec_swatchbooker(data: bytes) -> Palette:
    raw = _zip_entry(data, lambda n: n.lower() == "swatchbook.xml")
    if not raw:
        raise _err("Invalid format")
    p, stack, materials, order = Palette(), [], {}, []
    title = ident = mode = None
    comps: list[float] = []

    def inside(name):
        return any(n.lower() == name for n in stack)

    def start(tag, a):
        nonlocal title, ident, mode, comps
        t = tag.strip()
        if not t:
            return
        stack.append(t)
        if t.lower() == "color" and inside("materials"):
            title = ident = mode = None
            comps = []
        elif t.lower() == "values":
            mode = next((a[k] for k in ("model", "MODEL", "Model") if a.get(k) is not None), None)
            comps = []
        elif t.lower() == "swatch" and inside("book") and a.get("material") is not None:
            order.append(a["material"].strip())

    def end(tag, chars):
        nonlocal title, ident, comps
        t, content = tag.strip(), chars.strip()
        if not t:
            if stack:
                stack.pop()
            return
        tl = t.lower()
        if tl == "dc:title":
            if inside("materials"):
                title = content
            else:
                p.name = content
        elif tl == "dc:identifier":
            ident = content
        elif tl == "values":
            comps = [v for v in (_double(x) for x in re.split(r"\s+", content, flags=re.A)) if v is not None]
        elif tl == "color" and inside("materials"):
            name = title if title is not None else ident if ident is not None else f"Color {len(p.colors) + 1}"
            m, n, col = (mode or "").lower(), len(comps), None
            if mode is not None:
                if m == "rgb" and n >= 3:
                    col = PaletteColor.rgb(comps[0], comps[1], comps[2], comps[3] if n >= 4 else 1.0, name)
                elif m == "lab" and n >= 3:
                    col = PaletteColor.lab(comps[0], comps[1], comps[2], 1.0, name)
                elif m == "cmyk" and n >= 4:
                    col = PaletteColor.cmyk(*comps[:4], 1.0, name)
                elif m == "gray" and n >= 1:
                    col = PaletteColor.gray(comps[0], 1.0, name)
                elif m == "hsv" and n >= 3:
                    col = PaletteColor.hsb(*comps[:3], 1.0, name)
                elif m == "hsl" and n >= 3:
                    col = PaletteColor.hsl(*comps[:3], 1.0, name)
            if col is not None:
                materials[ident if ident is not None else f"color_{len(p.colors)}"] = col
                p.colors.append(col)
        if stack:
            stack.pop()

    _sax(raw, start, end)
    remaining, ordered = list(p.colors), []
    for material in order:  # book order first, then any colours the book does not reference
        col = materials.get(material)
        idx = next((i for i, c in enumerate(remaining) if c is col), -1)
        if idx >= 0:
            ordered.append(remaining.pop(idx))
    p.colors = ordered + remaining
    if not p.total_color_count:
        raise _err("Invalid format")
    return p


def _enc_swatchbooker(p: Palette) -> bytes:
    cols = p.all_colors()
    x = ('<?xml version="1.0" encoding="UTF-8"?>\n<SwatchBook xmlns="http://www.selapa.net/swatchbook/1.0/" '
         'xmlns:dc="http://purl.org/dc/elements/1.1/">\n  <metadata>\n'
         f"    <dc:title>{_xml_escaped(p.name)}</dc:title>\n  </metadata>\n  <materials>\n")
    for i, c in enumerate(cols):
        values = " ".join(_jstr(v) for v in c.components + ([c.alpha] if c.color_space == "rgb" else []))
        model = {"rgb": "RGB", "cmyk": "CMYK", "lab": "Lab", "gray": "GRAY"}[c.color_space]
        x += ('    <color usage="standard">\n      <metadata>\n'
              f"        <dc:title>{_xml_escaped(c.name)}</dc:title>\n        <dc:identifier>color_{i}</dc:identifier>\n"
              f'      </metadata>\n      <values model="{model}">\n        {values}\n      </values>\n    </color>\n')
    x += "  </materials>\n  <book>\n" + "".join(f'    <swatch material="color_{i}"/>\n' for i in range(len(cols)))
    return _zip([("swatchbook.xml", (x + "  </book>\n</SwatchBook>\n").encode(), False)])


# AFPALETTE: Affinity palette (marker-scanning reader) ------------------------------
def _af_skip_marker(r: _Reader, marker: bytes) -> None:
    try:
        r.seek_to(marker)
        r.skip(len(marker))
    except EOFError:
        pass


def _dec_afpalette(data: bytes) -> Palette:
    r = _Reader(data)
    try:
        has_bom = r.i32(True) in (0x414BFF00, 0x00FF4B41)
    except EOFError:
        has_bom = False
    if not has_bom:  # no BOM: look for an NClP marker anywhere
        r.seek_set(0)
        found = r.find(b"NClP")
        found = found if found >= 0 else r.find(b"PlCN")
        if found < 0:
            raise _err("Invalid BOM")
        r.seek_set(found + 4)
    for marker in (b"NClP", b"PlCN", None):
        if marker is None:
            raise _err("Invalid format")
        try:
            r.seek_to(marker)
            r.skip(4)
            break
        except EOFError:
            pass
    try:
        name_len = r.i32(True)
    except EOFError:
        raise _err("Invalid format") from None
    try:
        name = "" if name_len <= 0 else r.ascii(name_len)
    except EOFError:
        name = ""
    for marker in (b"VlaP", b"PalV", None):
        if marker is None:
            raise _err("Invalid format")
        try:
            r.seek_to(marker)
            r.skip(4)
            break
        except EOFError:
            pass
    try:
        count = r.i32(True)
    except EOFError:
        raise _err("Invalid format") from None
    colors: list[PaletteColor] = []
    unsupported = False
    for _ in range(count):
        try:
            try:
                r.seek_to(b"rloC")
            except EOFError:
                if not colors:
                    raise _err("Invalid format") from None
                break
            r.skip(4)
            r.try_skip(6)
            try:
                kind = r.ascii(4)
            except EOFError:
                if not colors:
                    raise _err("Invalid format") from None
                break
            if kind == "ABGR":
                _af_skip_marker(r, b"Dloc_")
                colors.append(PaletteColor.rgb(r.f32(True), r.f32(True), r.f32(True)))
            elif kind == "ABAL":
                _af_skip_marker(r, b"<loc_")
                l, a, b = (r.u16(True) for _ in range(3))
                colors.append(PaletteColor.lab(l / 65535.0 * 100.0, a / 65535.0 * 256.0 - 128.0,
                                               b / 65535.0 * 256.0 - 128.0).converted("rgb"))
            elif kind == "KYMC":
                _af_skip_marker(r, b"Hloc_")
                colors.append(PaletteColor.cmyk(r.f32(True), r.f32(True), r.f32(True), r.f32(True)))
            elif kind == "ALSH":
                _af_skip_marker(r, b"Dloc_")
                colors.append(PaletteColor.hsl(r.f32(True), r.f32(True), r.f32(True)))
            elif kind == "YARG":
                _af_skip_marker(r, b"<loc_")
                colors.append(PaletteColor.gray(r.f32(True)))
            else:
                unsupported = True
                raise _err("Cannot create color")
        except PaletteFormatError:
            if unsupported:
                raise _err("Unsupported palette type") from None
            if not colors:
                raise _err("Invalid format") from None
            break
        except Exception:
            if not colors:
                raise _err("Invalid format") from None
            break
    try:  # names section (VNaP) is optional
        idx = r.find(b"VNaP")
        idx = idx if idx >= 0 else r.find(b"PaNV")
        if idx >= 0:
            r.seek_set(idx + 4)
            r.u32(True)
            for i in range(min(r.i32(True), len(colors))):
                try:
                    colors[i].name = r.utf8(r.i32(True))
                except Exception:
                    break
    except Exception:
        pass
    if not colors:
        raise _err("Invalid format")
    return Palette(name, colors)


def _enc_afpalette(p: Palette) -> bytes:
    cols = p.all_colors()
    if not cols:
        raise _err("Too few colors")
    name = (p.name or "Palette").encode("ascii", "replace")
    out = bytearray(struct.pack("<II", 0x414BFF00, 11) + b"PlCN" + struct.pack("<I", len(name)) + name)
    out += b"PalV" + struct.pack("<I", len(cols))
    for c in cols:
        r, g, b, _ = c.to_rgba()
        out += b"rloC" + bytes(6) + b"ABGR" + b"Dloc_" + _pf("<", r) + _pf("<", g) + _pf("<", b)
    out += b"PaNV" + struct.pack("<II", 0, len(cols))
    for c in cols:
        nb = (c.name or "Color").encode()
        out += struct.pack("<I", len(nb)) + nb
    return bytes(out)


# XARA: .jcw ---------------------------------------------------------------------
_JCW_SPACES = {1: ("cmyk", "normal"), 8: ("cmyk", "normal"), 9: ("cmyk", "spot"), 2: ("rgb", "normal"),
               10: ("rgb", "normal"), 11: ("rgb", "spot"), 3: ("hsb", "normal"), 12: ("hsb", "normal"),
               13: ("hsb", "spot")}


def _dec_xara(data: bytes) -> Palette:
    r = _Reader(data)
    if r.ascii(3) != "JCW":
        raise _err("Invalid BOM")
    r.u8()  # version
    count, space_id, name_len = r.u16(True), r.u8(), r.u8()
    if space_id not in _JCW_SPACES:
        raise _err("Invalid format")
    space, ctype = _JCW_SPACES[space_id]
    colors = []
    for i in range(count):
        c0, c1, c2, c3 = (min(r.u16(True), 10000) / 10000.0 for _ in range(4))
        name = r.latin1(name_len).rstrip("\x00") or f"c{i}"
        if space == "rgb":
            colors.append(PaletteColor.rgb(c0, c1, c2, name=name, color_type=ctype))
        elif space == "cmyk":
            colors.append(PaletteColor.cmyk(c0, c1, c2, c3, name=name, color_type=ctype))
        else:
            colors.append(PaletteColor.hsb(c0, c1, c2, name=name, color_type=ctype))
    return Palette(colors=colors)


def _enc_xara(p: Palette) -> bytes:
    cols = _as_space(p.all_colors(), "rgb")
    out = bytearray(b"JCW\x01" + struct.pack("<H", len(cols) & 0xFFFF) + bytes([10, 14]))
    for c in cols:
        r, g, b, _ = c.to_rgba()
        out += struct.pack("<4H", *(_toint(v * 10000) & 0xFFFF for v in (r, g, b)), 0)
        out += c.name.encode("latin-1", "replace")[:14].ljust(14, b"\0")
    return bytes(out)


# KOFFICE / HPL / SKENCIL: small text formats --------------------------------------
_KOFFICE_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)(.*)$", re.A)
_HPL_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+).*$", re.A)
_SKENCIL_RE = re.compile(r"^\s*(\d*\.?\d+|\d+)\s+(\d*\.?\d+|\d+)\s+(\d*\.?\d+|\d+)\s+(.*)$", re.A)


def _dec_koffice(data: bytes) -> Palette:
    lines = _lines(_text(data))
    if "KDE RGB Palette" not in lines[0]:
        raise _err("Invalid format")
    colors = []
    for line in lines[1:]:
        m = _KOFFICE_RE.search(line)
        if m:
            rgb = [_int(m.group(i)) for i in (1, 2, 3)]
            if None not in rgb:
                colors.append(PaletteColor.rgb(*(_clamp(v / 255.0) for v in rgb), name=m.group(4).strip()))
    return Palette(colors=colors)


def _enc_koffice(p: Palette) -> bytes:
    return ("KDE RGB Palette\n" + "".join("\t".join(map(str, _rgb_ints(c))) + (f"\t{c.name}" if c.name else "") + "\n"
                                          for c in _as_space(p.all_colors(), "rgb"))).encode()


def _dec_hpl(data: bytes) -> Palette:
    lines = _lines(_text(data))
    if "Palette" not in lines[0] or len(lines) < 2 or "Version 4.0" not in lines[1]:
        raise _err("Invalid format")
    colors, name = [], ""
    for line in lines[3:]:
        t = line.strip()
        if not t:
            continue
        if t.startswith(";"):
            name = t[1:].strip() or name
            continue
        m = _HPL_RE.search(line)
        if m:
            rgb = [_int(m.group(i)) for i in (1, 2, 3)]
            if None not in rgb:
                colors.append(PaletteColor.rgb(*(_clamp(v / 255.0) for v in rgb), name=name))
                name = ""
    return Palette(colors=colors)


def _enc_hpl(p: Palette) -> bytes:
    s = "Palette\nVersion 4.0\n-----------\n"
    for c in _as_space(p.all_colors(), "rgb"):
        s += (f"; {c.name}\n" if c.name else "") + " ".join(map(str, _rgb_ints(c))) + "\n"
    return s.encode()


def _dec_skencil(data: bytes) -> Palette:
    lines = _lines(_text(data))
    if "##Sketch RGBPalette 0" not in lines[0]:
        raise _err("Invalid format")
    colors = []
    for line in lines[1:]:
        m = _SKENCIL_RE.search(line)
        if m:
            v = [_double(m.group(i)) for i in (1, 2, 3)]
            if None not in v:
                colors.append(PaletteColor.rgb(*v, name=m.group(4).strip()))
    return Palette(colors=colors)


def _enc_skencil(p: Palette) -> bytes:
    s = "##Sketch RGBPalette 0\n"
    for c in _as_space(p.all_colors(), "rgb"):
        s += " ".join(_decfmt(v, 6, 6) for v in c.to_rgba()[:3]) + (f" {c.name}" if c.name else "") + "\n"
    return s.encode()


# VGA_24BIT / VGA_18BIT: raw RGB triplets, optional '; Names: ' header ------------
def _dec_vga(data: bytes, bits: int) -> Palette:
    data, names = bytes(data), []
    if data.startswith(b"; Names: "):
        # DEVIATION: split at the newline byte; upstream uses the decoded char index as a byte offset.
        newline = data.find(b"\n")
        if newline > 0:
            names = [s.strip() for s in _text(data[9:newline]).split(", ")]
            data = data[newline + 1:]
    if not data or len(data) % 3:
        raise _err("Invalid format")
    if bits == 18 and max(data) > 63:
        raise _err("Invalid format")
    scale = 255.0 if bits == 24 else 63.0
    return Palette(colors=[PaletteColor.rgb(data[i] / scale, data[i + 1] / scale, data[i + 2] / scale,
                                            name=names[i // 3] if i // 3 < len(names) else "")
                           for i in range(0, len(data), 3)])


def _enc_vga(p: Palette, bits: int) -> bytes:
    cols = _as_space(p.all_colors(), "rgb")
    names = [c.name for c in cols if c.name]
    out = bytearray(f"; Names: {', '.join(names)}\n".encode() if names else b"")
    for c in cols:
        if bits == 18:
            out += bytes(min(max(_kround(v * 63), 0), 63) for v in c.to_rgba()[:3])
        else:
            out += bytes(_rgb_ints(c))
    return bytes(out)


# KRITA: .kpl (zip: mimetype, colorset.xml, profiles.xml) ----------------------------
def _dec_krita(data: bytes) -> Palette:
    xml = _zip_entry(data, lambda n: n.lower() == "colorset.xml" or n.lower().endswith("/colorset.xml"))
    if xml is None:
        raise _err("Invalid format")
    p, glob, group = Palette(), [], None
    entry: dict = {}

    def reset():
        entry.update(name="", spot=False, color=None, row=_INT_MAX, col=_INT_MAX)

    def ordered(items):
        return [c for _, _, c in sorted(items, key=lambda t: (t[0], t[1]))]

    def num(a, *keys):
        return _double(next((a[k] for k in keys if a.get(k) is not None), None))

    def start(tag, a):
        nonlocal group
        t = tag.split(":", 1)[-1].strip().lower()
        ctype = "spot" if entry.get("spot") else "normal"
        if t == "colorset":
            if _xd(a.get("name")):
                p.name = _xd(a["name"])
        elif t == "group":
            group = (_xd(a.get("name")) or "", [])
        elif t == "colorsetentry":
            reset()
            entry.update(name=_xd(a.get("name")) or "", spot=(a.get("spot") or "").lower() == "true")
        elif t == "position":
            row, col = _int(a.get("row")), _int(a.get("column"))
            entry.update(row=_INT_MAX if row is None else row, col=_INT_MAX if col is None else col)
        elif t in ("srgb", "rgb"):
            v = [num(a, k) for k in "rgb"]
            entry["color"] = None if None in v else PaletteColor.rgb(*v, name=entry["name"], color_type=ctype)
        elif t == "cmyk":
            v = [num(a, k) for k in "cmyk"]
            entry["color"] = None if None in v else PaletteColor.cmyk(*v, name=entry["name"], color_type=ctype)
        elif t == "lab":
            v = [num(a, "L", "l"), num(a, "a"), num(a, "b")]
            entry["color"] = None if None in v else PaletteColor.lab(*v, name=entry["name"], color_type=ctype)
        elif t == "gray":
            v = num(a, "g", "gray")
            entry["color"] = None if v is None else PaletteColor.gray(v, name=entry["name"], color_type=ctype)

    def end(tag, chars):
        nonlocal group
        t = tag.split(":", 1)[-1].strip().lower()
        if t == "colorsetentry":
            if entry["color"] is not None:
                (group[1] if group is not None else glob).append((entry["row"], entry["col"], entry["color"]))
            reset()
        elif t == "group":
            if group is not None and group[1]:
                p.groups.append(ColorGroup(group[0], ordered(group[1])))
            group = None

    reset()
    _sax(xml, start, end)
    p.colors = ordered(glob)
    if not p.colors and not p.groups:
        raise _err("Invalid format")
    return p


def _enc_krita(p: Palette) -> bytes:
    if not p.total_color_count:
        raise _err("Too few colors")
    columns = max(math.ceil(math.sqrt(max(len(p.colors), max((len(g.colors) for g in p.groups), default=0), 1))), 1)

    def rows(n):
        return 0 if n <= 0 else math.ceil(n / columns)

    def clean(s):
        return _xml_escaped(re.sub(r"[\r\n\t]+", " ", s).strip())

    def entries(colors, indent, prefix):
        s = ""
        for i, c in enumerate(colors):
            cid = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{prefix}_{i + 1}").strip("_") or "color"
            r, g, b = (_decfmt(_clamp(v), 6) for v in c.to_rgba()[:3])  # DecimalFormat("0.######")
            spot = "true" if c.color_type == "spot" else "false"
            s += (f'{indent}<ColorSetEntry name="{clean(c.name)}" id="{cid}" bitdepth="F32" spot="{spot}">\n'
                  f'{indent}  <sRGB r="{r}" g="{g}" b="{b}"/>\n'
                  f'{indent}  <Position row="{i // columns}" column="{i % columns}"/>\n{indent}</ColorSetEntry>\n')
        return s

    x = ('<?xml version="1.0" encoding="UTF-8"?>\n'
         f'<Colorset name="{clean(p.name)}" comment="" columns="{columns}" rows="{rows(len(p.colors))}" '
         'readonly="false" version="1.0">\n' + entries(p.colors, "  ", "global"))
    for i, g in enumerate(p.groups):
        x += (f'  <Group name="{clean(g.name)}" rows="{rows(len(g.colors))}">\n'
              + entries(g.colors, "    ", f"group_{i + 1}") + "  </Group>\n")
    x += "</Colorset>\n"
    profiles = '<?xml version="1.0" encoding="UTF-8"?>\n<Profiles/>\n'
    return _zip([("mimetype", b"application/x-krita-palette", True), ("colorset.xml", x.encode(), False),
                 ("profiles.xml", profiles.encode(), False)])


# ---------------------------------------------------------------------------
# Format registry (PaletteFormat.kt) and public API
# ---------------------------------------------------------------------------

_FORMAT_TABLE = [  # (id, upstream enum name, extensions, withPaletteName, description, encoder, decoder)
    ("acb", "ACB", ["acb"], True, "Adobe Color Book", _enc_acb, _dec_acb),
    ("aco", "ACO", ["aco"], True, "Adobe Photoshop Swatch", _enc_aco, _dec_aco),
    ("act", "ACT", ["act"], False, "Adobe Color Tables", _enc_act, _dec_act),
    ("android_xml", "ANDROID_XML", ["xml"], False, "Android XML Palette file", _enc_android_xml, _dec_android_xml),
    ("ase", "ASE", ["ase"], True, "Adobe Swatch Exchange", _enc_ase, _dec_ase),
    ("basic_xml", "BASIC_XML", ["xml"], True, "Basic XML palette format", _enc_basic_xml, _dec_basic_xml),
    ("corel_painter", "COREL_PAINTER", ["txt"], False, "Corel Painter Swatches", _enc_corel_painter, _dec_corel_painter),
    ("corel_draw", "COREL_DRAW", ["xml"], True, "CorelDraw XML", _enc_corel_draw, _dec_corel_draw),
    ("scribus_xml", "SCRIBUS_XML", ["xml"], True, "Scribus XML swatches", _enc_scribus_xml, _dec_scribus_xml),
    ("corel_palette", "COREL_PALETTE", ["cpl"], True, "Corel Palette", _enc_corel_palette, _dec_corel_palette),
    ("csv", "CSV", ["csv"], False, "CSV Palette", _enc_csv, _dec_csv),
    ("dcp", "DCP", ["dcp"], True, "ColorPaletteCodable binary format", _enc_dcp, _dec_dcp),
    ("gimp", "GIMP", ["gpl"], True, "GIMP gpl format", _enc_gimp, _dec_gimp),
    ("hex_rgba", "HEX_RGBA", ["hex"], False, "Hex RGBA coded files", _enc_hex_rgba, _dec_hex_rgba),
    ("image", "IMAGE", ["png", "jpg", "jpeg"], False, "image-based palette coder", _enc_image, _dec_image),
    ("json", "JSON", ["jsoncolorpalette", "json"], True, "ColorPaletteCodable JSON format", _enc_json, _dec_json),
    ("open_office", "OPEN_OFFICE", ["soc"], False, "OpenOffice palette format (.soc)", _enc_open_office,
     _dec_open_office),
    ("paint_net", "PAINT_NET", ["txt"], True, "Paint.NET palette file (.txt)", _enc_paint_net, _dec_paint_net),
    ("paint_shop_pro", "PAINT_SHOP_PRO", ["psppalette", "pal"], True, "Paint Shop Pro palette (.pal, .psppalette)",
     _enc_paint_shop_pro, _dec_paint_shop_pro),
    ("rgba", "RGBA", ["rgba", "txt"], False, "RGBA encoded text files (.rgba, .txt)",
     lambda p: _enc_hex_lines(p, "RGBA"), lambda d: _dec_hex_lines(d, "RGBA")),
    ("rgb", "RGB", ["rgb", "txt"], False, "RGB encoded text files (.rgb, .txt)",
     lambda p: _enc_hex_lines(p, "RGB"), lambda d: _dec_hex_lines(d, "RGB")),
    ("riff", "RIFF", ["pal"], False, "Microsoft RIFF palette (.pal)", _enc_riff, _dec_riff),
    ("sketch", "SKETCH", ["sketchpalette"], False, "Sketch palette file (.sketchpalette)", _enc_sketch, _dec_sketch),
    ("skp", "SKP", ["skp"], False, "SKP Palette", _enc_skp, _dec_skp),
    ("svg", "SVG", ["svg"], True, "Scalable Vector Graphics palette (.svg)", _enc_svg, _dec_svg),
    ("swift", "SWIFT", ["swift"], True, "(export only) Swift source file (.swift)", _enc_swift, _dec_swift),
    ("kotlin", "KOTLIN", ["kt"], True, "(export only) Kotlin/Jetpack Compose source file (.kt)", _enc_kotlin,
     _dec_kotlin),
    ("corel_draw_v3", "COREL_DRAW_V3", ["pal"], True, "Corel Draw V3 file (.pal)", _enc_corel_draw_v3,
     _dec_corel_draw_v3),
    ("clf", "CLF", ["clf"], True, "LAB colors", _enc_clf, _dec_clf),
    ("swatches", "SWATCHES", ["swatches"], True, "Procreate swatches", _enc_swatches, _dec_swatches),
    ("infinite_painter", "INFINITE_PAINTER", ["clrs"], True, "Infinite Painter swatch palette",
     _enc_infinite_painter, _dec_infinite_painter),
    ("autodesk_color_book", "AUTODESK_COLOR_BOOK", ["acb"], True, "Autodesk Color Book (unencrypted only) (.acb)",
     _enc_autodesk_color_book, _dec_autodesk_color_book),
    ("simple_palette", "SIMPLE_PALETTE", ["color-palette"], True, "Simple Palette format", _enc_simple_palette,
     _dec_simple_palette),
    ("swatchbooker", "SWATCHBOOKER", ["sbz"], True, "Swatchbooker .sbz file", _enc_swatchbooker, _dec_swatchbooker),
    ("afpalette", "AFPALETTE", ["afpalette"], True, "Affinity Designer .afpalette file", _enc_afpalette,
     _dec_afpalette),
    ("xara", "XARA", ["jcw"], True, "Xara palette file (.jcw)", _enc_xara, _dec_xara),
    ("koffice", "KOFFICE", ["colors"], False, "KOffice palette file (.colors)", _enc_koffice, _dec_koffice),
    ("hpl", "HPL", ["hpl"], True, "Homesite Palette file (.hpl)", _enc_hpl, _dec_hpl),
    ("skencil", "SKENCIL", ["spl"], True, "Skencil Palette file (.spl)", _enc_skencil, _dec_skencil),
    ("vga_24bit", "VGA_24BIT", ["vga24"], False, "24-bit RGB VGA (3 bytes RRGGBB)",
     lambda p: _enc_vga(p, 24), lambda d: _dec_vga(d, 24)),
    ("vga_18bit", "VGA_18BIT", ["vga18"], False, "18-bit RGB VGA (3 bytes RRGGBB)",
     lambda p: _enc_vga(p, 18), lambda d: _dec_vga(d, 18)),
    ("krita", "KRITA", ["kpl"], True, "KRITA Palette file (.kpl)", _enc_krita, _dec_krita),
]

# Every upstream coder implements both directions (SWIFT/KOTLIN are "export only" in upstream's
# comments, but their decoders are implemented and the app probes them on import).
FORMATS: dict[str, dict] = {
    fid: {"name": name, "extensions": list(ext), "with_palette_name": with_name, "can_encode": True,
          "can_decode": True, "description": desc}
    for fid, name, ext, with_name, desc, _, _ in _FORMAT_TABLE
}
_CODERS = {fid: (enc, dec) for fid, _, _, _, _, enc, dec in _FORMAT_TABLE}

# PaletteFormatHelper.entries: enum order with CSV and HEX_RGBA moved to the end (HEX_RGBA, CSV).
EXPORT_ORDER: list[str] = [f for f in FORMATS if f not in ("csv", "hex_rgba")] + ["hex_rgba", "csv"]


def formats_for_filename(filename: str) -> list[str]:
    """PaletteFormat.matchingFilename: every format whose extension ends the name (?query/#fragment ignored)."""
    norm = filename.split("?", 1)[0].split("#", 1)[0].lower()
    return [fid for fid, f in FORMATS.items() if any(norm.endswith("." + e) for e in f["extensions"])]


def decode_order_for_filename(filename: str) -> list[str]:
    """Import probing order of the app (PaletteFormatHelper.entriesFor): matching formats, then EXPORT_ORDER."""
    return list(dict.fromkeys(formats_for_filename(filename) + EXPORT_ORDER))


def _format_id(fmt: str) -> str:
    fid = str(fmt).lower()
    if fid not in FORMATS:
        raise PaletteFormatError(f"Unknown palette format: {fmt!r}")
    return fid


def encode(palette: Palette, fmt: str) -> bytes:
    """Encode ``palette`` with the coder for ``fmt`` (format id or upstream enum name)."""
    fid = _format_id(fmt)
    try:
        return _CODERS[fid][0](palette)
    except Exception as exc:
        raise PaletteFormatError(f"Cannot encode {FORMATS[fid]['name']} palette: {exc}") from exc


def decode(data: bytes, fmt: str | None = None, filename: str | None = None) -> tuple[Palette, str]:
    """Decode palette bytes; returns ``(palette, format_id)``.

    With ``fmt`` that coder is used as is. Otherwise formats are probed in
    ``decode_order_for_filename(filename or "")`` order and the first one that decodes to a palette
    with at least one colour wins (PaletteToolsComponent's import loop).
    """
    data = bytes(data)
    if fmt is not None:
        fid = _format_id(fmt)
        try:
            return _CODERS[fid][1](data), fid
        except Exception as exc:
            raise PaletteFormatError(f"Cannot decode {FORMATS[fid]['name']} palette: {exc}") from exc
    for fid in decode_order_for_filename(filename or ""):
        try:
            palette = _CODERS[fid][1](data)
        except Exception:
            continue
        if palette.total_color_count > 0:
            return palette, fid
    raise PaletteFormatError(f"Unrecognised palette file{f' {filename!r}' if filename else ''}: "
                             f"none of the {len(FORMATS)} palette formats could read it")
