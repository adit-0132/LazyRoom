"""Safe image decoding / encoding with colour management and metadata control.

* Decoding goes through Pillow (+ pillow-heif) with a pixel-count limit; 16-bit
  PNG/TIFF use OpenCV to keep full precision; camera RAW uses rawpy (LibRaw)
  when installed.
* Embedded ICC profiles are converted to sRGB with LittleCMS (Pillow ImageCms),
  so the pipeline always works in sRGB.
* EXIF orientation is applied on load; exported files get orientation=1.
* Encoding writes to a temporary file in the destination folder and renames it
  atomically.  Existing files are never overwritten unless explicitly allowed.
"""
from __future__ import annotations

import hashlib
import io
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

from .errors import ErrorCode, LazyRoomError
from .image import normalize_channels, opaque_if_full, to_uint8, to_uint16

try:  # optional HEIF/AVIF support
    import pillow_heif

    pillow_heif.register_heif_opener()
    HAVE_HEIF = True
except Exception:  # pragma: no cover
    HAVE_HEIF = False

try:  # optional RAW support
    import rawpy  # type: ignore

    HAVE_RAW = True
except Exception:  # pragma: no cover
    HAVE_RAW = False

try:  # optional JPEG XL support (imagecodecs, BSD-3; bundles libjxl)
    import imagecodecs as _imagecodecs  # type: ignore

    HAVE_JXL = bool(_imagecodecs.JPEGXL.available)
except Exception:  # pragma: no cover
    HAVE_JXL = False
JXL_HINT = "pip install 'lazyroom[jxl]' (installs imagecodecs, which bundles libjxl)"

DEFAULT_MAX_PIXELS = 250_000_000
Image.MAX_IMAGE_PIXELS = DEFAULT_MAX_PIXELS

RAW_EXTS = {".dng", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".orf", ".rw2", ".raf",
            ".pef", ".srw", ".x3f", ".3fr", ".erf", ".kdc", ".mos", ".mrw", ".iiq", ".rwl"}

# extension -> (Pillow format, supports alpha, supports 16-bit, lossy)
WRITE_FORMATS: dict[str, tuple[str, bool, bool, bool]] = {
    ".jpg": ("JPEG", False, False, True),
    ".jpeg": ("JPEG", False, False, True),
    ".png": ("PNG", True, True, False),
    ".webp": ("WEBP", True, False, True),
    ".avif": ("AVIF", True, False, True),
    ".heic": ("HEIF", True, False, True),
    ".heif": ("HEIF", True, False, True),
    ".tif": ("TIFF", True, True, False),
    ".tiff": ("TIFF", True, True, False),
    ".bmp": ("BMP", False, False, False),
    ".gif": ("GIF", True, False, False),
    ".jp2": ("JPEG2000", True, False, True),
    ".ico": ("ICO", True, False, False),
    ".tga": ("TGA", True, False, False),
    ".qoi": ("QOI", True, False, False),
    ".ppm": ("PPM", False, False, False),
    ".pdf": ("PDF", False, False, True),
}
if HAVE_JXL:
    WRITE_FORMATS[".jxl"] = ("JXL", True, True, True)

READ_EXTS = sorted(set(WRITE_FORMATS) | RAW_EXTS | {".psd", ".pcx", ".sgi", ".dds", ".pgm", ".pbm", ".xbm", ".icns", ".jfif"})

_SRGB_PROFILE = ImageCms.createProfile("sRGB")
_SRGB_ICC_BYTES = ImageCms.ImageCmsProfile(_SRGB_PROFILE).tobytes()


@dataclass
class ImageMeta:
    path: str | None = None
    format: str | None = None
    width: int = 0
    height: int = 0
    mode: str | None = None
    bit_depth: int = 8
    has_alpha: bool = False
    frames: int = 1
    animated: bool = False
    dpi: tuple[float, float] | None = None
    icc_profile_name: str | None = None
    color_converted_from: str | None = None
    exif: bytes | None = None
    xmp: bytes | None = None
    orientation_applied: int | None = None
    file_size: int | None = None
    sha256: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("exif", "xmp", "extra")}
        d["has_exif"] = bool(self.exif)
        d["has_xmp"] = bool(self.xmp)
        if self.extra:
            d.update(self.extra)
        return d


def file_sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _check_exists(path: Path) -> None:
    if not path.exists():
        raise LazyRoomError(ErrorCode.FILE_NOT_FOUND, f"File not found: {path}",
                              hint="Check the path; relative paths are resolved against the working directory.")
    if not path.is_file():
        raise LazyRoomError(ErrorCode.FILE_NOT_FOUND, f"Not a regular file: {path}")


def _icc_to_srgb(pil: Image.Image, meta: ImageMeta) -> Image.Image:
    icc = pil.info.get("icc_profile")
    if not icc:
        return pil
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        name = ImageCms.getProfileDescription(src).strip()
        meta.icc_profile_name = name
        if "srgb" in name.lower() or "iec61966" in name.lower():
            return pil
        mode = pil.mode
        if mode not in ("RGB", "RGBA", "CMYK", "L", "LA"):
            return pil
        out_mode = "RGBA" if mode in ("RGBA", "LA") else "RGB"
        if mode == "LA":
            pil = pil.convert("RGBA")
            mode = "RGBA"
        converted = ImageCms.profileToProfile(pil, src, _SRGB_PROFILE, outputMode=out_mode,
                                              renderingIntent=ImageCms.Intent.PERCEPTUAL)
        meta.color_converted_from = name
        return converted if converted is not None else pil
    except Exception as exc:  # malformed profile: keep pixels, note it
        meta.extra["icc_warning"] = f"ICC profile ignored: {exc}"
        return pil


def probe(path: str | os.PathLike) -> ImageMeta:
    """Read header-level metadata without decoding all pixels."""
    p = Path(path)
    _check_exists(p)
    meta = ImageMeta(path=str(p), file_size=p.stat().st_size)
    if _is_jxl(p):
        return _load_jxl(p, 0, DEFAULT_MAX_PIXELS)[1]
    if p.suffix.lower() in RAW_EXTS:
        meta.format = "RAW"
        if HAVE_RAW:
            try:
                with rawpy.imread(str(p)) as raw:
                    s = raw.sizes
                    meta.width, meta.height = s.width, s.height
                    meta.bit_depth = 16
            except Exception as exc:
                raise LazyRoomError(ErrorCode.DECODE_FAILED, f"Cannot read RAW file {p.name}: {exc}") from exc
        return meta
    try:
        with Image.open(p) as im:
            meta.format = im.format
            meta.width, meta.height = im.size
            meta.mode = im.mode
            meta.frames = getattr(im, "n_frames", 1)
            meta.animated = bool(getattr(im, "is_animated", False))
            meta.has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
            meta.bit_depth = 16 if im.mode in ("I;16", "I;16B", "I;16L", "I", "RGB;16") else 8
            dpi = im.info.get("dpi")
            if dpi:
                meta.dpi = (float(dpi[0]), float(dpi[1]))
            meta.exif = im.info.get("exif")
            orient = im.getexif().get(0x0112)
            if orient in (5, 6, 7, 8):
                meta.width, meta.height = meta.height, meta.width
    except Image.DecompressionBombError as exc:
        raise LazyRoomError(ErrorCode.RESOURCE_LIMIT, str(exc),
                              hint="Raise the max-pixels limit in the configuration if the file is trusted.") from exc
    except Exception as exc:
        raise LazyRoomError(ErrorCode.UNSUPPORTED_FORMAT if "cannot identify" in str(exc) else ErrorCode.DECODE_FAILED,
                              f"Cannot read image header of {p.name}: {exc}") from exc
    return meta


def _png_is_16bit(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(33)
        return head[:8] == b"\x89PNG\r\n\x1a\n" and head[24] == 16
    except Exception:
        return False


def _read_cv16(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)  # unicode-safe on Windows
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None or img.dtype != np.uint16:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA if img.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    return img


def load_image(path: str | os.PathLike, *, frame: int = 0, max_pixels: int | None = None,
               raw_options: dict[str, Any] | None = None) -> tuple[np.ndarray, ImageMeta]:
    """Decode an image file into the float32 working buffer.  Returns (pixels, meta)."""
    p = Path(path)
    _check_exists(p)
    limit = max_pixels or DEFAULT_MAX_PIXELS
    if p.suffix.lower() in RAW_EXTS:
        return _load_raw(p, raw_options or {})
    if _is_jxl(p):
        return _load_jxl(p, frame, limit)
    meta = ImageMeta(path=str(p), file_size=p.stat().st_size)
    old_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = limit
    try:
        with Image.open(p) as im:
            meta.format = im.format
            meta.frames = getattr(im, "n_frames", 1)
            meta.animated = bool(getattr(im, "is_animated", False))
            if frame:
                im.seek(frame)
            if im.size[0] * im.size[1] > limit:
                raise LazyRoomError(ErrorCode.RESOURCE_LIMIT, f"Image has {im.size[0] * im.size[1]} pixels; limit is {limit}")
            meta.exif = im.info.get("exif")
            meta.xmp = im.info.get("xmp") or im.info.get("XML:com.adobe.xmp")
            if isinstance(meta.xmp, str):
                meta.xmp = meta.xmp.encode("utf-8")
            dpi = im.info.get("dpi")
            if dpi:
                meta.dpi = (float(dpi[0]), float(dpi[1]))
            meta.mode = im.mode
            orient = im.getexif().get(0x0112)
            sixteen = im.mode in ("I;16", "I;16B", "I;16L", "I") or (
                im.format == "PNG" and _png_is_16bit(p)) or (im.format == "TIFF" and im.mode.startswith("I;16"))
            arr: np.ndarray | None = None
            if sixteen and not frame:
                arr = _read_cv16(p)
                if arr is not None:
                    meta.bit_depth = 16
                    if orient and orient != 1:
                        arr = _apply_orientation(arr, orient)
                        meta.orientation_applied = orient
            if arr is None:
                im.load()
                pil = im
                if orient and orient != 1 and not frame:
                    pil = ImageOps.exif_transpose(im)
                    meta.orientation_applied = orient
                pil = _icc_to_srgb(pil, meta)
                if pil.mode == "P":
                    pil = pil.convert("RGBA" if "transparency" in pil.info else "RGB")
                elif pil.mode in ("I", "I;16", "I;16B", "I;16L"):
                    a = np.asarray(pil, dtype=np.float32)
                    pil = None
                    arr = np.clip(a / (65535.0 if a.max() > 255 else 255.0), 0, 1)
                elif pil.mode == "F":
                    arr = np.asarray(pil, dtype=np.float32)
                    pil = None
                elif pil.mode == "CMYK":
                    pil = pil.convert("RGB")
                elif pil.mode not in ("RGB", "RGBA", "L", "LA"):
                    pil = pil.convert("RGBA")
                if pil is not None:
                    arr = np.asarray(pil)
    except LazyRoomError:
        raise
    except Image.DecompressionBombError as exc:
        raise LazyRoomError(ErrorCode.RESOURCE_LIMIT, str(exc)) from exc
    except Exception as exc:
        code = ErrorCode.UNSUPPORTED_FORMAT if "cannot identify" in str(exc) else ErrorCode.DECODE_FAILED
        raise LazyRoomError(code, f"Cannot decode {p.name}: {exc}",
                              hint="The file may be corrupt, truncated, or in an unsupported format.") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit
    img = normalize_channels(arr)
    img = opaque_if_full(img)
    meta.height, meta.width = img.shape[:2]
    meta.has_alpha = img.shape[2] == 4
    return img, meta


def _apply_orientation(arr: np.ndarray, orient: int) -> np.ndarray:
    ops = {
        2: lambda a: a[:, ::-1],
        3: lambda a: a[::-1, ::-1],
        4: lambda a: a[::-1, :],
        5: lambda a: np.swapaxes(a, 0, 1),
        6: lambda a: np.rot90(a, -1),
        7: lambda a: np.rot90(a, 1)[::-1, ::-1],
        8: lambda a: np.rot90(a, 1),
    }
    f = ops.get(orient)
    return np.ascontiguousarray(f(arr)) if f else arr


def _load_raw(p: Path, opts: dict[str, Any]) -> tuple[np.ndarray, ImageMeta]:
    if not HAVE_RAW:
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, "RAW decoding requires the optional 'rawpy' package",
                              hint="pip install 'lazyroom[raw]'")
    meta = ImageMeta(path=str(p), format="RAW", file_size=p.stat().st_size, bit_depth=16)
    try:
        with rawpy.imread(str(p)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=opts.get("use_camera_wb", True),
                use_auto_wb=opts.get("use_auto_wb", False),
                no_auto_bright=opts.get("no_auto_bright", False),
                output_bps=16,
                output_color=rawpy.ColorSpace.sRGB,
                half_size=opts.get("half_size", False),
            )
    except Exception as exc:
        raise LazyRoomError(ErrorCode.DECODE_FAILED, f"Cannot decode RAW file {p.name}: {exc}") from exc
    img = normalize_channels(rgb)
    meta.height, meta.width = img.shape[:2]
    try:
        with Image.open(p) as im:  # many RAWs carry a TIFF-style EXIF block
            meta.exif = im.info.get("exif")
    except Exception:
        pass
    return img, meta


# ---------------------------------------------------------------------------
# JPEG XL (optional, via imagecodecs/libjxl)
# ---------------------------------------------------------------------------

_JXL_BOX_SIG = b"\x00\x00\x00\x0cJXL \r\n\x87\n"


def _is_jxl(p: Path) -> bool:
    if p.suffix.lower() == ".jxl":
        return True
    try:
        with open(p, "rb") as f:
            head = f.read(12)
    except OSError:
        return False
    return head[:2] == b"\xff\x0a" or head == _JXL_BOX_SIG


def jxl_boxes(data: bytes) -> list[tuple[bytes, bytes]]:
    """ISOBMFF boxes of a containerised JPEG XL file as (type, payload); [] for a bare codestream."""
    out, pos = [], 0
    if data[:12] != _JXL_BOX_SIG:
        return out
    while pos + 8 <= len(data):
        size, typ, hdr = int.from_bytes(data[pos:pos + 4], "big"), data[pos + 4:pos + 8], 8
        if size == 1:
            size, hdr = int.from_bytes(data[pos + 8:pos + 16], "big"), 16
        elif size == 0:
            size = len(data) - pos
        if size < hdr:
            raise ValueError("corrupt JPEG XL box")
        out.append((typ, data[pos + hdr:pos + size]))
        pos += size
    return out


class _JxlBits:
    """LSB-first bit reader for JPEG XL headers (ISO/IEC 18181-1)."""

    def __init__(self, data: bytes) -> None:
        self.d, self.pos = data, 0

    def u(self, n: int) -> int:
        v = 0
        for i in range(n):
            v |= ((self.d[self.pos >> 3] >> (self.pos & 7)) & 1) << i
            self.pos += 1
        return v

    def u32(self, *dist: tuple[int, int]) -> int:  # dist entries: (bits, offset); Val(v) == (0, v)
        n, off = dist[self.u(2)]
        return off + (self.u(n) if n else 0)


def jxl_header(data: bytes) -> dict[str, Any]:
    """Size and animation header of a JPEG XL file (bare codestream or container).

    ``anim_bitpos`` is the codestream bit offset of the AnimationHeader (animated files only).
    """
    if data[:2] == b"\xff\x0a":
        cs = data
    else:
        boxes = jxl_boxes(data)
        cs = next((p for t, p in boxes if t == b"jxlc"), b"") or b"".join(p[4:] for t, p in boxes if t == b"jxlp")
    if cs[:2] != b"\xff\x0a":
        raise ValueError("not a JPEG XL file")
    br = _JxlBits(cs)
    br.u(16)
    size_dist = ((9, 1), (13, 1), (18, 1), (30, 1))
    ratios = ((1, 1), (12, 10), (4, 3), (3, 2), (16, 9), (5, 4), (2, 1))

    def size_header(div8_y, plain_y, div8_x, plain_x):
        div8 = br.u(1)
        h = (br.u32(*div8_y) if isinstance(div8_y, tuple) else br.u(div8_y) + 1) * 8 if div8 else br.u32(*plain_y)
        ratio = br.u(3)
        if ratio:
            return h * ratios[ratio - 1][0] // ratios[ratio - 1][1], h
        w = (br.u32(*div8_x) if isinstance(div8_x, tuple) else br.u(div8_x) + 1) * 8 if div8 else br.u32(*plain_x)
        return w, h

    w, h = size_header(5, size_dist, 5, size_dist)
    info: dict[str, Any] = {"width": w, "height": h, "animated": False}
    if not br.u(1) and br.u(1):  # !all_default and extra_fields
        br.u(3)  # orientation
        if br.u(1):  # intrinsic size
            size_header(5, size_dist, 5, size_dist)
        if br.u(1):  # preview
            pv_div8 = ((0, 16), (0, 32), (5, 1), (9, 33))
            pv = ((6, 1), (8, 65), (10, 321), (12, 1345))
            size_header(pv_div8, pv, pv_div8, pv)
        if br.u(1):
            info.update(animated=True, anim_bitpos=br.pos,
                        tps_numerator=br.u32((0, 100), (0, 1000), (10, 1), (30, 1)),
                        tps_denominator=br.u32((0, 1), (0, 1001), (8, 1), (10, 1)),
                        num_loops=br.u32((0, 0), (3, 0), (16, 0), (32, 0)))
    return info


def _load_jxl(p: Path, frame: int, limit: int) -> tuple[np.ndarray, ImageMeta]:
    if not HAVE_JXL:
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE,
                              f"Decoding JPEG XL ({p.name}) requires the optional 'imagecodecs' package",
                              details={"backend": "imagecodecs"}, hint=JXL_HINT)
    data = p.read_bytes()
    meta = ImageMeta(path=str(p), format="JXL", file_size=len(data))
    try:
        info = jxl_header(data)
        boxes = jxl_boxes(data)
    except (ValueError, IndexError) as exc:
        raise LazyRoomError(ErrorCode.DECODE_FAILED, f"Cannot read JPEG XL header of {p.name}: {exc}") from exc
    if info["width"] * info["height"] > limit:
        raise LazyRoomError(ErrorCode.RESOURCE_LIMIT,
                              f"Image has {info['width'] * info['height']} pixels; limit is {limit}")
    try:
        arr = _imagecodecs.jpegxl_decode(data, index=frame)
    except Exception as exc:
        raise LazyRoomError(ErrorCode.DECODE_FAILED, f"Cannot decode {p.name}: {exc}",
                              hint="The file may be corrupt, truncated, or use unsupported JPEG XL features.") from exc
    meta.animated = bool(info["animated"])
    if meta.animated:  # counting frames would decode all of them (unbounded memory): left unknown
        meta.frames = None  # type: ignore[assignment]
    meta.mode = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}.get(arr.shape[2] if arr.ndim == 3 else 1)
    meta.bit_depth = 8 if arr.dtype == np.uint8 else 16
    for typ, payload in boxes:
        if typ == b"Exif" and len(payload) > 4:
            meta.exif = payload[4 + int.from_bytes(payload[:4], "big"):]
        elif typ == b"xml ":
            meta.xmp = payload
    img = opaque_if_full(normalize_channels(arr))
    meta.height, meta.width = img.shape[:2]
    meta.has_alpha = img.shape[2] == 4
    return img, meta


def _encode_jxl(img: np.ndarray, quality: int, lossless: bool, bit_depth: int, exif: bytes | None,
                xmp: bytes | None, extra: dict[str, Any] | None) -> bytes:
    """JPEG XL: quality 1..99 -> libjxl distance via JxlEncoderDistanceFromQuality; >= 100 or lossless -> lossless."""
    from .image import has_alpha

    extra = extra or {}
    arr = to_uint16(img) if bit_depth == 16 else to_uint8(img)
    if not has_alpha(img):
        arr = np.ascontiguousarray(arr[..., :3])
    kw: dict[str, Any] = {"effort": int(extra.get("effort", 7)), "decodingspeed": int(extra.get("decoding_speed", 0))}
    if lossless or quality >= 100:
        kw["lossless"] = True
    else:
        kw["level"] = max(1, int(quality))
    try:
        cs = bytes(_imagecodecs.jpegxl_encode(arr, **kw))
    except Exception as exc:
        raise LazyRoomError(ErrorCode.ENCODE_FAILED, f"Failed to encode JPEG XL: {exc}") from exc
    if not exif and not xmp:
        return cs
    box = lambda t, p: (8 + len(p)).to_bytes(4, "big") + t + p  # noqa: E731
    out = _JXL_BOX_SIG + box(b"ftyp", b"jxl \x00\x00\x00\x00jxl ")
    if exif:
        out += box(b"Exif", b"\x00\x00\x00\x00" + (exif[6:] if exif.startswith(b"Exif\x00\x00") else exif))
    if xmp:
        out += box(b"xml ", xmp)
    return out + box(b"jxlc", cs)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def format_for_path(path: str | os.PathLike, fmt: str | None = None) -> tuple[str, str]:
    """Return (extension, pillow_format) for a destination path / explicit format."""
    if fmt:
        f = fmt.lower().lstrip(".")
        ext = {"jpeg": ".jpg", "tiff": ".tif", "heif": ".heic", "jpeg2000": ".jp2"}.get(f, "." + f)
    else:
        ext = Path(path).suffix.lower()
    if ext == ".jxl" and not HAVE_JXL:
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, "Writing JPEG XL requires the optional 'imagecodecs' package",
                              details={"backend": "imagecodecs"}, hint=JXL_HINT)
    if ext not in WRITE_FORMATS:
        raise LazyRoomError(ErrorCode.UNSUPPORTED_FORMAT, f"Cannot write format '{ext or fmt}'",
                              details={"supported": sorted(WRITE_FORMATS)},
                              hint=f"Use one of: {', '.join(sorted(WRITE_FORMATS))}")
    return ext, WRITE_FORMATS[ext][0]


def _filter_exif(exif_bytes: bytes | None, policy: str) -> bytes | None:
    if not exif_bytes or policy == "strip":
        return None
    try:
        ex = Image.Exif()
        ex.load(exif_bytes)
        ex[0x0112] = 1  # orientation already applied to pixels
        if policy == "keep_no_gps":
            ex.pop(0x8825, None)  # GPSInfo IFD
        return ex.tobytes()
    except Exception:
        return None


def encode_image(img: np.ndarray, fmt: str, *, quality: int = 92, bit_depth: int = 8, lossless: bool = False,
                 meta: ImageMeta | None = None, metadata: str = "keep", dpi: tuple[float, float] | None = None,
                 background: tuple[float, float, float] = (1.0, 1.0, 1.0), extra: dict[str, Any] | None = None) -> bytes:
    """Encode the working buffer.  ``metadata`` is one of keep | keep_no_gps | strip."""
    ext, pil_fmt = format_for_path("x" + fmt if fmt.startswith(".") else "x." + fmt)
    _, alpha_ok, deep_ok, _ = WRITE_FORMATS[ext]
    from .image import composite_on, has_alpha

    if has_alpha(img) and not alpha_ok:
        img = composite_on(img, background)
    if pil_fmt == "JXL":
        exif = _filter_exif(meta.exif, metadata) if meta is not None else None
        xmp = meta.xmp if meta is not None and metadata != "strip" else None
        return _encode_jxl(img, quality, lossless, bit_depth, exif, xmp, extra)
    buf = io.BytesIO()
    if bit_depth == 16 and deep_ok:
        arr16 = to_uint16(img)
        if ext == ".png" or ext in (".tif", ".tiff"):
            code = cv2.COLOR_RGBA2BGRA if arr16.shape[2] == 4 else cv2.COLOR_RGB2BGR
            ok, enc = cv2.imencode(ext, cv2.cvtColor(arr16, code))
            if not ok:
                raise LazyRoomError(ErrorCode.ENCODE_FAILED, f"OpenCV failed to encode 16-bit {ext}")
            return enc.tobytes()
    arr = to_uint8(img)
    mode = "RGBA" if arr.shape[2] == 4 else "RGB"
    pil = Image.fromarray(arr, mode)
    kw: dict[str, Any] = {}
    if pil_fmt in ("JPEG", "WEBP", "AVIF", "HEIF", "JPEG2000"):
        kw["quality"] = int(quality)
    if pil_fmt == "JPEG":
        kw.update(optimize=True, progressive=True, subsampling=0 if quality >= 90 else 2)
    if pil_fmt == "WEBP":
        kw["lossless"] = bool(lossless)
        kw["method"] = 6
    if pil_fmt == "AVIF" and lossless:
        kw["quality"] = 100
    if pil_fmt == "PNG":
        kw["optimize"] = True
    if pil_fmt == "TIFF":
        kw["compression"] = "tiff_deflate"
    if pil_fmt == "JPEG2000" and lossless:
        kw.pop("quality", None)
        kw["irreversible"] = False
    if pil_fmt == "GIF":
        pil = pil.convert("RGBA").quantize(256, method=Image.Quantize.FASTOCTREE) if mode == "RGBA" else pil.quantize(256)
    if pil_fmt not in ("GIF", "ICO", "QOI", "PPM", "BMP", "TGA"):
        kw["icc_profile"] = _SRGB_ICC_BYTES
    if dpi or (meta and meta.dpi):
        kw["dpi"] = dpi or meta.dpi  # type: ignore[union-attr]
    if meta is not None and pil_fmt in ("JPEG", "WEBP", "PNG", "TIFF", "AVIF", "HEIF"):
        ex = _filter_exif(meta.exif, metadata)
        if ex:
            kw["exif"] = ex
        if meta.xmp and metadata != "strip" and pil_fmt in ("JPEG", "WEBP", "PNG", "TIFF"):
            kw["xmp"] = meta.xmp
    if extra:
        kw.update(extra)
    try:
        pil.save(buf, format=pil_fmt, **kw)
    except Exception as exc:
        raise LazyRoomError(ErrorCode.ENCODE_FAILED, f"Failed to encode {pil_fmt}: {exc}") from exc
    return buf.getvalue()


def atomic_write(path: str | os.PathLike, data: bytes, *, overwrite: bool = False) -> Path:
    p = Path(path)
    if p.exists() and not overwrite:
        raise LazyRoomError(ErrorCode.OUTPUT_EXISTS, f"Refusing to overwrite existing file: {p}",
                              hint="Choose another output path, or pass overwrite=true if replacing it is intended.")
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pm-", suffix=p.suffix, dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if os.name != "nt":  # mkstemp creates 0600 files; give outputs normal permissions (umask applied)
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def save_image(img: np.ndarray, path: str | os.PathLike, *, fmt: str | None = None, overwrite: bool = False,
               **kwargs: Any) -> dict[str, Any]:
    ext, _ = format_for_path(path, fmt)
    data = encode_image(img, ext, **kwargs)
    out = atomic_write(path, data, overwrite=overwrite)
    return {"path": str(out), "format": ext.lstrip("."), "bytes": len(data),
            "width": int(img.shape[1]), "height": int(img.shape[0])}


def supported_formats() -> dict[str, Any]:
    return {
        "read": READ_EXTS,
        "write": sorted(WRITE_FORMATS),
        "write_alpha": sorted(k for k, v in WRITE_FORMATS.items() if v[1]),
        "write_16bit": sorted(k for k, v in WRITE_FORMATS.items() if v[2]),
        "raw_available": HAVE_RAW,
        "heif_available": HAVE_HEIF,
        "jxl_available": HAVE_JXL,
        "metadata_policies": ["keep", "keep_no_gps", "strip"],
    }
