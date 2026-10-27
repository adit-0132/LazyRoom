"""Compression lab — port of ImageToolbox ``feature/compression-lab`` (Apache-2.0, T8RIN).

Sources: ``CompressionLabComponent.kt`` (format list, defaults, search modes, metric pipeline),
``CompressionLabControls.kt`` (ranges shown in the UI, result card text), ``core/domain ImageFormat.kt``
(format titles, compression ranges), ``Quality.kt`` (``coerceIn`` defaults), ``core/data .../compressor/*``
(what each backend does with the quality value), ``lib/opencv-tools ImageDiffTool`` (SSIM / PSNR, shared
port in :mod:`._image_diff`).

Behaviour kept from the source:

* formats are evaluated in ImageToolbox's ``ImageFormat.entries`` order (selection order is irrelevant);
  default selection = JPG, WEBP Lossy, AV1 Lossy, JXL Lossy;
* ``qualityRange``: only formats whose first compression type is *Quality* are searched; the others
  are encoded once with fixed settings (``Quality.Base().coerceIn(format)``);
* Manual: quality clamped into the format range; TargetQuality: binary search for the *lowest* quality
  whose SSIM >= target/100 (fallback: max quality, flagged ``target_satisfied=false``); TargetSize:
  binary search for the *highest* quality whose size <= target kB * 1024 (fallback: min quality,
  flagged); one encode per quality value (memoised), so at most ceil(log2(101)) = 7 encodes per format;
* metrics on a copy scaled to fit 1024 px (bilinear), decoded result scaled the same way, RGBA,
  ImageDiffTool SSIM (Gaussian 11x11, sigma 1.5, per-channel mean) and PSNR (255^2 / MSE);
* failures do not stop the run; they are listed like ImageToolbox's "failed formats" line.

Encoders are LazyRoom's (``core.io.encode_image``); see the port report for per-format mapping and
the formats that have no encoder here (MozJPEG, Jpegli, PNG Lossy/OxiPNG/ImageQuant, AV2, VVC, JXL).
"""
from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, features

from ..core.errors import invalid
from ..core.image import resize, to_uint8
from ..core.io import HAVE_HEIF, atomic_write, encode_image
from ..core.params import Bool, Choice, File, Int, StringList, Text
from ._image_diff import metric, to_rgba8
from .registry import out_path, read_image, tool

F32 = np.float32
METRIC_MAX_SIZE = 1024  # CompressionLabComponent.MetricMaxSize
DEFAULT_FORMATS = ["jpg", "webp_lossy", "avif_lossy", "jxl_lossy"]  # DefaultFormats
SEARCH_MODES = ["manual", "target_quality", "target_size"]  # CompressionSearchMode.entries
ICON_SIZES = [16, 32, 48, 64, 128, 256]  # IcoBackend.ICON_SIZES


def human_file_size(n: int) -> str:
    """``humanFileSize`` (ByteUtils.kt), '.' decimal separator."""
    if -1024 < n < 1024:
        return f"{n} B"
    units = "kMGTPE"
    i = 0
    while n <= -999950 or n >= 999950:
        n //= 1024
        i += 1
    return f"{n / 1024.0:.1f} {units[i]}B"


@dataclass(frozen=True)
class LabFormat:
    id: str
    title: str
    extension: str
    quality_range: tuple[int, int] | None  # CompressionLabComponent.qualityRange(format)
    encoder: Callable[[np.ndarray, int | None], bytes] | None = None
    unavailable: str | None = None
    lossless: bool = False
    note: str = ""


def _q(fmt: LabFormat, value: int | None) -> dict[str, Any]:
    """``qualityFor(format, value)`` as a description of the encoder setting actually used."""
    fid = fmt.id
    if value is None:  # Quality.Base().coerceIn(format)
        if fid.startswith("avif"):
            return {"type": "Avif", "quality_value": 50, "effort": 0}
        if fid.startswith(("heic", "heif")):
            return {"type": "Heic", "quality_value": 100}
        if fid in ("tif", "tiff"):
            return {"type": "Tiff", "compression_scheme": 5, "compression": "JPEG"}
        return {"type": "Base", "quality_value": 100}
    lo, hi = fmt.quality_range or (0, 100)
    v = max(lo, min(hi, int(value)))
    if fid.startswith("avif"):
        return {"type": "Avif", "quality_value": v, "effort": 0}
    if fid.startswith(("heic", "heif")):
        return {"type": "Heic", "quality_value": v}
    return {"type": "Base", "quality_value": v}


# -- encoders (LazyRoom's) ---------------------------------------------------

def _enc(ext: str, **kw) -> Callable[[np.ndarray, int | None], bytes]:
    def f(img: np.ndarray, q: int | None) -> bytes:
        return encode_image(img, ext, quality=100 if q is None else int(q), metadata="strip", **kw)
    return f


def _enc_jp2(j2k: bool) -> Callable[[np.ndarray, int | None], bytes]:
    def f(img: np.ndarray, q: int | None) -> bytes:  # JP2Encoder.setVisualQuality(q): PSNR target in dB
        extra: dict[str, Any] = {"quality_mode": "dB", "quality_layers": [float(100 if q is None else q)]}
        if j2k:
            extra["no_jp2"] = True  # raw J2K codestream
        return encode_image(img, ".jp2", metadata="strip", extra=extra)
    return f


def _enc_ico(img: np.ndarray, q: int | None) -> bytes:
    """IcoBackend: 16..256 px icons, each ``ResizeType.Fit`` (centred, transparent canvas), 32-bit BMP."""
    rgba = img if img.shape[2] == 4 else np.dstack([img, np.ones(img.shape[:2], F32)])
    h, w = rgba.shape[:2]
    s = 256 / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = np.zeros((256, 256, 4), F32)
    x0, y0 = (256 - nw) // 2, (256 - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resize(rgba, nw, nh)
    return encode_image(canvas, ".ico", metadata="strip",
                        extra={"sizes": [(s_, s_) for s_ in ICON_SIZES], "bitmap_format": "bmp"})


def _avif_ok() -> str | None:
    return None if features.check("avif") else "Pillow was built without AVIF support"


def _heif_ok() -> str | None:
    if not HAVE_HEIF:
        return "pillow-heif is not installed"
    try:
        import pillow_heif

        if not any("265" in k or "hevc" in str(v).lower() for k, v in pillow_heif.libheif_info().get("encoders", {}).items()):
            return "libheif has no HEVC encoder"
    except Exception as exc:  # pragma: no cover
        return f"pillow-heif unusable: {exc}"
    return None


def _formats() -> list[LabFormat]:
    """``ImageFormat.entries`` in order, mapped to LazyRoom encoders."""
    webp = None if features.check("webp") else "Pillow was built without WebP support"
    jp2 = None if features.check("jpg_2000") else "Pillow was built without OpenJPEG"
    avif, heif = _avif_ok(), _heif_ok()
    na = lambda why: why  # noqa: E731
    heic_lossless = _enc(".heic", extra={"quality": -1, "chroma": 444})
    return [
        LabFormat("jpg", "JPG", "jpg", (0, 100), _enc(".jpg")),
        LabFormat("jpeg", "JPEG", "jpeg", (0, 100), _enc(".jpg")),
        LabFormat("mozjpeg", "MozJPEG", "jpg", (0, 100), unavailable=na("no MozJPEG encoder (ImageToolbox uses Aire)")),
        LabFormat("jpegli", "Jpegli", "jpg", (0, 100), unavailable=na("no Jpegli encoder")),
        LabFormat("png_lossless", "PNG Lossless", "png", None, _enc(".png"), lossless=True),
        LabFormat("png_lossy", "PNG Lossy", "png", None,
                  unavailable=na("no Aire PNG quantizer (Wu, 512 colours, Jarvis-Judice-Ninke dithering)")),
        LabFormat("oxipng", "OxiPNG", "png", None, unavailable=na("no OxiPNG optimizer")),
        LabFormat("imagequant", "ImageQuant", "png", (0, 100), unavailable=na("no libimagequant with quality control")),
        LabFormat("bmp", "BMP", "bmp", None, _enc(".bmp"), lossless=True),
        LabFormat("webp_lossless", "WEBP Lossless", "webp", None, _enc(".webp", lossless=True, extra={"method": 0}),
                  webp, True, "Bitmap.compress(WEBP_LOSSLESS, 100): quality 100 as lossless effort, method 0"),
        LabFormat("webp_lossy", "WEBP Lossy", "webp", (0, 100), _enc(".webp"), webp),
        LabFormat("avif_lossless", "AV1 Lossless", "avif", None, _enc(".avif", lossless=True,
                                                                     extra={"subsampling": "4:4:4"}), avif, True,
                  "near-lossless: Pillow cannot select the identity matrix, so this is quality 100 YUV 4:4:4"),
        LabFormat("avif_lossy", "AV1 Lossy", "avif", (1, 100), _enc(".avif"), avif),
        LabFormat("av2_lossless", "AV2 Lossless", "avif", None, unavailable=na("no AV2 encoder")),
        LabFormat("av2_lossy", "AV2 Lossy", "avif", (1, 100), unavailable=na("no AV2 encoder")),
        LabFormat("heic_lossless", "HEIC Lossless", "heic", None, heic_lossless, heif, True,
                  "x265 lossless in YCbCr 4:4:4 (ImageToolbox default losslessYcbcr)"),
        LabFormat("heic_lossy", "HEIC Lossy", "heic", (0, 100), _enc(".heic"), heif),
        LabFormat("heif_lossless", "HEIF Lossless", "heif", None, heic_lossless, heif, True,
                  "same encoder as HEIC Lossless"),
        LabFormat("heif_lossy", "HEIF Lossy", "heif", (0, 100), _enc(".heic"), heif),
        LabFormat("vvc_lossless", "VVC Lossless", "heif", None, unavailable=na("no VVC (H.266) encoder")),
        LabFormat("vvc_lossy", "VVC Lossy", "heif", (1, 100), unavailable=na("no VVC (H.266) encoder")),
        LabFormat("jxl_lossless", "JXL Lossless", "jxl", None, unavailable=na("no JPEG XL encoder")),
        LabFormat("jxl_lossy", "JXL Lossy", "jxl", (1, 100), unavailable=na("no JPEG XL encoder")),
        LabFormat("jixel_lossless", "Jixel Lossless", "jxl", None, unavailable=na("no JPEG XL (Jixel) encoder")),
        LabFormat("jixel_lossy", "Jixel Lossy", "jxl", (1, 100), unavailable=na("no JPEG XL (Jixel) encoder")),
        LabFormat("tif", "TIF", "tif", None, _enc(".tif", extra={"compression": "jpeg"}),
                  note="ImageToolbox's default Quality.Tiff(5) is an ordinal -> JPEG-compressed TIFF"),
        LabFormat("tiff", "TIFF", "tiff", None, _enc(".tif", extra={"compression": "jpeg"}),
                  note="ImageToolbox's default Quality.Tiff(5) is an ordinal -> JPEG-compressed TIFF"),
        LabFormat("jp2", "JP2", "jp2", (20, 100), _enc_jp2(False), jp2, note="quality = target PSNR in dB"),
        LabFormat("j2k", "J2K", "j2k", (20, 100), _enc_jp2(True), jp2, note="quality = target PSNR in dB"),
        LabFormat("qoi", "QOI", "qoi", None, _enc(".qoi"), lossless=True),
        LabFormat("ico", "ICO", "ico", None, _enc_ico, note="16, 32, 48, 64, 128, 256 px icons"),
        LabFormat("gif", "GIF", "gif", None, _enc(".gif"),
                  note="ImageToolbox searches GIF quality 0..100 (native GifEncoder); LazyRoom's GIF encoder "
                       "has no quality setting, so GIF is evaluated with fixed settings"),
    ]


FORMAT_IDS = [f.id for f in _formats()]
_ALIASES = {"webp": "webp_lossy", "avif": "avif_lossy", "av1_lossy": "avif_lossy", "av1_lossless": "avif_lossless",
            "png": "png_lossless", "heic": "heic_lossy", "heif": "heif_lossy", "jxl": "jxl_lossy"}


def resolve_formats(names: list[str]) -> list[LabFormat]:
    table = _formats()
    by_key: dict[str, LabFormat] = {}
    for f in table:
        by_key[f.id] = f
        by_key[f.title.lower().replace(" ", "_")] = f
    chosen = set()
    for n in names:
        k = n.strip().lower().replace(" ", "_").replace("-", "_").lstrip(".")
        k = _ALIASES.get(k, k)
        if k not in by_key:
            raise invalid(f"Unknown compression lab format {n!r}", param="formats", valid=FORMAT_IDS)
        chosen.add(by_key[k].id)
    if not chosen:
        raise invalid("Select at least one format", param="formats")
    return [f for f in table if f.id in chosen]  # sortedBy(AvailableFormats::indexOf)


# -- metrics ---------------------------------------------------------------------

def _fit_metrics(rgba8: np.ndarray) -> np.ndarray:
    """``Bitmap.scaleForMetrics`` (bilinear ``Bitmap.scale``) on premultiplied RGBA8."""
    h, w = rgba8.shape[:2]
    m = max(w, h)
    if m <= METRIC_MAX_SIZE:
        return rgba8
    s = METRIC_MAX_SIZE / m
    return cv2.resize(rgba8, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_LINEAR)


def decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.uint8).astype(F32) / 255.0


@dataclass
class Candidate:
    quality_value: int | None
    quality: dict[str, Any]
    data: bytes
    encoding_time_ms: float
    ssim: float | None = None
    target_satisfied: bool = True


class _Lab:
    def __init__(self, img: np.ndarray):
        # the benchmark is an 8-bit bitmap in ImageToolbox: quantise once (as encode_image does) so a
        # lossless round trip compares equal
        self.img = to_uint8(img).astype(F32) / 255.0
        self.metric_source = _fit_metrics(to_rgba8(self.img))
        self.encodes = 0

    def _decoded_metric_img(self, data: bytes) -> np.ndarray:
        return _fit_metrics(to_rgba8(decode(data)))

    def ssim(self, data: bytes) -> float:
        return float(metric(self.metric_source, self._decoded_metric_img(data), "SSIM"))

    def encode(self, fmt: LabFormat, q: int | None, with_ssim: bool) -> Candidate:
        quality = _q(fmt, q)
        t0 = time.perf_counter()
        data = fmt.encoder(self.img, None if q is None else quality.get("quality_value"))
        dt = (time.perf_counter() - t0) * 1000
        self.encodes += 1
        return Candidate(q, quality, data, dt, self.ssim(data) if with_ssim else None)

    def find_candidate(self, fmt: LabFormat, mode: str, manual: int, target_quality: int, target_kb: int) -> Candidate:
        rng = fmt.quality_range
        if fmt.id == "gif":
            rng = None  # see LabFormat note
        if rng is None:
            c = self.encode(fmt, None, with_ssim=mode == "target_quality")
            if mode == "target_quality":
                c.target_satisfied = (c.ssim or 0.0) >= target_quality / 100.0
            elif mode == "target_size":
                c.target_satisfied = len(c.data) <= target_kb * 1024
            return c
        cache: dict[int, Candidate] = {}

        def at(q: int, with_ssim: bool = False) -> Candidate:
            c = cache.get(q)
            if c is not None and (not with_ssim or c.ssim is not None):
                return c
            c = self.encode(fmt, q, with_ssim)
            cache[q] = c
            return c

        lo, hi = rng
        if mode == "manual":
            return at(max(lo, min(hi, manual)))
        best: Candidate | None = None
        if mode == "target_quality":
            target = target_quality / 100.0
            while lo <= hi:
                q = (lo + hi) // 2
                c = at(q, with_ssim=True)
                if (c.ssim or 0.0) >= target:
                    best, hi = c, q - 1
                else:
                    lo = q + 1
            if best is None:
                c = at(rng[1], with_ssim=True)
                best = Candidate(c.quality_value, c.quality, c.data, c.encoding_time_ms, c.ssim, False)
            return best
        target_bytes = target_kb * 1024
        while lo <= hi:
            q = (lo + hi) // 2
            c = at(q)
            if len(c.data) <= target_bytes:
                best, lo = c, q + 1
            else:
                hi = q - 1
        if best is None:
            c = at(rng[0])
            best = Candidate(c.quality_value, c.quality, c.data, c.encoding_time_ms, c.ssim, False)
        return best

    def to_result(self, fmt: LabFormat, c: Candidate) -> dict[str, Any]:
        dec = self._decoded_metric_img(c.data)
        ssim = c.ssim if c.ssim is not None else float(metric(self.metric_source, dec, "SSIM"))
        psnr = float(metric(self.metric_source, dec, "PSNR"))
        h, w = self.img.shape[:2]
        size = len(c.data)
        row: dict[str, Any] = {
            "format": fmt.id, "title": fmt.title, "extension": fmt.extension,
            "quality_value": c.quality_value,
            "encoder_setting": f"Q {c.quality_value}" if c.quality_value is not None else "fixed settings",
            "quality": c.quality, "width": int(w), "height": int(h),
            "size_bytes": size, "size": human_file_size(size),
            "bits_per_pixel": round(size * 8 / (w * h), 4),
            "encoding_time_ms": round(c.encoding_time_ms, 1),
            "ssim": round(ssim, 6),
            "psnr": None if math.isinf(psnr) else round(psnr, 3),
            "psnr_infinite": math.isinf(psnr),
            "target_satisfied": bool(c.target_satisfied),
        }
        if fmt.note:
            row["note"] = fmt.note
        return row


def run_lab(img: np.ndarray, formats: list[LabFormat], mode: str = "manual", quality: int = 85,
            target_quality: int = 95, target_size_kb: int = 500) -> tuple[list[dict], list[dict], list[bytes], int]:
    lab = _Lab(img)
    results, failed, blobs = [], [], []
    for fmt in formats:
        if fmt.unavailable or fmt.encoder is None:
            failed.append({"format": fmt.id, "title": fmt.title, "reason": fmt.unavailable or "no encoder"})
            continue
        try:
            c = lab.find_candidate(fmt, mode, quality, target_quality, target_size_kb)
            results.append(lab.to_result(fmt, c))
            blobs.append(c.data)
        except Exception as exc:  # runSuspendCatching: record and continue
            failed.append({"format": fmt.id, "title": fmt.title, "reason": str(exc) or type(exc).__name__})
    return results, failed, blobs, lab.encodes


@tool("compression_lab", "Compression lab", category="analysis", params=[
    File("input", "", file_kind="image", required=True, description="Benchmark image (encoded at full size)"),
    StringList("formats", DEFAULT_FORMATS,
               description="Formats to compare (ImageToolbox ids, evaluated in ImageToolbox's order): "
                           + ", ".join(FORMAT_IDS) + ". Default: JPG, WEBP Lossy, AV1 Lossy, JXL Lossy"),
    Choice("mode", "manual", choices=SEARCH_MODES,
           description="manual: fixed quality; target_quality: lowest quality reaching the SSIM target; "
                       "target_size: highest quality not exceeding the size target"),
    Int("quality", 85, min=0, max=100, description="manual mode quality (clamped into each format's range)"),
    Int("target_quality", 95, min=1, max=100, unit="% SSIM",
        description="target_quality mode: minimum SSIM x 100 (ImageToolbox slider 50..100)"),
    Int("target_size_kb", 500, min=1, max=100_000, unit="kB",
        description="target_size mode: maximum size in kB (1 kB = 1024 bytes; ImageToolbox slider 10..10000)"),
    Text("output_dir", "", description="Optional folder to write each chosen encoding ({stem}_{format}.{ext})"),
    Bool("overwrite", False),
], source={"project": "ImageToolbox", "feature": "Compression Lab (CompressionLabComponent)"})
def compression_lab(api, input, formats, mode, quality, target_quality, target_size_kb, output_dir, overwrite):
    """Compare output formats on one image: size, encode time, SSIM and PSNR at a fixed quality, or the
    quality found by ImageToolbox's binary search for a target SSIM or a target file size."""
    fmts = resolve_formats(formats)
    img, _ = read_image(api, input)
    src = api.ws.policy.check_read(input)
    results, failed, blobs, encodes = run_lab(img, fmts, mode, quality, target_quality, target_size_kb)
    if output_dir:
        od = api.ws.policy.check_write_dir(output_dir)
        od.mkdir(parents=True, exist_ok=True)
        stem = Path(input).stem
        for row, data in zip(results, blobs):
            p = out_path(api, str(od / f"{stem}_{row['format']}.{row['extension']}"), overwrite=overwrite)
            atomic_write(p, data, overwrite=overwrite)
            row["output"] = str(p)
    h, w = img.shape[:2]
    s = min(1.0, METRIC_MAX_SIZE / max(w, h))
    res: dict[str, Any] = {
        "input": str(src), "width": int(w), "height": int(h), "input_bytes": src.stat().st_size,
        "mode": mode, "metric_size": [max(1, round(w * s)), max(1, round(h * s))],
        "results": results, "failed": failed, "encodes": encodes,
    }
    if mode == "manual":
        res["quality"] = quality
    elif mode == "target_quality":
        res["target_ssim"] = target_quality / 100.0
    else:
        res["target_size_bytes"] = target_size_kb * 1024
    return res
