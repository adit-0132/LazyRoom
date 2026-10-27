"""ImageToolbox format tools: JXL Tools (JPEG <-> JXL lossless transcoding, images/animations <-> animated JXL)
and the Telegram sticker preset.  PNG Lossy / OxiPNG / Jpegli / multi-size ICO are ``format_convert`` choices
(encoders in ``_itb_codecs``).  Registered by importing this module (``tools/itb_extras.py`` does so).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

from ..core.errors import ErrorCode, LazyRoomError, invalid
from ..core.image import has_alpha
from ..core.io import atomic_write, encode_image
from ..core.params import Bool, Choice, File, Int, StringList, Text
from . import _itb_codecs as codecs
from .registry import expand_inputs, out_path, read_image, save, tool

JXL_SOURCE = {"project": "ImageToolbox", "feature": "JXL Tools", "license": "Apache-2.0",
              "backend": "imagecodecs (libjxl, BSD-3)"}


def _require_jxl(transcode: bool = False):
    from ..core.io import HAVE_JXL, JXL_HINT

    if HAVE_JXL:
        import imagecodecs

        if not transcode or hasattr(imagecodecs, "jpegxl_encode_jpeg"):
            return imagecodecs
    raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE,
                          "JPEG XL support requires the optional 'imagecodecs' package"
                          + (" with jpegxl_encode_jpeg (lossless JPEG transcoding)" if transcode else ""),
                          details={"backend": "imagecodecs"}, hint=JXL_HINT)


def _files(api, inputs, recursive, exts):
    files = []
    for item in inputs:
        p = Path(item).expanduser()
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            files.extend(str(f) for f in sorted(it) if f.is_file() and f.suffix.lower() in exts)
        else:
            files.extend(expand_inputs(api, [item], recursive))
    if not files:
        raise invalid("No input files matched", hint=f"Expected {', '.join(sorted(exts))} files")
    return files


@tool("jxl_transcode", "JPEG <-> JPEG XL lossless transcoding", category="convert", params=[
    StringList("inputs", [], min_items=1, description="Files, folders or glob patterns"),
    Text("output_dir", ""),
    Choice("mode", "jpeg_to_jxl", choices=["jpeg_to_jxl", "jxl_to_jpeg"],
           description="jpeg_to_jxl: recompress the JPEG bitstream into JXL (reversible, ~20% smaller); "
                       "jxl_to_jpeg: rebuild the bit-identical original JPEG from a transcoded JXL"),
    Bool("verify", True, description="jpeg_to_jxl: check that the JXL reconstructs the original JPEG byte for byte"),
    Bool("recursive", False), Bool("overwrite", False),
], source=JXL_SOURCE)
def jxl_transcode(api, inputs, output_dir, mode, verify, recursive, overwrite):
    """ImageToolbox JXL Tools "JPEG to JXL" / "JXL to JPEG": lossless transcoding (libjxl JPEG reconstruction)."""
    ic = _require_jxl(transcode=True)
    to_jxl = mode == "jpeg_to_jxl"
    files = _files(api, inputs, recursive, {".jpg", ".jpeg", ".jfif"} if to_jxl else {".jxl"})
    od = api.ws.policy.check_write_dir(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        try:
            src = api.ws.policy.check_read(f)
            data = src.read_bytes()
            dst = api.ws.policy.check_write(od / (src.stem + (".jxl" if to_jxl else ".jpg")), overwrite=overwrite)
            res = {"input": str(src), "input_bytes": len(data)}
            if to_jxl:
                if data[:3] != b"\xff\xd8\xff":
                    raise LazyRoomError(ErrorCode.UNSUPPORTED_FORMAT, f"{src.name} is not a JPEG file")
                try:
                    out = bytes(ic.jpegxl_encode_jpeg(data))
                except Exception as exc:
                    raise LazyRoomError(ErrorCode.ENCODE_FAILED, f"libjxl cannot transcode {src.name}: {exc}",
                                          hint="Progressive/arithmetic/CMYK or corrupt JPEGs may not be transcodable; "
                                               "use format_convert with format='jxl' to re-encode.") from exc
                if verify:
                    back = bytes(ic.jpegxl_decode_jpeg(out, numthreads=1))
                    if back != data:
                        raise LazyRoomError(ErrorCode.OPERATION_FAILED,
                                              f"Reconstructed JPEG differs from {src.name}; nothing written")
                    res["verified"] = True
            else:
                try:
                    out = bytes(ic.jpegxl_decode_jpeg(data, numthreads=1))
                except Exception as exc:
                    raise LazyRoomError(ErrorCode.DECODE_FAILED,
                                          f"{src.name} has no JPEG reconstruction data: {exc}",
                                          hint="Only JXL files made by lossless JPEG transcoding can be turned back "
                                               "into the original JPEG; use format_convert(format='jpg') to re-encode.") \
                        from exc
            atomic_write(dst, out, overwrite=overwrite)
            res.update(output=str(dst), output_bytes=len(out), status="ok",
                       saved_percent=round(100 * (1 - len(out) / max(1, len(data))), 2))
            results.append(res)
        except LazyRoomError as e:
            results.append({"input": f, "status": "failed", "error": e.to_dict()})
    return {"results": results, "succeeded": sum(r["status"] == "ok" for r in results),
            "failed": sum(r["status"] != "ok" for r in results), "mode": mode}


def _flexible_center_crop(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """ImageToolbox createJxlAnimation frame fitting: ResizeType.Flexible (longer side -> max(w, h)), then
    ResizeType.CenterCrop on a transparent w x h canvas (no further scaling)."""
    fw, fh = codecs.telegram_sticker_size(img.shape[1], img.shape[0], max(w, h))
    rgba = img if has_alpha(img) else np.dstack([img[..., :3], np.ones(img.shape[:2], np.float32)])
    pil = Image.fromarray((np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")
    if (fw, fh) != pil.size:
        pil = pil.resize((fw, fh), Image.Resampling.LANCZOS)
    if (fw, fh) == (w, h):
        return np.asarray(pil, dtype=np.float32) / 255.0
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.paste(pil, ((w - fw) // 2, (h - fh) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


@tool("jxl_from_frames", "Images / animation to animated JPEG XL", category="animation", params=[
    StringList("inputs", [], min_items=1, file_kind="image",
               description="Still images (one frame each, in order) or a single animated GIF/APNG/WebP"),
    Text("output", "", description="Output .jxl path"),
    Int("frame_ms", 1000, min=1, max=60000,
        description="Frame duration for still images (ImageToolbox default 1000 ms); an animated source keeps "
                    "its own durations unless override_durations is set"),
    Bool("override_durations", False, description="Use frame_ms for every frame of an animated source too"),
    Int("quality", 50, min=1, max=100, description="Lossy quality (ImageToolbox default 50)"),
    Bool("lossless", False, description="Mathematically lossless (ImageToolbox: 'lossy' switch off)"),
    Int("effort", 2, min=1, max=10, description="libjxl effort (ImageToolbox default 2)"),
    Bool("overwrite", False),
], source=JXL_SOURCE)
def jxl_from_frames(api, inputs, output, frame_ms, override_durations, quality, lossless, effort, overwrite):
    """ImageToolbox "Images to JXL" / "GIF|APNG|WEBP to JXL": encode frames as an animated JPEG XL.

    Stills are fitted to the first image's size like ImageToolbox (longer side scaled to the canvas, centred,
    cropped/padded with transparency).  Loops forever (num_loops = 0)."""
    _require_jxl()
    files = expand_inputs(api, inputs)
    dst = out_path(api, output, overwrite=overwrite)
    if dst.suffix.lower() != ".jxl":
        raise invalid("output must end with .jxl")
    frames, durations = [], []
    if len(files) == 1:
        src = api.ws.policy.check_read(files[0])
        try:
            with Image.open(src) as im:
                if getattr(im, "n_frames", 1) > 1:
                    for fr in ImageSequence.Iterator(im):
                        frames.append(np.asarray(fr.convert("RGBA"), dtype=np.float32) / 255.0)
                        durations.append(int(fr.info.get("duration", im.info.get("duration", 100)) or 100))
        except LazyRoomError:
            raise
        except Exception:
            frames = []
    animated_source = bool(frames)
    if animated_source:
        if override_durations:
            durations = [frame_ms] * len(frames)
    else:
        size = None
        for f in files:
            img, _ = read_image(api, f)
            if size is None:
                size = (img.shape[1], img.shape[0])
            frames.append(_flexible_center_crop(img, *size))
        durations = [frame_ms] * len(frames)
    durations = [max(10, d) for d in durations]  # MIN_FRAME_DELAY_MILLIS
    data, info = codecs.encode_jxl_animation(frames, durations, quality=quality, lossless=lossless, effort=effort)
    atomic_write(dst, data, overwrite=overwrite)
    return {"output": str(dst), "bytes": len(data), "frames": len(frames), "durations_ms": durations,
            "animated_source": animated_source, "width": int(frames[0].shape[1]), "height": int(frames[0].shape[0]),
            **info}


@tool("jxl_extract_frames", "Extract frames from (animated) JPEG XL", category="animation", params=[
    File("input", "", file_kind="image", required=True, description=".jxl file"),
    Text("output_dir", ""),
    Choice("format", "png", choices=["png", "jpg", "webp"]),
    Bool("overwrite", False),
], source=JXL_SOURCE)
def jxl_extract_frames(api, input, output_dir, format, overwrite):
    """ImageToolbox "JXL to Images": save every frame of a JPEG XL animation as an image file.

    Per-frame durations are not exposed by the imagecodecs backend; the header tick is reported instead."""
    ic = _require_jxl()
    from ..core.image import normalize_channels, opaque_if_full
    from ..core.io import jxl_header

    src = api.ws.policy.check_read(input)
    data = src.read_bytes()
    try:
        info = jxl_header(data)
    except (ValueError, IndexError) as exc:
        raise LazyRoomError(ErrorCode.DECODE_FAILED, f"Cannot read the JPEG XL header of {src.name}: {exc}") from exc
    limit = api.ws.config.max_pixels
    if limit and info["width"] * info["height"] > limit:
        raise LazyRoomError(ErrorCode.RESOURCE_LIMIT,
                              f"Image has {info['width'] * info['height']} pixels per frame; limit is {limit}")
    try:
        single = ic.jpegxl_decode(data, index=0)
        arr = ic.jpegxl_decode(data) if info["animated"] else single[None]
    except Exception as exc:
        raise LazyRoomError(ErrorCode.DECODE_FAILED, f"Cannot decode {src.name}: {exc}") from exc
    if arr.ndim == single.ndim:
        arr = arr[None]
    od = api.ws.policy.check_write_dir(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    outs = [save(api, opaque_if_full(normalize_channels(fr)), str(od / f"{src.stem}_{i + 1:04d}.{format}"),
                 overwrite=overwrite)["path"] for i, fr in enumerate(arr)]
    res = {"frames": outs, "count": len(outs), "animated": info["animated"]}
    if info["animated"]:
        res["tick_ms"] = round(1000.0 * info["tps_denominator"] / info["tps_numerator"], 3)
        res["loops"] = info["num_loops"]
        res["note"] = "Per-frame durations are not exposed by the decoder; files written by jxl_from_frames use 1 tick per stored frame."
    return res


@tool("telegram_sticker", "Telegram sticker (512 px PNG)", category="convert", params=[
    StringList("inputs", [], min_items=1, file_kind="image"), Text("output_dir", ""),
    Bool("recursive", False), Bool("overwrite", False),
], source={"project": "ImageToolbox", "feature": "Resize and Convert: Telegram preset / telegram_sticker export profile",
           "license": "Apache-2.0"})
def telegram_sticker(api, inputs, output_dir, recursive, overwrite):
    """ImageToolbox Telegram preset: longer side scaled to exactly 512 px (Lanczos3), PNG lossless.

    Telegram also limits static stickers to 512 KB; files above that are flagged (ImageToolbox does not check)."""
    files = expand_inputs(api, inputs, recursive)
    od = api.ws.policy.check_write_dir(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        try:
            img, _ = read_image(api, f)
            out = codecs.telegram_sticker(img)
            dst = api.ws.policy.check_write(od / (Path(f).stem + ".png"), overwrite=overwrite)
            data = encode_image(out, ".png", metadata="strip")
            atomic_write(dst, data, overwrite=overwrite)
            r = {"input": f, "output": str(dst), "status": "ok", "width": int(out.shape[1]),
                 "height": int(out.shape[0]), "bytes": len(data), "within_512kb": len(data) <= 512 * 1024}
            if not r["within_512kb"]:
                r["warning"] = "Telegram rejects static stickers above 512 KB; try format_convert(format='png_lossy')."
            results.append(r)
        except LazyRoomError as e:
            results.append({"input": f, "status": "failed", "error": e.to_dict()})
    return {"results": results, "succeeded": sum(r["status"] == "ok" for r in results),
            "failed": sum(r["status"] != "ok" for r in results)}
