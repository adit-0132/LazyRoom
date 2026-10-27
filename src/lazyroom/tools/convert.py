"""Conversion tools: animations, PDF, target-size compression, checksums, base64, renaming, ASCII, SVG tracing."""
from __future__ import annotations

import base64 as b64
import hashlib
import io
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageSequence

from ..core.errors import ErrorCode, LazyRoomError, invalid
from ..core.image import fit_within
from ..core.io import encode_image, format_for_path, load_image
from ..core.params import Bool, Choice, Color, File, Float, Int, StringList, Text
from .registry import expand_inputs, out_path, read_image, save, tool

F32 = np.float32


# ---------------------------------------------------------------------------
# Animations (GIF / APNG / WebP)
# ---------------------------------------------------------------------------

def _frames(path: Path) -> tuple[list[Image.Image], list[int], int]:
    with Image.open(path) as im:
        frames, durations = [], []
        for fr in ImageSequence.Iterator(im):
            frames.append(fr.convert("RGBA"))
            durations.append(int(fr.info.get("duration", im.info.get("duration", 100)) or 100))
        loop = int(im.info.get("loop", 0))
    return frames, durations, loop


def _save_anim(frames: list[Image.Image], durations: list[int], loop: int, dst: Path, quality: int) -> None:
    ext = dst.suffix.lower()
    if ext == ".gif":
        pal = [f.convert("RGBA").quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.FLOYDSTEINBERG)
               for f in frames]
        pal[0].save(dst, save_all=True, append_images=pal[1:], duration=durations, loop=loop, disposal=2,
                    optimize=False)
    elif ext == ".png":
        frames[0].save(dst, format="PNG", save_all=True, append_images=frames[1:], duration=durations, loop=loop,
                       default_image=False)
    elif ext == ".webp":
        frames[0].save(dst, save_all=True, append_images=frames[1:], duration=durations, loop=loop,
                       quality=quality, method=4)
    elif ext in (".avif",):
        frames[0].save(dst, save_all=True, append_images=frames[1:], duration=durations, loop=loop, quality=quality)
    else:
        raise invalid(f"Animated output must be .gif, .png (APNG), .webp or .avif, got {ext}")


@tool("animation_convert", "Convert / edit animation", category="animation", params=[
    File("input", "", file_kind="image", required=True, description="Animated GIF, APNG or WebP"),
    Text("output", "", description=".gif / .png (APNG) / .webp / .avif"),
    Float("speed", 1.0, min=0.05, max=20, description="Playback speed multiplier"),
    Int("max_size", 0, min=0, max=8000, description="Downscale long edge (0 = keep)"),
    Int("quality", 85, min=1, max=100), Text("steps_preset", "", description="Optional preset id applied to every frame"),
    Bool("overwrite", False),
], source={"project": "ImageToolbox", "feature": "GIF/APNG/WEBP/JXL tools"})
def animation_convert(api, input, output, speed, max_size, quality, steps_preset, overwrite):
    """Convert between animated formats, change speed/size, optionally grade every frame with a preset."""
    src = api.ws.policy.check_read(input)
    dst = out_path(api, output, overwrite=overwrite)
    frames, durations, loop = _frames(src)
    if steps_preset:
        from ..core.context import RenderContext
        from ..core.pipeline import render, steps_from_list

        steps = steps_from_list(api.presets.get(steps_preset)["steps"])
        new = []
        for i, f in enumerate(frames):
            arr = np.asarray(f, dtype=np.float32) / 255
            out, _ = render(arr, steps, RenderContext(seed=1))
            new.append(Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGBA" if out.shape[2] == 4 else "RGB").convert("RGBA"))
        frames = new
    if max_size:
        for i, f in enumerate(frames):
            if max(f.size) > max_size:
                s = max_size / max(f.size)
                frames[i] = f.resize((max(1, round(f.width * s)), max(1, round(f.height * s))), Image.Resampling.LANCZOS)
    durations = [max(10, int(round(d / speed))) for d in durations]
    _save_anim(frames, durations, loop, dst, quality)
    return {"output": str(dst), "frames": len(frames), "duration_ms": sum(durations)}


@tool("animation_extract_frames", "Extract animation frames", category="animation", params=[
    File("input", "", file_kind="image", required=True), Text("output_dir", ""),
    Choice("format", "png", choices=["png", "jpg", "webp"]), Bool("overwrite", False),
])
def animation_extract_frames(api, input, output_dir, format, overwrite):
    """Save every frame of a GIF/APNG/WebP animation as an image file."""
    src = api.ws.policy.check_read(input)
    od = api.ws.policy.check_write_dir(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    frames, durations, _ = _frames(src)
    outs = []
    for i, f in enumerate(frames):
        arr = np.asarray(f, dtype=np.float32) / 255
        outs.append(save(api, arr, str(od / f"{src.stem}_{i + 1:04d}.{format}"), overwrite=overwrite)["path"])
    return {"frames": outs, "durations_ms": durations}


@tool("animation_from_images", "Create animation from images", category="animation", params=[
    StringList("inputs", [], min_items=1, file_kind="image"), Text("output", ""),
    Int("frame_ms", 100, min=10, max=60000), Int("loop", 0, min=0, max=1000, description="0 = infinite"),
    Int("max_size", 1080, min=0, max=8000), Int("quality", 85, min=1, max=100), Bool("overwrite", False),
])
def animation_from_images(api, inputs, output, frame_ms, loop, max_size, quality, overwrite):
    """Assemble images into an animated GIF / APNG / WebP (frames are fitted to the first frame's size)."""
    files = expand_inputs(api, inputs)
    dst = out_path(api, output, overwrite=overwrite)
    frames = []
    size = None
    for f in files:
        img, _ = read_image(api, f)
        if max_size:
            img, _ = fit_within(img, max_size)
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGBA" if img.shape[2] == 4 else "RGB").convert("RGBA")
        if size is None:
            size = pil.size
        elif pil.size != size:
            canvas = Image.new("RGBA", size, (0, 0, 0, 0))
            pil.thumbnail(size, Image.Resampling.LANCZOS)
            canvas.paste(pil, ((size[0] - pil.width) // 2, (size[1] - pil.height) // 2))
            pil = canvas
        frames.append(pil)
    _save_anim(frames, [frame_ms] * len(frames), loop, dst, quality)
    return {"output": str(dst), "frames": len(frames)}


# ---------------------------------------------------------------------------
@tool("images_to_pdf", "Images to PDF", category="convert", params=[
    StringList("inputs", [], min_items=1, file_kind="image"), Text("output", ""),
    Choice("page_size", "fit", choices=["fit", "a4", "letter"]), Int("dpi", 300, min=72, max=1200),
    Int("quality", 90, min=1, max=100), Bool("overwrite", False),
], source={"project": "ImageToolbox", "feature": "PDF tools"})
def images_to_pdf(api, inputs, output, page_size, dpi, quality, overwrite):
    """Combine images into a multi-page PDF (one image per page)."""
    files = expand_inputs(api, inputs)
    dst = out_path(api, output, overwrite=overwrite)
    pages = []
    for f in files:
        img, _ = read_image(api, f)
        from ..core.image import composite_on

        rgb = composite_on(img) if img.shape[2] == 4 else img
        pil = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")
        if page_size != "fit":
            pw, ph = (8.27, 11.69) if page_size == "a4" else (8.5, 11.0)
            if pil.width > pil.height:
                pw, ph = ph, pw
            W, H = int(pw * dpi), int(ph * dpi)
            pil.thumbnail((int(W * 0.94), int(H * 0.94)), Image.Resampling.LANCZOS)
            page = Image.new("RGB", (W, H), "white")
            page.paste(pil, ((W - pil.width) // 2, (H - pil.height) // 2))
            pil = page
        pages.append(pil)
    pages[0].save(dst, format="PDF", save_all=True, append_images=pages[1:], resolution=dpi, quality=quality)
    return {"output": str(dst), "pages": len(pages)}


@tool("compress_to_size", "Compress to target file size", category="convert", params=[
    File("input", "", file_kind="image", required=True), Text("output", ""),
    Int("target_kb", 500, min=5, max=200000), Choice("format", "jpg", choices=["jpg", "webp", "avif"]),
    Bool("allow_resize", True, description="Downscale when the quality floor cannot reach the target"),
    Bool("keep_metadata", False), Bool("overwrite", False),
], source={"project": "ImageToolbox", "feature": "Image Shrinking / Limits Resize"})
def compress_to_size(api, input, output, target_kb, format, allow_resize, keep_metadata, overwrite):
    """Binary-search the encoder quality (and size if needed) to fit under a file-size limit."""
    img, meta = read_image(api, input)
    dst = out_path(api, output, overwrite=overwrite)
    target = target_kb * 1024
    ext = "." + format
    md = "keep" if keep_metadata else "strip"
    cur = img
    for _ in range(8):
        lo, hi, best = 5, 98, None
        while lo <= hi:
            q = (lo + hi) // 2
            data = encode_image(cur, ext, quality=q, meta=meta, metadata=md)
            if len(data) <= target:
                best = (q, data)
                lo = q + 1
            else:
                hi = q - 1
        if best and best[0] >= 35 or (best and not allow_resize):
            break
        if not allow_resize:
            break
        cur, _ = fit_within(cur, int(max(cur.shape[:2]) * 0.8))
    if not best:
        raise LazyRoomError(ErrorCode.OPERATION_FAILED, f"Cannot reach {target_kb} KB",
                              hint="Allow resizing or raise the target size.")
    from ..core.io import atomic_write

    atomic_write(dst, best[1], overwrite=overwrite)
    return {"output": str(dst), "bytes": len(best[1]), "quality": best[0], "width": int(cur.shape[1]),
            "height": int(cur.shape[0])}


@tool("checksum", "File checksums", category="convert", params=[
    StringList("inputs", [], min_items=1), Choice("algorithm", "sha256", choices=["md5", "sha1", "sha256", "sha512", "blake2b"]),
], source={"project": "ImageToolbox", "feature": "Checksum tools"})
def checksum(api, inputs, algorithm):
    """Compute file hashes (to verify integrity or find exact duplicates)."""
    out = []
    for f in inputs:
        p = api.ws.policy.check_read(f)
        h = hashlib.new(algorithm)
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out.append({"path": str(p), algorithm: h.hexdigest(), "bytes": p.stat().st_size})
    return {"results": out}


@tool("base64", "Base64 encode / decode", category="convert", params=[
    Choice("mode", "encode", choices=["encode", "decode"]),
    File("input", "", file_kind="any", description="encode: image file to encode"),
    Text("data", "", max_length=50_000_000, description="decode: base64 text (optionally a data: URI)"),
    Text("output", "", description="decode: where to write the image"), Bool("overwrite", False),
], source={"project": "ImageToolbox", "feature": "Base64 tools"})
def base64(api, mode, input, data, output, overwrite):
    """Encode an image file as base64 (data URI) or decode base64 back into a validated image file."""
    if mode == "encode":
        p = api.ws.policy.check_read(input)
        raw = p.read_bytes()
        if len(raw) > 30 * 1024 * 1024:
            raise invalid("File too large to base64-encode inline (> 30 MB)")
        mime = Image.MIME.get((Image.open(p).format or "").upper(), "application/octet-stream")
        return {"data_uri": f"data:{mime};base64," + b64.b64encode(raw).decode(), "bytes": len(raw)}
    s = data.split(",", 1)[1] if data.startswith("data:") else data
    try:
        raw = b64.b64decode(s, validate=False)
        Image.open(io.BytesIO(raw)).verify()
    except Exception as exc:
        raise LazyRoomError(ErrorCode.DECODE_FAILED, f"Not a valid base64 image: {exc}") from exc
    from ..core.io import atomic_write

    dst = out_path(api, output, overwrite=overwrite)
    atomic_write(dst, raw, overwrite=overwrite)
    return {"output": str(dst), "bytes": len(raw)}


@tool("batch_rename", "Batch rename (copy or move)", category="convert", destructive=True, params=[
    StringList("inputs", [], min_items=1), Text("template", "{date}_{index:03d}{ext}",
                                                 description="Tokens: {stem} {ext} {index} {parent} {date} {exif_date} {width} {height}"),
    Choice("mode", "copy", choices=["copy", "move"]), Text("output_dir", "", description="Required for copy"),
    Int("start", 1, min=0, max=1_000_000), Bool("dry_run", True, description="Only show the plan"),
    Bool("confirm", False, description="Required (moving renames originals)"),
], source={"project": "ImageToolbox", "feature": "Batch Rename"})
def batch_rename(api, inputs, template, mode, output_dir, start, dry_run, confirm):
    """Rename many files with a template; dry-run by default; never overwrites existing files."""
    import time as _t

    files = expand_inputs(api, inputs)
    plan = []
    names = set()
    for i, f in enumerate(files):
        p = Path(f)
        try:
            with Image.open(p) as im:
                w, h = im.size
                dt = im.getexif().get_ifd(0x8769).get(0x9003) or im.getexif().get(0x0132) or ""
        except Exception:
            w = h = 0
            dt = ""
        exif_date = str(dt).replace(":", "").replace(" ", "_")[:15] if dt else ""
        try:
            name = template.format(stem=p.stem, ext=p.suffix.lower(), index=start + i, parent=p.parent.name,
                                   date=_t.strftime("%Y%m%d", _t.localtime(p.stat().st_mtime)),
                                   exif_date=exif_date or _t.strftime("%Y%m%d_%H%M%S", _t.localtime(p.stat().st_mtime)),
                                   width=w, height=h)
        except (KeyError, ValueError, IndexError) as exc:
            raise invalid(f"Bad template: {exc}") from None
        if "/" in name or "\\" in name:
            raise invalid("Template must produce plain file names")
        folder = Path(output_dir) if (mode == "copy" and output_dir) else p.parent
        dst = folder / name
        if str(dst) in names:
            raise invalid(f"Template produces duplicate name {name}; include {{index}}")
        names.add(str(dst))
        plan.append((p, dst))
    if mode == "copy" and not output_dir:
        raise invalid("output_dir is required for copy mode")
    if dry_run:
        return {"dry_run": True, "plan": [{"from": str(a), "to": str(b)} for a, b in plan]}
    done = []
    for a, b in plan:
        api.ws.policy.check_read(a)
        api.ws.policy.check_write(b, overwrite=False)
        if mode == "move" and str(a) in {str(x) for x in api.ws.policy.protected}:
            raise LazyRoomError(ErrorCode.PATH_NOT_ALLOWED, f"{a} is the source of an open project")
        b.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copy2 if mode == "copy" else shutil.move)(str(a), str(b))
        done.append({"from": str(a), "to": str(b)})
    return {"renamed": done, "mode": mode}


ASCII_RAMP = " .:-=+*#%@"


@tool("ascii_art", "ASCII art", category="convert", params=[
    File("input", "", file_kind="image", required=True), Int("columns", 100, min=10, max=400),
    Text("charset", ASCII_RAMP, max_length=100), Bool("invert", False),
    Text("output", "", description="Optional .txt (text) or image path (renders the text)"), Bool("overwrite", False),
], source={"project": "ImageToolbox", "feature": "ASCII art"})
def ascii_art(api, input, columns, charset, invert, output, overwrite):
    """Convert an image to ASCII art text (optionally rendered to an image)."""
    img, _ = read_image(api, input)
    g = np.clip(img[..., :3], 0, 1) @ np.array([0.299, 0.587, 0.114], F32)
    h, w = g.shape
    rows = max(1, int(h / w * columns * 0.5))
    small = cv2.resize(g, (columns, rows), interpolation=cv2.INTER_AREA)
    if invert:
        small = 1 - small
    chars = charset or ASCII_RAMP
    idx = np.clip((small * (len(chars) - 1)).round().astype(int), 0, len(chars) - 1)
    text = "\n".join("".join(chars[i] for i in row) for row in idx)
    res: dict[str, Any] = {"text": text, "columns": columns, "rows": rows}
    if output:
        if output.lower().endswith(".txt"):
            from ..core.io import atomic_write

            atomic_write(out_path(api, output, overwrite=overwrite), text.encode(), overwrite=overwrite)
            res["output"] = output
        else:
            from PIL import ImageDraw, ImageFont

            f = ImageFont.load_default(size=14)
            cw, ch = 9, 16
            pil = Image.new("RGB", (columns * cw + 8, rows * ch + 8), "white")
            ImageDraw.Draw(pil).multiline_text((4, 4), text, font=f, fill="black", spacing=2)
            res["output"] = save(api, np.asarray(pil, dtype=F32) / 255, output, overwrite=overwrite)
    return res


@tool("svg_trace", "Trace raster to SVG", category="convert", params=[
    File("input", "", file_kind="image", required=True), Text("output", "", description=".svg path"),
    Int("colors", 8, min=2, max=64), Float("detail", 0.5, min=0.05, max=1.0, step=0.05,
                                           description="Higher keeps more small shapes / exact outlines"),
    Int("max_size", 1200, min=64, max=4000), Bool("overwrite", False),
], source={"project": "ImageToolbox", "feature": "SVG maker"})
def svg_trace(api, input, output, colors, detail, max_size, overwrite):
    """Vectorise an image: colour quantisation + contour tracing into filled SVG paths (layered, with holes)."""
    img, _ = read_image(api, input)
    small, _ = fit_within(np.clip(img[..., :3], 0, 1), max_size)
    small = cv2.bilateralFilter(small.astype(F32), 7, 0.08, 5)
    h, w = small.shape[:2]
    data = small.reshape(-1, 3).astype(F32)
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(data, colors, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1e-3),
                                    2, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h, w)
    min_area = (1 - detail) * 0.0005 * w * h + 2
    eps = (1.05 - detail) * 2.0
    counts = np.bincount(labels.ravel(), minlength=colors)
    order = np.argsort(-counts)  # paint large regions first
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">']
    bg = centers[order[0]]
    parts.append(f'<rect width="{w}" height="{h}" fill="#{"".join(f"{int(round(c * 255)):02x}" for c in bg)}"/>')
    paths = 0
    for k in order[1:]:
        mask = (labels == k).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        col = "#" + "".join(f"{int(round(c * 255)):02x}" for c in centers[k])
        d = []
        for i, c in enumerate(cnts):
            if cv2.contourArea(c) < min_area:
                continue
            c = cv2.approxPolyDP(c, eps, True)
            if len(c) < 3:
                continue
            d.append("M" + " L".join(f"{p[0][0]} {p[0][1]}" for p in c) + " Z")
        if d:
            parts.append(f'<path fill="{col}" fill-rule="evenodd" d="{" ".join(d)}"/>')
            paths += 1
    parts.append("</svg>")
    svg = "\n".join(parts)
    from ..core.io import atomic_write

    dst = out_path(api, output, overwrite=overwrite)
    atomic_write(dst, svg.encode(), overwrite=overwrite)
    return {"output": str(dst), "paths": paths, "bytes": len(svg), "size": [w, h]}


@tool("format_convert", "Convert image format", category="convert", params=[
    StringList("inputs", [], min_items=1, file_kind="image"), Text("output_dir", ""),
    Choice("format", "webp", choices=["jpg", "png", "webp", "avif", "heic", "tif", "bmp", "gif", "jp2", "qoi", "ico", "tga",
                                      "jxl", "png_lossy", "oxipng", "jpegli", "ico_multi"],
           description="ImageToolbox extras: jxl (JPEG XL, optional imagecodecs), png_lossy (Wu palette quantiser, "
                       "<= max_colors), oxipng (optional pyoxipng), jpegli (optional pyjpegli), ico_multi "
                       "(16/32/48/64/128/256 px in one .ico)"),
    Int("quality", 90, min=1, max=100), Choice("metadata", "keep", choices=["keep", "keep_no_gps", "strip"]),
    Bool("recursive", False), Bool("overwrite", False),
    Bool("lossless", False, description="Lossless mode for webp / avif / jp2 / jxl"),
    Int("effort", -1, min=-1, max=10,
        description="-1 = default. jxl: libjxl effort 1..10 (default 7; ImageToolbox uses 2); png_lossy: zlib "
                    "level 0..9 (default 7); oxipng: preset level 0..6 (default 6)"),
    Int("max_colors", 512, min=2, max=1024, description="png_lossy: palette size (ImageToolbox default 512)"),
], source={"project": "ImageToolbox", "feature": "Format Conversion"})
def format_convert(api, inputs, output_dir, format, quality, metadata, recursive, overwrite, lossless=False,
                   effort=-1, max_colors=512):
    """Convert images to another format (synchronously; use batch_start for very large sets)."""
    from . import _itb_codecs as itbc

    files = expand_inputs(api, inputs, recursive)
    od = api.ws.policy.check_write_dir(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    special = format in itbc.SPECIAL_FORMATS
    ext = itbc.check_special(format, effort) if special else format_for_path("x", format)[0]
    if ext == ".jxl" and effort == 0:
        raise invalid("JPEG XL effort must be within 1..10 (or -1 for the default)", param="effort")
    results = []
    for f in files:
        try:
            img, meta = read_image(api, f)
            dst = api.ws.policy.check_write(od / (Path(f).stem + ext), overwrite=overwrite)
            if special:
                data = itbc.encode_special(format, img, quality=quality, effort=effort, max_colors=max_colors,
                                           meta=meta, metadata=metadata)
            else:
                data = encode_image(img, ext, quality=quality, meta=meta, metadata=metadata, lossless=lossless,
                                    extra={"effort": effort} if ext == ".jxl" and effort > 0 else None)
            from ..core.io import atomic_write

            atomic_write(dst, data, overwrite=overwrite)
            results.append({"input": f, "output": str(dst), "status": "ok", "bytes": len(data)})
        except LazyRoomError as e:
            results.append({"input": f, "status": "failed", "error": e.to_dict()})
    return {"results": results, "succeeded": sum(r["status"] == "ok" for r in results),
            "failed": sum(r["status"] != "ok" for r in results)}
