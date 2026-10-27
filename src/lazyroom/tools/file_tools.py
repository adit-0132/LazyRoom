"""File-oriented tools ported from ImageToolbox: cipher (file encryption), ZIP archives,
audio cover extraction, code preview images and a classical document scanner.

See docs/dev/port_reports/file_tools.md for the feature mapping and deviations.
Passwords are tool parameters only: they are never echoed in results or error details
(the operation log records ``tools.run`` arguments as ``<dict[N]>``).
"""
from __future__ import annotations

import glob as _glob
import io
import math
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..core.errors import ErrorCode, LazyRoomError, invalid
from ..core.io import atomic_write
from ..core.params import Bool, Choice, Color, File, Float, Int, Points, StringList, Text, parse_color
from . import _cipher
from . import _code_preview as _cp
from ._code_themes import THEMES as _THEMES
from .registry import out_path, read_image, save, tool

MAX_CIPHER_BYTES = 2 * 1024 ** 3  # whole-file in memory, like ImageToolbox


def _src(feature: str) -> dict[str, str]:
    return {"project": "ImageToolbox", "feature": feature}


def _read_path(api: Any, path: str) -> Path:
    if not path:
        raise invalid("An input path is required")
    p = api.ws.policy.check_read(path)
    if not p.is_file():
        raise LazyRoomError(ErrorCode.FILE_NOT_FOUND, f"File not found: {path}")
    return p


def _check_out(api: Any, path: Path | str, overwrite: bool, source: Path | None = None) -> Path:
    p = api.ws.policy.check_write(path, overwrite=overwrite)
    if source is not None and p == source:
        raise invalid("The output must be a different file than the input", hint="Choose another output path.")
    return p


def _write_bytes(api: Any, path: Path | str, data: bytes, overwrite: bool, source: Path | None = None) -> Path:
    p = _check_out(api, path, overwrite, source)
    atomic_write(p, data, overwrite=overwrite)
    return p


# ---------------------------------------------------------------------------
# Cipher
# ---------------------------------------------------------------------------

def _cipher_input(api: Any, input: str) -> tuple[Path, bytes]:
    p = _read_path(api, input)
    if p.stat().st_size > MAX_CIPHER_BYTES:
        raise LazyRoomError(ErrorCode.RESOURCE_LIMIT, "File is too large to encrypt in memory (limit 2 GiB)")
    return p, p.read_bytes()


def _need_password(password: str) -> None:
    if not password:
        raise invalid("A password (key) is required", param="password",
                      hint="Use generate_password to create a random one.")


@tool("encrypt_file", "Encrypt file (Cipher)", category="file", params=[
    File("input", "", file_kind="any", required=True, description="Any file"),
    Text("password", "", max_length=4096, description="Password / key (never logged or echoed)"),
    Choice("algorithm", _cipher.DEFAULT_ALGORITHM, choices=_cipher.SUPPORTED_NAMES,
           description="ImageToolbox cipher (see cipher_algorithms)"),
    Text("output", "", description="Output path (default: 'enc-<name>' next to the input)"),
    Bool("overwrite", False),
], source=_src("Cipher"))
def encrypt_file(api, input, password, algorithm, output, overwrite):
    """Encrypt a file in ImageToolbox's versioned format (PBKDF2-HMAC-SHA256, 210,000 iterations, random salt
    and nonce; AES-256-GCM by default).  Files decrypt in ImageToolbox and with decrypt_file."""
    c = _cipher
    _need_password(password)
    src, data = _cipher_input(api, input)
    p = _check_out(api, Path(output) if output else src.with_name("enc-" + src.name), overwrite, src)
    blob = c.encrypt(data, password, algorithm)
    atomic_write(p, blob, overwrite=overwrite)
    d = c.lookup(algorithm)
    res: dict[str, Any] = {"output": str(p), "bytes": len(blob), "algorithm": d.name, "cipher": d.cipher,
                           "key_bits": c.key_size_bits(d.name), "iterations": c.PBKDF2_ITERATIONS,
                           "authenticated": c.is_authenticated(d.name)}
    if c.has_magic(data):
        res["warning"] = "This file is already encrypted; encrypting it again adds an additional encryption layer."
    if not c.is_authenticated(d.name):
        res["note"] = ("This algorithm has no integrity check: a wrong password may decrypt to garbage instead of "
                       "failing.  AES/GCM/NoPadding is authenticated.")
    return res


@tool("decrypt_file", "Decrypt file (Cipher)", category="file", params=[
    File("input", "", file_kind="any", required=True, description="File encrypted by ImageToolbox or encrypt_file"),
    Text("password", "", max_length=4096, description="Password / key (never logged or echoed)"),
    Choice("algorithm", _cipher.DEFAULT_ALGORITHM, choices=_cipher.SUPPORTED_NAMES,
           description="Only used for legacy files without the ITBXCRYP header; versioned files name their "
                       "algorithm in the header"),
    Text("output", "", description="Output path (default: 'dec-<name>' next to the input)"),
    Bool("overwrite", False),
], source=_src("Cipher"))
def decrypt_file(api, input, password, algorithm, output, overwrite):
    """Decrypt an ImageToolbox-encrypted file (versioned ITBXCRYP format, or legacy headerless files with the
    chosen algorithm).  A wrong password or corrupted data gives a DECODE_FAILED error."""
    c = _cipher
    _need_password(password)
    src, data = _cipher_input(api, input)
    p = _check_out(api, Path(output) if output else src.with_name("dec-" + src.name), overwrite, src)
    plain, info = c.decrypt(data, password, algorithm)
    atomic_write(p, plain, overwrite=overwrite)
    res = {"output": str(p), "bytes": len(plain), **info}
    if not info["authenticated"]:
        res["note"] = ("No integrity check for this algorithm: verify the output, a wrong password can produce "
                       "garbage" + ("" if info.get("padding_checked") else " without any error") + ".")
    return res


@tool("cipher_algorithms", "Cipher algorithms", category="file", params=[
    Bool("include_unsupported", True, description="Also list ImageToolbox algorithms LazyRoom cannot run"),
], source=_src("Cipher"))
def cipher_algorithms(api, include_unsupported):
    """ImageToolbox's cipher list (BouncyCastle + Conscrypt, after its BROKEN filter) with what LazyRoom
    supports through the optional 'cryptography' package and why the rest is unsupported."""
    c = _cipher
    try:
        rows = c.describe_algorithms(probe=True)
        backend = True
    except LazyRoomError as e:
        if e.code != ErrorCode.BACKEND_UNAVAILABLE:
            raise
        rows, backend = c.describe_algorithms(probe=False), False
    shown = rows if include_unsupported else [r for r in rows if r["supported"]]
    return {"algorithms": shown, "default": c.DEFAULT_ALGORITHM, "supported": sum(r["supported"] for r in rows),
            "total": len(rows), "backend_installed": backend,
            "format": {"magic": c.MAGIC.decode(), "version": c.VERSION, "kdf": "PBKDF2WithHmacSHA256",
                       "iterations": c.PBKDF2_ITERATIONS, "salt_bytes": c.SALT_SIZE,
                       "aad": "header (authenticated ciphers)"}}


@tool("generate_password", "Generate random password", category="file", params=[
    Int("length", 18, min=1, max=4096, description="ImageToolbox generates 18 characters"),
], source=_src("Cipher"))
def generate_password(api, length):
    """Random password from [0-9A-Za-z] using a CSPRNG (ImageToolbox's random key generator)."""
    c = _cipher
    return {"password": c.generate_random_string(length), "length": length, "alphabet": "0-9A-Za-z",
            "entropy_bits": round(length * math.log2(len(c.RANDOM_CHARS)), 1)}


# ---------------------------------------------------------------------------
# Archives (ZIP)
# ---------------------------------------------------------------------------

def _expand_files(api: Any, inputs: list[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(os.path.expanduser(item))
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            files.extend(sorted(f for f in it if f.is_file()))
        elif any(ch in item for ch in "*?["):
            files.extend(sorted(Path(f) for f in _glob.glob(os.path.expanduser(item), recursive=recursive)
                                if Path(f).is_file()))
        else:
            files.append(p)
    out, seen = [], set()
    for f in files:
        rp = api.ws.policy.check_read(f)
        if not rp.is_file():
            raise LazyRoomError(ErrorCode.FILE_NOT_FOUND, f"File not found: {f}")
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


@tool("archive_create", "Create ZIP archive", category="file", params=[
    StringList("inputs", [], min_items=1, description="Files, folders or glob patterns (stored flat, like "
                                                       "ImageToolbox; duplicate names become 'name (2).ext')"),
    Text("output", "", description=".zip path"),
    Choice("compression", "deflate", choices=["store", "deflate", "bzip2", "lzma", "xz", "zstd"]),
    Choice("level", "normal", choices=["fast", "normal", "maximum"], description="1 / 6 / 9 (Deflate, BZIP2)"),
    Text("password", "", max_length=4096, description="ImageToolbox writes AES-256 ZIPs: not implemented here"),
    Bool("recursive", False, description="Include files in sub-folders of folder inputs"),
    Bool("overwrite", False),
], source=_src("Archive tools"))
def archive_create(api, inputs, output, compression, level, password, recursive, overwrite):
    """Pack files into a ZIP archive (Store / Deflate / BZIP2 / LZMA; compression level Fast / Normal / Maximum)."""
    from . import _archive as A

    if password:
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE,
                              "Password-protected archive creation is not implemented: ImageToolbox writes WinZip "
                              "AES-256 ZIPs and no permissive AES-ZIP library is installed",
                              hint="Create the archive without a password, then protect it with encrypt_file.")
    method = A.compression_methods()[compression]
    if method is None:
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE,
                              f"{compression.upper()} compression inside ZIP is not supported by this Python's zipfile",
                              hint="Use deflate, bzip2, lzma or store.")
    if not output:
        raise invalid("An output .zip path is required")
    files = _expand_files(api, inputs, recursive)
    if not files:
        raise LazyRoomError(ErrorCode.FILE_NOT_FOUND, "No files selected", details={"inputs": inputs})
    dst = api.ws.policy.check_write(output, overwrite=overwrite)
    if dst in files:
        raise invalid("The output archive cannot also be an input")
    used: set[str] = set()
    sources = [(A.unique_name(f.name, used), f) for f in files]
    lvl = None if compression in ("store", "lzma") else A.LEVELS[level]
    size = A.write_zip(dst, sources, method, lvl, overwrite)
    return {"output": str(dst), "bytes": size, "entries": [n for n, _ in sources], "compression": compression,
            "level": level if lvl is not None else None}


def _limits(max_entries: int, max_entry_mb: float, max_total_mb: float, max_ratio: float):
    from . import _archive as A

    mb = 1024 * 1024
    return A.Limits(max_entries=max_entries, max_entry_bytes=int(max_entry_mb * mb),
                    max_total_bytes=int(max_total_mb * mb), max_ratio=max_ratio)


@tool("archive_list", "List ZIP archive", category="file", params=[
    File("input", "", file_kind="any", required=True),
    Text("password", "", max_length=4096, description="Optional: verify this password (never echoed)"),
    Int("max_entries", 100_000, min=1, max=1_000_000),
], source=_src("Archive tools"))
def archive_list(api, input, password, max_entries):
    """Entries (normalised safe paths, sizes, encryption) and the archive's encryption status
    (none / password_required / unsupported = WinZip AES)."""
    from . import _archive as A

    p = _read_path(api, input)
    with A.open_zip(p) as zf:
        entries, unsafe = A.list_entries(zf, A.Limits(max_entries=max_entries))
        status = A.encryption_status(zf)
        res: dict[str, Any] = {"input": str(p), "entries": entries, "count": len(entries),
                               "total_size": sum(e["size"] for e in entries), "encryption": status}
        if unsafe:
            res["skipped_entries"] = unsafe
        if password and status == "password_required":
            res["password_ok"] = A.verify_password(zf, _cipher.password_bytes(password))
    return res


@tool("archive_extract", "Extract ZIP archive", category="file", params=[
    File("input", "", file_kind="any", required=True, description="ZIP-based archive (.zip .cbz .jar .apk .epub ...)"),
    Text("output_dir", "", description="Destination folder (default: the archive's folder)"),
    Text("password", "", max_length=4096, description="For ZipCrypto-encrypted entries (never echoed)"),
    Bool("create_subfolder", True, description="Extract into a new '<archive name>' folder"),
    Bool("preserve_structure", True, description="Keep folders (off = all files flat)"),
    Bool("skip_hidden", False, description="Skip entries with a path segment starting with '.'"),
    StringList("entries", [], description="Only these entry paths (as listed by archive_list); empty = all"),
    Int("max_entries", 100_000, min=1, max=1_000_000),
    Float("max_entry_mb", 4096, min=1, max=65536, description="Per-file limit on decompressed bytes"),
    Float("max_total_mb", 8192, min=1, max=131072, description="Limit on all decompressed bytes"),
    Float("max_ratio", 1000, min=0, max=1_000_000,
          description="Abort on entries (> 1 MiB) whose decompressed/compressed ratio exceeds this (0 = off)"),
], source=_src("Archive tools"))
def archive_extract(api, input, output_dir, password, create_subfolder, preserve_structure, skip_hidden, entries,
                    max_entries, max_entry_mb, max_total_mb, max_ratio):
    """Extract a ZIP archive safely: rejects zip-slip paths, skips symlinks, enforces size / count / ratio limits
    on the actually decompressed data, never overwrites (name collisions become 'name (2).ext'), and removes
    everything it created if extraction fails."""
    from . import _archive as A

    p = _read_path(api, input)
    dest = Path(output_dir) if output_dir else p.parent
    dest = api.ws.policy.check_write_dir(dest)
    if dest.exists() and not dest.is_dir():
        raise invalid(f"Destination is not a folder: {dest}")
    limits = _limits(max_entries, max_entry_mb, max_total_mb, max_ratio)
    created: list[Path] = []
    with A.open_zip(p) as zf:
        try:
            if not dest.exists():
                dest.mkdir(parents=True)
                created.append(dest)
            root = dest
            if create_subfolder:
                root = A.unique_path(dest, A.archive_name(p.name), is_dir=True)
                root = api.ws.policy.check_write_dir(root)
                root.mkdir()
                created.append(root)
            res = A.extract(zf, root, password=_cipher.password_bytes(password) if password else None,
                            preserve=preserve_structure, skip_hidden=skip_hidden,
                            selected=set(entries) if entries else None, limits=limits,
                            check_write=lambda t: api.ws.policy.check_write(t, overwrite=False), created=created)
        except BaseException:
            A.rollback(created)
            raise
    return {"input": str(p), "output_dir": str(root), "extracted": len(res["files"]), "files": res["files"],
            "bytes": res["bytes"], "skipped": res["skipped"]}


# ---------------------------------------------------------------------------
# Audio cover extractor
# ---------------------------------------------------------------------------
_MIME_EXT = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/gif": ".gif",
             "image/webp": ".webp", "image/bmp": ".bmp"}


def _decode_picture(api: Any, data: bytes) -> np.ndarray:
    old = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = api.ws.config.max_pixels
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)
            alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
            arr = np.asarray(im.convert("RGBA" if alpha else "RGB"), np.float32) / 255.0
    except (OSError, Image.DecompressionBombError) as exc:
        raise LazyRoomError(ErrorCode.DECODE_FAILED, f"The embedded picture cannot be decoded: {exc}") from None
    finally:
        Image.MAX_IMAGE_PIXELS = old
    return arr


def _video_frame(path: Path) -> np.ndarray | None:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    finally:
        cap.release()


@tool("audio_cover_extract", "Extract audio cover art", category="file", params=[
    File("input", "", file_kind="any", required=True, description="MP3, FLAC, M4A/MP4, Ogg Vorbis / Opus"),
    Text("output", "", description="Output image (default: '<audio name>.<format>' next to the input)"),
    Choice("format", "png", choices=["png", "jpg", "webp", "original"],
           description="original = the embedded bytes unchanged"),
    Int("index", -1, min=-1, max=1000, description="Picture to extract (-1 = front cover, else the first)"),
    Bool("video_frame_fallback", True, description="No embedded picture: use the first video frame, if any"),
    Int("quality", 92, min=1, max=100), Bool("overwrite", False),
], source=_src("Audio Cover Extractor"))
def audio_cover_extract(api, input, output, format, index, video_frame_fallback, quality, overwrite):
    """Extract the embedded cover art of an audio file (ID3v2.2-2.4 APIC/PIC, FLAC PICTURE, MP4 covr, Ogg
    METADATA_BLOCK_PICTURE) and save it (PNG by default, like ImageToolbox)."""
    from . import _audio_cover as AC

    p = _read_path(api, input)
    try:
        kind, pics = AC.read_pictures(p)
    except (AC.FormatError, ValueError, IndexError, struct.error) as exc:
        raise LazyRoomError(ErrorCode.DECODE_FAILED, f"Cannot parse {p.name}: {exc}") from None
    if index >= len(pics) and pics:
        raise invalid(f"index {index} out of range: the file has {len(pics)} picture(s)")
    pic = AC.choose(pics, None if index < 0 else index)
    res: dict[str, Any] = {"input": str(p), "container": kind, "pictures": [x.describe() for x in pics]}
    if pic is None:
        frame = _video_frame(p) if video_frame_fallback else None
        if frame is None:
            raise LazyRoomError(ErrorCode.OPERATION_FAILED, "No image: the file has no embedded cover art",
                                  details={"container": kind})
        fmt = "png" if format == "original" else format
        dst = output or str(p.with_name(p.stem + "." + fmt))
        res.update(source="video_frame", output=save(api, frame, dst, overwrite=overwrite, quality=quality))
        return res
    res["selected"] = pics.index(pic)
    if format == "original":
        mime = pic.mime if pic.mime in _MIME_EXT else AC.sniff_mime(pic.data)
        dst = output or str(p.with_name(p.stem + _MIME_EXT.get(mime, ".bin")))
        w = _write_bytes(api, dst, pic.data, overwrite, source=p)
        res.update(source="embedded", output={"path": str(w), "bytes": len(pic.data), "mime": mime})
        return res
    img = _decode_picture(api, pic.data)
    dst = output or str(p.with_name(p.stem + "." + format))
    res.update(source="embedded", output=save(api, img, dst, overwrite=overwrite, quality=quality))
    return res


# ---------------------------------------------------------------------------
# Code preview
# ---------------------------------------------------------------------------

@tool("code_preview", "Code preview image", category="generate", params=[
        Text("code", _cp.DEFAULT_CODE, max_length=2_000_000, multiline=True, description="Source code"),
        File("input", "", file_kind="any", description="Optional text file to read the code from (UTF-8)"),
        Choice("language", "kotlin", choices=_cp.LANGUAGE_KEYS, description="highlight.js language key"),
        Choice("theme", "base16/dracula", choices=sorted(_THEMES), description="highlight.js theme"),
        Choice("background", "aurora", choices=[*_cp.BACKGROUND_PRESETS, "custom"]),
        StringList("background_colors", [], max_items=16, description="background=custom: gradient colours"),
        Text("title", "Main.kt", max_length=500), Int("font_size", 15, min=10, max=30),
        Int("outer_padding", 28, min=0, max=80), Int("inner_padding", 20, min=8, max=48),
        Int("corner_radius", 18, min=0, max=36), Int("canvas_corner_radius", 24, min=0, max=64),
        Float("rotation", 0.0, min=-15, max=15, unit="deg"),
        Bool("card_shadow", True), Color("shadow_color", "#00000066"), Int("shadow_blur", 14, min=0, max=40),
        Int("shadow_offset_x", 0, min=-30, max=30), Int("shadow_offset_y", 5, min=-30, max=30),
        Bool("canvas_background", True), Bool("window_controls", True), Bool("line_numbers", True),
        Bool("show_title", True), Bool("wrap_long_lines", True),
        Text("output", "", description="Output image (.png recommended)"),
        Int("quality", 92, min=1, max=100), Bool("overwrite", False),
], source=_src("Code Preview"))
def code_preview(api, code, input, language, theme, background, background_colors, title, font_size, outer_padding,
                 inner_padding, corner_radius, canvas_corner_radius, rotation, card_shadow, shadow_color, shadow_blur,
                 shadow_offset_x, shadow_offset_y, canvas_background, window_controls, line_numbers, show_title,
                 wrap_long_lines, output, quality, overwrite):
    """Render source code as a syntax-highlighted window card on a gradient canvas (ImageToolbox Code Preview:
    256 highlight.js themes, window controls, line numbers, wrapping, shadow, rotation)."""
    dst = out_path(api, output, overwrite=overwrite)
    if input:
        p = _read_path(api, input)
        if p.stat().st_size > 2_000_000:
            raise LazyRoomError(ErrorCode.RESOURCE_LIMIT, "Code file is larger than 2 MB")
        code = p.read_bytes().decode("utf-8", errors="replace")
    if background == "custom":
        if not background_colors:
            raise invalid("background=custom needs background_colors", param="background_colors")
        colors = tuple(parse_color(c, "background_colors") for c in background_colors)
    else:
        colors = tuple(parse_color(c) for c in _cp.BACKGROUND_PRESETS[background])
    params = _cp.CodePreviewParams(
        code=code, language=language, theme=theme, background_colors=colors, title=title, font_size=font_size,
        outer_padding=outer_padding, inner_padding=inner_padding, corner_radius=corner_radius,
        canvas_corner_radius=canvas_corner_radius, rotation=rotation, show_card_shadow=card_shadow,
        card_shadow_color=tuple(shadow_color), card_shadow_blur_radius=shadow_blur,
        card_shadow_offset_x=shadow_offset_x, card_shadow_offset_y=shadow_offset_y,
        show_canvas_background=canvas_background, show_window_controls=window_controls,
        show_line_numbers=line_numbers, show_title=show_title, wrap_long_lines=wrap_long_lines)
    img, info = _cp.render(params)
    saved = save(api, img.astype(np.float32) / 255.0, str(dst), overwrite=overwrite, quality=quality)
    return {"output": saved, **info}


# ---------------------------------------------------------------------------
# Document scanner
# ---------------------------------------------------------------------------

def _to_u8_rgb(img: np.ndarray) -> np.ndarray:
    rgb = img[..., :3]
    if img.shape[2] == 4:
        a = img[..., 3:4]
        rgb = rgb * a + (1 - a)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


@tool("document_scan", "Document scanner", category="file", params=[
    StringList("inputs", [], min_items=1, file_kind="image", description="Photos of pages, in page order"),
    Text("output", "", description="Folder for images (default: next to each input) or the .pdf path"),
    Choice("format", "jpg", choices=["jpg", "png", "webp", "pdf"]),
    Choice("mode", "color", choices=["color", "grayscale", "bw", "shadow_removal"],
           description="color = rectified only; bw = adaptive threshold; shadow_removal = flattened lighting"),
    Points("corners", [], min_items=0, max_items=4, min=0, max=1,
           description="Page corners [[x, y] x4] in 0..1 image coordinates (single input only); empty = detect"),
    Int("bw_block", 0, min=0, max=501, description="bw: adaptive threshold block size in px (0 = auto)"),
    Int("bw_offset", 10, min=-50, max=50, description="bw: threshold offset C"),
    Int("quality", 85, min=1, max=100), Int("pdf_dpi", 300, min=36, max=1200),
    Bool("overwrite", False),
], source=_src("Document Scanner"))
def document_scan(api, inputs, output, format, mode, corners, bw_block, bw_offset, quality, pdf_dpi, overwrite):
    """Scan documents from photos: detect the page quadrilateral (or use given corners), correct perspective,
    enhance (colour / greyscale / black & white / shadow removal) and save images or one multi-page PDF.
    Classical OpenCV replacement for ImageToolbox's ML Kit scanner."""
    from . import _docscan as D
    from .registry import expand_inputs

    files = expand_inputs(api, inputs)
    if not files:
        raise LazyRoomError(ErrorCode.FILE_NOT_FOUND, "No input images matched", details={"inputs": inputs})
    if corners and (len(corners) != 4 or len(files) != 1):
        raise invalid("corners needs exactly 4 points and a single input image", param="corners")
    if format == "pdf":
        if not output:
            raise invalid("format=pdf needs an output .pdf path")
        pdf_dst = out_path(api, output, overwrite=overwrite)
    pages, results = [], []
    for f in files:
        img, _ = read_image(api, f)
        rgb = _to_u8_rgb(img)
        h, w = rgb.shape[:2]
        if corners:
            quad = np.array([[x * (w - 1), y * (h - 1)] for x, y in corners], np.float64)
            info: dict[str, Any] = {"detected": False, "method": "user"}
        else:
            quad, info = D.detect_page(rgb)
        quad = D.order_corners(quad)
        page = D.enhance(D.rectify(rgb, quad), mode, block=bw_block, offset=bw_offset)
        entry = {"input": str(f), "corners_px": np.round(quad, 2).tolist(),
                 "corners": np.round(quad / [max(w - 1, 1), max(h - 1, 1)], 5).tolist(),
                 "width": int(page.shape[1]), "height": int(page.shape[0]), **info}
        if not info["detected"] and info["method"] != "user":
            entry["warning"] = "No page outline found; the whole image was used."
        if format == "pdf":
            pages.append(Image.fromarray(page, "L" if page.ndim == 2 else "RGB"))
        else:
            folder = Path(output) if output else Path(f).parent
            dst = folder / f"{Path(f).stem}_scan.{format}"
            arr = page.astype(np.float32) / 255.0
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            entry["output"] = save(api, arr, str(dst), overwrite=overwrite, quality=quality)
        results.append(entry)
    res: dict[str, Any] = {"pages": results, "mode": mode, "format": format}
    if format == "pdf":
        buf = io.BytesIO()
        first, rest = pages[0], pages[1:]
        first.save(buf, format="PDF", save_all=True, append_images=rest, resolution=float(pdf_dpi),
                   quality=quality)
        atomic_write(pdf_dst, buf.getvalue(), overwrite=overwrite)
        res["output"] = {"path": str(pdf_dst), "pages": len(pages), "bytes": len(buf.getvalue())}
    return res
