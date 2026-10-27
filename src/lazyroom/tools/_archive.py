"""ZIP archive helpers (port of ImageToolbox ``feature/archive-tools`` + ``lib/archive``).

ImageToolbox drives libarchive; Python's ``zipfile`` covers the ZIP subset:
Store / Deflate / BZIP2 / LZMA (and Zstandard when the interpreter's zipfile has it),
ZipCrypto extraction.  WinZip AES (what ImageToolbox writes when a password is set)
needs a library that is not installed, so it is reported as not implemented.

Safety: entry paths go through ``ArchivePath.safeSegments`` (no absolute paths, drive
letters, ``.``/``..`` segments, NUL, <= 4096 chars, <= 256 segments, segments truncated
to 240 chars) and every output path is re-checked to stay inside the destination folder
and against the workspace path policy.  Symlinks and other special entries are skipped.
Zip bombs are stopped by entry-count, per-entry and total limits enforced on the bytes
actually decompressed (not only the declared sizes) plus a compression-ratio limit.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..core.errors import ErrorCode, LazyRoomError

MAX_PATH_LENGTH = 4096
MAX_SEGMENTS = 256
MAX_SEGMENT_LENGTH = 240
FILE_PERMISSIONS = 0o644  # ArchiveEngine.FILE_PERMISSIONS = 420
COPY_BUFFER = 64 * 1024
LEVELS = {"fast": 1, "normal": 6, "maximum": 9}  # ArchiveCompressionLevel
_DRIVE = re.compile(r"^[A-Za-z]:/")

SUPPORTED_ARCHIVE_EXTENSIONS = [
    ".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst", ".tar.zstd", ".tar.z", ".tar.lz4", ".tar.lz", ".tar.lzip",
    ".tar.lzma", ".tar.br", ".warc.gz", ".tgz", ".tbz", ".tbz2", ".txz", ".tzst", ".taz", ".tlz", ".tbr", ".zip",
    ".zipx", ".7z", ".rar", ".cbr", ".cbz", ".cb7", ".cbt", ".apk", ".apks", ".xapk", ".aab", ".jar", ".aar", ".ipa",
    ".epub", ".whl", ".egg", ".nupkg", ".vsix", ".tar", ".gz", ".bz2", ".xz", ".zst", ".zstd", ".z", ".lz4", ".lz",
    ".lzip", ".lzma", ".br", ".uu", ".uue", ".cab", ".iso", ".lha", ".lzh", ".xar", ".ar", ".a", ".deb", ".udeb",
    ".cpio", ".rpm", ".warc", ".mtree"]


def compression_methods() -> dict[str, int | None]:
    """ImageToolbox ZipCompressionMethod -> zipfile constant (None = not available here)."""
    return {"store": zipfile.ZIP_STORED, "deflate": zipfile.ZIP_DEFLATED, "bzip2": zipfile.ZIP_BZIP2,
            "lzma": zipfile.ZIP_LZMA, "xz": None, "zstd": getattr(zipfile, "ZIP_ZSTANDARD", None)}


def safe_segments(path: str) -> list[str] | None:
    """``ArchivePath.safeSegments``."""
    if not path.strip() or "\x00" in path:
        return None
    normalized = path.replace("\\", "/")
    if len(normalized) > MAX_PATH_LENGTH:
        return None
    if normalized.startswith("/") or _DRIVE.search(normalized):
        return None
    segments = [s for s in normalized.split("/") if s.strip()]
    if not segments or len(segments) > MAX_SEGMENTS:
        return None
    if any(s in (".", "..") for s in segments):
        return None
    return [s[:MAX_SEGMENT_LENGTH] for s in segments]


def archive_name(filename: str) -> str:
    """Folder name for "create subfolder": file name without its archive extension."""
    lower = filename.lower()
    ext = next((e for e in SUPPORTED_ARCHIVE_EXTENSIONS if lower.endswith(e)), None)
    base = filename[:-len(ext)] if ext else (filename.rsplit(".", 1)[0] if "." in filename else filename)
    segs = safe_segments(base if base.strip() else "Archive")
    return segs[-1] if segs else "Archive"


def unique_name(name: str, used: set[str]) -> str:
    """``AndroidArchiveManager.uniqueName`` (case-insensitive, "name (2).ext")."""
    if name.lower() not in used:
        used.add(name.lower())
        return name
    base, ext = (name.rsplit(".", 1)[0], "." + name.rsplit(".", 1)[1]) if "." in name else (name, "")
    i = 2
    while True:
        cand = f"{base} ({i}){ext}"
        if cand.lower() not in used:
            used.add(cand.lower())
            return cand
        i += 1


def unique_path(folder: Path, name: str, is_dir: bool = False) -> Path:
    """``createUniqueFile`` / ``createUniqueDirectory``."""
    cand = folder / name
    if not cand.exists():
        return cand
    base, ext = (name, "") if is_dir or "." not in name else (name.rsplit(".", 1)[0], "." + name.rsplit(".", 1)[1])
    i = 2
    while True:
        cand = folder / f"{base} ({i}){ext}"
        if not cand.exists():
            return cand
        i += 1


def _is_aes(info: zipfile.ZipInfo) -> bool:
    return info.compress_type == 99


def _encrypted(info: zipfile.ZipInfo) -> bool:
    return bool(info.flag_bits & 0x1)


def _entry_kind(info: zipfile.ZipInfo) -> str:
    mode = info.external_attr >> 16
    if info.is_dir() or (mode and stat.S_ISDIR(mode)):
        return "dir"
    if mode and stat.S_ISLNK(mode):
        return "symlink"
    if mode and not stat.S_ISREG(mode) and stat.S_IFMT(mode):
        return "special"
    return "file"


def open_zip(path: Path) -> zipfile.ZipFile:
    if not zipfile.is_zipfile(path):
        raise LazyRoomError(ErrorCode.UNSUPPORTED_FORMAT, f"Not a ZIP archive: {path.name}",
                              hint="Only ZIP-based archives (.zip, .cbz, .jar, .apk, .epub, ...) are supported; "
                                   "tar/7z/rar are not ported.")
    try:
        return zipfile.ZipFile(path, metadata_encoding="utf-8")  # libarchive reads names as UTF-8
    except UnicodeDecodeError:
        return zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise LazyRoomError(ErrorCode.DECODE_FAILED, f"Corrupt ZIP archive: {exc}") from None


def encryption_status(zf: zipfile.ZipFile) -> str:
    infos = zf.infolist()
    if any(_is_aes(i) for i in infos):
        return "unsupported"
    if any(_encrypted(i) for i in infos):
        return "password_required"
    return "none"


@dataclass
class Limits:
    max_entries: int = 100_000
    max_entry_bytes: int = 64 * 1024 ** 3
    max_total_bytes: int = 128 * 1024 ** 3
    max_ratio: float = 0.0  # 0 = off


def _limit(msg: str, **details: Any) -> LazyRoomError:
    return LazyRoomError(ErrorCode.RESOURCE_LIMIT, msg, details=details,
                           hint="Raise the limit parameters only if you trust this archive.")


def list_entries(zf: zipfile.ZipFile, limits: Limits) -> tuple[list[dict], list[dict]]:
    infos = zf.infolist()
    if len(infos) > limits.max_entries:
        raise _limit("Archive contains too many entries", entries=len(infos), max_entries=limits.max_entries)
    entries, unsafe, seen = [], [], set()
    for info in infos:
        kind = _entry_kind(info)
        segs = safe_segments(info.filename)
        if segs is None:
            unsafe.append({"path": info.filename, "reason": "unsafe path"})
            continue
        if kind not in ("dir", "file"):
            unsafe.append({"path": info.filename, "reason": kind})
            continue
        path = "/".join(segs)
        if path in seen:
            continue
        seen.add(path)
        entries.append({"path": path, "size": info.file_size, "compressed_size": info.compress_size,
                        "is_directory": kind == "dir", "encrypted": _encrypted(info),
                        "method": _method_name(info.compress_type)})
    return entries, unsafe


def _method_name(m: int) -> str:
    return {0: "store", 8: "deflate", 9: "deflate64", 12: "bzip2", 14: "lzma", 93: "zstd", 95: "xz",
            98: "ppmd", 99: "aes"}.get(m, str(m))


def _password_error(info: zipfile.ZipInfo, password: bytes | None) -> LazyRoomError:
    if password:
        return LazyRoomError(ErrorCode.INVALID_PARAMS, "Wrong archive password", details={"entry": info.filename})
    return LazyRoomError(ErrorCode.INVALID_PARAMS, "This archive is password protected",
                           details={"entry": info.filename}, hint="Pass the archive password.")


def _open_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo, password: bytes | None):
    if _is_aes(info):
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE,
                              "WinZip AES-encrypted entries are not supported (no permissive AES-ZIP library "
                              "is installed); only legacy ZipCrypto archives can be extracted",
                              details={"entry": info.filename})
    if _encrypted(info) and not password:
        raise _password_error(info, None)
    try:
        return zf.open(info, pwd=password if _encrypted(info) else None)
    except RuntimeError as exc:
        if "password" in str(exc).lower():
            raise _password_error(info, password) from None
        raise
    except NotImplementedError as exc:
        raise LazyRoomError(ErrorCode.UNSUPPORTED_FORMAT, f"Unsupported ZIP compression: {exc}",
                              details={"entry": info.filename, "method": _method_name(info.compress_type)}) from None


def verify_password(zf: zipfile.ZipFile, password: bytes) -> bool:
    """``verifyArchivePassphrase``: decrypt the first encrypted file entry completely (CRC checked)."""
    for info in zf.infolist():
        if _encrypted(info) and _entry_kind(info) == "file":
            try:
                with _open_entry(zf, info, password) as f:
                    while f.read(COPY_BUFFER):
                        pass
                return True
            except (LazyRoomError, zipfile.BadZipFile):
                return False
    return True


def extract(zf: zipfile.ZipFile, dest_root: Path, *, password: bytes | None, preserve: bool, skip_hidden: bool,
            selected: set[str] | None, limits: Limits, check_write: Callable[[Path], Path],
            created: list[Path]) -> dict[str, Any]:
    """Extract into ``dest_root`` (already created).  ``created`` collects new paths for rollback."""
    infos = zf.infolist()
    if len(infos) > limits.max_entries:
        raise _limit("Archive contains too many entries", entries=len(infos), max_entries=limits.max_entries)
    root = dest_root.resolve()
    declared = 0
    actual = 0
    files: list[str] = []
    skipped: list[dict[str, str]] = []
    for info in infos:
        segs = safe_segments(info.filename)
        if segs is None:
            raise LazyRoomError(ErrorCode.PATH_NOT_ALLOWED, f"Unsafe archive entry path: {info.filename}",
                                  hint="The archive tries to write outside the destination (zip-slip); "
                                       "nothing was extracted.")
        kind = _entry_kind(info)
        if kind not in ("dir", "file"):
            skipped.append({"path": info.filename, "reason": kind})
            continue
        if info.file_size > limits.max_entry_bytes:
            raise _limit("Archive entry is too large", entry=info.filename, size=info.file_size)
        declared += info.file_size
        if declared > limits.max_total_bytes:
            raise _limit("Unpacked archive is too large", max_total_bytes=limits.max_total_bytes)
        if skip_hidden and any(s.startswith(".") for s in segs):
            skipped.append({"path": info.filename, "reason": "hidden"})
            continue
        norm = "/".join(segs)
        if selected is not None:
            ok = any(s.startswith(norm + "/") for s in selected) if kind == "dir" else norm in selected
            if not ok:
                continue
        out_segs = segs if preserve else [segs[-1]]
        if kind == "dir":
            if preserve:
                _make_dirs(root, out_segs, created)
            continue
        parent = _make_dirs(root, out_segs[:-1], created)
        target = unique_path(parent, out_segs[-1])
        if not _inside(target, root):
            raise LazyRoomError(ErrorCode.PATH_NOT_ALLOWED, f"Unsafe archive entry path: {info.filename}")
        target = check_write(target)
        if limits.max_ratio and info.compress_size and info.file_size / info.compress_size > limits.max_ratio \
                and info.file_size > 1024 * 1024:
            raise _limit("Suspicious compression ratio (possible zip bomb)", entry=info.filename,
                         ratio=round(info.file_size / info.compress_size, 1), max_ratio=limits.max_ratio)
        with _open_entry(zf, info, password) as src:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o666)
            created.append(target)
            entry_bytes = 0
            with os.fdopen(fd, "wb") as dst:
                while True:
                    try:
                        chunk = src.read(COPY_BUFFER)
                    except RuntimeError as exc:
                        raise _password_error(info, password) from exc
                    except zipfile.BadZipFile as exc:  # CRC mismatch: corrupt data or wrong ZipCrypto key
                        if _encrypted(info):
                            raise _password_error(info, password) from None
                        raise LazyRoomError(ErrorCode.DECODE_FAILED, f"Corrupt entry {info.filename}: {exc}") \
                            from None
                    if not chunk:
                        break
                    entry_bytes += len(chunk)
                    actual += len(chunk)
                    if entry_bytes > limits.max_entry_bytes:
                        raise _limit("Archive entry is too large", entry=info.filename)
                    if actual > limits.max_total_bytes:
                        raise _limit("Unpacked archive is too large", max_total_bytes=limits.max_total_bytes)
                    if limits.max_ratio and info.compress_size and entry_bytes > 1024 * 1024 \
                            and entry_bytes / info.compress_size > limits.max_ratio:
                        raise _limit("Suspicious compression ratio (possible zip bomb)", entry=info.filename,
                                     max_ratio=limits.max_ratio)
                    dst.write(chunk)
        files.append(str(target))
    return {"files": files, "skipped": skipped, "bytes": actual}


def _inside(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _make_dirs(root: Path, segs: list[str], created: list[Path]) -> Path:
    cur = root
    for s in segs:
        cur = cur / s
        if cur.is_symlink() or (cur.exists() and not cur.is_dir()):
            raise LazyRoomError(ErrorCode.PATH_NOT_ALLOWED, f"Cannot create directory over an existing entry: {cur}")
        if not cur.exists():
            cur.mkdir()
            created.append(cur)
        if not _inside(cur, root):
            raise LazyRoomError(ErrorCode.PATH_NOT_ALLOWED, f"Directory escapes the destination: {cur}")
    return cur


def rollback(created: list[Path]) -> None:
    """Remove what this extraction created (files first, then directories, newest first)."""
    for p in reversed(created):
        try:
            if p.is_dir() and not p.is_symlink():
                p.rmdir()  # only empty: anything not created here is kept
            else:
                p.unlink()
        except OSError:
            pass


def write_zip(dst: Path, sources: list[tuple[str, Path]], method: int, level: int | None, overwrite: bool) -> int:
    """Stream a ZIP to a temp file next to ``dst`` and atomically move it into place."""
    if dst.exists() and not overwrite:
        raise LazyRoomError(ErrorCode.OUTPUT_EXISTS, f"Refusing to overwrite existing file: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pm-", suffix=".zip", dir=str(dst.parent))
    try:
        with os.fdopen(fd, "wb") as raw:
            with zipfile.ZipFile(raw, "w", compression=method, compresslevel=level) as zf:
                for name, src in sources:
                    zi = zipfile.ZipInfo.from_file(src, arcname=name, strict_timestamps=False)
                    zi.external_attr = (stat.S_IFREG | FILE_PERMISSIONS) << 16
                    zi.compress_type = method
                    if level is not None:  # public attribute since Python 3.13 (``_compresslevel`` before)
                        setattr(zi, "compress_level" if hasattr(zi, "compress_level") else "_compresslevel", level)
                    with open(src, "rb") as fin, zf.open(zi, "w") as fout:
                        shutil.copyfileobj(fin, fout, COPY_BUFFER)
            raw.flush()
            os.fsync(raw.fileno())
        if os.name != "nt":
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, dst)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dst.stat().st_size
