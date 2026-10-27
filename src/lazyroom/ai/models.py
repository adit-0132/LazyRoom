"""Catalogue of optional ONNX models (ported from ImageToolbox's ``NeuralModel`` list).

Models are *never* shipped with LazyRoom.  ``catalog.json`` (next to this file)
records, for every model, the ImageToolbox download URL (HF ``T8RIN/imagetoolbox-models``),
size, SHA-256, speed tier, and — researched per model — the original project and
the licence of its *weights*.  ``restricted: true`` marks non-commercial,
copyleft or unverified licences.  See docs/MODELS.md.

A model becomes available only after an explicit, confirmed download
(``lazyroom models download NAME --yes``); the download is verified against
the catalogue SHA-256 and written atomically.  Zip "packs" (style transfer,
PaddleOCR) are extracted into a directory that is marked complete only after
verification.  ``@<task>`` names (e.g. ``@upscale``) are aliases for the
recommended default model of a task.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable

from ..core.errors import ErrorCode, LazyRoomError
from ..core.paths import ensure, models_dir

CATALOG_PATH = Path(__file__).with_name("catalog.json")

# task -> operation / tool that uses it
TASKS: dict[str, dict[str, Any]] = {
    "upscale": {"op": "ai.upscale", "title": "Upscaling"},
    "remove_background": {"op": "ai.remove_background", "title": "Background removal / subject mask"},
    "colorize": {"op": "ai.colorize", "title": "Colorization"},
    "dejpeg": {"op": "ai.dejpeg", "title": "JPEG artifact removal"},
    "denoise": {"op": "ai.denoise", "title": "Denoise / deblur"},
    "remove_artifacts": {"op": "ai.remove_artifacts", "title": "Artifact removal"},
    "enhance": {"op": "ai.enhance", "title": "Enhancement"},
    "restore_anime": {"op": "ai.restore_anime", "title": "Anime / cartoon restoration"},
    "restore_scans": {"op": "ai.restore_scans", "title": "Scans / documents"},
    "style_transfer": {"op": "ai.style_transfer", "title": "Style transfer"},
    "depth": {"op": "ai.depth_map", "title": "Depth estimation"},
    "detect_ai_generated": {"op": "tool:detect_ai_generated", "title": "AI-generated image detection"},
    "remove_watermark": {"op": "ai.remove_watermark", "title": "Watermark removal"},
    "inpaint": {"op": "ai.inpaint", "title": "Inpainting"},
    "ocr": {"op": "tool:ocr_paddle", "title": "Text recognition (PaddleOCR)"},
}


@functools.lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _build_registry() -> dict[str, dict[str, Any]]:
    return {m["name"]: m for m in _catalog()["models"]}


# name -> metadata (the catalogue; tests may add fake entries with register_model)
MODELS: dict[str, dict[str, Any]] = _build_registry()


def register_model(name: str, **meta: Any) -> None:
    meta.setdefault("name", name)
    MODELS[name] = meta


def task_defaults() -> dict[str, str]:
    return dict(_catalog().get("task_defaults", {}))


def models_for_task(task: str) -> list[str]:
    return [n for n, m in MODELS.items() if m.get("task") == task]


def default_model(task: str) -> str:
    d = task_defaults().get(task)
    if d and d in MODELS:
        return d
    names = models_for_task(task)
    if not names:
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, f"No catalogue model for task '{task}'")
    return names[0]


def resolve_name(name: str) -> str:
    """Canonical catalogue name: exact, case-insensitive, without extension, or ``@task`` alias."""
    if name in MODELS:
        return name
    if name.startswith("@") and name[1:] in TASKS:
        return default_model(name[1:])
    low = name.lower()
    for n in MODELS:
        if n.lower() == low:
            return n
    stems = [n for n in MODELS if n.lower().rsplit(".", 1)[0] == low]
    if len(stems) == 1:
        return stems[0]
    import difflib

    raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"Unknown model '{name}'",
                          details={"suggestions": difflib.get_close_matches(name, list(MODELS), 5, 0.4)},
                          hint="List models with: lazyroom models list")


def _is_pack(meta: dict[str, Any]) -> bool:
    return bool(meta.get("pack"))


def model_path(name: str) -> Path:
    """File (or extracted directory for zip packs) where the model lives once downloaded."""
    name = resolve_name(name)
    meta = MODELS[name]
    fn = meta.get("filename") or name
    if _is_pack(meta):
        return models_dir() / fn.rsplit(".", 1)[0]
    return models_dir() / fn


def _self_present(name: str) -> bool:
    meta = MODELS[name]
    p = model_path(name)
    if _is_pack(meta):
        return (p / ".complete").exists()
    return p.is_file() and p.stat().st_size > 0


def model_available(name: str) -> bool:
    """True when the model (and the models it depends on) is downloaded.

    ``@task`` checks whether *any* model of that task is installed (used by the
    operations' ``requires`` so they report unavailability cleanly)."""
    if name.startswith("@") and name[1:] in TASKS:
        return any(model_available(n) for n in models_for_task(name[1:]))
    try:
        name = resolve_name(name)
    except LazyRoomError:
        return False
    if not _self_present(name):
        return False
    return all(model_available(d) for d in MODELS[name].get("requires_models", []))


def download_command(name: str) -> str:
    return f"lazyroom models download {name} --yes"


def require_model(name: str) -> Path:
    """Path of a downloaded model or BACKEND_UNAVAILABLE with the exact download command."""
    name = resolve_name(name)
    if not model_available(name):
        meta = MODELS[name]
        missing = [n for n in [name, *meta.get("requires_models", [])] if not _self_present(n)]
        size = sum(int(MODELS[n].get("size") or 0) for n in missing)
        raise LazyRoomError(
            ErrorCode.BACKEND_UNAVAILABLE,
            f"Model '{name}' is not downloaded",
            details={"model": name, "missing": missing, "download_mb": round(size / 1e6, 1),
                     "license": meta.get("license"), "restricted": bool(meta.get("restricted")),
                     "command": download_command(name)},
            hint=f"Download it (after reviewing its licence) with: {download_command(name)}")
    return model_path(name)


def model_info(name: str) -> dict[str, Any]:
    name = resolve_name(name)
    d = dict(MODELS[name])
    d["available"] = model_available(name)
    d["path"] = str(model_path(name))
    return d


def list_models(task: str | None = None) -> list[dict[str, Any]]:
    out = []
    for name, m in MODELS.items():
        if task and m.get("task") != task:
            continue
        out.append({"name": name, "title": m.get("title"), "task": m.get("task"), "type": m.get("type"),
                    "license": m.get("license"), "restricted": bool(m.get("restricted")),
                    "size_mb": round(int(m.get("size") or 0) / 1e6, 1),
                    "speed": (m.get("speed") or {}).get("tier"),
                    "source": m.get("upstream_url"), "available": model_available(name)})
    return out


# ---------------------------------------------------------------------------
# downloads
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path, *, expected_size: int | None, progress: Callable | None) -> str:
    """Stream ``url`` into ``dest`` (a temp path) and return its SHA-256."""
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "LazyRoom model downloader"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or expected_size or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if expected_size and done > expected_size * 1.05 + (1 << 20):
                    raise LazyRoomError(ErrorCode.RESOURCE_LIMIT, "Download is larger than the catalogue size")
                if progress:
                    progress(done, total)
            f.flush()
            os.fsync(f.fileno())
    except LazyRoomError:
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, f"Download failed: {exc}",
                              hint="Check the network connection and retry.") from exc
    return h.hexdigest()


def _check_disk(target_dir: Path, size: int) -> None:
    free = shutil.disk_usage(target_dir).free
    need = int(size * 2.2) + (64 << 20)  # zip packs need room for archive + extracted files
    if size and free < need:
        raise LazyRoomError(ErrorCode.RESOURCE_LIMIT,
                              f"Not enough disk space: need ~{need / 1e6:.0f} MB, {free / 1e6:.0f} MB free in {target_dir}")


def _verify(meta: dict[str, Any], digest: str, path: Path) -> None:
    expected = (meta.get("sha256") or "").lower()
    if not expected:
        path.unlink(missing_ok=True)
        raise LazyRoomError(ErrorCode.INTERNAL, f"No checksum in the catalogue for {meta['name']}; refusing to install")
    if digest != expected:
        path.unlink(missing_ok=True)
        raise LazyRoomError(ErrorCode.DECODE_FAILED, "Downloaded model checksum mismatch",
                              details={"model": meta["name"], "expected": expected, "actual": digest},
                              hint="The download was corrupted or the upstream file changed; retry later.")


_MAX_PACK_BYTES = 4 << 30


def _extract_pack(meta: dict[str, Any], archive: Path, final_dir: Path) -> None:
    tmp_dir = final_dir.with_name(final_dir.name + f".part-{os.getpid()}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as z:
            if sum(i.file_size for i in z.infolist()) > _MAX_PACK_BYTES:
                raise LazyRoomError(ErrorCode.RESOURCE_LIMIT, "Model archive expands to more than 4 GB")
            for info in z.infolist():
                if info.is_dir():
                    continue
                base = os.path.basename(info.filename.replace("\\", "/"))  # flatten like ITB unzipModels
                if not base or base.startswith("."):
                    continue
                with z.open(info) as src, open(tmp_dir / base, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
        for spec in meta["pack"].get("files", []):
            p = tmp_dir / spec["name"]
            if not p.is_file():
                raise LazyRoomError(ErrorCode.DECODE_FAILED, f"Model archive lacks {spec['name']}")
            if spec.get("size") and p.stat().st_size != int(spec["size"]):
                raise LazyRoomError(ErrorCode.DECODE_FAILED, f"{spec['name']} has an unexpected size")
            if spec.get("sha256") and _sha256_file(p) != spec["sha256"]:
                raise LazyRoomError(ErrorCode.DECODE_FAILED, f"{spec['name']} checksum mismatch")
        for pattern in meta["pack"].get("require_globs", []):
            if not any(tmp_dir.glob(pattern)):
                raise LazyRoomError(ErrorCode.DECODE_FAILED, f"Model archive lacks a file matching {pattern}")
        (tmp_dir / ".complete").write_text(json.dumps({"sha256": meta["sha256"], "url": meta.get("url")}))
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp_dir, final_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _download_one(name: str, progress: Callable | None) -> dict[str, Any]:
    meta = MODELS[name]
    final = model_path(name)
    if _self_present(name):
        return {"name": name, "path": str(final), "status": "already present"}
    ensure(final.parent)
    _check_disk(final.parent, int(meta.get("size") or 0))
    tmp = final.with_name(final.name + (".zip" if _is_pack(meta) else "") + f".part-{os.getpid()}")
    try:
        digest = _fetch(meta["url"], tmp, expected_size=int(meta.get("size") or 0) or None, progress=progress)
        _verify(meta, digest, tmp)
        if _is_pack(meta):
            _extract_pack(meta, tmp, final)
        else:
            os.replace(tmp, final)  # atomic on the same filesystem
    finally:
        tmp.unlink(missing_ok=True)
    side = {k: meta.get(k) for k in ("name", "title", "url", "sha256", "size", "license", "upstream_url")}
    side_path = final.parent / (final.name + ".meta.json")
    side_tmp = side_path.with_name(side_path.name + ".tmp")
    side_tmp.write_text(json.dumps(side, indent=1))
    os.replace(side_tmp, side_path)
    return {"name": name, "path": str(final), "sha256": digest, "size": meta.get("size"), "status": "downloaded"}


def download_model(name: str, *, progress=None) -> dict[str, Any]:
    """Download a catalogue model (and its dependencies), verifying SHA-256; atomic install.

    Callers are responsible for obtaining the user's confirmation first (the CLI
    requires ``--yes``)."""
    name = resolve_name(name)
    meta = MODELS[name]
    deps = [_download_one(d, progress) for d in meta.get("requires_models", [])]
    res = _download_one(name, progress)
    res["license"] = meta.get("license")
    res["restricted"] = bool(meta.get("restricted"))
    if res["restricted"]:
        res["warning"] = (f"Licence of '{name}' is {meta.get('license')} — "
                          f"{meta.get('license_notes') or 'not cleared for unrestricted (e.g. commercial) use'}.")
    if deps:
        res["dependencies"] = deps
    return res


def delete_model(name: str) -> dict[str, Any]:
    name = resolve_name(name)
    p = model_path(name)
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink(missing_ok=True)
    (p.parent / (p.name + ".meta.json")).unlink(missing_ok=True)
    from . import runtime

    runtime.clear_sessions()
    return {"name": name, "status": "deleted"}
