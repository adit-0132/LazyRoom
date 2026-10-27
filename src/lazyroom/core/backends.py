"""Detection of optional external backends (G'MIC, Tesseract, ExifTool, ONNX models...).

Operations declare ``requires=[...]``; the registry reports them as
unavailable — with installation guidance — instead of failing obscurely.
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .errors import ErrorCode, LazyRoomError

INSTALL_HINTS = {
    "gmic": "Install the G'MIC command-line tool (Fedora: 'sudo dnf install gmic'; Debian/Ubuntu: "
            "'sudo apt install gmic'; Windows: download gmic_*_cli_win64.zip from https://gmic.eu/download.html) "
            "or set LAZYROOM_GMIC to the gmic executable.",
    "tesseract": "Install Tesseract OCR (Fedora: 'sudo dnf install tesseract'; Debian/Ubuntu: "
                 "'sudo apt install tesseract-ocr'; Windows: UB-Mannheim installer) or set LAZYROOM_TESSERACT.",
    "exiftool": "Install ExifTool (Fedora: 'sudo dnf install perl-Image-ExifTool'; Debian/Ubuntu: "
                "'sudo apt install libimage-exiftool-perl'; Windows: exiftool.org).",
    "onnxruntime": "pip install 'lazyroom[ai]' (installs onnxruntime).",
    "rawpy": "pip install 'lazyroom[raw]' (installs rawpy/LibRaw).",
    "pyside6": "pip install 'lazyroom[gui]' (Qt is used for headless OpenGL shader rendering).",
}


def _which(env: str, *names: str) -> str | None:
    override = os.environ.get(env)
    if override and Path(override).exists():
        return str(Path(override).resolve())
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    # bundled tools (AppImage / Windows portable) live next to the app
    from .paths import bundled_bin_dir

    for n in names:
        for cand in (bundled_bin_dir() / n, bundled_bin_dir() / (n + ".exe")):
            if cand.exists():
                return str(cand)
    return None


@functools.lru_cache(maxsize=None)
def gmic_path() -> str | None:
    p = _which("LAZYROOM_GMIC", "gmic")
    if not p:
        return None
    try:
        r = subprocess.run([p, "-version"], capture_output=True, timeout=20)
        return p if r.returncode == 0 or b"G'MIC" in (r.stdout + r.stderr) else None
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def tesseract_path() -> str | None:
    return _which("LAZYROOM_TESSERACT", "tesseract")


@functools.lru_cache(maxsize=None)
def exiftool_path() -> str | None:
    return _which("LAZYROOM_EXIFTOOL", "exiftool")


def _has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


CHECKS: dict[str, Callable[[], bool]] = {
    "gmic": lambda: gmic_path() is not None,
    "tesseract": lambda: tesseract_path() is not None,
    "exiftool": lambda: exiftool_path() is not None,
    "onnxruntime": lambda: _has_module("onnxruntime"),
    "rawpy": lambda: _has_module("rawpy"),
    "pyside6": lambda: _has_module("PySide6"),
}


def is_available(name: str) -> bool:
    if name.startswith("model:"):
        from ..ai.models import model_available  # lazy: optional subsystem

        return model_available(name.split(":", 1)[1])
    check = CHECKS.get(name)
    return bool(check and check())


def hint_for(name: str) -> str:
    if name.startswith("model:"):
        return (f"Download the model with: lazyroom models download {name.split(':', 1)[1]} "
                "(or the 'models_download' MCP action).")
    return INSTALL_HINTS.get(name, f"Install backend '{name}'.")


def require(name: str) -> None:
    if not is_available(name):
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, f"Required backend '{name}' is not available",
                              details={"backend": name}, hint=hint_for(name))


def report() -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for name in CHECKS:
        ok = is_available(name)
        out[name] = {"available": ok}
        if not ok:
            out[name]["hint"] = hint_for(name)
    if out["gmic"]["available"]:
        out["gmic"]["path"] = gmic_path()
    return out
