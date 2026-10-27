"""Detection of optional external backends (G'MIC, Tesseract, ExifTool, ONNX models...).

Operations declare ``requires=[...]``; the registry reports them as
unavailable — with installation guidance — instead of failing obscurely.
"""
from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .errors import ErrorCode, LazyRoomError

INSTALL_HINTS = {
    "gmic": "Install the G'MIC 3.x/4.x command-line tool (Fedora: 'sudo dnf install gmic'; Debian/Ubuntu: the "
            "distribution package can be the old 2.x series, so install the current .deb from "
            "https://gmic.eu/download.html; Windows: download gmic_*_cli_win64.zip from the same page) "
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


# 2.x lacks most of the fx_* filters the ImageToolbox ports call. Tested with 3.7.6 and 4.0.6.
GMIC_MIN_VERSION = (3, 0, 0)


@functools.lru_cache(maxsize=None)
def _gmic_probe() -> tuple[str | None, tuple[int, int, int] | None]:
    """(path, version) of the G'MIC CLI that would be used; version is None if it could not be read."""
    p = _which("LAZYROOM_GMIC", "gmic")
    if not p:
        return None, None
    try:
        r = subprocess.run([p, "-version"], capture_output=True, timeout=20)
    except Exception:
        return None, None
    out = (r.stdout + r.stderr).decode("utf-8", "replace")
    m = re.search(r"[Vv]ersion\s+(\d+)\.(\d+)\.(\d+)", out)
    if m:
        return p, (int(m[1]), int(m[2]), int(m[3]))
    return (p if r.returncode == 0 or "G'MIC" in out else None), None


def gmic_path() -> str | None:
    """Path of a usable G'MIC CLI (None if missing or older than GMIC_MIN_VERSION)."""
    p, v = _gmic_probe()
    return None if v is not None and v < GMIC_MIN_VERSION else p


def gmic_version() -> tuple[int, int, int] | None:
    return _gmic_probe()[1] if gmic_path() else None


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
    if name == "gmic":
        p, v = _gmic_probe()
        if p and v is not None and v < GMIC_MIN_VERSION:
            return (f"Found G'MIC {'.'.join(map(str, v))} at {p}, but LazyRoom needs "
                    f"{'.'.join(map(str, GMIC_MIN_VERSION))} or newer. " + INSTALL_HINTS["gmic"])
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
        v = gmic_version()
        if v:
            out["gmic"]["version"] = ".".join(map(str, v))
    return out
