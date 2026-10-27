"""Platform-aware application directories."""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP = "lazyroom"


def _xdg(var: str, fallback: str) -> Path:
    v = os.environ.get(var)
    return Path(v) if v else Path.home() / fallback


def config_dir() -> Path:
    if os.environ.get("LAZYROOM_HOME"):
        return Path(os.environ["LAZYROOM_HOME"]) / "config"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / "LazyRoom"
    return _xdg("XDG_CONFIG_HOME", ".config") / APP


def data_dir() -> Path:
    if os.environ.get("LAZYROOM_HOME"):
        return Path(os.environ["LAZYROOM_HOME"]) / "data"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "LazyRoom"
    return _xdg("XDG_DATA_HOME", ".local/share") / APP


def cache_dir() -> Path:
    if os.environ.get("LAZYROOM_HOME"):
        return Path(os.environ["LAZYROOM_HOME"]) / "cache"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "LazyRoom" / "Cache"
    return _xdg("XDG_CACHE_HOME", ".cache") / APP


def presets_dir() -> Path:
    return data_dir() / "presets"


def models_dir() -> Path:
    return data_dir() / "models"


def artifacts_dir() -> Path:
    return cache_dir() / "artifacts"


def package_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def bundled_bin_dir() -> Path:
    """Directory with bundled helper executables (frozen builds)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "bin"
    return Path(__file__).resolve().parent.parent / "bin"


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
