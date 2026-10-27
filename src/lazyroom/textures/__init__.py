"""Procedural texture generation (port of ImageToolbox's "Texture Generation" feature).

ImageToolbox (https://github.com/T8RIN/ImageToolbox, Apache-2.0) offers 170+
texture presets backed by several engines; each is ported here (see
``docs/dev/port_reports/textures.md`` for per-preset fidelity):

* ``fast-noise`` - ImageToolboxLibs' C++ procedural textures on FastNoiseLite (MIT),
  ported to numba (:mod:`.fastnoise_lite`, :mod:`.fastnoise`);
* ``pattern`` / ``raymarch`` - ImageToolbox's GLSL shaders, ported to numpy/numba;
* ``jhlabs`` - ImageToolboxLibs' Rust re-implementation of the JH Labs texture filters;
* ``gmic`` - G'MIC commands (external CLI, exact ImageToolboxLibs command strings).

Public API::

    from lazyroom.textures import render, list_presets, get_preset
    img = render("brushed_metal", 512, 512, seed=0, options={"radius": 4})
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ._registry import BACKENDS, CATEGORIES, PRESETS, TexturePreset, get_preset, parse_options

_BACKEND_MODULES = ("fastnoise", "patterns", "raymarch", "jhlabs", "gmic")
_loaded = False


def _load_all() -> None:
    global _loaded
    if not _loaded:
        import importlib

        for m in _BACKEND_MODULES:
            importlib.import_module(f"{__name__}.{m}")
        _loaded = True


def preset_ids() -> list[str]:
    _load_all()
    return list(PRESETS)


def list_presets(query: str | None = None, *, schema: bool = True) -> list[dict[str, Any]]:
    """Describe presets (optionally filtered by words matched against id/title/category/backend)."""
    _load_all()
    words = [w for w in (query or "").lower().replace("-", " ").split() if w]
    out = []
    for p in PRESETS.values():
        hay = f"{p.id} {p.id.replace('_', ' ')} {p.title} {p.category} {p.backend} {p.upstream}".lower()
        if words and not all(w in hay for w in words):
            continue
        out.append(p.describe(schema=schema))
    return out


def render(preset_id: str, width: int, height: int, seed: int = 0, options: dict[str, Any] | str | None = None, *,
           resolve_file: Callable[[str], Any] | None = None) -> np.ndarray:
    """Render ``preset_id`` at ``width`` x ``height`` -> float32 (H, W, 3|4) in 0..1.

    ``options`` (dict or JSON object string) are validated against the preset schema (INVALID_PARAMS).
    """
    return get_preset(preset_id).render(width, height, seed, resolve_file=resolve_file, **parse_options(options))


SEED_NOTE = ("seed 0 reproduces ImageToolbox's default output; other values re-seed the preset's random "
             "generator (fast-noise: noise seed 1337+seed; patterns with a seed: seed % 10001; jhlabs: Java "
             "Random / Perlin seed offsets; G'MIC: srand seed). Presets without randomness ignore it.")

__all__ = ["BACKENDS", "CATEGORIES", "PRESETS", "SEED_NOTE", "TexturePreset", "get_preset", "list_presets",
           "parse_options", "preset_ids", "render"]
