"""Texture preset registry (port of ImageToolbox's "Texture Generation" feature).

Every ImageToolbox ``TextureFilterType`` is registered as a :class:`TexturePreset`
with a snake_case id, its English title, UI category, typed parameter schema
(ImageToolbox ranges / defaults) and a render function.

Render contract (all backends):
    ``render_fn(width, height, seed, values) -> float32 (H, W, 3|4)``, sRGB 0..1,
    straight (non-premultiplied) alpha.  ``values`` are the validated parameter
    values (colours are ``(r, g, b, a)`` float tuples, choices are strings).
    ``seed == 0`` reproduces ImageToolbox's default output; other values vary the
    preset's random generator (see each backend module).
"""
from __future__ import annotations

import ast
import difflib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..core import backends
from ..core.errors import ErrorCode, LazyRoomError, invalid
from ..core.params import Param, params_schema, validate_params

CATEGORIES: dict[str, str] = {
    "nature": "Nature",
    "materials": "Materials",
    "space": "Space",
    "patterns": "Patterns",
    "abstract": "Abstract",
}

BACKENDS: dict[str, str] = {
    "fast-noise": "ImageToolboxLibs fast-noise procedural textures (C++ on FastNoiseLite), numba port",
    "pattern": "ImageToolbox GLSL pattern shaders (GPUImage), numpy/numba port",
    "raymarch": "ImageToolbox GLSL ray-marched 3D shapes (GPUImage), numba port",
    "jhlabs": "ImageToolboxLibs jhlabs (Rust) texture filters, numba port",
    "gmic": "G'MIC CLI command (exact ImageToolboxLibs command string)",
}

RenderFn = Callable[[int, int, int, dict], np.ndarray]


@dataclass
class TexturePreset:
    id: str
    title: str
    category: str
    backend: str
    upstream: str                      # ImageToolbox TextureFilterType enum name
    params: list[Param]
    render_fn: RenderFn
    requires: list[str] = field(default_factory=list)
    status: str = "faithful"           # faithful | approximate
    notes: str = ""
    description: str = ""

    # ------------------------------------------------------------------
    def validate(self, options: dict[str, Any] | None) -> dict[str, Any]:
        """Validate preset options (fills defaults); raises INVALID_PARAMS."""
        if options is not None and not isinstance(options, dict):
            raise invalid(f"Options for texture '{self.id}' must be a JSON object, got {type(options).__name__}",
                          hint="Pass e.g. {\"scale\": 0.01}.")
        return validate_params(self.params, options or {}, op_id=f"texture:{self.id}")

    def availability(self) -> tuple[bool, str | None]:
        for b in self.requires:
            if not backends.is_available(b):
                return False, backends.hint_for(b)
        return True, None

    def render(self, width: int, height: int, seed: int = 0, *,
               resolve_file: Callable[[str], Any] | None = None, **options: Any) -> np.ndarray:
        """Render the preset to a float32 (H, W, 3|4) image (alpha dropped when fully opaque)."""
        width, height = int(width), int(height)
        if width < 1 or height < 1:
            raise invalid(f"Texture size must be positive, got {width}x{height}")
        values = self.validate(options)
        for p in self.params:
            if p.kind == "file" and values.get(p.name) and resolve_file is not None:
                values[p.name] = str(resolve_file(values[p.name]))
        for b in self.requires:
            backends.require(b)
        out = self.render_fn(width, height, int(seed), values)
        out = np.asarray(out, dtype=np.float32)
        if out.ndim != 3 or out.shape[:2] != (height, width) or out.shape[2] not in (3, 4):
            raise LazyRoomError(ErrorCode.INTERNAL,
                                  f"Texture '{self.id}' produced shape {out.shape}, expected ({height}, {width}, 3|4)")
        out = np.nan_to_num(np.clip(out, 0.0, 1.0), nan=0.0)
        if out.shape[2] == 4 and float(out[..., 3].min()) >= 1.0:
            out = out[..., :3]
        return np.ascontiguousarray(out)

    def describe(self, *, schema: bool = True) -> dict[str, Any]:
        ok, hint = self.availability()
        d: dict[str, Any] = {"id": self.id, "title": self.title, "category": self.category,
                             "backend": self.backend, "upstream": self.upstream, "status": self.status,
                             "available": ok}
        if self.description:
            d["description"] = self.description
        if self.requires:
            d["requires"] = list(self.requires)
        if hint:
            d["install_hint"] = hint
        if self.notes:
            d["notes"] = self.notes
        if schema:
            d["params"] = [p.describe() for p in self.params]
            d["options_schema"] = params_schema(self.params)
        return d


PRESETS: dict[str, TexturePreset] = {}


def register(preset: TexturePreset) -> TexturePreset:
    if preset.id in PRESETS:
        raise ValueError(f"Duplicate texture preset {preset.id!r}")
    if preset.category not in CATEGORIES:
        raise ValueError(f"Unknown texture category {preset.category!r} for {preset.id}")
    if preset.backend not in BACKENDS:
        raise ValueError(f"Unknown texture backend {preset.backend!r} for {preset.id}")
    names = [p.name for p in preset.params]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate parameter names in texture {preset.id}: {names}")
    PRESETS[preset.id] = preset
    return preset


def get_preset(preset_id: str) -> TexturePreset:
    from . import _load_all

    _load_all()
    p = PRESETS.get(preset_id)
    if p is None:
        raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"Unknown texture preset '{preset_id}'",
                              details={"suggestions": difflib.get_close_matches(preset_id, list(PRESETS), 5)},
                              hint="Use the texture_presets tool to list valid preset ids.")
    return p


def parse_options(options: Any) -> dict[str, Any]:
    """Preset options given as a JSON object string (or a dict) -> dict; INVALID_PARAMS otherwise."""
    if options is None:
        return {}
    if isinstance(options, dict):
        return dict(options)
    if not isinstance(options, str):
        raise invalid(f"Texture options must be a JSON object, got {type(options).__name__}", param="options")
    text = options.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except ValueError as exc:
        try:  # a Python dict repr (e.g. str(dict) from an SDK caller) is accepted too
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            raise invalid(f"Texture options are not valid JSON: {exc}", param="options",
                          hint='Use a JSON object, e.g. {"scale": 0.01, "sky_color": "#6FA9DC"}.') from None
    if not isinstance(value, dict):
        raise invalid("Texture options must be a JSON object", param="options",
                      hint='Use a JSON object, e.g. {"scale": 0.01}.')
    return value


# ---------------------------------------------------------------------------
# Colour helpers shared by the backends
# ---------------------------------------------------------------------------

def argb_hex(argb: int) -> str:
    """Android ARGB colour int (possibly negative) -> '#RRGGBBAA' (or '#RRGGBB' when opaque)."""
    v = int(argb) & 0xFFFFFFFF
    a, rgb = v >> 24, v & 0xFFFFFF
    return f"#{rgb:06X}" if a == 255 else f"#{rgb:06X}{a:02X}"


def color_argb(c: Sequence[float]) -> int:
    """(r, g, b, a) floats 0..1 -> unsigned 32-bit ARGB int (8-bit channels, rounded)."""
    r, g, b, a = (max(0, min(255, int(round(float(x) * 255.0)))) for x in (tuple(c) + (1.0,))[:4])
    return (a << 24) | (r << 16) | (g << 8) | b


def colors_argb(colors: Sequence[Sequence[float]]) -> np.ndarray:
    """List of RGBA float tuples -> uint32 array of ARGB ints (for numba kernels)."""
    return np.array([color_argb(c) for c in colors], dtype=np.uint32)


def argb_to_rgba(packed: np.ndarray) -> np.ndarray:
    """uint32 ARGB array (H, W) -> float32 (H, W, 4) straight RGBA 0..1."""
    p = packed.astype(np.uint32)
    out = np.empty(p.shape + (4,), np.float32)
    out[..., 0] = (p >> 16) & 0xFF
    out[..., 1] = (p >> 8) & 0xFF
    out[..., 2] = p & 0xFF
    out[..., 3] = (p >> 24) & 0xFF
    out *= np.float32(1.0 / 255.0)
    return out
