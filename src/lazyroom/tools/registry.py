"""Utility tools: operations that produce data or new files rather than editing
one image's pipeline (palette extraction, OCR, QR codes, collages, stacking,
metadata, duplicates, conversions...).

Tools share the parameter system with operations, so they are documented and
validated the same way for every interface (``tools_list`` / ``tool_describe`` /
``tool_run``).  Every file access goes through the workspace path policy.
"""
from __future__ import annotations

import difflib
import importlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..core import backends
from ..core.errors import ErrorCode, LazyRoomError
from ..core.params import Param, params_schema, params_to_json, validate_params

TOOL_MODULES = ["analysis", "metadata", "codes", "compose", "generate", "convert", "ai_tools", "textures", "fractals", "presets_misc"]


@dataclass
class ToolSpec:
    name: str
    title: str
    fn: Callable[..., dict[str, Any]]
    params: list[Param] = field(default_factory=list)
    category: str = "utility"
    description: str = ""
    requires: list[str] = field(default_factory=list)
    destructive: bool = False
    source: dict[str, Any] = field(default_factory=dict)

    def availability(self) -> tuple[bool, str | None]:
        for b in self.requires:
            if not backends.is_available(b):
                return False, backends.hint_for(b)
        return True, None

    def describe(self) -> dict[str, Any]:
        ok, hint = self.availability()
        d = {"name": self.name, "title": self.title, "category": self.category, "description": self.description,
             "params": [p.describe() for p in self.params], "input_schema": params_schema(self.params),
             "destructive": self.destructive, "available": ok}
        if hint:
            d["install_hint"] = hint
        if self.requires:
            d["requires"] = self.requires
        if self.source:
            d["source"] = self.source
        return d


TOOLS: dict[str, ToolSpec] = {}
_loaded = False


def tool(name: str, title: str, *, params: Sequence[Param] = (), category: str = "utility",
         requires: Sequence[str] = (), destructive: bool = False, source: dict[str, Any] | None = None) -> Callable:
    def deco(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        if name in TOOLS:
            raise ValueError(f"Duplicate tool {name}")
        TOOLS[name] = ToolSpec(name=name, title=title, fn=fn, params=list(params), category=category,
                               description=(fn.__doc__ or "").strip(), requires=list(requires),
                               destructive=destructive, source=source or {})
        return fn

    return deco


def _ensure() -> None:
    global _loaded
    if not _loaded:
        for m in TOOL_MODULES:
            importlib.import_module(f"{__package__}.{m}")
        _loaded = True


def get_tool(name: str) -> ToolSpec:
    _ensure()
    t = TOOLS.get(name)
    if t is None:
        raise LazyRoomError(ErrorCode.UNKNOWN_TOOL, f"Unknown tool '{name}'",
                              details={"suggestions": difflib.get_close_matches(name, list(TOOLS), 5)},
                              hint="Use tools_list to discover tools.")
    return t


def list_tools(query: str | None = None) -> list[dict[str, Any]]:
    _ensure()
    out = []
    for t in sorted(TOOLS.values(), key=lambda t: (t.category, t.name)):
        if query and not any(q in (t.name + " " + t.title + " " + t.description).lower()
                             for q in query.lower().split()):
            continue
        ok, _ = t.availability()
        d = {"name": t.name, "title": t.title, "category": t.category,
             "summary": t.description.split("\n")[0][:160]}
        if not ok:
            d["available"] = False
        if t.destructive:
            d["destructive"] = True
        out.append(d)
    return out


def run_tool(api: Any, name: str, params: dict[str, Any]) -> dict[str, Any]:
    t = get_tool(name)
    for b in t.requires:
        backends.require(b)
    values = validate_params(t.params, params, op_id=f"tool:{name}")
    if t.destructive:
        api._require_confirm(bool(values.get("confirm")), f"run destructive tool '{name}'")
    t0 = time.perf_counter()
    res = t.fn(api, **values)
    res.setdefault("tool", name)
    res["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return res


# -- helpers for tool implementations -----------------------------------------

def read_image(api: Any, path: str):
    from ..core.io import load_image

    p = api.ws.policy.check_read(path)
    return load_image(p, max_pixels=api.ws.config.max_pixels)


def out_path(api: Any, path: str, *, overwrite: bool = False) -> Path:
    if not path:
        raise LazyRoomError(ErrorCode.INVALID_PARAMS, "An output path is required")
    return api.ws.policy.check_write(path, overwrite=overwrite)


def save(api: Any, img, path: str, *, overwrite: bool = False, quality: int = 92, meta=None,
         metadata: str = "strip") -> dict[str, Any]:
    from ..core.io import atomic_write, encode_image, format_for_path

    p = out_path(api, path, overwrite=overwrite)
    ext, _ = format_for_path(p)
    data = encode_image(img, ext, quality=quality, meta=meta, metadata=metadata)
    atomic_write(p, data, overwrite=overwrite)
    return {"path": str(p), "width": int(img.shape[1]), "height": int(img.shape[0]), "bytes": len(data)}


def expand_inputs(api: Any, inputs: list[str] | str, recursive: bool = False) -> list[str]:
    return api._expand_inputs(inputs, recursive)
