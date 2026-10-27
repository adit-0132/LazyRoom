"""Operation registry — the single source of truth for every editing capability.

Each operation is a pure function ``fn(img, ctx, **params) -> img`` plus a
declarative :class:`OpSpec`.  The CLI, MCP server, SDK and GUI all discover
operations exclusively through this registry, so a capability registered here
is automatically available to every client.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from . import backends
from .context import RenderContext
from .errors import ErrorCode, LazyRoomError
from .image import has_alpha, merge_alpha, split_alpha
from .params import Param, params_schema, params_to_json, validate_params

OpFn = Callable[..., np.ndarray]

# Top-level categories (namespaces of operation ids).
CATEGORIES: dict[str, str] = {
    "adjust": "Photographic tone & colour adjustments (exposure, white balance, curves, HSL, grading)",
    "effect": "Photographic effects (grain, vignette, bloom, sharpening, denoise, blur)",
    "geometry": "Crop, resize, rotate, flip, straighten, perspective, canvas",
    "draw": "Text, shapes, watermarks, image overlays, annotations",
    "filter": "ImageToolbox filter catalogue (500+ filters in 14 groups)",
    "ai": "Machine-learning operations (background removal, upscaling, depth...) — optional models",
    "generate": "Procedural content (gradients, noise, textures) composited into the image",
}

# ImageToolbox filter groups (used as ``group`` for the ``filter`` category).
FILTER_GROUPS: dict[str, str] = {
    "simple": "Simple effects", "color": "Color", "lut": "LUT", "light": "Light",
    "enhancement": "Enhancement", "artistic": "Artistic", "frames": "Frames", "effects": "Effects",
    "blur": "Blur", "pixelation": "Pixelation", "distortion": "Distortion", "glitch": "Glitch",
    "geometry": "Geometry", "dithering": "Dithering",
}


@dataclass
class OpSpec:
    id: str
    title: str
    fn: OpFn
    params: list[Param] = field(default_factory=list)
    category: str = "filter"
    group: str | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    alpha: str = "preserve"          # "preserve": fn gets RGB, alpha re-attached; "full": fn handles RGBA
    geometry: bool = False           # changes size/position of pixels -> cannot be masked/blended
    requires: list[str] = field(default_factory=list)
    deterministic: bool = True
    source: dict[str, Any] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)
    version: int = 1
    aliases: list[str] = field(default_factory=list)
    hidden: bool = False

    # ------------------------------------------------------------------
    def validate(self, values: dict[str, Any] | None) -> dict[str, Any]:
        return validate_params(self.params, values, op_id=self.id)

    def params_json(self, values: dict[str, Any]) -> dict[str, Any]:
        return params_to_json(self.params, values)

    def defaults(self) -> dict[str, Any]:
        return self.params_json(self.validate({}))

    def availability(self) -> tuple[bool, str | None]:
        for b in self.requires:
            if not backends.is_available(b):
                return False, backends.hint_for(b)
        return True, None

    def run(self, img: np.ndarray, ctx: RenderContext, values: dict[str, Any]) -> np.ndarray:
        for b in self.requires:
            backends.require(b)
        ctx.check()
        if self.alpha == "preserve" and has_alpha(img):
            rgb, a = split_alpha(img)
            out = self.fn(np.ascontiguousarray(rgb), ctx, **values)
            out = _check_output(self.id, out)
            if self.geometry and out.shape[:2] != rgb.shape[:2]:
                raise LazyRoomError(ErrorCode.INTERNAL, f"{self.id}: geometry op must use alpha='full'")
            if out.shape[2] == 4:  # op produced its own alpha: combine
                return np.dstack([out[..., :3], out[..., 3] * a]).astype(np.float32)
            return merge_alpha(out, a)
        out = self.fn(img, ctx, **values)
        return _check_output(self.id, out)

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "title": self.title}
        if self.group:
            d["group"] = self.group
        ok, _ = self.availability()
        if not ok:
            d["available"] = False
        return d

    def describe(self) -> dict[str, Any]:
        ok, hint = self.availability()
        d: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "group": self.group,
            "description": self.description,
            "version": self.version,
            "params": [p.describe() for p in self.params],
            "input_schema": params_schema(self.params),
            "defaults": self.defaults(),
            "geometry": self.geometry,
            "maskable": not self.geometry,
            "alpha": self.alpha,
            "deterministic": self.deterministic,
            "available": ok,
        }
        if self.tags:
            d["tags"] = self.tags
        if self.requires:
            d["requires"] = self.requires
        if hint:
            d["install_hint"] = hint
        if self.source:
            d["source"] = self.source
        if self.examples:
            d["examples"] = self.examples
        if self.aliases:
            d["aliases"] = self.aliases
        return d


def _check_output(op_id: str, out: Any) -> np.ndarray:
    if not isinstance(out, np.ndarray):
        raise LazyRoomError(ErrorCode.INTERNAL, f"Operation {op_id} returned {type(out).__name__}, not an image")
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    if out.ndim != 3 or out.shape[2] not in (3, 4) or out.shape[0] < 1 or out.shape[1] < 1:
        raise LazyRoomError(ErrorCode.INTERNAL, f"Operation {op_id} returned invalid shape {out.shape}")
    if out.dtype != np.float32:
        out = out.astype(np.float32)
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
REGISTRY: dict[str, OpSpec] = {}
_ALIASES: dict[str, str] = {}
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")


def register(spec: OpSpec) -> OpSpec:
    if not _ID_RE.match(spec.id):
        raise ValueError(f"Invalid operation id {spec.id!r}")
    if spec.id in REGISTRY:
        raise ValueError(f"Duplicate operation id {spec.id!r}")
    if spec.category not in CATEGORIES:
        raise ValueError(f"Unknown category {spec.category!r} for {spec.id}")
    names = [p.name for p in spec.params]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate parameter names in {spec.id}")
    REGISTRY[spec.id] = spec
    for a in spec.aliases:
        _ALIASES[a] = spec.id
    return spec


def op(id: str, title: str, *, params: Sequence[Param] = (), **kwargs: Any) -> Callable[[OpFn], OpFn]:
    """Decorator registering an operation.  Category defaults to the id prefix."""

    def deco(fn: OpFn) -> OpFn:
        category = kwargs.pop("category", id.split(".", 1)[0])
        desc = kwargs.pop("description", None) or (fn.__doc__ or "").strip()
        register(OpSpec(id=id, title=title, fn=fn, params=list(params), category=category,
                        description=desc, **kwargs))
        return fn

    return deco


def ensure_loaded() -> None:
    """Import all built-in operation modules (idempotent)."""
    from .. import ops  # noqa: F401  (registers on import)


def get_op(op_id: str) -> OpSpec:
    ensure_loaded()
    spec = REGISTRY.get(op_id) or REGISTRY.get(_ALIASES.get(op_id, ""))
    if spec is None:
        close = difflib.get_close_matches(op_id, list(REGISTRY), n=5, cutoff=0.5)
        if not close:
            q = op_id.split(".")[-1]
            close = [s.id for s in search_ops(q, limit=5)]
        raise LazyRoomError(ErrorCode.UNKNOWN_OPERATION, f"Unknown operation '{op_id}'",
                              details={"suggestions": close},
                              hint="Use list_operations / 'lazyroom ops list' to discover valid ids.")
    return spec


def all_ops(include_hidden: bool = False) -> list[OpSpec]:
    ensure_loaded()
    return [s for s in REGISTRY.values() if include_hidden or not s.hidden]


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def search_ops(query: str, *, category: str | None = None, group: str | None = None,
               limit: int = 50) -> list[OpSpec]:
    ensure_loaded()
    q = _tokens(query)
    scored: list[tuple[float, OpSpec]] = []
    for s in REGISTRY.values():
        if s.hidden or (category and s.category != category) or (group and s.group != group):
            continue
        hay_id = _tokens(s.id.replace(".", " ") + " " + s.title)
        hay = hay_id | _tokens(" ".join(s.tags)) | _tokens(s.description[:300])
        if not q:
            scored.append((0.0, s))
            continue
        score = 0.0
        for t in q:
            if t in hay_id:
                score += 3
            elif t in hay:
                score += 1
            elif any(h.startswith(t) for h in hay_id):
                score += 1.5
            elif any(h.startswith(t) for h in hay):
                score += 0.5
        if score > 0:
            scored.append((score, s))
    scored.sort(key=lambda t: (-t[0], t[1].id))
    return [s for _, s in scored[:limit]]


def list_ops(category: str | None = None, group: str | None = None) -> list[OpSpec]:
    ensure_loaded()
    return sorted((s for s in REGISTRY.values()
                   if not s.hidden and (category is None or s.category == category)
                   and (group is None or s.group == group)), key=lambda s: s.id)


def catalog_index() -> dict[str, Any]:
    """Compact hierarchical index for agents: category -> group -> [ids]."""
    ensure_loaded()
    out: dict[str, Any] = {}
    for s in sorted(REGISTRY.values(), key=lambda s: s.id):
        if s.hidden:
            continue
        cat = out.setdefault(s.category, {"description": CATEGORIES[s.category], "count": 0, "groups": {}})
        cat["count"] += 1
        cat["groups"].setdefault(s.group or "general", []).append(s.id)
    return out


def iter_specs(ids: Iterable[str]) -> list[OpSpec]:
    return [get_op(i) for i in ids]
