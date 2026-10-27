"""Non-destructive edit pipeline: ordered steps (adjustment layers) rendered on demand.

A pipeline is a JSON list of steps::

    {"id": "s1a2b3", "op": "adjust.exposure", "params": {"ev": 0.3},
     "enabled": true, "opacity": 1.0, "blend": "normal", "mask": null,
     "label": "Brighten", "origin": "agent"}

``op: "group"`` steps contain a nested ``steps`` list rendered as one unit
(sharing one mask / opacity — e.g. a "look" with a strength slider).
Sources are never modified; rendering produces new buffers.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .blend import BLEND_MODES, mix_step
from .context import RenderContext
from .errors import ErrorCode, LazyRoomError, invalid
from .masks import render_mask, validate_mask
from .registry import get_op

GROUP_OP = "group"
ORIGINS = ("user", "agent", "preset", "look", "style_match", "intent", "import", "system")


def new_id(prefix: str = "s") -> str:
    return prefix + uuid.uuid4().hex[:8]


@dataclass
class Step:
    op: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=new_id)
    enabled: bool = True
    opacity: float = 1.0
    blend: str = "normal"
    mask: dict[str, Any] | None = None
    label: str | None = None
    origin: str = "user"
    steps: list["Step"] | None = None  # for groups

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any], *, validate: bool = True) -> "Step":
        if not isinstance(d, dict):
            raise invalid(f"Step must be an object, got {type(d).__name__}")
        known = {"op", "params", "id", "enabled", "opacity", "blend", "mask", "label", "origin", "steps"}
        unknown = set(d) - known
        if unknown:
            raise invalid(f"Unknown step fields {sorted(unknown)}", valid=sorted(known))
        if "op" not in d:
            raise invalid("Step requires an 'op' field")
        origin = str(d.get("origin", "user"))
        if origin.split(":", 1)[0] not in ORIGINS:
            raise invalid(f"Unknown origin {origin!r}", valid=list(ORIGINS))
        s = cls(op=str(d["op"]), params=dict(d.get("params") or {}), id=str(d.get("id") or new_id()),
                enabled=bool(d.get("enabled", True)), opacity=float(d.get("opacity", 1.0)),
                blend=str(d.get("blend", "normal")), mask=d.get("mask"), label=d.get("label"),
                origin=origin)
        if s.op == GROUP_OP:
            sub = d.get("steps")
            if not isinstance(sub, list):
                raise invalid("Group step requires a 'steps' list")
            s.steps = [Step.from_dict(x, validate=validate) for x in sub]
            if s.params:
                raise invalid("Group steps take no params")
        if validate:
            s.validate()
        return s

    def validate(self) -> "Step":
        if not (0.0 <= self.opacity <= 1.0):
            raise invalid(f"Step opacity must be in 0..1, got {self.opacity}", step=self.id)
        if self.blend not in BLEND_MODES:
            raise invalid(f"Unknown blend mode {self.blend!r}", valid=list(BLEND_MODES), step=self.id)
        self.mask = validate_mask(self.mask)
        if self.op == GROUP_OP:
            for s in self.steps or []:
                s.validate()
            if any(get_op(s.op).geometry for s in self._flat() if s.op != GROUP_OP) and (
                    self.mask or self.opacity < 1 or self.blend != "normal"):
                raise invalid("A group containing geometry operations cannot be masked or blended", step=self.id)
            return self
        spec = get_op(self.op)
        self.op = spec.id  # canonicalise aliases
        values = spec.validate(self.params)
        self.params = spec.params_json(values)
        if spec.geometry and (self.mask or self.opacity < 1 or self.blend != "normal"):
            raise invalid(f"'{self.op}' changes geometry and cannot have a mask, opacity or blend mode",
                          step=self.id)
        return self

    def _flat(self) -> list["Step"]:
        out: list[Step] = []
        for s in self.steps or []:
            out.append(s)
            if s.op == GROUP_OP:
                out.extend(s._flat())
        return out

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "op": self.op}
        if self.op == GROUP_OP:
            d["steps"] = [s.to_dict() for s in self.steps or []]
        else:
            d["params"] = self.params
        d["enabled"] = self.enabled
        if self.opacity != 1.0:
            d["opacity"] = self.opacity
        if self.blend != "normal":
            d["blend"] = self.blend
        if self.mask:
            d["mask"] = self.mask
        if self.label:
            d["label"] = self.label
        d["origin"] = self.origin
        return d

    def fingerprint(self) -> str:
        d = self.to_dict()
        d.pop("label", None)
        d.pop("origin", None)
        return hashlib.blake2b(json.dumps(d, sort_keys=True, default=str).encode(), digest_size=12).hexdigest()

    @property
    def is_geometry(self) -> bool:
        if self.op == GROUP_OP:
            return any(s.is_geometry for s in self.steps or [])
        return get_op(self.op).geometry


def steps_from_list(items: list[dict[str, Any]] | None) -> list[Step]:
    return [Step.from_dict(d) for d in (items or [])]


def steps_to_list(steps: list[Step]) -> list[dict[str, Any]]:
    return [s.to_dict() for s in steps]


def find_step(steps: list[Step], step_id: str) -> tuple[list[Step], int]:
    """Locate a step (also inside groups). Returns (containing list, index)."""
    for i, s in enumerate(steps):
        if s.id == step_id:
            return steps, i
        if s.op == GROUP_OP and s.steps:
            try:
                return find_step(s.steps, step_id)
            except LazyRoomError:
                pass
    raise LazyRoomError(ErrorCode.STEP_NOT_FOUND, f"No step with id '{step_id}'",
                          details={"step_ids": [s.id for s in steps]})


# ---------------------------------------------------------------------------
# Render cache
# ---------------------------------------------------------------------------
class RenderCache:
    """Thread-safe LRU of intermediate buffers keyed by pipeline-prefix hashes."""

    def __init__(self, max_bytes: int = 768 * 1024 * 1024) -> None:
        self.max_bytes = max_bytes
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
                self.hits += 1
            else:
                self.misses += 1
            return v

    def put(self, key: str, value: np.ndarray) -> None:
        nb = value.nbytes
        if nb > self.max_bytes // 4:
            return
        with self._lock:
            if key in self._d:
                return
            value.setflags(write=False)
            self._d[key] = value
            self._bytes += nb
            while self._bytes > self.max_bytes and self._d:
                _, old = self._d.popitem(last=False)
                self._bytes -= old.nbytes

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._bytes = 0

    def info(self) -> dict[str, Any]:
        return {"entries": len(self._d), "bytes": self._bytes, "max_bytes": self.max_bytes,
                "hits": self.hits, "misses": self.misses}


GLOBAL_CACHE = RenderCache()


@dataclass
class StepReport:
    step_id: str
    op: str
    ms: float
    cached: bool
    skipped: bool = False
    size: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"step_id": self.step_id, "op": self.op, "ms": round(self.ms, 2), "cached": self.cached}
        if self.skipped:
            d["skipped"] = True
        if self.size:
            d["size"] = list(self.size)
        return d


def _chain_key(prev: str, step: Step) -> str:
    return hashlib.blake2b((prev + "|" + step.fingerprint()).encode(), digest_size=16).hexdigest()


def run_step(img: np.ndarray, step: Step, ctx: RenderContext) -> np.ndarray:
    sctx = ctx.derive(step_id=step.id)
    if step.op == GROUP_OP:
        out = img
        for s in step.steps or []:
            if s.enabled:
                out = run_step(out, s, sctx)
    else:
        spec = get_op(step.op)
        values = spec.validate(step.params)
        try:
            out = spec.run(img, sctx, values)
        except LazyRoomError:
            raise
        except MemoryError:
            raise
        except Exception as exc:
            raise LazyRoomError(ErrorCode.OPERATION_FAILED, f"Operation '{step.op}' failed: {exc}",
                                  details={"step_id": step.id, "op": step.op,
                                           "exception_type": type(exc).__name__}) from exc
    if out.shape[:2] != img.shape[:2]:
        return out  # geometry change: no mixing possible
    if step.mask or step.opacity < 1.0 or step.blend != "normal":
        mask = render_mask(step.mask, img, sctx)
        out = mix_step(img, out, step.blend, step.opacity, mask)
    return out


def render(source: np.ndarray, steps: list[Step], ctx: RenderContext, *, cache: RenderCache | None = None,
           base_key: str | None = None, upto: int | None = None,
           progress: Callable[[int, int, Step], None] | None = None) -> tuple[np.ndarray, list[StepReport]]:
    """Render ``steps`` (optionally only the first ``upto``) on ``source``.

    When ``cache`` and ``base_key`` are given, intermediate results are reused
    so that editing step *k* only re-renders steps *k..n*.
    """
    active = steps if upto is None else steps[:upto]
    reports: list[StepReport] = []
    keys: list[str] = []
    k = base_key or ""
    for s in active:
        if s.enabled:
            k = _chain_key(k, s)
        keys.append(k)
    img = source
    start = 0
    if cache is not None and base_key:
        for i in range(len(active) - 1, -1, -1):
            if not active[i].enabled:
                continue
            hit = cache.get(keys[i])
            if hit is not None:
                img = hit
                start = i + 1
                for j in range(start):
                    reports.append(StepReport(active[j].id, active[j].op, 0.0, True, skipped=not active[j].enabled))
                break
    for i in range(start, len(active)):
        s = active[i]
        ctx.check()
        if not s.enabled:
            reports.append(StepReport(s.id, s.op, 0.0, False, skipped=True))
            continue
        if progress:
            progress(i, len(active), s)
        t0 = time.perf_counter()
        img = run_step(img, s, ctx)
        ms = (time.perf_counter() - t0) * 1000
        reports.append(StepReport(s.id, s.op, ms, False, size=(int(img.shape[1]), int(img.shape[0]))))
        if cache is not None and base_key:
            cache.put(keys[i], img)
    return img, reports


def geometry_scale(steps: list[Step]) -> bool:
    return any(s.enabled and s.is_geometry for s in steps)
