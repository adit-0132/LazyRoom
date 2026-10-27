"""Project model: immutable source reference + edit pipeline + history.

Projects are stored as human-readable JSON (``*.lazyroom``), schema
``lazyroom.project/1`` (see docs/PROJECT_FORMAT.md).  Every mutation goes
through :meth:`Project.apply_actions`, which validates a batch of edit actions
transactionally and records a single history entry — so agents and humans
share the same undo/redo stack.
"""
from __future__ import annotations

import copy
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ErrorCode, LazyRoomError, invalid
from .pipeline import GROUP_OP, Step, find_step, steps_from_list, steps_to_list, new_id

SCHEMA = "lazyroom.project/1"
LEGACY_SCHEMA_PREFIX = "photomasta.project/"  # projects saved before the rename to LazyRoom (*.pmproj)
PROJECT_SUFFIXES = (".lazyroom", ".pmproj")
MAX_HISTORY = 1000

ACTION_TYPES = {
    "add": "Add a step: {op, params?, index?, mask?, opacity?, blend?, label?, enabled?, group_id?}",
    "add_group": "Add a group step: {steps:[...], label?, mask?, opacity?, index?}",
    "update": "Update a step: {step_id, params? (merged), replace_params?, mask?, opacity?, blend?, label?, enabled?}",
    "remove": "Remove a step: {step_id}",
    "move": "Move a step to a new index: {step_id, index}",
    "toggle": "Enable/disable a step: {step_id, enabled?} (flips when 'enabled' omitted)",
    "duplicate": "Duplicate a step: {step_id}",
    "set_steps": "Replace the whole pipeline: {steps:[...]}",
    "clear": "Remove all steps: {}",
}


def _now() -> float:
    return round(time.time(), 3)


@dataclass
class HistoryEntry:
    id: str
    time: float
    action: str
    description: str
    actor: str
    steps: list[dict[str, Any]]

    def to_dict(self, with_steps: bool = True) -> dict[str, Any]:
        d = {"id": self.id, "time": self.time, "action": self.action, "description": self.description,
             "actor": self.actor, "step_count": len(self.steps)}
        if with_steps:
            d["steps"] = self.steps
        return d


@dataclass
class History:
    entries: list[HistoryEntry] = field(default_factory=list)
    cursor: int = -1  # index of the entry matching the current state

    def record(self, action: str, description: str, actor: str, steps: list[dict[str, Any]]) -> HistoryEntry:
        # drop redo branch
        self.entries = self.entries[: self.cursor + 1]
        e = HistoryEntry(id=new_id("h"), time=_now(), action=action, description=description, actor=actor,
                         steps=copy.deepcopy(steps))
        self.entries.append(e)
        if len(self.entries) > MAX_HISTORY:
            self.entries = self.entries[-MAX_HISTORY:]
        self.cursor = len(self.entries) - 1
        return e

    @property
    def can_undo(self) -> bool:
        return self.cursor > 0

    @property
    def can_redo(self) -> bool:
        return self.cursor < len(self.entries) - 1

    def to_dict(self) -> dict[str, Any]:
        return {"cursor": self.cursor, "entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "History":
        if not d:
            return cls()
        es = [HistoryEntry(**{k: e[k] for k in ("id", "time", "action", "description", "actor", "steps")})
              for e in d.get("entries", [])]
        return cls(entries=es, cursor=int(d.get("cursor", len(es) - 1)))


@dataclass
class SourceRef:
    path: str
    sha256: str | None = None
    width: int = 0
    height: int = 0
    format: str | None = None
    file_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Project:
    source: SourceRef
    steps: list[Step] = field(default_factory=list)
    id: str = field(default_factory=lambda: "p" + uuid.uuid4().hex[:10])
    name: str = "Untitled"
    created: float = field(default_factory=_now)
    modified: float = field(default_factory=_now)
    seed: int = 0
    export: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    history: History = field(default_factory=History)
    path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def new(cls, source: SourceRef, name: str | None = None, steps: list[Step] | None = None) -> "Project":
        p = cls(source=source, name=name or Path(source.path).stem, steps=list(steps or []))
        p.seed = int(uuid.uuid4().int % (2 ** 31))
        p.history.record("create", f"Opened {Path(source.path).name}", "system", steps_to_list(p.steps))
        return p

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "id": self.id,
            "name": self.name,
            "created": self.created,
            "modified": self.modified,
            "seed": self.seed,
            "source": self.source.to_dict(),
            "steps": steps_to_list(self.steps),
            "export": self.export,
            "notes": self.notes,
            "meta": self.meta,
            "history": self.history.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], path: str | None = None) -> "Project":
        schema = d.get("schema", "")
        if not str(schema).startswith(("lazyroom.project/", LEGACY_SCHEMA_PREFIX)):
            raise LazyRoomError(ErrorCode.SCHEMA_VERSION, f"Not a LazyRoom project (schema={schema!r})")
        major = str(schema).split("/", 1)[1]
        if major != "1":
            raise LazyRoomError(ErrorCode.SCHEMA_VERSION, f"Unsupported project schema version {major}",
                                  hint="Upgrade LazyRoom to open this project.")
        src = d.get("source") or {}
        p = cls(source=SourceRef(**{k: src.get(k) for k in SourceRef.__dataclass_fields__}),
                steps=steps_from_list(d.get("steps")), id=d.get("id") or "p" + uuid.uuid4().hex[:10],
                name=d.get("name", "Untitled"), created=d.get("created", _now()), modified=d.get("modified", _now()),
                seed=int(d.get("seed", 0)), export=d.get("export") or {}, notes=d.get("notes", ""),
                history=History.from_dict(d.get("history")), path=path, meta=d.get("meta") or {})
        if not p.history.entries:
            p.history.record("load", "Loaded project", "system", steps_to_list(p.steps))
        return p

    def save(self, path: str | os.PathLike | None = None) -> str:
        from .io import atomic_write

        target = Path(path or self.path or "")
        if not str(target):
            raise invalid("No path given to save the project")
        if target.suffix.lower() != ".lazyroom":
            target = target.with_suffix(".lazyroom")
        self.modified = _now()
        data = json.dumps(self.to_dict(), indent=1, ensure_ascii=False).encode("utf-8")
        atomic_write(target, data, overwrite=True)
        self.path = str(target)
        return self.path

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Project":
        p = Path(path)
        if not p.exists():
            raise LazyRoomError(ErrorCode.FILE_NOT_FOUND, f"Project file not found: {p}")
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LazyRoomError(ErrorCode.DECODE_FAILED, f"Project file is not valid JSON: {exc}") from exc
        proj = cls.from_dict(d, path=str(p))
        # resolve a relative source path against the project folder
        sp = Path(proj.source.path)
        if not sp.is_absolute():
            proj.source.path = str((p.parent / sp).resolve())
        return proj

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------
    def apply_actions(self, actions: list[dict[str, Any]], *, actor: str = "user",
                      description: str | None = None) -> dict[str, Any]:
        """Apply a batch of edit actions atomically.  Returns ids of created steps."""
        if not isinstance(actions, list) or not actions:
            raise invalid("'actions' must be a non-empty list", hint=f"Action types: {', '.join(ACTION_TYPES)}")
        work = [Step.from_dict(s.to_dict(), validate=False) for s in self.steps]
        created: list[str] = []
        descs: list[str] = []
        for i, a in enumerate(actions):
            try:
                created_id, desc = _apply_one(work, a, actor)
            except LazyRoomError as e:
                e.details.setdefault("action_index", i)
                raise
            if created_id:
                created.append(created_id)
            descs.append(desc)
        self.steps = work
        self.modified = _now()
        entry = self.history.record("edit", description or "; ".join(descs)[:500], actor, steps_to_list(self.steps))
        return {"created_step_ids": created, "history_id": entry.id, "step_count": len(self.steps)}

    def undo(self, n: int = 1) -> HistoryEntry:
        if not self.history.can_undo:
            raise LazyRoomError(ErrorCode.NOTHING_TO_UNDO, "Nothing to undo")
        self.history.cursor = max(0, self.history.cursor - max(1, n))
        e = self.history.entries[self.history.cursor]
        self.steps = steps_from_list(e.steps)
        return e

    def redo(self, n: int = 1) -> HistoryEntry:
        if not self.history.can_redo:
            raise LazyRoomError(ErrorCode.NOTHING_TO_REDO, "Nothing to redo")
        self.history.cursor = min(len(self.history.entries) - 1, self.history.cursor + max(1, n))
        e = self.history.entries[self.history.cursor]
        self.steps = steps_from_list(e.steps)
        return e

    def revert_to(self, history_id: str, actor: str = "user") -> HistoryEntry:
        for e in self.history.entries:
            if e.id == history_id:
                self.steps = steps_from_list(e.steps)
                return self.history.record("revert", f"Reverted to state '{e.description}'", actor,
                                           steps_to_list(self.steps))
        raise LazyRoomError(ErrorCode.STEP_NOT_FOUND, f"No history entry '{history_id}'")

    def steps_at(self, history_id: str) -> list[Step]:
        for e in self.history.entries:
            if e.id == history_id:
                return steps_from_list(e.steps)
        raise LazyRoomError(ErrorCode.STEP_NOT_FOUND, f"No history entry '{history_id}'")

    def state(self, *, history: bool = False) -> dict[str, Any]:
        d = {
            "project_id": self.id,
            "name": self.name,
            "path": self.path,
            "source": self.source.to_dict(),
            "steps": steps_to_list(self.steps),
            "can_undo": self.history.can_undo,
            "can_redo": self.history.can_redo,
            "history_cursor": self.history.cursor,
            "history_length": len(self.history.entries),
            "export": self.export,
            "notes": self.notes,
        }
        if history:
            d["history"] = [e.to_dict(with_steps=False) for e in self.history.entries]
        return d


def _step_kwargs(a: dict[str, Any], origin_default: str) -> dict[str, Any]:
    d = {k: a[k] for k in ("op", "params", "mask", "opacity", "blend", "label", "enabled", "id", "steps")
         if k in a}
    d["origin"] = a.get("origin", origin_default)
    return d


def _apply_one(work: list[Step], a: dict[str, Any], actor: str) -> tuple[str | None, str]:
    if not isinstance(a, dict) or "type" not in a:
        raise invalid("Each action needs a 'type'", valid_types=list(ACTION_TYPES))
    t = a["type"]
    origin = "agent" if actor == "agent" else "user"
    if t == "add":
        allowed = {"type", "op", "params", "index", "mask", "opacity", "blend", "label", "enabled", "origin",
                   "group_id", "id"}
        _reject_unknown(a, allowed, t)
        step = Step.from_dict(_step_kwargs(a, origin))
        target = work
        if a.get("group_id"):
            lst, i = find_step(work, a["group_id"])
            if lst[i].op != GROUP_OP:
                raise invalid(f"Step {a['group_id']} is not a group")
            target = lst[i].steps  # type: ignore[assignment]
        idx = a.get("index")
        target.insert(len(target) if idx is None else max(0, min(int(idx), len(target))), step)
        return step.id, f"Add {step.op}"
    if t == "add_group":
        _reject_unknown(a, {"type", "steps", "label", "mask", "opacity", "blend", "index", "enabled", "origin", "id"}, t)
        d = _step_kwargs(a, origin)
        d["op"] = GROUP_OP
        step = Step.from_dict(d)
        idx = a.get("index")
        work.insert(len(work) if idx is None else max(0, min(int(idx), len(work))), step)
        return step.id, f"Add group '{step.label or step.id}'"
    if t == "update":
        _reject_unknown(a, {"type", "step_id", "params", "replace_params", "mask", "opacity", "blend", "label",
                            "enabled"}, t)
        lst, i = find_step(work, _req(a, "step_id"))
        cur = lst[i].to_dict()
        if "params" in a:
            if lst[i].op == GROUP_OP:
                raise invalid("Group steps have no params; update the child steps instead")
            cur["params"] = dict(a["params"]) if a.get("replace_params") else {**cur.get("params", {}), **a["params"]}
        for k in ("mask", "opacity", "blend", "label", "enabled"):
            if k in a:
                cur[k] = a[k]
        lst[i] = Step.from_dict(cur)
        return None, f"Update {lst[i].op}"
    if t == "remove":
        _reject_unknown(a, {"type", "step_id"}, t)
        lst, i = find_step(work, _req(a, "step_id"))
        s = lst.pop(i)
        return None, f"Remove {s.op}"
    if t == "move":
        _reject_unknown(a, {"type", "step_id", "index"}, t)
        lst, i = find_step(work, _req(a, "step_id"))
        s = lst.pop(i)
        idx = max(0, min(int(_req(a, "index")), len(lst)))
        lst.insert(idx, s)
        return None, f"Move {s.op} to {idx}"
    if t == "toggle":
        _reject_unknown(a, {"type", "step_id", "enabled"}, t)
        lst, i = find_step(work, _req(a, "step_id"))
        lst[i].enabled = bool(a["enabled"]) if "enabled" in a else not lst[i].enabled
        return None, f"{'Enable' if lst[i].enabled else 'Disable'} {lst[i].op}"
    if t == "duplicate":
        _reject_unknown(a, {"type", "step_id"}, t)
        lst, i = find_step(work, _req(a, "step_id"))
        d = lst[i].to_dict()
        d.pop("id")
        if "steps" in d:
            d["steps"] = _strip_ids(d["steps"])
        s = Step.from_dict(d)
        lst.insert(i + 1, s)
        return s.id, f"Duplicate {s.op}"
    if t == "set_steps":
        _reject_unknown(a, {"type", "steps"}, t)
        items = _req(a, "steps")
        if not isinstance(items, list):
            raise invalid("'steps' must be a list")
        new = [Step.from_dict({**d, "origin": d.get("origin", origin)} if isinstance(d, dict) else d)
               for d in items]
        work[:] = new
        return None, f"Set pipeline ({len(new)} steps)"
    if t == "clear":
        work.clear()
        return None, "Clear pipeline"
    raise invalid(f"Unknown action type {t!r}", valid_types=list(ACTION_TYPES))


def _strip_ids(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for d in items:
        d = dict(d)
        d.pop("id", None)
        if "steps" in d:
            d["steps"] = _strip_ids(d["steps"])
        out.append(d)
    return out


def _req(a: dict[str, Any], k: str) -> Any:
    if k not in a:
        raise invalid(f"Action '{a.get('type')}' requires '{k}'")
    return a[k]


def _reject_unknown(a: dict[str, Any], allowed: set[str], t: str) -> None:
    extra = set(a) - allowed
    if extra:
        raise invalid(f"Unknown fields {sorted(extra)} for action '{t}'", valid=sorted(allowed))
