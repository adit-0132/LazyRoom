"""Presets: built-in semantic looks (read-only, shipped as JSON) + user presets.

Preset file format (``lazyroom.preset/1``)::

    {"schema": "lazyroom.preset/1", "id": "warm-70s-film", "name": "Warm 70s film",
     "description": "...", "tags": ["vintage", "warm", "film"], "steps": [ ...step dicts... ]}
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ..core.errors import ErrorCode, LazyRoomError, invalid
from ..core.paths import ensure, package_data_dir, presets_dir
from ..core.pipeline import steps_from_list, steps_to_list

SCHEMA = "lazyroom.preset/1"


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:60] or "preset"


class PresetStore:
    def __init__(self, user_dir: Path | None = None, builtin_dir: Path | None = None) -> None:
        self.user_dir = user_dir or presets_dir()
        self.builtin_dir = builtin_dir or (package_data_dir() / "looks")

    def _load_dir(self, d: Path, kind: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if not d.exists():
            return out
        for f in sorted(d.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if data.get("schema") != SCHEMA or "id" not in data:
                continue
            data["kind"] = kind
            data["file"] = str(f)
            out[data["id"]] = data
        return out

    def all(self) -> dict[str, dict[str, Any]]:
        d = self._load_dir(self.builtin_dir, "look")
        d.update({k: v for k, v in self._load_dir(self.user_dir, "user").items()})
        return d

    def list(self, query: str | None = None, kind: str = "all") -> list[dict[str, Any]]:
        items = self.all().values()
        if kind != "all":
            items = [p for p in items if p["kind"] == kind]
        if query:
            words = [w for w in re.findall(r"[a-z0-9]+", query.lower())]
            def score(p: dict[str, Any]) -> int:
                hay = " ".join([p["id"], p.get("name", ""), p.get("description", ""), " ".join(p.get("tags", []))]).lower()
                return sum(w in hay for w in words)
            items = sorted((p for p in items if score(p) > 0), key=lambda p: -score(p))
        return [{"id": p["id"], "name": p.get("name", p["id"]), "kind": p["kind"],
                 "description": p.get("description", ""), "tags": p.get("tags", []),
                 "steps": len(p.get("steps", []))} for p in items]

    def get(self, preset_id: str) -> dict[str, Any]:
        p = self.all().get(preset_id)
        if p is None:
            import difflib

            raise LazyRoomError(ErrorCode.PRESET_NOT_FOUND, f"No preset '{preset_id}'",
                                  details={"suggestions": difflib.get_close_matches(preset_id, list(self.all()), 5)},
                                  hint="Use presets_list to see available looks and presets.")
        steps_from_list(p.get("steps", []))  # validate
        return {k: v for k, v in p.items() if k != "file"}

    def save(self, name: str, steps: list[dict[str, Any]], *, description: str = "", tags: list[str] | None = None,
             overwrite: bool = False, preset_id: str | None = None) -> dict[str, Any]:
        if not name.strip():
            raise invalid("Preset name must not be empty")
        steps = steps_to_list(steps_from_list(steps))
        pid = preset_id or slugify(name)
        existing = self.all().get(pid)
        if existing and existing["kind"] == "look":
            pid = pid + "-custom"
            existing = self.all().get(pid)
        if existing and not overwrite:
            raise LazyRoomError(ErrorCode.OUTPUT_EXISTS, f"Preset '{pid}' already exists",
                                  hint="Pass overwrite=true or choose another name.")
        data = {"schema": SCHEMA, "id": pid, "name": name, "description": description, "tags": tags or [],
                "created": time.time(), "steps": steps}
        f = ensure(self.user_dir) / f"{pid}.json"
        f.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        return {"id": pid, "path": str(f), "steps": len(steps)}

    def delete(self, preset_id: str) -> dict[str, Any]:
        p = self.all().get(preset_id)
        if p is None:
            raise LazyRoomError(ErrorCode.PRESET_NOT_FOUND, f"No preset '{preset_id}'")
        if p["kind"] != "user":
            raise LazyRoomError(ErrorCode.PATH_NOT_ALLOWED, "Built-in looks cannot be deleted")
        Path(p["file"]).unlink()
        return {"deleted": preset_id}
