"""In-memory workspace: open project sessions, render cache, artifacts, jobs, log."""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ..core.context import CancelToken, RenderContext
from ..core.errors import ErrorCode, LazyRoomError
from ..core.image import fit_within, size_of
from ..core.io import ImageMeta, file_sha256, load_image
from ..core.paths import artifacts_dir, data_dir, ensure
from ..core.pipeline import RenderCache, Step, render
from ..core.project import Project, SourceRef
from .config import Config
from .security import PathPolicy


class OperationLog:
    """Append-only JSONL log of every command (for observability / audit)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (data_dir() / "logs" / "operations.jsonl")
        self._lock = threading.Lock()

    def write(self, entry: dict[str, Any]) -> None:
        try:
            ensure(self.path.parent)
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out


class ArtifactStore:
    """Preview/compare images written for agents to look at (bounded, auto-pruned)."""

    def __init__(self, root: Path | None = None, keep: int = 300) -> None:
        self.root = ensure(root or artifacts_dir())
        self.keep = keep
        self._lock = threading.Lock()

    def new_path(self, prefix: str, ext: str = ".jpg") -> Path:
        with self._lock:
            name = f"{time.strftime('%Y%m%d-%H%M%S')}-{prefix}-{uuid.uuid4().hex[:6]}{ext}"
            self._prune()
            return self.root / name

    def _prune(self) -> None:
        files = sorted(self.root.glob("*"), key=lambda p: p.stat().st_mtime)
        for p in files[: max(0, len(files) - self.keep)]:
            try:
                p.unlink()
            except OSError:
                pass


class ProjectSession:
    """A loaded project with its decoded source and preview proxies."""

    def __init__(self, project: Project, ws: "Workspace") -> None:
        self.project = project
        self.ws = ws
        self.lock = threading.RLock()
        self._source: np.ndarray | None = None
        self._meta: ImageMeta | None = None
        self._proxies: dict[int, tuple[np.ndarray, float]] = {}
        self.warnings: list[str] = []
        self.listeners: list[Any] = []  # GUI callbacks on change

    @property
    def id(self) -> str:
        return self.project.id

    def source(self) -> tuple[np.ndarray, ImageMeta]:
        with self.lock:
            if self._source is None:
                path = self.ws.policy.check_read(self.project.source.path)
                img, meta = load_image(path, max_pixels=self.ws.config.max_pixels)
                img.setflags(write=False)
                self._source, self._meta = img, meta
            assert self._meta is not None
            return self._source, self._meta

    def proxy(self, max_size: int) -> tuple[np.ndarray, float]:
        with self.lock:
            src, _ = self.source()
            key = int(max_size)
            if key not in self._proxies:
                img, s = fit_within(src, key)
                img = np.ascontiguousarray(img)
                img.setflags(write=False)
                if len(self._proxies) > 4:
                    self._proxies.pop(next(iter(self._proxies)))
                self._proxies[key] = (img, s)
            return self._proxies[key]

    def render(self, *, max_size: int | None = None, steps: list[Step] | None = None, upto: int | None = None,
               cancel: CancelToken | None = None, use_cache: bool = True) -> tuple[np.ndarray, list, float]:
        """Render the pipeline. ``max_size`` None = full resolution. Returns (img, reports, scale)."""
        steps = self.project.steps if steps is None else steps
        if max_size:
            base, scale = self.proxy(max_size)
        else:
            base, scale = self.source()[0], 1.0
        ctx = RenderContext(scale=scale, seed=self.project.seed, preview=bool(max_size),
                            cancel=cancel or CancelToken(timeout=self.ws.config.render_timeout_s),
                            resolver=self.ws.resolve_aux_file)
        key = f"{self.project.source.sha256 or self.project.source.path}|{base.shape}|{scale:.6f}|{self.project.seed}"
        out, reports = render(base, steps, ctx, cache=self.ws.cache if use_cache else None, base_key=key, upto=upto)
        return out, reports, scale

    def notify(self, event: str) -> None:
        for cb in list(self.listeners):
            try:
                cb(event, self)
            except Exception:
                pass


class Workspace:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.load()
        self.policy = PathPolicy(self.config.read_roots, self.config.write_roots)
        self.cache = RenderCache(self.config.cache_mb * 1024 * 1024)
        self.artifacts = ArtifactStore(Path(self.config.artifact_dir) if self.config.artifact_dir else None)
        self.log = OperationLog()
        self.sessions: dict[str, ProjectSession] = {}
        self._lock = threading.RLock()
        from .jobs import JobManager

        self.jobs = JobManager(self)

    # ------------------------------------------------------------------
    def resolve_aux_file(self, path: str, kind: str) -> Path:
        p = self.policy.check_read(path)
        if not p.exists():
            raise LazyRoomError(ErrorCode.FILE_NOT_FOUND, f"{kind} file not found: {path}")
        return p

    def session(self, project_id: str) -> ProjectSession:
        s = self.sessions.get(project_id)
        if s is None:
            raise LazyRoomError(ErrorCode.PROJECT_NOT_FOUND, f"No open project with id '{project_id}'",
                                  details={"open_projects": list(self.sessions)},
                                  hint="Use project_create or project_open first (ids are per server session).")
        return s

    def create_project(self, source: str, name: str | None = None, steps: list[Step] | None = None) -> ProjectSession:
        path = self.policy.check_read(source)
        if not path.exists():
            raise LazyRoomError(ErrorCode.FILE_NOT_FOUND, f"Image not found: {source}")
        from ..core.io import probe

        meta = probe(path)
        ref = SourceRef(path=str(path), sha256=file_sha256(path), width=meta.width, height=meta.height,
                        format=meta.format, file_size=meta.file_size)
        proj = Project.new(ref, name=name, steps=steps)
        return self._add(proj)

    def open_project(self, path: str) -> ProjectSession:
        p = self.policy.check_read(path)
        proj = Project.load(p)
        for s in self.sessions.values():
            if s.project.path and Path(s.project.path) == Path(proj.path or ""):
                return s
        sess = self._add(proj)
        src = Path(proj.source.path)
        if not src.exists():
            sess.warnings.append(f"Source image is missing: {src}")
        elif proj.source.sha256 and file_sha256(src) != proj.source.sha256:
            sess.warnings.append("Source image changed on disk since the project was created")
        return sess

    def _add(self, proj: Project) -> ProjectSession:
        with self._lock:
            sess = ProjectSession(proj, self)
            self.sessions[proj.id] = sess
            self.policy.protect(proj.source.path)
            return sess

    def close(self, project_id: str) -> None:
        with self._lock:
            self.sessions.pop(project_id, None)
