"""Background jobs (batch processing) with progress, cancellation and reports.

A job never blocks the caller: ``submit`` returns immediately with a job id.
Each input file is an item with its own status; failures are isolated so a
corrupt file never aborts the rest of the batch.  A JSON report is written
next to the outputs when the job ends.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.context import CancelToken
from ..core.errors import ErrorCode, LazyRoomError, wrap_exception

TERMINAL = ("completed", "partial", "failed", "cancelled")


@dataclass
class JobItem:
    index: int
    input: str
    output: str | None = None
    status: str = "pending"   # pending | running | ok | failed | skipped | cancelled
    error: dict[str, Any] | None = None
    ms: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"index": self.index, "input": self.input, "output": self.output, "status": self.status,
             "ms": round(self.ms, 1)}
        if self.error:
            d["error"] = self.error
        if self.info:
            d["info"] = self.info
        return d


@dataclass
class Job:
    id: str
    kind: str
    params: dict[str, Any]
    items: list[JobItem]
    status: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    report_path: str | None = None
    error: dict[str, Any] | None = None
    cancel: CancelToken = field(default_factory=CancelToken)
    done_event: threading.Event = field(default_factory=threading.Event)

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for it in self.items:
            c[it.status] = c.get(it.status, 0) + 1
        return c

    def summary(self, items: bool = False, max_items: int = 200) -> dict[str, Any]:
        c = self.counts()
        total = len(self.items)
        finished = sum(c.get(k, 0) for k in ("ok", "failed", "skipped", "cancelled"))
        d: dict[str, Any] = {
            "job_id": self.id, "kind": self.kind, "status": self.status, "total": total,
            "succeeded": c.get("ok", 0), "failed": c.get("failed", 0), "skipped": c.get("skipped", 0),
            "cancelled": c.get("cancelled", 0), "progress": round(finished / total, 4) if total else 1.0,
            "created": self.created, "started": self.started, "finished": self.finished,
            "elapsed_s": round(((self.finished or time.time()) - (self.started or self.created)), 2),
        }
        if self.report_path:
            d["report_path"] = self.report_path
        if self.error:
            d["error"] = self.error
        if items:
            d["items"] = [it.to_dict() for it in self.items[:max_items]]
            if total > max_items:
                d["items_truncated"] = total - max_items
        else:
            fails = [it.to_dict() for it in self.items if it.status == "failed"][:20]
            if fails:
                d["failures"] = fails
        return d


ItemFn = Callable[[JobItem, CancelToken], None]


class JobManager:
    def __init__(self, ws: Any, max_jobs: int = 2) -> None:
        self.ws = ws
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._runner = ThreadPoolExecutor(max_workers=max_jobs, thread_name_prefix="pm-job")

    def submit(self, kind: str, inputs: list[str], fn: ItemFn, *, params: dict[str, Any] | None = None,
               outputs: list[str | None] | None = None, workers: int = 0, report_dir: str | None = None) -> Job:
        items = [JobItem(index=i, input=p, output=(outputs[i] if outputs else None)) for i, p in enumerate(inputs)]
        job = Job(id="j" + uuid.uuid4().hex[:10], kind=kind, params=params or {}, items=items)
        with self._lock:
            self.jobs[job.id] = job
        nworkers = workers or self.ws.config.batch_workers or max(1, min(4, (os.cpu_count() or 2) // 2))
        self._runner.submit(self._run, job, fn, nworkers, report_dir)
        return job

    def _run(self, job: Job, fn: ItemFn, nworkers: int, report_dir: str | None) -> None:
        job.status = "running"
        job.started = time.time()

        def one(item: JobItem) -> None:
            if job.cancel.cancelled:
                item.status = "cancelled"
                return
            item.status = "running"
            t0 = time.perf_counter()
            try:
                fn(item, job.cancel)
                if item.status == "running":
                    item.status = "ok"
            except LazyRoomError as e:
                item.status = "cancelled" if e.code == ErrorCode.CANCELLED else "failed"
                item.error = e.to_dict()
            except Exception as e:  # never let one file kill the batch
                item.status = "failed"
                err = wrap_exception(e).to_dict()
                err["traceback"] = traceback.format_exc(limit=4)
                item.error = err
            finally:
                item.ms = (time.perf_counter() - t0) * 1000

        try:
            with ThreadPoolExecutor(max_workers=nworkers, thread_name_prefix="pm-item") as pool:
                list(pool.map(one, job.items))
            c = job.counts()
            if job.cancel.cancelled:
                job.status = "cancelled"
            elif c.get("failed", 0) and c.get("ok", 0):
                job.status = "partial"
            elif c.get("failed", 0) and not c.get("ok", 0) and not c.get("skipped", 0):
                job.status = "failed"
            else:
                job.status = "completed"
        except Exception as e:  # pragma: no cover - defensive
            job.status = "failed"
            job.error = wrap_exception(e).to_dict()
        finally:
            job.finished = time.time()
            if report_dir:
                try:
                    p = Path(report_dir) / f"lazyroom-report-{job.id}.json"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps({**job.summary(items=True, max_items=10**9), "params": job.params},
                                            indent=1, default=str), encoding="utf-8")
                    job.report_path = str(p)
                except OSError:
                    pass
            self.ws.log.write({"t": time.time(), "event": "job_finished", **job.summary()})
            job.done_event.set()

    def get(self, job_id: str) -> Job:
        j = self.jobs.get(job_id)
        if j is None:
            raise LazyRoomError(ErrorCode.JOB_NOT_FOUND, f"No job with id '{job_id}'",
                                  details={"jobs": list(self.jobs)[-20:]})
        return j

    def cancel(self, job_id: str) -> Job:
        j = self.get(job_id)
        if j.status not in TERMINAL:
            j.cancel.cancel("cancelled by request")
        return j

    def wait(self, job_id: str, timeout: float | None = None) -> Job:
        j = self.get(job_id)
        j.done_event.wait(timeout)
        return j

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created)
