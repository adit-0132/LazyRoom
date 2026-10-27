"""Render context passed to every operation."""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .errors import ErrorCode, LazyRoomError


class CancelToken:
    """Cooperative cancellation + deadline shared by a render or a job."""

    def __init__(self, timeout: float | None = None) -> None:
        import time

        self._event = threading.Event()
        self._deadline = (time.monotonic() + timeout) if timeout else None
        self.reason: str | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        import time

        if self._event.is_set():
            raise LazyRoomError(ErrorCode.CANCELLED, f"Operation cancelled ({self.reason})")
        if self._deadline is not None and time.monotonic() > self._deadline:
            raise LazyRoomError(ErrorCode.TIMEOUT, "Operation exceeded its time limit",
                                  hint="Increase the timeout, render a smaller preview, or simplify the pipeline.")


FileResolver = Callable[[str, str], Path]


def default_resolver(path: str, kind: str) -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        raise LazyRoomError(ErrorCode.FILE_NOT_FOUND, f"{kind} file not found: {path}")
    return p.resolve()


@dataclass
class RenderContext:
    """Information an operation may need besides its parameters.

    ``scale`` is the ratio between the buffer being rendered and the
    full-resolution image (1.0 for exports, <1 for preview proxies).  Use
    :meth:`px` for any parameter measured in full-resolution pixels.
    """

    scale: float = 1.0
    seed: int = 0
    preview: bool = False
    cancel: CancelToken | None = None
    resolver: FileResolver = default_resolver
    step_id: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def px(self, value: float) -> float:
        return float(value) * self.scale

    def ipx(self, value: float, minimum: int = 1) -> int:
        return max(minimum, int(round(float(value) * self.scale)))

    def rng(self, salt: int | str = 0) -> np.random.Generator:
        h = hashlib.blake2b(f"{self.seed}:{self.step_id}:{salt}".encode(), digest_size=8).digest()
        return np.random.default_rng(int.from_bytes(h, "little"))

    def check(self) -> None:
        if self.cancel is not None:
            self.cancel.check()

    def resolve_file(self, path: str, kind: str = "file") -> Path:
        if not path:
            raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"A {kind} file path is required")
        return self.resolver(path, kind)

    def derive(self, **changes: Any) -> "RenderContext":
        d = dict(self.__dict__)
        d.update(changes)
        return RenderContext(**d)
