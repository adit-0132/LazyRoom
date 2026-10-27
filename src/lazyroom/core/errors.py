"""Structured, machine-readable errors shared by every interface (SDK, CLI, MCP, GUI).

Every failure surfaced to an agent carries a stable ``code`` (see ``ErrorCode``),
a human-readable ``message``, optional structured ``details`` and a ``hint``
describing how to recover.  CLI exit codes are derived from the code.
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    INVALID_PARAMS = "INVALID_PARAMS"
    UNKNOWN_OPERATION = "UNKNOWN_OPERATION"
    UNKNOWN_COMMAND = "UNKNOWN_COMMAND"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    DECODE_FAILED = "DECODE_FAILED"
    ENCODE_FAILED = "ENCODE_FAILED"
    PATH_NOT_ALLOWED = "PATH_NOT_ALLOWED"
    OUTPUT_EXISTS = "OUTPUT_EXISTS"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    PROJECT_NOT_FOUND = "PROJECT_NOT_FOUND"
    STEP_NOT_FOUND = "STEP_NOT_FOUND"
    PRESET_NOT_FOUND = "PRESET_NOT_FOUND"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    NOTHING_TO_UNDO = "NOTHING_TO_UNDO"
    NOTHING_TO_REDO = "NOTHING_TO_REDO"
    SCHEMA_VERSION = "SCHEMA_VERSION"
    OPERATION_FAILED = "OPERATION_FAILED"
    INTERNAL = "INTERNAL"


# Process exit codes used by the CLI.  0 = success, 1 = partial failure (batch).
EXIT_CODES: dict[ErrorCode, int] = {
    ErrorCode.INVALID_PARAMS: 2,
    ErrorCode.UNKNOWN_OPERATION: 2,
    ErrorCode.UNKNOWN_COMMAND: 2,
    ErrorCode.UNKNOWN_TOOL: 2,
    ErrorCode.SCHEMA_VERSION: 2,
    ErrorCode.FILE_NOT_FOUND: 3,
    ErrorCode.PROJECT_NOT_FOUND: 3,
    ErrorCode.STEP_NOT_FOUND: 3,
    ErrorCode.PRESET_NOT_FOUND: 3,
    ErrorCode.JOB_NOT_FOUND: 3,
    ErrorCode.UNSUPPORTED_FORMAT: 4,
    ErrorCode.DECODE_FAILED: 4,
    ErrorCode.ENCODE_FAILED: 4,
    ErrorCode.PATH_NOT_ALLOWED: 5,
    ErrorCode.OUTPUT_EXISTS: 5,
    ErrorCode.APPROVAL_REQUIRED: 5,
    ErrorCode.BACKEND_UNAVAILABLE: 6,
    ErrorCode.RESOURCE_LIMIT: 7,
    ErrorCode.TIMEOUT: 8,
    ErrorCode.CANCELLED: 9,
    ErrorCode.NOTHING_TO_UNDO: 10,
    ErrorCode.NOTHING_TO_REDO: 10,
    ErrorCode.OPERATION_FAILED: 11,
    ErrorCode.INTERNAL: 70,
}


class LazyRoomError(Exception):
    """Base error carrying a stable machine-readable code."""

    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = ErrorCode(code)
        self.message = message
        self.details = details or {}
        self.hint = hint

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, 1)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code.value, "message": self.message}
        if self.details:
            out["details"] = self.details
        if self.hint:
            out["hint"] = self.hint
        return out

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code.value}] {self.message}"


def invalid(message: str, **details: Any) -> LazyRoomError:
    hint = details.pop("hint", None)
    return LazyRoomError(ErrorCode.INVALID_PARAMS, message, details=details, hint=hint)


def wrap_exception(exc: BaseException) -> LazyRoomError:
    """Convert any exception into a LazyRoomError without hiding the cause."""
    if isinstance(exc, LazyRoomError):
        return exc
    if isinstance(exc, FileNotFoundError):
        return LazyRoomError(ErrorCode.FILE_NOT_FOUND, str(exc))
    if isinstance(exc, MemoryError):
        return LazyRoomError(
            ErrorCode.RESOURCE_LIMIT,
            "Out of memory while processing the image",
            hint="Reduce image size, render a smaller preview, or lower the batch concurrency.",
        )
    return LazyRoomError(
        ErrorCode.INTERNAL,
        f"{type(exc).__name__}: {exc}",
        details={"exception_type": type(exc).__name__},
        hint="This is likely a bug; the full traceback is in the operation log.",
    )
