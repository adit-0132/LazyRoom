"""Filesystem access policy for agent-facing interfaces.

* Reads are allowed only under ``read_roots`` (default: home, temp, cwd).
* Writes are allowed only under ``write_roots`` and never into sensitive
  locations (SSH keys, credentials, system directories, app binaries).
* Originals are protected: an open project's source file can never be an
  output target, and existing files are only replaced with ``overwrite=true``.
* Paths are resolved (symlinks followed) before checking, so ``..`` or links
  cannot escape the roots.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from ..core.errors import ErrorCode, LazyRoomError
from ..core.paths import artifacts_dir, cache_dir, data_dir

SENSITIVE_PARTS = {".ssh", ".gnupg", ".aws", ".kube", ".docker", ".password-store", "keyrings", ".netrc"}
SYSTEM_PREFIXES_POSIX = ("/etc", "/bin", "/sbin", "/usr", "/lib", "/lib64", "/boot", "/sys", "/proc", "/dev",
                         "/var/lib")


def _resolve(p: str | os.PathLike) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(p))))
    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class PathPolicy:
    def __init__(self, read_roots: Iterable[str] = (), write_roots: Iterable[str] = ()) -> None:
        defaults = [Path.home(), Path(tempfile.gettempdir()), Path.cwd()]
        self.read_roots = [_resolve(r) for r in read_roots] or [_resolve(d) for d in defaults]
        self.write_roots = [_resolve(r) for r in write_roots] or [_resolve(d) for d in defaults]
        # the application's own data is always accessible
        for d in (data_dir(), cache_dir(), artifacts_dir()):
            rd = _resolve(d)
            self.read_roots.append(rd)
            self.write_roots.append(rd)
        self.read_roots = list(dict.fromkeys(self.read_roots))
        self.write_roots = list(dict.fromkeys(self.write_roots))
        self.protected: set[Path] = set()

    def protect(self, path: str | os.PathLike) -> None:
        self.protected.add(_resolve(path))

    def check_read(self, path: str | os.PathLike) -> Path:
        p = _resolve(path)
        if not any(_is_under(p, r) for r in self.read_roots):
            raise LazyRoomError(ErrorCode.PATH_NOT_ALLOWED, f"Reading outside the allowed folders is not permitted: {p}",
                                  details={"read_roots": [str(r) for r in self.read_roots]},
                                  hint="Ask the user to add the folder to LAZYROOM_READ_ROOTS or the config read_roots.")
        if self._sensitive(p):
            raise LazyRoomError(ErrorCode.PATH_NOT_ALLOWED, f"Access to sensitive location refused: {p}")
        return p

    def check_write(self, path: str | os.PathLike, *, overwrite: bool = False) -> Path:
        p = _resolve(path)
        if not any(_is_under(p, r) for r in self.write_roots):
            raise LazyRoomError(ErrorCode.PATH_NOT_ALLOWED, f"Writing outside the allowed folders is not permitted: {p}",
                                  details={"write_roots": [str(r) for r in self.write_roots]},
                                  hint="Choose an output inside an allowed folder (see capabilities.security).")
        if self._sensitive(p) or self._system(p):
            raise LazyRoomError(ErrorCode.PATH_NOT_ALLOWED, f"Refusing to write to a protected location: {p}")
        if p in self.protected:
            raise LazyRoomError(ErrorCode.PATH_NOT_ALLOWED, f"Refusing to overwrite an original source image: {p}",
                                  hint="Export to a new file; originals are never modified.")
        if p.exists() and not overwrite:
            raise LazyRoomError(ErrorCode.OUTPUT_EXISTS, f"Output file already exists: {p}",
                                  hint="Choose a different name or pass overwrite=true.")
        if p.exists() and p.is_dir():
            raise LazyRoomError(ErrorCode.PATH_NOT_ALLOWED, f"Output path is a directory: {p}")
        return p

    def check_write_dir(self, path: str | os.PathLike) -> Path:
        p = _resolve(path)
        if not any(_is_under(p, r) for r in self.write_roots) or self._sensitive(p) or self._system(p):
            raise LazyRoomError(ErrorCode.PATH_NOT_ALLOWED, f"Writing into this folder is not permitted: {p}")
        return p

    @staticmethod
    def _sensitive(p: Path) -> bool:
        parts = set(p.parts)
        return bool(parts & SENSITIVE_PARTS)

    @staticmethod
    def _system(p: Path) -> bool:
        if sys.platform == "win32":
            s = str(p).lower()
            windir = os.environ.get("WINDIR", "C:\\Windows").lower()
            progs = [os.environ.get(k, "").lower() for k in ("PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA")]
            return s.startswith(windir) or any(x and s.startswith(x) for x in progs)
        s = str(p)
        return any(s == pre or s.startswith(pre + "/") for pre in SYSTEM_PREFIXES_POSIX)

    def describe(self) -> dict:
        return {"read_roots": [str(r) for r in self.read_roots], "write_roots": [str(r) for r in self.write_roots],
                "originals_protected": True, "overwrite_requires_flag": True}
