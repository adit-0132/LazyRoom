"""Shared infrastructure for the ImageToolbox filter catalogue port.

ImageToolbox (https://github.com/T8RIN/ImageToolbox) by T8RIN is licensed under
Apache-2.0; its filter implementations build on ImageToolboxLibs (Apache-2.0,
incl. a jhlabs filters port, Apache-2.0), Trickle (Apache-2.0), aire (MIT,
awxkee) and GPUImage (BSD-3).  The G'MIC-based filters call the external
G'MIC CLI (CeCILL) as a separate process.  See NOTICE / docs/THIRD_PARTY.md.

Every filter is registered as ``filter.<snake_case_name>`` with ``group`` set
to the ImageToolbox UI group and ``source`` attribution metadata.
"""
from __future__ import annotations

import contextlib
import functools
import os
import threading
import re
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from ...core import backends
from ...core.errors import ErrorCode, LazyRoomError
from ...core.params import Param
from ...core.paths import package_data_dir
from ...core.registry import FILTER_GROUPS, op
from .._util import F32, apply_lut3d, lut_from_hald_or_square

ITB_REPO = "https://github.com/T8RIN/ImageToolbox"


_SNAKE_SPECIAL = {"DoG": "dog"}


def snake(name: str) -> str:
    """ImageToolbox class name -> operation id part, e.g. FastGaussianBlur2D -> fast_gaussian_blur_2d,
    LUT512x512 -> lut_512x512, VHSNtsc -> vhs_ntsc, OldSchool8Bit -> old_school_8_bit."""
    if name in _SNAKE_SPECIAL:
        return _SNAKE_SPECIAL[name]
    s = re.sub(r"(\d+)x(\d+)", r"_\1x\2_", name)
    s = re.sub(r"(\d+)D(?=[A-Z]|$)", r"_\1d_", s)
    s = re.sub(r"(\d+)(?=[A-Z])", r"_\1_", s)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", s)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_").lower()


def _legacy_snake(name: str) -> str:
    """Id scheme used before v0.1 release candidates; kept as aliases so old projects still open."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    return s.strip("_").lower()


def itb_op(name: str, title: str, group: str, params: Sequence[Param] = (), *,
           upstream: str = "ImageToolbox", description: str | None = None, tags: Sequence[str] = (),
           requires: Sequence[str] = (), alpha: str = "preserve", geometry: bool = False,
           deterministic: bool = True, op_id: str | None = None, clamp: bool = True, **kw: Any) -> Callable:
    """Register an ImageToolbox filter port.

    ``name`` is the ImageToolbox filter class name without the ``Filter``
    suffix (e.g. ``"BlackAndWhiteFilm"``); ``upstream`` names the library that
    implements the algorithm in ImageToolbox (GPUImage, aire, Trickle, jhlabs,
    G'MIC, OpenCV, ImageToolbox).  ``clamp`` (default True) clips the output
    to 0..1 like ImageToolbox's 8-bit pipeline does after every filter.
    """
    if group not in FILTER_GROUPS:
        raise ValueError(f"Unknown ImageToolbox group {group!r}")
    oid = op_id or f"filter.{snake(name)}"
    legacy = f"filter.{_legacy_snake(name)}"
    if legacy != oid:
        kw["aliases"] = list(kw.get("aliases", [])) + [legacy]
    source = {"project": "ImageToolbox", "filter": name, "upstream": upstream, "license": "Apache-2.0",
              "url": ITB_REPO}
    register = op(oid, title, params=params, category="filter", group=group, description=description,
                  tags=list(tags), requires=list(requires), alpha=alpha, geometry=geometry,
                  deterministic=deterministic, source=source, **kw)

    def deco(fn: Callable) -> Callable:
        if not clamp:
            register(fn)
            return fn

        @functools.wraps(fn)
        def clamped(img, ctx, **values):
            return np.clip(fn(img, ctx, **values), 0.0, 1.0)

        register(clamped)
        return fn

    return deco


# ---------------------------------------------------------------------------
# Colour matrices (GPUImage / Android ColorMatrix semantics)
# ---------------------------------------------------------------------------

def color_matrix_3x3(img: np.ndarray, m: Sequence[float]) -> np.ndarray:
    """out = M @ rgb with M given row-major (9 values)."""
    M = np.asarray(m, F32).reshape(3, 3)
    return (img[..., :3] @ M.T).astype(F32)


def color_matrix_4x4(img: np.ndarray, m: Sequence[float], intensity: float = 1.0) -> np.ndarray:
    """GPUImageColorMatrixFilter: GLSL ``rgba * colorMatrix`` with the float[16] uploaded column-major,
    which equals ``A @ rgba`` for A = m reshaped row-major; mixed with the input by ``intensity``."""
    M = np.asarray(m, F32).reshape(4, 4)
    rgba = np.dstack([img[..., :3], np.ones(img.shape[:2], F32)])
    out = rgba @ M.T
    res = img[..., :3] + (out[..., :3] - img[..., :3]) * intensity
    return res.astype(F32)


def android_color_matrix(img: np.ndarray, m: Sequence[float]) -> np.ndarray:
    """Android ``ColorMatrix`` (4x5, row-major, offsets in 0..255 units) applied to RGB."""
    M = np.asarray(m, F32).reshape(4, 5)
    rgba = np.dstack([img[..., :3], np.ones(img.shape[:2], F32)])
    out = rgba @ M[:, :4].T + M[:, 4] / 255.0
    return out[..., :3].astype(F32)


# ---------------------------------------------------------------------------
# Bundled 512x512 lookup LUTs
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=32)
def bundled_lut(name: str) -> np.ndarray:
    from ...core.io import load_image

    path = package_data_dir() / "luts" / f"lookup_{name}.webp"
    if not path.exists():
        raise LazyRoomError(ErrorCode.FILE_NOT_FOUND, f"Bundled LUT '{name}' missing at {path}")
    img, _ = load_image(path)
    return lut_from_hald_or_square(img)


def apply_lut512(img: np.ndarray, name_or_lut: str | np.ndarray, intensity: float = 1.0) -> np.ndarray:
    lut = bundled_lut(name_or_lut) if isinstance(name_or_lut, str) else name_or_lut
    out = apply_lut3d(np.clip(img, 0, 1), lut)
    return (img + (out - img) * intensity).astype(F32)


# ---------------------------------------------------------------------------
# G'MIC bridge (external CLI, lossless float exchange via the .cimg format)
# ---------------------------------------------------------------------------

def _write_cimg(path: str, imgs: Sequence[np.ndarray]) -> None:
    with open(path, "wb") as f:
        f.write(f"{len(imgs)} float32 little_endian\n".encode())
        for a in imgs:
            h, w, c = a.shape
            f.write(f"{w} {h} 1 {c}\n".encode())
            f.write(np.ascontiguousarray(a.transpose(2, 0, 1), dtype="<f4").tobytes())


def _read_cimg(path: str) -> list[np.ndarray]:
    data = Path(path).read_bytes()
    nl = data.index(b"\n")
    header = data[:nl].decode().split()
    n, dtype = int(header[0]), header[1]
    endian = "<" if (len(header) < 3 or header[2] == "little_endian") else ">"
    np_t = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1", "uint8": "u1",
            "unsigned_char": "u1", "int": "i4", "int32": "i4", "unsigned_short": "u2", "uint16": "u2",
            "short": "i2", "int16": "i2", "unsigned_int": "u4", "uint32": "u4", "char": "i1", "int8": "i1"}.get(dtype)
    if np_t is None:
        raise LazyRoomError(ErrorCode.DECODE_FAILED, f"Unsupported .cimg pixel type {dtype}")
    pos = nl + 1
    out = []
    for _ in range(n):
        nl = data.index(b"\n", pos)
        dims = data[pos:nl].decode().split()
        w, h, d, c = (int(v) for v in dims[:4])
        pos = nl + 1
        if len(dims) > 4 and dims[4].startswith("#"):  # pragma: no cover - zlib variant never requested
            raise LazyRoomError(ErrorCode.DECODE_FAILED, "Compressed .cimg not supported")
        count = w * h * d * c
        if count == 0:
            continue
        nbytes = count * np.dtype(np_t).itemsize
        arr = np.frombuffer(data, dtype=endian + np_t, count=count, offset=pos).astype(F32)
        pos += nbytes
        arr = arr.reshape(c, d, h, w)[:, 0].transpose(1, 2, 0)
        out.append(arr)
    return out


_TLS = threading.local()


@contextlib.contextmanager
def gmic_single_thread():
    """Run G'MIC calls in this thread single-threaded (OMP_NUM_THREADS=1 in the child only).

    Some G'MIC filters draw random numbers from several threads, so they are only
    reproducible (after ``srand``) when evaluated on one thread.
    """
    prev = getattr(_TLS, "single", False)
    _TLS.single = True
    try:
        yield
    finally:
        _TLS.single = prev


def gmic_env(single_thread: bool = False) -> dict[str, str]:
    """Environment for the G'MIC child: an isolated config home (so user/updated command files are
    never loaded and nothing is written to the user's home) and optional single-threading."""
    from ...core.paths import cache_dir, ensure

    env = dict(os.environ)
    home = ensure(cache_dir() / "gmic-home")
    env.update({"HOME": str(home), "XDG_CONFIG_HOME": str(home / ".config"), "APPDATA": str(home),
                "USERPROFILE": str(home)})
    if single_thread or getattr(_TLS, "single", False):
        env["OMP_NUM_THREADS"] = "1"
    return env


def run_gmic(img: np.ndarray, command: str, *, preserve_alpha: bool = True, aux: Sequence[np.ndarray] = (),
             output_index: int = 0, timeout: float = 600.0, ctx: Any = None,
             single_thread: bool = False) -> np.ndarray:
    """Run a G'MIC pipeline on ``img`` (0..1 float) and return the result (0..1 float).

    Mirrors ImageToolboxLibs' native bridge: pixels are passed as 0..255
    floats; with ``preserve_alpha`` only RGB is processed and the original
    alpha is re-attached (resized if the geometry changed).
    """
    exe = backends.gmic_path()
    if not exe:
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, "The G'MIC CLI is not installed",
                              details={"backend": "gmic"}, hint=backends.hint_for("gmic"))
    alpha = img[..., 3] if img.shape[2] == 4 else None
    main = img[..., :3] if (preserve_alpha or alpha is None) else img
    imgs = [np.clip(main, 0, 1) * 255.0] + [np.clip(a, 0, 1) * 255.0 for a in aux]
    with tempfile.TemporaryDirectory(prefix="pm-gmic-") as td:
        src = os.path.join(td, "in.cimg")
        dst = os.path.join(td, "out.cimg")
        _write_cimg(src, imgs)
        args = [exe, "-v", "-1", "-network", "-1", "-input", src, *_split_cmd(command),
                "-keep", str(output_index), "-output", f"{dst},float32"]
        try:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=td,
                                    env=gmic_env(single_thread))
            while True:
                try:
                    _, err = proc.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    timeout -= 0.25
                    if (ctx is not None and ctx.cancel is not None and ctx.cancel.cancelled) or timeout <= 0:
                        proc.kill()
                        proc.communicate()
                        if timeout <= 0:
                            raise LazyRoomError(ErrorCode.TIMEOUT, "G'MIC command timed out")
                        raise LazyRoomError(ErrorCode.CANCELLED, "G'MIC command cancelled")
        except FileNotFoundError as exc:  # pragma: no cover
            raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, f"Cannot start G'MIC: {exc}") from exc
        if proc.returncode != 0 or not os.path.exists(dst):
            msg = err.decode("utf-8", "replace").strip().splitlines()
            raise LazyRoomError(ErrorCode.OPERATION_FAILED, "G'MIC command failed: " + (msg[-1] if msg else "?"),
                                  details={"command": command, "stderr": "\n".join(msg[-5:])})
        outs = _read_cimg(dst)
        if not outs:
            raise LazyRoomError(ErrorCode.OPERATION_FAILED, "G'MIC returned no image", details={"command": command})
        res = outs[0]
    res = np.clip(res / 255.0, 0, 1).astype(F32)
    c = res.shape[2]
    if c == 1:
        res = np.repeat(res, 3, axis=2)
    elif c == 2:
        res = np.dstack([np.repeat(res[..., :1], 3, axis=2), res[..., 1:2]])
    elif c > 4:
        res = res[..., :4]
    if preserve_alpha and alpha is not None:
        rgb = res[..., :3]
        a = alpha
        if a.shape != rgb.shape[:2]:
            a = cv2.resize(a, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        # like ImageToolbox's bridge (GmicAlphaMode.Preserve): any alpha G'MIC produced is discarded
        return np.dstack([rgb, a]).astype(F32)
    return res


def _split_cmd(command: str) -> list[str]:
    """Split a G'MIC pipeline into argv items, respecting double quotes."""
    out, cur, q = [], [], False
    for ch in command.strip():
        if ch == '"':
            q = not q
            cur.append(ch)
        elif ch.isspace() and not q:
            if cur:
                out.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def gmic_cmd(name: str, *args: Any) -> str:
    """Format a G'MIC command like ImageToolboxLibs' ``gmicCommand`` (comma-joined args)."""
    def fmt(a: Any) -> str:
        if isinstance(a, bool):
            return "1" if a else "0"
        if isinstance(a, float):
            s = f"{a:.6f}".rstrip("0").rstrip(".")
            return s if s not in ("", "-0") else "0"
        if hasattr(a, "value"):
            return str(a.value)
        return str(a)
    return name if not args else f"{name} " + ",".join(fmt(a) for a in args)


def gmic_pipeline(*cmds: str) -> str:
    return " ".join(cmds)
