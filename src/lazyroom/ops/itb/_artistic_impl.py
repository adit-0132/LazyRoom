"""Private helpers for the ImageToolbox "Artistic" group (see artistic.py).

Ports of the native implementations ImageToolbox uses:

* G'MIC bridge emulating ImageToolboxLibs' embedded interpreter (alpha modes,
  and its ``disableSmooth = true`` default, see :func:`gmic_itb`);
* Trickle (Apache-2.0): ``OilFilter.cpp``, ``lowpoly/*.c`` + ``LowPolyPipelineImpl``,
  ``popArtImpl`` / ``drawColorBehindImpl`` (cairo compositing);
* aire (MIT, awxkee): ``OilEffect.cpp``, ``RemoveShadows.cpp`` (+ ``MedianBlur.cpp``,
  ``Arithmetics.cpp``, ``Dilation.cpp``);
* ImageToolboxLibs jhlabs Rust port (Apache-2.0): ``CellularFilter``/``PointillizeFilter``
  and ``StampFilter``;
* GPUImage (BSD-3): ``GPUImageToonFilter``, ``GPUImageSmoothToonFilter``.

Sequential / per-pixel-branching loops are ``numba.njit(cache=True)``.
"""
from __future__ import annotations

import functools
import hashlib
import math
import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from numba import njit, prange

from ...core import backends
from ._common import gmic_pipeline, run_gmic

F32 = np.float32

# ---------------------------------------------------------------------------
# G'MIC bridge (ImageToolboxLibs semantics)
# ---------------------------------------------------------------------------

# ImageToolboxLibs runs G'MIC with ``GmicExecutionOptions.disableSmooth = true``:
# its patched interpreter skips every call of the built-in ``smooth`` command
# (and its argument).  These stdlib commands (used by the artistic filters)
# call ``smooth``; for ITB parity we load copies where ``smooth`` -> ``skip``.
# Not included: ``fx_highlight_bloom`` uses ``+smooth`` (skipping it makes the
# command fail in ImageToolbox, "Invalid selection [-2]"), so it keeps stock
# ``smooth`` to stay usable.
ITB_DISABLE_SMOOTH = True  # set False to run these filters with stock G'MIC smoothing
NOSMOOTH_COMMANDS = ("fx_engrave", "fx_lylejk_painting", "fx_painting", "fx_pen_drawing", "fx_rodilius",
                     "fx_smooth_anisotropic", "fx_stringify", "pencilbw", "poster_edges", "poster_hope",
                     "inpaint_pde")
_NOSMOOTH_ENV = "LAZYROOM_ITB_GMIC_NOSMOOTH"
_SMOOTH_TOKEN = re.compile(r'(?:^|(?<=[\s"{]))[-+]?smooth(?:\[[^\]]*\]|\.+)?(?=\s|$)')


def _reescape_spaces(code: str) -> str:
    """``$$cmd`` dumps print escaped spaces (``${cmd\\ arg}``) as plain spaces; restore them."""
    out, depth, quoted, i = [], 0, False, 0
    while i < len(code):
        ch = code[i]
        if ch == '"':
            quoted = not quoted
        elif not quoted:
            if ch == "$" and code[i + 1:i + 2] == "{":
                depth += 1
                out.append("${")
                i += 2
                continue
            if ch == "{" and depth:
                depth += 1
            elif ch == "}" and depth:
                depth -= 1
            elif ch == " " and depth:
                out.append("\\ ")
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


@functools.lru_cache(maxsize=4)
def _nosmooth_file(exe: str) -> str | None:
    """Write (once) a command file overriding NOSMOOTH_COMMANDS with ``smooth`` disabled.

    The definitions are dumped from the installed G'MIC itself (``$$name``) so
    they always match its stdlib version.  Returns None if that fails (the
    filters then run with stock ``smooth``).
    """
    args = [exe, "-v", "0"]
    for name in NOSMOOTH_COMMANDS:
        args += ["echo", f"@@{name}@@", "echo", f"$${name}"]
    args.append("-quit")
    try:
        from ._common import gmic_env

        proc = subprocess.run(args, capture_output=True, timeout=60, env=gmic_env())
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (proc.stderr + proc.stdout).decode("utf-8", "replace").splitlines()
    defs = []
    for k, line in enumerate(lines[:-1]):
        m = re.fullmatch(r"@@(\w+)@@", line.strip())
        code = lines[k + 1].strip()
        if m and code and not code.startswith("@@"):
            defs.append(f"{m.group(1)} : {_reescape_spaces(_SMOOTH_TOKEN.sub('skip', code))}")
    if not defs:
        return None
    text = "#@gmic\n" + "\n".join(defs) + "\n"
    # per-user cache (not a shared temp dir: the file is executed by G'MIC, so it must not be pre-creatable)
    from ...core.paths import cache_dir

    folder = cache_dir() / "gmic-itb"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"nosmooth-{hashlib.sha1(text.encode()).hexdigest()[:16]}.gmic"
    if not path.exists():
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    # Passed to G'MIC as `-command $LAZYROOM_ITB_GMIC_NOSMOOTH` (G'MIC substitutes
    # OS environment variables), which keeps paths with spaces intact.
    os.environ[_NOSMOOTH_ENV] = str(path)
    return str(path)


def gmic_itb(img: np.ndarray, ctx, command: str, *, mode: str = "preserve", no_smooth: bool = False) -> np.ndarray:
    """Run a G'MIC filter like ImageToolboxLibs' ``Gmic.runCancellable``.

    ``mode``: ``"preserve"`` (GmicAlphaMode.Preserve: only RGB goes through
    G'MIC, any alpha G'MIC produces is ignored; source alpha is re-attached by
    the registry), ``"process"`` (GmicAlphaMode.Process: RGBA goes through
    G'MIC and its alpha is the result alpha), ``"geometry"`` (Preserve mode for
    commands that change the image size: source alpha is resized).
    ``no_smooth`` emulates ITB's disabled ``smooth`` built-in.  The G'MIC
    random generator is seeded from ``ctx`` (ITB leaves it time-seeded).
    """
    pre = []
    if no_smooth and ITB_DISABLE_SMOOTH:
        exe = backends.gmic_path()
        patch = _nosmooth_file(exe) if exe else None
        if patch and not os.path.exists(patch):  # temp dir cleaned since: rebuild
            _nosmooth_file.cache_clear()
            patch = _nosmooth_file(exe)
        if patch:
            os.environ[_NOSMOOTH_ENV] = patch
            pre.append(f"-command ${_NOSMOOTH_ENV}")
    pre.append(f"srand {int(ctx.rng('gmic').integers(1, 2 ** 31 - 1))}")
    cmd = gmic_pipeline(*pre, command)
    h, w = img.shape[:2]
    if mode == "process":
        had_alpha = img.shape[2] == 4
        rgba = img if had_alpha else np.dstack([img, np.ones((h, w), F32)])
        res = run_gmic(np.ascontiguousarray(rgba), cmd, preserve_alpha=False, ctx=ctx)
        if res.shape[2] == 3:
            res = np.dstack([res, np.ones(res.shape[:2], F32)])
        if not had_alpha and float(res[..., 3].min()) >= 1.0 - 1e-6:
            res = res[..., :3]
        return res
    res = run_gmic(np.ascontiguousarray(img[..., :3]), cmd, ctx=ctx)[..., :3]
    if mode == "geometry" and img.shape[2] == 4:
        a = img[..., 3]
        if res.shape[:2] != (h, w):
            a = cv2.resize(a, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_LINEAR)
        res = np.dstack([res, a])
    return np.ascontiguousarray(res, dtype=F32)


def color255(c) -> tuple[int, int, int, int]:
    """(r, g, b, a) 0..1 floats -> 0..255 ints (Android ``Color.red()`` etc.)."""
    return tuple(int(round(float(v) * 255)) for v in c)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 8-bit helpers (ImageToolbox works on ARGB_8888 bitmaps)
# ---------------------------------------------------------------------------

def to8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def premultiplied8(img: np.ndarray) -> np.ndarray:
    """RGBA uint8 as stored in an Android ARGB_8888 bitmap (premultiplied)."""
    h, w = img.shape[:2]
    a = img[..., 3] if img.shape[2] == 4 else np.ones((h, w), F32)
    rgb = img[..., :3] * a[..., None]
    return to8(np.dstack([rgb, a]))


def unpremultiply(rgba8: np.ndarray) -> np.ndarray:
    """Premultiplied RGBA uint8 -> straight RGBA float32 0..1."""
    out = rgba8.astype(F32) / 255.0
    a = out[..., 3:4]
    out[..., :3] = np.where(a > 0, out[..., :3] / np.maximum(a, 1e-8), 0.0)
    return out


def finish_rgba(out: np.ndarray, had_alpha: bool) -> np.ndarray:
    """Drop a fully opaque alpha channel when the input had none."""
    if out.shape[2] == 4 and not had_alpha and float(out[..., 3].min()) >= 1.0 - 1e-6:
        return np.ascontiguousarray(out[..., :3])
    return np.ascontiguousarray(out, dtype=F32)


# ---------------------------------------------------------------------------
# Trickle OilFilter (per-channel mode of a 2r x 2r window)
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _trickle_oil(px: np.ndarray, r: int) -> np.ndarray:
    h, w, _ = px.shape
    out = np.empty((h, w, 3), np.uint8)
    for y in prange(h):
        hist = np.zeros((3, 256), np.int32)
        y0 = max(0, y - r)
        y1 = min(h, y + r)
        # window columns [x - r, x + r) (OilFilter.cpp: col in [-range, range))
        for col in range(0, min(w, r)):
            for yy in range(y0, y1):
                for c in range(3):
                    hist[c, px[yy, col, c]] += 1
        for x in range(w):
            if x > 0:
                cr = x - r - 1
                if cr >= 0:
                    for yy in range(y0, y1):
                        for c in range(3):
                            hist[c, px[yy, cr, c]] -= 1
                ca = x + r - 1
                if ca < w:
                    for yy in range(y0, y1):
                        for c in range(3):
                            hist[c, px[yy, ca, c]] += 1
            for c in range(3):
                best = 0
                bc = hist[c, 0]
                for i in range(1, 256):
                    if hist[c, i] > bc:
                        bc = hist[c, i]
                        best = i
                out[y, x, c] = best
    return out


def trickle_oil(img: np.ndarray, oil_range: int) -> np.ndarray:
    """Trickle ``OilFilter``: per-channel histogram mode; result is opaque (``RGB2Color``)."""
    px = premultiplied8(img)[..., :3]
    return _trickle_oil(np.ascontiguousarray(px), int(oil_range)).astype(F32) / 255.0


# ---------------------------------------------------------------------------
# aire oilEffect (note the upstream window bug: posY uses `i`, so only the
# diagonal (x+i, y+i) is scanned)
# ---------------------------------------------------------------------------

def aire_oil(img: np.ndarray, radius: int, levels: float) -> np.ndarray:
    h, w = img.shape[:2]
    p8 = to8(img[..., :3]).astype(np.int32)
    gray = (p8[..., 0] + p8[..., 1] + p8[..., 2]) // 3
    inten = np.clip(gray.astype(F32) * F32(levels), 0, 255).astype(np.uint8)
    r = int(radius)
    ipad = np.pad(inten, r, mode="edge")
    cpad = np.pad(img, ((r, r), (r, r), (0, 0)), mode="edge")
    best = ipad[0:h, 0:w].copy()
    out = cpad[0:h, 0:w].copy()
    for i in range(1, 2 * r + 1):  # offsets -r+1 .. r; first maximum wins
        cand = ipad[i:i + h, i:i + w]
        upd = cand > best
        best = np.where(upd, cand, best)
        out = np.where(upd[..., None], cpad[i:i + h, i:i + w], out)
    return out.astype(F32)


# ---------------------------------------------------------------------------
# aire removeShadows ("Sketch")
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _aire_median_channel(data: np.ndarray, size: int) -> np.ndarray:
    """aire ``medianBlurChannel`` incl. its sliding-histogram quirks (column
    ``size`` is never added, later removed) — reproduced exactly."""
    h, w = data.shape
    out = np.empty((h, w), np.uint8)
    for y in prange(h):
        hist = np.zeros(256, np.int64)
        n = 0
        i0 = max(0, y - size)
        i1 = min(h - 1, y + size)
        for j in range(min(size, w)):
            for i in range(i0, i1 + 1):
                hist[data[i, j]] += 1
                n += 1
        for x in range(w):
            if x > 0:
                xd = x - size
                if 0 <= xd < w:
                    for i in range(i0, i1 + 1):
                        hist[data[i, xd]] -= 1
                        n -= 1
                xa = x + size
                if 0 <= xa < w:
                    for i in range(i0, i1 + 1):
                        hist[data[i, xa]] += 1
                        n += 1
            if n > 0:
                t = n // 2
                k = 0
                while k < 256:
                    t -= hist[k]
                    if t <= 0:
                        break
                    k += 1
                out[y, x] = k & 0xFF
            else:
                out[y, x] = data[y, x]
    return out


def aire_remove_shadows(img: np.ndarray, kernel_size: int, median_size: int = 21, lanes: int = 16) -> np.ndarray:
    """aire ``removeShadows`` per channel: dilate -> median(21) -> 255-|median-src| -> stretch.

    ``absDiff`` runs in signed 8-bit SIMD lanes upstream (wrapping), except for
    the scalar tail of each row; reproduced for arm64/NEON (16 lanes).
    ``normalize`` uses min = 0 (``pickMaxMin`` starts from 0).
    """
    h, w = img.shape[:2]
    k = max(1, int(kernel_size))
    kern = np.ones((k, k), np.uint8)
    anchor = ((k - 1) // 2, (k - 1) // 2)  # aire dilation window [x-(k-1)//2, x+k//2]
    x_end = lanes * ((w - 1) // lanes)
    out = np.empty((h, w, 3), F32)
    for c in range(3):
        src = to8(img[..., c])
        dil = cv2.dilate(src, kern, anchor=anchor, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        med = _aire_median_channel(np.ascontiguousarray(dil), int(median_size)).astype(np.int16)
        d = med - src.astype(np.int16)
        a = np.abs(d)
        wrapped = np.abs(((d + 128) & 0xFF) - 128)  # int8 Sub + Abs
        a[:, :x_end] = wrapped[:, :x_end]
        chan = (255 - a).astype(F32)
        mx = float(chan.max())
        out[..., c] = np.ceil(chan / mx * 255.0) / 255.0 if mx > 0 else 0.0
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# Trickle lowpoly (xyzxqs lowpoly: sobel -> random dilution -> dedup -> Delaunay)
# ---------------------------------------------------------------------------

def _lowpoly_points(rgba8: np.ndarray, threshold: int, count: float, rng: np.random.Generator) -> np.ndarray:
    """``get_triangles`` vertex selection; returns (N, 2) int32 (x, y) sorted by x then y."""
    h, w = rgba8.shape[:2]
    a = rgba8[..., 3].astype(np.float32) / 255.0
    s = rgba8[..., 0].astype(np.int32) + rgba8[..., 1] + rgba8[..., 2]
    avg = np.zeros((h + 1, w + 1), np.int32)  # color_average(): 0 outside (x>=w or y>=h)
    avg[:h, :w] = ((s * a) / 3.0).astype(np.int32)
    g = lambda dx, dy: avg[1 + dy:h + dy, 1 + dx:w + dx]  # noqa: E731  (pixels x, y >= 1)
    gx = -g(-1, -1) + g(1, -1) - 2 * g(-1, 0) + 2 * g(1, 0) - g(-1, 1) + g(1, 1)
    gy = -g(-1, -1) - 2 * g(0, -1) - g(1, -1) + g(-1, 1) + 2 * g(0, 1) + g(1, 1)
    mag = np.sqrt((gx * gx + gy * gy).astype(np.float64)).astype(np.int32)
    ys, xs = np.nonzero(mag > threshold)  # row-major scan order, like sobel()
    pts = np.stack([xs + 1, ys + 1], axis=1).astype(np.int32)[: w * h // 2]
    n = len(pts)
    if n and int(count) <= n:  # dilution(): point_filter1 (random picks with replacement)
        idx = (rng.random(int(count)) * (n - 1)).astype(np.int64)
        pts = pts[idx]
    if len(pts):  # dedup(): qsort by (x, y) + remove duplicates
        pts = np.unique(pts, axis=0)
    return pts


def lowpoly(img: np.ndarray, count: float, fill: bool, rng: np.random.Generator) -> np.ndarray:
    """Trickle ``lowPoly`` (native ``generate``, threshold 50): Delaunay triangles
    filled (anti-aliased, over the source) with the colour at their centroid, or
    stroked outlines (width 0.0005 of the image, drawn twice) on transparency."""
    from scipy.spatial import Delaunay

    h, w = img.shape[:2]
    src8 = premultiplied8(img)
    pts = _lowpoly_points(src8, 50, count, rng)
    corners = np.array([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]], np.int32)
    verts = np.concatenate([pts, corners]) if len(pts) else corners
    verts = np.unique(verts, axis=0)
    canvas = src8.copy() if fill else np.zeros_like(src8)
    if len(verts) >= 3 and w > 1 and h > 1:
        try:
            tris = verts[Delaunay(verts.astype(np.float64)).simplices]
        except Exception:  # degenerate (collinear) point sets: nothing to draw
            tris = np.zeros((0, 3, 2), np.int32)
        cx = np.clip(tris[:, :, 0].sum(1) // 3, 0, w - 1)
        cy = np.clip(tris[:, :, 1].sum(1) // 3, 0, h - 1)
        cols = src8[cy, cx]
        # cairo puts integer vertex coordinates on pixel corners; OpenCV on centres.
        poly = ((tris.astype(np.float64) - 0.5) * 16).round().astype(np.int32)
        if fill:
            for p, col in zip(poly, cols):
                cv2.fillPoly(canvas, [p], tuple(int(v) for v in col), lineType=cv2.LINE_AA, shift=4)
        else:
            lw = min(1.0, 0.0005 * max(w, h))
            for p, col in zip(poly, cols):
                c = tuple(float(v) * lw for v in col)
                for _ in range(2):
                    cv2.polylines(canvas, [p], True, c, 1, lineType=cv2.LINE_AA, shift=4)
    return unpremultiply(canvas)


@njit(cache=True)
def _draw_sand(canvas: np.ndarray, pts: np.ndarray, cols: np.ndarray) -> None:
    h, w = canvas.shape[:2]
    for k in range(pts.shape[0]):
        x, y = pts[k, 0], pts[k, 1]
        a = cols[k, 3]
        for yy in range(y - 1, y + 1):
            for xx in range(x - 1, x + 1):
                if 0 <= xx < w and 0 <= yy < h:
                    for c in range(4):  # premultiplied SRC_OVER
                        canvas[yy, xx, c] = cols[k, c] + canvas[yy, xx, c] * (1.0 - a)


def sand_painting(img: np.ndarray, count: float, threshold: int, bg, rng: np.random.Generator) -> np.ndarray:
    """Trickle ``sandPainting`` + ``drawColorBehind``: radius-1 hairline circles
    (non-AA, i.e. the 2x2 pixel block around the point) in the source colour at
    the sampled edge points, composited over the background colour."""
    h, w = img.shape[:2]
    src8 = premultiplied8(img)
    pts = _lowpoly_points(src8, int(threshold), count, rng)
    straight = img if img.shape[2] == 4 else np.dstack([img, np.ones((h, w), F32)])
    canvas = np.zeros((h, w, 4), F32)
    if len(pts):
        c = straight[pts[:, 1], pts[:, 0]].astype(F32)
        cols = np.concatenate([c[:, :3] * c[:, 3:4], c[:, 3:4]], axis=1)
        _draw_sand(canvas, pts.astype(np.int64), np.ascontiguousarray(cols))
    bgp = np.array([bg[0] * bg[3], bg[1] * bg[3], bg[2] * bg[3], bg[3]], F32)
    out = canvas + bgp * (1.0 - canvas[..., 3:4])
    a = out[..., 3:4]
    rgb = np.where(a > 0, out[..., :3] / np.maximum(a, 1e-8), 0.0)
    return np.dstack([rgb, a]).astype(F32)


# ---------------------------------------------------------------------------
# Trickle popArt (cairo: half-size image in 4 quadrants, each blended with a
# square-harmony colour; the Android RGBA memory is read by cairo as BGRA)
# ---------------------------------------------------------------------------

def _sk_rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    v = mx / 255.0
    if d == 0:
        return 0.0, 0.0, v
    s = d / mx
    if r == mx:
        hh = (g - b) / d
    elif g == mx:
        hh = 2 + (b - r) / d
    else:
        hh = 4 + (r - g) / d
    hh *= 60
    if hh < 0:
        hh += 360
    return hh, s, v


def _sk_hsv_to_rgb(hh: float, s: float, v: float) -> tuple[int, int, int]:
    sb = max(0, min(255, int(math.floor(s * 255 + 0.5))))
    vb = max(0, min(255, int(math.floor(v * 255 + 0.5))))
    if sb == 0:
        return vb, vb, vb
    hx = 0.0 if (hh < 0 or hh >= 360) else hh / 60.0
    wi = math.floor(hx)
    f = hx - wi
    rnd = lambda x: int(math.floor(x + 0.5))  # noqa: E731  SkScalarRoundToInt
    p = rnd((1 - s) * vb)
    q = rnd((1 - s * f) * vb)
    t = rnd((1 - s * (1 - f)) * vb)
    return [(vb, t, p), (q, vb, p), (p, vb, t), (p, q, vb), (t, p, vb)][wi] if wi < 5 else (vb, p, q)


def square_harmony(r: int, g: int, b: int) -> list[tuple[int, int, int]]:
    hh, s, v = _sk_rgb_to_hsv(r, g, b)
    return [_sk_hsv_to_rgb((hh + k) % 360, s, v) for k in (0, 90, 180, 270)]


_LUM_W = np.array([0.11, 0.59, 0.30], F32)  # cairo (0.3, 0.59, 0.11) on channels read as BGR


def _lum(c):
    return c @ _LUM_W


def _clip_color(c, a):
    lum = _lum(c)[..., None]
    n = c.min(-1, keepdims=True)
    x = c.max(-1, keepdims=True)
    c = np.where(n < 0, lum + (c - lum) * lum / np.maximum(lum - n, 1e-12), c)
    c = np.where(x > a, lum + (c - lum) * (a - lum) / np.maximum(x - lum, 1e-12), c)
    return c


def _set_lum(c, a, lum):
    return _clip_color(c + (lum - _lum(c))[..., None], a)


def _set_sat(c, sat):
    mx = c.max(-1, keepdims=True)
    mn = c.min(-1, keepdims=True)
    rng_ = mx - mn
    return np.where(rng_ > 0, (c - mn) * sat[..., None] / np.maximum(rng_, 1e-12), 0.0)


def _blend(d, da, s, sa, mode: str):
    """pixman PDF blend on premultiplied colour: (1-sa)d + (1-da)s + B."""
    if mode == "multiply":
        b = s * d
    elif mode == "difference":
        b = np.abs(s * da - d * sa)
    elif mode == "color_burn":
        with np.errstate(divide="ignore", invalid="ignore"):
            v = sa * (da - sa * (da - d) / np.where(s == 0, 1, s))
        b = np.where(d >= da, sa * da,
                     np.where(sa * (da - d) >= s * da, 0.0, np.where(s == 0, 0.0, v)))
    elif mode == "soft_light":
        dda = np.where(da > 0, d / np.maximum(da, 1e-12), 0.0)
        low = d * sa - d * (da - d) * (sa - 2 * s) / np.maximum(da, 1e-12)
        hi1 = d * sa + (2 * s - sa) * d * ((16 * dda - 12) * dda + 3)
        hi2 = d * sa + (np.sqrt(d * da) - d) * (2 * s - sa)
        b = np.where(da <= 0, d * sa, np.where(2 * s < sa, low, np.where(4 * d <= da, hi1, hi2)))
    else:  # non-separable HSL modes (pixman blend_hsl_*), dc/sc premultiplied
        dc, sc = d, s
        a = sa * da
        if mode == "hsl_hue":
            c = _set_sat(sc * da, (dc.max(-1) - dc.min(-1)) * sa)
            b = _set_lum(c, a, _lum(dc) * sa)
        else:  # hsl_color
            b = _set_lum(sc * da, a, _lum(dc) * sa)
    return (1 - sa) * d + (1 - da) * s + b


def pop_art(img: np.ndarray, color, strength: float, mode: str) -> np.ndarray:
    h, w = img.shape[:2]
    hw, hh = w // 2, h // 2
    src = premultiplied8(img).astype(F32) / 255.0
    out = np.zeros((h, w, 4), F32)
    if hw < 1 or hh < 1:
        return out
    half = cv2.resize(src, (hw, hh), interpolation=cv2.INTER_AREA)
    r1, g1, b1 = color255(color)[:3]
    harm = square_harmony(r1, g1, b1)
    # cairo_set_source_rgba(...) argument orders per quadrant, mapped back to RGBA memory
    quads = [((0, 0), harm[0]),                                    # (b,g,r) on BGRA -> true colour
             ((hw, 0), (harm[1][2], harm[1][1], harm[1][0])),      # (r,g,b) -> appears as (b,g,r)
             ((0, hh), (harm[2][0], harm[2][2], harm[2][1])),      # (g,b,r) -> appears as (r,b,g)
             ((hw, hh), (harm[3][2], harm[3][0], harm[3][1]))]     # (g,r,b) -> appears as (b,r,g)
    sa = F32(strength)
    for (ox, oy), col in quads:
        d = half[..., :3]
        da = half[..., 3:4]
        s = np.array(col, F32) / 255.0 * sa
        rgb = _blend(d, da, np.broadcast_to(s, d.shape), sa, mode)
        alpha = sa + da * (1 - sa)
        out[oy:oy + hh, ox:ox + hw] = np.dstack([np.clip(rgb, 0, 1), alpha])
    a = out[..., 3:4]
    rgb = np.where(a > 0, out[..., :3] / np.maximum(a, 1e-8), 0.0)
    return np.dstack([np.clip(rgb, 0, 1), a]).astype(F32)


# ---------------------------------------------------------------------------
# jhlabs (Rust port) CellularFilter / PointillizeFilter
# ---------------------------------------------------------------------------

_M48 = (1 << 48) - 1


class _JavaRandom:
    def __init__(self, seed: int):
        self.s = (seed ^ 0x5DEECE66D) & _M48

    def next(self, bits: int) -> int:
        self.s = (self.s * 0x5DEECE66D + 0xB) & _M48
        v = self.s >> (48 - bits)
        return v - (1 << 32) if v >= (1 << 31) else v

    def next_int(self) -> int:
        return self.next(32)


@functools.lru_cache(maxsize=1)
def _perlin_tables() -> tuple[np.ndarray, np.ndarray]:
    rnd = _JavaRandom(0x4A484C616273)
    perm = np.zeros(514, np.int64)
    g2 = np.zeros((514, 2), F32)
    pos = lambda: rnd.next_int() & 0x7FFFFFFF  # noqa: E731
    for i in range(256):
        perm[i] = i
        pos()  # gradient1 (unused here) consumes one value
        g2[i, 0] = F32((pos() % 512) - 256) / F32(256.0)
        g2[i, 1] = F32((pos() % 512) - 256) / F32(256.0)
        ln = np.sqrt(g2[i, 0] * g2[i, 0] + g2[i, 1] * g2[i, 1], dtype=F32)
        g2[i] /= ln
        for _ in range(3):  # gradient3 consumes three values
            pos()
    for i in range(255, -1, -1):
        t = pos() % 256
        perm[i], perm[t] = perm[t], perm[i]
    for i in range(258):
        perm[256 + i] = perm[i]
        g2[256 + i] = g2[i]
    return perm, g2


@functools.lru_cache(maxsize=1)
def _cellular_probabilities() -> np.ndarray:
    vals = np.zeros(8192, np.int64)
    fact = F32(1.0)
    total = F32(0.0)
    for i in range(10):
        if i > 1:
            fact = F32(fact * F32(i))
        p = F32(F32(F32(2.5) ** i) * F32(np.exp(F32(-2.5))) / fact)
        start = int(total * F32(8192.0))
        total = F32(total + p)
        end = int(total * F32(8192.0))
        vals[min(start, 8192):min(end, 8192)] = i
    return vals


@njit(cache=True, inline="always")
def _jr_next(state: np.int64, bits: int):
    state = (state * np.int64(0x5DEECE66D) + np.int64(0xB)) & np.int64(_M48)
    v = state >> np.int64(48 - bits)
    if bits == 32 and v >= np.int64(1) << np.int64(31):
        v -= np.int64(1) << np.int64(32)
    return state, v


@njit(cache=True)
def _noise2(x, y, perm, g2):
    sx = x + np.float32(4096.0)
    ix = np.int64(np.int32(sx))
    x0 = ix & 255
    x1 = (x0 + 1) & 255
    rx = sx - np.float32(ix)
    nx = rx - np.float32(1.0)
    sy = y + np.float32(4096.0)
    iy = np.int64(np.int32(sy))
    y0 = iy & 255
    y1 = (y0 + 1) & 255
    ry = sy - np.float32(iy)
    ny = ry - np.float32(1.0)
    f = perm[x0]
    s = perm[x1]
    c0 = perm[f + y0]
    c1 = perm[s + y0]
    c2 = perm[f + y1]
    c3 = perm[s + y1]
    cx = rx * rx * (np.float32(3.0) - np.float32(2.0) * rx)
    cy = ry * ry * (np.float32(3.0) - np.float32(2.0) * ry)
    tl = rx * g2[c0, 0] + ry * g2[c0, 1]
    tr = nx * g2[c1, 0] + ry * g2[c1, 1]
    bl = rx * g2[c2, 0] + ny * g2[c2, 1]
    br = nx * g2[c3, 0] + ny * g2[c3, 1]
    top = tl + cx * (tr - tl)
    bot = bl + cx * (br - bl)
    return np.float32(1.5) * (top + cy * (bot - top))


@njit(cache=True)
def _check_cube(x, y, cube_x, cube_y, grid, randomness, res, probs, perm, g2):
    """cellular_check_cube; res rows: [distance, dx, dy, px, py] sorted by distance."""
    state = (np.int64(571) * np.int64(cube_x) + np.int64(23) * np.int64(cube_y)) ^ np.int64(0x5DEECE66D)
    state &= np.int64(_M48)
    if grid == 1 or grid == 2:
        npts = 1
    elif grid == 3 or grid == 4:
        npts = 2
    else:
        state, v = _jr_next(state, 32)
        npts = probs[v & 0x1FFF]
    for index in range(npts):
        weight = np.float32(1.0)
        if grid == 1:
            px = np.float32(0.5)
            py = np.float32(0.5)
        elif grid == 2:
            px = np.float32(0.75)
            py = np.float32(0.0) if (cube_x & 1) == 0 else np.float32(0.5)
        elif grid == 3:
            if index == 0:
                px = np.float32(0.207)
                py = np.float32(0.207)
            else:
                px = np.float32(0.707)
                py = np.float32(0.707)
                weight = np.float32(1.6)
        elif grid == 4:
            if (cube_y & 1) == 0:
                if index == 0:
                    px = np.float32(0.25)
                    py = np.float32(0.35)
                else:
                    px = np.float32(0.75)
                    py = np.float32(0.65)
            else:
                if index == 0:
                    px = np.float32(0.75)
                    py = np.float32(0.35)
                else:
                    px = np.float32(0.25)
                    py = np.float32(0.65)
        else:
            state, v = _jr_next(state, 24)
            px = np.float32(v) / np.float32(16777216.0)
            state, v = _jr_next(state, 24)
            py = np.float32(v) / np.float32(16777216.0)
        if randomness != 0.0:
            if grid == 1:
                state, v = _jr_next(state, 24)
                px += randomness * (np.float32(v) / np.float32(16777216.0) - np.float32(0.5))
                state, v = _jr_next(state, 24)
                py += randomness * (np.float32(v) / np.float32(16777216.0) - np.float32(0.5))
            elif grid != 0:
                px += randomness * _noise2(np.float32(271.0) * (np.float32(cube_x) + px),
                                           np.float32(271.0) * (np.float32(cube_y) + py), perm, g2)
                py += randomness * _noise2(np.float32(271.0) * (np.float32(cube_x) + px) + np.float32(89.0),
                                           np.float32(271.0) * (np.float32(cube_y) + py) + np.float32(137.0),
                                           perm, g2)
        dx = abs(x - px) * weight
        dy = abs(y - py) * weight
        d = np.sqrt(dx * dx + dy * dy)  # distancePower = 2
        pxx = np.float32(cube_x) + px
        pyy = np.float32(cube_y) + py
        if d < res[0, 0]:
            for k in range(5):
                res[2, k] = res[1, k]
                res[1, k] = res[0, k]
            res[0, 0] = d; res[0, 1] = dx; res[0, 2] = dy; res[0, 3] = pxx; res[0, 4] = pyy  # noqa: E702
        elif d < res[1, 0]:
            for k in range(5):
                res[2, k] = res[1, k]
            res[1, 0] = d; res[1, 1] = dx; res[1, 2] = dy; res[1, 3] = pxx; res[1, 4] = pyy  # noqa: E702
        elif d < res[2, 0]:
            res[2, 0] = d; res[2, 1] = dx; res[2, 2] = dy; res[2, 3] = pxx; res[2, 4] = pyy  # noqa: E702
    return res[2, 0]


@njit(cache=True)
def _cellular_evaluate(x, y, grid, randomness, res, probs, perm, g2):
    for k in range(3):
        res[k, 0] = np.inf
        res[k, 1] = 0.0
        res[k, 2] = 0.0
        res[k, 3] = 0.0
        res[k, 4] = 0.0
    ix = np.int64(np.int32(x))
    iy = np.int64(np.int32(y))
    fx = x - np.float32(ix)
    fy = y - np.float32(iy)
    one = np.float32(1.0)
    d = _check_cube(fx, fy, ix, iy, grid, randomness, res, probs, perm, g2)
    if d > fy:
        d = _check_cube(fx, fy + one, ix, iy - 1, grid, randomness, res, probs, perm, g2)
    if d > one - fy:
        d = _check_cube(fx, fy - one, ix, iy + 1, grid, randomness, res, probs, perm, g2)
    if d > fx:
        _check_cube(fx + one, fy, ix - 1, iy, grid, randomness, res, probs, perm, g2)
        if d > fy:
            d = _check_cube(fx + one, fy + one, ix - 1, iy - 1, grid, randomness, res, probs, perm, g2)
        if d > one - fy:
            d = _check_cube(fx + one, fy - one, ix - 1, iy + 1, grid, randomness, res, probs, perm, g2)
    if d > one - fx:
        d = _check_cube(fx - one, fy, ix + 1, iy, grid, randomness, res, probs, perm, g2)
        if d > fy:
            d = _check_cube(fx - one, fy + one, ix + 1, iy - 1, grid, randomness, res, probs, perm, g2)
        if d > one - fy:
            _check_cube(fx - one, fy - one, ix + 1, iy + 1, grid, randomness, res, probs, perm, g2)


@njit(cache=True)
def _smooth_step(a, b, v):
    if v < a:
        return np.float32(0.0)
    if v >= b:
        return np.float32(1.0)
    t = (v - a) / (b - a)
    return t * t * (np.float32(3.0) - np.float32(2.0) * t)


@njit(cache=True, parallel=True)
def _pointillize(src, scale, stretch, angle, grid, randomness, edge, fuzz, edge_color, probs, perm, g2):
    h, w, _ = src.shape
    out = np.empty((h, w, 4), np.uint8)
    ca = np.float32(np.cos(angle))
    sn = np.float32(np.sin(angle))
    for y in prange(h):
        res = np.empty((3, 5), np.float32)
        for x in range(w):
            nx = (ca * np.float32(x) + sn * np.float32(y)) / scale + np.float32(1000.0)
            ny = (-sn * np.float32(x) + ca * np.float32(y)) / (scale * stretch) + np.float32(1000.0)
            _cellular_evaluate(nx, ny, grid, randomness, res, probs, perm, g2)
            sx = min(max(np.int64(np.int32((res[0, 3] - np.float32(1000.0)) * scale)), 0), w - 1)
            sy = min(max(np.int64(np.int32((res[0, 4] - np.float32(1000.0)) * scale)), 0), h - 1)
            amount = np.float32(1.0) - _smooth_step(edge, edge + fuzz, res[0, 0])
            inv = np.float32(1.0) - amount
            for c in range(4):  # mix_colors(amount, edge_color, colour): truncation, clamp
                v = np.int64(np.float32(edge_color[c]) * inv + np.float32(src[sy, sx, c]) * amount)
                out[y, x, c] = min(max(v, 0), 255)
    return out


def pointillize(img: np.ndarray, scale: float, stretch: float, angle_rad: float, grid: int,
                randomness: float, edge: float, edge_color, fuzziness: float = 0.1) -> np.ndarray:
    h, w = img.shape[:2]
    a = img[..., 3] if img.shape[2] == 4 else np.ones((h, w), F32)
    src = to8(np.dstack([img[..., :3], a]))  # bitmap bytes read as straight ARGB upstream
    perm, g2 = _perlin_tables()
    ec = np.array(color255(edge_color), np.int64)
    out = _pointillize(src, np.float32(scale), np.float32(stretch), np.float32(angle_rad), int(grid),
                       np.float32(randomness), np.float32(edge), np.float32(fuzziness), ec,
                       _cellular_probabilities(), perm, g2)
    return out.astype(F32) / 255.0


# ---------------------------------------------------------------------------
# jhlabs (Rust port) StampFilter
# ---------------------------------------------------------------------------

def _gaussian_kernel_exact(radius: float) -> np.ndarray:
    half = int(math.ceil(radius))
    sigma = F32(radius / 3.0)
    s22 = F32(2.0) * sigma * sigma
    norm = F32(math.sqrt(2.0 * math.pi * float(sigma)))
    d = np.arange(-half, half + 1, dtype=F32) ** 2
    v = np.where(d > F32(radius * radius), F32(0.0), np.exp(-d / s22) / norm).astype(F32)
    return (v / v.sum()).astype(F32)


def _convolve_rows_exact(src: np.ndarray, kern: np.ndarray, premul: bool, unpremul: bool, axis: int) -> np.ndarray:
    """convolve_and_transpose_exact along ``axis`` (1 = rows, 0 = columns), ARGB as float 0..255,
    clamped edges, ``(v + 0.5) as i32`` rounding."""
    s = src.astype(F32)
    if premul:
        f = s[..., 3:4] / F32(255.0)
        s = np.concatenate([np.floor(s[..., :3] * f), s[..., 3:4]], axis=-1)
    k = kern.reshape(1, -1) if axis == 1 else kern.reshape(-1, 1)
    acc = cv2.filter2D(s, -1, k, borderType=cv2.BORDER_REPLICATE)
    if unpremul:
        a = acc[..., 3:4]
        scale = np.where((a != 0) & (a != 255), F32(255.0) / np.where(a == 0, 1, a), F32(1.0))
        acc = np.concatenate([acc[..., :3] * scale, a], axis=-1)
    return np.clip((acc + F32(0.5)).astype(np.int32), 0, 255).astype(np.uint8)


def rubber_stamp(img: np.ndarray, radius: float, threshold: float, softness: float, white, black) -> np.ndarray:
    h, w = img.shape[:2]
    a = img[..., 3] if img.shape[2] == 4 else np.ones((h, w), F32)
    px = to8(np.dstack([img[..., :3], a]))
    r = float(int(radius))
    if r > 0:
        kern = _gaussian_kernel_exact(r)
        tmp = _convolve_rows_exact(px, kern, True, False, axis=1)
        px = _convolve_rows_exact(tmp, kern, False, True, axis=0)
    bright = (px[..., 0].astype(np.int32) + px[..., 1] + px[..., 2]) // 3
    t = F32(threshold * 255.0)
    s = F32(softness * 255.0 * 0.5)
    lo, hi = t - s, t + s
    v = bright.astype(F32)
    tt = np.where(hi > lo, (v - lo) / np.where(hi > lo, hi - lo, 1), 0)
    f = np.where(v < lo, 0.0, np.where(v >= hi, 1.0, tt * tt * (3 - 2 * tt))).astype(F32)[..., None]
    bl = np.array(color255(black), F32)
    wh = np.array(color255(white), F32)
    out = np.clip((bl * (1 - f) + wh * f).astype(np.int32), 0, 255)
    return out.astype(F32) / 255.0


# ---------------------------------------------------------------------------
# GPUImage Toon / SmoothToon
# ---------------------------------------------------------------------------

def gpu_toon(img: np.ndarray, threshold: float, levels: float) -> np.ndarray:
    """GPUImageToonFilter: Sobel on the red channel (3x3 texel taps, clamp to
    edge), posterize ``floor(c*q+0.5)/q``, black where magnitude >= threshold."""
    r = cv2.copyMakeBorder(img[..., 0], 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    h, w = img.shape[:2]
    t = lambda dx, dy: r[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]  # noqa: E731
    gh = -t(-1, -1) - 2 * t(0, -1) - t(1, -1) + t(-1, 1) + 2 * t(0, 1) + t(1, 1)
    gv = -t(-1, 1) - 2 * t(-1, 0) - t(-1, -1) + t(1, 1) + 2 * t(1, 0) + t(1, -1)
    mag = np.sqrt(gh * gh + gv * gv)
    q = F32(levels)
    if q > 0:
        post = np.floor(img[..., :3] * q + F32(0.5)) / q
    else:  # 0/0 in the shader -> NaN -> written as 0
        post = np.zeros_like(img[..., :3])
    keep = (mag < F32(threshold)).astype(F32)
    return (post * keep[..., None]).astype(F32)


_GAUSS9 = np.array([0.05, 0.09, 0.12, 0.15, 0.18, 0.15, 0.12, 0.09, 0.05], F32)


def _gpu_blur_pass(img: np.ndarray, step: float, axis: int) -> np.ndarray:
    """9 taps at ``m * step`` px (m = -4..4) with GL_LINEAR + CLAMP_TO_EDGE sampling, written
    as one FIR kernel (linear-interpolation weights distributed to integer offsets)."""
    offs = np.arange(-4, 5) * float(step)
    lo = int(np.floor(offs.min()))
    kern = np.zeros(int(np.floor(offs.max())) - lo + 2, np.float64)
    for wk, o in zip(_GAUSS9, offs):
        k = int(np.floor(o))
        f = o - k
        kern[k - lo] += wk * (1 - f)
        kern[k - lo + 1] += wk * f
    k2 = kern.astype(F32).reshape(1, -1) if axis == 1 else kern.astype(F32).reshape(-1, 1)
    anchor = (-lo, 0) if axis == 1 else (0, -lo)
    acc = cv2.filter2D(np.ascontiguousarray(img[..., :3]), -1, k2, anchor=anchor, borderType=cv2.BORDER_REPLICATE)
    return np.rint(np.clip(acc, 0, 1) * 255) / 255  # 8-bit framebuffer


def gpu_smooth_toon(img: np.ndarray, blur_size: float, threshold: float, levels: float) -> np.ndarray:
    """GPUImageSmoothToonFilter = GPUImageGaussianBlurFilter(blurSize) -> Toon.

    The blur's vertex shader swaps the texel offsets: the first pass steps
    vertically by ``blurSize*H/W`` px, the second horizontally by
    ``blurSize*W/H`` px (9 linear taps, clamp to edge).
    """
    h, w = img.shape[:2]
    b = _gpu_blur_pass(img, blur_size * h / w, axis=0)
    b = _gpu_blur_pass(b.astype(F32), blur_size * w / h, axis=1)
    return gpu_toon(b.astype(F32), threshold, levels)
