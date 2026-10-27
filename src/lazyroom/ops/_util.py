"""Shared building blocks for operation implementations.

Everything here works on float32 channels-last buffers (see core.image).
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from ..core.color import linear_to_srgb, srgb_to_linear, LUMA_709
from ..core.errors import ErrorCode, LazyRoomError

F32 = np.float32


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(F32, copy=False)


def lerp(a, b, t):
    return a + (b - a) * t


def luma(img: np.ndarray) -> np.ndarray:
    return (img[..., :3] @ LUMA_709).astype(F32)


def gaussian(img: np.ndarray, sigma: float, border: int = cv2.BORDER_REFLECT101) -> np.ndarray:
    if sigma <= 0.05:
        return img
    out = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma), borderType=border)
    return out.reshape(img.shape).astype(F32, copy=False)


def box(img: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return img
    k = 2 * int(radius) + 1
    return cv2.blur(img, (k, k), borderType=cv2.BORDER_REFLECT101).reshape(img.shape).astype(F32, copy=False)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float = 1e-3) -> np.ndarray:
    """He et al. guided filter (grayscale guide). Edge-aware smoothing in O(N)."""
    I = guide.astype(F32)
    p = src.astype(F32)
    r = max(1, int(radius))
    mean_I = box(I, r)
    mean_p = box(p, r)
    corr_I = box(I * I, r)
    corr_Ip = box(I * p, r)
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return (box(a, r) * I + box(b, r)).astype(F32)


def apply_1d_lut(x: np.ndarray, lut: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Apply a sampled curve (len N) mapping [lo, hi] -> values, linear interpolation, clamped input."""
    n = len(lut) - 1
    t = np.clip((x - lo) / (hi - lo), 0, 1) * n
    i0 = np.floor(t).astype(np.int32)
    i0 = np.minimum(i0, n - 1)
    f = (t - i0).astype(F32)
    lut = lut.astype(F32)
    return (lut[i0] * (1 - f) + lut[i0 + 1] * f).astype(F32)


def monotone_curve(points: Sequence[Sequence[float]], n: int = 4096) -> np.ndarray:
    """Monotone cubic (PCHIP) curve through control points, sampled on [0,1]."""
    pts = sorted((float(p[0]), float(p[1])) for p in points)
    xs, ys = [], []
    for x, y in pts:  # dedupe x
        if xs and abs(x - xs[-1]) < 1e-6:
            ys[-1] = y
        else:
            xs.append(x)
            ys.append(y)
    grid = np.linspace(0, 1, n, dtype=np.float64)
    if len(xs) == 0:
        return grid.astype(F32)
    if len(xs) == 1:
        return np.full(n, ys[0], F32)
    if len(xs) == 2:
        return np.interp(grid, xs, ys).astype(F32)
    from scipy.interpolate import PchipInterpolator

    f = PchipInterpolator(np.array(xs), np.array(ys), extrapolate=False)
    out = f(np.clip(grid, xs[0], xs[-1]))
    out = np.where(grid < xs[0], ys[0], np.where(grid > xs[-1], ys[-1], out))
    return out.astype(F32)


def smooth_curve(points: Sequence[Sequence[float]], n: int = 4096) -> np.ndarray:
    """Natural cubic spline (allows overshoot, like many editors' curves); clamped to [0,1]."""
    pts = sorted((float(p[0]), float(p[1])) for p in points)
    if len(pts) < 3:
        return monotone_curve(pts, n)
    from scipy.interpolate import CubicSpline

    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    keep = np.concatenate([[True], np.diff(xs) > 1e-6])
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 3:
        return monotone_curve(list(zip(xs, ys)), n)
    cs = CubicSpline(xs, ys, bc_type="natural")
    grid = np.linspace(0, 1, n)
    out = cs(np.clip(grid, xs[0], xs[-1]))
    out = np.where(grid < xs[0], ys[0], np.where(grid > xs[-1], ys[-1], out))
    return np.clip(out, 0, 1).astype(F32)


def in_linear(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """Run ``fn`` on linear-light values and return encoded values."""

    @functools.wraps(fn)
    def wrapper(rgb: np.ndarray) -> np.ndarray:
        return linear_to_srgb(fn(srgb_to_linear(rgb)))

    return wrapper


def preserve_luma_scale(rgb: np.ndarray, new_luma: np.ndarray, old_luma: np.ndarray | None = None) -> np.ndarray:
    """Scale RGB so that its luma becomes ``new_luma`` while keeping chroma ratios."""
    if old_luma is None:
        old_luma = luma(rgb)
    ratio = new_luma / np.maximum(old_luma, 1e-5)
    out = rgb * ratio[..., None]
    # keep near-black pixels stable
    return np.where(old_luma[..., None] < 1e-5, new_luma[..., None], out).astype(F32)


def zone_weights(y: np.ndarray, balance: float = 0.0, blending: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft shadows / midtones / highlights weights from a 0..1 lightness map.

    ``balance`` (-1..1) moves the shadow/highlight split; ``blending`` (0..1)
    controls the overlap width.
    """
    pivot = 0.5 + 0.25 * balance
    width = 0.15 + 0.35 * blending
    sh = np.clip((pivot - y) / width + 0.5, 0, 1)
    hi = np.clip((y - pivot) / width + 0.5, 0, 1)
    sh = sh * sh * (3 - 2 * sh)
    hi = hi * hi * (3 - 2 * hi)
    mid = np.clip(1 - sh - hi, 0, 1)
    mid = np.maximum(mid, 4 * y * (1 - y) * (1 - np.maximum(sh, hi)))
    return sh.astype(F32), mid.astype(F32), hi.astype(F32)


def hue_to_ab(hue_deg: float, amount: float) -> tuple[float, float]:
    """OKLab (a, b) offset of given hue angle (degrees, 0=red..) and magnitude."""
    # Align hue wheel with HSV-style degrees: compute OKLab angle of the sRGB hue.
    from ..core.color import hsv_to_rgb, rgb_to_oklab

    rgb = hsv_to_rgb(np.array([[[hue_deg / 360.0 % 1.0, 1.0, 1.0]]], F32))
    lab = rgb_to_oklab(rgb)[0, 0]
    ang = np.arctan2(lab[2], lab[1])
    return float(np.cos(ang) * amount), float(np.sin(ang) * amount)


def hue_distance_weight(h: np.ndarray, center: float, width: float) -> np.ndarray:
    """Smooth circular window on hue (all in degrees)."""
    d = np.abs((h - center + 180.0) % 360.0 - 180.0)
    t = np.clip(1 - d / width, 0, 1)
    return (t * t * (3 - 2 * t)).astype(F32)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def remap(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, *, interpolation: int = cv2.INTER_LINEAR,
          border: str = "reflect", fill: Sequence[float] | None = None) -> np.ndarray:
    """cv2.remap wrapper with named border modes: reflect | clamp | wrap | transparent | constant."""
    bmodes = {"reflect": cv2.BORDER_REFLECT101, "clamp": cv2.BORDER_REPLICATE, "wrap": cv2.BORDER_WRAP,
              "constant": cv2.BORDER_CONSTANT, "transparent": cv2.BORDER_CONSTANT}
    bm = bmodes.get(border, cv2.BORDER_REFLECT101)
    src = img
    if border == "transparent" and img.shape[2] == 3:
        src = np.dstack([img, np.ones(img.shape[:2], F32)])
    value = tuple(fill) if fill is not None else (0.0,) * src.shape[2]
    if len(value) < src.shape[2]:
        value = tuple(value) + (1.0,) * (src.shape[2] - len(value))
    chans = src.shape[2]
    if chans <= 4:
        out = cv2.remap(src, map_x.astype(F32), map_y.astype(F32), interpolation, borderMode=bm,
                        borderValue=value[:4] if chans == 4 else value[:3])
    else:  # pragma: no cover
        out = np.dstack([cv2.remap(src[..., c], map_x.astype(F32), map_y.astype(F32), interpolation,
                                   borderMode=bm, borderValue=value[c]) for c in range(chans)])
    return out.astype(F32)


def pixel_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """(X, Y) pixel-centre coordinates as float32 meshgrids."""
    return np.meshgrid(np.arange(w, dtype=F32), np.arange(h, dtype=F32))


def norm_grid(h: int, w: int, aspect: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates centred on the image, y in [-1,1]; x scaled by aspect if requested."""
    X, Y = pixel_grid(h, w)
    nx = (X + 0.5) / w * 2 - 1
    ny = (Y + 0.5) / h * 2 - 1
    if aspect:
        nx = nx * (w / h)
    return nx.astype(F32), ny.astype(F32)


# ---------------------------------------------------------------------------
# LUTs
# ---------------------------------------------------------------------------

def apply_lut3d(rgb: np.ndarray, lut: np.ndarray, domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Trilinear 3D LUT. ``lut`` shape (N, N, N, 3) indexed [r, g, b]."""
    n = lut.shape[0]
    dmin = np.asarray(domain_min, F32)
    dmax = np.asarray(domain_max, F32)
    x = np.clip((rgb[..., :3] - dmin) / (dmax - dmin), 0, 1) * (n - 1)
    i0 = np.minimum(np.floor(x).astype(np.int32), n - 2)
    f = (x - i0).astype(F32)
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    L = lut.astype(F32)
    c000 = L[r0, g0, b0]
    c100 = L[r0 + 1, g0, b0]
    c010 = L[r0, g0 + 1, b0]
    c110 = L[r0 + 1, g0 + 1, b0]
    c001 = L[r0, g0, b0 + 1]
    c101 = L[r0 + 1, g0, b0 + 1]
    c011 = L[r0, g0 + 1, b0 + 1]
    c111 = L[r0 + 1, g0 + 1, b0 + 1]
    c00 = c000 + (c100 - c000) * fr
    c10 = c010 + (c110 - c010) * fr
    c01 = c001 + (c101 - c001) * fr
    c11 = c011 + (c111 - c011) * fr
    c0 = c00 + (c10 - c00) * fg
    c1 = c01 + (c11 - c01) * fg
    return (c0 + (c1 - c0) * fb).astype(F32)


@functools.lru_cache(maxsize=64)
def load_cube(path: str) -> tuple[np.ndarray, tuple, tuple, np.ndarray | None]:
    """Parse an Adobe/Resolve .cube file -> (lut3d[r,g,b,3], dmin, dmax, lut1d|None)."""
    size3 = size1 = None
    dmin, dmax = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    rows: list[list[float]] = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise LazyRoomError(ErrorCode.FILE_NOT_FOUND, f"Cannot read LUT {path}: {exc}") from exc
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        k = s.split()
        key = k[0].upper()
        if key == "LUT_3D_SIZE":
            size3 = int(k[1])
        elif key == "LUT_1D_SIZE":
            size1 = int(k[1])
        elif key == "DOMAIN_MIN":
            dmin = [float(v) for v in k[1:4]]
        elif key == "DOMAIN_MAX":
            dmax = [float(v) for v in k[1:4]]
        elif key in ("TITLE", "LUT_3D_INPUT_RANGE", "LUT_1D_INPUT_RANGE"):
            if key.endswith("INPUT_RANGE"):
                dmin = [float(k[1])] * 3
                dmax = [float(k[2])] * 3
            continue
        else:
            try:
                rows.append([float(v) for v in k[:3]])
            except ValueError:
                continue
    data = np.array(rows, dtype=F32)
    if size3:
        if len(data) < size3 ** 3:
            raise LazyRoomError(ErrorCode.DECODE_FAILED, f"LUT {path}: expected {size3 ** 3} rows, got {len(data)}")
        # .cube: red changes fastest -> reshape to [b, g, r] then transpose to [r, g, b]
        lut = data[: size3 ** 3].reshape(size3, size3, size3, 3).transpose(2, 1, 0, 3).copy()
        return lut, tuple(dmin), tuple(dmax), None
    if size1:
        return np.zeros((2, 2, 2, 3), F32), tuple(dmin), tuple(dmax), data[:size1]
    raise LazyRoomError(ErrorCode.DECODE_FAILED, f"{path} is not a valid .cube LUT (no LUT_3D_SIZE)")


def lut_from_hald_or_square(img: np.ndarray) -> np.ndarray:
    """Convert a LUT image to a 3D LUT.

    Supports the GPUImage 512x512 layout (8x8 tiles of 64x64, blue = tile index,
    red = x, green = y) and square Hald CLUT images (level^3 x level^3).
    """
    h, w = img.shape[:2]
    rgb = img[..., :3].astype(F32)
    if (h, w) == (512, 512):
        n = 64
        lut = np.empty((n, n, n, 3), F32)
        for b in range(n):
            ty, tx = divmod(b, 8)
            tile = rgb[ty * 64:(ty + 1) * 64, tx * 64:(tx + 1) * 64]  # [g, r]
            lut[:, :, b] = tile.transpose(1, 0, 2)
        return lut
    level = round((h * w) ** (1 / 6))
    n = level * level
    if h == w and n ** 3 == h * w:
        flat = rgb.reshape(-1, 3)  # r fastest, then g, then b
        return flat.reshape(n, n, n, 3).transpose(2, 1, 0, 3).copy()
    raise LazyRoomError(ErrorCode.UNSUPPORTED_FORMAT,
                          f"LUT image {w}x{h} is neither a 512x512 GPUImage lookup nor a Hald CLUT")


def identity_lut(n: int = 33) -> np.ndarray:
    g = np.linspace(0, 1, n, dtype=F32)
    r, gg, b = np.meshgrid(g, g, g, indexing="ij")
    return np.stack([r, gg, b], axis=-1)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def value_noise(h: int, w: int, scale: float, rng: np.random.Generator, octaves: int = 1) -> np.ndarray:
    """Smooth fractal noise in 0..1 using bicubic-upscaled random grids."""
    out = np.zeros((h, w), F32)
    amp, total = 1.0, 0.0
    s = max(1.0, scale)
    for _ in range(max(1, octaves)):
        gh, gw = max(2, int(np.ceil(h / s)) + 2), max(2, int(np.ceil(w / s)) + 2)
        grid = rng.random((gh, gw), dtype=np.float32)
        up = cv2.resize(grid, (int(gw * s), int(gh * s)), interpolation=cv2.INTER_CUBIC)
        out += amp * up[:h, :w]
        total += amp
        amp *= 0.5
        s = max(1.0, s / 2)
    return np.clip(out / total, 0, 1).astype(F32)
