"""ImageToolbox "LUT" group filters.

Every bundled look (and "512x512 2D LUT") is ``LUT512x512Filter`` ->
Trickle ``applyLut``: each channel is reduced to 6 bits (``v / 4``), the
512x512 lookup image is read at ``x = (b % 8) * 64 + r``, ``y = (b / 8) * 64 + g``
(nearest cell, no interpolation) and the result is
``round(q*4 + strength * (lut - q*4))``.  "3D LUT" is Trickle ``applyCubeLut``
(trilinear .cube lookup, result truncated to 8 bits, mixed with the input).
"""
from __future__ import annotations

import functools

import cv2
import numpy as np
from numba import njit, prange

from ...core.errors import ErrorCode, LazyRoomError
from ...core.io import load_image
from ...core.params import File, Float
from ...core.paths import package_data_dir
from .._util import load_cube
from ._common import itb_op

GROUP = "lut"
F32 = np.float32


def _strength():
    return Float("strength", 1.0, min=0, max=1, step=0.01, title="Strength")


@njit(cache=True, inline="always")
def _u8(f):
    if f <= 0.0:
        return 0
    if f >= 1.0:
        return 255
    return int(f * 255.0 + 0.5)


@njit(cache=True, parallel=True)
def _apply_lut512(img, lut, intensity):
    """Trickle applyLutImpl (float32 math, std::round = half away from zero)."""
    h, w = img.shape[:2]
    out = np.empty((h, w, 3), np.float32)
    s = np.float32(intensity)
    for y in prange(h):
        for x in range(w):
            r = _u8(img[y, x, 0]) // 4
            g = _u8(img[y, x, 1]) // 4
            b = _u8(img[y, x, 2]) // 4
            lx = (b % 8) * 64 + r
            ly = (b // 8) * 64 + g
            q = (r, g, b)
            for c in range(3):
                base = q[c] * 4
                v = np.float32(base) + s * np.float32(lut[ly, lx, c] - base)
                out[y, x, c] = np.floor(np.float64(v) + 0.5) / 255.0
    return out


@functools.lru_cache(maxsize=1)
def _identity_lut512() -> np.ndarray:
    """ImageToolbox ``R.drawable.lookup`` (verified pixel-exact): cell value round(i * 255 / 63)."""
    y, x = np.mgrid[0:512, 0:512]
    idx = np.stack([x % 64, y % 64, (y // 64) * 8 + x // 64], axis=-1)
    return np.rint(idx * 255.0 / 63.0).astype(np.int32)


def _lut_u8(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rint(np.clip(img[..., :3], 0, 1) * 255.0).astype(np.int32))


@functools.lru_cache(maxsize=32)
def _bundled(name: str) -> np.ndarray:
    path = package_data_dir() / "luts" / f"lookup_{name}.webp"
    if not path.exists():
        raise LazyRoomError(ErrorCode.FILE_NOT_FOUND, f"Bundled LUT '{name}' missing at {path}")
    return _lut_u8(load_image(path)[0])


def apply_trickle_lut512(img: np.ndarray, lut: np.ndarray, strength: float) -> np.ndarray:
    """Apply a (512, 512, 3) int 0..255 lookup image the way Trickle ``applyLut`` does."""
    return _apply_lut512(np.ascontiguousarray(img[..., :3], dtype=F32), lut, float(strength))


# ---------------------------------------------------------------------------
# Bundled looks (each ImageToolbox filter = LUT512x512Filter(strength, R.drawable.lookup_<name>))
# ---------------------------------------------------------------------------

_BUNDLED = [
    ("Amatorka", "Amatorka", "amatorka", ["amatorka", "warm", "vintage"]),
    ("BleachBypass", "Bleach Bypass", "bleach_bypass", ["bleach bypass", "desaturated", "contrast", "film"]),
    ("Candlelight", "Candlelight", "candlelight", ["candlelight", "warm", "orange"]),
    ("Celluloid", "Celluloid", "celluloid", ["celluloid", "film", "faded"]),
    ("Coffee", "Coffee", "coffee", ["coffee", "brown", "warm"]),
    ("DropBlues", "Drop Blues", "drop_blues", ["blue", "cool", "desaturated"]),
    ("EdgyAmber", "Edgy Amber", "edgy_amber", ["amber", "orange", "contrast"]),
    ("FallColors", "Fall Colors", "fall_colors", ["autumn", "fall", "warm"]),
    ("FilmStock50", "Film Stock 50", "filmstock_50", ["film", "analog", "film stock"]),
    ("FoggyNight", "Foggy Night", "foggy_night", ["fog", "night", "cool", "faded"]),
    ("GoldenForest", "Golden Forest", "golden_forest", ["golden", "forest", "green", "warm"]),
    ("Greenish", "Greenish", "greenish", ["green", "tint"]),
    ("Kodak", "Kodak", "kodak", ["kodak", "film", "analog"]),
    ("MissEtikate", "Miss Etikate", "miss_etikate", ["miss etikate", "soft", "pastel"]),
    ("RetroYellow", "Retro Yellow", "retro_yellow", ["retro", "yellow", "vintage"]),
    ("SoftElegance", "Soft Elegance", "soft_elegance_1", ["soft", "elegant", "pastel"]),
    ("SoftEleganceVariant", "Soft Elegance Variant", "soft_elegance_2", ["soft", "elegant", "pastel"]),
]


def _make_bundled(name: str, title: str, drawable: str, tags: list[str]) -> None:
    def fn(img, ctx, strength):
        return apply_trickle_lut512(img, _bundled(drawable), strength)

    fn.__name__ = drawable
    itb_op(name, title, GROUP, [_strength()], upstream="Trickle",
           description=f"'{title}' look: ImageToolbox's bundled lookup_{drawable} 512x512 LUT "
                       "(Trickle nearest-cell lookup on 6-bit channels) blended by strength.",
           tags=["lut", "look", "preset", *tags])(fn)


for _args in _BUNDLED:
    _make_bundled(*_args)


@itb_op("LUT512x512", "512x512 2D LUT", GROUP, [
    _strength(),
    File("target_lut_image", "", file_kind="image", title="Target LUT image",
         description="Square 512x512 lookup image (8x8 tiles of 64x64; empty = identity lookup)"),
], upstream="Trickle", tags=["lut", "lookup", "color grading", "look"])
def lut512x512(img, ctx, strength, target_lut_image):
    """Apply a user 512x512 lookup image (GPUImage/ImageToolbox layout) with Trickle's nearest-cell lookup."""
    if not target_lut_image:
        lut = _identity_lut512()
    else:
        lut_img, _ = load_image(ctx.resolve_file(target_lut_image, "image"))
        h, w = lut_img.shape[:2]
        if h != w:
            raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"LUT image must be square, got {w}x{h}",
                                  details={"param": "target_lut_image"},
                                  hint="Use a 512x512 lookup image (8x8 tiles of 64x64 cells).")
        if (h, w) != (512, 512):
            interp = cv2.INTER_AREA if h > 512 else cv2.INTER_LINEAR
            lut_img = cv2.resize(lut_img[..., :3], (512, 512), interpolation=interp)
        lut = _lut_u8(lut_img)
    return apply_trickle_lut512(img, lut, strength)


# ---------------------------------------------------------------------------
# 3D LUT (.cube)
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _apply_cube(img, lut, intensity):
    """Trickle getColor + applyCubeLutImpl, float32 like the C++/Eigen code.

    ``lut`` is indexed [r, g, b]; interpolation runs along red, then green,
    then blue, the LUT value is truncated to int (``int lutR = c * 255``) and
    mixed as ``round(in * (1 - s) + lut * s)``.
    """
    h, w = img.shape[:2]
    n1 = lut.shape[0] - 1
    nf = np.float32(n1)
    s = np.float32(intensity)
    inv = np.float32(1.0) - s
    out = np.empty((h, w, 3), np.float32)
    one = np.float32(1.0)
    for y in prange(h):
        for x in range(w):
            vin = (_u8(img[y, x, 0]), _u8(img[y, x, 1]), _u8(img[y, x, 2]))
            fr = np.float32(np.float32(vin[0]) / np.float32(255.0)) * nf
            fg = np.float32(np.float32(vin[1]) / np.float32(255.0)) * nf
            fb = np.float32(np.float32(vin[2]) / np.float32(255.0)) * nf
            r0 = int(fr)
            g0 = int(fg)
            b0 = int(fb)
            r1 = min(r0 + 1, n1)
            g1 = min(g0 + 1, n1)
            b1 = min(b0 + 1, n1)
            dr = fr - np.float32(r0)
            dg = fg - np.float32(g0)
            db = fb - np.float32(b0)
            for c in range(3):
                c00 = lut[r0, g0, b0, c] * (one - dr) + lut[r1, g0, b0, c] * dr
                c01 = lut[r0, g1, b0, c] * (one - dr) + lut[r1, g1, b0, c] * dr
                c10 = lut[r0, g0, b1, c] * (one - dr) + lut[r1, g0, b1, c] * dr
                c11 = lut[r0, g1, b1, c] * (one - dr) + lut[r1, g1, b1, c] * dr
                c0 = c00 * (one - dg) + c01 * dg
                c1 = c10 * (one - dg) + c11 * dg
                cv = c0 * (one - db) + c1 * db
                lv = int(cv * np.float32(255.0))
                v = np.float32(vin[c]) * inv + np.float32(lv) * s
                o = np.floor(np.float64(v) + 0.5) if v >= 0 else np.ceil(np.float64(v) - 0.5)
                out[y, x, c] = min(max(o, 0.0), 255.0) / 255.0
    return out


@itb_op("CubeLut", "3D LUT", GROUP, [
    _strength(),
    File("target_cube_lut_file", "", file_kind="lut", title="Target 3D LUT File (.cube / .CUBE)",
         description="Adobe/Resolve .cube 3D LUT (empty = no change)"),
], upstream="Trickle", tags=["lut", "3d lut", "cube", "color grading", "look"])
def cube_lut(img, ctx, strength, target_cube_lut_file):
    """Apply a .cube 3D LUT (trilinear, DOMAIN_MIN/MAX ignored like Trickle) blended by strength."""
    if not target_cube_lut_file:
        return img
    path = ctx.resolve_file(target_cube_lut_file, "lut")
    lut, _dmin, _dmax, lut1d = load_cube(str(path))
    if lut1d is not None:
        raise LazyRoomError(ErrorCode.UNSUPPORTED_FORMAT, f"{path.name} is a 1D LUT; a 3D .cube LUT is required",
                              details={"param": "target_cube_lut_file"}, hint="Use a .cube file with LUT_3D_SIZE.")
    return _apply_cube(np.ascontiguousarray(img[..., :3], dtype=F32), np.ascontiguousarray(lut, dtype=F32),
                       float(strength))
