"""Private implementation helpers for the ImageToolbox "Distortion" group.

Ports of (all Apache-2.0 unless noted):
  * ImageToolbox procedural GLSL shaders (``ProceduralShaders.kt``) and the
    ``Anaglyph`` Canvas tool,
  * GPUImage (BSD-3) bulge / swirl / sphere-refraction / glass-sphere shaders,
  * the ImageToolboxLibs jhlabs Rust transform filters (Pinch, Ripple, Twirl,
    Water, SphereLensDistortion),
  * aire (MIT, awxkee) convex / fractal glass / marble / perlin distortion /
    water effects (incl. siv::PerlinNoise, MIT),
  * ImageToolbox opencv-tools ``SeamCarver.distort`` (forward-energy seam carving).

Buffers are float32 0..1, RGB or RGBA (straight alpha).  Coordinate-mapping
filters build inverse maps and sample with ``cv2.remap`` using the upstream
edge handling.  GPU (GLES) filters sample premultiplied textures like Android
does; RGB inputs are treated as opaque and only grow an alpha channel when the
upstream filter produces transparency.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from numba import njit

F32 = np.float32
TAU = 6.28318530718
_WORKERS = max(1, min(8, os.cpu_count() or 1))


def _parallel_rows(h: int, rows: int, fn) -> None:
    """Run ``fn(y0, y1)`` over row strips (numpy / OpenCV / nogil numba release the GIL)."""
    strips = [(y0, min(h, y0 + rows)) for y0 in range(0, h, rows)]
    if len(strips) == 1 or _WORKERS == 1:
        for s in strips:
            fn(*s)
        return
    with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
        for f in [ex.submit(fn, *s) for s in strips]:
            f.result()


# ---------------------------------------------------------------------------
# Alpha / sampling helpers
# ---------------------------------------------------------------------------

def has_alpha(img: np.ndarray) -> bool:
    return img.shape[2] == 4


def premultiply(img: np.ndarray) -> np.ndarray:
    out = np.array(img, dtype=F32, copy=True)
    out[..., :3] *= out[..., 3:4]
    return out


def unpremultiply_(rgba: np.ndarray) -> np.ndarray:
    """In-place unpremultiply of an owned RGBA buffer."""
    a = rgba[..., 3:4]
    np.divide(rgba[..., :3], a, out=rgba[..., :3], where=a > 1e-6)
    np.copyto(rgba[..., :3], 0.0, where=a <= 1e-6)
    return rgba


def with_alpha(rgb: np.ndarray, alpha: np.ndarray, had_alpha: bool) -> np.ndarray:
    """Attach ``alpha``; for RGB inputs only when the filter actually produced transparency."""
    if not had_alpha and float(alpha.min()) >= 1.0 - 1e-6:
        return rgb
    return np.dstack([rgb[..., :3], alpha]).astype(F32, copy=False)


def to_rgba(img: np.ndarray) -> np.ndarray:
    if has_alpha(img):
        return img
    return np.dstack([img, np.ones(img.shape[:2], F32)])


def finish(out: np.ndarray, had_alpha: bool) -> np.ndarray:
    if had_alpha or out.shape[2] == 3:
        return out
    return with_alpha(out[..., :3], out[..., 3], False)


def _clean(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=F32)
    if not (m.flags.writeable and m.flags.c_contiguous) or m.base is not None and not m.flags.owndata:
        m = np.array(m, dtype=F32, copy=True)
    np.nan_to_num(m, copy=False, nan=-1e4, posinf=1e4, neginf=-1e4)
    np.clip(m, -3e4, 3e4, out=m)
    return m


def remap(src: np.ndarray, mx: np.ndarray, my: np.ndarray, border: int = cv2.BORDER_REPLICATE,
          interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    out = cv2.remap(src, _clean(mx), _clean(my), interp, borderMode=border, borderValue=(0.0, 0.0, 0.0, 0.0))
    if src.ndim == 3 and out.ndim == 2:
        out = out[..., None]
    return out


def gather(img: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """``img[iy, ix]`` for integer-valued float maps already clamped to the image (nearest remap)."""
    return remap(img, ix, iy, interp=cv2.INTER_NEAREST)


def gl_texture(img: np.ndarray) -> np.ndarray:
    """What GLUtils.texImage2D uploads: premultiplied RGBA (RGB inputs are opaque, kept as RGB)."""
    return premultiply(img) if has_alpha(img) else img


def gl_output(prem: np.ndarray, had_alpha: bool) -> np.ndarray:
    return unpremultiply_(prem) if had_alpha else prem


def gl_sample(tex: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """GLES ``texture2D`` with LINEAR filtering and CLAMP_TO_EDGE at texture coords (u, v)."""
    h, w = tex.shape[:2]
    return remap(tex, u * F32(w) - F32(0.5), v * F32(h) - F32(0.5))


def jh_apply(img: np.ndarray, edge: int, mapfn) -> np.ndarray:
    """jhlabs (ImageToolboxLibs Rust) inverse mapping, bilinear.  ``mapfn(X, Y) -> (sx, sy)`` in pixels.

    edge: 1 = CLAMP; 3 = RGB_CLAMP (clamped colour, alpha 0 outside -> transparency, also for RGB input).
    """
    h, w = img.shape[:2]
    had = has_alpha(img)
    src = np.ascontiguousarray(img, dtype=F32)
    if edge == 1:
        out = np.empty_like(src)
    else:
        rgb_src = np.ascontiguousarray(src[..., :3])
        a_src = np.ascontiguousarray(src[..., 3]) if had else np.ones((h, w), F32)
        out_rgb = np.empty((h, w, 3), F32)
        out_a = np.empty((h, w), F32)
    xs = np.arange(w, dtype=F32)[None, :]

    def strip(y0, y1):
        ys = np.arange(y0, y1, dtype=F32)[:, None]
        X, Y = np.broadcast_to(xs, (y1 - y0, w)), np.broadcast_to(ys, (y1 - y0, w))
        sx, sy = mapfn(X, Y)
        sx, sy = _clean(sx), _clean(sy)
        if edge == 1:
            out[y0:y1] = remap(src, sx, sy).reshape(y1 - y0, w, -1)
        else:
            out_rgb[y0:y1] = remap(rgb_src, sx, sy)
            out_a[y0:y1] = remap(a_src, sx, sy, border=cv2.BORDER_CONSTANT)

    _parallel_rows(h, 128, strip)
    if edge == 1:
        return out
    return with_alpha(out_rgb, out_a, had)


def gl_apply(img: np.ndarray, shader) -> np.ndarray:
    """Run a GLES fragment ``shader(u, v, tex, rows) -> premultiplied RGB(A) strip`` over row strips."""
    h, w = img.shape[:2]
    had = has_alpha(img)
    tex = np.ascontiguousarray(gl_texture(img), dtype=F32)
    us = ((np.arange(w, dtype=F32) + 0.5) / w)[None, :]
    results = {}

    def strip(y0, y1):
        vs = ((np.arange(y0, y1, dtype=F32) + 0.5) / h)[:, None]
        u, v = np.broadcast_to(us, (y1 - y0, w)), np.broadcast_to(vs, (y1 - y0, w))
        results[y0] = shader(u, v, tex)

    _parallel_rows(h, 128, strip)
    out = np.concatenate([results[k] for k in sorted(results)], axis=0)
    if out.shape[2] == 4:
        return unpremultiply_(out) if had else with_alpha(out[..., :3], out[..., 3], False)
    return out


# ---------------------------------------------------------------------------
# GPUImage shaders
# ---------------------------------------------------------------------------

def gpu_bulge(img, radius, scale):
    ar = img.shape[0] / img.shape[1]
    r = max(radius, 1e-12)

    def shader(u, v, tex):
        du, dv = u - F32(0.5), v - F32(0.5)
        tv = v * F32(ar) + F32(0.5 - 0.5 * ar) - F32(0.5)
        dist = np.sqrt(du * du + tv * tv)
        percent = 1.0 - ((F32(r) - dist) / F32(r)) * F32(scale)
        percent *= percent
        inside = dist < radius
        return gl_sample(tex, np.where(inside, du * percent + F32(0.5), u),
                         np.where(inside, dv * percent + F32(0.5), v))

    return gl_apply(img, shader)


def gpu_swirl(img, radius, angle, cx, cy):
    r = max(radius, 1e-12)

    def shader(u, v, tex):
        dx, dy = u - F32(cx), v - F32(cy)
        dist = np.sqrt(dx * dx + dy * dy)
        percent = (F32(r) - dist) / F32(r)
        theta = percent * percent * F32(angle * 8.0)
        s, c = np.sin(theta), np.cos(theta)
        inside = dist < radius
        return gl_sample(tex, np.where(inside, dx * c - dy * s + F32(cx), u),
                         np.where(inside, dx * s + dy * c + F32(cy), v))

    return gl_apply(img, shader)


def gpu_sphere(img, radius, eta, glass: bool):
    """GPUImageSphereRefractionFilter / GPUImageGlassSphereFilter (centre 0.5, 0.5); transparent outside."""
    ar = img.shape[0] / img.shape[1]
    r = max(radius, 1e-12)
    ls = 1.0 / math.sqrt(1.5)

    def shader(u, v, tex):
        dx = u - F32(0.5)
        dy = v * F32(ar) + F32(0.5 - 0.5 * ar) - F32(0.5)
        d = np.sqrt(dx * dx + dy * dy)
        check = ((d <= radius) & (radius > 0)).astype(F32)  # step(distance, radius)
        dn = d / F32(r)
        depth = F32(r) * np.sqrt(np.maximum(1.0 - dn * dn, 0.0))
        ln = np.sqrt(dx * dx + dy * dy + depth * depth)
        ln[ln <= 0] = 1.0
        nx, ny, nz = dx / ln, dy / ln, depth / ln
        k = 1.0 - F32(eta * eta) * (1.0 - nz * nz)  # dot(N, I) = -nz
        f = np.where(k < 0, F32(0.0), F32(-eta) * nz + np.sqrt(np.maximum(k, 0.0)))
        rx, ry = -f * nx, -f * ny  # refract((0, 0, -1), N, eta).xy
        if glass:
            rx, ry = rx * F32(-2.0), ry * F32(-2.0)
        sample = gl_sample(tex, (rx + 1.0) * 0.5, (ry + 1.0) * 0.5)
        if not glass:
            if sample.shape[2] == 3:
                sample = np.dstack([sample, np.ones(check.shape, F32)])
            return sample * check[..., None]
        color = sample[..., :3]
        lighting = 2.5 * (1.0 - np.power(np.clip(nz, 0.0, 1.0), 0.25))
        spec = np.power(np.clip(F32(-0.5 * ls) * nx + F32(0.5 * ls) * ny + F32(ls) * nz, 0.0, 1.0), 15.0)
        color = color + (lighting + 0.8 * spec)[..., None]
        np.clip(color, 0.0, 1.0, out=color)  # 8-bit framebuffer
        return np.dstack([color * check[..., None], check]).astype(F32)

    return gl_apply(img, shader)  # glass writes (colour, 1) * check: its alpha ignores the texture


# ---------------------------------------------------------------------------
# jhlabs (ImageToolboxLibs Rust port) transform filters
# ---------------------------------------------------------------------------

def _jh_centre(w, h, cxr, cyr, radius):
    cx, cy = F32(cxr * w), F32(cyr * h)
    if radius == 0:
        radius = float(min(cx, cy))
    return float(cx), float(cy), float(radius)


def jh_pinch(img, angle, cxr, cyr, radius, amount):
    h, w = img.shape[:2]
    cx, cy, radius = _jh_centre(w, h, cxr, cyr, radius)
    r = F32(max(radius, 1e-12))

    def mapfn(X, Y):
        dx, dy = X - F32(cx), Y - F32(cy)
        d2 = dx * dx + dy * dy
        inside = (d2 <= radius * radius) & (d2 != 0)
        d = np.sqrt(d2)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            pinch = np.power(np.sin(F32(math.pi / 2) * d / r), F32(-amount))
            t = 1.0 - d / r
            ang = F32(angle) * t * t
            s, c = np.sin(ang), np.cos(ang)
            return (np.where(inside, cx + pinch * (c * dx - s * dy), X),
                    np.where(inside, cy + pinch * (s * dx + c * dy), Y))

    return jh_apply(img, 3, mapfn)


def jh_twirl(img, angle, cxr, cyr, radius):
    h, w = img.shape[:2]
    cx, cy, radius = _jh_centre(w, h, cxr, cyr, radius)

    def mapfn(X, Y):
        dx, dy = X - F32(cx), Y - F32(cy)
        d = np.sqrt(dx * dx + dy * dy)
        inside = (d <= radius) & (radius > 0)
        ang = F32(angle) * (F32(radius) - d) / F32(max(radius, 1e-12))
        s, c = np.sin(ang), np.cos(ang)
        return np.where(inside, cx + c * dx - s * dy, X), np.where(inside, cy + s * dx + c * dy, Y)

    return jh_apply(img, 1, mapfn)


def jh_water(img, wavelength, amplitude, phase, cxr, cyr, radius):
    h, w = img.shape[:2]
    cx, cy, radius = _jh_centre(w, h, cxr, cyr, radius)

    def mapfn(X, Y):
        dx, dy = X - F32(cx), Y - F32(cy)
        d = np.sqrt(dx * dx + dy * dy)
        inside = (d <= radius) & (d != 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            amount = (F32(amplitude) * np.sin(d / F32(wavelength) * F32(TAU) - F32(phase))
                      * (F32(radius) - d) / F32(max(radius, 1e-12)) / d)
            k = amount * F32(wavelength)
            return np.where(inside, X + dx * k, X), np.where(inside, Y + dy * k, Y)

    return jh_apply(img, 1, mapfn)


def jh_ripple(img, xa, xw, ya, yw):
    def mapfn(X, Y):
        return X + np.sin(Y / F32(xw)) * F32(xa), Y + np.sin(X / F32(yw)) * F32(ya)

    return jh_apply(img, 3, mapfn)


def jh_sphere_lens(img, refraction, radius, cxr, cyr):
    h, w = img.shape[:2]
    cx, cy = float(F32(cxr * w)), float(F32(cyr * h))
    radius = max(radius, 0.001)
    r2 = F32(radius * radius)
    half_pi = F32(math.pi / 2)

    def mapfn(X, Y):
        dx, dy = X - F32(cx), Y - F32(cy)
        x2, y2 = dx * dx, dy * dy
        inside = ~(y2 >= r2 - x2)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.sqrt(np.maximum((1.0 - x2 / r2 - y2 / r2) * r2, 0.0))
            z2 = z * z
            xa = np.arccos(np.clip(dx / np.sqrt(x2 + z2), -1, 1))
            sx2 = half_pi - xa - np.arcsin(np.sin(half_pi - xa) / F32(refraction))
            ya = np.arccos(np.clip(dy / np.sqrt(y2 + z2), -1, 1))
            sy2 = half_pi - ya - np.arcsin(np.sin(half_pi - ya) / F32(refraction))
            return np.where(inside, X - np.tan(sx2) * z, X), np.where(inside, Y - np.tan(sy2) * z, Y)

    return jh_apply(img, 1, mapfn)


# ---------------------------------------------------------------------------
# ImageToolbox procedural shaders (ProceduralShaders.kt)
# ---------------------------------------------------------------------------

def _fract(x):
    return x - np.floor(x)


def _hash(x, y):
    """``hash(vec2)`` from ProceduralShaders.kt, evaluated in float32 like GLSL highp."""
    hx = _fract(x * F32(0.1031))
    hy = _fract(y * F32(0.1030))
    hz = _fract(x * F32(0.0973))
    d = hx * (hy + F32(33.33)) + hy * (hz + F32(33.33)) + hz * (hx + F32(33.33))
    return _fract(((hx + d) + (hy + d)) * (hz + d))


def _hash2(x, y):
    return _hash(x, y), _hash(x + F32(17.17), y + F32(47.23))


def _angle_of(x, y):
    return np.where(x * x + y * y < 1e-8, F32(0.0), np.arctan2(y, x)).astype(F32)


def _rot(x, y, ang):
    c, s = np.cos(ang), np.sin(ang)
    return c * x - s * y, s * x + c * y


class _CellTable:
    """hash2(cell + add...) of every integer cell in [lo - 1, hi + 1], flattened (exact float32 inputs)."""

    LIMIT = 4_000_000

    def __init__(self, cx, cy):
        self.lox, self.loy = int(cx.min()) - 1, int(cy.min()) - 1
        self.nx = int(cx.max()) + 2 - self.lox
        self.ny = int(cy.max()) + 2 - self.loy
        self.ok = self.nx * self.ny <= self.LIMIT

    def table(self, add):
        X, Y = np.meshgrid(np.arange(self.lox, self.lox + self.nx, dtype=F32),
                           np.arange(self.loy, self.loy + self.ny, dtype=F32))
        for a in add:
            X, Y = X + F32(a), Y + F32(a)
        h0, h1 = _hash2(X, Y)
        return h0.ravel(), h1.ravel()

    def index(self, cx, cy):
        return ((cy - F32(self.loy)) * F32(self.nx) + (cx - F32(self.lox))).astype(np.int32)


def _cell_hash2(cx, cy, add):
    """Direct hash2 for float32 integer-valued cells (fallback for huge cell ranges)."""
    for a in add:
        cx, cy = cx + F32(a), cy + F32(a)
    return _hash2(cx, cy)


class Field:
    """``toField`` / ``sampleField`` of ProceduralShaders.kt for a w x h image.

    ``sampleField`` mirrors the texture coordinates (``1 - |mod(uv, 2) - 1|``) and
    clamps to half a texel; that is exactly bilinear sampling with OpenCV's
    BORDER_REFLECT at pixel coordinates ``uv * size - 0.5`` (period 2 * size).
    """

    def __init__(self, h, w, offset_x=0.0, offset_y=0.0, scale=1.0, rotation=0.0):
        short = max(1, min(w, h))
        self.h, self.w = h, w
        self.ax, self.ay = w / short, h / short
        self.ox, self.oy = offset_x, offset_y
        self.scale = scale
        t = math.radians(rotation)
        self.c, self.s = math.cos(t), math.sin(t)
        # d(pixel)/d(q) of the affine field -> pixel mapping
        self.jxx = self.scale * self.c / self.ax * w
        self.jxy = -self.scale * self.s / self.ax * w
        self.jyx = self.scale * self.s / self.ay * h
        self.jyy = self.scale * self.c / self.ay * h

    def to_field(self, u, v):
        dx = (u - F32(0.5 + self.ox)) * F32(self.ax)
        dy = (v - F32(0.5 + self.oy)) * F32(self.ay)
        k = 1.0 / self.scale
        return (F32(self.c * k) * dx + F32(self.s * k) * dy), (F32(-self.s * k) * dx + F32(self.c * k) * dy)

    def pixels(self, qx, qy):
        """Pixel coordinates of ``sampleField(q)`` (sample them with BORDER_REFLECT)."""
        X = F32(self.jxx) * qx + F32(self.jxy) * qy + F32((0.5 + self.ox) * self.w - 0.5)
        Y = F32(self.jyx) * qx + F32(self.jyy) * qy + F32((0.5 + self.oy) * self.h - 0.5)
        return X, Y


def _lum_gradient(L, field, qx, qy, delta):
    X, Y = field.pixels(qx, qy)

    def lum(ox, oy):
        return remap(L, X + F32(ox), Y + F32(oy), border=cv2.BORDER_REFLECT)

    dxx, dxy = delta * field.jxx, delta * field.jyx  # q + (delta, 0)
    dyx, dyy = delta * field.jxy, delta * field.jyy  # q + (0, delta)
    gx = lum(dxx, dxy) - lum(-dxx, -dxy)
    gy = lum(dyx, dyy) - lum(-dyx, -dyy)
    k = F32(1.0 / (2.0 * delta))
    return gx * k, gy * k


@njit(cache=True, nogil=True)
def _voronoi_kernel(gx, gy, t0, t1, lox, loy, nx, jitter, lx, ly, best):
    """Nearest jittered cell centre over the 3x3 neighbourhood (float32 like GLSL highp)."""
    half = np.float32(0.5)
    flox = np.float32(lox)
    floy = np.float32(loy)
    for j in range(gx.shape[0]):
        for i in range(gx.shape[1]):
            x = gx[j, i]
            y = gy[j, i]
            cx = np.float32(math.floor(x))
            cy = np.float32(math.floor(y))
            base = int(cy - floy) * nx + int(cx - flox)
            nearest = np.float32(100.0)
            bl_x = np.float32(0.0)
            bl_y = np.float32(0.0)
            bi = base
            for oy in range(-1, 2):
                for ox in range(-1, 2):
                    idx = base + oy * nx + ox
                    kx = cx + np.float32(ox)
                    ky = cy + np.float32(oy)
                    ddx = x - kx - half - (t0[idx] - half) * jitter
                    ddy = y - ky - half - (t1[idx] - half) * jitter
                    d = ddx * ddx + ddy * ddy
                    if d < nearest:
                        nearest = d
                        bl_x = ddx
                        bl_y = ddy
                        bi = idx
            lx[j, i] = bl_x
            ly[j, i] = bl_y
            best[j, i] = bi


def _voronoi(px, py, freq, seed, jitter, random_add):
    """3x3 cell search of the BrokenCells / BrokenGlass shaders.

    Returns (local.x, local.y, random.x, random.y) where random = hash2(chosen + random_add...) - 0.5.
    """
    gx, gy = px * F32(freq), py * F32(freq)
    cx, cy = np.floor(gx), np.floor(gy)
    cells = _CellTable(cx, cy)
    lx = np.zeros(px.shape, F32)
    ly = np.zeros(px.shape, F32)
    if cells.ok:
        t0, t1 = cells.table((seed,))
        best = np.empty(px.shape, np.int64)
        _voronoi_kernel(np.ascontiguousarray(gx), np.ascontiguousarray(gy), t0, t1, cells.lox, cells.loy,
                        cells.nx, np.float32(jitter), lx, ly, best)
        r0, r1 = cells.table(random_add) if tuple(random_add) != (seed,) else (t0, t1)
        rx, ry = r0.take(best), r1.take(best)
        return lx, ly, rx - F32(0.5), ry - F32(0.5)
    # huge cell range (extreme scale / aspect): direct per-pixel hashing
    nearest = np.full(px.shape, 100.0, F32)
    bx, by = cx.copy(), cy.copy()
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            kx, ky = cx + F32(ox), cy + F32(oy)
            hx, hy = _cell_hash2(kx, ky, (seed,))
            ddx = gx - kx - F32(0.5) - (hx - F32(0.5)) * F32(jitter)
            ddy = gy - ky - F32(0.5) - (hy - F32(0.5)) * F32(jitter)
            d = ddx * ddx + ddy * ddy
            upd = d < nearest
            np.copyto(nearest, d, where=upd)
            np.copyto(lx, ddx, where=upd)
            np.copyto(ly, ddy, where=upd)
            np.copyto(bx, kx, where=upd)
            np.copyto(by, ky, where=upd)
    rx, ry = _cell_hash2(bx, by, random_add)
    return lx, ly, rx - F32(0.5), ry - F32(0.5)


def _hex_local(px, py, freq):
    gx, gy = px * F32(freq), py * F32(freq)
    sx, sy = F32(1.7320508), F32(3.0)
    ax = np.mod(gx, sx) - sx * F32(0.5)
    ay = np.mod(gy, sy) - sy * F32(0.5)
    bx = np.mod(gx - sx * F32(0.5), sx) - sx * F32(0.5)
    by = np.mod(gy - sy * F32(0.5), sy) - sy * F32(0.5)
    use_a = ax * ax + ay * ay < bx * bx + by * by
    return np.where(use_a, ax, bx), np.where(use_a, ay, by)


def _effect(name, px, py, v, field, src, L, rows):
    """Return ('q', qx, qy, shade) for field distortions or ('uv', u, v) for direct texture lookups."""
    qx, qy = px, py
    shade = None
    inten = float(v.get("intensity", 0.0))
    if name == "RadialShimmer":
        r = np.sqrt(px * px + py * py)
        qx, qy = _rot(px, py, F32(inten * 0.3) * np.sin(np.power(r, F32(v["spacing"])) * F32(v["count"] * TAU)))
    elif name == "HelixWaves":
        r = np.sqrt(px * px + py * py)
        f = v["frequency"]
        phase = _angle_of(px, py) * F32(max(1.0, math.floor(f * 0.25))) + r * F32(f * TAU)
        wave = np.sin(phase)
        mode = v["mode"]
        if mode < 0.5:
            k = 1.0 + wave * F32(inten * 0.2)
            qx, qy = px * k, py * k
        elif mode < 1.5:
            qx, qy = _rot(px, py, wave * F32(inten * 0.3))
        else:
            qx = px + np.cos(phase) * F32(inten * 0.06)
            qy = py + wave * F32(inten * 0.06)
        shade = 1.0 + F32(v["lighting"] * 0.3) * np.cos(phase)
    elif name == "GridSineDistortion2":
        f = F32(v["frequency"] * TAU)
        k = F32(inten * 0.1)
        qx = px + np.sin(px * f) * np.cos(py * f) * k
        qy = py + np.sin(py * f) * np.cos(px * f) * k
    elif name == "Quicksilver":
        gx, gy = _lum_gradient(L, field, px, py, 0.005)
        wave = np.sin(L[rows] * F32(v["frequency"] * TAU))  # sampleField(p) is the pixel itself
        dk, ik = F32(v["displacement"] * 0.1), F32(inten * 0.04)
        qx = px + gx * dk + wave * ik
        qy = py + gy * dk + np.cos(wave * 3.0) * ik
    elif name == "Gallium":
        for _ in range(int(v["iterations"])):
            gx, gy = _lum_gradient(L, field, qx, qy, 0.007)
            k = F32(inten * 0.08) / (1.0 + np.sqrt(gx * gx + gy * gy))
            qx, qy = qx + gx * k, qy + gy * k
    elif name == "Rubidium":
        gx, gy = _lum_gradient(L, field, px, py, 0.008)
        b = F32(v["balance"])
        mx = gx + (-gy - gx) * b
        my = gy + (gx - gy) * b
        k = F32(v["displacement"]) / (1.0 + np.sqrt(gx * gx + gy * gy))
        col = src[rows]
        ik = F32(inten * 0.2)
        qx = px + mx * k + (col[..., 0] - col[..., 1]) * ik
        qy = py + my * k + (col[..., 2] - col[..., 1]) * ik
    elif name in ("GlassRectTiles", "GlassHexTiles", "GlassTriangleTiles"):
        f = v["frequency"]
        dist = v["distortion"]
        if name == "GlassRectTiles":
            lx = _fract(px * F32(f) + F32(0.5)) - F32(0.5)
            ly = _fract(py * F32(f) + F32(0.5)) - F32(0.5)
            k = (F32(inten) + F32(dist) * (lx * lx + ly * ly) * F32(4.0)) / F32(f)
        elif name == "GlassHexTiles":
            lx, ly = _hex_local(px, py, f)
            k = (F32(inten) + F32(dist) * (lx * lx + ly * ly)) / F32(f)
        else:
            gx = (px - py / F32(1.7320508)) * F32(f)
            gy = (py * F32(1.1547005)) * F32(f)
            cx, cy = np.floor(gx), np.floor(gy)
            off = np.where((gx - cx) + (gy - cy) < 1.0, F32(0.3333333), F32(0.6666667))
            l0x, l0y = gx - (cx + off), gy - (cy + off)
            lx, ly = l0x + l0y * F32(0.5), l0y * F32(0.8660254)
            k = (F32(inten) + F32(dist) * (lx * lx + ly * ly) * F32(3.0)) / F32(f)
        qx, qy = px + lx * k, py + ly * k
    elif name == "BrokenCells":
        f, seed = v["frequency"], v["seed"]
        lx, ly, rx, ry = _voronoi(px, py, f, seed, v["variability"], (seed,))
        k, dist = F32(inten / f), F32(v["distortion"])
        qx = px + (rx + lx * dist) * k
        qy = py + (ry + ly * dist) * k
    elif name == "BrokenGlass":
        f, seed = v["frequency"], v["seed"]
        lx, ly, rx, ry = _voronoi(px, py, f, seed, 0.8, (seed, 19.3))
        tx, ty = _rot(lx, ly, rx * F32(inten * 3.0))
        qx = px + (tx - lx + rx * F32(inten)) / F32(f)
        qy = py + (ty - ly + ry * F32(inten)) / F32(f)
    elif name == "SpiralArms":
        r = np.sqrt(px * px + py * py)
        ang = _angle_of(px, py) / F32(TAU)
        return ("uv", _fract(ang * F32(v["count"]) + r * F32(v["intensity"])), _fract(r))
    elif name == "LogarithmicSpiral":
        lr = np.log(np.maximum(np.sqrt(px * px + py * py), F32(1e-5)))
        ang = _angle_of(px, py) / F32(TAU)
        return ("uv", _fract(ang * F32(v["count"]) + lr * F32(v["intensity"])), _fract(ang + lr))
    elif name in ("SpiralDroste", "SquareSpiralDroste"):
        sa = F32(v.get("shape_aspect", 1.0))
        ptx, pty = px, (py * sa if name == "SquareSpiralDroste" else py)
        if name == "SpiralDroste":
            r = np.maximum(np.sqrt(ptx * ptx + pty * pty), F32(1e-5))
        else:
            r = np.maximum(np.maximum(np.abs(ptx), np.abs(pty)), F32(1e-5))
        ang = _angle_of(ptx, pty)
        period = F32(math.log(v["intensity"]))
        dist = F32(v["distortion"])
        lrad = np.log(r / F32(0.5)) - dist * ang
        layer = np.floor(lrad / period)
        rad = np.exp(np.mod(lrad, period)) * F32(0.5 / v["intensity"])
        if name == "SpiralDroste":
            turn = ang + dist * layer * F32(TAU)
            qx, qy = rad * np.cos(turn), rad * np.sin(turn)
        else:
            qx, qy = _rot(ptx / r * rad, pty / r * rad, dist * layer * F32(TAU))
            qy = qy / sa
    elif name == "SquareFresnel":
        r = np.maximum(np.maximum(np.abs(px), np.abs(py)), F32(1e-5))
        k = 1.0 / (1.0 + np.floor(r * F32(v["intensity"] * 2.0)))
        qx, qy = px * k, py * k
    elif name == "RadialInterpolate":
        r = np.sqrt(px * px + py * py)
        rad, wd, cnt = F32(v["radius"]), F32(v["width"]), F32(v["count"])
        inside = (np.abs(r - rad) < wd * F32(0.5)).astype(F32)  # 1 - step(width / 2, |r - radius|)
        mapped = rad + np.floor((r - rad) / wd * cnt) / cnt * wd
        k = 1.0 + (mapped / np.maximum(r, F32(1e-5)) - 1.0) * (inside * F32(inten))
        qx, qy = px * k, py * k
    else:  # pragma: no cover
        raise ValueError(name)
    return ("q", qx, qy, shade)


def procedural(img: np.ndarray, name: str, values: dict, rows_per_strip: int = 128) -> np.ndarray:
    """Run one ImageToolbox procedural distortion shader on ``img``."""
    h, w = img.shape[:2]
    had = has_alpha(img)
    src = np.ascontiguousarray(gl_texture(img), dtype=F32)
    field = Field(h, w, values.get("offset_x", 0.0), values.get("offset_y", 0.0),
                  values.get("scale", 1.0), values.get("rotation", 0.0))
    L = None
    if name in ("Quicksilver", "Gallium", "Rubidium"):
        L = np.ascontiguousarray(src[..., :3] @ np.array([0.2126, 0.7152, 0.0722], F32), dtype=F32)
    map_x = np.empty((h, w), F32)
    map_y = np.empty((h, w), F32)
    shade = np.ones((h, w), F32) if name == "HelixWaves" else None
    uu = ((np.arange(w, dtype=F32) + 0.5) / w)[None, :]

    def strip(y0, y1):
        vv = ((np.arange(y0, y1, dtype=F32) + 0.5) / h)[:, None]
        px, py = field.to_field(uu, vv)
        res = _effect(name, px, py, values, field, src, L, slice(y0, y1))
        if res[0] == "uv":
            map_x[y0:y1] = res[1] * F32(w) - F32(0.5)
            map_y[y0:y1] = res[2] * F32(h) - F32(0.5)
        else:
            map_x[y0:y1], map_y[y0:y1] = field.pixels(res[1], res[2])
            if shade is not None:
                shade[y0:y1] = res[3]

    _parallel_rows(h, rows_per_strip, strip)
    out = remap(src, map_x, map_y, border=cv2.BORDER_REFLECT)
    if shade is not None:
        out[..., :3] *= shade[..., None]
        np.clip(out[..., :3], 0.0, out[..., 3:4] if had else 1.0, out=out[..., :3])  # 8-bit framebuffer
    return gl_output(out, had)


# ---------------------------------------------------------------------------
# Anaglyph (ImageToolbox GlitchTool, Android Canvas with PorterDuff ADD)
# ---------------------------------------------------------------------------

def anaglyph(img: np.ndarray, shift: int) -> np.ndarray:
    """Red layer (shader translated by -t: pixel x shows x + t, REPEAT), cyan layer (x - t), then the
    original drawn again with the cyan colour matrix (drawBitmap ignores the shader), all PorterDuff ADD."""
    w = img.shape[1]
    t = shift % w if w else 0
    if not has_alpha(img):
        out = np.empty_like(img, dtype=F32)
        out[..., 0] = np.roll(img[..., 0], -t, axis=1)
        for c in (1, 2):
            np.add(np.roll(img[..., c], t, axis=1), img[..., c], out=out[..., c])
        np.minimum(out, 1.0, out=out)
        return out
    prem = premultiply(img)
    left = np.roll(prem, -t, axis=1)
    right = np.roll(prem, t, axis=1)
    out = np.empty_like(prem)
    out[..., 0] = left[..., 0]
    out[..., 1:3] = right[..., 1:3] + prem[..., 1:3]
    out[..., 3] = left[..., 3] + right[..., 3] + prem[..., 3]
    np.minimum(out, 1.0, out=out)
    return unpremultiply_(out)


# ---------------------------------------------------------------------------
# aire effects
# ---------------------------------------------------------------------------

def _stack_blur_rgb(rgb: np.ndarray, radius: int) -> np.ndarray:
    """Stack blur == separable triangle kernel (weights r+1-|i|) with clamped edges."""
    if radius < 1:
        return rgb
    k = (radius + 1 - np.abs(np.arange(-radius, radius + 1))).astype(np.float64)
    k = (k / k.sum()).astype(F32)
    return cv2.sepFilter2D(rgb, -1, k, k, borderType=cv2.BORDER_REPLICATE)


def _bezier_lut(p0, p1, p2, p3, n=4097):
    """aire BezierInterpolator.getInterpolation sampled on [0, 1] (Newton solve, 13 steps, tol 1e-3)."""
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    cx = 3 * p0
    bx = 3 * (p2 - p0) - cx
    axx = 1 - cx - bx
    cy = 3 * p1
    by = 3 * (p3 - p1) - cy
    ay = 1 - cy - by
    x = t.copy()
    active = np.ones(n, bool)
    for _ in range(1, 14):
        z = x * (cx + x * (bx + x * axx)) - t
        active &= ~(np.abs(z) < 1e-3)
        d = cx + x * (2 * bx + 3 * axx * x)
        x = np.where(active, x - z / np.where(d == 0, 1e-12, d), x)
    return t, (x * (cy + x * (by + x * ay))).astype(F32)


_EASE_OUT_CUBIC = _bezier_lut(0.33, 1.0, 0.68, 1.0)


def aire_convex(img: np.ndarray, strength: float, scale: float) -> np.ndarray:
    """aire ConvexEffect; ``scale`` = preview scale (distances are measured in full-resolution pixels)."""
    h, w = img.shape[:2]
    src = np.ascontiguousarray(img, dtype=F32)
    blurred = src.copy()
    blurred[..., :3] = _stack_blur_rgb(np.ascontiguousarray(src[..., :3]), max(0, int(round(27 * scale))))
    cx, cy = w / 2.0, h / 2.0
    max_d = max(min(w / 2.0, h / 2.0), 1e-12)
    k = F32(strength / 5000.0 / scale)
    dx = (np.arange(w, dtype=F32) - F32(cx))[None, :]
    out = np.empty_like(src)

    def strip(y0, y1):
        dy = (np.arange(y0, y1, dtype=F32) - F32(cy))[:, None]
        dist = np.sqrt(dx * dx + dy * dy)
        factor = 1.0 - k * dist
        nx = np.clip(np.trunc(F32(cx) + dx * factor), 0, w - 1)
        ny = np.clip(np.trunc(F32(cy) + dy * factor), 0, h - 1)
        t = np.interp(np.clip(1.0 - dist / F32(max_d), 0.0, 1.0), *_EASE_OUT_CUBIC).astype(F32)
        t = t.reshape(y1 - y0, w, 1)
        bg = gather(src, nx, ny).reshape(y1 - y0, w, -1)
        fg = gather(blurred, nx, ny).reshape(y1 - y0, w, -1)
        out[y0:y1] = fg * (1.0 - t) + bg * t  # mix(blurred, source, easeOutCubic(1 - d / maxD))

    _parallel_rows(h, 128, strip)
    return out


def aire_fractal_glass(img: np.ndarray, glass_size: float, amplitude: float) -> np.ndarray:
    h, w = img.shape[:2]
    dw = int(np.float32(w) * np.float32(glass_size))
    xs = np.arange(w)
    if dw > 0:
        counter = np.where(xs <= dw, xs, (xs - dw - 1) % dw + 1)
        disp = np.floor(counter.astype(np.float32) / np.float32(dw) * np.float32(255.0)).astype(np.float32)
    else:  # 0/0 -> NaN -> 0 on the first column, x/0 -> inf -> saturated 255 afterwards
        disp = np.where(xs == 0, 0.0, 255.0).astype(np.float32)
    displ = disp / np.float32(255.0) * (np.float32(w) * np.float32(amplitude))
    nx = np.clip(np.floor(displ + xs.astype(np.float32)).astype(np.int64) % w, 0, w - 1)
    return np.take(img, nx, axis=1)


def aire_water(img: np.ndarray, fraction: float, fx: float, ax: float, fy: float, ay: float) -> np.ndarray:
    h, w = img.shape[:2]
    x_move = int(np.float32(w) * np.float32(fraction))
    y_move = int(np.float32(h) * np.float32(fraction))
    ys = np.arange(h, dtype=np.float64)
    xs = np.arange(w, dtype=np.float64)
    dx = (x_move * np.sin(2 * np.pi * ys / h * fx)).astype(F32) * F32(ax) if ax != 0 else np.zeros(h, F32)
    dy = (y_move * np.cos(2 * np.pi * xs / w * fy)).astype(F32) * F32(ay) if ay != 0 else np.zeros(w, F32)
    nx = np.clip(np.trunc(xs.astype(F32)[None, :] + dx[:, None]), 0, w - 1)
    ny = np.clip(np.trunc(ys.astype(F32)[:, None] + dy[None, :]), 0, h - 1)
    return gather(np.ascontiguousarray(img, dtype=F32), nx, ny)


@njit(cache=True, nogil=True)
def _grad(hsh, x, y, z):
    hh = hsh & 15
    u = x if hh < 8 else y
    if hh < 4:
        v = y
    elif hh == 12 or hh == 14:
        v = x
    else:
        v = z
    return (u if (hh & 1) == 0 else -u) + (v if (hh & 2) == 0 else -v)


@njit(cache=True, nogil=True)
def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


@njit(cache=True, nogil=True)
def _lerp(a, b, t):
    return a + (b - a) * t


@njit(cache=True, nogil=True)
def _noise3(perm, x, y, z):
    fx0 = math.floor(x)
    fy0 = math.floor(y)
    fz0 = math.floor(z)
    ix = int(fx0) & 255
    iy = int(fy0) & 255
    iz = int(fz0) & 255
    fx = x - fx0
    fy = y - fy0
    fz = z - fz0
    u = _fade(fx)
    v = _fade(fy)
    w = _fade(fz)
    A = (perm[ix & 255] + iy) & 255
    B = (perm[(ix + 1) & 255] + iy) & 255
    AA = (perm[A] + iz) & 255
    AB = (perm[(A + 1) & 255] + iz) & 255
    BA = (perm[B] + iz) & 255
    BB = (perm[(B + 1) & 255] + iz) & 255
    p0 = _grad(perm[AA], fx, fy, fz)
    p1 = _grad(perm[BA], fx - 1, fy, fz)
    p2 = _grad(perm[AB], fx, fy - 1, fz)
    p3 = _grad(perm[BB], fx - 1, fy - 1, fz)
    p4 = _grad(perm[(AA + 1) & 255], fx, fy, fz - 1)
    p5 = _grad(perm[(BA + 1) & 255], fx - 1, fy, fz - 1)
    p6 = _grad(perm[(AB + 1) & 255], fx, fy - 1, fz - 1)
    p7 = _grad(perm[(BB + 1) & 255], fx - 1, fy - 1, fz - 1)
    q0 = _lerp(p0, p1, u)
    q1 = _lerp(p2, p3, u)
    q2 = _lerp(p4, p5, u)
    q3 = _lerp(p6, p7, u)
    r0 = _lerp(q0, q1, v)
    r1 = _lerp(q2, q3, v)
    return _lerp(r0, r1, w)


@njit(cache=True, nogil=True)
def _octave2d_01(perm, xs, ys, octaves, amplitude, out):
    """float(siv::PerlinNoise::octave2D_01(x, y, octaves) * amplitude) on the grid xs x ys."""
    for j in range(ys.shape[0]):
        for i in range(xs.shape[0]):
            x = xs[i]
            y = ys[j]
            res = 0.0
            amp = 1.0
            for _ in range(octaves):
                res += _noise3(perm, x, y, 0.34567) * amp
                x *= 2
                y *= 2
                amp *= 0.5
            if res <= -1.0:
                res = 0.0
            elif res >= 1.0:
                res = 1.0
            else:
                res = res * 0.5 + 0.5
            out[j, i] = res * amplitude


def aire_noise_warp(img: np.ndarray, intensity: float, turbulence: float, amplitude: float,
                    rng: np.random.Generator, scale: float, colorize: bool) -> np.ndarray:
    """aire ``marbleEffect`` (colorize=False) / ``perlinDistortion`` (colorize=True)."""
    h, w = img.shape[:2]
    perm = rng.permutation(256).astype(np.int64)  # siv::PerlinNoise is time-seeded upstream
    # noise is sampled at float(x) * 0.01f in full-resolution pixel units
    xs = (np.arange(w, dtype=F32) / F32(scale) * F32(0.01)).astype(np.float64)
    ys = (np.arange(h, dtype=F32) / F32(scale) * F32(0.01)).astype(np.float64)
    noise = np.empty((h, w), F32)
    amp = float(F32(amplitude))
    _parallel_rows(h, 64, lambda y0, y1: _octave2d_01(perm, xs, ys[y0:y1], 4, amp, noise[y0:y1]))
    disp = np.clip(np.trunc(F32(127) * (F32(1) + noise)), 0, 255).astype(np.intp)
    ang = 2 * np.pi * np.arange(256) / 256.0 * float(F32(turbulence))
    iw = F32(intensity) * F32(w)
    sin_t = (-np.sin(ang)).astype(F32) * (-iw)
    cos_t = np.cos(ang).astype(F32) * iw
    xg = np.arange(w, dtype=F32)[None, :]
    yg = np.arange(h, dtype=F32)[:, None]
    sx = np.clip(np.floor(xg + sin_t[disp]), 0, w - 1)
    sy = np.clip(np.floor(yg + cos_t[disp]), 0, h - 1)
    rgb = gather(np.ascontiguousarray(img[..., :3], dtype=F32), sx, sy)
    if colorize:
        rgb += (F32(127.0 / 255.0) * noise)[..., None]
        rgb /= F32(1.25)
    if has_alpha(img):
        a = gather(np.ascontiguousarray(img[..., 3], dtype=F32), np.broadcast_to(xg, (h, w)), sy)
        return np.dstack([rgb, a])  # aire copies alpha from (sourceY, x)
    return rgb


# ---------------------------------------------------------------------------
# Content aware distortion (opencv-tools SeamCarver.distort)
# ---------------------------------------------------------------------------

@njit(cache=True, nogil=True)
def _carve_width(img, gray, target):
    """Remove forward-energy vertical seams in place until ``target`` columns remain; returns the width."""
    H, W, C = img.shape
    cols = W
    prev = np.zeros(W)
    cur = np.zeros(W)
    back = np.zeros((H, W), np.int32)
    seam = np.zeros(H, np.int32)
    while cols > target:
        if cols <= 1:
            for r in range(H):
                seam[r] = 0
        else:
            for c in range(cols):
                prev[c] = 0.0
            for row in range(1, H):
                for col in range(cols):
                    up = gray[row - 1, col]
                    left = gray[row, col - 1 if col > 0 else 0]
                    right = gray[row, col + 1 if col < cols - 1 else cols - 1]
                    cost_up = abs(right - left)
                    cost_left = abs(up - left) + cost_up
                    cost_right = abs(up - right) + cost_up
                    best = prev[col] + cost_up
                    bc = col
                    if col > 0:
                        t = prev[col - 1] + cost_left
                        if t < best:
                            best = t
                            bc = col - 1
                    if col < cols - 1:
                        t = prev[col + 1] + cost_right
                        if t < best:
                            best = t
                            bc = col + 1
                    cur[col] = best
                    back[row, col] = bc
                tmp = prev
                prev = cur
                cur = tmp
            min_col = 0
            min_val = np.inf
            for c in range(cols):
                if prev[c] < min_val:
                    min_val = prev[c]
                    min_col = c
            col = min_col
            for row in range(H - 1, -1, -1):
                seam[row] = col
                if row > 0:
                    col = back[row, col]
        for r in range(H):
            s = seam[r]
            for c in range(s, cols - 1):
                gray[r, c] = gray[r, c + 1]
                for k in range(C):
                    img[r, c, k] = img[r, c + 1, k]
        cols -= 1
    return cols


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def seam_distort(img: np.ndarray, percent: float, max_side: int = 512) -> np.ndarray:
    amount = min(max(percent, 0.0), 95.0)
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    target_scale = 1.0 - amount / 100.0
    small = img
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        small = cv2.resize(img, (max(1, _round_half_up(w * s)), max(1, _round_half_up(h * s))),
                           interpolation=cv2.INTER_AREA)
        if small.ndim == 2:
            small = small[..., None]
    sh, sw = small.shape[:2]
    tw = max(1, _round_half_up(sw * target_scale))
    th = max(1, _round_half_up(sh * target_scale))
    u8 = np.clip(small[..., :3] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float64)  # COLOR_RGBA2GRAY of the 8-bit Mat
    data = np.array(small, dtype=F32, copy=True)  # carved in place
    nw = _carve_width(data, gray, tw)
    data = np.ascontiguousarray(data[:, :nw].transpose(1, 0, 2))
    gray = np.ascontiguousarray(gray[:, :nw].T)
    nh = _carve_width(data, gray, th)
    data = np.ascontiguousarray(data[:, :nh].transpose(1, 0, 2))
    out = cv2.resize(data, (w, h), interpolation=cv2.INTER_CUBIC)
    if out.ndim == 2:
        out = out[..., None]
    return out.astype(F32, copy=False)
