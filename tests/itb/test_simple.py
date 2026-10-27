"""ImageToolbox "Simple effects" group: range sweeps + semantic checks.

Several tests compare the vectorised ports against small, literal
transliterations of the upstream loops (jhlabs Rust, Trickle C++, GPUImage
GLSL) on tiny images.
"""
import math

import cv2
import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.params import Choice, Float, Int
from lazyroom.core.registry import all_ops, get_op
from lazyroom.ops.itb import _simple_impl as S
from lazyroom.ops.itb.simple import _MATRICES
from tests.conftest import synthetic_image

SPECS = [s for s in all_ops(include_hidden=True) if s.group == "simple"]
IDS = {s.id for s in SPECS}


def run(op_id, img, seed=0, **params):
    spec = get_op(op_id)
    return spec.run(img, RenderContext(seed=seed), spec.validate(params))


def flat(v, w=24, h=20, alpha=False):
    img = np.full((h, w, 3), v, np.float32)
    if alpha:
        img = np.dstack([img, np.full((h, w), 0.5, np.float32)])
    return img


def test_group_complete():
    assert len(SPECS) == 66
    for s in SPECS:
        assert s.source["project"] == "ImageToolbox"


def _extremes(spec):
    out = [{}]
    for p in spec.params:
        if isinstance(p, (Float, Int)):
            out += [{p.name: p.min}, {p.name: p.max}]
        elif isinstance(p, Choice):
            out += [{p.name: c} for c in p.choices]
    return out


@pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])
@pytest.mark.parametrize("alpha", [False, True])
def test_param_extremes(spec, alpha):
    img = synthetic_image(48, 36, alpha=alpha)
    for vals in _extremes(spec):
        out = spec.run(img, RenderContext(seed=3), spec.validate(vals))
        assert out.dtype == np.float32 and np.isfinite(out).all()
        assert out.shape[2] == img.shape[2]
        if not spec.geometry:
            assert out.shape == img.shape
        assert out.min() >= 0 and out.max() <= 1


# ---------------------------------------------------------------------------
# Colour matrices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,matrix", [(n, m) for n, _, m, _ in _MATRICES])
def test_color_matrices(name, matrix):
    img = synthetic_image(32, 24, alpha=True)
    from lazyroom.ops.itb._common import snake

    out = run(f"filter.{snake(name)}", img)
    M = np.asarray(matrix, np.float32).reshape(3, 3)
    ref = np.clip(np.einsum("ij,hwj->hwi", M, img[..., :3]), 0, 1)
    assert np.allclose(out[..., :3], ref, atol=1e-5)
    assert np.array_equal(out[..., 3], img[..., 3])  # alpha untouched


def test_achromatopsia_is_gray():
    out = run("filter.achromatopsia", synthetic_image(32, 24))
    assert np.allclose(out[..., 0], out[..., 1]) and np.allclose(out[..., 1], out[..., 2])


# ---------------------------------------------------------------------------
# GPUImage
# ---------------------------------------------------------------------------

def test_black_and_white_and_negative():
    img = synthetic_image(32, 24, alpha=True)
    bw = run("filter.black_and_white", img)
    assert np.allclose(bw[..., 0], bw[..., 1]) and np.allclose(bw[..., 0], bw[..., 2])
    assert np.allclose(bw[..., 0], img[..., :3] @ np.array([0.2125, 0.7154, 0.0721], np.float32), atol=1e-6)
    neg = run("filter.negative", img)
    assert np.allclose(neg[..., :3], 1 - img[..., :3]) and np.array_equal(neg[..., 3], img[..., 3])


def test_cga_palette_and_opaque():
    out = run("filter.cga_color_space", synthetic_image(64, 48, alpha=True))
    pal = {(0, 0, 0), (255, 255, 255), (85, 255, 255), (255, 85, 255)}
    assert {tuple(c) for c in np.rint(out[..., :3] * 255).astype(int).reshape(-1, 3)} <= pal
    assert np.all(out[..., 3] == 1)


def test_gpu_laplacian_flat_is_half():
    out = run("filter.laplacian", flat(0.3))
    assert np.allclose(out, 0.5, atol=1e-6)


def _nms_ref(rgb):
    """Literal port of the GLSL (clamp-to-edge sampling)."""
    h, w = rgb.shape[:2]
    r = rgb[..., 0]
    g = lambda y, x: r[min(max(y, 0), h - 1), min(max(x, 0), w - 1)]  # noqa: E731
    out = np.zeros_like(rgb)
    step = lambda e, x: 1.0 if x >= e else 0.0  # noqa: E731
    for y in range(h):
        for x in range(w):
            c = r[y, x]
            m = 1.0 - step(c, g(y - 1, x))
            m = m * 1.0 - step(c, g(y - 1, x - 1))
            m = m * 1.0 - step(c, g(y, x - 1))
            m = m * 1.0 - step(c, g(y + 1, x - 1))
            mx = max(c, g(y + 1, x), g(y + 1, x + 1), g(y, x + 1), g(y - 1, x + 1))
            out[y, x] = rgb[y, x] * step(mx, c) * m
    return np.clip(out, 0, 1)


def test_non_maximum_suppression_matches_shader():
    rgb = np.random.default_rng(1).integers(0, 6, (12, 14, 3)).astype(np.float32) / 5
    out = run("filter.non_maximum_suppression", rgb)
    assert np.allclose(out, _nms_ref(rgb))
    assert np.all(run("filter.non_maximum_suppression", flat(0.4)) == 0)  # ties suppress everything


def test_weak_pixel_binary():
    out = run("filter.weak_pixel", synthetic_image(48, 36, alpha=True))
    assert set(np.unique(out[..., :3])) <= {0.0, 1.0}
    assert np.all(out[..., 3] == 1)
    assert np.all(run("filter.weak_pixel", flat(0.2))[..., :3] == 1)   # 9 * 0.2 >= 1.5
    assert np.all(run("filter.weak_pixel", flat(0.15))[..., :3] == 0)  # 9 * 0.15 < 1.5


# ---------------------------------------------------------------------------
# aire
# ---------------------------------------------------------------------------

def test_aire_edges():
    assert np.all(run("filter.laplacian_simple", flat(0.6)) == 0)
    assert np.all(run("filter.sobel_simple", flat(0.6)) == 0)
    step = np.zeros((10, 12, 3), np.float32)
    step[:, 6:] = 1.0
    sob = run("filter.sobel_simple", step)
    assert sob[5, 5, 0] == 1.0 and sob[5, 6, 0] == 1.0 and sob[5, 2, 0] == 0
    sob_rev = run("filter.sobel_simple", 1 - step)  # bright -> dark: negative gradient saturates to 0
    assert np.all(sob_rev == 0)
    out = run("filter.laplacian_simple", synthetic_image(32, 24, alpha=True))
    assert np.allclose(out[..., 0], out[..., 2]) and np.all(out[..., 3] == 1)


def test_equalize_histogram_stretches_luma_keeps_gray():
    g = np.linspace(0.4, 0.6, 40, dtype=np.float32)
    img = np.repeat(np.repeat(g[None, :, None], 10, 0), 3, 2)
    out = run("filter.equalize_histogram", img)
    assert out.min() < 0.05 and out.max() > 0.95
    assert np.allclose(out[..., 0], out[..., 1], atol=1e-6) and np.allclose(out[..., 1], out[..., 2], atol=1e-6)
    assert np.all(np.diff(out[0, :, 0]) >= 0)


# ---------------------------------------------------------------------------
# jhlabs
# ---------------------------------------------------------------------------

def _despeckle_ref(q):
    """Literal port of despeckle_exact (8-bit ints)."""
    h, w = q.shape[:2]
    out = q.copy()

    def adj(c, a, b):
        if c < a:
            c += 1
        if c < b:
            c += 1
        if c > a:
            c -= 1
        if c > b:
            c -= 1
        return c

    for y in range(h):
        for x in range(w):
            for ch in range(3):
                c = int(q[y, x, ch])
                iy, ix = 0 < y < h - 1, 0 < x < w - 1
                if iy:
                    c = adj(c, q[y - 1, x, ch], q[y + 1, x, ch])
                if ix:
                    c = adj(c, q[y, x - 1, ch], q[y, x + 1, ch])
                if ix and iy:
                    c = adj(c, q[y - 1, x - 1, ch], q[y + 1, x + 1, ch])
                    c = adj(c, q[y + 1, x - 1, ch], q[y - 1, x + 1, ch])
                out[y, x, ch] = c
    return out


def test_despeckle_matches_reference():
    q = np.random.default_rng(2).integers(0, 256, (11, 13, 3))
    out = run("filter.despeckle", q.astype(np.float32) / 255)
    assert np.array_equal(np.rint(out * 255).astype(int), _despeckle_ref(q))


def _reduce_noise_ref(q):
    h, w = q.shape[:2]
    out = np.empty_like(q)
    for y in range(h):
        for x in range(w):
            for ch in range(3):
                vals = []
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        vals.append(q[yy, xx, ch] if 0 <= yy < h and 0 <= xx < w else q[y, x, ch])
                nb = vals[:4] + vals[5:]
                out[y, x, ch] = min(max(vals[4], min(nb)), max(nb))
    return out


def test_reduce_noise_matches_reference():
    q = np.random.default_rng(3).integers(0, 256, (10, 12, 3))
    out = run("filter.reduce_noise", q.astype(np.float32) / 255)
    assert np.array_equal(np.rint(out * 255).astype(int), _reduce_noise_ref(q))


def test_equalize_per_channel():
    q = np.random.default_rng(4).integers(60, 120, (16, 16, 3))
    out = run("filter.equalize", q.astype(np.float32) / 255)
    for ch in range(3):
        cdf = np.cumsum(np.bincount(q[..., ch].ravel(), minlength=256))
        lut = np.floor(cdf * 255 / q[..., ch].size + 0.5)
        assert np.array_equal(np.rint(out[..., ch] * 255), lut[q[..., ch]])
    assert out.max() == 1.0


def test_simple_solarize():
    img = np.array([[[0.0, 0.5, 1.0]]], np.float32)
    assert np.allclose(run("filter.simple_solarize", img), [[[1.0, 0.0, 1.0]]])


def test_java_look_and_feel_checkerboard():
    img = synthetic_image(10, 8)
    out = run("filter.java_look_and_feel", img)
    assert np.allclose(out[0, 0], img[0, 0]) and np.allclose(out[1, 1], img[1, 1])
    assert np.allclose(out[0, 1], 0.75 * 0.6 + 0.25 * img[0, 1], atol=1e-6)
    rgba = run("filter.java_look_and_feel", synthetic_image(10, 8, alpha=True))
    assert rgba[0, 1, 3] >= 0.75  # alpha mixed with the opaque grey (premultiplied mix)


# ---------------------------------------------------------------------------
# Trickle
# ---------------------------------------------------------------------------

def test_hdr_flat():
    out = run("filter.hdr", flat(0.3))
    assert np.allclose(out, 2 * 0.3 * 0.3, atol=1e-5)  # overlay of a flat image with itself; no edges


def test_gotham_blue_and_opaque():
    out = run("filter.gotham", synthetic_image(48, 36, alpha=True))
    assert np.all(out[..., 3] == 1)
    assert out[..., 2].mean() > out[..., 0].mean()


def _trickle_pixels(q):
    """Android RGBA bytes as Trickle sees them: Color.R() is the blue byte, Color.B() the red byte."""
    return q[..., 2].astype(np.int64), q[..., 1].astype(np.int64), q[..., 0].astype(np.int64)


def _gotham_ref(q):
    """Literal (integer) port of GothamFilter::procImage on byte-swapped pixels."""
    R, G, B = _trickle_pixels(q)
    R, G, B = (np.clip(c - 50, 0, 255) for c in (R, G, B))
    table = np.array([int((i / 255.0) ** (1 / 0.7) * 255.0) for i in range(256)])
    R, G, B = table[R], table[G], table[B]
    i = (R / 255.0 + G / 255.0 + B / 255.0) / 3.0
    h = 233.69 - 120
    r = i * (1 - 0.2)
    g = i * (1 + (0.2 * np.cos(h * 3.14159265 / 180.0)) / np.cos((60 - h) * 3.14159265 / 180.0))
    b = 3 * i - (r + g)
    ri, gi, bi = (r * 255).astype(np.int64), (g * 255).astype(np.int64), (b * 255).astype(np.int64)
    ok = (bi <= 255) & (ri >= 0) & (gi >= 0) & (bi >= 0) & (ri <= 255) & (gi <= 255)
    R, G, B = np.where(ok, bi, R), np.where(ok, gi, G), np.where(ok, ri, B)  # RGB2Color(rgb.b, rgb.g, rgb.r)
    R, G, B = (np.clip((128 + (c - 128) * 1.2).astype(np.int64), 0, 255) for c in (R, G, B))
    return np.stack([B, G, R], axis=-1)  # code-space B is the red byte


def test_gotham_matches_trickle_reference():
    q = np.rint(synthetic_image(40, 30) * 255).astype(np.uint8)
    out = np.rint(run("filter.gotham", q.astype(np.float32) / 255) * 255)
    diff = np.abs(out - _gotham_ref(q))
    assert np.mean(diff <= 3) > 0.99  # float vs truncating int arithmetic


def _hdr_ref(q):
    """Literal port of HDRFilter (GaussianBlurFilter 0.6 interior, overlay, SharpenFilter)."""
    o = q.astype(np.float64)
    h, w = o.shape[:2]
    k = np.exp(-0.5 / 0.36 * np.add.outer(np.arange(-1, 2) ** 2, np.arange(-1, 2) ** 2))
    k /= k.sum()
    s = o.copy()
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            s[y, x] = np.trunc((o[y - 1:y + 2, x - 1:x + 2] * k[..., None]).sum((0, 1)))
    on, sn = o / 255, s / 255
    over = np.trunc(np.where(sn <= 0.5, 2 * sn * on, 1 - 2 * (1 - on) * (1 - sn)) * 255)
    edge = np.zeros_like(over)
    lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            edge[y, x] = np.clip((over[y - 1:y + 2, x - 1:x + 2] * lap[..., None]).sum((0, 1)), 0, 255)
    return np.clip(over + edge, 0, 255)


def test_hdr_matches_trickle_reference():
    q = np.rint(synthetic_image(30, 24) * 255).astype(np.uint8)
    out = np.rint(run("filter.hdr", q.astype(np.float32) / 255) * 255)
    diff = np.abs(out - _hdr_ref(q))
    assert np.mean(diff <= 3) > 0.98  # float vs truncating int arithmetic


def _tv_ref(q):
    """Literal port of TvFilter on Android RGBA bytes (Trickle reads them as 0xAABBGGRR ints)."""
    h, w = q.shape[:2]
    out = q.copy()
    for x in range(w):
        for y in range(0, h, 4):
            r = g = 0
            for k in range(4):
                if y + k < h:
                    r += int(q[y + k, x, 2]) // 4   # Color.R() == blue byte
                    g += int(q[y + k, x, 1]) // 4
            for k in range(3):
                if y + k < h:
                    out[y + k, x] = [(0, 0, r), (0, g, 0), (0, 0, 0)][k]
    return out


def test_simple_old_tv_matches_reference():
    q = np.random.default_rng(5).integers(0, 256, (11, 7, 3))
    out = run("filter.simple_old_tv", q.astype(np.float32) / 255)
    assert np.array_equal(np.rint(out * 255).astype(int), _tv_ref(q))
    rgba = run("filter.simple_old_tv", synthetic_image(8, 9, alpha=True))
    assert np.all(rgba[0::4, :, 3] == 1) and np.all(rgba[2::4, :, :3] == 0)


def test_simple_sketch():
    out = run("filter.simple_sketch", flat(0.5, alpha=True))
    assert np.all(out[1:-1, 1:-1, :3] == 1) and np.all(out[..., 3] == 1)
    img = synthetic_image(40, 30)
    out = run("filter.simple_sketch", img)
    assert set(np.unique(out[1:-1, 1:-1])) <= {0.0, 1.0}
    q = np.rint(img * 255).astype(np.float64)
    gray = np.trunc(((q[..., 2] * 0.3 + q[..., 1] * 0.59) + q[..., 0] * 0.11).astype(np.float32))
    assert np.allclose(out[0, :, 0] * 255, gray[0], atol=1e-3)  # border keeps the grey value


# ---------------------------------------------------------------------------
# OpenCV based / chains
# ---------------------------------------------------------------------------

def test_old_tv_seeded_and_green():
    img = synthetic_image(40, 30)
    a, b = run("filter.old_tv", img, seed=1), run("filter.old_tv", img, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, run("filter.old_tv", img, seed=2))
    assert np.all(a[..., 1] >= a[..., 0] - 1e-6) and np.all(a[..., 0] >= a[..., 2] - 1e-6)


@pytest.mark.parametrize("cmap", ["autumn", "jet", "viridis", "deep_green"])
def test_color_map_matches_opencv(cmap):
    img = synthetic_image(32, 24)
    out = run("filter.color_map", img, color_map=cmap)
    g = cv2.cvtColor(np.rint(img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    ref = cv2.applyColorMap(g, getattr(cv2, f"COLORMAP_{cmap.upper().replace('_', '')}"))[..., ::-1] / 255.0
    assert np.allclose(out, ref, atol=1e-6)


def test_average_distance_and_luminance_gradient():
    assert np.all(run("filter.average_distance", flat(0.4)) == 0)
    ad = run("filter.average_distance", synthetic_image(32, 24))
    assert ad.max() == pytest.approx(1.0)
    lg = run("filter.luminance_gradient", flat(0.4, alpha=True))
    assert np.allclose(lg[..., :3], [0.0, 0.5, 0.0]) and np.all(lg[..., 3] == 1)


def test_moire_removes_periodic_stripes():
    x = np.arange(128, dtype=np.float32)
    img = np.full((128, 128, 3), 0.5, np.float32) + 0.3 * np.sin(2 * np.pi * x / 4)[None, :, None]
    out = run("filter.moire", img)
    amp = lambda a: np.abs(np.fft.fft(a[64, :, 0] - a[64, :, 0].mean()))[32]  # noqa: E731
    assert amp(out) < 0.05 * amp(img)


def test_auto_perspective_rectifies_quad():
    h, w = 300, 400
    img = np.full((h, w, 3), 0.15, np.float32)
    quad = np.array([[90, 60], [330, 40], [350, 250], [70, 270]], np.int32)
    cv2.fillConvexPoly(img, quad, (0.95, 0.95, 0.95))
    out = run("filter.auto_perspective", img)
    assert out.shape[:2] != img.shape[:2]
    assert abs(out.shape[1] - 260) < 25 and abs(out.shape[0] - 215) < 25
    assert out[..., :3].mean() > 0.85  # essentially only the bright document remains
    # nothing found -> unchanged (needs >= ~90 px: below that upstream's 1.2% border margin is < 1 px
    # and the full-frame contour itself qualifies as a quad)
    plain = flat(0.5, 200, 150)
    assert np.array_equal(run("filter.auto_perspective", plain), plain)


def test_opaque_outputs_use_premultiplied_colour():
    img = synthetic_image(24, 20, alpha=True)
    img[..., 3] = 0.0
    out = run("filter.color_map", img, color_map="jet")
    jet0 = cv2.applyColorMap(np.zeros((1, 1), np.uint8), cv2.COLORMAP_JET)[0, 0, ::-1] / 255.0
    assert np.allclose(out[..., :3], jet0) and np.all(out[..., 3] == 1)
    assert S.bitmap_rgb(img).max() == 0
    assert math.isclose(float(run("filter.negative", img)[..., 3].max()), 0.0)
