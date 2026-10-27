"""Tests for the ImageToolbox "Blur" group port."""
import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.params import Float, Int
from lazyroom.core.registry import all_ops, get_op
from lazyroom.ops.itb import _blur_impl as B
from tests.conftest import synthetic_image

SPECS = [s for s in all_ops(include_hidden=True) if s.group == "blur"]
IDS = [s.id for s in SPECS]


def _run(op_id, img, seed=0, **params):
    spec = get_op(op_id)
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    return spec.run(img, RenderContext(seed=seed), spec.validate(params))


def _hf_energy(img):
    rgb = img[..., :3]
    return float(np.mean(np.abs(np.diff(rgb, axis=0))) + np.mean(np.abs(np.diff(rgb, axis=1))))


def test_group_is_complete():
    assert len(SPECS) == 36


# ---------------------------------------------------------------------------
# Parameter extremes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("extreme", ["min", "max"])
@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_param_extremes(spec, extreme):
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    vals = {}
    for p in spec.params:
        if isinstance(p, (Int, Float)):
            vals[p.name] = p.min if extreme == "min" else p.max
    for alpha in (False, True):
        img = synthetic_image(44, 33, alpha=alpha)
        out = spec.run(img, RenderContext(seed=3), spec.validate(vals))
        if spec.id == "filter.shuffle_blur":   # unmatched pixels become transparent, like upstream
            assert out.shape[:2] == img.shape[:2] and out.shape[2] in (3, 4)
        else:
            assert out.shape == img.shape
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
        assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("spec", [s for s in SPECS if any(p.name in ("edge_mode", "transfer_function")
                                                           for p in s.params)], ids=lambda s: s.id)
def test_every_choice_runs(spec):
    img = synthetic_image(40, 30)
    for p in spec.params:
        if p.name in ("edge_mode", "transfer_function"):
            for c in p.choices:
                out = spec.run(img, RenderContext(), spec.validate({p.name: c}))
                assert out.shape == img.shape and np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Semantics
# ---------------------------------------------------------------------------

SMOOTHING = ["filter.box_blur", "filter.tent_blur", "filter.gaussian_box_blur", "filter.native_stack_blur",
             "filter.linear_box_blur", "filter.linear_tent_blur", "filter.linear_gaussian_box_blur",
             "filter.linear_stack_blur", "filter.linear_fast_gaussian_blur", "filter.linear_fast_gaussian_blur_next",
             "filter.linear_gaussian_blur", "filter.gaussian_blur", "filter.fast_gaussian_blur_2d",
             "filter.fast_gaussian_blur_3d", "filter.fast_gaussian_blur_4d", "filter.bilaterial_blur",
             "filter.fast_bilaterial_blur", "filter.median_blur", "filter.motion_blur", "filter.poisson_blur",
             "filter.circle_blur", "filter.cross_blur", "filter.ring_blur", "filter.star_blur",
             "filter.enhanced_zoom_blur", "filter.zoom_blur", "filter.radial_tilt_shift", "filter.linear_tilt_shift",
             "filter.adaptive_blur", "filter.lens_blur", "filter.fast_blur", "filter.stack_blur"]


@pytest.mark.parametrize("op_id", SMOOTHING)
def test_blur_reduces_high_frequency_energy(op_id):
    img = synthetic_image(80, 60)
    noisy = np.clip(img + np.random.default_rng(1).normal(0, 0.15, img.shape).astype(np.float32), 0, 1)
    out = _run(op_id, noisy)
    assert _hf_energy(out) < 0.8 * _hf_energy(noisy)


NORMALISED = ["filter.box_blur", "filter.tent_blur", "filter.gaussian_box_blur", "filter.native_stack_blur",
              "filter.linear_box_blur", "filter.linear_gaussian_blur", "filter.gaussian_blur",
              "filter.fast_gaussian_blur_2d", "filter.fast_gaussian_blur_3d", "filter.fast_gaussian_blur_4d",
              "filter.bilaterial_blur", "filter.fast_bilaterial_blur", "filter.median_blur", "filter.motion_blur",
              "filter.poisson_blur", "filter.circle_blur", "filter.cross_blur", "filter.ring_blur",
              "filter.star_blur", "filter.enhanced_zoom_blur", "filter.zoom_blur", "filter.radial_tilt_shift",
              "filter.linear_tilt_shift", "filter.adaptive_blur", "filter.lens_blur", "filter.diffuse",
              "filter.fast_blur", "filter.stack_blur", "filter.bokeh", "filter.linear_stack_blur"]


@pytest.mark.parametrize("op_id", NORMALISED)
def test_constant_image_is_preserved(op_id):
    img = np.full((30, 40, 3), (0.2, 0.55, 0.8), np.float32)
    out = _run(op_id, img)
    assert np.allclose(out, img, atol=2.5e-3)


@pytest.mark.parametrize("op_id,params", [
    ("filter.box_blur", {"radius": 0}),
    ("filter.gaussian_blur", {"radius": 0}),
    ("filter.fast_blur", {"radius": 0}),
    ("filter.stack_blur", {"radius": 0}),
    ("filter.lens_blur", {"radius": 0}),
    ("filter.diffuse", {"scale": 0}),
    ("filter.zoom_blur", {"blur_size": 0}),
    ("filter.enhanced_zoom_blur", {"strength": 0}),
    ("filter.fast_gaussian_blur_2d", {"radius": 1}),
    ("filter.fast_gaussian_blur_3d", {"radius": 1}),
    ("filter.fast_gaussian_blur_4d", {"radius": 1}),
    ("filter.shuffle_blur", {"radius": 0}),
])
def test_identity_settings(op_id, params):
    img = synthetic_image(50, 40)
    assert np.allclose(_run(op_id, img, **params), img, atol=1e-5)


def test_opaque_rgba_matches_rgb():
    rgb = synthetic_image(50, 40)
    rgba = np.dstack([rgb, np.ones(rgb.shape[:2], np.float32)])
    for op_id in ("filter.gaussian_blur", "filter.fast_gaussian_blur_2d", "filter.bokeh", "filter.adaptive_blur",
                  "filter.linear_box_blur", "filter.stack_blur"):
        a = _run(op_id, rgb)
        b = _run(op_id, rgba)
        assert np.allclose(a, b[..., :3], atol=1e-5), op_id
        assert np.allclose(b[..., 3], 1.0, atol=1e-5), op_id


def test_alpha_is_blurred_premultiplied():
    img = np.zeros((40, 40, 4), np.float32)
    img[:, :20] = (1.0, 0.0, 0.0, 1.0)      # opaque red on the left, transparent (black) on the right
    out = _run("filter.box_blur", img, radius=5)
    mid = out[20, 18:23]
    assert 0.05 < mid[2, 3] < 0.95            # alpha blurred across the edge
    assert np.allclose(mid[mid[:, 3] > 0.05, 0], 1.0, atol=1e-4)   # no dark fringe in the colour


def test_bokeh_dilates():
    img = synthetic_image(60, 45)
    out = _run("filter.bokeh", img, size=9, amount=6)
    assert (out >= img - 1e-6).all()
    assert out.mean() > img.mean()


def test_tilt_shift_keeps_focus_and_blurs_far_field():
    img = np.clip(synthetic_image(80, 80) + np.random.default_rng(2).normal(0, 0.2, (80, 80, 3)), 0, 1)
    img = img.astype(np.float32)
    out = _run("filter.radial_tilt_shift", img, radius=0.3)
    assert np.allclose(out[40, 40], img[40, 40], atol=1e-6)
    assert _hf_energy(out[:15, :15]) < 0.7 * _hf_energy(img[:15, :15])
    out = _run("filter.linear_tilt_shift", img, size=0.2, angle=0)
    assert np.allclose(out[:, 40], img[:, 40], atol=1e-6)        # sharp vertical band through the centre


def test_shuffle_blur_copies_existing_pixels_and_threshold():
    img = synthetic_image(40, 30)
    out = _run("filter.shuffle_blur", img, radius=70, threshold=1.0)
    cols = {tuple(c) for c in np.round(img.reshape(-1, 3), 5)}
    assert all(tuple(c) in cols for c in np.round(out[..., :3].reshape(-1, 3), 5))
    rgba = np.dstack([img, np.ones(img.shape[:2], np.float32)])
    out = _run("filter.shuffle_blur", rgba, radius=10, threshold=-1.0)   # luma > 1 never true
    assert np.allclose(out[..., 3], 0.0)


def test_random_ops_are_seeded():
    img = synthetic_image(40, 30)
    for op_id in ("filter.poisson_blur", "filter.shuffle_blur", "filter.frosted_glass"):
        a = _run(op_id, img, seed=1)
        b = _run(op_id, img, seed=1)
        c = _run(op_id, img, seed=2)
        assert np.allclose(a, b), op_id
        assert not np.allclose(a, c), op_id


def test_edge_mode_only_changes_borders():
    img = synthetic_image(60, 50)
    a = _run("filter.gaussian_blur", img, radius=4, sigma=3, edge_mode="clamp")
    b = _run("filter.gaussian_blur", img, radius=4, sigma=3, edge_mode="wrap")
    assert np.allclose(a[6:-6, 6:-6], b[6:-6, 6:-6], atol=1e-6)
    assert not np.allclose(a[:3], b[:3])


def test_linear_light_blur_differs_from_perceptual():
    img = np.zeros((30, 30, 3), np.float32)
    img[:, ::2] = 1.0
    lin = _run("filter.linear_box_blur", img, radius=3)
    per = _run("filter.box_blur", img, radius=3)
    # averaging black/white in linear light is brighter once re-encoded
    assert lin[15, 15, 0] > per[15, 15, 0] + 0.1


# ---------------------------------------------------------------------------
# Kernel-level checks against literal re-implementations of upstream loops
# ---------------------------------------------------------------------------

def test_libblur_box_window():
    rng = np.random.default_rng(0)
    a = rng.random((17, 23, 3)).astype(np.float32)
    k = 7
    r = k // 2
    tmp = np.zeros_like(a)
    ref = np.zeros_like(a)
    for x in range(a.shape[1]):
        tmp[:, x] = a[:, np.clip(np.arange(x - r + 1, x + r + 1), 0, a.shape[1] - 1)].mean(axis=1)
    for y in range(a.shape[0]):
        ref[y] = tmp[np.clip(np.arange(y - r + 1, y + r + 1), 0, a.shape[0] - 1)].mean(axis=0)
    assert np.allclose(B.lb_box(a, k, k), ref, atol=1e-6)


def test_stack_blur_is_triangle():
    rng = np.random.default_rng(1)
    a = rng.random((21, 19, 3)).astype(np.float32)
    r = 6
    w = (r + 1 - np.abs(np.arange(-r, r + 1))) / (r + 1) ** 2
    tmp = np.zeros_like(a)
    ref = np.zeros_like(a)
    for x in range(a.shape[1]):
        tmp[:, x] = (a[:, np.clip(np.arange(x - r, x + r + 1), 0, a.shape[1] - 1)] * w[None, :, None]).sum(1)
    for y in range(a.shape[0]):
        ref[y] = (tmp[np.clip(np.arange(y - r, y + r + 1), 0, a.shape[0] - 1)] * w[:, None, None]).sum(0)
    assert np.allclose(B.triangle_blur(a, r), ref, atol=1e-6)


def _literal_fg2(x, r, mode):
    n = len(x)
    x = list(map(float, x))
    buf = [0.0] * 1024
    dif = s = 0.0
    for y in range(-2 * r, n):
        if y >= 0:
            x[y] = s / (r * r)
            dif += buf[(y - r) & 1023] - 2 * buf[y & 1023]
        elif y + r >= 0:
            dif -= 2 * buf[y & 1023]
        i = y + r
        if not 0 <= i < n:
            i = {"clamp": min(max(i, 0), n - 1), "wrap": i % n, "reflect": n - (i % n) - 1,
                 "reflect_101": (n - 1) - (i % (n - 1))}[mode]
        v = x[i]
        dif += v
        s += dif
        buf[(y + r) & 1023] = v
    return np.array(x)


@pytest.mark.parametrize("mode", B.EDGE_MODES)
def test_fast_gaussian_matches_literal_loop(mode):
    rng = np.random.default_rng(2)
    a = rng.random((19, 26, 1)).astype(np.float32)
    ref = a[..., 0].astype(np.float64).copy()
    for x in range(ref.shape[1]):
        ref[:, x] = _literal_fg2(ref[:, x], 5, mode)
    for y in range(ref.shape[0]):
        ref[y] = _literal_fg2(ref[y], 5, mode)
    assert np.allclose(B.fast_gaussian(a, 5, mode, 2)[..., 0], ref, atol=1e-6)


def test_fast_gaussian_kernels_are_box_powers():
    for degree, fn in ((2, lambda d, r: B.fast_gaussian(d, r, "clamp", 2)),
                       (3, lambda d, r: B.fast_gaussian(d, r, "clamp", 3)),
                       (4, B.fast_gaussian_4d)):
        d = np.zeros((1, 61, 1), np.float32)
        d[0, 30] = 1.0
        out = np.ascontiguousarray(np.repeat(d, 1, axis=0))
        resp = fn(np.ascontiguousarray(np.tile(out, (3, 1, 1))), 4)[1, :, 0]
        box = np.ones(4) / 4
        ker = box
        for _ in range(degree - 1):
            ker = np.convolve(ker, box)
        assert np.isclose(resp.sum(), 1.0, atol=1e-6)
        nz = resp[resp > 1e-9]
        assert np.allclose(np.sort(nz), np.sort(ker), atol=1e-6), degree


def test_bilateral_series_matches_brute_force():
    rng = np.random.default_rng(3)
    a = rng.random((16, 18, 3)).astype(np.float32)
    ks, ss, rs = 7, 2.0, 1.0
    p = ks // 2
    pa = B.pad(a, p, p, p, p, "reflect_101").astype(np.float64)
    num = np.zeros(a.shape)
    den = np.zeros(a.shape)
    for dy in range(ks):
        for dx in range(ks):
            ws = np.exp(-((p - dx) ** 2 + (p - dy) ** 2) / (2 * ss * ss))
            nb = pa[dy:dy + a.shape[0], dx:dx + a.shape[1]]
            wr = ws * np.exp(-((a - nb) ** 2) / (2 * rs * rs))
            num += wr * nb
            den += wr
    assert np.allclose(B.bilateral(a, ks, ss, rs, "reflect_101"), num / den, atol=1e-5)


def test_dilate_matches_opencv():
    import cv2

    rng = np.random.default_rng(4)
    a = rng.random((25, 31, 3)).astype(np.float32)
    for size, sides in ((9, 5), (12, 6), (15, 3)):
        k = B.bokeh_kernel(size, sides)
        ref = cv2.dilate(a, k.astype(np.uint8), borderType=cv2.BORDER_REFLECT_101)
        assert np.allclose(B.dilate(a, k), ref), (size, sides)


@pytest.mark.parametrize("mode", B.EDGE_MODES)
def test_correlate2d_paths_match_brute_force(mode):
    rng = np.random.default_rng(5)
    a = rng.random((30, 41, 3)).astype(np.float32)
    for k in (B.kernel_circle(15), B.kernel_ring(9), B.motion_kernel(11, 30.0), B.kernel_circle(45)):
        kh, kw = k.shape
        p = B.pad(a, kh // 2, kh - 1 - kh // 2, kw // 2, kw - 1 - kw // 2, mode).astype(np.float64)
        ref = sum(k[j, i] * p[j:j + a.shape[0], i:i + a.shape[1]]
                  for j in range(kh) for i in range(kw) if k[j, i])
        assert np.allclose(B.correlate2d(a, k, mode), ref, atol=1e-5)


def test_motion_kernel():
    k = B.motion_kernel(9, 0.0)
    assert np.isclose(k.sum(), 1.0)
    assert np.count_nonzero(k[4]) == np.count_nonzero(k)   # horizontal line through the centre row
    assert np.isclose(B.motion_kernel(25, 45.0).sum(), 1.0)


def test_shaped_kernels():
    assert B.kernel_circle(3) is None              # aire rejects kernels < 5
    for fn in (B.kernel_circle, B.kernel_cross, B.kernel_ring, B.kernel_star):
        k = fn(15)
        assert k.shape == (15, 15) and np.isclose(k.sum(), 1.0)
    assert np.count_nonzero(B.kernel_star(9)) == 17


def test_rounding_helpers():
    assert B.nearest_odd(10) == 11 and B.nearest_odd(11) == 11 and B.nearest_odd(0) == 1
    assert B.kround(2.5) == 3 and B.kround(2.49) == 2
    assert B.create_box_gauss(1.0, 3) == [1, 1, 1]
