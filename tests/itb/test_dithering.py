"""ImageToolbox "Dithering" group: parameter extremes + semantic checks against the Trickle/GPUImage/aire rules."""
import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.registry import all_ops, get_op
from tests.conftest import synthetic_image

SPECS = [s for s in all_ops(include_hidden=True) if s.group == "dithering"]

# Trickle ops that output pure 0/255 channels
TRICKLE = [s.id for s in SPECS if s.source.get("upstream") == "Trickle"]
DIFFUSION = ["floyd_steinberg", "false_floyd_steinberg", "jarvis_judice_ninke", "stucki", "burkes", "sierra",
             "two_row_sierra", "sierra_lite", "atkinson"]


def _run(op_id, img, seed=0, **params):
    spec = get_op(op_id)
    return spec.run(img, RenderContext(seed=seed), spec.validate(params))


def test_group_complete():
    assert len(SPECS) == 25
    assert len(TRICKLE) == 20


def _extremes(spec):
    lo, hi = {}, {}
    for p in spec.params:
        if p.kind in ("int", "float"):
            lo[p.name], hi[p.name] = p.min, p.max
        elif p.kind == "bool":
            lo[p.name], hi[p.name] = False, True
    return lo, hi


@pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])
@pytest.mark.parametrize("end", ["min", "max"])
def test_param_extremes(spec, end):
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    lo, hi = _extremes(spec)
    img = synthetic_image(64, 48, alpha=True)
    out = spec.run(img, RenderContext(seed=3), spec.validate(lo if end == "min" else hi))
    assert out.shape == img.shape and out.dtype == np.float32
    assert np.isfinite(out).all()


@pytest.mark.parametrize("op_id", TRICKLE)
@pytest.mark.parametrize("gray", [False, True])
def test_trickle_outputs_only_palette_colours(op_id, gray):
    img = synthetic_image(80, 60)
    out = _run(op_id, img, gray_scale=gray)
    assert set(np.unique(out)) <= {0.0, 1.0}
    if gray:
        assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
    else:
        assert len(np.unique(out.reshape(-1, 3), axis=0)) > 2  # colour mode dithers channels independently


@pytest.mark.parametrize("op_id", TRICKLE)
def test_trickle_gray_mode_reads_red_channel(op_id):
    img = np.zeros((40, 50, 3), np.float32)
    img[..., 1:] = 1.0  # cyan: R=0 -> Trickle's "gray" (= red byte) is black everywhere
    out = _run(op_id, img, gray_scale=True)
    if op_id == "filter.random_dithering":  # 0 < rand() % 256 fails only when the random threshold is 0
        assert out.mean() < 0.02
    else:
        assert out.max() == 0.0


def _ref_diffusion(u8, kernel, threshold):
    """Literal Python transcription of Trickle's per-channel loop (full frame, C++ truncating division)."""
    from lazyroom.ops.itb._dithering_impl import KERNELS

    taps, div = KERNELS[kernel]
    h, w = u8.shape
    err = np.zeros((h + 3, w + 4), np.int64)
    out = np.zeros((h, w), np.int64)
    for y in range(h):
        for x in range(w):
            v = u8[y, x] + err[y, x + 2]
            if v < threshold:
                e, out[y, x] = v, 0
            else:
                e, out[y, x] = v - 255, 255
            for dx, dy, n in taps:
                p = n * e
                err[y + dy, x + dx + 2] += abs(p) // div * (1 if p >= 0 else -1)
    return out


@pytest.mark.parametrize("kernel", DIFFUSION)
def test_error_diffusion_matches_reference(kernel):
    rng = np.random.default_rng(5)
    u8 = rng.integers(0, 256, (23, 31, 3))
    img = (u8 / 255.0).astype(np.float32)
    out = _run(f"filter.{kernel}_dithering", img, threshold=140)
    for c in range(3):
        ref = _ref_diffusion(u8[..., c], kernel, 140)
        assert np.array_equal(np.rint(out[..., c] * 255).astype(int), ref)


@pytest.mark.parametrize("kernel", ["floyd_steinberg", "jarvis_judice_ninke", "stucki", "burkes", "sierra"])
def test_error_diffusion_preserves_mean(kernel):
    img = np.full((96, 128, 3), 77 / 255, np.float32)
    out = _run(f"filter.{kernel}_dithering", img, threshold=128)
    assert abs(out.mean() - 77 / 255) < 0.03


def test_bayer_formula():
    from lazyroom.ops.itb._dithering_impl import BAYER

    rng = np.random.default_rng(1)
    u8 = rng.integers(0, 256, (17, 19, 3))
    out = _run("filter.bayer_four_dithering", (u8 / 255.0).astype(np.float32), threshold=150)
    m, div = BAYER[4]
    y, x = np.mgrid[0:17, 0:19]
    t = m[x % 4, y % 4][..., None]  # Trickle indexes matrix[x % f][y % f]
    ref = np.where(u8 + u8 * t // div < 150, 0, 255)
    assert np.array_equal(np.rint(out * 255).astype(int), ref)


def test_clustered_and_threshold_formula():
    from lazyroom.ops.itb._dithering_impl import CLUSTERED

    rng = np.random.default_rng(2)
    u8 = rng.integers(0, 256, (13, 21, 3))
    img = (u8 / 255.0).astype(np.float32)
    m, k = CLUSTERED[8]
    y, x = np.mgrid[0:13, 0:21]
    ref = np.where(u8 > (m[y % 8, x % 8] * k)[..., None], 255, 0)
    assert np.array_equal(np.rint(_run("filter.clustered_8x8_dithering", img) * 255).astype(int), ref)
    ref = np.where(u8 < 99, 0, 255)
    assert np.array_equal(np.rint(_run("filter.simple_threshold_dithering", img, threshold=99) * 255).astype(int), ref)


def test_left_to_right_rows_are_independent():
    img = synthetic_image(60, 40)
    out = _run("filter.left_to_right_dithering", img)
    top = _run("filter.left_to_right_dithering", img[:10])
    assert np.array_equal(out[:10], top)


def test_random_dithering_seeded():
    img = np.full((40, 40, 3), 0.5, np.float32)
    a = _run("filter.random_dithering", img, seed=1)
    b = _run("filter.random_dithering", img, seed=1)
    c = _run("filter.random_dithering", img, seed=2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert abs(a.mean() - 0.5) < 0.1
    # one threshold per pixel shared by the channels: a grey input stays grey
    assert np.array_equal(a[..., 0], a[..., 2])


def test_color_halftone():
    white = np.ones((40, 50, 3), np.float32)
    black = np.zeros((40, 50, 3), np.float32)
    assert _run("filter.color_halftone", white).min() == 1.0
    assert _run("filter.color_halftone", black).mean() < 0.15
    assert _run("filter.color_halftone", synthetic_image(50, 40), radius=0).min() == 1.0  # zero grid -> white
    mid = _run("filter.color_halftone", np.full((60, 60, 3), 0.5, np.float32), radius=4)
    assert 0.2 < mid.mean() < 0.9 and mid.std() > 0.2


def test_gpuimage_halftone_and_crosshatch_are_bw():
    img = synthetic_image(120, 90)
    for op_id in ("filter.halftone", "filter.crosshatch"):
        out = _run(op_id, img)
        assert set(np.unique(out)) <= {0.0, 1.0}
        assert np.array_equal(out[..., 0], out[..., 1])
    assert _run("filter.crosshatch", np.ones((30, 30, 3), np.float32)).min() == 1.0
    dark = _run("filter.crosshatch", np.zeros((200, 200, 3), np.float32)).mean()
    light = _run("filter.crosshatch", np.full((200, 200, 3), 0.9, np.float32)).mean()
    assert dark < light
    ht_dark = _run("filter.halftone", np.full((300, 300, 3), 0.1, np.float32), size=0.02).mean()
    ht_light = _run("filter.halftone", np.full((300, 300, 3), 0.9, np.float32), size=0.02).mean()
    assert ht_dark < ht_light and ht_light > 0.9


def test_quantizier_palette_size():
    img = synthetic_image(120, 90)
    for n in (2, 7, 32):
        out = _run("filter.quantizier", img, max_colors=n)
        assert len(np.unique(out.reshape(-1, 3), axis=0)) <= n
    flat = np.zeros((30, 30, 3), np.float32)
    flat[:, :15] = (1.0, 0.0, 0.0)
    flat[:, 15:] = (0.0, 0.0, 1.0)
    out = _run("filter.quantizier", flat, max_colors=16)
    assert np.abs(out - flat).max() < 2 / 255  # two colours survive (box means of 8-bit values)


def test_wu_quantizer_nearest_mapping_exact():
    from lazyroom.ops.itb._dithering_impl import remap_nearest

    rng = np.random.default_rng(4)
    img = rng.random((20, 25, 3)).astype(np.float32)
    pal = np.column_stack([rng.integers(0, 256, (37, 3)), np.full(37, 255)]).astype(np.int64)
    out = np.rint(remap_nearest(img, pal) * 255).astype(int).reshape(-1, 3)
    q = np.rint(img * 255).astype(int).reshape(-1, 3)
    wb = np.where(q[:, 0] < 128, 3, 2)
    d = 2 * (q[:, None, 0] - pal[None, :, 0]) ** 2 + 4 * (q[:, None, 1] - pal[None, :, 1]) ** 2 \
        + wb[:, None] * (q[:, None, 2] - pal[None, :, 2]) ** 2
    best = d.min(axis=1)
    got = 2 * (q[:, 0] - out[:, 0]) ** 2 + 4 * (q[:, 1] - out[:, 1]) ** 2 + wb * (q[:, 2] - out[:, 2]) ** 2
    assert np.array_equal(got, best)


@pytest.mark.gmic
def test_posterized_dithering_runs():
    spec = get_op("filter.posterized_dithering")
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(hint)
    img = synthetic_image(64, 48)
    a = spec.run(img, RenderContext(), spec.validate({}))
    b = spec.run(img, RenderContext(), spec.validate({"mixer_mode": "multiply", "quantize_colors": 4}))
    assert a.shape == img.shape and not np.allclose(a, b)
