"""ImageToolbox "Pixelation" group: parameter extremes + semantic checks."""
import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.registry import all_ops, get_op
from tests.conftest import synthetic_image

SPECS = [s for s in all_ops(include_hidden=True) if s.group == "pixelation"]

# Extremes whose output size explodes (Ministeck: resolution x piece size px on the long side)
# or that take minutes (Shapeism: 16 scales at precision 10) are tamed so the test stays fast;
# every other numeric parameter is pinned to its bound.
TAME = {"filter.ministeck": {"max": {"resolution": 32, "piece_size": 4}},
        "filter.shapeism": {"max": {"scale": 3, "precision": 3}}}


def _run(op_id, img, _seed=7, _scale=1.0, **params):
    spec = get_op(op_id)
    return spec.run(img, RenderContext(seed=_seed, scale=_scale), spec.validate(params))


def test_group_size():
    assert len(SPECS) == 47


def _extreme(spec, which):
    vals = {}
    for p in spec.params:
        if p.kind in ("int", "float") and p.min is not None and p.max is not None:
            vals[p.name] = p.min if which == "min" else p.max
    vals.update(TAME.get(spec.id, {}).get(which, {}))
    return vals


@pytest.mark.parametrize("which", ["min", "max"])
@pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])
def test_param_extremes(spec, which):
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    img = synthetic_image(64, 48)
    out = spec.run(img, RenderContext(seed=3), spec.validate(_extreme(spec, which)))
    assert out.dtype == np.float32 and out.ndim == 3 and out.shape[2] in (3, 4)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    if not spec.geometry:
        assert out.shape[:2] == img.shape[:2]


# ---------------------------------------------------------------------------
# PixelationTool family
# ---------------------------------------------------------------------------

TOOL_IDS = ["filter.pixelation", "filter.enhanced_pixelation", "filter.circle_pixelation",
            "filter.enhanced_circle_pixelation", "filter.diamond_pixelation", "filter.enhanced_diamond_pixelation",
            "filter.simple_weave_pixelation", "filter.cross_pixelization", "filter.micro_macro_pixelization",
            "filter.nucleus_pixelization", "filter.orbital_pixelization", "filter.pulse_grid_pixelization",
            "filter.radial_weave_pixelization", "filter.staggered_pixelization", "filter.vortex_pixelization"]


def test_pixelation_constant_blocks_and_cell_centre_colour():
    img = synthetic_image(160, 120)
    out = _run("filter.pixelation", img, pixel_size=25)
    assert out.shape == img.shape  # squares overlap -> fully opaque
    # square(25): resolution 21, size 25 -> pixel (50, 50) lies only in the shape (row 3, col 3),
    # centred at 2.5 * 21 = 52.5, whose colour is the source pixel (52, 52)
    assert np.allclose(out[50, 50], img[52, 52], atol=1e-5)
    assert np.allclose(out[44:60, 44:60], img[52, 52], atol=1e-5)
    same_right = np.all(np.abs(out[:, 1:] - out[:, :-1]) < 1e-6, axis=2).mean()
    assert same_right > 0.85


@pytest.mark.parametrize("op_id", TOOL_IDS + ["filter.stroke_pixelation"])
def test_tool_uniform_image_stays_uniform(op_id):
    img = np.full((60, 80, 3), (0.2, 0.5, 0.7), np.float32)
    out = _run(op_id, img)
    covered = out[..., 3] > 0.01 if out.shape[2] == 4 else np.ones(out.shape[:2], bool)
    if op_id == "filter.stroke_pixelation":
        return  # dots over a black background
    assert np.allclose(out[covered][:, :3], (0.2, 0.5, 0.7), atol=1e-4)


def test_circle_pixelation_has_transparent_gaps():
    out = _run("filter.circle_pixelation", synthetic_image(96, 72), pixel_size=24)
    assert out.shape[2] == 4
    assert (out[..., 3] < 0.01).any() and (out[..., 3] > 0.99).any()


def test_stroke_pixelation_background():
    img = synthetic_image(96, 72)
    out = _run("filter.stroke_pixelation", img, pixel_size=20, background_color="#FF0000")
    assert out.shape[2] == 3  # opaque background -> no alpha
    red = np.all(np.abs(out - (1.0, 0.0, 0.0)) < 1e-4, axis=2)
    assert 0.2 < red.mean() < 0.95


def test_pixel_size_scales_with_preview():
    img = synthetic_image(160, 120)
    full = _run("filter.pixelation", img, pixel_size=40)
    import cv2

    half = cv2.resize(img, (80, 60), interpolation=cv2.INTER_AREA)
    prev = _run("filter.pixelation", half, _scale=0.5, pixel_size=40)
    # same number of blocks: compare block counts along a row
    def edges(row):
        return int((np.abs(np.diff(row, axis=0)).sum(axis=1) > 1e-4).sum())
    assert abs(edges(full[60]) - edges(prev[30])) <= 1


# ---------------------------------------------------------------------------
# Native filters
# ---------------------------------------------------------------------------

def _quadrants(h=96, w=128):
    img = np.zeros((h, w, 3), np.float32)
    img[: h // 2, : w // 2] = (1, 0, 0)
    img[: h // 2, w // 2:] = (0, 1, 0)
    img[h // 2:, : w // 2] = (0, 0, 1)
    img[h // 2:, w // 2:] = (1, 1, 0)
    return img


def _colour_set(a):
    return {tuple(np.round(c, 4)) for c in a.reshape(-1, a.shape[2])}


def test_crystallize_uses_source_colours_and_rng():
    img = _quadrants()
    out = _run("filter.crystallize", img, amount=0.3)
    assert out.shape == img.shape
    assert _colour_set(out) <= _colour_set(img)
    a = _run("filter.crystallize", synthetic_image(96, 72), _seed=1)
    b = _run("filter.crystallize", synthetic_image(96, 72), _seed=2)
    assert not np.allclose(a, b)


def test_crystallize_stroke():
    out = _run("filter.crystallize", _quadrants(), amount=0.3, stroke_color="#FFFFFF")
    white = np.all(out > 0.999, axis=2)
    assert white.mean() > 0.05


def test_voronoi_crystallize_without_border_uses_source_colours():
    img = _quadrants()
    out = _run("filter.voronoi_crystallize", img, border_thickness=0, scale=12)
    assert _colour_set(out[..., :3]) <= _colour_set(img)
    bordered = _run("filter.voronoi_crystallize", img, border_thickness=0.4, scale=12)
    assert (bordered.sum(axis=2) < 0.05).mean() > 0.01  # black cell borders


def test_equalize_histogram_pixelation_single_tile_stretches_value():
    img = synthetic_image(96, 72) * 0.5 + 0.2
    out = _run("filter.equalize_histogram_pixelation", img, grid_size_x=1, grid_size_y=1)
    v_in, v_out = img.max(axis=2), out.max(axis=2)
    assert v_out.max() == pytest.approx(1.0, abs=1e-5)
    assert v_out.min() <= 0.02
    # monotone mapping of the value
    order = np.argsort(v_in.ravel())
    assert (np.diff(v_out.ravel()[order]) >= -0.0101).all()


def test_polka_dot_dots_and_background():
    img = synthetic_image(96, 72)
    out = _run("filter.polka_dot", img, radius=5, spacing=4, background_color="#00000000")
    assert out.shape[2] == 4
    assert out[0, 0, 3] == 0.0  # corner of the first cell is outside the dot
    assert np.allclose(out[5, 5], np.append(img[5, 5], 1.0))  # dot centre
    assert np.allclose(out[3, 6, :3], img[5, 5])  # inside the same dot
    opaque = _run("filter.polka_dot", img, radius=5, spacing=4)
    assert opaque.shape[2] == 3 and np.allclose(opaque[0, 0], 0.0)


@pytest.mark.parametrize("shape", [0, 1, 2, 3])
def test_smear_mix_zero_is_identity(shape):
    img = synthetic_image(64, 48)
    out = _run("filter.smear", img, mix=0.0, shape=shape, distance=10)
    assert np.allclose(out, img, atol=1e-6)


@pytest.mark.parametrize("shape", [0, 2, 3])
def test_smear_full_mix_only_source_colours(shape):
    img = _quadrants(48, 64)
    out = _run("filter.smear", img, mix=1.0, shape=shape, distance=6, density=0.5)
    assert _colour_set(out) <= _colour_set(img)
    assert not np.allclose(out, img)


def test_smear_blob_replay_matches_forward_painting():
    """The backward replay used for circles / squares equals naive forward painting."""
    from lazyroom.ops.itb import _pixelation_impl as impl

    img = synthetic_image(40, 30)
    for shape, mix in ((2, 0.6), (3, 1.0)):
        out = impl.smear(img, 0.0, 0.3, mix, 3, shape)
        ref = img.copy()
        h, w = img.shape[:2]
        mask = (1 << 48) - 1
        st = (567 ^ 0x5DEECE66D) & mask

        def nxt():
            nonlocal st
            st = (st * 0x5DEECE66D + 0xB) & mask
            v = st >> 16
            return v - (1 << 32) if v >= 1 << 31 else v
        r = 4
        for _ in range(int(np.float32(2.0 * 0.3 * w * h / r))):
            sx, sy = (nxt() & 0x7FFFFFFF) % w, (nxt() & 0x7FFFFFFF) % h
            c = img[sy, sx].copy()
            for x in range(sx - r, sx + r + 1):
                for y in range(sy - r, sy + r + 1):
                    if 0 <= x < w and 0 <= y < h and (shape == 3 or (x - sx) ** 2 + (y - sy) ** 2 <= r * r):
                        ref[y, x] = ref[y, x] * (1 - mix) + c * mix
        assert np.abs(out - ref).max() < 2e-3


def test_weave_gaps_transparent_and_no_gap_identity():
    img = synthetic_image(96, 72)
    out = _run("filter.weave", img)
    assert out.shape[2] == 4 and (out[..., 3] == 0).any()
    same = _run("filter.weave", img, x_gap=0, y_gap=0)
    assert np.allclose(same, img, atol=1e-6)


def test_ascii_draws_characters():
    img = np.ones((80, 120, 3), np.float32)
    out = _run("filter.ascii", img, gradient="█", font_size=10, background_color="#000000")
    assert out.shape == img.shape
    assert 0.3 < (out.mean(axis=2) > 0.5).mean() < 1.0
    grey = _run("filter.ascii", img, gradient="#", font_size=10, background_color="#00000000", gray_scale=True)
    assert grey.shape[2] == 4
    drawn = grey[..., 3] > 0.5
    assert drawn.any() and np.allclose(grey[drawn][:, :3], 0x88 / 255, atol=1e-4)


# ---------------------------------------------------------------------------
# G'MIC filters: exact ImageToolboxLibs command strings and alpha semantics
# ---------------------------------------------------------------------------

EXPECTED_GMIC = {
    "filter.blocks_3d": "fx_blocks3d 32,0,4,1.5,30,60,45,50,50,0,-50,-100,0.5,0.7,1,1,0,0,0,128",
    "filter.chessboard_overlay": "fx_chessboard 64,64,0,0,0,0.25,0,0,0,255,255,255,255,255",
    "filter.circle_abstraction": "fx_circle_abstraction 8,5,0.8,0,1,1,1",
    "filter.dice_mosaic": "fx_dices 2,24,1",
    "filter.diffusion_tensors": "fx_diffusiontensors 10,5,3,1,0.15,1,0,3",
    "filter.ellipsionism": "fx_ellipsionism 20,10,0.5,0.7,3,0.5",
    "filter.faded_array": "fx_array_fade 2,2,0,0,80,90,0,0",
    "filter.flip_and_rotate_blocks": "fx_flip_blocks 4,4,3,1,0",
    "filter.make_seamless": "fx_make_seamless 0",
    "filter.mineral_mosaic": "fx_mineral_mosaic 1,2,1,100,0",
    "filter.ministeck": "fx_ministeck 8,64,8,2,100,0.3,0",
    "filter.mirrored_array": "fx_array_mirror 1,0,0,2,0,0,0",
    "filter.old_school_8_bit": "fx_8bits 25,800,16",
    "filter.puzzle": "fx_puzzle 5,5,0.5,0,0,0.3,100,0.2,255,100,0,0,0,0,0,0",
    "filter.quadtree": "fx_quadtree 0,1024,0.5,0,3,1.5,1,1",
    "filter.random_color_array": "fx_array_color 5,5,0.5",
    "filter.random_patch_array": "fx_array_random 5,5,7,7,0",
    "filter.random_shade_stripes": "fx_shade_stripes 30,1,0.8,1.3,0,0",
    "filter.rebuild_from_similar_blocks": "fx_rebuild_from_similar_blocks 5,10,0.75,1",
    "filter.shapeism": "fx_shapeism 2,7,0.38,0,1,5,32,8,3,0,5,0.5,1,0,0,0,255",
    "filter.shuffle_patches": "fx_shuffle_patches 0,0,64,0,20,0",
    "filter.super_pixels": "fx_superpixels 16,10,5,1,1,1,0,0,0,255",
    "filter.taquin": "fx_taquin 7,7,0,50,5,0,0,0,0,255,0",
    "filter.tileable_rotation": "fx_rotate_tileable 45,8,0",
}
PADDED = {"filter.make_seamless": "power2", "filter.tileable_rotation": "square"}


@pytest.mark.parametrize("op_id", sorted(EXPECTED_GMIC))
def test_gmic_default_command_strings(op_id, monkeypatch):
    from lazyroom.ops.itb import _pixelation_impl as impl

    seen = []

    def fake_run_gmic(img, command, **kw):
        seen.append(command)
        return img

    monkeypatch.setattr(impl, "run_gmic", fake_run_gmic)
    spec = get_op(op_id)
    assert spec.requires == ["gmic"] and spec.source["upstream"] == "G'MIC"
    spec.run(synthetic_image(32, 24), RenderContext(seed=1), spec.validate({}))
    cmd = seen[0]
    assert cmd.startswith("srand ")
    body = cmd.split(" ", 2)[2]
    if op_id in PADDED:
        assert EXPECTED_GMIC[op_id] in body
        assert body.startswith("_gmic_android_w={w} _gmic_android_h={h} ")
        assert ("{2^ceil(log2(max(1,w)))}" in body) == (PADDED[op_id] == "power2")
        assert body.endswith("{int((h-$_gmic_android_h)/2+$_gmic_android_h-1)}")
    else:
        assert body == EXPECTED_GMIC[op_id]


def test_shuffle_patches_gradient_reconstruction_is_padded(monkeypatch):
    from lazyroom.ops.itb import _pixelation_impl as impl

    seen = []
    monkeypatch.setattr(impl, "run_gmic", lambda img, command, **kw: seen.append(command) or img)
    _run("filter.shuffle_patches", synthetic_image(32, 24), reconstruct="gradients")
    assert "fx_shuffle_patches 0,1,64,0,20,0" in seen[0] and "_gmic_android_w" in seen[0]


def test_gmic_preserve_mode_keeps_source_alpha():
    pytest.importorskip("lazyroom.core.backends")
    spec = get_op("filter.old_school_8_bit")
    if not spec.availability()[0]:
        pytest.skip("G'MIC unavailable")
    img = synthetic_image(64, 48, alpha=True)
    out = _run("filter.old_school_8_bit", img)
    assert out.shape == img.shape and np.allclose(out[..., 3], img[..., 3], atol=1e-6)


def test_gmic_process_mode_opacity_zero_chessboard_is_identity():
    spec = get_op("filter.chessboard_overlay")
    if not spec.availability()[0]:
        pytest.skip("G'MIC unavailable")
    img = synthetic_image(64, 48)
    out = _run("filter.chessboard_overlay", img, opacity=0.0)
    assert out.shape == img.shape and np.allclose(out, img, atol=2e-3)


def test_gmic_random_filters_are_deterministic():
    spec = get_op("filter.random_color_array")
    if not spec.availability()[0]:
        pytest.skip("G'MIC unavailable")
    img = synthetic_image(64, 48)
    a = _run("filter.random_color_array", img, _seed=5)
    b = _run("filter.random_color_array", img, _seed=5)
    c = _run("filter.random_color_array", img, _seed=6)
    assert np.allclose(a, b) and not np.allclose(a, c)
