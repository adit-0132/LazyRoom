"""ImageToolbox "Artistic" group: extremes + semantic checks."""
import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.errors import LazyRoomError
from lazyroom.core.registry import all_ops, get_op
from tests.conftest import synthetic_image

SPECS = [s for s in all_ops(include_hidden=True) if s.group == "artistic"]


def _run(op_id, img, **values):
    spec = get_op(op_id)
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    return spec.run(img, RenderContext(seed=5), spec.validate(values))


def test_group_has_all_filters():
    assert len(SPECS) == 40


# Upstream G'MIC errors (also in ImageToolbox): with threshold=1024 no string survives on a small
# image and fx_stringify's `sort_list[3--1]` fails.
UPSTREAM_FAILURES = {("filter.stringify", "max")}


@pytest.mark.parametrize("extreme", ["min", "max"])
@pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])
def test_numeric_extremes(spec, extreme):
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    values = {p.name: getattr(p, extreme) for p in spec.params if p.kind in ("int", "float")}
    img = synthetic_image(48, 36, alpha=True)
    if (spec.id, extreme) in UPSTREAM_FAILURES:
        with pytest.raises(LazyRoomError, match="G'MIC command failed"):
            spec.run(img, RenderContext(seed=1), spec.validate(values))
        return
    out = spec.run(img, RenderContext(seed=1), spec.validate(values))
    assert out.dtype == np.float32 and out.ndim == 3 and out.shape[2] in (3, 4)
    assert np.isfinite(out).all()
    if not spec.geometry:
        assert out.shape[:2] == img.shape[:2]


# --- GPUImage toon family ---------------------------------------------------

def test_toon_flat_image_is_posterized_without_edges():
    img = np.full((20, 24, 3), 0.43, np.float32)
    out = _run("filter.toon", img, threshold=0.2, quantization_levels=10)
    assert np.allclose(out, 0.4, atol=1e-6)  # floor(4.3 + 0.5) / 10
    out = _run("filter.smooth_toon", img, blur_size=2.0, threshold=0.2, quantization_levels=4)
    assert np.allclose(out, 0.5, atol=1e-6)


def test_toon_draws_black_edges():
    img = np.zeros((20, 20, 3), np.float32)
    img[:, 10:] = 1.0
    out = _run("filter.toon", img)
    assert np.all(out[:, 9:11] == 0)  # step edge -> black outline
    assert np.allclose(out[:, :8], 0) and np.allclose(out[:, 12:], 1)


# --- jhlabs -----------------------------------------------------------------

def test_rubber_stamp_two_colours():
    out = _run("filter.rubber_stamp", synthetic_image(64, 48), brush_softness=0.0, radius=0.0,
               first_color="#FF0000", second_color="#0000FF")
    cols = {tuple(c) for c in np.round(out.reshape(-1, out.shape[2])[:, :3] * 255).astype(int)}
    assert cols <= {(255, 0, 0), (0, 0, 255)} and len(cols) == 2


def test_pointillize_without_border_only_uses_source_colours():
    img = synthetic_image(40, 30)
    out = _run("filter.pointillize", img, border_thickness=5.0, scale=8)
    src = {tuple(c) for c in np.round(img.reshape(-1, 3) * 255).astype(int)}
    got = {tuple(c) for c in np.round(out.reshape(-1, out.shape[2])[:, :3] * 255).astype(int)}
    assert got <= src
    assert len(got) < len(src) / 4  # cells -> few colours


def test_java_random_matches_java():
    from lazyroom.ops.itb._artistic_impl import _JavaRandom

    assert _JavaRandom(0).next_int() == -1155484576
    assert _JavaRandom(42).next_int() == -1170105035


# --- aire -------------------------------------------------------------------

def test_oil_picks_existing_pixels():
    img = synthetic_image(40, 30)
    out = _run("filter.oil", img, radius=3, strength=1.0)
    src = {tuple(c) for c in np.round(img.reshape(-1, 3) * 255).astype(int)}
    got = {tuple(c) for c in np.round(out[..., :3].reshape(-1, 3) * 255).astype(int)}
    assert got <= src


def test_sketch_stretches_each_channel_to_white():
    out = _run("filter.sketch", synthetic_image(64, 48))
    assert np.allclose(out[..., :3].reshape(-1, 3).max(0), 1.0)


def test_aire_median_quirk_matches_reference_semantics():
    from lazyroom.ops.itb._artistic_impl import _aire_median_channel

    flat = np.full((30, 40), 77, np.uint8)
    assert np.all(_aire_median_channel(flat, 21) == 77)


# --- Trickle ----------------------------------------------------------------

def test_enhanced_oil_flat_and_mode():
    flat = np.full((16, 16, 3), 0.6, np.float32)
    assert np.allclose(_run("filter.enhanced_oil", flat, strength=3), np.round(0.6 * 255) / 255)
    img = synthetic_image(40, 30)
    out = _run("filter.enhanced_oil", img, strength=2)
    for c in range(3):  # per-channel mode -> values come from that channel
        vals = lambda x: set(np.round(x[..., c] * 255).astype(int).ravel())  # noqa: E731
        assert vals(out) <= vals(img)


def test_pop_art_quadrants():
    img = synthetic_image(64, 48)
    out = _run("filter.pop_art", img, strength=0.0)
    q = [out[:24, :32], out[:24, 32:], out[24:, :32], out[24:, 32:]]
    for t in q[1:]:
        assert np.allclose(t, q[0], atol=1e-6)  # strength 0 -> four copies of the half-size image
    out = _run("filter.pop_art", synthetic_image(65, 49), color="#FF0000")
    # odd size: cairo leaves the last row/column transparent
    assert out.shape[2] == 4 and np.all(out[:, -1, 3] == 0) and np.all(out[-1, :, 3] == 0)


def test_square_harmony_like_android():
    from lazyroom.ops.itb._artistic_impl import square_harmony

    assert square_harmony(255, 0, 0) == [(255, 0, 0), (128, 255, 0), (0, 255, 255), (128, 0, 255)]


def test_low_poly_fill_and_outline():
    img = synthetic_image(64, 48)
    filled = _run("filter.low_poly", img, strength=200, fill=True)
    assert filled.shape[2] == 3
    outline = _run("filter.low_poly", img, strength=200, fill=False)
    assert outline.shape[2] == 4 and outline[..., 3].mean() < 0.5


def test_sand_painting_background():
    out = _run("filter.sand_painting", synthetic_image(64, 48), strength=50, background_color="#00FF00")
    green = np.all(np.abs(out[..., :3] - [0, 1, 0]) < 1e-6, axis=-1)
    assert 0.8 < green.mean() < 1.0


# --- G'MIC ------------------------------------------------------------------

def test_reescape_spaces_in_substitutions():
    from lazyroom.ops.itb._artistic_impl import _reescape_spaces

    assert _reescape_spaces('check ${is_image_arg $1}" && a b" x') == 'check ${is_image_arg\\ $1}" && a b" x'


def test_gmic_charcoal_is_monochrome():
    out = _run("filter.charcoal", synthetic_image(48, 36))
    assert np.allclose(out[..., 0], out[..., 1], atol=1e-6) and np.allclose(out[..., 1], out[..., 2], atol=1e-6)


def test_gmic_smooth_disabled_like_imagetoolbox():
    # fx_pen_drawing's amplitude only feeds `smooth`, which ImageToolbox skips
    img = synthetic_image(48, 36)
    a = _run("filter.pen_drawing", img, amplitude=0.0)
    b = _run("filter.pen_drawing", img, amplitude=30.0)
    assert np.allclose(a, b, atol=1e-6)


def test_gmic_warhol_tiles_geometry():
    # tiles are the image scaled by 1/max(x_tiles, y_tiles): 3x2 tiles of 16x12 -> 48x24
    img = synthetic_image(48, 36, alpha=True)
    out = _run("filter.warhol", img, x_tiles=3, y_tiles=2)
    assert out.shape == (24, 48, 4)


def test_gmic_engrave_processes_alpha():
    img = synthetic_image(48, 36, alpha=True)
    out = _run("filter.engrave", img)
    assert out.shape == img.shape


@pytest.mark.parametrize("mode", ["multiply", "color_burn", "soft_light", "hsl_color", "hsl_hue", "difference"])
def test_pop_art_overlay_modes(mode):
    img = synthetic_image(40, 30, alpha=True)
    out = _run("filter.pop_art", img, overlay_mode=mode, color="#3366CC", strength=0.7)
    assert out.shape == img.shape and np.isfinite(out).all()
    ref = _run("filter.pop_art", img, overlay_mode=mode, strength=0.0)
    assert not np.allclose(out, ref)
