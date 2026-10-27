"""ImageToolbox "Frames" group: parameter extremes + semantic checks."""
import math

import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.errors import LazyRoomError
from lazyroom.core.registry import all_ops, get_op
from tests.conftest import synthetic_image

GROUP = "frames"
SPECS = [s for s in all_ops(include_hidden=True) if s.group == GROUP and s.id.startswith("filter.")]


def run(op_id, img, ctx=None, **params):
    spec = get_op(op_id)
    return spec.run(img, ctx or RenderContext(seed=3), spec.validate(params))


# density 100 makes G'MIC's remove_pixels fail (upstream too, see test_painting_frame_density_100_fails_like_upstream)
CAPS = {("filter.painting_frame", "density"): 99}


def _extreme(spec, which):
    vals = {p.name: (p.min if which == "min" else p.max) for p in spec.params if p.kind in ("float", "int")}
    if which == "max":
        vals.update({k[1]: v for k, v in CAPS.items() if k[0] == spec.id})
    return vals


def test_group_complete():
    assert len(SPECS) == 12


@pytest.mark.parametrize("which", ["min", "max"])
@pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])
def test_param_extremes(spec, which):
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    vals = spec.validate(_extreme(spec, which))
    for alpha in (False, True):
        img = synthetic_image(64, 48, alpha=alpha)
        out = spec.run(img, RenderContext(seed=5), vals)
        assert out.dtype == np.float32 and out.ndim == 3 and out.shape[2] in (3, 4)
        assert np.isfinite(out).all()
        if not spec.geometry:
            assert out.shape[:2] == img.shape[:2]


# ---------------------------------------------------------------------------
# Canvas frames (ImageToolbox)
# ---------------------------------------------------------------------------

def test_border_frame_geometry_and_colour():
    img = synthetic_image(40, 30)
    out = run("filter.border_frame", img, horizontal_border_thickness=5, vertical_border_thickness=7,
              border_color="#FF0000")
    assert out.shape == (30 + 14, 40 + 10, 3)
    assert np.allclose(out[:7], [1, 0, 0]) and np.allclose(out[:, :5], [1, 0, 0])
    assert np.array_equal(out[7:37, 5:45], img)


def test_border_frame_rgba_composites_over_colour():
    img = np.zeros((10, 10, 4), np.float32)
    img[..., 1] = 1.0
    img[..., 3] = 0.5
    out = run("filter.border_frame", img, horizontal_border_thickness=2, vertical_border_thickness=2,
              border_color="#0000FF")
    assert out.shape == (14, 14, 4)
    assert np.allclose(out[7, 7], [0, 0.5, 0.5, 1.0], atol=1e-5)


def test_drop_shadow_padding_and_shadow():
    img = synthetic_image(40, 30)
    out = run("filter.drop_shadow", img, blur_radius=6, offset_x=0, offset_y=4)
    # padding: left/right ceil(6), top ceil(6 - 4) = 2, bottom ceil(6 + 4) = 10
    assert out.shape == (30 + 2 + 10, 40 + 12, 4)
    assert np.allclose(out[2:32, 6:46, :3], img) and np.all(out[2:32, 6:46, 3] == 1)
    below = out[33:36, 10:40, 3]
    assert below.min() > 0.05 and np.allclose(out[33:36, 10:40, :3], 0)  # black shadow under the image
    assert out[0, 0, 3] < 0.05


def test_drop_shadow_without_blur_is_hard():
    img = synthetic_image(20, 20)
    out = run("filter.drop_shadow", img, blur_radius=0, offset_x=3, offset_y=3, shadow_color="#00FF00")
    assert out.shape == (23, 23, 4)
    assert np.allclose(out[21, 10], [0, 1, 0, 1]) and out[1, 21, 3] == 0


@pytest.mark.parametrize("side", ["start", "end", "top", "bottom"])
def test_side_fade_gradient(side):
    img = synthetic_image(40, 30)
    out = run("filter.side_fade", img, strength=0.5, side=side)
    a = out[..., 3]
    prof = a[15] if side in ("start", "end") else a[:, 20]
    if side in ("end", "bottom"):
        prof = prof[::-1]
    n = 20 if side in ("start", "end") else 15
    assert np.all(np.diff(prof[:n]) > 0) and np.all(prof[n:] == 1)
    assert prof[0] < 0.1
    assert np.allclose(out[..., :3], img)


def test_side_fade_zero_strength_identity():
    img = synthetic_image(40, 30)
    assert np.array_equal(run("filter.side_fade", img, strength=0), img)


def test_torn_edge_clears_borders_deterministically():
    img = synthetic_image(80, 60)
    a = run("filter.torn_edge", img)
    b = run("filter.torn_edge", img, ctx=RenderContext(seed=99))  # java.util.Random seeded by size + params
    assert np.array_equal(a, b)
    alpha = a[..., 3]
    assert alpha[0, 0] == 0 and alpha[-1, -1] == 0 and alpha[0, 40] == 0  # first row always inside a tooth
    assert np.all(alpha[17:-17, 17:-17] == 1)
    assert np.allclose(a[..., :3], img)
    c = run("filter.torn_edge", img, tooth_height=8)
    assert not np.array_equal(a, c)


def test_torn_edge_no_edges_is_opaque():
    img = synthetic_image(40, 30)
    out = run("filter.torn_edge", img, top_edge=False, right_edge=False, bottom_edge=False, left_edge=False)
    assert np.all(out[..., 3] == 1)


def test_torn_edge_teeth_reference():
    # tooth heights drawn from java.util.Random((31 * w + h) * 31 + TornEdgeParams.hashCode())
    from lazyroom.ops.itb._geometry_impl import JavaRandom, kotlin_hash

    h = kotlin_hash(16, 22, 22, True, False, False, False)
    r = JavaRandom((31 * 80 + 60) * 31 + h)
    first = r.next_int(16) + 1
    out = run("filter.torn_edge", synthetic_image(80, 60), right_edge=False, bottom_edge=False, left_edge=False)
    # the top path starts at (w, first tooth): the right-most column is cleared down to that depth
    col = out[:, -1, 3]
    assert math.isclose(float((1 - col).sum()), first - 0.5, abs_tol=1.0)


# ---------------------------------------------------------------------------
# G'MIC frames
# ---------------------------------------------------------------------------

GMIC = [s for s in SPECS if "gmic" in s.requires]


@pytest.mark.parametrize("spec", GMIC, ids=[s.id for s in GMIC])
def test_gmic_frames_run(spec):
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    for alpha in (False, True):
        img = synthetic_image(64, 48, alpha=alpha)
        out = spec.run(img, RenderContext(), spec.validate({}))
        assert np.isfinite(out).all() and out.shape[2] in (3, 4)
        if alpha:
            assert out.shape[2] == 4
        if spec.geometry:
            assert out.shape[0] >= 48 and out.shape[1] >= 64  # frames grow the canvas


@pytest.mark.parametrize("op_id", ["filter.fuzzy_frame", "filter.painting_frame"])
def test_gmic_random_frames_are_seeded(op_id):
    if not get_op(op_id).availability()[0]:
        pytest.skip("gmic unavailable")
    img = synthetic_image(64, 48)
    a = run(op_id, img, ctx=RenderContext(seed=1))
    b = run(op_id, img, ctx=RenderContext(seed=1))
    assert np.array_equal(a, b)


def test_painting_frame_density_100_fails_like_upstream():
    # G'MIC's remove_pixels fails when asked to remove 0 defect pixels (density 100, default settings);
    # the stdlib bundled by ImageToolbox (G'MIC 4.0.2) has the same definition, so ImageToolbox errors too.
    if not get_op("filter.painting_frame").availability()[0]:
        pytest.skip("gmic unavailable")
    with pytest.raises(LazyRoomError):
        run("filter.painting_frame", synthetic_image(64, 48), density=100)


def test_preserve_mode_frames_keep_source_alpha_semantics():
    # Polaroid/Smooth frames run in GmicAlphaMode.Preserve: RGB in, G'MIC alpha discarded
    if not get_op("filter.polaroid_frame").availability()[0]:
        pytest.skip("gmic unavailable")
    out = run("filter.polaroid_frame", synthetic_image(64, 48))
    assert out.shape[2] == 3
    out = run("filter.smooth_frame", synthetic_image(64, 48, alpha=True))
    assert out.shape[2] == 4 and out.shape[:2] == (48 + 2 * round(0.05 * 64), 64 + 2 * round(0.05 * 64))
