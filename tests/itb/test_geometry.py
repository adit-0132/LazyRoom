"""ImageToolbox "Geometry" group: parameter extremes + semantic checks."""
import json

import cv2
import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.registry import all_ops, get_op
from tests.conftest import synthetic_image

GROUP = "geometry"
SPECS = [s for s in all_ops(include_hidden=True) if s.group == GROUP and s.id.startswith("filter.")]

# Extremes that would only make the test slow (seam insertion up to 16384 px) are capped.
CAPS = {("filter.seam_carving", "width"): 130, ("filter.seam_carving", "height"): 90,
        ("filter.expand_image", "top"): 40, ("filter.expand_image", "start"): 40,
        ("filter.expand_image", "end"): 40, ("filter.expand_image", "bottom"): 40}


def run(op_id, img, **params):
    spec = get_op(op_id)
    return spec.run(img, RenderContext(seed=3), spec.validate(params))


def _extreme(spec, which):
    vals = {}
    for p in spec.params:
        if p.kind in ("float", "int"):
            v = p.min if which == "min" else p.max
            if (spec.id, p.name) in CAPS and which == "max":
                v = CAPS[(spec.id, p.name)]
            vals[p.name] = v
    return vals


def _check(spec, img, out):
    assert out.dtype == np.float32 and out.ndim == 3 and out.shape[2] in (3, 4)
    assert np.isfinite(out).all()
    assert out.shape[0] >= 1 and out.shape[1] >= 1
    if not spec.geometry:
        assert out.shape[:2] == img.shape[:2]


def test_group_complete():
    assert len(SPECS) == 30


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
        _check(spec, img, out)


# ---------------------------------------------------------------------------
# jhlabs transforms
# ---------------------------------------------------------------------------

def test_offset_is_circular_shift():
    img = synthetic_image(80, 60)
    out = run("filter.offset", img, offset_x=0.25, offset_y=0.5)
    assert np.array_equal(out, np.roll(img, (30, 20), axis=(0, 1)))
    assert np.array_equal(run("filter.offset", img, offset_x=0, offset_y=0), img)


def test_shear_zero_is_identity():
    img = synthetic_image(64, 48)
    out = run("filter.shear", img, x_angle=0, y_angle=0)
    assert out.shape[2] == 3 and np.allclose(out, img, atol=1e-4)


def test_shear_creates_transparent_corner():
    out = run("filter.shear", synthetic_image(64, 48), x_angle=30, y_angle=0)
    assert out.shape[2] == 4 and out[0, 0, 3] == 0 and out[-1, -1, 3] > 0.99


def test_distort_perspective_identity_corners():
    img = synthetic_image(64, 48)
    out = run("filter.distort_perspective", img, top_left=[0, 0], top_right=[1, 0], bottom_left=[0, 1],
              bottom_right=[1, 1])
    assert np.allclose(out[..., :3], img, atol=1e-3)


def test_arc_outside_is_transparent():
    out = run("filter.arc", synthetic_image(80, 60))
    assert out.shape[2] == 4
    assert out[0, 0, 3] == 0 and out[30, 40, 3] == 0  # corner and the hole in the middle
    assert out[..., 3].max() == 1.0


def test_feedback_without_motion_is_identity():
    img = synthetic_image(64, 48)
    out = run("filter.feedback", img, distance=0, angle=0, rotation=0, zoom=0)
    assert np.allclose(out, img, atol=1e-4)


def test_kaleidoscope_is_symmetric():
    # theta = triangle(atan2(dy, dx) ...) is even in dy -> mirror symmetry about the centre row (y = 24.5)
    img = synthetic_image(65, 49)
    out = run("filter.kaleidoscope", img, sides=2)
    assert np.allclose(out[1:25], out[25:49][::-1], atol=1e-5)


@pytest.mark.parametrize("kind", ["rect_to_polar", "polar_to_rect", "invert_in_circle"])
def test_polar_coordinates_types(kind):
    img = synthetic_image(64, 48)
    out = run("filter.polar_coordinates", img, type=kind)
    assert out.shape == img.shape and not np.allclose(out, img)


# ---------------------------------------------------------------------------
# Procedural shaders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_id", ["filter.circular_mirror", "filter.progressive_scaling"])
def test_procedural_zero_intensity_is_identity(op_id):
    img = synthetic_image(64, 48)
    out = run(op_id, img, intensity=0)
    assert np.allclose(out, img, atol=1e-3)


@pytest.mark.parametrize("op_id", ["filter.hex_kaleidoscope", "filter.smooth_kaleidoscope",
                                   "filter.random_kaleidoscope_grid", "filter.inversion_fractal"])
def test_procedural_changes_image_and_keeps_range(op_id):
    img = synthetic_image(64, 48)
    out = run(op_id, img)
    assert out.shape == img.shape and not np.allclose(out, img)
    assert out.min() >= img.min() - 1e-3 and out.max() <= img.max() + 1e-3  # only resamples the input


def test_random_kaleidoscope_seed_changes_result():
    img = synthetic_image(64, 48)
    a = run("filter.random_kaleidoscope_grid", img, seed=1)
    b = run("filter.random_kaleidoscope_grid", img, seed=2)
    assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
# OpenCV / Trickle tools
# ---------------------------------------------------------------------------

def test_mirror_left_to_right_symmetric():
    img = synthetic_image(80, 60)
    out = run("filter.mirror", img, center=0.5, side="left_to_right")
    assert out.shape == (60, 80, 3) and np.array_equal(out, out[:, ::-1])
    assert np.array_equal(out[:, :40], img[:, :40])
    out = run("filter.mirror", img, center=0.25, side="bottom_to_top")
    assert out.shape == (2 * 15, 80, 3) and np.array_equal(out, out[::-1])


def test_crop_to_content_crops_border():
    img = np.zeros((50, 70, 3), np.float32)
    img[10:30, 20:45] = 0.7
    out = run("filter.crop_to_content", img, tolerance=0.1)
    assert out.shape == (20, 25, 4)
    assert np.allclose(out[:, :-1, :3], 0.7) and np.all(out[:, :-1, 3] == 1)
    assert np.all(out[:, -1, 3] == 0)  # upstream copies width-1 columns


def test_crop_to_content_nothing_to_crop_returns_input():
    img = np.zeros((20, 30, 3), np.float32)
    assert run("filter.crop_to_content", img).shape == img.shape


def test_crop_or_perspective_rectifies_quad():
    img = synthetic_image(100, 80)
    out = run("filter.crop_or_perspective", img, top_left=[0.2, 0.25], top_right=[0.7, 0.25],
              bottom_left=[0.2, 0.75], bottom_right=[0.7, 0.75])
    assert out.shape[:2] == (40, 50)
    assert np.allclose(out[1:-1, 1:-1], img[21:59, 21:69], atol=1e-3)
    absolute = run("filter.crop_or_perspective", img, top_left=[20, 20], top_right=[70, 20],
                   bottom_left=[20, 60], bottom_right=[70, 60], absolute=True)
    assert absolute.shape[:2] == (40, 50)


def test_auto_crop_band():
    img = np.full((120, 100, 3), 0.5, np.float32)
    rng = np.random.default_rng(0)
    img[40:80] = rng.random((40, 100, 3)).astype(np.float32)  # rows 40..79 are Canny edge-candidate rows
    out = run("filter.auto_crop", img, tolerance=10)
    # candidates [0, 40..79, 120]; the (79, 120) window scores best -> rows 80..119 are kept
    assert np.array_equal(out, img[80:120])
    flat = np.full((60, 50, 3), 0.3, np.float32)
    assert np.array_equal(run("filter.auto_crop", flat), flat)  # no edge rows -> unchanged


def test_expand_image_keeps_source_and_size():
    img = synthetic_image(64, 48)
    out = run("filter.expand_image", img, top=10, start=6, end=4, bottom=8)
    assert out.shape == (48 + 18, 64 + 10, 3)
    centre = out[10:58, 6:70]
    assert np.abs(centre - img).max() < 0.25  # only blended near the seams
    assert np.allclose(centre[12:-12, 12:-12], img[12:-12, 12:-12], atol=1 / 255)
    out2 = run("filter.expand_image", img, top=10, start=6, end=4, bottom=8)
    assert np.array_equal(out, out2)
    assert run("filter.expand_image", img, top=0, start=0, end=0, bottom=0) is not None


def test_expand_image_rgba():
    img = synthetic_image(40, 30, alpha=True)
    out = run("filter.expand_image", img, top=5, start=5, end=5, bottom=5)
    assert out.shape == (40, 50, 4)


def test_seam_carving_resizes():
    img = synthetic_image(60, 40)
    assert run("filter.seam_carving", img).shape == img.shape  # 0x0 = unchanged
    out = run("filter.seam_carving", img, width=50, height=35)
    assert out.shape == (35, 50, 3)
    out = run("filter.seam_carving", img, width=75, height=40, backward_energy=True)
    assert out.shape == (40, 75, 3)


def test_seam_carving_keeps_high_energy_feature():
    img = np.full((40, 60, 3), 0.5, np.float32)
    img[:, 30] = 1.0
    img[:, 31] = 0.0
    out = run("filter.seam_carving", img, width=40, height=40)
    assert out.shape == (40, 40, 3)
    assert (out[..., 0] > 0.9).any(axis=1).all() and (out[..., 0] < 0.1).any(axis=1).all()


def test_seam_carving_protection_and_removal_mask(tmp_path):
    from lazyroom.core.io import save_image

    img = synthetic_image(60, 40)
    mask = np.zeros((40, 60, 3), np.float32)
    mask[:, 10:20] = 1.0
    p = tmp_path / "mask.png"
    save_image(mask, p)
    prot = run("filter.seam_carving", img, width=45, height=40, mask=str(p))
    assert prot.shape == (40, 45, 3)
    for r in range(40):  # seams never cross the protected band: it survives contiguously in every row
        band = img[r, 10:20]
        assert any(np.allclose(prot[r, o:o + 10], band, atol=1 / 255) for o in range(0, 36))
    removed = run("filter.seam_carving", img, width=60, height=40, mask=str(p), use_mask_as_removal=True)
    assert removed.shape == (40, 60, 3)


def test_deskew_straightens_rotated_lines():
    img = np.ones((400, 600, 3), np.float32)
    for y in range(60, 360, 20):
        cv2.line(img, (40, y), (560, y), (0, 0, 0), 2)
    m = cv2.getRotationMatrix2D((300, 200), 5, 1.0)
    skewed = cv2.warpAffine(img, m, (600, 400), borderMode=cv2.BORDER_REPLICATE)
    out = run("filter.deskew", skewed, max=15, allow_crop=False)
    assert out.shape[:2] != skewed.shape[:2]  # rotated onto a larger canvas
    gray = (out[..., 0] < 0.5).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 200, minLineLength=50, maxLineGap=4).reshape(-1, 4)
    ang = np.degrees(np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0]))
    assert abs(np.median(ang)) < 1.0
    cropped = run("filter.deskew", skewed, max=15, allow_crop=True)
    assert cropped.shape[0] < out.shape[0] and cropped.shape[1] < out.shape[1]


def test_lens_correction_profile_and_explicit_params(tmp_path):
    img = synthetic_image(96, 72)
    assert np.array_equal(run("filter.lens_correction", img), img)  # nothing bundled -> unchanged
    profile = {"calib_dimension": {"w": 1920, "h": 1080},
               "fisheye_params": {"camera_matrix": [[1672.4, 0, 933.6], [0, 1677.6, 539.5], [0, 0, 1]],
                                  "distortion_coeffs": [0.25, -0.44, 3.9, -8.2]}}
    p = tmp_path / "lens.json"
    p.write_text(json.dumps(profile))
    a = run("filter.lens_correction", img, lens_profile=str(p), strength=1.0)
    b = run("filter.lens_correction", img, strength=1.0,
            camera_matrix=[1672.4, 0, 933.6, 0, 1677.6, 539.5, 0, 0, 1], distortion_coeffs=[0.25, -0.44, 3.9, -8.2],
            calib_width=1920, calib_height=1080)
    assert a.shape == img.shape and np.allclose(a, b)
    assert not np.allclose(a, img)
    with pytest.raises(Exception):
        run("filter.lens_correction", img, camera_matrix=[1, 2, 3])


# ---------------------------------------------------------------------------
# G'MIC
# ---------------------------------------------------------------------------

GMIC = [s for s in SPECS if "gmic" in s.requires]


@pytest.mark.parametrize("spec", GMIC, ids=[s.id for s in GMIC])
def test_gmic_ops_run_rgb_and_rgba(spec):
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    for alpha in (False, True):
        img = synthetic_image(64, 48, alpha=alpha)
        out = spec.run(img, RenderContext(), spec.validate({}))
        _check(spec, img, out)
        if alpha:
            assert out.shape[2] == 4
        assert not np.allclose(out[..., :3].mean(), 0)


def test_gmic_polar_transform_expression_choices():
    spec = get_op("filter.polar_transform")
    if not spec.availability()[0]:
        pytest.skip("gmic unavailable")
    img = synthetic_image(64, 48)
    a = run("filter.polar_transform", img, radius_expression="identity", angle_expression="identity")
    b = run("filter.polar_transform", img, radius_expression="wavy", angle_expression="spiral")
    assert np.allclose(a[..., :3], img, atol=0.02)  # r, a -> identity
    assert not np.allclose(a, b)


def test_square_to_circle_transparent_corners():
    if not get_op("filter.square_to_circle").availability()[0]:
        pytest.skip("gmic unavailable")
    out = run("filter.square_to_circle", synthetic_image(64, 48))
    assert out.shape[2] == 4 and out[0, 0, 3] < 0.01 and out[24, 32, 3] > 0.99
