"""ImageToolbox "Distortion" group: parameter extremes + semantic checks."""
import cv2
import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.registry import all_ops, get_op
from tests.conftest import synthetic_image

GROUP = "distortion"
SPECS = [s for s in all_ops(include_hidden=True) if s.group == GROUP and s.id.startswith("filter.")]
FAST = [s for s in SPECS if "gmic" not in s.requires]
GMIC = [s for s in SPECS if "gmic" in s.requires]


def run(op_id, img, ctx=None, **params):
    spec = get_op(op_id)
    return spec.run(img, ctx or RenderContext(seed=3), spec.validate(params))


def _numeric(spec):
    return [p for p in spec.params if p.kind in ("float", "int")]


def _check(spec, img, out):
    assert out.dtype == np.float32 and out.ndim == 3 and out.shape[2] in (3, 4)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    if not spec.geometry:
        assert out.shape[:2] == img.shape[:2]
    if img.shape[2] == 4:
        assert out.shape[2] == 4


def _gmic_ok(spec):
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")


def test_group_complete():
    assert len(SPECS) == 43
    assert len(GMIC) == 10


@pytest.mark.parametrize("which", ["min", "max"])
@pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])
def test_param_extremes_all(spec, which):
    _gmic_ok(spec)
    vals = spec.validate({p.name: (p.min if which == "min" else p.max) for p in _numeric(spec)})
    for alpha in (False, True):
        img = synthetic_image(48, 36, alpha=alpha)
        _check(spec, img, spec.run(img, RenderContext(seed=5), vals))


@pytest.mark.parametrize("spec", FAST, ids=[s.id for s in FAST])
def test_param_extremes_each(spec):
    """Every numeric parameter at its min and max with the others at their defaults."""
    img = synthetic_image(40, 30, alpha=True)
    for p in _numeric(spec):
        for v in (p.min, p.max):
            out = spec.run(img, RenderContext(seed=5), spec.validate({p.name: v}))
            _check(spec, img, out)


@pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])
def test_input_not_modified(spec):
    _gmic_ok(spec)
    img = synthetic_image(40, 30, alpha=True)
    before = img.copy()
    spec.run(img, RenderContext(seed=1), spec.validate({}))
    assert np.array_equal(img, before)


# ---------------------------------------------------------------------------
# Identity / no-op settings of the coordinate-mapping filters
# ---------------------------------------------------------------------------

IDENTITY = [
    ("filter.glass_rect_tiles", dict(intensity=0, distortion=0)),
    ("filter.glass_rect_tiles", dict(intensity=0, distortion=0, rotation=37, scale=2.5, offset_x=0.3)),
    ("filter.glass_hex_tiles", dict(intensity=0, distortion=0)),
    ("filter.glass_triangle_tiles", dict(intensity=0, distortion=0)),
    ("filter.radial_shimmer", dict(intensity=0)),
    ("filter.grid_sine_distortion2", dict(intensity=0)),
    ("filter.helix_waves", dict(intensity=0)),
    ("filter.broken_cells", dict(intensity=0)),
    ("filter.broken_glass", dict(intensity=0)),
    ("filter.gallium", dict(intensity=0)),
    ("filter.quicksilver", dict(intensity=0, displacement=0)),
    ("filter.rubidium", dict(intensity=0, displacement=0)),
    ("filter.radial_interpolate", dict(intensity=0)),
    ("filter.bulge_distortion", dict(scale=0)),
    ("filter.swirl_distortion", dict(angle=0)),
    ("filter.twirl", dict(angle=0)),
    ("filter.pinch", dict(angle=0, amount=0)),
    ("filter.water_drop", dict(amplitude=0)),
    ("filter.ripple", dict(x_amplitude=0, y_amplitude=0)),
    ("filter.sphere_lens_distortion", dict(refraction_index=1)),
    ("filter.fractal_glass", dict(amplitude=0)),
    ("filter.water_effect", dict(amplitude_x=0, amplitude_y=0)),
    ("filter.distortion", dict(strength=0)),
    ("filter.marble", dict(strength=0)),
]


@pytest.mark.parametrize("op_id,params", IDENTITY, ids=[f"{o}-{i}" for i, (o, _) in enumerate(IDENTITY)])
def test_identity_settings(op_id, params):
    img = synthetic_image(81, 61, alpha=True)
    out = run(op_id, img, **params)
    assert out.shape == img.shape
    # GPU filters work on premultiplied textures (colour of alpha-0 pixels is lost, like on Android)
    np.testing.assert_allclose(out[..., :3] * out[..., 3:], img[..., :3] * img[..., 3:], atol=2e-4)
    np.testing.assert_allclose(out[..., 3], img[..., 3], atol=2e-4)
    rgb = synthetic_image(81, 61)
    np.testing.assert_allclose(run(op_id, rgb, **params), rgb, atol=2e-4)


@pytest.mark.parametrize("spec", FAST, ids=[s.id for s in FAST])
def test_defaults_change_image(spec):
    img = synthetic_image(80, 60)
    out = spec.run(img, RenderContext(seed=2), spec.validate({}))
    assert np.abs(out[..., :3] - img).mean() > 1e-3


# ---------------------------------------------------------------------------
# Family-specific semantics
# ---------------------------------------------------------------------------

def test_alpha_moves_with_pixels():
    img = synthetic_image(80, 60, alpha=True)
    out = run("filter.twirl", img, angle=180, radius=1)
    assert out.shape[2] == 4
    assert np.abs(out[..., 3] - img[..., 3]).max() > 0.05


def test_rgb_clamp_edges_become_transparent():
    """jhlabs RGB_CLAMP (Pinch, Ripple): samples outside the source get alpha 0, even for RGB input."""
    img = synthetic_image(80, 60)
    out = run("filter.ripple", img, x_amplitude=10, y_amplitude=10)
    assert out.shape[2] == 4 and out[..., 3].min() < 0.5
    out = run("filter.twirl", img, angle=200, radius=1)  # CLAMP edges stay opaque
    assert out.shape[2] == 3


def test_sphere_refraction_transparent_outside():
    img = synthetic_image(80, 60)
    for op in ("filter.sphere_refraction", "filter.glass_sphere_refraction"):
        out = run(op, img)
        assert out.shape[2] == 4
        assert out[0, 0, 3] == 0 and out[30, 40, 3] == 1


def test_swirl_and_bulge_leave_outside_untouched():
    img = synthetic_image(80, 60)
    out = run("filter.swirl_distortion", img, radius=0.2)
    np.testing.assert_allclose(out[:5], img[:5], atol=1e-4)
    out = run("filter.bulge_distortion", img, radius=0.2)
    np.testing.assert_allclose(out[:5], img[:5], atol=1e-4)
    assert np.abs(out[25:35, 35:45] - img[25:35, 35:45]).max() > 0.01


def test_anaglyph_channels():
    img = synthetic_image(60, 40)
    out = run("filter.anaglyph", img, shift=5)
    np.testing.assert_allclose(out[..., 0], np.roll(img[..., 0], -5, axis=1), atol=1e-6)
    g = np.minimum(1.0, np.roll(img[..., 1], 5, axis=1) + img[..., 1])
    np.testing.assert_allclose(out[..., 1], g, atol=1e-6)


def test_nearest_neighbour_filters_copy_pixels():
    """aire water / fractal glass / marble only move pixels (no new colours)."""
    img = synthetic_image(64, 48)
    palette = {tuple(p) for p in img.reshape(-1, 3).round(6)}
    for op, kw in (("filter.water_effect", {}), ("filter.fractal_glass", dict(strength=0.1, amplitude=0.3)),
                   ("filter.marble", dict(strength=0.05))):
        out = run(op, img, **kw)
        assert {tuple(p) for p in out.reshape(-1, 3).round(6)} <= palette, op


def test_fractal_glass_is_column_permutation():
    img = synthetic_image(64, 48)
    out = run("filter.fractal_glass", img, strength=0.1, amplitude=0.2)
    cols = [int(np.argmin(np.abs(img[:, :, :] - out[:, x:x + 1, :]).sum(axis=(0, 2)))) for x in range(64)]
    np.testing.assert_allclose(out, img[:, cols], atol=1e-6)


def test_marble_seeded():
    img = synthetic_image(64, 48)
    a = run("filter.marble", img, RenderContext(seed=1), strength=0.1)
    b = run("filter.marble", img, RenderContext(seed=1), strength=0.1)
    c = run("filter.marble", img, RenderContext(seed=2), strength=0.1)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_procedural_seed_changes_pattern():
    img = synthetic_image(64, 48)
    a = run("filter.broken_cells", img, seed=0)
    b = run("filter.broken_cells", img, seed=1)
    assert np.abs(a - b).mean() > 1e-3


def test_procedural_mirror_sampling_matches_fold():
    """sampleField's mirror fold + half-texel clamp == bilinear with BORDER_REFLECT."""
    from lazyroom.ops.itb import _distortion_impl as impl

    img = synthetic_image(61, 41)
    h, w = img.shape[:2]
    vals = dict(intensity=0.0, distortion=0.0, frequency=8.0, scale=0.3, rotation=30.0, offset_x=0.4)
    out = impl.procedural(img, "GlassRectTiles", vals)
    f = impl.Field(h, w, 0.4, 0.0, 0.3, 30.0)
    u = ((np.arange(w) + 0.5) / w)[None, :]
    v = ((np.arange(h) + 0.5) / h)[:, None]
    px, py = f.to_field(u, v)
    X, Y = px * f.scale, py * f.scale
    uu = (f.c * X - f.s * Y) / f.ax + 0.5 + f.ox
    vv = (f.s * X + f.c * Y) / f.ay + 0.5 + f.oy
    uu = np.clip(1 - np.abs(np.mod(uu, 2) - 1), 0.5 / w, 1 - 0.5 / w)
    vv = np.clip(1 - np.abs(np.mod(vv, 2) - 1), 0.5 / h, 1 - 0.5 / h)
    ref = cv2.remap(img, (uu * w - 0.5).astype(np.float32), (vv * h - 0.5).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    np.testing.assert_allclose(out, ref, atol=1e-3)


def test_voronoi_kernel_matches_numpy_path():
    from lazyroom.ops.itb import _distortion_impl as impl

    rng = np.random.default_rng(0)
    px = (rng.random((50, 60), dtype=np.float32) - 0.5) * 3
    py = (rng.random((50, 60), dtype=np.float32) - 0.5) * 2
    a = impl._voronoi(px, py, 8.0, 17, 0.8, (17, 19.3))
    limit = impl._CellTable.LIMIT
    impl._CellTable.LIMIT = 0
    try:
        b = impl._voronoi(px, py, 8.0, 17, 0.8, (17, 19.3))
    finally:
        impl._CellTable.LIMIT = limit
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_preview_scale_consistency_px_params():
    """px parameters scale with ctx.scale so a half-size preview matches the downscaled export."""
    big = synthetic_image(160, 120)
    small = cv2.resize(big, (80, 60), interpolation=cv2.INTER_AREA)
    full = run("filter.ripple", big, RenderContext(seed=1), x_amplitude=6, x_wavelength=20, y_amplitude=0)
    prev = run("filter.ripple", small, RenderContext(seed=1, scale=0.5), x_amplitude=6, x_wavelength=20,
               y_amplitude=0)
    down = cv2.resize(full[..., :3], (80, 60), interpolation=cv2.INTER_AREA)
    assert np.abs(prev[5:-5, 5:-5, :3] - down[5:-5, 5:-5]).mean() < 0.03


def test_content_aware_distortion_keeps_size():
    img = synthetic_image(90, 70, alpha=True)
    out = run("filter.distortion", img, strength=60)
    assert out.shape == img.shape


# ---------------------------------------------------------------------------
# G'MIC
# ---------------------------------------------------------------------------

CHOICE_SAMPLES = {"channel": ["all", "rgb", "luminance", "lab_a", "hsv_value", "cmyk", "ryb_blue", "alpha"]}


@pytest.mark.gmic
@pytest.mark.parametrize("spec", GMIC, ids=[s.id for s in GMIC])
def test_gmic_choices_run(spec):
    """Every enumerated value (a sample of the 47 channels) is accepted by the dev G'MIC CLI."""
    _gmic_ok(spec)
    img = synthetic_image(32, 24)
    for p in spec.params:
        if p.kind != "choice":
            continue
        for c in CHOICE_SAMPLES.get(p.name, p.choices):
            _check(spec, img, spec.run(img, RenderContext(seed=1), spec.validate({p.name: c})))
    for p in spec.params:
        if p.kind == "bool":
            _check(spec, img, spec.run(img, RenderContext(seed=1), spec.validate({p.name: not p.default})))


@pytest.mark.gmic
def test_gmic_random_filters_deterministic_per_seed():
    _gmic_ok(get_op("filter.crease"))
    img = synthetic_image(48, 36)
    a = run("filter.crease", img, RenderContext(seed=1))
    b = run("filter.crease", img, RenderContext(seed=1))
    c = run("filter.crease", img, RenderContext(seed=2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


@pytest.mark.gmic
def test_gmic_lens_transparent_boundary_adds_alpha():
    _gmic_ok(get_op("filter.lens_distortion"))
    img = synthetic_image(64, 48)
    out = run("filter.lens_distortion", img, amplitude=0.5)
    assert out.shape[2] == 4 and out[0, 0, 3] < 0.5 and out[24, 32, 3] > 0.99
    out = run("filter.lens_distortion", img, amplitude=0.5, boundary="mirror")
    assert out.shape[2] == 3


@pytest.mark.gmic
def test_gmic_water_reflection_changes_geometry():
    spec = get_op("filter.water_reflection")
    _gmic_ok(spec)
    assert spec.geometry
    out = run("filter.water_reflection", synthetic_image(64, 48))
    assert out.shape[2] == 4 and out.shape[:2] != (48, 64)


@pytest.mark.gmic
def test_gmic_displacement(tmp_path):
    from lazyroom.core.io import save_image

    _gmic_ok(get_op("filter.displacement"))
    img = synthetic_image(64, 48)
    assert np.array_equal(run("filter.displacement", img), img)  # no map -> unchanged (like ImageToolbox)
    flat = np.full((48, 64, 3), 0.5, np.float32)
    save_image(flat, tmp_path / "flat.png")
    out = run("filter.displacement", img, horizontal_map=str(tmp_path / "flat.png"))
    assert np.abs(out[..., :3] - img).mean() < 0.005  # mid grey (128) -> ~0.03 px shift
    ramp = np.repeat(np.linspace(0, 1, 64, dtype=np.float32)[None, :, None], 48, axis=0).repeat(3, axis=2)
    save_image(ramp, tmp_path / "ramp.png")
    out = run("filter.displacement", img, horizontal_map=str(tmp_path / "ramp.png"), strength_x=0.2,
              strength_y=0.0)
    assert np.abs(out[..., :3] - img).mean() > 0.01
