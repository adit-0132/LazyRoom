"""ImageToolbox "Color" group: extremes + semantic checks."""
import math
import re

import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.registry import all_ops, get_op
from tests.conftest import synthetic_image

SPECS = [s for s in all_ops(include_hidden=True) if s.group == "color"]


def _run(op_id, img, **values):
    spec = get_op(op_id)
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    return spec.run(img, RenderContext(seed=3), spec.validate(values))


@pytest.fixture(scope="module")
def ref_path(tmp_path_factory):
    from lazyroom.core.io import save_image

    p = tmp_path_factory.mktemp("ref") / "reference.png"
    ref = synthetic_image(80, 60, seed=5)[..., ::-1] * np.array([1.0, 0.8, 0.5], np.float32)
    save_image(np.ascontiguousarray(ref, np.float32), p)
    return str(p)


def test_group_complete():
    assert len(SPECS) == 43


def _extreme_values(spec, which, ref_path):
    vals = {}
    for p in spec.params:
        if p.kind in ("float", "int") and p.min is not None and p.max is not None:
            vals[p.name] = p.min if which == "min" else p.max
        elif p.kind == "file":
            vals[p.name] = ref_path
    return vals


@pytest.mark.parametrize("which", ["min", "max"])
@pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])
def test_extremes(spec, which, ref_path):
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    for alpha in (False, True):
        img = synthetic_image(48, 36, alpha=alpha)
        out = spec.run(img, RenderContext(seed=1), spec.validate(_extreme_values(spec, which, ref_path)))
        assert out.dtype == np.float32 and out.shape[:2] == img.shape[:2] and out.shape[2] in (3, 4)
        assert np.isfinite(out).all()
        assert out.min() >= 0.0 and out.max() <= 1.0


# ---------------------------------------------------------------------------
# G'MIC: exact ImageToolboxLibs command strings (defaults)
# ---------------------------------------------------------------------------

EXPECTED_GMIC = {
    "filter.black_and_white_film":
        "fx_bwfilmsimulate 15,0.299,0,0.587,0,0.114,0,1.05,1.2,0,30,0.08,18,10,5,2,0.8,0,8,16,4",
    "filter.boost_chromaticity": "fx_boost_chroma 75,3",
    "filter.channel_processing": "fx_channel_processing 5,12,0,0,0,0,100,256,0,0,2,2,12",
    "filter.color_temperature": "fx_tk_colortemp 8,0",
    "filter.desaturate_norm": "fx_gcd_norm_eq 0.6,0.4,2.2",
    "filter.graduated_color_accent": "fx_gca 255,152,0,2.5,0,0",
    "filter.hsl_adjustment": "gcd_hsl 1.1,0.05,0,35,0.35,0,1,1.05",
    "filter.hsv_equalizer": "fx_hsv_equalizer 0,25,55,8,0.2,0.04,200,70,-6,0.14,-0.03,330,45,4,0.1,0.06",
    "filter.lomo": "fx_lomo 20",
    "filter.saturation_equalizer": "Saturation_EQ_p -12,-6,4,8,12,8,4,-2,-8,8,14,6,-4,-8,4,10,12,8,0",
    "filter.selective_desaturation": "fx_selective_desaturation 229,57,53,1,5,4,0",
    "filter.specific_saturation": "gcd_hio_levels 1.2,0.9,35,0.08,1,1.1,1,0",
    "filter.target_color_spot": "fx_gcd_color_target 50,50,255,167,38,1",
    "filter.temperature_balance": "gcd_temp_balance 0.2,0.1,1.15,1",
    "filter.unpurple": "unpurple 1,0,0,0.33,5,0,0",
}


def _parse(cmd):
    cmd = re.sub(r"^srand \d+ ", "", cmd)
    name, _, args = cmd.partition(" ")
    return name, [float(a) for a in args.split(",")] if args else []


@pytest.mark.parametrize("op_id", sorted(EXPECTED_GMIC))
def test_gmic_command_strings(op_id, monkeypatch):
    from lazyroom.ops.itb import color

    seen = []
    monkeypatch.setattr(color, "run_gmic", lambda img, cmd, **kw: (seen.append(cmd), img)[1])
    spec = get_op(op_id)
    spec.fn(synthetic_image(32, 24), RenderContext(seed=1), **spec.validate({}))
    assert _parse(seen[0]) == _parse(EXPECTED_GMIC[op_id])


@pytest.mark.gmic
def test_bw_film_is_gray_without_toning():
    out = _run("filter.black_and_white_film", synthetic_image(64, 48), saturation=0.0)
    assert np.abs(out[..., 0] - out[..., 1]).max() < 2 / 255 and np.abs(out[..., 1] - out[..., 2]).max() < 2 / 255


# ---------------------------------------------------------------------------
# Matrices / channel filters
# ---------------------------------------------------------------------------

def test_color_matrices_identity_and_convention():
    img = synthetic_image(40, 30)
    assert np.allclose(_run("filter.color_matrix_3x3", img), img, atol=1e-6)
    assert np.allclose(_run("filter.color_matrix_4x4", img), img, atol=1e-6)
    swap = [0, 1, 0, 1, 0, 0, 0, 0, 1]  # out_r = g, out_g = r
    out = _run("filter.color_matrix_3x3", img, matrix=swap)
    assert np.allclose(out[..., 0], img[..., 1], atol=1e-6) and np.allclose(out[..., 1], img[..., 0], atol=1e-6)
    # GPUImage sepia matrix: rows are output channels (GL column-major upload) -> warm, R > G > B on white
    sepia = [0.3588, 0.7044, 0.1368, 0, 0.2990, 0.5870, 0.1140, 0, 0.2392, 0.4696, 0.0940, 0, 0, 0, 0, 1]
    w = _run("filter.color_matrix_4x4", np.full((4, 4, 3), 0.7, np.float32), matrix=sepia)[0, 0]
    assert w[0] > w[1] > w[2]


def test_color_matrix4x4_alpha_row():
    img = synthetic_image(20, 16, alpha=True)
    m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5]
    out = _run("filter.color_matrix_4x4", img, matrix=m)
    assert np.allclose(out[..., 3], img[..., 3] * 0.5, atol=1e-6)


def test_channel_mix_identity_and_mix():
    img = synthetic_image(40, 30)
    out = _run("filter.channel_mix", img, into_red=0, into_green=0, into_blue=0)
    assert np.allclose(out, img, atol=1e-6)
    out = _run("filter.channel_mix", img, into_red=255, blue_green=255, into_green=0, into_blue=0)
    assert np.allclose(out[..., 0], img[..., 1], atol=1e-6)


def test_rgb_filter_and_mask():
    img = synthetic_image(40, 30)
    out = _run("filter.rgb", img, color="#FF0000")
    assert np.allclose(out[..., 0], img[..., 0]) and out[..., 1:].max() == 0
    assert np.allclose(_run("filter.color_mask", img, color="#FFFFFF"), img)
    out = _run("filter.color_mask", img)  # cyan mask kills red
    assert out[..., 0].max() == 0 and np.allclose(out[..., 1:], img[..., 1:])


def test_opacity_alpha():
    img = synthetic_image(40, 30, alpha=True)
    out = _run("filter.opacity", img, opacity=0.25)
    assert np.allclose(out[..., :3], img[..., :3]) and np.allclose(out[..., 3], img[..., 3] * 0.25, atol=1e-6)
    out = _run("filter.opacity", synthetic_image(40, 30), opacity=0.5)
    assert out.shape[2] == 4 and np.allclose(out[..., 3], 0.5)


def test_posterize_levels():
    out = _run("filter.posterize", synthetic_image(40, 30), levels=4)
    assert np.allclose(np.round(out * 4), out * 4, atol=1e-5)
    assert len(np.unique(np.round(out * 4))) <= 5


def test_solarize():
    img = synthetic_image(40, 30)
    assert np.allclose(_run("filter.solarize", img, threshold=1.0), 1 - img, atol=1e-6)
    lum = img @ np.array([0.2125, 0.7154, 0.0721], np.float32)
    out = _run("filter.solarize", img, threshold=0.5)
    bright = lum > 0.5
    assert np.allclose(out[bright], img[bright]) and np.allclose(out[~bright], 1 - img[~bright], atol=1e-6)


def test_false_color_endpoints():
    img = np.zeros((2, 2, 3), np.float32)
    img[1] = 1.0
    out = _run("filter.false_color", img, first_color="#102030", second_color="#F0E0D0")
    assert np.allclose(out[0, 0], np.array([16, 32, 48]) / 255, atol=1e-6)
    assert np.allclose(out[1, 0], np.array([240, 224, 208]) / 255, atol=2e-3)


def test_hue_rotation():
    img = synthetic_image(40, 30)
    assert np.abs(_run("filter.hue", img, hue=0) - img).max() < 0.01  # YIQ round trip
    gray = np.full((4, 4, 3), 0.4, np.float32)
    assert np.abs(_run("filter.hue", gray, hue=120) - gray).max() < 1e-3
    red = np.zeros((2, 2, 3), np.float32)
    red[..., 0] = 0.8
    assert not np.allclose(_run("filter.hue", red, hue=120), red, atol=0.05)


def test_color_balance_zero_is_identity_and_preserves_lightness():
    img = synthetic_image(40, 30)
    assert np.abs(_run("filter.color_balance", img) - img).max() < 1e-4
    out = _run("filter.color_balance", img, values=[0.3, 0, -0.2, 0, 0.2, 0, 0, 0, 0.4])
    l_in = (img.max(-1) + img.min(-1)) / 2
    l_out = (out.max(-1) + out.min(-1)) / 2
    assert np.abs(l_in - l_out).max() < 1e-4 and np.abs(out - img).max() > 0.02


def test_tone_curves():
    img = synthetic_image(40, 30)
    assert np.array_equal(_run("filter.tone_curves", img), img)
    out = _run("filter.tone_curves", img, red=[[0, 1], [1, 0]])
    assert np.abs(out[..., 0] - (1 - img[..., 0])).max() < 2 / 255
    assert np.abs(out[..., 1:] - img[..., 1:]).max() < 2 / 255
    ramp = np.repeat(np.linspace(0, 1, 256, dtype=np.float32)[None, :, None], 3, axis=2)
    out = _run("filter.tone_curves", ramp, rgb=[[0, 0], [0.5, 0.75], [1, 1]])
    assert np.all(np.diff(out[0, :, 0]) >= -1e-6) and abs(out[0, 128, 0] - 0.75) < 0.02


# ---------------------------------------------------------------------------
# Grayscale / tint family
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_id,kw", [("filter.grayscale", {}), ("filter.saturation", {"strength": 0.0,
                                                                                     "enable_tonemapping": False})])
def test_gray_outputs(op_id, kw):
    out = _run(op_id, synthetic_image(40, 30), **kw)
    assert np.abs(out[..., 0] - out[..., 1]).max() < 1e-4 and np.abs(out[..., 1] - out[..., 2]).max() < 1e-4


def test_grayscale_is_linear_luma():
    out = _run("filter.grayscale", np.full((2, 2, 3), 0.5, np.float32))
    assert abs(out[0, 0, 0] - ((0.5 + 0.055) / 1.055) ** 2.4) < 1e-4  # aire writes linear values back


def test_saturation_and_vibrance_neutral():
    img = synthetic_image(40, 30)
    assert np.abs(_run("filter.saturation", img, strength=1.0, enable_tonemapping=False) - img).max() < 2e-3
    assert np.allclose(_run("filter.vibrance", img, vibrance=0.0), img)
    gray = np.full((4, 4, 3), 0.3, np.float32)
    assert np.allclose(_run("filter.vibrance", gray, vibrance=5.0), gray)


def test_monochrome():
    img = synthetic_image(40, 30)
    assert np.abs(_run("filter.monochrome", img, strength=0.0) - img).max() < 3e-3
    gray = np.full((2, 2, 3), 0.5, np.float32)
    out = _run("filter.monochrome", gray, color="#FF0000")
    assert out[0, 0, 0] > out[0, 0, 1] and np.isclose(out[0, 0, 1], out[0, 0, 2])


def test_tri_tone_endpoints():
    img = np.zeros((1, 3, 3), np.float32)
    img[0, 1] = 127 / 255
    img[0, 2] = 1.0
    out = _run("filter.tri_tone", img, shadows_color="#000080", midtones_color="#008000", highlights_color="#800000")
    assert np.allclose(out[0, 0], [0, 0, 128 / 255], atol=1e-3)
    assert np.allclose(out[0, 1], [0, 128 / 255, 0], atol=1e-3)
    assert np.allclose(out[0, 2], [128 / 255, 0, 0], atol=1e-3)


def test_gradient_map():
    img = np.zeros((1, 2, 3), np.float32)
    img[0, 1] = 1.0
    out = _run("filter.gradient_map", img, gradient=[[0, "#FF0000"], [1, "#0000FF"]])
    assert np.allclose(out[0, 0], [1, 0, 0]) and np.allclose(out[0, 1], [0, 0, 1])
    out = _run("filter.gradient_map", img)
    assert np.allclose(out[0, 0], np.array([0x9E, 0x01, 0x42]) / 255, atol=1e-6)


def test_color_poster_uses_palette():
    from lazyroom.ops.itb import _color_impl as ci

    img = synthetic_image(40, 30)
    out = _run("filter.color_poster", img, strength=0.5, color="#4DFFE4")
    h, s, _ = ci.sk_rgb_to_hsv(0x4D, 0xFF, 0xE4)
    pal = {ci.sk_hsv_to_rgb(h, s, i / 100) for i in range(2, 99, 25)}
    got = {tuple(int(round(v)) for v in c) for c in np.unique((out * 255).reshape(-1, 3), axis=0)}
    assert got <= pal


def test_color_overlay():
    img = synthetic_image(40, 30)
    assert np.allclose(_run("filter.color_overlay", img, color="#FF000000"), img, atol=1e-6)
    out = _run("filter.color_overlay", img, color="#00FF00")
    assert np.allclose(out, [0, 1, 0])
    rgba = synthetic_image(40, 30, alpha=True)
    out = _run("filter.color_overlay", rgba, color=[1, 0, 0, 0.5])
    assert np.allclose(out[..., 3], 0.5 + rgba[..., 3] * 0.5, atol=1e-6)


def test_replace_and_remove_color():
    img = synthetic_image(40, 30)
    img[:5, :5] = 0.0
    out = _run("filter.replace_color", img, tolerance=0.0, color_to_replace="#000000", target_color="#FFFFFF")
    assert np.allclose(out[:5, :5], 1.0)
    out = _run("filter.remove_color", img, tolerance=0.0)
    assert out.shape[2] == 4 and np.allclose(out[:5, :5, 3], 0) and out[10:, 10:, 3].min() == 1.0


def test_trickle_channel_order(tmp_path):
    """Trickle's byte-level filters must not swap R/B (Android RGBA bytes vs ARGB colour ints)."""
    from lazyroom.core.io import save_image

    img = np.zeros((4, 4, 3), np.float32)
    img[:2] = [1.0, 0.0, 0.0]
    out = _run("filter.replace_color", img, color_to_replace="#FF0000", target_color="#0000FF")
    assert np.allclose(out[:2], [0, 0, 1]) and np.allclose(out[2:], 0)
    assert np.allclose(_run("filter.color_overlay", img, color="#FF0000"), [1, 0, 0])
    out = _run("filter.color_poster", np.full((2, 2, 3), [0.9, 0.1, 0.0], np.float32), strength=0.0, color="#FF0000")
    assert out[0, 0, 0] > 0.8 and out[0, 0, 2] == 0
    out = _run("filter.tri_tone", np.ones((2, 2, 3), np.float32), highlights_color="#FF8000")
    assert np.allclose(out, [1, 128 / 255, 0], atol=1e-3)
    ref = tmp_path / "red.png"
    save_image(np.dstack([np.linspace(0.6, 1, 64 * 48).reshape(48, 64), np.full((48, 64), 0.1, np.float32),
                          np.full((48, 64), 0.05, np.float32)]).astype(np.float32), ref)
    out = _run("filter.palette_transfer", synthetic_image(48, 36), target_image=str(ref))
    assert out[..., 0].mean() > out[..., 2].mean() + 0.3


@pytest.mark.parametrize("op_id", ["filter.hue_shift_bands", "filter.hue_shift_fan", "filter.hue_shift_tunnel"])
def test_hue_shift_family(op_id):
    img = synthetic_image(40, 30)
    assert np.abs(_run(op_id, img, intensity=0.0) - img).max() < 1e-4
    red = np.zeros((4, 4, 3), np.float32)
    red[..., 0] = 1.0
    out = _run(op_id, red, intensity=0.0, hue=0.5)
    assert np.allclose(out, [0, 1, 1], atol=1e-5)
    assert np.abs(_run(op_id, img) - img).max() > 0.05


def test_neon_flat_is_black():
    flat = np.full((20, 20, 3), 0.6, np.float32)
    assert np.abs(_run("filter.neon", flat)).max() < 1e-5
    out = _run("filter.neon", synthetic_image(40, 30), color="#00FF00")
    assert out[..., 0].max() == 0 and out[..., 1].max() > 0.2


@pytest.mark.parametrize("op_id,kw", [("filter.palette_transfer", {}),
                                      ("filter.palette_transfer_variant", {"color_space": "oklab"}),
                                      ("filter.palette_transfer_variant", {"color_space": "lab"}),
                                      ("filter.palette_transfer_variant", {"color_space": "luv"}),
                                      ("filter.palette_transfer_variant", {"color_space": "lalphabeta"})])
def test_palette_transfer(op_id, kw, ref_path):
    from lazyroom.core.io import load_image

    img = synthetic_image(48, 36)
    same = _run(op_id, img, strength=0.0, target_image=ref_path, **kw)
    assert np.abs(same - img).max() < 0.01
    out = _run(op_id, img, target_image=ref_path, **kw)
    ref, _ = load_image(ref_path)
    target = ref[..., :3].mean((0, 1))
    # mean colour moves towards the reference
    assert np.abs(out.mean((0, 1)) - target).sum() < np.abs(img.mean((0, 1)) - target).sum()


def test_procedural_field_aspect():
    from lazyroom.ops.itb import _color_impl as ci

    x, y = ci.procedural_field(10, 20, 0, 0, 1, 0)
    assert math.isclose(float(x.max() - x.min()), (19 / 20) * 2, rel_tol=1e-5)
    assert math.isclose(float(y.max() - y.min()), 9 / 10, rel_tol=1e-5)
