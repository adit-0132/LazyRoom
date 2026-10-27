"""ImageToolbox texture generation port: presets, options validation, tools and the generate.texture op."""
import json

import numpy as np
import pytest

from lazyroom.core import backends
from lazyroom.core.context import RenderContext
from lazyroom.core.errors import ErrorCode, LazyRoomError
from lazyroom.textures import PRESETS, get_preset, list_presets, preset_ids, render
from lazyroom.textures.fastnoise_lite import FastNoiseLite
from tests.conftest import synthetic_image

IDS = preset_ids()
HAS_GMIC = backends.is_available("gmic")


def _skip_unavailable(preset):
    if preset.requires and not all(backends.is_available(b) for b in preset.requires):
        pytest.skip(f"backend unavailable: {preset.requires}")


# ---------------------------------------------------------------------------
# catalogue
# ---------------------------------------------------------------------------

def test_catalogue_covers_every_imagetoolbox_texture():
    assert len(IDS) == 177 and len(set(IDS)) == 177
    by_backend = {}
    for p in PRESETS.values():
        by_backend[p.backend] = by_backend.get(p.backend, 0) + 1
        assert p.category in ("nature", "materials", "space", "patterns", "abstract")
        assert p.title and p.upstream and p.status in ("faithful", "approximate")
    assert by_backend == {"fast-noise": 122, "pattern": 28, "raymarch": 15, "jhlabs": 9, "gmic": 3}
    assert len({p.upstream for p in PRESETS.values()}) == 177


def test_defaults_inside_declared_ranges():
    for p in PRESETS.values():
        vals = p.validate({})  # raises if a default violated its own schema
        assert set(vals) == {q.name for q in p.params}


# ---------------------------------------------------------------------------
# every preset renders
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset_id", IDS)
def test_preset_renders_defaults(preset_id):
    p = get_preset(preset_id)
    _skip_unavailable(p)
    a = p.render(96, 64, seed=3)
    assert a.dtype == np.float32 and a.shape[:2] == (64, 96) and a.shape[2] in (3, 4)
    assert np.isfinite(a).all() and a.min() >= 0.0 and a.max() <= 1.0
    b = p.render(96, 64, seed=3)
    assert b.shape == a.shape and np.array_equal(a, b), "not deterministic for a fixed seed"


@pytest.mark.parametrize("preset_id", ["cloud", "voronoi_gems", "brushed_metal", "plasma"])
def test_seed_varies_random_presets(preset_id):
    assert not np.array_equal(render(preset_id, 64, 48, seed=0), render(preset_id, 64, 48, seed=1))


def test_raymarch_background_is_transparent():
    img = render("sphere_3d", 64, 64)
    assert img.shape[2] == 4 and img[0, 0, 3] == 0.0 and img[32, 32, 3] == 1.0


# ---------------------------------------------------------------------------
# faithful-port anchors (values from the compiled upstream C++)
# ---------------------------------------------------------------------------

def test_fastnoiselite_matches_upstream():
    xs, ys = np.array([12.5, 1000.3, -345.6]), np.array([-7.25, 2000.7, 78.9])
    cases = {
        (1337, 0.01, 0, 0, 3, 0.5, 1, 1): [0.097617, -0.056511, -0.251429],
        (42, 0.02, 1, 1, 5, 0.52, 1, 1): [-0.378894, 0.081886, -0.240403],
        (7, 0.05, 2, 0, 3, 0.5, 0, 4): [-0.867093, -0.702743, -0.886515],
        (-3, 0.03, 3, 2, 3, 0.5, 1, 1): [0.891104, 0.945234, 0.879002],
        (99, 0.04, 4, 3, 4, 0.5, 1, 1): [-0.102062, 0.326801, 0.257345],
        (5, 0.07, 5, 0, 3, 0.5, 1, 1): [-0.484326, 0.83328, 0.216455],
    }
    for (seed, freq, nt, ft, octv, gain, cd, cr), expected in cases.items():
        n = FastNoiseLite(seed)
        n.frequency, n.noise_type, n.fractal_type, n.octaves, n.gain = freq, nt, ft, octv, gain
        n.cellular_distance_function, n.cellular_return_type = cd, cr
        assert np.allclose(n.get_noise(xs, ys), expected, atol=2e-5), (seed, nt)


@pytest.mark.parametrize("preset_id,expected", [
    ("lava", [0xffea8032, 0xffcb290a, 0xffc4280a, 0xffcc290a]),
    ("brick", [0xffb8ad99, 0xff983b28, 0xffa3462f, 0xffa3462f]),
    ("geode", [0xff120e20, 0xff5c3a8b, 0xfff4dda6, 0xff201b28]),
    ("mandelbrot", [0xff0c1f3e, 0xff2f3d7a, 0xff0c1f3e, 0xff02030b]),
])
def test_fast_noise_pixels_match_upstream(preset_id, expected):
    img = render(preset_id, 40, 30)
    for (x, y), argb in zip(((0, 0), (13, 7), (39, 29), (20, 15)), expected):
        want = np.array([(argb >> 16) & 255, (argb >> 8) & 255, argb & 255]) / 255.0
        assert np.abs(img[y, x, :3] - want).max() <= 2.5 / 255, (preset_id, x, y)


# ---------------------------------------------------------------------------
# options validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("options", [
    {"no_such_option": 1},              # unknown option
    {"scale": 5.0},                     # out of range (0.001..0.05)
    {"scale": "big"},                   # wrong type
    {"mortar_color": "not-a-colour"},   # bad colour
    '{"scale": ',                       # invalid JSON
    "[1, 2]",                           # not an object
])
def test_invalid_options_are_invalid_params(options):
    with pytest.raises(LazyRoomError) as e:
        render("brick", 32, 32, options=options)
    assert e.value.code == ErrorCode.INVALID_PARAMS


def test_unknown_preset_and_bad_size_are_invalid_params():
    for call in (lambda: render("no_such_texture", 8, 8), lambda: render("brick", 0, 8)):
        with pytest.raises(LazyRoomError) as e:
            call()
        assert e.value.code == ErrorCode.INVALID_PARAMS


def test_choice_and_int_options_validated():
    with pytest.raises(LazyRoomError) as e:
        render("cellular", 16, 16, options={"grid_type": "pentagonal"})
    assert e.value.code == ErrorCode.INVALID_PARAMS
    with pytest.raises(LazyRoomError) as e:
        render("check", 16, 16, options={"scale_x": 2.5})
    assert e.value.code == ErrorCode.INVALID_PARAMS
    img = render("check", 16, 16, options=json.dumps({"scale_x": 4, "light_color": "#FF0000"}))
    assert img.shape == (16, 16, 3)


def test_options_change_output():
    a = render("brick", 48, 32)
    b = render("brick", 48, 32, options={"mortar_color": "#00FF00", "scale": 0.03})
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("LAZYROOM_HOME", str(tmp_path / "home"))
    from lazyroom.api import LazyRoom

    return LazyRoom()


def test_texture_tool_writes_file(api, tmp_path):
    from lazyroom.core.io import load_image

    out = tmp_path / "brick.png"
    res = api.tool_run("texture", {"preset": "brick", "width": 80, "height": 50, "seed": 2,
                                   "options": '{"scale": 0.03}', "output": str(out)})
    assert out.exists() and res["output"]["width"] == 80 and res["preset"] == "brick"
    assert res["options"]["scale"] == 0.03
    img, _ = load_image(out)
    assert img.shape[:2] == (50, 80)
    with pytest.raises(LazyRoomError) as e:  # existing file without overwrite
        api.tool_run("texture", {"preset": "brick", "width": 8, "height": 8, "output": str(out)})
    assert e.value.code == ErrorCode.OUTPUT_EXISTS


def test_texture_tool_rejects_bad_options(api, tmp_path):
    for opts in ('{"bogus": 1}', "not json"):
        with pytest.raises(LazyRoomError) as e:
            api.tool_run("texture", {"preset": "cloud", "width": 8, "height": 8, "options": opts,
                                     "output": str(tmp_path / "x.png")})
        assert e.value.code == ErrorCode.INVALID_PARAMS
    assert not (tmp_path / "x.png").exists()


def test_texture_tool_alpha_output(api, tmp_path):
    from lazyroom.core.io import load_image

    out = tmp_path / "sphere.png"
    api.tool_run("texture", {"preset": "sphere_3d", "width": 40, "height": 40, "output": str(out)})
    img, _ = load_image(out)
    assert img.shape[2] == 4 and img[0, 0, 3] == 0.0


def test_texture_presets_tool(api):
    res = api.tool_run("texture_presets", {})
    assert res["count"] == 177 and sum(res["categories"].values()) == 177
    brick = next(p for p in res["presets"] if p["id"] == "brick")
    assert brick["title"] == "Brick" and brick["category"] == "materials"
    assert brick["options"]["scale"]["min"] == 0.001  # compact schema for broad listings
    full = api.tool_run("texture_presets", {"query": "brick"})["presets"][0]
    assert "scale" in full["options_schema"]["properties"] and full["params"]
    space = api.tool_run("texture_presets", {"query": "space"})
    assert space["count"] == 12 and all(p["category"] == "space" for p in space["presets"])
    assert api.tool_run("texture_presets", {"query": "raymarch"})["count"] == 15


def test_list_presets_query_words():
    assert [p["id"] for p in list_presets("hex cube")] == ["hex_cube_pattern"]


# ---------------------------------------------------------------------------
# generate.texture operation
# ---------------------------------------------------------------------------

def _op():
    from lazyroom.core.registry import get_op

    return get_op("generate.texture")


@pytest.mark.parametrize("alpha", [False, True])
def test_op_overlays_texture(alpha):
    spec = _op()
    img = synthetic_image(96, 72, alpha=alpha)
    vals = spec.validate({"preset": "check", "opacity": 0.5, "blend": "multiply"})
    out = spec.run(img, RenderContext(seed=1), vals)
    assert out.shape[:2] == img.shape[:2] and out.shape[2] == img.shape[2] and np.isfinite(out).all()
    assert not np.allclose(out, img)
    out2 = spec.run(img, RenderContext(seed=1), vals)
    assert np.array_equal(out, out2)


def test_op_transparent_texture_keeps_image():
    spec = _op()
    img = synthetic_image(64, 64)
    out = spec.run(img, RenderContext(), spec.validate({"preset": "sphere_3d"}))
    assert np.allclose(out[0, 0], img[0, 0], atol=1e-5)  # background of the 3D sphere is transparent
    assert not np.allclose(out[32, 32], img[32, 32])


def test_op_preview_matches_export_scale():
    spec = _op()
    vals = spec.validate({"preset": "hatch_pattern", "options": '{"frequency": 4}'})
    full = spec.run(np.zeros((64, 96, 3), np.float32), RenderContext(), vals)
    half = spec.run(np.zeros((32, 48, 3), np.float32), RenderContext(scale=0.5, preview=True), vals)
    import cv2

    ref = cv2.resize(full, (48, 32), interpolation=cv2.INTER_AREA)
    assert np.abs(half - ref).mean() < 0.02


def test_op_invalid_options():
    spec = _op()
    with pytest.raises(LazyRoomError) as e:
        spec.run(synthetic_image(16, 16), RenderContext(), spec.validate({"preset": "brick", "options": '{"x": 1}'}))
    assert e.value.code == ErrorCode.INVALID_PARAMS


@pytest.mark.skipif(not HAS_GMIC, reason="G'MIC CLI not available")
def test_gmic_presets_marked():
    for pid in ("organic_fibers", "gmic_reaction_diffusion", "truchet"):
        assert get_preset(pid).requires == ["gmic"]
