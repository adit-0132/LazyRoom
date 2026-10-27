"""Fractal generator: every type renders, engine behaviour matches the Rust fractal_engine unit tests,
ImageToolbox planning/normalisation, and the `fractal` / `fractal_types` tools."""
import json
import math
from decimal import Decimal

import numpy as np
import pytest

from lazyroom.core.errors import ErrorCode, LazyRoomError
from lazyroom.fractals import _escape, _raymarch
from lazyroom.fractals.catalog import SPECS, TYPE_NAMES
from lazyroom.fractals.palettes import PALETTES, sample_colors
from lazyroom.fractals.render import (EngineParams, engine_work_plan, render_engine, render_fractal, resolve_plan,
                                        _density_multiplier)

TEST_PALETTE = [0xFF05051A, 0xFF26BCE1, 0xFFF8E16C]  # render.rs tests::palette()
BACKGROUND = np.array([0, 0, 0, 255], np.uint8)


# ----------------------------------------------------------------------------- engine-level helpers

def engine_params(type_id: int, width: int, height: int, max_iterations: int | None = None, **kw) -> EngineParams:
    """render.rs ``tests::parameters()`` + ``configure_parameters(type_id)``."""
    p = dict(center=(-0.5, 0.0, 3.0), power=2.0, bailout=4.0, jr=-0.8, ji=0.156, phr=-0.5, dist=3.5)
    cfg = {
        103: dict(center=(3.0, 3.0, 2.2)), 104: dict(center=(0.0, 0.0, 2.2)), 105: dict(center=(0.0, 0.0, 2.2)),
        106: dict(center=(0.0, 0.0, 3.0)), 107: dict(center=(0.0, 0.0, 3.0)),
        108: dict(center=(0.0, 0.0, 4.0), bailout=100.0), 109: dict(center=(0.4, 0.0, 4.444444444444445)),
        110: dict(center=(-0.1, 0.5, 4.2), jr=0.4, ji=1.0, power=0.0),
        111: dict(center=(math.pi * 0.5, math.pi * 0.5, 145.0), jr=math.pi), 112: dict(center=(2.5, 2.5, 12.0)),
        113: dict(center=(-7.2, -7.9, 720.0), jr=-15.0, ji=-19.0, power=1.0),
        114: dict(center=(16.5, 17.5, 200.0), jr=34.0, ji=1.0, power=5.0),
        115: dict(center=(-34.0, -21.0, 5500.0), jr=-55.0, ji=-1.0, power=-42.0),
        116: dict(center=(0.0, 0.0, 4.0), jr=-1.4, ji=1.6, power=1.0, phr=0.7),
        117: dict(center=(0.0, 0.0, 4.5), jr=-2.24, ji=0.43, power=-0.65, phr=-2.43),
        118: dict(center=(1.0, 0.0, 3.0), jr=0.85, ji=0.9, power=0.4, phr=7.7),
        119: dict(center=(-0.5, -0.25, 3.5), jr=0.9, ji=-0.6013, power=2.0, phr=0.5),
        120: dict(center=(0.0, 0.0, 28.0), jr=0.008, ji=0.05), 121: dict(center=(0.0, 5.0, 11.0)),
        122: dict(center=(0.0, 4.0, 12.0)), 123: dict(center=(0.5, 0.5, 1.2)), 124: dict(center=(0.5, 0.5, 1.2)),
        126: dict(center=(0.0, 1.25, 5.0)),
        131: dict(center=(0.0, 0.0, 4.0)), 132: dict(center=(0.0, 0.0, 6.0), jr=1.1, ji=0.6),
        133: dict(center=(-0.5, 0.0, 3.0)), 134: dict(center=(0.0, 0.0, 3.0), jr=0.36228, ji=-0.0777),
        135: dict(center=(0.0, 0.0, 4.0)), 136: dict(center=(0.0, 0.0, 4.0), jr=1.0, ji=0.1),
        137: dict(center=(-0.5, 0.0, 3.0)), 138: dict(center=(-0.5, 0.0, 3.0)), 139: dict(center=(0.0, 0.0, 4.0)),
        140: dict(center=(0.0, 0.0, 3.5), jr=0.1, ji=-0.1, bailout=10_000.0),
        141: dict(center=(0.0, 0.0, 4.0), jr=1.1, ji=0.6), 142: dict(center=(0.0, 0.0, 2.4), jr=-0.75, ji=-0.2),
        143: dict(center=(0.0, 0.0, 6.0), bailout=50.0), 144: dict(center=(0.0, 0.0, 6.0), bailout=50.0, jr=-0.2,
                                                                   ji=0.7),
        145: dict(center=(0.0, 0.0, 4.0), bailout=50.0), 146: dict(center=(0.0, 0.0, 4.0), bailout=50.0, jr=0.065,
                                                                   ji=0.122),
        147: dict(center=(0.0, 0.0, 4.0), bailout=100.0), 148: dict(center=(0.0, 0.0, 4.0), bailout=100.0),
        149: dict(center=(0.0, 0.0, 1.8), bailout=100.0), 150: dict(center=(0.0, 0.0, 4.0), bailout=100.0),
        1006: dict(jr=2.0, ji=1.2, dist=9.0), 1007: dict(jr=1.7, ji=1.5, dist=9.0),
        1008: dict(jr=1.3, ji=1.35, power=1.12, dist=8.0), 1009: dict(jr=1.5, ji=1.0, power=0.5, dist=5.0),
        1010: dict(jr=-0.2, ji=0.8, power=8.0, dist=4.0), 1011: dict(dist=3.5, quat=(-0.2, 0.6, 0.3, 0.0)),
        1012: dict(jr=1.5, ji=1.0, power=0.5, dist=5.0),
        1013: dict(jr=2.24, ji=0.43, power=-0.65, phr=-2.43, dist=3.0),
        1014: dict(jr=10.0, ji=28.0, power=8.0 / 3.0, dist=5.0), 1015: dict(jr=0.2, ji=0.2, power=5.7, dist=5.0),
    }
    for vid in (125, 127, 128, 129, 130):
        cfg[vid] = dict(center=(0.0, 0.0, 2.2))
    p.update(cfg.get(type_id, {}))
    if type_id >= 1001:
        p.setdefault("quat", (-0.2, 0.7, 0.0, 0.0))
        if type_id not in cfg or "center" not in cfg[type_id]:
            p["center"] = (0.0, 0.0, 2.0)
        if type_id in (1001, 1002, 1003, 1004, 1005):
            p["power"] = 8.0 if type_id == 1001 else 2.0
    if max_iterations is None:
        max_iterations = {1009: 192, 1011: 256}.get(type_id, 96)
    cx, cy, span = p["center"]
    args = dict(kind=type_id, max_iterations=max_iterations, center_x=Decimal(repr(cx)), center_y=Decimal(repr(cy)),
                span=Decimal(repr(span)), aspect=width / height, power=p["power"], bailout=p["bailout"],
                julia_real=p["jr"], julia_imaginary=p["ji"], phoenix_real=p["phr"], camera_distance=p["dist"],
                quaternion=p.get("quat", (-0.2, 0.7, 0.0, 0.0)), palette=list(TEST_PALETTE))
    args.update(kw)
    return EngineParams(**args)


def render_type(type_id: int, width: int, height: int, **kw) -> np.ndarray:
    return render_engine(engine_params(type_id, width, height, **kw), width, height)[0]


ALL_IDS = sorted(s.stable_id for s in SPECS.values())


# ----------------------------------------------------------------------------- catalogue

def test_catalogue_counts():
    assert len(SPECS) == 77 and len(ALL_IDS) == 77
    assert sum(s.itb_ui for s in SPECS.values()) == 75  # ImageToolbox UI; h_tree / hilbert_curve engine-only
    assert {s.key for s in SPECS.values() if not s.itb_ui} == {"h_tree", "hilbert_curve"}
    assert len(PALETTES) == 76


# ----------------------------------------------------------------------------- every type (ImageToolbox defaults)

@pytest.mark.parametrize("name", TYPE_NAMES)
def test_every_type_renders_deterministically(name):
    a, info = render_fractal(name, 40, 30)
    b, _ = render_fractal(name, 40, 30)
    assert a.shape == (30, 40, 4) and a.dtype == np.uint8
    assert np.isfinite(a.astype(np.float32)).all()
    assert np.array_equal(a, b)
    assert info["iterations"] >= 1


# ----------------------------------------------------------------------------- ports of render.rs unit tests

@pytest.mark.parametrize("type_id", ALL_IDS)
def test_every_kind_renders_non_uniform_pixels(type_id):
    px = render_type(type_id, 48, 48).reshape(-1, 4)
    assert (px != px[0]).any(), type_id


def test_new_types_render_distinct_and_added_escape_types_differ():
    new_ids = [i for i in ALL_IDS if 108 <= i <= 150 or 1006 <= i <= 1015]
    outs = {render_type(i, 40, 40).tobytes() for i in new_ids}
    assert len(outs) == len(new_ids)
    added = {i: render_type(i, 64, 64).astype(int) for i in range(141, 151)}
    ids = list(added)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            diff = np.abs(added[ids[a]][..., :3] - added[ids[b]][..., :3]).sum(-1)
            assert (diff > 48).sum() > 64 * 64 / 20, (ids[a], ids[b])


@pytest.mark.parametrize("type_id", [i for i in ALL_IDS if 109 <= i <= 130 or i >= 1013])
def test_density_types_are_deterministic(type_id):
    assert np.array_equal(render_type(type_id, 32, 24), render_type(type_id, 32, 24))


def _occupancy(px):
    return (px.reshape(-1, 4) != BACKGROUND).any(-1).mean()


def test_kleinian_default_has_visible_bounded_occupancy():
    assert 0.05 <= _occupancy(render_type(1009, 96, 54)) < 0.20


def test_kleinian_app_defaults_have_visible_bounded_occupancy():
    yaw, pitch = math.radians(-35.0), math.radians(20.0)
    px = render_type(1009, 96, 54, max_iterations=192, camera_yaw=yaw, camera_pitch=pitch)
    flat = px.reshape(-1, 4)
    occupied = (flat != BACKGROUND).any(-1)
    assert 0.05 <= occupied.mean() < 0.20
    assert (flat[:, :3].max(-1) >= 24).mean() >= 0.04
    floor = render_type(1009, 96, 54, max_iterations=192, camera_yaw=yaw, camera_pitch=pitch,
                        show_floor=True).reshape(-1, 4)
    retained = (occupied & (flat == floor).all(-1)).sum()
    assert retained >= occupied.sum() * 4 // 5


def test_kleinian_budgeted_preview_stays_visible():
    px = render_type(1009, 56, 96, max_iterations=16, camera_yaw=math.radians(-35.0), camera_pitch=math.radians(20.0))
    assert (px[..., :3].max(-1) >= 24).sum() >= 56 * 96 / 100


def test_ray_marched_floor_is_opt_in_and_visible():
    without = render_type(1001, 96, 54)
    with_floor = render_type(1001, 96, 54, show_floor=True)
    assert not np.array_equal(without, with_floor)
    bottom = slice(27, None)
    assert _occupancy(with_floor[bottom]) > _occupancy(without[bottom])


def test_density_work_plans_are_resolution_aware_and_exact():
    assert engine_work_plan(engine_params(109, 1920, 1080, 800), 1920, 1080) == {"work": 500_000_000,
                                                                                "samples": 622_408}
    assert engine_work_plan(engine_params(109, 640, 480, 16), 640, 480) == {"work": 5_222_400, "samples": 153_600}
    assert engine_work_plan(engine_params(110, 640, 480, 320), 640, 480) == {"work": 1_331_200, "steps": 40_960}
    lorenz = engine_params(1014, 1920, 1080, 1000, supersampling=4)
    assert engine_work_plan(lorenz, 1920, 1080) == {"work": 5_241_600, "steps": 352_000}
    assert _density_multiplier(640 * 480, 1) == 1 and _density_multiplier(640 * 480 + 1, 1) == 2
    assert _density_multiplier(4 * 640 * 480, 1) == 2 and _density_multiplier(4 * 640 * 480 + 1, 1) == 3


@pytest.mark.parametrize("type_id", list(range(1001, 1013)))
def test_ray_march_work_matches_plan_and_four_k_rejects(type_id):
    ep = engine_params(type_id, 1080, 1080, 256 if type_id == 1011 else 320)
    plan = engine_work_plan(ep, 1080, 1080)
    assert plan is not None and plan["work"] == 1080 * 1080 * (plan["steps"] + 6) * plan["distance_iterations"]
    assert plan["work"] <= 500_000_000
    assert engine_work_plan(ep, 3840, 2160) is None


def test_buddhabrot_fixed_escape_radius_collatz_configurable():
    def r(type_id, bailout, it):
        return render_type(type_id, 32, 24, max_iterations=it, bailout=bailout, aspect=4 / 3)

    assert np.array_equal(r(109, 2.0, 96), r(109, 100.0, 96))
    assert not np.array_equal(r(108, 2.0, 96), r(108, 100.0, 96))
    low = r(109, 2.0, 16).reshape(-1, 4)
    assert (low != low[0]).any()


def test_deep_zoom_at_1e_100_is_non_uniform():
    ep = engine_params(1, 32, 8, 600, center_x=Decimal("-2"), center_y=Decimal("0"), span=Decimal("4E-100"),
                       aspect=4.0)
    px, info = render_engine(ep, 32, 8)
    assert info.get("perturbation")
    flat = px.reshape(-1, 4)
    assert (flat != flat[0]).any()


def test_exact_center_changes_pixels_beyond_double_precision():
    shifted = "-1." + "9" * 100
    a = render_engine(engine_params(1, 24, 8, 600, center_x=Decimal("-2"), center_y=Decimal(0),
                                    span=Decimal("4E-100"), aspect=2.0), 24, 8)[0]
    b = render_engine(engine_params(1, 24, 8, 600, center_x=Decimal(shifted), center_y=Decimal(0),
                                    span=Decimal("4E-100"), aspect=2.0), 24, 8)[0]
    assert not np.array_equal(a, b)


def test_julia_variants_start_at_the_pixel_and_use_the_julia_constant():
    pr, pi, jr, ji = 0.25, -0.4, -0.8, 0.156
    _, _, _, _, ang = _escape.iterate_escape(106, pr, pi, 1, 2.0, 4.0, jr, ji, -0.5, 0.0, 1.0)
    vr, vi = abs(pr) ** 2 - abs(pi) ** 2 + jr, 2 * abs(pr) * abs(pi) + ji
    assert abs(ang - (math.atan2(vi, vr) / (2 * math.pi) + 1.0) % 1.0) < 1e-12
    _, _, _, _, ang = _escape.iterate_escape(107, pr, pi, 1, 2.0, 4.0, jr, ji, -0.5, 0.0, 1.0)
    pre, pim = pr * pr - pi * pi, pr * pi + pi * pr
    assert abs(ang - (math.atan2(pim + ji, abs(pre) + jr) / (2 * math.pi) + 1.0) % 1.0) < 1e-12


def test_quaternion_julia_preserves_direction():
    a = _raymarch.quaternion_julia(0.8, 0.1, 0.2, -0.2, 0.8, 0.0, 0.0, 16)
    b = _raymarch.quaternion_julia(0.1, 0.8, 0.2, -0.2, 0.8, 0.0, 0.0, 16)
    assert math.isfinite(a) and math.isfinite(b) and abs(a - b) > 1e-6


def test_lyapunov_derivative_uses_state_before_update():
    seq = np.array([False, True, True])  # "ABB"
    pr, pi = 3.2, 3.7
    actual = _escape.lyapunov_exponent(pr, pi, seq, 64)
    state = 0.5
    for i in range(32):
        rate = pi if seq[i % 3] else pr
        state = rate * state * (1 - state)
    total = 0.0
    for i in range(64):
        rate = pi if seq[i % 3] else pr
        total += math.log(max(abs(rate * (1 - 2 * state)), 1e-15))
        state = rate * state * (1 - state)
    assert abs(actual - total / 64) < 1e-12


def test_distance_estimator_reference_samples():
    # distance_estimators.rs tests
    q = _raymarch.quaternion_cubic(0.37, -0.21, 0.54, -0.2, 0.6, 0.3, 0.0, 16)
    assert abs(q - 0.021641987147598706) < 1e-12
    h1 = _raymarch.hybrid(0.37, -0.21, 0.54, 8.0, -0.2, 0.8, 0.3, 12)
    h2 = _raymarch.hybrid(-1.5, -1.5, -1.3, 8.0, -0.2, 0.8, 0.3, 12)
    assert abs(h1 - -0.016340226140528893) < 1e-12 and abs(h2 - 1.1338524354114028) < 1e-12
    vals = [_raymarch.octahedral_ifs(0.37, -0.21, 0.54, 2.0, 1.2, 10),
            _raymarch.icosahedral_ifs(0.37, -0.21, 0.54, 1.7, 1.5, 10, _raymarch.icosahedral_normals()),
            _raymarch.apollonian(0.37, -0.21, 0.54, 1.3, 1.35, 1.12, 10),
            _raymarch.kleinian(0.37, -0.21, 0.54, 1.0, 1.2, 1.5, 10), h1, q,
            _raymarch.sierpinski_gasket(0.37, -0.21, 0.54, 1.5, 1.0, 0.5, 10)]
    assert all(math.isfinite(v) for v in vals)
    assert len({round(v * 1e9) for v in vals}) == len(vals)


def test_palette_interpolation_and_sampling():
    assert _escape.interp_color(0xFF000000, 0xFFFFFFFF, 0.5) == 0xFF808080
    assert sample_colors([0xFF000000, 0xFFFFFFFF], 3) == [0xFF000000, 0xFF808080, 0xFFFFFFFF]


# ----------------------------------------------------------------------------- ImageToolbox planning

def test_adaptive_iteration_policy_and_plans():
    m = SPECS["mandelbrot"]
    assert resolve_plan(m, 320, "adaptive", Decimal(3), 640, 480, 1) == (320, 1)
    assert resolve_plan(m, 320, "adaptive", Decimal(3), 1280, 960, 1) == (387, 1)  # + ln(4) * 48
    assert resolve_plan(m, 320, "adaptive", Decimal("0.003"), 640, 480, 1) == (392, 1)  # + 3 decades * 24
    assert resolve_plan(m, 320, "fixed", Decimal("0.003"), 1280, 960, 1) == (320, 1)
    assert resolve_plan(m, 320, "scale_with_zoom", Decimal("0.003"), 1280, 960, 1) == (392, 1)
    assert resolve_plan(m, 320, "adaptive", Decimal(3), 8192, 8192, 4)[1] == 1  # supersampling budget
    bulb = SPECS["mandelbulb"]
    assert resolve_plan(bulb, 320, "adaptive", Decimal(3), 1024, 1024, 1) == (53, 1)


def test_zoom_rotation_and_options():
    base, info = render_fractal("mandelbrot", 64, 48)
    zoomed, zinfo = render_fractal("mandelbrot", 64, 48, zoom=4.0)
    assert zinfo["span"] == "0.75" and not np.array_equal(base, zoomed)
    rot, _ = render_fractal("mandelbrot", 64, 48, rotation=180.0)
    assert (np.rot90(base, 2) == rot).all(-1).mean() > 0.97
    same, _ = render_fractal("mandelbrot", 64, 48, rotation=360.0)
    assert np.array_equal(base, same)
    banded, _ = render_fractal("julia", 48, 32, options={"coloring": "banded", "julia_real": -0.4,
                                                          "palette_colors": ["#000000", "#FF0000", "#FFFF00"]})
    assert banded.shape == (32, 48, 4)
    lorenz, li = render_fractal("lorenz", 48, 32, options={"sigma": 12.0, "camera_yaw": 30}, zoom=2.0)
    assert li["camera"]["distance"] == 2.5 and lorenz.shape == (32, 48, 4)
    sponge, si = render_fractal("menger_sponge", 40, 30, options={"show_floor": False, "supersampling": 2})
    assert si["show_floor"] is False and si["supersampling"] == 2


@pytest.mark.parametrize("options,match", [
    ({"nope": 1}, "Unknown option"),
    ({"power": 99}, "outside the allowed range"),
    ({"julia_real": 0.1}, "Unknown option"),          # mandelbrot has no Julia constant
    ({"coloring": "rainbow"}, "must be one of"),
    ({"palette_colors": ["#000000"]}, "list of 2"),
    ({"span": "-1"}, "span"),
    ({"center_x": "abc"}, "decimal"),
])
def test_invalid_options_raise_invalid_params(options, match):
    with pytest.raises(LazyRoomError, match=match) as e:
        render_fractal("mandelbrot", 16, 16, options=options)
    assert e.value.code == ErrorCode.INVALID_PARAMS


def test_lyapunov_sequence_option_and_validation():
    a, _ = render_fractal("lyapunov", 32, 24, options={"sequence": "AABAB"})
    b, _ = render_fractal("lyapunov", 32, 24)
    assert not np.array_equal(a, b)
    with pytest.raises(LazyRoomError) as e:
        render_fractal("lyapunov", 8, 8, options={"sequence": "ABC"})
    assert e.value.code == ErrorCode.INVALID_PARAMS


def test_ray_marched_size_limit_matches_imagetoolbox():
    with pytest.raises(LazyRoomError) as e:
        render_fractal("mandelbulb", 4096, 4096)
    assert e.value.code == ErrorCode.INVALID_PARAMS


# ----------------------------------------------------------------------------- tools

@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("LAZYROOM_HOME", str(tmp_path / "home"))
    from lazyroom.api import LazyRoom

    return LazyRoom()


def test_fractal_tool_writes_files(api, tmp_path):
    from lazyroom.core.io import load_image

    out = tmp_path / "julia.png"
    res = api.tool_run("fractal", {"type": "julia", "width": 64, "height": 48, "palette": "fire",
                                   "options": json.dumps({"julia_real": -0.7, "julia_imaginary": 0.27}),
                                   "output": str(out)})
    assert out.exists() and res["output"]["width"] == 64 and res["type"] == "julia"
    img, _ = load_image(out)
    assert img.shape[:2] == (48, 64) and img.shape[2] == 3
    out2 = tmp_path / "bulb.png"
    res = api.tool_run("fractal", {"type": "mandelbulb", "width": 40, "height": 30, "zoom": 1.5,
                                   "options": '{"inside_color": "#00000000", "show_floor": false}',
                                   "output": str(out2)})
    img2, _ = load_image(out2)
    assert img2.shape == (30, 40, 4) and res["camera"]["distance"] == pytest.approx(3.2 / 1.5)
    # interfaces (CLI key=value, MCP) may hand over an already-parsed object instead of a JSON string
    res = api.tool_run("fractal", {"type": "lorenz", "width": 24, "height": 16, "options": {"sigma": 11.0},
                                   "output": str(tmp_path / "obj.png")})
    assert res["type"] == "lorenz"
    with pytest.raises(LazyRoomError) as e:  # existing output without overwrite
        api.tool_run("fractal", {"type": "julia", "width": 8, "height": 8, "output": str(out)})
    assert e.value.code == ErrorCode.OUTPUT_EXISTS


def test_fractal_types_tool(api):
    res = api.tool_run("fractal_types", {})
    assert res["count"] == 77 and len(res["palettes"]) == 76
    by_name = {t["name"]: t for t in res["types"]}
    assert set(by_name) == set(TYPE_NAMES)
    assert "julia_real" in by_name["julia"]["options"] and "camera_yaw" in by_name["kleinian"]["options"]
    assert by_name["lorenz"]["options"]["sigma"]["default"] == 10.0
    one = api.tool_run("fractal_types", {"type": "hopalong"})
    assert one["count"] == 1 and one["types"][0]["default_iterations"] == 1000


@pytest.mark.parametrize("params", [
    {"type": "not_a_fractal"},
    {"options": "{not json"},
    {"options": "[1, 2]"},
    {"center": [0.1]},
    {"width": 0},
    {"iterations": 99999},
    {"palette": "nope"},
])
def test_fractal_tool_invalid_params(api, tmp_path, params):
    p = {"width": 16, "height": 16, "output": str(tmp_path / "x.png"), **params}
    with pytest.raises(LazyRoomError) as e:
        api.tool_run("fractal", p)
    assert e.value.code == ErrorCode.INVALID_PARAMS
