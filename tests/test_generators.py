"""noise_image (FastNoiseLite / ImageToolbox Noise Generation) and gradient_image (Gradient Maker + mesh)."""
import json

import numpy as np
import pytest

from lazyroom.core.errors import ErrorCode, LazyRoomError
from lazyroom.tools import _fastnoise as fnl
from lazyroom.tools import _gradient_impl as gi

# Reference samples produced by the upstream C++ (ImageToolboxLibs FastNoiseLite.h + fast-noise.cpp colour
# mapping, g++ x86-64, no fast-math): config, mode (0 = 2D, 1 = 3D, 2 = warp 2D, 3 = warp 3D), z,
# [(x, y, noise, 8-bit grey)].  Config order: seed, frequency, noise, rotation, fractal, octaves, lacunarity,
# gain, weighted, ping-pong, cellular distance, cellular return, jitter, warp type, warp amplitude.
UPSTREAM = {
    "itb_default": ((1337, 0.01, 0, 0, 0, 3, 2.0, 0.5, 0.0, 2.0, 1, 1, 1.0, 0, 1.0), 0, 0.0,
                    [(0, 0, 0.0, 119), (5, 3, 0.3293265700340271, 162), (15, 7, 0.7320005893707275, 217),
                     (10, 1, 0.39570382237434387, 170)]),
    "os2s_fbm_weighted": ((42, 0.05, 1, 0, 1, 4, 2.0, 0.5, 0.7, 2.0, 1, 1, 1.0, 0, 1.0), 0, 0.0,
                          [(0, 0, -1.3998557101569555e-29, 119), (5, 3, -0.02912767231464386, 115),
                           (15, 7, 0.0837908610701561, 130), (10, 1, -0.16123858094215393, 99)]),
    "cellular_euclid_div": ((7, 0.08, 2, 0, 0, 3, 2.0, 0.5, 0.0, 2.0, 0, 6, 0.8, 0, 1.0), 0, 0.0,
                            [(0, 0, -0.5218348503112793, 57), (5, 3, -0.17652499675750732, 97),
                             (15, 7, -0.43918734788894653, 66), (10, 1, -0.12695205211639404, 103)]),
    "perlin_ridged": ((-5, 0.04, 3, 0, 2, 5, 2.5, 0.6, 0.2, 2.0, 1, 1, 1.0, 0, 1.0), 0, 0.0,
                      [(0, 0, 1.0, 255), (5, 3, 0.4242742359638214, 174), (15, 7, 0.23298998177051544, 149),
                       (10, 1, 0.5167087316513062, 187)]),
    "value_cubic_pingpong": ((99, 0.03, 4, 0, 3, 3, 2.0, 0.5, 0.0, 3.5, 1, 1, 1.0, 0, 1.0), 0, 0.0,
                             [(0, 0, -0.3852151036262512, 72), (5, 3, -0.19710059463977814, 95),
                              (15, 7, -0.43830791115760803, 66), (10, 1, -0.4008105397224426, 71)]),
    "value_3d_xz": ((1, 0.06, 5, 2, 1, 2, 2.0, 0.5, 0.0, 2.0, 1, 1, 1.0, 0, 1.0), 1, 3.5,
                    [(0, 0, 0.2847261428833008, 156), (5, 3, -0.06349897384643555, 111),
                     (15, 7, -0.04767773300409317, 113), (10, 1, -0.02814638987183571, 115)]),
    "os2_3d_xy_warp_independent": ((2024, 0.02, 0, 1, 5, 3, 2.0, 0.5, 0.0, 2.0, 1, 1, 1.0, 1, 40.0), 3, 7.0,
                                   [(0, 0, 0.3401670753955841, 163), (5, 3, 0.40058571100234985, 171),
                                    (15, 7, -0.11246880143880844, 105), (10, 1, 0.5167543888092041, 187)]),
    "perlin_warp_progressive_grid": ((11, 0.02, 3, 0, 4, 3, 2.0, 0.5, 0.0, 2.0, 1, 1, 1.0, 2, 25.0), 2, 0.0,
                                     [(0, 0, 0.48683711886405945, 183), (5, 3, 0.4458385705947876, 177),
                                      (15, 7, 0.10500913113355637, 132), (10, 1, 0.4834039807319641, 182)]),
}


def _grid(cfg, mode, z, w=16, h=8):
    seed, freq, nt, rt, ft, oc, lac, gain, ws, pp, cd, cr, jit, wt, wamp = cfg
    ic, fc = fnl.make_config(seed, freq, fnl.NOISE_TYPES[nt], fnl.ROTATION_TYPES[rt], fnl.FRACTAL_TYPES[ft], oc,
                             lac, gain, ws, pp, fnl.CELLULAR_DISTANCE[cd], fnl.CELLULAR_RETURN[cr], jit,
                             fnl.DOMAIN_WARP_TYPES[wt], wamp)
    n = np.empty((h, w), np.float32)
    fnl.noise_grid(n, ic, fc, mode in (1, 3), np.float32(z), mode in (2, 3))
    return n


@pytest.mark.parametrize("name", list(UPSTREAM))
def test_fastnoise_matches_upstream_cpp_bit_exactly(name):
    cfg, mode, z, samples = UPSTREAM[name]
    n = _grid(cfg, mode, z)
    rgba = np.empty(n.shape + (4,))
    fnl.noise_to_rgba(n, np.array(fnl.color_to_lab(0, 0, 0)), np.array(fnl.color_to_lab(255, 255, 255)),
                      255.0, 255.0, rgba)
    for x, y, value, grey in samples:
        assert n[y, x] == np.float32(value), (name, x, y)
        assert tuple(rgba[y, x]) == (grey, grey, grey, 255), (name, x, y)


@pytest.mark.parametrize("noise_type", fnl.NOISE_TYPES)
@pytest.mark.parametrize("fractal_type", fnl.FRACTAL_TYPES)
def test_noise_types_and_fractals_finite_deterministic(noise_type, fractal_type):
    cfg = (1337, 0.02, fnl.NOISE_TYPES.index(noise_type), 0, fnl.FRACTAL_TYPES.index(fractal_type), 4, 2.0, 0.5,
           0.3, 2.0, 1, 1, 1.0, 0, 1.0)
    a = _grid(cfg, 0, 0.0, 48, 32)
    assert np.isfinite(a).all() and np.array_equal(a, _grid(cfg, 0, 0.0, 48, 32))
    assert a.std() > 0
    if fractal_type in ("none", "fbm", "ridged", "ping_pong"):
        assert np.abs(a).max() <= 1.0 + 1e-5


@pytest.mark.parametrize("dist", range(4))
@pytest.mark.parametrize("ret", range(7))
def test_cellular_modes(dist, ret):
    a = _grid((3, 0.05, 2, 0, 0, 3, 2.0, 0.5, 0.0, 2.0, dist, ret, 0.9, 0, 1.0), 0, 0.0, 40, 24)
    assert np.isfinite(a).all() and a.std() > 0


@pytest.mark.parametrize("rotation", range(3))
@pytest.mark.parametrize("warp", range(3))
@pytest.mark.parametrize("fractal", (0, 1, 4, 5))
def test_3d_rotation_and_domain_warp(rotation, warp, fractal):
    cfg = (9, 0.03, 0, rotation, fractal, 3, 2.0, 0.5, 0.0, 2.0, 1, 1, 1.0, warp, 30.0)
    a = _grid(cfg, 3, 2.0, 32, 24)
    b = _grid(cfg, 2, 0.0, 32, 24)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert not np.array_equal(a, _grid(cfg, 1, 2.0, 32, 24))  # warping moves the samples


def test_rotation_type_only_affects_3d():
    base = (5, 0.03, 3, 0, 1, 3, 2.0, 0.5, 0.0, 2.0, 1, 1, 1.0, 0, 1.0)
    rot = (5, 0.03, 3, 1, 1, 3, 2.0, 0.5, 0.0, 2.0, 1, 1, 1.0, 0, 1.0)
    assert np.array_equal(_grid(base, 0, 0.0), _grid(rot, 0, 0.0))
    assert not np.array_equal(_grid(base, 1, 1.0), _grid(rot, 1, 1.0))


def test_domain_warp_fractals_without_warp_are_single_noise():
    # ImageToolbox never calls DomainWarp(): GetNoise with a DomainWarp* fractal type = single noise
    plain = _grid((1, 0.02, 0, 0, 0, 3, 2.0, 0.5, 0.0, 2.0, 1, 1, 1.0, 0, 1.0), 0, 0.0)
    for ft in (4, 5):
        assert np.array_equal(plain, _grid((1, 0.02, 0, 0, ft, 3, 2.0, 0.5, 0.0, 2.0, 1, 1, 1.0, 0, 1.0), 0, 0.0))


def test_fractal_bounding_matches_upstream():
    assert fnl.fractal_bounding(3, 0.5) == np.float32(1 / 1.75)
    assert fnl.fractal_bounding(1, 0.5) == np.float32(1.0)


# ----------------------------------------------------------------------------- tools

@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("LAZYROOM_HOME", str(tmp_path / "home"))
    from lazyroom.api import LazyRoom

    return LazyRoom()


def _load(path):
    from lazyroom.core.io import load_image

    return load_image(path)[0]


def test_noise_image_tool_itb_default(api, tmp_path):
    out = tmp_path / "noise.png"
    res = api.tool_run("noise_image", {"width": 64, "height": 40, "output": str(out)})
    img = _load(out)
    assert img.shape == (40, 64, 3) and np.isfinite(img).all()
    assert np.allclose(img[..., 0], img[..., 1]) and np.allclose(img[..., 1], img[..., 2])  # grey (LAB b/w)
    assert img[0, 0, 0] == pytest.approx(119 / 255, abs=1e-6)  # upstream pixel (0, 0) of the default config
    assert -1 <= res["range"][0] <= res["range"][1] <= 1


@pytest.mark.parametrize("params", [
    {"noise_type": "cellular", "fractal_type": "ping_pong", "cellular_return_type": "distance2_sub",
     "cellular_distance_function": "hybrid", "octaves": 5, "ping_pong_strength": 4.0},
    {"noise_type": "value_cubic", "fractal_type": "ridged", "weighted_strength": 1.5, "gain": -0.7},
    {"noise_type": "perlin", "dimensions": "3d", "z": 12.5, "rotation_type_3d": "improve_xz_planes"},
    {"noise_type": "open_simplex2s", "domain_warp": True, "fractal_type": "domain_warp_independent",
     "domain_warp_type": "basic_grid", "domain_warp_amp": 80.0},
    {"color_a": "#102030", "color_b": "#FFC0A080", "frequency": -0.05, "seed": -42},
    {"kind": "fbm"}, {"kind": "white"}, {"kind": "worley", "scale": 8}, {"kind": "turbulence"},
    {"kind": "ridged"}, {"kind": "value"},
])
def test_noise_image_tool_variants(api, tmp_path, params):
    out = tmp_path / "n.png"
    api.tool_run("noise_image", {"width": 48, "height": 32, "output": str(out), **params})
    img = _load(out)
    assert img.shape[:2] == (32, 48) and np.isfinite(img).all()
    assert img.shape[2] == (4 if params.get("color_b") == "#FFC0A080" else 3)


@pytest.mark.parametrize("params", [
    {"frequency": 0.9}, {"noise_type": "simplex"}, {"octaves": 0}, {"seed": 2 ** 40}, {"domain_warp_amp": 5000},
])
def test_noise_image_invalid_params(api, tmp_path, params):
    with pytest.raises(LazyRoomError) as e:
        api.tool_run("noise_image", {"width": 8, "height": 8, "output": str(tmp_path / "x.png"), **params})
    assert e.value.code == ErrorCode.INVALID_PARAMS


# ----------------------------------------------------------------------------- gradients

STOPS = [(0.0, (1.0, 0.0, 0.0, 1.0)), (1.0, (0.0, 0.0, 1.0, 1.0))]


def test_linear_gradient_geometry():
    pos, pre = gi.prepare_stops(STOPS, sweep=False)
    img = gi.eval_gradient(gi.linear_t(100, 50, 0.0), pos, pre, "clamp")
    assert img[25, 0, 0] > 0.98 and img[25, -1, 2] > 0.98       # 0 deg: left -> right
    img = gi.eval_gradient(gi.linear_t(100, 50, 90.0), pos, pre, "clamp")
    assert img[-1, 50, 0] > 0.97 and img[0, 50, 2] > 0.97       # 90 deg: bottom -> top


def test_radial_sweep_and_tile_modes():
    pos, pre = gi.prepare_stops(STOPS, sweep=False)
    t = gi.radial_t(64, 64, (0.5, 0.5), 0.25)
    clamp = gi.eval_gradient(t, pos, pre, "clamp")
    assert clamp[32, 32, 0] > 0.9 and clamp[0, 0, 2] == 1.0  # t = 0.707 px / 8 px radius
    decal = gi.eval_gradient(t, pos, pre, "decal")
    assert decal[0, 0, 3] == 0.0 and decal[32, 32, 3] == 1.0
    rep = gi.eval_gradient(np.array([0.25, 1.25]), pos, pre, "repeated")
    assert np.allclose(rep[0], rep[1])
    mir = gi.eval_gradient(np.array([0.25, 1.75]), pos, pre, "mirror")
    assert np.allclose(mir[0], mir[1])
    sw = gi.sweep_t(64, 64, (0.5, 0.5), 90.0)
    assert sw.min() >= 0 and sw.max() < 1
    # sweep start (t = 0) is rotated by +90 deg (clockwise in screen space): straight below the centre
    assert sw[60, 32] < 0.01 or sw[60, 32] > 0.99


def test_stop_preprocessing_like_skia_and_imagetoolbox():
    stops = [(0.5, (1, 0, 0, 1)), (0.5, (0, 1, 0, 1)), (0.2, (0, 0, 1, 1))]
    pos, pre = gi.prepare_stops(stops, sweep=False)
    assert list(pos) == [0.0, 0.2, 0.5, 1.0]  # sorted, de-duplicated, implicit 0 / 1 stops
    assert np.allclose(pre[0], (0, 0, 1, 1)) and np.allclose(pre[-1], (1, 0, 0, 1))
    pos, _ = gi.prepare_stops(stops, sweep=True)  # sweep keeps duplicates (hard stop)
    assert list(pos) == [0.0, 0.2, 0.5, 0.5, 1.0]


def test_mesh_gradient_corners_and_continuity():
    pts = np.array([[[0, 0], [1, 0]], [[0, 1], [1, 1]]], float)
    cols = np.array([[[1, 0, 0, 1], [0, 1, 0, 1]], [[0, 0, 1, 1], [1, 1, 1, 1]]], float)
    img = gi.render_mesh(64, 48, pts, cols, 16)
    assert img.shape == (48, 64, 4) and (img[..., 3] == 1).all()
    # the border cubics start with zero slope, so colours change fast near the control points
    ctrl = cols.reshape(4, 4)[:, :3]
    for k, (y, x) in enumerate([(0, 0), (0, -1), (-1, 0), (-1, -1)]):
        assert np.argmin(np.linalg.norm(ctrl - img[y, x, :3], axis=1)) == k
    assert np.abs(np.diff(img, axis=1)).max() < 0.15 and np.abs(np.diff(img, axis=0)).max() < 0.15


def test_compose_lerp_endpoints_and_oklab_midpoint():
    a = np.array([[1.0, 0.0, 0.0, 1.0]])
    b = np.array([[0.0, 0.0, 1.0, 1.0]])
    out = gi.compose_lerp(a, b, np.array([0.0, 0.5, 1.0]))
    assert np.allclose(out[0], a[0]) and np.allclose(out[2], b[0])
    assert np.allclose(out[1, :3], (0.554, 0.326, 0.637), atol=0.01)  # Oklab midpoint of red / blue


def test_path_measure_straight_cubic_samples_by_t():
    pos = gi.sample_cubic_by_length(gi._cubic_x((0.0, 0.0), (1.0, 0.0), 1), [0.25, 0.5, 0.75])
    assert np.allclose([p[0] for p in pos], [0.296875, 0.5, 0.703125])  # x(t) = 1.5t(1 - t) + t^3 (single segment)


KIND_TILES = [(k, t) for k in ("linear", "radial") for t in gi.TILE_MODES] + \
    [(k, "clamp") for k in ("sweep", "angular", "diamond", "reflected", "mesh")]


@pytest.mark.parametrize("kind,tile_mode", KIND_TILES)
def test_gradient_image_tool_kinds(api, tmp_path, kind, tile_mode):
    out = tmp_path / "g.png"
    api.tool_run("gradient_image", {"width": 48, "height": 32, "kind": kind, "tile_mode": tile_mode,
                                    "radius": 0.3, "output": str(out)})
    img = _load(out)
    assert img.shape[:2] == (32, 48) and np.isfinite(img).all() and img.std() > 0


def test_gradient_image_palettes_and_mesh_options(api, tmp_path):
    out = tmp_path / "p.png"
    api.tool_run("gradient_image", {"width": 40, "height": 20, "palette": "viridis", "angle": 0, "output": str(out)})
    img = _load(out)
    assert np.allclose(img[10, 0], np.array([0x44, 0x01, 0x54]) / 255, atol=0.03)  # first Viridis colour
    mesh = [[[0, 0, "#FF0000"], [0.5, -0.1, "#00FF00"], [1, 0, "#0000FF"]],
            [[0, 1, "#FFFFFF"], [0.5, 1.1, "#000000"], [1, 1, "#FFFF00"]]]
    res = api.tool_run("gradient_image", {"width": 40, "height": 30, "kind": "mesh", "mesh": json.dumps(mesh),
                                          "resolution": 8, "output": str(tmp_path / "m.png")})
    assert res["resolution"] == 8
    res = api.tool_run("gradient_image", {"width": 20, "height": 20, "kind": "mesh", "mesh": mesh,  # parsed object
                                          "output": str(tmp_path / "mo.png")})
    assert res["resolution"] == 16
    res = api.tool_run("gradient_image", {"width": 40, "height": 30, "kind": "mesh", "grid_size": 6,
                                          "resolution": 64, "seed": 3, "output": str(tmp_path / "r.png")})
    assert res["resolution"] == 24 and res["notes"]
    a = _load(tmp_path / "r.png")
    api.tool_run("gradient_image", {"width": 40, "height": 30, "kind": "mesh", "grid_size": 6, "resolution": 24,
                                    "seed": 3, "output": str(tmp_path / "r2.png")})
    assert np.array_equal(a, _load(tmp_path / "r2.png"))  # deterministic random mesh colours
    api.tool_run("gradient_image", {"width": 40, "height": 30, "kind": "mesh", "grid_size": 3, "palette": "fire",
                                    "output": str(tmp_path / "pf.png")})


@pytest.mark.parametrize("mesh", ["{bad json", "[[[0, 0, \"#FF0000\"]]]",
                                  "[[[0,0,\"#F00\"],[1,0,\"#0F0\"]],[[0,1,\"#00F\"]]]",
                                  "[[[0,0,\"nope\"],[1,0,\"#0F0\"]],[[0,1,\"#00F\"],[1,1,\"#FFF\"]]]"])
def test_gradient_image_invalid_mesh(api, tmp_path, mesh):
    with pytest.raises(LazyRoomError) as e:
        api.tool_run("gradient_image", {"width": 8, "height": 8, "kind": "mesh", "mesh": mesh,
                                        "output": str(tmp_path / "x.png")})
    assert e.value.code == ErrorCode.INVALID_PARAMS


@pytest.mark.parametrize("params", [{"kind": "conic"}, {"tile_mode": "wrap"}, {"radius": 2}, {"palette": "nope"},
                                    {"grid_size": 9}])
def test_gradient_image_invalid_params(api, tmp_path, params):
    with pytest.raises(LazyRoomError) as e:
        api.tool_run("gradient_image", {"width": 8, "height": 8, "output": str(tmp_path / "x.png"), **params})
    assert e.value.code == ErrorCode.INVALID_PARAMS
