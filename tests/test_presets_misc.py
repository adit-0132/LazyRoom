"""ImageToolbox preset catalogues: crop shapes, line styles, colour library, palette formats,
shader presets, collage layouts, export profiles (see docs/dev/port_reports/presets_misc.md)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.errors import ErrorCode, LazyRoomError
from lazyroom.core.registry import get_op
from lazyroom.ops import shapes as S
from tests.conftest import synthetic_image


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("LAZYROOM_HOME", str(tmp_path / "home"))
    from lazyroom.api import LazyRoom

    return LazyRoom()


def _run(op_id, img, **params):
    spec = get_op(op_id)
    return spec.run(img, RenderContext(), spec.validate(params))


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

CROP_SHAPES = ["rect", "rounded_corners", "cut_corners", "oval", "squircle", "octagon", "pentagon", "clover", "pill",
               "bookmark", "burger", "shuriken", "explosion", "map", "material_star", "small_material_star",
               "shield", "egg", "droplet", "arrow", "kotlin", "heart", "star"]


def test_catalogue_matches_imagetoolbox_crop_list():
    ids = list(S.SHAPES)
    assert ids[:23] == CROP_SHAPES
    assert ids[23:53] == [f"m3_{n}" for n in S.M3_SHAPES] and len(S.M3_SHAPES) == 30
    assert ids[53] == "image_mask"
    assert set(ids[54:]) == {"heart_cubic", "simple_heart"}
    spec = get_op("geometry.crop_to_shape")
    assert spec.geometry and spec.alpha == "full"
    assert list(next(p for p in spec.params if p.name == "shape").choices) == ids


@pytest.mark.parametrize("shape", list(S.SHAPES))
@pytest.mark.parametrize("hw", [(1, 1), (2, 7), (37, 23), (96, 160)])
def test_every_shape_renders(shape, hw):
    h, w = hw
    m = S.SHAPES[shape](h, w)
    assert m.shape == (h, w) and m.dtype == np.float32
    assert np.isfinite(m).all() and m.min() >= 0 and m.max() <= 1
    if min(h, w) >= 23 and shape not in ("rect", "image_mask"):
        frac = m.mean()
        assert 0.15 < frac < 0.999, (shape, frac)
        # anti-aliased: some partial coverage on the outline
        assert ((m > 0.02) & (m < 0.98)).any()


@pytest.mark.parametrize("shape", list(S.SHAPES))
@pytest.mark.parametrize("alpha", [False, True])
def test_crop_to_shape_op_rgb_rgba(shape, alpha, tmp_path):
    img = synthetic_image(64, 48, alpha=alpha)
    params = {"shape": shape}
    if shape == "image_mask":
        from lazyroom.core.io import save_image

        mk = np.zeros((10, 10, 4), np.float32)
        mk[2:8, 2:8] = 1
        save_image(mk, tmp_path / "mask.png")
        params["mask_image"] = str(tmp_path / "mask.png")
    out = _run("geometry.crop_to_shape", img, **params)
    assert out.shape == (48, 64, 4) and np.isfinite(out).all()
    src_a = img[..., 3] if alpha else np.ones(img.shape[:2], np.float32)
    assert (out[..., 3] <= src_a + 1e-6).all()
    assert np.allclose(out[..., :3], img[..., :3])  # colours untouched, only alpha masked
    if shape != "image_mask":
        assert np.allclose(out[..., 3], src_a * S.render_shape(shape, 48, 64))


def test_crop_to_shape_extremes_and_aspect():
    img = synthetic_image(120, 80)
    for cr in (0, 50):
        for shape in ("rounded_corners", "cut_corners"):
            out = _run("geometry.crop_to_shape", img, shape=shape, corner_radius=cr)
            assert out.shape == (80, 120, 4)
            if cr == 0:
                assert out[..., 3].min() == 1.0
    out = _run("geometry.crop_to_shape", img, shape="oval", aspect_ratio="1:1")
    assert out.shape == (80, 80, 4)
    out = _run("geometry.crop_to_shape", img, shape="oval", aspect_ratio="9:16")
    assert out.shape[:2] == (80, 45)
    for rot in (-180, 37.5, 180):
        out = _run("geometry.crop_to_shape", img, shape="kotlin", rotation=rot)
        assert out.shape == (80, 120, 4) and np.isfinite(out).all()


def test_shape_areas_are_exact():
    m = S.SHAPES["oval"](301, 517)
    assert m.sum() == pytest.approx(math.pi * 517 * 301 / 4, rel=2e-3)
    m = S.render_shape("rounded_corners", 200, 300, corner_radius=50)  # radius = 100 px
    assert m.sum() == pytest.approx(300 * 200 - (4 - math.pi) * 100 ** 2, rel=2e-3)
    m = S.render_shape("cut_corners", 100, 200, corner_radius=30)
    assert m.sum() == pytest.approx(200 * 100 - 2 * 30 * 30, rel=1e-6)
    m = S.SHAPES["kotlin"](1000, 1000)  # notch triangle (0,0)-(1000,1000)-(473.5,500) removed from the square
    assert m.sum() == pytest.approx(1000 * 1000 - 0.5 * 1000 * (1000 - 473.5), rel=1e-4)


def test_rotation_quarter_turn_matches_rot90():
    a = S.render_shape("bookmark", 64, 64)
    b = S.render_shape("bookmark", 64, 64, rotation=90)
    assert np.abs(np.rot90(a, -1) - b).mean() < 0.01  # +90 deg = clockwise in image coordinates


def test_rotation_keeps_shape_inside():
    m = S.render_shape("rect", 50, 100, rotation=45)
    assert m[0, 0] == 0 and m[-1, -1] == 0 and m[25, 50] == 1


def test_material_shapes_are_normalized_and_centred():
    for name in S.M3_SHAPES:
        poly = S._m3_cached(name)
        l, t, r, b = poly.bounds()
        assert max(r - l, b - t) == pytest.approx(1.0, abs=1e-9), name
        assert (l + r) / 2 == pytest.approx(0.5, abs=1e-9) and (t + b) / 2 == pytest.approx(0.5, abs=1e-9)
        c = np.array(poly.cubics)
        assert np.allclose(c[:, 6:8], np.roll(c[:, 0:2], -1, axis=0), atol=1e-4), name  # contiguous


def test_rounded_polygon_circle_matches_circle():
    # RoundedPolygon.circle: 8 fully rounded corners approximating the unit circle
    poly = S._rp_circle(8)
    pts = S.flatten(poly.contours(), tol=1e-4)[0]
    r = np.hypot(pts[:, 0], pts[:, 1])
    assert r.min() > 0.999 and r.max() < 1.001


def test_svg_path_parser():
    cs = S.parse_svg_path("M10 0h5v5H10z m1 1 l1 0 0 1Z")
    assert len(cs) == 2 and cs[0][0].shape == (4, 8) and cs[0][1].all()
    # compact numbers like ImageToolbox's Shuriken path: "25.4025.2998" = 25.4025, 0.2998
    cs = S.parse_svg_path("M0 0C1 2 3 4 25.4025.2998L-1e-1 2Z")
    assert np.allclose(cs[0][0][0], [0, 0, 1, 2, 3, 4, 25.4025, 0.2998])
    assert cs[0][0][1][6] == pytest.approx(-0.1)


def test_image_mask_uses_alpha(tmp_path):
    from lazyroom.core.io import save_image

    mk = np.zeros((20, 20, 4), np.float32)
    mk[:, :10] = 1  # left half opaque
    save_image(mk, tmp_path / "m.png")
    out = _run("geometry.crop_to_shape", synthetic_image(40, 20), shape="image_mask", mask_image=str(tmp_path / "m.png"))
    assert out[10, 5, 3] == pytest.approx(1) and out[10, 35, 3] == pytest.approx(0)
    with pytest.raises(LazyRoomError):
        _run("geometry.crop_to_shape", synthetic_image(40, 20), shape="image_mask")


# ---------------------------------------------------------------------------
# Line styles
# ---------------------------------------------------------------------------

def _line_cov(style, **kw):
    img = np.zeros((100, 500, 3), np.float32)
    out = _run("draw.styled_line", img, points=[[0.1, 0.5], [0.9, 0.5]], line_style=style, color="#FFFFFF", **kw)
    return out[..., 0]


@pytest.mark.parametrize("style", list(S.LINE_STYLES))
@pytest.mark.parametrize("alpha", [False, True])
def test_line_styles_render(style, alpha):
    img = synthetic_image(80, 60, alpha=alpha)
    out = _run("draw.styled_line", img, points=[[0.1, 0.2], [0.5, 0.8], [0.9, 0.3]], line_style=style,
               color="#FF000080", closed=True)
    assert out.shape == img.shape and np.isfinite(out).all()
    assert np.abs(out - img).max() > 0.1


def test_line_style_geometry():
    sw = 20 * 100 / 500  # 20 pt on a 100 px short side = 4 px
    plain = _line_cov("none")
    assert plain[:, 250].sum() == pytest.approx(sw, abs=0.15)
    assert plain[50, 50 - 3] == 0 and plain[50, 50 - 1] > 0.9  # round cap reaches 2 px before x = 50
    row = _line_cov("dashed", dash_size=50, gap_size=50)[50, 50:450]  # 10 px dashes, 10 + 4 px gaps
    on = row > 0.5
    starts = np.flatnonzero(on[1:] & ~on[:-1]) + 1
    assert np.allclose(np.diff(starts), 24, atol=1)  # period = dash + gap + stroke width
    dd = _line_cov("dot_dashed")[50, 30:470] > 0.5
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, dd.astype(int), 0])))[::2]
    assert set(runs[:-1].tolist()) <= {20, 21, 5, 6}  # 4w dash (+caps) and w/4 dot (+caps); last may be cut
    zz = _line_cov("zigzag")
    assert 0 < zz.sum() < plain.sum()
    st = _line_cov("stamped", stamp_shape="oval", stamp_spacing=0)
    assert st.sum() == pytest.approx(math.pi * (sw / 2) ** 2 * 100, rel=0.08)  # 100 circles of diameter w


def test_stamped_shapes_cover_icon_catalogue():
    assert len(S.STAMP_SHAPES) == 51  # IconShape.entriesNoRandom
    for k in S.STAMP_SHAPES:
        m = S.line_style_mask(np.array([[5.0, 20.0], [95.0, 20.0]]), 40, 100, "stamped", stamp_shape=k)
        assert m.max() > 0, k


# ---------------------------------------------------------------------------
# Colour library
# ---------------------------------------------------------------------------

def test_color_library_lookups(api):
    res = api.tool_run("color_library", {})
    assert res["total_colors"] > 30000 and "MIT" in json.dumps(res["licenses"])
    res = api.tool_run("color_library", {"color": "#0000FF"})
    assert res["exact"] and res["nearest"][0]["name"] == "Blue" and res["nearest"][0]["distance"] == 0
    res = api.tool_run("color_library", {"color": "#0000FE", "limit": 3})
    assert not res["exact"] and res["nearest"][0]["distance"] > 0 and len(res["nearest"]) == 3
    res = api.tool_run("color_library", {"query": "navy blue"})
    assert res["colors"][0]["name"] == "Navy Blue" and res["colors"][0]["hex"] == "#000080"
    assert all("navy blue" in c["name"].lower() or c["name"].lower() in "navy blue" for c in res["colors"])
    res = api.tool_run("color_library", {"query": "#FF000"})
    assert res["matches"] >= 1 and all("ff000" in c["hex"].lower() for c in res["colors"])
    with pytest.raises(LazyRoomError):
        api.tool_run("color_library", {"color": "not-a-colour"})


def test_color_library_data_is_verified_subset():
    from lazyroom.core.paths import package_data_dir

    data = json.loads((package_data_dir() / "color_library.json").read_text())
    c = data["counts"]
    assert c["total"] == len(data["colors"]) == c["from_meodai_color_names"] + c["from_ntc_js"] + c["css_named_colors"]
    assert all(k.startswith("#") and len(k) == 7 for k in data["colors"])


# ---------------------------------------------------------------------------
# Shader presets
# ---------------------------------------------------------------------------

def test_bundled_shader_presets_are_valid():
    from lazyroom.core.paths import package_data_dir
    from lazyroom.ops.itb._effects_impl import parse_shader

    files = sorted((package_data_dir() / "shaders").glob("*.itshader"))
    assert len(files) == 5
    types = set()
    for p in files:
        preset = json.loads(p.read_text())
        assert set(preset) == {"version", "name", "params", "shader"} and preset["version"] == 1
        _, params, _ = parse_shader(p.read_text(), "")
        types |= {q["type"] for q in params}
    assert types == {"float", "int", "bool", "color", "vec2"}


def test_shader_presets_tool(api, tmp_path):
    res = api.tool_run("shader_presets", {})
    assert {p["name"] for p in res["presets"]} == {"Vignette", "Duotone", "Posterize", "RGB Split", "Threshold"}
    show = api.tool_run("shader_presets", {"action": "show", "name": "duotone"})
    assert show["preset"]["name"] == "Duotone" and json.loads(show["filter_shader_params"]["values"])["uIntensity"] == 1.0
    out = api.tool_run("shader_presets", {"action": "export", "name": "Posterize", "path": str(tmp_path / "p.itshader")})
    assert json.loads((tmp_path / "p.itshader").read_text())["name"] == "Posterize" and out["output"]
    custom = json.loads((tmp_path / "p.itshader").read_text())
    custom["name"] = "My/Posterize"
    (tmp_path / "c.itshader").write_text(json.dumps(custom))
    imp = api.tool_run("shader_presets", {"action": "import", "path": str(tmp_path / "c.itshader")})
    assert imp["file"].endswith("My_Posterize.itshader") and str(tmp_path / "home") in imp["file"]
    names = {(p["name"], p["origin"]) for p in api.tool_run("shader_presets", {})["presets"]}
    assert ("My/Posterize", "user") in names
    (tmp_path / "bad.itshader").write_text('{"version": 1, "name": "x", "params": [], "shader": "void main(){}"}')
    with pytest.raises(LazyRoomError) as e:
        api.tool_run("shader_presets", {"action": "import", "path": str(tmp_path / "bad.itshader")})
    assert e.value.code == ErrorCode.INVALID_PARAMS


def test_shader_presets_run_in_filter_shader():
    from lazyroom.core.paths import package_data_dir
    from lazyroom.core.registry import REGISTRY, ensure_loaded

    ensure_loaded()
    if "filter.shader" not in REGISTRY:
        pytest.skip("filter.shader not loaded (LAZYROOM_ITB_ONLY)")
    img = synthetic_image(40, 30)
    for p in sorted((package_data_dir() / "shaders").glob("*.itshader")):
        try:
            out = _run("filter.shader", img, shader=p.read_text())
        except LazyRoomError as exc:
            if exc.code == ErrorCode.BACKEND_UNAVAILABLE:
                pytest.skip(f"OpenGL unavailable: {exc}")
            raise
        assert out.shape[:2] == img.shape[:2] and np.abs(out[..., :3] - img).max() > 0.01


# ---------------------------------------------------------------------------
# Export profiles
# ---------------------------------------------------------------------------

def test_export_profiles_catalogue(api):
    res = api.tool_run("export_profiles", {})
    assert res["count"] == 61
    ids = [p["id"] for p in res["profiles"]]
    assert ids[0] == "web_optimized" and "telegram_sticker" in ids and ids[-1] == "twitch_profile"
    ig = api.tool_run("export_profiles", {"platform": "instagram"})
    assert ig["count"] == 6 and all(p["format"] == "jpg" and p["quality"] == 90 for p in ig["profiles"])


def test_export_profile_sizes():
    from lazyroom.tools.presets_misc import _profiles, apply_profile_size

    P = _profiles()
    img = synthetic_image(320, 240)
    assert apply_profile_size(img, P["web_thumbnail"]).shape[:2] == (480, 640)  # flexible: long side = 640
    tall = synthetic_image(200, 400)
    assert apply_profile_size(tall, P["telegram_sticker"]).shape[:2] == (512, 256)
    sq = apply_profile_size(img, P["instagram_square"])  # upscaled x4.5 to cover, centre-cropped
    assert sq.shape[:2] == (1080, 1080)
    big = synthetic_image(1600, 1200)
    hdr = apply_profile_size(big, P["x_header"])  # larger than the canvas: centre crop at native scale
    assert hdr.shape[:2] == (500, 1500)
    assert np.allclose(hdr, big[350:850, 50:1550])


def test_export_profile_tool_writes_files(api, photo_path, tmp_path):
    od = tmp_path / "out"
    res = api.tool_run("export_profile", {"inputs": [str(photo_path)], "profile": "telegram_custom_emoji",
                                          "output_dir": str(od)})
    assert res["succeeded"] == 1
    from PIL import Image

    with Image.open(od / "photo_telegram_custom_emoji.png") as im:
        assert im.size == (100, 75) and im.format == "PNG"
    res = api.tool_run("export_profile", {"inputs": [str(photo_path)], "profile": "web_thumbnail",
                                          "output_dir": str(od)})
    with Image.open(od / "photo_web_thumbnail.webp") as im:
        assert im.size == (640, 480) and im.format == "WEBP"


# ---------------------------------------------------------------------------
# Collage layouts (lib/collages)
# ---------------------------------------------------------------------------

from lazyroom.tools._collage_layouts import (  # noqa: E402
    DEFAULT_CORNER_RADIUS,
    DEFAULT_SPACING,
    layout_ids,
    load_layouts,
    render_collage,
)

# Upstream CollageLayoutFactory.COLLAGE_MAP: 316 layouts, grouped by frame count as the picker does.
UPSTREAM_COUNTS = {1: 1, 2: 14, 3: 48, 4: 28, 5: 36, 6: 20, 7: 15, 8: 19, 9: 14, 10: 11, 11: 11, 12: 11,
                   13: 11, 14: 11, 15: 10, 16: 11, 17: 12, 18: 11, 19: 11, 20: 11}
RED, BLUE, GREEN = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)
WHITE = (1.0, 1.0, 1.0, 1.0)


def _solid(rgb, h=30, w=40, alpha=None):
    img = np.empty((h, w, 3 if alpha is None else 4), np.float32)
    img[..., :3] = rgb
    if alpha is not None:
        img[..., 3] = alpha
    return img


def _synthetic(i: int, rgba: bool) -> np.ndarray:
    h, w = 24 + 3 * i, 32 + 5 * i
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.dstack([x / (w - 1), y / (h - 1), np.full((h, w), (i % 7) / 6, np.float32)])
    if rgba:
        img = np.dstack([img, 0.25 + 0.75 * x / (w - 1)])
    return img.astype(np.float32)


def test_layout_count_matches_upstream():
    layouts = load_layouts()
    assert len(layouts) == 316
    ids = [lay["id"] for lay in layouts]
    assert len(set(ids)) == len(ids)
    assert layout_ids() == ids
    counts: dict[int, int] = {}
    for lay in layouts:
        assert lay["images"] == len(lay["frames"]) >= 1
        counts[lay["images"]] = counts.get(lay["images"], 0) + 1
    assert counts == UPSTREAM_COUNTS
    for n, c in UPSTREAM_COUNTS.items():
        assert len(layout_ids(n)) == c
        assert all(lay["images"] == n for lay in layouts if lay["id"] in layout_ids(n))
    # picker order: frame count, then asset name in byte order; upstream's 15_2 really has 17 frames
    assert layout_ids(2)[:4] == ["collage_2_0", "collage_2_1", "collage_2_10", "collage_2_11"]
    assert "collage_15_2" in layout_ids(17)
    assert load_layouts() is layouts  # cached


def test_frames_inside_canvas():
    for lay in load_layouts():
        for i, f in enumerate(lay["frames"]):
            assert f["index"] == i
            l, t, r, b = f["bound"]
            assert 0 <= l < r <= 1 and 0 <= t < b <= 1, lay["id"]
            pts = np.asarray(f["points"], float).reshape(-1, 2)
            assert ((pts >= 0) & (pts <= 1)).all(), lay["id"]
            assert len(pts) >= 3 or "path" in f, lay["id"]


def test_every_layout_renders():
    w, h = 120, 90
    for lay in load_layouts():
        imgs = [_synthetic(i, rgba=bool(i % 2)) for i in range(lay["images"])]
        for kwargs in ({}, {"spacing": 2, "corner_radius": 6}):
            out = render_collage(imgs, lay["id"], w, h, **kwargs)
            assert out.shape == (h, w, 4) and out.dtype == np.float32, lay["id"]
            assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1, lay["id"]


def test_upstream_defaults():
    assert DEFAULT_SPACING == 10.0 and DEFAULT_CORNER_RADIUS == 0.0
    out = render_collage([_solid(RED), _solid(BLUE)], "collage_2_0", 120, 90)
    assert np.allclose(out[0, 0], WHITE)  # 2 * spacing outer margin shows the white background


def test_frame_colours_collage_2_0():
    out = render_collage([_solid(RED), _solid(BLUE, 50, 20)], "collage_2_0", 120, 90, spacing=0, corner_radius=0)
    assert np.allclose(out[:, :60, :3], RED, atol=1e-5)
    assert np.allclose(out[:, 60:, :3], BLUE, atol=1e-5)
    assert np.allclose(out[..., 3], 1.0)


def test_spacing_leaves_background_between_frames():
    bg = (0.2, 0.9, 0.3, 1.0)
    out = render_collage([_solid(RED), _solid(BLUE)], "collage_2_0", 120, 90, spacing=10, background=bg)
    # left frame [0, 60) shrinks to x in [20, 50], right [60, 120) to [70, 100]; y in [20, 70]
    assert np.allclose(out[45, 55:66], bg, atol=1e-6)  # gap between frames
    assert np.allclose(out[45, 5], bg) and np.allclose(out[5, 35], bg)  # outer margin
    assert np.allclose(out[45, 35, :3], RED, atol=1e-5)
    assert np.allclose(out[45, 85, :3], BLUE, atol=1e-5)
    no_gap = render_collage([_solid(RED), _solid(BLUE)], "collage_2_0", 120, 90, spacing=0, background=bg)
    assert not np.isclose(no_gap[..., :3], bg[:3], atol=1e-3).all(axis=-1).any()


def test_corner_radius_rounds_corners():
    bg = (0.0, 1.0, 0.0, 1.0)
    sharp = render_collage([_solid(RED), _solid(BLUE)], "collage_2_0", 120, 90, spacing=0, corner_radius=0,
                           background=bg)
    assert np.allclose(sharp[0, 0, :3], RED) and np.allclose(sharp[89, 119, :3], BLUE)
    rounded = render_collage([_solid(RED), _solid(BLUE)], "collage_2_0", 120, 90, spacing=0, corner_radius=20,
                             background=bg)
    for y, x in ((0, 0), (89, 0), (0, 119), (89, 119), (0, 59), (0, 60)):
        assert np.allclose(rounded[y, x], bg), (y, x)
    assert np.allclose(rounded[45, 30, :3], RED) and np.allclose(rounded[45, 90, :3], BLUE)
    assert ((rounded[..., 0] > 0.01) & (rounded[..., 0] < 0.99)).any()  # anti-aliased arcs


def test_path_layout_collage_3_8_circle():
    out = render_collage([_solid(RED), _solid(BLUE), _solid(GREEN)], "collage_3_8", 120, 120, spacing=0)
    assert np.allclose(out[60, 60, :3], GREEN)  # centre circle (frame 2, path item)
    assert np.allclose(out[60, 10, :3], RED) and np.allclose(out[60, 110, :3], BLUE)
    assert np.allclose(out[60, 35, :3], GREEN) and np.allclose(out[10, 30, :3], RED)


def test_alpha_and_transparent_background():
    half = _solid(RED, alpha=0.5)
    out = render_collage([half, _solid(BLUE)], "collage_2_0", 120, 90, spacing=5, background=(0, 0, 0, 0))
    assert np.allclose(out[45, 60], 0.0)  # gap: transparent background
    assert np.allclose(out[45, 30], (1.0, 0.0, 0.0, 0.5), atol=1e-5)
    out = render_collage([half, _solid(BLUE)], "collage_2_0", 120, 90, spacing=0, background=WHITE)
    assert np.allclose(out[45, 30], (1.0, 0.5, 0.5, 1.0), atol=1e-5)


def test_missing_images_leave_frames_empty():
    out = render_collage([_solid(RED)], "collage_2_0", 120, 90, spacing=0, background=WHITE)
    assert np.allclose(out[:, :60, :3], RED) and np.allclose(out[:, 60:], WHITE)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        render_collage([_solid(RED)], "collage_nope", 10, 10)
    with pytest.raises(ValueError):
        render_collage([_solid(RED)], "collage_1_0", 10, 10, spacing=-1)
    with pytest.raises(ValueError):
        render_collage([np.zeros((4, 4), np.float32)], "collage_1_0", 10, 10)


def test_collage_tools(api, tmp_path):
    from lazyroom.core.io import save_image

    res = api.tool_run("collage_layouts", {"images": 2})
    assert res["count"] == UPSTREAM_COUNTS[2] and res["layouts"][0]["id"] == "collage_2_0"
    assert api.tool_run("collage_layouts", {})["count"] == 316
    paths = []
    for i, c in enumerate((RED, BLUE)):
        save_image(_solid(c), tmp_path / f"in{i}.png")
        paths.append(str(tmp_path / f"in{i}.png"))
    out = api.tool_run("collage_layout", {"inputs": paths, "width": 200, "height": 100, "spacing": 0,
                                          "output": str(tmp_path / "c.png")})
    assert out["layout"] == "collage_2_0" and out["output"]["width"] == 200
    from lazyroom.core.io import load_image

    img, _ = load_image(tmp_path / "c.png")
    assert np.allclose(img[50, 20, :3], RED, atol=0.02) and np.allclose(img[50, 180, :3], BLUE, atol=0.02)
    with pytest.raises(LazyRoomError):
        api.tool_run("collage_layout", {"inputs": paths, "layout": "nope", "output": str(tmp_path / "d.png")})


# ---------------------------------------------------------------------------
# Palette file formats (lib/palette)
# ---------------------------------------------------------------------------

import io  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import zipfile  # noqa: E402

from lazyroom.tools import _palette_formats as pf  # noqa: E402
from lazyroom.tools._palette_formats import (  # noqa: E402
    EXPORT_ORDER,
    FORMATS,
    ColorGroup,
    Palette,
    PaletteColor,
    PaletteFormatError,
    decode,
    decode_order_for_filename,
    encode,
    formats_for_filename,
)

_C = PaletteColor.rgb
_ALL = list(FORMATS)

# Upstream PaletteFormat enum order.
_ENUM = ["ACB", "ACO", "ACT", "ANDROID_XML", "ASE", "BASIC_XML", "COREL_PAINTER", "COREL_DRAW", "SCRIBUS_XML",
         "COREL_PALETTE", "CSV", "DCP", "GIMP", "HEX_RGBA", "IMAGE", "JSON", "OPEN_OFFICE", "PAINT_NET",
         "PAINT_SHOP_PRO", "RGBA", "RGB", "RIFF", "SKETCH", "SKP", "SVG", "SWIFT", "KOTLIN", "COREL_DRAW_V3", "CLF",
         "SWATCHES", "INFINITE_PAINTER", "AUTODESK_COLOR_BOOK", "SIMPLE_PALETTE", "SWATCHBOOKER", "AFPALETTE", "XARA",
         "KOFFICE", "HPL", "SKENCIL", "VGA_24BIT", "VGA_18BIT", "KRITA"]

# RGB precision of each format for 8-bit-exact inputs (k/255); everything else round trips exactly.
_TOL = {"ase": 1e-6, "dcp": 1e-6, "afpalette": 1e-6, "clf": 1e-6, "corel_draw": 1e-6, "krita": 1e-6,
        "skencil": 1e-6, "swatches": 1e-9, "aco": 1 / 65535, "xara": 1e-4, "swift": 5e-5,
        "simple_palette": 1 / 255, "corel_draw_v3": 0.01, "vga_18bit": 0.5 / 63 + 1e-9}
_NO_NAMES = {"act", "infinite_painter"}  # formats that cannot store colour names
_KEEPS_PALETTE_NAME = {"acb", "basic_xml", "corel_draw", "scribus_xml", "corel_palette", "dcp", "gimp", "json",
                       "paint_net", "skp", "swift", "swatches", "infinite_painter", "autodesk_color_book",
                       "simple_palette", "swatchbooker", "afpalette", "krita"}
_KEEPS_GROUPS = {"ase", "corel_draw", "dcp", "json", "swatches", "krita"}


def _source() -> Palette:
    return Palette("Round Trip",
                   [_C(1, 0, 0, name="Red"), _C(0, 1, 0, name="Green"), _C(0, 0, 1, name="Blue"),
                    _C(64 / 255, 128 / 255, 191 / 255, name="Accent")],
                   [ColorGroup("Group One", [_C(1, 1, 0, name="Yellow"), _C(12 / 255, 34 / 255, 56 / 255, name="Deep")])])


def _zip_bytes(*entries: tuple[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i, (name, data) in enumerate(entries):
            zf.writestr(zipfile.ZipInfo(name), data,
                        zipfile.ZIP_STORED if i == 0 and name == "mimetype" else zipfile.ZIP_DEFLATED)
    return buf.getvalue()


# --- registry ---------------------------------------------------------------------------------------------


def test_formats_registry_mirrors_upstream_enum():
    assert [f["name"] for f in FORMATS.values()] == _ENUM
    assert _ALL == [n.lower() for n in _ENUM]
    assert FORMATS["json"]["extensions"] == ["jsoncolorpalette", "json"]
    assert FORMATS["image"]["extensions"] == ["png", "jpg", "jpeg"]
    assert FORMATS["paint_shop_pro"]["extensions"] == ["psppalette", "pal"]
    assert FORMATS["gimp"]["with_palette_name"] and not FORMATS["act"]["with_palette_name"]
    assert all(f["can_encode"] and f["can_decode"] and f["description"] for f in FORMATS.values())
    assert EXPORT_ORDER[-2:] == ["hex_rgba", "csv"]
    assert sorted(EXPORT_ORDER) == sorted(_ALL) and len(EXPORT_ORDER) == len(_ALL)
    assert EXPORT_ORDER[:10] == _ALL[:10]


def test_filename_matching_uses_extension_boundaries_and_returns_all_candidates():
    assert formats_for_filename("Palette.XML") == ["android_xml", "basic_xml", "corel_draw", "scribus_xml"]
    assert formats_for_filename("not-a-palette-pal") == []
    assert formats_for_filename("x.GPL?download=1#frag") == ["gimp"]
    assert formats_for_filename("a.pal") == ["paint_shop_pro", "riff", "corel_draw_v3"]
    assert formats_for_filename("swatch.txt") == ["corel_painter", "paint_net", "rgba", "rgb"]
    order = decode_order_for_filename("book.acb")
    assert order[:2] == ["acb", "autodesk_color_book"] and order[2:] == [f for f in EXPORT_ORDER if f not in order[:2]]
    assert decode_order_for_filename("") == EXPORT_ORDER


# --- round trips --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fid", _ALL)
def test_round_trip_every_format(fid):
    src = _source()
    data = encode(src, fid)
    assert data
    palette, got = decode(data, fid)
    assert got == fid
    want, flat = src.all_colors(), palette.all_colors()
    assert len(flat) == len(want) == palette.total_color_count == 6
    tol = _TOL.get(fid, 1e-9)
    for have, exp in zip(flat, want, strict=True):
        assert have.to_rgba()[:3] == pytest.approx(exp.to_rgba()[:3], abs=tol), (fid, have, exp)
        assert have.to_rgba()[3] == pytest.approx(1.0)
    if fid not in _NO_NAMES:
        assert [c.name for c in flat] == [c.name for c in want]
    if fid in _KEEPS_PALETTE_NAME:
        assert palette.name == "Round Trip"
    elif fid == "kotlin":
        assert palette.name == "ExportedPalette"  # upstream reads the `object` name back
    else:
        assert palette.name == ""
    if fid in _KEEPS_GROUPS:
        assert [(g.name, len(g.colors)) for g in palette.groups] == [("Group One", 2)]
        assert len(palette.colors) == 4
    elif fid == "autodesk_color_book":  # pages are written without names
        assert [(g.name, len(g.colors)) for g in palette.groups] == [("Color Page 1", 2)]
    else:
        assert palette.groups == []


_SPACES = {
    "cmyk": PaletteColor.cmyk(0.1, 0.2, 0.3, 0.4, name="C1"),
    "lab": PaletteColor.lab(53.5, 20.25, -30.75, name="L1"),
    "gray": PaletteColor.gray(0.25, name="G1"),
}
_SPACE_CASES = [  # (format, colour space, component tolerance)
    ("ase", "cmyk", 1e-6), ("ase", "lab", 1e-5), ("ase", "gray", 1e-7),
    ("dcp", "cmyk", 1e-6), ("dcp", "lab", 1e-5), ("dcp", "gray", 1e-7),
    ("json", "cmyk", 0), ("json", "lab", 0), ("json", "gray", 0),
    ("swatchbooker", "cmyk", 0), ("swatchbooker", "lab", 0), ("swatchbooker", "gray", 0),
    ("corel_draw", "cmyk", 1e-6), ("corel_draw", "lab", 2e-4), ("corel_draw", "gray", 1e-6),
    ("aco", "cmyk", 1 / 65535), ("aco", "gray", 1e-4),
    ("acb", "cmyk", 1 / 255), ("acb", "lab", 1.0), ("acb", "gray", 1 / 255),
    ("scribus_xml", "cmyk", 1 / 255), ("corel_draw_v3", "cmyk", 0.005), ("clf", "lab", 0),
]


@pytest.mark.parametrize("fid,space,tol", _SPACE_CASES)
def test_color_spaces_round_trip(fid, space, tol):
    src = _SPACES[space]
    palette, _ = decode(encode(Palette("Spaces", [src]), fid), fid)
    (out,) = palette.all_colors()
    assert out.color_space == space and out.name == src.name
    assert out.components == pytest.approx(src.components, abs=tol + 1e-12)


def test_aco_converts_lab_to_rgb_and_acb_mixed_spaces_fall_back_to_rgb():
    lab = _SPACES["lab"]
    (out,) = decode(encode(Palette(colors=[lab]), "aco"), "aco")[0].colors
    assert out.color_space == "rgb" and out.to_rgba() == pytest.approx(lab.to_rgba(), abs=1 / 65535)
    mixed = decode(encode(Palette(colors=[_SPACES["cmyk"], _C(1, 0, 0, name="R")]), "acb"), "acb")[0]
    assert [c.color_space for c in mixed.colors] == ["rgb", "rgb"]


@pytest.mark.parametrize("fid", ["json", "dcp", "rgba", "hex_rgba", "paint_net", "image", "sketch", "basic_xml"])
def test_alpha_round_trip(fid):
    palette, _ = decode(encode(Palette(colors=[_C(1, 0, 0, a=128 / 255, name="Half")]), fid), fid)
    assert palette.colors[0].to_rgba()[3] == pytest.approx(128 / 255, abs=1e-6)


# --- upstream PaletteCoderCompatibilityTest ports ---------------------------------------------------------------


@pytest.mark.parametrize("fid", _ALL)
def test_every_decoder_rejects_empty_input(fid):
    with pytest.raises(PaletteFormatError):
        decode(b"", fid)


@pytest.mark.parametrize("fid", _ALL)
def test_encoded_palettes_are_detected_by_filename_without_false_positives(fid):
    src = Palette("Detection", [_C((i % 5) / 4, (i % 4) / 3, (i % 3) / 2, name=f"Color {i}") for i in range(20)])
    data = encode(src, fid)
    filename = "PALETTE." + FORMATS[fid]["extensions"][0].upper()
    detected = next(f for f in decode_order_for_filename(filename)
                    if _decodes_with_colors(data, f))
    assert detected == fid
    palette, got = decode(data, filename=filename)
    assert got == fid and palette.total_color_count > 0


def _decodes_with_colors(data: bytes, fid: str) -> bool:
    try:
        return decode(data, fid)[0].total_color_count > 0
    except PaletteFormatError:
        return False


def test_infinite_painter_uses_signed_argb_colors():
    src = b'{\n  "colors": [-16777216, -1, -65536, -16711936, -16776961],\n  "name": "Basic Hex"\n}'
    palette, _ = decode(src, "infinite_painter")
    assert palette.name == "Basic Hex" and len(palette.colors) == 5
    assert [c.to_argb() for c in palette.colors] == [-16777216, -1, -65536, -16711936, -16776961]
    text = encode(palette, "infinite_painter").decode()
    assert '"name": "Basic Hex"' in text and "-16777216" in text and "-16776961" in text
    transparent_red = encode(Palette(colors=[_C(1, 0, 0, a=0.0)]), "infinite_painter").decode()
    assert transparent_red == '{\n    "colors": [\n        -65536\n    ]\n}'  # kotlinx prettyPrint, alpha forced


def test_krita_decoder_reads_documented_colorset_structure():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<Colorset name="Demo Palette" comment="" columns="2" rows="1" readonly="false" version="1.0">
  <ColorSetEntry name="Red" id="red_1" bitdepth="F32" spot="false">
    <Position row="0" column="0"/>
    <sRGB r="1.0" g="0.0" b="0.0"/>
  </ColorSetEntry>
  <Group name="Group One" rows="1">
    <ColorSetEntry name="Green Spot" id="green_1" bitdepth="F32" spot="true">
      <Position row="0" column="0"/>
      <CMYK c="1.0" m="0.0" y="1.0" k="0.0"/>
    </ColorSetEntry>
  </Group>
</Colorset>"""
    data = _zip_bytes(("mimetype", b"application/x-krita-palette"), ("colorset.xml", xml.encode()),
                      ("profiles.xml", b'<?xml version="1.0" encoding="UTF-8"?><Profiles/>'))
    palette, _ = decode(data, "krita")
    assert palette.name == "Demo Palette"
    assert [c.name for c in palette.colors] == ["Red"]
    assert [g.name for g in palette.groups] == ["Group One"]
    spot = palette.groups[0].colors[0]
    assert (spot.name, spot.color_type, spot.color_space) == ("Green Spot", "spot", "cmyk")


def test_krita_encoder_writes_mimetype_and_documented_xml():
    palette = Palette("Krita Export", [_C(1, 0, 0, name="Red")],
                      [ColorGroup("Group One", [PaletteColor.cmyk(1, 0, 1, 0, name="CMYK Color")])])
    data = encode(palette, "krita")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        infos = zf.infolist()
        assert [i.filename for i in infos] == ["mimetype", "colorset.xml", "profiles.xml"]
        assert infos[0].compress_type == zipfile.ZIP_STORED
        assert zf.read("mimetype") == b"application/x-krita-palette"
        xml = zf.read("colorset.xml").decode()
    assert "<Colorset" in xml and "<ColorSetEntry" in xml
    assert '<sRGB r="1" g="0" b="0"/>\n    <Position row="0" column="0"/>' in xml
    decoded, _ = decode(data, "krita")
    assert decoded.total_color_count == 2 and decoded.name == "Krita Export"
    assert decoded.groups[0].name == "Group One"


def test_gimp_encoder_always_writes_version_2_header():
    text = encode(Palette(colors=[_C(1, 0, 0, name="Primary\tRed")]), "gimp").decode()
    assert text.split("\n")[:5] == ["GIMP Palette", "Name: ", "Columns: 0", "#Colors: 1", "255\t0\t0\tPrimary Red"]


def test_basic_xml_decoder_works_without_namespace_aware_parser():
    xml = b'<?xml version="1.0" encoding="utf-8"?>\n<palette name="Basic">\n  <color name="Red" r="255" g="0" b="0" a="255" />\n</palette>'
    palette, _ = decode(xml, "basic_xml")
    assert palette.name == "Basic" and [c.name for c in palette.colors] == ["Red"]


def test_android_colors_xml_decoder_reads_plain_resources():
    xml = b'<?xml version="1.0" encoding="utf-8"?>\n<resources>\n  <color name="accent">#80FF0000</color>\n</resources>'
    (color,) = decode(xml, "android_xml")[0].colors
    assert color.name == "accent"
    assert color.to_rgba() == pytest.approx((1.0, 0.0, 0.0, 128 / 255))


def test_act_encoder_stays_within_standard_file_size():
    assert len(encode(Palette(colors=[_C(1, 0, 0, name="Named Color")]), "act")) == 772


def test_riff_encoder_writes_little_endian_chunk_size():
    data = encode(Palette(colors=[_C(1, 0, 0, name="Red"), _C(0, 1, 0)]), "riff")
    assert struct.unpack_from("<i", data, 4)[0] == len(data) - 8


# --- golden bytes (hand-built from the upstream layouts) -------------------------------------------------------


def _adobe_string(s: str) -> bytes:
    return struct.pack(">I", len(s)) + s.encode("utf-16-be")


def test_aco_golden_v1_then_v2():
    palette = Palette("ignored", [_C(1.0, 0.5, 0.0, name="Hi"), PaletteColor.gray(0.25),
                                  PaletteColor.cmyk(0.0, 1.0, 0.5, 0.0, name="C")])
    records = [struct.pack(">5H", 0, 0xFFFF, 0x7FFF, 0, 0),  # RGB, 0.5 * 65535 truncated
               struct.pack(">5H", 8, 2500, 0, 0, 0),  # gray * 10000
               struct.pack(">5H", 2, 0xFFFF, 0, 65535 - 32767, 0xFFFF)]  # CMYK stored as 65535 - ink
    names = ["Hi", "", "C"]
    expected = struct.pack(">HH", 1, 3) + b"".join(records)
    expected += struct.pack(">HH", 2, 3) + b"".join(r + _adobe_string(n) for r, n in zip(records, names, strict=True))
    assert encode(palette, "aco") == expected
    # a v1-only file (no v2 section) is accepted; v2 names win when present
    v1_only, _ = decode(struct.pack(">HH", 1, 1) + records[0], "aco")
    assert [c.name for c in v1_only.colors] == [""]
    assert [c.name for c in decode(expected, "aco")[0].colors] == names
    with pytest.raises(PaletteFormatError):  # must start with version 1
        decode(struct.pack(">HH", 2, 1) + records[0] + _adobe_string("x"), "aco")


def test_aco_lab_ab_are_signed():  # documented deviation: upstream reads a/b unsigned
    data = struct.pack(">HH", 1, 1) + struct.pack(">HHhhH", 7, 5000, -2000, 1500, 0)
    (lab,) = decode(data, "aco")[0].colors
    assert lab.color_space == "lab" and lab.components == pytest.approx([50.0, -20.0, 15.0])


def test_ase_golden_header_and_blocks():
    palette = Palette("x", [_C(1, 0, 0, name="R")], [ColorGroup("G", [PaletteColor.gray(0.5, name="g")])])

    def block(kind: int, body: bytes) -> bytes:
        return struct.pack(">HI", kind, len(body)) + body

    def name(s: str) -> bytes:
        return struct.pack(">H", len(s) + 1) + s.encode("utf-16-be") + b"\0\0"

    expected = b"ASEF" + struct.pack(">HHI", 1, 0, 4)
    expected += block(0x0001, name("R") + b"RGB " + struct.pack(">fffH", 1, 0, 0, 0))
    expected += block(0xC001, name("G"))
    expected += block(0x0001, name("g") + b"Gray" + struct.pack(">fH", 0.5, 0))
    expected += struct.pack(">HI", 0xC002, 0)
    assert encode(palette, "ase") == expected
    spot = Palette(colors=[PaletteColor.cmyk(0, 0, 0, 1, name="S", color_type="spot")])
    assert encode(spot, "ase").endswith(b"CMYK" + struct.pack(">ffffH", 0, 0, 0, 1, 1))
    lab = encode(Palette(colors=[PaletteColor.lab(50, 10, -10)]), "ase")
    assert lab.endswith(b"LAB " + struct.pack(">fffH", 0.5, 10, -10, 0))  # L stored as L / 100


def test_act_golden_768_and_772_layouts():
    full = Palette(colors=[_C(i / 255, (255 - i) / 255, (i * 7 % 256) / 255) for i in range(256)])
    data = encode(full, "act")
    assert data == b"".join(bytes((i, 255 - i, i * 7 % 256)) for i in range(256))
    assert len(decode(data, "act")[0].colors) == 256
    short = encode(Palette(colors=[_C(0.5, 1, 0), _C(0, 0, 1)]), "act")
    assert short[:6] == bytes((127, 255, 0, 0, 0, 255))  # channels truncated
    assert short[6:768] == bytes(762) and short[768:] == b"\x00\x02\xff\xff"
    # count + transparency index trailer: 3 colours, colour 1 transparent
    palette, _ = decode(bytes(range(9)) + bytes(759) + b"\x00\x03\x00\x01", "act")
    assert [c.to_rgba()[3] for c in palette.colors] == [1.0, 0.0, 1.0]
    with pytest.raises(PaletteFormatError):
        decode(bytes(767), "act")


def test_riff_golden_layout():
    data = encode(Palette(colors=[_C(1, 0, 0, name="Red"), _C(0, 1, 0)]), "riff")
    names = b"Red\n"  # names joined by '\n', even length -> no pad byte
    body = b"PAL " + b"data" + struct.pack("<IHH", 12, 3, 2) + b"\xff\0\0\0" + b"\0\xff\0\0"
    body += b"name" + struct.pack("<I", len(names)) + names
    assert data == b"RIFF" + struct.pack("<I", len(body)) + body
    odd = encode(Palette(colors=[_C(0, 0, 0, name="Odd")]), "riff")
    assert odd.endswith(b"name" + struct.pack("<I", 3) + b"Odd\0")
    assert [c.name for c in decode(data, "riff")[0].colors] == ["Red", ""]


def test_gpl_golden_text():
    palette = Palette("My\tPalette", [_C(0.5, 1, 0, name="Half"), _C(0.25, 0.75, 1)])
    assert encode(palette, "gimp") == (b"GIMP Palette\nName: My Palette\nColumns: 0\n#Colors: 2\n"
                                       b"128\t255\t0\tHalf\n64\t191\t255\n")  # Math.round, half up
    text = b"\xef\xbb\xbfGIMP Palette\r\nName: Web\r\nColumns: 4\r\n# comment\r\n 12  34 56   Some name\r\n300 1 1 bad\r\n"
    palette, _ = decode(text, "gimp")
    assert palette.name == "Web"
    assert [(c.name, c.to_rgba()[:3]) for c in palette.colors] == [("Some name", (12 / 255, 34 / 255, 56 / 255))]
    with pytest.raises(PaletteFormatError):
        decode(b"JASC-PAL\n0100\n0\n", "gimp")


def test_dcp_and_cpl_golden_headers():
    dcp = encode(Palette("P", [_C(1, 0, 0, name="R")]), "dcp")
    assert dcp[:4] == b"\x9c\x7d\x01\x00"  # BOM 32156 and version 1, little endian
    assert dcp[4:8] == b"\x01\x00P\x00"  # Pascal UTF-16LE palette name
    assert dcp[8:10] == b"\x01\x00" and dcp[10:11] == b"\xea"  # one group (the global colours), group tag
    cpl = encode(Palette("Pal", [_C(1, 0.5, 0, name="Orange")]), "corel_palette")
    assert cpl == (b"\xdc\xdc\x03Pal\x01\x00" + b"\x05\x00" + bytes(6) + bytes((0, 127, 255, 0)) + b"\x06Orange")


def test_less_common_decode_paths():
    # CPL 0xCDDC: header table, spot palette type 3 => "long" records holding two colour models each
    head = struct.pack(">H", 0xCDDC) + struct.pack("<i", 3)
    base = len(head) + 24
    head += struct.pack("<6i", 0, base, 1, base + 5, 2, base + 7)
    rec = struct.pack("<IH", 0, 5) + bytes(6) + bytes((0, 128, 255, 0))  # BGR fractions
    rec += struct.pack("<H", 2) + bytes(4) + bytes((10, 20, 30, 40)) + b"\x03Abc"  # CMYK percentages
    cpl, _ = decode(head + b"\x04Spot" + struct.pack("<HH", 3, 1) + rec, "corel_palette")
    assert cpl.name == "Spot" and [(c.name, c.color_type, c.color_space) for c in cpl.colors] == [
        ("Abc", "spot", "rgb"), ("Abc", "spot", "cmyk")]
    assert cpl.colors[0].components == pytest.approx([1.0, 128 / 255, 0.0])
    assert cpl.colors[1].components == pytest.approx([0.1, 0.2, 0.3, 0.4])
    # ACB: "$$$/...=Title" titles, CMYK stored as 255 - ink, empty name+code records skipped
    acb = b"8BCB" + struct.pack(">HH", 1, 3000) + b"".join(
        _adobe_string(s) for s in ("$$$/colorbook/X/title=Pantone X", "PANTONE ", " C", ""))
    acb += struct.pack(">4H", 2, 0, 0, 2) + _adobe_string("100") + b"   100" + bytes((0, 255, 255, 255))
    acb += _adobe_string("") + b"      " + bytes(4) + b"spflspot"
    book, _ = decode(acb, "acb")
    assert book.name == "Pantone X" and [(c.name, c.components) for c in book.colors] == [("100", [1, 0, 0, 0])]
    # Affinity: CMYK (KYMC) and gray (YARG) records plus the PaNV names section
    af = struct.pack("<II", 0x414BFF00, 11) + b"PlCN" + struct.pack("<I", 2) + b"AF" + b"PalV" + struct.pack("<I", 2)
    af += b"rloC" + bytes(6) + b"KYMC" + b"Hloc_" + struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
    af += b"rloC" + bytes(6) + b"YARG" + b"<loc_" + struct.pack("<f", 0.5)
    af += b"PaNV" + struct.pack("<III", 0, 2, 1) + b"k" + struct.pack("<I", 1) + b"g"
    aff, _ = decode(af, "afpalette")
    assert aff.name == "AF" and [(c.name, c.color_space) for c in aff.colors] == [("k", "cmyk"), ("g", "gray")]
    with pytest.raises(PaletteFormatError, match="Unsupported palette type"):
        decode(af.replace(b"KYMC", b"XXXX"), "afpalette")
    # Xara JCW: CMYK spot colour space 9, NUL-padded Latin-1 names, unnamed -> "c<index>"
    jcw = b"JCW\x01" + struct.pack("<H", 2) + bytes((9, 4)) + struct.pack("<4H", 1000, 2000, 3000, 4000) + b"ab\0\0"
    jcw += struct.pack("<4H", 0, 0, 0, 10000) + bytes(4)
    xara, _ = decode(jcw, "xara")
    assert [(c.name, c.color_type) for c in xara.colors] == [("ab", "spot"), ("c1", "spot")]
    assert xara.colors[0].components == pytest.approx([0.1, 0.2, 0.3, 0.4])
    # Swatchbooker: <book> order wins; HSV materials become RGB
    sb = ('<SwatchBook xmlns="http://www.selapa.net/swatchbook/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">'
          '<materials><color><metadata><dc:identifier>a</dc:identifier></metadata><values model="RGB">1 0 0</values>'
          '</color><color><metadata><dc:title>B</dc:title><dc:identifier>b</dc:identifier></metadata>'
          '<values model="HSV">0.3333333333333333 1 1</values></color></materials>'
          '<book><swatch material="b"/><swatch material="a"/></book></SwatchBook>')
    swb, _ = decode(_zip_bytes(("swatchbook.xml", sb.encode())), "swatchbooker")
    assert [c.name for c in swb.colors] == ["B", "a"]
    assert swb.colors[0].to_rgba() == pytest.approx((0, 1, 0, 1))
    # CorelDraw XML: colours may reference a named colorspace template
    cdr = (b'<palette name="T"><colorspaces><cs name="Spot1"><color cs="CMYK" tints="0,1,1,0"/></cs></colorspaces>'
           b'<colors><page><color name="UsesSpot" cs="Spot1"/></page></colors></palette>')
    (spot,) = decode(cdr, "corel_draw")[0].colors
    assert (spot.name, spot.color_space, spot.components) == ("UsesSpot", "cmyk", [0, 1, 1, 0])


def test_hex_and_rounding_quirks_follow_upstream():
    assert _C(1, 0.5, 0).hex() == "#FF7F00"  # channels truncated like hexString()
    assert _C(1, 0.5, 0, a=0.5).hex() == "#FF7F007F"
    assert encode(Palette(colors=[_C(1, 0.5, 0, name="o")]), "hex_rgba") == b"; o\n#ff7f00\n"
    # COREL_DRAW_V3 uses kotlin.math.round (ties to even): 12.5 -> 12, 37.5 -> 38
    cdr = encode(Palette(colors=[PaletteColor.cmyk(0.125, 0.375, 0, 0, name="T")]), "corel_draw_v3")
    assert cdr == b'"T"    12    38    0    0\r\n'
    assert encode(Palette(colors=[_C(1, 0, 0, name="R"), _C(0, 0, 1)]), "rgba") == b"#ff0000ff R\r\n#0000ffff"
    assert encode(Palette(colors=[_C(1, 0, 0, name="R")]), "csv") == b"#ff0000, R\n"
    assert encode(Palette("N", [_C(0, 0, 0, name="K")]), "paint_net").endswith(b"; Palette Name: N\r\n; K\r\nFF000000\r\n")


def test_json_matches_kotlinx_layout():
    data = encode(Palette("P", [_C(1, 0, 0, name="Red"), PaletteColor.gray(0.5, alpha=0.25, color_type="spot")]), "json")
    text = re.sub(r'"id":"[0-9a-f-]{36}"', '"id":"ID"', data.decode())
    assert text == ('{"name":"P","colors":[{"name":"Red","colorSpace":"RGB","colorComponents":[1.0,0.0,0.0],'
                    '"id":"ID"},{"colorType":"Spot","colorSpace":"Gray","colorComponents":[0.5],"alpha":0.25,'
                    '"id":"ID"}],"id":"ID"}')
    assert json.loads(data)["colors"][1]["colorType"] == "Spot"
    with pytest.raises(PaletteFormatError):  # component count must match the colour space
        decode(b'{"colors":[{"colorSpace":"RGB","colorComponents":[1.0]}]}', "json")


# --- colour model -------------------------------------------------------------------------------------------


def test_color_conversions_match_palette_color_kt():
    assert PaletteColor.cmyk(0.2, 0.4, 0.6, 0.5).to_rgba() == pytest.approx((0.4, 0.3, 0.2, 1.0))
    assert PaletteColor.gray(0.3, alpha=0.5).to_rgba() == (0.3, 0.3, 0.3, 0.5)
    red_lab = _C(1, 0, 0).to_lab()  # androidx ColorUtils.colorToLAB (values from AOSP ColorUtilsTest)
    assert red_lab[:3] == pytest.approx((53.23, 80.11, 67.22), abs=0.01)
    assert _C(0, 1, 0).to_lab()[:3] == pytest.approx((87.74, -86.18, 83.18), abs=0.01)
    assert _C(0, 0, 1).to_lab()[:3] == pytest.approx((32.30, 79.19, -107.86), abs=0.011)
    assert PaletteColor.lab(*red_lab[:3]).to_rgba() == (1.0, 0.0, 0.0, 1.0)
    mid = PaletteColor.lab(50, 0, 0).to_rgba()  # LABToColor result is 8-bit, read back as float32 / 255
    assert mid[0] == mid[1] == mid[2] and round(mid[0] * 255) == 119
    assert _C(2, -1, 0.5).components == [1.0, 0.0, 0.5]  # factory clamps
    with pytest.raises(PaletteFormatError):
        PaletteColor("bad", "cmyk", [0, 0, 0])
    assert _C(0.25, 0.5, 0.75).converted("cmyk").converted("rgb").to_rgba() == pytest.approx((0.25, 0.5, 0.75, 1.0))


# --- decode API, probing and hardening -----------------------------------------------------------------------


def test_decode_by_filename_probing():
    gpl = encode(_source(), "gimp")
    assert decode(gpl, filename="colors.gpl")[1] == "gimp"
    assert decode(gpl, filename="mystery.bin")[1] == "gimp"  # falls back to the app's probing order
    assert decode(gpl)[1] == "gimp"
    riff = encode(_source(), "riff")
    assert decode(riff, filename="x.pal")[1] == "riff"  # PAINT_SHOP_PRO is tried first and rejects it
    cdr3 = encode(_source(), "corel_draw_v3")
    assert decode(cdr3, filename="x.pal")[1] == "corel_draw_v3"
    adsk = encode(_source(), "autodesk_color_book")
    assert decode(adsk, filename="book.acb")[1] == "autodesk_color_book"
    assert decode(encode(_source(), "acb"), filename="book.acb")[1] == "acb"
    with pytest.raises(PaletteFormatError):
        decode(b"\x00\x01\x02\x03 nope!", filename="x.gpl")  # 10 bytes: not VGA, no hex
    with pytest.raises(PaletteFormatError):
        decode(gpl, "no_such_format")
    assert issubclass(PaletteFormatError, ValueError)


@pytest.mark.parametrize("fid", ["acb", "corel_palette", "image", "swatches", "afpalette", "krita"])
def test_encoders_reject_empty_palettes_like_upstream(fid):
    with pytest.raises(PaletteFormatError):
        encode(Palette("Empty"), fid)


def test_xml_dtd_and_zip_size_limits(monkeypatch):
    bomb = b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaa">]><resources><color name="x">&a;</color></resources>'
    with pytest.raises(PaletteFormatError, match="DTD"):
        decode(bomb, "android_xml")
    data = encode(_source(), "krita")
    monkeypatch.setattr(pf, "_ZIP_LIMIT", 64)
    with pytest.raises(PaletteFormatError, match="too large"):
        decode(data, "krita")


def test_documented_decode_deviations():
    xml = b'<resources><color name="a">#FF0000</color><color name="b">#8000FF00</color></resources>'
    a, b = decode(xml, "android_xml")[0].colors  # #RRGGBB is RGB (upstream would read blue)
    assert a.to_rgba() == (1.0, 0.0, 0.0, 1.0) and b.to_rgba() == pytest.approx((0, 1, 0, 128 / 255))
    named = Palette(colors=[_C(1, 0, 0, name="Grün"), _C(0, 0, 1, name="Blau")])
    for fid in ("vga_24bit", "vga_18bit"):
        assert [c.name for c in decode(encode(named, fid), fid)[0].colors] == ["Grün", "Blau"]


def test_image_coder_png_swatches_and_name_trailer():
    data = encode(Palette(colors=[_C(1, 0, 0, name="R"), _C(0, 0, 1, a=0.5, name="B")]), "image")
    assert data.startswith(b"\x89PNG") and data.endswith(b"\n; IMAGE_NAMES: R|B\n")
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        assert im.size == (64, 32) and im.mode == "RGBA"
    palette, _ = decode(data, "image")
    assert [c.name for c in palette.colors] == ["R", "B"]
    assert palette.colors[1].to_rgba() == pytest.approx((0, 0, 1, 127 / 255))


def test_palette_tools_round_trip(api, tmp_path):
    cols = ["Brick=#B22222", "#00FF00", "Sky=#87CEEBCC"]
    out = api.tool_run("palette_export", {"colors": cols, "name": "Test", "output": str(tmp_path / "p.gpl")})
    assert out["format"] == "gimp" and (tmp_path / "p.gpl").read_text().startswith("GIMP Palette")
    names = [c["name"] for c in out["palette"]["colors"]]
    assert names[0] == "Brick" and names[1] == "Green" and names[2] == "Sky"  # unnamed -> colour library
    imp = api.tool_run("palette_import", {"input": str(tmp_path / "p.gpl"), "swatch_output": str(tmp_path / "s.png")})
    assert imp["format"] == "gimp" and imp["palette"]["name"] == "Test"
    assert [c["hex"][:7] for c in imp["palette"]["colors"]] == ["#B22222", "#00FF00", "#87CEEB"]
    assert (tmp_path / "s.png").exists()
    # convert GPL -> ASE -> ACO keeps colours
    api.tool_run("palette_export", {"source": str(tmp_path / "p.gpl"), "output": str(tmp_path / "p.ase")})
    api.tool_run("palette_export", {"source": str(tmp_path / "p.ase"), "format": "aco",
                                    "output": str(tmp_path / "p.bin")})
    back = api.tool_run("palette_import", {"input": str(tmp_path / "p.bin"), "format": "aco"})
    assert [c["hex"][:7] for c in back["palette"]["colors"]] == ["#B22222", "#00FF00", "#87CEEB"]
    assert (tmp_path / "p.ase").read_bytes()[:4] == b"ASEF"


def test_palette_export_from_image_and_errors(api, photo_path, tmp_path):
    out = api.tool_run("palette_export", {"input": str(photo_path), "count": 8, "output": str(tmp_path / "i.act")})
    assert out["format"] == "act" and 1 <= out["palette"]["total_colors"] <= 8
    assert len((tmp_path / "i.act").read_bytes()) in (768, 772)
    with pytest.raises(LazyRoomError):  # format cannot be inferred
        api.tool_run("palette_export", {"colors": ["#000000"], "output": str(tmp_path / "x.unknown")})
    with pytest.raises(LazyRoomError):  # nothing or several sources
        api.tool_run("palette_export", {"output": str(tmp_path / "y.gpl")})
    (tmp_path / "bad.aco").write_bytes(b"")
    with pytest.raises(LazyRoomError):
        api.tool_run("palette_import", {"input": str(tmp_path / "bad.aco"), "format": "aco"})
