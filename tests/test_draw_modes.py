"""ImageToolbox Draw port: draw.paint / flood_fill / spot_heal / warp / shape_layer."""
import json
import math

import cv2
import numpy as np
import pytest

import lazyroom.ops.draw_modes  # noqa: F401  (registers the operations; the lead wires ops/__init__)
from lazyroom.core.context import RenderContext
from lazyroom.core.errors import ErrorCode, LazyRoomError
from lazyroom.core.registry import get_op
from lazyroom.ops import _draw_paths as P
from lazyroom.ops import _draw_raster as R
from lazyroom.ops.draw_modes import MODES, PATH_MODES
from lazyroom.ops.shapes import LINE_STYLES
from tests.conftest import synthetic_image


def run(op_id, img, ctx=None, **params):
    spec = get_op(op_id)
    return spec.run(img, ctx or RenderContext(seed=1), spec.validate(params))


def white(h=200, w=300):
    return np.ones((h, w, 3), np.float32)


def changed(out, base, thr=0.02):
    return np.abs(out[..., :3] - base[..., :3]).max(-1) > thr


@pytest.fixture
def brush_png(tmp_path):
    from lazyroom.core.io import save_image

    im = np.zeros((20, 20, 4), np.float32)
    im[..., 0] = 1.0
    im[..., 3] = 1.0
    p = tmp_path / "brush.png"
    save_image(im, p)
    return str(p)


# ---------------------------------------------------------------------------
# every mode x path mode, every line style
# ---------------------------------------------------------------------------

PTS = [[0.2, 0.3], [0.4, 0.5], [0.6, 0.4], [0.8, 0.7]]


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("path_mode", list(PATH_MODES))
def test_every_mode_and_path_mode(mode, path_mode, brush_png):
    img = synthetic_image(96, 72)
    vals = {"strokes": [PTS], "mode": mode, "path_mode": path_mode, "width": 6, "color": "#FF00FF"}
    if mode == "filter_brush":
        vals.update(filter="adjust.saturation", filter_params={"saturation": -100})
    if mode == "image_brush":
        vals["image"] = brush_png
    if mode in ("text", "image_brush") and path_mode in ("spray", "flood_fill"):
        with pytest.raises(LazyRoomError) as e:
            run("draw.paint", img, **vals)
        assert e.value.code == ErrorCode.INVALID_PARAMS
        return
    out = run("draw.paint", img, **vals)
    assert out.shape == img.shape and out.dtype == np.float32 and np.isfinite(out).all()
    if mode == "eraser":  # nothing earlier in the step to erase
        assert np.allclose(out, img)
    else:
        assert changed(out, img, 0.005).any(), f"{mode}/{path_mode} drew nothing"


@pytest.mark.parametrize("style", list(LINE_STYLES))
@pytest.mark.parametrize("mode", ["pen", "highlighter", "neon"])
def test_line_styles(style, mode):
    img = white()
    out = run("draw.paint", img, strokes=[[[0.1, 0.5], [0.9, 0.5]]], mode=mode, line_style=style, width=12,
              color="#0000FF", path_mode="line")
    assert changed(out, img).any()


def test_dashes_create_gaps():
    img = white()
    solid = run("draw.paint", img, strokes=[[[0.05, 0.5], [0.95, 0.5]]], width=6, path_mode="line")
    dash = run("draw.paint", img, strokes=[[[0.05, 0.5], [0.95, 0.5]]], width=6, path_mode="line",
               line_style="dashed", dash_size=10, gap_size=10)
    row_s, row_d = solid[100, 20:280, 0], dash[100, 20:280, 0]
    assert (row_s < 0.1).all()
    on = row_d < 0.5
    assert on.any() and (~on).any()
    runs = np.flatnonzero(np.diff(on.astype(int)) != 0)
    assert len(runs) >= 10  # many dash boundaries
    # intervals = [dash, gap + width] = [10, 16] -> period 26 px (round caps only lengthen the dashes)
    period = np.diff(runs[::2]).mean()
    assert abs(period - 26) < 1.5


def test_dot_dashed_and_stamped_and_zigzag_differ():
    img = white()
    outs = {s: run("draw.paint", img, strokes=[[[0.1, 0.5], [0.9, 0.5]]], width=16, path_mode="line",
                   line_style=s) for s in LINE_STYLES}
    cov = {s: (1 - o[..., 0]).sum() for s, o in outs.items()}
    assert cov["dashed"] < cov["none"] and cov["dot_dashed"] < cov["none"]
    assert cov["zigzag"] < cov["none"] and cov["stamped"] < cov["none"]
    assert len({round(v) for v in cov.values()}) == len(cov)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_filled_star_covers_center_and_vertices():
    img = white(300, 300)
    out = run("draw.paint", img, strokes=[[[0.1, 0.1], [0.9, 0.9]]], path_mode="star", filled=True,
              vertices=5, inner_radius=0.5, regular=True)
    ink = out[..., 0] < 0.5
    assert ink[150, 150]
    # vertices: regular star of radius 120 px, first vertex on top (ImageToolbox orientation)
    for k in range(5):
        a = math.radians(-90 + k * 72)
        x, y = 150 + 112 * math.cos(a), 150 + 112 * math.sin(a)
        assert ink[int(round(y)), int(round(x))], k
    # between two outer vertices (on the outer circle) it is concave -> empty
    a = math.radians(-90 + 36)
    assert not ink[int(150 + 112 * math.sin(a)), int(150 + 112 * math.cos(a))]


def test_polygon_vertices_match_imagetoolbox_formula():
    p0, p1 = (10.0, 20.0), (110.0, 80.0)
    (pts, closed), = P.polygon_path(p0, p1, 5, 0, False)
    assert closed and len(pts) == 5
    assert np.allclose(pts[0], (60, 20))  # top centre (height < 0 in PathHelper flips sin)
    (reg, _), = P.polygon_path((0.0, 0.0), (20.0, 200.0), 6, 0, True)
    # upstream used |height| / 2 = 100 px for this tall box; fixed to min(w, h) / 2 = 10 px
    assert np.allclose(np.hypot(*(reg - (10, 100)).T), 10)


def test_rect_rotation_and_corner_radius():
    (pts, _), = P.rect_path((0.0, 0.0), (100.0, 50.0), 90, 0)
    assert np.allclose(pts.min(0), (25, -25)) and np.allclose(pts.max(0), (75, 75))
    (rr, _), = P.rect_path((0.0, 0.0), (100.0, 50.0), 0, 0.5)
    assert np.allclose(rr.min(0), (0, 0), atol=1e-6) and len(rr) > 8


def test_arrow_head_exists_and_points_forward():
    img = white()
    line = run("draw.paint", img, strokes=[[[0.2, 0.5], [0.8, 0.5]]], path_mode="line", width=6)
    arrow = run("draw.paint", img, strokes=[[[0.2, 0.5], [0.8, 0.5]]], path_mode="line_arrow", width=6,
                arrow_size=3, arrow_angle=150)
    extra = changed(arrow, line, 0.3)
    ys, xs = np.nonzero(extra)
    assert len(xs) > 20, "no arrow head on an exactly horizontal line"
    assert xs.min() > 0.8 * 300 - 6 * 3 - 4 and xs.max() <= 0.8 * 300 + 4  # behind the tip only
    assert ys.min() < 100 - 5 and ys.max() > 100 + 5  # both wings
    double = run("draw.paint", img, strokes=[[[0.2, 0.5], [0.8, 0.5]]], path_mode="double_line_arrow", width=6)
    xs2 = np.nonzero(changed(double, line, 0.3))[1]
    assert (xs2 < 0.2 * 300 + 25).any() and (xs2 > 0.8 * 300 - 25).any()


def test_arrow_head_geometry_helper():
    main = np.array([[0.0, 0.0], [100.0, 0.0]])
    paths = P.add_arrows(main, 10, 3, 150, start=False, end=True)
    head = paths[0][0][-1]  # first head line continues the main contour
    assert np.allclose(head, (100 - 30 * math.cos(math.radians(30)), 30 * math.sin(math.radians(30))), atol=1e-6)
    assert np.allclose(paths[1][0][1], (100 - 30 * math.cos(math.radians(30)), -15), atol=1e-6)


def test_free_path_quad_smoothing_passes_midpoints_and_end():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    f = P.free_path(pts)
    assert np.allclose(f[0], pts[0]) and np.allclose(f[-1], pts[-1])
    assert np.min(np.hypot(*(f - (10, 5)).T)) < 1e-9  # midpoint of the last two points


def test_highlighter_overlap_does_not_accumulate():
    img = white()
    loop = [[0.2, 0.5], [0.8, 0.5], [0.8, 0.2], [0.5, 0.2], [0.5, 0.8]]  # crosses itself
    out = run("draw.paint", img, strokes=[loop], mode="highlighter", color="#000000", width=20)
    vals = out[..., 0][out[..., 0] < 0.99]
    assert np.isclose(vals.min(), 0.6, atol=1e-3)  # one layer of alpha 0.4, never 0.36 (= 0.6 * 0.6)


def test_highlighter_square_caps_pen_round_caps():
    img = white()
    pen = run("draw.paint", img, strokes=[[[0.3, 0.5], [0.7, 0.5]]], path_mode="line", width=20, alpha=1)
    hl = run("draw.paint", img, strokes=[[[0.3, 0.5], [0.7, 0.5]]], path_mode="line", width=20, alpha=1,
             mode="highlighter")
    # square cap extends half the width past the end, also at the corner of the cap
    assert hl[91, 81, 0] < 0.05 and pen[91, 81, 0] > 0.9


def test_neon_white_core_and_coloured_glow():
    img = np.zeros((200, 300, 3), np.float32)
    out = run("draw.paint", img, strokes=[[[0.2, 0.5], [0.8, 0.5]]], mode="neon", color="#FF0000", width=10,
              path_mode="line", softness=30)
    assert np.allclose(out[100, 150], 1.0, atol=1e-3)  # white core
    glow = out[100 + 15, 150]
    assert glow[0] > 0.05 and glow[0] > 5 * glow[1]  # red glow outside the core


def test_softness_blurs_edges():
    img = white()
    hard = run("draw.paint", img, strokes=[[[0.2, 0.5], [0.8, 0.5]]], width=20, path_mode="line")
    soft = run("draw.paint", img, strokes=[[[0.2, 0.5], [0.8, 0.5]]], width=20, path_mode="line", softness=20)
    assert (np.abs(hard - soft).max() > 0.3) and soft[100 + 20, 150, 0] < 0.99 and hard[100 + 20, 150, 0] > 0.999


def test_outlined_shape_fill_colour():
    img = white()
    out = run("draw.paint", img, strokes=[[[0.2, 0.2], [0.8, 0.8]]], path_mode="rect", filled=False,
              fill_color="#00FF00", color="#0000FF", width=6)
    assert np.allclose(out[100, 150], (0, 1, 0), atol=1e-3)
    assert np.allclose(out[int(0.2 * 200), 150], (0, 0, 1), atol=1e-3)


def test_lasso_fills_interior():
    img = white()
    out = run("draw.paint", img, strokes=[[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]], path_mode="lasso")
    assert out[100, 150, 0] < 0.01


def test_spray_density_and_determinism():
    img = white()
    a = run("draw.paint", img, strokes=[[[0.2, 0.5], [0.8, 0.5]]], path_mode="spray", width=30, density=20)
    b = run("draw.paint", img, strokes=[[[0.2, 0.5], [0.8, 0.5]]], path_mode="spray", width=30, density=20)
    c = run("draw.paint", img, RenderContext(seed=2), strokes=[[[0.2, 0.5], [0.8, 0.5]]], path_mode="spray",
            width=30, density=20)
    d = run("draw.paint", img, strokes=[[[0.2, 0.5], [0.8, 0.5]]], path_mode="spray", width=30, density=200)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    ink = lambda o: (1 - o[..., 0]).sum()
    assert ink(d) > 3 * ink(a)
    ys, _ = np.nonzero(changed(a, img))
    assert np.abs(ys - 100).max() <= 30 + 3  # disc of radius = width


# ---------------------------------------------------------------------------
# eraser, effects, text, image
# ---------------------------------------------------------------------------

def test_eraser_removes_earlier_strokes_only():
    img = synthetic_image(120, 90)
    drawn = run("draw.paint", img, strokes=[[[0.1, 0.5], [0.9, 0.5]]], width=10, path_mode="line")
    erased = run("draw.paint", img, strokes=[{"points": [[0.1, 0.5], [0.9, 0.5]], "path_mode": "line"},
                                             {"points": [[0.5, 0.2], [0.5, 0.8]], "mode": "eraser", "width": 20}],
                 width=10)
    assert np.allclose(erased[45, 60], img[45, 60], atol=1e-5)  # stroke erased back to the photo
    assert np.allclose(erased[45, 20], drawn[45, 20], atol=1e-5)  # elsewhere the stroke remains


def test_privacy_blur_inside_only():
    img = synthetic_image(160, 120)
    out = run("draw.paint", img, strokes=[[[0.3, 0.3], [0.7, 0.7]]], mode="privacy_blur", path_mode="rect",
              filled=True, blur_radius=50)
    inside = (slice(45, 80), slice(55, 105))
    assert out[inside].std() < img[inside].std()
    assert np.array_equal(out[:30], img[:30])


def test_pixelation_makes_constant_cells():
    rng = np.random.default_rng(0)
    img = rng.random((200, 200, 3)).astype(np.float32)
    out = run("draw.paint", img, strokes=[[[0.0, 0.0], [1.0, 1.0]]], mode="pixelation", path_mode="rect",
              filled=True, pixel_size=50)  # 10 px cells
    cell = out[23:28, 23:28]
    assert np.ptp(cell.reshape(-1, 3), axis=0).max() < 1e-4


def test_pixelation_crop_matches_full_image_grid():
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.random((300, 300, 3)).astype(np.float32), (0, 0), 3)
    full = run("draw.paint", img, strokes=[[[0.0, 0.0], [1.0, 1.0]]], mode="pixelation", path_mode="rect",
               filled=True, pixel_size=37)
    part = run("draw.paint", img, strokes=[[[0.5, 0.5], [0.8, 0.8]]], mode="pixelation", path_mode="rect",
               filled=True, pixel_size=37)
    assert np.allclose(full[170:220, 170:220], part[170:220, 170:220], atol=1e-5)


def test_filter_brush_applies_registered_op_inside_stroke():
    img = synthetic_image(120, 90)
    out = run("draw.paint", img, strokes=[[[0.0, 0.5], [1.0, 0.5]]], mode="filter_brush", width=20,
              path_mode="line", filter="adjust.saturation", filter_params={"saturation": -100})
    band = out[40:50, 20:100]
    assert np.ptp(band, axis=2).max() < 0.02  # grey inside
    assert np.array_equal(out[:20], img[:20])


def test_text_on_path_and_repeat():
    img = white()
    one = run("draw.paint", img, strokes=[[[0.05, 0.6], [0.95, 0.6]]], mode="text", text="Hi", width=30,
              path_mode="line")
    rep = run("draw.paint", img, strokes=[[[0.05, 0.6], [0.95, 0.6]]], mode="text", text="Hi", width=30,
              path_mode="line", text_repeat=True)
    ys, xs = np.nonzero(changed(one, img))
    assert len(xs) and ys.max() <= 120 + 8 and ys.min() >= 120 - 30  # baseline on the path, glyphs above
    assert changed(rep, img).sum() > 4 * changed(one, img).sum()


def test_image_brush_stamps_along_path(brush_png):
    img = white()
    out = run("draw.paint", img, strokes=[[[0.1, 0.5], [0.9, 0.5]]], mode="image_brush", image=brush_png,
              width=20, path_mode="line", interval=20)
    red = (out[..., 0] > 0.9) & (out[..., 1] < 0.1)
    xs = np.nonzero(red[100])[0]
    gaps = np.flatnonzero(np.diff(xs) > 1)
    assert 4 <= len(gaps) + 1 <= 7  # 240 px path / (20 + 20) px advance = 6 copies
    assert red[100, 30]  # first copy centred on the path start


# ---------------------------------------------------------------------------
# flood fill, spot heal, warp, shape layer
# ---------------------------------------------------------------------------

def test_flood_fill_tolerance():
    img = white(100, 100)
    img[:, 50:] = 0.9  # 230 vs 255: diff 25
    img[40:60, 40:60] = 0.0
    lo = run("draw.flood_fill", img, point=[0.1, 0.1], tolerance=0.05, color="#FF0000")
    hi = run("draw.flood_fill", img, point=[0.1, 0.1], tolerance=0.2, color="#FF0000")
    red = lambda o: (o[..., 0] > 0.99) & (o[..., 1] < 0.01)
    assert red(lo)[:, :50].sum() == 50 * 100 - 10 * 20 and not red(lo)[:, 50:].any()
    assert red(hi)[:, 50:].sum() == 50 * 100 - 10 * 20 and not red(hi)[45, 45]


def test_flood_fill_region_is_4_connected():
    rgba = np.zeros((5, 5, 4), np.float32)
    rgba[..., 3] = 1
    rgba[1, 1] = rgba[2, 2] = (1, 1, 1, 1)  # diagonal neighbours only
    mask, box = P.flood_fill_region(rgba, None, 1, 1, 0.0)
    assert mask.sum() == 1 and box == (1, 1, 2, 2)


def test_flood_fill_path_mode_with_privacy_blur():
    rng = np.random.default_rng(3)
    img = rng.random((80, 80, 3)).astype(np.float32) * 0.1
    img[:, 40:] = 1.0
    out = run("draw.paint", img, strokes=[[[0.9, 0.5]]], mode="privacy_blur", path_mode="flood_fill",
              tolerance=0.05)
    assert np.array_equal(out[:, :38], img[:, :38])


@pytest.mark.parametrize("method", ["telea", "ns"])
def test_spot_heal_removes_blemish(method):
    h, w = 120, 160
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    clean = np.dstack([x / w, y / h, np.full_like(x, 0.5)]).astype(np.float32)
    spot = clean.copy()
    cv2.circle(spot, (80, 60), 6, (0.0, 0.0, 0.0), -1)
    out = run("draw.spot_heal", spot, strokes=[[[80 / w, 60 / h]]], width=20, method=method)
    e0 = np.abs(spot - clean)[50:70, 70:90].mean()
    e1 = np.abs(out - clean)[50:70, 70:90].mean()
    assert e1 < 0.2 * e0
    assert np.array_equal(out[:, :60], spot[:, :60])


def test_spot_heal_lama_requires_model_or_works():
    img = synthetic_image(64, 48)
    try:
        out = run("draw.spot_heal", img, strokes=[[[0.5, 0.5]]], width=8, method="lama")
    except LazyRoomError as e:
        assert e.code in (ErrorCode.BACKEND_UNAVAILABLE,), e
        assert "download" in (e.hint or "")
    else:
        assert out.shape == img.shape


def _band_centroid(img):
    blue = (img[..., 2] > 0.5) & (img[..., 0] < 0.5)
    return np.nonzero(blue)[1].mean()


def test_warp_move_pushes_content_along_drag():
    img = white(200, 300)
    img[:, 140:160] = (0.1, 0.2, 0.9)
    right = run("draw.warp", img, strokes=[[[0.45 + 0.01 * i, 0.5] for i in range(11)]], radius=60,
                strength=0.8, warp_mode="move")
    left = run("draw.warp", img, strokes=[[[0.55 - 0.01 * i, 0.5] for i in range(11)]], radius=60,
               strength=0.8, warp_mode="move")
    row = slice(95, 105)
    assert _band_centroid(right[row]) > _band_centroid(img[row]) + 3
    assert _band_centroid(left[row]) < _band_centroid(img[row]) - 3
    assert np.array_equal(right[:20], img[:20])  # outside the brush


def test_warp_grow_and_shrink():
    img = white(200, 200)
    cv2.circle(img, (100, 100), 15, (0.0, 0.0, 0.0), -1)
    t = np.linspace(0, 6 * np.pi, 60)
    circ = np.stack([0.5 + 0.01 * np.cos(t), 0.5 + 0.01 * np.sin(t)], 1).tolist()
    grow = run("draw.warp", img, strokes=[circ], radius=60, strength=0.3, hardness=0.2, warp_mode="grow")
    shrink = run("draw.warp", img, strokes=[circ], radius=60, strength=0.3, hardness=0.2, warp_mode="shrink")
    area = lambda o: (o[..., 0] < 0.5).sum()
    assert area(grow) > 1.1 * area(img) and area(shrink) < 0.9 * area(img)


def test_warp_swirl_direction_and_mix():
    img = white(200, 200)
    img[95:105, 100:180] = 0.0  # bar to the right of the centre
    t = np.linspace(0, 4 * np.pi, 40)
    circ = np.stack([0.5 + 0.005 * np.cos(t), 0.5 + 0.005 * np.sin(t)], 1).tolist()
    cw = run("draw.warp", img, strokes=[circ], radius=80, strength=0.6, hardness=0.0, warp_mode="swirl_cw")
    ccw = run("draw.warp", img, strokes=[circ], radius=80, strength=0.6, hardness=0.0, warp_mode="swirl_ccw")
    ys_cw = np.nonzero(cw[:, 140:150, 0] < 0.5)[0].mean()
    ys_ccw = np.nonzero(ccw[:, 140:150, 0] < 0.5)[0].mean()
    # WarpEngine samples src at R(+a)(p - c) for swirl_cw: on screen (y down) the content turns
    # counter-clockwise, i.e. the bar's right part moves up; swirl_ccw the opposite (same as upstream)
    assert ys_cw < 99 < ys_ccw
    mix = run("draw.warp", img, strokes=[[[0.6, 0.5], [0.62, 0.5]]], radius=40, strength=1.0, warp_mode="mix")
    assert mix[100, 120, 0] < 0.99 or mix[90, 120, 0] < 0.99


def test_warp_engine_skips_last_substep_like_upstream():
    pix = np.zeros((50, 200, 3), np.float32)
    from lazyroom.ops._draw_warp import _segment

    dx = np.zeros((50, 200), np.float32)
    dy = np.zeros_like(dx)
    _segment(dx, dy, pix, 0, 0, 50.0, 25.0, 150.0, 25.0, 40.0, 0.0, 1.0, 0)
    # step = r/2 = 20 -> 5 sub-steps applied for i = 0..4 (dab centres 50, 70, 90, 110, 130): the last
    # sub-step (130 -> 150) is left to the next segment, so x = 175 (45 px from 130) is untouched
    assert dx[25, 175] == 0 and dx[25, 160] != 0


def test_shape_layer_shapes_and_shadow():
    img = white()
    for shape in ["line", "arrow", "double_arrow", "line_arrow", "double_line_arrow", "rect", "oval", "triangle",
                  "polygon", "star"]:
        out = run("draw.shape_layer", img, shape=shape, width=0.4, height=0.2)
        assert changed(out, img).any(), shape
    plain = run("draw.shape_layer", img, shape="rect", width=0.3, height=0.3, color="#FFFFFF00")
    assert np.allclose(plain, img)
    shad = run("draw.shape_layer", img, shape="rect", width=0.3, height=0.3, color="#FF0000", shadow=True,
               shadow_dx=0.05, shadow_dy=0.05, shadow_blur=0.01)
    # shadow below-right of the red square
    red = (shad[..., 0] > 0.9) & (shad[..., 1] < 0.1)
    ys, xs = np.nonzero(red)
    assert shad[ys.max() + 3, xs.max() + 3, 0] < 0.5


def test_shape_layer_rotation_and_flip():
    img = white()
    a = run("draw.shape_layer", img, shape="arrow", width=0.5, height=0.2)
    f = run("draw.shape_layer", img, shape="arrow", width=0.5, height=0.2, flip_h=True)
    r = run("draw.shape_layer", img, shape="arrow", width=0.5, height=0.2, rotation=180)
    assert np.allclose(f, a[:, ::-1], atol=0.02)  # centred layer: flip == mirror
    assert np.abs(r - f).mean() < 0.01


# ---------------------------------------------------------------------------
# determinism, preview scale, alpha, params
# ---------------------------------------------------------------------------

def test_determinism_all_ops():
    img = synthetic_image(96, 72)
    for op_id, vals in [("draw.paint", {"path_mode": "spray"}), ("draw.paint", {}),
                        ("draw.warp", {"strokes": [PTS], "warp_mode": "swirl_cw"}),
                        ("draw.spot_heal", {"strokes": [PTS]}), ("draw.flood_fill", {}),
                        ("draw.shape_layer", {"shadow": True})]:
        a = run(op_id, img, RenderContext(seed=5), **vals)
        b = run(op_id, img, RenderContext(seed=5), **vals)
        assert np.array_equal(a, b), op_id


@pytest.mark.parametrize("vals", [
    {"width": 16, "softness": 6},
    {"mode": "neon", "width": 10},
    {"mode": "highlighter", "width": 24, "line_style": "dashed", "dash_size": 20, "gap_size": 16},
    {"path_mode": "star", "filled": True},
    {"mode": "privacy_blur", "width": 30},
    {"mode": "text", "text": "Hello", "width": 40},
])
def test_preview_scale_matches_full_render(vals):
    full = synthetic_image(320, 240)
    half = cv2.resize(full, (160, 120), interpolation=cv2.INTER_AREA)
    a = run("draw.paint", full, RenderContext(seed=1), strokes=[PTS], **vals)
    b = run("draw.paint", half, RenderContext(seed=1, scale=0.5), strokes=[PTS], **vals)
    a_small = cv2.resize(a, (160, 120), interpolation=cv2.INTER_AREA)
    assert np.abs(a_small - b).mean() < 0.012


def test_warp_preview_scale():
    full = white(200, 300)
    full[:, 140:160] = (0.1, 0.2, 0.9)
    half = cv2.resize(full, (150, 100), interpolation=cv2.INTER_AREA)
    pts = [[0.45 + 0.01 * i, 0.5] for i in range(11)]
    a = run("draw.warp", full, strokes=[pts], radius=60, strength=0.8)
    b = run("draw.warp", half, RenderContext(scale=0.5), strokes=[pts], radius=60, strength=0.8)
    assert np.abs(cv2.resize(a, (150, 100), interpolation=cv2.INTER_AREA) - b).mean() < 0.02


def test_alpha_images():
    img = synthetic_image(96, 72, alpha=True)
    img[:, :20, 3] = 0.0
    out = run("draw.paint", img, strokes=[[[0.02, 0.5], [0.9, 0.5]]], width=8, path_mode="line", color="#FF0000",
              alpha=0.5)
    assert out.shape == img.shape
    assert np.isclose(out[36, 10, 3], 0.5, atol=1e-3) and np.allclose(out[36, 10, :3], (1, 0, 0), atol=1e-3)
    er = run("draw.paint", img, strokes=[{"points": [[0.02, 0.5], [0.9, 0.5]], "path_mode": "line"},
                                         {"points": [[0.0, 0.5], [1.0, 0.5]], "mode": "eraser", "width": 24}],
             width=8)
    assert np.allclose(er[..., 3], img[..., 3], atol=1e-5)  # erasing a stroke restores the original alpha
    for op_id, vals in [("draw.warp", {"strokes": [PTS]}), ("draw.spot_heal", {"strokes": [PTS]}),
                        ("draw.flood_fill", {}), ("draw.shape_layer", {}),
                        ("draw.paint", {"mode": "pixelation"}), ("draw.paint", {"mode": "privacy_blur"})]:
        o = run(op_id, img, **vals)
        assert o.shape == img.shape and np.isfinite(o).all() and o[..., 3].min() >= 0, op_id


def test_rgb_stays_rgb():
    img = synthetic_image(64, 48)
    for op_id in ["draw.paint", "draw.flood_fill", "draw.shape_layer"]:
        assert run(op_id, img).shape == img.shape


def test_strokes_param_forms():
    spec = get_op("draw.paint")
    one = spec.validate({"strokes": [[0.1, 0.2], [0.3, 0.4]]})["strokes"]
    assert one == [{"points": [(0.1, 0.2), (0.3, 0.4)]}]
    js = spec.validate({"strokes": json.dumps([{"points": [{"x": 0.1, "y": 0.2}], "mode": "neon",
                                                "color": "#FF0000"}])})["strokes"]
    assert js[0]["mode"] == "neon" and js[0]["color"] == (1.0, 0.0, 0.0, 1.0)
    back = spec.params_json(spec.validate({"strokes": js}))["strokes"]
    assert back == [{"points": [[0.1, 0.2]], "mode": "neon", "color": "#FF0000"}]
    json.dumps(spec.describe())


@pytest.mark.parametrize("vals", [
    {"strokes": [{"points": [[0.1, 0.1]], "colour": "#FFF"}]},
    {"strokes": [{"points": []}]},
    {"strokes": [[[0.1, "a"]]]},
    {"strokes": [[[5.0, 0.1]]]},
    {"strokes": "not json"},
    {"strokes": [{"points": [[0.1, 0.1]], "mode": "crayon"}]},
])
def test_invalid_strokes(vals):
    with pytest.raises(LazyRoomError) as e:
        get_op("draw.paint").validate(vals)
    assert e.value.code == ErrorCode.INVALID_PARAMS


@pytest.mark.parametrize("vals", [
    {"mode": "filter_brush"},
    {"mode": "filter_brush", "filter": "no.such_op"},
    {"mode": "filter_brush", "filter": "geometry.rotate"},
    {"mode": "filter_brush", "filter": "adjust.saturation", "filter_params": {"nope": 1}},
    {"mode": "image_brush"},
    {"mode": "text", "path_mode": "spray"},
])
def test_invalid_render_params(vals):
    with pytest.raises(LazyRoomError) as e:
        run("draw.paint", synthetic_image(32, 24), strokes=[PTS], **vals)
    assert e.value.code == ErrorCode.INVALID_PARAMS


def test_empty_strokes_are_noops():
    img = synthetic_image(32, 24)
    for op_id in ["draw.paint", "draw.spot_heal", "draw.warp"]:
        assert np.array_equal(run(op_id, img, strokes=[]), img)


# ---------------------------------------------------------------------------
# rasteriser / stroker
# ---------------------------------------------------------------------------

def test_rasterizer_area_and_nonzero_union():
    sq = np.array([[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]])
    cov, box = R.rasterize([sq], 50, 50)
    assert abs(cov.sum() - 400) < 1e-3
    cov2, _ = R.rasterize([sq, sq + 5], 50, 50)  # overlapping, same winding -> union
    assert abs(cov2.sum() - (400 + 400 - 15 * 15)) < 1e-2 and cov2.max() <= 1.0
    tri = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]])
    c3, _ = R.rasterize([tri + 0.37], 40, 40)
    assert abs(c3.sum() - 200) < 0.2


def test_stroke_outline_area():
    line = np.array([[20.0, 50.0], [80.0, 50.0]])
    for cap, extra in [("butt", 0.0), ("square", 10 * 10), ("round", math.pi * 25)]:
        cov, _ = R.rasterize(R.stroke_outline(line, False, 10, cap, "round"), 100, 100)
        assert abs(cov.sum() - (60 * 10 + extra)) < 2.0, cap
    sq = np.array([[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]])
    cov, _ = R.rasterize(R.stroke_outline(sq, True, 10, "butt", "miter"), 100, 100)
    assert abs(cov.sum() - (70 * 70 - 50 * 50)) < 2.0
    dot, _ = R.rasterize(R.stroke_outline(np.array([[50.0, 50.0]]), False, 10, "round", "round"), 100, 100)
    assert abs(dot.sum() - math.pi * 25) < 1.0
