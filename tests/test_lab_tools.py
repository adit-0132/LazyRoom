"""ImageToolbox analysis labs: pixel-by-pixel compare (ImageDiffTool), compare renders, the animated
before/after GIF, histograms (RGB / Brightness / Camera), image scopes, multi-frame fusion and the
compression lab — see docs/dev/port_reports/color_tools.md."""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest
from PIL import Image

import lazyroom.tools._compression_lab as CL
import lazyroom.tools._fusion as FU
import lazyroom.tools._scopes as sc
import lazyroom.tools.lab_tools  # noqa: F401  (registers the lab tools)
from lazyroom.core.errors import LazyRoomError
from lazyroom.core.io import encode_image, load_image, save_image
from lazyroom.tools import _compare_render as cr
from lazyroom.tools import _histogram as hg
from lazyroom.tools import _image_diff as idf
from tests.conftest import synthetic_image


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("LAZYROOM_HOME", str(tmp_path / "home"))
    from lazyroom.api import LazyRoom

    return LazyRoom()


@pytest.fixture
def pair(tmp_path):
    a = synthetic_image(160, 120)
    a[40:60, 30:90] = 0.8
    b = a.copy()
    b[40:60, 30:90] = 0.0
    pa, pb = tmp_path / "a.png", tmp_path / "b.png"
    save_image(a, pa)
    save_image(b, pb)
    return pa, pb


# ---------------------------------------------------------------------------
# ImageDiffTool
# ---------------------------------------------------------------------------

def test_identical_images_scores():
    a = idf.to_rgba8(synthetic_image(64, 48))
    assert idf.metric(a, a, "SSIM") == pytest.approx(1.0, abs=1e-6)
    assert math.isinf(idf.metric(a, a, "PSNR"))
    assert idf.metric(a, a, "NCC") == pytest.approx(1.0, abs=1e-9)
    assert idf.metric(a, a, "MAE") == 0 and idf.metric(a, a, "RMSE") == 0 and idf.metric(a, a, "AE") == 0


def test_known_error_values():
    a = np.zeros((10, 10, 4), np.uint8)
    a[..., 3] = 255
    b = a.copy()
    b[..., 0] = 10  # one channel of four differs by 10 everywhere
    assert idf.metric(a, b, "MAE") == pytest.approx(2.5)          # mean over RGBA
    assert idf.metric(a, b, "RMSE") == pytest.approx(5.0)         # sqrt(100 / 4)
    assert idf.metric(a, b, "PSNR") == pytest.approx(10 * math.log10(255 ** 2 / 25))
    r = idf.highlight_differences(a, b, "AE", threshold=3)        # 10 > 3 * 2.55
    assert r.score == 100 and r.mask.all() and tuple(r.highlighted[0, 0]) == (255, 0, 0, 255)
    assert idf.highlight_differences(a, b, "AE", threshold=4).score == 0  # 10 <= 10.2


def test_resize_and_pad_matches_opencv_utils():
    src = np.full((50, 100, 4), 200, np.uint8)
    out = idf.resize_and_pad(src, 80, 80)          # wider than target: width fits, height = int(80 / 2)
    assert out.shape == (80, 80, 4) and (out[:40] == 200).all() and (out[40:] == 0).all()


def test_compare_tool_pixel_by_pixel(api, pair, tmp_path):
    pa, pb = pair
    res = api.tool_run("compare", {"a": str(pa), "b": str(pb), "pixel_metric": "all",
                                   "highlight_output": str(tmp_path / "hl.png")})
    assert {"metrics", "size_a", "size_b"} <= set(res)  # original keys kept
    m = res["pixel_by_pixel"]["methods"]
    assert set(m) == {"ssim", "ae", "mae", "ncc", "psnr", "rmse"}
    assert m["ae"]["score"] == 20 * 60 and 0 < m["ssim"]["score"] < 1 and m["psnr"]["score"] > 10
    hl = np.asarray(Image.open(tmp_path / "hl.png"))
    assert (hl[50, 60] == [255, 0, 0]).all() and not (hl[5, 5] == [255, 0, 0]).all()
    json.dumps(res, allow_nan=False)
    same = api.tool_run("compare", {"a": str(pa), "b": str(pa), "pixel_metric": "psnr"})
    assert same["pixel_by_pixel"]["methods"]["psnr"] == {"score": None, "infinite": True, "highlighted_pixels": 0,
                                                        "highlighted_fraction": 0.0}


def test_compare_default_unchanged(api, pair):
    pa, pb = pair
    res = api.tool_run("compare", {"a": str(pa), "b": str(pb)})
    assert "pixel_by_pixel" not in res and res["metrics"]["identical"] is False


# ---------------------------------------------------------------------------
# compare renders and GIF
# ---------------------------------------------------------------------------

def test_animation_params_match_kotlin_tests():
    for d in range(2, 9):
        fr = cr.AnimationParams(duration_seconds=d).frames()
        assert sum(t for _, t in fr) == d * 1000
        assert [p for p, t in fr if t == 500] == [0.0, 1.0]
        assert all(0 <= p <= 1 for p, _ in fr) and all(t % 10 == 0 for _, t in fr)
    p = cr.AnimationParams()
    assert p.position_at(0) == 0 and p.position_at(499) == 0 and p.position_at(2000) == 1 and p.position_at(4000) == 0
    for t in range(0, 4001, 50):
        assert p.position_at(t) + p.position_at(t + 2000) == pytest.approx(1.0, abs=1e-5)
    assert cr.output_size(4000, 2000) == (2048, 1024) and cr.output_size(120, 80) == (120, 80)
    assert cr.output_size(2 ** 31 - 1, 1) == (2048, 1) and cr.output_size(8000, 4000, 720) == (720, 360)
    assert sum(t for _, t in cr.AnimationParams(duration_seconds=-5).frames()) == 2000


def test_compare_animation_gif(api, pair, tmp_path):
    pa, pb = pair
    out = tmp_path / "ba.gif"
    res = api.tool_run("compare_animation", {"before": str(pa), "after": str(pb), "output": str(out)})
    assert res["frame_count"] == 62 and res["duration_ms"] == 4000
    im = Image.open(out)
    assert im.n_frames == 62 and im.size == (160, 120) and im.info.get("loop") == 0
    total = 0
    for i in range(im.n_frames):
        im.seek(i)
        total += im.info["duration"]
    assert total == 4000
    im.seek(0)
    first = np.asarray(im.convert("RGB")).astype(int)
    assert abs(first[100, 10] - np.array([255, 0, 0])).max() < 40  # before image (red patch) at position 0
    v = api.tool_run("compare_animation", {"before": str(pa), "after": str(pb), "output": str(tmp_path / "v.gif"),
                                           "direction": "vertical", "duration_seconds": 2, "show_labels": False})
    assert v["frame_count"] == 22 and all(f["bounds"][0] == 0 and f["bounds"][2] == 160 for f in v["frames"])


@pytest.mark.parametrize("mode", ["slide", "side_by_side", "transparency", "pixel_by_pixel"])
def test_compare_visual_modes(api, pair, tmp_path, mode):
    pa, pb = pair
    res = api.tool_run("compare_visual", {"before": str(pa), "after": str(pb), "mode": mode,
                                          "output": str(tmp_path / f"{mode}.png")})
    img = np.asarray(Image.open(res["output"]["path"]).convert("RGB")).astype(int)
    if mode == "slide":  # left half = before, right half = after
        assert res["split_x"] == 80 and img[50, 40].tolist() != [0, 0, 0] and img[50, 85].tolist() == [0, 0, 0]
    if mode == "side_by_side":
        assert img.shape[1] > 320
    if mode == "pixel_by_pixel":
        assert res["score"] == 1200


def test_compare_visual_toggle_gif(api, pair, tmp_path):
    pa, pb = pair
    res = api.tool_run("compare_visual", {"before": str(pa), "after": str(pb), "mode": "toggle",
                                          "output": str(tmp_path / "t.gif")})
    assert Image.open(res["output"]["path"]).n_frames == 2


# ---------------------------------------------------------------------------
# histograms
# ---------------------------------------------------------------------------

def test_itb_histogram_bins_and_brightness():
    img = np.zeros((10, 20, 3), np.float32)
    img[:, :10] = 1.0  # white half, black half
    h = hg.itb_histogram(img)
    assert len(h["red"]) == 256 and len(h["brightness"]) == 101
    assert h["red"][255] == 100 and h["red"][0] == 100 and h["brightness"][100] == 100 and h["brightness"][0] == 100
    grey = np.full((4, 4, 3), 119 / 255, np.float32)  # sRGB 119 -> L* ~50
    assert np.argmax(hg.itb_histogram(grey)["brightness"]) in (49, 50)


def test_histogram_tool_keeps_keys_and_renders(api, pair, tmp_path):
    pa, _ = pair
    res = api.tool_run("histogram", {"input": str(pa)})
    assert {"red", "green", "blue", "luma", "bins"} <= set(res) and "itb" not in res
    for t in ("rgb", "brightness", "camera"):
        r = api.tool_run("histogram", {"input": str(pa), "itb_histogram": True, "render_type": t,
                                       "render_output": str(tmp_path / f"h_{t}.png")})
        assert sum(r["itb"]["red"]) == 160 * 120 and r["render"]["width"] == 1000
        assert Image.open(tmp_path / f"h_{t}.png").size == (1000, 500)


def _scope_solid(rgb255, n=4):
    return np.full((n, n, 3), np.array(rgb255, np.float32) / 255.0, np.float32)


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------

def test_waveform_of_grey_ramp_is_a_diagonal():
    w = 256
    ramp = np.repeat(np.repeat(np.linspace(0, 1, w, dtype=np.float32)[None, :, None], 16, 0), 3, 2)
    d = sc.waveform_data(ramp, 256)
    assert d["size"] == 256 and d["data"].shape == (256, 256, 4)
    occupied = np.argwhere(d["counts"][0] > 0)  # (row, col)
    assert len(occupied) == 256
    assert np.all(occupied[:, 0] == 255 - occupied[:, 1])  # level rises left -> right
    res = sc.waveform(ramp)
    assert res["columns"][0]["red"]["p50"] < 0.2 < 0.8 < res["columns"][-1]["red"]["p50"]
    assert res["clipping"]["red"]["black"] == pytest.approx(1 / w, abs=1e-6) and res["max_bin_count"]["green"] == 16


def test_vectorscope_of_pure_red_sits_at_red_angle():
    red = _scope_solid((255, 0, 0), 8)
    d = sc.vectorscope_data(red, 384)
    ys, xs = np.nonzero(d["counts"])
    assert len(xs) == 1 and ys[0] == 0  # Cr clamps to 1 -> top row
    assert tuple(d["data"][ys[0], xs[0]]) == (255, 0, 0, 255)
    res = sc.vectorscope(red)
    assert res["peak"]["angle_deg"] == pytest.approx(103.0, abs=0.5)  # atan2(0.5, -0.1146)
    assert res["hue_groups"][0]["fraction"] == 1.0 and res["neutral_fraction"] == 0
    grey = sc.vectorscope(_scope_solid((128, 128, 128)))
    assert grey["neutral_fraction"] == 1.0 and grey["saturation"]["mean"] < 0.01


def test_histogram_data_and_brightness_bins():
    img = np.concatenate([_scope_solid((255, 255, 255)), _scope_solid((0, 0, 0)), _scope_solid((119, 119, 119))], 1)
    h = sc.histogram_data(img)
    assert h["red"][255] == 16 and h["red"][0] == 16 and h["brightness"].sum() == 48
    assert h["brightness"][100] == 16 and h["brightness"][0] == 16 and h["brightness"][50] == 16  # L*(119) = 50.0


def test_image_scopes_tool(api, photo_path, tmp_path):
    res = api.tool_run("image_scopes", {"input": str(photo_path), "output_dir": str(tmp_path / "scopes"),
                                        "render_width": 200, "render_height": 160})
    for k in ("waveform", "vectorscope", "histogram"):
        r = res[k]["render"]
        assert (tmp_path / "scopes" / f"photo_{k}.png").exists() and (r["width"], r["height"]) == (200, 160)
    assert res["waveform"]["size"] == 256 and res["vectorscope"]["size"] == 384
    assert "intensity" not in res["waveform"]
    json.dumps(res)
    one = api.tool_run("image_scopes", {"input": str(photo_path), "scope": "vectorscope", "resolution": 32,
                                        "include_data": True, "background": "#00000000",
                                        "vectorscope_output": str(tmp_path / "v.png")})
    assert one["vectorscope"]["size"] == 64 and len(one["vectorscope"]["intensity"]) == 64
    assert set(one) >= {"vectorscope"} and "waveform" not in one
    from lazyroom.core.io import load_image

    v, _ = load_image(tmp_path / "v.png")
    assert v.shape[2] == 4 and v[..., 3].max() > 0.5 and v[0, 0, 3] == 0


def test_scope_renders_draw_data():
    img = synthetic_image(120, 90)
    wf = sc.render_waveform(sc.waveform_data(img), 256, 256)
    assert wf.shape == (256, 256, 3) and wf.max() > 0.3
    vs = sc.render_vectorscope(sc.vectorscope_data(img), 256, 256)
    assert vs.shape == (256, 256, 3) and vs[128, 128].max() > 0  # graticule crosses the centre
    hs = sc.render_histogram(sc.histogram_data(img)["brightness"], sc.LUMA_CURVE, 300, 100)
    assert hs.shape == (100, 300, 3) and hs[-1].mean() > 0


# =============================================================================
# Multi-frame fusion + compression lab
# =============================================================================

def _solid(rgb, w=48, h=32):
    a = np.zeros((h, w, 3), np.float32)
    a[:] = np.array(rgb, np.float32) / 255
    return a


def _f(u8):
    return u8.astype(np.float32) / 255


# ---------------------------------------------------------------------------
# Multi-frame fusion (MultiFrameFusionEngineTest / FusionParamsTest equivalents)
# ---------------------------------------------------------------------------

def test_fusion_params_normalized_like_imagetoolbox():
    n = FU.FusionParams(contrast_weight=-1, saturation_weight=3, exposure_weight=4, focus_radius=8,
                        focus_strength=20, light_trail_threshold=2, trail_strength=-1, trail_persistence=4).normalized()
    assert (n.contrast_weight, n.saturation_weight, n.exposure_weight) == (0.0, 2.0, 2.0)
    assert n.focus_radius == 9 and n.focus_strength == 6.0
    assert (n.light_trail_threshold, n.trail_strength, n.trail_persistence) == (0.5, 0.0, 1.0)
    assert FU.FusionParams(focus_radius=21).normalized().focus_radius == 21
    assert FU.FusionParams(focus_radius=0).normalized().focus_radius == 1


def test_median_removes_outlier_frame():
    n, o = _solid((24, 96, 168)), _solid((240, 16, 32))
    out, _ = FU.fuse([n, o, n], FU.FusionParams(mode="median", align_images=False))
    assert out.shape == (32, 48, 3) and np.abs(out[0, 0].astype(int) - [24, 96, 168]).max() <= 1


def test_long_exposure_averages_in_linear_light():
    out, _ = FU.fuse([_solid((0, 0, 0)), _solid((255, 255, 255))], FU.FusionParams(mode="long_exposure",
                                                                                   align_images=False))
    assert 184 <= out[0, 0, 0] <= 188  # 255 * 0.5^(1/2.2)


def test_light_and_motion_trails():
    bg = (20, 24, 28)
    f2, f3 = _solid(bg), _solid(bg)
    f2[8:20, 4:14] = (1, 0, 0)
    f3[8:20, 28:38] = (0, 0, 1)
    out, _ = FU.fuse([_solid(bg), f2, f3], FU.FusionParams(mode="light_trails", align_images=False,
                                                           light_trail_threshold=0.05, trail_strength=1))
    assert np.abs(out[4, 20].astype(int) - bg).max() <= 2
    assert out[12, 8, 0] > 180 and out[12, 32, 2] > 180
    frames = []
    for x in (4, 20, 36):
        f = _solid((8, 8, 8))
        f[8:20, x:x + 6] = 1
        frames.append(f)
    out, _ = FU.fuse(frames, FU.FusionParams(mode="motion_trails", align_images=False, trail_strength=1,
                                             trail_persistence=0.5))
    old, mid, new = out[12, 6, 0], out[12, 22, 0], out[12, 38, 0]
    assert 8 < old < mid < new


def _half_sharp(sharp_left):
    y, x = np.mgrid[0:64, 0:96]
    sharp = (x < 48) if sharp_left else (x >= 48)
    v = np.where(sharp, np.where((x + y) % 2 == 0, 24, 232), 128).astype(np.uint8)
    return np.repeat(v[..., None], 3, 2)


def _gradient(w=96, h=64):
    y, x = np.mgrid[0:h, 0:w]
    return np.dstack([20 + x * 180 // w, 25 + y * 170 // h, 30 + (x + y) * 160 // (w + h)]).astype(np.uint8)


def _brightness(img, m):
    return np.clip((img.astype(np.float32) * m).astype(int), 0, 255).astype(np.uint8)


def test_focus_stack_keeps_sharp_halves_and_exposure():
    out, _ = FU.fuse([_f(_half_sharp(True)), _f(_half_sharp(False))],
                     FU.FusionParams(mode="focus", align_images=False, focus_radius=3, focus_strength=4))
    assert abs(int(out[20, 10, 0]) - int(out[20, 11, 0])) > 80
    assert abs(int(out[20, 82, 0]) - int(out[20, 83, 0])) > 80
    base = _gradient()
    out, _ = FU.fuse([_f(_brightness(base, 0.4)), _f(base), _f(_brightness(base, 1.8))],
                     FU.FusionParams(mode="focus", align_images=False, focus_radius=3, focus_strength=4))
    lum = lambda im: (im.astype(int).sum(2) // 3)  # noqa: E731
    assert abs(lum(out).mean() - lum(base).mean()) < 15
    assert np.abs(np.diff(lum(out).mean(1))).max() < 10


def test_exposure_fusion_of_brackets():
    base = _gradient()
    out, info = FU.fuse([_f(_brightness(base, 0.45)), _f(base), _f(_brightness(base, 1.7))],
                        FU.FusionParams(mode="exposure", align_images=False))
    assert out.shape == base.shape and 20 < out.mean() < 235
    assert info["params"]["mode"] == "exposure"


def _feature_image(w=320, h=240):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (24, 28, 36)
    for i in range(70):
        c = (40 + i * 37 % 200, 40 + i * 71 % 200, 40 + i * 97 % 200)
        x, y = i * 83 % (w - 24), i * 47 % (h - 24)
        if i % 2 == 0:
            cv2.circle(img, (x + 8, y + 8), 4 + i % 7, c, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x, y), (x + 7 + i % 13, y + 7 + i % 11), c, -1)
    return img


def test_alignment_recovers_shift_and_crops_to_overlap():
    base = _feature_image()
    shifted = np.zeros_like(base)
    shifted[6:, 9:] = base[:-6, :-9]
    out, info = FU.fuse([_f(base), _f(shifted)], FU.FusionParams(mode="median"))
    rec = info["alignment"][1]
    assert rec["aligned"] and rec["matches"] >= FU.MIN_FEATURE_MATCHES
    assert abs(rec["translation"][0] + 9) < 1.5 and abs(rec["translation"][1] + 6) < 1.5
    assert 240 <= out.shape[1] <= 320 and 180 <= out.shape[0] <= 240
    assert info["crop"] == {"x": 0, "y": 0, "width": out.shape[1], "height": out.shape[0]}


def test_largest_valid_rectangle():
    m = np.full((40, 50), 255, np.uint8)
    m[:5] = 0
    m[:, 45:] = 0
    m[20, 10] = 0
    assert FU.largest_valid_rectangle(m) == (11, 5, 34, 35)
    assert FU.largest_valid_rectangle(np.zeros((4, 4), np.uint8)) is None


def test_fusion_tool_reduces_noise(api, tmp_path):
    clean = synthetic_image(96, 72, seed=1)
    rng = np.random.default_rng(3)
    files = []
    for i in range(7):
        p = tmp_path / f"n{i}.png"
        save_image(np.clip(clean + rng.normal(0, 0.08, clean.shape).astype(np.float32), 0, 1), p)
        files.append(str(p))
    single = load_image(files[0])[0]
    rmse = lambda a: float(np.sqrt(((a[..., :3] - clean) ** 2).mean()))  # noqa: E731
    for mode in ("median", "long_exposure"):
        res = api.tool_run("multi_frame_fusion", {"inputs": files, "mode": mode, "align_images": False,
                                                  "output": str(tmp_path / f"{mode}.png")})
        fused = load_image(res["output"]["path"])[0]
        assert res["frames"] == 7 and fused.shape[:2] == clean.shape[:2]
        assert rmse(fused) < 0.6 * rmse(single)
        json.dumps(res)


def test_fusion_tool_every_mode_and_limits(api, tmp_path):
    files = []
    for i in range(13):
        p = tmp_path / f"f{i}.png"
        save_image(synthetic_image(64, 48, seed=i), p)
        files.append(str(p))
    for mode in FU.FUSION_MODES:
        res = api.tool_run("multi_frame_fusion", {"inputs": files[:3], "mode": mode,
                                                  "output": str(tmp_path / f"o_{mode}.png")})
        assert res["mode"] == mode and res["output"]["width"] > 0 and len(res["alignment"]) == 3
        json.dumps(res)
    res = api.tool_run("multi_frame_fusion", {"inputs": files + files[:2], "mode": "median", "align_images": False,
                                              "output": str(tmp_path / "cap.png")})
    assert res["frames"] == FU.MAX_IMAGES and res["ignored_inputs"] == [files[12]]
    with pytest.raises(LazyRoomError):
        api.tool_run("multi_frame_fusion", {"inputs": [files[0], files[0]], "output": str(tmp_path / "x.png")})


# ---------------------------------------------------------------------------
# Compression lab
# ---------------------------------------------------------------------------

def test_human_file_size():
    assert CL.human_file_size(1023) == "1023 B"
    assert CL.human_file_size(1536) == "1.5 kB"
    assert CL.human_file_size(5 * 1024 * 1024) == "5.0 MB"


def test_resolve_formats_order_and_aliases():
    fm = CL.resolve_formats(["avif", "JPG", "WEBP Lossy", "png"])
    assert [f.id for f in fm] == ["jpg", "png_lossless", "webp_lossy", "avif_lossy"]
    with pytest.raises(LazyRoomError):
        CL.resolve_formats(["nope"])


def test_lab_lossless_and_unavailable(api, photo_path):
    res = api.tool_run("compression_lab", {"input": str(photo_path),
                                           "formats": ["png_lossless", "jpg", "jxl_lossy", "qoi"], "quality": 100})
    json.dumps(res, allow_nan=False)
    rows = {r["format"]: r for r in res["results"]}
    assert [r["format"] for r in res["results"]] == ["jpg", "png_lossless", "qoi"]
    for k in ("png_lossless", "qoi"):
        assert rows[k]["ssim"] == 1.0 and rows[k]["psnr"] is None and rows[k]["psnr_infinite"]
        assert rows[k]["quality_value"] is None and rows[k]["encoder_setting"] == "fixed settings"
    assert rows["jpg"]["quality_value"] == 100 and rows["jpg"]["ssim"] > 0.98 and rows["jpg"]["psnr"] > 40
    assert rows["jpg"]["size_bytes"] > 0 and rows["jpg"]["encoding_time_ms"] >= 0
    assert [f["format"] for f in res["failed"]] == ["jxl_lossy"] and res["failed"][0]["reason"]


def test_lab_target_size_binary_search(api, photo_path):
    img = load_image(photo_path)[0]
    target_kb = 8
    res = api.tool_run("compression_lab", {"input": str(photo_path), "formats": ["jpg", "webp_lossy"],
                                           "mode": "target_size", "target_size_kb": target_kb})
    for row in res["results"]:
        assert row["target_satisfied"] and row["size_bytes"] <= target_kb * 1024
        q = row["quality_value"]
        if q < 100:  # highest quality under the target: next quality must not fit (binary search on size)
            ext = ".jpg" if row["format"] == "jpg" else ".webp"
            nxt = encode_image(CL.to_uint8(img).astype(np.float32) / 255, ext, quality=q + 1, metadata="strip")
            assert len(nxt) > target_kb * 1024
    tiny = api.tool_run("compression_lab", {"input": str(photo_path), "formats": ["jpg", "png_lossless"],
                                            "mode": "target_size", "target_size_kb": 1})
    rows = {r["format"]: r for r in tiny["results"]}
    assert rows["jpg"]["quality_value"] == 0 and not rows["jpg"]["target_satisfied"]
    assert not rows["png_lossless"]["target_satisfied"]


def test_lab_target_quality_binary_search(api, photo_path, tmp_path):
    res = api.tool_run("compression_lab", {"input": str(photo_path), "formats": ["jpg", "jp2"],
                                           "mode": "target_quality", "target_quality": 90,
                                           "output_dir": str(tmp_path / "out")})
    assert res["target_ssim"] == 0.9
    for row in res["results"]:
        assert row["target_satisfied"] and row["ssim"] >= 0.9
        assert (tmp_path / "out" / row["output"].split("/")[-1]).stat().st_size == row["size_bytes"]
    jp2 = next(r for r in res["results"] if r["format"] == "jp2")
    assert 20 <= jp2["quality_value"] <= 100
    assert res["encodes"] <= 2 * 7
