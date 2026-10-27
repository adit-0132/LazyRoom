"""Core engine tests: colour correctness, pipeline semantics, projects, IO robustness, path policy, goldens."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lazyroom.core import color as C
from lazyroom.core.context import RenderContext
from lazyroom.core.errors import ErrorCode, LazyRoomError
from lazyroom.core.io import load_image, save_image
from lazyroom.core.pipeline import Step, render, steps_from_list
from lazyroom.core.project import Project, SourceRef
from lazyroom.style.metrics import delta_e76
from tests.conftest import synthetic_image

PHOTOS = Path(__file__).resolve().parent / "data" / "photos"
GOLDEN = Path(__file__).resolve().parent / "data" / "golden"


def run(steps, img=None):
    out, _ = render(synthetic_image(96, 72) if img is None else img, steps_from_list(steps), RenderContext(seed=5))
    return out


# --------------------------------------------------------------------------- colour
def test_transfer_functions_roundtrip_and_reference_values():
    x = np.linspace(-0.2, 1.2, 10001, dtype=np.float32)
    assert np.abs(C.linear_to_srgb(C.srgb_to_linear(x)) - x).max() < 2e-6
    assert abs(float(C.srgb_to_linear(np.array([0.5], np.float32))[0]) - 0.21404) < 1e-4
    big = np.random.default_rng(0).random((80, 80, 3), dtype=np.float32)  # exercises the numba path
    assert np.allclose(C.srgb_to_linear(big), C._np_srgb_to_linear(big), atol=1e-6)


def test_oklab_and_lab_reference_values():
    white = np.ones((1, 1, 3), np.float32)
    assert np.allclose(C.rgb_to_oklab(white)[0, 0], [1, 0, 0], atol=1e-4)
    assert np.allclose(C.rgb_to_lab(white)[0, 0], [100, 0, 0], atol=0.05)
    red = np.array([[[1, 0, 0]]], np.float32)
    assert np.allclose(C.rgb_to_oklab(red)[0, 0], [0.62796, 0.22486, 0.12585], atol=2e-4)  # Ottosson
    rnd = np.random.default_rng(1).random((64, 64, 3), dtype=np.float32)
    assert np.abs(C.oklab_to_rgb(C.rgb_to_oklab(rnd)) - rnd).max() < 1e-4


def test_white_balance_is_neutral_at_zero_and_warms_at_positive():
    grey = np.full((4, 4, 3), 0.5, np.float32)
    assert np.allclose(run([{"op": "adjust.white_balance", "params": {}}], grey), grey, atol=1e-5)
    warm = run([{"op": "adjust.white_balance", "params": {"temperature": 40}}], grey)[0, 0]
    assert warm[0] > warm[2]


def test_exposure_doubles_linear_light():
    img = np.full((4, 4, 3), 0.3, np.float32)
    out = run([{"op": "adjust.exposure", "params": {"ev": 1}}], img)
    assert np.allclose(C.srgb_to_linear(out), 2 * C.srgb_to_linear(img), rtol=1e-4)


def test_tone_ops_keep_endpoints():
    ramp = np.repeat(np.linspace(0, 1, 64, dtype=np.float32)[None, :, None], 3, axis=2)
    for step in ({"op": "adjust.contrast", "params": {"amount": 60}},
                 {"op": "adjust.contrast", "params": {"amount": -60}},
                 {"op": "adjust.curves", "params": {"rgb": [[0, 0], [0.3, 0.2], [0.7, 0.8], [1, 1]]}}):
        out = run([step], ramp)
        assert abs(out[0, 0, 0]) < 1e-3 and abs(out[0, -1, 0] - 1) < 1e-3
        assert np.all(np.diff(out[0, :, 0]) >= -1e-5), "tone curve must stay monotone"


# --------------------------------------------------------------------------- pipeline
def test_disabled_opacity_mask_and_groups():
    img = synthetic_image(96, 72)
    e = {"op": "adjust.exposure", "params": {"ev": 1}}
    assert np.allclose(run([{**e, "enabled": False}], img), img)
    assert np.allclose(run([{**e, "opacity": 0.0}], img), img, atol=1e-6)
    radial = run([{**e, "mask": {"shapes": [{"type": "ellipse", "center": [0.5, 0.5], "radius": [0.2, 0.2],
                                             "feather": 0}]}}], img)
    assert np.allclose(radial[0, 0], img[0, 0], atol=1e-6) and radial[36, 48].mean() > img[36, 48].mean()
    group = run([{"op": "group", "steps": [e, {"op": "adjust.invert", "params": {}}], "opacity": 1.0}], img)
    assert np.allclose(group, run([e, {"op": "adjust.invert", "params": {}}], img), atol=1e-6)


def test_geometry_steps_cannot_be_masked():
    with pytest.raises(LazyRoomError) as e:
        Step.from_dict({"op": "geometry.rotate90", "params": {}, "mask": {"shapes": [{"type": "all"}]}})
    assert e.value.code == ErrorCode.INVALID_PARAMS


def test_unknown_params_and_ops_give_suggestions():
    with pytest.raises(LazyRoomError) as e:
        Step.from_dict({"op": "adjust.exposure", "params": {"evv": 1}})
    assert "ev" in str(e.value.details)
    with pytest.raises(LazyRoomError) as e:
        Step.from_dict({"op": "adjust.exposre", "params": {}})
    assert e.value.code == ErrorCode.UNKNOWN_OPERATION and "adjust.exposure" in e.value.details["suggestions"]


def test_render_cache_reuses_prefix():
    from lazyroom.core.pipeline import RenderCache

    cache = RenderCache()
    steps = steps_from_list([{"op": "adjust.exposure", "params": {"ev": 0.5}},
                             {"op": "adjust.saturation", "params": {"saturation": 20}}])
    img = synthetic_image(64, 48)
    render(img, steps, RenderContext(), cache=cache, base_key="k")
    steps[1].params["saturation"] = 30
    _, rep = render(img, steps, RenderContext(), cache=cache, base_key="k")
    assert rep[0].cached and not rep[1].cached


# --------------------------------------------------------------------------- project
def test_project_roundtrip_undo_redo_revert_and_atomicity(tmp_path):
    p = Project.new(SourceRef(path=str(tmp_path / "x.png")))
    p.apply_actions([{"type": "add", "op": "adjust.exposure", "params": {"ev": 0.5}}], actor="agent")
    r = p.apply_actions([{"type": "add", "op": "effect.vignette", "params": {"amount": -20}}])
    vid = r["created_step_ids"][0]
    with pytest.raises(LazyRoomError):  # second action invalid -> nothing applied
        p.apply_actions([{"type": "remove", "step_id": vid}, {"type": "add", "op": "adjust.exposure",
                                                                "params": {"ev": 99}}])
    assert len(p.steps) == 2
    p.undo()
    assert len(p.steps) == 1
    p.redo()
    assert len(p.steps) == 2
    first = p.history.entries[1].id
    p.revert_to(first)
    assert len(p.steps) == 1 and p.history.can_undo
    path = p.save(tmp_path / "proj")
    assert path.endswith(".lazyroom")
    q = Project.load(path)
    assert q.to_dict()["steps"] == p.to_dict()["steps"] and q.history.cursor == p.history.cursor
    data = json.loads(Path(path).read_text())
    data["schema"] = "lazyroom.project/2"
    Path(path).write_text(json.dumps(data))
    with pytest.raises(LazyRoomError) as e:
        Project.load(path)
    assert e.value.code == ErrorCode.SCHEMA_VERSION
    # projects saved before the rename (photomasta.project/1, *.pmproj) still open; re-saving writes *.lazyroom
    data["schema"] = "photomasta.project/1"
    legacy = tmp_path / "old.pmproj"
    legacy.write_text(json.dumps(data))
    old = Project.load(str(legacy))
    assert old.to_dict()["steps"] == p.to_dict()["steps"]
    assert old.save().endswith("old.lazyroom") and legacy.exists()


# --------------------------------------------------------------------------- IO
def test_16bit_png_precision_preserved(tmp_path):
    ramp = np.linspace(0, 1, 4096, dtype=np.float32).reshape(64, 64)
    img = np.dstack([ramp] * 3)
    save_image(img, tmp_path / "a.png", bit_depth=16)
    back, meta = load_image(tmp_path / "a.png")
    assert meta.bit_depth == 16 and np.abs(back - img).max() < 2 / 65535


def test_exif_orientation_applied(tmp_path):
    arr = np.zeros((20, 40, 3), np.uint8)
    arr[:, :20] = 255
    im = Image.fromarray(arr)
    ex = im.getexif()
    ex[0x0112] = 6  # rotate 90 CW on display
    im.save(tmp_path / "o.jpg", exif=ex.tobytes(), quality=95)
    img, meta = load_image(tmp_path / "o.jpg")
    assert img.shape[:2] == (40, 20) and meta.orientation_applied == 6


def test_unicode_paths(tmp_path):
    p = tmp_path / "фото 写真 é.png"
    save_image(synthetic_image(32, 24), p)
    img, _ = load_image(p)
    assert img.shape[:2] == (24, 32)


def test_malformed_inputs(tmp_path):
    (tmp_path / "trunc.jpg").write_bytes((PHOTOS / "rocket.jpg").read_bytes()[:2000])
    (tmp_path / "junk.png").write_bytes(os.urandom(500))
    for name in ("trunc.jpg", "junk.png"):
        with pytest.raises(LazyRoomError) as e:
            load_image(tmp_path / name)
        assert e.value.code in (ErrorCode.DECODE_FAILED, ErrorCode.UNSUPPORTED_FORMAT)
    with pytest.raises(LazyRoomError) as e:
        load_image(tmp_path / "missing.png")
    assert e.value.code == ErrorCode.FILE_NOT_FOUND
    big = Image.new("L", (6000, 6000))
    big.save(tmp_path / "big.png")
    with pytest.raises(LazyRoomError) as e:
        load_image(tmp_path / "big.png", max_pixels=1_000_000)
    assert e.value.code == ErrorCode.RESOURCE_LIMIT


def test_export_formats_and_metadata_policy(tmp_path):
    img = synthetic_image(64, 48, alpha=True)
    from lazyroom.core.io import ImageMeta

    ex = Image.Exif()
    ex[0x8825] = {2: (1.0, 2.0, 3.0)}  # GPS IFD
    ex[0x010F] = "Cam"
    meta = ImageMeta(exif=ex.tobytes())
    for fmt in ("jpg", "png", "webp", "tif", "avif", "bmp", "gif", "qoi"):
        r = save_image(img, tmp_path / f"o.{fmt}", meta=meta, metadata="keep_no_gps")
        back, m = load_image(r["path"])
        assert back.shape[:2] == (48, 64), fmt
    with Image.open(tmp_path / "o.jpg") as im:
        e = im.getexif()
        assert e.get(0x010F) == "Cam" and 0x8825 not in e
    save_image(img, tmp_path / "s.jpg", meta=meta, metadata="strip")
    with Image.open(tmp_path / "s.jpg") as im:
        assert not im.getexif()
    with pytest.raises(LazyRoomError) as e:
        save_image(img, tmp_path / "o.png")
    assert e.value.code == ErrorCode.OUTPUT_EXISTS


# --------------------------------------------------------------------------- path policy
def test_path_policy(tmp_path):
    from lazyroom.app.security import PathPolicy

    pol = PathPolicy([str(tmp_path)], [str(tmp_path)])
    src = tmp_path / "orig.jpg"
    src.write_bytes(b"x")
    pol.protect(src)
    with pytest.raises(LazyRoomError) as e:
        pol.check_write(src, overwrite=True)
    assert e.value.code == ErrorCode.PATH_NOT_ALLOWED
    with pytest.raises(LazyRoomError):
        pol.check_read("/etc/passwd")
    with pytest.raises(LazyRoomError):
        pol.check_write(tmp_path / ".ssh" / "authorized_keys")
    with pytest.raises(LazyRoomError):
        pol.check_write(tmp_path / ".." / ".." / "escape.png")
    assert pol.check_write(tmp_path / "new.png") == (tmp_path / "new.png").resolve()


# --------------------------------------------------------------------------- golden images
GOLDEN_CASES = {
    "warm70s": [{"op": "group", "steps": "look:warm-70s-film"}],
    "teal_orange": [{"op": "group", "steps": "look:teal-orange"}],
    "bw": [{"op": "adjust.black_white", "params": {}}, {"op": "adjust.contrast", "params": {"amount": 30}}],
    "grade": [{"op": "adjust.color_grading", "params": {"shadows_hue": 200, "shadows_saturation": 40,
                                                        "highlights_hue": 40, "highlights_saturation": 30}},
              {"op": "adjust.hsl", "params": {"green_saturation": -50, "orange_luminance": 20}}],
    "geometry": [{"op": "geometry.straighten", "params": {"angle": 4}},
                 {"op": "geometry.resize", "params": {"width": 200, "mode": "long_edge", "method": "lanczos3"}}],
    "itb_mix": [{"op": "filter.sepia", "params": {}}, {"op": "filter.floyd_steinberg_dithering", "params": {}}],
}


def _resolve(steps):
    from lazyroom.presets.store import PresetStore

    out = []
    for s in steps:
        if s.get("op") == "group" and isinstance(s.get("steps"), str):
            out.append({"op": "group", "steps": [x for x in PresetStore().get(s["steps"].split(":")[1])["steps"]
                                                 if x["op"] != "effect.grain"]})
        else:
            out.append(s)
    return out


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_golden_images(name):
    """Renders must stay perceptually identical to the committed goldens (mean ΔE < 0.6, p99 < 3)."""
    img, _ = load_image(PHOTOS / "coffee.png")
    out = np.clip(run(_resolve(GOLDEN_CASES[name]), img), 0, 1)
    gpath = GOLDEN / f"{name}.png"
    if not gpath.exists():  # first run creates the golden (commit it)
        GOLDEN.mkdir(parents=True, exist_ok=True)
        save_image(out, gpath, bit_depth=16)
        pytest.skip(f"golden {name} created")
    ref, _ = load_image(gpath)
    assert ref.shape == out.shape
    de = delta_e76(out[..., :3], ref[..., :3])
    assert float(de.mean()) < 0.6 and float(np.quantile(de, 0.99)) < 3.0
