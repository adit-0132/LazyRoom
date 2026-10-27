"""AI subsystem: catalogue integrity, op registration / unavailability, tiler correctness with
synthetic ONNX models, downloads, and (slow) real inference with small downloaded models."""
from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")

from lazyroom.ai import models, processors, runtime  # noqa: E402
from lazyroom.core.context import CancelToken, RenderContext  # noqa: E402
from lazyroom.core.errors import ErrorCode, LazyRoomError  # noqa: E402
from lazyroom.core.registry import get_op, list_ops  # noqa: E402

AI_HOME = os.environ.get("LAZYROOM_AI_TEST_HOME", "/tmp/pm_ai_home")
ITB_TYPES = {"UPSCALE", "REMOVE_BG", "COLORIZE", "DE_JPEG", "DENOISE", "ARTIFACTS", "ENHANCE", "ANIME", "SCANS",
             "STYLE_TRANSFER", "DEPTH", "AI_DETECTION"}
OPS = {"ai.upscale": "upscale", "ai.remove_background": "remove_background", "ai.colorize": "colorize",
       "ai.dejpeg": "dejpeg", "ai.denoise": "denoise", "ai.remove_artifacts": "remove_artifacts",
       "ai.enhance": "enhance", "ai.restore_anime": "restore_anime", "ai.restore_scans": "restore_scans",
       "ai.style_transfer": "style_transfer", "ai.depth_map": "depth", "ai.inpaint": "inpaint",
       "ai.remove_watermark": "remove_watermark"}


# ---------------------------------------------------------------------------
# Minimal ONNX protobuf writer (the 'onnx' package is not a dependency)
# ---------------------------------------------------------------------------
def _varint(n: int) -> bytes:
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _f(num: int, wire: int, payload: bytes | int) -> bytes:
    key = _varint((num << 3) | wire)
    if wire == 0:
        return key + _varint(int(payload))
    if wire == 5:
        return key + payload  # type: ignore[operator]
    return key + _varint(len(payload)) + payload  # type: ignore[arg-type]


def _s(num: int, text: str) -> bytes:
    return _f(num, 2, text.encode())


FLOAT, FLOAT16 = 1, 10


def _value_info(name: str, elem: int, dims) -> bytes:
    shape = b"".join(_f(1, 2, _f(1, 0, d) if isinstance(d, int) else _s(2, d)) for d in dims)
    tensor = _f(1, 0, elem) + _f(2, 2, shape)
    return _s(1, name) + _f(2, 2, _f(1, 2, tensor))


def _attr(name: str, value) -> bytes:
    if isinstance(value, str):
        return _s(1, name) + _f(20, 0, 3) + _f(4, 2, value.encode())
    if isinstance(value, int):
        return _s(1, name) + _f(20, 0, 2) + _f(3, 0, value)
    return _s(1, name) + _f(20, 0, 7) + b"".join(_f(8, 0, v) for v in value)  # INTS


def _node(op: str, inputs, outputs, **attrs) -> bytes:
    return (b"".join(_s(1, i) for i in inputs) + b"".join(_s(2, o) for o in outputs) + _s(4, op)
            + b"".join(_f(5, 2, _attr(k, v)) for k, v in attrs.items()))


def _init(name: str, arr: np.ndarray) -> bytes:
    arr = np.asarray(arr, np.float32)
    return (b"".join(_f(1, 0, d) for d in arr.shape) + _f(2, 0, FLOAT) + _s(8, name)
            + _f(9, 2, arr.tobytes()))


def make_model(path: Path, nodes, inputs, outputs, inits=()) -> Path:
    graph = (b"".join(_f(1, 2, n) for n in nodes) + _s(2, "g") + b"".join(_f(5, 2, i) for i in inits)
             + b"".join(_f(11, 2, _value_info(*i)) for i in inputs)
             + b"".join(_f(12, 2, _value_info(*o)) for o in outputs))
    model = _f(1, 0, 8) + _s(2, "lazyroom-test") + _f(7, 2, graph) + _f(8, 2, _s(1, "") + _f(2, 0, 13))
    path.write_bytes(model)
    return path


DYN = ["n", 3, "h", "w"]


def identity_model(tmp: Path, name="fake_identity.onnx", elem=FLOAT, dims=DYN) -> Path:
    return make_model(tmp / name, [_node("Identity", ["input"], ["output"])],
                      [("input", elem, dims)], [("output", elem, dims)])


def upscale2_model(tmp: Path) -> Path:
    return make_model(tmp / "fake_x2.onnx",
                      [_node("Resize", ["input", "", "scales"], ["output"], mode="nearest",
                             coordinate_transformation_mode="asymmetric", nearest_mode="floor")],
                      [("input", FLOAT, DYN)], [("output", FLOAT, ["n", 3, "h2", "w2"])],
                      [_init("scales", np.array([1, 1, 2, 2], np.float32))])


def tile_mean_model(tmp: Path) -> Path:
    """Output = per-tile channel mean broadcast over the tile (makes tile seams measurable)."""
    return make_model(tmp / "fake_mean.onnx",
                      [_node("ReduceMean", ["input"], ["m"], axes=[2, 3], keepdims=1),
                       _node("Sub", ["input", "input"], ["z"]), _node("Add", ["z", "m"], ["output"])],
                      [("input", FLOAT, DYN)], [("output", FLOAT, DYN)])


def strength_model(tmp: Path) -> Path:
    return make_model(tmp / "fake_strength.onnx", [_node("Mul", ["input", "strength"], ["output"])],
                      [("input", FLOAT, DYN), ("strength", FLOAT, [1, 1])], [("output", FLOAT, DYN)])


@pytest.fixture
def empty_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LAZYROOM_HOME", str(tmp_path / "home"))
    runtime.clear_sessions()
    return tmp_path


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
def test_catalog_integrity():
    cat = json.loads(models.CATALOG_PATH.read_text(encoding="utf-8"))
    ms = cat["models"]
    names = [m["name"] for m in ms]
    assert len(names) == len(set(names)), "duplicate model names"
    itb = [m for m in ms if m["in_itb_catalog"]]
    assert len(itb) == 131  # every NeuralModel entry of ImageToolbox
    assert {m["type"] for m in itb} == ITB_TYPES
    for m in ms:
        assert m["task"] in models.TASKS, m["name"]
        assert re.fullmatch(r"[0-9a-f]{64}", m["sha256"]), m["name"]
        assert m["url"].startswith("https://huggingface.co/T8RIN/imagetoolbox-models/resolve/main/")
        assert m["url"].endswith(m["name"]) and m["size"] > 0
        assert m["title"] and m["license"] and isinstance(m["restricted"], bool), m["name"]
        assert "upstream_project" in m and "license_source" in m
        if m["license"].lower().startswith("unknown"):
            assert m["restricted"], f"{m['name']}: unknown licence must be restricted"
        if re.search(r"(^|-)NC(-|$)", m["license"]):
            assert m["restricted"], m["name"]
        if m.get("pack"):
            assert m["name"].endswith(".zip")
        for dep in m.get("requires_models", []):
            assert dep in names
    for m in itb:
        assert m["speed"]["tier"] in {"VeryFast", "Fast", "Normal", "Slow", "VerySlow"}
    for task, name in cat["task_defaults"].items():
        assert name in names and models.MODELS[name]["task"] == task
    assert set(cat["task_defaults"]) == set(models.TASKS)
    for m in ms:
        if m["type"] == "AI_DETECTION":
            assert m["ai_detection"]["output"]["kind"] in ("softmax", "sigmoid")


def test_resolve_names():
    assert models.resolve_name("u2netp") == "u2netp.onnx"
    assert models.resolve_name("U2NETP.ONNX") == "u2netp.onnx"
    assert models.resolve_name("@upscale") == models.default_model("upscale")
    with pytest.raises(LazyRoomError) as e:
        models.resolve_name("no-such-model")
    assert e.value.code == ErrorCode.INVALID_PARAMS


def test_list_models_shape(empty_home):
    rows = models.list_models()
    assert len(rows) == len(models.MODELS)
    assert all(not r["available"] for r in rows)
    assert {"name", "task", "license", "restricted", "size_mb", "source", "available"} <= set(rows[0])


# ---------------------------------------------------------------------------
# Operations / tools
# ---------------------------------------------------------------------------
def test_ops_registered():
    ids = {s.id for s in list_ops(category="ai")}
    assert set(OPS) <= ids
    for op_id, task in OPS.items():
        spec = get_op(op_id)
        assert spec.category == "ai" and "onnxruntime" in spec.requires
        model = next(p for p in spec.params if p.name == "model")
        assert list(model.choices) == models.models_for_task(task)
        assert model.default == models.default_model(task)
        json.dumps(spec.describe())
    assert get_op("ai.upscale").geometry is True


def test_ops_report_backend_unavailable(empty_home):
    img = np.random.default_rng(0).random((32, 48, 3), dtype=np.float32)
    for op_id, task in OPS.items():
        spec = get_op(op_id)
        ok, hint = spec.availability()
        assert not ok and "lazyroom models download" in hint
        with pytest.raises(LazyRoomError) as e:
            spec.run(img, RenderContext(), spec.validate({"style": "x.png"} if task == "style_transfer" else {}))
        assert e.value.code == ErrorCode.BACKEND_UNAVAILABLE
        # the op itself (past the registry check) names the exact model + command
        vals = spec.validate({"style": "x.png"} if task == "style_transfer" else {})
        with pytest.raises(LazyRoomError) as e:
            spec.fn(img, RenderContext(), **vals)
        assert e.value.code == ErrorCode.BACKEND_UNAVAILABLE
        assert f"lazyroom models download {vals['model']} --yes" in e.value.hint


def test_tools_and_subject_mask_unavailable(empty_home):
    from lazyroom.ai.segmentation import subject_mask
    from lazyroom.tools.registry import get_tool

    for name in ("detect_ai_generated", "ocr_paddle"):
        ok, hint = get_tool(name).availability()
        assert not ok and "models download" in hint
    with pytest.raises(LazyRoomError) as e:
        subject_mask(np.zeros((8, 8, 3), np.float32), RenderContext())
    assert e.value.code == ErrorCode.BACKEND_UNAVAILABLE
    assert "lazyroom models download" in e.value.hint


# ---------------------------------------------------------------------------
# Tiler (port of TileGrid / AiProcessor.drawTile)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("w,h,limit,overlap", [(1000, 700, 512, 16), (513, 512, 512, 16), (2048, 129, 128, 32),
                                               (77, 300, 64, 16)])
def test_tile_grid_upstream_formula(w, h, limit, overlap):
    g = runtime.TileGrid.from_size(w, h, limit, overlap)
    step = limit - overlap
    assert g.columns == (1 if w <= limit else -(-(w - overlap) // step))
    assert g.rows == (1 if h <= limit else -(-(h - overlap) // step))
    cover = np.zeros((h, w), int)
    for t in g.tiles():
        assert t.width <= limit and t.height <= limit and t.width > 0 and t.height > 0
        cover[t.y:t.y + t.height, t.x:t.x + t.width] += 1
    assert cover.min() >= 1


@pytest.mark.parametrize("shape,chunk,overlap", [((37, 53), 128, 16), ((300, 257), 128, 32), ((129, 700), 256, 16),
                                                 ((520, 515), 512, 16), ((64, 64), 128, 16)])
def test_tiled_identity_reconstructs(tmp_path, shape, chunk, overlap):
    p = identity_model(tmp_path)
    rng = np.random.default_rng(sum(shape))
    img = rng.random((*shape, 4), dtype=np.float32)
    out = runtime.process_image("fake_identity.onnx", p, img, RenderContext(), chunk_size=chunk, overlap=overlap)
    assert out.shape == img.shape
    np.testing.assert_allclose(out, img, atol=1e-6)


def test_tiled_x2_nearest_upscale(tmp_path):
    p = upscale2_model(tmp_path)
    img = np.random.default_rng(1).random((150, 211, 3), dtype=np.float32)
    out = runtime.process_image("fake_x2.onnx", p, img, RenderContext(), chunk_size=128, overlap=16)
    ref = np.repeat(np.repeat(img, 2, 0), 2, 1)
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, atol=1e-6)
    rgba = np.dstack([img, np.linspace(0, 1, img.size // 3, dtype=np.float32).reshape(img.shape[:2])])
    out4 = runtime.process_image("fake_x2.onnx", p, rgba, RenderContext(), chunk_size=128, overlap=16)
    np.testing.assert_allclose(out4[..., 3], np.repeat(np.repeat(rgba[..., 3], 2, 0), 2, 1))


def test_overlap_blend_is_smoothstep(tmp_path):
    """Two tiles with different constant outputs: the left band of tile 2 must follow ITB's smoothStep ramp."""
    p = tile_mean_model(tmp_path)
    img = np.zeros((100, 200, 3), np.float32)
    img[:, 100:] = 1.0  # tile 0 = [0,128) mean 0.21875, tile 1 = [112,200) mean 1
    out = runtime.process_image("fake_mean.onnx", p, img, RenderContext(), chunk_size=128, overlap=16)
    m0 = img[:, :128].mean()
    m1 = img[:, 112:].mean()
    row = out[50, :, 0]
    np.testing.assert_allclose(row[:112], m0, atol=1e-6)
    np.testing.assert_allclose(row[128:], m1, atol=1e-6)
    ramp = runtime.smooth_step_ramp(16)
    np.testing.assert_allclose(row[112:128], m0 * (1 - ramp) + m1 * ramp, atol=1e-6)
    assert ramp[0] == 0 and ramp[-1] == 1


def test_strength_control_tensor(tmp_path):
    p = strength_model(tmp_path)
    img = np.random.default_rng(2).random((40, 40, 3), dtype=np.float32)
    out = runtime.process_image("fake_strength.onnx", p, img, RenderContext(), strength=40)
    np.testing.assert_allclose(out, img * 0.4, atol=1e-6)


def test_fp16_and_fixed_size_models(tmp_path):
    img = np.random.default_rng(3).random((100, 150, 3), dtype=np.float32)
    p16 = identity_model(tmp_path, "fake_fp16.onnx", FLOAT16)
    out = runtime.process_image("fake_fp16.onnx", p16, img, RenderContext(), chunk_size=128)
    np.testing.assert_allclose(out, img, atol=1e-3)
    pfix = identity_model(tmp_path, "fake_fixed.onnx", FLOAT, [1, 3, 64, 64])
    info = runtime.ModelInfo("fake_fixed.onnx", runtime.load_session(pfix))
    assert info.tile_limit == 64 and info.tensor_size(10, 20) == (64, 64)
    out = runtime.process_image("fake_fixed.onnx", pfix, img, RenderContext(), chunk_size=512, overlap=16)
    np.testing.assert_allclose(out, img, atol=1e-6)


def test_model_info_port():
    assert runtime.name_scale("RealESRGAN_x4plus.ort") == 4
    assert runtime.name_scale("2xVHS2HD-RealPLKSR.onnx") == 2
    assert runtime.name_scale("DAT_2_x3.onnx") == 3
    assert runtime.name_scale("1x-DeBink-v4.onnx") == 1
    assert runtime.name_scale("zero_dcepp.onnx") == 1


def test_cancellation(tmp_path):
    p = identity_model(tmp_path)
    tok = CancelToken()
    tok.cancel("test")
    with pytest.raises(LazyRoomError) as e:
        runtime.process_image("fake_identity.onnx", p, np.zeros((300, 300, 3), np.float32),
                              RenderContext(cancel=tok), chunk_size=128)
    assert e.value.code == ErrorCode.CANCELLED


def test_lama_processor_with_fake_model(tmp_path):
    """LaMa I/O contract: inputs 'image' (0..1) + binary 'mask'; normal model outputs 0..255."""
    p = make_model(tmp_path / "fake_lama.onnx",
                   [_node("Mul", ["image", "k"], ["output"])],
                   [("image", FLOAT, [1, 3, 512, 512]), ("mask", FLOAT, [1, 1, 512, 512])],
                   [("output", FLOAT, [1, 3, 512, 512])], [_init("k", np.array([255.0], np.float32))])
    img = np.full((60, 80, 3), 0.5, np.float32)
    out = processors.lama_inpaint("LaMa_512.onnx", p, img, np.ones((60, 80), np.float32))
    assert out.shape == img.shape
    assert abs(float(out.mean()) - 127 / 255) < 0.01  # trunc(0.5*255) = 127


def test_inpaint_op_with_fake_lama(empty_home):
    """ai.inpaint end-to-end: mask spec -> LaMa contract -> only the masked region changes."""
    mdir = Path(models.model_path("LaMa_512.onnx")).parent
    mdir.mkdir(parents=True)
    make_model(mdir / "LaMa_512.onnx", [_node("Mul", ["mask", "k"], ["output"])],  # output: 255 inside the hole
               [("image", FLOAT, [1, 3, 512, 512]), ("mask", FLOAT, [1, 1, 512, 512])],
               [("output", FLOAT, [1, 1, 512, 512])], [_init("k", np.full((1, 3, 1, 1), 255.0, np.float32))])
    img = np.random.default_rng(5).random((64, 96, 3), dtype=np.float32) * 0.5
    spec = get_op("ai.inpaint")
    mask = {"shapes": [{"type": "rect", "x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}]}
    out = spec.run(img, RenderContext(), spec.validate({"mask": json.dumps(mask)}))
    np.testing.assert_allclose(out[:8], img[:8], atol=1e-6)  # outside the mask untouched
    assert float(out[30:34, 45:50].mean()) > 0.95  # hole filled by the (fake) model
    unchanged = spec.run(img, RenderContext(), spec.validate({}))  # no mask -> input (SpotHealFilter)
    np.testing.assert_array_equal(unchanged, img)


def test_ai_detector_contract_with_fake_model(tmp_path):
    p = make_model(tmp_path / "fake_det.onnx", [_node("ReduceMean", ["input"], ["output"], axes=[2, 3], keepdims=0)],
                   [("input", FLOAT, [1, 3, 224, 224])], [("output", FLOAT, [1, 3])])
    img = np.zeros((50, 80, 3), np.float32)
    img[..., 0] = 1.0
    meta = {"ai_detection": {"input_size": 224, "resize_size": 256, "resize_mode": "ShortestSideCenterCrop",
                             "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
                             "output": {"kind": "softmax", "ai_class_indices": [0]}}}
    res = processors.detect_ai_generated(meta, p, img)
    logits = np.array([1.0, -1.0, -1.0])  # (x - 0.5) / 0.5 per channel
    expected = np.exp(logits[0]) / np.exp(logits).sum()
    assert res["ai_probability"] == pytest.approx(expected, abs=1e-3)
    assert res["verdict"] == "likely_ai"
    meta["ai_detection"]["output"] = {"kind": "sigmoid"}
    assert processors.detect_ai_generated(meta, p, img)["ai_probability"] == pytest.approx(1 / (1 + np.exp(-1)),
                                                                                           abs=1e-3)


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
def _fake_entry(name: str, src: Path, sha: str | None = None, **extra) -> None:
    models.register_model(name, title="fake", type="UPSCALE", task="upscale", url=src.as_uri(),
                          size=src.stat().st_size, sha256=sha or hashlib.sha256(src.read_bytes()).hexdigest(),
                          license="MIT", restricted=False, **extra)


def test_download_verifies_checksum_atomically(empty_home):
    src = identity_model(empty_home, "src_model.onnx")
    try:
        _fake_entry("fake_dl.onnx", src)
        res = models.download_model("fake_dl.onnx")
        assert res["status"] == "downloaded" and models.model_available("fake_dl.onnx")
        assert models.download_model("fake_dl.onnx")["status"] == "already present"
        _fake_entry("fake_bad.onnx", src, sha="0" * 64)
        with pytest.raises(LazyRoomError) as e:
            models.download_model("fake_bad.onnx")
        assert e.value.code == ErrorCode.DECODE_FAILED
        mdir = models.model_path("fake_bad.onnx").parent
        assert not models.model_path("fake_bad.onnx").exists()
        assert not [p for p in mdir.iterdir() if ".part" in p.name]
    finally:
        models.MODELS.pop("fake_dl.onnx", None)
        models.MODELS.pop("fake_bad.onnx", None)


def test_download_zip_pack(empty_home):
    payload = b"onnx-bytes"
    zp = empty_home / "pack.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("nested/dir/a.onnx", payload)
        z.writestr("LICENSE.txt", b"MIT")
    try:
        _fake_entry("fake_pack.zip", zp, pack={"files": [{"name": "a.onnx", "size": len(payload),
                                                          "sha256": hashlib.sha256(payload).hexdigest()}]})
        res = models.download_model("fake_pack.zip")
        d = Path(res["path"])
        assert d.is_dir() and (d / "a.onnx").read_bytes() == payload and (d / ".complete").exists()
        assert models.model_available("fake_pack.zip")
        _fake_entry("fake_pack2.zip", zp, pack={"files": [{"name": "a.onnx", "sha256": "1" * 64}]})
        with pytest.raises(LazyRoomError):
            models.download_model("fake_pack2.zip")
        assert not models.model_available("fake_pack2.zip")
    finally:
        models.MODELS.pop("fake_pack.zip", None)
        models.MODELS.pop("fake_pack2.zip", None)


# ---------------------------------------------------------------------------
# Processors without models
# ---------------------------------------------------------------------------
def test_depth_effects_all_render():
    rng = np.random.default_rng(4)
    rgb = rng.random((40, 60, 3), dtype=np.float32)
    depth = np.tile(np.linspace(0, 1, 30, dtype=np.float32), (20, 1))
    for eff in processors.DEPTH_EFFECTS:
        out, a = processors.render_depth_effect(rgb, None, depth, eff)
        assert out.shape == rgb.shape and np.isfinite(out).all(), eff
        if eff == "cutout":
            assert a is not None and a.min() < 0.5 < a.max()


def test_custom_depth_map_roundtrip():
    depth = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (16, 1))
    spectral, _ = processors.render_depth_effect(np.zeros((16, 64, 3), np.float32), None, depth, "map")
    np.testing.assert_allclose(processors.depth_from_image(spectral), depth, atol=0.02)
    gray = np.repeat(depth[..., None], 3, axis=2)
    np.testing.assert_allclose(processors.depth_from_image(gray), depth, atol=0.005)


def test_lbfgs_port_minimises_quadratic():
    a = np.linspace(1, 5, 50).astype(np.float32)
    target = np.linspace(-1, 1, 50).astype(np.float32)

    def f(x):
        r = x - target
        return float((a * r * r).sum()), (2 * a * r).astype(np.float32)

    x = processors.lbfgs(np.zeros(50, np.float32), f, steps=50)
    np.testing.assert_allclose(x, target, atol=1e-3)


def test_ocr_geometry_helpers():
    from lazyroom.ai import ocr

    pts = np.array([(x, y) for x in range(10, 31) for y in range(5, 11)], np.float32)
    hull = ocr.convex_hull(pts)
    rect = ocr.min_area_rect(hull, True)
    xs, ys = [p[0] for p in rect], [p[1] for p in rect]
    assert (min(xs), max(xs), min(ys), max(ys)) == pytest.approx((10, 30, 5, 10), abs=1e-4)
    big = ocr.unclip(rect, 1.5)
    assert abs(ocr._signed_area(big)) > abs(ocr._signed_area(rect))
    assert ocr.order_clockwise([(30, 10), (10, 5), (30, 5), (10, 10)])[0] == (10, 5)


# ---------------------------------------------------------------------------
# Real models (downloaded with: LAZYROOM_HOME=/tmp/pm_ai_home lazyroom models download NAME --yes)
# ---------------------------------------------------------------------------
@pytest.fixture
def ai_home(monkeypatch):
    monkeypatch.setenv("LAZYROOM_HOME", AI_HOME)
    runtime.clear_sessions()
    return AI_HOME


def _need(name: str) -> None:
    if not models.model_available(name):
        pytest.skip(f"{name} not downloaded into {AI_HOME} "
                    f"(LAZYROOM_HOME={AI_HOME} lazyroom models download {name} --yes)")


def _photo(name: str) -> np.ndarray:
    from lazyroom.core.io import load_image

    img, _ = load_image(Path(__file__).parent / "data" / "photos" / name)
    return img[..., :3]


@pytest.mark.slow
def test_real_remove_background_u2netp(ai_home):
    _need("u2netp.onnx")
    img = _photo("chelsea.png")
    spec = get_op("ai.remove_background")
    out = spec.run(img, RenderContext(), spec.validate({"model": "u2netp.onnx"}))
    assert out.shape == (*img.shape[:2], 4)
    a = out[..., 3]
    assert a[a.shape[0] // 3, a.shape[1] // 2] > 0.8  # the cat's face is foreground
    assert (a < 0.1).mean() > 0.05 and (a > 0.9).mean() > 0.2
    np.testing.assert_array_equal(out[..., :3], img)
    # same model drives the 'subject' mask type
    from lazyroom.core.masks import render_mask

    m = render_mask({"shapes": [{"type": "subject", "model": "u2netp"}]}, img, RenderContext())
    np.testing.assert_allclose(m, a, atol=1e-5)


@pytest.mark.slow
def test_real_upscale_realesrgan_x4v3(ai_home):
    _need("RealESRGAN-x4v3.ort")
    img = _photo("chelsea.png")[:120, :160]
    spec = get_op("ai.upscale")
    tiled = spec.run(img, RenderContext(), spec.validate({"model": "RealESRGAN-x4v3.ort", "chunk_size": 128,
                                                          "overlap": 16}))
    single = spec.run(img, RenderContext(), spec.validate({"model": "RealESRGAN-x4v3.ort",
                                                           "enable_chunking": False}))
    assert tiled.shape == single.shape == (480, 640, 3)
    assert float(np.abs(tiled - single).mean()) < 0.005  # overlap blending hides tile seams
    import cv2

    down = cv2.resize(single, (160, 120), interpolation=cv2.INTER_AREA)
    assert float(np.abs(down - img).mean()) < 0.05  # content preserved
