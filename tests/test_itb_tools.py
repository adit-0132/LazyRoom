"""Tests for the ImageToolbox utility ports: image cutting, photomosaic, PDF tools (tools/itb_tools.py)
and the draw-to-erase operation (ops/erase.py)."""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import lazyroom.ops.erase  # noqa: F401  (registers draw.erase_to_alpha)
import lazyroom.tools.itb_tools as it
from lazyroom.core.context import RenderContext
from lazyroom.core.errors import ErrorCode, LazyRoomError
from lazyroom.core.io import load_image, save_image
from lazyroom.core.params import validate_params
from lazyroom.core.registry import REGISTRY
from lazyroom.tools.registry import TOOLS
from tests.conftest import synthetic_image

pdfium = pytest.importorskip("pypdfium2")
from pypdf import PdfReader, PdfWriter  # noqa: E402
from pypdf.generic import (ArrayObject, DecodedStreamObject, DictionaryObject, FloatObject,  # noqa: E402
                           NameObject, NumberObject, TextStringObject)

ITB_TOOLS = sorted(n for n, t in TOOLS.items() if t.fn.__module__ == it.__name__)


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("LAZYROOM_HOME", str(tmp_path / "home"))
    from lazyroom.api import LazyRoom

    return LazyRoom()


def run(api, _tool, **params):
    t = TOOLS[_tool]
    return t.fn(api, **validate_params(t.params, params, op_id=_tool))


def err(api, code, _tool, **params) -> LazyRoomError:
    with pytest.raises(LazyRoomError) as e:
        run(api, _tool, **params)
    assert e.value.code == code, e.value
    return e.value


@pytest.fixture
def images(tmp_path):
    out = []
    for i in range(5):
        p = tmp_path / f"img{i}.png"
        save_image(synthetic_image(80 + 10 * i, 60, alpha=(i == 1), seed=i), p)
        out.append(str(p))
    return out


def make_pdf(path: Path, n: int = 3, *, extras: bool = True, sizes=None) -> str:
    """Synthetic PDF: Helvetica text 'Page k', a red/blue image, link + text annotations, and (extras)
    an AcroForm, an OpenAction script, embedded files, JavaScript names and document info."""
    w = PdfWriter()
    for k in range(n):
        pw, ph = (sizes[k] if sizes else (200, 300))
        page = w.add_blank_page(pw, ph)
        img = DecodedStreamObject()
        arr = np.zeros((8, 8, 3), np.uint8)
        arr[..., 0] = 255 if k % 2 == 0 else 0
        arr[..., 2] = 0 if k % 2 == 0 else 255
        img.set_data(arr.tobytes())
        img.update({NameObject("/Type"): NameObject("/XObject"), NameObject("/Subtype"): NameObject("/Image"),
                    NameObject("/Width"): NumberObject(8), NameObject("/Height"): NumberObject(8),
                    NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
                    NameObject("/BitsPerComponent"): NumberObject(8)})
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/XObject"): DictionaryObject({NameObject("/Im0"): w._add_object(img)}),
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        cs = DecodedStreamObject()
        cs.set_data(f"q 80 0 0 80 20 20 cm /Im0 Do Q BT /F1 24 Tf 20 {ph - 60} Td (Page {k + 1}) Tj ET".encode())
        page.replace_contents(cs)
        annots = ArrayObject()
        for st, rect in (("/Link", [10, 10, 60, 30]), ("/Text", [100, 100, 120, 120]),
                         ("/Highlight", [20, 200, 90, 220]), ("/Widget", [120, 10, 180, 30])):
            a = DictionaryObject({NameObject("/Type"): NameObject("/Annot"), NameObject("/Subtype"): NameObject(st),
                                  NameObject("/Rect"): ArrayObject([FloatObject(v) for v in rect])})
            annots.append(w._add_object(a))
        page[NameObject("/Annots")] = annots
    if extras:
        root = w.root_object
        js = DictionaryObject({NameObject("/S"): NameObject("/JavaScript"), NameObject("/JS"): TextStringObject("app.alert(1)")})
        root[NameObject("/OpenAction")] = w._add_object(js)
        root[NameObject("/AcroForm")] = DictionaryObject({NameObject("/Fields"): ArrayObject()})
        root[NameObject("/Names")] = DictionaryObject({
            NameObject("/EmbeddedFiles"): DictionaryObject({NameObject("/Names"): ArrayObject()}),
            NameObject("/JavaScript"): DictionaryObject({NameObject("/Names"): ArrayObject()})})
        w.add_metadata({"/Title": "Secret title", "/Author": "Someone"})
    with open(path, "wb") as f:
        w.write(f)
    return str(path)


@pytest.fixture
def pdf(tmp_path):
    return make_pdf(tmp_path / "doc.pdf")


def render_page(path, index=0, scale=1.0) -> np.ndarray:
    doc = pdfium.PdfDocument(str(path))
    return doc[index].render(scale=scale, fill_color=(255, 255, 255, 255), rev_byteorder=True).to_numpy()


# =============================================================================
# registry / schema
# =============================================================================

def test_all_tools_registered_and_describable():
    expected = {"image_cutting", "photomosaic", "pdf_info", "pdf_from_images", "pdf_to_images", "pdf_merge",
                "pdf_split", "pdf_remove_pages", "pdf_rotate", "pdf_rearrange", "pdf_crop", "pdf_crop_presets",
                "pdf_page_numbers", "pdf_watermark", "pdf_signature", "pdf_protect", "pdf_unlock", "pdf_compress",
                "pdf_grayscale", "pdf_dark_mode", "pdf_repair", "pdf_metadata", "pdf_extract_text", "pdf_flatten",
                "pdf_extract_images", "pdf_zip", "pdf_print", "pdf_remove_annotations", "pdf_compare",
                "pdf_resize_pages", "pdf_sanitize", "pdf_contact_sheet", "pdf_searchable", "pdf_from_comic",
                "limits_resize", "screenshot_framing"}
    assert expected == set(ITB_TOOLS)
    for name in ITB_TOOLS:
        d = TOOLS[name].describe()
        json.dumps(d)
        assert d["source"]["project"] == "ImageToolbox"
        assert all(n.startswith("pdf_") for n in ITB_TOOLS if TOOLS[n].category == "pdf")


# =============================================================================
# Image cutting
# =============================================================================

def _grid(h=10, w=10):
    y, x = np.mgrid[0:h, 0:w]
    return np.dstack([x / 10, y / 10, np.zeros((h, w))]).astype(np.float32)


def test_cut_and_merge_semantics():
    img = _grid()
    out = it.cut_and_merge(img, [0.2, 0.5], None, False, False)  # remove columns 2..4
    assert out.shape == (10, 7, 3)
    np.testing.assert_array_equal(out[:, :2], img[:, :2])
    np.testing.assert_array_equal(out[:, 2:], img[:, 5:])
    out = it.cut_and_merge(img, None, [0.3, 0.6], False, True)  # keep rows 3..5
    np.testing.assert_array_equal(out, img[3:6])
    out = it.cut_and_merge(img, [0.2, 0.5], [0.3, 0.6], True, True)  # both inverse: crop
    np.testing.assert_array_equal(out, img[3:6, 2:5])
    out = it.cut_and_merge(img, [0.2, 0.5], [0.3, 0.6], True, False)  # keep band, then drop rows
    np.testing.assert_array_equal(out, np.concatenate([img[:3, 2:5], img[6:, 2:5]], 0))
    np.testing.assert_array_equal(it.cut_and_merge(img, [0.0, 1.0], [0.0, 1.0], False, False), img)
    # removing everything throws upstream -> original image returned untouched
    np.testing.assert_array_equal(it.cut_and_merge(img, [0.0001, 0.9999], None, False, False), img)
    # start == end after rounding is not a valid cut
    np.testing.assert_array_equal(it.cut_and_merge(img, [0.21, 0.24], None, False, False), img)


def test_image_cutting_single_and_batch(api, images, tmp_path):
    r = run(api, "image_cutting", inputs=[images[0]], vertical=[0.25, 0.5], output=str(tmp_path / "cut.png"))
    assert (r["outputs"][0]["width"], r["outputs"][0]["height"]) == (60, 60)
    src, _ = load_image(images[0])
    out, _ = load_image(tmp_path / "cut.png")
    assert np.abs(out[:, :20] - src[:, :20]).max() < 1 / 255 + 1e-6
    r = run(api, "image_cutting", inputs=images[:3], horizontal=[0.5, 1.0], inverse_horizontal=True,
            output_dir=str(tmp_path / "batch"), name_template="{index}_{stem}.png")
    assert r["count"] == 3
    assert [Path(o["path"]).name for o in r["outputs"]] == ["1_img0.png", "2_img1.png", "3_img2.png"]
    assert all(o["height"] == 30 for o in r["outputs"])
    assert load_image(tmp_path / "batch" / "2_img1.png")[0].shape[2] == 4  # alpha kept


def test_image_cutting_errors(api, images, tmp_path):
    err(api, ErrorCode.INVALID_PARAMS, "image_cutting", inputs=images[:2], output=str(tmp_path / "x.png"))
    err(api, ErrorCode.INVALID_PARAMS, "image_cutting", inputs=[images[0]], vertical=[0.6, 0.2],
        output=str(tmp_path / "x.png"))
    err(api, ErrorCode.INVALID_PARAMS, "image_cutting", inputs=[images[0]], vertical=[0.2], output="x.png")
    err(api, ErrorCode.INVALID_PARAMS, "image_cutting", inputs=images, output_dir=str(tmp_path / "o"),
        name_template="{nope}.png")
    run(api, "image_cutting", inputs=[images[0]], output=str(tmp_path / "same.png"))
    err(api, ErrorCode.OUTPUT_EXISTS, "image_cutting", inputs=[images[0]], output=str(tmp_path / "same.png"))


# =============================================================================
# Photomosaic: ports of the upstream unit tests + end-to-end
# =============================================================================

def _labs(vals):
    return np.array(vals, np.float64)


def test_tile_matcher_upstream_cases():
    dark, light = [10.0, 0, 0], [90.0, 0, 0]
    assert list(it._tile_match(_labs([light, dark]), _labs([dark, light]), 2, 0)) == [1, 0]
    t, close = [50.0, 0, 0], [51.0, 0, 0]
    assert list(it._tile_match(_labs([t] * 4), _labs([t, close]), 2, 1)) == [0, 1, 1, 0]
    assert list(it._tile_match(_labs([t] * 6), _labs([t, close]), 6, 5)) == [0, 1, 0, 1, 0, 1]
    assert set(it._tile_match(_labs([t] * 12), _labs([t, [80.0, 0, 0]]), 4, 0)) == {0}
    res = it._tile_match(_labs([[0.0, 0, 0]] * 7), _labs([[i * 30.0, 0, 0] for i in range(3)]), 5, 50)
    assert len(res) == 7 and set(res) <= {0, 1, 2}
    assert len(it._tile_match(np.zeros((0, 3)), _labs([[0.0, 0, 0]]), 1, 5)) == 0
    with pytest.raises(ValueError):
        it._tile_match(_labs([[0.0, 0, 0]]), np.zeros((0, 3)), 1, 0)


def test_tile_matcher_repeat_avoidance_across_rows():
    cols, d = 8, 3
    res = it._tile_match(_labs([[50.0, 0, 0]] * (cols * 8)), _labs([[50.0, 0, 0]] * 32), cols, d)
    for i, ti in enumerate(res):
        for j in range(i):
            if abs(i // cols - j // cols) + abs(i % cols - j % cols) <= d:
                assert res[j] != ti


def _layout(width=1000, height=700, aspects=None, repeat=0, detail=0.5):
    aspects = aspects or [1.0] * 8
    labs = _labs([[float(i), 0, 0] for i in range(len(aspects))])
    return it._adaptive_layout(width, height, 10, labs, np.array(aspects, np.float32), repeat,
                               lambda x, y: np.zeros(3), lambda y: detail)


def test_adaptive_layout_upstream_cases():
    pl = _layout(1200, 800, [0.75, 1, 1.5, 1.8])
    rows: dict = {}
    for p in pl:
        rows.setdefault(float(p[2]), []).append(p)
    keys = sorted(rows)
    assert keys[0] == 0
    for k in keys:
        row = sorted(rows[k], key=lambda p: p[1])
        assert row[0][1] == pytest.approx(0, abs=1e-3) and row[-1][3] == pytest.approx(1200, abs=1e-3)
        for a, b in zip(row, row[1:]):
            assert a[3] == pytest.approx(b[1], abs=1e-3)
    for a, b in zip(keys, keys[1:]):
        assert rows[a][0][4] == pytest.approx(b, abs=1e-3)
    assert rows[keys[-1]][0][4] >= 800
    aspects = [0.6, 1.0, 1.4, 2.1]
    for idx, l, t, r, b in _layout(aspects=aspects, repeat=3):
        assert (r - l) / (b - t) == pytest.approx(aspects[idx], abs=1e-3)
    low, high = _layout(detail=0.0), _layout(detail=1.0)
    assert high[0][4] - high[0][2] < low[0][4] - low[0][2] and len(high) > len(low)
    pl = _layout(aspects=[1.0] * 4, repeat=3)
    for i, p in enumerate(pl):
        assert p[0] not in [q[0] for q in pl[max(0, i - 3):i]]


def test_rgb_to_lab_reference():
    lab = it._rgb8_to_lab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]]))
    assert lab[0] == pytest.approx([100, 0, 0], abs=0.01)
    assert lab[1] == pytest.approx([0, 0, 0], abs=1e-6)
    assert lab[2] == pytest.approx([53.24, 80.09, 67.20], abs=0.05)


@pytest.fixture
def tiles(tmp_path):
    rng = np.random.default_rng(1)
    d = tmp_path / "tiles"
    d.mkdir()
    for i in range(24):
        w, h = int(rng.integers(20, 60)), int(rng.integers(20, 60))
        save_image(np.ones((h, w, 3), np.float32) * rng.random(3).astype(np.float32), d / f"t{i:02d}.png")
    return d


@pytest.mark.parametrize("layout", ["adaptive", "grid"])
def test_photomosaic_end_to_end_deterministic(api, tmp_path, tiles, layout):
    target = tmp_path / "target.png"
    save_image(synthetic_image(240, 180, seed=5), target)
    outs = []
    for k in range(2):
        r = run(api, "photomosaic", input=str(target), tiles=[str(tiles)], columns=12, layout=layout,
                color_blend=0.0, output=str(tmp_path / f"m_{layout}{k}.png"))
        assert r["output"]["width"] == 240 and r["output"]["height"] == 180 and r["tiles_used"] == 24
        outs.append(load_image(r["output"]["path"])[0])
    np.testing.assert_array_equal(outs[0], outs[1])
    src = load_image(target)[0]
    # every pixel comes from a (solid) tile, so it differs from the gradient target; colours still follow it
    assert np.abs(outs[0] - src).mean() > 0.01
    assert np.abs(outs[0].mean((0, 1)) - src.mean((0, 1))).max() < 0.2
    blended = run(api, "photomosaic", input=str(target), tiles=[str(tiles / "*.png")], columns=12, layout=layout,
                  color_blend=0.6, output=str(tmp_path / f"b_{layout}.png"))
    b = load_image(blended["output"]["path"])[0]
    assert np.abs(b - src).mean() < np.abs(outs[0] - src).mean()  # target colour blend pulls towards target


def test_photomosaic_errors_and_preview(api, tmp_path, tiles, images):
    err(api, ErrorCode.FILE_NOT_FOUND, "photomosaic", input=images[0], tiles=[images[0]],
        output=str(tmp_path / "x.png"))
    err(api, ErrorCode.INVALID_PARAMS, "photomosaic", input=images[0], tiles=[str(tiles)], columns=5,
        output=str(tmp_path / "x.png"))
    r = run(api, "photomosaic", input=images[0], tiles=[str(tiles)], preview=True, max_tiles=10,
            output=str(tmp_path / "p.png"))
    assert r["tile_size"] == 48 and r["tiles_used"] == 10


# =============================================================================
# Erase / restore by drawing (draw.erase_to_alpha)
# =============================================================================

def _erase(img, **values):
    spec = REGISTRY["draw.erase_to_alpha"]
    return spec.run(img, RenderContext(), spec.validate(values))


def test_erase_and_restore_rect():
    spec = REGISTRY["draw.erase_to_alpha"]
    assert spec.alpha == "full" and spec.category == "draw"
    img = synthetic_image(80, 60)
    rect = {"shapes": [{"type": "rect", "x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}]}
    out = _erase(img, mask=rect)
    assert out.shape == (60, 80, 4)
    assert out[30, 40, 3] == 0 and out[2, 2, 3] == 1
    np.testing.assert_array_equal(out[..., :3], img)  # colours untouched
    back = _erase(out, mask={"shapes": [{"type": "rect", "x": 0.25, "y": 0.25, "w": 0.25, "h": 0.5}]},
                  mode="restore")
    assert back[30, 30, 3] == pytest.approx(1) and back[30, 50, 3] == 0


def test_erase_soft_brush_and_json_mask():
    img = synthetic_image(100, 100, alpha=True)
    stroke = {"shapes": [{"type": "brush", "strokes": [{"points": [[0.2, 0.5], [0.8, 0.5]], "radius": 0.08,
                                                         "hardness": 0.2}]}]}
    out = _erase(img, mask=json.dumps(stroke))
    a = out[..., 3]
    assert a[50, 50] < 0.3 * img[50, 50, 3]  # soft brush: ~86 % coverage at the stroke centre
    assert np.any((a > 0.05) & (a < img[..., 3] - 0.05))  # soft edge = partial transparency
    assert np.all(a <= img[..., 3] + 1e-6)
    restored = _erase(out, mask=stroke, mode="restore")
    assert np.all(restored[..., 3] >= a - 1e-6) and restored[50, 50, 3] > 0.85  # SrcOver: a + c (1 - a)
    hard = {"shapes": [{"type": "brush", "strokes": [{"points": [[0.2, 0.5], [0.8, 0.5]], "radius": 0.08,
                                                       "hardness": 1.0}]}]}
    assert _erase(_erase(img, mask=hard), mask=hard, mode="restore")[50, 50, 3] == pytest.approx(1.0)


def test_erase_noop_and_errors():
    img = synthetic_image(30, 20)
    np.testing.assert_array_equal(_erase(img), img)
    spec = REGISTRY["draw.erase_to_alpha"]
    assert spec.defaults() == {"mask": None, "mode": "erase"}
    json.dumps(spec.describe())
    for bad in ({"shapes": []}, {"shapes": [{"type": "nope"}]}, "{not json", 5):
        with pytest.raises(LazyRoomError):
            spec.validate({"mask": bad})


# =============================================================================
# PDF tools
# =============================================================================

def test_pdf_info_and_metadata(api, pdf, tmp_path):
    info = run(api, "pdf_info", input=pdf)
    assert info["status"] == "open" and info["pages"] == 3
    assert info["page_info"][0] == {"width": 200.0, "height": 300.0, "rotation": 0}
    assert info["metadata"]["title"] == "Secret title"
    assert run(api, "pdf_metadata", input=pdf)["metadata"]["author"] == "Someone"
    r = run(api, "pdf_metadata", input=pdf, title="New", keywords="a, b", output=str(tmp_path / "m.pdf"))
    assert r["metadata"]["title"] == "New" and r["metadata"]["keywords"] == "a, b"
    assert r["metadata"]["author"] == "Someone"  # empty fields are kept
    r = run(api, "pdf_metadata", input=pdf, reset=True, output=str(tmp_path / "m2.pdf"))
    assert r["metadata"]["title"] is None and r["metadata"]["author"] is None


def test_pdf_info_missing_and_broken(api, tmp_path):
    err(api, ErrorCode.FILE_NOT_FOUND, "pdf_info", input=str(tmp_path / "missing.pdf"))
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf at all")
    assert run(api, "pdf_info", input=str(tmp_path / "broken.pdf"))["status"] == "failure"
    err(api, ErrorCode.DECODE_FAILED, "pdf_split", input=str(tmp_path / "broken.pdf"),
        output=str(tmp_path / "o.pdf"))


def test_pdf_from_images_and_to_images(api, images, tmp_path):
    r = run(api, "pdf_from_images", inputs=images, quality=100, output=str(tmp_path / "imgs.pdf"))
    assert r["pages"] == 5
    rd = PdfReader(tmp_path / "imgs.pdf")
    assert [float(p.mediabox.width) for p in rd.pages] == [80, 90, 100, 110, 120]
    r = run(api, "pdf_from_images", inputs=images, scale_percent=50, scale_direction="up",
            output=str(tmp_path / "up.pdf"))
    assert {float(p.mediabox.width) for p in PdfReader(tmp_path / "up.pdf").pages} == {60.0}
    r = run(api, "pdf_to_images", input=str(tmp_path / "imgs.pdf"), output_dir=str(tmp_path / "pages"),
            pages="1,3", scale_percent=100)
    assert r["count"] == 2 and r["dpi"] == 72
    page1 = load_image(r["outputs"][0]["path"])[0]
    src = load_image(images[0])[0]
    assert page1.shape[:2] == (60, 80)
    assert np.abs(page1 - src[..., :3]).mean() < 0.02  # lossless page renders back to the image
    r = run(api, "pdf_to_images", input=str(tmp_path / "imgs.pdf"), output_dir=str(tmp_path / "p2"), pages="2",
            scale_percent=200, format="jpg")
    assert r["outputs"][0]["width"] == 180 and r["outputs"][0]["path"].endswith("imgs_2.jpg")


def test_pdf_merge_split_remove_rearrange(api, pdf, tmp_path):
    other = make_pdf(tmp_path / "other.pdf", 2, extras=False)
    r = run(api, "pdf_merge", inputs=[pdf, other], output=str(tmp_path / "merged.pdf"))
    assert r["pages"] == 5
    run(api, "pdf_merge", inputs=[str(tmp_path / "doc.pdf"), str(tmp_path / "o*.pdf")],
        output=str(tmp_path / "merged2.pdf"))
    r = run(api, "pdf_split", input=str(tmp_path / "merged.pdf"), pages="2-3,5", output=str(tmp_path / "s.pdf"))
    texts = [p.extract_text() for p in PdfReader(tmp_path / "s.pdf").pages]
    assert texts == ["Page 2", "Page 3", "Page 2"]
    r = run(api, "pdf_remove_pages", input=pdf, pages="1,3", output=str(tmp_path / "rm.pdf"))
    assert r["pages"] == 1 and PdfReader(tmp_path / "rm.pdf").pages[0].extract_text() == "Page 2"
    r = run(api, "pdf_rearrange", input=pdf, order="3-1,2", output=str(tmp_path / "re.pdf"))
    assert [p.extract_text() for p in PdfReader(tmp_path / "re.pdf").pages] == ["Page 3", "Page 2", "Page 1",
                                                                                 "Page 2"]
    err(api, ErrorCode.INVALID_PARAMS, "pdf_remove_pages", input=pdf, pages="1-3", output=str(tmp_path / "x.pdf"))
    err(api, ErrorCode.INVALID_PARAMS, "pdf_split", input=pdf, pages="0-2", output=str(tmp_path / "x.pdf"))
    err(api, ErrorCode.INVALID_PARAMS, "pdf_split", input=pdf, pages="2-9", output=str(tmp_path / "x.pdf"))
    err(api, ErrorCode.INVALID_PARAMS, "pdf_split", input=pdf, pages="a", output=str(tmp_path / "x.pdf"))
    err(api, ErrorCode.INVALID_PARAMS, "pdf_split", input=pdf, pages="3-1", output=str(tmp_path / "x.pdf"))
    err(api, ErrorCode.INVALID_PARAMS, "pdf_split", input=pdf, output=str(tmp_path / "x.png"))
    err(api, ErrorCode.FILE_NOT_FOUND, "pdf_merge", inputs=[str(tmp_path / "none*.pdf")],
        output=str(tmp_path / "x.pdf"))


def test_pdf_rotate(api, pdf, tmp_path):
    r = run(api, "pdf_rotate", input=pdf, angle="90", pages="2", output=str(tmp_path / "r.pdf"))
    assert [p.rotation for p in PdfReader(tmp_path / "r.pdf").pages] == [0, 90, 0]
    r = run(api, "pdf_rotate", input=str(tmp_path / "r.pdf"), rotations=[270, 270], output=str(tmp_path / "r2.pdf"))
    assert [p.rotation for p in PdfReader(tmp_path / "r2.pdf").pages] == [270, 0, 0]
    r = run(api, "pdf_rotate", input=str(tmp_path / "r.pdf"), auto=True, output=str(tmp_path / "r3.pdf"))
    assert r["rotations"] == [0, 270, 0]
    assert {p.rotation for p in PdfReader(tmp_path / "r3.pdf").pages} == {0}
    err(api, ErrorCode.INVALID_PARAMS, "pdf_rotate", input=pdf, rotations=[45], output=str(tmp_path / "x.pdf"))


def test_pdf_crop_and_presets(api, pdf, tmp_path):
    run(api, "pdf_crop", input=pdf, rect=[0.1, 0.2, 0.6, 0.7], pages="1", output=str(tmp_path / "c.pdf"))
    pages = PdfReader(tmp_path / "c.pdf").pages
    assert [round(float(v), 3) for v in pages[0].cropbox] == [20, 90, 120, 240]
    assert [float(v) for v in pages[1].cropbox] == [0, 0, 200, 300]
    # rotation-aware crop (90 degrees)
    rot = run(api, "pdf_rotate", input=pdf, angle="90", output=str(tmp_path / "rot.pdf"))["output"]
    run(api, "pdf_crop", input=rot, rect=[0.0, 0.0, 0.5, 1.0], output=str(tmp_path / "c90.pdf"))
    cb = PdfReader(tmp_path / "c90.pdf").pages[0].cropbox
    assert [round(float(v), 3) for v in cb] == [0, 0, 200, 150]
    assert run(api, "pdf_crop_presets")["presets"] == []
    run(api, "pdf_crop_presets", action="save", name="Left half", rect=[0, 0, 0.5, 1])
    run(api, "pdf_crop_presets", action="save", name="left HALF", rect=[0, 0, 0.4, 1])  # case-insensitive upsert
    ps = run(api, "pdf_crop_presets")["presets"]
    assert len(ps) == 1 and ps[0]["rect"] == [0, 0, 0.4, 1]
    run(api, "pdf_crop", input=pdf, preset="Left Half", output=str(tmp_path / "cp.pdf"))
    assert round(float(PdfReader(tmp_path / "cp.pdf").pages[0].cropbox.width), 3) == 80
    run(api, "pdf_crop_presets", action="delete", name="left half")
    assert run(api, "pdf_crop_presets")["presets"] == []
    err(api, ErrorCode.PRESET_NOT_FOUND, "pdf_crop", input=pdf, preset="nope", output=str(tmp_path / "x.pdf"))
    err(api, ErrorCode.INVALID_PARAMS, "pdf_crop", input=pdf, rect=[0.5, 0, 0.4, 1], output=str(tmp_path / "x.pdf"))
    err(api, ErrorCode.INVALID_PARAMS, "pdf_crop_presets", action="save", name=" ")


def test_pdf_page_numbers_watermark_signature(api, pdf, images, tmp_path):
    run(api, "pdf_page_numbers", input=pdf, label_format="{n}/{total} ✓", position="top_right",
        color="#00FF00", output=str(tmp_path / "n.pdf"))
    texts = [p.extract_text() for p in PdfReader(tmp_path / "n.pdf").pages]
    assert "1/3" in texts[0] and "3/3" in texts[2] and "✓" not in texts[0]  # unencodable glyph dropped
    img = render_page(tmp_path / "n.pdf")
    green = (img[..., 1] > 200) & (img[..., 0] < 80) & (img[..., 2] < 80)
    ys, xs = np.nonzero(green)
    assert ys.max() < 60 and xs.min() > 100  # top right corner
    run(api, "pdf_watermark", input=pdf, text="DRAFT", rotation=0, opacity=1.0, color="#00FF00", pages="2",
        output=str(tmp_path / "w.pdf"))
    pages = PdfReader(tmp_path / "w.pdf").pages
    assert "DRAFT" in pages[1].extract_text() and "DRAFT" not in pages[0].extract_text()
    page2 = render_page(tmp_path / "w.pdf", 1)
    m = (page2[..., 1] > 200) & (page2[..., 0] < 80) & (page2[..., 2] < 80)
    ys, xs = np.nonzero(m)
    assert abs(xs.mean() - 100) < 10 and abs(ys.mean() - 150) < 15  # centred
    assert xs.max() - xs.min() == pytest.approx(100, abs=8)  # 50 % of the page width
    err(api, ErrorCode.INVALID_PARAMS, "pdf_watermark", input=pdf, text="✓✓", output=str(tmp_path / "x.pdf"))
    run(api, "pdf_signature", input=pdf, image=images[1], x=0.9, y=0.9, size=0.3, opacity=1.0,
        output=str(tmp_path / "sig.pdf"))
    a = render_page(tmp_path / "sig.pdf").astype(int)
    b = render_page(pdf).astype(int)
    diff = np.abs(a - b).sum(-1) > 30
    ys, xs = np.nonzero(diff)
    assert xs.min() >= 135 and ys.max() <= 50  # clamped into the top-right corner (y from the bottom)


def test_pdf_protect_unlock(api, pdf, tmp_path):
    run(api, "pdf_protect", input=pdf, new_password="pw", output=str(tmp_path / "p.pdf"))
    info = run(api, "pdf_info", input=str(tmp_path / "p.pdf"))
    assert info["status"] == "needs_password"
    assert run(api, "pdf_info", input=str(tmp_path / "p.pdf"), password="pw")["status"] == "unlocked"
    e = err(api, ErrorCode.INVALID_PARAMS, "pdf_split", input=str(tmp_path / "p.pdf"), output=str(tmp_path / "x.pdf"))
    assert e.details.get("needs_password")
    err(api, ErrorCode.INVALID_PARAMS, "pdf_unlock", input=str(tmp_path / "p.pdf"), password="bad",
        output=str(tmp_path / "x.pdf"))
    r = run(api, "pdf_unlock", input=str(tmp_path / "p.pdf"), password="pw", output=str(tmp_path / "u.pdf"))
    assert r["was_encrypted"] and not PdfReader(tmp_path / "u.pdf").is_encrypted
    # tools that rasterise accept the password too
    run(api, "pdf_to_images", input=str(tmp_path / "p.pdf"), password="pw", output_dir=str(tmp_path / "pi"))
    err(api, ErrorCode.INVALID_PARAMS, "pdf_to_images", input=str(tmp_path / "p.pdf"), output_dir=str(tmp_path / "pj"))
    err(api, ErrorCode.INVALID_PARAMS, "pdf_protect", input=pdf, output=str(tmp_path / "x.pdf"))


def test_pdf_compress_and_grayscale(api, images, tmp_path):
    src = run(api, "pdf_from_images", inputs=images, quality=100, output=str(tmp_path / "s.pdf"))
    c = run(api, "pdf_compress", input=src["output"], quality=0.3, output=str(tmp_path / "c.pdf"))
    assert c["images"] == 5 and c["bytes"] < src["bytes"]
    g = run(api, "pdf_grayscale", input=src["output"], output=str(tmp_path / "g.pdf"))
    assert g["images"] == 5
    img = render_page(tmp_path / "g.pdf").astype(int)
    assert np.abs(img[..., 0] - img[..., 1]).max() <= 12 and np.abs(img[..., 1] - img[..., 2]).max() <= 12
    page = PdfReader(tmp_path / "g.pdf").pages[1]
    assert "/SMask" in page["/Resources"]["/XObject"]["/Im0"].get_object()  # alpha of img1 kept


def test_pdf_dark_mode(api, pdf, tmp_path):
    for theme in ("warm", "negative", "custom"):
        run(api, "pdf_dark_mode", input=pdf, theme=theme, custom_color="#102030", custom_blend_mode="multiply",
            output=str(tmp_path / f"d_{theme}.pdf"))
    light = render_page(pdf)
    dark = render_page(tmp_path / "d_warm.pdf")
    assert dark[5, 190].mean() < 80 and light[5, 190].mean() > 250  # white paper became dark
    neg = render_page(tmp_path / "d_negative.pdf")
    assert np.abs(neg[5, 190].astype(int) - 0).max() < 5  # pure inversion
    red_pixel = neg[260, 60]
    assert red_pixel[0] < 50 and red_pixel[1] > 200  # red image inverted to cyan (no desaturation)
    assert PdfReader(tmp_path / "d_warm.pdf").pdf_header >= "%PDF-1.4"
    assert "Page 1" in PdfReader(tmp_path / "d_warm.pdf").pages[0].extract_text()  # text stays vector


def test_pdf_repair_and_extract_text(api, pdf, tmp_path):
    r = run(api, "pdf_repair", input=pdf, output=str(tmp_path / "rep.pdf"))
    assert r["pages"] == 3
    r = run(api, "pdf_extract_text", input=pdf, output=str(tmp_path / "t.txt"))
    assert r["text"].startswith("--- Page 1 ---\nPage 1\n\n--- Page 2 ---")
    assert (tmp_path / "t.txt").read_text() == r["text"]


def test_pdf_sanitize(api, pdf, tmp_path):
    r = run(api, "pdf_sanitize", input=pdf, output=str(tmp_path / "s.pdf"))
    assert {"/OpenAction", "/AcroForm", "/EmbeddedFiles", "/JavaScript"} <= set(r["removed"])
    rd = PdfReader(tmp_path / "s.pdf")
    assert rd.metadata is None or not rd.metadata.get("/Title")
    assert "/OpenAction" not in rd.trailer["/Root"] and "/AcroForm" not in rd.trailer["/Root"]
    assert all(len(p.get("/Annots", [])) == 0 for p in rd.pages)
    run(api, "pdf_sanitize", input=pdf, metadata=False, annotations=False, scripts=False,
        output=str(tmp_path / "s2.pdf"))
    rd = PdfReader(tmp_path / "s2.pdf")
    assert rd.metadata["/Title"] == "Secret title" and "/OpenAction" in rd.trailer["/Root"]
    kinds = [a.get_object()["/Subtype"] for a in rd.pages[0]["/Annots"]]
    assert kinds == ["/Link", "/Text", "/Highlight"]  # widgets removed with forms


def test_pdf_remove_annotations(api, pdf, tmp_path):
    def kinds(path, i=0):
        return [a.get_object()["/Subtype"] for a in PdfReader(path).pages[i].get("/Annots", [])]

    r = run(api, "pdf_remove_annotations", input=pdf, output=str(tmp_path / "a.pdf"))
    assert r["removed"] == 3 and kinds(tmp_path / "a.pdf") == ["/Text", "/Highlight", "/Widget"]
    run(api, "pdf_remove_annotations", input=pdf, types=["markup"], pages="2", output=str(tmp_path / "b.pdf"))
    assert kinds(tmp_path / "b.pdf", 1) == ["/Link", "/Widget"] and len(kinds(tmp_path / "b.pdf", 0)) == 4
    run(api, "pdf_remove_annotations", input=pdf, types=it.ANNOTATION_TYPES, output=str(tmp_path / "c.pdf"))
    assert kinds(tmp_path / "c.pdf") == []
    err(api, ErrorCode.INVALID_PARAMS, "pdf_remove_annotations", input=pdf, types=["bogus"],
        output=str(tmp_path / "x.pdf"))


def test_pdf_flatten(api, pdf, tmp_path):
    r = run(api, "pdf_flatten", input=pdf, quality=0.5, output=str(tmp_path / "f.pdf"))
    assert r["dpi"] == 186
    rd = PdfReader(tmp_path / "f.pdf")
    assert all(p.extract_text().strip() == "" for p in rd.pages)  # text is now pixels
    a, b = render_page(pdf).astype(int), render_page(tmp_path / "f.pdf").astype(int)
    assert a.shape == b.shape and np.abs(a - b).mean() < 6
    rot = run(api, "pdf_rotate", input=pdf, angle="90", output=str(tmp_path / "rot.pdf"))["output"]
    run(api, "pdf_flatten", input=rot, output=str(tmp_path / "frot.pdf"))
    assert render_page(tmp_path / "frot.pdf").shape == render_page(rot).shape  # displayed orientation kept


def test_pdf_print(api, pdf, tmp_path):
    r = run(api, "pdf_print", input=pdf, pages_per_sheet="2", output=str(tmp_path / "p.pdf"))
    assert r["sheets"] == 2
    page = PdfReader(tmp_path / "p.pdf").pages[0]
    assert (float(page.mediabox.width), float(page.mediabox.height)) == (200, 300)  # original: first page box
    r = run(api, "pdf_print", input=pdf, pages_per_sheet="4", orientation="horizontal", page_size="a4",
            margin_percent=5, output=str(tmp_path / "p4.pdf"))
    page = PdfReader(tmp_path / "p4.pdf").pages[0]
    assert r["sheets"] == 1 and (float(page.mediabox.width), float(page.mediabox.height)) == (842, 595)
    img = render_page(tmp_path / "p4.pdf")
    reds = (img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] < 60)
    assert reds[: img.shape[0] // 2, : img.shape[1] // 2].any() and reds[img.shape[0] // 2:, : img.shape[1] // 2].any()
    rot = run(api, "pdf_rotate", input=pdf, angle="90", output=str(tmp_path / "rot.pdf"))["output"]
    run(api, "pdf_print", input=rot, output=str(tmp_path / "prot.pdf"))
    page = PdfReader(tmp_path / "prot.pdf").pages[0]
    assert (float(page.mediabox.width), float(page.mediabox.height)) == (300, 200)  # displayed orientation


def test_pdf_compare(api, pdf, tmp_path):
    other = make_pdf(tmp_path / "other.pdf", 2, extras=False)
    run(api, "pdf_watermark", input=other, text="X", opacity=1.0, color="#000000", output=str(tmp_path / "o2.pdf"))
    r = run(api, "pdf_compare", first=pdf, second=str(tmp_path / "o2.pdf"), highlight_color="#00FF00",
            output=str(tmp_path / "cmp.pdf"))
    assert r["pages"] == 3 and r["scores"][0] > 0 and r["scores"][2] > 0
    img = render_page(tmp_path / "cmp.pdf")
    assert ((img[..., 1] > 200) & (img[..., 0] < 60)).any()
    same = run(api, "pdf_compare", first=pdf, second=pdf, output=str(tmp_path / "same.pdf"))
    assert same["scores"] == [0.0, 0.0, 0.0]
    for ct in it.COMPARISON_TYPES:
        run(api, "pdf_compare", first=pdf, second=str(tmp_path / "o2.pdf"), comparison_type=ct,
            output=str(tmp_path / f"cmp_{ct}.pdf"))


def test_highlight_differences_metrics():
    a = np.full((40, 50, 4), 255, np.uint8)
    b = a.copy()
    b[10:20, 10:20, :3] = 0
    for ct, expect in (("ae", 100.0), ("ssim", None), ("mae", None), ("rmse", None), ("psnr", None), ("ncc", None)):
        out, score = it.highlight_differences(a, b, ct, (255, 0, 0, 255))
        hl = (out[..., 0] == 255) & (out[..., 1] == 0)
        assert hl[15, 15] and not hl[35, 45], ct
        if expect is not None:
            assert score == expect
    out, score = it.highlight_differences(a, a, "psnr", (255, 0, 0, 255))
    assert score == float("inf") and not (out[..., 1] == 0).any()


def test_pdf_resize_pages(api, pdf, tmp_path):
    run(api, "pdf_resize_pages", input=pdf, page_size="a4", mode="fit", output=str(tmp_path / "r.pdf"))
    rd = PdfReader(tmp_path / "r.pdf")
    p = rd.pages[0]
    assert [float(v) for v in p.mediabox] == [0, 0, 595, 842] and [float(v) for v in p.cropbox] == [0, 0, 595, 842]
    s = min(595 / 200, 842 / 300)  # fit
    off_x, off_y = (595 - 200 * s) / 2, (842 - 300 * s) / 2
    link = [float(v) for v in p["/Annots"][0].get_object()["/Rect"]]
    assert link == pytest.approx([10 * s + off_x, 10 * s + off_y, 60 * s + off_x, 30 * s + off_y], abs=1e-3)
    img = render_page(tmp_path / "r.pdf")
    reds = (img[..., 0] > 200) & (img[..., 1] < 60)
    ys, xs = np.nonzero(reds)
    assert xs.min() == pytest.approx(20 * s + off_x, abs=2) and xs.max() == pytest.approx(100 * s + off_x, abs=2)
    run(api, "pdf_resize_pages", input=pdf, page_size="letter", mode="stretch", output=str(tmp_path / "st.pdf"))
    assert float(PdfReader(tmp_path / "st.pdf").pages[0].mediabox.width) == 612


def test_pdf_extract_images_and_zip(api, pdf, images, tmp_path):
    r = run(api, "pdf_extract_images", input=pdf, output=str(tmp_path / "i.zip"))
    assert r["images"] == 2  # red + blue (identical decoded images deduplicated by MD5)
    with zipfile.ZipFile(tmp_path / "i.zip") as z:
        assert sorted(z.namelist()) == ["extracted_0.png", "extracted_1.png"]
    jp = run(api, "pdf_from_images", inputs=images[:2], quality=80, output=str(tmp_path / "j.pdf"))
    r = run(api, "pdf_extract_images", input=jp["output"], output=str(tmp_path / "j.zip"))
    with zipfile.ZipFile(tmp_path / "j.zip") as z:
        names = sorted(z.namelist())
        assert any(n.endswith(".jpg") for n in names)
        assert z.read([n for n in names if n.endswith(".jpg")][0])[:2] == b"\xff\xd8"  # original JPEG bytes
    empty = make_pdf(tmp_path / "e.pdf", 1, extras=False)
    w = PdfWriter(clone_from=PdfReader(empty))
    del w.pages[0]["/Resources"]["/XObject"]
    w.write(tmp_path / "noimg.pdf")
    r = run(api, "pdf_extract_images", input=str(tmp_path / "noimg.pdf"), output=str(tmp_path / "n.zip"))
    assert r["images"] == 0 and not (tmp_path / "n.zip").exists()
    r = run(api, "pdf_zip", input=pdf, interval=2, output=str(tmp_path / "parts.zip"))
    assert r["documents"] == ["doc_0.pdf", "doc_1.pdf"]
    with zipfile.ZipFile(tmp_path / "parts.zip") as z:
        assert [len(PdfReader(io.BytesIO(z.read(n))).pages) for n in r["documents"]] == [2, 1]
    err(api, ErrorCode.INVALID_PARAMS, "pdf_zip", input=pdf, output=str(tmp_path / "x.pdf"))


def test_pdf_contact_sheet(api, images, tmp_path):
    r = run(api, "pdf_contact_sheet", inputs=images, columns=2, rows=2, caption_fields=it.CAPTION_FIELDS,
            custom_caption_text="Hello\n\nWorld", output=str(tmp_path / "cs.pdf"))
    assert r["pages"] == 2
    rd = PdfReader(tmp_path / "cs.pdf")
    p = rd.pages[0]
    assert [round(float(v), 2) for v in p.mediabox] == [0, 0, 595.28, 841.89]
    t = p.extract_text()
    assert "001 — img0.png — img0" in t and "80 × 60" in t and "Hello" in t and "World" in t
    assert len(p["/Resources"]["/XObject"]) == 4
    r = run(api, "pdf_contact_sheet", inputs=images[:1], caption_fields=[], output=str(tmp_path / "nc.pdf"))
    assert PdfReader(tmp_path / "nc.pdf").pages[0].extract_text().strip() == ""
    err(api, ErrorCode.INVALID_PARAMS, "pdf_contact_sheet", inputs=images, caption_fields=["x"],
        output=str(tmp_path / "x.pdf"))
    assert it._human_size(500) == "500 B" and it._human_size(1536) == "1.5 kB"
    assert it._human_size(5 * 1024 * 1024) == "5.0 MB"


HOCR = """<div class='ocr_page' id='page_1' title='image "x.png"; bbox 0 0 400 200; ppageno 0'>
<span class='ocr_line' title="bbox 10 20 300 60"><span class='ocrx_word' id='w1' title='bbox 10 20 120 60; x_wconf 95'>Hello</span>
<span class='ocrx_word' id='w2' title='bbox 140 20 300 60; x_wconf 90'><strong>W&amp;rld</strong></span></span></div>"""


def test_hocr_parse_and_searchable_pdf(api, tmp_path):
    box, words = it.parse_hocr(HOCR)
    assert box == (400.0, 200.0) and [w[4] for w in words] == ["Hello", "W&rld"]
    img = tmp_path / "scan.png"
    save_image(np.ones((100, 200, 3), np.float32), img)  # hOCR box is 2x the image size -> scaled
    (tmp_path / "scan.hocr").write_text(HOCR)
    (tmp_path / "scan.txt").write_text("first line\n\nsecond line\n")
    img2 = tmp_path / "scan2.png"
    save_image(np.ones((100, 200, 3), np.float32), img2)
    r = run(api, "pdf_searchable", inputs=[str(img), str(img2)], engine="sidecar",
            ocr_files=[str(tmp_path / "scan.hocr"), str(tmp_path / "scan.txt")], output=str(tmp_path / "s.pdf"))
    rd = PdfReader(tmp_path / "s.pdf")
    assert r["pages"] == 2
    assert "Hello" in rd.pages[0].extract_text() and "W&rld" in rd.pages[0].extract_text()
    assert "first line" in rd.pages[1].extract_text() and "second line" in rd.pages[1].extract_text()
    page = render_page(tmp_path / "s.pdf")
    assert page.min() > 250  # text layer is invisible (rendering mode 3)
    content = rd.pages[0].get_contents().get_data().decode("latin-1")
    assert "3 Tr" in content and "Tz" in content
    err(api, ErrorCode.INVALID_PARAMS, "pdf_searchable", inputs=[str(img)], engine="sidecar", ocr_files=[],
        output=str(tmp_path / "x.pdf"))
    from lazyroom.core import backends

    if not backends.is_available("tesseract"):
        err(api, ErrorCode.BACKEND_UNAVAILABLE, "pdf_searchable", inputs=[str(img)], output=str(tmp_path / "x.pdf"))


def test_pdf_from_comic(api, images, tmp_path):
    from PIL import Image

    arc = tmp_path / "book.cbz"
    wide = tmp_path / "wide.jpg"
    Image.new("RGB", (300, 100), (200, 30, 30)).save(wide, quality=90)
    with zipfile.ZipFile(arc, "w") as z:
        z.write(images[0], "pages/p10.png")
        z.write(images[2], "pages/p2.png")
        z.write(wide, "pages/p1.jpg")
        z.writestr("readme.txt", "ignore me")
    r = run(api, "pdf_from_comic", input=str(arc), output=str(tmp_path / "c.pdf"))
    rd = PdfReader(tmp_path / "c.pdf")
    assert r["images"] == 3 and [float(p.mediabox.width) for p in rd.pages] == [300, 100, 80]  # natural order
    assert rd.trailer["/Root"]["/ViewerPreferences"]["/Direction"] == "/L2R"
    xo = rd.pages[0]["/Resources"]["/XObject"]["/Im0"].get_object()
    assert xo["/Filter"] == "/DCTDecode" and xo.get_data()[:2] == b"\xff\xd8"  # original JPEG embedded as-is
    r = run(api, "pdf_from_comic", input=str(arc), split_wide_pages=True, reading_direction="right_to_left",
            skip_cover=True, reverse_page_order=True, page_size="a5", output=str(tmp_path / "c2.pdf"))
    rd = PdfReader(tmp_path / "c2.pdf")
    # skip cover (p1.jpg), reverse -> p10, p2; both are wider than 1.2x their height -> split in two
    assert len(rd.pages) == 4 and {float(p.mediabox.width) for p in rd.pages} == {420}
    run(api, "pdf_from_comic", input=str(arc), split_wide_pages=True, reading_direction="right_to_left",
        output=str(tmp_path / "c3.pdf"))
    assert len(PdfReader(tmp_path / "c3.pdf").pages) == 6  # every spread split into two pages
    (tmp_path / "x.cbr").write_bytes(b"Rar!\x1a\x07\x00junk")
    err(api, ErrorCode.UNSUPPORTED_FORMAT, "pdf_from_comic", input=str(tmp_path / "x.cbr"),
        output=str(tmp_path / "x.pdf"))
    with zipfile.ZipFile(tmp_path / "empty.cbz", "w") as z:
        z.writestr("a.txt", "x")
    err(api, ErrorCode.INVALID_PARAMS, "pdf_from_comic", input=str(tmp_path / "empty.cbz"),
        output=str(tmp_path / "x.pdf"))


def test_pdf_path_policy_and_overwrite(api, pdf, tmp_path):
    err(api, ErrorCode.PATH_NOT_ALLOWED, "pdf_repair", input=pdf, output="/proc/forbidden.pdf")
    run(api, "pdf_repair", input=pdf, output=str(tmp_path / "o.pdf"))
    err(api, ErrorCode.OUTPUT_EXISTS, "pdf_repair", input=pdf, output=str(tmp_path / "o.pdf"))
    assert run(api, "pdf_repair", input=pdf, output=str(tmp_path / "o.pdf"), overwrite=True)["pages"] == 3


def test_tool_run_through_registry(api, images, tmp_path):
    try:
        res = api.tool_run("image_cutting", {"inputs": [images[0]], "output": str(tmp_path / "via_api.png")})
    except ImportError as e:  # another tool module is mid-edit elsewhere
        pytest.skip(f"tool registry import failed: {e}")
    assert res["tool"] == "image_cutting" and res["count"] == 1 and "ms" in res


# =============================================================================
# Resize by Limits / Screenshot framing
# =============================================================================

def test_limits_size_semantics():
    ls = it.limits_size
    assert ls(400, 300, 200, 0, "recode", False) == (200, 150)
    assert ls(400, 300, 0, 100, "skip", False) == (133, 100)  # shrinking happens in every mode
    assert ls(400, 300, 200, 200, "recode", False) == (200, 150)
    assert ls(100, 50, 200, 200, "recode", False) == (100, 50)
    assert ls(100, 50, 200, 200, "skip", False) is None
    assert ls(100, 50, 200, 200, "zoom", False) == (200, 100)
    assert ls(100, 50, 300, 0, "zoom", False) == (300, 150)
    assert ls(100, 50, 0, 75, "zoom", False) == (150, 75)
    assert ls(50, 100, 200, 100, "zoom", False) == (50, 100)
    assert ls(50, 100, 200, 100, "zoom", True) == (100, 200)  # limit box rotated for portrait images
    assert ls(300, 300, 100, 100, "recode", False) == (100, 100)


def test_limits_resize_tool(api, images, tmp_path):
    r = run(api, "limits_resize", inputs=images, width=100, height=100, mode="skip", output_dir=str(tmp_path / "l"))
    assert len(r["skipped"]) == 3 and r["count"] == 2  # 80x60, 90x60, 100x60 already fit inside the box
    assert all(max(o["width"], o["height"]) <= 100 for o in r["outputs"])
    r = run(api, "limits_resize", inputs=[images[0]], width=160, mode="zoom", output=str(tmp_path / "z.png"))
    assert (r["outputs"][0]["width"], r["outputs"][0]["height"]) == (160, 120)
    err(api, ErrorCode.INVALID_PARAMS, "limits_resize", inputs=images, output=str(tmp_path / "x.png"), width=10)
    err(api, ErrorCode.INVALID_PARAMS, "limits_resize", inputs=[images[0]], output=str(tmp_path / "x.png"))


def test_screenshot_framing(api, tmp_path):
    src = tmp_path / "shot.png"
    img = synthetic_image(200, 100, seed=4)
    save_image(img, src)
    r = run(api, "screenshot_framing", input=str(src), preset="paper", output=str(tmp_path / "f.png"))
    # unit = 1, blur = 4, offset = 2, inset = 12 + 8 + 2 = 22
    assert (r["output"]["width"], r["output"]["height"]) == (244, 144)
    out = load_image(tmp_path / "f.png")[0]
    assert out.shape[2] == 3
    np.testing.assert_allclose(out[72, 122], img[50, 100], atol=2 / 255)  # image centred, 1:1
    assert np.abs(out[1, 1] - np.array([0xF8, 0xFA, 0xFC]) / 255).max() < 3 / 255  # gradient start colour
    assert out[122 + 3, 122].mean() < out[1, 122].mean() - 0.01  # soft shadow below the image
    assert np.abs(out[23, 23] - img[0, 0]).max() > 0.05  # rounded corner shows background
    r = run(api, "screenshot_framing", input=str(src), aspect_ratio="1:1", shadow=False, gradient=False,
            preset="custom", start_color="#102030", output=str(tmp_path / "sq.png"))
    assert r["output"]["width"] == r["output"]["height"] == 224 and r["colors"][0] == "#102030"
    out = load_image(tmp_path / "sq.png")[0]
    assert np.abs(out[0, 0] - np.array([0x10, 0x20, 0x30]) / 255).max() < 1 / 255 + 1e-6
    auto = run(api, "screenshot_framing", input=str(src), output=str(tmp_path / "auto.png"))
    assert len(auto["colors"]) == 2 and auto["colors"][0] != auto["colors"][1]
