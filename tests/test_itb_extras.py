"""ImageToolbox extras: watermark ports, hidden (LSB / FD) watermarks, filter templates (text / QR) and formats.

See docs/dev/port_reports/itb_extras.md.  Run with LAZYROOM_ITB_ONLY empty (templates need all ITB filters).
"""
from __future__ import annotations

import base64
import datetime as dt
import io
import io as _io
import math
import random
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from lazyroom.core import io as pmio
from lazyroom.core.context import RenderContext
from lazyroom.core.errors import ErrorCode, LazyRoomError
from lazyroom.core.io import save_image
from lazyroom.core.registry import get_op
from lazyroom.ops import _wm_render as R
from lazyroom.ops import watermark_ext  # noqa: F401  (registers the draw.* watermark ops)
from lazyroom.tools import _itb_codecs as itbc
from lazyroom.tools import _itb_formats  # noqa: F401  (registers the format tools)
from lazyroom.tools import _itb_stego as S
from lazyroom.tools import _itb_templates as T
from lazyroom.tools import itb_extras  # noqa: F401  (registers the tools)
from lazyroom.tools._itb_template_table import FILTERS, FONTS, PALETTES
from tests.conftest import synthetic_image

UI = "^^.core.filters.presentation.model."
HDR = T.LINK_HEADER


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("LAZYROOM_HOME", str(tmp_path / "home"))
    from lazyroom.api import LazyRoom

    return LazyRoom()


def run(op_id, img, **params):
    spec = get_op(op_id)
    return spec.run(img, RenderContext(), spec.validate(params))


def u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# ===========================================================================
# Visible watermarks (ops/watermark_ext.py)
# ===========================================================================

def test_ops_registered_with_imagetoolbox_defaults():
    t = get_op("draw.repeating_text").defaults()
    assert t["text"] == "Watermark" and t["size"] == 0.1 and t["rotation"] == 45 and t["opacity"] == 0.5
    assert t["repeat"] is True and t["blend"] == "src_over" and t["color"] == "#000000"
    s = get_op("draw.stamp_watermark").defaults()
    assert s["position"] == "bottom_right" and s["padding"] == 20 and s["size"] == 0.2 and s["text"] == "Stamp"
    assert get_op("draw.timestamp_watermark").defaults()["format"] == "dd/MM/yyyy HH:mm"
    assert get_op("draw.timestamp_watermark").deterministic is False
    assert len(R.BLEND_MODES) == 29 and R.BLEND_MODES[3] == "src_over"


def test_repeating_text_tiles_on_a_grid_from_the_origin():
    img = np.full((300, 400, 3), 1.0, np.float32)
    out = run("draw.repeating_text", img, rotation=0, size=0.2, opacity=1.0)
    tile = R.text_bitmap("Watermark", 0.2, 400, (0, 0, 0, 1), (0, 0, 0, 0), "sans-serif", 1.0)
    th, tw = tile.shape[:2]
    assert tw == round(400 * 0.2)
    assert out.min() < 0.5  # text drawn
    np.testing.assert_allclose(out[:th, :tw], out[th:2 * th, tw:2 * tw], atol=1e-6)  # BitmapShader REPEAT
    np.testing.assert_allclose(out[:th, :tw], out[:th, 2 * tw:3 * tw], atol=1e-6)


def test_repeating_text_alpha_is_applied_twice_like_androidwm():
    img = np.ones((200, 400, 3), np.float32)
    out = run("draw.repeating_text", img, repeat=False, rotation=0, size=0.9, opacity=0.5, position=[0.0, 0.0])
    # glyph cores: black text, alpha 0.5 in the text paint and 0.5 again in the bitmap paint -> 25 %
    assert abs(float(out.min()) - 0.75) < 0.02
    out = run("draw.repeating_text", img, repeat=False, rotation=0, size=0.9, opacity=0.5, position=[0.0, 0.0],
              background="#FF000080")
    # background colour (alpha 128/255) is only scaled by the bitmap paint alpha
    a = 128 / 255 * 0.5
    tile = R.text_bitmap("Watermark", 0.9, 400, (0, 0, 0, 1), (1, 0, 0, 128 / 255), "sans-serif", 0.5)
    y = tile.shape[0] - 2
    np.testing.assert_allclose(out[y, 2], [1.0, 1 - a, 1 - a], atol=0.01)
    assert np.allclose(out[tile.shape[0] + 3:], 1.0)  # nothing below the single bitmap


def test_repeating_text_single_position_and_porter_duff_modes():
    img = synthetic_image(200, 160)
    out = run("draw.repeating_text", img, repeat=False, rotation=0, position=[0.5, 0.5], opacity=1.0)
    diff = np.abs(out - img).max(axis=2) > 1e-4
    ys, xs = np.nonzero(diff)
    assert xs.min() >= 100 and ys.min() >= 80
    assert np.allclose(run("draw.repeating_text", img, blend="dst"), img)
    cleared = run("draw.repeating_text", img, blend="clear")
    assert cleared.shape[2] == 4 and float(cleared[..., 3].max()) == 0.0  # Clear over the whole tiled canvas
    single_clear = run("draw.repeating_text", img, blend="clear", repeat=False, rotation=0, position=[0.5, 0.5])
    assert single_clear[..., 3].min() == 0.0 and single_clear[0, 0, 3] == 1.0  # only inside the bitmap rect


def test_composite_matches_skia_formulas():
    d = np.array([[[0.2, 0.3, 0.4, 0.5]]], np.float32)
    s = np.array([[[0.6, 0.1, 0.0, 0.6]]], np.float32)
    np.testing.assert_allclose(R.composite(d, s, "src_over")[0, 0], s[0, 0] + d[0, 0] * 0.4, atol=1e-6)
    np.testing.assert_allclose(R.composite(d, s, "dst_in")[0, 0], d[0, 0] * 0.6, atol=1e-6)
    np.testing.assert_allclose(R.composite(d, s, "xor")[0, 0], s[0, 0] * 0.5 + d[0, 0] * 0.4, atol=1e-6)
    m = R.composite(d, s, "multiply")[0, 0]  # Skia kMultiply: S(1-Da) + D(1-Sa) + S*D
    np.testing.assert_allclose(m[:3], s[0, 0, :3] * 0.5 + d[0, 0, :3] * 0.4 + s[0, 0, :3] * d[0, 0, :3], atol=1e-6)
    for mode in R.BLEND_MODES:
        out = R.composite(d, s, mode)
        assert out.shape == d.shape and np.isfinite(out).all()


def test_rotation_matches_android_bounding_box():
    layer = np.ones((20, 100, 4), np.float32)
    r = R.rotate_layer(layer, 45)
    side = round((100 + 20) * math.sqrt(0.5))
    assert abs(r.shape[0] - side) <= 1 and abs(r.shape[1] - side) <= 1
    r90 = R.rotate_layer(layer, 90)
    assert r90.shape[:2] == (100, 20)


def test_repeating_image_scales_logo_to_width_fraction(tmp_path):
    logo = tmp_path / "logo.png"
    save_image(synthetic_image(60, 30, seed=2), logo)
    img = np.full((240, 320, 3), 0.5, np.float32)
    out = run("draw.repeating_image", img, image=str(logo), size=0.25, rotation=0, opacity=1.0)
    lw = round(320 * 0.25)
    lh = round(30 * lw / 60)
    np.testing.assert_allclose(out[:lh, :lw], out[lh:2 * lh, lw:2 * lw], atol=1e-6)
    assert not np.allclose(out[:lh, :lw], 0.5)


@pytest.mark.parametrize("pos,check", [("bottom_right", "br"), ("top_left", "tl"), ("center", "c")])
def test_stamp_placement_and_padding(pos, check):
    w, h, pad = 400, 300, 20
    img = np.ones((h, w, 3), np.float32)
    out = run("draw.stamp_watermark", img, position=pos, padding=pad, background="#000000", color="#FFFFFF",
              opacity=1.0)
    ys, xs = np.nonzero(out.max(axis=2) < 0.5)
    vpad = pad - 6 * pad / 20
    width = xs.max() - xs.min() + 1
    assert abs(width - (0.2 * w - 2 * pad)) <= 2  # text bitmap shrunk by twice the padding
    if check == "br":
        assert abs((w - 1 - xs.max()) - pad) <= 1 and abs((h - 1 - ys.max()) - vpad) <= 1
    elif check == "tl":
        assert abs(xs.min() - pad) <= 1 and abs(ys.min() - vpad) <= 1
    else:
        assert abs((xs.min() + xs.max()) / 2 - w / 2) <= 1.5 and abs((ys.min() + ys.max()) / 2 - h / 2) <= 1.5


def test_stamp_box_has_rounded_corners_and_unscaled_alpha():
    img = np.ones((300, 400, 3), np.float32)
    out = run("draw.stamp_watermark", img, background="#000000", opacity=0.3)
    ys, xs = np.nonzero(out.max(axis=2) < 0.5)
    assert out[ys.max(), xs.max()].min() > 0.5  # corner pixel left uncovered by the 12 px radius
    assert out[(ys.min() + ys.max()) // 2, xs.min() + 1].max() < 0.05  # box opaque although alpha = 0.3


def test_java_simple_date_format():
    d = dt.datetime(2023, 7, 4, 18, 5, 9, 7000)
    f = R.java_date_format
    assert f("dd/MM/yyyy HH:mm", d) == "04/07/2023 18:05"
    assert f("EEEE, MMMM d, yyyy h:mm a", d) == "Tuesday, July 4, 2023 6:05 PM"
    assert f("EEE MMM yy", d) == "Tue Jul 23"
    assert f("yyyy-MM-dd'T'HH:mm:ss.SSS", d) == "2023-07-04T18:05:09.007"
    assert f("'o''clock' hh K k", d) == "o'clock 06 6 18"
    assert f("D u", d) == "185 2"
    with pytest.raises(R.DateFormatError):
        f("yyyy-MM-dd Q", d)
    # ImageToolbox timestamp(): an invalid pattern prints Date().time (epoch millis)
    assert R.timestamp_text("Q", "2023:07:04 18:05:09") == str(int(dt.datetime(2023, 7, 4, 18, 5, 9).timestamp() * 1000))


def test_parse_date_sources():
    assert R.parse_date("2021:12:31 23:59:58") == dt.datetime(2021, 12, 31, 23, 59, 58)
    assert R.parse_date("2021-12-31T23:59:58") == dt.datetime(2021, 12, 31, 23, 59, 58)
    assert R.parse_date("1700000000000").year == 2023
    assert abs((R.parse_date("") - dt.datetime.now().astimezone()).total_seconds()) < 5
    from lazyroom.core.errors import LazyRoomError

    with pytest.raises(LazyRoomError):
        R.parse_date("yesterday")


def test_timestamp_op_uses_explicit_date_deterministically():
    img = synthetic_image(320, 240)
    a = run("draw.timestamp_watermark", img, date="2020:01:02 03:04:05", format="yyyy")
    b = run("draw.timestamp_watermark", img, date="2020:01:02 03:04:05", format="yyyy")
    c = run("draw.timestamp_watermark", img, date="2021:01:02 03:04:05", format="yyyy")
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    ref = run("draw.stamp_watermark", img, text="2020")
    np.testing.assert_allclose(a, ref, atol=1e-6)


# ===========================================================================
# Hidden watermarks: reference implementation translated literally from Watermark.cpp
# (uint32 RGBA_8888 words on a little-endian device, shifts {16, 8, 0})
# ===========================================================================

def _words(rgba):
    a = rgba[..., 3] if rgba.shape[2] == 4 else np.full(rgba.shape[:2], 255, np.uint8)
    r, g, b = (rgba[..., i].astype(np.uint32) for i in range(3))
    return (r | (g << 8) | (b << 16) | (a.astype(np.uint32) << 24)).astype(np.uint32)


def _unwords(words):
    return np.stack([(words >> s) & 0xFF for s in (0, 8, 16, 24)], -1).astype(np.uint8)


def _frame_bit(magic, wm, is_image, index):
    if index < 32:
        return (magic >> (31 - index)) & 1
    if index == 32:
        return int(is_image)
    if index < 40:
        return 0
    if index < 64:
        return (len(wm) >> (63 - index)) & 1
    p = index - 64
    return (wm[p // 8] >> (7 - p % 8)) & 1


def ref_lsb_embed(rgba, wm: bytes, is_image: bool):
    words = _words(rgba).copy()
    h, w = words.shape
    shifts = (16, 8, 0)
    for index in range(64 + len(wm) * 8):
        pi = index // 3
        x, y = pi % w, pi // w
        shift = shifts[index % 3]
        bit = _frame_bit(0x4C534257, wm, is_image, index)
        words[y, x] = (int(words[y, x]) & ~(1 << shift) & 0xFFFFFFFF) | (bit << shift)
    return _unwords(words)


def ref_lsb_extract(rgba, is_image: bool):
    words = _words(rgba)
    h, w = words.shape
    shifts = (16, 8, 0)

    def read_bit(index):
        pi = index // 3
        return (int(words[pi // w, pi % w]) >> shifts[index % 3]) & 1

    magic = 0
    for i in range(32):
        magic = (magic << 1) | read_bit(i)
    if magic != 0x4C534257 or read_bit(32) != int(is_image):
        return None
    length = 0
    for i in range(40, 64):
        length = (length << 1) | read_bit(i)
    out = bytearray(length)
    for i in range(length * 8):
        out[i // 8] = ((out[i // 8] << 1) | read_bit(64 + i)) & 0xFF
    return bytes(out)


_C = np.array([[math.cos((2 * c + 1) * f * math.pi / 16) for c in range(8)] for f in range(8)])
_SC = np.array([math.sqrt(0.125)] + [0.5] * 7)


def _ref_fdct(inp):
    return np.einsum("yx,vy,hx->vh", inp, _C, _C) * _SC[:, None] * _SC[None, :]


def _ref_idct(spec):
    return np.einsum("vh,vy,hx->yx", spec * _SC[:, None] * _SC[None, :], _C, _C)


def _ref_read_fd_bit(words, bx, by, shift):
    inp = ((words[by * 8:by * 8 + 8, bx * 8:bx * 8 + 8] >> shift) & 0xFF).astype(np.float64)
    sp = _ref_fdct(inp)
    return sp[2][3] >= sp[3][2]


def ref_fd_embed(rgba, wm: bytes, is_image: bool):
    words = _words(rgba).copy()
    cols = words.shape[1] // 8
    shifts = (16, 8, 0)
    for index in range(64 + len(wm) * 8):
        bit = bool(_frame_bit(0x4644574D, wm, is_image, index))
        enc = 0
        for replica in range(3):
            phys = index * 3 + replica
            block = phys // 3
            bx, by, shift = block % cols, block // cols, shifts[phys % 3]
            ok = False
            for attempt in (1, 2, 3):
                inp = ((words[by * 8:by * 8 + 8, bx * 8:bx * 8 + 8] >> shift) & 0xFF).astype(np.float64)
                sp = _ref_fdct(inp)
                diff = sp[2][3] - sp[3][2]
                target = 80.0 * attempt if bit else -80.0 * attempt
                adj = (target - diff) / 2.0
                sp[2][3] += adj
                sp[3][2] -= adj
                o = _ref_idct(sp)
                comp = np.floor(np.clip(o, 0, 255) + 0.5).astype(np.uint32)
                blk = words[by * 8:by * 8 + 8, bx * 8:bx * 8 + 8]
                words[by * 8:by * 8 + 8, bx * 8:bx * 8 + 8] = (blk & ~np.uint32(0xFF << shift)) | (comp << shift)
                if _ref_read_fd_bit(words, bx, by, shift) == bit:
                    ok = True
                    break
            enc += ok
        assert enc > 1
    return _unwords(words)


def ref_fd_extract(rgba, is_image: bool):
    words = _words(rgba)
    cols = words.shape[1] // 8
    shifts = (16, 8, 0)

    def frame_bit(index):
        ones = 0
        for replica in range(3):
            phys = index * 3 + replica
            block = phys // 3
            ones += _ref_read_fd_bit(words, block % cols, block // cols, shifts[phys % 3])
        return int(ones > 1)

    magic = 0
    for i in range(32):
        magic = (magic << 1) | frame_bit(i)
    if magic != 0x4644574D or frame_bit(32) != int(is_image):
        return None
    length = 0
    for i in range(40, 64):
        length = (length << 1) | frame_bit(i)
    out = bytearray(length)
    for i in range(length * 8):
        out[i // 8] = ((out[i // 8] << 1) | frame_bit(64 + i)) & 0xFF
    return bytes(out)


@pytest.fixture(scope="module")
def photo8():
    return u8(synthetic_image(96, 88, seed=5))


def test_modified_utf8_matches_jni():
    assert S.to_modified_utf8("A\x00é") == b"A\xc0\x80\xc3\xa9"
    assert S.to_modified_utf8("😀") == b"\xed\xa0\xbd\xed\xb8\x80"  # surrogate pair, 3 bytes each (CESU-8)
    for s in ("plain", "ünï €", "😀 x \x00 y", ""):
        assert S.from_modified_utf8(S.to_modified_utf8(s)) == s
    assert S.from_modified_utf8(b"abc\x00def") == "abc"  # NewStringUTF(c_str()) stops at NUL


def test_lsb_layout_is_bit_identical_to_reference(photo8):
    payload = S.to_modified_utf8("© ITB 😀")
    ours = S.lsb_embed(photo8, payload, False)
    ref = ref_lsb_embed(photo8, payload, False)[..., :3]
    assert np.array_equal(ours, ref)
    # channel order of the first pixel: B, G, R carry the first three magic bits (0x4C = 0b0100_1100)
    assert [ours[0, 0, 2] & 1, ours[0, 0, 1] & 1, ours[0, 0, 0] & 1] == [0, 1, 0]
    assert int(np.abs(ours.astype(int) - photo8).max()) <= 1


def test_lsb_interoperates_both_ways(photo8):
    payload = S.to_modified_utf8("hidden: ünï 😀")
    assert ref_lsb_extract(S.lsb_embed(photo8, payload, False), False) == payload
    assert S.lsb_extract(ref_lsb_embed(photo8, payload, False), False) == payload
    img_payload = S.image_payload_lsb(u8(synthetic_image(8, 6))).encode()
    marked = ref_lsb_embed(photo8, img_payload, True)
    det = S.detect(marked)
    assert det.method == "lsb" and det.kind == "image" and det.image.size == (8, 6)
    assert S.lsb_extract(marked, False) is None  # the is-image flag is checked


def test_fd_interoperates_both_ways(photo8):
    payload = b"FD"
    ours = S.fd_embed(photo8, payload, False)
    ref = ref_fd_embed(photo8, payload, False)[..., :3]
    assert int(np.abs(ours.astype(int) - ref.astype(int)).max()) <= 1  # same algorithm (float order may differ)
    assert ref_fd_extract(ours, False) == payload
    assert S.fd_extract(ref, False) == payload
    assert S.detect(ref).method == "fd"


def test_capacity_errors_and_frame_header(photo8):
    with pytest.raises(S.StegoError):
        S.lsb_embed(photo8[:4, :4], b"x" * 10, False)
    with pytest.raises(S.StegoError):
        S.fd_embed(photo8, b"x" * 40, False)  # 11x12 blocks < 64 + 320 bits
    bits = S.frame_bits(S.FD_MAGIC, b"ab", True)
    assert len(bits) == 64 + 16 and bits[32] == 1 and not bits[33:40].any()
    assert int("".join(map(str, bits[40:64])), 2) == 2


def test_semi_transparent_pixels_use_premultiplied_storage():
    rgba = u8(synthetic_image(64, 64, alpha=True))
    rgba[..., 3] = np.maximum(rgba[..., 3], 200)
    payload = b"alpha!"
    marked = S.lsb_embed(rgba, payload, False)
    assert np.array_equal(marked[..., 3], rgba[..., 3])
    prem = S._premul(marked[..., :3], marked[..., 3])  # what Android sees after decoding the PNG
    flat = prem[..., [2, 1, 0]].reshape(-1)
    bits = S.frame_bits(S.LSB_MAGIC, payload, False)
    assert np.array_equal(flat[:len(bits)] & 1, bits)
    assert S.lsb_extract(marked, False) == payload


def test_fd_survives_high_quality_jpeg_but_lsb_does_not():
    img = u8(synthetic_image(256, 192, seed=9))
    fd = S.fd_embed(img, b"jpeg", False)
    lsb = S.lsb_embed(img, b"jpeg", False)

    def jpeg(a, q):
        b = io.BytesIO()
        Image.fromarray(a).save(b, "JPEG", quality=q)
        return np.asarray(Image.open(b))

    assert S.fd_extract(jpeg(fd, 95), False) == b"jpeg"
    assert S.lsb_extract(jpeg(lsb, 95), False) is None


def test_fd_image_payload_follows_bitmap_to_fd_string():
    wm = u8(synthetic_image(80, 60))
    cap = S.fd_payload_capacity(960, 1280)
    enc = S.image_payload_fd(wm, cap)
    assert enc is not None and len(enc) <= cap
    assert Image.open(io.BytesIO(base64.b64decode(enc))).format == "JPEG"
    assert S.image_payload_fd(wm, 100) is None  # cannot fit: ImageToolbox fails too


def test_hidden_watermark_tools_round_trip(api, tmp_path):
    src = tmp_path / "in.png"
    save_image(synthetic_image(320, 240), src)
    out = tmp_path / "marked.png"
    r = api.tool_run("watermark_hidden_embed", {"input": str(src), "output": str(out), "text": "© 2026 Ünïcødé 😀"})
    assert r["verified"] and r["method"] == "lsb" and r["kind"] == "text"
    x = api.tool_run("watermark_hidden_extract", {"input": str(out)})
    assert x["found"] and x["text"] == "© 2026 Ünïcødé 😀" and x["method"] == "lsb"
    # cross-check the file with the literal reference reader
    arr = np.asarray(Image.open(out).convert("RGB"))
    assert S.from_modified_utf8(ref_lsb_extract(arr, False)) == "© 2026 Ünïcødé 😀"
    clean = api.tool_run("watermark_hidden_extract", {"input": str(src)})
    assert clean["found"] is False


def test_hidden_image_watermark_tool(api, tmp_path):
    src, logo = tmp_path / "in.png", tmp_path / "logo.png"
    save_image(synthetic_image(400, 300), src)
    save_image(synthetic_image(120, 90, seed=4), logo)
    out = tmp_path / "marked.png"
    r = api.tool_run("watermark_hidden_embed", {"input": str(src), "output": str(out), "image": str(logo),
                                                 "size": 0.1})
    assert r["kind"] == "image"
    x = api.tool_run("watermark_hidden_extract", {"input": str(out), "image_output": str(tmp_path / "wm.png")})
    assert x["kind"] == "image" and x["image"]["width"] == 40 and x["image"]["height"] == 30
    assert (tmp_path / "wm.png").exists()


def test_hidden_watermark_tool_rejects_lossy_lsb_and_verifies_fd_jpeg(api, tmp_path):
    from lazyroom.core.errors import LazyRoomError

    src = tmp_path / "in.png"
    save_image(synthetic_image(320, 240), src)
    with pytest.raises(LazyRoomError) as e:
        api.tool_run("watermark_hidden_embed", {"input": str(src), "output": str(tmp_path / "a.jpg"), "text": "x"})
    assert e.value.code.value == "INVALID_PARAMS"
    r = api.tool_run("watermark_hidden_embed", {"input": str(src), "output": str(tmp_path / "a.jpg"), "text": "fd",
                                                 "method": "fd"})
    assert r["verified"] and "warning" in r
    assert api.tool_run("watermark_hidden_extract", {"input": str(tmp_path / "a.jpg")})["text"] == "fd"


# ===========================================================================
# Filter templates
# ===========================================================================

@pytest.mark.parametrize("value,expected", [
    (0.5, "0.5"), (1, "1.0"), (100, "100.0"), (0.001, "0.001"), (1e-4, "1.0E-4"), (1e7, "1.0E7"),
    (12345678, "1.2345678E7"), (-16777216, "-1.6777216E7"), (2.5e-5, "2.5E-5"), (0.1, "0.1"), (1 / 3, "0.33333334"),
    (7000.0, "7000.0"), (-0.35, "-0.35"), (0.0, "0.0"), (9999999.0, "9999999.0"),
])
def test_java_float_to_string(value, expected):
    assert T.jfloat(value) == expected


def test_template_validity_rule():
    assert T.is_valid_template(HDR + "x+" + UI + "UiBrightnessFilter:Float:0.5")
    assert not T.is_valid_template("x+" + UI + "UiBrightnessFilter:Float:0.5")
    assert not T.is_valid_template(HDR + "hello")
    from lazyroom.core.errors import LazyRoomError

    with pytest.raises(LazyRoomError):
        T.import_template("https://example.com/?x+y")


def _b(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _imp(*filters, name="Look"):
    return T.import_template(HDR + name + "+" + ",".join(filters))


def _params(res, i=0):
    s = res["steps"][i]
    spec = get_op(s["op"])
    return s["op"], spec.params_json(spec.validate(s["params"]))


# Template strings written exactly as ImageToolbox's FilterSerializationUtils/Mappings build them.
ITB_CASES = [
    (UI + "UiBrightnessFilter:Float:-0.35", "filter.brightness", {"bias": -0.35}),
    (UI + "UiGaussianBlurFilter:Triple{Float$Float$BlurEdgeMode}:30.0$7.5$Clamp", "filter.gaussian_blur",
     {"radius": 30, "sigma": 7.5, "edge_mode": "clamp"}),
    (UI + "UiColorMatrix3x3Filter:FloatArray:0.393$0.769$0.189$0.349$0.686$0.168$0.272$0.534$0.131",
     "filter.color_matrix_3x3", {"matrix": [0.393, 0.769, 0.189, 0.349, 0.686, 0.168, 0.272, 0.534, 0.131]}),
    (UI + "UiVignetteFilter:Triple{Float$Float$ColorModel}:0.1$0.9$-16711936", "filter.vignette",
     {"start": 0.1, "end": 0.9, "color": "#00FF00"}),
    (UI + "UiColorOverlayFilter:FilterValueWrapper{ColorModel}:-2130771968", "filter.color_overlay",
     {"color": "#FF000080"}),
    (UI + "UiFastGaussianBlur2DFilter:Pair{Float$BlurEdgeMode}:25.0$Wrap", "filter.fast_gaussian_blur_2d",
     {"radius": 25, "edge_mode": "wrap"}),
    (UI + "UiAtkinsonDitheringFilter:Pair{Float$Boolean}:128.0$true", "filter.atkinson_dithering",
     {"threshold": 128, "gray_scale": True}),
    (UI + "UiClaheHSVFilter:ClaheParams:0.75$4$6$64", "filter.clahe_hsv",
     {"threshold": 0.75, "grid_size_x": 4, "grid_size_y": 6, "bins_count": 64}),
    (UI + "UiLinearGaussianBlurFilter:LinearGaussianParams:31$12.5$Reflect$REC709", "filter.linear_gaussian_blur",
     {"strength": 31, "sigma": 12.5, "edge_mode": "reflect", "transfer_function": "rec709"}),
    (UI + "UiMirrorFilter:Pair{Float$MirrorSide}:0.3$TopToBottom", "filter.mirror",
     {"center": 0.3, "side": "top_to_bottom"}),
    (UI + "UiPolarCoordinatesFilter:PolarCoordinatesType:INVERT_IN_CIRCLE", "filter.polar_coordinates",
     {"type": "invert_in_circle"}),
    (UI + "UiFalseColorFilter:Pair{ColorModel$ColorModel}:-65536$-16776961", "filter.false_color",
     {"first_color": "#FF0000", "second_color": "#0000FF"}),
    (UI + "UiDropShadowFilter:DropShadowParams:-16777216$4.0$12.0$20.0", "filter.drop_shadow",
     {"shadow_color": "#000000", "offset_x": 4.0, "offset_y": 12.0, "blur_radius": 20.0}),
    (UI + "UiSideFadeFilter:SideFadeParams:Top$0.3", "filter.side_fade", {"side": "top", "strength": 0.3}),
    (UI + "UiEnhancedZoomBlurFilter:EnhancedZoomBlurParams:40$6.0$0.3$0.6$2.0$90.0", "filter.enhanced_zoom_blur",
     {"radius": 40, "sigma": 6.0, "blur_center_x": 0.3, "blur_center_y": 0.6, "strength": 2.0, "angle": 90.0}),
    (UI + "UiCropOrPerspectiveFilter:CropOrPerspectiveParams:0.0*0.1$1.0*0.0$0.05*1.0$0.95*0.9$false",
     "filter.crop_or_perspective", {"top_left": [0.0, 0.1], "top_right": [1.0, 0.0], "bottom_left": [0.05, 1.0],
                                    "bottom_right": [0.95, 0.9], "absolute": False}),
    (UI + "UiDistortPerspectiveFilter:DistortPerspectiveParams:0.1$0.0$0.9$0.05$0.0$1.0$1.0$0.95$true",
     "filter.distort_perspective", {"top_left": [0.1, 0.0], "top_right": [0.9, 0.05], "bottom_left": [0.0, 1.0],
                                    "bottom_right": [1.0, 0.95], "clip": True}),
    (UI + "UiWarholFilter:GmicFilterParams:" + "$".join(_b(v) for v in ("3", "4", "2.5", "30.0")), "filter.warhol",
     {"x_tiles": 3, "y_tiles": 4, "smoothness": 2.5, "color": 30.0}),
    (UI + "UiBrokenGlassFilter:ProceduralParams:"
     + _b("intensity=0.6;frequency=12.0;seed=42.0;offsetX=0.1;offsetY=0.0;scale=2.0;rotation=30.0"),
     "filter.broken_glass", {"intensity": 0.6, "frequency": 12.0, "seed": 42, "offset_x": 0.1, "scale": 2.0,
                             "rotation": 30.0}),
    (UI + "UiBokehLightsFilter:ProceduralParams:" + _b("count=10.0;color:lightColor=-16711681"),
     "filter.bokeh_lights", {"count": 10, "color": "#00FFFF", "blades": 6}),
    (UI + "UiGradientMapFilter:GradientPalette:custom;ffff0000;ff0000ff", "filter.gradient_map",
     {"gradient": [[0.0, "#FF0000"], [1.0, "#0000FF"]]}),
    (UI + "UiSmearFilter:SmearParams:30.0$0.5$0.8$150$1", "filter.smear",
     {"angle": 30.0, "density": 0.5, "mix": 0.8, "distance": 150, "shape": 1}),
    (UI + "UiRubberStampFilter:RubberStampParams:0.4$0.2$0.1$-16777216$-1", "filter.rubber_stamp",
     {"threshold": 0.4, "brush_softness": 0.2, "radius": 0.1, "first_color": "#000000", "second_color": "#FFFFFF"}),
    (UI + "UiClustered2x2DitheringFilter:Boolean:true", "filter.clustered_2x2_dithering", {"gray_scale": True}),
    (UI + "UiJetFilter:Unit:Unit", "filter.color_map", {"color_map": "jet"}),
    (UI + "UiTriToneFilter:Triple{ColorModel$ColorModel$ColorModel}:-16777216$-8355712$-1", "filter.tri_tone",
     {"shadows_color": "#000000", "midtones_color": "#808080", "highlights_color": "#FFFFFF"}),
]


@pytest.mark.parametrize("line,op_id,expected", ITB_CASES, ids=[c[1] for c in ITB_CASES])
def test_import_itb_built_strings(line, op_id, expected):
    res = _imp(line)
    assert not res["skipped"], res["warnings"]
    got_op, params = _params(res)
    assert got_op == op_id
    for k, v in expected.items():
        assert params[k] == v, (k, params[k], v)


@pytest.mark.parametrize("line,op_id,expected", [c for c in ITB_CASES if not c[0].startswith(
    (UI + "UiClustered2x2", UI + "UiBokehLights", UI + "UiBrokenGlass"))], ids=lambda c: str(c)[:40])
def test_export_is_byte_identical_to_itb(line, op_id, expected):
    res = _imp(line)
    out = T.export_template("Look", res["steps"])
    assert out["template"] == HDR + "Look+" + line


def test_export_procedural_and_gmic_match_itb_serialisation():
    step = {"op": "filter.broken_glass", "params": {"intensity": 0.6, "seed": 42}}
    out = T.export_template("P", [step])["template"]
    value = out.split(":")[-1]
    assert out.startswith(HDR + "P+" + UI + "UiBrokenGlassFilter:ProceduralParams:")
    assert base64.b64decode(value).decode() == ("intensity=0.6;frequency=8.0;seed=42.0;offsetX=0.0;offsetY=0.0;"
                                                "scale=1.0;rotation=0.0")
    step = {"op": "filter.blurred_frame", "params": {"color_balance": True, "normalization": "equalize",
                                                     "outline_color": "#FF0000"}}
    vals = [base64.b64decode(v).decode() for v in T.export_template("G", [step])["template"].split(":")[-1].split("$")]
    assert vals[5] == "true" and vals[7] == "Equalize" and vals[9] == str(-65536) and vals[0] == "30.0"


def test_import_whole_template_with_name_and_aliases():
    res = _imp(UI + "UiBrightnessFilter:Float:0.2",
               "^^.feature.filters.data.model.ContrastFilter:Float:1.5",  # data-model class names (older templates)
               UI + "UiDisplacementFilter:null",  # ImageToolbox writes ':null' for values it cannot serialise
               UI + "UiNoSuchFilter:Float:1.0",
               UI + "UiVHSNtscFilter:NtscParams:2.0$0.5",  # missing fields fall back to defaults (floatAt/intAt)
               name="My Look 2")
    assert res["name"] == "My Look 2" and len(res["steps"]) == 3 and len(res["skipped"]) == 2
    assert res["steps"][1]["op"] == "filter.contrast"
    _, p = _params(res, 2)
    assert p["amount"] == 2.0 and p["chroma_bleed"] == 0.5 and p["use_field"] == "interleaved_upper"
    assert any("null" in w for w in res["warnings"])


def test_tone_curves_import_export():
    groups = [[0.0, 0.1, 0.5, 0.6, 1.0, 0.9], [0.0, 0.0, 0.25, 0.3, 0.75, 0.8, 1.0, 1.0], [0.0, 0.2, 0.4, 0.6, 1.0],
              [0.0, 0.0, 1.0, 1.0]] + [[0.0, 0.0, 1.0, 1.0]] * 7 + [[0.0, 0.5, 1.0, 0.5]] * 6 + \
             [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5]]
    line = UI + "UiToneCurvesFilter:ToneCurvesParams:" + "$".join("*".join(T.jfloat(x) for x in g) for g in groups)
    res = _imp(line)
    _, p = _params(res)
    assert p["rgb"] == [[0.0, 0.1], [0.5, 0.6], [1.0, 0.9]]
    assert p["green"] == [[0.0, 0.0], [0.25, 0.2], [0.5, 0.4], [0.75, 0.6], [1.0, 1.0]]  # legacy 5-value curve
    assert not res["warnings"]
    out = T.export_template("Look", res["steps"])["template"]
    back = _params(_imp(out[len(HDR) + 5:]))[1]
    assert back == p
    hue = list(groups)
    hue[12] = [0.0, 0.7, 1.0, 0.5]
    res2 = _imp(UI + "UiToneCurvesFilter:ToneCurvesParams:" + "$".join("*".join(T.jfloat(x) for x in g) for g in hue))
    assert any("not supported" in w for w in res2["warnings"])


def test_gradient_palettes():
    res = _imp(UI + "UiGradientMapFilter:GradientPalette:Viridis")
    _, p = _params(res)
    assert len(p["gradient"]) == len(PALETTES["Viridis"]) == 5 and p["gradient"][0][1] == "#440154"
    assert T.export_template("x", res["steps"])["template"].endswith(":GradientPalette:Viridis")
    uneven = {"op": "filter.gradient_map", "params": {"gradient": [[0, "#000000"], [0.2, "#FF0000"], [1, "#FFFFFF"]]}}
    out = T.export_template("x", [uneven])
    assert out["template"].split(":")[-1].startswith("custom;ff000000;") and out["warnings"]


def test_ascii_font_and_string():
    line = UI + "UiAsciiFilter:AsciiParams:" + _b("@#+. ") + "$12.0$-1$true$9"
    _, p = _params(_imp(line))
    assert p["gradient"] == "@#+. " and p["font_size"] == 12.0 and p["background_color"] == "#FFFFFF"
    assert p["gray_scale"] is True and p["font"] == FONTS[9] == "DejaVu"


def test_every_itb_filter_round_trips_at_defaults():
    bad = []
    for name, e in FILTERS.items():
        if e.get("unsupported") or name in ("PaletteTransfer", "PaletteTransferVariant"):
            continue
        spec = get_op(e["op"])
        params = {**spec.defaults(), **e.get("fixed", {})}
        out = T.export_template("t", [{"op": e["op"], "params": params}])
        back = T.import_template(out["template"])
        got = spec.params_json(spec.validate(back["steps"][0]["params"]))
        if got != spec.params_json(spec.validate(params)):
            bad.append(name)
    assert not bad
    assert len(FILTERS) >= 540


def _rand_value(rng, p, leaf):
    kind = leaf[0]
    extra = leaf[2] if len(leaf) > 2 else None
    if isinstance(extra, dict) and "index" in extra:
        return rng.choice(extra["index"])
    if (kind.startswith("E:") or kind == "G:sel") and isinstance(extra, dict):
        return rng.choice(list(extra.values()))
    if p.kind == "color":
        a = 255 if kind == "CF" else rng.choice([255, 128, 64])
        return "#" + "".join(f"{rng.randrange(256):02X}" for _ in range(3)) + ("" if a == 255 else f"{a:02X}")
    if p.kind == "bool":
        return rng.random() < 0.5
    if p.kind == "int":
        lo, hi = int(p.min if p.min is not None else 0), int(p.max if p.max is not None else 100)
        return rng.randint(lo, hi)
    if p.kind == "float":
        lo, hi = (p.min if p.min is not None else -10), (p.max if p.max is not None else 10)
        return min(max(round(rng.uniform(lo, hi), 3), lo), hi)
    if p.kind == "point":
        lo, hi = (p.min if p.min is not None else 0), (p.max if p.max is not None else 1)
        return [round(rng.uniform(lo, hi), 3), round(rng.uniform(lo, hi), 3)]
    if p.kind == "float_list":
        n = p.length or len(p.default)
        return [round(rng.uniform(-2, 2), 3) for _ in range(n)]
    if p.kind == "points":
        xs = sorted(rng.sample(range(1, 99), 2))
        return [[0.0, round(rng.random(), 3)]] + [[x / 100, round(rng.random(), 3)] for x in xs] + [[1.0, 0.5]]
    if p.kind == "gradient":
        n = rng.randint(2, 6)
        return [[i / (n - 1), "#" + "".join(f"{rng.randrange(256):02X}" for _ in range(3))] for i in range(n)]
    if p.kind == "text":
        return rng.choice(["Comfortaa", "Jura", "sans-serif"]) if kind == "FONT" else "".join(
            rng.choice("abc@#. ü") for _ in range(5))
    return None


def test_every_itb_filter_round_trips_random_params():
    rng = random.Random(1234)
    bad = []
    for name, e in FILTERS.items():
        if e.get("unsupported") or any(lf[0] == "FILE" for lf in e.get("m") or []):
            continue
        spec = get_op(e["op"])
        byname = {p.name: p for p in spec.params}
        for _ in range(3):
            params = {**spec.defaults(), **e.get("fixed", {})}
            for leaf in e.get("m") or []:
                v = _rand_value(rng, byname[leaf[1]], leaf)
                if v is not None:
                    params[leaf[1]] = v
            if e["t"] == "ToneCurvesParams":
                for c in ("red", "green", "blue"):
                    params[c] = _rand_value(rng, byname[c], ["TC", c])
            want = spec.params_json(spec.validate(params))
            back = T.import_template(T.export_template("r", [{"op": e["op"], "params": want}])["template"])
            got = spec.params_json(spec.validate(back["steps"][0]["params"]))
            if got != want:
                bad.append((name, {k: (want[k], got[k]) for k in want if want[k] != got[k]}))
    assert not bad, bad[:5]


def test_export_rejects_non_itb_ops_with_a_clear_list():
    from lazyroom.core.errors import LazyRoomError

    steps = [{"op": "filter.brightness", "params": {"bias": 0.1}}, {"op": "adjust.exposure", "params": {"ev": 1}},
             {"op": "filter.displacement", "params": {}}]
    with pytest.raises(LazyRoomError) as e:
        T.export_template("x", steps)
    ops = {u["op"] for u in e.value.details["unsupported"]}
    assert ops == {"adjust.exposure", "filter.displacement"}
    out = T.export_template("x", steps, skip_unsupported=True)
    assert out["filters"] == 1 and len(out["warnings"]) == 2


def test_export_flattens_groups_and_reports_dropped_settings():
    steps = [{"op": "group", "opacity": 0.5, "steps": [{"op": "filter.sepia", "params": {}},
                                                       {"op": "filter.negative", "params": {}, "enabled": False}]},
             {"op": "filter.brightness", "params": {"bias": 0.1}, "mask": {"type": "linear"}}]
    out = T.export_template("a+b", steps)
    assert out["filters"] == 2 and out["name"] == "a_b"
    assert len(out["warnings"]) == 4


def test_template_tools_qr_round_trip(api, tmp_path):
    steps = [{"op": "filter.brightness", "params": {"bias": 0.25}},
             {"op": "filter.gaussian_blur", "params": {"radius": 12, "sigma": 4, "edge_mode": "wrap"}},
             {"op": "filter.vignette", "params": {"start": 0.2, "color": "#102030"}},
             {"op": "filter.warhol", "params": {"x_tiles": 3}}]
    qr, txt = tmp_path / "t.png", tmp_path / "t.imtbx_template"
    r = api.tool_run("template_export", {"steps": steps, "name": "My look", "qr_output": str(qr),
                                         "text_output": str(txt)})
    assert r["qr"]["verified"] and r["filters"] == 4 and txt.read_text() == r["template"]
    import zxingcpp

    decoded = [c.text for c in zxingcpp.read_barcodes(np.asarray(Image.open(qr).convert("RGB")))]
    assert decoded == [r["template"]]
    imp = api.tool_run("template_import", {"input": str(qr), "name": "From QR"})
    assert imp["imported"] == 4 and imp["source"] == "qr" and imp["template_name"] == "My look"
    pr = api.preset_get(imp["preset"]["id"])
    assert [s["op"] for s in pr["steps"]] == [s["op"] for s in steps]
    assert pr["steps"][1]["params"]["edge_mode"] == "wrap"
    again = api.tool_run("template_export", {"preset": imp["preset"]["id"], "name": "My look"})
    assert again["template"] == r["template"]
    from_file = api.tool_run("template_import", {"input": str(txt), "save_preset": False})
    assert from_file["source"] == "file" and "preset" not in from_file


def test_template_import_applies_to_project(api, tmp_path):
    src = tmp_path / "p.png"
    save_image(synthetic_image(64, 48), src)
    pid = api.project_create(str(src))["project_id"]
    r = api.tool_run("template_import", {"text": HDR + "T+" + UI + "UiSepiaFilter:Unit:Unit,"
                                                  + UI + "UiBrightnessFilter:Float:0.3",
                                         "save_preset": False, "project_id": pid})
    assert r["project"]["steps"] == 1
    st = api.project_state(pid)
    group = st["steps"][-1]
    assert group["op"] == "group" and [s["op"] for s in group["steps"]] == ["filter.sepia", "filter.brightness"]


def test_hidden_text_filename_placeholder(api, tmp_path):
    src = tmp_path / "holiday.photo.png"
    save_image(synthetic_image(200, 150), src)
    out = tmp_path / "m.png"
    api.tool_run("watermark_hidden_embed", {"input": str(src), "output": str(out), "text": "(c) {filename}"})
    assert api.tool_run("watermark_hidden_extract", {"input": str(out)})["text"] == "(c) holiday.photo"


# ===========================================================================
# Formats: JPEG XL (core/io + JXL tools), PNG Lossy, multi-size ICO, Telegram sticker, OxiPNG / Jpegli
# ===========================================================================

needs_jxl = pytest.mark.skipif(not pmio.HAVE_JXL, reason="imagecodecs (JPEG XL) not installed")


def _has(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


def _u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def _jpeg_bytes(img, quality=88):
    buf = _io.BytesIO()
    Image.fromarray(_u8(img[..., :3])).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


# --------------------------------------------------------------------------- JPEG XL in core/io
@needs_jxl
def test_jxl_listed_in_formats():
    fm = pmio.supported_formats()
    assert ".jxl" in fm["write"] and ".jxl" in fm["read"] and ".jxl" in fm["write_alpha"]
    assert ".jxl" in fm["write_16bit"] and fm["jxl_available"] is True
    assert pmio.format_for_path("x.jxl") == (".jxl", "JXL")


@needs_jxl
def test_jxl_lossless_roundtrip_rgba_and_16bit(tmp_path):
    img = synthetic_image(64, 48, alpha=True)
    r = pmio.save_image(img, tmp_path / "a.jxl", lossless=True)
    back, meta = pmio.load_image(r["path"])
    assert meta.format == "JXL" and meta.has_alpha and back.shape == (48, 64, 4)
    assert np.array_equal(_u8(back), _u8(img))
    rgb = synthetic_image(40, 30)
    r = pmio.save_image(rgb, tmp_path / "b16.jxl", bit_depth=16, lossless=True)
    back, meta = pmio.load_image(r["path"])
    assert meta.bit_depth == 16 and back.shape == (30, 40, 3)
    assert np.abs(back - rgb).max() < 1 / 65535 + 1e-6


@needs_jxl
def test_jxl_lossy_quality_and_effort(tmp_path):
    img = synthetic_image(96, 64)
    small = pmio.encode_image(img, ".jxl", quality=30)
    big = pmio.encode_image(img, ".jxl", quality=95)
    assert len(small) < big.__len__() < len(pmio.encode_image(img, ".jxl", lossless=True))
    assert pmio.encode_image(img, ".jxl", quality=100) == pmio.encode_image(img, ".jxl", lossless=True)
    fast = pmio.encode_image(img, ".jxl", quality=80, extra={"effort": 1})
    assert fast[:2] == b"\xff\x0a"
    p = tmp_path / "q.jxl"
    p.write_bytes(big)
    back, _ = pmio.load_image(p)
    assert np.abs(back - img).mean() < 0.02


@needs_jxl
def test_jxl_exif_xmp_container_and_strip(tmp_path):
    ex = Image.Exif()
    ex[0x010F] = "Cam"
    ex[0x8825] = {2: (1.0, 2.0, 3.0)}
    meta = pmio.ImageMeta(exif=ex.tobytes(), xmp=b"<x:xmpmeta/>")
    img = synthetic_image(32, 24)
    keep = pmio.encode_image(img, ".jxl", meta=meta, metadata="keep_no_gps")
    assert keep[:12] == b"\x00\x00\x00\x0cJXL \r\n\x87\n"
    types = [t for t, _ in pmio.jxl_boxes(keep)]
    assert types[:2] == [b"JXL ", b"ftyp"] and b"Exif" in types and b"xml " in types and types[-1] == b"jxlc"
    p = tmp_path / "m.jxl"
    p.write_bytes(keep)
    back, m = pmio.load_image(p)
    e = Image.Exif()
    e.load(m.exif)
    assert e.get(0x010F) == "Cam" and 0x8825 not in e and m.xmp == b"<x:xmpmeta/>"
    assert np.abs(back - img).mean() < 0.03
    strip = pmio.encode_image(img, ".jxl", meta=meta, metadata="strip")
    assert strip[:2] == b"\xff\x0a"


@needs_jxl
def test_jxl_header_parser_sizes_and_limits(tmp_path):
    import imagecodecs

    for w, h in [(64, 48), (100, 37), (7, 1000), (4096, 8), (1, 1), (513, 512), (160, 90)]:
        cs = bytes(imagecodecs.jpegxl_encode(np.zeros((h, w, 3), np.uint8), lossless=True))
        info = pmio.jxl_header(cs)
        assert (info["width"], info["height"], info["animated"]) == (w, h, False)
    p = tmp_path / "big.jxl"
    p.write_bytes(bytes(imagecodecs.jpegxl_encode(np.zeros((200, 300, 3), np.uint8), lossless=True)))
    with pytest.raises(LazyRoomError) as e:
        pmio.load_image(p, max_pixels=1000)
    assert e.value.code == ErrorCode.RESOURCE_LIMIT
    assert pmio.probe(p).width == 300
    with pytest.raises(ValueError):
        pmio.jxl_header(b"\x89PNG\r\n\x1a\n" + bytes(20))


@needs_jxl
def test_jxl_detected_by_signature_without_extension(tmp_path):
    p = tmp_path / "noext.img"
    p.write_bytes(pmio.encode_image(synthetic_image(20, 10), ".jxl", lossless=True))
    img, meta = pmio.load_image(p)
    assert meta.format == "JXL" and img.shape == (10, 20, 3)


def test_jxl_unavailable_means_backend_error(tmp_path, monkeypatch):
    monkeypatch.setattr(pmio, "HAVE_JXL", False)
    with pytest.raises(LazyRoomError) as e:
        pmio.format_for_path("out.jxl")
    assert e.value.code == ErrorCode.BACKEND_UNAVAILABLE and "lazyroom[jxl]" in e.value.hint
    p = tmp_path / "x.jxl"
    p.write_bytes(b"\xff\x0a" + bytes(30))
    with pytest.raises(LazyRoomError) as e:
        pmio.load_image(p)
    assert e.value.code == ErrorCode.BACKEND_UNAVAILABLE


def test_jxl_absent_from_format_lists_without_imagecodecs():
    code = ("import sys; sys.modules['imagecodecs'] = None\n"
            "from lazyroom.core import io\n"
            "f = io.supported_formats()\n"
            "print('.jxl' in f['write'], '.jxl' in f['read'], f['jxl_available'])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False", "False"]


@needs_jxl
@pytest.mark.skipif(not shutil.which("magick"), reason="ImageMagick not installed")
def test_jxl_reads_files_written_by_other_encoders(tmp_path):
    if b"JXL" not in subprocess.run(["magick", "-list", "format"], capture_output=True).stdout:
        pytest.skip("ImageMagick built without JPEG XL")
    src = tmp_path / "s.png"
    img = synthetic_image(80, 60, alpha=True)
    pmio.save_image(img, src)
    subprocess.run(["magick", str(src), "-quality", "100", str(tmp_path / "ll.jxl")], check=True)
    back, meta = pmio.load_image(tmp_path / "ll.jxl")
    assert np.array_equal(_u8(back), _u8(img))
    flip = tmp_path / "f.png"
    pmio.save_image(img[::-1].copy(), flip)
    subprocess.run(["magick", "-delay", "50", str(src), str(flip), "-loop", "0", str(tmp_path / "an.jxl")], check=True)
    info = pmio.jxl_header((tmp_path / "an.jxl").read_bytes())
    assert info["animated"]
    _, meta = pmio.load_image(tmp_path / "an.jxl")
    assert meta.animated and meta.frames is None  # frames are not counted (would decode all)


# --------------------------------------------------------------------------- JXL tools
@needs_jxl
def test_jxl_transcode_roundtrip_bit_exact(api, tmp_path):
    jpg = tmp_path / "in" / "photo.jpg"
    jpg.parent.mkdir()
    jpg.write_bytes(_jpeg_bytes(synthetic_image(320, 240)))
    r = api.tool_run("jxl_transcode", {"inputs": [str(jpg.parent)], "output_dir": str(tmp_path / "jxl")})
    res = r["results"][0]
    assert r["succeeded"] == 1 and res["verified"] and res["output_bytes"] < res["input_bytes"]
    jxl = tmp_path / "jxl" / "photo.jxl"
    pix, meta = pmio.load_image(jxl)
    assert meta.format == "JXL" and pix.shape == (240, 320, 3)
    r = api.tool_run("jxl_transcode", {"inputs": [str(jxl)], "output_dir": str(tmp_path / "back"),
                                       "mode": "jxl_to_jpeg"})
    assert r["succeeded"] == 1
    assert (tmp_path / "back" / "photo.jpg").read_bytes() == jpg.read_bytes()


@needs_jxl
def test_jxl_transcode_errors_are_reported(api, tmp_path):
    plain = tmp_path / "plain.jxl"
    plain.write_bytes(pmio.encode_image(synthetic_image(32, 24), ".jxl", quality=80))
    r = api.tool_run("jxl_transcode", {"inputs": [str(plain)], "output_dir": str(tmp_path / "o"),
                                       "mode": "jxl_to_jpeg"})
    assert r["failed"] == 1 and r["results"][0]["error"]["code"] == "DECODE_FAILED"
    fake = tmp_path / "fake.jpg"
    fake.write_bytes(b"not a jpeg at all")
    r = api.tool_run("jxl_transcode", {"inputs": [str(fake)], "output_dir": str(tmp_path / "o")})
    assert r["failed"] == 1 and r["results"][0]["error"]["code"] == "UNSUPPORTED_FORMAT"
    assert not (tmp_path / "o" / "fake.jxl").exists()


@needs_jxl
def test_set_jxl_tick_and_plan():
    import imagecodecs

    frames = np.stack([_u8(synthetic_image(32, 24, seed=s)) for s in range(3)])
    data = bytes(imagecodecs.jpegxl_encode(frames, level=90))
    h = pmio.jxl_header(data)
    assert (h["tps_numerator"], h["tps_denominator"], h["num_loops"]) == (10, 1, 0)
    for tick in (1, 7, 250, 1000, 1024):
        out = itbc.set_jxl_tick(data, tick)
        h2 = pmio.jxl_header(out)
        assert (h2["tps_numerator"], h2["tps_denominator"], h2["num_loops"]) == (1000, tick, 0)
        assert len(out) == len(data)
        dec = imagecodecs.jpegxl_decode(out)
        assert dec.shape[0] == 3
    with pytest.raises(ValueError):
        itbc.set_jxl_tick(data, 2000)
    assert itbc.jxl_tick_plan([1000, 1000]) == (1000, [1, 1])
    assert itbc.jxl_tick_plan([2000, 4000]) == (1000, [2, 4])
    assert itbc.jxl_tick_plan([100, 200, 100]) == (100, [1, 2, 1])
    assert itbc.jxl_tick_plan([1500]) == (750, [2])


def _ffprobe_durations(path):
    if not shutil.which("ffprobe"):
        return None
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "frame=duration_time", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True)
    vals = [float(v) for v in out.stdout.split() if v.strip()]
    return vals or None


@needs_jxl
def test_jxl_from_still_frames_and_extract(api, tmp_path):
    paths = []
    for i, (w, h) in enumerate([(80, 60), (60, 80), (80, 60)]):
        p = tmp_path / f"f{i}.png"
        pmio.save_image(synthetic_image(w, h, seed=i), p)
        paths.append(str(p))
    out = tmp_path / "anim.jxl"
    r = api.tool_run("jxl_from_frames", {"inputs": paths, "output": str(out), "frame_ms": 250, "lossless": True})
    assert r["frames"] == 3 and r["tick_ms"] == 250 and r["stored_frames"] == 3
    assert (r["width"], r["height"]) == (80, 60)
    hdr = pmio.jxl_header(out.read_bytes())
    assert hdr["animated"] and hdr["tps_numerator"] == 1000 and hdr["tps_denominator"] == 250
    d = _ffprobe_durations(out)
    if d is not None:
        assert d == pytest.approx([0.25, 0.25, 0.25])
    ex = api.tool_run("jxl_extract_frames", {"input": str(out), "output_dir": str(tmp_path / "frames")})
    assert ex["count"] == 3 and ex["animated"] and ex["tick_ms"] == 250.0
    f0, _ = pmio.load_image(ex["frames"][0])
    assert np.array_equal(_u8(f0), _u8(synthetic_image(80, 60, seed=0)))
    # the portrait frame: longer side scaled to 80 (ResizeType.Flexible) then centre-cropped to 80x60
    f1, _ = pmio.load_image(ex["frames"][1])
    assert f1.shape[:2] == (60, 80) and f1.shape[2] == 4
    assert f1[30, 0, 3] == 0 and f1[30, 40, 3] == 1.0  # transparent side bars, opaque centre


@needs_jxl
def test_jxl_from_animated_gif_keeps_durations(api, tmp_path):
    frames = [Image.fromarray(_u8(synthetic_image(40, 30, seed=s))) for s in range(3)]
    gif = tmp_path / "a.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=[100, 200, 100], loop=0)
    out = tmp_path / "a.jxl"
    r = api.tool_run("jxl_from_frames", {"inputs": [str(gif)], "output": str(out)})
    assert r["animated_source"] and r["durations_ms"] == [100, 200, 100]
    assert r["tick_ms"] == 100 and r["stored_frames"] == 4
    d = _ffprobe_durations(out)
    if d is not None:
        assert sum(d) == pytest.approx(0.4)
    r = api.tool_run("jxl_from_frames", {"inputs": [str(gif)], "output": str(tmp_path / "b.jxl"),
                                         "override_durations": True, "frame_ms": 2000})
    assert r["tick_ms"] == 1000 and r["stored_frames"] == 6
    one = tmp_path / "one.png"
    pmio.save_image(synthetic_image(20, 10), one)
    r = api.tool_run("jxl_from_frames", {"inputs": [str(one)], "output": str(tmp_path / "one.jxl")})
    assert r["stored_frames"] == 1 and r["tick_ms"] is None
    with pytest.raises(LazyRoomError):
        api.tool_run("jxl_from_frames", {"inputs": [str(one)], "output": str(tmp_path / "one.png")})


# --------------------------------------------------------------------------- PNG lossy (aire Wu quantiser)
def test_wu_palette_exact_for_few_colours_and_alpha_fader():
    a = np.zeros((30, 30, 4), np.uint8)
    a[..., 3] = 255
    a[:10] = (200, 10, 10, 255)
    a[10:20] = (10, 200, 10, 255)
    a[20:] = (10, 10, 200, 255)
    pal = itbc.wu_palette(a, 8)
    assert {tuple(p) for p in pal} == {(200, 10, 10, 255), (10, 200, 10, 255), (10, 10, 200, 255)}
    a[0, :5] = (0, 0, 0, 0)
    a[1, :5] = (50, 60, 70, 128)
    pal = itbc.wu_palette(a, 8)
    assert tuple(pal[0]) == (0, 0, 0, 0)          # transparent entry first (m_transparentPixelIndex)
    assert (50, 60, 70, 136) in {tuple(p) for p in pal}  # alpha fader: 128 + 128 % 15
    q = itbc.remap_nearest(a, pal)
    assert tuple(q[0, 0]) == (0, 0, 0, 0) and tuple(q[25, 25]) == (10, 10, 200, 255)


@pytest.mark.parametrize("n", [2, 16, 64, 256, 512])
def test_png_lossy_colour_budget(n):
    img = synthetic_image(160, 120, alpha=True)
    q = itbc.png_lossy_pixels(img, n)
    pal = itbc.wu_palette(_u8(img), n)
    assert len(pal) <= n
    used = {tuple(c) for c in q.reshape(-1, 4)}
    assert used <= {tuple(p) for p in pal}
    if n >= 64:
        m = img[..., 3] > 0.5
        assert np.abs(q[..., :3] / 255.0 - img[..., :3])[m].mean() < 0.05


def test_png_lossy_encoding_palette_and_truecolour():
    img = synthetic_image(160, 120)
    data = itbc.png_lossy_encode(img, 64, 9)
    im = Image.open(_io.BytesIO(data))
    assert im.mode == "P" and len(im.getcolors(1024)) <= 64 and "transparency" not in im.info
    assert np.array_equal(np.asarray(im.convert("RGBA")), itbc.png_lossy_pixels(img, 64))
    rgba = synthetic_image(160, 120, alpha=True)
    im = Image.open(_io.BytesIO(itbc.png_lossy_encode(rgba, 32)))
    assert im.mode == "P" and "transparency" in im.info
    rich = np.random.default_rng(1).random((64, 64, 3)).astype(np.float32)
    im = Image.open(_io.BytesIO(itbc.png_lossy_encode(rich, 1024)))
    assert im.mode == "RGB" and 256 < len(im.getcolors(70000)) <= 1024
    with pytest.raises(LazyRoomError):
        itbc.png_lossy_encode(img, 1)
    with pytest.raises(LazyRoomError):
        itbc.png_lossy_encode(img, 64, 10)


def test_format_convert_png_lossy(api, tmp_path):
    src = tmp_path / "in.png"
    pmio.save_image(synthetic_image(200, 150), src)
    r = api.tool_run("format_convert", {"inputs": [str(src)], "output_dir": str(tmp_path / "o"),
                                        "format": "png_lossy", "max_colors": 48})
    assert r["succeeded"] == 1
    out = tmp_path / "o" / "in.png"
    with Image.open(out) as im:
        assert len(im.convert("RGB").getcolors(100000)) <= 48
    assert out.stat().st_size < src.stat().st_size
    with pytest.raises(LazyRoomError):
        api.tool_run("format_convert", {"inputs": [str(src)], "output_dir": str(tmp_path / "o2"),
                                        "format": "png_lossy", "effort": 10})


# --------------------------------------------------------------------------- ICO
def test_ico_multi_layout_matches_imagetoolbox():
    img = synthetic_image(300, 150, alpha=True)
    data = itbc.ico_encode(img)
    reserved, typ, count = struct.unpack_from("<HHH", data, 0)
    assert (reserved, typ, count) == (0, 1, 6)
    offset = 6 + 16 * 6
    for i, size in enumerate(itbc.ICO_SIZES):
        w, h, colors, res, planes, bpp, nbytes, off = struct.unpack_from("<BBBBHHII", data, 6 + 16 * i)
        assert (w, h) == ((0, 0) if size == 256 else (size, size))
        assert (colors, res, planes, bpp, off) == (0, 0, 1, 32, offset)
        hdr = struct.unpack_from("<IiiHHII", data, off)
        assert hdr == (40, size, 2 * size, 1, 32, 0, size * size * 4 + ((size + 31) // 32) * 4 * size)
        assert nbytes == 40 + size * size * 4 + ((size + 31) // 32) * 4 * size
        offset += nbytes
    assert offset == len(data)
    im = Image.open(_io.BytesIO(data))
    assert im.info["sizes"] == {(s, s) for s in itbc.ICO_SIZES}
    im.size = (64, 64)
    px = np.asarray(im.convert("RGBA"))
    assert px[0, 32, 3] == 0 and px[63, 32, 3] == 0      # 2:1 image centred: transparent top/bottom bands
    assert px[32, 32, 3] > 200
    # BMP rows are stored bottom-up in BGRA order
    off16 = struct.unpack_from("<I", data, 6 + 12)[0]
    ref = np.asarray(itbc._fit_on_square(Image.fromarray(_u8(img), "RGBA"), 16))
    first_row = np.frombuffer(data, np.uint8, 16 * 4, off16 + 40).reshape(16, 4)
    assert np.array_equal(first_row, ref[15][:, [2, 1, 0, 3]])


def test_format_convert_ico_multi_and_plain_ico_unchanged(api, tmp_path):
    src = tmp_path / "logo.png"
    pmio.save_image(synthetic_image(64, 64, alpha=True), src)
    r = api.tool_run("format_convert", {"inputs": [str(src)], "output_dir": str(tmp_path / "m"),
                                        "format": "ico_multi"})
    assert r["succeeded"] == 1
    with Image.open(tmp_path / "m" / "logo.ico") as im:
        assert im.info["sizes"] == {(s, s) for s in (16, 32, 48, 64, 128, 256)}
    r = api.tool_run("format_convert", {"inputs": [str(src)], "output_dir": str(tmp_path / "p"), "format": "ico"})
    expected = pmio.encode_image(pmio.load_image(src)[0], ".ico", quality=90, meta=pmio.load_image(src)[1])
    assert (tmp_path / "p" / "logo.ico").read_bytes() == expected


# --------------------------------------------------------------------------- Telegram sticker
def test_telegram_sticker_sizes():
    assert itbc.telegram_sticker_size(1920, 1080) == (512, 288)
    assert itbc.telegram_sticker_size(1080, 1920) == (288, 512)
    assert itbc.telegram_sticker_size(300, 300) == (512, 512)      # upscales like ImageToolbox
    assert itbc.telegram_sticker_size(100, 50) == (512, 256)
    assert itbc.telegram_sticker_size(1000, 3) == (512, 1)
    assert itbc.telegram_sticker_size(333, 1000) == (170, 512)     # Kotlin Float -> Int truncation


def test_telegram_sticker_tool(api, tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.png"
    pmio.save_image(synthetic_image(400, 300), a)
    pmio.save_image(synthetic_image(120, 240, alpha=True), b)
    r = api.tool_run("telegram_sticker", {"inputs": [str(a), str(b)], "output_dir": str(tmp_path / "tg")})
    assert r["succeeded"] == 2
    sizes = {x["input"]: (x["width"], x["height"]) for x in r["results"]}
    assert sizes == {str(a): (512, 384), str(b): (256, 512)}
    for x in r["results"]:
        with Image.open(x["output"]) as im:
            assert im.format == "PNG" and max(im.size) == 512
        assert x["within_512kb"] is (x["bytes"] <= 512 * 1024)
    with Image.open(tmp_path / "tg" / "b.png") as im:
        assert im.mode == "RGBA"


# --------------------------------------------------------------------------- format_convert extras
@needs_jxl
def test_format_convert_jxl_lossless_effort(api, tmp_path):
    src = tmp_path / "s.png"
    img = synthetic_image(100, 80, alpha=True)
    pmio.save_image(img, src)
    r = api.tool_run("format_convert", {"inputs": [str(src)], "output_dir": str(tmp_path / "o"), "format": "jxl",
                                        "lossless": True, "effort": 3})
    assert r["succeeded"] == 1
    back, _ = pmio.load_image(tmp_path / "o" / "s.jxl")
    assert np.array_equal(_u8(back), _u8(img))
    with pytest.raises(LazyRoomError):
        api.tool_run("format_convert", {"inputs": [str(src)], "output_dir": str(tmp_path / "o"), "format": "jxl",
                                        "effort": 0})


def test_format_convert_existing_defaults_unchanged(api, tmp_path):
    src = tmp_path / "s.png"
    img = synthetic_image(60, 40)
    pmio.save_image(img, src)
    r = api.tool_run("format_convert", {"inputs": [str(src)], "output_dir": str(tmp_path / "o")})
    assert r["succeeded"] == 1
    loaded, meta = pmio.load_image(src)
    assert (tmp_path / "o" / "s.webp").read_bytes() == pmio.encode_image(loaded, ".webp", quality=90, meta=meta,
                                                                          metadata="keep")


@pytest.mark.skipif(not _has("oxipng"), reason="pyoxipng not installed")
def test_format_convert_oxipng_is_lossless(api, tmp_path):
    src = tmp_path / "s.png"
    img = synthetic_image(120, 90, alpha=True)
    pmio.save_image(img, src)
    r = api.tool_run("format_convert", {"inputs": [str(src)], "output_dir": str(tmp_path / "o"),
                                        "format": "oxipng", "effort": 2})
    assert r["succeeded"] == 1
    back, _ = pmio.load_image(tmp_path / "o" / "s.png")
    assert np.array_equal(_u8(back), _u8(img))
    plain = pmio.encode_image(img, ".png", extra={"optimize": False, "compress_level": 1})
    assert (tmp_path / "o" / "s.png").stat().st_size < len(plain)
    with pytest.raises(LazyRoomError):
        api.tool_run("format_convert", {"inputs": [str(src)], "output_dir": str(tmp_path / "o2"),
                                        "format": "oxipng", "effort": 7})


@pytest.mark.skipif(not _has("pyjpegli"), reason="pyjpegli not installed")
def test_format_convert_jpegli_with_metadata(api, tmp_path):
    img = synthetic_image(120, 90)
    ex = Image.Exif()
    ex[0x010F] = "Cam"
    src = tmp_path / "s.jpg"
    Image.fromarray(_u8(img)).save(src, "JPEG", quality=95, exif=ex.tobytes())
    r = api.tool_run("format_convert", {"inputs": [str(src)], "output_dir": str(tmp_path / "o"),
                                        "format": "jpegli", "quality": 80})
    assert r["succeeded"] == 1
    with Image.open(tmp_path / "o" / "s.jpg") as im:
        assert im.format == "JPEG" and im.getexif().get(0x010F) == "Cam" and im.info.get("icc_profile")
        back = np.asarray(im.convert("RGB"), np.float32) / 255
    assert np.abs(back - img).mean() < 0.05


def test_optional_encoders_report_backend_unavailable(api, tmp_path, monkeypatch):
    src = tmp_path / "s.png"
    pmio.save_image(synthetic_image(20, 20), src)
    for fmt, mod in (("oxipng", "oxipng"), ("jpegli", "pyjpegli")):
        monkeypatch.setitem(sys.modules, mod, None)
        with pytest.raises(LazyRoomError) as e:
            api.tool_run("format_convert", {"inputs": [str(src)], "output_dir": str(tmp_path / "o"), "format": fmt})
        assert e.value.code == ErrorCode.BACKEND_UNAVAILABLE and "lazyroom[" in e.value.hint
