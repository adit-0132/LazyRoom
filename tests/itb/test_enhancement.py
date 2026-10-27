"""ImageToolbox "Enhancement" group: ranges, exact G'MIC commands and semantic checks."""
import math

import cv2
import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.registry import all_ops, get_op
from tests.conftest import synthetic_image

SPECS = [s for s in all_ops() if s.group == "enhancement"]


def _run(op_id, img, ctx=None, **values):
    spec = get_op(op_id)
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    return spec.run(img, ctx or RenderContext(seed=3), spec.validate(values))


def _hf(img):
    """High-frequency energy (variance of the Laplacian of the luma)."""
    g = img[..., :3].mean(axis=2).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def test_group_is_complete():
    assert len(SPECS) == 33


@pytest.mark.parametrize("extreme", ["min", "max"])
@pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])
def test_param_extremes(spec, extreme):
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    vals = {p.name: getattr(p, extreme) for p in spec.params
            if p.kind in ("float", "int") and getattr(p, extreme, None) is not None}
    img = synthetic_image(64, 48)  # G'MIC split_details (Details Equalizer) needs >= ~64x48
    out = spec.run(img, RenderContext(seed=1), spec.validate(vals))
    assert out.shape == img.shape and out.dtype == np.float32
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


# ---------------------------------------------------------------------------
# G'MIC: exact ImageToolbox command strings
# ---------------------------------------------------------------------------

DEFAULT_COMMANDS = {
    "alpha_sharpen": "fx_sharpen_alpha 100,5,50,100,0",
    "banding_denoise": "banding_denoise_v2 5,5,5,10,0",
    "clean_text": "afre_cleantext 8,1,80,95",
    "constrained_sharpen": "iain_constrained_sharpen 0.75,2.5,1,5,0,11,1",
    "dcp_dehaze": "jeje_dehaze 5,1,0.2,1,0,0,0,0",
    "deblur": "fx_deblur 2,10,20,0.1,1,0,0,24",
    "denoise_smooth": "afre_denoisesmooth 3,10",
    "descreen": "fx_pahlsson_descreen 7,40,1",
    "details_equalizer": "fx_equalize_details 5,0.5,0,0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,0,32",
    "freaky_details": "fx_freaky_details 2,10,1,7",
    "gold_meinel_sharpen": "fx_unsharp_goldmeinel 1,5,1,1,1,0,0,24",
    "gradient_sharpen": "gcd_sharpen_gradient 0.8,1.4,1",
    "hessian_sharpen": "jeje_hessian_sharpen 3,1,1,0",
    "local_variance_normalization": "jeje_normalize_local_variance 60,6,10,1,0",
    "magic_details": "fx_magic_details 6,3,15,-0.5,2,12",
    "mighty_details": "fx_mighty_details 25,1,25,1,12",
    "multiscale_sharpen": "fx_sharpen_multiscale 15,20,0",
    "remove_hot_pixels": "remove_hotpixels 3,10",
    "remove_scratches": "fx_remove_scratches 72,2,4",
    "richardson_lucy_sharpen": "fx_unsharp_richardsonlucy 1,10,1,1",
    "smooth_skin": "fx_smooth_skin 2,0.5,1,1,1,50,50,5,2,0.2,3,1,0.05",
    "texture_sharpen": "fx_sharpen_texture 1,4,0",
    "tone_sharpen": "gcd_sharpen_tones 1.2,128,1",
    "unstrip": "jeje_unstrip 1,20,4,1,0",
    "whiten_sharpen": "jeje_whiten_frequency 50,0",
}


@pytest.fixture
def captured(monkeypatch):
    from lazyroom.ops.itb import enhancement

    cmds = []

    def fake(img, command, **kw):
        cmds.append(command)
        return img

    monkeypatch.setattr(enhancement, "run_gmic", fake)
    return cmds


@pytest.mark.parametrize("name", sorted(DEFAULT_COMMANDS))
def test_gmic_default_commands(name, captured):
    spec = get_op(f"filter.{name}")
    spec.fn(synthetic_image(16, 12), RenderContext(), **spec.validate({}))
    assert captured == [DEFAULT_COMMANDS[name]]


def test_gmic_enum_and_bool_mapping(captured):
    spec = get_op("filter.constrained_sharpen")
    spec.fn(synthetic_image(16, 12), RenderContext(),
            **spec.validate({"channel": "key", "value_action": "none"}))
    spec = get_op("filter.deblur")
    spec.fn(synthetic_image(16, 12), RenderContext(),
            **spec.validate({"channel": "ycbcr", "regularization": "total_variation", "parallelism": "sixteen_threads"}))
    spec = get_op("filter.unstrip")
    spec.fn(synthetic_image(16, 12), RenderContext(), **spec.validate({"normalize": False, "fft_preview": True}))
    assert captured == ["iain_constrained_sharpen 0.75,2.5,1,5,0,31,0",
                        "fx_deblur 2,10,20,0.1,2,11,5,24",
                        "jeje_unstrip 1,20,4,0,1"]


def test_gmic_pixel_params_follow_preview_scale(captured):
    spec = get_op("filter.richardson_lucy_sharpen")
    spec.fn(synthetic_image(16, 12), RenderContext(scale=0.5), **spec.validate({"sigma": 3}))
    assert captured == ["fx_unsharp_richardsonlucy 1.5,10,1,1"]


@pytest.mark.gmic
def test_details_equalizer_neutral_by_default():
    img = synthetic_image(48, 36)
    out = _run("filter.details_equalizer", img)
    assert np.abs(out - img).max() < 1.5 / 255


@pytest.mark.gmic
def test_details_equalizer_small_image_parallel_tiling():
    # G'MIC's parallel tiling fails on tiny tiles; the op retries untiled
    img = synthetic_image(64, 48)
    out = _run("filter.details_equalizer", img, spatial_overlap=0, parallelism="sixteen_threads", fine_gain=1)
    assert out.shape == img.shape and np.isfinite(out).all()


@pytest.mark.gmic
@pytest.mark.parametrize("op_id", ["filter.richardson_lucy_sharpen", "filter.gold_meinel_sharpen",
                                   "filter.multiscale_sharpen", "filter.constrained_sharpen"])
def test_gmic_sharpeners_add_detail(op_id):
    img = cv2.GaussianBlur(synthetic_image(64, 48), (0, 0), 1.2)
    assert _hf(_run(op_id, img)) > _hf(img) * 1.2


@pytest.mark.gmic
def test_remove_hot_pixels():
    img = np.full((32, 32, 3), 0.3, np.float32)
    img[10, 12] = 1.0
    out = _run("filter.remove_hot_pixels", img)
    assert np.allclose(out[10, 12], 0.3, atol=2 / 255)


# ---------------------------------------------------------------------------
# GPUImage
# ---------------------------------------------------------------------------

def test_sharpen_identity_and_direction():
    img = cv2.GaussianBlur(synthetic_image(64, 48), (0, 0), 1.0)
    assert np.allclose(_run("filter.sharpen", img, sharpness=0), img, atol=1e-6)
    assert _hf(_run("filter.sharpen", img, sharpness=1)) > _hf(img) * 2
    assert _hf(_run("filter.sharpen", img, sharpness=-0.2)) < _hf(img)


def _kuwahara_shader(img, radius, rows, cols):
    """Literal transcription of GPUImageKuwaharaFilter's fragment shader (bilinear, clamp-to-edge)."""
    h, w = img.shape[:2]

    def tex(u, v):
        x, y = u * w - 0.5, v * h - 0.5
        x0, y0 = math.floor(x), math.floor(y)
        fx, fy = x - x0, y - y0

        def p(xx, yy):
            return img[min(max(yy, 0), h - 1), min(max(xx, 0), w - 1)].astype(np.float64)

        return ((1 - fx) * (1 - fy) * p(x0, y0) + fx * (1 - fy) * p(x0 + 1, y0)
                + (1 - fx) * fy * p(x0, y0 + 1) + fx * fy * p(x0 + 1, y0 + 1))

    out = np.zeros((len(rows), len(cols), 3), np.float64)
    n = (radius + 1) ** 2
    quads = [((-radius, 0), (-radius, 0)), ((-radius, 0), (0, radius)), ((0, radius), (0, radius)),
             ((0, radius), (-radius, 0))]
    for r_i, yy in enumerate(rows):
        for k, xx in enumerate(cols):
            uv = ((xx + 0.5) / w, (yy + 0.5) / h)
            best, min_s = None, 1e2
            for (j0, j1), (i0, i1) in quads:
                m, s = np.zeros(3), np.zeros(3)
                for j in range(j0, j1 + 1):
                    for i in range(i0, i1 + 1):
                        c = tex(uv[0] + i / 768.0, uv[1] + j / 1024.0)
                        m += c
                        s += c * c
                m /= n
                sig = np.abs(s / n - m * m).sum()
                if sig < min_s:
                    min_s, best = sig, m
            out[r_i, k] = best
    return out


# sample spacing is width/768 x height/1024 px: cover sub-pixel and multi-pixel spacings + borders
@pytest.mark.parametrize("shape,radius", [((14, 20), 2), ((1100, 900), 2), ((1300, 5), 1)])
def test_kuwahara_matches_shader(shape, radius):
    from lazyroom.ops.itb import _enhancement_impl as impl

    rng = np.random.default_rng(1)
    img = np.ascontiguousarray(rng.random(shape + (3,), dtype=np.float32))
    cols = sorted({0, 1, shape[1] // 2, shape[1] - 2, shape[1] - 1})
    rows = sorted({0, 1, shape[0] // 2, shape[0] - 1})
    ref = _kuwahara_shader(img, radius, rows, cols)
    got = impl.gpu_kuwahara(img, radius)[np.ix_(rows, cols)]
    assert np.abs(got - ref).max() < 1e-5


def test_kuwahara_radius_zero_and_flat():
    img = synthetic_image(40, 30)
    assert np.allclose(_run("filter.kuwahara", img, radius=0), img)
    flat = np.full((30, 40, 3), 0.42, np.float32)
    assert np.allclose(_run("filter.kuwahara", flat), 0.42, atol=1e-5)


# ---------------------------------------------------------------------------
# aire
# ---------------------------------------------------------------------------

def test_unsharp_blends_towards_aire_blur():
    from lazyroom.ops.itb import _enhancement_impl as impl

    img = synthetic_image(48, 36)
    assert np.allclose(_run("filter.unsharp", img, intensity=0), img, atol=1e-6)
    k = impl.aire_gauss_kernel(5, 4.0)
    assert np.allclose(k, k[::-1]) and abs(k.sum() - 1) < 1e-6
    blur = cv2.sepFilter2D(img, -1, k, k, borderType=cv2.BORDER_REPLICATE)
    assert np.allclose(_run("filter.unsharp", img, intensity=1), np.clip(blur, 0, 1), atol=1e-5)
    assert _hf(_run("filter.unsharp", img)) < _hf(img)


def test_anisotropic_diffusion_reference_and_flat():
    from lazyroom.ops.itb import _enhancement_impl as impl

    flat = np.full((20, 24, 3), 100 / 255, np.float32)
    assert np.allclose(_run("filter.anisotropic_diffusion", flat), flat, atol=1e-6)

    def literal(u8, steps, cond, diff):  # aire AnisotropicDiffusion.cpp, segments last-to-first
        buf = u8.copy()
        h, w = buf.shape[:2]
        seg = h // 4
        segs = [(0, seg), (seg, 2 * seg), (2 * seg, 3 * seg), (3 * seg, h)]
        for _ in range(steps):
            for a, b in reversed(segs):
                for y in range(a, b):
                    for x in range(w):
                        nx, ny, px, py = min(x + 1, w - 1), min(y + 1, h - 1), max(x - 1, 0), max(y - 1, 0)
                        for c in range(3):
                            loc = np.float32(buf[y, x, c])
                            tot = np.float32(0)
                            for yy, xx in ((py, px), (py, nx), (ny, px), (ny, nx)):
                                d = np.float32(buf[yy, xx, c]) - loc
                                k = d / np.float32(cond)
                                tot += np.float32(1.0 / (1.0 + k * k)) * d
                            upd = loc + loc * (np.float32(diff) * tot)
                            buf[y, x, c] = min(max(math.ceil(upd), 0), 255)
        return buf

    rng = np.random.default_rng(2)
    u8 = rng.integers(90, 110, (13, 9, 3), dtype=np.uint8)
    got = impl.to_u8(impl.anisotropic_diffusion(u8.astype(np.float32) / 255, 2, 0.6, 0.5))
    assert np.array_equal(got, literal(u8, 2, 0.6, 0.5))


def test_dehaze_dark_channel_matches_aire():
    from lazyroom.ops.itb import _enhancement_impl as impl

    def literal(u8, radius, lanes=16):  # getDarkChannel incl. the red-only SIMD blocks
        h, w = u8.shape[:2]
        out = np.zeros((h, w), np.uint8)
        for i in range(h):
            for j in range(w):
                mv = 255
                for y in range(-radius, radius + 1):
                    row = u8[min(max(y + i, 0), h - 1)]
                    x = -radius
                    while x + lanes <= radius and x + j + lanes < w:
                        pos = min(max(x + j, 0), w - 1)
                        mv = min(mv, int(row[pos:pos + lanes, 0].min()))
                        x += lanes
                    while x <= radius:
                        if x + j >= w:
                            break
                        mv = min(mv, int(row[max(x + j, 0)].min()))
                        x += 1
                out[i, j] = mv
        return out

    rng = np.random.default_rng(4)
    u8 = rng.integers(0, 256, (14, 45, 3), dtype=np.uint8)
    rmin = cv2.erode(np.ascontiguousarray(u8[..., 0]), np.ones((1, 16), np.uint8), anchor=(0, 0),
                     borderType=cv2.BORDER_REPLICATE)
    rows = impl._dark_rows(u8, rmin, 9, 16)
    dark = cv2.erode(rows, np.ones((19, 1), np.uint8), borderType=cv2.BORDER_REPLICATE)
    assert np.array_equal(dark, literal(u8, 9))


def test_dehaze_increases_contrast():
    img = synthetic_image(80, 60)
    hazy = (img * 0.5 + 0.45).astype(np.float32)
    out = _run("filter.dehaze", hazy, radius=5)
    assert out.std() > hazy.std() * 1.2
    assert out.mean() < hazy.mean()


# ---------------------------------------------------------------------------
# Trickle / jhlabs
# ---------------------------------------------------------------------------

def test_auto_white_balance_stretches_channels():
    rng = np.random.default_rng(5)
    img = rng.random((40, 50, 3)).astype(np.float32)
    img[..., 0] = 0.2 + 0.4 * img[..., 0]
    img[..., 2] = 0.1 + 0.3 * img[..., 2]
    out = _run("filter.auto_white_balance", img)
    for c in range(3):
        assert out[..., c].min() < 0.02 and out[..., c].max() > 0.98
    assert np.allclose(_run("filter.auto_white_balance", img, strength=0), img, atol=1e-6)


def test_auto_white_balance_ignores_transparent_pixels():
    img = np.dstack([synthetic_image(30, 20) * 0.5 + 0.25, np.ones((20, 30), np.float32)])
    img[:5, :, 3] = 0
    img[:5, :, :3] = 0.9
    out = _run("filter.auto_white_balance", img)
    assert np.allclose(out[:5], img[:5])
    assert np.allclose(out[..., 3], img[..., 3])


def test_high_pass():
    flat = np.full((24, 32, 3), 0.7, np.float32)
    assert np.allclose(_run("filter.high_pass", flat), 0.5, atol=1e-5)
    img = synthetic_image(48, 36)
    out = _run("filter.high_pass", img)
    assert abs(float(out.mean()) - 0.5) < 0.05 and out.std() < img.std()
    rgba = synthetic_image(48, 36, alpha=True)
    out = _run("filter.high_pass", rgba)
    assert out.shape == rgba.shape and np.allclose(out[..., 3], rgba[..., 3])


# ---------------------------------------------------------------------------
# OpenCV red-eye removal
# ---------------------------------------------------------------------------

def test_red_eye_model_loads_and_finds_no_face_in_synthetic_image():
    from lazyroom.ops.itb import _enhancement_impl as impl

    bgr = (synthetic_image(96, 72) * 255).astype(np.uint8)
    assert len(impl.detect_faces(bgr)) == 0


def test_red_eye_correction(monkeypatch):
    from lazyroom.ops.itb import _enhancement_impl as impl

    img = np.full((120, 160, 3), (0.35, 0.55, 0.45), np.float32)
    eyes = [(60.0, 50.0), (100.0, 52.0)]
    for x, y in eyes:
        cv2.circle(img, (int(x), int(y)), 5, (0.9, 0.15, 0.1), -1)
    face = np.zeros((1, 15), np.float32)
    face[0, :4] = (40, 20, 90, 100)
    face[0, 4:8] = (*eyes[0], *eyes[1])
    monkeypatch.setattr(impl, "detect_faces", lambda bgr: face)
    out = _run("filter.auto_remove_red_eyes", img)
    for x, y in eyes:
        px = out[int(y), int(x)]
        assert abs(px[0] - px[1]) < 1e-6 and abs(px[1] - px[2]) < 1e-6  # grey (B + G) / 2
        assert abs(px[0] - round((0.15 + 0.1) * 255 / 2) / 255) < 1.5 / 255
    # pixels outside the 35 px eye boxes are untouched
    assert np.allclose(out[:20], img[:20]) and np.allclose(out[90:], img[90:])
    # a high threshold leaves everything unchanged
    assert np.allclose(_run("filter.auto_remove_red_eyes", img, threshold=250), img)
