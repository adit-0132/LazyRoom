"""ImageToolbox "Glitch" group: parameter extremes + semantic checks."""
import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.errors import LazyRoomError
from lazyroom.core.registry import all_ops, get_op
from lazyroom.ops.itb import _glitch_impl as G
from tests.conftest import synthetic_image

SPECS = [s for s in all_ops(include_hidden=True) if s.group == "glitch"]
M64 = (1 << 64) - 1


def _spec(op_id):
    spec = get_op(op_id)
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    return spec


def _premul(img):
    """Colour as stored by Android (premultiplied): fully transparent pixels have no colour."""
    return np.dstack([img[..., :3] * img[..., 3:4], img[..., 3:4]]) if img.shape[2] == 4 else img


def _run(op_id, img, render_seed=5, **values):
    spec = _spec(op_id)
    return spec.run(img, RenderContext(seed=render_seed), spec.validate(values))


def test_group_has_all_filters():
    assert len(SPECS) == 31


@pytest.mark.parametrize("alpha", [False, True], ids=["rgb", "rgba"])
@pytest.mark.parametrize("extreme", ["min", "max"])
@pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])
def test_numeric_extremes(spec, extreme, alpha):
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    values = {p.name: getattr(p, extreme) for p in spec.params if p.kind in ("int", "float")}
    img = synthetic_image(48, 36, alpha=alpha)
    out = spec.run(img, RenderContext(seed=1), spec.validate(values))
    assert out.dtype == np.float32 and out.ndim == 3 and out.shape[2] in (3, 4)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    if not spec.geometry:
        assert out.shape[:2] == img.shape[:2]


# ---------------------------------------------------------------------------
# VHS NTSC (ntsc-rs port): RNG / hash / noise / filter primitives
# ---------------------------------------------------------------------------

def _siphash24_ref(k0, k1, data: bytes) -> int:
    """Straightforward SipHash-2-4 reference (used to check the numba version)."""
    def rotl(x, b):
        return ((x << b) | (x >> (64 - b))) & M64

    v0, v1 = k0 ^ 0x736F6D6570736575, k1 ^ 0x646F72616E646F6D
    v2, v3 = k0 ^ 0x6C7967656E657261, k1 ^ 0x7465646279746573

    def rounds(n):
        nonlocal v0, v1, v2, v3
        for _ in range(n):
            v0 = (v0 + v1) & M64; v1 = rotl(v1, 13) ^ v0; v0 = rotl(v0, 32)  # noqa: E702
            v2 = (v2 + v3) & M64; v3 = rotl(v3, 16) ^ v2  # noqa: E702
            v0 = (v0 + v3) & M64; v3 = rotl(v3, 21) ^ v0  # noqa: E702
            v2 = (v2 + v1) & M64; v1 = rotl(v1, 17) ^ v2; v2 = rotl(v2, 32)  # noqa: E702

    n = len(data)
    for i in range(0, n - n % 8, 8):
        m = int.from_bytes(data[i:i + 8], "little")
        v3 ^= m
        rounds(2)
        v0 ^= m
    b = ((n & 0xFF) << 56) | int.from_bytes(data[n - n % 8:], "little")
    v3 ^= b
    rounds(2)
    v0 ^= b
    v2 ^= 0xFF
    rounds(4)
    return v0 ^ v1 ^ v2 ^ v3


def test_siphash_reference_vector_and_numba_port():
    # Official SipHash-2-4 test vector (key 00..0f, message 00..0e)
    assert _siphash24_ref(0x0706050403020100, 0x0F0E0D0C0B0A0908, bytes(range(15))) == 0xA129CA6149BE45E5
    rng = np.random.default_rng(0)
    for n in (0, 4, 8, 12, 16, 20, 28, 32):
        msg = rng.integers(0, 256, n).astype(np.uint8)
        assert G.seeder_finalize(msg) == _siphash24_ref(0, 0, bytes(msg))
    # Seeder bytes: u64 seed + u64 index as appended by _siphash's "extra" argument
    pre = G.seeder((7, 8), (3, 8), (0, 8))
    assert G.seeder_finalize(pre, 42, 8) == _siphash24_ref(0, 0, bytes(pre) + (42).to_bytes(8, "little"))
    assert G.seeder_finalize(pre, 1, 4) == _siphash24_ref(0, 0, bytes(pre) + (1).to_bytes(4, "little"))


def test_xoshiro256pp_and_splitmix_reference_values():
    # rand_xoshiro reference() tests (values from the C reference implementations)
    s = np.array([1, 2, 3, 4], np.uint64)
    expected = [41943041, 58720359, 3588806011781223, 3591011842654386, 9228616714210784205,
                9973669472204895162, 14011001112246962877, 12406186145184390807,
                15849039046786891736, 10450023813501588000]
    assert [int(G._xo_next(s)) for _ in expected] == expected
    s = np.empty(4, np.uint64)
    G._xo_seed(np.uint64(1477776061723855037), s)  # Xoshiro::seed_from_u64 = 4 SplitMix64 outputs
    assert [int(v) for v in s] == [1985237415132408290, 2979275885539914483, 13511426838097143398,
                                   8488337342461049707]


def test_xoshiro_jump_matches_reference_algorithm():
    def nxt(st):
        res = (((st[0] + st[3]) & M64) << 23 | ((st[0] + st[3]) & M64) >> 41) & M64
        res = (res + st[0]) & M64
        t = (st[1] << 17) & M64
        st[2] ^= st[0]; st[3] ^= st[1]; st[1] ^= st[2]; st[0] ^= st[3]; st[2] ^= t  # noqa: E702
        st[3] = ((st[3] << 45) | (st[3] >> 19)) & M64
        return res

    jump = [0x180EC6D33CFD0ABA, 0xD5A61266F0C9392C, 0xA9582618E03FC9AA, 0x39ABDC4529B1661C]
    ref = [11, 22, 33, 44]
    acc = [0, 0, 0, 0]
    for j in jump:
        for b in range(64):
            if j >> b & 1:
                acc = [a ^ v for a, v in zip(acc, ref)]
            nxt(ref)
    s = np.array([11, 22, 33, 44], np.uint64)
    G._xo_jump(s)
    assert [int(v) for v in s] == acc


def test_simplex_noise_properties():
    for seed in (0, 12345, -7):
        for x in (-3.0, 0.0, 5.0, 17.0):  # 1D simplex is zero on the integer lattice
            assert G._simplex1d(np.float32(x), seed) == 0.0
        assert G._simplex2d(np.float32(0.0), np.float32(0.0), seed) == 0.0
    xs = np.arange(0, 50, 0.01, dtype=np.float32)
    v = np.array([G._simplex1d(x, 3) for x in xs])
    assert np.abs(np.diff(v)).max() < 0.2 and v.std() > 0.05  # continuous, non-trivial


def test_iir_filter_matches_scipy_lfilter():
    from scipy.signal import lfilter, lfilter_zi

    rng = np.random.default_rng(1)
    x = rng.random((3, 200)).astype(np.float32)
    for tf in (G.make_lowpass(1300000.0, G.NTSC_RATE), G.make_lowpass(600000.0, G.NTSC_RATE).cascade_self(3),
               G.make_butterworth(1900000.0, G.NTSC_RATE), G.make_notch(0.5, 2.0).with_scale(4.0)):
        a = np.concatenate([[1.0], tf.den]).astype(np.float64)
        b = tf.num.astype(np.float64)
        y = x.copy()
        tf.apply(y, first_sample=False, delay=0)
        np.testing.assert_allclose(y, lfilter(b, a, x.astype(np.float64), axis=1), atol=2e-4)
        y = x.copy()
        tf.apply(y, first_sample=True, delay=0)
        ref = np.stack([lfilter(b, a, row, zi=lfilter_zi(b, a) * row[0])[0] for row in x.astype(np.float64)])
        np.testing.assert_allclose(y, ref, atol=2e-4)
    # delay shifts the output left, padding with the last sample
    tf = G.make_lowpass(1300000.0, G.NTSC_RATE)
    y0, y2 = x.copy(), x.copy()
    tf.apply(y0, False, 0)
    tf.apply(y2, False, 2)
    np.testing.assert_allclose(y2[:, :-2], y0[:, 2:], atol=1e-6)


def test_cascade_equals_repeated_filtering():
    tf = G.make_lowpass(600000.0, G.NTSC_RATE)
    x = np.random.default_rng(2).random((2, 100)).astype(np.float32)
    once = x.copy()
    tf.cascade_self(3).apply(once)
    thrice = x.copy()
    for _ in range(3):
        tf.apply(thrice)
    np.testing.assert_allclose(once, thrice, atol=1e-4)


def test_vhs_ntsc_amount_zero_is_identity():
    img = synthetic_image(80, 60)
    np.testing.assert_allclose(_run("filter.vhs_ntsc", img, amount=0.0), img, atol=1e-6)


def test_vhs_ntsc_clean_signal_path_preserves_image():
    # (ntsc-rs filters start from a zero state, so the first columns darken; keep the image wide enough)
    img = synthetic_image(320, 240)
    clean = dict(head_switching_enabled=False, tracking_noise_enabled=False, composite_noise_enabled=False,
                 ringing_enabled=False, luma_noise_enabled=False, chroma_noise_enabled=False, vhs_enabled=False,
                 luma_smear=0.0, composite_sharpening=0.0, snow=0.0, chroma_phase_noise_intensity=0.0)
    out = _run("filter.vhs_ntsc", img, **clean)
    assert np.abs(out.mean((0, 1)) - img.mean((0, 1))).max() < 0.01  # colour balance kept
    assert np.abs(out - img).mean() < 0.06  # only band-limiting (chroma blur) remains
    for field in ("both", "upper", "lower", "alternating", "interleaved_lower"):
        o = _run("filter.vhs_ntsc", img, use_field=field, **clean)
        assert np.abs(o - img).mean() < 0.06


def test_vhs_ntsc_seeded_and_damaging():
    img = synthetic_image(96, 72)
    a = _run("filter.vhs_ntsc", img, render_seed=1, seed=3)
    b = _run("filter.vhs_ntsc", img, render_seed=99, seed=3)
    c = _run("filter.vhs_ntsc", img, seed=4)
    np.testing.assert_array_equal(a, b)  # explicit seed parameter, independent of the render seed
    assert np.abs(a - c).max() > 0.01
    assert np.abs(a - img).mean() > 0.03


# ---------------------------------------------------------------------------
# Trickle glitches
# ---------------------------------------------------------------------------

def test_block_glitch_zero_strength_is_identity():
    img = synthetic_image(64, 48, alpha=True)
    np.testing.assert_allclose(_run("filter.block_glitch", img, strength=0.0), img, atol=1e-6)


def test_block_glitch_moves_whole_pixels():
    img = synthetic_image(64, 48)
    out = _run("filter.block_glitch", img, strength=1.0, block_size=0.1)
    assert not np.allclose(out, img)
    pix = {tuple(p) for p in img.reshape(-1, 3)}
    assert all(tuple(p) in pix for p in out.reshape(-1, 3))


def test_glitch_variant_channel_split_direction():
    # Android RGBA bytes read as little-endian ints: red comes from x - shift, blue from x + shift
    img = synthetic_image(100, 20)
    out = _run("filter.glitch_variant", img, repeat_count=0, channel_shift=0.5, max_offset=0.0)
    s = 5  # int(100 / 10 * 0.5)
    np.testing.assert_allclose(out[:, s:, 0], img[:, :-s, 0], atol=1e-6)
    np.testing.assert_allclose(out[:, :-s, 2], img[:, s:, 2], atol=1e-6)
    np.testing.assert_allclose(out[..., 1], img[..., 1], atol=1e-6)


def test_glitch_variant_identity():
    img = synthetic_image(64, 48, alpha=True)
    out = _run("filter.glitch_variant", img, repeat_count=0, channel_shift=0.0)
    np.testing.assert_allclose(_premul(out), _premul(img), atol=1e-5)


def test_pixel_melt():
    img = synthetic_image(64, 48)
    np.testing.assert_allclose(_run("filter.pixel_melt", img, strength=0.0), img, atol=1e-6)
    np.testing.assert_allclose(_run("filter.pixel_melt", img, max_drop=0), img, atol=1e-6)
    out = _run("filter.pixel_melt", img, strength=0.5, max_drop=10)
    assert not np.allclose(out, img)
    for x in (0, 17, 63):  # pixels only move down their own column
        col = {tuple(p) for p in img[:, x]}
        assert all(tuple(p) in col for p in out[:, x])


def test_vhs_zero_strength_only_darkens_every_third_row():
    img = synthetic_image(64, 48)
    out = _run("filter.vhs", img, strength=0.0)
    exp = img.copy()
    exp[0::3] *= 0.92
    np.testing.assert_allclose(out, exp, atol=1e-5)


def test_crt_curvature_neutral_and_chroma():
    img = synthetic_image(96, 72)
    out = _run("filter.crt_curvature", img, curvature=0.0, vignette=0.0, chroma=0.0)
    assert np.abs(out - img).mean() < 0.01
    y, x = np.mgrid[0:72, 0:96].astype(np.float32)  # smooth ramps: off-by-one sampling stays small
    ramp = np.dstack([x / 95, y / 71, 1 - x / 95]).astype(np.float32)
    out = _run("filter.crt_curvature", ramp, curvature=0.0, vignette=0.0, chroma=0.05)
    s = int(96 * np.float32(0.05))
    np.testing.assert_allclose(out[:, 20:70, 0], ramp[:, 20 - s:70 - s, 0], atol=0.025)
    np.testing.assert_allclose(out[:, 20:70, 2], ramp[:, 20 + s:70 + s, 2], atol=0.025)
    vig = _run("filter.crt_curvature", img, curvature=0.0, vignette=1.0, chroma=0.0)
    assert vig[0, 0].sum() < img[0, 0].sum() * 0.2  # corners darkened


# ---------------------------------------------------------------------------
# aire
# ---------------------------------------------------------------------------

def test_enhanced_glitch_halftone_and_opaque_output():
    img = synthetic_image(64, 48, alpha=True)
    out = _run("filter.enhanced_glitch", img, channel_shift_x=0.0, corruption_size=0.0, amount=1)
    assert out.shape == (48, 64, 3)  # aire writes alpha = 255
    prem = img[..., :3] * img[..., 3:4] ** 2  # premultiplied pixels multiplied by alpha again
    factor = np.where(np.arange(48) % 4 < 2, 1 - 38 / 255, 1.0)[:, None, None]
    np.testing.assert_allclose(out[:-1], (prem * factor)[:-1], atol=1e-5)
    assert out[-1].max() == 0.0  # upstream copies only height - 1 rows


def test_enhanced_glitch_shifts_green_blue_only():
    img = synthetic_image(100, 50)
    out = _run("filter.enhanced_glitch", img, channel_shift_x=0.1, corruption_size=0.0, amount=1)
    factor = np.where(np.arange(50) % 4 < 2, 1 - 38 / 255, 1.0)[:, None]
    np.testing.assert_allclose(out[:-1, :, 0], (img[..., 0] * factor)[:-1], atol=1e-5)
    # G/B move by (+10 px, +5 rows)
    np.testing.assert_allclose(out[5:-1, 10:, 1], (img[:-6, :-10, 1] * factor[5:-1]), atol=1e-5)


def test_wind_stagger():
    img = synthetic_image(64, 48)
    out = _run("filter.horizontal_wind_stagger", img, strength=0.0)
    np.testing.assert_allclose(out[..., :3], img, atol=1e-5)
    assert np.allclose(out[..., 3], 1.0) if out.shape[2] == 4 else True
    out = _run("filter.horizontal_wind_stagger", img, strength=1.0, amount=10, color="#FFFFFF")
    assert out.shape[2] == 3
    # 10 streams of int(48 / 10) = 4 rows; rows from the last streamed one (39) on are copied verbatim
    np.testing.assert_allclose(out[39:], img[39:], atol=1e-6)
    assert np.all(out[:39, -1] == 1.0, axis=-1).sum() > 10  # right ends of squeezed rows show the colour
    out = _run("filter.horizontal_wind_stagger", img, strength=1.0, amount=10)
    assert out.shape[2] == 4 and (out[:39, -1, 3] == 0.0).sum() > 10  # default clear colour is transparent


def test_color_anomaly_is_per_channel():
    g = np.repeat(np.linspace(0, 1, 64, dtype=np.float32)[None, :, None], 8, axis=0).repeat(3, axis=2)
    out = _run("filter.color_anomaly", g, cutoff=0.7)
    np.testing.assert_allclose(out[..., 0], out[..., 1], atol=1e-6)
    np.testing.assert_allclose(out[..., 1], out[..., 2], atol=1e-6)


# ---------------------------------------------------------------------------
# ImageToolbox native
# ---------------------------------------------------------------------------

def test_jpeg_glitch():
    img = synthetic_image(96, 72)
    clean = _run("filter.glitch", img, repeat_count=0)
    assert np.abs(clean - img).mean() < 0.03  # plain q100 JPEG round trip
    out = _run("filter.glitch", img)
    assert np.abs(out - img).mean() > np.abs(clean - img).mean()


PROCEDURAL = ["filter.column_streak", "filter.concentric_circle_breaks", "filter.concentric_square_breaks",
              "filter.inline_breaks", "filter.radial_streak", "filter.random_color_dispersion",
              "filter.spiral_breaks", "filter.streak_circles", "filter.streak_expand", "filter.tiled_streak"]


@pytest.mark.parametrize("op_id", PROCEDURAL)
def test_procedural_zero_intensity_is_identity(op_id):
    img = synthetic_image(64, 48, alpha=True)
    out = _run(op_id, img, intensity=0.0, rotation=30.0, scale=1.7, offset_x=0.2)
    np.testing.assert_allclose(_premul(out), _premul(img), atol=2e-4)


@pytest.mark.parametrize("op_id", PROCEDURAL)
def test_procedural_default_changes_image(op_id):
    img = synthetic_image(96, 72)
    assert np.abs(_run(op_id, img) - img).max() > 0.05


def test_streak_expand_stretches_centre_row():
    img = synthetic_image(64, 48)
    out = _run("filter.streak_expand", img, intensity=1.0, size=0.5)
    # rows within size/2 of the centre all sample the centre row (pixel row 23.5 -> rows 23/24)
    centre = 0.5 * (img[23] + img[24])
    np.testing.assert_allclose(out[20], centre, atol=1e-4)
    np.testing.assert_allclose(out[27], centre, atol=1e-4)


# ---------------------------------------------------------------------------
# G'MIC
# ---------------------------------------------------------------------------

def test_gmic_random_filters_are_deterministic():
    img = synthetic_image(64, 48)
    for op_id in ("filter.analog_damage", "filter.breaks"):
        a = _run(op_id, img, render_seed=3)
        b = _run(op_id, img, render_seed=3)
        c = _run(op_id, img, render_seed=4)
        np.testing.assert_array_equal(a, b)
        assert np.abs(a - c).max() > 0
    a = _run("filter.chromatic_aberrations", img, render_seed=3, primary_type="random")
    b = _run("filter.chromatic_aberrations", img, render_seed=3, primary_type="random")
    np.testing.assert_array_equal(a, b)


def test_buffer_error_changes_size():
    img = synthetic_image(64, 48, alpha=True)
    out = _run("filter.buffer_error", img)
    assert out.shape == (24, 32, 4)  # 50% x 50% buffer, alpha resized


def test_streak_without_mask_colour_is_identity():
    img = synthetic_image(64, 48) * 0.5 + 0.25  # no pure red pixels
    out = _run("filter.streak", img)
    np.testing.assert_allclose(out, img, atol=1.5 / 255)


def test_jpeg_artifacts_quality():
    img = synthetic_image(64, 48)
    hi = _run("filter.jpeg_artifacts", img, quality=100)
    lo = _run("filter.jpeg_artifacts", img, quality=5)
    assert np.abs(hi - img).mean() < np.abs(lo - img).mean()


def test_pixel_sort_threshold_order_is_validated():
    _spec("filter.pixel_sort")
    with pytest.raises(LazyRoomError):
        _run("filter.pixel_sort", synthetic_image(32, 24), lower_threshold=80, upper_threshold=20)


def test_breaks_processes_alpha():
    out = _run("filter.breaks", synthetic_image(48, 36, alpha=True), boundary="transparent")
    assert out.shape[2] == 4
