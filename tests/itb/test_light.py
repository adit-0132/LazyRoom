"""Tests for the ImageToolbox "Light" group port (filter.* ops with group == "light")."""
import math

import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.params import Float, Int
from lazyroom.core.registry import all_ops, ensure_loaded, get_op
from lazyroom.ops.itb import _light_impl as L
from lazyroom.ops.itb import light as light_mod
from tests.conftest import synthetic_image

ensure_loaded()
LIGHT_OPS = [s for s in all_ops(include_hidden=True) if s.group == "light"]
IMG = synthetic_image(48, 36)


def run(op_id, img=IMG, ctx=None, **params):
    spec = get_op(op_id)
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    return spec.run(img, ctx or RenderContext(seed=3), spec.validate(params))


def gray_ramp(n=21):
    ramp = np.repeat(np.linspace(0, 1, n, dtype=np.float32)[None, :, None], 3, axis=2)
    return np.repeat(ramp, 2, axis=0)


def test_all_54_registered():
    assert len(LIGHT_OPS) == 54


# ---------------------------------------------------------------------------
# (1) every op at parameter minima and maxima
# ---------------------------------------------------------------------------

def _extreme(spec, which):
    vals = {}
    for p in spec.params:
        if isinstance(p, (Float, Int)) and p.min is not None and p.max is not None:
            vals[p.name] = p.min if which == "min" else p.max
    return vals


@pytest.mark.parametrize("which", ["min", "max"])
@pytest.mark.parametrize("spec", LIGHT_OPS, ids=[s.id for s in LIGHT_OPS])
def test_param_extremes(spec, which):
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    out = spec.run(IMG, RenderContext(seed=1), spec.validate(_extreme(spec, which)))
    assert out.shape == IMG.shape and out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0 and out.max() <= 1


@pytest.mark.parametrize("spec", [s for s in LIGHT_OPS if "gmic" not in s.requires], ids=lambda s: s.id)
def test_preview_scale_and_tiny_images(spec):
    """Preview proxies (ctx.scale < 1) and images smaller than the CLAHE grid must work."""
    tiny = synthetic_image(7, 5)
    out = spec.run(tiny, RenderContext(scale=0.25), spec.validate({}))
    assert out.shape == tiny.shape and np.isfinite(out).all()


# ---------------------------------------------------------------------------
# (2) semantic checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_id,params", [
    ("filter.brightness", {"bias": 0}),
    ("filter.contrast", {"gain": 1}),
    ("filter.exposure", {"exposure": 0}),
    ("filter.gamma", {"gamma": 1}),
    ("filter.levels", {}),
    ("filter.haze", {"distance": 0, "slope": 0}),
    ("filter.glow", {"amount": 0}),
    ("filter.rays", {"opacity": 0}),
    ("filter.bloom", {"strength": 0, "gamma": 1}),
    ("filter.highlights_and_shadows", {"shadows": 1}),
    ("filter.bokeh_lights", {"intensity": 0}),
    ("filter.fireworks", {"intensity": 0}),
])
def test_identity_settings(op_id, params):
    out = run(op_id, **params)
    assert np.allclose(out, IMG, atol=2e-3), op_id


def test_white_balance_neutral_is_near_identity():
    out = run("filter.white_balance", temperature=5000, tint=0)
    assert np.allclose(out, IMG, atol=5e-3)


def test_white_balance_warm_vs_cool():
    warm = run("filter.white_balance", temperature=9000, tint=0)
    cool = run("filter.white_balance", temperature=2000, tint=0)
    assert (warm[..., 0] - warm[..., 2]).mean() > (cool[..., 0] - cool[..., 2]).mean()


@pytest.mark.parametrize("op_id", ["filter.aces_filmic_tone_mapping", "filter.hable_filmic_tone_mapping",
                                   "filter.uchimura", "filter.logarithmic_tone_mapping", "filter.drago",
                                   "filter.hejl_burgess_tone_mapping", "filter.aldridge"])
def test_tone_mappers_monotone_and_grey(op_id):
    out = run(op_id, img=gray_ramp())
    row = out[0, :, 0]
    assert np.all(np.diff(row) >= -1e-5), row
    assert np.allclose(out[..., 0], out[..., 1], atol=1e-5) and np.allclose(out[..., 1], out[..., 2], atol=1e-5)


def test_aces_hill_equals_aces_filmic_like_upstream():
    assert np.array_equal(run("filter.aces_hill_tone_mapping", exposure=1.3),
                          run("filter.aces_filmic_tone_mapping", exposure=1.3))


def test_aces_filmic_known_value():
    # sRGB 0.5 -> linear (aire threshold) -> ACES fit -> sRGB
    lin = ((0.5 + 0.055) / 1.055) ** 2.4
    x = lin
    y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    exp = 1.055 * y ** (1 / 2.4) - 0.055
    out = run("filter.aces_filmic_tone_mapping", img=np.full((2, 2, 3), 0.5, np.float32))
    assert abs(float(out[0, 0, 0]) - exp) < 1e-4


def test_mobius_identity_below_transition():
    ramp = gray_ramp()
    out = run("filter.mobius", img=ramp, exposure=1, transition=0.9, peak=1)
    lin = L.aire_srgb_to_linear(ramp[0, :, 0])
    keep = lin <= 0.9
    assert np.allclose(out[0, keep, 0], ramp[0, keep, 0], atol=1e-4)


def test_gamma_is_normalised_power_curve():
    # Lead decision: upstream's raw 0..255 power is a bug; we apply out = in ** gamma on 0..1 values.
    img = np.full((2, 2, 3), 0.5, np.float32)
    out = run("filter.gamma", img=img, gamma=2)
    assert abs(float(out[0, 0, 0]) - 0.25) < 2e-3
    assert np.allclose(run("filter.gamma", img=img, gamma=1), img, atol=2e-3)


def _low_contrast():
    img = synthetic_image(256, 192)
    return (0.4 + 0.2 * img).astype(np.float32)


EQUALIZERS = ["filter.clahe", "filter.clahe_hsl", "filter.clahe_hsv", "filter.clahe_lab", "filter.clahe_luv",
              "filter.clahe_oklab", "filter.clahe_oklch", "filter.clahe_jzazbz", "filter.equalize_histogram_adaptive",
              "filter.equalize_histogram_adaptive_hsl", "filter.equalize_histogram_adaptive_hsv",
              "filter.equalize_histogram_adaptive_lab", "filter.equalize_histogram_adaptive_luv",
              "filter.equalize_histogram_hsv"]


@pytest.mark.parametrize("op_id", EQUALIZERS)
def test_equalization_increases_contrast(op_id):
    # tiles must hold enough pixels: upstream's clip limit is int(threshold * tile_area / bins)
    img = _low_contrast()
    defaults = get_op(op_id).defaults()
    extra = {"grid_size_x": 2, "grid_size_y": 2} if "grid_size_x" in defaults else {}
    if "threshold" in defaults:
        extra["threshold"] = 2.0
    out = run(op_id, img=img, **extra)
    assert out.mean(axis=2).std() > img.mean(axis=2).std() * 1.3


def test_clahe_negative_threshold_is_near_identity():
    # clip limit 0 -> uniform histogram -> linear mapping (tile area a multiple of 256 here)
    img = synthetic_image(320, 240)
    out = run("filter.clahe", img=img, threshold=-1, grid_size_x=2, grid_size_y=2)
    assert np.abs(out - img).mean() < 2 / 255


def test_clahe_small_tiles_collapse_like_upstream():
    # 96x72 with an 8x8 grid: 108-px tiles, int(0.5 * 108 / 256) == 0 -> every bin clipped to 0
    out = run("filter.clahe", img=synthetic_image(96, 72))
    assert out.mean() < 0.15


def test_global_hsv_equalization_keeps_hue():
    from lazyroom.core.color import rgb_to_hsv

    img = _low_contrast()
    out = run("filter.equalize_histogram_hsv", img=img)
    h0, h1 = rgb_to_hsv(img), rgb_to_hsv(out)
    sel = (h0[..., 1] > 0.2) & (h1[..., 1] > 0.2)
    dh = np.abs((h1[..., 0] - h0[..., 0] + 0.5) % 1.0 - 0.5) * 360
    assert np.median(dh[sel]) < 3


def _rust_round(x):
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def _scalar_ahe(q, bins, gw, gh, threshold):
    """Literal transliteration of histogram_equalization 0.2.4 ``clahe_impl_u16`` (per-pixel loops)."""
    height, width = q.shape
    tw, th = width // gw, height // gh
    ntx, nty = width // tw, height // th
    maxb = bins - 1
    f32 = np.float32
    hists = []
    for h in range(nty):
        row = []
        for w in range(ntx):
            sx, sy = w * tw, h * th
            ex = width if w + 1 == ntx else (w + 1) * tw
            ey = height if h + 1 == nty else (h + 1) * th
            b = [0] * bins
            for y in range(sy, ey):
                for x in range(sx, ex):
                    b[min(int(q[y, x]), maxb)] += 1
            if threshold is not None:
                tv = threshold * (ex - sx) * (ey - sy) / bins
                cl = int(tv) if tv > 0 else 0
                excess = sum(v for v in b if v > cl)
                b = [min(v, cl) if v > cl else v for v in b]
                b = [v + int(excess / bins) for v in b]
            b = list(np.cumsum(b))
            mn = min(b)
            area = (ey - sy) * (ex - sx)
            for i in range(bins):
                if area == mn:  # 1/0 -> inf; NaN/inf ignored by Rust min/max -> max_bins
                    b[i] = maxb
                else:
                    b[i] = int(max(min(_rust_round(maxb * (b[i] - mn) / (area - mn)), maxb), 0))
            row.append(b)
        hists.append(row)
    out = np.zeros(q.shape, np.float32)
    for y in range(height):
        for x in range(width):
            cxf = f32((f32(x) - f32(tw) / f32(2)) / f32(tw))
            ryf = f32((f32(y) - f32(th) / f32(2)) / f32(th))
            x1 = f32((f32(x) - (f32(int(cxf)) + f32(0.5)) * f32(tw)) / f32(tw))
            y1 = f32((f32(y) - (f32(int(ryf)) + f32(0.5)) * f32(th)) / f32(th))
            v = min(int(q[y, x]), maxb)
            r, c = min(int(max(ryf, 0)), nty - 1), min(int(max(cxf, 0)), ntx - 1)
            r1, c1 = min(r + 1, nty - 1), min(c + 1, ntx - 1)

            def lerp(a, b, t):
                return f32(f32(a) * (f32(1) - t) + f32(b) * t)
            it = lerp(lerp(hists[r][c][v], hists[r][c1][v], x1), lerp(hists[r1][c][v], hists[r1][c1][v], x1), y1)
            out[y, x] = float(int(max(min(it, maxb), 0)))
    return out


@pytest.mark.parametrize("bins,grid,threshold,const", [
    (16, (3, 2), 0.5, False), (16, (3, 2), None, False), (256, (4, 4), 2.0, False), (3, (2, 3), -1.0, False),
    (128, (2, 2), None, True), (16, (5, 5), 10.0, False),
])
def test_ahe_map_matches_scalar_transliteration(bins, grid, threshold, const):
    rng = np.random.default_rng(bins + grid[0])
    q = rng.integers(0, bins, (23, 31)).astype(np.float32)
    if const:
        q[:] = 0
    assert np.array_equal(L.ahe_map(q, bins, grid[0], grid[1], threshold), _scalar_ahe(q, bins, *grid, threshold))


def test_lab_chroma_storage_keeps_negative_chroma():
    # Lead decision: upstream clamps a*/b* below -1 (bug); we keep the exact chroma so greens/blues survive.
    green = np.zeros((4, 4, 3), np.float32)
    green[..., 1] = 0.8
    _, a, _ = L.cu_xyz_to_lab(*L.cu_rgb_to_xyz(0.0, 0.8, 0.0))
    assert a < -1
    _, side = L.encode_plane(green, L.SPACES["lab"], 127.0)
    assert np.allclose(side[..., 0], a, atol=1e-3)
    rgb = np.random.default_rng(1).random((24, 24, 3), dtype=np.float32)
    for space in ("lab", "luv"):
        q, side = L.encode_plane(rgb, L.SPACES[space], 255.0)
        out = L.decode_plane(q, side, L.SPACES[space], 255.0)
        assert np.abs(out - rgb).max() < 0.03, space


def test_equalize_colour_round_trip_without_mapping():
    """encode -> decode with an identity mapping reproduces the input up to upstream quantisation."""
    rng = np.random.default_rng(0)
    rgb = rng.random((32, 32, 3), dtype=np.float32)
    for space in ("ycgco", "hsv", "hsl", "oklab", "oklch"):
        q, side = L.encode_plane(rgb, L.SPACES[space], 255.0)
        out = L.decode_plane(q, side, L.SPACES[space], 255.0)
        assert np.abs(out - rgb).max() < 0.02, space


def test_bloom_brightens_highlights():
    img = synthetic_image(96, 72)
    out = run("filter.bloom", img=img, threshold=0.3, strength=2, gamma=1)
    assert np.all(out >= img - 1e-5) and out.mean() > img.mean()


def test_glow_brightens():
    out = run("filter.glow", amount=0.3)
    assert np.all(out >= IMG - 1e-6) and out.mean() > IMG.mean()


def test_light_effects_flat_bump_is_half_brightness():
    out = run("filter.light_effects", bump_height=0)
    assert np.allclose(out, IMG * 0.5, atol=1e-4)


def test_vignette_center_and_corner():
    img = np.full((40, 60, 3), 0.8, np.float32)
    out = run("filter.vignette", img=img, start=0.1, end=0.5, color="#000000")
    assert np.allclose(out[20, 30], 0.8, atol=1e-3)
    assert out[0, 0, 0] < 0.2


def test_haze_brightens_with_negative_distance():
    out = run("filter.haze", distance=-0.2, slope=0)
    assert np.allclose(out, np.clip((IMG + 0.2) / 1.2, 0, 1), atol=1e-5)


@pytest.mark.parametrize("op_id", ["filter.flare", "filter.sparkle"])
def test_flare_and_sparkle_mix_towards_colour(op_id):
    out = run(op_id, color="#FFFFFF")
    assert np.all(out >= IMG - 1e-5) and out.mean() > IMG.mean()
    black = run(op_id, color="#000000")
    assert np.all(black <= IMG + 1e-5)


def test_flare_preview_matches_full_resolution():
    full = synthetic_image(120, 90)
    small = full[::2, ::2]
    a = run("filter.flare", img=full)[::2, ::2]
    b = run("filter.flare", img=small, ctx=RenderContext(scale=0.5))
    assert np.abs(a - b).mean() < 0.02


def test_sparkle_ray_lengths_use_java_random_seed_371():
    r = L.JavaRandom(371)
    # java.util.Random(371).nextGaussian() -- value fixed by the JDK algorithm
    g = r.next_gaussian()
    r2 = L.JavaRandom(371)
    assert g == r2.next_gaussian() and -5 < g < 5


def test_java_random_matches_jdk_sequence():
    # new java.util.Random(42).nextInt() == -1170105035 (well-known JDK value)
    assert L.JavaRandom(42).next_int() == -1170105035


@pytest.mark.parametrize("op_id", ["filter.bokeh_lights", "filter.fireworks"])
def test_procedural_lights_add_light_and_depend_on_seed(op_id):
    a = run(op_id, seed=1)
    b = run(op_id, seed=2)
    assert np.all(a >= IMG - 1e-6) and a.mean() > IMG.mean()
    assert not np.allclose(a, b)
    assert np.array_equal(a, run(op_id, seed=1))


def test_rays_only_adds_where_bright():
    dark = np.full((30, 40, 3), 0.1, np.float32)
    out = run("filter.rays", img=dark, threshold=0.5)
    assert np.allclose(out, dark)


# ---------------------------------------------------------------------------
# G'MIC command strings (must equal ImageToolboxLibs' builders for the defaults)
# ---------------------------------------------------------------------------

EXPECTED_GMIC = {
    "filter.cinematic_dream": "fx_highlight_bloom 75,45,55,28,18 fx_pop_shadows 0.55,6,1",
    "filter.dynamic_contrast": "rep_dynamic_contrast 12,128,220",
    "filter.equalize_light": "fx_equalize_light 88,1,12",
    "filter.equalize_shadow": "fx_equalize_shadow 0.8",
    "filter.highlight_synthesis": "iain_highlight_synthesis 1.1,0.7,1.15,1",
    "filter.light_glow": "fx_lightglow 30,0.5,8,0.8,12",
    "filter.light_patch": "fx_light_patch 5,0.7,2.5,12",
    "filter.local_contrast_enhancement": "fx_LCE 80,1.2,1.2,1.2,12",
    "filter.pop_shadows": "fx_pop_shadows 0.75,5,1",
    "filter.relief_light": "fx_light_relief 0.3,0.2,0.2,0,1,50,50,5,0.5,0,0",
    "filter.retinex": "fx_retinex 75,16,1,1,1,5,15,80,250",
    "filter.shadow_patch": "fx_shadow_patch 0.7,0",
    "filter.tone_enhance": "gcd_tone_enhance 0.6,0.9,0.5,1.1,128,0.4,1,0.6,0.2,3,5,0,0",
    "filter.zone_system": "fx_zonesystem 2,9,1.1,1.15,8,245",
}


@pytest.mark.parametrize("op_id", sorted(EXPECTED_GMIC))
def test_gmic_command_strings(op_id, monkeypatch):
    captured = {}

    def fake(img, cmd, **kw):
        captured["cmd"] = cmd
        return img

    monkeypatch.setattr(light_mod, "run_gmic", fake)
    spec = get_op(op_id)
    spec.fn(IMG, RenderContext(), **spec.validate({}))
    cmd = captured["cmd"]
    if cmd.startswith("srand "):  # random-based filters are seeded for determinism
        cmd = cmd.split(" ", 2)[2]
    assert cmd == EXPECTED_GMIC[op_id]


def test_gmic_px_params_scale_with_preview(monkeypatch):
    captured = {}
    monkeypatch.setattr(light_mod, "run_gmic", lambda img, cmd, **kw: captured.setdefault("cmd", cmd) and img)
    spec = get_op("filter.local_contrast_enhancement")
    spec.fn(IMG, RenderContext(scale=0.5), **spec.validate({}))
    assert captured["cmd"].startswith("fx_LCE 40,")


def test_choice_indices_match_upstream_enums():
    assert light_mod._enum_index(light_mod._CHANNELS, "luminance") == 12
    assert light_mod._enum_index(light_mod._CHANNELS, "ryb_blue") == 46
    assert light_mod._enum_index(light_mod._LIGHTGLOW_MODES, "overlay") == 8
    assert light_mod._enum_index(light_mod._COLOR_SPACES6, "ycbcr") == 5
