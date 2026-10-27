"""ImageToolbox colour tools: model conversions, mixing, harmonies, shading, colour picking, palette PDF,
colour_info extensions, Material You (material-color-utilities port + DynamicTheme), androidx Palette,
adjust.color_curves and the curves preset codec — see docs/dev/port_reports/color_tools.md.

Reference values marked "MCU test" come from material-color-utilities' own unit tests (hct_test,
tonal_palette_test, score_test, quantizer_wu_test, scheme_tonal_spot_test, scheme_monochrome_test,
scheme_fidelity_test); curve expectations come from ImageToolbox's lib/curves androidTests
(ColorCurvesFilterInstrumentedTest.kt, CurvesPresetCodecTest.kt).
"""
from __future__ import annotations

import colorsys
import io
import json
import math
import zipfile

import numpy as np
import pytest

import lazyroom.ops.curves_ext as ce
import lazyroom.tools._curves_presets as cp
import lazyroom.tools.color_tools  # noqa: F401  (registers the colour tools)
from lazyroom.core.context import RenderContext
from lazyroom.core.errors import LazyRoomError
from lazyroom.core.io import save_image
from lazyroom.core.registry import get_op
from lazyroom.tools import _androidx_palette as AP
from lazyroom.tools import _color_models as cm
from lazyroom.tools import _material as M
from tests.conftest import synthetic_image


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("LAZYROOM_HOME", str(tmp_path / "home"))
    from lazyroom.api import LazyRoom

    return LazyRoom()


# ---------------------------------------------------------------------------
# colour models (textbook values)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rgb,hsv,hsl,cmyk", [
    ((255, 0, 0), (0, 1, 1), (0, 1, 0.5), (0, 1, 1, 0)),
    ((0, 255, 0), (120, 1, 1), (120, 1, 0.5), (1, 0, 1, 0)),
    ((51, 102, 204), (220, 0.75, 0.8), (220, 0.6, 0.5), (0.75, 0.5, 0, 0.2)),
    ((128, 128, 128), (0, 0, 128 / 255), (0, 0, 128 / 255), (0, 0, 0, 1 - 128 / 255)),
    ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0, 1)),
])
def test_hsv_hsl_cmyk(rgb, hsv, hsl, cmyk):
    assert cm.rgb_to_hsv(*rgb) == pytest.approx(hsv, abs=1e-5)
    assert cm.rgb_to_hsl(*rgb) == pytest.approx(hsl, abs=1e-5)
    assert cm.rgb_to_cmyk(*rgb) == pytest.approx(cmyk, abs=1e-5)
    assert cm.hsv_to_rgb(*cm.rgb_to_hsv(*rgb)) == rgb
    assert cm.hsl_to_rgb(*cm.rgb_to_hsl(*rgb)) == rgb


def test_inverse_conversions_textbook():
    assert cm.hsl_to_rgb(120, 1, 0.25) == (0, 128, 0)
    assert cm.hsv_to_rgb(300, 1, 0.5) == (128, 0, 128)
    assert cm.cmyk_to_rgb(0, 1, 1, 0) == (255, 0, 0)
    assert cm.cmyk_to_rgb(0.5, 0.25, 0, 0.2) == (102, 153, 204)


def test_xyz_lab_d65():
    assert cm.rgb_to_xyz(255, 255, 255) == pytest.approx((95.05, 100.0, 108.9), abs=0.01)
    assert cm.xyz_to_lab(*cm.rgb_to_xyz(255, 255, 255)) == pytest.approx((100, 0, 0), abs=0.02)
    assert cm.xyz_to_lab(*cm.rgb_to_xyz(255, 0, 0)) == pytest.approx((53.24, 80.09, 67.20), abs=0.05)
    assert cm.xyz_to_lab(*cm.rgb_to_xyz(0, 0, 255)) == pytest.approx((32.30, 79.19, -107.86), abs=0.1)
    for rgb in [(12, 200, 99), (250, 3, 128), (0, 0, 0), (1, 2, 3)]:
        assert cm.xyz_to_rgb(*cm.lab_to_xyz(*cm.xyz_to_lab(*cm.rgb_to_xyz(*rgb)))) == rgb


def test_parse_models_and_formats():
    blue = (255, 51, 102, 204)
    for text in ["#3366CC", "3366cc", "51, 102, 204", "rgb(51, 102, 204)", "hsv(220, 75%, 80%)",
                 "hsl(220, 60%, 50%)", "cmyk(75%, 50%, 0%, 20%)", "lab(45.0342, 18.7212, -57.8593)"]:
        assert cm.parse(text)[0] == blue, text
    assert cm.parse("#80FF0000")[0] == (0, 128, 255, 0)                      # CSS #RRGGBBAA
    assert cm.parse("#80FF0000", hex_alpha_first=True)[0] == (128, 255, 0, 0)  # Android #AARRGGBB
    f = cm.describe(blue)["formats"]
    assert f["rgb"] == "rgb(51, 102, 204)" and f["hsl"] == "hsl(220, 60%, 50%)" and f["hsv"] == "hsv(220, 75%, 80%)"
    assert f["cmyk"] == "cmyk(75%, 50%, 0%, 20%)" and f["rgba"] == "rgba(51, 102, 204, 1)"
    assert f["cmyk_field"] == "75, 50, 0, 20" and f["hex"] == "#3366CC"


def test_color_names():
    assert cm.color_name((255, 255, 0, 0)) == "Red"
    c, matched = cm.color_from_name("red")
    assert c == (255, 255, 0, 0) and matched == "Red"
    assert cm.color_from_name("zzzz-no-such-colour-qq")[1] == "Olo"  # containment either way (sic)
    assert cm.color_from_name("\u2603\u2603") == ((255, 0, 0, 0), None)  # no match: ImageToolbox falls back to black


@pytest.mark.parametrize("kind", cm.HARMONY_TYPES)
def test_harmony_hue_offsets(kind):
    base = (255, 51, 102, 204)
    cols = cm.harmony(base, kind)
    assert cols[0] == base and len(cols) == len(cm.HARMONY_OFFSETS[kind])
    h0 = cm.rgb_to_hsv(*base[1:])[0]
    for c, off in zip(cols, cm.HARMONY_OFFSETS[kind]):
        h = cm.rgb_to_hsv(*c[1:])[0]
        d = (h - h0 - off + 180) % 360 - 180
        assert abs(d) < 1.5, (kind, off, h)


def test_mixing_and_shading():
    red, blue = (255, 255, 0, 0), (255, 0, 0, 255)
    assert cm.mix_with(red, blue, 3) == [red, (255, 127, 0, 127), blue]  # blendARGB truncates
    sh = cm.shading((255, 51, 102, 204), 5)
    assert [len(v) for v in sh.values()] == [5, 5, 5]
    assert all(row[0] == (255, 51, 102, 204) for row in sh.values())
    assert sh["shades"][-1] == (255, 10, 20, 40)  # 4 / (5 + 1 - 1) = 0.8 towards black (max 90 %)
    assert sh["tints"][-1][1:] == (187, 204, 238)  # 4 / 6 towards white (max 80 %)


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

def test_color_convert_tool(api):
    r = api.tool_run("color_convert", {"value": "hsl(220, 60%, 50%)"})
    assert r["hex"] == "#3366CC" and r["input"]["model"] == "hsl" and r["name"]
    assert api.tool_run("color_convert", {"value": "Red"})["hex"] == "#FF0000"
    json.dumps(r, allow_nan=False)


def test_mix_harmony_shading_tools(api, tmp_path):
    r = api.tool_run("color_mix", {"color": "#FF0000", "mix_with": "#0000FF", "variations": 3,
                                   "swatch_output": str(tmp_path / "m.png")})
    assert [c["hex"] for c in r["colors"]] == ["#FF0000", "#7F007F", "#0000FF"] and r["swatch"]["width"] == 3 * 96
    h = api.tool_run("color_harmonies", {"color": "#FF0000"})
    assert h["harmonies"]["complementary"]["colors"][1]["hex"] == "#00FFFF"
    assert [c["hex"] for c in h["harmonies"]["triadic"]["colors"]] == ["#FF0000", "#00FF00", "#0000FF"]
    assert [c["hex"] for c in h["harmonies"]["square"]["colors"]] == ["#FF0000", "#80FF00", "#00FFFF", "#8000FF"]
    s = api.tool_run("color_shading", {"color": "#3366CC", "variations": 4})
    assert len(s["tints"]) == len(s["tones"]) == len(s["shades"]) == 4


def test_color_pick(api, tmp_path):
    img = np.zeros((40, 60, 3), np.float32)
    img[:, 30:] = [0.2, 0.4, 0.8]
    img[10, 5] = [1, 0, 0]
    p = tmp_path / "p.png"
    save_image(img, p)
    r = api.tool_run("color_pick", {"input": str(p), "point": [0.75, 0.5]})
    assert r["hex"] == "#3366CC" and r["position_px"] == [45, 20]
    assert api.tool_run("color_pick", {"input": str(p), "point": [5, 10], "unit": "px"})["hex"] == "#FF0000"
    avg = api.tool_run("color_pick", {"input": str(p), "point": [5, 10], "unit": "px", "radius": 1})
    assert avg["pixels"] == 9 and avg["rgb"] == [28, 0, 0]  # 255 / 9


def test_color_info_backwards_compatible(api):
    r = api.tool_run("color_info", {"color": "#3366CC"})
    for k in ("hex", "rgb", "alpha", "hsl", "hsv", "cmyk", "lab", "oklab", "harmonies"):
        assert k in r
    assert r["harmonies"]["tetradic"] and r["itb"]["harmonies"]["tetradic"] == ["#3366CC", "#4C33CC", "#CC9933",
                                                                                 "#B3CC33"]
    assert r["name"] == r["itb"]["name"]


# ---------------------------------------------------------------------------
# palette PDF
# ---------------------------------------------------------------------------

def _pdf_text(path):
    from pypdf import PdfReader

    rd = PdfReader(str(path))
    return len(rd.pages), "\n".join(pg.extract_text() for pg in rd.pages)


def test_palette_pdf_from_colors(api, tmp_path):
    cols = [f"C{i}=#{i * 5:02X}{(255 - i * 5):02X}80" for i in range(30)]
    out = tmp_path / "p.pdf"
    r = api.tool_run("palette_pdf", {"colors": cols, "palette_name": "My Palette", "output": str(out)})
    n, text = _pdf_text(out)
    assert n == r["pages"] == 3 and r["layout"]["colors_per_page"] == [12, 12, 6]
    assert "My Palette" in text and "C0" in text and "#00FF80" in text and "C29" in text
    r2 = api.tool_run("palette_pdf", {"colors": cols[:5], "columns": 6, "show_hex_values": False,
                                      "show_palette_name": False, "output": str(tmp_path / "q.pdf")})
    n, text = _pdf_text(tmp_path / "q.pdf")
    assert n == 1 and "#00FF80" not in text and "C4" in text and r2["layout"]["caption_height"] == 26


def test_palette_pdf_from_palette_file(api, tmp_path):
    gpl = tmp_path / "pal.gpl"
    gpl.write_text("GIMP Palette\nName: Sunset\nColumns: 3\n#\n255 0 0\tRed\n0 255 0\tGreen\n255 0 0\tRed\n")
    r = api.tool_run("palette_pdf", {"input": str(gpl), "show_source_filename": True, "output": str(tmp_path / "g.pdf")})
    n, text = _pdf_text(tmp_path / "g.pdf")
    assert r["source_type"] == "palette_file" and r["palette_name"] == "Sunset" and len(r["colors"]) == 2
    assert "pal.gpl" in text and "Sunset" in text and "Green" in text


# =============================================================================
# Material You + androidx Palette
# =============================================================================

# --- HCT / CAM16 -------------------------------------------------------------

@pytest.mark.parametrize("argb,hue,chroma,tone", [
    (0xFFFF0000, 27.408, 113.358, 53.233),   # MCU test (red)
    (0xFF00FF00, 142.140, 108.410, 87.737),  # MCU test (green)
    (0xFF0000FF, 282.788, 87.231, 32.302),   # MCU test (blue)
    (0xFFFFFFFF, 209.492, 2.869, 100.0),     # MCU test (white)
])
def test_hct_known_values(argb, hue, chroma, tone):
    h = M.hct_from_argb(argb)
    assert h.hue == pytest.approx(hue, abs=2e-3)
    assert h.chroma == pytest.approx(chroma, abs=2e-3)
    assert h.tone == pytest.approx(tone, abs=2e-3)


def test_hct_round_trip_and_gamut_mapping():
    rng = np.random.default_rng(3)
    for v in rng.integers(0, 1 << 24, 300):
        argb = 0xFF000000 | int(v)
        h = M.Hct(argb)
        assert M.Hct.from_hct(h.hue, h.chroma, h.tone).to_int() == argb
    # out-of-gamut request: tone preserved, chroma reduced
    h = M.Hct.from_hct(120.0, 200.0, 50.0)
    assert abs(h.tone - 50.0) < 0.5 and h.chroma < 200


def test_hctsolver_tables_match_viewing_conditions():
    vc = M.VC
    derived = np.diag(vc.rgb_d) @ np.array(M.XYZ_TO_CAM16RGB) @ np.array(M.SRGB_TO_XYZ)
    ratio = np.array(M._SCALED_DISCOUNT_FROM_LINRGB) / derived
    assert np.allclose(ratio, ratio[0, 0], rtol=1e-6)
    assert np.allclose(np.array(M._LINRGB_FROM_SCALED_DISCOUNT) @ np.array(M._SCALED_DISCOUNT_FROM_LINRGB), np.eye(3),
                       atol=1e-9)


def test_tonal_palette_blue():  # MCU test
    tp = M.TonalPalette.from_int(0xFF0000FF)
    expected = {100: 0xFFFFFFFF, 95: 0xFFF1EFFF, 90: 0xFFE0E0FF, 80: 0xFFBEC2FF, 70: 0xFF9DA3FF, 60: 0xFF7C84FF,
                50: 0xFF5A64FF, 40: 0xFF343DFF, 30: 0xFF0000EF, 20: 0xFF0001AC, 10: 0xFF00006E, 0: 0xFF000000}
    assert {t: tp.tone(t) for t in expected} == expected


def test_key_color_finds_requested_chroma():
    tp = M.TonalPalette.from_hue_and_chroma(270.0, 16.0)
    k = tp.key_color
    assert abs(k.hue - 270.0) < 1.0 and abs(k.chroma - 16.0) < 1.0


# --- quantizers / score --------------------------------------------------------

@pytest.mark.parametrize("pixels,expected", [
    ([0xFF141216], {0xFF141216}), ([0xFFFF0000], {0xFFFF0000}), ([0xFF0000FF] * 5, {0xFF0000FF}),
    ([0xFFFF0000] * 2 + [0xFF00FF00] * 3, {0xFFFF0000, 0xFF00FF00}),
    ([0xFFFF0000, 0xFF00FF00, 0xFF0000FF], {0xFFFF0000, 0xFF00FF00, 0xFF0000FF}),
])
def test_wu_and_celebi_on_solid_colours(pixels, expected):  # MCU quantizer tests
    px = np.array(pixels, np.int64)
    assert set(M.QuantizerWu().quantize(px, 128)) == expected
    res = M.quantize_celebi(px, 128)
    assert set(res) == expected and sum(res.values()) == len(pixels)


def test_score():  # MCU score tests
    assert M.score({0xFFFF0000: 1, 0xFF00FF00: 1, 0xFF0000FF: 1}) == [0xFFFF0000, 0xFF00FF00, 0xFF0000FF]
    assert M.score({0xFF000000: 1}) == [0xFF4285F4]          # greys -> Google Blue fallback
    assert M.score({0xFF808080: 3, 0xFFFFFFFF: 2}) == [0xFF4285F4]
    assert M.score({0xFF008772: 1, 0xFF318477: 1}) == [0xFF008772]
    assert M.score({0xFF008772: 1, 0xFF008587: 1, 0xFF007EBC: 1}, desired=2) == [0xFF007EBC, 0xFF008772]


def test_java_random_matches_jdk():
    r = M.JavaRandom(42)
    assert [r.next_int(10) for _ in range(5)] == [0, 3, 8, 4, 0]  # new java.util.Random(42).nextInt(10) x5


def test_source_colors_from_image():
    img = np.zeros((60, 90, 3), np.uint8)
    img[:, :45] = (200, 30, 30)
    img[:, 45:] = (30, 60, 200)
    seeds, pops = M.source_colors_from_image(img)
    hues = sorted(M.Hct(a).hue for a, _ in seeds)
    assert len(seeds) == 2 and sum(pops.values()) == 60 * 90
    assert hues[0] < 40 and 250 < hues[1] < 300


# --- schemes -----------------------------------------------------------------

def _hexes(cs, *roles):
    return [M.argb_to_hex(cs[r]) for r in roles]


def test_scheme_tonal_spot_blue():  # MCU test values (SchemeTonalSpot, seed 0xff0000ff)
    light = M.to_color_scheme(M.make_scheme(0xFF0000FF, "tonal_spot", False, 0.0))
    dark = M.to_color_scheme(M.make_scheme(0xFF0000FF, "tonal_spot", True, 0.0))
    assert _hexes(light, "primary", "primaryContainer", "surface", "onSurface") == \
        ["#555992", "#E0E0FF", "#FBF8FF", "#1B1B21"]
    assert _hexes(dark, "primary", "primaryContainer", "onPrimaryContainer", "surface") == \
        ["#BEC2FF", "#3E4278", "#E0E0FF", "#131318"]


def test_scheme_monochrome_and_fidelity():
    mono_l = M.to_color_scheme(M.make_scheme(0xFF0000FF, "monochrome", False, 0.0))
    mono_d = M.to_color_scheme(M.make_scheme(0xFF0000FF, "monochrome", True, 0.0))
    assert _hexes(mono_l, "primary", "onPrimaryContainer") == ["#000000", "#FFFFFF"]
    assert _hexes(mono_d, "primary", "primaryContainer", "onPrimaryContainer") == ["#FFFFFF", "#D4D4D4", "#000000"]
    fid = M.to_color_scheme(M.make_scheme(0xFF0000FF, "fidelity", False, 0.0))
    assert M.argb_to_hex(fid["primaryContainer"]) == "#0000FF"  # container keeps the source tone


@pytest.mark.parametrize("variant", M.VARIANTS)
@pytest.mark.parametrize("dark", [False, True])
def test_scheme_contrast_requirements(variant, dark):
    cs = M.to_color_scheme(M.make_scheme(0xFF6750A4, variant, dark, 0.0))
    for fg, bg, need in (("onPrimary", "primary", 4.5), ("onSurface", "surface", 4.5),
                         ("onSecondaryContainer", "secondaryContainer", 3.0), ("onError", "error", 4.5)):
        ratio = M.ratio_of_tones(M.Hct(cs[fg]).tone, M.Hct(cs[bg]).tone)
        assert ratio >= need - 0.05, (fg, bg, ratio)
    hi = M.to_color_scheme(M.make_scheme(0xFF6750A4, variant, dark, 1.0))
    assert M.ratio_of_tones(M.Hct(hi["onSurface"]).tone, M.Hct(hi["surface"]).tone) >= 11


def test_itb_color_scheme_rules():
    seed = 0xFFBCE870
    light = M.itb_color_scheme(seed, "tonal_spot", False)
    dark = M.itb_color_scheme(seed, "tonal_spot", True)
    assert len(light) == len(M.COMPOSE_ROLES) and light["background"] == light["surface"]
    # TonalSpot keeps the seed's chroma for the primary palette (TonalPalette.fromInt)
    assert abs(M.Hct(light["primary"]).chroma - M.Hct(M.TonalPalette.from_int(seed).tone(40)).chroma) < 1
    # withContrastingSurfaces (light only): surface is tone 97
    assert abs(M.Hct(light["surface"]).tone - 97) < 0.6
    # expressive on-colours reach 4.5:1 (androidx ColorUtils contrast)
    for role in ("primary", "secondary", "tertiary"):
        for cs in (light, dark):
            assert M.calculate_contrast(cs["on" + role.capitalize()], cs[role]) >= 4.5
    inv = M.itb_color_scheme(seed, "tonal_spot", False, invert=True)
    assert inv["primary"] == light["primary"] ^ 0xFFFFFF and inv["scrim"] == light["scrim"]
    amo = M.itb_color_scheme(seed, "tonal_spot", True, amoled=True)
    assert amo["surface"] == 0xFF000000 and amo["background"] == 0xFF000000
    gray = M.itb_color_scheme(seed, "tonal_spot", False, color_blind="achromatopsia")
    p = gray["primary"]
    assert (p >> 16) & 255 == (p >> 8) & 255 == p & 255


def test_compose_code_format():
    code = M.as_code_string(M.itb_color_scheme(0xFF6750A4), False)
    lines = code.splitlines()
    assert lines[0] == "val lightColorScheme = ColorScheme(" and lines[-1] == ")"
    assert len(lines) == len(M.CODE_ROLES) + 2 and lines[1].startswith("    background = Color(0xff")


# --- androidx palette ------------------------------------------------------------

def test_androidx_palette_two_colours():
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :120] = (40, 90, 200)
    img[:, 120:] = (230, 200, 20)
    pal = AP.generate(img)
    assert [(s.hex, s.population) for s in pal.swatches] == [("#2858C8", 7600), ("#E0C810", 5120)]  # 5-bit approx.
    assert pal.dominant.hex == "#2858C8" and pal.selected["vibrant"].hex == "#2858C8"
    assert AP.extract_primary_color(img) == 0xFF2858C8


def test_androidx_palette_filters_black_white_and_scales():
    img = np.zeros((400, 300, 3), np.uint8)
    img[:200] = 255
    img[200:300] = (20, 160, 60)
    img[300:350] = (224, 136, 24)  # skin-tone hue band, S <= 0.82: DEFAULT_FILTER "near red I-line"
    pal = AP.generate(img)
    assert [s.hex for s in pal.swatches] == ["#10A038"]
    small = AP.scale_bitmap_down(img)
    assert small.shape[0] * small.shape[1] <= 113 * 113 + 300


def test_palette_quantizes_many_colours_to_max():
    y, x = np.mgrid[0:240, 0:320]
    img = (np.dstack([x / 320, y / 240, 0.5 + 0.5 * np.sin(x / 30)]) * 255).astype(np.uint8)
    pal = AP.generate(img, maximum_color_count=12)
    assert 1 <= len(pal.swatches) <= 12
    data = AP.named_palette_data(pal)
    assert abs(sum(d["percent"] for d in data) - 1) < 0.3 and len({d["name"] for d in data}) == len(data)


# --- tools ---------------------------------------------------------------------

@pytest.fixture
def two_tone_path(tmp_path):
    img = np.zeros((80, 120, 3), np.float32)
    img[:, :70] = (0.15, 0.35, 0.8)
    img[:, 70:] = (0.95, 0.85, 0.1)
    p = tmp_path / "two.png"
    save_image(img, p)
    return p


def test_material_you_scheme_tool(api, two_tone_path, tmp_path):
    res = api.tool_run("material_you_scheme", {"input": str(two_tone_path), "swatch_output": str(tmp_path / "s.png")})
    assert res["key_color_source"] == "imagetoolbox" and res["key_color"].startswith("#")
    assert set(res["schemes"]) == {"light", "dark"} and len(res["schemes"]["light"]) == len(M.COMPOSE_ROLES)
    assert res["seeds"] and res["seeds"][0]["population"] > 0
    assert set(res["tonal_palettes"]) == {"primary", "secondary", "tertiary", "neutral", "neutralVariant", "error"}
    assert "val darkColorScheme" in res["compose_code"] and res["swatch"]["width"] > 0
    assert (tmp_path / "s.png").exists()
    seeded = api.tool_run("material_you_scheme", {"seed": "#0000FF", "style": "monochrome", "theme": "dark"})
    assert seeded["key_color"] == "#0000FF" and list(seeded["schemes"]) == ["dark"]
    assert seeded["schemes"]["dark"]["primary"] == "#FFFFFF"
    mcu = api.tool_run("material_you_scheme", {"input": str(two_tone_path), "key_color_source": "mcu_score",
                                               "theme": "light"})
    assert mcu["key_color"] == mcu["seeds"][0]["hex"]


def test_palette_simple_tool(api, two_tone_path, tmp_path):
    res = api.tool_run("palette_simple", {"input": str(two_tone_path), "count": 8, "include_targets": True,
                                          "swatch_output": str(tmp_path / "p.png")})
    hexes = {c["hex"] for c in res["colors"]}
    assert hexes == {"#2058C8", "#F0D818"}  # 5-bit quantised (0.15,0.35,0.8) and (0.95,0.85,0.1) and all(c["name"] for c in res["colors"])
    assert abs(sum(c["percent"] for c in res["colors"]) - 1) < 1e-6
    assert res["dominant"]["hex"] == "#2058C8" and set(res["targets"]) >= {"vibrant", "muted"}
    assert (tmp_path / "p.png").exists()


# =============================================================================
# adjust.color_curves + curves preset codec
# =============================================================================

def _run(img, **params):
    spec = get_op("adjust.color_curves")
    return spec.run(img, RenderContext(), spec.validate(params))


def _solid(rgb255, n=4):
    return np.full((n, n, 3), np.array(rgb255, np.float32) / 255.0, np.float32)


def _render(name, points, rgb255):
    return _run(_solid(rgb255), **{name: points})[0, 0]


def _flat(name, value, rgb255):
    pts = [[0, value], [0.5, value], [1, value]] if name in ce.CENTERED else [[0, value], [1, value]]
    return _render(name, pts, rgb255)


def _hsv(rgb):
    return colorsys.rgb_to_hsv(*[float(v) for v in rgb])


def _hue_dist(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _luma(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


# ---------------------------------------------------------------------------
# adjust.color_curves
# ---------------------------------------------------------------------------

def test_identity_is_exact_noop():
    img = synthetic_image(64, 48)
    assert np.array_equal(_run(img), img)
    defaults = ce.default_params()
    for name in ce.CURVE_NAMES:  # explicitly passing each default (and a collinear identity) changes nothing
        assert np.array_equal(_run(img, **{name: defaults[name]}), img), name
    assert np.array_equal(_run(img, rgb=[[0, 0], [0.5, 0.5], [1, 1]], hue_vs_sat=[[0.3, 0.5], [0.6, 0.5]]), img)
    rgba = synthetic_image(32, 24, alpha=True)
    out = _run(rgba, rgb=[[0, 0], [0.5, 0.7], [1, 1]])
    assert out.shape == rgba.shape and np.array_equal(out[..., 3], rgba[..., 3])


def test_rgb_master_curve_brightens_monotonically():
    ramp = np.repeat(np.linspace(0, 1, 256, dtype=np.float32)[None, :, None], 3, 2)
    out = _run(ramp, rgb=[[0, 0], [0.5, 0.7], [1, 1]])
    assert out.mean() > ramp.mean() + 0.1
    assert np.all(np.diff(out[0, :, 0]) >= -1e-6)
    assert out[0, 0, 0] == pytest.approx(0, abs=1e-6) and out[0, -1, 0] == pytest.approx(1, abs=1e-6)
    # GPUImage spline passes through the integer control point (127 -> 178)
    assert ce.gpuimage_spline_offsets([[0, 0], [0.5, 0.7], [1, 1]])[127] == 178 - 127
    assert not ce.gpuimage_spline_offsets([[0, 0], [1, 1]]).any()


def test_per_channel_curve_only_touches_its_channel():
    img = synthetic_image(64, 48)
    out = _run(img, red=[[0, 0], [0.5, 0.75], [1, 1]])
    assert (out[..., 0] - img[..., 0]).mean() > 0.05
    # identity G/B LUTs are only resampled at OpenGL texel centres (< 0.5/255 shift)
    assert np.abs(out[..., 1:] - img[..., 1:]).max() < 0.5 / 255 + 1e-5


def test_cmyk_and_lab_curves():
    grey = _solid((128, 128, 128))
    more_cyan = _run(grey, cyan=[[0, 0.2], [1, 1]])[0, 0]
    assert more_cyan[0] < 0.45 and abs(more_cyan[1] - 128 / 255) < 0.01 and abs(more_cyan[2] - 128 / 255) < 0.01
    brighter = _run(grey, lab_l=[[0, 0], [0.5, 0.7], [1, 1]])[0, 0]
    assert brighter.mean() > 0.6 and np.ptp(brighter) < 0.01  # L only: stays neutral
    warmer = _run(grey, lab_b=[[0, 0], [0.5, 0.6], [1, 1]])[0, 0]
    assert warmer[0] > warmer[2]  # +b = towards yellow


def test_relation_curves_match_imagetoolbox_instrumented_tests():
    source = (204, 102, 68)
    s_hsv = _hsv(np.array(source) / 255)
    for name in ("hue_vs_sat", "luma_vs_sat", "sat_vs_sat"):
        r = _hsv(_flat(name, 0.25, source))
        assert r[1] < s_hsv[1] and _hue_dist(r[0] * 360, s_hsv[0] * 360) < 2
        boosted = _hsv(_flat(name, 1.0, (180, 150, 140)))[1]
        assert _hsv(np.array((180, 150, 140)) / 255)[1] < boosted < 0.85
    lighter = _flat("hue_vs_luma", 0.6, source)
    assert _luma(lighter) > _luma(np.array(source) / 255)
    assert _hue_dist(_hsv(lighter)[0] * 360, s_hsv[0] * 360) < 6
    assert _hsv(_flat("hue_vs_luma", 0.0, source))[2] < 0.02
    grey = _flat("hue_vs_luma", 0.0, (160, 160, 160)) * 255
    assert np.abs(grey - 160).max() <= 2
    assert _luma(_flat("hue_vs_luma", 0.3, (230, 20, 10))) < 0.02
    dark = np.array((16, 24, 28)) / 255
    assert abs(_luma(_flat("hue_vs_luma", 1.0, (16, 24, 28))) - _luma(dark)) < 0.03
    assert 85 <= _hsv(_flat("hue_vs_hue", 0.75, (255, 0, 0)))[0] * 360 <= 95


def test_reference_style_relation_curves():
    bright_red, dark_red = (230, 20, 10), (64, 6, 3)
    desc = [[0, 0.75], [0.35, 0.75], [0.65, 0], [1, 0]]
    asc = [[0, 0.25], [0.35, 0.25], [0.65, 0.75], [1, 0.75]]
    c = ce.CurvesValue(0.5)
    c.replace_points(desc)
    assert not c.is_default and c.to_lut()[230] < 0.05
    r = _hsv(_render("luma_vs_sat", desc, bright_red))
    assert r[1] < 0.05 and r[2] > 0.85
    assert 60 <= _hsv(_render("luma_vs_hue", asc, bright_red))[0] * 360 <= 120
    assert 240 <= _hsv(_render("luma_vs_hue", asc, dark_red))[0] * 360 <= 320
    assert 0.25 <= _hsv(_flat("hue_vs_sat", 1.0, (160, 160, 160)))[1] <= 0.4
    assert 0.25 <= _hsv(_render("sat_vs_sat", [[0, 0.8], [0.3, 0.8], [0.75, 0.2], [1, 0.2]], bright_red))[1] <= 0.5


def test_curve_value_interpolation_rules():
    c = ce.CurvesValue(0.5)
    c.replace_points([[0, 0.7], [0, 0.7], [1, 0.7]])  # centred curves keep a control point at x=0
    assert len(c.copy().points) == 3 and c.copy().points[1][0] == 0
    hue = ce.CurvesValue(0.5, circular=True)
    hue.replace_points([[0, 0.5], [0.3, 0.1], [0.65, 0.9], [1, 0.5]])
    lut, it = hue.to_lut(), hue.interpolate()
    assert abs(lut[0] - lut[-1]) < 0.01 and np.all(np.diff(it[:, 0]) > 0) and it[:, 1].min() >= 0
    lin = ce.CurvesValue(0.5)
    lin.replace_points([[0, 0.5], [0.35, 0.9], [0.65, 0.1], [1, 0.5]])
    it, lut = lin.interpolate(), lin.to_lut()
    assert it[0, 1] == pytest.approx(0.9, abs=1e-5) and it[-1, 1] == pytest.approx(0.1, abs=1e-5)
    assert lut[int(0.3 * 255)] == pytest.approx(0.9, abs=1e-5) and lut[int(0.7 * 255)] == pytest.approx(0.1, abs=1e-5)
    many = ce.CurvesValue()
    many.replace_points([[i / 30, i / 30] for i in range(31)])
    assert len(many.points) == 16 and many.points[-1] == [1.0, 1.0]  # MaxPointCount


def test_hue_vs_hue_rotates_hue():
    out = _run(_solid((255, 0, 0)), hue_vs_hue=[[0.5, 0.5 + 1 / 6]])  # flat +1/6 everywhere (single point)
    assert _hue_dist(_hsv(out[0, 0])[0] * 360, 60) < 2


def test_color_wheels_match_imagetoolbox_tests():
    tri = np.concatenate([_solid(c, 8) for c in ((32, 32, 32), (128, 128, 128), (224, 224, 224))], 1)
    out = _run(tri, wheel_shadows=[0, -0.7], wheel_edges=1.0)
    d_sh = np.abs(out[0, 2] - tri[0, 2]).sum()
    d_hi = np.abs(out[0, 20] - tri[0, 20]).sum()
    assert d_sh > d_hi and abs(out[0, 2, 0] * 255 - 32) >= 1
    pts = {"wheel_shadows": [0, -0.85], "wheel_midtones": [0, 0.85], "wheel_highlights": [0.85, 0]}
    src = (56, 56, 56)
    blended = _run(_solid(src), **pts, wheel_edges=0.0)[0, 0]
    isolated = _run(_solid(src), **pts, wheel_edges=1.0)[0, 0]
    bias = lambda c: c[0] - (c[1] + c[2]) / 2  # noqa: E731
    assert bias(isolated) > bias(blended)
    assert abs(_luma(isolated) - 56 / 255) < 0.08
    cyan = _run(_solid((128, 128, 128)), wheel_midtones=[0, 0.8], wheel_edges=1.0)[0, 0]
    assert cyan[1] > cyan[0] and cyan[2] > cyan[0]


def test_numba_kernel_matches_numpy_reference():
    img = synthetic_image(96, 64, seed=3)
    for p in ({"rgb": [[0, 0], [0.3, 0.2], [1, 1]], "blue": [[0, 0.1], [1, 0.9]]},
              {"magenta": [[0, 0], [0.5, 0.6], [1, 1]], "black": [[0, 0.1], [1, 1]]},
              {"lab_l": [[0, 0.1], [1, 1]], "lab_a": [[0, 0], [0.5, 0.55], [1, 1]]},
              {"hue_vs_sat": [[0.3, 0.8]], "hue_vs_hue": [[0.2, 0.3], [0.6, 0.7]], "hue_vs_luma": [[0.5, 0.7]],
               "luma_vs_sat": [[0.35, 0.9], [0.65, 0.1]], "luma_vs_hue": [[0.5, 0.6]], "sat_vs_sat": [[0.5, 0.3]]},
              {"wheel_shadows": [0.3, 0.2], "wheel_highlights": [-0.5, 0.1], "wheel_edges": 0.9}):
        st = ce.state_from_params(p)
        assert np.abs(ce.apply_state(img, st) - ce.apply_state_numpy(img, st)).max() < 1e-4, p


def test_control_points_legacy_and_wheel_normalisation():
    cps = [[0.0, 0.2, 0.5, 0.8, 1.0]]  # legacy five-value Y-only curve
    st = ce.CurvesState.from_control_points(cps)
    assert st.curves["rgb"].points == [[0.0, 0.0], [0.25, 0.20000000298023224], [0.5, 0.5],
                                       [0.75, 0.800000011920929], [1.0, 1.0]]
    assert st.edges == 1.0  # ImageToolbox falls back to edges = 1 when the entry is missing
    st = ce.state_from_params({"wheel_shadows": [1, 1]})
    assert math.hypot(*st.wheels["shadows"]) == pytest.approx(1, abs=1e-6)


# ---------------------------------------------------------------------------
# CurvesPresetCodec
# ---------------------------------------------------------------------------

def test_codec_roundtrips_all_curves_and_wheels():
    pts = ce.CurvesState().control_points()
    pts[0] = [0, 0.1, 0.5, 0.8, 1, 0.9]
    pts[11] = [0, 0.5, 0.3, 0.75, 1, 0.5]
    pts[17] = [0.2, -0.1]
    pts[20] = [0.7]
    original = ce.CurvesState.from_control_points(pts)
    text = cp.encode(original)
    assert cp.decode(text).control_points() == original.control_points()
    assert text.startswith('{\n  "format": "ImageToolbox Curves",\n  "version": 1,\n  "curves": [\n')
    assert '    [[0.0, 0.1], [0.5, 0.8], [1.0, 0.9]],\n' in text
    assert text.endswith('  "colorWheels": [[0.2, -0.1], [0.0, 0.0], [0.0, 0.0], [0.7]]\n}\n')
    assert cp.encode(ce.CurvesState()).count("[[0.0, 0.5], [1.0, 0.5]]") == 6


def test_java_float_formatting():
    assert [cp.java_float_str(v) for v in (0, 1, 0.5, 0.1, 1 / 3, -0.25, 1e-4, 0.001)] == \
        ["0.0", "1.0", "0.5", "0.1", "0.33333334", "-0.25", "1.0E-4", "0.001"]


def test_codec_imports_lightroom_and_acv():
    xmp = """<crs:ToneCurvePV2012 xmlns:crs="crs" xmlns:rdf="rdf">
  <rdf:Seq><rdf:li>0, 10</rdf:li><rdf:li>128, 200</rdf:li><rdf:li>255, 245</rdf:li></rdf:Seq>
</crs:ToneCurvePV2012>
<crs:ToneCurvePV2012Red xmlns:crs="crs" xmlns:rdf="rdf">
  <rdf:Seq><rdf:li>0, 0</rdf:li><rdf:li>255, 220</rdf:li></rdf:Seq>
</crs:ToneCurvePV2012Red>"""
    dec = cp.decode(xmp).control_points()
    f = lambda v: float(np.float32(np.float32(v) / np.float32(255)))  # noqa: E731
    assert dec[0][3] == f(200) and dec[1][3] == f(220)
    assert dec[2] == ce.CurvesState().control_points()[2]
    tmpl = "s = {\n  ToneCurvePV2012 = { 0, 0, 64, 80, 255, 255, },\n}"
    d2 = cp.decode(tmpl).control_points()
    assert d2[0][2] == f(64) and d2[0][3] == f(80)
    pts = ce.CurvesState().control_points()
    pts[2] = [0, 0, 0.5, 0.75, 1, 1]
    assert cp.decode(cp.encode_xmp(ce.CurvesState.from_control_points(pts))).control_points()[2][3] == f(191)
    acv = bytes([0, 4, 0, 1, 0, 3, 0, 0, 0, 0, 0, 200, 0, 128, 0, 255, 0, 255])
    d3 = cp.decode(acv).control_points()
    assert d3[0][2] == f(128) and d3[0][3] == f(200)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Preset Pack/Seashell.xmp", xmp.replace("128, 200", "128, 190"))
        z.writestr("__MACOSX/._Seashell.xmp", "junk")
    st, fmt = cp.decode_with_format(buf.getvalue())
    assert st.control_points()[0][3] == f(190) and fmt == "zip:xmp"


@pytest.mark.parametrize("data,msg", [
    (b"", "empty"), ("   ", "empty"), ('{"format": "Other"}', "Unsupported curve preset format"),
    ('{"format": "ImageToolbox Curves", "curves": [[0, 0, 1, 1]], "colorWheels": [0,0,0,0,0,0,0.5]}', "incomplete"),
    (bytes([0, 2, 0, 1]), "Unsupported ACV version"), (bytes([0, 4, 0, 1, 0, 3, 0]), "truncated"),
    ("<x>nothing</x>", "No supported tone curves"),
])
def test_codec_errors(data, msg):
    with pytest.raises(cp.CurvesPresetError, match=msg):
        cp.decode(data)


def test_preset_tools_roundtrip_and_apply(api, tmp_path):
    params = {"rgb": [[0, 0], [0.5, 0.65], [1, 1]], "hue_vs_sat": [[0, 0.5], [0.33, 0.8], [1, 0.5]],
              "wheel_shadows": [0.1, -0.2]}
    out = api.tool_run("curves_preset_export", {**params, "output": str(tmp_path / "look.itcurve")})
    assert out["bytes"] == (tmp_path / "look.itcurve").stat().st_size and out["format"] == "itcurve"
    assert set(out["active_editor_types"]) == {"RGB", "HueVsSat", "ColorWheels"}
    imp = api.tool_run("curves_preset_import", {"input": str(tmp_path / "look.itcurve")})
    assert imp["detected_format"] == "itcurve" and imp["active_curves"] == ["rgb", "hue_vs_sat"]
    img = synthetic_image(48, 32)
    direct = _run(img, **params)
    assert np.allclose(_run(img, **imp["params"]), direct, atol=1e-6)
    assert np.allclose(_run(img, preset_file=str(tmp_path / "look.itcurve")), direct, atol=1e-6)
    assert np.allclose(_run(img, preset=out["text"]), direct, atol=1e-6)
    # preset + explicit override: the explicit non-default curve wins
    over = _run(img, preset=out["text"], rgb=[[0, 0], [0.5, 0.3], [1, 1]])
    assert over.mean() < direct.mean()
    # ACV -> itcurve conversion through `source`
    (tmp_path / "c.acv").write_bytes(bytes([0, 4, 0, 1, 0, 3, 0, 0, 0, 0, 0, 200, 0, 128, 0, 255, 0, 255]))
    conv = api.tool_run("curves_preset_export", {"source": str(tmp_path / "c.acv"), "format": "xmp"})
    assert "<rdf:li>128, 200</rdf:li>" in conv["text"] and "output" not in conv
    with pytest.raises(LazyRoomError):
        api.tool_run("curves_preset_export", {**params, "output": str(tmp_path / "look.itcurve")})
    (tmp_path / "bad.itcurve").write_text("{}")
    with pytest.raises(LazyRoomError):
        api.tool_run("curves_preset_import", {"input": str(tmp_path / "bad.itcurve")})
