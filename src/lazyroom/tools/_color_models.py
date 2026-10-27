"""Colour models and colour-tool maths of ImageToolbox (feature/color-tools, feature/pick-color,
core/ui ColorCopyFormatSelectionDialog, lib/colors), ported with Android's numeric behaviour.

* HSV: Skia ``SkRGBToHSV`` / ``SkHSVToColor`` (what ``android.graphics.Color.colorToHSV`` /
  ``HSVToColor`` call), float32 maths, ``floor(x + 0.5)`` rounding.
* HSL, XYZ, LAB, blending: androidx ``ColorUtils`` (``RGBToHSL``/``HSLToColor``,
  ``RGBToXYZ``/``XYZToLAB``/``LABToXYZ``/``XYZToColor`` with D65/2° white 95.047/100/108.883 and the
  0.4124/0.3576/... matrices, ``blendARGB`` with float32 maths and ``(int)`` truncation).
* CMYK: ColorInfoDisplay ``toCMYK``/``cmykToColor`` and the copy dialog ``toCmykString``.
* Harmonies (7 types), mixing (``mixWith``) and shading (tints / tones / shades) from
  ColorHarmoniesUtils.kt, ColorMixingUtils.kt and ColorShading.kt.
* Colour names: lib/colors ``ColorNameParser`` over ``data/color_library.json``.

Colours are passed around as 8-bit ``(a, r, g, b)`` tuples or packed ARGB ints like on Android.
"""
from __future__ import annotations

import functools
import json
import math
from typing import Any

import numpy as np

f32 = np.float32

HARMONY_TYPES = ["complementary", "analogous", "analogous_complementary", "triadic", "split_complementary",
                 "tetradic", "square"]  # HarmonyType.entries order
HARMONY_OFFSETS = {  # ColorHarmoniesUtils.applyHarmony: hue offsets in HSV (first entry = the colour itself)
    "complementary": [0, 180],
    "analogous": [0, 30, -30],
    "analogous_complementary": [0, 30, -30, 180],
    "triadic": [0, 120, -120],
    "split_complementary": [0, 150, -150],
    "tetradic": [0, 30, 180, 210],
    "square": [0, 90, 180, 270],
}
TONE_GRAY = (0xFF, 0x8E, 0x91, 0x8F)  # ColorShading: tones mix with Color(0xff8e918f)


# ---------------------------------------------------------------------------
# packing helpers
# ---------------------------------------------------------------------------

def pack(a: int, r: int, g: int, b: int) -> int:
    return ((a & 255) << 24) | ((r & 255) << 16) | ((g & 255) << 8) | (b & 255)


def unpack(argb: int) -> tuple[int, int, int, int]:
    return (argb >> 24) & 255, (argb >> 16) & 255, (argb >> 8) & 255, argb & 255


def from_float(c) -> tuple[int, int, int, int]:
    """LazyRoom (r, g, b, a) floats -> 8-bit (a, r, g, b) like Compose ``toArgb`` ((v*255+0.5).toInt())."""
    r, g, b = (int(f32(f32(v) * f32(255.0) + f32(0.5))) for v in c[:3])
    a = int(f32(f32(c[3]) * f32(255.0) + f32(0.5))) if len(c) > 3 else 255
    return a, r, g, b


def to_float(argb8) -> tuple[float, float, float, float]:
    a, r, g, b = argb8
    return r / 255.0, g / 255.0, b / 255.0, a / 255.0


def jround(x: float) -> int:
    """Java/Kotlin ``Math.round`` / ``roundToInt`` / Skia ``SkScalarRoundToInt``: floor(x + 0.5)."""
    return int(math.floor(float(x) + 0.5))


def hex_rgb(argb8) -> str:
    """ColorInfoDisplay ``Color.toHex()`` / ``ColorUtil.colorToHex``: #RRGGBB."""
    _, r, g, b = argb8
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_itb(argb8) -> str:
    """``Color.hex()``: #RRGGBB when opaque, else Android order #AARRGGBB (uppercase)."""
    a, r, g, b = argb8
    return hex_rgb(argb8) if a == 255 else f"#{a:02X}{r:02X}{g:02X}{b:02X}"


def hex_css(argb8) -> str:
    """LazyRoom / CSS convention: #RRGGBB or #RRGGBBAA."""
    a, r, g, b = argb8
    return hex_rgb(argb8) if a == 255 else f"#{r:02X}{g:02X}{b:02X}{a:02X}"


def parse_hex(s: str, alpha_first: bool = True) -> tuple[int, int, int, int]:
    """``HexUtil.hexToColor`` (Android ``parseColor``): #RGB? no - #RRGGBB or #AARRGGBB.

    ``alpha_first=False`` reads 8-digit values as #RRGGBBAA (CSS / LazyRoom) instead.
    """
    h = s.strip().lstrip("#")
    if len(h) == 3:  # convenience, not accepted by Android
        h = "".join(c * 2 for c in h)
    if len(h) not in (6, 8) or any(c not in "0123456789abcdefABCDEF" for c in h):
        raise ValueError(f"Invalid hex colour {s!r}")
    v = [int(h[i:i + 2], 16) for i in range(0, len(h), 2)]
    if len(v) == 3:
        return 255, v[0], v[1], v[2]
    if alpha_first:
        return v[0], v[1], v[2], v[3]
    return v[3], v[0], v[1], v[2]


# ---------------------------------------------------------------------------
# HSV (Skia) and HSL (androidx ColorUtils)
# ---------------------------------------------------------------------------

def rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    """``Color.colorToHSV`` -> SkRGBToHSV (float32)."""
    mn, mx = min(r, g, b), max(r, g, b)
    delta = mx - mn
    v = f32(mx) / f32(255.0)
    if delta == 0:
        return 0.0, 0.0, float(v)
    s = f32(delta) / f32(mx)
    if r == mx:
        h = f32(g - b) / f32(delta)
    elif g == mx:
        h = f32(2) + f32(b - r) / f32(delta)
    else:
        h = f32(4) + f32(r - g) / f32(delta)
    h = f32(h * f32(60))
    if h < 0:
        h = f32(h + f32(360))
    return float(h), float(s), float(v)


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """``Color.HSVToColor`` -> SkHSVToColor."""
    s = f32(min(1.0, max(0.0, s)))
    v = f32(min(1.0, max(0.0, v)))
    v_byte = jround(f32(v * f32(255)))
    if abs(float(s)) <= 1.0 / 4096:
        return v_byte, v_byte, v_byte
    hx = f32(0) if (h < 0 or h >= 360) else f32(f32(h) / f32(60))
    w = f32(math.floor(hx))
    f = f32(hx - w)
    one = f32(1)
    p = jround(f32(f32(one - s) * v * f32(255)))
    q = jround(f32(f32(one - f32(s * f)) * v * f32(255)))
    t = jround(f32(f32(one - f32(s * f32(one - f))) * v * f32(255)))
    return {0: (v_byte, t, p), 1: (q, v_byte, p), 2: (p, v_byte, t), 3: (p, q, v_byte),
            4: (t, p, v_byte)}.get(int(w), (v_byte, p, q))


def rgb_to_hsl(r: int, g: int, b: int) -> tuple[float, float, float]:
    """androidx ``ColorUtils.RGBToHSL`` (float32)."""
    rf, gf, bf = f32(r) / f32(255), f32(g) / f32(255), f32(b) / f32(255)
    mx, mn = max(rf, gf, bf), min(rf, gf, bf)
    d = f32(mx - mn)
    l = f32(f32(mx + mn) / f32(2))
    if mx == mn:
        h = s = f32(0)
    else:
        if mx == rf:
            h = f32(math.fmod(float(f32(gf - bf) / d), 6.0))
        elif mx == gf:
            h = f32(f32(bf - rf) / d + f32(2))
        else:
            h = f32(f32(rf - gf) / d + f32(4))
        s = f32(d / f32(f32(1) - f32(abs(f32(f32(2) * l) - f32(1)))))
    h = f32(math.fmod(float(f32(h * f32(60))), 360.0))
    if h < 0:
        h = f32(h + f32(360))
    return float(min(360.0, max(0.0, h))), float(min(1.0, max(0.0, s))), float(min(1.0, max(0.0, l)))


def hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    """androidx ``ColorUtils.HSLToColor``."""
    h, s, l = f32(h), f32(s), f32(l)
    c = f32(f32(f32(1) - f32(abs(f32(f32(2) * l) - f32(1)))) * s)
    m = f32(l - f32(f32(0.5) * c))
    x = f32(c * f32(f32(1) - f32(abs(f32(math.fmod(float(f32(h / f32(60))), 2.0)) - f32(1)))))
    seg = int(h) // 60
    R = lambda v: jround(f32(f32(255) * f32(v)))  # noqa: E731
    if seg == 0:
        rgb = (R(c + m), R(x + m), R(m))
    elif seg == 1:
        rgb = (R(x + m), R(c + m), R(m))
    elif seg == 2:
        rgb = (R(m), R(c + m), R(x + m))
    elif seg == 3:
        rgb = (R(m), R(x + m), R(c + m))
    elif seg == 4:
        rgb = (R(x + m), R(m), R(c + m))
    elif seg in (5, 6):
        rgb = (R(c + m), R(m), R(x + m))
    else:
        rgb = (0, 0, 0)
    return tuple(min(255, max(0, v)) for v in rgb)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# XYZ / LAB (androidx ColorUtils)
# ---------------------------------------------------------------------------
XYZ_WHITE = (95.047, 100.0, 108.883)
XYZ_EPSILON = 0.008856
XYZ_KAPPA = 903.3


def rgb_to_xyz(r: int, g: int, b: int) -> tuple[float, float, float]:
    def lin(c):
        c /= 255.0
        return c / 12.92 if c < 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    sr, sg, sb = lin(r), lin(g), lin(b)
    return (100 * (sr * 0.4124 + sg * 0.3576 + sb * 0.1805),
            100 * (sr * 0.2126 + sg * 0.7152 + sb * 0.0722),
            100 * (sr * 0.0193 + sg * 0.1192 + sb * 0.9505))


def xyz_to_lab(x: float, y: float, z: float) -> tuple[float, float, float]:
    def piv(c):
        return c ** (1 / 3.0) if c > XYZ_EPSILON else (XYZ_KAPPA * c + 16) / 116

    fx, fy, fz = piv(x / XYZ_WHITE[0]), piv(y / XYZ_WHITE[1]), piv(z / XYZ_WHITE[2])
    return max(0.0, 116 * fy - 16), 500 * (fx - fy), 200 * (fy - fz)


def lab_to_xyz(l: float, a: float, b: float) -> tuple[float, float, float]:
    fy = (l + 16) / 116
    fx, fz = a / 500 + fy, fy - b / 200
    xr = fx ** 3 if fx ** 3 > XYZ_EPSILON else (116 * fx - 16) / XYZ_KAPPA
    yr = fy ** 3 if l > XYZ_KAPPA * XYZ_EPSILON else l / XYZ_KAPPA
    zr = fz ** 3 if fz ** 3 > XYZ_EPSILON else (116 * fz - 16) / XYZ_KAPPA
    return xr * XYZ_WHITE[0], yr * XYZ_WHITE[1], zr * XYZ_WHITE[2]


def xyz_to_rgb(x: float, y: float, z: float) -> tuple[int, int, int]:
    rgb = ((x * 3.2406 + y * -1.5372 + z * -0.4986) / 100,
           (x * -0.9689 + y * 1.8758 + z * 0.0415) / 100,
           (x * 0.0557 + y * -0.2040 + z * 1.0570) / 100)
    out = []
    for c in rgb:
        c = 1.055 * c ** (1 / 2.4) - 0.055 if c > 0.0031308 else 12.92 * c
        out.append(min(255, max(0, jround(c * 255))))
    return tuple(out)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# CMYK (ColorInfoDisplay.kt / ColorCopyFormatSelectionDialog.kt)
# ---------------------------------------------------------------------------

def rgb_to_cmyk(r: int, g: int, b: int) -> tuple[float, float, float, float]:
    rf, gf, bf = f32(r) / f32(255), f32(g) / f32(255), f32(b) / f32(255)
    k = f32(f32(1) - max(rf, gf, bf))
    den = f32(f32(1) - k)

    def comp(v):
        if den == 0:
            return f32(0)
        return f32(f32(f32(1) - v - k) / den)

    return float(comp(rf)), float(comp(gf)), float(comp(bf)), float(k)


def cmyk_to_rgb(c: float, m: float, y: float, k: float) -> tuple[int, int, int]:
    """``cmykToColor``: 255*(1-c)*(1-k) -> Compose Color(r/255f) -> toArgb ((v*255+0.5).toInt())."""
    out = []
    for v in (c, m, y):
        x = f32(f32(f32(255) * f32(f32(1) - f32(v))) * f32(f32(1) - f32(k)))
        out.append(min(255, max(0, int(f32(f32(x / f32(255)) * f32(255) + f32(0.5))))))
    return tuple(out)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# blending / harmonies / mixing / shading
# ---------------------------------------------------------------------------

def blend_argb(c1, c2, ratio: float) -> tuple[int, int, int, int]:
    """androidx ``ColorUtils.blendARGB`` (float32, ``(int)`` truncation)."""
    ratio = f32(ratio)
    inv = f32(f32(1) - ratio)
    return tuple(int(f32(f32(f32(x) * inv) + f32(f32(y) * ratio))) for x, y in zip(c1, c2))  # type: ignore


def harmony(argb8, kind: str) -> list[tuple[int, int, int, int]]:
    """``Color.applyHarmony(type)``: HSV hue rotations, alpha kept."""
    a, r, g, b = argb8
    h, s, v = rgb_to_hsv(r, g, b)
    out = []
    for off in HARMONY_OFFSETS[kind]:
        hh = float(f32(math.fmod(float(f32(f32(h) + f32(off) + (f32(360) if off < 0 else f32(0)))), 360.0)))
        out.append((a, *hsv_to_rgb(hh, s, v)))
    return out


def mix_with(argb8, other8, variations: int, max_percent: float = 1.0) -> list[tuple[int, int, int, int]]:
    """``Color.mixWith(color, variations, maxPercent)``."""
    den = f32(f32(f32(variations) + f32(f32(f32(1) - f32(max_percent)) * f32(10))) - f32(1))
    return [blend_argb(argb8, other8, float(f32(f32(i) / den))) for i in range(variations)]


def shading(argb8, variations: int) -> dict[str, list[tuple[int, int, int, int]]]:
    """ColorShading: tints (white, 0.8), tones (0xff8e918f, 0.9), shades (black, 0.9)."""
    return {"tints": mix_with(argb8, (255, 255, 255, 255), variations, 0.8),
            "tones": mix_with(argb8, TONE_GRAY, variations, 0.9),
            "shades": mix_with(argb8, (255, 0, 0, 0), variations, 0.9)}


def relative_luminance(argb8) -> float:
    """Compose ``Color.luminance()`` (sRGB EOTF, Rec.709 weights)."""
    _, r, g, b = argb8

    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast_ratio(c1, c2) -> float:
    l1, l2 = relative_luminance(c1) + 0.05, relative_luminance(c2) + 0.05
    return max(l1, l2) / min(l1, l2)


# ---------------------------------------------------------------------------
# colour names (lib/colors ColorNameParser over data/color_library.json)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _names() -> tuple[list[str], list[str], np.ndarray]:
    from ..core.paths import package_data_dir

    data = json.loads((package_data_dir() / "color_library.json").read_text(encoding="utf-8"))
    hexes = list(data["colors"])
    names = [data["colors"][h] for h in hexes]
    rgb = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in hexes], np.float64) / 255.0
    return hexes, names, rgb


def color_name(argb8) -> str:
    """``ColorNameParser.parseColorName``: exact hex, else nearest (Euclidean RGB 0..1)."""
    hexes, names, rgb = _names()
    h = hex_rgb(argb8)
    try:
        return names[hexes.index(h)]
    except ValueError:
        pass
    c = np.array(argb8[1:], np.float64) / 255.0
    return names[int(np.argmin(((rgb - c) ** 2).sum(1)))]


def color_from_name(name: str) -> tuple[tuple[int, int, int, int], str | None]:
    """``ColorNameParser.parseColorFromNameSingle``: exact (case-insensitive) name, else first containment
    match either way, else Black.  Returns (colour, matched name or None when it fell back to black)."""
    hexes, names, _ = _names()
    n = name.strip().lower()
    for h, nm in zip(hexes, names):
        if nm.lower() == n:
            return parse_hex(h), nm
    for h, nm in zip(hexes, names):
        low = nm.lower()
        if n in low or low in n:
            return parse_hex(h), nm
    return (255, 0, 0, 0), None


# ---------------------------------------------------------------------------
# everything about one colour
# ---------------------------------------------------------------------------

def _r(v: float, d: int = 4) -> float:
    return round(float(v), d)


def describe(argb8, *, name: bool = True) -> dict[str, Any]:
    """All models for one colour (ImageToolbox values + rounded strings)."""
    a, r, g, b = argb8
    h, s, v = rgb_to_hsv(r, g, b)
    hh, hs, hl = rgb_to_hsl(r, g, b)
    c, m, y, k = rgb_to_cmyk(r, g, b)
    x, yy, z = rgb_to_xyz(r, g, b)
    L, A, B = xyz_to_lab(x, yy, z)
    alpha_s = f"{jround(a / 255.0 * 100) / 100:g}"
    out: dict[str, Any] = {
        "hex": hex_css(argb8), "hex_itb": hex_itb(argb8), "argb_int": pack(a, r, g, b) - (1 << 32 if a >= 128 else 0),
        "rgb": [r, g, b], "alpha": _r(a / 255.0, 4), "alpha_8bit": a,
        "hsv": [_r(h, 3), _r(s), _r(v)], "hsl": [_r(hh, 3), _r(hs), _r(hl)],
        "cmyk": [_r(c), _r(m), _r(y), _r(k)],
        "xyz": [_r(x), _r(yy), _r(z)], "lab": [_r(L), _r(A), _r(B)],
        "formats": {  # ColorInfoDisplay fields + ColorCopyFormat strings
            "hex": hex_itb(argb8),
            "rgb_field": f"{r}, {g}, {b}",
            "hsv_field": f"{jround(h)}, {jround(f32(s) * f32(100))}, {jround(f32(v) * f32(100))}",
            "hsl_field": f"{jround(hh)}, {jround(f32(hs) * f32(100))}, {jround(f32(hl) * f32(100))}",
            "cmyk_field": ", ".join(str(jround(f32(t) * f32(100))) for t in (c, m, y, k)),
            "rgb": f"rgb({r}, {g}, {b})", "rgba": f"rgba({r}, {g}, {b}, {alpha_s})",
            "hsl": f"hsl({jround(hh)}, {jround(f32(hs) * f32(100))}%, {jround(f32(hl) * f32(100))}%)",
            "hsv": f"hsv({jround(h)}, {jround(f32(s) * f32(100))}%, {jround(f32(v) * f32(100))}%)",
            "cmyk": "cmyk(" + ", ".join(f"{jround(f32(t) * f32(100))}%" for t in (c, m, y, k)) + ")",
        },
    }
    if name:
        nm = color_name(argb8)
        out["name"] = nm
        out["formats"]["name"] = nm
        out["formats"]["hex_and_name"] = f"{nm} - {hex_itb(argb8)}"
    return out


MODELS = ["auto", "hex", "rgb", "rgba", "hsv", "hsl", "cmyk", "lab", "xyz", "name"]


def _nums(text: str) -> list[float]:
    import re

    s = re.sub(r"^[a-zA-Z]+\s*\(|\)\s*$", "", text.strip())
    parts = [p for p in re.split(r"[,\s/]+", s.replace("%", "")) if p]
    return [float(p) for p in parts]


def parse(value: str, model: str = "auto", *, hex_alpha_first: bool = False) -> tuple[tuple[int, int, int, int], str]:
    """Parse a colour written in one of ImageToolbox's models.

    Field syntax follows ColorInfoDisplay (``"r, g, b"``, ``"h, s%, v%"`` as integers 0..100,
    ``"c, m, y, k"`` 0..100) and CSS-like copies (``rgb(...)``, ``hsl(...)``, ``cmyk(...)``) are
    accepted too.  LAB/XYZ use androidx ColorUtils ranges (L 0..100, XYZ 0..~100).
    Returns ((a, r, g, b), model used).
    """
    v = value.strip()
    m = model.lower()
    if m == "auto":
        low = v.lower()
        if low.startswith("#") or (len(low) in (6, 8) and all(ch in "0123456789abcdef" for ch in low)):
            m = "hex"
        else:
            for pre in ("rgba", "rgb", "hsv", "hsb", "hsl", "cmyk", "lab", "xyz"):
                if low.startswith(pre + "(") or low.startswith(pre + " "):
                    m = {"hsb": "hsv"}.get(pre, pre)
                    break
            else:
                try:
                    n = _nums(v)
                    m = {3: "rgb", 4: "rgba"}.get(len(n), "name")
                except ValueError:
                    m = "name"
    if m == "hex":
        return parse_hex(v, alpha_first=hex_alpha_first), m
    if m == "name":
        c, matched = color_from_name(v)
        return c, m
    n = _nums(v)
    if m == "rgb":
        if len(n) != 3:
            raise ValueError("rgb needs 3 values 0..255")
        return (255, *[min(255, max(0, int(round(x)))) for x in n]), m
    if m == "rgba":
        if len(n) != 4:
            raise ValueError("rgba needs r, g, b (0..255) and alpha (0..1)")
        a = n[3] if n[3] <= 1 else n[3] / 255.0
        return (jround(min(1.0, max(0.0, a)) * 255), *[min(255, max(0, int(round(x)))) for x in n[:3]]), m
    if m == "hsv":
        if len(n) != 3:
            raise ValueError("hsv needs h (0..360), s and v (0..100)")
        return (255, *hsv_to_rgb(n[0], n[1] / 100.0, n[2] / 100.0)), m
    if m == "hsl":
        if len(n) != 3:
            raise ValueError("hsl needs h (0..360), s and l (0..100)")
        return (255, *hsl_to_rgb(n[0], n[1] / 100.0, n[2] / 100.0)), m
    if m == "cmyk":
        if len(n) != 4:
            raise ValueError("cmyk needs 4 values 0..100")
        return (255, *cmyk_to_rgb(*[x / 100.0 for x in n])), m
    if m == "lab":
        if len(n) != 3:
            raise ValueError("lab needs L (0..100), a, b")
        return (255, *xyz_to_rgb(*lab_to_xyz(*n))), m
    if m == "xyz":
        if len(n) != 3:
            raise ValueError("xyz needs X, Y, Z (0..~100)")
        return (255, *xyz_to_rgb(*n)), m
    raise ValueError(f"Unknown colour model {model!r}")
