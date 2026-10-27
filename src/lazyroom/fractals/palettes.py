"""ImageToolbox ``GradientPalette`` presets (core/domain GradientPalette.kt, Apache-2.0).

Generated from the upstream Kotlin source: ARGB ints exactly as ImageToolbox builds them
(``rgb()`` helpers rounded with Kotlin ``roundToInt``).  Order = ``GradientPalette.entries``.
Titles are the English strings ImageToolbox shows for each palette.
"""
from __future__ import annotations

import math

import numpy as np

# key: (title, [ARGB, ...])
PALETTES: dict[str, tuple[str, list[int]]] = {
    "classic": ("Classic", [0xFF05051A, 0xFF123EAB, 0xFF26BCE1, 0xFFF8E16C, 0xFFF06A24, 0xFF3A0812]),
    "fire": ("Fire", [0xFF090004, 0xFF5A0900, 0xFFD33A00, 0xFFFFA51F, 0xFFFFFFB0]),
    "ocean": ("Ocean", [0xFF020B22, 0xFF063B73, 0xFF087E8B, 0xFF31C5C0, 0xFFD8FFF2]),
    "viridis": ("Viridis", [0xFF440154, 0xFF3B528B, 0xFF21918C, 0xFF5EC962, 0xFFFDE725]),
    "magma": ("Magma", [0xFF000004, 0xFF3B0F70, 0xFF8C2981, 0xFFDE4968, 0xFFFE9F6D, 0xFFFCFDBF]),
    "inferno": ("Inferno", [0xFF000004, 0xFF420A68, 0xFF932667, 0xFFDD513A, 0xFFFCA50A, 0xFFFCFFA4]),
    "plasma": ("Plasma", [0xFF0D0887, 0xFF6A00A8, 0xFFB12A90, 0xFFE16462, 0xFFFCA636, 0xFFF0F921]),
    "turbo": ("Turbo", [0xFF30123B, 0xFF4145AB, 0xFF2A9DF4, 0xFF20D5A5, 0xFF8BEB55, 0xFFF9D423, 0xFFF36B1B, 0xFF7A0403]),
    "twilight": ("Twilight", [0xFF20134E, 0xFF6D3580, 0xFFC75D75, 0xFFF6B36B, 0xFF8ED1C5, 0xFF315A8A, 0xFF20134E]),
    "ice": ("Ice", [0xFF02040F, 0xFF102A56, 0xFF2D78B7, 0xFF8BE4F0, 0xFFF5FFFF]),
    "forest": ("Forest", [0xFF07150A, 0xFF174B2B, 0xFF3B7D3A, 0xFF9BBF48, 0xFFF1E7A1]),
    "neon": ("Neon", [0xFF050011, 0xFF7400B8, 0xFFFF007A, 0xFFFFC800, 0xFF00F5D4, 0xFF0077FF]),
    "cividis": ("Cividis", [0xFF00204C, 0xFF2E4A7D, 0xFF666870, 0xFFA08A5B, 0xFFD6AF3C, 0xFFFFE945]),
    "cubehelix": ("Cubehelix", [0xFF000000, 0xFF1D2B53, 0xFF5E3C99, 0xFFB35C8C, 0xFFE3A35D, 0xFFFFFFFF]),
    "spectral": ("Spectral", [0xFF9E0142, 0xFFD53E4F, 0xFFF46D43, 0xFFFEE08B, 0xFFE6F598, 0xFF66C2A5, 0xFF3288BD, 0xFF5E4FA2]),
    "aurora": ("Aurora", [0xFF07152B, 0xFF44318D, 0xFF0B8F9C, 0xFF35D07F, 0xFFD9F36A]),
    "sunset": ("Sunset", [0xFF10143D, 0xFF4A236B, 0xFFA33B69, 0xFFF06A4D, 0xFFFFC56E, 0xFFFFF0B3]),
    "copper": ("Copper", [0xFF080403, 0xFF3A1C12, 0xFF814425, 0xFFC8783E, 0xFFF0B878, 0xFFFFE0B8]),
    "rocket": ("Rocket", [0xFF03051A, 0xFF3F1B43, 0xFF841E5A, 0xFFCB1B4F, 0xFFF06043, 0xFFF6B48F, 0xFFFAEBDD]),
    "mako": ("Mako", [0xFF0B0405, 0xFF342032, 0xFF3B496C, 0xFF357BA3, 0xFF39A7A5, 0xFF8BDAB2, 0xFFDEF5E5]),
    "amethyst": ("Amethyst", [0xFF10002B, 0xFF240046, 0xFF5A189A, 0xFF9D4EDD, 0xFFE0AAFF, 0xFFFFF0FF]),
    "vaporwave": ("Vaporwave", [0xFF17002E, 0xFF5800A3, 0xFFB5179E, 0xFFF72585, 0xFF4CC9F0, 0xFF00F5D4]),
    "earth": ("Earth", [0xFF071A12, 0xFF1D4D32, 0xFF607D3B, 0xFFB49A55, 0xFFD9C9A2, 0xFFF2EFE6]),
    "rainbow": ("Rainbow", [0xFFFF0000, 0xFFFF8000, 0xFFFFFF00, 0xFF00FF00, 0xFF00FFFF, 0xFF0000FF, 0xFF8000FF, 0xFFFF0080]),
    "cool": ("Cool", [0xFF00FFFF, 0xFF20DFFF, 0xFF40BFFF, 0xFF609FFF, 0xFF8080FF, 0xFF9F60FF, 0xFFBF40FF, 0xFFFF00FF]),
    "hot": ("Hot", [0xFF000000, 0xFF400000, 0xFF800000, 0xFFBF4000, 0xFFFF8000, 0xFFFFBF40, 0xFFFFFF80, 0xFFFFFFFF]),
    "purple_dream": ("Purple Dream", [0xFF0D001A, 0xFF1A0033, 0xFF400066, 0xFF660099, 0xFF8C26BF, 0xFFB34DE6, 0xFFD999FF, 0xFFFFE6FF]),
    "lava": ("Lava", [0xFF0D0000, 0xFF1A0000, 0xFF4D0000, 0xFF800000, 0xFFB31A00, 0xFFD94D00, 0xFFFF8000, 0xFFFFCC33]),
    "galaxy": ("Galaxy", [0xFF060013, 0xFF0D0026, 0xFF260059, 0xFF4D0080, 0xFF801AB3, 0xFFB340CC, 0xFFD973D9, 0xFFFFB3E6]),
    "mint": ("Mint", [0xFF002626, 0xFF004D4D, 0xFF1A6666, 0xFF338080, 0xFF4D998C, 0xFF80CCB3, 0xFFB3EBD9, 0xFFD9FFF2]),
    "cherry": ("Cherry", [0xFF26000D, 0xFF4D001A, 0xFF730026, 0xFF990033, 0xFFBF1A4D, 0xFFE64D73, 0xFFFF8CA6, 0xFFFFCCD9]),
    "xf_alternating_grey": ("Alternating Grey", [0xFF000000, 0xFFDCD8D8, 0xFFB4B4B4, 0xFF909090, 0xFF686C6C, 0xFF444848, 0xFF242028, 0xFF000000]),
    "xf_blues": ("Blues", [0xFF000000, 0xFF0018CC, 0xFF3CD4FC, 0xFFB8FCFC, 0xFF00F8FC, 0xFF0038FC, 0xFF000094, 0xFF000000]),
    "xf_chromatic": ("Chromatic", [0xFF303030, 0xFF3060E0, 0xFF98E098, 0xFFE04030, 0xFFD030E0, 0xFF30E098, 0xFFE0E060, 0xFF283030]),
    "xf_default": ("Default", [0xFF000000, 0xFFFC00FC, 0xFF7CFC7C, 0xFF700054, 0xFF387044, 0xFF400020, 0xFF204030, 0xFF000000]),
    "xf_default_white": ("Default White", [0xFF000000, 0xFFFC9C3C, 0xFF3C7C5C, 0xFF7C3C7C, 0xFF9CDC7C, 0xFFDC9C9C, 0xFFFC5CBC, 0xFFFCFCFC]),
    "xf_fire_storm": ("Fire Storm", [0xFF000000, 0xFFEA0E87, 0xFFF96421, 0xFFABD102, 0xFF3EFF41, 0xFF01CCB1, 0xFF2361FA, 0xFF8D0BE6]),
    "xf_froth3": ("Froth 3", [0xFF000000, 0xFF980000, 0xFF680000, 0xFF00A800, 0xFF007800, 0xFF0000D0, 0xFF000088, 0xFF000058]),
    "xf_froth316": ("Froth 3-16", [0xFF000000, 0xFFD40000, 0xFF7C0000, 0xFF00FC00, 0xFF00A800, 0xFF005000, 0xFF0000D4, 0xFF000050]),
    "xf_froth6": ("Froth 6", [0xFF000000, 0xFF700070, 0xFF880000, 0xFF9C9C00, 0xFF00B400, 0xFF0000C8, 0xFF00E0E0, 0xFFFCFCFC]),
    "xf_froth616": ("Froth 6-16", [0xFF000000, 0xFFA000A0, 0xFFA00000, 0xFFA0A000, 0xFF00A000, 0xFF0000A0, 0xFF00A0A0, 0xFFFCFCFC]),
    "xf_gamma1": ("Gamma 1", [0xFF000000, 0xFF5C6060, 0xFF848880, 0xFFA8A4A0, 0xFFC0BCC4, 0xFFD4D4DC, 0xFFE8E8F0, 0xFFFCFCFC]),
    "xf_gamma2": ("Gamma 2", [0xFF000000, 0xFF242420, 0xFF484840, 0xFF6C6C68, 0xFF8C9094, 0xFFB4B4B4, 0xFFDCD4DC, 0xFFFCFCFC]),
    "xf_glasses1": ("3D Glasses 1", [0xFF000000, 0xFF480000, 0xFF900000, 0xFFD80000, 0xFF000020, 0xFF00006C, 0xFF0000B4, 0xFF0000FC]),
    "xf_glasses2": ("3D Glasses 2", [0xFF000000, 0xFF400020, 0xFF800040, 0xFFD00060, 0xFF100090, 0xFF6000B0, 0xFFA000D0, 0xFFF000F0]),
    "xf_good_ega": ("Good EGA", [0xFF000000, 0xFFFC5400, 0xFFFC7E00, 0xFFFCA800, 0xFFFCD200, 0xFFFCFC00, 0xFFFCFC7E, 0xFFFCFCFC]),
    "xf_green": ("Green", [0xFF000000, 0xFF00D800, 0xFF00B400, 0xFF009000, 0xFF006C00, 0xFF004800, 0xFF002000, 0xFF000000]),
    "xf_grey": ("Grey", [0xFF000000, 0xFFDCDCDC, 0xFFB7B7B7, 0xFF939393, 0xFF6E6E6E, 0xFF4A4A4A, 0xFF252525, 0xFF000000]),
    "xf_grid": ("Grid", [0xFF000000, 0xFF2400FF, 0xFF4900FF, 0xFF6D00FF, 0xFF9200FF, 0xFFB600FF, 0xFFDB00FF, 0xFFFF00FF]),
    "xf_headache2": ("Headache 2", [0xFFF00000, 0xFFF02000, 0xFFF04400, 0xFFF46800, 0xFF006C90, 0xFF0048B4, 0xFF0024D8, 0xFF008080]),
    "xf_headache": ("Headache", [0xFF000000, 0xFF00DC20, 0xFF00B844, 0xFFF46800, 0xFFF48C00, 0xFF0048B4, 0xFF0024D8, 0xFF008080]),
    "xf_landscape": ("Landscape", [0xFF000000, 0xFF0000A8, 0xFF0000A8, 0xFF405300, 0xFF402E00, 0xFF400E00, 0xFFD2D2FF, 0xFFFFFFFF]),
    "xf_lyapunov": ("Lyapunov", [0xFF000000, 0xFFDCA000, 0xFFB87C00, 0xFF905400, 0xFF6C3000, 0xFF480C00, 0xFF240000, 0xFF000000]),
    "xf_neon": ("XF Neon", [0xFF000000, 0xFFE0486C, 0xFF000000, 0xFF00E400, 0xFF000000, 0xFFACAC00, 0xFF303000, 0xFF000000]),
    "xf_paint_jet": ("PaintJet", [0xFF18140C, 0xFFC44448, 0xFFF0E848, 0xFFBC306C, 0xFF18140C, 0xFFC44448, 0xFFF0E848, 0xFF2874C4]),
    "xf_royal": ("Royal", [0xFF000000, 0xFF5C007C, 0xFF8C24AC, 0xFFD8B8E0, 0xFFFCFCB4, 0xFFFCFC20, 0xFFA88C24, 0xFF3C0050]),
    "xf_topo": ("Topo", [0xFF000000, 0xFF4484E8, 0xFF3C8C10, 0xFF64B438, 0xFF94E46C, 0xFFE8E468, 0xFFE8AC24, 0xFFFCFCFC]),
    "xf_volcano": ("Volcano", [0xFF000000, 0xFFFC1C00, 0xFFFCFC3C, 0xFFFCE4CC, 0xFFFC983C, 0xFFFC2800, 0xFFB40000, 0xFF3C3C3C]),
    "rgb": ("RGB", [0xFFFF0000, 0xFFFFFF00, 0xFF00FF00, 0xFF00FFFF, 0xFF0000FF, 0xFFFF00FF, 0xFFFF0000]),
    "ryb": ("RYB", [0xFFF20D0D, 0xFFFF8C00, 0xFFFFF20D, 0xFF1A66E6, 0xFF730DBF, 0xFFF20D0D]),
    "cmyk": ("CMYK", [0xFF00FFFF, 0xFF0033FF, 0xFFFF00FF, 0xFFFF3300, 0xFFFFFF00, 0xFF000000, 0xFF00FFFF]),
    "hsv_wheel": ("HSV Wheel", [0xFFFF0000, 0xFFFF8000, 0xFFFFFF00, 0xFF00FF00, 0xFF00FFFF, 0xFF0000FF, 0xFF8000FF, 0xFFFF00FF, 0xFFFF0000]),
    "red_channel": ("Red Channel", [0xFF000000, 0xFF400000, 0xFF8C0000, 0xFFD90D05, 0xFFFF7340, 0xFFFFFFFF]),
    "green_channel": ("Green Channel", [0xFF000000, 0xFF003808, 0xFF008514, 0xFF1FD12E, 0xFF8CFF7A, 0xFFFFFFFF]),
    "blue_channel": ("Blue Channel", [0xFF000000, 0xFF030847, 0xFF051F9E, 0xFF0D66F2, 0xFF66D1FF, 0xFFFFFFFF]),
    "heatmap": ("Heatmap", [0xFF000000, 0xFF400066, 0xFFBF001A, 0xFFFF5900, 0xFFFFF200, 0xFFFFFFFF]),
    "cold_fire": ("Cold Fire", [0xFF00051F, 0xFF0040BF, 0xFF00F2FF, 0xFFFFFFFF, 0xFFFFA600, 0xFFBF000D, 0xFF140000]),
    "ultraviolet": ("Ultraviolet", [0xFF03000D, 0xFF140047, 0xFF5900B3, 0xFFB800FF, 0xFFFF33D1, 0xFF73D9FF]),
    "toxic_waste": ("Toxic Waste", [0xFF000500, 0xFF0A2900, 0xFF337A00, 0xFF94E600, 0xFFE6FF0D, 0xFF1F5926]),
    "blood_moon": ("Blood Moon", [0xFF030000, 0xFF1F0003, 0xFF610005, 0xFFB80D05, 0xFFFF470D, 0xFFFFB84D]),
    "abyss": ("Abyss", [0xFF000005, 0xFF00051F, 0xFF001F3D, 0xFF00617A, 0xFF14B8AD, 0xFF8CFFE0]),
    "electric_candy": ("Electric Candy", [0xFF050014, 0xFF4000D9, 0xFFFF00B8, 0xFFFF590D, 0xFFE6FF00, 0xFF00FFB8, 0xFF050014]),
    "black_gold": ("Black Gold", [0xFF000000, 0xFF140E03, 0xFF472E06, 0xFF9E6B0F, 0xFFF2BF38, 0xFFFFF2AD, 0xFF000000]),
    "ghost": ("Ghost", [0xFF000000, 0xFF141A24, 0xFF3D4D61, 0xFF7A9EB3, 0xFFC7EBF0, 0xFFFFFFFF, 0xFF000000]),
    "grayscale": ("Grayscale", [0xFF000000, 0xFF404040, 0xFF909090, 0xFFFFFFFF]),
    "soft_rainbow": ("Soft rainbow", [0xFF3FCEBC, 0xFF3CBCEB, 0xFF5F96E7, 0xFF816FE3, 0xFF9F5EE2, 0xFFBD4CE0, 0xFFDE589F, 0xFFFF645E, 0xFFFDA859, 0xFFFAEC54, 0xFF9EE671, 0xFF67E282, 0xFF3FCEBC]),
}


PALETTE_NAMES: list[str] = list(PALETTES)


def palette_colors(name: str) -> list[int]:
    """ARGB ints of a named ImageToolbox palette."""
    return list(PALETTES[name][1])


def _kround(v: float) -> int:
    """Kotlin ``Double.roundToInt`` (round half up)."""
    return int(math.floor(v + 0.5))


def _interp_argb(first: int, second: int, fraction: float) -> int:
    out = 0
    for shift in (24, 16, 8, 0):
        s = (first >> shift) & 0xFF
        e = (second >> shift) & 0xFF
        out |= max(0, min(255, _kround(s + (e - s) * fraction))) << shift
    return out


def sample_colors(colors: list[int], count: int) -> list[int]:
    """``GradientPalette.sampleColors(count)``: ``count`` evenly spaced colours (ARGB)."""
    if count <= 0:
        raise ValueError("count must be > 0")
    if count == 1:
        return [colors[0]]
    out = []
    last = len(colors) - 1
    for i in range(count):
        scaled = i / (count - 1) * last
        fi = min(max(int(math.floor(scaled)), 0), last)
        si = min(fi + 1, last)
        out.append(_interp_argb(colors[fi], colors[si], scaled - fi))
    return out


def argb_to_rgba(colors) -> np.ndarray:
    """ARGB ints -> (N, 4) float32 RGBA in 0..1."""
    a = np.asarray(colors, dtype=np.int64)
    return np.stack([(a >> 16) & 0xFF, (a >> 8) & 0xFF, a & 0xFF, (a >> 24) & 0xFF], -1).astype(np.float32) / 255.0


def rgba_to_argb(c) -> int:
    """(r, g, b, a) floats 0..1 -> ARGB int (rounded like Compose ``Color.toArgb``)."""
    r, g, b, a = (max(0, min(255, int(math.floor(float(v) * 255.0 + 0.5)))) for v in c)
    return a << 24 | r << 16 | g << 8 | b
