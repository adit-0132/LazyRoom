"""ImageToolbox Palette Tools as tools: the Material You scheme generator and the "Default" (simple
colours) palette mode.  Algorithms live in ``_material`` (material-color-utilities + DynamicTheme.kt)
and ``_androidx_palette`` (androidx.palette + lib/colors)."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import invalid
from ..core.params import Bool, Choice, File, Float, Int, Text, parse_color
from . import _androidx_palette as AP
from . import _material as M
from .registry import read_image, save, tool

F32 = np.float32
ITB = "ImageToolbox"


def _hex(argb: int) -> str:
    """ImageToolbox ``Color.toHex()`` without the ARGB ambiguity: '#RRGGBB', or '#RRGGBBAA' if translucent."""
    a = (argb >> 24) & 255
    return M.argb_to_hex(argb) if a == 255 else f"{M.argb_to_hex(argb)}{a:02X}"


def _hct(argb: int) -> dict[str, float]:
    h = M.Hct(argb)
    return {"hue": round(h.hue, 3), "chroma": round(h.chroma, 3), "tone": round(h.tone, 3)}


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except Exception:  # pragma: no cover - Pillow without FreeType
        return ImageFont.load_default()


def _ink(argb: int) -> tuple[int, int, int]:
    return (0, 0, 0) if M.lstar_from_argb(argb | 0xFF000000) > 55 else (255, 255, 255)


def _render_schemes(schemes: dict[str, dict[str, int]], cols: int = 4) -> np.ndarray:
    from PIL import Image, ImageDraw

    cw, ch, pad, head = 260, 64, 8, 40
    roles = list(M.COMPOSE_ROLES)
    rows = -(-len(roles) // cols)
    block_h = head + rows * (ch + pad)
    img = Image.new("RGB", (cols * (cw + pad) + pad, len(schemes) * block_h + pad), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f_head, f = _font(22), _font(14)
    for b, (theme, cs) in enumerate(schemes.items()):
        y0 = b * block_h + pad
        d.text((pad, y0 + 8), f"{theme} scheme", fill=(0, 0, 0), font=f_head)
        for i, role in enumerate(roles):
            r, c = divmod(i, cols)
            x, y = pad + c * (cw + pad), y0 + head + r * (ch + pad)
            argb = cs[role]
            d.rectangle([x, y, x + cw - 1, y + ch - 1], fill=((argb >> 16) & 255, (argb >> 8) & 255, argb & 255))
            ink = _ink(argb)
            d.text((x + 8, y + 10), role, fill=ink, font=f)
            d.text((x + 8, y + 34), _hex(argb), fill=ink, font=f)
    return np.asarray(img, F32) / 255.0


def _render_palette_rows(colors: list[dict[str, Any]]) -> np.ndarray:
    """ImageToolbox ImageColorPalette rows: colour circle, name, hex."""
    from PIL import Image, ImageDraw

    w, rh, pad = 560, 56, 4
    img = Image.new("RGB", (w, max(1, len(colors)) * (rh + pad) + pad), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = _font(17)
    for i, c in enumerate(colors):
        y = pad + i * (rh + pad)
        d.rounded_rectangle([pad, y, w - pad - 1, y + rh - 1], radius=16, fill=(243, 243, 246))
        d.ellipse([pad + 8, y + 9, pad + 46, y + 47], fill=tuple(c["rgb"]))
        d.text((pad + 58, y + 17), c["name"], fill=(0, 0, 0), font=f)
        d.text((w - pad - 16, y + 17), c["hex"].upper(), fill=(0, 0, 0), font=f, anchor="ra")
    return np.asarray(img, F32) / 255.0


@tool("material_you_scheme", "Material You colour scheme", category="color", params=[
    File("input", "", file_kind="image",
         description="Image: provides the key colour (unless 'seed' is given) and the MCU seed colours"),
    Text("seed", "", max_length=64, description="Key colour ('#RRGGBB' or a colour name); overrides the image"),
    Choice("key_color_source", "imagetoolbox", choices=["imagetoolbox", "mcu_score"],
           description="imagetoolbox: androidx Palette dominant colour blended 50% with the vibrant swatch "
                       "(ImageToolbox extractPrimaryColor); mcu_score: best seed of material-color-utilities "
                       "(QuantizerCelebi 128 colours + Score)"),
    Choice("style", "tonal_spot", choices=list(M.ITB_STYLES), description="ImageToolbox PaletteStyle"),
    Choice("theme", "both", choices=["both", "light", "dark"]),
    Float("contrast", 0.0, min=-1, max=1, step=0.01, description="Contrast level (-1 reduced .. 0 standard .. 1 high)"),
    Bool("invert_colors", False, description="ImageToolbox 'Invert colors' (RGB xor 0xFFFFFF on every role)"),
    Bool("amoled", False, description="AMOLED mode (dark scheme only): black surfaces, darkened roles"),
    Choice("color_blind", "none", choices=["none", *M.COLOR_BLIND_TYPES],
           description="ImageToolbox colour-blind simulation applied to the scheme"),
    Int("max_seeds", 4, min=1, max=16, description="Number of MCU seed colours to report from the image"),
    Text("swatch_output", "", description="Optional PNG/JPG rendering of every scheme role"),
    Bool("overwrite", False),
], source={"project": ITB, "feature": "Palette Tools: Material You (lib/dynamic-theme + MaterialKolor)"})
def material_you_scheme(api, input, seed, key_color_source, style, theme, contrast, invert_colors, amoled,
                        color_blind, max_seeds, swatch_output, overwrite):
    """Material You (Material 3) light/dark colour schemes from an image or a seed colour, as ImageToolbox
    generates them (MaterialKolor DynamicScheme + ImageToolbox's error/on-colour/surface adjustments);
    also returns the material-color-utilities seed colours of the image and the tonal palettes."""
    if not input and not seed:
        raise invalid("Give an input image or a seed colour", param="input")
    res: dict[str, Any] = {}
    seeds: list[tuple[int, float | None]] = []
    if input:
        img, _ = read_image(api, input)
        u8 = AP.to_u8(img)
        ranked, pops = M.source_colors_from_image(
            np.dstack([u8, (np.clip(img[..., 3], 0, 1) * 255 + 0.5).astype(np.uint8)]) if img.shape[-1] == 4 else u8,
            desired=max_seeds)
        seeds = ranked
        total = float(sum(pops.values())) or 1.0
        res["seeds"] = [{"hex": _hex(a), **_hct(a), "population": pops.get(a, 0),
                         "proportion": round(pops.get(a, 0) / total, 4),
                         "score": None if s is None else round(s, 4), "fallback": s is None} for a, s in ranked]
    if seed:
        r, g, b, _ = parse_color(seed, "seed")
        key = M.argb_from_rgb(*(int(round(v * 255)) for v in (r, g, b)))
        src = "seed"
    elif key_color_source == "mcu_score":
        key, src = seeds[0][0], "mcu_score"
    else:
        key, src = AP.extract_primary_color(img), "imagetoolbox"
    res["key_color"] = M.argb_to_hex(key)
    if (key >> 24) & 255 != 255:
        res["key_color_alpha"] = (key >> 24) & 255  # ImageToolbox blends with transparent when no vibrant swatch
    res["key_color_source"] = src
    res["key_color_hct"] = _hct(key)
    themes = ["light", "dark"] if theme == "both" else [theme]
    cb = None if color_blind == "none" else color_blind
    raw = {t: M.itb_color_scheme(key, style, t == "dark", contrast, invert=invert_colors, amoled=amoled,
                                 color_blind=cb) for t in themes}
    res["style"], res["contrast"] = style, contrast
    res["schemes"] = {t: {role: _hex(v) for role, v in cs.items()} for t, cs in raw.items()}
    sch = M.itb_dynamic_scheme(key, style, False, contrast)
    res["tonal_palettes"] = {name: {"key_color": M.argb_to_hex(p.key_color.to_int()), "hue": round(p.hue, 3),
                                    "chroma": round(p.chroma, 3), "tones": M.iter_tones(p)}
                             for name, p in sch.palettes().items()}
    res["compose_code"] = "\n\n".join(M.as_code_string(raw[t], t == "dark") for t in themes)
    if swatch_output:
        res["swatch"] = save(api, _render_schemes(raw), swatch_output, overwrite=overwrite)
    return res


@tool("palette_simple", "Simple colours palette", category="color", params=[
    File("input", "", file_kind="image", required=True),
    Int("count", 32, min=1, max=128, description="Maximum colour count (androidx Palette maximumColorCount)"),
    Bool("include_targets", False,
         description="Also return the dominant swatch and the six androidx Palette targets "
                     "(light/dark vibrant, vibrant, light/dark muted, muted)"),
    Text("swatch_output", "", description="Optional PNG/JPG list rendering (circle, name, hex)"),
    Bool("overwrite", False),
], source={"project": ITB, "feature": "Palette Tools: Default palette (androidx Palette + lib/colors)"})
def palette_simple(api, input, count, include_targets, swatch_output, overwrite):
    """ImageToolbox's default palette: androidx Palette swatches (median cut on a 112x112-area image,
    black/white/skin-tone filter) named from the colour library, with population percentages, one entry
    per name, sorted by luminance then hue."""
    img, _ = read_image(api, input)
    pal = AP.generate(AP.to_u8(img), maximum_color_count=count)
    colors = AP.named_palette_data(pal)
    for c in colors:
        c["percent"] = round(c["percent"], 6)
    res: dict[str, Any] = {"colors": colors, "count": len(colors)}
    if include_targets:
        def sw(s):
            return None if s is None else {"hex": s.hex, "rgb": [(s.rgb >> 16) & 255, (s.rgb >> 8) & 255, s.rgb & 255],
                                           "population": s.population}
        res["dominant"] = sw(pal.dominant)
        res["targets"] = {k: sw(v) for k, v in pal.selected.items()}
    if swatch_output:
        res["swatch"] = save(api, _render_palette_rows(colors), swatch_output, overwrite=overwrite)
    return res
