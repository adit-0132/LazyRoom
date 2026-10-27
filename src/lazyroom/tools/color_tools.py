"""ImageToolbox colour tools: conversion between colour models, mixing, harmonies, shading, picking a
colour from an image, palette contact-sheet PDF, plus (registered from private modules) the Material You
scheme generator, the "simple colours" palette and the curves preset codec.

Sources: feature/color-tools, feature/pick-color, feature/palette-pdf, feature/palette-tools,
lib/colors, lib/dynamic-theme, lib/curves (T8RIN/ImageToolbox, Apache-2.0).
See docs/dev/port_reports/color_tools.md.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import ErrorCode, LazyRoomError, invalid
from ..core.io import READ_EXTS, atomic_write
from ..core.params import Bool, Choice, Color, File, Float, Int, Point, StringList, Text
from . import _color_models as cm
from .registry import out_path, read_image, save, tool

F32 = np.float32
ITB = "ImageToolbox"
_NAMED = re.compile(r"^\s*(?:(?P<name>[^=]*?)\s*=\s*)?(?P<color>.+?)\s*$")  # 'Name=#RRGGBB' or a colour


def _src(feature: str) -> dict[str, str]:
    return {"project": ITB, "feature": feature}


def _c8(color) -> tuple[int, int, int, int]:
    return cm.from_float(color)


def _entry(c8, name: bool = True) -> dict[str, Any]:
    d = {"hex": cm.hex_css(c8), "hex_itb": cm.hex_itb(c8), "rgb": list(c8[1:]), "alpha": round(c8[0] / 255, 4)}
    if name:
        d["name"] = cm.color_name(c8)
    return d


def _swatch_strip(api, rows: list[list[tuple[int, int, int, int]]], path: str, overwrite: bool,
                  cell: int = 96) -> dict[str, Any]:
    n = max(len(r) for r in rows)
    img = np.zeros((cell * len(rows), cell * n, 4), F32)
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            img[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = np.array(cm.to_float(c), F32)
    return save(api, img if img[..., 3].min() < 1 else img[..., :3], path, overwrite=overwrite)


# =============================================================================
# conversion
# =============================================================================

@tool("color_convert", "Convert colour between models", category="color", params=[
    Text("value", "#3366CC", max_length=200,
         description="Colour in any model: '#RRGGBB', 'rgb(51, 102, 204)' or '51, 102, 204', 'hsv(220, 75%, 80%)', "
                     "'hsl(...)', 'cmyk(75%, 50%, 0%, 20%)', 'lab(45, 18.7, -57.9)', 'xyz(...)' or a colour name"),
    Choice("model", "auto", choices=cm.MODELS,
           description="Model of 'value' (auto = detect from the syntax). Field syntax follows ImageToolbox's "
                       "Color Tools: HSV/HSL/CMYK percentages as 0..100 numbers"),
    Choice("hex_alpha_order", "rrggbbaa", choices=["rrggbbaa", "aarrggbb"],
           description="How to read 8-digit hex: CSS/LazyRoom #RRGGBBAA or Android/ImageToolbox #AARRGGBB"),
], source=_src("Color Tools: color info (HEX/RGB/HSV/HSL/CMYK/Name) + copy formats"))
def color_convert(api, value, model, hex_alpha_order):
    """Convert one colour to every model ImageToolbox offers (HEX, RGB, RGBA, HSV, HSL, CMYK, colour
    name) plus androidx ColorUtils XYZ / CIE Lab, with the app's copy-format strings."""
    try:
        c8, used = cm.parse(value, model, hex_alpha_first=hex_alpha_order == "aarrggbb")
    except ValueError as e:
        raise invalid(str(e), param="value") from None
    out = cm.describe(c8)
    out["input"] = {"value": value, "model": used}
    if used == "name":
        _, matched = cm.color_from_name(value)
        out["input"]["matched_name"] = matched
        if matched is None:
            out["input"]["note"] = "No colour name matched; ImageToolbox falls back to black"
    return out


# =============================================================================
# mixing / harmonies / shading
# =============================================================================

@tool("color_mix", "Colour mixing", category="color", params=[
    Color("color", "#3366CC", description="Selected colour"),
    Color("mix_with", "#FFFF00", description="Colour to mix with (ImageToolbox defaults to the theme's tertiary "
                                             "colour, else yellow)"),
    Int("variations", 3, min=2, max=20, description="Number of steps (ImageToolbox: 2..20, default 3)"),
    Text("swatch_output", "", description="Optional swatch strip image"), Bool("overwrite", False),
], source=_src("Color Tools: Color Mixing (mixWith)"))
def color_mix(api, color, mix_with, variations, swatch_output, overwrite):
    """Blend from ``color`` to ``mix_with`` in ``variations`` steps (androidx ColorUtils.blendARGB in sRGB,
    step i at i / (variations - 1))."""
    a, b = _c8(color), _c8(mix_with)
    cols = cm.mix_with(a, b, variations)
    res: dict[str, Any] = {"colors": [dict(_entry(c), ratio=round(i / (variations - 1), 6)) for i, c in enumerate(cols)]}
    if swatch_output:
        res["swatch"] = _swatch_strip(api, [cols], swatch_output, overwrite)
    return res


@tool("color_harmonies", "Colour harmonies", category="color", params=[
    Color("color", "#3366CC"),
    Choice("type", "all", choices=["all"] + cm.HARMONY_TYPES,
           description="ImageToolbox HarmonyType (HSV hue rotations): complementary +180; analogous ±30; "
                       "analogous_complementary ±30, +180; triadic ±120; split_complementary ±150; tetradic +30, "
                       "+180, +210; square +90, +180, +270"),
    Text("swatch_output", "", description="Optional swatch image (one row per harmony)"), Bool("overwrite", False),
], source=_src("Color Tools: Color Harmonies (applyHarmony)"))
def color_harmonies(api, color, type, swatch_output, overwrite):
    """Harmonious colours of ``color`` (the first entry of every list is the colour itself)."""
    c8 = _c8(color)
    kinds = cm.HARMONY_TYPES if type == "all" else [type]
    rows = {k: cm.harmony(c8, k) for k in kinds}
    res: dict[str, Any] = {"color": _entry(c8),
                           "harmonies": {k: {"hue_offsets": cm.HARMONY_OFFSETS[k], "colors": [_entry(c) for c in v]}
                                         for k, v in rows.items()}}
    if swatch_output:
        res["swatch"] = _swatch_strip(api, list(rows.values()), swatch_output, overwrite)
    return res


@tool("color_shading", "Colour shading (tints, tones, shades)", category="color", params=[
    Color("color", "#3366CC"),
    Int("variations", 5, min=2, max=20, description="Steps per row (ImageToolbox: 2..20, default 5)"),
    Text("swatch_output", "", description="Optional swatch image (rows: tints, tones, shades)"),
    Bool("overwrite", False),
], source=_src("Color Tools: Color Shading"))
def color_shading(api, color, variations, swatch_output, overwrite):
    """Tints (towards white, up to 80 %), tones (towards grey #8E918F, up to 90 %) and shades (towards
    black, up to 90 %) exactly as ImageToolbox computes them (blendARGB steps)."""
    rows = cm.shading(_c8(color), variations)
    res: dict[str, Any] = {k: [_entry(c) for c in v] for k, v in rows.items()}
    if swatch_output:
        res["swatch"] = _swatch_strip(api, [rows["tints"], rows["tones"], rows["shades"]], swatch_output, overwrite)
    return res


# =============================================================================
# pick colour
# =============================================================================

@tool("color_pick", "Pick colour from image", category="color", params=[
    File("input", "", file_kind="image", required=True),
    Point("point", (0.5, 0.5), description="Position: normalised 0..1 (default) or pixels with unit='px'"),
    Choice("unit", "normalized", choices=["normalized", "px"]),
    Int("radius", 0, min=0, max=500, unit="px",
        description="0 = the single pixel (ImageToolbox); > 0 = average of the (2r+1)^2 square around it"),
    Choice("average", "srgb", choices=["srgb", "linear"],
           description="Area average in sRGB values or in linear light"),
], source=_src("Pick Color From Image (ImageColorDetector.calculateColorInPixel)"))
def color_pick(api, input, point, unit, radius, average):
    """Colour at a point (or the mean of a square area) of an image, in every colour model + name."""
    img, _ = read_image(api, input)
    h, w = img.shape[:2]
    if unit == "px":
        x, y = int(point[0]), int(point[1])
    else:
        if not (0 <= point[0] <= 1 and 0 <= point[1] <= 1):
            raise invalid("normalised point must be within 0..1", param="point")
        x, y = int(np.float32(point[0]) * w), int(np.float32(point[1]) * h)  # lerp(...).toInt()
    x, y = min(max(x, 0), w - 1), min(max(y, 0), h - 1)
    x0, x1, y0, y1 = max(0, x - radius), min(w, x + radius + 1), max(0, y - radius), min(h, y + radius + 1)
    area = np.clip(img[y0:y1, x0:x1], 0, 1).astype(np.float64)
    if area.shape[2] == 4:  # average colour weighted by alpha (straight alpha in, straight out)
        a = area[..., 3:4]
        wsum = max(float(a.sum()), 1e-12)
        if average == "linear":
            from ..core.color import linear_to_srgb, srgb_to_linear

            lin = srgb_to_linear(area[..., :3].astype(F32)).astype(np.float64)
            rgb = linear_to_srgb(((lin * a).sum((0, 1)) / wsum).astype(F32)[None, None])[0, 0]
        else:
            rgb = (area[..., :3] * a).sum((0, 1)) / wsum
        alpha = float(a.mean())
    else:
        if average == "linear":
            from ..core.color import linear_to_srgb, srgb_to_linear

            rgb = linear_to_srgb(srgb_to_linear(area.astype(F32)).mean((0, 1))[None, None])[0, 0]
        else:
            rgb = area.mean((0, 1))
        alpha = 1.0
    c8 = cm.from_float((*[float(v) for v in rgb], 1.0))  # ImageToolbox drops alpha: Color(r, g, b)
    out = cm.describe(c8)
    out.update(position_px=[x, y], image_size=[w, h], area=[x0, y0, x1 - x0, y1 - y0],
               pixels=int((x1 - x0) * (y1 - y0)), source_alpha=round(alpha, 4))
    return out


# =============================================================================
# palette PDF (feature/palette-pdf)
# =============================================================================

def _image_colors(img: np.ndarray, count: int):
    """``extractImageColorPalette(image, maximumColorCount)``: androidx Palette swatches, named, distinct by
    name, sorted by luminance then hue."""
    from ._androidx_palette import extract_image_color_palette
    from ._palette_pdf import PdfColor, argb_hex

    out = []
    for e in extract_image_color_palette(img, maximum_color_count=count):
        r, g, b = e["rgb"]
        argb = cm.pack(255, r, g, b)
        out.append(PdfColor(argb=argb, name=e["name"], hex=argb_hex(argb)))
    return out


def _is_image_source(path: Path) -> bool:
    from . import _palette_formats as pf

    pal_fmts = [f for f in pf.formats_for_filename(path.name) if f != "image"]
    if pal_fmts:
        try:
            pal, _ = pf.decode(path.read_bytes(), filename=path.name)
            if pal.total_color_count > 0:
                return False
        except pf.PaletteFormatError:
            pass
    return path.suffix.lower() in READ_EXTS


@tool("palette_pdf", "Palette PDF contact sheet", category="color", params=[
    File("input", "", file_kind="any", description="Source: an image (palette is extracted) or a palette file "
                                                    "(.gpl, .ase, .aco, ... any format palette_import reads)"),
    StringList("colors", [], description="Or: colours as '#RRGGBB(AA)' or 'Name=#RRGGBB' (named from the colour "
                                         "library when no name is given)"),
    Text("palette_format", "", max_length=40, description="Palette file format id (empty = detect)"),
    Text("palette_name", "", max_length=200, description="Palette name (colour lists; overrides a file's name)"),
    Int("maximum_color_count", 32, min=4, max=64, description="Colours extracted from an image (4..64)"),
    Int("columns", 3, min=1, max=6), Float("margin", 24.0, min=0, max=72, unit="pt", description="Page margins"),
    Float("spacing", 8.0, min=0, max=36, unit="pt", description="Colour spacing"),
    Bool("include_source_image", True, description="Place the source image above the colour grid on the first page"),
    Bool("show_source_filename", False, description="Source filename at the top of each page"),
    Bool("show_palette_name", True, description="Name stored inside the palette below the filename"),
    Bool("show_color_names", True), Bool("show_hex_values", True),
    Int("image_dpi", 300, min=72, max=600, description="Resolution of the embedded source image"),
    Text("output", "", description="Output .pdf path"), Bool("overwrite", False),
], source=_src("Palette PDF (PalettePdfRenderer)"))
def palette_pdf(api, input, colors, palette_format, palette_name, maximum_color_count, columns, margin, spacing,
                include_source_image, show_source_filename, show_palette_name, show_color_names,
                show_hex_values, image_dpi, output, overwrite):
    """Arrange palette colours (and optionally the source image) on A4 PDF pages: a vector contact sheet
    with colour cards, names and HEX values (ImageToolbox Palette PDF layout and defaults)."""
    from . import _palette_formats as pf
    from ._palette_pdf import PalettePdf, PdfColor, PdfParams, argb_hex

    if bool(input) == bool(colors):
        raise invalid("Give exactly one of input (image or palette file) or colors")
    p = out_path(api, output, overwrite=overwrite)
    if p.suffix.lower() != ".pdf":
        raise invalid("output must be a .pdf file", param="output")
    image = None
    name: str | None = None
    filename = ""
    source_type = "colors"
    if input:
        src = api.ws.policy.check_read(input)
        filename = src.name
        if palette_format or not _is_image_source(src):
            source_type = "palette_file"
            try:
                pal, fmt = pf.decode(src.read_bytes(), fmt=palette_format or None, filename=src.name)
            except pf.PaletteFormatError as e:
                raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"Could not read palette {src.name}: {e}",
                                      details={"param": "input"}) from None
            seen, cols = set(), []
            for c in pal.all_colors():
                if c.alpha <= 0:
                    continue
                argb = c.to_argb() & 0xFFFFFFFF
                if (argb, c.name) in seen:  # distinctBy { argb to name }
                    continue
                seen.add((argb, c.name))
                cols.append(PdfColor(argb=argb, name=c.name, hex=argb_hex(argb)))
            name = pal.name
        else:
            source_type = "image"
            image, _ = read_image(api, input)
            cols = _image_colors(image, maximum_color_count)
    else:
        cols = []
        for s in colors:
            m = _NAMED.match(s)
            try:
                c8, _ = cm.parse(m.group("color").strip() if m else s, "auto")
            except (ValueError, AttributeError):
                raise invalid(f"Bad colour entry {s!r}", param="colors") from None
            nm = ((m.group("name") or "").strip() if m else "") or cm.color_name(c8)
            argb = cm.pack(*c8)
            cols.append(PdfColor(argb=argb, name=nm, hex=argb_hex(argb)))
    if palette_name:
        name = palette_name
    if not cols:
        raise invalid("Palette must contain at least one color")
    params = PdfParams(columns=columns, margin=margin, spacing=spacing, include_source_image=include_source_image,
                       show_source_filename=show_source_filename, show_palette_name=show_palette_name,
                       show_color_names=show_color_names, show_hex_values=show_hex_values)
    doc = PalettePdf(image, cols, params, filename, name, image_dpi=image_dpi)
    data = doc.render()
    atomic_write(p, data, overwrite=overwrite)
    return {"output": str(p), "bytes": len(data), "pages": len(doc.pages), "source_type": source_type,
            "palette_name": name, "colors": [{"name": c.name, "hex": c.hex} for c in cols], "layout": doc.layout}


# Tools defined in private modules (registered on import): Material You scheme + simple-colours palette
# (lib/dynamic-theme, androidx.palette, material-color-utilities ports) and the curves preset codec.
from . import _curves_presets, _palette_tools  # noqa: E402,F401
