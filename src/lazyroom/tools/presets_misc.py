"""ImageToolbox preset catalogues as tools: colour library, palette file formats, shader presets,
collage layouts and built-in export profiles.

Data files (``src/lazyroom/data``): ``color_library.json``, ``shaders/*.itshader``,
``collage_layouts.json``, ``export_profiles.json``.  Codecs: ``_palette_formats`` (lib/palette port),
``_collage_layouts`` (lib/collages port).  See docs/dev/port_reports/presets_misc.md.
"""
from __future__ import annotations

import colorsys
import functools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import ErrorCode, LazyRoomError, invalid
from ..core.io import atomic_write, encode_image
from ..core.params import Bool, Choice, Color, File, Float, Int, StringList, Text, color_to_hex, parse_color
from ..core.paths import data_dir, package_data_dir
from . import _palette_formats as pf
from .registry import expand_inputs, out_path, read_image, save, tool

F32 = np.float32
ITB = "ImageToolbox"


# =============================================================================
# Colour library (feature/color-library + lib/colors ColorNameParser)
# =============================================================================

@functools.lru_cache(maxsize=1)
def _library() -> tuple[list[str], list[str], np.ndarray, dict[str, Any]]:
    data = json.loads((package_data_dir() / "color_library.json").read_text(encoding="utf-8"))
    hexes = list(data["colors"])
    names = [data["colors"][h] for h in hexes]
    rgb = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in hexes], np.float64) / 255.0
    meta = {k: v for k, v in data.items() if k != "colors"}
    return hexes, names, rgb, meta


def _entry(i: int, distance: float | None = None) -> dict[str, Any]:
    hexes, names, rgb, _ = _library()
    d = {"name": names[i], "hex": hexes[i], "rgb": [int(round(v * 255)) for v in rgb[i]]}
    if distance is not None:
        d["distance"] = round(float(distance), 6)
    return d


def color_name(color) -> str:
    """``ColorNameParser.parseColorName``: exact hex match, else the nearest name (Euclidean RGB 0..1)."""
    hexes, names, rgb, _ = _library()
    c = np.array(parse_color(color)[:3], np.float64)
    return names[int(np.argmin(((rgb - c) ** 2).sum(1)))]


@tool("color_library", "Colour library (named colours)", category="color", params=[
    Text("query", "", max_length=200,
         description="Search: name contains the query (or the query contains the name), or the hex contains it"),
    Text("color", "", max_length=64, description="Find the nearest named colours to this colour (#RRGGBB, ...)"),
    Int("limit", 50, min=1, max=5000, description="Maximum number of results per list"),
], source={"project": ITB, "feature": "Color Library (lib/colors color_names.json, licence-verified subset)"})
def color_library(api, query, color, limit):
    """Search ~31k named colours by name or hex, and name any colour (nearest match).

    Without arguments returns the library summary and licences.  Search follows ImageToolbox's colour
    library (case-insensitive, sorted by name; exact name matches are listed first).  ``color`` returns
    the nearest names by RGB distance; the first is ImageToolbox's ``parseColorName`` result.
    """
    hexes, names, rgb, meta = _library()
    res: dict[str, Any] = {"total_colors": len(hexes)}
    if query.strip():
        q = query.strip().lower()
        idx = [i for i, (h, n) in enumerate(zip(hexes, names))
               if q in n.lower() or (n.lower() in q) or q.lstrip("#") in h.lower().lstrip("#")]
        idx.sort(key=lambda i: (names[i].lower() != q, names[i]))
        res["query"] = query
        res["matches"] = len(idx)
        res["colors"] = [_entry(i) for i in idx[:limit]]
    if color.strip():
        try:
            c = np.array(parse_color(color.strip())[:3], np.float64)
        except LazyRoomError:
            raise invalid(f"Not a colour: {color!r}", param="color") from None
        d = np.sqrt(((rgb - c) ** 2).sum(1))
        order = np.argsort(d, kind="stable")[:limit]
        res["color"] = color_to_hex(c)
        res["exact"] = bool(d[order[0]] == 0)
        res["nearest"] = [_entry(int(i), d[i]) for i in order]
    if not query.strip() and not color.strip():
        res.update(meta)
    return res


# =============================================================================
# Palette file formats (feature/palette-tools + lib/palette)
# =============================================================================

_ENCODABLE = [f for f in pf.EXPORT_ORDER if pf.FORMATS[f]["can_encode"]]
_DECODABLE = [f for f in pf.FORMATS if pf.FORMATS[f]["can_decode"]]


def _palette_json(p: "pf.Palette") -> dict[str, Any]:
    def col(c: "pf.PaletteColor") -> dict[str, Any]:
        r, g, b, a = c.to_rgba()
        d = {"name": c.name, "hex": c.hex(), "rgba": [round(v, 6) for v in (r, g, b, a)],
             "color_space": c.color_space, "components": [round(float(v), 6) for v in c.components]}
        if c.color_type != "global":
            d["color_type"] = c.color_type
        return d

    return {"name": p.name, "colors": [col(c) for c in p.colors],
            "groups": [{"name": g.name, "colors": [col(c) for c in g.colors]} for g in p.groups],
            "total_colors": p.total_color_count}


def _format_for_output(fmt: str, path: str) -> str:
    if fmt != "auto":
        return fmt
    for f in pf.formats_for_filename(path):
        if f in _ENCODABLE:
            return f
    raise invalid(f"Cannot infer a palette format from {Path(path).name!r}", param="format",
                  hint="Pass format explicitly (e.g. 'gimp' for .gpl, 'ase', 'aco').", choices=_ENCODABLE)


def _image_palette(api, path: str, count: int) -> list[tuple[str, tuple[float, float, float]]]:
    """Image -> named colours (approximates lib/colors ``extractImageColorPalette``)."""
    from .registry import run_tool

    cols = run_tool(api, "palette", {"input": path, "count": min(count, 32), "method": "median_cut"})["colors"]
    seen, out = set(), []
    for c in cols:
        rgb = tuple(v / 255 for v in c["rgb"])
        name = color_name(rgb)
        if name not in seen:  # distinctBy { name }
            seen.add(name)
            out.append((name, rgb))

    def lum(c):  # Compose Color.luminance()
        lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    out.sort(key=lambda t: (lum(t[1]), colorsys.rgb_to_hsv(*t[1])[0]))
    return out


_NAMED = re.compile(r"^\s*(?:(?P<name>.*?)\s*[=:]\s*)?(?P<color>#?[0-9A-Fa-f]{3,8}|\S+)\s*$")


@tool("palette_export", "Export palette file", category="color", params=[
    StringList("colors", [], description="Colours as '#RRGGBB(AA)' or 'Name=#RRGGBB' (ImageToolbox names them "
                                          "with the colour library when no name is given)"),
    File("input", "", file_kind="image", description="Or: build the palette from this image"),
    Int("count", 32, min=1, max=128, description="Max colours when building from an image"),
    File("source", "", file_kind="any", description="Or: convert this palette file (any supported format)"),
    Text("name", "", max_length=200, description="Palette name (for formats that store one)"),
    Choice("format", "auto", choices=["auto"] + _ENCODABLE,
           description="Palette format; auto = from the output extension"),
    Bool("name_colors", True, description="Name unnamed colours from the colour library"),
    Text("output", "", description="Output file (.gpl, .ase, .aco, .act, .pal, ...)"), Bool("overwrite", False),
], source={"project": ITB, "feature": "Palette Tools (lib/palette, 41 formats)"})
def palette_export(api, colors, input, count, source, name, format, name_colors, output, overwrite):
    """Write a palette in one of ImageToolbox's palette formats (Adobe ACO/ASE/ACT/ACB, GIMP GPL, RIFF PAL,
    Paint.NET, Krita, Procreate, CorelDraw, Scribus, SVG, Swift/Kotlin code, ...)."""
    if sum(bool(x) for x in (colors, input, source)) != 1:
        raise invalid("Give exactly one of colors, input (image) or source (palette file)")
    fmt = _format_for_output(format, output)
    if source:
        pal, _ = pf.decode(api.ws.policy.check_read(source).read_bytes(), filename=source)
        if name:
            pal.name = name
    else:
        entries: list[tuple[str, tuple[float, ...]]] = []
        if input:
            entries = [(n, c) for n, c in _image_palette(api, input, count)]
        for s in colors:
            m = _NAMED.match(s)
            if not m:
                raise invalid(f"Bad colour entry {s!r}", param="colors")
            try:
                rgba = parse_color(m.group("color").strip())
            except LazyRoomError:
                raise invalid(f"Bad colour entry {s!r}", param="colors") from None
            n = (m.group("name") or "").strip() or (color_name(rgba) if name_colors else "")
            entries.append((n, rgba))
        pal = pf.Palette(name=name, colors=[pf.PaletteColor.rgb(*c[:3], a=c[3] if len(c) > 3 else 1.0, name=n)
                                            for n, c in entries], groups=[])
    if pal.total_color_count == 0:
        raise invalid("The palette has no colours")
    try:
        data = pf.encode(pal, fmt)
    except pf.PaletteFormatError as e:
        raise invalid(f"Cannot encode as {fmt}: {e}", param="format") from None
    p = out_path(api, output, overwrite=overwrite)
    atomic_write(p, data, overwrite=overwrite)
    return {"output": str(p), "format": fmt, "bytes": len(data), "palette": _palette_json(pal)}


@tool("palette_import", "Import palette file", category="color", params=[
    File("input", "", file_kind="any", required=True, description="Palette file"),
    Choice("format", "auto", choices=["auto"] + _DECODABLE,
           description="auto = try the formats matching the extension first, then all others (like ImageToolbox)"),
    Text("swatch_output", "", description="Optional PNG/JPG swatch image of the colours"), Bool("overwrite", False),
], source={"project": ITB, "feature": "Palette Tools (lib/palette, 41 formats)"})
def palette_import(api, input, format, swatch_output, overwrite):
    """Read a palette file: colours (hex, RGBA, original colour space/components), groups and name."""
    p = api.ws.policy.check_read(input)
    try:
        pal, fmt = pf.decode(p.read_bytes(), fmt=None if format == "auto" else format, filename=p.name)
    except pf.PaletteFormatError as e:
        raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"Could not read palette {p.name}: {e}",
                              details={"param": "input"}) from None
    res = {"input": str(p), "format": fmt, "palette": _palette_json(pal)}
    if swatch_output:
        cols = pal.all_colors()
        n = len(cols)
        per = min(n, 16)
        rows = -(-n // per)
        sw = np.zeros((rows * 64, per * 64, 4), F32)
        for i, c in enumerate(cols):
            r, q = divmod(i, per)
            sw[r * 64:(r + 1) * 64, q * 64:(q + 1) * 64] = c.to_rgba()
        res["swatch"] = save(api, sw if sw[..., 3].min() < 1 else sw[..., :3], swatch_output, overwrite=overwrite)
    return res


# =============================================================================
# Shader Studio presets (.itshader)
# =============================================================================

def _user_shader_dir() -> Path:
    return data_dir() / "shaders"


def _shader_files() -> list[tuple[str, Path]]:
    out = [("bundled", p) for p in sorted((package_data_dir() / "shaders").glob("*.itshader"))]
    d = _user_shader_dir()
    if d.is_dir():
        out += [("user", p) for p in sorted(d.glob("*.itshader"))]
    return out


def _parse_itshader(text: str) -> tuple[dict, list[dict]]:
    from ..ops.itb._effects_impl import parse_shader

    _, params, _ = parse_shader(text, "")
    try:
        preset = json.loads(text)
    except json.JSONDecodeError:
        preset = None
    if not isinstance(preset, dict):
        raise invalid("Not an .itshader preset (JSON with version, name, params, shader)")
    return preset, params


def _find_shader(name: str) -> tuple[str, Path, dict]:
    hits = []
    for origin, p in _shader_files():
        try:
            preset = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(preset.get("name", "")).lower() == name.strip().lower() or p.stem.lower() == name.strip().lower():
            hits.append((origin, p, preset))
    if not hits:
        raise invalid(f"No shader preset named {name!r}", param="name",
                      hint="Use action='list' to see the available presets.")
    return hits[-1]  # user presets override bundled ones with the same name


def _sanitized(name: str) -> str:  # ShaderStudioMapper.sanitizedFileName
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "shader"


@tool("shader_presets", "Shader presets (.itshader)", category="presets", params=[
    Choice("action", "list", choices=["list", "show", "import", "export"],
           description="list presets; show one (its JSON is the 'shader' param of filter.shader); "
                       "import a .itshader file into the user library; export one to a file"),
    Text("name", "", max_length=200, description="Preset name (show / export)"),
    Text("path", "", description="import: .itshader file to add; export: destination file"),
    Bool("overwrite", False),
], source={"project": ITB, "feature": "Shader Studio library (.itshader presets)"})
def shader_presets(api, action, name, path, overwrite):
    """Library of ImageToolbox Shader Studio presets for ``filter.shader``.

    Bundled examples live in the package (ImageToolbox itself ships none); imported presets are stored in
    the LazyRoom data folder (``<data>/shaders``).  Presets are validated like ImageToolbox's ShaderParser.
    """
    if action == "list":
        items = []
        for origin, p in _shader_files():
            try:
                preset, params = _parse_itshader(p.read_text(encoding="utf-8"))
            except LazyRoomError as e:
                items.append({"file": str(p), "origin": origin, "valid": False, "error": e.message})
                continue
            items.append({"name": preset.get("name", p.stem), "origin": origin, "file": str(p),
                          "params": [{"name": q["name"], "type": q["type"]} for q in params]})
        return {"presets": items, "user_library": str(_user_shader_dir()),
                "hint": "action='show' returns the preset JSON to pass as filter.shader's 'shader'."}
    if action == "show":
        origin, p, preset = _find_shader(name)
        text = p.read_text(encoding="utf-8")
        _, params = _parse_itshader(text)
        defaults = {q["name"]: q["default"] for q in preset.get("params", [])}
        return {"name": preset.get("name"), "origin": origin, "file": str(p), "preset": preset,
                "filter_shader_params": {"shader": text, "values": json.dumps(defaults)}}
    if action == "import":
        if not path:
            raise invalid("action='import' needs path", param="path")
        text = api.ws.policy.check_read(path).read_text(encoding="utf-8")
        preset, _ = _parse_itshader(text)
        d = _user_shader_dir()
        d.mkdir(parents=True, exist_ok=True)
        dst = api.ws.policy.check_write(d / f"{_sanitized(str(preset['name']))}.itshader", overwrite=overwrite)
        atomic_write(dst, text.encode("utf-8"), overwrite=overwrite)
        return {"imported": preset["name"], "file": str(dst)}
    # export
    if not path:
        raise invalid("action='export' needs path", param="path")
    _, p, preset = _find_shader(name)
    dst = out_path(api, path, overwrite=overwrite)
    atomic_write(dst, p.read_bytes(), overwrite=overwrite)
    return {"exported": preset.get("name"), "output": str(dst)}


# =============================================================================
# Collage layouts (lib/collages)
# =============================================================================

@tool("collage_layouts", "Collage layouts (list)", category="compose", params=[
    Int("images", 0, min=0, max=20, description="Only layouts for this many images (0 = all)"),
], source={"project": ITB, "feature": "Collage Maker layouts (lib/collages)"})
def collage_layouts(api, images):
    """ImageToolbox's collage layout catalogue: ids, image count and frame geometry (normalised)."""
    from . import _collage_layouts as cl

    lays = [l for l in cl.load_layouts() if not images or l["images"] == images]
    return {"count": len(lays), "layouts": [{"id": l["id"], "images": l["images"],
                                             "frames": [f["bound"] for f in l["frames"]]} for l in lays]}


@tool("collage_layout", "Collage from layout", category="compose", params=[
    StringList("inputs", [], min_items=1, file_kind="image"),
    Text("layout", "", max_length=64, description="Layout id from collage_layouts (empty = first layout for "
                                                    "the number of inputs)"),
    Int("width", 2000, min=16, max=20000), Int("height", 2000, min=16, max=20000),
    Float("spacing", -1, min=-1, max=1000, unit="px",
          description="Space between frames in output px (-1 = ImageToolbox default 10)"),
    Float("corner_radius", -1, min=-1, max=1000, unit="px",
          description="Frame corner radius in output px (-1 = ImageToolbox default 0)"),
    Color("background", "#FFFFFF"), Text("output", ""), Bool("overwrite", False),
], source={"project": ITB, "feature": "Collage Maker (lib/collages)"})
def collage_layout(api, inputs, layout, width, height, spacing, corner_radius, background, output, overwrite):
    """Arrange images in one of ImageToolbox's ~300 collage layouts (polygon frames, spacing, rounded corners)."""
    from ..core.image import fit_within
    from . import _collage_layouts as cl

    files = expand_inputs(api, inputs)
    if not files:
        raise invalid("No input images", param="inputs")
    imgs = [fit_within(read_image(api, f)[0], 4000)[0] for f in files]
    lid = layout or next(iter(cl.layout_ids(len(imgs))), "")
    if lid not in cl.layout_ids():
        raise invalid(f"Unknown collage layout {lid!r}", param="layout",
                      hint="Use collage_layouts to list layout ids.")
    kw: dict[str, Any] = {"background": tuple(background)}
    if spacing >= 0:
        kw["spacing"] = spacing
    if corner_radius >= 0:
        kw["corner_radius"] = corner_radius
    canvas = cl.render_collage(imgs, lid, width, height, **kw)
    if canvas[..., 3].min() >= 1:
        canvas = canvas[..., :3]
    return {"layout": lid, "images": len(imgs), "output": save(api, canvas, output, overwrite=overwrite)}


# =============================================================================
# Built-in export profiles (core/domain BuiltInImageExportProfile)
# =============================================================================

@functools.lru_cache(maxsize=1)
def _profiles() -> dict[str, dict[str, Any]]:
    data = json.loads((package_data_dir() / "export_profiles.json").read_text(encoding="utf-8"))
    return {p["id"]: p for p in data["profiles"]}


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if (img.shape[1], img.shape[0]) == (w, h):
        return img
    from ..ops.resample import resample

    return resample(img, max(1, w), max(1, h), "lanczos3", "linear")  # ImageScaleMode.Lanczos3, Linear space


def apply_profile_size(img: np.ndarray, prof: dict[str, Any]) -> np.ndarray:
    """``AndroidImageScaler.scaleImage`` for the profile's resize type."""
    ih, iw = img.shape[:2]
    tw, th = prof["width"], prof["height"]
    if prof["resize"] == "flexible":  # flexibleResize, ResizeAnchor.Default
        mx = max(tw, th)
        if ih >= iw:
            return _resize(img, int(mx * (iw / ih)), mx)
        return _resize(img, mx, int(mx * (ih / iw)))
    # performCenterCrop(upscaleToFitCanvas = true): no downscale, centred on a tw x th canvas
    if (tw, th) == (iw, ih):
        return img
    f = max(1.0, max(tw / iw, th / ih))
    ow, oh = int(iw * f), int(ih * f)
    src = _resize(img, ow, oh)
    if src.shape[2] == 3:
        src = np.dstack([src, np.ones(src.shape[:2], F32)])
    canvas = np.zeros((th, tw, 4), F32)
    x0, y0 = (tw - ow) // 2, (th - oh) // 2
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    cw, ch = min(ow - sx0, tw - dx0), min(oh - sy0, th - dy0)
    canvas[dy0:dy0 + ch, dx0:dx0 + cw] = src[sy0:sy0 + ch, sx0:sx0 + cw]
    return canvas if img.shape[2] == 4 or canvas[..., 3].min() < 1 else canvas[..., :3]


@tool("export_profiles", "Export profiles (list)", category="convert", params=[
    Text("platform", "", max_length=40, description="Filter by platform (Instagram, X, YouTube, Web, ...)"),
], source={"project": ITB, "feature": "Built-in export profiles (61)"})
def export_profiles(api, platform):
    """ImageToolbox's built-in export profiles: size, resize type, format and quality per platform."""
    items = [p for p in _profiles().values() if not platform or p["platform"].lower() == platform.lower()]
    return {"count": len(items), "profiles": items}


@tool("export_profile", "Export with profile", category="convert", params=[
    StringList("inputs", [], min_items=1, file_kind="image"),
    Choice("profile", "web_optimized", choices=list(_profiles())),
    Text("output_dir", ""), Bool("recursive", False), Bool("overwrite", False),
], source={"project": ITB, "feature": "Built-in export profiles (Resize & Convert)"})
def export_profile(api, inputs, profile, output_dir, recursive, overwrite):
    """Export images with an ImageToolbox built-in profile (e.g. instagram_square, youtube_thumbnail).

    Resize types follow ImageToolbox: 'flexible' scales the long side to the profile size; 'center_crop'
    places the image centred on the target canvas at its own scale (upscaled only if it is smaller than the
    canvas) — so large photos are cropped, not downscaled.  Output: <stem>_<profile>.<ext>.
    """
    prof = _profiles()[profile]
    files = expand_inputs(api, inputs, recursive)
    od = api.ws.policy.check_write_dir(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    ext = "." + prof["format"]
    results = []
    for f in files:
        try:
            img, meta = read_image(api, f)
            out = apply_profile_size(img, prof)
            dst = api.ws.policy.check_write(od / f"{Path(f).stem}_{profile}{ext}", overwrite=overwrite)
            data = encode_image(out, ext, quality=prof["quality"], lossless=prof.get("lossless", False),
                                meta=meta, metadata="keep" if prof.get("keep_exif") else "strip")
            atomic_write(dst, data, overwrite=overwrite)
            results.append({"input": f, "output": str(dst), "status": "ok", "width": int(out.shape[1]),
                            "height": int(out.shape[0]), "bytes": len(data)})
        except LazyRoomError as e:
            results.append({"input": f, "status": "failed", "error": e.to_dict()})
    return {"profile": prof, "results": results, "succeeded": sum(r["status"] == "ok" for r in results),
            "failed": sum(r["status"] != "ok" for r in results)}
