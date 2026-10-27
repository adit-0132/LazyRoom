"""Texture generation tools (port of ImageToolbox's "Texture Generation", 170+ presets).

``texture`` renders one preset to a file; ``texture_presets`` lists the presets
with their categories, titles and option schemas.  Preset options are passed as
a JSON object and validated against the preset's own schema.
"""
from __future__ import annotations

from collections import Counter

from ..core.params import Bool, Choice, Int, Text, params_to_json
from ..textures import SEED_NOTE, get_preset, list_presets, parse_options, preset_ids
from .registry import out_path, save, tool

_SOURCE = {"project": "ImageToolbox", "feature": "Texture Generation", "license": "Apache-2.0",
           "url": "https://github.com/T8RIN/ImageToolbox"}


@tool("texture", "Generate texture", category="generate", params=[
    Choice("preset", "brushed_metal", choices=preset_ids(),
           description="Texture preset id (see texture_presets)"),
    Int("width", 1024, min=1, max=16000), Int("height", 1024, min=1, max=16000),
    Int("seed", 0, min=0, max=1_000_000, description=SEED_NOTE),
    Text("options", "{}", description="JSON object of preset-specific options (see texture_presets); "
                                      "omitted options use ImageToolbox's defaults"),
    Text("output", ""), Bool("overwrite", False),
], source=_SOURCE)
def texture(api, preset, width, height, seed, options, output, overwrite):
    """Render an ImageToolbox texture preset (noise materials, nature, space, patterns, 3D shapes, G'MIC) to a file."""
    p = get_preset(preset)
    opts = p.validate(parse_options(options))
    out_path(api, output, overwrite=overwrite)  # fail fast before rendering
    img = p.render(width, height, seed, resolve_file=lambda path: api.ws.policy.check_read(path), **opts)
    return {"output": save(api, img, output, overwrite=overwrite), "preset": p.id, "title": p.title,
            "backend": p.backend, "seed": seed, "options": params_to_json(p.params, opts)}


@tool("texture_presets", "List texture presets", category="generate", params=[
    Text("query", "", description="Optional words matched against id, title, category and backend "
                                   "(e.g. 'space', 'wood', 'raymarch')"),
], source=_SOURCE)
def texture_presets(api, query):
    """List texture presets with category, title, backend, availability and option schema.

    Up to 12 matches return the full JSON Schema of each preset's options; broader listings return a
    compact schema per option (type, default, range or choices) to keep the response small."""
    presets = list_presets(query or None)
    res = {"count": len(presets), "categories": dict(Counter(p["category"] for p in presets)),
           "backends": dict(Counter(p["backend"] for p in presets)), "seed": SEED_NOTE}
    if len(presets) > FULL_SCHEMA_LIMIT:
        presets = [_compact(p) for p in presets]
        res["hint"] = "Query a preset id (e.g. query='brick') for its full JSON option schema and notes."
    res["presets"] = presets
    return res


FULL_SCHEMA_LIMIT = 12
_COMPACT_KEYS = ("type", "default", "min", "max", "choices")


def _compact(p: dict) -> dict:
    d = {k: p[k] for k in ("id", "title", "category", "backend", "status", "available") if k in p}
    if "requires" in p:
        d["requires"] = p["requires"]
    d["options"] = {q["name"]: {k: q[k] for k in _COMPACT_KEYS if q.get(k) is not None} for q in p["params"]}
    return d
