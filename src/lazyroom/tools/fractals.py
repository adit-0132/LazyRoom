"""Fractal generator tools: port of ImageToolbox "Fractals" (feature/fractal-generation) and its
native Rust fractal-engine (ImageToolboxLibs, Apache-2.0).  See ``lazyroom.fractals``.
"""
from __future__ import annotations

import ast
import json
from typing import Any

import numpy as np

from ..core.errors import invalid
from ..core.params import Bool, Choice, Float, FloatList, Int, Text
from ..fractals.catalog import COLORINGS, ITERATION_POLICIES, SPECS, TYPE_NAMES, describe_type
from ..fractals.palettes import PALETTE_NAMES, PALETTES
from ..fractals.render import max_output_pixels, render_fractal
from .registry import out_path, save, tool

SOURCE = {"project": "ImageToolbox", "feature": "Fractals (feature/fractal-generation)",
          "engine": "ImageToolboxLibs libs/fractal-engine (Rust)", "license": "Apache-2.0"}


def _options(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        val = json.loads(text)
    except json.JSONDecodeError as e:
        try:  # interfaces that pre-parse JSON hand the Text param a stringified Python dict
            val = ast.literal_eval(text)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            raise invalid(f"'options' is not valid JSON: {e.msg}", param="options") from None
    if not isinstance(val, dict):
        raise invalid("'options' must be a JSON object", param="options")
    return val


@tool("fractal", "Generate fractal", category="generate", params=[
    Choice("type", "mandelbrot", choices=TYPE_NAMES, labels={k: s.title for k, s in SPECS.items()},
           description="Fractal formula (75 ImageToolbox types + 2 engine-only: h_tree, hilbert_curve)"),
    Int("width", 1080, min=1, max=8192), Int("height", 1080, min=1, max=8192),
    Int("iterations", 0, min=0, max=8192,
        description="Base iterations (0 = type default). ImageToolbox clamps to 16..8192 and the adaptive "
                    "iteration policy then adds zoom/resolution bonuses"),
    FloatList("center", [], min_length=0, max_length=2,
              description="[x, y] centre in the fractal plane for 2D types; empty = type default"),
    Float("zoom", 1.0, min=1e-6, max=1e300,
          description="2D: visible span = default span / zoom.  3D: camera distance = default distance / zoom"),
    Float("rotation", 0.0, min=-360.0, max=360.0, unit="deg",
          description="View rotation about the image centre (LazyRoom extension; 0 = ImageToolbox framing)"),
    Choice("palette", "classic", choices=PALETTE_NAMES, labels={k: v[0] for k, v in PALETTES.items()},
           description="ImageToolbox gradient palette (options.palette_colors overrides)"),
    Text("options", "{}", multiline=True,
         description="JSON object of type-specific options (power, bailout, julia_real, coefficients, camera_*, "
                     "coloring, supersampling, ...). See the fractal_types tool"),
    Int("seed", 0, min=0, max=2_147_483_647,
        description="Accepted for API uniformity. Every ImageToolbox fractal type is deterministic "
                    "(Buddhabrot uses a Halton sequence), so the seed has no effect"),
    Text("output", ""), Bool("overwrite", False),
], source=SOURCE)
def fractal(api, type, width, height, iterations, center, zoom, rotation, palette, options, seed, output, overwrite):
    """Render any of ImageToolbox's fractal types (escape-time, density/attractor, IFS, geometric curves,
    ray-marched 3D, 3D attractors) with the same parameters, defaults and palettes."""
    opts = _options(options)
    out_path(api, output, overwrite=overwrite)  # fail fast before a long render
    rgba, info = render_fractal(type, width, height, iterations=iterations, center=center or None, zoom=zoom,
                                rotation=rotation, palette=palette, options=opts)
    img = rgba.astype(np.float32) / np.float32(255.0)
    if rgba[..., 3].min() == 255:
        img = img[..., :3]
    return {"output": save(api, img, output, overwrite=overwrite), **info}


@tool("fractal_types", "List fractal types", category="generate", params=[
    Text("type", "", description="Only describe this type (empty = all)"),
], source=SOURCE)
def fractal_types(api, type):
    """List the fractal types of the `fractal` tool with their defaults and option schemas."""
    if type:
        spec = SPECS.get(type)
        if spec is None:
            raise invalid(f"Unknown fractal type '{type}'", param="type", valid=TYPE_NAMES)
        specs = [spec]
    else:
        specs = list(SPECS.values())
    types = []
    for s in specs:
        d = describe_type(s)
        d["max_output_pixels"] = max_output_pixels(s)
        types.append(d)
    return {"count": len(types), "types": types, "palettes": {k: v[0] for k, v in PALETTES.items()},
            "colorings": COLORINGS, "iteration_policies": ITERATION_POLICIES}
