"""``generate.texture``: overlay an ImageToolbox texture preset onto the image.

The texture is rendered at the full-resolution-equivalent size (image size /
render scale / texture ``scale``) and resampled to the buffer, so previews on
proxies show the same texture as the final export.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core.blend import BLEND_MODES, layer_over
from ..core.params import Choice, Float, Int, Text
from ..core.registry import op
from ..textures import SEED_NOTE, get_preset, parse_options, preset_ids
from ._util import F32

MAX_TEXTURE_PIXELS = 64_000_000  # safety cap for the full-resolution-equivalent render


@op("generate.texture", "Texture (ImageToolbox)", params=[
    Choice("preset", "brushed_metal", choices=preset_ids(), description="Texture preset id (texture_presets tool)"),
    Text("options", "{}", description="JSON object of preset-specific options (see texture_presets)"),
    Float("scale", 1.0, min=0.05, max=10.0, step=0.05,
          description="Texture zoom: the texture is rendered at image size / scale and resampled"),
    Float("opacity", 1.0, min=0, max=1, step=0.01),
    Choice("blend", "normal", choices=list(BLEND_MODES)),
    Int("seed", 0, min=0, max=1_000_000, description=SEED_NOTE),
], alpha="full", tags=["texture", "procedural", "noise", "pattern", "material", "imagetoolbox", "overlay",
                       "generate", "3d"],
    source={"project": "ImageToolbox", "feature": "Texture Generation", "license": "Apache-2.0",
            "url": "https://github.com/T8RIN/ImageToolbox"})
def texture(img, ctx, preset, options, scale, opacity, blend, seed):
    """Composite one of ImageToolbox's 170+ procedural textures (materials, nature, space, patterns,
    ray-marched 3D shapes, G'MIC) over the image with a blend mode; transparent texture areas keep the image."""
    h, w = img.shape[:2]
    p = get_preset(preset)
    opts = p.validate(parse_options(options))
    f = 1.0 / (max(float(ctx.scale), 1e-6) * float(scale))
    tw, th = max(1, int(round(w * f))), max(1, int(round(h * f)))
    if tw * th > MAX_TEXTURE_PIXELS:
        k = (MAX_TEXTURE_PIXELS / (tw * th)) ** 0.5
        tw, th = max(1, int(tw * k)), max(1, int(th * k))
    ctx.check()
    layer = p.render(tw, th, seed, resolve_file=lambda path: ctx.resolve_file(path, "image"), **opts)
    if (tw, th) != (w, h):
        interp = cv2.INTER_AREA if tw * th > w * h else cv2.INTER_LINEAR
        if layer.shape[2] == 4:  # resample premultiplied to avoid dark fringes around transparent areas
            pm = np.dstack([layer[..., :3] * layer[..., 3:4], layer[..., 3]])
            pm = cv2.resize(pm, (w, h), interpolation=interp)
            a = np.clip(pm[..., 3:4], 0, 1)
            layer = np.dstack([np.where(a > 1e-6, pm[..., :3] / np.maximum(a, 1e-6), 0.0), a]).astype(F32)
        else:
            layer = cv2.resize(layer, (w, h), interpolation=interp)
    if layer.shape[2] == 3:
        layer = np.dstack([layer, np.ones((h, w), F32)])
    return layer_over(img, np.clip(layer, 0, 1).astype(F32), blend, opacity)
