"""Erase / restore background by drawing (ImageToolbox ``feature/erase-background``, ``BitmapEraser``).

ImageToolbox draws the user's paths on the image with ``BlendMode.Clear`` (erase:
anti-aliased, optionally softened with a ``BlurMaskFilter``) or, in "recover"
mode, paints the original image back through an ``ImageShader`` with SrcOver.
Here the drawn area is a LazyRoom mask spec (brush strokes, lasso polygons,
rectangles, ellipses... see :mod:`lazyroom.core.masks`), whose value is the
paint coverage ``c`` in 0..1:

* erase   (``Clear`` with coverage ``c``):  alpha' = alpha * (1 - c)
* restore (SrcOver of the opaque original): alpha' = alpha + c * (1 - alpha)

Colour channels are left untouched, so restoring brings back the pixels that
earlier alpha-only steps (e.g. ``ai.remove_background``) hid.  Auto-erase is
``ai.remove_background``; "trim image" is ``geometry.trim`` with tolerance 0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.errors import invalid
from ..core.masks import SHAPE_TYPES, render_mask, validate_mask
from ..core.params import Choice, Param
from ..core.registry import op

F32 = np.float32


@dataclass
class MaskSpec(Param):
    """A mask spec object (``{"shapes": [...], "invert": false, "feather": 0, "density": 1}``) or null."""

    def __post_init__(self) -> None:
        self.kind = "mask"

    def coerce(self, value: Any) -> dict[str, Any] | None:
        if isinstance(value, str):
            if not value.strip():
                return None
            try:
                value = json.loads(value)
            except ValueError:
                raise invalid(f"Parameter '{self.name}' must be a mask object or JSON", param=self.name) from None
        return validate_mask(value)

    def json_schema(self) -> dict[str, Any]:
        return {"type": ["object", "null"], **self._schema_base(),
                "description": "Mask spec: {'shapes': [{'type': ...}], 'invert', 'feather', 'density'}; shape types: "
                               + ", ".join(SHAPE_TYPES)}


@op("draw.erase_to_alpha", "Erase / restore background (draw)", params=[
    MaskSpec("mask", None, description="Area to paint (e.g. brush strokes or a lasso polygon); null = no change"),
    Choice("mode", "erase", choices=["erase", "restore"],
           description="erase: make the painted area transparent; restore: make it opaque again"),
], alpha="full", tags=["erase", "eraser", "background", "transparent", "restore", "recover", "cutout", "brush"],
    source={"project": "ImageToolbox", "feature": "Erase Background (BitmapEraser)"})
def erase_to_alpha(img, ctx, mask, mode):
    """Paint transparency into the image (erase) or paint it back (restore) through a mask spec."""
    if not mask:
        return img
    cov = render_mask(mask, img, ctx)
    a = img[..., 3] if img.shape[2] == 4 else np.ones(img.shape[:2], F32)
    a = a * (1 - cov) if mode == "erase" else a + cov * (1 - a)
    return np.dstack([img[..., :3], np.clip(a, 0, 1)]).astype(F32)
