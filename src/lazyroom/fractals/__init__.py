"""Fractal generation: port of ImageToolbox "Fractals" (feature/fractal-generation) and the native Rust
fractal-engine from ImageToolboxLibs (both Apache-2.0, (c) T8RIN), re-implemented with numba.

* :mod:`.catalog`  - the 77 fractal types (75 exposed by ImageToolbox + 2 engine-only) with their
  ImageToolbox defaults, capabilities and option schemas.
* :mod:`.palettes` - ImageToolbox ``GradientPalette`` presets (76).
* :mod:`.render`   - parameter normalisation / render planning and the engine entry point.
* ``_escape`` / ``_density`` / ``_raymarch`` - numba kernels (2D per-pixel, density/IFS/geometry, 3D).
"""
from .catalog import SPECS, TYPE_NAMES, describe_type
from .palettes import PALETTE_NAMES, PALETTES
from .render import EngineParams, render_engine, render_fractal

__all__ = ["SPECS", "TYPE_NAMES", "describe_type", "PALETTE_NAMES", "PALETTES", "EngineParams", "render_engine",
           "render_fractal"]
