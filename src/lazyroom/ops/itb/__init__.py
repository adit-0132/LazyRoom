"""ImageToolbox filter catalogue (ported). One module per ImageToolbox UI group.

``LAZYROOM_ITB_ONLY=group1,group2`` restricts which groups are imported
(development isolation while groups are being ported in parallel).
"""
import importlib
import os

GROUP_MODULES = ["simple", "color", "lut", "light", "enhancement", "artistic", "frames", "effects",
                 "blur", "pixelation", "distortion", "glitch", "geometry", "dithering"]

_only = {g.strip() for g in os.environ.get("LAZYROOM_ITB_ONLY", "").split(",") if g.strip()}

for _m in GROUP_MODULES:
    if _only and _m not in _only:
        continue
    try:
        importlib.import_module(f"{__name__}.{_m}")
    except ModuleNotFoundError as _e:  # group module not created yet
        if _e.name != f"{__name__}.{_m}":
            raise
