"""Subject segmentation for the 'subject' mask type (uses a downloaded REMOVE_BG model)."""
from __future__ import annotations

import numpy as np

from ..core.errors import ErrorCode, LazyRoomError


def subject_mask(img: np.ndarray, ctx, model: str | None = None) -> np.ndarray:
    """Soft foreground mask (H, W) in 0..1 from a background-removal model.

    ``model`` picks a specific catalogue entry; otherwise the task default is used
    when installed, else the smallest installed background-removal model."""
    from . import models, processors

    if model:
        name = models.resolve_name(model)
        if models.MODELS[name].get("task") != "remove_background":
            raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"'{name}' is not a background-removal model",
                                  details={"models": models.models_for_task("remove_background")})
        candidates = [name]
    else:
        default = models.default_model("remove_background")
        rest = sorted(models.models_for_task("remove_background"), key=lambda n: int(models.MODELS[n].get("size") or 0))
        candidates = [default] + [n for n in rest if n != default]
    for name in candidates:
        if models.model_available(name):
            return processors.background_mask(name, models.model_path(name), np.ascontiguousarray(img[..., :3]), ctx)
    if model:
        models.require_model(candidates[0])  # raises with the exact download command
    default = models.default_model("remove_background")
    raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, "No subject-segmentation (background removal) model is installed",
                          details={"models": models.models_for_task("remove_background")},
                          hint=f"Download one with: {models.download_command(default)} "
                               "(see 'lazyroom models list' for alternatives)")
