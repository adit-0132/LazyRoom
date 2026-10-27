import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
_dev_gmic = ROOT / ".tools" / "gmic" / "gmic"
if _dev_gmic.exists() and "LAZYROOM_GMIC" not in os.environ:
    os.environ["LAZYROOM_GMIC"] = str(_dev_gmic)
os.environ.setdefault("LAZYROOM_HOME", str(ROOT / ".pytest_home"))


def synthetic_image(w: int = 160, h: int = 120, alpha: bool = False, seed: int = 0) -> np.ndarray:
    """Deterministic test photo: gradients, colour patches, edges, noise, a 'face-ish' skin patch."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = x / (w - 1)
    img[..., 1] = y / (h - 1)
    img[..., 2] = 1 - (x / (w - 1)) * 0.6
    patches = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (0.9, 0.7, 0.55), (0.5, 0.5, 0.5)]
    pw = w // len(patches)
    for i, c in enumerate(patches):
        img[h * 2 // 3:, i * pw:(i + 1) * pw] = c
    cy, cx = h // 3, w // 2
    disk = (x - cx) ** 2 + (y - cy) ** 2 < (min(w, h) // 6) ** 2
    img[disk] = (0.85, 0.62, 0.5)
    img[h // 2 - 2:h // 2 + 2, :] = 0.05
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    if alpha:
        a = np.clip(1.2 - np.hypot(x - w / 2, y - h / 2) / (0.5 * max(w, h)), 0, 1).astype(np.float32)
        img = np.dstack([img, a])
    return img


@pytest.fixture
def rgb():
    return synthetic_image()


@pytest.fixture
def rgba():
    return synthetic_image(alpha=True)


@pytest.fixture
def photo_path(tmp_path):
    from lazyroom.core.io import save_image

    p = tmp_path / "photo.png"
    save_image(synthetic_image(320, 240), p)
    return p
