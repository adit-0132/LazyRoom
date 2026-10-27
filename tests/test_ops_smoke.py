"""Every registered operation must run with its defaults, be deterministic, and return a sane buffer."""
import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.registry import all_ops
from tests.conftest import synthetic_image

SPECS = all_ops(include_hidden=True)


def _needs_file(spec):
    return any(p.kind == "file" and getattr(p, "required", False) for p in spec.params)


@pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])
def test_op_defaults(spec):
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    if _needs_file(spec):
        pytest.skip("requires an input file parameter")
    img = synthetic_image(96, 72)
    ctx = RenderContext(seed=7)
    vals = spec.validate({})
    out = spec.run(img, ctx, vals)
    assert out.dtype == np.float32 and out.ndim == 3 and out.shape[2] in (3, 4)
    assert np.isfinite(out).all()
    if not spec.geometry:
        assert out.shape[:2] == img.shape[:2], "non-geometry op changed the size"
    if spec.deterministic:
        out2 = spec.run(img, RenderContext(seed=7), vals)
        assert out2.shape == out.shape
        assert np.allclose(out, out2, atol=1e-5), "op is not deterministic for a fixed seed"


@pytest.mark.parametrize("spec", [s for s in SPECS if not s.geometry], ids=lambda s: s.id)
def test_op_rgba_preserves_alpha_shape(spec):
    ok, _ = spec.availability()
    if not ok or _needs_file(spec):
        pytest.skip("unavailable")
    img = synthetic_image(64, 48, alpha=True)
    out = spec.run(img, RenderContext(), spec.validate({}))
    assert out.shape[:2] == img.shape[:2]


def test_describe_all_ops_serializable():
    import json

    for s in SPECS:
        json.dumps(s.describe())
