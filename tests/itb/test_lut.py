"""ImageToolbox "LUT" group: Trickle applyLut / applyCubeLut semantics and the 512x512 layout."""
import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.errors import LazyRoomError
from lazyroom.core.registry import all_ops, get_op
from tests.conftest import synthetic_image

SPECS = [s for s in all_ops(include_hidden=True) if s.group == "lut"]
BUNDLED = [s.id for s in SPECS if s.id not in ("filter.lut_512x512", "filter.cube_lut")]


def _run(op_id, img, **params):
    spec = get_op(op_id)
    return spec.run(img, RenderContext(), spec.validate(params))


def _u8(a):
    return np.rint(np.asarray(a) * 255).astype(int)


def _grid_image():
    """Every 6-bit (r, g, b) cell once, at 8-bit value 4*q + 1."""
    q = np.arange(64)
    r, g, b = np.meshgrid(q, q, q, indexing="ij")
    rgb = np.stack([r, g, b], -1).reshape(512, 512, 3)
    return ((rgb * 4 + 1) / 255.0).astype(np.float32), rgb


def test_group_complete():
    assert len(SPECS) == 19 and len(BUNDLED) == 17


@pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])
@pytest.mark.parametrize("strength", [0.0, 1.0])
def test_param_extremes(spec, strength):
    img = synthetic_image(64, 48, alpha=True)
    out = spec.run(img, RenderContext(), spec.validate({"strength": strength}))
    assert out.shape == img.shape and out.dtype == np.float32 and np.isfinite(out).all()


def test_layout_matches_gpuimage_lookup_interpretation():
    """Trickle's x=(b%8)*64+r, y=(b/8)*64+g read == _util.lut_from_hald_or_square (GPUImage shader layout)."""
    from lazyroom.ops._util import lut_from_hald_or_square
    from lazyroom.ops.itb.lut import _bundled

    lut_img = _bundled("kodak")
    lut3d = lut_from_hald_or_square((lut_img / 255.0).astype(np.float32))
    img, q = _grid_image()
    out = _run("filter.kodak", img, strength=1.0)
    expect = lut3d[q[..., 0], q[..., 1], q[..., 2]]
    assert np.abs(out - expect).max() < 1e-6


def test_identity_lut_image(tmp_path):
    from lazyroom.core.io import save_image
    from lazyroom.ops._util import identity_lut, lut_from_hald_or_square
    from lazyroom.ops.itb.lut import _identity_lut512

    ident = _identity_lut512()
    # the generated ImageToolbox identity lookup is the GPUImage identity layout
    assert np.abs(lut_from_hald_or_square(ident / 255.0) - identity_lut(64)).max() < 0.5 / 255
    path = tmp_path / "identity.png"
    save_image((ident / 255.0).astype(np.float32), path)
    img = synthetic_image(96, 72)
    from_file = _run("filter.lut_512x512", img, target_lut_image=str(path))
    default = _run("filter.lut_512x512", img)
    assert np.array_equal(from_file, default)
    q = _u8(img) // 4
    assert np.array_equal(_u8(from_file), np.rint(q * 255 / 63).astype(int))
    assert np.abs(from_file - img).max() <= 4 / 255 + 1e-6


def test_channel_swap_lut_image(tmp_path):
    from lazyroom.core.io import save_image
    from lazyroom.ops.itb.lut import _identity_lut512

    swapped = _identity_lut512()[..., ::-1]
    path = tmp_path / "swap.png"
    save_image((swapped / 255.0).astype(np.float32), path)
    img = synthetic_image(64, 48)
    out = _run("filter.lut_512x512", img, target_lut_image=str(path))
    assert np.abs(out - img[..., ::-1]).max() <= 4 / 255 + 1e-6


def test_lut_image_must_be_square(tmp_path):
    from lazyroom.core.io import save_image

    path = tmp_path / "bad.png"
    save_image(np.zeros((64, 128, 3), np.float32), path)
    with pytest.raises(LazyRoomError):
        _run("filter.lut_512x512", synthetic_image(32, 24), target_lut_image=str(path))


@pytest.mark.parametrize("op_id", BUNDLED)
def test_strength_zero_is_6bit_quantized_input(op_id):
    """Trickle mixes from q*4 (the 6-bit cell), so strength 0 returns the input with its 2 low bits cleared."""
    img = synthetic_image(64, 48)
    out = _run(op_id, img, strength=0.0)
    assert np.array_equal(_u8(out), (_u8(img) // 4) * 4)


@pytest.mark.parametrize("op_id", BUNDLED)
def test_strength_blends_linearly(op_id):
    img = synthetic_image(64, 48)
    full = _u8(_run(op_id, img, strength=1.0))
    half = _u8(_run(op_id, img, strength=0.5))
    base = (_u8(img) // 4) * 4
    assert np.abs(half - (base + full) / 2).max() <= 0.5 + 1e-6
    assert np.abs(full - base).mean() > 1.0  # the look actually changes the image


def test_soft_elegance_variants_use_different_luts():
    img = synthetic_image(64, 48)
    assert not np.array_equal(_run("filter.soft_elegance", img), _run("filter.soft_elegance_variant", img))


def _write_cube(path, n, fn):
    g = np.linspace(0, 1, n)
    lines = ["TITLE \"t\"", f"LUT_3D_SIZE {n}"]
    for b in g:
        for gg in g:
            for r in g:  # red changes fastest
                lines.append("%.6f %.6f %.6f" % tuple(fn(r, gg, b)))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_cube_lut_identity_and_strength(tmp_path):
    cube = _write_cube(tmp_path / "id.cube", 17, lambda r, g, b: (r, g, b))
    img = synthetic_image(80, 60)
    assert np.array_equal(_run("filter.cube_lut", img), img)  # empty path: unchanged
    out0 = _run("filter.cube_lut", img, target_cube_lut_file=cube, strength=0.0)
    assert np.array_equal(_u8(out0), _u8(img))
    out1 = _run("filter.cube_lut", img, target_cube_lut_file=cube, strength=1.0)
    assert np.abs(_u8(out1) - _u8(img)).max() <= 1  # int(c * 255) truncation may drop 1 level


def test_cube_lut_orientation(tmp_path):
    cube = _write_cube(tmp_path / "swap.cube", 9, lambda r, g, b: (b, g, 1 - r))
    img = synthetic_image(80, 60)
    out = _u8(_run("filter.cube_lut", img, target_cube_lut_file=cube))
    u = _u8(img)
    assert np.abs(out[..., 0] - u[..., 2]).max() <= 1
    assert np.abs(out[..., 1] - u[..., 1]).max() <= 1
    assert np.abs(out[..., 2] - (255 - u[..., 0])).max() <= 1


def test_cube_lut_rejects_1d(tmp_path):
    p = tmp_path / "one.cube"
    p.write_text("LUT_1D_SIZE 2\n0 0 0\n1 1 1\n")
    with pytest.raises(LazyRoomError):
        _run("filter.cube_lut", synthetic_image(16, 16), target_cube_lut_file=str(p))
