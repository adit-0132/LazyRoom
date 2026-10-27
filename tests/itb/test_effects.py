"""ImageToolbox "Effects" group: extremes + semantic checks."""
import json

import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.errors import ErrorCode, LazyRoomError
from lazyroom.core.registry import all_ops, get_op
from lazyroom.ops.itb import _effects_impl as fx
from tests.conftest import synthetic_image

SPECS = [s for s in all_ops(include_hidden=True) if s.group == "effects"]


def _spec(op_id):
    spec = get_op(op_id)
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    return spec


def _run(op_id, img, **values):
    spec = _spec(op_id)
    return spec.run(img, RenderContext(seed=3), spec.validate(values))


def _q(x):
    return np.round(np.clip(x, 0, 1) * 255) / 255


def test_group_complete():
    assert len(SPECS) == 41


def _extreme_values(spec, which):
    vals = {}
    for p in spec.params:
        if p.kind in ("float", "int") and p.min is not None and p.max is not None:
            vals[p.name] = p.min if which == "min" else p.max
    return vals


@pytest.mark.parametrize("which", ["min", "max"])
@pytest.mark.parametrize("spec", SPECS, ids=[s.id for s in SPECS])
def test_extremes(spec, which):
    ok, hint = spec.availability()
    if not ok:
        pytest.skip(f"backend unavailable: {hint}")
    vals = _extreme_values(spec, which)
    for img in (synthetic_image(64, 48), synthetic_image(40, 30, alpha=True)):
        out = spec.run(img, RenderContext(seed=1), spec.validate(vals))
        assert out.dtype == np.float32 and out.ndim == 3 and out.shape[2] in (3, 4)
        assert np.isfinite(out).all()
        assert out.min() >= 0 and out.max() <= 1
        if not spec.geometry:
            assert out.shape[:2] == img.shape[:2]


# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------

def test_morphology_order_relations():
    img = synthetic_image(80, 60)
    q = _q(img)
    dil = _run("filter.dilation", img, size=7)
    ero = _run("filter.erode", img, size=7)
    opn = _run("filter.opening", img, size=7)
    cls = _run("filter.closing", img, size=7)
    assert (dil >= q - 1e-6).all() and (ero <= q + 1e-6).all()
    assert (opn <= q + 1e-6).all() and (cls >= q - 1e-6).all()
    grad = _run("filter.morphological_gradient", img, size=7)
    assert np.allclose(grad, dil - ero, atol=1e-6)
    top = _run("filter.top_hat", img, size=7)
    assert np.allclose(top, q - opn, atol=1e-6)
    black = _run("filter.black_hat", img, size=7)
    assert np.allclose(black, cls - q, atol=1e-6)


def test_morphology_kernels():
    k = fx.morph_kernel(25, True)  # disk of radius 25/3 centred in a 25x25 window
    assert k.shape == (25, 25) and k[12, 12] == 1 and k[12, 12 + 8] == 1 and k[12, 12 + 9] == 0 and k[0, 0] == 0
    assert fx.morph_kernel(4, False).sum() == 16
    img = synthetic_image(40, 30)
    assert np.allclose(_run("filter.dilation", img, size=1), _q(img), atol=1e-6)
    # a single bright pixel dilates to the kernel shape
    dot = np.zeros((41, 41, 3), np.float32)
    dot[20, 20] = 1
    out = _run("filter.dilation", dot, size=25, use_circle_kernel=True)[..., 0]
    assert np.array_equal(out > 0.5, np.pad(k, 8).astype(bool))
    box = _run("filter.dilation", dot, size=5, use_circle_kernel=False)[..., 0]
    assert (box > 0.5).sum() == 25


def test_morphology_rgba_mode_morphs_alpha():
    img = synthetic_image(48, 36, alpha=True)
    out = _run("filter.erode", img, size=9)
    assert out.shape[2] == 4 and out[..., 3].min() <= img[..., 3].min() + 1e-6
    assert (out[..., 3] <= _q(img[..., 3]) + 1e-6).all()


# ---------------------------------------------------------------------------
# Edges / analysis
# ---------------------------------------------------------------------------

def test_canny_binary_gray():
    out = _run("filter.canny", synthetic_image(80, 60))
    assert set(np.unique(out)) <= {0.0, 1.0} and out.max() == 1.0
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])


def test_sobel_flat_is_black_and_edges_found():
    flat = np.full((30, 40, 3), 0.4, np.float32)
    assert _run("filter.sobel_edge_detection", flat).max() < 1e-6
    img = np.zeros((30, 40, 3), np.float32)
    img[:, 20:] = 1
    out = _run("filter.sobel_edge_detection", img)
    assert out[15, 19:21, 0].min() > 0.9 and out[15, :15].max() < 1e-6
    assert np.array_equal(out[..., 0], out[..., 2])


def test_error_level_analysis_normalised():
    out = _run("filter.error_level_analysis", synthetic_image(80, 60), quality=50)
    assert out.min() == 0.0 and abs(out.max() - 1.0) < 1e-6


def test_copy_move_detects_cloned_region():
    img = synthetic_image(160, 120, seed=4)
    img += np.random.default_rng(1).normal(0, 0.05, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    img[70:110, 100:150] = img[10:50, 10:60]
    out = _run("filter.copy_move_detection", img, retain=4, coefficient=1)
    changed = np.abs(out - _q(img)).max(axis=2) > 0.05
    assert changed[75:105, 105:145].mean() > 0.9 and changed[15:45, 15:55].mean() > 0.9
    tiny = synthetic_image(12, 10)
    assert np.allclose(_run("filter.copy_move_detection", tiny), tiny)


def test_java_random_matches_reference():
    # values printed by a JDK: new Random(42).nextInt(256) x3, new Random(0).nextFloat() x3
    assert fx.java_random_next_int(42, 256, 3) == [186, 13, 174]
    assert np.allclose(fx._java_random_floats(0, 3), [0.73096776, 0.831441, 0.24053639], atol=1e-7)


# ---------------------------------------------------------------------------
# jhlabs
# ---------------------------------------------------------------------------

def test_chrome_gray_and_flat_value():
    flat = np.full((10, 12, 3), 0.3, np.float32)
    out = _run("filter.chrome", flat, amount=0.5, exposure=1.0)
    expect = np.floor((1 - np.exp(-(0.5 + np.sin(0.5)))) * 255) / 255
    assert np.allclose(out, expect, atol=1e-6)
    out = _run("filter.chrome", synthetic_image(40, 30))
    assert np.array_equal(out[..., 0], out[..., 1])


def test_contour_draws_on_level_boundaries():
    # horizontal 8-bit ramp: brightness levels of 5 change at 52, 103, 154 and 205; weak edges let the
    # contour colour through (the source is composited over it with the edge strength as alpha)
    ramp = np.repeat((np.arange(256, dtype=np.float32) / 255)[None, :, None], 3, axis=2)
    img = np.repeat(ramp, 12, axis=0)
    out = _run("filter.contour", img, levels=5, scale=1, offset=0, color="#FF0000")
    changed = np.abs(out - img).max(axis=2) > 1e-6
    assert sorted(set(np.nonzero(changed)[1].tolist())) == [52, 103, 154, 205]
    assert changed[1:-1, 52].all() and not changed[0].any() and not changed[-1].any()
    assert out[5, 103, 0] > 0.95 and out[5, 103, 1] < 0.45
    # a strong edge is left unchanged (full-strength composite returns the source)
    step = np.zeros((20, 40, 3), np.float32)
    step[:, 20:] = 0.9
    assert np.allclose(_run("filter.contour", step), _q(step), atol=1e-6)
    assert np.allclose(_run("filter.contour", img, levels=0), img, atol=1e-6)


def test_dissolve_density():
    img = synthetic_image(80, 60)
    out = _run("filter.dissolve", img, density=0.3, softness=0)
    assert out.shape[2] == 4 and set(np.unique(out[..., 3])) <= {0.0, 1.0}
    assert abs(out[..., 3].mean() - 0.3) < 0.05
    assert np.allclose(_run("filter.dissolve", img, density=1, softness=0.5)[..., 3], 1)
    assert np.allclose(_run("filter.dissolve", img, density=0, softness=0)[..., 3], 0)


def test_dog_flat_and_equal_radii():
    flat = np.full((30, 40, 3), 0.5, np.float32)
    assert _run("filter.dog", flat, radius=1, second_radius=3)[1:-1, 1:-1].max() == 0
    img = synthetic_image(60, 40)
    out = _run("filter.dog", img, radius=2, second_radius=6)
    assert abs(out.max() - 1.0) < 1e-6  # normalised


# ---------------------------------------------------------------------------
# aire / Trickle / GPUImage lookup
# ---------------------------------------------------------------------------

def test_convolution_identity_and_q15_saturation():
    img = synthetic_image(40, 30)
    assert np.allclose(_run("filter.convolution_3x3", img), _q(img), atol=1e-6)
    doubled = _run("filter.convolution_3x3", img, kernel=[0, 0, 0, 0, 2, 0, 0, 0, 0])
    assert np.allclose(doubled, _q(img), atol=1e-6)  # Q15 weights saturate at ~1.0
    half = _run("filter.convolution_3x3", img, kernel=[0, 0, 0, 0, 0.5, 0, 0, 0, 0])
    assert np.allclose(half, np.floor(_q(img) * 255 * 0.5 + 0.5) / 255, atol=1 / 255 + 1e-6)


def test_emboss_zero_intensity_is_identity():
    img = synthetic_image(40, 30)
    assert np.allclose(_run("filter.emboss", img, intensity=0), _q(img), atol=1e-6)


def test_threshold_uses_red_channel():
    img = np.zeros((4, 6, 3), np.float32)
    img[:, :3, 0] = 0.8  # red
    img[:, 3:, 2] = 1.0  # blue
    out = _run("filter.threshold", img, level=128)
    assert np.all(out[:, :3] == 1) and np.all(out[:, 3:] == 0)


def test_noise_and_grain():
    img = synthetic_image(50, 40)
    q8 = np.round(img * 255).astype(np.uint8)
    out = np.round(_run("filter.noise", img, threshold=64) * 255).astype(np.uint8)
    assert np.all(out >= q8) and np.array_equal(out | q8, out)
    assert np.allclose(_run("filter.noise", img, threshold=1), _q(img), atol=1e-6)
    assert np.allclose(_run("filter.grain", img, intensity=0), _q(img), atol=1e-6)
    g = _run("filter.grain", np.full((20, 20, 3), 0.5, np.float32), intensity=0.2)
    d = g - 0.5
    assert d.std() > 0.05 and np.allclose(d[..., 0], d[..., 2], atol=1e-6)  # monochrome grain


def test_lookup_without_and_with_lut(tmp_path):
    from lazyroom.core.io import save_image

    img = synthetic_image(40, 30)
    out = _run("filter.lookup", img, intensity=-1)  # unbound texture -> color * (1 - intensity)
    assert np.allclose(out, np.clip(img * 2, 0, 1), atol=1e-5)
    assert np.allclose(_run("filter.lookup", img, intensity=1), 0)
    # identity GPUImage 512x512 LUT
    lut = np.zeros((512, 512, 3), np.float32)
    g = np.arange(64, dtype=np.float32) / 63
    for b in range(64):
        ty, tx = divmod(b, 8)
        lut[ty * 64:(ty + 1) * 64, tx * 64:(tx + 1) * 64, 0] = g[None, :]
        lut[ty * 64:(ty + 1) * 64, tx * 64:(tx + 1) * 64, 1] = g[:, None]
        lut[ty * 64:(ty + 1) * 64, tx * 64:(tx + 1) * 64, 2] = b / 63
    p = tmp_path / "identity.png"
    save_image(lut, p)
    out = _run("filter.lookup", img, intensity=1, lut=str(p))
    assert np.abs(out - img).max() < 0.01


# ---------------------------------------------------------------------------
# G'MIC
# ---------------------------------------------------------------------------

def test_crt_filters_upscale():
    img = synthetic_image(32, 24)
    assert _run("filter.crt_phosphors", img, upscale_factor=3).shape[:2] == (72, 96)
    rgba = synthetic_image(32, 24, alpha=True)
    out = _run("filter.crt_scanlines", rgba, upscale_factor=2)
    assert out.shape == (48, 64, 4)


def test_gmic_random_filters_deterministic():
    img = synthetic_image(48, 36)
    for op in ("filter.cracks", "filter.ghost", "filter.filaments", "filter.dirty"):
        a = _run(op, img)
        b = _run(op, img)
        assert np.allclose(a, b, atol=1e-5), op


def test_filaments_transparent_mode_outputs_alpha():
    out = _run("filter.filaments", synthetic_image(48, 36), color_mode="white_on_transparent")
    assert out.shape[2] == 4 and out[..., 3].min() < 0.5


@pytest.mark.parametrize("command", ["blur 3", "fx_sharpen_texture 1,4,0", "a x", "+mirror x a x",
                                     "repeat 2 { blur 1 } n 0,255", "f \"i*0.5\"", "resize 50%,50%"])
def test_raw_gmic_allows_normal_filters(command):
    img = synthetic_image(48, 36)
    out = _run("filter.raw_gmic", img, command=command)
    assert np.isfinite(out).all() and out.ndim == 3


def test_raw_gmic_blur_changes_image():
    img = synthetic_image(48, 36)
    out = _run("filter.raw_gmic", img, command="blur 3")
    assert out.shape == img.shape and np.abs(out - img).mean() > 0.005


@pytest.mark.parametrize("command", [
    "exec \"touch /tmp/pm_pwned\"",
    "x \"rm -rf ~\"",
    "blur 3 x \"id\"",
    "-x 1,\"id\"",
    "rv x \"id\"",
    "nm a x \"id\"",
    "o /tmp/x.png",
    "output /tmp/x.png",
    "+o[0] out.png",
    "outputn foo",
    "i /etc/passwd",
    "input /etc/passwd",
    "/etc/passwd",
    "blur 3 ../../secret.png",
    "photo.png",
    "~/Pictures",
    "C:\\\\x.png",
    "download http://example.com/a.png",
    "https://example.com/a.png",
    "i http://example.com/a.png",
    "m \"foo : x id\" foo",
    "command /tmp/evil.gmic",
    "shell",
    "parallel \"x id\"",
    "fx_tk_video3D 1",
    "fx_pack 1,2",
    "delete /tmp/file",
    "createdir 1,dir",
    "camera",
    "display",
    "w 100,100",
    "blur $HOME",
    "${\"-x id\"}",
    "v={'x'} $v \"id\"",
    "{`[120]`} \"id\"",
    "(120) {0,t} \"id\"",
    "eval \"run('x id')\"",
    "f \"isfile([47,101,116,99])\"",
    "network 1",
    "update",
    "sample lena",
    "x_life",
    "echo_file /tmp/a,b",
    "",
])
def test_raw_gmic_rejects_dangerous_commands(command):
    spec = get_op("filter.raw_gmic")
    img = synthetic_image(32, 24)
    with pytest.raises(LazyRoomError) as ei:
        spec.run(img, RenderContext(), spec.validate({"command": command}))
    assert ei.value.code == ErrorCode.INVALID_PARAMS


def test_raw_gmic_policy_in_description():
    d = get_op("filter.raw_gmic").describe()["description"]
    assert "Safety policy" in d and "network" in d


# ---------------------------------------------------------------------------
# Shader (headless OpenGL through PySide6)
# ---------------------------------------------------------------------------

IDENTITY_SHADER = """
precision mediump float;
varying highp vec2 textureCoordinate;
uniform sampler2D inputImageTexture;
void main() { gl_FragColor = texture2D(inputImageTexture, textureCoordinate); }
"""

GAIN_SHADER = """
varying highp vec2 textureCoordinate;
uniform sampler2D inputImageTexture;
uniform lowp float gain;
uniform vec4 tint;
uniform vec2 offset;
uniform bool useTint;
uniform int mode;
void main() {
    vec4 c = texture2D(inputImageTexture, textureCoordinate);
    vec3 rgb = c.rgb * gain;
    if (useTint) rgb = mix(rgb, tint.rgb, offset.x);
    if (mode == 1) rgb = vec3(textureCoordinate, 0.0);
    gl_FragColor = vec4(rgb, c.a);
}
"""


def _run_shader(img, **values):
    try:
        return _run("filter.shader", img, **values)
    except LazyRoomError as exc:
        if exc.code == ErrorCode.BACKEND_UNAVAILABLE:
            pytest.skip(f"OpenGL unavailable: {exc}")
        raise


def test_shader_empty_is_noop():
    img = synthetic_image(20, 16)
    assert np.array_equal(_run("filter.shader", img), img)


def test_shader_identity_and_uniforms():
    img = synthetic_image(40, 30)
    out = _run_shader(img, shader=IDENTITY_SHADER)
    assert out.shape == img.shape and np.abs(out - _q(img)).max() < 1.5 / 255
    half = _run_shader(img, shader=GAIN_SHADER, values=json.dumps({"gain": 0.5}))
    assert np.abs(half - _q(img) * 0.5).max() < 1.5 / 255
    coords = _run_shader(img, shader=GAIN_SHADER, values='{"mode": 1}')
    # textureCoordinate (0,0) is the top-left pixel (GPUImage on Android)
    assert coords[0, 0, 0] < 0.05 and coords[0, 0, 1] < 0.05 and coords[-1, -1, 0] > 0.95


def test_shader_itshader_preset():
    preset = {"version": 1, "name": "Tint", "shader": GAIN_SHADER, "params": [
        {"name": "gain", "type": "float", "default": 1.0, "min": 0, "max": 2},
        {"name": "tint", "type": "color", "default": "#FF0000"},
        {"name": "offset", "type": "vec2", "default": [1.0, 0.0]},
        {"name": "useTint", "type": "bool", "default": True},
        {"name": "mode", "type": "int", "default": 0},
    ]}
    img = synthetic_image(24, 20)
    out = _run_shader(img, shader=json.dumps(preset))
    assert np.allclose(out[..., 0], 1, atol=1 / 255) and out[..., 1:].max() < 1.5 / 255
    out = _run_shader(img, shader=json.dumps(preset), values='{"useTint": false}')
    assert np.abs(out - _q(img)).max() < 1.5 / 255


@pytest.mark.parametrize("shader,values", [
    ("void main() { gl_FragColor = vec4(1.0); }", ""),  # missing GPUImage contract
    (IDENTITY_SHADER.replace("gl_FragColor", "gl_FragData[0]"), ""),
    (IDENTITY_SHADER.replace("void main()", "uniform sampler2D other; void main()"), ""),
    (IDENTITY_SHADER.replace("void main()", "uniform float iTime; void main()"), ""),
    (GAIN_SHADER, '{"nope": 1}'),
    (GAIN_SHADER, '{"gain": "high"}'),
    ('{"version": 2, "name": "x", "shader": "void main(){}"}', ""),
])
def test_shader_validation_errors(shader, values):
    spec = get_op("filter.shader")
    with pytest.raises(LazyRoomError) as ei:
        spec.run(synthetic_image(8, 8), RenderContext(), spec.validate({"shader": shader, "values": values}))
    assert ei.value.code == ErrorCode.INVALID_PARAMS


def test_shader_compile_error_reported():
    bad = IDENTITY_SHADER.replace("texture2D(inputImageTexture, textureCoordinate)", "undefined_fn(1.0)")
    try:
        _run("filter.shader", synthetic_image(8, 8), shader=bad)
    except LazyRoomError as exc:
        if exc.code == ErrorCode.BACKEND_UNAVAILABLE:
            pytest.skip("OpenGL unavailable")
        assert exc.code == ErrorCode.INVALID_PARAMS and "compile" in str(exc)
    else:
        pytest.fail("compile error not reported")
