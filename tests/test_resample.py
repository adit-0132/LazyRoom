"""Extended resampling (ops/resample.py, port of pic-scale 0.6.6) and its geometry.resize wiring."""
import time

import cv2
import numpy as np
import pytest

from lazyroom.core.context import RenderContext
from lazyroom.core.registry import get_op
from lazyroom.ops import resample as R
from lazyroom.ops.geometry import resize_buffer
from tests.conftest import synthetic_image

NAMES = list(R.KERNELS)
CONV = [n for n in NAMES if R.KERNELS[n].kind != "nearest"]

RESIZE_FUNCTION = [  # aire ResizeFunction, in enum order 0..48
    "bilinear", "nearest", "cubic", "mitchell_netravalli", "catmull_rom", "hermite", "b_spline", "hann",
    "bicubic", "hamming", "hanning", "blackman", "welch", "quadric", "gaussian", "sphinx", "bartlett",
    "robidoux", "robidoux_sharp", "spline16", "spline36", "spline64", "kaiser", "bartlett_hann", "box",
    "bohman", "lanczos2", "lanczos3", "lanczos4", "lanczos2_jinc", "lanczos3_jinc", "lanczos4_jinc",
    "ewa_hanning", "ewa_robidoux", "ewa_blackman", "ewa_quadric", "ewa_robidoux_sharp", "ewa_lanczos3_jinc",
    "ginseng", "ewa_ginseng", "ewa_lanczos_sharp", "ewa_lanczos4_sharpest", "ewa_lanczos_soft", "haasn_soft",
    "lagrange2", "lagrange3", "lanczos6", "lanczos6_jinc", "area",
]
SCALE_COLOR_SPACE = [  # aire ScaleColorSpace, in enum order 0..18
    "srgb", "lab", "linear", "luv", "sigmoidal", "xyz", "linear_f32_srgb", "linear_f32_rec709",
    "linear_f32_gamma_2_2", "linear_f32_gamma_2_8", "lch", "oklab_srgb", "oklab_rec709", "oklab_gamma_2_2",
    "oklab_gamma_2_8", "jzazbz_srgb", "jzazbz_rec709", "jzazbz_gamma_2_2", "jzazbz_gamma_2_8",
]


def test_tables_match_upstream_enums():
    assert NAMES == RESIZE_FUNCTION
    assert R.SCALE_COLOR_SPACES == SCALE_COLOR_SPACE


@pytest.mark.parametrize("name", NAMES)
def test_down_and_upscale_finite(name):
    img = synthetic_image(96, 72, alpha=True)
    for w, h in [(41, 29), (37, 53), (193, 145), (120, 90)]:
        out = R.resample(img, w, h, name)
        assert out.shape == (h, w, 4) and out.dtype == np.float32
        assert np.isfinite(out).all()
        assert out[..., 3].min() >= 0 and out[..., 3].max() <= 1


@pytest.mark.parametrize("name", NAMES)
def test_constant_image_stays_constant(name):
    rgb = np.full((50, 60, 3), 0.37, np.float32)
    rgba = np.dstack([rgb, np.full((50, 60), 0.6, np.float32)])
    # (75, 62): width scale 0.8 has output centres exactly between two inputs
    for w, h in [(23, 17), (131, 97), (75, 62), (60, 7)]:
        assert np.abs(R.resample(rgb, w, h, name) - 0.37).max() < 1e-5
        out = R.resample(rgba, w, h, name)
        assert np.abs(out[..., :3] - 0.37).max() < 1e-5 and np.abs(out[..., 3] - 0.6).max() < 1e-5


@pytest.mark.parametrize("name", CONV)
def test_weight_rows_normalised(name):
    for n_in, n_out in [(50, 23), (50, 131), (64, 80), (9, 4), (4, 9), (4, 5), (2, 3), (1, 7), (7, 1)]:
        m = R.weight_matrix(n_in, n_out, R.KERNELS[name])
        assert m.shape == (n_out, n_in)
        assert np.allclose(np.asarray(m.sum(axis=1)).ravel(), 1.0, atol=1e-5)


@pytest.mark.parametrize("name", NAMES)
def test_mirror_symmetric(name):
    """Symmetric tap windows and zero-phase EWA smoothing: resampling commutes with flips."""
    img = synthetic_image(61, 47)
    for w, h in [(29, 21), (150, 113), (75, 62)]:
        ref = R.resample(img, w, h, name)
        assert np.abs(R.resample(img[:, ::-1], w, h, name)[:, ::-1] - ref).max() < 1e-5
        assert np.abs(R.resample(img[::-1], w, h, name)[::-1] - ref).max() < 1e-5


SMOOTH = ["bilinear", "cubic", "mitchell_netravalli", "catmull_rom", "hermite", "b_spline", "bicubic", "robidoux",
          "robidoux_sharp", "spline16", "spline36", "spline64", "lanczos2", "lanczos3", "lanczos4", "lanczos6",
          "lanczos3_jinc", "lagrange2", "lagrange3", "ginseng", "ewa_robidoux", "ewa_lanczos3_jinc",
          "ewa_lanczos_sharp", "area"]


@pytest.mark.parametrize("name", SMOOTH)
def test_round_trip_2x(name):
    img = cv2.GaussianBlur(synthetic_image(96, 72), (0, 0), 1.5)
    back = R.resample(R.resample(img, 192, 144, name), 96, 72, name)
    assert np.abs(back - img)[4:-4, 4:-4].mean() < 0.012


def test_area_and_nearest_match_opencv():
    img = synthetic_image(97, 71)
    for w, h in [(48, 35), (31, 23), (13, 11), (194, 142), (150, 100)]:
        assert np.abs(R.resample(img, w, h, "area") - cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)).max() < 1e-5
        assert np.array_equal(R.resample(img, w, h, "nearest"),
                              cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST_EXACT))


def test_kernel_values():
    k = {n: R.KERNELS[n].weights for n in NAMES if R.KERNELS[n].kernel is not None}
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    assert np.allclose(k["lanczos3"](x), [1, 0.6079271, 0, -0.1350949, 0], atol=1e-6)
    assert np.allclose(k["bicubic"](x), [1, 0.5625, 0, -0.0625, 0])
    assert np.allclose(k["catmull_rom"](x), [1, 0.5625, 0, -0.0625, 0])
    assert np.allclose(k["lagrange2"](x), [1, 0.5625, 0, -0.0625, 0])
    assert np.allclose(k["spline36"](np.array([0.0, 1.0, 2.0, 3.0])), [1, 0, 0, 0])
    assert np.allclose(k["spline64"](np.array([0.0, 1.0, 2.0, 3.0, 4.0])), [1, 0, 0, 0, 0])
    assert np.allclose(k["quadric"](x), [0.75, 0.5, 0.125, 0, 0])
    assert np.isclose(k["gaussian"](np.array(0.0)), 1 / (np.sqrt(2 * np.pi) * 0.35))
    # fixed upstream bugs: jinc(0) and the Bartlett / Bartlett-Hann windows
    assert np.isclose(R._jinc(np.array(0.0)), 0.5)
    assert np.isclose(k["lanczos3_jinc"](np.array(0.0)), 0.25)
    assert np.allclose(k["bartlett"](x), [1, 0.5, 0, 0, 0])
    assert np.allclose(k["bartlett_hann"](np.array([0.0, 2.0])), [1, 0])
    # upstream quirk kept: blackman's window is not centred (ring-shaped kernel)
    assert k["blackman"](np.array(0.0)) < 1e-3 < k["blackman"](np.array(1.0))


def test_zero_sum_rows_fall_back_to_nearest():
    # haasn_soft is non-zero only within 0.37 px when upscaling; 4 -> 5 has a centre at 1.5
    m = R.weight_matrix(4, 5, R.KERNELS["haasn_soft"]).toarray()
    assert np.allclose(m.sum(1), 1) and (m.max(1) == 1).all()


@pytest.mark.parametrize("colorspace", SCALE_COLOR_SPACE)
def test_colorspaces(colorspace):
    img = synthetic_image(64, 48, alpha=True)
    rgb = img[..., :3]
    # pure-gamma decoding (x^(1/2.2), x^(1/2.8)) has infinite slope at 0: float32 noise of the
    # Oklab/Jzazbz matrices on exactly-zero channels becomes ~2-3/255 depending on the BLAS/CPU
    # (2.7/255 measured on GitHub's Ubuntu runners; same as upstream f32)
    tol = 2e-2 if colorspace.startswith(("oklab_gamma", "jzazbz_gamma")) else 1e-4
    assert np.abs(R.decode_space(R.encode_space(rgb, colorspace), colorspace) - rgb).max() < tol
    for name in ["lanczos3", "ewa_robidoux", "area", "nearest"]:
        for w, h in [(29, 21), (100, 75)]:
            out = R.resample(img, w, h, name, colorspace)
            assert out.shape == (h, w, 4) and np.isfinite(out).all()
            ref = R.resample(img, w, h, name)
            assert np.abs(out - ref)[..., :3].mean() < 0.1  # same picture, different blending
    const = np.full((30, 40, 3), 0.42, np.float32)
    assert np.abs(R.resample(const, 17, 13, "lanczos3", colorspace) - 0.42).max() < 2e-4


def test_unknown_names_raise():
    from lazyroom.core.errors import LazyRoomError

    img = synthetic_image(16, 12)
    with pytest.raises(LazyRoomError):
        R.resample(img, 8, 6, "nope")
    with pytest.raises(LazyRoomError):
        R.resample(img, 8, 6, "lanczos3", "nope")


def test_resize_buffer_routing():
    img = synthetic_image(80, 60, alpha=True)
    # sRGB + OpenCV names: unchanged OpenCV path
    srgb = resize_buffer(img, 40, 30, "lanczos")
    assert np.array_equal(srgb, resize_buffer(img, 40, 30, "lanczos", "srgb"))
    # linear light + OpenCV name: OpenCV kernel between conversions
    lin = resize_buffer(img, 40, 30, "area", "linear")
    manual = R.decode_space(resize_buffer(R.encode_space(img, "linear"), 40, 30, "area"), "linear")
    assert np.allclose(lin, manual, atol=1e-6) and not np.allclose(lin, resize_buffer(img, 40, 30, "area"))
    # other spaces: OpenCV names go to resample() (lanczos -> lanczos4, auto -> area/lanczos)
    assert np.array_equal(resize_buffer(img, 40, 30, "lanczos", "oklab_srgb"),
                          R.resample(img, 40, 30, "lanczos4", "oklab_srgb"))
    assert np.array_equal(resize_buffer(img, 40, 30, "auto", "lab"), R.resample(img, 40, 30, "area", "lab"))
    assert np.array_equal(resize_buffer(img, 40, 30, "ewa_robidoux"), R.resample(img, 40, 30, "ewa_robidoux"))


def test_resize_op_colorspace_param():
    spec = get_op("geometry.resize")
    assert [p for p in spec.params if p.name == "colorspace"][0].choices == SCALE_COLOR_SPACE
    img = synthetic_image(96, 72)
    for vals in [{}, {"width": 48, "colorspace": "linear"}, {"width": 50, "method": "lanczos3", "colorspace": "lch"},
                 {"scale": 150, "method": "ewa_lanczos_sharp", "colorspace": "jzazbz_srgb"}]:
        out = spec.run(img, RenderContext(), spec.validate(vals))
        assert out.dtype == np.float32 and np.isfinite(out).all()
    assert spec.run(img, RenderContext(), spec.validate({"width": 48})).shape == (36, 48, 3)


def test_speed_lanczos3():
    img = np.random.default_rng(0).random((2000, 3000, 3), dtype=np.float32)
    R.resample(img[:64, :64], 32, 32, "lanczos3")  # warm up imports
    t0 = time.perf_counter()
    out = R.resample(img, 1500, 1000, "lanczos3")
    elapsed = time.perf_counter() - t0
    assert out.shape == (1000, 1500, 3)
    assert elapsed < 3.0, f"lanczos3 3000x2000 -> 1500x1000 took {elapsed:.2f}s"
