"""ImageToolbox "Blur" group filters.

Each op ports the code path ImageToolbox actually runs (aire -> libblur
0.19.3 / aire C++, Trickle C++, the native jhlabs port of ImageToolboxLibs,
GPUImage, G'MIC).  Kernels live in ``_blur_impl``; see
docs/dev/port_reports/blur.md for the per-filter fidelity notes.
"""
from __future__ import annotations

import math

import numpy as np

from ...core.params import Choice, Float, Int
from . import _blur_impl as B
from ._common import gmic_cmd, gmic_pipeline, itb_op, run_gmic

GROUP = "blur"
F32 = np.float32


def _edge_param(default: str = "reflect_101") -> Choice:
    return Choice("edge_mode", default, choices=B.EDGE_MODES, labels=B.EDGE_LABELS, title="Edge mode",
                  description="How pixels outside the image are synthesised")


def _tf_param() -> Choice:
    return Choice("transfer_function", "srgb", choices=B.TRANSFER_FUNCS, labels=B.TRANSFER_LABELS,
                  title="Transfer function", description="Blur in linear light obtained with this transfer curve")


def _spx(ctx, v: float) -> int:
    """Scale a full-resolution pixel count for preview proxies (Kotlin rounding)."""
    return B.kround(ctx.px(v))


def _odd_px(ctx, k: int) -> int:
    return k if ctx.scale == 1.0 else max(1, B.nearest_odd(ctx.px(k)))


# ---------------------------------------------------------------------------
# libblur box family
# ---------------------------------------------------------------------------

@itb_op("BoxBlur", "Box blur", GROUP, [
    Int("radius", 10, min=0, max=100, unit="px", title="Radius", description="Kernel size is 2*radius+1"),
], upstream="aire", alpha="full", tags=["blur", "box", "mean"])
def box_blur(img, ctx, radius):
    """libblur ``box_blur`` (running sum over 2r taps, clamped edges)."""
    k = 2 * max(0, _spx(ctx, radius)) + 1
    return B.run_premul(img, lambda d: B.lb_box(d, k, k))


@itb_op("TentBlur", "Tent Blur", GROUP, [
    Int("sigma", 10, min=1, max=300, unit="px", title="Sigma", description="Rounded to the nearest odd value"),
], upstream="aire", alpha="full", tags=["blur", "tent", "triangle"])
def tent_blur(img, ctx, sigma):
    """libblur ``tent_blur``: two box passes sized by ``create_box_gauss``."""
    s = ctx.px(B.nearest_odd(sigma))
    return B.run_premul(img, lambda d: B.lb_clt(d, s, 2))


@itb_op("GaussianBoxBlur", "Gaussian Box Blur", GROUP, [
    Int("sigma", 10, min=1, max=300, unit="px", title="Sigma"),
], upstream="aire", alpha="full", tags=["blur", "gaussian", "box"])
def gaussian_box_blur(img, ctx, sigma):
    """libblur ``gaussian_box_blur``: three box passes sized by ``create_box_gauss``."""
    s = ctx.px(sigma)
    return B.run_premul(img, lambda d: B.lb_clt(d, s, 3))


@itb_op("NativeStackBlur", "Native Stack Blur", GROUP, [
    Int("radius", 25, min=3, max=250, unit="px", title="Radius"),
], upstream="aire", alpha="full", tags=["blur", "stack blur", "fast"])
def native_stack_blur(img, ctx, radius):
    """libblur ``stack_blur``: triangular kernel (r+1-|i|)/(r+1)^2, clamped edges."""
    r = max(1, _spx(ctx, min(max(radius, 1), 1449)))
    return B.run_premul(img, lambda d: B.triangle_blur(d, r))


# ---------------------------------------------------------------------------
# libblur linear-light variants
# ---------------------------------------------------------------------------

@itb_op("LinearBoxBlur", "Linear Box Blur", GROUP, [
    Int("radius", 10, min=1, max=300, unit="px", title="Radius"),
    _tf_param(),
], upstream="aire", alpha="full", tags=["blur", "box", "linear light"])
def linear_box_blur(img, ctx, radius, transfer_function):
    k = 2 * max(0, _spx(ctx, radius)) + 1
    return B.run_premul(img, lambda d: B.in_transfer(d, transfer_function, lambda x: B.lb_box(x, k, k)))


@itb_op("LinearTentBlur", "Linear Tent Blur", GROUP, [
    Int("sigma", 11, min=1, max=300, unit="px", title="Sigma", description="Rounded to the nearest odd value"),
    _tf_param(),
], upstream="aire", alpha="full", tags=["blur", "tent", "linear light"])
def linear_tent_blur(img, ctx, sigma, transfer_function):
    s = ctx.px(B.nearest_odd(sigma))
    return B.run_premul(img, lambda d: B.in_transfer(d, transfer_function, lambda x: B.lb_clt(x, s, 2)))


@itb_op("LinearGaussianBoxBlur", "Linear Gaussian Box Blur", GROUP, [
    Int("sigma", 10, min=1, max=300, unit="px", title="Sigma"),
    _tf_param(),
], upstream="aire", alpha="full", tags=["blur", "gaussian", "box", "linear light"])
def linear_gaussian_box_blur(img, ctx, sigma, transfer_function):
    s = ctx.px(sigma)
    return B.run_premul(img, lambda d: B.in_transfer(d, transfer_function, lambda x: B.lb_clt(x, s, 3)))


@itb_op("LinearStackBlur", "Linear Stack Blur", GROUP, [
    Int("radius", 10, min=1, max=300, unit="px", title="Radius"),
    _tf_param(),
], upstream="aire", alpha="full", tags=["blur", "stack blur", "linear light"])
def linear_stack_blur(img, ctx, radius, transfer_function):
    r = max(1, _spx(ctx, min(radius, 2000)))
    return B.run_premul(img, lambda d: B.in_transfer(d, transfer_function, lambda x: B.triangle_blur(x, r)))


@itb_op("LinearFastGaussianBlur", "Linear Fast Gaussian Blur", GROUP, [
    Int("radius", 10, min=1, max=300, unit="px", title="Radius", description="libblur clamps the radius to 255"),
    _tf_param(),
    _edge_param(),
], upstream="aire", alpha="full", tags=["blur", "gaussian", "fast", "linear light"])
def linear_fast_gaussian_blur(img, ctx, radius, transfer_function, edge_mode):
    r = max(1, _spx(ctx, min(radius, 255)))
    return B.run_premul(img, lambda d: B.in_transfer(
        d, transfer_function, lambda x: B.fast_gaussian(x, r, edge_mode, 2)))


@itb_op("LinearFastGaussianBlurNext", "Linear Fast Gaussian Blur Next", GROUP, [
    Int("radius", 10, min=1, max=300, unit="px", title="Radius", description="libblur clamps the radius to 152"),
    _tf_param(),
    _edge_param(),
], upstream="aire", alpha="full", tags=["blur", "gaussian", "fast", "linear light"])
def linear_fast_gaussian_blur_next(img, ctx, radius, transfer_function, edge_mode):
    r = max(1, _spx(ctx, min(radius, 152)))
    return B.run_premul(img, lambda d: B.in_transfer(
        d, transfer_function, lambda x: B.fast_gaussian(x, r, edge_mode, 3)))


@itb_op("LinearGaussianBlur", "Linear Gaussian Blur", GROUP, [
    Int("strength", 25, min=3, max=600, unit="px", title="Strength",
        description="Kernel size (rounded to the nearest odd value)"),
    Float("sigma", 10, min=1, max=50, step=0.01, unit="px", title="Sigma"),
    _edge_param(),
    _tf_param(),
], upstream="aire", alpha="full", tags=["blur", "gaussian", "linear light"])
def linear_gaussian_blur(img, ctx, strength, sigma, edge_mode, transfer_function):
    k = _odd_px(ctx, B.nearest_odd(strength))
    kern = B.gaussian_kernel_1d(k, ctx.px(sigma))
    return B.run_premul(img, lambda d: B.in_transfer(
        d, transfer_function, lambda x: B.sep_correlate(x, kern, kern, edge_mode)))


# ---------------------------------------------------------------------------
# Gaussian family
# ---------------------------------------------------------------------------

@itb_op("GaussianBlur", "Gaussian blur", GROUP, [
    Int("radius", 25, min=0, max=100, unit="px", title="Radius", description="Kernel size is 2*radius+1"),
    Float("sigma", 10, min=1, max=100, step=0.01, unit="px", title="Sigma"),
    _edge_param(),
], upstream="aire", alpha="full", tags=["blur", "gaussian", "soften"])
def gaussian_blur(img, ctx, radius, sigma, edge_mode):
    """libblur ``gaussian_blur`` (separable sampled gaussian, kernel 2r+1)."""
    k = 2 * max(0, _spx(ctx, radius)) + 1
    if k == 1:
        return img
    kern = B.gaussian_kernel_1d(k, ctx.px(sigma))
    return B.run_premul(img, lambda d: B.sep_correlate(d, kern, kern, edge_mode))


@itb_op("FastGaussianBlur2D", "Fast Gaussian Blur 2D", GROUP, [
    Int("radius", 10, min=1, max=300, unit="px", title="Radius"),
    _edge_param(),
], upstream="aire", alpha="full", tags=["blur", "gaussian", "fast"])
def fast_gaussian_blur_2d(img, ctx, radius, edge_mode):
    """libblur ``fast_gaussian`` (2nd-degree integrated box filter, in place)."""
    r = max(1, _spx(ctx, min(radius, 319)))
    return B.run_premul(img, lambda d: B.fast_gaussian(d, r, edge_mode, 2))


@itb_op("FastGaussianBlur3D", "Fast Gaussian Blur 3D", GROUP, [
    Int("radius", 10, min=1, max=500, unit="px", title="Radius", description="libblur clamps the radius to 280"),
    _edge_param(),
], upstream="aire", alpha="full", tags=["blur", "gaussian", "fast"])
def fast_gaussian_blur_3d(img, ctx, radius, edge_mode):
    """libblur ``fast_gaussian_next`` (3rd-degree integrated box filter, in place)."""
    r = max(1, _spx(ctx, min(radius, 280)))
    return B.run_premul(img, lambda d: B.fast_gaussian(d, r, edge_mode, 3))


@itb_op("FastGaussianBlur4D", "Fast Gaussian Blur 4D", GROUP, [
    Float("radius", 10, min=1, max=100, step=0.01, unit="px", title="Radius"),
], upstream="aire", alpha="full", tags=["blur", "gaussian", "fast"])
def fast_gaussian_blur_4d(img, ctx, radius):
    """aire ``gaussianApproximation4D``: 4th-degree integrated filter on R, G, B (alpha untouched)."""
    r = max(1, _spx(ctx, B.kround(radius)))
    return B.run_premul_rgb(img, lambda d: B.fast_gaussian_4d(d, r))


# ---------------------------------------------------------------------------
# Bilateral, median, motion, poisson
# ---------------------------------------------------------------------------

@itb_op("BilaterialBlur", "Bilaterial blur", GROUP, [
    Int("radius", 12, min=1, max=50, unit="px", title="Radius", description="Window size is 2*radius+1"),
    Float("sigma", 10, min=1, max=100, step=0.1, title="Sigma",
          description="Intensity (range) sigma on 0..1 values - aire passes the two sigmas swapped"),
    Float("spatial_sigma", 3, min=1, max=100, step=0.1, unit="px", title="Spatial Sigma",
          description="Acts as the spatial gaussian sigma (in pixels) after aire's argument swap"),
    _edge_param(),
], upstream="aire", alpha="full", tags=["blur", "bilateral", "edge preserving", "denoise"])
def bilaterial_blur(img, ctx, radius, sigma, spatial_sigma, edge_mode):
    """libblur ``bilateral_filter`` (per-channel range weights, square window)."""
    k = 2 * max(0, _spx(ctx, radius)) + 1
    return B.run_premul(img, lambda d: B.bilateral(d, k, ctx.px(spatial_sigma), sigma, edge_mode))


@itb_op("FastBilaterialBlur", "Fast Bilaterial Blur", GROUP, [
    Int("size", 11, min=1, max=200, title="Size",
        description="Grid-space kernel size (rounded to the nearest odd value)"),
    Float("sigma", 10, min=1, max=100, step=0.1, title="Sigma", description="Spatial sigma of the bilateral grid"),
    Float("spatial_sigma", 3, min=1, max=100, step=0.1, title="Spatial Sigma",
          description="Range sigma of the bilateral grid (ImageToolbox labels it 'Spatial Sigma')"),
], upstream="aire", alpha="full", tags=["blur", "bilateral", "edge preserving", "fast"])
def fast_bilaterial_blur(img, ctx, size, sigma, spatial_sigma):
    """libblur ``fast_bilateral_filter`` (bilateral grid, one plane per channel)."""
    k = B.nearest_odd(size)

    def fn(d):
        return np.dstack([B.fast_bilateral_plane(d[..., c], k, sigma, spatial_sigma) for c in range(d.shape[2])])

    return B.run_premul(img, fn)


@itb_op("MedianBlur", "Median Blur", GROUP, [
    Int("radius", 10, min=0, max=100, unit="px", title="Radius",
        description="aire passes 2*radius+1 to libblur as its radius: the window is (4*radius+3)^2"),
], upstream="aire", alpha="full", tags=["blur", "median", "denoise"])
def median_blur(img, ctx, radius):
    """libblur ``median_blur`` with radius 2r+1 (aire passes the kernel size as libblur's radius)."""
    big_r = 2 * max(0, _spx(ctx, radius)) + 1
    return B.run_premul(img, lambda d: B.median_libblur(d, big_r))


@itb_op("MotionBlur", "Motion Blur", GROUP, [
    Int("size", 25, min=0, max=201, unit="px", title="Size", description="Rounded to the nearest odd value"),
    Float("angle", 45, min=0, max=360, step=0.01, title="Angle", description="Degrees"),
    _edge_param(),
], upstream="aire", alpha="full", tags=["blur", "motion", "directional"])
def motion_blur(img, ctx, size, angle, edge_mode):
    """libblur ``motion_blur``: Bresenham line kernel + 2D filter."""
    k = _odd_px(ctx, B.nearest_odd(size))
    kern = B.motion_kernel(k, angle)
    return B.run_premul(img, lambda d: B.correlate2d(d, kern, edge_mode))


@itb_op("PoissonBlur", "Poisson Blur", GROUP, [
    Int("radius", 10, min=3, max=200, unit="px", title="Radius", description="Kernel size is 2*radius+1"),
], upstream="aire", alpha="full", tags=["blur", "poisson", "random"])
def poisson_blur(img, ctx, radius):
    """aire ``poissonBlur``: random Poisson(k) 1D kernel, separable, clamped edges."""
    k = 2 * max(1, _spx(ctx, radius)) + 1
    kern = B.poisson_kernel(k, ctx.rng("poisson"))
    return B.run_premul(img, lambda d: B.sep_correlate(d, kern, kern, "clamp"))


# ---------------------------------------------------------------------------
# Shaped 2D kernels (aire ConvolveKernels via convolve2D, REFLECT_101)
# ---------------------------------------------------------------------------

def _shaped(img, ctx, size, make):
    k = _odd_px(ctx, B.nearest_odd(size))
    kern = make(k)
    if kern is None:
        return img
    return B.run_premul(img, lambda d: B.correlate2d(d, kern, "reflect_101"))


_SIZE_DESC = "Kernel size in pixels (rounded to the nearest odd value)"


@itb_op("CircleBlur", "Circle Blur", GROUP, [
    Int("size", 25, min=3, max=200, unit="px", title="Size", description=_SIZE_DESC),
], upstream="aire", alpha="full", tags=["blur", "circle", "disc", "kernel"])
def circle_blur(img, ctx, size):
    return _shaped(img, ctx, size, B.kernel_circle)


@itb_op("CrossBlur", "Cross Blur", GROUP, [
    Int("size", 25, min=3, max=200, unit="px", title="Size", description=_SIZE_DESC),
], upstream="aire", alpha="full", tags=["blur", "cross", "diagonal", "kernel"])
def cross_blur(img, ctx, size):
    return _shaped(img, ctx, size, B.kernel_cross)


@itb_op("RingBlur", "Ring Blur", GROUP, [
    Int("size", 25, min=3, max=200, unit="px", title="Size", description=_SIZE_DESC),
], upstream="aire", alpha="full", tags=["blur", "ring", "arc", "kernel"])
def ring_blur(img, ctx, size):
    return _shaped(img, ctx, size, B.kernel_ring)


@itb_op("StarBlur", "Star Blur", GROUP, [
    Int("size", 25, min=3, max=200, unit="px", title="Size", description=_SIZE_DESC),
], upstream="aire", alpha="full", tags=["blur", "star", "plus", "kernel"])
def star_blur(img, ctx, size):
    return _shaped(img, ctx, size, B.kernel_star)


@itb_op("Bokeh", "Bokeh", GROUP, [
    Int("size", 6, min=1, max=150, unit="px", title="Size"),
    Int("amount", 6, min=3, max=40, title="Amount", description="Number of polygon sides"),
], upstream="aire", alpha="full", tags=["blur", "bokeh", "dilate", "aperture"])
def bokeh(img, ctx, size, amount):
    """aire ``morphology(DILATE)`` with a filled-polygon bokeh kernel, REFLECT_101 edges."""
    kern = B.bokeh_kernel(_spx(ctx, size), amount)
    if kern is None:
        return img
    if img.shape[2] == 4:
        return B.run_premul(img, lambda d: B.dilate(d, kern))
    return B.dilate(np.ascontiguousarray(img, dtype=F32), kern)


# ---------------------------------------------------------------------------
# Zoom / tilt-shift
# ---------------------------------------------------------------------------

@itb_op("EnhancedZoomBlur", "Enhanced Zoom Blur", GROUP, [
    Int("radius", 25, min=1, max=100, title="Radius", description="Number of taps is 2*radius+1"),
    Float("sigma", 5, min=1, max=100, step=0.01, title="Sigma", description="Gaussian sigma over the taps"),
    Float("blur_center_x", 0.5, min=0, max=1, step=0.01, title="Blur center x"),
    Float("blur_center_y", 0.5, min=0, max=1, step=0.01, title="Blur center y"),
    Float("strength", 1, min=0, max=3, step=0.01, title="Strength"),
    Int("angle", 135, min=0, max=360, title="Angle", description="Degrees"),
], upstream="aire", alpha="full", tags=["blur", "zoom", "radial"])
def enhanced_zoom_blur(img, ctx, radius, sigma, blur_center_x, blur_center_y, strength, angle):
    """aire ``ZoomBlur``: gaussian-weighted nearest samples along (center - p) * strength / 100."""
    h, w = img.shape[:2]
    kern = B.gaussian_kernel_1d(2 * radius + 1, sigma)
    a = np.float32(angle) * np.float32(math.pi) / np.float32(180.0)
    cx = np.float32(math.floor(np.float32(w) * np.float32(blur_center_x)))
    cy = np.float32(math.floor(np.float32(h) * np.float32(blur_center_y)))
    return B.run_premul(img, lambda d: B.aire_zoom_blur(
        np.ascontiguousarray(d, dtype=F32), kern, cx, cy, np.float32(strength), np.cos(a), np.sin(a)))


@itb_op("ZoomBlur", "Zoom blur", GROUP, [
    Float("blur_center_x", 0.5, min=0, max=1, step=0.01, title="Blur center x"),
    Float("blur_center_y", 0.5, min=0, max=1, step=0.01, title="Blur center y"),
    Float("blur_size", 5, min=0, max=10, step=0.01, title="Blur size"),
], upstream="GPUImage", alpha="full", tags=["blur", "zoom", "radial"])
def zoom_blur(img, ctx, blur_center_x, blur_center_y, blur_size):
    """GPUImageZoomBlurFilter: 9 bilinear taps along (center - uv) * size / 100."""
    return B.run_premul(img, lambda d: B.gpu_zoom_blur(d, blur_center_x, blur_center_y, blur_size))


@itb_op("RadialTiltShift", "Tilt Shift", GROUP, [
    Int("blur_radius", 25, min=1, max=100, unit="px", title="Blur radius", description="Gaussian kernel size"),
    Float("sigma", 10, min=1, max=50, step=0.01, unit="px", title="Sigma"),
    Float("center_x", 0.5, min=0, max=1, step=0.01, title="Center X"),
    Float("center_y", 0.5, min=0, max=1, step=0.01, title="Center Y"),
    Float("radius", 0.2, min=0, max=1, step=0.01, title="Radius", description="Sharp area, fraction of the diagonal"),
], upstream="aire", alpha="full", tags=["blur", "tilt shift", "miniature", "radial"])
def radial_tilt_shift(img, ctx, blur_radius, sigma, center_x, center_y, radius):
    """aire ``tiltShift``: blend towards a gaussian blur by (distance / (diag * radius))^3."""
    k = max(1, _spx(ctx, blur_radius))

    def fn(d):
        blurred = B.aire_gauss_blur(d, k, ctx.px(sigma))
        frac = B.radial_fraction(d.shape[0], d.shape[1], center_x, center_y, radius)
        return B.tilt_blend(d, blurred, frac)

    return B.run_premul(img, fn)


@itb_op("LinearTiltShift", "Linear Tilt-Shift", GROUP, [
    Int("blur_radius", 25, min=1, max=100, unit="px", title="Blur radius", description="Gaussian kernel size"),
    Float("sigma", 10, min=1, max=50, step=0.01, unit="px", title="Sigma"),
    Float("center_x", 0.5, min=0, max=1, step=0.01, title="Center X"),
    Float("center_y", 0.5, min=0, max=1, step=0.01, title="Center Y"),
    Float("size", 0.2, min=0, max=1, step=0.01, title="Size", description="Sharp band width, fraction of the extent"),
    Int("angle", 45, min=0, max=360, title="Angle", description="Degrees"),
], upstream="aire", alpha="full", tags=["blur", "tilt shift", "miniature", "linear"])
def linear_tilt_shift(img, ctx, blur_radius, sigma, center_x, center_y, size, angle):
    """aire ``horizontalTiltShift``: band blend by (distance / (extent * size))^3."""
    k = max(1, _spx(ctx, blur_radius))
    a = float(np.float32(angle) * np.float32(math.pi) / np.float32(180.0))

    def fn(d):
        blurred = B.aire_gauss_blur(d, k, ctx.px(sigma))
        frac = B.linear_fraction(d.shape[0], d.shape[1], center_x, center_y, size, a)
        return B.tilt_blend(d, blurred, frac)

    return B.run_premul(img, fn)


# ---------------------------------------------------------------------------
# jhlabs (ImageToolboxLibs native port)
# ---------------------------------------------------------------------------

@itb_op("AdaptiveBlur", "Adaptive Blur", GROUP, [
    Int("radius", 15, min=1, max=100, unit="px", title="Radius"),
    Int("threshold", 35, min=0, max=255, title="Threshold",
        description="Neighbours differing by more than this (0..255) are ignored"),
], upstream="jhlabs", alpha="full", tags=["blur", "smart blur", "edge preserving", "adaptive"])
def adaptive_blur(img, ctx, radius, threshold):
    """jhlabs SmartBlurFilter: per-channel thresholded gaussian, horizontal then vertical."""
    r = max(1, _spx(ctx, radius))
    return B.run_premul(img, lambda d: B.smart_blur(d, r, threshold))


@itb_op("Diffuse", "Diffuse", GROUP, [
    Float("scale", 50, min=0, max=500, step=0.01, unit="px", title="Scale", description="Maximum displacement"),
], upstream="jhlabs", alpha="full", tags=["blur", "diffuse", "scatter", "noise"])
def diffuse(img, ctx, scale):
    """jhlabs DiffuseFilter (native port): hashed per-pixel displacement, bilinear, clamped edges."""
    return B.run_premul(img, lambda d: B.diffuse(d, ctx.px(scale)))


@itb_op("LensBlur", "Lens Blur", GROUP, [
    Int("radius", 10, min=0, max=50, unit="px", title="Radius"),
    Int("sides", 5, min=3, max=12, title="Sides", description="Unused by ImageToolbox's native implementation"),
    Float("bloom", 2, min=0, max=10, step=0.01, title="Bloom"),
    Int("threshold", 255, min=0, max=255, title="Threshold", description="Bloom brightness threshold (0..255)"),
], upstream="jhlabs", alpha="full", tags=["blur", "lens", "bloom", "gaussian"])
def lens_blur(img, ctx, radius, sides, bloom, threshold):
    """ImageToolboxLibs native LensBlurFilter: bloom bright pixels, then jhlabs gaussian (sigma = r/3)."""
    r = max(0, _spx(ctx, radius))

    def fn(d):
        if threshold < 255:  # brightness = (r + g + b) / 3 on 8-bit values can never exceed 255
            q = np.round(np.clip(d[..., :3], 0, 1) * 255.0)
            bright = (np.floor(q.sum(axis=2) / 3.0) > threshold)[..., None]
            d = d.copy()
            d[..., :3] = np.where(bright, np.minimum(np.floor(q * np.float32(bloom)), 255.0) / 255.0, d[..., :3])
        if r <= 0:
            return d
        kern = B.jh_gaussian_kernel(float(r))
        return B.sep_correlate(d, kern, kern, "clamp")

    return B.run_premul(img, fn)


# ---------------------------------------------------------------------------
# Trickle
# ---------------------------------------------------------------------------

@itb_op("FastBlur", "Fast blur", GROUP, [
    Float("scale", 0.5, min=0.1, max=1, step=0.01, title="Scale", description="Downscale factor before blurring"),
    Int("radius", 5, min=0, max=100, unit="px", title="Radius", description="Box radius in downscaled pixels"),
], upstream="Trickle", alpha="full", tags=["blur", "fast", "box", "downscale"])
def fast_blur(img, ctx, scale, radius):
    """Trickle ``fastBlurImpl``: bilinear downscale, (2r+1) box blur, upscale; output is opaque."""
    if radius < 1:
        return img
    r = max(1, _spx(ctx, radius))
    out = B.trickle_scaled(B.premul(img), scale, lambda d: B.centered_box(d, r))
    if img.shape[2] == 4:  # Trickle writes 0xFF alpha over the premultiplied colours
        out = np.dstack([out[..., :3], np.ones(img.shape[:2], F32)])
    return out


@itb_op("StackBlur", "Stack blur", GROUP, [
    Float("scale", 0.5, min=0.1, max=1, step=0.01, title="Scale", description="Downscale factor before blurring"),
    Int("radius", 10, min=0, max=100, unit="px", title="Radius", description="Stack radius in downscaled pixels"),
], upstream="Trickle", alpha="full", tags=["blur", "stack blur", "fast", "downscale"])
def stack_blur(img, ctx, scale, radius):
    """Trickle ``stackBlurImpl``: bilinear downscale, Klingemann stack blur on RGB, upscale."""
    if radius < 1:
        return img
    r = max(1, _spx(ctx, radius))

    def blur(d):
        out = d.copy()
        out[..., :3] = B.triangle_blur(np.ascontiguousarray(d[..., :3]), r)
        return out

    data = B.premul(img)
    out = B.trickle_scaled(data, scale, blur)
    return B.unpremul(out) if img.shape[2] == 4 else out


# ShuffleBlurFilter.radiusMapping (Kotlin Float arithmetic)
_SHUFFLE_RADIUS = [F32(0.0), F32(0.001)] + [F32(0.005) + F32(0.005) * F32(i) for i in range(200)]


@itb_op("ShuffleBlur", "Shuffle Blur", GROUP, [
    Int("radius", 35, min=0, max=70, title="Radius", description="Shuffle strength step"),
    Float("threshold", 1, min=-1, max=1, step=0.01, title="Threshold",
          description=">= 0: shuffle pixels with linear luma <= threshold; < 0: luma > |threshold|"),
], upstream="Trickle", alpha="full", tags=["blur", "shuffle", "scatter", "random"])
def shuffle_blur(img, ctx, radius, threshold):
    """Trickle ``shuffleBlurImpl``: copy a random pixel from a position-dependent window.

    Pixels that do not pass the luma threshold are left transparent, like upstream.
    """
    strength = float(_SHUFFLE_RADIUS[radius])
    seed = int(ctx.rng("shuffle").integers(0, 2 ** 63 - 1))
    out = B.unpremul(B.shuffle_blur(B.premul(img), threshold, strength, seed))
    if img.shape[2] == 3 and out[..., 3].min() >= 1.0:  # nothing became transparent: keep RGB
        return out[..., :3]
    return out


# ---------------------------------------------------------------------------
# G'MIC
# ---------------------------------------------------------------------------

@itb_op("FrostedGlass", "Frosted Glass", GROUP, [
    Float("x_amplitude", 40, min=0, max=400, step=0.01, unit="px", title="X amplitude"),
    Float("y_amplitude", 40, min=0, max=400, step=0.01, unit="px", title="Y amplitude"),
    Float("x_smoothness", 1, min=0, max=5, step=0.01, unit="px", title="X Smoothness"),
    Float("y_smoothness", 1, min=0, max=5, step=0.01, unit="px", title="Y Smoothness"),
    Float("edge_attenuation", 0, min=0, max=1, step=0.01, title="Edge Attenuation"),
    Float("edge_influence", 2, min=0, max=10, step=0.01, unit="px", title="Edge Influence"),
    Int("scale", 0, min=0, max=16, title="Scale", description="Noise scale (0 = per-pixel random noise)"),
], upstream="G'MIC", requires=["gmic"], alpha="full", tags=["blur", "glass", "frosted", "distort"])
def frosted_glass(img, ctx, x_amplitude, y_amplitude, x_smoothness, y_smoothness, edge_attenuation,
                  edge_influence, scale):
    """G'MIC ``fx_textured_glass`` (alpha processed as a 4th channel, like GmicAlphaMode.Process).

    G'MIC's random generator is seeded from the render context (``srand``) for determinism.
    """
    seed = int(ctx.rng("frosted_glass").integers(0, 2 ** 31 - 1))
    cmd = gmic_pipeline(
        gmic_cmd("srand", seed),
        gmic_cmd("fx_textured_glass", float(ctx.px(x_amplitude)), float(ctx.px(y_amplitude)),
                 float(ctx.px(x_smoothness)), float(ctx.px(y_smoothness)), float(edge_attenuation),
                 float(ctx.px(edge_influence)), int(scale)),
    )
    return run_gmic(img, cmd, ctx=ctx, preserve_alpha=False)
