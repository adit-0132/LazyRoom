"""Photographic effects (category ``effect``)."""
from __future__ import annotations

import cv2
import numpy as np

from ..core.color import linear_to_srgb, srgb_to_linear, rgb_to_oklab, oklab_to_rgb
from ..core.params import Bool, Choice, Color, Float, Int, Point
from ..core.registry import op
from ._util import F32, gaussian, luma, norm_grid


@op("effect.grain", "Film grain", params=[
    Float("amount", 25, min=0, max=100, title="Amount"),
    Float("size", 1.5, min=0.5, max=12, step=0.1, unit="px", title="Grain size"),
    Float("roughness", 50, min=0, max=100, title="Roughness", description="Irregularity of grain clumps"),
    Float("color", 0, min=0, max=100, title="Colour noise", description="0 = monochrome grain"),
    Float("highlight_response", 30, min=0, max=100, advanced=True,
          description="How much grain shows in highlights (film has less grain in highlights)"),
    Int("seed", 0, min=0, max=1_000_000, advanced=True),
], tags=["grain", "film", "noise", "analog", "texture", "vintage"], deterministic=True,
    examples=[{"amount": 20, "size": 1.8}])
def grain(img, ctx, amount, size, roughness, color, highlight_response, seed):
    """Luminance-dependent, resolution-independent film grain (deterministic per seed)."""
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    rng = ctx.rng(f"grain{seed}")
    s = max(0.35, ctx.px(size))
    def layer():
        n = rng.standard_normal((h, w), dtype=np.float32)
        n = gaussian(n, s * 0.6)
        if roughness > 0:
            coarse = gaussian(rng.standard_normal((h, w), dtype=np.float32), s * 2.0)
            n = n * (1 - roughness / 200) + coarse * (roughness / 100)
        sd = float(n.std()) or 1.0
        return n / sd
    n = layer()
    y = np.clip(luma(img), 0, 1)
    resp = 0.35 + 0.65 * (4 * y * (1 - y)) ** 0.7  # strongest in midtones
    hl = np.clip((y - 0.5) * 2, 0, 1)
    resp = resp * (1 - hl * (1 - highlight_response / 100) * 0.6)
    strength = amount / 100.0 * 0.12
    out = img + (n * strength * resp)[..., None]
    if color > 0:
        cn = np.stack([layer(), layer(), layer()], -1)
        cn -= cn.mean(-1, keepdims=True)
        out = out + cn * (strength * color / 100.0 * 0.6) * resp[..., None]
    return out.astype(F32)


@op("effect.vignette", "Vignette", params=[
    Float("amount", -30, min=-100, max=100, title="Amount", description="Negative darkens edges, positive lightens"),
    Float("midpoint", 50, min=0, max=100, title="Midpoint", description="Size of the unaffected centre"),
    Float("roundness", 0, min=-100, max=100, title="Roundness", description="-100 = follows frame, 100 = circle"),
    Float("feather", 50, min=0, max=100, title="Feather"),
    Point("center", (0.5, 0.5), min=0, max=1, title="Centre"),
    Color("color", "#000000", advanced=True, description="Vignette colour (for amount < 0)"),
], tags=["vignette", "edges", "darken corners", "frame", "lens"])
def vignette(img, ctx, amount, midpoint, roundness, feather, center, color):
    """Lightroom-style post-crop vignette, applied in linear light."""
    if amount == 0:
        return img
    h, w = img.shape[:2]
    X, Y = np.meshgrid((np.arange(w, dtype=F32) + 0.5) / w, (np.arange(h, dtype=F32) + 0.5) / h)
    dx = (X - center[0]) * 2
    dy = (Y - center[1]) * 2
    r = roundness / 100.0
    aspect = w / h
    if r >= 0:  # blend frame-shaped ellipse (r=0) -> true circle (r=1)
        ell = np.sqrt(dx ** 2 + dy ** 2)
        circ = np.sqrt((dx * aspect) ** 2 + dy ** 2) * np.sqrt(2) / np.sqrt(aspect ** 2 + 1)
        d = ell * (1 - r) + circ * r
    else:  # towards a rounded rectangle (superellipse)
        p = 2.0 - r * 6.0
        d = (np.abs(dx) ** p + np.abs(dy) ** p) ** (1 / p)
    inner = midpoint / 100.0
    soft = 0.05 + feather / 100.0 * 1.2
    t = np.clip((d - inner) / soft, 0, 1)
    t = (t * t * (3 - 2 * t))[..., None]
    k = amount / 100.0
    lin = srgb_to_linear(img)
    if k < 0:
        col = srgb_to_linear(np.array(color[:3], F32))
        out = lin + (col - lin) * (-k * t)
    else:
        out = lin + (1 - lin) * (k * t) * 0.8
    return linear_to_srgb(out)


@op("effect.bloom", "Bloom / glow", params=[
    Float("threshold", 0.7, min=0, max=1, step=0.01, description="Brightness above which light blooms"),
    Float("intensity", 40, min=0, max=200, title="Intensity"),
    Float("radius", 40, min=1, max=500, unit="px"),
    Float("knee", 0.2, min=0, max=1, step=0.01, title="Soft knee", advanced=True),
    Color("tint", "#FFFFFF", description="Colour of the glow"),
], tags=["bloom", "glow", "dreamy", "highlights", "soft light", "diffusion"])
def bloom(img, ctx, threshold, intensity, radius, knee, tint):
    """Physically motivated bloom: bright-pass in linear light, multi-scale blur, additive."""
    lin = srgb_to_linear(np.clip(img, 0, None))
    y = lin @ np.array([0.2126, 0.7152, 0.0722], F32)
    t_lin = float(srgb_to_linear(np.array([threshold], F32))[0])
    kn = max(knee * t_lin, 1e-4)
    soft = np.clip(y - t_lin + kn, 0, 2 * kn)
    soft = soft * soft / (4 * kn)
    contrib = np.maximum(soft, y - t_lin) / np.maximum(y, 1e-4)
    bright = lin * contrib[..., None]
    r = ctx.px(radius)
    acc = np.zeros_like(bright)
    for f, wgt in ((0.15, 0.25), (0.4, 0.3), (1.0, 0.45)):
        acc += gaussian(bright, max(0.5, r * f)) * wgt
    acc *= np.array(srgb_to_linear(np.array(tint[:3], F32)), F32)
    return linear_to_srgb(lin + acc * intensity / 100.0 * 2)


@op("effect.diffusion", "Diffusion (soft / dreamy / Orton)", params=[
    Float("amount", 30, min=0, max=100),
    Float("radius", 25, min=1, max=400, unit="px"),
    Choice("mode", "screen", choices=["screen", "lighten", "soft_light", "normal"],
           description="screen = Pro-Mist highlight glow; soft_light = Orton effect"),
    Float("highlights_only", 50, min=0, max=100, description="Restrict the effect to brighter areas"),
], tags=["diffusion", "dreamy", "soft focus", "orton", "pro mist", "haze", "ethereal"])
def diffusion(img, ctx, amount, radius, mode, highlights_only):
    """Soft-focus diffusion filter: blend a blurred copy over the image."""
    from ..core.blend import blend_rgb

    blurred = gaussian(img, ctx.px(radius))
    top = blend_rgb(np.clip(img, 0, 1), np.clip(blurred, 0, 1), mode)
    y = np.clip(luma(blurred), 0, 1)
    m = 1 - highlights_only / 100.0 + highlights_only / 100.0 * y ** 1.5
    w = (amount / 100.0) * m
    return (img + (top - img) * w[..., None]).astype(F32)


@op("effect.halation", "Halation", params=[
    Float("amount", 30, min=0, max=100),
    Float("threshold", 0.75, min=0, max=1, step=0.01),
    Float("radius", 20, min=1, max=300, unit="px"),
    Color("color", "#FF3A10", description="Halation colour (film: red-orange)"),
], tags=["halation", "film", "cinestill", "glow", "red glow", "analog"])
def halation(img, ctx, amount, threshold, radius, color):
    """Film halation: red-orange glow bleeding around bright highlights."""
    y = luma(np.clip(img, 0, 1))
    mask = np.clip((y - threshold) / max(1 - threshold, 1e-3), 0, 1)
    glow = gaussian(mask, ctx.px(radius))
    glow = np.clip(glow - mask * 0.5, 0, 1)
    col = np.array(color[:3], F32)
    lin = srgb_to_linear(img)
    out = lin + glow[..., None] * srgb_to_linear(col) * amount / 100.0 * 1.5
    return linear_to_srgb(out)


@op("effect.sharpen", "Sharpen (unsharp mask)", params=[
    Float("amount", 50, min=0, max=500, unit="%"),
    Float("radius", 1.0, min=0.1, max=50, step=0.1, unit="px"),
    Float("threshold", 0, min=0, max=0.3, step=0.005, description="Ignore differences below this"),
    Bool("luminance_only", True, description="Sharpen luminance only (avoids colour fringes)"),
], tags=["sharpen", "unsharp mask", "crisp", "detail"])
def sharpen(img, ctx, amount, radius, threshold, luminance_only):
    """Unsharp-mask sharpening (on luminance by default)."""
    r = max(0.3, ctx.px(radius))
    k = amount / 100.0
    if luminance_only:
        y = luma(img)
        d = y - gaussian(y, r)
        if threshold > 0:
            d = np.where(np.abs(d) < threshold, 0, d)
        return (img + (d * k)[..., None]).astype(F32)
    d = img - gaussian(img, r)
    if threshold > 0:
        d = np.where(np.abs(d) < threshold, 0, d)
    return (img + d * k).astype(F32)


@op("effect.denoise", "Noise reduction", params=[
    Float("luminance", 30, min=0, max=100, title="Luminance noise"),
    Float("color", 30, min=0, max=100, title="Colour noise"),
    Float("detail", 50, min=0, max=100, description="Preserve fine detail"),
    Choice("method", "nlmeans", choices=["nlmeans", "bilateral", "guided"]),
], tags=["denoise", "noise reduction", "smooth", "clean"])
def denoise(img, ctx, luminance, color, detail, method):
    """Luminance/chroma noise reduction (non-local means by default) in OKLab."""
    lab = rgb_to_oklab(np.clip(img, 0, 1))
    L, ab = lab[..., 0], lab[..., 1:]
    if luminance > 0:
        if method == "nlmeans":
            L8 = np.clip(L * 255, 0, 255).astype(np.uint8)
            hval = 1 + luminance / 100 * 14 * (1 - detail / 200)
            den = cv2.fastNlMeansDenoising(L8, None, h=hval, templateWindowSize=7, searchWindowSize=21)
            Ld = L - (L8.astype(F32) - den.astype(F32)) / 255.0
        elif method == "bilateral":
            Ld = cv2.bilateralFilter(L, 0, luminance / 100 * 0.08, ctx.px(2 + luminance / 20))
        else:
            from ._util import guided_filter

            Ld = guided_filter(L, L, ctx.ipx(2 + luminance / 25), eps=(luminance / 100) ** 2 * 0.004)
        keep = detail / 100 * 0.5
        L = Ld + (L - Ld) * keep
    if color > 0:
        s = ctx.px(1 + color / 100 * 6)
        ab = np.dstack([gaussian(ab[..., 0], s), gaussian(ab[..., 1], s)])
    lab = np.dstack([L, ab]).astype(F32)
    return oklab_to_rgb(lab)


@op("effect.blur", "Gaussian blur", params=[Float("radius", 5, min=0, max=500, unit="px")],
    tags=["blur", "soften", "gaussian"])
def blur(img, ctx, radius):
    """Gaussian blur (radius ≈ 2σ)."""
    return gaussian(img, ctx.px(radius) / 2)


@op("effect.chromatic_aberration", "Chromatic aberration", params=[
    Float("amount", 2.0, min=-20, max=20, step=0.1, unit="px", description="Red/blue radial misalignment at the corners"),
], tags=["chromatic aberration", "lens", "fringe", "retro", "glitch"])
def chromatic_aberration(img, ctx, amount):
    """Radial lateral chromatic aberration (red/blue channels scaled in opposite directions)."""
    h, w = img.shape[:2]
    a = ctx.px(amount) / (0.5 * np.hypot(w, h))
    out = img.copy()
    for c, s in ((0, 1 + a), (2, 1 - a)):
        M = np.float32([[s, 0, (1 - s) * w / 2], [0, s, (1 - s) * h / 2]])
        out[..., c] = cv2.warpAffine(img[..., c], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return out


@op("effect.light_leak", "Light leak", params=[
    Float("amount", 40, min=0, max=100),
    Color("color", "#FF7A30"),
    Float("angle", 30, min=0, max=360, unit="deg", description="Direction the leak comes from"),
    Float("size", 50, min=5, max=100, description="How far it reaches into the frame"),
    Int("seed", 0, min=0, max=1_000_000, advanced=True),
], tags=["light leak", "polaroid", "film", "vintage", "lomo", "analog"])
def light_leak(img, ctx, amount, color, angle, size, seed):
    """Warm light leak entering from one edge (screen blended), with organic variation."""
    h, w = img.shape[:2]
    nx, ny = norm_grid(h, w, aspect=False)
    th = np.radians(angle)
    d = nx * np.cos(th) + ny * np.sin(th)  # -1..1 across the frame
    reach = size / 100.0 * 2
    t = np.clip((d - (1 - reach)) / max(reach, 1e-3), 0, 1)
    from ._util import value_noise

    n = value_noise(h, w, max(h, w) / 4, ctx.rng(f"leak{seed}"), octaves=2)
    t = np.clip(t * (0.6 + 0.8 * n), 0, 1) ** 1.5
    col = np.array(color[:3], F32)
    top = 1 - (1 - np.clip(img, 0, 1)) * (1 - col)
    return (img + (top - img) * (t * amount / 100.0)[..., None]).astype(F32)
