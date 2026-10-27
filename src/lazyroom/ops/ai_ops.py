"""AI operations (category ``ai``) — ImageToolbox "AI tools" parity on ONNX Runtime.

Every operation has a ``model`` choice listing the compatible catalogue entries
(src/lazyroom/ai/catalog.json).  Nothing is bundled: an operation reports
itself unavailable until at least one compatible model is downloaded
(``requires`` contains ``model:@<task>``), and running it with a model that is
not downloaded raises BACKEND_UNAVAILABLE whose hint is the exact
``lazyroom models download NAME --yes`` command.
"""
from __future__ import annotations

import json

import numpy as np

from ..ai import models
from ..core.errors import invalid
from ..core.params import Bool, Choice, File, Float, Int, Text
from ..core.registry import op
from ._util import F32

SOURCE = {"project": "ImageToolbox", "feature": "AI tools", "license": "Apache-2.0",
          "files": "feature/ai-tools/**, lib/neural-tools/**"}


def _model_param(task: str, description: str = "") -> Choice:
    names = models.models_for_task(task)
    labels = {}
    for n in names:
        m = models.MODELS[n]
        lic = m.get("license") or "unknown"
        flag = ", restricted licence" if m.get("restricted") else ""
        labels[n] = f"{m.get('title')} — {int(m.get('size') or 0) / 1e6:.0f} MB, {lic}{flag}"
    return Choice("model", models.default_model(task), choices=names, labels=labels, title="Model",
                  description=(description or "Catalogue model to use") +
                  ". Must be downloaded first: lazyroom models download NAME --yes")


def _req(task: str) -> list[str]:
    return ["onnxruntime", f"model:@{task}"]


def _tile_params() -> list:
    return [
        Int("chunk_size", 512, min=128, max=2048, title="Chunk size", unit="px", advanced=True,
            description="Tile size for chunked inference (ImageToolbox offers 128..2048, powers of two)"),
        Int("overlap", 16, min=16, max=128, title="Overlap", unit="px", advanced=True,
            description="Tile overlap blended with a smooth-step ramp"),
        Bool("enable_chunking", True, title="Enable chunking", advanced=True,
             description="Process large images in tiles (models that cannot be tiled ignore this)"),
    ]


def _strength(desc: str = "Model strength (only for models with a strength input, e.g. FBCNN)") -> Float:
    return Float("strength", 65.0, min=0, max=100, step=1, title="Strength", description=desc)


def _non_chunkable(name: str, task: str) -> bool:
    """NeuralModel.isNonChunkable."""
    return ("ddcolor" in name or task in ("remove_background", "style_transfer", "depth", "detect_ai_generated")
            or name in ("watermark_mit_b5_sigmoid.onnx", "uvdoc_grid.onnx"))


def _generic(task: str, img, ctx, model, strength=65.0, chunk_size=512, overlap=16, enable_chunking=True):
    from ..ai import runtime

    name = models.resolve_name(model)
    path = models.require_model(name)
    if overlap >= chunk_size:
        raise invalid("overlap must be smaller than chunk_size")
    est = runtime.name_scale(name)
    runtime.check_output_size(img.shape[1] * est, img.shape[0] * est)
    return runtime.process_image(name, path, img, ctx, strength=strength, chunk_size=chunk_size, overlap=overlap,
                                 enable_chunking=enable_chunking, non_chunkable=_non_chunkable(name, task),
                                 opt_level=runtime.opt_level_for(name))


# ---------------------------------------------------------------------------
@op("ai.upscale", "AI upscale", params=[_model_param("upscale", "Upscaling model (x2/x3/x4)"), *_tile_params()],
    requires=_req("upscale"), geometry=True, alpha="full", source=SOURCE,
    tags=["ai", "upscale", "super resolution", "enlarge", "esrgan", "swinir", "dat"],
    examples=[{"model": "RealESRGAN-x4v3.ort"}])
def upscale(img, ctx, model, chunk_size, overlap, enable_chunking):
    """Neural super-resolution (Real-ESRGAN, SwinIR, DAT, UltraSharp, ...); output is 2-4x larger.
    Tiled with overlap blending exactly like ImageToolbox; alpha is upscaled nearest-neighbour."""
    return _generic("upscale", img, ctx, model, chunk_size=chunk_size, overlap=overlap,
                    enable_chunking=enable_chunking)


def _simple_op(op_id: str, title: str, task: str, doc: str, tags: list[str], strength: bool = False) -> None:
    params = [_model_param(task)] + ([_strength()] if strength else []) + _tile_params()

    def fn(img, ctx, model, chunk_size, overlap, enable_chunking, strength=65.0):
        return _generic(task, img, ctx, model, strength=strength, chunk_size=chunk_size, overlap=overlap,
                        enable_chunking=enable_chunking)

    fn.__doc__ = doc
    fn.__name__ = op_id.split(".")[1]
    op(op_id, title, params=params, requires=_req(task), source=SOURCE, tags=["ai", *tags])(fn)


_simple_op("ai.colorize", "AI colorize", "colorize",
           "Colourise grayscale / black-and-white photos (DDColor, ColorizerV2, SpongeColor).",
           ["colorize", "colourise", "black and white", "restore"])
_simple_op("ai.dejpeg", "AI remove JPEG artifacts", "dejpeg",
           "Remove JPEG compression artifacts (FBCNN with a strength control, 1x models per quality range).",
           ["jpeg", "compression", "artifacts", "deblock"], strength=True)
_simple_op("ai.denoise", "AI denoise / deblur", "denoise",
           "Remove noise or blur (SCUNet, NAFNet, ISO denoise, noise toners...).",
           ["denoise", "noise", "deblur", "grain"])
_simple_op("ai.remove_artifacts", "AI remove artifacts", "remove_artifacts",
           "Remove aliasing, halftone, dithering, codec and broadcast artifacts.",
           ["artifacts", "halftone", "aliasing", "dither", "codec"])
_simple_op("ai.enhance", "AI enhance", "enhance",
           "General enhancement models (low-light Zero-DCE++, exposure correction, debanding, sharpening, deraining).",
           ["enhance", "low light", "exposure", "deband", "sharpen", "derain"])
_simple_op("ai.restore_anime", "AI restore anime", "restore_anime",
           "Restore anime / cartoon frames (compression, jagged lines).", ["anime", "cartoon", "restore"])


@op("ai.restore_scans", "AI restore scans", params=[_model_param("restore_scans"), *_tile_params()],
    requires=_req("restore_scans"), alpha="full", source=SOURCE,
    tags=["ai", "scan", "document", "book", "unwarp", "dewarp"])
def restore_scans(img, ctx, model, chunk_size, overlap, enable_chunking):
    """Clean up book / document scans; 'uvdoc_grid.onnx' flattens curved pages (UVDoc unwarping)."""
    name = models.resolve_name(model)
    if name == "uvdoc_grid.onnx":
        from ..ai import processors

        return processors.uvdoc_unwarp(models.require_model(name), img, ctx)
    return _generic("restore_scans", img, ctx, name, chunk_size=chunk_size, overlap=overlap,
                    enable_chunking=enable_chunking)


@op("ai.remove_background", "AI remove background", params=[_model_param("remove_background")],
    requires=_req("remove_background"), alpha="full", source=SOURCE,
    tags=["ai", "background", "remove background", "cutout", "matting", "segmentation", "subject"])
def remove_background(img, ctx, model):
    """Cut out the subject: the model's mask becomes the alpha channel (U2Net, ISNet, BiRefNet, MODNet,
    RMBG, InSPyReNet, YOLO11-seg).  Pixels that were already transparent keep their alpha."""
    from ..ai import processors

    name = models.resolve_name(model)
    path = models.require_model(name)
    alpha = img[..., 3] if img.shape[2] == 4 else None
    mask = processors.background_mask(name, path, np.ascontiguousarray(img[..., :3]), ctx)
    return np.dstack([img[..., :3], processors.heal_alpha(mask, alpha)]).astype(F32)


@op("ai.style_transfer", "AI style transfer", params=[
    _model_param("style_transfer"),
    File("style", "", file_kind="image", required=True, title="Style image",
         description="Reference image whose colours / textures are transferred"),
    Float("strength", 65.0, min=0, max=100, step=1, title="Strength", description="0 = unchanged, 100 = full style"),
], requires=_req("style_transfer"), source=SOURCE, tags=["ai", "style transfer", "artistic", "painting"])
def style_transfer(img, ctx, model, style, strength):
    """Neural style transfer from a reference image (Magenta arbitrary stylization, MicroAST, AesFA,
    VGG19 Gram-matrix optimisation)."""
    from ..ai import processors
    from ..core.io import load_image

    name = models.resolve_name(model)
    path = models.require_model(name)
    sty, _ = load_image(ctx.resolve_file(style, "style image"))
    return processors.style_transfer(name, path, img[..., :3], np.ascontiguousarray(sty[..., :3]), strength, ctx)


@op("ai.depth_map", "AI depth effects", params=[
    _model_param("depth", "Depth Anything V2 / 3 model"),
    Choice("effect", "map", choices=["map", "lens_blur", "background_blur", "portrait", "color_grade", "depth_edges",
                                     "normal_map", "stereo", "depth_mask", "cutout", "focus_color", "fog", "relight"],
           title="Effect", description="map = colour-coded depth (near = red, far = violet)"),
    Float("strength", 65.0, min=0, max=100, step=1, title="Strength"),
    Float("focus", 65.0, min=0, max=100, step=1, title="Focus depth",
          description="Depth in focus (0 = far, 100 = near); zero plane for 'stereo'"),
    Float("focus_range", 18.0, min=1, max=50, step=1, title="Focus range"),
    Float("light_angle", 315.0, min=0, max=360, step=1, title="Light angle", unit="deg"),
    File("depth_image", "", file_kind="image", title="Custom depth map",
         description="Optional depth image (grayscale near = white, or this op's spectral map) used instead of a model"),
], requires=_req("depth"), alpha="full", source=SOURCE,
    tags=["ai", "depth", "depth map", "bokeh", "lens blur", "portrait", "normal map", "3d", "anaglyph", "fog"])
def depth_map(img, ctx, model, effect, strength, focus, focus_range, light_angle, depth_image):
    """Estimate depth (Depth Anything) and render one of ImageToolbox's 13 depth effects."""
    from ..ai import processors
    from ..core.io import load_image

    rgb = np.ascontiguousarray(img[..., :3])
    alpha = img[..., 3] if img.shape[2] == 4 else None
    if depth_image:
        di, _ = load_image(ctx.resolve_file(depth_image, "depth map"))
        depth = processors.depth_from_image(di)
    else:
        name = models.resolve_name(model)
        depth = processors.infer_depth(name, models.require_model(name), rgb, ctx)
    out, a = processors.render_depth_effect(rgb, alpha, depth, effect, strength=strength, focus=focus,
                                            focus_range=focus_range, light_angle=light_angle)
    return np.dstack([out, a]).astype(F32) if a is not None else out


def _load_mask(img, ctx, mask_image: str, mask: str) -> np.ndarray:
    from ..core.masks import render_mask, validate_mask

    h, w = img.shape[:2]
    if mask_image:
        from ..core.image import gray, resize
        from ..core.io import load_image

        mi, _ = load_image(ctx.resolve_file(mask_image, "mask image"))
        m = mi[..., 3] if mi.shape[2] == 4 and float(mi[..., 3].min()) < 1.0 else gray(mi[..., :3])
        return np.clip(resize(m[..., None].astype(F32), w, h)[..., 0], 0, 1)
    if mask:
        try:
            spec = json.loads(mask)
        except json.JSONDecodeError as exc:
            raise invalid(f"mask must be a JSON mask spec: {exc}") from None
        return np.clip(render_mask(validate_mask(spec), img, ctx), 0, 1)
    return np.zeros((h, w), F32)  # no mask: SpotHealFilter returns the input unchanged


def _composite(img, filled, m, keep_unmasked):
    if not keep_unmasked:
        return filled
    a = m[..., None]
    return (img[..., :3] * (1 - a) + filled * a).astype(F32)


@op("ai.inpaint", "AI inpaint (LaMa)", params=[
    _model_param("inpaint", "LaMa inpainting model"),
    File("mask_image", "", file_kind="image", title="Mask image",
         description="White (or opaque) = region to fill; resized to the image"),
    Text("mask", "", max_length=100_000, title="Mask spec",
         description="Alternative to mask_image: JSON mask spec (same format as step masks)"),
    Bool("keep_unmasked", True, title="Keep unmasked pixels",
         description="Only replace the masked region (ImageToolbox returns the whole 512 px LaMa result upscaled)"),
], requires=_req("inpaint"), source=SOURCE, tags=["ai", "inpaint", "erase", "remove object", "heal", "lama"])
def inpaint(img, ctx, model, mask_image, mask, keep_unmasked):
    """Fill a masked region with LaMa (the model runs at 512x512 on the whole image, like ImageToolbox's
    spot-heal filter).  Without a mask the image is returned unchanged."""
    from ..ai import processors

    name = models.resolve_name(model)
    path = models.require_model(name)
    m = _load_mask(img, ctx, mask_image, mask)
    if float(m.max()) <= 0:
        return img[..., :3].copy()
    filled = processors.lama_inpaint(name, path, img[..., :3], m, ctx)
    return _composite(img, filled, m, keep_unmasked)


@op("ai.remove_watermark", "AI remove watermark", params=[
    _model_param("remove_watermark", "Watermark detector (uses LaMa_512.onnx for filling)"),
    Bool("keep_unmasked", True, title="Keep unmasked pixels",
         description="Only replace detected watermark pixels (ImageToolbox returns the whole LaMa result)"),
], requires=_req("remove_watermark"), source=SOURCE, tags=["ai", "watermark", "remove watermark", "logo", "lama"])
def remove_watermark(img, ctx, model, keep_unmasked):
    """Detect common watermarks (SegFormer MiT-B5 mask, threshold 0.18) and inpaint them with LaMa."""
    from ..ai import processors

    name = models.resolve_name(model)
    path = models.require_model(name)
    m = processors.watermark_mask(path, img[..., :3], ctx)
    if float(m.max()) <= 0:
        return img[..., :3].copy()
    lama = models.MODELS[name].get("requires_models", ["LaMa_512.onnx"])[0]
    filled = processors.lama_inpaint(lama, models.require_model(lama), img[..., :3], m, ctx)
    return _composite(img, filled, m, keep_unmasked)
