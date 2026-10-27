"""ImageToolbox analysis "labs" as tools: visual compare renders and the animated before/after GIF
(feature/compare), image scopes (lib/curves + libs/histogram), compression lab (feature/compression-lab)
and multi-frame fusion (feature/multi-frame-fusion).

The scopes, compression lab and fusion tools live in private modules (``_scopes``, ``_compression_lab``,
``_fusion``) and are registered by importing them here.  See docs/dev/port_reports/color_tools.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import invalid
from ..core.image import fit_within
from ..core.io import atomic_write
from ..core.params import Bool, Choice, Color, File, Float, Int, Text
from . import _compare_render as cr
from .registry import out_path, read_image, save, tool

F32 = np.float32
ITB = "ImageToolbox"
COMPARE_MODES = ["slide", "side_by_side", "toggle", "transparency", "pixel_by_pixel"]


def _load_compare(api, path: str) -> np.ndarray:
    """CompareComponent.getBitmapByUri: decoded with the longest side limited to 4096 px."""
    img, _ = read_image(api, path)
    return fit_within(img, 4096)[0]


def _rgb_on(img: np.ndarray, bg) -> np.ndarray:
    if img.shape[2] == 4:
        a = img[..., 3:4]
        return img[..., :3] * a + np.array(bg[:3], F32) * (1 - a)
    return img[..., :3]


def _resize_exact(img: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(np.ascontiguousarray(img, dtype=F32), (w, h), interpolation=interp)


@tool("compare_visual", "Visual compare (slide / side by side / toggle / transparency / pixel by pixel)",
      category="analysis", params=[
          File("before", "", file_kind="image", required=True, description="First image ('before')"),
          File("after", "", file_kind="image", required=True, description="Second image ('after')"),
          Choice("mode", "slide", choices=COMPARE_MODES,
                 description="ImageToolbox CompareType: slide (before|after split), side_by_side, toggle (tap: "
                             "2-frame GIF), transparency (after over before at 'progress' opacity), pixel_by_pixel"),
          Float("progress", 50.0, min=0, max=100, step=0.5,
                description="Slide position / transparency in percent (ImageToolbox default 50)"),
          Float("threshold", 4.0, min=0, max=100, step=0.5, description="Pixel-by-pixel tolerance (default 4)"),
          Choice("comparison_type", "ae", choices=["ssim", "ae", "mae", "ncc", "psnr", "rmse"],
                 description="Pixel-by-pixel metric (ImageToolbox default AE)"),
          Color("highlight_color", "#FF0000", description="Pixel-by-pixel highlight colour"),
          Choice("orientation", "horizontal", choices=["horizontal", "vertical"],
                 description="side_by_side: Row (landscape) or Column (portrait) layout"),
          Bool("labels", False, description="Draw the file names on the images (CompareLabel)"),
          Int("toggle_ms", 1000, min=100, max=10000, description="toggle: time each image is shown"),
          Color("background", "#00000000", description="Letterbox colour where an image does not fill its area"),
          Text("output", "", description="Output image (toggle needs .gif)"), Bool("overwrite", False),
      ], source={"project": ITB, "feature": "Compare (CompareType, ImageDiffTool)"})
def compare_visual(api, before, after, mode, progress, threshold, comparison_type, highlight_color, orientation,
                   labels, toggle_ms, background, output, overwrite):
    """Render one of ImageToolbox's compare views to an image file.

    slide = the app's export (getOverlappedImage: 'after' canvas, the left ``progress`` % replaced by the
    'before' image scaled to the same size); pixel_by_pixel = ImageDiffTool highlight of 'before' with the
    score; side_by_side / transparency / toggle reproduce the on-screen layouts (not exportable in the app).
    """
    if not output:
        raise invalid("output is required", param="output")
    b0 = _load_compare(api, before)
    a0 = _load_compare(api, after)
    res: dict[str, Any] = {"mode": mode}
    names = (Path(before).name, Path(after).name)
    if mode == "pixel_by_pixel":
        from ._image_diff import pixel_by_pixel

        pb, hl = pixel_by_pixel(b0, a0, [comparison_type], threshold=threshold, highlight_color=highlight_color,
                                highlight_method=comparison_type)
        res["pixel_by_pixel"] = pb
        res["score"] = pb["methods"][comparison_type]["score"]
        res["output"] = save(api, hl, output, overwrite=overwrite)
        return res
    if mode == "slide":
        h, w = a0.shape[:2]
        canvas = a0.copy()
        bs = _resize_exact(b0, w, h)
        if bs.shape[2] != canvas.shape[2]:
            canvas = np.dstack([canvas, np.ones(canvas.shape[:2], F32)]) if canvas.shape[2] == 3 else canvas
            bs = np.dstack([bs, np.ones(bs.shape[:2], F32)]) if bs.shape[2] == 3 else bs
        cut = int(round(w * progress / 100))
        if cut > 0:  # drawBitmap(createBitmap(scaled, 0, 0, cut, h)) at (0, 0): source-over
            src = bs[:, :cut]
            if src.shape[2] == 4:
                sa = src[..., 3:4]
                dst = canvas[:, :cut]
                out_a = sa + dst[..., 3:4] * (1 - sa)
                rgb = (src[..., :3] * sa + dst[..., :3] * dst[..., 3:4] * (1 - sa)) / np.maximum(out_a, 1e-6)
                canvas[:, :cut] = np.dstack([rgb, out_a])
            else:
                canvas[:, :cut] = src
        if labels:
            rgb = _rgb_on(canvas, background)
            cr.draw_label(rgb, names[0], at_start=True, text_size=min(w, h) * 0.035)
            cr.draw_label(rgb, names[1], at_start=False, text_size=min(w, h) * 0.035)
            canvas = rgb
        res.update(split_x=cut)
        res["output"] = save(api, canvas, output, overwrite=overwrite)
        return res
    bg = np.array(background, F32)
    if mode == "side_by_side":
        horiz = orientation == "horizontal"
        cw, ch = max(b0.shape[1], a0.shape[1]), max(b0.shape[0], a0.shape[0])
        div = max(1, round(min(cw, ch) * 0.002))
        W, H = (2 * cw + div, ch) if horiz else (cw, 2 * ch + div)
        canvas = np.empty((H, W, 4), F32)
        canvas[:] = bg
        for i, im in enumerate((b0, a0)):
            cell = np.empty((ch, cw, 4), F32)
            cell[:] = bg
            _fit_rgba(cell, im)
            x, y = ((cw + div) * i, 0) if horiz else (0, (ch + div) * i)
            canvas[y:y + ch, x:x + cw] = cell
        line = np.array([0x79 / 255, 0x74 / 255, 0x7E / 255, 1], F32)  # M3 outline divider
        if horiz:
            canvas[:, cw:cw + div] = line
        else:
            canvas[ch:ch + div, :] = line
        if labels:
            rgb = _rgb_on(canvas, (1, 1, 1))
            ts = min(cw, ch) * 0.035
            cr.draw_label(rgb, names[0], at_start=True, text_size=ts)
            if horiz:
                sub = rgb[:, cw + div:]
                cr.draw_label(sub, names[1], at_start=True, text_size=ts)
            else:
                cr.draw_label(rgb, names[1], at_start=False, text_size=ts)
            canvas = rgb
        res["layout"] = "row" if horiz else "column"
        res["output"] = save(api, _maybe_opaque(canvas), output, overwrite=overwrite)
        return res
    # transparency / toggle: both images ContentScale.Fit into the first image's frame
    h, w = b0.shape[:2]
    lb = np.empty((h, w, 4), F32)
    lb[:] = bg
    _fit_rgba(lb, b0)
    la = np.empty((h, w, 4), F32)
    la[:] = bg
    _fit_rgba(la, a0)
    if mode == "transparency":
        t = progress / 100.0
        a = la[..., 3:4] * t
        out_a = a + lb[..., 3:4] * (1 - a)
        rgb = (la[..., :3] * a + lb[..., :3] * lb[..., 3:4] * (1 - a)) / np.maximum(out_a, 1e-6)
        canvas = np.dstack([rgb, out_a])
        if labels:
            canvas = _rgb_on(canvas, (1, 1, 1))
            cr.draw_label(canvas, names[0], at_start=True, text_size=min(w, h) * 0.035)
            cr.draw_label(canvas, names[1], at_start=False, text_size=min(w, h) * 0.035)
        res["opacity"] = round(t, 4)
        res["output"] = save(api, _maybe_opaque(canvas), output, overwrite=overwrite)
        return res
    # toggle
    p = out_path(api, output, overwrite=overwrite)
    if p.suffix.lower() != ".gif":
        raise invalid("toggle mode writes an animated GIF: use a .gif output", param="output")
    frames = []
    for i, layer in enumerate((lb, la)):
        rgb = _rgb_on(layer, (1, 1, 1))
        if labels:
            cr.draw_label(rgb, names[i], at_start=i == 0, text_size=min(w, h) * 0.035)
        frames.append(rgb)
    pal, _ = cr.shared_palette(frames)
    idx = [cr.quantize(f, pal) for f in frames]
    data = cr.write_gif([((0, 0, w, h), idx[0], toggle_ms), ((0, 0, w, h), idx[1], toggle_ms)], w, h,
                        bytes(pal.getpalette()[:768]))
    atomic_write(p, data, overwrite=overwrite)
    res["output"] = {"path": str(p), "width": w, "height": h, "bytes": len(data), "frames": 2}
    return res


def _fit_rgba(canvas: np.ndarray, img: np.ndarray) -> None:
    """ContentScale.Fit of ``img`` into an RGBA float canvas (source-over)."""
    h, w = canvas.shape[:2]
    left, top, dw, dh = cr._fit_rect(img.shape[1], img.shape[0], w, h)
    x0, y0, x1, y1 = int(round(left)), int(round(top)), int(round(left + dw)), int(round(top + dh))
    if x1 <= x0 or y1 <= y0:
        return
    r = _resize_exact(img, x1 - x0, y1 - y0)
    if r.ndim == 2:
        r = np.repeat(r[..., None], 3, 2)
    if r.shape[2] == 3:
        r = np.dstack([r, np.ones(r.shape[:2], F32)])
    dst = canvas[y0:y1, x0:x1]
    sa = r[..., 3:4]
    out_a = sa + dst[..., 3:4] * (1 - sa)
    rgb = (r[..., :3] * sa + dst[..., :3] * dst[..., 3:4] * (1 - sa)) / np.maximum(out_a, 1e-6)
    canvas[y0:y1, x0:x1] = np.dstack([rgb, out_a])


def _maybe_opaque(img: np.ndarray) -> np.ndarray:
    if img.shape[2] == 4 and float(img[..., 3].min()) >= 1.0:
        return img[..., :3]
    return img


@tool("compare_animation", "Animated before/after GIF", category="analysis", params=[
    File("before", "", file_kind="image", required=True),
    File("after", "", file_kind="image", required=True),
    Int("duration_seconds", 4, min=2, max=8, description="Loop duration (2..8 s, default 4)"),
    Choice("direction", "horizontal", choices=["horizontal", "vertical"],
           description="Movement direction of the divider"),
    Bool("show_labels", True, description="Draw the Before/After labels"),
    Int("max_size", 2048, min=720, max=4096,
        description="Longest side of the GIF (ImageToolbox offers 720, 1080, 2048, 4096; never upscales)"),
    Text("before_label", "Before", max_length=200), Text("after_label", "After", max_length=200),
    Text("output", "", description="Output .gif path"), Bool("overwrite", False),
], source={"project": ITB, "feature": "Compare: Animated before/after (CompareAnimationRenderer, writeGif)"})
def compare_animation(api, before, after, duration_seconds, direction, show_labels, max_size, before_label,
                      after_label, output, overwrite):
    """Export a looping before/after GIF: the divider holds 0.5 s at each end and sweeps with smoothstep
    easing in 50 ms frames; the 'after' image is revealed from the left (or top).  Size follows the
    'before' image fitted within ``max_size``; 'after' is fitted (letterboxed on white) into that frame."""
    p = out_path(api, output, overwrite=overwrite)
    if p.suffix.lower() != ".gif":
        raise invalid("compare_animation writes a GIF: use a .gif output", param="output")
    b0 = _load_compare(api, before)
    a0 = _load_compare(api, after)
    params = cr.AnimationParams(duration_seconds=duration_seconds, is_vertical=direction == "vertical",
                                show_labels=show_labels, max_size=max_size)
    data, meta = cr.render_animation(b0, a0, params, before_label=before_label, after_label=after_label)
    atomic_write(p, data, overwrite=overwrite)
    frames = meta["frames"]
    return {"output": {"path": str(p), "width": meta["width"], "height": meta["height"], "bytes": len(data)},
            "frame_count": len(frames), "duration_ms": sum(f["delay_ms"] for f in frames),
            "direction": direction, "frames": frames}


# Tools defined in private modules (registered on import): image scopes (waveform / vector scope /
# histogram), compression lab and multi-frame fusion.
from . import _compression_lab, _fusion, _scopes  # noqa: E402,F401
