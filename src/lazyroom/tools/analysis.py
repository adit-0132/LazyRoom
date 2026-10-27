"""Analysis tools: palette, histogram, comparison, duplicate detection, colour info."""
from __future__ import annotations

import colorsys
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.color import rgb_to_oklab, rgb_to_lab
from ..core.image import fit_within, resize
from ..core.params import Bool, Choice, Color, File, Float, Int, StringList, Text, color_to_hex, parse_color
from .registry import expand_inputs, read_image, save, tool

F32 = np.float32


@tool("palette", "Extract colour palette", category="analysis", params=[
    File("input", "", file_kind="image", required=True),
    Int("count", 6, min=1, max=32, description="Number of colours"),
    Choice("method", "kmeans", choices=["kmeans", "median_cut"]),
    Text("swatch_output", "", description="Optional path to save a swatch image (PNG/JPG)"),
    Bool("overwrite", False),
])
def palette(api, input, count, method, swatch_output, overwrite):
    """Dominant colours (OKLab k-means or median cut) with weights, hex, RGB and HSL values."""
    img, _ = read_image(api, input)
    small, _ = fit_within(img[..., :3], 400)
    small = np.clip(small, 0, 1)
    if method == "median_cut":
        from PIL import Image

        pil = Image.fromarray((small * 255).astype(np.uint8))
        q = pil.quantize(count, method=Image.Quantize.MEDIANCUT)
        pal = np.array(q.getpalette()[: count * 3], F32).reshape(-1, 3) / 255
        counts = np.bincount(np.asarray(q).ravel(), minlength=count)[: len(pal)].astype(float)
        cols = [(pal[i], counts[i]) for i in np.argsort(-counts)]
    else:
        from ..style.analysis import palette as kpal

        cols = [(np.array(parse_color(c["hex"])[:3]), c["weight"]) for c in kpal(rgb_to_oklab(small), count)]
    total = sum(w for _, w in cols) or 1
    out = []
    for rgb, w in cols:
        h, l, s = colorsys.rgb_to_hls(*[float(v) for v in rgb])
        out.append({"hex": color_to_hex(rgb), "rgb": [int(round(v * 255)) for v in rgb],
                    "hsl": [round(h * 360, 1), round(s * 100, 1), round(l * 100, 1)], "weight": round(w / total, 4)})
    res: dict[str, Any] = {"colors": out}
    if swatch_output:
        sw = np.zeros((120, 120 * len(out), 3), F32)
        for i, c in enumerate(out):
            sw[:, i * 120:(i + 1) * 120] = np.array(c["rgb"], F32) / 255
        res["swatch"] = save(api, sw, swatch_output, overwrite=overwrite)
    return res


@tool("histogram", "Histogram", category="analysis", params=[
    File("input", "", file_kind="image", required=True),
    Int("bins", 64, min=8, max=256),
    Bool("itb_histogram", False, description="Also return ImageToolbox's histogram data: 256-bin R/G/B counts and "
                                             "the 101-bin Brightness (CIE L*) histogram"),
    Text("render_output", "", description="Optional path: render the ImageToolbox histogram chart (PNG/JPG/...)"),
    Choice("render_type", "rgb", choices=["rgb", "brightness", "camera"],
           description="Chart type (ImageToolbox HistogramType: RGB, Brightness, Camera-like RGB)"),
    Int("render_width", 1000, min=100, max=4000), Int("render_height", 500, min=50, max=4000),
    Text("harmonization_color", "", description="Optional theme colour the RGB/Brightness line colours are "
                                                "blended with (25 % / 10 %), like ImageToolbox's primary colour"),
    Color("background", "#1B1B1F", description="Chart background (the app draws on its theme background)"),
    Bool("overwrite", False),
])
def histogram(api, input, bins, itb_histogram=False, render_output="", render_type="rgb", render_width=1000,
              render_height=500, harmonization_color="", background=(0x1B / 255, 0x1B / 255, 0x1F / 255, 1.0),
              overwrite=False):
    """Per-channel (R, G, B, luma) histograms and summary statistics; optionally ImageToolbox's RGB /
    Brightness / Camera-like RGB histogram data and a rendered chart."""
    img, _ = read_image(api, input)
    x = np.clip(fit_within(img[..., :3], 1024)[0], 0, 1)
    y = x @ np.array([0.2126, 0.7152, 0.0722], F32)
    out: dict[str, Any] = {"bins": bins}
    for name, ch in (("red", x[..., 0]), ("green", x[..., 1]), ("blue", x[..., 2]), ("luma", y)):
        h, _ = np.histogram(ch, bins=bins, range=(0, 1))
        out[name] = {"counts": h.tolist(), "mean": round(float(ch.mean()), 4),
                     "median": round(float(np.median(ch)), 4), "std": round(float(ch.std()), 4)}
    if itb_histogram or render_output:
        from ._histogram import itb_histogram as _itb, render_histogram

        data = _itb(img)
        if itb_histogram:
            out["itb"] = data
        if render_output:
            harm = None
            if harmonization_color:
                from ._color_models import from_float

                harm = from_float(parse_color(harmonization_color, "harmonization_color"))
            chart = render_histogram(data, render_type, render_width, render_height, background=background,
                                     harmonization=harm)
            out["render"] = save(api, chart, render_output, overwrite=overwrite)
            out["render"]["type"] = render_type
    return out


@tool("compare", "Compare images", category="analysis", params=[
    File("a", "", file_kind="image", required=True), File("b", "", file_kind="image", required=True),
    Text("diff_output", "", description="Optional path for a visual difference heat map"),
    Bool("overwrite", False),
    Choice("pixel_metric", "none", choices=["none", "all", "ssim", "ae", "mae", "ncc", "psnr", "rmse"],
           description="ImageToolbox 'Pixel by pixel' metrics (ImageDiffTool, 8-bit RGBA, b resized+padded to a)"),
    Float("threshold", 4.0, min=0, max=100, step=0.5,
          description="Pixel-by-pixel tolerance 0..100 (ImageToolbox default 4)"),
    Color("highlight_color", "#FF0000", description="Colour painted on differing pixels"),
    Text("highlight_output", "", description="Optional path: image a with differing pixels highlighted "
                                             "(uses pixel_metric, or AE when 'all'/'none')"),
])
def compare(api, a, b, diff_output, overwrite, pixel_metric="none", threshold=4.0,
            highlight_color=(1.0, 0.0, 0.0, 1.0), highlight_output=""):
    """Perceptual comparison: ΔE, SSIM, PSNR, colour-distribution distance; optional diff heat map and
    ImageToolbox's pixel-by-pixel metrics (SSIM, AE, MAE, NCC, PSNR, RMSE) with a highlight image."""
    from ..style.metrics import compare_images, delta_e76

    ia, _ = read_image(api, a)
    ib, _ = read_image(api, b)
    res: dict[str, Any] = {"metrics": compare_images(ia, ib), "size_a": [ia.shape[1], ia.shape[0]],
                           "size_b": [ib.shape[1], ib.shape[0]]}
    if diff_output:
        a3, _ = fit_within(ia[..., :3], 1600)
        b3 = resize(ib[..., :3], a3.shape[1], a3.shape[0])
        de = delta_e76(np.clip(a3, 0, 1), np.clip(b3, 0, 1))
        heat = cv2.applyColorMap(np.clip(de / 20 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        res["diff"] = save(api, cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(F32) / 255, diff_output,
                           overwrite=overwrite)
    if pixel_metric != "none" or highlight_output:
        from ._image_diff import pixel_by_pixel

        methods = ["ssim", "ae", "mae", "ncc", "psnr", "rmse"] if pixel_metric in ("all", "none") else [pixel_metric]
        if pixel_metric == "none":
            methods = []
        hl_method = pixel_metric if pixel_metric not in ("all", "none") else "ae"
        pb, hl = pixel_by_pixel(ia, ib, methods, threshold=threshold, highlight_color=highlight_color,
                                highlight_method=hl_method if highlight_output else None)
        res["pixel_by_pixel"] = pb
        if highlight_output:
            res["highlight"] = save(api, hl, highlight_output, overwrite=overwrite)
            res["highlight"]["method"] = hl_method
    return res


def _dhash(img: np.ndarray, size: int = 16) -> np.ndarray:
    g = cv2.resize(np.clip(img[..., :3], 0, 1) @ np.array([0.299, 0.587, 0.114], F32), (size + 1, size),
                   interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).ravel()


def _phash(img: np.ndarray) -> np.ndarray:
    g = cv2.resize(np.clip(img[..., :3], 0, 1) @ np.array([0.299, 0.587, 0.114], F32), (32, 32),
                   interpolation=cv2.INTER_AREA)
    d = cv2.dct(g.astype(np.float32))[:8, :8].ravel()[1:]
    return d > np.median(d)


@tool("duplicates", "Find duplicate / similar images", category="analysis", params=[
    StringList("inputs", [], min_items=1, file_kind="image", description="Files, folders or glob patterns"),
    Bool("recursive", True),
    Float("threshold", 0.12, min=0, max=0.5, step=0.01,
          description="Max normalised hash distance to call two images similar (0 = identical)"),
])
def duplicates(api, inputs, recursive, threshold):
    """Group visually identical or near-identical images using perceptual hashes (pHash + dHash)."""
    files = expand_inputs(api, inputs, recursive)
    hashes, errors = [], []
    from ..core.io import load_image

    for f in files:
        try:
            im, _ = load_image(api.ws.policy.check_read(f), max_pixels=api.ws.config.max_pixels)
            small, _ = fit_within(im, 256)
            hashes.append((f, np.concatenate([_phash(small), _dhash(small)])))
        except Exception as exc:
            errors.append({"input": f, "error": str(exc)})
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    if n:
        H = np.array([h for _, h in hashes])
        for i in range(n):
            d = (H[i + 1:] != H[i]).mean(1)
            for j in np.where(d <= threshold)[0]:
                jj = i + 1 + int(j)
                pairs.append({"a": hashes[i][0], "b": hashes[jj][0], "distance": round(float(d[j]), 4)})
                parent[find(jj)] = find(i)
    groups: dict[int, list[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(hashes[i][0])
    dup_groups = [g for g in groups.values() if len(g) > 1]
    return {"scanned": n, "groups": dup_groups, "pairs": pairs[:1000], "errors": errors}


@tool("color_info", "Colour conversions & harmonies", category="analysis", params=[
    Color("color", "#3366CC"),
])
def color_info(api, color):
    """Convert a colour between HEX/RGB/HSL/HSV/CMYK/Lab/OKLab/XYZ, name it (colour library) and list
    harmonious colours (``itb``: ImageToolbox's models, copy formats and 7 harmony types)."""
    r, g, b, a = color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    hv, sv, vv = colorsys.rgb_to_hsv(r, g, b)
    k = 1 - max(r, g, b)
    cmyk = [0, 0, 0, 100] if k >= 1 else [round((1 - c - k) / (1 - k) * 100, 1) for c in (r, g, b)] + [round(k * 100, 1)]
    arr = np.array([[[r, g, b]]], F32)
    lab = rgb_to_lab(arr)[0, 0]
    ok = rgb_to_oklab(arr)[0, 0]

    def rot(deg):
        rr, gg, bb = colorsys.hls_to_rgb(((h * 360 + deg) % 360) / 360, l, s)
        return color_to_hex((rr, gg, bb))

    from . import _color_models as cm

    c8 = cm.from_float(color)
    itb = cm.describe(c8)
    itb["harmonies"] = {t: [cm.hex_itb(c) for c in cm.harmony(c8, t)] for t in cm.HARMONY_TYPES}
    return {"hex": color_to_hex(color), "rgb": [round(v * 255) for v in (r, g, b)], "alpha": round(a, 3),
            "hsl": [round(h * 360, 1), round(s * 100, 1), round(l * 100, 1)],
            "hsv": [round(hv * 360, 1), round(sv * 100, 1), round(vv * 100, 1)], "cmyk": cmyk,
            "lab": [round(float(v), 2) for v in lab], "oklab": [round(float(v), 4) for v in ok],
            "harmonies": {"complementary": [rot(180)], "analogous": [rot(-30), rot(30)],
                          "triadic": [rot(120), rot(240)], "split_complementary": [rot(150), rot(210)],
                          "tetradic": [rot(90), rot(180), rot(270)]},
            "name": itb["name"], "xyz": itb["xyz"], "itb": itb}
