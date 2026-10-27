"""Model-specific pre/post-processing, ported from ImageToolbox.

Sources (Apache-2.0, T8RIN/ImageToolbox):
  lib/neural-tools/.../bgremover/{GenericBackgroundRemover,YoloSegRemover,*BackgroundRemover}.kt
  lib/neural-tools/.../inpaint/{LaMaProcessor,WatermarkRemoverProcessor}.kt
  lib/neural-tools/.../scans/UvDocUnwarper.kt
  feature/ai-tools/.../data/{DepthProcessor,StyleTransferProcessor,
      OptimizationStyleTransferProcessor,AiImageDetectorProcessor}.kt

All functions take float32 sRGB images in 0..1 (``(H, W, 3)`` unless noted) and
an optional render context used for cancellation.  ImageToolbox works on 8-bit
Android bitmaps; we keep float precision and use OpenCV resampling (bilinear for
``Bitmap.scale``, Lanczos-4 for Aire Lanczos3).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.errors import ErrorCode, LazyRoomError
from . import runtime as rt

F32 = np.float32
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], F32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], F32)


def _resize(x: np.ndarray, w: int, h: int, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    if x.shape[1] == w and x.shape[0] == h:
        return x
    out = cv2.resize(np.ascontiguousarray(x), (int(w), int(h)), interpolation=interp)
    if x.ndim == 3 and out.ndim == 2:
        out = out[..., None]
    return out


def _nchw(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None].astype(F32))


def _round(v: float) -> int:
    """Kotlin ``roundToInt`` (half away from zero for positives)."""
    return int(math.floor(v + 0.5))


def _flat(v: Any) -> np.ndarray:
    return np.asarray(v, dtype=F32).reshape(-1)


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------
# (name prefix, trainedSize) in AndroidAiToolsRepository.asBgRemover order
BG_REMOVERS = [("u2netp", 320), ("u2net", 320), ("inspyrenet", 1024), ("RMBG_1.4", 1024),
               ("birefnet_swin_tiny", 1024), ("modnet_portrait_matting", 512), ("isnet", 1024), ("yolo", 1024)]


def bg_trained_size(name: str) -> int:
    for prefix, size in BG_REMOVERS:
        if name.startswith(prefix):
            return size
    raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"{name} is not a background-removal model")


def background_mask(name: str, path: Path, rgb: np.ndarray, ctx=None) -> np.ndarray:
    """Foreground alpha (H, W) in 0..1 (GenericBackgroundRemover / YoloSegRemover)."""
    size = bg_trained_size(name)
    if name.startswith("yolo"):
        return _yolo_mask(path, rgb, size, ctx)
    H, W = rgb.shape[:2]
    sess = rt.load_session(path)
    x = _resize(np.clip(rgb, 0, 1), size, size)  # Aire bilinear, no normalisation (as upstream)
    out = rt.run(sess, {sess.input_names[0]: _nchw(x)}, ctx)[0]
    m = np.asarray(out, F32)
    while m.ndim > 2:
        m = m[0]
    m = np.clip(np.round(m * 255), 0, 255) / 255.0
    return np.clip(_resize(m.astype(F32), W, H), 0, 1)


def _yolo_mask(path: Path, rgb: np.ndarray, size: int, ctx) -> np.ndarray:
    H, W = rgb.shape[:2]
    sess = rt.load_session(path)
    x = _resize(np.clip(rgb, 0, 1), size, size)
    outs = rt.run(sess, {sess.input_names[0]: _nchw(x)}, ctx)
    det, proto = np.asarray(outs[0], F32)[0], np.asarray(outs[1], F32)[0]  # (4+80+32, N), (32, ph, pw)
    ph, pw = proto.shape[1:]
    scores = det[4:84]
    best = scores.argmax(axis=0)
    best_score = scores.max(axis=0)
    keep = np.nonzero(best_score >= 0.45)[0]
    order = keep[np.argsort(-best_score[keep], kind="stable")]
    boxes = det[:4]

    def iou(a: int, b: int) -> float:
        ax1, ay1 = boxes[0, a] - boxes[2, a] / 2, boxes[1, a] - boxes[3, a] / 2
        ax2, ay2 = boxes[0, a] + boxes[2, a] / 2, boxes[1, a] + boxes[3, a] / 2
        bx1, by1 = boxes[0, b] - boxes[2, b] / 2, boxes[1, b] - boxes[3, b] / 2
        bx2, by2 = boxes[0, b] + boxes[2, b] / 2, boxes[1, b] + boxes[3, b] / 2
        inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)

    selected: list[int] = []
    rest = list(order)
    while rest:
        b = rest.pop(0)
        selected.append(int(b))
        rest = [r for r in rest if not (best[r] == best[b] and iou(b, r) > 0.5)]
    mask = np.zeros((ph, pw), F32)
    for d in selected[:8]:
        cx, cy, bw, bh = boxes[:, d]
        left = min(max(int((cx - bw / 2) / size * pw), 0), pw - 1)
        top = min(max(int((cy - bh / 2) / size * ph), 0), ph - 1)
        right = min(max(int((cx + bw / 2) / size * pw), 0), pw - 1)
        bottom = min(max(int((cy + bh / 2) / size * ph), 0), ph - 1)
        coeffs = det[84:116, d]
        s = np.tensordot(coeffs, proto[:, top:bottom + 1, left:right + 1], axes=1)
        v = 1 / (1 + np.exp(-s))
        mask[top:bottom + 1, left:right + 1] = np.maximum(mask[top:bottom + 1, left:right + 1], v)
    alpha = np.where(mask >= 0.55, 255, np.where(mask <= 0.35, 0, ((mask - 0.35) / 0.2 * 255).astype(np.int32)))
    alpha = np.clip(alpha, 0, 255).astype(F32) / 255
    return np.clip(_resize(alpha, W, H), 0, 1)


def heal_alpha(mask: np.ndarray, original_alpha: np.ndarray | None) -> np.ndarray:
    """``Bitmap.healAlpha``: where the source was not fully opaque, keep its alpha."""
    if original_alpha is None:
        return mask
    return np.where(original_alpha < 1.0, original_alpha, mask).astype(F32)


# ---------------------------------------------------------------------------
# LaMa inpainting / watermark removal
# ---------------------------------------------------------------------------
LAMA_SIZE = 512


def lama_inpaint(name: str, path: Path, rgb: np.ndarray, mask: np.ndarray, ctx=None) -> np.ndarray:
    """LaMaProcessor.inpaint: whole image at 512x512, mask = (value > 0), result resized back."""
    H, W = rgb.shape[:2]
    fast = "FAST" in name  # isFastModel -> output already 0..1
    sess = rt.load_session(path)
    x = _resize(np.clip(rgb, 0, 1), LAMA_SIZE, LAMA_SIZE, cv2.INTER_LANCZOS4)
    m = _resize(mask.astype(F32), LAMA_SIZE, LAMA_SIZE, cv2.INTER_NEAREST)
    m = (m > 0.5 / 255).astype(F32)  # 8-bit red channel > 0
    names = sess.input_names
    img_name = "image" if "image" in names else names[0]
    mask_name = "mask" if "mask" in names else names[1]
    out = np.asarray(rt.run(sess, {img_name: _nchw(np.clip(x, 0, 1)), mask_name: m[None, None]}, ctx)[0], F32)[0]
    amp = 255.0 if fast else 1.0
    y = np.clip(np.trunc(out[:3] * amp), 0, 255).transpose(1, 2, 0) / 255.0
    return np.clip(_resize(y.astype(F32), W, H, cv2.INTER_LANCZOS4), 0, 1)


WATERMARK_SIZE = 512
WATERMARK_THRESHOLD = 0.18


def watermark_mask(path: Path, rgb: np.ndarray, ctx=None) -> np.ndarray:
    """WatermarkRemoverProcessor.findWatermarkMask -> binary (H, W) mask."""
    H, W = rgb.shape[:2]
    sess = rt.load_session(path)
    x = _resize(np.clip(rgb, 0, 1), WATERMARK_SIZE, WATERMARK_SIZE, cv2.INTER_LANCZOS4)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    name = "input" if "input" in sess.input_names else sess.input_names[0]
    prob = np.asarray(rt.run(sess, {name: _nchw(x)}, ctx)[0], F32).reshape(WATERMARK_SIZE, WATERMARK_SIZE)
    m = (prob >= WATERMARK_THRESHOLD).astype(F32)
    return _resize(m, W, H, cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# UVDoc document unwarping
# ---------------------------------------------------------------------------
UVDOC_W, UVDOC_H = 496, 720


def uvdoc_unwarp(path: Path, img: np.ndarray, ctx=None) -> np.ndarray:
    """UvDocUnwarper.unwarp: predict a [-1, 1] sampling grid and remap the full-resolution image."""
    H, W = img.shape[:2]
    sess = rt.load_session(path)
    x = _resize(np.clip(img[..., :3], 0, 1), UVDOC_W, UVDOC_H)
    grid = np.asarray(rt.run(sess, {sess.input_names[0]: _nchw(x)}, ctx)[0], F32)[0]  # (2, gh, gw)
    gh, gw = grid.shape[1:]
    gc = (np.arange(W, dtype=F32) / (W - 1) * (gw - 1)) if W > 1 else np.zeros(W, F32)
    gr = (np.arange(H, dtype=F32) / (H - 1) * (gh - 1)) if H > 1 else np.zeros(H, F32)
    mx, my = np.meshgrid(gc, gr)
    nx = cv2.remap(np.ascontiguousarray(grid[0]), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ny = cv2.remap(np.ascontiguousarray(grid[1]), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    sx = np.clip((nx + 1) * 0.5 * (W - 1), 0, W - 1).astype(F32)
    sy = np.clip((ny + 1) * 0.5 * (H - 1), 0, H - 1).astype(F32)
    out = cv2.remap(np.ascontiguousarray(img), sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if out.ndim == 2:
        out = out[..., None]
    return out.astype(F32)


# ---------------------------------------------------------------------------
# Depth (DepthProcessor.kt)
# ---------------------------------------------------------------------------
DEPTH_OUTPUT_NAMES = ("predicted_depth", "depth", "output")
DEPTH_EFFECTS = ["map", "lens_blur", "background_blur", "portrait", "color_grade", "depth_edges", "normal_map",
                 "stereo", "depth_mask", "cutout", "focus_color", "fog", "relight"]
_SPECTRAL = np.array([(94, 79, 162), (50, 136, 189), (102, 194, 165), (171, 221, 164), (230, 245, 152),
                      (255, 255, 191), (254, 224, 139), (253, 174, 97), (244, 109, 67), (213, 62, 79),
                      (158, 1, 66)], F32) / 255
FOG_COLOR = np.array([222, 232, 242], F32) / 255
FAR_DEPTH_COLOR = np.array([35, 72, 104], F32) / 255
NEAR_DEPTH_COLOR = np.array([245, 151, 79], F32) / 255
EDGE_DARK = np.array([18, 21, 28], F32) / 255
EDGE_LIGHT = np.array([237, 242, 248], F32) / 255
LUMA = np.array([0.2126, 0.7152, 0.0722], F32)


def depth_contract(name: str) -> str:
    if name.startswith("depth_anything_v2_"):
        return "disparity"
    if name.startswith("depth_anything_v3_mono_"):
        return "direct_with_sky"
    if name.startswith("depth_anything_v3_"):
        return "direct"
    return "disparity"


def _depth_input_size(w: int, h: int, max_side: int) -> tuple[int, int]:
    scale = max_side / max(w, h)

    def aligned(v: int) -> int:
        return min(max(_round(v * scale / 14) * 14, 14), max_side)

    return aligned(w), aligned(h)


def _percentile_sorted(sorted_vals: np.ndarray, fraction: float) -> float:
    return float(np.quantile(sorted_vals, min(max(fraction, 0.0), 1.0)))


def infer_depth(name: str, path: Path, rgb: np.ndarray, ctx=None) -> np.ndarray:
    """Normalised depth field (near = 1, far = 0) at model resolution."""
    sess = rt.load_session(path)
    inp = sess.inputs[0]
    shape = list(inp.shape or [])
    if len(shape) not in (4, 5):
        raise LazyRoomError(ErrorCode.OPERATION_FAILED, f"Unsupported depth input rank {len(shape)}")
    contract = depth_contract(name)
    fw = shape[-1] if isinstance(shape[-1], int) and shape[-1] > 0 else None
    fh = shape[-2] if isinstance(shape[-2], int) and shape[-2] > 0 else None
    H, W = rgb.shape[:2]
    if fw and fh:
        s = min(fw / W, fh / H)
        cw, ch = min(max(_round(W * s), 1), fw), min(max(_round(H * s), 1), fh)
        left, top = (fw - cw) // 2, (fh - ch) // 2
        scaled = _resize(rgb, cw, ch)
        frame = np.pad(scaled, ((top, fh - ch - top), (left, fw - cw - left), (0, 0)), mode="edge")
        inset = 7 if contract == "direct_with_sky" else 0
        content = (left, top, cw, ch)
    else:
        w2, h2 = _depth_input_size(W, H, 518 if contract == "disparity" else 504)
        frame = _resize(rgb, w2, h2)
        inset = 0
        content = (0, 0, w2, h2)
    fh_, fw_ = frame.shape[:2]
    t = _nchw((np.clip(frame, 0, 1) - IMAGENET_MEAN) / IMAGENET_STD)
    if len(shape) == 5:
        t = t[:, None]
    outs = rt.run(sess, {inp.name: t}, ctx)
    names = sess.output_names
    idx = next((i for i, n in enumerate(names) if n in DEPTH_OUTPUT_NAMES), 0)
    depth = _flat(outs[idx]).copy()
    if depth.size != fw_ * fh_:
        raise LazyRoomError(ErrorCode.OPERATION_FAILED, f"Unexpected depth output size {depth.size}, "
                                                          f"expected {fw_ * fh_}")
    if contract == "direct_with_sky":
        sky = _flat(outs[-1])
        if sky.size == depth.size:
            _apply_sky_depth(depth, sky)
    elif contract == "direct" and "confidence" in names:
        conf = _flat(outs[names.index("confidence")])
        if conf.size == depth.size:
            _suppress_uncertain_background(depth, conf, fw_, fh_)
    depth = depth.reshape(fh_, fw_)
    left, top, cw, ch = content
    if inset or (left, top, cw, ch) != (0, 0, fw_, fh_):
        depth = depth[top + inset: top + ch - inset, left + inset: left + cw - inset]
    return _normalize_depth(depth, direct=contract != "disparity")


def _apply_sky_depth(depth: np.ndarray, sky: np.ndarray) -> None:
    non_sky = sky < 0.3
    vals = depth[non_sky & np.isfinite(depth)]
    sky_count = int((~non_sky).sum())
    if vals.size <= 10 or sky_count <= 10:
        return
    far = _percentile_sorted(np.sort(vals), 0.99)
    depth[sky >= 0.3] = far


def _suppress_uncertain_background(depth: np.ndarray, conf: np.ndarray, w: int, h: int) -> None:
    finite = conf[np.isfinite(conf)]
    if finite.size <= 10:
        return
    thr = _percentile_sorted(np.sort(finite), 0.3)
    cand = (conf <= thr).reshape(h, w).astype(np.uint8)
    n, labels = cv2.connectedComponents(cand, connectivity=4)
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    border_cand = np.concatenate([cand[0], cand[-1], cand[:, 0], cand[:, -1]]).astype(bool)
    lab = np.unique(border[border_cand])
    bg = (np.isin(labels, lab) & cand.astype(bool)).reshape(-1)
    if bg.sum() <= 10:
        return
    fg = depth[~bg & np.isfinite(depth)]
    if fg.size <= 10:
        return
    depth[bg] = _percentile_sorted(np.sort(fg), 0.99)


def _normalize_depth(values: np.ndarray, *, direct: bool) -> np.ndarray:
    v = values.astype(np.float64)
    if direct:
        with np.errstate(divide="ignore", invalid="ignore"):
            nv = np.where(np.isfinite(v) & (v > 1e-6), 1.0 / v, np.nan)
        fin = np.sort(nv[np.isfinite(nv)])
        if fin.size:
            lo, hi = _percentile_sorted(fin, 0.02), _percentile_sorted(fin, 0.98)
        else:
            lo, hi = math.inf, -math.inf
    else:
        nv = v
        fin = nv[np.isfinite(nv)]
        lo, hi = (float(fin.min()), float(fin.max())) if fin.size else (math.inf, -math.inf)
    rng = hi - lo
    if not math.isfinite(rng) or rng <= 1e-6:
        return np.zeros(values.shape, F32)
    out = (nv - lo) / rng
    out = np.where(np.isfinite(out), np.clip(out, 0, 1), 0.0)
    return out.astype(F32)


def depth_from_image(img: np.ndarray) -> np.ndarray:
    """``loadDepthMap``: decode a user depth map (grayscale luma, or the spectral palette of the
    'map' effect when > 5% of pixels are coloured).  Loaded at <= 518 px like upstream."""
    rgb = np.clip(img[..., :3], 0, 1)
    h, w = rgb.shape[:2]
    if max(w, h) > 518:
        s = 518 / max(w, h)
        rgb = _resize(rgb, max(_round(w * s), 1), max(_round(h * s), 1), cv2.INTER_AREA)
    q = np.round(rgb * 255)
    spectral = ((q.max(axis=2) - q.min(axis=2)) > 12).sum() > q.shape[0] * q.shape[1] * 0.05
    if not spectral:
        return (q @ LUMA / 255).astype(F32)
    px = q.reshape(-1, 3)
    best_pos = np.zeros(len(px), F32)
    best_d = np.full(len(px), np.inf, F32)
    pal = _SPECTRAL * 255
    for i in range(len(pal) - 1):
        a, b = pal[i], pal[i + 1]
        delta = b - a
        amt = np.clip(((px - a) @ delta) / float(delta @ delta), 0, 1)
        proj = a + amt[:, None] * delta
        d = ((px - proj) ** 2).sum(axis=1)
        better = d < best_d
        best_d = np.where(better, d, best_d)
        best_pos = np.where(better, (i + amt) / (len(pal) - 1), best_pos)
    return best_pos.reshape(q.shape[:2]).astype(F32)


def sample_field(field: np.ndarray, W: int, H: int) -> np.ndarray:
    """``FieldSampler``: align-corners bilinear resampling of a (fh, fw) field to (H, W)."""
    fh, fw = field.shape

    def axis(n_out: int, n_in: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = np.arange(n_out, dtype=np.float64) * (n_in - 1) / max(n_out - 1, 1)
        lo = np.clip(pos.astype(np.int64), 0, n_in - 1)
        hi = np.minimum(lo + 1, n_in - 1)
        return lo, hi, (pos - lo).astype(F32)

    xl, xr, xm = axis(W, fw)
    yt, yb, ym = axis(H, fh)
    top = field[yt][:, xl] + (field[yt][:, xr] - field[yt][:, xl]) * xm
    bot = field[yb][:, xl] + (field[yb][:, xr] - field[yb][:, xl]) * xm
    return (top + (bot - top) * ym[:, None]).astype(F32)


def _smooth(e0: float, e1: float, v: np.ndarray) -> np.ndarray:
    x = np.clip((v - e0) / max(e1 - e0, 1e-6), 0, 1)
    return x * x * (3 - 2 * x)


def _mix(a: np.ndarray, b: np.ndarray, amount: np.ndarray | float) -> np.ndarray:
    m = np.clip(amount, 0, 1)
    if isinstance(m, np.ndarray) and m.ndim == 2:
        m = m[..., None]
    return a * (1 - m) + b * m


def _blur_frame(rgb: np.ndarray, radius: int) -> np.ndarray:
    H, W = rgb.shape[:2]
    scale = min(1.0, 1536.0 / max(W, H))
    w, h = max(_round(W * scale), 1), max(_round(H * scale), 1)
    small = _resize(rgb, w, h)
    r = max(min(radius, min(w, h) // 2), 1)
    k = 2 * r + 1
    b = cv2.blur(small, (k, 1), borderType=cv2.BORDER_REPLICATE)
    b = cv2.blur(b, (1, k), borderType=cv2.BORDER_REPLICATE)
    xs = (np.arange(W, dtype=np.int64) * w) // W
    ys = (np.arange(H, dtype=np.int64) * h) // H
    return b[ys][:, xs]


def _shifted(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """a[y+dy, x+dx] with clamping."""
    H, W = a.shape[:2]
    ys = np.clip(np.arange(H) + dy, 0, H - 1)
    xs = np.clip(np.arange(W) + dx, 0, W - 1)
    return a[ys][:, xs]


def render_depth_effect(rgb: np.ndarray, alpha: np.ndarray | None, depth: np.ndarray, effect: str, *,
                        strength: float = 65, focus: float = 65, focus_range: float = 18,
                        light_angle: float = 315) -> tuple[np.ndarray, np.ndarray | None]:
    """DepthProcessor.renderEffect.  Returns (rgb, alpha)."""
    H, W = rgb.shape[:2]
    s = strength / 100
    sc = min(max(s, 0.0), 1.0)
    rng = max(focus_range / 100, 0.01)
    f = focus / 100

    def up(field: np.ndarray) -> np.ndarray:
        return sample_field(field, W, H)

    def selection(d: np.ndarray) -> np.ndarray:
        return 1 - _smooth(rng * 0.55, rng, np.abs(d - f))

    def gray3(x: np.ndarray) -> np.ndarray:
        g = np.round(np.clip(x @ LUMA, 0, 1) * 255) / 255
        return np.repeat(g[..., None], 3, axis=2)

    if effect == "relight":
        a = math.radians(light_angle)
        lx, ly = math.cos(a), math.sin(a)
        nx = (_shifted(depth, 0, -1) - _shifted(depth, 0, 1)) * 2.5
        ny = (_shifted(depth, -1, 0) - _shifted(depth, 1, 0)) * 2.5
        inv = 1 / np.sqrt(nx * nx + ny * ny + 1)
        light = np.clip((nx * lx + ny * ly + 0.65) * inv, -1, 1).astype(F32)
        factor = np.clip(1 + up(light) * s * 0.8, 0.25, 1.8)
        return np.clip(rgb * factor[..., None], 0, 1), alpha
    d = up(depth)
    if effect == "map":
        pos = np.clip(d, 0, 1) * (len(_SPECTRAL) - 1)
        i0 = np.clip(pos.astype(np.int64), 0, len(_SPECTRAL) - 1)
        i1 = np.minimum(i0 + 1, len(_SPECTRAL) - 1)
        out = _mix(_SPECTRAL[i0], _SPECTRAL[i1], pos - i0)
    elif effect == "fog":
        out = _mix(rgb, FOG_COLOR, (1 - d) ** 1.35 * sc * 0.9)
    elif effect == "lens_blur":
        blur = _blur_frame(rgb, max(_round(s * 36), 1))
        out = _mix(rgb, blur, _smooth(rng * 0.2, rng, np.abs(d - f)) * s)
    elif effect == "background_blur":
        blur = _blur_frame(rgb, max(_round(sc * 36), 1))
        out = _mix(rgb, blur, np.power(np.clip(1 - d, 0, None), 1.45) * sc)
    elif effect == "portrait":
        sel = selection(d)
        bg = (1 - sel) * sc
        muted = _mix(rgb, gray3(rgb), bg * 0.6)
        light = 1 - bg * 0.22 + sel * sc * 0.08
        out = muted * light[..., None]
    elif effect == "color_grade":
        grade = _mix(FAR_DEPTH_COLOR[None, None], NEAR_DEPTH_COLOR[None, None], _smooth(0.08, 0.92, d))
        out = _mix(rgb, grade, sc * 0.38)
    elif effect in ("depth_edges", "normal_map"):
        hs, vs = W / depth.shape[1], H / depth.shape[0]
        left, right = _shifted(d, 0, -1), _shifted(d, 0, 1)
        top, bottom = _shifted(d, -1, 0), _shifted(d, 1, 0)
        if effect == "depth_edges":
            mag = np.sqrt(((right - left) * hs) ** 2 + ((bottom - top) * vs) ** 2)
            edge = _smooth(0.008, 0.075, mag) * sc
            lum = rgb @ LUMA
            target = np.where((lum > 0.55)[..., None], EDGE_DARK, EDGE_LIGHT)
            out = _mix(rgb, target, edge)
        else:
            relief = 2 + sc * 22
            nx = (left - right) * relief * hs
            ny = (top - bottom) * relief * vs
            inv = 1 / np.sqrt(nx * nx + ny * ny + 1)
            out = np.dstack([nx * inv * 0.5 + 0.5, ny * inv * 0.5 + 0.5, inv])
    elif effect == "stereo":
        max_shift = min(W * 0.035, 96.0) * sc
        shift = np.floor((d - f) * max_shift + 0.5).astype(np.int64)
        xs = np.arange(W)[None, :]
        ys = np.arange(H)[:, None]
        lft = rgb[ys, np.clip(xs + shift, 0, W - 1)]
        rgt = rgb[ys, np.clip(xs - shift, 0, W - 1)]
        out = np.dstack([lft[..., 0], rgt[..., 1], rgt[..., 2]])
    elif effect == "depth_mask":
        out = np.repeat(selection(d)[..., None], 3, axis=2)
    elif effect == "cutout":
        base = alpha if alpha is not None else np.ones((H, W), F32)
        return rgb, (base * selection(d)).astype(F32)
    elif effect == "focus_color":
        out = _mix(rgb, gray3(rgb), (1 - selection(d)) * sc)
    else:
        raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"Unknown depth effect {effect}")
    return np.clip(out, 0, 1).astype(F32), alpha


# ---------------------------------------------------------------------------
# Style transfer (StyleTransferProcessor.kt)
# ---------------------------------------------------------------------------
MAX_CONTENT_SIZE, ARBITRARY_CONTENT_SIZE, DEFAULT_CONTENT_SIZE = 2048, 1024, 500
STYLE_IMAGE_SIZE = 256


def _scaled_for_prediction(x: np.ndarray, max_size: int) -> np.ndarray:
    h, w = x.shape[:2]
    s = max_size / max(w, h)
    return _resize(x, max(int(w * s), 1), max(int(h * s), 1))


def _scaled_for_inference(x: np.ndarray, max_size: int) -> np.ndarray:
    h, w = x.shape[:2]
    if max(w, h) <= max_size:
        return x
    return _scaled_for_prediction(x, max_size)


def _center_crop(x: np.ndarray, size: int) -> np.ndarray:
    h, w = x.shape[:2]
    c = min(w, h)
    y0, x0 = (h - c) // 2, (w - c) // 2
    return _resize(x[y0:y0 + c, x0:x0 + c], size, size)


def _pad_to(x: np.ndarray, multiple: int) -> np.ndarray:
    h, w = x.shape[:2]
    return np.pad(x, ((0, -(-h // multiple) * multiple - h), (0, -(-w // multiple) * multiple - w), (0, 0)),
                  mode="edge")


def style_pack_kind(name: str) -> str:
    return {"arbitrary_style_transfer_onnx.zip": "arbitrary", "microast_style_transfer_onnx.zip": "microast",
            "aesfa_style_transfer_onnx.zip": "aesfa",
            "vgg19_optimization_style_transfer_onnx.zip": "vgg19"}[name]


def style_transfer(name: str, pack_dir: Path, content: np.ndarray, style: np.ndarray, strength: float,
                   ctx=None) -> np.ndarray:
    kind = style_pack_kind(name)
    if strength <= 0:
        return content.copy()
    if kind == "vgg19":
        return vgg19_style_transfer(pack_dir, content, style, strength, ctx)
    amount = min(max(strength / 100, 0.0), 1.0)
    H, W = content.shape[:2]
    if max(W, H) <= MAX_CONTENT_SIZE:
        max_size = DEFAULT_CONTENT_SIZE
    elif kind == "arbitrary":
        max_size = ARBITRARY_CONTENT_SIZE
    else:
        max_size = MAX_CONTENT_SIZE
    x = _scaled_for_inference(content, max_size)
    if kind == "arbitrary":
        pred = rt.load_session(pack_dir / "style_mobilenet.onnx", opt_level="all")
        tr = rt.load_session(pack_dir / "transformer_separable.onnx", opt_level="all")

        def predict(img: np.ndarray) -> np.ndarray:
            return _flat(rt.run(pred, {pred.input_names[0]: img[None].astype(F32)}, ctx)[0])[:100]

        # ImageGetter.getImage(uri, 256) then scaledForPrediction(256)
        style_emb = predict(_scaled_for_prediction(_scaled_for_inference(style, STYLE_IMAGE_SIZE), STYLE_IMAGE_SIZE))
        if amount < 1:
            content_emb = predict(_scaled_for_prediction(x, STYLE_IMAGE_SIZE))
            style_emb = content_emb + (style_emb - content_emb) * amount
        names = tr.input_names
        cname = next(i.name for i in tr.inputs if (i.shape or [None])[-1] == 3)
        sname = next(n for n in names if n != cname)
        y = np.asarray(rt.run(tr, {cname: x[None].astype(F32), sname: style_emb.reshape(1, 1, 1, 100)}, ctx)[0],
                       F32)[0]
        out = np.clip(y[: x.shape[0], : x.shape[1], :3], 0, 1)
    else:
        prefix = "microast" if kind == "microast" else "aesfa"
        pred = rt.load_session(pack_dir / f"{prefix}_style.onnx", opt_level="all")
        tr = rt.load_session(pack_dir / f"{prefix}_transform.onnx", opt_level="all")
        normalize = kind == "aesfa"

        def to_in(img: np.ndarray) -> np.ndarray:
            v = np.clip(img, 0, 1)
            return _nchw((v - IMAGENET_MEAN) / IMAGENET_STD if normalize else v)

        if kind == "microast":
            st = _scaled_for_prediction(style, STYLE_IMAGE_SIZE)
            code = _flat(rt.run(pred, {pred.input_names[0]: to_in(st)}, ctx)[0])[:512]
            multiple = 4
            feeds = {"style_code": code.reshape(1, 512, 1, 1), "strength": np.array([1.0], F32)}
        else:
            st = _center_crop(style, STYLE_IMAGE_SIZE)
            outs = rt.run(pred, {pred.input_names[0]: to_in(st)}, ctx)
            feeds = {"style_high": _flat(outs[0])[:1152].reshape(1, 128, 3, 3),
                     "style_low": _flat(outs[1])[:1152].reshape(1, 128, 3, 3)}
            multiple = 8
        xp = _pad_to(x, multiple)
        feeds["content"] = to_in(xp)
        y = np.asarray(rt.run(tr, feeds, ctx)[0], F32)[0, :3]
        if normalize:
            y = y * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
        y = y.transpose(1, 2, 0)[: x.shape[0], : x.shape[1]]
        out = np.clip(x + (y - x) * amount, 0, 1)
    if out.shape[:2] != (H, W):
        out = _resize(out.astype(F32), W, H)
    return np.clip(out, 0, 1).astype(F32)


# --- VGG19 optimisation (OptimizationStyleTransferProcessor.kt) ---------------
VGG_MEAN = np.array([123.675, 116.28, 103.53], F32)
VGG_PIXEL_BUDGET = 384 * 384
VGG_STEPS = 240


def _vgg_prepare(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(math.sqrt(VGG_PIXEL_BUDGET / (w * h)), 1.0)
    tw = -(-max(_round(w * s), 16) // 16) * 16
    th = -(-max(_round(h * s), 16) // 16) * 16
    x = _resize(img, tw, th)
    return (np.clip(x, 0, 1) * 255 - VGG_MEAN).transpose(2, 0, 1)[None].astype(F32)


def _cubic(x1, f1, g1, x2, f2, g2, lo=None, hi=None) -> float:
    lo = min(x1, x2) if lo is None else lo
    hi = max(x1, x2) if hi is None else hi
    d1 = g1 + g2 - 3.0 * (f1 - f2) / (x1 - x2)
    sq = d1 * d1 - g1 * g2
    if sq < 0 or not math.isfinite(sq):
        return (lo + hi) / 2
    d2 = math.sqrt(sq)
    if x1 <= x2:
        pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
    else:
        pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
    return min(max(pos, lo), hi)


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)))


def _strong_wolfe(x, t, d, loss, g, gtd, evaluate):
    c1, c2, max_ls = 1e-4, 0.9, 25

    def at(step):
        return evaluate((x + d * F32(step)).astype(F32))

    f_new, g_new = at(t)
    gtd_new = _dot(g_new, d)
    t_prev, f_prev, g_prev, gtd_prev = 0.0, loss, g.copy(), gtd
    bracket = None
    it = 0
    while it < max_ls:
        if f_new > loss + c1 * t * gtd or (it > 1 and f_new >= f_prev):
            bracket = [[t_prev, f_prev, g_prev, gtd_prev], [t, f_new, g_new, gtd_new]]
            break
        if abs(gtd_new) <= -c2 * gtd:
            return t, f_new, g_new
        if gtd_new >= 0:
            bracket = [[t_prev, f_prev, g_prev, gtd_prev], [t, f_new, g_new, gtd_new]]
            break
        nxt = _cubic(t_prev, f_prev, gtd_prev, t, f_new, gtd_new, t + 0.01 * (t - t_prev), t * 10)
        t_prev, f_prev, g_prev, gtd_prev = t, f_new, g_new, gtd_new
        t = nxt
        f_new, g_new = at(t)
        gtd_new = _dot(g_new, d)
        it += 1
    pts = bracket or [[0.0, loss, g.copy(), gtd], [t, f_new, g_new, gtd_new]]
    insuf = False
    dmax = float(np.abs(d).max()) if d.size else 0.0
    while it < max_ls:
        lo_i = 0 if pts[0][1] <= pts[1][1] else 1
        hi_i = 1 - lo_i
        low, high = pts[lo_i], pts[hi_i]
        if abs(high[0] - low[0]) * dmax < 1e-9:
            break
        t = _cubic(pts[0][0], pts[0][1], pts[0][3], pts[1][0], pts[1][1], pts[1][3])
        mn, mx = min(pts[0][0], pts[1][0]), max(pts[0][0], pts[1][0])
        eps = 0.1 * (mx - mn)
        if min(mx - t, t - mn) < eps:
            if insuf or t >= mx or t <= mn:
                t = mx - eps if abs(t - mx) < abs(t - mn) else mn + eps
                insuf = False
            else:
                insuf = True
        else:
            insuf = False
        f_new, g_new = at(t)
        gtd_new = _dot(g_new, d)
        it += 1
        if f_new > loss + c1 * t * gtd or f_new >= low[1]:
            pts[hi_i] = [t, f_new, g_new, gtd_new]
        else:
            if abs(gtd_new) <= -c2 * gtd:
                return t, f_new, g_new
            if gtd_new * (high[0] - low[0]) >= 0:
                pts[hi_i] = low
            pts[lo_i] = [t, f_new, g_new, gtd_new]
    best = min(pts, key=lambda p: p[1])
    return best[0], best[1], best[2]


def lbfgs(x: np.ndarray, evaluate, steps: int = VGG_STEPS, history: int = 10, ctx=None) -> np.ndarray:
    """Port of ``optimizeLbfgs`` (PyTorch-style L-BFGS with strong-Wolfe line search)."""
    x = x.astype(F32).copy()
    loss, g = evaluate(x)
    d = np.zeros_like(x)
    prev_g = None
    prev_t = 0.0
    old_dirs: list[np.ndarray] = []
    old_steps: list[np.ndarray] = []
    rho: list[float] = []
    h_diag = 1.0
    for i in range(steps):
        if ctx is not None:
            ctx.check()
        if i == 0:
            d = -g
        else:
            y = g - prev_g
            s = d * F32(prev_t)
            ys = _dot(y, s)
            if ys > 1e-10:
                if len(old_dirs) == history:
                    old_dirs.pop(0)
                    old_steps.pop(0)
                    rho.pop(0)
                old_dirs.append(y)
                old_steps.append(s)
                rho.append(1.0 / ys)
                h_diag = ys / _dot(y, y)
            al = [0.0] * len(old_dirs)
            q = (-g).astype(F32)
            for k in range(len(old_dirs) - 1, -1, -1):
                al[k] = _dot(old_steps[k], q) * rho[k]
                q = (q + old_dirs[k] * F32(-al[k])).astype(F32)
            d = (q * F32(h_diag)).astype(F32)
            for k in range(len(old_dirs)):
                be = _dot(old_dirs[k], d) * rho[k]
                d = (d + old_steps[k] * F32(al[k] - be)).astype(F32)
        prev_g = g.copy()
        gtd = _dot(g, d)
        if gtd >= -1e-9:
            break
        t0 = min(1.0, 1.0 / float(np.abs(g.astype(np.float64)).sum())) if i == 0 else 1.0
        t, loss, g = _strong_wolfe(x, t0, d, loss, g, gtd, evaluate)
        x = (x + d * F32(t)).astype(F32)
        prev_t = t
    return x


def vgg19_style_transfer(pack_dir: Path, content: np.ndarray, style: np.ndarray, strength: float,
                         ctx=None, steps: int = VGG_STEPS) -> np.ndarray:
    H, W = content.shape[:2]
    c = _vgg_prepare(content)
    s = _vgg_prepare(_scaled_for_inference(style, 512))
    tsess = rt.load_session(pack_dir / "vgg19_targets.onnx", opt_level="all")
    grams = rt.run(tsess, {"image": s}, ctx, output_names=[f"gram{i}" for i in range(1, 6)])
    content_t = rt.run(tsess, {"image": c}, ctx, output_names=["target_content"])[0]
    gsess = rt.load_session(pack_dir / "vgg19_gradient.onnx", opt_level="all")
    feeds = {"content_target": np.asarray(content_t, F32),
             "strength": np.array([min(max(strength / 100, 0.0), 1.0)], F32)}
    for i, gm in enumerate(grams, 1):
        feeds[f"gram{i}"] = np.asarray(gm, F32)
    shape = c.shape

    def evaluate(flat: np.ndarray):
        feeds["image"] = flat.reshape(shape)
        grad, loss = rt.run(gsess, feeds, ctx, output_names=["gradient", "total_loss"])
        return float(np.asarray(loss).reshape(-1)[0]), np.asarray(grad, F32).reshape(-1)

    x = lbfgs(c.reshape(-1), evaluate, steps=steps, ctx=ctx).reshape(shape)[0]
    img = np.clip(np.round(x.transpose(1, 2, 0) + VGG_MEAN), 0, 255) / 255
    return np.clip(_resize(img.astype(F32), W, H), 0, 1)


# ---------------------------------------------------------------------------
# AI-generated image detection (AiImageDetectorProcessor.kt)
# ---------------------------------------------------------------------------
def detect_ai_generated(meta: dict[str, Any], path: Path, rgb: np.ndarray, ctx=None) -> dict[str, Any]:
    con = meta["ai_detection"]
    S = int(con["input_size"])
    H, W = rgb.shape[:2]
    if con.get("resize_mode", "Stretch") == "Stretch":
        x = _resize(rgb, S, S)
    else:
        scale = int(con["resize_size"]) / min(W, H)
        sw, sh = max(_round(W * scale), S), max(_round(H * scale), S)
        r = _resize(rgb, sw, sh)
        x0, y0 = (sw - S) // 2, (sh - S) // 2
        x = r[y0:y0 + S, x0:x0 + S]
    t = (np.clip(x, 0, 1) - np.array(con["mean"], F32)) / np.array(con["std"], F32)
    sess = rt.load_session(path)
    logits = _flat(rt.run(sess, {sess.input_names[0]: _nchw(t)}, ctx)[0]).astype(np.float64)
    if logits.size == 0:
        raise LazyRoomError(ErrorCode.OPERATION_FAILED, "AI detector returned an empty output")
    out = con["output"]
    if out["kind"] == "sigmoid":
        p = 1 / (1 + math.exp(-logits[0]))
    else:
        idx = out["ai_class_indices"]
        if any(i >= logits.size for i in idx):
            raise LazyRoomError(ErrorCode.OPERATION_FAILED, f"AI class indices {idx} outside {logits.size} classes")
        e = np.exp(logits - logits.max())
        p = float(e[idx].sum() / max(e.sum(), 1e-12))
    p = min(max(p, 0.0), 1.0)
    verdict = "likely_ai" if p >= 0.7 else "likely_human" if p <= 0.3 else "uncertain"
    return {"ai_probability": round(p, 4), "verdict": verdict, "logits": [round(float(v), 4) for v in logits[:16]]}
