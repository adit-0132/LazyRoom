"""Multi-frame fusion — port of ImageToolbox ``feature/multi-frame-fusion`` (Apache-2.0, T8RIN).

Sources: ``MultiFrameFusionEngine.kt`` (OpenCV pipeline), ``FusionMode.kt``, ``FusionParams.kt``
(defaults, ranges, ``normalized()``), ``AndroidMultiFrameFusionProcessor.kt`` (distinct inputs,
2..12 frames, full resolution for the saved result).

Pipeline (same order, constants and OpenCV calls as the engine):

1. Every frame becomes 8-bit RGB (``Utils.bitmapToMat`` gives premultiplied RGBA, then RGBA->RGB, so
   transparent pixels become black); frames 2..n are resized to frame 1 with ``INTER_AREA``.
2. Optional alignment to frame 1: ORB (1800 features) on grey, brute-force Hamming kNN (k=2),
   Lowe ratio 0.76, >= 12 matches, RANSAC homography (3 px), sanity check (warped edge lengths within
   0.55..1.8 of the frame), ``warpPerspective`` (bilinear, black border); a warp is kept only if it
   covers >= 45 % of the frame.  The intersection of the warped coverage masks is the common area.
3. Optional crop to the largest axis-aligned rectangle inside the common area (histogram/stack
   algorithm), only if it is >= 64 px per side and >= 25 % of the frame.
4. Fusion mode: Mertens exposure fusion, Laplacian focus stack with exposure normalisation, per-pixel
   median, linear-light (gamma 2.2) long exposure, light trails, motion trails.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import cv2
import numpy as np

from ..core import fast as _fast  # noqa: F401  (sets NUMBA_CACHE_DIR before numba is imported)
from ..core.errors import ErrorCode, LazyRoomError
from ..core.params import Bool, Choice, Float, Int, StringList, Text
from .registry import expand_inputs, read_image, save, tool

import numba as nb  # noqa: E402

F32 = np.float32

# FusionMode.entries (snake_case ids; titles from strings.xml)
FUSION_MODES = ["exposure", "focus", "median", "long_exposure", "light_trails", "motion_trails"]
MODE_TITLES = {
    "exposure": "Exposure Fusion", "focus": "Focus Stacking", "median": "Median Stack",
    "long_exposure": "Long Exposure", "light_trails": "Light Trails", "motion_trails": "Motion Trails",
}

# FusionParams.companion
MIN_IMAGES, MAX_IMAGES = 2, 12
MIN_WEIGHT, MAX_WEIGHT = 0.0, 2.0
MIN_FOCUS_RADIUS, MAX_FOCUS_RADIUS = 1, 21
MIN_FOCUS_STRENGTH, MAX_FOCUS_STRENGTH = 1.0, 6.0
MIN_LIGHT_TRAIL_THRESHOLD, MAX_LIGHT_TRAIL_THRESHOLD = 0.0, 0.5
MIN_TRAIL_STRENGTH, MAX_TRAIL_STRENGTH = 0.0, 1.0
MIN_TRAIL_PERSISTENCE, MAX_TRAIL_PERSISTENCE = 0.1, 1.0

# MultiFrameFusionEngine.companion
MAX_FEATURES = 1800
MIN_FEATURE_MATCHES = 12
LOWE_RATIO = 0.76
RANSAC_REPROJECTION_THRESHOLD = 3.0
MIN_VALID_ALIGNMENT_FRACTION = 0.45
MIN_TRANSFORM_SCALE = 0.55
MAX_TRANSFORM_SCALE = 1.8
MIN_CROP_SIDE = 64
MIN_CROP_AREA_FRACTION = 0.25
FOCUS_WEIGHT_EPSILON = 0.5
MIN_BRIGHTNESS = 1.0
MIN_EXPOSURE_SCALE = 0.25
MAX_EXPOSURE_SCALE = 4.0
LINEAR_GAMMA = 2.2


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


@dataclass
class FusionParams:
    mode: str = "exposure"
    align_images: bool = True
    crop_to_overlap: bool = True
    contrast_weight: float = 1.0
    saturation_weight: float = 1.0
    exposure_weight: float = 1.0
    focus_radius: int = 7
    focus_strength: float = 2.0
    light_trail_threshold: float = 0.08
    trail_strength: float = 1.0
    trail_persistence: float = 0.7

    def normalized(self) -> "FusionParams":
        """``FusionParams.normalized()``: clamp everything; focus radius forced odd (even -> +1)."""
        r = int(_clamp(int(self.focus_radius), MIN_FOCUS_RADIUS, MAX_FOCUS_RADIUS))
        if r % 2 == 0:
            r += 1
        r = min(r, MAX_FOCUS_RADIUS)
        return replace(
            self,
            contrast_weight=_clamp(float(self.contrast_weight), MIN_WEIGHT, MAX_WEIGHT),
            saturation_weight=_clamp(float(self.saturation_weight), MIN_WEIGHT, MAX_WEIGHT),
            exposure_weight=_clamp(float(self.exposure_weight), MIN_WEIGHT, MAX_WEIGHT),
            focus_radius=r,
            focus_strength=_clamp(float(self.focus_strength), MIN_FOCUS_STRENGTH, MAX_FOCUS_STRENGTH),
            light_trail_threshold=_clamp(float(self.light_trail_threshold), MIN_LIGHT_TRAIL_THRESHOLD,
                                         MAX_LIGHT_TRAIL_THRESHOLD),
            trail_strength=_clamp(float(self.trail_strength), MIN_TRAIL_STRENGTH, MAX_TRAIL_STRENGTH),
            trail_persistence=_clamp(float(self.trail_persistence), MIN_TRAIL_PERSISTENCE, MAX_TRAIL_PERSISTENCE),
        )


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------

def to_rgb8(img: np.ndarray) -> np.ndarray:
    """Float 0..1 RGB(A) -> uint8 RGB as ``bitmapToMat`` + ``COLOR_RGBA2RGB`` (premultiplied colour)."""
    x = np.clip(img, 0, 1)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, 2)
    if x.shape[2] == 4:
        a8 = np.rint(x[..., 3:4] * 255.0)
        c8 = np.rint(x[..., :3] * 255.0)
        return np.ascontiguousarray(np.floor((c8 * a8 + 127.0) / 255.0).astype(np.uint8))
    return np.ascontiguousarray(np.rint(x[..., :3] * 255.0).astype(np.uint8))


def normalize_sizes(images: list[np.ndarray]) -> list[np.ndarray]:
    h, w = images[0].shape[:2]
    return [images[0]] + [im if im.shape[:2] == (h, w) else cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
                          for im in images[1:]]


# ---------------------------------------------------------------------------
# Alignment (ORB + homography) and overlap crop
# ---------------------------------------------------------------------------

def _features(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(MAX_FEATURES)
    kps, desc = orb.detectAndCompute(gray, None)
    if not kps or desc is None or len(desc) == 0:
        return None
    return kps, desc


def _is_sane_homography(H: np.ndarray, w: int, h: int) -> bool:
    corners = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], np.float64)
    c = cv2.perspectiveTransform(corners, H).reshape(4, 2)
    if not np.isfinite(c).all():
        return False
    d = lambda i, j: float(np.hypot(*(c[i] - c[j])))  # noqa: E731
    width = max(d(0, 1), d(2, 3))
    height = max(d(0, 3), d(1, 2))
    return (w * MIN_TRANSFORM_SCALE <= width <= w * MAX_TRANSFORM_SCALE
            and h * MIN_TRANSFORM_SCALE <= height <= h * MAX_TRANSFORM_SCALE)


def find_homography(source: np.ndarray, reference) -> tuple[np.ndarray | None, dict[str, Any]]:
    info: dict[str, Any] = {}
    feats = _features(source)
    if feats is None:
        info["reason"] = "no features in frame"
        return None, info
    kps, desc = feats
    rkps, rdesc = reference
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(desc, rdesc, k=2)
    src_pts, ref_pts = [], []
    for pair in matches:
        if len(pair) >= 2 and pair[0].distance < LOWE_RATIO * pair[1].distance:
            src_pts.append(kps[pair[0].queryIdx].pt)
            ref_pts.append(rkps[pair[0].trainIdx].pt)
    info["matches"] = len(src_pts)
    if len(src_pts) < MIN_FEATURE_MATCHES:
        info["reason"] = f"only {len(src_pts)} good matches (< {MIN_FEATURE_MATCHES})"
        return None, info
    H, inl = cv2.findHomography(np.float32(src_pts).reshape(-1, 1, 2), np.float32(ref_pts).reshape(-1, 1, 2),
                                cv2.RANSAC, RANSAC_REPROJECTION_THRESHOLD)
    if H is None or H.size == 0:
        info["reason"] = "homography estimation failed"
        return None, info
    info["inliers"] = int(inl.sum()) if inl is not None else None
    if not _is_sane_homography(H, source.shape[1], source.shape[0]):
        info["reason"] = "homography rejected by sanity check"
        return None, info
    return H, info


@nb.njit(cache=True)
def _largest_rect(mask: np.ndarray):  # pragma: no cover - compiled
    """``largestValidRectangle``: maximal rectangle of non-zero cells (row histograms + stack)."""
    height, width = mask.shape
    hist = np.zeros(width, np.int64)
    stack = np.zeros(width + 1, np.int64)
    best_area = 0
    bx = by = bw = bh = 0
    for y in range(height):
        for x in range(width):
            if mask[y, x] != 0:
                hist[x] += 1
            else:
                hist[x] = 0
        size = 0
        for x in range(width + 1):
            cur = 0 if x == width else hist[x]
            while size > 0 and hist[stack[size - 1]] > cur:
                size -= 1
                bar = stack[size]
                rh = hist[bar]
                left = 0 if size == 0 else stack[size - 1] + 1
                rw = x - left
                area = rw * rh
                if area > best_area:
                    best_area = area
                    bx, by, bw, bh = left, y - rh + 1, rw, rh
            stack[size] = x
            size += 1
    return best_area, bx, by, bw, bh


def largest_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask.size == 0:
        return None
    area, x, y, w, h = _largest_rect(np.ascontiguousarray(mask, dtype=np.uint8))
    return None if area == 0 else (int(x), int(y), int(w), int(h))


def align_to_first(images: list[np.ndarray], crop_to_overlap: bool):
    ref = images[0]
    h, w = ref.shape[:2]
    common = np.full((h, w), 255, np.uint8)
    ref_feats = _features(ref)
    frames: list[dict[str, Any]] = [{"index": 0, "reference": True}]
    out = [ref]
    for i, src in enumerate(images[1:], start=1):
        rec: dict[str, Any] = {"index": i, "aligned": False}
        H = None
        if ref_feats is None:
            rec["reason"] = "no features in reference frame"
        else:
            H, info = find_homography(src, ref_feats)
            rec.update(info)
        if H is not None:
            warped = cv2.warpPerspective(src, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0)
            wmask = cv2.warpPerspective(np.full(src.shape[:2], 255, np.uint8), H, (w, h), flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            valid = float(np.count_nonzero(wmask)) / max(1, wmask.size)
            rec["valid_fraction"] = round(valid, 4)
            rec["homography"] = [[round(float(v), 6) for v in row] for row in H]
            rec["translation"] = [round(float(H[0, 2]), 3), round(float(H[1, 2]), 3)]
            if valid >= MIN_VALID_ALIGNMENT_FRACTION:
                src = warped
                np.bitwise_and(common, wmask, out=common)
                rec["aligned"] = True
            else:
                rec["reason"] = f"warp covers {valid:.0%} of the frame (< 45%)"
        out.append(src)
        frames.append(rec)
    crop = None
    if crop_to_overlap:
        rect = largest_valid_rectangle(common)
        if rect is not None:
            x, y, rw, rh = rect
            if rw >= MIN_CROP_SIDE and rh >= MIN_CROP_SIDE and rw * rh >= w * h * MIN_CROP_AREA_FRACTION:
                out = [np.ascontiguousarray(im[y:y + rh, x:x + rw]) for im in out]
                crop = {"x": x, "y": y, "width": rw, "height": rh}
    return out, frames, crop


# ---------------------------------------------------------------------------
# Fusion modes
# ---------------------------------------------------------------------------

def _to_u8(x: np.ndarray) -> np.ndarray:
    """``convertTo(CV_8UC3, 255.0)``: saturate_cast (round half to even, clamp)."""
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


# convertTo(CV_32FC3, 1/255) + Core.pow(2.2) evaluated once per 8-bit level
_LINEAR_LUT = np.power(np.arange(256, dtype=F32) * F32(1.0 / 255.0), F32(LINEAR_GAMMA)).astype(F32)


def _linear(img8: np.ndarray) -> np.ndarray:
    return _LINEAR_LUT[img8]


def _to_rgb8_from_linear(x: np.ndarray) -> np.ndarray:  # Mat.toRgb8 (Core.pow uses |x| for fractional powers)
    return _to_u8(cv2.pow(np.ascontiguousarray(x, dtype=F32), 1.0 / LINEAR_GAMMA))


def _intensity(x: np.ndarray) -> np.ndarray:  # max(R, G, B)
    return np.maximum(np.maximum(x[..., 0], x[..., 1]), x[..., 2])


def exposure_fusion(images, p: FusionParams) -> np.ndarray:
    merger = cv2.createMergeMertens(p.contrast_weight, p.saturation_weight, p.exposure_weight)
    return _to_u8(merger.process(list(images)))


def focus_stack(images, p: FusionParams) -> np.ndarray:
    h, w = images[0].shape[:2]
    acc = np.zeros((h, w, 3), F32)
    wsum = np.zeros((h, w), F32)
    bright = [max(float(np.mean(im.reshape(-1, 3), axis=0).mean()), MIN_BRIGHTNESS) for im in images]
    target = sorted(bright)[len(bright) // 2]
    k = (p.focus_radius, p.focus_radius)
    for im, b in zip(images, bright):
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        sharp = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        weight = cv2.GaussianBlur(sharp, k, 0)
        weight = np.power(weight + F32(FOCUS_WEIGHT_EPSILON), F32(p.focus_strength))
        scale = _clamp(target / b, MIN_EXPOSURE_SCALE, MAX_EXPOSURE_SCALE)
        acc += im.astype(F32) * F32(scale / 255.0) * weight[..., None]
        wsum += weight
    return _to_u8(acc / wsum[..., None])


def median_stack(images) -> np.ndarray:
    n = len(images)
    h = images[0].shape[0]
    out = np.empty_like(images[0])
    rows = max(1, (64 << 20) // max(1, images[0][0].size * n))  # ~64 MB per chunk
    for y0 in range(0, h, rows):
        st = np.stack([im[y0:y0 + rows] for im in images], 0)
        if n % 2 == 1:
            out[y0:y0 + rows] = np.partition(st, n // 2, axis=0)[n // 2]
        else:
            part = np.partition(st, [n // 2 - 1, n // 2], axis=0)
            s = part[n // 2 - 1].astype(F32) * F32(0.5) + part[n // 2].astype(F32) * F32(0.5)
            out[y0:y0 + rows] = np.clip(np.rint(s), 0, 255).astype(np.uint8)  # addWeighted -> saturate_cast
    return out


def _linear_average(images) -> np.ndarray:
    acc = np.zeros(images[0].shape, F32)
    for im in images:
        acc += _linear(im)
    return acc * F32(1.0 / len(images))


def long_exposure(images) -> np.ndarray:
    return _to_rgb8_from_linear(_linear_average(images))


def light_trails(images, p: FusionParams) -> np.ndarray:
    base = _linear_average(images)
    brightest = _linear(images[0])
    b_int = _intensity(brightest)
    for im in images[1:]:
        cand = _linear(im)
        c_int = _intensity(cand)
        m = c_int > b_int  # Core.compare(CMP_GT) + copyTo(mask)
        np.copyto(brightest, cand, where=m[..., None])
        np.copyto(b_int, c_int, where=m)
    t = p.light_trail_threshold
    mask = np.maximum(b_int - _intensity(base) - F32(t), 0) * F32(1.0 / (1.0 - t))
    mask = np.sqrt(np.minimum(mask, 1)) * F32(p.trail_strength)
    return _to_rgb8_from_linear(base + (brightest - base) * mask[..., None])


def motion_trails(images, p: FusionParams) -> np.ndarray:
    acc = np.zeros(images[0].shape, F32)
    wsum = 0.0
    last = len(images) - 1
    for i, im in enumerate(images):
        wgt = float(p.trail_persistence) ** (last - i)
        acc += _linear(im) * F32(wgt)
        wsum += wgt
    avg = acc * F32(1.0 / wsum)
    latest = _linear(images[-1])
    s = F32(p.trail_strength)
    return _to_rgb8_from_linear(latest * (F32(1.0) - s) + avg * s)


def fuse(frames: list[np.ndarray], params: FusionParams) -> tuple[np.ndarray, dict[str, Any]]:
    """Fuse float 0..1 RGB(A) frames; returns (uint8 RGB result, info)."""
    if len(frames) < MIN_IMAGES:
        raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"Multi-frame fusion needs at least {MIN_IMAGES} images")
    p = params.normalized()
    images = normalize_sizes([to_rgb8(f) for f in frames])
    info: dict[str, Any] = {"alignment": None, "crop": None}
    if p.align_images:
        images, align_info, crop = align_to_first(images, p.crop_to_overlap)
        info["alignment"] = align_info
        info["crop"] = crop
    mode = p.mode
    if mode == "exposure":
        out = exposure_fusion(images, p)
    elif mode == "focus":
        out = focus_stack(images, p)
    elif mode == "median":
        out = median_stack(images)
    elif mode == "long_exposure":
        out = long_exposure(images)
    elif mode == "light_trails":
        out = light_trails(images, p)
    elif mode == "motion_trails":
        out = motion_trails(images, p)
    else:
        raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"Unknown fusion mode {mode!r}")
    info["params"] = asdict(p)
    return out, info


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool("multi_frame_fusion", "Multi-frame fusion", category="compose", params=[
    StringList("inputs", [], min_items=2, file_kind="image",
               description="2..12 frames of the same scene (first = alignment reference; for motion_trails "
                           "put the newest frame last). Duplicates are dropped and frames after the 12th ignored, "
                           "like ImageToolbox."),
    Choice("mode", "exposure", choices=FUSION_MODES, labels=MODE_TITLES,
           description="exposure: Mertens fusion of brackets; focus: sharpest regions (focus stacking); "
                       "median: remove transient objects/noise; long_exposure: linear-light average; "
                       "light_trails: keep bright changes; motion_trails: fading history of moving subjects"),
    Bool("align_images", True, description="Align frames to the first one (ORB features + RANSAC homography)"),
    Bool("crop_to_overlap", True, description="Crop to the area covered by every aligned frame"),
    Float("contrast_weight", 1.0, min=MIN_WEIGHT, max=MAX_WEIGHT, step=0.1, description="exposure mode"),
    Float("saturation_weight", 1.0, min=MIN_WEIGHT, max=MAX_WEIGHT, step=0.1, description="exposure mode"),
    Float("exposure_weight", 1.0, min=MIN_WEIGHT, max=MAX_WEIGHT, step=0.1,
          description="exposure mode: well-exposedness weight"),
    Int("focus_radius", 7, min=MIN_FOCUS_RADIUS, max=MAX_FOCUS_RADIUS, unit="px",
        description="focus mode: Gaussian window of the sharpness map (even values become the next odd one)"),
    Float("focus_strength", 2.0, min=MIN_FOCUS_STRENGTH, max=MAX_FOCUS_STRENGTH, step=0.5,
          description="focus mode: exponent applied to the sharpness weights"),
    Float("light_trail_threshold", 0.08, min=MIN_LIGHT_TRAIL_THRESHOLD, max=MAX_LIGHT_TRAIL_THRESHOLD, step=0.05,
          description="light_trails: minimum brightening (linear light) that counts as a trail"),
    Float("trail_strength", 1.0, min=MIN_TRAIL_STRENGTH, max=MAX_TRAIL_STRENGTH, step=0.1,
          description="light_trails / motion_trails strength"),
    Float("trail_persistence", 0.7, min=MIN_TRAIL_PERSISTENCE, max=MAX_TRAIL_PERSISTENCE, step=0.1,
          description="motion_trails: weight decay per older frame"),
    Text("output", "", description="Output image path"), Bool("overwrite", False),
], source={"project": "ImageToolbox", "feature": "Multi-frame fusion (MultiFrameFusionEngine)"})
def multi_frame_fusion(api, inputs, mode, align_images, crop_to_overlap, contrast_weight, saturation_weight,
                       exposure_weight, focus_radius, focus_strength, light_trail_threshold, trail_strength,
                       trail_persistence, output, overwrite):
    """Fuse 2-12 frames of one scene: exposure fusion, focus stacking, median stack, long exposure,
    light trails or motion trails, with optional feature-based alignment and overlap crop."""
    files = list(dict.fromkeys(expand_inputs(api, inputs, False)))
    ignored = files[MAX_IMAGES:]
    files = files[:MAX_IMAGES]
    if len(files) < MIN_IMAGES:
        raise LazyRoomError(ErrorCode.INVALID_PARAMS,
                              f"Multi-frame fusion needs at least {MIN_IMAGES} distinct images, got {len(files)}")
    if not output:
        raise LazyRoomError(ErrorCode.INVALID_PARAMS, "An output path is required")
    frames = [read_image(api, f)[0] for f in files]
    p = FusionParams(mode=mode, align_images=align_images, crop_to_overlap=crop_to_overlap,
                     contrast_weight=contrast_weight, saturation_weight=saturation_weight,
                     exposure_weight=exposure_weight, focus_radius=focus_radius, focus_strength=focus_strength,
                     light_trail_threshold=light_trail_threshold, trail_strength=trail_strength,
                     trail_persistence=trail_persistence)
    out8, info = fuse(frames, p)
    res: dict[str, Any] = {"output": save(api, out8.astype(F32) / 255.0, output, overwrite=overwrite),
                           "frames": len(files), "inputs": files, "mode": mode, **info}
    if ignored:
        res["ignored_inputs"] = ignored
    if info["alignment"] is not None:
        res["aligned_frames"] = sum(1 for f in info["alignment"][1:] if f.get("aligned"))
    return res
