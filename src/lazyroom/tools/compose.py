"""Multi-image tools: split, stitch, collage, stacking, HDR fusion, focus stacking, panorama."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.errors import ErrorCode, LazyRoomError, invalid
from ..core.image import composite_on, ensure_rgba, fit_within, resize
from ..core.params import Bool, Choice, Color, File, Float, Int, StringList, Text
from .registry import expand_inputs, read_image, save, tool

F32 = np.float32


def _load_all(api, inputs, recursive=False, max_count=500):
    files = expand_inputs(api, inputs, recursive)
    if not files:
        raise LazyRoomError(ErrorCode.FILE_NOT_FOUND, "No input images")
    if len(files) > max_count:
        raise invalid(f"Too many images ({len(files)} > {max_count})")
    return files, [read_image(api, f)[0] for f in files]


@tool("split", "Split into tiles", category="compose", params=[
    File("input", "", file_kind="image", required=True),
    Int("rows", 2, min=1, max=100), Int("columns", 2, min=1, max=100),
    Text("output_dir", ""), Text("name_template", "{stem}_r{row}_c{col}.png"), Bool("overwrite", False),
], source={"project": "ImageToolbox", "feature": "Image Splitting"})
def split(api, input, rows, columns, output_dir, name_template, overwrite):
    """Cut an image into a rows x columns grid of tiles (e.g. Instagram grid posts)."""
    img, _ = read_image(api, input)
    if not output_dir:
        raise invalid("output_dir is required")
    od = api.ws.policy.check_write_dir(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    stem = Path(input).stem
    out = []
    for r in range(rows):
        for c in range(columns):
            y0, y1 = round(r * h / rows), round((r + 1) * h / rows)
            x0, x1 = round(c * w / columns), round((c + 1) * w / columns)
            name = name_template.format(stem=stem, row=r + 1, col=c + 1, index=r * columns + c + 1)
            out.append(save(api, img[y0:y1, x0:x1], str(od / name), overwrite=overwrite))
    return {"tiles": out}


@tool("stitch", "Stitch images side by side", category="compose", params=[
    StringList("inputs", [], min_items=2, file_kind="image"),
    Choice("direction", "horizontal", choices=["horizontal", "vertical"]),
    Choice("fit", "match", choices=["match", "none"], description="match: scale all to the same height/width"),
    Int("spacing", 0, min=0, max=2000), Color("background", "#FFFFFF"),
    Choice("align", "center", choices=["start", "center", "end"]),
    Text("output", ""), Bool("overwrite", False),
], source={"project": "ImageToolbox", "feature": "Image Stitching"})
def stitch(api, inputs, direction, fit, spacing, background, align, output, overwrite):
    """Join images into one strip (horizontal or vertical)."""
    _, imgs = _load_all(api, inputs)
    imgs = [ensure_rgba(i) for i in imgs]
    horiz = direction == "horizontal"
    if fit == "match":
        target = min(i.shape[0] if horiz else i.shape[1] for i in imgs)
        imgs = [resize(i, round(i.shape[1] * target / i.shape[0]), target) if horiz else
                resize(i, target, round(i.shape[0] * target / i.shape[1])) for i in imgs]
    cross = max(i.shape[0] if horiz else i.shape[1] for i in imgs)
    along = sum(i.shape[1] if horiz else i.shape[0] for i in imgs) + spacing * (len(imgs) - 1)
    H, W = (cross, along) if horiz else (along, cross)
    canvas = np.empty((H, W, 4), F32)
    canvas[:] = np.array(background, F32)
    pos = 0
    for i in imgs:
        ih, iw = i.shape[:2]
        free = (cross - (ih if horiz else iw))
        off = {"start": 0, "center": free // 2, "end": free}[align]
        if horiz:
            region = canvas[off:off + ih, pos:pos + iw]
            pos += iw + spacing
        else:
            region = canvas[pos:pos + ih, off:off + iw]
            pos += ih + spacing
        a = i[..., 3:4]
        region[..., :3] = i[..., :3] * a + region[..., :3] * (1 - a)
        region[..., 3:4] = a + region[..., 3:4] * (1 - a)
    if canvas[..., 3].min() >= 1:
        canvas = canvas[..., :3]
    return {"output": save(api, canvas, output, overwrite=overwrite)}


@tool("collage", "Collage", category="compose", params=[
    StringList("inputs", [], min_items=1, file_kind="image"),
    Int("width", 2400, min=64, max=20000), Int("height", 0, min=0, max=20000, description="0 = automatic"),
    Int("columns", 0, min=0, max=50, description="0 = automatic (near-square grid)"),
    Choice("layout", "grid", choices=["grid", "rows_justified"],
           description="grid: equal cells (centre-cropped); rows_justified: keep aspect, justify rows"),
    Int("spacing", 12, min=0, max=500), Int("corner_radius", 0, min=0, max=500),
    Color("background", "#FFFFFF"), Text("output", ""), Bool("overwrite", False),
], source={"project": "ImageToolbox", "feature": "Collage Maker"})
def collage(api, inputs, width, height, columns, layout, spacing, corner_radius, background, output, overwrite):
    """Arrange images in a grid or justified rows with spacing, rounded corners and background colour."""
    _, imgs = _load_all(api, inputs, max_count=200)
    imgs = [ensure_rgba(fit_within(i, 3000)[0]) for i in imgs]
    n = len(imgs)
    cols = columns or max(1, math.ceil(math.sqrt(n)))
    placements: list[tuple[np.ndarray, int, int]] = []
    if layout == "grid":
        rows = math.ceil(n / cols)
        cw = (width - spacing * (cols + 1)) / cols
        ch = cw if not height else (height - spacing * (rows + 1)) / rows
        H = height or int(round(rows * ch + spacing * (rows + 1)))
        for k, im in enumerate(imgs):
            r, c = divmod(k, cols)
            tw, th = max(1, int(cw)), max(1, int(ch))
            s = max(tw / im.shape[1], th / im.shape[0])
            sc = resize(im, math.ceil(im.shape[1] * s), math.ceil(im.shape[0] * s))
            y0, x0 = (sc.shape[0] - th) // 2, (sc.shape[1] - tw) // 2
            placements.append((sc[y0:y0 + th, x0:x0 + tw], int(spacing + c * (cw + spacing)),
                               int(spacing + r * (ch + spacing))))
    else:
        rows_list = [imgs[i:i + cols] for i in range(0, n, cols)]
        y = spacing
        for row in rows_list:
            ratios = [i.shape[1] / i.shape[0] for i in row]
            rh = (width - spacing * (len(row) + 1)) / sum(ratios)
            x = spacing
            for im, rr in zip(row, ratios):
                tw, th = max(1, int(round(rh * rr))), max(1, int(round(rh)))
                placements.append((resize(im, tw, th), int(x), int(y)))
                x += tw + spacing
            y += int(round(rh)) + spacing
        H = height or int(y)
    canvas = np.empty((H, width, 4), F32)
    canvas[:] = np.array(background, F32)
    for im, x, y in placements:
        if corner_radius:
            from ..ops.geometry import shape_mask

            im = im.copy()
            im[..., 3] *= shape_mask("rounded", im.shape[0], im.shape[1],
                                     roundness=min(1.0, 2 * corner_radius / max(1, min(im.shape[:2]))))
        h, w = im.shape[:2]
        h, w = min(h, H - y), min(w, width - x)
        if h <= 0 or w <= 0:
            continue
        region = canvas[y:y + h, x:x + w]
        a = im[:h, :w, 3:4]
        region[..., :3] = im[:h, :w, :3] * a + region[..., :3] * (1 - a)
        region[..., 3:4] = a + region[..., 3:4] * (1 - a)
    if canvas[..., 3].min() >= 1:
        canvas = canvas[..., :3]
    return {"output": save(api, canvas, output, overwrite=overwrite), "images": n}


def _align_ecc(ref_gray: np.ndarray, img: np.ndarray, mode=cv2.MOTION_HOMOGRAPHY) -> np.ndarray:
    g = (np.clip(img[..., :3], 0, 1) @ np.array([0.299, 0.587, 0.114], F32)).astype(F32)
    s = min(1.0, 1000 / max(g.shape))
    small_ref = cv2.resize(ref_gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    small = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    warp = np.eye(3, 3, dtype=np.float32) if mode == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(small_ref, small, warp, mode,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
    except cv2.error:
        return img
    if mode == cv2.MOTION_HOMOGRAPHY:
        S = np.diag([s, s, 1]).astype(np.float32)
        full = np.linalg.inv(S) @ warp @ S
        return cv2.warpPerspective(img, full, (img.shape[1], img.shape[0]),
                                   flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    warp = warp.copy()
    warp[:, 2] /= s
    return cv2.warpAffine(img, warp, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def _prepare_stack(api, inputs, align):
    files, imgs = _load_all(api, inputs, max_count=300)
    base = imgs[0][..., :3]
    h, w = base.shape[:2]
    imgs = [resize(i[..., :3], w, h) if i.shape[:2] != (h, w) else i[..., :3] for i in imgs]
    if align and len(imgs) > 1:
        ref = (np.clip(base, 0, 1) @ np.array([0.299, 0.587, 0.114], F32)).astype(F32)
        imgs = [imgs[0]] + [_align_ecc(ref, i) for i in imgs[1:]]
    return files, imgs


@tool("stack", "Image stacking", category="compose", params=[
    StringList("inputs", [], min_items=2, file_kind="image"),
    Choice("mode", "mean", choices=["mean", "median", "max", "min", "sigma_clip"],
           description="mean/median: noise reduction; max: star trails / light painting; min: remove bright objects"),
    Bool("align", True, description="Align frames (ECC homography) before stacking"),
    Text("output", ""), Bool("overwrite", False),
], source={"project": "ImageToolbox", "feature": "Image Stacking"})
def stack(api, inputs, mode, align, output, overwrite):
    """Combine aligned exposures of the same scene (noise reduction, star trails, crowd removal)."""
    files, imgs = _prepare_stack(api, inputs, align)
    arr = np.stack(imgs, 0)
    if mode == "mean":
        out = arr.mean(0)
    elif mode == "median":
        out = np.median(arr, 0)
    elif mode == "max":
        out = arr.max(0)
    elif mode == "min":
        out = arr.min(0)
    else:
        mu, sd = arr.mean(0), arr.std(0) + 1e-6
        keep = np.abs(arr - mu) <= 2 * sd
        out = (arr * keep).sum(0) / np.maximum(keep.sum(0), 1)
    return {"output": save(api, out.astype(F32), output, overwrite=overwrite), "frames": len(files)}


@tool("hdr_merge", "HDR / exposure fusion", category="compose", params=[
    StringList("inputs", [], min_items=2, file_kind="image", description="Bracketed exposures of the same scene"),
    Choice("method", "mertens", choices=["mertens", "debevec"],
           description="mertens: exposure fusion (no EXIF needed); debevec: radiance map + tone mapping (needs exposure times)"),
    Bool("align", True), Float("gamma", 1.0, min=0.2, max=3), Text("output", ""), Bool("overwrite", False),
])
def hdr_merge(api, inputs, method, align, gamma, output, overwrite):
    """Merge bracketed exposures into a single well-exposed image."""
    files, imgs = _prepare_stack(api, inputs, align)
    u8 = [(np.clip(i, 0, 1) * 255).astype(np.uint8) for i in imgs]
    if method == "mertens":
        out = cv2.createMergeMertens().process([cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in u8])
        out = cv2.cvtColor(np.clip(out, 0, 1).astype(F32), cv2.COLOR_BGR2RGB)
    else:
        times = []
        from PIL import Image

        for f in files:
            with Image.open(api.ws.policy.check_read(f)) as im:
                et = im.getexif().get_ifd(0x8769).get(0x829A)
            if not et:
                raise invalid(f"{Path(f).name} has no ExposureTime EXIF; use method='mertens'")
            times.append(float(et))
        bgr = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in u8]
        t = np.array(times, np.float32)
        resp = cv2.createCalibrateDebevec().process(bgr, t)
        hdr = cv2.createMergeDebevec().process(bgr, t, resp)
        ldr = cv2.createTonemapReinhard(gamma=2.2).process(hdr)
        out = cv2.cvtColor(np.nan_to_num(np.clip(ldr, 0, 1)).astype(F32), cv2.COLOR_BGR2RGB)
    if gamma != 1.0:
        out = np.power(np.clip(out, 0, 1), 1 / gamma)
    return {"output": save(api, out.astype(F32), output, overwrite=overwrite), "frames": len(files), "method": method}


@tool("focus_stack", "Focus stacking", category="compose", params=[
    StringList("inputs", [], min_items=2, file_kind="image", description="Shots focused at different distances"),
    Bool("align", True), Int("smoothing", 5, min=1, max=50, description="Blend smoothing radius (px)"),
    Text("output", ""), Bool("overwrite", False),
])
def focus_stack(api, inputs, align, smoothing, output, overwrite):
    """Combine the sharpest regions of each frame (Laplacian energy, soft-weighted blend)."""
    files, imgs = _prepare_stack(api, inputs, align)
    weights = []
    for im in imgs:
        g = (np.clip(im, 0, 1) @ np.array([0.299, 0.587, 0.114], F32)).astype(F32)
        lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F, ksize=3))
        weights.append(cv2.GaussianBlur(lap, (0, 0), smoothing))
    W = np.stack(weights, 0) ** 4 + 1e-12
    W /= W.sum(0, keepdims=True)
    out = (np.stack(imgs, 0) * W[..., None]).sum(0)
    return {"output": save(api, out.astype(F32), output, overwrite=overwrite), "frames": len(files)}


@tool("panorama", "Panorama stitching", category="compose", params=[
    StringList("inputs", [], min_items=2, file_kind="image", description="Overlapping photos"),
    Choice("mode", "panorama", choices=["panorama", "scans"], description="scans: flat documents / affine"),
    Bool("crop", True, description="Crop to the largest rectangle without empty areas"),
    Text("output", ""), Bool("overwrite", False),
])
def panorama(api, inputs, mode, crop, output, overwrite):
    """Stitch overlapping photos into a panorama (OpenCV Stitcher: feature matching, warping, seam blending)."""
    files, imgs = _load_all(api, inputs, max_count=60)
    bgr = [cv2.cvtColor((np.clip(fit_within(i[..., :3], 3000)[0], 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
           for i in imgs]
    st = cv2.Stitcher.create(cv2.Stitcher_PANORAMA if mode == "panorama" else cv2.Stitcher_SCANS)
    status, pano = st.stitch(bgr)
    if status != cv2.Stitcher_OK:
        names = {1: "not enough overlapping features", 2: "homography estimation failed",
                 3: "camera parameter adjustment failed"}
        raise LazyRoomError(ErrorCode.OPERATION_FAILED, f"Panorama stitching failed: {names.get(status, status)}",
                              hint="Use photos with 20-50% overlap, same exposure, taken from one position.")
    out = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB).astype(F32) / 255
    if crop:
        mask = (pano.max(2) > 0).astype(np.uint8)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        rows = np.where(mask.mean(1) > 0.995)[0]
        cols = np.where(mask[rows[0]:rows[-1] + 1].min(0) > 0)[0] if len(rows) else []
        if len(rows) and len(cols):
            out = out[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return {"output": save(api, out, output, overwrite=overwrite), "frames": len(files)}
