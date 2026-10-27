"""PaddleOCR (PP-OCRv5 / v6) text recognition — port of ImageToolbox
``lib/neural-tools/.../ocr/PaddleOCR.kt`` (DB text detection, angle classifier,
CTC recogniser).  Model packs are the zips published by ImageToolbox (det/rec/cls
ONNX + dictionary), downloaded on request via :mod:`lazyroom.ai.models`.

Upstream quirks kept on purpose: detection normalises channels in B, G, R order
with the ImageNet mean/std given in R, G, B order; classifier/recogniser inputs
are BGR scaled to [-1, 1]; boxes use ImageToolbox's own convex hull, minimum-area
rectangle and polygon offset ("unclip") code rather than OpenCV/pyclipper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..core.errors import ErrorCode, LazyRoomError
from . import runtime as rt

F32 = np.float32
_DET_MEAN = np.array([0.485, 0.456, 0.406], F32)
_DET_STD = np.array([0.229, 0.224, 0.225], F32)


@dataclass
class PackFiles:
    det: Path
    rec: Path
    cls: Path
    dict: Path


def resolve_pack(folder: Path) -> PackFiles:
    files = sorted(p for p in folder.rglob("*") if p.is_file())

    def pick(exact: str, key: str, ext: str) -> Path | None:
        for p in files:
            if p.name == exact or (key in p.name.lower() and p.suffix.lower() == ext):
                return p
        return None

    det, rec, cls = pick("det.onnx", "det", ".onnx"), pick("rec.onnx", "rec", ".onnx"), pick("cls.onnx", "cls", ".onnx")
    dct = pick("ppocrv5_dict.txt", "dict", ".txt")
    if not (det and rec and cls and dct):
        raise LazyRoomError(ErrorCode.BACKEND_UNAVAILABLE, f"Incomplete PaddleOCR model pack in {folder}")
    return PackFiles(det, rec, cls, dct)


def _scale(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)


# --- geometry helpers (TextDetector) ------------------------------------------
def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    pts = [tuple(p) for p in points.tolist()]
    if len(pts) < 3:
        return pts
    pts.sort(key=lambda p: (p[0], p[1]))
    lower: list = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def min_area_rect(points, convex: bool) -> list[tuple[float, float]]:
    hull = points if convex else convex_hull(np.asarray(points, F32))
    if len(hull) < 3:
        return []
    best: list = []
    min_area = math.inf
    for i in range(len(hull)):
        p1, p2 = hull[i], hull[(i + 1) % len(hull)]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        ln = math.hypot(dx, dy)
        if ln < 1e-6:
            continue
        ex, ey = dx / ln, dy / ln
        nx, ny = -ey, ex
        rel = [(p[0] - p1[0], p[1] - p1[1]) for p in hull]
        proj = [rx * ex + ry * ey for rx, ry in rel]
        orth = [rx * nx + ry * ny for rx, ry in rel]
        a0, a1, b0, b1 = min(proj), max(proj), min(orth), max(orth)
        area = (a1 - a0) * (b1 - b0)
        if area < min_area and a1 - a0 > 1e-3 and b1 - b0 > 1e-3:
            min_area = area
            best = [(p1[0] + ex * a0 + nx * b0, p1[1] + ey * a0 + ny * b0),
                    (p1[0] + ex * a1 + nx * b0, p1[1] + ey * a1 + ny * b0),
                    (p1[0] + ex * a1 + nx * b1, p1[1] + ey * a1 + ny * b1),
                    (p1[0] + ex * a0 + nx * b1, p1[1] + ey * a0 + ny * b1)]
    if best:
        return best
    xs, ys = [p[0] for p in hull], [p[1] for p in hull]
    return [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]


def _inside_mask(quad, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    pos = np.zeros(xs.shape, bool)
    neg = np.zeros(xs.shape, bool)
    for i in range(len(quad)):
        p1, p2 = quad[i], quad[(i + 1) % len(quad)]
        c = (p2[0] - p1[0]) * (ys - p1[1]) - (p2[1] - p1[1]) * (xs - p1[0])
        pos |= c > 0
        neg |= c < 0
    return ~(pos & neg)


def box_score(prob: np.ndarray, quad) -> float:
    h, w = prob.shape
    x0 = min(max(math.floor(min(p[0] for p in quad)), 0), w - 1)
    x1 = min(max(math.ceil(max(p[0] for p in quad)), 0), w - 1)
    y0 = min(max(math.floor(min(p[1] for p in quad)), 0), h - 1)
    y1 = min(max(math.ceil(max(p[1] for p in quad)), 0), h - 1)
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(F32)
    m = _inside_mask(quad, xs + 0.5, ys + 0.5)
    return float(prob[y0:y1 + 1, x0:x1 + 1][m].mean()) if m.any() else 0.0


def _signed_area(pts) -> float:
    return sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
               for i in range(len(pts))) / 2


def _perimeter(pts) -> float:
    return sum(math.dist(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))


def _norm(v):
    ln = math.hypot(*v)
    return None if ln < 1e-6 else (v[0] / ln, v[1] / ln)


def unclip(box, ratio: float = 1.5):
    per = _perimeter(box)
    if per <= 1e-6:
        return []
    offset = abs(_signed_area(box)) * ratio / per
    ccw = _signed_area(box) > 0
    out = []
    n = len(box)
    for i in range(n):
        prev, cur, nxt = box[(i - 1) % n], box[i], box[(i + 1) % n]
        d1 = _norm((cur[0] - prev[0], cur[1] - prev[1]))
        d2 = _norm((nxt[0] - cur[0], nxt[1] - cur[1]))
        if d1 is None or d2 is None:
            continue
        n1 = (d1[1], -d1[0]) if ccw else (-d1[1], d1[0])
        n2 = (d2[1], -d2[0]) if ccw else (-d2[1], d2[0])
        a = (cur[0] + n1[0] * offset, cur[1] + n1[1] * offset)
        b = (cur[0] + n2[0] * offset, cur[1] + n2[1] * offset)
        cr = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cr) < 1e-6:
            out.append(cur)
            continue
        t = ((b[0] - a[0]) * d2[1] - (b[1] - a[1]) * d2[0]) / cr
        out.append((a[0] + d1[0] * t, a[1] + d1[1] * t))
    return out if len(out) >= 3 else []


def order_clockwise(pts):
    if len(pts) != 4:
        return pts
    cx, cy = sum(p[0] for p in pts) / 4, sum(p[1] for p in pts) / 4
    s = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    tl = min(range(4), key=lambda i: s[i][0] + s[i][1])
    return [s[(tl + i) % 4] for i in range(4)]


def sort_boxes(boxes):
    by_top = sorted(boxes, key=lambda b: min(p[1] for p in b))
    out = []
    i = 0
    while i < len(by_top):
        ref = min(p[1] for p in by_top[i])
        j = i
        group = []
        while j < len(by_top) and abs(min(p[1] for p in by_top[j]) - ref) <= 10:
            group.append(by_top[j])
            j += 1
        group.sort(key=lambda b: min(p[0] for p in b))
        out.extend(group)
        i = j
    return out


# --- pipeline -------------------------------------------------------------------
def detect(sess: rt.Session, rgb: np.ndarray, ctx=None) -> list[list[tuple[float, float]]]:
    H, W = rgb.shape[:2]
    ratio = 960 / max(W, H) if max(W, H) > 960 else 1.0
    rw = max(((int(math.floor(W * ratio + 0.5)) + 31) // 32) * 32, 32)
    rh = max(((int(math.floor(H * ratio + 0.5)) + 31) // 32) * 32, 32)
    x = _scale(rgb, rw, rh)
    bgr = x[..., ::-1]
    t = ((bgr - _DET_MEAN) / _DET_STD).transpose(2, 0, 1)[None].astype(F32)
    name = "x" if "x" in sess.input_names else sess.input_names[0]
    prob = np.asarray(rt.run(sess, {name: t}, ctx)[0], F32).reshape(-1)[: rh * rw].reshape(rh, rw)
    binary = (prob > 0.3).astype(np.uint8)
    n, labels = cv2.connectedComponents(binary, connectivity=8)
    sizes = np.bincount(labels.ravel())
    comps = sorted(range(1, n), key=lambda k: -sizes[k])[:1000]
    sx, sy = W / rw, H / rh
    boxes = []
    ys_all, xs_all = np.nonzero(labels)
    lab_all = labels[ys_all, xs_all]
    order = np.argsort(lab_all, kind="stable")
    ys_all, xs_all, lab_all = ys_all[order], xs_all[order], lab_all[order]
    starts = np.searchsorted(lab_all, np.arange(n + 1))
    for k in comps:
        if ctx is not None and len(boxes) % 50 == 0:
            ctx.check()
        if sizes[k] < 4:
            continue
        pts = np.stack([xs_all[starts[k]:starts[k + 1]], ys_all[starts[k]:starts[k + 1]]], 1).astype(F32)
        hull = convex_hull(pts)
        if len(hull) < 3:
            continue
        rect = min_area_rect(hull, True)
        if not rect or box_score(prob, rect) < 0.6:
            continue
        expanded = unclip(rect, 1.5)
        er = min_area_rect(expanded, False) if expanded else []
        if not er or min(math.dist(er[i], er[(i + 1) % len(er)]) for i in range(len(er))) < 3:
            continue
        clipped = [(min(max(p[0], 0), rw - 1) * sx, min(max(p[1], 0), rh - 1) * sy) for p in er]
        boxes.append(order_clockwise(clipped))
    return sort_boxes(boxes)


def crop_text(rgb: np.ndarray, pts) -> np.ndarray:
    w = max(int(max(math.dist(pts[0], pts[1]), math.dist(pts[2], pts[3]))), 1)
    h = max(int(max(math.dist(pts[0], pts[3]), math.dist(pts[1], pts[2]))), 1)
    m = cv2.getPerspectiveTransform(np.array(pts, F32), np.array([(0, 0), (w, 0), (w, h), (0, h)], F32))
    crop = cv2.warpPerspective(rgb, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if h / w >= 1.5:
        crop = np.ascontiguousarray(np.rot90(crop, k=-1))  # postRotate(90): clockwise
    return crop


def _line_tensor(img: np.ndarray, width: int, target_w: int) -> np.ndarray:
    r = _scale(img, width, 48)
    out = np.zeros((3, 48, target_w), F32)
    out[:, :, :width] = ((r[..., ::-1] - 0.5) / 0.5).transpose(2, 0, 1)
    return out


def classify(sess: rt.Session, crops: list[np.ndarray], ctx=None) -> list[bool]:
    flags: list[bool] = []
    for i in range(0, len(crops), 6):
        batch = crops[i:i + 6]
        t = np.stack([_line_tensor(c, min(max(math.ceil(48 * c.shape[1] / c.shape[0]), 1), 192), 192)
                      for c in batch])
        out = np.asarray(rt.run(sess, {sess.input_names[0]: t}, ctx)[0], F32).reshape(len(batch), -1)
        flags += [bool(o[1] > o[0] and o[1] > 0.9) for o in out]
    return flags


def recognize(sess: rt.Session, crops: list[np.ndarray], chars: list[str], ctx=None) -> list[tuple[str, float]]:
    res: list[tuple[str, float]] = [("", 0.0)] * len(crops)
    order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / crops[i].shape[0])
    for b in range(0, len(order), 6):
        idx = order[b:b + 6]
        max_wh = max([320 / 48] + [crops[i].shape[1] / crops[i].shape[0] for i in idx])
        tw = max(math.ceil(48 * max_wh), 1)
        t = np.stack([_line_tensor(crops[i], min(max(math.ceil(48 * crops[i].shape[1] / crops[i].shape[0]), 1), tw),
                                   tw) for i in idx])
        out = np.asarray(rt.run(sess, {sess.input_names[0]: t}, ctx)[0], F32)
        for j, i in enumerate(idx):
            seq = out[j]
            ids, probs = seq.argmax(axis=1), seq.max(axis=1)
            text, ps = [], []
            t_ = 0
            while t_ < len(ids):
                c = int(ids[t_])
                if c == 0:
                    t_ += 1
                    continue
                end = t_ + 1
                while end < len(ids) and ids[end] == c:
                    end += 1
                if c < len(chars):
                    text.append(chars[c])
                    ps.append(float(probs[t_:end].mean()))
                t_ = end
            res[i] = ("".join(text), float(np.mean(ps)) if ps else 0.0)
    return res


def run_ocr(pack_dir: Path, rgb: np.ndarray, ctx=None) -> dict:
    files = resolve_pack(pack_dir)
    det = rt.load_session(files.det, opt_level="basic")
    rec = rt.load_session(files.rec, opt_level="basic")
    cls = rt.load_session(files.cls, opt_level="basic")
    lines = files.dict.read_text(encoding="utf-8").splitlines()
    chars = ["blank"] + lines + [" "]
    x = np.clip(rgb[..., :3], 0, 1).astype(F32)
    boxes = detect(det, x, ctx)
    if not boxes:
        return {"text": "", "lines": [], "accuracy": 0}
    crops = [crop_text(x, b) for b in boxes]
    classified = [False] * len(crops)

    def classify_idx(ids: list[int]) -> None:
        if not ids:
            return
        flags = classify(cls, [crops[i] for i in ids], ctx)
        for i, f in zip(ids, flags):
            classified[i] = True
            if f:
                crops[i] = np.ascontiguousarray(np.rot90(crops[i], 2))

    classify_idx([i for i, c in enumerate(crops) if c.shape[1] / c.shape[0] < 0.5])
    results = recognize(rec, crops, chars, ctx)
    low = [i for i, r in enumerate(results) if not classified[i] and r[1] < 0.65]
    if low:
        classify_idx(low)
        again = recognize(rec, [crops[i] for i in low], chars, ctx)
        for i, r in zip(low, again):
            if r[1] > results[i][1]:
                results[i] = r
    out = []
    for b, (text, conf) in zip(boxes, results):
        if conf >= 0.5 and text.strip():
            xs, ys = [p[0] for p in b], [p[1] for p in b]
            out.append({"text": text, "confidence": round(conf, 4),
                        "bbox": [math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys))],
                        "polygon": [[round(p[0], 1), round(p[1], 1)] for p in b]})
    acc = round(float(np.mean([o["confidence"] for o in out])) * 100) if out else 0
    return {"text": "\n".join(o["text"] for o in out), "lines": out, "accuracy": acc}


def to_hocr(result: dict, width: int, height: int) -> str:
    from html import escape

    rows = [f'<div class="ocr_page" id="page_1" title="bbox 0 0 {width} {height}">']
    for i, ln in enumerate(result["lines"], 1):
        l, t, r, b = ln["bbox"]
        rows.append(f'  <span class="ocr_line" id="line_{i}" title="bbox {l} {t} {r} {b}"><span class="ocrx_word" '
                    f'id="word_{i}" title="bbox {l} {t} {r} {b}">{escape(ln["text"], quote=True)}</span></span>')
    rows.append("</div>")
    return "\n".join(rows)
