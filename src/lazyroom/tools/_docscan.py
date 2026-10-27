"""Classical document scanner (replacement for ImageToolbox's ML Kit document scanner).

ImageToolbox delegates to Google ML Kit (Android only).  This is an OpenCV pipeline:

1. page detection on a <= 1000 px proxy: several binarisations (Canny edges of a
   text-suppressed image, Otsu threshold, Otsu on saturation) -> external contours ->
   convex 4-gons from ``approxPolyDP`` scored by area x rectangularity; the best quad is
   refined by fitting a line to the contour points of every side and intersecting them;
2. perspective rectification (``warpPerspective``, output size from the longest opposite
   sides);
3. enhancement: ``color`` (none), ``grayscale``, ``bw`` (adaptive Gaussian threshold of
   the shadow-normalised luminance united with a global Otsu split) or ``shadow_removal`` (per-channel division by a
   dilated + median-blurred background estimate, which flattens lighting and shadows).
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

MODES = ["color", "grayscale", "bw", "shadow_removal"]
DETECT_SIZE = 1000


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # clockwise on screen starting near -pi (left)
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def _quad_score(quad: np.ndarray, area_img: float) -> float:
    area = cv2.contourArea(quad.astype(np.float32))
    if area < 0.12 * area_img or not cv2.isContourConvex(quad.astype(np.float32)):
        return -1.0
    (_, (rw, rh), _) = cv2.minAreaRect(quad.astype(np.float32))
    rect = area / max(rw * rh, 1e-6)
    q = order_corners(quad)
    angles = []
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cosv = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        angles.append(cosv)
    if max(angles) > 0.7:  # a corner sharper than ~45 degrees from square: not a page
        return -1.0
    return area / area_img * (0.5 + 0.5 * rect)


def _side_contrast(smooth: np.ndarray, quad: np.ndarray) -> int:
    """Number of quad sides with a clear luminance step across them (sides on the image border count)."""
    h, w = smooth.shape
    q = order_corners(quad)
    c = q.mean(axis=0)
    good = 0
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        if (abs(a[0] - b[0]) < 1e-6 and min(a[0], w - 1 - a[0]) <= 3) or \
                (abs(a[1] - b[1]) < 1e-6 and min(a[1], h - 1 - a[1]) <= 3):
            good += 1
            continue
        ab = b - a
        n = np.array([-ab[1], ab[0]]) / (np.linalg.norm(ab) + 1e-9)
        if np.dot((a + b) / 2 - c, n) < 0:
            n = -n  # outward normal
        ts = np.linspace(0.15, 0.85, 24)[:, None]
        pts = a + ts * ab
        inner, outer = pts - 4 * n, pts + 4 * n
        ok = (outer[:, 0] >= 0) & (outer[:, 0] < w) & (outer[:, 1] >= 0) & (outer[:, 1] < h)
        if ok.sum() < 6:
            good += 1  # the side runs along / outside the image border
            continue
        vi = smooth[np.clip(inner[ok, 1].astype(int), 0, h - 1), np.clip(inner[ok, 0].astype(int), 0, w - 1)]
        vo = smooth[outer[ok, 1].astype(int), outer[ok, 0].astype(int)]
        if np.median(np.abs(vi.astype(np.int16) - vo.astype(np.int16))) >= 12:
            good += 1
    return good


def _binarisations(gray: np.ndarray, rgb: np.ndarray) -> list[np.ndarray]:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    smooth = cv2.morphologyEx(cv2.GaussianBlur(gray, (5, 5), 0), cv2.MORPH_CLOSE, k, iterations=2)
    med = float(np.median(smooth))
    edges = cv2.Canny(smooth, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)) or 50)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    out = [edges]
    _, otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out += [otsu, 255 - otsu]
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    _, s_otsu = cv2.threshold(cv2.GaussianBlur(sat, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    out.append(cv2.morphologyEx(s_otsu, cv2.MORPH_OPEN, k))
    return out


def _refine(quad: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Fit a line to the contour points belonging to each side; intersect neighbouring lines."""
    q = order_corners(quad)
    pts = contour.reshape(-1, 2).astype(np.float64)
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        ab = b - a
        L = np.linalg.norm(ab)
        if L < 1e-6:
            return q
        t = ((pts - a) @ ab) / (L * L)
        rel = pts - a
        d = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / L
        sel = pts[(t > 0.1) & (t < 0.9) & (d < max(3.0, 0.02 * L))]
        if len(sel) < 5:
            return q
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        m = np.array([d1, -d2]).T
        if abs(np.linalg.det(m)) < 1e-9:
            return q
        s = np.linalg.solve(m, p2 - p1)
        out.append(p1 + s[0] * d1)
    out = np.array(out)
    # reject refinements that moved a corner implausibly far
    if np.max(np.linalg.norm(out - q, axis=1)) > 0.05 * np.linalg.norm(q[2] - q[0]):
        return q
    return order_corners(out)


def _refine_edges(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Sub-pixel refinement on the full-resolution image: along each side, find the strongest
    luminance step on the normal (+-search px), fit a robust line, intersect neighbours."""
    h, w = gray.shape
    q = order_corners(quad)
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    search = max(4.0, 0.006 * math.hypot(h, w))
    offs = np.arange(-search, search + 0.5, 0.5, dtype=np.float32)
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 10:
            return q
        u = ab / L
        n = np.array([-u[1], u[0]])
        ts = np.linspace(0.08, 0.92, 60)
        base = a[None, :] + ts[:, None] * ab[None, :]
        px = (base[:, None, 0] + offs[None, :] * n[0]).astype(np.float32)
        py = (base[:, None, 1] + offs[None, :] * n[1]).astype(np.float32)
        prof = cv2.remap(g, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        d = np.abs(np.diff(prof, axis=1))
        k = np.argmax(d, axis=1)
        strength = d[np.arange(len(k)), k]
        good = strength > max(4.0, 0.35 * float(np.median(strength)))
        if good.sum() < 8:
            return q
        km = k.astype(np.float64)
        # parabolic sub-sample peak
        for j in np.nonzero(good & (k > 0) & (k < d.shape[1] - 1))[0]:
            y0, y1, y2 = d[j, k[j] - 1], d[j, k[j]], d[j, k[j] + 1]
            den = y0 - 2 * y1 + y2
            if abs(den) > 1e-6:
                km[j] += 0.5 * (y0 - y2) / den
        off = offs[0] + (km + 0.5) * 0.5
        pts = base + off[:, None] * n[None, :]
        vx, vy, x0, y0 = cv2.fitLine(pts[good].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        m = np.array([d1, -d2]).T
        if abs(np.linalg.det(m)) < 1e-9:
            return q
        sol = np.linalg.solve(m, p2 - p1)
        out.append(p1 + sol[0] * d1)
    out = np.array(out)
    if np.max(np.linalg.norm(out - q, axis=1)) > 2.5 * search:
        return q
    return order_corners(out)


def detect_page(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect the page quadrilateral in a uint8 RGB image.  Returns (corners px, info)."""
    h, w = rgb.shape[:2]
    s = min(1.0, DETECT_SIZE / max(h, w))
    small = cv2.resize(rgb, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                       interpolation=cv2.INTER_AREA) if s < 1 else rgb
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), 2.0)
    area_img = float(sh * sw)
    best: tuple[float, np.ndarray, np.ndarray, str] | None = None
    names = ["edges", "otsu", "otsu_inverted", "saturation"]
    for name, binary in zip(names, _binarisations(gray, small)):
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in (0.015, 0.02, 0.03, 0.05):
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) != 4:
                    continue
                quad = approx.reshape(4, 2).astype(np.float64)
                # ignore quads that are just the image border
                if np.all((quad[:, 0] <= 2) | (quad[:, 0] >= sw - 3)) and np.all((quad[:, 1] <= 2) | (quad[:, 1] >= sh - 3)):
                    continue
                score = _quad_score(quad, area_img)
                if score > 0 and _side_contrast(smooth, quad) < 4:
                    score = -1.0
                if score > 0 and (best is None or score > best[0]):
                    best = (score, quad, c, name)
                break
    if best is None:
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)
        return corners, {"detected": False, "method": "full_image"}
    score, quad, contour, name = best
    refined = _refine(quad, contour)
    corners = order_corners((refined + 0.5) / s - 0.5)
    corners = _refine_edges(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), corners)
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    return corners, {"detected": True, "method": name, "score": round(float(score), 4)}


def rectify(img: np.ndarray, corners: np.ndarray, max_side: int = 20000) -> np.ndarray:
    tl, tr, br, bl = order_corners(corners)
    W = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))) + 1
    H = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))) + 1
    k = min(1.0, max_side / max(W, H))
    W, H = max(2, int(W * k)), max(2, int(H * k))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), dst)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _odd(v: float, lo: int = 3) -> int:
    v = max(lo, int(v))
    return v if v % 2 else v + 1


def _background(ch: np.ndarray) -> np.ndarray:
    n = min(ch.shape[:2])
    dil = cv2.dilate(ch, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(n / 50, 7),) * 2))
    return cv2.medianBlur(dil, min(_odd(n / 12, 21), 255))


def remove_shadows(rgb: np.ndarray) -> np.ndarray:
    out = np.empty_like(rgb)
    for c in range(rgb.shape[2]):
        ch = rgb[..., c]
        bg = _background(ch).astype(np.float32)
        out[..., c] = np.clip(ch.astype(np.float32) / np.maximum(bg, 1.0) * 255.0, 0, 255).astype(np.uint8)
    return out


def enhance(rgb: np.ndarray, mode: str, *, block: int = 0, offset: int = 10) -> np.ndarray:
    """uint8 RGB in, uint8 RGB (or single-channel for grayscale/bw) out."""
    if mode == "color":
        return rgb
    if mode == "shadow_removal":
        return remove_shadows(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if mode == "grayscale":
        return gray
    if mode == "bw":
        flat = remove_shadows(gray[..., None])[..., 0]
        bs = _odd(block) if block else _odd(min(gray.shape) / 30, 15)
        local = cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, offset)
        # the background is flat (~255) after normalisation, so a global Otsu split also fills large dark
        # areas that a purely local threshold would hollow out
        t, _ = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return np.where((local == 0) | (flat < min(t, 200)), 0, 255).astype(np.uint8)
    raise ValueError(mode)


def corner_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(order_corners(a) - order_corners(b), axis=1)))


def diag(h: int, w: int) -> float:
    return math.hypot(h, w)
