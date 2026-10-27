"""Collage Maker layout catalogue and renderer, ported from ImageToolbox (Apache-2.0).

Upstream (T8RIN/ImageToolbox, ``lib/collages``):

* ``frames/*FrameImage.kt`` + ``utils/CollageLayoutFactory.kt`` -> ``data/collage_layouts.json``.
  The JSON was produced by compiling the upstream Kotlin against minimal ``android.*`` stubs
  and dumping ``CollageLayoutFactory.loadFrameImages()``. DSL layouts are evaluated at their
  initial parameter values, so the draggable dividers aren't included. Layout order is the
  upstream picker order: asset names in byte order, stable-sorted by frame count.
* ``view/FramePhotoLayout.kt`` (frame metrics), ``view/FrameImageView.kt`` (``setSpace``,
  ``buildRealPath``, ``buildRealClearPath``, ``drawImage``, centre-crop image matrix) and
  ``utils/GeometryUtils.kt`` (polygon shrinking, circle-corner paths, Jarvis hull) -> below.

Rendering model (mirrors ``FramePhotoLayout.createImage``): the canvas is filled with the
background colour. Then, for each frame in list order (later frames draw on top), the image is
scaled to *cover* the frame's bound rectangle, centred, bilinear-filtered, clipped to the frame
polygon, shrunk by ``spacing`` with ``corner_radius`` round corners, and has its clear area
removed. The result is composited source-over inside the frame's integer bound rectangle.

``spacing`` and ``corner_radius`` are in output pixels. Upstream measures them in preview-view
pixels (sliders 0..50, defaults 10 / 0, ``CollageParams``) and multiplies them by the export
``outputScale``, so this matches upstream exported at ``outputScale == 1``. The aspect ratio
(upstream default 1:1) is set by ``width``/``height``. Every layout is normalised to [0, 1] and
stretched to the canvas, as ``bound * viewSize`` does upstream.

Deviations from upstream:

* Frames without an image (``len(images) < frames``) are left empty, showing the background.
  This is what upstream draws for a frame with no bitmap. Extra images are ignored.
* Anti-aliasing uses 4x4 supersampled coverage with the non-zero winding rule, not Skia's
  scan converter. Images are pre-shrunk with INTER_AREA when they're downscaled.
* Degenerate geometry that upstream turns into NaN or a spike to (0, 0) is guarded instead:
  a corner whose angle bisector is undefined stays sharp, and a path scale below zero
  (spacing larger than the frame) is clamped to 0. The Jarvis hull loop is bounded.
* Background gradients (``backgroundShader``), user pan/zoom/rotate, snap-to-borders and
  divider dragging aren't ported. With the initial cover matrix, snap-to-borders is a no-op.
* ``PhotoItem.hasBackground`` isn't used by any upstream layout, so it isn't implemented.
* Curved paths (circle, heart) are stored pre-flattened: 256 segments per circle, 32 per
  quadratic. Corner arcs are flattened at render time to within 0.05 px.

Upstream quirks that are kept:

* ``collage_15_2`` really has 17 frames, so like upstream it's listed with the 17-image layouts.
* Integer frame metrics (``computeFrameMetrics``) can leave 1-px seams or overlaps when
  ``spacing == 0``.
* Corner rounding with a radius larger than half an edge self-intersects, as upstream's path does.
"""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

F32 = np.float32
_DATA = Path(__file__).resolve().parent.parent / "data" / "collage_layouts.json"

#: Upstream defaults (feature/collage-maker ``CollageParams``); upstream slider ranges are 0..50.
DEFAULT_SPACING = 10.0
DEFAULT_CORNER_RADIUS = 0.0
DEFAULT_BACKGROUND = (1.0, 1.0, 1.0, 1.0)  # CollageMakerComponent: Color.White

_SS = 4  # coverage supersampling per axis
_ARC_TOLERANCE = 0.05  # max chord error (px) when flattening corner arcs


# --------------------------------------------------------------------------- catalogue
@functools.lru_cache(maxsize=1)
def _catalogue() -> tuple[list[dict], dict[str, dict]]:
    with open(_DATA, encoding="utf-8") as fh:
        layouts = json.load(fh)["layouts"]
    return layouts, {lay["id"]: lay for lay in layouts}


def load_layouts() -> list[dict]:
    """All upstream layouts in picker order (cached; treat as read-only)."""
    return _catalogue()[0]


def layout_ids(images: int | None = None) -> list[str]:
    """Layout ids, optionally only those with exactly ``images`` frames (upstream picker filter)."""
    return [lay["id"] for lay in load_layouts() if images is None or lay["images"] == images]


# --------------------------------------------------------------------------- GeometryUtils port
def _hull(pts: Sequence[tuple[float, float]]) -> set[int]:
    """GeometryUtils.jarvis: indices on the convex hull (integer-truncated coordinates, like upstream)."""
    n = len(pts)
    if n < 3:
        return set(range(n))
    xi = [int(x) for x, _ in pts]
    yi = [int(y) for _, y in pts]

    def ccw(p: int, q: int, r: int) -> bool:
        return (yi[q] - yi[p]) * (xi[r] - xi[q]) - (xi[q] - xi[p]) * (yi[r] - yi[q]) < 0

    left = 0
    for i in range(1, n):
        if xi[i] < xi[left]:
            left = i
    hull, p = set(), left
    for _ in range(n + 1):  # upstream: do { ... } while (p != leftMost)
        q = (p + 1) % n
        for i in range(n):
            if ccw(p, i, q):
                q = i
        hull.add(p)
        p = q
        if p == left:
            break
    return hull


def _coeffs(a, b) -> tuple[float, float, float]:
    return b[1] - a[1], a[0] - b[0], b[0] * a[1] - a[0] * b[1]


def _intersect(a, b, c, d, e, f):
    """Solve ax+by=c, dx+ey=f; None when there's no unique solution."""
    det = a * e - b * d
    if det == 0:
        return None
    return (c * e - b * f) / det, (a * f - c * d) / det


def _point_on_segment(a, b, dist):
    if dist == 0:
        return a
    dab = math.hypot(a[0] - b[0], a[1] - b[1])
    if dab == 0:
        return a
    dx = abs(a[0] - b[0]) * dist / dab
    dy = abs(a[1] - b[1]) * dist / dab
    return (a[0] - dx if a[0] > b[0] else a[0] + dx), (a[1] - dy if a[1] > b[1] else a[1] + dy)


def _point_on_bisector(a, b, c, d):
    ab, ac = _coeffs(a, b), _coeffs(a, c)
    vb = ac[0] * b[0] + ac[1] * b[1] + ac[2]
    vc = ab[0] * c[0] + ab[1] * c[1] + ab[2]
    s1, s2 = math.hypot(ab[0], ab[1]), math.hypot(ac[0], ac[1])
    if vc > 0:
        l1 = (ab[0], ab[1], d * s1 - ab[2])
    else:
        l1 = (-ab[0], -ab[1], d * s1 + ab[2])
    if vb > 0:
        l2 = (ac[0], ac[1], d * s2 - ac[2])
    else:
        l2 = (-ac[0], -ac[1], d * s2 + ac[2])
    return _intersect(*l1, *l2)


def _arc(a, b, c, r, concave):
    """GeometryUtils.createArc -> (centre, tangent point on AB, start deg, sweep deg) or None."""
    o = _point_on_bisector(a, b, c, r)
    if o is None:
        return None
    dma = math.hypot(a[0] - o[0], a[1] - o[1])
    if dma == 0:
        return None
    d = math.sqrt(max(dma * dma - r * r, 0.0))
    t1 = _point_on_segment(a, b, d)
    t2 = _point_on_segment(a, c, d)
    half = math.acos(min(1.0, r / dma))
    start = math.atan2(t1[1] - o[1], t1[0] - o[0])
    sweep = math.atan2(t2[1] - o[1], t2[0] - o[0]) - start
    if not concave:
        sweep = 2 * half
    start_deg, sweep_deg = math.degrees(start), math.degrees(sweep)
    tmp = math.degrees(2 * half)
    if abs(tmp - abs(sweep_deg)) > 1:
        sweep_deg = -tmp
    return o, t1, start_deg, sweep_deg


def _arc_points(o, r, start_deg, sweep_deg) -> list[tuple[float, float]]:
    sweep = math.radians(sweep_deg)
    step = 2 * math.acos(1 - _ARC_TOLERANCE / r) if r > _ARC_TOLERANCE else math.pi / 2
    n = max(1, math.ceil(abs(sweep) / step))
    st = math.radians(start_deg)
    return [(o[0] + r * math.cos(st + sweep * k / n), o[1] + r * math.sin(st + sweep * k / n)) for k in range(n + 1)]


def _round_corners(pts: Sequence[tuple[float, float]], corner: float) -> list[tuple[float, float]]:
    """GeometryUtils.createPathWithCircleCorner flattened to a closed polyline."""
    n = len(pts)
    if corner == 0 or n < 3:
        return list(pts)
    hull = _hull(pts)
    out: list[tuple[float, float]] = []
    for i in range(n):
        arc = _arc(pts[i], pts[i - 1], pts[(i + 1) % n], corner, i not in hull)
        if arc is None:  # upstream would emit a spike to (0, 0) here
            out.append(pts[i])
            continue
        o, t1, start, sweep = arc
        out.append(t1)
        out.extend(_arc_points(o, corner, start, sweep))  # arcTo: implicit lineTo(arc start) + arc
    return out


def _bounds(pts) -> tuple[float, float, float, float]:
    a = np.asarray(pts, np.float64)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def _regular_polygon(size: float, sides: int, corner: float):
    """GeometryUtils.createRegularPolygonPath(outPath, size, n, corner) -> (polyline, Path.computeBounds)."""
    r, c = size / 2, size / 2
    section = 2 * math.pi / sides
    verts = [(c + r * math.cos(section * i), c + r * math.sin(section * i)) for i in range(sides)]
    # computeBounds covers control points; for <=90 deg corner arcs those are the sharp vertices.
    return _round_corners(verts, corner), _bounds(verts)


def _rectangle(w: float, h: float, corner: float):
    """GeometryUtils.createRectanglePath -> (polyline, Path.computeBounds)."""
    verts = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    return _round_corners(verts, corner), _bounds(verts)


def _shrink_default(pts, space, bound):
    """GeometryUtils.shrinkPath: 2*space towards the canvas border, space towards neighbours."""
    if space == 0:
        return list(pts)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    left, top, right, bottom = bound
    out = []
    for x, y in pts:
        sx = 2 * space if (left == 0 and x < cx) or (right == 1 and x >= cx) else space
        sy = 2 * space if (top == 0 and y < cy) or (bottom == 1 and y >= cy) else space
        nx = x if abs(cx - x) < 1 else (x + sx if x < cx else x - sx)
        ny = y if abs(cy - y) < 1 else (y + sy if y < cy else y - sy)
        out.append((nx, ny))
    return out


def _center_index_3_3(points, bound) -> int:
    """FrameImageView.findCenterPointIndex (normalised points)."""
    idx = 0
    if bound[0] == 0 and bound[1] == 0:
        best = 1.0
        for i, (x, y) in enumerate(points):
            if 0 < x < 1 and 0 < y < 1 and x < best:
                idx, best = i, x
    else:
        best = 0.0
        for i, (x, y) in enumerate(points):
            if 0 < x < 1 and 0 < y < 1 and x > best:
                idx, best = i, x
    return idx


def _shrink_3_3(pts, ci, space, bound):
    """GeometryUtils.shrinkPathCollage_3_3."""
    n = len(pts)
    li, ri = (ci - 1) % n, (ci + 1) % n
    center, lp, rp = pts[ci], pts[li], pts[ri]
    left, top, right, bottom = bound
    out = []
    for i, (x, y) in enumerate(pts):
        sx = 2 * space if (left == 0 and x < center[0]) or (right == 1 and x >= center[0]) else space
        sy = 2 * space if (top == 0 and y < center[1]) or (bottom == 1 and y >= center[1]) else space
        nx = ny = 0.0  # upstream leaves PointF() when the neighbours aren't vertically aligned
        if lp[0] == rp[0]:
            if lp[0] < center[0]:
                nx = x + sx if x <= center[0] else x - sx
            else:
                nx = x + sx if x < center[0] else x - sx
            if i not in (li, ri, ci):
                ny = y + sy if y < center[1] else y - sy
            elif i in (li, ri):
                ny = y - space if y < center[1] else y + space
            else:
                ny = y
        out.append((nx, ny))
    return out


def _shrink_point(a, b, c, dab, dac, same_b, same_c):
    """GeometryUtils.shrinkPoint: offset vertex A by dab from line AB and dac from line AC."""
    ab, ac = _coeffs(a, b), _coeffs(a, c)
    s1, s2 = math.hypot(ab[0], ab[1]), math.hypot(ac[0], ac[1])
    m, p = dab * s1 - ab[2], -dab * s1 - ab[2]
    n, q = dac * s2 - ac[2], -dac * s2 - ac[2]
    for k1, k2 in ((m, n), (m, q), (p, n), (p, q)):
        pt = _intersect(ab[0], ab[1], k1, ac[0], ac[1], k2)
        if pt is None:
            continue
        sign_c = (ab[0] * pt[0] + ab[1] * pt[1] + ab[2]) * (ab[0] * c[0] + ab[1] * c[1] + ab[2])
        sign_b = (ac[0] * pt[0] + ac[1] * pt[1] + ac[2]) * (ac[0] * b[0] + ac[1] * b[1] + ac[2])
        if (sign_c > 0) == same_c and (sign_b > 0) == same_b:
            return pt
    return None


def _shrink_common(pts, space, smap):
    """GeometryUtils.commonShrinkPath (per-vertex distances from the shrink map)."""
    if space == 0:
        return list(pts)
    hull = _hull(pts)
    n = len(pts)
    out = []
    for i in range(n):
        convex = i in hull
        pt = _shrink_point(pts[i], pts[i - 1], pts[(i + 1) % n],
                           smap[i][0] * space, smap[i][1] * space, convex, convex)
        out.append(pt if pt is not None else (0.0, 0.0))  # upstream: PointF(0, 0)
    return out


def _place_path(spec, vw, vh, f, space, corner, clear: bool):
    """FrameImageView.buildRealPath / buildRealClearPath -> polyline in view pixels."""
    pts = np.asarray(spec["points"], np.float64)
    bl, bt, br, bb = spec["bounds"]
    pw, ph = br - bl, bb - bt
    rb = spec["ratio_bound"]
    sr = spec["scale_ratio"]
    ns = 0.0 if clear else space * 2
    if f.get("fit_bound"):
        rx = sr * (vw * (rb[2] - rb[0]) - 2 * ns) / pw
        ry = sr * (vh * (rb[3] - rb[1]) - 2 * ns) / ph
    else:
        rx = ry = min(sr * (vh - 2 * ns) / ph, sr * (vw - 2 * ns) / pw)
    rx, ry = max(rx, 0.0), max(ry, 0.0)
    pts = pts * (rx, ry)
    bounds = (bl * rx, bt * ry, br * rx, bb * ry)
    cm = f.get("corner_method")
    if cm == "3_6":
        poly, bounds = _regular_polygon(min(bounds[2] - bounds[0], bounds[3] - bounds[1]), 6, corner)
        pts = np.asarray(poly, np.float64)
    elif cm == "3_13":
        poly, bounds = _rectangle(bounds[2] - bounds[0], bounds[3] - bounds[1], corner)
        pts = np.asarray(poly, np.float64)
    bw, bh = bounds[2] - bounds[0], bounds[3] - bounds[1]
    sm = f.get("shrink_method")
    if clear:
        if sm == "3_6":
            x = vw - bw / 2 if rb[0] > 0 else -bw / 2
            y = vh / 2 - bh / 2
        elif spec["center_in_clear_bound"]:
            x = rb[0] * vw + (vw / 2 - bw / 2)
            y = rb[1] * vh + (vh / 2 - bh / 2)
        else:
            x, y = rb[0] * vw, rb[1] * vh
            if spec["center_horizontal"]:
                x = vw / 2 - bw / 2
            if spec["center_vertical"]:
                y = vh / 2 - bh / 2
    elif sm in ("3_6", "3_8"):
        x, y = vw / 2 - bw / 2, vh / 2 - bh / 2
    else:
        ox = ns / rx if rx else 0.0
        oy = ns / ry if ry else 0.0
        x = rb[2] * vw - bw - ox if spec["align_parent_right"] else rb[0] * vw + ox
        y = rb[1] * vh + oy
        if spec["center_horizontal"]:
            x = vw / 2 - bw / 2
        if spec["center_vertical"]:
            y = vh / 2 - bh / 2
    return pts + (x, y)


def _frame_shapes(f: dict, vw: float, vh: float, space: float, corner: float):
    """FrameImageView.setSpace -> (visible polygon, clear polygon or None) in view pixels."""
    clear = None
    if f.get("clear_area_points"):
        cpts = [(x * vw, y * vh) for x, y in f["clear_area_points"]]
        clear = _round_corners(cpts, corner)
    elif f.get("clear_path"):
        clear = _place_path(f["clear_path"], vw, vh, f, space, corner, clear=True)

    if f.get("path"):
        return _place_path(f["path"], vw, vh, f, space, corner, clear=False), clear

    pts = [(x * vw, y * vh) for x, y in f["points"]]
    sm, bound = f.get("shrink_method"), f["bound"]
    if sm == "3_3":
        shrunk = _shrink_3_3(pts, _center_index_3_3(f["points"], bound), space, bound)
    elif sm == "using_map" and f.get("shrink_map"):
        shrunk = [(x + mx * space, y + my * space) for (x, y), (mx, my) in zip(pts, f["shrink_map"])]
    elif sm == "common" and f.get("shrink_map"):
        shrunk = _shrink_common(pts, space, f["shrink_map"])
    else:
        shrunk = _shrink_default(pts, 0.0 if f.get("disable_shrink") else space, bound)
    return _round_corners(shrunk, corner), clear


# --------------------------------------------------------------------------- rasterisation
def _coverage(polys, w: int, h: int) -> np.ndarray:
    """Anti-aliased coverage (h, w) of closed polygons (pixel edges on integers), non-zero winding.

    Exactly the fraction of a 4x4 sample grid per pixel that is inside. It works per sub-scanline
    on the sorted edge crossings, accumulating the inside spans straight into 1x buffers.
    """
    out = np.zeros((h, w), F32)
    segs = []
    for p in polys:
        p = np.asarray(p, np.float64)
        if len(p) >= 3:
            segs.append(np.concatenate([p, np.roll(p, -1, axis=0)], axis=1))
    if not segs:
        return out
    e = np.concatenate(segs)
    e = e[np.isfinite(e).all(axis=1) & (e[:, 1] != e[:, 3])]
    if not len(e):
        return out
    up = e[:, 3] > e[:, 1]
    ya = np.where(up, e[:, 1], e[:, 3])
    yb = np.where(up, e[:, 3], e[:, 1])
    xa = np.where(up, e[:, 0], e[:, 2])
    slope = (np.where(up, e[:, 2], e[:, 0]) - xa) / (yb - ya)

    # sub-scanline k (centre (k + .5) / SS) crosses an edge when ya <= centre < yb
    k0 = np.clip(np.ceil(ya * _SS - 0.5), 0, h * _SS).astype(np.int64)
    k1 = np.clip(np.ceil(yb * _SS - 0.5), 0, h * _SS).astype(np.int64)
    cnt = np.maximum(k1 - k0, 0)
    total = int(cnt.sum())
    if not total:
        return out
    ei = np.repeat(np.arange(len(e)), cnt)
    k = k0[ei] + np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    cx = xa[ei] + ((k + 0.5) / _SS - ya[ei]) * slope[ei]
    # samples s >= ceil(cx * SS - .5) lie right of the crossing
    s = np.clip(np.ceil(cx * _SS - 0.5), 0, w * _SS).astype(np.int64)
    d = np.where(up, 1, -1)[ei]
    order = np.lexsort((s, k))
    k, s, d = k[order], s[order], d[order]
    after = np.cumsum(d)
    first = np.r_[True, k[1:] != k[:-1]]
    row_base = (after - d)[np.maximum.accumulate(np.where(first, np.arange(total), 0))]
    after -= row_base
    before = after - d
    edge = np.where((before == 0) & (after != 0), 1.0, np.where((before != 0) & (after == 0), -1.0, 0.0))
    keep = edge != 0
    row, s, edge = k[keep] // _SS, s[keep], edge[keep]
    # span [a, b) in samples -> per pixel c: SS*[c > a//SS] + (SS - a%SS)*[c == a//SS], minus the same at b
    px, rem = s // _SS, s % _SS
    stride = w + 2
    for r0 in range(0, h, 512):
        r1 = min(h, r0 + 512)
        lo, hi = np.searchsorted(row, [r0, r1])
        if lo == hi:
            continue
        base = (row[lo:hi] - r0) * stride
        n = (r1 - r0) * stride
        step = np.bincount(base + px[lo:hi] + 1, weights=edge[lo:hi] * _SS, minlength=n)
        point = np.bincount(base + px[lo:hi], weights=edge[lo:hi] * (_SS - rem[lo:hi]), minlength=n)
        acc = np.cumsum(step.reshape(r1 - r0, stride), axis=1) + point.reshape(r1 - r0, stride)
        out[r0:r1] = acc[:, :w] * (1.0 / (_SS * _SS))
    return out


def _cover(img: np.ndarray, vw: int, vh: int) -> np.ndarray:
    """FrameImageView.createMatrixToDrawImageInCenterView: scale to cover the view, centred.

    Returns (vh, vw, 3) for RGB input, otherwise premultiplied (vh, vw, 4).
    """
    img = np.asarray(img, F32)
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"expected an (H, W, 3|4) image, got shape {img.shape}")
    if img.shape[2] == 4:  # premultiply before resampling
        img = np.concatenate([img[..., :3] * img[..., 3:4], img[..., 3:4]], axis=2)
    ih, iw = img.shape[:2]
    ratio = max(vw / iw, vh / ih)
    sx = sy = ratio
    if ratio < 1:  # pre-shrink so the bilinear warp doesn't alias
        nw, nh = max(1, round(iw * ratio)), max(1, round(ih * ratio))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        sx, sy = iw * ratio / nw, ih * ratio / nh
        ih, iw = nh, nw
    m = np.array([[sx, 0, (0.5 - iw / 2) * sx + vw / 2 - 0.5],
                  [0, sy, (0.5 - ih / 2) * sy + vh / 2 - 0.5]], np.float64)
    return cv2.warpAffine(img, m, (vw, vh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _frame_metrics(bound, width: int, height: int) -> tuple[int, int, int, int]:
    """FramePhotoLayout.computeFrameMetrics -> (left, top, frame width, frame height)."""
    l, t, r, b = bound
    left, top = int(width * l), int(height * t)
    fw = width - left if r == 1 else int(width * (r - l) + 0.5)
    fh = height - top if b == 1 else int(height * (b - t) + 0.5)
    return left, top, fw, fh


# --------------------------------------------------------------------------- public renderer
def render_collage(images: list[np.ndarray], layout_id: str, width: int, height: int,
                   spacing: float = DEFAULT_SPACING, corner_radius: float = DEFAULT_CORNER_RADIUS,
                   background=DEFAULT_BACKGROUND) -> np.ndarray:
    """Render ``layout_id`` at ``width`` x ``height``; returns float32 (height, width, 4), straight alpha.

    ``images[i]`` goes into frame ``i`` (upstream ``updateImages`` order). ``spacing`` and
    ``corner_radius`` are output pixels; ``background`` is (r, g, b[, a]) in 0..1.
    """
    by_id = _catalogue()[1]
    if layout_id not in by_id:
        raise ValueError(f"unknown collage layout {layout_id!r}")
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError("width and height must be >= 1")
    if spacing < 0 or corner_radius < 0:
        raise ValueError("spacing and corner_radius must be >= 0")
    bg = np.asarray(background, F32).reshape(-1)
    if bg.size == 3:
        bg = np.append(bg, F32(1.0))
    if bg.size != 4:
        raise ValueError("background must be (r, g, b) or (r, g, b, a)")
    bg = np.clip(bg, 0, 1)
    canvas = np.empty((height, width, 4), F32)
    canvas[:] = np.append(bg[:3] * bg[3], bg[3])

    for frame, image in zip(by_id[layout_id]["frames"], images):
        left, top, fw, fh = _frame_metrics(frame["bound"], width, height)
        x0, y0 = max(left, 0), max(top, 0)
        x1, y1 = min(left + fw, width), min(top + fh, height)
        if fw < 1 or fh < 1 or x1 <= x0 or y1 <= y0:
            continue
        shape, clear = _frame_shapes(frame, float(fw), float(fh), float(spacing), float(corner_radius))
        cov = _coverage([shape], fw, fh)
        if clear is not None:
            cov *= 1.0 - _coverage([clear], fw, fh)
        sl = (slice(y0 - top, y1 - top), slice(x0 - left, x1 - left))
        src = _cover(image, fw, fh)[sl]
        cov = cov[sl][..., None]
        a = cov * src[..., 3:4] if src.shape[2] == 4 else cov  # layer alpha
        region = canvas[y0:y1, x0:x1]
        region *= 1.0 - a  # source-over with a premultiplied layer
        src = src[..., :3]
        src *= cov
        region[..., :3] += src
        region[..., 3:4] += a

    alpha = canvas[..., 3:4]
    np.divide(canvas[..., :3], alpha, out=canvas[..., :3], where=alpha > 1e-6)
    canvas[..., :3] *= alpha > 1e-6
    return np.clip(canvas, 0, 1, out=canvas)
