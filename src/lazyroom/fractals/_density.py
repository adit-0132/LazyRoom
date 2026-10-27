"""Numba port of fractal_engine ``attractors.rs`` + density helpers of ``render.rs`` (Apache-2.0).

Buddhabrot (Halton sampling), the 2D maps (Hopalong, Martin, ..., Gumowski-Mira), the deterministic
affine IFS (Barnsley fern, ...) and geometric constructions (Pythagoras tree, H-tree, Heighway dragon,
Koch snowflake, Hilbert curve) and the 3D attractors (Pickover, Lorenz, Rossler).  The upstream
recursions are run with explicit stacks (Hilbert keeps the exact recursion order because it joins
consecutive points).

Viewport array ``vp`` layout: [cx, cy, span, aspect, width, height, rot, cos, sin, minx, maxx, miny, maxy]
where ``rot``/``cos``/``sin`` implement LazyRoom's optional view rotation (0 = upstream exact) and
min/max is the (rotated) viewport bounding box used for culling and clipping.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ._escape import fclamp, fmax, fmin, palette_sample, rround

EPS = 2.220446049250313e-16


# ---------------------------------------------------------------- viewport helpers

@njit(cache=True)
def plot_density(dens, vp, x, y, radius):
    cx, cy, span, aspect = vp[0], vp[1], vp[2], vp[3]
    width = int(vp[4])
    height = int(vp[5])
    hspan = span * aspect
    if vp[6] != 0.0:
        dx, dy = x - cx, y - cy
        sx = vp[7] * dx - vp[8] * dy
        sy = vp[8] * dx + vp[7] * dy
        nx = (sx + hspan * 0.5) / hspan
        ny = (span * 0.5 - sy) / span
    else:
        nx = (x - (cx - hspan * 0.5)) / hspan
        ny = ((cy + span * 0.5) - y) / span
    if not (0.0 <= nx < 1.0) or not (0.0 <= ny < 1.0):
        return
    px = int(nx * width)
    py = int(ny * height)
    r2 = radius * radius
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            if ox * ox + oy * oy > r2:
                continue
            qx = px + ox
            qy = py + oy
            if qx >= 0 and qy >= 0 and qx < width and qy < height:
                dens[qy * width + qx] += 1


@njit(cache=True)
def plot_density_r1(dens, vp, x, y):
    """``plot_density(.., radius = 1)`` unrolled: centre + 4 neighbours (same bounds rules)."""
    cx, cy, span, aspect = vp[0], vp[1], vp[2], vp[3]
    width = int(vp[4])
    height = int(vp[5])
    hspan = span * aspect
    if vp[6] != 0.0:
        dx, dy = x - cx, y - cy
        sx = vp[7] * dx - vp[8] * dy
        sy = vp[8] * dx + vp[7] * dy
        nx = (sx + hspan * 0.5) / hspan
        ny = (span * 0.5 - sy) / span
    else:
        nx = (x - (cx - hspan * 0.5)) / hspan
        ny = ((cy + span * 0.5) - y) / span
    if not (0.0 <= nx < 1.0) or not (0.0 <= ny < 1.0):
        return
    px = int(nx * width)
    py = int(ny * height)
    base = py * width + px
    dens[base] += 1
    if py > 0:
        dens[base - width] += 1
    if py + 1 < height:
        dens[base + width] += 1
    if px > 0:
        dens[base - 1] += 1
    if px + 1 < width:
        dens[base + 1] += 1


@njit(cache=True)
def projected_length(vp, length):
    hspan = vp[2] * vp[3]
    return max(length * vp[4] / hspan, length * vp[5] / vp[2])


@njit(cache=True)
def intersects_circle(vp, cx, cy, r):
    return cx + r >= vp[9] and cx - r <= vp[10] and cy + r >= vp[11] and cy - r <= vp[12]


@njit(cache=True)
def intersects_bounds(vp, minx, maxx, miny, maxy):
    return maxx >= vp[9] and minx <= vp[10] and maxy >= vp[11] and miny <= vp[12]


@njit(cache=True)
def plot_line(dens, vp, sx, sy, ex, ey):
    # clip_line_to_viewport (Liang-Barsky)
    dx = ex - sx
    dy = ey - sy
    near = 0.0
    far = 1.0
    for k in range(4):
        if k == 0:
            direction, distance = -dx, sx - vp[9]
        elif k == 1:
            direction, distance = dx, vp[10] - sx
        elif k == 2:
            direction, distance = -dy, sy - vp[11]
        else:
            direction, distance = dy, vp[12] - sy
        if abs(direction) < EPS:
            if distance < 0.0:
                return
            continue
        ratio = distance / direction
        if direction < 0.0:
            near = fmax(near, ratio)
        else:
            far = fmin(far, ratio)
        if near > far:
            return
    ax, ay = sx + dx * near, sy + dy * near
    bx, by = sx + dx * far, sy + dy * far
    hspan = vp[2] * vp[3]
    pdx = (bx - ax) * vp[4] / hspan
    pdy = (by - ay) * vp[5] / vp[2]
    steps = int(fmax(math.ceil(math.hypot(pdx, pdy)), 1.0))
    for i in range(steps + 1):
        f = i / steps
        plot_density_r1(dens, vp, ax + (bx - ax) * f, ay + (by - ay) * f)


# ---------------------------------------------------------------- Buddhabrot

@njit(cache=True)
def halton(index, base):
    f = 1.0
    r = 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r


@njit(cache=True)
def _in_main_bulbs(x, y):
    y2 = y * y
    s = x - 0.25
    q = s * s + y2
    return q * (q + s) <= 0.25 * y2 or (x + 1.0) * (x + 1.0) + y2 <= 0.0625


@njit(parallel=True, cache=True)
def buddhabrot(chunks, vp, samples, orbit_limit):
    """``buddhabrot_density``; samples are split over ``chunks`` (one density buffer each)."""
    n = chunks.shape[0]
    per = (samples + n - 1) // n
    min_escape = max(20, orbit_limit // 10)
    for ch in prange(n):
        dens = chunks[ch]
        orb_r = np.empty(orbit_limit, np.float64)
        orb_i = np.empty(orbit_limit, np.float64)
        start = ch * per + 1
        end = min(samples, (ch + 1) * per)
        for idx in range(start, end + 1):
            cr = -2.5 + 3.5 * halton(idx, 2)
            ci = -1.5 + 3.0 * halton(idx, 3)
            if _in_main_bulbs(cr, ci):
                continue
            zr = 0.0
            zi = 0.0
            esc = -1
            cnt = 0
            for it in range(orbit_limit):
                zr, zi = (zr * zr - zi * zi) + cr, (zr * zi + zi * zr) + ci
                orb_r[cnt] = zr
                orb_i[cnt] = zi
                cnt += 1
                if not (math.isfinite(zr) and math.isfinite(zi)) or zr * zr + zi * zi > 4.0:
                    esc = it + 1
                    break
            if esc >= min_escape:
                for k in range(cnt):
                    plot_density_r1(dens, vp, orb_i[k], -orb_r[k])


# ---------------------------------------------------------------- 2D maps

@njit(cache=True)
def attractor_step(kind, x, y, a, b, c, d):
    sign = math.copysign(1.0, x)
    if kind == 110:  # Hopalong
        return y - sign * math.sqrt(abs(b * x - c)), a - x
    if kind == 111:  # Martin
        return y - math.sin(x), a - x
    if kind == 112:  # Gingerbreadman
        return 1.0 - y + abs(x), x
    if kind == 113:  # Chip
        first = math.log(fmax(abs(b * x - c), 0.001))
        second = math.log(fmax(abs(c * x - b), 0.001))
        return y - sign * math.cos(first * first) * math.atan(second * second), a - x
    if kind == 114:  # Quadruptwo
        first = math.log(fmax(abs(b * x - c), 0.001))
        second = abs(c * x - b)
        return y - sign * math.sin(first) * math.atan(second * second), a - x
    if kind == 115:  # Threeply
        term = math.sin(x) * math.cos(b) + c - x * math.sin(a + b + c)
        return y - sign * abs(term), a - x
    if kind == 116:  # Clifford
        return math.sin(a * y) + c * math.cos(a * x), math.sin(b * x) + d * math.cos(b * y)
    if kind == 117:  # De Jong
        return math.sin(a * y) - math.cos(b * x), math.sin(c * x) - math.cos(d * y)
    if kind == 118:  # Ikeda
        ang = c - d / (1.0 + x * x + y * y)
        cs = math.cos(ang)
        sn = math.sin(ang)
        return a + b * (x * cs - y * sn), b * (x * sn + y * cs)
    if kind == 119:  # Tinkerbell
        return x * x - y * y + a * x + b * y, 2.0 * x * y + c * x + d * y
    # 120 Gumowski-Mira
    fx = b * x + 2.0 * (1.0 - b) * x * x / (1.0 + x * x)
    nx = y + a * (1.0 - b * y * y) * y + fx
    fn = b * nx + 2.0 * (1.0 - b) * nx * nx / (1.0 + nx * nx)
    return nx, -x + fn


@njit(cache=True)
def attractor_2d(dens, vp, kind, a, b, c, d, steps):
    if kind == 112 or kind == 120:
        x, y = -0.1, 0.0
    elif kind == 119:
        x, y = -0.72, -0.64
    else:
        x, y = 0.1, 0.0
    for _ in range(100):
        x, y = attractor_step(kind, x, y, a, b, c, d)
    for _ in range(steps):
        x, y = attractor_step(kind, x, y, a, b, c, d)
        if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > 1.0e7 or abs(y) > 1.0e7:
            break
        plot_density(dens, vp, x, y, 2)


# ---------------------------------------------------------------- affine IFS

@njit(cache=True)
def _ifs_node(dens, vp, maps, bounds, xx, xy, yx, yy, tx, ty, depth, st, sd, top):
    """Process one ``draw_affine_ifs`` call; children are pushed on (st, sd). Returns the new stack top."""
    minx = maxx = miny = maxy = 0.0
    for k in range(4):  # AffineBounds::transformed corner order
        px = bounds[0] if k < 2 else bounds[1]
        py = bounds[2] if k % 2 == 0 else bounds[3]
        qx = xx * px + xy * py + tx
        qy = yx * px + yy * py + ty
        if k == 0:
            minx = maxx = qx
            miny = maxy = qy
        else:
            minx = min(minx, qx)
            maxx = max(maxx, qx)
            miny = min(miny, qy)
            maxy = max(maxy, qy)
    if not intersects_bounds(vp, minx, maxx, miny, maxy):
        return top
    hspan = vp[2] * vp[3]
    pw = (maxx - minx) * vp[4] / hspan
    ph = (maxy - miny) * vp[5] / vp[2]
    if depth >= 64 or max(pw, ph) <= 0.75:
        plot_density_r1(dens, vp, (minx + maxx) * 0.5, (miny + maxy) * 0.5)
        return top
    for j in range(maps.shape[0] - 1, -1, -1):
        ixx, ixy, iyx, iyy, itx, ity = maps[j, 0], maps[j, 1], maps[j, 2], maps[j, 3], maps[j, 4], maps[j, 5]
        st[top, 0] = xx * ixx + xy * iyx
        st[top, 1] = xx * ixy + xy * iyy
        st[top, 2] = yx * ixx + yy * iyx
        st[top, 3] = yx * ixy + yy * iyy
        st[top, 4] = xx * itx + xy * ity + tx
        st[top, 5] = yx * itx + yy * ity + ty
        sd[top] = depth + 1
        top += 1
    return top


@njit(cache=True)
def _ifs_frontier(dens, vp, maps, bounds, target):
    """Breadth-first expansion from the identity transform until ``target`` open nodes exist."""
    cap = target * maps.shape[0] + 64
    qa = np.empty((cap, 6), np.float64)
    qd = np.empty(cap, np.int64)
    qa[0, 0], qa[0, 1], qa[0, 2], qa[0, 3], qa[0, 4], qa[0, 5] = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
    qd[0] = 0
    head = 0
    tail = 1
    tmp = np.empty((maps.shape[0], 6), np.float64)
    tmpd = np.empty(maps.shape[0], np.int64)
    while head < tail and tail - head < target:
        n = _ifs_node(dens, vp, maps, bounds, qa[head, 0], qa[head, 1], qa[head, 2], qa[head, 3], qa[head, 4],
                      qa[head, 5], qd[head], tmp, tmpd, 0)
        head += 1
        if tail + n > cap:  # compact
            qa[:tail - head] = qa[head:tail]
            qd[:tail - head] = qd[head:tail]
            tail -= head
            head = 0
        for k in range(n - 1, -1, -1):
            qa[tail] = tmp[k]
            qd[tail] = tmpd[k]
            tail += 1
    return qa[head:tail].copy(), qd[head:tail].copy()


@njit(parallel=True, cache=True)
def _ifs_parallel(chunks, vp, maps, bounds, nodes, depths):
    nch = chunks.shape[0]
    cap = 64 * maps.shape[0] + 8
    for ch in prange(nch):
        dens = chunks[ch]
        st = np.empty((cap, 6), np.float64)
        sd = np.empty(cap, np.int64)
        for i in range(ch, nodes.shape[0], nch):
            st[0] = nodes[i]
            sd[0] = depths[i]
            top = 1
            while top > 0:
                top -= 1
                top = _ifs_node(dens, vp, maps, bounds, st[top, 0], st[top, 1], st[top, 2], st[top, 3], st[top, 4],
                                st[top, 5], sd[top], st, sd, top)


def affine_ifs(dens, vp, maps, bounds, threads=1):
    """``draw_affine_ifs`` (depth-first over all map compositions; counts are order independent)."""
    nodes, depths = _ifs_frontier(dens, vp, maps, bounds, max(64, threads * 16))
    if nodes.shape[0] == 0:
        return
    pixels = dens.shape[0]
    nch = max(1, min(threads, (256 << 20) // max(pixels * 4, 1), nodes.shape[0]))
    chunks = np.zeros((nch, pixels), np.int32)
    _ifs_parallel(chunks, vp, maps, bounds, nodes, depths)
    dens += chunks.sum(axis=0, dtype=np.int64)


# ---------------------------------------------------------------- geometric constructions

@njit(cache=True)
def pythagoras_tree(dens, vp):
    st = np.empty((256, 4), np.float64)
    sd = np.empty(256, np.int64)
    st[0, 0], st[0, 1], st[0, 2], st[0, 3] = -0.34, -0.95, 0.34, -0.95
    sd[0] = 64
    top = 1
    while top > 0:
        top -= 1
        bsx, bsy, bex, bey = st[top, 0], st[top, 1], st[top, 2], st[top, 3]
        depth = sd[top]
        ex, ey = bex - bsx, bey - bsy
        elen = math.hypot(ex, ey)
        ox, oy = -ey, ex
        tsx, tsy = bsx + ox, bsy + oy
        tex, tey = bex + ox, bey + oy
        ccx = (bsx + bex + ox) * 0.5
        ccy = (bsy + bey + oy) * 0.5
        if not intersects_circle(vp, ccx, ccy, elen * 4.0):
            continue
        plot_line(dens, vp, bsx, bsy, bex, bey)
        plot_line(dens, vp, bex, bey, tex, tey)
        plot_line(dens, vp, tex, tey, tsx, tsy)
        plot_line(dens, vp, tsx, tsy, bsx, bsy)
        if depth <= 1 or projected_length(vp, elen) <= 0.6:
            continue
        tx, ty = tex - tsx, tey - tsy
        apx = tsx + (tx - ty) * 0.5
        apy = tsy + (ty + tx) * 0.5
        st[top, 0], st[top, 1], st[top, 2], st[top, 3] = apx, apy, tex, tey
        sd[top] = depth - 1
        top += 1
        st[top, 0], st[top, 1], st[top, 2], st[top, 3] = tsx, tsy, apx, apy
        sd[top] = depth - 1
        top += 1


@njit(cache=True)
def h_tree(dens, vp):
    st = np.empty((256, 3), np.float64)
    sh = np.empty(256, np.bool_)
    sd = np.empty(256, np.int64)
    st[0, 0], st[0, 1], st[0, 2] = 0.0, 0.0, 0.72
    sh[0] = True
    sd[0] = 64
    top = 1
    while top > 0:
        top -= 1
        cx, cy, hl = st[top, 0], st[top, 1], st[top, 2]
        horiz = sh[top]
        depth = sd[top]
        if not intersects_circle(vp, cx, cy, hl * 4.0):
            continue
        if horiz:
            fx, fy, sx, sy = cx - hl, cy, cx + hl, cy
        else:
            fx, fy, sx, sy = cx, cy - hl, cx, cy + hl
        plot_line(dens, vp, fx, fy, sx, sy)
        if depth <= 1 or projected_length(vp, hl * 2.0) <= 0.6:
            continue
        child = hl / math.sqrt(2.0)
        st[top, 0], st[top, 1], st[top, 2] = sx, sy, child
        sh[top] = not horiz
        sd[top] = depth - 1
        top += 1
        st[top, 0], st[top, 1], st[top, 2] = fx, fy, child
        sh[top] = not horiz
        sd[top] = depth - 1
        top += 1


@njit(cache=True)
def heighway_dragon(dens, vp):
    st = np.empty((256, 4), np.float64)
    sd = np.empty(256, np.int64)
    st[0, 0], st[0, 1], st[0, 2], st[0, 3] = -0.65, -0.15, 0.65, -0.15
    sd[0] = 64
    top = 1
    while top > 0:
        top -= 1
        sx, sy, ex, ey = st[top, 0], st[top, 1], st[top, 2], st[top, 3]
        depth = sd[top]
        dx, dy = ex - sx, ey - sy
        length = math.hypot(dx, dy)
        cx, cy = (sx + ex) * 0.5, (sy + ey) * 0.5
        if not intersects_circle(vp, cx, cy, length * 1.25):
            continue
        if depth == 0 or projected_length(vp, length) <= 0.6:
            plot_line(dens, vp, sx, sy, ex, ey)
            continue
        mx, my = cx + dy * 0.5, cy - dx * 0.5
        st[top, 0], st[top, 1], st[top, 2], st[top, 3] = ex, ey, mx, my
        sd[top] = depth - 1
        top += 1
        st[top, 0], st[top, 1], st[top, 2], st[top, 3] = sx, sy, mx, my
        sd[top] = depth - 1
        top += 1


@njit(cache=True)
def koch_snowflake(dens, vp):
    st = np.empty((512, 4), np.float64)
    sd = np.empty(512, np.int64)
    vx = np.array([-0.84, 0.84, 0.0])
    vy = np.array([-0.48, -0.48, 0.975])
    sin60 = -math.sqrt(3.0) * 0.5
    for edge in range(3):
        st[0, 0], st[0, 1] = vx[edge], vy[edge]
        st[0, 2], st[0, 3] = vx[(edge + 1) % 3], vy[(edge + 1) % 3]
        sd[0] = 32
        top = 1
        while top > 0:
            top -= 1
            sx, sy, ex, ey = st[top, 0], st[top, 1], st[top, 2], st[top, 3]
            depth = sd[top]
            length = math.hypot(ex - sx, ey - sy)
            cx, cy = (sx + ex) * 0.5, (sy + ey) * 0.5
            if not intersects_circle(vp, cx, cy, length * 0.8):
                continue
            if depth == 0 or projected_length(vp, length) <= 0.6:
                plot_line(dens, vp, sx, sy, ex, ey)
                continue
            ddx, ddy = (ex - sx) / 3.0, (ey - sy) / 3.0
            fx, fy = sx + ddx, sy + ddy
            tx, ty = sx + ddx * 2.0, sy + ddy * 2.0
            pkx = fx + ddx * 0.5 - ddy * sin60
            pky = fy + ddx * sin60 + ddy * 0.5
            segs = ((tx, ty, ex, ey), (pkx, pky, tx, ty), (fx, fy, pkx, pky), (sx, sy, fx, fy))
            for s in segs:
                st[top, 0], st[top, 1], st[top, 2], st[top, 3] = s[0], s[1], s[2], s[3]
                sd[top] = depth - 1
                top += 1


@njit(cache=True)
def hilbert_curve(dens, vp):
    st = np.empty((512, 6), np.float64)
    sd = np.empty(512, np.int64)
    st[0, 0], st[0, 1], st[0, 2], st[0, 3], st[0, 4], st[0, 5] = -0.9, -0.9, 1.8, 0.0, 0.0, 1.8
    sd[0] = 32
    top = 1
    has_prev = False
    prx = 0.0
    pry = 0.0
    while top > 0:
        top -= 1
        ox, oy, axx, axy, ayx, ayy = st[top, 0], st[top, 1], st[top, 2], st[top, 3], st[top, 4], st[top, 5]
        depth = sd[top]
        c1x, c1y = ox + axx, oy + axy
        c2x, c2y = ox + ayx, oy + ayy
        c3x, c3y = ox + axx + ayx, oy + axy + ayy
        minx = min(min(ox, c1x), min(c2x, c3x))
        maxx = max(max(ox, c1x), max(c2x, c3x))
        miny = min(min(oy, c1y), min(c2y, c3y))
        maxy = max(max(oy, c1y), max(c2y, c3y))
        if not intersects_bounds(vp, minx, maxx, miny, maxy):
            has_prev = False
            continue
        cell = max(math.hypot(axx, axy), math.hypot(ayx, ayy))
        if depth == 0 or projected_length(vp, cell) <= 0.6:
            px = ox + (axx + ayx) * 0.5
            py = oy + (axy + ayy) * 0.5
            if has_prev:
                plot_line(dens, vp, prx, pry, px, py)
            has_prev = True
            prx, pry = px, py
            continue
        hxx, hxy, hyx, hyy = axx * 0.5, axy * 0.5, ayx * 0.5, ayy * 0.5
        d1 = depth - 1
        # children in reverse order (stack): 4, 3, 2, 1
        st[top, 0], st[top, 1], st[top, 2], st[top, 3], st[top, 4], st[top, 5] = (
            ox + hxx + ayx, oy + hxy + ayy, -hyx, -hyy, -hxx, -hxy)
        sd[top] = d1
        top += 1
        st[top, 0], st[top, 1], st[top, 2], st[top, 3], st[top, 4], st[top, 5] = (
            ox + hxx + hyx, oy + hxy + hyy, hxx, hxy, hyx, hyy)
        sd[top] = d1
        top += 1
        st[top, 0], st[top, 1], st[top, 2], st[top, 3], st[top, 4], st[top, 5] = (
            ox + hxx, oy + hxy, hxx, hxy, hyx, hyy)
        sd[top] = d1
        top += 1
        st[top, 0], st[top, 1], st[top, 2], st[top, 3], st[top, 4], st[top, 5] = (ox, oy, hyx, hyy, hxx, hxy)
        sd[top] = d1
        top += 1


# ---------------------------------------------------------------- 3D attractors

@njit(cache=True)
def project(cam, px, py, pz, width, height, aspect):
    """Camera::project without the final bounds test -> (ok, x, y, depth)."""
    rx, ry, rz = px - cam[0], py - cam[1], pz - cam[2]
    depth = rx * cam[3] + ry * cam[4] + rz * cam[5]
    if not math.isfinite(depth) or depth <= 1.0e-6:
        return False, 0.0, 0.0, 0.0
    horiz = (rx * cam[6] + ry * cam[7] + rz * cam[8]) / (depth * cam[12] * aspect)
    vert = (rx * cam[9] + ry * cam[10] + rz * cam[11]) / (depth * cam[12])
    return True, (horiz + 1.0) * 0.5 * width, (1.0 - vert) * 0.5 * height, depth


@njit(cache=True)
def attractor_3d(dens, cam, width, height, aspect, kind, a, b, c, d, steps, rot, rc, rs):
    if kind == 1013:
        transient = 500
        x, y, z = 0.1, 0.1, 0.1
    elif kind == 1014:
        transient = 1000
        x, y, z = 0.1, 0.0, 0.0
    else:
        transient = 2000
        x, y, z = 0.1, 0.0, 0.0
    for i in range(transient + steps):
        if kind == 1013:
            x, y, z = (math.sin(a * y) - z * math.cos(b * x), z * math.sin(c * x) - math.cos(d * y), math.sin(x))
        elif kind == 1014:
            dt = 0.005
            x, y, z = (x + a * (y - x) * dt, y + (x * (b - z) - y) * dt, z + (x * y - c * z) * dt)
        else:
            dt = 0.02
            x, y, z = (x + (-y - z) * dt, y + (x + a * y) * dt, z + (b + z * (x - c)) * dt)
        if i < transient:
            continue
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            break
        if kind == 1013:
            vx, vy, vz = x * 0.3, y * 0.3, z * 0.3
        elif kind == 1014:
            vx, vy, vz = x * 0.05, y * 0.05, (z - 25.0) * 0.05
        else:
            vx, vy, vz = x * 0.1, y * 0.1, z * 0.1
        ok, sx, sy, depth = project(cam, vx, vy, vz, width, height, aspect)
        if not ok:
            continue
        if rot:
            hx, hy = sx - width * 0.5, sy - height * 0.5
            sx, sy = width * 0.5 + rc * hx + rs * hy, height * 0.5 - rs * hx + rc * hy
        if sx < -1.0 or sy < -1.0 or sx > width + 1.0 or sy > height + 1.0:
            continue
        weight = int(fclamp(rround(24.0 / fmax(depth, 0.25)), 1.0, 24.0))
        cxp = int(math.floor(sx))
        cyp = int(math.floor(sy))
        for oy in range(-1, 2):
            for ox in range(-1, 2):
                qx = cxp + ox
                qy = cyp + oy
                if qx < 0 or qy < 0 or qx >= width or qy >= height:
                    continue
                fall = 2 if (ox == 0 and oy == 0) else 1
                dens[qy * width + qx] += weight * fall


# ---------------------------------------------------------------- density -> colours

@njit(cache=True)
def write_density(dens, out, cycles, offset, inside, pal):
    h, w = out.shape[0], out.shape[1]
    mx = 0
    for i in range(dens.shape[0]):
        if dens[i] > mx:
            mx = dens[i]
    logmax = fmax(math.log(mx + 1.0), 1.0)
    for y in range(h):
        for x in range(w):
            v = dens[y * w + x]
            if v == 0:
                c = inside
            else:
                inten = math.log(v + 1.0) / logmax
                c = palette_sample(pal, (0.06 + inten ** 0.58 * 0.92) * cycles + offset)
            out[y, x, 0] = (c >> 16) & 0xFF
            out[y, x, 1] = (c >> 8) & 0xFF
            out[y, x, 2] = c & 0xFF
            out[y, x, 3] = (c >> 24) & 0xFF
