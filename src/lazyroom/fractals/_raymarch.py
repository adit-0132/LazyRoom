"""Numba port of fractal_engine 3D ray marching (``render.rs`` + ``distance_estimators.rs``, Apache-2.0).

Distance estimators: Mandelbulb, Mandelbox, Menger sponge, Sierpinski tetrahedron ("pyramid"),
quaternion Julia, octahedral / icosahedral IFS, Apollonian gasket, Kleinian, hybrid Mandelbulb-Julia,
quaternion cubic, Sierpinski gasket.  Shading (diffuse / specular / rim / fog), the optional
checkerboard floor and supersampling follow ``render_3d_sample`` / ``shade_floor`` / ``render_row``.

Camera array ``cam`` layout: origin(3), forward(3), right(3), up(3), tan(fov/2).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ._escape import divide_rounded, fclamp, fmax, mix_color, palette_sample, row_order, shade_color

FLOOR_OFFSET = 1.6


@njit(cache=True)
def _len(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


@njit(cache=True)
def _normalize(x, y, z):
    ln = math.sqrt(x * x + y * y + z * z)
    if ln > 1.0e-15:
        return x / ln, y / ln, z / ln
    return x, y, z


@njit(cache=True)
def _sort_desc(x, y, z):
    if x < y:
        x, y = y, x
    if x < z:
        x, z = z, x
    if y < z:
        y, z = z, y
    return x, y, z


@njit(cache=True)
def _qmul(ax, ay, az, aw, bx, by, bz, bw):
    return (aw * bx + bw * ax + ay * bz - az * by,
            aw * by + bw * ay + az * bx - ax * bz,
            aw * bz + bw * az + ax * by - ay * bx,
            aw * bw - ax * bx - ay * by - az * bz)


@njit(cache=True)
def mandelbulb(px, py, pz, power, iterations):
    zx, zy, zz = px, py, pz
    deriv = 1.0
    radius = 0.0
    for _ in range(iterations):
        radius = _len(zx, zy, zz)
        if radius > 2.5:
            break
        if radius < 1.0e-12:
            return 0.0
        theta = math.acos(fclamp(zz / radius, -1.0, 1.0))
        phi = math.atan2(zy, zx)
        deriv = radius ** (power - 1.0) * power * deriv + 1.0
        pr = radius ** power
        pt = theta * power
        pp = phi * power
        zx = math.sin(pt) * math.cos(pp) * pr + px
        zy = math.sin(pt) * math.sin(pp) * pr + py
        zz = math.cos(pt) * pr + pz
    return 0.5 * math.log(radius) * radius / fmax(abs(deriv), 1.0e-12)


@njit(cache=True)
def mandelbox(px, py, pz, iterations):
    zx, zy, zz = px, py, pz
    deriv = 1.0
    scale = -1.65
    min_r2 = 0.25
    for _ in range(iterations):
        zx = fclamp(zx, -1.0, 1.0) * 2.0 - zx
        zy = fclamp(zy, -1.0, 1.0) * 2.0 - zy
        zz = fclamp(zz, -1.0, 1.0) * 2.0 - zz
        r2 = zx * zx + zy * zy + zz * zz
        if r2 < min_r2:
            f = 1.0 / min_r2
            zx, zy, zz = zx * f, zy * f, zz * f
            deriv *= f
        elif r2 < 1.0:
            f = 1.0 / fmax(r2, 1.0e-12)
            zx, zy, zz = zx * f, zy * f, zz * f
            deriv *= f
        zx, zy, zz = zx * scale + px, zy * scale + py, zz * scale + pz
        deriv = deriv * abs(scale) + 1.0
    return _len(zx, zy, zz) / fmax(abs(deriv), 1.0e-12)


@njit(cache=True)
def _rem2(v):
    r = np.fmod(v, 2.0)
    if r < 0.0:
        r += 2.0
    return r


@njit(cache=True)
def menger(px, py, pz, iterations):
    qx, qy, qz = abs(px) - 1.0, abs(py) - 1.0, abs(pz) - 1.0
    outside = _len(fmax(qx, 0.0), fmax(qy, 0.0), fmax(qz, 0.0))
    dist = outside + min(fmax(fmax(qx, qy), qz), 0.0)
    scale = 1.0
    for _ in range(iterations):
        rx = _rem2(px * scale) - 1.0
        ry = _rem2(py * scale) - 1.0
        rz = _rem2(pz * scale) - 1.0
        scale *= 3.0
        ax = abs(1.0 - abs(rx) * 3.0)
        ay = abs(1.0 - abs(ry) * 3.0)
        az = abs(1.0 - abs(rz) * 3.0)
        da = fmax(ax, ay)
        db = fmax(ay, az)
        dc = fmax(az, ax)
        cut = (min(min(da, db), dc) - 1.0) / scale
        dist = fmax(dist, cut)
    return dist


@njit(cache=True)
def tetrahedron(px, py, pz, iterations):
    vx = (0.68, -0.68, -0.68, 0.68)
    vy = (0.68, -0.68, 0.68, -0.68)
    vz = (0.68, 0.68, -0.68, -0.68)
    scale = 1.0
    for _ in range(iterations):
        nx, ny, nz = vx[0], vy[0], vz[0]
        dx, dy, dz = px - nx, py - ny, pz - nz
        nd = dx * dx + dy * dy + dz * dz
        for k in range(1, 4):
            dx, dy, dz = px - vx[k], py - vy[k], pz - vz[k]
            d = dx * dx + dy * dy + dz * dz
            if d < nd:
                nx, ny, nz = vx[k], vy[k], vz[k]
                nd = d
        px, py, pz = (px - nx) * 2.0, (py - ny) * 2.0, (pz - nz) * 2.0
        scale *= 2.0
    return _len(px, py, pz) / scale - 0.012


@njit(cache=True)
def quaternion_julia(px, py, pz, q0, q1, q2, q3, iterations):
    # scalar-first catalogue constant -> scalar-last math type: (x, y, z, w) = (c.y, c.z, c.w, c.x)
    cx, cy, cz, cw = q1, q2, q3, q0
    x, y, z, w = py, pz, 0.0, px
    deriv = 1.0
    r2 = x * x + y * y + z * z + w * w
    for _ in range(iterations):
        deriv *= 2.0 * fmax(math.sqrt(r2), 1.0e-8)
        x, y, z, w = 2.0 * w * x, 2.0 * w * y, 2.0 * w * z, w * w - x * x - y * y - z * z
        x, y, z, w = x + cx, y + cy, z + cz, w + cw
        r2 = x * x + y * y + z * z + w * w
        if r2 > 16.0:
            break
    radius = fmax(math.sqrt(r2), 1.0e-12)
    return 0.5 * math.log(radius) * radius / fmax(deriv, 1.0e-12)


@njit(cache=True)
def octahedral_ifs(px, py, pz, object_scale, fold, iterations):
    x, y, z = px / object_scale, py / object_scale, pz / object_scale
    scale = fold + 1.0
    deriv = 1.0
    for _ in range(min(max(iterations, 4), 18)):
        x, y, z = _sort_desc(abs(x), abs(y), abs(z))
        x, y, z = x * scale - (scale - 1.0), y * scale - (scale - 1.0), z * scale - (scale - 1.0)
        deriv *= scale
        if _len(x, y, z) > 100.0:
            break
    return (_len(x, y, z) - 2.0) / fmax(abs(deriv), 1.0e-12) * object_scale


@njit(cache=True)
def icosahedral_ifs(px, py, pz, object_scale, fold, iterations, normals):
    x, y, z = px / object_scale, py / object_scale, pz / object_scale
    scale = fold + 1.0
    deriv = 1.0
    for _ in range(min(max(iterations, 4), 18)):
        for k in range(5):
            nx, ny, nz = normals[k, 0], normals[k, 1], normals[k, 2]
            d = x * nx + y * ny + z * nz
            if d < 0.0:
                x, y, z = x - nx * (2.0 * d), y - ny * (2.0 * d), z - nz * (2.0 * d)
        x, y, z = _sort_desc(abs(x), abs(y), abs(z))
        x, y, z = x * scale - (scale - 1.0), y * scale - (scale - 1.0), z * scale - (scale - 1.0)
        deriv *= scale
        if _len(x, y, z) > 100.0:
            break
    return (_len(x, y, z) - 2.0) / fmax(abs(deriv), 1.0e-12) * object_scale


@njit(cache=True)
def apollonian(px, py, pz, object_scale, fold, packing, iterations):
    x, y, z = px / object_scale, py / object_scale, pz / object_scale
    scale = (1.0 + packing) * fold
    deriv = 1.0
    for _ in range(min(max(iterations, 4), 18)):
        x, y, z = abs(x), abs(y), abs(z)
        r2 = fmax(x * x + y * y + z * z, 1.0e-15)
        p2 = packing * packing
        if r2 < p2:
            f = 1.0 / p2
            x, y, z = x * f, y * f, z * f
            deriv *= f
        elif r2 < 1.0:
            f = 1.0 / r2
            x, y, z = x * f, y * f, z * f
            deriv *= f
        x, y, z = _sort_desc(x, y, z)
        x, y, z = x * scale - packing, y * scale - packing, z * scale - packing
        deriv *= scale
        if _len(x, y, z) > 100.0:
            break
    return (_len(x, y, z) - 1.0) / fmax(abs(deriv), 1.0e-12) * object_scale


@njit(cache=True)
def kleinian(px, py, pz, object_scale, fold, min_radius, iterations):
    x, y, z = px / object_scale, py / object_scale, pz / object_scale
    scale = fold + 1.0
    deriv = 1.0
    for _ in range(min(max(iterations, 6), 16)):
        x, y, z = _sort_desc(abs(x), abs(y), abs(z))
        x = x * scale - 1.0 * (scale - 1.0)
        y = y * scale - 1.0 * (scale - 1.0)
        z = z * scale - 1.3 * (scale - 1.0)
        r2 = fmax(x * x + y * y + z * z, 1.0e-15)
        inv = fclamp(1.0 / r2, 1.0, 3.0)
        x, y, z = x * inv + 0.0, y * inv + 0.0, z * inv + (-min_radius)
        deriv *= inv * scale
    return (_len(x, y, z) - 0.4) / fmax(abs(deriv), 1.0e-12) * object_scale


@njit(cache=True)
def hybrid(px, py, pz, power, jx, jy, jz, iterations):
    power = fclamp(power, 2.0, 16.0)
    vx, vy, vz = px, py, pz
    deriv = 1.0
    radius = _len(vx, vy, vz)
    for it in range(min(max(iterations, 6), 24)):
        if radius > 2.0:
            break
        if radius < 1.0e-12:
            return 0.0
        theta = math.acos(fclamp(vz / radius, -1.0, 1.0))
        phi = math.atan2(vy, vx)
        deriv = radius ** (power - 1.0) * power * deriv + 1.0
        pr = radius ** power
        pt = theta * power
        pp = phi * power
        if it & 1 == 0:
            ax, ay, az = px, py, pz
        else:
            ax, ay, az = jx, jy, jz
        vx = math.sin(pt) * math.cos(pp) * pr + ax
        vy = math.sin(pt) * math.sin(pp) * pr + ay
        vz = math.cos(pt) * pr + az
        radius = _len(vx, vy, vz)
    return 0.5 * math.log(fmax(radius, 1.0e-12)) * radius / fmax(abs(deriv), 1.0e-12)


@njit(cache=True)
def quaternion_cubic(px, py, pz, q0, q1, q2, q3, iterations):
    cx, cy, cz, cw = q1, q2, q3, q0
    x, y, z, w = py, pz, 0.0, px
    dx, dy, dz, dw = 0.0, 0.0, 0.0, 1.0
    r2 = x * x + y * y + z * z + w * w
    for _ in range(min(max(iterations, 6), 24)):
        if math.sqrt(r2) > 2.0:
            break
        sx, sy, sz, sw = _qmul(x, y, z, w, x, y, z, w)
        tx, ty, tz, tw = _qmul(0.0, 0.0, 0.0, 3.0, sx, sy, sz, sw)
        dx, dy, dz, dw = _qmul(tx, ty, tz, tw, dx, dy, dz, dw)
        x, y, z, w = _qmul(x, y, z, w, sx, sy, sz, sw)
        x, y, z, w = x + cx, y + cy, z + cz, w + cw
        r2 = x * x + y * y + z * z + w * w
    radius = fmax(math.sqrt(r2), 1.0e-12)
    return 0.5 * math.log(radius) * radius / fmax(math.sqrt(dx * dx + dy * dy + dz * dz + dw * dw), 1.0e-12)


@njit(cache=True)
def sierpinski_gasket(px, py, pz, object_scale, fold, min_radius, iterations):
    x, y, z = px / object_scale, py / object_scale, pz / object_scale
    sf = 2.0 + fold * 0.5
    mr2 = min_radius ** 2
    scale = 1.0
    for _ in range(min(max(iterations, 5), 16)):
        if x + y < 0.0:
            ox = x
            x = -y
            y = -ox
        if x + z < 0.0:
            ox = x
            x = -z
            z = -ox
        if y + z < 0.0:
            oy = y
            y = -z
            z = -oy
        r2 = fmax(x * x + y * y + z * z, 1.0e-15)
        if r2 < mr2:
            inv = mr2 / r2
            x, y, z = x * inv, y * inv, z * inv
            scale *= inv
        x, y, z = x * sf - (sf - 1.0), y * sf - (sf - 1.0), z * sf - (sf - 1.0)
        scale *= sf
    return (_len(x, y, z) - 1.0) / fmax(abs(scale), 1.0e-12) * object_scale


@njit(cache=True)
def distance_estimate(kind, x, y, z, it, power, a, b, c, q, normals):
    if kind == 1001:
        return mandelbulb(x, y, z, power, it)
    if kind == 1002:
        return mandelbox(x, y, z, it)
    if kind == 1003:
        return menger(x, y, z, it)
    if kind == 1004:
        return tetrahedron(x, y, z, it)
    if kind == 1005:
        return quaternion_julia(x, y, z, q[0], q[1], q[2], q[3], it)
    if kind == 1006:
        return octahedral_ifs(x, y, z, a, b, it)
    if kind == 1007:
        return icosahedral_ifs(x, y, z, a, b, it, normals)
    if kind == 1008:
        return apollonian(x, y, z, a, b, power, it)
    if kind == 1009:
        return kleinian(x, y, z, a, b, power, it)
    if kind == 1010:
        return hybrid(x, y, z, power, a, b, 0.3, it)
    if kind == 1011:
        return quaternion_cubic(x, y, z, q[0], q[1], q[2], q[3], it)
    if kind == 1012:
        return sierpinski_gasket(x, y, z, a, b, power, it)
    return math.inf


@njit(cache=True)
def _shade_floor(px, pz, rdx, rdy, rdz, travel, max_dist, primary, secondary, inside):
    checker = (int(math.floor(px * 1.25)) + int(math.floor(pz * 1.25))) % 2
    surface = secondary if checker == 0 else primary
    lx, ly, lz = _normalize(-0.45, 0.8, -0.38)
    diffuse = fmax(0.0 * lx + 1.0 * ly + 0.0 * lz, 0.0)
    hx, hy, hz = _normalize(lx - rdx, ly - rdy, lz - rdz)
    spec = fmax(0.0 * hx + 1.0 * hy + 0.0 * hz, 0.0) ** 36.0
    shaded = shade_color(surface, 0.2 + diffuse * 0.7, spec * 0.35)
    fog = fclamp((travel / max_dist) ** 1.7, 0.0, 0.88)
    return mix_color(shaded, inside, fog)


@njit(cache=True)
def sample_3d(kind, cam, px, py, width, height, aspect, rot, rc, rs, max_steps, it, power, a, b, c, q, normals,
              cam_dist, kleinian_eps, show_floor, target_y, primary, secondary, cycles, offset, inside, pal):
    tan = cam[12]
    sx = (px / width * 2.0 - 1.0) * aspect * tan
    sy = (1.0 - py / height * 2.0) * tan
    if rot:
        sx, sy = rc * sx + rs * sy, -rs * sx + rc * sy
    rdx, rdy, rdz = _normalize(cam[3] + cam[6] * sx + cam[9] * sy,
                               cam[4] + cam[7] * sx + cam[10] * sy,
                               cam[5] + cam[8] * sx + cam[11] * sy)
    max_dist = cam_dist + 12.0
    floor_hit = False
    fpx = fpz = fdist = 0.0
    if show_floor and rdy < -1.0e-9:
        fh = target_y - FLOOR_OFFSET
        if cam[1] > fh:
            dist = (fh - cam[1]) / rdy
            if 0.0 <= dist <= max_dist:
                floor_hit = True
                fpx = cam[0] + rdx * dist
                fpz = cam[2] + rdz * dist
                fdist = dist
    ray_limit = fdist if floor_hit else max_dist
    travel = 0.0
    hit = False
    hx = hy = hz = 0.0
    hstep = 0
    for step in range(max_steps):
        x = cam[0] + rdx * travel
        y = cam[1] + rdy * travel
        z = cam[2] + rdz * travel
        d = distance_estimate(kind, x, y, z, it, power, a, b, c, q, normals)
        if not math.isfinite(d):
            break
        eps = 3.5e-4 if kleinian_eps else 0.00045 * (1.0 + travel * 0.08)
        if d < eps:
            hit = True
            hx, hy, hz = x, y, z
            hstep = step
            break
        travel += fclamp(d, eps * 0.5, 0.75)
        if travel > ray_limit:
            break
    if not hit:
        if floor_hit:
            return _shade_floor(fpx, fpz, rdx, rdy, rdz, fdist, max_dist, primary, secondary, inside)
        return inside
    e = 0.0009 * (1.0 + travel * 0.05)
    nx = (distance_estimate(kind, hx + e, hy + 0.0, hz + 0.0, it, power, a, b, c, q, normals)
          - distance_estimate(kind, hx - e, hy - 0.0, hz - 0.0, it, power, a, b, c, q, normals))
    ny = (distance_estimate(kind, hx + 0.0, hy + e, hz + 0.0, it, power, a, b, c, q, normals)
          - distance_estimate(kind, hx - 0.0, hy - e, hz - 0.0, it, power, a, b, c, q, normals))
    nz = (distance_estimate(kind, hx + 0.0, hy + 0.0, hz + e, it, power, a, b, c, q, normals)
          - distance_estimate(kind, hx - 0.0, hy - 0.0, hz - e, it, power, a, b, c, q, normals))
    nx, ny, nz = _normalize(nx, ny, nz)
    lx, ly, lz = _normalize(-0.45, 0.8, -0.38)
    diffuse = fmax(nx * lx + ny * ly + nz * lz, 0.0)
    hvx, hvy, hvz = _normalize(lx - rdx, ly - rdy, lz - rdz)
    spec = fmax(nx * hvx + ny * hvy + nz * hvz, 0.0) ** 28.0
    rim = (1.0 - abs(nx * (rdx * -1.0) + ny * (rdy * -1.0) + nz * (rdz * -1.0))) ** 2.0
    detail = hstep / max_steps
    pos = (0.12 + diffuse * 0.52 + rim * 0.18 + detail * 0.18) * cycles + offset
    surface = palette_sample(pal, pos)
    shaded = shade_color(surface, 0.28 + diffuse * 0.72, spec * 0.55)
    fog = fclamp((travel / max_dist) ** 1.7, 0.0, 0.82)
    return mix_color(shaded, inside, fog)


@njit(parallel=True, cache=True)
def render_3d(out, kind, cam, ss, aspect, rot, rc, rs, max_steps, it, power, a, b, c, q, normals, cam_dist,
              show_floor, target_y, primary, secondary, cycles, offset, inside, pal):
    height, width = out.shape[0], out.shape[1]
    count = ss * ss
    kl = kind == 1009
    order = row_order(height)
    for i in prange(height):
        y = order[i]
        for x in range(width):
            asum = 0
            rsum = 0
            gsum = 0
            bsum = 0
            for sy in range(ss):
                for sx in range(ss):
                    color = sample_3d(kind, cam, x + (sx + 0.5) / ss, y + (sy + 0.5) / ss, width, height, aspect,
                                      rot, rc, rs, max_steps, it, power, a, b, c, q, normals, cam_dist, kl,
                                      show_floor, target_y, primary, secondary, cycles, offset, inside, pal)
                    al = (color >> 24) & 0xFF
                    asum += al
                    rsum += ((color >> 16) & 0xFF) * al
                    gsum += ((color >> 8) & 0xFF) * al
                    bsum += (color & 0xFF) * al
            den = max(asum, 1)
            out[y, x, 0] = divide_rounded(rsum, den)
            out[y, x, 1] = divide_rounded(gsum, den)
            out[y, x, 2] = divide_rounded(bsum, den)
            out[y, x, 3] = divide_rounded(asum, count)


def icosahedral_normals() -> np.ndarray:
    phi = (1.0 + math.sqrt(5.0)) * 0.5
    raw = [(1.0, phi, 0.0), (0.0, 1.0, phi), (phi, 0.0, 1.0), (-1.0, phi, 0.0), (0.0, -1.0, phi)]
    out = np.empty((5, 3), np.float64)
    for i, (x, y, z) in enumerate(raw):
        ln = math.sqrt(x * x + y * y + z * z)
        out[i] = (x / ln, y / ln, z / ln) if ln > 1e-15 else (x, y, z)
    return out


__all__ = ["render_3d", "distance_estimate", "icosahedral_normals"]
