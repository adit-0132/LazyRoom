"""numba port of ImageToolbox's ray-marching texture shader (RaymarchTextureGenerator.kt, Apache-2.0, T8RIN).

One kernel serves all 15 shapes (``kind`` selects the signed distance function).
The GLSL is followed line by line; arithmetic is float64 (the GPU runs highp
float32).  GLSL ``rotate(a)`` returns ``mat2(c,-s,s,c)`` (column-major), so
``rotate(a) * v == (c*v.x + s*v.y, -s*v.x + c*v.y)``.
"""
from __future__ import annotations

import math

import numpy as np

from ._nb import njit, prange

# shape selectors (order of TextureFilterType)
SPHERE, CUBE, CUBE_STAIR, CUBE_SPHERES, EMPTIED_CUBE, TORUS, MOBIUS_TORUS, GYRO_RINGS, TORUS_GEM, \
    CUBE_OCTAHEDRON, INFINITE_SPHEROIDS, INFINITE_STRUCTURE, INFINITE_CYLINDERS, TRUCHET_RINGS, \
    TRUCHET_PIPES = range(15)

# uniform vector layout
U_SIZE, U_THICKNESS, U_ROUNDNESS, U_COUNT, U_INNER = 0, 1, 2, 3, 4
U_RXC, U_RXS, U_RYC, U_RYS, U_RZC, U_RZS = 5, 6, 7, 8, 9, 10
N_UNIFORMS = 11

PI = 3.14159265359  # the shader's constant


@njit(cache=True, inline="always")
def _box(px, py, pz, bx, by, bz):
    qx = abs(px) - bx
    qy = abs(py) - by
    qz = abs(pz) - bz
    mx = max(qx, 0.0)
    my = max(qy, 0.0)
    mz = max(qz, 0.0)
    return math.sqrt(mx * mx + my * my + mz * mz) + min(max(qx, max(qy, qz)), 0.0)


@njit(cache=True, inline="always")
def _torus(px, py, pz, radius, width):
    a = math.sqrt(px * px + pz * pz) - radius
    return math.sqrt(a * a + py * py) - width


@njit(cache=True, inline="always")
def _rep(v):
    # GLSL mod(v + 0.5, 1.0) - 0.5
    w = v + 0.5
    return w - math.floor(w) - 0.5


@njit(cache=True)
def _stairs(px, py, pz):
    d = 10.0
    for i in range(6):
        st = i / 6.0
        d = min(d, _box(px, py - (st - 0.5), pz - (st - 0.5), 0.65, 0.09, 0.65 - st * 0.5))
    return d


@njit(cache=True)
def _mobius(px, py, pz, thickness, count):
    angle = math.atan2(pz, px)
    a = angle * count * 0.5
    c = math.cos(a)
    s = math.sin(a)
    vx = math.sqrt(px * px + pz * pz) - 0.65
    vy = py
    qx = c * vx + s * vy
    qy = -s * vx + c * vy
    return _box(qx, qy, 0.0, thickness, 0.035, 1.0)


@njit(cache=True)
def _truchet(px, py, pz, thickness, pipes):
    cx = math.floor(px + 0.5)
    cy = math.floor(py + 0.5)
    cz = math.floor(pz + 0.5)
    qx = _rep(px)
    qy = _rep(py)
    qz = _rep(pz)
    # fract(sin(dot(cell, vec3(127.1, 311.7, 74.7))) * 43758.5453), evaluated in float32 like the GPU
    dd = (np.float32(cx) * np.float32(127.1) + np.float32(cy) * np.float32(311.7)
          + np.float32(cz) * np.float32(74.7))
    hv = np.float32(np.sin(dd)) * np.float32(43758.5453)
    h = float(hv - np.floor(hv))
    if h > 0.5:
        qx = -qx
    if h > 0.75 or h < 0.25:
        tx, ty, tz = qy, qz, qx  # q = q.yzx
        qx, qy, qz = tx, ty, tz
    if pipes:
        ax = qx - 0.5
        ay = qy - 0.5
        la = math.sqrt(ax * ax + ay * ay) - 0.5
        a = math.sqrt(la * la + qz * qz)
        bx = qx + 0.5
        by = qy + 0.5
        lb = math.sqrt(bx * bx + by * by) - 0.5
        b = math.sqrt(lb * lb + qz * qz)
        return min(a, b) - thickness * 0.45
    return _torus(qx, qy, qz, 0.34, thickness * 0.45)


@njit(cache=True)
def _shape(kind, px, py, pz, u):
    th = u[U_THICKNESS]
    rd = u[U_ROUNDNESS]
    if kind == SPHERE:
        return math.sqrt(px * px + py * py + pz * pz) - 0.75
    if kind == CUBE:
        return _box(px, py, pz, 0.65, 0.65, 0.65) - rd
    if kind == CUBE_STAIR:
        return _stairs(px, py, pz) - rd
    if kind == CUBE_SPHERES:
        ax = abs(px) - 0.43
        ay = abs(py) - 0.43
        az = abs(pz) - 0.43
        return math.sqrt(ax * ax + ay * ay + az * az) - th
    if kind == EMPTIED_CUBE:
        return max(_box(px, py, pz, 0.7, 0.7, 0.7) - rd, -(math.sqrt(px * px + py * py + pz * pz) - u[U_INNER]))
    if kind == TORUS:
        return _torus(px, py, pz, 0.62, th)
    if kind == MOBIUS_TORUS:
        return _mobius(px, py, pz, th, u[U_COUNT]) - rd * 0.25
    if kind == GYRO_RINGS:
        return min(_torus(px, py, pz, 0.68, th * 0.5),
                   min(_torus(py, px, pz, 0.68, th * 0.5), _torus(px, pz, py, 0.68, th * 0.5)))
    if kind == TORUS_GEM:
        return (abs(math.sqrt(px * px + pz * pz) - 0.62) + abs(py) - th) * 0.7071 - rd
    if kind == CUBE_OCTAHEDRON:
        return max(_box(px, py, pz, 0.65, 0.65, 0.65), (abs(px) + abs(py) + abs(pz) - 1.2) * 0.57735) - rd
    bound = _box(px, py, pz, 2.4, 2.4, 2.4)
    if kind == INFINITE_SPHEROIDS:
        rx = _rep(px)
        ry = _rep(py)
        rz = _rep(pz)
        return max(math.sqrt(rx * rx + ry * ry + rz * rz) - th * 1.5, bound)
    if kind == INFINITE_STRUCTURE:
        rx = _rep(px)
        ry = _rep(py)
        rz = _rep(pz)
        d = min(_box(rx, ry, rz, 0.5, th, th), min(_box(rx, ry, rz, th, 0.5, th), _box(rx, ry, rz, th, th, 0.5)))
        return max(d - rd, bound)
    if kind == INFINITE_CYLINDERS:
        rx = _rep(px)
        rz = _rep(pz)
        return max(math.sqrt(rx * rx + rz * rz) - th, bound)
    if kind == TRUCHET_RINGS:
        return max(_truchet(px, py, pz, th, False), bound)
    return max(_truchet(px, py, pz, th, True), bound)


@njit(cache=True)
def _scene(kind, px, py, pz, u):
    # p.yz = rotate(rx) * p.yz
    c = u[U_RXC]
    s = u[U_RXS]
    y = c * py + s * pz
    z = -s * py + c * pz
    py, pz = y, z
    # p.xz = rotate(ry) * p.xz
    c = u[U_RYC]
    s = u[U_RYS]
    x = c * px + s * pz
    z = -s * px + c * pz
    px, pz = x, z
    # p.xy = rotate(rz) * p.xy
    c = u[U_RZC]
    s = u[U_RZS]
    x = c * px + s * py
    y = -s * px + c * py
    px, py = x, y
    size = u[U_SIZE]
    return _shape(kind, px / size, py / size, pz / size, u) * size


@njit(cache=True)
def _environment(dx, dy, dz, roughness, env, has_env):
    if has_env:
        uu = math.atan2(dz, dx) / (2.0 * PI) + 0.5
        vv = math.acos(min(max(dy, -1.0), 1.0)) / PI
        h = env.shape[0]
        w = env.shape[1]
        # GL_LINEAR + GL_CLAMP_TO_EDGE on texel centres
        fx = uu * w - 0.5
        fy = vv * h - 0.5
        x0 = math.floor(fx)
        y0 = math.floor(fy)
        tx = fx - x0
        ty = fy - y0
        ix0 = min(max(int(x0), 0), w - 1)
        ix1 = min(max(int(x0) + 1, 0), w - 1)
        iy0 = min(max(int(y0), 0), h - 1)
        iy1 = min(max(int(y0) + 1, 0), h - 1)
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        for k in range(4):
            a = env[iy0, ix0, k] + (env[iy0, ix1, k] - env[iy0, ix0, k]) * tx
            b = env[iy1, ix0, k] + (env[iy1, ix1, k] - env[iy1, ix0, k]) * tx
            v = a + (b - a) * ty
            if k == 0:
                s0 = v
            elif k == 1:
                s1 = v
            elif k == 2:
                s2 = v
            else:
                s3 = v
        inv = 1.0 / max(s3, 0.00001)
        return s0 * inv, s1 * inv, s2 * inv
    t = dy * 0.5 + 0.5
    skr = 0.06 + (0.75 - 0.06) * t
    skg = 0.09 + (0.87 - 0.09) * t
    skb = 0.16 + (1.0 - 0.16) * t
    e = 70.0 + (3.0 - 70.0) * roughness
    strip = math.pow(max(0.0, 1.0 - abs(dy - 0.3)), e)
    return skr + 1.0 * strip * 0.7, skg + 0.88 * strip * 0.7, skb + 0.67 * strip * 0.7


@njit(cache=True)
def _normalize(x, y, z):
    inv = 1.0 / math.sqrt(x * x + y * y + z * z)
    return x * inv, y * inv, z * inv


@njit(cache=True, parallel=True)
def render(kind, width, height, u, cam, obj, bg, env, has_env):
    """``cam`` = [aspect, distance, yaw, pitch, fov, metallic, roughness, transmission, light_az, light_el]
    (angles in degrees); ``obj``/``bg`` = RGBA 0..1; ``env`` = premultiplied RGBA float32 (H, W, 4)."""
    out = np.empty((height, width, 4), np.float32)
    aspect = cam[0]
    yaw = cam[2] * PI / 180.0
    pitch = cam[3] * PI / 180.0
    ox = cam[1] * (math.sin(yaw) * math.cos(pitch))
    oy = cam[1] * math.sin(pitch)
    oz = cam[1] * (math.cos(yaw) * math.cos(pitch))
    fx, fy, fz = _normalize(-ox, -oy, -oz)
    # right = normalize(cross(forward, vec3(0, 1, 0)))
    rx, ry, rz = _normalize(fy * 0.0 - fz * 1.0, fz * 0.0 - fx * 0.0, fx * 1.0 - fy * 0.0)
    # up = cross(right, forward)
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx
    tan_half = math.tan(cam[4] * PI / 180.0 * 0.5)
    metallic = cam[5]
    roughness = cam[6]
    transmission = cam[7]
    az = cam[8] * PI / 180.0
    el = cam[9] * PI / 180.0
    lx = math.sin(az) * math.cos(el)
    ly = math.sin(el)
    lz = math.cos(az) * math.cos(el)
    for j in prange(height):
        tcy = (j + 0.5) / height
        vy = -(tcy * 2.0 - 1.0)
        for i in range(width):
            tcx = (i + 0.5) / width
            vx = (tcx * 2.0 - 1.0) * aspect
            dx, dy, dz = _normalize(fx + tan_half * (vx * rx + vy * ux),
                                    fy + tan_half * (vx * ry + vy * uy),
                                    fz + tan_half * (vx * rz + vy * uz))
            distance = 0.0
            hit = False
            for _ in range(128):
                st = _scene(kind, ox + dx * distance, oy + dy * distance, oz + dz * distance, u)
                if abs(st) < 0.001 * max(1.0, distance):
                    hit = True
                    break
                distance += max(abs(st) * 0.6, 0.0005)
                if distance > 30.0:
                    break
            if not hit:
                out[j, i, 0] = bg[0]
                out[j, i, 1] = bg[1]
                out[j, i, 2] = bg[2]
                out[j, i, 3] = bg[3]
                continue
            px = ox + dx * distance
            py = oy + dy * distance
            pz = oz + dz * distance
            e = 0.001
            nx, ny, nz = _normalize(
                _scene(kind, px + e, py, pz, u) - _scene(kind, px - e, py, pz, u),
                _scene(kind, px, py + e, pz, u) - _scene(kind, px, py - e, pz, u),
                _scene(kind, px, py, pz + e, u) - _scene(kind, px, py, pz - e, u))
            if nx * dx + ny * dy + nz * dz > 0.0:
                nx, ny, nz = -nx, -ny, -nz
            diffuse = max(nx * lx + ny * ly + nz * lz, 0.0)
            hx, hy, hz = _normalize(lx - dx, ly - dy, lz - dz)
            specular = math.pow(max(nx * hx + ny * hy + nz * hz, 0.0), 160.0 + (4.0 - 160.0) * roughness)
            fresnel = math.pow(1.0 - max(-(nx * dx + ny * dy + nz * dz), 0.0), 5.0)
            # reflect(ray, normal)
            dn = nx * dx + ny * dy + nz * dz
            fx_ = dx - 2.0 * dn * nx
            fy_ = dy - 2.0 * dn * ny
            fz_ = dz - 2.0 * dn * nz
            er, eg, eb = _environment(fx_, fy_, fz_, roughness, env, has_env)
            # refract(ray, normal, 0.67)
            eta = 0.67
            k = 1.0 - eta * eta * (1.0 - dn * dn)
            if k < 0.0:
                gx = 0.0
                gy = 0.0
                gz = 0.0
            else:
                sk = eta * dn + math.sqrt(k)
                gx = eta * dx - sk * nx
                gy = eta * dy - sk * ny
                gz = eta * dz - sk * nz
            gr, gg, gb = _environment(gx, gy, gz, roughness, env, has_env)
            mix_r = metallic * 0.75 + fresnel * 0.2
            spec_k = specular * (1.0 - roughness * 0.6)
            glass_mix = 0.08 + fresnel * 0.92
            for ch in range(3):
                tint = obj[ch]
                matte = tint * (0.18 + diffuse * 0.82)
                envc = er if ch == 0 else (eg if ch == 1 else eb)
                glsc = gr if ch == 0 else (gg if ch == 1 else gb)
                reflected = envc * (1.0 + (tint - 1.0) * metallic)
                col = matte + (reflected - matte) * mix_r
                col += spec_k * (0.25 + (tint - 0.25) * metallic)
                glass = glsc * tint
                gm = glass + (reflected - glass) * glass_mix
                col = col + (gm - col) * transmission
                out[j, i, ch] = min(max(col, 0.0), 1.0)
            out[j, i, 3] = obj[3]
    return out
