"""Numba port of fractal_engine ``render.rs`` 2D per-pixel renderers (Apache-2.0, ImageToolboxLibs).

Escape-time iteration (``iterate_escape_time``), Magnet I/II, Lyapunov, Sierpinski carpet/triangle,
Collatz, the double-precision perturbation sampler used for deep zoom, palette sampling and the
supersampled row renderer.  Complex arithmetic is written out component-wise in the same
evaluation order as the Rust ``Complex`` operators so float rounding matches.  Colours are ARGB
ints held in int64, exactly like the engine's ``u32`` colours.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

TWO_PI = 2.0 * math.pi
LN2 = math.log(2.0)
INF = math.inf


# ---------------------------------------------------------------- scalar helpers (Rust semantics)

@njit(cache=True)
def fmin(a, b):  # f64::min: NaN-ignoring
    if a != a:
        return b
    if b != b:
        return a
    return a if a < b else b


@njit(cache=True)
def fmax(a, b):  # f64::max: NaN-ignoring
    if a != a:
        return b
    if b != b:
        return a
    return a if a > b else b


@njit(cache=True)
def fclamp(x, lo, hi):  # f64::clamp (NaN passes through)
    if x < lo:
        x = lo
    if x > hi:
        x = hi
    return x


@njit(cache=True)
def rround(x):  # f64::round: half away from zero
    t = float(math.trunc(x))
    if abs(x - t) >= 0.5:
        t += 1.0 if x > 0.0 else -1.0
    return t


@njit(cache=True)
def rem1(x):  # x.rem_euclid(1.0)
    r = np.fmod(x, 1.0)
    if r < 0.0:
        r += 1.0
    return r


@njit(cache=True)
def norm_angle(re, im):
    return rem1(math.atan2(im, re) / TWO_PI + 1.0)


# ---------------------------------------------------------------- colours

@njit(cache=True)
def interp_color(first, second, fraction):
    out = 0
    for shift in (24, 16, 8, 0):
        s = (first >> shift) & 0xFF
        e = (second >> shift) & 0xFF
        v = fclamp(rround(s + (e - s) * fraction), 0.0, 255.0)
        out |= int(v) << shift
    return out


@njit(cache=True)
def palette_sample(pal, position):
    if math.isfinite(position):
        wrapped = rem1(position)
    else:
        wrapped = 0.0
    n = pal.shape[0]
    scaled = wrapped * (n - 1)
    fi = int(math.floor(scaled))
    si = min(fi + 1, n - 1)
    return interp_color(pal[fi], pal[si], scaled - fi)


@njit(cache=True)
def shade_color(color, light, specular):
    out = ((color >> 24) & 0xFF) << 24
    for shift in (16, 8, 0):
        v = fclamp(rround(((color >> shift) & 0xFF) * light + 255.0 * specular), 0.0, 255.0)
        out |= int(v) << shift
    return out


@njit(cache=True)
def mix_color(first, second, amount):
    return interp_color(first, second, fclamp(amount, 0.0, 1.0))


# ---------------------------------------------------------------- complex helpers (math.rs order)

@njit(cache=True)
def cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@njit(cache=True)
def cpowf(re, im, power):
    if power == 2.0:
        return re * re - im * im, re * im + im * re
    ip = rround(power)
    if abs(power - ip) < 1.0e-10 and ip > 0.0:
        rr, ri = 1.0, 0.0
        fr, fi = re, im
        rem = int(ip)
        while rem > 0:
            if rem & 1 == 1:
                rr, ri = cmul(rr, ri, fr, fi)
            rem >>= 1
            if rem > 0:
                fr, fi = cmul(fr, fi, fr, fi)
        return rr, ri
    radius = (re * re + im * im) ** (power * 0.5)
    angle = math.atan2(im, re) * power
    return radius * math.cos(angle), radius * math.sin(angle)


@njit(cache=True)
def cdiv(nr, ni, dr, di):
    """Complex::checked_div -> (ok, re, im)."""
    if not (math.isfinite(dr) and math.isfinite(di)):
        return False, 0.0, 0.0
    if abs(dr) >= abs(di):
        if dr == 0.0:
            return False, 0.0, 0.0
        ratio = di / dr
        scaled = dr + di * ratio
        return True, (nr + ni * ratio) / scaled, (ni - nr * ratio) / scaled
    ratio = dr / di
    scaled = di + dr * ratio
    return True, (nr * ratio + ni) / scaled, (ni * ratio - nr) / scaled


@njit(cache=True)
def csin(re, im):
    return math.sin(re) * math.cosh(im), math.cos(re) * math.sinh(im)


@njit(cache=True)
def ccos(re, im):
    return math.cos(re) * math.cosh(im), -math.sin(re) * math.sinh(im)


@njit(cache=True)
def csinh(re, im):
    return math.sinh(re) * math.cos(im), math.cosh(re) * math.sin(im)


@njit(cache=True)
def cexp(re, im):
    m = math.exp(re)
    return m * math.cos(im), m * math.sin(im)


@njit(cache=True)
def clog(re, im):
    n2 = re * re + im * im
    if n2 <= 0.0 or not math.isfinite(n2):
        return False, 0.0, 0.0
    return True, 0.5 * math.log(n2), math.atan2(im, re)


# ---------------------------------------------------------------- escape-time iteration

@njit(cache=True)
def _is_julia_start(kind):
    return (kind == 2 or kind == 106 or kind == 107 or kind == 132 or kind == 134 or kind == 136
            or kind == 141 or kind == 142 or kind == 144 or kind == 146 or kind == 140)


@njit(cache=True)
def iterate_escape(kind, pr, pi, max_it, power, bailout, jr, ji, phr, phi, nova):
    """render.rs ``iterate_escape_time`` -> (iterations, escaped, smooth, orbit, angle)."""
    if _is_julia_start(kind):
        zr, zi, cr, ci = pr, pi, jr, ji
    elif kind == 131 or kind == 149:
        zr, zi, cr, ci = pr, pi, pr, pi
    elif kind == 135 or kind == 150 or kind == 11:
        zr, zi, cr, ci = 1.0, 0.0, pr, pi
    elif kind == 139:
        zr, zi, cr, ci = 0.5, 0.0, pr, pi
    elif kind == 12:
        zr, zi, cr, ci = pr, pi, 0.0, 0.0
    else:
        zr, zi, cr, ci = 0.0, 0.0, pr, pi
    prr, pri = 0.0, 0.0
    orbit = INF
    mag2 = zr * zr + zi * zi
    it = 0
    escaped = False
    bail2 = bailout * bailout
    ok_log, lr, li = clog(cr, ci)
    # Exact-cycle detection (Brent): if z repeats bit-for-bit the orbit is periodic and can never escape,
    # so the pixel is "inside" exactly as after max_it iterations. Only for kinds whose step is a pure
    # function of z (not Phoenix / Spider / Man-o'-War, which carry extra state).
    cycle_check = kind != 10 and kind != 137 and kind != 138
    saved_r, saved_i = zr, zi
    next_save = 1
    while it < max_it:
        orr, ori = zr, zi
        converged = False
        if kind == 1 or kind == 5 or kind == 2:
            a, b = cpowf(orr, ori, power)
            zr, zi = a + cr, b + ci
        elif kind == 3 or kind == 106:
            a, b = cpowf(abs(orr), abs(ori), power)
            zr, zi = a + cr, b + ci
        elif kind == 4 or kind == 6:
            a, b = cpowf(orr, -ori, power)
            zr, zi = a + cr, b + ci
        elif kind == 7 or kind == 107:
            a, b = cpowf(orr, ori, power)
            zr, zi = abs(a) + cr, b + ci
        elif kind == 8:
            a, b = cpowf(abs(orr), abs(ori), power)
            zr, zi = abs(a) + cr, abs(b) + ci
        elif kind == 9:
            a, b = cpowf(orr, -abs(ori), power)
            zr, zi = a + cr, b + ci
        elif kind == 10:
            a, b = cpowf(orr, ori, power)
            mr, mi = cmul(prr, pri, phr, phi)
            nr, ni = (a + cr) + mr, (b + ci) + mi
            prr, pri = orr, ori
            zr, zi = nr, ni
        elif kind == 11 or kind == 12:
            nr, ni = cpowf(orr, ori, power)
            nr, ni = nr - 1.0, ni - 0.0
            dr, di = cpowf(orr, ori, power - 1.0)
            dr, di = dr * power, di * power
            ok, qr, qi = cdiv(nr, ni, dr, di)
            if not ok:
                it += 1
                break
            relax = nova if kind == 11 else 1.0
            xr = (orr - qr * relax) + cr
            xi = (ori - qi * relax) + ci
            dxr, dxi = xr - orr, xi - ori
            converged = dxr * dxr + dxi * dxi < 1.0e-20
            zr, zi = xr, xi
        elif kind == 131 or kind == 132:
            off = -1.0 if orr >= 0.0 else 1.0
            zr, zi = cmul(orr + off, ori + 0.0, cr, ci)
        elif kind == 133 or kind == 134:
            sr, si = cmul(orr, ori, orr, ori)
            ar, ai = sr + cr, si + ci
            p2r, p2i = cmul(ar, ai, ar, ai)
            zr, zi = (sr + p2r) + cr, (si + p2i) + ci
        elif kind == 135 or kind == 136:
            sr, si = csin(orr, ori)
            zr, zi = cmul(cr, ci, sr, si)
        elif kind == 137:
            nr, ni = cmul(orr, ori, orr, ori)
            nr, ni = nr + cr, ni + ci
            cr, ci = cr * 0.5 + nr, ci * 0.5 + ni
            zr, zi = nr, ni
        elif kind == 138:
            nr, ni = cmul(orr, ori, orr, ori)
            nr, ni = (nr + prr) + cr, (ni + pri) + ci
            prr, pri = orr, ori
            zr, zi = nr, ni
        elif kind == 139:
            ar, ai = cmul(cr, ci, orr, ori)
            zr, zi = cmul(ar, ai, 1.0 - orr, 0.0 - ori)
        elif kind == 140:
            cs = math.cos(ori)
            sn = math.sin(orr)
            if abs(cs) < 1.0e-14 or abs(sn) < 1.0e-14:
                zr, zi = INF, INF
            else:
                zr, zi = orr / cs + cr, ori / sn + ci
        elif kind == 141:
            tr, ti = cmul(orr, ori, cr, ci)
            off = -1.0 if ti >= 0.0 else 1.0
            zr, zi = cmul(orr + off, ori + 0.0, cr, ci)
        elif kind == 142:
            sr, si = cmul(orr, ori, orr, ori)
            if orr > 0.0:
                zr, zi = sr - 1.0, si - 0.0
            else:
                zr, zi = (sr - 1.0) + cr * orr, (si - 0.0) + ci * orr
        elif kind == 143 or kind == 144:
            a, b = ccos(orr, ori)
            zr, zi = a + cr, b + ci
        elif kind == 145 or kind == 146:
            a, b = csinh(orr, ori)
            zr, zi = a + cr, b + ci
        elif kind == 147:
            sr, si = cmul(orr, ori, orr, ori)
            cbr, cbi = cmul(sr, si, orr, ori)
            ok, qr, qi = cdiv(cbr, cbi, 1.0 + orr * orr, ori * ori)
            if ok:
                zr, zi = qr + cr, qi + ci
            else:
                zr, zi = INF, INF
        elif kind == 148:
            sr, si = cmul(orr, ori, orr, ori)
            ar, ai = cmul(orr, ori, sr, si)
            br, bi = cmul(cr - 1.0, ci - 0.0, orr, ori)
            zr, zi = (ar + br) - cr, (ai + bi) - ci
        elif kind == 149:
            sr, si = cmul(orr, ori, orr, ori)
            ok, qr, qi = cdiv(cr, ci, orr, ori)
            if ok:
                zr, zi = sr + qr, si + qi
            else:
                zr, zi = sr + INF, si + INF
        elif kind == 150:
            # complex_pow(c, z) = exp(z * log(c)); log(c) is loop-invariant (hoisted, bit-identical)
            if ok_log:
                er, ei = cmul(orr, ori, lr, li)
                zr, zi = cexp(er, ei)
            else:
                zr, zi = INF, INF
        it += 1
        mag2 = zr * zr + zi * zi
        orbit = fmin(orbit, fmin(fmin(abs(zr), abs(zi)), abs(math.sqrt(fmax(mag2, 0.0)) - 1.0)))
        if converged or not math.isfinite(mag2) or mag2 > bail2:
            escaped = True
            break
        if cycle_check:
            if zr == saved_r and zi == saved_i:
                break
            if it == next_save:
                saved_r, saved_i = zr, zi
                next_save *= 2
    if math.isfinite(mag2) and mag2 > 1.0:
        safe = mag2
    else:
        safe = fmax(bail2, 4.0)
    smooth = float(it)
    if escaped and kind != 12 and kind != 11:
        v = it + 1.0 - math.log(math.log(math.sqrt(safe))) / math.log(fmax(power, 2.0))
        if math.isfinite(v):
            smooth = v
    if not math.isfinite(orbit):
        orbit = 1.0
    return it, escaped, smooth, orbit, norm_angle(zr, zi)


@njit(cache=True)
def iterate_magnet(kind, cr, ci, max_it, bailout):
    zr, zi = 0.0, 0.0
    orbit = INF
    it = 0
    escaped = False
    lim = fmax(bailout, 100.0)
    lim = lim * lim
    while it < max_it:
        if kind == 101:
            sr, si = cmul(zr, zi, zr, zi)
            nr, ni = (sr + cr) - 1.0, (si + ci) - 0.0
            dr, di = (zr * 2.0 + cr) - 2.0, (zi * 2.0 + ci) - 0.0
        else:
            c1r, c1i = cr - 1.0, ci - 0.0
            c2r, c2i = cr - 2.0, ci - 0.0
            comr, comi = cmul(c1r, c1i, c2r, c2i)
            z3r, z3i = cpowf(zr, zi, 3.0)
            t1r, t1i = cmul(c1r, c1i, zr, zi)
            nr, ni = (z3r + t1r * 3.0) + comr, (z3i + t1i * 3.0) + comi
            zzr, zzi = cmul(zr, zi, zr, zi)
            t2r, t2i = cmul(c2r, c2i, zr, zi)
            dr = ((zzr * 3.0 + t2r * 3.0) + comr) + 1.0
            di = ((zzi * 3.0 + t2i * 3.0) + comi) + 0.0
        ok, qr, qi = cdiv(nr, ni, dr, di)
        if not ok:
            escaped = True
            it += 1
            break
        zr, zi = cmul(qr, qi, qr, qi)
        it += 1
        d1r, d1i = zr - 1.0, zi - 0.0
        dist = d1r * d1r + d1i * d1i
        orbit = fmin(orbit, math.sqrt(dist))
        if not (math.isfinite(zr) and math.isfinite(zi)) or zr * zr + zi * zi > lim:
            escaped = True
            break
        if dist < 1.0e-20:
            break
    return it, escaped, float(it), fmax(orbit, 1.0e-15), norm_angle(zr, zi)


@njit(cache=True)
def perturbation_sample(ref_r, ref_i, julia, dre, dim, max_it, bailout):
    if julia:
        dzr, dzi, dcr, dci = dre, dim, 0.0, 0.0
    else:
        dzr, dzi, dcr, dci = 0.0, 0.0, dre, dim
    bail2 = bailout * bailout
    orbit = INF
    mag2 = 0.0
    ar = ref_r[0] + dzr
    ai = ref_i[0] + dzi
    it = 0
    ridx = 0
    escaped = False
    while it < max_it:
        rr = ref_r[ridx]
        ri = ref_i[ridx]
        nr = ref_r[ridx + 1]
        ni = ref_i[ridx + 1]
        if not (math.isfinite(nr) and math.isfinite(ni)):
            escaped = True
            break
        odr, odi = dzr, dzi
        dzr = 2.0 * (rr * odr - ri * odi) + odr * odr - odi * odi + dcr
        dzi = 2.0 * (rr * odi + ri * odr) + 2.0 * odr * odi + dci
        ar = nr + dzr
        ai = ni + dzi
        it += 1
        mag2 = ar * ar + ai * ai
        orbit = fmin(orbit, fmin(fmin(abs(ar), abs(ai)), abs(math.sqrt(fmax(mag2, 0.0)) - 1.0)))
        if not math.isfinite(mag2) or mag2 > bail2:
            escaped = True
            break
        rmag = nr * nr + ni * ni
        if ridx > 0 and math.isfinite(rmag) and mag2 < 1.0e-6 * rmag:
            dzr = ar - ref_r[0]
            dzi = ai - ref_i[0]
            ridx = 0
        else:
            ridx += 1
    safe = mag2 if (math.isfinite(mag2) and mag2 > 1.0) else fmax(bail2, 4.0)
    smooth = float(it)
    if escaped:
        v = it + 1.0 - math.log(math.log(math.sqrt(safe))) / LN2
        if math.isfinite(v):
            smooth = v
    if not math.isfinite(orbit):
        orbit = 1.0
    return it, escaped, smooth, orbit, norm_angle(ar, ai)


@njit(cache=True)
def color_escape(it, escaped, smooth, orbit, angle, max_it, coloring, cycles, offset, inside, pal):
    if not escaped:
        return inside
    frac = it / max_it
    if coloring == 0:
        pos = smooth / max_it
    elif coloring == 1:
        pos = math.floor(frac * 64.0) / 64.0
    elif coloring == 2:
        pos = -math.log(fclamp(orbit, 1.0e-15, 1.0)) / 16.0 + frac * 0.15
    else:
        pos = angle + frac * 0.1
    return palette_sample(pal, pos * cycles + offset)


@njit(cache=True)
def lyapunov_exponent(pr, pi, seq, max_it):
    state = 0.5
    n = seq.shape[0]
    for idx in range(32):
        rate = pi if seq[idx % n] else pr
        state = rate * state * (1.0 - state)
    total = 0.0
    for idx in range(max_it):
        rate = pi if seq[idx % n] else pr
        deriv = fmax(abs(rate * (1.0 - 2.0 * state)), 1.0e-15)
        total += math.log(deriv)
        state = rate * state * (1.0 - state)
    return total / max_it


@njit(cache=True)
def color_lyapunov(pr, pi, seq, max_it, cycles, offset, pal):
    exponent = lyapunov_exponent(pr, pi, seq, max_it)
    if exponent < 0.0:
        pos = 0.42 + fclamp(-exponent / 2.5, 0.0, 1.0) * 0.58
    else:
        pos = 0.18 * (1.0 - fclamp(exponent / 2.5, 0.0, 1.0))
    return palette_sample(pal, pos * cycles + offset)


@njit(cache=True)
def sierpinski_carpet(pr, pi, requested):
    x = (pr + 1.0) * 0.5
    y = (pi + 1.0) * 0.5
    if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
        return -1.0
    eps = 2.220446049250313e-16
    x = min(x, 1.0 - eps)
    y = min(y, 1.0 - eps)
    depth = min(max(requested, 1), 14)
    for i in range(depth):
        x *= 3.0
        y *= 3.0
        if int(math.floor(x)) == 1 and int(math.floor(y)) == 1:
            return -1.0
        x = x - math.trunc(x)
        y = y - math.trunc(y)
        if i + 1 == depth:
            return (i + 1) / depth
    return 1.0


@njit(cache=True)
def sierpinski_triangle(pr, pi, requested):
    tx, ty = 0.0, 0.96
    lx, ly = -1.0, -0.78
    rx, ry = 1.0, -0.78
    den = (ly - ry) * (tx - rx) + (rx - lx) * (ty - ry)
    a = ((ly - ry) * (pr - rx) + (rx - lx) * (pi - ry)) / den
    b = ((ry - ty) * (pr - rx) + (tx - rx) * (pi - ry)) / den
    c = 1.0 - a - b
    if a < 0.0 or b < 0.0 or c < 0.0:
        return -1.0
    depth = min(max(requested, 1), 18)
    for i in range(depth):
        if a >= 0.5:
            a = a * 2.0 - 1.0
            b *= 2.0
            c *= 2.0
        elif b >= 0.5:
            b = b * 2.0 - 1.0
            a *= 2.0
            c *= 2.0
        elif c >= 0.5:
            c = c * 2.0 - 1.0
            a *= 2.0
            b *= 2.0
        else:
            return -1.0
        if i + 1 == depth:
            return (i + 1) / depth
    return 1.0


@njit(cache=True)
def color_collatz(pr, pi, max_it, bailout, cycles, offset, pal):
    vr, vi = pr, pi
    mind = INF
    bail2 = bailout * bailout
    pos = -1.0
    found = False
    for it in range(max_it):
        m2 = vr * vr + vi * vi
        if not math.isfinite(m2) or m2 > bail2:
            lg = math.log(fmax(m2, 1.0)) * 0.5
            smooth = it + 1.0 - math.log(fmax(lg / LN2, 1.0e-4)) / LN2
            pos = rem1(smooth / 50.0)
            found = True
            break
        mind = fmin(mind, math.sqrt(m2))
        rl = math.pi * vr
        im = math.pi * vi
        cosr, cosi = math.cos(rl) * math.cosh(im), -math.sin(rl) * math.sinh(im)
        ar, ai = 2.0 + vr * 7.0, 0.0 + vi * 7.0
        br, bi = 2.0 + vr * 5.0, 0.0 + vi * 5.0
        mr, mi = cmul(br, bi, cosr, cosi)
        vr, vi = (ar - mr) * 0.25, (ai - mi) * 0.25
    if not found:
        pos = rem1(mind * 2.0) if math.isfinite(mind) else 0.0
    return palette_sample(pal, pos * cycles + offset)


@njit(cache=True)
def sample_2d(kind, pr, pi, max_it, power, bailout, jr, ji, phr, phi, nova, coloring, cycles, offset, inside,
              pal, seq):
    if kind == 101 or kind == 102:
        it, esc, sm, orb, ang = iterate_magnet(kind, pr, pi, max_it, bailout)
        return color_escape(it, esc, sm, orb, ang, max_it, coloring, cycles, offset, inside, pal)
    if kind == 103:
        return color_lyapunov(pr, pi, seq, max_it, cycles, offset, pal)
    if kind == 104 or kind == 105:
        d = sierpinski_carpet(pr, pi, max_it) if kind == 104 else sierpinski_triangle(pr, pi, max_it)
        if d < 0.0:
            return inside
        return palette_sample(pal, (0.2 + d * 0.72) * cycles + offset)
    if kind == 108:
        return color_collatz(pr, pi, max_it, bailout, cycles, offset, pal)
    it, esc, sm, orb, ang = iterate_escape(kind, pr, pi, max_it, power, bailout, jr, ji, phr, phi, nova)
    return color_escape(it, esc, sm, orb, ang, max_it, coloring, cycles, offset, inside, pal)


@njit(cache=True)
def row_order(height):
    """Interleaved row order so static prange chunks get rows spread over the whole image (load balance)."""
    k = 16
    order = np.empty(height, np.int64)
    n = 0
    for r in range(k):
        for y in range(r, height, k):
            order[n] = y
            n += 1
    return order


@njit(cache=True)
def divide_rounded(n, d):
    return (n + d // 2) // d


@njit(parallel=True, cache=True)
def render_2d(out, kind, ss, cx, cy, span, aspect, rot, rc, rs, max_it, power, bailout, jr, ji, phr, phi, nova,
              coloring, cycles, offset, inside, pal, seq, use_pert, pert_julia, ref_r, ref_i):
    """render.rs ``render_row`` + ``render_2d_sample`` (unpremultiplied output, RGBA uint8)."""
    height, width = out.shape[0], out.shape[1]
    count = ss * ss
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
                    px = x + (sx + 0.5) / ss
                    py = y + (sy + 0.5) / ss
                    nx = px / width
                    ny = py / height
                    dre = (nx - 0.5) * span * aspect
                    dim = (0.5 - ny) * span
                    if rot:
                        dre, dim = rc * dre + rs * dim, -rs * dre + rc * dim
                    if use_pert:
                        it, esc, sm, orb, ang = perturbation_sample(ref_r, ref_i, pert_julia, dre, dim, max_it,
                                                                    bailout)
                        color = color_escape(it, esc, sm, orb, ang, max_it, coloring, cycles, offset, inside, pal)
                    else:
                        color = sample_2d(kind, cx + dre, cy + dim, max_it, power, bailout, jr, ji, phr, phi, nova,
                                          coloring, cycles, offset, inside, pal, seq)
                    a = (color >> 24) & 0xFF
                    asum += a
                    rsum += ((color >> 16) & 0xFF) * a
                    gsum += ((color >> 8) & 0xFF) * a
                    bsum += (color & 0xFF) * a
            den = max(asum, 1)
            out[y, x, 0] = divide_rounded(rsum, den)
            out[y, x, 1] = divide_rounded(gsum, den)
            out[y, x, 2] = divide_rounded(bsum, den)
            out[y, x, 3] = divide_rounded(asum, count)


@njit(cache=True)
def argb_to_rgba8(colors, out):
    """Write an (H*W) ARGB int64 buffer into an (H, W, 4) uint8 RGBA image."""
    h, w = out.shape[0], out.shape[1]
    for y in range(h):
        for x in range(w):
            c = colors[y * w + x]
            out[y, x, 0] = (c >> 16) & 0xFF
            out[y, x, 1] = (c >> 8) & 0xFF
            out[y, x, 2] = c & 0xFF
            out[y, x, 3] = (c >> 24) & 0xFF


def empty_ref():
    return np.zeros(2, np.float64), np.zeros(2, np.float64)
