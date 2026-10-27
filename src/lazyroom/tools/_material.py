"""Material You colour maths: a Python port of Google's material-color-utilities (MCU, Apache-2.0)
as shipped by MaterialKolor 5.0.1 (the library ImageToolbox uses), plus ImageToolbox's own
``getColorScheme`` post-processing (lib/dynamic-theme/DynamicTheme.kt, Apache-2.0).

Ported pieces (MCU Java/Kotlin names):
  utils (ColorUtils / MathUtils), Cam16 + ViewingConditions.DEFAULT, HctSolver, Hct, Blend.cam16Ucs,
  TonalPalette (+ KeyColor), QuantizerMap, QuantizerWu, QuantizerWsmeans (PointProviderLab,
  java.util.Random(0x42688)), QuantizerCelebi, Score, DislikeAnalyzer, TemperatureCache, Contrast,
  ContrastCurve, ToneDeltaPair, DynamicColor, MaterialDynamicColors (2021 spec as in the 2024/2025
  releases: tone-30 "on container" roles), DynamicScheme and the nine scheme variants.

All colour maths uses Python floats (IEEE doubles, like the JVM); colours are ARGB ints.
Known deviation: MCU Java/Kotlin compute some Wu-quantizer sums in 32-bit ints that can overflow
for large boxes; this port uses exact integers (the TypeScript/Dart behaviour).
"""
from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np

# =============================================================================
# MathUtils / ColorUtils
# =============================================================================


def _signum(x: float) -> int:
    return -1 if x < 0 else (0 if x == 0 else 1)


def _jround(x: float) -> int:
    """java.lang.Math.round(double)."""
    return int(math.floor(x + 0.5))


def _lerp(start: float, stop: float, amount: float) -> float:
    return (1.0 - amount) * start + amount * stop


def _clamp_int(lo: int, hi: int, v: int) -> int:
    return lo if v < lo else (hi if v > hi else v)


def _clamp_double(lo: float, hi: float, v: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def sanitize_degrees_int(d: int) -> int:
    d = int(math.fmod(d, 360))
    return d + 360 if d < 0 else d


def sanitize_degrees(d: float) -> float:
    d = math.fmod(d, 360.0)
    return d + 360.0 if d < 0 else d


def rotation_direction(frm: float, to: float) -> float:
    return 1.0 if sanitize_degrees(to - frm) <= 180.0 else -1.0


def difference_degrees(a: float, b: float) -> float:
    return 180.0 - abs(abs(a - b) - 180.0)


def _mat_mul(row: tuple, m: tuple) -> tuple:
    return (row[0] * m[0][0] + row[1] * m[0][1] + row[2] * m[0][2],
            row[0] * m[1][0] + row[1] * m[1][1] + row[2] * m[1][2],
            row[0] * m[2][0] + row[1] * m[2][1] + row[2] * m[2][2])


SRGB_TO_XYZ = ((0.41233895, 0.35762064, 0.18051042),
               (0.2126, 0.7152, 0.0722),
               (0.01932141, 0.11916382, 0.95034478))
XYZ_TO_SRGB = ((3.2413774792388685, -1.5376652402851851, -0.49885366846268053),
               (-0.9691452513005321, 1.8758853451067872, 0.04156585616912061),
               (0.05562093689691305, -0.20395524564742123, 1.0571799111220335))
WHITE_POINT_D65 = (95.047, 100.0, 108.883)


def argb_from_rgb(r: int, g: int, b: int) -> int:
    return (255 << 24) | ((r & 255) << 16) | ((g & 255) << 8) | (b & 255)


def alpha_from_argb(argb: int) -> int:
    return (argb >> 24) & 255


def red_from_argb(argb: int) -> int:
    return (argb >> 16) & 255


def green_from_argb(argb: int) -> int:
    return (argb >> 8) & 255


def blue_from_argb(argb: int) -> int:
    return argb & 255


def linearized(c: int | float) -> float:
    n = c / 255.0
    if n <= 0.040449936:
        return n / 12.92 * 100.0
    return ((n + 0.055) / 1.055) ** 2.4 * 100.0


def delinearized(c: float) -> int:
    n = c / 100.0
    d = n * 12.92 if n <= 0.0031308 else 1.055 * n ** (1.0 / 2.4) - 0.055
    return _clamp_int(0, 255, _jround(d * 255.0))


def argb_from_linrgb(lin: tuple) -> int:
    return argb_from_rgb(delinearized(lin[0]), delinearized(lin[1]), delinearized(lin[2]))


def argb_from_xyz(x: float, y: float, z: float) -> int:
    m = XYZ_TO_SRGB
    lr = m[0][0] * x + m[0][1] * y + m[0][2] * z
    lg = m[1][0] * x + m[1][1] * y + m[1][2] * z
    lb = m[2][0] * x + m[2][1] * y + m[2][2] * z
    return argb_from_rgb(delinearized(lr), delinearized(lg), delinearized(lb))


def xyz_from_argb(argb: int) -> tuple:
    return _mat_mul((linearized(red_from_argb(argb)), linearized(green_from_argb(argb)),
                     linearized(blue_from_argb(argb))), SRGB_TO_XYZ)


def _lab_f(t: float) -> float:
    e = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    return t ** (1.0 / 3.0) if t > e else (kappa * t + 16) / 116


def _lab_invf(ft: float) -> float:
    e = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    ft3 = ft * ft * ft
    return ft3 if ft3 > e else (116 * ft - 16) / kappa


def argb_from_lab(l: float, a: float, b: float) -> int:
    wp = WHITE_POINT_D65
    fy = (l + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    return argb_from_xyz(_lab_invf(fx) * wp[0], _lab_invf(fy) * wp[1], _lab_invf(fz) * wp[2])


def lab_from_argb(argb: int) -> tuple:
    lr, lg, lb = linearized(red_from_argb(argb)), linearized(green_from_argb(argb)), linearized(blue_from_argb(argb))
    m = SRGB_TO_XYZ
    x = m[0][0] * lr + m[0][1] * lg + m[0][2] * lb
    y = m[1][0] * lr + m[1][1] * lg + m[1][2] * lb
    z = m[2][0] * lr + m[2][1] * lg + m[2][2] * lb
    wp = WHITE_POINT_D65
    fx, fy, fz = _lab_f(x / wp[0]), _lab_f(y / wp[1]), _lab_f(z / wp[2])
    return (116.0 * fy - 16, 500.0 * (fx - fy), 200.0 * (fy - fz))


def y_from_lstar(lstar: float) -> float:
    return 100.0 * _lab_invf((lstar + 16.0) / 116.0)


def lstar_from_y(y: float) -> float:
    return _lab_f(y / 100.0) * 116.0 - 16.0


def lstar_from_argb(argb: int) -> float:
    return 116.0 * _lab_f(xyz_from_argb(argb)[1] / 100.0) - 16.0


def argb_from_lstar(lstar: float) -> int:
    c = delinearized(y_from_lstar(lstar))
    return argb_from_rgb(c, c, c)


def argb_to_hex(argb: int, alpha: bool = False) -> str:
    """'#RRGGBB' (or '#AARRGGBB' with ``alpha``), upper case."""
    return f"#{argb & 0xFFFFFFFF:08X}" if alpha else f"#{argb & 0xFFFFFF:06X}"


def hex_to_argb(s: str) -> int:
    h = s.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) == 6:
        return 0xFF000000 | int(h, 16)
    if len(h) == 8:  # #RRGGBBAA (LazyRoom convention) -> ARGB
        v = int(h, 16)
        return ((v & 0xFF) << 24) | (v >> 8)
    raise ValueError(f"bad hex colour {s!r}")


# =============================================================================
# ViewingConditions + Cam16
# =============================================================================

XYZ_TO_CAM16RGB = ((0.401288, 0.650173, -0.051461),
                   (-0.250268, 1.204414, 0.045854),
                   (-0.002079, 0.048952, 0.953127))
CAM16RGB_TO_XYZ = ((1.8620678, -1.0112547, 0.14918678),
                   (0.38752654, 0.62144744, -0.00897398),
                   (-0.01584150, -0.03412294, 1.0499644))


class ViewingConditions:
    __slots__ = ("n", "aw", "nbb", "ncb", "c", "nc", "rgb_d", "fl", "fl_root", "z")

    def __init__(self, n, aw, nbb, ncb, c, nc, rgb_d, fl, fl_root, z):
        self.n, self.aw, self.nbb, self.ncb, self.c, self.nc = n, aw, nbb, ncb, c, nc
        self.rgb_d, self.fl, self.fl_root, self.z = rgb_d, fl, fl_root, z

    @staticmethod
    def make(white_point, adapting_luminance, background_lstar, surround, discounting_illuminant):
        background_lstar = max(0.1, background_lstar)
        m = XYZ_TO_CAM16RGB
        xyz = white_point
        rw = xyz[0] * m[0][0] + xyz[1] * m[0][1] + xyz[2] * m[0][2]
        gw = xyz[0] * m[1][0] + xyz[1] * m[1][1] + xyz[2] * m[1][2]
        bw = xyz[0] * m[2][0] + xyz[1] * m[2][1] + xyz[2] * m[2][2]
        f = 0.8 + surround / 10.0
        c = _lerp(0.59, 0.69, (f - 0.9) * 10.0) if f >= 0.9 else _lerp(0.525, 0.59, (f - 0.8) * 10.0)
        d = 1.0 if discounting_illuminant else f * (1.0 - (1.0 / 3.6) * math.exp((-adapting_luminance - 42.0) / 92.0))
        d = _clamp_double(0.0, 1.0, d)
        nc = f
        rgb_d = (d * (100.0 / rw) + 1.0 - d, d * (100.0 / gw) + 1.0 - d, d * (100.0 / bw) + 1.0 - d)
        k = 1.0 / (5.0 * adapting_luminance + 1.0)
        k4 = k * k * k * k
        k4f = 1.0 - k4
        fl = k4 * adapting_luminance + 0.1 * k4f * k4f * math.cbrt(5.0 * adapting_luminance)
        n = y_from_lstar(background_lstar) / white_point[1]
        z = 1.48 + math.sqrt(n)
        nbb = 0.725 / n ** 0.2
        ncb = nbb
        fs = [(fl * rgb_d[0] * rw / 100.0) ** 0.42, (fl * rgb_d[1] * gw / 100.0) ** 0.42,
              (fl * rgb_d[2] * bw / 100.0) ** 0.42]
        rgb_a = [400.0 * v / (v + 27.13) for v in fs]
        aw = (2.0 * rgb_a[0] + rgb_a[1] + 0.05 * rgb_a[2]) * nbb
        return ViewingConditions(n, aw, nbb, ncb, c, nc, rgb_d, fl, fl ** 0.25, z)


VC = ViewingConditions.make(WHITE_POINT_D65, 200.0 / math.pi * y_from_lstar(50.0) / 100.0, 50.0, 2.0, False)


class Cam16:
    __slots__ = ("hue", "chroma", "j", "q", "m", "s", "jstar", "astar", "bstar")

    def __init__(self, hue, chroma, j, q, m, s, jstar, astar, bstar):
        self.hue, self.chroma, self.j, self.q, self.m, self.s = hue, chroma, j, q, m, s
        self.jstar, self.astar, self.bstar = jstar, astar, bstar

    def distance(self, other: "Cam16") -> float:
        dj, da, db = self.jstar - other.jstar, self.astar - other.astar, self.bstar - other.bstar
        return 1.41 * math.sqrt(dj * dj + da * da + db * db) ** 0.63

    @staticmethod
    def from_int(argb: int) -> "Cam16":
        rl, gl, bl = linearized(red_from_argb(argb)), linearized(green_from_argb(argb)), linearized(blue_from_argb(argb))
        x = 0.41233895 * rl + 0.35762064 * gl + 0.18051042 * bl
        y = 0.2126 * rl + 0.7152 * gl + 0.0722 * bl
        z = 0.01932141 * rl + 0.11916382 * gl + 0.95034478 * bl
        return Cam16.from_xyz(x, y, z)

    @staticmethod
    def from_xyz(x: float, y: float, z: float, vc: ViewingConditions = VC) -> "Cam16":
        m = XYZ_TO_CAM16RGB
        rt = x * m[0][0] + y * m[0][1] + z * m[0][2]
        gt = x * m[1][0] + y * m[1][1] + z * m[1][2]
        bt = x * m[2][0] + y * m[2][1] + z * m[2][2]
        rd, gd, bd = vc.rgb_d[0] * rt, vc.rgb_d[1] * gt, vc.rgb_d[2] * bt
        raf = (vc.fl * abs(rd) / 100.0) ** 0.42
        gaf = (vc.fl * abs(gd) / 100.0) ** 0.42
        baf = (vc.fl * abs(bd) / 100.0) ** 0.42
        ra = _signum(rd) * 400.0 * raf / (raf + 27.13)
        ga = _signum(gd) * 400.0 * gaf / (gaf + 27.13)
        ba = _signum(bd) * 400.0 * baf / (baf + 27.13)
        a = (11.0 * ra + -12.0 * ga + ba) / 11.0
        b = (ra + ga - 2.0 * ba) / 9.0
        u = (20.0 * ra + 20.0 * ga + 21.0 * ba) / 20.0
        p2 = (40.0 * ra + 20.0 * ga + ba) / 20.0
        atan_deg = math.degrees(math.atan2(b, a))
        hue = atan_deg + 360.0 if atan_deg < 0 else (atan_deg - 360.0 if atan_deg >= 360 else atan_deg)
        hue_rad = math.radians(hue)
        ac = p2 * vc.nbb
        j = 100.0 * (ac / vc.aw) ** (vc.c * vc.z)
        q = 4.0 / vc.c * math.sqrt(j / 100.0) * (vc.aw + 4.0) * vc.fl_root
        hue_prime = hue + 360 if hue < 20.14 else hue
        e_hue = 0.25 * (math.cos(math.radians(hue_prime) + 2.0) + 3.8)
        p1 = 50000.0 / 13.0 * e_hue * vc.nc * vc.ncb
        t = p1 * math.hypot(a, b) / (u + 0.305)
        alpha = (1.64 - 0.29 ** vc.n) ** 0.73 * t ** 0.9
        c = alpha * math.sqrt(j / 100.0)
        mm = c * vc.fl_root
        s = 50.0 * math.sqrt((alpha * vc.c) / (vc.aw + 4.0))
        jstar = (1.0 + 100.0 * 0.007) * j / (1.0 + 0.007 * j)
        mstar = 1.0 / 0.0228 * math.log1p(0.0228 * mm)
        return Cam16(hue, c, j, q, mm, s, jstar, mstar * math.cos(hue_rad), mstar * math.sin(hue_rad))

    @staticmethod
    def from_jch(j: float, c: float, h: float, vc: ViewingConditions = VC) -> "Cam16":
        q = 4.0 / vc.c * math.sqrt(j / 100.0) * (vc.aw + 4.0) * vc.fl_root
        m = c * vc.fl_root
        alpha = c / math.sqrt(j / 100.0)
        s = 50.0 * math.sqrt((alpha * vc.c) / (vc.aw + 4.0))
        hr = math.radians(h)
        jstar = (1.0 + 100.0 * 0.007) * j / (1.0 + 0.007 * j)
        mstar = 1.0 / 0.0228 * math.log1p(0.0228 * m)
        return Cam16(h, c, j, q, m, s, jstar, mstar * math.cos(hr), mstar * math.sin(hr))

    @staticmethod
    def from_ucs(jstar: float, astar: float, bstar: float, vc: ViewingConditions = VC) -> "Cam16":
        m = math.hypot(astar, bstar)
        m2 = math.expm1(m * 0.0228) / 0.0228
        c = m2 / vc.fl_root
        h = math.atan2(bstar, astar) * (180.0 / math.pi)
        if h < 0.0:
            h += 360.0
        j = jstar / (1.0 - (jstar - 100.0) * 0.007)
        return Cam16.from_jch(j, c, h, vc)

    def xyz(self, vc: ViewingConditions = VC) -> tuple:
        alpha = 0.0 if (self.chroma == 0.0 or self.j == 0.0) else self.chroma / math.sqrt(self.j / 100.0)
        t = (alpha / (1.64 - 0.29 ** vc.n) ** 0.73) ** (1.0 / 0.9)
        hr = math.radians(self.hue)
        e_hue = 0.25 * (math.cos(hr + 2.0) + 3.8)
        ac = vc.aw * (self.j / 100.0) ** (1.0 / vc.c / vc.z)
        p1 = e_hue * (50000.0 / 13.0) * vc.nc * vc.ncb
        p2 = ac / vc.nbb
        hs, hc = math.sin(hr), math.cos(hr)
        gamma = 23.0 * (p2 + 0.305) * t / (23.0 * p1 + 11.0 * t * hc + 108.0 * t * hs)
        a, b = gamma * hc, gamma * hs
        ra = (460.0 * p2 + 451.0 * a + 288.0 * b) / 1403.0
        ga = (460.0 * p2 - 891.0 * a - 261.0 * b) / 1403.0
        ba = (460.0 * p2 - 220.0 * a - 6300.0 * b) / 1403.0
        out = []
        for comp in (ra, ga, ba):
            base = max(0.0, (27.13 * abs(comp)) / (400.0 - abs(comp)))
            out.append(_signum(comp) * (100.0 / vc.fl) * base ** (1.0 / 0.42))
        rf, gf, bf = out[0] / vc.rgb_d[0], out[1] / vc.rgb_d[1], out[2] / vc.rgb_d[2]
        m = CAM16RGB_TO_XYZ
        return (rf * m[0][0] + gf * m[0][1] + bf * m[0][2],
                rf * m[1][0] + gf * m[1][1] + bf * m[1][2],
                rf * m[2][0] + gf * m[2][1] + bf * m[2][2])

    def to_int(self) -> int:
        return argb_from_xyz(*self.xyz())


# =============================================================================
# HctSolver
# =============================================================================

_SCALED_DISCOUNT_FROM_LINRGB = ((0.001200833568784504, 0.002389694492170889, 0.0002795742885861124),
                                (0.0005891086651375999, 0.0029785502573438758, 0.0003270666104008398),
                                (0.00010146692491640572, 0.0005364214359186694, 0.0032979401770712076))
_LINRGB_FROM_SCALED_DISCOUNT = ((1373.2198709594231, -1100.4251190754821, -7.278681089101213),
                                (-271.815969077903, 559.6580465940733, -32.46047482791194),
                                (1.9622899599665666, -57.173814538844006, 308.7233197812385))
_Y_FROM_LINRGB = (0.2126, 0.7152, 0.0722)
# MCU hard-codes these 255 values; they are the linearised mid-points between 8-bit sRGB codes.
_CRITICAL_PLANES = tuple(linearized(i + 0.5) for i in range(255))


def _sanitize_radians(a: float) -> float:
    return math.fmod(a + math.pi * 8, math.pi * 2)


def _true_delinearized(c: float) -> float:
    n = c / 100.0
    d = n * 12.92 if n <= 0.0031308 else 1.055 * n ** (1.0 / 2.4) - 0.055
    return d * 255.0


def _chromatic_adaptation(c: float) -> float:
    af = abs(c) ** 0.42
    return _signum(c) * 400.0 * af / (af + 27.13)


def _hue_of(lin: tuple) -> float:
    sd = _mat_mul(lin, _SCALED_DISCOUNT_FROM_LINRGB)
    ra, ga, ba = _chromatic_adaptation(sd[0]), _chromatic_adaptation(sd[1]), _chromatic_adaptation(sd[2])
    a = (11.0 * ra + -12.0 * ga + ba) / 11.0
    b = (ra + ga - 2.0 * ba) / 9.0
    return math.atan2(b, a)


def _cyclic(a: float, b: float, c: float) -> bool:
    return _sanitize_radians(b - a) < _sanitize_radians(c - a)


def _set_coordinate(src: tuple, coord: float, tgt: tuple, axis: int) -> tuple:
    t = (coord - src[axis]) / (tgt[axis] - src[axis])
    return (src[0] + (tgt[0] - src[0]) * t, src[1] + (tgt[1] - src[1]) * t, src[2] + (tgt[2] - src[2]) * t)


def _nth_vertex(y: float, n: int) -> tuple:
    kr, kg, kb = _Y_FROM_LINRGB
    ca = 0.0 if n % 4 <= 1 else 100.0
    cb = 0.0 if n % 2 == 0 else 100.0
    if n < 4:
        g, b = ca, cb
        r = (y - g * kg - b * kb) / kr
        return (r, g, b) if 0.0 <= r <= 100.0 else (-1.0, -1.0, -1.0)
    if n < 8:
        b, r = ca, cb
        g = (y - r * kr - b * kb) / kg
        return (r, g, b) if 0.0 <= g <= 100.0 else (-1.0, -1.0, -1.0)
    r, g = ca, cb
    b = (y - r * kr - g * kg) / kb
    return (r, g, b) if 0.0 <= b <= 100.0 else (-1.0, -1.0, -1.0)


def _bisect_to_segment(y: float, target_hue: float) -> tuple:
    left = right = (-1.0, -1.0, -1.0)
    left_hue = right_hue = 0.0
    initialized = False
    uncut = True
    for n in range(12):
        mid = _nth_vertex(y, n)
        if mid[0] < 0:
            continue
        mid_hue = _hue_of(mid)
        if not initialized:
            left = right = mid
            left_hue = right_hue = mid_hue
            initialized = True
            continue
        if uncut or _cyclic(left_hue, mid_hue, right_hue):
            uncut = False
            if _cyclic(left_hue, target_hue, mid_hue):
                right, right_hue = mid, mid_hue
            else:
                left, left_hue = mid, mid_hue
    return left, right


def _bisect_to_limit(y: float, target_hue: float) -> tuple:
    left, right = _bisect_to_segment(y, target_hue)
    left_hue = _hue_of(left)
    for axis in range(3):
        if left[axis] != right[axis]:
            if left[axis] < right[axis]:
                l_plane = math.floor(_true_delinearized(left[axis]) - 0.5)
                r_plane = math.ceil(_true_delinearized(right[axis]) - 0.5)
            else:
                l_plane = math.ceil(_true_delinearized(left[axis]) - 0.5)
                r_plane = math.floor(_true_delinearized(right[axis]) - 0.5)
            for _ in range(8):
                if abs(r_plane - l_plane) <= 1:
                    break
                m_plane = math.floor((l_plane + r_plane) / 2.0)
                mid = _set_coordinate(left, _CRITICAL_PLANES[m_plane], right, axis)
                mid_hue = _hue_of(mid)
                if _cyclic(left_hue, target_hue, mid_hue):
                    right, r_plane = mid, m_plane
                else:
                    left, left_hue, l_plane = mid, mid_hue, m_plane
    return ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2, (left[2] + right[2]) / 2)


def _inverse_chromatic_adaptation(adapted: float) -> float:
    aa = abs(adapted)
    base = max(0.0, 27.13 * aa / (400.0 - aa))
    return _signum(adapted) * base ** (1.0 / 0.42)


_T_INNER = 1 / (1.64 - 0.29 ** VC.n) ** 0.73


def _find_result_by_j(hue_rad: float, chroma: float, y: float) -> int:
    j = math.sqrt(y) * 11.0
    e_hue = 0.25 * (math.cos(hue_rad + 2.0) + 3.8)
    p1 = e_hue * (50000.0 / 13.0) * VC.nc * VC.ncb
    hs, hc = math.sin(hue_rad), math.cos(hue_rad)
    kr, kg, kb = _Y_FROM_LINRGB
    for it in range(5):
        jn = j / 100.0
        alpha = 0.0 if (chroma == 0.0 or j == 0.0) else chroma / math.sqrt(jn)
        t = (alpha * _T_INNER) ** (1.0 / 0.9)
        ac = VC.aw * jn ** (1.0 / VC.c / VC.z)
        p2 = ac / VC.nbb
        gamma = 23.0 * (p2 + 0.305) * t / (23.0 * p1 + 11 * t * hc + 108.0 * t * hs)
        a, b = gamma * hc, gamma * hs
        ra = (460.0 * p2 + 451.0 * a + 288.0 * b) / 1403.0
        ga = (460.0 * p2 - 891.0 * a - 261.0 * b) / 1403.0
        ba = (460.0 * p2 - 220.0 * a - 6300.0 * b) / 1403.0
        lin = _mat_mul((_inverse_chromatic_adaptation(ra), _inverse_chromatic_adaptation(ga),
                        _inverse_chromatic_adaptation(ba)), _LINRGB_FROM_SCALED_DISCOUNT)
        if lin[0] < 0 or lin[1] < 0 or lin[2] < 0:
            return 0
        fnj = kr * lin[0] + kg * lin[1] + kb * lin[2]
        if fnj <= 0:
            return 0
        if it == 4 or abs(fnj - y) < 0.002:
            if lin[0] > 100.01 or lin[1] > 100.01 or lin[2] > 100.01:
                return 0
            return argb_from_linrgb(lin)
        j = j - (fnj - y) * j / (2 * fnj)
    return 0


def solve_to_int(hue: float, chroma: float, lstar: float) -> int:
    if chroma < 0.0001 or lstar < 0.0001 or lstar > 99.9999:
        return argb_from_lstar(lstar)
    hue = sanitize_degrees(hue)
    hr = math.radians(hue)
    y = y_from_lstar(lstar)
    exact = _find_result_by_j(hr, chroma, y)
    if exact != 0:
        return exact
    return argb_from_linrgb(_bisect_to_limit(y, hr))


# =============================================================================
# Hct
# =============================================================================

class Hct:
    __slots__ = ("hue", "chroma", "tone", "argb")

    def __init__(self, argb: int):
        self.argb = argb & 0xFFFFFFFF
        cam = Cam16.from_int(self.argb)
        self.hue = cam.hue
        self.chroma = cam.chroma
        self.tone = lstar_from_argb(self.argb)

    @staticmethod
    def from_hct(hue: float, chroma: float, tone: float) -> "Hct":
        return Hct(solve_to_int(hue, chroma, tone))

    @staticmethod
    def from_int(argb: int) -> "Hct":
        return Hct(argb)

    def to_int(self) -> int:
        return self.argb

    @staticmethod
    def is_yellow(hue: float) -> bool:
        return 105.0 <= hue < 125.0

    def __repr__(self) -> str:
        return f"Hct({argb_to_hex(self.argb)}, h={self.hue:.3f}, c={self.chroma:.3f}, t={self.tone:.3f})"


def hct_from_argb(argb: int) -> Hct:
    return Hct(argb)


def blend_cam16_ucs(frm: int, to: int, amount: float) -> int:
    """MCU ``Blend.cam16Ucs`` (MaterialKolor ``Color.blend``)."""
    a, b = Cam16.from_int(frm), Cam16.from_int(to)
    j = a.jstar + (b.jstar - a.jstar) * amount
    aa = a.astar + (b.astar - a.astar) * amount
    bb = a.bstar + (b.bstar - a.bstar) * amount
    return Cam16.from_ucs(j, aa, bb).to_int()


def blend_harmonize(design: int, source: int) -> int:
    """MCU ``Blend.harmonize``."""
    fh, th = Hct(design), Hct(source)
    diff = difference_degrees(fh.hue, th.hue)
    rot = min(diff * 0.5, 15.0)
    out_hue = sanitize_degrees(fh.hue + rot * rotation_direction(fh.hue, th.hue))
    return Hct.from_hct(out_hue, fh.chroma, fh.tone).to_int()


# =============================================================================
# TonalPalette
# =============================================================================

class _KeyColor:
    MAX_CHROMA = 200.0

    def __init__(self, hue: float, chroma: float):
        self.hue, self.chroma = hue, chroma
        self._cache: dict[int, float] = {}

    def _max_chroma(self, tone: int) -> float:
        if tone not in self._cache:
            self._cache[tone] = Hct.from_hct(self.hue, self.MAX_CHROMA, tone).chroma
        return self._cache[tone]

    def create(self) -> Hct:
        pivot, step, eps = 50, 1, 0.01
        lo, hi = 0, 100
        while lo < hi:
            mid = (lo + hi) // 2
            ascending = self._max_chroma(mid) < self._max_chroma(mid + step)
            sufficient = self._max_chroma(mid) >= self.chroma - eps
            if sufficient:
                if abs(lo - pivot) < abs(hi - pivot):
                    hi = mid
                else:
                    if lo == mid:
                        return Hct.from_hct(self.hue, self.chroma, lo)
                    lo = mid
            else:
                if ascending:
                    lo = mid + step
                else:
                    hi = mid
        return Hct.from_hct(self.hue, self.chroma, lo)


class TonalPalette:
    def __init__(self, hue: float, chroma: float, key_color: Hct | None = None):
        self.hue = hue
        self.chroma = chroma
        self._key = key_color
        self._cache: dict[int, int] = {}

    @staticmethod
    def from_int(argb: int) -> "TonalPalette":
        return TonalPalette.from_hct(Hct(argb))

    @staticmethod
    def from_hct(hct: Hct) -> "TonalPalette":
        return TonalPalette(hct.hue, hct.chroma, hct)

    @staticmethod
    def from_hue_and_chroma(hue: float, chroma: float) -> "TonalPalette":
        return TonalPalette(hue, chroma, None)

    @property
    def key_color(self) -> Hct:
        if self._key is None:  # computed lazily (only key-colour roles need it)
            self._key = _KeyColor(self.hue, self.chroma).create()
        return self._key

    def tone(self, tone: int) -> int:
        tone = int(tone)
        c = self._cache.get(tone)
        if c is None:
            if tone == 99 and Hct.is_yellow(self.hue):
                c = _average_argb(self.tone(98), self.tone(100))
            else:
                c = Hct.from_hct(self.hue, self.chroma, tone).to_int()
            self._cache[tone] = c
        return c

    def get_hct(self, tone: float) -> Hct:
        return Hct.from_hct(self.hue, self.chroma, tone)


def _average_argb(a: int, b: int) -> int:
    def rnd(x, y):  # Math.round((x + y) / 2f)
        return int(math.floor((x + y) / 2.0 + 0.5))
    return (255 << 24) | (rnd(red_from_argb(a), red_from_argb(b)) << 16) | \
        (rnd(green_from_argb(a), green_from_argb(b)) << 8) | rnd(blue_from_argb(a), blue_from_argb(b))


# =============================================================================
# Quantizers
# =============================================================================

def pixels_to_argb(img_u8: np.ndarray) -> np.ndarray:
    """(H, W, 3|4) uint8 -> flat int64 ARGB array (alpha 255 when RGB)."""
    a = img_u8.reshape(-1, img_u8.shape[-1]).astype(np.int64)
    alpha = a[:, 3] if a.shape[1] == 4 else np.full(len(a), 255, np.int64)
    return (alpha << 24) | (a[:, 0] << 16) | (a[:, 1] << 8) | a[:, 2]


def quantizer_map(pixels: np.ndarray) -> dict[int, int]:
    """QuantizerMap: colour -> count, in first-occurrence (LinkedHashMap) order."""
    uniq, first, counts = np.unique(pixels, return_index=True, return_counts=True)
    order = np.argsort(first, kind="stable")
    return {int(uniq[i]): int(counts[i]) for i in order}


_WU_BITS = 5
_WU_N = 33


class _Box:
    __slots__ = ("r0", "r1", "g0", "g1", "b0", "b1", "vol")

    def __init__(self):
        self.r0 = self.r1 = self.g0 = self.g1 = self.b0 = self.b1 = self.vol = 0


class QuantizerWu:
    def quantize(self, pixels: np.ndarray, color_count: int) -> list[int]:
        cmap = quantizer_map(pixels)
        self._histogram(cmap)
        boxes_count = self._create_boxes(color_count)
        return self._create_result(boxes_count)

    def _histogram(self, cmap: dict[int, int]) -> None:
        keys = np.fromiter(cmap.keys(), np.int64, len(cmap))
        cnts = np.fromiter(cmap.values(), np.int64, len(cmap))
        r, g, b = (keys >> 16) & 255, (keys >> 8) & 255, keys & 255
        sh = 8 - _WU_BITS
        idx = (((r >> sh) + 1) * _WU_N + ((g >> sh) + 1)) * _WU_N + ((b >> sh) + 1)
        size = _WU_N ** 3
        w = np.bincount(idx, cnts, size)
        mr = np.bincount(idx, r * cnts, size)
        mg = np.bincount(idx, g * cnts, size)
        mb = np.bincount(idx, b * cnts, size)
        m2 = np.bincount(idx, (r * r + g * g + b * b) * cnts, size)

        def cum(a, dtype):
            a = np.rint(a).astype(dtype).reshape(_WU_N, _WU_N, _WU_N)
            return a.cumsum(0).cumsum(1).cumsum(2).ravel().tolist()

        # createMoments(): 3-D prefix sums (exact integers; see module docstring)
        self.w, self.mr, self.mg, self.mb = (cum(x, np.int64) for x in (w, mr, mg, mb))
        self.m2 = [float(v) for v in cum(m2, np.int64)]

    @staticmethod
    def _i(r, g, b):
        return (r * _WU_N + g) * _WU_N + b

    def _vol(self, c: _Box, m: list) -> float:
        i = self._i
        return (m[i(c.r1, c.g1, c.b1)] - m[i(c.r1, c.g1, c.b0)] - m[i(c.r1, c.g0, c.b1)] + m[i(c.r1, c.g0, c.b0)]
                - m[i(c.r0, c.g1, c.b1)] + m[i(c.r0, c.g1, c.b0)] + m[i(c.r0, c.g0, c.b1)] - m[i(c.r0, c.g0, c.b0)])

    def _bottom(self, c: _Box, d: str, m: list):
        i = self._i
        if d == "r":
            return -m[i(c.r0, c.g1, c.b1)] + m[i(c.r0, c.g1, c.b0)] + m[i(c.r0, c.g0, c.b1)] - m[i(c.r0, c.g0, c.b0)]
        if d == "g":
            return -m[i(c.r1, c.g0, c.b1)] + m[i(c.r1, c.g0, c.b0)] + m[i(c.r0, c.g0, c.b1)] - m[i(c.r0, c.g0, c.b0)]
        return -m[i(c.r1, c.g1, c.b0)] + m[i(c.r1, c.g0, c.b0)] + m[i(c.r0, c.g1, c.b0)] - m[i(c.r0, c.g0, c.b0)]

    def _top(self, c: _Box, d: str, p: int, m: list):
        i = self._i
        if d == "r":
            return m[i(p, c.g1, c.b1)] - m[i(p, c.g1, c.b0)] - m[i(p, c.g0, c.b1)] + m[i(p, c.g0, c.b0)]
        if d == "g":
            return m[i(c.r1, p, c.b1)] - m[i(c.r1, p, c.b0)] - m[i(c.r0, p, c.b1)] + m[i(c.r0, p, c.b0)]
        return m[i(c.r1, c.g1, p)] - m[i(c.r1, c.g0, p)] - m[i(c.r0, c.g1, p)] + m[i(c.r0, c.g0, p)]

    def _variance(self, c: _Box) -> float:
        dr, dg, db = self._vol(c, self.mr), self._vol(c, self.mg), self._vol(c, self.mb)
        xx = self._vol(c, self.m2)
        hyp = dr * dr + dg * dg + db * db
        return xx - hyp / float(self._vol(c, self.w))

    def _maximize(self, c, d, first, last, wr, wg, wb, ww):
        br, bg, bb, bw = (self._bottom(c, d, m) for m in (self.mr, self.mg, self.mb, self.w))
        mx, cut = 0.0, -1
        for i in range(first, last):
            hr = br + self._top(c, d, i, self.mr)
            hg = bg + self._top(c, d, i, self.mg)
            hb = bb + self._top(c, d, i, self.mb)
            hw = bw + self._top(c, d, i, self.w)
            if hw == 0:
                continue
            temp = float(hr * hr + hg * hg + hb * hb) / hw
            hr, hg, hb, hw = wr - hr, wg - hg, wb - hb, ww - hw
            if hw == 0:
                continue
            temp += float(hr * hr + hg * hg + hb * hb) / hw
            if temp > mx:
                mx, cut = temp, i
        return cut, mx

    def _cut(self, one: _Box, two: _Box) -> bool:
        wr, wg, wb, ww = (self._vol(one, m) for m in (self.mr, self.mg, self.mb, self.w))
        cr, mr = self._maximize(one, "r", one.r0 + 1, one.r1, wr, wg, wb, ww)
        cg, mg = self._maximize(one, "g", one.g0 + 1, one.g1, wr, wg, wb, ww)
        cb, mb = self._maximize(one, "b", one.b0 + 1, one.b1, wr, wg, wb, ww)
        if mr >= mg and mr >= mb:
            if cr < 0:
                return False
            d = "r"
        elif mg >= mr and mg >= mb:
            d = "g"
        else:
            d = "b"
        two.r1, two.g1, two.b1 = one.r1, one.g1, one.b1
        if d == "r":
            one.r1 = cr
            two.r0, two.g0, two.b0 = one.r1, one.g0, one.b0
        elif d == "g":
            one.g1 = cg
            two.r0, two.g0, two.b0 = one.r0, one.g1, one.b0
        else:
            one.b1 = cb
            two.r0, two.g0, two.b0 = one.r0, one.g0, one.b1
        one.vol = (one.r1 - one.r0) * (one.g1 - one.g0) * (one.b1 - one.b0)
        two.vol = (two.r1 - two.r0) * (two.g1 - two.g0) * (two.b1 - two.b0)
        return True

    def _create_boxes(self, max_colors: int) -> int:
        self.cubes = [_Box() for _ in range(max_colors)]
        vv = [0.0] * max_colors
        first = self.cubes[0]
        first.r1 = first.g1 = first.b1 = _WU_N - 1
        generated = max_colors
        nxt = 0
        i = 1
        while i < max_colors:
            if self._cut(self.cubes[nxt], self.cubes[i]):
                vv[nxt] = self._variance(self.cubes[nxt]) if self.cubes[nxt].vol > 1 else 0.0
                vv[i] = self._variance(self.cubes[i]) if self.cubes[i].vol > 1 else 0.0
            else:
                vv[nxt] = 0.0
                i -= 1
            nxt = 0
            temp = vv[0]
            for j in range(1, i + 1):
                if vv[j] > temp:
                    temp, nxt = vv[j], j
            if temp <= 0.0:
                generated = i + 1
                break
            i += 1
        return generated

    def _create_result(self, count: int) -> list[int]:
        out = []
        for i in range(count):
            c = self.cubes[i]
            w = self._vol(c, self.w)
            if w > 0:
                r, g, b = self._vol(c, self.mr) // w, self._vol(c, self.mg) // w, self._vol(c, self.mb) // w
                out.append((255 << 24) | ((r & 255) << 16) | ((g & 255) << 8) | (b & 255))
        return list(dict.fromkeys(out))  # LinkedHashMap keys


class JavaRandom:
    """java.util.Random (48-bit LCG) – QuantizerWsmeans seeds it with 0x42688."""

    def __init__(self, seed: int):
        self.seed = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def _next(self, bits: int) -> int:
        self.seed = (self.seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        v = self.seed >> (48 - bits)
        return v - (1 << 32) if v >= (1 << 31) else v

    def next_int(self, bound: int) -> int:
        r = self._next(31)
        m = bound - 1
        if bound & m == 0:
            return (bound * r) >> 31
        u = r
        while True:
            r = u % bound
            x = (u - r + m) & 0xFFFFFFFF
            if x < (1 << 31):  # int overflow check of `u - r + m < 0`
                return r
            u = self._next(31)


def _lab_points(argbs: np.ndarray) -> np.ndarray:
    """Vectorised ColorUtils.labFromArgb (same double maths as lab_from_argb)."""
    rgb = np.stack([(argbs >> 16) & 255, (argbs >> 8) & 255, argbs & 255], -1).astype(np.float64) / 255.0
    lin = np.where(rgb <= 0.040449936, rgb / 12.92 * 100.0, ((rgb + 0.055) / 1.055) ** 2.4 * 100.0)
    m = np.array(SRGB_TO_XYZ)
    xyz = (m[:, 0] * lin[:, :1] + m[:, 1] * lin[:, 1:2] + m[:, 2] * lin[:, 2:3]) / np.array(WHITE_POINT_D65)
    e, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > e, np.power(xyz, 1.0 / 3.0), (kappa * xyz + 16) / 116)
    return np.stack([116.0 * f[:, 1] - 16, 500.0 * (f[:, 0] - f[:, 1]), 200.0 * (f[:, 1] - f[:, 2])], -1)


def quantize_wsmeans(pixels: np.ndarray, starting: list[int], max_colors: int,
                     max_iterations: int = 10, min_movement: float = 3.0) -> dict[int, int]:
    """QuantizerWsmeans (weighted spherical k-means in L*a*b*).  The distance-matrix bookkeeping
    reproduces MCU's in-place sorted rows; the per-point loop is vectorised (same results)."""
    rnd = JavaRandom(0x42688)
    cmap = quantizer_map(pixels)
    px = np.fromiter(cmap.keys(), np.int64, len(cmap))
    counts = np.fromiter(cmap.values(), np.int64, len(cmap))
    points = _lab_points(px)
    n_points = len(px)
    k = min(max_colors, n_points)
    if starting:
        k = min(k, len(starting))
    clusters = [list(lab_from_argb(c)) for c in starting[:k]]
    while len(clusters) < k:  # MCU leaves additional clusters unset (never happens with Wu seeds)
        clusters.append([0.0, 0.0, 0.0])
    idx = np.array([rnd.next_int(k) for _ in range(n_points)], np.int64)
    # distanceToIndexMatrix rows: mutable [distance, index] objects, sorted in place each iteration
    rows = [[[-1.0, -1] for _ in range(k)] for _ in range(k)]
    sums = np.zeros(k, np.int64)
    for iteration in range(max_iterations):
        cl = np.array(clusters, np.float64)
        for i in range(k):
            for j in range(i + 1, k):
                d = (cl[i, 0] - cl[j, 0]) ** 2 + (cl[i, 1] - cl[j, 1]) ** 2 + (cl[i, 2] - cl[j, 2]) ** 2
                d = float(d)
                rows[j][i][0], rows[j][i][1] = d, i
                rows[i][j][0], rows[i][j][1] = d, j
            rows[i].sort(key=lambda e: e[0])  # Arrays.sort: stable
        row_d = np.array([[e[0] for e in r] for r in rows], np.float64)
        diff = points[:, None, :] - cl[None, :, :]
        dist = (diff * diff).sum(-1)
        prev_d = dist[np.arange(n_points), idx]
        allowed = row_d[idx] < 4 * prev_d[:, None]
        masked = np.where(allowed, dist, np.inf)
        best = masked.argmin(1)
        best_d = masked[np.arange(n_points), best]
        improve = best_d < prev_d
        moved = improve & (np.abs(np.sqrt(best_d) - np.sqrt(prev_d)) > min_movement)
        idx = np.where(moved, best, idx)
        if not moved.any() and iteration != 0:
            break
        sums = np.bincount(idx, counts, k).astype(np.int64)
        for c in range(3):
            comp = np.bincount(idx, points[:, c] * counts, k)
            for i in range(k):
                if sums[i]:
                    clusters[i][c] = comp[i] / sums[i]
        for i in range(k):
            if sums[i] == 0:
                clusters[i] = [0.0, 0.0, 0.0]
    out: dict[int, int] = {}
    for i in range(k):
        if sums[i] == 0:
            continue
        c = argb_from_lab(*clusters[i])
        if c not in out:
            out[c] = int(sums[i])
    return out


def quantize_celebi(pixels: np.ndarray, max_colors: int) -> dict[int, int]:
    """QuantizerCelebi: Wu seeds refined by WSMeans.  Returns {argb: population}."""
    wu = QuantizerWu().quantize(pixels, max_colors)
    return quantize_wsmeans(pixels, wu, max_colors)


# =============================================================================
# Score
# =============================================================================

def score(colors_to_population: dict[int, int], desired: int = 4, fallback: int = 0xFF4285F4,
          filter: bool = True, with_scores: bool = False) -> list:
    """MCU ``Score.score``: rank quantised colours for use as theme seeds."""
    hcts, hue_pop, total = [], [0] * 360, 0.0
    for argb, pop in colors_to_population.items():
        h = Hct(argb)
        hcts.append(h)
        hue_pop[int(math.floor(h.hue))] += pop
        total += pop
    excited = [0.0] * 360
    for hue in range(360):
        prop = hue_pop[hue] / total
        for i in range(hue - 14, hue + 16):
            excited[sanitize_degrees_int(i)] += prop
    scored = []
    for h in hcts:
        prop = excited[sanitize_degrees_int(_jround(h.hue))]
        if filter and (h.chroma < 5.0 or prop <= 0.01):
            continue
        s = prop * 100.0 * 0.7 + (h.chroma - 48.0) * (0.1 if h.chroma < 48.0 else 0.3)
        scored.append((h, s))
    scored.sort(key=lambda t: -t[1])  # stable, descending
    chosen: list = []
    for diff in range(90, 14, -1):
        chosen.clear()
        for h, s in scored:
            if not any(difference_degrees(h.hue, c[0].hue) < diff for c in chosen):
                chosen.append((h, s))
            if len(chosen) >= desired:
                break
        if len(chosen) >= desired:
            break
    if not chosen:
        return [(fallback, None)] if with_scores else [fallback]
    return [(h.to_int(), s) for h, s in chosen] if with_scores else [h.to_int() for h, _ in chosen]


# =============================================================================
# DislikeAnalyzer, TemperatureCache, Contrast
# =============================================================================

def is_disliked(h: Hct) -> bool:
    return 90.0 <= _jround(h.hue) <= 111.0 and _jround(h.chroma) > 16.0 and _jround(h.tone) < 65.0


def fix_if_disliked(h: Hct) -> Hct:
    return Hct.from_hct(h.hue, h.chroma, 70.0) if is_disliked(h) else h


class TemperatureCache:
    def __init__(self, input_hct: Hct):
        self.input = input_hct
        self._by_hue: list[Hct] | None = None
        self._by_temp: list[Hct] | None = None
        self._temps: dict[int, float] | None = None
        self._complement: Hct | None = None

    @staticmethod
    def raw_temperature(c: Hct) -> float:
        lab = lab_from_argb(c.to_int())
        hue = sanitize_degrees(math.degrees(math.atan2(lab[2], lab[1])))
        chroma = math.hypot(lab[1], lab[2])
        return -0.5 + 0.02 * chroma ** 1.07 * math.cos(math.radians(sanitize_degrees(hue - 50.0)))

    def hcts_by_hue(self) -> list[Hct]:
        if self._by_hue is None:
            self._by_hue = [Hct.from_hct(float(h), self.input.chroma, self.input.tone) for h in range(361)]
        return self._by_hue

    def temps(self) -> dict[int, float]:  # keyed by id(): Java's Hct has identity equality
        if self._temps is None:
            self._temps = {id(h): self.raw_temperature(h) for h in self.hcts_by_hue() + [self.input]}
        return self._temps

    def hcts_by_temp(self) -> list[Hct]:
        if self._by_temp is None:
            t = self.temps()
            self._by_temp = sorted(self.hcts_by_hue() + [self.input], key=lambda h: t[id(h)])
        return self._by_temp

    def coldest(self) -> Hct:
        return self.hcts_by_temp()[0]

    def warmest(self) -> Hct:
        return self.hcts_by_temp()[-1]

    def relative_temperature(self, h: Hct) -> float:
        t = self.temps()
        rng = t[id(self.warmest())] - t[id(self.coldest())]
        diff = t[id(h)] - t[id(self.coldest())]
        return 0.5 if rng == 0.0 else diff / rng

    @staticmethod
    def _between(angle: float, a: float, b: float) -> bool:
        return a <= angle <= b if a < b else (a <= angle or angle <= b)

    def complement(self) -> Hct:
        if self._complement is not None:
            return self._complement
        t = self.temps()
        coldest_hue, coldest_t = self.coldest().hue, t[id(self.coldest())]
        warmest_hue, warmest_t = self.warmest().hue, t[id(self.warmest())]
        rng = warmest_t - coldest_t
        start_cw = self._between(self.input.hue, coldest_hue, warmest_hue)
        start = warmest_hue if start_cw else coldest_hue
        end = coldest_hue if start_cw else warmest_hue
        smallest = 1000.0
        answer = self.hcts_by_hue()[_jround(self.input.hue)]
        comp_rel = 1.0 - self.relative_temperature(self.input)
        addend = 0.0
        while addend <= 360.0:
            hue = sanitize_degrees(start + addend)
            if self._between(hue, start, end):
                cand = self.hcts_by_hue()[_jround(hue)]
                rel = (t[id(cand)] - coldest_t) / rng
                err = abs(comp_rel - rel)
                if err < smallest:
                    smallest, answer = err, cand
            addend += 1.0
        self._complement = answer
        return answer

    def analogous(self, count: int = 5, divisions: int = 12) -> list[Hct]:
        start_hue = _jround(self.input.hue)
        by_hue = self.hcts_by_hue()
        start_hct = by_hue[start_hue]
        last = self.relative_temperature(start_hct)
        all_colors = [start_hct]
        total_abs = 0.0
        for i in range(360):
            h = by_hue[sanitize_degrees_int(start_hue + i)]
            tt = self.relative_temperature(h)
            total_abs += abs(tt - last)
            last = tt
        addend = 1
        step = total_abs / divisions
        total = 0.0
        last = self.relative_temperature(start_hct)
        while len(all_colors) < divisions:
            h = by_hue[sanitize_degrees_int(start_hue + addend)]
            tt = self.relative_temperature(h)
            total += abs(tt - last)
            desired = len(all_colors) * step
            ok = total >= desired
            index_addend = 1
            while ok and len(all_colors) < divisions:
                all_colors.append(h)
                desired = (len(all_colors) + index_addend) * step
                ok = total >= desired
                index_addend += 1
            last = tt
            addend += 1
            if addend > 360:
                while len(all_colors) < divisions:
                    all_colors.append(h)
                break
        answers = [self.input]
        ccw = int(math.floor((count - 1.0) / 2.0))
        for i in range(1, ccw + 1):
            index = -i
            while index < 0:
                index += len(all_colors)
            if index >= len(all_colors):
                index %= len(all_colors)
            answers.insert(0, all_colors[index])
        cw = count - ccw - 1
        for i in range(1, cw + 1):
            index = i
            if index >= len(all_colors):
                index %= len(all_colors)
            answers.append(all_colors[index])
        return answers


def ratio_of_ys(y1: float, y2: float) -> float:
    lighter = max(y1, y2)
    darker = y1 if lighter == y2 else y2
    return (lighter + 5.0) / (darker + 5.0)


def ratio_of_tones(t1: float, t2: float) -> float:
    return ratio_of_ys(y_from_lstar(t1), y_from_lstar(t2))


def contrast_lighter(tone: float, ratio: float) -> float:
    if tone < 0.0 or tone > 100.0:
        return -1.0
    dark_y = y_from_lstar(tone)
    light_y = ratio * (dark_y + 5.0) - 5.0
    if light_y < 0.0 or light_y > 100.0:
        return -1.0
    real = ratio_of_ys(light_y, dark_y)
    if real < ratio and abs(real - ratio) > 0.04:
        return -1.0
    v = lstar_from_y(light_y) + 0.4
    return -1.0 if (v < 0 or v > 100) else v


def contrast_darker(tone: float, ratio: float) -> float:
    if tone < 0.0 or tone > 100.0:
        return -1.0
    light_y = y_from_lstar(tone)
    dark_y = ((light_y + 5.0) / ratio) - 5.0
    if dark_y < 0.0 or dark_y > 100.0:
        return -1.0
    real = ratio_of_ys(light_y, dark_y)
    if real < ratio and abs(real - ratio) > 0.04:
        return -1.0
    v = lstar_from_y(dark_y) - 0.4
    return -1.0 if (v < 0 or v > 100) else v


def contrast_lighter_unsafe(tone: float, ratio: float) -> float:
    v = contrast_lighter(tone, ratio)
    return 100.0 if v < 0.0 else v


def contrast_darker_unsafe(tone: float, ratio: float) -> float:
    return max(0.0, contrast_darker(tone, ratio))


# =============================================================================
# DynamicColor
# =============================================================================

class ContrastCurve:
    __slots__ = ("low", "normal", "medium", "high")

    def __init__(self, low, normal, medium, high):
        self.low, self.normal, self.medium, self.high = low, normal, medium, high

    def get(self, level: float) -> float:
        if level <= -1.0:
            return self.low
        if level < 0.0:
            return _lerp(self.low, self.normal, (level - (-1)) / 1)
        if level < 0.5:
            return _lerp(self.normal, self.medium, (level - 0) / 0.5)
        if level < 1.0:
            return _lerp(self.medium, self.high, (level - 0.5) / 0.5)
        return self.high


class ToneDeltaPair:
    __slots__ = ("role_a", "role_b", "delta", "polarity", "stay_together")

    def __init__(self, role_a, role_b, delta, polarity, stay_together):
        self.role_a, self.role_b, self.delta = role_a, role_b, delta
        self.polarity, self.stay_together = polarity, stay_together


def tone_prefers_light_foreground(tone: float) -> bool:
    return _jround(tone) < 60


def tone_allows_light_foreground(tone: float) -> bool:
    return _jround(tone) <= 49


def foreground_tone(bg_tone: float, ratio: float) -> float:
    lighter = contrast_lighter_unsafe(bg_tone, ratio)
    darker = contrast_darker_unsafe(bg_tone, ratio)
    lr = ratio_of_tones(lighter, bg_tone)
    dr = ratio_of_tones(darker, bg_tone)
    if tone_prefers_light_foreground(bg_tone):
        negligible = abs(lr - dr) < 0.1 and lr < ratio and dr < ratio
        return lighter if (lr >= ratio or lr >= dr or negligible) else darker
    return darker if (dr >= ratio or dr >= lr) else lighter


class DynamicColor:
    def __init__(self, name: str, palette: Callable, tone: Callable, is_background: bool = False,
                 background: Callable | None = None, second_background: Callable | None = None,
                 contrast_curve: ContrastCurve | None = None, tone_delta_pair: Callable | None = None):
        self.name, self.palette, self.tone = name, palette, tone
        self.is_background = is_background
        self.background, self.second_background = background, second_background
        self.contrast_curve, self.tone_delta_pair = contrast_curve, tone_delta_pair

    def get_argb(self, s: "DynamicScheme") -> int:
        return self.get_hct(s).to_int()

    def get_hct(self, s: "DynamicScheme") -> Hct:
        key = ("hct", self.name)
        if key not in s._cache:
            s._cache[key] = self.palette(s).get_hct(self.get_tone(s))
        return s._cache[key]

    def get_tone(self, s: "DynamicScheme") -> float:
        key = ("tone", self.name)
        if key not in s._cache:
            s._cache[key] = self._tone(s)
        return s._cache[key]

    def _tone(self, s: "DynamicScheme") -> float:
        decreasing = s.contrast_level < 0
        if self.tone_delta_pair is not None:
            p = self.tone_delta_pair(s)
            role_a, role_b, delta, polarity, stay = p.role_a, p.role_b, p.delta, p.polarity, p.stay_together
            bg_tone = self.background(s).get_tone(s)
            a_nearer = (polarity == "nearer" or (polarity == "lighter" and not s.is_dark)
                        or (polarity == "darker" and s.is_dark))
            nearer, farther = (role_a, role_b) if a_nearer else (role_b, role_a)
            am_nearer = self.name == nearer.name
            exp = 1 if s.is_dark else -1
            n_contrast = nearer.contrast_curve.get(s.contrast_level)
            f_contrast = farther.contrast_curve.get(s.contrast_level)
            n_init = nearer.tone(s)
            n_tone = n_init if ratio_of_tones(bg_tone, n_init) >= n_contrast else foreground_tone(bg_tone, n_contrast)
            f_init = farther.tone(s)
            f_tone = f_init if ratio_of_tones(bg_tone, f_init) >= f_contrast else foreground_tone(bg_tone, f_contrast)
            if decreasing:
                n_tone = foreground_tone(bg_tone, n_contrast)
                f_tone = foreground_tone(bg_tone, f_contrast)
            if (f_tone - n_tone) * exp < delta:
                f_tone = _clamp_double(0, 100, n_tone + delta * exp)
                if (f_tone - n_tone) * exp < delta:
                    n_tone = _clamp_double(0, 100, f_tone - delta * exp)
            if 50 <= n_tone < 60:
                if exp > 0:
                    n_tone = 60
                    f_tone = max(f_tone, n_tone + delta * exp)
                else:
                    n_tone = 49
                    f_tone = min(f_tone, n_tone + delta * exp)
            elif 50 <= f_tone < 60:
                if stay:
                    if exp > 0:
                        n_tone = 60
                        f_tone = max(f_tone, n_tone + delta * exp)
                    else:
                        n_tone = 49
                        f_tone = min(f_tone, n_tone + delta * exp)
                else:
                    f_tone = 60 if exp > 0 else 49
            return float(n_tone if am_nearer else f_tone)

        answer = self.tone(s)
        if self.background is None:
            return answer
        bg_tone = self.background(s).get_tone(s)
        desired = self.contrast_curve.get(s.contrast_level)
        if ratio_of_tones(bg_tone, answer) < desired:
            answer = foreground_tone(bg_tone, desired)
        if decreasing:
            answer = foreground_tone(bg_tone, desired)
        if self.is_background and 50 <= answer < 60:
            answer = 49 if ratio_of_tones(49, bg_tone) >= desired else 60
        if self.second_background is not None:
            bg1 = self.background(s).get_tone(s)
            bg2 = self.second_background(s).get_tone(s)
            upper, lower = max(bg1, bg2), min(bg1, bg2)
            if ratio_of_tones(upper, answer) >= desired and ratio_of_tones(lower, answer) >= desired:
                return answer
            light = contrast_lighter(upper, desired)
            dark = contrast_darker(lower, desired)
            avail = [v for v in (light, dark) if v != -1]
            if tone_prefers_light_foreground(bg1) or tone_prefers_light_foreground(bg2):
                return 100.0 if light == -1 else light
            if len(avail) == 1:
                return avail[0]
            return 0.0 if dark == -1 else dark
        return answer


# =============================================================================
# DynamicScheme + variants
# =============================================================================

VARIANTS = ("monochrome", "neutral", "tonal_spot", "vibrant", "expressive", "fidelity", "content", "rainbow",
            "fruit_salad")


class DynamicScheme:
    def __init__(self, source: Hct, variant: str, is_dark: bool, contrast_level: float,
                 primary: TonalPalette, secondary: TonalPalette, tertiary: TonalPalette,
                 neutral: TonalPalette, neutral_variant: TonalPalette, error: TonalPalette | None = None):
        self.source_color_hct = source
        self.source_color_argb = source.to_int()
        self.variant = variant
        self.is_dark = is_dark
        self.contrast_level = contrast_level
        self.primary_palette = primary
        self.secondary_palette = secondary
        self.tertiary_palette = tertiary
        self.neutral_palette = neutral
        self.neutral_variant_palette = neutral_variant
        self.error_palette = error or TonalPalette.from_hue_and_chroma(25.0, 84.0)
        self._cache: dict = {}

    def palettes(self) -> dict[str, TonalPalette]:
        return {"primary": self.primary_palette, "secondary": self.secondary_palette,
                "tertiary": self.tertiary_palette, "neutral": self.neutral_palette,
                "neutralVariant": self.neutral_variant_palette, "error": self.error_palette}


def _rotated_hue(source: Hct, hues: tuple, rotations: tuple) -> float:
    sh = source.hue
    if len(rotations) == 1:
        return sanitize_degrees(sh + rotations[0])
    for i in range(len(hues) - 1):
        if hues[i] < sh < hues[i + 1]:
            return sanitize_degrees(sh + rotations[i])
    return sh


def make_scheme(source: Hct | int, variant: str, is_dark: bool, contrast: float = 0.0) -> DynamicScheme:
    """``SchemeTonalSpot/Neutral/Vibrant/Expressive/Rainbow/FruitSalad/Monochrome/Fidelity/Content``
    (2021 spec, MCU defaults)."""
    if isinstance(source, int):
        source = Hct(source)
    h, c = source.hue, source.chroma
    f = TonalPalette.from_hue_and_chroma
    if variant == "tonal_spot":
        p = (f(h, 36.0), f(h, 16.0), f(sanitize_degrees(h + 60.0), 24.0), f(h, 6.0), f(h, 8.0))
    elif variant == "neutral":
        p = (f(h, 12.0), f(h, 8.0), f(h, 16.0), f(h, 2.0), f(h, 2.0))
    elif variant == "vibrant":
        hues = (0.0, 41.0, 61.0, 101.0, 131.0, 181.0, 251.0, 301.0, 360.0)
        p = (f(h, 200.0), f(_rotated_hue(source, hues, (18.0, 15.0, 10.0, 12.0, 15.0, 18.0, 15.0, 12.0, 12.0)), 24.0),
             f(_rotated_hue(source, hues, (35.0, 30.0, 20.0, 25.0, 30.0, 35.0, 30.0, 25.0, 25.0)), 32.0),
             f(h, 10.0), f(h, 12.0))
    elif variant == "expressive":
        hues = (0.0, 21.0, 51.0, 121.0, 151.0, 191.0, 271.0, 321.0, 360.0)
        p = (f(sanitize_degrees(h + 240.0), 40.0),
             f(_rotated_hue(source, hues, (45.0, 95.0, 45.0, 20.0, 45.0, 90.0, 45.0, 45.0, 45.0)), 24.0),
             f(_rotated_hue(source, hues, (120.0, 120.0, 20.0, 45.0, 20.0, 15.0, 20.0, 120.0, 120.0)), 32.0),
             f(sanitize_degrees(h + 15.0), 8.0), f(sanitize_degrees(h + 15.0), 12.0))
    elif variant == "rainbow":
        p = (f(h, 48.0), f(h, 16.0), f(sanitize_degrees(h + 60.0), 24.0), f(h, 0.0), f(h, 0.0))
    elif variant == "fruit_salad":
        p = (f(sanitize_degrees(h - 50.0), 48.0), f(sanitize_degrees(h - 50.0), 36.0), f(h, 36.0), f(h, 10.0),
             f(h, 16.0))
    elif variant == "monochrome":
        p = (f(h, 0.0), f(h, 0.0), f(h, 0.0), f(h, 0.0), f(h, 0.0))
    elif variant in ("fidelity", "content"):
        tc = TemperatureCache(source)
        tert = tc.complement() if variant == "fidelity" else tc.analogous(3, 6)[2]
        p = (f(h, c), f(h, max(c - 32.0, c * 0.5)), TonalPalette.from_int(fix_if_disliked(tert).to_int()),
             f(h, c / 8.0), f(h, c / 8.0 + 4.0))
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return DynamicScheme(source, variant, is_dark, contrast, *p)


# =============================================================================
# MaterialDynamicColors (2021 spec)
# =============================================================================

def _is_fidelity(s: DynamicScheme) -> bool:
    return s.variant in ("fidelity", "content")


def _is_mono(s: DynamicScheme) -> bool:
    return s.variant == "monochrome"


def _find_desired_chroma_by_tone(hue: float, chroma: float, tone: float, by_decreasing: bool) -> float:
    answer = tone
    closest = Hct.from_hct(hue, chroma, tone)
    if closest.chroma < chroma:
        peak = closest.chroma
        while closest.chroma < chroma:
            answer += -1.0 if by_decreasing else 1.0
            pot = Hct.from_hct(hue, chroma, answer)
            if peak > pot.chroma:
                break
            if abs(pot.chroma - chroma) < 0.4:
                break
            if abs(pot.chroma - chroma) < abs(closest.chroma - chroma):
                closest = pot
            peak = max(peak, pot.chroma)
    return answer


CC = ContrastCurve
_PRI = lambda s: s.primary_palette  # noqa: E731
_SEC = lambda s: s.secondary_palette  # noqa: E731
_TER = lambda s: s.tertiary_palette  # noqa: E731
_NEU = lambda s: s.neutral_palette  # noqa: E731
_NV = lambda s: s.neutral_variant_palette  # noqa: E731
_ERR = lambda s: s.error_palette  # noqa: E731


class MaterialDynamicColors:
    """All M3 roles (2021 spec); attribute access returns the role's DynamicColor."""

    def __init__(self):
        D = DynamicColor
        R = self
        hs = lambda s: R.surface_bright if s.is_dark else R.surface_dim  # highestSurface  # noqa: E731
        dk = lambda d, l: (lambda s: d if s.is_dark else l)  # noqa: E731

        def key(name, pal):
            return D(name, pal, lambda s: pal(s).key_color.tone)

        self.primary_palette_key_color = key("primary_palette_key_color", _PRI)
        self.secondary_palette_key_color = key("secondary_palette_key_color", _SEC)
        self.tertiary_palette_key_color = key("tertiary_palette_key_color", _TER)
        self.neutral_palette_key_color = key("neutral_palette_key_color", _NEU)
        self.neutral_variant_palette_key_color = key("neutral_variant_palette_key_color", _NV)

        self.background = D("background", _NEU, dk(6.0, 98.0), True)
        self.on_background = D("on_background", _NEU, dk(90.0, 10.0), False, lambda s: R.background, None,
                               CC(3.0, 3.0, 4.5, 7.0))
        self.surface = D("surface", _NEU, dk(6.0, 98.0), True)
        self.surface_dim = D("surface_dim", _NEU,
                             lambda s: 6.0 if s.is_dark else CC(87.0, 87.0, 80.0, 75.0).get(s.contrast_level), True)
        self.surface_bright = D("surface_bright", _NEU,
                                lambda s: CC(24.0, 24.0, 29.0, 34.0).get(s.contrast_level) if s.is_dark else 98.0,
                                True)

        def cc2(dark, light):
            return lambda s: (dark.get(s.contrast_level) if s.is_dark else
                              (light.get(s.contrast_level) if isinstance(light, CC) else light))

        self.surface_container_lowest = D("surface_container_lowest", _NEU, cc2(CC(4.0, 4.0, 2.0, 0.0), 100.0), True)
        self.surface_container_low = D("surface_container_low", _NEU,
                                       cc2(CC(10.0, 10.0, 11.0, 12.0), CC(96.0, 96.0, 96.0, 95.0)), True)
        self.surface_container = D("surface_container", _NEU,
                                   cc2(CC(12.0, 12.0, 16.0, 20.0), CC(94.0, 94.0, 92.0, 90.0)), True)
        self.surface_container_high = D("surface_container_high", _NEU,
                                        cc2(CC(17.0, 17.0, 21.0, 25.0), CC(92.0, 92.0, 88.0, 85.0)), True)
        self.surface_container_highest = D("surface_container_highest", _NEU,
                                           cc2(CC(22.0, 22.0, 26.0, 30.0), CC(90.0, 90.0, 84.0, 80.0)), True)
        self.on_surface = D("on_surface", _NEU, dk(90.0, 10.0), False, hs, None, CC(4.5, 7.0, 11.0, 21.0))
        self.surface_variant = D("surface_variant", _NV, dk(30.0, 90.0), True)
        self.on_surface_variant = D("on_surface_variant", _NV, dk(80.0, 30.0), False, hs, None,
                                    CC(3.0, 4.5, 7.0, 11.0))
        self.inverse_surface = D("inverse_surface", _NEU, dk(90.0, 20.0), False)
        self.inverse_on_surface = D("inverse_on_surface", _NEU, dk(20.0, 95.0), False, lambda s: R.inverse_surface,
                                    None, CC(4.5, 7.0, 11.0, 21.0))
        self.outline = D("outline", _NV, dk(60.0, 50.0), False, hs, None, CC(1.5, 3.0, 4.5, 7.0))
        self.outline_variant = D("outline_variant", _NV, dk(30.0, 80.0), False, hs, None, CC(1.0, 1.0, 3.0, 4.5))
        self.shadow = D("shadow", _NEU, lambda s: 0.0, False)
        self.scrim = D("scrim", _NEU, lambda s: 0.0, False)
        self.surface_tint = D("surface_tint", _PRI, dk(80.0, 40.0), True)

        def pair(a, b, polarity, stay):
            return lambda s: ToneDeltaPair(getattr(R, a), getattr(R, b), 10.0, polarity, stay)

        # primary
        self.primary = D("primary", _PRI,
                         lambda s: (100.0 if s.is_dark else 0.0) if _is_mono(s) else (80.0 if s.is_dark else 40.0),
                         True, hs, None, CC(3.0, 4.5, 7.0, 7.0), pair("primary_container", "primary", "nearer", False))
        self.on_primary = D("on_primary", _PRI,
                            lambda s: (10.0 if s.is_dark else 90.0) if _is_mono(s) else (20.0 if s.is_dark else 100.0),
                            False, lambda s: R.primary, None, CC(4.5, 7.0, 11.0, 21.0))

        def pc_tone(s):
            if _is_fidelity(s):
                return s.source_color_hct.tone
            if _is_mono(s):
                return 85.0 if s.is_dark else 25.0
            return 30.0 if s.is_dark else 90.0

        self.primary_container = D("primary_container", _PRI, pc_tone, True, hs, None, CC(1.0, 1.0, 3.0, 4.5),
                                   pair("primary_container", "primary", "nearer", False))

        def opc_tone(s):
            if _is_fidelity(s):
                return foreground_tone(R.primary_container.tone(s), 4.5)
            if _is_mono(s):
                return 0.0 if s.is_dark else 100.0
            return 90.0 if s.is_dark else 30.0

        self.on_primary_container = D("on_primary_container", _PRI, opc_tone, False, lambda s: R.primary_container,
                                      None, CC(3.0, 4.5, 7.0, 11.0))
        self.inverse_primary = D("inverse_primary", _PRI, dk(40.0, 80.0), False, lambda s: R.inverse_surface, None,
                                 CC(3.0, 4.5, 7.0, 7.0))
        # secondary
        self.secondary = D("secondary", _SEC, dk(80.0, 40.0), True, hs, None, CC(3.0, 4.5, 7.0, 7.0),
                           pair("secondary_container", "secondary", "nearer", False))
        self.on_secondary = D("on_secondary", _SEC,
                              lambda s: (10.0 if s.is_dark else 100.0) if _is_mono(s) else (20.0 if s.is_dark else 100.0),
                              False, lambda s: R.secondary, None, CC(4.5, 7.0, 11.0, 21.0))

        def sc_tone(s):
            initial = 30.0 if s.is_dark else 90.0
            if _is_mono(s):
                return 30.0 if s.is_dark else 85.0
            if not _is_fidelity(s):
                return initial
            return _find_desired_chroma_by_tone(s.secondary_palette.hue, s.secondary_palette.chroma, initial,
                                                not s.is_dark)

        self.secondary_container = D("secondary_container", _SEC, sc_tone, True, hs, None, CC(1.0, 1.0, 3.0, 4.5),
                                     pair("secondary_container", "secondary", "nearer", False))

        def osc_tone(s):
            if _is_mono(s):
                return 90.0 if s.is_dark else 10.0
            if not _is_fidelity(s):
                return 90.0 if s.is_dark else 30.0
            return foreground_tone(R.secondary_container.tone(s), 4.5)

        self.on_secondary_container = D("on_secondary_container", _SEC, osc_tone, False,
                                        lambda s: R.secondary_container, None, CC(3.0, 4.5, 7.0, 11.0))
        # tertiary
        self.tertiary = D("tertiary", _TER,
                          lambda s: (90.0 if s.is_dark else 25.0) if _is_mono(s) else (80.0 if s.is_dark else 40.0),
                          True, hs, None, CC(3.0, 4.5, 7.0, 7.0), pair("tertiary_container", "tertiary", "nearer", False))
        self.on_tertiary = D("on_tertiary", _TER,
                             lambda s: (10.0 if s.is_dark else 90.0) if _is_mono(s) else (20.0 if s.is_dark else 100.0),
                             False, lambda s: R.tertiary, None, CC(4.5, 7.0, 11.0, 21.0))

        def tc_tone(s):
            if _is_mono(s):
                return 60.0 if s.is_dark else 49.0
            if not _is_fidelity(s):
                return 30.0 if s.is_dark else 90.0
            return fix_if_disliked(s.tertiary_palette.get_hct(s.source_color_hct.tone)).tone

        self.tertiary_container = D("tertiary_container", _TER, tc_tone, True, hs, None, CC(1.0, 1.0, 3.0, 4.5),
                                    pair("tertiary_container", "tertiary", "nearer", False))

        def otc_tone(s):
            if _is_mono(s):
                return 0.0 if s.is_dark else 100.0
            if not _is_fidelity(s):
                return 90.0 if s.is_dark else 30.0
            return foreground_tone(R.tertiary_container.tone(s), 4.5)

        self.on_tertiary_container = D("on_tertiary_container", _TER, otc_tone, False,
                                       lambda s: R.tertiary_container, None, CC(3.0, 4.5, 7.0, 11.0))
        # error
        self.error = D("error", _ERR, dk(80.0, 40.0), True, hs, None, CC(3.0, 4.5, 7.0, 7.0),
                       pair("error_container", "error", "nearer", False))
        self.on_error = D("on_error", _ERR, dk(20.0, 100.0), False, lambda s: R.error, None, CC(4.5, 7.0, 11.0, 21.0))
        self.error_container = D("error_container", _ERR, dk(30.0, 90.0), True, hs, None, CC(1.0, 1.0, 3.0, 4.5),
                                 pair("error_container", "error", "nearer", False))
        self.on_error_container = D("on_error_container", _ERR,
                                    lambda s: (90.0 if s.is_dark else 10.0) if _is_mono(s) else (90.0 if s.is_dark else 30.0),
                                    False, lambda s: R.error_container, None, CC(3.0, 4.5, 7.0, 11.0))

        # fixed roles
        def fixed(prefix, pal, t_fix, t_dim, t_on, t_onv):
            fx, fd = f"{prefix}_fixed", f"{prefix}_fixed_dim"
            setattr(self, fx, D(fx, pal, lambda s: t_fix[0] if _is_mono(s) else t_fix[1], True, hs, None,
                                CC(1.0, 1.0, 3.0, 4.5), pair(fx, fd, "lighter", True)))
            setattr(self, fd, D(fd, pal, lambda s: t_dim[0] if _is_mono(s) else t_dim[1], True, hs, None,
                                CC(1.0, 1.0, 3.0, 4.5), pair(fx, fd, "lighter", True)))
            setattr(self, f"on_{prefix}_fixed",
                    D(f"on_{prefix}_fixed", pal, lambda s: t_on[0] if _is_mono(s) else t_on[1], False,
                      lambda s: getattr(R, fd), lambda s: getattr(R, fx), CC(4.5, 7.0, 11.0, 21.0)))
            setattr(self, f"on_{prefix}_fixed_variant",
                    D(f"on_{prefix}_fixed_variant", pal, lambda s: t_onv[0] if _is_mono(s) else t_onv[1], False,
                      lambda s: getattr(R, fd), lambda s: getattr(R, fx), CC(3.0, 4.5, 7.0, 11.0)))

        fixed("primary", _PRI, (40.0, 90.0), (30.0, 80.0), (100.0, 10.0), (90.0, 30.0))
        fixed("secondary", _SEC, (80.0, 90.0), (70.0, 80.0), (10.0, 10.0), (25.0, 30.0))
        fixed("tertiary", _TER, (40.0, 90.0), (30.0, 80.0), (100.0, 10.0), (90.0, 30.0))


MDC = MaterialDynamicColors()

# MaterialKolor ``DynamicScheme.toColorScheme()``: Compose ColorScheme role -> MCU role
COMPOSE_ROLES = {
    "primary": "primary", "onPrimary": "on_primary", "primaryContainer": "primary_container",
    "onPrimaryContainer": "on_primary_container", "inversePrimary": "inverse_primary",
    "secondary": "secondary", "onSecondary": "on_secondary", "secondaryContainer": "secondary_container",
    "onSecondaryContainer": "on_secondary_container", "tertiary": "tertiary", "onTertiary": "on_tertiary",
    "tertiaryContainer": "tertiary_container", "onTertiaryContainer": "on_tertiary_container",
    "background": "background", "onBackground": "on_background", "surface": "surface", "onSurface": "on_surface",
    "surfaceVariant": "surface_variant", "onSurfaceVariant": "on_surface_variant", "surfaceTint": "surface_tint",
    "inverseSurface": "inverse_surface", "inverseOnSurface": "inverse_on_surface", "error": "error",
    "onError": "on_error", "errorContainer": "error_container", "onErrorContainer": "on_error_container",
    "outline": "outline", "outlineVariant": "outline_variant", "scrim": "scrim", "surfaceBright": "surface_bright",
    "surfaceContainer": "surface_container", "surfaceContainerHigh": "surface_container_high",
    "surfaceContainerHighest": "surface_container_highest", "surfaceContainerLow": "surface_container_low",
    "surfaceContainerLowest": "surface_container_lowest", "surfaceDim": "surface_dim",
    "primaryFixed": "primary_fixed", "primaryFixedDim": "primary_fixed_dim", "onPrimaryFixed": "on_primary_fixed",
    "onPrimaryFixedVariant": "on_primary_fixed_variant", "secondaryFixed": "secondary_fixed",
    "secondaryFixedDim": "secondary_fixed_dim", "onSecondaryFixed": "on_secondary_fixed",
    "onSecondaryFixedVariant": "on_secondary_fixed_variant", "tertiaryFixed": "tertiary_fixed",
    "tertiaryFixedDim": "tertiary_fixed_dim", "onTertiaryFixed": "on_tertiary_fixed",
    "onTertiaryFixedVariant": "on_tertiary_fixed_variant",
}


def to_color_scheme(s: DynamicScheme) -> dict[str, int]:
    return {k: getattr(MDC, v).get_argb(s) for k, v in COMPOSE_ROLES.items()}


# =============================================================================
# ImageToolbox DynamicTheme.kt: Context.getColorScheme(dynamicColor = false)
# =============================================================================

ITB_STYLES = ("tonal_spot", "neutral", "vibrant", "expressive", "rainbow", "fruit_salad", "monochrome", "fidelity",
              "content")
COLOR_BLIND_TYPES = ("protanomaly", "deuteranomaly", "tritanomaly", "protanopia", "deuteranopia", "tritanopia",
                     "achromatomaly", "achromatopsia")
_COLOR_BLIND = {
    "protanomaly": (0.817, 0.183, 0.0, 0.333, 0.667, 0.0, 0.0, 0.125, 0.875),
    "deuteranomaly": (0.8, 0.2, 0.0, 0.258, 0.742, 0.0, 0.0, 0.142, 0.858),
    "tritanomaly": (0.967, 0.033, 0.0, 0.0, 0.733, 0.267, 0.0, 0.183, 0.817),
    "protanopia": (0.567, 0.433, 0.0, 0.558, 0.442, 0.0, 0.0, 0.242, 0.758),
    "deuteranopia": (0.625, 0.375, 0.0, 0.7, 0.3, 0.0, 0.0, 0.3, 0.7),
    "tritanopia": (0.95, 0.05, 0.0, 0.0, 0.433, 0.567, 0.0, 0.475, 0.525),
    "achromatomaly": (0.618, 0.320, 0.062, 0.163, 0.775, 0.062, 0.163, 0.320, 0.516),
    "achromatopsia": (0.299, 0.587, 0.114, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114),
}
F32 = np.float32


def _compose_color(r: float, g: float, b: float, a: float = 1.0) -> int:
    """Compose ``Color(red, green, blue, alpha)`` in sRGB: stored as 8-bit, (x * 255f + 0.5f).toInt()."""
    def q(v):
        v = min(max(F32(v), F32(0.0)), F32(1.0))
        return int(v * F32(255.0) + F32(0.5))
    return (q(a) << 24) | (q(r) << 16) | (q(g) << 8) | q(b)


def _channels(argb: int) -> tuple:
    return (F32(alpha_from_argb(argb)) / F32(255.0), F32(red_from_argb(argb)) / F32(255.0),
            F32(green_from_argb(argb)) / F32(255.0), F32(blue_from_argb(argb)) / F32(255.0))


def composite_over(fg: int, bg: int) -> int:
    """Compose ``Color.compositeOver`` (float32 maths, 8-bit result)."""
    fa, fr, fgc, fb = _channels(fg)
    ba, br, bgc, bb = _channels(bg)
    a = fa + ba * (F32(1.0) - fa)

    def comp(f, b):
        return F32(0.0) if a == 0 else (f * fa + b * ba * (F32(1.0) - fa)) / a
    return _compose_color(comp(fr, br), comp(fgc, bgc), comp(fb, bb), a)


def with_alpha(argb: int, alpha: float) -> int:
    """Compose ``Color.copy(alpha = ...)``."""
    _, r, g, b = _channels(argb)
    return _compose_color(r, g, b, alpha)


def blend_argb(c1: int, c2: int, ratio: float) -> int:
    """androidx ``ColorUtils.blendARGB`` (float32, truncating)."""
    ratio = F32(ratio)
    inv = F32(1.0) - ratio
    out = []
    for sh in (24, 16, 8, 0):
        v = F32((c1 >> sh) & 255) * inv + F32((c2 >> sh) & 255) * ratio
        out.append(int(v))
    return (out[0] << 24) | (out[1] << 16) | (out[2] << 8) | out[3]


def calculate_luminance(argb: int) -> float:
    """androidx ``ColorUtils.calculateLuminance``."""
    def lin(c):
        c = c / 255.0
        return c / 12.92 if c < 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    y = 100 * (lin(red_from_argb(argb)) * 0.2126 + lin(green_from_argb(argb)) * 0.7152
               + lin(blue_from_argb(argb)) * 0.0722)
    return y / 100


def calculate_contrast(fg: int, bg: int) -> float:
    """androidx ``ColorUtils.calculateContrast`` (background must be opaque)."""
    if alpha_from_argb(fg) < 255:
        fg = composite_colors(fg, bg)
    l1 = calculate_luminance(fg) + 0.05
    l2 = calculate_luminance(bg) + 0.05
    return max(l1, l2) / min(l1, l2)


def composite_colors(fg: int, bg: int) -> int:
    """androidx ``ColorUtils.compositeColors``."""
    ba, fa = alpha_from_argb(bg), alpha_from_argb(fg)
    a = 0xFF - (((0xFF - ba) * (0xFF - fa)) // 0xFF)

    def comp(f, b):
        return 0 if a == 0 else ((0xFF * f * fa) + (b * ba * (0xFF - fa))) // (a * 0xFF)
    return (a << 24) | (comp(red_from_argb(fg), red_from_argb(bg)) << 16) | \
        (comp(green_from_argb(fg), green_from_argb(bg)) << 8) | comp(blue_from_argb(fg), blue_from_argb(bg))


def _secondary_chroma(h: Hct) -> float:
    return min(max(h.chroma / 3.0, 14.0), 28.0)


def _tertiary_chroma(h: Hct) -> float:
    return min(max(h.chroma / 2.0, 20.0), 42.0)


def _tertiary_hue(h: Hct) -> float:
    return (h.hue + 60.0) % 360.0  # Kotlin Double.mod


def _dynamic_error_chroma(style: str) -> float:
    return {"neutral": 56.0, "tonal_spot": 72.0, "expressive": 76.0, "vibrant": 80.0}.get(style, 72.0)


def _dynamic_error_hue(h: Hct) -> float:
    bps = (0.0, 3.0, 13.0, 23.0, 33.0, 43.0, 153.0, 273.0, 360.0)
    hues = (12.0, 22.0, 32.0, 12.0, 22.0, 32.0, 22.0, 12.0)
    idx = max((i for i, b in enumerate(bps) if h.hue >= b), default=-1)
    idx = min(idx, len(hues) - 1)
    return hues[idx] if 0 <= idx < len(hues) else hues[-1]


def itb_dynamic_scheme(primary_argb: int, style: str, is_dark: bool, contrast: float) -> DynamicScheme:
    """The DynamicScheme ImageToolbox builds for ``ColorTuple(primary)`` (no other tuple colours)."""
    hct = Hct(primary_argb)
    if style != "tonal_spot":
        return make_scheme(hct, style, is_dark, contrast)
    a1 = TonalPalette.from_int(primary_argb)
    a2 = TonalPalette.from_hue_and_chroma(hct.hue, _secondary_chroma(hct))
    a3 = TonalPalette.from_hue_and_chroma(_tertiary_hue(hct), _tertiary_chroma(hct))
    n1 = TonalPalette.from_hue_and_chroma(hct.hue, min(hct.chroma / 12.0, 4.0))
    n2 = TonalPalette.from_int(n1.tone(90))
    return DynamicScheme(hct, "tonal_spot", is_dark, contrast, a1, a2, a3, n1, n2,
                         TonalPalette.from_hue_and_chroma(25.0, 84.0))


_INVERT_ROLES = ("primary", "onPrimary", "primaryContainer", "onPrimaryContainer", "inversePrimary", "secondary",
                 "onSecondary", "secondaryContainer", "onSecondaryContainer", "tertiary", "onTertiary",
                 "tertiaryContainer", "onTertiaryContainer", "background", "onBackground", "surface", "onSurface",
                 "surfaceVariant", "onSurfaceVariant", "surfaceTint", "inverseSurface", "inverseOnSurface", "error",
                 "onError", "errorContainer", "onErrorContainer", "outline", "outlineVariant", "surfaceBright",
                 "surfaceDim", "surfaceContainer", "surfaceContainerHigh", "surfaceContainerHighest",
                 "surfaceContainerLow", "surfaceContainerLowest", "primaryFixed", "primaryFixedDim", "onPrimaryFixed",
                 "onPrimaryFixedVariant", "secondaryFixed", "secondaryFixedDim", "onSecondaryFixed",
                 "onSecondaryFixedVariant", "tertiaryFixed", "tertiaryFixedDim", "onTertiaryFixed",
                 "onTertiaryFixedVariant")

_AMOLED = {  # role -> darken fraction (None = pure black)
    "primary": 0.3, "onPrimary": 0.1, "primaryContainer": 0.3, "onPrimaryContainer": 0.1, "inversePrimary": 0.3,
    "secondary": 0.3, "onSecondary": 0.1, "secondaryContainer": 0.3, "onSecondaryContainer": 0.1,
    "tertiary": 0.3, "onTertiary": 0.1, "tertiaryContainer": 0.3, "onTertiaryContainer": 0.1,
    "background": None, "onBackground": 0.1, "surface": None, "onSurface": 0.1, "surfaceVariant": 0.1,
    "onSurfaceVariant": 0.1, "inverseSurface": 0.5, "inverseOnSurface": 0.1, "error": 0.3,
    "onError": 0.1, "errorContainer": 0.3, "onErrorContainer": 0.1, "outline": 0.2, "outlineVariant": 0.2,
    "scrim": 0.5, "surfaceBright": 0.5, "surfaceDim": 0.5, "surfaceContainer": 0.5, "surfaceContainerHigh": 0.5,
    "surfaceContainerHighest": 0.5, "surfaceContainerLow": 0.5, "surfaceContainerLowest": 0.5,
    "primaryFixed": 0.3, "primaryFixedDim": 0.3, "onPrimaryFixed": 0.1, "onPrimaryFixedVariant": 0.1,
    "secondaryFixed": 0.3, "secondaryFixedDim": 0.3, "onSecondaryFixed": 0.1, "onSecondaryFixedVariant": 0.1,
    "tertiaryFixed": 0.3, "tertiaryFixedDim": 0.3, "onTertiaryFixed": 0.1, "onTertiaryFixedVariant": 0.1,
}


def _to_color_blind(argb: int, kind: str) -> int:
    a, r, g, b = (float(v) for v in _channels(argb))
    m = _COLOR_BLIND[kind]
    nr = float(F32(m[0] * r + m[1] * g + m[2] * b))
    ng = float(F32(m[3] * r + m[4] * g + m[5] * b))
    nb = float(F32(m[6] * r + m[7] * g + m[8] * b))
    return _compose_color(min(max(nr, 0), 1), min(max(ng, 0), 1), min(max(nb, 0), 1), a)


def itb_color_scheme(primary_argb: int, style: str = "tonal_spot", is_dark: bool = False, contrast: float = 0.0,
                     invert: bool = False, amoled: bool = False, color_blind: str | None = None) -> dict[str, int]:
    """ImageToolbox ``Context.getColorScheme(dynamicColor = false, colorTuple = ColorTuple(primary))``:
    Compose ColorScheme role name -> ARGB int."""
    scheme = itb_dynamic_scheme(primary_argb, style, is_dark, contrast)
    cs = to_color_scheme(scheme)
    # withDynamicErrorColors(style, isDarkTheme, errorSeed = null)
    pal = TonalPalette.from_hue_and_chroma(_dynamic_error_hue(Hct(cs["primary"])), _dynamic_error_chroma(style))
    cs["error"] = blend_cam16_ucs(pal.tone(70 if is_dark else 30), cs["primary"], 0.15)
    cs["onError"] = pal.tone(22 if is_dark else 84)
    cs["errorContainer"] = blend_cam16_ucs(pal.tone(30 if is_dark else 78), cs["primaryContainer"], 0.15)
    cs["onErrorContainer"] = pal.tone(84 if is_dark else 22)
    # withExpressiveOnColors(isDarkTheme)
    tones = list(range(22, 9, -1)) if is_dark else [96, 97, 98]
    for role, on in (("primary", "onPrimary"), ("secondary", "onSecondary"), ("tertiary", "onTertiary")):
        p = TonalPalette.from_int(cs[role])
        for t in tones:
            cand = p.tone(t)
            if calculate_contrast(cand, cs[role]) >= 4.5:
                cs[on] = cand
                break
    # withContrastingSurfaces(surfaceSeed = null, isDarkTheme)
    if not is_dark:
        seed = Hct(cs["surface"])

        def surf(tone, mult=1.0):
            return TonalPalette.from_hue_and_chroma(seed.hue, min(seed.chroma * mult, 12.0)).tone(tone)
        cs.update(surface=surf(97), surfaceBright=surf(99), surfaceDim=surf(82, 1.8), surfaceContainerLowest=surf(100),
                  surfaceContainerLow=surf(94, 1.2), surfaceContainer=surf(91, 1.4), surfaceContainerHigh=surf(88, 1.6),
                  surfaceContainerHighest=surf(85, 1.8), surfaceVariant=surf(85, 1.8))
    if amoled and is_dark:
        for role, frac in _AMOLED.items():
            cs[role] = 0xFF000000 if frac is None else blend_argb(cs[role], 0xFF000000, frac)
    if color_blind:
        for role in _INVERT_ROLES:  # toColorBlind covers the same roles as invertColors
            cs[role] = _to_color_blind(cs[role], color_blind)
    if invert:
        for role in _INVERT_ROLES:
            cs[role] = cs[role] ^ 0x00FFFFFF
    # outlineVariant = onSecondaryContainer@0.2 over surfaceColorAtElevation(6.dp); background = surface
    elev_alpha = float((F32(4.5) * F32(math.log(F32(6.0) + F32(1.0))) + F32(2.0)) / F32(100.0))
    elevated = composite_over(with_alpha(cs["surfaceTint"], elev_alpha), cs["surface"])
    cs["outlineVariant"] = composite_over(with_alpha(cs["onSecondaryContainer"], 0.2), elevated)
    cs["background"] = cs["surface"]
    return cs


# ImageToolbox MaterialYouPalette.kt ``asCodeString`` role order
CODE_ROLES = ("background", "error", "errorContainer", "inverseOnSurface", "inversePrimary", "inverseSurface",
              "onBackground", "onError", "onErrorContainer", "onPrimary", "onPrimaryContainer", "onSecondary",
              "onSecondaryContainer", "onSurface", "onSurfaceVariant", "onTertiary", "onTertiaryContainer", "outline",
              "outlineVariant", "primary", "primaryContainer", "scrim", "secondary", "secondaryContainer", "surface",
              "surfaceTint", "surfaceVariant", "tertiary", "tertiaryContainer", "surfaceBright", "surfaceDim",
              "surfaceContainer", "surfaceContainerHigh", "surfaceContainerHighest", "surfaceContainerLow",
              "surfaceContainerLowest")


def _color_code(argb: int) -> str:
    """``Color.colorCode``: "Color(0xff" + toHex().drop(1).lowercase() + ")" (toHex drops an FF alpha)."""
    h = f"#{argb & 0xFFFFFFFF:08X}".replace("#FF", "#", 1)
    return f"Color(0xff{h[1:].lower()})"


def as_code_string(cs: dict[str, int], is_dark: bool) -> str:
    name = f"{'dark' if is_dark else 'light'}ColorScheme"
    body = "\n".join(f"    {r} = {_color_code(cs[r])}," for r in CODE_ROLES)
    return f"val {name} = ColorScheme(\n{body}\n)"


def source_colors_from_image(img_u8: np.ndarray, max_colors: int = 128, desired: int = 4,
                             resize_area: int = 112 * 112) -> tuple[list[tuple[int, float | None]], dict[int, int]]:
    """MCU image -> seeds: (downscale to ``resize_area`` like WallpaperColors / Flutter) QuantizerCelebi(128)
    then Score.  Returns ([(argb, score)], celebi {argb: population})."""
    h, w = img_u8.shape[:2]
    if h * w > resize_area:
        s = math.sqrt(resize_area / (h * w))
        nh, nw = max(1, int(math.ceil(h * s))), max(1, int(math.ceil(w * s)))
        ys = np.minimum(((np.arange(nh) + 0.5) * h / nh).astype(np.int64), h - 1)
        xs = np.minimum(((np.arange(nw) + 0.5) * w / nw).astype(np.int64), w - 1)
        img_u8 = img_u8[ys][:, xs]
    px = pixels_to_argb(img_u8)
    px = px[(px >> 24) == 255] if img_u8.shape[-1] == 4 else px  # MCU drops non-opaque pixels
    if len(px) == 0:
        return [(0xFF4285F4, None)], {}
    result = quantize_celebi(px, max_colors)
    return score(result, desired=desired, with_scores=True), result


def iter_tones(pal: TonalPalette, tones: Iterable[int] = range(0, 101, 10)) -> dict[str, str]:
    return {str(t): argb_to_hex(pal.tone(t)) for t in tones}
