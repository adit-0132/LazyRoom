"""androidx.palette (AOSP, Apache-2.0) port: ``Palette.Builder`` defaults, ``ColorCutQuantizer``
(5-bit histogram + volume-ordered median cut), ``Swatch``, ``Target`` scoring and the dominant swatch,
plus the ImageToolbox helpers built on it:

* ``extract_primary_color``  – lib/dynamic-theme ``Bitmap.extractPrimaryColor`` (dominant blended 50 %
  with vibrant, ``ColorUtils.blendARGB``) and the ``saturate(2f)`` pre-pass of ``updateColorByImage``.
* ``extract_image_color_palette`` – lib/colors ``extractImageColorPalette`` (ImageToolbox's "Default"
  palette mode: named swatches with population percentages).

Arithmetic follows the Java code (float32 where Java uses ``float``, Java PriorityQueue heap order for
the swatch list).  Down-scaling uses nearest-neighbour centre sampling, standing in for
``Bitmap.createScaledBitmap(..., filter = false)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

F32 = np.float32
_W = 5  # QUANTIZE_WORD_WIDTH
_MASK = (1 << _W) - 1
DEFAULT_RESIZE_AREA = 112 * 112
DEFAULT_MAX_COLORS = 16


# -----------------------------------------------------------------------------
# ColorUtils helpers (float32 like androidx.core.graphics.ColorUtils)
# -----------------------------------------------------------------------------

def rgb_to_hsl(r, g, b):
    """``ColorUtils.RGBToHSL`` for scalars or numpy int arrays (float32 maths)."""
    rf = np.asarray(r, F32) / F32(255.0)
    gf = np.asarray(g, F32) / F32(255.0)
    bf = np.asarray(b, F32) / F32(255.0)
    mx = np.maximum(rf, np.maximum(gf, bf))
    mn = np.minimum(rf, np.minimum(gf, bf))
    d = mx - mn
    l = (mx + mn) / F32(2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(mx == rf, np.fmod((gf - bf) / d, F32(6.0)),
                     np.where(mx == gf, (bf - rf) / d + F32(2.0), (rf - gf) / d + F32(4.0))).astype(F32)
        s = (d / (F32(1.0) - np.abs(F32(2.0) * l - F32(1.0)))).astype(F32)
    mono = mx == mn
    h = np.where(mono, F32(0.0), h)
    s = np.where(mono, F32(0.0), s)
    h = np.fmod(h * F32(60.0), F32(360.0)).astype(F32)
    h = np.where(h < 0, h + F32(360.0), h)
    return (np.clip(h, 0, 360).astype(F32), np.clip(s, 0, 1).astype(F32), np.clip(l, 0, 1).astype(F32))


def blend_argb(c1: int, c2: int, ratio: float) -> int:
    """``ColorUtils.blendARGB``."""
    ratio = F32(ratio)
    inv = F32(1.0) - ratio
    out = [int(F32((c1 >> sh) & 255) * inv + F32((c2 >> sh) & 255) * ratio) for sh in (24, 16, 8, 0)]
    return (out[0] << 24) | (out[1] << 16) | (out[2] << 8) | out[3]


# -----------------------------------------------------------------------------
# Swatch / Target
# -----------------------------------------------------------------------------

@dataclass
class Swatch:
    rgb: int          # opaque ARGB
    population: int

    @property
    def hsl(self) -> tuple[float, float, float]:
        h, s, l = rgb_to_hsl((self.rgb >> 16) & 255, (self.rgb >> 8) & 255, self.rgb & 255)
        return float(h), float(s), float(l)

    @property
    def hex(self) -> str:
        return f"#{self.rgb & 0xFFFFFF:06X}"


@dataclass
class Target:
    name: str
    saturation: tuple = (0.0, 0.5, 1.0)   # min, target, max
    lightness: tuple = (0.0, 0.5, 1.0)
    weights: tuple = (0.24, 0.52, 0.24)   # saturation, luma, population
    exclusive: bool = True

    def normalized_weights(self) -> tuple:
        w = [F32(x) for x in self.weights]
        total = F32(0.0)
        for x in w:
            if x > 0:
                total += x
        if total != 0:
            w = [x / total if x > 0 else x for x in w]
        return tuple(w)


_LIGHT = (0.55, 0.74, 1.0)
_NORMAL = (0.3, 0.5, 0.7)
_DARK = (0.0, 0.26, 0.45)
_VIBRANT = (0.35, 1.0, 1.0)
_MUTED = (0.0, 0.3, 0.4)
TARGETS = (
    Target("light_vibrant", _VIBRANT, _LIGHT), Target("vibrant", _VIBRANT, _NORMAL),
    Target("dark_vibrant", _VIBRANT, _DARK), Target("light_muted", _MUTED, _LIGHT),
    Target("muted", _MUTED, _NORMAL), Target("dark_muted", _MUTED, _DARK),
)


# -----------------------------------------------------------------------------
# ColorCutQuantizer
# -----------------------------------------------------------------------------

def _approx_rgb888(q: int) -> int:
    r, g, b = (q >> (2 * _W)) & _MASK, (q >> _W) & _MASK, q & _MASK
    return 0xFF000000 | ((r << 3) & 255) << 16 | ((g << 3) & 255) << 8 | ((b << 3) & 255)


def _default_filter_allowed(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Palette.DEFAULT_FILTER: not white (L>=0.95), not black (L<=0.05), not near the red I-line."""
    white = l >= F32(0.95)
    black = l <= F32(0.05)
    red_i = (h >= F32(10.0)) & (h <= F32(37.0)) & (s <= F32(0.82))
    return ~(white | black | red_i)


class _Vbox:
    __slots__ = ("q", "lo", "hi", "population", "min_r", "max_r", "min_g", "max_g", "min_b", "max_b")

    def __init__(self, q: "ColorCutQuantizer", lo: int, hi: int):
        self.q, self.lo, self.hi = q, lo, hi
        self.fit()

    def volume(self) -> int:
        return (self.max_r - self.min_r + 1) * (self.max_g - self.min_g + 1) * (self.max_b - self.min_b + 1)

    def color_count(self) -> int:
        return 1 + self.hi - self.lo

    def fit(self) -> None:
        cols = self.q.colors[self.lo:self.hi + 1]
        r, g, b = (cols >> (2 * _W)) & _MASK, (cols >> _W) & _MASK, cols & _MASK
        self.min_r, self.max_r = int(r.min()), int(r.max())
        self.min_g, self.max_g = int(g.min()), int(g.max())
        self.min_b, self.max_b = int(b.min()), int(b.max())
        self.population = int(self.q.hist[cols].sum())

    def longest_dimension(self) -> str:
        rl, gl, bl = self.max_r - self.min_r, self.max_g - self.min_g, self.max_b - self.min_b
        if rl >= gl and rl >= bl:
            return "r"
        if gl >= rl and gl >= bl:
            return "g"
        return "b"

    def split_point(self) -> int:
        dim = self.longest_dimension()
        cols = self.q.colors[self.lo:self.hi + 1]
        r, g, b = (cols >> (2 * _W)) & _MASK, (cols >> _W) & _MASK, cols & _MASK
        if dim == "g":
            key = (g << (2 * _W)) | (r << _W) | b
        elif dim == "b":
            key = (b << (2 * _W)) | (g << _W) | r
        else:
            key = cols
        order = np.argsort(key, kind="stable")  # keys are distinct, any sort gives Arrays.sort's result
        self.q.colors[self.lo:self.hi + 1] = cols[order]
        mid = self.population // 2  # int / 2 in Java
        csum = np.cumsum(self.q.hist[self.q.colors[self.lo:self.hi + 1]])
        hit = np.nonzero(csum >= mid)[0]
        if len(hit):
            return min(self.hi - 1, self.lo + int(hit[0]))
        return self.lo

    def split(self) -> "_Vbox":
        sp = self.split_point()
        new = _Vbox(self.q, sp + 1, self.hi)
        self.hi = sp
        self.fit()
        return new

    def average(self) -> Swatch:
        cols = self.q.colors[self.lo:self.hi + 1]
        pop = self.q.hist[cols].astype(np.int64)
        tot = int(pop.sum())
        r, g, b = (cols >> (2 * _W)) & _MASK, (cols >> _W) & _MASK, cols & _MASK

        def mean(c):  # Math.round(sum / (float) total)
            v = F32(int((pop * c).sum())) / F32(tot)
            return int(math.floor(float(v + F32(0.5))))
        rm, gm, bm = mean(r), mean(g), mean(b)
        rgb = 0xFF000000 | (((rm << 3) & 255) << 16) | (((gm << 3) & 255) << 8) | ((bm << 3) & 255)
        return Swatch(rgb, tot)


class _JavaPQ:
    """java.util.PriorityQueue with VBOX_COMPARATOR_VOLUME (largest volume first); iteration = heap order."""

    def __init__(self):
        self.a: list[_Vbox] = []

    @staticmethod
    def _cmp(x: _Vbox, y: _Vbox) -> int:
        return y.volume() - x.volume()

    def offer(self, x: _Vbox) -> None:
        k = len(self.a)
        self.a.append(x)
        while k > 0:
            parent = (k - 1) >> 1
            e = self.a[parent]
            if self._cmp(x, e) >= 0:
                break
            self.a[k] = e
            k = parent
        self.a[k] = x

    def poll(self) -> _Vbox | None:
        if not self.a:
            return None
        result = self.a[0]
        x = self.a.pop()
        n = len(self.a)
        if n > 0:
            k, half = 0, n >> 1
            while k < half:
                child = 2 * k + 1
                c = self.a[child]
                right = child + 1
                if right < n and self._cmp(c, self.a[right]) > 0:
                    child = right
                    c = self.a[child]
                if self._cmp(x, c) <= 0:
                    break
                self.a[k] = c
                k = child
            self.a[k] = x
        return result

    def __len__(self) -> int:
        return len(self.a)


class ColorCutQuantizer:
    def __init__(self, pixels_rgb: np.ndarray, max_colors: int, use_default_filter: bool = True):
        p = pixels_rgb.reshape(-1, pixels_rgb.shape[-1]).astype(np.int64)
        q = ((p[:, 0] >> 3) << (2 * _W)) | ((p[:, 1] >> 3) << _W) | (p[:, 2] >> 3)
        self.hist = np.bincount(q, minlength=1 << (3 * _W)).astype(np.int64)
        if use_default_filter:
            idx = np.arange(1 << (3 * _W))
            h, s, l = rgb_to_hsl(((idx >> (2 * _W)) & _MASK) << 3, ((idx >> _W) & _MASK) << 3, (idx & _MASK) << 3)
            self.hist[~_default_filter_allowed(h, s, l)] = 0
        self.use_filter = use_default_filter
        self.colors = np.nonzero(self.hist > 0)[0].astype(np.int64)
        if len(self.colors) <= max_colors:
            self.swatches = [Swatch(_approx_rgb888(int(c)), int(self.hist[c])) for c in self.colors]
        else:
            pq = _JavaPQ()
            pq.offer(_Vbox(self, 0, len(self.colors) - 1))
            while len(pq) < max_colors:
                vb = pq.poll()
                if vb is not None and vb.color_count() > 1:
                    pq.offer(vb.split())
                    pq.offer(vb)
                else:  # Java returns here without re-offering the polled box
                    break
            out = []
            for vb in pq.a:
                sw = vb.average()
                if not self._ignored(sw):
                    out.append(sw)
            self.swatches = out

    def _ignored(self, sw: Swatch) -> bool:
        if not self.use_filter:
            return False
        h, s, l = rgb_to_hsl((sw.rgb >> 16) & 255, (sw.rgb >> 8) & 255, sw.rgb & 255)
        return not bool(_default_filter_allowed(h, s, l))


# -----------------------------------------------------------------------------
# Palette
# -----------------------------------------------------------------------------

def scale_bitmap_down(img_u8: np.ndarray, resize_area: int = DEFAULT_RESIZE_AREA) -> np.ndarray:
    """``Palette.Builder.scaleBitmapDown``: area-limited, ceil()ed size, unfiltered (nearest) scaling."""
    h, w = img_u8.shape[:2]
    if resize_area <= 0 or w * h <= resize_area:
        return img_u8
    ratio = math.sqrt(resize_area / float(w * h))
    nw, nh = int(math.ceil(w * ratio)), int(math.ceil(h * ratio))
    ys = np.minimum(((np.arange(nh) + 0.5) * h / nh).astype(np.int64), h - 1)
    xs = np.minimum(((np.arange(nw) + 0.5) * w / nw).astype(np.int64), w - 1)
    return img_u8[ys][:, xs]


@dataclass
class Palette:
    swatches: list[Swatch]
    selected: dict[str, Swatch | None] = field(default_factory=dict)

    @property
    def dominant(self) -> Swatch | None:
        best, mx = None, -(1 << 31)
        for sw in self.swatches:
            if sw.population > mx:
                best, mx = sw, sw.population
        return best

    def color_for(self, target: str, default: int) -> int:
        sw = self.selected.get(target)
        return sw.rgb if sw is not None else default


def generate(img_u8: np.ndarray, maximum_color_count: int = DEFAULT_MAX_COLORS,
             resize_area: int = DEFAULT_RESIZE_AREA, use_default_filter: bool = True) -> Palette:
    """``Palette.from(bitmap).maximumColorCount(n).generate()`` (default filter and 6 default targets)."""
    small = scale_bitmap_down(img_u8, resize_area)
    swatches = ColorCutQuantizer(small[..., :3], maximum_color_count, use_default_filter).swatches
    pal = Palette(swatches)
    dom = pal.dominant
    max_pop = dom.population if dom is not None else 1
    used: set[int] = set()
    for t in TARGETS:
        ws, wl, wp = t.normalized_weights()
        best, best_score = None, F32(0.0)
        for sw in swatches:
            h, s, l = (F32(v) for v in sw.hsl)
            if not (F32(t.saturation[0]) <= s <= F32(t.saturation[2]) and
                    F32(t.lightness[0]) <= l <= F32(t.lightness[2]) and sw.rgb not in used):
                continue
            sc = F32(0.0)
            if ws > 0:
                sc += ws * (F32(1.0) - abs(s - F32(t.saturation[1])))
            if wl > 0:
                sc += wl * (F32(1.0) - abs(l - F32(t.lightness[1])))
            if wp > 0:
                sc += wp * (F32(sw.population) / F32(max_pop))
            if best is None or sc > best_score:
                best, best_score = sw, sc
        if best is not None and t.exclusive:
            used.add(best.rgb)
        pal.selected[t.name] = best
    return pal


# -----------------------------------------------------------------------------
# ImageToolbox helpers
# -----------------------------------------------------------------------------

def to_u8(img: np.ndarray) -> np.ndarray:
    """Float 0..1 (or uint8) RGB(A) -> uint8 RGB (8-bit like an Android ARGB_8888 bitmap)."""
    if img.dtype == np.uint8:
        return img[..., :3]
    return (np.clip(img[..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def saturate(img_u8: np.ndarray, saturation: float = 2.0, max_dim: int = 600) -> np.ndarray:
    """``DynamicThemeState.saturate``: draws the bitmap at (0, 0) into a min(w,600) x min(h,600) bitmap
    (i.e. a top-left crop, no scaling) through ``ColorMatrix.setSaturation`` (Rec.709-ish weights)."""
    crop = img_u8[:max_dim, :max_dim, :3].astype(F32) / F32(255.0)
    inv = F32(1.0) - F32(saturation)
    r, g, b = F32(0.213) * inv, F32(0.715) * inv, F32(0.072) * inv
    s = F32(saturation)
    m = np.array([[r + s, g, b], [r, g + s, b], [r, g, b + s]], F32)
    out = np.clip(crop @ m.T, 0, 1)
    return (out * F32(255.0) + F32(0.5)).astype(np.uint8)


def extract_primary_color(img: np.ndarray, blend_with_vibrant: bool = True, default: int = 0,
                          saturated: bool = False) -> int:
    """``Bitmap.extractPrimaryColor(default = 0, blendWithVibrant = true)`` -> ARGB int.

    When the image has no vibrant swatch the default (0 = transparent black) is blended in, exactly as
    ImageToolbox does, so the result can carry alpha 0x7F; Material You only uses its RGB hue/chroma.
    ``saturated`` applies the ``saturate(2f)`` pre-pass used when theming the app from an image."""
    u8 = to_u8(img)
    if saturated:
        u8 = saturate(u8)
    pal = generate(u8)
    dom = pal.dominant.rgb if pal.dominant is not None else default
    if blend_with_vibrant:
        return blend_argb(dom, pal.color_for("vibrant", default), 0.5)
    return dom


def _compose_luminance(argb: int) -> float:
    """Compose ``Color.luminance()`` (sRGB EOTF, Rec.709 weights, float result)."""
    def eotf(c):
        x = float(F32(c) / F32(255.0))
        return ((x + 0.055) / 1.055) ** 2.4 if x >= 0.04045 else x / 12.92
    y = 0.2126 * eotf((argb >> 16) & 255) + 0.7152 * eotf((argb >> 8) & 255) + 0.0722 * eotf(argb & 255)
    return float(F32(min(max(y, 0.0), 1.0)))


def _skia_hue(argb: int) -> float:
    """``android.graphics.Color.RGBToHSV`` hue (SkRGBToHSV, float32)."""
    r, g, b = (argb >> 16) & 255, (argb >> 8) & 255, argb & 255
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d == 0:
        return 0.0
    if r == mx:
        h = F32(g - b) / F32(d)
    elif g == mx:
        h = F32(2.0) + F32(b - r) / F32(d)
    else:
        h = F32(4.0) + F32(r - g) / F32(d)
    h = h * F32(60.0)
    if h < 0:
        h += F32(360.0)
    return float(h)


def extract_image_color_palette(img: np.ndarray, maximum_color_count: int = 32) -> list[dict]:
    """lib/colors ``extractImageColorPalette``: androidx Palette swatches, named with the colour library
    (ColorNameParser), percent of the swatch population, distinct by name, sorted by luminance then hue."""
    return named_palette_data(generate(to_u8(img), maximum_color_count=maximum_color_count))


def named_palette_data(pal: Palette) -> list[dict]:
    """The ``PaletteData`` list of ``extractImageColorPalette`` for an already generated palette."""
    from ._color_models import color_name as _name  # lib/colors ColorNameParser port

    def color_name(c):
        return _name((255, *[int(round(v * 255)) for v in c[:3]]))

    total = float(F32(sum(sw.population for sw in pal.swatches)))
    out, seen = [], set()
    for sw in pal.swatches:
        rgb = ((sw.rgb >> 16) & 255, (sw.rgb >> 8) & 255, sw.rgb & 255)
        name = color_name(tuple(c / 255.0 for c in rgb))
        if name in seen:
            continue
        seen.add(name)
        out.append({"hex": sw.hex, "rgb": list(rgb), "name": name, "population": sw.population,
                    "percent": float(F32(sw.population) / F32(total)) if total else 0.0, "_argb": sw.rgb})
    out.sort(key=lambda d: (_compose_luminance(d["_argb"]), _skia_hue(d["_argb"])))
    for d in out:
        d.pop("_argb")
    return out
