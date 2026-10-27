"""Generators that create new images: gradients (ImageToolbox Gradient Maker + mesh gradients),
procedural noise (ImageToolbox Noise Generation = FastNoiseLite), solid canvases."""
from __future__ import annotations

import ast
import json

import numpy as np

from ..core.errors import invalid
from ..core.params import Bool, Choice, Color, Float, Gradient, Int, Point, Text, parse_color
from ..fractals.palettes import PALETTE_NAMES, PALETTES, argb_to_rgba, palette_colors, sample_colors
from ..ops._util import value_noise
from ..ops.generate import eval_gradient
from . import _fastnoise as fnl
from . import _gradient_impl as gi
from .registry import save, tool

F32 = np.float32
ITB_KINDS = ("linear", "radial", "sweep")
LEGACY_KINDS = ("angular", "diamond", "reflected")


def _rgb_or_rgba(img: np.ndarray) -> np.ndarray:
    return img if img[..., 3].min() < 1 else img[..., :3]


def _parse_mesh(text: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        rows = json.loads(text)
    except json.JSONDecodeError as e:
        try:  # interfaces that pre-parse JSON hand the Text param a stringified Python list
            rows = ast.literal_eval(text)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            raise invalid(f"'mesh' is not valid JSON: {e.msg}", param="mesh") from None
    if not isinstance(rows, list) or not 2 <= len(rows) <= 6:
        raise invalid("'mesh' must be a list of 2..6 rows", param="mesh")
    ncols = len(rows[0]) if isinstance(rows[0], list) else 0
    if not 2 <= ncols <= 6 or any(not isinstance(r, list) or len(r) != ncols for r in rows):
        raise invalid("every mesh row must contain the same number (2..6) of points", param="mesh")
    pts = np.zeros((len(rows), ncols, 2))
    cols = np.zeros((len(rows), ncols, 4))
    for y, row in enumerate(rows):
        for x, p in enumerate(row):
            if not isinstance(p, (list, tuple)) or len(p) != 3:
                raise invalid("mesh points must be [x, y, colour]", param="mesh")
            try:
                px, py = float(p[0]), float(p[1])
            except (TypeError, ValueError):
                raise invalid("mesh point coordinates must be numbers", param="mesh") from None
            if not (np.isfinite(px) and np.isfinite(py)):
                raise invalid("mesh point coordinates must be finite", param="mesh")
            pts[y, x] = (px, py)
            # Compose Color -> 8-bit sRGB
            cols[y, x] = np.floor(np.array(parse_color(p[2], "mesh")) * 255.0 + 0.5) / 255.0
    return pts, cols


@tool("gradient_image", "Create gradient image", category="generate", params=[
    Int("width", 1920, min=1, max=16000), Int("height", 1080, min=1, max=16000),
    Gradient("gradient", [[0, "#4158D0"], [0.5, "#C850C0"], [1, "#FFCC70"]]),
    Choice("palette", "none", choices=["none", *PALETTE_NAMES],
           labels={"none": "None (use gradient)", **{k: v[0] for k, v in PALETTES.items()}},
           description="ImageToolbox gradient preset: replaces the stops by the palette colours, evenly spaced "
                       "(mesh: colours sampled row by row)"),
    Choice("kind", "linear", choices=["linear", "radial", "sweep", "angular", "diamond", "reflected", "mesh"],
           description="linear / radial / sweep / mesh follow ImageToolbox; angular, diamond, reflected are "
                       "LazyRoom extras"),
    Float("angle", 45, min=0, max=360, unit="deg",
          description="linear: direction (0 = left to right, 90 = bottom to top); sweep: rotation of the start; "
                      "reflected: direction"),
    Point("center", (0.5, 0.5), min=0, max=1, description="radial / sweep / angular / diamond centre"),
    Float("radius", 0.5, min=0, max=1, description="radial: radius as a fraction of max(width, height) / 2"),
    Choice("tile_mode", "clamp", choices=gi.TILE_MODES, description="linear / radial: colour outside the gradient"),
    Text("mesh", "", multiline=True,
         description="mesh: JSON grid [[[x, y, colour], ...], ...] (2..6 rows x 2..6 points, x/y normalised); "
                     "empty = regular grid_size x grid_size grid"),
    Int("grid_size", 2, min=2, max=6, description="mesh: points per side of the generated grid"),
    Int("resolution", 16, min=1, max=64,
        description="mesh: subdivisions between control points (ImageToolbox caps it at 24/32/40/64 for grids "
                    "of 6/5/4/<=3 points)"),
    Int("seed", 0, min=0, max=1_000_000,
        description="mesh: >0 gives random opaque colours like ImageToolbox's default mesh; 0 = colours sampled "
                    "from the palette or gradient"),
    Text("output", ""), Bool("overwrite", False),
], source={"project": "ImageToolbox", "feature": "Gradient Maker / Mesh Gradients", "license": "Apache-2.0"})
def gradient_image(api, width, height, gradient, palette, kind, angle, center, radius, tile_mode, mesh, grid_size,
                   resolution, seed, output, overwrite):
    """Linear / radial / sweep gradients with tile modes and ImageToolbox palettes, mesh gradients
    (control-point grid, Oklab colour interpolation), plus angular / diamond / reflected extras."""
    stops = gradient
    if palette != "none":
        cols = argb_to_rgba(palette_colors(palette))
        last = len(cols) - 1
        stops = [(i / last, tuple(float(v) for v in c)) for i, c in enumerate(cols)]
    result: dict = {}
    if kind == "mesh":
        if mesh.strip():
            pts, cols = _parse_mesh(mesh)
        else:
            n = grid_size
            ys, xs = np.mgrid[0:n, 0:n]
            pts = np.stack([xs / (n - 1.0), ys / (n - 1.0)], -1)
            if seed > 0:
                rgb = np.random.default_rng(seed).integers(0, 256, (n, n, 3)) / 255.0
                cols = np.concatenate([rgb, np.ones((n, n, 1))], -1)
            elif palette != "none":
                cols = argb_to_rgba(sample_colors(palette_colors(palette), n * n)).reshape(n, n, 4)
            else:
                t = np.linspace(0.0, 1.0, n * n)
                cols = np.floor(eval_gradient(stops, t).astype(np.float64) * 255.0 + 0.5).reshape(n, n, 4) / 255.0
        limit = gi.max_mesh_resolution(max(pts.shape[0], pts.shape[1]))
        if resolution > limit:
            result["notes"] = [f"resolution reduced to {limit} (ImageToolbox limit for this grid size)"]
            resolution = limit
        img = gi.render_mesh(width, height, pts, cols.astype(np.float64), resolution)
        result["resolution"] = resolution
    elif kind in ITB_KINDS:
        pos, pre = gi.prepare_stops(stops, sweep=kind == "sweep")
        if kind == "linear":
            t = gi.linear_t(width, height, angle)
            img = gi.eval_gradient(t, pos, pre, tile_mode)
        elif kind == "radial":
            t = gi.radial_t(width, height, center, radius)
            img = gi.eval_gradient(t, pos, pre, tile_mode)
        else:
            t = gi.sweep_t(width, height, center, angle)
            img = gi.eval_gradient(t, pos, pre, "clamp")
    else:  # LazyRoom extras (unchanged legacy geometry)
        X, Y = np.meshgrid((np.arange(width, dtype=F32) + 0.5) / width, (np.arange(height, dtype=F32) + 0.5) / height)
        dx, dy = X - center[0], (Y - center[1]) * height / width
        th = np.radians(angle)
        if kind == "reflected":
            t = np.abs(dx * np.cos(th) + dy * np.sin(th)) * 2
        elif kind == "diamond":
            t = (np.abs(dx) + np.abs(dy)) / 0.75
        else:
            t = (np.arctan2(dy, dx) / (2 * np.pi)) % 1.0
        img = eval_gradient(stops, np.clip(t, 0, 1))
    return {"output": save(api, _rgb_or_rgba(img), output, overwrite=overwrite), **result}


@tool("noise_image", "Generate noise texture", category="generate", params=[
    Int("width", 1024, min=1, max=16000), Int("height", 1024, min=1, max=16000),
    Choice("kind", "fastnoise", choices=["fastnoise", "white", "value", "fbm", "turbulence", "ridged", "worley"],
           description="fastnoise = ImageToolbox Noise Generation (FastNoiseLite, all options below). The other "
                       "values are LazyRoom's earlier generators (use scale / octaves / seed only)"),
    Choice("noise_type", "open_simplex2", choices=fnl.NOISE_TYPES),
    Float("frequency", 0.01, min=-0.5, max=0.5, step=0.001),
    Choice("fractal_type", "none", choices=fnl.FRACTAL_TYPES,
           description="domain_warp_* only change GetNoise when domain_warp is enabled (FastNoiseLite semantics)"),
    Int("octaves", 3, min=1, max=10),
    Float("lacunarity", 2.0, min=-50, max=50),
    Float("gain", 0.5, min=-10, max=10),
    Float("weighted_strength", 0.0, min=-3, max=3),
    Float("ping_pong_strength", 2.0, min=0, max=20),
    Choice("cellular_distance_function", "euclidean_sq", choices=fnl.CELLULAR_DISTANCE),
    Choice("cellular_return_type", "distance", choices=fnl.CELLULAR_RETURN),
    Float("cellular_jitter", 1.0, min=-10, max=10),
    Choice("domain_warp_type", "open_simplex2", choices=fnl.DOMAIN_WARP_TYPES),
    Float("domain_warp_amp", 1.0, min=-2000, max=2000, title="Amplitude"),
    Bool("domain_warp", False,
         description="Apply FastNoiseLite DomainWarp() to the coordinates before GetNoise (LazyRoom option: "
                     "ImageToolbox never calls DomainWarp, so its warp settings have no visible effect there)"),
    Choice("rotation_type_3d", "none", choices=fnl.ROTATION_TYPES, description="Only affects 3D sampling"),
    Choice("dimensions", "2d", choices=["2d", "3d"],
           description="2d = ImageToolbox GetNoise(x, y); 3d samples the plane at 'z' with GetNoise(x, y, z)"),
    Float("z", 0.0, min=-1e6, max=1e6, description="3D sampling plane"),
    Int("seed", 1337, min=-2_147_483_648, max=2_147_483_647),
    Float("scale", 64, min=1, max=4000, description="Legacy kinds only: feature size in pixels"),
    Color("color_a", "#000000"), Color("color_b", "#FFFFFF"),
    Text("output", ""), Bool("overwrite", False),
], source={"project": "ImageToolbox", "feature": "Noise Generation", "library": "FastNoiseLite 1.1.1 (MIT)"})
def noise_image(api, width, height, kind, noise_type, frequency, fractal_type, octaves, lacunarity, gain,
                weighted_strength, ping_pong_strength, cellular_distance_function, cellular_return_type,
                cellular_jitter, domain_warp_type, domain_warp_amp, domain_warp, rotation_type_3d, dimensions, z, seed,
                scale, color_a, color_b, output, overwrite):
    """Procedural noise: ImageToolbox / FastNoiseLite (OpenSimplex2(S), cellular, Perlin, value(-cubic);
    FBm / ridged / ping-pong / domain-warp fractals; LAB black->white mapping), or legacy white / value /
    fBm / turbulence / ridged / Worley generators."""
    if kind == "fastnoise":
        ic, fc = fnl.make_config(seed, frequency, noise_type, rotation_type_3d, fractal_type, octaves, lacunarity,
                                 gain, weighted_strength, ping_pong_strength, cellular_distance_function,
                                 cellular_return_type, cellular_jitter, domain_warp_type, domain_warp_amp)
        n = np.empty((height, width), np.float32)
        fnl.noise_grid(n, ic, fc, dimensions == "3d", np.float32(z), bool(domain_warp))
        a8 = [int(np.floor(v * 255.0 + 0.5)) for v in color_a]
        b8 = [int(np.floor(v * 255.0 + 0.5)) for v in color_b]
        rgba = np.empty((height, width, 4), np.float64)
        fnl.noise_to_rgba(n, np.array(fnl.color_to_lab(*a8[:3])), np.array(fnl.color_to_lab(*b8[:3])),
                          float(a8[3]), float(b8[3]), rgba)
        img = (rgba / 255.0).astype(F32)
        return {"output": save(api, _rgb_or_rgba(img), output, overwrite=overwrite),
                "range": [float(np.nanmin(n)), float(np.nanmax(n))]}
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    if kind == "white":
        n = rng.random((height, width), dtype=np.float32)
    elif kind == "worley":
        npts = max(4, int(width * height / (scale * scale)))
        pts = rng.random((npts, 2)).astype(F32) * [width, height]
        from scipy.spatial import cKDTree

        X, Y = np.meshgrid(np.arange(width, dtype=F32), np.arange(height, dtype=F32))
        d, _ = cKDTree(pts).query(np.stack([X.ravel(), Y.ravel()], 1), k=1, workers=-1)
        n = (d.reshape(height, width) / scale).astype(F32)
        n = np.clip(n / (n.max() or 1), 0, 1)
    else:
        n = value_noise(height, width, scale, rng, octaves=1 if kind == "value" else octaves)
        if kind == "turbulence":
            n = np.abs(n * 2 - 1)
        elif kind == "ridged":
            n = 1 - np.abs(n * 2 - 1)
            n = n * n
        lo, hi = np.percentile(n, 1), np.percentile(n, 99)
        n = np.clip((n - lo) / max(hi - lo, 1e-6), 0, 1)
    a, b = np.array(color_a, F32), np.array(color_b, F32)
    img = a + (b - a) * n[..., None]
    return {"output": save(api, _rgb_or_rgba(img), output, overwrite=overwrite)}


@tool("canvas", "Create solid canvas", category="generate", params=[
    Int("width", 1080, min=1, max=20000), Int("height", 1080, min=1, max=20000), Color("color", "#FFFFFF"),
    Text("output", ""), Bool("overwrite", False),
])
def canvas(api, width, height, color, output, overwrite):
    """Blank canvas (solid colour or transparent) to start a composition or collage."""
    img = np.empty((height, width, 4), F32)
    img[:] = np.array(color, F32)
    return {"output": save(api, img if color[3] < 1 else img[..., :3], output, overwrite=overwrite)}
