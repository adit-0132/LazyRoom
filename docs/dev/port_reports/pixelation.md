# Port report: Pixelation group (47 filters)

Module: `src/lazyroom/ops/itb/pixelation.py`; helpers: `src/lazyroom/ops/itb/_pixelation_impl.py`;
tests: `tests/itb/test_pixelation.py`.

**Summary:** 46 faithful, 1 approximate (ASCII), 0 not ported.

## General notes

* **PixelationTool family** (Pixelation, Circle/Diamond/Enhanced* Pixelation, *Pixelization, Stroke):
  this is ImageToolbox's `PixelationTool.render` with the exact `PixelationCommands` layer stacks:
  shape centres at `(i - 0.5) * resolution + offset` for `i in 0..int(dim / resolution + 1)`, the colour of
  the source pixel at the clamped centre, alpha `int(layerAlpha * A8)`, row-major drawing order, src-over
  onto a transparent canvas. The Android canvas uses ANTI_ALIAS paint. Here edge coverage is the exact pixel
  area for axis-aligned squares and a distance-based estimate for circles and 45° squares, so edge pixels
  can differ by a fraction of a pixel. Gaps the layers leave uncovered stay **transparent**, as in
  ImageToolbox. Those filters (Circle, Diamond, Nucleus, Pulse Grid at default) therefore return RGBA even
  for RGB input. Layer geometry is defined in full-resolution px and multiplied by `ctx.scale`
  (resolution ≥ 0.5 px on previews).
* **G'MIC filters**: the commands are built with `gmic_cmd` in the same argument order and enum values as
  the ImageToolboxLibs builders (or ImageToolbox's own `RawGmicFilter` for 3D Blocks). They include the
  `withPowerOfTwoPadding` / `withSquarePadding` wrappers and Puzzle's trailing `0`.
  `tests/itb/test_pixelation.py::test_gmic_default_command_strings` asserts every default command string.
  * The ImageToolboxLibs alpha modes are reproduced:
    * **Preserve**: only RGB goes to G'MIC. The source alpha is re-attached, bilinearly resampled if the
      size changed, and G'MIC's alpha output is ignored.
    * **Process**: RGBA goes to G'MIC, with an opaque alpha added for RGB input. Used by Chessboard
      Overlay, Circle Abstraction, Puzzle, Shapeism, Super Pixels and Taquin.
  * G'MIC seeds its RNG from the clock, so every pipeline is prefixed with `srand <ctx.rng>` to make it
    deterministic. Filters with an explicit seed parameter still apply their own `srand`.
  * Pixel-size parameters (chessboard sizes and offsets, flip-block sizes, ellipse radii, Shapeism sizes,
    super-pixel size and border, patch size, Mineral Mosaic area and smoothness) are scaled by `ctx.scale`
    and clamped to the G'MIC range. They are unchanged at full resolution.
  * Each command was verified with the dev CLI, which is **G'MIC 3.7.6**; ImageToolbox bundles 4.0.2.
* Random ops (aire Crystallize) use `ctx.rng`. Smear uses jhlabs' fixed `java.util.Random(567)`
  sequence, which is reproduced exactly.

## Filters

| Filter | op id | upstream | status | notes |
|---|---|---|---|---|
| Ascii | filter.ascii | ImageToolbox (lib/ascii ASCIIConverter) | approximate | Same algorithm: `round(w/fontSize)` columns, bilinear downscale, Rec.709 luma → gradient threshold map (`1 - i/(n-1)`, first threshold ≤ luma), text drawn at `(x*fs, baseline y*fs)` in grid-x-major order, coloured (or grey 0x88 with luma alpha), background via drawColor. Differences: glyphs come from Pillow/FreeType with the system font (`sans-serif` + per-glyph fontconfig fallback) instead of Android Roboto; glyph origins are rounded to whole pixels for fractional font sizes; when fewer than 5 columns would result (ImageToolbox throws) the font size is reduced to width/5. Gradient text is deduplicated / whitespace-stripped like the ImageToolbox UI. |
| Blocks3D | filter.blocks_3d | G'MIC | faithful | `fx_blocks3d` with ImageToolbox's raw 20-argument string (outline colour default 0x80000000). Output is square (longest side) → `geometry=True`. |
| ChessboardOverlay | filter.chessboard_overlay | G'MIC | faithful | `fx_chessboard`, alpha processed; sizes/offsets in px. |
| CircleAbstraction | filter.circle_abstraction | G'MIC | faithful | `fx_circle_abstraction`, alpha processed. The G'MIC command itself fails (`1,{h/2*density%},1,1,1` < 1 px is parsed as a file name; identical code in the 4.0.2 stdlib used by ImageToolbox) when a colour layer's skeleton is tiny, typically on small images. Only in that case it is retried on a 2×/4× nearest-upscaled copy and area-downscaled back. Slow upstream (≈14 s for 1.1 MP). |
| CirclePixelation | filter.circle_pixelation | ImageToolbox | faithful | `circle(v)` layers. Transparent gaps (see general notes); AA estimate. |
| CrossPixelization | filter.cross_pixelization | ImageToolbox | faithful | `cross(v)`; implementation file `CrossPixelationFilter.kt`. |
| Crystallize | filter.crystallize | aire | faithful | aire `crystallize`: integer sites in `[1,w-1]×[1,h-1]`, jc_voronoi cells (sites sorted by (y,x), duplicates pruned, bbox floor/ceil + 10 inflation, remap of the diagram rect onto the image). Cells are clipped exactly with a scipy/Qhull mirror construction. For every graph edge (CCW angle order of the midpoint) the bounding box of the triangle (site, edge) is filled with the colour at ceil(site), including aire's in-loop mutation of `v0`. The stroke is drawn with the aire DDA + 8-bit-style straight-alpha blend. Sites come from `ctx.rng` instead of the clock. The cluster count `w*h*0.01*amount` is divided by `scale²` on previews and is at least 1 (aire throws on 0). |
| DiamondPixelation | filter.diamond_pixelation | ImageToolbox | faithful | `diamond(v)` layers; AA estimate for 45° squares. |
| DiceMosaic | filter.dice_mosaic | G'MIC | faithful | `fx_dices`; output size changes → `geometry=True`. |
| DiffusionTensors | filter.diffusion_tensors | G'MIC | faithful | `fx_diffusiontensors`. |
| Ellipsionism | filter.ellipsionism | G'MIC | faithful | `fx_ellipsionism`; radii in px. |
| EnhancedCirclePixelation | filter.enhanced_circle_pixelation | ImageToolbox | faithful | `enhancedCircle(v)` layers. |
| EnhancedDiamondPixelation | filter.enhanced_diamond_pixelation | ImageToolbox | faithful | `enhancedDiamond(v)` layers. |
| EnhancedPixelation | filter.enhanced_pixelation | ImageToolbox | faithful | `enhancedSquare(v)` layers. |
| EqualizeHistogramPixelation | filter.equalize_histogram_pixelation | aire | faithful | aire `equalize_histogram_squares`: tiles of `W//gx × H//gy` (last row/column extends to the border), 101-bin HSV value histogram, CDF normalised by `(cdf - cdf[0]) / (N - cdf[0])` rounded to 1/100, all-black tiles → V = 1 (Rust `NaN.min(100)` quirk). Value replacement = HSV round trip (RGB scaled by V'/V) in float instead of 8-bit. Tile size is guarded to ≥ 1 px (aire panics when the grid exceeds the image size). |
| FadedArray | filter.faded_array | G'MIC | faithful | `fx_array_fade`; `repeat` sizing enlarges → `geometry=True`. |
| FlipAndRotateBlocks | filter.flip_and_rotate_blocks | G'MIC | faithful | `fx_flip_blocks`; block sizes in px; 47-entry channel choice. |
| MakeSeamless | filter.make_seamless | G'MIC | faithful | `fx_make_seamless` inside `withPowerOfTwoPadding`. |
| MicroMacroPixelization | filter.micro_macro_pixelization | ImageToolbox | faithful | `microMacro(v)` layers. |
| MineralMosaic | filter.mineral_mosaic | G'MIC | faithful | `fx_mineral_mosaic`; area/smoothness scaled for previews. |
| Ministeck | filter.ministeck | G'MIC | faithful | `fx_ministeck`; output = resolution × piece size px → `geometry=True` (resolution/piece size are output-geometry parameters, not scaled). |
| MirroredArray | filter.mirrored_array | G'MIC | faithful | `fx_array_mirror`; `expand_size` enlarges → `geometry=True`. |
| NucleusPixelization | filter.nucleus_pixelization | ImageToolbox | faithful | `nucleus(v)` layers (partially transparent result). |
| OldSchool8Bit | filter.old_school_8_bit | G'MIC | faithful | `fx_8bits`. |
| OrbitalPixelization | filter.orbital_pixelization | ImageToolbox | faithful | `orbital(v)` layers. |
| Pixelation | filter.pixelation | ImageToolbox | faithful | `square(v)`: resolution `v-4`, size `v` (overlapping squares → opaque). Test checks the exact cell-centre colour. |
| PolkaDot | filter.polka_dot | Trickle | faithful | Trickle `polkaDot` (dots of radius r on a `2r+spacing` grid, exact disc test, centre colour, rest transparent) + `drawColorBehind` (image over colour). Channel-symmetric, so Trickle's RGBA/ARGB byte-order quirk does not apply. |
| PulseGridPixelization | filter.pulse_grid_pixelization | ImageToolbox | faithful | `pulseGrid(v)` layers. |
| Puzzle | filter.puzzle | G'MIC | faithful | `fx_puzzle` (16 args incl. trailing 0), alpha processed. |
| Quadtree | filter.quadtree | G'MIC | faithful | `fx_quadtree` (Preserve: G'MIC's alpha output dropped like ImageToolboxLibs). |
| RadialWeavePixelization | filter.radial_weave_pixelization | ImageToolbox | faithful | `radialWeave(v)` layers. |
| RandomColorArray | filter.random_color_array | G'MIC | faithful | `fx_array_color` (+ `srand`). |
| RandomPatchArray | filter.random_patch_array | G'MIC | faithful | `fx_array_random`; output size changes → `geometry=True`. Source tile counts are capped at the image size, because G'MIC errors when tiles < 1 px (this only affects inputs that would fail otherwise). |
| RandomShadeStripes | filter.random_shade_stripes | G'MIC | faithful | `fx_shade_stripes` (+ `srand`). |
| RebuildFromSimilarBlocks | filter.rebuild_from_similar_blocks | G'MIC | faithful | `fx_rebuild_from_similar_blocks`. Registered `deterministic=False`: the command's nearest-block search is a forced-parallel (`:`) G'MIC fill, which races, so results vary between runs. This is also true in ImageToolbox. It is deterministic with `OMP_NUM_THREADS=1`. |
| Shapeism | filter.shapeism | G'MIC | faithful | `fx_shapeism`, alpha processed; max/min sizes in px. |
| ShufflePatches | filter.shuffle_patches | G'MIC | faithful | `fx_shuffle_patches`; power-of-two padded for Gradients/Laplacians like ImageToolboxLibs. |
| SimpleWeavePixelation | filter.simple_weave_pixelation | ImageToolbox | faithful | `simpleWeave(v)` layers. |
| Smear | filter.smear | jhlabs | faithful | Rust jhlabs SmearFilter: `java.util.Random(567)` sequence reproduced exactly (signed `%`, `& 0x7fffffff`), crosses / Bresenham lines / circles / squares, `mix_colors` on all four channels. Mixing is float, not truncated 8-bit. Circles and squares are replayed backwards with per-pixel remaining weight and union-find skipping. This equals forward painting within 1/2048 (verified against a naive forward painter in the tests) and avoids ~10⁹ paints at default settings. |
| StaggeredPixelization | filter.staggered_pixelization | ImageToolbox | faithful | `staggered(v)` layers. |
| StrokePixelation | filter.stroke_pixelation | ImageToolbox (+Trickle) | faithful | `stroke(v)` layers + Trickle `drawColorBehind`. |
| SuperPixels | filter.super_pixels | G'MIC | faithful | `fx_superpixels`, alpha processed; size/border in px. |
| Taquin | filter.taquin | G'MIC | faithful | `fx_taquin`, alpha processed; crops to a tile multiple → `geometry=True`. |
| TileableRotation | filter.tileable_rotation | G'MIC | faithful | `fx_rotate_tileable` inside `withSquarePadding` (cropped back to the input size). |
| VoronoiCrystallize | filter.voronoi_crystallize | jhlabs | faithful | Rust jhlabs CrystallizeFilter/CellularFilter: per-cube `JavaRandom(571x+23y)`, Poisson point table, grid types 0–4 incl. Perlin jitter (`PerlinNoise(0x4a484c616273)` tables), the asymmetric neighbour-cube search, `sample()` that ignores rotation/stretch, and the double division of the edge thickness (`smoothStep(0, e, (F2-F1)/e)`). Float instead of f32/8-bit. As upstream: `turbulence` < 2 and `amount` have no effect on Crystallize. |
| VortexPixelization | filter.vortex_pixelization | ImageToolbox | faithful | `vortex(v)` layers. |
| Weave | filter.weave | jhlabs | faithful | Rust jhlabs WeaveFilter (image colours, square threads, shaded crossings, 4×4 crossing matrix, widths ≥ 1); gaps are transparent (RGBA output). |

## Things the lead should know

* **RGBA output for RGB input** (faithful to ImageToolbox, whose canvases start transparent): Circle,
  Diamond, Nucleus and Pulse Grid pixelation, Weave, Polka Dot and ASCII with a transparent background,
  and Voronoi Crystallize with a translucent border colour. When everything is opaque the alpha channel is dropped again.
* `filter.rebuild_from_similar_blocks` is `deterministic=False` (a G'MIC thread race). An optional
  `env={"OMP_NUM_THREADS": "1"}` argument on the shared `run_gmic` would make it deterministic. I did not
  touch `_common.py`.
* `filter.ascii` imports `find_font` from `lazyroom.ops.draw` and calls `fc-match` with the `charset=`
  argument for per-glyph fallback. The default "░▒▓█" gradient needs a font that has block elements.
* Op ids come from the shared `snake()`: `filter.blocks_3d` and `filter.old_school_8_bit`.
* Timing on a 12 MP image, measured on a heavily loaded machine (load average ≈ 15 from parallel agents):
  * ≈ 1–2 s: pixelation family, Weave, Polka Dot, Equalize Histogram.
  * ≈ 3 s: ASCII and Voronoi Crystallize.
  * ≈ 5 s: aire Crystallize (Qhull on about 600k points).
  * Smear depends on its parameters.
  * The G'MIC filters are as slow as G'MIC itself; Circle Abstraction and Shapeism are very slow.
