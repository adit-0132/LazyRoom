# Port report: ImageToolbox "Draw" (+ markup shape layers)

Modules: `src/lazyroom/ops/draw_modes.py` (operations, parameter types, drawing canvas, mode
renderers), `src/lazyroom/ops/_draw_raster.py` (stroker, nonzero rasteriser, blurs, numba
compositing kernels), `src/lazyroom/ops/_draw_paths.py` (path geometry, arrows, spray, flood fill,
line styles, markup shapes), `src/lazyroom/ops/_draw_warp.py` (warp engine).
Tests: `tests/test_draw_modes.py`.

Operations: `draw.paint`, `draw.flood_fill`, `draw.spot_heal`, `draw.warp`, `draw.shape_layer`.
They are registered on `import lazyroom.ops.draw_modes`; **`ops/__init__.py` still has to import
the module** (see "Wiring").

## Sources read

ImageToolbox (Apache-2.0), `feature/draw`:
`domain/{DrawMode,DrawPathMode,DrawLineStyle,Warp,PathPaint,DrawBehavior}.kt`,
`data/AndroidImageDrawApplier.kt` (the export renderer: the reference for every formula),
`data/utils/DrawRepeatedPath.kt`, `presentation/DrawContent.kt` (defaults: alpha 0.4 for the
highlighter, softness 35 Pt for neon, width clamps), `presentation/components/BitmapDrawer.kt`
(pointer handling: quad smoothing, warp strokes, flood fill on pointer-up),
`DrawCommittedPath.kt`, `controls/DrawContentControls.kt` (which controls each mode shows, path
modes / modes offered per line style), `utils/{PathHelper,DrawUtils,DrawGradient,DrawOutlinedFill,
FloodFill,DrawPathModeUtils}.kt`, `element/*ParamsSelector.kt` (slider ranges).
`core/domain/.../Pt.kt`, `core/ui/.../helper/DrawUtils.kt` (`Offset.rotate`),
`core/settings/.../SettingsState.kt` (default width 20 Pt, black).
`lib/opencv-tools/.../spot_heal/{SpotHealer,HealType}.kt`, `feature/filters/.../SpotHealFilter.kt`,
`PixelationFilter.kt`, `NativeStackBlurFilter.kt`, `lib/neural-tools/.../LaMaProcessor.kt`.
Trickle (Apache-2.0, T8RIN): `WarpEngine.kt`, `cpp/WarpEngine.cpp`.
`feature/markup-layers`: `domain/{MarkupLayer,ShapeLayerModeExt}.kt`,
`data/utils/{ShapeLayerRenderer,LayersRenderer,PictureLayerShadowRenderer,TextLayerMetrics}.kt`.
Android/Skia behaviour (SkStroke joins/caps, SkDashPathEffect, SkPath1DPathEffect, BlurMaskFilter
radius-to-sigma, `setShadowLayer`, HWUI `drawLayoutOnPath`) from knowledge of those sources, not
from a local copy.

Reused LazyRoom code: `ops/shapes.py` (`_Measure`, `_dash_pieces`, `_stamp`, `parse_svg_path`,
`shape_contours`, `STAMP_SHAPES`, `LINE_STYLES`), `ops/itb/blur.native_stack_blur` (libblur stack
blur), `ops/itb/_pixelation_impl` (`_pixelate_layer`, `prem_to_image`: ImageToolbox
`PixelationTool`), `ops/draw.load_font`, `core.blend.layer_over`, `core.io.load_image`, and
optionally `ai.models` / `ai.processors.lama_inpaint`.

## Feature table

| ImageToolbox feature | LazyRoom | Status | Notes |
|---|---|---|---|
| Pen | `draw.paint` `mode=pen` | faithful | Round caps/joins (square caps + miter joins for sharp-edged path modes), colour x alpha, BlurMaskFilter softness (sigma = 0.57735 r + 0.5). |
| Highlighter | `mode=highlighter` | faithful | Square caps, Android's default miter join (limit 4), default alpha 0.4 (`alpha=-1`). The whole path is one coverage, so self-overlaps never darken. |
| Neon | `mode=neon` | faithful | Opaque white core + `setShadowLayer(softness)` in the draw colour at alpha 0.8 (softness 0 removes the glow); default softness 35 Pt; alpha is ignored like upstream (control hidden). |
| Eraser (draw screen toggle) | `mode=eraser` | faithful | Clears the drawing layer (strokes earlier in the same step) with round caps + softness; always a free path (PathHelper ignores the path mode while erasing); line style ignored (export does). Cannot erase earlier pipeline steps: they are already part of the image. |
| Privacy blur | `mode=privacy_blur`, `blur_radius` 5..50 | faithful | Native stack blur radius = `int(blur_radius / 1000 * longest side)`, run on `image.overlay(drawing)` and drawn through the path (`pathEffectPaint`). Computed on the stroke box + radius margin (exact). |
| Pixelation paint | `mode=pixelation`, `pixel_size` 10..75 | faithful | Stack blur 20/1000 x longest side, then `PixelationTool` square layer with cell = `pixel_size / 1000 x longest side`; computed on a crop with the cell grid kept aligned to the image origin (test compares against a full-image run). |
| Custom filter brush | `mode=filter_brush`, `filter`, `filter_params` | faithful | Any registered non-geometry op via the registry, run on the whole composite, drawn through the path. Unknown ids / geometry ops / bad params -> INVALID_PARAMS. |
| Text (on path) | `mode=text`, `text`, `font`, `text_repeat`, `interval` | approximate | Glyphs placed like HWUI `drawLayoutOnPath` (RSXform at each glyph's mid-advance, baseline on the path, glyphs beyond the end pinned to the end); repeat logic of `drawRepeatedTextOnPath` incl. its partial last copy. Glyphs are rasterised by PIL/FreeType and rotated bilinearly; kerning from PIL pair lengths. Text size = width. Gradients not ported. |
| Image brush | `mode=image_brush`, `image`, `interval` | approximate | Copies every (bitmap width + interval) along the path, image fitted inside width x width, paint alpha applied, softness blurs the bitmap rectangle with a clamp shader (Skia semantics). Deviation: centred on the path and rotated once (see bugs). No bundled default sticker: `image` is required. |
| Spot heal (OpenCV) | `draw.spot_heal` `method=telea` | faithful | 8-bit RGB -> XYZ, `cv2.inpaint` (radius 3, Telea), XYZ -> RGB, mask = painted path > 0, result drawn through the path. Computed on the stroke box + margin (exact for this local algorithm). |
| Spot heal (Navier-Stokes) | `method=ns` | new | `HealType.NS` exists in `SpotHealer` but ImageToolbox never selects it. |
| Spot heal (LaMa) | `method=lama` | approximate | Calls `ai.models.require_model("@inpaint")` + `ai.processors.lama_inpaint`; raises BACKEND_UNAVAILABLE with the download command when the model is missing (upstream silently falls back to OpenCV). |
| Warp: Move, Grow, Shrink, Swirl CW/CCW, Mix | `draw.warp` `warp_mode`, `radius`, `strength`, `hardness` | faithful | Trickle `WarpEngine.cpp`: dab spacing radius/2 incl. its loop bounds, smoothstep(hardness, 1, t) falloff, amp 0.03 / 0.2 / 1, additive backward displacement map, one engine per gesture, final pointer-up dab, mix lerps towards the colour under the brush centre. Float maths, premultiplied RGBA (see deviations). |
| Path: Free | `path_mode=free` | faithful | BitmapDrawer: `moveTo(p0)`, `quadTo(prev, mid(prev, cur))` per point, `lineTo(last)`. A single point draws a dot (zero-length line with caps). |
| Path: Line, Arrow, Double arrow, Line arrow, Double line arrow | `line`, `arrow`, `double_arrow`, `line_arrow`, `double_line_arrow`, `arrow_size`, `arrow_angle` | faithful | `drawEndArrow`/`drawStartArrow`: head vector = last - position(length - width x size), rotated by +-angle (default 150), first end-head line continues the main contour. Bug fix: see below. |
| Path: Lasso | `lasso` | faithful | Smoothed free path, filled. |
| Path: Rect / Outlined rect | `rect` + `filled`, `rotation`, `corner_radius`, `fill_color` | faithful | `addRoundRect` with radius = min(w, h) x corner (<= 0.5), rotated about the centre; sharp edge (square caps, miter joins). Contour start/direction follow Skia's CCW `addRRect` (only matters for dash phase). |
| Path: Oval / Outlined oval | `oval` | faithful | Ellipse from the (inverted) PathHelper rect; sharp edge. |
| Path: Triangle / Outlined triangle | `triangle` | faithful | Base on the first point's row, apex at the last point's row. |
| Path: Polygon / Outlined polygon | `polygon` + `vertices`, `rotation`, `regular` | faithful (bug fix) | PathHelper formulas incl. its flipped y (top = max y). Regular radius fixed to min(w, h)/2. |
| Path: Star / Outlined star | `star` + `vertices`, `inner_radius`, `rotation`, `regular` | faithful (bug fix) | Same as polygon with alternating inner radius. |
| Outlined fill colour | `fill_color` | faithful | `DrawOutlinedFill`: drawn before the outline with the stroke paint (softness / neon glow kept, no path effect); an opaque fill colour takes the stroke alpha, a translucent one keeps its own. |
| Path: Spray | `path_mode=spray`, `density`, `particle_size`, `square_particles` | approximate | `density` particles per pointer event uniform in a disc of radius = width (`sqrt(U) x width`), particle = oval/rect `(x, y, x + size, y + size)`, filled. Deviations: seeded `ctx.rng` (determinism); the first point also sprays; gaps longer than width/2 between points are subdivided (an agent's two-point stroke sprays a band instead of two blobs); `particle_size` is in image px (upstream: px of the on-screen canvas). |
| Path: Flood fill | `path_mode=flood_fill`, `tolerance`; `draw.flood_fill` | faithful | 4-connected region whose 8-bit A, R, G, B are each within round(tolerance x 255) of the seed, computed on image + strokes so far; combinable with every mode (e.g. privacy blur a region). |
| Line style: Dashed | `line_style=dashed`, `dash_size`, `gap_size` | faithful | `dashPathEffect([size, gap + width])` per contour, dashes stroked with the paint caps. |
| Line style: Dot dashed | `dot_dashed` | faithful | `[4w, 2w, w/4, 2w]`. |
| Line style: Zigzag | `zigzag`, `zigzag_ratio` | faithful | Stamped chevron (thickness width/ratio), advance = width, filled. |
| Line style: Stamped | `stamped`, `stamp_shape`, `stamp_spacing` | faithful | `stampedPathEffect(shape sized width x width, advance spacing + width, Morph)`, filled; shapes from `shapes.STAMP_SHAPES` (default Material star). |
| Line style restrictions | — | faithful | Like the export: ignored by eraser, effect modes, text and image brush; dashes do nothing on filled paths (SkDashPath skips fill styles); zigzag/stamped on filled shapes stamp the outline. |
| Gradient strokes / fills (`GradientPalette`, `DrawGradient.kt`) | — | not implemented | Not in the requested scope. |
| `DrawBehavior.Background` (draw on a new canvas) | — | not implemented | Use a generate/canvas op first, then `draw.paint`. |
| Live preview helpers (line angle, helper grid, magnifier, zoom) | — | not applicable | UI only. Mirroring helpers exist in `DrawUtils.kt` but are unused by the draw screen. |
| Markup shape layer | `draw.shape_layer` | faithful | `ShapeLayerRenderer`: line, filled block arrow / double arrow (`buildFilledArrowPath`), line arrows (head length min(len/2 or len/3, box height x scale)), rect, oval, triangle (apex on top), polygon/star (markup's corrected geometry), filled or outlined + fill colour, round caps/joins, drop shadow (alpha extracted, BlurMaskFilter, tinted, offset), shadow-padding centring, layer rotation / flips / opacity / blend mode. Lengths are fractions of the shorter side (the layer system is relative to the editor canvas: default 16 px line = 0.016, shadow 6/12 px = 0.006/0.012, assuming a ~1000 px canvas). |
| Markup text layer | `draw.text` (existing) | partly covered | Covered: text, font, size, colour, outline, shadow, background box, alignment, rotation, opacity, blend. Missing in `draw.text`: underline / line-through decorations, `scaleX`/`skewX` transform, gradient fills, layer flips. |
| Markup picture / sticker layer | `draw.image` (existing) | partly covered | Missing in `draw.image`: drop shadow, corner radius, flips. |
| Markup project files, layer grouping, locking | — | not applicable | Editor/project features (LazyRoom has pipelines/projects). |

Existing ops already covering other parts: `draw.text`, `draw.image`, `draw.watermark`,
`draw.border`, `draw.shape` (simple annotations), `draw.styled_line` (shapes.py: a single polyline
with ImageToolbox line styles), `draw.erase_to_alpha` (erase-background screen), `ai.inpaint`.

## Parameters, units and defaults

* Points are normalised image coordinates (0..1, -1..2 accepted); each point is a pointer event.
  Line, arrows and shapes use the first ("down") and last ("up") points.
* `strokes` (new param kind `strokes`): list of strokes; a stroke is `[[x, y], ...]` or
  `{"points": [...], <any other draw.paint parameter>}` overriding the step parameters for that
  stroke, so one step can mix modes, colours and shapes. A bare single stroke or a JSON string is
  accepted. `filter_params` uses a new param kind `object` (dict or JSON).
* Lengths (`width`, `softness`, `dash_size`, `gap_size`, `stamp_spacing`, `particle_size`,
  `interval`, warp `radius`, heal `radius`) are full-resolution pixels (`unit="px"`, scaled with
  `ctx.px` for previews). **`-1` means ImageToolbox's default expressed in Pt** (1 Pt = 1/500 of the
  shorter side, computed on the rendered buffer, so it is preview-consistent): width 20 Pt,
  softness 35 Pt for neon / 0 otherwise, dash 10 Pt, gap 20 Pt, stamp spacing 20 Pt, warp radius
  20 Pt. `alpha=-1` means 0.4 for the highlighter and 1 otherwise. ImageToolbox's own sizes are Pt;
  to reproduce an ImageToolbox setting of N Pt pass `N * min(width, height) / 500` px.
* `blur_radius` and `pixel_size` keep ImageToolbox's unit (1/1000 of the longer side).

## Algorithms

* **Stroker** (`_draw_raster.stroke_outline`): Skia's construction: left offsets with joins, end
  cap, reversed right offsets, start cap; outer joins round (arc), miter (limit 4, bevel beyond)
  or bevel; inner joins routed through the pivot (`HandleInnerJoin`); closed contours give two
  loops; zero-length sub-paths draw the cap shape (dot / square). Arcs are flattened to 0.05 px.
* **Rasteriser** (`_draw_raster.rasterize`): nonzero winding resolved per sub-scanline (16 per
  row) with exact horizontal span coverage (difference-array accumulation, numba). The stroke
  outline's winding equals the number of covering segment quads/joins, so a nonzero fill is the
  exact union: self-crossing or tightly curved strokes never double their alpha. (`shapes.
  fill_polylines` accumulates signed area, which adds the anti-aliasing of overlapping pieces;
  fine for single shapes, not for strokes, hence the separate rasteriser.)
* **Compositing**: premultiplied float32, source-over, in-place numba kernels on the stroke's box.
  The step keeps ImageToolbox's model: a transparent drawing layer over the image, allocated only
  when an eraser stroke follows other strokes (source-over is associative otherwise).
* **Softness / glows**: sigma = 0.57735 r + 0.5; sigma >= 2 uses three box blurs
  (d = floor(sigma x 3 sqrt(2 pi) / 4 + 0.5), the approximation Skia's raster blur uses), smaller
  sigmas an exact Gaussian.
* **Warp**: numba port of `applyStrokeToPoint` / `interpolateAndApplyStroke` on a displacement map
  covering the gesture's box; rendering with `cv2.remap` (bilinear, replicated border).
* **Flood fill**: one numba pass computes the composite's 8-bit ARGB tolerance mask, then
  `cv2.floodFill` (4-connectivity) extracts the seed's component: same pixel set as the Kotlin
  scanline queue algorithm, which also yields the 4-connected component.

## Deviations

* Floating-point, premultiplied pipeline (ImageToolbox: 8-bit ARGB canvases). Spot-heal keeps
  8-bit XYZ inpainting inside the mask only.
* Alpha: warp and spot heal process alpha as a channel (premultiplied warp; alpha inpainted with
  float `cv2.inpaint`). Upstream forces the warp output opaque and drops alpha in SpotHealer.
* Warp edge sampling clamps coordinates (upstream extrapolates for negative coordinates).
* A single-point warp gesture applies one dab (upstream ignores taps).
* Arrows on exactly horizontal/vertical lines keep their heads; the `preLast != Offset.Zero` check
  is dropped.
* Regular polygons/stars use min(w, h)/2.
* Image brush stamps are centred on the path and rotated once.
* Dashes on closed contours restart at the contour start without merging the first and last dash
  (Skia joins them); visible only as a seam at the start corner.
* Spray: deterministic seeded RNG, first point sprays too, long gaps subdivided (width/2).
* LaMa without a downloaded model raises instead of falling back to OpenCV.
* Repeated text with an interval that makes the step < 1 px is clamped to 1 px; an image-brush
  advance (width + interval) <= 0 is clamped to 1 px.
* Hardness 1 (smoothstep edge0 = edge1) is defined as a step instead of the C++ NaN.

## Upstream bugs found

1. `PathHelper.drawPolygon` / `drawStar`: `top = max(y)`, `bottom = min(y)` make `height <= 0`, so a
   regular shape's radius is `min(width, height) / 2 = -|height| / 2`: it ignores the box width and
   overflows tall thin boxes. (Fixed: radius magnitude min(|w|, |h|)/2, sign kept so orientation
   is unchanged.) Side effect kept: regular and non-regular shapes rotate in opposite directions.
2. `drawEndArrow`/`drawStartArrow` draw the head only if `abs(v.x) < size*w && abs(v.y) < size*w`
   (strict), which fails for exactly horizontal/vertical shafts, and skip it when the measured
   point is exactly (0, 0). (Fixed.)
3. `drawRepeatedBitmapOnPath` rotates twice (canvas `rotate(degree)` and the bitmap matrix
   `postRotate(degree)`), so images turn by 2x the path angle and their centre lands at
   `pos + R * (w/2, h/2)` instead of on the path. (Fixed.)
4. `WarpEngine::interpolateAndApplyStroke` runs sub-steps `i = 0..steps-1` with
   `from = start + step * max(i - 1, 0)`: the first dab is a zero move and the last sub-step of each
   segment is never applied (left to the next segment; lost on the last one). Kept (faithful),
   covered by a test.
5. `WarpEngine::lerpColor` / `sampleBilinear` force alpha to 255 and treat premultiplied pixels as
   straight; negative sample coordinates extrapolate. (Alpha handled properly here.)
6. `smoothstep(hardness, 1, t)` divides by zero for hardness = 1 (NaN displacement in C++).
7. `SWIRL_CW` samples the source at the clockwise-rotated position, so on screen the content turns
   counter-clockwise (and `SWIRL_CCW` clockwise). Kept as-is (test documents it).
8. `drawRepeatedTextOnPath` with empty text: `textWidth + interval = 0` makes
   `fullRepeats = Int.MAX_VALUE` (effectively a hang). Empty text is a no-op here.
9. LaMa spot heal silently falls back to OpenCV when the model is missing.
10. Live preview vs export: the preview applies the line style to the eraser paint, the export does
    not (export behaviour ported).

## AI inpainting (read-only look at `src/lazyroom/ai/`)

`ai.models` has the `inpaint` task (`LaMa_512.onnx`, `LaMa_512_FAST.onnx`, URLs and checksums)
and `ai.processors.lama_inpaint(name, path, rgb, mask, ctx)` reproduces `LaMaProcessor` (512 px
whole-image inference). `draw.spot_heal method=lama` calls exactly `models.resolve_name("@inpaint")`,
`models.require_model(name)` and `processors.lama_inpaint(...)`, and composites the result only
inside the painted path (like `SpotHealFilter` + `clipBitmap`). Needed for it to work: the user
downloads the model (`lazyroom models download LaMa_512.onnx --yes`; onnxruntime 1.31 is already
installed). If the ai agent renames these functions, `_inpaint_lama` in `draw_modes.py` is the only
call site. Not verified with the real 208 MB model (not downloaded; no network).

## Performance

12 MP (4000 x 3000 RGB float32), three strokes (one 40-point wave across the image, two
diagonals), warm numba cache, minimum of 3-5 runs. First number: quiet machine; second: the same
machine later while it was swapping (4 GB swap, CPU at 1.8 GHz) with other agents running.

| Case | ms |
|---|---|
| pen, auto width (120 px) | 97 / 170 |
| highlighter + dashed | 92 / 169 |
| neon (glow sigma ~ 122 px) | 230 / 251 |
| spray, density 200 | 57 / 85 |
| zigzag / stamped | 88 / 98 |
| eraser after 3 strokes (drawing layer) | 296 / 305 |
| text, repeated | 272 / 629 |
| privacy blur, width 200 px | 357 / 910 |
| pixelation, width 200 px | 954 / 1812 |
| flood fill (draw.paint / draw.flood_fill, region ~ whole image) | 637-756 / 1025 |
| spot heal (Telea / NS), 60 px stroke | 65-95 |
| warp move / mix, radius 200 px | 161 / 126 |
| shape layer (block arrow + shadow) | 269 |
| filter brush | (cost of the chosen op on 12 MP) x number of strokes, e.g. `filter.sepia` ~1.05 s each |

Before the numba compositing kernels the pen case took 1.35 s and neon 8 s (pure numpy
temporaries and a 727-tap Gaussian).

## Tests

`LAZYROOM_ITB_ONLY=none LAZYROOM_HOME=/tmp/pm_draw_home .venv/bin/python -m pytest tests/test_draw_modes.py -q -x`
-> **201 passed** (~4 s with a warm numba cache). Coverage: every mode x path mode (126 cases,
including the INVALID_PARAMS text/image + spray/flood-fill combinations), every line style with pen /
highlighter / neon, dash period and gaps, filled star centre/vertices/concavity, polygon vertex
formula and the regular-radius fix, rect rotation and corner radius, arrow-head presence/position
(also exactly horizontal), arrow-head geometry, quad smoothing, highlighter overlap (no alpha
accumulation), square vs round caps, neon core/glow, softness, outlined fill colour, lasso, spray
radius/density/determinism, eraser semantics, privacy blur locality, pixelation cells and
crop-vs-full-image equality, filter brush through the registry, text on path + repeat, image brush
spacing, flood-fill tolerance and 4-connectivity, flood fill + privacy blur, spot heal error drop
(Telea and NS, >80 % lower error, untouched outside), LaMa unavailable -> BACKEND_UNAVAILABLE with
the download hint, warp move direction (both ways), grow/shrink area, swirl direction, mix,
the upstream sub-step quirk, shape-layer shapes/shadow/flip/rotation, determinism of all ops,
preview scale 0.5 vs downscaled full render (pen+softness, neon, dashed highlighter, filled star,
privacy blur, text, warp), RGBA inputs (alpha output, eraser restores alpha), RGB stays RGB, strokes
param forms + JSON round trip, invalid strokes / filters / images, rasteriser areas (union,
fractional), stroke cap areas and closed miter outlines.

`tests/test_ops_smoke.py -k draw` with `lazyroom.ops.draw_modes` imported before collection
(`import lazyroom.ops.draw_modes; pytest.main(["tests/test_ops_smoke.py", "-k", "draw"])`):
**19 passed, 2 skipped** (the skips are the existing `draw.image`, which needs a file). All op
examples validate and render.

## Wiring

* `src/lazyroom/ops/__init__.py`: add `from . import draw_modes  # noqa: F401  (ImageToolbox Draw modes)`.
* GUI: the form is auto-generated. `strokes` (kind `strokes`) and `filter_params` (kind `object`)
  fall back to the JSON text field. A canvas stroke-capture tool would be the natural editor for
  `strokes`: the mask brush in `gui/controller.py` (`strokeBegin` / `strokeTo`) already records
  normalised points and could append `{"points": [...]}` to `draw.paint` / `draw.spot_heal` /
  `draw.warp` steps. Shape path modes only need the drag start and end. The `-1` "automatic"
  values show as -1 on the sliders (described in each parameter).
