# Port report: ImageToolbox "Artistic" group (40 filters)

Module: `src/lazyroom/ops/itb/artistic.py` (registrations), helpers in
`src/lazyroom/ops/itb/_artistic_impl.py`, tests in `tests/itb/test_artistic.py`.

## Sources read

* ImageToolbox `Ui*Filter.kt` (ranges, defaults, `GmicParameterInfo`), `data/model/*Filter.kt`
  (value → builder mapping), `GMICFilterTransformation.kt`, `GPUFilterTransformation.kt`,
  `JhFilterTransformation.kt`, `EnumMappings.kt`, strings.xml.
* ImageToolboxLibs G'MIC: builders `filters/{Artistic,Creative,Composite,Detail,GmicCommon,RawGmic}Filters.kt`,
  `Gmic.kt`, `model/Gmic{ExecutionOptions,Options,AlphaMode}.kt`, `cpp/gmic_jni.cpp` (alpha
  handling), the bundled `gmic.cpp` (4.0.2, patched) and `gmic_stdlib_community.h` (decoded to
  the 4.0.2 stdlib source to diff against the dev CLI's 3.7.6 stdlib).
* Trickle C++: `OilFilter.cpp`, `ColorGetter.h`/`Util.h`, `ProcessingFilters.cpp` (`popArtImpl`,
  `drawColorBehindImpl`), `lowpoly/*.c` + `lowpoly-lib.cpp`, Kotlin `LowPolyPipelineImpl.kt`,
  `EffectsPipelineImpl.kt` (`squareHarmony`).
* aire C++: `effect/OilEffect.cpp`, `pipelines/RemoveShadows.cpp`, `blur/MedianBlur.cpp`,
  `base/{Dilation,ArbitraryUtil,Arithmetics}.cpp`, `algo/EigenUtils.h`.
* jhlabs: the native Rust port `jhlabs/src/main/rust/jhlabs/src/lib.rs` (Cellular/Pointillize,
  StampFilter, JavaRandom, Perlin noise, gaussian_blur_exact); the Java classes are only
  parameter holders.
* GPUImage: `GPUImageToonFilter`, `GPUImageSmoothToonFilter`, `GPUImageGaussianBlurFilter`,
  `GPUImageTwoPassTextureSamplingFilter`, `GPUImage3x3TextureSamplingFilter`, `GPUImageFilterGroup`.

Validation done: compiled the aire median (`medianBlurChannel`) and dilation C++ into small
harnesses and compared (median: bit-exact on random data incl. the quirks; dilation: window
offsets for even kernels); Trickle oil vs a brute-force transliteration (exact); lowpoly Sobel
point extraction vs a transliteration (exact); JavaRandom vs Java reference values; the G'MIC
no-smooth patch (below) vs a patch built from the stdlib source (identical outputs).

## G'MIC filters: how ImageToolbox really runs them

All 30 G'MIC filters use the exact builder command strings (`gmic_cmd`, enum `.value`s,
booleans 0/1, colours as 0..255 ints) and run through `_artistic_impl.gmic_itb`, which
reproduces ImageToolboxLibs' native bridge:

* **`smooth` is disabled in ImageToolbox.** `GmicExecutionOptions.disableSmooth` defaults to
  `true`, and ImageToolboxLibs' patched `gmic.cpp` then skips every `smooth` built-in call
  (with its argument). This changes the look of 13 of our filters a lot (mean abs. difference vs
  stock G'MIC up to 100/255: Ink Wash, Pencil Portrait, Stringify, Painting, Hope Poster...).
  The CLI cannot disable a built-in, so `gmic_itb` dumps the affected stdlib commands from the
  installed G'MIC (`$$name`), replaces `smooth` by `skip` (same "consume the argument" semantics),
  writes them to `<tmp>/lazyroom-itb/nosmooth-<hash>.gmic` and loads it with
  `-command $LAZYROOM_ITB_GMIC_NOSMOOTH` (env var, so paths with spaces work through
  `run_gmic`'s splitter). Patched: `fx_engrave, fx_lylejk_painting, fx_painting, fx_pen_drawing,
  fx_rodilius, fx_smooth_anisotropic, fx_stringify, pencilbw, poster_edges, poster_hope,
  inpaint_pde` (found by tracing `smooth` calls with `-debug`). Switch:
  `_artistic_impl.ITB_DISABLE_SMOOTH`.
* **Exception `fx_highlight_bloom`**: it uses `+smooth`; skipping it removes an image and the
  command fails ("Invalid selection [-2]"), i.e. Dream Painting (and the light group's Cinematic
  Dream) error out in ImageToolbox. We keep stock `smooth` for that command.
* **Alpha modes**: `GmicAlphaMode.Preserve` (default) sends RGB only and ignores any alpha G'MIC
  returns (source alpha re-attached); `Process` (Engrave, Marker Drawing) sends RGBA and uses
  G'MIC's alpha. Note `run_gmic(preserve_alpha=True)` multiplies the source alpha by a returned
  alpha channel, which differs from ITB, so the wrapper passes RGB only and drops G'MIC's alpha.
* **Randomness**: the pipeline starts with `srand <ctx.rng>`. Six filters stay non-reproducible
  because G'MIC draws random numbers from OpenMP threads (Brushify, Felt Pen, Marker Drawing,
  Old Photograph, Pencil Portrait, Stringify). They are deterministic with `OMP_NUM_THREADS=1`
  but much slower, so they are registered with `deterministic=False`.
* **Version**: the dev CLI is G'MIC 3.7.6, ImageToolbox embeds 4.0.2. I diffed every stdlib
  command these filters reach. The only changes are messages, an equivalent substitution syntax
  in `fx_stringify`, and the `append_tiles` rewrite (Warhol, same-size tiles so no effect).
  Built-in C++ commands were not compared.
* Parameters are passed unscaled, as ImageToolbox does. The exception is Marker Drawing's
  "Length (px)" / "Radius (px)", which use `unit="px"` and `ctx.px`.

## Table

| Filter | op id | upstream | status | notes |
|---|---|---|---|---|
| Brushify | filter.brushify | G'MIC | faithful | `fx_brushify`; deterministic=False (OpenMP RNG) |
| Charcoal | filter.charcoal | G'MIC | faithful | `fx_charcoal` |
| ColorAbstraction | filter.color_abstraction | G'MIC | faithful | `fx_color_abstraction` |
| Cubism | filter.cubism | G'MIC | faithful | `fx_cubism` |
| Cutout | filter.cutout | G'MIC | faithful | `fx_cutout` |
| DodgeSketch | filter.dodge_sketch | G'MIC | faithful | `fx_dodgesketch` |
| Doodle | filter.doodle | G'MIC | faithful | `fx_doodle …,0`; extremely slow on big images |
| DreamPainting | filter.dream_painting | G'MIC | approximate | `fx_painting` (no smooth) + `fx_highlight_bloom` with stock smooth; in ImageToolbox the bloom step errors (see above) |
| Engrave | filter.engrave | G'MIC | faithful | `fx_engrave_preview`, Process alpha, no smooth |
| FeltPen | filter.felt_pen | G'MIC | faithful | `fx_feltpen`; deterministic=False |
| GraphicNovel | filter.graphic_novel | G'MIC | faithful | `fx_poster_edges …,12,p,20` + `fx_retinex …,16,2(Lab),1,1,r,15,80,250`; no smooth |
| Hedcut | filter.hedcut | G'MIC | faithful | `hedcut` |
| HopePoster | filter.hope_poster | G'MIC | faithful | `fx_poster_hope`; its only smoothing is skipped, so `smoothness` has no effect (as in ITB) |
| HoughSketch | filter.hough_sketch | G'MIC | faithful | `fx_houghsketchbw`; "transparent" colour models stay opaque (Preserve mode). `radius=0` makes G'MIC fail and is run as 1 |
| InkWash | filter.ink_wash | G'MIC | faithful | `fx_ink_wash`, no smooth |
| Linify | filter.linify | G'MIC | faithful | `fx_linify …,0`; `resolution=0` makes G'MIC fail and is run as 0.01. Very slow |
| LylejkPainting | filter.lylejk_painting | G'MIC | faithful | `fx_lylejk_painting`, no smooth |
| MarkerDrawing | filter.marker_drawing | G'MIC | faithful | `fx_marker_drawing`, Process alpha, RGBA background colour; length/radius px-scaled; deterministic=False |
| OldPhotograph | filter.old_photograph | G'MIC | faithful | `fx_old_photo`; deterministic=False |
| Painting | filter.painting | G'MIC | faithful | `fx_painting`, no smooth |
| PenDrawing | filter.pen_drawing | G'MIC | faithful | `fx_pen_drawing`; amplitude only feeds the skipped smooth, so it has no effect (as in ITB) |
| PencilPortrait | filter.pencil_portrait | G'MIC | faithful | `fx_pencil_portraitbw` + RGB; no smooth; deterministic=False |
| PosterEdges | filter.poster_edges | G'MIC | faithful | `fx_poster_edges`, no smooth |
| Rodilius | filter.rodilius | G'MIC | faithful | `fx_rodilius` (47 channel choices); no smooth |
| SharpAbstract | filter.sharp_abstract | G'MIC | faithful | `fx_sharp_abstract` |
| SmoothAbstract | filter.smooth_abstract | G'MIC | faithful | `fx_smooth_abstract`; `inpaint_pde` patched (edge-oriented mode smooths) |
| StainedGlass | filter.stained_glass | G'MIC | faithful | `fx_stained_glass` |
| Stringify | filter.stringify | G'MIC | faithful | `fx_stringify`; no smooth; deterministic=False. Upstream fails ("sort_list") when no string survives the threshold (e.g. threshold 1024 on small images) and raises OPERATION_FAILED |
| VectorPainting | filter.vector_painting | G'MIC | faithful | `fx_vector_painting` |
| Warhol | filter.warhol | G'MIC | faithful | `warhol`; geometry op: tiles are the image scaled by 1/max(x,y), so the output is `w*x/max × h*y/max` (rounded); source alpha resized |
| EnhancedOil | filter.enhanced_oil | Trickle | faithful | per-channel histogram mode over the window `[-r, r)` (asymmetric, ties → lowest value), result opaque (`RGB2Color`); numba sliding histogram; radius px-scaled |
| LowPoly | filter.low_poly | Trickle | approximate | Same pipeline: Sobel (threshold 50, premultiplied `(r+g+b)*a/3`, at most w·h/2 points in scan order), random picks with replacement, dedup, Delaunay + 4 corners, centroid colour. Differences: the RNG (ctx.rng instead of libc `random()`), scipy Delaunay tie-breaking, and OpenCV AA rasterisation instead of cairo. Outline mode (`fill=false`, cairo 0.0005-wide stroke drawn twice) is approximated by AA lines scaled by the width |
| Oil | filter.oil | aire | faithful | reproduces aire's bug: `posY` uses `i`, so only the diagonal `(x+i, y+i)` is scanned; the first max of `clamp(((r+g+b)/3)*levels)` wins; radius px-scaled |
| Pointillize | filter.pointillize | jhlabs | faithful | Rust port of Cellular/Pointillize in numba (JavaRandom per cube, Poisson table, Perlin `noise2` for grid types 2-4, f32), `edgeColor·ss + colour·(1-ss)` with fuzziness 0.1; scale px-scaled. `turbulence`/`amount` have no effect upstream (kept for parity) |
| PopArt | filter.pop_art | Trickle | faithful | cairo emulation: 2× box-downscaled image in 4 quadrants, each filled with a square-harmony colour (Skia HSV maths) at alpha=strength using the pixman PDF operator. Reproduces cairo reading the RGBA memory as BGRA: the per-quadrant channel orders are permuted and HSL luma weights are effectively (0.11, 0.59, 0.30). Odd sizes leave the last row/column transparent, as upstream |
| RubberStamp | filter.rubber_stamp | jhlabs | faithful | Rust StampFilter: radius `int(r·max(w,h)/64)`, `gaussian_blur_exact` (premultiplied, 8-bit rounding per pass), `smoothstep(t±s/2)` on `(r+g+b)/3`, colour mix with truncation; output alpha = mixed colour alpha |
| SandPainting | filter.sand_painting | Trickle | approximate | Same point selection as LowPoly (user threshold). Android `drawCircle(r=1)` with a non-AA hairline is rendered as the 2×2 block around the point. Unpremultiplied source colour, then `drawColorBehind`. RNG differs |
| Sketch | filter.sketch | aire | faithful | `removeShadows` per channel: dilate k×k (window `[x-(k-1)/2, x+k/2]`), the quirky 21-px sliding-histogram median (bit-exact), int8-wrapping `absDiff` on the SIMD part of each row (16 NEON lanes assumed, scalar tail exact), `255-d`, `ceil(v·255/max)` (min forced to 0). Kernel and median size px-scaled |
| SmoothToon | filter.smooth_toon | GPUImage | faithful | 9-tap blur (0.05..0.18) with the shader's swapped offsets (vertical step `b·H/W` px, then horizontal `b·W/H` px, linear + clamp, 8-bit FBOs), then Toon. Upstream bug: `GPUImageSmoothToonFilter.onInitialized()` resets blur 0.5 / threshold 0.2 / levels 10 after ITB's setters, so in ITB the sliders do nothing. We honour them (defaults identical) |
| Toon | filter.toon | GPUImage | faithful | Sobel on the red channel (3×3 texel taps, clamp), `floor(c·q+0.5)/q`, black where `mag >= threshold`; q=0 → 0 (GPU NaN) |

**Counts:** 37 faithful, 3 approximate (DreamPainting, LowPoly, SandPainting), 0 not ported.

## Timings (2 MP, 1732×1155 RGB)

Measured on a 12-thread machine while other agents ran (load ≈ 17, swapping), so the numbers
are pessimistic. G'MIC: op time with ITB's disabled smooth (op) where measured, otherwise CLI
time with stock G'MIC.

| Filter | time | | Filter | time |
|---|---|---|---|---|
| Doodle | ~280 s (CLI) | | Linify | ~384 s (CLI) |
| ColorAbstraction | ~52 s (CLI) | | Brushify | ~34 s (CLI) |
| DreamPainting | 22 s (op) | | Stringify | 18 s (op) |
| Rodilius | 14 s (op) | | Engrave | 13 s (op) |
| Painting | 11 s (op) | | Cubism | ~11 s (CLI) |
| GraphicNovel | 2.7 s (op) | | HoughSketch | ~9.7 s (CLI) |
| SharpAbstract | ~8.8 s (CLI) | | Cutout | ~5.2 s (CLI) |
| SmoothAbstract | 4.4 s (op) | | StainedGlass | ~4.9 s (CLI) |
| OldPhotograph | ~5.0 s (CLI) | | Hedcut | ~4.4 s (CLI) |
| MarkerDrawing | ~4.1 s (CLI) | | Warhol | ~3.6 s (CLI) |
| PenDrawing | 3.6 s (op) | | LylejkPainting | 3.5 s (op) |
| DodgeSketch | ~3.4 s (CLI) | | FeltPen | ~2.9 s (CLI) |
| PencilPortrait | 2.7 s (op) | | PosterEdges | 1.7 s (op) |
| Charcoal | 1.7 s (op) | | VectorPainting | 1.4 s (op) |
| HopePoster | 0.9 s (op; 49 s with stock smooth) | | InkWash | 0.7 s (op; 5 s stock) |
| EnhancedOil | 0.7 s | | LowPoly | 1.0 s |
| Oil | 0.4 s | | Pointillize | 0.4 s |
| PopArt | 0.8 s | | RubberStamp | 1.0 s |
| SandPainting | 0.6 s | | Sketch | 0.8 s |
| SmoothToon | 0.7 s | | Toon | 0.2 s |

The native ports are in the 0.2–1 s range at 2 MP. At 12 MP on the loaded machine they took
4–24 s, mostly numpy memory traffic with swapping.

## Tests

`tests/itb/test_artistic.py`:
* every op at all-numeric-minima and all-numeric-maxima on a small RGBA image;
  `stringify@max` is expected to raise (upstream G'MIC failure, see table);
* Toon/SmoothToon posterisation and edges, RubberStamp two colours, Pointillize/Oil only reuse
  source colours, Sketch per-channel stretch, Enhanced Oil per-channel mode, Pop Art quadrants,
  all six overlay modes, odd-size gap, square harmony, Low Poly fill/outline, Sand Painting
  background, JavaRandom, the aire median on flat input;
* G'MIC: Charcoal monochrome, Warhol geometry, Engrave alpha, `$$` dump re-escaping, and the
  disabled smooth (Pen Drawing's amplitude has no effect).
