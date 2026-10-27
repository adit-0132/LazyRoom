# Port report: ImageToolbox Noise Generation, Gradient Maker and mesh gradients

Tools (all in `src/lazyroom/tools/generate.py`): `noise_image`, `gradient_image` (rewritten for parity),
`canvas` (unchanged).  FastNoiseLite port: `src/lazyroom/tools/_fastnoise.py` (+ lookup tables
`_fastnoise_tables.py`, extracted verbatim from the header; MIT notice kept).  Gradient / mesh engine:
`src/lazyroom/tools/_gradient_impl.py`.  Palettes shared with the fractal port:
`src/lazyroom/fractals/palettes.py` (76 `GradientPalette` presets generated from the Kotlin source).
Tests: `tests/test_generators.py` (157 cases).

## Sources actually read

* ImageToolbox `feature/noise-generation`: `NoiseParams.kt` (defaults), the six enums, `AndroidNoiseGenerator.kt`
  (enum ordinals passed to the native call), `NoiseParamsSelection.kt` (slider ranges / visibility),
  `NoiseGenerationComponent.kt` (default 1000x1000).
* ImageToolboxLibs `libs/fast-noise`: vendored `FastNoiseLite.h` 1.1.1, `fast-noise.cpp` (image loop and
  colour mapping), `ColorUtils.cpp/.h`, `ColorGetter.h`, `FastNoise.kt`, `CMakeLists.txt` (no fast-math).
* ImageToolbox `feature/gradient-maker`: `AndroidGradientMaker.kt` (`createGradient`, `drawMeshGradient`,
  `PointData`, `cubicPathX/Y`), `UiGradientState.kt` (`getBrush`, `UiMeshGradientState`, `generateMesh`),
  `GradientMakerComponent.kt` (palette application, mesh grid/resolution limits), `GradientPropertiesSelector.kt`,
  `GradientMakerControls.kt`, `GradientMakerAppColorSchemeHandler.kt` (default stops), `TileModeSelector.kt`.
* ImageToolbox `feature/mesh-gradients/MeshGradientsComponent.kt` (only downloads a remote image pack).
* `core/domain/.../GradientPalette.kt`, English `strings.xml`.

## noise_image (ImageToolbox "Noise Generation" = FastNoiseLite)

`kind="fastnoise"` (default) is the ImageToolbox generator; parameters mirror `NoiseParams` with the same
defaults (seed 1337, frequency 0.01, OpenSimplex2, no fractal, 3 octaves, lacunarity 2, gain 0.5, weighted 0,
ping-pong 2, EuclideanSq / Distance, jitter 1, warp OpenSimplex2, amplitude 1) and the slider ranges.

| Item | param | status | notes |
|---|---|---|---|
| Noise types OpenSimplex2, OpenSimplex2S, Cellular, Perlin, ValueCubic, Value | `noise_type` | faithful | float32 port, C++ int32 wrap-around hashing; bit-exact vs the compiled header |
| Fractal None / FBm / Ridged / PingPong | `fractal_type` | faithful | bit-exact; keeps the vendored copy's 2D FBm weighting `FastMin(noise + 1, 2)` |
| Fractal DomainWarpProgressive / Independent | `fractal_type` | faithful | `GetNoise` treats them as single noise (exactly what ImageToolbox renders); they drive the warp when `domain_warp` is on |
| Octaves, lacunarity, gain, weighted strength, ping-pong strength | same names | faithful | `CalculateFractalBounding` reproduced in float32. Octaves allowed 1..10 (ImageToolbox UI 1..5) |
| Cellular distance (4) / return type (7) / jitter | `cellular_*` | faithful | bit-exact for all 28 combinations tested |
| Domain warp type / amplitude | `domain_warp_type`, `domain_warp_amp` | faithful | ImageToolbox passes them to FastNoiseLite but never calls `DomainWarp()`, so they have no visible effect there |
| DomainWarp (single, progressive, independent; OpenSimplex2, OpenSimplex2Reduced, BasicGrid; 2D and 3D) | `domain_warp=true` | faithful (extension to use it) | Ported and bit-exact; applying it before `GetNoise` is a LazyRoom option (default off = ImageToolbox) |
| Rotation type 3D | `rotation_type_3d` | faithful | Only affects 3D noise/warp; ImageToolbox samples 2D so it is a no-op there (also hidden in its UI) |
| 3D noise | `dimensions="3d"`, `z` | faithful (extension to use it) | All 3D noise types + transforms ported, bit-exact; sampling the plane `z` is a LazyRoom option |
| Seed | `seed` | faithful | Full int32 range (ImageToolbox UI ±10000) |
| Colour mapping | `color_a`, `color_b` | faithful | `fast-noise.cpp`: `fraction = (n + 1) / 2`, LAB lerp between black and white (`colorToLAB` with the upstream 0.045 sRGB threshold and float/double roundings), `labToColor`, `std::round`. Bit-exact for the default black/white; other endpoint colours (and alpha lerp) are a LazyRoom generalisation |
| Default size | `width`, `height` | adaptation | ImageToolbox 1000x1000; LazyRoom keeps its previous 1024x1024 |
| `white`, `value`, `fbm`, `turbulence`, `ridged`, `worley` | `kind` | not upstream | LazyRoom's previous generators, kept unchanged for compatibility (`scale`, `octaves`, `seed`) |

Verification: a C++ harness was compiled from the **unmodified** upstream `FastNoiseLite.h` and
`ColorUtils.cpp` plus the `fast-noise.cpp` pixel loop (g++ -O2, x86-64, no fast-math; empty stubs for the
Android headers).  210 configurations — every noise x fractal type, every cellular distance x return type,
every rotation type in 3D, 2D/3D domain warps for every warp type and warp fractal, extreme parameters
(negative frequency, lacunarity 49, gain -9, INT_MAX seed, amplitude -2000) — gave **bit-identical noise
floats and identical 8-bit output in 210/210 cases**.  Eight upstream samples are embedded in the tests.
Caveat: ImageToolbox's arm64 builds use clang, which may fuse `a*b+c` into FMA, and libc++ may resolve
`pow(float, float)` to `powf` in `LinearSRGBTosRGB`; on a device the result can therefore differ in the last
float bit (very rarely one grey level).

Performance (1024x1024): single-octave noise 6–66 ms, 5-octave FBm 30–210 ms.

## gradient_image (ImageToolbox "Gradient Maker")

| Item | param | status | notes |
|---|---|---|---|
| Linear | `kind="linear"`, `angle` | faithful | `getBrush` geometry: offset = centre + (cos, sin)·half-diagonal, clamped into the image, y flipped, start = size − end. 0° = left→right, 90° = bottom→top |
| Radial | `kind="radial"`, `center`, `radius` | faithful | centre = size·center, radius = max(w, h)/2·radius (min 0.01) |
| Sweep | `kind="sweep"`, `center`, `angle` | approximate (tiny) | `SweepGradient` + `setRotate(angle, cx, cy)` local matrix; clockwise from 3 o'clock in screen space. Uses exact `atan2`; Skia evaluates the angle with a polynomial approximation (difference far below one 8-bit step). No tile mode, as in Compose `sweepGradient` |
| Tile modes clamp / repeated / mirror / decal | `tile_mode` | faithful | Skia tiling formulas (mirror = `|(t-1) - 2·floor((t-1)/2) - 1|`, decal = transparent outside) |
| Colour stops | `gradient` | faithful | Sorted (stable), duplicate positions dropped except for sweep, single stop duplicated (ImageToolbox); implicit 0/1 stops and monotonic pinning (Skia); evaluated at pixel centres |
| Colour interpolation | — | faithful for opaque stops | sRGB-encoded, premultiplied (Android hwui creates gradient shaders with `kInterpolateColorsInPremul_Flag` — taken from AOSP, not verifiable locally). Translucent stops therefore blend premultiplied |
| Presets | `palette` | faithful | `setGradientPalette`: the 76 `GradientPalette` presets as evenly spaced stops (`index / lastIndex`) |
| Default stops / angle / size | — | adaptation | ImageToolbox derives the initial 3 stops from the Material You theme (not reproducible) and starts at 0°, 1000x1000; LazyRoom keeps its previous defaults (3-stop gradient, 45°, 1920x1080) |
| Dithering | — | faithful | none (Android `Paint()` default) |
| `angular`, `diamond`, `reflected` | `kind` | not upstream | LazyRoom extras, unchanged |

## gradient_image `kind="mesh"` (Gradient Maker mesh mode)

| Item | param | status | notes |
|---|---|---|---|
| Control grid | `grid_size` (2..6, default 2), `mesh` (JSON) | faithful | `generateMesh(n)`: regular n x n grid of normalised points. `mesh` accepts any 2..6 x 2..6 grid `[[[x, y, colour], ...], ...]` (points may be dragged outside 0..1 like in the editor) |
| Resolution | `resolution` (default 16) | faithful | Clamped to 24 / 32 / 40 / 64 for grids of 6 / 5 / 4 / <=3 points (`meshResolutionMax`), with a note in the result |
| Colours from a preset | `palette` | faithful | `setMeshGradientPalette`: `GradientPalette.sampleColors(n·n)` row by row (Kotlin `roundToInt`) |
| Random colours | `seed` > 0 | approximate | ImageToolbox's default mesh uses `Color(Random.nextInt()).copy(alpha = 1)`; LazyRoom draws the same kind of opaque random colours from a seeded NumPy generator (deterministic, different sequence) |
| Colours from `gradient` | (no palette, seed 0) | adaptation | The gradient is sampled at n·n evenly spaced positions (8-bit), LazyRoom's way to reuse the stops |
| Subdivision (`PointData`) | — | faithful | Horizontal pass with `cubicPathX` (first / middle / last segment handles), vertical pass with `cubicPathY` over the subdivided columns, including upstream's use of the row length (`points[y].lastIndex`) to pick the vertical handle type |
| `PathMeasure.getPosition` | — | faithful | Skia `SkContourMeasure`: cubic subdivided only when "too curvy" (tolerance 0.5 in path units), distance -> t by linear interpolation inside segments, cubic evaluated at t; zero-length paths give `Offset.Unspecified` (those triangles are skipped).  f64 instead of Skia f32 |
| Colour interpolation | — | approximate | Compose `lerp(Color, Color)`: Oklab, intermediate stored as half-float L/a/b + 10-bit alpha, result quantised to 8-bit sRGB (all replicated); Oklab matrices are Ottosson's linear-sRGB form in f64 instead of Compose's D50-adapted float connector — differences of at most ±1/255 |
| Drawing (`drawVertices(TRIANGLES)`) | — | approximate | Own Gouraud rasteriser: pixel-centre coverage, premultiplied barycentric colour interpolation, source-over in index order on a transparent canvas.  Skia's exact edge ownership rules are not reproduced (only affects which of two identical-colour neighbours draws a boundary pixel) |

## Not ported

* "Gradient Overlay" / "Mesh Gradient Overlay" (drawing a gradient on top of existing images with an alpha)
  — out of scope for these generator tools (LazyRoom has its own `generate.gradient` operation).
* The "Mesh Gradients" screen: it only downloads a pack of pre-rendered images from an unlicensed remote
  repository; deliberately **not** downloaded or bundled.  Only the generation algorithm (above) is ported.
* Palette import from files / colour picking from an image inside the Gradient Maker UI (LazyRoom has
  separate `palette_import` / `palette` tools).

## Backward compatibility

* `gradient_image`: all previous parameters and defaults are kept.  `linear` and `radial` now use
  ImageToolbox's geometry (outputs differ from before); `angular`, `diamond`, `reflected` are unchanged;
  `mesh` is now the ImageToolbox mesh gradient (the old inverse-distance blobs are gone) and `seed` now selects
  random mesh colours instead of random point positions.  New: `sweep`, `palette`, `radius`, `tile_mode`,
  `mesh`, `grid_size`, `resolution`.
* `noise_image`: all previous parameters are kept and the previous algorithms remain available as `kind`
  values.  The default `kind` is now `fastnoise` (ImageToolbox).  Defaults changed to ImageToolbox's
  `seed=1337` (was 0) and `octaves=3` (was 5), which also changes the legacy kinds when those parameters are
  omitted.  `seed` now accepts the full int32 range (legacy kinds mask it to 32 bits).
