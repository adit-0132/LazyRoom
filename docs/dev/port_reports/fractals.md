# Port report: ImageToolbox "Fractals" (fractal generation)

Tools: `fractal`, `fractal_types` (`src/lazyroom/tools/fractals.py`).
Package: `src/lazyroom/fractals/` — `catalog.py` (types, defaults, option schemas), `palettes.py`
(76 `GradientPalette` presets), `render.py` (ImageToolbox normalisation + render planning + engine entry
point `render_engine`), numba kernels `_escape.py` (2D per-pixel), `_density.py` (Buddhabrot, maps, IFS,
geometric curves, 3D attractors), `_raymarch.py` (distance estimators, shading, floor).
Tests: `tests/test_fractals.py` (226 cases).

## Sources actually read

* ImageToolbox `feature/fractal-generation`: `FractalFormula.kt` (75 formulas, UI defaults, capabilities,
  coefficient specs), `FractalParams.kt` (normalisation, ranges), `FractalIterationPolicy.kt`,
  `FractalRenderRequest.kt` (`resolvePlan`, work budget, `maxOutputPixelsFor`), `FractalViewport.kt`,
  `FractalCamera.kt`, `FractalQuaternion.kt`, `FractalCoefficients.kt`, `NativeFractalRendererBackend.kt`
  (`toNativeRequest`: coefficient -> wire mapping, degrees -> radians, FOV 45, Lyapunov "AB"),
  `FractalParamsSelection.kt` (slider ranges), `FractalLabels.kt` + `strings.xml` (titles),
  `FractalGenerationComponent.kt` (default 1080x1080 output).
* ImageToolbox `core/domain/.../GradientPalette.kt` (palettes; the Python table is generated from it).
* ImageToolboxLibs `libs/fractal-engine`: `FractalType.kt`, `FractalRenderRequest.kt`, `FractalEngineBridge.kt`
  (stable ids, validation, work estimates) and **all of the Rust engine** `render.rs`, `math.rs`,
  `attractors.rs`, `distance_estimators.rs`, `decimal.rs` (including its unit tests).

## Type catalogue

ImageToolbox exposes **75** formulas; the native engine knows **77** (`HTree`, `HilbertCurve` are
engine-only).  All 77 are ported; `fractal_types` reports `itb_ui: false` for the two extras.
Timing column: 1024x1024, ImageToolbox defaults (adaptive iterations), best of 2 runs, 12 threads,
measured while other agents were running (load average 4.5–9.7), see "Performance".

| Type | `type=` | upstream (stable id) | family | status | s @1024² | notes / type-specific options |
|---|---|---|---|---|---|---|
| Mandelbrot | `mandelbrot` | Mandelbrot (1) | escape-time | faithful | 0.12 | perturbation deep zoom (span <= 1e-12, power 2) with exact decimal reference orbit; options: power, bailout |
| Julia | `julia` | Julia (2) | escape-time | faithful | 0.10 | perturbation deep zoom like Mandelbrot; options: power, bailout, julia_real, julia_imaginary |
| Burning ship | `burning_ship` | BurningShip (3) | escape-time | faithful | 0.21 | options: power, bailout |
| Tricorn | `tricorn` | Tricorn (4) | escape-time | faithful | 0.08 | options: power, bailout |
| Multibrot | `multibrot` | Multibrot (5) | escape-time | faithful | 0.14 | options: power, bailout |
| Multicorn | `multicorn` | Multicorn (6) | escape-time | faithful | 0.11 | options: power, bailout |
| Celtic | `celtic` | Celtic (7) | escape-time | faithful | 0.14 | options: power, bailout |
| Buffalo | `buffalo` | Buffalo (8) | escape-time | faithful | 0.23 | options: power, bailout |
| Perpendicular burning ship | `perpendicular_burning_ship` | PerpendicularBurningShip (9) | escape-time | faithful | 0.11 | options: power, bailout |
| Phoenix | `phoenix` | Phoenix (10) | escape-time | faithful | 0.20 | uses previous z (no cycle shortcut); options: power, bailout, phoenix_real, phoenix_imaginary |
| Nova | `nova` | Nova (11) | escape-time | faithful | 0.23 | relaxation option; Newton/Nova escape = convergence (no smooth colouring, as upstream); options: power, bailout, nova_relaxation |
| Newton | `newton` | Newton (12) | escape-time | faithful | 0.09 | convergence |dz|^2 < 1e-20 counts as escape (upstream); options: power, bailout |
| Magnet I | `magnet_i` | MagnetI (101) | escape-time | faithful | 0.54 | separate magnet iteration, bailout max(r, 100); options: bailout |
| Magnet II | `magnet_ii` | MagnetII (102) | escape-time | faithful | 0.71 | separate magnet iteration, bailout max(r, 100); options: bailout |
| Lyapunov | `lyapunov` | Lyapunov (103) | escape-time | faithful | 1.12 | sequence 'AB' (ImageToolbox); engine `sequence` exposed as option; options: sequence |
| Sierpinski carpet | `sierpinski_carpet` | SierpinskiCarpet (104) | escape-time | faithful | 0.03 | depth = clamp(iterations, 1, 14); no iteration slider in ImageToolbox |
| Sierpinski triangle | `sierpinski_triangle` | SierpinskiTriangle (105) | escape-time | faithful | 0.02 | depth = clamp(iterations, 1, 18); sparse at default depth exactly like upstream |
| Burning Ship Julia | `burning_ship_julia` | BurningShipJulia (106) | escape-time | faithful | 0.07 | options: power, bailout, julia_real, julia_imaginary |
| Celtic Julia | `celtic_julia` | CelticJulia (107) | escape-time | faithful | 0.14 | options: power, bailout, julia_real, julia_imaginary |
| Collatz | `collatz` | Collatz (108) | escape-time | faithful | 0.13 | own colouring (smooth escape / min distance); options: bailout |
| Buddhabrot | `buddhabrot` | Buddhabrot (109) | 2D density | faithful | 0.09 | Halton(2,3) samples; samples split over threads, integer counts summed (order independent) |
| Hopalong | `hopalong` | Hopalong (110) | 2D density | faithful | 0.04 | coefficients a, b, c (c = engine power slot); options: a, b, c |
| Martin | `martin` | Martin (111) | 2D density | faithful | 0.04 | coefficient a; options: a |
| Gingerbread man | `gingerbreadman` | Gingerbreadman (112) | 2D density | faithful | 0.06 |  |
| Chip | `chip` | Chip (113) | 2D density | faithful | 0.08 | coefficients a, b, c in -100..100; options: a, b, c |
| Quadruptwo | `quadruptwo` | Quadruptwo (114) | 2D density | faithful | 0.13 | coefficients a, b, c; options: a, b, c |
| Threeply | `threeply` | Threeply (115) | 2D density | faithful | 0.10 | coefficients a, b, c; options: a, b, c |
| Clifford attractor | `clifford` | Clifford (116) | 2D density | faithful | 0.07 | a..d in -3..3; options: a, b, c, d |
| De Jong attractor | `de_jong` | DeJong (117) | 2D density | faithful | 0.04 | a..d in -3..3; options: a, b, c, d |
| Ikeda attractor | `ikeda` | Ikeda (118) | 2D density | faithful | 0.06 | a 0..2, b 0..1, c -10..10, d 0..20; options: a, b, c, d |
| Tinkerbell map | `tinkerbell` | Tinkerbell (119) | 2D density | faithful | 0.03 | a..d in -5..5; options: a, b, c, d |
| Gumowski–Mira | `gumowski_mira` | GumowskiMira (120) | 2D density | faithful | 0.03 | a, b in -1..1; options: a, b |
| Barnsley fern | `barnsley_fern` | BarnsleyFern (121) | 2D density | faithful | 0.29 | deterministic IFS subdivision (draw_affine_ifs), parallel subtrees (verified == serial) |
| IFS dragon | `ifs_dragon` | IFSDragon (122) | 2D density | faithful | 3.00 | same IFS traversal; ~126 M tree nodes at 1024^2 |
| IFS twig | `ifs_twig` | IFSTwig (123) | 2D density | faithful | 0.05 | IFS |
| Christmas tree | `christmas_tree` | ChristmasTree (124) | 2D density | faithful | 0.06 | IFS |
| Vicsek cross | `vicsek_cross` | VicsekCross (125) | 2D density | faithful | 0.05 | IFS |
| Pythagoras tree | `pythagoras_tree` | PythagorasTree (126) | 2D density | faithful | 0.13 | recursive squares (explicit stack) |
| H-tree | `h_tree` | HTree (127) | 2D density | faithful | 0.80 | engine-only type (not in the ImageToolbox UI); defaults from FractalType.kt |
| Heighway dragon | `heighway_dragon` | HeighwayDragon (128) | 2D density | faithful | 0.25 | recursive segments |
| Koch snowflake | `koch_snowflake` | KochSnowflake (129) | 2D density | faithful | 0.02 | recursive segments |
| Hilbert curve | `hilbert_curve` | HilbertCurve (130) | 2D density | faithful | 0.89 | engine-only type; recursion order preserved (consecutive points joined) |
| Barnsley Mandelbrot | `barnsley_mandelbrot` | BarnsleyMandelbrot (131) | escape-time | faithful | 0.26 | options: bailout |
| Barnsley Julia | `barnsley_julia` | BarnsleyJulia (132) | escape-time | faithful | 0.13 | options: bailout, julia_real, julia_imaginary |
| Alpha Mandelbrot | `alpha_mandelbrot` | AlphaMandelbrot (133) | escape-time | faithful | 0.15 | options: bailout |
| Alpha Mandelbrot Julia | `alpha_mandelbrot_julia` | AlphaMandelbrotJulia (134) | escape-time | faithful | 0.06 | options: bailout, julia_real, julia_imaginary |
| Mandelbrot sine | `mandelbrot_sine` | MandelbrotSine (135) | escape-time | faithful | 1.28 | options: bailout |
| Julia sine | `julia_sine` | JuliaSine (136) | escape-time | faithful | 0.95 | options: bailout, julia_real, julia_imaginary |
| Spider | `spider` | Spider (137) | escape-time | faithful | 0.17 | options: bailout |
| Man-o’-War | `man_o_war` | ManOWar (138) | escape-time | faithful | 0.05 | options: bailout |
| Lambda | `lambda` | Lambda (139) | escape-time | faithful | 0.39 | options: bailout |
| Thorn | `thorn` | Thorn (140) | escape-time | faithful | 0.20 | coefficients a, b feed the Julia constant (as toNativeRequest); options: bailout, a, b |
| Barnsley II | `barnsley_ii` | BarnsleyII (141) | escape-time | faithful | 0.11 | options: bailout, julia_real, julia_imaginary |
| Barnsley III | `barnsley_iii` | BarnsleyIII (142) | escape-time | faithful | 0.07 | options: bailout, julia_real, julia_imaginary |
| Mandelbrot cosine | `mandelbrot_cosine` | MandelbrotCosine (143) | escape-time | faithful | 1.31 | options: bailout |
| Julia cosine | `julia_cosine` | JuliaCosine (144) | escape-time | faithful | 2.17 | options: bailout, julia_real, julia_imaginary |
| Mandelbrot sinh | `mandelbrot_sinh` | MandelbrotSinh (145) | escape-time | faithful | 1.23 | options: bailout |
| Julia sinh | `julia_sinh` | JuliaSinh (146) | escape-time | faithful | 1.48 | options: bailout, julia_real, julia_imaginary |
| Feather | `feather` | Feather (147) | escape-time | faithful | 0.52 | options: bailout |
| Cactus | `cactus` | Cactus (148) | escape-time | faithful | 0.10 | options: bailout |
| Zubieta | `zubieta` | Zubieta (149) | escape-time | faithful | 0.10 | options: bailout |
| Tetration | `tetration` | Tetration (150) | escape-time | faithful | 1.87 | exp(z log c); log c hoisted out of the loop (bit-identical); options: bailout |
| Mandelbulb | `mandelbulb` | Mandelbulb (1001) | ray-marched 3D | faithful | 2.48 | power 2..12; options: power, camera_yaw, camera_pitch, camera_distance, show_floor, floor_primary_color, floor_secondary_color |
| Mandelbox | `mandelbox` | Mandelbox (1002) | ray-marched 3D | faithful | 1.01 | scale -1.65 fixed upstream; options: camera_yaw, camera_pitch, camera_distance, show_floor, floor_primary_color, floor_secondary_color |
| Menger sponge | `menger_sponge` | MengerSponge (1003) | ray-marched 3D | faithful | 0.45 | options: camera_yaw, camera_pitch, camera_distance, show_floor, floor_primary_color, floor_secondary_color |
| Sierpinski pyramid | `sierpinski_tetrahedron` | SierpinskiTetrahedron (1004) | ray-marched 3D | faithful | 0.60 | ImageToolbox title 'Sierpinski pyramid'; options: camera_yaw, camera_pitch, camera_distance, show_floor, floor_primary_color, floor_secondary_color |
| Julia set 3D | `quaternion_julia` | QuaternionJulia (1005) | ray-marched 3D | faithful | 0.58 | title 'Julia set 3D'; quaternion option; options: quaternion_x, quaternion_y, quaternion_z, quaternion_w, camera_yaw, camera_pitch, camera_distance, show_floor, floor_primary_color, floor_secondary_color |
| Octahedral IFS | `octahedral_ifs` | OctahedralIFS (1006) | ray-marched 3D | faithful | 0.46 | scale/fold options; options: scale, fold, camera_yaw, camera_pitch, camera_distance, show_floor, floor_primary_color, floor_secondary_color |
| Icosahedral IFS | `icosahedral_ifs` | IcosahedralIFS (1007) | ray-marched 3D | faithful | 0.82 | scale/fold options; options: scale, fold, camera_yaw, camera_pitch, camera_distance, show_floor, floor_primary_color, floor_secondary_color |
| Apollonian gasket | `apollonian_gasket` | ApollonianGasket (1008) | ray-marched 3D | faithful | 0.44 | scale/fold/minimum_radius; options: scale, fold, minimum_radius, camera_yaw, camera_pitch, camera_distance, show_floor, floor_primary_color, floor_secondary_color |
| Kleinian | `kleinian` | Kleinian (1009) | ray-marched 3D | faithful | 1.84 | scale/fold/minimum_radius; fixed epsilon 3.5e-4; options: scale, fold, minimum_radius, camera_yaw, camera_pitch, camera_distance, show_floor, floor_primary_color, floor_secondary_color |
| Hybrid Mandelbulb–Julia | `hybrid_mandelbulb_julia` | HybridMandelbulbJulia (1010) | ray-marched 3D | faithful | 1.67 | power + Julia constant (z = 0.3); options: power, julia_real, julia_imaginary, camera_yaw, camera_pitch, camera_distance, show_floor, floor_primary_color, floor_secondary_color |
| Quaternion cubic | `quaternion_cubic` | QuaternionCubic (1011) | ray-marched 3D | faithful | 0.67 | quaternion option; options: quaternion_x, quaternion_y, quaternion_z, quaternion_w, camera_yaw, camera_pitch, camera_distance, show_floor, floor_primary_color, floor_secondary_color |
| Sierpinski gasket | `sierpinski_gasket` | SierpinskiGasket (1012) | ray-marched 3D | faithful | 0.53 | scale/fold/minimum_radius; options: scale, fold, minimum_radius, camera_yaw, camera_pitch, camera_distance, show_floor, floor_primary_color, floor_secondary_color |
| Pickover | `pickover` | Pickover (1013) | 3D attractor | faithful | 0.03 | density projection through the orbit camera; options: a, b, c, d, camera_yaw, camera_pitch, camera_distance |
| Lorenz | `lorenz` | Lorenz (1014) | 3D attractor | faithful | 0.02 | sigma/rho/beta; options: sigma, rho, beta, camera_yaw, camera_pitch, camera_distance |
| Rössler | `rossler` | Rossler (1015) | 3D attractor | faithful | 0.01 | a/b/c; options: a, b, c, camera_yaw, camera_pitch, camera_distance |

Common options for every type: `coloring` (smooth / banded / orbit_trap / angle — escape-time only, as
upstream), `palette_cycles` (0.05..64), `palette_offset` (wrapped to 0..1), `palette_colors` (custom palette,
2..64 colours), `inside_color` (interior / background), `supersampling` (1..4; "sampling density" for density
types), `iteration_policy` (fixed / scale_with_zoom / adaptive, default adaptive), and for 2D types
`center_x`, `center_y`, `span` (numbers **or decimal strings**, for deep zoom beyond double precision).
3D types additionally take `camera_yaw`, `camera_pitch` (degrees), `camera_distance`, and ray-marched types
`show_floor` (default **on**, as ImageToolbox's `supportsFloor`), `floor_primary_color`,
`floor_secondary_color`.  Coefficient options also accept their slot letter (`a`..`d`) as alias.

## What is reproduced (all "faithful" rows)

* **Kernels** are line-by-line ports of the Rust engine in f64 with the same evaluation order (complex
  arithmetic written component-wise like `math.rs`; `powf` integer fast path; `checked_div`; Rust
  `f64::min/max/clamp/round/rem_euclid/signum` semantics).  Colours are ARGB ints exactly like the engine's
  `u32` (`interpolate_color` with Rust rounding, `shade_color`, `mix_color`), palette sampling with
  `rem_euclid`.  Supersampling averages premultiplied sums and writes the engine's *unpremultiplied* mode.
* **Planning**: ImageToolbox `FractalParams.normalized()` (power 2..12, bailout 2..1e12, cycles, offset wrap,
  span clamp 1e-300..1e6 and the 1e-12 floor unless deep zoom is available, camera yaw wrap / pitch ±89 /
  distance 0.5..20, quaternion ±2, coefficient ranges, floor only for ray-marched types), the adaptive
  iteration policy (`+24` per decade of zoom vs span 3, `+48·ln(pixels / 640·480)`), `resolvePlan`
  (supersampling reduction, ray-march iteration search, Kotlin `Long.toInt()` wrap kept), and the engine's
  `work_plan` (Buddhabrot samples, density steps with the ceil-sqrt resolution multiplier, `ray_march_plan`).
  ImageToolbox refuses ray-marched renders above `maxOutputPixelsFor` (e.g. Mandelbulb ≈ 2.08 MP); LazyRoom
  raises `INVALID_PARAMS` with the limit instead of rendering something ImageToolbox cannot.
* **Wire mapping** (`toNativeRequest`): coefficient `a`/`b`/`c`/`d` replace Julia re / Julia im / power /
  Phoenix re when the formula defines them (e.g. Thorn's a/b become the Julia constant, Lorenz σ/ρ/β), camera
  degrees -> radians, FOV 45°, Lyapunov sequence "AB", camera target 0.
* **Deep zoom**: `PerturbationReference` reproduced with Python big integers (fixed point with
  `-floor(log10 span) + 20` fractional digits, truncating division like num-bigint, reference stop at
  |z| > 1e100) and the same glitch-rebasing perturbation loop.  Exact decimal centres are accepted via
  `options.center_x/center_y/span` strings.
* **Density renderers** (`attractors.rs`): Halton Buddhabrot with main-bulb rejection and `(im, -re)`
  plotting, all 2D maps with their start points / warm-up / divergence break, `draw_affine_ifs`
  subdivision, Pythagoras / H-tree / Heighway / Koch / Hilbert recursions (explicit stacks; Hilbert keeps the
  recursion order because consecutive points are joined), line clipping and plotting, 3D attractor transient
  / scale / depth-weighted 3x3 splats, the log-density palette mapping.
* **Ray marching**: all distance estimators, camera, epsilon schedule (Kleinian fixed 3.5e-4), step clamp,
  central-difference normals, diffuse / specular / rim / step-detail palette position, fog, checkerboard
  floor and its shading.

Pure optimisations that keep pixels identical (verified):
* Interleaved row order for `prange` load balance.
* **Exact-cycle early exit** (Brent) in escape-time iteration for formulas whose step is a pure function of z
  (not Phoenix / Spider / Man-o'-War): when z repeats bit-for-bit the orbit is periodic and can never escape,
  so the pixel is "inside" exactly as after `max_iterations`.  Verified by rendering every escape-time type in
  smooth / orbit-trap / angle colouring with and without the check: 0 differing pixels.
* Tetration's loop-invariant `log(c)` is hoisted.
* Buddhabrot and affine IFS run in parallel with one integer density buffer per thread, summed afterwards
  (integer counts are order independent; IFS parallel result verified equal to the serial traversal for all
  five IFS types).

## Deviations / LazyRoom extensions (not upstream)

| Item | status | notes |
|---|---|---|
| `rotation` | extension | View rotation about the image centre (escape-time: rotated sample deltas; density: rotated plotting with the rotated viewport's bounding box for culling/clipping; 3D: camera-plane roll).  `0` (default, or any multiple of 360) gives the upstream framing bit-for-bit. |
| `zoom` | extension | Convenience: span = default span / zoom (2D), camera distance = default distance / zoom (3D). ImageToolbox zooms interactively; `options.span` / `camera_distance` are the upstream quantities. |
| `center` (top level) | adaptation | Float pair; ignored (with a note) for 3D types whose camera orbits the origin.  Use `options.center_x/center_y` strings for exact decimal centres. |
| `seed` | no effect | Requested parameter; every ImageToolbox fractal is deterministic (Buddhabrot uses a Halton sequence), so it is accepted and ignored (documented in the parameter). |
| `iterations` for every type | extension | ImageToolbox hides the slider for geometric / some 3D formulas but the engine uses the value; LazyRoom exposes it for all types (0 = type default). |
| Validation | adaptation | Out-of-range options raise `INVALID_PARAMS` (LazyRoom convention) where ImageToolbox's `normalized()` would clamp; values that pass validation are normalised exactly like upstream. |
| Transcendentals | approximate (ulp level) | glibc `sin/cos/exp/log/pow/sinh/cosh/acos/atan2` instead of Android bionic/Rust std: last-bit differences may flip an occasional pixel near colour-band boundaries. |
| Output alpha | adaptation | Engine "unpremultiplied" mode written to float RGBA (Android stores premultiplied; identical for the default opaque palettes/inside colour).  RGB is saved when the result is fully opaque. |

## Not ported

* Interactive preview / pan / pinch zoom / camera-orbit gestures and the camera gesture guide (UI only);
  render-session cancellation API (renders are synchronous).
* Nothing of the rendering algorithm is missing.

## Verification

No Rust toolchain is available and the bundled `libfractal_engine.so` is Android-only, so pixels could not
be compared against the native binary.  Instead the engine's own Rust unit tests were ported to
`tests/test_fractals.py` and pass: every stable id renders non-uniform pixels; new types render distinct and
the ten escape types 141..150 differ pairwise; density types are deterministic; Kleinian occupancy
(5–20 %) for engine and app defaults incl. the floor-retention check and the 16-iteration preview;
ray-marched floor opt-in; exact density work plans (`622 408` Buddhabrot samples @1920x1080, `153 600`,
`40 960`, `352 000`); ray-march plans fit 1080² and reject 4K; Buddhabrot ignores bailout while Collatz
does not; deep zoom at 1e-100 is non-uniform and an exact centre 1e-100 away changes pixels; Julia-variant
first-iteration angles; quaternion-Julia direction; Lyapunov derivative order; distance-estimator reference
samples (`quaternion_cubic` 0.021641987147598706, hybrid -0.016340226140528893 / 1.1338524354114028 to 1e-12);
`interpolate_color(black, white, 0.5) == 0xFF808080`.  Plus ImageToolbox planning (adaptive iterations,
supersampling budget, Mandelbulb @1024² -> 53 iterations), tool behaviour and invalid-option errors, and a
visual contact sheet of all 77 types.

## Performance

1024x1024 with ImageToolbox defaults, warm numba cache, 12 threads, other agents running in parallel
(load average 4.5–9.7 during the run, so these are pessimistic):

* 2D escape-time (40 types): median 0.15 s, max 2.17 s (Julia cosine); all < 3 s.
* 2D density (22 types): median 0.07 s, max 3.00 s (IFS dragon: ~126 M subdivision nodes, 16 s CPU over
  12 threads); everything else ≤ 0.9 s.
* Ray-marched 3D (12 types; the ImageToolbox budget gives e.g. Mandelbulb 53 steps x 8 DE iterations at
  1024²): 0.44–2.48 s (Mandelbulb 2.48 s, Kleinian 1.84 s, hybrid 1.67 s, Mandelbox 1.01 s, others < 0.9 s).
* 3D attractors: ≤ 0.03 s.
* First call after installing compiles the numba kernels (~20 s once; cached in `__pycache__`).
