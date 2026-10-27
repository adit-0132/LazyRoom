# ImageToolbox port report: Distortion group

Module: `src/lazyroom/ops/itb/distortion.py` (registrations, G'MIC command builders) and
`src/lazyroom/ops/itb/_distortion_impl.py` (procedural shader engine, GPUImage, jhlabs, aire and
seam-carving ports). Tests: `tests/itb/test_distortion.py`.

**Summary:** 43 of 43 filters ported, all `faithful` (same algorithm). The deviations listed in the notes are
deliberate, mostly about determinism, preview scaling and alpha edge cases.

## Conventions used by this group

* **Coordinate mapping:** every coordinate-mapping filter builds a vectorised inverse map and samples it with
  `cv2.remap`, matching the upstream sampler:
  * GLES `texture2D` (LINEAR + CLAMP_TO_EDGE) → bilinear, `BORDER_REPLICATE` at `uv * size - 0.5`.
  * The procedural `sampleField` mirror fold (`1 - |mod(uv, 2) - 1|` plus a half-texel clamp) is exactly bilinear
    with `BORDER_REFLECT`. A test checks this against an explicit fold.
  * jhlabs Rust sampler → bilinear at pixel coordinates (`x = index`). CLAMP is `BORDER_REPLICATE`.
    RGB_CLAMP uses the clamped colour plus alpha 0 outside the image.
  * aire nearest-neighbour code (truncation, `static_cast<int>`) → exact integer gathers.
* **Alpha:** every op is `alpha="full"`, so alpha moves with the pixels.
  * GPU filters sample premultiplied textures, as Android does, and unpremultiply the result.
  * RGB inputs stay RGB unless the upstream filter itself creates transparency. In that case the op returns RGBA,
    just as the Android bitmap would be transparent there. This applies to:
    * the Sphere/Glass sphere outside area,
    * jhlabs RGB_CLAMP edges (Pinch, Ripple),
    * G'MIC "Transparent" boundaries,
    * Water Reflection.
* **G'MIC:** commands use `GmicAlphaMode.Process` semantics: RGBA always goes in as 4 channels, like Android
  ARGB bitmaps. All definitions used were compared with the stdlib bundled in ImageToolboxLibs (G'MIC 4.0.2,
  extracted from `gmic_stdlib_community.h`) and are identical in the dev CLI (3.7.6). These commands were checked:
  * `fx_crease`, `fisheye`, `fx_flower`/`flower`, `fx_distort_lens`/`undistort`, `fx_powertwirl`,
    `fx_jr_deform`/`jr_deform`, `fx_shockwaves`, `fx_warp_by_intensity`, `fx_reflect`/`water`, `transform_polar`,
    `euclidean2polar`, `autocrop`.

  The ITB Android build patches out G'MIC `smooth` (`disableSmooth`). None of these commands call `smooth`, so
  that patch makes no difference here.

## Filters

| Filter | op id | upstream | status | notes |
|---|---|---|---|---|
| Anaglyph | filter.anaglyph | ImageToolbox | faithful | Canvas emulation of the left/right shifted, REPEAT-tiled shader layers with the colour matrices and PorterDuff ADD (premultiplied). The final `drawBitmap` with the same paint adds G and B a second time, exactly as upstream; drawBitmap ignores the shader. Shift is in px and scaled for previews. |
| BrokenCells | filter.broken_cells | ImageToolbox | faithful | Procedural GLSL port. `hash`/`hash2` are evaluated in float32 like GLSL highp, and a numba 3x3 Voronoi kernel reads a per-cell hash table with exact float32 inputs. A per-pixel numpy fallback handles huge cell ranges. |
| BrokenGlass | filter.broken_glass | ImageToolbox | faithful | Same as BrokenCells, with jitter 0.8, the `+19.3` random hash and rotation of the local vector. |
| Gallium | filter.gallium | ImageToolbox | faithful | Iterative luminance-gradient advection: 4 reflected-border luminance remaps per iteration. This is the slowest procedural filter (about 4 s for 12 MP at the default 5 iterations on a heavily loaded machine). |
| GlassHexTiles | filter.glass_hex_tiles | ImageToolbox | faithful | Procedural GLSL port. |
| GlassRectTiles | filter.glass_rect_tiles | ImageToolbox | faithful | Procedural GLSL port. |
| GlassTriangleTiles | filter.glass_triangle_tiles | ImageToolbox | faithful | Procedural GLSL port. |
| GridSineDistortion2 | filter.grid_sine_distortion2 | ImageToolbox | faithful | Procedural GLSL port. |
| HelixWaves | filter.helix_waves | ImageToolbox | faithful | Procedural GLSL port, including the `shade` lighting term clamped like the 8-bit framebuffer. `mode` stays an Int slider (0..2) as upstream. |
| LogarithmicSpiral | filter.logarithmic_spiral | ImageToolbox | faithful | `sampleInput(fract(uv))`, sampled directly in texture space. |
| Quicksilver | filter.quicksilver | ImageToolbox | faithful | Procedural GLSL port. `sampleField(p)` is the pixel itself, so it is read directly. |
| RadialInterpolate | filter.radial_interpolate | ImageToolbox | faithful | Procedural GLSL port. |
| RadialShimmer | filter.radial_shimmer | ImageToolbox | faithful | Procedural GLSL port. |
| Rubidium | filter.rubidium | ImageToolbox | faithful | Procedural GLSL port. |
| SpiralArms | filter.spiral_arms | ImageToolbox | faithful | `sampleInput(fract(uv))`. |
| SpiralDroste | filter.spiral_droste | ImageToolbox | faithful | Procedural GLSL port. |
| SquareFresnel | filter.square_fresnel | ImageToolbox | faithful | Procedural GLSL port. |
| SquareSpiralDroste | filter.square_spiral_droste | ImageToolbox | faithful | Procedural GLSL port. |
| BulgeDistortion | filter.bulge_distortion | GPUImage | faithful | Exact GLSL math with the aspect-corrected distance and centre fixed at 0.5,0.5. The GPUImageRenderer odd-width padding column, an upstream quirk, is not reproduced; this applies to all four GPUImage filters. |
| GlassSphereRefraction | filter.glass_sphere_refraction | GPUImage | faithful | `2*refract`, mirrored, plus grazing and specular lighting. Output is `(colour, 1) * inside`, so it is transparent outside the sphere. |
| SphereRefraction | filter.sphere_refraction | GPUImage | faithful | `texture(refract) * inside`, transparent outside the sphere. |
| SwirlDistortion | filter.swirl_distortion | GPUImage | faithful | No aspect correction, as upstream. The radius is in texture units, so the swirl region is elliptical on non-square images. |
| Pinch | filter.pinch | jhlabs | faithful | ImageToolboxLibs Rust `PinchFilter` math. Radius is relative to `max(w,h)/2`, and radius 0 uses `min(cx, cy)`. RGB_CLAMP edges are done in straight alpha. Upstream writes RGB_CLAMP pixels into premultiplied memory, which yields bright fringes at partially transparent edges; that artefact is not reproduced. |
| Ripple | filter.ripple | jhlabs | faithful | Sine wave with RGB_CLAMP edges, so borders become transparent as on Android. Amplitudes and wavelengths are in px and scaled for previews. |
| SphereLensDistortion | filter.sphere_lens_distortion | jhlabs | faithful | Rust `SphereFilter` math with CLAMP edges. The default refraction index is 2.5 (Ui default). The data class default of 1.5 is not used by the UI. |
| Twirl | filter.twirl | jhlabs | faithful | CLAMP edges. |
| WaterDrop | filter.water_drop | jhlabs | faithful | Rust `WaterFilter` with CLAMP edges. Wavelength, amplitude and radius are relative to `max(w,h)/2`, as ImageToolbox sets them. |
| Convex | filter.convex | aire | faithful | `ConvexEffect`: truncating nearest-neighbour squeeze, `mix(stackBlur(27), src, easeOutCubic(1 - d/maxD))`. The bezier interpolator is solved exactly (Newton, 13 steps, 1e-3) into a lookup table. Stack blur is computed as its exact separable triangle kernel with clamped edges, so aire's 8-bit mul/shift rounding is not reproduced. Blur radius and distances are scaled for previews. |
| FractalGlass | filter.fractal_glass | aire | faithful | Includes the counter quirk: the first period has dw+1 steps and later periods dw, plus uint8 truncation. |
| Marble | filter.marble | aire | faithful | siv::PerlinNoise `octave2D_01` (4 octaves, z = 0.34567, double precision) is ported to numba and run in threaded row blocks. Upstream seeds the permutation from the wall clock, so it is non-deterministic; here it comes from `ctx.rng`, which makes it deterministic per render seed. Alpha is copied from `(sourceY, x)` as in aire. Noise coordinates are in full-resolution px. |
| PerlinDistortion | filter.perlin_distortion | aire | faithful | As Marble, plus `(src + 127*noise)/1.25`. |
| WaterEffect | filter.water_effect | aire | faithful | ImageToolbox passes `fractionSize = 0.2*size`. Uses truncating nearest-neighbour sampling. |
| Distortion | filter.distortion | OpenCV | faithful | `SeamCarver.distort`: INTER_AREA downscale to at most 512 px, forward-energy seam removal (numba) to `1 - strength%` in width then height, then INTER_CUBIC back to the original size. Gray energy comes from the 8-bit image as upstream. Pixel data stays float instead of 8-bit. |
| Crease | filter.crease | G'MIC | faithful | `fx_crease amp,freq,boundary`. A `srand <seed>` prefix from `ctx.rng` makes it deterministic; upstream noise is unseeded. Amplitude is in px and scaled for previews. |
| Displacement | filter.displacement | G'MIC | faithful | Exact ImageToolboxLibs `Displacement` command string, with aux maps passed via `run_gmic(aux=[h, v or h])`. The map files are `File` params. With no map the input is returned unchanged, as in ImageToolbox. |
| FishEye | filter.fish_eye | G'MIC | faithful | Exact call `fisheye radius,amplitude`. **Upstream quirk:** G'MIC's `fisheye` signature is `center_x,center_y,radius,amplitude`, so the "Radius" and "Amplitude" sliders actually set the centre X%/Y%, with radius 50 and amplitude 1.2. This is kept for fidelity and documented in the parameter descriptions. |
| Flower | filter.flower | G'MIC | faithful | `fx_flower`. |
| LensDistortion | filter.lens_distortion | G'MIC | faithful | `fx_distort_lens`. With the default Transparent boundary the corners become transparent; RGBA is returned only when that happens. |
| PowerTwirl | filter.power_twirl | G'MIC | faithful | `fx_powertwirl`. |
| RandomDeformations | filter.random_deformations | G'MIC | faithful | `fx_jr_deform` with all 8 args. G'MIC ignores the first one (`recompute`). Seeded with `srand`; amplitude is in px and scaled for previews. |
| ShockWaves | filter.shock_waves | G'MIC | faithful | `fx_shockwaves`. All 47 channel modes were verified with the dev CLI. |
| WarpByIntensity | filter.warp_by_intensity | G'MIC | faithful | `fx_warp_by_intensity`. The x/y factors (px per intensity level) are scaled with the preview scale. All 47 channel modes were verified. |
| WaterReflection | filter.water_reflection | G'MIC | faithful | `fx_reflect` with ARGB colour `0x406EA0BE` split to r,g,b,a. **Changes the image size** because `fx_reflect` ends with `autocrop`, so it is registered with `geometry=True`. Seeded with `srand`; wave amplitude and softness are in px and scaled for previews. |

## Notes for the lead

* **G'MIC numbers:** they are formatted by `gmic_cmd`, which writes `30` rather than Kotlin's `30.0`. The value is
  the same; every default command was printed and checked:
  * `fx_crease 30,10,3`
  * `fisheye 70,1`
  * `fx_flower 50,50,75,50,6,0,3`
  * `fx_distort_lens 0.1,0,0,50,50,0`
  * `fx_powertwirl 1,0,50,50,1,3,0`
  * `fx_jr_deform 0,5,1,10,1,0,0,1`
  * `fx_shockwaves 10,10,20,0`
  * `fx_warp_by_intensity 0.04,0.04,128,128,0,1,3,0`
  * `fx_reflect 50,1,110,160,190,64,0,1.5,0,-3.3,7,1.5`
* **Choice names:** these are snake_case of the upstream enum names. For example `YCbCr` becomes `y_cb_cr`,
  `NoneOption` becomes `none`, and `SpreadNoise` becomes `spread_noise`. The `channel` choices carry the upstream
  names as labels.
* **Numba:** used with `cache=True, nogil=True` for the Voronoi kernel, the Perlin noise and the seam carving. No
  numba `parallel=True` is used, to avoid the workqueue threading layer being unsafe across threads. Parallelism
  comes from a `ThreadPoolExecutor` over row strips, since numpy, OpenCV and nogil numba all release the GIL. The
  first call of each kernel compiles it, which takes about 2 s; after that it comes from the cache.
* **Speed at 12 MP (4000x3000):** measured while the machine was heavily loaded (load average about 20 on 12
  cores):
  * Every non-G'MIC filter takes 0.3–2.3 s wall, except Gallium at about 4 s.
  * The G'MIC filters take 4–19 s, which is the external CLI's own time.


## Lead decisions

* **FishEye** — fixed to `fisheye 50,50,radius,amplitude`; upstream passes the slider values as the centre position, so its sliders did not control radius/strength. Status: approximate (bug fixed).
