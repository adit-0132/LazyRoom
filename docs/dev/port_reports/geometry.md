# Port report: ImageToolbox "Geometry" group (30 filters)

Module: `src/lazyroom/ops/itb/geometry.py`, helpers: `src/lazyroom/ops/itb/_geometry_impl.py`,
tests: `tests/itb/test_geometry.py`.

Summary: **29 faithful, 1 approximate, 0 not ported.**

| Filter | op id | upstream | status | notes |
|---|---|---|---|---|
| Arc | `filter.arc` | jhlabs | faithful | Formula of the ImageToolboxLibs Rust port (`ArcFilter`), edge ZERO (outside transparent), bilinear. Radius/height scaled by max(w,h)/2 like ITB. |
| AutoCrop | `filter.auto_crop` | OpenCV | faithful | opencv-tools `AutoCropper`: RGBA2GRAY, median 3, Canny(100,200), row-mean threshold `(10-s)*20.5+50`, Sobel(CV_16U,2,2,5) band score, crop rows `top+1 .. bottom-1`. Returns the input when upstream would return null/throw (no candidates / no positive score). |
| CircleTransform | `filter.circle_transform` | G'MIC | faithful | `fx_circle_transform`, exact argument order/enums; GmicAlphaMode.Process (RGBA always passed). |
| CircularMirror | `filter.circular_mirror` | ImageToolbox | faithful | Procedural GLSL shader (`ProceduralShaders.kt`) evaluated per pixel in numba (float64; GPU is float32), sampled with GL_LINEAR + half-texel clamp + mirrored repeat. |
| ConformalMap | `filter.conformal_map` | G'MIC | faithful | `fx_conformal_maps` with the builder's fixed `"z"`, `0,0,0` tail; Process alpha. |
| CropOrPerspective | `filter.crop_or_perspective` | OpenCV | faithful | AutoStraighten Manual -> FreeCrop: output = min opposite edge lengths (truncated), `warpPerspective` INTER_LINEAR, transparent border. Absolute corners are full-res px (scaled with `ctx.px`). |
| CropToContent | `filter.crop_to_content` | Trickle | faithful | Native `cropToContentImpl`, including its quirks: compares the pixel's B,G,R bytes with the colour's R,G,B (little-endian RGBA read), works on premultiplied values, and copies only `right-left` columns so the last output column is transparent. Returns input when nothing differs. |
| Deskew | `filter.deskew` | OpenCV | faithful | AutoStraighten Deskew: BGR2GRAY on the RGBA mat (as upstream), fastNlMeans(h=3), Otsu inv, HoughLinesP(1, 1deg, 200, w/12, w/150), median angle, 90deg pre-rotation for "landscape", warpAffine BORDER_REPLICATE, optional largest-rotated-rect crop (guarded against empty rects). fastNlMeans dominates runtime on large images (as upstream). |
| DistortPerspective | `filter.distort_perspective` | jhlabs | faithful | `PerspectiveFilter.setCorners` (scaled mode) of the Rust port: inverse of `unitSquareToQuad` (float32), offset = int(min corner), edge RGB_CLAMP. `clip` is exposed but, like the native port, has no effect in corner mode. Same canvas size. |
| Droste | `filter.droste` | G'MIC | faithful | `fx_droste`, Process alpha. |
| EquirectangularProjection | `filter.equirectangular_projection` | G'MIC | faithful | `fx_equirectangular2nadirzenith`, Process alpha (marked geometry: size may change). |
| ExpandImage | `filter.expand_image` | OpenCV (opencv-tools, pure Kotlin) | faithful | `ImageExpander` patch-based texture synthesis ported line by line to numba (placements, search area, anchor + random candidates, weighted ARGB error, seam blending, projection fill). `kotlin.random.Random(seed)` (XorWow + `nextInt(from, until)`) reproduced and verified bit-exact against the real kotlin-stdlib. Float32 `mixColors` truncation reproduced (it can leave alpha 254 in blended pixels, as upstream). |
| Feedback | `filter.feedback` | jhlabs | faithful | Rust `FeedbackFilter`: 3 iterations, bilinear ZERO-edge sample of the original, `composite_normal` with extra alpha `255/(i+1)` (integer division). Distance is px (scaled). |
| HexKaleidoscope | `filter.hex_kaleidoscope` | ImageToolbox | faithful | Procedural shader (see CircularMirror). |
| InversionFractal | `filter.inversion_fractal` | ImageToolbox | faithful | Procedural shader (see CircularMirror). |
| Kaleidoscope | `filter.kaleidoscope` | jhlabs | faithful | Rust `KaleidoscopeFilter`, CLAMP edge. Upstream quirk kept: ITB passes the 0..360 "angle"/"secondary angle" sliders to jhlabs unconverted, i.e. they act as radians. Radius is px (scaled). |
| LensCorrection | `filter.lens_correction` | OpenCV | approximate | Algorithm is the opencv-tools one (Gyroflow-style JSON profile, K scaled by image/calib size, `D = -coeffs * strength`, `cv2.fisheye.undistortImage` with Knew=K, opaque output). Deviation: ImageToolbox falls back to a bundled sample profile when no file is chosen; LazyRoom bundles no profiles (unlicensed source), so with no `lens_profile` file and no explicit `camera_matrix`/`distortion_coeffs` the image is returned unchanged. Extra params `camera_matrix` (9), `distortion_coeffs` (4), `calib_width/height` allow using a profile without a file. Processing is float instead of 8-bit RGB. |
| Mirror | `filter.mirror` | ImageToolbox | faithful | Canvas implementation: keep `min(center, w-center)` px on the chosen side and append its flip (output size changes). |
| Offset | `filter.offset` | jhlabs | faithful | Wrap offset by `round(v*size)` px = exact circular shift (`np.roll`). |
| PoincareDisk | `filter.poincare_disk` | G'MIC | faithful | `fx_poincare_disk` incl. colours as 0..255 ints; Process alpha; output is square (geometry). |
| PolarCoordinates | `filter.polar_coordinates` | jhlabs | faithful | Rust `PolarFilter` (3 types), CLAMP edge. |
| PolarTransform | `filter.polar_transform` | G'MIC | faithful | `fx_transform_polar` with the 4+4 expression presets (choices `wavy/identity/inverse/angle` and `identity/mirror/radius/spiral`, labels show the expressions). Spaces are removed from the quoted expressions because the G'MIC CLI re-quotes argv items containing spaces (which broke the command); spaces are insignificant in G'MIC math. |
| ProgressiveScaling | `filter.progressive_scaling` | ImageToolbox | faithful | Procedural shader (see CircularMirror). |
| RandomKaleidoscopeGrid | `filter.random_kaleidoscope_grid` | ImageToolbox | faithful | Procedural shader; the shader `hash()` is evaluated in float32 like the GPU (FMA contraction on a device could still differ in rare cells). |
| SeamCarving | `filter.seam_carving` | OpenCV | faithful | opencv-tools `SeamCarver.carve` in numba: forward energy (default) or Sobel backward energy, hard mask protection with soft (+1e9) fallback, removal mask (-1e9) mode, batched insertion (15% of width) with neighbour-averaged pixels, width then height via transpose. Target size 0 (either side) = unchanged, like `IntegerSize.isZero()`. Minor: the mask is carried as its gray/binary channel, so inserted mask pixels average gray values instead of RGBA bytes (only matters for near-black coloured masks). `mask`/`use_mask_as_removal` exist in the upstream params but not in ITB's UI (marked advanced). |
| Shear | `filter.shear` | jhlabs | faithful | Rust `ShearFilter` (same canvas size, RGB_CLAMP edge). `resize` exposed but ignored, like the native port. |
| SmoothKaleidoscope | `filter.smooth_kaleidoscope` | ImageToolbox | faithful | Procedural shader (4 weighted mirrored samples). |
| SquareToCircle | `filter.square_to_circle` | G'MIC | faithful | `fx_square_circle_alt`, Process alpha. |
| StereographicProjection | `filter.stereographic_projection` | G'MIC | faithful | `fx_project_stereographic` incl. the quoted `"cx,cy"` tail; square output (geometry). When the radius point equals the centre G'MIC's maths is undefined and the command errors (also in 4.0.2): the input is returned unchanged instead. |
| Tunnel | `filter.tunnel` | G'MIC | faithful | `fx_tunnel`; GmicAlphaMode.Preserve (RGB processed, source alpha re-attached). |

## Notes

* **G'MIC alpha modes.** ImageToolboxLibs' JNI bridge always passes RGBA in Process mode
  (`gmicProcessAlpha`) and, in Preserve mode, passes RGB and *discards* any alpha produced by
  G'MIC (source alpha is bilinearly resampled when the size changes). `_geometry_impl.gmic_apply`
  reproduces both; `run_gmic(preserve_alpha=True)` alone differs (it multiplies by G'MIC's alpha).
* **G'MIC version.** The dev CLI is 3.7.6, ImageToolbox bundles 4.0.2. I decompressed ITB's
  `gmic_stdlib_community.h` and the stdlib embedded in `libgmic.so.376`: the definitions of all
  17 `fx_*` commands used by geometry+frames (and their main helpers: `transform_polar`, `tunnel`,
  `frame_*`, `polaroid`, `drop_shadow`, `vignette`, `remove_pixels`, `blend`, ...) are identical.
* **Determinism.** Every G'MIC pipeline is prefixed with `srand <ctx.rng>` (G'MIC is clock-seeded).
* **Performance** (12 MP, machine at load avg 10-20 during measurement, so idle numbers are
  lower): procedural shaders 0.8-3 s (numba kernels emit remap coordinates, cv2.remap samples);
  jhlabs transforms 1-4 s (row-banded maps); ExpandImage ~1.4 s; SeamCarving ~0.15 s per seam;
  Deskew is dominated by `fastNlMeansDenoising` (as upstream).
* Size-changing ops (`geometry=True`, `alpha="full"`): AutoCrop, CropOrPerspective, CropToContent,
  Deskew, ExpandImage, Mirror, SeamCarving, PoincareDisk, StereographicProjection,
  EquirectangularProjection.  Same-size ops that create transparency use `alpha="full"` and return
  RGBA only when transparency was actually produced (or the input had alpha).
