# Port report: ImageToolbox "Blur" group (36 filters)

Module: `src/lazyroom/ops/itb/blur.py` (registrations), kernels in
`src/lazyroom/ops/itb/_blur_impl.py`, tests in `tests/itb/test_blur.py`.

## Sources actually read

* ImageToolbox `Ui*Filter.kt` (ranges/defaults), `data/model/*Filter.kt`,
  `EnumMappings.kt`, `AireConvolution.kt`, `Rounding.kt`, param items
  (`BilaterialBlurParamsItem.kt` ...) for the UI -> data-class mapping.
* aire 0.18.1 (the version pinned in ImageToolbox `libs.versions.toml`; the local
  clone is 0.18.1 + a CI-only commit): Kotlin pipelines + JNI argument order,
  C++ (`PoissonBlur.cpp`, `Convolve1D.cpp`, `GaussBlur.cpp` 4D,
  `ZoomBlur.hpp`, `TiltShift.cpp`, `EigenUtils.h` bokeh kernel,
  `ConvolveKernels.kt`) and the Rust JNI layer `aire-filters/src/{blur,morph}.rs`.
* libblur **0.19.3** (version in aire-filters' `Cargo.lock`; downloaded from
  crates.io because aire builds it from a local path) and fast_morphology 0.3.1.
* Trickle C++ (`Utils.cpp` fast/stack blur, `ProcessingFilters.cpp` shuffle blur,
  `ColorUtils.cpp`).
* ImageToolboxLibs: the **native Rust jhlabs port** (`jhlabs/src/main/rust/jhlabs/src/lib.rs`
  — `JhFilter.filter` always goes through it, not through the Java classes),
  GPUImage `GPUImageZoomBlurFilter`, G'MIC `FrostedGlass.kt`.

## Conventions used for the whole group

* **Alpha**: Android bitmaps are premultiplied and every native implementation works
  on the raw premultiplied RGBA memory, so all ops are `alpha="full"`: RGBA input is
  premultiplied, all four channels are blurred, then un-premultiplied (alpha is
  blurred like in ImageToolbox). Filters that skip the alpha channel upstream
  (aire 4D gaussian, Trickle stack blur) blur premultiplied RGB only and keep alpha.
* **Precision**: float32/float64 maths instead of upstream's 8-bit (or 16-bit for
  the linear variants) storage; no intermediate quantisation and no fixed-point
  kernel weights (`ConvolutionMode::FixedPoint`), no `+0.5`/truncation biases of the
  integer paths. Kernels, window alignment and edge rules are reproduced exactly.
* **Edge modes** (`BlurEdgeMode` -> choices `clamp, wrap, reflect, reflect_101`, labels
  Clamp / Wrap / Mirror / Mirror 101): libblur's index formulas are reproduced
  (`edge_index`), including its non-periodic behaviour when a kernel is larger than
  the image; OpenCV's own border handling is only used where it is provably identical.
* **Transfer functions** (`TransferFunc` -> `srgb, rec709, gamma2p2, gamma2p8`):
  libblur's exact curves (its sRGB/Rec.709 constants, pure gamma), applied through
  65536-entry tables like libblur; alpha stays linear (`may_have_alpha = true`).
* **Kotlin rounding** (`roundToInt` = half up, `roundToNearestOdd`) and upstream
  clamps (libblur radius limits 319/280/255/152/1449) are applied on full-resolution
  values; pixel parameters are then scaled with `ctx.px` (`unit="px"`).
* Randomness (Poisson kernel, shuffle blur, G'MIC frosted glass) is seeded from
  `ctx.rng` (upstream is clock-seeded).

## Filters

| Filter | op id | upstream | status | notes |
|---|---|---|---|---|
| AdaptiveBlur | `filter.adaptive_blur` | jhlabs | faithful | Native jhlabs `SmartBlurFilter`: gaussian sigma=r/3, taps +-r, per-channel inclusion if \|c-n\| <= threshold (0..255), out-of-image taps use the centre pixel; horizontal then vertical (numba). UI defaults (15, 35) used (impl class default differs). |
| BilaterialBlur | `filter.bilaterial_blur` | aire | faithful | libblur `bilateral_filter`: square (2r+1)^2 window, separable spatial gaussian, **per-channel** range weights on 0..1 values, edge modes via arena. aire passes the two sigmas **swapped** (JNI argument order), so UI "Sigma" (default 10) is the range sigma and UI "Spatial Sigma" (default 3) the spatial one — reproduced. Evaluated exactly (|err| < 1e-6) with a power series in the centre value: one separable convolution per term instead of O(r^2) per pixel. |
| Bokeh | `filter.bokeh` | aire | faithful | `getBokehEffect` polygon (Bresenham outline + 4-connected flood fill, radius 0.85*size/2) + `morphology(DILATE)`, REFLECT_101; RGB mode for opaque input. Size < 3 is rejected by aire (exception) -> returned unchanged. |
| BoxBlur | `filter.box_blur` | aire | faithful | libblur `box_blur(2r+1)`: note libblur's window is **2r samples [x-r+1, x+r]** (half-pixel shift), clamped edges; reproduced with an anchored `cv2.blur`. Upstream's multi-threaded ring-buffer path mis-initialises the first rows of each horizontal tile (device-dependent seams) — not reproduced. |
| CircleBlur | `filter.circle_blur` | aire | faithful | `ConvolveKernels.circle` (disc of radius size/3) via `convolve2D` REFLECT_101. Size 3 throws in aire (needs >= 5) -> returned unchanged. |
| CrossBlur | `filter.cross_blur` | aire | faithful | `ConvolveKernels.cross` (main diagonal only, distance < size/2+1). |
| Diffuse | `filter.diffuse` | jhlabs | faithful | Native port: offset from the integer hash `(x*0x9e3779b1) ^ (y*0x85ebca77)` (angle = low byte, distance = next 16 bits * scale), bilinear, CLAMP edges. Bilinear via `cv2.remap` (1/32 px sub-pixel precision). UI default 50 used. |
| EnhancedZoomBlur | `filter.enhanced_zoom_blur` | aire | faithful | aire `ZoomBlur`: 2r+1 gaussian-weighted **nearest (truncated)** samples at p + m*(c-p)*strength/100 * (cos a, sin a), clamped. Taps < 1e-9 of the peak weight are skipped (no visible effect). |
| FastBilaterialBlur | `filter.fast_bilaterial_blur` | aire | faithful | libblur `fast_bilateral_filter` (bilateral grid, one plane per channel) ported as written, including its quirks: the grid convolution pairs taps off by one (right half shifted), and `trilinear_interpolation` always has zero fractions so slicing is a floor-cell lookup (visibly blocky, like upstream). UI "Sigma" = grid spatial sigma, UI "Spatial Sigma" = range sigma. Parameters are in grid units (not px-scaled). |
| FastBlur | `filter.fast_blur` | Trickle | approximate | Trickle `fastBlurImpl`: bilinear downscale by `scale`, symmetric (2r+1) box (clamped), upscale; output alpha forced to 255 like upstream. Deviations: Android `createScaledBitmap(filter=true)` resampling approximated by `cv2.INTER_LINEAR`; upstream scales back to `(int)(w/scale)` which can differ from the input by a pixel — we always return the input size. |
| FastGaussianBlur2D | `filter.fast_gaussian_blur_2d` | aire | faithful | libblur `fast_gaussian` (2nd-degree integrated box, kernel = box_r * box_r / r^2) ported literally in numba, vertical then horizontal, **in place**, so the reflect/wrap read-ahead of already-blurred samples near the bottom/right edges is reproduced. Radius clamp 319. |
| FastGaussianBlur3D | `filter.fast_gaussian_blur_3d` | aire | faithful | libblur `fast_gaussian_next` (3rd degree, read offset 3r/2), literal in-place port; libblur clamps the radius to 280 (UI allows 500). |
| FastGaussianBlur4D | `filter.fast_gaussian_blur_4d` | aire | faithful | aire C++ `gaussianApproximation4D` (4th degree, clamped), literal port; alpha channel is not blurred upstream (kept). |
| FrostedGlass | `filter.frosted_glass` | G'MIC | faithful | Exact `fx_textured_glass x_amp,y_amp,x_smooth,y_smooth,edge_att,edge_infl,scale` (alpha processed, `GmicAlphaMode.Process`); prefixed with `srand <ctx seed>` for determinism. Amplitudes / smoothness / edge influence are pixel values and are scaled with `ctx.px` for previews. Verified with the dev CLI. |
| GaussianBlur | `filter.gaussian_blur` | aire | faithful | libblur `gaussian_blur`: sampled, normalised gaussian of 2r+1 taps (separable), selected edge mode. Fixed-point weights not reproduced (float). Radius 0 -> identity. |
| GaussianBoxBlur | `filter.gaussian_box_blur` | aire | faithful | libblur `gaussian_box_blur`: 3 libblur box passes sized by `create_box_gauss` (ported with f32 maths, incl. its "make the half-width odd" rule). |
| LensBlur | `filter.lens_blur` | jhlabs | faithful | ImageToolbox's native jhlabs `LensBlurFilter` is **not** the Java FFT polygon-aperture filter: it multiplies pixels with brightness ((r+g+b)/3) > threshold by `bloom`, then applies the jhlabs gaussian (sigma=r/3, clamped). `sides` is ignored upstream (kept as a parameter for parity). Upstream premultiplies the already premultiplied buffer once more; we use a single premultiplication (only matters for RGBA). |
| LinearBoxBlur | `filter.linear_box_blur` | aire | faithful | linearize -> libblur box (same 2r window as BoxBlur) -> gamma. |
| LinearFastGaussianBlur | `filter.linear_fast_gaussian_blur` | aire | faithful | linearize -> `fast_gaussian_u16` (radius clamp 255) -> gamma. |
| LinearFastGaussianBlurNext | `filter.linear_fast_gaussian_blur_next` | aire | faithful | linearize -> `fast_gaussian_next_u16` (radius clamp 152) -> gamma. |
| LinearGaussianBlur | `filter.linear_gaussian_blur` | aire | faithful | kernel size = nearest odd of "strength", libblur gaussian kernel, edge mode, transfer function. Tail taps below 1e-9 of the peak are trimmed. |
| LinearGaussianBoxBlur | `filter.linear_gaussian_box_blur` | aire | faithful | linear-light `gaussian_box_blur`. |
| LinearStackBlur | `filter.linear_stack_blur` | aire | faithful | linear-light libblur stack blur (triangular kernel). |
| LinearTentBlur | `filter.linear_tent_blur` | aire | faithful | linear-light `tent_blur` with sigma = nearest odd. |
| LinearTiltShift | `filter.linear_tilt_shift` | aire | faithful | aire `horizontalTiltShift`: blur = aire `gaussBlurU8` with **kernel size = blur radius** (even sizes give the asymmetric aire kernel), blend factor clamp((d/(extent*size))^3) with d along the rotated axis (f32 maths, so the 45 deg default resolves to the same branch as upstream), mix weighted by blurred alpha, source alpha kept. |
| MedianBlur | `filter.median_blur` | aire | faithful | aire passes `2r+1` as libblur's *radius*, so the window is (4r+3) rows. Literal numba port of libblur's per-row sliding 8-bit histograms with shrinking windows at the borders, including the upstream quirk that column `R` is never added (so from x >= 2R the histogram carries negative counts). 8-bit quantised like upstream. |
| MotionBlur | `filter.motion_blur` | aire | faithful | libblur `generate_motion_kernel` (f32 maths, saturating casts, Bresenham) + `filter_2d` correlation with the edge mode. Size 0 -> 1 (nearest odd). |
| NativeStackBlur | `filter.native_stack_blur` | aire | faithful | libblur `stack_blur`: weights (r+1-\|i\|)/(r+1)^2, clamped edges — computed exactly as two anchored box passes on an edge-padded buffer. |
| PoissonBlur | `filter.poisson_blur` | aire | faithful | aire `poissonBlur`: 1D kernel of 2r+1 Poisson(lambda=2r+1) samples, normalised, separable `convolve1D` with clamped edges. Same distribution, seeded from `ctx.rng` instead of the clock. |
| RadialTiltShift | `filter.radial_tilt_shift` | aire | faithful | aire `tiltShift`: same blur as above, factor clamp((dist/(diagonal*radius))^3). |
| RingBlur | `filter.ring_blur` | aire | faithful | `ConvolveKernels.ring` measures the distance from (size, size) — outside the kernel — so the "ring" is a quarter arc in the bottom-right; reproduced as is. |
| ShuffleBlur | `filter.shuffle_blur` | Trickle | faithful | Trickle `shuffleBlurImpl`: threshold on linear luma (Trickle's 0.045 sRGB knee, f32/f64 mix reproduced), random source from a window that grows with the pixel position (`rand() % (y+1) * strength`), alpha from the current pixel; non-matching pixels stay transparent (output gets an alpha channel then). `rand()` replaced by a counter-based generator seeded from `ctx.rng`. Strength table = `radiusMapping` in Kotlin float maths. |
| StackBlur | `filter.stack_blur` | Trickle | approximate | Trickle `stackBlurImpl`: bilinear downscale, Klingemann stack blur on RGB (alpha untouched), upscale. Same resampling / output-size caveats as FastBlur. |
| StarBlur | `filter.star_blur` | aire | faithful | `ConvolveKernels.star` (centre row + centre column). |
| TentBlur | `filter.tent_blur` | aire | faithful | libblur `tent_blur(nearest odd sigma)`: 2 box passes from `create_box_gauss`. UI default 10 (impl class default 15). |
| ZoomBlur | `filter.zoom_blur` | GPUImage | faithful | `GPUImageZoomBlurFilter`: 9 taps (0.18, 0.15, 0.12, 0.09, 0.05 symmetric) at uv +- k * (center - uv) * size / 100, bilinear with clamp-to-edge (numba). |

**Totals: 34 faithful, 2 approximate (FastBlur, StackBlur — Android bitmap scaling), 0 not ported.**

## Things worth knowing

* Op ids come from the shared `snake()`: `FastGaussianBlur2D` -> `filter.fast_gaussian_blur_2d`
  (same for 3D/4D). Left as-is to follow the contract; a nicer id would need a change in
  `_common.snake` or an explicit `op_id`.
* `ShuffleBlur` can return RGBA for an RGB input (pixels outside the threshold become
  transparent, as in ImageToolbox); with the default threshold (1.0) every pixel matches and
  RGB is returned.
* Several ImageToolbox labels are misleading because of upstream argument swaps
  (bilateral sigmas, fast-bilateral "Spatial Sigma" = range sigma); parameter
  descriptions document the actual effect.
* Performance (12 MP, defaults, measured on a heavily loaded 12-core box, so roughly 2x
  pessimistic): most filters 0.3–2.5 s; slowest are BilaterialBlur (~3.5–7 s, exact
  evaluation), MedianBlur (~4 s, O(radius) per pixel like upstream), EnhancedZoomBlur
  (~2–3 s) and FrostedGlass (G'MIC, ~3.5 s). Numba kernels are `cache=True`; the first
  call in a fresh environment pays the JIT compile. Extreme parameters (601-tap linear gaussian,
  r=100 median/zoom, 150 px bokeh) are proportionally slower, as upstream.
