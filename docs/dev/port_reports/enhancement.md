# Port report: Enhancement group (33 filters)

Module: `src/lazyroom/ops/itb/enhancement.py`, helpers `src/lazyroom/ops/itb/_enhancement_impl.py`,
tests `tests/itb/test_enhancement.py`, bundled model `src/lazyroom/data/models/face_detection_yunet_2026may.onnx`.

**Summary: 31 faithful, 2 approximate, 0 not ported.**

## G'MIC filters (25)

Every command string is built with the same argument order, enum `value` integers and 0/1 booleans as the
ImageToolboxLibs builder (`com.t8rin.gmic.filters.*`); defaults/ranges come from the `Ui*Filter` classes.
`test_gmic_default_commands` pins all 25 default command strings. All 25 were run with the dev CLI (3.7.6) at
defaults, at every numeric min/max and with every enum choice (0 failures), and are deterministic (no `srand`
needed). The stdlib definitions of all 25 commands and their `_`-helpers are identical in G'MIC 3.7.6 (dev CLI)
and 4.0.2 (ImageToolbox). Pixel-distance arguments (marked `unit="px"`) are scaled by `ctx.px/ipx`; at full
resolution the command is byte-for-byte the ImageToolbox one.

| Filter | op id | upstream | status | notes |
|---|---|---|---|---|
| AlphaSharpen | `filter.alpha_sharpen` | G'MIC | faithful | `fx_sharpen_alpha` |
| BandingDenoise | `filter.banding_denoise` | G'MIC | faithful | `banding_denoise_v2`; `space` (bilateral spatial) scaled for previews |
| CleanText | `filter.clean_text` | G'MIC | faithful | `afre_cleantext` |
| ConstrainedSharpen | `filter.constrained_sharpen` | G'MIC | faithful | `iain_constrained_sharpen`; own 32-entry channel enum |
| DcpDehaze | `filter.dcp_dehaze` | G'MIC | faithful | `jeje_dehaze`; `scale` (window radius) scaled for previews |
| Deblur | `filter.deblur` | G'MIC | faithful | `fx_deblur` |
| DenoiseSmooth | `filter.denoise_smooth` | G'MIC | faithful | `afre_denoisesmooth` (note: the G'MIC command itself desaturates and barely smooths at defaults) |
| Descreen | `filter.descreen` | G'MIC | faithful | `fx_pahlsson_descreen` |
| DetailsEqualizer | `filter.details_equalizer` | G'MIC | approximate | `fx_equalize_details`. (1) ImageToolbox's native bridge runs with `GmicExecutionOptions.disableSmooth = true`, which skips G'MIC's built-in `smooth`; the "Diffusion" band smoothing is therefore a no-op in ImageToolbox but applied here (identical for smoothness 0, the default, and for Gaussian/Bilateral). (2) G'MIC's parallel tiling fails on small images (tiny tiles: `_ac: Invalid assignment request of shared instance`); the op then retries with parallelism `OneThread` (tiling affects only speed). Images below ~64x48 can still fail inside `split_details`. |
| FreakyDetails | `filter.freaky_details` | G'MIC | faithful | `fx_freaky_details` |
| GoldMeinelSharpen | `filter.gold_meinel_sharpen` | G'MIC | faithful | `fx_unsharp_goldmeinel` |
| GradientSharpen | `filter.gradient_sharpen` | G'MIC | faithful | `gcd_sharpen_gradient` |
| HessianSharpen | `filter.hessian_sharpen` | G'MIC | faithful | `jeje_hessian_sharpen` (`repeat` is a float like upstream) |
| LocalVarianceNormalization | `filter.local_variance_normalization` | G'MIC | faithful | `jeje_normalize_local_variance` |
| MagicDetails | `filter.magic_details` | G'MIC | faithful | `fx_magic_details` |
| MightyDetails | `filter.mighty_details` | G'MIC | approximate | `fx_mighty_details` runs as G'MIC intends. Its first step is `+smooth`; with ImageToolbox's disabled `smooth` no copy is created and the next `-[1] [0]` fails ("Invalid selection [1]", reproduced with the CLI), so the filter errors in ImageToolbox. |
| MultiscaleSharpen | `filter.multiscale_sharpen` | G'MIC | faithful | `fx_sharpen_multiscale` |
| RemoveHotPixels | `filter.remove_hot_pixels` | G'MIC | faithful | `remove_hotpixels` |
| RemoveScratches | `filter.remove_scratches` | G'MIC | faithful | `fx_remove_scratches` |
| RichardsonLucySharpen | `filter.richardson_lucy_sharpen` | G'MIC | faithful | `fx_unsharp_richardsonlucy` |
| SmoothSkin | `filter.smooth_skin` | G'MIC | faithful | `fx_smooth_skin` |
| TextureSharpen | `filter.texture_sharpen` | G'MIC | faithful | `fx_sharpen_texture` |
| ToneSharpen | `filter.tone_sharpen` | G'MIC | faithful | `gcd_sharpen_tones` |
| Unstrip | `filter.unstrip` | G'MIC | faithful | `jeje_unstrip` |
| WhitenSharpen | `filter.whiten_sharpen` | G'MIC | faithful | `jeje_whiten_frequency` |

## Native filters (8)

| Filter | op id | upstream | status | notes |
|---|---|---|---|---|
| AnisotropicDiffusion | `filter.anisotropic_diffusion` | aire | faithful | `blur/AnisotropicDiffusion.cpp`: in-place 8-bit update from the 4 *diagonal* neighbours, `v + v*diffusion*sum(g(d)*d)`, `ceil`. Upstream updates 4 row segments concurrently in place (a data race); reproduced deterministically with the most likely interleaving (segment boundary rows see old/new neighbours). Bit-exact against a literal transcription (test). numba, ~3 s for 20 iterations at 12 MP. |
| AutoRemoveRedEyes | `filter.auto_remove_red_eyes` | OpenCV | faithful | `opencv-tools RedEyeRemover`: YuNet `FaceDetectorYN` (same bundled model, thresholds 0.7/0.3/5000, full-resolution input), eye boxes `max(35, 0.16*min(face w,h))` around the eye landmarks, mask `R > threshold && R > sat(B+G)`, `fillHoles` (flood fill from the box corner), 3x dilate, replace with `round(sat(B+G)/2)` grey. Upstream quirk kept: if the box corner is itself "red" (very red skin) the whole box turns grey. Deviations: alpha is preserved (upstream converts RGBA->BGR->RGBA, making the image opaque); the 35 px minimum eye size follows the preview scale. |
| AutoWhiteBalance | `filter.auto_white_balance` | Trickle | faithful | `AutoWhiteBalanceFilter.cpp`: per-channel histograms (alpha 0 pixels ignored and left untouched), `cut = n*clip/100`, low/high percentiles exactly as upstream; the 8-bit LUT is applied as the continuous stretch it samples (< 1 8-bit level difference, keeps float precision). Channel-symmetric, so Trickle's R/B byte-order swap does not matter. |
| Dehaze | `filter.dehaze` | aire | faithful | `DehazeDarkChannel.cpp`, bit-exact against a literal transcription: dark channel including the upstream SIMD bug (16-pixel blocks use `MinOfLanes(r)` three times, i.e. red only; NEON 16 lanes assumed), atmospheric light = mean of the top 1 %, `t = max(1-w*dark/A, 1-w*min(rgb)/A)`, truncating cast. |
| HighPass | `filter.high_pass` | jhlabs | faithful | The ImageToolboxLibs jhlabs port runs natively (Rust `lib.rs`): `(c + 255 - gaussian(c)) / 2` with `gaussian_kernel_exact` (sigma = r/3, taps beyond r zeroed), clamp edges, premultiplied blur, alpha kept. Computed in float (no 8-bit intermediate rounding). |
| Kuwahara | `filter.kuwahara` | GPUImage | faithful | `GPUImageKuwaharaFilter` shader: hard-coded texel size `(1/768, 1/1024)` (sample spacing = width/768 x height/1024 px, independent of resolution), bilinear clamp-to-edge fetches, variance of the interpolated samples, strict `<` selection from `min_sigma2 = 1e2`. Evaluated exactly via separable corner-pair products (matches a literal shader transcription to 1e-7). The shader writes alpha 1.0; here alpha is preserved. ~3-5 s at 12 MP (radius 9). |
| Sharpen | `filter.sharpen` | GPUImage | faithful | `GPUImageSharpenFilter`: `c*(1+4s) - s*(L+R+T+B)`, clamp-to-edge. |
| Unsharp | `filter.unsharp` | aire | faithful | aire's `unsharp` actually blends towards a blur: `c - (c - gauss)*intensity` with a 5-tap, sigma-4 kernel (`compute1DGaussianKernel`), replicate edges. Upstream also blurs alpha (premultiplied RGBA); here alpha is preserved. Kernel scaled for previews only. Float, no 8-bit rounding. |

## Notes for the lead

* **New data file**: `src/lazyroom/data/models/face_detection_yunet_2026may.onnx` (230 KB, copied from
  ImageToolbox `lib/opencv-tools/src/main/assets`; YuNet / OpenCV Zoo, MIT). NOTICE / `docs/THIRD_PARTY.md`
  need an entry (shared files, not edited). Loading it with OpenCV 5.0 prints one harmless
  `setPreferableTarget ... new graph engine` warning per process.
* **ImageToolbox disables G'MIC's built-in `smooth`** (`GmicExecutionOptions.disableSmooth = true` by default,
  `gmic.cpp` patched to skip the command, including `+smooth`). `run_gmic` does not emulate this, so any G'MIC
  filter in any group that calls `smooth` (e.g. anisotropic smoothing filters) differs from ImageToolbox.
  A custom `smooth` definition cannot override the built-in in the CLI (tested).
* Native ports operate on straight (unpremultiplied) RGB with alpha re-attached, whereas aire works on
  Android's premultiplied RGBA; results are identical for opaque images.
