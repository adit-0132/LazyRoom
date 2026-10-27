# Port report: Dithering (25 filters)

Module: `src/lazyroom/ops/itb/dithering.py`, helpers `src/lazyroom/ops/itb/_dithering_impl.py`,
tests `tests/itb/test_dithering.py`.

Sources read: Trickle `trickle/src/main/cpp/Dithering.cpp` + `pipeline/DitheringPipelineImpl.kt`,
ImageToolboxLibs `jhlabs/src/main/rust/jhlabs/src/lib.rs` (`"ColorHalftoneFilter"`), GPUImage
`GPUImageHalftoneFilter.java` / `GPUImageCrosshatchFilter.java`, G'MIC `CreativeFilters.kt`
(`PosterizedDithering`) + the `fx_pdithered` stdlib definition, aire `jni/Base.cpp` (`paletteImpl`),
`algo/WuQuantizer.cpp`, `base/RemapPalette.cpp`, `algo/NearestColorSearch.hpp`, `algo/CoverTreePoint.h`.

## Shared Trickle behaviour (all 20 Trickle filters)

* 8-bit integer maths exactly as in C++ (`v + v*M/div`, `num*error/div` truncating toward zero,
  `< threshold` / `> threshold` comparisons as written per filter). Output channels are pure 0/255.
* "Gray Scale" = Trickle's `int gray = r`: only the **red** channel is dithered and written to R, G, B
  (no luminance conversion — this is upstream behaviour, kept on purpose).
* Error diffusion: plain raster order (left to right, top to bottom, no serpentine), one independent
  error buffer per channel.  Upstream border bug **not** replicated: Trickle's diffusion loops run
  `y < height - 1|2` and `x` from 1|2 to `width - 1|2` while the row pointer starts at column 0, so
  the last 1-2 rows and the last 2-4 columns of the output are never written (transparent black).  The port dithers the whole frame with the same kernel and drops error that
  leaves the image; interior pixels near the right edge can therefore differ slightly.
* `threshold` 1..255 (default 200) and `gray_scale` (default false) mirror the Ui classes.
* Performance (12 MP, numba `cache=True`, machine under heavy parallel load): error diffusion 0.6-1.4 s,
  ordered/threshold 0.25-0.45 s.

| Filter | op id | upstream | status | notes |
|---|---|---|---|---|
| FloydSteinbergDithering | `filter.floyd_steinberg_dithering` | Trickle | faithful | 7,3,5,1 /16; border note above |
| FalseFloydSteinbergDithering | `filter.false_floyd_steinberg_dithering` | Trickle | faithful | 3,3,2 /8 (right, down, down-right) |
| JarvisJudiceNinkeDithering | `filter.jarvis_judice_ninke_dithering` | Trickle | faithful | 12 taps /48 |
| StuckiDithering | `filter.stucki_dithering` | Trickle | faithful | 12 taps /42 |
| BurkesDithering | `filter.burkes_dithering` | Trickle | faithful | 7 taps /32 |
| SierraDithering | `filter.sierra_dithering` | Trickle | faithful | 10 taps /32 |
| TwoRowSierraDithering | `filter.two_row_sierra_dithering` | Trickle | faithful | 7 taps /16 |
| SierraLiteDithering | `filter.sierra_lite_dithering` | Trickle | faithful | Trickle's `sierraLiteImpl` uses the **Two-Row Sierra** kernel (4,3 / 1,2,3,2,1 /16), not the classic 2/4,1/4,1/4 — ported as written |
| AtkinsonDithering | `filter.atkinson_dithering` | Trickle | faithful | 6 taps of error/8 |
| LeftToRightDithering | `filter.left_to_right_dithering` | Trickle | faithful | per-row running error, `|delta| < 10` ignored, delta from the raw value; full frame like upstream |
| BayerTwoDithering | `filter.bayer_two_dithering` | Trickle | faithful | `matrix[x%2][y%2]`, `v + v*M/5 < t` |
| BayerThreeDithering | `filter.bayer_three_dithering` | Trickle | faithful | `/10` |
| BayerFourDithering | `filter.bayer_four_dithering` | Trickle | faithful | `/17` |
| BayerEightDithering | `filter.bayer_eight_dithering` | Trickle | faithful | `/65` |
| Clustered2x2Dithering | `filter.clustered_2x2_dithering` | Trickle | faithful | `v > M[y%2][x%2]*64` (gray_scale only, no threshold param, like upstream) |
| Clustered4x4Dithering | `filter.clustered_4x4_dithering` | Trickle | faithful | `*16` |
| Clustered8x8Dithering | `filter.clustered_8x8_dithering` | Trickle | faithful | `*4` |
| YililomaDithering | `filter.yililoma_dithering` | Trickle | faithful | 4x4 matrix (1..16) `*16`, `v > t` |
| SimpleThresholdDithering | `filter.simple_threshold_dithering` | Trickle | faithful | `v < threshold -> 0` |
| RandomDithering | `filter.random_dithering` | Trickle | faithful | one `rand() % 256` threshold per pixel shared by R,G,B; RNG is `ctx.rng` (seeded, deterministic) instead of time-seeded `std::rand`. The `threshold` slider exists in ImageToolbox but is ignored upstream — kept for parity, documented as unused |
| ColorHalftone | `filter.color_halftone` | jhlabs | faithful | ImageToolboxLibs' Rust `libjhlabs` implementation (rotated grids, 5-neighbour dots, `dot = (1-v²)·half·1.414`, smoothstep edge, `(255·f) as u32`); f64 instead of f32 arithmetic. `radius` is px (`ctx.px`), angles are integer degrees (roundTo 0) for C/M/Y = R/G/B. Radius 0 -> white, as upstream (NaN grid, `f32::min` ignores NaN). Verified against a literal Python transcription |
| Halftone | `filter.halftone` | GPUImage | faithful | shader ported in float32 at pixel-centre texture coords (row 0 = t 0); cell-centre colour sampled bilinearly with clamp-to-edge (cv2.remap); `size` is a fraction of the width (resolution independent, no px scaling). Luminance dot product rounded once (as a fused GPU dot) |
| Crosshatch | `filter.crosshatch` | GPUImage | faithful | four hatch layers below lum 1.0/0.75/0.5/0.3 with GLSL `mod`; spacing clamp `max(s, 1/2048)` as in `setCrossHatchSpacing` before the output size is known (never active in the 0.001..0.05 range). Luminance rounded once so pure white stays unhatched |
| PosterizedDithering | `filter.posterized_dithering` | G'MIC | faithful | `fx_pdithered gamma,contrast,brightness,smoothness,quantize_colors,smooth_colors,mode,color_intensity` (same order/ranges as the Kotlin builder; mixer enum -> 0..5). `smoothness` and `smooth_colors` are G'MIC blur sigmas in px, scaled with `ctx.px` (identical at export scale 1). Verified with the dev CLI |
| Quantizier | `filter.quantizier` | aire | approximate | Palette: aire's 4-D Xiaolin Wu quantizer ported exactly (33 bins/axis, `alphaThreshold=15, alphaFader=15`, `AdjustMoments` integer-division test, uint32 running sums, float32 moments in the same summation order, and the C++ `sqr(int)` 32-bit overflow / ARM64 saturating float->int conversions in `Maximize`/`CalculateVariance`). Remap: aire's `RemapPalette` JJN "dithering" is a no-op upstream (errors are written into destination pixels that are overwritten later), so plain nearest-colour mapping is applied. **Deviations:** (1) nearest colour is the exact minimum of aire's `CoverTreePoint` metric (2dR²+4dG²+w·dB²+dA², w = 3 if R<128 else 2), whereas aire queries a cover tree built on that non-metric squared distance, which can occasionally return a non-nearest entry; (2) alpha is preserved, not quantized (upstream quantizes RGBA; identical for opaque images) |

Counts: **24 faithful, 1 approximate, 0 not ported.**
