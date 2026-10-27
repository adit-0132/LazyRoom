# Port report: ImageToolbox "Frames" group (12 filters)

Module: `src/lazyroom/ops/itb/frames.py`, helpers: `src/lazyroom/ops/itb/_frames_impl.py`
(G'MIC bridge shared from `_geometry_impl.gmic_apply`), tests: `tests/itb/test_frames.py`.

Summary: **11 faithful, 1 approximate, 0 not ported.**

| Filter | op id | upstream | status | notes |
|---|---|---|---|---|
| BlurredFrame | `filter.blurred_frame` | G'MIC | faithful | `fx_frame_blur`, all 17 args in builder order (colours as 0..255 ints); Process alpha. Canvas grows. |
| BorderFrame | `filter.border_frame` | ImageToolbox | faithful | Canvas filled with the colour, image drawn centred (source-over). Thickness in px (scaled). |
| CubeFrame | `filter.cube_frame` | G'MIC | faithful | `fx_frame_cube`; Process alpha. Same size. |
| DropShadow | `filter.drop_shadow` | ImageToolbox | approximate | Same construction as `DropShadowFilter` (padding `ceil(max(r -/+ offset, 0))`, `extractAlpha` + `BlurMaskFilter(NORMAL)` with Skia's `sigma = 0.57735*r + 0.5`, SRC_IN tint, image drawn on top). Deviation: Gaussian blur via OpenCV instead of Skia's box-blur approximation, and the fractional shadow offset is rounded to whole pixels (non-filtered bitmap draw). |
| FuzzyFrame | `filter.fuzzy_frame` | G'MIC | faithful | `fx_frame_fuzzy` with RGBA colour; Process alpha. Upstream is clock-seeded; here seeded via `srand <ctx.rng>`. |
| MirrorFrame | `filter.mirror_frame` | G'MIC | faithful | `fx_frame_mirror`; Process alpha. Canvas grows. |
| PaintingFrame | `filter.painting_frame` | G'MIC | faithful | `fx_frame_painting` (+ builder's trailing `0`); Process alpha; seeded via `srand`. The serial number only draws the binary "barcode". Density 100 makes G'MIC's `remove_pixels` fail (asked to remove 0 pixels); ITB's bundled 4.0.2 stdlib has the same definition, so this errors upstream too (test documents it). |
| PolaroidFrame | `filter.polaroid_frame` | G'MIC | faithful | `fx_polaroid`; GmicAlphaMode.Preserve like the builder, so G'MIC's shadow/rotation alpha is discarded and the corners come out black-opaque exactly as in ImageToolbox. |
| RoundedFrame | `filter.rounded_frame` | G'MIC | faithful | `fx_frame_round` with RGBA colour + antialias flag; Process alpha. |
| SideFade | `filter.side_fade` | ImageToolbox | faithful | `SideFadeParams.Relative` (the UI mode): DST_IN with a transparent->opaque linear gradient of length `round(size*strength)` sampled at pixel centres; length 0 = degenerate gradient = unchanged. UI default side `end`. |
| SmoothFrame | `filter.smooth_frame` | G'MIC | faithful | `fx_frame_smooth`; GmicAlphaMode.Preserve (source alpha resized to the new canvas). |
| TornEdge | `filter.torn_edge` | ImageToolbox | faithful | `createTornPaths` reproduced exactly: `java.util.Random((31*w+h)*31 + TornEdgeParams.hashCode())` (Java LCG + Kotlin data-class hash, verified against a real JVM), same vertex order; paths cleared with anti-aliased coverage (4x4 supersampling instead of Skia's analytic AA). Tooth sizes are px (scaled); the hash uses the unscaled values like ITB. |

## Notes

* All G'MIC frame sizes are relative (% of the image) in the stdlib definitions, so no
  `ctx.px` scaling is needed for them.
* G'MIC definitions of the used commands are identical in the dev CLI (3.7.6) and ITB's bundled
  4.0.2 stdlib (see geometry.md).
* Performance at 12 MP (machine under heavy load): BorderFrame 0.6 s, TornEdge ~1.5 s,
  SideFade ~2 s, DropShadow ~2.5 s; G'MIC frames depend on the CLI.
