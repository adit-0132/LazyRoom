# Port report: LUT (19 filters)

Module: `src/lazyroom/ops/itb/lut.py`, tests `tests/itb/test_lut.py`.

Sources read: ImageToolbox `LUT512x512Filter.kt`, `CubeLutFilter.kt`, the 17 bundled-look filters
(`<Name>Filter.kt`, each `ChainTransformation(LUT512x512Filter(value to ImageModel(R.drawable.lookup_*)))`),
the Ui classes, Trickle `EffectsPipelineImpl.kt` and `cpp/ProcessingFilters.cpp`
(`applyLutImpl`, `parseCubeFile`, `getColor`, `applyCubeLutImpl`).

## Key finding: ImageToolbox does not use the GPUImage lookup shader

All bundled looks and "512x512 2D LUT" run Trickle `applyLut` on the CPU:

```
q = byte / 4                                   # every channel reduced to 6 bits (0..63)
lut pixel at x = (qb % 8) * 64 + qr, y = (qb / 8) * 64 + qg   # nearest cell, no interpolation
out = round(q*4 + strength * (lut - q*4))      # float32, std::round (half away from zero)
```

So the result has 64 levels per channel at strength 1, and **strength 0 is not an identity**: it
returns the input with its two low bits cleared (`(v // 4) * 4`).  The port reproduces this exactly
(numba kernel, float32 maths).  The channel order was checked against the C++ (uint32 little-endian
read of RGBA bytes: `pixel & 0xff` is red), and the 512x512 layout was verified to be the same one
`_util.lut_from_hald_or_square` implements for the GPUImage shader (blue = 8x8 tile index, red = x,
green = y inside the tile): `test_layout_matches_gpuimage_lookup_interpretation` checks that the
Trickle lookup of every 6-bit cell equals `lut_from_hald_or_square(img)[r, g, b]` for a bundled LUT,
and `test_identity_lut_image` checks the identity lookup round-trip.

`Amatorka` was switched from the trilinear `_common.apply_lut512` path to this faithful Trickle
path for consistency with the other 16 looks (`_common.apply_lut512`/`bundled_lut` are no longer
used by this module).  Mean difference between the two on a test photo is ~0.003 (< 1/255), max ~0.09.

The default LUT of "512x512 2D LUT" is ImageToolbox's `R.drawable.lookup`; it is generated
procedurally (`round(i * 255 / 63)`, verified pixel-exact against the upstream `lookup.webp`), so no
extra data file is needed.  Bundled `lookup_*.webp` files in `src/lazyroom/data/luts/` are
byte-identical to the upstream drawables.

| Filter | op id | upstream | status | notes |
|---|---|---|---|---|
| Amatorka | `filter.amatorka` | Trickle | faithful | `lookup_amatorka`; implementation changed from trilinear to Trickle nearest-cell (see above) |
| BleachBypass | `filter.bleach_bypass` | Trickle | faithful | `lookup_bleach_bypass` |
| Candlelight | `filter.candlelight` | Trickle | faithful | `lookup_candlelight` |
| Celluloid | `filter.celluloid` | Trickle | faithful | `lookup_celluloid` |
| Coffee | `filter.coffee` | Trickle | faithful | `lookup_coffee` |
| DropBlues | `filter.drop_blues` | Trickle | faithful | `lookup_drop_blues` |
| EdgyAmber | `filter.edgy_amber` | Trickle | faithful | `lookup_edgy_amber` |
| FallColors | `filter.fall_colors` | Trickle | faithful | `lookup_fall_colors` |
| FilmStock50 | `filter.film_stock50` | Trickle | faithful | `lookup_filmstock_50` |
| FoggyNight | `filter.foggy_night` | Trickle | faithful | `lookup_foggy_night` |
| GoldenForest | `filter.golden_forest` | Trickle | faithful | `lookup_golden_forest` |
| Greenish | `filter.greenish` | Trickle | faithful | `lookup_greenish` |
| Kodak | `filter.kodak` | Trickle | faithful | `lookup_kodak` |
| MissEtikate | `filter.miss_etikate` | Trickle | faithful | `lookup_miss_etikate` |
| RetroYellow | `filter.retro_yellow` | Trickle | faithful | `lookup_retro_yellow` |
| SoftElegance | `filter.soft_elegance` | Trickle | faithful | single LUT `lookup_soft_elegance_1` (no chaining upstream) |
| SoftEleganceVariant | `filter.soft_elegance_variant` | Trickle | faithful | single LUT `lookup_soft_elegance_2` |
| LUT512x512 | `filter.lut_512x512` | Trickle | faithful | params `strength` (0..1, default 1) + `target_lut_image` (File, image; empty = identity lookup). Same lookup as above. Differences in edge handling only: a non-square LUT image raises `INVALID_PARAMS` (upstream silently returns the input); a square LUT that is not 512x512 is resized with OpenCV (area/bilinear) instead of Coil `size(512)` + `Bitmap.scale` |
| CubeLut | `filter.cube_lut` | Trickle | faithful | params `strength` + `target_cube_lut_file` (File, lut; empty = input unchanged, like upstream). Trickle `getColor` trilinear in float32 with the same axis order (red, then green, then blue), LUT value truncated with `int(c * 255)`, mixed as `round(in*(1-s) + lut*s)`; `DOMAIN_MIN/MAX` ignored exactly like Trickle. Parsing uses `_util.load_cube` (more tolerant than Trickle's parser, which would mis-read `LUT_3D_INPUT_RANGE`/indented comment lines as data); a 1D-only .cube raises `UNSUPPORTED_FORMAT` (upstream returns the input unchanged) |

Performance (12 MP, machine under heavy parallel load): 0.6-1.2 s per look, cube LUT ~0.3 s.

Counts: **19 faithful, 0 approximate, 0 not ported.**
