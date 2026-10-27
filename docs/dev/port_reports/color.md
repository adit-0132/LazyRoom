# Port report — group `color` (43 filters)

Module: `src/lazyroom/ops/itb/color.py` (+ helpers in `_color_impl.py`), tests: `tests/itb/test_color.py`.

**Summary:** 42 faithful, 1 approximate, 0 not ported.

## General notes (apply to several filters)

* **Alpha / premultiplication.** GPUImage shaders and Trickle/jhlabs native code run on Android's
  *premultiplied* bitmap memory. The port works on straight alpha. Opaque pixels give the same result;
  semi-transparent pixels can differ slightly. Filters that change alpha on purpose use
  `alpha="full"`: Opacity, Posterize, ColorMatrix4x4, ColorMask, ColorOverlay, ReplaceColor and
  RemoveColor.
* **Shaders that force alpha = 1** (RGB filter, the Sobel/RGB stages of Neon, and TriTone's
  ramp alpha) would flatten transparent images onto black. The port keeps the image alpha instead.
  For TriTone, the ramp colours' own alpha is multiplied in.
* **8-bit quantisation.** Upstream rounds to 8 bits between stages and on output. The port keeps
  float precision, except where the algorithm itself is defined on 8-bit values: ColorMask (bitwise
  AND), Replace/RemoveColor (8-bit distance test) and ColorPoster (palette built with Skia's 8-bit HSV
  conversion).
* **Trickle byte order (re-checked).** Trickle filters built on `ColorGetter.h`'s `Color` class read
  Android ARGB_8888 pixels (R,G,B,A bytes) as little-endian ints, so `R()` and `B()` are swapped. None of
  the five Trickle functions used here go through that class:
  * `colorPosterizeImpl`, `replaceColorImpl`, `transferPaletteImpl` and `tritoneImpl` read and write the
    pixels as `uint8_t` bytes (`[0]`=R … `[3]`=A).
  * Colour params arrive as Compose `Color.toArgb()` ints (0xAARRGGBB) and are decoded with `>>16` = red
    (`ColorToRGB`/`ColorToARGB`/`mixColors`). The `generateShades` palette is also ARGB.
  * `drawColorAboveImpl` deliberately passes (b, g, r) to cairo, whose ARGB32 surface stores bytes
    B,G,R,A on little-endian, so red still lands in Android's R byte.

  So R/B are not swapped in any of these ports. `test_trickle_channel_order` guards this with asymmetric
  colours.
* **G'MIC.** All 15 commands are built exactly like the ImageToolboxLibs Kotlin builders: same
  argument order, enum `.value` ints, booleans as 0/1, colours as 8-bit R,G,B. Each one was run with the
  dev CLI (3.7.6). I decompressed ITB's bundled stdlib (4.0.2, from `gmic_stdlib_community.h`) and the
  3.7.6 stdlib from `libgmic.so`. The definitions of all 15 commands and the sub-commands they use are
  identical. ITB's bridge disables G'MIC's `smooth` command (`disableSmooth = true`), but none of these
  commands reach `smooth`, so that has no effect. The test `test_gmic_command_strings` checks every
  default command string against the Kotlin builder output.

## Filters

| Filter | op id | upstream | status | notes |
|---|---|---|---|---|
| BlackAndWhiteFilm | `filter.black_and_white_film` | G'MIC | faithful | `fx_bwfilmsimulate`. G'MIC's RNG is time-seeded, so the command is prefixed with `srand <seed from ctx.rng>` to make it deterministic (the grain is random upstream anyway). `radius` (local-contrast radius, px) is scaled by the preview scale. |
| BoostChromaticity | `filter.boost_chromaticity` | G'MIC | faithful | `fx_boost_chroma`. |
| ChannelMix | `filter.channel_mix` | jhlabs | faithful | Formula from the ImageToolboxLibs jhlabs Rust port, written as the equivalent 3x3 matrix (it is linear). Upstream integer truncation is not reproduced. |
| ChannelProcessing | `filter.channel_processing` | G'MIC | faithful | `fx_channel_processing`. 47 channel choices, upstream order. |
| ColorBalance | `filter.color_balance` | GPUImage | faithful | Shader ported to a numba kernel; preserveLuminosity = true (GPUImage default). One `values` list of 9 floats, like ITB's `FloatArray`: [highlights rgb, midtones rgb, shadows rgb]. **Upstream quirk kept:** `ColorBalanceFilter.kt` passes `value[6]` (shadows red) as the midtones blue shift. |
| ColorMask | `filter.color_mask` | jhlabs | faithful | `pixel &= mask` on 8-bit ARGB, alpha included. Mask bytes 0xFF/0x00 skip quantisation. |
| ColorMatrix3x3 | `filter.color_matrix_3x3` | aire | faithful | `out = M·rgb`, M row-major (aire `colorMatrix`). |
| ColorMatrix4x4 | `filter.color_matrix_4x4` | GPUImage | faithful | `textureColor * colorMatrix` with the array uploaded column-major, i.e. `out_j = Σ_i arr[4j+i]·v_i` (rows are output channels, applied to r,g,b,a, `alpha="full"`). Uses its own code because the alpha row and column matter here (see below). |
| ColorOverlay | `filter.color_overlay` | Trickle | faithful | Trickle `drawColorAbove` = cairo OVER of the colour over the whole canvas. Alpha is changed too (`a' = ca + a(1-ca)`), as upstream. |
| ColorPoster | `filter.color_poster` | Trickle | faithful | `generateShades` (Android/Skia HSV↔RGB with its rounding, step `round(max(50·strength, 1))`, V = 2..98 %) followed by `colorPosterize` (nearest palette colour, Euclidean RGB, first minimum wins; numba). |
| ColorTemperature | `filter.color_temperature` | G'MIC | faithful | `fx_tk_colortemp`. |
| DesaturateNorm | `filter.desaturate_norm` | G'MIC | faithful | `fx_gcd_norm_eq`. |
| FalseColor | `filter.false_color` | GPUImage | faithful | `mix(first, second, dot(rgb, (0.2125, 0.7154, 0.0721)))`. |
| GradientMap | `filter.gradient_map` | ImageToolbox | faithful | Lookup position is `(77R+150G+29B)/256`; colours interpolate linearly between stops. The default is ITB's `Spectral` palette as 8 evenly spaced `Gradient` stops. A continuous lookup replaces the 256-entry table (the only difference is 8-bit rounding). The gradient's alpha is ignored, as upstream. |
| GraduatedColorAccent | `filter.graduated_color_accent` | G'MIC | faithful | `fx_gca`. |
| Grayscale | `filter.grayscale` | aire | faithful | Weighted sum of sRGB→linear channels (aire's 0.045 threshold). **Upstream quirk kept:** the linear value is written back without sRGB re-encoding, so the output is darker than a perceptual grey. |
| HslAdjustment | `filter.hsl_adjustment` | G'MIC | faithful | `gcd_hsl`. |
| HsvEqualizer | `filter.hsv_equalizer` | G'MIC | faithful | `fx_hsv_equalizer 0,…` (the builder's leading `false`). |
| Hue | `filter.hue` | GPUImage | faithful | Rotates YIQ chroma by `-(hue % 360)°`, folded into one 3x3 matrix (same maths). The ITB slider range 0..255 is kept. |
| HueShiftBands | `filter.hue_shift_bands` | ImageToolbox | faithful | ITB procedural GLSL: `toField` (aspect, offset, scale, rotation), shader `rgbToHsv`/`hsvToRgb`, hue += `p.y·intensity + hue`, saturation × factor. |
| HueShiftFan | `filter.hue_shift_fan` | ImageToolbox | faithful | Same, hue += `atan2(p)/τ·intensity + hue`. |
| HueShiftTunnel | `filter.hue_shift_tunnel` | ImageToolbox | faithful | Same, hue += `‖p‖·intensity + hue`. |
| Lomo | `filter.lomo` | G'MIC | faithful | `fx_lomo`. |
| Monochrome | `filter.monochrome` | aire | faithful | aire `MonochromeToneMapper`: overlay of the colour on linear BT.601 luma, mixed by strength, re-encoded to sRGB (exposure 1). |
| Neon | `filter.neon` | ImageToolbox | faithful | ITB chain: GPUImage Sharpen(strength) → Sobel edge detection (greyscale, 3x3 taps `amount` texels apart, clamp-to-edge) → RGB filter(colour). Sobel is done with OpenCV separable filters. `amount` is scaled by the preview scale. Alpha kept (see general notes). |
| Opacity | `filter.opacity` | GPUImage | faithful | `alpha *= opacity` (`alpha="full"`; RGB input gets an alpha channel). Note: upstream writes this into premultiplied memory without rescaling RGB, which on Android may brighten semi-transparent results. Not reproduced. |
| PaletteTransfer | `filter.palette_transfer` | Trickle | faithful | Reinhard mean/std transfer in Trickle's CIELab (sRGB threshold 0.045, population std). The reference is loaded fit-within 1000 px, like ITB's `loadBitmap(1000)`. **Upstream quirk kept:** a target channel with std = 0 maps to 0. `target_image` is a required File param (ITB's string name; it is the palette source). |
| PaletteTransferVariant | `filter.palette_transfer_variant` | aire | faithful | aire `copyPalette` → palette-transfer 0.2 crate (downloaded from crates.io and read) with colorutils-rs 0.7.5 conversions: OKLab / CIELab / CIELuv / lαβ, colorutils' exact sRGB curve. A channel with an inf/NaN/0 scale is left unchanged, as upstream. |
| Posterize | `filter.posterize` | GPUImage | faithful | `floor(c·L + 0.5)/L` on every channel including alpha (`alpha="full"`), as the shader does. |
| RGB | `filter.rgb` | GPUImage | faithful | Per-channel multiply. Alpha kept (the shader writes 1.0, see general notes). |
| RemoveColor | `filter.remove_color` | Trickle | faithful | ReplaceColor with a transparent target. |
| ReplaceColor | `filter.replace_color` | Trickle | faithful | 8-bit RGB distance / 255 ≤ `cbrt(3)·tolerance` (upstream uses cbrt, kept) → target ARGB (alpha included). |
| SaturationEqualizer | `filter.saturation_equalizer` | G'MIC | faithful | `Saturation_EQ_p`. |
| Saturation | `filter.saturation` | aire | faithful | OKLab a,b × strength → linear RGB → optional ACES-Hill tone map (default on, as ITB) → sRGB (colorutils curve). |
| SelectiveDesaturation | `filter.selective_desaturation` | G'MIC | faithful | `fx_selective_desaturation`. |
| Solarize | `filter.solarize` | GPUImage | faithful | `abs(step(lum, threshold) - rgb)`: pixels with luminance ≤ threshold are inverted. |
| SpecificSaturation | `filter.specific_saturation` | G'MIC | faithful | `gcd_hio_levels`. |
| TargetColorSpot | `filter.target_color_spot` | G'MIC | faithful | `fx_gcd_color_target`. x/y kept as percentages, as upstream. |
| TemperatureBalance | `filter.temperature_balance` | G'MIC | faithful | `gcd_temp_balance`. |
| ToneCurves | `filter.tone_curves` | GPUImage | approximate | Exposes only the RGB part of ITB's curves editor as four `Points` params (rgb/luma composite, red, green, blue). This part is ported exactly: ITB `replacePoints` normalisation (clamp, sort, unique x, ends pinned to 0/1, max 16 points), the GPUImage natural cubic spline (bit-identical to SciPy's natural spline on the int points), the composite + per-channel offset LUT, and the GL_LINEAR 256-texel sampling; all-default curves are an exact identity. **Not exposed:** the CMYK, Lab, hue/luma/sat relation curves and the colour wheels stored in the same `ToneCurvesParams` (identity in the default value). |
| TriTone | `filter.tri_tone` | Trickle | faithful | BT.601 luma: 0..127 → shadow→mid ramp, 128..255 → mid→highlight ramp (continuous version of the 256-entry LUT). Alpha: see general notes. |
| Unpurple | `filter.unpurple` | G'MIC | faithful | `unpurple`. `blur_standard_deviation` (px) is scaled by the preview scale. |
| Vibrance | `filter.vibrance` | aire | faithful | `c += clamp((max - mean)·v)`, the same boost added to all channels, as aire does. Integer truncation not reproduced. |

## Shared-code issue reported during the port (now fixed by the lead)

`_common.color_matrix_4x4` used to compute `rgba @ M`, the transpose of GPUImage's
`textureColor * colorMatrix` (column-major upload, so `rgba @ M.T` is correct). The lead has fixed it.
`filter.color_matrix_4x4` keeps its own implementation for two reasons:
* it also writes the alpha output row (`alpha="full"`);
* it returns RGB when the alpha row is the identity.

Its convention matches the fixed helper; `test_color_matrices_identity_and_convention` checks it with
the GPUImage sepia matrix.

## Performance

Measured on a 12 MP image, but the machine was heavily loaded during the run (load average 14–24 on
12 cores, swap in use; a trivial per-pixel multiply took ~0.8–1.4 s instead of ~0.15 s).

* Per-pixel numpy filters are processed in row bands (`_color_impl.chunked`) to avoid full-size
  temporaries. Under that load they measured 0.7–2 s wall.
* The heavier ones measured 3–8 s wall: the palette transfers (colour-space conversions plus statistics
  passes), Saturation, Monochrome and the HueShift family. Scaled by the ~5x load inflation, all should
  be under 2 s on an idle machine.
* ColorBalance and ColorPoster use parallel numba kernels.
* G'MIC filters take about 0.25–1 s each on small images, dominated by process start-up.
