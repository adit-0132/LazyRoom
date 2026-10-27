# Port report: colour and analysis utilities (compare, histograms and scopes, colour tools, Material You, curves, compression lab, palette PDF, multi-frame fusion)

Source: T8RIN/ImageToolbox `feature/compare`, `feature/color-tools`, `feature/pick-color`,
`feature/palette-tools`, `feature/curves`, `feature/compression-lab`, `feature/palette-pdf`,
`feature/multi-frame-fusion`. Libraries: `lib/colors`, `lib/dynamic-theme`, `lib/curves`,
`lib/opencv-tools` (`ImageDiffTool`), and ImageToolboxLibs `libs/histogram` (Kotlin plus native C++).
All of these are Apache-2.0.

**Summary.** 15 new tools and 1 new operation. The `compare`, `histogram` and `color_info` tools were
extended; every change is additive and keeps the old keys. 122 tests pass. No new dependencies.

## Files

| File | Contents |
|---|---|
| `src/lazyroom/tools/color_tools.py` | Tools `color_convert`, `color_mix`, `color_harmonies`, `color_shading`, `color_pick` and `palette_pdf`. Importing it also registers the tools in `_palette_tools` and `_curves_presets`. |
| `src/lazyroom/tools/lab_tools.py` | Tools `compare_visual` and `compare_animation`. Importing it also registers the tools in `_scopes`, `_compression_lab` and `_fusion`. |
| `src/lazyroom/tools/_color_models.py` | Colour models (Skia HSV, androidx ColorUtils HSL/XYZ/LAB/blendARGB, CMYK), harmonies, mixing, shading, colour names and copy formats. |
| `src/lazyroom/tools/_image_diff.py` | `ImageDiffTool`: SSIM, AE, MAE, NCC, PSNR and RMSE, plus the highlight image. Uses fused numba kernels. Also used by the compression lab. |
| `src/lazyroom/tools/_compare_render.py` | `CompareAnimationParams`, `CompareAnimationRenderer` and a GIF89a assembler that mirrors `writeGif`. |
| `src/lazyroom/tools/_histogram.py` | `HistogramGenerator.generate` (RGB plus L* brightness) and the `ImageHistogram`/`LineChart` render. |
| `src/lazyroom/tools/_palette_pdf.py` | `PalettePdfRenderer`, written to PDF by a small vector PDF writer of our own. |
| `src/lazyroom/tools/_material.py` | material-color-utilities port and ImageToolbox `DynamicTheme.getColorScheme` (with `ColorBlindUtils`). |
| `src/lazyroom/tools/_androidx_palette.py` | androidx.palette port, plus `extractPrimaryColor` and `extractImageColorPalette`. |
| `src/lazyroom/tools/_palette_tools.py` | Tools `material_you_scheme` and `palette_simple`. |
| `src/lazyroom/ops/curves_ext.py`, `src/lazyroom/ops/_curves_kernel.py` | Operation `adjust.color_curves` and its fused numba kernel. The kernel is module-level so numba's disk cache works across processes. |
| `src/lazyroom/tools/_curves_presets.py` | `CurvesPresetCodec` and tools `curves_preset_export` and `curves_preset_import`. |
| `src/lazyroom/tools/_scopes.py` | Waveform, vector scope and histogram preview (native `generateScope` plus `ImageScopeOverlay` drawing), and tool `image_scopes`. |
| `src/lazyroom/tools/_compression_lab.py` | Tool `compression_lab`. |
| `src/lazyroom/tools/_fusion.py` | Tool `multi_frame_fusion` (`MultiFrameFusionEngine`). |
| `src/lazyroom/tools/analysis.py` | Additive changes only:<ul><li>`compare`: `pixel_metric`, `threshold`, `highlight_color`, `highlight_output`</li><li>`histogram`: `itb_histogram`, `render_*`, `harmonization_color`, `background`, `overwrite`</li><li>`color_info`: new keys `name`, `xyz`, `itb`</li></ul> |
| `tests/test_color_tools.py` (88 tests), `tests/test_lab_tools.py` (34 tests) | Tests. |

## Feature table

Status: **faithful** means the same algorithm, constants and parameters as the source.
**Approximate** means the behaviour is reproduced but the pixels differ, and the reason is given.
**Not implemented** means the feature was not ported, and the reason is given.

### Compare (`feature/compare`, `lib/opencv-tools`)

| ImageToolbox feature | LazyRoom | Status |
|---|---|---|
| Pixel by pixel. The README says "7 methods", but `ComparisonType` has 6: SSIM, AE, MAE, NCC, PSNR, RMSE | `compare` (`pixel_metric=all\|ssim\|…`, `threshold` default 4, `highlight_color` default red, `highlight_output`); `compare_visual mode=pixel_by_pixel` | faithful. 8-bit premultiplied RGBA with alpha included, `resizeAndPad` of the second image, 11×11 Gaussian window with σ 1.5, C1/C2 for 0..255 values, tolerance mapping 0..100, per-channel mean SSIM and minimum-channel mask, NCC flat-window rule. Inputs are capped at 4096 px like `getBitmapByUri`. |
| Slide (export = `getOverlappedImage`) | `compare_visual mode=slide`, `progress` default 50 | faithful. The canvas is the "after" image. The left `progress` % is replaced by "before" scaled to the same size. |
| Side by side | `compare_visual mode=side_by_side`, `orientation` | approximate. The app only shows this on screen; we render its Row/Column layout with Fit cells and a divider. |
| Toggle tap | `compare_visual mode=toggle` (2-frame GIF, `toggle_ms`) | approximate. This is an interactive view in the app; we render it as a looping GIF. |
| Transparency | `compare_visual mode=transparency` (after over before at `progress` % opacity) | approximate. On-screen only in the app; we use the same alpha rule. |
| Filename labels (`CompareLabel`) | `compare_visual labels=true` | approximate. Same corners, but a fixed dark label style instead of theme colours. |
| Animated before/after GIF (`CompareAnimationParams`, `CompareAnimationRenderer`, `writeGif`) | `compare_animation`: `duration_seconds` 2..8 (default 4), `direction`, `show_labels`, `max_size` 720..4096 (default 2048), label texts | faithful timeline and geometry:<ul><li>500 ms holds and 50 ms smoothstep travel frames, so 62 frames for 4 s</li><li>`outputSize` never upscales</li><li>white canvas, before fitted, after clipped to `[0, edge]`</li><li>divider `max(1, 0.004·min)`</li><li>labels at `0.045·min` in an `argb(170,0,0,0)` rounded box</li><li>loop forever, dispose 1</li><li>first frame full size, later frames limited to `frameBounds`</li></ul>Pixel quantisation differs (see Deviations). |
| Rotate / swap buttons | none | not implemented. Pure UI: swap the two inputs, or use `geometry.rotate` first. |

### Histograms and scopes (`libs/histogram`, `lib/curves`)

| ImageToolbox feature | LazyRoom | Status |
|---|---|---|
| RGB, Brightness and Camera-like RGB histogram data (`HistogramGenerator.generate`) | `histogram itb_histogram=true` returns `itb.red/green/blue` (256 bins) and `itb.brightness` (101 bins of rounded CIE L*) | faithful. Native maths, including the library's 0.045 sRGB threshold. The image is fitted to 1024 px like Coil's `.size(1024)`. |
| Histogram chart (`ImageHistogram` / `LineChart`) | `histogram render_output`, `render_type=rgb\|brightness\|camera` | approximate. Same curved path, gradient alphas (0.8 to 0.4, Camera opaque with `BlendMode.Lighten`), base colours, 25 %/10 % harmonisation and 8×4 dashed grid. Rasterised by us, and the theme background is a parameter. |
| Waveform (`generateScope` type 0) | `image_scopes scope=waveform` | faithful data. The render is approximate. |
| Vector scope (BT.709 Cb/Cr, mean colour per bin, log1p intensity) | `image_scopes scope=vectorscope` | faithful data. The render is approximate. |
| Curves-screen histogram preview (`CurveChannelHistogram`) | `image_scopes scope=histogram`, `histogram_channel` | faithful data. The render is approximate. |

### Colour tools (`feature/color-tools`, `feature/pick-color`, `lib/colors`)

| ImageToolbox feature | LazyRoom | Status |
|---|---|---|
| Colour info fields: HEX, RGB, HSV, HSL, CMYK, Name (display and parsing) | `color_convert` (any model to all), `color_info` → `itb` | faithful:<ul><li>Skia `SkRGBToHSV`/`SkHSVToColor` and androidx `RGBToHSL`/`HSLToColor` in float32 with Java rounding</li><li>`toCMYK`/`cmykToColor`</li><li>`ColorNameParser` (exact hex, then nearest) and `parseColorFromNameSingle` (exact, then containment either way, then black)</li></ul> |
| Copy formats: HEX, Name, HEX + Name, RGB, RGBA, HSL, HSV, CMYK | `formats` in `color_convert`, `color_pick` and `color_info.itb` | faithful. Identical strings. |
| LAB and XYZ (the models ImageToolbox uses for palettes, via androidx `ColorUtils`) | `color_convert` model `lab` / `xyz`, output keys `lab` and `xyz` | faithful to androidx `ColorUtils`: D65 white 95.047/100/108.883, 0.4124… matrices, κ 903.3, ε 0.008856. |
| Colour mixing (`mixWith`, variations 2..20, default 3) | `color_mix` | faithful. `blendARGB` in float32 with `(int)` truncation. |
| Harmonies: complementary, analogous, analogous + complementary, triadic, split complementary, tetradic, square | `color_harmonies` (one type or all), `color_info.itb.harmonies` | faithful. HSV hue rotations, alpha kept. ImageToolbox's tetradic is +30/+180/+210, unlike the old `color_info` tetradic. |
| Shading: tints (white, max 80 %), tones (#8E918F, 90 %), shades (black, 90 %), variations 2..20 (default 5) | `color_shading` | faithful. Includes the float32 `blendARGB` quirk: alpha can come out as 254. |
| Pick colour from image (`calculateColorInPixel`) | `color_pick`: `point` normalised or in px | faithful. `lerp(...).toInt()` mapping; alpha is dropped, as in the app (`source_alpha` is reported as well). |
| Area average | `color_pick radius>0`, `average=srgb\|linear` | LazyRoom extension. ImageToolbox only picks single pixels. |
| Magnifier, zoom, pan | none | not implemented. UI only. |

### Palettes and Material You (`feature/palette-tools`, `lib/dynamic-theme`, androidx.palette, material-color-utilities)

| ImageToolbox feature | LazyRoom | Status |
|---|---|---|
| "Simple colors" = Palette Tools Default mode (`extractImageColorPalette`) | `palette_simple` (`count` 1..128, default 32; `include_targets`) | faithful androidx Palette:<ul><li>112² area scale-down</li><li>5-bit histogram</li><li>default filter</li><li>Vbox median cut over a Java PriorityQueue</li><li>targets with weights</li></ul>Results are named, distinct by name and sorted by luminance then hue. This differs from our `palette` tool (OKLab k-means / PIL median cut). |
| Material You key colour (`extractPrimaryColor`: dominant swatch blended 50 % with vibrant) | `material_you_scheme key_color_source=imagetoolbox` | faithful. Nearest-neighbour stands in for unfiltered `createScaledBitmap`. |
| Material You scheme: 9 `PaletteStyle`s, light/dark, contrast −1..1, invert colours | `material_you_scheme` (`style`, `theme`, `contrast`, `invert_colors`) | faithful port of `getColorScheme(dynamicColor=false)`:<ul><li>custom TonalSpot `DynamicScheme` from `ColorTuple`; the other styles use SchemeX</li><li>`toColorScheme`</li><li>`withDynamicErrorColors`</li><li>`withExpressiveOnColors`</li><li>`withContrastingSurfaces`</li><li>`invertColors`</li><li>`outlineVariant` = onSecondaryContainer @ 0.2 over `surfaceColorAtElevation(6.dp)`</li><li>background = surface</li></ul>Output: 48 Compose roles to hex for light and dark, tonal palettes, and `compose_code` (the "copy as Compose code" text). |
| AMOLED and colour-blind themes (`ColorBlindUtils`) | `material_you_scheme amoled`, `color_blind` | faithful. These are app-theme options; the palette screen does not offer them. |
| Seed colours of an image (MCU `QuantizerCelebi` = Wu + WSMeans, then `Score`) | `material_you_scheme` → `seeds`; `key_color_source=mcu_score` | faithful MCU port. Not used by ImageToolbox itself; added as requested. |
| Palette edit, import/export (41 formats) | `palette_import` / `palette_export` (another agent's) | out of scope. Used read-only by `palette_pdf`. |

### Curves (`feature/curves`, `lib/curves`)

| ImageToolbox feature | LazyRoom | Status |
|---|---|---|
| RGB master and R/G/B (GPUImage tone curve: natural cubic spline, additive master) | `adjust.color_curves` params `rgb`, `red`, `green`, `blue` | faithful |
| CMYK curves | `cyan`, `magenta`, `yellow`, `black` | faithful |
| Lab curves | `lab_l`, `lab_a`, `lab_b` | faithful |
| Hue vs Sat, Hue vs Hue, Hue vs Luma, Luma vs Sat, Luma vs Hue, Sat vs Sat | `hue_vs_sat`, `hue_vs_hue`, `hue_vs_luma`, `luma_vs_sat`, `luma_vs_hue`, `sat_vs_sat` | faithful. Catmull-Rom / centred Hermite / periodic Hermite interpolation and 256-entry byte LUTs sampled like `GL_LINEAR`. |
| Colour wheels (OKLab shadows / midtones / highlights, edges) | `wheel_shadows`, `wheel_midtones`, `wheel_highlights`, `wheel_edges` | faithful |
| Curves presets (`CurvesPresetCodec`): `.itcurve` JSON v1, Lightroom XMP / `.lrtemplate`, Photoshop `.acv`, ZIP | `curves_preset_export` (`format=itcurve\|xmp`), `curves_preset_import` (returns params usable directly by the op), op param `preset` / `preset_file` | faithful. `.itcurve` output is byte-identical, including Java `Float.toString`. XMP carries only the RGB curves, and the tool warns when other curves are dropped. |
| Live preview and editor UI | none | not implemented. UI only. |

### Compression lab (`feature/compression-lab`)

| ImageToolbox feature | LazyRoom | Status |
|---|---|---|
| Format comparison: size, bits per pixel, encode time, SSIM, PSNR | `compression_lab` | faithful pipeline:<ul><li>`ImageFormat.entries` order</li><li>metrics on a copy fitted to 1024 px, using our `ImageDiffTool` port</li><li>default selection JPG, WEBP Lossy, AVIF Lossy, JXL Lossy</li></ul>Encoders are LazyRoom's (`encode_image`). |
| Search modes Manual / TargetQuality / TargetSize | `mode`, `quality` (85), `target_quality` (95, SSIM×100), `target_size_kb` (500) | faithful. Same binary searches with memoised encodes (at most 7 per format), per-format quality ranges, fallbacks flagged `target_satisfied=false`, and a single encode for fixed-settings formats. |
| Formats with no encoder here: MozJPEG, Jpegli, PNG Lossy, OxiPNG, ImageQuant, AV2 ×2, VVC ×2, JXL ×2, Jixel ×2 | listed in `failed` with a reason | not implemented. No permissive encoder is installed. 19 formats work. |

### Palette PDF (`feature/palette-pdf`)

| ImageToolbox feature | LazyRoom | Status |
|---|---|---|
| Contact sheet (`PalettePdfRenderer`) with every option: maximum colour count 4..64 (32), columns 1..6 (3), margins 0..72 (24), spacing 0..36 (8), include source image, show names, show HEX, show source filename, show palette name | `palette_pdf` (same names and defaults) | faithful layout:<ul><li>A4 595×842 pt</li><li>header 22/12 pt</li><li>image ≤ 220 pt high in a rounded 8 pt card with an `argb(28,0,0,0)` border</li><li>card height `clamp(w·0.62 + caption, 64, 160)`</li><li>caption 38/26/0</li><li>ellipsizing</li><li>page numbers</li></ul>Fonts differ (see Deviations). |
| Source detection: palette file (`matchingFilename` and decoder probing) or image | `input` (image or any palette file `palette_import` reads), `palette_format` | faithful. File palettes drop alpha-0 colours and are `distinctBy(argb, name)`. Image palettes use `extractImageColorPalette` (androidx port). |
| List of colours as the source | `colors` (`Name=#hex` or `#hex`), `palette_name` | LazyRoom extension. |

### Multi-frame fusion (`feature/multi-frame-fusion`)

| ImageToolbox feature | LazyRoom | Status |
|---|---|---|
| `FusionMode`: Exposure, Focus, Median, Long exposure, Light trails, Motion trails | `multi_frame_fusion mode` | faithful. Same OpenCV calls and constants: Mertens, Laplacian focus with exposure normalisation, median, gamma 2.2 linear-light long exposure, and both trail modes. |
| `FusionParams` with defaults, ranges and `normalized()` (odd focus radius, 2..12 distinct frames) | all params, same names in snake_case | faithful |
| Alignment and crop to overlap | `align_images`, `crop_to_overlap` | faithful. ORB (1800 features), ratio 0.76, ≥ 12 matches, RANSAC 3 px, 0.55..1.8 sanity check, ≥ 45 % coverage; largest-rectangle crop with the 64 px / 25 % rule. |
| 1024 px preview mode | none | not implemented. The tool always runs at full resolution, which is what the app saves. |

## Algorithms (summary)

- **ImageDiffTool**. Per-pixel maps are computed by fused numba kernels that follow the Kotlin `Core.*` chains:
  - AE: maximum channel `absdiff`; the score is the number of pixels above `t·2.55`.
  - MAE: channel mean of `absdiff`.
  - RMSE: square root of the channel-mean squared error.
  - PSNR: `10·log10(255²/MSE)`, highlighted through a Gaussian-blurred local MSE.
  - SSIM: Gaussian moments, clamped to 0..1.
  - NCC: global centred correlation, and a local map that falls back to `1 − |μa−μb|/255` in flat windows.
- **GIF**. Before/after layers are rendered once and quantised once to a shared 256-colour palette. Each
  frame is a palette-index composite at the divider position, cropped to `frameBounds`. Each sub-frame
  is LZW-encoded with Pillow and assembled into a GIF89a with a NETSCAPE loop block and per-frame graphic
  control extensions (delay, disposal 1).
- **Histogram**. `round(L*)` uses the native code's float32 sRGB→XYZ→Lab chain. The chart is
  supersampled 2×; the cubic segments (control points at the x mid-point) are inverted numerically.
- **Colour models**. Skia HSV, androidx HSL/XYZ/LAB, and the CMYK formulas from the Kotlin
  sources, with float32 intermediates and Java `Math.round` semantics where Android uses them.
- **Material You**:
  - CAM16 / HCT: default viewing conditions, `HctSolver` critical-plane bisection.
  - Tonal palettes: `KeyColor` search, tone 99 for yellow hues.
  - Quantizers: Wu (33³ moments), WSMeans (Lab, `java.util.Random(0x42688)`), Celebi.
  - `Score`: chroma target 48, filters, hue spacing 90 → 15°.
  - `DynamicColor` tone solving (contrast curves, tone-delta pairs), `MaterialDynamicColors` 2021 spec,
    `DislikeAnalyzer`, `TemperatureCache`.
- **androidx Palette**:
  - `ColorCutQuantizer`: 5-bit, Vbox priority queue, float32 averages.
  - Default filter: L ≤ 0.05, L ≥ 0.95, I-line hue 10..37 / S ≤ 0.82.
  - Target scoring: weights 0.24 / 0.52 / 0.24.
- **Curves**:
  - `CurvesValue` interpolation: Catmull-Rom; centred and periodic Hermite for the relation curves.
  - GPUImage tone curve: natural cubic spline on integer 0..255 points, additive master.
  - The rest of `buildCurvesFilter` in order: CMYK shader → Lab shader → HSV relation group → OKLab
    colour wheels.
- **Palette PDF**. Coordinates are the Android canvas mapped to PDF space. Rounded rectangles are cubic
  Bézier paths, with clipping for the swatch and image. Alpha uses ExtGState `ca`/`CA`. Text uses
  standard-14 fonts (Helvetica, Helvetica-Bold, Courier) with WinAnsi encoding, measured with pypdf's
  bundled AFM metrics. The source image is embedded as a JPEG (DCTDecode) at `image_dpi`, default 300.
- **Fusion and compression lab**: see the module docstrings of `_fusion.py` and `_compression_lab.py`.

## Licences of ported code (NOTICE)

| Origin | Licence | Ported into |
|---|---|---|
| ImageToolbox (feature/*, lib/colors, lib/dynamic-theme, lib/curves, lib/opencv-tools) and ImageToolboxLibs `libs/histogram`, T8RIN | Apache-2.0 | everything here. Already covered by the existing NOTICE line. |
| material-color-utilities, Google LLC | Apache-2.0 | `_material.py`. **New NOTICE line needed.** |
| androidx.palette and androidx.core `ColorUtils`, The Android Open Source Project | Apache-2.0 | `_androidx_palette.py`, `_color_models.py`. **New NOTICE line needed.** |
| Skia `SkRGBToHSV` / `SkHSVToColor` numeric behaviour (Google) | BSD-3-Clause | `_color_models.py` (a few lines of arithmetic). **Mention in NOTICE / THIRD_PARTY.** |
| GPUImage for Android tone-curve filter (CyberAgent) | Apache-2.0 | `curves_ext.py`. Already in NOTICE. |
| MaterialKolor (the Kotlin port ImageToolbox links) | MIT | not copied. Its source was not available offline; behaviour follows MCU Java. |

No new dependencies, mandatory or optional. The port uses numpy, OpenCV, Pillow, numba and pypdf, which
are already mandatory; HEIF/AVIF in the compression lab uses pillow-heif when installed.

## Deviations

1. **Material You** (MCU ported without the MaterialKolor source):
   - Follows MCU Java at the default spec (2021) and phone platform.
   - `com.materialkolor.ktx.blend`, used for ImageToolbox's 15 % error-colour blend, is assumed to be `Blend.cam16Ucs`.
   - WSMeans uses `java.util.Random(0x42688)`; MaterialKolor may use Kotlin's RNG. This only affects `seeds`.
   - Wu box sums use exact integers. MCU Java can overflow 32-bit ints on huge boxes.
   - Seeds are computed on a ≤ 112×112 copy and skip non-opaque pixels.
   - When there is no vibrant swatch, the ImageToolbox key colour is blended with transparent black, as in
     the app. The result is reported as `key_color` plus `key_color_alpha`.
2. **androidx Palette**: unfiltered `createScaledBitmap` is approximated by centre-sampled nearest
   neighbour. The `saturate(2f)` variant, used only for the app theme, is a helper and is not exposed.
3. **Compare animation GIF**: the app's GIF encoder quantises every frame on its own. We quantise both
   layers once to a shared 256-colour median-cut palette with Floyd-Steinberg dithering, plus pure white
   for the canvas and divider. Frame count, delays, frame bounds and disposal are the app's. The divider
   is drawn at pixel precision without anti-aliasing. Labels use fontconfig's bold sans font instead of
   Roboto Bold.
4. **Compare renders**: side by side, toggle and transparency have no export in the app; their exported
   layouts are our own. The slide export matches the app and, like the app, draws no labels by default.
5. **Histogram chart and scope renders** use the editor's base colours on a configurable background,
   with no theme harmonisation unless `harmonization_color` is given. Grid dashes and anti-aliasing are
   approximated. All the data is exact.
6. **Colour tools**:
   - `blendARGB` float32 truncation is kept, so tints can come out with alpha 254, exactly like Android.
   - 8-digit hex input defaults to CSS `#RRGGBBAA`; `hex_alpha_order=aarrggbb` gives Android's reading.
   - `hex_itb` outputs use Android's `#AARRGGBB`.
   - Name lookup uses LazyRoom's licence-verified subset of the colour library (31,308 of 33,219
     names), so some names differ from the app.
7. **Curves**:
   - Processing stays in float: no 8-bit rounding of the input or of the intermediate HSV framebuffer.
     The LUTs are 8-bit, as upstream.
   - Point lists without x = 0 / x = 1 endpoints get neutral endpoints added. The app always stores them.
   - All-zero wheels skip the wheel stage.
   - When both a preset and explicit non-default params are given, the params override the preset. The
     app replaces the whole state on import.
8. **Scopes** are computed on the full-resolution image. The app uses its on-screen preview bitmap.
9. **Compression lab** encoders are LazyRoom's:
   - WebP lossless uses method 0.
   - AVIF "lossless" is quality 100 at 4:4:4 (about 49 dB), because Pillow cannot select the identity
     matrix.
   - HEIC lossless uses pillow-heif quality −1.
   - TIFF is JPEG-compressed. This is ImageToolbox's quirk, where the TIFF setting 5 is used as a list
     index.
   - JP2 quality is a target PSNR in dB.
   - ICO is rendered as multi-size icons.
   - GIF is encoded with fixed settings.
   - No "best" result is marked, matching the app.
10. **Palette PDF**:
    - Text uses standard-14 Helvetica/Courier instead of Roboto / Droid Sans Mono, so ellipsizing points
      differ slightly.
    - Characters outside WinAnsi become `?`.
    - The source image is embedded as JPEG at 300 dpi. Android embeds the bitmap itself.
11. **No cancellation**: the tool API has no cancellation hook, so long tools (compression lab, fusion)
    run to the end.

## Performance

Measured on this shared 12-thread machine, warm numba cache. The timings were noisy because other
agents' tests were running at the same time.

| Tool | Input | Time |
|---|---|---|
| `compare pixel_metric=all` | 4000×3000 vs 4000×3000 JPEG | ≈ 13 s. Per metric: AE/MAE/RMSE/PSNR 0.2–0.4 s, SSIM ≈ 4 s, NCC ≈ 4 s (20 Gaussian blurs on 4-channel float32 per metric). The existing perceptual metrics account for the rest. |
| `compare_animation` | 12 MP pair → 2048 px, 62 frames | ≈ 3.4 s (4096 px: ≈ 11 s, 12.8 MB) |
| `histogram` with ITB data and render | 12 MP | ≈ 3 s |
| `image_scopes scope=all` | 12 MP | ≈ 10 s with the three renders |
| `adjust.color_curves`, every stage active | 12 MP | ≈ 1.7 s (fused numba kernel). The first ever call compiles in about 16 s; later processes take ≈ 0.4 s (cached). |
| `material_you_scheme` (image, both themes, seeds) | 12 MP | ≈ 1.3 s |
| `palette_simple` | 12 MP | ≈ 0.7 s |
| `palette_pdf` from an image | 12 MP | ≈ 0.8 s |
| `multi_frame_fusion` | 3 × 6 MP, no alignment | 1.6–2.7 s per mode |
| `compression_lab target_size`, default formats | 12 MP | ≈ 54 s (21 encodes, mostly AVIF and WebP) |

## Tests

`LAZYROOM_ITB_ONLY=none LAZYROOM_HOME=/tmp/pm_color_home .venv/bin/python -m pytest tests/test_color_tools.py tests/test_lab_tools.py -q`
gives **122 passed** (88 + 34). `tests/integration/test_cli.py` and `tests/integration/test_mcp.py`: **5 passed**.

What the tests check:
- **Colour conversions** against textbook values: HSV/HSL/CMYK of primaries and #3366CC; Lab of
  white/red/blue ±0.05–0.1; round trips; copy-format strings; names and the black fallback.
- **Harmonies**: measured hue offsets for all 7 types.
- **Mixing and shading**: exact blendARGB values.
- **Picking**: point mapping in normalised and pixel units, and the area mean.
- **ImageDiffTool**:
  - Identical images give SSIM 1, NCC 1, infinite PSNR (returned as JSON `null` with `"infinite": true`)
    and zero errors.
  - Analytic MAE/RMSE/PSNR values.
  - Threshold mapping and `resizeAndPad`.
- **Compare animation**:
  - Ports ImageToolbox's `CompareAnimationParamsTest`: exact loop, holds, symmetry, `outputSize`, and
    bounded durations.
  - The GIF has 62 frames totalling 4000 ms, loops, and the vertical bounds are correct.
- **Compare renders**: all modes.
- **Histograms**: bin counts and L* brightness; the old tool keys are kept.
- **`color_info`**: backward compatible.
- **Palette PDF**: page count and text (palette name, colour names, HEX) via pypdf, for a colour list and
  a GIMP palette file; column and caption options.
- **Material You**:
  - MCU unit-test values: HCT of red/green/blue/white, `TonalPalette(0xff0000ff)`, Score ordering and
    fallback, Wu and Celebi.
  - `SchemeTonalSpot(0xff0000ff)`: light primary #555992, container #E0E0FF, surface #FBF8FF; dark
    primary #BEC2FF.
  - Monochrome and Fidelity schemes.
  - Contrast ≥ 4.5 for every variant.
  - ImageToolbox's rules: error hues and chroma, expressive on-colours, contrasting surfaces, Compose-code
    format.
- **androidx Palette**: two-colour images, filters, colour count.
- **Curves**:
  - Identity is an exact no-op for every space.
  - Expectations from ImageToolbox's `ColorCurvesFilterInstrumentedTest`.
  - Colour wheels.
  - The numba kernel matches the numpy reference (≤ 1.4e-5).
  - Codec round trips, Java float formatting, Lightroom/ACV/ZIP import and error cases (from
    `CurvesPresetCodecTest`).
- **Scopes**:
  - A grey ramp gives a diagonal waveform.
  - Pure red lands at 103° in the vector scope.
  - Brightness bins, tool outputs and renders.
- **Fusion**:
  - ImageToolbox's engine test cases (median, long exposure 184..188, trails, focus, exposure).
  - Alignment recovers a 9×6 px shift and crops to the overlap.
  - On 7 noisy frames the fused RMSE is below 0.6 × a single frame's.
  - Parameter normalisation and the 12-frame cap.
- **Compression lab**:
  - Lossless PNG/QOI give SSIM 1 and PSNR `null` (infinite).
  - The target-size search returns the highest quality that fits.
  - The target-quality search reaches its SSIM.
  - `output_dir` writes the encoded files.

## Wiring for the lead

- `src/lazyroom/tools/registry.py` `TOOL_MODULES`: add `"color_tools"` and `"lab_tools"`. Importing
  these also registers the private modules `_palette_tools`, `_curves_presets`, `_scopes`,
  `_compression_lab` and `_fusion`.
- `src/lazyroom/ops/__init__.py`: add `from . import curves_ext  # noqa: F401  (ImageToolbox colour curves: adjust.color_curves)`.
- Dependencies: none.
- NOTICE: add "This product includes software ported from material-color-utilities (Google LLC,
  Apache-2.0), androidx.palette and androidx.core ColorUtils (The Android Open Source Project,
  Apache-2.0) and colour-conversion arithmetic from Skia (Google LLC, BSD-3-Clause)."
- Optional follow-up for the owner of `presets_misc.py`: `_image_palette` approximates
  `extractImageColorPalette` with PIL median cut. `_androidx_palette.extract_image_color_palette` is the
  faithful version.
