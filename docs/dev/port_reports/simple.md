# Port report: "Simple effects" group (`simple`)

Module: `src/lazyroom/ops/itb/simple.py` (registrations) and
`src/lazyroom/ops/itb/_simple_impl.py` (algorithms). Tests: `tests/itb/test_simple.py`.

66 catalogue entries: **66 faithful, 0 approximate, 0 not ported** (see the notes on the
upstream bugs that are reproduced on purpose, and on the `ColorMap` entry).

## Conventions that apply to the whole group

* **Float maths.** Upstream works on 8-bit pixels and truncates between steps. The port keeps
  float precision. It quantises to 8 bits only where the algorithm is defined on integers:
  histograms, integer thresholds, ±1-level steps, integer division and 8-bit luma. Those
  filters are Equalize, EqualizeHistogram, Despeckle (steps in 1/255 units), SimpleOldTv,
  SimpleSketch, LaplacianSimple/SobelSimple luma, the ColorMap and LuminanceGradient grey,
  and Moire peak detection. The tests check Gotham and HDR against literal integer
  transliterations of the Trickle C++: 99% of values match within 3 levels; HDR has a
  handful of edge pixels up to 6 levels off, where the Laplacian amplifies truncation.
* **Alpha.** Android bitmaps are premultiplied.
  * Filters that write alpha = 255 upstream are registered with `alpha="full"`. They process
    the premultiplied colour (helper `bitmap_rgb`) and return an opaque result, like
    ImageToolbox. These are: CGAColorSpace, NonMaximumSuppression, WeakPixel,
    LaplacianSimple, SobelSimple, Gotham, HDR, SimpleSketch, ColorMap, AverageDistance and
    LuminanceGradient.
  * SimpleOldTv makes rows 0–2 of each 4-row group opaque. OldTv's haze also transforms
    alpha. JavaLookAndFeel mixes in premultiplied space. AutoPerspective warps premultiplied
    RGBA.
  * All other filters keep alpha and process straight RGB. Upstream applies them to
    premultiplied RGB, which only differs on semi-transparent pixels for non-linear filters.
* **Trickle byte order.** Trickle's C++ reads the RGBA bytes of an `ARGB_8888` bitmap as
  little-endian ints. Its `Color.R()` therefore returns the *blue* byte and `B()` the red byte.
  This is resolved explicitly for the asymmetric Trickle filters: Gotham, SimpleOldTv and
  SimpleSketch.

## Filters

| Filter | op id | upstream | status | notes |
|---|---|---|---|---|
| Achromatomaly | filter.achromatomaly | aire | faithful | `ColorMatrices.Assistance.ACHROMATOMALY` via `Aire.colorMatrix`: out = clamp(M·rgb), alpha untouched (checked in aire `Adjustments.cpp`) |
| Achromatopsia | filter.achromatopsia | aire | faithful | colour matrix (Assistance.ACHROMATOPSIA) |
| AutoPerspective | filter.auto_perspective | OpenCV | faithful | Full port of opencv-tools `AutoStraighten(Perspective)`. Covers `PerspectiveDetector` (Otsu, Canny and adaptive masks; quad scoring with border, fill and cosine rules) and `DocumentDetector` (median, RGB+Luv channels, threshold/Otsu/Canny cascade). Then `getPerspectiveTransform` + `warpPerspective` (INTER_LINEAR, BORDER_REPLICATE) to the max-edge size. Geometry op. Image unchanged if no quad is found. Like upstream, on images under ~90 px the full-frame contour can qualify (the 1.2% border margin is < 1 px), which crops 1 px. |
| AutumnTones | filter.autumn_tones | aire | faithful | colour matrix (existing reference op, unchanged) |
| AverageDistance | filter.average_distance | OpenCV | faithful | `ImageForensics.averageDistance`: \|x − 4-neighbour mean\| (filter2D, REFLECT_101), global min-max normalise over all channels; opaque |
| BlackAndWhite | filter.black_and_white | GPUImage | faithful | `GPUImageLuminanceFilter`: dot(rgb, 0.2125/0.7154/0.0721) |
| Browni | filter.browni | aire | faithful | colour matrix |
| CGAColorSpace | filter.cga_color_space | GPUImage | faithful | `GPUImageCGAColorspaceFilter`: 200×320 sample grid. The GL_LINEAR sample at the cell origin is emulated (texel = u·W − 0.5, clamp to edge). Nearest of black/white/cyan/magenta with the shader's tie-break order. Opaque. |
| CaramelDarkness | filter.caramel_darkness | aire | faithful | colour matrix |
| CodaChrome | filter.coda_chrome | aire | faithful | colour matrix |
| ColorExplosion | filter.color_explosion | aire | faithful | colour matrix |
| ColorMap | filter.color_map | OpenCV | faithful | opencv-tools `ColorMap.apply`: 8-bit grey → `cv::applyColorMap`; opaque. **Catalogue note:** `ColorMapFilter.kt` defines 22 separate ITB filters (Autumn, Bone, Jet, Winter, Rainbow, Ocean, Summer, Spring, Cool Variant, HSV, Pink, Hot, Parula, Magma, Inferno, Plasma, Viridis, Cividis, Twilight, Twilight Shifted, Turbo, Deep Green). They are exposed as one op with a `color_map` choice (default `autumn`, labels = ITB titles). The title is "Color Map" rather than the catalogue's "Autumn". |
| ColorfulSwirl | filter.colorful_swirl | aire | faithful | colour matrix |
| Cool | filter.cool | aire | faithful | colour matrix |
| Cyberpunk | filter.cyberpunk | aire | faithful | colour matrix |
| DeepPurple | filter.deep_purple | aire | faithful | colour matrix |
| Despeckle | filter.despeckle | jhlabs | faithful | `despeckle_exact` (Rust): sequential ±1-level pepper-and-salt nudges against the vertical, horizontal and both diagonal pairs, interior rules as upstream. numba kernel; matches a literal port exactly in tests. |
| Deutaromaly | filter.deutaromaly | aire | faithful | colour matrix (Assistance.DEUTAROMALY) |
| Deuteranopia | filter.deuteranopia | aire | faithful | colour matrix (Assistance.DEUTARONOPIA) |
| DigitalCode | filter.digital_code | aire | faithful | colour matrix |
| ElectricGradient | filter.electric_gradient | aire | faithful | colour matrix |
| Equalize | filter.equalize | jhlabs | faithful | Rust `equalize`: independent per-channel 8-bit histogram, map = round(cdf·255/N) |
| EqualizeHistogram | filter.equalize_histogram | aire | faithful | `histogram_equalization::hist_equal_yuv_rgba` (0.2.4) with full-range YCgCo from the `yuv` crate. 256-bin Y histogram, map = round(255·(cdf − cdf_min)/(N − cdf_min)), alpha kept. Chroma is kept in float (upstream: 8-bit planes, ≤ 0.5 level difference). |
| FantasyLandscape | filter.fantasy_landscape | aire | faithful | colour matrix |
| FuturisticGradient | filter.futuristic_gradient | aire | faithful | colour matrix |
| GoldenHour | filter.golden_hour | aire | faithful | colour matrix |
| Gotham | filter.gotham | Trickle | faithful | `GothamFilter`: brightness −50 → gamma 0.7 → `setHueSaturationIntesity(233.69, 0.2, 0.3)` → contrast 1.2; opaque. Upstream quirk reproduced: hue and saturation are fixed, the intensity argument is ignored, and pixels whose blue would exceed 255 fail `checkRGB` and keep their colour. Byte swap resolved (result is bluish). Float maths. |
| GreenSun | filter.green_sun | aire | faithful | colour matrix |
| HDR | filter.hdr | Trickle | faithful | `HDRFilter`: 3×3 gaussian σ 0.6 on interior pixels only → overlay(original, blur) → `SharpenFilter` (orig + clamp(4-neighbour Laplacian, 0..255), interior only); opaque. Float maths. |
| HotSummer | filter.hot_summer | aire | faithful | colour matrix |
| JavaLookAndFeel | filter.java_look_and_feel | jhlabs | faithful | Rust `JavaLnFFilter`: pixels with (x&1) ≠ (y&1) mixed 75% with opaque #999999. The mix is done on premultiplied ARGB, so alpha → 0.75 + 0.25a. |
| Laplacian | filter.laplacian | GPUImage | faithful | `GPUImageLaplacianFilter` kernel (0.5 1 0.5 / 1 −6 1 / 0.5 1 0.5) + 0.5, clamp-to-edge sampling, alpha kept |
| LaplacianSimple | filter.laplacian_simple | aire | faithful | `Aire.laplacian` → libblur `laplacian`: 3×3 (−1…8…−1) on the fixed-point luma (218R + 733G + 74B) >> 10, REFLECT_101, saturated to u8; grey, opaque |
| LavenderDream | filter.lavender_dream | aire | faithful | colour matrix |
| LemonadeLight | filter.lemonade_light | aire | faithful | colour matrix |
| LuminanceGradient | filter.luminance_gradient | OpenCV | faithful | `ImageForensics.luminanceGradient`: 8-bit grey, Sobel 3×3. R = −cos(atan2(sx, sy))/2 + .5, G = −sin(…)/2 + .5, B = min-max normalised magnitude. Opaque. |
| Moire | filter.moire | OpenCV | approximate (bug fixed) | opencv-tools `Moire.remove(AUTO)`, per channel: DFT, quadrant shift (odd row/column untouched), 8-bit log-magnitude peaks > 180 outside ±10 bins. PCA direction → type (default VERTICAL), radius-8 notch on the matching peaks, Butterworth LP (d0 = 80 bins, n = 10), inverse with `DFT_SCALE \| DFT_REAL_OUTPUT`. **Upstream bug reproduced:** the Butterworth is merged into a 2-channel (B, B) matrix, so `mulSpectrums` multiplies the spectrum by B·(1+i). OpenCV's real-output inverse then does not give back the filtered image; a plain photo comes out strongly distorted (mean error ~60/255 in a test, clipped). Periodic stripes are still removed. **Lead decision:** fixed — LazyRoom uses the intended real filter `cv2.merge([bw, zeros])`, so photos are preserved while moiré peaks are removed. Forward DFT uses real input + `DFT_COMPLEX_OUTPUT` (same spectrum, half the cost). |
| Negative | filter.negative | GPUImage | faithful | `GPUImageColorInvertFilter`: 1 − rgb, alpha kept |
| NightMagic | filter.night_magic | aire | faithful | colour matrix |
| NightVision | filter.night_vision | aire | faithful | colour matrix |
| NonMaximumSuppression | filter.non_maximum_suppression | GPUImage | faithful | `GPUImageNonMaximumSuppressionFilter`. Reproduces the shader's operator-precedence quirk (`multiplier * 1.0 - step(...)` is a subtraction), so a pixel survives only if its red value is > top, top-left, left and bottom-left, and ≥ the other four neighbours. Colour kept, opaque. Matches a literal GLSL transliteration in tests. |
| OldTv | filter.old_tv | ImageToolbox | faithful | ITB chain `GrainFilter(0.36)` → `HazeFilter(0 to 0.3)` → `MonochromeFilter(1, #1C3A00)`. Grain: aire gaussian N(0, 127·0.36) truncated to int, same value on RGB; aire seeds it from the clock, here it uses `ctx.rng`. Haze: GPUImage on all four channels, d = y·0.3. Monochrome: aire `MonochromeToneMapper` in linear light with aire's sRGB curve (0.045 knee). Fused numba kernel. |
| OrangeHaze | filter.orange_haze | aire | faithful | colour matrix |
| Pastel | filter.pastel | aire | faithful | colour matrix |
| PinkDream | filter.pink_dream | aire | faithful | colour matrix |
| Polaroid | filter.polaroid | aire | faithful | colour matrix |
| Protanopia | filter.protanopia | aire | faithful | colour matrix (Assistance.PROTANOPIA) |
| Protonomaly | filter.protonomaly | aire | faithful | colour matrix (Assistance.PROTANOMALY) |
| PurpleMist | filter.purple_mist | aire | faithful | colour matrix |
| RainbowWorld | filter.rainbow_world | aire | faithful | colour matrix |
| RedSwirl | filter.red_swirl | aire | faithful | colour matrix |
| ReduceNoise | filter.reduce_noise | jhlabs | faithful | `reduce_noise_exact`: clamp each channel to [min, max] of its 8 neighbours. Out-of-image neighbours count as the centre, so border pixels are unchanged. Matches a literal port exactly in tests. |
| Sepia | filter.sepia | aire | faithful | colour matrix |
| SimpleOldTv | filter.simple_old_tv | Trickle | faithful | `TvFilter` (gap 4), per column and 4-row group: row 0 = (0, 0, Σ blue/4) (the code's "R" is the blue byte), row 1 = (0, Σ green/4, 0), row 2 = black (upstream `min(0, b)` bug), row 3 untouched. Integer division per pixel; rows 0–2 opaque. Matches a literal port exactly in tests. |
| SimpleSketch | filter.simple_sketch | Trickle | faithful | `SketchFilter`: grey = trunc(0.3·B + 0.59·G + 0.11·R) (byte-swapped weights). Interior pixels become black if \|grey − bottom-right grey\| ≥ 7, else white; border pixels keep the grey. Opaque. |
| SimpleSolarize | filter.simple_solarize | jhlabs | faithful | Rust `SolarizeFilter` transfer: v > .5 ? 2(v − .5) : 2(.5 − v) |
| SobelSimple | filter.sobel_simple | aire | faithful | `Aire.sobel` → libblur `sobel` (crate 0.19.3 in `/tmp/ref/crates`): separable [−1 0 1] row pass, then [1 2 1] column pass on the fixed-point luma. Each pass is saturated to u8, so only left-dark → right-bright edges respond. Grey, opaque. Caveats: aire builds against a local libblur fork (path dependency) that could not be checked; libblur's top-edge row preload (replicate) is approximated by REFLECT_101 (1-px row). |
| SoftSpringLight | filter.soft_spring_light | aire | faithful | colour matrix |
| SpacePortal | filter.space_portal | aire | faithful | colour matrix |
| SpectralFire | filter.spectral_fire | aire | faithful | colour matrix |
| Sunrise | filter.sunrise | aire | faithful | colour matrix |
| Tritanopia | filter.tritanopia | aire | faithful | colour matrix (Assistance.TRITANOPIA) |
| Tritonomaly | filter.tritonomaly | aire | faithful | colour matrix (Assistance.TRITONOMALY) |
| Vintage | filter.vintage | aire | faithful | colour matrix |
| Warm | filter.warm | aire | faithful | colour matrix |
| WeakPixel | filter.weak_pixel | GPUImage | faithful | `GPUImageWeakPixelInclusionFilter`: white where the 3×3 red sum ≥ 1.5 and the centre ≥ 0.01, else black; opaque |

## Performance

The development machine was heavily loaded during the port (load average 10–18 on 12 cores,
with swapping), so wall-clock numbers are pessimistic. Despeckle, OldTv and Gotham are numba
kernels; the rest is numpy/OpenCV.

Measured on a 12 MP (4000×3000) image under that load:

* Moire: ~4.7 s. Three full-size complex DFT/IDFT pairs; the channels run in 3 threads.
* EqualizeHistogram, SimpleSketch, LuminanceGradient, Sobel/LaplacianSimple: ~1.6–2.1 s.
* Everything else: < 1 s. The 42 colour matrices take ~0.4 s each, in the shared
  `color_matrix_3x3`.
