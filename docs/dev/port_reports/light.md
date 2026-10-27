# Port report: ImageToolbox "Light" group (54 filters)

Modules: `src/lazyroom/ops/itb/light.py` (registrations), `src/lazyroom/ops/itb/_light_impl.py`
(private helpers, numba kernels), tests `tests/itb/test_light.py`.

Summary: **54 faithful, 0 approximate, 0 not ported.** Every filter keeps the ImageToolbox `Ui*Filter`
ranges and defaults (UI defaults win where the implementation class differs: Gamma 1.0, Rays opacity 0.75).

General notes

* Sources actually ported: aire C++ (`color/ConvolveToneMapper.cpp`, `color/tone/*.hpp`, `eotf-inl.h`,
  `Adjustments.cpp`, `jni/Base.cpp`); for CLAHE/equalisation the Rust crate **histogram_equalization 0.2.4**
  that aire-filters links (+ **colorutils-rs 0.7.5** colour conversions, **yuv 0.8.6** YCgCo), versions taken
  from `aire-filters/Cargo.lock` and downloaded to `/tmp/ref/crates/` (all `BSD-3-Clause OR Apache-2.0` —
  the lead may want to add them to NOTICE/THIRD_PARTY); Trickle `ProcessingFilters.cpp::bloomImpl`;
  ImageToolboxLibs' **native Rust jhlabs** (`libs/jhlabs/src/main/rust/jhlabs/src/lib.rs` — this is what
  `JhFilter.filter()` runs, not the Java jhlabs code); GPUImage GLSL; ImageToolbox's procedural GLSL
  (`ProceduralShaders.kt`); G'MIC builders in `ImageToolboxLibs/libs/gmic`.
* Float pipeline: the 8-bit intermediate truncations of the native code are not replicated (outputs are
  clamped 0..1 by `itb_op`). NaNs produced by upstream maths are mapped to 0 (what the native float→u8
  conversion yields), inf to the clamp limit.
* Performance (12 MP, shared/loaded machine, incl. ~0.4 s framework overhead): all non-G'MIC filters
  0.3–2.5 s. Per-pixel upstream loops are numba kernels (`cache=True, parallel=True`); the per-channel aire
  tone curves (and Gamma / Bloom's gamma) are evaluated once on a 65 536-entry table (upstream input is
  8-bit; max deviation vs. direct evaluation 2.4e-5). First use JIT-compiles the kernels (cached on disk).
* Validation: vectorised (C)LAHE is checked against a literal transliteration of the crate's loops
  (`test_ahe_map_matches_scalar_transliteration`, exact equality); numba kernels were cross-checked
  against independent numpy implementations (max diff ≤ 2.5e-3 at a few rounding boundaries);
  G'MIC command strings are asserted equal to the Kotlin builders' output for the defaults.
* G'MIC: the dev CLI is 3.7.6, ImageToolbox bundles 4.0.2. I extracted the 4.0.2 stdlib from
  `gmic_stdlib_community.h` (`/tmp/ref/gmic_stdlib402/stdlib.gmic`) and ran every light-group command with
  both: outputs are bit-identical except LightPatch/ShadowPatch, which are random (see below).

| Filter | op id | upstream | status | notes |
|---|---|---|---|---|
| AcesFilmicToneMapping | filter.aces_filmic_tone_mapping | aire | faithful | aire sRGB decode (threshold 0.045) → `x*e` → Narkowicz fit clamped → sRGB encode. |
| AcesHillToneMapping | filter.aces_hill_tone_mapping | aire | faithful | Upstream `aire::acesHill` instantiates `AcesFilmicToneMapper`, so it is identical to ACES Filmic; reproduced as-is (the unused `AcesHillToneMapper` is not called). |
| Aldridge | filter.aldridge | aire | faithful | `AldridgeToneMapper` incl. final `pow(.,2.4)` before sRGB encode; cutoff 0 / negative cutoff give NaN→black like upstream. |
| Bloom | filter.bloom | Trickle | faithful | Reads raw RGBA bytes (no ColorGetter R/B swap involved). Soft-knee bright pass in 0..255 units, Gaussian kernel radius r / sigma r/2 truncated, clamped edges, `pow(x,1/gamma)`. Radius is px-scaled (`ctx.ipx`). |
| BokehLights | filter.bokeh_lights | ImageToolbox | faithful | Procedural GLSL ported (hash/hash2 in float32, field transform, polygonal discs, HSV hue jitter, `min(1, rgb + light*intensity)` = unpremultiplied form of the shader). Assumes texture y = 0 at the top. Light colour default `-675749` = #F5B05B. |
| Brightness | filter.brightness | aire | faithful | Lead's port verified: `adjustment(gain=1,bias)` = rgb + bias, range/default match. |
| CinematicDream | filter.cinematic_dream | G'MIC | faithful | Lead's port verified: params, ranges, defaults and command identical to the Kotlin builder. **Upstream issue:** ImageToolboxLibs runs G'MIC with `GmicExecutionOptions.disableSmooth = true` (default), which skips the builtin `smooth`; `fx_highlight_bloom` starts with `+smooth …` followed by `-.. .`, so in ImageToolbox this pipeline most likely fails ("Invalid selection [-2]"). LazyRoom runs the intended G'MIC filter. No other light-group G'MIC command uses `smooth`. |
| Clahe | filter.clahe | aire | faithful | `clahe_yuv_rgba`: CLAHE on 8-bit full-range YCgCo Y (256 bins). Crate quirks reproduced: clip excess counts the whole bin value, `int(threshold*area/bins)` clip limit (small tiles → limit 0 → collapse), extrapolating bilinear interpolation in half-border tiles, Rust NaN-ignoring min/max. Upstream panics when grid > image size; we clamp the tile size to ≥ 1 px. |
| ClaheHSL | filter.clahe_hsl | aire | faithful | As above on colorutils HSL lightness, `bins_count` levels; hue stored as whole degrees and saturation quantised like upstream's u16 planes. |
| ClaheHSV | filter.clahe_hsv | aire | faithful | As above on HSV value. |
| ClaheJzazbz | filter.clahe_jzazbz | aire | faithful | colorutils Jzazbz at 200 nits; Jz(white) ≈ 0.22 so only ~22 % of the bins are used (upstream behaviour, visible posterisation). |
| ClaheLAB | filter.clahe_lab | aire | faithful | **Upstream bug reproduced:** the crate stores a*/b* as `(v*100+100) as u16`, so every a*/b* below −1 saturates to −1 (greens/blues lose chroma). Lead may decide to "fix" it (would become approximate). |
| ClaheLUV | filter.clahe_luv | aire | faithful | Same u*/v* storage bug as LAB, reproduced. |
| ClaheOklab | filter.clahe_oklab | aire | faithful | L quantised to `bins-1`, a/b kept as float (as upstream). |
| ClaheOklch | filter.clahe_oklch | aire | faithful | L as Oklab; C/h float → numerically equal to ClaheOklab (as upstream). |
| Contrast | filter.contrast | aire | faithful | `gain*(rgb-0.5)+0.5`. Param named `gain` (aire argument), title "Contrast". |
| Drago | filter.drago | aire | faithful | Vectorised (SIMD) path of `DragoToneMapper` (Lwa = 1, bias 0.85, Lmax = threshold, Rec.601 luma); threshold 0 → NaN → black like upstream. Upstream's scalar tail (last 1–4 px per row) uses a slightly different Lmax; not replicated. |
| DynamicContrast | filter.dynamic_contrast | G'MIC | faithful | `rep_dynamic_contrast a,t,i`. |
| EqualizeHistogramAdaptive | filter.equalize_histogram_adaptive | aire | faithful | `ahe_yuv_rgba` (no clipping), YCgCo Y, 256 bins. |
| EqualizeHistogramAdaptiveHSL | filter.equalize_histogram_adaptive_hsl | aire | faithful | `ahe_hsl_rgba`. |
| EqualizeHistogramAdaptiveHSV | filter.equalize_histogram_adaptive_hsv | aire | faithful | `ahe_hsv_rgba`. |
| EqualizeHistogramAdaptiveLAB | filter.equalize_histogram_adaptive_lab | aire | faithful | `ahe_lab_rgba`; a*/b* u16 storage bug reproduced (see ClaheLAB). |
| EqualizeHistogramAdaptiveLUV | filter.equalize_histogram_adaptive_luv | aire | faithful | `ahe_luv_rgba`; u*/v* storage bug reproduced. |
| EqualizeHistogramHSV | filter.equalize_histogram_hsv | aire | faithful | `hist_equal_hsv_rgba` global CDF mapping of V. |
| EqualizeLight | filter.equalize_light | G'MIC | faithful | `fx_equalize_light amount,mode,channel` (enum indices). |
| EqualizeShadow | filter.equalize_shadow | G'MIC | faithful | `fx_equalize_shadow a`. |
| Exposure | filter.exposure | GPUImage | faithful | `rgb * 2^exposure`. |
| Fireworks | filter.fireworks | ImageToolbox | faithful | Procedural GLSL ported; `mode` kept as an Int slider 0..2 like ImageToolbox. Trail contributions < 1e-7 are skipped (below float noise). |
| Flare | filter.flare | jhlabs | faithful | Native Rust `FlareFilter` (custom centre, integer radius = r·max(w,h)/2, Perlin `noise1` with the Rust seed `0x4a484c616273` and `java.util.Random` replica). Distances are evaluated in full-resolution pixels so previews match exports. |
| Gamma | filter.gamma | aire | faithful | **Upstream quirk reproduced:** aire builds `lut[i] = clamp(i^gamma, 0, 255)` on the raw 0..255 code value (not normalised), so gamma ≠ 1 is extreme (γ=2 → almost white, γ=0.5 → near black). UI default 1.0 (impl class says 1.5). |
| Glow | filter.glow | jhlabs | faithful | Native Rust: `rgb + 4*amount*gauss(rgb, radius 2)`, jhlabs kernel (sigma r/3); radius px-scaled for previews. |
| HableFilmicToneMapping | filter.hable_filmic_tone_mapping | aire | faithful | Uncharted-2 curve without white normalisation (as aire). |
| Haze | filter.haze | GPUImage | faithful | `(c-d)/(1-d)`, `d = y*slope + distance`; assumes texture y = 0 at the top (only matters with slope ≠ 0). Upstream also applies it to alpha; alpha is preserved here. |
| HejlBurgessToneMapping | filter.hejl_burgess_tone_mapping | aire | faithful | SIMD path (offset 0.0004; scalar tail uses 0.004). |
| HighlightSynthesis | filter.highlight_synthesis | G'MIC | faithful | `iain_highlight_synthesis r,g,b,srgb`. |
| HighlightsAndShadows | filter.highlights_and_shadows | GPUImage | faithful | ImageToolbox's `GPUImageHighlightShadowWideRangeFilter(shadows, 1)` shader; lum = 0 → black (GPU 0/0). Param `shadows`. |
| Levels | filter.levels | jhlabs | faithful | Native Rust transfer incl. `high += 1/255` when low == high. |
| LightEffects | filter.light_effects | jhlabs | faithful | Native Rust `LightFilter`, which is a simplified bump shader (not the full Java jhlabs Phong model): blurred brightness gradient, default distant light (az 270°, el 30°) → flat areas become 0.5×. Bump softness px-scaled. |
| LightGlow | filter.light_glow | G'MIC | faithful | `fx_lightglow density,amp,mode,opacity,channel`. |
| LightPatch | filter.light_patch | G'MIC | faithful | `fx_light_patch …` prefixed with `srand <ctx.rng>`: the command uses `noise` and is clock-seeded upstream; seeding makes renders deterministic. |
| LocalContrastEnhancement | filter.local_contrast_enhancement | G'MIC | faithful | `fx_LCE`; spatial radius (a Gaussian sigma in px) is scaled with `ctx.px` (identical at full resolution). |
| LogarithmicToneMapping | filter.logarithmic_tone_mapping | aire | faithful | `log(1+L)/log(1+e)` on Rec.601 luma; exposure 0 → black (NaN upstream). |
| Mobius | filter.mobius | aire | faithful | **SIMD path reproduced:** its `twos` constant is 1.0 (scalar tail uses 2.0), so a/b differ from the textbook Möbius formula; pixels ≤ transition unchanged. |
| PopShadows | filter.pop_shadows | G'MIC | faithful | `fx_pop_shadows s,scale,normalize`. |
| Rays | filter.rays | jhlabs | faithful | Native Rust: 16 exact-bilinear samples towards the centre, average of samples with brightness ≥ threshold, composited with opacity. UI default opacity 0.75. |
| ReliefLight | filter.relief_light | G'MIC | faithful | `fx_light_relief` (11 args). "Opacity as heightmap" only sees RGB, as with ImageToolbox's preserve-alpha mode. |
| Retinex | filter.retinex | G'MIC | faithful | `fx_retinex`; low/middle/high scales (Gaussian sigmas in px) scaled with `ctx.px`. |
| ShadowPatch | filter.shadow_patch | G'MIC | faithful | `fx_shadow_patch` prefixed with `srand` (uses `plasma`, random upstream). |
| Sparkle | filter.sparkle | jhlabs | faithful | Native Rust `SparkleFilter`, ray lengths from `java.util.Random(371).nextGaussian()` (exact LCG replica). Full-resolution pixel coordinates for previews. |
| ToneEnhance | filter.tone_enhance | G'MIC | faithful | `gcd_tone_enhance` (13 args). |
| Uchimura | filter.uchimura | aire | faithful | Gran-Turismo curve (`step` evaluated as x ≥ edge). |
| Vignette | filter.vignette | GPUImage | faithful | `mix(rgb, colour, smoothstep(start,end,|uv-0.5|))`. GPUImage writes alpha = 1; alpha is preserved here (identical for opaque images). |
| WhiteBalance | filter.white_balance | GPUImage | faithful | GLSL `mat3` constants are column-major, so the GPU applies the transposed YIQ matrices — reproduced. Temperature/tint mapping from `setTemperature`/`setTint`. |
| ZoneSystem | filter.zone_system | G'MIC | faithful | `fx_zonesystem`; gamma is clamped to ≥ 0.01 because the command takes `log(gamma)` and G'MIC errors at the slider minimum 0 (ImageToolbox would fail there too). |


## Lead decisions (bug fixes over parity)

* **Gamma** — corrected to a normalised curve (`out = in ** gamma`). Upstream raises the raw 0–255 code
  value to the power, which makes the filter unusable for any value other than 1.0. Status: approximate (bug fixed).
* **CLAHE LAB/LUV and Equalize Histogram Adaptive LAB/LUV** — a\*/b\* (u\*/v\*) are kept exactly instead of
  the upstream `(v*100+100) as u16` storage that clamps all negative chroma (greens/blues). Status: approximate (bug fixed).
