# Port report: ImageToolbox "Glitch" group (31 filters)

Module: `src/lazyroom/ops/itb/glitch.py` (registrations), helpers in
`src/lazyroom/ops/itb/_glitch_impl.py`, tests in `tests/itb/test_glitch.py`.

## Sources read

* ImageToolbox: `Ui*Filter.kt` (ranges, defaults, `GmicParameterInfo`), `data/model/*Filter.kt`,
  `GMICFilterTransformation.kt`, `ProceduralFilter.kt`, `utils/gpu/{ProceduralShaders,
  ProceduralGpuImageFilter,BitmapGpuImageFilter}.kt`, `enums/ProceduralEffect.kt`,
  `utils/glitch/tools/JpegGlitch.kt`, `params/{NtscParams,GlitchParams}.kt`,
  `widget/filterItem/NtscParamsItem.kt` (NTSC param index -> field mapping), `ColorUtils.kt`, strings.xml.
* ImageToolboxLibs G'MIC: `filters/{Artifact,Composite,Creative,Deformation,Transform,GmicCommon,RawGmic}Filters.kt`,
  `Gmic.kt`, `model/*`, `cpp/gmic_jni.cpp` (alpha handling, output size).
* Trickle: `cpp/ProcessingFilters.cpp` (glitchVariant, vhsGlitch, blockGlitch, crtCurvature,
  pixelMelt), `pipeline/EffectsPipelineImpl.kt`, `NtscSettings.kt`, `rust/ntsc_jni/src/lib.rs`.
* ntsc-rs (`ntsc-rs/ntsc-rs`, rev `add90f5`, the revision pinned by `ntsc_jni/Cargo.toml`; cloned to
  `/tmp/ref_ntsc/ntsc-rs`): `crates/ntscrs/src/{ntsc,filter,shift,yiq_fielding,random,noise,noise/*}.rs`,
  `settings/standard.rs`; plus the exact dependency sources `rand 0.9.2`, `rand_core 0.9.5`,
  `rand_xoshiro 0.7.0`, `siphasher 1.0.2` (`/tmp/ref_ntsc/crates`).
* aire: `shift/Glitch.cpp`, `shift/WindStagger.cpp`, `halftone/Halftone.cpp`, `color/Blend.h`,
  `color/ConvolveToneMapper.cpp`, `color/tone/AldridgeToneMapper.hpp`, `color/eotf-inl.h`,
  `algo/MathUtils.hpp`, `jni/{ShiftPipelines,ToneMappingPipelines,AcquireBitmapPixels}.cpp`,
  `pipeline/ShiftPipelineImpl.kt`.
* G'MIC stdlib (dev CLI 3.7.6) and the 4.0.2 stdlib embedded in ImageToolboxLibs (decoded copy in
  `/tmp/distort_stdlib/stdlib_itb.gmic`): every stdlib command reached by these filters is
  identical in both versions and none of them calls `smooth` (which ImageToolbox disables).

## Cross-cutting notes

* **Android byte order**: Trickle C++ reads RGBA_8888 pixels as little-endian `uint32`, so its
  "r" (`>> 16`) is blue and "b" (`& 0xFF`) is red. Net effect reproduced: in GlitchVariant, VHS and
  CRT curvature red is taken from `x - shift` and blue from `x + shift`.
* **Premultiplied alpha**: Trickle/aire/GPU code operates on premultiplied bitmaps. Ports that mix
  channels of different pixels (or sample bilinearly) premultiply, process, and unpremultiply;
  fully transparent pixels therefore lose their colour, exactly as on Android.
* **Randomness**: libc `rand()` / clock-seeded engines are replaced by `ctx.rng` draws (same
  distributions; per-pixel draws use a SplitMix64 hash of (stream seed, index) so the parallel
  numba kernels are deterministic). Explicit upstream seeds keep their meaning: HuffmanGlitch,
  OffsetStripes (G'MIC `srand $seed`), VHS NTSC (ntsc-rs seed, independent of the render seed),
  the procedural shaders' `seed` uniforms and Glitch's `seed` (a byte position, not an RNG).
* **G'MIC determinism**: AnalogDamage, Breaks, ChromaticAberrations (random type) and PixelSort
  (random criterion/mask) prefix `srand <ctx.rng>`. That alone is not enough: `noise` and `u()`
  in parallel fills use per-thread RNGs, so those runs also set `OMP_NUM_THREADS=1` for the G'MIC
  subprocess (reference-counted env var in `_glitch_impl._gmic_single_thread`). Verified
  bit-identical repeated runs on 600x400. (Other groups chose `deterministic=False` instead.)
* **G'MIC alpha**: Preserve mode passes RGB and keeps the source alpha, ignoring alpha G'MIC
  returns (fx_streak's `to_rgba`); Process mode (Breaks, OffsetStripes, RowShifter) passes RGBA.
* **px scaling**: G'MIC amplitudes/smoothness in pixels (aberrations, breaks) use `ctx.px`;
  PixelMelt's max drop is scaled by `ctx.scale`; VHS NTSC multiplies ntsc-rs's bandwidth scales by
  `ctx.scale` (like its "scale with video size"). All identity at export scale.
* **Performance** (12 MP RGBA, machine load ~10): numba-parallel kernels, 0.3-2.1 s
  (VHS NTSC ~1-2 s, Glitch/EnhancedGlitch ~1.7 s); G'MIC filters run at the CLI's speed (the
  random ones single-threaded).

## Table

| Filter | op id | upstream | status | notes |
|---|---|---|---|---|
| AnalogDamage | filter.analog_damage | G'MIC | faithful | Exact `AnalogDamage` pipeline: power-of-two padded `fx_dirty …,0,1` + red/cyan `fx_chromatic_aberrations` + `fx_jpeg_artefacts`; seeded, single-threaded |
| Breaks | filter.breaks | G'MIC | faithful | `fx_breaks`, Process alpha (transparent boundary adds alpha); seeded, single-threaded |
| BufferError | filter.buffer_error | G'MIC | faithful | `fx_buffer_error`; the pixel buffer is re-read at width%/height%, so the output size changes (50%x50% by default, as in ImageToolbox) -> `geometry=True`, source alpha resized |
| ChromaticAberrations | filter.chromatic_aberrations | G'MIC | faithful | `fx_chromatic_aberrations` with colours as Android `Color.red/green/blue`; seeded when a type is "random" |
| HuffmanGlitch | filter.huffman_glitch | G'MIC | faithful | `fx_huffman_glitches` (seed parameter) |
| JpegArtifacts | filter.jpeg_artifacts | G'MIC | faithful | `fx_jpeg_artefacts` |
| MessWithBits | filter.mess_with_bits | G'MIC | faithful | `fx_mess_with_bits` (47 channel choices) |
| OffsetStripes | filter.offset_stripes | G'MIC | faithful | `fx_offset_stripes` (seed parameter), Process alpha |
| PixelSort | filter.pixel_sort | G'MIC | faithful | `fx_pixelsort` (mask enum values 1..3); `lower > upper` raises INVALID_PARAMS like the Kotlin `require` |
| RowShifter | filter.row_shifter | G'MIC | faithful | `fx_row_shift`, Process alpha. The stdlib pads by 2 px (`z 0,0,{w+1},{h+1},2`) and its final crop uses the padded size, so the output is 2 px wider/taller (same in 4.0.2) -> `geometry=True` |
| Streak | filter.streak | G'MIC | faithful | `fx_streak` incl. mask alpha; RGB input as in Preserve mode (so a mask alpha < 255 never matches, as upstream) |
| BlockGlitch | filter.block_glitch | Trickle | faithful | blockGlitch: blocks of max(4, w·size) px copied from (±w·0.15·s, ±block) offsets with probability 0.4·s |
| CrtCurvature | filter.crt_curvature | Trickle | faithful | crtCurvature: truncating barrel warp, R/B offset (byte-order corrected), quadratic vignette |
| GlitchVariant | filter.glitch_variant | Trickle | faithful | glitchVariant: random 1-5 px bands shifted (later iterations win), R/B split taken from the original |
| PixelMelt | filter.pixel_melt | Trickle | faithful | pixelMelt per-column persistent drops; `max_drop = 0` is undefined upstream (`rand() % 0`) and is treated as no-op |
| VHS | filter.vhs | Trickle | faithful | vhsGlitch: sine wobble (`seed` = time), 8%·s line jitter, R/B split, shared ±20·s/255 noise, rows `y % 3 == 0` x0.92 |
| VHSNtsc | filter.vhs_ntsc | Trickle | faithful | Line-by-line port of ntsc-rs `NtscEffect` (YIQ fields/interleaving, IIR filters with scipy-style initial conditions, QAM modulation/4 demodulators, head switching, tracking noise, snow, edge wave, VHS tape filters, chroma loss/delay/phase noise, vertical blend) with ImageToolbox's `toNtscSettings()` amount scaling and the JNI sanitising. RNG streams reproduced (SipHash-2-4 Seeder over Rust `Hash` bytes, Xoshiro256++/SplitMix64, rand 0.9 float conversion, geometric sampler, jump) and verified against reference vectors; IIR filters checked against scipy `lfilter`. Not compared end-to-end with the Android binary (no Rust toolchain); float rounding (no FMA) and keeping float instead of u8 truncation give sub-LSB differences; straight instead of premultiplied RGB for semi-transparent pixels. All 49 sliders + 8 enum selectors + 8 toggles exposed (advanced ones flagged) |
| EnhancedGlitch | filter.enhanced_glitch | aire | faithful | aire glitchEffect: G/B shift, corruption bands, 2-px halftone lines at 15%, premultiplied output made opaque, last row black (aire copies only h-1 rows). Wiring kept as in ImageToolbox: `EnhancedGlitchFilter` passes channelShiftX as the corruption X shift and aire's `ShiftPipelineImpl.glitch` passes channelShiftX as the Y shift, so **channel_shift_y and corruption_shift_x have no effect** (documented in the param descriptions) |
| HorizontalWindStagger | filter.horizontal_wind_stagger | aire | approximate | Algorithm identical (gaussian-profiled per-stream lengths, linear resampling, last streamed row restored). Deviation: ImageToolbox's `Color.toAbgr()` permutes the clear colour's channels (R<-G, G<-B, B<-A, A<-R) which gives an unrelated/invalid premultiplied colour on Android; we use the picked colour. Default (transparent) is identical |
| ColorAnomaly | filter.color_anomaly | aire | faithful | `AldridgeFilter(0.5 to cutoff)`: aire Aldridge tone mapper in linear light with aire's 0.045 sRGB threshold; negative curve values (NaN pow upstream) -> 0 |
| Glitch | filter.glitch | ImageToolbox | approximate | JpegGlitch algorithm exact (q100 JPEG, header length always 417 because the Kotlin signed-byte check never matches, float32 byte-index maths, `floor(amount·2.56)` byte). The JPEG bytes come from OpenCV/libjpeg-turbo (q100, 4:2:0, optimized Huffman) instead of Android Skia, so corrupted positions and the lenient decode can differ; output is opaque (JPEG) like upstream. libjpeg prints "Corrupt JPEG data" warnings to stderr |
| ColumnStreak | filter.column_streak | ImageToolbox | faithful | ProceduralShaders.kt GLSL on the CPU (float32, `toField`/`sampleField` mirror-repeat, GL_LINEAR clamp-to-edge sampling of the premultiplied texture), incl. offset/scale/rotation uniforms |
| ConcentricCircleBreaks | filter.concentric_circle_breaks | ImageToolbox | faithful | as above |
| ConcentricSquareBreaks | filter.concentric_square_breaks | ImageToolbox | faithful | as above |
| InlineBreaks | filter.inline_breaks | ImageToolbox | faithful | as above |
| RadialStreak | filter.radial_streak | ImageToolbox | faithful | as above |
| RandomColorDispersion | filter.random_color_dispersion | ImageToolbox | faithful | as above (R/G/B from three field samples, max alpha) |
| SpiralBreaks | filter.spiral_breaks | ImageToolbox | faithful | as above |
| StreakCircles | filter.streak_circles | ImageToolbox | faithful | as above (up to 64 circles, ring-quantised samples mixed sequentially) |
| StreakExpand | filter.streak_expand | ImageToolbox | faithful | as above |
| TiledStreak | filter.tiled_streak | ImageToolbox | faithful | as above |

Totals: 29 faithful, 2 approximate, 0 not ported.
