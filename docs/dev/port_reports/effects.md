# Port report: ImageToolbox "Effects" group (41 filters)

Module: `src/lazyroom/ops/itb/effects.py`, helpers: `src/lazyroom/ops/itb/_effects_impl.py`,
tests: `tests/itb/test_effects.py` (extremes for every op + semantic checks, RawGmic safety and Shader tests).

Summary: **38 faithful, 3 approximate, 0 not ported.**

| Filter | op id | upstream | status | notes |
|---|---|---|---|---|
| BlackHat | filter.black_hat | aire (fast_morphology) | faithful | closing - image, saturating u8. Same aire kernels (box, or disk of radius size/3 in a size x size window), REFLECT_101, anchor size/2. Like ITB, images with any non-opaque pixel run in RGBA mode on premultiplied values (alpha is morphed too). ITB throws for a circle kernel with size < 5; we build the disk anyway (size 1 = identity). |
| Closing | filter.closing | aire | faithful | dilate then erode; see BlackHat. |
| Dilation | filter.dilation | aire | faithful | see BlackHat. Verified pixel-exact against cv2.dilate with the aire kernel. |
| Erode | filter.erode | aire | faithful | see BlackHat. |
| MorphologicalGradient | filter.morphological_gradient | aire | faithful | dilate - erode, saturating. |
| Opening | filter.opening | aire | faithful | erode then dilate. |
| TopHat | filter.top_hat | aire | faithful | image - opening, saturating. |
| Canny | filter.canny | OpenCV | faithful | Same calls as `ImageProcessing.canny`: RGBA2GRAY, medianBlur 3, Canny(t1, t2). Alpha is preserved (ITB returns an opaque bitmap). |
| Chrome | filter.chrome | jhlabs | faithful | Port of the ImageToolboxLibs Rust `ChromeFilter` (4-neighbour (r+g+b)/3 bump, 1-exp exposure curve, grey output), not the original Java one. |
| Contour | filter.contour | jhlabs | faithful | Rust `contour_exact`, incl. f32 table maths, Rust saturating casts/i32 wrap at levels 0/1, `composite_normal(source, colour, strength)`: the source is composited *over* the contour colour, so weak edges show the colour and strong edges stay unchanged. `scale` is an Int 0..1 because ITB's slider uses roundTo=0. |
| Convolution3x3 | filter.convolution_3x3 | aire (libblur) | faithful | libblur `filter_2d_rgba` u8 path: correlation, REFLECT_101, weights quantised to Q15 and saturated to i16 (effective weights clip to about [-1, 1]), rounding shift, saturation (ARM/neon behaviour). RGBA mode convolves alpha too. |
| CopyMoveDetection | filter.copy_move_detection | OpenCV | approximate | Same algorithm (16x16 block DCT, retain x retain coefficients / q -> u8, stable signed-byte sort, shift-vector voting > 10, blocks painted with java.util.Random(trunc(|shift|)) colours, 20/80 blend), with swapped R/B gray weights as upstream (COLOR_BGR2GRAY on RGBA). Deviations: DCT computed with separable float64 filters (coefficients with abs < 1e-6 set to 0 to mimic cv::dct's exact zeros; extremely rare rounding/sign differences possible); q = 0 (the UI default) reproduces ARM's saturating float->int conversion of x/0 (positive -> 255, else 0); retain > 16 is clamped to 16 (upstream throws); painted blocks keep the source alpha (upstream's 3-value Scalar writes alpha 0, making painted blocks 80% transparent); a memory guard raises RESOURCE_LIMIT above 768 MiB of signatures. Painting uses an equivalent "last write wins" max filter. Worst case ~20 s on a 12 MP noise image. |
| ErrorLevelAnalysis | filter.error_level_analysis | OpenCV | faithful | JPEG re-encode at quality, absdiff, NORM_MINMAX. libjpeg build may differ from Android's. Alpha preserved (ITB returns opaque). |
| Cracks | filter.cracks | G'MIC | faithful | `fx_cracks density,relief,r,g,b,a,channel`. Run single-threaded (see notes). |
| CrtPhosphors | filter.crt_phosphors | G'MIC | faithful | `fx_crt_phosphors` with `upscaleFactor-1` like the Kotlin builder; geometry op, output is upscale_factor x larger (ITB shrinks only its preview input by the factor). |
| CrtScanlines | filter.crt_scanlines | G'MIC | faithful | `fx_crt_scanlines`; geometry op, output upscale_factor x larger. |
| Dirty | filter.dirty | G'MIC | faithful | `fx_dirty` inside ImageToolboxLibs' power-of-two padding wrapper (exact string). |
| Filaments | filter.filaments | G'MIC | faithful | `fx_filaments`, GmicAlphaMode.Process (RGBA in, G'MIC alpha out, opaque alpha added for RGB input). Single-threaded. |
| FillHoles | filter.fill_holes | G'MIC | faithful | `fill_holes` with FillHolesChannelS enum values. |
| Fractalize | filter.fractalize | G'MIC | faithful | `fractalize detail`. |
| FreakyBlackAndWhite | filter.freaky_black_and_white | G'MIC | faithful | `fx_freaky_bw`. |
| Ghost | filter.ghost | G'MIC | faithful | `fx_ghost`. Single-threaded. |
| Grid | filter.grid | G'MIC | faithful | `grid h%,v%,x%,y%,opacity,r,g,b,a` (ITB builds this RawGmicFilter string itself). |
| Isophotes | filter.isophotes | G'MIC | faithful | `fx_isophotes`; G'MIC's alpha is ignored as in GmicAlphaMode.Preserve. |
| LocalOrientation | filter.local_orientation | G'MIC | faithful | `fx_local_orientation`. |
| MetallicLook | filter.metallic_look | G'MIC | faithful | `fx_tk_metallic`. |
| OldMovieStripes | filter.old_movie_stripes | G'MIC | faithful | `fx_stripes_y`. |
| RainAndSnow | filter.rain_and_snow | G'MIC | faithful | `fx_rain ... rv blend screen,opacity`. |
| RetroFade | filter.retro_fade | G'MIC | faithful | `fx_retrofade`. |
| SixtiesCinema | filter.sixties_cinema | G'MIC | faithful | `fx_gb_cfx`. |
| Skeleton | filter.skeleton | G'MIC | faithful | `fx_skeleton`. |
| Dissolve | filter.dissolve | jhlabs | faithful | Rust port: alpha *= smoothstep(min, max, rand) with the exact java.util.Random(0) float sequence (verified against a JDK). Treats alpha as straight (Android data is premultiplied). |
| DoG | filter.dog | jhlabs | faithful | Rust `box_blur_exact` (3 integer box passes + fractional pass, premultiplied) for both radii, `max(blur2 - blur1, 0)`, integer normalisation. Includes jhlabs' `blurFractional` quirk (first/last column/row replaced by pixels of the first line). Radii in px, scaled for previews. |
| Emboss | filter.emboss | aire | faithful | aire emboss matrix through the same Q15 libblur path (weights saturate at +-1, so intensities above 1 look like 1 at the centre/corners). Alpha preserved (aire's RGB mode returns opaque). |
| Grain | filter.grain | aire | faithful | One N(0, 127*intensity) sample per pixel, truncated, added to R, G and B (monochrome), saturated. Random values come from ctx.rng (upstream: time-seeded std::default_random_engine). |
| Lookup | filter.lookup | GPUImage | faithful | GPUImageLookupFilter maths. ImageToolbox never binds the lookup texture, so the shader samples an incomplete texture = (0,0,0,1) and the result is `color * (1 - intensity)` (default -1 = x2 brightening); reproduced. Extra: optional `lut` File (512x512 GPUImage lookup or Hald CLUT) applies a real LUT with the same trilinear sampling. |
| Noise | filter.noise | Trickle | faithful | `value | (rand() % threshold)` per 8-bit channel (channel-symmetric, so Trickle's R/B swap does not matter). threshold 0 is a no-op (upstream divides by zero). |
| RawGmic | filter.raw_gmic | G'MIC | approximate | Runs the user pipeline (GmicAlphaMode.Preserve, geometry may change) behind a strict safety policy (see below), with `network -1` prepended, 300 s timeout. Default command changed from ITB's `fx_gmicky 0` (needs to download a sample image from gmic.eu, which the no-network policy forbids) to `blur 3`. ITB's auxiliary input images are not supported. |
| Shader | filter.shader | GPUImage (custom GLSL) | approximate | Real GPU rendering: headless Qt OpenGL (PySide6, QOffscreenSurface + FBO, GL_RGBA8 textures, GL_LINEAR + CLAMP_TO_EDGE, GPUImage quad so textureCoordinate (0,0) = top-left) in a separate `python -I` process (a GL crash cannot take down the app). Accepts GLSL source or a full `.itshader` JSON preset; `values` JSON sets uniforms (float/int/bool/color->vec4/vec2). Validation follows ShaderValidator/ShaderParser. Deviations: GLSL ES 1.00 is compiled as desktop GLSL 1.20 (precision qualifiers ignored, i.e. highp everywhere; ES-only `#version`/`#extension GL_OES_standard_derivatives` lines stripped); invalid/failed shaders raise INVALID_PARAMS with the validator/compiler messages instead of silently returning the input; for plain GLSL the declared uniforms become parameters (GL default 0 when unset). Without PySide6 or an OpenGL-capable display (X11/Wayland; tried offscreen, wayland, xcb) it raises BACKEND_UNAVAILABLE with a hint; tests skip in that case. Empty shader = no-op (no GL needed). |
| SobelEdgeDetection | filter.sobel_edge_detection | GPUImage | faithful | Grayscale pass (0.2125/0.7154/0.0721) quantised to 8 bits like the intermediate RGBA8 FBO, Sobel at +-line width with bilinear clamp-to-edge sampling, magnitude. Line width in px (scaled for previews, fractional offsets bilinear). Alpha preserved (GPUImage writes alpha 1). |
| Threshold | filter.threshold | aire | faithful | aire thresholds the **red** channel only (`R > level` -> white, written to R, G, B); reproduced although the UI calls it "Luminance threshold". |

## RawGmic safety policy (`_effects_impl.check_raw_gmic`)

Rejected with `INVALID_PARAMS` (hint explains the policy; also documented in the op description):

* non-printable/non-ASCII characters, backslashes, single quotes (math strings), backquotes;
* any `$` substitution except loop/list counters (`$>`, `$<`, `$!`, `$%`, `$^`, `$|`): no variables, environment variables, `${"-pipeline"}`, `$$cmd`;
* `{...}` substitutions producing strings / image names / text (`{/...}`, `{*...}`, `{^...}`, `{...,t}`, `{...,n}`, `{...,b|f|x|^|@...}`);
* math functions with side effects: `run( ext( fsize( isfile( isdir( date( files( display( set(`;
* commands (any `+`/`-` prefix and `[selection]`/`.` suffix) from a denylist: exec/x/xo/shell/run, input/i and all `input_*` (incl. `input_cached`), output/o and all `output*`/on/op/ow/ox/ot, command/m, download, network, update/up, camera, delete, createdir, cursor, screen, select, wait, window/w*, display/d*, sample/sp, lorem, echo_file, files, video/file batch commands, plus name prefixes (`input`, `output`, `display`, `exec`, `x_`, `file`, `path_`, `update`, `upload`, `download`, `camera`, `window`, `pink_input`, `gmd2`, `parse_`, `reference`), plus 170 further G'MIC 4.0.2 stdlib commands that (transitively) execute programs or write/delete files (call-graph scan of `stdlib.gmic`, e.g. `fx_tk_video3D`, `fx_pack`, `fx_send_feedback`, `fx_denoise`);
* a bare `x` token, except as the axis argument of `a`/`append`/`mirror`/`s`/`split` when that command is itself in command position (`nm a x "id"` is rejected);
* anything that looks like a file path or URL in any token or comma-separated argument: `://`, `http/ftp/file/www` prefixes, leading `/`, `\`, `~`, drive letters / `type:` prefixes, `../`, `dir/sub/`, `name.ext`.

Quoting is ignored by the check (words inside strings and sub-pipelines are checked too), so it is conservative and can reject harmless pipelines (e.g. `f "x"`; write `f "x*1"`). Network loading is additionally disabled inside G'MIC (`network -1`), which also neutralises stdlib commands that fetch resources from gmic.eu. Remaining risk: resource exhaustion (huge images / endless loops) is only bounded by the 300 s timeout.

## Notes for the lead

* **G'MIC determinism**: every G'MIC op prefixes `srand <ctx.rng>`. `fx_cracks`, `fx_filaments` and `fx_ghost` draw random numbers in multi-threaded math expressions, so they additionally run with `OMP_NUM_THREADS=1` (reference-counted os.environ change, same mechanism as `_glitch_impl.py`). The two modules keep **independent** reference counts; with concurrent renders across both groups the variable could be left at 1 (slower G'MIC, never wrong results). Suggest moving one shared helper into `_common`.
* **Shader backend**: it needs PySide6 (optional `gui` extra) and a display. `core/backends.py` has no `opengl` check, so the op is listed as available and raises BACKEND_UNAVAILABLE at run time; adding an `opengl`/`pyside6` backend check to `core/backends.py` would let `requires=[...]` report it up front.
* The dev CLI is G'MIC 3.7.6 (ImageToolbox bundles 4.0.2). All 18 group commands were verified to run with it (and the generated strings checked against the Kotlin builders). This machine also has `~/.config/gmic/update376.gmic` (G'MIC's auto-downloaded stdlib update), which the CLI loads at startup.
* Timings on a 12 MP image (busy machine, includes registry overhead): morphology 1.3-2 s (size 150 disk ~3 s), Canny 1.3 s, ELA 1 s, Chrome 2.3 s, Contour 2.6 s, Dissolve 3 s, DoG 1.9 s, Convolution/Emboss 1.6-3.4 s, Grain/Noise 1.2-1.3 s, Sobel 2.1 s, Lookup 1.1 s; CopyMoveDetection 20+ s (O(N log N) over 12M blocks).
* Test run: `LAZYROOM_ITB_ONLY=effects .venv/bin/python -m pytest tests/itb/test_effects.py tests/test_ops_smoke.py -q -x` passes (~35 s); the Shader GL tests skip without a display.
