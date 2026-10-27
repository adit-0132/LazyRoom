# Port report: ImageToolbox extras (watermarking, filter templates, formats)

Upstream: ImageToolbox by T8RIN (Apache-2.0) — `feature/watermarking`, `core/filters` (template model),
`feature/filters/data/utils/serialization`, `feature/scan-qr-code`, `feature/format-conversion`, `feature/jxl-tools`,
`feature/weight-resize`, `core/data` (image formats); ImageToolboxLibs (Apache-2.0) `libs/androidwm` (AndroidWM by
huangyz0918, maintained there by T8RIN; the C++ `Watermark.cpp` implements the hidden watermarks).

## Code

| File | Contents |
|---|---|
| `src/lazyroom/ops/watermark_ext.py` | ops `draw.repeating_text`, `draw.repeating_image`, `draw.stamp_watermark`, `draw.timestamp_watermark` |
| `src/lazyroom/ops/_wm_render.py` | AndroidWM text bitmap, rotation, tiling, Skia Porter-Duff/blend modes, `SimpleDateFormat` port |
| `src/lazyroom/tools/itb_extras.py` | tools `watermark_hidden_embed`, `watermark_hidden_extract`, `template_import`, `template_export` (+ imports `_itb_formats`) |
| `src/lazyroom/tools/_itb_stego.py` | bit-compatible LSB / FD (8x8 DCT) embed + extract, modified UTF-8, payload codecs |
| `src/lazyroom/tools/_itb_templates.py` | template envelope + value codec (Java `Float.toString`, Base64, enums, colours, ...) |
| `src/lazyroom/tools/_itb_template_table.py` | generated: 541 ImageToolbox filter classes -> `filter.*` op + per-component param mapping, 76 gradient palettes, 27 fonts |
| `src/lazyroom/tools/_itb_template_gen.py` | dev-time generator of the table (reads the Kotlin sources + live registry, verifies every default); not imported at runtime |
| `src/lazyroom/tools/_itb_formats.py`, `src/lazyroom/tools/_itb_codecs.py` | format tools `jxl_transcode`, `jxl_from_frames`, `jxl_extract_frames`, `telegram_sticker`; encoders (PNG Lossy, ICO, OxiPNG, Jpegli, animated JXL) |
| `src/lazyroom/core/io.py`, `src/lazyroom/tools/convert.py` | additive: optional JPEG XL backend; new `format_convert` choices/params (section 3) |
| `tests/test_itb_extras.py` | tests (section "Tests") |

## Summary

| ImageToolbox feature | LazyRoom | Status | Notes |
|---|---|---|---|
| Watermark: Text (repeat on/off, X/Y offset, angle, alpha, overlay mode, size, colour, background, font) | `draw.repeating_text` | faithful (layout); approximate (glyphs) | AndroidWM geometry, grid tiling from the origin, rotation bbox, alpha applied twice, 29 Skia modes. Fonts: system font instead of Roboto; text drawn at final scale (sharper) |
| Watermark: Image | `draw.repeating_image` | faithful | logo width = size x image width; single-step resize instead of Coil->1024->size |
| Watermark: Stamp (text) | `draw.stamp_watermark` | faithful (layout) | 9 positions, padding / 0.7 x padding, flexible shrink, 12 px rounded box, text-only alpha |
| Watermark: Time stamp | `draw.timestamp_watermark` | faithful + explicit `date` | `SimpleDateFormat` port (English names); empty `date` = now like ImageToolbox |
| `{filename}` placeholder | `watermark_hidden_embed` text | faithful | ops only see pixels: put the name into `text` yourself |
| Digital watermark (text / image, LSB) | `watermark_hidden_embed` / `watermark_hidden_extract` | faithful, bit-identical | same frame, bit order (B, G, R), payloads; automatic detection order |
| Digital watermark (FD = invisible without LSB) | same tools, `method="fd"` | faithful | same DCT, strength, retries, redundancy; float summation order may move a pixel by 1 |
| Filter template -> string / file / QR | `template_export` | faithful (byte-identical strings) | QR verified by decoding it back |
| Scan QR / import template string or file | `template_import` | faithful | -> preset (`presets.save`) and optional project group |
| JPEG XL read/write (+16-bit, EXIF/XMP) | `.jxl` in `core/io.py`, `format_convert(format="jxl")` | faithful (libjxl, optional) | absent from format lists without imagecodecs; `BACKEND_UNAVAILABLE` + hint |
| JXL Tools: JPEG <-> JXL lossless | `jxl_transcode` | faithful | byte-identical JPEG reconstruction verified |
| JXL Tools: images/animations <-> JXL | `jxl_from_frames`, `jxl_extract_frames` | approximate | always loops; durations via tick rewrite / repeated frames |
| PNG Lossy | `format_convert(format="png_lossy", max_colors)` | faithful algorithm | aire Wu quantiser; no dithering (as ImageToolbox actually outputs) |
| ICO 16/32/48/64/128/256 | `format_convert(format="ico_multi")` | faithful | ImageToolbox byte layout |
| Telegram sticker (512 px) | `telegram_sticker` | faithful | + 512 KB check |
| OxiPNG / Jpegli | `format_convert(format="oxipng" / "jpegli")` | faithful / approximate (optional) | Jpegli baseline, not progressive |
| MozJPEG / ImageQuant | — | not implemented | no permissive package (MozJPEG encoder absent; libimagequant GPL) |

## 1. Watermarking

### 1.1 What `draw.watermark` already covers, and what differs

`draw.watermark` (left unchanged) renders single or tiled text/logo with anchor+margin placement; its tiled
layout uses explicit spacing and half-offset rows and has no blend modes.  ImageToolbox (AndroidWM `Watermark`)
instead builds one bitmap (text or image), rotates it about its centre into its bounding box
(`Matrix.setRotate` + `Bitmap.createBitmap(.., filter = true)`), and either tiles it with
`BitmapShader(REPEAT, REPEAT)` over `drawRect(clipBounds)` — a seamless grid starting at (0, 0), period = rotated
bitmap size — or draws it once with its top-left corner at (`positionX` x width, `positionY` x height).  The paint
carries the alpha and the overlay mode (`BlendingMode` = Skia `SkBlendMode`, 29 values).  These are new ops.

Defaults are ImageToolbox's: text "Watermark", size 0.1, black, angle 45, alpha 0.5, repeat on, `SrcOver`;
image size 0.1; stamp "Stamp", size 0.2, padding 20 px, `BottomRight`; time format `dd/MM/yyyy HH:mm`.

### 1.2 Text bitmap (`BitmapUtils.textAsBitmap`)

* Text size `size * 200` px; bitmap width = `min(bounds.width + 20, measureText)`, height
  `int(int(-ascent + 1) + descent + 3) * lines`, background colour fills the bitmap, then the bitmap is scaled so its
  width is `background width * size`.  LazyRoom computes the same geometry and draws the glyphs directly at the
  final scale (ImageToolbox up-scales a small raster, which blurs large watermarks): **deviation, sharper text**.
* `Paint.setAlpha(alpha)` replaces the text colour's own alpha; the bitmap is then drawn with paint alpha again, so
  repeated text has opacity alpha^2 and its background alpha x bg alpha — reproduced and tested.  Stamps draw with
  paint alpha 255, so only the text gets alpha (the stamp box is opaque) — reproduced.
* Multi-line text: ImageToolbox's `StaticLayout` (spacing x2 + 2) is taller than the bitmap height it allocates,
  so lines after the first are clipped in ImageToolbox.  LazyRoom keeps the same line pitch but sizes the bitmap
  to show every line (**deviation, bug not reproduced**).  Widths are measured on the concatenated text as
  upstream.
* Fonts: ImageToolbox uses Roboto / its bundled fonts; LazyRoom resolves `font` through fontconfig
  (`sans-serif` = e.g. Noto Sans), so glyph shapes and the exact text-box aspect differ slightly.

### 1.3 Stamp layout (`AndroidWatermarkApplier.drawStamp`)

Vertical padding = `padding - 6 * padding / 20`; the text bitmap is resized with
`ImageScaler.scaleImage(w - 2*hp, h - 2*vp, ResizeType.Flexible)` (`flexibleResize`, `ResizeAnchor.Default`:
long side = max of the two targets); a non-transparent background gets `RoundedCornersTransformation(12)`;
`CanvasUtils.drawBitmap(position, horizontalPadding, verticalPadding)` places it.  `padding` and the 12 px radius are
full-resolution pixels (`unit="px"`, scaled for previews).  The overlay mode is kept as a param (ImageToolbox hides
the selector for stamps but still applies `params.overlayMode`, default `SrcOver`).

### 1.4 Time stamp: which date?

ImageToolbox calls `timestamp(format)` with no date, i.e. **the current time when the image is processed**
(`SimpleDateFormat(format, Locale.getDefault()).format(Date())`); it does **not** read EXIF.  An invalid pattern
makes `timestamp()` fall back to `Date().time.toString()` (epoch milliseconds) — reproduced.

Ops only receive pixels, so `draw.timestamp_watermark` has an explicit `date` param:

* empty (default) = now at render time, like ImageToolbox (the op is therefore declared `deterministic=False`);
* EXIF `YYYY:MM:DD HH:MM:SS`, ISO 8601, or epoch milliseconds.

Agent workflow for the capture date: `metadata_read(input=photo)` -> take
`exif.exif_ifd.DateTimeOriginal` (or `exif.DateTime`) -> `draw.timestamp_watermark(date="2023:07:14 18:22:05",
format="dd/MM/yyyy HH:mm")`.  The pattern language is `java.text.SimpleDateFormat` (`G y Y M L w W D d F E u a H k
K h m s S z Z X`, quotes, `''`), with English month/day names (deviation: ImageToolbox uses the device locale);
`w`/`W` use US week rules.

### 1.5 Blend modes

`_wm_render.composite` implements Skia's `SkBlendMode` on premultiplied float RGBA: the 15 Porter-Duff/coefficient
modes exactly, and the 14 advanced modes as `Sc(1-Da) + Dc(1-Sa) + Sa*Da*B(Dc, Sc)` with LazyRoom's W3C
`blend_rgb` for `B` (Skia uses the same W3C formulas).  As in Android, modes apply to the whole canvas when tiling
(e.g. `clear` erases everything) and to the bitmap rectangle for a single watermark; a result with alpha is
returned as RGBA.

### 1.6 Hidden ("digital") watermarks

ImageToolbox's Digital watermark = AndroidWM `WatermarkBuilder.setInvisibleWMListener(isLSB)` +
`WatermarkDetector.create(bitmap).detect()`; the algorithms are in ImageToolboxLibs
`libs/androidwm/src/main/cpp/Watermark.cpp` and are ported bit-for-bit in `_itb_stego.py`:

* **Frame**: magic (32 bits, MSB first; `0x4C534257` "LSBW" or `0x4644574D` "FDWM"), 1 "is image" bit, 7 zero bits,
  24-bit byte length, payload bytes MSB first.  Text payload = `GetStringUTFChars` = Java *modified UTF-8*
  (U+0000 -> `C0 80`, supplementary characters as two 3-byte surrogates); extraction mirrors
  `NewStringUTF(result.c_str())` (stops at a NUL byte).  Image payload = Base64 (NO_WRAP) of a PNG (LSB) or of a
  JPEG chosen by `bitmapToFdString` (quality 85..35 step 10, then 3/4 downscales) for FD.  The hidden image is
  `size` x the background width (ImageToolbox "Watermark size", default 0.1).
* **LSB**: frame bit *i* -> pixel *i // 3*, channel by the `uint32` shift `{16, 8, 0}[i % 3]` of an Android
  `RGBA_8888` pixel.  On little-endian ARM those shifts are **blue, green, red** (the C++ comments say ARGB, but the
  memory order of `RGBA_8888` is R,G,B,A).  Capacity = 3 bits per pixel.
* **FD**: bit *i* in 8x8 block *i* (row-major, `width / 8` blocks per row), written in the B, G and R channels:
  orthonormal 8x8 DCT, coefficients (2,3)/(3,2) pushed to a difference of +-80, retried with 160 and 240 if
  rounding/clamping flips it; failing on more than one replica aborts (ImageToolbox: `ERROR_PIXELS_NOT_ENOUGH`).
  Reading: majority of `D[2][3] >= D[3][2]`.  Capacity = one bit per 8x8 block.  The DCT is vectorised (matrix
  products) instead of the C++ quadruple loops: identical maths, different float summation order, so an output
  pixel can differ by 1 in rare .5 rounding cases; extraction results are identical.
* **Detection** follows `AutomaticDetectionTask`: LSB text, LSB image, FD text, FD image; an empty text result means
  "no watermark" (as `checkHiddenWatermark` does).
* **Premultiplied alpha**: Android bitmaps are premultiplied and ImageToolbox embeds into premultiplied values (its
  PNG encoder un-premultiplies).  `_itb_stego` emulates this for pixels with alpha < 255 (premultiply with
  `SkMulDiv255Round`, embed, un-premultiply), so semi-transparent images interoperate; fully transparent pixels and
  channels equal to an even alpha cannot hold a 1 bit in ImageToolbox either.  The embed tool re-decodes its own
  output and fails with a clear message when bits did not survive.
* Interoperability is tested against a literal Python translation of `Watermark.cpp` working on little-endian
  `uint32` RGBA words: ours == reference for LSB (bit-identical files), reference-extract(ours) and
  ours-extract(reference) both recover the payload for LSB and FD.

**Warning (also in the tool docs): LSB watermarks are destroyed by any lossy re-encoding (JPEG, lossy WebP/AVIF/HEIC),
resizing, cropping that shifts pixels, colour-profile conversion or palette quantisation.**  `watermark_hidden_embed`
therefore refuses LSB output to a lossy format, writes lossless WebP when `.webp` is chosen, and verifies every file it
writes by decoding it again.  FD survives high-quality JPEG (q95 tested) but not strong compression or resizing.
ImageToolbox's own preview path (`originalSize = false`) may down-scale large backgrounds to 1024 px before embedding;
its save path uses the original size, which is what the tool does.

## 2. Filter templates

### 2.1 Format (`FilterSerializationUtils.kt`, `Mappings.kt`, `AndroidFilterParamsInteractor`)

```
https://github.com/T8RIN/ImageToolbox?<name>+<filter>,<filter>,...
<filter> = ^^.core.filters.presentation.model.Ui<Name>Filter:<value type>:<value>
```

`^^` replaces `com.t8rin.imagetoolbox`; templates saved by ImageToolbox contain `Ui*Filter` classes (older
`^^.feature.filters.data.model.<Name>Filter` names are accepted on import).  `<value type>` is the runtime simple
name: `Float`, `Unit`, `FloatArray`, `FilterValueWrapper{ColorModel}`, `Pair{Float$Boolean}`,
`Triple{Float$Float$BlurEdgeMode}`, `Quad{...}`, a `*Params` data class, `GmicFilterParams`, `ProceduralParams`,
`GradientPalette`, `PolarCoordinatesType`, `String`.  Components are joined with `$` (points inside
`CropOrPerspectiveParams` with `*`, tone curves `*` within a curve), floats are printed with Java `Float.toString`
(shortest float32 digits; scientific `1.0E-4` outside [1e-3, 1e7)), colours are signed ARGB ints, enums by name,
strings / G'MIC values Base64 (UTF-8), procedural values as Base64 of `key=value;...;color:key=int`.  A template is
valid only if it contains the link header, `^^` (or the real package) and `Filter` (`isValidTemplateFilter`), which
is also how the QR scanner recognises it.  The QR code is plain byte-mode content (ImageToolbox uses QRose with
automatic error correction; LazyRoom uses segno, level L boosted) — only the content must match, and the export
tool decodes its own QR with zxing-cpp to verify.  `text_output` writes the same string as an `.imtbx_template` file
(ImageToolbox's template file format, importable there).

### 2.2 Mapping ImageToolbox values to `filter.*` params

`_itb_template_table.py` is generated by `_itb_template_gen.py` from the Kotlin sources: for each of the **541**
`Ui*Filter` classes (519 named filters + 22 colour-map classes that share `filter.color_map` with a fixed choice) it
records the value type and, per value component, the kind (float, int, bool, ARGB colour, enum, string, float array,
point, gradient palette, tone curves, G'MIC number/toggle/selection/colour, procedural key, font, file) and the
target op param.  The LazyRoom ports mirror `paramsInfo`, so components map positionally; procedural and NTSC values
map by name; 6 filters needed explicit overrides (Ascii, DropShadow, SideFade, ToneCurves, three HueShift procedural
effects).  Enum names map to choices by normalised name or upstream order (BlurEdgeMode, TransferFunc, MirrorSide,
FadeSide, PopArtBlendingMode, PolarCoordinatesType, PaletteTransferSpace, G'MIC selections; NTSC int codes map to
choices by index).  Verification: building ImageToolbox's own default template string for every filter and
importing it reproduces the op defaults for all filters except three whose LazyRoom *op defaults* deviate from
ImageToolbox (ColorOverlay and Monochrome store alpha/channel 0x4C where Compose rounds to 0x4D; RawGmic's default
command differs) — the imported values are ImageToolbox's in those cases.

Coverage: 538 classes import/export; 3 cannot (`DisplacementParams`, `SeamCarvingParams`, `ShaderParams` have no
`toPair` case: ImageToolbox itself writes `<class>:null` and drops them on import — LazyRoom skips them with a
warning on import and lists them as unsupported on export).

### 2.3 Import (`template_import`)

Input: a QR code image (zxing-cpp, the first decoded text that is a valid template), an `.imtbx_template`/text file,
or a raw string.  Behaviour mirrors `toFiltersList`: unknown classes and unparsable values are skipped (ImageToolbox
drops them too), a value type different from the current one falls back to the filter defaults (ImageToolbox's
`callBy(emptyMap())`), missing NTSC fields fall back per field (`floatAt/intAt/booleanAt`), procedural values are
clamped like `ProceduralEffect.resolve`, out-of-range numbers are clamped to the LazyRoom range with a warning.
Floats are imported as the shortest decimal of the float32 value (`"0.3"` -> 0.3).  The steps are saved with
`presets.save` (name = template name unless `name` is given) and, with `project_id`, appended to the project as one
group (`project.edit` / `add_group`, origin `import`).  The result lists `warnings` and `skipped` filters.  Decoded
strings are untrusted data.

Lenient vs ImageToolbox: ImageToolbox keeps only the text between the first and second `+`; a `+` inside a Base64
value (possible only for strings containing `>`/`~`/non-ASCII) truncates the template there.  Import rejoins the
pieces (reads everything ImageToolbox reads, and more); export warns when its output would hit this ImageToolbox bug.

### 2.4 Export (`template_export`)

Input: a preset id or a step list; groups are flattened; disabled steps are skipped; step opacity/blend/mask and
group settings cannot be represented and produce warnings.  Steps whose op is not an ImageToolbox filter (or one of
the 3 unserialisable ones) make the call fail with an `INVALID_PARAMS` error listing them (`details.unsupported`),
unless `skip_unsupported=true`.  Op params that have no ImageToolbox counterpart (e.g. `filter.lookup`'s `lut`,
`filter.tone_curves` beyond RGB) are dropped with a warning when non-default.  Output strings are byte-identical to
what ImageToolbox writes for the same values (tested for 23 filters covering every value family).  Special cases:

* Boolean-valued filters (Clustered 2x2/4x4/8x8, Yililoma dithering): ImageToolbox's `toPair` has no Boolean case,
  so ImageToolbox *writes* `:null` and loses them, but its reader accepts `Boolean:true`; LazyRoom writes the
  readable form (warning).
* Filters with an image/file argument (LUT 512x512, Cube LUT, Lens correction, Palette transfer (+variant)): the file
  is serialised as an empty part exactly like ImageToolbox (`Pair{Float$ImageModel}:1.0$`), with a warning; on import
  the file param keeps LazyRoom's default, and the two palette-transfer filters, whose target image is required,
  are skipped.
* Tone curves: ImageToolbox stores 17 curves + 4 colour-wheel entries; `filter.tone_curves` implements the first
  four (RGB/luma, R, G, B).  Export writes the other curves/wheels as ImageToolbox defaults; import warns when they
  are not neutral.  Legacy 5-value curves are read as evenly spaced Y values.
* Gradient map: named palettes are written by name (76 palettes extracted), others as `custom;aarrggbb;...`;
  unevenly spaced stops are resampled to 33 evenly spaced colours (warning) because ImageToolbox palettes are evenly
  spaced.
* ASCII font: ImageToolbox stores a font ordinal (0 = System; custom fonts are written as System); LazyRoom maps
  ordinals to the 26 bundled family names and `sans-serif` to System.
* Template names containing `+` or `\` (separators) are sanitised to `_` (warning).

## 3. Formats

Sources: ImageToolbox `core/data/.../image/utils/compressor/*Backend.kt` (format encoders), `feature/format-conversion`, `feature/jxl-tools`, `feature/weight-resize`, the export profiles; aire (MIT) for PNG Lossy.  Code: the JXL backend in `core/io.py` (additive), new `format_convert` choices and params in `tools/convert.py` (existing params/defaults/output unchanged, tested byte-identical), encoders in `tools/_itb_codecs.py`, tools `jxl_transcode`, `jxl_from_frames`, `jxl_extract_frames`, `telegram_sticker` in `tools/_itb_formats.py`.

`core/io.py` additions (all guarded by the optional import): `HAVE_JXL`/`JXL_HINT`, the `.jxl` entry of `WRITE_FORMATS` (hence `READ_EXTS`, `supported_formats()` + `jxl_available`), a `BACKEND_UNAVAILABLE` check in `format_for_path`, JXL hooks in `probe`/`load_image` (extension or signature), `_load_jxl`, the `encode_image` JXL branch (`_encode_jxl`) and a ~70-line header parser (`jxl_header`, `jxl_boxes`).  The parser is the one non-trivial part: imagecodecs has no header/info call, so it is the only way to enforce `max_pixels` before libjxl allocates the image (decompression-bomb guard, like Pillow's for other formats) and to detect animation.  `load_image` decodes only the requested frame; the frame count of an animated JXL is reported as unknown (`frames: null`) because counting would decode every frame (unbounded memory). `jxl_extract_frames` checks the per-frame pixel limit before decoding; it then decodes all frames at once (explicit request).

### 3.1 Feature table

| ImageToolbox feature (source) | LazyRoom | Status | Notes |
|---|---|---|---|
| JXL Lossless / JXL Lossy format (`JxlBackend`, awxkee jxl-coder / libjxl) | `.jxl` read/write in `core/io.py` (`load_image`, `probe`, `encode_image`, `save_image`, `export`, batch); `format_convert(format="jxl", lossless, effort, quality)` | faithful (libjxl) | quality 1..99 -> libjxl `JxlEncoderDistanceFromQuality` (same mapping as jxl-coder), quality 100 or `lossless` -> mathematically lossless; effort 1..10; 8- and 16-bit; RGB/RGBA. Default effort is libjxl's 7 (ImageToolbox default 2: pass `effort=2`). EXIF/XMP written in an ISOBMFF container (`Exif`/`xml ` boxes) per the metadata policy, bare codestream otherwise. Optional backend: without `imagecodecs` `.jxl` is absent from all format lists and reads/writes raise `BACKEND_UNAVAILABLE` with an install hint. |
| Jixel Lossless / Jixel Lossy (`JixelCoder`, effort 1..2) | — | not implemented | Second JXL encoder in ImageToolbox; same file format, covered by the libjxl path. |
| JXL `channels` (RGBA/RGB/Monochrome) and `speed` (decoding speed 0..4) | automatic RGB/RGBA; `encode_image(extra={"decoding_speed": n})` | approximate | Channel layout follows the image's alpha; monochrome output and decoding speed are not exposed as tool params. |
| JXL Tools: JPEG -> JXL / JXL -> JPEG (`JxlCoder.Convenience.construct` / `reconstructJPEG`) | tool `jxl_transcode` (`mode=jpeg_to_jxl|jxl_to_jpeg`, `verify`) | faithful | libjxl lossless JPEG recompression (`jpegxl_encode_jpeg`); `verify=true` rebuilds the JPEG and refuses to write unless it is byte-identical. JXLs without reconstruction data fail with `DECODE_FAILED` + hint. Tested: bit-exact round trip, ~20% smaller. |
| JXL Tools: Images -> JXL (`createJxlAnimation`, `AnimatedJxlParams` delay 1000 ms, quality 50, effort 2, lossy) | tool `jxl_from_frames` (still images) | approximate | Same defaults and frame fitting (ResizeType.Flexible to the first image's longer side, then CenterCrop on a transparent canvas). Loops forever (ImageToolbox default `repeatCount=1`): the backend cannot write `num_loops`. Frame timing: imagecodecs writes 10 ticks/s and 1 tick per frame; LazyRoom rewrites the AnimationHeader to a tick of 1..1024 ms in place (same bit length, verified by libjxl decoding and by ffprobe durations); longer/irregular durations are realised by repeating frames (reported as `stored_frames`). |
| JXL Tools: GIF / APNG / WebP -> JXL (`gif2JXL`, `apng2JXL`, WebP converter) | tool `jxl_from_frames` with one animated input | approximate | Keeps source durations exactly (common tick + repeated frames); refuses plans that would need > max(1000, 4x frames) stored frames (use `override_durations`). |
| JXL Tools: JXL -> Images (`extractFramesFromJxl`) | tool `jxl_extract_frames` (png/jpg/webp) | faithful (frames) | Per-frame durations are not exposed by imagecodecs; the header tick and loop count are reported. Frame subset selection (`ImageFrames`) not implemented (all frames). |
| JXL Tools: JXL -> GIF / APNG / WebP, merge JXLs | — | not implemented | Use `jxl_extract_frames` then `animation_from_images` (uniform frame time). Per-frame JXL durations would need a FrameHeader parser. |
| PNG Lossy (`PngLossyBackend` -> aire `toPNG`: XIAOLING_WU quantiser, JARVIS_JUDICE_NINKE dithering, COVER_TREE mapper, maxColors 2..1024 default 512, compression level 0..9 default 7) | `format_convert(format="png_lossy", max_colors, effort=zlib level)`; helpers `png_lossy_pixels/encode` in `tools/_itb_codecs.py` | faithful algorithm, approximate bits | Port of aire's 4-D (ARGB) Wu quantiser: 33^4 cells (SIDEPIXSHIFT 3), alpha threshold 15 and alpha fader 15 (`a + a % 15`), the `sqr(n)/cells` integer-division rule for sqrt weighting with `long *= double` truncation, Wu cut/variance splitting, transparent entry first, truncating averages. Mapping uses aire's weighted metric `2dR²+4dG²+(R<128?3:2)dB²+dA²` with an exact nearest search (aire's cover tree on squared distances is approximate). **Dithering: none, like ImageToolbox's actual output** — aire's `RemapPalette::remap` adds the JJN error into output pixels that are overwritten later, so the requested Jarvis-Judice-Ninke diffusion has no effect. Differences: 64-bit/float64 arithmetic (aire's `int sqr()` and uint32 `xarea` sums overflow on large images, so palettes can differ there); the global `m_transparentPixelIndex` is reset per image; palette written as an indexed PNG when <= 256 colours (identical pixels, smaller file; ImageToolbox writes RGBA truecolour); fully opaque images are kept opaque (aire's alpha truncation can produce alpha-254 entries). 12 MP in ~7 s. |
| OxiPNG (`OxiPngBackend`, level 0..6) | `format_convert(format="oxipng", effort=level)` | faithful (optional) | `pyoxipng` (MIT) `optimize_from_memory(level)` = oxipng preset; default 6 (ImageToolbox coerces its quality to 0..6, i.e. 6 by default). Lossless; metadata chunks kept per policy. `BACKEND_UNAVAILABLE` + hint when absent. |
| Jpegli (`JpegliBackend`, quality, progressive=true) | `format_convert(format="jpegli", quality)` | approximate (optional) | `pyjpegli` (BSD-3, bundles google/jpegli). pyjpegli exposes no progressive switch, so output is baseline (ImageToolbox: progressive). EXIF (per policy) and the sRGB ICC are inserted as APP1/APP2. Alpha is flattened on white like other non-alpha formats. |
| MozJPEG (`MozJpegBackend`, Aire.mozjpeg) | — | not implemented | No permissive Python MozJPEG *encoder* binding: imagecodecs' wheel is built without MozJPEG (`MOZJPEG.available == False`); `mozjpeg-lossless-optimization` (BSD-3) only re-optimises existing JPEGs losslessly, which is not the MozJPEG encoder. |
| ImageQuant (`PngImageQuantBackend`, libimagequant) | — | not implemented | libimagequant is GPL-3.0-or-later (or commercial); `imagequant`, `libimagequant` and `pngquant` Python packages bundle/wrap it. Not permissive. |
| ICO (`IcoBackend`: 16/32/48/64/128/256 in one file) | `format_convert(format="ico_multi")`; helper `ico_encode` | faithful | Byte layout identical to ImageToolbox: ICONDIR + 16-byte entries (256 stored as 0, planes 1, 32 bpp), 40-byte BITMAPINFOHEADER (height x2, biSizeImage = XOR+AND), bottom-up BGRA, zero AND mask; each size = ResizeType.Fit on a transparent square (Kotlin Float->Int truncation, centred). Resampling Lanczos (ImageToolbox uses the app's default scale mode). Dimensions are clamped to >= 1 px (ImageToolbox would fail on extreme aspect ratios). The existing `format="ico"` (Pillow, sizes <= image size) is unchanged. |
| Telegram sticker (`Preset.Telegram` / `telegram_sticker` export profile: 512x512, ResizeType.Flexible, PNG lossless, Lanczos3) | tool `telegram_sticker` | faithful | Flexible + ResizeAnchor.Default: longer side becomes 512 (also upscales), other side `(512 * aspect).toInt()` in float32. Output PNG lossless, metadata stripped. Adds `within_512kb` / warning for Telegram's 512 KB static-sticker limit (ImageToolbox does not check). |
| Weight resize (`AndroidWeightImageScaler.scaleByMaxBytes`) | existing `compress_to_size` (not modified: outside the allowed edits) | approximate (pre-existing) | ImageToolbox: target = maxBytes − 2048·max(1, size/5 MB); quality 100->15 in steps of 1, then size x0.93 steps at quality 15; any format. LazyRoom: binary search on quality (jpg/webp/avif), then x0.8 downscale. |

### 3.2 Formats and compatibility notes

* **JPEG XL interop**: files written by LazyRoom decode in ImageMagick (libjxl 0.11) and FFmpeg; JXLs written by ImageMagick (lossless, lossy, animated) load in LazyRoom (lossless pixels identical). Animated JXL durations written by `jxl_from_frames` were checked with `ffprobe` (0.25 s per frame for `frame_ms=250`).
* `jxl_header()` (core/io.py) parses the SizeHeader and the start of ImageMetadata (bare codestream or `jxlc`/`jxlp` container boxes) to enforce the max-pixels limit before decoding and to detect animation; JXL files are also detected by signature when the extension is missing.
* Decoded JXL pixels are taken as sRGB; JXLs with a non-sRGB embedded profile are not colour-converted (imagecodecs does not return the ICC profile).
* The tick rewrite only touches the 17-bit AnimationHeader written by imagecodecs (`tps 10/1, loops 0, no timecodes`); it verifies the original bits and re-parses the result, otherwise raises `INTERNAL`.
* Animated JXL output always loops forever.

### 3.3 Packages evaluated (PyPI metadata / licences checked 2026-10-14)

| Package | Version | Licence | Verdict |
|---|---|---|---|
| imagecodecs | 2026.10.10 | BSD-3-Clause (bundles libjxl 0.12 BSD-3, highway Apache-2.0/BSD-3, brotli MIT, lcms2 MIT, libjpeg-turbo IJG/BSD, ...; see `imagecodecs/licenses/`) | **chosen** for JPEG XL: read/write, 16-bit, lossless JPEG transcoding (`jpegxl_encode_jpeg/decode_jpeg`), multi-frame. Large wheel (~86 MB installed). No MozJPEG in the wheel. |
| pillow-jxl-plugin | 1.3.8 | GPL-3.0 (repository LICENSE; no licence in PyPI metadata) | rejected (copyleft) |
| pylibjxl | 0.5.0 | BSD-3-Clause | not chosen: uint8 only, no animation, very new; transcoding available but imagecodecs covers it |
| pyoxipng | 9.1.1 | MIT (oxipng MIT) | **chosen** (optional) for OxiPNG |
| pyjpegli | 0.3.2 | BSD-3-Clause (bundles google/jpegli BSD-3, highway, libjpeg-turbo, skcms) | **chosen** (optional) for Jpegli; RGB only, no progressive control |
| mozjpeg-lossless-optimization | 1.3.2 | BSD-3-Clause | not an encoder (lossless re-optimisation only) -> MozJPEG not implemented |
| imagequant | 1.1.5 | binding BSD-3 but bundles libimagequant GPL-3.0+ | rejected |
| libimagequant | 2.17.0.0 | GPL-3.0+ | rejected |
| pngquant | 3.0.3 | wrapper BSD, wraps GPL pngquant binary | rejected |
| simplejpeg | 1.9.0 | MIT | libjpeg-turbo only (no MozJPEG/jpegli) -> not useful |

**Installed into `.venv` for evaluation/tests** (with `uv pip install --python .venv/bin/python ...`): `imagecodecs==2026.10.10`, `pyoxipng==9.1.1`, `pyjpegli==0.3.2`. Nothing else was installed; no runtime network use.

### 3.4 Format deviations

1. PNG Lossy: no dithering (matches ImageToolbox's effective output, not its intent); exact nearest-colour search instead of aire's cover tree; float64/int64 arithmetic instead of aire's overflowing 32-bit `sqr`/`xarea`; indexed PNG when <= 256 colours; opaque inputs stay opaque.
2. JXL: default effort 7 (libjxl) instead of ImageToolbox's 2 in `format_convert`/export (`jxl_from_frames` uses 2); no Jixel encoder; no monochrome/decoding-speed params in tools; animations always loop forever; non-1-tick-divisible durations stored as repeated frames; per-frame durations of foreign animated JXLs are not reported.
3. Jpegli: baseline instead of progressive (binding limitation).
4. ICO / Telegram: Lanczos resampling; dimensions clamped to >= 1 px.
5. MozJPEG and ImageQuant: not implemented (no permissive package).

## Deviations (summary)

* Visible watermarks: system fonts instead of Roboto/bundled fonts; text drawn at final scale (no upscaling blur);
  multi-line text not clipped; date names in English; `{filename}` placeholder only in the hidden-watermark tool.
* Hidden watermarks: FD output may differ by 1 in rare pixels (float summation order); payload PNG/JPEG bytes come
  from Pillow instead of Android's encoders (any decoder reads them).
* Templates: lenient `+` handling on import; Boolean filters exported in the readable form; QR symbol layout differs
  (same content).
* Formats: see 3.4.

## Tests

`tests/test_itb_extras.py` (run with `LAZYROOM_ITB_ONLY= LAZYROOM_HOME=/tmp/pm_extras_home .venv/bin/python -m
pytest tests/test_itb_extras.py -q -x`):

* watermark ops: ImageToolbox defaults, grid tiling period, alpha^2 vs background alpha, single placement, Porter-Duff
  semantics (clear/dst on canvas vs rect), rotation bounding box, image watermark scale, stamp placement/padding for
  3 positions, rounded opaque stamp box, `SimpleDateFormat` patterns and fallback, date parsing, deterministic
  explicit-date rendering;
* hidden watermarks: modified UTF-8, LSB bit-identical to a literal `Watermark.cpp` translation (incl. B,G,R order),
  LSB/FD interop both ways, capacity errors, frame header, premultiplied alpha emulation, FD vs LSB under JPEG,
  FD image payload sizing, tool round trips (text, image, `{filename}`), lossy-output refusal and FD+JPEG verification;
* templates: Java `Float.toString` (15 cases), validity rule, 26 ImageToolbox-built strings imported with
  non-default values (colour, enum, float array, pair/triple/quad, every `*Params` family, G'MIC, procedural with
  colour, palettes, booleans, colour maps), byte-identical export of 23 of them, procedural/G'MIC serialisation,
  whole-template import with legacy class names, `:null` and unknown filters, NTSC fallbacks, tone curves (incl.
  legacy and ignored curves), gradient palettes, ASCII font, default round trip of all 538 filters, randomised
  round trip (3 random parameter sets x 533 filters), export errors listing non-ImageToolbox ops, group flattening,
  QR export -> zxing decode -> import -> preset -> re-export identical, template file import, apply to a project;
* formats (31): JXL format lists, lossless 8/16-bit round trips, lossy quality/effort, EXIF/XMP container + strip, header parser sizes and max-pixel limit, signature detection, `BACKEND_UNAVAILABLE` (monkeypatched and in a subprocess with imagecodecs blocked: `.jxl` absent from read/write lists), ImageMagick-written JXL interop (skipped without `magick`), `jxl_transcode` bit-exact round trip and errors, tick rewrite + plan, `jxl_from_frames` (stills, animated GIF durations, override, single frame; durations checked with ffprobe when available), `jxl_extract_frames`, Wu palette exactness and alpha fader, PNG lossy colour budgets / palette vs truecolour / validation, `format_convert` png_lossy, ICO byte layout and Pillow/ImageMagick readability, unchanged plain `ico`, Telegram sizes and tool, `format_convert` jxl/oxipng/jpegli, unchanged default `format_convert` output, optional encoders unavailable.

Result: **133 passed** (102 watermark/template + 31 formats) in ~8 s with imagecodecs, pyoxipng and pyjpegli installed.  `tests/test_core.py` + `tests/integration/test_cli.py`: **26 passed**.  The four new ops were also run like `tests/test_ops_smoke.py` (defaults, RGB/RGBA/2x3 px, determinism, preview scale, `describe()` JSON) — the smoke test itself only sees them once `ops/__init__.py` imports `watermark_ext`.

## Wiring for the lead

* `tools/registry.py`: add `"itb_extras"` to `TOOL_MODULES` (it imports `_itb_formats` itself).
* `ops/__init__.py`: import `watermark_ext` (registers the four `draw.*` ops).
* `pyproject.toml` optional extras (the install hints already use these names):
  `jxl = ["imagecodecs>=2026.8.16"]` (tested with 2026.10.10; that release line fixed lossless JXL with libjxl 0.12),
  `oxipng = ["pyoxipng>=9.1"]`, `jpegli = ["pyjpegli>=0.3.2"]`, optionally `formats = [all three]` and adding them
  to `all`.  No new mandatory dependency (segno, zxing-cpp, numba, Pillow, OpenCV are already required).
* NOTICE / docs/THIRD_PARTY.md:
  * ImageToolbox (Apache-2.0, T8RIN): watermarking, filter-template format, format backends, Telegram/ICO presets.
  * ImageToolboxLibs `libs/androidwm` (Apache-2.0; AndroidWM by huangyz0918, Copyright 2018): hidden-watermark
    algorithms (LSB / FD frames) and the watermark layout.
  * aire (MIT, Radzivon Bartoshyk / awxkee): PNG Lossy Wu quantiser port.
  * imagecodecs (BSD-3, Christoph Gohlke) bundling libjxl (BSD-3, JPEG XL Project Authors), highway
    (Apache-2.0 / BSD-3), brotli (MIT), lcms2 (MIT), libjpeg-turbo (IJG/BSD) and others — ship
    `imagecodecs/licenses/*` in frozen builds.
  * pyoxipng (MIT, Nick Frasser) wrapping oxipng (MIT).
  * pyjpegli (BSD-3, Maksim Burtsev) bundling google/jpegli (BSD-3), highway, libjpeg-turbo, skcms (BSD-3).
* Packages installed into `.venv` during this work (with `uv pip install --python .venv/bin/python`):
  `imagecodecs==2026.10.10`, `pyoxipng==9.1.1`, `pyjpegli==0.3.2`.  Nothing else.
* `tools/_itb_template_gen.py` is a dev script living in the package only because of file ownership; it can move to
  `scripts/` (it imports nothing at runtime and is never imported by the app).
