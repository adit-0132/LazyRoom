# Port report: ImageToolbox utility features (cutting, photomosaic, PDF tools, draw-to-erase, extras)

Modules: `src/lazyroom/tools/itb_tools.py` (tools), `src/lazyroom/ops/erase.py` (op `draw.erase_to_alpha`).
Tests: `tests/test_itb_tools.py` (41 tests; run with `LAZYROOM_ITB_ONLY=none .venv/bin/python -m pytest tests/test_itb_tools.py -q`).

Sources: `/tmp/ref/ImageToolbox/feature/{image-cutting,photomosaic,pdf-tools,erase-background,limits-resize,screenshot-framing}`,
`lib/opencv-tools/.../ImageDiffTool.kt` (PDF compare), English strings from `core/resources/.../values/strings.xml`.

**Wiring still needed (shared files were not touched):** add `"itb_tools"` to `TOOL_MODULES` in
`src/lazyroom/tools/registry.py` and `erase` to the imports in `src/lazyroom/ops/__init__.py`.

**Dependencies:**
* `pypdfium2` (5.14.0, Apache-2.0 / BSD-3; bundles PDFium, BSD-3; cp313 wheels) was installed into `.venv`. Add it to
  `[project] dependencies` in `pyproject.toml`. It is imported lazily: without it, `pdf_to_images`, `pdf_flatten`,
  `pdf_print` and `pdf_compare` raise `BACKEND_UNAVAILABLE` with an install hint, and every other tool still works.
* `pypdf` is already a dependency. Recommend raising the floor to `pypdf>=6`: the code uses
  `compress_identical_objects` (≥ 4.3, both argument spellings handled), `pypdf._codecs.core_font_metrics` (Helvetica
  AFM widths, with a fallback) and `pypdf.generic._image_xobject._xobj_to_image` (image extraction).
* `cryptography` (already present transitively) is needed for AES in `pdf_protect` and for opening AES-encrypted
  PDFs. Consider declaring it explicitly.
* PyMuPDF was **not** used (AGPL).
* `pillow-jxl-plugin` was tried and **uninstalled**: it is GPL-3.0-or-later (see "Skipped").

Summary (table rows): **21 faithful, 16 approximate, 2 not implemented**. That is 36 tools plus 1 op; Photomosaic has two rows (adaptive/grid), and crop + crop presets share one row.

## Image Cutting, Photomosaic

| Feature | tool / op | status | notes |
|---|---|---|---|
| Image Cutting (`AndroidImageCutter.cutAndMerge`, `CutParams`, `PivotPair`) | `image_cutting` | faithful | Exact semantics: `toCutBounds` (round-half-up, `(0,1)` and invalid/empty ranges mean "no cut"); vertical cut first, then horizontal; inverse keeps the band; both inverse is a crop; any internal failure (e.g. both parts empty) returns the image unchanged, like `runSuspendCatching { … } ?: image`. `isRtl` mirrors the UI layout direction only and is not needed. Batch mode (`output_dir` + `{stem}{ext}{index}` template) replaces the app's save dialog. |
| Photomosaic, adaptive layout (`AndroidPhotomosaicMaker` + `AdaptiveTileLayout`) | `photomosaic` (`layout=adaptive`) | faithful | Algorithm reproduced exactly: `PhotomosaicParams` ranges/defaults (columns 10–100/40, blend 0–0.6/0.12, repeat 0–10/2, max tiles 10–500/300); columns ≤ width; tile size `clamp(W/cols, 32, 192)` capped by the 24 MiB budget (preview: 48 px tiles, target ≤ 1024); Flexible long-side tile scaling; tile Lab = integer mean of pixels with alpha > 16; `rgbToLab` (D65); colour sampled with an integer-mean window of radius `min(W,H)/256`; `detailAt` (24 samples, offset `min/128`, gain 4); row heights `base·(1.5 − 0.85·detail)`; `selectTile` with repeat avoidance and a "least-recently-used" fallback; rows justified to the full width. Kotlin `Float` arithmetic and `roundToInt` are emulated in float32. Target colour blend uses `alpha = round(blend·255)`. The four upstream `AdaptiveTileLayoutTest` cases are ported. Pixel-level differences come only from resampling: OpenCV area/bilinear instead of Coil + Aire `ImageScaler` + Skia `drawBitmap(RectF)` (tile rects snapped to pixel centres). |
| Photomosaic, grid (`TileMatcher`) | `photomosaic` (`layout=grid`) | approximate | `TileMatcher.match` is exact: Manhattan repeat avoidance, falling back to the farthest recent use; all upstream `TileMatcherTest` cases are ported. Upstream never wires TileMatcher into the maker, so the grid geometry is LazyRoom's: square cells `W/cols`, cell Lab = integer mean over the cell, tiles centre-cropped to square. |

## PDF tools (`AndroidPdfManager` / `AndroidPdfHelper` / `PdfUtils`)

Common points:
* **Password handling.** Every tool takes `password` (upstream `masterPassword`). Outputs are written without
  security, like `isAllSecurityToBeRemoved = true`.
* **Page lists.** Pages are 1-based text such as `'1-3,5'`. Out-of-range pages are rejected, whereas upstream
  `getPageSafe` clamps them.
* **Output paths** go through the workspace path policy.
* **Orphaned objects** (replaced images, removed pages) are pruned before writing.
* **Text drawing** (page numbers, watermark, captions, OCR layer) uses the standard Helvetica font with WinAnsi
  encoding instead of the app's embedded multi-script Noto font. Metrics (width, ascent 718, descent −207) come from
  the Adobe AFM. Characters outside WinAnsi are dropped by the same mechanism as `cleanPdfText`.
* **Rasterisation** uses PDFium (pypdfium2), the same engine as Android's `PdfRenderer`. It renders on white,
  draws annotations, and applies the 4096 px / 12 MP clamps of `safeRenderDpi`.

| Feature | tool / op | status | notes |
|---|---|---|---|
| Preview PDF / `checkPdf` / `getPdfMetadata` | `pdf_info` | approximate | No viewer (GUI concern). Reports `open` / `needs_password` / `unlocked` / `failure` like `PdfCheckResult`, plus page count, sizes, rotations, version and the six metadata fields. |
| Images to PDF (`createPdf`, `prepareImagesForPdf`, `createPdfFromPreparedImages`) | `pdf_from_images` | faithful | Percentage preset with Flexible resize; `ImageScaleDirection` none/up/down (bicubic to the widest/narrowest width); page size = image pixels in points; JPEG at `quality/100`, or lossless Flate at 100; alpha goes to an `/SMask` (Lossless/JPEGFactory). OpenCV resampling instead of Aire. The existing Pillow-based `images_to_pdf` tool is separate. |
| PDF to Images (`extractPages`) | `pdf_to_images` | faithful | `dpi = 72·scale%`, same render clamps, white background, page selection; output naming via template. |
| Merge PDF (`mergePdfs`) | `pdf_merge` | faithful | Concatenates the pages in order (pypdf `append` instead of `PDFMergerUtility`). Folders and globs are expanded to `*.pdf`. |
| Split PDF (`splitPdf`) | `pdf_split` | faithful | Imports the selected pages in order into a new document with the same version. |
| Remove PDF pages (`removePdfPages`) | `pdf_remove_pages` | faithful | Distinct indices, descending; "Can't remove all pages". |
| Rotate PDF (`rotatePdf`, `detectPdfAutoRotations`) | `pdf_rotate` | faithful | `/Rotate = (rotate + angle) % 360` per page; `auto` = rotate to the majority orientation (upstream helper used by the UI). `angle` + `pages` is a convenience on top of the per-page `rotations` list. |
| Rearrange PDF (`rearrangePdf`) | `pdf_rearrange` | faithful | New order may repeat or omit pages; descending ranges (`'5-1'`) are allowed. |
| Crop PDF (`cropPdf`, `PDRectangle.crop`) + crop presets (`PdfCropPresetRepository`) | `pdf_crop`, `pdf_crop_presets` | faithful | Rotation-aware crop-box maths for 0/90/180/270. Presets are named rects with a case-insensitive upsert; blank names and empty rects are rejected. Stored as JSON in `<data>/itb/pdf_crop_presets.json` instead of DataStore; listed newest first. |
| Page Numbers (`addPageNumbers`) | `pdf_page_numbers` | approximate | Same layout maths: font size makes the label `size %` of the page width; 9 positions with 10/20 pt insets; ascent/descent/centre baseline offsets; crop-box origin; `{n}`/`{total}`; colour with alpha via `ExtGState ca`. Deviation: Helvetica metrics instead of the Noto font, so text width and baseline differ slightly. |
| Watermark PDF (`addWatermark`) | `pdf_watermark` | approximate | Same maths: text width = `size %` of the page width, `Matrix.getRotateInstance(-angle, centre)` then `Td(-w/2, -vCenter)`, alpha = opacity. UI defaults: 315°, 0.3, 50 %, gray. Same font deviation as page numbers. |
| Signature (`addSignature`) | `pdf_signature` | faithful | Size/aspect clamping, centre `(x·W, y·H)` with **y measured from the bottom** (as upstream and its preview), clamped onto the page, lossless image with `/SMask`, alpha via `ca`. The signature drawing pad and the saved-signatures library are UI features and are not ported (pass an image file). |
| Protect PDF (`protectPdf`) | `pdf_protect` | faithful | user password = owner password, all permissions, 128-bit key. pypdf writes AES-128. PDFBox's 128-bit policy may choose RC4-128 or AES-128 depending on `preferAES`, so the cipher can differ but the protection is equivalent. |
| Unlock PDF (`unlockPdf`) | `pdf_unlock` | faithful | Open with the password and save without security. |
| Repair PDF (`repairPdf`) | `pdf_repair` | approximate | Same idea (lenient load plus clean rewrite), but pypdf's `strict=False` recovery is not PDFBox's parser, so the set of repairable files differs. |
| Metadata (`changePdfMetadata`, `setMetadata`) | `pdf_metadata` | faithful | Null fields are kept and given fields replace (`value ?: old`); `reset` = a fresh Info with creation/modification dates (`setMetadata(null)`). With no output it only reads. An empty string means "keep" (fields cannot be set to an empty string). |
| Sanitize PDF (`sanitizePdf`) | `pdf_sanitize` | faithful | Same removals: Info + catalog/page `/Metadata`; `/OpenAction`, `/AA`, names `/JavaScript`; `/AcroForm`; names `/EmbeddedFiles`; annotations (all, or only widgets / file attachments). Reports the keys removed. |
| Remove annotations (`removeAnnotations`, `PdfAnnotationType`) | `pdf_remove_annotations` | faithful | PDFBox class mapping reproduced by `/Subtype`: `markup` matches every markup subtype (Text, FileAttachment, Line, Stamp, Square/Circle, text markups, FreeText, Polygon, PolyLine, Caret, Ink, Sound); `unknown` matches anything PDFBox maps to `PDAnnotationUnknown`; all types = clear. Default `link`. |
| Compress PDF (`compressPdf`, `transformImages`) | `pdf_compress` | approximate | Re-encodes every top-level page image XObject as JPEG at `quality` (lossless at 1; alpha to `/SMask`). Deviations: decoding by pypdf instead of `PDImageXObject.image`; an image shared by several pages is converted once (upstream creates one copy per page); Pillow/libjpeg instead of `Bitmap.compress`. |
| Grayscale (`convertToGrayscale`) | `pdf_grayscale` | approximate | Same transform as aire `saturation(0, tonemap=false)` (OKLab chroma = 0) at JPEG 0.8, applied to images only (text and vectors untouched, like upstream). Same decode/encode deviations as compress. |
| PDF Dark Mode (`convertToDarkMode`, `PdfDarkModeTheme`) | `pdf_dark_mode` | faithful | White underlay (PREPEND), then white fills with `Saturation` (skipped for Negative), `Difference`, then the theme colour with `Screen` (or the custom colour and blend mode); version raised to ≥ 1.4. Vector content stays vector. |
| PDF to Text (`stripText`) | `pdf_extract_text` | approximate | Same output format (`--- Page N ---` blocks, `.txt`), but pypdf `extract_text` instead of `PDFTextStripper`, so ordering and spacing can differ. The app's "navigate to OCR" flow is `pdf_to_images` + `ocr` / `pdf_searchable`. |
| Flatten PDF (`flattenPdf`) | `pdf_flatten` | approximate | `dpi = 72 + 228·quality`, page raster stored as JPEG(quality) or lossless. Deliberate fix: pages use the displayed (rotated) size at origin 0. Upstream draws into `PDPage(cropBox)` at (0,0), which shifts pages with a non-zero crop origin and stretches rotated pages. |
| Extract Images (`extractImagesFromPdf`, `getAllImages`) | `pdf_extract_images` | approximate | Images of every page, recursing into forms, de-duplicated by object and by MD5 of the PNG; names `extracted_N.ext`; writes nothing when there are no images. JPEG/JPEG2000 streams are stored as their original bytes (`.jpg`/`.jp2`); everything else (including CCITT) as PNG, while upstream labels CCITT `tiff`. |
| Zip PDF (`convertToZip`) | `pdf_zip` | faithful | Chunks of `interval` pages; entries `<stem>_<k>.pdf`. |
| Print PDF (`printPdf`, `PrintPdfParams`) | `pdf_print` | approximate | N-up mapping {1,2,4,6,8,9,12,16}, horizontal swaps rows/cols, margin %, scale ≤ 1, centring, render dpi `(72+228q)·scale ≥ 36`. Upstream quirk kept: with `orientation=original` the page size is ignored (first page's box is used). Deviation: source/sheet sizes are rotation-aware (displayed size) instead of the raw crop box. |
| Compare PDF (`comparePdfs`, `createPdfDifferenceBitmap`, `ImageDiffTool`) | `pdf_compare` | faithful | Renders at 144 dpi, scales to ≤ 1600 px, uses white for missing pages. Full `ImageDiffTool.highlightDifferences` port: AE/MAE/RMSE/PSNR/SSIM/NCC, threshold 4, Gaussian window ≤ 11 with σ 1.5, `resizeAndPad`, highlight colour. Output pages are JPEG 0.9. Per-page scores are also returned. |
| Resize PDF Pages (`resizePdfPages`) | `pdf_resize_pages` | faithful | Fit/fill/stretch content transform (prepended `cm`), annotation rects moved, media/crop box set; all 81 `PageSize` entries (snake-case ids). |
| PDF Contact Sheet (`createContactSheet` + caption builder) | `pdf_contact_sheet` | approximate | Identical A4 layout maths: margin ≤ ⅓, cells, a caption block of `4 + lines·11` pt, 300-dpi decode clamp 64–4096, centred images, 9 pt captions, `fitPdfWidth` ellipsis. Identical caption rules: zero-padded sequence number, ` — ` / ` • ` joins, `W × H`, `humanFileSize`, camera make/model merge, exposure formatting, date, parent folder, custom lines. Deviations: Helvetica font; EXIF read with Pillow. |
| Searchable PDF (Recognize Text → `createSearchablePdf`, `parseHocrData`) | `pdf_searchable` | approximate | Text-layer maths identical: hOCR page/word boxes scaled to the page; font size `0.72·(0.84·h) + 0.28·(w/em)`; `Tz` clamped to 80–125; invisible rendering mode (3); baseline offset; 300-line plain-text fallback. Deviations: (1) the hOCR page regex is made quote-aware, because upstream's regex never matches Tesseract's `title='image "…"; bbox …'`, so its scaling only worked at 100 %; (2) OCR comes from the Tesseract CLI (`hocr`) or sidecar `.hocr`/`.txt` files, not Tesseract4Android/PaddleOCR; (3) the Helvetica/WinAnsi text layer cannot hold non-Latin scripts. |
| Comic book to PDF (`convertComicBookToPdf`) | `pdf_from_comic` | approximate | Same page logic: natural sort, skip cover, reverse, split spreads wider than 1.2× their height in reading order, `/ViewerPreferences /Direction`, original JPEG embedded at quality 100 without splitting, auto page size ≤ 14 400 pt or a fixed size with centred fit. Deviation: only ZIP/CBZ and TAR/CBT (stdlib). RAR/CBR and 7z/CB7 need libarchive/unrar and are rejected with a hint. Encrypted ZIPs: ZipCrypto only. |
| Preview PDF (viewer) | – | not implemented | Interactive viewer; belongs to the GUI. `pdf_info` and `pdf_to_images` cover the non-visual parts. |
| Palette PDF (`feature/palette-pdf`, listed in the PDF tools screen) | – | not implemented | Needs androidx Palette (MMCQ), the colour-name table and palette-file decoders (ACO/ASE/GPL…). Out of scope here. |

Unused upstream helpers (`PdfContentStreamEditor.removeText`) were not ported.

## Erase background by drawing

| Feature | tool / op | status | notes |
|---|---|---|---|
| Erase Background, manual eraser/recover (`BitmapEraser`, `EraseBackgroundComponent`) | `draw.erase_to_alpha` (`alpha="full"`) | approximate | Compositing is faithful: erase is `BlendMode.Clear` with coverage, `α' = α(1−c)`; restore is SrcOver of the opaque original, `α' = α + c(1−α)`; colours untouched. Deviations: (1) the painted area is a LazyRoom mask spec (brush strokes with radius/hardness/opacity, lasso polygon, rect, ellipse…), rasterised with OpenCV and Gaussian softness rather than Skia paths with `BlurMaskFilter`; (2) a non-destructive step cannot see the original source, so restore re-opaques the current RGB (correct after alpha-only steps such as `ai.remove_background`) and assumes an opaque original; (3) the flood-fill draw mode has no mask-spec equivalent. Auto-erase = `ai.remove_background`; "trim image" = `geometry.trim` with `tolerance=0`. |

## Other small features found while surveying `feature/*`

| Feature | tool / op | status | notes |
|---|---|---|---|
| Resize by Limits (`AndroidLimitsImageScaler`, `LimitsResizeType`) | `limits_resize` | faithful | Exact `resizeWithLimits`: 0 = unlimited; Auto Rotate swaps the box for portrait images; over-limit images are always shrunk to fit (float32 aspect, `toInt` truncation); in-limit images: skip (no output, reported), recode (unchanged), zoom (enlarge). Resampling uses OpenCV area/cubic instead of the app's `ImageScaleMode`. Batch via `output_dir`. |
| Screenshot framing (`renderScreenshotFramingBitmap`, `GradientBackgroundPreset`) | `screenshot_framing` | approximate | Same geometry: unit = min side/100, blur/offset/inset, aspect-ratio padding (all 23 ratios + free), 16.7 MP / 8192 px cap, diagonal two-colour gradient, rounded corners, black shadow (Skia σ = 0.577·r + 0.5, dy = blur/2) clipped outside the image; all 30 preset colours. Deviations: a fixed 16.7 MP cap instead of the memory-based one; `auto` colours approximate androidx Palette (5-bit RGB bins instead of MMCQ boxes, with the same default filter and VIBRANT scoring, 50/50 dominant/vibrant blend, then OKLab lerps to white 0.6 and black 0.18). |

## Skipped features (and why)

* **jxl-tools.** No acceptable JPEG XL codec.
  * `pillow-jxl-plugin` has cp313 wheels but is **GPL-3.0-or-later**, which is incompatible with shipping in this
    Apache-2.0 app, so it was uninstalled.
  * `imagecodecs` (BSD-3; bundles libjxl including lossless JPEG↔JXL) has cp313 wheels, but it is ~140 MB and also
    bundles libheif (LGPL), so it was not added. It is the recommended route if JXL is wanted, as an optional extra.
* **multi-frame-fusion.** Not small (~300 lines: ORB + RANSAC alignment, overlap crop, linear-light long exposure,
  light trails, motion trails). A `multi_frame_fusion` tool is being added concurrently in another module, so it
  was left alone here.
* **mesh-gradients.** The feature only downloads a remote PNG collection (network). The mesh renderer lives in
  gradient-maker and overlaps `gradient_image`.
* **document-scanner.** ML Kit and camera are Android-only. The OpenCV corner detector (~150 lines) is a follow-up
  candidate.
* **compression-lab.** Mostly covered by `compress_to_size` + `compare`; the JXL/HEIC/VVC encoders are unavailable.
* **code-preview.** Needs Pygments plus highlight.js theme parsing; not small.
* **palette-pdf.** See the table above.
* **markup-layers.** Very large (~4k lines plus UI); `draw.*` covers single layers.
* **weight-resize.** `compress_to_size` exists.
* **cipher, archive-tools.** Not image editing; RAR also has licensing issues.
* **load-net-image.** Network access.
* **wallpapers-export, quick-tiles, settings, etc.** Android/UI-only.
* **audio-cover-extractor.** Out of scope.
