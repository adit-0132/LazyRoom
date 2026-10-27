# Port report: ImageToolbox preset catalogues ("presets_misc")

Code:
* `src/lazyroom/ops/shapes.py` — `SHAPES` (id -> `render(h, w, **params)` -> float32 mask), ops
  `geometry.crop_to_shape` and `draw.styled_line` (imported from `ops/__init__.py`).
* `src/lazyroom/tools/presets_misc.py` — tools `color_library`, `palette_export`, `palette_import`,
  `shader_presets`, `collage_layouts`, `collage_layout`, `export_profiles`, `export_profile`
  (registered in `TOOL_MODULES`). Private codecs: `tools/_palette_formats.py` (lib/palette port),
  `tools/_collage_layouts.py` (lib/collages port).
* Data: `data/color_library.json`, `data/shaders/*.itshader`, `data/collage_layouts.json`,
  `data/export_profiles.json`.
* Tests: `tests/test_presets_misc.py` (568 test cases):
  `LAZYROOM_ITB_ONLY=none .venv/bin/python -m pytest tests/test_presets_misc.py -q -x` passes
  (the one GL test, which runs the bundled shaders through `filter.shader`, is skipped unless the effects group is loaded; it passes with
  `LAZYROOM_ITB_ONLY=effects`).

Upstream sources: ImageToolbox (Apache-2.0) `core/resources/.../shapes`, `feature/crop`, `lib/cropper`,
`feature/draw`, `lib/colors`, `feature/color-library`, `lib/palette`, `feature/palette-tools`,
`feature/shader-studio`, `core/filters/.../shader`, `lib/collages`, `feature/collage-maker`,
`core/domain/.../BuiltInImageExportProfile.kt`; AOSP androidx (Apache-2.0) `compose/material3/MaterialShapes.kt`
and `graphics/graphics-shapes` (`RoundedPolygon`, `CornerRounding`, `Cubic`, `Shapes.kt`), fetched from
androidx-main (ImageToolbox pins material3 1.5.0-alpha29; the shape definitions are the stable ones).

## Summary

| Catalogue | Items | faithful | approximate | not ported |
|---|---|---|---|---|
| Crop shapes (`geometry.crop_to_shape`, `SHAPES`) | 56 (23 crop + 30 Material 3 + image mask + 2 icon-only hearts) | 55 | 1 | 0 |
| Draw line styles (`draw.styled_line`) | 5 styles, 51 stamp shapes | 5 | 0 | 0 |
| Colour library (`color_library`) | 33,219 upstream names | search + nearest | data: 31,308 shipped | 1,911 names (licence unverifiable) |
| Palette formats (`palette_export` / `palette_import`) | 42 formats | 38 (4 with a documented bug fix) | 4 (IMAGE; 3 zip containers byte-differ) | 0 |
| Shader Studio presets (`shader_presets`) | 0 bundled upstream | library format/validation | — | nothing to port; 5 LazyRoom examples |
| Collage layouts (`collage_layouts`, `collage_layout`) | 316 layouts | 316 geometry | rendering details (see below) | gradients, per-image gestures |
| Built-in export profiles (`export_profiles`, `export_profile`) | 61 profiles | 61 | — | — |
| Other preset lists | see last section | | | |

## 1. Crop shapes

Rendering: every outline is built from the upstream path data / algorithm as cubic Béziers and
filled with an exact-area scanline rasteriser (signed-area accumulation, font-rs style, numba), so edges are
analytically anti-aliased at any size (~0.1 s for 12 MP). Skia's own AA differs by sub-pixel amounts only.
Paths and their scaling rules were extracted programmatically from the Kotlin (no hand transcription):
`Path().apply{}` shapes scale by size / (baseWidth, baseHeight); `PathShape` scales by size / the path's
*control-point* bounds (Android `Path.computeBounds` = Skia `getBounds`) about the origin; cropper
`CustomPathOutline`s are scaled to their bounds and moved to (0, 0) like `CropAgent`; Material shapes use
ImageToolbox's `RoundedPolygon.toShape()` (normalised polygon scaled by the crop size).
The 30 Material 3 shapes are built by a Python port of graphics-shapes: `RoundedCorner` (rounding radius +
smoothing, cut allocation between neighbouring corners, flanking curves, circular arcs), `RoundedPolygon`
cubic list construction (first-corner split), `normalized()` with approximate bounds, `circle`, `rectangle`,
`star`, Compose `Matrix.rotateZ/scale`, and `MaterialShapes.customPolygon/doRepeat` (incl. mirroring).
Order and ids follow `DefaultOutlineProperties.kt` (ids 0–22, Material shapes 500–529, image mask 23).

`geometry.crop_to_shape` params: `shape`; `corner_radius` (0..50 %, default 20, "Radius" slider of
`CropMaskSelection`, used by rounded/cut corners); `aspect_ratio` — the crop-frame presets of
`DomainAspectRatio.defaultList` (free, original, 1:1, 10:16, … 2:1; `custom` is `geometry.crop_aspect`),
applied as the largest centred crop of that ratio (the shape is stretched over the crop rect, as upstream);
`rotation` (LazyRoom extension: rotates the outline about the centre and shrinks it uniformly if it would
leave the frame — upstream's -45..45 rotation rotates the *image*, which is `geometry.rotate`); `mask_image`.
Deviation: the default shape is `rounded_corners` (upstream's crop screen defaults to Rect because it is also
the plain crop tool; a plain rect crop here is `geometry.crop`). Output is RGBA, alpha × mask; RGB values are untouched.

| Shape | id | upstream definition | status | notes |
|---|---|---|---|---|
| Rect | `rect` | `RectCropShape` | faithful |  |
| Rounded Corners | `rounded_corners` | `RoundedCornerShape(percent)`; radius 0..50 % of the shorter side (default 20) | faithful | Corner arcs are cubic quarter-circle approximations (Skia uses conics; < 0.03 % radius error). |
| Cut Corners | `cut_corners` | `CutCornerShape(percent)`; 0..50 % (default 20) | faithful |  |
| Oval | `oval` | Kotlin `Path()` (base 1000 x 1000), scaled by size / base | faithful |  |
| Squircle | `squircle` | Kotlin `Path()` (base 1000 x 1000), scaled by size / base | faithful |  |
| Octagon | `octagon` | Kotlin `Path()` (base 1000 x 1000), scaled by size / base | faithful |  |
| Rounded Pentagon | `pentagon` | Kotlin `Path()` (base 1224.19 x 1137.39), scaled by size / base | faithful | README calls it "Rounded Pentagon". |
| Clover | `clover` | Kotlin `Path()` (base 200 x 200), scaled by size / base | faithful |  |
| Pill | `pill` | `PathShape` data, scaled by size / control-point bounds about the origin | faithful |  |
| Bookmark | `bookmark` | `PathShape` data, scaled by size / control-point bounds about the origin | faithful |  |
| Burger | `burger` | `PathShape` data, scaled by size / control-point bounds about the origin | faithful |  |
| Shuriken | `shuriken` | `PathShape` data, scaled by size / control-point bounds about the origin | faithful |  |
| Explosion | `explosion` | `PathShape` data, scaled by size / control-point bounds about the origin | faithful |  |
| Map | `map` | `PathShape` data, scaled by size / control-point bounds about the origin | faithful |  |
| Material Star | `material_star` | Kotlin `Path()` (base 865.081 x 865.081), scaled by size / base | faithful |  |
| Small Material Star | `small_material_star` | Kotlin `Path()` (base 2695.79 x 2680.65), scaled by size / base | faithful |  |
| Shield | `shield` | `PathShape` data, scaled by size / control-point bounds about the origin | faithful |  |
| Egg | `egg` | `PathShape` data, scaled by size / control-point bounds about the origin | faithful |  |
| Droplet | `droplet` | `PathShape` data, scaled by size / control-point bounds about the origin | faithful |  |
| Arrow | `arrow` | `PathShape` data, scaled by size / control-point bounds about the origin | faithful |  |
| Kotlin Logo | `kotlin` | Kotlin `Path()` (base 1000 x 1000), scaled by size / base | faithful | README: "Kotlin Logo". |
| Heart | `heart` | lib/cropper `Paths.Favorite` (CustomPathOutline: scaled to bounds, moved to 0,0) | faithful |  |
| Star | `star` | lib/cropper `Paths.Star` (CustomPathOutline) | faithful |  |
| Material Slanted | `m3_slanted` | `MaterialShapes.Slanted` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Arch | `m3_arch` | `MaterialShapes.Arch` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Fan | `m3_fan` | `MaterialShapes.Fan` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Arrow | `m3_arrow` | `MaterialShapes.Arrow` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Semi Circle | `m3_semi_circle` | `MaterialShapes.SemiCircle` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Oval | `m3_oval` | `MaterialShapes.Oval` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Triangle | `m3_triangle` | `MaterialShapes.Triangle` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Diamond | `m3_diamond` | `MaterialShapes.Diamond` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Clam Shell | `m3_clam_shell` | `MaterialShapes.ClamShell` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Gem | `m3_gem` | `MaterialShapes.Gem` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Sunny | `m3_sunny` | `MaterialShapes.Sunny` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Very Sunny | `m3_very_sunny` | `MaterialShapes.VerySunny` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Cookie 4 | `m3_cookie_4` | `MaterialShapes.Cookie4Sided` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Cookie 6 | `m3_cookie_6` | `MaterialShapes.Cookie6Sided` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Cookie 9 | `m3_cookie_9` | `MaterialShapes.Cookie9Sided` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Cookie 12 | `m3_cookie_12` | `MaterialShapes.Cookie12Sided` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Ghostish | `m3_ghostish` | `MaterialShapes.Ghostish` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Clover 4 | `m3_clover_4` | `MaterialShapes.Clover4Leaf` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Clover 8 | `m3_clover_8` | `MaterialShapes.Clover8Leaf` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Burst | `m3_burst` | `MaterialShapes.Burst` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Soft Burst | `m3_soft_burst` | `MaterialShapes.SoftBurst` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Boom | `m3_boom` | `MaterialShapes.Boom` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Soft Boom | `m3_soft_boom` | `MaterialShapes.SoftBoom` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Flower | `m3_flower` | `MaterialShapes.Flower` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Puffy | `m3_puffy` | `MaterialShapes.Puffy` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Puffy Diamond | `m3_puffy_diamond` | `MaterialShapes.PuffyDiamond` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Pixel Circle | `m3_pixel_circle` | `MaterialShapes.PixelCircle` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Pixel Triangle | `m3_pixel_triangle` | `MaterialShapes.PixelTriangle` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Bun | `m3_bun` | `MaterialShapes.Bun` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Material Heart | `m3_heart` | `MaterialShapes.Heart` -> `RoundedPolygon.normalized().toShape()` | faithful |  |
| Image Mask | `image_mask` | `ImageMaskOutline` | approximate | Mask image stretched with bilinear filtering; its alpha is the mask. Deviation: masks without an alpha channel use their luminance (upstream would keep the whole image). |
| Heart (icon shape) | `heart_cubic` | Kotlin `Path()` (base 1 x 1), scaled by size / base | faithful | Icon-shape catalogue only (`HeartShape`); not in the crop list. |
| Simple Heart (icon shape) | `simple_heart` | Kotlin `Path()` (base 24 x 24), scaled by size / base | faithful | Icon-shape catalogue only (`SimpleHeartShape`, same path as Heart but base-scaled). |

## 2. Draw line styles (`draw.styled_line`)

Port of `DrawLineStyle` as rendered by `AndroidImageDrawApplier.asPathEffect` for the Pen mode (round caps
and joins). The polyline is given as normalised points (`closed` optional); sizes use ImageToolbox `Pt`
(`px = min(w, h) * pt / 500`), so previews and exports match without `ctx.px`. Line width 1..100 pt
(default 20 = `defaultDrawLineWidth`), colour default black (`defaultDrawColor`). Strokes are the union of
round-capped segments (distance-based AA); stamped/zigzag outputs are filled like Skia's `Path1DPathEffect`.

| Style | id | upstream | status | notes |
|---|---|---|---|---|
| Default | `none` | plain stroke, `StrokeCap.Round` / `StrokeJoin.Round` | faithful | |
| Dashed | `dashed` | `dashPathEffect([size, gap + strokeWidth], 0)` | faithful | dash/gap 0..100 pt (defaults 10 / 20). Skia's special handling of a dash crossing the start of a *closed* contour is not reproduced. |
| Dot Dashed | `dot_dashed` | `dashPathEffect([4w, 2w, w/4, 2w], 0)` | faithful | |
| Zigzag | `zigzag` | `stampedPathEffect(zigzag path, advance = w, Morph)` | faithful | ratio 0.5..20 (default 4); exact stamp path incl. the (-w/4, -w/4) offset. |
| Stamped | `stamped` | `stampedPathEffect(shape outline w × w, advance = spacing + w, Morph)` | faithful | spacing 0..100 pt (default 20); stamp choices = `IconShape.entriesNoRandom` (51: squircle, rounded 15/25/35/45 %, cut 25/35/50 %, the ITB shapes and 25 Material shapes), default Material Star. Skia `kMorph` reproduced: stamp points bent along the path via position/tangent at `distance + x` (pinned to the path length); lines stay straight, curve control points are bent. |

Not ported: gradient strokes along the path (`supportsGradient`), freehand smoothing of touch input (points are
used as given), other draw modes (neon, highlighter, privacy blur, warp, spot heal: interactive tools, not
preset lists).

## 3. Colour library (`color_library`)

Upstream data: `lib/colors/src/main/assets/color_names.json`, 33,219 `#AARRGGBB -> name` entries, no
attribution or licence notice in the repository. Provenance check (against the current upstream lists):
29,908 entries match meodai/color-names exactly (MIT, © David Aerne), 1,350 more match ntc.js
(Name that Color, Chirag Mehta, CC BY 2.5), 50 more are CSS named colours (W3C keywords, some with
ImageToolbox's "Variant" suffix); 1,911 entries (e.g. "Klein Blue", "Aurometalsaurus", numbered/suffixed
duplicates) match no clearly licensed source. Per the licensing rule, **only the 31,308 verified entries are
shipped**, with the MIT text and the CC BY 2.5 attribution embedded in `data/color_library.json`
(`licenses`, `counts`). The excluded entries are documented here, not shipped.

| Item | status | notes |
|---|---|---|
| Name data | approximate | 31,308 / 33,219 names (verified subset, see above). |
| Search (name / hex) | faithful | `ColorLibraryComponent.updateSearch`: name contains query, query contains name, or hex contains query; case-insensitive; sorted by name. Addition: exact (case-insensitive) name matches are listed first. |
| Nearest name | faithful | `ColorNameParser.parseColorName`: exact hex, else minimum Euclidean RGB distance (0..1); the tool returns the `limit` nearest with distances (first = upstream result). Also used to name unnamed palette colours. |
| Favourites | not ported | per-user UI state. |

## 4. Palette formats (`palette_export`, `palette_import`)

`_palette_formats.py` ports `lib/palette` (PaletteFormat enum: 42 formats; models `Palette`, `ColorGroup`,
`PaletteColor` with RGB/CMYK/LAB/Gray conversions, BytesReader/Writer, CSV/XML helpers). Byte layouts,
endianness, string encodings, scaling/rounding and quirks follow the Kotlin; golden-byte tests cover ACO
v1+v2, ASE, ACT 768/772, RIFF, GPL, DCP and CPL headers, and every format round-trips names / palette name /
groups where it can store them. Safety: DTD/entity declarations are rejected before XML parsing; zip entries
are capped at 50 MB. Format auto-detection follows `PaletteToolsComponent`: formats matching the extension
first, then the export order (`PaletteFormatHelper`), first decode with ≥ 1 colour wins.

Tools: `palette_export` writes `colors` ("#hex" or "Name=#hex"; unnamed colours get colour-library names like
ImageToolbox), an image's palette (`input`, `count`), or converts another palette file (`source`, keeps colour
spaces and groups); format from `format` or the output extension. `palette_import` returns name, groups,
colours (hex, RGBA, original colour space + components, spot/normal type) and an optional swatch image.
The image palette is **approximate**: ImageToolbox uses androidx `Palette` (MMCQ, max 32 by default, up to 128)
named via `parseColorName`, distinct by name, sorted by luminance then hue; LazyRoom uses its `palette`
tool (median cut, ≤ 32 colours) with the same naming, de-duplication and sort.

| Format (enum) | id | extensions | encode | decode | status | notes |
|---|---|---|---|---|---|---|
| ACB | acb | acb | ✓ | ✓ | faithful | one colour space per file (mixed → RGB); `$$$…=Title` handling |
| ACO | aco | aco | ✓ | ✓ | faithful + fix | writes v1+v2; LAB is written as RGB (as upstream). Fix: LAB a/b read as signed 16-bit (Adobe spec; upstream reads unsigned → garbage for negative values) |
| ACT | act | act | ✓ | ✓ | faithful | 768/772-byte layouts; no names |
| ANDROID_XML | android_xml | xml | ✓ | ✓ | faithful + fix | Fix: `#RRGGBB`/`#RGB` read as RGB (upstream applies ARGB byte order, so `#FF0000` became blue) |
| ASE | ase | ase | ✓ | ✓ | faithful | groups; RGB/CMYK/LAB/Gray; global/spot/normal; the format stores no palette name |
| BASIC_XML | basic_xml | xml | ✓ | ✓ | faithful | |
| COREL_PAINTER | corel_painter | txt | ✓ | ✓ | faithful | |
| COREL_DRAW | corel_draw | xml | ✓ | ✓ | faithful | random guid as upstream; pages → groups; colour-space templates |
| SCRIBUS_XML | scribus_xml | xml | ✓ | ✓ | faithful | RGB + CMYK hex |
| COREL_PALETTE | corel_palette | cpl | ✓ | ✓ | faithful | writes 0xDCDC/BGR; reads all versions incl. long/spot records; non-ASCII palette names read back as `�` (upstream quirk) |
| CSV | csv | csv | ✓ | ✓ | faithful | alpha dropped; a one-line file loses its names |
| DCP | dcp | dcp | ✓ | ✓ | faithful | |
| GIMP | gimp | gpl | ✓ | ✓ | faithful | rounds half-up |
| HEX_RGBA | hex_rgba | hex | ✓ | ✓ | faithful | |
| IMAGE | image | png, jpg, jpeg | ✓ | ✓ | approximate | Pillow instead of Android Bitmap: same layout (32×32 swatch per colour, names appended after the PNG end) but different PNG bytes; Android premultiplied-alpha rounding not copied; decodes PNG/JPEG/WebP/GIF/BMP/ICO |
| JSON | json | jsoncolorpalette, json | ✓ | ✓ | faithful | kotlinx layout incl. random `id`s |
| OPEN_OFFICE | open_office | soc | ✓ | ✓ | faithful | |
| PAINT_NET | paint_net | txt | ✓ | ✓ | faithful | mixed `\n`/`\r\n` line endings as upstream |
| PAINT_SHOP_PRO | paint_shop_pro | psppalette, pal | ✓ | ✓ | faithful | |
| RGBA | rgba | rgba, txt | ✓ | ✓ | faithful | |
| RGB | rgb | rgb, txt | ✓ | ✓ | faithful | |
| RIFF | riff | pal | ✓ | ✓ | faithful | plus upstream's non-standard `name` chunk |
| SKETCH | sketch | sketchpalette | ✓ | ✓ | faithful | |
| SKP | skp | skp | ✓ | ✓ | faithful | |
| SVG | svg | svg | ✓ | ✓ | faithful | groups flattened on decode; group rows overlap (upstream quirk) |
| SWIFT | swift | swift | ✓ | ✓ | faithful | labelled export-only upstream but a decoder exists and the app probes it; unnamed colours dropped on decode |
| KOTLIN | kotlin | kt | ✓ | ✓ | faithful | as SWIFT; decode drops duplicate colours; name reads back as `ExportedPalette` |
| COREL_DRAW_V3 | corel_draw_v3 | pal | ✓ | ✓ | faithful | CMYK in whole percent, half-to-even |
| CLF | clf | clf | ✓ | ✓ | faithful | LAB |
| SWATCHES | swatches | swatches | ✓ | ✓ | approximate | Procreate HSB; same zip entries/order/compression, container bytes differ (no data descriptors, timestamps) |
| INFINITE_PAINTER | infinite_painter | clrs | ✓ | ✓ | faithful | signed ARGB ints; alpha forced opaque; no names |
| AUTODESK_COLOR_BOOK | autodesk_color_book | acb | ✓ | ✓ | faithful | group names read back as "Color Page N" |
| SIMPLE_PALETTE | simple_palette | color-palette | ✓ | ✓ | faithful | linear sRGB, 4 decimals |
| SWATCHBOOKER | swatchbooker | sbz | ✓ | ✓ | approximate | zip container bytes differ (content faithful); RGB/CMYK/Lab/GRAY/HSV/HSL |
| AFPALETTE | afpalette | afpalette | ✓ | ✓ | faithful | marker scanner incl. upstream's naive matcher |
| XARA | xara | jcw | ✓ | ✓ | faithful | names truncated to 14 Latin-1 bytes |
| KOFFICE | koffice | colors | ✓ | ✓ | faithful | |
| HPL | hpl | hpl | ✓ | ✓ | faithful | |
| SKENCIL | skencil | spl | ✓ | ✓ | faithful | unnamed colours dropped on decode |
| VGA_24BIT | vga_24bit | vga24 | ✓ | ✓ | faithful + fix | Fix: `; Names:` header cut at the newline byte (upstream uses a char index as byte offset, breaking non-ASCII names) |
| VGA_18BIT | vga_18bit | vga18 | ✓ | ✓ | faithful + fix | 6-bit, half-to-even; same header fix |
| KRITA | krita | kpl | ✓ | ✓ | approximate | zip container bytes differ (content faithful); writes sRGB; reads sRGB/CMYK/LAB/Gray |

Other notes: XML goes through ElementTree replayed as SAX events, so an undeclared namespace prefix is a
parse error (upstream's non-namespace SAX parser accepts it for Android/Basic/Scribus XML); Java number
formatting (`Double.toString`, `%.6f`, `DecimalFormat`) is emulated, the last digit may differ in rare ties.
Without a recognised extension, ACT/VGA_24BIT/CSV/RGB accept almost anything (as in the app).

## 5. Shader Studio presets (`shader_presets`)

ImageToolbox bundles **no** shader presets or templates: Shader Studio starts from an empty draft and user
presets live in DataStore (`AndroidShaderPresetRepository`); the only bundled `assets` are unrelated (colour
names, collage previews, emoji SVGs, Lottie, the YuNet model, filter update JSON). So there is nothing to port.

| Item | status | notes |
|---|---|---|
| Bundled presets | not ported | none exist upstream. |
| `.itshader` file format | faithful | Library files use `ShaderPresetCodec.toItShaderJson`'s layout; import validates with the `filter.shader` port of `ShaderParser`/`ShaderValidator`; file names via `sanitizedFileName`. |
| Library: list / show / import / export | faithful | `show` returns the preset JSON to pass as `filter.shader`'s `shader` plus default `values`. User presets go to `<LazyRoom data>/shaders/` and override bundled ones with the same name. |
| Library: delete / share / in-app editor | not ported | delete = remove the file; editing is plain text. |
| Example presets (`data/shaders`) | LazyRoom-authored | Vignette, Duotone, Posterize, RGB Split, Threshold — written for this port (not from ImageToolbox) to cover every parameter type (float, int, bool, color, vec2); validated by `parse_shader` and rendered through `filter.shader` in the tests. |

## 6. Collage layouts (`collage_layouts`, `collage_layout`)

`data/collage_layouts.json` holds all **316** layouts of `CollageLayoutFactory.COLLAGE_MAP` (1–20 images,
per-count distribution checked against upstream). It was generated by compiling the real upstream layout and
geometry Kotlin (with stand-ins for Android classes) and dumping each layout through upstream's own loader —
no hand-porting; layouts with draggable dividers are evaluated at their initial positions; circle/heart paths
are flattened to polylines. Cross-check: every layout rendered like upstream's preview generator, median IoU
0.988 vs the upstream frame shapes (minimum 0.945, from integer rounding at different resolutions).

`_collage_layouts.render_collage` reproduces frame placement and integer rounding, spacing shrink rules
(2× at the canvas border, 1× between frames, the five special shrink methods), rounded corners incl. concave
corners, circle/heart/hexagon/rectangle paths and clear areas; each image covers its frame bound, centred
(upstream's initial placement); later frames draw on top; 4×4 supersampled AA with upstream's fill rule.
Defaults from the Collage Maker: spacing 10, corner radius 0, white background (upstream sliders 0..50).

| Item | status | notes |
|---|---|---|
| 316 layout geometries | faithful | evaluated at initial divider positions; upstream quirks kept (`collage_15_2` has 17 frames, 1-px seams at spacing 0). |
| Spacing / corner radius / background | approximate | units are output pixels; upstream measures them in on-screen preview pixels × export scale (device dependent), so they match only when the export equals the preview size. |
| Image placement | approximate | initial cover-and-centre placement; resampling bilinear with area pre-shrink instead of Android's filter. |
| Degenerate geometry | approximate | cases where upstream draws spikes/NaN are guarded (corner stays sharp, negative path scale clamped, hull loop capped). |
| Fewer images than frames | faithful | empty frames show the background. |
| Gradient backgrounds, per-image pan/zoom/rotate, divider dragging, snap-to-border | not ported | interactive editor features. |

The existing grid/justified `collage` tool is unchanged.

## 7. Built-in export profiles (`export_profiles`, `export_profile`)

`data/export_profiles.json`: all **61** `BuiltInImageExportProfile.entries` (ids, platforms, English titles,
size, format, quality, resize type), extracted from the Kotlin, incl. the inline `telegram_sticker` and the
Flexible override of `telegram_custom_emoji`.

| Item | status | notes |
|---|---|---|
| Catalogue (61 profiles, 18 platforms) | faithful | |
| `flexible` resize | faithful | `flexibleResize` with `ResizeAnchor.Default`: long side = max(width, height), truncating ints; upscales smaller images like upstream. |
| `center_crop` resize | faithful | `performCenterCrop(upscaleToFitCanvas = true)`: upscale factor max(tw/w, th/h) clamped ≥ 1, image drawn centred on a tw × th canvas — large photos are **cropped at native scale, not downscaled** (upstream behaviour, documented in the tool). Offsets are integer (upstream draws at float offsets). |
| Scaling filter | faithful | Lanczos3 in linear light (`ImageScaleMode.Lanczos3`, `ScaleColorSpace.Default = Linear`) via LazyRoom's pic-scale port (`ops/resample.py`; see its report for kernel-level differences). |
| Encoding | approximate | JPG q90 / WebP lossy q85·q80 / PNG; EXIF stripped (`keepExif = false`; `null` → strip). Transparent pixels in JPG output are composited on white (Android's encoder yields black). Output name `<stem>_<profile>.<ext>`. |

## 8. Other bundled preset lists

| Catalogue | status | where / why |
|---|---|---|
| ASCII art gradients (`Gradient.NORMAL … NUMERICAL`, 8) | faithful (already) | `filter.ascii` (pixelation port) lists all 8 in its `gradient` description; the `ascii_art` tool accepts any charset. Nothing added. |
| Bundled fonts (29 TTFs: Montserrat, Comfortaa, Caveat, … Unisource) | not ported | font binaries with individual licences (OFL/others) not documented per file upstream; LazyRoom uses system fonts or a font path (`font` params). |
| Crop aspect-ratio presets (`DomainAspectRatio.defaultList`) | faithful | `geometry.crop_to_shape.aspect_ratio`; any ratio via `geometry.crop_aspect`. |
| Resize presets (`Preset.Percentage` 100…50, `AspectRatio(fit/fill)`, `Telegram`) | covered | percentages / aspect fit-fill = `geometry.resize`, `geometry.crop_aspect`, `geometry.canvas`; Telegram = `export_profile telegram_sticker` (same 512 × 512 PNG Flexible settings). No separate tool. |
| Crop shapes / icon shapes (`IconShape.entriesNoRandom`) | faithful | sections 1–2. Material shapes that ImageToolbox does not use (Circle, Square, Pill, Pentagon, Cookie7Sided) are not included. |
| Draw line styles | faithful | section 2. |
| Draw path modes (rect, oval, line, arrows, polygon, star, lasso, spray, flood fill) | not ported here | interactive drawing tools rather than preset lists; rect/ellipse/line/arrow/polygon exist as `draw.shape`. |
| Watermark types (`WatermarkingType`: Text, Image, Stamp.Text, Stamp.Time) | partly covered | Text/Image/Stamp.Text ≈ existing `draw.watermark` (single/tiled). Stamp.Time (date-format stamp) and the hidden/LSB digital watermark are features with their own algorithms, not preset data — not ported here. |
| Collage layouts | faithful | section 6. |
| Export profiles | faithful | section 7. |
| Emoji / sticker SVGs (`core/resources/assets/svg`, 389 files) | not ported | used as draw/watermark stamps; their source and licence are not documented in the repository (licence unclear → not shipped). `draw.image` / `draw.watermark` accept any image file. |
| Shader presets | n/a | none upstream (section 5). |
| Wallpapers export | not ported | Android-only: exports the device's current home/lock wallpapers; no desktop equivalent. |
| Lottie animations, confetti types, switch/slider/shape UI types | not ported | app UI only, no image-processing meaning. |
