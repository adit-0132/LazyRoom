# Porting ImageToolbox filters to LazyRoom

This guide is the contract for porting the ~520 filters of
[ImageToolbox](https://github.com/T8RIN/ImageToolbox) (Apache-2.0) into
`src/lazyroom/ops/itb/<group>.py`.  The catalogue with every filter, its
group, English title and source files is `docs/dev/itb_catalog.json`.

## Where the upstream sources are (local clones)

| What | Path |
|---|---|
| UI filter (title, params, **ranges and defaults**) | `/tmp/ref/ImageToolbox/core/filters/src/main/java/com/t8rin/imagetoolbox/core/filters/presentation/model/Ui<Name>Filter.kt` |
| ITB implementation | `/tmp/ref/ImageToolbox/feature/filters/src/main/java/com/t8rin/imagetoolbox/feature/filters/data/model/<Name>Filter.kt` (+ `../utils/**`, `../transformation/**`) |
| Param data classes / enums | `/tmp/ref/ImageToolbox/core/filters/src/main/java/com/t8rin/imagetoolbox/core/filters/domain/model/{params,enums}/` |
| English strings | `/tmp/ref/ImageToolbox/core/resources/src/main/res/values/strings.xml` |
| aire (MIT) Kotlin API / C++ / Rust | `/tmp/ref/aire/aire/src/main/java/com/awxkee/aire/`, `/tmp/ref/aire/aire/src/main/cpp/`, `/tmp/ref/aire/aire-filters/src/` |
| Trickle (Apache-2.0) | `/tmp/ref/Trickle/trickle/src/main/{java,cpp,rust}` |
| ImageToolboxLibs (Apache-2.0): jhlabs, gpuimage (GLSL in Java strings), gmic command builders, fast-noise, ... | `/tmp/ref/ImageToolboxLibs/libs/` |
| G'MIC command builders (exact command strings) | `/tmp/ref/ImageToolboxLibs/libs/gmic/src/main/java/com/t8rin/gmic/filters/*.kt` |
| OpenCV helpers used by ITB | `/tmp/ref/ImageToolbox/lib/opencv-tools/`, `/tmp/ref/ImageToolbox/lib/*` |
| G'MIC CLI (dev) | `.tools/gmic/gmic` (`gmic -h fx_name` prints a command's doc) |

Always read the *actual* implementation (not just the README) and port the
algorithm faithfully.  Note: `Ui*Filter` names sometimes differ slightly from
the implementation file name (e.g. `...Pixelization` vs `...PixelationFilter.kt`).

## Conventions

* One module per ImageToolbox group: `src/lazyroom/ops/itb/<group>.py`
  (`simple, color, lut, light, enhancement, artistic, frames, effects, blur,
  pixelation, distortion, glitch, geometry, dithering`).  Private helpers go in
  `src/lazyroom/ops/itb/_<group>_impl.py`.  **Do not edit shared files**
  (`core/**`, `ops/_util.py`, `ops/itb/_common.py`, `ops/itb/__init__.py`,
  other groups' modules) — other people are editing in parallel.  If you need a
  shared helper, copy it into your private module.
* Register with `itb_op(Name, title, GROUP, params, upstream=..., requires=..., tags=[...])`
  from `._common`.  `Name` = the catalogue `name` (ImageToolbox class name
  without `Ui`/`Filter`).  This gives the operation id `filter.<snake_case(Name)>`.
  `title` = catalogue `title` (English).  `upstream` = the library that actually
  implements it in ImageToolbox: `GPUImage`, `aire`, `Trickle`, `jhlabs`,
  `G'MIC`, `OpenCV`, `ImageToolbox`, `fast-noise`, ...
* Parameters mirror the `Ui*Filter` `paramsInfo`: **same ranges, same
  defaults** (the default `value` of the Ui class).  Name parameters in
  `snake_case` after their string resource (`R.string.threshold` ->
  `threshold`); give a `title` (English string) and, when helpful, a one-line
  `description`.  Use the param types in `lazyroom.core.params`:
  `Float, Int, Bool, Choice, Color, Point, Points, FloatList, Gradient, File, Text`.
  * Integer-valued sliders (`roundTo = 0`, `isInteger = true`) -> `Int`.
  * Enums -> `Choice` with snake_case choices (keep upstream order, map back to the upstream value inside the op).
  * `ColorModel` -> `Color` (value is an `(r, g, b, a)` float tuple 0..1; Android ints like `-8355712` are ARGB).
  * `ImageModel`/`FileModel` inputs -> `File(..., file_kind="image"|"lut")` and load via `ctx.resolve_file(path)` + `lazyroom.core.io.load_image`.
  * Pixel-distance parameters (radii, sizes, offsets in px) get `unit="px"` and must be scaled with `ctx.px(v)` / `ctx.ipx(v)` so previews match exports.
  * Filters with no parameters (`Unit`) -> `params=[]`.
* Buffer conventions: input `img` is float32 `(H, W, 3)` sRGB-encoded 0..1
  (alpha is split off and re-attached automatically).  If the filter reads or
  writes alpha, or changes the image size, pass `alpha="full"` (and
  `geometry=True` when the size changes) and handle `(H, W, 3|4)` input.
  Outputs are clamped to 0..1 automatically (ImageToolbox is 8-bit).
* Upstream 8-bit math: convert constants (e.g. thresholds in 0..255) to 0..1.
  Android `Bitmap` colours are ARGB ints.  GPUImage shaders work on 0..1 floats
  (texture coordinates normalised 0..1, `texelWidth = 1/width`).
* Randomness: never use global RNG; use `ctx.rng(salt)` (deterministic).
* Performance: vectorise with numpy/OpenCV (`cv2.remap` for distortions,
  `cv2.filter2D`/`GaussianBlur`/`blur` for convolutions).  Inherently
  sequential loops (error diffusion, scanline algorithms) must use
  `numba.njit(cache=True)`.  No Python per-pixel loops over the image.
  Target: < 2 s on a 12 MP image for typical filters.
* G'MIC filters: build the exact command with `gmic_cmd(name, *args)` /
  `gmic_pipeline(...)` (same argument order and value mapping as the Kotlin
  builder, including enum `.value` integers and Booleans as 0/1), run with
  `run_gmic(img, cmd, ctx=ctx, preserve_alpha=...)` and declare
  `requires=["gmic"], upstream="G'MIC"`.  Check parameter ranges from the
  Kotlin `inRange(...)` calls.  Verify every G'MIC filter actually runs with
  the dev CLI.
* Reference examples (already done): `filter.brightness`, `filter.cinematic_dream`
  (light.py), `filter.autumn_tones` (simple.py), `filter.amatorka` (lut.py).

## Tests

* The generic smoke test (`tests/test_ops_smoke.py`) runs every operation with
  defaults, on RGB and RGBA, and checks determinism.
* Add `tests/itb/test_<group>.py` with: (1) a parametrised test running each op
  of the group at parameter minima and maxima (numeric params) — must not
  crash and must return finite values of the right shape; (2) a handful of
  semantic assertions per sub-family (e.g. grayscale filters return R=G=B,
  pixelation produces constant blocks, blur reduces high-frequency energy,
  dithering output uses only the palette colours, LUT strength 0 is identity).
* Run: see "Isolation while porting in parallel" below.
  (conftest sets `LAZYROOM_GMIC` to the dev G'MIC binary).

## Report

At the end, write `docs/dev/port_reports/<group>.md` with a table:
`| Filter | op id | upstream | status | notes |` where status is one of
`faithful` (same algorithm), `approximate` (explain the deviation),
`not ported` (explain why).  Never mark something faithful that isn't.

## Isolation while porting in parallel

Other groups are being ported concurrently and may be temporarily broken.
Always run tests with `LAZYROOM_ITB_ONLY=<your groups>` so only your modules
are imported, e.g.:

```bash
cd <repository root>
LAZYROOM_ITB_ONLY=blur .venv/bin/python -m pytest tests/itb/test_blur.py tests/test_ops_smoke.py -q -x
```
