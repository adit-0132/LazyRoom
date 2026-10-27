# Third-party code, data and dependencies

LazyRoom is licensed under the Apache License 2.0 (see LICENSE). It contains ports of, bundles, or depends on
the following works. Licences of **AI model weights** are tracked separately in docs/MODELS.md.

## Ported source code (algorithms re-implemented in Python from the original sources)

| Work | Author / project | Licence | Used for |
|---|---|---|---|
| ImageToolbox | T8RIN (Malik Mukhametzyanov) — github.com/T8RIN/ImageToolbox | Apache-2.0 | Filter catalogue (520 filters: parameters, defaults, algorithms), looks/LUT assets, feature parity |
| ImageToolboxLibs (incl. jhlabs Rust port, G'MIC command builders, fast-noise, gpuimage) | T8RIN | Apache-2.0 | Filter algorithms, G'MIC command strings |
| Trickle | T8RIN | Apache-2.0 | Dithering, glitch, Gotham/HDR/etc. filter algorithms |
| aire | Radzivon Bartoshyk (awxkee) | MIT | Tone mapping, CLAHE, blurs, colour matrices, morphology, ... |
| pic-scale | Radzivon Bartoshyk | BSD-3-Clause OR Apache-2.0 | Resampling kernels |
| libblur, histogram_equalization, colorutils-rs, yuv (Rust crates) | Radzivon Bartoshyk | BSD-3-Clause OR Apache-2.0 | Blur and equalisation algorithms, colour-space conversions |
| palette-transfer (crate) | — | see crate (MIT/Apache) | PaletteTransferVariant |
| GPUImage for Android | CyberAgent, Inc. | Apache-2.0 | Shader-based filters (ported GLSL maths) |
| GPUImage | Brad Larson | BSD-3-Clause | Original shaders / lookup LUT images (Amatorka, MissEtikate, SoftElegance) |
| Java Image Filters (jhlabs) | Jerry Huxtable | Apache-2.0 | Distortion and effect filters |
| FastNoiseLite | Jordan Peck & contributors | MIT | Noise generation |
| ntsc-rs | valadaptive | MIT / ISC / Apache-2.0 | VHS NTSC emulation |
| Clatter | — | MIT / Apache-2.0 / zlib | Simplex noise used by the NTSC port |
| OKLab | Björn Ottosson | public domain / MIT | Perceptual colour space |

## Bundled data

| Data | Source | Licence |
|---|---|---|
| `data/luts/lookup_*.webp` (17 lookup LUTs) | ImageToolbox repository | Apache-2.0 (GPUImage-origin LUTs: BSD-3) |
| `data/models/face_detection_yunet_2026may.onnx` | OpenCV Zoo YuNet (via ImageToolbox assets) | MIT |
| `tests/data/photos/*` | scikit-image sample data | public domain / CC0 (see tests/data/photos/README.md) |

Not bundled (licence unclear): ImageToolbox's downloadable Cube-LUT pack, mesh-gradient pack and lens profiles
(github.com/T8RIN/ImageToolboxRemoteResources has no licence).

## Runtime dependencies (Python packages)

| Package | Licence | Notes |
|---|---|---|
| numpy, scipy | BSD-3-Clause | |
| opencv-python-headless | Apache-2.0 | wheels bundle FFmpeg (LGPL-2.1+), libvpx, libaom, OpenBLAS |
| Pillow | MIT-CMU (HPND) | uses LittleCMS (MIT), libjpeg-turbo, libwebp, libavif, … |
| numba / llvmlite | BSD-2-Clause | LLVM (Apache-2.0 with LLVM exception) |
| pydantic | MIT | |
| mcp (Model Context Protocol SDK) | MIT | starlette/uvicorn (BSD-3) |
| piexif | MIT | |
| segno | BSD-3-Clause | QR codes |
| zxing-cpp | Apache-2.0 | barcode read/write |
| pypdf | BSD-3-Clause | |
| PySide6 (optional `[gui]`) | LGPL-3.0 (Qt) | dynamic linking; Qt source available from qt.io |
| rawpy (optional `[raw]`) | MIT | LibRaw: LGPL-2.1 / CDDL-1.0 |
| onnxruntime (optional `[ai]`) | MIT | |
| pillow-heif (optional `[heif]`) | BSD-3-Clause | wheels bundle libheif (LGPL-3.0), libde265 (LGPL-3.0) and **x265 (GPL-2.0-or-later)** |
| anthropic (optional `[llm]`) | MIT | only used when the Anthropic LLM backend is configured |

## External programs (optional, invoked as separate processes)

| Program | Licence | Notes |
|---|---|---|
| G'MIC CLI | CeCILL-2.1 (GPL-compatible) / CeCILL-C for libgmic | 167 `filter.*` operations; bundled in release builds as a separate executable |
| Tesseract OCR | Apache-2.0 | `ocr` tool |

## Binary distributions

Release bundles (AppImage, Flatpak, Windows installer) are aggregates of separately licensed components. When a
bundle includes pillow-heif (x265, GPL-2.0-or-later) and/or the G'MIC CLI (CeCILL-2.1), those components are
distributed under their own terms (source available from their upstream projects; offer in packaging/SOURCES.md),
and LazyRoom's own Apache-2.0 code is compatible with GPLv3 for such combined distribution. Builds that must
stay GPL-free can omit `[heif]` and G'MIC; LazyRoom degrades gracefully (HEIC import and G'MIC filters report
`BACKEND_UNAVAILABLE`).
