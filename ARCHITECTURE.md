# LazyRoom architecture

LazyRoom is an image editor whose **primary user is an AI agent**. The desktop
GUI is one client of the same system; nothing can be done in the GUI that an
agent cannot do through the MCP server, CLI or Python SDK.

```
 AI agent ── MCP (stdio / Streamable HTTP) ─┐
 Shell / CI ── CLI (`lazyroom`, JSON) ────┤
 Python code ── SDK (`lazyroom.LazyRoom`)┼──►  Application API  (src/lazyroom/api.py)
 Human ── Qt Quick GUI (controller.py) ──────┘      │  commands: projects, edits, render, export,
                                                      │  presets, style, batch jobs, tools, intent
                                                      ▼
                                   Workspace (app/): sessions, path policy, jobs, artifacts, op log
                                                      │
                                   Non-destructive engine (core/): Project → History → Steps
                                   (op + params + mask + opacity + blend), render cache
                                                      │
            Operations registry (core/registry.py) ── 600+ ops: adjust.*, effect.*, geometry.*,
            draw.*, generate.*, filter.* (ImageToolbox catalogue), ai.*
                                                      │
            Backends: numpy · OpenCV · numba · Pillow/LittleCMS · scipy · G'MIC CLI (optional)
                      ONNX Runtime (optional) · Tesseract (optional) · rawpy/LibRaw (optional)
```

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Language / core | Python 3.11+ with numpy, OpenCV, numba, scipy, Pillow | Fastest path to a very large, testable operation catalogue; mature imaging + ML ecosystem; first-class MCP SDK. Heavy per-pixel loops are numba-compiled (parallel, cached). |
| Agent protocol | MCP via the official Python SDK (v2, `MCPServer`) | Standard for agent tool use; stdio for local clients, Streamable HTTP for remote/local services. |
| GUI | PySide6 (Qt 6) + Qt Quick Controls **Material** style (Material 3) | Native, GPU-accelerated, high-DPI, cross-platform (Linux/Windows), LGPL; Material 3 controls out of the box. |
| Working space | float32 sRGB-encoded RGB(A), straight alpha, unclamped between steps | Matches ImageToolbox/GPUImage filter semantics; avoids 8-bit banding; ops convert to linear light or OKLab internally where physically meaningful (exposure, white balance, blur-in-linear, grading). |
| Colour management | ICC → sRGB on import (LittleCMS via Pillow), sRGB ICC embedded on export | One well-defined working space; documented in docs/PROJECT_FORMAT.md. |
| Project format | JSON `lazyroom.project/1` (`.lazyroom`) | Human- and agent-readable, diffable; versioned schema. |
| Tool surface for agents | ~27 task-level MCP tools; operations are *data* | Hundreds of operations would flood an agent's context; `list_operations`/`describe_operations` give hierarchical discovery. |
| ImageToolbox parity | Faithful ports (read from the real Kotlin/C++/Rust/GLSL sources) + G'MIC CLI for the 167 G'MIC-based filters | Same algorithms and parameter ranges as the Android app; see FEATURES.md and docs/dev/port_reports/. |

## Layers

### Core engine (`src/lazyroom/core`)
* `params.py` — typed parameters (Float, Int, Bool, Choice, Color, Point(s), FloatList, File, Text, Gradient,
  StringList) with strict validation, JSON Schema generation and GUI metadata. Parameters with `unit="px"` are in
  full-resolution pixels and are rescaled for preview proxies, so previews look like exports.
* `registry.py` — `OpSpec` + `@op` decorator. Every capability is registered once with id, category/group,
  parameter schema, alpha policy, geometry flag, backend requirements, provenance (`source`), examples.
* `pipeline.py` — `Step` (op, params, enabled, opacity, blend, mask, label, origin), group steps (nested
  pipelines sharing one mask/opacity, used for looks and style matches), rendering with a prefix-hash LRU cache
  (editing step *k* re-renders only steps *k..n*), per-step timing reports.
* `masks.py` — resolution-independent local-adjustment masks: rect, ellipse/radial, linear gradient, polygon,
  brush strokes, luminance/colour/hue ranges, skin, image masks and AI subject masks; add/subtract/intersect,
  feather, invert, density.
* `blend.py` — 24 blend modes; adjustment-layer mixing and layer-over compositing.
* `project.py` — project model, transactional edit actions (add, add_group, update, remove, move, toggle,
  duplicate, set_steps, clear), snapshot-based undo/redo/revert shared by humans and agents.
* `io.py` — safe decoding (pixel limits, ICC conversion, EXIF orientation, 16-bit PNG/TIFF via OpenCV, RAW via
  rawpy, HEIF/AVIF), encoding with metadata policy (keep / keep_no_gps / strip), atomic writes, no silent
  overwrites.
* `color.py` + `fast.py` — colour science (sRGB/linear, XYZ/Lab, OKLab, HSV/HSL, YCbCr, Bradford white balance)
  with numba-parallel fused kernels for large buffers.
* `context.py` — render context (preview scale, deterministic RNG per step, cancellation/timeouts, file resolver
  bound to the path policy).

### Operations (`src/lazyroom/ops`)
* `adjust.*` — photographic controls: exposure, white balance (relative/Kelvin/auto), contrast, brightness,
  tone (highlights/shadows/whites/blacks), levels, curves (RGB/R/G/B/luminance), HSL mixer, saturation/vibrance,
  colour balance, colour grading wheels, split toning, fade, channel mixer, B&W mixer, selective colour, dehaze,
  clarity/texture, auto tone, LUT (.cube / Hald / 512²), colour transfer.
* `effect.*` — grain, vignette, bloom, diffusion (Orton/Pro-Mist), halation, sharpen, denoise, blur, chromatic
  aberration, light leak.
* `geometry.*` — crop (normalised/pixels/aspect), resize (49 resampling kernels × 19 scaling colour spaces),
  rotate, rotate90, flip, straighten (auto-detect), perspective, keystone, canvas, trim, lens distortion,
  shape crop.
* `draw.*` — text, watermark, shapes/annotations, image overlay layers, borders.
* `generate.*` — gradient overlay, gradient map, procedural texture overlay.
* `filter.*` — the ImageToolbox catalogue (520 filters in 14 groups), see FEATURES.md.
* `ai.*` — optional ONNX models (downloaded on request).

### Application (`src/lazyroom/app`, `api.py`)
`LazyRoom` (api.py) is the single command layer and the Python SDK. `@command` registers each method for
discovery and logs every call (JSONL operation log). The `Workspace` owns open project sessions (decoded source +
preview proxies), the path policy (`app/security.py`), the background job manager (`app/jobs.py`) and preview
artifacts.

### Interfaces
* MCP server — `src/lazyroom/mcp/server.py`
* CLI — `src/lazyroom/cli/main.py`
* GUI — `src/lazyroom/gui/` (QML in `gui/qml/`)

### Style & intent
* `style/analysis.py` — measurable style descriptors (tone quantiles, black/white points, zone tints of
  near-neutral pixels, chroma distribution, hue bands, palette, grain, vignette, colour temperature).
* `style/match.py` — reference matching that fits an editable pipeline (see docs/STYLE_MATCHING.md).
* `intent/planner.py` — deterministic natural-language → steps engine; optional LLM backends (`intent/llm.py`).

## Security model (summary)
Local-first; no network access unless a user configures an LLM endpoint or requests a model download. Path
policy limits reads/writes to allowed roots, refuses sensitive/system locations, protects every open project's
source file, and requires `overwrite=true` to replace files. Destructive actions need `confirm=true`. The MCP
HTTP transport binds to loopback with a bearer token and DNS-rebinding protection. G'MIC runs as a subprocess
with an isolated config home and networking disabled; the raw G'MIC filter rejects file/exec/network commands.
Image metadata, OCR and barcode text are treated as untrusted data. See docs/SECURITY.md.
