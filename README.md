# LazyRoom

**An AI-native, cross-platform image toolbox — built so AI agents can do everything a human can, with a
beautiful desktop app on top.**

LazyRoom is a desktop equivalent of [ImageToolbox](https://github.com/T8RIN/ImageToolbox) for **Linux and
Windows**, re-architected around one principle: *the AI agent is the primary user.* Every capability is a
documented, validated operation that agents reach through **MCP**, a **headless CLI** or a **Python SDK**. The
Material 3 desktop GUI is just another client of the same engine.

* **600+ operations** — photographic adjustments (exposure, white balance, curves, HSL, colour-grading wheels,
  split toning, fade, LUTs, dehaze, clarity…), effects (grain, bloom, halation, diffusion, vignette…), geometry
  (crop, 49 resampling kernels × 19 scaling colour spaces, perspective, straighten…), text/watermarks/annotations,
  and **all 520 ImageToolbox filters** across 14 groups — ported from the original sources (see
  [docs/FILTER_PARITY.md](docs/FILTER_PARITY.md)).
* **Non-destructive projects** — ordered steps with masks (radial, gradient, brush, luminance/colour/skin ranges,
  AI subject), opacity and blend modes; groups; undo/redo shared by humans and agents; open JSON format.
* **Reference style matching** — "make photo A look like photo B": fits an *editable* grade (white balance, tone
  curve, 3-way grading, saturation, HSL, grain, vignette), reports confidence and limitations, and beats classic
  colour transfer by ~2.6× in a ground-truth benchmark ([docs/STYLE_MATCHING.md](docs/STYLE_MATCHING.md)).
* **Natural-language looks** — semantic presets (70s film, teal & orange, Polaroid, faded analog, editorial
  portrait, urban, earthy…) and a transparent intent planner; optional local/remote LLM backends.
* **Batch processing** — folders/globs, per-image normalisation for consistent series, per-file status, JSON
  reports, cancellation, resume.
* **Utility tools** — palette, OCR, QR/barcodes, collage, stitch, stack, HDR fusion, focus stacking, panorama,
  metadata edit/strip, duplicates, animations, PDF, compress-to-size, SVG tracing, textures, fractals, …
* **Local-first & safe** — originals never modified, no silent overwrites, path policy, untrusted metadata never
  treated as instructions, no network unless you opt in.

## Install

```bash
# from source (Python 3.11–3.13)
git clone https://github.com/adit-0132/LazyRoom.git && cd LazyRoom
python -m venv .venv && . .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ".[all]"                                # GUI, RAW, AI, HEIF, HTTP extras
# optional system tools
sudo dnf install gmic tesseract   # Fedora  (Ubuntu/Debian: sudo apt install tesseract-ocr)
# G'MIC 3.x or 4.x is needed for the G'MIC-based filters. Ubuntu/Debian's own gmic package can be the old
# 2.x series: install the current .deb from https://gmic.eu/download.html instead (Windows: the CLI zip).
```

Pre-built bundles: `LazyRoom-<version>-x86_64.AppImage` (Linux) and the Windows installer / portable ZIP built by
CI (see [DEVELOPMENT.md](DEVELOPMENT.md#packaging)).

## Use it

**Desktop:** `lazyroom-gui [photo.jpg]` — drag & drop a photo, pick tools from the rail, adjust in the right
panel, compare before/after (`\`), ask the AI panel ("faded 1990s magazine look"), export (`Ctrl+E`).

**Agents (MCP):** register once — `claude mcp add lazyroom -- lazyroom-mcp` (other clients:
`lazyroom agent-setup`). Then ask your agent things like *"Give ~/Pictures/beach.jpg a warm, faded 1970s film
look and save it next to the original"*. See [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md) and
[MCP_REFERENCE.md](MCP_REFERENCE.md).

**CLI:**
```bash
lazyroom ops list --category adjust
lazyroom apply in.jpg out.jpg --op adjust.white_balance temperature=20 --op effect.grain amount=15
lazyroom style match photo.jpg reference.jpg --export matched.jpg
lazyroom batch ./shoot --output-dir ./graded --preset faded-analog --normalize both
```
Full reference: [CLI_REFERENCE.md](CLI_REFERENCE.md).

**Python:**
```python
from lazyroom import LazyRoom
pm = LazyRoom()
p = pm.project_create("photo.jpg")["project_id"]
pm.preset_apply(p, "teal-orange", strength=0.8)
pm.export(p, "photo-teal.jpg")
```

## Documentation

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layers, decisions, colour pipeline, security model |
| [FEATURES.md](FEATURES.md) | Feature parity with ImageToolbox and the directive (status, tests, platforms) |
| [docs/FILTER_PARITY.md](docs/FILTER_PARITY.md) | All 520 ImageToolbox filters, one by one |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Every operation and its parameters |
| [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md) · [MCP_REFERENCE.md](MCP_REFERENCE.md) · [CLI_REFERENCE.md](CLI_REFERENCE.md) | Interfaces |
| [docs/PROJECT_FORMAT.md](docs/PROJECT_FORMAT.md) · [docs/STYLE_MATCHING.md](docs/STYLE_MATCHING.md) · [docs/SECURITY.md](docs/SECURITY.md) · [docs/MODELS.md](docs/MODELS.md) | Specs |
| [DEVELOPMENT.md](DEVELOPMENT.md) · [TESTING.md](TESTING.md) · [STATUS.md](STATUS.md) | Contributing, tests, current status |
| [LICENSE](LICENSE) · [NOTICE](NOTICE) · [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) | Licences |

## License

Apache License 2.0. LazyRoom ports algorithms from ImageToolbox, ImageToolboxLibs and Trickle (Apache-2.0,
T8RIN), aire and related crates (MIT/BSD/Apache, awxkee), GPUImage, jhlabs and others — see NOTICE and
docs/THIRD_PARTY.md. AI model weights are never bundled; each has its own licence (docs/MODELS.md).
