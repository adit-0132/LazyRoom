# Project status

_Last updated: 2026-10-27. This file is the durable record of progress; update it with every milestone._

## Milestones

| # | Milestone | State |
|---|---|---|
| 1 | Research & architecture (ImageToolbox survey, catalogue, decisions in ARCHITECTURE.md) | done |
| 2 | Headless core (IO, engine, non-destructive pipeline, projects/history, core ops, CLI) | done |
| 3 | Agent infrastructure (MCP server, discovery, previews, batch jobs, errors, tests) | done |
| 4 | Desktop GUI (Qt Quick / Material 3) | done (see gaps) |
| 5 | Advanced processing / ImageToolbox parity | filters done (520/520); textures, fractals, noise/gradients, AI models, misc presets, PDF/photomosaic/cutting **in progress** (parallel workstreams) |
| 6 | Reference-driven editing (analysis, matching, evaluation, looks) | done (benchmarked) |
| 7 | Hardening, packaging, docs | AppImage built & smoke-tested on Fedora (before the rename — needs a rebuild); Windows installer + CI defined (not executed locally); docs written |

## Verified facts (as of last update)

* Test suite: 3,433 passed / 10 skipped (Fedora 43, Python 3.13) before the new workstreams; scenario tests A–F pass.
* Operations: 579 (520 `filter.*` ImageToolbox ports: 496 faithful, 24 approximate, 0 missing).
* Style matching benchmark: mean ΔE 13.17 vs 14.03 (unedited) vs 33.9–35.2 (Reinhard/MKL/histogram).
* AppImage (310 MB, built 2026-10-14 under the old name PhotoMasta) ran CLI, MCP and GUI (offscreen) on Fedora 43.
  It has not been rebuilt since the rename to LazyRoom.

## Decisions log

* 2026-10-27: renamed PhotoMasta → LazyRoom (package `lazyroom`; commands `lazyroom`, `lazyroom-mcp`,
  `lazyroom-gui`; env `LAZYROOM_*`; projects `*.lazyroom`). Old `*.pmproj` projects still open.
* Python core + numpy/OpenCV/numba; PySide6/QML Material 3 GUI; official MCP SDK v2 (`MCPServer`).
* Working space float32 sRGB-encoded, straight alpha; ICC → sRGB on import.
* ImageToolbox ports are faithful by default; **upstream bugs that make a filter unusable are fixed** and the
  filter is marked *approximate* (Moiré, Gamma, CLAHE/Equalize LAB/LUV, FishEye). Harmless quirks are kept.
* G'MIC 3.x and 4.x are supported (tested with 3.7.6 and 4.0.6); 2.x is treated as unavailable. G'MIC 4
  changed its random generator, so seeded G'MIC filters give different (equally valid) results than on 3.x,
  and two filters needed small argument adaptations (offset stripes, marker drawing).
* G'MIC runs as an isolated subprocess (own config home, `-network -1`, optional single-thread for RNG
  determinism); G'MIC is optional at runtime.
* Remote resources without a licence (ImageToolboxRemoteResources: Cube-LUT pack, mesh gradients, lens profiles)
  are not bundled. AI model weights are downloaded on explicit request only, with per-model licence metadata.
* pillow-heif is an optional extra because its wheels include x265 (GPL-2.0-or-later).
* Style matching transfers the *grade* (relative to the source) and applies it at confidence-based strength —
  chosen after the benchmark showed absolute-statistics matching (and classic colour transfer) copies scene content.

## Known gaps / next steps

1. Finish & integrate the parallel workstreams (textures, fractals/noise/gradients, AI models, misc presets,
   PDF/photomosaic/cutting/erase) — wire new modules, run the full suite, regenerate docs, update FEATURES.md.
2. GUI: brush painting of masks on the canvas, gradient/point editors,
   settings dialog (LLM backend, path roots), batch dialog, tool runner UI (tools are currently agent/CLI only).
3. Performance: re-measure on an idle machine; tile very large exports; GPU acceleration is not used yet.
4. Windows: run CI, verify the installer and paths/Unicode handling on Windows 10/11.
5. Flatpak: generate python-deps with flatpak-pip-generator and build.
6. Issues found in real-photo trials (2026-10), not fixed yet:
   * `match_style` in `look` mode can return a near-identity tone curve for low-key, strongly graded references
     while still reporting high similarity: the similarity is measured against its own conservative target, not
     the reference. Needs tone transfer for strongly styled references and a reference-based similarity.
   * Mask feathering blurs at full resolution with a sigma proportional to the image size (~100 s per strongly
     feathered step on a 14 MP photo). Blurring at low resolution and upsampling measured ~0.1 s (max diff 0.02).
   * Several exports started concurrently can exceed the 10-minute render timeout; exports should be queued.
   * No person/subject segmentation model is installed by default, so faces/people cannot be masked
     automatically until an optional model is downloaded.
7. Optional: JPEG XL support (no Python codec wheel in the dependency set yet), file encryption tool,
   internet image loading (deliberately off: local-first).
