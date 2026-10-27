# Testing

```bash
.venv/bin/python -m pytest -q                       # everything (~3–5 min on 12 cores)
.venv/bin/python -m pytest tests/itb/test_blur.py  # one ImageToolbox group
.venv/bin/python -m pytest -m "not slow"            # skip slow model tests
```

Tests set `LAZYROOM_HOME` to temporary folders and never touch the user's real configuration or photos.
G'MIC-backed tests run when the G'MIC CLI is found (PATH, `LAZYROOM_GMIC` or `.tools/gmic/gmic`) and skip
otherwise; GUI tests run offscreen (`QT_QPA_PLATFORM=offscreen`) and skip without PySide6.

## What is covered

| Area | Tests |
|---|---|
| Every operation (600+) | `tests/test_ops_smoke.py`: defaults on RGB and RGBA, output shape/dtype/finiteness, determinism for a fixed seed, JSON-serialisable descriptions |
| ImageToolbox filters | `tests/itb/test_<group>.py`: parameter extremes for every filter, exact G'MIC command strings, line-by-line reference checks against upstream loops where feasible, family-specific semantics (greyscale outputs, palette-only dithering, block constancy, LUT layout, RNG reproduction vs JDK/Kotlin…) |
| Resampling | `tests/test_resample.py`: all 49 kernels up/down, normalisation, round trips, colour spaces, speed |
| Engine | pipeline composition, masks, groups, project serialisation, undo/redo/revert (via API, CLI, MCP and GUI tests) |
| MCP | `tests/integration/test_mcp.py`: real client over stdio — discovery, edits, inline previews, structured errors, overwrite/original protection, undo, resources |
| CLI | `tests/integration/test_cli.py`: JSON output, exit codes, project round trip, batch partial failure + resume, generic `call` |
| GUI | `tests/integration/test_gui.py`: QML loads offscreen, open/edit/live slider/commit/undo/preset/AI panel |
| Agent scenarios A–F | `tests/integration/test_scenarios.py` (see below) |
| Generators / tools / AI | `tests/test_textures.py`, `tests/test_fractals.py`, `tests/test_generators.py`, `tests/test_presets_misc.py`, `tests/test_itb_tools.py`, `tests/test_ai.py` |

### Required agent scenarios

| Scenario | Test | What it proves |
|---|---|---|
| A — natural-language vintage grade | `test_scenario_a_…` | discovery → inspect → plan → preview (warmer, lifted shadows measured) → refine → export → preset + history |
| B — reference style transfer | `test_scenario_b_…` | match_style yields editable steps + report, compare image, export, reproducible re-render |
| C — consistent batch (24 images) | `test_scenario_c_…` | background job, per-image normalisation recorded in the report, outputs more consistent than inputs |
| D — fully headless | `test_scenario_d_…` | CLI-only discovery, project, preview artifact, export, verification |
| E — human refinement | `test_scenario_e_…` | agent edits via MCP → human edits in the GUI controller → both kept in history |
| F — failure recovery | `test_scenario_f_…` | invalid params / unknown op rejected with codes & suggestions; corrupt + non-image files isolated; resume skips finished files |

## Benchmarks

* Style matching: `python scripts/eval_style_match.py` → docs/eval/style_match.md (ground-truth ΔE vs baselines).
* Operation timings are reported per step in every render (`render_preview` → `steps[].ms`); port reports list
  12 MP timings measured during development (on a heavily loaded machine — treat as upper bounds).

## Platforms

Executed on Fedora 43 (x86_64, Python 3.13). Windows builds and tests are defined in CI
(`.github/workflows/ci.yml`) but have **not** been executed locally.
