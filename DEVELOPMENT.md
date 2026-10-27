# Development

## Setup

```bash
uv venv --python 3.13 .venv              # or: python3.13 -m venv .venv
uv pip install --python .venv -e ".[all,dev]" pyinstaller
```
Optional tools: G'MIC CLI (`gmic`, 3.x or 4.x — older 2.x builds are reported as unavailable), Tesseract. For development without root, a portable G'MIC can live in
`.tools/gmic/gmic`; the test suite picks it up automatically (tests/conftest.py sets `LAZYROOM_GMIC`).

Environment variables: `LAZYROOM_HOME` (put config/data/cache under one folder — use it for tests),
`LAZYROOM_GMIC`, `LAZYROOM_TESSERACT`, `LAZYROOM_READ_ROOTS` / `LAZYROOM_WRITE_ROOTS`,
`LAZYROOM_APPROVAL`, `LAZYROOM_LLM_*`, `LAZYROOM_MCP_TOKEN`, `LAZYROOM_ITB_ONLY` (import only some
ImageToolbox filter groups — development isolation).

## Layout

```
src/lazyroom/
  api.py            application API = command layer = Python SDK
  core/             engine: params, registry, pipeline, masks, blend, project/history, io, colour (+numba fast paths)
  ops/              operations: adjust, effect, geometry, draw, generate, resample, itb/ (ImageToolbox catalogue)
  app/              workspace, security policy, jobs, config
  style/            analysis, matching, metrics, colour transfer
  intent/           natural-language planner (+ optional LLM backends)
  tools/            utility tools (data/new files)
  presets/          preset store; data/looks/*.json built-in looks
  ai/               optional ONNX models
  mcp/              MCP server + client registration snippets
  cli/              CLI
  gui/              PySide6 controller + QML (gui/qml)
scripts/            gen_docs.py (reference docs), eval_style_match.py (benchmark)
packaging/          PyInstaller spec, AppImage, Windows installer, Flatpak manifest
docs/dev/           porting guide, catalogue, per-group port reports
```

## Adding an operation

```python
from lazyroom.core.params import Float
from lazyroom.core.registry import op

@op("effect.my_glow", "My glow", params=[Float("amount", 30, min=0, max=100)], tags=["glow"])
def my_glow(img, ctx, amount):
    """One-line description shown to agents."""
    return img + amount / 100 * 0.1
```
Rules: pure function of `(img, ctx, **params)`; float32 sRGB 0..1 in/out; use `ctx.px()` for pixel distances,
`ctx.rng()` for randomness, `ctx.check()` in long loops; declare `alpha="full"`/`geometry=True` when relevant.
The smoke test automatically exercises every registered operation. ImageToolbox ports follow
docs/dev/ITB_PORTING.md.

## Adding a tool

See `src/lazyroom/tools/registry.py` — `@tool(name, title, params=[...])`, `fn(api, **values) -> dict`; always
go through `read_image`, `out_path`/`save` (path policy) and add the module to `TOOL_MODULES`.

## Regenerating docs

`python scripts/gen_docs.py` regenerates docs/OPERATIONS.md, MCP_REFERENCE.md, CLI_REFERENCE.md,
docs/FILTER_PARITY.md and the packaged agent guide. Run it after changing operations, tools, MCP tools or CLI.

## Packaging

* **Linux AppImage:** `packaging/linux/build_appimage.sh` → `dist/LazyRoom-<v>-x86_64.AppImage`
  (GUI by default; `LazyRoom.AppImage mcp`, `LazyRoom.AppImage ops list`, … run the CLI). Set
  `LAZYROOM_BUNDLE_GMIC=<folder>` to ship a G'MIC CLI inside the bundle (`bin/`).
* **Windows:** `packaging\windows\build_windows.ps1` → PyInstaller folder, portable ZIP and (with Inno Setup 6)
  a per-user installer that can add the `lazyroom` CLI to PATH.
* **Flatpak:** `packaging/flatpak/io.github.lazyroom.LazyRoom.yml` (experimental; see comments there).
* **CI:** `.github/workflows/ci.yml` tests on Ubuntu and Windows (Python 3.12/3.13) and builds the AppImage and
  the Windows installer.

The PyInstaller spec ships the `lazyroom` package as `.py` files (not inside the archive) because numba's
on-disk cache needs real source files; the cache itself goes to the user cache directory.
