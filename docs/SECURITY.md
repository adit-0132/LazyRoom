# Security and privacy

LazyRoom is **local-first**: images never leave the machine unless the user explicitly configures a remote LLM
endpoint *and* enables `llm.send_preview_image` (only text and numeric image statistics are sent otherwise), or
explicitly requests an AI model download.

## Filesystem policy (`src/lazyroom/app/security.py`)

* Reads allowed only under `read_roots`; writes only under `write_roots` (defaults: home, temp, current directory;
  configure with `lazyroom config set read_roots '[...]'` or `LAZYROOM_READ_ROOTS` / `LAZYROOM_WRITE_ROOTS`).
* Paths are resolved (symlinks, `..`) before checking; sensitive locations (`.ssh`, `.gnupg`, `.aws`, keyrings,
  …) and system directories are always refused for writing.
* **Originals are protected**: the source file of every open project can never be an export target.
* Existing files are never replaced unless `overwrite=true` is passed; writes are atomic (temp file + rename).
* Destructive actions (deleting presets, moving/renaming originals) require `confirm=true`;
  `approval_policy="never"` refuses them entirely.

## Untrusted input

* Decoding limits: max pixels (default 250 MP, decompression-bomb guard), file-type detection by content, format
  errors mapped to `DECODE_FAILED` / `UNSUPPORTED_FORMAT` without crashing batches.
* Metadata, OCR text, barcode contents and file names are treated as **data, never instructions**; MCP results
  label them as untrusted.
* Render timeouts and cooperative cancellation (`CancelToken`) bound runaway operations; batch failures are
  isolated per file.

## External processes

* **G'MIC** runs as a subprocess with an isolated config home (user command files and updates are never loaded,
  nothing is written to the user's home) and networking disabled (`-network -1`).
* `filter.raw_gmic` (user-supplied G'MIC pipelines) rejects shell execution, file I/O, network access, variable
  substitution and side-effecting math functions via a conservative denylist + path/URL detection; it is bounded
  by a timeout.
* `filter.shader` (custom GLSL) runs in a separate process with Qt's offscreen OpenGL, so a driver crash cannot
  take the app down.
* Tesseract and fc-match are invoked with fixed argument lists (no shell).

## Network interfaces

* MCP stdio: inherits the client's trust (local process).
* MCP Streamable HTTP: binds to `127.0.0.1` by default; requires `Authorization: Bearer <token>` (random 256-bit
  token stored with 0600 permissions, or `LAZYROOM_MCP_TOKEN`); DNS-rebinding protection (Host/Origin
  allow-lists) on loopback; binding to other interfaces requires `--allow-remote`.

## Agent safety defaults

* Preview-first: `render_preview` never writes outside the artifact cache; `dry_run` on export, apply and batch.
* Every command is logged to `<data dir>/logs/operations.jsonl` (command, arguments summary, duration, status,
  error code) for audit.

## Reporting

Please report vulnerabilities privately to the maintainers before public disclosure.
