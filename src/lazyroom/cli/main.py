"""``lazyroom`` command-line interface.

Output: JSON when stdout is not a terminal or with ``--json`` (agents/scripts),
readable text otherwise.  Every command maps onto the same API used by the MCP
server and GUI.  Exit codes: 0 ok, 1 partial batch failure, 2 invalid input,
3 not found, 4 decode/encode, 5 path/permission, 6 backend missing,
7 resource limit, 8 timeout, 9 cancelled, 10 nothing to undo/redo,
11 operation failed, 70 internal error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from ..core.errors import ErrorCode, LazyRoomError, wrap_exception
from ..core.project import PROJECT_SUFFIXES

EPILOG = """examples:
  lazyroom ops list --category adjust
  lazyroom ops describe adjust.color_grading
  lazyroom apply in.jpg out.jpg --op adjust.exposure ev=0.4 --op effect.grain amount=20
  lazyroom apply in.jpg out.jpg --preset warm-70s-film
  lazyroom project create in.jpg --save edit.lazyroom --op adjust.white_balance temperature=15
  lazyroom project preview edit.lazyroom --out preview.jpg --compare side_by_side
  lazyroom style match in.jpg reference.jpg --save matched.lazyroom --export matched.jpg
  lazyroom batch ~/Pictures/trip --output-dir out/ --preset faded-analog --normalize both --wait
  lazyroom tools run palette input=photo.jpg count=5
  lazyroom intent "warm, faded 70s film look with muted greens" --image in.jpg
  lazyroom mcp                      # MCP server on stdio (for Claude Desktop/Code, Cursor, ...)
  lazyroom call ops.list '{"query": "vignette"}'
"""


# ---------------------------------------------------------------------------
def parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        low = text.lower()
        if low in ("true", "false"):
            return low == "true"
        return text


def parse_kv(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"Expected key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = parse_value(v)
    return out


def ops_to_steps(op_args: list[list[str]] | None) -> list[dict[str, Any]]:
    steps = []
    for group in op_args or []:
        op_id, *kv = group
        steps.append({"op": op_id, "params": parse_kv(kv)})
    return steps


def load_steps_file(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("steps", [])
    if not isinstance(data, list):
        raise LazyRoomError(ErrorCode.INVALID_PARAMS, "--steps file must contain a list of steps or {steps: [...]}")
    return data


# ---------------------------------------------------------------------------
class Out:
    def __init__(self, json_mode: bool, pretty: bool) -> None:
        self.json_mode = json_mode
        self.pretty = pretty

    def emit(self, result: Any, human: Any = None) -> None:
        if self.json_mode:
            print(json.dumps({"ok": True, "result": result}, indent=2 if self.pretty else None, default=str,
                             ensure_ascii=False))
        elif human is not None:
            print(human(result) if callable(human) else human)
        else:
            print(json.dumps(result, indent=2, default=str, ensure_ascii=False))

    def error(self, err: LazyRoomError) -> None:
        if self.json_mode:
            print(json.dumps({"ok": False, "error": err.to_dict()}, indent=2 if self.pretty else None, default=str))
        else:
            msg = f"error [{err.code.value}]: {err.message}"
            if err.hint:
                msg += f"\nhint: {err.hint}"
            if err.details:
                msg += "\ndetails: " + json.dumps(err.details, default=str)[:2000]
            print(msg, file=sys.stderr)


def _table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    widths = [min(w, 60) for w in widths]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join("  ".join(str(c)[:60].ljust(w) for c, w in zip(r, widths)) for r in rows)
    return f"{line}\n{sep}\n{body}"


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lazyroom", description="AI-native image toolbox (headless CLI).",
                                epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true", help="force JSON output (default when stdout is not a TTY)")
    p.add_argument("--pretty", action="store_true", help="indent JSON output")
    p.add_argument("--config", help="path to a config.json")
    p.add_argument("--version", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("info", help="capabilities manifest (operations, formats, backends, workflows)")
    sub.add_parser("formats", help="supported formats")
    sub.add_parser("backends", help="optional backends and AI models")

    ops = sub.add_parser("ops", help="discover editing operations").add_subparsers(dest="sub")
    o = ops.add_parser("list")
    o.add_argument("--category")
    o.add_argument("--group")
    o.add_argument("--query", "-q")
    o.add_argument("--limit", type=int, default=1000)
    o = ops.add_parser("describe")
    o.add_argument("ids", nargs="+")

    i = sub.add_parser("inspect", help="metadata, statistics and style analysis of an image")
    i.add_argument("path")
    i.add_argument("--thumbnail", action="store_true")

    a = sub.add_parser("apply", help="apply operations / a preset / a steps file to an image and export")
    a.add_argument("input")
    a.add_argument("output")
    _pipeline_args(a)
    _export_args(a)

    pr = sub.add_parser("project", help="non-destructive projects (.lazyroom)").add_subparsers(dest="sub")
    c = pr.add_parser("create")
    c.add_argument("source")
    c.add_argument("--save", required=True, help="project file to write")
    c.add_argument("--name")
    _pipeline_args(c)
    s = pr.add_parser("show")
    s.add_argument("project")
    s.add_argument("--history", action="store_true")
    e = pr.add_parser("edit", help="apply edit actions (JSON) or --op additions; saves the project")
    e.add_argument("project")
    e.add_argument("--actions", help="JSON list of actions or @file.json")
    e.add_argument("--op", nargs="+", action="append", metavar=("OP", "KEY=VALUE"))
    e.add_argument("--description")
    for name in ("undo", "redo"):
        u = pr.add_parser(name)
        u.add_argument("project")
        u.add_argument("--steps", type=int, default=1)
    v = pr.add_parser("preview")
    v.add_argument("project")
    v.add_argument("--out", help="copy the preview JPEG here")
    v.add_argument("--size", type=int, default=1024)
    v.add_argument("--compare", choices=["none", "side_by_side", "split"], default="none")
    v.add_argument("--version-id", default="current", help="original | current | <history id>")
    x = pr.add_parser("export")
    x.add_argument("project")
    x.add_argument("output")
    _export_args(x)
    cm = pr.add_parser("compare")
    cm.add_argument("project")
    cm.add_argument("--a", default="original")
    cm.add_argument("--b", default="current")

    ps = sub.add_parser("preset", help="looks and presets").add_subparsers(dest="sub")
    q = ps.add_parser("list")
    q.add_argument("--query", "-q")
    q.add_argument("--kind", choices=["all", "look", "user"], default="all")
    q = ps.add_parser("show")
    q.add_argument("id")
    q = ps.add_parser("save")
    q.add_argument("name")
    q.add_argument("--from-project")
    q.add_argument("--steps", help="steps JSON file")
    q.add_argument("--description", default="")
    q.add_argument("--tags", default="")
    q.add_argument("--overwrite", action="store_true")
    q = ps.add_parser("delete")
    q.add_argument("id")
    q.add_argument("--confirm", action="store_true")

    st = sub.add_parser("style", help="style analysis and reference matching").add_subparsers(dest="sub")
    q = st.add_parser("analyze")
    q.add_argument("path")
    q = st.add_parser("match")
    q.add_argument("source", help="image or .lazyroom")
    q.add_argument("reference")
    q.add_argument("--save", help="write the resulting project")
    q.add_argument("--export", help="export the matched image")
    q.add_argument("--strength", type=float, default=None, help="0..1 (default: automatic, confidence-based)")
    q.add_argument("--mode", choices=["parametric", "parametric+lut"], default="parametric")
    q.add_argument("--matching", choices=["look", "exact"], default="look",
                   help="look: transfer the grade only; exact: match the reference statistics (same scene)")
    q.add_argument("--no-skin", action="store_true")
    q.add_argument("--overwrite", action="store_true")

    b = sub.add_parser("batch", help="process many images with one pipeline (runs in the foreground)")
    b.add_argument("inputs", nargs="+", help="files, folders or glob patterns")
    b.add_argument("--output-dir", required=True)
    _pipeline_args(b)
    b.add_argument("--project", help="use this .lazyroom's pipeline")
    b.add_argument("--recursive", action="store_true")
    b.add_argument("--naming", default="{stem}{ext}")
    b.add_argument("--format")
    b.add_argument("--quality", type=int, default=92)
    b.add_argument("--metadata", choices=["keep", "keep_no_gps", "strip"], default="keep")
    b.add_argument("--overwrite", action="store_true")
    b.add_argument("--skip-existing", action="store_true", help="resume: skip files whose output exists")
    b.add_argument("--normalize", choices=["none", "exposure", "white_balance", "both"], default="none")
    b.add_argument("--match-reference")
    b.add_argument("--workers", type=int, default=0)
    b.add_argument("--dry-run", action="store_true")

    t = sub.add_parser("tools", help="utility tools").add_subparsers(dest="sub")
    q = t.add_parser("list")
    q.add_argument("--query", "-q")
    q = t.add_parser("describe")
    q.add_argument("name")
    q = t.add_parser("run")
    q.add_argument("name")
    q.add_argument("params", nargs="*", help="key=value pairs (values parsed as JSON when possible)")
    q.add_argument("--params-json")

    it = sub.add_parser("intent", help="translate a natural-language request into edit steps")
    it.add_argument("request")
    it.add_argument("--image", help="image (or .lazyroom) to tailor the plan to")
    it.add_argument("--save", help="write a project with the planned steps (needs --image)")
    it.add_argument("--llm", action="store_true", help="use the configured LLM backend")

    cl = sub.add_parser("call", help="call any API command with JSON arguments")
    cl.add_argument("command", help="e.g. ops.list, render.apply (see 'lazyroom info')")
    cl.add_argument("args", nargs="?", default="{}", help="JSON object or @file.json")

    m = sub.add_parser("mcp", help="run the MCP server")
    m.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    m.add_argument("--host", default="127.0.0.1")
    m.add_argument("--port", type=int, default=8765)
    m.add_argument("--allow-remote", action="store_true", help="allow binding to non-loopback addresses")

    g = sub.add_parser("gui", help="launch the desktop application")
    g.add_argument("path", nargs="?")

    md = sub.add_parser("models", help="optional AI models").add_subparsers(dest="sub")
    md.add_parser("list")
    q = md.add_parser("download")
    q.add_argument("name")
    q.add_argument("--yes", action="store_true", help="confirm the download")

    cf = sub.add_parser("config", help="show or edit configuration").add_subparsers(dest="sub")
    cf.add_parser("show")
    q = cf.add_parser("set")
    q.add_argument("key")
    q.add_argument("value")

    ag = sub.add_parser("agent-setup", help="print MCP registration snippets for AI clients")
    ag.add_argument("client", nargs="?", default="all",
                    choices=["all", "claude-code", "claude-desktop", "cursor", "vscode", "codex", "gemini", "generic"])
    return p


def _pipeline_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--op", nargs="+", action="append", metavar=("OP", "KEY=VALUE"),
                   help="add an operation step, e.g. --op adjust.exposure ev=0.5 (repeatable)")
    p.add_argument("--steps", help="JSON file with a list of steps")
    p.add_argument("--preset", help="preset / look id")


def _export_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--format")
    p.add_argument("--quality", type=int, default=92)
    p.add_argument("--bit-depth", type=int, choices=[8, 16], default=8)
    p.add_argument("--metadata", choices=["keep", "keep_no_gps", "strip"], default="keep")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")


# ---------------------------------------------------------------------------
def _api(args: argparse.Namespace):
    from ..api import LazyRoom
    from ..app.config import Config

    return LazyRoom(config=Config.load(args.config) if args.config else None)


def _collect_steps(args) -> tuple[list[dict[str, Any]] | None, str | None]:
    steps = load_steps_file(getattr(args, "steps", None)) + ops_to_steps(getattr(args, "op", None))
    preset = getattr(args, "preset", None)
    if preset and steps:
        raise LazyRoomError(ErrorCode.INVALID_PARAMS, "Use either --preset or --op/--steps, not both")
    return (steps or None), preset


def _open(api, path: str) -> str:
    if path.lower().endswith(PROJECT_SUFFIXES):
        return api.project_open(path)["project_id"]
    return api.project_create(path)["project_id"]


def run(args: argparse.Namespace, out: Out) -> int:
    cmd = args.cmd
    if args.version:
        from .. import __version__

        out.emit({"version": __version__}, lambda r: f"lazyroom {r['version']}")
        return 0
    if cmd is None:
        build_parser().print_help()
        return 0
    if cmd == "mcp":
        from ..mcp.server import serve

        serve(transport=args.transport, host=args.host, port=args.port, allow_remote=args.allow_remote,
              config_path=args.config)
        return 0
    if cmd == "gui":
        from ..gui.app import main as gui_main

        return gui_main([args.path] if args.path else [])
    if cmd == "agent-setup":
        from ..mcp.clients import snippets

        out.emit(snippets(args.client), lambda r: "\n\n".join(f"## {k}\n{v}" for k, v in r.items()))
        return 0
    api = _api(args)
    if cmd == "info":
        cap = api.capabilities()
        out.emit(cap, lambda c: (
            f"LazyRoom {c['version']} (API {c['api_version']})\n{c['summary']}\n\nOperations: "
            f"{c['operations']['total']} ({c['operations']['available']} available)\n" +
            "\n".join(f"  {k:9} {v['count']:4}  {v['description']}" for k, v in c["operations"]["categories"].items()) +
            "\n\nBackends: " + ", ".join(f"{k}={'yes' if v['available'] else 'no'}" for k, v in c["backends"].items()) +
            "\nRun 'lazyroom info --json' for the full machine-readable manifest."))
    elif cmd == "formats":
        out.emit(api.list_formats())
    elif cmd == "backends":
        out.emit(api.backends_status())
    elif cmd == "ops":
        if args.sub == "list":
            r = api.list_operations(category=args.category, group=args.group, query=args.query, limit=args.limit)
            out.emit(r, lambda r: _table([[o["id"], o["title"], o.get("group", ""), "" if o.get("available", True)
                                            else "unavailable"] for o in r["operations"]],
                                          ["id", "title", "group", ""]) + f"\n{r['total']} operations")
        elif args.sub == "describe":
            out.emit(api.describe_operations(args.ids))
        else:
            raise LazyRoomError(ErrorCode.INVALID_PARAMS, "use: ops list | ops describe")
    elif cmd == "inspect":
        out.emit(api.inspect_image(args.path, thumbnail=args.thumbnail))
    elif cmd == "apply":
        steps, preset = _collect_steps(args)
        r = api.apply(args.input, args.output, steps=steps, preset=preset, format=args.format, quality=args.quality,
                      bit_depth=args.bit_depth, metadata=args.metadata, overwrite=args.overwrite, dry_run=args.dry_run)
        out.emit(r, lambda r: f"wrote {r['output']} ({r.get('width')}x{r.get('height')}, {r.get('bytes')} bytes)"
                 if not r.get("dry_run") else json.dumps(r, indent=2))
    elif cmd == "project":
        return _project(api, args, out)
    elif cmd == "preset":
        if args.sub == "list":
            r = api.presets_list(query=args.query, kind=args.kind)
            out.emit(r, lambda r: _table([[p["id"], p["kind"], p["name"], ", ".join(p["tags"][:5])] for p in r["presets"]],
                                          ["id", "kind", "name", "tags"]))
        elif args.sub == "show":
            out.emit(api.preset_get(args.id))
        elif args.sub == "save":
            pid = api.project_open(args.from_project)["project_id"] if args.from_project else None
            steps = load_steps_file(args.steps) if args.steps else None
            out.emit(api.preset_save(args.name, steps=steps, project_id=pid, description=args.description,
                                     tags=[t for t in args.tags.split(",") if t], overwrite=args.overwrite))
        elif args.sub == "delete":
            out.emit(api.preset_delete(args.id, confirm=args.confirm))
    elif cmd == "style":
        if args.sub == "analyze":
            r = api.style_analyze(args.path)
            out.emit(r, lambda r: "\n".join(["Style: " + "; ".join(r["style"]["descriptors"]),
                                             json.dumps({k: r["style"][k] for k in ("tone", "color")}, indent=1)]))
        elif args.sub == "match":
            pid = _open(api, args.source)
            r = api.style_match(pid, args.reference, strength=args.strength, mode=args.mode,
                                matching=args.matching, preserve_skin=not args.no_skin)
            if args.save:
                r["saved"] = api.project_save(pid, args.save)["path"]
            if args.export:
                r["export"] = api.export(pid, args.export, overwrite=args.overwrite)
            out.emit(r, lambda r: (f"similarity {r['report']['similarity']} ({r['report']['confidence']}), "
                                   f"loss {r['report']['loss_initial']} -> {r['report']['loss_final']}\n" +
                                   "\n".join(f"  {s['op']}: {json.dumps(s['params'])}" for s in r["steps"]) +
                                   ("\nlimitations:\n  - " + "\n  - ".join(r["report"]["limitations"])
                                    if r["report"]["limitations"] else "")))
    elif cmd == "batch":
        steps, preset = _collect_steps(args)
        pid = api.project_open(args.project)["project_id"] if args.project else None
        r = api.batch_start(args.inputs, args.output_dir, steps=steps, preset=preset, project_id=pid,
                            recursive=args.recursive, naming=args.naming, format=args.format, quality=args.quality,
                            metadata=args.metadata, overwrite=args.overwrite, skip_existing=args.skip_existing,
                            normalize=args.normalize, match_reference=args.match_reference, workers=args.workers,
                            dry_run=args.dry_run)
        if args.dry_run:
            out.emit(r)
            return 0
        job_id = r["job_id"]
        last = -1.0
        while True:
            s = api.job_wait(job_id, timeout=1.0)
            if not out.json_mode and s["progress"] != last:
                print(f"\r{s['succeeded'] + s['failed'] + s['skipped']}/{s['total']} done, {s['failed']} failed",
                      end="", file=sys.stderr, flush=True)
                last = s["progress"]
            if s["status"] in ("completed", "partial", "failed", "cancelled"):
                break
        if not out.json_mode:
            print(file=sys.stderr)
        final = api.job_status(job_id, items=True)
        out.emit(final, lambda r: f"{r['status']}: {r['succeeded']} ok, {r['failed']} failed, {r['skipped']} skipped"
                 + (f"\nreport: {r['report_path']}" if r.get("report_path") else "")
                 + "".join(f"\n  FAILED {f['input']}: {f['error']['message']}" for f in r.get("items", [])
                           if f["status"] == "failed"))
        return 0 if final["status"] == "completed" else 1
    elif cmd == "tools":
        if args.sub == "list":
            r = api.tools_list(args.query)
            out.emit(r, lambda r: _table([[t["name"], t["category"], t["title"], "" if t.get("available", True)
                                           else "unavailable"] for t in r["tools"]], ["name", "category", "title", ""]))
        elif args.sub == "describe":
            out.emit(api.tool_describe(args.name))
        elif args.sub == "run":
            params = parse_kv(args.params)
            if args.params_json:
                params.update(json.loads(args.params_json))
            out.emit(api.tool_run(args.name, params))
    elif cmd == "intent":
        pid = _open(api, args.image) if args.image else None
        r = api.intent_plan(args.request, project_id=pid, apply=bool(args.save), use_llm=True if args.llm else None)
        if args.save:
            r["saved"] = api.project_save(pid, args.save)["path"]
        out.emit(r, lambda r: "\n".join([f"engine: {r['engine']}"] +
                                        [f"  '{i['phrase']}' -> {i['effect']}" for i in r["interpretation"]] +
                                        [f"note: {n}" for n in r["notes"]] +
                                        [json.dumps(a) for a in r["actions"]]))
    elif cmd == "call":
        return _call(api, args, out)
    elif cmd == "models":
        from ..ai import models

        if args.sub == "list":
            out.emit({"models": models.list_models()})
        elif args.sub == "download":
            if not args.yes:
                raise LazyRoomError(ErrorCode.APPROVAL_REQUIRED, "Model downloads need --yes (they fetch large files)")
            out.emit(models.download_model(args.name))
    elif cmd == "config":
        from ..app.config import Config

        cfg = Config.load(args.config)
        if args.sub == "show":
            out.emit(cfg.to_dict())
        elif args.sub == "set":
            obj, key = cfg, args.key
            if "." in key:
                head, key = key.split(".", 1)
                obj = getattr(cfg, head)
            if not hasattr(obj, key):
                raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"Unknown config key {args.key}")
            setattr(obj, key, parse_value(args.value))
            out.emit({"saved": str(cfg.save(args.config)), args.key: getattr(obj, key)})
    return 0


def _project(api, args, out: Out) -> int:
    sub = args.sub
    if sub == "create":
        steps, preset = _collect_steps(args)
        r = api.project_create(args.source, name=args.name, steps=steps)
        pid = r["project_id"]
        if preset:
            api.preset_apply(pid, preset)
        path = api.project_save(pid, args.save)["path"]
        out.emit(api.project_state(pid) | {"path": path})
        return 0
    pid = api.project_open(args.project)["project_id"]
    if sub == "show":
        out.emit(api.project_state(pid, history=args.history))
    elif sub == "edit":
        actions: list[dict[str, Any]] = []
        if args.actions:
            raw = Path(args.actions[1:]).read_text(encoding="utf-8") if args.actions.startswith("@") else args.actions
            actions = json.loads(raw)
        actions += [{"type": "add", **s} for s in ops_to_steps(args.op)]
        r = api.project_edit(pid, actions, actor="user", description=args.description)
        api.project_save(pid)
        out.emit(r)
    elif sub in ("undo", "redo"):
        r = (api.project_undo if sub == "undo" else api.project_redo)(pid, args.steps)
        api.project_save(pid)
        out.emit(r)
    elif sub == "preview":
        r = api.render_preview(pid, max_size=args.size, compare=args.compare, version=args.version_id)
        if args.out:
            import shutil

            api.ws.policy.check_write(args.out, overwrite=True)
            shutil.copyfile(r["artifact"]["path"], args.out)
            r["artifact"]["copied_to"] = args.out
        out.emit(r, lambda r: f"preview: {r['artifact'].get('copied_to') or r['artifact']['path']} "
                              f"({r['render_ms']} ms)")
    elif sub == "export":
        r = api.export(pid, args.output, format=args.format, quality=args.quality, bit_depth=args.bit_depth,
                       metadata=args.metadata, overwrite=args.overwrite, dry_run=args.dry_run)
        out.emit(r, lambda r: f"wrote {r['output']}" if not r.get("dry_run") else json.dumps(r, indent=2))
    elif sub == "compare":
        out.emit(api.compare_versions(pid, a=args.a, b=args.b))
    return 0


def _call(api, args, out: Out) -> int:
    from ..api import COMMANDS

    if args.command not in COMMANDS:
        import difflib

        raise LazyRoomError(ErrorCode.UNKNOWN_COMMAND, f"Unknown command '{args.command}'",
                              details={"suggestions": difflib.get_close_matches(args.command, list(COMMANDS), 5)},
                              hint="See 'lazyroom info --json' -> commands")
    raw = Path(args.args[1:]).read_text(encoding="utf-8") if args.args.startswith("@") else args.args
    kwargs = json.loads(raw or "{}")
    if not isinstance(kwargs, dict):
        raise LazyRoomError(ErrorCode.INVALID_PARAMS, "Arguments must be a JSON object")
    fn = getattr(api, COMMANDS[args.command]["method"])
    try:
        result = fn(**kwargs)
    except TypeError as exc:
        raise LazyRoomError(ErrorCode.INVALID_PARAMS, f"Bad arguments for {args.command}: {exc}") from exc
    out.emit(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = Out(json_mode=args.json or not sys.stdout.isatty(), pretty=args.pretty)
    try:
        return run(args, out)
    except LazyRoomError as e:
        out.error(e)
        return e.exit_code
    except KeyboardInterrupt:
        out.error(LazyRoomError(ErrorCode.CANCELLED, "Interrupted"))
        return 130
    except json.JSONDecodeError as e:
        err = LazyRoomError(ErrorCode.INVALID_PARAMS, f"Invalid JSON: {e}")
        out.error(err)
        return err.exit_code
    except Exception as e:  # pragma: no cover - last resort, never crash with a bare traceback
        err = wrap_exception(e)
        out.error(err)
        return err.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
