"""One-time registration snippets for MCP-capable AI clients.

No client discovers arbitrary local MCP servers automatically; each needs the
server command registered once.  ``lazyroom agent-setup`` prints these.
"""
from __future__ import annotations

import json
import shutil
import sys


def _command() -> tuple[str, list[str]]:
    import os
    from pathlib import Path

    if os.environ.get("APPIMAGE"):  # AppImage: the mount point is temporary; register the AppImage itself
        return os.environ["APPIMAGE"], ["mcp"]
    if getattr(sys, "frozen", False):  # bundled app: the CLI executable sits next to the interpreter stub
        return str(Path(sys.executable).with_name("lazyroom" + (".exe" if sys.platform == "win32" else ""))), ["mcp"]
    beside = Path(sys.executable).with_name("lazyroom-mcp" + (".exe" if sys.platform == "win32" else ""))
    if beside.exists():
        return str(beside), []
    exe = shutil.which("lazyroom-mcp")
    if exe:
        return exe, []
    return sys.executable, ["-m", "lazyroom.mcp.server"]


def snippets(client: str = "all") -> dict[str, str]:
    cmd, args = _command()
    server = {"command": cmd, "args": args}
    out = {
        "claude-code": f"claude mcp add lazyroom -- {cmd} {' '.join(args)}".strip(),
        "claude-desktop": "Add to claude_desktop_config.json (Settings > Developer > Edit Config):\n"
                          + json.dumps({"mcpServers": {"lazyroom": server}}, indent=2),
        "cursor": "Add to ~/.cursor/mcp.json (or .cursor/mcp.json in a project):\n"
                  + json.dumps({"mcpServers": {"lazyroom": server}}, indent=2),
        "vscode": "Add to .vscode/mcp.json (VS Code / GitHub Copilot agent mode):\n"
                  + json.dumps({"servers": {"lazyroom": {"type": "stdio", **server}}}, indent=2),
        "codex": "Add to ~/.codex/config.toml:\n[mcp_servers.lazyroom]\ncommand = " + json.dumps(cmd)
                 + "\nargs = " + json.dumps(args),
        "gemini": "Add to ~/.gemini/settings.json:\n" + json.dumps({"mcpServers": {"lazyroom": server}}, indent=2),
        "generic": "stdio: " + " ".join([cmd, *args]) + "\nStreamable HTTP: lazyroom mcp --transport http "
                   "(http://127.0.0.1:8765/mcp, header 'Authorization: Bearer <token>' — token printed at start)",
    }
    return out if client == "all" else {client: out[client]}
