"""MCP server integration tests: a real MCP client talks to the server over stdio."""
import json
import os
import sys

import anyio
import pytest

from lazyroom.core.io import save_image
from tests.conftest import synthetic_image

pytestmark = pytest.mark.timeout(240)


def _data(res):
    assert res.content, res
    return json.loads(res.content[0].text)


def test_mcp_stdio_end_to_end(tmp_path):
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    src = tmp_path / "in.jpg"
    save_image(synthetic_image(320, 240), src)
    env = dict(os.environ, LAZYROOM_HOME=str(tmp_path / "home"), LAZYROOM_ITB_ONLY=os.environ.get("LAZYROOM_ITB_ONLY", ""))
    params = StdioServerParameters(command=sys.executable, args=["-m", "lazyroom.mcp.server"], env=env)

    async def run():
        async with Client(params, read_timeout_seconds=120) as c:
            tools = {t.name for t in (await c.list_tools()).tools}
            for expected in ("list_operations", "describe_operations", "open_image", "edit_project",
                             "render_preview", "export_image", "match_style", "start_batch", "run_tool"):
                assert expected in tools
            assert len(tools) <= 30, "tool surface must stay compact"
            ops = _data(await c.call_tool("list_operations", {"category": "adjust"}))
            assert ops["total"] >= 20
            desc = _data(await c.call_tool("describe_operations", {"ids": ["adjust.exposure"]}))
            assert desc["operations"][0]["params"][0]["name"] == "ev"
            st = _data(await c.call_tool("open_image", {"path": str(src)}))
            pid = st["project_id"]
            ed = _data(await c.call_tool("edit_project", {"project_id": pid, "actions": [
                {"type": "add", "op": "adjust.exposure", "params": {"ev": 0.5}},
                {"type": "add", "op": "effect.vignette", "params": {"amount": -30}}]}))
            assert ed["step_count"] == 2
            pv = await c.call_tool("render_preview", {"project_id": pid, "max_size": 256})
            kinds = [b.type for b in pv.content]
            assert "image" in kinds, "preview must be returned inline"
            bad = await c.call_tool("edit_project", {"project_id": pid, "actions": [
                {"type": "add", "op": "adjust.exposure", "params": {"ev": 99}}]})
            assert bad.is_error
            err = json.loads(bad.content[0].text)["error"]
            assert err["code"] == "INVALID_PARAMS" and "range" in err["message"]
            out = tmp_path / "out.jpg"
            ex = _data(await c.call_tool("export_image", {"project_id": pid, "output": str(out)}))
            assert out.exists() and ex["width"] == 320
            again = await c.call_tool("export_image", {"project_id": pid, "output": str(out)})
            assert again.is_error and json.loads(again.content[0].text)["error"]["code"] == "OUTPUT_EXISTS"
            orig = await c.call_tool("export_image", {"project_id": pid, "output": str(src), "overwrite": True})
            assert orig.is_error and json.loads(orig.content[0].text)["error"]["code"] == "PATH_NOT_ALLOWED"
            h = _data(await c.call_tool("project_history", {"project_id": pid, "action": "undo"}))
            assert h["steps"] == []
            res = await c.read_resource("lazyroom://manifest")
            assert "operations" in json.loads(res.contents[0].text)

    anyio.run(run)
