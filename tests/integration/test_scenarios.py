"""End-to-end versions of the six agent scenarios required by the project directive.

Each scenario is driven through a real MCP client (in-process transport) exactly as an agent would, except E
which also uses the GUI controller (offscreen) for the human-refinement step.
"""
import json
import os
from pathlib import Path

import anyio
import numpy as np
import pytest

from lazyroom.core.io import load_image, save_image
from tests.conftest import synthetic_image

pytestmark = pytest.mark.timeout(600)
PHOTOS = Path(__file__).resolve().parents[1] / "data" / "photos"


def data(res):
    assert not res.is_error, res.content[0].text
    return json.loads(res.content[0].text)


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("LAZYROOM_HOME", str(tmp_path / "home"))
    from lazyroom.api import LazyRoom
    from lazyroom.mcp.server import build_server

    return build_server(LazyRoom())


def run_with_client(server, fn):
    from mcp import Client

    async def main():
        async with Client(server, read_timeout_seconds=300) as c:
            return await fn(c)

    return anyio.run(main)


def test_scenario_a_natural_language_vintage_grade(server, tmp_path):
    src = PHOTOS / "coffee.png"

    async def flow(c):
        caps = data(await c.call_tool("get_capabilities", {}))
        assert caps["operations"]["total"] >= 550 and "adjust" in caps["operations"]["categories"]
        insp = data(await c.call_tool("inspect_image", {"path": str(src)}))
        assert "descriptors" in insp["analysis"]
        pid = data(await c.call_tool("open_image", {"path": str(src)}))["project_id"]
        request = ("Give this a warm, brownish, retro film look with muted greens, soft contrast, faded "
                   "shadows, subtle grain, and a nostalgic 1970s feeling.")
        plan = data(await c.call_tool("plan_edit", {"request": request, "project_id": pid, "apply": True}))
        assert plan["interpretation"] and plan["edit"]["step_count"] >= 1
        pv = await c.call_tool("render_preview", {"project_id": pid, "max_size": 384, "compare": "side_by_side"})
        assert any(b.type == "image" for b in pv.content)
        stats_before = data(await c.call_tool("render_preview", {"project_id": pid, "version": "original",
                                                                 "include_image": False}))["stats"]
        stats_after = json.loads(pv.content[0].text)["stats"]
        # warmer: red/blue ratio increases; faded: shadows lifted
        rb = lambda s: s["mean_rgb"][0] / max(s["mean_rgb"][2], 1e-3)
        assert rb(stats_after) > rb(stats_before)
        assert stats_after["lightness"]["p05"] > stats_before["lightness"]["p05"]
        # refine: agent tweaks the grain amount inside the look group
        state = data(await c.call_tool("get_project_state", {"project_id": pid}))
        grain = [s for g in state["steps"] for s in g.get("steps", [g]) if s["op"] == "effect.grain"][0]
        data(await c.call_tool("edit_project", {"project_id": pid, "actions": [
            {"type": "update", "step_id": grain["id"], "params": {"amount": 12}}]}))
        out = tmp_path / "vintage.jpg"
        data(await c.call_tool("export_image", {"project_id": pid, "output": str(out)}))
        preset = data(await c.call_tool("save_preset", {"name": "My 70s", "project_id": pid,
                                                        "tags": ["vintage"]}))
        hist = data(await c.call_tool("project_history", {"project_id": pid}))
        saved = data(await c.call_tool("save_project", {"project_id": pid, "path": str(tmp_path / "v.lazyroom")}))
        return out, preset, hist, saved

    out, preset, hist, saved = run_with_client(server, flow)
    assert out.exists() and Path(preset["path"]).exists() and Path(saved["path"]).exists()
    assert len(hist["entries"]) >= 3 and any(e["actor"] == "agent" for e in hist["entries"])


def test_scenario_b_reference_style_transfer(server, tmp_path):
    from lazyroom.core.context import RenderContext
    from lazyroom.core.pipeline import render, steps_from_list
    from lazyroom.presets.store import PresetStore

    ref_img, _ = load_image(PHOTOS / "chelsea.png")
    look = [s for s in PresetStore().get("teal-orange")["steps"]]
    graded, _ = render(ref_img, steps_from_list(look), RenderContext())
    ref = tmp_path / "reference.png"
    save_image(np.clip(graded, 0, 1), ref)

    async def flow(c):
        pid = data(await c.call_tool("open_image", {"path": str(PHOTOS / "astronaut.png")}))["project_id"]
        m = data(await c.call_tool("match_style", {"project_id": pid, "reference": str(ref)}))
        assert m["report"]["similarity"] >= 0 and "limitations" in m["report"]
        assert {s["op"] for s in m["steps"]} >= {"adjust.white_balance", "adjust.curves"}
        state = data(await c.call_tool("get_project_state", {"project_id": pid}))
        assert state["steps"][-1]["origin"] == "style_match"
        cmp = await c.call_tool("compare_versions", {"project_id": pid})
        assert any(b.type == "image" for b in cmp.content)
        out = tmp_path / "matched.jpg"
        data(await c.call_tool("export_image", {"project_id": pid, "output": str(out)}))
        return m, out

    m, out = run_with_client(server, flow)
    assert out.exists()
    # reproducibility: the report's steps re-render to the same pixels
    a, _ = load_image(PHOTOS / "astronaut.png")
    r1, _ = render(a, steps_from_list(m["steps"]), RenderContext(seed=3))
    r2, _ = render(a, steps_from_list(m["steps"]), RenderContext(seed=3))
    assert np.allclose(r1, r2)


def test_scenario_c_consistent_batch(server, tmp_path):
    folder = tmp_path / "shoot"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(24):  # differently exposed / tinted shots of a scene
        img = synthetic_image(160, 120, seed=i) * rng.uniform(0.6, 1.3) + rng.uniform(-0.05, 0.05, 3)
        save_image(np.clip(img, 0, 1).astype(np.float32), folder / f"img_{i:03d}.jpg")
    out = tmp_path / "graded"

    async def flow(c):
        job = data(await c.call_tool("start_batch", {"inputs": [str(folder)], "output_dir": str(out),
                                                     "preset": "earthy-muted", "normalize": "both",
                                                     "naming": "{stem}-graded{ext}"}))
        st = data(await c.call_tool("job_status", {"job_id": job["job_id"], "wait_seconds": 240,
                                                   "include_items": True}))
        return st

    st = run_with_client(server, flow)
    assert st["status"] == "completed" and st["succeeded"] == 24
    assert Path(st["report_path"]).exists()
    report = json.loads(Path(st["report_path"]).read_text())
    assert all("per_image_steps" in it["info"] for it in report["items"])
    # consistency: graded medians are closer together than the originals
    med = lambda p: float(np.median(load_image(p)[0][..., :3]))
    before = [med(p) for p in sorted(folder.glob("*.jpg"))]
    after = [med(p) for p in sorted(out.glob("*-graded.jpg"))]
    assert np.std(after) < np.std(before)


def test_scenario_d_headless_cli(tmp_path):
    import subprocess
    import sys

    env = dict(os.environ, LAZYROOM_HOME=str(tmp_path / "home"))
    run = lambda *a: json.loads(subprocess.run([sys.executable, "-m", "lazyroom.cli.main", "--json", *a],
                                               capture_output=True, text=True, env=env, timeout=240).stdout)
    assert run("info")["result"]["operations"]["total"] >= 550
    desc = run("ops", "describe", "adjust.exposure")["result"]["operations"][0]
    assert desc["params"][0]["name"] == "ev"
    proj = tmp_path / "e.lazyroom"
    assert run("project", "create", str(PHOTOS / "rocket.jpg"), "--save", str(proj), "--op", "adjust.exposure",
               "ev=0.5")["ok"]
    pv = run("project", "preview", str(proj), "--out", str(tmp_path / "p.jpg"), "--size", "256")["result"]
    assert Path(pv["artifact"]["copied_to"]).exists()
    ex = run("project", "export", str(proj), str(tmp_path / "final.webp"))["result"]
    assert Path(ex["output"]).exists() and ex["format"] == "webp"


def test_scenario_e_human_refines_agent_edit(server, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    proj = tmp_path / "shared.lazyroom"

    async def agent(c):
        pid = data(await c.call_tool("open_image", {"path": str(PHOTOS / "chelsea.png")}))["project_id"]
        data(await c.call_tool("apply_preset", {"project_id": pid, "preset_id": "warm-editorial-portrait"}))
        data(await c.call_tool("save_project", {"project_id": pid, "path": str(proj)}))

    run_with_client(server, agent)
    # Human opens the same project in the GUI, inspects and overrides an agent-generated adjustment.
    from PySide6.QtGui import QGuiApplication

    from lazyroom.api import LazyRoom
    from lazyroom.gui.controller import Controller, PreviewProvider

    app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841
    ctl = Controller(LazyRoom(), PreviewProvider())
    ctl.openFile(str(proj))
    rows = ctl.stepsModel.items()
    assert any(r["origin"] != "user" for r in rows), "agent steps must be visible as such"
    wb = next(r for r in rows if r["op"] == "adjust.white_balance")
    ctl.selectStep(wb["id"])
    ctl.setParam("temperature", -10)            # human override
    ctl.toggleStep(next(r for r in rows if r["op"] == "effect.diffusion")["id"])  # human disables a step
    ctl.saveProject("")
    from lazyroom.core.project import Project

    p = Project.load(proj)
    hist = [e.actor for e in p.history.entries]
    assert "agent" in hist and "user" in hist, "history keeps both agent and human edits"
    flat = [s for g in p.steps for s in (g.steps or [g])]
    assert next(s for s in flat if s.op == "adjust.white_balance").params["temperature"] == -10
    assert not next(s for s in flat if s.op == "effect.diffusion").enabled


def test_scenario_f_failure_recovery(server, tmp_path):
    folder = tmp_path / "mixed"
    folder.mkdir()
    save_image(synthetic_image(80, 60), folder / "ok1.png")
    save_image(synthetic_image(80, 60, seed=2), folder / "ok2.jpg")
    (folder / "corrupt.jpg").write_bytes(b"\xff\xd8\xff\xdbgarbage")
    (folder / "notes.txt.png").write_text("this is not an image")
    out = tmp_path / "out"

    async def flow(c):
        bad = await c.call_tool("start_batch", {"inputs": [str(folder)], "output_dir": str(out),
                                                "steps": [{"op": "adjust.exposure", "params": {"ev": 42}}]})
        assert bad.is_error and json.loads(bad.content[0].text)["error"]["code"] == "INVALID_PARAMS"
        typo = await c.call_tool("start_batch", {"inputs": [str(folder)], "output_dir": str(out),
                                                 "steps": [{"op": "adjust.exposur", "params": {}}]})
        err = json.loads(typo.content[0].text)["error"]
        assert err["code"] == "UNKNOWN_OPERATION" and "adjust.exposure" in err["details"]["suggestions"]
        job = data(await c.call_tool("start_batch", {"inputs": [str(folder)], "output_dir": str(out),
                                                     "steps": [{"op": "filter.sepia", "params": {}}]}))
        st = data(await c.call_tool("job_status", {"job_id": job["job_id"], "wait_seconds": 120,
                                                   "include_items": True}))
        retry = data(await c.call_tool("start_batch", {"inputs": [str(folder)], "output_dir": str(out),
                                                       "steps": [{"op": "filter.sepia", "params": {}}],
                                                       "skip_existing": True}))
        st2 = data(await c.call_tool("job_status", {"job_id": retry["job_id"], "wait_seconds": 120}))
        return st, st2

    st, st2 = run_with_client(server, flow)
    assert st["status"] == "partial" and st["succeeded"] == 2 and st["failed"] == 2
    codes = {Path(i["input"]).name: i["error"]["code"] for i in st["items"] if i["status"] == "failed"}
    assert set(codes) == {"corrupt.jpg", "notes.txt.png"}
    assert all(c in ("DECODE_FAILED", "UNSUPPORTED_FORMAT") for c in codes.values())
    assert st2["skipped"] == 2  # safe recovery: valid outputs preserved, not recomputed
