"""CLI integration: JSON output, exit codes, project round trip, batch with partial failure."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lazyroom.core.io import save_image
from tests.conftest import synthetic_image

pytestmark = pytest.mark.timeout(300)


def run(args, tmp_path, check_json=True):
    env = dict(os.environ, LAZYROOM_HOME=str(tmp_path / "home"))
    r = subprocess.run([sys.executable, "-m", "lazyroom.cli.main", "--json", *args], capture_output=True,
                       text=True, env=env, timeout=240)
    data = json.loads(r.stdout) if check_json and r.stdout.strip() else None
    return r.returncode, data, r.stderr


def test_cli_discovery_and_apply(tmp_path):
    code, d, _ = run(["ops", "list", "--category", "adjust"], tmp_path)
    assert code == 0 and d["ok"] and d["result"]["total"] >= 20
    code, d, _ = run(["ops", "describe", "nope.op"], tmp_path)
    assert code == 2 and d["error"]["code"] == "UNKNOWN_OPERATION"
    src = tmp_path / "a.png"
    save_image(synthetic_image(200, 150), src)
    out = tmp_path / "a_out.webp"
    code, d, err = run(["apply", str(src), str(out), "--op", "adjust.exposure", "ev=0.3",
                        "--op", "adjust.curves", "rgb=[[0,0.05],[1,0.95]]"], tmp_path)
    assert code == 0, err
    assert out.exists() and d["result"]["width"] == 200
    code, d, _ = run(["apply", str(src), str(out)], tmp_path)
    assert code == 5 and d["error"]["code"] == "OUTPUT_EXISTS"
    code, d, _ = run(["apply", str(tmp_path / "missing.jpg"), str(tmp_path / "x.jpg")], tmp_path)
    assert code == 3 and d["error"]["code"] == "FILE_NOT_FOUND"


def test_cli_project_roundtrip(tmp_path):
    src = tmp_path / "b.jpg"
    save_image(synthetic_image(240, 160), src)
    proj = tmp_path / "b.lazyroom"
    code, d, err = run(["project", "create", str(src), "--save", str(proj), "--preset", "faded-analog"], tmp_path)
    assert code == 0, err
    code, d, _ = run(["project", "edit", str(proj), "--op", "effect.vignette", "amount=-40"], tmp_path)
    assert code == 0 and d["result"]["step_count"] == 2
    code, d, _ = run(["project", "undo", str(proj)], tmp_path)
    assert code == 0 and len(d["result"]["steps"]) == 1
    code, d, _ = run(["project", "preview", str(proj), "--out", str(tmp_path / "pv.jpg"), "--size", "128"], tmp_path)
    assert code == 0 and (tmp_path / "pv.jpg").exists()
    code, d, _ = run(["project", "export", str(proj), str(tmp_path / "b_out.jpg")], tmp_path)
    assert code == 0 and (tmp_path / "b_out.jpg").exists()
    state = json.loads(proj.read_text())
    assert state["schema"] == "lazyroom.project/1" and len(state["steps"]) == 1


def test_cli_batch_partial_failure(tmp_path):
    d_in = tmp_path / "in"
    d_in.mkdir()
    for i in range(3):
        save_image(synthetic_image(120, 90, seed=i), d_in / f"img{i}.jpg")
    (d_in / "corrupt.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    out = tmp_path / "out"
    code, d, err = run(["batch", str(d_in), "--output-dir", str(out), "--preset", "cool-urban"], tmp_path)
    assert code == 1, err  # partial failure exit code
    r = d["result"]
    assert r["status"] == "partial" and r["succeeded"] == 3 and r["failed"] == 1
    failed = [i for i in r["items"] if i["status"] == "failed"][0]
    assert failed["input"].endswith("corrupt.jpg") and failed["error"]["code"] in ("DECODE_FAILED", "UNSUPPORTED_FORMAT")
    assert Path(r["report_path"]).exists()
    # resume: re-run skipping existing outputs
    code, d, _ = run(["batch", str(d_in), "--output-dir", str(out), "--preset", "cool-urban", "--skip-existing"], tmp_path)
    assert d["result"]["skipped"] == 3


def test_cli_call_and_tools(tmp_path):
    code, d, _ = run(["call", "presets.list", '{"query": "film"}'], tmp_path)
    assert code == 0 and any(p["id"] == "warm-70s-film" for p in d["result"]["presets"])
    code, d, _ = run(["tools", "run", "color_info", "color=#FF8800"], tmp_path)
    assert code == 0 and d["result"]["rgb"] == [255, 136, 0]
    code, d, _ = run(["intent", "make it cooler and moodier with a vignette"], tmp_path)
    assert code == 0 and any(a.get("op") == "effect.vignette" for a in d["result"]["actions"])
