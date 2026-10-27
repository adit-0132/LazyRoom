"""Natural-language planning: rule engine semantics and the provider-agnostic LLM path (stub server)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lazyroom.app.config import Config, LLMConfig
from lazyroom.intent.planner import plan


def ops_of(res):
    out = []
    for a in res["actions"]:
        out += [s["op"] for s in a.get("steps", [])] if a["type"] == "add_group" else [a["op"]]
    return out


def test_rules_vintage_request_maps_to_look_and_attributes():
    r = plan("Give this a warm, brownish, retro film look with muted greens, soft contrast, faded shadows, "
             "subtle grain, and a nostalgic 1970s feeling.")
    assert r["engine"] == "rules" and r["base_look"] == "warm-70s-film"
    ops = ops_of(r)
    for op in ("adjust.white_balance", "adjust.hsl", "adjust.fade", "effect.grain", "adjust.contrast"):
        assert op in ops
    hsl = [s for a in r["actions"] for s in a.get("steps", []) if s["op"] == "adjust.hsl"][0]
    assert hsl["params"]["green_saturation"] < 0


def test_rules_intensity_and_skin_protection():
    weak = plan("slightly cooler")
    strong = plan("much cooler")
    t = lambda r: [a for a in r["actions"] if a.get("op") == "adjust.white_balance"][0]["params"]["temperature"]
    assert t(strong) < t(weak) < 0
    r = plan("make the shadows warmer without changing skin tones")
    step = [a for a in r["actions"] if a.get("op") == "adjust.color_grading"][0]
    assert step["params"]["shadows_saturation"] > 0 and step["mask"]["shapes"][1]["type"] == "skin"


def test_rules_image_aware_tempering():
    bright = {"tone": {"median": 0.8}, "color": {"warmth": 0.0}}
    r = plan("brighter", analysis=bright)
    ev = [a for a in r["actions"] if a.get("op") == "adjust.exposure"][0]["params"]["ev"]
    assert 0 < ev < 0.4 and any("already bright" in n for n in r["notes"])


def test_rules_unknown_request_is_explained():
    r = plan("qwertz zxcv")
    assert r["actions"] == [] and r["notes"]


class _Handler(BaseHTTPRequestHandler):
    reply: dict = {}
    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append((self.path, self.headers.get("Authorization"), body))
        payload = json.dumps({"choices": [{"message": {"content": json.dumps(_Handler.reply)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def stub_llm(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    monkeypatch.setenv("STUB_KEY", "secret-token")
    cfg = Config()
    cfg.llm = LLMConfig(provider="openai_compatible", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                        model="stub-model", api_key_env="STUB_KEY")
    yield cfg
    srv.shutdown()


def test_llm_openai_compatible_path(stub_llm):
    _Handler.reply = {"actions": [
        {"type": "add", "op": "adjust.white_balance", "params_json": "{\"temperature\": 25}", "label": "Warm"},
        {"type": "add", "op": "effect.grain", "params_json": "{\"amount\": 15}", "label": "Grain"}],
        "explanation": "warm with grain"}
    r = plan("warm with grain", config=stub_llm)
    assert r["engine"] == "llm:openai_compatible:stub-model"
    assert ops_of(r) == ["adjust.white_balance", "effect.grain"]
    path, auth, body = _Handler.seen[-1]
    assert path == "/v1/chat/completions" and auth == "Bearer secret-token"
    assert body["response_format"] == {"type": "json_object"}
    assert "adjust.exposure" in body["messages"][0]["content"]  # operation reference was provided


def test_llm_invalid_output_falls_back_to_rules(stub_llm):
    _Handler.reply = {"actions": [{"type": "add", "op": "adjust.nope", "params_json": "{}", "label": "x"}],
                      "explanation": "bad"}
    r = plan("cooler with a vignette", config=stub_llm)
    assert r["engine"] == "rules" and "LLM planning failed" in r["notes"][0]
    assert "effect.vignette" in ops_of(r)
