"""Runtime configuration (file: <config_dir>/config.json, env overrides)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.paths import config_dir


@dataclass
class LLMConfig:
    """Optional in-app language-model backend for natural-language editing.

    provider: "none" | "openai_compatible" | "anthropic".  ``openai_compatible``
    works with OpenAI, Ollama, LM Studio, vLLM, llama.cpp server, OpenRouter...
    The API key is read from the environment variable named in ``api_key_env``
    (never stored in the config file).
    """
    provider: str = "none"
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    timeout: float = 120.0
    send_preview_image: bool = False  # images leave the machine only if explicitly enabled


@dataclass
class Config:
    # Filesystem policy ------------------------------------------------------
    read_roots: list[str] = field(default_factory=list)   # empty = home + temp + cwd
    write_roots: list[str] = field(default_factory=list)  # empty = home + temp + cwd
    allow_overwrite: bool = True        # still requires overwrite=true on each call
    approval_policy: str = "explicit"   # explicit: destructive actions need confirm=true; "never": refuse them
    # Resources ----------------------------------------------------------------
    max_pixels: int = 250_000_000
    max_file_mb: int = 1024
    preview_size: int = 1280
    render_timeout_s: float = 600.0
    batch_workers: int = 0              # 0 = auto
    cache_mb: int = 768
    # Network ------------------------------------------------------------------
    http_host: str = "127.0.0.1"
    http_port: int = 8765
    # Optional LLM -------------------------------------------------------------
    llm: LLMConfig = field(default_factory=LLMConfig)
    # Misc -----------------------------------------------------------------------
    artifact_dir: str = ""

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        p = Path(path) if path else config_dir() / "config.json"
        data: dict[str, Any] = {}
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
        llm = LLMConfig(**{k: v for k, v in (data.pop("llm", {}) or {}).items() if k in LLMConfig.__dataclass_fields__})
        cfg = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__}, llm=llm)
        # environment overrides
        env = os.environ
        if env.get("LAZYROOM_READ_ROOTS"):
            cfg.read_roots = env["LAZYROOM_READ_ROOTS"].split(os.pathsep)
        if env.get("LAZYROOM_WRITE_ROOTS"):
            cfg.write_roots = env["LAZYROOM_WRITE_ROOTS"].split(os.pathsep)
        if env.get("LAZYROOM_APPROVAL"):
            cfg.approval_policy = env["LAZYROOM_APPROVAL"]
        if env.get("LAZYROOM_LLM_PROVIDER"):
            cfg.llm.provider = env["LAZYROOM_LLM_PROVIDER"]
            cfg.llm.base_url = env.get("LAZYROOM_LLM_BASE_URL", cfg.llm.base_url)
            cfg.llm.model = env.get("LAZYROOM_LLM_MODEL", cfg.llm.model)
            cfg.llm.api_key_env = env.get("LAZYROOM_LLM_API_KEY_ENV", cfg.llm.api_key_env)
        return cfg

    def save(self, path: str | os.PathLike | None = None) -> Path:
        p = Path(path) if path else config_dir() / "config.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
