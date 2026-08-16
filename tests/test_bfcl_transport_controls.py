from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

from r0b0bench.endpoint import Endpoint
from r0b0bench.lanes.bfcl import _env


def load_bfcl_run_module():
    path = Path(__file__).parents[1] / "scripts/bfcl/bfcl_run.py"
    spec = importlib.util.spec_from_file_location("r0b0bench_bfcl_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bfcl_transport_applies_muse_generation_controls(monkeypatch) -> None:
    module = load_bfcl_run_module()
    monkeypatch.setenv("BFCL_MAX_TOKENS", "8192")
    monkeypatch.setenv("R0B0BENCH_REASONING_STRENGTH", "low")
    monkeypatch.setenv(
        "R0B0BENCH_CHAT_TEMPLATE_KWARGS", '{"enable_thinking":false}'
    )

    handler = module.R0b0OpenAICompletionsHandler.__new__(
        module.R0b0OpenAICompletionsHandler
    )
    handler.model_name = "muse"
    handler.temperature = 0.001
    handler.generate_with_backoff = lambda **kwargs: kwargs
    inference_data = {
        "message": [{"role": "user", "content": "call the tool"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }

    kwargs = handler._query_FC(inference_data)

    assert kwargs["model"] == "muse"
    assert kwargs["max_tokens"] == 8192
    assert kwargs["extra_body"] == {
        "chat_template_kwargs": {
            "enable_thinking": False,
            "reasoning_strength": "low",
        }
    }
    assert kwargs["tools"] == inference_data["tools"]
    assert "inference_input_log" in inference_data


def test_bfcl_ast_registers_shared_transport(monkeypatch) -> None:
    script_dir = Path(__file__).parents[1] / "scripts/bfcl"
    monkeypatch.syspath_prepend(str(script_dir))
    bfcl_run = importlib.import_module("bfcl_run")
    bfcl_ast_run = importlib.import_module("bfcl_ast_run")

    bfcl_ast_run.register_model()
    config = bfcl_ast_run.MODEL_CONFIG_MAPPING[bfcl_ast_run.REGISTRY]

    assert config.model_handler is bfcl_run.R0b0OpenAICompletionsHandler


def test_bfcl_safe_defaults_bound_concurrency_and_retries(monkeypatch, tmp_path) -> None:
    for name in ("BFCL_NUM_THREADS", "BFCL_HTTP_TIMEOUT", "BFCL_MAX_RETRIES"):
        monkeypatch.delenv(name, raising=False)

    endpoint = Endpoint("http://127.0.0.1:8000/v1", "qwen38-27b")
    try:
        env = _env(endpoint, tmp_path)
    finally:
        endpoint.close()

    assert env["BFCL_NUM_THREADS"] == "4"
    assert env["BFCL_HTTP_TIMEOUT"] == "600"
    assert env["BFCL_MAX_RETRIES"] == "1"
