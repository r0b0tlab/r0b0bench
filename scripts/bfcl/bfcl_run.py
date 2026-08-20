#!/usr/bin/env python3
"""Run official BFCL v4 multi_turn_base against an OpenAI-compatible endpoint.

The served model name, endpoint, and private BFCL project root come only from
runtime environment variables. The evaluator itself is the unmodified official
bfcl-eval package.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:30000/v1")
os.environ.setdefault("BFCL_NUM_THREADS", "4")
os.environ.setdefault("BFCL_HTTP_TIMEOUT", "600")
os.environ.setdefault("BFCL_MAX_RETRIES", "1")
os.environ.setdefault("BFCL_MAX_TOKENS", "8192")
os.environ.setdefault("R0B0BENCH_REASONING_STRENGTH", "low")

from bfcl_eval._llm_response_generation import main as generation_main  # type: ignore[import-not-found]
from bfcl_eval.constants.eval_config import RESULT_PATH, SCORE_PATH, TEST_IDS_TO_GENERATE_PATH  # type: ignore[import-not-found]
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig  # type: ignore[import-not-found]
from bfcl_eval.eval_checker.eval_runner import main as evaluation_main  # type: ignore[import-not-found]
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler  # type: ignore[import-not-found]
from bfcl_eval.utils import get_directory_structure_by_category, get_file_name_by_category, load_dataset_entry  # type: ignore[import-not-found]

CATEGORY = "multi_turn_base"
EXPECTED_ROWS = 200
REGISTRY = os.environ.get("R0B0BENCH_BFCL_MODEL_REGISTRY", "r0b0bench-openai-FC")
MODEL_NAME = os.environ.get("R0B0BENCH_SERVED_MODEL", "openai-compatible-model")
_TIMING_LOCK = threading.Lock()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _record_timing(row: dict[str, object]) -> None:
    path = os.environ.get("R0B0BENCH_BFCL_TIMING_PATH")
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _TIMING_LOCK:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


def _response_timing(response: Any) -> dict[str, Any]:
    usage = _field(response, "usage", {}) or {}
    choices = _field(response, "choices", []) or []
    choice = choices[0] if choices else {}
    details = _field(usage, "completion_tokens_details", {}) or {}
    tool_calls = _field(_field(choice, "message", {}) or {}, "tool_calls", []) or []
    return {
        "http_status": 200,
        "prompt_tokens": int(_field(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(_field(usage, "completion_tokens", 0) or 0),
        "reasoning_tokens": int(_field(details, "reasoning_tokens", 0) or 0),
        "finish_reason": _field(choice, "finish_reason"),
        "tool_calls": len(tool_calls),
    }


class R0b0OpenAICompletionsHandler(OpenAICompletionsHandler):
    """Official BFCL OpenAI transport with explicit generation controls."""

    def _query_FC(self, inference_data: dict):  # type: ignore[no-untyped-def]
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {
            "message": repr(message),
            "tools": tools,
        }
        template_kwargs: dict[str, object] = {}
        if raw_template_kwargs := os.environ.get("R0B0BENCH_CHAT_TEMPLATE_KWARGS"):
            parsed = json.loads(raw_template_kwargs)
            if not isinstance(parsed, dict):
                raise ValueError("R0B0BENCH_CHAT_TEMPLATE_KWARGS must decode to an object")
            template_kwargs.update(parsed)
        template_kwargs.setdefault(
            "reasoning_strength", os.environ["R0B0BENCH_REASONING_STRENGTH"]
        )
        kwargs = {
            "messages": message,
            "model": self.model_name,
            "temperature": self.temperature,
            "store": False,
            "max_tokens": int(os.environ["BFCL_MAX_TOKENS"]),
            "extra_body": {"chat_template_kwargs": template_kwargs},
        }
        if tools:
            kwargs["tools"] = tools
        started = time.perf_counter()
        try:
            raw_response = self.generate_with_backoff(**kwargs)
        except Exception as exc:
            _record_timing(
                {
                    "case_id": inference_data.get("id") or inference_data.get("test_id"),
                    "http_status": 0,
                    "elapsed_s": time.perf_counter() - started,
                    "error": type(exc).__name__,
                }
            )
            raise
        sdk_elapsed = None
        if isinstance(raw_response, tuple) and len(raw_response) == 2:
            response, sdk_elapsed = raw_response
        else:
            response = raw_response
        elapsed = time.perf_counter() - started
        timing = _response_timing(response)
        timing.update(
            {
                "case_id": inference_data.get("id") or inference_data.get("test_id"),
                "elapsed_s": elapsed,
                "sdk_elapsed_s": float(sdk_elapsed) if sdk_elapsed is not None else None,
                "e2e_output_tok_s": (
                    float(timing["completion_tokens"]) / elapsed
                    if int(timing["completion_tokens"]) and elapsed > 0
                    else 0.0
                ),
            }
        )
        _record_timing(timing)
        return response, (float(sdk_elapsed) if sdk_elapsed is not None else elapsed)


def _patch_timeout() -> None:
    import httpx

    timeout_s = float(os.environ.get("BFCL_HTTP_TIMEOUT", "3600"))

    def build(self):  # type: ignore[no-untyped-def]
        kwargs = {"timeout": httpx.Timeout(timeout_s, connect=60.0)}
        if key := os.getenv("OPENAI_API_KEY"):
            kwargs["api_key"] = key
        if base_url := os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = base_url
        if headers := os.getenv("OPENAI_DEFAULT_HEADERS"):
            kwargs["default_headers"] = json.loads(headers)
        kwargs["max_retries"] = int(os.environ.get("BFCL_MAX_RETRIES", "3"))
        return kwargs

    OpenAICompletionsHandler._build_client_kwargs = build  # type: ignore[method-assign]


def register_model() -> None:
    _patch_timeout()
    MODEL_CONFIG_MAPPING[REGISTRY] = ModelConfig(
        model_name=MODEL_NAME,
        display_name=os.environ.get("R0B0BENCH_BFCL_MODEL_DISPLAY", MODEL_NAME),
        url=os.environ.get("R0B0BENCH_BFCL_MODEL_URL", "local://openai-compatible"),
        org=os.environ.get("R0B0BENCH_BFCL_MODEL_ORG", "r0b0tlab"),
        license=os.environ.get("R0B0BENCH_BFCL_MODEL_LICENSE", "MIT"),
        model_handler=R0b0OpenAICompletionsHandler,  # type: ignore[arg-type]
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    )


def result_file() -> Path:
    return RESULT_PATH / REGISTRY.replace("/", "_") / get_directory_structure_by_category(CATEGORY) / get_file_name_by_category(CATEGORY, is_result_file=True)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dataset_ids() -> list[str]:
    rows = load_dataset_entry(CATEGORY)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"{CATEGORY} count drift: {len(rows)} != {EXPECTED_ROWS}")
    return [row["id"] for row in rows]


def good_rows() -> dict[str, dict]:
    return {
        row.get("id"): row
        for row in read_jsonl(result_file())
        if row.get("id") and not (isinstance(row.get("result"), str) and row["result"].startswith("Error during inference:"))
    }


def validate_rows(expected: int, ids: list[str] | None = None) -> dict:
    rows = read_jsonl(result_file())
    wanted = set(ids or dataset_ids())
    selected = [row for row in rows if row.get("id") in wanted]
    errors = [row for row in selected if isinstance(row.get("result"), str) and row["result"].startswith("Error during inference:")]
    row_ids = [row.get("id") for row in selected]
    duplicates = sorted({row_id for row_id in row_ids if row_ids.count(row_id) > 1})
    report = {"result_file": str(result_file()), "rows": len(selected), "expected": expected, "unique_ids": len(set(row_ids)), "duplicate_ids": duplicates, "inference_errors": len(errors)}
    if len(selected) != expected or len(set(row_ids)) != expected or duplicates or errors:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def generation_args(*, run_ids: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model=[REGISTRY], test_category=[CATEGORY], temperature=0.001,
        include_input_log=True, exclude_state_log=False, num_gpus=1,
        num_threads=int(os.environ.get("BFCL_NUM_THREADS", "4")),
        gpu_memory_utilization=0.9, backend="vllm", skip_server_setup=True,
        local_model_path=None, result_dir=None, allow_overwrite=False,
        run_ids=run_ids,
    )


def write_ids(ids: list[str]) -> None:
    TEST_IDS_TO_GENERATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_IDS_TO_GENERATE_PATH.write_text(json.dumps({CATEGORY: ids}, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inspect", "canary", "full", "resume", "evaluate", "strip-errors", "status"])
    args = parser.parse_args()
    root = Path(os.environ.get("BFCL_PROJECT_ROOT", ""))
    if not root.is_absolute():
        raise SystemExit("BFCL_PROJECT_ROOT must be an absolute private run root")
    root.mkdir(parents=True, exist_ok=True)
    register_model()
    ids = dataset_ids()
    if args.mode == "inspect":
        print(json.dumps({"category": CATEGORY, "rows": len(ids), "model": MODEL_NAME, "base_url": os.environ["OPENAI_BASE_URL"], "score_root": str(SCORE_PATH / REGISTRY.replace("/", "_"))}, indent=2))
        return 0
    if args.mode == "status":
        rows = read_jsonl(result_file())
        errors = sum(1 for row in rows if isinstance(row.get("result"), str) and row["result"].startswith("Error during inference:"))
        print(json.dumps({"rows": len(rows), "errors": errors, "missing": len(set(ids) - set(good_rows()))}, indent=2))
        return 0
    if args.mode == "strip-errors":
        rows = read_jsonl(result_file())
        good = [row for row in rows if not (isinstance(row.get("result"), str) and row["result"].startswith("Error during inference:"))]
        if result_file().exists():
            result_file().write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in good), encoding="utf-8")
        print(json.dumps({"kept": len(good), "stripped": len(rows) - len(good), "result_file": str(result_file())}, indent=2))
        return 0
    if args.mode == "canary":
        canary = ids[:3]
        write_ids(canary)
        generation_main(generation_args(run_ids=True))
        print(json.dumps({"canary": validate_rows(len(canary), canary)}, indent=2))
        return 0
    if args.mode == "full":
        generation_main(generation_args(run_ids=False))
    elif args.mode == "resume":
        missing = [row_id for row_id in ids if row_id not in good_rows()]
        if missing:
            write_ids(missing)
            generation_main(generation_args(run_ids=True))
    elif args.mode == "evaluate":
        pass
    else:
        raise AssertionError(args.mode)
    report = validate_rows(EXPECTED_ROWS)
    evaluation_main([REGISTRY], [CATEGORY], None, None, False)
    (root / f"{args.mode}-validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"validated": report, "score_root": str(SCORE_PATH / REGISTRY.replace("/", "_"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
