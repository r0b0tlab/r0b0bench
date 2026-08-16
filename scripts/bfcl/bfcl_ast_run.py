#!/usr/bin/env python3
"""Run official BFCL v4 AST categories against an OpenAI-compatible endpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

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
from bfcl_run import R0b0OpenAICompletionsHandler

CATEGORIES = ("multiple", "parallel", "parallel_multiple")
EXPECTED_PER_CATEGORY = 200
REGISTRY = os.environ.get("R0B0BENCH_BFCL_MODEL_REGISTRY", "r0b0bench-openai-FC")
MODEL_NAME = os.environ.get("R0B0BENCH_SERVED_MODEL", "openai-compatible-model")


def _patch_timeout() -> None:
    import httpx

    timeout_s = float(os.environ.get("BFCL_HTTP_TIMEOUT", "3600"))

    def build(self):  # type: ignore[no-untyped-def]
        kwargs: dict[str, object] = {"timeout": httpx.Timeout(timeout_s, connect=60.0)}
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


def result_file(category: str) -> Path:
    return RESULT_PATH / REGISTRY.replace("/", "_") / get_directory_structure_by_category(category) / get_file_name_by_category(category, is_result_file=True)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ids_for(category: str) -> list[str]:
    rows = load_dataset_entry(category)
    if len(rows) != EXPECTED_PER_CATEGORY:
        raise RuntimeError(f"{category} count drift: {len(rows)} != {EXPECTED_PER_CATEGORY}")
    return [row["id"] for row in rows]


def good_rows(category: str) -> dict[str, dict]:
    return {
        str(row["id"]): row
        for row in read_jsonl(result_file(category))
        if row.get("id") and not (isinstance(row.get("result"), str) and row["result"].startswith("Error during inference:"))
    }


def strip_errors(category: str) -> dict:
    path = result_file(category)
    rows = read_jsonl(path)
    good = [row for row in rows if not (isinstance(row.get("result"), str) and row["result"].startswith("Error during inference:"))]
    if path.exists():
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in good), encoding="utf-8")
    return {"category": category, "kept": len(good), "stripped": len(rows) - len(good), "result_file": str(path)}


def validate_category(category: str, expected: int) -> dict:
    rows = read_jsonl(result_file(category))
    ids = [str(row.get("id")) for row in rows]
    errors = [row for row in rows if isinstance(row.get("result"), str) and row["result"].startswith("Error during inference:")]
    duplicates = sorted({row_id for row_id in ids if ids.count(row_id) > 1})
    report = {"category": category, "result_file": str(result_file(category)), "rows": len(rows), "expected": expected, "unique_ids": len(set(ids)), "duplicate_ids": duplicates, "inference_errors": len(errors)}
    if len(rows) != expected or len(set(ids)) != expected or duplicates or errors:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def generation_args(categories: list[str], *, run_ids: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model=[REGISTRY], test_category=categories, temperature=0.001,
        include_input_log=True, exclude_state_log=False, num_gpus=1,
        num_threads=int(os.environ.get("BFCL_NUM_THREADS", "12")),
        gpu_memory_utilization=0.9, backend="vllm", skip_server_setup=True,
        local_model_path=None, result_dir=None, allow_overwrite=False,
        run_ids=run_ids,
    )


def write_ids(mapping: dict[str, list[str]]) -> None:
    TEST_IDS_TO_GENERATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_IDS_TO_GENERATE_PATH.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inspect", "canary", "resume", "evaluate", "strip-errors", "status"])
    args = parser.parse_args()
    root = Path(os.environ.get("BFCL_PROJECT_ROOT", ""))
    if not root.is_absolute():
        raise SystemExit("BFCL_PROJECT_ROOT must be an absolute private run root")
    root.mkdir(parents=True, exist_ok=True)
    register_model()
    inventory = {category: ids_for(category) for category in CATEGORIES}
    if args.mode == "inspect":
        print(json.dumps({"categories": list(CATEGORIES), "rows_per_category": EXPECTED_PER_CATEGORY, "model": MODEL_NAME, "base_url": os.environ["OPENAI_BASE_URL"], "score_root": str(SCORE_PATH / REGISTRY.replace("/", "_"))}, indent=2))
        return 0
    if args.mode == "status":
        print(json.dumps({category: {"rows": len(read_jsonl(result_file(category))), "missing": len(set(inventory[category]) - set(good_rows(category)))} for category in CATEGORIES}, indent=2))
        return 0
    if args.mode == "strip-errors":
        print(json.dumps([strip_errors(category) for category in CATEGORIES], indent=2))
        return 0
    if args.mode == "canary":
        mapping = {category: ids[:8] for category, ids in inventory.items()}
        write_ids(mapping)
        generation_main(generation_args(list(CATEGORIES), run_ids=True))
        reports = {category: validate_category(category, len(ids)) for category, ids in mapping.items()}
        print(json.dumps({"canary": reports}, indent=2))
        return 0
    if args.mode == "resume":
        mapping = {category: [row_id for row_id in ids if row_id not in good_rows(category)] for category, ids in inventory.items()}
        mapping = {category: ids for category, ids in mapping.items() if ids}
        if mapping:
            write_ids(mapping)
            generation_main(generation_args(list(mapping), run_ids=True))
    elif args.mode == "evaluate":
        pass
    else:
        raise AssertionError(args.mode)
    reports = [validate_category(category, EXPECTED_PER_CATEGORY) for category in CATEGORIES]
    evaluation_main([REGISTRY], list(CATEGORIES), None, None, False)
    (root / f"{args.mode}-validation.json").write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"validated": reports, "score_root": str(SCORE_PATH / REGISTRY.replace("/", "_"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
