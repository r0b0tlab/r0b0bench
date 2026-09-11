#!/usr/bin/env python3
"""Run the frozen BFCL v4 multi_turn_base structural-hard20 subset.

This is a deterministic structural-complexity proxy selected from official BFCL
metadata. It is not the official 200-case category score and is not described
as the semantically 20 hardest cases.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("BFCL_NUM_THREADS", "1")
os.environ.setdefault("BFCL_HTTP_TIMEOUT", "1200")
os.environ.setdefault("BFCL_MAX_RETRIES", "1")
os.environ.setdefault("BFCL_MAX_TOKENS", "8192")

from bfcl_eval._llm_response_generation import main as generation_main  # type: ignore[import-not-found]
from bfcl_eval.constants.eval_config import (  # type: ignore[import-not-found]
    POSSIBLE_ANSWER_PATH,
    PROMPT_PATH,
    RESULT_PATH,
    SCORE_PATH,
    TEST_IDS_TO_GENERATE_PATH,
)
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig  # type: ignore[import-not-found]
from bfcl_eval.eval_checker.eval_runner import main as evaluation_main  # type: ignore[import-not-found]
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker  # type: ignore[import-not-found]
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler  # type: ignore[import-not-found]
from bfcl_eval.utils import (  # type: ignore[import-not-found]
    get_directory_structure_by_category,
    get_file_name_by_category,
    load_dataset_entry,
)

CATEGORY = "multi_turn_base"
LABEL = "BFCL v4 multi_turn_base structural-hard20"
EXPECTED = 20
SOURCE_COUNT = 200
BFCL_VERSION = "2025.12.17"
DATASET_SHA256 = "1a21a995d06fd6f20ba55de7bced30ef953ec35e998f502ec2ecf4d66ef1c43a"
GROUND_TRUTH_SHA256 = "1fee67823b317571649177dd89d63969feaae4e810cc7448ee55ba797fb7c8fc"
MANIFEST_SHA256 = "0860da504a3db2c3cd73647ecdc2a5ecdb1793d7a5cf3f4f004912f0ef314d4e"
SELECTED_IDS_SHA256 = "28158003bca5abf7e6922a8aa460d0bed208b3636757349460000b85812b384f"
SELECTED_FEATURES_SHA256 = "65c7889c2f78f29bd71e2d74854a11234a20dbb02e83186ae7a4c9050b79cbca"
SELECTOR_POLICY_SHA256 = "03999d7f8e22669d5ec51a9920d0e79a6350005f95f7d305e5835e18479dc0f2"
REQUIREMENTS_SHA256 = "58d14dc53550f928cf0e014c6a4f03e796ad87a26b398a5e10ad2313b4ce1ff2"
REGISTRY = os.environ.get("Q200_BFCL_REGISTRY", "qwen38-flash-next-hard20-FC")
MODEL_NAME = os.environ.get("Q200_SERVED_MODEL", "qwen38-flash-next-w4a16")
REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "artifacts" / "bfcl-v4-multi-turn-hard20-v1.json"
REQUIREMENTS_PATH = REPO_ROOT / "requirements-bfcl.txt"
FRESHNESS_NAME = "bfcl-hard20-freshness.json"
TIMING_LOCK = threading.Lock()
TIMING_EVENTS: dict[str, list[dict[str, Any]]] = {}
REQUIRED_CHAT_KWARGS = {
    "enable_thinking": True,
    "thinking": True,
    "reasoning_effort": "low",
}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise RuntimeError(f"blank JSONL line {line_number}: {path}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise RuntimeError(f"non-object JSONL row {line_number}: {path}")
        rows.append(row)
    return rows


def numeric_case_id(row_id: str) -> int:
    match = re.fullmatch(r"multi_turn_base_(\d+)", row_id)
    if match is None:
        raise RuntimeError(f"unexpected BFCL ID: {row_id}")
    return int(match.group(1))


def structural_key(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        -len(row["question"]),
        -len(row["path"]),
        -len(row.get("involved_classes", [])),
        numeric_case_id(str(row["id"])),
    )


def feature_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "involved_class_count": len(row.get("involved_classes", [])),
        "required_path_length": len(row["path"]),
        "user_turn_count": len(row["question"]),
    }


def _structural_triplet(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        len(row["question"]),
        len(row["path"]),
        len(row.get("involved_classes", [])),
    )


def validate_contract() -> tuple[dict[str, Any], list[str]]:
    if importlib.metadata.version("bfcl-eval") != BFCL_VERSION:
        raise RuntimeError("bfcl-eval version drift")
    dataset_path = PROMPT_PATH / "BFCL_v4_multi_turn_base.json"
    truth_path = POSSIBLE_ANSWER_PATH / "BFCL_v4_multi_turn_base.json"
    if sha256_file(dataset_path) != DATASET_SHA256:
        raise RuntimeError("BFCL source dataset hash drift")
    if sha256_file(truth_path) != GROUND_TRUTH_SHA256:
        raise RuntimeError("BFCL ground-truth hash drift")
    if sha256_file(REQUIREMENTS_PATH) != REQUIREMENTS_SHA256:
        raise RuntimeError("BFCL dependency lock hash drift")
    raw_manifest = MANIFEST_PATH.read_bytes()
    if sha256_bytes(raw_manifest) != MANIFEST_SHA256:
        raise RuntimeError("hard-20 manifest hash drift")
    manifest = json.loads(raw_manifest)
    if manifest.get("label") != LABEL:
        raise RuntimeError("hard-20 label drift")
    rows = load_dataset_entry(CATEGORY)
    if len(rows) != SOURCE_COUNT:
        raise RuntimeError(f"BFCL source count drift: {len(rows)} != {SOURCE_COUNT}")
    expected_source_ids = {f"multi_turn_base_{index}" for index in range(SOURCE_COUNT)}
    source_ids = [str(row.get("id")) for row in rows]
    if len(set(source_ids)) != SOURCE_COUNT or set(source_ids) != expected_source_ids:
        raise RuntimeError("BFCL source ID domain drift")

    ranked = sorted(rows, key=structural_key)
    derived_rows = ranked[:EXPECTED]
    derived = [str(row["id"]) for row in derived_rows]
    selected_features = manifest.get("selected")
    if not isinstance(selected_features, list):
        raise RuntimeError("hard-20 selected feature list missing")
    selected = [str(entry.get("id")) for entry in selected_features]
    if selected != derived or len(set(selected)) != EXPECTED:
        raise RuntimeError("hard-20 selector does not reproduce the frozen manifest")
    independently_projected = [feature_projection(row) for row in derived_rows]
    if selected_features != independently_projected:
        raise RuntimeError("hard-20 manifest feature metadata differs from official source")

    policy = manifest.get("selection_policy")
    provenance = manifest.get("selection_provenance")
    if not isinstance(policy, dict) or not isinstance(provenance, dict):
        raise RuntimeError("hard-20 selection provenance is incomplete")
    hashes = {
        "canonical_selected_ids_sha256": sha256_bytes(canonical(selected)),
        "canonical_selected_feature_projection_sha256": sha256_bytes(canonical(selected_features)),
        "canonical_selector_policy_sha256": sha256_bytes(canonical(policy)),
    }
    expected_hashes = {
        "canonical_selected_ids_sha256": SELECTED_IDS_SHA256,
        "canonical_selected_feature_projection_sha256": SELECTED_FEATURES_SHA256,
        "canonical_selector_policy_sha256": SELECTOR_POLICY_SHA256,
    }
    if hashes != expected_hashes or provenance != expected_hashes:
        raise RuntimeError("hard-20 canonical selection hash drift")

    rank20, rank21 = ranked[EXPECTED - 1], ranked[EXPECTED]
    boundary_triplet = _structural_triplet(rank20)
    tied = [row for row in rows if _structural_triplet(row) == boundary_triplet]
    selected_tied = [row for row in derived_rows if _structural_triplet(row) == boundary_triplet]
    expected_boundary = {
        "rank_20_id": str(rank20["id"]),
        "rank_21_id": str(rank21["id"]),
        "boundary_structural_key": {
            "user_turn_count": boundary_triplet[0],
            "required_path_length": boundary_triplet[1],
            "involved_class_count": boundary_triplet[2],
        },
        "tied_source_case_count": len(tied),
        "selected_from_boundary_tie_count": len(selected_tied),
        "disclosure": "The rank-20 boundary is deterministic but arbitrary within a 20-way structural tie; numeric case ID ascending selects the first six tied cases.",
    }
    if manifest.get("selection_boundary") != expected_boundary:
        raise RuntimeError("hard-20 boundary/tie disclosure drift")
    return manifest, selected


def chat_kwargs() -> dict[str, Any]:
    raw = os.environ.get("Q200_CHAT_TEMPLATE_KWARGS")
    value = json.loads(raw) if raw else dict(REQUIRED_CHAT_KWARGS)
    if value != REQUIRED_CHAT_KWARGS:
        raise RuntimeError(f"BFCL hard-20 requires exact native-thinking kwargs: {REQUIRED_CHAT_KWARGS}")
    return value


def required_binding(name: str, pattern: str | None = None) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is required")
    if pattern is not None and re.fullmatch(pattern, value) is None:
        raise RuntimeError(f"{name} has invalid format")
    return value


def endpoint_contract() -> tuple[str, str]:
    base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    if not base.endswith("/v1"):
        raise RuntimeError("OPENAI_BASE_URL must end in /v1")
    request = urllib.request.Request(base + "/models", method="GET")
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read())
    names = {str(row.get("id")) for row in payload.get("data", []) if isinstance(row, dict)}
    if MODEL_NAME not in names:
        raise RuntimeError(f"served model {MODEL_NAME!r} absent from /v1/models")
    return base, MODEL_NAME


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def record_timing_event(case_id: Any, row: dict[str, Any]) -> None:
    if not isinstance(case_id, str) or numeric_case_id(case_id) < 0:
        raise RuntimeError(f"missing/invalid BFCL case ID in timing event: {case_id!r}")
    with TIMING_LOCK:
        TIMING_EVENTS.setdefault(case_id, []).append(row)


class Q200OpenAICompletionsHandler(OpenAICompletionsHandler):
    """Official BFCL handler with frozen native-thinking request controls."""

    def _pre_query_processing_FC(self, inference_data, test_entry):  # type: ignore[no-untyped-def]
        value = super()._pre_query_processing_FC(inference_data, test_entry)
        value["q200_case_id"] = test_entry["id"]
        return value

    def _query_FC(self, inference_data: dict):  # type: ignore[no-untyped-def]
        messages: list[dict[str, Any]] = inference_data["message"]
        tools = inference_data["tools"]
        case_id = inference_data.get("q200_case_id")
        inference_data["inference_input_log"] = {"message": repr(messages), "tools": tools}
        kwargs: dict[str, Any] = {
            "messages": messages,
            "model": self.model_name,
            "temperature": self.temperature,
            "store": False,
            "max_tokens": int(os.environ["BFCL_MAX_TOKENS"]),
            "extra_body": {"chat_template_kwargs": chat_kwargs()},
        }
        if tools:
            kwargs["tools"] = tools
        started = time.perf_counter()
        try:
            raw_response = self.generate_with_backoff(**kwargs)
        except Exception as exc:
            record_timing_event(
                case_id,
                {
                    "http_status": 0,
                    "elapsed_s": time.perf_counter() - started,
                    "error": type(exc).__name__,
                },
            )
            raise
        if isinstance(raw_response, tuple) and len(raw_response) == 2:
            response, sdk_elapsed = raw_response
        else:
            response, sdk_elapsed = raw_response, None
        elapsed = time.perf_counter() - started
        usage = _field(response, "usage", {}) or {}
        choices = _field(response, "choices", []) or []
        choice = choices[0] if choices else {}
        message = _field(choice, "message", {}) or {}
        completion_tokens = int(_field(usage, "completion_tokens", 0) or 0)
        record_timing_event(
            case_id,
            {
                "http_status": 200,
                "elapsed_s": elapsed,
                "sdk_elapsed_s": float(sdk_elapsed) if sdk_elapsed is not None else None,
                "prompt_tokens": int(_field(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": completion_tokens,
                "finish_reason": _field(choice, "finish_reason"),
                "tool_calls": len(_field(message, "tool_calls", []) or []),
                "e2e_output_tok_s": completion_tokens / elapsed if completion_tokens and elapsed > 0 else 0.0,
            },
        )
        return response, float(sdk_elapsed) if sdk_elapsed is not None else elapsed


def patch_timeout() -> None:
    import httpx

    timeout_s = float(os.environ["BFCL_HTTP_TIMEOUT"])

    def build(self):  # type: ignore[no-untyped-def]
        return {
            "timeout": httpx.Timeout(timeout_s, connect=60.0),
            "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
            "base_url": os.environ["OPENAI_BASE_URL"],
            "max_retries": int(os.environ["BFCL_MAX_RETRIES"]),
        }

    OpenAICompletionsHandler._build_client_kwargs = build  # type: ignore[method-assign]


def register_model() -> None:
    patch_timeout()
    MODEL_CONFIG_MAPPING[REGISTRY] = ModelConfig(
        model_name=MODEL_NAME,
        display_name="Qwen3.8 Flash-Next NVFP4 W4A16 SM121 (Q200-v2 BFCL structural-hard20)",
        url="https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-NVFP4-W4A16-sm121",
        org="r0b0tlab",
        license="other",
        model_handler=Q200OpenAICompletionsHandler,  # type: ignore[arg-type]
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    )


def result_file() -> Path:
    return RESULT_PATH / REGISTRY.replace("/", "_") / get_directory_structure_by_category(CATEGORY) / get_file_name_by_category(CATEGORY, is_result_file=True)


def score_file() -> Path:
    return SCORE_PATH / REGISTRY.replace("/", "_") / get_directory_structure_by_category(CATEGORY) / get_file_name_by_category(CATEGORY, is_score_file=True)


def timing_path() -> Path:
    raw = required_binding("Q200_BFCL_TIMING_PATH")
    target = Path(raw)
    if not target.is_absolute():
        raise RuntimeError("Q200_BFCL_TIMING_PATH must be absolute")
    return target


def validate_results(selected: list[str]) -> dict[str, Any]:
    rows = read_jsonl(result_file())
    ids = [row.get("id") for row in rows]
    errors = [
        row.get("id")
        for row in rows
        if "Error during inference:" in json.dumps(row, ensure_ascii=False)
    ]
    if len(rows) != EXPECTED or len(set(ids)) != EXPECTED or set(ids) != set(selected) or errors:
        raise RuntimeError(
            json.dumps(
                {
                    "rows": len(rows),
                    "unique_ids": len(set(ids)),
                    "ids_match": set(ids) == set(selected),
                    "inference_errors": errors,
                },
                sort_keys=True,
            )
        )
    return {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "ids": selected,
        "result_sha256": sha256_file(result_file()),
    }


def validate_score(selected: list[str]) -> dict[str, Any]:
    rows = read_jsonl(score_file())
    if not rows:
        raise RuntimeError("official BFCL score file is empty")
    header = rows[0]
    if header.get("total_count") != EXPECTED:
        raise RuntimeError(f"official BFCL score total drift: {header.get('total_count')} != {EXPECTED}")
    correct = header.get("correct_count")
    accuracy = header.get("accuracy")
    if not isinstance(correct, int) or isinstance(correct, bool) or not 0 <= correct <= EXPECTED:
        raise RuntimeError("official BFCL correct_count is invalid")
    if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
        raise RuntimeError("official BFCL accuracy is invalid")
    accuracy = float(accuracy)
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise RuntimeError("official BFCL accuracy is non-finite or outside [0,1]")
    expected_accuracy = correct / EXPECTED
    if not math.isclose(accuracy, expected_accuracy, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("official BFCL accuracy arithmetic mismatch")
    failures = rows[1:]
    incorrect = EXPECTED - correct
    failure_ids = [row.get("id") for row in failures]
    if len(failures) != incorrect:
        raise RuntimeError("official BFCL failure-body count mismatch")
    if len(set(failure_ids)) != len(failure_ids) or not set(failure_ids).issubset(set(selected)):
        raise RuntimeError("official BFCL failure-body ID mismatch")
    if any(row.get("valid") is not False for row in failures):
        raise RuntimeError("official BFCL failure body contains a non-failure row")
    return {
        "correct_count": correct,
        "incorrect_count": incorrect,
        "total_count": EXPECTED,
        "accuracy": accuracy,
        "failure_rows": len(failures),
        "score_sha256": sha256_file(score_file()),
    }


def generation_args() -> SimpleNamespace:
    return SimpleNamespace(
        model=[REGISTRY],
        test_category=[CATEGORY],
        temperature=0.001,
        include_input_log=True,
        exclude_state_log=False,
        num_gpus=1,
        num_threads=int(os.environ["BFCL_NUM_THREADS"]),
        gpu_memory_utilization=0.9,
        backend="vllm",
        skip_server_setup=True,
        local_model_path=None,
        result_dir=None,
        allow_overwrite=False,
        run_ids=True,
    )


def write_selected_ids(selected: list[str]) -> None:
    TEST_IDS_TO_GENERATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_IDS_TO_GENERATE_PATH.write_text(json.dumps({CATEGORY: selected}, indent=2) + "\n", encoding="utf-8")


def write_timing_summary(selected: list[str]) -> dict[str, Any]:
    with TIMING_LOCK:
        observed = set(TIMING_EVENTS)
        if observed != set(selected):
            raise RuntimeError(
                f"timing case coverage mismatch: missing={sorted(set(selected) - observed)} outside={sorted(observed - set(selected))}"
            )
        rows: list[dict[str, Any]] = []
        for case_id in selected:
            events = TIMING_EVENTS[case_id]
            errors = [event for event in events if event.get("error") or event.get("http_status") != 200]
            if not events or errors:
                raise RuntimeError(f"timing transport errors for {case_id}: {errors}")
            rows.append(
                {
                    "case_id": case_id,
                    "request_count": len(events),
                    "error_count": 0,
                    "all_http_200": True,
                    "elapsed_s": sum(float(event["elapsed_s"]) for event in events),
                    "prompt_tokens": sum(int(event.get("prompt_tokens", 0)) for event in events),
                    "completion_tokens": sum(int(event.get("completion_tokens", 0)) for event in events),
                    "tool_calls": sum(int(event.get("tool_calls", 0)) for event in events),
                    "finish_reasons": [event.get("finish_reason") for event in events],
                }
            )
    target = timing_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return validate_timing(selected)


def validate_timing(selected: list[str]) -> dict[str, Any]:
    target = timing_path()
    rows = read_jsonl(target)
    ids = [row.get("case_id") for row in rows]
    if len(rows) != EXPECTED or ids != selected or len(set(ids)) != EXPECTED:
        raise RuntimeError("BFCL timing sidecar is not exactly one ordered row per selected ID")
    for row in rows:
        if row.get("all_http_200") is not True or row.get("error_count") != 0:
            raise RuntimeError("BFCL timing sidecar contains transport errors")
        if not isinstance(row.get("request_count"), int) or row["request_count"] < 1:
            raise RuntimeError("BFCL timing request_count is invalid")
        for key in ("elapsed_s", "prompt_tokens", "completion_tokens", "tool_calls"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise RuntimeError(f"BFCL timing metric is invalid: {key}")
    return {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "error_count": 0,
        "timing_sha256": sha256_file(target),
    }


def _module_file_hash(value: Any) -> dict[str, str]:
    module = value if inspect.ismodule(value) else inspect.getmodule(value)
    if module is None:
        raise RuntimeError(f"cannot resolve module for {value!r}")
    path = Path(inspect.getfile(module)).resolve()
    return {"module": module.__name__, "path": str(path), "sha256": sha256_file(path)}


def package_tree_binding() -> dict[str, Any]:
    dist = importlib.metadata.distribution("bfcl-eval")
    records: list[dict[str, Any]] = []
    for entry in dist.files or []:
        relative = str(entry)
        if "__pycache__" in relative or relative.endswith((".pyc", ".pyo")):
            continue
        path = Path(str(dist.locate_file(entry))).resolve()
        if not path.is_file():
            continue
        records.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    records.sort(key=lambda row: row["path"])
    return {
        "distribution": "bfcl-eval",
        "version": BFCL_VERSION,
        "file_count": len(records),
        "manifest_sha256": sha256_bytes(canonical(records)),
    }


def runtime_binding() -> dict[str, Any]:
    settings = {
        "temperature": 0.001,
        "max_tokens": int(os.environ["BFCL_MAX_TOKENS"]),
        "http_timeout_s": float(os.environ["BFCL_HTTP_TIMEOUT"]),
        "max_retries": int(os.environ["BFCL_MAX_RETRIES"]),
        "num_threads": int(os.environ["BFCL_NUM_THREADS"]),
        "include_input_log": True,
        "partial_eval": True,
        "run_ids": True,
        "chat_template_kwargs": chat_kwargs(),
    }
    if settings["num_threads"] != 1:
        raise RuntimeError("claim-bearing BFCL hard-20 requires BFCL_NUM_THREADS=1")
    return {
        "adapter": _module_file_hash(Q200OpenAICompletionsHandler),
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "selector_source_sha256": sha256_bytes(inspect.getsource(structural_key).encode("utf-8")),
        "requirements_bfcl_sha256": sha256_file(REQUIREMENTS_PATH),
        "official_modules": {
            "generation": _module_file_hash(generation_main),
            "evaluation": _module_file_hash(evaluation_main),
            "multi_turn_checker": _module_file_hash(multi_turn_checker),
            "openai_handler": _module_file_hash(OpenAICompletionsHandler),
        },
        "package_tree": package_tree_binding(),
        "settings": settings,
    }


def root_files(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def freshness_path(root: Path) -> Path:
    return root / FRESHNESS_NAME


def start_fresh_run(root: Path, binding_sha256: str) -> dict[str, Any]:
    existing = root_files(root)
    if existing:
        raise RuntimeError(f"run mode requires a fresh BFCL_PROJECT_ROOT; found files: {existing}")
    timing = timing_path()
    if timing.exists():
        raise RuntimeError(f"run mode requires a fresh timing sidecar path: {timing}")
    witness = {
        "schema": "r0b0tlab.qwen38.bfcl_freshness.v1",
        "mode": "run",
        "started_utc": utc_now(),
        "project_root": str(root),
        "preexisting_files": [],
        "preexisting_timing": False,
        "runtime_binding_sha256": binding_sha256,
        "generation_complete": False,
    }
    freshness_path(root).write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return witness


def complete_generation_witness(root: Path, witness: dict[str, Any], result: Mapping[str, Any], timing: Mapping[str, Any]) -> dict[str, Any]:
    if score_file().exists():
        raise RuntimeError("stale BFCL score artifact existed before official evaluation")
    witness.update(
        {
            "generation_complete": True,
            "generation_completed_utc": utc_now(),
            "result_sha256": result["result_sha256"],
            "timing_sha256": timing["timing_sha256"],
        }
    )
    freshness_path(root).write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return witness


def validate_freshness(root: Path, result_sha256: str, timing_sha256: str, binding_sha256: str) -> dict[str, Any]:
    path = freshness_path(root)
    if not path.exists():
        raise RuntimeError("fresh-run witness is missing")
    witness = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "schema": "r0b0tlab.qwen38.bfcl_freshness.v1",
        "mode": "run",
        "preexisting_files": [],
        "preexisting_timing": False,
        "runtime_binding_sha256": binding_sha256,
        "generation_complete": True,
        "result_sha256": result_sha256,
        "timing_sha256": timing_sha256,
    }
    for key, expected in checks.items():
        if witness.get(key) != expected:
            raise RuntimeError(f"fresh-run witness mismatch: {key}")
    return {"freshness_sha256": sha256_file(path), "witness": witness}


def build_summary(root: Path, manifest: Mapping[str, Any], selected: list[str], binding: Mapping[str, Any]) -> dict[str, Any]:
    base_url, _ = endpoint_contract()
    image_id = required_binding("Q200_IMAGE_ID", r"sha256:[0-9a-f]{64}")
    profile_id = required_binding("Q200_PROFILE_ID", r"[0-9a-f]{64}")
    candidate_id = required_binding("Q200_CANDIDATE_ID")
    result = validate_results(selected)
    timing = validate_timing(selected)
    score = validate_score(selected)
    binding_sha = sha256_bytes(canonical(binding))
    freshness = validate_freshness(root, result["result_sha256"], timing["timing_sha256"], binding_sha)
    identity = {
        "model": MODEL_NAME,
        "base_url": base_url,
        "image_id": image_id,
        "profile_id": profile_id,
        "candidate_id": candidate_id,
        "chat_template_kwargs": chat_kwargs(),
        "bfcl_eval_version": BFCL_VERSION,
        "category": CATEGORY,
        "label": LABEL,
        "manifest_sha256": MANIFEST_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "ground_truth_sha256": GROUND_TRUTH_SHA256,
        "selected_ids_sha256": SELECTED_IDS_SHA256,
        "selected_features_sha256": SELECTED_FEATURES_SHA256,
        "selector_policy_sha256": SELECTOR_POLICY_SHA256,
        "runtime_binding_sha256": binding_sha,
    }
    return {
        "schema": "r0b0tlab.qwen38.q200_v2_bfcl_hard20_run.v1",
        "status": "SCORED",
        "family": "bfcl_hard20",
        "label": LABEL,
        "interpretation": manifest["interpretation"],
        "identity": identity,
        "identity_sha256": sha256_bytes(canonical(identity)),
        "selection_policy": manifest["selection_policy"],
        "selection_boundary": manifest["selection_boundary"],
        "runtime_binding": binding,
        "freshness": freshness,
        "results": result,
        "timing": timing,
        "score": score,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inspect", "run", "resume", "evaluate", "status"])
    args = parser.parse_args()
    root = Path(os.environ.get("BFCL_PROJECT_ROOT", ""))
    if not root.is_absolute():
        raise SystemExit("BFCL_PROJECT_ROOT must be an absolute private run root")
    root.mkdir(parents=True, exist_ok=True)
    manifest, selected = validate_contract()
    register_model()
    if args.mode == "inspect":
        print(
            json.dumps(
                {
                    "category": CATEGORY,
                    "label": LABEL,
                    "selected_ids": selected,
                    "manifest_sha256": MANIFEST_SHA256,
                    "selection_boundary": manifest["selection_boundary"],
                },
                indent=2,
            )
        )
        return 0
    if args.mode == "status":
        rows = read_jsonl(result_file())
        present = {row.get("id") for row in rows}
        print(
            json.dumps(
                {
                    "rows": len(rows),
                    "selected_present": len(set(selected) & present),
                    "missing": sorted(set(selected) - present),
                    "freshness_witness": freshness_path(root).exists(),
                },
                indent=2,
            )
        )
        return 0
    if args.mode == "resume":
        raise RuntimeError("resume is disabled for claim-bearing structural-hard20 runs; use a fresh root with run")

    endpoint_contract()
    required_binding("Q200_IMAGE_ID", r"sha256:[0-9a-f]{64}")
    required_binding("Q200_PROFILE_ID", r"[0-9a-f]{64}")
    required_binding("Q200_CANDIDATE_ID")
    chat_kwargs()
    binding = runtime_binding()
    binding_sha = sha256_bytes(canonical(binding))

    if args.mode == "run":
        witness = start_fresh_run(root, binding_sha)
        write_selected_ids(selected)
        generation_main(generation_args())
        result = validate_results(selected)
        timing = write_timing_summary(selected)
        complete_generation_witness(root, witness, result, timing)
    else:
        result = validate_results(selected)
        timing = validate_timing(selected)
        validate_freshness(root, result["result_sha256"], timing["timing_sha256"], binding_sha)
        if score_file().exists():
            raise RuntimeError("evaluate mode requires no pre-existing score artifact")

    evaluation_main([REGISTRY], [CATEGORY], None, None, True)
    summary = build_summary(root, manifest, selected, binding)
    output = root / "bfcl-hard20-summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
