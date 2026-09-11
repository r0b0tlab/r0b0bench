#!/usr/bin/env python3
"""Fail-closed merger for Q200-v2 text-180 + official BFCL hard-20."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

TEXT_SCHEMA = "r0b0tlab.qwen38.quality_text_180_run.v2"
BFCL_SCHEMA = "r0b0tlab.qwen38.q200_v2_bfcl_hard20_run.v1"
OUTPUT_SCHEMA = "r0b0tlab.qwen38.q200_v2_closeout.v1"
TEXT_DATASET_SHA256 = "74623ab9b075120cd6f7a93059cc16d8817a6039dd20118b8f0350279f8b1ed6"
BFCL_MANIFEST_SHA256 = "0860da504a3db2c3cd73647ecdc2a5ecdb1793d7a5cf3f4f004912f0ef314d4e"
BFCL_LABEL = "BFCL v4 multi_turn_base structural-hard20"
EXPECTED_CHAT = {"enable_thinking": True, "thinking": True, "reasoning_effort": "low"}


def load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-summary", type=Path, required=True)
    parser.add_argument("--bfcl-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text, text_sha = load(args.text_summary)
    bfcl, bfcl_sha = load(args.bfcl_summary)

    require(text.get("schema") == TEXT_SCHEMA, "text summary schema mismatch")
    require(text.get("status") == "SCORED", "text-180 is not fully scored")
    require(text.get("dataset_sha256") == TEXT_DATASET_SHA256, "text dataset hash mismatch")
    require(text.get("dataset_count") == 180 and text.get("rows") == 180, "text-180 count mismatch")
    require(text.get("transport_complete") is True and text.get("transport_count") == 180, "text transport incomplete")
    require(text.get("grade_complete") is True, "text grades incomplete")
    require(text.get("ungraded_count") == 0 and text.get("grader_error_count") == 0, "text grades contain ungraded/errors")
    require(text.get("correct_count", 0) + text.get("incorrect_count", 0) == 180, "text score total mismatch")
    require(text.get("chat_template_kwargs") == EXPECTED_CHAT, "text thinking profile mismatch")

    require(bfcl.get("schema") == BFCL_SCHEMA and bfcl.get("status") == "SCORED", "BFCL hard-20 is not fully scored")
    require(bfcl.get("label") == BFCL_LABEL, "BFCL subset label mismatch")
    identity_raw = bfcl.get("identity")
    require(isinstance(identity_raw, Mapping), "BFCL identity missing")
    assert isinstance(identity_raw, Mapping)
    identity = identity_raw
    require(bfcl.get("identity_sha256") == hashlib.sha256(canonical(identity)).hexdigest(), "BFCL identity hash mismatch")
    require(identity.get("manifest_sha256") == BFCL_MANIFEST_SHA256, "BFCL manifest hash mismatch")
    require(identity.get("chat_template_kwargs") == EXPECTED_CHAT, "BFCL thinking profile mismatch")
    for key in ("dataset_sha256", "ground_truth_sha256", "selected_ids_sha256", "selected_features_sha256", "selector_policy_sha256", "runtime_binding_sha256"):
        require(is_sha256(identity.get(key)), f"BFCL identity hash missing/invalid: {key}")

    binding_raw = bfcl.get("runtime_binding")
    require(isinstance(binding_raw, Mapping), "BFCL runtime binding missing")
    assert isinstance(binding_raw, Mapping)
    require(hashlib.sha256(canonical(binding_raw)).hexdigest() == identity.get("runtime_binding_sha256"), "BFCL runtime binding hash mismatch")
    require(is_sha256(binding_raw.get("wrapper_sha256")), "BFCL wrapper hash missing")
    require(is_sha256(binding_raw.get("selector_source_sha256")), "BFCL selector source hash missing")
    require(is_sha256(binding_raw.get("requirements_bfcl_sha256")), "BFCL requirements hash missing")
    official_modules = binding_raw.get("official_modules")
    require(isinstance(official_modules, Mapping), "BFCL official module bindings missing")
    assert isinstance(official_modules, Mapping)
    require(set(official_modules) == {"generation", "evaluation", "multi_turn_checker", "openai_handler"}, "BFCL official module binding set mismatch")
    for name, module_raw in official_modules.items():
        require(isinstance(module_raw, Mapping) and is_sha256(module_raw.get("sha256")), f"BFCL official module hash missing: {name}")
    settings = binding_raw.get("settings")
    require(isinstance(settings, Mapping), "BFCL runtime settings missing")
    assert isinstance(settings, Mapping)
    require(settings.get("num_threads") == 1 and settings.get("run_ids") is True and settings.get("partial_eval") is True, "BFCL execution settings mismatch")
    require(settings.get("chat_template_kwargs") == EXPECTED_CHAT, "BFCL runtime thinking settings mismatch")
    package_tree = binding_raw.get("package_tree")
    require(isinstance(package_tree, Mapping), "BFCL package tree binding missing")
    assert isinstance(package_tree, Mapping)
    require(package_tree.get("version") == "2025.12.17", "BFCL package version mismatch")
    require(isinstance(package_tree.get("file_count"), int) and package_tree.get("file_count", 0) > 0, "BFCL package manifest is empty")
    require(is_sha256(package_tree.get("manifest_sha256")), "BFCL package tree hash missing")

    results_raw = bfcl.get("results")
    require(isinstance(results_raw, Mapping), "BFCL result binding missing")
    assert isinstance(results_raw, Mapping)
    require(results_raw.get("rows") == 20 and results_raw.get("unique_ids") == 20, "BFCL result set mismatch")
    require(isinstance(results_raw.get("ids"), list) and len(results_raw.get("ids", [])) == len(set(results_raw.get("ids", []))) == 20, "BFCL result IDs mismatch")
    require(is_sha256(results_raw.get("result_sha256")), "BFCL result hash missing")

    timing_raw = bfcl.get("timing")
    require(isinstance(timing_raw, Mapping), "BFCL timing binding missing")
    assert isinstance(timing_raw, Mapping)
    require(timing_raw.get("rows") == 20 and timing_raw.get("unique_ids") == 20 and timing_raw.get("error_count") == 0, "BFCL timing coverage/error mismatch")
    require(is_sha256(timing_raw.get("timing_sha256")), "BFCL timing hash missing")

    freshness_raw = bfcl.get("freshness")
    require(isinstance(freshness_raw, Mapping) and is_sha256(freshness_raw.get("freshness_sha256")), "BFCL fresh-run witness missing")
    assert isinstance(freshness_raw, Mapping)
    witness = freshness_raw.get("witness")
    require(isinstance(witness, Mapping), "BFCL fresh-run witness body missing")
    assert isinstance(witness, Mapping)
    require(witness.get("mode") == "run" and witness.get("preexisting_files") == [] and witness.get("preexisting_timing") is False and witness.get("generation_complete") is True, "BFCL fresh-run witness is invalid")
    require(witness.get("result_sha256") == results_raw.get("result_sha256") and witness.get("timing_sha256") == timing_raw.get("timing_sha256"), "BFCL fresh-run artifact hashes mismatch")
    require(witness.get("runtime_binding_sha256") == identity.get("runtime_binding_sha256"), "BFCL fresh-run runtime binding mismatch")
    score_raw = bfcl.get("score")
    require(isinstance(score_raw, Mapping), "BFCL score missing")
    assert isinstance(score_raw, Mapping)
    score = score_raw
    require(score.get("total_count") == 20, "BFCL score total mismatch")
    require(score.get("correct_count", 0) + score.get("incorrect_count", 0) == 20, "BFCL correct/incorrect mismatch")
    require(score.get("failure_rows") == score.get("incorrect_count"), "BFCL failure-body arithmetic mismatch")
    accuracy = score.get("accuracy")
    if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
        raise ValueError("BFCL accuracy invalid")
    accuracy_value = float(accuracy)
    require(math.isfinite(accuracy_value) and 0.0 <= accuracy_value <= 1.0, "BFCL accuracy invalid")
    require(math.isclose(accuracy_value, float(score.get("correct_count", 0)) / 20.0, rel_tol=0.0, abs_tol=1e-12), "BFCL accuracy arithmetic mismatch")
    require(is_sha256(score.get("score_sha256")), "BFCL score hash missing")

    for key in ("model", "image_id", "profile_id", "candidate_id"):
        require(text.get(key) == identity.get(key), f"cross-lane identity mismatch: {key}")

    text_families_raw = text.get("families")
    require(isinstance(text_families_raw, Mapping), "text family summary missing")
    assert isinstance(text_families_raw, Mapping)
    text_families = text_families_raw
    families = {name: dict(stats) for name, stats in text_families.items()}
    families["bfcl_hard20"] = {
        "n": 20,
        "transported": 20,
        "grade_complete": 20,
        "correct": score["correct_count"],
        "incorrect": score["incorrect_count"],
        "ungraded": 0,
        "grader_errors": 0,
        "accuracy_pct": round(100.0 * score["correct_count"] / 20, 2),
        "scorer": "official bfcl-eval partial evaluation",
    }
    correct = int(text["correct_count"]) + int(score["correct_count"])
    incorrect = int(text["incorrect_count"]) + int(score["incorrect_count"])
    require(correct + incorrect == 200, "Q200-v2 total mismatch")
    output = {
        "schema": OUTPUT_SCHEMA,
        "status": "SCORED",
        "suite": "Q200-v2",
        "definition": "frozen text-180 plus model-independent official BFCL v4 multi_turn_base structural-hard20 subset",
        "model": text["model"],
        "image_id": text["image_id"],
        "profile_id": text["profile_id"],
        "candidate_id": text["candidate_id"],
        "chat_template_kwargs": EXPECTED_CHAT,
        "total_count": 200,
        "correct_count": correct,
        "incorrect_count": incorrect,
        "accuracy_pct": round(100.0 * correct / 200, 2),
        "families": families,
        "source_artifacts": {
            "text_summary_sha256": text_sha,
            "bfcl_summary_sha256": bfcl_sha,
            "text_dataset_sha256": TEXT_DATASET_SHA256,
            "bfcl_manifest_sha256": BFCL_MANIFEST_SHA256,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
