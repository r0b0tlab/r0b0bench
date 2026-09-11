from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT180 = ROOT / "artifacts" / "quality-text-180-v2.jsonl"
HISTORICAL200 = ROOT / "artifacts" / "quality-200.jsonl"
BFCL20 = ROOT / "artifacts" / "bfcl-v4-multi-turn-hard20-v1.json"
EXPECTED_TEXT_SHA = "74623ab9b075120cd6f7a93059cc16d8817a6039dd20118b8f0350279f8b1ed6"
EXPECTED_MANIFEST_SHA = "0860da504a3db2c3cd73647ecdc2a5ecdb1793d7a5cf3f4f004912f0ef314d4e"
EXPECTED_IDS = [
    "multi_turn_base_109",
    "multi_turn_base_131",
    "multi_turn_base_97",
    "multi_turn_base_110",
    "multi_turn_base_162",
    "multi_turn_base_170",
    "multi_turn_base_180",
    "multi_turn_base_183",
    "multi_turn_base_184",
    "multi_turn_base_186",
    "multi_turn_base_188",
    "multi_turn_base_116",
    "multi_turn_base_117",
    "multi_turn_base_194",
    "multi_turn_base_2",
    "multi_turn_base_33",
    "multi_turn_base_55",
    "multi_turn_base_60",
    "multi_turn_base_68",
    "multi_turn_base_72",
]


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_q200_v2_text180_is_an_exact_non_agentic_projection() -> None:
    historical = rows(HISTORICAL200)
    text = rows(TEXT180)
    assert hashlib.sha256(TEXT180.read_bytes()).hexdigest() == EXPECTED_TEXT_SHA
    assert text == [row for row in historical if row["family"] != "agentic_coding"]
    assert len(text) == 180
    assert Counter(row["family"] for row in text) == {
        "gsm8k": 80,
        "humaneval": 40,
        "ifeval": 40,
        "hard_reasoning": 20,
    }
    assert not any(row["id"].startswith("agentic-") for row in text)


def test_bfcl_hard20_manifest_is_frozen_unique_and_model_independent() -> None:
    assert hashlib.sha256(BFCL20.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA
    manifest = json.loads(BFCL20.read_text(encoding="utf-8"))
    ids = [row["id"] for row in manifest["selected"]]
    assert ids == EXPECTED_IDS
    assert len(ids) == len(set(ids)) == 20
    assert manifest["bfcl_eval_version"] == "2025.12.17"
    assert manifest["category"] == "multi_turn_base"
    assert manifest["selection_policy"]["model_independent"] is True
    assert manifest["label"] == "BFCL v4 multi_turn_base structural-hard20"
    assert manifest["selection_boundary"] == {
        "rank_20_id": "multi_turn_base_72",
        "rank_21_id": "multi_turn_base_102",
        "boundary_structural_key": {
            "user_turn_count": 5,
            "required_path_length": 7,
            "involved_class_count": 2,
        },
        "tied_source_case_count": 20,
        "selected_from_boundary_tie_count": 6,
        "disclosure": "The rank-20 boundary is deterministic but arbitrary within a 20-way structural tie; numeric case ID ascending selects the first six tied cases.",
    }
    assert manifest["selection_provenance"] == {
        "canonical_selected_ids_sha256": "28158003bca5abf7e6922a8aa460d0bed208b3636757349460000b85812b384f",
        "canonical_selected_feature_projection_sha256": "65c7889c2f78f29bd71e2d74854a11234a20dbb02e83186ae7a4c9050b79cbca",
        "canonical_selector_policy_sha256": "03999d7f8e22669d5ec51a9920d0e79a6350005f95f7d305e5835e18479dc0f2",
    }
    assert manifest["source_dataset_count"] == 200
    assert manifest["source_dataset_sha256"] == "1a21a995d06fd6f20ba55de7bced30ef953ec35e998f502ec2ecf4d66ef1c43a"
    assert manifest["source_ground_truth_sha256"] == "1fee67823b317571649177dd89d63969feaae4e810cc7448ee55ba797fb7c8fc"


def _summary_pair(tmp_path: Path) -> tuple[Path, Path]:
    hex_hash = "d" * 64
    identity = {
        "model": "qwen38-flash-next-w4a16",
        "image_id": "sha256:" + "a" * 64,
        "profile_id": "b" * 64,
        "candidate_id": "candidate",
    }
    text = {
        "schema": "r0b0tlab.qwen38.quality_text_180_run.v2",
        "status": "SCORED",
        "dataset_sha256": EXPECTED_TEXT_SHA,
        "dataset_count": 180,
        "rows": 180,
        "transport_complete": True,
        "transport_count": 180,
        "grade_complete": True,
        "ungraded_count": 0,
        "grader_error_count": 0,
        "correct_count": 150,
        "incorrect_count": 30,
        "chat_template_kwargs": {"enable_thinking": True, "thinking": True, "reasoning_effort": "low"},
        "families": {"gsm8k": {"n": 80, "correct": 70, "incorrect": 10}},
        **identity,
    }
    runtime_binding = {
        "adapter": {"module": "adapter", "path": "/adapter.py", "sha256": hex_hash},
        "wrapper_sha256": hex_hash,
        "selector_source_sha256": hex_hash,
        "requirements_bfcl_sha256": hex_hash,
        "official_modules": {
            name: {"module": name, "path": f"/{name}.py", "sha256": hex_hash}
            for name in ("generation", "evaluation", "multi_turn_checker", "openai_handler")
        },
        "package_tree": {
            "distribution": "bfcl-eval",
            "version": "2025.12.17",
            "file_count": 100,
            "manifest_sha256": hex_hash,
        },
        "settings": {
            "num_threads": 1,
            "run_ids": True,
            "partial_eval": True,
            "chat_template_kwargs": {"enable_thinking": True, "thinking": True, "reasoning_effort": "low"},
        },
    }
    runtime_binding_sha = hashlib.sha256(
        json.dumps(runtime_binding, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    bfcl_identity = {
        **identity,
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "dataset_sha256": hex_hash,
        "ground_truth_sha256": hex_hash,
        "selected_ids_sha256": hex_hash,
        "selected_features_sha256": hex_hash,
        "selector_policy_sha256": hex_hash,
        "runtime_binding_sha256": runtime_binding_sha,
        "chat_template_kwargs": {"enable_thinking": True, "thinking": True, "reasoning_effort": "low"},
    }
    bfcl = {
        "schema": "r0b0tlab.qwen38.q200_v2_bfcl_hard20_run.v1",
        "status": "SCORED",
        "label": "BFCL v4 multi_turn_base structural-hard20",
        "identity": bfcl_identity,
        "identity_sha256": hashlib.sha256(
            json.dumps(bfcl_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        ).hexdigest(),
        "runtime_binding": runtime_binding,
        "results": {
            "rows": 20,
            "unique_ids": 20,
            "ids": EXPECTED_IDS,
            "result_sha256": hex_hash,
        },
        "timing": {
            "rows": 20,
            "unique_ids": 20,
            "error_count": 0,
            "timing_sha256": hex_hash,
        },
        "freshness": {
            "freshness_sha256": hex_hash,
            "witness": {
                "mode": "run",
                "preexisting_files": [],
                "preexisting_timing": False,
                "generation_complete": True,
                "result_sha256": hex_hash,
                "timing_sha256": hex_hash,
                "runtime_binding_sha256": runtime_binding_sha,
            },
        },
        "score": {
            "correct_count": 14,
            "incorrect_count": 6,
            "total_count": 20,
            "accuracy": 0.7,
            "failure_rows": 6,
            "score_sha256": hex_hash,
        },
    }
    text_path = tmp_path / "text.json"
    bfcl_path = tmp_path / "bfcl.json"
    text_path.write_text(json.dumps(text), encoding="utf-8")
    bfcl_path.write_text(json.dumps(bfcl), encoding="utf-8")
    return text_path, bfcl_path


def test_q200_v2_closeout_requires_exact_180_plus_20(tmp_path: Path) -> None:
    text, bfcl = _summary_pair(tmp_path)
    output = tmp_path / "closeout.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "close_q200_v2.py"), "--text-summary", str(text), "--bfcl-summary", str(bfcl), "--output", str(output)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    closeout = json.loads(output.read_text(encoding="utf-8"))
    assert closeout["status"] == "SCORED"
    assert closeout["total_count"] == 200
    assert closeout["correct_count"] == 164
    assert closeout["incorrect_count"] == 36
    assert closeout["families"]["bfcl_hard20"]["n"] == 20


def test_q200_v2_closeout_rejects_cross_lane_identity_drift(tmp_path: Path) -> None:
    text, bfcl = _summary_pair(tmp_path)
    value = json.loads(bfcl.read_text(encoding="utf-8"))
    value["identity"]["profile_id"] = "c" * 64
    value["identity_sha256"] = hashlib.sha256(
        json.dumps(value["identity"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    bfcl.write_text(json.dumps(value), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "close_q200_v2.py"), "--text-summary", str(text), "--bfcl-summary", str(bfcl), "--output", str(tmp_path / "out.json")],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "cross-lane identity mismatch" in result.stderr
