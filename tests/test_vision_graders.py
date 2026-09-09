"""Unit tests for the r0b0bench-vision v1.0 graders (scripts/vision/run_vision.py).

These lock the deterministic grading contract: MC letter extraction, MMVP paired
accuracy, RealWorldQA letter/word handling, and the official OCRBench containment
rule (including the HME100k space-stripping variant).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISION_DIR = ROOT / "scripts" / "vision"


def _load():
    spec = importlib.util.spec_from_file_location("run_vision", VISION_DIR / "run_vision.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_vision"] = mod
    spec.loader.exec_module(mod)
    return mod


rv = _load()


def _row(suite, gold, subaxis=None, dataset_name=None):
    return {
        "id": "x",
        "suite": suite,
        "subaxis": subaxis,
        "dataset_name": dataset_name,
        "gold": gold,
    }


def test_extract_choice_basic():
    assert rv.extract_choice("C", "ABCDEF") == "C"
    assert rv.extract_choice("(C)", "ABCDEF") == "C"
    assert rv.extract_choice("c.", "ABCDEF") == "C"
    assert rv.extract_choice("The answer is C.", "ABCDEF") == "C"
    assert rv.extract_choice("", "ABCDEF") is None
    assert rv.extract_choice("Zebra", "ABCDEF") is None


def test_extract_choice_respects_allowed_letters():
    assert rv.extract_choice("(a)", "AB") == "A"
    assert rv.extract_choice("b", "AB") == "B"
    assert rv.extract_choice("c", "AB") is None


def test_grade_cvbench():
    ok, got = rv.grade(_row("cvbench", "(C)"), "C")
    assert ok and got == "C"
    ok, _ = rv.grade(_row("cvbench", "(E)"), "A")
    assert not ok


def test_grade_mmvp():
    ok, _ = rv.grade(_row("mmvp", "(a)"), "A")
    assert ok
    ok, _ = rv.grade(_row("mmvp", "(b)"), "A")
    assert not ok


def test_grade_realworldqa_letter_and_word():
    ok, _ = rv.grade(_row("realworldqa", "C"), "C")
    assert ok
    ok, _ = rv.grade(_row("realworldqa", "Yes"), "yes!")
    assert ok
    ok, _ = rv.grade(_row("realworldqa", "Green"), "The color is green")
    assert ok
    ok, _ = rv.grade(_row("realworldqa", "Green"), "red")
    assert not ok


def test_grade_ocrbench_official_containment():
    ok, _ = rv.grade(
        _row("ocrbench", ["CENTRE"], subaxis="Regular Text Recognition", dataset_name="IIIT5K"),
        "Centre",
    )
    assert ok
    # official rule is containment, not equality
    ok, _ = rv.grade(
        _row("ocrbench", ["FRIEND"], subaxis="Regular Text Recognition", dataset_name="IIIT5K"),
        "FRIENDSHIP",
    )
    assert ok
    ok, _ = rv.grade(
        _row("ocrbench", ["MARKET"], subaxis="Regular Text Recognition", dataset_name="IIIT5K"),
        "shop",
    )
    assert not ok


def test_grade_ocrbench_hme_strips_spaces():
    row = _row(
        "ocrbench",
        ["x = 1"],
        subaxis="Handwritten Mathematical Expression Recognition",
        dataset_name="HME100k",
    )
    ok, _ = rv.grade(row, "x=1")
    assert ok
    boxed = _row(
        "ocrbench",
        ["3 2 + 5 = \\boxed { 3 }"],
        subaxis="Handwritten Mathematical Expression Recognition",
        dataset_name="HME100k",
    )
    ok, _ = rv.grade(boxed, "32+5=37")
    assert not ok  # official rule requires the boxed-form answer string


def test_mmvp_paired_metric():
    rows = [{"id": f"mmvp:{i}", "passed": i % 4 in (1, 2)} for i in range(1, 301)]
    out = rv.mmvp_paired(rows)
    assert out["pairs"] == 150
    assert out["paired_accuracy"] == 0.5  # odd pairs both-correct, even pairs both-wrong


def test_ocrbench_official_tally():
    rows = [
        {"id": "ocr:0", "subaxis": "Regular Text Recognition", "passed": True},
        {"id": "ocr:1", "subaxis": "Regular Text Recognition", "passed": False},
        {"id": "ocr:2", "subaxis": "HME", "passed": True},
    ]
    out = rv.ocrbench_official(rows)
    assert out["Regular Text Recognition"] == {"score": 1, "n": 2}
    assert out["HME"] == {"score": 1, "n": 1}


def test_wilson():
    lo, hi = rv.wilson(50, 100)
    assert abs(lo - 0.4038) < 0.01 and abs(hi - 0.5962) < 0.01
    assert rv.wilson(0, 0) is None


def test_contract_is_frozen():
    import json

    contract = json.loads((VISION_DIR / "benchmark.json").read_text())
    assert contract["name"] == "r0b0bench-vision"
    assert contract["total_rows"] == sum(s["rows"] for s in contract["suites"])
    ids = [s["id"] for s in contract["suites"]]
    assert ids == ["cvbench", "mmvp", "realworldqa", "ocrbench"]
    for s in contract["suites"]:
        assert len(s["revision"]) == 40
