"""AST score files must not collapse onto a longer category name."""

from pathlib import Path

from r0b0bench.lanes.bfcl import _parse_ast_scores


def _write(dirpath: Path, name: str, correct: int) -> None:
    header = {
        "accuracy": correct / 200,
        "correct_count": correct,
        "total_count": 200,
    }
    (dirpath / name).write_text(
        __import__("json").dumps(header) + "\n", encoding="utf-8"
    )


def test_ast_categories_do_not_share_parallel_multiple(tmp_path: Path) -> None:
    _write(tmp_path, "BFCL_v4_multiple_score.json", 70)
    _write(tmp_path, "BFCL_v4_parallel_score.json", 103)
    _write(tmp_path, "BFCL_v4_parallel_multiple_score.json", 42)
    parsed = _parse_ast_scores(tmp_path)
    cats = parsed["categories"]
    assert cats["multiple"]["correct_count"] == 70
    assert cats["parallel"]["correct_count"] == 103
    assert cats["parallel_multiple"]["correct_count"] == 42
    assert parsed["micro_correct"] == 215
    assert parsed["micro_total"] == 600
    paths = {info["path"] for info in cats.values()}
    assert len(paths) == 3
