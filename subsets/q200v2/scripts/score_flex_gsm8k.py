#!/usr/bin/env python3
"""Flex-extract GSM8K answers from a quality-run JSONL file.

Extraction precedence is frozen: last ``\\boxed{...}`` number, then the last
bold number on an explicit answer statement, then the last bold number, then
the last number anywhere. TeX thousands separators such as ``70{,}000`` are
normalized before extraction. This scorer never executes generated code and
does not treat the first plausible number as an answer.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_EXPLICIT_ANSWER_RE = re.compile(
    r"\b(?:final\s+answer|answer\s+is|therefore|thus|hence|started\s+with|paid\s+a\s+total)\b",
    flags=re.IGNORECASE,
)


def _number_in(text: str) -> str | None:
    normalized = text.replace("\u2212", "-")
    normalized = re.sub(r"(?<=\d)\{,\}(?=\d)", "", normalized)
    matches = list(_NUMBER_RE.finditer(normalized))
    if not matches:
        return None
    return matches[-1].group(0).replace(",", "").rstrip(".")


def final_answer(text: str) -> str | None:
    if not text:
        return None
    # Search all boxed spans first, from the end.  The conservative pattern
    # handles one nested brace pair without pretending to parse arbitrary TeX.
    boxed = re.findall(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", text, flags=re.DOTALL)
    for span in reversed(boxed):
        value = _number_in(span)
        if value is not None:
            return value
    explicit_answer_numbers: list[str] = []
    for line in text.splitlines():
        if not _EXPLICIT_ANSWER_RE.search(line):
            continue
        for span in re.findall(r"\*\*(.+?)\*\*", line, flags=re.DOTALL):
            value = _number_in(span)
            if value is not None:
                explicit_answer_numbers.append(value)
    if explicit_answer_numbers:
        return explicit_answer_numbers[-1]
    bold_numbers: list[str] = []
    for span in re.findall(r"\*\*(.+?)\*\*", text, flags=re.DOTALL):
        value = _number_in(span)
        if value is not None:
            bold_numbers.append(value)
    if bold_numbers:
        return bold_numbers[-1]
    return _number_in(text)


def normalize_number(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        decimal = Decimal(text)
    except InvalidOperation:
        return None
    if not decimal.is_finite():
        return None
    rendered = format(decimal, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def score_rows(rows: Iterable[Mapping[str, Any]], quality_set: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    qset: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    for record in quality_set:
        row_id = record.get("id")
        if not isinstance(row_id, str) or row_id in qset:
            errors.append(f"invalid/duplicate quality-set id: {row_id}")
        else:
            qset[row_id] = record
    observed: set[str] = set()
    count = passed = 0
    failures: list[dict[str, str]] = []
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id in observed:
            errors.append(f"invalid/duplicate result id: {row_id}")
            continue
        observed.add(row_id)
        source = qset.get(row_id)
        if source is None:
            errors.append(f"unknown result id: {row_id}")
            continue
        if source.get("family") != "gsm8k":
            continue
        count += 1
        expected = normalize_number(source.get("reference"))
        got = normalize_number(final_answer(str(row.get("content", row.get("text", "")) or "")))
        ok = expected is not None and got is not None and Decimal(expected) == Decimal(got)
        if ok:
            passed += 1
        else:
            failures.append({"id": row_id, "expected": expected or "INVALID_REFERENCE", "got": got or "NO-EXTRACT"})
    missing = sorted(set(qset) - observed)
    errors.extend(f"missing result id: {row_id}" for row_id in missing)
    return {
        "gsm8k_flex": {"n": count, "passed": passed, "pct": round(100.0 * passed / count, 2) if count else 0.0},
        "failures_head": failures[:8],
        "errors": errors,
        "complete": not errors,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {line_number}: {path}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL line {line_number}: {path}")
            result.append(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows_file", type=Path)
    parser.add_argument("quality_set", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = score_rows(_read_jsonl(args.rows_file), _read_jsonl(args.quality_set))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"score failed: {exc}", file=sys.stderr)
        return 2
    report["rows_file"] = str(args.rows_file)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
