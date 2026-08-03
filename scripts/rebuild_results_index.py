#!/usr/bin/env python3
"""Build results/index.json and results/LEADERBOARD.md from entries/*.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ROOT / "results" / "entries"
INDEX = ROOT / "results" / "index.json"
BOARD = ROOT / "results" / "LEADERBOARD.md"


def load_entries() -> list[dict]:
    rows = []
    for p in sorted(ENTRIES.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {p.name}: {exc}", file=sys.stderr)
            continue
        d["_file"] = p.name
        rows.append(d)
    return rows


def metric(entry: dict, *path, default=None):
    cur = entry.get("scores") or {}
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return default if cur is None else cur


def main() -> int:
    entries = load_entries()
    index = {
        "schema_version": 1,
        "n_entries": len(entries),
        "entries": [
            {
                "entry_id": e.get("entry_id"),
                "file": e.get("_file"),
                "model": (e.get("model") or {}).get("display_name") or (e.get("model") or {}).get("id"),
                "profile": (e.get("harness") or {}).get("profile"),
                "invalid_for_publish": e.get("invalid_for_publish"),
                "infra_errors_total": e.get("infra_errors_total"),
                "gsm8k": metric(e, "gsm8k", "accuracy"),
                "humaneval_pass1": metric(e, "humaneval", "pass@1"),
                "qa": metric(e, "qa", "accuracy"),
                "ifeval": metric(e, "ifeval", "accuracy"),
                "bfcl_mt": metric(e, "bfcl_mt", "accuracy"),
                "bfcl_ast_micro": metric(e, "bfcl_ast", "micro_accuracy"),
                "niah": metric(e, "niah", "status"),
                "ttft_ms": metric(e, "latency", "ttft_ms_mean"),
                "c1_tok_s": (metric(e, "concurrency", "ladder_aggregate_tok_s") or {}).get("1")
                if isinstance(metric(e, "concurrency", "ladder_aggregate_tok_s"), dict)
                else None,
            }
            for e in entries
        ],
    }
    INDEX.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# r0b0bench leaderboard",
        "",
        f"Entries: **{len(entries)}** · regenerated from `results/entries/*.json`",
        "",
        "Comparable only within the same profile and disclosed scorer variants.",
        "",
        "| entry_id | model | profile | GSM8K | HE@1 | QA | IFEval | BFCL-MT | ASTµ | NIAH | invalid |",
        "|----------|-------|---------|------:|-----:|---:|-------:|-------:|-----:|------|---------|",
    ]
    for row in index["entries"]:
        lines.append(
            "| {entry_id} | {model} | {profile} | {gsm8k} | {he} | {qa} | {ife} | {mt} | {ast} | {niah} | {inv} |".format(
                entry_id=row.get("entry_id"),
                model=(row.get("model") or "")[:40],
                profile=row.get("profile"),
                gsm8k=_fmt(row.get("gsm8k")),
                he=_fmt(row.get("humaneval_pass1")),
                qa=_fmt(row.get("qa")),
                ife=_fmt(row.get("ifeval")),
                mt=_fmt(row.get("bfcl_mt")),
                ast=_fmt(row.get("bfcl_ast_micro")),
                niah=row.get("niah") or "—",
                inv=row.get("invalid_for_publish"),
            )
        )
    lines += ["", "## Files", ""]
    for e in entries:
        lines.append(f"- [`{e.get('_file')}`](entries/{e.get('_file')}) — {e.get('entry_id')}")
    lines.append("")
    BOARD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {INDEX} ({len(entries)} entries)")
    print(f"wrote {BOARD}")
    return 0


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


if __name__ == "__main__":
    raise SystemExit(main())
