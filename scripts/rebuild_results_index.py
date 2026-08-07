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


def _spec_str(e: dict) -> str:
    spec = (e.get("runtime") or {}).get("speculative") or {}
    method = spec.get("method", "none")
    if method == "none":
        return "none"
    parts = [method]
    if spec.get("K"):
        parts.append(f"K={spec['K']}")
    if spec.get("draft_tokens"):
        parts.append(f"draft={spec['draft_tokens']}")
    if spec.get("config"):
        parts.append(spec["config"])
    return " ".join(parts)


def _decode_tok_s(e: dict) -> float | None:
    return metric(e, "throughput", "decode_median_client_tok_s")


def _prefill_tok_s(e: dict) -> float | None:
    # check common locations
    v = metric(e, "throughput", "prefill_median_prompt_tok_s")
    if v is None:
        v = metric(e, "throughput", "prefill_median_prompt_tok_s_proxy")
    return v


def _conc_ladder(e: dict) -> dict:
    raw = metric(e, "concurrency", "ladder_aggregate_tok_s")
    return raw if isinstance(raw, dict) else {}


def main() -> int:
    entries = load_entries()
    index = {
        "schema_version": 2,
        "n_entries": len(entries),
        "entries": [
            {
                "entry_id": e.get("entry_id"),
                "file": e.get("_file"),
                "model": (e.get("model") or {}).get("display_name") or (e.get("model") or {}).get("id"),
                "model_family": (e.get("model") or {}).get("family", ""),
                "profile": (e.get("harness") or {}).get("profile"),
                "hardware": (e.get("runtime") or {}).get("hardware", ""),
                "speculative": _spec_str(e),
                "invalid_for_publish": e.get("invalid_for_publish"),
                "infra_errors_total": e.get("infra_errors_total"),
                # Quality
                "gsm8k": metric(e, "gsm8k", "accuracy"),
                "humaneval_pass1": metric(e, "humaneval", "pass@1"),
                "qa": metric(e, "qa", "accuracy"),
                "ifeval": metric(e, "ifeval", "accuracy"),
                "bfcl_mt": metric(e, "bfcl_mt", "accuracy"),
                "bfcl_ast_micro": metric(e, "bfcl_ast", "micro_accuracy"),
                # Performance
                "decode_tok_s": _decode_tok_s(e),
                "prefill_tok_s": _prefill_tok_s(e),
                "ttft_ms": metric(e, "latency", "ttft_ms_mean"),
                "e2el_ms": metric(e, "latency", "e2el_ms_mean"),
                "conc_c1": _conc_ladder(e).get("1"),
                "conc_c2": _conc_ladder(e).get("2"),
                "conc_c4": _conc_ladder(e).get("4"),
                "conc_c6": _conc_ladder(e).get("6"),
                # Long context
                "niah": metric(e, "niah", "status"),
            }
            for e in entries
        ],
    }
    INDEX.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    # --- LEADERBOARD.md ---
    lines = [
        "# r0b0bench leaderboard",
        "",
        f"Entries: **{len(entries)}** · regenerated from `results/entries/*.json`",
        "",
        "Comparable only within the same profile and disclosed scorer variants.",
        "",
        "## Quality",
        "",
        "| entry_id | model | spec | GSM8K | HE@1 | QA | IFEval | BFCL-MT | ASTµ | NIAH | invalid |",
        "|----------|-------|------|------:|-----:|---:|-------:|-------:|-----:|------|---------|",
    ]
    for row in index["entries"]:
        lines.append(
            "| {eid} | {model} | {spec} | {gsm} | {he} | {qa} | {ife} | {mt} | {ast} | {niah} | {inv} |".format(
                eid=row.get("entry_id"),
                model=(row.get("model") or "")[:35],
                spec=row.get("speculative", ""),
                gsm=_fmt(row.get("gsm8k")),
                he=_fmt(row.get("humaneval_pass1")),
                qa=_fmt(row.get("qa")),
                ife=_fmt(row.get("ifeval")),
                mt=_fmt(row.get("bfcl_mt")),
                ast=_fmt(row.get("bfcl_ast_micro")),
                niah=row.get("niah") or "—",
                inv=row.get("invalid_for_publish"),
            )
        )

    # Performance table
    lines += [
        "",
        "## Performance",
        "",
        "| entry_id | model | spec | decode tok/s | prefill tok/s | TTFT ms | c1 agg | c2 agg | c4 agg | c6 agg |",
        "|----------|-------|------|------------:|-------------:|-------:|-------:|-------:|-------:|-------:|",
    ]
    for row in index["entries"]:
        lines.append(
            "| {eid} | {model} | {spec} | {dec} | {pre} | {ttft} | {c1} | {c2} | {c4} | {c6} |".format(
                eid=row.get("entry_id"),
                model=(row.get("model") or "")[:35],
                spec=row.get("speculative", ""),
                dec=_fmt(row.get("decode_tok_s")),
                pre=_fmt(row.get("prefill_tok_s")),
                ttft=_fmt(row.get("ttft_ms")),
                c1=_fmt(row.get("conc_c1")),
                c2=_fmt(row.get("conc_c2")),
                c4=_fmt(row.get("conc_c4")),
                c6=_fmt(row.get("conc_c6")),
            )
        )

    # Entry files
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
