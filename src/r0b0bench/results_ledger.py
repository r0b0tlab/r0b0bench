"""Results ledger: record and compare package-produced runs."""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ENTRY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")


def results_root() -> Path:
    # package lives in src/r0b0bench → repo root is parents[2] when editable
    here = Path(__file__).resolve()
    # .../src/r0b0bench/results_ledger.py → parents[2] = repo root
    candidate = here.parents[2]
    if (candidate / "results" / "entries").is_dir():
        return candidate / "results"
    # installed wheel fallback: cwd
    cwd = Path.cwd() / "results"
    if cwd.is_dir():
        return cwd
    raise FileNotFoundError("results/ directory not found (run from r0b0bench checkout)")


def list_entries() -> list[dict[str, Any]]:
    root = results_root()
    out = []
    for p in sorted((root / "entries").glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        d["_path"] = str(p)
        out.append(d)
    return out


def show_entry(entry_id: str) -> dict[str, Any]:
    root = results_root()
    path = root / "entries" / f"{entry_id}.json"
    if not path.exists():
        # allow bare filename
        matches = list((root / "entries").glob(f"*{entry_id}*.json"))
        if len(matches) == 1:
            path = matches[0]
        else:
            raise FileNotFoundError(f"entry not found: {entry_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _lane_summary(report: dict[str, Any], lane_id: str) -> dict[str, Any]:
    for r in report.get("lanes") or []:
        if r.get("lane_id") == lane_id:
            return r.get("summary") or {}
    # scores dict fallback
    scores = report.get("scores") or {}
    if lane_id in scores and isinstance(scores[lane_id], dict):
        return scores[lane_id]
    return {}


def entry_from_report(
    report: dict[str, Any],
    *,
    entry_id: str,
    model_display: str | None = None,
    hardware: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if not ENTRY_ID_RE.match(entry_id):
        raise ValueError(f"invalid entry_id {entry_id!r}")

    def S(name: str) -> dict[str, Any]:
        return _lane_summary(report, name)

    mt = S("bfcl_mt")
    ast = S("bfcl_ast")
    lat = S("latency")
    stream = lat.get("stream") if isinstance(lat.get("stream"), dict) else lat
    conc = S("concurrency")
    thr = S("throughput")
    qa = S("qa")
    ife = S("ifeval")
    he = S("humaneval")
    gsm = S("gsm8k")
    niah = S("niah")

    # AST micro from nested score if needed
    micro = ast.get("micro_accuracy")
    if micro is None and isinstance(ast.get("score"), dict):
        micro = ast["score"].get("micro_accuracy")
    if micro is None and report.get("scores", {}).get("bfcl_ast_micro"):
        bam = report["scores"]["bfcl_ast_micro"]
        if isinstance(bam, dict):
            micro = bam.get("accuracy")

    entry = {
        "schema_version": 1,
        "entry_id": entry_id,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": report.get("model") or "unknown",
            "display_name": model_display or report.get("model") or entry_id,
        },
        "runtime": {
            "hardware": hardware or "",
            "base_url_omitted": True,
        },
        "harness": {
            "name": "r0b0bench",
            "version": report.get("r0b0bench_version") or report.get("version") or "",
            "profile": report.get("profile") or "core-subset",
        },
        "scores": {
            "canary": {"status": _status(report, "canary"), **{k: S("canary").get(k) for k in ("passed", "n") if k in S("canary")}},
            "bfcl_mt": {
                "status": _status(report, "bfcl_mt"),
                "accuracy": mt.get("accuracy") or (mt.get("score") or {}).get("accuracy"),
                "correct": (mt.get("score") or {}).get("correct") or (mt.get("score") or {}).get("primary", {}).get("correct_count"),
                "total": (mt.get("score") or {}).get("total") or (mt.get("score") or {}).get("primary", {}).get("total_count") or 200,
            },
            "bfcl_ast": {
                "status": _status(report, "bfcl_ast"),
                "micro_accuracy": micro,
                "categories": (ast.get("score") or {}).get("categories") if isinstance(ast.get("score"), dict) else None,
            },
            "latency": {
                "status": _status(report, "latency"),
                "ttft_ms_mean": stream.get("ttft_ms_mean"),
                "itl_ms_mean": stream.get("itl_ms_mean"),
                "e2el_ms_mean": stream.get("e2el_ms_mean"),
            },
            "concurrency": {
                "status": _status(report, "concurrency"),
                "ladder_aggregate_tok_s": {
                    str(r.get("concurrency")): r.get("aggregate_output_tok_s")
                    for r in (conc.get("rows") or [])
                    if isinstance(r, dict)
                },
            },
            "throughput": {
                "status": _status(report, "throughput"),
                "decode_median_client_tok_s": (thr.get("decode") or {}).get("median_client_output_tok_s")
                if isinstance(thr.get("decode"), dict)
                else thr.get("decode_median_client_tok_s"),
            },
            "qa": {
                "status": _status(report, "qa"),
                "accuracy": qa.get("accuracy"),
                "correct": qa.get("correct"),
                "n": qa.get("n"),
            },
            "ifeval": {
                "status": _status(report, "ifeval"),
                "accuracy": ife.get("accuracy"),
                "correct": ife.get("correct"),
                "n": ife.get("n"),
                "method": ife.get("method"),
            },
            "humaneval": {
                "status": _status(report, "humaneval"),
                "pass@1": he.get("pass@1") or (he.get("pass_at") or {}).get("pass@1") if isinstance(he.get("pass_at"), dict) else he.get("pass_at"),
                "n": he.get("n") or 164,
            },
            "gsm8k": {
                "status": _status(report, "gsm8k"),
                "accuracy": gsm.get("accuracy"),
                "correct": gsm.get("correct"),
                "n": gsm.get("n"),
                "method": gsm.get("method"),
            },
            "niah": {
                "status": _status(report, "niah") or niah.get("status"),
                "depths": niah.get("depths") or niah.get("rows") or (report.get("scores") or {}).get("niah_long1m", {}).get("rows"),
                "max_model_len": niah.get("max_model_len"),
            },
        },
        "infra_errors_total": int(report.get("infra_errors_total") or 0),
        "invalid_for_publish": bool(report.get("invalid_for_publish", False)),
        "disclosures": list(report.get("notes") or []),
        "artifact_roots_private": [],
    }
    if notes:
        entry["disclosures"].append(notes)
    return entry


def _status(report: dict[str, Any], lane_id: str) -> str | None:
    for r in report.get("lanes") or []:
        if r.get("lane_id") == lane_id:
            return r.get("status")
    return None


def write_entry(entry: dict[str, Any], *, force: bool = False) -> Path:
    root = results_root()
    path = root / "entries" / f"{entry['entry_id']}.json"
    if path.exists() and not force:
        raise FileExistsError(f"entry exists: {path} (use --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    return path


def rebuild_index() -> tuple[Path, Path]:
    # prefer in-repo script
    root = results_root().parent
    script = root / "scripts" / "rebuild_results_index.py"
    if script.exists():
        import runpy

        runpy.run_path(str(script), run_name="__main__")
    else:
        # minimal fallback
        entries = list_entries()
        idx = {"schema_version": 1, "n_entries": len(entries), "entries": [{"entry_id": e.get("entry_id")} for e in entries]}
        (results_root() / "index.json").write_text(json.dumps(idx, indent=2) + "\n")
    return results_root() / "index.json", results_root() / "LEADERBOARD.md"


def compare_entries(a_id: str, b_id: str) -> str:
    a = show_entry(a_id)
    b = show_entry(b_id)
    keys = [
        ("gsm8k", "accuracy"),
        ("humaneval", "pass@1"),
        ("qa", "accuracy"),
        ("ifeval", "accuracy"),
        ("bfcl_mt", "accuracy"),
        ("bfcl_ast", "micro_accuracy"),
        ("latency", "ttft_ms_mean"),
        ("niah", "status"),
    ]
    lines = [
        f"compare {a.get('entry_id')}  vs  {b.get('entry_id')}",
        f"model  {(a.get('model') or {}).get('display_name')}  |  {(b.get('model') or {}).get('display_name')}",
        "",
        f"{'metric':24} {'A':>12} {'B':>12} {'delta':>12}",
    ]
    for lane, field in keys:
        va = (a.get("scores") or {}).get(lane, {}).get(field) if isinstance((a.get("scores") or {}).get(lane), dict) else None
        vb = (b.get("scores") or {}).get(lane, {}).get(field) if isinstance((b.get("scores") or {}).get(lane), dict) else None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
            lines.append(f"{lane+'.'+field:24} {va:12.4f} {vb:12.4f} {delta:+12.4f}")
        else:
            lines.append(f"{lane+'.'+field:24} {str(va):>12} {str(vb):>12} {'':>12}")
    return "\n".join(lines) + "\n"
