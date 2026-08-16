from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from r0b0bench.config import LaneResult, write_json
from r0b0bench.endpoint import Endpoint

BFCL_PY = Path(os.environ.get("R0B0BENCH_BFCL_PYTHON") or os.environ.get("BFCL_PYTHON") or "")
# Prefer explicit env; otherwise use in-repo official adapter under scripts/bfcl
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SCRIPTS = _REPO_ROOT / "scripts" / "bfcl"
BFCL_SCRIPTS = Path(os.environ.get("R0B0BENCH_BFCL_SCRIPTS") or (_DEFAULT_SCRIPTS if _DEFAULT_SCRIPTS.is_dir() else ""))


def _env(ep: Endpoint, project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = str(project_root)
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY") or "EMPTY"
    # base_url should be .../v1
    env["OPENAI_BASE_URL"] = ep.base_url.rstrip("/")
    env["R0B0BENCH_SERVED_MODEL"] = ep.model
    env["PYTHONUNBUFFERED"] = "1"
    # GB10 unified memory: BFCL at C=8 can trigger concurrent CUDA compilation
    # and globally OOM the serving host. The public safe default is C=4 with a
    # bounded request lifetime and no retry fan-out; explicit env values remain
    # an intentional profile override.
    env.setdefault("BFCL_NUM_THREADS", os.environ.get("BFCL_NUM_THREADS", "4"))
    env.setdefault("BFCL_HTTP_TIMEOUT", os.environ.get("BFCL_HTTP_TIMEOUT", "600"))
    env.setdefault("BFCL_MAX_RETRIES", os.environ.get("BFCL_MAX_RETRIES", "1"))
    return env


def _run(cmd: list[str], env: dict[str, str], log_path: Path, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        return subprocess.run(
            cmd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )


def _parse_mt_score(score_dir: Path) -> dict[str, Any]:
    # Find multi_turn score json
    hits = list(score_dir.rglob("*multi_turn_base*score*.json")) + list(score_dir.rglob("*multi_turn*score*.json"))
    if not hits:
        # any score json
        hits = list(score_dir.rglob("*score*.json"))
    summary: dict[str, Any] = {"score_files": [str(p) for p in hits[:10]]}
    for path in hits:
        try:
            text = path.read_text(encoding="utf-8")
            # score files are often JSONL: first line overall
            first = text.splitlines()[0] if text.strip() else ""
            if first.startswith("{"):
                row = json.loads(first)
                if "accuracy" in row or "ACCURACY_RATE" in row or "correct_count" in row:
                    summary["primary"] = row
                    summary["primary_path"] = str(path)
                    acc = row.get("accuracy", row.get("ACCURACY_RATE"))
                    if acc is not None:
                        summary["accuracy"] = float(acc)
                    if "correct_count" in row and "total_count" in row:
                        summary["correct"] = row["correct_count"]
                        summary["total"] = row["total_count"]
                    break
        except Exception:
            continue
    # CSV fallback
    for csv in score_dir.rglob("*.csv"):
        try:
            lines = csv.read_text(encoding="utf-8").splitlines()
            summary.setdefault("csv", []).append(str(csv))
            for line in lines:
                if "multi_turn" in line.lower() or "Multi Turn" in line:
                    summary.setdefault("csv_rows", []).append(line)
        except Exception:
            pass
    return summary


def _parse_ast_scores(score_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"categories": {}}
    for cat in ("multiple", "parallel", "parallel_multiple"):
        hits = list(score_dir.rglob(f"*{cat}*score*.json"))
        for path in hits:
            try:
                first = path.read_text(encoding="utf-8").splitlines()[0]
                row = json.loads(first)
                out["categories"][cat] = {
                    "path": str(path),
                    "accuracy": row.get("accuracy", row.get("ACCURACY_RATE")),
                    "correct_count": row.get("correct_count"),
                    "total_count": row.get("total_count"),
                    "raw": row,
                }
                break
            except Exception:
                continue
    # micro
    accs = []
    correct = total = 0
    for cat, info in out["categories"].items():
        if info.get("correct_count") is not None and info.get("total_count"):
            correct += int(info["correct_count"])
            total += int(info["total_count"])
        elif info.get("accuracy") is not None and info.get("total_count"):
            # estimate
            pass
    if total:
        out["micro_accuracy"] = correct / total
        out["micro_correct"] = correct
        out["micro_total"] = total
    return out


def run_bfcl_mt(ep: Endpoint, out_dir: Path, cfg: dict[str, Any]) -> LaneResult:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    script = BFCL_SCRIPTS / "bfcl_run.py"
    if not BFCL_PY.exists() or not script.exists():
        summary = {
            "error": "bfcl runner missing",
            "bfcl_python": str(BFCL_PY),
            "script": str(script),
            "configuration": "set R0B0BENCH_BFCL_PYTHON and R0B0BENCH_BFCL_SCRIPTS to an official bfcl-eval adapter",
        }
        write_json(out_dir / "bfcl_mt.json", summary)
        return LaneResult(lane_id="bfcl_mt", status="ERROR", summary=summary, infra_errors=1, elapsed_s=time.perf_counter() - t0)

    project = out_dir / "bfcl-project"
    project.mkdir(parents=True, exist_ok=True)
    env = _env(ep, project)
    log = out_dir / "bfcl_mt.log"

    # Prefer resume/full against isolated root (fresh = full)
    # If prior results exist in project, resume
    rc = _run([str(BFCL_PY), str(script), "full"], env=env, log_path=log, timeout=None)
    if rc.returncode != 0:
        # try resume path if partial
        rc2 = _run([str(BFCL_PY), str(script), "resume"], env=env, log_path=log, timeout=None)
        if rc2.returncode != 0:
            summary = {
                "status": "ERROR",
                "returncode": rc2.returncode or rc.returncode,
                "log": str(log),
                "project": str(project),
            }
            write_json(out_dir / "bfcl_mt.json", summary)
            return LaneResult(
                lane_id="bfcl_mt",
                status="ERROR",
                summary=summary,
                artifacts={"log": str(log)},
                infra_errors=1,
                elapsed_s=time.perf_counter() - t0,
            )

    # score dir under BFCL_PROJECT_ROOT/score
    score_dir = project / "score"
    # bfcl may write under result/score relative to package default when RESULT_PATH uses project root
    # Also search result tree
    cand_score = list(project.rglob("score")) or [score_dir]
    parsed = {}
    for sd in cand_score:
        if sd.is_dir():
            parsed = _parse_mt_score(sd)
            if parsed.get("accuracy") is not None or parsed.get("primary"):
                break

    # copy key artifacts
    for pat in ("*multi_turn_base*result*.json", "*multi_turn_base*score*.json"):
        for p in project.rglob(pat):
            try:
                shutil.copy2(p, out_dir / p.name)
            except Exception:
                pass

    summary = {
        "status": "PASS",
        "category": cfg.get("category", "multi_turn_base"),
        "expected_rows": cfg.get("expected_rows", 200),
        "project": str(project),
        "score": parsed,
        "model": ep.model,
        "base_url": ep.base_url,
    }
    if parsed.get("accuracy") is not None:
        summary["accuracy"] = parsed["accuracy"]
    if parsed.get("accuracy") is None and not parsed.get("primary"):
        summary["status"] = "ERROR"
        summary["error"] = "official BFCL MT score was not parsed"
        write_json(out_dir / "bfcl_mt.json", summary)
        return LaneResult(
            lane_id="bfcl_mt",
            status="ERROR",
            summary=summary,
            artifacts={"bfcl_mt.json": str(out_dir / "bfcl_mt.json"), "log": str(log)},
            infra_errors=1,
            elapsed_s=time.perf_counter() - t0,
        )
    write_json(out_dir / "bfcl_mt.json", summary)
    return LaneResult(
        lane_id="bfcl_mt",
        status="PASS",
        summary=summary,
        artifacts={"bfcl_mt.json": str(out_dir / "bfcl_mt.json"), "log": str(log)},
        elapsed_s=time.perf_counter() - t0,
    )


def run_bfcl_ast(ep: Endpoint, out_dir: Path, cfg: dict[str, Any]) -> LaneResult:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    script = BFCL_SCRIPTS / "bfcl_ast_run.py"
    if not BFCL_PY.exists() or not script.exists():
        summary = {
            "error": "bfcl ast runner missing",
            "script": str(script),
            "configuration": "set R0B0BENCH_BFCL_PYTHON and R0B0BENCH_BFCL_SCRIPTS to an official bfcl-eval adapter",
        }
        write_json(out_dir / "bfcl_ast.json", summary)
        return LaneResult(lane_id="bfcl_ast", status="ERROR", summary=summary, infra_errors=1, elapsed_s=time.perf_counter() - t0)

    project = out_dir / "bfcl-ast-project"
    project.mkdir(parents=True, exist_ok=True)
    env = _env(ep, project)
    log = out_dir / "bfcl_ast.log"
    cats = list(cfg.get("categories") or ["multiple", "parallel", "parallel_multiple"])

    steps = [
        [str(BFCL_PY), str(script), "strip-errors"],
        [str(BFCL_PY), str(script), "resume"],  # fills all missing across cats with BFCL_NUM_THREADS
        [str(BFCL_PY), str(script), "evaluate"],
    ]

    for cmd in steps:
        rc = _run(cmd, env=env, log_path=log)
        if rc.returncode != 0:
            summary = {"status": "ERROR", "failed_cmd": cmd, "returncode": rc.returncode, "log": str(log)}
            write_json(out_dir / "bfcl_ast.json", summary)
            return LaneResult(
                lane_id="bfcl_ast",
                status="ERROR",
                summary=summary,
                artifacts={"log": str(log)},
                infra_errors=1,
                elapsed_s=time.perf_counter() - t0,
            )

    score_dirs = [p for p in project.rglob("score") if p.is_dir()]
    parsed: dict[str, Any] = {}
    for sd in score_dirs:
        parsed = _parse_ast_scores(sd)
        if parsed.get("categories"):
            break

    for pat in ("*multiple*result*.json", "*parallel*result*.json", "*score*.json"):
        for p in project.rglob(pat):
            try:
                dest = out_dir / p.name
                if not dest.exists():
                    shutil.copy2(p, dest)
            except Exception:
                pass

    summary = {
        "status": "PASS",
        "categories": cats,
        "project": str(project),
        "score": parsed,
        "model": ep.model,
        "base_url": ep.base_url,
    }
    if parsed.get("micro_accuracy") is not None:
        summary["micro_accuracy"] = parsed["micro_accuracy"]
        summary["micro_correct"] = parsed.get("micro_correct")
        summary["micro_total"] = parsed.get("micro_total")
    if set(parsed.get("categories", {})) != set(cats) or parsed.get("micro_accuracy") is None:
        summary["status"] = "ERROR"
        summary["error"] = "official BFCL AST scores were not parsed for every category"
        write_json(out_dir / "bfcl_ast.json", summary)
        return LaneResult(
            lane_id="bfcl_ast",
            status="ERROR",
            summary=summary,
            artifacts={"bfcl_ast.json": str(out_dir / "bfcl_ast.json"), "log": str(log)},
            infra_errors=1,
            elapsed_s=time.perf_counter() - t0,
        )
    write_json(out_dir / "bfcl_ast.json", summary)
    return LaneResult(
        lane_id="bfcl_ast",
        status="PASS",
        summary=summary,
        artifacts={"bfcl_ast.json": str(out_dir / "bfcl_ast.json"), "log": str(log)},
        elapsed_s=time.perf_counter() - t0,
    )


def run_quality_stub(lane_id: str, out_dir: Path, meta: dict[str, Any]) -> LaneResult:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "not_implemented_in_rc1",
        "lane": lane_id,
        **meta,
        "message": "Quality lane scaffolded; subset locks and scorers land in next RC.",
    }
    write_json(out_dir / f"{lane_id}.json", summary)
    return LaneResult(
        lane_id=lane_id,
        status="NOT_IMPLEMENTED",
        summary=summary,
        artifacts={f"{lane_id}.json": str(out_dir / f"{lane_id}.json")},
        elapsed_s=time.perf_counter() - t0,
    )
