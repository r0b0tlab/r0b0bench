from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

from r0b0bench.config import LaneResult, write_json
from r0b0bench.endpoint import Endpoint


def _bfcl_available() -> bool:
    return importlib.util.find_spec("bfcl_eval") is not None


def run_bfcl_mt(ep: Endpoint, out_dir: Path, cfg: dict[str, Any]) -> LaneResult:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not _bfcl_available():
        summary = {
            "error": "bfcl-eval not installed in this environment",
            "hint": "pip install 'r0b0bench[bfcl]' or use image tag with bfcl extra",
            "expected_rows": cfg.get("expected_rows", 200),
            "category": cfg.get("category", "multi_turn_base"),
            "endpoint": ep.base_url,
            "model": ep.model,
        }
        write_json(out_dir / "bfcl_mt.json", summary)
        return LaneResult(
            lane_id="bfcl_mt",
            status="NOT_IMPLEMENTED",
            summary=summary,
            artifacts={"bfcl_mt.json": str(out_dir / "bfcl_mt.json")},
            infra_errors=0,
            elapsed_s=time.perf_counter() - t0,
        )
    # Full BFCL orchestration is heavy; RC1 records intent + doctor hook.
    # Prefer import-lane from prior official runs or install bfcl extra and use external runner.
    summary = {
        "status": "requires_external_runner",
        "message": (
            "bfcl-eval is importable. Run official multi_turn_base via project scripts "
            "or r0b0bench import-lane from a completed evidence tree. "
            "Inline full 200-case generation will ship in a follow-up RC."
        ),
        "package": cfg.get("package"),
        "category": cfg.get("category"),
        "expected_rows": cfg.get("expected_rows", 200),
        "temperature": cfg.get("temperature", 0.001),
        "num_threads": cfg.get("num_threads", 1),
        "model": ep.model,
        "base_url": ep.base_url,
    }
    write_json(out_dir / "bfcl_mt.json", summary)
    return LaneResult(
        lane_id="bfcl_mt",
        status="NOT_IMPLEMENTED",
        summary=summary,
        artifacts={"bfcl_mt.json": str(out_dir / "bfcl_mt.json")},
        elapsed_s=time.perf_counter() - t0,
    )


def run_bfcl_ast(ep: Endpoint, out_dir: Path, cfg: dict[str, Any]) -> LaneResult:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "requires_external_runner",
        "categories": cfg.get("categories", ["multiple", "parallel", "parallel_multiple"]),
        "expected_rows_per_category": cfg.get("expected_rows_per_category", 200),
        "bfcl_available": _bfcl_available(),
        "model": ep.model,
        "base_url": ep.base_url,
        "message": "Use import-lane from AST-600 evidence or external bfcl_ast_run until inlined.",
    }
    write_json(out_dir / "bfcl_ast.json", summary)
    return LaneResult(
        lane_id="bfcl_ast",
        status="NOT_IMPLEMENTED",
        summary=summary,
        artifacts={"bfcl_ast.json": str(out_dir / "bfcl_ast.json")},
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
