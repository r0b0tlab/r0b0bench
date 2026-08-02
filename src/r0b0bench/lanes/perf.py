from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from r0b0bench.config import LaneResult, write_json
from r0b0bench.endpoint import Endpoint
from r0b0bench.lanes.concurrency import run_concurrency
from r0b0bench.lanes.latency import run_latency
from r0b0bench.lanes.throughput import run_throughput


def run_perf(ep: Endpoint, out_dir: Path, cfg: dict[str, Any]) -> LaneResult:
    """Composite systems performance package: latency + concurrency + throughput.

    Kept as a single lane id for backward-compatible core/core-subset profiles.
    Prefer explicit `latency` / `concurrency` / `throughput` lanes when available.
    """
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = str(cfg.get("mode") or "full").lower()
    parts: list[str] = []
    if mode == "legacy_sweep_only":
        # old portable sweep only via concurrency lane levels
        from r0b0bench.lanes.concurrency import run_concurrency as _rc

        legacy_cfg = {
            "levels": cfg.get("levels") or [1, 2, 4, 8, 12, 16],
            "reps": cfg.get("reps") or 3,
            "drop_first_rep": cfg.get("drop_first_rep", True),
            "output_tokens": cfg.get("output_tokens") or 256,
            "temperature": cfg.get("temperature") or 0,
        }
        res = _rc(ep, out_dir / "legacy_concurrency", legacy_cfg)
        summary = {"mode": mode, "legacy": res.summary}
        write_json(out_dir / "summary.json", summary)
        return LaneResult(
            lane_id="perf",
            status=res.status,
            summary=summary,
            artifacts={"summary.json": str(out_dir / "summary.json")},
            infra_errors=res.infra_errors,
            elapsed_s=time.perf_counter() - t0,
        )

    # full package
    lat_cfg = dict(cfg.get("latency") or {})
    conc_cfg = dict(cfg.get("concurrency") or {})
    thr_cfg = dict(cfg.get("throughput") or {})
    # inherit a few top-level knobs
    if "levels" in cfg and "levels" not in conc_cfg:
        conc_cfg["levels"] = cfg["levels"]
    if "reps" in cfg and "reps" not in lat_cfg:
        lat_cfg.setdefault("reps", cfg["reps"])
    if "drop_first_rep" in cfg:
        lat_cfg.setdefault("drop_first_rep", cfg["drop_first_rep"])
        conc_cfg.setdefault("drop_first_rep", cfg["drop_first_rep"])
        thr_cfg.setdefault("drop_first_rep", cfg["drop_first_rep"])

    results = {}
    infra = 0
    for name, fn, c in (
        ("latency", run_latency, lat_cfg),
        ("concurrency", run_concurrency, conc_cfg),
        ("throughput", run_throughput, thr_cfg),
    ):
        parts.append(name)
        r = fn(ep, out_dir / name, c)
        results[name] = {
            "status": r.status,
            "infra_errors": r.infra_errors,
            "summary": r.summary,
            "elapsed_s": r.elapsed_s,
        }
        infra += int(r.infra_errors or 0)
        write_json(out_dir / name / "lane_result.json", r.model_dump())

    statuses = [results[p]["status"] for p in parts]
    if any(s == "ERROR" for s in statuses) or infra:
        status = "ERROR" if infra else "FAIL"
    elif all(s == "PASS" for s in statuses):
        status = "PASS"
    else:
        status = "FAIL"

    summary = {
        "method": "composite_latency_concurrency_throughput",
        "mode": mode,
        "parts": parts,
        "results": {k: {"status": v["status"], "infra_errors": v["infra_errors"]} for k, v in results.items()},
        "detail": results,
    }
    write_json(out_dir / "summary.json", summary)
    return LaneResult(
        lane_id="perf",
        status=status,
        summary=summary,
        artifacts={"summary.json": str(out_dir / "summary.json")},
        infra_errors=infra,
        elapsed_s=time.perf_counter() - t0,
    )
