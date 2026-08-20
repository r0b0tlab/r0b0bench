from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from r0b0bench.config import LaneResult, write_json
from r0b0bench.endpoint import Endpoint
from r0b0bench.thinking import effective_max_tokens


def _one(ep: Endpoint, prompt: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    status, body, elapsed = ep.chat_completions(
        {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    )
    usage = body.get("usage") or {}
    comp = int(usage.get("completion_tokens") or 0)
    return {
        "http_status": status,
        "ok": status == 200 and bool(body.get("choices")),
        "elapsed_s": elapsed,
        "completion_tokens": comp,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "client_output_tokens_per_second": (comp / elapsed) if elapsed > 0 and comp else 0.0,
    }


def run_concurrency(ep: Endpoint, out_dir: Path, cfg: dict[str, Any]) -> LaneResult:
    """Concurrency ladder (default C1/C2/C4/C6) with fixed output budget."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = [int(x) for x in (cfg.get("levels") or [1, 2, 4, 6])]
    reps = int(cfg.get("reps") or 3)
    drop_first = bool(cfg.get("drop_first_rep", True))
    out_tok = effective_max_tokens(int(cfg.get("output_tokens") or 512), "concurrency")
    temperature = float(cfg.get("temperature") or 0)
    # short prompt; concurrency measures decode scaling under load
    prompt = str(
        cfg.get("prompt")
        or (
            "Count upward from 1 using space-separated integers until you approach the token budget. "
            "Do not stop early. Digits and spaces only."
        )
    )

    rows = []
    for c in levels:
        rep_stats = []
        for rep in range(1, reps + 1):
            # enough in-flight work to saturate level c
            n_req = max(c * 2, c + 1)
            t_batch = time.perf_counter()
            results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=c) as pool:
                futs = [pool.submit(_one, ep, prompt, out_tok, temperature) for _ in range(n_req)]
                for f in as_completed(futs):
                    results.append(f.result())
            wall = time.perf_counter() - t_batch
            ok = [r for r in results if r["ok"]]
            fail = len(results) - len(ok)
            comp = sum(r["completion_tokens"] for r in ok)
            client_tps = [r["client_output_tokens_per_second"] for r in ok if r["client_output_tokens_per_second"]]
            rep_stats.append(
                {
                    "rep": rep,
                    "concurrency": c,
                    "n_requests": n_req,
                    "completed": len(ok),
                    "failed": fail,
                    "wall_s": wall,
                    "aggregate_output_tok_s": (comp / wall) if wall > 0 else 0.0,
                    "median_client_output_tok_s": statistics.median(client_tps) if client_tps else None,
                    "mean_e2el_ms": statistics.mean(r["elapsed_s"] * 1000 for r in ok) if ok else None,
                }
            )
            write_json(out_dir / f"c{c}-r{rep}.json", rep_stats[-1])
        stable = rep_stats[1:] if drop_first and len(rep_stats) > 1 else rep_stats

        def mean_key(key: str) -> float | None:
            vals = [float(r[key]) for r in stable if r.get(key) is not None]
            return statistics.mean(vals) if vals else None

        rows.append(
            {
                "concurrency": c,
                "output_tokens": out_tok,
                "repetitions_present": len(rep_stats),
                "warmup_rep_dropped": 1 if drop_first and len(rep_stats) > 1 else 0,
                "aggregate_output_tok_s": mean_key("aggregate_output_tok_s"),
                "median_client_output_tok_s": mean_key("median_client_output_tok_s"),
                "mean_e2el_ms": mean_key("mean_e2el_ms"),
                "completed": sum(int(r["completed"]) for r in stable),
                "failed": sum(int(r["failed"]) for r in stable),
            }
        )

    summary = {
        "method": "openai_portable_concurrency_ladder",
        "backend": "openai_portable",
        "levels": levels,
        "reps": reps,
        "drop_first_rep": drop_first,
        "output_tokens": out_tok,
        "rows": rows,
        "total_failed": sum(int(r["failed"] or 0) for r in rows),
    }
    write_json(out_dir / "summary.json", summary)
    infra = 1 if summary["total_failed"] and all(r["completed"] == 0 for r in rows) else 0
    status = "PASS" if summary["total_failed"] == 0 else ("ERROR" if infra else "FAIL")
    return LaneResult(
        lane_id="concurrency",
        status=status,
        summary=summary,
        artifacts={"summary.json": str(out_dir / "summary.json")},
        infra_errors=infra,
        elapsed_s=time.perf_counter() - t0,
    )
