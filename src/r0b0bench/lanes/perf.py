from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from r0b0bench.config import LaneResult, write_json
from r0b0bench.endpoint import Endpoint


def _one_request(ep: Endpoint, prompt: str, max_tokens: int) -> dict[str, Any]:
    status, body, elapsed = ep.chat_completions(
        {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    )
    usage = body.get("usage") or {}
    return {
        "http_status": status,
        "elapsed_s": elapsed,
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "ok": status == 200 and bool(body.get("choices")),
    }


def run_perf(ep: Endpoint, out_dir: Path, cfg: dict[str, Any]) -> LaneResult:
    """Portable OpenAI-chat concurrency sweep (not vllm bench exec)."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = [int(x) for x in (cfg.get("levels") or [1, 2, 4, 8, 16])]
    reps = int(cfg.get("reps") or 3)
    drop_first = bool(cfg.get("drop_first_rep", True))
    out_tok = int(cfg.get("output_tokens") or 256)
    # approximate input with repeated filler (not exact token count without tokenizer)
    filler = ("benchmark filler token sequence for throughput measurement. " * 80).strip()
    prompt = filler + "\nReply with a single word: OK"

    rows = []
    for c in levels:
        rep_stats = []
        for rep in range(1, reps + 1):
            n_req = max(8, c * 4)
            t_batch = time.perf_counter()
            results = []
            with ThreadPoolExecutor(max_workers=c) as pool:
                futs = [pool.submit(_one_request, ep, prompt, out_tok) for _ in range(n_req)]
                for f in as_completed(futs):
                    results.append(f.result())
            wall = time.perf_counter() - t_batch
            ok = [r for r in results if r["ok"]]
            fail = len(results) - len(ok)
            comp = sum(r["completion_tokens"] for r in ok)
            out_tps = comp / wall if wall > 0 else 0.0
            rps = len(ok) / wall if wall > 0 else 0.0
            lat = [r["elapsed_s"] * 1000 for r in ok]
            rep_stats.append(
                {
                    "rep": rep,
                    "concurrency": c,
                    "n_requests": n_req,
                    "completed": len(ok),
                    "failed": fail,
                    "wall_s": wall,
                    "output_throughput_tok_s": out_tps,
                    "request_throughput_rps": rps,
                    "mean_e2el_ms": statistics.mean(lat) if lat else None,
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
                "repetitions_present": len(rep_stats),
                "warmup_rep_dropped": 1 if drop_first and len(rep_stats) > 1 else 0,
                "output_throughput_tok_s": mean_key("output_throughput_tok_s"),
                "request_throughput_rps": mean_key("request_throughput_rps"),
                "mean_e2el_ms": mean_key("mean_e2el_ms"),
                "completed": sum(int(r["completed"]) for r in stable),
                "failed": sum(int(r["failed"]) for r in stable),
            }
        )

    summary = {
        "method": f"openai_portable_chat approx_in/out={out_tok}; levels={levels}; reps={reps} drop_first={drop_first}",
        "backend": "openai_portable",
        "rows": rows,
        "total_failed": sum(int(r["failed"] or 0) for r in rows),
    }
    write_json(out_dir / "summary.json", summary)
    infra = 1 if summary["total_failed"] and all(r["completed"] == 0 for r in rows) else 0
    status = "PASS" if summary["total_failed"] == 0 else ("ERROR" if infra else "FAIL")
    return LaneResult(
        lane_id="perf",
        status=status,
        summary=summary,
        artifacts={"summary.json": str(out_dir / "summary.json")},
        infra_errors=infra,
        elapsed_s=time.perf_counter() - t0,
    )
