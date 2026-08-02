from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

from r0b0bench.config import LaneResult, write_json
from r0b0bench.endpoint import Endpoint


def run_latency(ep: Endpoint, out_dir: Path, cfg: dict[str, Any]) -> LaneResult:
    """C1 streaming latency: TTFT + ITL + e2e. Non-stream e2e backup."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    reps = int(cfg.get("reps") or 5)
    drop_first = bool(cfg.get("drop_first_rep", True))
    max_tokens = int(cfg.get("output_tokens") or 128)
    prompt = str(
        cfg.get("prompt")
        or "Write a short factual paragraph about copper conductivity. Keep it under 80 words."
    )
    temperature = float(cfg.get("temperature") or 0)

    stream_rows = []
    for rep in range(1, reps + 1):
        status, stats = ep.chat_completions_stream(
            {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"thinking": False},
            }
        )
        row = {"rep": rep, "http_status": status, **stats}
        stream_rows.append(row)
        write_json(out_dir / f"stream-r{rep}.json", row)

    nonstream_rows = []
    for rep in range(1, min(3, reps) + 1):
        status, body, elapsed = ep.chat_completions(
            {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"thinking": False},
            }
        )
        usage = body.get("usage") or {}
        nonstream_rows.append(
            {
                "rep": rep,
                "http_status": status,
                "ok": status == 200 and bool(body.get("choices")),
                "elapsed_s": elapsed,
                "e2el_ms": elapsed * 1000.0,
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            }
        )
        write_json(out_dir / f"nonstream-r{rep}.json", nonstream_rows[-1])

    def stable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return rows[1:] if drop_first and len(rows) > 1 else rows

    def mean_of(rows: list[dict[str, Any]], key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None and r.get("ok", True)]
        return statistics.mean(vals) if vals else None

    s = stable([r for r in stream_rows if r.get("ok")])
    ns = stable([r for r in nonstream_rows if r.get("ok")])
    summary = {
        "method": "openai_chat_stream_c1_latency",
        "reps": reps,
        "drop_first_rep": drop_first,
        "output_tokens": max_tokens,
        "stream": {
            "n_stable": len(s),
            "ttft_ms_mean": mean_of(s, "ttft_ms"),
            "itl_ms_mean": mean_of(s, "itl_ms_mean"),
            "itl_ms_p50_mean": mean_of(s, "itl_ms_p50"),
            "itl_ms_p95_mean": mean_of(s, "itl_ms_p95"),
            "e2el_ms_mean": mean_of(s, "e2el_ms"),
            "rows": stream_rows,
        },
        "nonstream": {
            "n_stable": len(ns),
            "e2el_ms_mean": mean_of(ns, "e2el_ms"),
            "rows": nonstream_rows,
        },
        "failed": sum(1 for r in stream_rows if not r.get("ok"))
        + sum(1 for r in nonstream_rows if not r.get("ok")),
    }
    write_json(out_dir / "summary.json", summary)
    infra = 1 if all(not r.get("ok") for r in stream_rows) else 0
    status = "PASS" if summary["failed"] == 0 and s else ("ERROR" if infra else "FAIL")
    return LaneResult(
        lane_id="latency",
        status=status,
        summary=summary,
        artifacts={"summary.json": str(out_dir / "summary.json")},
        infra_errors=infra,
        elapsed_s=time.perf_counter() - t0,
    )
