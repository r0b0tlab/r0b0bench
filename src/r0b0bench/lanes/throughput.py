from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

from r0b0bench.config import LaneResult, write_json
from r0b0bench.endpoint import Endpoint


def _chat(ep: Endpoint, prompt: str, max_tokens: int) -> dict[str, Any]:
    status, body, elapsed = ep.chat_completions(
        {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"thinking": False},
        }
    )
    usage = body.get("usage") or {}
    comp = int(usage.get("completion_tokens") or 0)
    prompt_tok = int(usage.get("prompt_tokens") or 0)
    return {
        "http_status": status,
        "ok": status == 200 and bool(body.get("choices")),
        "elapsed_s": elapsed,
        "completion_tokens": comp,
        "prompt_tokens": prompt_tok,
        "client_output_tokens_per_second": (comp / elapsed) if elapsed > 0 and comp else None,
        "server_prompt_tokens_per_second": (prompt_tok / elapsed) if elapsed > 0 and prompt_tok else None,
        "finish_reason": ((body.get("choices") or [{}])[0] or {}).get("finish_reason"),
    }


def run_throughput(ep: Endpoint, out_dir: Path, cfg: dict[str, Any]) -> LaneResult:
    """Decode + prefill throughput (C1).

    - decode: fixed long generation (default 2048 out) × N, report median client decode tok/s
    - prefill: long prompt / short generation, report server prompt tok/s proxy
    """
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    decode_out = int(cfg.get("decode_output_tokens") or 2048)
    decode_reps = int(cfg.get("decode_reps") or 5)
    drop_first = bool(cfg.get("drop_first_rep", True))
    prefill_target_tokens = int(cfg.get("prefill_prompt_tokens") or 14000)
    prefill_out = int(cfg.get("prefill_output_tokens") or 16)
    prefill_reps = int(cfg.get("prefill_reps") or 3)

    decode_prompt = str(
        cfg.get("decode_prompt")
        or (
            "Write a detailed technical essay on NVFP4 KV cache design for MoE LLMs. "
            "Continue with dense factual prose until the token budget is exhausted."
        )
    )
    # Build ~prefill_target_tokens via repeated filler (usage reports actual prompt_tokens).
    unit = "neutral archival observation about weather tools books roads and ordinary daily events. "
    # ~8–10 tokens/unit rough; overshoot then model counts exactly
    approx_units = max(50, prefill_target_tokens // 8)
    prefill_prompt = (
        "Summarize the following archive in one short sentence.\n\n" + (unit * approx_units)
    )

    decode_rows = []
    for rep in range(1, decode_reps + 1):
        row = {"rep": rep, "kind": "decode", **_chat(ep, decode_prompt, decode_out)}
        decode_rows.append(row)
        write_json(out_dir / f"decode-r{rep}.json", row)

    prefill_rows = []
    for rep in range(1, prefill_reps + 1):
        row = {"rep": rep, "kind": "prefill", **_chat(ep, prefill_prompt, prefill_out)}
        prefill_rows.append(row)
        write_json(out_dir / f"prefill-r{rep}.json", row)

    def stable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ok = [r for r in rows if r.get("ok")]
        return ok[1:] if drop_first and len(ok) > 1 else ok

    d_stable = stable(decode_rows)
    p_stable = stable(prefill_rows)
    d_tps = [float(r["client_output_tokens_per_second"]) for r in d_stable if r.get("client_output_tokens_per_second")]
    p_tps = [float(r["server_prompt_tokens_per_second"]) for r in p_stable if r.get("server_prompt_tokens_per_second")]
    p_prompt = [int(r["prompt_tokens"]) for r in p_stable if r.get("prompt_tokens")]

    summary = {
        "method": "openai_portable_c1_throughput",
        "backend": "openai_portable",
        "drop_first_rep": drop_first,
        "decode": {
            "output_tokens_requested": decode_out,
            "reps": decode_reps,
            "n_stable": len(d_stable),
            "median_client_output_tok_s": statistics.median(d_tps) if d_tps else None,
            "mean_client_output_tok_s": statistics.mean(d_tps) if d_tps else None,
            "rows": decode_rows,
        },
        "prefill": {
            "prompt_tokens_target": prefill_target_tokens,
            "output_tokens": prefill_out,
            "reps": prefill_reps,
            "n_stable": len(p_stable),
            "median_prompt_tokens": statistics.median(p_prompt) if p_prompt else None,
            "median_server_prompt_tok_s": statistics.median(p_tps) if p_tps else None,
            "mean_server_prompt_tok_s": statistics.mean(p_tps) if p_tps else None,
            "note": "prompt tok/s is e2e wall proxy (includes short decode); not pure prefill kernel metric",
            "rows": prefill_rows,
        },
        "failed": sum(1 for r in decode_rows + prefill_rows if not r.get("ok")),
    }
    write_json(out_dir / "summary.json", summary)
    infra = 1 if all(not r.get("ok") for r in decode_rows) else 0
    status = "PASS" if summary["failed"] == 0 and d_tps else ("ERROR" if infra else "FAIL")
    return LaneResult(
        lane_id="throughput",
        status=status,
        summary=summary,
        artifacts={"summary.json": str(out_dir / "summary.json")},
        infra_errors=infra,
        elapsed_s=time.perf_counter() - t0,
    )
