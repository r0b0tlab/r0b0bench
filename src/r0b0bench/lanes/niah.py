from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from r0b0bench.config import LaneResult, write_json
from r0b0bench.endpoint import Endpoint


def _find_subseq(hay: list[int], needle: list[int]) -> int:
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i : i + n] == needle:
            return i
    raise RuntimeError("marker token sequence not found in template")


def _depths_from_max(m: int, fractions: list[float], gen_reserve: int) -> list[int]:
    u = m - gen_reserve
    if u < 2048:
        raise RuntimeError(f"usable context U={u} too small (M={m}, reserve={gen_reserve})")
    depths = []
    for f in fractions:
        d = int(u * f)
        d = max(1024, min(d, u))
        depths.append(d)
    # unique sorted
    out = sorted(set(depths))
    while len(out) < 3 and out[-1] < u:
        out.append(min(u, out[-1] + max(1, (u - out[-1]) // 2)))
    return out[:3] if len(out) >= 3 else out


def run_niah(
    ep: Endpoint,
    out_dir: Path,
    cfg: dict[str, Any],
    tokenizer_path: str | None = None,
) -> LaneResult:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    m = ep.max_model_len()
    if m is None:
        return LaneResult(
            lane_id="niah",
            status="ERROR",
            summary={"error": "MAX_CONTEXT_UNKNOWN"},
            infra_errors=1,
            elapsed_s=time.perf_counter() - t0,
        )

    fractions = list(cfg.get("fractions") or [0.25, 0.5, 0.9])
    gen_reserve = int(cfg.get("generation_reserve") or 64)
    needle_frac = float(cfg.get("needle_fraction") or 0.5)
    code = str(cfg.get("code") or "R0B0-NIAH-7K3M")
    depths = _depths_from_max(m, fractions, gen_reserve)

    tok_path = tokenizer_path
    if not tok_path:
        return LaneResult(
            lane_id="niah",
            status="ERROR",
            summary={"error": "tokenizer path required (--tokenizer)", "max_model_len": m, "depths": depths},
            infra_errors=1,
            elapsed_s=time.perf_counter() - t0,
        )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=False, trust_remote_code=True)
    marker = "<<<R0B0BENCH_NIAH_SLOT>>>"
    instruction = (
        "You are given a long archival record. A secret verification code appears exactly once. "
        "Read carefully and remember the code.\n"
        f"{marker}\n"
        "Question: What is the secret verification code? Reply with only the code, nothing else."
    )
    messages = [{"role": "user", "content": instruction}]
    try:
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    base_ids = list(tok.encode(rendered, add_special_tokens=False))
    mids = tok.encode(marker, add_special_tokens=False)
    pos = _find_subseq(base_ids, mids)
    prefix, suffix = base_ids[:pos], base_ids[pos + len(mids) :]
    needle = tok.encode(f"\nIMPORTANT SECRET VERIFICATION CODE: {code}\n", add_special_tokens=False)
    unit = tok.encode(
        " The archival record contains neutral observations about weather, tools, books, roads, and ordinary daily events.",
        add_special_tokens=False,
    )

    rows = []
    for depth in depths:
        insertion = depth - len(prefix) - len(suffix)
        if insertion < len(needle):
            rows.append({"depth": depth, "passed": False, "status": "SKIP", "reason": "too_small"})
            continue
        remaining = insertion - len(needle)
        left_n = int(remaining * needle_frac)
        right_n = remaining - left_n
        left = (unit * ((left_n // len(unit)) + 3))[:left_n]
        right = (unit * ((right_n // len(unit)) + 3))[:right_n]
        input_ids = prefix + left + needle + right + suffix
        if len(input_ids) != depth:
            # rebuild
            insertion = depth - len(prefix) - len(suffix)
            remaining = insertion - len(needle)
            left_n = max(0, int(remaining * needle_frac))
            right_n = max(0, remaining - left_n)
            left = (unit * ((left_n // len(unit)) + 3))[:left_n]
            right = (unit * ((right_n // len(unit)) + 3))[:right_n]
            input_ids = prefix + left + needle + right + suffix
        status, body, elapsed = ep.completions(
            {"prompt": input_ids, "temperature": 0, "max_tokens": gen_reserve, "skip_special_tokens": True}
        )
        text = ""
        if status == 200 and body.get("choices"):
            ch = body["choices"][0]
            text = (ch.get("text") or "").strip()
        passed = code in text
        rows.append(
            {
                "depth": depth,
                "max_model_len": m,
                "prompt_tokens_constructed": len(input_ids),
                "prompt_sha256": hashlib.sha256(json.dumps(input_ids).encode()).hexdigest(),
                "needle_fraction_effective": (len(prefix) + left_n) / max(1, len(input_ids)),
                "elapsed_s": elapsed,
                "http_status": status,
                "response_text": text,
                "passed": passed,
                "usage": body.get("usage"),
            }
        )

    summary = {
        "max_model_len": m,
        "depths": depths,
        "fractions": fractions,
        "generation_reserve": gen_reserve,
        "code": code,
        "pass_count": sum(1 for r in rows if r.get("passed")),
        "total": len(rows),
        "all_passed": bool(rows) and all(r.get("passed") for r in rows),
        "depth_policy": "frac_of_max_context",
    }
    write_json(out_dir / "niah.json", {"summary": summary, "rows": rows})
    infra = 1 if any(r.get("http_status") not in (200, None) and r.get("status") != "SKIP" for r in rows) else 0
    # treat all HTTP non-200 as infra
    if any(r.get("http_status") not in (200, None) for r in rows if r.get("status") != "SKIP"):
        infra = 1
    status = "PASS" if summary["all_passed"] else ("ERROR" if infra else "FAIL")
    return LaneResult(
        lane_id="niah",
        status=status,
        summary=summary,
        artifacts={"niah.json": str(out_dir / "niah.json")},
        infra_errors=infra,
        elapsed_s=time.perf_counter() - t0,
    )
