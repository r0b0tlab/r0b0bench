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
    out = sorted(set(depths))
    while len(out) < 3 and out[-1] < u:
        out.append(min(u, out[-1] + max(1, (u - out[-1]) // 2)))
    return out[:3] if len(out) >= 3 else out


def _load_encode_fn(tokenizer_path: str | None):
    """Return encode(text)->list[int]. Prefer tokenizers lib (robust); fall back to HF."""
    if tokenizer_path:
        tok_json = Path(tokenizer_path)
        if tok_json.is_dir():
            cand = tok_json / "tokenizer.json"
        else:
            cand = tok_json
        if cand.exists():
            try:
                from tokenizers import Tokenizer

                tok = Tokenizer.from_file(str(cand))

                def encode(text: str) -> list[int]:
                    return list(tok.encode(text).ids)

                return encode, {"backend": "tokenizers", "path": str(cand)}
            except Exception:
                pass
        try:
            from transformers import AutoTokenizer

            hf = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=False)

            def encode_hf(text: str) -> list[int]:
                return list(hf.encode(text, add_special_tokens=False))

            return encode_hf, {"backend": "transformers", "path": tokenizer_path}
        except Exception as exc:
            raise RuntimeError(f"failed to load tokenizer from {tokenizer_path}: {exc}") from exc
    raise RuntimeError("tokenizer path required for NIAH (--tokenizer pointing at model/tokenizer.json dir)")


def run_niah(
    ep: Endpoint,
    out_dir: Path,
    cfg: dict[str, Any],
    tokenizer_path: str | None = None,
) -> LaneResult:
    """Max-context NIAH: depths = fractions of (max_model_len - reserve).

    Uses server /v1/chat/completions/render for exact chat-template token parity when available.
    """
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
    # Always max-context against advertised max_model_len (NIAH is the capacity test).
    depths = _depths_from_max(m, fractions, gen_reserve)
    kv_tokens = ep.kv_cache_size_tokens()
    request_timeout = float(cfg.get("request_timeout_s") or 7200)
    stop_on_infra = bool(cfg.get("stop_on_infra_error", True))

    try:
        encode, tok_meta = _load_encode_fn(tokenizer_path)
    except Exception as exc:
        return LaneResult(
            lane_id="niah",
            status="ERROR",
            summary={
                "error": str(exc),
                "max_model_len": m,
                "kv_cache_size_tokens": kv_tokens,
                "depths": depths,
            },
            infra_errors=1,
            elapsed_s=time.perf_counter() - t0,
        )

    # Prefer context-stable marker without angle-bracket BPE splits.
    marker = str(cfg.get("marker") or "R0B0BENCH_NIAH_SLOT")
    instruction = (
        "You are given a long archival record. A secret verification code appears exactly once. "
        "Read carefully and remember the code.\n"
        f"{marker}\n"
        "Question: What is the secret verification code? Reply with only the code, nothing else."
    )
    messages = [{"role": "user", "content": instruction}]
    template_source = "unknown"
    try:
        rendered = ep.chat_render(messages)
        base_ids = list(rendered.get("token_ids") or [])
        if not base_ids:
            raise RuntimeError("render returned empty token_ids")
        template_source = "server:/v1/chat/completions/render"
    except Exception as render_exc:
        # Fallback: HF chat template if available
        try:
            from transformers import AutoTokenizer

            hf = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            try:
                import os

                env_kwargs = json.loads(os.environ.get("R0B0BENCH_CHAT_TEMPLATE_KWARGS") or "{}")
                tmpl_kw = {}
                if "enable_thinking" in env_kwargs:
                    tmpl_kw["enable_thinking"] = env_kwargs["enable_thinking"]
                text = hf.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **tmpl_kw
                )
            except TypeError:
                text = hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            base_ids = list(hf.encode(text, add_special_tokens=False))
            template_source = f"transformers_chat_template (render_failed={render_exc})"
        except Exception as exc:
            return LaneResult(
                lane_id="niah",
                status="ERROR",
                summary={
                    "error": f"template_render_failed: render={render_exc}; hf={exc}",
                    "max_model_len": m,
                    "depths": depths,
                },
                infra_errors=1,
                elapsed_s=time.perf_counter() - t0,
            )

    mids = encode(marker)
    try:
        pos = _find_subseq(base_ids, mids)
    except RuntimeError:
        # last resort: search bare marker without surrounding punctuation variants
        raise_err = None
        pos = None
        for cand in (marker, f" {marker}", f"{marker}\n"):
            try:
                mids = encode(cand)
                pos = _find_subseq(base_ids, mids)
                break
            except Exception as e:  # noqa: BLE001
                raise_err = e
        if pos is None:
            return LaneResult(
                lane_id="niah",
                status="ERROR",
                summary={
                    "error": f"marker_not_found_in_template: {raise_err}",
                    "template_source": template_source,
                    "marker": marker,
                },
                infra_errors=1,
                elapsed_s=time.perf_counter() - t0,
            )

    prefix, suffix = base_ids[:pos], base_ids[pos + len(mids) :]
    needle = encode(f"\nIMPORTANT SECRET VERIFICATION CODE: {code}\n")
    unit = encode(
        " The archival record contains neutral observations about weather, tools, books, roads, and ordinary daily events."
    )

    long_ep = ep.with_timeout(request_timeout)
    rows = []
    try:
        for depth in depths:
            insertion = depth - len(prefix) - len(suffix)
            if insertion < len(needle):
                rows.append({"depth": depth, "passed": False, "status": "SKIP", "reason": "too_small"})
                continue
            remaining = insertion - len(needle)
            left_n = int(remaining * needle_frac)
            right_n = remaining - left_n
            left = (unit * ((left_n // max(1, len(unit))) + 3))[:left_n]
            right = (unit * ((right_n // max(1, len(unit))) + 3))[:right_n]
            input_ids = prefix + left + needle + right + suffix
            if len(input_ids) != depth:
                if len(input_ids) < depth:
                    pad = unit * ((depth - len(input_ids)) // max(1, len(unit)) + 2)
                    input_ids = (input_ids + pad)[:depth]
                else:
                    input_ids = input_ids[:depth]
            t1 = time.perf_counter()
            try:
                status, body, elapsed = long_ep.completions(
                    {
                        "prompt": input_ids,
                        "temperature": 0,
                        "max_tokens": gen_reserve,
                        "skip_special_tokens": True,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "depth": depth,
                        "max_model_len": m,
                        "prompt_tokens_constructed": len(input_ids),
                        "elapsed_s": time.perf_counter() - t1,
                        "http_status": None,
                        "status": "ERROR",
                        "response_text": str(exc)[:1000],
                        "passed": False,
                        "infra_errors": 1,
                    }
                )
                if stop_on_infra:
                    break
                continue
            text = ""
            finish = None
            if status == 200 and body.get("choices"):
                ch = body["choices"][0]
                text = (ch.get("text") or "").strip()
                finish = ch.get("finish_reason")
            passed = code in text
            row_status = "PASS" if passed else ("ERROR" if status != 200 else "FAIL")
            rows.append(
                {
                    "depth": depth,
                    "max_model_len": m,
                    "fraction_of_usable": round(depth / max(1, m - gen_reserve), 4),
                    "prompt_tokens_constructed": len(input_ids),
                    "prompt_sha256": hashlib.sha256(json.dumps(input_ids).encode()).hexdigest(),
                    "needle_fraction_effective": (len(prefix) + left_n) / max(1, len(input_ids)),
                    "elapsed_s": elapsed,
                    "http_status": status,
                    "finish_reason": finish,
                    "response_text": text[:500],
                    "passed": passed,
                    "status": row_status,
                    "usage": body.get("usage"),
                    "infra_errors": 0 if status == 200 else 1,
                }
            )
            write_json(out_dir / f"depth-{depth}.json", rows[-1])
            if status != 200 and stop_on_infra:
                break
    finally:
        long_ep.close()

    measured = [r for r in rows if r.get("status") != "SKIP"]
    summary = {
        "max_model_len": m,
        "kv_cache_size_tokens": kv_tokens,
        "depths": depths,
        "fractions": fractions,
        "generation_reserve": gen_reserve,
        "code": code,
        "marker": marker,
        "template_source": template_source,
        "tokenizer": tok_meta,
        "depth_policy": "frac_of_max_context",
        "note": (
            "Depths are always 25/50/90% of (max_model_len - generation_reserve). "
            "This is the standard max-context capacity package; physical KV shortfall surfaces as infra ERROR."
        ),
        "pass_count": sum(1 for r in measured if r.get("passed")),
        "total": len(measured),
        "all_passed": bool(measured) and all(r.get("passed") for r in measured) and len(measured) >= len(depths),
        "rows": rows,
    }
    write_json(out_dir / "niah.json", summary)
    infra = sum(int(r.get("infra_errors") or 0) for r in rows)
    if any(r.get("http_status") not in (200, None) for r in rows if r.get("status") != "SKIP"):
        infra = max(infra, 1)
    status = "PASS" if summary["all_passed"] and infra == 0 else ("ERROR" if infra else "FAIL")
    return LaneResult(
        lane_id="niah",
        status=status,
        summary=summary,
        artifacts={"niah.json": str(out_dir / "niah.json")},
        infra_errors=infra,
        elapsed_s=time.perf_counter() - t0,
    )
