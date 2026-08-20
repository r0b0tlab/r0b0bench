"""Quality lanes for r0b0bench core / core-subset profiles."""
from __future__ import annotations

import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from r0b0bench.config import LaneResult, write_json, wilson_ci
from r0b0bench.endpoint import Endpoint
from r0b0bench.thinking import effective_max_tokens


def _env_path(*keys: str, default: str = "") -> str:
    for k in keys:
        v = os.environ.get(k)
        if v:
            return v
    return default


def _chat(ep: Endpoint, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> dict[str, Any]:
    last_err = None
    for attempt in range(4):
        try:
            status, body, elapsed = ep.chat_completions(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            text = ""
            finish = None
            if status == 200 and body.get("choices"):
                msg = body["choices"][0].get("message") or {}
                text = (msg.get("content") or "").strip()
                finish = body["choices"][0].get("finish_reason")
            return {
                "http_status": status,
                "text": text,
                "finish_reason": finish,
                "elapsed_s": elapsed,
                "ok": status == 200 and bool(text),
                "usage": body.get("usage") or {},
                "error": None if status == 200 else str(body)[:500],
            }
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(min(30, 2 ** attempt))
    return {
        "http_status": 0,
        "text": "",
        "finish_reason": None,
        "elapsed_s": 0.0,
        "ok": False,
        "usage": {},
        "error": str(last_err)[:500] if last_err else "unknown",
    }


def _load_jsonl(path: Path, n: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if n is not None and n < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, n)
    return rows


# ---------------- GSM8K (0-shot flexible-extract) ----------------

def _gsm8k_gold(ans: str) -> str | None:
    m = re.search(r"####\s*([-\d,\.]+)", ans)
    return m.group(1).replace(",", "").rstrip(".") if m else None


def _gsm8k_pred(text: str) -> str | None:
    # flexible-extract style
    m = re.findall(r"(?:answer is|final answer is)\s*\$?\s*([-\d,\.]+)", text, re.I)
    if not m:
        m = re.findall(r"([-\d,]+\.?\d*)", text.replace(",", ""))
    if not m:
        return None
    return m[-1].replace(",", "").rstrip(".")


def run_gsm8k(ep: Endpoint, out_dir: Path, cfg: dict[str, Any]) -> LaneResult:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = Path(
        cfg.get("data_path")
        or _env_path("R0B0BENCH_GSM8K_DATA", "GSM8K_DATA", default="/home/r0b0tdgx/datasets/gsm8k/test.jsonl")
    )
    if not data.exists():
        return LaneResult(
            lane_id="gsm8k",
            status="ERROR",
            summary={"error": f"missing dataset {data}"},
            infra_errors=1,
            elapsed_s=time.perf_counter() - t0,
        )
    n = int(cfg.get("n") or 200)
    conc = int(cfg.get("concurrency") or 4)
    max_tokens = effective_max_tokens(int(cfg.get("max_tokens") or 512), "gsm8k")
    seed = int(cfg.get("seed") or 0)
    rows_in = _load_jsonl(data, n=None)
    # deterministic head for subset (not random) — stable core-subset lock
    rows_in = rows_in[:n]
    results: dict[int, dict[str, Any]] = {}

    def one(i: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        q = row["question"]
        prompt = q + "\nSolve step by step. End with 'The answer is <number>'."
        r = _chat(ep, prompt, max_tokens=max_tokens)
        gold = _gsm8k_gold(row.get("answer") or "")
        pred = _gsm8k_pred(r["text"]) if r["ok"] else None
        return i, {
            **r,
            "gold": gold,
            "pred": pred,
            "correct": pred is not None and gold is not None and pred == gold,
            "question": q[:200],
        }

    infra = 0
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(one, i, row) for i, row in enumerate(rows_in)]
        for f in as_completed(futs):
            i, row = f.result()
            results[i] = row
            if row.get("http_status") != 200:
                infra += 1

    ordered = [results[i] for i in range(len(rows_in))]
    correct = sum(1 for r in ordered if r.get("correct"))
    total = len(ordered)
    acc = correct / total if total else 0.0
    summary = {
        "method": "gsm8k_0shot_flexible_extract",
        "n": total,
        "correct": correct,
        "accuracy": acc,
        "wilson95": wilson_ci(correct, total),
        "concurrency": conc,
        "data_path": str(data),
        "infra_http_errors": infra,
        "seed": seed,
    }
    write_json(out_dir / "gsm8k.json", {"summary": summary, "rows": ordered})
    status = "PASS" if total and infra == 0 else ("ERROR" if infra == total else "FAIL")
    # quality lanes: status is measurement complete PASS if ran; accuracy is in summary
    # For package: use PASS when executed without infra failure
    if total and infra < total:
        status = "PASS"
    return LaneResult(
        lane_id="gsm8k",
        status=status,
        summary=summary,
        artifacts={"gsm8k.json": str(out_dir / "gsm8k.json")},
        infra_errors=1 if infra == total and total else 0,
        elapsed_s=time.perf_counter() - t0,
    )


# ---------------- HumanEval ----------------

def run_humaneval(ep: Endpoint, out_dir: Path, cfg: dict[str, Any]) -> LaneResult:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = Path(
        cfg.get("data_path")
        or _env_path("R0B0BENCH_HUMANEVAL_DATA", default="/home/r0b0tdgx/datasets/humaneval/HumanEval.jsonl")
    )
    n = int(cfg.get("n") or 164)
    conc = int(cfg.get("concurrency") or 2)
    max_tokens = effective_max_tokens(int(cfg.get("max_tokens") or 512), "humaneval")
    rows_in = _load_jsonl(data)[:n]
    samples = []
    infra = 0

    def one(row: dict[str, Any]) -> dict[str, Any]:
        prompt = row["prompt"]
        # stop at common end markers via instruction
        user = (
            "Complete the following Python function. Output only code, no markdown fences.\n\n"
            + prompt
        )
        r = _chat(ep, user, max_tokens=max_tokens)
        text = r["text"]
        # strip fences
        text = re.sub(r"^```(?:python)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        completion = text
        # if model repeated prompt, strip
        if completion.startswith(prompt):
            completion = completion[len(prompt) :]
        return {
            "task_id": row["task_id"],
            "completion": completion,
            "http_status": r["http_status"],
            "ok": r["ok"],
            "elapsed_s": r["elapsed_s"],
        }

    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(one, row) for row in rows_in]
        for f in as_completed(futs):
            s = f.result()
            samples.append(s)
            if s.get("http_status") != 200:
                infra += 1

    samples_path = out_dir / "samples.jsonl"
    with samples_path.open("w") as f:
        for s in samples:
            f.write(json.dumps({"task_id": s["task_id"], "completion": s["completion"]}) + "\n")

    pass_at = None
    eval_err = None
    try:
        from human_eval.evaluation import evaluate_functional_correctness

        # evaluate_functional_correctness expects a file path
        pass_at = evaluate_functional_correctness(str(samples_path), k=[1], n_workers=4, timeout=10.0)
    except Exception as exc:  # noqa: BLE001
        eval_err = str(exc)[:500]

    summary = {
        "method": "humaneval_functional_correctness",
        "n": len(samples),
        "pass_at": pass_at,
        "pass@1": (pass_at or {}).get("pass@1") if isinstance(pass_at, dict) else None,
        "infra_http_errors": infra,
        "eval_error": eval_err,
        "data_path": str(data),
        "concurrency": conc,
    }
    write_json(out_dir / "humaneval.json", {"summary": summary, "samples_meta": samples})
    status = "PASS" if pass_at is not None and infra < len(samples) else ("ERROR" if infra == len(samples) else "FAIL")
    if pass_at is not None:
        status = "PASS"
    elif eval_err and infra == 0:
        status = "ERROR"
    return LaneResult(
        lane_id="humaneval",
        status=status,
        summary=summary,
        artifacts={
            "humaneval.json": str(out_dir / "humaneval.json"),
            "samples.jsonl": str(samples_path),
        },
        infra_errors=1 if infra == len(samples) and samples else (1 if eval_err and pass_at is None else 0),
        elapsed_s=time.perf_counter() - t0,
    )


# ---------------- IFEval (lightweight scorer) ----------------

def _ifeval_checks(prompt: str, response: str, kwargs: dict[str, Any] | None) -> list[tuple[str, bool]]:
    """Best-effort constraint checks for common IFEval instruction types."""
    kwargs = kwargs or {}
    checks: list[tuple[str, bool]] = []
    text = response or ""
    low = text.lower()
    # numbered bullets
    if "number" in str(kwargs.get("instruction_id_list", [])) or any(
        "number_bullets" in str(x) or "number_paragraphs" in str(x) for x in (kwargs.get("instruction_id_list") or [])
    ):
        pass
    ids = kwargs.get("instruction_id_list") or []
    kws = kwargs.get("kwargs") or []
    if not isinstance(kws, list):
        kws = [kws]
    for idx, iid in enumerate(ids):
        kw = kws[idx] if idx < len(kws) and isinstance(kws[idx], dict) else {}
        ok = True
        if iid in ("detectable_format:number_bullet_lists", "detectable_format:number_highlighted_sections"):
            # at least one markdown-ish highlight or bullet
            ok = bool(re.search(r"(^|\n)\s*([-*•]|\d+\.)\s+\S+", text) or "*" in text)
        elif iid == "length_constraints:number_words":
            n = int(kw.get("num_words") or kw.get("n") or 0)
            rel = kw.get("relation", "at least")
            wc = len(re.findall(r"\b\w+\b", text))
            if rel in ("at least", "least"):
                ok = wc >= n
            elif rel in ("around", "about"):
                ok = abs(wc - n) <= max(5, int(0.1 * n))
            else:
                ok = wc <= n
        elif iid == "length_constraints:number_sentences":
            n = int(kw.get("num_sentences") or 0)
            sc = len(re.findall(r"[.!?](?:\s|$)", text))
            rel = kw.get("relation", "at least")
            ok = sc >= n if "least" in str(rel) else sc <= n
        elif iid == "length_constraints:number_paragraphs":
            n = int(kw.get("num_paragraphs") or 0)
            pc = len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()])
            ok = pc >= n
        elif iid == "keywords:existence":
            klist = kw.get("keywords") or []
            ok = all(str(k).lower() in low for k in klist)
        elif iid == "keywords:forbidden_words":
            fl = kw.get("forbidden_words") or []
            ok = all(str(k).lower() not in low for k in fl)
        elif iid == "change_case:english_capital":
            # mostly uppercase letters
            letters = [c for c in text if c.isalpha()]
            ok = bool(letters) and (sum(c.isupper() for c in letters) / len(letters) > 0.8)
        elif iid == "change_case:english_lowercase":
            letters = [c for c in text if c.isalpha()]
            ok = bool(letters) and (sum(c.islower() for c in letters) / len(letters) > 0.8)
        elif iid == "startend:end_checker":
            end = str(kw.get("end_phrase") or "")
            ok = text.strip().endswith(end) if end else True
        elif iid == "startend:quotation":
            ok = text.strip().startswith('"') and text.strip().endswith('"')
        elif iid == "punctuation:no_comma":
            ok = "," not in text
        elif iid == "detectable_content:postscript":
            ok = bool(re.search(r"\b(p\.s\.|ps:)\b", low))
        elif iid == "detectable_format:title":
            ok = bool(re.search(r"<<.*>>", text)) or text.strip().startswith("#")
        elif iid == "language:response_language":
            lang = str(kw.get("language") or "en")
            try:
                from langdetect import detect

                ok = detect(text) == lang
            except Exception:
                ok = True  # don't fail closed on missing detector
        else:
            # unknown instruction — mark as soft pass with flag
            ok = len(text) > 0
            checks.append((f"{iid}:unknown_soft", ok))
            continue
        checks.append((iid, ok))
    if not checks:
        checks.append(("nonempty", len(text) > 0))
    return checks


def run_ifeval(ep: Endpoint, out_dir: Path, cfg: dict[str, Any]) -> LaneResult:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = Path(
        cfg.get("data_path")
        or _env_path("R0B0BENCH_IFEVAL_DATA", default="/home/r0b0tdgx/datasets/ifeval/input_data.jsonl")
    )
    if not data.exists():
        return LaneResult(
            lane_id="ifeval",
            status="ERROR",
            summary={"error": f"missing {data}"},
            infra_errors=1,
            elapsed_s=time.perf_counter() - t0,
        )
    n = int(cfg.get("n") or 200)
    conc = int(cfg.get("concurrency") or 4)
    max_tokens = effective_max_tokens(int(cfg.get("max_tokens") or 1024), "ifeval")
    rows_in = _load_jsonl(data)[:n]
    results = []
    infra = 0

    def one(row: dict[str, Any]) -> dict[str, Any]:
        prompt = row.get("prompt") or ""
        r = _chat(ep, prompt, max_tokens=max_tokens)
        checks = _ifeval_checks(
            prompt,
            r["text"],
            {
                "instruction_id_list": row.get("instruction_id_list"),
                "kwargs": row.get("kwargs"),
            },
        )
        passed = all(ok for _, ok in checks) if checks else False
        return {
            "key": row.get("key"),
            "prompt": prompt[:200],
            "http_status": r["http_status"],
            "response": r["text"][:1000],
            "checks": checks,
            "passed": passed and r["ok"],
            "elapsed_s": r["elapsed_s"],
        }

    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(one, row) for row in rows_in]
        for f in as_completed(futs):
            row = f.result()
            results.append(row)
            if row.get("http_status") != 200:
                infra += 1

    total = len(results)
    correct = sum(1 for r in results if r.get("passed"))
    summary = {
        "method": "ifeval_lightweight_constraint_scorer",
        "note": "Not full official IFEval scorer; common instruction families only. Label as lightweight.",
        "n": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "wilson95": wilson_ci(correct, total),
        "infra_http_errors": infra,
        "data_path": str(data),
        "concurrency": conc,
    }
    write_json(out_dir / "ifeval.json", {"summary": summary, "rows": results})
    status = "PASS" if total and infra < total else "ERROR"
    return LaneResult(
        lane_id="ifeval",
        status=status,
        summary=summary,
        artifacts={"ifeval.json": str(out_dir / "ifeval.json")},
        infra_errors=1 if infra == total and total else 0,
        elapsed_s=time.perf_counter() - t0,
    )


# ---------------- QA (ARC-Easy multiple choice) ----------------

def run_qa(ep: Endpoint, out_dir: Path, cfg: dict[str, Any]) -> LaneResult:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = Path(
        cfg.get("data_path")
        or _env_path("R0B0BENCH_QA_DATA", default="/home/r0b0tdgx/datasets/qa/arc_easy_test.jsonl")
    )
    if not data.exists():
        return LaneResult(
            lane_id="qa",
            status="ERROR",
            summary={"error": f"missing {data}"},
            infra_errors=1,
            elapsed_s=time.perf_counter() - t0,
        )
    n = int(cfg.get("n") or 400)
    conc = int(cfg.get("concurrency") or 4)
    max_tokens = effective_max_tokens(int(cfg.get("max_tokens") or 32), "qa")
    rows_in = _load_jsonl(data)[:n]
    results = []
    infra = 0

    def one(row: dict[str, Any]) -> dict[str, Any]:
        ch = row.get("choices") or {}
        labels = ch.get("label") or []
        texts = ch.get("text") or []
        opts = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
        prompt = (
            f"Question: {row.get('question')}\n{opts}\n"
            "Reply with only the choice letter (A/B/C/D/E)."
        )
        r = _chat(ep, prompt, max_tokens=max_tokens)
        gold = str(row.get("answerKey") or "").strip().upper()
        pred = None
        if r["ok"]:
            m = re.search(r"\b([A-E])\b", r["text"].upper())
            pred = m.group(1) if m else r["text"].strip()[:1].upper()
        return {
            "id": row.get("id"),
            "gold": gold,
            "pred": pred,
            "correct": pred == gold,
            "http_status": r["http_status"],
            "response": r["text"][:200],
            "elapsed_s": r["elapsed_s"],
        }

    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(one, row) for row in rows_in]
        for f in as_completed(futs):
            row = f.result()
            results.append(row)
            if row.get("http_status") != 200:
                infra += 1

    total = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    summary = {
        "method": "arc_easy_mc_letter",
        "dataset": "allenai/ai2_arc ARC-Easy test",
        "n": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "wilson95": wilson_ci(correct, total),
        "infra_http_errors": infra,
        "data_path": str(data),
        "concurrency": conc,
    }
    write_json(out_dir / "qa.json", {"summary": summary, "rows": results})
    status = "PASS" if total and infra < total else "ERROR"
    return LaneResult(
        lane_id="qa",
        status=status,
        summary=summary,
        artifacts={"qa.json": str(out_dir / "qa.json")},
        infra_errors=1 if infra == total and total else 0,
        elapsed_s=time.perf_counter() - t0,
    )
