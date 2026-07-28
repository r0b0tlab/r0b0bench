from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from r0b0bench.config import LaneResult, write_json
from r0b0bench.endpoint import Endpoint


def run_canary(ep: Endpoint, out_dir: Path, cfg: dict[str, Any] | None = None) -> LaneResult:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    cases = [
        {
            "id": "identity",
            "messages": [{"role": "user", "content": "Reply with exactly R0B0BENCH_OK and nothing else."}],
            "max_tokens": 256,
            "expect": "R0B0BENCH_OK",
        },
        {
            "id": "zh_arithmetic",
            "messages": [{"role": "user", "content": "只回答数字：17乘以19等于多少？"}],
            "max_tokens": 256,
            "expect": "323",
        },
        {
            "id": "structured",
            "messages": [
                {
                    "role": "user",
                    "content": "Return one compact JSON object with keys alpha and beta, set to integers 2 and 3. No prose.",
                }
            ],
            "max_tokens": 256,
            "expect_json": {"alpha": 2, "beta": 3},
        },
        {
            "id": "tool_call",
            "messages": [{"role": "user", "content": "What is the current weather in Tokyo? Use the provided tool."}],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 512,
            "expect_tool": True,
        },
        {
            "id": "needle",
            "messages": [
                {
                    "role": "user",
                    "content": ("filler red blue green. " * 800)
                    + " The verification code is A9Q7. "
                    + ("filler one two three. " * 800)
                    + "What is the verification code? Reply with only the code.",
                }
            ],
            "max_tokens": 256,
            "expect": "A9Q7",
        },
    ]
    results = []
    checks: dict[str, bool] = {}
    for case in cases:
        payload: dict[str, Any] = {
            "messages": case["messages"],
            "temperature": 0,
            "max_tokens": case["max_tokens"],
        }
        for k in ("tools", "tool_choice"):
            if k in case:
                payload[k] = case[k]
        status, body, elapsed = ep.chat_completions(payload)
        results.append({"id": case["id"], "http_status": status, "elapsed_s": elapsed, "response": body})
        ok = status == 200 and bool(body.get("choices"))
        msg = body.get("choices", [{}])[0].get("message", {}) if ok else {}
        content = (msg.get("content") or "").strip()
        blob = "\n".join(
            x
            for x in (content, str(msg.get("reasoning") or ""), str(msg.get("reasoning_content") or ""))
            if x
        )
        if case.get("expect"):
            ok = ok and case["expect"] in blob
        elif case.get("expect_json"):
            parsed_ok = False
            for candidate in (content, blob):
                try:
                    s, e = candidate.find("{"), candidate.rfind("}")
                    if s >= 0 and e > s and json.loads(candidate[s : e + 1]) == case["expect_json"]:
                        parsed_ok = True
                        break
                except Exception:
                    pass
            ok = ok and parsed_ok
        elif case.get("expect_tool"):
            ok = ok and bool(msg.get("tool_calls"))
        checks[case["id"]] = bool(ok)

    summary = {"checks": checks, "passed": all(checks.values()), "n": len(checks)}
    write_json(out_dir / "canary.json", {"summary": summary, "results": results})
    status = "PASS" if summary["passed"] else "FAIL"
    # canary failure is infra-ish if all HTTP fail; model wrong answers are FAIL but not infra_errors
    infra = 0
    if all(r["http_status"] != 200 for r in results):
        infra = 1
        status = "ERROR"
    return LaneResult(
        lane_id="canary",
        status=status,
        summary=summary,
        artifacts={"canary.json": str(out_dir / "canary.json")},
        infra_errors=infra,
        elapsed_s=time.perf_counter() - t0,
    )
