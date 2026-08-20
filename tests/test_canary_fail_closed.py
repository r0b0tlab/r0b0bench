from r0b0bench.lanes.canary import _check_case
from r0b0bench.thinking import effective_generation_reserve, effective_max_tokens


def test_structured_budget_exhaustion_is_a_real_failure() -> None:
    case = {"id": "structured", "expect_json": {"alpha": 2, "beta": 3}}
    body = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": "", "reasoning_content": "still reasoning"},
            }
        ],
        "usage": {"reasoning_tokens": 8192},
    }
    assert _check_case(case, 200, body) is False


def test_structured_json_response_passes() -> None:
    case = {"id": "structured", "expect_json": {"alpha": 2, "beta": 3}}
    body = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": '{"alpha": 2, "beta": 3}'},
            }
        ]
    }
    assert _check_case(case, 200, body) is True


def test_tool_canary_requires_a_tool_call() -> None:
    case = {"id": "tool_call", "expect_tool": True}
    body = {"choices": [{"message": {"content": "I cannot call tools"}}]}
    assert _check_case(case, 200, body) is False


def test_thinking_on_uses_complete_answer_budgets(monkeypatch) -> None:
    monkeypatch.setenv("R0B0BENCH_CHAT_TEMPLATE_KWARGS", '{"thinking":true,"enable_thinking":true}')
    assert effective_max_tokens(32, "qa") == 32768
    assert effective_max_tokens(512, "gsm8k") == 49152
    assert effective_max_tokens(512, "concurrency") == 8192
    assert effective_generation_reserve(256) == 8192


def test_thinking_off_preserves_profile_budgets(monkeypatch) -> None:
    monkeypatch.setenv("R0B0BENCH_CHAT_TEMPLATE_KWARGS", '{"thinking":false,"enable_thinking":false}')
    assert effective_max_tokens(32, "qa") == 32
    assert effective_max_tokens(512, "gsm8k") == 512
    assert effective_generation_reserve(256) == 256
