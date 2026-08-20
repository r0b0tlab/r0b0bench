from r0b0bench.lanes.canary import _check_case


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
