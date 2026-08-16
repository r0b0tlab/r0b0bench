from __future__ import annotations

import json

import httpx
import pytest

from r0b0bench.endpoint import Endpoint


def test_chat_template_environment_override_merges_native_muse_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "R0B0BENCH_CHAT_TEMPLATE_KWARGS",
        '{"reasoning_strength":"low"}',
    )
    endpoint = Endpoint("http://test/v1", "muse")
    try:
        body = endpoint._chat_body({
            "messages": [{"role": "user", "content": "test"}],
            "chat_template_kwargs": {"thinking": False},
        })
    finally:
        endpoint.close()

    assert body["chat_template_kwargs"] == {
        "thinking": False,
        "reasoning_strength": "low",
    }


@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content"])
def test_streaming_reasoning_sets_ttft_without_counting_completion_chunks(
    reasoning_field: str,
) -> None:
    events = [
        {"choices": [{"delta": {reasoning_field: "thinking"}}]},
        {"choices": [{"delta": {"content": "final "}}]},
        {
            "choices": [
                {"delta": {"content": "answer"}, "finish_reason": "stop"}
            ]
        },
        {
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        },
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode(),
        )

    endpoint = Endpoint("http://test/v1", "muse")
    endpoint._client.close()
    endpoint._client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        status, stats = endpoint.chat_completions_stream(
            {"messages": [{"role": "user", "content": "test"}]}
        )
    finally:
        endpoint.close()

    assert status == 200
    assert stats["ok"] is True
    assert stats["ttft_ms"] is not None
    assert stats["text"] == "final answer"
    assert stats["stream_completion_chunks"] == 2
    assert stats["itl_ms_mean"] is not None
    assert stats["finish_reason"] == "stop"
    assert stats["usage"] == {"prompt_tokens": 10, "completion_tokens": 3}
