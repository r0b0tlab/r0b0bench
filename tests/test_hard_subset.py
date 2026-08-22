"""Tests for the hard-subset profile and MultiChallenge protocol."""
from __future__ import annotations

import hashlib
import json
from typing import cast

import pytest

from r0b0bench.config import load_profile
from r0b0bench.endpoint import Endpoint
from r0b0bench.lanes.multichallenge import (
    _generate_responses,
    _load_rows,
    _validate_judgments,
    parse_judge_verdict,
)


def test_hard_subset_profile_loads():
    profile = load_profile("hard-subset")
    assert profile["profile"] == "hard-subset"
    assert profile["lane_order"] == [
        "canary",
        "multichallenge",
        "bfcl_mt",
        "canary_end",
    ]
    assert "tau2" not in profile
    assert profile["multichallenge"]["n_rows"] is None
    assert profile["multichallenge"]["expected_rows"] == 266
    assert profile["systems"]["bfcl_mt"]["category"] == "multi_turn_base"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reasoning here.\nVerdict: YES", "YES"),
        ("Reasoning here.\nFINAL VERDICT - NO.", "NO"),
        ("The criterion is satisfied. YES", "YES"),
        ("YES because the criterion is satisfied", None),
        ("maybe", None),
        ("", None),
    ],
)
def test_parse_judge_verdict(text, expected):
    assert parse_judge_verdict(text) == expected


def test_mc_load_rows_struct_conversation(tmp_path):
    """HF parquet's struct-of-lists conversation is normalized."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "question_id": ["q1"],
            "axis": ["INFERENCE_MEMORY"],
            "conversation": pa.array(
                [{"role": ["user", "assistant", "user"], "content": ["hi", "hello", "remember?"]}],
                type=pa.struct(
                    [
                        pa.field("role", pa.list_(pa.string())),
                        pa.field("content", pa.list_(pa.string())),
                    ]
                ),
            ),
            "target_question": ["Did the response remember the greeting?"],
            "pass_criteria": ["YES"],
            "num_turns": [3],
        }
    )
    path = tmp_path / "mc.parquet"
    pq.write_table(table, path)

    rows = _load_rows(path)
    assert rows[0]["conversation"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "remember?"},
    ]


class _FakeEndpoint:
    model = "target-model"
    base_url = "http://target.test/v1/"

    def __init__(self):
        self.payloads = []

    def chat_completions(self, payload):
        self.payloads.append(payload)
        return 200, {
            "choices": [
                {
                    "message": {"content": "I remembered the greeting."},
                    "finish_reason": "stop",
                }
            ]
        }, 0.1


def test_generation_sends_released_conversation_not_rubric_question():
    endpoint = _FakeEndpoint()
    conversation = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "what did I say?"},
    ]
    dataset_rows = [
        {
            "question_id": "q1",
            "axis": "INFERENCE_MEMORY",
            "conversation": conversation,
            "target_question": "Did the response remember the greeting?",
            "pass_criteria": "YES",
            "num_turns": 3,
        }
    ]

    responses = _generate_responses(
        cast(Endpoint, endpoint),
        dataset_rows,
        concurrency=1,
        max_tokens=2048,
        temperature=0.0,
        data_revision="rev1",
    )

    assert endpoint.payloads[0]["messages"] == conversation
    assert all(
        message["content"] != "Did the response remember the greeting?"
        for message in endpoint.payloads[0]["messages"]
    )
    assert responses[0]["response"] == "I remembered the greeting."
    assert responses[0]["criterion"] == "Did the response remember the greeting?"


def test_imported_judgments_are_hash_bound():
    answer = "I remembered the greeting."
    criterion = "Did the response remember the greeting?"
    responses = [
        {
            "question_id": "q1",
            "response": answer,
            "response_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "criterion": criterion,
            "criterion_sha256": hashlib.sha256(criterion.encode()).hexdigest(),
        }
    ]
    judgments = [
        {
            "question_id": "q1",
            "response_sha256": responses[0]["response_sha256"],
            "criterion_sha256": responses[0]["criterion_sha256"],
            "judge_provider": "openai-codex",
            "judge_model": "gpt-5.6-sol",
            "verdict": "YES",
            "reasoning": "The response satisfies the criterion.",
        }
    ]
    validated = _validate_judgments(judgments, responses)
    assert validated["q1"]["verdict"] == "YES"

    tampered = json.loads(json.dumps(judgments))
    tampered[0]["response_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="response hash mismatch"):
        _validate_judgments(tampered, responses)
