from r0b0bench.results_ledger import entry_from_report


def test_hard_subset_report_preserves_multichallenge_and_end_canary():
    report = {
        "r0b0bench_version": "1.0.0rc2",
        "profile": "hard-subset",
        "model": "Ornith-1.5-35B-A3B",
        "invalid_for_publish": False,
        "infra_errors_total": 0,
        "lanes": [
            {
                "lane_id": "canary",
                "status": "PASS",
                "summary": {"passed": True, "n": 5},
            },
            {
                "lane_id": "multichallenge",
                "status": "PASS",
                "summary": {
                    "accuracy": 0.5,
                    "correct": 133,
                    "rows": 266,
                    "by_axis": {"INFERENCE_MEMORY": {"correct": 50, "total": 100}},
                    "judge_models": ["gpt-5.6-sol"],
                    "judge_providers": ["openai-codex"],
                    "data_revision": "rev",
                    "dataset_sha256": "a" * 64,
                    "comparable_only_with_same_judge": True,
                },
            },
            {
                "lane_id": "bfcl_mt",
                "status": "PASS",
                "summary": {"accuracy": 0.6},
            },
            {
                "lane_id": "canary_end",
                "status": "PASS",
                "summary": {"passed": True, "n": 5},
            },
        ],
    }

    entry = entry_from_report(
        report,
        entry_id="ornith-hard-subset-test",
        model_display="Ornith test",
        hardware="1x GB10",
    )

    score = entry["scores"]["multichallenge"]
    assert score["status"] == "PASS"
    assert score["accuracy"] == 0.5
    assert score["correct"] == 133
    assert score["total"] == 266
    assert score["judge_models"] == ["gpt-5.6-sol"]
    assert score["comparable_only_with_same_judge"] is True
    assert entry["scores"]["canary_end"] == {"status": "PASS", "passed": True, "n": 5}
