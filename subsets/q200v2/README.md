# Q200v2 — frozen quality kit (text180 + BFCL structural-hard20)

Campaign kit published **as a subset** of r0b0bench so the identity that
[`docs/PROCEDURES.md` §1](../../docs/PROCEDURES.md) already pins is verifiable here.

## What it is

| Part | Contents |
|---|---|
| `artifacts/quality-text-180-v2.jsonl` | **180** frozen text-quality rows — `gsm8k` 80, `humaneval` 40, `ifeval` 40, `hard_reasoning` 20. Rows are `{id, family, grade, prompt, reference}`; grades are `numeric_exact` (gsm8k), `exec` (humaneval, docker sandbox), `ifeval_strict` (ifeval) and `manual` (hard_reasoning). |
| `artifacts/bfcl-v4-multi-turn-hard20-v1.json` | **20** selected ids from the official BFCL v4 `multi_turn_base` category (`bfcl-eval==2025.12.17`), chosen by a deterministic structural ranking. |
| `artifacts/quality-200.jsonl` | the **parent** 200-row set. `quality-text-180-v2` is the exact projection of this file minus the `agentic_coding` 20 rows — asserted by the test. |
| `scripts/` + `docker/q200_sandbox_driver.py` | the kit harness: text lane (`run_quality_set.py`), closure/manual-evidence tool (`close_q200_v2.py`), BFCL lane wrapper (`run_bfcl_hard20.py`), local GSM8K grader (`score_flex_gsm8k.py`), and the humaneval sandbox driver the lane execs inside the candidate image. |
| `tests/test_q200_v2.py` | identity test: dataset hash, projection property, family counts, BFCL selection + boundary/provenance hashes. |

## Identity (verify the bytes, don't trust the prose)

```
74623ab9b075120cd6f7a93059cc16d8817a6039dd20118b8f0350279f8b1ed6  artifacts/quality-text-180-v2.jsonl
ca35650e0bf4c9997772276c15a7116afd553a305b10c88182f52f050b76e066  artifacts/quality-200.jsonl
0860da504a3db2c3cd73647ecdc2a5ecdb1793d7a5cf3f4f004912f0ef314d4e  artifacts/bfcl-v4-multi-turn-hard20-v1.json
```

The first hash is the `dataset_sha256` published in `docs/PROCEDURES.md` §1, so a ledger entry
citing Q200v2 can now be re-derived from bytes in this repository. `MANIFEST.sha256` covers every
file in the subset; run

```
sha256sum -c subsets/q200v2/MANIFEST.sha256
pytest subsets/q200v2/tests
```

## Provenance notes

- **Origin kit.** `qwen38-flash-next-w4a16/q200v2ar-20260829T141533Z-runner` (the campaign kit used
  by the Qwen3.8-Flash-Next W4A16, Ling-3.0-flash-VL and GLM-5.3-Flash campaigns). Files under this
  directory are **byte-frozen**: not reformatted, not re-linted, not path-rewritten — with the single
  documented exception below.
- **One sanitized file.** `artifacts/quality-200.manifest.json` originally carried an absolute
  campaign path in its `source` field. That path is replaced (`"source": "artifacts/quality-200.jsonl"`,
  `source_path_redacted: true`) and the file's original byte hash is preserved inside it as
  `original_file_sha256` (`2b1b898e3eabacbb2c5ff5abaa463538d6db813ca78df4ca1ffeb62bf2c00a83`). No other
  byte of any kit file was altered, and the dataset hashes above are untouched.
- **The BFCL part is a selection, not a fork.** Only ids plus selection metadata are published here;
  the cases and ground truth come from the official BFCL v4 dataset via `bfcl-eval==2025.12.17`
  (`requirements-bfcl.txt`). The manifest states its own reading: a *deterministic structural-complexity
  proxy subset — not the 20 semantically hardest cases and not an official full-category leaderboard
  score*. The rank-20/rank-21 tie boundary and the six tied cases selected from it are disclosed in
  `selection_boundary`, and the canonical id/feature/policy hashes are in `selection_provenance`.
- **`hard_reasoning` is manual-review by design.** Those rows ship `grade: manual` and stay `ungraded`
  until `--manual-evidence <file>` supplies an `independent_manual_review` record; re-running the same
  `--run-id` resumes and re-grades without regenerating responses.
- **Budget rule (fail closed).** The frozen contract accepts only `finish_reason == "stop"`; a
  length-truncated row is transport-failed even if its grader passes, and must be disclosed rather than
  rescued by raising the cap.

## Running the kit

Text lane (default set is this subset's file):

```
python3 subsets/q200v2/scripts/run_quality_set.py \
  --base-url http://127.0.0.1:8000 \
  --run-id <run-id> \
  --image-id sha256:<candidate-image-id> \
  --profile-id <profile-id> \
  --candidate-id <candidate-id> \
  --admission-config <admission.json> \
  [--manual-evidence <manual-review.json>] \
  [--chat-template-kwargs '{"enable_thinking": true}']      # model-specific thinking switch
```

- Run it **from a directory you want the outputs in**: the lane writes `<run-id>.rows.jsonl` and
  `<run-id>.summary.json` relative to the working directory.
- `--set` defaults to `artifacts/quality-text-180-v2.jsonl`, so pass the path explicitly when running
  from outside this directory.
- `--image-id/--profile-id/--candidate-id` bind the run identity; they are recorded per row.
- The humaneval grader execs `docker run … <image-id> python3 /opt/r0b0tlab/q200_sandbox_driver.py`
  (`--memory=256m --memory-swap=256m`, no `--entrypoint`), so the candidate image must bake
  `docker/q200_sandbox_driver.py` at `/opt/r0b0tlab/` and clear its ENTRYPOINT — otherwise every
  humaneval row reports `HARNESS_BLOCK` (vLLM images set `ENTRYPOINT ["vllm"]`, whose CLI eats the
  args and turns the 256 MB cap into rc 137 with empty stdout). See `docs/PROCEDURES.md` §1.

BFCL lane (install `requirements-bfcl.txt` first; `BFCL_PROJECT_ROOT` must point at a fresh project
root — resume is disabled for claim-bearing runs):

```
BFCL_PROJECT_ROOT=<dir> python3 subsets/q200v2/scripts/run_bfcl_hard20.py <inspect|run|evaluate|status>
```

Serve side: `max_model_len ≥ prompt + BFCL_MAX_TOKENS` (32 K was used in the campaigns; a 16 K serve
returns HTTP 400 "maximum context length is 16384"). "Failed to decode the model response" printed
once per turn is the normal text-only path, not a harness failure.

## Licensing and attribution

The kit is MIT (© 2026 r0b0tlab), same license as this repository. The quality rows are
generated-answer items derived from public upstream sources — BFCL / gorilla (Apache-2.0), GSM8K
(MIT), HumanEval (MIT), IFEval (Apache-2.0) — which retain their own terms; see
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
