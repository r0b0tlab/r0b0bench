# Profiles

The public CLI exposes four profiles:

## `core-subset` (default RC path)

- Systems block: canary, BFCL-MT, BFCL-AST, latency, concurrency, throughput, and NIAH
- Quality subsets: QA 400, IFEval 200, HumanEval 164, GSM8K 200

## `core`

- Same systems block
- Full quality populations when the frozen datasets are configured

## `systems`

- Systems block only; useful for serving/configuration diagnostics
- Never use a systems-only report as a core quality claim

## `hard-subset` (hard multiturn quality)

- Lanes: `canary` → `multichallenge` → `bfcl_mt` → `canary_end`
- `multichallenge` — ScaleAI MultiChallenge test split (full 266 rows):
  generate from the released conversation, then grade the response with an
  independent LLM using the released instance-level YES/NO rubric. Scores are
  comparable only when the judge identity is the same. The lane fails closed
  if no independent judge (or hash-bound imported judgments) is configured.
- `bfcl_mt` — official BFCL V4 multi_turn_base ×200 (shared systems config).
- `canary_end` — end-of-run canary replay (fail-closed if the endpoint
  degraded during the run).
- This profile does NOT include the full systems block (no AST/latency/
  concurrency/throughput/NIAH) — it is a quality profile, not a systems
  profile. τ²-bench airline/retail/telecom subsets were removed by user
  decision on 2026-08-21.
- Think-on is enforced by the runtime env `R0B0BENCH_CHAT_TEMPLATE_KWARGS`;
  per-lane `max_tokens` floors (8_192 when thinking is on) guarantee the
  model has room to finish answers.
- MultiChallenge judge configuration:
  `R0B0BENCH_MULTICHALLENGE_JUDGE_BASE_URL`,
  `R0B0BENCH_MULTICHALLENGE_JUDGE_MODEL`, and optionally
  `R0B0BENCH_MULTICHALLENGE_JUDGE_API_KEY`. For a two-stage/offline judge,
  use the hash-bound `R0B0BENCH_MULTICHALLENGE_RESPONSES_PATH` and
  `R0B0BENCH_MULTICHALLENGE_JUDGMENTS_PATH` inputs.
- Stage the pinned 266-row parquet at the profile path or set
  `R0B0BENCH_MULTICHALLENGE_DATASET`; the lane verifies the profile's SHA-256
  before sending any target requests.

## Systems block

Shared YAML: `src/r0b0bench/profiles/_systems_block.yaml`

| Lane | Behavior |
|------|----------|
| canary | deterministic API and schema checks |
| bfcl_mt | official BFCL v4 `multi_turn_base` ×200 |
| bfcl_ast | official BFCL `multiple` + `parallel` + `parallel_multiple` ×200 each |
| latency | C1 streaming TTFT / ITL |
| concurrency | C1–C6 decode ladder |
| throughput | C1 decode and prefill proxy |
| niah | 0.25/0.50/0.90 × (max_model_len − 64) |

A filtered `--only` run is diagnostic and is marked `invalid_for_publish` by
`report.json`. A claim-bearing core report must contain every listed lane with
status `PASS` and zero infrastructure errors.
