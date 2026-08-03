# Profiles

The public CLI exposes three profiles:

## `core-subset` (default RC path)

- Systems block: canary, BFCL-MT, BFCL-AST, latency, concurrency, throughput, and NIAH
- Quality subsets: QA 400, IFEval 200, HumanEval 164, GSM8K 200

## `core`

- Same systems block
- Full quality populations when the frozen datasets are configured

## `systems`

- Systems block only; useful for serving/configuration diagnostics
- Never use a systems-only report as a core quality claim

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
