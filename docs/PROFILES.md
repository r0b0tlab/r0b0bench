# Profiles

Only two user profiles:

## `core-subset` (default RC path)

- Systems block (mandatory)
- Quality subsets (QA 400, IFEval 200, HE 164, GSM8K 200) — scorers land post-RC1

## `core`

- Same systems block
- Full quality populations (7715 / 541 / 164 / …)

## Systems block

Shared YAML: `src/r0b0bench/profiles/_systems_block.yaml`

| Lane | Behavior |
|------|----------|
| canary | 5 checks |
| bfcl_mt | multi_turn_base ×200 |
| bfcl_ast | AST-600 |
| perf | c1–16 portable |
| niah | 0.25/0.50/0.90 × (max_model_len − 64) |

No `systems-only` profile.
