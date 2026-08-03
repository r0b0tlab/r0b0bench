# AGENTS.md — working on / running r0b0bench

This file is for **coding agents** (Hermes, Cursor, etc.).

## What this repo is

- **Runnable** OpenAI-compatible **benchmark client** (`r0b0bench` CLI + optional container).
- **Not** a model server. Do not bake weights into this image.
- **Exactly three public profiles:** `core` | `core-subset` | `systems`. The first two always include the systems block; `systems` runs only that block.

## Systems block (every profile)

- Order: `canary` → `bfcl_mt` → `bfcl_ast` → `latency` → `concurrency` → `throughput` → `niah`

- **NIAH** depths = 25% / 50% / 90% of `(max_model_len − 64)` from `/v1/models`. Do not hardcode 8k/16k/28k.
- **Latency/concurrency/throughput** use the portable OpenAI chat backend (label method strings; never mix with vllm-bench rows).

## Hard rules

1. **Output outside checkout** — `--output` must not be inside the git tree (except gitignored `out/`). Private evidence must not be committed.
2. **No BFCL forks** — use official `bfcl-eval` categories/names; do not invent leaderboard-looking composites.
3. **Canary taxonomy** — stop the run on infra/auth/schema death; ordinary wrong answers are scored (canary content fails → FAIL, not silent continue for publish).
4. **No score washing** — do not weaken verifiers to make a model pass.
5. **`--skip-systems`** is debug-only; report must set `invalid_for_publish=true`.
6. **Do not claim r0b0bench results** without a package-produced `report.json`.
7. Secrets only via env; never print tokens.

## Common commands

```bash
pip install -e .
r0b0bench doctor --base-url "$URL" --model "$MODEL"
r0b0bench run --profile core-subset --base-url "$URL" --model "$MODEL" \
  --tokenizer "$TOK" --output /tmp/r0b0bench-out
r0b0bench run --profile core-subset --only canary,niah,latency,concurrency,throughput \
  --base-url "$URL" --model "$MODEL" --tokenizer "$TOK" --output /tmp/r0b0bench-out
```

## Definition of done (agent task)

- `r0b0bench profiles` prints `core`, `core-subset`, and `systems`
- `doctor` green against target endpoint
- `run` writes `report.json` + per-lane artifacts under `--output`
- No credentials or absolute private host paths in commits
- NIAH summary includes discovered `max_model_len` and derived depths

## RC1 honesty

Quality lanes (QA/IFEval/HE/GSM8K) are executable in rc2 when their frozen datasets are configured. Systems lanes **canary / BFCL-MT / BFCL-AST / latency / concurrency / throughput / NIAH** are executable with the official BFCL adapter; missing adapters or scorers must remain explicit `ERROR`/`NOT_IMPLEMENTED`. Do not pretend unimplemented lanes passed.
