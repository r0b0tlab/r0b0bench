# AGENTS.md — working on / running r0b0bench

This file is for **coding agents** (Hermes, Cursor, etc.).

## What this repo is

- **Runnable** OpenAI-compatible **benchmark client** (`r0b0bench` CLI + optional container).
- **Not** a model server. Do not bake weights into this image.
- **Exactly two profiles:** `core` | `core-subset`. Both **always** include the systems block.

## Systems block (every profile)

Order: `canary` → `bfcl_mt` → `bfcl_ast` → `perf` → `niah`

- **NIAH** depths = 25% / 50% / 90% of `(max_model_len − 64)` from `/v1/models`. Do not hardcode 8k/16k/28k.
- **Perf** default backend is portable OpenAI chat concurrency (label method string; never mix with vllm-bench rows).

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
r0b0bench run --profile core-subset --only canary,niah,perf \
  --base-url "$URL" --model "$MODEL" --tokenizer "$TOK" --output /tmp/r0b0bench-out
```

## Definition of done (agent task)

- `r0b0bench profiles` prints only `core` and `core-subset`
- `doctor` green against target endpoint
- `run` writes `report.json` + per-lane artifacts under `--output`
- No credentials or absolute private host paths in commits
- NIAH summary includes discovered `max_model_len` and derived depths

## RC1 honesty

Quality lanes (QA/IFEval/HE/GSM8K) may return `NOT_IMPLEMENTED` until scorers land. Systems lanes **canary / niah / perf** are executable; BFCL may be `NOT_IMPLEMENTED` without `[bfcl]` extra or import. Do not pretend unimplemented lanes passed.
