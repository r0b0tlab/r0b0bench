# r0b0bench

Reproducible **OpenAI-compatible endpoint** benchmarks for humans and agents.

> **Status:** `v1.0.0rc1` client package. Two profiles only: **`core`** and **`core-subset`**.  
> Both profiles always include the **systems block** (canary, BFCL-MT, BFCL-AST, perf, NIAH).  
> Quality scorers for QA/IFEval/HumanEval/GSM8K are scaffolded in RC1 (`NOT_IMPLEMENTED` until next RC).  
> Private campaign dumps are not automatically r0b0bench results — run through this package (or hash-valid import).

## Profiles

| Profile | Systems block | Quality |
|---------|---------------|---------|
| **`core-subset`** | canary + BFCL multi_turn_base 200 + BFCL AST-600 + perf c1–16 + NIAH@max-context | QA@400, IFEval@200, HE@164, GSM8K@200 (subset locks; scorers next RC) |
| **`core`** | **same systems block** | Full 8,620-class pillars (QA 7715, IFEval 541, HE 164, …) |

### Systems block (mandatory)

1. **canary** — 5 deterministic API checks  
2. **bfcl_mt** — official BFCL v4 `multi_turn_base` (200) via `bfcl-eval` (import or external runner in RC1)  
3. **bfcl_ast** — `multiple` + `parallel` + `parallel_multiple` (600)  
4. **perf** — portable concurrency sweep c1/2/4/8/16  
5. **niah** — 3 depths at **25% / 50% / 90%** of `(max_model_len − 64)` from `/v1/models`

There is **no** standalone `systems` profile.

## Quick start (Docker)

```bash
docker pull ghcr.io/r0b0tlab/r0b0bench:v1.0.0-rc1   # after publish

docker run --rm --network host \
  -e R0B0BENCH_BASE_URL=http://127.0.0.1:18082/v1 \
  -e R0B0BENCH_MODEL=xyz-aquila-nvfp4 \
  -v /tmp/r0b0bench-out:/out \
  -v /path/to/tokenizer_or_model:/tokenizer:ro \
  ghcr.io/r0b0tlab/r0b0bench:v1.0.0-rc1 \
  run --profile core-subset \
      --base-url http://127.0.0.1:18082/v1 \
      --model xyz-aquila-nvfp4 \
      --tokenizer /tokenizer \
      --output /out
```

## Quick start (local)

```bash
cd r0b0bench
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
# optional: pip install -e '.[bfcl]'

r0b0bench profiles
r0b0bench doctor --base-url http://127.0.0.1:18082/v1 --model xyz-aquila-nvfp4

r0b0bench run --profile core-subset \
  --base-url http://127.0.0.1:18082/v1 \
  --model xyz-aquila-nvfp4 \
  --tokenizer /path/to/model_or_tokenizer \
  --output /tmp/r0b0bench-out
```

`--output` must be **outside** the git checkout (or a gitignored `out/` name).

## Aquila NVFP4 example

- Weights: https://huggingface.co/r0b0tlab/XYZ-Aquila-mini-NVFP4  
- Runtime: `ghcr.io/r0b0tlab/xyz-aquila-mini-nvfp4-vllm:v0.25.0-sm121-702f4814`  
- Docs: https://github.com/r0b0tlab/xyz-aquila-mini-nvfp4-sm121-vllm  

Aquila is an **example endpoint**, not a third profile.

## CLI

```text
r0b0bench profiles
r0b0bench doctor --base-url URL --model NAME
r0b0bench run --profile core|core-subset --base-url URL --model NAME --output DIR [--tokenizer PATH] [--only canary,niah,perf]
r0b0bench report --run-dir DIR
```

## Principles

- Official BFCL/IFEval scorers stay unforked and version-pinned when used.  
- Canaries stop on infrastructure failure; wrong model answers are scores, not silent stops (except canary hard-fail on total HTTP death).  
- No opaque single composite score in v1.  
- Subset metrics always report `@n` and Wilson CI when implemented.  
- Raw traces stay in `--output`; do not commit them.

## Documents

- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)  
- [Reference method](docs/REFERENCE_METHOD_2026-07-21.md)  
- [AGENTS.md](AGENTS.md) — for coding agents  
- [Container notes](docs/CONTAINER.md)  

## License

MIT for original r0b0bench code and docs. Third-party benchmarks remain under their own terms — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Standard systems package (v1.0.0rc2+)

Profiles: `systems`, `core-subset`, `core`.

Systems lanes (always, via `lane_order: [systems]`):

| Lane | What it measures |
|------|------------------|
| `canary` | semantic / tool / stop smoke |
| `bfcl_mt` | BFCL multi-turn base (200) |
| `bfcl_ast` | BFCL AST micro (multiple/parallel/parallel_multiple) |
| `latency` | C1 streaming TTFT + ITL + e2e |
| `concurrency` | C1/C2/C4/C6 decode ladder |
| `throughput` | C1 decode (2048 out) + ~14k prefill proxy |
| `niah` | **max-context** NIAH at 25/50/90% of `max_model_len` |

Optional composite: `--only perf` runs latency+concurrency+throughput as one lane.

```bash
r0b0bench run --profile systems \
  --base-url http://HOST:PORT/v1 --model MODEL \
  --tokenizer /path/to/hf-or-tokenizer.json-dir \
  --output /tmp/r0b0bench-out
```
