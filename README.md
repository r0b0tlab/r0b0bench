# r0b0bench

Reproducible **OpenAI-compatible endpoint** benchmarks for humans and agents.

> **Status:** `v1.0.0rc2` client package. Profiles: **`core`**, **`core-subset`**, **`systems`**.  
> Systems package: canary, BFCL-MT, BFCL-AST, **latency**, **concurrency**, **throughput**, max-context **NIAH**.  
> Quality lanes (QA / IFEval / HumanEval / GSM8K) are executable in rc2 (subset sizes on `core-subset`).  
> Private campaign dumps are not automatically r0b0bench results — run through this package (or hash-valid import).

## Profiles

| Profile | Systems block | Quality |
|---------|---------------|---------|
| **`core-subset`** | canary + BFCL-MT 200 + BFCL AST-600 + latency + concurrency + throughput + NIAH@max-context | QA@400, IFEval@200, HE@164, GSM8K@200 |
| **`core`** | **same systems block** | Full 8,620-class pillars (when datasets configured) |
| **`systems`** | systems only | — |

### Systems block (mandatory on core profiles)

1. **canary** — deterministic API checks  
2. **bfcl_mt** — BFCL v4 `multi_turn_base` (200)  
3. **bfcl_ast** — multiple + parallel + parallel_multiple (600)  
4. **latency** — C1 streaming TTFT / ITL  
5. **concurrency** — C1–C6 decode ladder  
6. **throughput** — C1 decode + prefill proxy  
7. **niah** — 25% / 50% / 90% of `(max_model_len − 64)`

## Results ledger (record & compare)

See **[results/README.md](results/README.md)** and **[docs/RESULTS.md](docs/RESULTS.md)**.

```bash
r0b0bench results list
r0b0bench results show dsv4-flash-native-v11-core-subset-20260803
r0b0bench results compare <entry_a> <entry_b>
r0b0bench results add --report /path/to/report.json --entry-id my-run-20260803
```

Entries live in `results/entries/*.json`. Generated table: [`results/LEADERBOARD.md`](results/LEADERBOARD.md).

## Quick start (local)

```bash
cd r0b0bench
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
# optional: pip install -e '.[bfcl]' human-eval langdetect

r0b0bench profiles
r0b0bench doctor --base-url http://127.0.0.1:8888/v1 --model deepseek-v4-flash-dspark

r0b0bench run --profile core-subset \
  --base-url http://127.0.0.1:8888/v1 \
  --model deepseek-v4-flash-dspark \
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
r0b0bench run --profile core|core-subset|systems --base-url URL --model NAME --output DIR
r0b0bench report --run-dir DIR
r0b0bench results list|show|compare|add|rebuild-index
```

## Principles

- Official BFCL scorers stay unforked when used.  
- Canaries stop on infrastructure failure; wrong model answers are scores.  
- No opaque single composite score in v1.  
- Subset metrics always report `@n` (and Wilson CI when present).  
- Raw traces stay in `--output`; do not commit them.  
- Results ledger entries are sanitized summaries only.

## Documents

- [Results ledger](docs/RESULTS.md)  
- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)  
- [AGENTS.md](AGENTS.md) — for coding agents  
- [Container notes](docs/CONTAINER.md)  

## License

MIT for original r0b0bench code and docs. Third-party benchmarks remain under their own terms — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
