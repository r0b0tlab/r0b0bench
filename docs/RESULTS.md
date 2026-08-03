# Results ledger

Package-produced scores are recorded under [`results/`](../results/).

## Why

Private campaign directories are **not** automatically comparable. The ledger keeps a small, sanitized JSON entry per run so humans and agents can:

1. Record a freeze’s scores without committing multi‑GB traces  
2. Compare models / images over time  
3. Fail closed on missing disclosures / infra errors  

## Commands

```bash
r0b0bench results list
r0b0bench results show <entry_id>
r0b0bench results compare <entry_a> <entry_b>
r0b0bench results rebuild-index

r0b0bench results add \
  --report /path/to/run/report.json \
  --entry-id my-model-core-subset-20260803 \
  --model-display "My Model" \
  --hardware "2x GB10 TP=2" \
  --notes "optional disclosure"
```

## Files

| Path | Purpose |
|------|---------|
| `results/SCHEMA.json` | Entry schema |
| `results/templates/entry.json` | Blank template |
| `results/entries/*.json` | One file per recorded run |
| `results/index.json` | Machine summary (generated) |
| `results/LEADERBOARD.md` | Human table (generated) |

## Comparison rules

- Same **profile** (`core` vs `core-subset`)  
- Same **scorer disclosure** (e.g. IFEval lightweight vs official)  
- `invalid_for_publish=false` and `infra_errors_total=0` for primary rows  
- Label latency backend (portable OpenAI vs vllm-bench) when mixing  

## Seed entry

`dsv4-flash-native-v11-core-subset-20260803` — DeepSeek-V4-Flash DSpark native-v11 dual-GB10 core-subset freeze.
