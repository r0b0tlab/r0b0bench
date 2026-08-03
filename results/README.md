# r0b0bench results ledger

Record and compare **package-produced** runs. Private campaign dumps are not results until imported (or re-run) through r0b0bench.

## Layout

```text
results/
  README.md                 # this file
  SCHEMA.json               # entry schema
  index.json                # regenerated summary table
  LEADERBOARD.md            # human-readable comparison
  templates/entry.json      # blank template
  entries/<entry_id>.json   # one file per recorded run
```

## Record a run

### From a package `report.json`

```bash
r0b0bench results add \
  --report /path/to/run/report.json \
  --entry-id my-model-core-subset-20260803 \
  --model-display "My Model 7B" \
  --hardware "2x GB10 TP=2" \
  --notes "optional free text"
```

### Manual / import sanitized JSON

```bash
cp results/templates/entry.json results/entries/my-entry.json
# edit scores + identity
r0b0bench results rebuild-index
```

## Compare

```bash
r0b0bench results list
r0b0bench results show dsv4-flash-native-v11-core-subset-20260803
r0b0bench results compare \
  dsv4-flash-native-v11-core-subset-20260803 \
  another-entry-id
```

## Rules

1. **Only package metrics** — lane scores must come from r0b0bench (or hash-valid import with `harness.name=r0b0bench`).  
2. **No private secrets** — strip API keys; prefer omitting internal IPs (`base_url` optional).  
3. **Disclosures required** when scorers are lightweight / subset sizes / alternate datasets.  
4. **infra_errors_total must be 0** for `invalid_for_publish=false`.  
5. **Do not mix** backends (e.g. openai_portable vs vllm-bench) in one comparable cell without labeling.  
6. Rebuild `index.json` + `LEADERBOARD.md` after every add (`results rebuild-index`).

## Seed entry

| entry_id | model | profile | gsm8k@200 | he pass@1 | bfcl_mt | niah |
|----------|-------|---------|----------:|----------:|--------:|------|
| dsv4-flash-native-v11-core-subset-20260803 | DeepSeek-V4-Flash DSpark | core-subset | 0.950 | 0.909 | 0.755 | PASS 1M 25/50/90 |

See [docs/RESULTS.md](../docs/RESULTS.md).
