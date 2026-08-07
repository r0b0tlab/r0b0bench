# r0b0bench leaderboard

Entries: **3** · regenerated from `results/entries/*.json`

Comparable only within the same profile and disclosed scorer variants.

## Quality

| entry_id | model | spec | GSM8K | HE@1 | QA | IFEval | BFCL-MT | ASTµ | NIAH | invalid |
|----------|-------|------|------:|-----:|---:|-------:|-------:|-----:|------|---------|
| dsv4-flash-native-v11-core-subset-20260803 | DeepSeek-V4-Flash DSpark NVFP4 (dua | dspark K=6 | 0.950 | 0.909 | 0.960 | 0.795 | 0.755 | 0.362 | PASS | False |
| inkling-small-marlin-baseline-20260806 | Inkling-Small NVFP4 (dual GB10, Mar | none | 0.810 | 0.732 | 0.245 | 0.425 | 0.540 | 0.302 | ERROR | True |
| inkling-small-marlin-mtp-20260807 | Inkling-Small NVFP4 MTP 8-1-9 (dual | mtp draft=8 8-1-9 | — | — | — | — | — | — | — | True |

## Performance

| entry_id | model | spec | decode tok/s | prefill tok/s | TTFT ms | c1 agg | c2 agg | c4 agg | c6 agg |
|----------|-------|------|------------:|-------------:|-------:|-------:|-------:|-------:|-------:|
| dsv4-flash-native-v11-core-subset-20260803 | DeepSeek-V4-Flash DSpark NVFP4 (dua | dspark K=6 | 40.164 | 29203.666 | 221.884 | 86.931 | 145.889 | 239.311 | 337.761 |
| inkling-small-marlin-baseline-20260806 | Inkling-Small NVFP4 (dual GB10, Mar | none | 13.779 | — | — | 14.091 | 26.653 | 46.828 | 48.694 |
| inkling-small-marlin-mtp-20260807 | Inkling-Small NVFP4 MTP 8-1-9 (dual | mtp draft=8 8-1-9 | 16.044 | — | — | 13.683 | 29.251 | 47.363 | 47.188 |

## Files

- [`dsv4-flash-native-v11-core-subset-20260803.json`](entries/dsv4-flash-native-v11-core-subset-20260803.json) — dsv4-flash-native-v11-core-subset-20260803
- [`inkling-small-marlin-baseline-20260806.json`](entries/inkling-small-marlin-baseline-20260806.json) — inkling-small-marlin-baseline-20260806
- [`inkling-small-marlin-mtp-20260807.json`](entries/inkling-small-marlin-mtp-20260807.json) — inkling-small-marlin-mtp-20260807

