# r0b0bench leaderboard

Entries: **5** · regenerated from `results/entries/*.json`

Comparable only within the same profile and disclosed scorer variants.

## Quality

| entry_id | model | spec | GSM8K | HE@1 | QA | IFEval | BFCL-MT | ASTµ | NIAH | invalid |
|----------|-------|------|------:|-----:|---:|-------:|-------:|-----:|------|---------|
| dsv4-flash-native-v11-core-subset-20260803 | DeepSeek-V4-Flash DSpark NVFP4 (dua | dspark K=6 | 0.950 | 0.909 | 0.960 | 0.795 | 0.755 | 0.362 | PASS | False |
| inkling-small-marlin-baseline-20260806 | Inkling-Small NVFP4 (dual GB10, Mar | none | 0.810 | 0.732 | 0.245 | 0.425 | 0.540 | 0.302 | PASS | False |
| inkling-small-marlin-mtp-20260807 | Inkling-Small NVFP4 MTP 8-1-9 (dual | mtp draft=8 8-1-9 | 0.770 | 0.713 | 0.237 | 0.480 | 0.575 | 0.302 | NOT_RUN | False |
| ling-3.0-flash-nvfp4-basear-24k-20260807 | Ling-3.0-flash NVFP4 (base AR, CUDA | none | 0.920 | 0.713 | 0.960 | 0.855 | 0.595 | 0.310 | PASS | False |
| ling-3.0-flash-nvfp4-mtp1-graphs-20260807 | Ling-3.0-flash NVFP4 (MTP scale 1,  | none | 0.945 | 0.659 | 0.960 | 0.830 | 0.620 | 0.307 | PASS | False |

## Performance

| entry_id | model | spec | decode tok/s | prefill tok/s | TTFT ms | c1 agg | c2 agg | c4 agg | c6 agg |
|----------|-------|------|------------:|-------------:|-------:|-------:|-------:|-------:|-------:|
| dsv4-flash-native-v11-core-subset-20260803 | DeepSeek-V4-Flash DSpark NVFP4 (dua | dspark K=6 | 40.164 | 29203.666 | 221.884 | 86.931 | 145.889 | 239.311 | 337.761 |
| inkling-small-marlin-baseline-20260806 | Inkling-Small NVFP4 (dual GB10, Mar | none | 13.779 | — | — | 14.091 | 26.653 | 46.828 | 48.694 |
| inkling-small-marlin-mtp-20260807 | Inkling-Small NVFP4 MTP 8-1-9 (dual | mtp draft=8 8-1-9 | 15.831 | — | — | 16.388 | 29.983 | 47.876 | 63.222 |
| ling-3.0-flash-nvfp4-basear-24k-20260807 | Ling-3.0-flash NVFP4 (base AR, CUDA | none | 21.909 | 12021.781 | 232.760 | 21.755 | 47.547 | 88.538 | 113.953 |
| ling-3.0-flash-nvfp4-mtp1-graphs-20260807 | Ling-3.0-flash NVFP4 (MTP scale 1,  | none | 32.830 | 5952.850 | 227.885 | 39.023 | 51.836 | 96.890 | 126.867 |

## Files

- [`dsv4-flash-native-v11-core-subset-20260803.json`](entries/dsv4-flash-native-v11-core-subset-20260803.json) — dsv4-flash-native-v11-core-subset-20260803
- [`inkling-small-marlin-baseline-20260806.json`](entries/inkling-small-marlin-baseline-20260806.json) — inkling-small-marlin-baseline-20260806
- [`inkling-small-marlin-mtp-20260807.json`](entries/inkling-small-marlin-mtp-20260807.json) — inkling-small-marlin-mtp-20260807
- [`ling-3.0-flash-nvfp4-basear-24k-20260807.json`](entries/ling-3.0-flash-nvfp4-basear-24k-20260807.json) — ling-3.0-flash-nvfp4-basear-24k-20260807
- [`ling-3.0-flash-nvfp4-mtp1-graphs-20260807.json`](entries/ling-3.0-flash-nvfp4-mtp1-graphs-20260807.json) — ling-3.0-flash-nvfp4-mtp1-graphs-20260807

