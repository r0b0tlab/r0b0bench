# r0b0bench leaderboard

Entries: **10** · regenerated from `results/entries/*.json`

Comparable only within the same profile and disclosed scorer variants.

## Quality

| entry_id | model | spec | GSM8K | HE@1 | QA | IFEval | BFCL-MT | ASTµ | NIAH | invalid |
|----------|-------|------|------:|-----:|---:|-------:|-------:|-----:|------|---------|
| dsv4-flash-native-v11-core-subset-20260803 | DeepSeek-V4-Flash DSpark NVFP4 (dua | dspark K=6 | 0.950 | 0.909 | 0.960 | 0.795 | 0.755 | 0.362 | PASS | False |
| inkling-small-marlin-baseline-20260806 | Inkling-Small NVFP4 (dual GB10, Mar | none | 0.810 | 0.732 | 0.245 | 0.425 | 0.540 | 0.302 | PASS | False |
| inkling-small-marlin-mtp-20260807 | Inkling-Small NVFP4 MTP 8-1-9 (dual | mtp draft=8 8-1-9 | 0.770 | 0.713 | 0.237 | 0.480 | 0.575 | 0.302 | NOT_RUN | False |
| ling-3.0-flash-nvfp4-basear-24k-20260807 | Ling-3.0-flash NVFP4 (base AR, CUDA | none | 0.920 | 0.713 | 0.960 | 0.855 | 0.595 | 0.310 | PASS | False |
| ling-3.0-flash-nvfp4-mtp1-graphs-20260807 | Ling-3.0-flash NVFP4 (MTP scale 1,  | none | 0.945 | 0.659 | 0.960 | 0.830 | 0.620 | 0.307 | PASS | False |
| nemotron-lightning-mtp-k1-core-subset-20260810 | NVIDIA Nemotron 3.5 Lightning 30B-A | mtp K=1 | 0.945 | 0.860 | 0.953 | 0.775 | 0.380 | 0.043 | PASS | False |
| nemotron-lightning-mtp-k1-thinking-on-core-subset-20260810 | NVIDIA Nemotron 3.5 Lightning 30B-A | mtp K=1 | 0.910 | 0.939 | 0.958 | 0.775 | 0.665 | 0.292 | PASS | False |
| qwen38-27b-nvfp4-sglang-dflash2-k8-core-subset-20260819 | Qwen3.8-27B NVFP4 (r0b0tlab 4-of-4) | dflash2 K=8 draft=8 block=8, think-off | 0.865 | 0.872 | 0.963 | 0.820 | 0.690 | 0.273 | PASS | False |
| qwen38-27b-nvfp4-sglang-dflash2-k8-core-subset-thinkon-low-20260819 | Qwen3.8-27B NVFP4 (r0b0tlab 4-of-4) | dflash2 K=8 draft=8 block=8, think-on reasoning_effort=low | 0.970 | 0.976 | 0.960 | 0.845 | 0.715 | 0.305 | PASS | False |
| qwen38-27b-nvfp4-vllm-dflash2-k8-core-subset-20260820 | Qwen3.8-27B NVFP4 (r0b0tlab 4-of-4) | dflash2 K=8 draft=8 num_speculative_tokens=8, z-lab Qwen3.8-27B-DFlash2 draft, think-off | 0.870 | 0.890 | 0.963 | 0.825 | 0.565 | 0.270 | PASS | False |

## Performance

| entry_id | model | spec | decode tok/s | prefill tok/s | TTFT ms | c1 agg | c2 agg | c4 agg | c6 agg |
|----------|-------|------|------------:|-------------:|-------:|-------:|-------:|-------:|-------:|
| dsv4-flash-native-v11-core-subset-20260803 | DeepSeek-V4-Flash DSpark NVFP4 (dua | dspark K=6 | 40.164 | 29203.666 | 221.884 | 86.931 | 145.889 | 239.311 | 337.761 |
| inkling-small-marlin-baseline-20260806 | Inkling-Small NVFP4 (dual GB10, Mar | none | 13.779 | 14109.463 | — | 14.091 | 26.653 | 46.828 | 48.694 |
| inkling-small-marlin-mtp-20260807 | Inkling-Small NVFP4 MTP 8-1-9 (dual | mtp draft=8 8-1-9 | 15.831 | 13235.308 | — | 16.388 | 29.983 | 47.876 | 63.222 |
| ling-3.0-flash-nvfp4-basear-24k-20260807 | Ling-3.0-flash NVFP4 (base AR, CUDA | none | 21.909 | 12021.781 | 232.760 | 21.755 | 47.547 | 88.538 | 113.953 |
| ling-3.0-flash-nvfp4-mtp1-graphs-20260807 | Ling-3.0-flash NVFP4 (MTP scale 1,  | none | 32.830 | 5952.850 | 227.885 | 39.023 | 51.836 | 96.890 | 126.867 |
| nemotron-lightning-mtp-k1-core-subset-20260810 | NVIDIA Nemotron 3.5 Lightning 30B-A | mtp K=1 | 90.954 | 28815.643 | 92.329 | 103.411 | 180.465 | 304.276 | 376.315 |
| nemotron-lightning-mtp-k1-thinking-on-core-subset-20260810 | NVIDIA Nemotron 3.5 Lightning 30B-A | mtp K=1 | 89.289 | 2031.666 | 93.628 | 99.470 | 151.647 | 217.046 | 252.004 |
| qwen38-27b-nvfp4-sglang-dflash2-k8-core-subset-20260819 | Qwen3.8-27B NVFP4 (r0b0tlab 4-of-4) | dflash2 K=8 draft=8 block=8, think-off | 26.022 | 22662.780 | 214.609 | 68.610 | 124.305 | 211.981 | 276.450 |
| qwen38-27b-nvfp4-sglang-dflash2-k8-core-subset-thinkon-low-20260819 | Qwen3.8-27B NVFP4 (r0b0tlab 4-of-4) | dflash2 K=8 draft=8 block=8, think-on reasoning_effort=low | 30.453 | 26320.080 | 216.447 | 61.063 | 113.852 | 187.481 | 252.588 |
| qwen38-27b-nvfp4-vllm-dflash2-k8-core-subset-20260820 | Qwen3.8-27B NVFP4 (r0b0tlab 4-of-4) | dflash2 K=8 draft=8 num_speculative_tokens=8, z-lab Qwen3.8-27B-DFlash2 draft, think-off | 21.770 | 826.400 | 259.074 | 67.057 | 121.492 | 211.523 | 279.250 |

## Files

- [`dsv4-flash-native-v11-core-subset-20260803.json`](entries/dsv4-flash-native-v11-core-subset-20260803.json) — dsv4-flash-native-v11-core-subset-20260803
- [`inkling-small-marlin-baseline-20260806.json`](entries/inkling-small-marlin-baseline-20260806.json) — inkling-small-marlin-baseline-20260806
- [`inkling-small-marlin-mtp-20260807.json`](entries/inkling-small-marlin-mtp-20260807.json) — inkling-small-marlin-mtp-20260807
- [`ling-3.0-flash-nvfp4-basear-24k-20260807.json`](entries/ling-3.0-flash-nvfp4-basear-24k-20260807.json) — ling-3.0-flash-nvfp4-basear-24k-20260807
- [`ling-3.0-flash-nvfp4-mtp1-graphs-20260807.json`](entries/ling-3.0-flash-nvfp4-mtp1-graphs-20260807.json) — ling-3.0-flash-nvfp4-mtp1-graphs-20260807
- [`nemotron-lightning-mtp-k1-core-subset-20260810.json`](entries/nemotron-lightning-mtp-k1-core-subset-20260810.json) — nemotron-lightning-mtp-k1-core-subset-20260810
- [`nemotron-lightning-mtp-k1-thinking-on-core-subset-20260810.json`](entries/nemotron-lightning-mtp-k1-thinking-on-core-subset-20260810.json) — nemotron-lightning-mtp-k1-thinking-on-core-subset-20260810
- [`qwen38-27b-nvfp4-sglang-dflash2-k8-core-subset-20260819.json`](entries/qwen38-27b-nvfp4-sglang-dflash2-k8-core-subset-20260819.json) — qwen38-27b-nvfp4-sglang-dflash2-k8-core-subset-20260819
- [`qwen38-27b-nvfp4-sglang-dflash2-k8-core-subset-thinkon-low-20260819.json`](entries/qwen38-27b-nvfp4-sglang-dflash2-k8-core-subset-thinkon-low-20260819.json) — qwen38-27b-nvfp4-sglang-dflash2-k8-core-subset-thinkon-low-20260819
- [`qwen38-27b-nvfp4-vllm-dflash2-k8-core-subset-20260820.json`](entries/qwen38-27b-nvfp4-vllm-dflash2-k8-core-subset-20260820.json) — qwen38-27b-nvfp4-vllm-dflash2-k8-core-subset-20260820

