# r0b0bench-vision v1.0

Frozen vision-language benchmark, 2026-09-09. Four complementary axes, deterministic
graders, single-image requests, thinking-off. **4,703 rows.**

## Suites (all pinned by HF revision)

| id | rows | axis | grading | license | revision |
|---|---:|---|---|---|---|
| `cvbench` | 2,638 | vision-centric 2D+3D: Count / Relation / Depth / Distance | MC letter (A–F) | Apache-2.0 | `bc284db50d03…` |
| `mmvp` | 300 (150 CLIP-blind pairs) | fine visual discrimination LLMs are blind to | 2-choice + paired metric | MIT | `37eafecab8a3…` |
| `realworldqa` | 765 | real-world scenes / spatial understanding | letter or normalized containment | CC-BY-ND-4.0 | `17e7f75e092e…` |
| `ocrbench` | 1,000 | text-in-image: recognition, formulas, tables, charts, KIE | official OCRBench containment | MIT | `92a54bd13843…` |

Sources: `nyu-visionx/CV-Bench`, `MMVP/MMVP`, `xai-org/RealworldQA`, `echo840/OCRBench`.
The exact revisions, row counts and grader ids live in `benchmark.json` (hash-bound into
every run summary as `contract_sha256`).

## Protocol (frozen)

- `chat_template_kwargs = {"enable_thinking": false}`, `temperature = 0`
- `max_tokens`: 32 for cvbench/mmvp/realworldqa, 64 for ocrbench
- image transport: base64 data URL (jpeg/png/webp/gif/bmp detected from magic bytes)
- appended instruction: `Answer with the letter or value only.`
- default 4 workers (the serve must allow ≥ 4 concurrent: `max_num_seqs ≥ 4`)

## Graders (deterministic, no LLM judge)

- **cvbench / mmvp**: extract the answer letter — exact single-letter reply first, else the
  first standalone allowed letter; compare against the gold letter (`(C)` → `C`).
- **mmvp paired**: a pair (questions 2i-1, 2i) scores only when *both* are correct — the
  headline paired accuracy is over 150 pairs.
- **realworldqa**: single-letter gold → letter match; otherwise normalized containment
  (lowercase, punctuation-stripped, whitespace-collapsed).
- **ocrbench**: the official rule from `Yuliang-Liu/MultimodalOCR` `OCRBench/example.py` —
  lowercased answer containment in the prediction; `HME100k` additionally strips all spaces.
  Per-type scores + the 1,000-point composite (recognition 300 + scene VQA 200 + doc VQA 200
  + KIE 200 + HME 100).

## Outputs

- `rows.jsonl` — one record per row: `id, suite, subaxis, prompt_sha256, image_sha256, gold,
  response, extracted, passed, finish_reason, elapsed_seconds, usage, image_id, profile_id,
  candidate_id, model`. Raw responses stay local; published evidence carries aggregates only.
- `summary.json` — per-suite accuracy + Wilson 95 % CI, per-subaxis breakdown, MMVP paired
  accuracy, OCRBench official type scores, identity bindings (`runner_sha256`,
  `contract_sha256`, dataset revisions), mean row latency.

## Usage

```bash
# fetch pinned data (once)
hf download nyu-visionx/CV-Bench   --repo-type dataset --revision bc284db50d036958861cb60cdd7b77612052ce0d --local-dir ~/rbv-data/cv-bench
hf download MMVP/MMVP              --repo-type dataset --revision 37eafecab8a3940c50c2ade5b36de69dbc99a8cf --local-dir ~/rbv-data/mmvp
hf download xai-org/RealworldQA    --repo-type dataset --revision 17e7f75e092e47169732462ea3cdfebe911105dd --local-dir ~/rbv-data/realworldqa
hf download echo840/OCRBench       --repo-type dataset --revision 92a54bd1384387c178d5a07140a2d85e0a3d12e1 --local-dir ~/rbv-data/ocrbench

# run (tmux for durability)
python3 run_vision.py --base-url http://127.0.0.1:8000 --data-dir ~/rbv-data \
  --out-dir ~/rbv-out/<run-id> --model <served-model> \
  --image-id sha256:<container-id> --profile-id <serve-profile-id> --candidate-id <label> \
  --workers 4 [--only cvbench,mmvp] [--max-rows 20]
```

## Scope and boundaries

- **Image-only** — the Ling-3.0-flash-VL fork does not support video, so every video
  benchmark (Video-MME, MMVU, …) is out of scope.
- **Single image per request** in all four suites; multi-image was verified to work on the
  serve but is not exercised here.
- **MMVP's paired metric has 150 pairs → ±8 pp** at p≈0.5; treat its headline as directional.
- **CC-BY-ND-4.0 (RealWorldQA)**: scores are published; no derived redistribution of images.
- Estimated wall time on one GB10 (Ling-3.0-flash-VL-NVFP4-MP, 4 workers, thinking-off):
  **~25–35 min** for the full 4,703 rows.
