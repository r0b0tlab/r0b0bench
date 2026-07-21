# Third-Party Notices

This repository currently contains an original implementation specification and
no vendored benchmark datasets, prompts, evaluator source, or model outputs.
The MIT License applies only to original r0b0bench material.

The planned implementation will integrate or depend on the following upstream
projects and datasets. Their own licenses and terms continue to apply:

- Berkeley Function-Calling Leaderboard (BFCL):
  https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- BFCL dataset:
  https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
- lm-evaluation-harness:
  https://github.com/EleutherAI/lm-evaluation-harness
- IFEval dataset and methodology:
  https://huggingface.co/datasets/google/IFEval
  https://arxiv.org/abs/2311.07911
- HumanEval:
  https://github.com/openai/human-eval
- Public task datasets named in the implementation plan, including GSM8K,
  ARC-Challenge, PIQA, Winogrande, TruthfulQA, and MMLU.

Before implementation or release, the project must create a machine-readable
per-dataset manifest containing source URL, exact revision, license, attribution,
redistribution status, content hash, item count, publishable fields, and
contamination-risk classification. Dataset content must be fetched from pinned
sources rather than committed here unless redistribution is independently
verified.

The 2026-07-21 calibration identity file publishes only task names, zero-based
indices, and prompt/instruction/task-ID hashes needed to recompute the reference
calibration digests. It does not redistribute benchmark content and does not
satisfy the required per-dataset license/revision manifest. Exact immutable
upstream revisions and license/publication policies therefore remain explicit
standalone-implementation release blockers.
