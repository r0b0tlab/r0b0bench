# r0b0bench

A reviewed specification for a reproducible, provenance-bound benchmark suite for
OpenAI-compatible model endpoints.

> **Status: specification and current reference method.** A run-specific private
> harness is exercising the 8,620-case lifecycle, but the model-neutral package,
> contracts, and CLI have not yet been ported or released. Private campaign output
> is not automatically an r0b0bench result.

## Standard core suite

`r0b0bench-core-v1-rc2` candidate contains 8,620 logical cases:

| Pillar | Component | Cases | Primary output |
|---|---|---:|---|
| Tool use | Official BFCL v4 `multi_turn_base` | 200 | Official BFCL accuracy |
| Knowledge and reasoning | Frozen generated-answer tasks | 7,715 | Per-task accuracy and transparent macro/micro summaries |
| Instruction following | IFEval | 541 | Four official strict/loose metrics |
| Code | HumanEval | 164 | pass@1 in a digest-pinned, networkless sandbox |

The knowledge/reasoning, instruction-following, and code lanes total 8,420
cases; adding 200 official BFCL cases yields the 8,620-case core.

BFCL and the other components are independent lanes under one standardized run
identity. Non-BFCL tasks are not inserted into BFCL's official category namespace,
and v1 will not publish one opaque composite score. BFCL counts toward the 8,620
logical total while retaining its separate official lane metric. Two pre-admission
105-case calibration repeats are excluded from the standard-suite count.

## Current reference method

The 2026-07-21 `r0b0bench-core-v1-rc2` candidate replaced the historical short
completion caps with a fixed 32,768-token ceiling for BFCL, generated-answer,
IFEval, and HumanEval generation. Non-BFCL model requests use concurrency 16;
official BFCL evaluation uses a separately versioned adapter profile and remains
one thread. The method also requires two hash-bound calibration repeats,
exact-stack BFCL canaries, private immutable evidence, endpoint-attempt ledgers,
and separate lane reporting.

See [Current reference method and porting boundary](docs/REFERENCE_METHOD_2026-07-21.md).
This evidence updates the implementation contract but does not make the public
repository runnable or authorize publication of private campaign results.

## Core principles

- Official BFCL and IFEval scorers remain unmodified and version-pinned.
- Canaries stop for infrastructure, provenance, persistence, or sandbox faults—not
  ordinary wrong, empty, truncated, or unparsable model output.
- Request envelopes are suite-wide and immutable; no model-specific prompt,
  parser, token-budget, or reasoning-mode exceptions.
- Raw prompts, responses, traces, paths, and operational metadata remain private in
  a runtime root that must resolve outside the source checkout; in-checkout roots
  are rejected. Public reports are generated from explicit allowlisted schemas.
- Resume is semantic and hash-bound, not based on file existence.
- Runtime profile and public-benchmark contamination risk accompany every score.
- A completed low score is still a valid measurement. Release gates are separate.

## Documents

- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [Current reference method and porting boundary](docs/REFERENCE_METHOD_2026-07-21.md)
- [Calibration identity hash preimage](docs/CALIBRATION_IDENTITY_2026-07-21.json)
- [Independent review history](docs/REVIEW_HISTORY.md)
- [Third-party notices and source boundaries](THIRD_PARTY_NOTICES.md)
- [Contributing](CONTRIBUTING.md)

## Current milestone

The next milestone is to port the audited reference behavior into a reusable,
model-neutral package without copying deployment-specific paths or runtime identity:

1. freeze strict suite/dataset/request/result schemas for the 32,768-token
   reference contract;
2. add atomic private storage, descendant owner/lock verification, and one
   endpoint lease;
3. implement an exact-stack official BFCL vertical slice with opaque-state
   canonicalization and call/result binding;
4. add generated-answer, IFEval, and sandboxed HumanEval lanes;
5. qualify a fresh 8,620-case reference run and safe public projection before
   freezing v1.

There is currently no install command because there is no released package or
working CLI. Please do not report benchmark scores as r0b0bench results until the
implementation and release gates in the plan are complete.

## Upstream projects

r0b0bench plans to integrate, without misrepresenting:

- [Berkeley Function-Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [IFEval](https://arxiv.org/abs/2311.07911)
- [OpenAI HumanEval](https://github.com/openai/human-eval)

All third-party software, datasets, and benchmark content remain governed by their
respective licenses and terms.

## License

Original r0b0bench documentation and future original code are licensed under the
[MIT License](LICENSE). This does not relicense third-party benchmark software,
datasets, prompts, or test material.
