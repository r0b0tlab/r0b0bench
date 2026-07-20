# r0b0bench v1 Standardized Suite Implementation Plan

> **For implementers:** Assume zero prior context. Follow the RED → GREEN →
> regression sequence. The repository currently publishes a reviewed specification,
> not a working benchmark runner. Do not publish package artifacts, benchmark
> scores, or leaderboard claims without completing the release gates below.

**Goal:** Build `r0b0bench`, a reproducible OpenAI-compatible endpoint benchmark
that runs official BFCL plus a frozen 8,420-item quality/instruction/code suite
under one provenance-bound lifecycle, without altering upstream scorers or
mistaking scoreable model behavior for infrastructure failure.

**Canonical repository:** `https://github.com/r0b0tlab/r0b0bench`

**Package / CLI:** planned Python package `r0b0bench`; planned command `r0b0bench`

**License:** MIT for original r0b0bench documentation and future code. Third-party
software, datasets, and benchmark material remain under their own licenses. Do
not redistribute benchmark datasets; fetch pinned revisions at setup time.

---

## 1. Executive architecture decision

Do **not** fork BFCL and do **not** insert GSM8K, MMLU, IFEval, or HumanEval into
BFCL's official category namespace. That would create scores that look official
but are not comparable to Berkeley's leaderboard.

Instead, build one outer orchestrator with independent, version-pinned lanes:

1. **Tool-use lane:** official `bfcl-eval==2025.12.17`, official
   `multi_turn_base`, official evaluator, 200 cases.
2. **Knowledge/reasoning lane:** 7,715 generated-answer cases across GSM8K,
   ARC-Challenge, PIQA, Winogrande, TruthfulQA MC1, and eight frozen MMLU
   subjects.
3. **Instruction-following lane:** official pinned IFEval scorer, 541 cases.
4. **Code lane:** HumanEval pass@1, 164 cases, in a digest-pinned networkless
   sandbox.
5. **Diagnostics:** completion coverage, extraction failures, length stops,
   empty-answer rate, latency, and token use. Diagnostics are never silently
   folded into accuracy.

The standardized core therefore contains **8,620 cases**: 200 BFCL plus 8,420
quality/instruction/code items. The scorecard reports each pillar separately.
There is no opaque single overall score in v1.

### Why this is the right boundary

- BFCL remains citable and bit-for-bit official.
- Quality scorers can evolve only by creating a new r0b0bench suite version.
- Model-format failures remain visible instead of being hidden by a permissive
  universal parser.
- A broken endpoint stops a run; a model that emits a wrong, empty, truncated,
  or unparsable answer is scored and the full suite continues.
- Archived evidence can be imported and validated without making private paths
  or prompts public.

---

## 2. Non-negotiable methodological rules

1. **Canaries test infrastructure, not intelligence.** A canary may stop a run
   only after the failure maps to the shared infrastructure/provenance taxonomy:
   endpoint or authentication failure, invalid HTTP/API schema, dataset or scorer
   drift, foreign/duplicate evidence, owner-lock loss, sandbox preflight failure,
   corrupt persistence, or a classified harness exception. A BFCL worker or
   evaluator crash is not a separate stop class: map it to `HARNESS_EXCEPTION`,
   `SCORER_DRIFT`, or another listed cause and apply the resume rules below.
   Wrong answers, missing tool calls, model-generated decode errors, empty
   content, `finish_reason=length`, and extraction failures are scoreable model
   behavior and must not stop the full run.
2. **No per-model adaptive token budget.** Every completion envelope is fixed by
   the suite contract. Changing one creates a new suite hash/version.
3. **Reasoning modes are distinct model variants.** `default`, `thinking-on`, and
   `thinking-off` runs may all exist, but their request bodies and hashes are
   bound separately. Never auto-disable thinking to rescue a score.
4. **Official scorers stay official.** Do not modify BFCL's evaluator or IFEval's
   scorer. If an upstream defect must be patched, record a separate adapter
   version and never label the result official until equivalence is demonstrated.
5. **No score mixing across suite hashes.** Comparison is allowed only when the
   suite contract, prompts, dataset pins, scorers, token envelopes, and sandbox
   contract match exactly. Cross-suite display must be explicitly marked
   non-comparable.
6. **Retries are transport-only.** Retry timeouts, connection resets, 429, and
   retryable 5xx responses under a frozen policy. Never retry a valid but wrong,
   empty, truncated, or unparsable model response.
7. **One endpoint owner.** BFCL and quality lanes run serially under one advisory
   endpoint/model lock. Do not overlap benchmark traffic with throughput tests.
8. **Raw evidence is private by default and external to the source checkout.**
   Resolve the repository root and private run root before binding a run, and
   reject any run root equal to or nested beneath the repository. `.gitignore`
   is defense in depth, never the privacy boundary. Public export contains
   path-free aggregates, hashes, versions, counts, and sanitized runtime
   identity—not raw prompts, API secrets, private hostnames, or absolute paths.
9. **Complete measurement is not a release PASS.** `COMPLETE` means every case
   has a valid persisted verdict. Release policy is a separate consumer of the
   scorecard.
10. **No benchmark-specific model patching.** Prompt/parser changes apply to all
    models in a new suite version; no model-name conditionals are permitted.

---

## 3. Frozen r0b0bench-core-v1 scope

| Pillar | Task | Count | Primary metric |
|---|---|---:|---|
| Tool use | BFCL v4 `multi_turn_base` | 200 | Official BFCL accuracy |
| Reasoning | GSM8K test, 0-shot | 1,319 | Exact match, flexible extract |
| Knowledge | ARC-Challenge test | 1,172 | Accuracy |
| Commonsense | PIQA validation | 1,838 | Accuracy |
| Commonsense | Winogrande XL validation | 1,267 | Accuracy |
| Truthfulness | TruthfulQA MC1 validation | 817 | MC1 accuracy |
| Knowledge | MMLU abstract algebra | 100 | Accuracy |
| Knowledge | MMLU business ethics | 100 | Accuracy |
| Knowledge | MMLU clinical knowledge | 265 | Accuracy |
| Knowledge | MMLU college biology | 144 | Accuracy |
| Knowledge | MMLU computer security | 100 | Accuracy |
| Knowledge | MMLU conceptual physics | 235 | Accuracy |
| Knowledge | MMLU high-school world history | 237 | Accuracy |
| Knowledge | MMLU international law | 121 | Accuracy |
| Instructions | IFEval train | 541 | Official strict/loose prompt + instruction metrics |
| Code | HumanEval test | 164 | pass@1 |

Verified subtotals:

- Generated-answer base tasks: 7,715
- Eight MMLU subjects: 1,302
- Balanced quality plus IFEval: 8,256
- Quality plus HumanEval: 8,420
- Full r0b0bench core including BFCL: 8,620

### Initial fixed request envelopes

These values are implementation defaults for `r0b0bench-core-v1-rc1`. Phase 8
must calibrate them before the immutable `v1` tag. If calibration changes a
value, create `rc2`; never mutate a released contract.

| Lane | Temperature | Maximum completion tokens | Concurrency |
|---|---:|---:|---:|
| BFCL | Official `0.001` | Upstream BFCL behavior | 1 thread |
| Multiple choice | 0 | 2,048 | 1 |
| GSM8K | 0 | 2,048 | 1 |
| IFEval | 0 | 4,096 | 1 |
| HumanEval | 0 | 2,048 | 1 |

For non-BFCL lanes, RC1 uses a 10-second connect timeout, 900-second response
timeout, and at most three wire attempts. Retry only connection/reset/timeout
failures and HTTP 408, 429, 500, 502, 503, or 504, with deterministic 2-second
then 10-second backoff and no jitter. Do not retry any valid 2xx model response
or nonlisted 4xx. Persist every attempt, distinguish total wall time from the
scored attempt latency, and flag post-send timeout retries as transport-ambiguous.
BFCL retains its pinned upstream request policy; the package/tree hash binds it.

Canonical non-BFCL RC1 requests send `seed=0`; endpoint rejection of that field
is a compatibility failure for the seeded canonical profile, not permission to
drop it mid-run. An explicitly versioned `seed-uncontrolled` profile may be
added later but is not directly comparable. Official BFCL remains
`upstream-unseeded` because injecting a seed would alter its request contract.
Record visible, reasoning (when exposed), and total completion tokens separately.

Rationale: the historical 192-token MC envelope repeatedly terminated a
reasoning model before its final answer. A fixed generous envelope lets the
model stop naturally, while token count and latency expose the efficiency cost.
There is no early stop or cap escalation based on model identity or observed
quality.

### Answer contracts

- **MC:** ordered, versioned extractor accepts only an explicit final answer
  marker, boxed letter, `answer is X`, or a final line containing only one valid
  choice letter. It must never infer from option frequency or arbitrary letters
  in reasoning.
- **Response fields:** generated-answer, IFEval, and HumanEval lanes score only
  user-visible assistant `content`. Preserve `reasoning_content` for diagnostics,
  but never scavenge an answer from hidden reasoning. BFCL scores structured tool
  calls through its official handler.
- **GSM8K:** use the pinned lm-eval flexible-extract semantics for a 0-shot chat
  prompt. Preserve exact numeric normalization and expose extraction failure
  separately.
- **IFEval:** score with the exact pinned official implementation; do not create
  a replacement heuristic.
- **HumanEval:** extract one Python candidate under a versioned rule and execute
  once in the frozen sandbox. No repair loop.
- **BFCL:** use the official model handler and evaluator. BFCL decode/tool errors
  are model outcomes unless the harness itself crashes or loses rows.

### Comparability and claim language

- Label the 200-case tool row exactly `Official BFCL v4 multi_turn_base`; never
  call it full BFCL V4 or BFCL overall.
- Label generated-answer results `r0b0bench-core-v1 <task>` because the frozen
  chat prompts, answer contracts, and completion envelopes are part of this
  suite. Do not present them as drop-in lm-eval leaderboard rows unless an
  explicit equivalence test proves that claim.
- Label IFEval metrics with dataset revision, scorer package/hash, strict/loose,
  and prompt/instruction level.
- Label HumanEval `pass@1`, one sample, with extraction version and sandbox
  digest. Never compare it to pass@k or repair-loop results.
- Report model/runtime/variant identity next to every score. A reasoning-mode or
  chat-template/runtime change is a different variant even under the same model
  weights.
- Treat runtime profile identity separately from benchmark-method identity.
  Speculative decoding, KV dtype, quantization, chat template, parser, container
  image, and distributed topology qualify the result; never present a
  profile-qualified deployment score as runtime-neutral model capability.
- These are public benchmarks with material contamination risk. Record declared
  model training cutoff and known benchmark exposure, assign each task a
  contamination-risk label, and make no contamination-free or broad
  generalization claim. Add overlap checks where feasible; a private/rotating
  holdout is a future leaderboard-resistance lane, not a replacement score.

---

## 4. Prior-art gate and reuse decisions

### Reuse directly

- `bfcl-eval==2025.12.17`, corresponding to official leaderboard commit
  `f7cf7359b7ac615a0b294831c5ba2bc95ee4a000`.
- BFCL's in-process `generation_main` and `evaluation_main` APIs after registering
  the served model in the same worker process.
- `lm_eval==0.4.12` IFEval scorer for the legacy-equivalent v1 contract, pinned
  by package version and source hash.
- Dataset counts, fingerprints, and content hashes recovered from the Hy3
  Balanced Chat-8 run.
- Digest-pinned, networkless, read-only, capability-dropped HumanEval sandbox
  design from the recovered quality runner.

### Reuse as regression evidence, not production modules

Historical campaign runners are private, run-specific evidence and may contain
hard-coded paths or deployment identity. Reimplement only their documented,
attributed behavior behind clean adapter contracts; never import private campaign
scripts at runtime or commit private archive rows to this repository.

### Do not merge into r0b0bench core

- **HermesBench** measures models inside the
  real Hermes Agent harness and has a different unit of evaluation. Keep it as a
  future optional external lane; do not rename or absorb it.
- **Inspect Evals' BFCL port** is useful architectural prior art, but its own
  documentation lists category gaps, implementation differences, and possible
  false negatives. Use official BFCL as the scoring authority.
- **The old Hy3 request caps** (`192` MC, `512` GSM8K) become an importable
  historical profile only. They are not the r0b0bench v1 standard.
- **Legacy shell controllers** such as `bfcl_then_quality.sh`,
  `finalize_bfcl_then_quality.sh`, and `run_hy3_quality_suite.sh` are run-bound
  evidence, not reusable orchestration. Their lock/manifest interfaces have
  drifted; do not copy or invoke them from r0b0bench.

### Authoritative upstream references

- BFCL code and methodology:
  `https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard`
- BFCL official leaderboard: `https://gorilla.cs.berkeley.edu/leaderboard.html`
- BFCL dataset: `https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard`
- lm-evaluation-harness: `https://github.com/EleutherAI/lm-evaluation-harness`
- IFEval dataset/paper: `https://huggingface.co/datasets/google/IFEval` and
  `https://arxiv.org/abs/2311.07911`
- HumanEval source: `https://github.com/openai/human-eval`

---

## 5. Repository layout to create

```text
r0b0bench/
├── AGENTS.md
├── LICENSE
├── NOTICE
├── THIRD_PARTY_NOTICES.md
├── README.md
├── pyproject.toml
├── uv.lock
├── contracts/
│   ├── suites/r0b0bench-core-v1-rc1.yaml
│   ├── datasets/v1.json
│   ├── prompts/v1/*.txt
│   ├── algorithms/request-v1.json
│   ├── algorithms/response-normalization-v1.json
│   ├── algorithms/extraction-v1.json
│   ├── schemas/run-manifest-v1.json
│   ├── schemas/result-row-v1.json
│   └── schemas/scorecard-v1.json
├── src/r0b0bench/
│   ├── __init__.py
│   ├── cli.py
│   ├── contract.py
│   ├── endpoint.py
│   ├── identity.py
│   ├── manifest.py
│   ├── persistence.py
│   ├── locking.py
│   ├── retry.py
│   ├── orchestrator.py
│   ├── validation.py
│   ├── scorecard.py
│   ├── report.py
│   ├── export.py
│   ├── adapters/
│   │   ├── base.py
│   │   ├── bfcl.py
│   │   ├── bfcl_worker.py
│   │   ├── generated_answer.py
│   │   ├── ifeval.py
│   │   └── humaneval.py
│   ├── scoring/
│   │   ├── multiple_choice.py
│   │   ├── gsm8k.py
│   │   └── humaneval.py
│   └── sandbox/
│       └── humaneval.py
├── tests/
│   ├── fixtures/
│   │   ├── synthetic_bfcl/
│   │   ├── synthetic_response_failures/
│   │   └── mock_endpoint/
│   ├── unit/
│   ├── contract/
│   ├── integration/
│   └── security/
├── scripts/
│   ├── import_archived_evidence.py
│   ├── public_safety_scan.py
│   └── build_third_party_notices.py
└── docs/
    ├── METHODOLOGY.md
    ├── DATASETS.md
    ├── SCORING.md
    ├── REPRODUCIBILITY.md
    ├── SECURITY.md
    └── ADDING_A_LANE.md
```

### External run layout

Runtime state never lives inside the Git checkout. The default state root is
`${XDG_STATE_HOME:-$HOME/.local/state}/r0b0bench`; an explicit override must
also resolve outside the repository. The canonical layout is:

```text
<external-state-root>/runs/<run-id>/
├── intent.json
├── run-manifest.json
├── run-manifest.sha256
├── status.json
├── owner.json
├── private/rows/<lane>/<task>/<ordinal>.json
├── private/raw/bfcl/{result,score}/...
├── summaries/<lane>.json
├── reports/{scorecard.json,REPORT.md,report.html}
└── private/logs/<lane>.log
```

At startup, resolve symlinks for both roots and fail before creating files if
the run root is the repository or one of its descendants. Create run
directories as `0700` and private files as `0600` unless deliberately frozen
read-only. Public exports also default to an external staging root; only their
strict allowlisted projection may later be copied into a publication artifact.
The repository intentionally does not ignore `runs/`, `private/`, or
`archives/`, so an accidental in-checkout tree remains visible to Git and fails
the publication scan.

`run-manifest.json` and its sidecar become read-only after endpoint/suite binding.
Mutable progress lives only in atomically replaced `status.json`. Every result row
uses a zero-padded numeric ordinal filename and contains the logical stable item
ID inside the validated row, preventing benchmark-controlled path traversal.
Every row also contains the run ID, suite hash, task ID, prompt hash, request
hash, response hash, attempt metadata, verdict class, scorer version/hash, and
latency/token fields.

---

## 6. Failure taxonomy

Every item or run transition must use one of these explicit classes:

### Infrastructure failures — stop or leave run resumable

- `ENDPOINT_UNREACHABLE`
- `AUTHENTICATION_FAILED`
- `API_SCHEMA_INVALID`
- `DATASET_DRIFT`
- `SCORER_DRIFT`
- `SANDBOX_PREFLIGHT_FAILED`
- `PERSISTENCE_CORRUPT`
- `FOREIGN_OR_DUPLICATE_ROW`
- `OWNER_LOCK_LOST`
- `HARNESS_EXCEPTION`

Use only this list for canary termination. `ENDPOINT_UNREACHABLE`,
`AUTHENTICATION_FAILED`, `SANDBOX_PREFLIGHT_FAILED`, `OWNER_LOCK_LOST`, and a
non-deterministic `HARNESS_EXCEPTION` may become `INCOMPLETE_RESUMABLE` only when
the immutable binding is unchanged and every persisted prefix row revalidates.
`DATASET_DRIFT`, `SCORER_DRIFT`, `FOREIGN_OR_DUPLICATE_ROW`, and
`PERSISTENCE_CORRUPT` make the run `INVALID`. `API_SCHEMA_INVALID` is resumable
only when no response evidence has been admitted and the same bound endpoint can
be corrected without changing methodology; otherwise it is `INVALID`.

For BFCL specifically, an evaluator that cannot initialize or score because its
package/tree or canonical data changed is `SCORER_DRIFT`; an unexpected worker
process failure with unchanged identities is `HARNESS_EXCEPTION`. Authentication
maps to `AUTHENTICATION_FAILED`, invalid endpoint responses map to
`API_SCHEMA_INVALID`, and row ownership violations map to
`FOREIGN_OR_DUPLICATE_ROW`. Officially scoreable model-generated tool/decode
errors remain model outcomes even if they lower or zero the canary score.

### Model outcomes — score and continue

- `CORRECT`
- `WRONG`
- `UNEXTRACTABLE`
- `EMPTY_CONTENT`
- `LENGTH_LIMITED`
- `TOOL_DECODE_FAILED`
- `TOOL_SELECTION_WRONG`
- `CODE_FAILED`
- `INSTRUCTION_FAILED`

A row may carry multiple diagnostics (for example `WRONG`, `UNEXTRACTABLE`, and
`LENGTH_LIMITED`) but exactly one primary score outcome. Never turn a model
outcome into an infrastructure failure merely because a canary threshold is
crossed.

---

## 7. CLI contract

```bash
r0b0bench doctor --suite r0b0bench-core-v1-rc1
r0b0bench inspect --base-url http://127.0.0.1:8888/v1 --model MODEL
r0b0bench run \
  --suite r0b0bench-core-v1-rc1 \
  --base-url http://127.0.0.1:8888/v1 \
  --model MODEL \
  --variant default \
  --run-id MODEL_core_v1_DATE
r0b0bench resume --run-id MODEL_core_v1_DATE
r0b0bench status --run-id MODEL_core_v1_DATE --json
r0b0bench validate --run-id MODEL_core_v1_DATE
r0b0bench score --run-id MODEL_core_v1_DATE
r0b0bench report --run-id MODEL_core_v1_DATE
r0b0bench compare RUN_A RUN_B
r0b0bench export --run-id MODEL_core_v1_DATE --public-dir "$PUBLIC_EXPORT_ROOT/RUN_ID"
```

API keys come only from environment variables or an external credential command;
they are never written to config, manifests, rows, logs, reports, or command
examples. `inspect` and `doctor` perform no benchmark requests beyond explicit
schema/health probes.

`run` owns the immutable manifest and one endpoint lease. `status`, `validate`,
`score`, `report`, `compare`, and `export` are endpoint-independent and make no
inference or health-probe requests. `validate` never repairs evidence; `score`
and `report` consume only validated persisted rows and never implicitly rerun a
benchmark.

Exit codes:

- `0`: requested operation complete and structurally valid, regardless of score
- `1`: an explicitly requested external `gate` policy failed
- `2`: incomplete/resumable infrastructure failure
- `4`: invalid user input or contract
- `5`: provenance/security violation

The `run` command exits `0` when the suite is complete even if accuracy is low;
accuracy is the measurement. Only an explicit external policy command such as
`r0b0bench gate` may convert a structurally complete low score into exit `1`.

---

## 8. Implementation tasks (TDD, in order)

For every RED task below, tests must collect cleanly. Defer imports inside test
helpers or provide only an interface declaration so the focused run fails with a
clear missing-behavior assertion—not `ImportError`, collection ERROR, or a
network/setup accident. Run the unchanged baseline separately from intentional
RED tests, and never weaken the contract to manufacture GREEN. After each GREEN
phase, make one narrow local commit containing only that phase, record its full
SHA in the development log, and inspect the staged path list. Do not push
intermediate phase commits to the canonical remote; push only an accepted phase
candidate after its public-safety and exact-SHA review gates pass.

### Phase 0 — Create the local project boundary

#### Task 0.1: Initialize only the new repository

**Create:** repository skeleton, `AGENTS.md`, `.gitignore`, `README.md`, license
files, and `pyproject.toml`.

**Dependencies:** Python 3.12. Keep base installation lightweight: `click`,
`pydantic>=2.8,<3` in strict mode, `PyYAML`, `httpx`, and `jsonschema`. Define
optional extras: `bfcl` (`bfcl-eval==2025.12.17`, `soundfile`), `quality`
(`datasets==4.3.0`, `lm-eval==0.4.12`), `full` (both runtime lanes), and `dev`
(`pytest`, `ruff`, `mypy`, security/build tooling). Lock exact transitive versions
for every extra with `uv.lock`.

**Commands:**

```bash
git clone https://github.com/r0b0tlab/r0b0bench.git
cd r0b0bench
uv venv --python 3.12
uv sync --extra full --extra dev
```

**Verify:** import `r0b0bench`; `r0b0bench --help`; clean `git diff --check`.
Do not replace or retarget the canonical remote during implementation.

#### Task 0.2: Complete license and attribution audit

**Create:** `THIRD_PARTY_NOTICES.md` and a machine-readable per-dataset
publication manifest with source URL, exact revision, license, attribution,
redistribution status, content hash, item count, publishable fields, and
contamination-risk label. Record whether prompts, datasets, code, model outputs,
or only IDs/hashes may be redistributed. Default to fetching datasets rather
than vendoring them.

**Test:** `tests/security/test_third_party_notices.py` requires every contract
source to have URL, license, revision, attribution, redistribution/publication
policy, and contamination-risk classification.

---

### Phase 1 — Freeze strict contracts before runners

#### Task 1.1: Write failing suite-contract tests

**Create:**

- `tests/contract/test_suite_contract.py`
- `tests/contract/test_dataset_contract.py`
- `tests/contract/test_contract_hash.py`
- `contracts/suites/r0b0bench-core-v1-rc1.yaml`
- `contracts/datasets/v1.json`

Test exact task set, counts, request envelopes, unknown-key rejection, strict
booleans/integers, range bounds, duplicate task IDs, normalized round-trip, and
stable canonical hash. Require the total `8,620` and quality subtotal `8,420`.

**RED:** tests fail because no parser exists.

#### Task 1.2: Implement strict immutable contract models

**Create:** `src/r0b0bench/contract.py`.

- Reject unknown fields and implicit type coercion.
- Normalize to canonical JSON before hashing.
- Include referenced prompt/scorer/dataset content hashes in the suite identity.
- Include request serialization, response-field normalization, extraction,
  retry-policy, and sandbox semantic-contract hashes in the suite identity.
- Do not let comments or filesystem paths affect the canonical suite hash.
- Resolve all relative references beneath the repository contract root; reject
  traversal and symlinks escaping it.

**GREEN:** contract tests pass. Add mutation tests proving one changed token cap,
prompt byte, dataset revision, scorer hash, request/response algorithm contract,
retry policy, or sandbox digest changes suite ID.

#### Task 1.3: Freeze prompt and extractor fixtures

**Create:** prompt files and `tests/unit/test_mc_extractor.py`,
`tests/unit/test_gsm8k_extractor.py`.

MC fixtures must cover explicit final markers, boxed choices, final-line choices,
multiple conflicting markers (last valid explicit final marker wins), lowercase,
invalid choice range, option letters in reasoning, empty content, truncated
reasoning, and prompt-injection-like answer text. GSM8K fixtures must match the
pinned lm-eval flexible-extract behavior.

**Implement:** `scoring/multiple_choice.py` and `scoring/gsm8k.py` only after RED.
Do not include model-name branches.

---

### Phase 2 — Build provenance, atomic rows, and endpoint ownership

#### Task 2.1: Freeze manifest and row schemas

**Create:** JSON schemas and tests for immutable intent/manifest, mutable status,
owner identity, result rows, lane summary, and scorecard.

Required manifest fields include suite ID/hash, run ID, UTC creation time, model
name, variant name, sanitized base URL identity, endpoint catalog snapshot,
normalized request extras hash, package/scorer versions, dataset hashes, retry
policy, concurrency, runner package version, git tree/commit identity or installed
wheel SHA-256, and sandbox digest. Secret values must be structurally impossible
to serialize.

#### Task 2.2: Implement atomic persistence

**Create:** `persistence.py` and failure-injection tests.

Use same-directory temp files, flush, `fsync`, atomic replacement, and directory
`fsync`. Create run directories as user-private (`0700`) and evidence files as
`0600` unless explicitly read-only; reject ancestor/final symlinks, non-regular
files, foreign ownership, and unexpected hard links. Validate an existing row
semantically before resuming. A nonempty file is not automatically valid.
Quarantine corrupt/foreign rows; never silently skip them.

If directory `fsync` fails after `os.replace`, preserve the replaced target and
report an ambiguous durability failure; never unlink the newly committed target.
Resolve state by reopening/revalidating the target and on restart.

Test crashes before write, before replace, and after replace; duplicate IDs;
wrong prompt hash; wrong suite hash; malformed JSON; short write; and stale temp
files.

#### Task 2.3: Implement owner and endpoint locks

**Create:** `locking.py` and tests with real child processes.

Bind owner PID plus `/proc` start ticks, run ID, suite hash, endpoint identity,
and model identity in mutable `owner.json`, never in immutable
`run-manifest.json`. Use advisory locks under a user-private cache directory keyed
by endpoint/model hash. The top-level `run` process acquires and holds the one
`EndpointLease` across every adapter; adapters must never acquire, replace, or
release endpoint ownership. Verify the inherited descriptor in workers. A stale
PID must not be mistaken for a live owner after PID reuse.

#### Task 2.4: Implement endpoint and response normalization

**Create:** `endpoint.py`, `retry.py`, and a deterministic mock OpenAI server
fixture.

Normalize `content`, `reasoning_content`, tool calls, usage, finish reason, raw
response ID, and HTTP attempt history without discarding the original response
hash. Empty `content` with populated reasoning is a model outcome, not an API
schema failure. Require strict mandatory fields/counters, allow documented
optional fields and nested `null` detail objects, and tolerate additive provider
metadata without scoring it. Retry only frozen transport classes.

Test 200 valid, 200 empty, reasoning-only content, legal nested-null usage,
additive provider fields, malformed required counters, 200 malformed tool
arguments, 400, 401, 429, retryable 5xx, timeout then success, and timeout
exhaustion using sanitized real-shape fixtures.

---

### Phase 3 — Implement the generated-answer quality lane

#### Task 3.1: Dataset loaders and drift gates

**Create:** `adapters/generated_answer.py` and dataset tests using cached/small
fixtures. Pin repository, config, split, revision, count, Hugging Face
fingerprint, and canonical content SHA-256 for all 13 base tasks.

The full preflight loads and verifies all datasets before sending any model
request. Dataset drift is infrastructure failure.

#### Task 3.2: Per-item runner and resume

For each item:

1. Render the frozen prompt.
2. Compute prompt and request hashes.
3. Reuse only a fully valid bound row.
4. Send one request under the fixed envelope.
5. Classify response diagnostics.
6. Score using the versioned extractor.
7. Atomically persist the row.
8. Continue regardless of model score.

Add integration test where every canary answer is empty or length-limited; the
runner must still attempt every fixture item and produce a complete summary.

#### Task 3.3: Base-task summaries

Report per task: expected/completed/scored counts, correct, accuracy, extraction
failures, empty content, length stops, transport retries, completion-token total,
and latency p50/p95. Publish three explicitly named summaries in addition to all
per-task scores:

1. `base_task_macro_13`: unweighted mean across the 13 base tasks.
2. `base_item_micro`: item-weighted accuracy across all 7,715 base rows.
3. `mmlu8_macro_accuracy`: unweighted mean of the eight selected MMLU subjects,
   feeding `quality_family_macro_accuracy`, the unweighted mean of six benchmark
   families: GSM8K, ARC-Challenge, PIQA, Winogrande, TruthfulQA MC1, and MMLU8.

The family macro is the preferred high-level quality summary because it prevents
the eight MMLU subjects from receiving eight votes; the 13-task macro and micro
remain transparent diagnostics. Per-task scores are primary. Never call any of
these BFCL or overall r0b0bench accuracy.

---

### Phase 4 — Integrate official BFCL without forking it

#### Task 4.1: Isolated BFCL worker

**Create:** `adapters/bfcl_worker.py`.

The parent launches a dedicated Python worker with `BFCL_PROJECT_ROOT` set to
`<external-run-root>/private/raw/bfcl` **before importing `bfcl_eval`**. The
resolved worker root must pass the same outside-repository check. Inside that
same worker:

1. Register the served model in `MODEL_CONFIG_MAPPING`.
2. Validate package version, installed wheel/full package-tree hash, official
   dataset count, ordered ID set, and dataset hash. Pinning selected source files
   alone is insufficient.
3. Call `generation_main` in-process with `skip_server_setup=True`, one thread,
   and official temperature.
4. Validate 200 rows and 200 unique IDs. Classify persisted row errors by cause:
   transport, authentication, endpoint-schema, or harness exceptions are
   infrastructure failures; official BFCL decode/tool-format failures that the
   evaluator can score remain model outcomes.
5. Call `evaluation_main` in-process and require it to score the complete
   official category without synthesizing or editing rows.
6. Parse the official score artifact and write a path-free lane summary.

Do not register in the parent and then invoke the bare `bfcl` CLI; that loses the
in-memory model mapping.

#### Task 4.2: BFCL canary semantics

Run three official IDs to prove persistence and evaluator compatibility. A
wrong tool call or model-generated decode failure is acceptable and scoreable.
Stop only through the shared taxonomy above: an unrepresentable HTTP response is
`API_SCHEMA_INVALID`; invalid row ownership is `FOREIGN_OR_DUPLICATE_ROW`; and a
worker/evaluator failure must first be classified as `HARNESS_EXCEPTION` or
`SCORER_DRIFT`. Never use a low or zero partial score as a stop condition.

Delete/separate partial canary output before the full run so it cannot
masquerade as a 200-row completion. Use distinct `private/raw/bfcl-canary` and
`private/raw/bfcl` roots. On full-run resume, parse BFCL JSONL line by line, preserve and
hash every complete valid prefix row, and repair only a provably incomplete final
line; malformed middle rows, duplicate IDs, foreign IDs, or bound-prefix drift
are provenance failures. Bind completion to worker PID/start time, suite/run
identity, package version, category, dataset hash, prefix hash, and score artifact
hash.

#### Task 4.3: Private BFCL archive regression

Add an opt-in local regression test that reads an external, hash-bound BFCL
archive only through `R0B0BENCH_ARCHIVE_ROOT`. It must verify frozen source
hashes, expected/unique row counts, absence of duplicates and top-level transport
errors, and equality with the archive's bound official aggregate.

Do not hard-code or publish model-specific private scores. Do not ship private raw
prompts, rows, input logs, or absolute paths in test fixtures. Public CI uses
synthetic BFCL rows for schema/error coverage; the archive regression is a local
pre-release gate and cleanly skips only when the external archive is intentionally
unavailable.

---

### Phase 5 — IFEval and HumanEval lanes

#### Task 5.1: Official IFEval adapter

**Create:** `adapters/ifeval.py`.

Pin `lm_eval==0.4.12` and the scorer source hash. Validate 541 rows. Persist
strict/loose prompt-level booleans and instruction-level arrays, then aggregate
all four official metrics. A failed instruction is a model outcome.

Tests must recompute every persisted row from its response and reject modified
scorer output, wrong instruction lists, and dataset drift.

#### Task 5.2: Hardened HumanEval sandbox

**Create:** `sandbox/humaneval.py`, `adapters/humaneval.py`, and security tests.

Required container policy:

- digest-pinned Python image
- `--network none`
- read-only root filesystem
- drop all capabilities
- no new privileges
- bounded PIDs, memory, CPU, and wall time
- private per-item work directory
- no-follow, single-link, owner-checked file creation
- deterministic hash seed

Run positive execution and explicit socket-denial preflights before any model
code. If sandbox isolation fails, stop as infrastructure failure. If candidate
code fails, score it and continue.

No repair attempts; primary metric is pass@1.

---

### Phase 6 — Orchestration, resume, and full-suite semantics

#### Task 6.1: Build a serial lane state machine

**Create:** `orchestrator.py` with a typed `LaneAdapter` protocol owned by
`adapters/base.py`: `preflight`, `run_or_resume`, `validate`, and `summarize`.
Adapters may write only their assigned row/raw namespaces through the persistence
API; the orchestrator alone updates global status, transitions lanes, and emits
completion markers. Keep adapters internal in v1; defer third-party entry-point
plugins until built-in contracts and provenance are stable. Use explicit states:

`CREATED → PREFLIGHT → BOUND → BFCL → BASE_QUALITY → IFEVAL → HUMANEVAL → VALIDATING → COMPLETE`

Any infrastructure fault becomes `INCOMPLETE_RESUMABLE` or `INVALID` depending
on whether provenance remains trustworthy. Model outcomes never change the lane
transition.

Historical archive imports are reports, not live runs. They never enter this
state machine, receive an endpoint lease, become resumable, or satisfy
`validate`, scorecard completeness, release admission, or the 8,620-case count.

#### Task 6.2: Resume correctness

Test termination at every state and mid-task, including `SIGINT`, `SIGTERM`,
and an ungraceful worker kill. Graceful signals finish or atomically abandon the
current row, remove no valid evidence, publish no completion marker, and leave a
resumable status. Resume must:

- reacquire the same endpoint/model lock,
- verify endpoint and suite identity,
- validate all existing rows,
- regenerate only absent/invalid owned rows,
- never change prompts, budgets, retry policy, or scorer,
- preserve one run ID and one immutable manifest,
- refuse endpoint/model/variant drift.

#### Task 6.3: Full count validator

`r0b0bench validate` requires exactly 8,620 expected case identities, complete
lane summaries, official BFCL score, all IFEval aggregates, all HumanEval
verdicts, no foreign/duplicate rows, and matching hashes. Low accuracy is valid;
missing evidence is not.

---

### Phase 7 — Scorecard, report, compare, and safe export

#### Task 7.1: Scorecard schema

Create separate pillars:

- `bfcl_multi_turn_base_accuracy`
- 13 per-task knowledge/reasoning accuracies
- `mmlu8_macro_accuracy`
- `base_task_macro_13`
- `base_item_micro`
- `quality_family_macro_accuracy`
- four official IFEval metrics
- `humaneval_pass_at_1`
- format diagnostics by task
- efficiency metrics by lane
- completeness/provenance status

For every proportion, publish numerator, denominator, decimal score, and a
supplemental 95% Wilson interval. Keep official BFCL/IFEval/HumanEval primary
values untouched; intervals are r0b0bench diagnostics. For same-suite model
comparisons, join by stable item ID and report paired correctness deltas in
addition to aggregate deltas.

Do not create a composite rank in v1. If a future composite is desired, version
its weights separately and publish the full formula.

#### Task 7.2: Comparison rules

`compare` rejects different suite hashes by default. It labels model variant,
reasoning mode, runtime identity, and request-extra hash. Report absolute and
relative deltas per pillar with sample counts; do not hide uncertainty or
missing lanes.

#### Task 7.3: Public export safety

Create a strict allowlisted public projection schema containing only run/manifest
digests, suite/adapter versions, counts, aggregate metrics, normalized diagnostic
categories, and publishable dataset/scorer identifiers/hashes. Exclude prompts,
generated text, reasoning, tool calls, headers, response bodies, tracebacks,
credentials, local paths, container commands, and raw BFCL artifacts. Do not use
a broad recursive redactor as the primary boundary; serialization into the
public schema must make private fields unrepresentable.

Run a second scanner that fails on home paths, LAN addresses, hostnames,
bearer/API keys, `.env`, raw prompts, tracebacks, and private project roots. Test
malicious values embedded in model output and metadata. `report` and `export`
must be endpoint-independent, must never make inference requests, and must never
implicitly rerun scoring.

---

### Phase 8 — Fix the historical truncation issue and freeze v1

#### Task 8.1: Import a private historical canary

The opt-in, hash-bound local regression importer must reproduce the row count,
score, extraction, empty-content, and length-stop classifications recorded in an
external archive manifest supplied through `R0B0BENCH_ARCHIVE_ROOT`.

This verifies only that the importer reproduces the archive's bound historical
classifications. It does not prove current adapter equivalence, live-run failure
semantics, or implementation correctness; those require synthetic/public
fixtures plus fresh contract and integration gates. Emit a report classified
`HISTORICAL_PARTIAL_IMPORT`; do not assign it a live-run status. It is neither
`INVALID` nor an infrastructure failure, but it cannot satisfy full-count
validation, produce a releasable scorecard, be resumed, or authorize a
benchmark-result claim.

Do not copy private rows or model-specific aggregates into the public repository.
Add equivalent synthetic public fixtures for empty content, reasoning-only
content, length stops, unextractable answers, and valid final markers.

#### Task 8.2: Fixed-envelope calibration (before v1 only)

Against a representative reasoning endpoint and a non-reasoning endpoint, run
a frozen stratified calibration set. Test candidate fixed envelopes; do not
adapt within a run. Choose the smallest power-of-two envelope up to 4,096 that
eliminates budget-induced length stops on the reasoning calibration set without
changing prompts or reasoning mode.

Decision rule:

- If 2,048 eliminates length stops, keep rc1 values.
- If it does not, create rc2 with 4,096 for that lane.
- If 4,096 still truncates, freeze the limitation transparently or defer v1;
  never add a model-specific exception.

Re-run only the calibration set while tuning. Once v1 is frozen, no cap ladder
may be used for official model comparisons.

#### Task 8.3: End-to-end reference qualification

Run, in order:

1. Mock endpoint full miniature suite.
2. Synthetic fixtures plus opt-in private archive imports when available.
3. One live canary across every lane.
4. One fresh, uninterrupted, pre-bound 8,620-case live reference run.
5. A separate forced-interruption/resume qualification run or miniature fixture;
   never manufacture a crash inside the canonical reference run.
6. A preregistered 125-case reproducibility subset run twice without changing
   settings: 20 BFCL IDs, five IDs from each of 13 base tasks, 20 IFEval IDs, and
   20 HumanEval IDs. Report exact response/verdict agreement and metric variance;
   these repeats are diagnostics and never replace first-run primary scores.
7. Public export and safety scan.

The reference score may be low. Qualification requires structural correctness,
not a target accuracy. Run the long reference job in a durable `tmux` session or
equivalent process supervisor, with atomic progress visible through
`r0b0bench status`. Emit bounded progress at least every 30 minutes and on lane
transitions; never rely on an attached terminal for ownership or resume.

#### Task 8.4: Freeze immutable v1

When all gates pass:

- Rename/copy the accepted RC contract to `r0b0bench-core-v1.yaml`.
- Record canonical suite hash and dependency lock hash.
- Tag locally as `v0.1.0-rc1` first.
- Reinstall from the built wheel in a clean venv and rerun mock plus opt-in
  private-archive gates.
- Push implementation commits only after their exact-SHA release gates pass.
  Package artifacts, tags, and benchmark-score announcements require a separate
  explicit release approval.

---

## 9. Test and quality commands

```bash
cd r0b0bench
uv sync --extra full --extra dev

# Fast contract/unit lane
uv run pytest tests/contract tests/unit -q

# Persistence, mock endpoint, resume, and sandbox security
uv run pytest tests/integration tests/security -q

# Optional private-archive evidence equivalence
uv run pytest tests/regression -q

# Static gates
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src/r0b0bench
uv run python scripts/public_safety_scan.py --repository .
uv run python scripts/public_safety_scan.py --public-export "$PUBLIC_EXPORT_ROOT/RUN_ID"
git diff --check

# Build/reinstall gate
uv build
python3 -m venv /tmp/r0b0bench-wheel-test
/tmp/r0b0bench-wheel-test/bin/pip install dist/r0b0bench-*.whl
/tmp/r0b0bench-wheel-test/bin/r0b0bench --version
```

Network/GPU integration tests must carry explicit markers and never run under
the default unit test command.

---

## 10. Acceptance criteria

r0b0bench v1 is ready only when all conditions hold:

- [ ] Official BFCL is unmodified and pinned by version plus installed
      wheel/full package-tree hash.
- [ ] The suite contract resolves to exactly 8,620 cases.
- [ ] Dataset revisions, counts, fingerprints, and content hashes are frozen.
- [ ] Every prompt, extractor, scorer, request envelope, and sandbox digest is
      included in the canonical suite identity.
- [ ] Request serialization, response normalization, and transport-retry
      semantics are versioned and hash-bound, not inferred from package version.
- [ ] Canonical quality requests use the frozen seed policy; BFCL's upstream
      unseeded policy and the 125-case repeat diagnostics are reported.
- [ ] Legal optional/nested-null and additive provider metadata validate without
      weakening required fields or score semantics.
- [ ] Canaries stop only on infrastructure/provenance faults.
- [ ] A mock model returning empty/truncated output for every item still reaches
      a complete, scoreable full-suite result.
- [ ] Transport retries cannot change valid model outcomes.
- [ ] Existing rows are semantically revalidated before resume.
- [ ] One top-level endpoint lease spans every adapter; volatile owner identity
      remains outside the immutable manifest.
- [ ] Private run roots and raw BFCL roots resolve outside the source checkout;
      in-checkout roots are rejected rather than merely ignored.
- [ ] Historical partial imports remain non-resumable diagnostic reports and
      cannot satisfy live-run validation or release criteria.
- [ ] Any supplied private BFCL archive reproduces its bound official aggregate without exposing model-specific results.
- [ ] Any supplied private canary archive reproduces its bound classification
      counts without exposing model-specific results.
- [ ] Fixed-envelope calibration removes the historical budget-induced failure
      mode or transparently documents a suite-wide remaining ceiling.
- [ ] IFEval official metrics recompute from persisted rows.
- [ ] HumanEval good-code and network-denial preflights pass in the pinned
      sandbox.
- [ ] One fresh, uninterrupted, pre-bound live 8,620-case reference run validates
      with exact counts.
- [ ] Forced interruption resumes without duplicated or foreign rows.
- [ ] Reports expose per-task scores, format diagnostics, and efficiency
      separately.
- [ ] Comparison rejects incompatible suite hashes by default.
- [ ] Runtime-profile identity and public-benchmark contamination risk accompany
      every applicable score; no runtime-neutral or contamination-free claim is
      made.
- [ ] Every dataset has a license/attribution/redistribution/publication manifest.
- [ ] Public export is an allowlisted schema projection containing no secrets,
      private paths/hosts, prompts, model text/reasoning, tool calls, tracebacks,
      container commands, or raw BFCL artifacts.
- [ ] Unit, integration, regression, security, Ruff, formatting, mypy, build,
      clean-wheel, and diff gates pass.
- [ ] No package, release tag, or public benchmark score exists before separate explicit release approval.

---

## 11. Deliverables

1. Local, tested `r0b0bench` Python package and CLI.
2. Immutable `r0b0bench-core-v1` contract and canonical hash.
3. Official BFCL adapter with unmodified score provenance.
4. Generated-answer, IFEval, and sandboxed HumanEval adapters.
5. Atomic, resumable, ownership-bound run engine.
6. Path-safe JSON/Markdown/HTML scorecard.
7. Opt-in hash-bound private-archive regression report plus synthetic public
   fixtures; no private raw rows or model-specific aggregates committed.
8. One fresh complete live reference run, one forced-resume proof, and the
   125-case reproducibility-repeat report.
9. Methodology, datasets, scoring, security, reproduction, and attribution docs.
10. Release-ready implementation, with package/tag/score publication still gated by explicit approval.

---

## 12. Explicitly deferred from v1

- Full official BFCL V4 leaderboard coverage beyond `multi_turn_base`
- Live/search BFCL categories requiring external paid services
- HermesBench real-agent lane
- Throughput/concurrency benchmarking as a scored quality lane
- One-number composite ranking
- Web leaderboard service or hosted submission API
- Adaptive judge models or LLM-as-judge scoring
- Dataset redistribution
- Model-specific prompts, parsers, or reasoning controls

These can be added only as separately versioned profiles after core-v1 is stable.
