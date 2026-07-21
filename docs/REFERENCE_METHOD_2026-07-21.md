# Current Reference Method and Porting Boundary

Updated: 2026-07-21

This document records the method currently being exercised by the private
reference campaign that informs r0b0bench implementation. It is implementation
evidence, not a released r0b0bench package, public benchmark result, or approval
to publish private rows.

## Standard-suite accounting

The standard suite contains **8,620 logical cases**:

- Official BFCL v4 `multi_turn_base`: 200
- Generated-answer tasks: 7,715
- IFEval: 541
- HumanEval: 164

BFCL cases count toward the standard-suite total while retaining a separate
official BFCL lane metric. The project does not average BFCL, generated-answer,
IFEval, and HumanEval values into an opaque combined accuracy.

Two pre-admission calibration repeats contain 105 cases each. Calibration evidence is
excluded from the 8,620 standard-suite count and cannot be relabeled or reused as
full-run evidence.

## Fixed reference request contract

The current reference campaign uses a fixed 32,768-token completion-envelope
candidate. This is `r0b0bench-core-v1-rc2` candidate input, not a released suite
identity and not a guarantee that no full-suite model response can reach the
ceiling.

| Lane | Temperature | top_p | Seed | Stop | Maximum completion tokens | Concurrency |
|---|---:|---:|---:|---:|---:|---:|
| BFCL adapter profile | 0.001 | omitted | omitted | omitted | 32,768 | 1 thread |
| Generated answer | 0 | 1 | 0 | none | 32,768 | 16 |
| IFEval | 0 | 1 | 0 | none | 32,768 | 16 |
| HumanEval generation | 0 | 1 | 0 | none | 32,768 | 16 |

There are no model-specific prompt, parser, token-budget, or reasoning-mode
exceptions. Successful usage must contain an exact nonnegative integer
`completion_tokens` no greater than 32,768. The outgoing BFCL request, BFCL
endpoint ledger, and final BFCL result validator enforce the same ceiling.

The non-BFCL transport policy uses a 10-second connect timeout, a 3,600-second
response timeout, at most three attempts, deterministic 2-second then 10-second
backoff, and retries only connection/timeout failures or HTTP 408, 429, 500, 502,
503, and 504. A valid but wrong, empty, truncated, unparsable, or
tool-decode-failed model response is scored and is never retried as
infrastructure.

The BFCL result remains official with respect to the pinned dataset and
unmodified evaluator. Its request profile is separately identified: the pinned
OpenAI handler sends messages, model, temperature, `store=false`, optional tools,
and the adapter's explicit `max_tokens=32768`; it omits top_p, seed, and stop.
This is not described as byte-identical leaderboard request behavior. The
run-specific reference handler inherits a `RateLimitError` retry decorator and
the OpenAI SDK's default retry behavior, which can hide wire attempts from an
outer ledger. `r0b0bench-core-v1-rc2` therefore requires `max_retries=0` on the
SDK client and one adapter-owned retry loop around the actual SDK `create` call:
three total wire attempts; deterministic 2-second then 10-second backoff; no
jitter; retry only connection reset/timeout and HTTP 408, 429, 500, 502, 503,
and 504; never retry a valid 2xx or any other 4xx. Exhaustion persists the final
attempt and a terminal infrastructure failure. Every attempted call, including
the exhausted terminal call, receives its own contiguous endpoint-ledger record.
The upstream decorator must be bypassed or replaced—not wrapped as a second retry
layer.

## Calibration admission

Before the 8,420 non-BFCL rows are admitted, two independent 105-case repeats
must each use fresh requests for the same frozen calibration identities:

- generated answer: zero-based indices 0-4 from each of the 13 tasks, in suite
  task order (65 cases);
- IFEval: zero-based indices 0-19 (20 cases);
- HumanEval: zero-based indices 0-19 (20 cases).

The identity projection uses UTF-8 JSON with sorted object keys, compact
separators, `ensure_ascii=false`, and list order preserved. Every displayed
digest is lowercase SHA-256 of those exact bytes. Prompt and task-ID digests are
SHA-256 of the exact UTF-8 source string; `instruction_ids_sha256` is SHA-256 of
the raw instruction-ID list encoded with the same canonical JSON algorithm.
Generated entries contain `task`, `index`, and `prompt_sha256`; IFEval entries
contain `index`, `prompt_sha256`, and `instruction_ids_sha256`; HumanEval entries
contain `index`, `task_id_sha256`, and `prompt_sha256`. A lane digest hashes that
lane's ordered entry list. The combined digest hashes exactly
`{"generated_answer": <list>, "ifeval": <list>, "humaneval": <list>}` under the
same canonical algorithm. The complete hash preimage is published as
[`CALIBRATION_IDENTITY_2026-07-21.json`](CALIBRATION_IDENTITY_2026-07-21.json)
(file SHA-256
`4e7bb6c2dcd8c2bae7fe93b94ec1defb9c366cf289464cac4a7c195964f23e28`).
The current pinned-data identity digests are:

- generated answer: `e11304fd3f2a3fdfd4b9df9b8a6c8f7d1de83d58192a9750c68c6f7bc6645163`;
- IFEval: `f7fe54bf2e049ed829d4f1821313b5482c48438f247b87ae46f7a2d30b90be98`;
- HumanEval: `436ed06adb79ee7fff8f4627b8326a3eab305dff248da3361b01c3ebdae2ff0f`;
- combined lane map: `47bcc014b84ce0ad0b4d86936407c6895a8ae31c75640b14522a07300ada7d7e`.

The same identities are requested twice under distinct immutable execution
scopes `calibration-repeat-01` and `calibration-repeat-02`. Neither repeat's
rows may satisfy or be reused by `full` scope. Each repeat must:

- use distinct hash-bound execution scopes;
- contain the exact generated-answer, IFEval, and HumanEval case sets;
- contain zero `finish_reason=length` results;
- keep maximum completion usage at or below 80% of the fixed ceiling;
- recompute every row, scorer output, sandbox verdict, and evidence index;
- reject copied/relabelled rows, gaps, foreign files, and stale failure ledgers.

This is an admission check for the fixed suite contract, not a score threshold.

## Exact-stack BFCL lifecycle

A no-request fixture is insufficient for the official BFCL adapter because
multi-turn evaluators can create opaque simulator objects. The reusable adapter
must therefore prove the following lifecycle before a long run:

1. Run an isolated three-case exact-stack canary using the real pinned BFCL
   package, handler, evaluator, multi-turn state, owner locks, and endpoint.
2. Canonicalize mappings, sequences, primitives, Pydantic values, bytes, and only
   an explicit allowlist of evaluator-owned opaque types. Unknown types fail
   closed; arbitrary global `str()`/`repr()` fallback is prohibited.
3. Bind every endpoint attempt to the canonical input ID/hash, request hash,
   endpoint epoch, response identity, token usage, and final result row.
4. Require contiguous immutable endpoint ordinals, refuse overwrites, and reject
   duplicate, foreign, missing, or orphan calls.
5. Re-run the official evaluator from validated result rows before accepting the
   score artifact.
6. Keep canary and full output roots distinct so canary rows cannot satisfy the
   200-case count.

Official decode and tool-format failures remain scoreable model outcomes. HTTP,
endpoint-schema, package, scorer, ownership, persistence, or evaluator lifecycle
failures stop the run as infrastructure or provenance failures.

## Evidence and ownership requirements

The current reference implementation established the following porting
requirements:

- A new run root must be absolute, owner-only, outside the source checkout, and
  beneath the configured private run parent.
- Existing/resumed roots must satisfy the same location, ownership, mode,
  symlink, hardlink, and Git-checkout restrictions.
- The controller must leave the new root pristine until manifest preparation
  owns it; controller logs live outside the unbound root.
- The owner token binds a live ancestor PID and process-start epoch. Delegated
  evaluator subprocesses inherit and verify the exact owner/workload lock file
  descriptors.
- Every request persists a full recomputable endpoint epoch, including transport,
  HTTP, malformed-success, and success paths.
- A terminal logical-request failure invalidates that run ordinal and cannot
  coexist with a later success under a reset attempt budget.
- Evidence directories are exact enumerations, not permissive glob matches.
- Files are owner-only (`0600`, or stricter read-only after freeze) and
  directories are `0700`.
- HumanEval verdicts remain bound to a digest-pinned, networkless, read-only,
  capability-dropped sandbox invocation and its supporting response evidence.
- Full checkpoint payload hashing is an admission operation. Later stage checks
  should use a cheap bound mount/inode/size/time/container/process guard and
  repeat the expensive hash only when that guard changes.

## Reporting contract

A complete public scorecard reports:

- `standard_suite_logical_items = 8620`;
- component counts for BFCL, generated answer, IFEval, and HumanEval;
- separate lane metrics and diagnostics;
- calibration counts separately and explicitly excluded;
- suite, manifest, runtime-profile, scorer, dataset, and evidence-root digests;
- no prompts, responses, reasoning, tool calls, candidate code, private paths,
  hostnames, credentials, traces, or raw official-evaluator artifacts.

`COMPLETE` means every expected case has validated evidence. It does not imply a
release threshold or a good model score.

## Repository status

The public repository remains specification-only until the run-specific
reference behavior is ported into model-neutral modules, contract tests, and a
clean-installable package. The private campaign must not be copied verbatim:
run-bound paths, runtime identity, model identity, and deployment-specific
checkpoint information are configuration/evidence, not reusable package source.

The next implementation candidate must treat the 32,768-token envelope and the
requirements above as a new suite-contract revision. It must pass contract,
unit, integration, security, exact-stack canary, clean-wheel, public-safety, and
exact-commit review gates before the repository is described as runnable.
