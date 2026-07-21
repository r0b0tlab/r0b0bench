# Independent Review History

## 2026-07-20 — Pre-implementation methodology and architecture review

Two independent read-only reviews examined the proposed methodology and the
available predecessor runner patterns.

### Review boundary

The r0b0bench repository and implementation did not yet exist, so the architecture
review provided design guidance rather than source-code approval. No reviewer
approved an implementation, package, benchmark score, or release.

### Findings incorporated

- Real-model canaries are diagnostic and cannot abort a requested full measurement
  for wrong, empty, truncated, or unparsable scoreable output.
- The historical short multiple-choice completion envelope is not the v1 standard;
  v1 uses fixed suite-wide envelopes calibrated before freeze.
- Only user-visible answer content is scored outside BFCL; hidden reasoning remains
  diagnostic and is never substituted as the answer.
- Canonical quality requests use a frozen seed policy; BFCL uses a distinct,
  version-pinned request profile and receives separate reproducibility diagnostics.
- Runtime profile and public-benchmark contamination risk qualify result claims.
- Response validation accepts documented nullable/additive provider metadata while
  keeping required fields strict.
- One parent-owned endpoint lease spans every adapter.
- Immutable logical manifests are separated from mutable process ownership.
- Resume revalidates row semantics, prompts, datasets, scorers, adapters, and
  endpoint identity.
- BFCL receives a private run-scoped root before import.
- Atomic storage uses private ordinal rows and preserves a replaced target if a
  post-replacement durability step fails.
- Public output is an allowlisted projection rather than a recursive redaction of
  arbitrary private evidence.

A new exact-commit review is required after implementation exists and before any
package, tag, score, or release publication.

## 2026-07-21 — Reference-harness method update

A private, run-specific campaign implemented and exercised the 8,620-case
lifecycle. This is operational input to the standalone design, not source-code
approval for r0b0bench and not a public r0b0bench score.

The repository specification now records these reproduced requirements:

- BFCL's 200 cases count toward the 8,620 logical total while retaining a
  separate official lane metric; two 105-case calibrations are excluded.
- The `r0b0bench-core-v1-rc2` candidate uses a fixed 32,768-token ceiling
  across BFCL, generated-answer, IFEval, and HumanEval generation; calibration
  qualifies the envelope but does not guarantee no full-suite length stops.
- Calibration admission uses two independent hash-bound execution scopes, zero
  length stops, and an 80% maximum-usage ceiling.
- The official BFCL adapter needs an exact-stack live canary, deterministic
  allowlisted handling of opaque evaluator state, one-to-one endpoint-call/result
  binding, and score recomputation from validated rows. “Official” applies to the
  pinned dataset/evaluator; the adapter request profile is versioned separately,
  SDK retries are disabled, and one adapter-owned bounded loop records every wire
  attempt.
- Delegated evaluators verify a live ancestor owner and explicitly inherited lock
  descriptors.
- New and resumed evidence roots enforce the same private-parent, owner, mode,
  symlink, hardlink, and Git-checkout boundary.
- A pristine root is bound by preparation before runner logs or evidence are
  created within it.
- Terminal logical-request failures cannot coexist with later success under a
  reset attempt budget.
- Full checkpoint payload hashing belongs at admission; later stages use a cheap
  bound guard and rehash only after identity drift.

The update itself is documentation-only. No package, runner, public raw evidence,
benchmark score, tag, or release is added, and a fresh exact-commit implementation
review remains required after standalone source exists.
