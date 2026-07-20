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
- Canonical quality requests use a frozen seed policy; BFCL retains its official
  upstream request behavior and receives separate reproducibility diagnostics.
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
