# Contributing

r0b0bench is currently in specification stage. Contributions should preserve the
methodology and evidence boundaries in `docs/IMPLEMENTATION_PLAN.md`.

## Before proposing implementation

- Open or reference a focused issue describing the contract being implemented.
- Keep BFCL and other official scorers unmodified.
- Add collecting RED contract tests before production behavior.
- Do not commit benchmark datasets, raw prompts, model responses, credentials,
  hostnames, private paths, traces, or private archive aggregates.
- Treat changes to prompts, token envelopes, extractors, scorers, retry semantics,
  request/response normalization, or sandbox policy as suite-version changes.

## Verification expectations

Implementation changes will eventually require unit, contract, integration,
security, packaging, clean-wheel, public-safety, and exact-commit review gates.
Until those gates exist, do not describe the repository as runnable or publish
results under the r0b0bench name.

By contributing, you agree that your original contributions are licensed under
the repository's MIT License. Third-party material must retain its original
license and attribution.
