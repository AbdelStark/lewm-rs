# Contributing

The full contributor's guide is in
[`CONTRIBUTING.md`](https://github.com/AbdelStark/lewm-rs/blob/main/CONTRIBUTING.md).
Highlights below.

## Where work happens

- **Code**: PRs to `main` on GitHub.
- **Design**: RFC PRs in `specs/rfcs/`. Start from
  `specs/rfcs/0000-template.md`.
- **Decisions**: ADR PRs in `specs/adr/`. Start from
  `specs/adr/0000-template.md`.
- **Docs**: PRs to `docs/src/` for this site.
- **Issues**: Bug reports and feature requests on GitHub.

## Local gate

Run `make check` before sending a PR. The CI mirrors it. See
[Local quality gate](../reproducing/quality-gate.md).

For a fast local pre-flight, set up `pre-commit` once (see the same
page); it runs gitleaks, ruff, `cargo fmt --check`, and the cheap
project validators on every staged change.

## Release process

Releases are cut by the maintainer following the
[`RELEASE.md`](https://github.com/AbdelStark/lewm-rs/blob/main/RELEASE.md)
runbook. The pipeline is fully automated once a `vX.Y.Z` tag is pushed:
reproducible binary builds (linux musl + macOS arm), the GHCR
container image (cosign-signed, build-provenance-attested), the
CycloneDX SBOM, the export verifier smoke, and the draft GitHub
release. The maintainer's only remaining task is the **draft →
publish** promotion after reviewing the release notes.

## RFC / ADR rhythm

- Spec changes precede code changes when the change is non-trivial.
- The reviewer's first question on any non-trivial PR is: "Is the
  contract in an RFC?"
- For bug fixes and small refactors, no RFC required.

## Coding conventions

- `cargo fmt` is required; CI enforces.
- `clippy -- -D warnings` is required; CI enforces.
- Public functions in `crates/lewm-core` carry full docstrings as
  specified in [RFC 0015 §3](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0015-documentation-paper-and-demo.md).
- Cross-link RFC IDs in source (`// See RFC 0003 §4.2.1`) where the
  contract is non-obvious.

## Parity hygiene

Any change that touches `lewm-core`'s model code **must** preserve
the parity tests. If a refactor requires updating the dumps, the PR
must also update the dump-producer script and explain *why* the dumps
needed to change.

## Testing

- Unit tests next to the source (`#[cfg(test)] mod tests` or
  `src/lib.rs`).
- Integration tests under `crates/<crate>/tests/`.
- Parity tests under `crates/lewm-core/tests/parity_*.rs`, gated
  behind `parity-fixtures` feature.
- Resume-parity test under `crates/lewm-train/tests/resume_parity.rs`.

## Commit messages

Conventional Commits encouraged but not enforced. Reference the
relevant RFC or ADR in the message body when the commit implements a
specific clause:

```text
feat(core): add AdaLN-zero modulation head (RFC 0002 §4.7.2)

Implements ConditionalBlock::ada_ln_modulation with zero-init,
verified against parity_predictor.

Closes #42.
```
