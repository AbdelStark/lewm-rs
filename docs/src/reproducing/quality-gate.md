# Local quality gate

> Run this before sending a PR.

```sh
CARGO_INCREMENTAL=0 make check
```

## Pre-commit hooks (optional, but recommended)

For a fast local checkpoint before `make check`, install the
[pre-commit](https://pre-commit.com/) framework once and let it run the
short, deterministic checks on every staged change:

```sh
pipx install pre-commit         # or: pip install pre-commit
pre-commit install              # writes .git/hooks/pre-commit
pre-commit run --all-files      # one-off sweep
```

The hook set is defined in `.pre-commit-config.yaml` at the repo root:

- `gitleaks protect --staged` — secret-scan the staged diff.
- `ruff check` + `ruff format --check` — Python lint on `python/` and
  `scripts/` against `python/pyproject.toml`.
- `cargo fmt --all -- --check` — Rust formatting.
- `scripts/check_layers.py`, `scripts/check_specs.py`,
  `scripts/check_jobs.py`, `scripts/check_nondet.py` — the cheap project
  validators.

The hooks are a strictly weaker subset of `make check`; they exist to
catch the high-confidence issues (secrets, formatting) before the heavier
clippy / cargo check / cargo deny passes run.

`make check` is the union of:

- `make fmt` — `cargo fmt --all`. Whitespace and import-order normalisation.
- `make lint` — `cargo clippy --workspace --all-targets -- -D warnings`.
- `make py-lint` — Ruff on `python/` + `scripts/` (config in
  `python/pyproject.toml`); falls back to `py_compile` if Ruff is
  absent.
- `cargo check --workspace --all-targets` — full compile-check.
- `python3 scripts/check_layers.py` — layer-dependency map verification
  (INV-003).
- `python3 scripts/check_specs.py` — RFC cross-reference / glossary
  enforcement.
- `python3 scripts/check_jobs.py` — HF Jobs spec validation.
- `python3 scripts/check_otel_infra.py` — OTLP infrastructure config
  validation.
- `python3 scripts/check_nondet.py` — non-determinism scan.
- `python3 -m py_compile python/*.py scripts/*.py` — syntax sweep on
  the Python helpers.
- `python3 python/cost_ledger.py check --path reports/cost.md --cap-usd 200`
  — cost-cap check.
- `cargo deny check` — license / advisory / source policy.
- `cargo audit ...` — vulnerability scan (with project-specific
  ignores for unmaintained transitive deps; see `Makefile`).

## The longer gate: `make accept`

For release prep:

```sh
make accept
```

`make accept` runs:

- `make check` (above).
- `make test` (`cargo test --workspace --all-features`).
- `make docs` (`cargo doc --workspace --no-deps` with
  `RUSTDOCFLAGS=-D warnings`).
- `python/Makefile` Python gate.
- `scripts/check_release_blockers.py` without `--allow-open`; this fails while
  any release blocker in `conformance/release_blockers.json` is still open.
- `scripts/check_phase_a_handoff.py`; this keeps the F1/F3 operator handoff
  aligned with the release blockers and prevents dry-run / execute / upload
  commands from drifting into the wrong stage.
- Future hooks for hub-artifact verification and release inventory.

This is the gate the maintainer runs before tagging a release. See
[`reports/release_checklist.md`](https://github.com/AbdelStark/lewm-rs/blob/main/reports/release_checklist.md).

## Targeted gates

- Training-only: `cargo test -p lewm-train --all-features --locked`.
- Parity-only (with dumps available):

  ```sh
  LEWM_REFERENCE_SAFETENSORS=... LEWM_PARITY_DUMPS=... \
      cargo test -p lewm-core --features parity-fixtures --locked
  ```

- Docs build: `make docs` or directly
  `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked`.

## CI parity

The CI workflows in `.github/workflows/` mirror these gates:

| Workflow | Gate equivalent |
|----------|-----------------|
| `ci.yml`         | `make check` + `make test-fast` |
| `specs.yml`      | `scripts/check_specs.py`, `scripts/check_layers.py` |
| `conformance.yml`| Conformance suite under `conformance/` |
| `docs.yml`       | `make docs` + paper / mdbook builds |
| `nightly.yml`    | Full `make accept` overnight |
| `release.yml`    | Release-tag artifact build |

A green `ci.yml` is *necessary* but not *sufficient* for release.
The nightly + release workflows have stricter gates.
