# Local quality gate

> Run this before sending a PR.

```sh
CARGO_INCREMENTAL=0 make check
```

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
- `python3 python/cost_ledger.py check --path reports/cost.md --cap-usd
  200` — cost-cap check.
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
- Future hooks for hub-artifact verification and release inventory.

This is the gate the maintainer runs before tagging a release. See
[`reports/release_checklist.md`](https://github.com/AbdelStark/lewm-rs/blob/main/reports/release_checklist.md).

## Targeted gates

- Training-only: `cargo test -p lewm-train --all-features --locked`.
- Parity-only (with dumps available):
```text
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
