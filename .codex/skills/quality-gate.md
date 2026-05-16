---
name: quality-gate
description: Local quality and release-gate recovery. Activate when `make check` or `make accept` fails, when reproducing CI failures locally, or when figuring out which gate corresponds to which CI workflow. The local gates are designed to mirror CI exactly — anything that passes `CARGO_INCREMENTAL=0 make check` should pass `.github/workflows/ci.yml`.
prerequisites: Rust 1.89.0 (rustup will install from `rust-toolchain.toml`), Python 3.13 on PATH, `cargo-deny`, `cargo-audit`
---

# Local Quality Gate

<purpose>
This project has a layered gate: `make check` (every commit), `make accept` (release readiness), and CI workflows that re-run subsets in matrix mode. This skill maps failures to fixes.
</purpose>

<context>
**`make check`** (≈3-5 min cold; ~30s warm) consists of, in order:

1. `cargo fmt --all` — formatting, denied if any file changes.
2. `cargo clippy --workspace --all-targets -- -D warnings` — lints with warnings denied.
3. `make py-lint` — Ruff via `python/pyproject.toml`; falls back to `py_compile` when Ruff missing.
4. `cargo check --workspace --all-targets` — type/borrow-check.
5. `python3 scripts/check_layers.py` — inter-crate dep allowlist (see `crate-layering.md`).
6. `python3 scripts/check_specs.py` — frontmatter + links + traceability (see `rfc-adr-process.md`).
7. `python3 scripts/check_jobs.py` — `jobs/*.yaml` validation (see `hf-jobs-cost.md`).
8. `python3 scripts/check_otel_infra.py` — Otel infra-as-code shape.
9. `python3 scripts/check_train_so100_job.py` — SO-100 full-training job contract.
10. `python3 scripts/check_nondet.py` — banned RNG / nondet patterns (see `determinism-rng.md`).
11. `python3 -m py_compile python/…` — fast Python syntax sweep for ungated files.
12. `python3 python/cost_ledger.py check --path reports/cost.md --cap-usd 200` — total spend ≤ $200.
13. `cargo deny check` — license + ban list + advisory DB.
14. `cargo audit … --ignore RUSTSEC-2024-0436 --ignore RUSTSEC-2026-0009 --ignore RUSTSEC-2025-0141` — CVE scan with scoped waivers per ADR.

**`make accept`** = `make check` + `make test` + `make docs` + `make -C python check` + `scripts/check_hub_artifacts.py` + `scripts/check_release_inventory.sh`.

**CI workflows** (`.github/workflows/`):
- `ci.yml` — fmt, clippy matrix (default / cpu-only / parity-fixtures × Linux/macOS), build matrix, tests, parity (with HF dump cache).
- `specs.yml` — `check_specs.py` + Lychee external-link checker on PRs that touch `PRD.md` / `specs/`.
- `conformance.yml` — weekly `make accept` against `main`.
- `docs.yml` + `gh-pages.yml` — docsite build/publish.
- `nightly.yml` — extended/`_slow_` test runs.
- `release.yml` — release pipeline.

Canonical local invocation that matches CI most closely:
```
CARGO_INCREMENTAL=0 make check
```
</context>

<procedure>
**On any `make check` failure:**

1. Read the FIRST failing step's full output. Many later steps cascade from one earlier failure; fixing step N often clears N+1.
2. Locate which gate failed (steps 1–14 above). Apply the matching recovery:

| Failing step          | Recovery                                                                                                     |
|-----------------------|--------------------------------------------------------------------------------------------------------------|
| 1 fmt                  | `make fmt`, then re-run.                                                                                     |
| 2 clippy               | Fix the diagnostic. Do **not** `#[allow(unwrap_used / unsafe_code / unimplemented / dbg_macro / print_stdout)]`. |
| 3 py-lint              | `ruff check --fix --config python/pyproject.toml python scripts`. See `python-helpers.md`.                   |
| 4 cargo check          | Build error: read the rustc message; if backend-feature-related, try `--no-default-features --features cpu-only` |
| 5 check_layers         | See `crate-layering.md`. Move the symbol, do not relax the allowlist.                                        |
| 6 check_specs          | See `rfc-adr-process.md`. Likely frontmatter, broken link, or missing traceability row.                      |
| 7 check_jobs           | See `hf-jobs-cost.md`. Likely missing `--timeout`, banned hardware, or missing env key.                      |
| 8 check_otel_infra     | Otel-stack shape drift in `infra/otel/`; re-align with the contract.                                         |
| 9 check_train_so100    | SO-100 full-training YAML contract drift; align with leash + RFC 0012.                                       |
| 10 check_nondet        | See `determinism-rng.md`. Replace `thread_rng` / banned pattern.                                             |
| 11 py_compile          | Syntax error in a Python helper. Fix and re-run.                                                             |
| 12 cost ledger          | Total > $200. STOP. Do not bypass.                                                                            |
| 13 cargo deny          | New dep license / ban: see `<adding_dependency>` workflow in `CLAUDE.md`.                                    |
| 14 cargo audit         | New RUSTSEC: upgrade if possible; else ADR + scoped `--ignore` in `Makefile`. Do not bypass silently.        |

3. After a fix, re-run **only** the failing step first (faster feedback), then the full `make check` once.

4. If `make check` passes but CI fails: ensure you ran with `CARGO_INCREMENTAL=0`. Also confirm `rustup show active-toolchain` resolves to 1.89.0.

**On `make test` failure:**

- Re-run focused: `cargo test -p <crate> <test_name> -- --nocapture`.
- If `parity_*` fails, see `parity-testing.md`.
- If `resume_*` fails, see `determinism-rng.md`.
- If macOS-only failure on `cuda` feature, build with `--no-default-features --features cpu-only`.

**On `make accept` failure:**

- Usually `make check` or `make test`. Recur into those.
- If `check_hub_artifacts.py` fails, see `release-and-artifacts.md`.
- If `check_release_inventory.sh` fails, the release manifest is out of sync — update `reports/release_checklist.md`.

**On CI-only failure:**

- Look at the workflow log for the exact step name. Reproduce locally with the same feature args:
  - `--no-default-features --features cpu-only` (Linux + macOS matrix)
  - `--features parity-fixtures` (parity job; needs `HF_TOKEN` for dumps)
- The `parity` job will gracefully fall back to shape-only on forks without `HF_TOKEN`.
</procedure>

<patterns>
<do>
— Run `CARGO_INCREMENTAL=0 make check` before every commit. It's faster than chasing CI failures.
— Prime `target/advisory-db/` once per fresh container with `cargo audit fetch`. `make check`'s audit step expects it.
— Use `make test-fast` during inner-loop development; reserve full `make test` for pre-commit.
— Keep the `Makefile` as the source of truth for command shapes — if you reproduce manually, copy the exact flags from there.
</do>
<dont>
— Don't add `#[allow(clippy::unwrap_used)]` or similar to "make clippy pass." The lints are policy.
— Don't add new RUSTSEC ignores to the `Makefile` without an ADR. Each existing `--ignore` has a date-bounded waiver.
— Don't disable `make check` in CI to ship a fix. Reproduce locally, fix the cause.
— Don't run `make check` without `CARGO_INCREMENTAL=0` when comparing to CI — stale incremental artifacts can hide / reveal warnings inconsistently.
</dont>
</patterns>

<examples>
Typical fix loop for a clippy regression:
```
$ CARGO_INCREMENTAL=0 make check
error: used `.unwrap()` on a `Result` value
  --> crates/lewm-train/src/optim.rs:42:18

# Fix: replace `.unwrap()` with `?` (propagate) or `.expect("…")` (only allowed in tests)

$ cargo clippy -p lewm-train --all-targets -- -D warnings   # quick re-check
$ CARGO_INCREMENTAL=0 make check                            # full re-run
```
</examples>

<troubleshooting>
| Symptom                                                       | Cause                                                  | Fix                                                                       |
|---------------------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------------------------|
| `make check` step 14 (audit) fails offline                    | No advisory DB                                         | `cargo audit fetch` to prime `target/advisory-db/`                        |
| `cargo deny check` slow                                       | License classifier walking dep tree                    | Run `cargo deny check licenses` first to narrow                            |
| CI fails clippy on `parity-fixtures` only                     | Feature-gated code with stricter lints                 | Build locally with `--features parity-fixtures` to repro                   |
| macOS CI fails build but Linux passes                          | Default `cuda` feature implicitly enabled              | Use `--no-default-features --features cpu-only` on macOS                  |
| `make accept` works locally but `conformance.yml` fails        | Hub artifact / release-inventory drift                  | See `release-and-artifacts.md`; reconcile `reports/release_checklist.md`  |
</troubleshooting>

<references>
- `Makefile` — source of truth for gate commands
- `.github/workflows/ci.yml`, `specs.yml`, `conformance.yml` — CI mirror
- `clippy.toml`, `rustfmt.toml`, `deny.toml` — lint policy files
- `scripts/check_*.py` — individual validators
- `reports/release_checklist.md` — release-readiness manifest
