<identity>
lewm-rs — pure-Rust workspace reproducing LeWorldModel (Maes et al., 2026): training, planning, CPU/GPU inference, and Hugging Face Hub publication. Spec-first, parity-verified, single-author repository governed by `PRD.md` and `specs/` (RFCs 0001–0018).
</identity>

<stack>
| Layer            | Technology              | Version   | Notes                                              |
|------------------|-------------------------|-----------|----------------------------------------------------|
| Toolchain        | Rust                    | 1.89.0    | Pinned in `rust-toolchain.toml`; edition 2024     |
| Build            | Cargo workspace         | resolver 2| 8 crates under `crates/`                           |
| ML framework     | Burn                    | =0.20.1   | Pinned exact; `burn-cuda`, `burn-ndarray` backends |
| CPU inference    | Tract                   | =0.22.1   | Pinned exact; ONNX (opset 17 fixed-batch)          |
| Python edge      | Python                  | 3.13      | `uv`-managed; Ruff lint; under `python/`           |
| Compute          | HF Jobs                 | —         | Hard cap **$200**; soft cap $100; per-session $20 |
| Hub              | huggingface.co          | —         | `abdelstark/lewm-rs-pusht`, `…-so100`, `…-demo`    |
| Docs             | mdBook                  | —         | Source `docs/src/`; build `make docsite`           |
| Lint policy      | `cargo clippy -D warn`  | —         | `unsafe_code=deny`, `unwrap_used=deny`             |
</stack>

<structure>
```
crates/                  # 8-crate Cargo workspace. Dep layering enforced.
├── lewm-core/           # ViT, predictor, AdaLN, SIGReg, JEPA, init, import/export. NO deps on other lewm crates.
├── lewm-data/           # PushT HDF5 + LeRobot v2.1; deps: lewm-core
├── lewm-hub/            # HF Hub helpers; deps: lewm-core
├── lewm-telemetry/      # OTel + nvml; deps: lewm-core
├── lewm-plan/           # CEM planner + eval; deps: lewm-core, lewm-data, lewm-telemetry
├── lewm-train/          # Trainer + lewm-train/lewm-reference-record bins; deps: lewm-core, lewm-data, lewm-telemetry, lewm-hub, lewm-plan
├── lewm-infer/          # Tract + burn-cpu inference; lewm-infer bin; deps: lewm-core, lewm-telemetry. NO CUDA/autodiff.
├── lewm-gpu/            # burn-cuda glue; deps: lewm-core, lewm-infer. Only crate that may import burn-cuda. Per RFC 0007.
specs/                   # SOURCE OF TRUTH. RFCs are binding once Accepted. [READ ONLY without ADR]
├── rfcs/                # 0001–0018, all Accepted v1.x
├── adr/                 # Numbered ADRs; supersede RFC clauses
├── TECHNICAL_SPECIFICATION.md, glossary.md, traceability-matrix.md
PRD.md                   # Product contract. [READ ONLY]
configs/                 # pusht.toml, so100.toml, …_eval/_warmstart. [READ ONLY without RFC update]
jobs/                    # HF Jobs YAML. train_*.yaml gated by .ml-intern config.
scripts/                 # Local validation scripts: check_specs/layers/jobs/nondet/otel/…
docs/                    # mdBook site (Concepts → Architecture → Training → Results → Reference)
python/                  # Edge helpers (export_onnx, convert_reference, upload_*, cost_ledger…). Ruff-linted.
infra/otel/              # Optional self-hosted OTel stack
paper/                   # Paper-style writeup (CC-BY-4.0)
reports/                 # Training/inference reports + cost ledger
tests/                   # Cross-crate spec/conformance harnesses (per-crate tests live in crates/*/tests)
conformance/             # Conformance gate stubs
.ml-intern/cli_agent_config.json   # AGENT SAFETY LEASH. Read before acting. See <boundaries>.
.codex/skills/           # Modular agent skills. Symlinked at .claude/skills and .agents/skills.
```
</structure>

<commands>
| Task                        | Command                                                                          | Notes                                                       |
|-----------------------------|----------------------------------------------------------------------------------|-------------------------------------------------------------|
| Format                      | `make fmt` or `cargo fmt --all`                                                  | rustfmt config in `rustfmt.toml`                            |
| Lint (Rust)                 | `make lint` (= `cargo clippy --workspace --all-targets -- -D warnings`)          | `-D warnings` is non-negotiable                             |
| Lint (Python)               | `make py-lint`                                                                   | Ruff via `python/pyproject.toml`; degrades to `py_compile`  |
| Cargo check                 | `cargo check --workspace --all-targets`                                          | Used inside `make check`                                    |
| Unit + integration tests    | `make test` (= `cargo test --workspace --all-features`)                          | Includes `_slow_` tests                                     |
| Fast tests                  | `make test-fast`                                                                 | Skips `_slow_`; lib/bin only                                |
| Crate-focused (training)    | `cargo test -p lewm-train --all-features --locked`                               | Use when changing training                                  |
| Benchmarks                  | `make bench`                                                                     | Criterion under `crates/*/benches`                          |
| Local quality gate          | `CARGO_INCREMENTAL=0 make check`                                                 | Fmt, clippy, py-lint, check, 6 Python validators, deny, audit. RUN BEFORE COMMIT. |
| Release gate                | `make accept`                                                                    | `check` + tests + docs + python/Makefile + hub-artifact + release-inventory      |
| Rustdoc                     | `make docs` (= `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`)     | Doc warnings denied                                         |
| Docsite (mdBook)            | `make docsite`                                                                   | Requires `cargo install mdbook`                             |
| Spec validation             | `python3 scripts/check_specs.py`                                                 | Frontmatter + links + traceability                          |
| Crate-layer invariants      | `python3 scripts/check_layers.py`                                                | Enforces inter-crate dep allowlist                          |
| Jobs validation             | `python3 scripts/check_jobs.py`                                                  | Validates `jobs/*.yaml` shapes                              |
| Nondet lint                 | `python3 scripts/check_nondet.py`                                                | Bans `thread_rng` etc. per RFC 0013                         |
| Smoke train (PushT, CPU)    | `cargo run -p lewm-train -- --config configs/pusht.toml --device cpu --output-dir /tmp/lewm-smoke smoke --steps 50 --batch-size 4` | Local validation; no HF spend       |
| Bounded train               | `cargo run -p lewm-train -- --config configs/pusht.toml --device cpu --output-dir /tmp/out --max-steps 10 train`                  | Real data-plane path                |
| Launch HF job (allowed)     | `scripts/launch_hf_job.py jobs/smoke_pusht.yaml`                                 | ⚠ COSTS MONEY. Allowed jobs only — see `<boundaries>`.      |
| Launch HF job (gated)       | `scripts/launch_hf_job.py jobs/train_pusht.yaml`                                 | ⚠ REQUIRES HUMAN APPROVAL                                   |
| Profile (CPU)               | `scripts/run_local.sh flamegraph <name> -- <cargo-flamegraph args…>`             | Outputs to `profiling/flamegraphs/<git_sha>/`               |
| Profile (GPU)               | `scripts/run_local.sh nsys <name> -- <cmd…>`                                     | Needs `lewm-telemetry/nvtx` feature                         |
</commands>

<conventions>
<code_style>
- Naming: snake_case (functions, modules, files), CamelCase (types, traits), SCREAMING_SNAKE (consts). Crate names use kebab-case (`lewm-core`), library identifiers use snake (`lewm_core`).
- Imports: rustfmt enforces `imports_granularity = "Crate"`, `group_imports = "StdExternalCrate"`, `imports_layout = "HorizontalVertical"`. Do not hand-format imports.
- Width: 100 cols (`max_width = 100`). Trailing comma vertical.
- Edition: 2024. `rust-version = "1.89"` is pinned via the workspace.
- Errors: each crate exposes a typed error (e.g. `LewmCoreError`) via `thiserror`. Bins may use `anyhow`. Never silently `?`-bubble into `anyhow` from library code.
- Logging: `tracing` macros; structured fields. Avoid `println!`/`eprintln!` in libs (clippy `print_stdout = warn`).
- Determinism: no `std::collections::HashMap` for iteration order in observed paths — see `skills/determinism-rng.md`. No `thread_rng` (lint-banned).
- Docs: every public item carries rustdoc per RFC 0015: shape, errors, invariants, examples. `missing_docs = "warn"`.
</code_style>

<patterns>
<do>
— Match the existing crate's error type and re-export pattern; do not introduce new error crates.
— Add new dependencies to `[workspace.dependencies]` first, then reference with `{ workspace = true }` in the crate.
— For Burn modules, derive `Module` + `Debug`; place tensor-shape contracts in module-level rustdoc with `RFC 0002 §X` references.
— RNG: take a `&mut ChaCha20Rng` (or sub-stream key) per RFC 0013; never seed inside library functions.
— Numerical changes that touch encoder/predictor/SIGReg/pred_proj: run the parity tests — see `skills/parity-testing.md`.
— Configs: add new fields with `serde(default)` and a `validate` impl; update `configs/*.toml` and the appropriate RFC.
— Conventional commits: `feat(scope): …`, `fix(scope): …`, `docs(scope): …`, `ci(scope): …`, `chore(scope): …`. Sign off with `-s` (DCO required).
</do>
<dont>
— Don't add a `lewm-*` crate dep without updating `scripts/check_layers.py`'s allowlist and `specs/rfcs/0001-…`.
— Don't introduce `burn-cuda`, `burn-autodiff`, or `nvml-wrapper` into `lewm-infer` — it must build on CUDA-less hosts (RFC 0007). Put CUDA wiring in `lewm-gpu`.
— Don't `.unwrap()` or `.expect()` in non-test code paths (clippy denies `unwrap_used`, warns `expect_used`).
— Don't introduce `unsafe` blocks (`unsafe_code = "deny"` workspace-wide).
— Don't paste TODO/FIXME into committed code without an issue link; clippy warns `todo` and `unimplemented` is denied.
— Don't bypass the lints with `#[allow(...)]` for `unwrap_used`, `unsafe_code`, `unimplemented`, `dbg_macro`, or `print_stdout` without a comment naming the RFC clause that permits it.
— Don't commit anything with `dbg!` (denied) or stray `println!` (warned in libs).
</dont>
</patterns>

<commit_conventions>
- Conventional Commits, scoped to a crate or topic: `feat(lewm-core): …`, `fix(infer): …`, `docs(rfc-0008): …`.
- `Signed-off-by:` trailer required (DCO). Use `git commit -s`.
- Keep changes small and tied to one issue; reference with `Closes #<n>` only when acceptance is met.
- Never include tool-branding lines (e.g., "generated by …") in commit messages or PR bodies.
- PR template at `.github/PULL_REQUEST_TEMPLATE.md` requires: Problem, Solution, Traceability, Validation, Caveats.
- Update `specs/traceability-matrix.md` when changing user-visible / contract behavior.
- Update `CHANGELOG.md` under `## [Unreleased]` for any user-facing change.
</commit_conventions>
</conventions>

<workflows>
<bug_fix>
1. Identify the failing RFC/test ID (search `specs/traceability-matrix.md` if applicable).
2. Reproduce locally with the narrowest test: `cargo test -p <crate> <test_name> -- --nocapture`.
3. Implement fix in the owning crate; preserve module boundaries.
4. Run `CARGO_INCREMENTAL=0 make check` — it MUST pass before commit.
5. Re-run the focused test, then `cargo test -p <crate> --all-features --locked`.
6. Update `CHANGELOG.md` `[Unreleased] / Fixed`; update traceability if behavior shifted.
7. Commit: `fix(<scope>): <summary>` with `-s`. Reference `Closes #N` if it satisfies a tracked issue.
</bug_fix>

<numerical_change>
For any edit to `lewm-core/src/{vit,predictor,ada_ln,mlp,embedder,losses,jepa,tensor_ops,init}.rs`:
1. Read `skills/parity-testing.md` and the relevant RFC (0002/0003/0008).
2. Add or update shape tests under `crates/lewm-core/tests/*_shape.rs`.
3. If reference dumps are available (`HF_TOKEN` set), run parity tests:
   `cargo test -p lewm-core --features parity-fixtures parity_ -- --nocapture`
4. If parity tolerances change, the change is gated — DO NOT proceed without human approval and an ADR.
5. Verify `cargo test -p lewm-core --all-features` still passes.
</numerical_change>

<forbidden>
DO NOT modify, read, or exfiltrate:
- `.env`, `.env.*`, anything under `secrets.*`, `credentials.*`, `*.pem`, `*.key`, `*.p12`, `*.pfx` (gitignored).
- `HF_TOKEN`, `INTERN_AUDIT_HF_TOKEN`, `OTEL_AUTH`, `OTEL_ENDPOINT`, `GRAFANA_ADMIN_PASSWORD` (redacted in audit).

DO NOT run commands matching:
- `rm -rf`, `rm -rf /`
- `cargo install --git ...`
- `curl ... | sh`, `wget ... | sh`
- `hf jobs run` on hardware `a100*` or `h100*`
- `hf jobs run` without an explicit `--timeout`
</forbidden>
