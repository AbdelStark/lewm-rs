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

<adding_dependency>
1. Confirm the license is in `deny.toml`'s allowlist (MIT, Apache-2.0, BSD-2/3, ISC, Zlib, Unlicense, CC0, MPL-2.0, Unicode-3.0, CDLA-Permissive-2.0, HDF5).
2. Add to `[workspace.dependencies]` in root `Cargo.toml` with an EXACT version pin where the existing pattern uses `=X.Y.Z` (e.g., Burn family, Tract family) or a caret/range otherwise.
3. Reference from the crate with `{ workspace = true }` — never inline-version.
4. Run `cargo deny check` and `cargo audit` (both included in `make check`).
5. If a new RUSTSEC advisory shows up, document the waiver path in an ADR; do not silently `--ignore` in `Makefile`.
6. Update `CHANGELOG.md` `[Unreleased] / Added`.
</adding_dependency>

<launching_an_hf_job>
1. Confirm the job filename appears in `.ml-intern/cli_agent_config.json` → `jobs_allowed`. If it's in `jobs_human_approval_required`, STOP and ask the user before proceeding.
2. Confirm hardware in the YAML is one of `cpu-basic|cpu-xl|l4x1|a10g-large`. Never `a100*` or `h100*` (denied).
3. Confirm `timeout:` is set (regex-banned otherwise).
4. State the estimated cost (see `python/hf_pricing.py` / `reports/cost.md`) and the running session-cap remaining before launching.
5. Run via `scripts/launch_hf_job.py jobs/<file>.yaml`.
6. Append the result (job id, cost, status) to `reports/cost.md`; re-run `python python/cost_ledger.py check --path reports/cost.md --cap-usd 200` (already in `make check`).
</launching_an_hf_job>

<adding_a_skill_or_rfc>
- New RFC: copy `specs/rfcs/0000-template.md`, assign the next number, set `Status: Draft`, link from `specs/README.md` §1.2. Run `python3 scripts/check_specs.py` before commit.
- New ADR: copy `specs/adr/0000-template.md`, assign the next number, name it precisely (`adr/NNNN-<kebab-decision>.md`). ADRs are immutable once Accepted — supersede with a new ADR.
- New skill: drop `<name>.md` into `.codex/skills/`, register it in `.codex/skills/_index.md`. The `.claude/skills` and `.agents/skills` symlinks pick it up automatically.
</adding_a_skill_or_rfc>
</workflows>

<boundaries>
The `.ml-intern/cli_agent_config.json` file is the authoritative agent leash for this repo. The mirror below is for fast lookup — if it disagrees with the JSON, the JSON wins.

<forbidden>
DO NOT modify, read, or exfiltrate:
- `.env`, `.env.*`, anything under `secrets.*`, `credentials.*`, `*.pem`, `*.key`, `*.p12`, `*.pfx` (gitignored).
- `HF_TOKEN`, `INTERN_AUDIT_HF_TOKEN`, `OTEL_AUTH`, `OTEL_ENDPOINT`, `GRAFANA_ADMIN_PASSWORD` (redacted in audit).

DO NOT run commands matching:
- `rm -rf`, `rm -rf /`
- `git push --force`, `git push -f`, `git push --force-with-lease`
- `git checkout main`, `git reset --hard`
- `cargo install --git ...`
- `curl ... | sh`, `wget ... | sh`
- `hf jobs run` on hardware `a100*` or `h100*`
- `hf jobs run` without an explicit `--timeout`

DO NOT push to `main` directly. The CODEOWNER (`@AbdelStark`) reviews everything.
</forbidden>

<gated>
Modify ONLY with explicit human approval (the agent must stop and ask):

| Glob / path                              | Reason                                               |
|------------------------------------------|------------------------------------------------------|
| `crates/lewm-core/src/losses/**`         | Loss math is the contract; changes invalidate parity |
| `specs/**`, `PRD.md`                     | Spec set; changes require RFC/ADR process            |
| `configs/pusht.toml`, `configs/so100.toml` | Locked training configs                            |
| `rust-toolchain.toml`, `Cargo.lock`      | Repro contract (RFC 0011/0013)                       |
| `jobs/train_pusht.yaml`, `jobs/train_so100.yaml` | Full training runs (cost-gated)              |
| Any HF-Hub upload to `abdelstark/lewm-rs-*` non-`smoke/` paths | Public artifacts                |
| Lockfile changes, dep version bumps       | License + audit + waiver review needed              |

Commands requiring confirmation per the leash: `hf jobs run`, `hf upload`, `git push`, `cargo publish`, `cargo add`, `uv add`.
</gated>

<writable>
Free to create/modify without prior approval (still subject to `make check`):
- `reports/**`, `python/**` (except secret material), `jobs/<new smoke/short>.yaml` only if first added to the allowlist, `configs/overrides/**`, `.ml-intern/sessions/**`, `.codex/skills/**`, `docs/src/**`.
- Source code inside any `crates/*/src/` and `crates/*/tests/` is editable subject to RFC conformance and parity preservation.
</writable>

<safety_checks>
Before ANY destructive or shared-state action (file delete, force-overwrite, `git push`, HF upload, job launch):
1. State the command and what it touches.
2. State the irreversible impact and the cost in $ (for HF Jobs / uploads).
3. Wait for confirmation. Approval once does NOT generalize to repeated calls.
</safety_checks>
</boundaries>

<troubleshooting>
<known_issues>
| Symptom                                                              | Cause                                                            | Fix                                                                                  |
|----------------------------------------------------------------------|------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| `error: failed to load HDF5 plugin` when reading `pusht_expert_*.h5` | Blosc filter not on path                                         | `export HDF5_PLUGIN_PATH=$(python -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)')` |
| Clippy fails on `unwrap_used` in a test                              | Tests allowed via `clippy.toml` — check you're really in tests   | Move `.unwrap()` into `#[cfg(test)]` or use `?`                                       |
| `cargo deny check` fails on a new dep license                        | License not in `deny.toml` allowlist                             | Either swap dep or add to ADR with rationale; do not edit `deny.toml` casually        |
| `cargo audit` reports a new RUSTSEC                                  | Upstream vulnerability                                           | Upgrade if possible; else open ADR and add scoped `--ignore` to `Makefile`            |
| `make check` fails at `check_layers.py`                              | A crate took on a forbidden inter-crate dep                      | Move the code, not the allowlist. See `skills/crate-layering.md`                      |
| `make check` fails at `check_nondet.py`                              | `thread_rng` / banned RNG entry-point in code                    | Use an RFC-0013 ChaCha20 sub-stream. See `skills/determinism-rng.md`                  |
| Parity test fails with `L∞ > 1e-4`                                   | Numerical drift in encoder/predictor                             | DO NOT loosen tolerance. Bisect commit, fix root cause, escalate if needed            |
| ONNX export load fails in Tract                                      | Opset / dynamic shapes                                           | Tract uses opset 17 fixed-batch; see `python/export_onnx.py` and RFC 0007             |
| HF Job CLI errors "hardware denied"                                  | Used `a100*` / `h100*`                                           | Use `l4x1` or `a10g-large`. Per leash, this is non-negotiable                         |
| `cargo bench` ICEs on macOS with `cuda` feature                      | `cuda` feature enabled by default on `lewm-train`/`lewm-plan`    | Build with `--no-default-features --features cpu-only` on macOS                       |
</known_issues>

<recovery_cascade>
When `make check` fails, in order:
1. Read the FIRST failing step's full output — later steps may cascade.
2. If fmt: run `make fmt`, then re-run `make check`.
3. If clippy: address the diagnostic. Do NOT `#[allow]` denied lints.
4. If py-lint: `ruff check --fix --config python/pyproject.toml python scripts` and re-verify.
5. If `check_layers.py` / `check_nondet.py` / `check_jobs.py`: open the relevant skill (`skills/crate-layering.md`, `skills/determinism-rng.md`, `skills/hf-jobs-cost.md`).
6. If `cargo deny` / `cargo audit`: see <adding_dependency>.
7. If still stuck after 2 passes, summarize the failure and ask the human.
</recovery_cascade>
</troubleshooting>

<environment>
- Harness: Claude Code on the web (ephemeral container; fresh clone per session). State that isn't committed is lost.
- File system scope: this repository tree only.
- Network: governed by the environment's network policy.
- Tools: Bash, Read/Edit/Write, GitHub MCP (scoped to `abdelstark/lewm-rs`), no `gh` CLI, no direct shell GitHub access.
- Human interaction: asynchronous; pause and ask via `AskUserQuestion` rather than guess on gated actions.
- Designated working branch this session: `claude/agentic-context-framework-e8B3p` (declared at session start; do not push elsewhere without explicit permission).
</environment>

<skills>
Modular skills live in `.codex/skills/` and are accessible via `.claude/skills/` and `.agents/skills/` symlinks. Load a skill file when entering its domain.

| Skill                                | When to load                                                         |
|--------------------------------------|----------------------------------------------------------------------|
| `crate-layering.md`                  | Adding/changing crate deps; `check_layers.py` failures               |
| `parity-testing.md`                  | Any change to `lewm-core/src/{vit,predictor,ada_ln,mlp,embedder,losses,jepa}.rs` |
| `burn-modules.md`                    | Implementing/modifying a `burn::module::Module`                      |
| `determinism-rng.md`                 | RNG, seeds, ordering, checkpoint resume                              |
| `hf-jobs-cost.md`                    | Launching HF Jobs, editing `jobs/*.yaml`, updating `reports/cost.md` |
| `rfc-adr-process.md`                 | Editing/adding RFCs or ADRs; PR contradicts a clause                 |
| `quality-gate.md`                    | `make check` / `make accept` failures, CI repro locally              |
| `python-helpers.md`                  | Editing anything under `python/` or `scripts/`                       |
| `release-and-artifacts.md`           | Hub uploads, model cards, release tags                               |

Registry: `.codex/skills/_index.md`.
</skills>

<memory>
<project_decisions>
- 2026-05-12: Specs locked at v1.0.0 (Accepted). RFCs 0001–0018 govern contracts; deviation requires ADR. See `specs/README.md`.
- 2026-05-13: ADR 0001 pins the PushT reference architecture (dims, init, AdaLN-zero) and forbids drift without a new ADR.
- 2026-05-14: ADR 0002 — Burn 0.20.1 → bincode 2.0.1 RUSTSEC waiver with date bound; tracked in `Makefile` audit line.
- 2026-05-15: `lewm-gpu` carved out so `lewm-infer` stays free of CUDA / autodiff / NVML (RFC 0007). `scripts/check_layers.py` enforces.
- 2026-05-15: Tract ONNX export pinned to opset 17, `dynamo=False`, fixed-batch (no dynamic_axes). Driven by Tract 0.22.1 limits.
- 2026-05-15: HF Jobs hardware leash: `cpu-basic|cpu-xl|l4x1|a10g-large` only. `a100*`/`h100*` denied. Hard cap $200, session cap $20.
</project_decisions>

<lessons_learned>
- LayerNorm `eps=1e-12` and exact-erf GELU were both required for L∞ < 1e-4 parity (PR #217). Do not "round" these.
- Predictor's `T=num_frames` is fixed; `lewm-infer bench --history-steps` must default to 3 to avoid Tract shape clash.
- `action_dim` is INFERRED from the action smoother Conv1d weight shape (10-DOF) — not the raw config value (2-DOF). Anywhere this is logged or exported, infer it.
- `make check` runs `cargo audit` against a local DB at `target/advisory-db/cargo-audit`; in offline-only environments, prime this cache before running `make check`.
- `CARGO_INCREMENTAL=0` is required for the canonical `make check` invocation to match CI behavior.
</lessons_learned>
</memory>
