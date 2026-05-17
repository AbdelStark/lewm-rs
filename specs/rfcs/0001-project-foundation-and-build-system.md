---
rfc: "0001"
title: "Project foundation, workspace layout, build system"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§5.1", "§16 Appendix C"]
depends_on: []
related: ["0011", "0016", "0017", "0018"]
---

# RFC 0001 — Project foundation, workspace layout, build system

> **Status:** Accepted · **Version:** 1.0.0
>
> Establishes the immovable shape of `lewm-rs`: how the workspace is laid out, which toolchain pins it relies on, how crates and features and profiles are organized, and the leash that the agent (ml-intern) operates under. Everything else builds on this.

---

## 1. Introduction

### 1.1 Motivation

The PRD specifies the seven Rust crates, three binaries, Python edge layer, and operational topology. This RFC turns those into a buildable, testable, releasable artifact set. It is the **first** RFC because every later RFC names files and modules that must exist somewhere, and every later RFC depends on a stable toolchain and a stable feature surface.

### 1.2 Goals

1. Define a workspace structure that scales to seven crates without circular deps and that maps 1-to-1 to the PRD's deliverable list.
2. Pin a single Rust toolchain, a single Burn version, a single Tract version, and a documented upgrade path.
3. Provide CI-compatible build profiles for development, release, benchmarking, and reproducible release-binary production.
4. Encode the layer invariants (INV-001 through INV-004) as enforceable checks.
5. Encode the ml-intern leash in a configuration file that the agent reads at session start.

### 1.3 Non-goals

- CI workflow content — handled by [RFC 0011](0011-ci-cd-and-release-engineering.md).
- Per-crate API surfaces — handled by the crate-specific RFCs.
- Container image internals — handled by RFC 0011.
- Release process — handled by RFC 0011.

### 1.4 Stakeholders

- Implementer: developer of any crate.
- Reviewer: anyone reading a PR; relies on layer invariants.
- Operator: the human launching HF Jobs; reads profile and feature documentation.
- Agent (ml-intern): reads the leash file.

---

## 2. Conventions

Per [`specs/README.md`](../README.md) §2. The glossary section "Workspace terms" is binding for this RFC.

---

## 3. Toolchain and dependency pinning

### 3.1 Rust toolchain

The project pins a single toolchain version in `rust-toolchain.toml` at the repo root:

```toml
# rust-toolchain.toml
[toolchain]
channel    = "1.95.0"        # stable; >= Burn 0.21.0 MSRV (1.92), per ADR 0003
profile    = "default"
components = ["rustfmt", "clippy", "rust-src", "rust-analyzer"]
targets    = ["x86_64-unknown-linux-gnu", "aarch64-apple-darwin"]
```

**RFC0001-001 [MUST]** — The toolchain pin **MUST** match the version in `rust-toolchain.toml` on every CI runner and developer machine. CI fails fast if `rustc -V` does not match.

**RFC0001-002 [MUST]** — Bumping the channel **REQUIRES** an ADR that captures: (a) the upstream changelog scan for breakage, (b) a re-run of all parity tests, (c) confirmation that Burn supports the new channel.

`edition = "2024"` is used in every member crate (`Cargo.toml` workspace inheritance).

### 3.2 Burn pin

```toml
# workspace dependency block (excerpt); per ADR 0003
burn          = { version = "=0.21.0", default-features = false }
burn-cuda     = { version = "=0.21.0" }
burn-ndarray  = { version = "=0.21.0" }
burn-autodiff = { version = "=0.21.0" }
burn-import   = { version = "=0.21.0" }
burn-train    = { version = "=0.21.0", default-features = false }
```

**RFC0001-003 [MUST]** — Burn version is locked with the `=` prefix (exact match). Upgrades require an ADR that includes a green parity-test run on the new version.

### 3.3 Tract pin

```toml
tract           = "=0.22.1"
tract-onnx      = "=0.22.1"
tract-nnef      = "=0.22.1"
```

**RFC0001-004 [MUST]** — Tract version is locked at `=0.22.1`. The inference report cites this version verbatim.

### 3.4 Other workspace dependencies

The complete `[workspace.dependencies]` block is reproduced in §A.1 of this RFC. The set is **closed** under normal evolution: adding a new third-party dependency to the workspace **MUST** go through `cargo deny check` plus a one-line PR note. CI runs `cargo deny` on every PR (see [RFC 0016 §5](0016-security-and-supply-chain.md)).

### 3.5 Python edge

A `python/` directory contains scripts run only at dataset prep and weight conversion edges. They depend on:

```
# python/pyproject.toml — extract
[project]
name = "lewm-rs-tools"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
  "torch>=2.4,<3.0",
  "safetensors>=0.5",
  "huggingface-hub>=0.25",
  "datasets>=3.0",
  "trackio>=0.0.5",
  "numpy>=2.0",
  "av>=12.0",                # MP4 decode via PyAV
  "h5py>=3.11",
  "pyarrow>=17.0",
  "tqdm>=4.66",
]
```

A `uv.lock` file commits the lockfile. The Python edge **MUST NOT** be in the training-run hot path; it runs once during Phase 0/4 dataset prep and during the per-checkpoint upload.

---

## 4. Workspace layout

### 4.1 Top-level directory

```
lewm-rs/
├── Cargo.toml                  # workspace manifest
├── Cargo.lock                  # committed
├── rust-toolchain.toml         # pin
├── rustfmt.toml                # formatting config (§4.5)
├── clippy.toml                 # lint config (§4.6)
├── deny.toml                   # cargo-deny config (RFC 0016)
├── Makefile                    # convenience targets (§4.7)
├── Dockerfile                  # multi-stage build (RFC 0011 §5)
├── README.md
├── LICENSE                     # MIT
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CHANGELOG.md
├── .editorconfig
├── .gitignore
├── .gitattributes
├── .github/
│   ├── workflows/{ci,release,specs,nightly,docs}.yml
│   ├── ISSUE_TEMPLATE/{bug,feature,parity}.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── CODEOWNERS
├── crates/
│   ├── lewm-core/
│   ├── lewm-data/
│   ├── lewm-train/
│   ├── lewm-plan/
│   ├── lewm-infer/
│   ├── lewm-telemetry/
│   └── lewm-hub/
├── python/
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── convert_reference.py
│   ├── decode_so100_to_h5.py
│   ├── upload_checkpoints.py
│   ├── plot_curves.py
│   └── tests/                    # pytest
├── scripts/
│   ├── check_layers.py
│   ├── check_specs.py
│   ├── check_unused_deps.sh
│   ├── bench_to_report.py
│   ├── cost_ledger.py
│   └── run_local.sh
├── jobs/
│   ├── smoke_pusht.yaml
│   ├── short_pusht.yaml
│   ├── train_pusht.yaml
│   ├── smoke_so100.yaml
│   ├── short_so100.yaml
│   ├── train_so100.yaml
│   └── eval.yaml
├── configs/
│   ├── pusht.toml
│   ├── so100.toml
│   ├── pusht_warmstart.toml      # encoder pre-warm config for SO-100
│   └── overrides/                # CLI-override TOMLs
├── reports/
│   ├── pusht_schema.md
│   ├── pusht_training.md
│   ├── so100_training.md
│   ├── inference.md
│   ├── cost.md
│   ├── parity.md
│   └── lambda_sweep.md
├── paper/
│   ├── lewm-rs.md
│   ├── lewm-rs.pdf
│   ├── figures/
│   └── bibliography.bib
├── specs/                          # this directory
└── .ml-intern/
    ├── cli_agent_config.json       # leash, §7
    ├── README.md
    └── prompts/system.md
```

**RFC0001-005 [MUST]** — The above directory layout is the canonical structure. Adding a new top-level directory requires updating both this RFC and `Appendix C` of the master spec.

### 4.2 Crate manifests

Each crate `crates/<name>/Cargo.toml` follows this template:

```toml
[package]
name        = "lewm-<name>"
version     = { workspace = true }
edition     = { workspace = true }
license     = { workspace = true }
authors     = { workspace = true }
repository  = { workspace = true }
description = "<one-line>"

[lib]
path = "src/lib.rs"

# optional: bin section for binary-bearing crates (lewm-train, lewm-plan, lewm-infer)
[[bin]]
name = "lewm-train"
path = "src/bin/lewm-train.rs"

[features]
default = []
# crate-specific features declared here; see §5

[dependencies]
# pulled from workspace deps
burn          = { workspace = true }
serde         = { workspace = true, features = ["derive"] }
# ...

[dev-dependencies]
proptest = "1"
insta    = "1"

[lints]
workspace = true
```

**RFC0001-006 [MUST]** — Every crate **MUST** inherit `package.version`, `edition`, `license`, `authors`, `repository`, and `lints` from the workspace.

### 4.3 Binary entry points

The deliverable binaries are:

| Binary | Crate | Path | Purpose |
|--------|-------|------|---------|
| `lewm-train` | `lewm-train` | `src/bin/lewm-train.rs` | Trainer CLI (subcommands per RFC 0005 §4) |
| `lewm-eval` | `lewm-plan` | `src/bin/lewm-eval.rs` | Eval CLI (RFC 0006 §4) |
| `lewm-infer` | `lewm-infer` | `src/bin/lewm-infer.rs` | Tract CPU planner CLI (RFC 0007 §6) |

Each binary is a `clap`-derived argument parser whose top-level commands route to functions in the library crate.

### 4.4 Workspace manifest

```toml
# Cargo.toml (workspace root)
[workspace]
resolver = "2"
members  = [
  "crates/lewm-core",
  "crates/lewm-data",
  "crates/lewm-train",
  "crates/lewm-plan",
  "crates/lewm-infer",
  "crates/lewm-telemetry",
  "crates/lewm-hub",
]
default-members = ["crates/lewm-core", "crates/lewm-data"]

[workspace.package]
version    = "0.1.0"
edition    = "2024"
license    = "MIT"
authors    = ["Abdel <abdel@starkware.co>"]
repository = "https://github.com/AbdelStark/lewm-rs"
rust-version = "1.95"

[workspace.dependencies]
# see Appendix A.1

[workspace.lints.rust]
unsafe_code                      = "deny"
unreachable_pub                  = "warn"
missing_docs                     = "warn"
missing_debug_implementations    = "warn"
rust_2024_compatibility          = "warn"

[workspace.lints.clippy]
all                              = { level = "warn", priority = -1 }
pedantic                         = { level = "warn", priority = -1 }
unwrap_used                      = "deny"
expect_used                      = "warn"        # allowed with invariant string
panic                            = "warn"
todo                             = "warn"
unimplemented                    = "deny"
dbg_macro                        = "deny"
print_stdout                     = "warn"        # use tracing instead
multiple_crate_versions          = "allow"       # Burn pulls older serde transitively
module_name_repetitions          = "allow"
must_use_candidate               = "allow"
```

**RFC0001-007 [MUST]** — The workspace lints **MUST** apply via `[lints] workspace = true` in every member crate. CI runs `cargo clippy --workspace --all-targets -- -D warnings`.

### 4.5 Formatting

```toml
# rustfmt.toml
edition = "2024"
max_width = 100
hard_tabs = false
tab_spaces = 4
newline_style = "Unix"
use_field_init_shorthand = true
use_try_shorthand = true
imports_granularity = "Crate"
imports_layout = "HorizontalVertical"
group_imports = "StdExternalCrate"
reorder_imports = true
reorder_modules = true
struct_field_align_threshold = 0
trailing_semicolon = true
trailing_comma = "Vertical"
match_block_trailing_comma = true
fn_single_line = false
where_single_line = false
```

**RFC0001-008 [MUST]** — CI runs `cargo fmt --all -- --check`. PRs fail on formatting diffs.

### 4.6 Linting

```toml
# clippy.toml
avoid-breaking-exported-api = false
cognitive-complexity-threshold = 30
type-complexity-threshold = 250
max-fn-params-bools = 1
allow-expect-in-tests = true
allow-unwrap-in-tests = true
```

### 4.7 Makefile

```makefile
# Makefile (excerpt)
.PHONY: fmt lint test test-fast bench docs check accept clean

fmt:
	cargo fmt --all

lint:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace --all-features

test-fast:
	cargo test --workspace --lib --bins -- --skip "_slow_"

bench:
	cargo bench --workspace

docs:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

check: fmt lint
	cargo check --workspace --all-targets
	python scripts/check_layers.py
	python scripts/check_specs.py
	cargo deny check
	cargo audit

accept:
	$(MAKE) check
	$(MAKE) test
	$(MAKE) docs
	$(MAKE) -C python check
	scripts/check_release_inventory.sh

clean:
	cargo clean
```

`make accept` is the canonical release gate referenced by the master spec §12.2.

---

## 5. Cargo features

Features are conservative: most behavior is selected at runtime via config, not compile-time. The following are the only cross-cutting features.

| Feature | Crates | Default? | Purpose |
|---------|--------|----------|---------|
| `cuda` | `lewm-train`, `lewm-plan` | yes (under `default`) | Enable `burn-cuda` backend |
| `cpu-only` | `lewm-train`, `lewm-plan` | no | Force `NdArray` backend; CUDA deps absent |
| `metal` | `lewm-train` | no | Burn Metal backend (Apple Silicon dev) |
| `tract-onnx` | `lewm-infer` | yes | ONNX path |
| `tract-nnef` | `lewm-infer` | yes | NNEF path |
| `python-bridge` | `lewm-hub` | no | PyO3 wrappers (dev only; RFC 0010 §5) |
| `parity-fixtures` | `lewm-core` | no | Bundle parity reference dumps for tests |
| `slow-tests` | all | no | Long-running tests, run in nightly CI only |

**RFC0001-009 [MUST NOT]** — A feature **MUST NOT** change a public API signature; it may only add code or alter implementation. (This is the invariant that keeps `cargo build -p lewm-core` and `cargo build -p lewm-core --features parity-fixtures` produce libraries with the same public surface.)

**RFC0001-010 [MUST]** — `cargo hack check --feature-powerset --workspace` passes in CI nightly.

---

## 6. Build profiles

```toml
# Cargo.toml — profiles
[profile.dev]
opt-level = 1               # without this Burn tensor ops are unusable in tests
debug = true
overflow-checks = true
lto = false
codegen-units = 256
incremental = true

[profile.release]
opt-level = 3
debug = "line-tables-only"  # keep enough symbols for crash logs
overflow-checks = false
lto = "thin"
codegen-units = 1
strip = false
panic = "abort"

[profile.release-lto]
inherits = "release"
lto = "fat"
codegen-units = 1
incremental = false
panic = "abort"

[profile.bench]
inherits = "release"
debug = "line-tables-only"
lto = "thin"

[profile.dev-fast]            # for snap test iteration
inherits = "dev"
opt-level = 0
debug = true
```

**RFC0001-011 [MUST]** — Release binaries published to GitHub Releases **MUST** be built under `release-lto`. CI verifies via `--profile release-lto`.

**RFC0001-012 [MUST]** — `panic = "abort"` in release profiles. Reasoning: the trainer's crash-resume protocol is **resume**-based (state persisted to disk), not **unwind**-based. Unwinding from a CUDA stream is undefined; abort is the safe stance.

---

## 7. ml-intern leash

The agent is a force multiplier and a blast radius hazard. The leash is encoded in two places: a JSON config that the agent reads at session start, and a system prompt that reinforces the same rules in natural language.

### 7.1 `cli_agent_config.json`

```jsonc
// .ml-intern/cli_agent_config.json
{
  "schema_version": "1.0.0",
  "project": "lewm-rs",
  "namespace": "abdelstark",
  "billing": {
    "hard_cap_usd": 200,
    "soft_cap_usd": 100,
    "per_job_default_timeout": "30m"
  },
  "hardware_allowed": ["cpu-basic", "cpu-xl", "l4x1", "a10g-large"],
  "hardware_denied":  ["a100-large", "a100-xl", "h100", "h100-xl"],
  "jobs_allowed": [
    "smoke_pusht.yaml",
    "short_pusht.yaml",
    "smoke_so100.yaml",
    "short_so100.yaml",
    "eval.yaml"
  ],
  "jobs_human_approval_required": [
    "train_pusht.yaml",
    "train_so100.yaml"
  ],
  "files_readonly_glob": [
    "crates/lewm-core/src/losses/**",
    "specs/**",
    "PRD.md",
    "configs/pusht.toml",
    "configs/so100.toml"
  ],
  "files_writable_glob": [
    "reports/**",
    "python/**",
    "jobs/**",
    "configs/overrides/**"
  ],
  "audit": {
    "session_log_repo": "abdelstark/lewm-rs-intern-audit",
    "upload_at_session_end": true
  },
  "command_denylist": [
    "rm -rf",
    "git push --force",
    "git checkout main",
    "cargo install --git",
    "hf jobs run --hardware a100*",
    "hf jobs run --hardware h100*"
  ],
  "command_require_confirmation": [
    "hf jobs run",
    "hf upload",
    "git push",
    "cargo publish"
  ]
}
```

**RFC0001-013 [MUST]** — Every ml-intern session **MUST** load this file before issuing its first command. CI verifies the schema with `scripts/check_intern_config.py`.

**RFC0001-014 [MUST]** — Edits to `cli_agent_config.json` are tracked separately from edits to `prompts/system.md`; both **MUST** be reviewed by a human.

### 7.2 System prompt

The system prompt is a short, declarative reinforcement of the JSON rules. It lives at `.ml-intern/prompts/system.md` and is uploaded with every session log.

A minimal opening:

> You are operating in the `lewm-rs` project. The PRD is at `/PRD.md` and is binding. The spec set is at `/specs/` and is binding. The hard cost cap is 200 USD; the soft cap is 100 USD. Do not launch any job whose YAML is in `jobs_human_approval_required` without an explicit human go-ahead. Do not edit any file matching `files_readonly_glob`. When in doubt, stop and ask.

---

## 8. Quickstart contract

**RFC0001-015 [MUST]** — A new contributor with Rust installed must be able to:

```bash
git clone https://github.com/AbdelStark/lewm-rs
cd lewm-rs
make test
```

…and see green in ≤ 5 minutes on a recent laptop. The `make test` target is exactly `cargo test --workspace --all-features` minus any test gated by `--features slow-tests`.

This is the realization of NFR-030.

---

## 9. Container image contract

A multi-stage Dockerfile produces a `~ 400 MB` runtime image based on `nvidia/cuda:12.4.1-runtime-ubuntu22.04`. The Dockerfile contract is fully specified in [RFC 0011 §5](0011-ci-cd-and-release-engineering.md); this RFC only asserts that one **MUST** exist at `/Dockerfile` and **MUST** be built and pushed to GHCR by CI on every tag.

---

## 10. Testing strategy

### 10.1 Test inventory (this RFC's scope)

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0001-QS-001 | `quickstart_in_5_minutes` | shell (CI) | NFR-030 |
| TST-0001-LAYERS-001 | `check_layers_no_violations` | python | INV-001..004 |
| TST-0001-FEATURES-001 | `cargo hack feature powerset` | shell | RFC0001-010 |
| TST-0001-FMT-001 | `cargo fmt --check` | shell | RFC0001-008 |
| TST-0001-CLIPPY-001 | `cargo clippy -D warnings` | shell | RFC0001-007 |
| TST-0001-PIN-001 | `verify rust-toolchain matches` | shell | RFC0001-001 |
| TST-0001-INTERN-001 | `validate intern config schema` | python | RFC0001-013 |

### 10.2 Layer check

`scripts/check_layers.py` parses each crate's `Cargo.toml` and asserts:

```python
# pseudocode
ALLOWED_DEPS = {
    "lewm-core":      set(),
    "lewm-data":      {"lewm-core"},
    "lewm-hub":       {"lewm-core"},
    "lewm-telemetry": {"lewm-core"},
    "lewm-plan":      {"lewm-core", "lewm-data", "lewm-telemetry"},
    "lewm-train":     {"lewm-core", "lewm-data", "lewm-telemetry", "lewm-hub", "lewm-plan"},
    "lewm-infer":     {"lewm-core", "lewm-telemetry"},
}
```

…and additionally that `lewm-infer` does **not** depend on `burn-cuda` or `burn-autodiff` (INV-003).

### 10.3 Quickstart smoke

`TST-0001-QS-001` runs `make test` from a clean checkout in a fresh container; asserts duration < 5 minutes and exit code 0. Listed in the CI matrix.

---

## 11. Operational considerations

### 11.1 Observability

Build commands print structured logs only when `RUST_LOG=lewm=info,cargo=warn` is set. No `print_stdout` in production code (clippy `print_stdout = warn`).

### 11.2 Runbook

- **"My toolchain doesn't match the pin"** — `rustup toolchain install $(grep '^channel' rust-toolchain.toml | cut -d'"' -f2)`.
- **"`make test` is slow"** — first run is cold; subsequent runs hit the `sccache` cache if `RUSTC_WRAPPER=sccache`.
- **"`cargo deny` failed for a new dep"** — add the dep's license to `deny.toml` allowlist if MIT-compatible; otherwise reject the dep.

### 11.3 Capacity

CI uses `ubuntu-22.04` (4 vCPU, 16 GB RAM) for non-GPU jobs, `ubuntu-22.04-gpu-l4` for GPU jobs. Build cache via `Swatinem/rust-cache@v2`.

---

## 12. Performance considerations

`cargo build` cold ≤ 5 min on `ubuntu-22.04`, warm ≤ 30 s. `cargo test --workspace` cold ≤ 8 min, warm ≤ 90 s.

---

## 13. Security considerations

- `unsafe_code = "deny"` workspace-wide. Any `unsafe` block requires a justifying comment and review by a code owner.
- Supply chain: `cargo deny`, `cargo audit`. See [RFC 0016 §5](0016-security-and-supply-chain.md).
- The ml-intern leash is enforced both by config and prompt; trust boundary is "the agent might be tricked, the JSON cannot be."
- Git LFS files limited to `≤ 50 MB`. Larger fixtures live on HF Hub.

---

## 14. Alternatives considered

- **A1 — Cargo virtual workspace with no top-level lib.** Adopted (current design). Alternative: a single library crate. Rejected because the deliverables call for crate-level reuse (`lewm-core` linked from `lewm-train`, `lewm-plan`, `lewm-infer` with different feature flags).
- **A2 — Two repos (model vs. infra).** Rejected; the project is small enough that one repo is simpler, and the spec set's tight coupling makes a two-repo split a paperwork tax.
- **A3 — `xtask` pattern instead of `Makefile`.** Considered; we have both `make accept` and `cargo xtask check` planned. For v1 we ship `make` only and revisit if non-Linux contributors object (rare given the GPU dep).
- **A4 — Different Burn version.** Originally pinned `0.20.1` (PRD time, API stable); upgraded to `0.21.0` per ADR 0003 once the API surface and `bincode 2.0.1` exposure remained equivalent.

---

## 15. Acceptance criteria

- [ ] `Cargo.toml` workspace manifest matches §4.4.
- [ ] All seven crates exist at their canonical paths.
- [ ] `rust-toolchain.toml` exists and CI enforces the pin.
- [ ] `make test`, `make lint`, `make docs`, `make accept` all defined.
- [ ] `scripts/check_layers.py` and `scripts/check_specs.py` exist and pass.
- [ ] `.ml-intern/cli_agent_config.json` exists and matches §7.1.
- [ ] CI matrix executes all of TST-0001-*.

---

## 16. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | Burn 0.21.x soft-yanks an API we use | M | H | Pinned `=0.21.0`; ADR-gated upgrades and weekly dependency review |
| R-2 | Rust 1.95.0 has a regression for our hot loop | L | M | `rust-toolchain.toml` allows pinned override; rollback ADR template ready |
| R-3 | Workspace grows beyond seven crates | M | L | New crate requires updating §4.1 and `traceability-matrix.md` |
| R-4 | Layer invariants get tedious to enforce manually | L | M | `scripts/check_layers.py` is automatic |

---

## 17. Open questions

None.

---

## A. Appendix — full `[workspace.dependencies]` block

```toml
[workspace.dependencies]
# Burn ecosystem (per ADR 0003)
burn          = { version = "=0.21.0", default-features = false }
burn-cuda     = { version = "=0.21.0" }
burn-ndarray  = { version = "=0.21.0" }
burn-autodiff = { version = "=0.21.0" }
burn-import   = { version = "=0.21.0" }
burn-train    = { version = "=0.21.0", default-features = false }

# Inference
tract           = "=0.22.1"
tract-onnx      = "=0.22.1"
tract-nnef      = "=0.22.1"

# Data
hdf5-metno      = "0.10"
parquet         = "56"
arrow-array     = "56"
arrow-schema    = "56"
image           = "0.25"
safetensors     = "0.5"
ndarray         = "0.16"

# Concurrency
tokio           = { version = "1", features = ["rt-multi-thread", "macros", "fs", "sync", "signal"] }
crossbeam       = "0.8"
parking_lot     = "0.12"
rayon           = "1.10"

# Telemetry
tracing               = "0.1"
tracing-subscriber    = { version = "0.3", features = ["env-filter", "json", "fmt"] }
tracing-opentelemetry = "0.27"
opentelemetry         = { version = "0.27", features = ["trace", "metrics"] }
opentelemetry-otlp    = { version = "0.27", features = ["grpc-tonic", "trace", "metrics"] }
opentelemetry_sdk     = "0.27"

# Serialization, config, CLI
serde     = { version = "1", features = ["derive"] }
serde_json = "1"
toml      = "0.8"
clap      = { version = "4.5", features = ["derive", "env", "wrap_help"] }
config    = "0.14"
validator = { version = "0.18", features = ["derive"] }

# Error handling
anyhow    = "1"
thiserror = "2"
miette    = { version = "7", features = ["fancy"] }

# RNG
rand           = { version = "0.8", default-features = false, features = ["std", "std_rng"] }
rand_chacha    = "0.3"
rand_distr     = "0.4"
blake3         = "1"

# HF Hub
hf-hub = { version = "0.4", default-features = false, features = ["tokio", "ureq"] }

# Utilities
chrono     = { version = "0.4", default-features = false, features = ["clock", "serde"] }
indicatif  = "0.17"
humantime  = "2"
bytes      = "1"
glob       = "0.3"
walkdir    = "2"
once_cell  = "1"
itertools  = "0.13"
regex      = "1"

# Testing
proptest   = "1"
insta      = "1"
criterion  = "0.5"
```

---

## 18. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0001.*
