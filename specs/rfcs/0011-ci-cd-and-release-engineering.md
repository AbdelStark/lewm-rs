---
rfc: "0011"
title: "CI/CD, release engineering, container images, reproducibility"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§5.1 layout", "§6.5 HF Jobs"]
depends_on: ["0001", "0008", "0009", "0010"]
related: ["0013", "0016"]
---

# RFC 0011 — CI/CD, release engineering, container images, reproducibility

> **Status:** Accepted · **Version:** 1.0.0
>
> The release process is the gate between a green local commit and a published artifact. This RFC specifies the GitHub Actions workflows, the container image, the release artifacts, the conformance gate (`make accept`), and the reproducible-build contract.

---

## 1. Introduction

### 1.1 Motivation

A reproducible release pipeline turns "Abdel pressed go on his laptop" into "anyone who runs the workflow gets the same bytes." That property underpins the project's reproducibility goals (NFR-050, NFR-051) and the conformance gate (master spec §12).

### 1.2 Goals

1. Specify the GitHub Actions workflow matrix and each workflow's gates.
2. Specify the Dockerfile (multi-stage) producing the GHCR image used by HF Jobs.
3. Specify the release pipeline — tag → GitHub release → Hub artifacts.
4. Specify the reproducible-build contract.
5. Specify the lint and clippy rules at the CI boundary.

### 1.3 Non-goals

- Source-level lints — covered by [RFC 0001 §4.5/4.6](0001-project-foundation-and-build-system.md).
- Threat model — covered by [RFC 0016](0016-security-and-supply-chain.md).

---

## 2. Conventions

- `gha` — GitHub Actions.
- `runner` — a GitHub-hosted or self-hosted runner.
- "Green CI" — every required workflow passed on the PR.

---

## 3. Workflow matrix

```
.github/workflows/
├── ci.yml          # PR + push to main; the gate that blocks merge
├── release.yml     # on tag v*; produces release artifacts
├── specs.yml       # validates the spec set
├── nightly.yml     # daily; runs slow tests + dep drift checks
├── docs.yml        # builds and publishes rustdoc + paper
└── conformance.yml # weekly; runs the full conformance suite
```

### 3.1 `ci.yml` — the PR gate

**Trigger:** pull_request, push to `main`.

**Matrix:**

```yaml
strategy:
  fail-fast: false
  matrix:
    os: [ubuntu-22.04]
    rust: [1.85.0]   # pinned per RFC 0001
    feature_set:
      - default
      - cpu-only
      - parity-fixtures
    include:
      - os: macos-14-arm
        rust: 1.85.0
        feature_set: cpu-only
```

**Jobs:**

1. `fmt` — `cargo fmt --all -- --check`.
2. `clippy` — `cargo clippy --workspace --all-targets -- -D warnings`.
3. `build` — `cargo build --workspace --features ${feature_set}`.
4. `test` — `cargo test --workspace --features ${feature_set} -- --skip "_slow_"`.
5. `parity` — only on `parity-fixtures` build; runs `parity_*` tests. Requires the parity fixture and dumps; fetched from the `abdelstark/lewm-rs-parity-dumps` HF dataset using a CI secret token.
6. `deny` — `cargo deny check`.
7. `audit` — `cargo audit --deny warnings`.
8. `layers` — `python scripts/check_layers.py`.
9. `specs` — `python scripts/check_specs.py` (frontmatter, link integrity, traceability).
10. `docs-build` — `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`.
11. `python-edge` — `cd python && uv run pytest`.
12. `quickstart` — TST-0001-QS-001 (make test in ≤ 5 min on fresh checkout).
13. `coverage` — `cargo tarpaulin --out Xml`; uploaded to Codecov.

**RFC0011-001 [MUST]** — All jobs except `coverage` and `parity` are *required* to merge. `coverage` is informational; `parity` is required when the touched files include `lewm-core` or `lewm-train::convert`.

**RFC0011-002 [MUST]** — Total CI wall time on a green PR **MUST** be ≤ **15 minutes** on the default matrix entry.

### 3.2 `release.yml` — on tag

**Trigger:** tag matching `^v\d+\.\d+\.\d+$`.

**Jobs:**

1. `build-linux-static` — produce static-linked `lewm-train`, `lewm-eval`, `lewm-infer` for x86_64-unknown-linux-musl using `cargo zigbuild`. Profile: `release-lto`.
2. `build-macos-arm` — same three binaries for aarch64-apple-darwin.
3. `container` — multi-stage Dockerfile (§5); push to `ghcr.io/abdelstark/lewm-rs:<tag>` and `:latest`.
4. `release-notes` — generate from `CHANGELOG.md` and the merged-PR list since prior tag.
5. `github-release` — create a draft release; attach binaries; publish.
6. `hub-models` — re-tag the most recent Hub artifacts with the version.
7. `verify-reproducible` — re-runs `build-linux-static` in a clean container and diffs the binaries against the published ones.

**RFC0011-003 [MUST]** — Step 7 fails the release if the binary diff is non-empty (modulo timestamps stripped by `objcopy --remove-section=.note.gnu.build-id` and `cargo` strip flags). This is the realization of NFR-050.

### 3.3 `specs.yml` — spec set validator

**Trigger:** pull_request touching `specs/**` or `PRD.md` or `traceability-matrix.md`.

**Jobs:**

1. `check-frontmatter` — every RFC and ADR has valid YAML frontmatter.
2. `check-links` — `lychee --no-progress specs/**/*.md`.
3. `check-traceability` — `python scripts/check_specs.py --check-traceability` (PRD reqs ↔ FRs ↔ Tests).
4. `check-glossary` — every domain term used in any spec is defined in `glossary.md`.
5. `check-rfc-numbering` — no gaps, no duplicates.
6. `check-status-lifecycle` — `Accepted` RFCs reference only `Accepted` RFCs (transitive).

**RFC0011-004 [MUST]** — Any spec PR that fails any check is blocked from merge.

### 3.4 `nightly.yml`

**Trigger:** scheduled `0 6 * * *` UTC.

**Jobs:**

1. `slow-tests` — `cargo test --workspace --features slow-tests -- --include-ignored`.
2. `feature-powerset` — `cargo hack check --feature-powerset --workspace`.
3. `bench-baselines` — run `criterion` benches against committed baselines; alert on regression > 5 %.
4. `dep-update-preview` — `cargo update --dry-run` plus deny/audit; opens an automation PR if updates are clean.
5. `parity-against-latest-hub` — re-pull the reference checkpoint from HF and re-run parity. Catches upstream drift.

### 3.5 `docs.yml`

**Trigger:** push to `main`.

**Jobs:**

1. `rustdoc-build` — `cargo doc --workspace --no-deps`. Publish to GitHub Pages.
2. `paper-build` — render `paper/lewm-rs.md` to PDF via `pandoc` with `eisvogel` template; commit to Pages.
3. `spec-build` — render `specs/` to a static site (using `mdbook` or equivalent); publish to Pages.

### 3.6 `conformance.yml`

**Trigger:** scheduled `0 6 * * 0` (Sundays) or manual dispatch.

**Job:** `make accept` end-to-end, including:

- Build all crates with `release-lto`.
- Run full conformance suite (every TST-* ID).
- Verify the Hub artifacts exist with the right hashes (`TST-0010-ART-*`).
- Verify the cost ledger total ≤ 200 USD.
- Generate the conformance report.

**RFC0011-005 [MUST]** — `make accept` exit code 0 is the binding "shipped" signal.

---

## 4. CI hardware matrix

| Workflow | Hardware | Reason |
|----------|----------|--------|
| `ci.yml` | `ubuntu-22.04` (4 vCPU) | sufficient for build + tests + parity |
| `ci.yml` (parity job) | `ubuntu-22.04` | NdArray CPU only |
| `nightly.yml` (bench) | `ubuntu-22.04-gpu-l4` (self-hosted) | benches need a GPU |
| `release.yml` | `ubuntu-22.04` (binary builds), `ubuntu-22.04-arm` if needed | static cross compile |
| `conformance.yml` | `ubuntu-22.04` + L4 self-hosted | matches CI matrix; bench gates require GPU |

**RFC0011-006 [MUST]** — Self-hosted runners are tagged `lewm-rs-gpu` and scope-restricted to this repo (no fork PRs).

---

## 5. Container image

### 5.1 Dockerfile (multi-stage)

```dockerfile
# syntax=docker/dockerfile:1.7

# ── Stage 1: builder ─────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config curl ca-certificates \
    libssl-dev libhdf5-dev libstdc++-12-dev \
 && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.85.0 --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /src
COPY rust-toolchain.toml ./
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY scripts ./scripts

# Hot cache: dummy build then real
RUN --mount=type=cache,target=/root/.cargo \
    --mount=type=cache,target=/src/target \
    cargo build --release --workspace --bins

# ── Stage 2: runtime ─────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-103 libssl3 ca-certificates python3.11 python3-pip \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/target/release/lewm-train /usr/local/bin/lewm-train
COPY --from=builder /src/target/release/lewm-eval  /usr/local/bin/lewm-eval
COPY --from=builder /src/target/release/lewm-infer /usr/local/bin/lewm-infer
COPY python /opt/lewm-rs/python
COPY configs /opt/lewm-rs/configs

RUN pip install --no-cache-dir uv
WORKDIR /opt/lewm-rs
RUN uv pip install --system -r python/requirements.txt

ENV RUST_LOG=lewm=info,burn=warn \
    HF_HOME=/cache/hf \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["lewm-train"]
```

**RFC0011-007 [MUST]** — Image size in the runtime stage **MUST** be ≤ **800 MB**. Verified by CI.

**RFC0011-008 [MUST]** — Image is tagged with both `<git_sha>` and `<version>`. The `latest` tag follows the most recent release.

**RFC0011-009 [MUST]** — Image is signed with cosign (keyless OIDC) and the signature is published.

### 5.2 Vulnerability scanning

Trivy runs on every container build:

```yaml
- name: Trivy scan
  uses: aquasecurity/trivy-action@latest
  with:
    image-ref: ghcr.io/abdelstark/lewm-rs:${{ github.sha }}
    severity: HIGH,CRITICAL
    exit-code: 1
```

**RFC0011-010 [MUST]** — HIGH and CRITICAL vulnerabilities **MUST** block the release. The vulnerability list is logged with an issue auto-opened for triage.

---

## 6. Lint and clippy at the boundary

Repeated for emphasis from RFC 0001:

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

Workspace lints (Cargo.toml `[workspace.lints]`):

- `unsafe_code = "deny"`.
- `clippy::unwrap_used = "deny"`.
- `clippy::expect_used = "warn"` (allowed with explicit invariant comment).
- `clippy::dbg_macro = "deny"`.
- `clippy::print_stdout = "warn"`.
- `clippy::todo = "warn"`.
- `clippy::unimplemented = "deny"`.

Additional CI-specific:

- `RUSTFLAGS="-D warnings"` on build.
- `RUSTDOCFLAGS="-D warnings"` on docs.

---

## 7. Release pipeline

```
1. Author opens PR bumping `version` in `Cargo.toml` and updating CHANGELOG.md.
2. PR merges to main after CI green.
3. Author creates a signed tag `git tag -s v0.1.0 -m "release v0.1.0"`.
4. Push the tag: `git push origin v0.1.0`.
5. release.yml fires.
6. release.yml's `github-release` step publishes the GitHub release with:
   - lewm-train, lewm-eval, lewm-infer (linux x86_64-musl, macos arm64)
   - SHA-256 checksums
   - release notes from CHANGELOG.md
7. release.yml's `hub-models` step tags the most recent Hub artifacts with `v0.1.0`.
8. release.yml's `verify-reproducible` step rebuilds in a clean container and diffs.
```

### 7.1 Versioning

The crate workspace version (in `Cargo.toml`) and the spec set version (`specs/README.md`) are bumped together; they share the same SemVer.

**RFC0011-011 [MUST]** — `version` in `Cargo.toml` matches the most recent tag at every commit on `main`. Pre-release commits carry the next planned version with `-pre` suffix.

### 7.2 Changelog

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/) v1.1.0 conventions:

```markdown
# Changelog

## [Unreleased]

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security

## [0.1.0] - 2026-XX-XX

### Added
- Initial public release. Reproduces LeWorldModel PushT result in pure Rust.
```

---

## 8. Reproducible-build contract

**RFC0011-012 [MUST]** — Two builds of the same git SHA, with the pinned toolchain, on `ubuntu-22.04`, produce **byte-identical** `lewm-train`, `lewm-eval`, `lewm-infer` binaries. Verified by CI step `verify-reproducible`.

Techniques used:

- Toolchain pinned in `rust-toolchain.toml`.
- `SOURCE_DATE_EPOCH` set in CI environment.
- Strip flags: `RUSTFLAGS="-C strip=symbols"` for release.
- Build path normalization: `--remap-path-prefix=$(pwd)=/src`.
- No `cargo-vendor` indirection (`Cargo.lock` is the only source of dep versions).

---

## 9. Testing strategy (CI gates)

This RFC's own tests:

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0011-CI-001 | `ci_full_matrix_passes` | meta | each PR runs all required jobs |
| TST-0011-FMT-001 | `cargo_fmt_clean` | unit | RFC0001-008 |
| TST-0011-LINT-001 | `cargo_clippy_clean` | unit | RFC0011-006 |
| TST-0011-DOC-001 | `cargo_doc_clean` | unit | NFR-032 |
| TST-0011-LEAK-001 | `no_handle_leak_on_smoke` | integration | NFR-022 |
| TST-0011-REPRO-001 | `reproducible_release_binary` | integration | NFR-050 |
| TST-0011-IMG-001 | `image_size_under_800mb` | integration | RFC0011-007 |
| TST-0011-TRIVY-001 | `trivy_no_high_critical` | integration | RFC0011-010 |
| TST-0011-COSIGN-001 | `image_cosign_signature_valid` | integration | RFC0011-009 |
| TST-0011-SPEC-001 | `specs_yml_passes_on_initial_commit` | unit | §3.3 |
| TST-0011-TIME-001 | `ci_total_wall_under_15_min` | meta | RFC0011-002 |

---

## 10. Operational considerations

### 10.1 Observability

CI jobs emit telemetry via the GitHub Actions logs and a small `scripts/ci_metrics.py` that uploads job duration and outcome to a private HF dataset for trending.

### 10.2 Runbook

- **"CI takes > 15 minutes."** — check `Swatinem/rust-cache@v2` cache hit; cold runs are excused once per dep bump.
- **"Trivy reports a HIGH that we can't patch."** — file an ADR documenting acceptance; CI override is via a labelled "trivy-waiver" with a 60-day expiry.
- **"Release verify-reproducible diffs."** — usually a non-frozen timestamp; check `objcopy` strip args.

### 10.3 Capacity

GitHub-hosted runner minutes are budgeted at ~ 100 hours/month at the project's pace. Plenty within free tier.

---

## 11. Performance considerations

Build cache hit rate target: ≥ 80 % on PRs with no dep changes. Cold build cap: 5 minutes.

---

## 12. Security considerations

- Self-hosted runner scope-restricted.
- All secrets via OIDC where possible (e.g., AWS-style federated tokens for HF if available; otherwise repo secrets).
- Cosign keyless signing of the container.
- See [RFC 0016](0016-security-and-supply-chain.md).

---

## 13. Alternatives considered

- **A1 — Pulumi for infra.** Out of scope.
- **A2 — Nix for reproducible builds.** Considered. Adds learning curve; `rust-toolchain.toml` + `Cargo.lock` are sufficient for v1.
- **A3 — Buildkite / CircleCI.** Rejected: GitHub Actions is free and sufficient.

---

## 14. Acceptance criteria

- [ ] All workflows exist and pass on a fresh clone + commit + push to `main`.
- [ ] Container image builds and is pushed by release pipeline.
- [ ] `make accept` exit 0 on `conformance.yml`.
- [ ] Reproducible-build check passes on release.

---

## 15. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | GitHub Actions outage on release | L | M | Re-run; tags are immutable |
| R-2 | Trivy false positive blocks release | M | L | Waiver process documented |
| R-3 | CI flakes on parity test | L | M | Re-pull dumps with checksums |
| R-4 | Container image size creep | M | L | CI gate enforces 800 MB |

---

## 16. Open questions

OQ-2011-1 — Whether to maintain a separate `nightly-cuda-12.5` matrix entry to catch driver drift early. Defer to v2.

---

## 17. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0011.*
