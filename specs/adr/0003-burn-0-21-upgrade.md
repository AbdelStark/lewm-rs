---
adr: "0003"
title: "Burn 0.21 upgrade and Tract stability"
status: Implemented
date: 2026-05-17
authors: ["Abdel"]
tracks_rfc: ["0001", "0002", "0007", "0016"]
supersedes: []
superseded_by: null
pr: null
---

# ADR 0003 — Burn 0.21 Upgrade And Tract Stability

## Context

ADR 0002 froze the Burn parity stack at `=0.20.1` to land the first
Burn-backed `lewm-core` slice on a documented compiler contract. Burn
`0.21.0` was released and stabilises the API surface relevant to this
project: `Tensor`, `Module` derive, `Param`, `RunningState`, `Recorder`,
`AdamW`, `Linear`, `LayerNorm`, `Dropout`, `gelu`, `softmax`, and the
`backend::Backend` trait shape used by every numerics file in
`lewm-core`. Rust `1.95.0` (already pinned in `rust-toolchain.toml`)
clears Burn 0.21's new `rust-version = "1.92"` floor with margin.

Tract is already on its latest stable line: `=0.22.1`. Tract `0.23.0`
exists only as `0.23.0-dev.5` (a pre-release with one yanked sibling).
Pulling a dev release into the inference path would violate the
`tract = "=X"` exact-pin policy in RFC 0007 and weaken the
release-train guarantees in RFC 0016. We therefore leave Tract pinned
at `=0.22.1` until upstream cuts a stable `0.23.0`.

## Decision

Bump every Burn workspace dep (`burn`, `burn-core`, `burn-cuda`,
`burn-ndarray`, `burn-autodiff`, `burn-import`, `burn-train`) from
`=0.20.1` to `=0.21.0`. Keep Tract at `=0.22.1`. Keep
`rust-toolchain.toml = "1.95.0"`. The `bincode 2.0.1` transitive
advisory (`RUSTSEC-2025-0141`) waiver from ADR 0002 still applies —
Burn 0.21 still pulls `bincode 2.0.1` through `burn-core`.

## Alternatives considered

- **Stay on Burn `=0.20.1`** — rejected. The previous pin was a
  scope-limiting decision for the first slice; the v1 architecture is
  now landed and the upgrade unblocks future autodiff/Recorder fixes
  that ship in `0.21.x`.
- **Move to Tract `0.23.0-dev.5`** — rejected. Dev pre-releases
  violate the `=` exact-pin RFC 0007 policy, and the alphabetical
  sibling `0.23.0-dev.1` is yanked, signalling churn.
- **Migrate `Ignored<T>` later** — rejected. Burn 0.21 deprecates the
  wrapper and the project enforces `-D warnings`, so the migration is
  required at upgrade time.

## Consequences

### Positive

- New Burn 0.21 features available (linear/conv fusion, expanded
  signal/loss libraries, `BackendTypes` alias) for future work.
- Code base sheds the deprecated `Ignored<T>` wrapper in favour of the
  recommended `#[module(skip)]` attribute; all five wrapper sites in
  `lewm-core` (`vit`, `mlp`, `jepa`, `predictor`, `losses/sigreg`)
  are migrated.
- `B::ad_enabled()` callers updated to the new `(&device)` signature
  required by 0.21's `Backend` trait.

### Negative

- `Module` derive output changes for any struct that used
  `Ignored<T>` vs `#[module(skip)]`. Both bypass serialization, but
  any externally-archived checkpoint that round-tripped the deprecated
  field metadata will need re-conversion. We have no such fixtures in
  the repo (only the safetensors export path is normative for
  cross-version interop, and it is unaffected).
- `cubecl-zspace` `Shape::dims` is now a method, not a field. Two
  call sites (`lewm-core/src/import.rs`, `lewm-train/src/bin/
  lewm-reference-record.rs`) migrated to `Shape::as_slice()`.

### Neutral or to revisit

- The `RUSTSEC-2025-0141` `bincode 2.0.1` waiver from ADR 0002 still
  applies. Revisit when upstream Burn cuts a release that drops the
  unmaintained crate.
- Re-evaluate Tract when `0.23.0` is published as a stable release.

## Implementation

Implemented in the same PR that updates `Cargo.toml`,
`crates/lewm-core/src/{mlp,vit,jepa,predictor,losses/sigreg,import}.rs`,
`crates/lewm-train/src/{eval.rs,bin/lewm-reference-record.rs}`,
and the corresponding documentation lines (root project-instructions
stack table, RFC 0001 / 0002 / 0007 / 0016 version references). The 212-test
workspace lib suite, clippy under `-D warnings`, spec / layer /
jobs / nondet validators, the Python validators, and the cost
ledger all pass after the migration.

## References

- ADR 0002 — Burn compiler floor and audit waiver
- RFC 0001 — Project foundation and build system
- RFC 0002 — Core model architecture
- RFC 0007 — Tract inference and ONNX export
- RFC 0016 — Security and supply chain
- Burn 0.21.0 release notes: <https://github.com/tracel-ai/burn/releases/tag/v0.21.0>

---

*End of ADR 0003.*
