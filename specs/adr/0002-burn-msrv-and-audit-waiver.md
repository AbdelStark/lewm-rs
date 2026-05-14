---
adr: "0002"
title: "Burn compiler floor and audit waiver"
status: Implemented
date: 2026-05-14
authors: ["Abdel"]
tracks_rfc: ["0001", "0011", "0016"]
supersedes: []
superseded_by: null
pr: null
---

# ADR 0002 — Burn Compiler Floor And Audit Waiver

## Context

The Burn parity stack is pinned to Burn `0.20.1` in RFC 0001. Adding Burn as a
direct `lewm-core` dependency fails under the previous Rust `1.85` repo
contract because Burn `0.20.1` requires Rust `1.89`. Burn `0.21` is not an
immediate substitute because it raises the compiler floor again to Rust `1.92`
and would expand the parity surface while the v1 architecture is still being
landed.

Enabling Burn `0.20.1` also brings `bincode 2.0.1` through `burn-core`, which
currently has `RUSTSEC-2025-0141` as an unmaintained advisory.

## Decision

Bump the repo toolchain contract to Rust `1.89.0` and keep Burn pinned at
`0.20.1` for the first Burn-backed `lewm-core` implementation slice.

Add a scoped `cargo audit` waiver for `RUSTSEC-2025-0141` while it is only
pulled transitively through Burn. Revisit this waiver no later than
2026-06-30, and sooner if Burn publishes a compatible `0.20.x` fix or the
project intentionally moves to a newer Burn/Rust pair.

## Alternatives considered

- **Stay on Rust 1.85** — rejected because it prevents compiling the pinned Burn
  version and blocks issues #26-#33.
- **Jump to Burn 0.21 and Rust 1.92** — rejected for this slice because it
  changes both the compiler and Burn API surface at the same time.
- **Avoid a direct Burn dependency in `lewm-core`** — rejected because it keeps
  the same blocker hidden until the real module implementation starts.

## Consequences

### Positive

- Burn module work can proceed on a documented compiler contract.
- CI, Docker, RFC examples, and local toolchains now agree on one Rust version.
- The transitive audit waiver is explicit and date-bounded.

### Negative

- Contributors need Rust `1.89.0` installed.
- The lockfile now includes Burn's CPU backend dependency tree.

### Neutral or to revisit

- The bincode advisory remains an upstream Burn dependency concern until the
  waiver is removed or Burn is upgraded.

## Implementation

Implemented by the PR that bumps `rust-toolchain.toml`, CI `RUST_TOOLCHAIN`,
Docker builder image, workspace `rust-version`, and adds a direct Burn compile
smoke in `lewm-core`.

## References

- RFC 0001
- RFC 0011
- RFC 0016
- `RUSTSEC-2025-0141`

---

*End of ADR 0002.*
