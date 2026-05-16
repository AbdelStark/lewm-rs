# Workspace map

> **Motivation.** The `lewm-rs` Cargo workspace is eight crates with
> tight, deliberate boundaries. This page is the map: who depends on
> whom, what each one owns, and where to read source.
>
> **Position.** Top of [Part VIII — Workspace and crates](./workspace.md).
>
> **What you should leave with.** A clear picture of the dependency
> tree, the dependency invariant (INV-003), and a pointer per crate.

## 1. The eight crates

| Crate | Role | Lines | Spec |
|-------|------|------:|------|
| [`lewm-core`](./lewm-core.md) | Model architecture, losses, init, parity helpers, exports | ~3 k | [RFC 0002], [RFC 0003] |
| [`lewm-data`](./lewm-data.md) | Dataset loaders, window sampling, transforms, prefetch | ~1.5 k | [RFC 0004] |
| [`lewm-train`](./lewm-train.md) | Trainer, optimizer, schedule, checkpoint, resume, CLI | ~3 k | [RFC 0005] |
| [`lewm-plan`](./lewm-plan.md) | CEM planner, PushT eval, SO-100 eval, reports | ~1 k | [RFC 0006] |
| [`lewm-infer`](./lewm-infer.md) | ONNX/Tract runner, CPU CEM, parity eval CLI | ~2 k | [RFC 0007] |
| [`lewm-telemetry`](./lewm-telemetry.md) | JSONL emission, OTLP exporter | ~0.4 k | [RFC 0009] |
| [`lewm-hub`](./lewm-hub.md) | HF Hub upload helpers, model card | ~0.5 k | [RFC 0010] |
| [`lewm-gpu`](./lewm-gpu.md) | CUDA-specific helpers (feature-gated) | ~0.2 k | [RFC 0007], [RFC 0014] |

## 2. The dependency tree

```text
                       lewm-core (no deps on other workspace crates)
                          ▲      ▲      ▲      ▲      ▲
                          │      │      │      │      │
                       ┌──┴──┐ ┌─┴───┐ ┌┴────┐ ┌┴────┐ ┌┴────┐
                       │data │ │plan │ │train│ │infer│ │ gpu │
                       └─┬───┘ └─┬───┘ └──┬──┘ └──┬──┘ └─────┘
                         │       │        │       │
                         └───────┴────────┴───────┘
                                 │
                              telemetry
                                 │
                                 │     (telemetry is consumed by anyone
                                 │      that wants to emit metrics)
                                 │
                              ┌──┴──┐
                              │ hub │   (post-run upload; consumed by
                              └─────┘    train and CLI only)
```

The arrows mean "depends on", with arrowheads pointing up the
dependency tree. `lewm-core` is the foundation; everything else depends
on it. `lewm-infer` deliberately does *not* depend on `lewm-train`
(it has no need for training state) — see INV-003 below.

## 3. The dependency invariants

Three workspace-level invariants are pinned in [RFC 0001]:

- **INV-001 [MUST]** — Every workspace crate uses the same
  `Cargo.lock`. There is no per-crate lock file.
- **INV-002 [MUST]** — Every crate compiles cleanly under `cargo
  clippy -- -D warnings`. The workspace-wide clippy lint table is in
  the root `Cargo.toml` `[workspace.lints]`.
- **INV-003 [MUST]** — `lewm-infer` does **not** depend on
  `burn-autodiff`, `burn-cuda`, `burn-train`, or `lewm-train`. The
  inference binary must be small and minimal-dependency.

INV-003 is the most operationally important. It ensures the CPU
inference binary can be built without a CUDA toolkit and without
autograd, which makes the binary suitable for embedded / CPU-only
deployment.

## 4. Workspace-wide configuration

| File | Role |
|------|------|
| `Cargo.toml` (root) | Workspace members, profiles, shared dep versions, lint table. |
| `rust-toolchain.toml` | Pinned Rust toolchain (currently 1.89.0). |
| `clippy.toml` | Clippy configuration overrides. |
| `rustfmt.toml` | Rustfmt configuration. |
| `deny.toml` | `cargo-deny` policy for licenses, advisories, sources. |
| `.gitleaks.toml` | Secret-scan configuration. |
| `Makefile` | Local-developer gates (`make check`, `make accept`). |

The pinned toolchain version is the single source of truth for what
"a build works" means. Bumping it requires an ADR.

## 5. The binaries

The workspace produces three binaries:

| Binary | Crate | Purpose |
|--------|-------|---------|
| `lewm-train` | `lewm-train` | Run training: smoke, parity, train, eval, convert. |
| `lewm-eval` | `lewm-plan` | Run eval on a checkpoint. |
| `lewm-infer` | `lewm-infer` | CPU inference: bench, plan, eval (parity vs dumps), demo bridge. |

Each is a standalone, statically-linked binary in release mode. The
Docker image (`ghcr.io/abdelstark/lewm-rs:latest`) ships `lewm-train`
plus its Python helpers; the inference binary is built separately for
laptop deployment.

## 6. Where to read

For each crate, the relevant page in this section gives:

- A one-paragraph "what it owns" summary.
- The public API surface.
- A pointer to the spec RFC.
- A pointer to the source.

Start with [`lewm-core`](./lewm-core.md), which everything else
depends on.

[RFC 0001]: ../reference/rfcs.md
[RFC 0002]: ../reference/rfcs.md
[RFC 0003]: ../reference/rfcs.md
[RFC 0004]: ../reference/rfcs.md
[RFC 0005]: ../reference/rfcs.md
[RFC 0006]: ../reference/rfcs.md
[RFC 0007]: ../reference/rfcs.md
[RFC 0009]: ../reference/rfcs.md
[RFC 0010]: ../reference/rfcs.md
[RFC 0014]: ../reference/rfcs.md
