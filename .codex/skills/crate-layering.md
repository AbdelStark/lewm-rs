---
name: crate-layering
description: Enforce the lewm-rs workspace's strict inter-crate dependency layering. Activate when adding/modifying any `[dependencies]` block in a `crates/*/Cargo.toml`, when `python3 scripts/check_layers.py` fails, or when refactoring crate boundaries. The layering is binding per RFC 0001 §4 and `scripts/check_layers.py` is run inside `make check`.
prerequisites: Python 3.13+ on PATH, write access to `crates/`
---

# Crate Layering

<purpose>
The 8-crate workspace has a one-way dep graph. Adding `lewm-X = { workspace = true }` to a crate that isn't allowed to consume it breaks `make check`. This skill encodes the allowlist and the patterns to use when "I need symbol Y in crate X."
</purpose>

<context>
The allowlist (mirror of `scripts/check_layers.py::ALLOWED_DEPS`):

| Crate            | May depend on                                                 |
|------------------|---------------------------------------------------------------|
| `lewm-core`      | (none)                                                        |
| `lewm-data`      | `lewm-core`                                                   |
| `lewm-hub`       | `lewm-core`                                                   |
| `lewm-telemetry` | `lewm-core`                                                   |
| `lewm-plan`      | `lewm-core`, `lewm-data`, `lewm-telemetry`                    |
| `lewm-train`     | `lewm-core`, `lewm-data`, `lewm-telemetry`, `lewm-hub`, `lewm-plan`     |
| `lewm-infer`     | `lewm-core`, `lewm-telemetry` (NO `burn-cuda`, `burn-autodiff`, `nvml-wrapper`) |
| `lewm-gpu`       | `lewm-core`, `lewm-infer` (the ONLY crate that may pull `burn-cuda`)    |

Reverse rule (key): `lewm-core` depends on nothing in this workspace. `lewm-infer` MUST stay CUDA-free so it builds on minimal CI runners and edge hosts. CUDA glue lives exclusively in `lewm-gpu`.

`scripts/check_layers.py` also pins:
- The set of workspace `members` (8 crates listed in `EXPECTED_MEMBERS`).
- The forbidden-direct-dep list for `lewm-infer` (`burn-cuda`, `burn-autodiff`, `nvml-wrapper`).
</context>

<procedure>
1. Identify the symbol you want to share and which crates need it.
2. Locate it on the dep graph: does the consuming crate already have a path to the producing crate?
   - **YES** → add `producing-crate = { workspace = true }` to the consumer's `[dependencies]`, then import normally. Run `python3 scripts/check_layers.py`.
   - **NO** → see step 3.
3. The dep would violate the allowlist. Pick the right resolution:
   - If the symbol is **algorithmic / pure / data-only** (e.g. a config struct, a math kernel), move it DOWN to `lewm-core`. This is usually correct.
   - If the symbol is **infrastructure** (telemetry, hub upload), move it INTO the appropriate sibling crate (`lewm-telemetry`, `lewm-hub`) and let both consumers pull it.
   - If the consumer is `lewm-infer` and the symbol needs CUDA, the symbol does NOT belong in `lewm-infer`. Implement it in `lewm-gpu` (or behind a feature flag in a crate `lewm-gpu` consumes).
4. After any move, run from repo root:
   ```
   cargo check --workspace --all-targets
   python3 scripts/check_layers.py
   cargo clippy --workspace --all-targets -- -D warnings
   ```
5. If `lewm-train`'s allowlist needs widening, update both `scripts/check_layers.py::ALLOWED_DEPS` AND RFC 0001 §4 in the same PR.
</procedure>

<patterns>
<do>
— Express new ML-shape primitives in `lewm-core::tensor_ops` (or a new module under `lewm-core`).
— Add new optional backends behind a Cargo feature in the existing crate before spawning a new crate.
— When adding a CUDA-only helper, place it in `lewm-gpu`, export via `pub fn`, and call from downstream binaries (not from `lewm-infer` lib code).
</do>
<dont>
— Don't add `burn-cuda` or `burn-autodiff` to `lewm-infer`'s `Cargo.toml`, even behind a feature flag. The script greps the raw deps section.
— Don't relax the allowlist to "make the test pass." The allowlist IS the contract.
— Don't introduce a new `lewm-*` crate without an RFC update and CI wiring (workflow matrix, `EXPECTED_MEMBERS`).
</dont>
</patterns>

<examples>
Example: "I need `Jepa<B>` available to `lewm-infer`'s eval runner."

`Jepa<B>` lives in `lewm-core::jepa`. `lewm-infer → lewm-core` is already allowed. Solution:

```toml
# crates/lewm-infer/Cargo.toml
[dependencies]
lewm-core = { workspace = true }
```

```rust
// crates/lewm-infer/src/runner/burn.rs
use lewm_core::jepa::Jepa;
```

Run `python3 scripts/check_layers.py` — should print no diagnostics.

Counter-example: "I want to call a `burn-cuda` kernel from `lewm-infer::runner`."

This breaks RFC 0007 and the layer check. Instead: put the kernel call in `lewm-gpu::load_cuda_runner`, expose a backend-erased trait in `lewm-infer`, and have the downstream binary wire `lewm-gpu` to the trait.
</examples>

<troubleshooting>
| Symptom                                                   | Cause                                                  | Fix                                                                |
|-----------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------------------|
| `check_layers.py: crate X depends on Y (forbidden)`        | Direct workspace dep outside allowlist                 | Remove the dep; relocate the symbol per step 3 above               |
| `lewm-infer forbidden direct dependency: burn-cuda`        | CUDA dep leaked into `lewm-infer`                      | Move to `lewm-gpu`; consume via trait + downstream binary           |
| `lewm-train allowed deps mismatch`                          | Added a crate to `lewm-train` without script update    | Update `ALLOWED_DEPS["lewm-train"]` AND RFC 0001 §4 in same PR     |
| Workspace `members` count drift                             | Added/removed a crate                                  | Update `EXPECTED_MEMBERS` and `[workspace] members = [...]`        |
</troubleshooting>

<references>
- `scripts/check_layers.py` — enforcement
- `specs/rfcs/0001-project-foundation-and-build-system.md` — workspace contract
- `specs/rfcs/0007-tract-inference-and-onnx-export.md` — CUDA-free `lewm-infer` rationale
- `crates/lewm-gpu/Cargo.toml` — the only crate that may depend on `burn-cuda`
</references>
