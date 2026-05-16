---
name: burn-modules
description: Conventions for writing `burn::module::Module` types in lewm-rs. Activate when adding a new module, modifying tensor-shape contracts, or porting a PyTorch layer into Burn. Pinned to Burn `=0.20.1`; deviations from this version need an ADR. Modules in `lewm-core` are parity-tested — see `parity-testing.md` before changing numerics.
prerequisites: Read `crates/lewm-core/src/jepa.rs` and `crates/lewm-core/src/vit.rs` for canonical examples
---

# Burn Module Patterns

<purpose>
The Burn modules in `lewm-core` (`vit`, `predictor`, `ada_ln`, `mlp`, `embedder`, `jepa`) are the numerical core of the project. They must be backend-generic over `B: Backend`, parity-preserving, and deterministically initialized. This skill documents the conventions used throughout the crate.
</purpose>

<context>
- Burn dependency is workspace-pinned at `=0.20.1`. All `burn-*` sub-crates share the same pin.
- Backends used:
  - `burn_ndarray::NdArray<f32>` — CPU; CI, smoke, parity tests, `lewm-infer --backend burn-cpu`.
  - `burn_cuda::Cuda<f32>` / `Cuda<bf16>` — training (`lewm-train`, default feature `cuda`) and `lewm-gpu`.
  - `burn_autodiff::Autodiff<B>` — wraps a backend for reverse-mode AD during training.
- All public modules in `lewm-core` are `pub struct X<B: Backend>` with `#[derive(Module, Debug)]`.
- Tensor shape conventions documented at module level in rustdoc per RFC 0015. Use `[batch, …]` ordering.
- Init: every learnable parameter goes through `lewm_core::init` helpers (RFC 0013) — never call `Tensor::random` with an ad-hoc seed.
- Determinism: take RNG sub-streams via `lewm_core::rng::SubStream` keys, not `thread_rng` (lint-banned).
</context>

<procedure>
1. **Define the module** in its own file under `crates/lewm-core/src/`. Mirror the existing pattern:
   ```rust
   #[derive(Module, Debug)]
   pub struct MyHead<B: Backend> {
       linear: nn::Linear<B>,
       norm: nn::LayerNorm<B>,
   }
   ```

2. **Add a `Config` companion** with `validator::Validate` impl. Configs live in `crates/lewm-core/src/config.rs` if architecturally significant.

3. **Implement `forward`** with explicit `Tensor<B, D>` types and an inline shape comment block at the top of the function:
   ```rust
   /// Shape contract:
   /// - input `x`: `[batch, tokens, dim]`
   /// - output:    `[batch, tokens, dim_out]`
   pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> { … }
   ```

4. **Initialize parameters** through `lewm_core::init`. For new layer types, add an init helper there — do not inline `Tensor::random_…`.

5. **Add shape tests** under `crates/lewm-core/tests/<name>_shape.rs` using `NdArray<f32>`. Cover at least: minimal batch, expected input rank, edge case (`batch=1`).

6. **If the module is on the parity path**, add an entry to `python/param_name_map.py` (PyTorch → Burn name mapping) and confirm `lewm_core::import` handles it. Then run the parity tests per `parity-testing.md`.

7. **Export support**: if the module ships as a checkpoint, confirm `lewm_core::export::to_safetensors` emits deterministic bytes (use `cargo test -p lewm-core export`).

8. **Doc**: rustdoc with the shape contract, errors (if any), and an example. `missing_docs = "warn"` makes this a hard requirement for public items.
</procedure>

<patterns>
<do>
— Keep modules backend-generic: every function on a `pub struct X<B: Backend>` works for `B = NdArray<f32>`, `Cuda<f32>`, `Cuda<bf16>`, and `Autodiff<B>`.
— Use `nn::LayerNorm` with `eps = 1e-12` to preserve parity.
— Use `tensor_ops::exact_gelu` (or whatever the crate exposes for exact-erf GELU) — NOT `nn::Gelu::Approximate`.
— Return `Tensor<B, D>` from `forward`; don't unwrap to vectors except at crate boundaries.
— For optional sub-modules, prefer `Option<SubModule<B>>` over feature-flagged compile-time absence.
</do>
<dont>
— Don't add `#[cfg(feature = "cuda")]` inside `lewm-core`. Backend selection happens at the consumer (`lewm-train`, `lewm-gpu`). `lewm-core` stays backend-agnostic.
— Don't call `panic!` / `unwrap()` / `expect()` in module code. Bubble through `LewmCoreError` (`crates/lewm-core/src/errors.rs`).
— Don't `pub mod` a new file without re-exporting (or deliberately keeping it crate-private) — `unreachable_pub = "warn"` flags the inconsistency.
— Don't change parameter names that appear in `python/param_name_map.py` without updating the map AND `crates/lewm-core/src/import.rs`. Both must move together to preserve weight loading.
</dont>
</patterns>

<examples>
Example — a two-layer MLP head, parity-friendly:

```rust
#[derive(Module, Debug)]
pub struct MlpHead<B: Backend> {
    fc1: nn::Linear<B>,
    bn: nn::BatchNorm<B, 1>, // feature-axis BatchNorm1d, per RFC 0002
    fc2: nn::Linear<B>,
}

impl<B: Backend> MlpHead<B> {
    /// Shape contract:
    /// - input  `x`: `[batch, dim_in]`
    /// - output:    `[batch, dim_out]`
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.fc1.forward(x);
        let h = self.bn.forward(h);
        let h = crate::tensor_ops::exact_gelu(h);
        self.fc2.forward(h)
    }
}
```
</examples>

<troubleshooting>
| Symptom                                              | Cause                                               | Fix                                                                         |
|------------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------------------------------|
| `the trait bound … : Module is not satisfied`        | Field type lacks `Module` derive                     | Ensure every field is a Burn type or a `Module`-deriving struct              |
| Shape mismatch at runtime                            | Off-by-one in rank generic                           | Audit each `Tensor<B, D>`; add a shape test                                  |
| `cargo build --no-default-features --features cpu-only` fails on macOS | Inadvertent CUDA import       | Remove the import; backend-select at consumer site                           |
| Parity drift after refactor                          | Activation order changed                             | Diff `forward` against the previous version op-by-op                         |
| `missing_docs` warning                               | Public item without `///` doc                        | Add a doc comment with shape + errors + (if useful) one-line example         |
</troubleshooting>

<references>
- `crates/lewm-core/src/jepa.rs` — top-level wrapper composing encoder + projector + predictor + pred-proj
- `crates/lewm-core/src/vit.rs` — ViT-Tiny encoder reference
- `crates/lewm-core/src/predictor.rs` — `ConditionalBlock` + `ArPredictor`
- `crates/lewm-core/src/tensor_ops.rs` — activation kernels, causal mask helpers
- `crates/lewm-core/src/init.rs` — RFC-0013 initialization helpers
- `specs/rfcs/0002-core-model-architecture.md` — shape and dimension contracts
</references>
