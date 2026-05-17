# `lewm-core`

Core model architecture, loss functions, initialization helpers, and tensor
contracts for the Rust `LeWM` implementation. This crate is intentionally free
of data loading, training orchestration, telemetry export, and inference runner
concerns.

**Specs:** [RFC 0002 — core model architecture][rfc-0002],
[RFC 0003 — SIGReg and losses][rfc-0003],
[RFC 0008 — reference parity testing][rfc-0008],
[RFC 0013 — determinism and reproducibility][rfc-0013].

**Layering:** `lewm-core` is the workspace root. No other `lewm-*` crate may be
imported here; this is enforced by [`scripts/check_layers.py`][layers].

## Module map

| Module        | Responsibility                                                                           |
| ------------- | ---------------------------------------------------------------------------------------- |
| `ada_ln`      | AdaLN-zero conditioning layer used in the predictor.                                     |
| `config`      | Locked architectural configs (ViT-Tiny, embedder, predictor).                            |
| `embedder`    | Action embedder MLP with Conv1d-k1 smoothing.                                            |
| `errors`      | Crate error type (`LewmCoreError`).                                                      |
| `export`      | Deterministic Safetensors export for `Jepa<B>` modules.                                  |
| `import`      | Reference-checkpoint Safetensors loader with parity-preserving name map.                 |
| `init`        | Tensor initialization helpers, RNG-stream keyed per RFC 0013.                            |
| `jepa`        | Top-level `Jepa<B>` module: encoder + projector + predictor + pred-proj.                 |
| `losses`      | SIGReg sketch-based regularizer, prediction loss, collapse probe.                        |
| `mlp`         | Reusable two-layer MLP block (used by embedder, projector, pred-proj).                   |
| `predictor`   | Autoregressive latent predictor with AdaLN-zero blocks.                                  |
| `rng`         | Substream-keyed deterministic RNG.                                                       |
| `tensor_ops`  | Activation kernels, causal mask, positional embedding interpolation.                     |
| `vit`         | ViT-Tiny encoder with parity-validated forward path.                                     |

## Features

- `parity-fixtures` (default off): enables the activation-level parity test
  suite (`tests/parity_*.rs`). The tests skip gracefully when the
  `LEWM_PARITY_DUMPS` / `LEWM_REFERENCE_SAFETENSORS` environment variables are
  unset; the CI workflow caches and provides the dumps when `HF_TOKEN` is
  available.
- `slow-tests` (default off): opt-in flag for fixture-heavy tests that are
  excluded from `make test-fast`.

## Numerical contracts

The crate ships ten activation-level parity tests against the published
`quentinll/lewm-pusht` reference checkpoint:

| Stage           | Tolerance         | Test file                                  |
| --------------- | ----------------- | ------------------------------------------ |
| Encoder         | `L∞ < 1e-4`       | [`tests/parity_encoder.rs`][parity-enc]    |
| Action encoder  | `L∞ < 1e-4`       | [`tests/parity_action_encoder.rs`][parity-ae] |
| Predictor       | `L∞ < 1e-4`       | [`tests/parity_predictor.rs`][parity-pred] |
| `pred_proj`     | `L∞ < 1e-4`       | [`tests/parity_pred_proj.rs`][parity-pp]   |
| SIGReg          | `|Δ| < 1e-3`      | [`tests/parity_sigreg.rs`][parity-sigreg]  |
| Init shapes     | exact             | [`tests/parity_init.rs`][parity-init]      |

LayerNorm uses `eps = 1e-12` and GELU is the exact-erf variant, matching the
reference implementation byte-for-byte.

## Examples

```rust,no_run
use burn::backend::NdArray;
use lewm_core::{Jepa, JepaConfig};

let config = JepaConfig::default();
let device = Default::default();
let model: Jepa<NdArray> = Jepa::new(&config, &device);
```

See the full encoder/predictor/SIGReg flow in
[`tests/burn_compat.rs`][burn-compat] and the `crates/lewm-train` trainer for
real usage.

[rfc-0002]: ../../specs/rfcs/0002-core-model-architecture.md
[rfc-0003]: ../../specs/rfcs/0003-sigreg-and-loss-functions.md
[rfc-0008]: ../../specs/rfcs/0008-reference-parity-testing.md
[rfc-0013]: ../../specs/rfcs/0013-determinism-and-reproducibility.md
[layers]: ../../scripts/check_layers.py
[parity-enc]: tests/parity_encoder.rs
[parity-ae]: tests/parity_action_encoder.rs
[parity-pred]: tests/parity_predictor.rs
[parity-pp]: tests/parity_pred_proj.rs
[parity-sigreg]: tests/parity_sigreg.rs
[parity-init]: tests/parity_init.rs
[burn-compat]: tests/burn_compat.rs
