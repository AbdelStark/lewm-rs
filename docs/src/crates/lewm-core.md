# `lewm-core`

The model, the losses, the init recipes, the safetensors export, and the
parity helpers. Everything else in the workspace depends on this crate.

## What it owns

- **Module definitions**: `Jepa<B>`, `Vit<B>`, `ArPredictor<B>`,
  `Embedder<B>`, `Mlp<B>`, plus their building blocks.
- **Loss functions**: `prediction_loss`, `sigreg_loss`,
  `collapse_probes`.
- **Initialization helpers**: truncated normal, zero, AdaLN-zero,
  position embeddings.
- **Tensor ops**: GELU variants (Erf, TanhApprox), bilinear/bicubic
  pos-emb interpolation, masking helpers.
- **Configs**: `JepaConfig`, `VitConfig`, `PredictorConfig`,
  `EmbedderConfig`, `MlpConfig`.
- **Export**: safetensors writer, parameter-name mapper.
- **Import**: safetensors reader matching `python/param_name_map.py`.
- **RNG tree**: named substream system pinned by [RFC 0013].
- **Error types**: `LewmCoreError` enum.

## Module layout

```text
lewm-core/src/
├── lib.rs               # re-exports, error type
├── config.rs            # all *Config structs
├── init.rs              # truncated normal, zero init, AdaLN-zero init
├── tensor_ops.rs        # GELU, masking, pos-emb interpolation
├── vit.rs               # PatchEmbed, Attention, MlpBlock, EncoderBlock, Vit
├── embedder.rs          # ActionSmoother, Embedder
├── mlp.rs               # Mlp (projector / pred_proj)
├── ada_ln.rs            # AdaLnZero helper
├── predictor.rs         # ConditionalBlock, ArPredictor
├── jepa.rs              # Jepa top-level wrapper
├── losses/
│   ├── mod.rs
│   ├── prediction.rs    # MSE loss
│   ├── sigreg.rs        # SIGReg (F32 island)
│   └── collapse_probes.rs # TOL-007/008/009
├── rng.rs               # named substream tree
├── export.rs            # safetensors writer
├── import.rs            # safetensors reader
└── errors.rs            # LewmCoreError
```

## Public API surface

```rust,ignore
// In lewm_core (re-exported from lib.rs):
pub use crate::config::{
    EmbedderConfig, JepaConfig, MlpConfig, PredictorConfig, VitConfig, GeluVariant,
};
pub use crate::embedder::Embedder;
pub use crate::jepa::Jepa;
pub use crate::mlp::Mlp;
pub use crate::predictor::{ArPredictor, ConditionalBlock};
pub use crate::vit::{EncoderBlock, PatchEmbed, Vit};
pub use crate::losses::{prediction_loss, sigreg_loss, collapse_probes};
pub use crate::errors::LewmCoreError;
```

## Parity test suite

| Test | What it pins |
|------|--------------|
| `parity_encoder_cls.rs` | encoder CLS, F32 |
| `parity_encoder_all.rs` | encoder all-tokens, F32 |
| `parity_encoder_mixed_precision.rs` | encoder BF16 vs F32 |
| `parity_action_encoder.rs` | action encoder |
| `parity_predictor.rs` | predictor, F32 |
| `parity_predictor_mixed_precision.rs` | predictor BF16 vs F32 |
| `parity_projector.rs` | projector MLP |
| `parity_pred_proj.rs` | pred-proj MLP |
| `parity_sigreg.rs` | SIGReg scalar, seeded |
| `parity_sigreg_seedfree.rs` | SIGReg scalar, fresh sketch |

Gated behind feature `parity-fixtures` plus env vars
`LEWM_REFERENCE_SAFETENSORS` and `LEWM_PARITY_DUMPS`. See
[Parity tests](../parity/tests.md).

## Dependencies

- `burn` (= 0.21.0)
- `burn-ndarray` (CPU backend, for tests)
- `safetensors`
- `thiserror`
- (no other workspace crates)

## Source

[`crates/lewm-core`](https://github.com/AbdelStark/lewm-rs/tree/main/crates/lewm-core)

[RFC 0013]: ../reference/rfcs.md
