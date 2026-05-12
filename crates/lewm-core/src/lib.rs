//! Core model architecture, loss functions, initialization helpers, and tensor
//! contracts for the Rust `LeWM` implementation. This crate is intentionally free
//! of data loading, training orchestration, telemetry export, and inference
//! runner concerns; see [RFC 0002] and [RFC 0003] for the locked model and loss
//! contracts.
//!
//! [RFC 0002]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0002-core-model-architecture.md
//! [RFC 0003]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0003-sigreg-and-loss-functions.md
//!
//! ## Module index
//!
//! The public module surface is added by the phase-specific implementation
//! issues after the workspace scaffold lands.

pub mod config;
pub mod errors;
pub mod init;
pub mod tensor_ops;

pub use crate::config::{
    EmbedderConfig, GeluVariant, JepaConfig, MlpConfig, NormVariant, PredictorConfig, VitConfig,
    VitSize,
};
pub use crate::errors::LewmCoreError;
pub use crate::init::{
    InitTensor, MODEL_INIT_STREAM, ModelInitRng, model_init_rng, ones, substream_rng,
    substream_seed, trunc_normal, zeros,
};
pub use crate::tensor_ops::{
    BICUBIC_ALIGN_CORNERS, CausalMask, DeviceKey, PositionEmbedding, build_causal_mask, gelu_erf,
    gelu_tanh_approx, interpolate_pos_embed,
};
