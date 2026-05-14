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

pub mod ada_ln;
pub mod config;
pub mod embedder;
pub mod errors;
pub mod export;
pub mod init;
pub mod jepa;
pub mod losses;
pub mod mlp;
pub mod predictor;
pub mod rng;
pub mod tensor_ops;
pub mod vit;

pub use crate::ada_ln::{AdaLNZero, AdaLNZeroOutputs};
pub use crate::config::{
    EmbedderConfig, GeluVariant, JepaConfig, MlpConfig, NormVariant, PredictorConfig, VitConfig,
    VitSize,
};
pub use crate::embedder::Embedder;
pub use crate::errors::LewmCoreError;
pub use crate::init::{InitTensor, ModelInitRng, model_init_rng, ones, trunc_normal, zeros};
pub use crate::jepa::{Jepa, JepaLosses};
pub use crate::losses::{
    CLS_COSINE_PAIR_CEILING, CLS_MEAN_ABS_CEILING, CLS_VAR_FLOOR, CollapseProbe,
    CollapseProbeResult, CollapseThresholds, CollapseTrip, DEFAULT_SIGREG_KNOTS,
    DEFAULT_SIGREG_NUM_PROJ, DEFAULT_SIGREG_T_MAX, SigReg, SigRegConsts, prediction_loss,
    run_collapse_probe, run_collapse_probe_with_thresholds, sample_sigreg_projection,
};
pub use crate::mlp::Mlp;
pub use crate::predictor::{ArPredictor, ConditionalBlock};
pub use crate::rng::{
    CEM_STREAM, DATA_SHUFFLE_STREAM, DROPOUT_STREAM, MISC_STREAM, MODEL_INIT_STREAM,
    RFC_0013_STREAMS, RNG_STATE_BYTES, RngState, SIGREG_SKETCH_STREAM, deserialize_rng,
    is_registered_substream, rng_state, serialize_rng, substream_rng, substream_seed,
};
pub use crate::tensor_ops::{
    BICUBIC_ALIGN_CORNERS, CausalMask, DeviceKey, PositionEmbedding, build_causal_mask, gelu_erf,
    gelu_tanh_approx, interpolate_pos_embed,
};
pub use crate::vit::{
    Attention, EncoderBlock, MlpBlock, PatchEmbed, ViTEmbeddings, ViTOutput, Vit,
};
