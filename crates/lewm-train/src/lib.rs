//! Training orchestration, checkpoint state, resume semantics, optimization,
//! and mixed-precision policy for `LeWM` experiments. This crate is the library
//! surface behind the `lewm-train` binary; see [RFC 0005].
//!
//! [RFC 0005]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0005-training-system.md
//!
//! ## Module index
//!
//! Training modules are added by the phase-specific implementation issues after
//! the workspace scaffold lands.

pub mod config;

pub use crate::config::{
    CONFIG_SCHEMA_VERSION, CameraView, ConfigError, DatasetConfig, EvalConfig, LossConfig,
    OptimizerKind, PrecisionKind, RootConfig, SO100_ACTION_DIM, SO100_HELDOUT_EPISODES,
    SO100_WARMUP_STEPS, So100DatasetConfig, So100EvalConfig, TrainingConfig, load_root_config,
    to_pretty_toml,
};
