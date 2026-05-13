//! Planning and evaluation primitives, including `CEM` action search, `PushT`
//! planning evaluation, and `SO-100` latent trajectory metrics. This crate stays
//! separate from training and `Tract` inference runners; see [RFC 0006].
//!
//! [RFC 0006]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0006-planning-and-evaluation.md
//!
//! ## Module index
//!
//! Planning modules are added by the phase-specific implementation issues after
//! the workspace scaffold lands.

pub mod cem;
pub mod errors;

pub use crate::cem::{
    CEM_RNG_STREAM, Cem, CemCostModel, CemCostRequest, CemIterTrace, CemPlanInput, CemResult,
    DEFAULT_CEM_CHUNK_SIZE, DEFAULT_CEM_MAX_BATCH_BYTES,
};
pub use crate::errors::LewmPlanError;
