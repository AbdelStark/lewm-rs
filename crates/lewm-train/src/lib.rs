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

pub mod warmstart;

pub use crate::warmstart::{
    SO100_ACTION_DIM, TRANSFER_MODULE_PREFIXES, TensorRecord, TrainError, TrainStateRecord,
    WarmstartLoad, WarmstartProvenance, load_warmstart,
};
