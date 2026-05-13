//! Training orchestration, checkpoint state, resume semantics, optimization,
//! and mixed-precision policy for `LeWM` experiments. This crate is the library
//! surface behind the `lewm-train` binary; see [RFC 0005].
//!
//! [RFC 0005]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0005-training-system.md
//!
//! ## Module index
//!
//! - [`mixed_precision`] — precision policy and `F32` islands.
//! - [`optim`] — `AdamW` configuration and RFC 0005 decay/no-decay partitioning.
//! - [`schedule`] — cosine decay with linear warmup.
//! - [`step`] — gradient accumulation, clipping, and step guards.

pub mod mixed_precision;
pub mod optim;
pub mod schedule;
pub mod step;
