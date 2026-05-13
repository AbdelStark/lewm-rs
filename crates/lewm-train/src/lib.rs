//! Training orchestration, checkpoint state, resume semantics, optimization,
//! and mixed-precision policy for `LeWM` experiments. This crate is the library
//! surface behind the `lewm-train` binary; see [RFC 0005].
//!
//! [RFC 0005]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0005-training-system.md
//!
//! ## Module index
//!
//! - [`checkpoint`] — epoch checkpoint files, sidecars, atomic writes, and
//!   pruning.
//! - [`config`] owns the root training TOML schema, layered overrides, and
//!   reproducibility hash.
//! - [`mixed_precision`] — precision policy and `F32` islands.
//! - [`optim`] — `AdamW` configuration and RFC 0005 decay/no-decay partitioning.
//! - [`resume`] — run-directory resume detection, RNG restoration, and
//!   shutdown handling.
//! - [`schedule`] — cosine decay with linear warmup.
//! - [`step`] — gradient accumulation, clipping, and step guards.
//! - [`trainer`] — outer-loop state machine and trainer artifacts.
//!
pub mod checkpoint;
pub mod config;
pub mod mixed_precision;
pub mod optim;
pub mod resume;
pub mod schedule;
pub mod step;
pub mod trainer;
