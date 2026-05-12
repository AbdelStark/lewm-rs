//! Training orchestration, checkpoint state, resume semantics, optimization,
//! and mixed-precision policy for `LeWM` experiments. This crate is the library
//! surface behind the `lewm-train` binary; see [RFC 0005].
//!
//! [RFC 0005]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0005-training-system.md
//!
//! ## Module index
//!
//! - [`mixed_precision`] — precision policy and `F32` islands.

pub mod mixed_precision;
