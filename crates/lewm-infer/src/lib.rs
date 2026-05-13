//! CPU inference and export verification boundaries for the `Tract` deployment
//! path. This crate intentionally excludes `CUDA` and autodiff dependencies; see
//! [RFC 0007].
//!
//! [RFC 0007]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0007-tract-inference-and-onnx-export.md
//!
//! ## Module index
//!
//! - [`export`] locks the RFC 0007 ONNX export graph contract and verifier
//!   fallback contract.
//! - [`errors`] exposes the crate error type.

pub mod errors;
pub mod export;

pub use crate::errors::{InferError, InferResult};
