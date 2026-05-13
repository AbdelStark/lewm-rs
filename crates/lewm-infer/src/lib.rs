//! CPU inference and export verification boundaries for the `Tract` deployment
//! path. This crate intentionally excludes `CUDA` and autodiff dependencies; see
//! [RFC 0007].
//!
//! [RFC 0007]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0007-tract-inference-and-onnx-export.md
//!
//! ## Module index
//!
//! - [`plan`] contains CPU-side CEM action search for inference.
//! - [`preprocess`] contains RFC 0004-compatible image preprocessing for
//!   inference inputs.
//! - [`runner`] contains the CPU inference runner trait and Tract-backed loaders.

pub mod plan;
pub mod preprocess;
pub mod runner;
