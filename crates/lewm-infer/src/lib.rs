//! CPU inference and export verification boundaries for the `Tract` deployment
//! path. This crate intentionally excludes `CUDA` and autodiff dependencies; see
//! [RFC 0007].
//!
//! [RFC 0007]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0007-tract-inference-and-onnx-export.md
//!
//! ## Module index
//!
//! - [`runner`] contains the CPU inference runner trait and Tract-backed loaders.

pub mod runner;
