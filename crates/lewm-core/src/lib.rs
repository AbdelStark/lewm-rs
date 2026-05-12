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
