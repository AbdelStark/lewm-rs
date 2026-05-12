//! Data loading, preprocessing, batching, and dataset streaming boundaries for
//! `PushT` and `SO-100` inputs. This crate owns dataset shape normalization and
//! keeps Python-only preparation at the repository edge; see [RFC 0004].
//!
//! [RFC 0004]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0004-data-pipeline.md
//!
//! ## Module index
//!
//! Dataset modules are added by the phase-specific implementation issues after
//! the workspace scaffold lands.
