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

pub mod batch;
pub mod errors;
pub mod pusht;
pub mod transform;

pub use crate::batch::{
    Batch, BatchBackend, BatchDtype, BatchTensor, HostBackend, HostDevice, collate,
};
pub use crate::errors::DataError;
pub use crate::pusht::{PushtConfig, PushtDataset, Sample, SampleMeta, Split};
pub use crate::transform::{ActionNormalizer, ImagePreprocessor, InterpKind, TransformStats};
