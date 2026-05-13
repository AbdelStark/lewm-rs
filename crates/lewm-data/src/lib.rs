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
pub mod prefetch;
pub mod pusht;
pub mod so100;
pub mod stats;
pub mod transform;

pub use crate::batch::{
    Batch, BatchBackend, BatchDtype, BatchTensor, HostBackend, HostDevice, collate,
};
pub use crate::errors::DataError;
pub use crate::prefetch::{
    DATA_QUEUE_DEPTH_METRIC, Dataset, HostPrefetchDevice, HostPrefetcher, Prefetcher,
    PrefetcherConfig,
};
pub use crate::pusht::{PushtConfig, PushtDataset, Sample, SampleMeta, Split};
pub use crate::so100::{CameraView, SO100_HELD_OUT_EPISODES, So100Config, So100Dataset};
pub use crate::stats::{ComputeStatsConfig, DatasetStats, StatsDataset, compute_stats};
pub use crate::transform::{ActionNormalizer, ImagePreprocessor, InterpKind, TransformStats};
