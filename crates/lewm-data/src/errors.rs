//! Error types for dataset loading and sample extraction.

use std::path::PathBuf;

/// Data-plane failures surfaced by dataset loaders and transforms.
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    /// Filesystem operation failed while discovering or opening dataset shards.
    #[error("filesystem error at {path}: {source}")]
    Io {
        /// Path involved in the failing operation.
        path: PathBuf,
        /// Original I/O error.
        #[source]
        source: std::io::Error,
    },

    /// HDF5 operation failed while reading a shard.
    #[error("HDF5 error in {context}: {source}")]
    Hdf5 {
        /// Operation or dataset path being accessed.
        context: String,
        /// Original HDF5 error.
        #[source]
        source: hdf5_metno::Error,
    },

    /// Safetensors operation failed while reading or writing data stats.
    #[error("safetensors error at {path}: {source}")]
    Safetensors {
        /// Stats file involved in the failing operation.
        path: PathBuf,
        /// Original safetensors error.
        #[source]
        source: safetensors::SafeTensorError,
    },

    /// A dataset shard does not match the `PushT` schema contract.
    #[error("schema mismatch at {path}: expected {expected}, found {found}")]
    SchemaMismatch {
        /// Dataset or shard path that failed validation.
        path: String,
        /// Expected schema fragment.
        expected: String,
        /// Observed schema fragment.
        found: String,
    },

    /// Loader configuration is invalid before any shard can be opened.
    #[error("invalid data configuration: {0}")]
    InvalidConfig(String),

    /// Transform input or persisted transform statistics are invalid.
    #[error("invalid transform input: {0}")]
    InvalidTransform(String),

    /// Dataset discovery succeeded but no usable shards/windows were present.
    #[error("empty dataset: {0}")]
    EmptyDataset(String),
}

impl DataError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn hdf5(context: impl Into<String>, source: hdf5_metno::Error) -> Self {
        Self::Hdf5 {
            context: context.into(),
            source,
        }
    }

    pub(crate) fn safetensors(
        path: impl Into<PathBuf>,
        source: safetensors::SafeTensorError,
    ) -> Self {
        Self::Safetensors {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn schema(
        path: impl Into<String>,
        expected: impl Into<String>,
        found: impl Into<String>,
    ) -> Self {
        Self::SchemaMismatch {
            path: path.into(),
            expected: expected.into(),
            found: found.into(),
        }
    }
}
