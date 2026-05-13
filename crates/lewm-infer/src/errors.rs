//! Error types for `lewm-infer` export and CPU inference boundaries.

use std::path::PathBuf;

/// Error type used by `lewm-infer`.
#[derive(Debug, thiserror::Error)]
pub enum InferError {
    /// The model configuration cannot produce the requested inference graph.
    #[error("invalid model export contract: {reason}")]
    InvalidExportContract {
        /// Human-readable validation failure.
        reason: String,
    },

    /// A required filesystem path was not present.
    #[error("required path does not exist: {path}")]
    MissingPath {
        /// Path that was expected to exist.
        path: PathBuf,
    },

    /// Export backend failed while writing one graph.
    #[error("failed to export {graph} graph to {path}: {source}")]
    ExportFailed {
        /// Graph name.
        graph: &'static str,
        /// Target graph path.
        path: PathBuf,
        /// Backend error.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Filesystem I/O failed.
    #[error("I/O error at {path}: {source}")]
    Io {
        /// Path associated with the failure.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },

    /// JSON serialization failed.
    #[error("failed to serialize export metadata at {path}: {source}")]
    Json {
        /// Path associated with the failure.
        path: PathBuf,
        /// Underlying JSON error.
        source: serde_json::Error,
    },
}

impl InferError {
    /// Construct an invalid-export-contract error.
    pub fn invalid_export_contract(reason: impl Into<String>) -> Self {
        Self::InvalidExportContract {
            reason: reason.into(),
        }
    }
}

/// Result alias for `lewm-infer` operations.
pub type InferResult<T> = Result<T, InferError>;
