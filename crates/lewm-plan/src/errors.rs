//! Error types for planning and evaluation drivers.

use std::path::PathBuf;

/// Planning and evaluation failures surfaced by `lewm-plan`.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// An evaluation input was malformed before a model was invoked.
    #[error("invalid evaluation input: {0}")]
    InvalidInput(String),

    /// A specific SO-100 episode violated the latent-rollout contract.
    #[error("invalid SO-100 episode {episode_id}: {reason}")]
    InvalidEpisode {
        /// Episode identifier from the dataset split.
        episode_id: u32,
        /// Human-readable validation failure.
        reason: String,
    },

    /// A metric could not be computed from finite, non-degenerate vectors.
    #[error("metric computation failed: {0}")]
    Metric(String),

    /// Filesystem I/O failed for an output or input path.
    #[error("I/O error at {path}: {source}")]
    Io {
        /// Path involved in the failing operation.
        path: PathBuf,
        /// Original I/O error.
        #[source]
        source: std::io::Error,
    },

    /// JSON parsing failed for an input path.
    #[error("JSON parse error at {path}: {source}")]
    JsonDecode {
        /// Path involved in the failing operation.
        path: PathBuf,
        /// Original JSON error.
        #[source]
        source: serde_json::Error,
    },

    /// JSON rendering failed.
    #[error("JSON render error: {source}")]
    JsonEncode {
        /// Original JSON error.
        #[source]
        source: serde_json::Error,
    },

    /// Arrow batch construction failed before Parquet output.
    #[error("Arrow output error for {path}: {source}")]
    Arrow {
        /// Path involved in the failing operation.
        path: PathBuf,
        /// Original Arrow error.
        #[source]
        source: arrow_schema::ArrowError,
    },

    /// Parquet output failed.
    #[error("Parquet output error for {path}: {source}")]
    Parquet {
        /// Path involved in the failing operation.
        path: PathBuf,
        /// Original Parquet error.
        #[source]
        source: parquet::errors::ParquetError,
    },
}

impl EvalError {
    pub(crate) fn invalid_episode(episode_id: u32, reason: impl Into<String>) -> Self {
        Self::InvalidEpisode {
            episode_id,
            reason: reason.into(),
        }
    }

    /// Build an I/O error with the path that was being accessed.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Build a JSON decode error with the path that was being parsed.
    pub fn json_decode(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
        Self::JsonDecode {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn json_encode(source: serde_json::Error) -> Self {
        Self::JsonEncode { source }
    }

    pub(crate) fn arrow(path: impl Into<PathBuf>, source: arrow_schema::ArrowError) -> Self {
        Self::Arrow {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn parquet(path: impl Into<PathBuf>, source: parquet::errors::ParquetError) -> Self {
        Self::Parquet {
            path: path.into(),
            source,
        }
    }
}
