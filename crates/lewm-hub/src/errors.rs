//! Error types for Hub publication.

use std::path::PathBuf;
use std::time::Duration;

/// Failures surfaced by Hub clients and upload pipelines.
#[derive(Debug, thiserror::Error)]
pub enum HubError {
    /// `HF_TOKEN` was not present in the environment.
    #[error("HF_TOKEN is required for Hub publication")]
    TokenMissing,

    /// Token or user validation failed.
    #[error("Hub authentication failed: {0}")]
    Auth(String),

    /// A filesystem operation failed.
    #[error("filesystem error at {path}: {source}")]
    Io {
        /// Path involved in the failing operation.
        path: PathBuf,
        /// Original I/O error.
        #[source]
        source: std::io::Error,
    },

    /// A Hub JSON response could not be parsed.
    #[error("Hub JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// A local or remote path failed validation.
    #[error("invalid Hub path: {0}")]
    InvalidPath(String),

    /// A remote Hub API call returned an HTTP status.
    #[error("Hub HTTP {status}: {message}")]
    HttpStatus {
        /// HTTP status code.
        status: u16,
        /// Optional `Retry-After` delay for rate limits.
        retry_after: Option<Duration>,
        /// Human-readable status message.
        message: String,
    },

    /// A network operation failed before receiving a status code.
    #[error("Hub network error: {0}")]
    Network(String),

    /// The selected transport does not implement a live Hub operation.
    #[error("Hub transport does not support this operation: {0}")]
    Unsupported(String),
}

impl HubError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn is_retryable(&self) -> bool {
        match self {
            Self::HttpStatus { status, .. } => *status == 429 || *status >= 500,
            Self::Network(_) => true,
            Self::TokenMissing
            | Self::Auth(_)
            | Self::Io { .. }
            | Self::Json(_)
            | Self::InvalidPath(_)
            | Self::Unsupported(_) => false,
        }
    }

    pub(crate) fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::HttpStatus { retry_after, .. } => *retry_after,
            Self::TokenMissing
            | Self::Auth(_)
            | Self::Io { .. }
            | Self::Json(_)
            | Self::InvalidPath(_)
            | Self::Network(_)
            | Self::Unsupported(_) => None,
        }
    }
}
