//! Error types for planning and evaluation APIs.

use std::path::PathBuf;

/// Errors surfaced by `lewm-plan` algorithms.
#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum LewmPlanError {
    /// A CEM hyperparameter or runtime input is invalid.
    #[error("invalid CEM configuration: {reason}")]
    InvalidCemConfig {
        /// Concrete validation failure.
        reason: String,
    },

    /// The caller passed tensors or buffers with incoherent shapes.
    #[error("invalid CEM input: {reason}")]
    InvalidCemInput {
        /// Concrete validation failure.
        reason: String,
    },

    /// The configured cost model returned an invalid response.
    #[error("invalid CEM cost output: {reason}")]
    InvalidCemCost {
        /// Concrete validation failure.
        reason: String,
    },

    /// The cost model failed while scoring candidate actions.
    #[error("CEM cost evaluation failed: {reason}")]
    CostEvaluation {
        /// Concrete cost-model failure.
        reason: String,
    },

    /// The required RFC 0013 RNG sub-stream was unavailable.
    #[error("CEM RNG setup failed: {reason}")]
    Rng {
        /// Concrete RNG setup failure.
        reason: String,
    },
}

impl LewmPlanError {
    pub(crate) fn invalid_config(reason: impl Into<String>) -> Self {
        Self::InvalidCemConfig {
            reason: reason.into(),
        }
    }

    pub(crate) fn invalid_input(reason: impl Into<String>) -> Self {
        Self::InvalidCemInput {
            reason: reason.into(),
        }
    }

    pub(crate) fn invalid_cost(reason: impl Into<String>) -> Self {
        Self::InvalidCemCost {
            reason: reason.into(),
        }
    }

    pub(crate) fn rng(reason: impl Into<String>) -> Self {
        Self::Rng {
            reason: reason.into(),
        }
    }
}

/// Failures surfaced by planning and evaluation drivers.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// Filesystem operation failed while reading or writing an eval artifact.
    #[error("filesystem error at {path}: {source}")]
    Io {
        /// Path involved in the failing filesystem operation.
        path: PathBuf,
        /// Original I/O error.
        #[source]
        source: std::io::Error,
    },

    /// JSON serialization or parsing failed.
    #[error("JSON error while {context}: {source}")]
    Json {
        /// Operation being performed.
        context: String,
        /// Original JSON error.
        #[source]
        source: serde_json::Error,
    },

    /// TOML configuration parsing failed.
    #[error("TOML config error at {path}: {source}")]
    Toml {
        /// Config path being parsed.
        path: PathBuf,
        /// Original TOML parsing error.
        #[source]
        source: toml::de::Error,
    },

    /// Config values were syntactically valid but violated the eval contract.
    #[error("invalid eval configuration: {0}")]
    InvalidConfig(String),

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

    /// The `PushT` JSON-RPC sidecar failed or returned an invalid response.
    #[error("PushT RPC error: {0}")]
    Rpc(String),

    /// Runtime data failed validation before it could be sent to the simulator.
    #[error("invalid eval data: {0}")]
    InvalidData(String),

    /// Dataset transform failure from `lewm-data`.
    #[error(transparent)]
    Data(#[from] lewm_data::DataError),

    /// Arrow record-batch construction failed before writing Parquet.
    #[error(transparent)]
    Arrow(#[from] arrow_schema::ArrowError),

    /// Parquet writer failure.
    #[error(transparent)]
    Parquet(#[from] parquet::errors::ParquetError),

    /// Arrow record-batch construction failed for a specific artifact path.
    #[error("Arrow output error at {path}: {source}")]
    ArrowAt {
        /// Output path involved in the failing operation.
        path: PathBuf,
        /// Original Arrow error.
        #[source]
        source: arrow_schema::ArrowError,
    },

    /// Parquet writer failure for a specific artifact path.
    #[error("Parquet output error at {path}: {source}")]
    ParquetAt {
        /// Output path involved in the failing operation.
        path: PathBuf,
        /// Original Parquet error.
        #[source]
        source: parquet::errors::ParquetError,
    },
}

impl EvalError {
    /// Build an I/O error with path context.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Build a JSON error with operation context.
    pub fn json(context: impl Into<String>, source: serde_json::Error) -> Self {
        Self::Json {
            context: context.into(),
            source,
        }
    }

    /// Build a JSON decode error with path context.
    pub fn json_decode(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
        let path = path.into();
        Self::json(format!("parsing JSON at {}", path.display()), source)
    }

    /// Build a JSON encode error.
    pub fn json_encode(source: serde_json::Error) -> Self {
        Self::json("rendering JSON", source)
    }

    /// Build a TOML error with path context.
    pub fn toml(path: impl Into<PathBuf>, source: toml::de::Error) -> Self {
        Self::Toml {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn invalid_episode(episode_id: u32, reason: impl Into<String>) -> Self {
        Self::InvalidEpisode {
            episode_id,
            reason: reason.into(),
        }
    }

    pub(crate) fn arrow(path: impl Into<PathBuf>, source: arrow_schema::ArrowError) -> Self {
        Self::ArrowAt {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn parquet(path: impl Into<PathBuf>, source: parquet::errors::ParquetError) -> Self {
        Self::ParquetAt {
            path: path.into(),
            source,
        }
    }
}
