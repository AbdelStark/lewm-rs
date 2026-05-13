//! Shared runner traits and metadata.

use std::fmt;
use std::path::PathBuf;

/// Image element count for `(3, 224, 224)` F32 CHW pixels.
pub const IMAGE_ELEMENT_COUNT: usize = 3 * 224 * 224;

/// Batch size used by the predictor runner API.
pub const PREDICTOR_BATCH: usize = 1;

/// Runner backend format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunnerFormat {
    /// ONNX graph pair loaded through `tract-onnx`.
    Onnx,
    /// NNEF graph pair loaded through `tract-nnef`.
    Nnef,
    /// Burn-record-direct fallback.
    BurnDirect,
}

impl RunnerFormat {
    /// Return the stable lowercase format name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Onnx => "onnx",
            Self::Nnef => "nnef",
            Self::BurnDirect => "burn-direct",
        }
    }
}

impl fmt::Display for RunnerFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Metadata exposed by a runner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunnerMetadata {
    /// Runner backend format.
    pub format: RunnerFormat,
    /// Encoder graph path.
    pub encoder_path: PathBuf,
    /// Predictor graph path.
    pub predictor_path: PathBuf,
    /// Whether the backend was optimized before becoming runnable.
    pub optimized: bool,
    /// Intra-op thread target derived from available CPU parallelism.
    pub intra_op_threads: usize,
}

/// Common CPU inference runner interface.
pub trait InferenceRunner: Send {
    /// Encode one CHW image and return a flat latent embedding.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] when graph execution or output extraction fails.
    fn encode(&mut self, pixels: &[f32; IMAGE_ELEMENT_COUNT]) -> Result<Vec<f32>, RunnerError>;

    /// Predict latent rollouts from `(1, H, D)` history and `(1, H, A)` actions.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError`] when shape validation, graph execution, or output
    /// extraction fails.
    fn predict(
        &mut self,
        history: &[f32],
        actions: &[f32],
        h: usize,
        a: usize,
    ) -> Result<Vec<f32>, RunnerError>;

    /// Return runner metadata.
    fn metadata(&self) -> RunnerMetadata;
}

/// Runner error type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunnerError {
    /// No supported graph pair was found in a checkpoint directory.
    NoExportFound {
        /// Directory that was inspected.
        checkpoint_dir: PathBuf,
    },
    /// A graph pair exists, but the corresponding crate feature is disabled.
    FormatDisabled {
        /// Disabled format.
        format: RunnerFormat,
    },
    /// A required graph file is missing.
    MissingGraph {
        /// Missing graph path.
        path: PathBuf,
    },
    /// Tensor shape validation failed before graph execution.
    InvalidShape {
        /// Shape failure reason.
        reason: String,
    },
    /// Tract failed to load, optimize, run, or extract an output.
    Backend {
        /// Backend operation context.
        context: String,
        /// Backend error string.
        source: String,
    },
}

impl fmt::Display for RunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoExportFound { checkpoint_dir } => {
                write!(
                    f,
                    "no ONNX or NNEF graph pair found in {}",
                    checkpoint_dir.display()
                )
            },
            Self::FormatDisabled { format } => {
                write!(
                    f,
                    "{format} graph pair found, but the `{format}` feature is disabled"
                )
            },
            Self::MissingGraph { path } => {
                write!(f, "required graph is missing: {}", path.display())
            },
            Self::InvalidShape { reason } => write!(f, "invalid runner input shape: {reason}"),
            Self::Backend { context, source } => write!(f, "{context}: {source}"),
        }
    }
}

impl std::error::Error for RunnerError {}

/// Return available CPU parallelism for runner metadata and Tract plan options.
#[cfg(any(feature = "tract-onnx", feature = "tract-nnef"))]
pub(super) fn available_intra_op_threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

/// Validate predictor input lengths and return the inferred latent dimension.
///
/// # Errors
///
/// Returns [`RunnerError::InvalidShape`] when history/actions are inconsistent.
#[cfg(any(feature = "tract-onnx", feature = "tract-nnef"))]
pub(super) fn validate_predict_shapes(
    history: &[f32],
    actions: &[f32],
    h: usize,
    a: usize,
) -> Result<usize, RunnerError> {
    if h == 0 {
        return Err(RunnerError::InvalidShape {
            reason: "history length H must be non-zero".to_owned(),
        });
    }
    if a == 0 {
        return Err(RunnerError::InvalidShape {
            reason: "action dimension A must be non-zero".to_owned(),
        });
    }
    let action_expected = PREDICTOR_BATCH
        .checked_mul(h)
        .and_then(|len| len.checked_mul(a))
        .ok_or_else(|| RunnerError::InvalidShape {
            reason: "action shape overflowed usize".to_owned(),
        })?;
    if actions.len() != action_expected {
        return Err(RunnerError::InvalidShape {
            reason: format!("actions must have 1 * H * A = {action_expected} elements"),
        });
    }
    if history.len() % h != 0 {
        return Err(RunnerError::InvalidShape {
            reason: "history length must be divisible by H".to_owned(),
        });
    }
    let latent_dim = history.len() / h;
    if latent_dim == 0 {
        return Err(RunnerError::InvalidShape {
            reason: "latent dimension D must be non-zero".to_owned(),
        });
    }
    Ok(latent_dim)
}

/// Validate that a graph path exists.
///
/// # Errors
///
/// Returns [`RunnerError::MissingGraph`] when the path is absent.
#[cfg(any(feature = "tract-onnx", feature = "tract-nnef"))]
pub(super) fn require_graph(path: PathBuf) -> Result<PathBuf, RunnerError> {
    if path.exists() {
        Ok(path)
    } else {
        Err(RunnerError::MissingGraph { path })
    }
}
