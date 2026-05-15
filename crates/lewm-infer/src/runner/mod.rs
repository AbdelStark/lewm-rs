//! CPU and GPU inference runners for exported `LeWM` graphs.
//!
//! Two runner families are exposed:
//!
//! - The Tract CPU runners ([`TractOnnxRunner`], [`TractNnefRunner`]) execute
//!   the published ONNX/NNEF graph exports.
//! - The Burn-direct runners ([`BurnJepaRunner`]) execute the in-Rust `Jepa<B>`
//!   module against a pluggable Burn backend — `NdArray` for CPU, `Cuda` for
//!   NVIDIA GPUs. The Burn runner is the bridge for `--backend burn-cpu` /
//!   `--backend burn-cuda` CLI selection and the `eval` parity command.

mod loader;
mod traits;

#[cfg(any(feature = "burn-cpu", feature = "burn-cuda"))]
mod burn_runner;

#[cfg(feature = "tract-nnef")]
mod tract_nnef_runner;
#[cfg(feature = "tract-onnx")]
mod tract_onnx_runner;

pub use crate::runner::loader::{detect_checkpoint_format, load};
#[cfg(any(feature = "burn-cpu", feature = "burn-cuda"))]
pub use crate::runner::loader::{detect_safetensors, load_with_backend};
pub use crate::runner::traits::{
    IMAGE_ELEMENT_COUNT, InferenceRunner, PREDICTOR_BATCH, RunnerError, RunnerFormat,
    RunnerMetadata,
};

#[cfg(any(feature = "burn-cpu", feature = "burn-cuda"))]
pub use crate::runner::burn_runner::BurnJepaRunner;

#[cfg(feature = "tract-nnef")]
pub use crate::runner::tract_nnef_runner::TractNnefRunner;
#[cfg(feature = "tract-onnx")]
pub use crate::runner::tract_onnx_runner::TractOnnxRunner;

/// User-selectable inference backend.
///
/// Used by the CLI's `--backend` flag to pick a concrete runner without
/// changing the rest of the planning/eval pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Tract CPU runner driving the ONNX export.
    TractOnnx,
    /// Tract CPU runner driving the NNEF export.
    TractNnef,
    /// Burn-direct CPU runner via `NdArray`.
    BurnCpu,
    /// Burn-direct GPU runner via `burn-cuda`.
    BurnCuda,
}

impl BackendKind {
    /// Return the stable lowercase backend name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TractOnnx => "tract-onnx",
            Self::TractNnef => "tract-nnef",
            Self::BurnCpu => "burn-cpu",
            Self::BurnCuda => "burn-cuda",
        }
    }

    /// Parse a CLI-friendly backend name.
    ///
    /// # Errors
    ///
    /// Returns an error string listing the supported names when the input does
    /// not match any known backend.
    pub fn parse_cli(value: &str) -> Result<Self, String> {
        match value {
            "tract" | "tract-onnx" => Ok(Self::TractOnnx),
            "tract-nnef" => Ok(Self::TractNnef),
            "burn-cpu" => Ok(Self::BurnCpu),
            "burn-cuda" | "burn-gpu" => Ok(Self::BurnCuda),
            other => Err(format!(
                "unknown backend '{other}'; expected tract|tract-onnx|tract-nnef|burn-cpu|burn-cuda"
            )),
        }
    }

    /// Return true when the backend runs on GPU.
    #[must_use]
    pub const fn is_gpu(self) -> bool {
        matches!(self, Self::BurnCuda)
    }
}
