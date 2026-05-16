//! CPU inference runners for exported `LeWM` graphs.
//!
//! Two runner families are exposed:
//!
//! - The Tract CPU runners ([`TractOnnxRunner`], [`TractNnefRunner`]) execute
//!   the published ONNX/NNEF graph exports.
//! - The Burn-direct runner ([`BurnJepaRunner`]) executes the in-Rust `Jepa<B>`
//!   module against a pluggable Burn backend. The runner type is generic over
//!   `Backend`; the in-crate wiring covers `burn-cpu` (`NdArray`). GPU backends
//!   (CUDA, Wgpu, ...) live in the separate `lewm-gpu` crate per RFC 0007 so
//!   `lewm-infer` itself stays free of CUDA / autodiff / NVML deps.

mod loader;
mod traits;

#[cfg(feature = "burn-cpu")]
mod burn_runner;

#[cfg(feature = "tract-nnef")]
mod tract_nnef_runner;
#[cfg(feature = "tract-onnx")]
mod tract_onnx_runner;

pub use crate::runner::loader::{detect_checkpoint_format, load};
#[cfg(feature = "burn-cpu")]
pub use crate::runner::loader::{detect_safetensors, load_with_backend};
pub use crate::runner::traits::{
    IMAGE_ELEMENT_COUNT, InferenceRunner, PREDICTOR_BATCH, RunnerError, RunnerFormat,
    RunnerMetadata,
};

#[cfg(feature = "burn-cpu")]
pub use crate::runner::burn_runner::BurnJepaRunner;

#[cfg(feature = "tract-nnef")]
pub use crate::runner::tract_nnef_runner::TractNnefRunner;
#[cfg(feature = "tract-onnx")]
pub use crate::runner::tract_onnx_runner::TractOnnxRunner;

/// User-selectable inference backend.
///
/// Used by the CLI's `--backend` flag to pick a concrete runner without
/// changing the rest of the planning/eval pipeline.
///
/// GPU backends (CUDA, Wgpu, ...) are wired in the `lewm-gpu` crate; the
/// `lewm-infer` CLI accepts those names so callers can dispatch through a
/// downstream binary, but constructing them inside `lewm-infer` itself
/// returns [`RunnerError::FormatDisabled`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Tract CPU runner driving the ONNX export.
    TractOnnx,
    /// Tract CPU runner driving the NNEF export.
    TractNnef,
    /// Burn-direct CPU runner via `NdArray`.
    BurnCpu,
}

impl BackendKind {
    /// Return the stable lowercase backend name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TractOnnx => "tract-onnx",
            Self::TractNnef => "tract-nnef",
            Self::BurnCpu => "burn-cpu",
        }
    }

    /// Parse a CLI-friendly backend name.
    ///
    /// # Errors
    ///
    /// Returns an error string listing the supported names when the input does
    /// not match any known backend. CUDA / GPU backends are routed through the
    /// `lewm-gpu` crate; this parser rejects them so callers can short-circuit
    /// with a clear message.
    pub fn parse_cli(value: &str) -> Result<Self, String> {
        match value {
            "tract" | "tract-onnx" => Ok(Self::TractOnnx),
            "tract-nnef" => Ok(Self::TractNnef),
            "burn-cpu" => Ok(Self::BurnCpu),
            "burn-cuda" | "burn-gpu" => Err(format!(
                "backend '{value}' lives in the `lewm-gpu` crate per RFC 0007; \
                 use `lewm-gpu::load_cuda_runner` or a downstream binary that links it"
            )),
            other => Err(format!(
                "unknown backend '{other}'; expected tract|tract-onnx|tract-nnef|burn-cpu"
            )),
        }
    }

    /// Return true when the backend runs on GPU.
    ///
    /// All in-crate backends are CPU; GPU backends live in the `lewm-gpu`
    /// crate. The function takes `self` for symmetry with future GPU variants
    /// that may be re-introduced here once they have a matching feature flag.
    #[must_use]
    #[allow(clippy::unused_self)]
    pub const fn is_gpu(self) -> bool {
        false
    }
}
