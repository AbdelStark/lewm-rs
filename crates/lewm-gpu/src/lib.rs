//! GPU inference helpers for `LeWM` — keeps the `burn-cuda` dependency outside
//! `lewm-infer` per the RFC 0007 layering rule enforced by
//! `scripts/check_layers.py` (`burn-cuda` is in `INFER_BANNED_DEPS`).
//!
//! The crate is intentionally thin: it re-uses `lewm-infer`'s backend-generic
//! `BurnJepaRunner` and the `lewm-core::import` Safetensors loader, and only
//! adds the CUDA-specific glue. Higher-level callers (CLI, demo Space, paper
//! benchmarks) depend on this crate when they need a GPU runner; the
//! `lewm-infer` crate itself stays CPU-only.
//!
//! ## Example
//!
//! ```ignore
//! use lewm_core::JepaConfig;
//! use lewm_gpu::{LewmGpuError, load_cuda_runner};
//!
//! let runner = load_cuda_runner(
//!     std::path::Path::new("/path/to/weights.safetensors"),
//!     JepaConfig::default(),
//! )?;
//! # Ok::<_, LewmGpuError>(())
//! ```

#![cfg_attr(not(feature = "burn-cuda"), allow(unused))]

use std::path::Path;

use lewm_core::JepaConfig;
use lewm_infer::runner::{InferenceRunner, RunnerError};

/// Errors raised while loading the GPU runner.
#[derive(Debug, thiserror::Error)]
pub enum LewmGpuError {
    /// The crate was built without the `burn-cuda` feature.
    #[error(
        "lewm-gpu was built without the `burn-cuda` feature; rebuild with --features burn-cuda"
    )]
    CudaFeatureDisabled,
    /// The underlying runner failed to load.
    #[error("runner failure: {source}")]
    Runner {
        /// Original runner error.
        #[from]
        source: RunnerError,
    },
}

/// Construct a CUDA-backed inference runner from a Safetensors weights file.
///
/// # Errors
///
/// Returns [`LewmGpuError::CudaFeatureDisabled`] when the crate was built
/// without the `burn-cuda` feature, or [`LewmGpuError::Runner`] when the
/// underlying [`BurnJepaRunner`](lewm_infer::runner::BurnJepaRunner) fails to
/// load the weights, no CUDA device is available, or the GPU rejects the
/// model.
pub fn load_cuda_runner(
    safetensors_path: &Path,
    config: JepaConfig,
) -> Result<Box<dyn InferenceRunner>, LewmGpuError> {
    load_cuda_runner_impl(safetensors_path, config)
}

#[cfg(feature = "burn-cuda")]
fn load_cuda_runner_impl(
    safetensors_path: &Path,
    config: JepaConfig,
) -> Result<Box<dyn InferenceRunner>, LewmGpuError> {
    use burn_cuda::{Cuda, CudaDevice};
    use lewm_infer::runner::BurnJepaRunner;

    let device = CudaDevice::default();
    let runner =
        BurnJepaRunner::<Cuda>::from_safetensors(safetensors_path, config, device, "cuda")?;
    Ok(Box::new(runner))
}

#[cfg(not(feature = "burn-cuda"))]
fn load_cuda_runner_impl(
    _safetensors_path: &Path,
    _config: JepaConfig,
) -> Result<Box<dyn InferenceRunner>, LewmGpuError> {
    Err(LewmGpuError::CudaFeatureDisabled)
}

/// Return `true` when the crate was compiled with the `burn-cuda` feature.
///
/// Callers can use this to fall back to CPU inference instead of bubbling the
/// [`LewmGpuError::CudaFeatureDisabled`] up.
#[must_use]
pub const fn cuda_supported() -> bool {
    cfg!(feature = "burn-cuda")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_supported_matches_feature() {
        assert_eq!(cuda_supported(), cfg!(feature = "burn-cuda"));
    }

    #[cfg(not(feature = "burn-cuda"))]
    #[test]
    fn load_cuda_runner_errors_without_feature() {
        let result = load_cuda_runner(
            std::path::Path::new("/nonexistent.safetensors"),
            JepaConfig::default(),
        );
        assert!(matches!(result, Err(LewmGpuError::CudaFeatureDisabled)));
    }
}
