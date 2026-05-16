//! Checkpoint directory runner loader.

use std::path::Path;

#[cfg(feature = "burn-cpu")]
use crate::runner::BackendKind;
use crate::runner::traits::{InferenceRunner, RunnerError, RunnerFormat};

/// Detect the graph format available in a checkpoint directory.
pub fn detect_checkpoint_format(checkpoint_dir: &Path) -> Option<RunnerFormat> {
    if checkpoint_dir.join("encoder.onnx").exists()
        && checkpoint_dir.join("predictor.onnx").exists()
    {
        Some(RunnerFormat::Onnx)
    } else if checkpoint_dir.join("encoder.nnef").exists()
        && checkpoint_dir.join("predictor.nnef").exists()
    {
        Some(RunnerFormat::Nnef)
    } else {
        None
    }
}

/// Find a `.safetensors` checkpoint next to a checkpoint directory.
///
/// Searches, in order:
/// 1. `checkpoint_dir/weights.safetensors`
/// 2. The highest-numbered `step_*.safetensors` file inside `checkpoint_dir`.
/// 3. `checkpoint_dir/reference.safetensors` (used by parity dump bundles).
#[cfg(feature = "burn-cpu")]
pub fn detect_safetensors(checkpoint_dir: &Path) -> Option<std::path::PathBuf> {
    let weights = checkpoint_dir.join("weights.safetensors");
    if weights.exists() {
        return Some(weights);
    }
    let reference = checkpoint_dir.join("reference.safetensors");
    if reference.exists() {
        return Some(reference);
    }
    let dir_entries = std::fs::read_dir(checkpoint_dir).ok()?;
    let mut step_candidates = Vec::new();
    for entry in dir_entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("safetensors") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if let Some(rest) = stem.strip_prefix("step_")
            && let Ok(step) = rest.parse::<u64>()
        {
            step_candidates.push((step, path));
        }
    }
    step_candidates.sort_by_key(|(step, _)| *step);
    step_candidates.pop().map(|(_, path)| path)
}

/// Load the best available runner from a checkpoint directory.
///
/// The selection ladder is ONNX, then NNEF, then Burn-direct once that fallback
/// is implemented behind an ADR-gated feature.
///
/// # Errors
///
/// Returns [`RunnerError`] when no graph pair is found, the selected format is
/// disabled at compile time, or backend loading fails.
pub fn load(checkpoint_dir: &Path) -> Result<Box<dyn InferenceRunner>, RunnerError> {
    match detect_checkpoint_format(checkpoint_dir) {
        Some(RunnerFormat::Onnx) => load_onnx(checkpoint_dir),
        Some(RunnerFormat::Nnef) => load_nnef(checkpoint_dir),
        Some(RunnerFormat::BurnDirect) => Err(RunnerError::FormatDisabled {
            format: RunnerFormat::BurnDirect,
        }),
        None => Err(RunnerError::NoExportFound {
            checkpoint_dir: checkpoint_dir.to_path_buf(),
        }),
    }
}

#[cfg(feature = "tract-onnx")]
fn load_onnx(checkpoint_dir: &Path) -> Result<Box<dyn InferenceRunner>, RunnerError> {
    Ok(Box::new(crate::runner::TractOnnxRunner::new(
        checkpoint_dir,
    )?))
}

#[cfg(not(feature = "tract-onnx"))]
fn load_onnx(_checkpoint_dir: &Path) -> Result<Box<dyn InferenceRunner>, RunnerError> {
    Err(RunnerError::FormatDisabled {
        format: RunnerFormat::Onnx,
    })
}

#[cfg(feature = "tract-nnef")]
fn load_nnef(checkpoint_dir: &Path) -> Result<Box<dyn InferenceRunner>, RunnerError> {
    Ok(Box::new(crate::runner::TractNnefRunner::new(
        checkpoint_dir,
    )?))
}

#[cfg(not(feature = "tract-nnef"))]
fn load_nnef(_checkpoint_dir: &Path) -> Result<Box<dyn InferenceRunner>, RunnerError> {
    Err(RunnerError::FormatDisabled {
        format: RunnerFormat::Nnef,
    })
}

/// Load a runner for a given [`BackendKind`].
///
/// For `TractOnnx`/`TractNnef`, this reuses the existing checkpoint-directory
/// discovery and ignores `safetensors_path`. For `BurnCpu`, the runner is
/// constructed from the Safetensors weights (resolved via
/// `detect_safetensors` when `safetensors_path` is `None`).
///
/// GPU backends are wired in `lewm-gpu::load_cuda_runner` and are not callable
/// through this entry point.
///
/// # Errors
///
/// Returns [`RunnerError`] when the requested backend feature is disabled, the
/// checkpoint is missing, or backend loading fails.
#[cfg(feature = "burn-cpu")]
pub fn load_with_backend(
    backend: BackendKind,
    checkpoint_dir: &Path,
    safetensors_path: Option<&Path>,
    config: Option<lewm_core::JepaConfig>,
) -> Result<Box<dyn InferenceRunner>, RunnerError> {
    match backend {
        BackendKind::TractOnnx => load_onnx(checkpoint_dir),
        BackendKind::TractNnef => load_nnef(checkpoint_dir),
        BackendKind::BurnCpu => {
            let resolved = match safetensors_path {
                Some(path) => path.to_path_buf(),
                None => detect_safetensors(checkpoint_dir).ok_or_else(|| {
                    RunnerError::NoExportFound {
                        checkpoint_dir: checkpoint_dir.to_path_buf(),
                    }
                })?,
            };
            load_burn_cpu(&resolved, config.unwrap_or_default())
        },
    }
}

#[cfg(feature = "burn-cpu")]
fn load_burn_cpu(
    safetensors_path: &Path,
    config: lewm_core::JepaConfig,
) -> Result<Box<dyn InferenceRunner>, RunnerError> {
    use crate::runner::BurnJepaRunner;
    let device = burn_ndarray::NdArrayDevice::default();
    let runner = BurnJepaRunner::<burn_ndarray::NdArray<f32>>::from_safetensors(
        safetensors_path,
        config,
        device,
        "cpu",
    )?;
    Ok(Box::new(runner))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn load_reports_no_export_found() -> Result<(), Box<dyn std::error::Error>> {
        let root = unique_temp_dir("lewm-runner-empty")?;
        let error = load(&root).err().ok_or("expected no export error")?;
        assert!(error.to_string().contains("no ONNX or NNEF graph pair"));
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn detect_checkpoint_format_prefers_onnx() -> Result<(), Box<dyn std::error::Error>> {
        let root = unique_temp_dir("lewm-runner-detect")?;
        for file in [
            "encoder.onnx",
            "predictor.onnx",
            "encoder.nnef",
            "predictor.nnef",
        ] {
            fs::write(root.join(file), b"graph")?;
        }
        assert_eq!(detect_checkpoint_format(&root), Some(RunnerFormat::Onnx));
        fs::remove_dir_all(root)?;
        Ok(())
    }

    fn unique_temp_dir(prefix: &str) -> std::io::Result<PathBuf> {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        path.push(format!("{prefix}-{nanos}"));
        fs::create_dir_all(&path)?;
        Ok(path)
    }
}
