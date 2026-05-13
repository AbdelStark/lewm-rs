//! Checkpoint directory runner loader.

use std::path::Path;

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
