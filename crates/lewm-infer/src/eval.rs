//! Parity evaluation utilities for the inference runners.
//!
//! The eval module loads Safetensors dumps captured from the `LeWorldModel`
//! Python reference (the same dumps consumed by `lewm-core::tests::support`)
//! and compares them against the outputs of an [`InferenceRunner`]. The output
//! is a JSON report compatible with the existing reports/ directory format.
//!
//! The expected dump layout follows the
//! [`AbdelStark/lewm-rs-parity-dumps`](https://huggingface.co/datasets/AbdelStark/lewm-rs-parity-dumps)
//! contract:
//!
//! ```text
//! dumps/
//!   encoder/cls.safetensors          # (B*T, encoder.hidden_size)
//!   projector/output.safetensors     # (B*T, predictor.hidden_dim)
//!   action_encoder/output.safetensors
//!   predictor/output.safetensors
//!   pred_proj/output.safetensors
//! ```
//!
//! The fixture inputs (`pixels.npy`, `actions.npy`) ship with the repo at
//! `tests/fixtures/parity_fixture.npz`; the eval reads them via the same
//! `load_fixture()` path used by the parity tests.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::runner::{IMAGE_ELEMENT_COUNT, InferenceRunner, RunnerError};

/// Default L∞ pass threshold matching RFC 0008 (`encoder`/`projector` parity).
pub const DEFAULT_TOLERANCE: f32 = 1e-4;

/// Stage name for the projected encoder output (`Jepa::encode`).
pub const STAGE_PROJECTOR: &str = "projector_output";

/// Stage name for the prediction projector output (`Jepa::predict`).
pub const STAGE_PRED_PROJ: &str = "pred_proj_output";

/// Numerical comparison summary for one tensor pair.
#[derive(Debug, Clone, Serialize)]
pub struct StageStats {
    /// Stage label.
    pub stage: String,
    /// Number of compared scalars.
    pub element_count: usize,
    /// L∞ (max abs diff).
    pub linf: f32,
    /// Mean absolute difference.
    pub mean_abs: f32,
    /// Root-mean-square difference.
    pub rmse: f32,
    /// Mean reference magnitude (used to contextualise the absolute errors).
    pub ref_mean_abs: f32,
    /// Maximum reference magnitude.
    pub ref_max_abs: f32,
    /// Fraction of elements whose abs diff exceeds `tolerance`.
    pub fraction_above_tolerance: f32,
    /// Tolerance threshold the run was scored against.
    pub tolerance: f32,
    /// Whether the L∞ is within tolerance.
    pub pass: bool,
}

/// Wall-clock timing summary captured alongside numerical metrics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct LatencyStats {
    /// Mean encode latency (ms).
    pub encode_mean_ms: f64,
    /// Mean predict latency (ms).
    pub predict_mean_ms: f64,
    /// Number of encode calls timed.
    pub encode_calls: usize,
    /// Number of predict calls timed.
    pub predict_calls: usize,
}

/// Aggregated eval report.
#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    /// Runner backend label, e.g. `tract-onnx`, `burn-cpu`, `burn-cuda`.
    pub backend: String,
    /// Source of the reference dumps.
    pub reference_dumps: PathBuf,
    /// Tolerance threshold applied to every stage.
    pub tolerance: f32,
    /// Per-stage numerical statistics.
    pub stages: Vec<StageStats>,
    /// Wall-clock timing summary.
    pub latency: LatencyStats,
    /// Aggregate pass flag (all stages pass).
    pub pass: bool,
}

impl EvalReport {
    /// Convenience helper for the CLI: pretty-print the report as JSON.
    ///
    /// # Errors
    ///
    /// Returns [`serde_json::Error`] when the report cannot be serialized.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Errors raised while running the parity eval pipeline.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// A required dump file was missing.
    #[error("missing reference dump at {}", path.display())]
    MissingDump {
        /// Missing path.
        path: PathBuf,
    },
    /// A filesystem operation failed.
    #[error("I/O error at {}: {source}", path.display())]
    Io {
        /// Path involved in the failure.
        path: PathBuf,
        /// Original error.
        source: std::io::Error,
    },
    /// Safetensors deserialization failed.
    #[error("safetensors error at {}: {source}", path.display())]
    Safetensors {
        /// Path involved in the failure.
        path: PathBuf,
        /// Original error.
        source: safetensors::SafeTensorError,
    },
    /// A dump tensor had an unexpected dtype.
    #[error("dump tensor {name} has unsupported dtype {dtype:?}; expected F32")]
    InvalidDumpDtype {
        /// Tensor name.
        name: String,
        /// Encountered dtype.
        dtype: safetensors::Dtype,
    },
    /// Two slices used in a comparison had different lengths.
    #[error("stage {stage}: length mismatch — runner={runner_len}, reference={reference_len}")]
    LengthMismatch {
        /// Stage label.
        stage: String,
        /// Runner output length.
        runner_len: usize,
        /// Reference output length.
        reference_len: usize,
    },
    /// Runner execution failed.
    #[error("runner failure during {stage}: {source}")]
    Runner {
        /// Stage label.
        stage: String,
        /// Original runner error.
        source: RunnerError,
    },
    /// A scalar input vector had the wrong shape.
    #[error("expected {expected} F32 elements, got {actual}")]
    InvalidInputLength {
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        actual: usize,
    },
}

/// One reference dump tensor: shape + F32 values.
#[derive(Debug, Clone)]
pub struct DumpTensor {
    /// Row-major shape.
    pub shape: Vec<usize>,
    /// Row-major values.
    pub values: Vec<f32>,
}

/// Load `data` F32 tensor from a Safetensors file in the parity-dump format.
///
/// # Errors
///
/// Returns [`EvalError`] when the file is missing, cannot be parsed, or stores
/// a non-F32 `data` tensor.
pub fn load_dump_tensor(path: &Path) -> Result<DumpTensor, EvalError> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(EvalError::MissingDump {
                path: path.to_path_buf(),
            });
        },
        Err(source) => {
            return Err(EvalError::Io {
                path: path.to_path_buf(),
                source,
            });
        },
    };
    let safe =
        safetensors::SafeTensors::deserialize(&bytes).map_err(|source| EvalError::Safetensors {
            path: path.to_path_buf(),
            source,
        })?;
    let view = safe
        .tensor("data")
        .map_err(|source| EvalError::Safetensors {
            path: path.to_path_buf(),
            source,
        })?;
    if view.dtype() != safetensors::Dtype::F32 {
        return Err(EvalError::InvalidDumpDtype {
            name: "data".to_owned(),
            dtype: view.dtype(),
        });
    }
    let data = view.data();
    if !data.len().is_multiple_of(4) {
        return Err(EvalError::Safetensors {
            path: path.to_path_buf(),
            source: safetensors::SafeTensorError::JsonError(serde_json::error::Error::io(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "F32 data not 4-byte aligned",
                ),
            )),
        });
    }
    let values = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(DumpTensor {
        shape: view.shape().to_vec(),
        values,
    })
}

/// Compute numerical stats for one stage pair.
///
/// # Errors
///
/// Returns [`EvalError::LengthMismatch`] when the runner and reference slices
/// have different lengths.
pub fn compare_stage(
    stage: impl Into<String>,
    runner_output: &[f32],
    reference: &[f32],
    tolerance: f32,
) -> Result<StageStats, EvalError> {
    let stage = stage.into();
    if runner_output.len() != reference.len() {
        return Err(EvalError::LengthMismatch {
            stage,
            runner_len: runner_output.len(),
            reference_len: reference.len(),
        });
    }
    let mut linf = 0.0_f32;
    let mut sum_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut ref_sum_abs = 0.0_f64;
    let mut ref_max_abs = 0.0_f32;
    let mut count_above = 0_usize;
    for (actual, expected) in runner_output.iter().zip(reference) {
        let diff = (*actual - *expected).abs();
        if diff > linf {
            linf = diff;
        }
        sum_abs += f64::from(diff);
        sum_sq += f64::from(diff) * f64::from(diff);
        let ref_abs = expected.abs();
        ref_sum_abs += f64::from(ref_abs);
        if ref_abs > ref_max_abs {
            ref_max_abs = ref_abs;
        }
        if diff > tolerance {
            count_above += 1;
        }
    }
    let count = runner_output.len();
    let denom = usize_to_f64(count.max(1));
    let fraction = usize_to_f32(count_above) / usize_to_f32(count.max(1));
    Ok(StageStats {
        stage,
        element_count: count,
        linf,
        mean_abs: as_f32(sum_abs / denom),
        rmse: as_f32((sum_sq / denom).sqrt()),
        ref_mean_abs: as_f32(ref_sum_abs / denom),
        ref_max_abs,
        fraction_above_tolerance: fraction,
        tolerance,
        pass: linf <= tolerance,
    })
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f32(value: usize) -> f32 {
    value as f32
}

#[allow(clippy::cast_possible_truncation)]
fn as_f32(value: f64) -> f32 {
    if value.is_finite() {
        value as f32
    } else {
        f32::NAN
    }
}

/// Inputs for [`run_parity_eval`].
///
/// Packaged as a struct so the callers can grow without thrashing the
/// signature.
#[derive(Debug)]
pub struct ParityEvalInputs<'a> {
    /// CHW F32 pixels to feed the encoder.
    pub pixels: &'a [f32; IMAGE_ELEMENT_COUNT],
    /// Predictor history latents, `(history_steps * latent_dim,)`.
    pub history: &'a [f32],
    /// Predictor action window, `(history_steps * action_dim,)`.
    pub actions: &'a [f32],
    /// Number of history steps.
    pub history_steps: usize,
    /// Action dimension.
    pub action_dim: usize,
    /// Directory containing the parity dumps.
    pub dumps_dir: &'a Path,
    /// L∞ pass threshold.
    pub tolerance: f32,
    /// Backend label recorded in the report.
    pub backend: &'a str,
}

/// Run a parity comparison: encode `pixels`, predict on `actions`, compare to
/// reference dumps, time the calls, and return the aggregated report.
///
/// `dumps_dir` must follow the parity-dump layout described in the module
/// docs. Only the stages listed in [`available_stages`] are evaluated; missing
/// stage files are recorded as a soft skip — a stage that cannot be compared
/// is omitted from the report rather than failing the whole run, so partial
/// dump sets (encoder-only or predictor-only) still produce a useful report.
///
/// # Errors
///
/// Returns [`EvalError`] when reading dumps, running the runner, or comparing
/// outputs fails.
#[allow(clippy::needless_pass_by_value)]
pub fn run_parity_eval(
    runner: &mut dyn InferenceRunner,
    inputs: ParityEvalInputs<'_>,
) -> Result<EvalReport, EvalError> {
    use std::time::Instant;

    let ParityEvalInputs {
        pixels,
        history,
        actions,
        history_steps,
        action_dim,
        dumps_dir,
        tolerance,
        backend,
    } = inputs;
    let backend = backend.to_owned();
    let mut stages = Vec::new();
    let mut latency = LatencyStats::default();
    let mut encode_total_ms = 0.0_f64;
    let mut predict_total_ms = 0.0_f64;

    let projector_path = dumps_dir.join("projector/output.safetensors");
    let pred_proj_path = dumps_dir.join("pred_proj/output.safetensors");

    let projector_dump = match load_dump_tensor(&projector_path) {
        Ok(dump) => Some(dump),
        Err(EvalError::MissingDump { .. }) => None,
        Err(error) => return Err(error),
    };
    let pred_proj_dump = match load_dump_tensor(&pred_proj_path) {
        Ok(dump) => Some(dump),
        Err(EvalError::MissingDump { .. }) => None,
        Err(error) => return Err(error),
    };

    if let Some(dump) = projector_dump.as_ref() {
        let start = Instant::now();
        let encoded = runner.encode(pixels).map_err(|source| EvalError::Runner {
            stage: STAGE_PROJECTOR.to_owned(),
            source,
        })?;
        encode_total_ms += start.elapsed().as_secs_f64() * 1000.0;
        latency.encode_calls += 1;

        let reference_slice =
            first_window(&dump.values, encoded.len()).map_err(|()| EvalError::LengthMismatch {
                stage: STAGE_PROJECTOR.to_owned(),
                runner_len: encoded.len(),
                reference_len: dump.values.len(),
            })?;
        stages.push(compare_stage(
            STAGE_PROJECTOR,
            &encoded,
            reference_slice,
            tolerance,
        )?);
    }

    if let Some(dump) = pred_proj_dump.as_ref() {
        let start = Instant::now();
        let predicted = runner
            .predict(history, actions, history_steps, action_dim)
            .map_err(|source| EvalError::Runner {
                stage: STAGE_PRED_PROJ.to_owned(),
                source,
            })?;
        predict_total_ms += start.elapsed().as_secs_f64() * 1000.0;
        latency.predict_calls += 1;

        let reference_slice = first_window(&dump.values, predicted.len()).map_err(|()| {
            EvalError::LengthMismatch {
                stage: STAGE_PRED_PROJ.to_owned(),
                runner_len: predicted.len(),
                reference_len: dump.values.len(),
            }
        })?;
        stages.push(compare_stage(
            STAGE_PRED_PROJ,
            &predicted,
            reference_slice,
            tolerance,
        )?);
    }

    if latency.encode_calls > 0 {
        latency.encode_mean_ms = encode_total_ms / usize_to_f64(latency.encode_calls);
    }
    if latency.predict_calls > 0 {
        latency.predict_mean_ms = predict_total_ms / usize_to_f64(latency.predict_calls);
    }

    let pass = !stages.is_empty() && stages.iter().all(|stage| stage.pass);
    Ok(EvalReport {
        backend,
        reference_dumps: dumps_dir.to_path_buf(),
        tolerance,
        stages,
        latency,
        pass,
    })
}

/// Stage labels reported by [`run_parity_eval`].
#[must_use]
pub fn available_stages() -> &'static [&'static str] {
    &[STAGE_PROJECTOR, STAGE_PRED_PROJ]
}

fn first_window(values: &[f32], window: usize) -> Result<&[f32], ()> {
    if window > values.len() {
        Err(())
    } else {
        Ok(&values[..window])
    }
}

/// Return the latency stats as a plain map for downstream report writers.
#[must_use]
pub fn latency_as_map(latency: &LatencyStats) -> BTreeMap<&'static str, f64> {
    let mut map = BTreeMap::new();
    map.insert("encode_mean_ms", latency.encode_mean_ms);
    map.insert("predict_mean_ms", latency.predict_mean_ms);
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_stage_reports_linf_and_rmse() {
        let actual = [1.0_f32, 2.0, 3.0, 4.0];
        let expected = [1.0_f32, 2.5, 3.0, 3.5];
        let stats = compare_stage("test", &actual, &expected, 0.1).expect("stats");
        assert_eq!(stats.element_count, 4);
        assert!((stats.linf - 0.5).abs() < 1e-6);
        // mean abs = (0 + 0.5 + 0 + 0.5) / 4 = 0.25
        assert!((stats.mean_abs - 0.25).abs() < 1e-6);
        // rmse = sqrt((0 + 0.25 + 0 + 0.25)/4) = sqrt(0.125)
        assert!((stats.rmse - 0.125_f32.sqrt()).abs() < 1e-6);
        assert!(!stats.pass);
        assert!(stats.fraction_above_tolerance > 0.0);
    }

    #[test]
    fn compare_stage_passes_within_tolerance() {
        let actual = [0.0_f32, 0.0, 0.0];
        let expected = [0.0_f32, 0.00005, -0.00005];
        let stats = compare_stage("test", &actual, &expected, 1e-4).expect("stats");
        assert!(stats.pass);
        assert!(stats.fraction_above_tolerance.abs() < 1e-6);
    }

    #[test]
    fn compare_stage_rejects_length_mismatch() {
        let result = compare_stage("test", &[1.0_f32, 2.0], &[1.0_f32], 1e-4);
        assert!(matches!(result, Err(EvalError::LengthMismatch { .. })));
    }
}
