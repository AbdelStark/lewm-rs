//! Collapse detector probes for encoder CLS embeddings.
//!
//! The probe math is backend-neutral and consumes the already extracted
//! `last_hidden_state[:, 0, :]` CLS matrix. The trainer is responsible for
//! running the encoder with no-grad semantics before calling this helper.

use serde::{Deserialize, Serialize};

use crate::LewmCoreError;

/// RFC TOL-007 lower bound for per-dimension CLS variance.
pub const CLS_VAR_FLOOR: f32 = 0.05;

/// RFC TOL-008 upper bound for mean absolute CLS activation.
pub const CLS_MEAN_ABS_CEILING: f32 = 5.0;

/// RFC TOL-009 upper bound for mean pairwise CLS cosine.
pub const CLS_COSINE_PAIR_CEILING: f32 = 0.85;

/// Thresholds used to trip the collapse detector.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CollapseThresholds {
    /// Mean absolute CLS activation ceiling.
    pub mean_abs_cls_ceiling: f32,
    /// Mean per-dimension CLS variance floor.
    pub cls_variance_per_dim_floor: f32,
    /// Mean pairwise CLS cosine ceiling.
    pub mean_pairwise_cosine_ceiling: f32,
}

impl Default for CollapseThresholds {
    fn default() -> Self {
        Self {
            mean_abs_cls_ceiling: CLS_MEAN_ABS_CEILING,
            cls_variance_per_dim_floor: CLS_VAR_FLOOR,
            mean_pairwise_cosine_ceiling: CLS_COSINE_PAIR_CEILING,
        }
    }
}

/// Collapse detector metrics for a held-out CLS probe batch.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CollapseProbe {
    /// Mean absolute value across batch and feature dimensions.
    pub mean_abs_cls: f32,
    /// Mean over feature dimensions of the batch variance.
    pub cls_variance_per_dim_mean: f32,
    /// Mean cosine over unordered distinct CLS pairs.
    pub mean_pairwise_cosine: f32,
}

/// Per-threshold collapse trip flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollapseTrip {
    /// `true` when `mean_abs_cls` exceeds `mean_abs_cls_ceiling`.
    pub mean_abs_cls: bool,
    /// `true` when `cls_variance_per_dim_mean` is below `cls_variance_per_dim_floor`.
    pub cls_variance_per_dim_mean: bool,
    /// `true` when `mean_pairwise_cosine` exceeds `mean_pairwise_cosine_ceiling`.
    pub mean_pairwise_cosine: bool,
}

impl CollapseTrip {
    /// Return `true` when any collapse threshold tripped.
    pub fn any(self) -> bool {
        self.mean_abs_cls || self.cls_variance_per_dim_mean || self.mean_pairwise_cosine
    }
}

/// Collapse probe output plus the threshold decision.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CollapseProbeResult {
    /// Computed probe metrics.
    pub probe: CollapseProbe,
    /// Thresholds used for the trip decision.
    pub thresholds: CollapseThresholds,
    /// Per-threshold trip flags.
    pub trip: CollapseTrip,
}

impl CollapseProbeResult {
    /// Return `true` when any collapse threshold tripped.
    pub fn is_collapsed(&self) -> bool {
        self.trip.any()
    }
}

/// Run the collapse detector with RFC TOL-007/008/009 thresholds.
///
/// The `cls` buffer must be row-major `(batch, dim)` CLS embeddings extracted
/// from `last_hidden_state[:, 0, :]` under no-grad execution.
///
/// # Errors
///
/// Returns [`LewmCoreError::InvalidTensorOp`] when `batch < 2`, `dim == 0`,
/// the buffer length is not `batch * dim`, the shape overflows, or any CLS
/// value is not finite.
pub fn run_collapse_probe(
    cls: &[f32],
    batch: usize,
    dim: usize,
) -> Result<CollapseProbeResult, LewmCoreError> {
    run_collapse_probe_with_thresholds(cls, batch, dim, CollapseThresholds::default())
}

/// Run the collapse detector with explicit thresholds.
///
/// The variance uses the unbiased batch estimator to mirror the common
/// `var(cls, dim=batch)` tensor default. The pairwise cosine mean uses
/// unordered, non-self pairs. A pair of zero vectors is treated as cosine `1.0`
/// because identical zero CLS vectors are a collapsed encoder signature.
///
/// # Errors
///
/// Returns [`LewmCoreError::InvalidTensorOp`] when the shape, values, or
/// thresholds are not valid for the collapse probe.
pub fn run_collapse_probe_with_thresholds(
    cls: &[f32],
    batch: usize,
    dim: usize,
    thresholds: CollapseThresholds,
) -> Result<CollapseProbeResult, LewmCoreError> {
    validate_cls(cls, batch, dim)?;
    validate_thresholds(thresholds)?;

    let probe = CollapseProbe {
        mean_abs_cls: mean_abs(cls),
        cls_variance_per_dim_mean: mean_variance_per_dim(cls, batch, dim),
        mean_pairwise_cosine: mean_pairwise_cosine(cls, batch, dim),
    };
    let trip = CollapseTrip {
        mean_abs_cls: probe.mean_abs_cls > thresholds.mean_abs_cls_ceiling,
        cls_variance_per_dim_mean: probe.cls_variance_per_dim_mean
            < thresholds.cls_variance_per_dim_floor,
        mean_pairwise_cosine: probe.mean_pairwise_cosine > thresholds.mean_pairwise_cosine_ceiling,
    };

    Ok(CollapseProbeResult {
        probe,
        thresholds,
        trip,
    })
}

fn validate_cls(cls: &[f32], batch: usize, dim: usize) -> Result<(), LewmCoreError> {
    if batch < 2 {
        return Err(LewmCoreError::InvalidTensorOp {
            reason: "collapse probe batch must contain at least two CLS rows".to_owned(),
        });
    }
    if dim == 0 {
        return Err(LewmCoreError::InvalidTensorOp {
            reason: "collapse probe dim must be non-zero".to_owned(),
        });
    }

    let expected_len = batch
        .checked_mul(dim)
        .ok_or_else(|| LewmCoreError::InvalidTensorOp {
            reason: "collapse probe element count overflowed usize".to_owned(),
        })?;
    if cls.len() != expected_len {
        return Err(LewmCoreError::InvalidShape {
            expected: vec![batch, dim],
            found: vec![cls.len()],
        });
    }

    if cls.iter().any(|value| !value.is_finite()) {
        return Err(LewmCoreError::InvalidTensorOp {
            reason: "collapse probe CLS values must be finite".to_owned(),
        });
    }

    Ok(())
}

fn validate_thresholds(thresholds: CollapseThresholds) -> Result<(), LewmCoreError> {
    if !thresholds.mean_abs_cls_ceiling.is_finite()
        || thresholds.mean_abs_cls_ceiling < 0.0
        || !thresholds.cls_variance_per_dim_floor.is_finite()
        || thresholds.cls_variance_per_dim_floor < 0.0
        || !thresholds.mean_pairwise_cosine_ceiling.is_finite()
        || !(-1.0..=1.0).contains(&thresholds.mean_pairwise_cosine_ceiling)
    {
        return Err(LewmCoreError::InvalidTensorOp {
            reason: "collapse probe thresholds must be finite and within metric domains".to_owned(),
        });
    }

    Ok(())
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn mean_abs(cls: &[f32]) -> f32 {
    let sum = cls.iter().map(|value| f64::from(value.abs())).sum::<f64>();
    (sum / cls.len() as f64) as f32
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn mean_variance_per_dim(cls: &[f32], batch: usize, dim: usize) -> f32 {
    let mut variance_sum = 0.0f64;
    for feature in 0..dim {
        let mean = (0..batch)
            .map(|row| f64::from(cls[row * dim + feature]))
            .sum::<f64>()
            / batch as f64;
        let variance = (0..batch)
            .map(|row| {
                let centered = f64::from(cls[row * dim + feature]) - mean;
                centered * centered
            })
            .sum::<f64>()
            / (batch - 1) as f64;
        variance_sum += variance;
    }

    (variance_sum / dim as f64) as f32
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn mean_pairwise_cosine(cls: &[f32], batch: usize, dim: usize) -> f32 {
    let mut cosine_sum = 0.0f64;
    let mut pair_count = 0usize;

    for left in 0..batch {
        for right in (left + 1)..batch {
            cosine_sum += pairwise_cosine(cls, left, right, dim);
            pair_count += 1;
        }
    }

    (cosine_sum / pair_count as f64) as f32
}

fn pairwise_cosine(cls: &[f32], left: usize, right: usize, dim: usize) -> f64 {
    let left_offset = left * dim;
    let right_offset = right * dim;
    let mut dot = 0.0f64;
    let mut left_norm_sq = 0.0f64;
    let mut right_norm_sq = 0.0f64;

    for feature in 0..dim {
        let left_value = f64::from(cls[left_offset + feature]);
        let right_value = f64::from(cls[right_offset + feature]);
        dot += left_value * right_value;
        left_norm_sq += left_value * left_value;
        right_norm_sq += right_value * right_value;
    }

    match (left_norm_sq == 0.0, right_norm_sq == 0.0) {
        (true, true) => 1.0,
        (true, false) | (false, true) => 0.0,
        (false, false) => dot / (left_norm_sq.sqrt() * right_norm_sq.sqrt()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_probe_on_synthetic_collapsed_encoder() {
        let cls = vec![1.0; 4 * 3];

        let result = run_collapse_probe(&cls, 4, 3).expect("valid collapse probe");

        assert!(result.is_collapsed());
        assert!(!result.trip.mean_abs_cls);
        assert!(result.trip.cls_variance_per_dim_mean);
        assert!(result.trip.mean_pairwise_cosine);
        assert_close(result.probe.mean_abs_cls, 1.0);
        assert_close(result.probe.cls_variance_per_dim_mean, 0.0);
        assert_close(result.probe.mean_pairwise_cosine, 1.0);
    }

    #[test]
    fn collapse_probe_on_zero_vectors_trips_without_nan_cosine() {
        let cls = vec![0.0; 4 * 3];

        let result = run_collapse_probe(&cls, 4, 3).expect("valid collapse probe");

        assert!(result.is_collapsed());
        assert!(result.trip.cls_variance_per_dim_mean);
        assert!(result.trip.mean_pairwise_cosine);
        assert_close(result.probe.mean_pairwise_cosine, 1.0);
    }

    #[test]
    fn collapse_probe_on_synthetic_healthy_encoder() {
        let cls = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            -1.0, 0.0, 0.0, 0.0, //
            0.0, -1.0, 0.0, 0.0,
        ];

        let result = run_collapse_probe(&cls, 4, 4).expect("valid collapse probe");

        assert!(!result.is_collapsed());
        assert!(!result.trip.mean_abs_cls);
        assert!(!result.trip.cls_variance_per_dim_mean);
        assert!(!result.trip.mean_pairwise_cosine);
        assert_close(result.probe.mean_abs_cls, 0.25);
        assert_close(result.probe.cls_variance_per_dim_mean, 1.0 / 3.0);
        assert_close(result.probe.mean_pairwise_cosine, -1.0 / 3.0);
    }

    #[test]
    fn collapse_probe_threshold_edges_do_not_trip() {
        let thresholds = CollapseThresholds {
            mean_abs_cls_ceiling: 0.5,
            cls_variance_per_dim_floor: 2.0,
            mean_pairwise_cosine_ceiling: 0.0,
        };
        let cls = [2.0, 0.0, -2.0, 0.0];

        let result =
            run_collapse_probe_with_thresholds(&cls, 2, 2, thresholds).expect("valid probe");

        assert_close(result.probe.mean_abs_cls, 1.0);
        assert_close(result.probe.cls_variance_per_dim_mean, 4.0);
        assert_close(result.probe.mean_pairwise_cosine, -1.0);
        assert!(result.trip.mean_abs_cls);
        assert!(!result.trip.cls_variance_per_dim_mean);
        assert!(!result.trip.mean_pairwise_cosine);
    }

    #[test]
    fn collapse_probe_rejects_invalid_shape_and_values() {
        assert!(run_collapse_probe(&[0.0], 1, 1).is_err());
        assert!(run_collapse_probe(&[0.0, 1.0], 2, 0).is_err());
        assert!(run_collapse_probe(&[0.0, 1.0], 2, 2).is_err());
        assert!(run_collapse_probe(&[0.0, f32::NAN], 2, 1).is_err());
    }

    #[test]
    fn collapse_probe_rejects_invalid_thresholds() {
        let cls = [0.0, 1.0];
        let thresholds = CollapseThresholds {
            mean_abs_cls_ceiling: f32::INFINITY,
            ..CollapseThresholds::default()
        };

        assert!(run_collapse_probe_with_thresholds(&cls, 2, 1, thresholds).is_err());
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "actual={actual} expected={expected}"
        );
    }
}
