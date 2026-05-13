//! Inner training-step primitives.

use core::fmt;

/// RFC 0005 global gradient clipping norm.
pub const DEFAULT_MAX_GRAD_NORM: f64 = 1.0;

/// RFC 0005 `TOL-011` pre-clip gradient norm threshold.
pub const GRAD_EXPLOSION_THRESHOLD: f64 = 1.0e3;

/// Number of consecutive non-finite steps that triggers an abort.
pub const MAX_CONSECUTIVE_NON_FINITE_STEPS: u8 = 3;

/// Error returned by reusable training-step primitives.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StepError {
    /// Gradient accumulation requires at least one micro-batch.
    InvalidGradAccumSteps,
    /// At least one gradient vector is required.
    EmptyGradientSet,
    /// Every micro-batch gradient must have the same flat length.
    GradientShapeMismatch {
        /// Expected flat gradient length.
        expected: usize,
        /// Found flat gradient length.
        found: usize,
    },
    /// Gradient clipping requires a finite non-negative maximum norm.
    InvalidMaxGradNorm,
}

impl fmt::Display for StepError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGradAccumSteps => {
                formatter.write_str("grad_accum_steps must be greater than zero")
            },
            Self::EmptyGradientSet => {
                formatter.write_str("at least one gradient vector is required")
            },
            Self::GradientShapeMismatch { expected, found } => write!(
                formatter,
                "gradient shape mismatch: expected flat length {expected}, found {found}"
            ),
            Self::InvalidMaxGradNorm => {
                formatter.write_str("max_grad_norm must be finite and non-negative")
            },
        }
    }
}

impl std::error::Error for StepError {}

/// Result of global L2 gradient clipping.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipResult {
    /// Pre-clip global L2 norm.
    pub grad_norm_pre: f64,
    /// Post-clip global L2 norm.
    pub grad_norm_post: f64,
    /// Multiplicative factor applied to every gradient element.
    pub scale: f64,
    /// Whether clipping changed the gradient vector.
    pub clipped: bool,
}

/// Artifact kind emitted by step guards.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepArtifactKind {
    /// Non-finite loss or gradient artifact.
    NanDetected,
    /// Large pre-clip gradient norm artifact.
    GradExplosion,
}

/// Structured artifact descriptor produced by a step guard.
#[must_use]
#[derive(Clone, Debug, PartialEq)]
pub struct StepArtifact {
    /// Artifact kind.
    pub kind: StepArtifactKind,
    /// Optimizer step that produced the artifact.
    pub step: u64,
    /// Human-readable reason.
    pub reason: String,
    /// Optional total loss observed at the step.
    pub total_loss: Option<f64>,
    /// Optional pre-clip global gradient norm.
    pub grad_norm_pre: Option<f64>,
    /// Consecutive non-finite step count when relevant.
    pub consecutive_non_finite: Option<u8>,
}

impl StepArtifact {
    /// Returns the RFC 0005 artifact file name.
    pub fn file_name(&self) -> String {
        match self.kind {
            StepArtifactKind::NanDetected => format!("nan_detected_{:07}.json", self.step),
            StepArtifactKind::GradExplosion => format!("grad_explosion_{:07}.json", self.step),
        }
    }

    /// Returns a compact JSON representation suitable for writing to disk.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"kind\":\"{}\",\"step\":{},\"reason\":\"{}\",\"total_loss\":{},\"grad_norm_pre\":{},\"consecutive_non_finite\":{}}}",
            self.kind.as_str(),
            self.step,
            self.reason,
            option_f64_json(self.total_loss),
            option_f64_json(self.grad_norm_pre),
            self.consecutive_non_finite
                .map_or_else(|| "null".to_owned(), |value| value.to_string())
        )
    }
}

impl StepArtifactKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::NanDetected => "nan_detected",
            Self::GradExplosion => "grad_explosion",
        }
    }
}

/// Result of a non-finite guard check.
#[derive(Clone, Debug, PartialEq)]
pub enum NanGuardDecision {
    /// Step is finite and can be applied.
    Apply,
    /// Step should be skipped, but training can continue.
    Skip {
        /// Artifact to write before continuing.
        artifact: StepArtifact,
    },
    /// Training should abort after writing the artifact.
    Abort {
        /// Artifact to write before aborting.
        artifact: StepArtifact,
    },
}

/// Stateful RFC 0005 non-finite guard.
#[must_use]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NanGuard {
    consecutive_non_finite: u8,
    abort_after: u8,
}

impl Default for NanGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl NanGuard {
    /// Creates a guard with the RFC 0005 abort threshold.
    pub const fn new() -> Self {
        Self {
            consecutive_non_finite: 0,
            abort_after: MAX_CONSECUTIVE_NON_FINITE_STEPS,
        }
    }

    /// Creates a guard with an explicit abort threshold.
    pub const fn with_abort_after(abort_after: u8) -> Self {
        Self {
            consecutive_non_finite: 0,
            abort_after,
        }
    }

    /// Returns the current consecutive non-finite count.
    pub const fn consecutive_non_finite(self) -> u8 {
        self.consecutive_non_finite
    }

    /// Observes one step and returns the trainer decision.
    pub fn observe(&mut self, step: u64, total_loss: f64, gradients: &[f64]) -> NanGuardDecision {
        if total_loss.is_finite() && gradients.iter().all(|gradient| gradient.is_finite()) {
            self.consecutive_non_finite = 0;
            return NanGuardDecision::Apply;
        }

        self.consecutive_non_finite = self.consecutive_non_finite.saturating_add(1);
        let artifact = StepArtifact {
            kind: StepArtifactKind::NanDetected,
            step,
            reason: "non-finite loss or gradient".to_owned(),
            total_loss: Some(total_loss),
            grad_norm_pre: None,
            consecutive_non_finite: Some(self.consecutive_non_finite),
        };

        if self.abort_after > 0 && self.consecutive_non_finite >= self.abort_after {
            NanGuardDecision::Abort { artifact }
        } else {
            NanGuardDecision::Skip { artifact }
        }
    }
}

/// Scales a micro-batch loss by `1 / grad_accum_steps`.
///
/// This is the required pre-backward scaling for gradient accumulation.
///
/// # Errors
///
/// Returns [`StepError::InvalidGradAccumSteps`] when `grad_accum_steps` is zero.
pub fn scale_loss_for_accumulation(loss: f64, grad_accum_steps: u32) -> Result<f64, StepError> {
    if grad_accum_steps == 0 {
        return Err(StepError::InvalidGradAccumSteps);
    }

    Ok(loss / f64::from(grad_accum_steps))
}

/// Accumulates already-backpropagated micro-batch gradients as an average.
///
/// # Errors
///
/// Returns [`StepError::InvalidGradAccumSteps`] when `grad_accum_steps` is zero,
/// [`StepError::EmptyGradientSet`] when no micro-batch gradients are provided,
/// or [`StepError::GradientShapeMismatch`] when flat gradient lengths differ.
pub fn accumulate_scaled_gradients(
    microbatch_gradients: &[Vec<f64>],
    grad_accum_steps: u32,
) -> Result<Vec<f64>, StepError> {
    if grad_accum_steps == 0 {
        return Err(StepError::InvalidGradAccumSteps);
    }
    let Some(first) = microbatch_gradients.first() else {
        return Err(StepError::EmptyGradientSet);
    };

    let mut accumulated = vec![0.0; first.len()];
    let scale = 1.0 / f64::from(grad_accum_steps);

    for gradients in microbatch_gradients {
        if gradients.len() != accumulated.len() {
            return Err(StepError::GradientShapeMismatch {
                expected: accumulated.len(),
                found: gradients.len(),
            });
        }

        for (target, gradient) in accumulated.iter_mut().zip(gradients) {
            *target += gradient * scale;
        }
    }

    Ok(accumulated)
}

/// Computes the global flat-vector L2 norm.
pub fn global_l2_norm(gradients: &[f64]) -> f64 {
    gradients
        .iter()
        .map(|gradient| gradient * gradient)
        .sum::<f64>()
        .sqrt()
}

/// Clips gradients by global L2 norm in place.
///
/// # Errors
///
/// Returns [`StepError::InvalidMaxGradNorm`] when `max_grad_norm` is negative
/// or non-finite.
pub fn clip_global_norm(
    gradients: &mut [f64],
    max_grad_norm: f64,
) -> Result<ClipResult, StepError> {
    if !max_grad_norm.is_finite() || max_grad_norm < 0.0 {
        return Err(StepError::InvalidMaxGradNorm);
    }

    let grad_norm_pre = global_l2_norm(gradients);
    let should_clip = grad_norm_pre.is_finite() && grad_norm_pre > max_grad_norm;
    let scale = if should_clip && grad_norm_pre > 0.0 {
        max_grad_norm / grad_norm_pre
    } else {
        1.0
    };

    if should_clip {
        for gradient in &mut *gradients {
            *gradient *= scale;
        }
    }

    let grad_norm_post = global_l2_norm(gradients);
    Ok(ClipResult {
        grad_norm_pre,
        grad_norm_post,
        scale,
        clipped: should_clip,
    })
}

/// Returns a gradient-explosion artifact when the pre-clip norm exceeds `TOL-011`.
pub fn grad_explosion_artifact(step: u64, grad_norm_pre: f64) -> Option<StepArtifact> {
    (grad_norm_pre > GRAD_EXPLOSION_THRESHOLD).then(|| StepArtifact {
        kind: StepArtifactKind::GradExplosion,
        step,
        reason: "pre-clip gradient norm exceeded TOL-011".to_owned(),
        total_loss: None,
        grad_norm_pre: Some(grad_norm_pre),
        consecutive_non_finite: None,
    })
}

fn option_f64_json(value: Option<f64>) -> String {
    value.map_or_else(
        || "null".to_owned(),
        |value| {
            if value.is_finite() {
                value.to_string()
            } else {
                format!("\"{value}\"")
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grad_accumulation_loss_equiv_to_single_pass() -> Result<(), StepError> {
        let microbatch_losses = [6.0, 10.0];
        let scaled_total = microbatch_losses
            .iter()
            .map(|loss| scale_loss_for_accumulation(*loss, 2))
            .sum::<Result<f64, StepError>>()?;
        let accumulated = accumulate_scaled_gradients(&[vec![2.0, 4.0], vec![6.0, 8.0]], 2)?;

        assert_eq!(scaled_total.to_bits(), 8.0_f64.to_bits());
        assert_eq!(accumulated, vec![4.0, 6.0]);
        Ok(())
    }

    #[test]
    fn global_norm_clip_scales_flat_vector() -> Result<(), StepError> {
        let mut gradients = vec![3.0, 4.0];
        let result = clip_global_norm(&mut gradients, DEFAULT_MAX_GRAD_NORM)?;

        assert!((result.grad_norm_pre - 5.0).abs() <= f64::EPSILON);
        assert!((result.grad_norm_post - 1.0).abs() <= f64::EPSILON);
        assert!((result.scale - 0.2).abs() <= f64::EPSILON);
        assert!(result.clipped);
        assert!((gradients[0] - 0.6).abs() <= f64::EPSILON);
        assert!((gradients[1] - 0.8).abs() <= f64::EPSILON);
        Ok(())
    }

    #[test]
    fn finite_step_resets_nan_guard() {
        let mut guard = NanGuard::new();
        let first = guard.observe(1, f64::NAN, &[1.0]);
        let second = guard.observe(2, 1.0, &[1.0, 2.0]);

        assert!(matches!(first, NanGuardDecision::Skip { .. }));
        assert_eq!(second, NanGuardDecision::Apply);
        assert_eq!(guard.consecutive_non_finite(), 0);
    }

    #[test]
    fn three_nan_abort_path_tested() -> Result<(), String> {
        let mut guard = NanGuard::new();

        assert!(matches!(
            guard.observe(1, f64::NAN, &[1.0]),
            NanGuardDecision::Skip { .. }
        ));
        assert!(matches!(
            guard.observe(2, 1.0, &[f64::INFINITY]),
            NanGuardDecision::Skip { .. }
        ));

        let decision = guard.observe(3, f64::NAN, &[f64::NAN]);
        let NanGuardDecision::Abort { artifact } = decision else {
            return Err("third consecutive non-finite step must abort".to_owned());
        };

        assert_eq!(artifact.file_name(), "nan_detected_0000003.json");
        assert_eq!(artifact.consecutive_non_finite, Some(3));
        Ok(())
    }

    #[test]
    fn grad_explosion_artifact_is_emitted_without_abort() -> Result<(), String> {
        let artifact = grad_explosion_artifact(42, GRAD_EXPLOSION_THRESHOLD + 1.0)
            .ok_or_else(|| "expected grad explosion artifact".to_owned())?;

        assert_eq!(artifact.file_name(), "grad_explosion_0000042.json");
        assert!(artifact.to_json().contains("grad_explosion"));
        Ok(())
    }
}
