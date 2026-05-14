//! Output-equivalence verifier for exported inference graphs.

use std::fmt::{self, Write as _};

/// Default RFC 0007 export tolerance for the `L_inf` norm.
pub const DEFAULT_L_INF_TOLERANCE: f32 = 1.0e-4;

/// Export strategy selected for CPU inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportStrategy {
    /// Primary ONNX export path.
    Onnx,
    /// Tract-native NNEF fallback path.
    Nnef,
    /// Burn-record-direct fallback path.
    BurnDirect,
}

impl ExportStrategy {
    /// Return the stable strategy name used in model cards.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Onnx => "onnx",
            Self::Nnef => "nnef",
            Self::BurnDirect => "burn-direct",
        }
    }
}

impl fmt::Display for ExportStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Gate for the last-resort Burn-record-direct fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BurnDirectPolicy {
    /// Burn-direct fallback is disabled because no ADR approved it.
    Disabled,
    /// Burn-direct fallback is allowed by a named ADR.
    Enabled {
        /// ADR identifier or path approving Burn-direct fallback.
        adr: String,
    },
}

impl BurnDirectPolicy {
    /// Return a policy that disables Burn-direct fallback.
    pub const fn disabled() -> Self {
        Self::Disabled
    }

    /// Return a policy that enables Burn-direct fallback with an ADR reference.
    pub fn enabled(adr: impl Into<String>) -> Self {
        Self::Enabled { adr: adr.into() }
    }

    fn allow_burn_direct(&self) -> bool {
        matches!(self, Self::Enabled { .. })
    }
}

/// Fixed input used to compare Burn and exported-runner outputs.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedInput {
    /// Input fixture name.
    pub name: String,
    /// Row-major input tensor values.
    pub values: Vec<f32>,
}

impl FixedInput {
    /// Construct a fixed input fixture.
    pub fn new(name: impl Into<String>, values: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }
}

/// Reference Burn forward pass for the verifier.
pub trait BurnForward {
    /// Run the Burn model on a fixed input and return a flat row-major output.
    ///
    /// # Errors
    ///
    /// Returns [`VerifierError`] when the reference forward pass fails.
    fn forward(&self, input: &FixedInput) -> Result<Vec<f32>, VerifierError>;
}

/// Exported inference runner forward pass for the verifier.
pub trait InferenceForward {
    /// Return the export strategy represented by this runner.
    fn strategy(&self) -> ExportStrategy;

    /// Run the exported runner on a fixed input and return a flat row-major output.
    ///
    /// # Errors
    ///
    /// Returns [`VerifierError`] when runner execution fails.
    fn forward(&self, input: &FixedInput) -> Result<Vec<f32>, VerifierError>;
}

/// Successful verifier report.
#[derive(Debug, Clone, PartialEq)]
pub struct VerificationReport {
    /// Strategy that produced the exported-runner output.
    pub strategy: ExportStrategy,
    /// Fixed input fixture name.
    pub input_name: String,
    /// Number of compared output elements.
    pub output_len: usize,
    /// Measured `L_inf` value.
    pub l_inf: f32,
    /// Acceptance tolerance used for the check.
    pub tolerance: f32,
}

/// One fallback-ladder attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct VerificationAttempt {
    /// Strategy attempted or blocked.
    pub strategy: ExportStrategy,
    /// Attempt outcome.
    pub status: VerificationAttemptStatus,
}

/// Status for one verifier attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum VerificationAttemptStatus {
    /// The verifier passed and produced a report.
    Passed(VerificationReport),
    /// The verifier ran and failed with an actionable reason.
    Failed {
        /// Failure reason.
        reason: String,
    },
    /// The strategy was blocked before execution.
    Blocked {
        /// Blocker reason.
        reason: String,
    },
}

/// Final export-strategy decision.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportDecision {
    /// Selected strategy.
    pub selected: ExportStrategy,
    /// Passing verifier report for the selected strategy.
    pub report: VerificationReport,
    /// Ordered attempts made before selection.
    pub attempts: Vec<VerificationAttempt>,
}

impl ExportDecision {
    /// Render the model-card section recording this decision.
    pub fn model_card_section(&self) -> String {
        render_model_card_decision(self)
    }
}

/// Error type for export verification.
#[derive(Debug, Clone, PartialEq)]
pub enum VerifierError {
    /// Burn reference forward failed.
    BurnForwardFailed {
        /// Actionable failure reason.
        reason: String,
    },
    /// Exported runner forward failed.
    RunnerForwardFailed {
        /// Strategy whose runner failed.
        strategy: ExportStrategy,
        /// Actionable failure reason.
        reason: String,
    },
    /// Burn and runner outputs had different flattened lengths.
    OutputLengthMismatch {
        /// Strategy whose output was compared.
        strategy: ExportStrategy,
        /// Burn output length.
        burn_len: usize,
        /// Runner output length.
        runner_len: usize,
    },
    /// A non-finite value was observed.
    NonFiniteOutput {
        /// Strategy whose output was compared.
        strategy: ExportStrategy,
        /// Flat output index.
        index: usize,
    },
    /// `L_inf` exceeded the configured tolerance.
    ToleranceExceeded {
        /// Strategy whose output was compared.
        strategy: ExportStrategy,
        /// Measured `L_inf` value.
        l_inf: f32,
        /// Acceptance tolerance.
        tolerance: f32,
        /// Number of compared output elements.
        output_len: usize,
    },
    /// No export strategy passed verification.
    FallbackExhausted {
        /// Ordered attempt summary.
        attempts: Vec<VerificationAttempt>,
    },
}

impl fmt::Display for VerifierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BurnForwardFailed { reason } => {
                write!(f, "Burn reference forward failed: {reason}")
            },
            Self::RunnerForwardFailed { strategy, reason } => {
                write!(f, "{strategy} runner forward failed: {reason}")
            },
            Self::OutputLengthMismatch {
                strategy,
                burn_len,
                runner_len,
            } => write!(
                f,
                "{strategy} verifier output length mismatch: Burn returned {burn_len}, runner returned {runner_len}"
            ),
            Self::NonFiniteOutput { strategy, index } => write!(
                f,
                "{strategy} verifier saw non-finite output at flat index {index}"
            ),
            Self::ToleranceExceeded {
                strategy,
                l_inf,
                tolerance,
                output_len,
            } => write!(
                f,
                "{strategy} verifier failed: L_inf={l_inf:.8} exceeds tolerance {tolerance:.8} across {output_len} outputs"
            ),
            Self::FallbackExhausted { attempts } => write!(
                f,
                "no export strategy passed verification after {} attempts; inspect failed attempts and enable Burn-direct only with an ADR",
                attempts.len()
            ),
        }
    }
}

impl std::error::Error for VerifierError {}

/// Compare a Burn model against an exported runner using RFC 0007 tolerance.
///
/// # Errors
///
/// Returns [`VerifierError`] when either forward pass fails, output shapes do
/// not match, a non-finite value appears, or `L_inf` exceeds the tolerance.
pub fn verify(
    burn_model: &impl BurnForward,
    runner: &impl InferenceForward,
    fixed_input: &FixedInput,
) -> Result<VerificationReport, VerifierError> {
    verify_with_tolerance(burn_model, runner, fixed_input, DEFAULT_L_INF_TOLERANCE)
}

/// Compare a Burn model against an exported runner using a custom tolerance.
///
/// # Errors
///
/// Returns [`VerifierError`] when either forward pass fails, output shapes do
/// not match, a non-finite value appears, or `L_inf` exceeds the tolerance.
pub fn verify_with_tolerance(
    burn_model: &impl BurnForward,
    runner: &impl InferenceForward,
    fixed_input: &FixedInput,
    tolerance: f32,
) -> Result<VerificationReport, VerifierError> {
    let strategy = runner.strategy();
    let burn = burn_model.forward(fixed_input)?;
    let runner_output = runner.forward(fixed_input)?;
    let l_inf = l_inf_distance(strategy, &burn, &runner_output)?;
    let output_len = burn.len();

    if l_inf > tolerance {
        return Err(VerifierError::ToleranceExceeded {
            strategy,
            l_inf,
            tolerance,
            output_len,
        });
    }

    Ok(VerificationReport {
        strategy,
        input_name: fixed_input.name.clone(),
        output_len,
        l_inf,
        tolerance,
    })
}

/// Select the first export strategy that passes verification.
///
/// The caller supplies candidates in the intended order, normally ONNX, then
/// NNEF, then Burn-direct. Burn-direct is blocked unless an ADR-backed policy
/// explicitly enables it.
///
/// # Errors
///
/// Returns [`VerifierError::FallbackExhausted`] when every candidate fails or is
/// blocked.
pub fn pick_export_strategy(
    burn_model: &impl BurnForward,
    candidates: &[&dyn InferenceForward],
    fixed_input: &FixedInput,
    burn_direct_policy: &BurnDirectPolicy,
) -> Result<ExportDecision, VerifierError> {
    let mut attempts = Vec::new();

    for candidate in candidates {
        let strategy = candidate.strategy();
        if strategy == ExportStrategy::BurnDirect && !burn_direct_policy.allow_burn_direct() {
            attempts.push(VerificationAttempt {
                strategy,
                status: VerificationAttemptStatus::Blocked {
                    reason: "Burn-direct fallback requires an approving ADR per RFC0007-008"
                        .to_owned(),
                },
            });
            continue;
        }

        match verify_trait_object(burn_model, *candidate, fixed_input) {
            Ok(report) => {
                attempts.push(VerificationAttempt {
                    strategy,
                    status: VerificationAttemptStatus::Passed(report.clone()),
                });
                return Ok(ExportDecision {
                    selected: strategy,
                    report,
                    attempts,
                });
            },
            Err(error) => attempts.push(VerificationAttempt {
                strategy,
                status: VerificationAttemptStatus::Failed {
                    reason: error.to_string(),
                },
            }),
        }
    }

    Err(VerifierError::FallbackExhausted { attempts })
}

/// Render the model-card section recording the export decision.
pub fn render_model_card_decision(decision: &ExportDecision) -> String {
    let mut out = String::new();
    out.push_str("## Export verification\n\n");
    let _ = writeln!(
        &mut out,
        "- Selected strategy: `{}`",
        decision.selected.as_str()
    );
    let _ = writeln!(&mut out, "- Fixed input: `{}`", decision.report.input_name);
    let _ = writeln!(
        &mut out,
        "- Output elements: `{}`",
        decision.report.output_len
    );
    let _ = writeln!(
        &mut out,
        "- L_inf: `{:.8}` (tolerance `{:.8}`)\n",
        decision.report.l_inf, decision.report.tolerance
    );
    out.push_str("| Strategy | Status |\n|---|---|\n");
    for attempt in &decision.attempts {
        let status = match &attempt.status {
            VerificationAttemptStatus::Passed(report) => {
                format!("passed, L_inf `{:.8}`", report.l_inf)
            },
            VerificationAttemptStatus::Failed { reason } => format!("failed: {reason}"),
            VerificationAttemptStatus::Blocked { reason } => format!("blocked: {reason}"),
        };
        let _ = writeln!(&mut out, "| `{}` | {} |", attempt.strategy.as_str(), status);
    }
    out
}

fn verify_trait_object(
    burn_model: &impl BurnForward,
    runner: &dyn InferenceForward,
    fixed_input: &FixedInput,
) -> Result<VerificationReport, VerifierError> {
    let strategy = runner.strategy();
    let burn = burn_model.forward(fixed_input)?;
    let runner_output = runner.forward(fixed_input)?;
    let l_inf = l_inf_distance(strategy, &burn, &runner_output)?;
    let output_len = burn.len();

    if l_inf > DEFAULT_L_INF_TOLERANCE {
        return Err(VerifierError::ToleranceExceeded {
            strategy,
            l_inf,
            tolerance: DEFAULT_L_INF_TOLERANCE,
            output_len,
        });
    }

    Ok(VerificationReport {
        strategy,
        input_name: fixed_input.name.clone(),
        output_len,
        l_inf,
        tolerance: DEFAULT_L_INF_TOLERANCE,
    })
}

fn l_inf_distance(
    strategy: ExportStrategy,
    burn: &[f32],
    runner: &[f32],
) -> Result<f32, VerifierError> {
    if burn.len() != runner.len() {
        return Err(VerifierError::OutputLengthMismatch {
            strategy,
            burn_len: burn.len(),
            runner_len: runner.len(),
        });
    }

    let mut max = 0.0_f32;
    for (index, (lhs, rhs)) in burn.iter().zip(runner.iter()).enumerate() {
        if !lhs.is_finite() || !rhs.is_finite() {
            return Err(VerifierError::NonFiniteOutput { strategy, index });
        }
        max = max.max((lhs - rhs).abs());
    }
    Ok(max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct StaticBurn {
        output: Vec<f32>,
    }

    impl BurnForward for StaticBurn {
        fn forward(&self, _input: &FixedInput) -> Result<Vec<f32>, VerifierError> {
            Ok(self.output.clone())
        }
    }

    #[derive(Debug, Clone)]
    struct StaticRunner {
        strategy: ExportStrategy,
        output: Vec<f32>,
    }

    impl InferenceForward for StaticRunner {
        fn strategy(&self) -> ExportStrategy {
            self.strategy
        }

        fn forward(&self, _input: &FixedInput) -> Result<Vec<f32>, VerifierError> {
            Ok(self.output.clone())
        }
    }

    #[test]
    fn verifier_reports_linf_value() -> Result<(), Box<dyn std::error::Error>> {
        let burn = StaticBurn {
            output: vec![1.0, 2.0, 3.0],
        };
        let runner = StaticRunner {
            strategy: ExportStrategy::Onnx,
            output: vec![1.0, 2.0 + 5.0e-5, 3.0],
        };
        let input = FixedInput::new("encoder-fixed", vec![0.0, 1.0]);

        let report = verify(&burn, &runner, &input)?;

        assert_eq!(report.strategy, ExportStrategy::Onnx);
        assert_eq!(report.output_len, 3);
        assert!((report.l_inf - 5.0e-5).abs() < 1.0e-7);
        Ok(())
    }

    #[test]
    fn verifier_failure_is_actionable() -> Result<(), Box<dyn std::error::Error>> {
        let burn = StaticBurn {
            output: vec![1.0, 2.0, 3.0],
        };
        let runner = StaticRunner {
            strategy: ExportStrategy::Onnx,
            output: vec![1.0, 2.5, 3.0],
        };
        let input = FixedInput::new("predictor-fixed", vec![0.0]);

        let error = verify(&burn, &runner, &input)
            .err()
            .ok_or("expected tolerance failure")?;

        assert!(error.to_string().contains("L_inf=0.50000000"));
        assert!(error.to_string().contains("onnx"));
        Ok(())
    }

    #[test]
    fn fallback_picks_nnef_after_onnx_failure() -> Result<(), Box<dyn std::error::Error>> {
        let burn = StaticBurn {
            output: vec![1.0, 2.0, 3.0],
        };
        let onnx = StaticRunner {
            strategy: ExportStrategy::Onnx,
            output: vec![10.0, 2.0, 3.0],
        };
        let nnef = StaticRunner {
            strategy: ExportStrategy::Nnef,
            output: vec![1.0, 2.0 + 1.0e-5, 3.0],
        };
        let input = FixedInput::new("fixed", vec![0.0]);

        let decision = pick_export_strategy(
            &burn,
            &[&onnx, &nnef],
            &input,
            &BurnDirectPolicy::disabled(),
        )?;

        assert_eq!(decision.selected, ExportStrategy::Nnef);
        assert_eq!(decision.attempts.len(), 2);
        assert!(matches!(
            decision.attempts[0].status,
            VerificationAttemptStatus::Failed { .. }
        ));
        assert!(matches!(
            decision.attempts[1].status,
            VerificationAttemptStatus::Passed(_)
        ));
        Ok(())
    }

    #[test]
    fn burn_direct_fallback_requires_adr() -> Result<(), Box<dyn std::error::Error>> {
        let burn = StaticBurn {
            output: vec![1.0, 2.0, 3.0],
        };
        let onnx = StaticRunner {
            strategy: ExportStrategy::Onnx,
            output: vec![10.0, 2.0, 3.0],
        };
        let nnef = StaticRunner {
            strategy: ExportStrategy::Nnef,
            output: vec![1.0, 20.0, 3.0],
        };
        let burn_direct = StaticRunner {
            strategy: ExportStrategy::BurnDirect,
            output: vec![1.0, 2.0, 3.0],
        };
        let input = FixedInput::new("fixed", vec![0.0]);

        let error = pick_export_strategy(
            &burn,
            &[&onnx, &nnef, &burn_direct],
            &input,
            &BurnDirectPolicy::disabled(),
        )
        .err()
        .ok_or("expected exhausted fallback")?;

        let VerifierError::FallbackExhausted { attempts } = error else {
            return Err("expected fallback exhausted error".into());
        };
        assert_eq!(attempts.len(), 3);
        assert!(matches!(
            attempts[2].status,
            VerificationAttemptStatus::Blocked { .. }
        ));
        Ok(())
    }

    #[test]
    fn model_card_decision_records_selected_strategy() -> Result<(), Box<dyn std::error::Error>> {
        let burn = StaticBurn {
            output: vec![1.0, 2.0],
        };
        let onnx = StaticRunner {
            strategy: ExportStrategy::Onnx,
            output: vec![1.0, 2.0],
        };
        let input = FixedInput::new("encoder-fixed", vec![0.0]);

        let decision =
            pick_export_strategy(&burn, &[&onnx], &input, &BurnDirectPolicy::disabled())?;
        let model_card = decision.model_card_section();

        assert!(model_card.contains("Selected strategy: `onnx`"));
        assert!(model_card.contains("L_inf: `0.00000000`"));
        Ok(())
    }

    #[test]
    fn export_verifier_release_smoke() -> Result<(), Box<dyn std::error::Error>> {
        let burn = StaticBurn {
            output: vec![0.25, -0.5, 1.0],
        };
        let onnx = StaticRunner {
            strategy: ExportStrategy::Onnx,
            output: vec![0.25, -0.5, 1.0 + 1.0e-6],
        };
        let input = FixedInput::new("release-smoke", vec![1.0, 2.0, 3.0]);

        let decision =
            pick_export_strategy(&burn, &[&onnx], &input, &BurnDirectPolicy::disabled())?;

        assert_eq!(decision.selected, ExportStrategy::Onnx);
        assert!(decision.report.l_inf <= DEFAULT_L_INF_TOLERANCE);
        Ok(())
    }
}
