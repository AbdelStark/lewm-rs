//! Trainer outer-loop state machine and artifact contracts.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Number of steps included in the local smoke run.
pub const SMOKE_STEPS: u64 = 50;

/// First step included in the smoke slope regression.
pub const SMOKE_SLOPE_START_STEP: u64 = 10;

/// Default mini-eval cadence in epochs.
pub const DEFAULT_EVAL_EVERY_N_EPOCHS: u64 = 5;

/// Error returned by trainer state-machine helpers.
#[derive(Debug)]
pub enum TrainerError {
    /// Filesystem operation failed.
    Io {
        /// Path involved in the failed operation.
        path: PathBuf,
        /// Original I/O error.
        source: std::io::Error,
    },
    /// JSON serialization failed.
    Json {
        /// Original JSON error.
        source: serde_json::Error,
    },
    /// State transition is not allowed by RFC 0005.
    InvalidTransition {
        /// Source state.
        from: State,
        /// Destination state.
        to: State,
    },
    /// Transition artifacts require a non-empty transition id.
    EmptyTransitionId,
    /// Smoke slope test did not have enough finite points.
    InsufficientSmokePoints {
        /// Number of usable points.
        found: usize,
    },
    /// Smoke loss point had a non-finite value.
    NonFiniteSmokeLoss {
        /// Step with the invalid loss.
        step: u64,
        /// Loss value.
        loss: f64,
    },
    /// Smoke loss step could not be represented by the checked regression path.
    SmokeStepOutOfRange {
        /// Step outside the expected smoke range.
        step: u64,
    },
}

impl fmt::Display for TrainerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    formatter,
                    "trainer I/O error at {}: {source}",
                    path.display()
                )
            },
            Self::Json { source } => write!(formatter, "trainer JSON error: {source}"),
            Self::InvalidTransition { from, to } => {
                write!(formatter, "invalid trainer transition: {from:?} -> {to:?}")
            },
            Self::EmptyTransitionId => formatter.write_str("transition id cannot be empty"),
            Self::InsufficientSmokePoints { found } => {
                write!(
                    formatter,
                    "smoke slope requires at least 2 points, found {found}"
                )
            },
            Self::NonFiniteSmokeLoss { step, loss } => {
                write!(formatter, "non-finite smoke loss at step {step}: {loss}")
            },
            Self::SmokeStepOutOfRange { step } => {
                write!(formatter, "smoke step outside supported range: {step}")
            },
        }
    }
}

impl std::error::Error for TrainerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source } => Some(source),
            Self::InvalidTransition { .. }
            | Self::EmptyTransitionId
            | Self::InsufficientSmokePoints { .. }
            | Self::NonFiniteSmokeLoss { .. }
            | Self::SmokeStepOutOfRange { .. } => None,
        }
    }
}

impl From<serde_json::Error> for TrainerError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json { source }
    }
}

/// RFC 0005 training pipeline state.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum State {
    /// Load config, construct model/data/optimizer/scheduler, initialize telemetry.
    Init,
    /// Run the fixed-input parity check.
    ParityCheck,
    /// Run the 50-step local smoke test.
    Smoke,
    /// Run learning-rate warmup steps.
    Warmup,
    /// Run steady training epochs.
    Steady,
    /// Run one cooldown epoch at `lr_min`.
    Cooldown,
    /// Run full evaluation.
    Eval,
    /// Upload artifacts and reports.
    Upload,
    /// Final state.
    Done,
}

impl State {
    /// Return the canonical state name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Init => "INIT",
            Self::ParityCheck => "PARITY_CHECK",
            Self::Smoke => "SMOKE",
            Self::Warmup => "WARMUP",
            Self::Steady => "STEADY",
            Self::Cooldown => "COOLDOWN",
            Self::Eval => "EVAL",
            Self::Upload => "UPLOAD",
            Self::Done => "DONE",
        }
    }

    /// Return the next success state.
    pub const fn next_success(self) -> Option<Self> {
        match self {
            Self::Init => Some(Self::ParityCheck),
            Self::ParityCheck => Some(Self::Smoke),
            Self::Smoke => Some(Self::Warmup),
            Self::Warmup => Some(Self::Steady),
            Self::Steady => Some(Self::Cooldown),
            Self::Cooldown => Some(Self::Eval),
            Self::Eval => Some(Self::Upload),
            Self::Upload => Some(Self::Done),
            Self::Done => None,
        }
    }
}

/// Mutable state-machine cursor for the trainer outer loop.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainerStateMachine {
    /// Current trainer state.
    pub state: State,
    /// Current optimizer step.
    pub step: u64,
    /// Current epoch.
    pub epoch: u64,
}

impl TrainerStateMachine {
    /// Create a fresh trainer state machine at `INIT`.
    pub const fn fresh() -> Self {
        Self {
            state: State::Init,
            step: 0,
            epoch: 0,
        }
    }

    /// Create a state machine from a resumed step/epoch in `STEADY`.
    pub const fn resumed(step: u64, epoch: u64) -> Self {
        Self {
            state: State::Steady,
            step,
            epoch,
        }
    }

    /// Advance through the next success transition.
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError::InvalidTransition`] when called from `DONE`.
    pub fn advance_success(
        &mut self,
        wall_time_s: f64,
        checkpoint_written: bool,
    ) -> Result<TransitionRecord, TrainerError> {
        let Some(to) = self.state.next_success() else {
            return Err(TrainerError::InvalidTransition {
                from: State::Done,
                to: State::Done,
            });
        };
        self.transition_to(to, wall_time_s, checkpoint_written)
    }

    /// Transition to an explicit destination state.
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError::InvalidTransition`] when `from -> to` is outside
    /// the RFC 0005 success graph.
    pub fn transition_to(
        &mut self,
        to: State,
        wall_time_s: f64,
        checkpoint_written: bool,
    ) -> Result<TransitionRecord, TrainerError> {
        let from = self.state;
        if from.next_success() != Some(to) {
            return Err(TrainerError::InvalidTransition { from, to });
        }
        self.state = to;
        Ok(TransitionRecord {
            from,
            to,
            step: self.step,
            epoch: self.epoch,
            wall_time_s,
            checkpoint_written,
        })
    }
}

/// Transition JSON payload written as `transition_<ts>.json`.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct TransitionRecord {
    /// Source state.
    pub from: State,
    /// Destination state.
    pub to: State,
    /// Optimizer step at transition time.
    pub step: u64,
    /// Epoch at transition time.
    pub epoch: u64,
    /// Wall-clock seconds spent in the source state.
    pub wall_time_s: f64,
    /// Whether the required checkpoint was written for this transition.
    pub checkpoint_written: bool,
}

/// Per-epoch parity probe JSON payload.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ParityProbeReport {
    /// Optimizer step at which the probe was written.
    pub step: u64,
    /// Encoder CLS-vector L-infinity drift.
    pub encoder_cls_l_inf: f64,
    /// Predictor-output L-infinity drift.
    pub predictor_l_inf: f64,
    /// `SIGReg` scalar value observed by the probe.
    pub sigreg_value: f64,
}

/// Smoke-test loss observation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmokeLossPoint {
    /// Optimizer step.
    pub step: u64,
    /// Total loss at the step.
    pub loss: f64,
}

/// Return whether `from -> to` is an RFC 0005 success transition.
pub const fn is_valid_success_transition(from: State, to: State) -> bool {
    matches!(
        (from, to),
        (State::Init, State::ParityCheck)
            | (State::ParityCheck, State::Smoke)
            | (State::Smoke, State::Warmup)
            | (State::Warmup, State::Steady)
            | (State::Steady, State::Cooldown)
            | (State::Cooldown, State::Eval)
            | (State::Eval, State::Upload)
            | (State::Upload, State::Done)
    )
}

/// Return `transition_<id>.json`.
///
/// # Errors
///
/// Returns [`TrainerError::EmptyTransitionId`] when `transition_id` is empty.
pub fn transition_file_name(transition_id: &str) -> Result<String, TrainerError> {
    if transition_id.is_empty() {
        return Err(TrainerError::EmptyTransitionId);
    }
    Ok(format!("transition_{transition_id}.json"))
}

/// Write a transition artifact.
///
/// # Errors
///
/// Returns an error if `transition_id` is empty, the output directory cannot be
/// created, or the JSON artifact cannot be written.
pub fn write_transition_record(
    output_dir: impl AsRef<Path>,
    transition_id: &str,
    record: &TransitionRecord,
) -> Result<PathBuf, TrainerError> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir).map_err(|source| io_error(output_dir, source))?;
    let path = output_dir.join(transition_file_name(transition_id)?);
    let bytes = serde_json::to_vec_pretty(record)?;
    fs::write(&path, bytes).map_err(|source| io_error(&path, source))?;
    Ok(path)
}

/// Return `step_{N:07}.parity.json`.
pub fn parity_probe_file_name(step: u64) -> String {
    format!("step_{step:07}.parity.json")
}

/// Write a per-epoch parity probe artifact.
///
/// # Errors
///
/// Returns an error if the output directory cannot be created or the JSON
/// artifact cannot be written.
pub fn write_parity_probe(
    output_dir: impl AsRef<Path>,
    report: &ParityProbeReport,
) -> Result<PathBuf, TrainerError> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir).map_err(|source| io_error(output_dir, source))?;
    let path = output_dir.join(parity_probe_file_name(report.step));
    let bytes = serde_json::to_vec_pretty(report)?;
    fs::write(&path, bytes).map_err(|source| io_error(&path, source))?;
    Ok(path)
}

/// Compute the best-fit linear slope for smoke losses over steps 10..50.
///
/// # Errors
///
/// Returns an error when fewer than two usable points are present or any smoke
/// loss point is non-finite.
pub fn smoke_loss_slope(points: &[SmokeLossPoint]) -> Result<f64, TrainerError> {
    let selected = points
        .iter()
        .copied()
        .filter(|point| (SMOKE_SLOPE_START_STEP..=SMOKE_STEPS).contains(&point.step))
        .collect::<Vec<_>>();

    if selected.len() < 2 {
        return Err(TrainerError::InsufficientSmokePoints {
            found: selected.len(),
        });
    }
    for point in &selected {
        if !point.loss.is_finite() {
            return Err(TrainerError::NonFiniteSmokeLoss {
                step: point.step,
                loss: point.loss,
            });
        }
    }

    let count = selected.iter().map(|_point| 1.0).sum::<f64>();
    let mean_step = selected
        .iter()
        .map(|point| smoke_step_as_f64(point.step))
        .sum::<Result<f64, TrainerError>>()?
        / count;
    let mean_loss = selected.iter().map(|point| point.loss).sum::<f64>() / count;
    let numerator = selected
        .iter()
        .map(|point| {
            smoke_step_as_f64(point.step).map(|step| (step - mean_step) * (point.loss - mean_loss))
        })
        .sum::<Result<f64, TrainerError>>()?;
    let denominator = selected
        .iter()
        .map(|point| smoke_step_as_f64(point.step).map(|step| (step - mean_step).powi(2)))
        .sum::<Result<f64, TrainerError>>()?;

    Ok(numerator / denominator)
}

/// Return whether the smoke loss slope satisfies RFC 0005 §8.3.
///
/// # Errors
///
/// Returns errors from [`smoke_loss_slope`].
pub fn smoke_loss_decreases(points: &[SmokeLossPoint]) -> Result<bool, TrainerError> {
    Ok(smoke_loss_slope(points)? < 0.0)
}

/// Return whether mini-eval should run at `epoch`.
pub const fn should_run_eval(epoch: u64, eval_every_n_epochs: u64) -> bool {
    eval_every_n_epochs > 0 && epoch > 0 && epoch % eval_every_n_epochs == 0
}

fn io_error(path: impl Into<PathBuf>, source: std::io::Error) -> TrainerError {
    TrainerError::Io {
        path: path.into(),
        source,
    }
}

fn smoke_step_as_f64(step: u64) -> Result<f64, TrainerError> {
    u32::try_from(step)
        .map(f64::from)
        .map_err(|_| TrainerError::SmokeStepOutOfRange { step })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn state_machine_follows_nine_state_pipeline_and_writes_transition_json()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = TestDir::new("transitions")?;
        let mut machine = TrainerStateMachine::fresh();
        let expected = [
            State::ParityCheck,
            State::Smoke,
            State::Warmup,
            State::Steady,
            State::Cooldown,
            State::Eval,
            State::Upload,
            State::Done,
        ];

        for (index, state) in expected.iter().enumerate() {
            let wall_time_s = f64::from(u32::try_from(index)?);
            let record = machine.advance_success(wall_time_s, true)?;
            assert_eq!(record.to, *state);
            assert!(record.checkpoint_written);
            let path = write_transition_record(dir.path(), &format!("{index:02}"), &record)?;
            assert!(path.is_file());
        }

        assert_eq!(machine.state, State::Done);
        assert_eq!(fs::read_dir(dir.path())?.count(), expected.len());
        Ok(())
    }

    #[test]
    fn parity_probe_artifact_written() -> Result<(), Box<dyn std::error::Error>> {
        let dir = TestDir::new("parity")?;
        let report = ParityProbeReport {
            step: 144,
            encoder_cls_l_inf: 7.2e-5,
            predictor_l_inf: 9.5e-5,
            sigreg_value: 0.00731,
        };

        let path = write_parity_probe(dir.path(), &report)?;
        let raw = fs::read_to_string(&path)?;

        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("step_0000144.parity.json")
        );
        assert!(raw.contains("encoder_cls_l_inf"));
        Ok(())
    }

    #[test]
    fn local_smoke_loss_decreases() -> Result<(), Box<dyn std::error::Error>> {
        let smoke_steps = u32::try_from(SMOKE_STEPS)?;
        let decreasing = (1_u32..=smoke_steps)
            .map(|step| SmokeLossPoint {
                step: u64::from(step),
                loss: 100.0 - f64::from(step),
            })
            .collect::<Vec<_>>();
        let increasing = (1_u32..=smoke_steps)
            .map(|step| SmokeLossPoint {
                step: u64::from(step),
                loss: f64::from(step),
            })
            .collect::<Vec<_>>();

        assert!(smoke_loss_decreases(&decreasing)?);
        assert!(!smoke_loss_decreases(&increasing)?);
        Ok(())
    }

    #[test]
    fn eval_trigger_every_n_epochs() {
        assert!(!should_run_eval(0, DEFAULT_EVAL_EVERY_N_EPOCHS));
        assert!(!should_run_eval(4, DEFAULT_EVAL_EVERY_N_EPOCHS));
        assert!(should_run_eval(5, DEFAULT_EVAL_EVERY_N_EPOCHS));
        assert!(should_run_eval(10, DEFAULT_EVAL_EVERY_N_EPOCHS));
        assert!(!should_run_eval(10, 0));
    }

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(name: &str) -> Result<Self, Box<dyn std::error::Error>> {
            let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
            let path = std::env::temp_dir().join(format!(
                "lewm-train-trainer-{name}-{}-{}",
                process::id(),
                now.as_nanos()
            ));
            fs::create_dir(&path)?;
            Ok(Self { path })
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            if let Err(_err) = fs::remove_dir_all(&self.path) {}
        }
    }
}
