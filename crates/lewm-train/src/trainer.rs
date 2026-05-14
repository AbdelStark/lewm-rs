//! Trainer outer-loop state machine and artifact contracts.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::checkpoint::{
    CheckpointError, CheckpointPaths, CheckpointRngState, CheckpointWriteRequest, ParameterTensor,
    ParityProbe, save_checkpoint,
};
use crate::config::{DatasetConfig, DatasetSplit, RootConfig, TrainingConfig};
use crate::optim::OptimConfig;
use crate::pusht_lewm::{
    PUSHT_ACTION_DIM, PUSHT_MINIMAL_LEWM_LATENT_DIM, PUSHT_MINIMAL_LEWM_PARAM_COUNT,
    PUSHT_MINIMAL_LEWM_PARAMETER_SPECS, PushtMinimalLewmCore, PushtMinimalLewmExample,
    PushtMinimalLewmFeatures, loss_and_gradients, parameter_spec_for_flat_index,
};
use crate::schedule::CosineWarmup;
use crate::step::{
    DEFAULT_MAX_GRAD_NORM, NanGuard, NanGuardDecision, StepError, accumulate_scaled_gradients,
    clip_global_norm, grad_explosion_artifact, scale_loss_for_accumulation,
};
use lewm_data::{
    DataError, PushtConfig as DataPushtConfig, PushtDataset, Sample as PushtSample,
    SampleMeta as PushtSampleMeta, Split as DataSplit,
};
use serde::{Deserialize, Serialize};

/// Number of steps included in the local smoke run.
pub const SMOKE_STEPS: u64 = 50;

/// First step included in the smoke slope regression.
pub const SMOKE_SLOPE_START_STEP: u64 = 10;

/// Default mini-eval cadence in epochs.
pub const DEFAULT_EVAL_EVERY_N_EPOCHS: u64 = 5;

const SMOKE_LR_PEAK: f64 = 0.05;
const SMOKE_LR_MIN: f64 = 0.01;
const SMOKE_MAX_BATCH_SIZE: u64 = 1_024;
const PUSHT_FIXTURE_FRAME_SIZE: usize = 16;
const PUSHT_FIXTURE_SAMPLE_COUNT: usize = 128;

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
    /// Checkpoint persistence failed.
    Checkpoint {
        /// Original checkpoint error.
        source: CheckpointError,
    },
    /// Inner step primitive failed.
    Step {
        /// Original step error.
        source: StepError,
    },
    /// Dataset loading or sampling failed.
    Data {
        /// Original data error.
        source: DataError,
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
    /// Smoke command received an invalid step count.
    InvalidSmokeSteps {
        /// Requested smoke steps.
        steps: u64,
    },
    /// Smoke command received an invalid batch size.
    InvalidSmokeBatchSize {
        /// Requested smoke batch size.
        batch_size: u64,
    },
    /// The smoke guard rejected a synthetic step.
    SmokeGuardRejected {
        /// Rejected optimizer step.
        step: u64,
        /// Guard reason.
        reason: String,
    },
    /// `train` currently requires a bounded max-step run.
    TrainRequiresMaxSteps,
    /// The selected config is not the supported `PushT` training path.
    UnsupportedTrainDataset {
        /// Dataset kind from the root config.
        kind: String,
    },
    /// A caller-provided training data path is missing.
    MissingTrainDataPath {
        /// Missing data root or shard.
        path: PathBuf,
    },
    /// Train command received an invalid step count.
    InvalidTrainSteps {
        /// Requested training steps.
        steps: u64,
    },
    /// Train command received an invalid batch size.
    InvalidTrainBatchSize {
        /// Requested training batch size.
        batch_size: usize,
    },
    /// The train guard rejected a `PushT` tiny-JEPA step.
    TrainGuardRejected {
        /// Rejected optimizer step.
        step: u64,
        /// Guard reason.
        reason: String,
    },
    /// The bounded `PushT` model path does not yet restore checkpoints.
    TrainResumeUnsupported,
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
            Self::Checkpoint { source } => write!(formatter, "trainer checkpoint error: {source}"),
            Self::Step { source } => write!(formatter, "trainer step error: {source}"),
            Self::Data { source } => write!(formatter, "trainer data error: {source}"),
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
            Self::InvalidSmokeSteps { steps } => {
                write!(
                    formatter,
                    "smoke steps must be greater than zero, found {steps}"
                )
            },
            Self::InvalidSmokeBatchSize { batch_size } => write!(
                formatter,
                "smoke batch size must be in 1..={SMOKE_MAX_BATCH_SIZE}, found {batch_size}"
            ),
            Self::SmokeGuardRejected { step, reason } => {
                write!(formatter, "smoke guard rejected step {step}: {reason}")
            },
            Self::TrainRequiresMaxSteps => formatter
                .write_str("lewm-train train requires --max-steps until full training lands"),
            Self::UnsupportedTrainDataset { kind } => {
                write!(formatter, "unsupported train dataset kind: {kind}")
            },
            Self::MissingTrainDataPath { path } => {
                write!(
                    formatter,
                    "training data path does not exist: {}",
                    path.display()
                )
            },
            Self::InvalidTrainSteps { steps } => {
                write!(
                    formatter,
                    "train max-steps must be greater than zero, found {steps}"
                )
            },
            Self::InvalidTrainBatchSize { batch_size } => write!(
                formatter,
                "train batch size must be in 1..={}, found {batch_size}",
                u32::MAX
            ),
            Self::TrainGuardRejected { step, reason } => {
                write!(formatter, "train guard rejected step {step}: {reason}")
            },
            Self::TrainResumeUnsupported => formatter.write_str(
                "lewm-train train --resume-if-present is not supported for pusht-minimal-lewm yet; start a fresh output directory",
            ),
        }
    }
}

impl std::error::Error for TrainerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source } => Some(source),
            Self::Checkpoint { source } => Some(source),
            Self::Step { source } => Some(source),
            Self::Data { source } => Some(source),
            Self::InvalidTransition { .. }
            | Self::EmptyTransitionId
            | Self::InsufficientSmokePoints { .. }
            | Self::NonFiniteSmokeLoss { .. }
            | Self::SmokeStepOutOfRange { .. }
            | Self::InvalidSmokeSteps { .. }
            | Self::InvalidSmokeBatchSize { .. }
            | Self::SmokeGuardRejected { .. }
            | Self::TrainRequiresMaxSteps
            | Self::UnsupportedTrainDataset { .. }
            | Self::MissingTrainDataPath { .. }
            | Self::InvalidTrainSteps { .. }
            | Self::InvalidTrainBatchSize { .. }
            | Self::TrainGuardRejected { .. }
            | Self::TrainResumeUnsupported => None,
        }
    }
}

impl From<serde_json::Error> for TrainerError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json { source }
    }
}

impl From<CheckpointError> for TrainerError {
    fn from(source: CheckpointError) -> Self {
        Self::Checkpoint { source }
    }
}

impl From<StepError> for TrainerError {
    fn from(source: StepError) -> Self {
        Self::Step { source }
    }
}

impl From<DataError> for TrainerError {
    fn from(source: DataError) -> Self {
        Self::Data { source }
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
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct SmokeLossPoint {
    /// Optimizer step.
    pub step: u64,
    /// Total loss at the step.
    pub loss: f64,
    /// Pre-clip global gradient norm.
    pub grad_norm_pre: f64,
    /// Post-clip global gradient norm.
    pub grad_norm_post: f64,
    /// Learning rate used for the parameter update.
    pub learning_rate: f64,
}

/// Smoke run report written by the CLI smoke command.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SmokeRunReport {
    /// Report schema version.
    pub schema_version: String,
    /// Artifact kind.
    pub kind: String,
    /// Hash of the canonical loaded config.
    pub config_hash: String,
    /// Optional dataset directory passed to the smoke command.
    pub data_dir: Option<String>,
    /// Output directory used for this run.
    pub output_dir: String,
    /// Requested smoke steps.
    pub steps: u64,
    /// Requested batch size.
    pub batch_size: u64,
    /// Run seed.
    pub seed: u64,
    /// Requested device string.
    pub device: String,
    /// Best-fit loss slope over the canonical smoke window.
    pub loss_slope: f64,
    /// Whether the deterministic smoke losses decrease.
    pub loss_decreased: bool,
    /// First recorded loss.
    pub initial_loss: f64,
    /// Last recorded loss.
    pub final_loss: f64,
    /// Optimizer step saved as a checkpoint.
    pub checkpoint_step: u64,
    /// Whether the smoke checkpoint contains all required files.
    pub checkpoint_complete: bool,
    /// Smoke checkpoint files written beside the report.
    pub checkpoint_files: Vec<String>,
    /// Number of gradient-explosion events observed by the guard.
    pub grad_explosion_events: u64,
    /// Explicitly scopes this artifact while the full JEPA loop is pending.
    pub mode: String,
    /// Deterministic loss observations.
    pub losses: Vec<SmokeLossPoint>,
}

/// Train loss observation written by the bounded `PushT` train path.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct TrainLossPoint {
    /// Optimizer step.
    pub step: u64,
    /// Mean training loss at the step.
    pub loss: f64,
    /// Mean latent prediction loss at the step.
    pub pred_loss: f64,
    /// Mean latent scale regularization loss at the step.
    pub sigreg_proxy_loss: f64,
    /// Pre-clip global gradient norm.
    pub grad_norm_pre: f64,
    /// Post-clip global gradient norm.
    pub grad_norm_post: f64,
    /// Learning rate used for the parameter update.
    pub learning_rate: f64,
    /// Cumulative samples consumed by the run.
    pub samples_seen: u64,
}

/// Report written by the bounded `PushT` train command.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct TrainRunReport {
    /// Report schema version.
    pub schema_version: String,
    /// Artifact kind.
    pub kind: String,
    /// Hash of the canonical loaded config.
    pub config_hash: String,
    /// Output directory used for this run.
    pub output_dir: String,
    /// Optional caller-provided dataset directory.
    pub data_dir: Option<String>,
    /// Resolved dataset path or fixture descriptor.
    pub data_source: String,
    /// Number of physical dataset windows, or fixture samples.
    pub dataset_windows: usize,
    /// Requested max training steps.
    pub max_steps: u64,
    /// Completed training steps.
    pub steps_completed: u64,
    /// Effective per-step batch size.
    pub batch_size: usize,
    /// Run seed.
    pub seed: u64,
    /// Requested device string.
    pub device: String,
    /// First recorded loss.
    pub initial_loss: f64,
    /// Last recorded loss.
    pub final_loss: f64,
    /// Whether the final loss is lower than the first recorded loss.
    pub loss_decreased: bool,
    /// Optimizer step saved as a checkpoint.
    pub checkpoint_step: u64,
    /// Whether the train checkpoint contains all required files.
    pub checkpoint_complete: bool,
    /// Train checkpoint files written beside the report.
    pub checkpoint_files: Vec<String>,
    /// Number of gradient-explosion events observed by the guard.
    pub grad_explosion_events: u64,
    /// Explicitly scopes the bounded path while the full JEPA loop is pending.
    pub mode: String,
    /// Deterministic loss observations.
    pub losses: Vec<TrainLossPoint>,
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

/// Write scalar training-mechanics smoke artifacts for local/HF validation.
///
/// This validates the CLI, config, gradient accumulation, clipping,
/// non-finite guards, parameter updates, checkpoint writing, output directory,
/// and upload contract without claiming that the full JEPA loop has landed.
///
/// # Errors
///
/// Returns an error if the output directory cannot be created, a smoke
/// regression cannot be computed, or the artifact files cannot be written.
pub fn write_smoke_artifacts(
    output_dir: impl AsRef<Path>,
    config_hash: &str,
    data_dir: Option<&Path>,
    steps: u64,
    batch_size: u64,
    seed: u64,
    device: &str,
) -> Result<SmokeRunReport, TrainerError> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir).map_err(|source| io_error(output_dir, source))?;

    let outcome = run_scalar_smoke_training(steps, batch_size)?;
    let paths = write_scalar_smoke_checkpoint(output_dir, config_hash, seed, &outcome)?;
    let losses = outcome.losses.clone();
    let loss_slope = smoke_loss_slope(&losses)?;
    let report = SmokeRunReport {
        schema_version: "1.0.0".to_owned(),
        kind: "lewm-rs-smoke-report".to_owned(),
        config_hash: config_hash.to_owned(),
        data_dir: data_dir.map(|path| path.display().to_string()),
        output_dir: output_dir.display().to_string(),
        steps,
        batch_size,
        seed,
        device: device.to_owned(),
        loss_slope,
        loss_decreased: smoke_loss_decreases(&losses)?,
        initial_loss: first_loss(&losses),
        final_loss: last_loss(&losses),
        checkpoint_step: outcome.step,
        checkpoint_complete: paths.is_complete(),
        checkpoint_files: checkpoint_file_names(&paths),
        grad_explosion_events: outcome.grad_explosion_events,
        mode: "mechanics-smoke".to_owned(),
        losses,
    };

    let report_path = output_dir.join("smoke_report.json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .map_err(|source| io_error(&report_path, source))?;

    let losses_path = output_dir.join("smoke_losses.jsonl");
    let mut losses_jsonl = String::new();
    for point in &report.losses {
        losses_jsonl.push_str(&serde_json::to_string(point)?);
        losses_jsonl.push('\n');
    }
    fs::write(&losses_path, losses_jsonl).map_err(|source| io_error(&losses_path, source))?;

    Ok(report)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ScalarSmokeModel {
    weight: f64,
    bias: f64,
}

impl ScalarSmokeModel {
    const fn initial() -> Self {
        Self {
            weight: 0.0,
            bias: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ScalarAdamWParamState {
    first_moment: f64,
    second_moment: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ScalarAdamWState {
    step: i32,
    weight: ScalarAdamWParamState,
    bias: ScalarAdamWParamState,
}

#[derive(Clone, Debug, PartialEq)]
struct ScalarSmokeOutcome {
    losses: Vec<SmokeLossPoint>,
    model: ScalarSmokeModel,
    optimizer: ScalarAdamWState,
    step: u64,
    grad_explosion_events: u64,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct ScalarSmokeRecord {
    schema_version: &'static str,
    kind: &'static str,
    step: u64,
    weight: f64,
    bias: f64,
    adamw_step: i32,
}

fn run_scalar_smoke_training(
    steps: u64,
    batch_size: u64,
) -> Result<ScalarSmokeOutcome, TrainerError> {
    let total_steps = smoke_steps_as_u32(steps)?;
    let accumulation_steps = smoke_batch_size_as_u32(batch_size)?;
    let microbatch_capacity =
        usize::try_from(accumulation_steps).map_err(|_| TrainerError::InvalidSmokeBatchSize {
            batch_size: u64::from(accumulation_steps),
        })?;
    let schedule = CosineWarmup::from_parts(SMOKE_LR_PEAK, SMOKE_LR_MIN, 0, total_steps);
    let config = OptimConfig::new();
    let mut model = ScalarSmokeModel::initial();
    let mut optimizer = ScalarAdamWState::default();
    let mut guard = NanGuard::new();
    let mut losses = Vec::with_capacity(usize::try_from(total_steps).unwrap_or_default());
    let mut grad_explosion_events = 0_u64;

    for step in 1..=total_steps {
        let mut total_loss = 0.0;
        let mut microbatch_gradients = Vec::with_capacity(microbatch_capacity);
        for sample_index in 0..accumulation_steps {
            let (input, target) = smoke_sample(step, sample_index);
            let (loss, gradients) = scalar_loss_and_gradients(model, input, target);
            total_loss += scale_loss_for_accumulation(loss, accumulation_steps)?;
            microbatch_gradients.push(vec![gradients[0], gradients[1]]);
        }

        let mut gradients = accumulate_scaled_gradients(&microbatch_gradients, accumulation_steps)?;
        let step_u64 = u64::from(step);
        match guard.observe(step_u64, total_loss, &gradients) {
            NanGuardDecision::Apply => {},
            NanGuardDecision::Skip { artifact } | NanGuardDecision::Abort { artifact } => {
                return Err(TrainerError::SmokeGuardRejected {
                    step: step_u64,
                    reason: artifact.reason,
                });
            },
        }

        let clip = clip_global_norm(&mut gradients, DEFAULT_MAX_GRAD_NORM)?;
        if grad_explosion_artifact(step_u64, clip.grad_norm_pre).is_some() {
            grad_explosion_events += 1;
        }
        let learning_rate = schedule.lr(step);
        apply_scalar_adamw(
            &mut model,
            &gradients,
            &mut optimizer,
            learning_rate,
            &config,
        );
        losses.push(SmokeLossPoint {
            step: step_u64,
            loss: total_loss,
            grad_norm_pre: clip.grad_norm_pre,
            grad_norm_post: clip.grad_norm_post,
            learning_rate,
        });
    }

    Ok(ScalarSmokeOutcome {
        losses,
        model,
        optimizer,
        step: steps,
        grad_explosion_events,
    })
}

fn smoke_steps_as_u32(steps: u64) -> Result<u32, TrainerError> {
    if steps == 0 {
        return Err(TrainerError::InvalidSmokeSteps { steps });
    }
    u32::try_from(steps).map_err(|_| TrainerError::SmokeStepOutOfRange { step: steps })
}

fn smoke_batch_size_as_u32(batch_size: u64) -> Result<u32, TrainerError> {
    if !(1..=SMOKE_MAX_BATCH_SIZE).contains(&batch_size) {
        return Err(TrainerError::InvalidSmokeBatchSize { batch_size });
    }
    u32::try_from(batch_size).map_err(|_| TrainerError::InvalidSmokeBatchSize { batch_size })
}

fn smoke_sample(step: u32, sample_index: u32) -> (f64, f64) {
    let raw_bucket = ((u64::from(step) * 13) + (u64::from(sample_index) * 7)) % 23;
    let bucket = u32::try_from(raw_bucket).unwrap_or_default();
    let input = (f64::from(bucket) - 11.0) / 11.0;
    let target = (2.0 * input) - 0.5;
    (input, target)
}

fn scalar_loss_and_gradients(model: ScalarSmokeModel, input: f64, target: f64) -> (f64, [f64; 2]) {
    let prediction = model.weight.mul_add(input, model.bias);
    let residual = prediction - target;
    let loss = residual * residual;
    let grad_weight = 2.0 * residual * input;
    let grad_bias = 2.0 * residual;
    (loss, [grad_weight, grad_bias])
}

fn apply_scalar_adamw(
    model: &mut ScalarSmokeModel,
    gradients: &[f64],
    optimizer: &mut ScalarAdamWState,
    learning_rate: f64,
    config: &OptimConfig,
) {
    optimizer.step += 1;
    model.weight = adamw_update_scalar(
        model.weight,
        gradients[0],
        &mut optimizer.weight,
        optimizer.step,
        learning_rate,
        config,
        true,
    );
    model.bias = adamw_update_scalar(
        model.bias,
        gradients[1],
        &mut optimizer.bias,
        optimizer.step,
        learning_rate,
        config,
        false,
    );
}

fn adamw_update_scalar(
    value: f64,
    gradient: f64,
    state: &mut ScalarAdamWParamState,
    step: i32,
    learning_rate: f64,
    config: &OptimConfig,
    apply_weight_decay: bool,
) -> f64 {
    state.first_moment = config
        .beta1
        .mul_add(state.first_moment, (1.0 - config.beta1) * gradient);
    state.second_moment = config.beta2.mul_add(
        state.second_moment,
        (1.0 - config.beta2) * gradient * gradient,
    );

    let decayed_value = if apply_weight_decay {
        value * (1.0 - (learning_rate * config.weight_decay))
    } else {
        value
    };
    let corrected_first = state.first_moment / (1.0 - config.beta1.powi(step));
    let corrected_second = state.second_moment / (1.0 - config.beta2.powi(step));

    decayed_value - (learning_rate * corrected_first / (corrected_second.sqrt() + config.epsilon))
}

fn write_scalar_smoke_checkpoint(
    output_dir: &Path,
    config_hash: &str,
    seed: u64,
    outcome: &ScalarSmokeOutcome,
) -> Result<CheckpointPaths, TrainerError> {
    let parameters = vec![
        ParameterTensor::f32(
            "smoke_scalar.weight",
            vec![1],
            vec![f32_from_f64(outcome.model.weight)],
        )?,
        ParameterTensor::f32(
            "smoke_scalar.bias",
            vec![1],
            vec![f32_from_f64(outcome.model.bias)],
        )?,
    ];
    let parity = ParityProbe {
        encoder_cls_l_inf: 0.0,
        predictor_l_inf: 0.0,
        sigreg_value: last_loss(&outcome.losses),
    };
    let record = ScalarSmokeRecord {
        schema_version: "1.0.0",
        kind: "lewm-rs-scalar-smoke-record",
        step: outcome.step,
        weight: outcome.model.weight,
        bias: outcome.model.bias,
        adamw_step: outcome.optimizer.step,
    };
    let burn_record = serde_json::to_vec(&record)?;
    let request = CheckpointWriteRequest {
        output_dir,
        run_id: "smoke-mechanics-v1",
        step: outcome.step,
        epoch: 0,
        wall_time_s: 0.0,
        git_short_sha: option_env!("LEWM_GIT_SHA").unwrap_or("unknown"),
        config_hash,
        rng_state: CheckpointRngState {
            global_seed: seed,
            step_at_save: outcome.step,
            data_shuffle: "deterministic-scalar-grid-v1".to_owned(),
            sigreg_sketch: "disabled-for-smoke".to_owned(),
            dropout: "disabled-for-smoke".to_owned(),
            cem: "disabled-for-smoke".to_owned(),
            model_init: "scalar-smoke-initial-zeros".to_owned(),
        },
        metrics_last_step: smoke_checkpoint_metrics(outcome),
        burn_record: &burn_record,
        parameters: &parameters,
        parity: &parity,
    };

    save_checkpoint(&request).map_err(TrainerError::from)
}

fn smoke_checkpoint_metrics(outcome: &ScalarSmokeOutcome) -> BTreeMap<String, f64> {
    let mut metrics = BTreeMap::new();
    metrics.insert("loss/total".to_owned(), last_loss(&outcome.losses));
    metrics.insert(
        "optim/grad_norm_pre".to_owned(),
        outcome
            .losses
            .last()
            .map_or(0.0, |point| point.grad_norm_pre),
    );
    metrics.insert(
        "optim/grad_norm_post".to_owned(),
        outcome
            .losses
            .last()
            .map_or(0.0, |point| point.grad_norm_post),
    );
    metrics.insert(
        "smoke/grad_explosion_events".to_owned(),
        smoke_step_as_f64(outcome.grad_explosion_events).unwrap_or(0.0),
    );
    metrics
}

fn checkpoint_file_names(paths: &CheckpointPaths) -> Vec<String> {
    [
        &paths.model_burn,
        &paths.model_safetensors,
        &paths.sidecar,
        &paths.parity,
    ]
    .iter()
    .map(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .map_or_else(|| path.display().to_string(), str::to_owned)
    })
    .collect()
}

fn first_loss(points: &[SmokeLossPoint]) -> f64 {
    points.first().map_or(0.0, |point| point.loss)
}

fn last_loss(points: &[SmokeLossPoint]) -> f64 {
    points.last().map_or(0.0, |point| point.loss)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn f32_from_f64(value: f64) -> f32 {
    value as f32
}

/// Write bounded `PushT` training artifacts for local/container/HF validation.
///
/// This is a real data-plane train path for a minimal componentized `LeWM`
/// core: it consumes `PushT` windows when the dataset exists and falls back to
/// an explicit `PushT`-compatible fixture when the default local path is absent.
/// It does not claim to train the full Burn `ViT` stack.
///
/// # Errors
///
/// Returns an error if the config is not `PushT`, max steps are invalid, a
/// caller-provided data directory is missing, dataset reads fail, a train step
/// guard rejects the update, or artifact/checkpoint writes fail.
pub fn write_train_artifacts(
    output_dir: impl AsRef<Path>,
    root: &RootConfig,
    config_hash: &str,
    data_dir: Option<&Path>,
    max_steps: u64,
    seed: u64,
    device: &str,
) -> Result<TrainRunReport, TrainerError> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir).map_err(|source| io_error(output_dir, source))?;

    let mut source = open_pusht_training_source(root, data_dir)?;
    let outcome = run_pusht_minimal_lewm_training(
        &mut source,
        &root.training,
        root.loss.lambda_sigreg,
        max_steps,
    )?;
    let paths = write_pusht_minimal_lewm_checkpoint(output_dir, config_hash, seed, &outcome)?;
    let losses = outcome.losses.clone();
    let report = TrainRunReport {
        schema_version: "1.0.0".to_owned(),
        kind: "lewm-rs-train-report".to_owned(),
        config_hash: config_hash.to_owned(),
        output_dir: output_dir.display().to_string(),
        data_dir: data_dir.map(|path| path.display().to_string()),
        data_source: source.description(),
        dataset_windows: source.len(),
        max_steps,
        steps_completed: outcome.step,
        batch_size: outcome.batch_size,
        seed,
        device: device.to_owned(),
        initial_loss: first_train_loss(&losses),
        final_loss: last_train_loss(&losses),
        loss_decreased: last_train_loss(&losses) < first_train_loss(&losses),
        checkpoint_step: outcome.step,
        checkpoint_complete: paths.is_complete(),
        checkpoint_files: checkpoint_file_names(&paths),
        grad_explosion_events: outcome.grad_explosion_events,
        mode: "pusht-minimal-lewm".to_owned(),
        losses,
    };

    let report_path = output_dir.join("train_report.json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .map_err(|source| io_error(&report_path, source))?;

    let losses_path = output_dir.join("train_losses.jsonl");
    let mut losses_jsonl = String::new();
    for point in &report.losses {
        losses_jsonl.push_str(&serde_json::to_string(point)?);
        losses_jsonl.push('\n');
    }
    fs::write(&losses_path, losses_jsonl).map_err(|source| io_error(&losses_path, source))?;

    Ok(report)
}

#[derive(Debug)]
enum PushtTrainingSource {
    Hdf5 {
        dataset: Box<PushtDataset>,
        path: PathBuf,
    },
    Fixture {
        samples: Vec<PushtSample>,
        descriptor: String,
    },
}

impl PushtTrainingSource {
    fn len(&self) -> usize {
        match self {
            Self::Hdf5 { dataset, .. } => dataset.len(),
            Self::Fixture { samples, .. } => samples.len(),
        }
    }

    fn description(&self) -> String {
        match self {
            Self::Hdf5 { path, .. } => format!("pusht-hdf5:{}", path.display()),
            Self::Fixture { descriptor, .. } => descriptor.clone(),
        }
    }

    fn get(&self, index: usize) -> Result<PushtSample, TrainerError> {
        match self {
            Self::Hdf5 { dataset, .. } => dataset.get(index).map_err(TrainerError::from),
            Self::Fixture { samples, .. } => {
                let sample =
                    samples
                        .get(index % samples.len())
                        .ok_or_else(|| TrainerError::Data {
                            source: DataError::EmptyDataset(
                                "PushT-compatible fixture has no samples".to_owned(),
                            ),
                        })?;
                Ok(sample.clone())
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PushtMinimalLewmAdamWState {
    step: i32,
    params: [ScalarAdamWParamState; PUSHT_MINIMAL_LEWM_PARAM_COUNT],
}

impl Default for PushtMinimalLewmAdamWState {
    fn default() -> Self {
        Self {
            step: 0,
            params: [ScalarAdamWParamState::default(); PUSHT_MINIMAL_LEWM_PARAM_COUNT],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct PushtMinimalLewmOutcome {
    losses: Vec<TrainLossPoint>,
    model: PushtMinimalLewmCore,
    optimizer: PushtMinimalLewmAdamWState,
    step: u64,
    batch_size: usize,
    samples_seen: u64,
    grad_explosion_events: u64,
}

#[derive(Clone, Debug, Serialize)]
struct PushtMinimalLewmRecord {
    schema_version: &'static str,
    kind: &'static str,
    step: u64,
    params: Vec<f64>,
    adamw_step: i32,
    samples_seen: u64,
}

fn open_pusht_training_source(
    root: &RootConfig,
    data_dir: Option<&Path>,
) -> Result<PushtTrainingSource, TrainerError> {
    let DatasetConfig::Pusht(config) = &root.dataset else {
        return Err(TrainerError::UnsupportedTrainDataset {
            kind: dataset_kind_name(&root.dataset).to_owned(),
        });
    };
    let path = data_dir.map_or_else(|| config.root_path.clone(), Path::to_path_buf);

    if path.exists() {
        let dataset_config = DataPushtConfig {
            root_path: path.clone(),
            split: data_split(config.split),
            horizon: config.horizon,
            history_size: config.history_size,
            seed: Some(config.seed),
            validate_schema: true,
            stats_path: None,
        };
        let dataset = PushtDataset::new(dataset_config)?;
        return Ok(PushtTrainingSource::Hdf5 {
            dataset: Box::new(dataset),
            path,
        });
    }

    if data_dir.is_some() {
        return Err(TrainerError::MissingTrainDataPath { path });
    }

    Ok(PushtTrainingSource::Fixture {
        samples: pusht_fixture_samples(config.horizon)?,
        descriptor: format!(
            "pusht-compatible-fixture:{PUSHT_FIXTURE_SAMPLE_COUNT}-samples:{PUSHT_FIXTURE_FRAME_SIZE}x{PUSHT_FIXTURE_FRAME_SIZE}"
        ),
    })
}

fn run_pusht_minimal_lewm_training(
    source: &mut PushtTrainingSource,
    config: &TrainingConfig,
    lambda_sigreg: f64,
    max_steps: u64,
) -> Result<PushtMinimalLewmOutcome, TrainerError> {
    if !lambda_sigreg.is_finite() {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(
                "PushT minimal LeWM lambda_sigreg must be finite".to_owned(),
            ),
        });
    }
    let total_steps = train_steps_as_u32(max_steps)?;
    let batch_size_u32 = train_batch_size_as_u32(config.batch_size)?;
    let batch_size =
        usize::try_from(batch_size_u32).map_err(|_| TrainerError::InvalidTrainBatchSize {
            batch_size: config.batch_size,
        })?;
    let schedule = CosineWarmup::from_parts(
        config.lr_peak,
        config.lr_min,
        config.warmup_steps,
        total_steps.max(config.warmup_steps.saturating_add(1)),
    );
    let optim_config = OptimConfig::new()
        .with_beta1(config.betas.0)
        .with_beta2(config.betas.1)
        .with_weight_decay(config.weight_decay);
    let mut model = PushtMinimalLewmCore::initial();
    let mut optimizer = PushtMinimalLewmAdamWState::default();
    let mut guard = NanGuard::new();
    let mut losses = Vec::with_capacity(usize::try_from(total_steps).unwrap_or_default());
    let mut samples_seen = 0_u64;
    let mut grad_explosion_events = 0_u64;
    let sigreg_weight = lambda_sigreg.max(0.0);

    for step in 1..=total_steps {
        let mut total_loss = 0.0;
        let mut pred_loss = 0.0;
        let mut sigreg_proxy_loss = 0.0;
        let mut sample_gradients = Vec::with_capacity(batch_size);
        for sample_offset in 0..batch_size {
            let dataset_index = training_sample_index(step, sample_offset, batch_size);
            let sample = source.get(dataset_index)?;
            let example = minimal_lewm_example_from_sample(&sample, config.history_size)?;
            let (sample_loss, gradients) = loss_and_gradients(&model, example, sigreg_weight);
            total_loss += scale_loss_for_accumulation(sample_loss.total, batch_size_u32)?;
            pred_loss += scale_loss_for_accumulation(sample_loss.pred, batch_size_u32)?;
            sigreg_proxy_loss +=
                scale_loss_for_accumulation(sample_loss.sigreg_proxy, batch_size_u32)?;
            sample_gradients.push(gradients.to_vec());
            samples_seen = samples_seen.saturating_add(1);
        }

        let mut gradients = accumulate_scaled_gradients(&sample_gradients, batch_size_u32)?;
        let step_u64 = u64::from(step);
        match guard.observe(step_u64, total_loss, &gradients) {
            NanGuardDecision::Apply => {},
            NanGuardDecision::Skip { artifact } | NanGuardDecision::Abort { artifact } => {
                return Err(TrainerError::TrainGuardRejected {
                    step: step_u64,
                    reason: artifact.reason,
                });
            },
        }

        let clip = clip_global_norm(&mut gradients, config.grad_clip_norm)?;
        if grad_explosion_artifact(step_u64, clip.grad_norm_pre).is_some() {
            grad_explosion_events += 1;
        }
        let learning_rate = schedule.lr(step);
        apply_pusht_minimal_lewm_adamw(
            &mut model,
            &gradients,
            &mut optimizer,
            learning_rate,
            &optim_config,
        );
        losses.push(TrainLossPoint {
            step: step_u64,
            loss: total_loss,
            pred_loss,
            sigreg_proxy_loss,
            grad_norm_pre: clip.grad_norm_pre,
            grad_norm_post: clip.grad_norm_post,
            learning_rate,
            samples_seen,
        });
    }

    Ok(PushtMinimalLewmOutcome {
        losses,
        model,
        optimizer,
        step: max_steps,
        batch_size,
        samples_seen,
        grad_explosion_events,
    })
}

fn train_steps_as_u32(steps: u64) -> Result<u32, TrainerError> {
    if steps == 0 {
        return Err(TrainerError::InvalidTrainSteps { steps });
    }
    u32::try_from(steps).map_err(|_| TrainerError::InvalidTrainSteps { steps })
}

fn train_batch_size_as_u32(batch_size: usize) -> Result<u32, TrainerError> {
    if batch_size == 0 {
        return Err(TrainerError::InvalidTrainBatchSize { batch_size });
    }
    u32::try_from(batch_size).map_err(|_| TrainerError::InvalidTrainBatchSize { batch_size })
}

fn training_sample_index(step: u32, sample_offset: usize, batch_size: usize) -> usize {
    let step_index = usize::try_from(step.saturating_sub(1)).unwrap_or_default();
    step_index
        .saturating_mul(batch_size)
        .saturating_add(sample_offset)
}

fn minimal_lewm_example_from_sample(
    sample: &PushtSample,
    history_size: usize,
) -> Result<PushtMinimalLewmExample, TrainerError> {
    let action_values = sample.actions.len();
    let expected_action_values = sample
        .action_shape
        .0
        .checked_mul(PUSHT_ACTION_DIM)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform("PushT action shape overflow".to_owned()),
        })?;
    if sample.action_shape.1 != PUSHT_ACTION_DIM || action_values != expected_action_values {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "PushT minimal LeWM expects action shape (T, {PUSHT_ACTION_DIM}), found {:?} with {action_values} values",
                sample.action_shape
            )),
        });
    }

    let frame_count = sample.frame_shape.0;
    if frame_count == 0 {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform("PushT frame time is zero".to_owned()),
        });
    }
    let source_len = history_size.clamp(1, frame_count);
    let target_start = source_len.min(frame_count.saturating_sub(1));
    let target_len = frame_count.saturating_sub(target_start).max(1);

    Ok(PushtMinimalLewmExample {
        source: sample_temporal_features(sample, 0, source_len)?,
        target: sample_temporal_features(sample, target_start, target_len)?,
        action_mean: sample_action_mean(sample)?,
    })
}

fn sample_temporal_features(
    sample: &PushtSample,
    start_frame: usize,
    frame_count: usize,
) -> Result<PushtMinimalLewmFeatures, TrainerError> {
    if sample.frames_t.is_empty() {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform("PushT frame buffer is empty".to_owned()),
        });
    }
    let frame_stride = sample
        .frame_shape
        .1
        .checked_mul(sample.frame_shape.2)
        .and_then(|value| value.checked_mul(sample.frame_shape.3))
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform("PushT frame shape overflow".to_owned()),
        })?;
    let end_frame = start_frame
        .checked_add(frame_count)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform("PushT temporal feature range overflow".to_owned()),
        })?;
    let end_offset = end_frame
        .checked_mul(frame_stride)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform(
                "PushT temporal feature offset overflow".to_owned(),
            ),
        })?;
    if end_frame > sample.frame_shape.0 || end_offset > sample.frames_t.len() {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "PushT temporal feature range {start_frame}..{end_frame} exceeds frame shape {:?}",
                sample.frame_shape
            )),
        });
    }

    let start_offset = start_frame
        .checked_mul(frame_stride)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform("PushT temporal feature start overflow".to_owned()),
        })?;
    let values = &sample.frames_t[start_offset..end_offset];
    let mut sum = 0.0;
    let mut energy = 0.0;
    for value in values {
        let normalized = (f64::from(*value) / 255.0 * 2.0) - 1.0;
        sum += normalized;
        energy += normalized * normalized;
    }
    let denominator = usize_to_f64(values.len())?;
    Ok(PushtMinimalLewmFeatures {
        pixel_mean: sum / denominator,
        pixel_energy: energy / denominator,
        time_fraction: f64::from(sample.meta.start_frame) / 1_000.0,
    })
}

fn sample_action_mean(sample: &PushtSample) -> Result<[f64; PUSHT_ACTION_DIM], TrainerError> {
    let time = sample.action_shape.0;
    if time == 0 {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform("PushT action time is zero".to_owned()),
        });
    }
    let mut sums = [0.0; PUSHT_ACTION_DIM];
    for chunk in sample.actions.chunks_exact(PUSHT_ACTION_DIM) {
        for (dim, value) in chunk.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(TrainerError::Data {
                    source: DataError::InvalidTransform(
                        "PushT action value must be finite".to_owned(),
                    ),
                });
            }
            sums[dim] += f64::from(value);
        }
    }
    let denominator = usize_to_f64(time)?;
    Ok([sums[0] / denominator, sums[1] / denominator])
}

fn apply_pusht_minimal_lewm_adamw(
    model: &mut PushtMinimalLewmCore,
    gradients: &[f64],
    optimizer: &mut PushtMinimalLewmAdamWState,
    learning_rate: f64,
    config: &OptimConfig,
) {
    optimizer.step += 1;
    for (param_index, gradient) in gradients
        .iter()
        .copied()
        .enumerate()
        .take(PUSHT_MINIMAL_LEWM_PARAM_COUNT)
    {
        let spec = parameter_spec_for_flat_index(param_index);
        let updated = adamw_update_scalar(
            model.parameter(param_index),
            gradient,
            &mut optimizer.params[param_index],
            optimizer.step,
            learning_rate,
            config,
            spec.apply_weight_decay,
        );
        model.set_parameter(param_index, updated);
    }
}

fn write_pusht_minimal_lewm_checkpoint(
    output_dir: &Path,
    config_hash: &str,
    seed: u64,
    outcome: &PushtMinimalLewmOutcome,
) -> Result<CheckpointPaths, TrainerError> {
    let parameters = PUSHT_MINIMAL_LEWM_PARAMETER_SPECS
        .iter()
        .copied()
        .map(|spec| minimal_lewm_parameter_tensor(&outcome.model, spec))
        .collect::<Result<Vec<_>, _>>()?;
    let parity = ParityProbe {
        encoder_cls_l_inf: outcome.losses.last().map_or(0.0, |point| point.pred_loss),
        predictor_l_inf: last_train_loss(&outcome.losses),
        sigreg_value: outcome
            .losses
            .last()
            .map_or(0.0, |point| point.sigreg_proxy_loss),
    };
    let record = PushtMinimalLewmRecord {
        schema_version: "1.0.0",
        kind: "lewm-rs-pusht-minimal-lewm-record",
        step: outcome.step,
        params: outcome.model.flat_parameters().to_vec(),
        adamw_step: outcome.optimizer.step,
        samples_seen: outcome.samples_seen,
    };
    let burn_record = serde_json::to_vec(&record)?;
    let request = CheckpointWriteRequest {
        output_dir,
        run_id: "pusht-minimal-lewm-v1",
        step: outcome.step,
        epoch: 0,
        wall_time_s: 0.0,
        git_short_sha: option_env!("LEWM_GIT_SHA").unwrap_or("unknown"),
        config_hash,
        rng_state: CheckpointRngState {
            global_seed: seed,
            step_at_save: outcome.step,
            data_shuffle: "sequential-window-modulo-v1".to_owned(),
            sigreg_sketch: "minimal-lewm-latent-scale-proxy-v1".to_owned(),
            dropout: "disabled-for-minimal-lewm".to_owned(),
            cem: "disabled-for-minimal-lewm".to_owned(),
            model_init: "pusht-minimal-lewm-deterministic-init-v1".to_owned(),
        },
        metrics_last_step: train_checkpoint_metrics(outcome),
        burn_record: &burn_record,
        parameters: &parameters,
        parity: &parity,
    };

    save_checkpoint(&request).map_err(TrainerError::from)
}

fn minimal_lewm_parameter_tensor(
    model: &PushtMinimalLewmCore,
    spec: crate::pusht_lewm::PushtMinimalLewmParameterSpec,
) -> Result<ParameterTensor, TrainerError> {
    let group_values = model.parameter_values(spec);
    let values = group_values.iter().copied().map(f32_from_f64).collect();
    Ok(ParameterTensor::f32(
        spec.name,
        vec![PUSHT_MINIMAL_LEWM_LATENT_DIM],
        values,
    )?)
}

fn train_checkpoint_metrics(outcome: &PushtMinimalLewmOutcome) -> BTreeMap<String, f64> {
    let mut metrics = BTreeMap::new();
    metrics.insert("loss/train".to_owned(), last_train_loss(&outcome.losses));
    metrics.insert(
        "loss/prediction".to_owned(),
        outcome.losses.last().map_or(0.0, |point| point.pred_loss),
    );
    metrics.insert(
        "loss/sigreg_proxy".to_owned(),
        outcome
            .losses
            .last()
            .map_or(0.0, |point| point.sigreg_proxy_loss),
    );
    metrics.insert(
        "optim/grad_norm_pre".to_owned(),
        outcome
            .losses
            .last()
            .map_or(0.0, |point| point.grad_norm_pre),
    );
    metrics.insert(
        "optim/grad_norm_post".to_owned(),
        outcome
            .losses
            .last()
            .map_or(0.0, |point| point.grad_norm_post),
    );
    metrics.insert(
        "train/samples_seen".to_owned(),
        smoke_step_as_f64(outcome.samples_seen).unwrap_or(0.0),
    );
    metrics.insert(
        "train/grad_explosion_events".to_owned(),
        smoke_step_as_f64(outcome.grad_explosion_events).unwrap_or(0.0),
    );
    metrics
}

fn pusht_fixture_samples(horizon: usize) -> Result<Vec<PushtSample>, TrainerError> {
    (0..PUSHT_FIXTURE_SAMPLE_COUNT)
        .map(|index| pusht_fixture_sample(index, horizon))
        .collect()
}

fn pusht_fixture_sample(index: usize, horizon: usize) -> Result<PushtSample, TrainerError> {
    let pixel_count = horizon
        .checked_mul(PUSHT_FIXTURE_FRAME_SIZE)
        .and_then(|value| value.checked_mul(PUSHT_FIXTURE_FRAME_SIZE))
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform("PushT fixture pixel shape overflow".to_owned()),
        })?;
    let index_u32 = u32::try_from(index).map_err(|_| TrainerError::Data {
        source: DataError::InvalidTransform("PushT fixture index overflow".to_owned()),
    })?;
    let mut frames_t = Vec::with_capacity(pixel_count);
    for pixel_index in 0..pixel_count {
        let pixel_u32 = u32::try_from(pixel_index % 251).map_err(|_| TrainerError::Data {
            source: DataError::InvalidTransform("PushT fixture pixel index overflow".to_owned()),
        })?;
        let value = ((index_u32.saturating_mul(17)) + pixel_u32) % 255;
        frames_t.push(u8::try_from(value).map_err(|_| TrainerError::Data {
            source: DataError::InvalidTransform("PushT fixture pixel value overflow".to_owned()),
        })?);
    }

    let feature = (f64::from(index_u32 % 31) - 15.0) / 15.0;
    let mut actions = Vec::with_capacity(horizon.saturating_mul(PUSHT_ACTION_DIM));
    for timestep in 0..horizon {
        let time_fraction = usize_to_f64(timestep)? / usize_to_f64(horizon)?;
        actions.push(f32_from_f64((0.6 * feature) + (0.2 * time_fraction)));
        actions.push(f32_from_f64((-0.4 * feature) + (0.1 * time_fraction)));
    }

    Ok(PushtSample {
        frames_t,
        frame_shape: (
            horizon,
            PUSHT_FIXTURE_FRAME_SIZE,
            PUSHT_FIXTURE_FRAME_SIZE,
            3,
        ),
        actions,
        action_shape: (horizon, PUSHT_ACTION_DIM),
        meta: PushtSampleMeta {
            episode_id: index_u32,
            start_frame: 0,
            shard: 0,
        },
    })
}

fn data_split(split: DatasetSplit) -> DataSplit {
    match split {
        DatasetSplit::Train => DataSplit::Train,
        DatasetSplit::Eval => DataSplit::Eval,
    }
}

fn dataset_kind_name(dataset: &DatasetConfig) -> &'static str {
    match dataset {
        DatasetConfig::Pusht(_) => "pusht",
        DatasetConfig::So100(_) => "so100",
    }
}

fn first_train_loss(points: &[TrainLossPoint]) -> f64 {
    points.first().map_or(0.0, |point| point.loss)
}

fn last_train_loss(points: &[TrainLossPoint]) -> f64 {
    points.last().map_or(0.0, |point| point.loss)
}

fn usize_to_f64(value: usize) -> Result<f64, TrainerError> {
    u32::try_from(value)
        .map(f64::from)
        .map_err(|_| TrainerError::Data {
            source: DataError::InvalidTransform(format!("{value} does not fit u32")),
        })
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
                grad_norm_pre: 1.0,
                grad_norm_post: 1.0,
                learning_rate: 0.01,
            })
            .collect::<Vec<_>>();
        let increasing = (1_u32..=smoke_steps)
            .map(|step| SmokeLossPoint {
                step: u64::from(step),
                loss: f64::from(step),
                grad_norm_pre: 1.0,
                grad_norm_post: 1.0,
                learning_rate: 0.01,
            })
            .collect::<Vec<_>>();

        assert!(smoke_loss_decreases(&decreasing)?);
        assert!(!smoke_loss_decreases(&increasing)?);
        Ok(())
    }

    #[test]
    fn smoke_artifacts_are_written() -> Result<(), Box<dyn std::error::Error>> {
        let dir = TestDir::new("smoke-artifacts")?;
        let report = write_smoke_artifacts(
            dir.path(),
            "abc123",
            Some(Path::new("/tmp/data")),
            SMOKE_STEPS,
            4,
            7,
            "cpu",
        )?;

        assert_eq!(report.kind, "lewm-rs-smoke-report");
        assert_eq!(report.mode, "mechanics-smoke");
        assert!(report.loss_decreased);
        assert!(report.loss_slope < 0.0);
        assert!(report.initial_loss > report.final_loss);
        assert_eq!(report.checkpoint_step, SMOKE_STEPS);
        assert!(report.checkpoint_complete);
        assert_eq!(report.checkpoint_files.len(), 4);
        assert!(dir.path().join("smoke_report.json").is_file());
        assert!(dir.path().join("smoke_losses.jsonl").is_file());
        let loaded = crate::checkpoint::load_checkpoint(dir.path().join("step_0000050.json"))?;
        assert_eq!(loaded.sidecar.step, SMOKE_STEPS);
        assert_eq!(loaded.sidecar.rng_state.global_seed, 7);
        Ok(())
    }

    #[test]
    fn pusht_train_fixture_artifacts_are_written() -> Result<(), Box<dyn std::error::Error>> {
        let dir = TestDir::new("pusht-train-artifacts")?;
        let missing_data = dir.path().join("missing-pusht");
        let mut root = crate::config::RootConfig::default();
        let crate::config::DatasetConfig::Pusht(config) = &mut root.dataset else {
            return Err("expected PushT default dataset".into());
        };
        config.root_path = missing_data;

        let report = write_train_artifacts(dir.path(), &root, "abc123", None, 10, 7, "cpu")?;

        assert_eq!(report.kind, "lewm-rs-train-report");
        assert_eq!(report.mode, "pusht-minimal-lewm");
        assert!(report.data_source.starts_with("pusht-compatible-fixture"));
        assert_eq!(report.steps_completed, 10);
        assert_eq!(report.checkpoint_step, 10);
        assert!(report.checkpoint_complete);
        assert_eq!(report.checkpoint_files.len(), 4);
        assert!(
            report
                .losses
                .iter()
                .all(|point| point.pred_loss.is_finite())
        );
        assert!(
            report
                .losses
                .iter()
                .all(|point| point.sigreg_proxy_loss.is_finite())
        );
        assert!(dir.path().join("train_report.json").is_file());
        assert!(dir.path().join("train_losses.jsonl").is_file());
        let loaded = crate::checkpoint::load_checkpoint(dir.path().join("step_0000010.json"))?;
        assert_eq!(loaded.sidecar.step, 10);
        assert_eq!(loaded.sidecar.rng_state.global_seed, 7);
        assert_eq!(loaded.sidecar.run_id, "pusht-minimal-lewm-v1");
        assert!(
            loaded
                .sidecar
                .metrics_last_step
                .contains_key("loss/prediction")
        );
        assert!(
            loaded
                .sidecar
                .metrics_last_step
                .contains_key("loss/sigreg_proxy")
        );
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
