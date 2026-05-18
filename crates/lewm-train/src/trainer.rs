//! Trainer outer-loop state machine and artifact contracts.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::checkpoint::{
    CHECKPOINT_SCHEMA_VERSION, CheckpointError, CheckpointPaths, CheckpointRngState,
    CheckpointWriteRequest, ParameterTensor, ParityProbe, load_checkpoint, save_checkpoint,
};
use crate::config::{DatasetConfig, DatasetSplit, PushtTrainMode, RootConfig, So100DatasetConfig};
use crate::optim::{ADAMW_EPSILON, OptimConfig};
use crate::pusht_full::{
    PUSHT_BOUNDED_LEWM_MODE, PUSHT_BOUNDED_LEWM_RECORD_KIND, PUSHT_BOUNDED_LEWM_RUN_ID,
    PUSHT_LEGACY_FULL_LEWM_RECORD_KIND, PUSHT_LEGACY_FULL_LEWM_RUN_ID, PushtFullLewmCore,
    PushtFullLewmError, PushtFullLewmExample, PushtFullLewmImageFeatures, SO100_FULL_LEWM_RUN_ID,
};
use crate::resume::{
    RUN_ID_FILE, RestoredRngStreams, StartupMode, detect_resume, encode_rng, restore_rng_streams,
};
use crate::schedule::CosineWarmup;
use crate::step::{
    DEFAULT_MAX_GRAD_NORM, NanGuard, NanGuardDecision, StepError, accumulate_scaled_gradients,
    clip_global_norm, grad_explosion_artifact, scale_loss_for_accumulation,
};
use crate::warmstart::{
    TrainError as WarmstartError, TrainStateRecord, WarmstartProvenance, load_warmstart,
};
use burn_core::module::{AutodiffModule, Module};
use burn_core::record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder};
use burn_core::tensor::{Tensor, TensorData, backend::Backend as BurnBackend};
use burn_optim::{AdamWConfig, GradientsParams, Optimizer};
use lewm_core::{
    CEM_STREAM, DATA_SHUFFLE_STREAM, DROPOUT_STREAM, Jepa, MODEL_INIT_STREAM, SIGREG_SKETCH_STREAM,
    export::{ExportDType, ExportedTensor, collect_parameters},
    substream_rng,
};
use lewm_data::{
    DataError, ImagePreprocessor, PushtConfig as DataPushtConfig, PushtDataset,
    Sample as PushtSample, SampleMeta as PushtSampleMeta, So100Config as DataSo100Config,
    So100Dataset, Split as DataSplit,
};
use rand::Rng;
use safetensors::tensor::{Dtype, SafeTensors};
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
const PUSHT_ACTION_DIM: usize = 2;
const PUSHT_FULL_BURN_JEPA_MODE: &str = "pusht-full-burn-jepa";
const PUSHT_FULL_BURN_JEPA_RUN_ID: &str = "pusht-full-burn-jepa-v1";

type PushtBurnCpuBackend = burn_ndarray::NdArray<f32>;
type PushtBurnAutodiffBackend = burn_autodiff::Autodiff<PushtBurnCpuBackend>;

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
    /// Resume detection or RNG restoration failed.
    Resume {
        /// Original resume error.
        source: crate::resume::ResumeError,
    },
    /// Core deterministic helper failed.
    Core {
        /// Original core error.
        source: lewm_core::LewmCoreError,
    },
    /// Core Safetensors export failed.
    Export {
        /// Original export error.
        source: lewm_core::export::ExportError,
    },
    /// Burn record serialization failed.
    BurnRecord {
        /// Original recorder error.
        source: burn_core::record::RecorderError,
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
    /// SO-100 warm-start transfer failed.
    Warmstart {
        /// Original warm-start error.
        source: WarmstartError,
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
    /// The train guard rejected a `PushT` train step.
    TrainGuardRejected {
        /// Rejected optimizer step.
        step: u64,
        /// Guard reason.
        reason: String,
    },
    /// A resume checkpoint does not match the current run contract.
    ResumeCheckpointInvalid {
        /// Actionable validation failure.
        reason: String,
    },
    /// A resume checkpoint was written from a different config.
    ResumeConfigHashMismatch {
        /// Config hash stored in the checkpoint sidecar.
        checkpoint: String,
        /// Config hash for the current invocation.
        current: String,
    },
    /// A resume checkpoint was written from a different seed.
    ResumeSeedMismatch {
        /// Seed stored in the checkpoint sidecar.
        checkpoint: u64,
        /// Seed for the current invocation.
        current: u64,
    },
    /// A resume checkpoint is already beyond the requested target step.
    ResumeStepBeyondTarget {
        /// Checkpoint step.
        checkpoint_step: u64,
        /// Requested max steps.
        max_steps: u64,
    },
}

impl fmt::Display for TrainerError {
    #[allow(clippy::too_many_lines)]
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
            Self::Resume { source } => write!(formatter, "trainer resume error: {source}"),
            Self::Core { source } => write!(formatter, "trainer core error: {source}"),
            Self::Export { source } => write!(formatter, "trainer export error: {source}"),
            Self::BurnRecord { source } => {
                write!(formatter, "trainer Burn record error: {source}")
            },
            Self::Step { source } => write!(formatter, "trainer step error: {source}"),
            Self::Data { source } => write!(formatter, "trainer data error: {source}"),
            Self::Warmstart { source } => {
                write!(formatter, "trainer SO-100 warm-start error: {source}")
            },
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
            Self::ResumeCheckpointInvalid { reason } => {
                write!(formatter, "invalid PushT resume checkpoint: {reason}")
            },
            Self::ResumeConfigHashMismatch {
                checkpoint,
                current,
            } => write!(
                formatter,
                "resume config hash mismatch: checkpoint has {checkpoint}, current config has {current}; use the original config or start a fresh output directory"
            ),
            Self::ResumeSeedMismatch {
                checkpoint,
                current,
            } => write!(
                formatter,
                "resume seed mismatch: checkpoint has {checkpoint}, current run has {current}; use the original seed or start a fresh output directory"
            ),
            Self::ResumeStepBeyondTarget {
                checkpoint_step,
                max_steps,
            } => write!(
                formatter,
                "resume checkpoint step {checkpoint_step} is beyond requested --max-steps {max_steps}; increase --max-steps or start a fresh output directory"
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
            Self::Resume { source } => Some(source),
            Self::Core { source } => Some(source),
            Self::Export { source } => Some(source),
            Self::BurnRecord { source } => Some(source),
            Self::Step { source } => Some(source),
            Self::Data { source } => Some(source),
            Self::Warmstart { source } => Some(source),
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
            | Self::ResumeCheckpointInvalid { .. }
            | Self::ResumeConfigHashMismatch { .. }
            | Self::ResumeSeedMismatch { .. }
            | Self::ResumeStepBeyondTarget { .. } => None,
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

impl From<crate::resume::ResumeError> for TrainerError {
    fn from(source: crate::resume::ResumeError) -> Self {
        Self::Resume { source }
    }
}

impl From<lewm_core::LewmCoreError> for TrainerError {
    fn from(source: lewm_core::LewmCoreError) -> Self {
        Self::Core { source }
    }
}

impl From<lewm_core::export::ExportError> for TrainerError {
    fn from(source: lewm_core::export::ExportError) -> Self {
        Self::Export { source }
    }
}

impl From<burn_core::record::RecorderError> for TrainerError {
    fn from(source: burn_core::record::RecorderError) -> Self {
        Self::BurnRecord { source }
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

impl From<WarmstartError> for TrainerError {
    fn from(source: WarmstartError) -> Self {
        Self::Warmstart { source }
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

/// Provenance for a SO-100 warm-start transfer.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WarmstartReport {
    /// Source checkpoint path used for the warm-start.
    pub source_path: String,
    /// SHA-256 digest of the source checkpoint bytes as lowercase hex.
    pub source_sha256: String,
    /// Model parameters copied verbatim from the `PushT` checkpoint.
    pub transferred_parameters: Vec<String>,
    /// SO-100 action-encoder parameters intentionally kept from fresh init.
    pub preserved_action_encoder_parameters: Vec<String>,
    /// Optimizer-state entries discarded during transfer.
    pub dropped_optimizer_state_entries: usize,
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
    /// Explicitly scopes the bounded training implementation.
    pub mode: String,
    /// Optional SO-100 warm-start provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmstart: Option<WarmstartReport>,
    /// Deterministic loss observations.
    pub losses: Vec<TrainLossPoint>,
}

/// Inputs for [`write_train_artifacts`].
#[derive(Clone, Copy, Debug)]
pub struct TrainArtifactRequest<'a> {
    /// Output directory used for reports and checkpoints.
    pub output_dir: &'a Path,
    /// Loaded root config.
    pub root: &'a RootConfig,
    /// Twelve-hex canonical config hash.
    pub config_hash: &'a str,
    /// Optional caller-provided dataset directory.
    pub data_dir: Option<&'a Path>,
    /// Target optimizer step for this invocation.
    pub max_steps: u64,
    /// Run-global seed.
    pub seed: u64,
    /// Requested device string.
    pub device: &'a str,
    /// Resume from the latest complete checkpoint when `run_id.txt` exists.
    pub resume_if_present: bool,
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

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
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

fn step_from_sidecar_file_name(file_name: &str) -> Option<u64> {
    let step = file_name.strip_prefix("step_")?.strip_suffix(".json")?;
    if step.len() != 7 || !step.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    step.parse().ok()
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
/// This is a real data-plane train path for the config-shaped host `LeWM` module
/// stack: it consumes `PushT` windows when the dataset exists and falls back to
/// an explicit `PushT`-compatible fixture when the default local path is absent.
/// It keeps the full artifact contract while the Burn `ViT` parity path remains a
/// separate implementation milestone.
///
/// # Errors
///
/// Returns an error if the config is not `PushT`, max steps are invalid, a
/// caller-provided data directory is missing, dataset reads fail, a train step
/// guard rejects the update, or artifact/checkpoint writes fail.
pub fn write_train_artifacts(
    request: TrainArtifactRequest<'_>,
) -> Result<TrainRunReport, TrainerError> {
    let output_dir = request.output_dir;
    fs::create_dir_all(output_dir).map_err(|source| io_error(output_dir, source))?;

    match &request.root.dataset {
        DatasetConfig::Pusht(_) => write_pusht_train_artifacts(request),
        DatasetConfig::So100(_) => write_so100_train_artifacts(request),
    }
}

fn write_pusht_train_artifacts(
    request: TrainArtifactRequest<'_>,
) -> Result<TrainRunReport, TrainerError> {
    match request.root.experimental.pusht_train_mode {
        PushtTrainMode::BoundedModule => write_pusht_bounded_train_artifacts(request),
        PushtTrainMode::FullBurnJepa => write_pusht_burn_jepa_train_artifacts(request),
    }
}

fn write_pusht_bounded_train_artifacts(
    request: TrainArtifactRequest<'_>,
) -> Result<TrainRunReport, TrainerError> {
    let output_dir = request.output_dir;

    let source = open_pusht_training_source(request.root, request.data_dir)?;
    let startup = detect_resume(
        output_dir,
        request.resume_if_present,
        option_env!("LEWM_GIT_SHA").unwrap_or("unknown"),
    )?;
    let mut previous_losses = Vec::new();
    let start = match startup {
        StartupMode::Fresh => {
            write_train_run_id(output_dir, PUSHT_BOUNDED_LEWM_RUN_ID)?;
            PushtFullLewmTrainingStart::fresh(request.root, request.seed)?
        },
        StartupMode::Resume(plan) => {
            previous_losses = read_train_losses(output_dir)?;
            restore_pusht_full_lewm_start(
                &plan,
                request.root,
                request.config_hash,
                request.max_steps,
                request.seed,
                &previous_losses,
            )?
        },
    };

    if u64::from(start.start_step) == request.max_steps {
        let paths = CheckpointPaths::for_step(output_dir, request.max_steps);
        return write_train_report(
            output_dir,
            request,
            source.description(),
            source.len(),
            previous_losses,
            &paths,
            start.batch_size,
            start.grad_explosion_events,
            PUSHT_BOUNDED_LEWM_MODE,
            start.warmstart,
        );
    }

    let outcome = run_pusht_full_lewm_training(&source, request.root, request.max_steps, start)?;
    let paths = write_pusht_full_lewm_checkpoint(
        output_dir,
        request.config_hash,
        request.seed,
        request.max_steps,
        &outcome,
    )?;
    previous_losses.extend(outcome.losses.clone());
    write_train_report(
        output_dir,
        request,
        source.description(),
        source.len(),
        previous_losses,
        &paths,
        outcome.batch_size,
        outcome.grad_explosion_events,
        PUSHT_BOUNDED_LEWM_MODE,
        outcome.warmstart,
    )
}

fn write_pusht_burn_jepa_train_artifacts(
    request: TrainArtifactRequest<'_>,
) -> Result<TrainRunReport, TrainerError> {
    let output_dir = request.output_dir;

    let source = open_pusht_training_source(request.root, request.data_dir)?;
    let startup = detect_resume(
        output_dir,
        request.resume_if_present,
        option_env!("LEWM_GIT_SHA").unwrap_or("unknown"),
    )?;
    match startup {
        StartupMode::Fresh => write_train_run_id(output_dir, PUSHT_FULL_BURN_JEPA_RUN_ID)?,
        StartupMode::Resume(plan) => {
            return Err(TrainerError::ResumeCheckpointInvalid {
                reason: format!(
                    "full Burn/Jepa PushT resume is not implemented yet; found checkpoint step {} in {}",
                    plan.step,
                    plan.sidecar_path.display()
                ),
            });
        },
    }

    let outcome = run_pusht_burn_jepa_training(&source, request.root, request.max_steps)?;
    let paths = write_pusht_burn_jepa_checkpoint(
        output_dir,
        request.config_hash,
        request.seed,
        request.max_steps,
        &outcome,
    )?;
    write_train_report(
        output_dir,
        request,
        source.description(),
        source.len(),
        outcome.losses.clone(),
        &paths,
        outcome.batch_size,
        outcome.grad_explosion_events,
        PUSHT_FULL_BURN_JEPA_MODE,
        None,
    )
}

fn write_so100_train_artifacts(
    request: TrainArtifactRequest<'_>,
) -> Result<TrainRunReport, TrainerError> {
    let output_dir = request.output_dir;

    let source = open_so100_training_source(request.root)?;
    let startup = detect_resume(
        output_dir,
        request.resume_if_present,
        option_env!("LEWM_GIT_SHA").unwrap_or("unknown"),
    )?;
    let mut previous_losses = Vec::new();
    let start = match startup {
        StartupMode::Fresh => {
            write_train_run_id(output_dir, SO100_FULL_LEWM_RUN_ID)?;
            fresh_so100_full_lewm_start(request.root, request.seed)?
        },
        StartupMode::Resume(plan) => {
            previous_losses = read_train_losses(output_dir)?;
            restore_so100_full_lewm_start(
                &plan,
                request.root,
                request.config_hash,
                request.max_steps,
                request.seed,
                &previous_losses,
            )?
        },
    };

    if u64::from(start.start_step) == request.max_steps {
        let paths = CheckpointPaths::for_step(output_dir, request.max_steps);
        return write_train_report(
            output_dir,
            request,
            source.description(),
            source.len(),
            previous_losses,
            &paths,
            start.batch_size,
            start.grad_explosion_events,
            "so100-full-module-lewm",
            start.warmstart,
        );
    }

    let outcome = run_so100_full_lewm_training(&source, request.root, request.max_steps, start)?;
    let paths = write_so100_full_lewm_checkpoint(
        output_dir,
        request.config_hash,
        request.seed,
        request.max_steps,
        &outcome,
    )?;
    previous_losses.extend(outcome.losses.clone());
    write_train_report(
        output_dir,
        request,
        source.description(),
        source.len(),
        previous_losses,
        &paths,
        outcome.batch_size,
        outcome.grad_explosion_events,
        "so100-full-module-lewm",
        outcome.warmstart,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_train_report(
    output_dir: &Path,
    request: TrainArtifactRequest<'_>,
    data_source: String,
    dataset_windows: usize,
    losses: Vec<TrainLossPoint>,
    paths: &CheckpointPaths,
    batch_size: usize,
    grad_explosion_events: u64,
    mode: &str,
    warmstart: Option<WarmstartReport>,
) -> Result<TrainRunReport, TrainerError> {
    let checkpoint_step = paths
        .sidecar
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(step_from_sidecar_file_name)
        .unwrap_or(request.max_steps);
    let report = TrainRunReport {
        schema_version: "1.0.0".to_owned(),
        kind: "lewm-rs-train-report".to_owned(),
        config_hash: request.config_hash.to_owned(),
        output_dir: output_dir.display().to_string(),
        data_dir: request.data_dir.map(|path| path.display().to_string()),
        data_source,
        dataset_windows,
        max_steps: request.max_steps,
        steps_completed: checkpoint_step,
        batch_size,
        seed: request.seed,
        device: request.device.to_owned(),
        initial_loss: first_train_loss(&losses),
        final_loss: last_train_loss(&losses),
        loss_decreased: last_train_loss(&losses) < first_train_loss(&losses),
        checkpoint_step,
        checkpoint_complete: paths.is_complete(),
        checkpoint_files: checkpoint_file_names(paths),
        grad_explosion_events,
        mode: mode.to_owned(),
        warmstart,
        losses,
    };

    let losses_path = output_dir.join("train_losses.jsonl");
    let mut losses_jsonl = String::new();
    for point in &report.losses {
        losses_jsonl.push_str(&serde_json::to_string(point)?);
        losses_jsonl.push('\n');
    }
    fs::write(&losses_path, losses_jsonl).map_err(|source| io_error(&losses_path, source))?;

    let report_path = output_dir.join("train_report.json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .map_err(|source| io_error(&report_path, source))?;

    Ok(report)
}

/// Abstract dataset interface required by the full `LeWM` training loop.
///
/// Implementors expose a logical sample count and indexed sample access, so
/// [`run_full_lewm_training`] can drive both `PushT` and SO-100 datasets
/// through one shared loop. Implementations must:
///
/// - report a stable [`len`] for the duration of a training run (the loop
///   builds dataset indices via `index % len`-style shuffling);
/// - return semantically identical samples for identical `(index, run)`
///   pairs to preserve the RFC 0013 determinism guarantees;
/// - surface I/O or schema errors via [`TrainerError`] rather than panicking.
///
/// [`len`]: TrainingSampleSource::len
trait TrainingSampleSource {
    /// Number of indexable samples available in this dataset view.
    fn len(&self) -> usize;
    /// Materialize the sample at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError::Data`] when the underlying dataset rejects the
    /// index or the sample fails downstream validation.
    fn get(&self, index: usize) -> Result<PushtSample, TrainerError>;
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
    fn description(&self) -> String {
        match self {
            Self::Hdf5 { path, .. } => format!("pusht-hdf5:{}", path.display()),
            Self::Fixture { descriptor, .. } => descriptor.clone(),
        }
    }
}

impl TrainingSampleSource for PushtTrainingSource {
    fn len(&self) -> usize {
        match self {
            Self::Hdf5 { dataset, .. } => dataset.len(),
            Self::Fixture { samples, .. } => samples.len(),
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

#[derive(Debug)]
enum So100TrainingSource {
    Hdf5 {
        dataset: Box<So100Dataset>,
        path: PathBuf,
    },
}

impl So100TrainingSource {
    fn description(&self) -> String {
        match self {
            Self::Hdf5 { path, .. } => format!("so100-hdf5:{}", path.display()),
        }
    }
}

impl TrainingSampleSource for So100TrainingSource {
    fn len(&self) -> usize {
        match self {
            Self::Hdf5 { dataset, .. } => dataset.len(),
        }
    }

    fn get(&self, index: usize) -> Result<PushtSample, TrainerError> {
        match self {
            Self::Hdf5 { dataset, .. } => dataset.get(index).map_err(TrainerError::from),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct PushtFullLewmAdamWState {
    step: i32,
    params: Vec<ScalarAdamWParamState>,
}

impl PushtFullLewmAdamWState {
    fn new(param_count: usize) -> Self {
        Self {
            step: 0,
            params: vec![ScalarAdamWParamState::default(); param_count],
        }
    }
}

#[derive(Clone, Debug)]
struct PushtFullLewmOutcome {
    losses: Vec<TrainLossPoint>,
    model: PushtFullLewmCore,
    optimizer: PushtFullLewmAdamWState,
    rng_streams: RestoredRngStreams,
    warmstart: Option<WarmstartReport>,
    step: u64,
    batch_size: usize,
    samples_seen: u64,
    grad_explosion_events: u64,
}

#[derive(Clone, Debug)]
struct PushtBurnJepaOutcome {
    losses: Vec<TrainLossPoint>,
    model: Jepa<PushtBurnAutodiffBackend>,
    rng_streams: RestoredRngStreams,
    step: u64,
    batch_size: usize,
    samples_seen: u64,
    grad_explosion_events: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PushtFullLewmRecord {
    schema_version: String,
    kind: String,
    step: u64,
    params: Vec<f64>,
    adamw_step: i32,
    #[serde(default)]
    adamw_params: Vec<ScalarAdamWParamState>,
    samples_seen: u64,
    #[serde(default)]
    scheduler_total_steps: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    warmstart: Option<WarmstartReport>,
}

#[derive(Debug)]
struct PushtFullLewmTrainingStart {
    model: PushtFullLewmCore,
    optimizer: PushtFullLewmAdamWState,
    rng_streams: RestoredRngStreams,
    warmstart: Option<WarmstartReport>,
    start_step: u32,
    batch_size: usize,
    samples_seen: u64,
    grad_explosion_events: u64,
}

impl PushtFullLewmTrainingStart {
    fn fresh(root: &RootConfig, seed: u64) -> Result<Self, TrainerError> {
        let model = PushtFullLewmCore::new(&root.model, seed).map_err(full_lewm_error)?;
        Ok(Self {
            optimizer: PushtFullLewmAdamWState::new(model.parameter_count()),
            rng_streams: fresh_pusht_full_lewm_rng_streams(seed)?,
            model,
            warmstart: None,
            start_step: 0,
            batch_size: train_batch_size(root)?,
            samples_seen: 0,
            grad_explosion_events: 0,
        })
    }
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

fn open_so100_training_source(root: &RootConfig) -> Result<So100TrainingSource, TrainerError> {
    let DatasetConfig::So100(config) = &root.dataset else {
        return Err(TrainerError::UnsupportedTrainDataset {
            kind: dataset_kind_name(&root.dataset).to_owned(),
        });
    };

    let hdf5_path = config.hdf5_path.clone();
    if !hdf5_path.exists() {
        return Err(TrainerError::MissingTrainDataPath { path: hdf5_path });
    }

    let so100_config = DataSo100Config {
        hdf5_path: hdf5_path.clone(),
        split: data_split(config.split),
        horizon: config.horizon,
        history_size: config.history_size,
        seed: Some(config.seed),
        camera_view: so100_camera_view(config.camera_view),
        stats_path: None,
    };
    let dataset = So100Dataset::from_hdf5(so100_config)?;
    Ok(So100TrainingSource::Hdf5 {
        dataset: Box::new(dataset),
        path: hdf5_path,
    })
}

fn so100_camera_view(view: crate::config::CameraView) -> lewm_data::so100::CameraView {
    match view {
        crate::config::CameraView::Top => lewm_data::so100::CameraView::Top,
        crate::config::CameraView::Wrist => lewm_data::so100::CameraView::Wrist,
    }
}

fn write_train_run_id(output_dir: &Path, run_id: &str) -> Result<(), TrainerError> {
    let path = output_dir.join(RUN_ID_FILE);
    fs::write(&path, format!("{run_id}\n")).map_err(|source| io_error(&path, source))
}

fn read_train_losses(output_dir: &Path) -> Result<Vec<TrainLossPoint>, TrainerError> {
    let path = output_dir.join("train_losses.jsonl");
    let raw = match fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => return Err(io_error(&path, source)),
    };

    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).map_err(TrainerError::from))
        .collect()
}

fn fresh_so100_full_lewm_start(
    root: &RootConfig,
    seed: u64,
) -> Result<PushtFullLewmTrainingStart, TrainerError> {
    let mut start = PushtFullLewmTrainingStart::fresh(root, seed)?;
    if let Some(source_path) = root.training.warmstart_from.as_deref() {
        let warmstart = apply_so100_warmstart(&mut start.model, root, seed, source_path)?;
        start.optimizer = PushtFullLewmAdamWState::new(start.model.parameter_count());
        start.warmstart = Some(warmstart);
    }
    Ok(start)
}

fn apply_so100_warmstart(
    target_model: &mut PushtFullLewmCore,
    root: &RootConfig,
    seed: u64,
    source_path: &Path,
) -> Result<WarmstartReport, TrainerError> {
    let source_bytes = fs::read(source_path).map_err(|source| io_error(source_path, source))?;
    let source_record: PushtFullLewmRecord = serde_json::from_slice(&source_bytes)?;
    let mut source_config = root.model.clone();
    source_config.action_encoder.input_dim = root.model.action_encoder.smoothed_dim;
    let mut source_model = PushtFullLewmCore::new(&source_config, seed).map_err(full_lewm_error)?;
    validate_warmstart_source_record(&source_record, &source_model)?;
    restore_pusht_full_lewm_model_params(&mut source_model, &source_record.params)?;

    let initialized_so100 = full_lewm_train_state(target_model)?;
    let pusht_checkpoint = full_lewm_train_state(&source_model)?;
    let loaded = load_warmstart(
        &root.model,
        initialized_so100,
        &pusht_checkpoint,
        source_path,
    )?;
    apply_full_lewm_train_state(target_model, &loaded.state)?;
    Ok(WarmstartReport::from(loaded.provenance))
}

fn validate_warmstart_source_record(
    record: &PushtFullLewmRecord,
    source_model: &PushtFullLewmCore,
) -> Result<(), TrainerError> {
    if !is_supported_pusht_bounded_record_kind(&record.kind) {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "warm-start source has unexpected record kind {:?}",
                record.kind
            ),
        });
    }
    if record.schema_version != "1.1.0" {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "warm-start source schema_version {:?} is not supported; expected \"1.1.0\"",
                record.schema_version
            ),
        });
    }
    if record.params.len() != source_model.parameter_count() {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "warm-start source record has {} model parameters, expected {}",
                record.params.len(),
                source_model.parameter_count()
            ),
        });
    }
    if record.params.iter().any(|value| !value.is_finite()) {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: "warm-start source contains non-finite model parameters".to_owned(),
        });
    }
    Ok(())
}

fn is_supported_pusht_bounded_record_kind(kind: &str) -> bool {
    matches!(
        kind,
        PUSHT_BOUNDED_LEWM_RECORD_KIND | PUSHT_LEGACY_FULL_LEWM_RECORD_KIND
    )
}

fn is_supported_pusht_bounded_run_id(run_id: &str) -> bool {
    matches!(
        run_id,
        PUSHT_BOUNDED_LEWM_RUN_ID | PUSHT_LEGACY_FULL_LEWM_RUN_ID
    )
}

fn full_lewm_train_state(model: &PushtFullLewmCore) -> Result<TrainStateRecord, TrainerError> {
    let mut state = TrainStateRecord::default();
    for spec in model.parameter_specs() {
        let values = model
            .parameter_values(spec)
            .iter()
            .copied()
            .map(f32_from_f64)
            .collect();
        state.insert_model_param(
            spec.name.clone(),
            crate::warmstart::TensorRecord::new(spec.shape.clone(), values)?,
        );
    }
    Ok(state)
}

fn apply_full_lewm_train_state(
    model: &mut PushtFullLewmCore,
    state: &TrainStateRecord,
) -> Result<(), TrainerError> {
    let specs = model.parameter_specs().to_vec();
    let mut flat_index = 0;
    for spec in specs {
        let tensor =
            state
                .model
                .get(&spec.name)
                .ok_or_else(|| TrainerError::ResumeCheckpointInvalid {
                    reason: format!("warm-start output is missing {}", spec.name),
                })?;
        if tensor.shape != spec.shape {
            return Err(TrainerError::ResumeCheckpointInvalid {
                reason: format!(
                    "warm-start output for {} has shape {:?}, expected {:?}",
                    spec.name, tensor.shape, spec.shape
                ),
            });
        }
        for value in &tensor.values {
            model.set_parameter(flat_index, f64::from(*value));
            flat_index += 1;
        }
    }
    if flat_index != model.parameter_count() {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "warm-start output populated {flat_index} parameters, expected {}",
                model.parameter_count()
            ),
        });
    }
    Ok(())
}

impl From<WarmstartProvenance> for WarmstartReport {
    fn from(provenance: WarmstartProvenance) -> Self {
        Self {
            source_path: provenance.source_path.display().to_string(),
            source_sha256: provenance.source_sha256,
            transferred_parameters: provenance.transferred_parameters,
            preserved_action_encoder_parameters: provenance.preserved_action_encoder_parameters,
            dropped_optimizer_state_entries: provenance.dropped_optimizer_state_entries,
        }
    }
}

fn restore_pusht_full_lewm_start(
    plan: &crate::resume::ResumePlan,
    root: &RootConfig,
    config_hash: &str,
    max_steps: u64,
    seed: u64,
    previous_losses: &[TrainLossPoint],
) -> Result<PushtFullLewmTrainingStart, TrainerError> {
    if !is_supported_pusht_bounded_run_id(&plan.run_id) {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "expected bounded PushT run_id {PUSHT_BOUNDED_LEWM_RUN_ID:?}, found {:?}",
                plan.run_id
            ),
        });
    }
    if plan.step > max_steps {
        return Err(TrainerError::ResumeStepBeyondTarget {
            checkpoint_step: plan.step,
            max_steps,
        });
    }

    let loaded = load_checkpoint(&plan.sidecar_path)?;
    let sidecar = &loaded.sidecar;
    if sidecar.schema_version != CHECKPOINT_SCHEMA_VERSION {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "unsupported sidecar schema_version {:?}, expected {CHECKPOINT_SCHEMA_VERSION:?}",
                sidecar.schema_version
            ),
        });
    }
    if sidecar.config_hash != config_hash {
        return Err(TrainerError::ResumeConfigHashMismatch {
            checkpoint: sidecar.config_hash.clone(),
            current: config_hash.to_owned(),
        });
    }
    if sidecar.rng_state.global_seed != seed {
        return Err(TrainerError::ResumeSeedMismatch {
            checkpoint: sidecar.rng_state.global_seed,
            current: seed,
        });
    }
    if sidecar.rng_state.step_at_save != sidecar.step {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "rng_state.step_at_save {} does not match sidecar step {}",
                sidecar.rng_state.step_at_save, sidecar.step
            ),
        });
    }
    if let Some(last_loss) = previous_losses.last()
        && last_loss.step != sidecar.step
    {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "train_losses.jsonl ends at step {}, but latest checkpoint is step {}",
                last_loss.step, sidecar.step
            ),
        });
    }

    let record: PushtFullLewmRecord = serde_json::from_slice(&loaded.burn_record)?;
    validate_pusht_full_lewm_record(&record, root, sidecar.step, max_steps)?;
    let mut model = PushtFullLewmCore::new(&root.model, seed).map_err(full_lewm_error)?;
    restore_pusht_full_lewm_model_params(&mut model, &record.params)?;
    validate_full_lewm_safetensors(&model, &loaded.safetensors_bytes)?;
    let rng_streams = restore_rng_streams(&plan.rng_state)?;

    Ok(PushtFullLewmTrainingStart {
        model,
        optimizer: PushtFullLewmAdamWState {
            step: record.adamw_step,
            params: record.adamw_params,
        },
        rng_streams,
        warmstart: record.warmstart.clone(),
        start_step: train_steps_as_u32(sidecar.step)?,
        batch_size: train_batch_size(root)?,
        samples_seen: record.samples_seen,
        grad_explosion_events: metric_u64(
            sidecar.metrics_last_step.get("train/grad_explosion_events"),
        )
        .unwrap_or(0),
    })
}

fn restore_so100_full_lewm_start(
    plan: &crate::resume::ResumePlan,
    root: &RootConfig,
    config_hash: &str,
    max_steps: u64,
    seed: u64,
    previous_losses: &[TrainLossPoint],
) -> Result<PushtFullLewmTrainingStart, TrainerError> {
    if plan.run_id != SO100_FULL_LEWM_RUN_ID {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "expected run_id {SO100_FULL_LEWM_RUN_ID:?}, found {:?}",
                plan.run_id
            ),
        });
    }
    if plan.step > max_steps {
        return Err(TrainerError::ResumeStepBeyondTarget {
            checkpoint_step: plan.step,
            max_steps,
        });
    }

    let loaded = load_checkpoint(&plan.sidecar_path)?;
    let sidecar = &loaded.sidecar;
    if sidecar.schema_version != CHECKPOINT_SCHEMA_VERSION {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "unsupported sidecar schema_version {:?}, expected {CHECKPOINT_SCHEMA_VERSION:?}",
                sidecar.schema_version
            ),
        });
    }
    if sidecar.config_hash != config_hash {
        return Err(TrainerError::ResumeConfigHashMismatch {
            checkpoint: sidecar.config_hash.clone(),
            current: config_hash.to_owned(),
        });
    }
    if sidecar.rng_state.global_seed != seed {
        return Err(TrainerError::ResumeSeedMismatch {
            checkpoint: sidecar.rng_state.global_seed,
            current: seed,
        });
    }
    if sidecar.rng_state.step_at_save != sidecar.step {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "rng_state.step_at_save {} does not match sidecar step {}",
                sidecar.rng_state.step_at_save, sidecar.step
            ),
        });
    }
    if let Some(last_loss) = previous_losses.last()
        && last_loss.step != sidecar.step
    {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "train_losses.jsonl ends at step {}, but latest checkpoint is step {}",
                last_loss.step, sidecar.step
            ),
        });
    }

    let record: PushtFullLewmRecord = serde_json::from_slice(&loaded.burn_record)?;
    validate_so100_full_lewm_record(&record, root, sidecar.step, max_steps)?;
    let mut model = PushtFullLewmCore::new(&root.model, seed).map_err(full_lewm_error)?;
    restore_pusht_full_lewm_model_params(&mut model, &record.params)?;
    validate_full_lewm_safetensors(&model, &loaded.safetensors_bytes)?;
    let rng_streams = restore_rng_streams(&plan.rng_state)?;

    Ok(PushtFullLewmTrainingStart {
        model,
        optimizer: PushtFullLewmAdamWState {
            step: record.adamw_step,
            params: record.adamw_params,
        },
        rng_streams,
        warmstart: record.warmstart.clone(),
        start_step: train_steps_as_u32(sidecar.step)?,
        batch_size: train_batch_size(root)?,
        samples_seen: record.samples_seen,
        grad_explosion_events: metric_u64(
            sidecar.metrics_last_step.get("train/grad_explosion_events"),
        )
        .unwrap_or(0),
    })
}

fn validate_pusht_full_lewm_record(
    record: &PushtFullLewmRecord,
    root: &RootConfig,
    sidecar_step: u64,
    max_steps: u64,
) -> Result<(), TrainerError> {
    if !is_supported_pusht_bounded_record_kind(&record.kind) {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!("unexpected record kind {:?}", record.kind),
        });
    }
    if record.schema_version != "1.1.0" {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "record schema_version {:?} is not resumable; expected \"1.1.0\" with AdamW state",
                record.schema_version
            ),
        });
    }
    if record.step != sidecar_step {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "record step {} does not match sidecar step {sidecar_step}",
                record.step
            ),
        });
    }
    if record.scheduler_total_steps > max_steps {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "record scheduler_total_steps {} is greater than requested --max-steps {max_steps}",
                record.scheduler_total_steps
            ),
        });
    }
    let expected_adamw_step =
        i32::try_from(record.step).map_err(|_| TrainerError::ResumeCheckpointInvalid {
            reason: format!("record step {} exceeds AdamW step range", record.step),
        })?;
    if record.adamw_step != expected_adamw_step {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "record adamw_step {} does not match optimizer step {}",
                record.adamw_step, record.step
            ),
        });
    }
    let model = PushtFullLewmCore::new(&root.model, root.training.seed).map_err(full_lewm_error)?;
    if record.params.len() != model.parameter_count() {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "record has {} model parameters, expected {}",
                record.params.len(),
                model.parameter_count()
            ),
        });
    }
    if record.adamw_params.len() != model.parameter_count() {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "record has {} AdamW parameter states, expected {}",
                record.adamw_params.len(),
                model.parameter_count()
            ),
        });
    }
    if record.params.iter().any(|value| !value.is_finite()) {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: "record contains non-finite model parameters".to_owned(),
        });
    }
    if record
        .adamw_params
        .iter()
        .any(|state| !state.first_moment.is_finite() || !state.second_moment.is_finite())
    {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: "record contains non-finite AdamW moments".to_owned(),
        });
    }
    Ok(())
}

fn validate_so100_full_lewm_record(
    record: &PushtFullLewmRecord,
    root: &RootConfig,
    sidecar_step: u64,
    max_steps: u64,
) -> Result<(), TrainerError> {
    if record.kind != "lewm-rs-so100-full-module-lewm-record" {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!("unexpected record kind {:?}", record.kind),
        });
    }
    if record.schema_version != "1.1.0" {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "record schema_version {:?} is not resumable; expected \"1.1.0\" with AdamW state",
                record.schema_version
            ),
        });
    }
    if record.step != sidecar_step {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "record step {} does not match sidecar step {sidecar_step}",
                record.step
            ),
        });
    }
    if record.scheduler_total_steps > max_steps {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "record scheduler_total_steps {} is greater than requested --max-steps {max_steps}",
                record.scheduler_total_steps
            ),
        });
    }
    let expected_adamw_step =
        i32::try_from(record.step).map_err(|_| TrainerError::ResumeCheckpointInvalid {
            reason: format!("record step {} exceeds AdamW step range", record.step),
        })?;
    if record.adamw_step != expected_adamw_step {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "record adamw_step {} does not match optimizer step {}",
                record.adamw_step, record.step
            ),
        });
    }
    let model = PushtFullLewmCore::new(&root.model, root.training.seed).map_err(full_lewm_error)?;
    if record.params.len() != model.parameter_count() {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "record has {} model parameters, expected {}",
                record.params.len(),
                model.parameter_count()
            ),
        });
    }
    if record.adamw_params.len() != model.parameter_count() {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "record has {} AdamW parameter states, expected {}",
                record.adamw_params.len(),
                model.parameter_count()
            ),
        });
    }
    if record.params.iter().any(|value| !value.is_finite()) {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: "record contains non-finite model parameters".to_owned(),
        });
    }
    if record
        .adamw_params
        .iter()
        .any(|state| !state.first_moment.is_finite() || !state.second_moment.is_finite())
    {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: "record contains non-finite AdamW moments".to_owned(),
        });
    }
    Ok(())
}

fn restore_pusht_full_lewm_model_params(
    model: &mut PushtFullLewmCore,
    params: &[f64],
) -> Result<(), TrainerError> {
    if params.len() != model.parameter_count() {
        return Err(TrainerError::ResumeCheckpointInvalid {
            reason: format!(
                "cannot restore {} parameters into model with {} parameters",
                params.len(),
                model.parameter_count()
            ),
        });
    }
    for (index, value) in params.iter().copied().enumerate() {
        model.set_parameter(index, value);
    }
    Ok(())
}

fn validate_full_lewm_safetensors(
    model: &PushtFullLewmCore,
    bytes: &[u8],
) -> Result<(), TrainerError> {
    let tensors = SafeTensors::deserialize(bytes).map_err(CheckpointError::from)?;
    for spec in model.parameter_specs() {
        let tensor = tensors.tensor(&spec.name).map_err(CheckpointError::from)?;
        if tensor.dtype() != Dtype::F32 || tensor.shape() != spec.shape.as_slice() {
            return Err(TrainerError::ResumeCheckpointInvalid {
                reason: format!(
                    "safetensors mirror for {} has dtype {:?} shape {:?}, expected F32 {:?}",
                    spec.name,
                    tensor.dtype(),
                    tensor.shape(),
                    spec.shape
                ),
            });
        }
        let mirror_values = f32_values(tensor.data());
        let expected_values = model
            .parameter_values(spec)
            .iter()
            .copied()
            .map(f32_from_f64)
            .collect::<Vec<_>>();
        if mirror_values != expected_values {
            return Err(TrainerError::ResumeCheckpointInvalid {
                reason: format!(
                    "safetensors mirror for {} does not match the model record",
                    spec.name
                ),
            });
        }
    }
    Ok(())
}

fn fresh_pusht_full_lewm_rng_streams(seed: u64) -> Result<RestoredRngStreams, TrainerError> {
    Ok(RestoredRngStreams {
        data_shuffle: substream_rng(seed, DATA_SHUFFLE_STREAM)?,
        sigreg_sketch: substream_rng(seed, SIGREG_SKETCH_STREAM)?,
        dropout: substream_rng(seed, DROPOUT_STREAM)?,
        cem: substream_rng(seed, CEM_STREAM)?,
        model_init: substream_rng(seed, MODEL_INIT_STREAM)?,
    })
}

fn train_batch_size(root: &RootConfig) -> Result<usize, TrainerError> {
    let batch_size_u32 = train_batch_size_as_u32(root.training.batch_size)?;
    usize::try_from(batch_size_u32).map_err(|_| TrainerError::InvalidTrainBatchSize {
        batch_size: root.training.batch_size,
    })
}

/// Run the full `LeWM` training loop against any dataset source.
///
/// The dataset-specific work — extracting the dataset config and shaping
/// raw samples into `PushtFullLewmExample` instances — is supplied by the
/// caller via `build_example`. The log tag (`"pusht"` / `"so100"`) appears
/// in the periodic ETA `eprintln!` so operators can disambiguate runs.
#[allow(clippy::too_many_lines)]
fn run_full_lewm_training<S, F>(
    source: &S,
    root: &RootConfig,
    max_steps: u64,
    start: PushtFullLewmTrainingStart,
    log_tag: &str,
    build_example: F,
) -> Result<PushtFullLewmOutcome, TrainerError>
where
    S: TrainingSampleSource + ?Sized,
    F: Fn(&PushtSample) -> Result<PushtFullLewmExample, TrainerError>,
{
    if !root.loss.lambda_sigreg.is_finite() {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "{log_tag} full LeWM lambda_sigreg must be finite"
            )),
        });
    }
    let total_steps = train_steps_as_u32(max_steps)?;
    if start.start_step > total_steps {
        return Err(TrainerError::ResumeStepBeyondTarget {
            checkpoint_step: u64::from(start.start_step),
            max_steps,
        });
    }
    let batch_size_u32 =
        u32::try_from(start.batch_size).map_err(|_| TrainerError::InvalidTrainBatchSize {
            batch_size: start.batch_size,
        })?;
    let batch_size = start.batch_size;
    let schedule = CosineWarmup::from_parts(
        root.training.lr_peak,
        root.training.lr_min,
        root.training.warmup_steps,
        total_steps.max(root.training.warmup_steps.saturating_add(1)),
    );
    let optim_config = OptimConfig::new()
        .with_beta1(root.training.betas.0)
        .with_beta2(root.training.betas.1)
        .with_weight_decay(root.training.weight_decay);
    let mut model = start.model;
    let mut optimizer = start.optimizer;
    let mut rng_streams = start.rng_streams;
    let mut guard = NanGuard::new();
    let mut losses = Vec::with_capacity(usize::try_from(total_steps).unwrap_or_default());
    let mut samples_seen = start.samples_seen;
    let mut grad_explosion_events = start.grad_explosion_events;
    let sigreg_weight = root.loss.lambda_sigreg.max(0.0);
    let train_start = Instant::now(); // determinism-lint: allow Instant::now (wall-clock ETA only)

    for step in (start.start_step.saturating_add(1))..=total_steps {
        let mut total_loss = 0.0;
        let mut pred_loss = 0.0;
        let mut sigreg_proxy_loss = 0.0;
        let mut sample_gradients = Vec::with_capacity(batch_size);
        for sample_offset in 0..batch_size {
            let dataset_index = training_sample_index(
                step,
                sample_offset,
                batch_size,
                source.len(),
                &mut rng_streams,
            )?;
            let sample = source.get(dataset_index)?;
            let example = build_example(&sample)?;
            let (sample_loss, gradients) = model
                .loss_and_gradients(&example, sigreg_weight)
                .map_err(full_lewm_error)?;
            total_loss += scale_loss_for_accumulation(sample_loss.total, batch_size_u32)?;
            pred_loss += scale_loss_for_accumulation(sample_loss.pred, batch_size_u32)?;
            sigreg_proxy_loss +=
                scale_loss_for_accumulation(sample_loss.sigreg_proxy, batch_size_u32)?;
            sample_gradients.push(gradients);
            samples_seen = samples_seen.saturating_add(1);
        }
        let _sigreg_rng_audit_word = rng_streams.sigreg_sketch.next_u64();

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

        let clip = clip_global_norm(&mut gradients, root.training.grad_clip_norm)?;
        if grad_explosion_artifact(step_u64, clip.grad_norm_pre).is_some() {
            grad_explosion_events += 1;
        }
        let learning_rate = schedule.lr(step);
        apply_pusht_full_lewm_adamw(
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
        if step == 1 || step % 100 == 0 || step == total_steps {
            let elapsed_s = train_start.elapsed().as_secs_f64();
            let steps_done = u64::from(step)
                .saturating_sub(u64::from(start.start_step))
                .max(1);
            #[allow(clippy::cast_precision_loss)]
            let secs_per_step = elapsed_s / steps_done as f64;
            let eta_s = secs_per_step * f64::from(total_steps - step);
            eprintln!(
                "[{log_tag} step {step}/{total_steps}] loss={total_loss:.6} pred={pred_loss:.6} \
                 lr={learning_rate:.2e} grad={:.3} elapsed={elapsed_s:.0}s eta={eta_s:.0}s",
                clip.grad_norm_post,
            );
        }
    }

    Ok(PushtFullLewmOutcome {
        losses,
        model,
        optimizer,
        rng_streams,
        warmstart: start.warmstart,
        step: max_steps,
        batch_size,
        samples_seen,
        grad_explosion_events,
    })
}

fn run_pusht_full_lewm_training(
    source: &PushtTrainingSource,
    root: &RootConfig,
    max_steps: u64,
    start: PushtFullLewmTrainingStart,
) -> Result<PushtFullLewmOutcome, TrainerError> {
    let DatasetConfig::Pusht(dataset_config) = &root.dataset else {
        return Err(TrainerError::UnsupportedTrainDataset {
            kind: dataset_kind_name(&root.dataset).to_owned(),
        });
    };
    run_full_lewm_training(source, root, max_steps, start, "pusht", |sample| {
        full_lewm_example_from_sample(sample, dataset_config, &root.model)
    })
}

#[allow(clippy::too_many_lines)]
fn run_pusht_burn_jepa_training(
    source: &PushtTrainingSource,
    root: &RootConfig,
    max_steps: u64,
) -> Result<PushtBurnJepaOutcome, TrainerError> {
    let DatasetConfig::Pusht(dataset_config) = &root.dataset else {
        return Err(TrainerError::UnsupportedTrainDataset {
            kind: dataset_kind_name(&root.dataset).to_owned(),
        });
    };
    validate_pusht_burn_jepa_config(root)?;

    let total_steps = train_steps_as_u32(max_steps)?;
    let batch_size = train_batch_size(root)?;
    let batch_size_u32 = train_batch_size_as_u32(batch_size)?;
    let device = burn_ndarray::NdArrayDevice::default();
    PushtBurnAutodiffBackend::seed(&device, root.training.seed);
    let mut model = Jepa::<PushtBurnAutodiffBackend>::init_with_seed(
        root.model.clone(),
        root.training.seed,
        &device,
    )?;
    let mut optimizer = AdamWConfig::new()
        .with_beta_1(f32_from_f64(root.training.betas.0))
        .with_beta_2(f32_from_f64(root.training.betas.1))
        .with_epsilon(f32_from_f64(ADAMW_EPSILON))
        .with_weight_decay(f32_from_f64(root.training.weight_decay))
        .init::<PushtBurnAutodiffBackend, Jepa<PushtBurnAutodiffBackend>>();
    let schedule = CosineWarmup::from_parts(
        root.training.lr_peak,
        root.training.lr_min,
        root.training.warmup_steps,
        total_steps.max(root.training.warmup_steps.saturating_add(1)),
    );
    let mut rng_streams = fresh_pusht_full_lewm_rng_streams(root.training.seed)?;
    let mut losses = Vec::with_capacity(usize::try_from(total_steps).unwrap_or_default());
    let mut samples_seen = 0_u64;
    let image_preproc = pusht_burn_jepa_image_preprocessor(root)?;
    let train_start = Instant::now(); // determinism-lint: allow Instant::now (wall-clock ETA only)

    for step in 1..=total_steps {
        PushtBurnAutodiffBackend::seed(&device, rng_streams.dropout.next_u64());
        let (pixels, actions) = pusht_burn_jepa_batch(
            source,
            dataset_config,
            &root.model,
            &image_preproc,
            step,
            batch_size,
            &mut rng_streams,
            device,
        )?;
        let losses_this_step = model.criterion(
            pixels,
            actions,
            root.loss.lambda_sigreg,
            &mut rng_streams.sigreg_sketch,
        )?;
        let total_loss = burn_scalar(&losses_this_step.total.clone().inner())?;
        let pred_loss = burn_scalar(&losses_this_step.pred.clone().inner())?;
        let sigreg_proxy_loss = burn_scalar(&losses_this_step.sigreg.clone().inner())?;
        let step_u64 = u64::from(step);
        if !total_loss.is_finite() {
            return Err(TrainerError::TrainGuardRejected {
                step: step_u64,
                reason: format!("full Burn/Jepa loss is non-finite: {total_loss}"),
            });
        }
        let gradients = GradientsParams::from_grads(losses_this_step.total.backward(), &model);
        let learning_rate = schedule.lr(step);
        model = optimizer.step(learning_rate, model, gradients);
        samples_seen = samples_seen.saturating_add(u64::from(batch_size_u32));
        losses.push(TrainLossPoint {
            step: step_u64,
            loss: total_loss,
            pred_loss,
            sigreg_proxy_loss,
            grad_norm_pre: 0.0,
            grad_norm_post: 0.0,
            learning_rate,
            samples_seen,
        });
        if step == 1 || step % 100 == 0 || step == total_steps {
            let elapsed_s = train_start.elapsed().as_secs_f64();
            let eta_s = elapsed_s / f64::from(step) * f64::from(total_steps - step);
            eprintln!(
                "[pusht-burn-jepa step {step}/{total_steps}] loss={total_loss:.6} \
                 pred={pred_loss:.6} sigreg={sigreg_proxy_loss:.6} lr={learning_rate:.2e} \
                 elapsed={elapsed_s:.0}s eta={eta_s:.0}s",
            );
        }
    }

    Ok(PushtBurnJepaOutcome {
        losses,
        model,
        rng_streams,
        step: max_steps,
        batch_size,
        samples_seen,
        grad_explosion_events: 0,
    })
}

fn run_so100_full_lewm_training(
    source: &So100TrainingSource,
    root: &RootConfig,
    max_steps: u64,
    start: PushtFullLewmTrainingStart,
) -> Result<PushtFullLewmOutcome, TrainerError> {
    let DatasetConfig::So100(dataset_config) = &root.dataset else {
        return Err(TrainerError::UnsupportedTrainDataset {
            kind: dataset_kind_name(&root.dataset).to_owned(),
        });
    };
    run_full_lewm_training(source, root, max_steps, start, "so100", |sample| {
        full_lewm_example_from_so100_sample(sample, dataset_config, &root.model)
    })
}

fn validate_pusht_burn_jepa_config(root: &RootConfig) -> Result<(), TrainerError> {
    if !root.loss.lambda_sigreg.is_finite() {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(
                "full Burn/Jepa lambda_sigreg must be finite".to_owned(),
            ),
        });
    }
    if root.model.encoder.num_channels != 3 {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "full Burn/Jepa PushT requires RGB encoder input, found {} channels",
                root.model.encoder.num_channels
            )),
        });
    }
    let DatasetConfig::Pusht(dataset) = &root.dataset else {
        return Err(TrainerError::UnsupportedTrainDataset {
            kind: dataset_kind_name(&root.dataset).to_owned(),
        });
    };
    let expected_action_dim = dataset
        .raw_action_dim
        .checked_mul(dataset.frameskip)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform(
                "PushT full Burn/Jepa action packing overflow".to_owned(),
            ),
        })?;
    if root.model.action_encoder.input_dim != expected_action_dim {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "full Burn/Jepa PushT action dim must be raw_action_dim * frameskip = {expected_action_dim}, found {}",
                root.model.action_encoder.input_dim
            )),
        });
    }
    Ok(())
}

fn pusht_burn_jepa_image_preprocessor(
    root: &RootConfig,
) -> Result<ImagePreprocessor, TrainerError> {
    let target_size =
        u32::try_from(root.model.encoder.image_size).map_err(|_| TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "encoder image_size {} does not fit u32",
                root.model.encoder.image_size
            )),
        })?;
    Ok(ImagePreprocessor {
        target_size,
        ..ImagePreprocessor::default()
    })
}

#[allow(clippy::too_many_arguments)]
fn pusht_burn_jepa_batch(
    source: &PushtTrainingSource,
    dataset: &crate::config::PushtDatasetConfig,
    model: &lewm_core::JepaConfig,
    image_preproc: &ImagePreprocessor,
    step: u32,
    batch_size: usize,
    rng_streams: &mut RestoredRngStreams,
    device: burn_ndarray::NdArrayDevice,
) -> Result<
    (
        Tensor<PushtBurnAutodiffBackend, 5>,
        Tensor<PushtBurnAutodiffBackend, 3>,
    ),
    TrainerError,
> {
    let image_size = model.encoder.image_size;
    let pixel_count = batch_size
        .checked_mul(model.horizon)
        .and_then(|value| value.checked_mul(model.encoder.num_channels))
        .and_then(|value| value.checked_mul(image_size))
        .and_then(|value| value.checked_mul(image_size))
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform(
                "full Burn/Jepa pixel batch shape overflow".to_owned(),
            ),
        })?;
    let action_count = batch_size
        .checked_mul(model.horizon)
        .and_then(|value| value.checked_mul(model.action_encoder.input_dim))
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform(
                "full Burn/Jepa action batch shape overflow".to_owned(),
            ),
        })?;
    let mut pixels = Vec::with_capacity(pixel_count);
    let mut actions = Vec::with_capacity(action_count);

    for sample_offset in 0..batch_size {
        let dataset_index =
            training_sample_index(step, sample_offset, batch_size, source.len(), rng_streams)?;
        let sample = source.get(dataset_index)?;
        let (sample_pixels, sample_actions) =
            pusht_burn_jepa_sample_tensors(&sample, dataset, model, image_preproc)?;
        pixels.extend(sample_pixels);
        actions.extend(sample_actions);
    }

    let pixels = Tensor::<PushtBurnAutodiffBackend, 5>::from_data(
        TensorData::new(
            pixels,
            [
                batch_size,
                model.horizon,
                model.encoder.num_channels,
                image_size,
                image_size,
            ],
        ),
        &device,
    );
    let actions = Tensor::<PushtBurnAutodiffBackend, 3>::from_data(
        TensorData::new(
            actions,
            [batch_size, model.horizon, model.action_encoder.input_dim],
        ),
        &device,
    );
    Ok((pixels, actions))
}

fn pusht_burn_jepa_sample_tensors(
    sample: &PushtSample,
    dataset: &crate::config::PushtDatasetConfig,
    model: &lewm_core::JepaConfig,
    image_preproc: &ImagePreprocessor,
) -> Result<(Vec<f32>, Vec<f32>), TrainerError> {
    validate_pusht_burn_jepa_sample(sample, dataset, model)?;
    let mut pixels = Vec::with_capacity(
        model
            .horizon
            .saturating_mul(model.encoder.num_channels)
            .saturating_mul(model.encoder.image_size)
            .saturating_mul(model.encoder.image_size),
    );
    for frame_index in 0..model.horizon {
        pixels.extend(pusht_burn_jepa_frame_pixels(
            sample,
            frame_index,
            image_preproc,
        )?);
    }

    let mut actions =
        Vec::with_capacity(model.horizon.saturating_mul(model.action_encoder.input_dim));
    for action_index in 0..model.horizon {
        actions.extend(
            full_lewm_packed_action(
                sample,
                action_index,
                dataset.raw_action_dim,
                dataset.frameskip,
                model.action_encoder.input_dim,
            )?
            .into_iter()
            .map(f32_from_f64),
        );
    }
    Ok((pixels, actions))
}

fn validate_pusht_burn_jepa_sample(
    sample: &PushtSample,
    dataset: &crate::config::PushtDatasetConfig,
    model: &lewm_core::JepaConfig,
) -> Result<(), TrainerError> {
    validate_full_lewm_sample(sample, dataset)?;
    if sample.frame_shape.0 < model.horizon {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "full Burn/Jepa PushT requires {} frames, found {}",
                model.horizon, sample.frame_shape.0
            )),
        });
    }
    if sample.action_shape.0 < model.horizon {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "full Burn/Jepa PushT requires {} action steps, found {}",
                model.horizon, sample.action_shape.0
            )),
        });
    }
    Ok(())
}

fn pusht_burn_jepa_frame_pixels(
    sample: &PushtSample,
    frame_index: usize,
    image_preproc: &ImagePreprocessor,
) -> Result<Vec<f32>, TrainerError> {
    let frame_stride = sample
        .frame_shape
        .1
        .checked_mul(sample.frame_shape.2)
        .and_then(|value| value.checked_mul(sample.frame_shape.3))
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform("full Burn/Jepa frame shape overflow".to_owned()),
        })?;
    let start = frame_index
        .checked_mul(frame_stride)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform("full Burn/Jepa frame offset overflow".to_owned()),
        })?;
    let end = start
        .checked_add(frame_stride)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform("full Burn/Jepa frame range overflow".to_owned()),
        })?;
    let frame = sample
        .frames_t
        .get(start..end)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "full Burn/Jepa frame {frame_index} exceeds frame shape {:?}",
                sample.frame_shape
            )),
        })?;
    let src_h = u32::try_from(sample.frame_shape.1).map_err(|_| TrainerError::Data {
        source: DataError::InvalidTransform(format!(
            "frame height {} does not fit u32",
            sample.frame_shape.1
        )),
    })?;
    let src_w = u32::try_from(sample.frame_shape.2).map_err(|_| TrainerError::Data {
        source: DataError::InvalidTransform(format!(
            "frame width {} does not fit u32",
            sample.frame_shape.2
        )),
    })?;
    image_preproc
        .apply(frame, src_h, src_w)
        .map_err(TrainerError::from)
}

fn burn_scalar<B: BurnBackend>(tensor: &Tensor<B, 1>) -> Result<f64, TrainerError> {
    let values = tensor
        .to_data()
        .to_vec::<f32>()
        .map_err(|source| TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "Burn scalar tensor conversion failed: {source}"
            )),
        })?;
    let value = values.first().copied().ok_or_else(|| TrainerError::Data {
        source: DataError::InvalidTransform("Burn scalar tensor was empty".to_owned()),
    })?;
    Ok(f64::from(value))
}

#[allow(clippy::needless_pass_by_value)]
fn full_lewm_error(source: PushtFullLewmError) -> TrainerError {
    TrainerError::Data {
        source: DataError::InvalidTransform(format!("PushT full LeWM error: {source}")),
    }
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

fn training_sample_index(
    step: u32,
    sample_offset: usize,
    batch_size: usize,
    dataset_len: usize,
    rng_streams: &mut RestoredRngStreams,
) -> Result<usize, TrainerError> {
    if dataset_len == 0 {
        return Err(TrainerError::Data {
            source: DataError::EmptyDataset("PushT training source has no samples".to_owned()),
        });
    }
    let step_index = usize::try_from(step.saturating_sub(1)).unwrap_or_default();
    let sequential_index = step_index
        .saturating_mul(batch_size)
        .saturating_add(sample_offset);
    let shuffle_offset = usize::try_from(rng_streams.data_shuffle.next_u64()).unwrap_or_default();
    Ok(sequential_index.wrapping_add(shuffle_offset) % dataset_len)
}

fn full_lewm_example_from_sample(
    sample: &PushtSample,
    dataset: &crate::config::PushtDatasetConfig,
    model: &lewm_core::JepaConfig,
) -> Result<PushtFullLewmExample, TrainerError> {
    validate_full_lewm_sample(sample, dataset)?;
    if sample.frame_shape.0 <= model.history_size {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "PushT full LeWM requires at least history_size + 1 frames, found {} frames for history_size {}",
                sample.frame_shape.0, model.history_size
            )),
        });
    }

    let source = (0..model.history_size)
        .map(|frame_index| full_lewm_image_features(sample, frame_index))
        .collect::<Result<Vec<_>, _>>()?;
    let target_index = model.history_size;
    let target = vec![full_lewm_image_features(sample, target_index)?];
    let packed_actions = vec![full_lewm_packed_action(
        sample,
        target_index,
        dataset.raw_action_dim,
        dataset.frameskip,
        model.action_encoder.input_dim,
    )?];

    Ok(PushtFullLewmExample {
        source,
        target,
        packed_actions,
    })
}

fn validate_full_lewm_sample(
    sample: &PushtSample,
    dataset: &crate::config::PushtDatasetConfig,
) -> Result<(), TrainerError> {
    let action_values = sample.actions.len();
    let expected_action_values = sample
        .action_shape
        .0
        .checked_mul(dataset.raw_action_dim)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform("PushT action shape overflow".to_owned()),
        })?;
    if sample.action_shape.1 != dataset.raw_action_dim || action_values != expected_action_values {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "PushT full LeWM expects raw action shape (T, {}), found {:?} with {action_values} values",
                dataset.raw_action_dim, sample.action_shape
            )),
        });
    }
    if sample.frame_shape.3 != 3 {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "PushT full LeWM expects RGB frames, found {} channels",
                sample.frame_shape.3
            )),
        });
    }
    Ok(())
}

fn full_lewm_image_features(
    sample: &PushtSample,
    frame_index: usize,
) -> Result<PushtFullLewmImageFeatures, TrainerError> {
    let frame_stride = sample
        .frame_shape
        .1
        .checked_mul(sample.frame_shape.2)
        .and_then(|value| value.checked_mul(sample.frame_shape.3))
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform("PushT full LeWM frame shape overflow".to_owned()),
        })?;
    let start = frame_index
        .checked_mul(frame_stride)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform("PushT full LeWM frame offset overflow".to_owned()),
        })?;
    let end = start
        .checked_add(frame_stride)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform("PushT full LeWM frame range overflow".to_owned()),
        })?;
    let values = sample
        .frames_t
        .get(start..end)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "PushT full LeWM frame {frame_index} exceeds frame shape {:?}",
                sample.frame_shape
            )),
        })?;
    let mut sum = 0.0;
    let mut energy = 0.0;
    let mut channel_sum = [0.0; 3];
    for pixel in values.chunks_exact(3) {
        for channel in 0..3 {
            let normalized = (f64::from(pixel[channel]) / 255.0 * 2.0) - 1.0;
            sum += normalized;
            energy += normalized * normalized;
            channel_sum[channel] += normalized;
        }
    }

    let total_values = usize_to_f64(values.len())?;
    let pixels_per_channel = usize_to_f64(values.len() / 3)?;
    let absolute_frame = u64::from(sample.meta.start_frame)
        .saturating_add(u64::try_from(frame_index).unwrap_or(u64::MAX));
    let absolute_frame_f64 = u32::try_from(absolute_frame).map_or(f64::from(u32::MAX), f64::from);
    Ok(PushtFullLewmImageFeatures {
        pixel_mean: sum / total_values,
        pixel_energy: energy / total_values,
        channel_mean: [
            channel_sum[0] / pixels_per_channel,
            channel_sum[1] / pixels_per_channel,
            channel_sum[2] / pixels_per_channel,
        ],
        time_fraction: absolute_frame_f64 / 1_000.0,
    })
}

fn validate_so100_lewm_sample(
    sample: &PushtSample,
    config: &So100DatasetConfig,
) -> Result<(), TrainerError> {
    if sample.action_shape.1 != config.action_dim {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "SO-100 full LeWM expects action dim {}, found {:?}",
                config.action_dim, sample.action_shape
            )),
        });
    }
    if sample.frame_shape.3 != 3 {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "SO-100 full LeWM expects RGB frames, found {} channels",
                sample.frame_shape.3
            )),
        });
    }
    Ok(())
}

fn full_lewm_so100_packed_action(
    sample: &PushtSample,
    target_index: usize,
    action_dim: usize,
) -> Result<Vec<f64>, TrainerError> {
    let start = target_index
        .checked_mul(action_dim)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform("SO-100 action offset overflow".to_owned()),
        })?;
    let end = start
        .checked_add(action_dim)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform("SO-100 action range overflow".to_owned()),
        })?;
    let action = sample
        .actions
        .get(start..end)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "SO-100 full LeWM action index {target_index} out of range {:?}",
                sample.action_shape
            )),
        })?;
    let mut packed = Vec::with_capacity(action.len());
    for value in action {
        if !value.is_finite() {
            return Err(TrainerError::Data {
                source: DataError::InvalidTransform(
                    "SO-100 action value must be finite".to_owned(),
                ),
            });
        }
        packed.push(f64::from(*value));
    }
    Ok(packed)
}

fn full_lewm_example_from_so100_sample(
    sample: &PushtSample,
    config: &So100DatasetConfig,
    model: &lewm_core::JepaConfig,
) -> Result<PushtFullLewmExample, TrainerError> {
    validate_so100_lewm_sample(sample, config)?;
    if sample.frame_shape.0 <= model.history_size {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "SO-100 full LeWM requires at least history_size + 1 frames, found {} frames for history_size {}",
                sample.frame_shape.0, model.history_size
            )),
        });
    }

    let source = (0..model.history_size)
        .map(|frame_index| full_lewm_image_features(sample, frame_index))
        .collect::<Result<Vec<_>, _>>()?;
    let target_index = model.history_size;
    let target = vec![full_lewm_image_features(sample, target_index)?];
    let packed_actions = vec![full_lewm_so100_packed_action(
        sample,
        target_index,
        config.action_dim,
    )?];

    Ok(PushtFullLewmExample {
        source,
        target,
        packed_actions,
    })
}

fn full_lewm_packed_action(
    sample: &PushtSample,
    target_index: usize,
    raw_action_dim: usize,
    frameskip: usize,
    packed_dim: usize,
) -> Result<Vec<f64>, TrainerError> {
    let expected_dim = raw_action_dim
        .checked_mul(frameskip)
        .ok_or_else(|| TrainerError::Data {
            source: DataError::InvalidTransform("PushT action packing overflow".to_owned()),
        })?;
    if packed_dim != expected_dim {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "PushT full LeWM packed action dim must be raw_action_dim * frameskip = {expected_dim}, found {packed_dim}",
            )),
        });
    }

    let mut packed = Vec::with_capacity(packed_dim);
    for skip in 0..frameskip {
        let action_index = target_index
            .saturating_add(skip)
            .min(sample.action_shape.0 - 1);
        let start = action_index
            .checked_mul(raw_action_dim)
            .ok_or_else(|| TrainerError::Data {
                source: DataError::InvalidTransform(
                    "PushT packed action offset overflow".to_owned(),
                ),
            })?;
        let end = start
            .checked_add(raw_action_dim)
            .ok_or_else(|| TrainerError::Data {
                source: DataError::InvalidTransform(
                    "PushT packed action range overflow".to_owned(),
                ),
            })?;
        let action = sample
            .actions
            .get(start..end)
            .ok_or_else(|| TrainerError::Data {
                source: DataError::InvalidTransform(format!(
                    "PushT packed action range {start}..{end} exceeds action shape {:?}",
                    sample.action_shape
                )),
            })?;
        for value in action {
            if !value.is_finite() {
                return Err(TrainerError::Data {
                    source: DataError::InvalidTransform(
                        "PushT action value must be finite".to_owned(),
                    ),
                });
            }
            packed.push(f64::from(*value));
        }
    }

    Ok(packed)
}

fn apply_pusht_full_lewm_adamw(
    model: &mut PushtFullLewmCore,
    gradients: &[f64],
    optimizer: &mut PushtFullLewmAdamWState,
    learning_rate: f64,
    config: &OptimConfig,
) {
    optimizer.step += 1;
    for (param_index, gradient) in gradients
        .iter()
        .copied()
        .enumerate()
        .take(model.parameter_count())
    {
        let spec = model.parameter_spec_for_flat_index(param_index);
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

fn write_pusht_burn_jepa_checkpoint(
    output_dir: &Path,
    config_hash: &str,
    seed: u64,
    scheduler_total_steps: u64,
    outcome: &PushtBurnJepaOutcome,
) -> Result<CheckpointPaths, TrainerError> {
    let valid_model = outcome.model.valid();
    let parameters = collect_parameters(&valid_model)?
        .into_iter()
        .map(parameter_tensor_from_export)
        .collect::<Result<Vec<_>, _>>()?;
    let burn_record = burn_jepa_record_bytes(&valid_model)?;
    let parity = ParityProbe {
        encoder_cls_l_inf: outcome.losses.last().map_or(0.0, |point| point.pred_loss),
        predictor_l_inf: last_train_loss(&outcome.losses),
        sigreg_value: outcome
            .losses
            .last()
            .map_or(0.0, |point| point.sigreg_proxy_loss),
    };
    let request = CheckpointWriteRequest {
        output_dir,
        run_id: PUSHT_FULL_BURN_JEPA_RUN_ID,
        step: outcome.step,
        epoch: 0,
        wall_time_s: 0.0,
        git_short_sha: option_env!("LEWM_GIT_SHA").unwrap_or("unknown"),
        config_hash,
        rng_state: checkpoint_rng_state(seed, outcome.step, &outcome.rng_streams),
        metrics_last_step: burn_jepa_checkpoint_metrics(outcome, scheduler_total_steps),
        burn_record: &burn_record,
        parameters: &parameters,
        parity: &parity,
    };

    save_checkpoint(&request).map_err(TrainerError::from)
}

fn burn_jepa_record_bytes(model: &Jepa<PushtBurnCpuBackend>) -> Result<Vec<u8>, TrainerError> {
    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
    recorder
        .record(model.clone().into_record(), ())
        .map_err(TrainerError::from)
}

fn parameter_tensor_from_export(tensor: ExportedTensor) -> Result<ParameterTensor, TrainerError> {
    match tensor.dtype {
        ExportDType::F32 => {
            let values = checked_f32_values(tensor.bytes())?;
            Ok(ParameterTensor::f32(tensor.name, tensor.shape, values)?)
        },
        ExportDType::I64 => {
            let values = i64_values(tensor.bytes())?;
            Ok(ParameterTensor::i64(tensor.name, tensor.shape, values)?)
        },
    }
}

fn burn_jepa_checkpoint_metrics(
    outcome: &PushtBurnJepaOutcome,
    scheduler_total_steps: u64,
) -> BTreeMap<String, f64> {
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
    metrics.insert(
        "train/scheduler_total_steps".to_owned(),
        smoke_step_as_f64(scheduler_total_steps).unwrap_or(0.0),
    );
    metrics
}

fn write_full_lewm_checkpoint(
    output_dir: &Path,
    config_hash: &str,
    seed: u64,
    scheduler_total_steps: u64,
    outcome: &PushtFullLewmOutcome,
    run_id: &'static str,
    record_kind: &str,
) -> Result<CheckpointPaths, TrainerError> {
    let parameters = outcome
        .model
        .parameter_specs()
        .iter()
        .map(|spec| full_lewm_parameter_tensor(&outcome.model, spec))
        .collect::<Result<Vec<_>, _>>()?;
    let parity = ParityProbe {
        encoder_cls_l_inf: outcome.losses.last().map_or(0.0, |point| point.pred_loss),
        predictor_l_inf: last_train_loss(&outcome.losses),
        sigreg_value: outcome
            .losses
            .last()
            .map_or(0.0, |point| point.sigreg_proxy_loss),
    };
    let record = PushtFullLewmRecord {
        schema_version: "1.1.0".to_owned(),
        kind: record_kind.to_owned(),
        step: outcome.step,
        params: outcome.model.flat_parameters().to_vec(),
        adamw_step: outcome.optimizer.step,
        adamw_params: outcome.optimizer.params.clone(),
        samples_seen: outcome.samples_seen,
        scheduler_total_steps,
        warmstart: outcome.warmstart.clone(),
    };
    let burn_record = serde_json::to_vec(&record)?;
    let request = CheckpointWriteRequest {
        output_dir,
        run_id,
        step: outcome.step,
        epoch: 0,
        wall_time_s: 0.0,
        git_short_sha: option_env!("LEWM_GIT_SHA").unwrap_or("unknown"),
        config_hash,
        rng_state: checkpoint_rng_state(seed, outcome.step, &outcome.rng_streams),
        metrics_last_step: full_train_checkpoint_metrics(outcome),
        burn_record: &burn_record,
        parameters: &parameters,
        parity: &parity,
    };

    save_checkpoint(&request).map_err(TrainerError::from)
}

fn write_pusht_full_lewm_checkpoint(
    output_dir: &Path,
    config_hash: &str,
    seed: u64,
    scheduler_total_steps: u64,
    outcome: &PushtFullLewmOutcome,
) -> Result<CheckpointPaths, TrainerError> {
    write_full_lewm_checkpoint(
        output_dir,
        config_hash,
        seed,
        scheduler_total_steps,
        outcome,
        PUSHT_BOUNDED_LEWM_RUN_ID,
        PUSHT_BOUNDED_LEWM_RECORD_KIND,
    )
}

fn write_so100_full_lewm_checkpoint(
    output_dir: &Path,
    config_hash: &str,
    seed: u64,
    scheduler_total_steps: u64,
    outcome: &PushtFullLewmOutcome,
) -> Result<CheckpointPaths, TrainerError> {
    write_full_lewm_checkpoint(
        output_dir,
        config_hash,
        seed,
        scheduler_total_steps,
        outcome,
        SO100_FULL_LEWM_RUN_ID,
        "lewm-rs-so100-full-module-lewm-record",
    )
}

fn full_lewm_parameter_tensor(
    model: &PushtFullLewmCore,
    spec: &crate::pusht_full::PushtFullLewmParameterSpec,
) -> Result<ParameterTensor, TrainerError> {
    let values = model
        .parameter_values(spec)
        .iter()
        .copied()
        .map(f32_from_f64)
        .collect();
    Ok(ParameterTensor::f32(
        spec.name.clone(),
        spec.shape.clone(),
        values,
    )?)
}

fn checkpoint_rng_state(
    seed: u64,
    step: u64,
    rng_streams: &RestoredRngStreams,
) -> CheckpointRngState {
    CheckpointRngState {
        global_seed: seed,
        step_at_save: step,
        data_shuffle: encode_rng(&rng_streams.data_shuffle),
        sigreg_sketch: encode_rng(&rng_streams.sigreg_sketch),
        dropout: encode_rng(&rng_streams.dropout),
        cem: encode_rng(&rng_streams.cem),
        model_init: encode_rng(&rng_streams.model_init),
    }
}

fn full_train_checkpoint_metrics(outcome: &PushtFullLewmOutcome) -> BTreeMap<String, f64> {
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

fn metric_u64(value: Option<&f64>) -> Option<u64> {
    let value = *value?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Some(value as u64)
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn checked_f32_values(bytes: &[u8]) -> Result<Vec<f32>, TrainerError> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "F32 tensor payload has {} bytes, not a multiple of 4",
                bytes.len()
            )),
        });
    }
    Ok(f32_values(bytes))
}

fn i64_values(bytes: &[u8]) -> Result<Vec<i64>, TrainerError> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<i64>()) {
        return Err(TrainerError::Data {
            source: DataError::InvalidTransform(format!(
                "I64 tensor payload has {} bytes, not a multiple of 8",
                bytes.len()
            )),
        });
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
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
    eval_every_n_epochs > 0 && epoch > 0 && epoch.is_multiple_of(eval_every_n_epochs)
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
    use std::collections::BTreeSet;
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

        let report = write_train_artifacts(TrainArtifactRequest {
            output_dir: dir.path(),
            root: &root,
            config_hash: "abc123",
            data_dir: None,
            max_steps: 10,
            seed: 7,
            device: "cpu",
            resume_if_present: false,
        })?;

        assert_eq!(report.kind, "lewm-rs-train-report");
        assert_eq!(report.mode, PUSHT_BOUNDED_LEWM_MODE);
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
        assert_eq!(loaded.sidecar.run_id, PUSHT_BOUNDED_LEWM_RUN_ID);
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
    fn pusht_full_burn_jepa_mode_writes_jepa_checkpoint() -> Result<(), Box<dyn std::error::Error>>
    {
        let dir = TestDir::new("pusht-burn-jepa")?;
        let root = compact_pusht_burn_jepa_root(dir.path().join("missing-pusht"));

        let report = write_train_artifacts(TrainArtifactRequest {
            output_dir: dir.path(),
            root: &root,
            config_hash: "abc123",
            data_dir: None,
            max_steps: 1,
            seed: 11,
            device: "cpu",
            resume_if_present: false,
        })?;

        assert_eq!(report.mode, PUSHT_FULL_BURN_JEPA_MODE);
        assert_eq!(report.steps_completed, 1);
        assert_eq!(report.batch_size, 1);
        assert!(report.checkpoint_complete);
        let loaded = crate::checkpoint::load_checkpoint(dir.path().join("step_0000001.json"))?;
        assert_eq!(loaded.sidecar.run_id, PUSHT_FULL_BURN_JEPA_RUN_ID);
        assert_eq!(loaded.sidecar.rng_state.global_seed, 11);
        assert!(loaded.burn_record.len() > 1024);
        let tensors = SafeTensors::deserialize(&loaded.safetensors_bytes)?;
        let actual_names = tensors.names().into_iter().collect::<BTreeSet<_>>();
        assert_eq!(actual_names.len(), 58);
        assert!(
            !tensors
                .names()
                .iter()
                .any(|name| name.starts_with("sigreg.consts."))
        );
        assert!(tensors.tensor("encoder.embeddings.cls_token").is_ok());
        assert!(tensors.tensor("predictor.pos_embed").is_ok());
        Ok(())
    }

    #[test]
    fn pusht_bounded_labels_accept_legacy_resume_contract() {
        assert!(is_supported_pusht_bounded_run_id(PUSHT_BOUNDED_LEWM_RUN_ID));
        assert!(is_supported_pusht_bounded_run_id(
            PUSHT_LEGACY_FULL_LEWM_RUN_ID
        ));
        assert!(is_supported_pusht_bounded_record_kind(
            PUSHT_BOUNDED_LEWM_RECORD_KIND
        ));
        assert!(is_supported_pusht_bounded_record_kind(
            PUSHT_LEGACY_FULL_LEWM_RECORD_KIND
        ));
        assert!(!is_supported_pusht_bounded_run_id(
            "pusht-full-burn-jepa-v1"
        ));
        assert!(!is_supported_pusht_bounded_record_kind(
            "lewm-rs-pusht-full-burn-jepa-record"
        ));
    }

    #[test]
    fn so100_train_missing_hdf5_returns_missing_data_path() -> Result<(), Box<dyn std::error::Error>>
    {
        let dir = TestDir::new("so100-train-missing-data")?;
        let missing_data = dir.path().join("missing-so100.h5");
        let root = crate::config::RootConfig {
            dataset: crate::config::DatasetConfig::So100(crate::config::So100DatasetConfig {
                hdf5_path: missing_data.clone(),
                ..crate::config::So100DatasetConfig::default()
            }),
            ..crate::config::RootConfig::default()
        };

        let err = write_train_artifacts(TrainArtifactRequest {
            output_dir: dir.path(),
            root: &root,
            config_hash: "abc123",
            data_dir: None,
            max_steps: 10,
            seed: 7,
            device: "cpu",
            resume_if_present: false,
        })
        .expect_err("SO-100 train without HDF5 data must fail");

        match err {
            TrainerError::MissingTrainDataPath { path } => {
                assert_eq!(path, missing_data);
            },
            other => return Err(format!("unexpected error: {other}").into()),
        }
        Ok(())
    }

    fn compact_pusht_burn_jepa_root(root_path: PathBuf) -> crate::config::RootConfig {
        use lewm_core::{
            EmbedderConfig, GeluVariant, JepaConfig, MlpConfig, NormVariant, PredictorConfig,
            VitConfig, VitSize,
        };

        let mut root = crate::config::RootConfig::default();
        root.experimental.pusht_train_mode = crate::config::PushtTrainMode::FullBurnJepa;
        root.model = JepaConfig {
            encoder: VitConfig {
                size: VitSize::Tiny,
                image_size: 16,
                patch_size: 8,
                num_channels: 3,
                hidden_size: 8,
                num_hidden_layers: 1,
                num_attention_heads: 2,
                intermediate_size: 16,
                hidden_act: GeluVariant::TanhApprox,
                attention_probs_dropout_prob: 0.0,
                hidden_dropout_prob: 0.0,
                layer_norm_eps: 1.0e-12,
                use_cls_token: true,
                interpolate_pos_encoding: false,
                use_mask_token: false,
                pretrained: false,
            },
            action_encoder: EmbedderConfig {
                input_dim: 2,
                smoothed_dim: 2,
                emb_dim: 8,
                mlp_scale: 2,
            },
            predictor: PredictorConfig {
                num_frames: 3,
                depth: 1,
                heads: 2,
                mlp_dim: 16,
                dim_head: 4,
                input_dim: 8,
                hidden_dim: 8,
                output_dim: 8,
                action_emb_dim: 8,
                dropout: 0.0,
                emb_dropout: 0.0,
            },
            projector: MlpConfig {
                input_dim: 8,
                hidden_dim: 16,
                output_dim: 8,
                norm: NormVariant::None,
            },
            pred_proj: MlpConfig {
                input_dim: 8,
                hidden_dim: 16,
                output_dim: 8,
                norm: NormVariant::None,
            },
            history_size: 2,
            horizon: 4,
        };
        root.training.history_size = 2;
        root.training.horizon = 4;
        root.training.batch_size = 1;
        root.training.grad_accum_steps = 1;
        root.training.lr_peak = 1.0e-4;
        root.training.lr_min = 1.0e-4;
        root.training.warmup_steps = 0;
        root.loss.lambda_sigreg = 0.0;
        if let crate::config::DatasetConfig::Pusht(config) = &mut root.dataset {
            config.root_path = root_path;
            config.horizon = 4;
            config.history_size = 2;
            config.raw_action_dim = 2;
            config.frameskip = 1;
        }
        root
    }

    #[test]
    fn so100_warmstart_transfers_shared_modules_and_preserves_action_encoder()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = TestDir::new("so100-warmstart-start")?;
        let source_path = dir.path().join("pusht_source.mpk");
        let mut root = so100_root_with_data_path(dir.path().join("missing-so100.h5"));
        root.training.warmstart_from = Some(source_path.clone());

        let mut source_config = root.model.clone();
        source_config.action_encoder.input_dim = root.model.action_encoder.smoothed_dim;
        let mut source_model = PushtFullLewmCore::new(&source_config, 7)?;
        for index in 0..source_model.parameter_count() {
            source_model.set_parameter(index, f64::from(u32::try_from(index)?) + 100.0);
        }
        write_pusht_warmstart_source(&source_path, &source_model)?;

        let fresh_target = PushtFullLewmCore::new(&root.model, 7)?;
        let start = fresh_so100_full_lewm_start(&root, 7)?;
        let warmstart = start
            .warmstart
            .as_ref()
            .ok_or("expected warm-start provenance")?;

        assert_eq!(start.start_step, 0);
        assert_eq!(start.optimizer.step, 0);
        assert_eq!(warmstart.source_path, source_path.display().to_string());
        assert!(
            warmstart
                .transferred_parameters
                .iter()
                .any(|name| name.starts_with("encoder."))
        );
        assert!(
            warmstart
                .preserved_action_encoder_parameters
                .iter()
                .all(|name| name.starts_with("action_encoder."))
        );

        assert_shared_params_match_source(&start.model, &source_model)?;
        assert_action_encoder_matches_fresh(&start.model, &fresh_target)?;
        Ok(())
    }

    #[test]
    fn pusht_train_resume_restores_checkpoint_and_continues()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = TestDir::new("pusht-train-resume")?;
        let missing_data = dir.path().join("missing-pusht");
        let mut root = crate::config::RootConfig::default();
        let crate::config::DatasetConfig::Pusht(config) = &mut root.dataset else {
            return Err("expected PushT default dataset".into());
        };
        config.root_path = missing_data;

        let first = write_train_artifacts(TrainArtifactRequest {
            output_dir: dir.path(),
            root: &root,
            config_hash: "abc123",
            data_dir: None,
            max_steps: 3,
            seed: 7,
            device: "cpu",
            resume_if_present: false,
        })?;
        let resumed = write_train_artifacts(TrainArtifactRequest {
            output_dir: dir.path(),
            root: &root,
            config_hash: "abc123",
            data_dir: None,
            max_steps: 6,
            seed: 7,
            device: "cpu",
            resume_if_present: true,
        })?;

        assert_eq!(first.checkpoint_step, 3);
        assert_eq!(resumed.checkpoint_step, 6);
        assert_eq!(resumed.losses.first().map(|point| point.step), Some(1));
        assert_eq!(resumed.losses.last().map(|point| point.step), Some(6));
        assert!(dir.path().join("step_0000006.json").is_file());
        let loaded = crate::checkpoint::load_checkpoint(dir.path().join("step_0000006.json"))?;
        let record: PushtFullLewmRecord = serde_json::from_slice(&loaded.burn_record)?;
        assert_eq!(record.adamw_step, 6);
        assert_eq!(record.adamw_params.len(), record.params.len());
        Ok(())
    }

    fn so100_root_with_data_path(path: PathBuf) -> crate::config::RootConfig {
        crate::config::RootConfig {
            dataset: crate::config::DatasetConfig::So100(crate::config::So100DatasetConfig {
                hdf5_path: path,
                ..crate::config::So100DatasetConfig::default()
            }),
            model: lewm_core::JepaConfig {
                action_encoder: lewm_core::EmbedderConfig {
                    input_dim: crate::config::SO100_ACTION_DIM,
                    ..lewm_core::EmbedderConfig::default()
                },
                ..lewm_core::JepaConfig::default()
            },
            ..crate::config::RootConfig::default()
        }
    }

    fn write_pusht_warmstart_source(
        path: &Path,
        model: &PushtFullLewmCore,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let record = PushtFullLewmRecord {
            schema_version: "1.1.0".to_owned(),
            kind: PUSHT_BOUNDED_LEWM_RECORD_KIND.to_owned(),
            step: 42,
            params: model.flat_parameters().to_vec(),
            adamw_step: 42,
            adamw_params: vec![ScalarAdamWParamState::default(); model.parameter_count()],
            samples_seen: 2688,
            scheduler_total_steps: 42,
            warmstart: None,
        };
        fs::write(path, serde_json::to_vec(&record)?)?;
        Ok(())
    }

    fn assert_shared_params_match_source(
        warm_model: &PushtFullLewmCore,
        source_model: &PushtFullLewmCore,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for warm_spec in warm_model.parameter_specs().iter().filter(|spec| {
            ["encoder.", "predictor.", "projector.", "pred_proj."]
                .iter()
                .any(|prefix| spec.name.starts_with(prefix))
        }) {
            let source_spec = source_model
                .parameter_specs()
                .iter()
                .find(|source_spec| source_spec.name == warm_spec.name)
                .ok_or_else(|| format!("missing source spec {}", warm_spec.name))?;
            let warm_values = warm_model.parameter_values(warm_spec);
            let source_values = source_model.parameter_values(source_spec);
            assert_eq!(warm_values.len(), source_values.len());
            for (warm, source) in warm_values.iter().zip(source_values) {
                assert_eq!(
                    f32_from_f64(*warm).to_bits(),
                    f32_from_f64(*source).to_bits()
                );
            }
        }
        Ok(())
    }

    fn assert_action_encoder_matches_fresh(
        warm_model: &PushtFullLewmCore,
        fresh_model: &PushtFullLewmCore,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for warm_spec in warm_model
            .parameter_specs()
            .iter()
            .filter(|spec| spec.name.starts_with("action_encoder."))
        {
            let fresh_spec = fresh_model
                .parameter_specs()
                .iter()
                .find(|fresh_spec| fresh_spec.name == warm_spec.name)
                .ok_or_else(|| format!("missing fresh spec {}", warm_spec.name))?;
            let warm_values = warm_model.parameter_values(warm_spec);
            let fresh_values = fresh_model.parameter_values(fresh_spec);
            assert_eq!(warm_values.len(), fresh_values.len());
            for (warm, fresh) in warm_values.iter().zip(fresh_values) {
                assert_eq!(
                    f32_from_f64(*warm).to_bits(),
                    f32_from_f64(*fresh).to_bits()
                );
            }
        }
        Ok(())
    }

    #[test]
    fn pusht_train_resume_rejects_config_mismatch() -> Result<(), Box<dyn std::error::Error>> {
        let dir = TestDir::new("pusht-train-resume-config-mismatch")?;
        let missing_data = dir.path().join("missing-pusht");
        let mut root = crate::config::RootConfig::default();
        let crate::config::DatasetConfig::Pusht(config) = &mut root.dataset else {
            return Err("expected PushT default dataset".into());
        };
        config.root_path = missing_data;

        let _first = write_train_artifacts(TrainArtifactRequest {
            output_dir: dir.path(),
            root: &root,
            config_hash: "abc123",
            data_dir: None,
            max_steps: 3,
            seed: 7,
            device: "cpu",
            resume_if_present: false,
        })?;
        let err = write_train_artifacts(TrainArtifactRequest {
            output_dir: dir.path(),
            root: &root,
            config_hash: "def456",
            data_dir: None,
            max_steps: 6,
            seed: 7,
            device: "cpu",
            resume_if_present: true,
        })
        .expect_err("resume with a different config hash must fail");

        assert!(err.to_string().contains("resume config hash mismatch"));
        Ok(())
    }

    #[test]
    fn pusht_train_resume_rejects_corrupt_safetensors() -> Result<(), Box<dyn std::error::Error>> {
        let dir = TestDir::new("pusht-train-resume-corrupt-safetensors")?;
        let missing_data = dir.path().join("missing-pusht");
        let mut root = crate::config::RootConfig::default();
        let crate::config::DatasetConfig::Pusht(config) = &mut root.dataset else {
            return Err("expected PushT default dataset".into());
        };
        config.root_path = missing_data;

        let _first = write_train_artifacts(TrainArtifactRequest {
            output_dir: dir.path(),
            root: &root,
            config_hash: "abc123",
            data_dir: None,
            max_steps: 3,
            seed: 7,
            device: "cpu",
            resume_if_present: false,
        })?;
        fs::write(
            dir.path().join("step_0000003.safetensors"),
            b"not safetensors",
        )?;

        let err = write_train_artifacts(TrainArtifactRequest {
            output_dir: dir.path(),
            root: &root,
            config_hash: "abc123",
            data_dir: None,
            max_steps: 6,
            seed: 7,
            device: "cpu",
            resume_if_present: true,
        })
        .expect_err("resume with a corrupt safetensors mirror must fail");

        assert!(err.to_string().contains("safetensors"));
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
