//! Root training configuration loader, validation, overrides, and hashing.

use std::{
    env,
    error::Error,
    fmt, fs,
    path::{Path, PathBuf},
};

use lewm_core::config::JepaConfig;
use serde::{Deserialize, Serialize};
use validator::Validate;

/// Compiled-in root config schema version accepted by this crate.
pub const SUPPORTED_SCHEMA_VERSION: &str = "1.0.0";

/// SO-100 action dimension from RFC 0012.
pub const SO100_ACTION_DIM: usize = crate::warmstart::SO100_ACTION_DIM;

/// SO-100 warmup length from RFC 0012 section 10.
pub const SO100_WARMUP_STEPS: u32 = 500;

/// Pinned SO-100 held-out episode IDs.
pub const SO100_HELDOUT_EPISODES: [u32; 5] = [5, 14, 23, 31, 42];

/// Default `PushT` checkpoint used by the SO-100 warm-start arm.
pub const DEFAULT_SO100_WARMSTART_FROM: &str = "/checkpoints/lewm-rs-pusht/step_0014400.mpk";

/// Fully loaded root config plus its canonical reproducibility hash.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedRootConfig {
    /// Validated root config after file, environment, and CLI overrides.
    pub root: RootConfig,
    /// BLAKE3 hash of the canonical TOML representation, truncated to 12 hex chars.
    pub config_hash: String,
    /// Secret values sourced from environment variables and excluded from hashing.
    pub secrets: ConfigSecrets,
    /// Device override sourced from the environment and handled by the CLI/runtime layer.
    pub device: Option<String>,
}

/// Secret runtime inputs intentionally excluded from the serializable root config.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConfigSecrets {
    /// Optional Hugging Face token from `LEWM_HF_TOKEN`.
    pub hf_token: Option<String>,
}

/// Environment-derived overrides that are allowed to affect config loading.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EnvOverrides {
    /// Optional `LEWM_SEED` override for `training.seed`.
    pub seed: Option<u64>,
    /// Optional `LEWM_LR_PEAK` override for `training.lr_peak`.
    pub lr_peak: Option<f64>,
    /// Optional `LEWM_HF_TOKEN` secret value, intentionally excluded from `RootConfig`.
    pub hf_token: Option<String>,
    /// Optional `LEWM_OTEL_ENDPOINT` override for `observability.otel_endpoint`.
    pub otel_endpoint: Option<String>,
    /// Optional `LEWM_DEVICE` CLI-equivalent value, intentionally excluded from `RootConfig`.
    pub device: Option<String>,
}

impl EnvOverrides {
    /// Read the RFC 0018 allowlisted environment overrides from the current process.
    ///
    /// # Errors
    ///
    /// Returns an error when a present numeric environment variable cannot be
    /// parsed as the target config field type.
    pub fn from_process_env() -> Result<Self, ConfigError> {
        Ok(Self {
            seed: parse_optional_env("LEWM_SEED")?,
            lr_peak: parse_optional_env("LEWM_LR_PEAK")?,
            hf_token: optional_env_string("LEWM_HF_TOKEN"),
            otel_endpoint: optional_env_string("LEWM_OTEL_ENDPOINT"),
            device: optional_env_string("LEWM_DEVICE"),
        })
    }
}

/// Top-level training config composed from crate-specific sections.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RootConfig {
    /// Mandatory schema version, compared with [`SUPPORTED_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Dataset section.
    pub dataset: DatasetConfig,
    /// Model section reused from `lewm-core`.
    pub model: JepaConfig,
    /// Loss section.
    pub loss: LossConfig,
    /// Training loop section.
    pub training: TrainingConfig,
    /// Evaluation section.
    pub eval: EvalConfig,
    /// Observability section.
    pub observability: ObservabilityConfig,
    /// Hub publication section.
    pub hub: HubConfig,
    /// Inference/export section reserved for shared root configs.
    pub infer: InferConfig,
    /// Reserved section for explicit experimental overlays.
    pub experimental: ExperimentalConfig,
}

impl Default for RootConfig {
    fn default() -> Self {
        Self {
            schema_version: SUPPORTED_SCHEMA_VERSION.to_string(),
            dataset: DatasetConfig::default(),
            model: JepaConfig::default(),
            loss: LossConfig::default(),
            training: TrainingConfig::default(),
            eval: EvalConfig::default(),
            observability: ObservabilityConfig::default(),
            hub: HubConfig::default(),
            infer: InferConfig::default(),
            experimental: ExperimentalConfig::default(),
        }
    }
}

impl RootConfig {
    /// Validate the schema version, nested validator ranges, and cross-field invariants.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] with precise messages for schema, range, and
    /// cross-field validation failures.
    pub fn validate_all(&self) -> Result<(), ConfigError> {
        self.assert_schema_version()?;

        let mut errors = Vec::new();
        push_validation_errors(&mut errors, "model", self.model.validate());
        push_validation_errors(&mut errors, "dataset", self.dataset.validate_ranges());
        push_validation_errors(&mut errors, "loss", self.loss.validate());
        push_validation_errors(&mut errors, "training", self.training.validate());
        push_validation_errors(&mut errors, "eval", self.eval.validate_ranges());
        push_validation_errors(&mut errors, "observability", self.observability.validate());
        push_validation_errors(&mut errors, "hub", self.hub.validate());
        push_validation_errors(&mut errors, "infer", self.infer.validate());

        errors.extend(self.model.shape_errors());
        errors.extend(self.cross_field_errors());

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Validation { messages: errors })
        }
    }

    /// Return validation warnings that should be surfaced without rejecting the config.
    #[must_use]
    pub fn validation_warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.loss.sigreg_num_proj % 32 != 0 {
            warnings.push(
                "loss.sigreg_num_proj is not divisible by 32; this may reduce kernel efficiency"
                    .to_string(),
            );
        }
        warnings
    }

    fn assert_schema_version(&self) -> Result<(), ConfigError> {
        if self.schema_version == SUPPORTED_SCHEMA_VERSION {
            return Ok(());
        }

        Err(ConfigError::SchemaVersion {
            expected: SUPPORTED_SCHEMA_VERSION.to_string(),
            found: self.schema_version.clone(),
        })
    }

    fn cross_field_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let dataset_horizon = self.dataset.horizon();
        let dataset_history_size = self.dataset.history_size();
        let effective_batch = self.training.effective_batch();

        if dataset_horizon != self.training.horizon {
            errors.push("dataset.horizon must equal training.horizon".to_string());
        }

        if self.model.horizon != self.training.horizon {
            errors.push("model.horizon must equal training.horizon".to_string());
        }

        if dataset_history_size != self.training.history_size {
            errors.push("dataset.history_size must equal training.history_size".to_string());
        }

        if self.model.history_size != self.training.history_size {
            errors.push("model.history_size must equal training.history_size".to_string());
        }

        match effective_batch {
            Some(value) if !(32..=512).contains(&value) => errors.push(
                "training.batch_size * training.grad_accum_steps must be between 32 and 512"
                    .to_string(),
            ),
            None => errors.push(
                "training.batch_size * training.grad_accum_steps overflowed usize".to_string(),
            ),
            Some(_) => {},
        }

        if self.training.lr_min > self.training.lr_peak {
            errors
                .push("training.lr_min must be less than or equal to training.lr_peak".to_string());
        }

        if !(0.0..1.0).contains(&self.training.betas.0)
            || !(0.0..1.0).contains(&self.training.betas.1)
            || self.training.betas.0 >= self.training.betas.1
        {
            errors.push("training.betas must satisfy 0.0 <= beta1 < beta2 < 1.0".to_string());
        }

        let Some(available_eval_horizon) = self
            .model
            .predictor
            .num_frames
            .checked_sub(self.training.history_size)
        else {
            errors.push(
                "training.history_size must not exceed model.predictor.num_frames".to_string(),
            );
            return errors;
        };

        if self.eval.horizon_plan() > available_eval_horizon {
            errors.push(
                "eval.horizon_plan must be less than or equal to model.predictor.num_frames - training.history_size"
                    .to_string(),
            );
        }

        if let DatasetConfig::So100(dataset) = &self.dataset {
            errors.extend(self.so100_contract_errors(dataset));
        }

        errors
    }

    fn so100_contract_errors(&self, dataset: &So100DatasetConfig) -> Vec<String> {
        let mut errors = Vec::new();

        if dataset.action_dim != SO100_ACTION_DIM {
            errors.push(format!(
                "SO-100 requires dataset.action_dim {SO100_ACTION_DIM}, found {}",
                dataset.action_dim
            ));
        }

        if self.model.action_encoder.input_dim != SO100_ACTION_DIM {
            errors.push(format!(
                "SO-100 requires model.action_encoder.input_dim {SO100_ACTION_DIM}, found {}",
                self.model.action_encoder.input_dim
            ));
        }

        if self.training.warmup_steps != SO100_WARMUP_STEPS {
            errors.push(format!(
                "SO-100 requires training.warmup_steps {SO100_WARMUP_STEPS}, found {}",
                self.training.warmup_steps
            ));
        }

        match &self.eval {
            EvalConfig::So100Latent(eval) => {
                if eval.episode_ids != SO100_HELDOUT_EPISODES {
                    errors.push(format!(
                        "SO-100 eval.episode_ids must be {:?}, found {:?}",
                        SO100_HELDOUT_EPISODES, eval.episode_ids
                    ));
                }
            },
            EvalConfig::PushtSimulated(_) => {
                errors.push("SO-100 dataset requires so100_latent_rollout eval".to_string());
            },
        }

        errors
    }
}

/// Dataset configuration selected by the `kind` tag.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DatasetConfig {
    /// `PushT` HDF5 dataset configuration.
    Pusht(PushtDatasetConfig),
    /// `SO-100` dataset configuration.
    So100(So100DatasetConfig),
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self::Pusht(PushtDatasetConfig::default())
    }
}

impl DatasetConfig {
    fn horizon(&self) -> usize {
        match self {
            Self::Pusht(config) => config.horizon,
            Self::So100(config) => config.horizon,
        }
    }

    fn history_size(&self) -> usize {
        match self {
            Self::Pusht(config) => config.history_size,
            Self::So100(config) => config.history_size,
        }
    }

    fn validate_ranges(&self) -> Result<(), validator::ValidationErrors> {
        match self {
            Self::Pusht(config) => config.validate(),
            Self::So100(config) => config.validate(),
        }
    }
}

/// Dataset split selected by training or evaluation configs.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSplit {
    /// Training split.
    #[default]
    Train,
    /// Evaluation split.
    Eval,
}

/// `PushT` config represented in root TOML.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct PushtDatasetConfig {
    /// Path to the `PushT` HDF5 root or shard.
    pub root_path: PathBuf,
    /// Dataset split.
    pub split: DatasetSplit,
    /// Sample horizon.
    #[validate(range(min = 2, max = 64))]
    pub horizon: usize,
    /// Historical context size.
    #[validate(range(min = 1, max = 100))]
    pub history_size: usize,
    /// Deterministic data seed.
    pub seed: u64,
}

impl Default for PushtDatasetConfig {
    fn default() -> Self {
        Self {
            root_path: PathBuf::from("/data/lewm-pusht"),
            split: DatasetSplit::Train,
            horizon: 8,
            history_size: 3,
            seed: 0,
        }
    }
}

/// `SO-100` config represented in root TOML.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct So100DatasetConfig {
    /// Path to the `SO-100` dataset root.
    pub root_path: PathBuf,
    /// Path to the decoded RFC 0012 HDF5 mirror.
    pub hdf5_path: PathBuf,
    /// Camera view selected for v1 training.
    pub camera_view: CameraView,
    /// Path to persisted SO-100 action statistics.
    pub stats_path: PathBuf,
    /// Dataset split.
    pub split: DatasetSplit,
    /// Sample horizon.
    #[validate(range(min = 2, max = 64))]
    pub horizon: usize,
    /// Historical context size.
    #[validate(range(min = 1, max = 100))]
    pub history_size: usize,
    /// Per-step action dimensionality.
    #[validate(range(min = 1, max = 64))]
    pub action_dim: usize,
    /// Deterministic data seed.
    pub seed: u64,
}

impl Default for So100DatasetConfig {
    fn default() -> Self {
        Self {
            root_path: PathBuf::from("/data/lewm-so100"),
            hdf5_path: PathBuf::from("/data/so100/svla_so100_pickplace.h5"),
            camera_view: CameraView::Top,
            stats_path: PathBuf::from("/data/so100/stats.safetensors"),
            split: DatasetSplit::Train,
            horizon: 8,
            history_size: 3,
            action_dim: SO100_ACTION_DIM,
            seed: 0,
        }
    }
}

/// SO-100 camera view selection.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CameraView {
    /// Top camera.
    #[default]
    Top,
    /// Wrist camera.
    Wrist,
}

/// Training loss configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct LossConfig {
    /// `SIGReg` loss weight.
    #[validate(range(min = 0.0, max = 100.0))]
    pub lambda_sigreg: f64,
    /// Number of `SIGReg` spline knots.
    #[validate(range(min = 8, max = 64))]
    pub sigreg_knots: usize,
    /// Number of random `SIGReg` projections.
    #[validate(range(min = 64, max = 8192))]
    pub sigreg_num_proj: usize,
    /// Maximum `SIGReg` spline parameter.
    #[validate(range(min = 1.0, max = 10.0))]
    pub sigreg_t_max: f64,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            lambda_sigreg: 1.0,
            sigreg_knots: 17,
            sigreg_num_proj: 1024,
            sigreg_t_max: 3.0,
        }
    }
}

/// Optimizer family.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    /// `AdamW` optimizer.
    #[default]
    Adamw,
    /// Lion optimizer.
    Lion,
}

/// Mixed precision policy.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionKind {
    /// Full `f32` training.
    F32,
    /// `bf16` outer operations with `f32` numerically sensitive paths.
    #[default]
    Bf16Mixed,
}

/// Training loop configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingConfig {
    /// Historical context size.
    #[validate(range(min = 1, max = 100))]
    pub history_size: usize,
    /// Prediction horizon.
    #[validate(range(min = 2, max = 64))]
    pub horizon: usize,
    /// Physical batch size.
    #[validate(range(min = 1, max = 1024))]
    pub batch_size: usize,
    /// Gradient accumulation steps.
    #[validate(range(min = 1, max = 32))]
    pub grad_accum_steps: usize,
    /// Optimizer kind.
    pub optimizer: OptimizerKind,
    /// Peak learning rate.
    #[validate(range(min = 1.0e-7, max = 1.0e-2))]
    pub lr_peak: f64,
    /// Minimum learning rate.
    #[validate(range(min = 1.0e-9, max = 1.0e-3))]
    pub lr_min: f64,
    /// Warmup step count.
    #[validate(range(min = 0, max = 100_000))]
    pub warmup_steps: u32,
    /// `AdamW` weight decay.
    #[validate(range(min = 0.0, max = 1.0))]
    pub weight_decay: f64,
    /// `AdamW` beta coefficients.
    pub betas: (f64, f64),
    /// Epoch count.
    #[validate(range(min = 1, max = 1000))]
    pub epochs: u32,
    /// Precision policy.
    pub precision: PrecisionKind,
    /// Global seed.
    pub seed: u64,
    /// Global gradient clipping norm.
    #[validate(range(min = 0.1, max = 100.0))]
    pub grad_clip_norm: f64,
    /// Optional warm-start checkpoint path.
    pub warmstart_from: Option<PathBuf>,
    /// Evaluation cadence in epochs.
    #[validate(range(min = 1, max = 100))]
    pub eval_every_n_epochs: u32,
    /// Probe cadence in steps.
    #[validate(range(min = 10, max = 10_000))]
    pub probe_every_n_steps: u32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            history_size: 3,
            horizon: 8,
            batch_size: 64,
            grad_accum_steps: 2,
            optimizer: OptimizerKind::Adamw,
            lr_peak: 3.0e-4,
            lr_min: 1.0e-5,
            warmup_steps: 1000,
            weight_decay: 0.05,
            betas: (0.9, 0.95),
            epochs: 10,
            precision: PrecisionKind::Bf16Mixed,
            seed: 0,
            grad_clip_norm: 1.0,
            warmstart_from: None,
            eval_every_n_epochs: 5,
            probe_every_n_steps: 100,
        }
    }
}

impl TrainingConfig {
    fn effective_batch(&self) -> Option<usize> {
        self.batch_size.checked_mul(self.grad_accum_steps)
    }
}

/// Evaluation configuration selected by the `kind` tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvalConfig {
    /// Simulated `PushT` planning evaluation.
    PushtSimulated(PushtEvalConfig),
    /// Latent `SO-100` trajectory evaluation.
    #[serde(alias = "so100_latent_rollout")]
    So100Latent(So100EvalConfig),
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self::PushtSimulated(PushtEvalConfig::default())
    }
}

impl EvalConfig {
    fn horizon_plan(&self) -> usize {
        match self {
            Self::PushtSimulated(config) => config.horizon_plan,
            Self::So100Latent(config) => config.horizon_plan,
        }
    }

    fn validate_ranges(&self) -> Result<(), validator::ValidationErrors> {
        match self {
            Self::PushtSimulated(config) => config.validate(),
            Self::So100Latent(config) => config.validate(),
        }
    }
}

/// `PushT` simulated planning evaluation config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct PushtEvalConfig {
    /// Episode ids to evaluate.
    #[validate(length(min = 1, max = 1000))]
    pub episode_ids: Vec<u32>,
    /// Maximum environment steps per episode.
    #[validate(range(min = 1, max = 1000))]
    pub max_steps_per_episode: usize,
    /// Cross-entropy method iteration count.
    #[validate(range(min = 1, max = 100))]
    pub n_iter: usize,
    /// Candidate count per iteration.
    #[validate(range(min = 1, max = 100_000))]
    pub n_cand: usize,
    /// Elite count per iteration.
    #[validate(range(min = 1, max = 100_000))]
    pub n_elite: usize,
    /// Planning horizon.
    #[validate(range(min = 1, max = 64))]
    pub horizon_plan: usize,
    /// Initial CEM sigma.
    #[validate(range(min = 1.0e-6, max = 100.0))]
    pub sigma_init: f64,
    /// Minimum CEM sigma.
    #[validate(range(min = 0.0, max = 100.0))]
    pub sigma_min: f64,
}

impl Default for PushtEvalConfig {
    fn default() -> Self {
        Self {
            episode_ids: vec![
                0, 7, 13, 21, 28, 34, 41, 48, 55, 62, 69, 76, 83, 90, 97, 104, 111, 118, 125, 132,
                139, 146, 153, 160, 167, 174, 181, 188, 195, 202, 209, 216, 223, 230, 237, 244,
                251, 258, 265, 272, 279, 286, 293, 300, 307, 314, 321, 328, 335, 342,
            ],
            max_steps_per_episode: 100,
            n_iter: 5,
            n_cand: 1000,
            n_elite: 100,
            horizon_plan: 5,
            sigma_init: 1.0,
            sigma_min: 0.05,
        }
    }
}

/// `SO-100` latent trajectory evaluation config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct So100EvalConfig {
    /// Held-out episode IDs.
    #[validate(length(min = 1, max = 1000))]
    pub episode_ids: Vec<u32>,
    /// Number of episodes to evaluate.
    #[validate(range(min = 1, max = 1000))]
    pub episodes: usize,
    /// Planning horizon.
    #[validate(range(min = 1, max = 64))]
    pub horizon_plan: usize,
}

impl Default for So100EvalConfig {
    fn default() -> Self {
        Self {
            episode_ids: SO100_HELDOUT_EPISODES.to_vec(),
            episodes: 5,
            horizon_plan: 5,
        }
    }
}

/// Observability configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct ObservabilityConfig {
    /// Trackio run-name prefix.
    #[validate(length(min = 1, max = 128))]
    pub trackio_run_name_prefix: String,
    /// Environment variable name for the OTLP endpoint.
    #[validate(length(min = 1, max = 128))]
    pub otel_endpoint_env: String,
    /// Resolved OTLP endpoint override.
    pub otel_endpoint: Option<String>,
    /// `TensorBoard` output directory.
    #[validate(length(min = 1, max = 256))]
    pub tensorboard_dir: String,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            trackio_run_name_prefix: "lewm-rs-pusht".to_string(),
            otel_endpoint_env: "OTEL_EXPORTER_OTLP_ENDPOINT".to_string(),
            otel_endpoint: None,
            tensorboard_dir: "tb".to_string(),
        }
    }
}

/// Hugging Face Hub publication config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct HubConfig {
    /// Target model repository.
    #[validate(length(min = 1, max = 256))]
    pub model_repo: String,
    /// Whether to upload at the end of a run.
    pub upload_at_end: bool,
    /// Mid-run upload cadence in epochs, or zero for end-only upload.
    #[validate(range(min = 0, max = 1000))]
    pub upload_every_n_epochs: u32,
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            model_repo: "abdelstark/lewm-rs-pusht".to_string(),
            upload_at_end: true,
            upload_every_n_epochs: 0,
        }
    }
}

/// Inference/export config reserved for shared roots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate, Default)]
#[serde(default, deny_unknown_fields)]
pub struct InferConfig {
    /// Whether inference/export is enabled for this root config.
    pub enabled: bool,
    /// Optional exported model path.
    pub model_path: Option<PathBuf>,
}

/// Reserved config section for explicit experimental overlays.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate, Default)]
#[serde(default, deny_unknown_fields)]
pub struct ExperimentalConfig {}

/// Error type for loading, merging, validating, and hashing root configs.
#[derive(Debug)]
pub enum ConfigError {
    /// Config file read failed.
    Io {
        /// Path that was being read.
        path: PathBuf,
        /// Source IO error.
        source: std::io::Error,
    },
    /// TOML parsing or `toml::Value` deserialization failed.
    TomlDeserialize {
        /// Source TOML deserialize error.
        source: toml::de::Error,
    },
    /// TOML serialization failed while building canonical form.
    TomlSerialize {
        /// Source TOML serialize error.
        source: toml::ser::Error,
    },
    /// The input file did not declare `schema_version`.
    MissingSchemaVersion,
    /// The input schema version does not match the compiled-in version.
    SchemaVersion {
        /// Expected schema version.
        expected: String,
        /// Found schema version.
        found: String,
    },
    /// Environment variable parsing failed.
    EnvVar {
        /// Environment variable name.
        name: String,
        /// Raw environment variable value.
        value: String,
        /// Parse failure message.
        message: String,
    },
    /// Override path or value was invalid.
    Override {
        /// Override failure message.
        message: String,
    },
    /// Range or cross-field validation failed.
    Validation {
        /// Validation failure messages.
        messages: Vec<String>,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    formatter,
                    "failed to read config {}: {source}",
                    path.display()
                )
            },
            Self::TomlDeserialize { source } => {
                write!(formatter, "failed to parse TOML config: {source}")
            },
            Self::TomlSerialize { source } => {
                write!(
                    formatter,
                    "failed to serialize canonical TOML config: {source}"
                )
            },
            Self::MissingSchemaVersion => {
                formatter.write_str("config schema_version is required and must be \"1.0.0\"")
            },
            Self::SchemaVersion { expected, found } => write!(
                formatter,
                "config schema_version {found:?} is unsupported; expected {expected:?}. Update the config file or use a compatible binary"
            ),
            Self::EnvVar {
                name,
                value,
                message,
            } => write!(
                formatter,
                "environment override {name}={value:?} could not be parsed: {message}"
            ),
            Self::Override { message } => formatter.write_str(message),
            Self::Validation { messages } => write!(
                formatter,
                "config validation failed: {}",
                messages.join("; ")
            ),
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::TomlDeserialize { source } => Some(source),
            Self::TomlSerialize { source } => Some(source),
            Self::MissingSchemaVersion
            | Self::SchemaVersion { .. }
            | Self::EnvVar { .. }
            | Self::Override { .. }
            | Self::Validation { .. } => None,
        }
    }
}

/// Load a config file with default, file, environment, and CLI override layers.
///
/// # Errors
///
/// Returns a [`ConfigError`] when reading, parsing, merging, validation, or
/// canonical hash generation fails.
pub fn load_root(
    config_path: &Path,
    env_overrides: &EnvOverrides,
    cli_sets: &[(String, String)],
) -> Result<LoadedRootConfig, ConfigError> {
    let text = fs::read_to_string(config_path).map_err(|source| ConfigError::Io {
        path: config_path.to_path_buf(),
        source,
    })?;

    load_root_from_str(&text, env_overrides, cli_sets)
}

/// Serialize a root config to the canonical TOML used for hashing.
///
/// # Errors
///
/// Returns an error if TOML serialization fails.
pub fn canonical_toml(root: &RootConfig) -> Result<String, ConfigError> {
    toml::to_string_pretty(root).map_err(|source| ConfigError::TomlSerialize { source })
}

/// Return the 12-hex-character BLAKE3 hash of the canonical root config TOML.
///
/// # Errors
///
/// Returns an error if TOML serialization fails.
pub fn canonical_hash(root: &RootConfig) -> Result<String, ConfigError> {
    let text = canonical_toml(root)?;
    Ok(blake3_hex_12(text.as_bytes()))
}

fn load_root_from_str(
    text: &str,
    env_overrides: &EnvOverrides,
    cli_sets: &[(String, String)],
) -> Result<LoadedRootConfig, ConfigError> {
    let file_value: toml::Value =
        toml::from_str(text).map_err(|source| ConfigError::TomlDeserialize { source })?;
    ensure_schema_version_present(&file_value)?;

    let mut merged = toml::Value::try_from(RootConfig::default())
        .map_err(|source| ConfigError::TomlSerialize { source })?;
    merge_values(&mut merged, file_value);
    apply_env_overrides(&mut merged, env_overrides)?;

    for (key, value) in cli_sets {
        apply_set(&mut merged, key, value)?;
    }

    let root: RootConfig = merged
        .try_into()
        .map_err(|source| ConfigError::TomlDeserialize { source })?;
    root.validate_all()?;
    let config_hash = canonical_hash(&root)?;

    Ok(LoadedRootConfig {
        root,
        config_hash,
        secrets: ConfigSecrets {
            hf_token: env_overrides.hf_token.clone(),
        },
        device: env_overrides.device.clone(),
    })
}

fn ensure_schema_version_present(value: &toml::Value) -> Result<(), ConfigError> {
    let Some(table) = value.as_table() else {
        return Err(ConfigError::Override {
            message: "config root must be a TOML table".to_string(),
        });
    };

    if table.contains_key("schema_version") {
        Ok(())
    } else {
        Err(ConfigError::MissingSchemaVersion)
    }
}

fn merge_values(base: &mut toml::Value, overlay: toml::Value) {
    match (base, overlay) {
        (toml::Value::Table(base_table), toml::Value::Table(overlay_table)) => {
            let changes_tagged_variant = base_table
                .get("kind")
                .zip(overlay_table.get("kind"))
                .is_some_and(|(base_kind, overlay_kind)| base_kind != overlay_kind);
            if changes_tagged_variant {
                *base_table = overlay_table;
                return;
            }

            for (key, overlay_value) in overlay_table {
                match base_table.get_mut(&key) {
                    Some(base_value) => merge_values(base_value, overlay_value),
                    None => {
                        base_table.insert(key, overlay_value);
                    },
                }
            }
        },
        (base_value, overlay_value) => *base_value = overlay_value,
    }
}

fn apply_env_overrides(
    value: &mut toml::Value,
    env_overrides: &EnvOverrides,
) -> Result<(), ConfigError> {
    if let Some(seed) = env_overrides.seed {
        let seed = i64::try_from(seed).map_err(|source| ConfigError::EnvVar {
            name: "LEWM_SEED".to_string(),
            value: seed.to_string(),
            message: source.to_string(),
        })?;
        apply_value(value, "training.seed", toml::Value::Integer(seed))?;
    }

    if let Some(lr_peak) = env_overrides.lr_peak {
        apply_value(value, "training.lr_peak", toml::Value::Float(lr_peak))?;
    }

    if let Some(endpoint) = &env_overrides.otel_endpoint {
        apply_value(
            value,
            "observability.otel_endpoint",
            toml::Value::String(endpoint.clone()),
        )?;
    }

    Ok(())
}

fn apply_set(value: &mut toml::Value, key: &str, raw_value: &str) -> Result<(), ConfigError> {
    validate_override_key(key)?;
    let parsed = parse_override_value(raw_value)?;
    apply_value(value, key, parsed)
}

fn apply_value(
    value: &mut toml::Value,
    dotted_key: &str,
    new_value: toml::Value,
) -> Result<(), ConfigError> {
    let mut segments = dotted_key.split('.').peekable();
    let mut current = value.as_table_mut().ok_or_else(|| ConfigError::Override {
        message: "root config must be a TOML table".to_string(),
    })?;

    while let Some(segment) = segments.next() {
        if segments.peek().is_none() {
            current.insert(segment.to_string(), new_value);
            return Ok(());
        }

        let entry = current
            .entry(segment.to_string())
            .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
        current = entry.as_table_mut().ok_or_else(|| ConfigError::Override {
            message: format!("override path {dotted_key:?} crosses non-table key {segment:?}"),
        })?;
    }

    Err(ConfigError::Override {
        message: "override key must not be empty".to_string(),
    })
}

fn validate_override_key(key: &str) -> Result<(), ConfigError> {
    if key.is_empty() {
        return Err(ConfigError::Override {
            message: "override key must not be empty".to_string(),
        });
    }

    if !key.contains('.') {
        return Err(ConfigError::Override {
            message: "override key must be a dotted path".to_string(),
        });
    }

    if key.split('.').any(str::is_empty) {
        return Err(ConfigError::Override {
            message: "override key must not contain empty path segments".to_string(),
        });
    }

    Ok(())
}

fn parse_override_value(raw_value: &str) -> Result<toml::Value, ConfigError> {
    #[derive(Deserialize)]
    struct Wrapper {
        value: toml::Value,
    }

    if raw_value.is_empty() {
        return Err(ConfigError::Override {
            message: "override value must not be empty".to_string(),
        });
    }

    let wrapped = format!("value = {raw_value}");
    let parsed: Wrapper =
        toml::from_str(&wrapped).map_err(|source| ConfigError::TomlDeserialize { source })?;

    if matches!(parsed.value, toml::Value::Table(_)) {
        return Err(ConfigError::Override {
            message: "override value must be a TOML scalar or array".to_string(),
        });
    }

    Ok(parsed.value)
}

fn optional_env_string(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.is_empty())
}

fn parse_optional_env<T>(name: &str) -> Result<Option<T>, ConfigError>
where
    T: std::str::FromStr,
    T::Err: fmt::Display,
{
    let Some(value) = optional_env_string(name) else {
        return Ok(None);
    };

    value
        .parse::<T>()
        .map(Some)
        .map_err(|source| ConfigError::EnvVar {
            name: name.to_string(),
            value,
            message: source.to_string(),
        })
}

fn push_validation_errors(
    errors: &mut Vec<String>,
    section: &str,
    result: Result<(), validator::ValidationErrors>,
) {
    if let Err(source) = result {
        errors.push(format!("{section}: {source}"));
    }
}

fn blake3_hex_12(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    let hash = blake3::hash(bytes);
    let mut output = String::with_capacity(12);
    for byte in &hash.as_bytes()[..6] {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn Error>>;

    const PUSHT_FIXTURE: &str = include_str!("../../../configs/pusht.toml");
    const SO100_FIXTURE: &str = include_str!("../../../configs/so100.toml");
    const SO100_WARMSTART_FIXTURE: &str = include_str!("../../../configs/so100_warmstart.toml");

    fn load_fixture(
        text: &str,
        env_overrides: &EnvOverrides,
        cli_sets: &[(String, String)],
    ) -> Result<LoadedRootConfig, ConfigError> {
        load_root_from_str(text, env_overrides, cli_sets)
    }

    #[test]
    fn unknown_field_rejected() {
        let err = load_fixture(
            r#"
schema_version = "1.0.0"

[training]
wieght_decay = 0.05
"#,
            &EnvOverrides::default(),
            &[],
        )
        .expect_err("unknown fields must fail");

        assert!(err.to_string().contains("unknown field"));
        assert!(err.to_string().contains("wieght_decay"));
    }

    #[test]
    fn numeric_out_of_range_rejected() {
        let err = load_fixture(
            r#"
schema_version = "1.0.0"

[training]
lr_peak = 1.0
"#,
            &EnvOverrides::default(),
            &[],
        )
        .expect_err("range validation must fail");

        assert!(err.to_string().contains("training"));
        assert!(err.to_string().contains("lr_peak"));
    }

    #[test]
    fn cross_field_validation() {
        let err = load_fixture(
            r#"
schema_version = "1.0.0"

[eval]
horizon_plan = 64
"#,
            &EnvOverrides::default(),
            &[],
        )
        .expect_err("cross-field validation must fail");

        assert!(err.to_string().contains("eval.horizon_plan"));
        assert!(err.to_string().contains("model.predictor.num_frames"));
    }

    #[test]
    fn cli_set_overrides_file() -> TestResult {
        let loaded = load_fixture(
            r#"
schema_version = "1.0.0"

[training]
lr_peak = 2.0e-4
batch_size = 64
"#,
            &EnvOverrides::default(),
            &[
                ("training.lr_peak".to_string(), "1.0e-4".to_string()),
                ("training.batch_size".to_string(), "32".to_string()),
            ],
        )?;

        assert!((loaded.root.training.lr_peak - 1.0e-4).abs() <= f64::EPSILON);
        assert_eq!(loaded.root.training.batch_size, 32);

        Ok(())
    }

    #[test]
    fn env_overrides_file_but_not_cli() -> TestResult {
        let loaded = load_fixture(
            r#"
schema_version = "1.0.0"

[training]
lr_peak = 2.0e-4
seed = 5
"#,
            &EnvOverrides {
                seed: Some(42),
                lr_peak: Some(3.0e-4),
                ..EnvOverrides::default()
            },
            &[("training.lr_peak".to_string(), "1.0e-4".to_string())],
        )?;

        assert_eq!(loaded.root.training.seed, 42);
        assert!((loaded.root.training.lr_peak - 1.0e-4).abs() <= f64::EPSILON);

        Ok(())
    }

    #[test]
    fn secrets_and_device_do_not_affect_config_hash() -> TestResult {
        let base = load_fixture(
            r#"
schema_version = "1.0.0"
"#,
            &EnvOverrides::default(),
            &[],
        )?;
        let with_runtime_inputs = load_fixture(
            r#"
schema_version = "1.0.0"
"#,
            &EnvOverrides {
                hf_token: Some("hf_secret".to_string()),
                device: Some("cuda:0".to_string()),
                ..EnvOverrides::default()
            },
            &[],
        )?;

        assert_eq!(base.config_hash, with_runtime_inputs.config_hash);
        assert_eq!(
            with_runtime_inputs.secrets.hf_token.as_deref(),
            Some("hf_secret")
        );
        assert_eq!(with_runtime_inputs.device.as_deref(), Some("cuda:0"));

        Ok(())
    }

    #[test]
    fn config_hash_stable_under_reformat() -> TestResult {
        let a = load_fixture(
            r#"
schema_version = "1.0.0"

[training]
lr_peak = 3.0e-4
"#,
            &EnvOverrides::default(),
            &[],
        )?;
        let b = load_fixture(
            r#"
schema_version = "1.0.0"

[training]
lr_peak = 0.000300
"#,
            &EnvOverrides::default(),
            &[],
        )?;

        assert_eq!(a.config_hash, b.config_hash);
        assert_eq!(a.config_hash.len(), 12);

        Ok(())
    }

    #[test]
    fn config_hash_differs_on_meaningful_change() -> TestResult {
        let a = load_fixture(
            r#"
schema_version = "1.0.0"

[training]
lr_peak = 3.0e-4
"#,
            &EnvOverrides::default(),
            &[],
        )?;
        let b = load_fixture(
            r#"
schema_version = "1.0.0"

[training]
lr_peak = 4.0e-4
"#,
            &EnvOverrides::default(),
            &[],
        )?;

        assert_ne!(a.config_hash, b.config_hash);

        Ok(())
    }

    #[test]
    fn schema_version_mismatch_explicit_error() {
        let err = load_fixture(
            r#"
schema_version = "9.9.9"
"#,
            &EnvOverrides::default(),
            &[],
        )
        .expect_err("schema mismatch must fail");

        assert!(err.to_string().contains("schema_version"));
        assert!(err.to_string().contains("1.0.0"));
        assert!(err.to_string().contains("9.9.9"));
    }

    #[test]
    fn pusht_toml_loads_and_validates() -> TestResult {
        let loaded = load_fixture(PUSHT_FIXTURE, &EnvOverrides::default(), &[])?;

        assert_eq!(loaded.root.schema_version, SUPPORTED_SCHEMA_VERSION);
        assert_eq!(loaded.root.training.batch_size, 64);
        assert_eq!(loaded.root.training.effective_batch(), Some(128));
        assert_eq!(loaded.config_hash.len(), 12);
        assert!(loaded.root.validation_warnings().is_empty());

        let DatasetConfig::Pusht(dataset) = loaded.root.dataset else {
            return Err("expected PushT dataset config".into());
        };
        assert_eq!(dataset.horizon, 8);
        assert_eq!(dataset.history_size, 3);

        Ok(())
    }

    #[test]
    fn so100_toml_loads_and_validates() -> TestResult {
        let loaded = load_fixture(SO100_FIXTURE, &EnvOverrides::default(), &[])?;

        let DatasetConfig::So100(dataset) = loaded.root.dataset else {
            return Err("expected SO-100 dataset config".into());
        };
        assert_eq!(dataset.camera_view, CameraView::Top);
        assert_eq!(dataset.action_dim, SO100_ACTION_DIM);
        assert_eq!(loaded.root.model.action_encoder.input_dim, SO100_ACTION_DIM);
        assert_eq!(loaded.root.training.warmup_steps, SO100_WARMUP_STEPS);
        assert_eq!(loaded.root.training.warmstart_from, None);

        let EvalConfig::So100Latent(eval) = loaded.root.eval else {
            return Err("expected SO-100 latent eval config".into());
        };
        assert_eq!(eval.episode_ids, SO100_HELDOUT_EPISODES);

        Ok(())
    }

    #[test]
    fn so100_warmstart_matches_base_except_checkpoint() -> TestResult {
        let base = load_fixture(SO100_FIXTURE, &EnvOverrides::default(), &[])?;
        let mut warm = load_fixture(SO100_WARMSTART_FIXTURE, &EnvOverrides::default(), &[])?;

        assert_eq!(
            warm.root.training.warmstart_from.as_deref(),
            Some(Path::new(DEFAULT_SO100_WARMSTART_FROM))
        );
        warm.root.training.warmstart_from = None;
        assert_eq!(warm.root, base.root);

        Ok(())
    }
}
