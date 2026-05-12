//! Strict training configuration loading for `lewm-train`.

use std::{
    fs,
    path::{Path, PathBuf},
};

use lewm_core::JepaConfig;
use serde::{Deserialize, Serialize};
use validator::Validate;

/// Supported training configuration schema version.
pub const CONFIG_SCHEMA_VERSION: &str = "1.0.0";

/// SO-100 action dimension from RFC 0012.
pub const SO100_ACTION_DIM: usize = 6;

/// SO-100 warmup length from RFC 0012 section 10.
pub const SO100_WARMUP_STEPS: u32 = 500;

/// Pinned SO-100 held-out episode IDs.
pub const SO100_HELDOUT_EPISODES: [u32; 5] = [5, 14, 23, 31, 42];

/// Default `PushT` checkpoint used by the SO-100 warm-start arm.
pub const DEFAULT_SO100_WARMSTART_FROM: &str = "/checkpoints/lewm-rs-pusht/step_0014400.mpk";

/// Errors raised while loading or validating training configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The config file could not be read.
    #[error("could not read config at {path}: {source}")]
    Read {
        /// Path that failed to read.
        path: PathBuf,
        /// Filesystem error.
        #[source]
        source: std::io::Error,
    },

    /// TOML deserialization failed.
    #[error("invalid TOML config at {path}: {source}")]
    Parse {
        /// Path that failed to parse.
        path: PathBuf,
        /// TOML parser error.
        #[source]
        source: toml::de::Error,
    },

    /// TOML serialization failed.
    #[error("could not serialize config: {0}")]
    Serialize(#[from] toml::ser::Error),

    /// The schema version does not match the loader.
    #[error("unsupported config schema_version {found:?}; expected {expected:?}")]
    SchemaVersion {
        /// Parsed schema version.
        found: String,
        /// Loader-supported schema version.
        expected: &'static str,
    },

    /// The config is syntactically valid but semantically invalid.
    #[error("invalid config: {0}")]
    Validation(String),
}

/// Top-level training configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootConfig {
    /// Schema version for migration-safe loading.
    pub schema_version: String,
    /// Dataset section.
    pub dataset: DatasetConfig,
    /// JEPA model section.
    pub model: JepaConfig,
    /// Loss section.
    #[serde(default)]
    pub loss: LossConfig,
    /// Training section.
    pub training: TrainingConfig,
    /// Evaluation section.
    pub eval: EvalConfig,
    /// Optional observability section.
    #[serde(default)]
    pub observability: ObservabilityConfig,
    /// Optional Hub publication section.
    #[serde(default)]
    pub hub: HubConfig,
    /// Reserved experimental overrides.
    #[serde(default)]
    pub experimental: ExperimentalConfig,
}

impl RootConfig {
    /// Validate the loaded configuration contract.
    ///
    /// # Errors
    ///
    /// Returns an error when the schema version, nested numeric ranges, model
    /// shape contract, or dataset-specific invariants are invalid.
    pub fn validate_contract(&self) -> Result<(), ConfigError> {
        if self.schema_version != CONFIG_SCHEMA_VERSION {
            return Err(ConfigError::SchemaVersion {
                found: self.schema_version.clone(),
                expected: CONFIG_SCHEMA_VERSION,
            });
        }

        self.model
            .validate()
            .map_err(|errors| ConfigError::Validation(errors.to_string()))?;
        self.model
            .validate_shape_contract()
            .map_err(|errors| ConfigError::Validation(errors.join("; ")))?;
        self.loss
            .validate()
            .map_err(|errors| ConfigError::Validation(errors.to_string()))?;
        self.training
            .validate()
            .map_err(|errors| ConfigError::Validation(errors.to_string()))?;
        self.training.validate_betas()?;

        if self.training.history_size != self.model.history_size {
            return Err(ConfigError::Validation(format!(
                "training.history_size ({}) must equal model.history_size ({})",
                self.training.history_size, self.model.history_size
            )));
        }
        if self.training.horizon != self.model.horizon {
            return Err(ConfigError::Validation(format!(
                "training.horizon ({}) must equal model.horizon ({})",
                self.training.horizon, self.model.horizon
            )));
        }

        match (&self.dataset, &self.eval) {
            (DatasetConfig::So100(_), EvalConfig::So100LatentRollout(eval)) => {
                validate_so100_contract(self, eval)?;
            },
        }

        Ok(())
    }
}

/// Dataset config variants.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DatasetConfig {
    /// Pre-decoded SO-100 HDF5 dataset.
    So100(So100DatasetConfig),
}

/// SO-100 dataset config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct So100DatasetConfig {
    /// Path to the RFC 0012 HDF5 mirror.
    pub hdf5_path: PathBuf,
    /// Camera view selected for v1 training.
    pub camera_view: CameraView,
    /// Path to persisted SO-100 action stats.
    pub stats_path: PathBuf,
}

/// SO-100 camera view selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraView {
    /// Top camera.
    Top,
    /// Wrist camera.
    Wrist,
}

/// Loss config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct LossConfig {
    /// `SigReg` loss weight.
    #[validate(range(min = 0.0, max = 100.0))]
    pub lambda_sigreg: f64,
    /// Number of `SigReg` spline knots.
    #[validate(range(min = 8, max = 64))]
    pub sigreg_knots: usize,
    /// Number of `SigReg` random projections.
    #[validate(range(min = 64, max = 8192))]
    pub sigreg_num_proj: usize,
    /// Maximum `SigReg` spline time.
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

/// Training config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    /// Number of historical context frames.
    #[validate(range(min = 1, max = 100))]
    pub history_size: usize,
    /// Prediction horizon.
    #[validate(range(min = 2, max = 64))]
    pub horizon: usize,
    /// Per-step batch size.
    #[validate(range(min = 1, max = 1024))]
    pub batch_size: usize,
    /// Gradient accumulation factor.
    #[validate(range(min = 1, max = 32))]
    pub grad_accum_steps: usize,
    /// Optimizer kind.
    pub optimizer: OptimizerKind,
    /// Peak learning rate.
    #[validate(range(min = 1.0e-7, max = 1.0e-2))]
    pub lr_peak: f64,
    /// Final learning rate.
    #[validate(range(min = 1.0e-9, max = 1.0e-3))]
    pub lr_min: f64,
    /// Linear warmup steps.
    #[validate(range(min = 0, max = 100_000))]
    pub warmup_steps: u32,
    /// Weight decay.
    #[validate(range(min = 0.0, max = 1.0))]
    pub weight_decay: f64,
    /// Adam-style beta values.
    pub betas: [f64; 2],
    /// Training epochs.
    #[validate(range(min = 1, max = 1000))]
    pub epochs: u32,
    /// Numeric precision.
    pub precision: PrecisionKind,
    /// Random seed.
    pub seed: u64,
    /// Gradient clipping norm.
    #[validate(range(min = 0.1, max = 100.0))]
    pub grad_clip_norm: f64,
    /// Optional warm-start checkpoint path.
    #[serde(default)]
    pub warmstart_from: Option<PathBuf>,
    /// Evaluation cadence.
    #[validate(range(min = 1, max = 100))]
    pub eval_every_n_epochs: u32,
    /// Collapse-probe cadence.
    #[validate(range(min = 10, max = 10_000))]
    pub probe_every_n_steps: u32,
}

impl TrainingConfig {
    fn validate_betas(&self) -> Result<(), ConfigError> {
        let [beta1, beta2] = self.betas;
        if !(0.0..1.0).contains(&beta1) || !(0.0..1.0).contains(&beta2) {
            return Err(ConfigError::Validation(
                "training.betas values must be in [0.0, 1.0)".to_owned(),
            ));
        }
        if beta1 >= beta2 {
            return Err(ConfigError::Validation(
                "training.betas[0] must be less than training.betas[1]".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Optimizer selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    /// `AdamW` optimizer.
    Adamw,
    /// Lion optimizer.
    Lion,
}

/// Numeric precision selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionKind {
    /// Full precision.
    F32,
    /// Mixed BF16 precision.
    Bf16Mixed,
}

/// Evaluation config variants.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvalConfig {
    /// SO-100 latent rollout evaluation.
    So100LatentRollout(So100EvalConfig),
}

/// SO-100 latent rollout eval config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct So100EvalConfig {
    /// Held-out episode IDs.
    pub episode_ids: Vec<u32>,
}

/// Optional observability config.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ObservabilityConfig {
    /// Optional run-name prefix.
    pub trackio_run_name_prefix: Option<String>,
    /// Optional OpenTelemetry endpoint environment variable name.
    pub otel_endpoint_env: Option<String>,
    /// Optional `TensorBoard` output directory.
    pub tensorboard_dir: Option<PathBuf>,
}

/// Optional Hugging Face Hub config.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HubConfig {
    /// Optional model repository.
    pub model_repo: Option<String>,
    /// Upload final artifacts at the end of training.
    pub upload_at_end: bool,
    /// Upload every N epochs when non-zero.
    pub upload_every_n_epochs: u32,
}

/// Reserved experimental config.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExperimentalConfig {}

/// Load and validate a root config from disk.
///
/// # Errors
///
/// Returns an error when the file cannot be read, parsed, or validated.
pub fn load_root_config(path: impl AsRef<Path>) -> Result<RootConfig, ConfigError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let config = toml::from_str::<RootConfig>(&text).map_err(|source| ConfigError::Parse {
        path: path.to_path_buf(),
        source,
    })?;
    config.validate_contract()?;
    Ok(config)
}

/// Serialize a config to canonical pretty TOML.
///
/// # Errors
///
/// Returns an error when TOML serialization fails.
pub fn to_pretty_toml(config: &RootConfig) -> Result<String, ConfigError> {
    Ok(toml::to_string_pretty(config)?)
}

fn validate_so100_contract(root: &RootConfig, eval: &So100EvalConfig) -> Result<(), ConfigError> {
    if root.model.action_encoder.input_dim != SO100_ACTION_DIM {
        return Err(ConfigError::Validation(format!(
            "SO-100 requires model.action_encoder.input_dim {SO100_ACTION_DIM}, found {}",
            root.model.action_encoder.input_dim
        )));
    }
    if root.training.warmup_steps != SO100_WARMUP_STEPS {
        return Err(ConfigError::Validation(format!(
            "SO-100 requires training.warmup_steps {SO100_WARMUP_STEPS}, found {}",
            root.training.warmup_steps
        )));
    }
    if eval.episode_ids != SO100_HELDOUT_EPISODES {
        return Err(ConfigError::Validation(format!(
            "SO-100 eval.episode_ids must be {:?}, found {:?}",
            SO100_HELDOUT_EPISODES, eval.episode_ids
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;

    #[test]
    fn so100_toml_loads_and_validates() -> Result<(), ConfigError> {
        let config = load_root_config(repo_path("configs/so100.toml"))?;

        assert!(matches!(config.dataset, DatasetConfig::So100(_)));
        assert_eq!(config.model.action_encoder.input_dim, SO100_ACTION_DIM);
        assert_eq!(config.training.warmup_steps, SO100_WARMUP_STEPS);
        assert_eq!(config.training.warmstart_from, None);
        let EvalConfig::So100LatentRollout(eval) = config.eval;
        assert_eq!(eval.episode_ids, SO100_HELDOUT_EPISODES);
        Ok(())
    }

    #[test]
    fn so100_warmstart_matches_base_except_checkpoint() -> Result<(), ConfigError> {
        let base = load_root_config(repo_path("configs/so100.toml"))?;
        let mut warm = load_root_config(repo_path("configs/so100_warmstart.toml"))?;

        assert_eq!(
            warm.training.warmstart_from.as_deref(),
            Some(Path::new(DEFAULT_SO100_WARMSTART_FROM))
        );
        warm.training.warmstart_from = None;
        assert_eq!(warm, base);
        Ok(())
    }

    #[test]
    fn so100_unknown_fields_are_rejected() {
        let err = toml::from_str::<RootConfig>(
            r#"
schema_version = "1.0.0"

[dataset]
kind = "so100"
hdf5_path = "/data/so100/svla_so100_pickplace.h5"
camera_view = "top"
stats_path = "/data/so100/stats.safetensors"
typo = true

[model]

[training]
history_size = 3
horizon = 8
batch_size = 64
grad_accum_steps = 2
optimizer = "adamw"
lr_peak = 3.0e-4
lr_min = 1.0e-5
warmup_steps = 500
weight_decay = 0.05
betas = [0.9, 0.95]
epochs = 10
precision = "bf16_mixed"
seed = 0
grad_clip_norm = 1.0
eval_every_n_epochs = 5
probe_every_n_steps = 100

[eval]
kind = "so100_latent_rollout"
episode_ids = [5, 14, 23, 31, 42]
"#,
        )
        .expect_err("unknown dataset field should fail");

        assert!(err.to_string().contains("unknown field"));
    }

    fn repo_path(relative: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(relative)
    }
}
