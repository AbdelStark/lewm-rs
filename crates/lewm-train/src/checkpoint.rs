//! Epoch checkpoint file contract for `lewm-train`.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde::{Deserialize, Serialize};

/// Sidecar schema version for RFC 0005 checkpoints.
pub const CHECKPOINT_SCHEMA_VERSION: &str = "1.0";

/// Number of local epoch checkpoints retained after each save.
pub const DEFAULT_RETAINED_CHECKPOINTS: usize = 3;

/// Error returned by checkpoint persistence and loading APIs.
#[derive(Debug)]
pub enum CheckpointError {
    /// Filesystem operation failed.
    Io {
        /// Path involved in the failed operation.
        path: PathBuf,
        /// Original I/O error.
        source: std::io::Error,
    },
    /// Sidecar or parity JSON serialization failed.
    Json {
        /// Original JSON error.
        source: serde_json::Error,
    },
    /// Safetensors serialization or parsing failed.
    Safetensors {
        /// Original safetensors error.
        source: safetensors::SafeTensorError,
    },
    /// A parameter tensor shape does not match its flat data length.
    TensorShapeMismatch {
        /// Parameter name.
        name: String,
        /// Expected element count from the shape.
        expected: usize,
        /// Actual flat data length.
        found: usize,
    },
    /// A parameter tensor name was empty.
    EmptyTensorName,
    /// A checkpoint sidecar did not have a parent directory.
    SidecarWithoutParent {
        /// Sidecar path.
        path: PathBuf,
    },
    /// The requested checkpoint did not have all four RFC 0005 files.
    IncompleteCheckpoint {
        /// Step that was incomplete.
        step: u64,
    },
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    formatter,
                    "checkpoint I/O error at {}: {source}",
                    path.display()
                )
            },
            Self::Json { source } => write!(formatter, "checkpoint JSON error: {source}"),
            Self::Safetensors { source } => {
                write!(formatter, "checkpoint safetensors error: {source}")
            },
            Self::TensorShapeMismatch {
                name,
                expected,
                found,
            } => write!(
                formatter,
                "checkpoint tensor {name:?} expected {expected} elements from shape, found {found}"
            ),
            Self::EmptyTensorName => formatter.write_str("checkpoint tensor name cannot be empty"),
            Self::SidecarWithoutParent { path } => write!(
                formatter,
                "checkpoint sidecar has no parent directory: {}",
                path.display()
            ),
            Self::IncompleteCheckpoint { step } => {
                write!(formatter, "checkpoint step {step:07} is incomplete")
            },
        }
    }
}

impl std::error::Error for CheckpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source } => Some(source),
            Self::Safetensors { source } => Some(source),
            Self::TensorShapeMismatch { .. }
            | Self::EmptyTensorName
            | Self::SidecarWithoutParent { .. }
            | Self::IncompleteCheckpoint { .. } => None,
        }
    }
}

impl From<serde_json::Error> for CheckpointError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json { source }
    }
}

impl From<safetensors::SafeTensorError> for CheckpointError {
    fn from(source: safetensors::SafeTensorError) -> Self {
        Self::Safetensors { source }
    }
}

/// Four co-located checkpoint files for one optimizer step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointPaths {
    /// `MessagePack` Burn record path.
    pub model_burn: PathBuf,
    /// Safetensors model-parameter mirror path.
    pub model_safetensors: PathBuf,
    /// JSON sidecar path.
    pub sidecar: PathBuf,
    /// Per-epoch parity probe JSON path.
    pub parity: PathBuf,
}

impl CheckpointPaths {
    /// Return paths for `step` under `output_dir`.
    pub fn for_step(output_dir: impl AsRef<Path>, step: u64) -> Self {
        let output_dir = output_dir.as_ref();
        Self {
            model_burn: output_dir.join(step_file_name(step, "mpk")),
            model_safetensors: output_dir.join(step_file_name(step, "safetensors")),
            sidecar: output_dir.join(step_file_name(step, "json")),
            parity: output_dir.join(step_parity_file_name(step)),
        }
    }

    /// Whether all four checkpoint files exist.
    pub fn is_complete(&self) -> bool {
        self.model_burn.is_file()
            && self.model_safetensors.is_file()
            && self.sidecar.is_file()
            && self.parity.is_file()
    }
}

/// Names of files referenced from the JSON sidecar.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CheckpointFiles {
    /// Burn record file name.
    pub model_burn: String,
    /// Safetensors mirror file name.
    pub model_safetensors: String,
    /// Parity probe file name.
    pub parity: String,
}

/// Serializable RNG state captured in the sidecar.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CheckpointRngState {
    /// Run-global seed.
    pub global_seed: u64,
    /// Optimizer step at save time.
    pub step_at_save: u64,
    /// Serialized `rng:data_shuffle` state.
    pub data_shuffle: String,
    /// Serialized `rng:sigreg_sketch` state.
    pub sigreg_sketch: String,
    /// Serialized `rng:dropout` state.
    pub dropout: String,
    /// Serialized `rng:cem` state.
    pub cem: String,
    /// Serialized `rng:model_init` state.
    pub model_init: String,
}

/// Per-epoch parity probe payload written beside the checkpoint.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct ParityProbe {
    /// Encoder CLS-vector L-infinity drift.
    pub encoder_cls_l_inf: f64,
    /// Predictor-output L-infinity drift.
    pub predictor_l_inf: f64,
    /// `SIGReg` scalar value observed by the probe.
    pub sigreg_value: f64,
}

/// RFC 0005 checkpoint sidecar schema.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CheckpointSidecar {
    /// Sidecar schema version.
    pub schema_version: String,
    /// Run identifier.
    pub run_id: String,
    /// Optimizer step at the checkpoint boundary.
    pub step: u64,
    /// Epoch index at the checkpoint boundary.
    pub epoch: u64,
    /// Wall-clock training time in seconds.
    pub wall_time_s: f64,
    /// Git short SHA captured at save time.
    pub git_short_sha: String,
    /// Twelve-hex BLAKE3 hash of canonical TOML config bytes.
    pub config_hash: String,
    /// Serializable RNG substream state.
    pub rng_state: CheckpointRngState,
    /// Last-step scalar metrics.
    pub metrics_last_step: BTreeMap<String, f64>,
    /// Checkpoint file names co-located with this sidecar.
    pub checkpoint_files: CheckpointFiles,
}

/// Model-parameter tensor dtype mirrored into safetensors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParameterTensorDtype {
    /// IEEE-754 single precision float.
    F32,
    /// Signed 64-bit integer.
    I64,
}

impl ParameterTensorDtype {
    fn safetensors_dtype(self) -> Dtype {
        match self {
            Self::F32 => Dtype::F32,
            Self::I64 => Dtype::I64,
        }
    }
}

/// Flat model-parameter values mirrored into safetensors.
#[derive(Clone, Debug, PartialEq)]
pub enum ParameterTensorValues {
    /// IEEE-754 single precision values.
    F32(Vec<f32>),
    /// Signed 64-bit integer values.
    I64(Vec<i64>),
}

impl ParameterTensorValues {
    fn dtype(&self) -> ParameterTensorDtype {
        match self {
            Self::F32(_) => ParameterTensorDtype::F32,
            Self::I64(_) => ParameterTensorDtype::I64,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::I64(values) => values.len(),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            Self::F32(values) => f32_bytes(values),
            Self::I64(values) => i64_bytes(values),
        }
    }
}

/// Model-parameter tensor to mirror into safetensors.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterTensor {
    /// Stable parameter name from the model walker.
    pub name: String,
    /// Tensor shape in row-major order.
    pub shape: Vec<usize>,
    /// Flat parameter values.
    pub values: ParameterTensorValues,
}

impl ParameterTensor {
    /// Create a validated F32 parameter tensor.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::EmptyTensorName`] when `name` is empty, or
    /// [`CheckpointError::TensorShapeMismatch`] when `shape` does not match the
    /// flat data length.
    pub fn f32(
        name: impl Into<String>,
        shape: Vec<usize>,
        data: Vec<f32>,
    ) -> Result<Self, CheckpointError> {
        let tensor = Self {
            name: name.into(),
            shape,
            values: ParameterTensorValues::F32(data),
        };
        tensor.validate()?;
        Ok(tensor)
    }

    /// Create a validated I64 parameter tensor.
    ///
    /// # Errors
    ///
    /// Returns [`CheckpointError::EmptyTensorName`] when `name` is empty, or
    /// [`CheckpointError::TensorShapeMismatch`] when `shape` does not match the
    /// flat data length.
    pub fn i64(
        name: impl Into<String>,
        shape: Vec<usize>,
        data: Vec<i64>,
    ) -> Result<Self, CheckpointError> {
        let tensor = Self {
            name: name.into(),
            shape,
            values: ParameterTensorValues::I64(data),
        };
        tensor.validate()?;
        Ok(tensor)
    }

    fn validate(&self) -> Result<(), CheckpointError> {
        if self.name.is_empty() {
            return Err(CheckpointError::EmptyTensorName);
        }
        let expected = element_count(&self.shape, &self.name)?;
        if expected != self.values.len() {
            return Err(CheckpointError::TensorShapeMismatch {
                name: self.name.clone(),
                expected,
                found: self.values.len(),
            });
        }
        Ok(())
    }
}

/// Inputs required to write one checkpoint.
#[derive(Clone, Debug)]
pub struct CheckpointWriteRequest<'a> {
    /// Output directory for all checkpoint files.
    pub output_dir: &'a Path,
    /// Run identifier.
    pub run_id: &'a str,
    /// Optimizer step at the checkpoint boundary.
    pub step: u64,
    /// Epoch index at the checkpoint boundary.
    pub epoch: u64,
    /// Wall-clock training time in seconds.
    pub wall_time_s: f64,
    /// Git short SHA captured at save time.
    pub git_short_sha: &'a str,
    /// Twelve-hex BLAKE3 hash of canonical TOML config bytes.
    pub config_hash: &'a str,
    /// Serializable RNG state.
    pub rng_state: CheckpointRngState,
    /// Last-step scalar metrics.
    pub metrics_last_step: BTreeMap<String, f64>,
    /// Serialized Burn record bytes from `NamedMpkFileRecorder`.
    pub burn_record: &'a [u8],
    /// Model parameters to mirror into safetensors.
    pub parameters: &'a [ParameterTensor],
    /// Per-epoch parity probe payload.
    pub parity: &'a ParityProbe,
}

/// Checkpoint loaded from a complete sidecar.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedCheckpoint {
    /// Sidecar metadata.
    pub sidecar: CheckpointSidecar,
    /// Burn record bytes.
    pub burn_record: Vec<u8>,
    /// Safetensors mirror bytes.
    pub safetensors_bytes: Vec<u8>,
    /// Parity probe payload.
    pub parity: ParityProbe,
}

/// Return the RFC 0005 step file name for an extension.
pub fn step_file_name(step: u64, extension: &str) -> String {
    format!("step_{step:07}.{extension}")
}

/// Return the RFC 0005 parity probe file name.
pub fn step_parity_file_name(step: u64) -> String {
    format!("step_{step:07}.parity.json")
}

/// Hash already-canonical TOML bytes per RFC 0005/0018.
pub fn config_hash_from_canonical_toml(canonical_toml: &[u8]) -> String {
    let hash = blake3::hash(canonical_toml);
    hex_12(hash.as_bytes())
}

/// Write one RFC 0005 checkpoint and prune older local checkpoints.
///
/// # Errors
///
/// Returns an error when parameter validation, serialization, atomic file
/// writes, or pruning fails.
pub fn save_checkpoint(
    request: &CheckpointWriteRequest<'_>,
) -> Result<CheckpointPaths, CheckpointError> {
    for parameter in request.parameters {
        parameter.validate()?;
    }

    fs::create_dir_all(request.output_dir)
        .map_err(|source| io_error(request.output_dir, source))?;

    let paths = CheckpointPaths::for_step(request.output_dir, request.step);
    let safetensors_bytes = serialize_parameters_to_safetensors(request.parameters)?;
    let parity_bytes = serde_json::to_vec_pretty(request.parity)?;
    let sidecar = sidecar_from_request(request);
    let sidecar_bytes = serde_json::to_vec_pretty(&sidecar)?;

    write_atomic_bytes(&paths.model_burn, request.burn_record)?;
    write_atomic_bytes(&paths.model_safetensors, &safetensors_bytes)?;
    write_atomic_bytes(&paths.parity, &parity_bytes)?;
    write_atomic_bytes(&paths.sidecar, &sidecar_bytes)?;
    prune_checkpoints(request.output_dir, DEFAULT_RETAINED_CHECKPOINTS)?;

    Ok(paths)
}

/// Load a checkpoint sidecar.
///
/// # Errors
///
/// Returns an error if the sidecar cannot be read or parsed.
pub fn load_sidecar(path: impl AsRef<Path>) -> Result<CheckpointSidecar, CheckpointError> {
    let path = path.as_ref();
    let raw = fs::read(path).map_err(|source| io_error(path, source))?;
    serde_json::from_slice(&raw).map_err(CheckpointError::from)
}

/// Load a complete checkpoint from its sidecar path.
///
/// # Errors
///
/// Returns an error if the sidecar has no parent directory, any referenced file
/// is missing, or any payload cannot be read or parsed.
pub fn load_checkpoint(
    sidecar_path: impl AsRef<Path>,
) -> Result<LoadedCheckpoint, CheckpointError> {
    let sidecar_path = sidecar_path.as_ref();
    let sidecar = load_sidecar(sidecar_path)?;
    let output_dir =
        sidecar_path
            .parent()
            .ok_or_else(|| CheckpointError::SidecarWithoutParent {
                path: sidecar_path.to_path_buf(),
            })?;
    let paths = CheckpointPaths {
        model_burn: output_dir.join(&sidecar.checkpoint_files.model_burn),
        model_safetensors: output_dir.join(&sidecar.checkpoint_files.model_safetensors),
        sidecar: sidecar_path.to_path_buf(),
        parity: output_dir.join(&sidecar.checkpoint_files.parity),
    };
    if !paths.is_complete() {
        return Err(CheckpointError::IncompleteCheckpoint { step: sidecar.step });
    }

    let burn_record =
        fs::read(&paths.model_burn).map_err(|source| io_error(&paths.model_burn, source))?;
    let safetensors_bytes = fs::read(&paths.model_safetensors)
        .map_err(|source| io_error(&paths.model_safetensors, source))?;
    SafeTensors::deserialize(&safetensors_bytes)?;
    let parity_raw = fs::read(&paths.parity).map_err(|source| io_error(&paths.parity, source))?;
    let parity = serde_json::from_slice(&parity_raw)?;

    Ok(LoadedCheckpoint {
        sidecar,
        burn_record,
        safetensors_bytes,
        parity,
    })
}

/// Return complete checkpoints sorted by ascending step.
///
/// # Errors
///
/// Returns an error if the output directory cannot be read.
pub fn list_complete_checkpoints(
    output_dir: impl AsRef<Path>,
) -> Result<Vec<CheckpointPaths>, CheckpointError> {
    let output_dir = output_dir.as_ref();
    let entries = match fs::read_dir(output_dir) {
        Ok(entries) => entries,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => return Err(io_error(output_dir, source)),
    };

    let mut checkpoints = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| io_error(output_dir, source))?;
        let Some(step) = step_from_sidecar_path(&entry.path()) else {
            continue;
        };
        let paths = CheckpointPaths::for_step(output_dir, step);
        if paths.is_complete() {
            checkpoints.push(paths);
        }
    }

    checkpoints.sort_by_key(|paths| step_from_sidecar_path(&paths.sidecar).unwrap_or_default());
    Ok(checkpoints)
}

/// Return the latest complete checkpoint, if any.
///
/// # Errors
///
/// Returns an error if the output directory cannot be read.
pub fn latest_complete_checkpoint(
    output_dir: impl AsRef<Path>,
) -> Result<Option<CheckpointPaths>, CheckpointError> {
    let mut checkpoints = list_complete_checkpoints(output_dir)?;
    Ok(checkpoints.pop())
}

/// Load the latest complete checkpoint, if any.
///
/// # Errors
///
/// Returns an error if checkpoint discovery or loading fails.
pub fn load_latest_checkpoint(
    output_dir: impl AsRef<Path>,
) -> Result<Option<LoadedCheckpoint>, CheckpointError> {
    latest_complete_checkpoint(output_dir)?
        .map(|paths| load_checkpoint(paths.sidecar))
        .transpose()
}

/// Remove older local checkpoints and keep the newest `keep_last` complete steps.
///
/// # Errors
///
/// Returns an error if checkpoint discovery or file deletion fails.
pub fn prune_checkpoints(
    output_dir: impl AsRef<Path>,
    keep_last: usize,
) -> Result<Vec<CheckpointPaths>, CheckpointError> {
    let mut checkpoints = list_complete_checkpoints(output_dir)?;
    if checkpoints.len() <= keep_last {
        return Ok(Vec::new());
    }

    let remove_count = checkpoints.len() - keep_last;
    let removed = checkpoints.drain(..remove_count).collect::<Vec<_>>();
    for paths in &removed {
        remove_if_exists(&paths.model_burn)?;
        remove_if_exists(&paths.model_safetensors)?;
        remove_if_exists(&paths.parity)?;
        remove_if_exists(&paths.sidecar)?;
    }
    Ok(removed)
}

fn sidecar_from_request(request: &CheckpointWriteRequest<'_>) -> CheckpointSidecar {
    CheckpointSidecar {
        schema_version: CHECKPOINT_SCHEMA_VERSION.to_owned(),
        run_id: request.run_id.to_owned(),
        step: request.step,
        epoch: request.epoch,
        wall_time_s: request.wall_time_s,
        git_short_sha: request.git_short_sha.to_owned(),
        config_hash: request.config_hash.to_owned(),
        rng_state: request.rng_state.clone(),
        metrics_last_step: request.metrics_last_step.clone(),
        checkpoint_files: CheckpointFiles {
            model_burn: step_file_name(request.step, "mpk"),
            model_safetensors: step_file_name(request.step, "safetensors"),
            parity: step_parity_file_name(request.step),
        },
    }
}

fn serialize_parameters_to_safetensors(
    parameters: &[ParameterTensor],
) -> Result<Vec<u8>, CheckpointError> {
    let owned = parameters
        .iter()
        .map(|parameter| {
            parameter.validate().map(|()| {
                (
                    parameter.name.clone(),
                    parameter.shape.clone(),
                    parameter.values.dtype(),
                    parameter.values.bytes(),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut views = Vec::with_capacity(owned.len());
    for (name, shape, dtype, bytes) in &owned {
        let view = TensorView::new(dtype.safetensors_dtype(), shape.clone(), bytes)?;
        views.push((name.as_str(), view));
    }

    let metadata: Option<HashMap<String, String>> = None;
    safetensors::tensor::serialize(views, metadata).map_err(CheckpointError::from)
}

fn write_atomic_bytes(path: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|source| io_error(parent, source))?;

    let tmp_path = tmp_path_for(path);
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&tmp_path)
        .map_err(|source| io_error(&tmp_path, source))?;
    file.write_all(bytes)
        .map_err(|source| io_error(&tmp_path, source))?;
    file.sync_all()
        .map_err(|source| io_error(&tmp_path, source))?;
    drop(file);

    fs::rename(&tmp_path, path).map_err(|source| io_error(path, source))?;
    sync_parent_dir(path)
}

fn tmp_path_for(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .map_or_else(|| "checkpoint.tmp".to_owned(), |name| format!("{name}.tmp"));
    path.with_file_name(file_name)
}

fn sync_parent_dir(path: &Path) -> Result<(), CheckpointError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let directory = File::open(parent).map_err(|source| io_error(parent, source))?;
    directory
        .sync_all()
        .map_err(|source| io_error(parent, source))
}

fn remove_if_exists(path: &Path) -> Result<(), CheckpointError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(io_error(path, source)),
    }
}

fn step_from_sidecar_path(path: &Path) -> Option<u64> {
    let file_name = path.file_name()?.to_str()?;
    let step = file_name.strip_prefix("step_")?.strip_suffix(".json")?;
    if step.len() != 7 || !step.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    step.parse().ok()
}

fn element_count(shape: &[usize], name: &str) -> Result<usize, CheckpointError> {
    shape.iter().try_fold(1_usize, |accumulator, dim| {
        accumulator
            .checked_mul(*dim)
            .ok_or_else(|| CheckpointError::TensorShapeMismatch {
                name: name.to_owned(),
                expected: usize::MAX,
                found: 0,
            })
    })
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn hex_12(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(12);
    for byte in &bytes[..6] {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn io_error(path: impl Into<PathBuf>, source: std::io::Error) -> CheckpointError {
    CheckpointError::Io {
        path: path.into(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn checkpoint_roundtrip_burn() -> Result<(), Box<dyn std::error::Error>> {
        let dir = TestDir::new("roundtrip")?;
        let parameters = tiny_parameters()?;
        let parity = tiny_parity();
        let config_hash = config_hash_from_canonical_toml(b"[train]\nepochs = 1\n");
        let request = request(
            dir.path(),
            12,
            &config_hash,
            b"named-mpk-record",
            &parameters,
            &parity,
        );

        let paths = save_checkpoint(&request)?;
        let loaded = load_checkpoint(paths.sidecar)?;

        assert_eq!(loaded.burn_record, b"named-mpk-record");
        assert_eq!(loaded.sidecar.schema_version, CHECKPOINT_SCHEMA_VERSION);
        assert_eq!(loaded.sidecar.step, 12);
        assert_eq!(loaded.sidecar.epoch, 2);
        assert_eq!(loaded.sidecar.config_hash.len(), 12);
        assert_eq!(loaded.parity, parity);
        Ok(())
    }

    #[test]
    fn checkpoint_safetensors_mirror() -> Result<(), Box<dyn std::error::Error>> {
        let dir = TestDir::new("safetensors")?;
        let parameters = tiny_parameters()?;
        let parity = tiny_parity();
        let config_hash = config_hash_from_canonical_toml(b"seed = 7\n");
        let request = request(dir.path(), 7, &config_hash, b"record", &parameters, &parity);
        let paths = save_checkpoint(&request)?;

        let raw = fs::read(paths.model_safetensors)?;
        let tensors = SafeTensors::deserialize(&raw)?;
        let tensor = tensors.tensor("encoder.weight")?;

        assert_eq!(tensor.dtype(), Dtype::F32);
        assert_eq!(tensor.shape(), [2, 2]);
        assert_eq!(f32_values(tensor.data()), vec![1.0, 2.0, 3.0, 4.0]);
        let int_tensor = tensors.tensor("projector.norm.num_batches_tracked")?;
        assert_eq!(int_tensor.dtype(), Dtype::I64);
        assert_eq!(int_tensor.shape(), [1]);
        assert!(tensors.tensor("predictor.bias").is_ok());
        Ok(())
    }

    #[test]
    fn checkpoint_atomic_rename() -> Result<(), Box<dyn std::error::Error>> {
        let dir = TestDir::new("atomic")?;
        let path = dir.path().join("step_0000001.mpk");

        write_atomic_bytes(&path, b"first")?;
        write_atomic_bytes(&path, b"second")?;

        assert_eq!(fs::read(&path)?, b"second");
        assert!(!tmp_path_for(&path).exists());
        Ok(())
    }

    #[test]
    fn checkpoint_prune_keeps_three() -> Result<(), Box<dyn std::error::Error>> {
        let dir = TestDir::new("prune")?;
        let parameters = tiny_parameters()?;
        let parity = tiny_parity();
        let config_hash = config_hash_from_canonical_toml(b"seed = 11\n");

        for step in 1..=5 {
            let burn_record = [u8::try_from(step)?];
            let request = request(
                dir.path(),
                step,
                &config_hash,
                &burn_record,
                &parameters,
                &parity,
            );
            save_checkpoint(&request)?;
        }

        let checkpoints = list_complete_checkpoints(dir.path())?;
        let steps = checkpoints
            .iter()
            .filter_map(|paths| step_from_sidecar_path(&paths.sidecar))
            .collect::<Vec<_>>();

        assert_eq!(steps, vec![3, 4, 5]);
        assert!(!CheckpointPaths::for_step(dir.path(), 2).sidecar.exists());
        assert!(CheckpointPaths::for_step(dir.path(), 5).sidecar.exists());
        Ok(())
    }

    #[test]
    fn config_hash_uses_first_six_blake3_bytes_as_hex() {
        let hash = config_hash_from_canonical_toml(b"batch_size = 64\nepochs = 10\n");

        assert_eq!(hash.len(), 12);
        assert!(hash.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    fn request<'a>(
        output_dir: &'a Path,
        step: u64,
        config_hash: &'a str,
        burn_record: &'a [u8],
        parameters: &'a [ParameterTensor],
        parity: &'a ParityProbe,
    ) -> CheckpointWriteRequest<'a> {
        let mut metrics = BTreeMap::new();
        metrics.insert("loss/total".to_owned(), 0.0123);
        metrics.insert("optim/grad_norm_pre".to_owned(), 1.42);

        CheckpointWriteRequest {
            output_dir,
            run_id: "20260512-143002-9f3a-abcd",
            step,
            epoch: 2,
            wall_time_s: 42.5,
            git_short_sha: "9f3a8e2",
            config_hash,
            rng_state: CheckpointRngState {
                global_seed: 7,
                step_at_save: step,
                data_shuffle: "shuffle-state".to_owned(),
                sigreg_sketch: "sigreg-state".to_owned(),
                dropout: "dropout-state".to_owned(),
                cem: "cem-state".to_owned(),
                model_init: "model-init-state".to_owned(),
            },
            metrics_last_step: metrics,
            burn_record,
            parameters,
            parity,
        }
    }

    fn tiny_parameters() -> Result<Vec<ParameterTensor>, CheckpointError> {
        Ok(vec![
            ParameterTensor::f32("encoder.weight", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?,
            ParameterTensor::f32("predictor.bias", vec![2], vec![0.5, -0.5])?,
            ParameterTensor::i64("projector.norm.num_batches_tracked", vec![1], vec![7])?,
        ])
    }

    const fn tiny_parity() -> ParityProbe {
        ParityProbe {
            encoder_cls_l_inf: 7.2e-5,
            predictor_l_inf: 9.5e-5,
            sigreg_value: 0.00731,
        }
    }

    fn f32_values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(name: &str) -> Result<Self, Box<dyn std::error::Error>> {
            let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
            let path = std::env::temp_dir().join(format!(
                "lewm-train-checkpoint-{name}-{}-{}",
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
