//! SO-100 warm-start checkpoint transfer contract.
//!
//! This module owns the record-level warm-start policy from RFC 0012. The
//! device-backed `Jepa` loader can call this boundary once concrete Burn record
//! serialization lands in the training crate.

use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use lewm_core::JepaConfig;
use sha2::{Digest, Sha256};

/// SO-100 action dimensionality pinned by RFC 0012.
pub const SO100_ACTION_DIM: usize = 6;

/// Model module prefixes copied from the `PushT` checkpoint into SO-100.
pub const TRANSFER_MODULE_PREFIXES: [&str; 4] =
    ["encoder.", "predictor.", "projector.", "pred_proj."];

const ACTION_ENCODER_PREFIX: &str = "action_encoder.";
const SHA256_BUFFER_BYTES: usize = 64 * 1024;

/// Error type surfaced by warm-start loading.
#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    /// The SO-100 target config is not a valid warm-start target.
    #[error("invalid warm-start config: {0}")]
    InvalidConfig(String),

    /// A tensor record has incoherent shape metadata.
    #[error(
        "invalid tensor record: shape {shape:?} requires {expected_len} values, found {found_len}"
    )]
    InvalidTensorRecord {
        /// Tensor shape.
        shape: Vec<usize>,
        /// Flat values required by the shape.
        expected_len: usize,
        /// Flat values present in the record.
        found_len: usize,
    },

    /// The `PushT` checkpoint did not contain any transferable module parameters.
    #[error("warm-start checkpoint has no encoder, predictor, projector, or pred_proj parameters")]
    NoTransferableParameters,

    /// The freshly initialized SO-100 model has no action encoder parameters to preserve.
    #[error("fresh SO-100 model record has no action_encoder parameters to preserve")]
    MissingActionEncoder,

    /// The warm-start source bytes could not be hashed.
    #[error("could not hash warm-start source at {path}: {source}")]
    SourceHash {
        /// Path being hashed.
        path: PathBuf,
        /// Filesystem error from the hashing pass.
        #[source]
        source: std::io::Error,
    },
}

/// Minimal named tensor record used by the warm-start transfer boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorRecord {
    /// Tensor shape in row-major order.
    pub shape: Vec<usize>,
    /// Flat tensor values.
    pub values: Vec<f32>,
}

impl TensorRecord {
    /// Build a tensor record and validate that `shape` matches `values`.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape product overflows or does not match the
    /// flat value count.
    pub fn new(shape: Vec<usize>, values: Vec<f32>) -> Result<Self, TrainError> {
        let expected_len = shape.iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(*dim)
                .ok_or_else(|| TrainError::InvalidTensorRecord {
                    shape: shape.clone(),
                    expected_len: usize::MAX,
                    found_len: values.len(),
                })
        })?;
        if expected_len != values.len() {
            return Err(TrainError::InvalidTensorRecord {
                shape,
                expected_len,
                found_len: values.len(),
            });
        }
        Ok(Self { shape, values })
    }
}

/// Record-level training state used by warm-start transfer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainStateRecord {
    /// Named model tensors.
    pub model: BTreeMap<String, TensorRecord>,
    /// Named optimizer-state tensors. These are intentionally dropped during
    /// warm-start.
    pub optimizer_state: BTreeMap<String, TensorRecord>,
}

impl TrainStateRecord {
    /// Insert a named model tensor.
    pub fn insert_model_param(
        &mut self,
        name: impl Into<String>,
        tensor: TensorRecord,
    ) -> Option<TensorRecord> {
        self.model.insert(name.into(), tensor)
    }

    /// Insert a named optimizer-state tensor.
    pub fn insert_optimizer_state(
        &mut self,
        name: impl Into<String>,
        tensor: TensorRecord,
    ) -> Option<TensorRecord> {
        self.optimizer_state.insert(name.into(), tensor)
    }
}

/// Provenance emitted after applying a warm-start transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarmstartProvenance {
    /// Source checkpoint path used for the warm-start.
    pub source_path: PathBuf,
    /// SHA-256 digest of the source checkpoint bytes as lowercase hex.
    pub source_sha256: String,
    /// Model parameters copied verbatim from the `PushT` checkpoint.
    pub transferred_parameters: Vec<String>,
    /// SO-100 action-encoder parameters intentionally preserved from the fresh target.
    pub preserved_action_encoder_parameters: Vec<String>,
    /// Number of optimizer-state entries discarded to guarantee fresh `AdamW` state.
    pub dropped_optimizer_state_entries: usize,
}

/// Result of applying a warm-start transfer.
#[derive(Debug, Clone, PartialEq)]
pub struct WarmstartLoad {
    /// Warm-started model record with fresh optimizer state.
    pub state: TrainStateRecord,
    /// Provenance for the trainer preamble.
    pub provenance: WarmstartProvenance,
}

/// Apply the SO-100 warm-start transfer policy to record-level model state.
///
/// The target record must come from a freshly initialized SO-100 model. This
/// function overwrites only `encoder.*`, `predictor.*`, `projector.*`, and
/// `pred_proj.*` from the `PushT` checkpoint, leaves `action_encoder.*` untouched,
/// and drops all optimizer state.
///
/// # Errors
///
/// Returns an error when the target config is not SO-100-shaped, no transferable
/// parameters are present, the fresh target has no action encoder parameters, or
/// the checkpoint source cannot be hashed.
pub fn load_warmstart(
    so100_config: &JepaConfig,
    mut initialized_so100: TrainStateRecord,
    pusht_checkpoint: &TrainStateRecord,
    source_path: impl AsRef<Path>,
) -> Result<WarmstartLoad, TrainError> {
    validate_so100_target(so100_config)?;

    let source_path = source_path.as_ref();
    let source_sha256 = sha256_file_hex(source_path)?;

    let preserved_action_encoder_parameters = initialized_so100
        .model
        .keys()
        .filter(|name| name.starts_with(ACTION_ENCODER_PREFIX))
        .cloned()
        .collect::<Vec<_>>();
    if preserved_action_encoder_parameters.is_empty() {
        return Err(TrainError::MissingActionEncoder);
    }

    let mut transferred_parameters = Vec::new();
    for prefix in TRANSFER_MODULE_PREFIXES {
        for (name, tensor) in &pusht_checkpoint.model {
            if !name.starts_with(prefix) {
                continue;
            }
            initialized_so100.model.insert(name.clone(), tensor.clone());
            transferred_parameters.push(name.clone());
        }
    }
    if transferred_parameters.is_empty() {
        return Err(TrainError::NoTransferableParameters);
    }

    let dropped_optimizer_state_entries = initialized_so100.optimizer_state.len();
    initialized_so100.optimizer_state.clear();

    Ok(WarmstartLoad {
        state: initialized_so100,
        provenance: WarmstartProvenance {
            source_path: source_path.to_path_buf(),
            source_sha256,
            transferred_parameters,
            preserved_action_encoder_parameters,
            dropped_optimizer_state_entries,
        },
    })
}

fn validate_so100_target(config: &JepaConfig) -> Result<(), TrainError> {
    if let Err(errors) = config.validate_shape_contract() {
        return Err(TrainError::InvalidConfig(errors.join("; ")));
    }
    if config.action_encoder.input_dim != SO100_ACTION_DIM {
        return Err(TrainError::InvalidConfig(format!(
            "SO-100 warm-start requires action_encoder.input_dim {SO100_ACTION_DIM}, found {}",
            config.action_encoder.input_dim
        )));
    }
    Ok(())
}

fn sha256_file_hex(path: &Path) -> Result<String, TrainError> {
    let mut file = File::open(path).map_err(|source| TrainError::SourceHash {
        path: path.to_path_buf(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; SHA256_BUFFER_BYTES];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| TrainError::SourceHash {
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(hex_lower(&hasher.finalize()))
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    out
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use lewm_core::JepaConfig;

    use super::*;

    const CHECKPOINT_SHA256: &str =
        "5971668c84c3ab63515fdd54723e1b52bab0bfd3c757099a9abc3bb144a3e279";

    #[test]
    fn warmstart_copies_encoder() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let source = write_source_checkpoint(dir.path())?;
        let outcome = load_warmstart(
            &so100_config(),
            fresh_so100_record()?,
            &pusht_record()?,
            &source,
        )?;

        assert_eq!(
            outcome.state.model.get("encoder.block.weight"),
            pusht_record()?.model.get("encoder.block.weight")
        );
        assert_eq!(
            outcome.state.model.get("predictor.block.weight"),
            pusht_record()?.model.get("predictor.block.weight")
        );
        assert_eq!(
            outcome.state.model.get("projector.bn.running_mean"),
            pusht_record()?.model.get("projector.bn.running_mean")
        );
        assert_eq!(
            outcome.state.model.get("pred_proj.bn.running_var"),
            pusht_record()?.model.get("pred_proj.bn.running_var")
        );
        assert_eq!(outcome.provenance.source_sha256, CHECKPOINT_SHA256);
        assert_eq!(
            outcome.provenance.transferred_parameters,
            vec![
                "encoder.block.weight",
                "predictor.block.weight",
                "projector.bn.running_mean",
                "pred_proj.bn.running_var",
            ]
        );
        Ok(())
    }

    #[test]
    fn warmstart_reinits_action_encoder() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let source = write_source_checkpoint(dir.path())?;
        let fresh = fresh_so100_record()?;
        let outcome = load_warmstart(&so100_config(), fresh.clone(), &pusht_record()?, &source)?;

        assert_eq!(
            outcome.state.model.get("action_encoder.weight"),
            fresh.model.get("action_encoder.weight")
        );
        assert_ne!(
            outcome.state.model.get("action_encoder.weight"),
            pusht_record()?.model.get("action_encoder.weight")
        );
        assert_eq!(
            outcome.provenance.preserved_action_encoder_parameters,
            vec!["action_encoder.weight"]
        );
        Ok(())
    }

    #[test]
    fn warmstart_no_optim_state_transfer() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let source = write_source_checkpoint(dir.path())?;
        let outcome = load_warmstart(
            &so100_config(),
            fresh_so100_record()?,
            &pusht_record()?,
            &source,
        )?;

        assert!(outcome.state.optimizer_state.is_empty());
        assert_eq!(outcome.provenance.dropped_optimizer_state_entries, 1);
        Ok(())
    }

    fn so100_config() -> JepaConfig {
        let mut config = JepaConfig::default();
        config.action_encoder.input_dim = SO100_ACTION_DIM;
        config
    }

    fn fresh_so100_record() -> Result<TrainStateRecord, TrainError> {
        let mut record = TrainStateRecord::default();
        record.insert_model_param(
            "encoder.block.weight",
            TensorRecord::new(vec![2], vec![100.0, 101.0])?,
        );
        record.insert_model_param(
            "predictor.block.weight",
            TensorRecord::new(vec![2], vec![200.0, 201.0])?,
        );
        record.insert_model_param(
            "projector.bn.running_mean",
            TensorRecord::new(vec![2], vec![300.0, 301.0])?,
        );
        record.insert_model_param(
            "pred_proj.bn.running_var",
            TensorRecord::new(vec![2], vec![400.0, 401.0])?,
        );
        record.insert_model_param(
            "action_encoder.weight",
            TensorRecord::new(vec![3], vec![600.0, 601.0, 602.0])?,
        );
        record.insert_optimizer_state(
            "adamw.exp_avg.encoder.block.weight",
            TensorRecord::new(vec![2], vec![0.1, 0.2])?,
        );
        Ok(record)
    }

    fn pusht_record() -> Result<TrainStateRecord, TrainError> {
        let mut record = TrainStateRecord::default();
        record.insert_model_param(
            "encoder.block.weight",
            TensorRecord::new(vec![2], vec![1.0, 2.0])?,
        );
        record.insert_model_param(
            "predictor.block.weight",
            TensorRecord::new(vec![2], vec![3.0, 4.0])?,
        );
        record.insert_model_param(
            "projector.bn.running_mean",
            TensorRecord::new(vec![2], vec![5.0, 6.0])?,
        );
        record.insert_model_param(
            "pred_proj.bn.running_var",
            TensorRecord::new(vec![2], vec![7.0, 8.0])?,
        );
        record.insert_model_param(
            "action_encoder.weight",
            TensorRecord::new(vec![2], vec![9.0, 10.0])?,
        );
        record.insert_optimizer_state(
            "adamw.exp_avg.encoder.block.weight",
            TensorRecord::new(vec![2], vec![9.0, 9.0])?,
        );
        Ok(record)
    }

    fn write_source_checkpoint(dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let path = dir.join("tiny_pusht_ckpt.mpk");
        fs::write(&path, b"checkpoint bytes\n")?;
        Ok(path)
    }
}
