//! Resume detection and RNG restoration primitives for `lewm-train`.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

/// Name of the run identifier file in a training output directory.
pub const RUN_ID_FILE: &str = "run_id.txt";

/// Serialized `ChaCha20` state byte length: 32-byte seed + 16-byte word position.
pub const SERIALIZED_RNG_LEN: usize = 48;

/// Error returned by resume detection and restoration APIs.
#[derive(Debug)]
pub enum ResumeError {
    /// Filesystem operation failed.
    Io {
        /// Path involved in the failed operation.
        path: PathBuf,
        /// Original I/O error.
        source: std::io::Error,
    },
    /// JSON parsing failed.
    Json {
        /// Original JSON error.
        source: serde_json::Error,
    },
    /// A run directory already exists but resume mode was not enabled.
    RunDirOccupied {
        /// Output directory that already contains `run_id.txt`.
        output_dir: PathBuf,
    },
    /// The run identifier file exists but is empty.
    EmptyRunId {
        /// Path to `run_id.txt`.
        path: PathBuf,
    },
    /// Resume was requested but no complete sidecar was found.
    MissingCheckpoint {
        /// Output directory scanned for checkpoint sidecars.
        output_dir: PathBuf,
    },
    /// The latest sidecar points to a missing Burn record.
    MissingModelRecord {
        /// Expected Burn record path.
        path: PathBuf,
    },
    /// The run id in the sidecar does not match `run_id.txt`.
    RunIdMismatch {
        /// Run id from `run_id.txt`.
        expected: String,
        /// Run id from the checkpoint sidecar.
        found: String,
    },
    /// Base64 RNG state could not be decoded.
    InvalidBase64 {
        /// Reason the base64 payload was rejected.
        reason: String,
    },
    /// Serialized RNG bytes had the wrong length.
    InvalidRngState {
        /// Expected byte length.
        expected: usize,
        /// Actual byte length.
        found: usize,
    },
}

impl fmt::Display for ResumeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(
                    formatter,
                    "resume I/O error at {}: {source}",
                    path.display()
                )
            },
            Self::Json { source } => write!(formatter, "resume JSON error: {source}"),
            Self::RunDirOccupied { output_dir } => write!(
                formatter,
                "run directory already exists at {}; pass --resume-if-present to resume",
                output_dir.display()
            ),
            Self::EmptyRunId { path } => write!(formatter, "empty run id file: {}", path.display()),
            Self::MissingCheckpoint { output_dir } => write!(
                formatter,
                "no complete checkpoint sidecar found in {}",
                output_dir.display()
            ),
            Self::MissingModelRecord { path } => {
                write!(
                    formatter,
                    "checkpoint model record missing: {}",
                    path.display()
                )
            },
            Self::RunIdMismatch { expected, found } => {
                write!(
                    formatter,
                    "run id mismatch: expected {expected:?}, found {found:?}"
                )
            },
            Self::InvalidBase64 { reason } => {
                write!(formatter, "invalid base64 RNG state: {reason}")
            },
            Self::InvalidRngState { expected, found } => {
                write!(
                    formatter,
                    "invalid RNG state length: expected {expected}, found {found}"
                )
            },
        }
    }
}

impl std::error::Error for ResumeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source } => Some(source),
            Self::RunDirOccupied { .. }
            | Self::EmptyRunId { .. }
            | Self::MissingCheckpoint { .. }
            | Self::MissingModelRecord { .. }
            | Self::RunIdMismatch { .. }
            | Self::InvalidBase64 { .. }
            | Self::InvalidRngState { .. } => None,
        }
    }
}

impl From<serde_json::Error> for ResumeError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json { source }
    }
}

/// Startup decision for a training output directory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StartupMode {
    /// No existing run was detected; start fresh.
    Fresh,
    /// Resume from an existing checkpoint.
    Resume(Box<ResumePlan>),
}

/// Warning emitted when a sidecar SHA differs from the current binary SHA.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitShaWarning {
    /// Git SHA stored in the checkpoint sidecar.
    pub checkpoint_git_short_sha: String,
    /// Git SHA reported by the current binary.
    pub current_git_short_sha: String,
}

/// Training state selected after resume.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResumeState {
    /// Resume at the steady training loop.
    Steady,
}

/// Resume plan built from `run_id.txt` and the latest checkpoint sidecar.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResumePlan {
    /// Run identifier read from `run_id.txt`.
    pub run_id: String,
    /// Latest checkpoint step.
    pub step: u64,
    /// Latest checkpoint epoch.
    pub epoch: u64,
    /// Latest sidecar path.
    pub sidecar_path: PathBuf,
    /// Burn model/optimizer record path referenced by the sidecar.
    pub model_burn_path: PathBuf,
    /// Serialized RNG state from the sidecar.
    pub rng_state: ResumeRngState,
    /// Optional git-SHA mismatch warning.
    pub git_warning: Option<GitShaWarning>,
    /// State selected by RFC 0005 §7.2.
    pub resume_state: ResumeState,
}

/// Serialized RNG substream state from a checkpoint sidecar.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResumeRngState {
    /// Run-global seed.
    pub global_seed: u64,
    /// Optimizer step at save time.
    pub step_at_save: u64,
    /// Base64 `rng:data_shuffle` state.
    pub data_shuffle: String,
    /// Base64 `rng:sigreg_sketch` state.
    pub sigreg_sketch: String,
    /// Base64 `rng:dropout` state.
    pub dropout: String,
    /// Base64 `rng:cem` state.
    pub cem: String,
    /// Base64 `rng:model_init` state for auditing.
    pub model_init: String,
}

/// Restored RNG substreams.
#[derive(Clone, Debug)]
pub struct RestoredRngStreams {
    /// Restored data-shuffle RNG.
    pub data_shuffle: ChaCha20Rng,
    /// Restored `SIGReg` sketch RNG.
    pub sigreg_sketch: ChaCha20Rng,
    /// Restored dropout RNG.
    pub dropout: ChaCha20Rng,
    /// Restored `CEM` planner RNG.
    pub cem: ChaCha20Rng,
    /// Restored model-init RNG for audit comparison.
    pub model_init: ChaCha20Rng,
}

/// Shutdown signal handled by the trainer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShutdownSignal {
    /// `SIGTERM`.
    Sigterm,
    /// `SIGINT`.
    Sigint,
}

/// Result of a handled shutdown signal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignalExit {
    /// Signal that was handled.
    pub signal: ShutdownSignal,
    /// Step used for the emergency checkpoint.
    pub step: u64,
    /// Process exit code requested by RFC 0005.
    pub exit_code: i32,
    /// Whether the emergency checkpoint callback completed successfully.
    pub emergency_checkpoint_written: bool,
}

#[derive(Debug, Deserialize)]
struct ResumeSidecar {
    run_id: String,
    step: u64,
    epoch: u64,
    git_short_sha: String,
    rng_state: ResumeRngState,
    checkpoint_files: ResumeCheckpointFiles,
}

#[derive(Debug, Deserialize)]
struct ResumeCheckpointFiles {
    model_burn: String,
}

/// Detect whether an output directory should start fresh or resume.
///
/// # Errors
///
/// Returns [`ResumeError::RunDirOccupied`] when `run_id.txt` exists and
/// `resume_if_present` is false. Returns other errors when resume metadata is
/// missing, malformed, or incomplete.
pub fn detect_resume(
    output_dir: impl AsRef<Path>,
    resume_if_present: bool,
    current_git_short_sha: &str,
) -> Result<StartupMode, ResumeError> {
    let output_dir = output_dir.as_ref();
    let Some(run_id) = read_run_id(output_dir)? else {
        return Ok(StartupMode::Fresh);
    };

    if !resume_if_present {
        return Err(ResumeError::RunDirOccupied {
            output_dir: output_dir.to_path_buf(),
        });
    }

    let sidecar_path =
        latest_sidecar(output_dir)?.ok_or_else(|| ResumeError::MissingCheckpoint {
            output_dir: output_dir.to_path_buf(),
        })?;
    let sidecar = load_sidecar(&sidecar_path)?;
    if sidecar.run_id != run_id {
        return Err(ResumeError::RunIdMismatch {
            expected: run_id,
            found: sidecar.run_id,
        });
    }

    let model_burn_path = output_dir.join(&sidecar.checkpoint_files.model_burn);
    if !model_burn_path.is_file() {
        return Err(ResumeError::MissingModelRecord {
            path: model_burn_path,
        });
    }

    let git_warning = (sidecar.git_short_sha != current_git_short_sha).then(|| GitShaWarning {
        checkpoint_git_short_sha: sidecar.git_short_sha.clone(),
        current_git_short_sha: current_git_short_sha.to_owned(),
    });

    Ok(StartupMode::Resume(Box::new(ResumePlan {
        run_id: sidecar.run_id,
        step: sidecar.step,
        epoch: sidecar.epoch,
        sidecar_path,
        model_burn_path,
        rng_state: sidecar.rng_state,
        git_warning,
        resume_state: ResumeState::Steady,
    })))
}

/// Read `run_id.txt` from an output directory.
///
/// # Errors
///
/// Returns an error if the file exists but cannot be read or is empty.
pub fn read_run_id(output_dir: impl AsRef<Path>) -> Result<Option<String>, ResumeError> {
    let run_id_path = output_dir.as_ref().join(RUN_ID_FILE);
    match fs::read_to_string(&run_id_path) {
        Ok(raw) => {
            let run_id = raw.trim().to_owned();
            if run_id.is_empty() {
                Err(ResumeError::EmptyRunId { path: run_id_path })
            } else {
                Ok(Some(run_id))
            }
        },
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(io_error(run_id_path, source)),
    }
}

/// Restore all RNG substreams from sidecar strings.
///
/// # Errors
///
/// Returns an error if any base64 payload or serialized RNG state is invalid.
pub fn restore_rng_streams(state: &ResumeRngState) -> Result<RestoredRngStreams, ResumeError> {
    Ok(RestoredRngStreams {
        data_shuffle: decode_rng(&state.data_shuffle)?,
        sigreg_sketch: decode_rng(&state.sigreg_sketch)?,
        dropout: decode_rng(&state.dropout)?,
        cem: decode_rng(&state.cem)?,
        model_init: decode_rng(&state.model_init)?,
    })
}

/// Serialize a `ChaCha20Rng` as seed plus word position.
pub fn serialize_rng(rng: &ChaCha20Rng) -> [u8; SERIALIZED_RNG_LEN] {
    let mut bytes = [0_u8; SERIALIZED_RNG_LEN];
    bytes[..32].copy_from_slice(&rng.get_seed());
    bytes[32..].copy_from_slice(&rng.get_word_pos().to_le_bytes());
    bytes
}

/// Deserialize a `ChaCha20Rng` from seed plus word position.
///
/// # Errors
///
/// Returns [`ResumeError::InvalidRngState`] when `bytes` is not exactly
/// [`SERIALIZED_RNG_LEN`] bytes.
pub fn deserialize_rng(bytes: &[u8]) -> Result<ChaCha20Rng, ResumeError> {
    if bytes.len() != SERIALIZED_RNG_LEN {
        return Err(ResumeError::InvalidRngState {
            expected: SERIALIZED_RNG_LEN,
            found: bytes.len(),
        });
    }

    let mut seed = [0_u8; 32];
    seed.copy_from_slice(&bytes[..32]);
    let mut word_pos = [0_u8; 16];
    word_pos.copy_from_slice(&bytes[32..]);
    let mut rng = ChaCha20Rng::from_seed(seed);
    rng.set_word_pos(u128::from_le_bytes(word_pos));
    Ok(rng)
}

/// Encode a `ChaCha20Rng` as base64 sidecar text.
pub fn encode_rng(rng: &ChaCha20Rng) -> String {
    base64_encode(&serialize_rng(rng))
}

/// Decode a base64 sidecar RNG string.
///
/// # Errors
///
/// Returns an error if base64 decoding fails or the decoded RNG state has the
/// wrong byte length.
pub fn decode_rng(encoded: &str) -> Result<ChaCha20Rng, ResumeError> {
    let bytes = base64_decode(encoded)?;
    deserialize_rng(&bytes)
}

/// Handle a shutdown signal by writing an emergency checkpoint and returning
/// the RFC 0005 exit-code contract.
///
/// # Errors
///
/// Returns an error if `write_emergency_checkpoint` fails.
pub fn handle_shutdown_signal<F>(
    signal: ShutdownSignal,
    step: u64,
    write_emergency_checkpoint: F,
) -> Result<SignalExit, ResumeError>
where
    F: FnOnce(u64, ShutdownSignal) -> Result<(), ResumeError>,
{
    write_emergency_checkpoint(step, signal)?;
    Ok(SignalExit {
        signal,
        step,
        exit_code: 0,
        emergency_checkpoint_written: true,
    })
}

fn load_sidecar(path: &Path) -> Result<ResumeSidecar, ResumeError> {
    let raw = fs::read(path).map_err(|source| io_error(path, source))?;
    serde_json::from_slice(&raw).map_err(ResumeError::from)
}

fn latest_sidecar(output_dir: &Path) -> Result<Option<PathBuf>, ResumeError> {
    let entries = match fs::read_dir(output_dir) {
        Ok(entries) => entries,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(io_error(output_dir, source)),
    };

    let mut candidates = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|source| io_error(output_dir, source))?;
        let path = entry.path();
        if let Some(step) = step_from_sidecar_path(&path) {
            candidates.push((step, path));
        }
    }
    candidates.sort_by_key(|(step, _path)| *step);
    Ok(candidates.pop().map(|(_step, path)| path))
}

fn step_from_sidecar_path(path: &Path) -> Option<u64> {
    let file_name = path.file_name()?.to_str()?;
    let step = file_name.strip_prefix("step_")?.strip_suffix(".json")?;
    if step.len() != 7 || !step.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    step.parse().ok()
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);

    for chunk in bytes.chunks(3) {
        let first = chunk[0];
        let second = chunk.get(1).copied().unwrap_or(0);
        let third = chunk.get(2).copied().unwrap_or(0);

        encoded.push(char::from(TABLE[usize::from(first >> 2)]));
        encoded.push(char::from(
            TABLE[usize::from(((first & 0x03) << 4) | (second >> 4))],
        ));
        if chunk.len() > 1 {
            encoded.push(char::from(
                TABLE[usize::from(((second & 0x0f) << 2) | (third >> 6))],
            ));
        } else {
            encoded.push('=');
        }
        if chunk.len() > 2 {
            encoded.push(char::from(TABLE[usize::from(third & 0x3f)]));
        } else {
            encoded.push('=');
        }
    }

    encoded
}

fn base64_decode(encoded: &str) -> Result<Vec<u8>, ResumeError> {
    let bytes = encoded.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err(ResumeError::InvalidBase64 {
            reason: "length must be a multiple of 4".to_owned(),
        });
    }

    let mut decoded = Vec::with_capacity((bytes.len() / 4) * 3);
    for chunk in bytes.chunks(4) {
        let first = decode_base64_char(chunk[0])?;
        let second = decode_base64_char(chunk[1])?;
        let third = if chunk[2] == b'=' {
            None
        } else {
            Some(decode_base64_char(chunk[2])?)
        };
        let fourth = if chunk[3] == b'=' {
            None
        } else {
            Some(decode_base64_char(chunk[3])?)
        };

        if third.is_none() && fourth.is_some() {
            return Err(ResumeError::InvalidBase64 {
                reason: "invalid padding order".to_owned(),
            });
        }

        decoded.push((first << 2) | (second >> 4));
        if let Some(third) = third {
            decoded.push(((second & 0x0f) << 4) | (third >> 2));
            if let Some(fourth) = fourth {
                decoded.push(((third & 0x03) << 6) | fourth);
            }
        }
    }

    Ok(decoded)
}

fn decode_base64_char(byte: u8) -> Result<u8, ResumeError> {
    match byte {
        b'A'..=b'Z' => Ok(byte - b'A'),
        b'a'..=b'z' => Ok(byte - b'a' + 26),
        b'0'..=b'9' => Ok(byte - b'0' + 52),
        b'+' => Ok(62),
        b'/' => Ok(63),
        _ => Err(ResumeError::InvalidBase64 {
            reason: format!("invalid byte 0x{byte:02x}"),
        }),
    }
}

fn io_error(path: impl Into<PathBuf>, source: std::io::Error) -> ResumeError {
    ResumeError::Io {
        path: path.into(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn run_dir_without_resume_flag_is_occupied() -> Result<(), Box<dyn std::error::Error>> {
        let dir = TestDir::new("occupied")?;
        fs::write(dir.path().join(RUN_ID_FILE), "run-1\n")?;

        let err = detect_resume(dir.path(), false, "abc123").err();

        assert!(matches!(err, Some(ResumeError::RunDirOccupied { .. })));
        Ok(())
    }

    #[test]
    fn resume_rng_bitwise_identical() -> Result<(), Box<dyn std::error::Error>> {
        let mut original = ChaCha20Rng::from_seed([7_u8; 32]);
        for _ in 0..17 {
            let _ = original.next_u64();
        }
        let encoded = encode_rng(&original);
        let expected_next = original.next_u64();

        let mut restored = decode_rng(&encoded)?;

        assert_eq!(restored.next_u64(), expected_next);
        Ok(())
    }

    #[test]
    fn resume_detection_loads_latest_sidecar_and_git_warning()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = TestDir::new("resume")?;
        fs::write(dir.path().join(RUN_ID_FILE), "run-2\n")?;
        write_sidecar(dir.path(), "run-2", 1, "oldsha")?;
        write_sidecar(dir.path(), "run-2", 9, "oldsha")?;
        fs::write(dir.path().join("step_0000001.mpk"), b"old")?;
        fs::write(dir.path().join("step_0000009.mpk"), b"new")?;

        let mode = detect_resume(dir.path(), true, "newsha")?;
        let StartupMode::Resume(plan) = mode else {
            return Err("expected resume mode".into());
        };

        assert_eq!(plan.run_id, "run-2");
        assert_eq!(plan.step, 9);
        assert_eq!(plan.epoch, 3);
        assert_eq!(plan.resume_state, ResumeState::Steady);
        assert_eq!(
            plan.git_warning,
            Some(GitShaWarning {
                checkpoint_git_short_sha: "oldsha".to_owned(),
                current_git_short_sha: "newsha".to_owned(),
            })
        );
        let mut streams = restore_rng_streams(&plan.rng_state)?;
        assert_eq!(streams.data_shuffle.next_u64(), expected_rng_next(9));
        Ok(())
    }

    #[test]
    fn resume_via_sigterm_simulation() -> Result<(), Box<dyn std::error::Error>> {
        let dir = TestDir::new("sigterm")?;
        let marker = dir.path().join("emergency_step_0000042.json");

        let exit = handle_shutdown_signal(ShutdownSignal::Sigterm, 42, |step, signal| {
            let body = format!("{{\"step\":{step},\"signal\":\"{signal:?}\"}}\n");
            fs::write(&marker, body).map_err(|source| io_error(&marker, source))
        })?;

        assert_eq!(exit.exit_code, 0);
        assert!(exit.emergency_checkpoint_written);
        assert_eq!(
            fs::read_to_string(marker)?,
            "{\"step\":42,\"signal\":\"Sigterm\"}\n"
        );
        Ok(())
    }

    fn write_sidecar(
        output_dir: &Path,
        run_id: &str,
        step: u64,
        git_short_sha: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let rng = rng_after_step(step);
        let encoded = encode_rng(&rng);
        let sidecar = serde_json::json!({
            "run_id": run_id,
            "step": step,
            "epoch": 3,
            "git_short_sha": git_short_sha,
            "rng_state": {
                "global_seed": 7,
                "step_at_save": step,
                "data_shuffle": encoded,
                "sigreg_sketch": encoded,
                "dropout": encoded,
                "cem": encoded,
                "model_init": encoded
            },
            "checkpoint_files": {
                "model_burn": step_file_name(step, "mpk")
            }
        });
        let path = output_dir.join(step_file_name(step, "json"));
        fs::write(path, serde_json::to_vec_pretty(&sidecar)?)?;
        Ok(())
    }

    fn rng_after_step(step: u64) -> ChaCha20Rng {
        let mut rng = ChaCha20Rng::from_seed([7_u8; 32]);
        for _ in 0..step {
            let _ = rng.next_u64();
        }
        rng
    }

    fn expected_rng_next(step: u64) -> u64 {
        let mut rng = rng_after_step(step);
        rng.next_u64()
    }

    fn step_file_name(step: u64, extension: &str) -> String {
        format!("step_{step:07}.{extension}")
    }

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(name: &str) -> Result<Self, Box<dyn std::error::Error>> {
            let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
            let path = std::env::temp_dir().join(format!(
                "lewm-train-resume-{name}-{}-{}",
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
