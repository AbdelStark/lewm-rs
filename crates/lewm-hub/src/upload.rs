//! Idempotent upload helpers and retry policy.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use sha2::{Digest, Sha256};
use walkdir::WalkDir;

use crate::HubError;

/// Result status for one file upload decision.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum UploadStatus {
    /// Remote content already matched the local SHA-256 digest.
    Skipped,
    /// Local content was uploaded.
    Uploaded,
}

/// Aggregate result for file or folder upload operations.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct UploadResult {
    /// Final status for single-file uploads, or `Uploaded` for folders that uploaded at least one file.
    pub status: UploadStatus,
    /// Number of files considered.
    pub files_considered: usize,
    /// Number of files uploaded.
    pub files_uploaded: usize,
    /// Number of files skipped because they were already current.
    pub files_skipped: usize,
    /// Bytes uploaded in this call.
    pub bytes_uploaded: u64,
}

impl UploadResult {
    /// Build a result for one skipped file.
    pub fn skipped() -> Self {
        Self {
            status: UploadStatus::Skipped,
            files_considered: 1,
            files_uploaded: 0,
            files_skipped: 1,
            bytes_uploaded: 0,
        }
    }

    /// Build a result for one uploaded file.
    pub fn uploaded(bytes_uploaded: u64) -> Self {
        Self {
            status: UploadStatus::Uploaded,
            files_considered: 1,
            files_uploaded: 1,
            files_skipped: 0,
            bytes_uploaded,
        }
    }

    pub(crate) fn empty_folder() -> Self {
        Self {
            status: UploadStatus::Skipped,
            files_considered: 0,
            files_uploaded: 0,
            files_skipped: 0,
            bytes_uploaded: 0,
        }
    }

    pub(crate) fn merge(&mut self, other: Self) {
        self.files_considered += other.files_considered;
        self.files_uploaded += other.files_uploaded;
        self.files_skipped += other.files_skipped;
        self.bytes_uploaded += other.bytes_uploaded;
        if self.files_uploaded > 0 {
            self.status = UploadStatus::Uploaded;
        }
    }
}

/// Retry policy for Hub writes.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct RetryPolicy {
    /// Maximum HTTP 5xx or 429 retries.
    pub max_http_retries: usize,
    /// Maximum network-error retries.
    pub max_network_retries: usize,
    /// Initial backoff delay.
    pub base_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_http_retries: 5,
            max_network_retries: 3,
            base_delay: Duration::from_secs(1),
        }
    }
}

impl RetryPolicy {
    /// Policy variant used by tests to avoid sleeping.
    pub fn no_delay() -> Self {
        Self {
            base_delay: Duration::ZERO,
            ..Self::default()
        }
    }
}

/// Run a fallible Hub operation under the retry policy.
///
/// # Errors
///
/// Returns the final operation error when retry budget is exhausted, or returns
/// fail-fast for non-retryable errors.
pub fn with_backoff<T>(
    policy: RetryPolicy,
    mut operation: impl FnMut() -> Result<T, HubError>,
) -> Result<T, HubError> {
    let mut http_retries = 0_usize;
    let mut network_retries = 0_usize;
    loop {
        match operation() {
            Ok(value) => return Ok(value),
            Err(error) if error.is_retryable() => {
                let next_attempt = match error {
                    HubError::HttpStatus { .. } => {
                        if http_retries >= policy.max_http_retries {
                            None
                        } else {
                            http_retries += 1;
                            Some(http_retries)
                        }
                    },
                    HubError::Network(_) => {
                        if network_retries >= policy.max_network_retries {
                            None
                        } else {
                            network_retries += 1;
                            Some(network_retries)
                        }
                    },
                    HubError::TokenMissing
                    | HubError::Auth(_)
                    | HubError::Io { .. }
                    | HubError::Json(_)
                    | HubError::InvalidPath(_)
                    | HubError::Unsupported(_) => None,
                };
                let Some(attempt) = next_attempt else {
                    return Err(error);
                };
                sleep_before_retry(policy, &error, attempt);
            },
            Err(error) => return Err(error),
        }
    }
}

/// Retry module matching the RFC naming.
pub mod retry {
    pub use super::with_backoff;
}

fn sleep_before_retry(policy: RetryPolicy, error: &HubError, attempt: usize) {
    let delay = match error {
        HubError::Network(_) => policy.base_delay,
        HubError::HttpStatus { .. } => error
            .retry_after()
            .unwrap_or_else(|| exponential_delay(policy.base_delay, attempt)),
        HubError::TokenMissing
        | HubError::Auth(_)
        | HubError::Io { .. }
        | HubError::Json(_)
        | HubError::InvalidPath(_)
        | HubError::Unsupported(_) => Duration::ZERO,
    };
    if !delay.is_zero() {
        thread::sleep(delay);
    }
}

fn exponential_delay(base_delay: Duration, attempt: usize) -> Duration {
    let shift = u32::try_from(attempt.saturating_sub(1).min(31)).unwrap_or(31);
    let factor = 1_u32.checked_shl(shift).unwrap_or(1);
    base_delay.saturating_mul(factor)
}

/// Compute a file's SHA-256 digest as lowercase hex.
///
/// # Errors
///
/// Returns an error when the file cannot be opened or read.
pub fn sha256_file(path: impl AsRef<Path>) -> Result<String, HubError> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|source| HubError::io(path, source))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| HubError::io(path, source))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(lower_hex(&hasher.finalize()))
}

fn lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

pub(crate) fn folder_files(root: &Path) -> Result<Vec<PathBuf>, HubError> {
    if !root.is_dir() {
        return Err(HubError::InvalidPath(format!(
            "upload folder root is not a directory: {}",
            root.display()
        )));
    }

    let mut files = Vec::new();
    for entry in WalkDir::new(root).follow_links(false) {
        let entry = entry.map_err(|source| HubError::InvalidPath(source.to_string()))?;
        if entry.file_type().is_file() {
            files.push(entry.path().to_path_buf());
        }
    }
    files.sort();
    Ok(files)
}

pub(crate) fn remote_join(prefix: &str, relative: &Path) -> Result<String, HubError> {
    let relative = relative
        .to_str()
        .ok_or_else(|| HubError::InvalidPath("remote path must be UTF-8".to_owned()))?
        .replace('\\', "/");
    if prefix.is_empty() {
        validate_remote_path(&relative)?;
        return Ok(relative);
    }
    let joined = format!("{}/{}", prefix.trim_matches('/'), relative);
    validate_remote_path(&joined)?;
    Ok(joined)
}

pub(crate) fn validate_remote_path(remote: &str) -> Result<(), HubError> {
    if remote.is_empty() {
        return Err(HubError::InvalidPath(
            "remote path must not be empty".to_owned(),
        ));
    }
    if remote.starts_with('/') || remote.contains("..") {
        return Err(HubError::InvalidPath(format!(
            "remote path must be relative and must not contain '..': {remote}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_file_matches_known_digest() -> Result<(), Box<dyn std::error::Error>> {
        let dir = unique_temp_dir()?;
        let path = dir.join("payload.txt");
        std::fs::write(&path, b"hello")?;

        let digest = sha256_file(&path)?;

        assert_eq!(
            digest,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    fn unique_temp_dir() -> Result<PathBuf, std::io::Error> {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("lewm-hub-upload-test-{suffix}"));
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }
}
