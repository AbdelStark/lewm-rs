//! Hub client and transport-backed upload pipeline.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use serde_json::{Value, json};

use crate::HubError;
use crate::upload::{
    RetryPolicy, UploadResult, folder_files, remote_join, sha256_file, validate_remote_path,
    with_backoff,
};

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
const DEFAULT_NAMESPACE: &str = "abdelstark";
const DEFAULT_REVISION: &str = "main";
const USER_AGENT: &str = concat!("lewm-rs/", env!("CARGO_PKG_VERSION"));

/// Hub repository kind.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum RepoKind {
    /// Hugging Face model repository.
    Model,
    /// Hugging Face dataset repository.
    Dataset,
    /// Hugging Face Space repository.
    Space,
}

impl RepoKind {
    fn api_type(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Dataset => "dataset",
            Self::Space => "space",
        }
    }

    fn api_plural(self) -> &'static str {
        match self {
            Self::Model => "models",
            Self::Dataset => "datasets",
            Self::Space => "spaces",
        }
    }
}

/// Handle for a repository on the Hub.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RepoHandle {
    /// Full repo id, for example `abdelstark/lewm-rs-pusht`.
    pub repo_id: String,
    /// Repo kind.
    pub kind: RepoKind,
}

impl RepoHandle {
    /// Build a repo handle.
    ///
    /// # Errors
    ///
    /// Returns an error when the repo id is empty or malformed.
    pub fn new(repo_id: impl Into<String>, kind: RepoKind) -> Result<Self, HubError> {
        let repo_id = repo_id.into();
        validate_repo_id(&repo_id)?;
        Ok(Self { repo_id, kind })
    }
}

/// Metadata for a remote file.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RemoteFile {
    /// Remote path inside the repository.
    pub path: String,
    /// SHA-256 digest if the Hub metadata exposed one.
    pub sha256: Option<String>,
    /// Remote file size, if known.
    pub size_bytes: Option<u64>,
}

/// Transport request for creating or reusing a repo.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EnsureRepoRequest {
    /// Namespace or organization.
    pub namespace: String,
    /// Short repo name.
    pub name: String,
    /// Repo kind.
    pub kind: RepoKind,
    /// Whether the repo should be private.
    pub private: bool,
}

/// Abstraction over live or mocked Hub operations.
pub trait HubTransport {
    /// Resolve the authenticated user.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid credentials or transport failures.
    fn whoami(&mut self, token: &str) -> Result<String, HubError>;

    /// Create the repo if needed and return a handle.
    ///
    /// # Errors
    ///
    /// Returns an error for non-idempotent repo creation failures.
    fn ensure_repo(
        &mut self,
        token: &str,
        request: &EnsureRepoRequest,
    ) -> Result<RepoHandle, HubError>;

    /// Return remote metadata for a file, or `None` when it does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error for transport failures other than a missing file.
    fn file_metadata(
        &mut self,
        token: &str,
        repo: &RepoHandle,
        remote: &str,
    ) -> Result<Option<RemoteFile>, HubError>;

    /// Upload one file to a remote path.
    ///
    /// # Errors
    ///
    /// Returns an error when the upload fails.
    fn upload_file(
        &mut self,
        token: &str,
        repo: &RepoHandle,
        local: &Path,
        remote: &str,
        commit_message: &str,
        sha256: &str,
    ) -> Result<(), HubError>;

    /// Delete one remote file.
    ///
    /// # Errors
    ///
    /// Returns an error when the delete operation fails.
    fn delete_file(
        &mut self,
        token: &str,
        repo: &RepoHandle,
        remote: &str,
        commit_message: &str,
    ) -> Result<(), HubError>;
}

/// HTTP transport used by [`HubClient::from_env`].
#[derive(Debug, Clone)]
pub struct EnvironmentHubTransport {
    endpoint: String,
    agent: ureq::Agent,
}

impl Default for EnvironmentHubTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl EnvironmentHubTransport {
    /// Build a transport using `HF_ENDPOINT`, or the public Hub endpoint.
    pub fn new() -> Self {
        Self::with_endpoint(
            std::env::var("HF_ENDPOINT").unwrap_or_else(|_| DEFAULT_ENDPOINT.to_owned()),
        )
    }

    /// Build a transport for a specific Hub-compatible endpoint.
    pub fn with_endpoint(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into().trim_end_matches('/').to_owned(),
            agent: ureq::AgentBuilder::new().build(),
        }
    }
}

impl HubTransport for EnvironmentHubTransport {
    fn whoami(&mut self, token: &str) -> Result<String, HubError> {
        if token.trim().is_empty() {
            return Err(HubError::Auth("HF_TOKEN was empty".to_owned()));
        }
        let value = self.get_json(token, "/api/whoami-v2")?;
        value
            .get("name")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .ok_or_else(|| HubError::Auth("whoami response did not include a user name".to_owned()))
    }

    fn ensure_repo(
        &mut self,
        token: &str,
        request: &EnsureRepoRequest,
    ) -> Result<RepoHandle, HubError> {
        if request.kind == RepoKind::Space {
            return Err(HubError::Unsupported(
                "Space creation needs an explicit SDK and is not part of this client surface"
                    .to_owned(),
            ));
        }

        let payload = json!({
            "name": request.name,
            "organization": request.namespace,
            "private": request.private,
            "type": request.kind.api_type(),
        });
        match self.post_json(token, "/api/repos/create", &payload) {
            Ok(_) | Err(HubError::HttpStatus { status: 409, .. }) => RepoHandle::new(
                format!("{}/{}", request.namespace, request.name),
                request.kind,
            ),
            Err(error) => Err(error),
        }
    }

    fn file_metadata(
        &mut self,
        token: &str,
        repo: &RepoHandle,
        remote: &str,
    ) -> Result<Option<RemoteFile>, HubError> {
        let path = format!("/api/{}/{}", repo.kind.api_plural(), repo.repo_id);
        let response = self
            .agent
            .get(&self.url(&path))
            .query("blobs", "true")
            .set("Authorization", &format!("Bearer {token}"))
            .set("User-Agent", USER_AGENT)
            .call()
            .map_err(map_ureq_error);
        let value = match response {
            Ok(response) => response_json(response)?,
            Err(HubError::HttpStatus { status: 404, .. }) => return Ok(None),
            Err(error) => return Err(error),
        };
        Ok(find_remote_file(&value, remote))
    }

    fn upload_file(
        &mut self,
        token: &str,
        repo: &RepoHandle,
        local: &Path,
        remote: &str,
        commit_message: &str,
        _sha256: &str,
    ) -> Result<(), HubError> {
        let mode = self.fetch_upload_mode(token, repo, local, remote)?;
        if mode == "lfs" {
            return Err(HubError::Unsupported(
                "live Rust transport supports regular Hub commits; LFS payloads must use the Python upload sidecar"
                    .to_owned(),
            ));
        }

        let content = base64_file(local)?;
        let payload = ndjson([
            json!({
                "key": "header",
                "value": {
                    "summary": commit_message,
                    "description": "",
                },
            }),
            json!({
                "key": "file",
                "value": {
                    "content": content,
                    "path": remote,
                    "encoding": "base64",
                },
            }),
        ])?;
        self.post_ndjson(
            token,
            &format!(
                "/api/{}/{}/commit/{}",
                repo.kind.api_plural(),
                repo.repo_id,
                DEFAULT_REVISION
            ),
            &payload,
        )
    }

    fn delete_file(
        &mut self,
        token: &str,
        repo: &RepoHandle,
        remote: &str,
        commit_message: &str,
    ) -> Result<(), HubError> {
        let payload = ndjson([
            json!({
                "key": "header",
                "value": {
                    "summary": commit_message,
                    "description": "",
                },
            }),
            json!({
                "key": "deletedFile",
                "value": { "path": remote },
            }),
        ])?;
        self.post_ndjson(
            token,
            &format!(
                "/api/{}/{}/commit/{}",
                repo.kind.api_plural(),
                repo.repo_id,
                DEFAULT_REVISION
            ),
            &payload,
        )
    }
}

impl EnvironmentHubTransport {
    fn url(&self, path: &str) -> String {
        format!("{}{}", self.endpoint, path)
    }

    fn get_json(&self, token: &str, path: &str) -> Result<Value, HubError> {
        let response = self
            .agent
            .get(&self.url(path))
            .set("Authorization", &format!("Bearer {token}"))
            .set("User-Agent", USER_AGENT)
            .call()
            .map_err(map_ureq_error)?;
        response_json(response)
    }

    fn post_json(&self, token: &str, path: &str, payload: &Value) -> Result<Value, HubError> {
        let response = self
            .agent
            .post(&self.url(path))
            .set("Authorization", &format!("Bearer {token}"))
            .set("User-Agent", USER_AGENT)
            .send_json(payload)
            .map_err(map_ureq_error)?;
        response_json(response)
    }

    fn post_ndjson(&self, token: &str, path: &str, payload: &str) -> Result<(), HubError> {
        self.agent
            .post(&self.url(path))
            .set("Authorization", &format!("Bearer {token}"))
            .set("User-Agent", USER_AGENT)
            .set("Content-Type", "application/x-ndjson")
            .send_string(payload)
            .map_err(map_ureq_error)?;
        Ok(())
    }

    fn fetch_upload_mode(
        &self,
        token: &str,
        repo: &RepoHandle,
        local: &Path,
        remote: &str,
    ) -> Result<String, HubError> {
        let sample = sample_file(local)?;
        let size = std::fs::metadata(local)
            .map_err(|source| HubError::io(local, source))?
            .len();
        if size == 0 {
            return Ok("regular".to_owned());
        }
        let payload = json!({
            "files": [{
                "path": remote,
                "sample": BASE64_STANDARD.encode(sample),
                "size": size,
            }],
        });
        let value = self.post_json(
            token,
            &format!(
                "/api/{}/{}/preupload/{}",
                repo.kind.api_plural(),
                repo.repo_id,
                DEFAULT_REVISION
            ),
            &payload,
        )?;
        value
            .get("files")
            .and_then(Value::as_array)
            .and_then(|files| files.first())
            .and_then(|file| file.get("uploadMode"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .ok_or_else(|| {
                HubError::Network("Hub preupload response did not include uploadMode".to_owned())
            })
    }
}

fn response_json(response: ureq::Response) -> Result<Value, HubError> {
    let body = response
        .into_string()
        .map_err(|source| HubError::Network(format!("could not read Hub response: {source}")))?;
    Ok(serde_json::from_str(&body)?)
}

fn map_ureq_error(error: ureq::Error) -> HubError {
    match error {
        ureq::Error::Status(status, response) => {
            let retry_after = response.header("Retry-After").and_then(parse_retry_after);
            let message = response
                .into_string()
                .unwrap_or_else(|source| format!("could not read error response: {source}"));
            HubError::HttpStatus {
                status,
                retry_after,
                message,
            }
        },
        ureq::Error::Transport(error) => HubError::Network(error.to_string()),
    }
}

fn parse_retry_after(value: &str) -> Option<Duration> {
    value.trim().parse::<u64>().ok().map(Duration::from_secs)
}

fn find_remote_file(value: &Value, remote: &str) -> Option<RemoteFile> {
    value
        .get("siblings")?
        .as_array()?
        .iter()
        .find(|file| file.get("rfilename").and_then(Value::as_str) == Some(remote))
        .map(|file| RemoteFile {
            path: remote.to_owned(),
            sha256: file
                .get("lfs")
                .and_then(|lfs| lfs.get("sha256").or_else(|| lfs.get("oid")))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            size_bytes: file.get("size").and_then(Value::as_u64).or_else(|| {
                file.get("lfs")
                    .and_then(|lfs| lfs.get("size"))
                    .and_then(Value::as_u64)
            }),
        })
}

fn base64_file(path: &Path) -> Result<String, HubError> {
    let mut file = File::open(path).map_err(|source| HubError::io(path, source))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|source| HubError::io(path, source))?;
    Ok(BASE64_STANDARD.encode(bytes))
}

fn sample_file(path: &Path) -> Result<Vec<u8>, HubError> {
    let mut file = File::open(path).map_err(|source| HubError::io(path, source))?;
    let mut sample = vec![0_u8; 512];
    let read = file
        .read(&mut sample)
        .map_err(|source| HubError::io(path, source))?;
    sample.truncate(read);
    Ok(sample)
}

fn ndjson<const N: usize>(items: [Value; N]) -> Result<String, HubError> {
    let mut payload = String::new();
    for item in items {
        payload.push_str(&serde_json::to_string(&item)?);
        payload.push('\n');
    }
    Ok(payload)
}

/// Authenticated Hub client.
#[derive(Debug, Clone)]
pub struct HubClient<T = EnvironmentHubTransport> {
    transport: T,
    user: String,
    namespace: String,
    token: String,
    retry_policy: RetryPolicy,
}

impl HubClient<EnvironmentHubTransport> {
    /// Build a client from `HF_TOKEN`.
    ///
    /// # Errors
    ///
    /// Returns an error when `HF_TOKEN` is missing or authentication validation
    /// fails.
    pub fn from_env() -> Result<Self, HubError> {
        Self::from_env_with_transport(EnvironmentHubTransport::new())
    }
}

impl<T> HubClient<T>
where
    T: HubTransport,
{
    /// Build a client from `HF_TOKEN` using a supplied transport.
    ///
    /// # Errors
    ///
    /// Returns an error when `HF_TOKEN` is missing or transport authentication
    /// validation fails.
    pub fn from_env_with_transport(transport: T) -> Result<Self, HubError> {
        Self::from_env_with(transport, |key| std::env::var(key).ok())
    }

    /// Build a client from an injectable environment reader.
    ///
    /// # Errors
    ///
    /// Returns an error when `HF_TOKEN` is missing or transport authentication
    /// validation fails.
    pub fn from_env_with(
        mut transport: T,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, HubError> {
        let token = env("HF_TOKEN").ok_or(HubError::TokenMissing)?;
        let namespace = env("HF_NAMESPACE").unwrap_or_else(|| DEFAULT_NAMESPACE.to_owned());
        let user = transport.whoami(&token)?;
        Ok(Self {
            transport,
            user,
            namespace,
            token,
            retry_policy: RetryPolicy::default(),
        })
    }

    /// Build a client from explicit pieces.
    ///
    /// # Errors
    ///
    /// Returns an error when the token is empty or authentication validation
    /// fails.
    pub fn from_token(
        mut transport: T,
        token: impl Into<String>,
        namespace: impl Into<String>,
    ) -> Result<Self, HubError> {
        let token = token.into();
        if token.trim().is_empty() {
            return Err(HubError::TokenMissing);
        }
        let namespace = namespace.into();
        let user = transport.whoami(&token)?;
        Ok(Self {
            transport,
            user,
            namespace,
            token,
            retry_policy: RetryPolicy::default(),
        })
    }

    /// Override retry policy.
    #[must_use]
    pub fn with_retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    /// Authenticated Hub username.
    pub fn user(&self) -> &str {
        &self.user
    }

    /// Default upload namespace.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Access the transport for test inspection.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Ensure a repository exists.
    ///
    /// # Errors
    ///
    /// Returns an error when the repo name is invalid or the transport fails.
    pub fn ensure_repo(
        &mut self,
        name: &str,
        kind: RepoKind,
        private: bool,
    ) -> Result<RepoHandle, HubError> {
        validate_repo_name(name)?;
        let request = EnsureRepoRequest {
            namespace: self.namespace.clone(),
            name: name.to_owned(),
            kind,
            private,
        };
        let repo_id = format!("{}/{}", request.namespace, request.name);
        with_backoff(self.retry_policy, || {
            match self.transport.ensure_repo(&self.token, &request) {
                Ok(handle) => Ok(handle),
                Err(HubError::HttpStatus { status: 409, .. }) => RepoHandle::new(&repo_id, kind),
                Err(error) => Err(error),
            }
        })
    }

    /// Upload a single file if the remote SHA-256 differs.
    ///
    /// # Errors
    ///
    /// Returns an error when the local file cannot be read, remote metadata
    /// lookup fails, or upload retries are exhausted.
    pub fn upload_file(
        &mut self,
        repo: &RepoHandle,
        local: impl AsRef<Path>,
        remote: &str,
        commit_message: &str,
    ) -> Result<UploadResult, HubError> {
        validate_remote_path(remote)?;
        validate_commit_message(commit_message)?;
        let local = local.as_ref();
        if !local.is_file() {
            return Err(HubError::InvalidPath(format!(
                "upload source is not a file: {}",
                local.display()
            )));
        }
        let sha256 = sha256_file(local)?;
        if let Some(remote_meta) = with_backoff(self.retry_policy, || {
            self.transport.file_metadata(&self.token, repo, remote)
        })? && remote_meta.sha256.as_deref() == Some(sha256.as_str())
        {
            return Ok(UploadResult::skipped());
        }

        let bytes = std::fs::metadata(local)
            .map_err(|source| HubError::io(local, source))?
            .len();
        with_backoff(self.retry_policy, || {
            self.transport
                .upload_file(&self.token, repo, local, remote, commit_message, &sha256)
        })?;
        Ok(UploadResult::uploaded(bytes))
    }

    /// Upload a folder file-by-file with SHA-256 resume behavior.
    ///
    /// # Errors
    ///
    /// Returns an error when directory walking, local hashing, metadata lookup,
    /// or any upload fails after retries.
    pub fn upload_folder(
        &mut self,
        repo: &RepoHandle,
        local_dir: impl AsRef<Path>,
        remote_prefix: &str,
        commit_message: &str,
    ) -> Result<UploadResult, HubError> {
        validate_commit_message(commit_message)?;
        let local_dir = local_dir.as_ref();
        let mut aggregate = UploadResult::empty_folder();
        for path in folder_files(local_dir)? {
            let relative = path.strip_prefix(local_dir).map_err(|source| {
                HubError::InvalidPath(format!(
                    "could not compute relative path for {}: {source}",
                    path.display()
                ))
            })?;
            let remote = remote_join(remote_prefix, relative)?;
            let result = self.upload_file(repo, &path, &remote, commit_message)?;
            aggregate.merge(result);
        }
        Ok(aggregate)
    }

    /// Delete a remote file.
    ///
    /// # Errors
    ///
    /// Returns an error when the path or commit message is invalid, or delete
    /// retries are exhausted.
    pub fn delete_file(
        &mut self,
        repo: &RepoHandle,
        remote: &str,
        commit_message: &str,
    ) -> Result<(), HubError> {
        validate_remote_path(remote)?;
        validate_commit_message(commit_message)?;
        with_backoff(self.retry_policy, || {
            self.transport
                .delete_file(&self.token, repo, remote, commit_message)
        })
    }

    /// Redacted token marker for debugging.
    pub fn token_is_configured(&self) -> bool {
        !self.token.is_empty()
    }
}

fn validate_repo_id(repo_id: &str) -> Result<(), HubError> {
    let parts = repo_id.split('/').collect::<Vec<_>>();
    if parts.len() != 2 || parts.iter().any(|part| part.is_empty()) {
        return Err(HubError::InvalidPath(format!(
            "repo id must be namespace/name, found {repo_id}"
        )));
    }
    Ok(())
}

fn validate_repo_name(name: &str) -> Result<(), HubError> {
    if name.is_empty() || name.contains('/') || name.contains("..") {
        return Err(HubError::InvalidPath(format!(
            "repo name must be a single path segment, found {name}"
        )));
    }
    Ok(())
}

fn validate_commit_message(commit_message: &str) -> Result<(), HubError> {
    if commit_message.trim().is_empty() {
        return Err(HubError::InvalidPath(
            "commit message must not be empty".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::collections::VecDeque;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::upload::UploadStatus;

    #[derive(Debug, Default)]
    struct MockTransport {
        user: String,
        whoami_calls: usize,
        upload_calls: Vec<String>,
        delete_calls: Vec<String>,
        remote: HashMap<String, RemoteFile>,
        upload_failures: HashMap<String, VecDeque<HubError>>,
        ensure_repo_calls: usize,
    }

    impl MockTransport {
        fn with_user(user: &str) -> Self {
            Self {
                user: user.to_owned(),
                ..Self::default()
            }
        }

        fn remote_key(repo: &RepoHandle, remote: &str) -> String {
            format!("{}:{remote}", repo.repo_id)
        }

        fn fail_upload_once(&mut self, repo: &RepoHandle, remote: &str, error: HubError) {
            self.upload_failures
                .entry(Self::remote_key(repo, remote))
                .or_default()
                .push_back(error);
        }
    }

    impl HubTransport for MockTransport {
        fn whoami(&mut self, token: &str) -> Result<String, HubError> {
            self.whoami_calls += 1;
            if token == "bad" {
                return Err(HubError::Auth("bad token".to_owned()));
            }
            Ok(if self.user.is_empty() {
                "tester".to_owned()
            } else {
                self.user.clone()
            })
        }

        fn ensure_repo(
            &mut self,
            _token: &str,
            request: &EnsureRepoRequest,
        ) -> Result<RepoHandle, HubError> {
            self.ensure_repo_calls += 1;
            RepoHandle::new(
                format!("{}/{}", request.namespace, request.name),
                request.kind,
            )
        }

        fn file_metadata(
            &mut self,
            _token: &str,
            repo: &RepoHandle,
            remote: &str,
        ) -> Result<Option<RemoteFile>, HubError> {
            Ok(self.remote.get(&Self::remote_key(repo, remote)).cloned())
        }

        fn upload_file(
            &mut self,
            _token: &str,
            repo: &RepoHandle,
            local: &Path,
            remote: &str,
            _commit_message: &str,
            sha256: &str,
        ) -> Result<(), HubError> {
            let key = Self::remote_key(repo, remote);
            if let Some(queue) = self.upload_failures.get_mut(&key)
                && let Some(error) = queue.pop_front()
            {
                return Err(error);
            }
            self.upload_calls.push(remote.to_owned());
            let size_bytes = std::fs::metadata(local)
                .map_err(|source| HubError::io(local, source))?
                .len();
            self.remote.insert(
                key,
                RemoteFile {
                    path: remote.to_owned(),
                    sha256: Some(sha256.to_owned()),
                    size_bytes: Some(size_bytes),
                },
            );
            Ok(())
        }

        fn delete_file(
            &mut self,
            _token: &str,
            repo: &RepoHandle,
            remote: &str,
            _commit_message: &str,
        ) -> Result<(), HubError> {
            self.delete_calls.push(remote.to_owned());
            self.remote.remove(&Self::remote_key(repo, remote));
            Ok(())
        }
    }

    #[test]
    fn client_from_env_fails_without_token() {
        let result = HubClient::from_env_with(MockTransport::default(), |_| None);

        assert!(matches!(result, Err(HubError::TokenMissing)));
    }

    #[test]
    fn client_from_env_calls_whoami() -> Result<(), Box<dyn std::error::Error>> {
        let client =
            HubClient::from_env_with(MockTransport::with_user("abdel"), |key| match key {
                "HF_TOKEN" => Some("token".to_owned()),
                "HF_NAMESPACE" => Some("abdelstark".to_owned()),
                _ => None,
            })?;

        assert_eq!(client.user(), "abdel");
        assert_eq!(client.namespace(), "abdelstark");
        assert_eq!(client.transport().whoami_calls, 1);
        assert!(client.token_is_configured());
        Ok(())
    }

    #[test]
    fn ensure_repo_returns_handle() -> Result<(), Box<dyn std::error::Error>> {
        let mut client = test_client(MockTransport::default())?;
        let repo = client.ensure_repo("lewm-rs-test", RepoKind::Model, false)?;

        assert_eq!(repo.repo_id, "abdelstark/lewm-rs-test");
        assert_eq!(repo.kind, RepoKind::Model);
        assert_eq!(client.transport().ensure_repo_calls, 1);
        Ok(())
    }

    #[test]
    fn upload_idempotent_on_same_file() -> Result<(), Box<dyn std::error::Error>> {
        let dir = unique_temp_dir()?;
        let local = dir.join("weights.safetensors");
        std::fs::write(&local, b"same")?;
        let repo = RepoHandle::new("abdelstark/lewm-rs-test", RepoKind::Model)?;
        let sha256 = sha256_file(&local)?;
        let mut transport = MockTransport::default();
        transport.remote.insert(
            MockTransport::remote_key(&repo, "weights.safetensors"),
            RemoteFile {
                path: "weights.safetensors".to_owned(),
                sha256: Some(sha256),
                size_bytes: Some(4),
            },
        );
        let mut client = test_client(transport)?;

        let result =
            client.upload_file(&repo, &local, "weights.safetensors", "run: upload file")?;

        assert_eq!(result.status, UploadStatus::Skipped);
        assert!(client.transport().upload_calls.is_empty());
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn upload_retry_on_5xx() -> Result<(), Box<dyn std::error::Error>> {
        let dir = unique_temp_dir()?;
        let local = dir.join("weights.safetensors");
        std::fs::write(&local, b"payload")?;
        let repo = RepoHandle::new("abdelstark/lewm-rs-test", RepoKind::Model)?;
        let mut transport = MockTransport::default();
        transport.fail_upload_once(
            &repo,
            "weights.safetensors",
            HubError::HttpStatus {
                status: 500,
                retry_after: None,
                message: "temporary".to_owned(),
            },
        );
        let mut client = test_client(transport)?;

        let result =
            client.upload_file(&repo, &local, "weights.safetensors", "run: upload file")?;

        assert_eq!(result.status, UploadStatus::Uploaded);
        assert_eq!(client.transport().upload_calls, vec!["weights.safetensors"]);
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn folder_upload_resume_after_crash() -> Result<(), Box<dyn std::error::Error>> {
        let dir = unique_temp_dir()?;
        let folder = dir.join("folder");
        std::fs::create_dir_all(&folder)?;
        let first = folder.join("a.txt");
        let second = folder.join("b.txt");
        std::fs::write(&first, b"a")?;
        std::fs::write(&second, b"b")?;
        let repo = RepoHandle::new("abdelstark/lewm-rs-test", RepoKind::Model)?;
        let mut transport = MockTransport::default();
        transport.fail_upload_once(
            &repo,
            "artifacts/b.txt",
            HubError::Unsupported("simulated process exit".to_owned()),
        );
        let mut client = test_client(transport)?;

        let first_attempt = client.upload_folder(&repo, &folder, "artifacts", "run: upload folder");
        assert!(first_attempt.is_err());

        let result = client.upload_folder(&repo, &folder, "artifacts", "run: upload folder")?;

        assert_eq!(result.files_considered, 2);
        assert_eq!(result.files_skipped, 1);
        assert_eq!(result.files_uploaded, 1);
        assert_eq!(
            client.transport().upload_calls,
            vec!["artifacts/a.txt", "artifacts/b.txt"]
        );
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn delete_file_uses_transport() -> Result<(), Box<dyn std::error::Error>> {
        let repo = RepoHandle::new("abdelstark/lewm-rs-test", RepoKind::Model)?;
        let mut client = test_client(MockTransport::default())?;

        client.delete_file(&repo, "old.bin", "run: delete file")?;

        assert_eq!(client.transport().delete_calls, vec!["old.bin"]);
        Ok(())
    }

    fn test_client(transport: MockTransport) -> Result<HubClient<MockTransport>, HubError> {
        HubClient::from_token(transport, "token", "abdelstark")
            .map(|client| client.with_retry_policy(RetryPolicy::no_delay()))
    }

    fn unique_temp_dir() -> Result<PathBuf, std::io::Error> {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("lewm-hub-client-test-{suffix}"));
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }
}
