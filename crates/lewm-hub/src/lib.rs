//! `Hugging Face Hub` integration boundaries for model uploads, model-card
//! rendering, artifact manifests, and cost-ledger checks. This crate owns
//! publishing mechanics, not credentials or human billing controls; see
//! [RFC 0010].
//!
//! [RFC 0010]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0010-huggingface-hub-integration.md
//!
//! ## Module index
//!
//! - [`client`] — authenticated client and transport-backed repo/upload APIs.
//! - [`upload`] — SHA-256 idempotency and retry helpers.

pub mod client;
pub mod errors;
pub mod upload;

pub use crate::client::{
    EnsureRepoRequest, EnvironmentHubTransport, HubClient, HubTransport, RemoteFile, RepoHandle,
    RepoKind,
};
pub use crate::errors::HubError;
pub use crate::upload::{RetryPolicy, UploadResult, UploadStatus, sha256_file, with_backoff};
