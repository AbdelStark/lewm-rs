//! `Hugging Face Hub` integration boundaries for model uploads, model-card
//! rendering, artifact manifests, and cost-ledger checks. This crate owns
//! publishing mechanics, not credentials or human billing controls; see
//! [RFC 0010].
//!
//! [RFC 0010]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0010-huggingface-hub-integration.md
//!
//! ## Module index
//!
//! - [`model_card`] — model repository README rendering.

pub mod model_card;

pub use crate::model_card::{
    LEWM_CITATION_BIBTEX, LEWM_RS_CITATION_BIBTEX, ModelCardError, ModelCardMetadata, render,
};
