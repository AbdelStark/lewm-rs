//! `Hugging Face Hub` integration boundaries for model uploads, model-card
//! rendering, artifact manifests, and cost-ledger checks. This crate owns
//! publishing mechanics, not credentials or human billing controls; see
//! [RFC 0010].
//!
//! [RFC 0010]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0010-huggingface-hub-integration.md
//!
//! ## Module index
//!
//! - [`cost_ledger`] — RFC 0010 cost ledger parsing, append, and cap checks.

pub mod cost_ledger;

pub use crate::cost_ledger::{
    CostEntry, CostLedgerError, CostLedgerRow, HARD_CAP_USD_CENTS, UsdAmount, append_entry,
    cost_for_wall_time, read_ledger, rounded_billable_minutes, verify_ledger,
};
