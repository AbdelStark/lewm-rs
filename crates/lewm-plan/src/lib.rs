//! Planning and evaluation primitives, including `CEM` action search, `PushT`
//! planning evaluation, and `SO-100` latent trajectory metrics. This crate stays
//! separate from training and `Tract` inference runners; see [RFC 0006].
//!
//! [RFC 0006]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0006-planning-and-evaluation.md
//!
//! ## Module index
//!
//! - [`so100_eval`] implements the RFC 0006/0012 SO-100 latent-rollout metric
//!   contract.

pub mod errors;
pub mod so100_eval;

pub use crate::errors::EvalError;
pub use crate::so100_eval::{
    ActionVector, LatentVector, RecordedRolloutModel, So100Acceptance, So100Episode,
    So100EpisodeReport, So100EvalConfig, So100EvalReport, So100EvalRun, So100Evaluator,
    So100LatentTraceRow, So100Outcome, So100RolloutModel, WarmStartDelta, average_ranks,
    latent_mse_per_step, pairwise_distances, render_report_markdown, spearman_rank_correlation,
    trajectory_spearman, warm_start_delta, write_so100_outputs,
};
