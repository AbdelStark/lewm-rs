//! Planning and evaluation primitives, including `CEM` action search, `PushT`
//! planning evaluation, and `SO-100` latent trajectory metrics. This crate stays
//! separate from training and `Tract` inference runners; see [RFC 0006].
//!
//! [RFC 0006]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0006-planning-and-evaluation.md
//!
//! ## Module index
//!
//! - [`cem`] — Cross Entropy Method action search for normalized action
//!   sequences.
//! - [`pusht_eval`] — `PushT` eval loop and simulator RPC boundary.
//! - [`reports`] — eval artifact rendering and persistence.
//! - [`so100_eval`] — RFC 0006/0012 SO-100 latent-rollout metric contract.

pub mod cem;
pub mod errors;
pub mod pusht_eval;
pub mod reports;
pub mod so100_eval;

pub use crate::cem::{
    CEM_RNG_STREAM, Cem, CemCostModel, CemCostRequest, CemIterTrace, CemPlanInput, CemResult,
    DEFAULT_CEM_CHUNK_SIZE, DEFAULT_CEM_MAX_BATCH_BYTES,
};
pub use crate::errors::{EvalError, LewmPlanError};
pub use crate::pusht_eval::{
    EpisodeOutcome, EvalClock, MockPushtRpc, PushtCemConfig, PushtConfigFile, PushtEvalConfig,
    PushtEvalReport, PushtEvaluator, PushtObservation, PushtPlan, PushtPlanRequest, PushtPlanner,
    PushtRpc, StaticPushtPlanner, SubprocessPushtRpc, TrajectoryStep, TrajectorySummary, WallClock,
};
pub use crate::reports::{render_pusht_report, write_pusht_artifacts};
pub use crate::so100_eval::{
    ActionVector, LatentVector, RecordedRolloutModel, So100Acceptance, So100Episode,
    So100EpisodeReport, So100EvalConfig, So100EvalReport, So100EvalRun, So100Evaluator,
    So100LatentTraceRow, So100Outcome, So100RolloutModel, WarmStartDelta, average_ranks,
    latent_mse_per_step, pairwise_distances, render_report_markdown, spearman_rank_correlation,
    trajectory_spearman, warm_start_delta, write_so100_outputs,
};
