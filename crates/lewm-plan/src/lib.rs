//! Planning and evaluation primitives, including `CEM` action search, `PushT`
//! planning evaluation, and `SO-100` latent trajectory metrics. This crate stays
//! separate from training and `Tract` inference runners; see [RFC 0006].
//!
//! [RFC 0006]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0006-planning-and-evaluation.md
//!
//! ## Module index
//!
//! - [`pusht_eval`] — `PushT` eval loop and simulator RPC boundary.
//! - [`reports`] — eval artifact rendering and persistence.

pub mod errors;
pub mod pusht_eval;
pub mod reports;

pub use crate::errors::EvalError;
pub use crate::pusht_eval::{
    EpisodeOutcome, MockPushtRpc, PushtCemConfig, PushtConfigFile, PushtEvalConfig,
    PushtEvalReport, PushtEvaluator, PushtObservation, PushtPlan, PushtPlanRequest, PushtPlanner,
    PushtRpc, StaticPushtPlanner, SubprocessPushtRpc, TrajectoryStep, TrajectorySummary,
};
pub use crate::reports::{render_pusht_report, write_pusht_artifacts};
