//! Loss and training-diagnostic helpers for `lewm-core`.

pub mod collapse;
pub mod prediction;

pub use collapse::{
    CLS_COSINE_PAIR_CEILING, CLS_MEAN_ABS_CEILING, CLS_VAR_FLOOR, CollapseProbe,
    CollapseProbeResult, CollapseThresholds, CollapseTrip, run_collapse_probe,
    run_collapse_probe_with_thresholds,
};
pub use prediction::prediction_loss;
