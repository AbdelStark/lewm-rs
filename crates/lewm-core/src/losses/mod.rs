//! Loss and training-diagnostic helpers for `lewm-core`.

pub mod collapse;
pub mod prediction;
pub mod sigreg;

pub use collapse::{
    CLS_COSINE_PAIR_CEILING, CLS_MEAN_ABS_CEILING, CLS_VAR_FLOOR, CollapseProbe,
    CollapseProbeResult, CollapseThresholds, CollapseTrip, run_collapse_probe,
    run_collapse_probe_with_thresholds,
};
pub use prediction::prediction_loss;
pub use sigreg::{
    DEFAULT_SIGREG_KNOTS, DEFAULT_SIGREG_NUM_PROJ, DEFAULT_SIGREG_T_MAX, SigReg, SigRegConsts,
    sample_sigreg_projection,
};
