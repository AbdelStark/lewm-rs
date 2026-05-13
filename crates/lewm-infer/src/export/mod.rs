//! Export verification and fallback selection for deployable inference graphs.

pub mod verifier;

pub use crate::export::verifier::{
    BurnDirectPolicy, BurnForward, DEFAULT_L_INF_TOLERANCE, ExportDecision, ExportStrategy,
    FixedInput, InferenceForward, VerificationAttempt, VerificationAttemptStatus,
    VerificationReport, VerifierError, pick_export_strategy, render_model_card_decision, verify,
    verify_with_tolerance,
};
