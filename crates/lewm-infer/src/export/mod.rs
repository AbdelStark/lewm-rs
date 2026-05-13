//! Export contract for the `Tract` deployment path.
//!
//! The concrete Burn model is implemented outside this crate boundary. This
//! module owns the stable ONNX graph contract that a model-specific exporter
//! must satisfy: two graphs, opset 18, dynamic batch/history axes, and explicit
//! lowering of activation ops that have spotty runtime coverage. It also owns
//! the export verification and fallback-selection contract for deployable
//! inference graphs.

pub mod lowering;
pub mod onnx;
pub mod verifier;

pub use crate::export::lowering::{ActivationLowering, ActivationLoweringSet};
pub use crate::export::onnx::{
    AxisDim, DynamicAxis, ExportedOnnxGraph, GraphExportRequest, GraphKind, OnnxExportArtifacts,
    OnnxExportSpec, OnnxGraphSpec, OnnxModelExporter, TensorDType, TensorSpec, export_onnx_pair,
    write_export_metadata,
};
pub use crate::export::verifier::{
    BurnDirectPolicy, BurnForward, DEFAULT_L_INF_TOLERANCE, ExportDecision, ExportStrategy,
    FixedInput, InferenceForward, VerificationAttempt, VerificationAttemptStatus,
    VerificationReport, VerifierError, pick_export_strategy, render_model_card_decision, verify,
    verify_with_tolerance,
};
