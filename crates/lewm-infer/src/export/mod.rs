//! Export contract for the `Tract` deployment path.
//!
//! The concrete Burn model is implemented outside this crate boundary. This
//! module owns the stable ONNX graph contract that a model-specific exporter
//! must satisfy: two graphs, opset 18, dynamic batch/history axes, and explicit
//! lowering of activation ops that have spotty runtime coverage.

pub mod lowering;
pub mod onnx;

pub use crate::export::lowering::{ActivationLowering, ActivationLoweringSet};
pub use crate::export::onnx::{
    AxisDim, DynamicAxis, ExportedOnnxGraph, GraphExportRequest, GraphKind, OnnxExportArtifacts,
    OnnxExportSpec, OnnxGraphSpec, OnnxModelExporter, TensorDType, TensorSpec, export_onnx_pair,
    write_export_metadata,
};
