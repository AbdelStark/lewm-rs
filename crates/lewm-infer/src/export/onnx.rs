//! ONNX export graph contract for the `Tract` runtime.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use lewm_core::JepaConfig;
use serde::{Deserialize, Serialize};

use crate::errors::{InferError, InferResult};
use crate::export::lowering::ActivationLoweringSet;

/// RFC 0007 ONNX opset version.
pub const ONNX_OPSET_VERSION: u32 = 18;

/// Required encoder ONNX filename.
pub const ENCODER_ONNX_FILENAME: &str = "encoder.onnx";

/// Required predictor ONNX filename.
pub const PREDICTOR_ONNX_FILENAME: &str = "predictor.onnx";

/// Metadata sidecar written beside exported ONNX graphs.
pub const ONNX_EXPORT_METADATA_FILENAME: &str = "onnx_export.json";

/// ONNX tensor element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorDType {
    /// 32-bit floating-point tensor.
    F32,
}

impl fmt::Display for TensorDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => f.write_str("f32"),
        }
    }
}

/// A tensor axis with either a fixed size or a dynamic symbolic name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum AxisDim {
    /// Fixed positive dimension.
    Fixed(usize),
    /// Dynamic dimension with a stable ONNX symbolic name.
    Dynamic(String),
}

impl AxisDim {
    /// Return a dynamic axis.
    pub fn dynamic(name: impl Into<String>) -> Self {
        Self::Dynamic(name.into())
    }

    /// Return a fixed axis.
    pub const fn fixed(value: usize) -> Self {
        Self::Fixed(value)
    }

    /// Return the dynamic-axis name, if this dimension is dynamic.
    pub fn dynamic_name(&self) -> Option<&str> {
        match self {
            Self::Fixed(_) => None,
            Self::Dynamic(name) => Some(name),
        }
    }
}

/// Named dynamic ONNX axis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DynamicAxis {
    /// Zero-based axis index.
    pub axis: usize,
    /// Symbolic dimension name.
    pub name: String,
}

/// ONNX input or output tensor specification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Tensor name in the exported graph.
    pub name: String,
    /// Tensor element type.
    pub dtype: TensorDType,
    /// Tensor shape.
    pub shape: Vec<AxisDim>,
}

impl TensorSpec {
    /// Construct a tensor spec.
    pub fn new(name: impl Into<String>, dtype: TensorDType, shape: Vec<AxisDim>) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape,
        }
    }

    /// Return all dynamic axes in this tensor spec.
    pub fn dynamic_axes(&self) -> Vec<DynamicAxis> {
        self.shape
            .iter()
            .enumerate()
            .filter_map(|(axis, dim)| {
                dim.dynamic_name().map(|name| DynamicAxis {
                    axis,
                    name: name.to_owned(),
                })
            })
            .collect()
    }
}

/// Exported graph kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphKind {
    /// Encoder graph: pixels to latent embedding.
    Encoder,
    /// Predictor graph: latent history and actions to next latent embedding.
    Predictor,
}

impl GraphKind {
    /// Return the stable graph filename.
    pub const fn filename(self) -> &'static str {
        match self {
            Self::Encoder => ENCODER_ONNX_FILENAME,
            Self::Predictor => PREDICTOR_ONNX_FILENAME,
        }
    }

    /// Return the stable graph display name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Encoder => "encoder",
            Self::Predictor => "predictor",
        }
    }
}

/// Contract for one ONNX graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnnxGraphSpec {
    /// Graph kind.
    pub kind: GraphKind,
    /// Target opset version.
    pub opset_version: u32,
    /// Input tensors.
    pub inputs: Vec<TensorSpec>,
    /// Output tensors.
    pub outputs: Vec<TensorSpec>,
    /// Activation lowerings required while exporting this graph.
    pub activation_lowerings: ActivationLoweringSet,
}

impl OnnxGraphSpec {
    /// Construct the encoder graph spec from a `JEPA` config.
    pub fn encoder(config: &JepaConfig) -> Self {
        Self {
            kind: GraphKind::Encoder,
            opset_version: ONNX_OPSET_VERSION,
            inputs: vec![TensorSpec::new(
                "pixels",
                TensorDType::F32,
                vec![
                    AxisDim::dynamic("batch"),
                    AxisDim::fixed(config.encoder.num_channels),
                    AxisDim::fixed(config.encoder.image_size),
                    AxisDim::fixed(config.encoder.image_size),
                ],
            )],
            outputs: vec![TensorSpec::new(
                "embedding",
                TensorDType::F32,
                vec![
                    AxisDim::dynamic("batch"),
                    AxisDim::fixed(config.projector.output_dim),
                ],
            )],
            activation_lowerings: ActivationLoweringSet::rfc0007(),
        }
    }

    /// Construct the predictor graph spec from a `JEPA` config.
    pub fn predictor(config: &JepaConfig) -> Self {
        Self {
            kind: GraphKind::Predictor,
            opset_version: ONNX_OPSET_VERSION,
            inputs: vec![
                TensorSpec::new(
                    "history",
                    TensorDType::F32,
                    vec![
                        AxisDim::dynamic("batch"),
                        AxisDim::dynamic("history"),
                        AxisDim::fixed(config.predictor.input_dim),
                    ],
                ),
                TensorSpec::new(
                    "actions",
                    TensorDType::F32,
                    vec![
                        AxisDim::dynamic("batch"),
                        AxisDim::dynamic("history"),
                        AxisDim::fixed(config.action_encoder.input_dim),
                    ],
                ),
            ],
            outputs: vec![TensorSpec::new(
                "predicted_embedding",
                TensorDType::F32,
                vec![
                    AxisDim::dynamic("batch"),
                    AxisDim::dynamic("history"),
                    AxisDim::fixed(config.predictor.output_dim),
                ],
            )],
            activation_lowerings: ActivationLoweringSet::rfc0007(),
        }
    }

    /// Return every dynamic axis in graph inputs and outputs.
    pub fn dynamic_axes(&self) -> Vec<DynamicAxis> {
        self.inputs
            .iter()
            .chain(self.outputs.iter())
            .flat_map(TensorSpec::dynamic_axes)
            .collect()
    }

    /// Validate the RFC 0007 graph-level contract.
    ///
    /// # Errors
    ///
    /// Returns [`InferError::InvalidExportContract`] when opset, dynamic axes,
    /// tensor ranks, or activation lowerings do not match RFC 0007.
    pub fn validate(&self) -> InferResult<()> {
        if self.opset_version != ONNX_OPSET_VERSION {
            return Err(InferError::invalid_export_contract(format!(
                "{} graph opset must be {ONNX_OPSET_VERSION}, got {}",
                self.kind.as_str(),
                self.opset_version
            )));
        }

        self.activation_lowerings.validate_rfc0007()?;

        match self.kind {
            GraphKind::Encoder => self.validate_encoder(),
            GraphKind::Predictor => self.validate_predictor(),
        }
    }

    fn validate_encoder(&self) -> InferResult<()> {
        if self.inputs.len() != 1 || self.outputs.len() != 1 {
            return Err(InferError::invalid_export_contract(
                "encoder graph must have one input and one output",
            ));
        }

        let input = &self.inputs[0];
        let output = &self.outputs[0];
        if input.shape.len() != 4 || output.shape.len() != 2 {
            return Err(InferError::invalid_export_contract(
                "encoder graph must map rank-4 pixels to rank-2 embeddings",
            ));
        }

        require_dynamic_axis(input, 0, "batch")?;
        require_dynamic_axis(output, 0, "batch")?;
        Ok(())
    }

    fn validate_predictor(&self) -> InferResult<()> {
        if self.inputs.len() != 2 || self.outputs.len() != 1 {
            return Err(InferError::invalid_export_contract(
                "predictor graph must have history and action inputs plus one output",
            ));
        }

        for tensor in self.inputs.iter().chain(self.outputs.iter()) {
            if tensor.shape.len() != 3 {
                return Err(InferError::invalid_export_contract(format!(
                    "predictor tensor {} must be rank 3",
                    tensor.name
                )));
            }
            require_dynamic_axis(tensor, 0, "batch")?;
            require_dynamic_axis(tensor, 1, "history")?;
        }

        Ok(())
    }
}

/// Full ONNX export specification for encoder and predictor graphs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnnxExportSpec {
    /// Encoder graph specification.
    pub encoder: OnnxGraphSpec,
    /// Predictor graph specification.
    pub predictor: OnnxGraphSpec,
}

impl OnnxExportSpec {
    /// Build the RFC 0007 export spec from a `JEPA` config.
    ///
    /// # Errors
    ///
    /// Returns [`InferError::InvalidExportContract`] when the core model config
    /// does not satisfy its own shape contract.
    pub fn from_jepa_config(config: &JepaConfig) -> InferResult<Self> {
        if let Err(errors) = config.validate_shape_contract() {
            return Err(InferError::invalid_export_contract(errors.join("; ")));
        }

        let spec = Self {
            encoder: OnnxGraphSpec::encoder(config),
            predictor: OnnxGraphSpec::predictor(config),
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Validate both graph specs.
    ///
    /// # Errors
    ///
    /// Returns [`InferError::InvalidExportContract`] when any graph violates
    /// RFC 0007.
    pub fn validate(&self) -> InferResult<()> {
        self.encoder.validate()?;
        self.predictor.validate()?;
        Ok(())
    }
}

/// Request passed to a concrete ONNX graph exporter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphExportRequest {
    /// Burn checkpoint or checkpoint directory.
    pub checkpoint_path: PathBuf,
    /// ONNX graph contract to emit.
    pub graph: OnnxGraphSpec,
    /// Target ONNX path.
    pub output_path: PathBuf,
}

impl GraphExportRequest {
    /// Validate the graph request.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint path is missing or the graph contract
    /// is invalid.
    pub fn validate(&self) -> InferResult<()> {
        if !self.checkpoint_path.exists() {
            return Err(InferError::MissingPath {
                path: self.checkpoint_path.clone(),
            });
        }
        self.graph.validate()
    }
}

/// One exported ONNX graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedOnnxGraph {
    /// Graph kind.
    pub kind: GraphKind,
    /// Path to the exported ONNX file.
    pub path: PathBuf,
    /// Graph contract used to write the file.
    pub spec: OnnxGraphSpec,
}

/// Export result for the paired encoder and predictor graphs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnnxExportArtifacts {
    /// Exported encoder graph.
    pub encoder: ExportedOnnxGraph,
    /// Exported predictor graph.
    pub predictor: ExportedOnnxGraph,
}

/// Concrete backend that can export Burn modules into ONNX files.
pub trait OnnxModelExporter {
    /// Export one ONNX graph according to the request contract.
    ///
    /// # Errors
    ///
    /// Returns backend-specific failures when graph emission fails.
    fn export_graph(
        &self,
        request: &GraphExportRequest,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// Export encoder and predictor ONNX graphs.
///
/// # Errors
///
/// Returns [`InferError`] when the graph contract is invalid, a required path is
/// missing, filesystem setup fails, or the concrete exporter fails.
pub fn export_onnx_pair(
    config: &JepaConfig,
    checkpoint_path: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    exporter: &impl OnnxModelExporter,
) -> InferResult<OnnxExportArtifacts> {
    let spec = OnnxExportSpec::from_jepa_config(config)?;
    let checkpoint_path = checkpoint_path.as_ref().to_path_buf();
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir).map_err(|source| InferError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;

    let encoder = export_graph(&checkpoint_path, output_dir, spec.encoder.clone(), exporter)?;
    let predictor = export_graph(
        &checkpoint_path,
        output_dir,
        spec.predictor.clone(),
        exporter,
    )?;
    let artifacts = OnnxExportArtifacts { encoder, predictor };
    write_export_metadata(output_dir, &artifacts)?;
    Ok(artifacts)
}

/// Write export metadata beside the ONNX graphs.
///
/// # Errors
///
/// Returns [`InferError`] when serialization or file writing fails.
pub fn write_export_metadata(
    output_dir: impl AsRef<Path>,
    artifacts: &OnnxExportArtifacts,
) -> InferResult<PathBuf> {
    let path = output_dir.as_ref().join(ONNX_EXPORT_METADATA_FILENAME);
    let bytes = serde_json::to_vec_pretty(artifacts).map_err(|source| InferError::Json {
        path: path.clone(),
        source,
    })?;
    fs::write(&path, bytes).map_err(|source| InferError::Io {
        path: path.clone(),
        source,
    })?;
    Ok(path)
}

fn export_graph(
    checkpoint_path: &Path,
    output_dir: &Path,
    graph: OnnxGraphSpec,
    exporter: &impl OnnxModelExporter,
) -> InferResult<ExportedOnnxGraph> {
    let output_path = output_dir.join(graph.kind.filename());
    let request = GraphExportRequest {
        checkpoint_path: checkpoint_path.to_path_buf(),
        graph,
        output_path: output_path.clone(),
    };
    request.validate()?;

    exporter
        .export_graph(&request)
        .map_err(|source| InferError::ExportFailed {
            graph: request.graph.kind.as_str(),
            path: output_path.clone(),
            source,
        })?;

    Ok(ExportedOnnxGraph {
        kind: request.graph.kind,
        path: output_path,
        spec: request.graph,
    })
}

fn require_dynamic_axis(tensor: &TensorSpec, axis: usize, expected: &str) -> InferResult<()> {
    match tensor.shape.get(axis) {
        Some(AxisDim::Dynamic(name)) if name == expected => Ok(()),
        Some(_) => Err(InferError::invalid_export_contract(format!(
            "tensor {} axis {axis} must be dynamic `{expected}`",
            tensor.name
        ))),
        None => Err(InferError::invalid_export_contract(format!(
            "tensor {} is missing axis {axis}",
            tensor.name
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[derive(Debug, Default)]
    struct RecordingExporter {
        graphs: RefCell<Vec<GraphKind>>,
    }

    impl OnnxModelExporter for RecordingExporter {
        fn export_graph(
            &self,
            request: &GraphExportRequest,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            request.validate()?;
            self.graphs.borrow_mut().push(request.graph.kind);
            fs::write(
                &request.output_path,
                format!("test graph: {}", request.graph.kind.as_str()),
            )?;
            Ok(())
        }
    }

    #[test]
    fn onnx_export_encoder_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let config = JepaConfig::default();
        let spec = OnnxExportSpec::from_jepa_config(&config)?;
        spec.encoder.validate()?;

        assert_eq!(spec.encoder.opset_version, ONNX_OPSET_VERSION);
        assert_eq!(spec.encoder.kind.filename(), ENCODER_ONNX_FILENAME);
        assert_eq!(
            spec.encoder.inputs[0].dynamic_axes(),
            vec![DynamicAxis {
                axis: 0,
                name: "batch".to_owned()
            }]
        );
        assert_eq!(
            spec.encoder.outputs[0].shape,
            vec![AxisDim::dynamic("batch"), AxisDim::fixed(384)]
        );
        assert_eq!(
            spec.encoder.activation_lowerings.as_slice(),
            ActivationLoweringSet::rfc0007().as_slice()
        );

        Ok(())
    }

    #[test]
    fn onnx_export_predictor_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let config = JepaConfig::default();
        let spec = OnnxExportSpec::from_jepa_config(&config)?;
        spec.predictor.validate()?;

        assert_eq!(spec.predictor.opset_version, ONNX_OPSET_VERSION);
        assert_eq!(spec.predictor.kind.filename(), PREDICTOR_ONNX_FILENAME);
        for input in &spec.predictor.inputs {
            assert_eq!(
                &input.dynamic_axes()[..2],
                [
                    DynamicAxis {
                        axis: 0,
                        name: "batch".to_owned()
                    },
                    DynamicAxis {
                        axis: 1,
                        name: "history".to_owned()
                    }
                ]
            );
        }
        assert_eq!(
            spec.predictor.outputs[0].shape,
            vec![
                AxisDim::dynamic("batch"),
                AxisDim::dynamic("history"),
                AxisDim::fixed(384)
            ]
        );
        assert_eq!(
            spec.predictor.activation_lowerings.as_slice(),
            ActivationLoweringSet::rfc0007().as_slice()
        );

        Ok(())
    }

    #[test]
    fn export_pair_writes_two_graphs_and_metadata() -> Result<(), Box<dyn std::error::Error>> {
        let root = unique_temp_dir("lewm-onnx-export")?;
        let checkpoint = root.join("checkpoint.mpk");
        let output = root.join("out");
        fs::write(&checkpoint, b"checkpoint")?;

        let exporter = RecordingExporter::default();
        let artifacts = export_onnx_pair(&JepaConfig::default(), &checkpoint, &output, &exporter)?;

        assert_eq!(
            *exporter.graphs.borrow(),
            vec![GraphKind::Encoder, GraphKind::Predictor]
        );
        assert_eq!(artifacts.encoder.path, output.join(ENCODER_ONNX_FILENAME));
        assert_eq!(
            artifacts.predictor.path,
            output.join(PREDICTOR_ONNX_FILENAME)
        );
        assert!(artifacts.encoder.path.exists());
        assert!(artifacts.predictor.path.exists());
        assert!(output.join(ONNX_EXPORT_METADATA_FILENAME).exists());

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn predictor_requires_dynamic_history_axis() -> Result<(), Box<dyn std::error::Error>> {
        let config = JepaConfig::default();
        let mut spec = OnnxGraphSpec::predictor(&config);
        spec.inputs[0].shape[1] = AxisDim::fixed(config.history_size);

        let error = spec.validate().err().ok_or("expected validation error")?;
        assert!(error.to_string().contains("history"));
        Ok(())
    }

    #[test]
    fn activation_lowerings_cover_rfc0007_ops() -> Result<(), Box<dyn std::error::Error>> {
        let lowerings = ActivationLoweringSet::rfc0007();
        lowerings.validate_rfc0007()?;
        assert!(lowerings.as_slice().iter().any(|lowering| {
            lowering.activation_name() == "gelu_tanh_approx"
                && lowering.primitive_ops().contains(&"Tanh")
        }));
        assert!(lowerings.as_slice().iter().any(|lowering| {
            lowering.activation_name() == "silu" && lowering.primitive_ops().contains(&"Sigmoid")
        }));
        Ok(())
    }

    fn unique_temp_dir(prefix: &str) -> std::io::Result<PathBuf> {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        path.push(format!("{prefix}-{nanos}"));
        fs::create_dir_all(&path)?;
        Ok(path)
    }
}
