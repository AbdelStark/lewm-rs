//! Build a Burn `NamedMpkFileRecorder` record from converted reference tensors.
//!
//! NOTE: The Safetensors→Burn loader logic here is kept in-place for backwards
//! compatibility. A backend-generic re-implementation lives in
//! [`lewm_core::import`] and is used by the inference runners. The duplication
//! will converge in a follow-up cleanup.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use burn_core::module::{Module, ModuleMapper, Param};
use burn_core::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_core::tensor::{Int, Tensor, TensorData};
use burn_ndarray::{NdArray, NdArrayDevice};
use clap::Parser;
use lewm_core::{Jepa, JepaConfig, LewmCoreError};
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;

type CpuBackend = NdArray<f32>;

#[derive(Debug, Parser)]
#[command(
    name = "lewm-reference-record",
    about = "Convert a lewm-rs reference Safetensors mirror into a Burn NamedMpk record."
)]
struct Args {
    /// Converted reference Safetensors file produced by `python/convert_reference.py`.
    #[arg(long)]
    safetensors_in: PathBuf,
    /// Burn record output path. Must end in `.mpk`.
    #[arg(
        long,
        conflicts_with = "burn_record_in",
        required_unless_present = "burn_record_in"
    )]
    burn_record_out: Option<PathBuf>,
    /// Existing Burn record to verify against `--safetensors-in`.
    #[arg(
        long,
        conflicts_with = "burn_record_out",
        required_unless_present = "burn_record_out"
    )]
    burn_record_in: Option<PathBuf>,
}

#[derive(Debug)]
enum RecordBuildError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Safetensors(safetensors::SafeTensorError),
    Core(LewmCoreError),
    Recorder(burn_core::record::RecorderError),
    InvalidOutputPath(PathBuf),
    InvalidTensor {
        name: String,
        reason: String,
    },
    ParameterErrors(Vec<String>),
}

impl fmt::Display for RecordBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(formatter, "I/O error at {}: {source}", path.display())
            },
            Self::Safetensors(source) => write!(formatter, "safetensors error: {source}"),
            Self::Core(source) => write!(formatter, "core model error: {source}"),
            Self::Recorder(source) => write!(formatter, "Burn recorder error: {source}"),
            Self::InvalidOutputPath(path) => write!(
                formatter,
                "Burn record output path must end in .mpk: {}",
                path.display()
            ),
            Self::InvalidTensor { name, reason } => {
                write!(formatter, "invalid tensor {name}: {reason}")
            },
            Self::ParameterErrors(errors) => {
                write!(
                    formatter,
                    "reference tensor mapping failed: {}",
                    errors.join("; ")
                )
            },
        }
    }
}

impl Error for RecordBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Safetensors(source) => Some(source),
            Self::Core(source) => Some(source),
            Self::Recorder(source) => Some(source),
            Self::InvalidOutputPath(_) | Self::InvalidTensor { .. } | Self::ParameterErrors(_) => {
                None
            },
        }
    }
}

impl From<safetensors::SafeTensorError> for RecordBuildError {
    fn from(source: safetensors::SafeTensorError) -> Self {
        Self::Safetensors(source)
    }
}

impl From<LewmCoreError> for RecordBuildError {
    fn from(source: LewmCoreError) -> Self {
        Self::Core(source)
    }
}

impl From<burn_core::record::RecorderError> for RecordBuildError {
    fn from(source: burn_core::record::RecorderError) -> Self {
        Self::Recorder(source)
    }
}

#[allow(clippy::print_stdout)]
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if let Some(path) = args.burn_record_out {
        let summary =
            write_burn_record_from_safetensors(&args.safetensors_in, &path, JepaConfig::default())?;
        println!(
            "reference burn record: tensors={} out={}",
            summary.tensor_count,
            summary.output_path.display()
        );
    } else if let Some(path) = args.burn_record_in {
        let summary = verify_burn_record_against_safetensors(
            &args.safetensors_in,
            &path,
            JepaConfig::default(),
        )?;
        println!(
            "reference burn record verify: tensors={} max_abs_diff={:.8e} record={}",
            summary.tensor_count,
            summary.max_abs_diff,
            path.display()
        );
    }
    Ok(())
}

#[derive(Debug)]
struct RecordSummary {
    tensor_count: usize,
    output_path: PathBuf,
}

#[derive(Debug)]
struct VerifySummary {
    tensor_count: usize,
    max_abs_diff: f64,
}

fn write_burn_record_from_safetensors(
    safetensors_in: &Path,
    burn_record_out: &Path,
    config: JepaConfig,
) -> Result<RecordSummary, RecordBuildError> {
    require_mpk_path(burn_record_out)?;
    let tensors = load_safetensors(safetensors_in)?;
    let device = NdArrayDevice::default();
    let model = Jepa::<CpuBackend>::init(config.clone(), &device)?;
    let model = apply_tensors(model, &tensors, device)?;

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    recorder.record(model.into_record(), burn_record_out.to_path_buf())?;

    let reload_model = Jepa::<CpuBackend>::init(config, &device)?;
    let record = recorder.load(burn_record_out.to_path_buf(), &device)?;
    let _loaded = reload_model.load_record(record);

    Ok(RecordSummary {
        tensor_count: tensors.len(),
        output_path: burn_record_out.to_path_buf(),
    })
}

fn verify_burn_record_against_safetensors(
    safetensors_in: &Path,
    burn_record_in: &Path,
    config: JepaConfig,
) -> Result<VerifySummary, RecordBuildError> {
    require_mpk_path(burn_record_in)?;
    let expected = load_safetensors(safetensors_in)?;
    let device = NdArrayDevice::default();
    let model = Jepa::<CpuBackend>::init(config, &device)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    let record = recorder.load(burn_record_in.to_path_buf(), &device)?;
    let model = model.load_record(record);
    let actual = collect_model_tensors(&model)?;
    compare_tensors(&expected, &actual)
}

fn require_mpk_path(path: &Path) -> Result<(), RecordBuildError> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("mpk") => Ok(()),
        _ => Err(RecordBuildError::InvalidOutputPath(path.to_path_buf())),
    }
}

fn load_safetensors(path: &Path) -> Result<BTreeMap<String, LoadedTensor>, RecordBuildError> {
    let bytes = fs::read(path).map_err(|source| RecordBuildError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let safe_tensors = SafeTensors::deserialize(&bytes)?;
    let mut tensors = BTreeMap::new();
    for (name, tensor) in safe_tensors.iter() {
        let loaded = match tensor.dtype() {
            Dtype::F32 => LoadedTensor::F32 {
                shape: tensor.shape().to_vec(),
                values: read_f32_values(name, tensor.data())?,
            },
            Dtype::I64 => LoadedTensor::I64 {
                shape: tensor.shape().to_vec(),
                values: read_i64_values(name, tensor.data())?,
            },
            dtype => {
                return Err(RecordBuildError::InvalidTensor {
                    name: name.to_owned(),
                    reason: format!("unsupported dtype {dtype:?}; expected F32 or I64"),
                });
            },
        };
        tensors.insert(name.to_owned(), loaded);
    }
    Ok(tensors)
}

fn read_f32_values(name: &str, bytes: &[u8]) -> Result<Vec<f32>, RecordBuildError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(RecordBuildError::InvalidTensor {
            name: name.to_owned(),
            reason: format!("F32 byte length {} is not divisible by 4", bytes.len()),
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn read_i64_values(name: &str, bytes: &[u8]) -> Result<Vec<i64>, RecordBuildError> {
    if !bytes.len().is_multiple_of(8) {
        return Err(RecordBuildError::InvalidTensor {
            name: name.to_owned(),
            reason: format!("I64 byte length {} is not divisible by 8", bytes.len()),
        });
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
}

#[derive(Clone, Debug)]
enum LoadedTensor {
    F32 { shape: Vec<usize>, values: Vec<f32> },
    I64 { shape: Vec<usize>, values: Vec<i64> },
}

impl LoadedTensor {
    fn shape(&self) -> &[usize] {
        match self {
            Self::F32 { shape, .. } | Self::I64 { shape, .. } => shape,
        }
    }
}

fn apply_tensors(
    model: Jepa<CpuBackend>,
    tensors: &BTreeMap<String, LoadedTensor>,
    device: NdArrayDevice,
) -> Result<Jepa<CpuBackend>, RecordBuildError> {
    let mut mapper = ReferenceTensorMapper {
        tensors,
        device,
        stack: Vec::new(),
        used: BTreeSet::new(),
        errors: Vec::new(),
    };
    let model = model.map(&mut mapper);
    for name in tensors.keys() {
        if !mapper.used.contains(name) {
            mapper
                .errors
                .push(format!("tensor {name} does not match any JEPA parameter"));
        }
    }
    if mapper.errors.is_empty() {
        Ok(model)
    } else {
        Err(RecordBuildError::ParameterErrors(mapper.errors))
    }
}

fn collect_model_tensors(
    model: &Jepa<CpuBackend>,
) -> Result<BTreeMap<String, LoadedTensor>, RecordBuildError> {
    let mut visitor = TensorCollector {
        stack: Vec::new(),
        tensors: BTreeMap::new(),
        errors: Vec::new(),
    };
    model.visit(&mut visitor);
    if visitor.errors.is_empty() {
        Ok(visitor.tensors)
    } else {
        Err(RecordBuildError::ParameterErrors(visitor.errors))
    }
}

struct TensorCollector {
    stack: Vec<String>,
    tensors: BTreeMap<String, LoadedTensor>,
    errors: Vec<String>,
}

impl burn_core::module::ModuleVisitor<CpuBackend> for TensorCollector {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.stack.push(name.to_owned());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.stack.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<CpuBackend, D>>) {
        let name = self.stack.join(".");
        if is_generated_reference_tensor(&name) {
            return;
        }
        let shape = param.dims().to_vec();
        match param.val().into_data().to_vec::<f32>() {
            Ok(values) => {
                self.tensors
                    .insert(name, LoadedTensor::F32 { shape, values });
            },
            Err(error) => self.errors.push(format!("{name}: {error}")),
        }
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<CpuBackend, D, Int>>) {
        let name = self.stack.join(".");
        if is_generated_reference_tensor(&name) {
            return;
        }
        let shape = param.dims().to_vec();
        match param.val().into_data().to_vec::<i64>() {
            Ok(values) => {
                self.tensors
                    .insert(name, LoadedTensor::I64 { shape, values });
            },
            Err(error) => self.errors.push(format!("{name}: {error}")),
        }
    }
}

fn compare_tensors(
    expected: &BTreeMap<String, LoadedTensor>,
    actual: &BTreeMap<String, LoadedTensor>,
) -> Result<VerifySummary, RecordBuildError> {
    let mut errors = Vec::new();
    let mut max_abs_diff = 0.0_f64;
    for (name, expected_tensor) in expected {
        let Some(actual_tensor) = actual.get(name) else {
            errors.push(format!("record is missing tensor {name}"));
            continue;
        };
        match (expected_tensor, actual_tensor) {
            (
                LoadedTensor::F32 {
                    shape: expected_shape,
                    values: expected_values,
                },
                LoadedTensor::F32 {
                    shape: actual_shape,
                    values: actual_values,
                },
            ) => {
                if expected_shape != actual_shape {
                    errors.push(format!(
                        "{name} shape mismatch: expected {expected_shape:?}, found {actual_shape:?}"
                    ));
                    continue;
                }
                if expected_values.len() != actual_values.len() {
                    errors.push(format!(
                        "{name} length mismatch: expected {}, found {}",
                        expected_values.len(),
                        actual_values.len()
                    ));
                    continue;
                }
                for (expected, actual) in expected_values.iter().zip(actual_values) {
                    max_abs_diff = max_abs_diff.max(f64::from((expected - actual).abs()));
                }
            },
            (
                LoadedTensor::I64 {
                    shape: expected_shape,
                    values: expected_values,
                },
                LoadedTensor::I64 {
                    shape: actual_shape,
                    values: actual_values,
                },
            ) => {
                if expected_shape != actual_shape {
                    errors.push(format!(
                        "{name} shape mismatch: expected {expected_shape:?}, found {actual_shape:?}"
                    ));
                    continue;
                }
                if expected_values != actual_values {
                    errors.push(format!("{name} I64 values differ"));
                }
            },
            _ => errors.push(format!("{name} dtype mismatch")),
        }
    }
    for name in actual.keys() {
        if !expected.contains_key(name) {
            errors.push(format!("record has extra tensor {name}"));
        }
    }
    if errors.is_empty() {
        Ok(VerifySummary {
            tensor_count: expected.len(),
            max_abs_diff,
        })
    } else {
        Err(RecordBuildError::ParameterErrors(errors))
    }
}

struct ReferenceTensorMapper<'a> {
    tensors: &'a BTreeMap<String, LoadedTensor>,
    device: NdArrayDevice,
    stack: Vec<String>,
    used: BTreeSet<String>,
    errors: Vec<String>,
}

impl ModuleMapper<CpuBackend> for ReferenceTensorMapper<'_> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.stack.push(name.to_owned());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.stack.pop();
    }

    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<CpuBackend, D>>,
    ) -> Param<Tensor<CpuBackend, D>> {
        let name = self.current_name();
        let Some(loaded) = self.tensors.get(&name) else {
            if !is_generated_reference_tensor(&name) {
                self.errors.push(format!("missing F32 tensor for {name}"));
            }
            return param;
        };
        let LoadedTensor::F32 { shape, values } = loaded else {
            self.errors.push(format!("{name} is not F32"));
            return param;
        };
        if let Err(error) = validate_shape(&name, loaded.shape(), &param.lazy_shape().dims) {
            self.errors.push(error);
            return param;
        }
        let Ok(shape_array): Result<[usize; D], _> = shape.clone().try_into() else {
            self.errors.push(format!(
                "{name} rank mismatch: expected rank {D}, found {}",
                shape.len()
            ));
            return param;
        };
        self.used.insert(name);
        let (id, _old, mapper) = param.consume();
        let tensor = Tensor::<CpuBackend, D>::from_data(
            TensorData::new(values.clone(), shape_array),
            &self.device,
        );
        Param::from_mapped_value(id, tensor, mapper)
    }

    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<CpuBackend, D, Int>>,
    ) -> Param<Tensor<CpuBackend, D, Int>> {
        let name = self.current_name();
        let Some(loaded) = self.tensors.get(&name) else {
            if !is_generated_reference_tensor(&name) {
                self.errors.push(format!("missing I64 tensor for {name}"));
            }
            return param;
        };
        let LoadedTensor::I64 { shape, values } = loaded else {
            self.errors.push(format!("{name} is not I64"));
            return param;
        };
        if let Err(error) = validate_shape(&name, loaded.shape(), &param.lazy_shape().dims) {
            self.errors.push(error);
            return param;
        }
        let Ok(shape_array): Result<[usize; D], _> = shape.clone().try_into() else {
            self.errors.push(format!(
                "{name} rank mismatch: expected rank {D}, found {}",
                shape.len()
            ));
            return param;
        };
        self.used.insert(name);
        let (id, _old, mapper) = param.consume();
        let tensor = Tensor::<CpuBackend, D, Int>::from_data(
            TensorData::new(values.clone(), shape_array),
            &self.device,
        );
        Param::from_mapped_value(id, tensor, mapper)
    }
}

impl ReferenceTensorMapper<'_> {
    fn current_name(&self) -> String {
        self.stack.join(".")
    }
}

fn validate_shape(name: &str, found: &[usize], expected: &[usize]) -> Result<(), String> {
    if found == expected {
        Ok(())
    } else {
        Err(format!(
            "{name} shape mismatch: expected {expected:?}, found {found:?}"
        ))
    }
}

fn is_generated_reference_tensor(name: &str) -> bool {
    name.starts_with("sigreg.consts.")
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use burn_core::module::{ModuleVisitor, Param};
    use burn_core::tensor::{Bool, Tensor};
    use lewm_core::{
        EmbedderConfig, GeluVariant, MlpConfig, NormVariant, PredictorConfig, VitConfig, VitSize,
    };
    use safetensors::tensor::{Dtype, TensorView};

    use super::*;

    #[test]
    fn writes_and_loads_named_mpk_from_safetensors() -> Result<(), Box<dyn Error>> {
        let config = tiny_config();
        let device = NdArrayDevice::default();
        let model = Jepa::<CpuBackend>::init(config.clone(), &device)?;
        let tensors = synthetic_safetensors_payload(&model);
        let dir = tempfile::tempdir()?;
        let safetensors_path = dir.path().join("reference.safetensors");
        write_test_safetensors(&safetensors_path, &tensors)?;

        let burn_record_path = dir.path().join("reference.mpk");
        let summary =
            write_burn_record_from_safetensors(&safetensors_path, &burn_record_path, config)?;

        assert_eq!(summary.tensor_count, tensors.len());
        assert!(burn_record_path.is_file());
        assert!(fs::metadata(&burn_record_path)?.len() > 0);
        let verify = verify_burn_record_against_safetensors(
            &safetensors_path,
            &burn_record_path,
            tiny_config(),
        )?;
        assert_eq!(verify.tensor_count, tensors.len());
        assert!(verify.max_abs_diff <= f64::EPSILON);
        Ok(())
    }

    #[test]
    fn rejects_missing_model_tensor() -> Result<(), Box<dyn Error>> {
        let config = tiny_config();
        let device = NdArrayDevice::default();
        let model = Jepa::<CpuBackend>::init(config.clone(), &device)?;
        let mut tensors = synthetic_safetensors_payload(&model);
        let removed = tensors.keys().next().cloned().expect("non-empty model");
        tensors.remove(&removed);
        let dir = tempfile::tempdir()?;
        let safetensors_path = dir.path().join("reference.safetensors");
        write_test_safetensors(&safetensors_path, &tensors)?;

        let burn_record_path = dir.path().join("reference.mpk");
        let error =
            write_burn_record_from_safetensors(&safetensors_path, &burn_record_path, config)
                .expect_err("missing tensor must be rejected");

        assert!(
            error
                .to_string()
                .contains(&format!("missing F32 tensor for {removed}"))
        );
        Ok(())
    }

    fn tiny_config() -> JepaConfig {
        JepaConfig {
            encoder: VitConfig {
                size: VitSize::Tiny,
                image_size: 16,
                patch_size: 8,
                num_channels: 3,
                hidden_size: 8,
                num_hidden_layers: 1,
                num_attention_heads: 2,
                intermediate_size: 16,
                hidden_act: GeluVariant::TanhApprox,
                attention_probs_dropout_prob: 0.0,
                hidden_dropout_prob: 0.0,
                layer_norm_eps: 1.0e-12,
                use_cls_token: true,
                interpolate_pos_encoding: false,
                use_mask_token: false,
                pretrained: false,
            },
            action_encoder: EmbedderConfig {
                input_dim: 2,
                smoothed_dim: 2,
                emb_dim: 8,
                mlp_scale: 2,
            },
            predictor: PredictorConfig {
                num_frames: 2,
                depth: 1,
                heads: 2,
                mlp_dim: 16,
                dim_head: 4,
                input_dim: 8,
                hidden_dim: 8,
                output_dim: 8,
                action_emb_dim: 8,
                dropout: 0.0,
                emb_dropout: 0.0,
            },
            projector: MlpConfig {
                input_dim: 8,
                hidden_dim: 16,
                output_dim: 8,
                norm: NormVariant::BatchNorm1d,
            },
            pred_proj: MlpConfig {
                input_dim: 8,
                hidden_dim: 16,
                output_dim: 8,
                norm: NormVariant::BatchNorm1d,
            },
            history_size: 2,
            horizon: 3,
        }
    }

    #[derive(Default)]
    struct ShapeCollector {
        stack: Vec<String>,
        tensors: BTreeMap<String, LoadedTensor>,
        next_value: f32,
    }

    impl ModuleVisitor<CpuBackend> for ShapeCollector {
        fn enter_module(&mut self, name: &str, _container_type: &str) {
            self.stack.push(name.to_owned());
        }

        fn exit_module(&mut self, _name: &str, _container_type: &str) {
            self.stack.pop();
        }

        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<CpuBackend, D>>) {
            if is_generated_reference_tensor(&self.stack.join(".")) {
                return;
            }
            let shape = param.dims().to_vec();
            let count = shape.iter().product();
            let values = (0..count)
                .map(|_| {
                    self.next_value += 0.001;
                    self.next_value
                })
                .collect();
            self.tensors
                .insert(self.stack.join("."), LoadedTensor::F32 { shape, values });
        }

        fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<CpuBackend, D, Int>>) {
            self.tensors.insert(
                self.stack.join("."),
                LoadedTensor::I64 {
                    shape: param.dims().to_vec(),
                    values: vec![0; param.dims().iter().product()],
                },
            );
        }

        fn visit_bool<const D: usize>(&mut self, _param: &Param<Tensor<CpuBackend, D, Bool>>) {}
    }

    fn synthetic_safetensors_payload(model: &Jepa<CpuBackend>) -> BTreeMap<String, LoadedTensor> {
        let mut collector = ShapeCollector::default();
        model.visit(&mut collector);
        collector.tensors
    }

    fn write_test_safetensors(
        path: &Path,
        tensors: &BTreeMap<String, LoadedTensor>,
    ) -> Result<(), Box<dyn Error>> {
        let mut owned = Vec::new();
        for (name, tensor) in tensors {
            let (dtype, shape, bytes) = match tensor {
                LoadedTensor::F32 { shape, values } => (
                    Dtype::F32,
                    shape.clone(),
                    values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
                LoadedTensor::I64 { shape, values } => (
                    Dtype::I64,
                    shape.clone(),
                    values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
            };
            owned.push((name.clone(), dtype, shape, bytes));
        }

        let views = owned
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                TensorView::new(*dtype, shape.clone(), bytes).map(|view| (name.as_str(), view))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let metadata: Option<HashMap<String, String>> = None;
        safetensors::tensor::serialize_to_file(views, &metadata, path)?;
        Ok(())
    }
}
