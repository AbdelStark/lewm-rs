//! Deterministic Safetensors export helpers for `Jepa` parameters.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use burn::module::{Module, ModuleVisitor, Param};
use burn::tensor::{Bool, DataError, Int, Tensor, backend::Backend};
use serde::Serialize;

use crate::Jepa;

/// Safetensors dtype emitted by the core exporter.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExportDType {
    /// IEEE-754 single precision float.
    F32,
    /// Signed 64-bit integer.
    I64,
}

impl ExportDType {
    fn safetensors_name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::I64 => "I64",
        }
    }
}

/// One model tensor collected from Burn's module visitor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExportedTensor {
    /// Stable module path, for example `encoder.embeddings.patch_embed.proj.weight`.
    pub name: String,
    /// Safetensors dtype.
    pub dtype: ExportDType,
    /// Row-major tensor shape.
    pub shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl ExportedTensor {
    /// Return the serialized payload size in bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Return the little-endian tensor payload bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Summary returned after writing a Safetensors export.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExportSummary {
    /// Destination path written by the exporter.
    pub path: PathBuf,
    /// Number of tensors in the file.
    pub tensor_count: usize,
    /// Total file size in bytes.
    pub byte_len: usize,
}

/// Errors raised while collecting or writing Safetensors exports.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// The output path did not use the expected extension.
    #[error("Safetensors export path must end in .safetensors: {0}")]
    InvalidExtension(PathBuf),
    /// A filesystem operation failed.
    #[error("I/O error at {}: {source}", path.display())]
    Io {
        /// Path involved in the failed operation.
        path: PathBuf,
        /// Original I/O error.
        source: std::io::Error,
    },
    /// Tensor data could not be converted to the requested scalar dtype.
    #[error("tensor data conversion failed for {name}: {source}")]
    TensorData {
        /// Tensor name.
        name: String,
        /// Burn tensor data conversion error.
        source: DataError,
    },
    /// Tensor collection produced invalid or unsupported parameters.
    #[error("Safetensors parameter collection failed: {}", .0.join("; "))]
    ParameterErrors(Vec<String>),
    /// Safetensors header serialization failed.
    #[error("Safetensors header serialization failed: {source}")]
    Header {
        /// Original JSON serialization error.
        source: serde_json::Error,
    },
    /// A byte offset or allocation size overflowed.
    #[error("Safetensors export size overflowed while writing {context}")]
    SizeOverflow {
        /// Operation being performed.
        context: &'static str,
    },
}

/// Return the RFC 0005 step Safetensors file name.
#[must_use]
pub fn step_safetensors_file_name(step: u64) -> String {
    format!("step_{step:07}.safetensors")
}

/// Collect `Jepa` tensors in Burn module-visitor order.
///
/// This includes trainable float parameters, float running state such as
/// `BatchNorm` statistics, and integer parameters such as `BatchNorm`
/// `num_batches_tracked`.
///
/// # Errors
///
/// Returns [`ExportError::ParameterErrors`] when a tensor has an invalid name,
/// duplicate path, unsupported dtype, or incoherent shape/data length.
pub fn collect_parameters<B: Backend>(model: &Jepa<B>) -> Result<Vec<ExportedTensor>, ExportError> {
    let mut visitor = ExportVisitor::default();
    model.visit(&mut visitor);
    visitor.finish()
}

/// Serialize `Jepa` parameters to owned Safetensors bytes.
///
/// # Errors
///
/// Returns an error if model tensor collection fails or if the Safetensors
/// header cannot be encoded.
pub fn to_safetensors_bytes<B: Backend>(model: &Jepa<B>) -> Result<Vec<u8>, ExportError> {
    let tensors = collect_parameters(model)?;
    serialize_tensors(&tensors)
}

/// Write `Jepa` parameters to a Safetensors file.
///
/// # Errors
///
/// Returns an error if model tensor collection, serialization, directory
/// creation, file writing, or atomic rename fails.
pub fn to_safetensors<B: Backend>(
    model: &Jepa<B>,
    path: impl AsRef<Path>,
) -> Result<ExportSummary, ExportError> {
    let path = path.as_ref();
    require_safetensors_path(path)?;
    let tensors = collect_parameters(model)?;
    let bytes = serialize_tensors(&tensors)?;
    write_atomic_bytes(path, &bytes)?;
    Ok(ExportSummary {
        path: path.to_path_buf(),
        tensor_count: tensors.len(),
        byte_len: bytes.len(),
    })
}

/// Write `Jepa` parameters to `output_dir/step_{N:07d}.safetensors`.
///
/// # Errors
///
/// Returns the same errors as [`to_safetensors`].
pub fn to_step_safetensors<B: Backend>(
    model: &Jepa<B>,
    output_dir: impl AsRef<Path>,
    step: u64,
) -> Result<ExportSummary, ExportError> {
    let path = output_dir.as_ref().join(step_safetensors_file_name(step));
    to_safetensors(model, path)
}

#[derive(Default)]
struct ExportVisitor {
    stack: Vec<String>,
    tensors: Vec<ExportedTensor>,
    errors: Vec<String>,
}

impl<B: Backend> ModuleVisitor<B> for ExportVisitor {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.stack.push(name.to_owned());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.stack.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let name = self.current_name();
        let tensor = param.val();
        let shape = tensor.dims().to_vec();
        match tensor.to_data().to_vec::<f32>() {
            Ok(values) => self.push_f32(name, shape, &values),
            Err(source) => self
                .errors
                .push(ExportError::TensorData { name, source }.to_string()),
        }
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) {
        let name = self.current_name();
        let tensor = param.val();
        let shape = tensor.dims().to_vec();
        match tensor.to_data().to_vec::<i64>() {
            Ok(values) => self.push_i64(name, shape, &values),
            Err(source) => self
                .errors
                .push(ExportError::TensorData { name, source }.to_string()),
        }
    }

    fn visit_bool<const D: usize>(&mut self, _param: &Param<Tensor<B, D, Bool>>) {
        let name = self.current_name();
        self.errors.push(format!(
            "bool tensor {name} cannot be exported to the LeWM parameter Safetensors mirror"
        ));
    }
}

impl ExportVisitor {
    fn current_name(&mut self) -> String {
        let name = self.stack.join(".");
        if name.is_empty() {
            self.errors
                .push("encountered a tensor without a module path".to_owned());
        }
        name
    }

    fn push_f32(&mut self, name: String, shape: Vec<usize>, values: &[f32]) {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        self.push_tensor(name, ExportDType::F32, shape, values.len(), bytes);
    }

    fn push_i64(&mut self, name: String, shape: Vec<usize>, values: &[i64]) {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        self.push_tensor(name, ExportDType::I64, shape, values.len(), bytes);
    }

    fn push_tensor(
        &mut self,
        name: String,
        dtype: ExportDType,
        shape: Vec<usize>,
        value_count: usize,
        bytes: Vec<u8>,
    ) {
        if name.is_empty() {
            return;
        }
        match element_count(&shape) {
            Ok(expected) if expected == value_count => self.tensors.push(ExportedTensor {
                name,
                dtype,
                shape,
                bytes,
            }),
            Ok(expected) => self.errors.push(format!(
                "tensor {name} shape product {expected} does not match value count {value_count}"
            )),
            Err(message) => self.errors.push(format!("tensor {name} {message}")),
        }
    }

    fn finish(self) -> Result<Vec<ExportedTensor>, ExportError> {
        let mut errors = self.errors;
        let mut seen = BTreeSet::new();
        for tensor in &self.tensors {
            if !seen.insert(tensor.name.clone()) {
                errors.push(format!("duplicate tensor name {}", tensor.name));
            }
        }

        if errors.is_empty() {
            Ok(self.tensors)
        } else {
            Err(ExportError::ParameterErrors(errors))
        }
    }
}

#[derive(Serialize)]
struct TensorHeader {
    dtype: &'static str,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

fn serialize_tensors(tensors: &[ExportedTensor]) -> Result<Vec<u8>, ExportError> {
    let mut header = BTreeMap::new();
    let mut offset = 0usize;
    for tensor in tensors {
        let end = offset
            .checked_add(tensor.bytes.len())
            .ok_or(ExportError::SizeOverflow {
                context: "tensor data offsets",
            })?;
        header.insert(
            tensor.name.clone(),
            TensorHeader {
                dtype: tensor.dtype.safetensors_name(),
                shape: tensor.shape.clone(),
                data_offsets: [offset, end],
            },
        );
        offset = end;
    }

    let mut header_bytes =
        serde_json::to_vec(&header).map_err(|source| ExportError::Header { source })?;
    let padding = (8 - (header_bytes.len() % 8)) % 8;
    header_bytes.extend(std::iter::repeat_n(b' ', padding));
    let header_len = u64::try_from(header_bytes.len()).map_err(|_| ExportError::SizeOverflow {
        context: "header length",
    })?;
    let total_len = 8usize
        .checked_add(header_bytes.len())
        .and_then(|len| len.checked_add(offset))
        .ok_or(ExportError::SizeOverflow {
            context: "output buffer length",
        })?;

    let mut bytes = Vec::with_capacity(total_len);
    bytes.extend_from_slice(&header_len.to_le_bytes());
    bytes.extend_from_slice(&header_bytes);
    for tensor in tensors {
        bytes.extend_from_slice(&tensor.bytes);
    }
    debug_assert_eq!(bytes.len(), total_len);
    Ok(bytes)
}

fn require_safetensors_path(path: &Path) -> Result<(), ExportError> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("safetensors") => Ok(()),
        _ => Err(ExportError::InvalidExtension(path.to_path_buf())),
    }
}

fn write_atomic_bytes(path: &Path, bytes: &[u8]) -> Result<(), ExportError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|source| ExportError::Io {
        path: parent.to_path_buf(),
        source,
    })?;

    let tmp_path = tmp_path_for(path);
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&tmp_path)
        .map_err(|source| ExportError::Io {
            path: tmp_path.clone(),
            source,
        })?;
    file.write_all(bytes).map_err(|source| ExportError::Io {
        path: tmp_path.clone(),
        source,
    })?;
    file.sync_all().map_err(|source| ExportError::Io {
        path: tmp_path.clone(),
        source,
    })?;
    drop(file);

    fs::rename(&tmp_path, path).map_err(|source| ExportError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn tmp_path_for(path: &Path) -> PathBuf {
    let mut file_name = path
        .file_name()
        .map_or_else(|| ".lewm-core-export".into(), std::ffi::OsStr::to_os_string);
    file_name.push(".tmp");
    path.with_file_name(file_name)
}

fn element_count(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |acc, dim| {
        if *dim == 0 {
            return Err("shape contains a zero dimension".to_owned());
        }
        acc.checked_mul(*dim)
            .ok_or_else(|| "shape element count overflowed usize".to_owned())
    })
}
