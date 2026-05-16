//! Safetensors -> `Jepa` parameter loader.
//!
//! This module is the inverse of [`crate::export`]: it walks a `Jepa<B>` module
//! and replaces each parameter with the matching tensor from a Safetensors
//! payload. It is backend-generic so the same loader works for `NdArray`,
//! `LibTorch`, `Cuda`, `Wgpu`, and other Burn backends and is the foundation of
//! both the reference-record builder in `lewm-train` and the Burn-direct
//! inference runners in `lewm-infer`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::{Jepa, JepaConfig, LewmCoreError};

/// One tensor parsed from a Safetensors file.
#[derive(Clone, Debug, PartialEq)]
pub enum LoadedTensor {
    /// F32 tensor.
    F32 {
        /// Row-major shape.
        shape: Vec<usize>,
        /// Values in row-major order.
        values: Vec<f32>,
    },
    /// I64 tensor.
    I64 {
        /// Row-major shape.
        shape: Vec<usize>,
        /// Values in row-major order.
        values: Vec<i64>,
    },
}

impl LoadedTensor {
    /// Return the row-major shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32 { shape, .. } | Self::I64 { shape, .. } => shape,
        }
    }
}

/// Errors raised while loading or applying Safetensors tensors to a `Jepa<B>`.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    /// A filesystem operation failed.
    #[error("I/O error at {}: {source}", path.display())]
    Io {
        /// Path involved in the failed operation.
        path: PathBuf,
        /// Original I/O error.
        source: std::io::Error,
    },
    /// Safetensors deserialization failed.
    #[error("safetensors error at {}: {source}", path.display())]
    Safetensors {
        /// Path involved in the failed operation.
        path: PathBuf,
        /// Original error.
        source: safetensors::SafeTensorError,
    },
    /// A tensor entry could not be decoded.
    #[error("invalid tensor {name}: {reason}")]
    InvalidTensor {
        /// Tensor name.
        name: String,
        /// Reason text.
        reason: String,
    },
    /// One or more tensor entries did not align with the target `Jepa<B>` module.
    #[error("tensor mapping failed: {}", .0.join("; "))]
    Mapping(Vec<String>),
    /// `Jepa<B>` construction failed before the mapper ran.
    #[error("model init failed: {source}")]
    Init {
        /// Original core error.
        source: LewmCoreError,
    },
}

impl ImportError {
    fn from_core(source: LewmCoreError) -> Self {
        Self::Init { source }
    }
}

/// Read all F32/I64 tensors from a Safetensors file into a name-keyed map.
///
/// # Errors
///
/// Returns [`ImportError`] when reading the file fails, the bytes are not
/// valid Safetensors, or a tensor uses a dtype other than F32 or I64.
pub fn load_safetensors_tensors(
    path: &Path,
) -> Result<BTreeMap<String, LoadedTensor>, ImportError> {
    let bytes = fs::read(path).map_err(|source| ImportError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_safetensors_bytes(&bytes, path)
}

/// Parse a Safetensors byte payload into a name-keyed map.
///
/// # Errors
///
/// Returns [`ImportError`] when the bytes are not valid Safetensors, or a
/// tensor uses a dtype other than F32 or I64.
pub fn parse_safetensors_bytes(
    bytes: &[u8],
    origin: &Path,
) -> Result<BTreeMap<String, LoadedTensor>, ImportError> {
    let safe = safetensors::SafeTensors::deserialize(bytes).map_err(|source| {
        ImportError::Safetensors {
            path: origin.to_path_buf(),
            source,
        }
    })?;
    let mut tensors = BTreeMap::new();
    for (name, tensor) in safe.iter() {
        let loaded = match tensor.dtype() {
            safetensors::Dtype::F32 => LoadedTensor::F32 {
                shape: tensor.shape().to_vec(),
                values: read_f32_values(name, tensor.data())?,
            },
            safetensors::Dtype::I64 => LoadedTensor::I64 {
                shape: tensor.shape().to_vec(),
                values: read_i64_values(name, tensor.data())?,
            },
            dtype => {
                return Err(ImportError::InvalidTensor {
                    name: name.to_owned(),
                    reason: format!("unsupported dtype {dtype:?}; expected F32 or I64"),
                });
            },
        };
        tensors.insert(name.to_owned(), loaded);
    }
    Ok(tensors)
}

fn read_f32_values(name: &str, bytes: &[u8]) -> Result<Vec<f32>, ImportError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(ImportError::InvalidTensor {
            name: name.to_owned(),
            reason: format!("F32 byte length {} is not divisible by 4", bytes.len()),
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn read_i64_values(name: &str, bytes: &[u8]) -> Result<Vec<i64>, ImportError> {
    if !bytes.len().is_multiple_of(8) {
        return Err(ImportError::InvalidTensor {
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

/// Strict tensor presence policy used when applying tensors to a `Jepa<B>`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MissingPolicy {
    /// Fail when the Safetensors payload is missing a parameter required by the
    /// module. This is the default policy used by reference-record conversion
    /// and inference loading because both expect a complete mirror.
    Strict,
    /// Tolerate missing parameters and leave the original initialization in
    /// place. This is useful for partial overrides and is documented but not
    /// currently used by default.
    Lenient,
}

/// Override `Jepa<B>` parameters with tensors loaded from a Safetensors map.
///
/// Tensor names with the `sigreg.consts.` prefix are deterministic constants
/// generated at init time and are intentionally skipped — they are not stored
/// in Safetensors mirrors and must not trigger missing-tensor errors.
///
/// # Errors
///
/// Returns [`ImportError::Mapping`] when one or more tensors are missing,
/// have the wrong dtype, the wrong rank, or the wrong shape, or when the
/// payload contains tensors that do not match any parameter.
pub fn apply_tensors_to_jepa<B: Backend>(
    model: Jepa<B>,
    tensors: &BTreeMap<String, LoadedTensor>,
    device: &B::Device,
    policy: MissingPolicy,
) -> Result<Jepa<B>, ImportError> {
    let mut mapper = JepaTensorMapper {
        tensors,
        device,
        policy,
        stack: Vec::new(),
        used: BTreeSet::new(),
        errors: Vec::new(),
    };
    let model = model.map(&mut mapper);
    let mut errors = std::mem::take(&mut mapper.errors);
    for name in tensors.keys() {
        if !mapper.used.contains(name) {
            errors.push(format!("tensor {name} does not match any JEPA parameter"));
        }
    }
    if errors.is_empty() {
        Ok(model)
    } else {
        Err(ImportError::Mapping(errors))
    }
}

/// Load a `Jepa<B>` module from a Safetensors file using the locked default
/// config.
///
/// This is a convenience wrapper over [`load_jepa_from_safetensors_with_config`]
/// for callers that target the locked `PushT` config. SO-100 or custom callers
/// should use the explicit-config variant.
///
/// # Errors
///
/// Returns [`ImportError`] when reading, parsing, init, or mapping fails.
pub fn load_jepa_from_safetensors<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<Jepa<B>, ImportError> {
    load_jepa_from_safetensors_with_config(path, JepaConfig::default(), device)
}

/// Load a `Jepa<B>` module from a Safetensors file with an explicit config.
///
/// # Errors
///
/// Returns [`ImportError`] when reading, parsing, init, or mapping fails.
pub fn load_jepa_from_safetensors_with_config<B: Backend>(
    path: &Path,
    config: JepaConfig,
    device: &B::Device,
) -> Result<Jepa<B>, ImportError> {
    let tensors = load_safetensors_tensors(path)?;
    let model = Jepa::<B>::init(config, device).map_err(ImportError::from_core)?;
    apply_tensors_to_jepa(model, &tensors, device, MissingPolicy::Strict)
}

struct JepaTensorMapper<'a, B: Backend> {
    tensors: &'a BTreeMap<String, LoadedTensor>,
    device: &'a B::Device,
    policy: MissingPolicy,
    stack: Vec<String>,
    used: BTreeSet<String>,
    errors: Vec<String>,
}

impl<B: Backend> ModuleMapper<B> for JepaTensorMapper<'_, B> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.stack.push(name.to_owned());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.stack.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let name = self.current_name();
        let Some(loaded) = self.tensors.get(&name) else {
            if !is_generated_reference_tensor(&name) && self.policy == MissingPolicy::Strict {
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
        let tensor =
            Tensor::<B, D>::from_data(TensorData::new(values.clone(), shape_array), self.device);
        Param::from_mapped_value(id, tensor, mapper)
    }

    fn map_int<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D, Int>>,
    ) -> Param<Tensor<B, D, Int>> {
        let name = self.current_name();
        let Some(loaded) = self.tensors.get(&name) else {
            if !is_generated_reference_tensor(&name) && self.policy == MissingPolicy::Strict {
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
        let tensor = Tensor::<B, D, Int>::from_data(
            TensorData::new(values.clone(), shape_array),
            self.device,
        );
        Param::from_mapped_value(id, tensor, mapper)
    }
}

impl<B: Backend> JepaTensorMapper<'_, B> {
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

/// Names of `Jepa<B>` parameters that are deterministically generated at init
/// time and are intentionally not present in Safetensors mirrors.
fn is_generated_reference_tensor(name: &str) -> bool {
    name.starts_with("sigreg.consts.")
}

#[cfg(test)]
mod tests {
    use burn_ndarray::{NdArray, NdArrayDevice};

    use super::*;
    use crate::export::to_safetensors_bytes;

    type CpuBackend = NdArray<f32>;

    #[test]
    fn parse_safetensors_bytes_round_trips_jepa() {
        let device = NdArrayDevice::default();
        let model = Jepa::<CpuBackend>::init(JepaConfig::default(), &device)
            .expect("init reference Jepa for round-trip");
        let bytes = to_safetensors_bytes(&model).expect("serialize Jepa to safetensors");
        let tensors = parse_safetensors_bytes(&bytes, Path::new("inline://round-trip"))
            .expect("parse safetensors back");

        let reloaded =
            Jepa::<CpuBackend>::init(JepaConfig::default(), &device).expect("init Jepa for reload");
        let reloaded = apply_tensors_to_jepa(reloaded, &tensors, &device, MissingPolicy::Strict)
            .expect("apply tensors to fresh Jepa");
        let round_trip_bytes = to_safetensors_bytes(&reloaded).expect("re-serialize reloaded Jepa");

        assert_eq!(bytes, round_trip_bytes);
    }

    #[test]
    fn apply_tensors_rejects_unknown_names() -> Result<(), Box<dyn std::error::Error>> {
        let device = NdArrayDevice::default();
        let model = Jepa::<CpuBackend>::init(JepaConfig::default(), &device)?;
        let bytes = to_safetensors_bytes(&model)?;
        let mut tensors = parse_safetensors_bytes(&bytes, Path::new("inline://unknown-names"))?;
        tensors.insert(
            "encoder.does_not_exist".to_owned(),
            LoadedTensor::F32 {
                shape: vec![1],
                values: vec![0.0],
            },
        );

        let fresh = Jepa::<CpuBackend>::init(JepaConfig::default(), &device)?;
        let Err(err) = apply_tensors_to_jepa(fresh, &tensors, &device, MissingPolicy::Strict)
        else {
            return Err("expected mapping error".into());
        };
        let ImportError::Mapping(messages) = err else {
            return Err(format!("expected Mapping error variant, got: {err:?}").into());
        };
        assert!(
            messages
                .iter()
                .any(|message| message.contains("encoder.does_not_exist")),
            "expected unknown-name error, got: {messages:?}"
        );
        Ok(())
    }
}
