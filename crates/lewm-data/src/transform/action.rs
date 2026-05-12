//! Action normalization and transform-stat safetensors persistence.

use std::path::Path;

use safetensors::tensor::{Dtype, SafeTensors, TensorView};

use crate::DataError;

const MIN_STD: f32 = 1e-6;
const PIXEL_CHANNELS: usize = 3;
const CONTENT_HASH_LEN: usize = 32;

/// Per-dimension action normalizer.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionNormalizer {
    /// Per-dimension action mean.
    pub mean: Vec<f32>,
    /// Per-dimension action standard deviation with tiny std values clamped to `1.0`.
    pub std: Vec<f32>,
}

impl ActionNormalizer {
    /// Build a normalizer, replacing every `std < 1e-6` with `1.0`.
    ///
    /// # Errors
    ///
    /// Returns an error if means/stds are empty, lengths differ, or any value is not finite.
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Result<Self, DataError> {
        if mean.is_empty() {
            return Err(DataError::InvalidTransform(
                "action normalizer must have at least one dimension".to_string(),
            ));
        }
        if mean.len() != std.len() {
            return Err(DataError::InvalidTransform(format!(
                "action mean has {} dims but std has {} dims",
                mean.len(),
                std.len()
            )));
        }
        for (dim, value) in mean.iter().enumerate() {
            if !value.is_finite() {
                return Err(DataError::InvalidTransform(format!(
                    "action mean[{dim}] must be finite"
                )));
            }
        }

        let mut sanitized_std = Vec::with_capacity(std.len());
        for (dim, value) in std.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(DataError::InvalidTransform(format!(
                    "action std[{dim}] must be finite"
                )));
            }
            sanitized_std.push(if value < MIN_STD { 1.0 } else { value });
        }

        Ok(Self {
            mean,
            std: sanitized_std,
        })
    }

    /// Number of action dimensions.
    #[must_use]
    pub fn action_dim(&self) -> usize {
        self.mean.len()
    }

    /// Map raw action values to normalized action values.
    ///
    /// # Errors
    ///
    /// Returns an error when `src.len()` is not a multiple of the action dimension.
    pub fn apply(&self, src: &[f32]) -> Result<Vec<f32>, DataError> {
        self.map_actions(src, |value, mean, std| (value - mean) / std)
    }

    /// Map normalized action values back to raw action values.
    ///
    /// # Errors
    ///
    /// Returns an error when `src.len()` is not a multiple of the action dimension.
    pub fn inverse(&self, src: &[f32]) -> Result<Vec<f32>, DataError> {
        self.map_actions(src, |value, mean, std| (value * std) + mean)
    }

    fn map_actions(
        &self,
        src: &[f32],
        map_value: impl Fn(f32, f32, f32) -> f32,
    ) -> Result<Vec<f32>, DataError> {
        let action_dim = self.action_dim();
        if src.len() % action_dim != 0 {
            return Err(DataError::InvalidTransform(format!(
                "action buffer length {} is not divisible by action_dim {action_dim}",
                src.len()
            )));
        }

        let mut output = Vec::with_capacity(src.len());
        for (index, value) in src.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(DataError::InvalidTransform(format!(
                    "action value at flat index {index} must be finite"
                )));
            }
            let dim = index % action_dim;
            output.push(map_value(value, self.mean[dim], self.std[dim]));
        }
        Ok(output)
    }
}

/// Persisted transform statistics stored in `stats.safetensors`.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformStats {
    /// Per-dimension raw action mean from the training split.
    pub action_mean: Vec<f32>,
    /// Per-dimension raw action std from the training split.
    pub action_std: Vec<f32>,
    /// Informational pixel mean; the runtime image preprocessor remains authoritative.
    pub pixel_mean: [f32; PIXEL_CHANNELS],
    /// Informational pixel std; the runtime image preprocessor remains authoritative.
    pub pixel_std: [f32; PIXEL_CHANNELS],
    /// Number of training samples used to compute the statistics.
    pub n_train_samples: i64,
    /// BLAKE3 hash of the underlying dataset bytes.
    pub content_hash: [u8; CONTENT_HASH_LEN],
}

impl TransformStats {
    /// Build validated transform statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if action statistics cannot construct an [`ActionNormalizer`],
    /// pixel statistics are non-finite, or `n_train_samples` is negative.
    pub fn new(
        action_mean: Vec<f32>,
        action_std: Vec<f32>,
        pixel_mean: [f32; PIXEL_CHANNELS],
        pixel_std: [f32; PIXEL_CHANNELS],
        n_train_samples: i64,
        content_hash: [u8; CONTENT_HASH_LEN],
    ) -> Result<Self, DataError> {
        let normalizer = ActionNormalizer::new(action_mean, action_std)?;
        validate_pixel_stats(&pixel_mean, &pixel_std)?;
        if n_train_samples < 0 {
            return Err(DataError::InvalidTransform(format!(
                "n_train_samples must be non-negative, found {n_train_samples}"
            )));
        }
        Ok(Self {
            action_mean: normalizer.mean,
            action_std: normalizer.std,
            pixel_mean,
            pixel_std,
            n_train_samples,
            content_hash,
        })
    }

    /// Create an action normalizer from persisted stats.
    ///
    /// # Errors
    ///
    /// Returns an error when the persisted action stats are invalid.
    pub fn action_normalizer(&self) -> Result<ActionNormalizer, DataError> {
        ActionNormalizer::new(self.action_mean.clone(), self.action_std.clone())
    }

    /// Write stats to a safetensors file using the RFC 0004 key layout.
    ///
    /// # Errors
    ///
    /// Returns an error when tensor views cannot be constructed or the file cannot be written.
    pub fn save_safetensors(&self, path: impl AsRef<Path>) -> Result<(), DataError> {
        let path = path.as_ref();
        let action_mean_bytes = f32_bytes(&self.action_mean);
        let action_std_bytes = f32_bytes(&self.action_std);
        let pixel_mean_bytes = f32_bytes(&self.pixel_mean);
        let pixel_std_bytes = f32_bytes(&self.pixel_std);
        let n_train_samples_bytes = self.n_train_samples.to_le_bytes();

        let tensors = vec![
            tensor_view(
                "action_mean",
                Dtype::F32,
                vec![self.action_mean.len()],
                &action_mean_bytes,
            )?,
            tensor_view(
                "action_std",
                Dtype::F32,
                vec![self.action_std.len()],
                &action_std_bytes,
            )?,
            tensor_view(
                "pixel_mean",
                Dtype::F32,
                vec![PIXEL_CHANNELS],
                &pixel_mean_bytes,
            )?,
            tensor_view(
                "pixel_std",
                Dtype::F32,
                vec![PIXEL_CHANNELS],
                &pixel_std_bytes,
            )?,
            tensor_view(
                "n_train_samples",
                Dtype::I64,
                Vec::new(),
                &n_train_samples_bytes,
            )?,
            tensor_view(
                "content_hash",
                Dtype::U8,
                vec![CONTENT_HASH_LEN],
                &self.content_hash,
            )?,
        ];

        let metadata = None;
        safetensors::serialize_to_file(tensors, &metadata, path)
            .map_err(|source| DataError::safetensors(path, source))
    }

    /// Load stats from a safetensors file using the RFC 0004 key layout.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, safetensors parsing fails,
    /// required tensors are missing, or tensor dtypes/shapes are invalid.
    pub fn load_safetensors(path: impl AsRef<Path>) -> Result<Self, DataError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| DataError::io(path, source))?;
        let tensors = SafeTensors::deserialize(&bytes)
            .map_err(|source| DataError::safetensors(path, source))?;

        let action_mean = read_f32_vector(&tensors, "action_mean")?;
        let action_std = read_f32_vector(&tensors, "action_std")?;
        let pixel_mean = read_f32_array::<PIXEL_CHANNELS>(&tensors, "pixel_mean")?;
        let pixel_std = read_f32_array::<PIXEL_CHANNELS>(&tensors, "pixel_std")?;
        let n_train_samples = read_i64_scalar(&tensors, "n_train_samples")?;
        let content_hash = read_u8_array::<CONTENT_HASH_LEN>(&tensors, "content_hash")?;

        Self::new(
            action_mean,
            action_std,
            pixel_mean,
            pixel_std,
            n_train_samples,
            content_hash,
        )
    }
}

fn validate_pixel_stats(
    pixel_mean: &[f32; PIXEL_CHANNELS],
    pixel_std: &[f32; PIXEL_CHANNELS],
) -> Result<(), DataError> {
    for channel in 0..PIXEL_CHANNELS {
        if !pixel_mean[channel].is_finite() {
            return Err(DataError::InvalidTransform(format!(
                "pixel_mean[{channel}] must be finite"
            )));
        }
        if !pixel_std[channel].is_finite() || pixel_std[channel] == 0.0 {
            return Err(DataError::InvalidTransform(format!(
                "pixel_std[{channel}] must be finite and non-zero"
            )));
        }
    }
    Ok(())
}

fn tensor_view<'a>(
    name: &'static str,
    dtype: Dtype,
    shape: Vec<usize>,
    data: &'a [u8],
) -> Result<(&'static str, TensorView<'a>), DataError> {
    let view = TensorView::new(dtype, shape, data).map_err(|source| {
        DataError::InvalidTransform(format!("could not build safetensors view {name}: {source}"))
    })?;
    Ok((name, view))
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn read_f32_vector(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>, DataError> {
    let tensor = tensors.tensor(name).map_err(|source| {
        DataError::InvalidTransform(format!("missing tensor {name}: {source}"))
    })?;
    if tensor.dtype() != Dtype::F32 || tensor.shape().len() != 1 {
        return Err(DataError::InvalidTransform(format!(
            "{name} must have dtype F32 and shape (A,), found dtype {:?} shape {:?}",
            tensor.dtype(),
            tensor.shape()
        )));
    }
    parse_f32_data(name, tensor.data())
}

fn read_f32_array<const N: usize>(
    tensors: &SafeTensors<'_>,
    name: &str,
) -> Result<[f32; N], DataError> {
    let values = read_f32_vector(tensors, name)?;
    values.try_into().map_err(|values: Vec<f32>| {
        DataError::InvalidTransform(format!(
            "{name} must have {N} values, found {}",
            values.len()
        ))
    })
}

fn read_i64_scalar(tensors: &SafeTensors<'_>, name: &str) -> Result<i64, DataError> {
    let tensor = tensors.tensor(name).map_err(|source| {
        DataError::InvalidTransform(format!("missing tensor {name}: {source}"))
    })?;
    if tensor.dtype() != Dtype::I64 || !tensor.shape().is_empty() {
        return Err(DataError::InvalidTransform(format!(
            "{name} must have dtype I64 and scalar shape, found dtype {:?} shape {:?}",
            tensor.dtype(),
            tensor.shape()
        )));
    }
    let bytes = tensor.data();
    let value_bytes: [u8; std::mem::size_of::<i64>()] = bytes.try_into().map_err(|_| {
        DataError::InvalidTransform(format!(
            "{name} must have {} bytes, found {}",
            std::mem::size_of::<i64>(),
            bytes.len()
        ))
    })?;
    Ok(i64::from_le_bytes(value_bytes))
}

fn read_u8_array<const N: usize>(
    tensors: &SafeTensors<'_>,
    name: &str,
) -> Result<[u8; N], DataError> {
    let tensor = tensors.tensor(name).map_err(|source| {
        DataError::InvalidTransform(format!("missing tensor {name}: {source}"))
    })?;
    if tensor.dtype() != Dtype::U8 || tensor.shape() != [N] {
        return Err(DataError::InvalidTransform(format!(
            "{name} must have dtype U8 and shape ({N},), found dtype {:?} shape {:?}",
            tensor.dtype(),
            tensor.shape()
        )));
    }
    let bytes = tensor.data();
    bytes.try_into().map_err(|_| {
        DataError::InvalidTransform(format!("{name} must have {N} bytes, found {}", bytes.len()))
    })
}

fn parse_f32_data(name: &str, data: &[u8]) -> Result<Vec<f32>, DataError> {
    if data.len() % std::mem::size_of::<f32>() != 0 {
        return Err(DataError::InvalidTransform(format!(
            "{name} byte length {} is not divisible by {}",
            data.len(),
            std::mem::size_of::<f32>()
        )));
    }
    data.chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| {
            let bytes: [u8; std::mem::size_of::<f32>()] = chunk.try_into().map_err(|_| {
                DataError::InvalidTransform(format!("{name} contains an invalid f32 byte chunk"))
            })?;
            Ok(f32::from_le_bytes(bytes))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn assert_close(left: f32, right: f32, tolerance: f32) {
        assert!(
            (left - right).abs() <= tolerance,
            "left={left}, right={right}, diff={}",
            (left - right).abs()
        );
    }

    #[test]
    fn action_normalize_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let action_norm = ActionNormalizer::new(vec![1.0, -2.0], vec![2.0, 4.0])?;
        let raw = vec![3.0, 2.0, -1.0, -10.0];

        let mapped = action_norm.apply(&raw)?;
        let restored = action_norm.inverse(&mapped)?;

        assert_eq!(mapped, vec![1.0, 1.0, -1.0, -2.0]);
        for (left, right) in restored.iter().zip(raw.iter()) {
            assert_close(*left, *right, 1e-6);
        }
        Ok(())
    }

    #[test]
    fn action_normalize_zero_std_replace() -> Result<(), Box<dyn std::error::Error>> {
        let action_norm = ActionNormalizer::new(vec![1.0, 2.0, 3.0], vec![0.0, 1e-7, 2.0])?;
        let mapped = action_norm.apply(&[2.0, 4.0, 7.0])?;

        assert_eq!(action_norm.std, vec![1.0, 1.0, 2.0]);
        assert_eq!(mapped, vec![1.0, 2.0, 2.0]);
        Ok(())
    }

    #[test]
    fn transform_stats_safetensors_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("stats.safetensors");
        let stats = TransformStats::new(
            vec![1.0, -2.0],
            vec![2.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            17,
            [7_u8; CONTENT_HASH_LEN],
        )?;

        stats.save_safetensors(&path)?;
        let loaded = TransformStats::load_safetensors(&path)?;

        assert_eq!(loaded, stats);
        assert_eq!(loaded.action_normalizer()?.std, vec![2.0, 1.0]);
        Ok(())
    }

    proptest! {
        #[test]
        fn action_normalizer_roundtrip_property(
            (mean, std, raw) in (1_usize..8).prop_flat_map(|action_dim| {
                (
                    prop::collection::vec(-1.0_f32..1.0, action_dim),
                    prop::collection::vec(0.5_f32..2.0, action_dim),
                    prop::collection::vec(-1.0_f32..1.0, action_dim),
                )
            })
        ) {
            let normalizer = ActionNormalizer::new(mean, std)
                .map_err(|err| TestCaseError::fail(err.to_string()))?;
            let normalized = normalizer.apply(&raw)
                .map_err(|err| TestCaseError::fail(err.to_string()))?;
            let restored = normalizer.inverse(&normalized)
                .map_err(|err| TestCaseError::fail(err.to_string()))?;

            prop_assert_eq!(restored.len(), raw.len());
            for (left, right) in restored.iter().zip(raw.iter()) {
                prop_assert!((*left - *right).abs() <= 1e-6);
            }
        }
    }
}
