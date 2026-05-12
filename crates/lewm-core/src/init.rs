//! Deterministic model initialization helpers.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Normal};

use crate::LewmCoreError;

/// RFC 0013 model-initialization RNG sub-stream name.
pub const MODEL_INIT_STREAM: &str = "rng:model_init";

const RFC_0013_STREAMS: &[&str] = &[
    "rng:data_shuffle",
    MODEL_INIT_STREAM,
    "rng:sigreg_sketch",
    "rng:dropout",
    "rng:cem",
    "rng:misc",
];

/// Opaque RNG state for deterministic model initialization.
#[derive(Debug, Clone)]
pub struct ModelInitRng {
    inner: ChaCha20Rng,
}

impl ModelInitRng {
    fn new(inner: ChaCha20Rng) -> Self {
        Self { inner }
    }
}

/// A deterministic initialized `f32` buffer with shape metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct InitTensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl InitTensor {
    /// Create a shaped initializer buffer after checking shape/product coherence.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidShape`] when `values.len()` does not
    /// match the product of `shape`, or [`LewmCoreError::InvalidInit`] when the
    /// shape itself is invalid.
    pub fn from_values(shape: &[usize], values: Vec<f32>) -> Result<Self, LewmCoreError> {
        let expected_len = element_count(shape)?;
        if values.len() != expected_len {
            return Err(LewmCoreError::InvalidShape {
                expected: vec![expected_len],
                found: vec![values.len()],
            });
        }

        Ok(Self {
            shape: shape.to_vec(),
            values,
        })
    }

    /// Return the tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the flat row-major values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Consume the tensor into its flat values.
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }

    /// Return the number of scalar elements.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Return `true` when the tensor has no scalar elements.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Derive a deterministic 32-byte seed for an RFC 0013 RNG sub-stream.
pub fn substream_seed(global: u64, name: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&global.to_le_bytes());
    hasher.update(b"::");
    hasher.update(name.as_bytes());
    *hasher.finalize().as_bytes()
}

/// Create a `ChaCha20Rng` for a named RFC 0013 sub-stream.
///
/// # Errors
///
/// Returns [`LewmCoreError::RngSubstream`] when `name` is not one of the
/// registered RFC 0013 stream names.
pub fn substream_rng(global: u64, name: &str) -> Result<ChaCha20Rng, LewmCoreError> {
    if !RFC_0013_STREAMS.contains(&name) {
        return Err(LewmCoreError::RngSubstream {
            name: name.to_owned(),
        });
    }

    Ok(ChaCha20Rng::from_seed(substream_seed(global, name)))
}

/// Create the model-initialization RNG for a global seed.
///
/// # Errors
///
/// This only fails if the RFC 0013 model-init stream constant is changed to an
/// unregistered name.
pub fn model_init_rng(global: u64) -> Result<ModelInitRng, LewmCoreError> {
    substream_rng(global, MODEL_INIT_STREAM).map(ModelInitRng::new)
}

/// Initialize a flat buffer with exact rejection-sampled truncated normal draws.
///
/// `clip` is the absolute bound. To truncate at `±2σ`, pass `clip = 2.0 * std`.
///
/// # Errors
///
/// Returns an error for invalid shape products, non-finite parameters,
/// non-positive `std`, or non-positive `clip`.
pub fn trunc_normal(
    shape: &[usize],
    std: f32,
    clip: f32,
    rng: &mut ModelInitRng,
) -> Result<InitTensor, LewmCoreError> {
    let len = element_count(shape)?;
    validate_normal_params(std, clip)?;
    let normal = Normal::<f32>::new(0.0, std).map_err(|err| LewmCoreError::InvalidInit {
        reason: format!("normal distribution rejected std={std}: {err}"),
    })?;
    let mut values = Vec::with_capacity(len);

    while values.len() < len {
        let value = normal.sample(&mut rng.inner);
        if value.abs() <= clip {
            values.push(value);
        }
    }

    InitTensor::from_values(shape, values)
}

/// Initialize a shaped buffer filled with zeros.
///
/// # Errors
///
/// Returns an error when `shape` is empty, contains a zero dimension, or its
/// element count overflows `usize`.
pub fn zeros(shape: &[usize]) -> Result<InitTensor, LewmCoreError> {
    fill(shape, 0.0)
}

/// Initialize a shaped buffer filled with ones.
///
/// # Errors
///
/// Returns an error when `shape` is empty, contains a zero dimension, or its
/// element count overflows `usize`.
pub fn ones(shape: &[usize]) -> Result<InitTensor, LewmCoreError> {
    fill(shape, 1.0)
}

fn fill(shape: &[usize], value: f32) -> Result<InitTensor, LewmCoreError> {
    let len = element_count(shape)?;
    InitTensor::from_values(shape, vec![value; len])
}

fn element_count(shape: &[usize]) -> Result<usize, LewmCoreError> {
    if shape.is_empty() {
        return Err(LewmCoreError::InvalidInit {
            reason: "shape must contain at least one dimension".to_owned(),
        });
    }

    shape.iter().try_fold(1usize, |acc, dim| {
        if *dim == 0 {
            return Err(LewmCoreError::InvalidInit {
                reason: "shape dimensions must be non-zero".to_owned(),
            });
        }

        acc.checked_mul(*dim)
            .ok_or_else(|| LewmCoreError::InvalidInit {
                reason: "shape element count overflowed usize".to_owned(),
            })
    })
}

fn validate_normal_params(std: f32, clip: f32) -> Result<(), LewmCoreError> {
    if !std.is_finite() || std <= 0.0 {
        return Err(LewmCoreError::InvalidInit {
            reason: format!("std must be finite and positive, got {std}"),
        });
    }

    if !clip.is_finite() || clip <= 0.0 {
        return Err(LewmCoreError::InvalidInit {
            reason: format!("clip must be finite and positive, got {clip}"),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn substream_seed_distinct() {
        let streams = [
            "rng:data_shuffle",
            MODEL_INIT_STREAM,
            "rng:sigreg_sketch",
            "rng:dropout",
            "rng:cem",
            "rng:misc",
        ];
        let seeds = streams
            .iter()
            .map(|stream| substream_seed(0, stream))
            .collect::<BTreeSet<_>>();

        assert_eq!(seeds.len(), streams.len());
        assert_eq!(
            substream_seed(0, MODEL_INIT_STREAM),
            substream_seed(0, MODEL_INIT_STREAM)
        );
        assert_ne!(
            substream_seed(0, MODEL_INIT_STREAM),
            substream_seed(1, MODEL_INIT_STREAM)
        );
    }

    #[test]
    fn model_init_rng_is_reproducible() {
        let mut left_rng = model_init_rng(7).expect("registered stream");
        let mut right_rng = model_init_rng(7).expect("registered stream");

        let left = trunc_normal(&[4, 8], 0.02, 0.04, &mut left_rng).expect("valid init");
        let right = trunc_normal(&[4, 8], 0.02, 0.04, &mut right_rng).expect("valid init");

        assert_eq!(left.shape(), &[4, 8]);
        assert_eq!(bits(left.values()), bits(right.values()));
    }

    #[test]
    fn model_init_rng_is_distinct_from_other_streams() {
        assert_ne!(
            substream_seed(0, MODEL_INIT_STREAM),
            substream_seed(0, "rng:dropout")
        );
    }

    #[test]
    fn trunc_normal_rejects_values_outside_clip() {
        let mut rng = model_init_rng(0).expect("registered stream");
        let tensor = trunc_normal(&[256], 0.02, 0.01, &mut rng).expect("valid init");

        assert!(tensor.values().iter().all(|value| value.abs() <= 0.01));
    }

    #[test]
    fn zeros_and_ones_preserve_shape_and_values() {
        let z = zeros(&[2, 3]).expect("valid shape");
        let o = ones(&[2, 3]).expect("valid shape");

        assert_eq!(z.shape(), &[2, 3]);
        assert_eq!(z.values(), &[0.0; 6]);
        assert_eq!(o.shape(), &[2, 3]);
        assert_eq!(o.values(), &[1.0; 6]);
    }

    #[test]
    fn invalid_init_requests_are_errors() {
        let mut rng = model_init_rng(0).expect("registered stream");

        assert!(zeros(&[]).is_err());
        assert!(ones(&[2, 0]).is_err());
        assert!(trunc_normal(&[2], 0.0, 0.04, &mut rng).is_err());
        assert!(trunc_normal(&[2], 0.02, f32::INFINITY, &mut rng).is_err());
        assert!(substream_rng(0, "rng:unknown").is_err());
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }
}
