//! Backend-neutral tensor helpers used by the `LeWM` model.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::LewmCoreError;

/// The RFC 0002 position-embedding interpolation mode.
pub const BICUBIC_ALIGN_CORNERS: bool = false;

const GELU_TANH_COEFF: f32 = 0.044_715;
const CUBIC_COEFF: f64 = -0.75;

static CAUSAL_MASK_CACHE: OnceLock<Mutex<HashMap<CausalMaskKey, Arc<[f32]>>>> = OnceLock::new();

/// Device identity used to key backend-specific tensor caches.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceKey(String);

impl DeviceKey {
    /// Construct a cache key for a backend device.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidTensorOp`] when the device name is empty.
    pub fn new(name: impl Into<String>) -> Result<Self, LewmCoreError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(LewmCoreError::InvalidTensorOp {
                reason: "device name must not be empty".to_owned(),
            });
        }

        Ok(Self(name))
    }

    /// Return the canonical CPU cache key.
    pub fn cpu() -> Self {
        Self("cpu".to_owned())
    }

    /// Return the backend device name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for DeviceKey {
    fn default() -> Self {
        Self::cpu()
    }
}

/// A cached row-major causal attention mask.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalMask {
    seq_len: usize,
    device: DeviceKey,
    values: Arc<[f32]>,
}

impl CausalMask {
    /// Return the sequence length for both mask axes.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Return the device key used for cache lookup.
    pub fn device(&self) -> &DeviceKey {
        &self.device
    }

    /// Return row-major F32 mask values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Return `true` if two masks reuse the same cached allocation.
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.values, &other.values)
    }
}

/// A `(1, n_patch + 1, dim)` position embedding flattened in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionEmbedding {
    n_patch: usize,
    dim: usize,
    values: Vec<f32>,
}

impl PositionEmbedding {
    /// Build a shaped position embedding after validating the patch grid.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidTensorOp`] when `n_patch` is not a
    /// non-empty square grid, `dim` is zero, or `values` has the wrong length.
    pub fn from_values(
        n_patch: usize,
        dim: usize,
        values: Vec<f32>,
    ) -> Result<Self, LewmCoreError> {
        let expected_len = position_embedding_len(n_patch, dim)?;
        let _side = square_side(n_patch)?;

        if dim == 0 {
            return Err(LewmCoreError::InvalidTensorOp {
                reason: "position embedding dim must be non-zero".to_owned(),
            });
        }

        if values.len() != expected_len {
            return Err(LewmCoreError::InvalidShape {
                expected: vec![1, n_patch + 1, dim],
                found: vec![values.len()],
            });
        }

        Ok(Self {
            n_patch,
            dim,
            values,
        })
    }

    /// Return the number of patch tokens, excluding the class token.
    pub fn n_patch(&self) -> usize {
        self.n_patch
    }

    /// Return the embedding feature dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the flattened row-major embedding values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Consume the embedding into its flattened values.
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CausalMaskKey {
    seq_len: usize,
    device: DeviceKey,
}

/// Compute the tanh-approximate GELU used by the `ViT` encoder MLP.
pub fn gelu_tanh_approx(x: f32) -> f32 {
    let inner = (std::f32::consts::FRAC_2_SQRT_PI / std::f32::consts::SQRT_2)
        * (x + GELU_TANH_COEFF * x.powi(3));
    0.5 * x * (1.0 + inner.tanh())
}

/// Compute the erf-based GELU used by projection MLP heads.
pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
}

/// Build or fetch a cached upper-triangular F32 causal attention mask.
///
/// Values above the diagonal are `-inf`; all other entries are `0.0`.
///
/// # Errors
///
/// Returns [`LewmCoreError::InvalidTensorOp`] when `seq_len` is zero or the
/// square mask size overflows `usize`.
pub fn build_causal_mask(seq_len: usize, device: &DeviceKey) -> Result<CausalMask, LewmCoreError> {
    if seq_len == 0 {
        return Err(LewmCoreError::InvalidTensorOp {
            reason: "causal mask seq_len must be non-zero".to_owned(),
        });
    }

    let _len = seq_len
        .checked_mul(seq_len)
        .ok_or_else(|| LewmCoreError::InvalidTensorOp {
            reason: "causal mask element count overflowed usize".to_owned(),
        })?;
    let key = CausalMaskKey {
        seq_len,
        device: device.clone(),
    };
    let cache = CAUSAL_MASK_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| LewmCoreError::InvalidTensorOp {
        reason: "causal mask cache lock was poisoned".to_owned(),
    })?;
    let values = cache
        .entry(key)
        .or_insert_with(|| Arc::<[f32]>::from(build_causal_mask_values(seq_len)))
        .clone();

    Ok(CausalMask {
        seq_len,
        device: device.clone(),
        values,
    })
}

/// Bicubic-interpolate a `ViT` position embedding to `n_patch` patch tokens.
///
/// `align_corners` should be [`BICUBIC_ALIGN_CORNERS`] for RFC 0002 parity.
///
/// # Errors
///
/// Returns [`LewmCoreError::InvalidTensorOp`] when the source or target patch
/// count is not a non-empty square grid.
pub fn interpolate_pos_embed(
    pos: &PositionEmbedding,
    n_patch: usize,
    align_corners: bool,
) -> Result<PositionEmbedding, LewmCoreError> {
    let source_side = square_side(pos.n_patch)?;
    let target_side = square_side(n_patch)?;

    if pos.n_patch == n_patch {
        return Ok(pos.clone());
    }

    let mut values = Vec::with_capacity(position_embedding_len(n_patch, pos.dim)?);
    values.extend_from_slice(&pos.values[..pos.dim]);

    for out_y in 0..target_side {
        for out_x in 0..target_side {
            interpolate_patch(
                pos,
                source_side,
                target_side,
                out_y,
                out_x,
                align_corners,
                &mut values,
            );
        }
    }

    PositionEmbedding::from_values(n_patch, pos.dim, values)
}

fn build_causal_mask_values(seq_len: usize) -> Vec<f32> {
    let mut values = vec![0.0; seq_len * seq_len];
    for row in 0..seq_len {
        for col in (row + 1)..seq_len {
            values[row * seq_len + col] = f32::NEG_INFINITY;
        }
    }
    values
}

fn square_side(n_patch: usize) -> Result<usize, LewmCoreError> {
    if n_patch == 0 {
        return Err(LewmCoreError::InvalidTensorOp {
            reason: "patch count must be non-zero".to_owned(),
        });
    }

    let mut side = 1usize;
    while side
        .checked_mul(side)
        .is_some_and(|square| square < n_patch)
    {
        side += 1;
    }

    if side.checked_mul(side) != Some(n_patch) {
        return Err(LewmCoreError::InvalidTensorOp {
            reason: format!("patch count must form a square grid, got {n_patch}"),
        });
    }

    Ok(side)
}

fn position_embedding_len(n_patch: usize, dim: usize) -> Result<usize, LewmCoreError> {
    n_patch
        .checked_add(1)
        .and_then(|tokens| tokens.checked_mul(dim))
        .ok_or_else(|| LewmCoreError::InvalidTensorOp {
            reason: "position embedding length overflowed usize".to_owned(),
        })
}

#[allow(clippy::cast_possible_truncation)]
fn interpolate_patch(
    pos: &PositionEmbedding,
    source_side: usize,
    target_side: usize,
    out_y: usize,
    out_x: usize,
    align_corners: bool,
    values: &mut Vec<f32>,
) {
    let y_source = source_index(source_side, target_side, out_y, align_corners);
    let x_source = source_index(source_side, target_side, out_x, align_corners);
    let y_base = y_source.floor() as isize;
    let x_base = x_source.floor() as isize;
    let y_weights = cubic_weights(y_source, y_base);
    let x_weights = cubic_weights(x_source, x_base);

    for dim in 0..pos.dim {
        let mut value = 0.0f64;
        for (y_offset, y_weight) in y_weights.iter().enumerate() {
            let source_y = clamp_index(y_base + offset_index(y_offset), source_side);
            for (x_offset, x_weight) in x_weights.iter().enumerate() {
                let source_x = clamp_index(x_base + offset_index(x_offset), source_side);
                let patch_idx = source_y * source_side + source_x;
                let value_idx = (patch_idx + 1) * pos.dim + dim;
                value += f64::from(pos.values[value_idx]) * y_weight * x_weight;
            }
        }
        values.push(value as f32);
    }
}

#[allow(clippy::cast_precision_loss)]
fn source_index(input: usize, output: usize, index: usize, align_corners: bool) -> f64 {
    if align_corners {
        if output == 1 {
            0.0
        } else {
            index as f64 * (input - 1) as f64 / (output - 1) as f64
        }
    } else {
        (index as f64 + 0.5) * input as f64 / output as f64 - 0.5
    }
}

#[allow(clippy::cast_precision_loss)]
fn cubic_weights(source: f64, base: isize) -> [f64; 4] {
    [
        cubic_weight(source - (base - 1) as f64),
        cubic_weight(source - base as f64),
        cubic_weight(source - (base + 1) as f64),
        cubic_weight(source - (base + 2) as f64),
    ]
}

fn cubic_weight(distance: f64) -> f64 {
    let x = distance.abs();
    if x <= 1.0 {
        (CUBIC_COEFF + 2.0) * x.powi(3) - (CUBIC_COEFF + 3.0) * x.powi(2) + 1.0
    } else if x < 2.0 {
        CUBIC_COEFF * x.powi(3) - 5.0 * CUBIC_COEFF * x.powi(2) + 8.0 * CUBIC_COEFF * x
            - 4.0 * CUBIC_COEFF
    } else {
        0.0
    }
}

fn offset_index(offset: usize) -> isize {
    match offset {
        0 => -1,
        1 => 0,
        2 => 1,
        _ => 2,
    }
}

fn clamp_index(index: isize, side: usize) -> usize {
    if index < 0 {
        0
    } else {
        usize::try_from(index).map_or(side - 1, |idx| idx.min(side - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_variants_match_reference_values() {
        const INPUTS: &[f32] = &[-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
        const TANH_EXPECTED: &[f64] = &[
            -0.003_637_392_082,
            -0.158_808_009_392,
            -0.154_285_990_175,
            0.0,
            0.345_714_009_825,
            0.841_191_990_608,
            2.996_362_607_918,
        ];
        const ERF_EXPECTED: &[f64] = &[
            -0.004_049_694_095,
            -0.158_655_253_931,
            -0.154_268_769_363,
            0.0,
            0.345_731_230_637,
            0.841_344_746_069,
            2.995_950_305_905,
        ];

        for ((input, tanh_expected), erf_expected) in INPUTS
            .iter()
            .zip(TANH_EXPECTED.iter())
            .zip(ERF_EXPECTED.iter())
        {
            assert_close(f64::from(gelu_tanh_approx(*input)), *tanh_expected, 1.0e-6);
            assert_close(f64::from(gelu_erf(*input)), *erf_expected, 1.0e-6);
        }
    }

    #[test]
    fn causal_mask_is_upper_triangular_negative_infinity_and_cached() {
        let device = DeviceKey::cpu();
        let mask = build_causal_mask(4, &device).expect("valid mask");
        let cached = build_causal_mask(4, &device).expect("valid mask");

        assert_eq!(mask.seq_len(), 4);
        assert_eq!(mask.device(), &device);
        assert!(mask.shares_storage_with(&cached));
        assert_eq!(
            mask.values(),
            &[
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        );
    }

    #[test]
    fn interpolate_pos_embed_noops_when_patch_count_matches() {
        let pos = PositionEmbedding::from_values(4, 2, (0u16..10).map(f32::from).collect())
            .expect("valid embedding");

        let interpolated =
            interpolate_pos_embed(&pos, 4, BICUBIC_ALIGN_CORNERS).expect("valid interpolation");

        assert_eq!(interpolated, pos);
    }

    #[test]
    #[allow(
        clippy::excessive_precision,
        clippy::too_many_lines,
        clippy::unreadable_literal
    )]
    fn interpolate_pos_embed_matches_pytorch_bicubic_boundary_case() {
        const EXPECTED: &[f32] = &[
            -99.000000000,
            0.093967013,
            0.217390046,
            0.331394672,
            0.444010437,
            0.558015049,
            0.676186383,
            0.799218714,
            0.917390049,
            1.031394601,
            1.144010425,
            1.258015037,
            1.381438017,
            1.821889400,
            1.945312500,
            2.059317112,
            2.171932936,
            2.285937548,
            2.404108763,
            2.527141094,
            2.645312548,
            2.759317160,
            2.871932983,
            2.985937357,
            3.109360456,
            3.417954206,
            3.541377544,
            3.655381918,
            3.767997742,
            3.882002115,
            4.000173569,
            4.123206139,
            4.241377354,
            4.355381966,
            4.467997551,
            4.582002640,
            4.705425262,
            4.994574547,
            5.117997646,
            5.232002258,
            5.344617844,
            5.458622456,
            5.576794147,
            5.699826717,
            5.817997932,
            5.932002068,
            6.044618130,
            6.158622742,
            6.282045841,
            6.590639591,
            6.714062214,
            6.828067303,
            6.940682888,
            7.054687500,
            7.172858715,
            7.295891285,
            7.414062500,
            7.528067112,
            7.640683174,
            7.754687309,
            7.878110886,
            8.245037079,
            8.368460655,
            8.482465744,
            8.595081329,
            8.709085464,
            8.827257156,
            8.950289726,
            9.068460464,
            9.182465553,
            9.295081139,
            9.409086227,
            9.532508850,
            9.967491150,
            10.090913773,
            10.204918861,
            10.317534447,
            10.431539536,
            10.549710274,
            10.672742844,
            10.790914536,
            10.904918671,
            11.017534256,
            11.131539345,
            11.254962921,
            11.621889114,
            11.745312691,
            11.859316826,
            11.971933365,
            12.085937500,
            12.204109192,
            12.327140808,
            12.445312500,
            12.559317589,
            12.671933174,
            12.785937309,
            12.909359932,
            13.217954636,
            13.341377258,
            13.455382347,
            13.567996979,
            13.682002068,
            13.800173759,
            13.923206329,
            14.041377068,
            14.155382156,
            14.267997742,
            14.382002831,
            14.505425453,
            14.794574738,
            14.917998314,
            15.032002449,
            15.144618034,
            15.258622169,
            15.376793861,
            15.499826431,
            15.617998123,
            15.732001305,
            15.844617844,
            15.958622932,
            16.082046509,
            16.390638351,
            16.514062881,
            16.628067017,
            16.740684509,
            16.854686737,
            16.972858429,
            17.095891953,
            17.214063644,
            17.328067780,
            17.440681458,
            17.554687500,
            17.678110123,
            18.118562698,
            18.241983414,
            18.355989456,
            18.468605042,
            18.582611084,
            18.700780869,
            18.823812485,
            18.941984177,
            19.055990219,
            19.168605804,
            19.282609940,
            19.406032562,
        ];
        let values = std::iter::once(-99.0)
            .chain((0u16..196).map(|value| f32::from(value) / 10.0))
            .collect::<Vec<_>>();
        let pos = PositionEmbedding::from_values(196, 1, values).expect("valid embedding");

        let interpolated =
            interpolate_pos_embed(&pos, 144, BICUBIC_ALIGN_CORNERS).expect("valid interpolation");

        assert_eq!(interpolated.n_patch(), 144);
        assert_eq!(interpolated.dim(), 1);
        assert_eq!(interpolated.values().len(), EXPECTED.len());
        for (actual, expected) in interpolated.values().iter().zip(EXPECTED.iter()) {
            assert_close(f64::from(*actual), f64::from(*expected), 1.0e-6);
        }
    }

    #[test]
    fn tensor_ops_reject_invalid_shapes() {
        assert!(DeviceKey::new(" ").is_err());
        assert!(build_causal_mask(0, &DeviceKey::cpu()).is_err());
        assert!(PositionEmbedding::from_values(3, 1, vec![0.0; 4]).is_err());

        let pos = PositionEmbedding::from_values(4, 1, vec![0.0; 5]).expect("valid embedding");
        assert!(interpolate_pos_embed(&pos, 3, BICUBIC_ALIGN_CORNERS).is_err());
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}
