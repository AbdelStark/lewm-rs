//! `SIGReg` loss from RFC 0003.

use burn::module::RunningState;
use burn::tensor::{DType, Tensor, TensorData, backend::Backend};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Normal};

use crate::LewmCoreError;

/// Default number of random projection directions.
pub const DEFAULT_SIGREG_NUM_PROJ: usize = 1024;

/// Default number of frequency knots on `[0, 3]`.
pub const DEFAULT_SIGREG_KNOTS: usize = 17;

/// Default maximum frequency.
pub const DEFAULT_SIGREG_T_MAX: f32 = 3.0;

const MAX_ZERO_NORM_RESAMPLES: usize = 32;

/// `SIGReg` module with precomputed Epps-Pulley constants.
#[derive(burn::module::Module, Debug)]
pub struct SigReg<B: Backend> {
    consts: SigRegConsts<B>,
}

/// Precomputed `SIGReg` frequency, target-CF, window, and quadrature constants.
#[derive(burn::module::Module, Debug)]
pub struct SigRegConsts<B: Backend> {
    t_grid: RunningState<Tensor<B, 1>>,
    phi: RunningState<Tensor<B, 1>>,
    window: RunningState<Tensor<B, 1>>,
    trap: RunningState<Tensor<B, 1>>,
    #[module(skip)]
    num_proj: usize,
    #[module(skip)]
    knots: usize,
    #[module(skip)]
    t_max: f32,
}

impl<B: Backend> SigReg<B> {
    /// Initialize `SIGReg` with the RFC 0003 default hyperparameters.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidTensorOp`] when the fixed hyperparameters
    /// are invalid. This should only fail if the default constants are edited.
    pub fn init(device: &B::Device) -> Result<Self, LewmCoreError> {
        Self::init_with_config(
            DEFAULT_SIGREG_NUM_PROJ,
            DEFAULT_SIGREG_KNOTS,
            DEFAULT_SIGREG_T_MAX,
            device,
        )
    }

    /// Initialize `SIGReg` with explicit sketch and quadrature settings.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidTensorOp`] when `num_proj` is zero,
    /// `knots < 2`, or `t_max` is not finite and positive.
    pub fn init_with_config(
        num_proj: usize,
        knots: usize,
        t_max: f32,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        Ok(Self {
            consts: SigRegConsts::new(num_proj, knots, t_max, device)?,
        })
    }

    /// Return the immutable `SIGReg` constants.
    #[must_use]
    pub fn consts(&self) -> &SigRegConsts<B> {
        &self.consts
    }

    /// Sample `P` from the caller-owned `rng:sigreg_sketch` stream.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidTensorOp`] when `dim` is zero or the
    /// projection shape overflows.
    pub fn sample_projection(
        &self,
        dim: usize,
        rng: &mut ChaCha20Rng,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>, LewmCoreError> {
        sample_sigreg_projection(self.consts.num_proj, dim, rng, device)
    }

    /// Compute `L_sigreg(z)` using a newly sampled projection sketch.
    ///
    /// `rng` must be the run's `rng:sigreg_sketch` sub-stream owner.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidTensorOp`] for zero dimensions or
    /// projection-sampling failures.
    pub fn forward(
        &self,
        embeddings: Tensor<B, 3>,
        rng: &mut ChaCha20Rng,
    ) -> Result<Tensor<B, 1>, LewmCoreError> {
        let [_, _, dim] = validate_sigreg_input(&embeddings)?;
        let projection = self.sample_projection(dim, rng, &embeddings.device())?;
        self.forward_with_projection(embeddings, projection)
    }

    /// Compute `L_sigreg(z)` with an explicit projection matrix.
    ///
    /// This is the parity-test path: it isolates the Epps-Pulley arithmetic from
    /// RNG compatibility by letting callers pass the fixed sketch `P`.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidShape`] when `projection` is not
    /// `(num_proj, D)`, or [`LewmCoreError::InvalidTensorOp`] for zero input
    /// dimensions.
    pub fn forward_with_projection(
        &self,
        embeddings: Tensor<B, 3>,
        projection: Tensor<B, 2>,
    ) -> Result<Tensor<B, 1>, LewmCoreError> {
        let [batch, steps, dim] = validate_sigreg_input(&embeddings)?;
        validate_projection_shape(&projection, self.consts.num_proj, dim)?;

        let flattened_count =
            batch
                .checked_mul(steps)
                .ok_or_else(|| LewmCoreError::InvalidTensorOp {
                    reason: "SIGReg flattened batch element count overflowed usize".to_owned(),
                })?;

        let embeddings = embeddings.cast(DType::F32).reshape([flattened_count, dim]);
        let projection = projection.cast(DType::F32);
        let projected = embeddings.matmul(projection.transpose()); // (N, K)

        let frequencies = self.consts.t_grid().reshape([self.consts.knots, 1, 1]);
        let arg = frequencies * projected.transpose().unsqueeze_dim::<3>(0); // (J, K, N)
        let cos_stats = arg.clone().cos().mean_dim(2); // (J, K, 1)
        let sin_stats = arg.sin().mean_dim(2); // (J, K, 1)

        let phi = self.consts.phi().reshape([self.consts.knots, 1, 1]);
        let real_residual = cos_stats - phi;
        let residual = real_residual.clone() * real_residual + sin_stats.clone() * sin_stats; // (J, K, 1)

        let weights =
            (self.consts.window() * self.consts.trap()).reshape([self.consts.knots, 1, 1]);
        Ok((residual * weights).sum_dim(0).mean())
    }
}

impl<B: Backend> SigRegConsts<B> {
    /// Build `SIGReg` constants for `num_proj`, `knots`, and `t_max`.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidTensorOp`] when `num_proj` is zero,
    /// `knots < 2`, or `t_max` is not finite and positive.
    pub fn new(
        num_proj: usize,
        knots: usize,
        t_max: f32,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        validate_constants(num_proj, knots, t_max)?;

        let t_grid = build_t_grid(knots, t_max);
        let phi = build_phi(&t_grid);
        let trap = build_trap(knots, t_max);

        Ok(Self {
            t_grid: RunningState::new(tensor_from_values(t_grid, [knots], device)),
            phi: RunningState::new(tensor_from_values(phi.clone(), [knots], device)),
            window: RunningState::new(tensor_from_values(phi, [knots], device)),
            trap: RunningState::new(tensor_from_values(trap, [knots], device)),
            num_proj,
            knots,
            t_max,
        })
    }

    /// Return the number of projection directions.
    #[must_use]
    pub fn num_proj(&self) -> usize {
        self.num_proj
    }

    /// Return the number of frequency knots.
    #[must_use]
    pub fn knots(&self) -> usize {
        self.knots
    }

    /// Return the maximum frequency.
    #[must_use]
    pub fn t_max(&self) -> f32 {
        self.t_max
    }

    /// Return the RFC 0003 frequency grid tensor.
    #[must_use]
    pub fn t_grid(&self) -> Tensor<B, 1> {
        self.t_grid.value()
    }

    /// Return the standard-normal characteristic-function values.
    #[must_use]
    pub fn phi(&self) -> Tensor<B, 1> {
        self.phi.value()
    }

    /// Return the Gaussian window values.
    #[must_use]
    pub fn window(&self) -> Tensor<B, 1> {
        self.window.value()
    }

    /// Return the trapezoid-rule weights.
    #[must_use]
    pub fn trap(&self) -> Tensor<B, 1> {
        self.trap.value()
    }
}

/// Sample unit-norm `SIGReg` projection rows from `rng:sigreg_sketch`.
///
/// # Errors
///
/// Returns [`LewmCoreError::InvalidTensorOp`] when `num_proj` or `dim` is zero,
/// the projection shape overflows, or a row repeatedly samples zero norm.
pub fn sample_sigreg_projection<B: Backend>(
    num_proj: usize,
    dim: usize,
    rng: &mut ChaCha20Rng,
    device: &B::Device,
) -> Result<Tensor<B, 2>, LewmCoreError> {
    let values = sample_sigreg_projection_values(num_proj, dim, rng)?;
    Ok(tensor_from_values(values, [num_proj, dim], device))
}

fn validate_constants(num_proj: usize, knots: usize, t_max: f32) -> Result<(), LewmCoreError> {
    if num_proj == 0 {
        return Err(LewmCoreError::InvalidTensorOp {
            reason: "SIGReg num_proj must be non-zero".to_owned(),
        });
    }

    if knots < 2 {
        return Err(LewmCoreError::InvalidTensorOp {
            reason: "SIGReg knots must be at least 2".to_owned(),
        });
    }

    if !t_max.is_finite() || t_max <= 0.0 {
        return Err(LewmCoreError::InvalidTensorOp {
            reason: format!("SIGReg t_max must be finite and positive, got {t_max}"),
        });
    }

    Ok(())
}

fn validate_sigreg_input<B: Backend>(z: &Tensor<B, 3>) -> Result<[usize; 3], LewmCoreError> {
    let dims = z.dims();
    if dims.contains(&0) {
        return Err(LewmCoreError::InvalidTensorOp {
            reason: format!("SIGReg input dimensions must be non-zero, got {dims:?}"),
        });
    }

    Ok(dims)
}

fn validate_projection_shape<B: Backend>(
    projection: &Tensor<B, 2>,
    num_proj: usize,
    dim: usize,
) -> Result<(), LewmCoreError> {
    let dims = projection.dims();
    if dims != [num_proj, dim] {
        return Err(LewmCoreError::InvalidShape {
            expected: vec![num_proj, dim],
            found: dims.to_vec(),
        });
    }

    Ok(())
}

fn sample_sigreg_projection_values(
    num_proj: usize,
    dim: usize,
    rng: &mut ChaCha20Rng,
) -> Result<Vec<f32>, LewmCoreError> {
    if num_proj == 0 || dim == 0 {
        return Err(LewmCoreError::InvalidTensorOp {
            reason: format!("SIGReg projection shape must be non-zero, got [{num_proj}, {dim}]"),
        });
    }

    let len = num_proj
        .checked_mul(dim)
        .ok_or_else(|| LewmCoreError::InvalidTensorOp {
            reason: "SIGReg projection element count overflowed usize".to_owned(),
        })?;
    let normal = Normal::<f32>::new(0.0, 1.0).map_err(|err| LewmCoreError::InvalidTensorOp {
        reason: format!("SIGReg projection normal distribution failed: {err}"),
    })?;
    let mut values = vec![0.0; len];

    for row in values.chunks_exact_mut(dim) {
        sample_unit_row(row, normal, rng)?;
    }

    Ok(values)
}

fn sample_unit_row(
    row: &mut [f32],
    normal: Normal<f32>,
    rng: &mut ChaCha20Rng,
) -> Result<(), LewmCoreError> {
    for _ in 0..MAX_ZERO_NORM_RESAMPLES {
        let mut norm_sq = 0.0_f64;
        for value in row.iter_mut() {
            let draw = normal.sample(rng);
            *value = draw;
            norm_sq += f64::from(draw) * f64::from(draw);
        }

        if norm_sq.is_finite() && norm_sq > 0.0 {
            #[allow(clippy::cast_possible_truncation)]
            let inv_norm = (1.0 / norm_sq.sqrt()) as f32;
            for value in row {
                *value *= inv_norm;
            }
            return Ok(());
        }
    }

    Err(LewmCoreError::InvalidTensorOp {
        reason: "SIGReg sampled a zero-norm projection row too many times".to_owned(),
    })
}

fn build_t_grid(knots: usize, t_max: f32) -> Vec<f32> {
    let step = t_max / usize_to_f32(knots - 1);
    (0..knots).map(|index| usize_to_f32(index) * step).collect()
}

fn build_phi(t_grid: &[f32]) -> Vec<f32> {
    t_grid
        .iter()
        .map(|t| (-0.5 * t * t).exp())
        .collect::<Vec<_>>()
}

fn build_trap(knots: usize, t_max: f32) -> Vec<f32> {
    let step = t_max / usize_to_f32(knots - 1);
    let mut trap = vec![step; knots];
    trap[0] = step / 2.0;
    trap[knots - 1] = step / 2.0;
    trap
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f32(value: usize) -> f32 {
    value as f32
}

fn tensor_from_values<B: Backend, const D: usize>(
    values: Vec<f32>,
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::<B, D>::from_data(TensorData::new(values, shape), device).cast(DType::F32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::AutodiffBackend;

    type CpuBackend = burn_ndarray::NdArray<f32>;
    type CpuAutodiffBackend = burn_autodiff::Autodiff<CpuBackend>;

    #[derive(Clone, Copy)]
    struct ManualSigRegInput<'a> {
        embeddings: &'a [f32],
        batch: usize,
        steps: usize,
        dim: usize,
        projection: &'a [f32],
        num_proj: usize,
        knots: usize,
        t_max: f32,
    }

    #[test]
    fn sigreg_constants_match_rfc_values() -> Result<(), LewmCoreError> {
        let device = burn_ndarray::NdArrayDevice::default();
        let sigreg = SigReg::<CpuBackend>::init(&device)?;
        let consts = sigreg.consts();

        assert_eq!(consts.num_proj(), DEFAULT_SIGREG_NUM_PROJ);
        assert_eq!(consts.knots(), DEFAULT_SIGREG_KNOTS);
        assert_close(consts.t_max(), DEFAULT_SIGREG_T_MAX, 0.0);

        let t_grid = tensor_values(&consts.t_grid());
        let phi = tensor_values(&consts.phi());
        let window = tensor_values(&consts.window());
        let trap = tensor_values(&consts.trap());

        for (index, t) in t_grid.iter().copied().enumerate() {
            let expected_t = usize_to_f32(index) * 0.1875;
            assert_close(t, expected_t, 1.0e-7);
            let expected_phi = (-0.5 * expected_t * expected_t).exp();
            assert_close(phi[index], expected_phi, 1.0e-7);
            assert_close(window[index], expected_phi, 1.0e-7);
        }

        assert_close(trap[0], 0.09375, 1.0e-7);
        assert_close(trap[DEFAULT_SIGREG_KNOTS - 1], 0.09375, 1.0e-7);
        assert!(
            trap[1..DEFAULT_SIGREG_KNOTS - 1]
                .iter()
                .all(|weight| (*weight - 0.1875).abs() <= 1.0e-7)
        );
        Ok(())
    }

    #[test]
    fn sigreg_projection_rows_are_unit_norm() -> Result<(), LewmCoreError> {
        let device = burn_ndarray::NdArrayDevice::default();
        let mut rng = crate::substream_rng(0, crate::rng::SIGREG_SKETCH_STREAM)?;
        let projection = sample_sigreg_projection::<CpuBackend>(8, 5, &mut rng, &device)?;
        let values = tensor_values(&projection);

        for row in values.chunks_exact(5) {
            let norm_sq = row.iter().map(|value| value * value).sum::<f32>();
            assert_close(norm_sq, 1.0, 1.0e-6);
        }
        Ok(())
    }

    #[test]
    fn sigreg_forward_matches_manual_fixed_projection() -> Result<(), LewmCoreError> {
        let device = burn_ndarray::NdArrayDevice::default();
        let sigreg = SigReg::<CpuBackend>::init_with_config(2, 3, 2.0, &device)?;
        let z_values = vec![
            0.0, 0.5, //
            1.0, -0.5, //
            0.25, 0.75, //
            -1.0, 0.25,
        ];
        let projection_values = vec![
            1.0, 0.0, //
            0.0, 1.0,
        ];
        let z = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(z_values.clone(), [2, 2, 2]),
            &device,
        );
        let projection = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(projection_values.clone(), [2, 2]),
            &device,
        );

        let loss_tensor = sigreg.forward_with_projection(z, projection)?;
        let loss = scalar(&loss_tensor);
        let expected = manual_sigreg(ManualSigRegInput {
            embeddings: &z_values,
            batch: 2,
            steps: 2,
            dim: 2,
            projection: &projection_values,
            num_proj: 2,
            knots: 3,
            t_max: 2.0,
        });

        assert_close(loss, expected, 1.0e-6);
        Ok(())
    }

    #[test]
    fn sigreg_rng_determinism() -> Result<(), LewmCoreError> {
        let device = burn_ndarray::NdArrayDevice::default();
        let sigreg = SigReg::<CpuBackend>::init_with_config(16, 5, 3.0, &device)?;
        let z = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(
                (0_u16..24).map(|idx| f32::from(idx) / 17.0).collect(),
                [2, 3, 4],
            ),
            &device,
        );

        let mut left_rng = crate::substream_rng(7, crate::rng::SIGREG_SKETCH_STREAM)?;
        let mut right_rng = crate::substream_rng(7, crate::rng::SIGREG_SKETCH_STREAM)?;
        let left_tensor = sigreg.forward(z.clone(), &mut left_rng)?;
        let right_tensor = sigreg.forward(z, &mut right_rng)?;
        let left = scalar(&left_tensor);
        let right = scalar(&right_tensor);

        assert_eq!(left.to_bits(), right.to_bits());
        Ok(())
    }

    #[test]
    fn sigreg_rng_advances_across_calls() -> Result<(), LewmCoreError> {
        let device = burn_ndarray::NdArrayDevice::default();
        let sigreg = SigReg::<CpuBackend>::init_with_config(16, 5, 3.0, &device)?;
        let z = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(
                (0_u16..24).map(|idx| f32::from(idx) / 19.0).collect(),
                [2, 3, 4],
            ),
            &device,
        );
        let mut rng = crate::substream_rng(11, crate::rng::SIGREG_SKETCH_STREAM)?;

        let first_tensor = sigreg.forward(z.clone(), &mut rng)?;
        let second_tensor = sigreg.forward(z, &mut rng)?;
        let first = scalar(&first_tensor);
        let second = scalar(&second_tensor);

        assert_ne!(first.to_bits(), second.to_bits());
        Ok(())
    }

    #[test]
    fn sigreg_rejects_projection_shape_mismatch() -> Result<(), LewmCoreError> {
        let device = burn_ndarray::NdArrayDevice::default();
        let sigreg = SigReg::<CpuBackend>::init_with_config(2, 3, 2.0, &device)?;
        let z = Tensor::<CpuBackend, 3>::zeros([1, 2, 3], &device);
        let projection = Tensor::<CpuBackend, 2>::zeros([2, 2], &device);

        let err = sigreg
            .forward_with_projection(z, projection)
            .expect_err("shape mismatch should fail");

        assert!(matches!(err, LewmCoreError::InvalidShape { .. }));
        Ok(())
    }

    #[test]
    fn sigreg_gradient_flows_through_input() -> Result<(), LewmCoreError> {
        let device = burn_ndarray::NdArrayDevice::default();
        let sigreg = SigReg::<CpuAutodiffBackend>::init_with_config(2, 3, 2.0, &device)?;
        let z = Tensor::<CpuAutodiffBackend, 3>::from_data(
            TensorData::new(vec![0.0, 0.5, 1.0, -0.5], [1, 2, 2]),
            &device,
        )
        .require_grad();
        let projection = Tensor::<CpuAutodiffBackend, 2>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [2, 2]),
            &device,
        );

        let loss = sigreg.forward_with_projection(z.clone(), projection)?;
        let grads = loss.backward();
        let grad = z
            .grad(&grads)
            .ok_or_else(|| LewmCoreError::Other("SIGReg input gradient missing".to_owned()))?;
        let values = tensor_values_inner::<CpuAutodiffBackend, 3>(&grad);

        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| value.abs() > 1.0e-7));
        Ok(())
    }

    fn manual_sigreg(input: ManualSigRegInput<'_>) -> f32 {
        let flattened_count = input.batch * input.steps;
        let t_grid = build_t_grid(input.knots, input.t_max);
        let phi = build_phi(&t_grid);
        let trap = build_trap(input.knots, input.t_max);
        let mut total = 0.0_f32;

        for projection_index in 0..input.num_proj {
            let mut per_projection = 0.0_f32;
            for knot_index in 0..input.knots {
                let frequency = t_grid[knot_index];
                let mut cos_sum = 0.0_f32;
                let mut sin_sum = 0.0_f32;
                for sample_index in 0..flattened_count {
                    let mut dot = 0.0_f32;
                    for dim_index in 0..input.dim {
                        dot += input.projection[(projection_index * input.dim) + dim_index]
                            * input.embeddings[(sample_index * input.dim) + dim_index];
                    }
                    cos_sum += (frequency * dot).cos();
                    sin_sum += (frequency * dot).sin();
                }
                let cos_mean = cos_sum / usize_to_f32(flattened_count);
                let sin_mean = sin_sum / usize_to_f32(flattened_count);
                let real = cos_mean - phi[knot_index];
                per_projection +=
                    trap[knot_index] * phi[knot_index] * ((real * real) + (sin_mean * sin_mean));
            }
            total += per_projection;
        }

        total / usize_to_f32(input.num_proj)
    }

    fn scalar<B: Backend>(tensor: &Tensor<B, 1>) -> f32 {
        let values = tensor_values(tensor);
        values[0]
    }

    fn tensor_values<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Vec<f32> {
        tensor
            .to_data()
            .to_vec::<f32>()
            .expect("test tensor should contain f32 values")
    }

    fn tensor_values_inner<B: AutodiffBackend, const D: usize>(
        tensor: &Tensor<B::InnerBackend, D>,
    ) -> Vec<f32> {
        tensor
            .to_data()
            .to_vec::<f32>()
            .expect("test tensor should contain f32 values")
    }

    #[track_caller]
    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "expected {expected}, got {actual}, diff {diff}, tolerance {tolerance}"
        );
    }
}
