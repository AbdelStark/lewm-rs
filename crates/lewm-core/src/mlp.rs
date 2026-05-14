//! Burn-backed projector and prediction-projector MLP heads from RFC 0002.

use burn::module::{Ignored, Initializer, Param, ParamId, RunningState};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::{Int, Tensor, backend::Backend};

use crate::LewmCoreError;
use crate::config::{MlpConfig, NormVariant};
use crate::init::{ModelInitRng, model_init_rng, ones_param, trunc_normal_param, zeros_param};

const DEFAULT_NORM_EPSILON: f64 = 1.0e-5;
const DEFAULT_BATCH_NORM_MOMENTUM: f64 = 0.1;

/// Two-layer projector MLP used for `projector` and `pred_proj`.
#[derive(burn::module::Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    norm: NormBlock<B>,
    fc2: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    /// Initialize an MLP with the deterministic default model-init seed.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidInit`] when deterministic parameter
    /// initialization fails.
    pub fn init(config: &MlpConfig, device: &B::Device) -> Result<Self, LewmCoreError> {
        Self::init_with_seed(config, 0, device)
    }

    /// Initialize an MLP with an explicit global model-init seed.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidInit`] when deterministic parameter
    /// initialization fails.
    pub fn init_with_seed(
        config: &MlpConfig,
        seed: u64,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        let mut rng = model_init_rng(seed)?;
        Self::init_with_rng(config, &mut rng, device)
    }

    pub(crate) fn init_with_rng(
        config: &MlpConfig,
        rng: &mut ModelInitRng,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        let mut fc1 = linear_zeros(config.input_dim, config.hidden_dim, device);
        fc1.weight = trunc_normal_param([config.input_dim, config.hidden_dim], rng, device)?;
        fc1.bias = Some(zeros_param([config.hidden_dim], device)?);

        let mut fc2 = linear_zeros(config.hidden_dim, config.output_dim, device);
        fc2.weight = trunc_normal_param([config.hidden_dim, config.output_dim], rng, device)?;
        fc2.bias = Some(zeros_param([config.output_dim], device)?);

        Ok(Self {
            fc1,
            norm: NormBlock::init(config, device)?,
            fc2,
        })
    }

    /// Run `Linear -> feature-axis norm -> erf GELU -> Linear`.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let projected = self.fc1.forward(input);
        let shape = projected.dims();
        let hidden_dim = shape[D - 1];
        let flat_batch = shape[..D - 1].iter().product::<usize>();
        let normalized = self
            .norm
            .forward(projected.reshape([flat_batch, hidden_dim]))
            .reshape(shape);

        self.fc2.forward(gelu(normalized))
    }
}

#[derive(burn::module::Module, Debug)]
struct NormBlock<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
    running_mean: RunningState<Tensor<B, 1>>,
    running_var: RunningState<Tensor<B, 1>>,
    num_batches_tracked: Param<Tensor<B, 1, Int>>,
    variant: Ignored<NormVariant>,
    momentum: f64,
    epsilon: f64,
}

impl<B: Backend> NormBlock<B> {
    fn init(config: &MlpConfig, device: &B::Device) -> Result<Self, LewmCoreError> {
        Ok(Self {
            weight: ones_param([config.hidden_dim], device)?,
            bias: zeros_param([config.hidden_dim], device)?,
            running_mean: RunningState::new(Tensor::zeros([config.hidden_dim], device)),
            running_var: RunningState::new(Tensor::ones([config.hidden_dim], device)),
            num_batches_tracked: zero_int_param(device),
            variant: Ignored(config.norm),
            momentum: DEFAULT_BATCH_NORM_MOMENTUM,
            epsilon: DEFAULT_NORM_EPSILON,
        })
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        match self.variant.0 {
            NormVariant::BatchNorm1d => self.forward_batch_norm(input),
            NormVariant::LayerNorm => self.forward_layer_norm(input),
            NormVariant::None => input,
        }
    }

    fn forward_batch_norm(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        if B::ad_enabled() {
            self.forward_batch_norm_train(input)
        } else {
            self.forward_batch_norm_inference(input)
        }
    }

    fn forward_batch_norm_inference(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = input.device();
        let [_, hidden_dim] = input.dims();
        let mean = self
            .running_mean
            .value()
            .to_device(&device)
            .reshape([1, hidden_dim]);
        let var = self
            .running_var
            .value()
            .to_device(&device)
            .reshape([1, hidden_dim]);

        self.forward_normalized(input, mean, var)
    }

    fn forward_batch_norm_train(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = input.device();
        let [_, hidden_dim] = input.dims();
        let mean = input.clone().mean_dim(0);
        let var = input.clone().sub(mean.clone()).square().mean_dim(0);

        let running_mean = self.running_mean.value_sync().to_device(&device);
        let running_var = self.running_var.value_sync().to_device(&device);
        let next_running_mean = running_mean.mul_scalar(1.0 - self.momentum).add(
            mean.clone()
                .detach()
                .reshape([hidden_dim])
                .mul_scalar(self.momentum),
        );
        let next_running_var = running_var.mul_scalar(1.0 - self.momentum).add(
            var.clone()
                .detach()
                .reshape([hidden_dim])
                .mul_scalar(self.momentum),
        );

        self.running_mean.update(next_running_mean.detach());
        self.running_var.update(next_running_var.detach());

        self.forward_normalized(input, mean, var)
    }

    fn forward_layer_norm(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let (var, mean) = input.clone().var_mean_bias(1);
        self.forward_normalized(input, mean, var)
    }

    fn forward_normalized(
        &self,
        input: Tensor<B, 2>,
        mean: Tensor<B, 2>,
        var: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [_, hidden_dim] = input.dims();
        input
            .sub(mean)
            .div(var.add_scalar(self.epsilon).sqrt())
            .mul(self.weight.val().reshape([1, hidden_dim]))
            .add(self.bias.val().reshape([1, hidden_dim]))
    }
}

fn linear_zeros<B: Backend>(d_input: usize, d_output: usize, device: &B::Device) -> Linear<B> {
    LinearConfig::new(d_input, d_output)
        .with_initializer(Initializer::Zeros)
        .init(device)
}

fn zero_int_param<B: Backend>(device: &B::Device) -> Param<Tensor<B, 1, Int>> {
    Param::initialized(ParamId::new(), Tensor::from_data([0], device))
}

#[cfg(test)]
mod tests {
    use burn_ndarray::NdArrayDevice;

    use super::*;

    type CpuBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn batch_norm_state_matches_reference_shapes() {
        let device = NdArrayDevice::default();
        let config = MlpConfig {
            input_dim: 8,
            hidden_dim: 16,
            output_dim: 4,
            norm: NormVariant::BatchNorm1d,
        };

        let mlp = Mlp::<CpuBackend>::init_with_seed(&config, 41, &device)
            .expect("compact MLP should initialize");

        assert_eq!(mlp.norm.weight.dims(), [config.hidden_dim]);
        assert_eq!(mlp.norm.bias.dims(), [config.hidden_dim]);
        assert_eq!(mlp.norm.running_mean.value().dims(), [config.hidden_dim]);
        assert_eq!(mlp.norm.running_var.value().dims(), [config.hidden_dim]);
        assert_eq!(mlp.norm.num_batches_tracked.dims(), [1]);
    }
}
