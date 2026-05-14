//! Burn-backed action embedder from RFC 0002.

use burn::module::Initializer;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::{Tensor, backend::Backend};

use crate::LewmCoreError;
use crate::config::EmbedderConfig;
use crate::init::{ModelInitRng, model_init_rng, trunc_normal_param, zeros_param};

/// Action embedder that maps `(B, T, A)` actions to `(B, T, E_a)`.
#[derive(burn::module::Module, Debug)]
pub struct Embedder<B: Backend> {
    smoother: Conv1d<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> Embedder<B> {
    /// Initialize an action embedder with the deterministic default model-init seed.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::ConstructionFailed`] when the hidden width
    /// overflows, or [`LewmCoreError::InvalidInit`] when deterministic
    /// parameter initialization fails.
    pub fn init(config: &EmbedderConfig, device: &B::Device) -> Result<Self, LewmCoreError> {
        Self::init_with_seed(config, 0, device)
    }

    /// Initialize an action embedder with an explicit global model-init seed.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::ConstructionFailed`] when the hidden width
    /// overflows, or [`LewmCoreError::InvalidInit`] when deterministic
    /// parameter initialization fails.
    pub fn init_with_seed(
        config: &EmbedderConfig,
        seed: u64,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        let mut rng = model_init_rng(seed)?;
        Self::init_with_rng(config, &mut rng, device)
    }

    pub(crate) fn init_with_rng(
        config: &EmbedderConfig,
        rng: &mut ModelInitRng,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        let mlp_hidden_dim = config
            .emb_dim
            .checked_mul(config.mlp_scale)
            .ok_or_else(|| LewmCoreError::ConstructionFailed {
                reason: "action embedder emb_dim * mlp_scale overflowed usize".to_owned(),
            })?;
        let mut smoother = Conv1dConfig::new(config.input_dim, config.smoothed_dim, 1)
            .with_initializer(Initializer::Zeros)
            .init(device);
        smoother.weight =
            trunc_normal_param([config.smoothed_dim, config.input_dim, 1], rng, device)?;
        smoother.bias = Some(zeros_param([config.smoothed_dim], device)?);

        let mut fc1 = linear_zeros(config.smoothed_dim, mlp_hidden_dim, device);
        fc1.weight = trunc_normal_param([config.smoothed_dim, mlp_hidden_dim], rng, device)?;
        fc1.bias = Some(zeros_param([mlp_hidden_dim], device)?);

        let mut fc2 = linear_zeros(mlp_hidden_dim, config.emb_dim, device);
        fc2.weight = trunc_normal_param([mlp_hidden_dim, config.emb_dim], rng, device)?;
        fc2.bias = Some(zeros_param([config.emb_dim], device)?);

        Ok(Self { smoother, fc1, fc2 })
    }

    /// Run `Conv1d(k=1) -> Linear -> SiLU -> Linear`.
    pub fn forward(&self, actions: Tensor<B, 3>) -> Tensor<B, 3> {
        let smoothed = self
            .smoother
            .forward(actions.permute([0, 2, 1]))
            .permute([0, 2, 1]);
        self.fc2.forward(silu(self.fc1.forward(smoothed)))
    }
}

fn linear_zeros<B: Backend>(d_input: usize, d_output: usize, device: &B::Device) -> Linear<B> {
    LinearConfig::new(d_input, d_output)
        .with_initializer(Initializer::Zeros)
        .init(device)
}

#[cfg(test)]
mod tests {
    use burn_ndarray::NdArrayDevice;

    use super::*;

    type CpuBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn smoother_preserves_conv1d_kernel_one_contract() {
        let device = NdArrayDevice::default();
        let config = EmbedderConfig::default();
        let embedder = Embedder::<CpuBackend>::init(&config, &device)
            .expect("default embedder should initialize");

        assert_eq!(embedder.smoother.kernel_size, 1);
        assert_eq!(
            embedder.smoother.weight.dims(),
            [config.smoothed_dim, config.input_dim, 1]
        );
        assert_eq!(
            embedder
                .smoother
                .bias
                .as_ref()
                .expect("smoother bias should be present")
                .dims(),
            [config.smoothed_dim]
        );
    }
}
