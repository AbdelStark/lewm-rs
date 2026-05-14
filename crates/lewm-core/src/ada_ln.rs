//! AdaLN-zero conditioning helper from RFC 0002.

use burn::module::Initializer;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::{Tensor, backend::Backend};

use crate::LewmCoreError;

const MODULATION_PARTS: usize = 6;

/// AdaLN-zero modulation tensors for one conditional predictor block.
#[derive(Debug, Clone)]
pub struct AdaLNZeroOutputs<B: Backend> {
    /// Attention branch shift. Shape `(B, T, D)`.
    pub shift_msa: Tensor<B, 3>,
    /// Attention branch scale. Shape `(B, T, D)`.
    pub scale_msa: Tensor<B, 3>,
    /// Attention branch residual gate. Shape `(B, T, D)`.
    pub gate_msa: Tensor<B, 3>,
    /// MLP branch shift. Shape `(B, T, D)`.
    pub shift_mlp: Tensor<B, 3>,
    /// MLP branch scale. Shape `(B, T, D)`.
    pub scale_mlp: Tensor<B, 3>,
    /// MLP branch residual gate. Shape `(B, T, D)`.
    pub gate_mlp: Tensor<B, 3>,
}

/// Adaptive LayerNorm-zero conditioner.
#[derive(burn::module::Module, Debug)]
pub struct AdaLNZero<B: Backend> {
    linear: Linear<B>,
    hidden_dim: usize,
    action_emb_dim: usize,
}

impl<B: Backend> AdaLNZero<B> {
    /// Initialize an AdaLN-zero conditioner.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::ConstructionFailed`] when `6 * hidden_dim`
    /// overflows.
    pub fn init(
        hidden_dim: usize,
        action_emb_dim: usize,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        let modulation_dim = hidden_dim.checked_mul(MODULATION_PARTS).ok_or_else(|| {
            LewmCoreError::ConstructionFailed {
                reason: "AdaLN-zero 6 * hidden_dim overflowed usize".to_owned(),
            }
        })?;
        let linear = LinearConfig::new(action_emb_dim, modulation_dim)
            .with_initializer(Initializer::Zeros)
            .init(device);

        Ok(Self {
            linear,
            hidden_dim,
            action_emb_dim,
        })
    }

    /// Return the predictor hidden dimension `D`.
    pub const fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Return the action embedding dimension consumed by the conditioner.
    pub const fn action_emb_dim(&self) -> usize {
        self.action_emb_dim
    }

    /// Produce `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`.
    pub fn forward(&self, conditioning: Tensor<B, 3>) -> AdaLNZeroOutputs<B> {
        let modulation = self.linear.forward(silu(conditioning));
        let mut chunks = modulation.chunk(MODULATION_PARTS, 2);
        let gate_mlp = chunks.remove(5);
        let scale_mlp = chunks.remove(4);
        let shift_mlp = chunks.remove(3);
        let gate_msa = chunks.remove(2);
        let scale_msa = chunks.remove(1);
        let shift_msa = chunks.remove(0);

        AdaLNZeroOutputs {
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tensor;
    use burn_ndarray::NdArrayDevice;

    use super::*;

    type CpuBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn zero_init_returns_zero_modulations_for_any_conditioning() {
        let device = NdArrayDevice::default();
        let hidden_dim = 12;
        let action_emb_dim = 5;
        let adaln = AdaLNZero::<CpuBackend>::init(hidden_dim, action_emb_dim, &device)
            .expect("compact AdaLN-zero should initialize");
        let conditioning = Tensor::<CpuBackend, 3>::ones([2, 4, action_emb_dim], &device);

        let outputs = adaln.forward(conditioning);

        for tensor in [
            outputs.shift_msa,
            outputs.scale_msa,
            outputs.gate_msa,
            outputs.shift_mlp,
            outputs.scale_mlp,
            outputs.gate_mlp,
        ] {
            assert_eq!(tensor.dims(), [2, 4, hidden_dim]);
            assert!(
                tensor
                    .to_data()
                    .to_vec::<f32>()
                    .expect("f32 tensor data")
                    .iter()
                    .all(|value| value.abs() <= f32::EPSILON)
            );
        }
    }

    #[test]
    fn linear_projection_is_zero_initialized_with_reference_shape() {
        let device = NdArrayDevice::default();
        let hidden_dim = 12;
        let action_emb_dim = 5;
        let adaln = AdaLNZero::<CpuBackend>::init(hidden_dim, action_emb_dim, &device)
            .expect("compact AdaLN-zero should initialize");

        assert_eq!(adaln.linear.weight.dims(), [action_emb_dim, 6 * hidden_dim]);
        assert_eq!(
            adaln
                .linear
                .bias
                .as_ref()
                .expect("AdaLN-zero bias should be present")
                .dims(),
            [6 * hidden_dim]
        );
        assert!(
            adaln
                .linear
                .weight
                .val()
                .to_data()
                .to_vec::<f32>()
                .expect("f32 tensor data")
                .iter()
                .all(|value| value.abs() <= f32::EPSILON)
        );
    }
}
