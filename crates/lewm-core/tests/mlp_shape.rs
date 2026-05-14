//! RFC 0002 projector and prediction-projector `Mlp` shape-contract tests.

use burn::tensor::Tensor;
use burn_ndarray::NdArrayDevice;
use lewm_core::{Mlp, MlpConfig, NormVariant};

type CpuBackend = burn_ndarray::NdArray<f32>;

#[test]
fn projector_and_pred_proj_shapes() {
    let device = NdArrayDevice::default();

    for norm in [
        NormVariant::BatchNorm1d,
        NormVariant::LayerNorm,
        NormVariant::None,
    ] {
        for batch_size in [1usize, 4, 8] {
            for time_steps in [1usize, 3, 8] {
                let config = MlpConfig {
                    input_dim: 12,
                    hidden_dim: 24,
                    output_dim: 16,
                    norm,
                };
                let mlp = Mlp::<CpuBackend>::init_with_seed(&config, 31, &device)
                    .expect("compact MLP config should initialize");
                let tokens = Tensor::<CpuBackend, 3>::zeros(
                    [batch_size, time_steps, config.input_dim],
                    &device,
                );

                let projected = mlp.forward(tokens);

                assert_eq!(
                    projected.dims(),
                    [batch_size, time_steps, config.output_dim]
                );
            }
        }
    }
}

#[test]
fn mlp_accepts_flattened_feature_batches() {
    let device = NdArrayDevice::default();
    let config = MlpConfig {
        input_dim: 12,
        hidden_dim: 24,
        output_dim: 16,
        norm: NormVariant::BatchNorm1d,
    };
    let mlp = Mlp::<CpuBackend>::init_with_seed(&config, 37, &device)
        .expect("compact MLP config should initialize");
    let features = Tensor::<CpuBackend, 2>::zeros([11, config.input_dim], &device);

    let projected = mlp.forward(features);

    assert_eq!(projected.dims(), [11, config.output_dim]);
}
