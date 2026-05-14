//! RFC 0002 action `Embedder` shape-contract tests.

use burn::tensor::Tensor;
use burn_ndarray::NdArrayDevice;
use lewm_core::{Embedder, EmbedderConfig};

type CpuBackend = burn_ndarray::NdArray<f32>;

#[test]
fn action_embedder_maps_bta_to_bte() {
    let device = NdArrayDevice::default();

    for batch_size in [1usize, 4, 8] {
        for time_steps in [1usize, 3, 8, 16] {
            for action_dim in [6usize, 10] {
                let config = EmbedderConfig {
                    input_dim: action_dim,
                    smoothed_dim: action_dim,
                    emb_dim: 64,
                    mlp_scale: 2,
                };
                let embedder = Embedder::<CpuBackend>::init_with_seed(&config, 29, &device)
                    .expect("compact embedder config should initialize");
                let actions =
                    Tensor::<CpuBackend, 3>::zeros([batch_size, time_steps, action_dim], &device);

                let embeddings = embedder.forward(actions);

                assert_eq!(embeddings.dims(), [batch_size, time_steps, config.emb_dim]);
            }
        }
    }
}
