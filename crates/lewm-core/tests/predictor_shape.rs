//! RFC 0002 predictor shape-contract tests.

use burn::tensor::Tensor;
use burn_ndarray::NdArrayDevice;
use lewm_core::{ArPredictor, LewmCoreError, PredictorConfig};

type CpuBackend = burn_ndarray::NdArray<f32>;

#[test]
fn predictor_forward_shape() {
    let device = NdArrayDevice::default();
    let config = compact_config(4);
    let predictor = ArPredictor::<CpuBackend>::init(config.clone(), &device)
        .expect("compact predictor should initialize");

    for batch_size in [1, 2] {
        for seq_len in 1..=config.num_frames {
            let tokens =
                Tensor::<CpuBackend, 3>::ones([batch_size, seq_len, config.hidden_dim], &device);
            let actions = Tensor::<CpuBackend, 3>::ones(
                [batch_size, seq_len, config.action_emb_dim],
                &device,
            );

            let output = predictor
                .forward(tokens, actions)
                .expect("valid predictor prefix should run");

            assert_eq!(output.dims(), [batch_size, seq_len, config.hidden_dim]);
        }
    }
}

#[test]
fn predictor_rejects_sequence_longer_than_position_embedding() {
    let device = NdArrayDevice::default();
    let config = compact_config(2);
    let predictor = ArPredictor::<CpuBackend>::init(config.clone(), &device)
        .expect("compact predictor should initialize");
    let tokens = Tensor::<CpuBackend, 3>::ones([1, 3, config.hidden_dim], &device);
    let actions = Tensor::<CpuBackend, 3>::ones([1, 3, config.action_emb_dim], &device);

    let err = predictor
        .forward(tokens, actions)
        .expect_err("T > num_frames should fail");

    assert_eq!(err, LewmCoreError::SequenceTooLong { got: 3, max: 2 });
}

fn compact_config(num_frames: usize) -> PredictorConfig {
    PredictorConfig {
        num_frames,
        depth: 2,
        heads: 2,
        mlp_dim: 8,
        dim_head: 2,
        input_dim: 4,
        hidden_dim: 4,
        output_dim: 4,
        action_emb_dim: 5,
        dropout: 0.0,
        emb_dropout: 0.0,
    }
}
