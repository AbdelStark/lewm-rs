//! RFC 0002 AdaLN-zero shape and initialization tests.

use burn::tensor::Tensor;
use burn_ndarray::NdArrayDevice;
use lewm_core::AdaLNZero;

type CpuBackend = burn_ndarray::NdArray<f32>;

#[test]
fn adaln_zero_splits_six_feature_modulations() {
    let device = NdArrayDevice::default();
    let hidden_dim = 16;
    let action_emb_dim = 7;
    let adaln = AdaLNZero::<CpuBackend>::init(hidden_dim, action_emb_dim, &device)
        .expect("compact AdaLN-zero should initialize");
    let conditioning = Tensor::<CpuBackend, 3>::ones([3, 5, action_emb_dim], &device);

    let outputs = adaln.forward(conditioning);

    assert_eq!(outputs.shift_msa.dims(), [3, 5, hidden_dim]);
    assert_eq!(outputs.scale_msa.dims(), [3, 5, hidden_dim]);
    assert_eq!(outputs.gate_msa.dims(), [3, 5, hidden_dim]);
    assert_eq!(outputs.shift_mlp.dims(), [3, 5, hidden_dim]);
    assert_eq!(outputs.scale_mlp.dims(), [3, 5, hidden_dim]);
    assert_eq!(outputs.gate_mlp.dims(), [3, 5, hidden_dim]);
}
