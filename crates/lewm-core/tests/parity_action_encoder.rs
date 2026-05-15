//! TST-0008-AE-001: Action encoder parity test (RFC 0008 §5.3).

#![cfg(feature = "parity-fixtures")]

#[allow(dead_code)]
mod support;

use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArrayDevice;

type CpuBackend = burn_ndarray::NdArray<f32>;

const B: usize = 4;
const T: usize = 4;
const A: usize = 10;
const D: usize = 192;
const TOL: f32 = 1e-4;

#[test]
fn parity_action_encoder_output_within_1e4() {
    let device = NdArrayDevice::default();
    let (Some(model), Some(dumps), Ok(fixture)) = (
        support::try_load_reference_model(device),
        support::try_load_dumps(),
        support::load_fixture(),
    ) else {
        return;
    };

    let actions = Tensor::<CpuBackend, 3>::from_data(
        TensorData::new(fixture.actions.values, [B, T, A]),
        &device,
    );
    let ae_out = model.action_encoder().forward(actions);
    let actual: Vec<f32> = ae_out
        .reshape([B * T, D])
        .to_data()
        .to_vec()
        .expect("ae_out to vec");

    let diff = support::linf(&actual, &dumps.action_encoder_output.values);
    assert!(
        diff < TOL,
        "action_encoder L∞ = {diff:.3e} > {TOL:.0e} (TST-0008-AE-001)"
    );
}
