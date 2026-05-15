//! TST-0008-PRED-003: Prediction projector parity test (RFC 0008 §5.3).

#![cfg(feature = "parity-fixtures")]

#[allow(dead_code)]
mod support;

use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArrayDevice;

type CpuBackend = burn_ndarray::NdArray<f32>;

const B: usize = 4;
const T: usize = 4;
const C: usize = 3;
const H: usize = 224;
const W: usize = 224;
const A: usize = 10;
const D: usize = 192;
const TOL: f32 = 1e-4;

#[test]
fn parity_pred_proj_output_within_1e4() {
    let device = NdArrayDevice::default();
    let (Some(model), Some(dumps), Ok(fixture)) = (
        support::try_load_reference_model(&device),
        support::try_load_dumps(),
        support::load_fixture(),
    ) else {
        return;
    };

    let pixels = Tensor::<CpuBackend, 5>::from_data(
        TensorData::new(fixture.pixels.values, [B, T, C, H, W]),
        &device,
    );
    let actions = Tensor::<CpuBackend, 3>::from_data(
        TensorData::new(fixture.actions.values, [B, T, A]),
        &device,
    );

    // Projector output (B, T, D) serves as context.
    let context = model.encode(pixels).expect("encode");
    // Full predict pipeline: action_encoder → predictor → pred_proj.
    let pred_proj_out = model.predict(context, actions).expect("predict");
    let actual: Vec<f32> = pred_proj_out
        .reshape([B * T, D])
        .to_data()
        .to_vec()
        .expect("pred_proj_out to vec");

    let diff = support::linf(&actual, &dumps.pred_proj_output.values);
    assert!(
        diff < TOL,
        "pred_proj L∞ = {diff:.3e} > {TOL:.0e} (TST-0008-PRED-003)"
    );
}
