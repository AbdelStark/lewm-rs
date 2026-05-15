//! TST-0008-PRED-001/002: Predictor parity tests (RFC 0008 §5.3).

#![cfg(feature = "parity-fixtures")]

#[allow(dead_code)]
mod support;

use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArrayDevice;

type CpuBackend = burn_ndarray::NdArray<f32>;

const B: usize = 4;
const T: usize = 4;
/// Context frames fed to the predictor. The predictor supports num_frames=3;
/// the 4th fixture frame is the target used for loss, not a predictor input.
const T_CTX: usize = 3;
const C: usize = 3;
const H: usize = 224;
const W: usize = 224;
const A: usize = 10;
const D: usize = 192;
const TOL: f32 = 1e-4;

#[test]
fn parity_predictor_output_within_1e4() {
    let device = NdArrayDevice::default();
    let (Some(model), Some(dumps), Ok(fixture)) = (
        support::try_load_reference_model(device),
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

    // Projector output (B, T, D); slice to first T_CTX=3 frames as history context.
    let all_context = model.encode(pixels).expect("encode");
    let context_history = all_context.slice([0..B, 0..T_CTX, 0..D]);

    // Action embeddings: embed first T_CTX actions only.
    let action_history = actions.slice([0..B, 0..T_CTX, 0..A]);
    let action_emb = model.action_encoder().forward(action_history);

    let pred_out = model
        .predictor()
        .forward(context_history, action_emb)
        .expect("predictor forward");
    let actual: Vec<f32> = pred_out
        .reshape([B * T_CTX, D])
        .to_data()
        .to_vec()
        .expect("pred_out to vec");

    let diff = support::linf(&actual, &dumps.predictor_output.values);
    assert!(
        diff < TOL,
        "predictor L∞ = {diff:.3e} > {TOL:.0e} (TST-0008-PRED-001)"
    );
}

#[test]
fn parity_predictor_per_block_shape() {
    let device = NdArrayDevice::default();
    let (Some(_model), Some(dumps), Ok(_fixture)) = (
        support::try_load_reference_model(device),
        support::try_load_dumps(),
        support::load_fixture(),
    ) else {
        return;
    };

    for (i, block) in dumps.predictor_blocks.iter().enumerate() {
        assert_eq!(
            block.after_mlp.shape,
            vec![B, T_CTX, D],
            "predictor block {i} after_mlp shape mismatch (TST-0008-PRED-002)"
        );
        assert_eq!(
            block.after_attn.shape,
            vec![B, T_CTX, D],
            "predictor block {i} after_attn shape mismatch (TST-0008-PRED-002)"
        );
    }
}
