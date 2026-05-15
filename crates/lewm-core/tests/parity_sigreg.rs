//! `TST-0008-SIGREG-001`: `SigReg` parity test (RFC 0008 §5.3).

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
const K: usize = 1024;
const D: usize = 192;
const TOL: f32 = 1e-3;

#[test]
fn parity_sigreg_value_within_1e3() {
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

    // Projector output (B, T, D) is the embedding input to SIGReg.
    let embeddings = model.encode(pixels).expect("encode");

    // Load the reference projection matrix (K, D) recorded at seed=0.
    let projection = support::tensor_from_dump::<2>(&dumps.sigreg_projection, [K, D], device);

    let value = model
        .sigreg()
        .forward_with_projection(embeddings, projection)
        .expect("sigreg forward_with_projection");

    let actual: Vec<f32> = value.to_data().to_vec().expect("sigreg value to vec");
    let diff = (actual[0] - dumps.sigreg_value).abs();
    assert!(
        diff < TOL,
        "sigreg value |actual-expected| = {diff:.3e} > {TOL:.0e} (TST-0008-SIGREG-001); actual={}, expected={}",
        actual[0],
        dumps.sigreg_value
    );
}
