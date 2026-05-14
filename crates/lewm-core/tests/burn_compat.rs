//! Compile smoke for the pinned Burn backend surface.

use burn::tensor::backend::Backend;

type CpuBackend = burn_ndarray::NdArray<f32>;

#[test]
fn burn_ndarray_backend_compiles_under_pinned_toolchain() {
    fn assert_backend<B: Backend>() {}

    assert_backend::<CpuBackend>();
}
