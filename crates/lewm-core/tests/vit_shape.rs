//! RFC 0002 `ViT` encoder shape-contract tests.

use burn::tensor::Tensor;
use burn_ndarray::NdArrayDevice;
use lewm_core::{Vit, VitConfig};

type CpuBackend = burn_ndarray::NdArray<f32>;

#[test]
fn vit_forward_preserves_frame_batch_shape_contract() {
    let device = NdArrayDevice::default();
    let config = compact_vit_config();

    for batch_size in [1usize, 4, 8] {
        for time_steps in [1usize, 3, 8] {
            for image_size in [192usize, 224] {
                let model = Vit::<CpuBackend>::init_with_seed(config.clone(), 17, &device)
                    .expect("compact ViT config should initialize");
                let frame_count = batch_size * time_steps;
                let pixels = Tensor::<CpuBackend, 4>::zeros(
                    [frame_count, config.num_channels, image_size, image_size],
                    &device,
                );

                let output = model.forward(pixels);
                let expected_patches = runtime_patch_count(image_size, config.patch_size);

                assert_eq!(
                    output.last_hidden_state.dims(),
                    [frame_count, expected_patches + 1, config.hidden_size]
                );
                assert_eq!(
                    Vit::<CpuBackend>::cls_from(&output).dims(),
                    [frame_count, config.hidden_size]
                );
            }
        }
    }
}

fn compact_vit_config() -> VitConfig {
    VitConfig {
        hidden_size: 12,
        num_hidden_layers: 1,
        num_attention_heads: 3,
        intermediate_size: 24,
        interpolate_pos_encoding: true,
        ..VitConfig::default()
    }
}

fn runtime_patch_count(image_size: usize, patch_stride: usize) -> usize {
    let grid_size = ((image_size - patch_stride) / patch_stride) + 1;
    grid_size * grid_size
}
