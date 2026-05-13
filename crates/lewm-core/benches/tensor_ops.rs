#![allow(clippy::cast_precision_loss, missing_docs)]

use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use lewm_core::{
    BICUBIC_ALIGN_CORNERS, DeviceKey, PositionEmbedding, build_causal_mask, gelu_erf,
    gelu_tanh_approx, interpolate_pos_embed,
};

fn gelu_inputs() -> Vec<f32> {
    (0..1024)
        .map(|index| -8.0 + ((index as f32) * (16.0 / 1023.0)))
        .collect()
}

fn position_embedding() -> Option<PositionEmbedding> {
    let n_patch = 14 * 14;
    let dim = 16;
    let values = (0..((n_patch + 1) * dim))
        .map(|index| ((index % 97) as f32) / 97.0)
        .collect();

    PositionEmbedding::from_values(n_patch, dim, values).ok()
}

fn bench_gelu(c: &mut Criterion) {
    let inputs = gelu_inputs();

    c.bench_function("lewm_core/gelu_tanh_approx_1024", |b| {
        b.iter(|| {
            let mut sum = 0.0_f32;
            for value in &inputs {
                sum += gelu_tanh_approx(black_box(*value));
            }
            black_box(sum);
        });
    });

    c.bench_function("lewm_core/gelu_erf_1024", |b| {
        b.iter(|| {
            let mut sum = 0.0_f32;
            for value in &inputs {
                sum += gelu_erf(black_box(*value));
            }
            black_box(sum);
        });
    });
}

fn bench_tensor_ops(c: &mut Criterion) {
    let device = DeviceKey::cpu();
    let pos = position_embedding();

    c.bench_function("lewm_core/causal_mask_256_cached", |b| {
        b.iter(|| {
            let mask = build_causal_mask(black_box(256), black_box(&device));
            let _ = black_box(mask);
        });
    });

    if let Some(pos) = pos {
        c.bench_function("lewm_core/interpolate_pos_embed_14_to_16_dim16", |b| {
            b.iter(|| {
                let interpolated = interpolate_pos_embed(
                    black_box(&pos),
                    black_box(16 * 16),
                    BICUBIC_ALIGN_CORNERS,
                );
                let _ = black_box(interpolated);
            });
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10))
        .sample_size(10);
    targets = bench_gelu, bench_tensor_ops
}
criterion_main!(benches);
