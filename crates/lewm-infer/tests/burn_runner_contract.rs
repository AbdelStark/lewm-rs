//! Integration tests for the Burn-direct inference runner.
//!
//! These tests exercise the Safetensors → `BurnJepaRunner` path for the
//! `burn-cpu` feature. They're tagged `_slow_` because the locked `PushT`
//! architecture (224×224 input, 12-block `ViT`) is expensive to run on the
//! `NdArray` CPU backend, so the broader `make test-fast` gate skips them.

#![cfg(feature = "burn-cpu")]
#![allow(clippy::large_stack_arrays)]

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use burn_ndarray::{NdArray, NdArrayDevice};
use lewm_core::{Jepa, JepaConfig, export::to_safetensors};
use lewm_infer::runner::{BurnJepaRunner, IMAGE_ELEMENT_COUNT, InferenceRunner, RunnerFormat};

type CpuBackend = NdArray<f32>;

#[test]
fn _slow_burn_runner_loads_safetensors_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let dir = unique_temp_dir("lewm-burn-runner")?;
    let device = NdArrayDevice::default();
    let config = JepaConfig::default();
    let model = Jepa::<CpuBackend>::init(config.clone(), &device)?;
    let weights_path = dir.join("weights.safetensors");
    to_safetensors(&model, &weights_path)?;

    let mut runner = BurnJepaRunner::<CpuBackend>::from_safetensors(
        &weights_path,
        config.clone(),
        device,
        "cpu",
    )?;
    assert_eq!(runner.metadata().format, RunnerFormat::BurnDirect);
    assert!(!runner.metadata().optimized);
    assert!(runner.metadata().intra_op_threads >= 1);
    assert_eq!(runner.latent_dim(), config.predictor.input_dim);
    assert_eq!(runner.action_dim(), config.action_encoder.input_dim);

    let mut pixels_vec = vec![0.0_f32; IMAGE_ELEMENT_COUNT];
    pixels_vec[42] = 0.5;
    let pixels: Box<[f32; IMAGE_ELEMENT_COUNT]> = pixels_vec
        .into_boxed_slice()
        .try_into()
        .map_err(|_| "pixel buffer did not match encoder input shape")?;
    let latent: Vec<f32> = runner.encode(pixels.as_ref())?;
    assert_eq!(latent.len(), config.predictor.input_dim);
    assert!(latent.iter().all(|value| value.is_finite()));

    fs::remove_dir_all(dir)?;
    Ok(())
}

fn unique_temp_dir(prefix: &str) -> std::io::Result<PathBuf> {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    path.push(format!("{prefix}-{nanos}"));
    fs::create_dir_all(&path)?;
    Ok(path)
}
