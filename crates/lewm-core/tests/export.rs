//! RFC 0002/0005 Safetensors export tests.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use burn_ndarray::{NdArray, NdArrayDevice};
use lewm_core::export::{
    ExportDType, collect_parameters, step_safetensors_file_name, to_safetensors_bytes,
    to_step_safetensors,
};
use lewm_core::{
    EmbedderConfig, GeluVariant, Jepa, JepaConfig, MlpConfig, NormVariant, PredictorConfig,
    VitConfig, VitSize,
};
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;

type CpuBackend = NdArray<f32>;

#[test]
fn export_collects_jepa_parameters_in_module_visit_order() -> Result<(), Box<dyn std::error::Error>>
{
    let device = NdArrayDevice::default();
    let model = Jepa::<CpuBackend>::init(compact_config(), &device)?;
    let tensors = collect_parameters(&model)?;
    let names = tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        names.first().copied(),
        Some("encoder.embeddings.patch_embed.proj.weight")
    );
    assert!(
        names
            .iter()
            .position(|name| *name == "projector.norm.running_mean")
            < names
                .iter()
                .position(|name| *name == "projector.norm.running_var")
    );
    assert!(names.contains(&"projector.norm.num_batches_tracked"));
    assert!(names.contains(&"pred_proj.norm.running_mean"));

    let num_batches = tensors
        .iter()
        .find(|tensor| tensor.name == "projector.norm.num_batches_tracked")
        .expect("BatchNorm integer state should be exported");
    assert_eq!(num_batches.dtype, ExportDType::I64);
    assert_eq!(num_batches.shape, [1]);
    Ok(())
}

#[test]
fn export_writes_valid_safetensors_with_batch_norm_state() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = TestDir::new("core-export-safetensors")?;
    let device = NdArrayDevice::default();
    let model = Jepa::<CpuBackend>::init(compact_config(), &device)?;
    let summary = to_step_safetensors(&model, dir.path(), 42)?;

    assert_eq!(step_safetensors_file_name(42), "step_0000042.safetensors");
    assert_eq!(summary.path, dir.path().join("step_0000042.safetensors"));
    assert!(summary.byte_len > 8);

    let raw = fs::read(&summary.path)?;
    let exported = SafeTensors::deserialize(&raw)?;
    assert_eq!(exported.len(), summary.tensor_count);
    assert_tensor(
        &exported,
        "encoder.embeddings.patch_embed.proj.weight",
        Dtype::F32,
        &[8, 3, 8, 8],
    )?;
    assert_tensor(&exported, "projector.norm.running_mean", Dtype::F32, &[12])?;
    assert_tensor(&exported, "projector.norm.running_var", Dtype::F32, &[12])?;
    assert_tensor(
        &exported,
        "projector.norm.num_batches_tracked",
        Dtype::I64,
        &[1],
    )?;
    assert_tensor(&exported, "pred_proj.norm.running_mean", Dtype::F32, &[12])?;
    Ok(())
}

#[test]
fn export_bytes_are_deterministic() -> Result<(), Box<dyn std::error::Error>> {
    let device = NdArrayDevice::default();
    let first = Jepa::<CpuBackend>::init(compact_config(), &device)?;
    let second = Jepa::<CpuBackend>::init(compact_config(), &device)?;

    assert_eq!(
        to_safetensors_bytes(&first)?,
        to_safetensors_bytes(&second)?
    );
    Ok(())
}

fn compact_config() -> JepaConfig {
    JepaConfig {
        encoder: VitConfig {
            size: VitSize::Tiny,
            image_size: 16,
            patch_size: 8,
            num_channels: 3,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            hidden_act: GeluVariant::TanhApprox,
            attention_probs_dropout_prob: 0.0,
            hidden_dropout_prob: 0.0,
            layer_norm_eps: 1.0e-6,
            use_cls_token: true,
            interpolate_pos_encoding: false,
            use_mask_token: false,
            pretrained: false,
        },
        action_encoder: EmbedderConfig {
            input_dim: 2,
            smoothed_dim: 4,
            emb_dim: 8,
            mlp_scale: 2,
        },
        predictor: PredictorConfig {
            num_frames: 2,
            depth: 1,
            heads: 2,
            mlp_dim: 16,
            dim_head: 4,
            input_dim: 8,
            hidden_dim: 8,
            output_dim: 8,
            action_emb_dim: 8,
            dropout: 0.0,
            emb_dropout: 0.0,
        },
        projector: MlpConfig {
            input_dim: 8,
            hidden_dim: 12,
            output_dim: 8,
            norm: NormVariant::BatchNorm1d,
        },
        pred_proj: MlpConfig {
            input_dim: 8,
            hidden_dim: 12,
            output_dim: 8,
            norm: NormVariant::BatchNorm1d,
        },
        history_size: 2,
        horizon: 3,
    }
}

fn assert_tensor(
    exported: &SafeTensors<'_>,
    name: &str,
    dtype: Dtype,
    shape: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let tensor = exported.tensor(name)?;
    assert_eq!(tensor.dtype(), dtype, "{name} dtype");
    assert_eq!(tensor.shape(), shape, "{name} shape");
    Ok(())
}

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new(name: &str) -> Result<Self, std::io::Error> {
        let path = std::env::temp_dir().join(format!(
            "lewm-rs-{name}-{}-{}",
            std::process::id(),
            unique_suffix()
        ));
        fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn unique_suffix() -> u128 {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    u128::from(NEXT_ID.fetch_add(1, Ordering::Relaxed))
}
