//! Burn-direct inference runner that runs the `Jepa<B>` Burn module against a
//! pluggable backend (CPU `NdArray`, CUDA, `Wgpu`, ...).
//!
//! The runner is the bridge between the existing CPU [`InferenceRunner`]
//! contract and the Burn ecosystem. It loads weights from a Safetensors file
//! via [`lewm_core::load_jepa_from_safetensors_with_config`] and forwards each
//! `encode`/`predict` call through `Jepa::encode` / `Jepa::predict`.
//!
//! The runner output semantics differ from the Tract ONNX runner: the Tract
//! exports surface raw `ViT` patch tokens, while this runner uses the projected
//! CLS embedding — the same semantics the in-Rust trainer uses end-to-end. That
//! makes it the right runner for parity evaluation against the Python
//! reference's `projector_output` / `pred_proj_output` dumps.

use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use lewm_core::{
    ImportError, Jepa, JepaConfig, LewmCoreError, load_jepa_from_safetensors_with_config,
};

use crate::runner::traits::{
    IMAGE_ELEMENT_COUNT, InferenceRunner, RunnerError, RunnerFormat, RunnerMetadata,
    available_intra_op_threads, validate_predict_shapes,
};

/// Burn-backed inference runner.
///
/// The runner stores a fully loaded `Jepa<B>` plus its device handle and reuses
/// them across every encode/predict call. It is `Send` so it satisfies the
/// trait object bound used by [`crate::runner::load`].
#[derive(Debug)]
pub struct BurnJepaRunner<B: Backend> {
    model: Jepa<B>,
    device: B::Device,
    metadata: RunnerMetadata,
    backend_label: &'static str,
}

impl<B: Backend> BurnJepaRunner<B> {
    /// Load weights from a Safetensors file and prepare the runner.
    ///
    /// `backend_label` is recorded in the runner metadata and used in error
    /// strings so the diagnostics distinguish CPU/Cuda/Wgpu instances.
    ///
    /// # Errors
    ///
    /// Returns [`RunnerError::Backend`] when the Safetensors file cannot be
    /// loaded or when the parameter map does not align with `config`.
    pub fn from_safetensors(
        safetensors_path: &Path,
        config: JepaConfig,
        device: B::Device,
        backend_label: &'static str,
    ) -> Result<Self, RunnerError> {
        let model = load_jepa_from_safetensors_with_config::<B>(safetensors_path, config, &device)
            .map_err(|error| import_to_runner_error(safetensors_path, backend_label, &error))?;
        let metadata = RunnerMetadata {
            format: RunnerFormat::BurnDirect,
            encoder_path: safetensors_path.to_path_buf(),
            predictor_path: safetensors_path.to_path_buf(),
            optimized: false,
            intra_op_threads: available_intra_op_threads(),
        };
        Ok(Self {
            model,
            device,
            metadata,
            backend_label,
        })
    }

    /// Construct from an already-built `Jepa<B>`.
    ///
    /// This is the entry point used by tests where the module is initialized
    /// from a synthetic config rather than a Safetensors file.
    pub fn from_model(
        model: Jepa<B>,
        device: B::Device,
        weights_origin: PathBuf,
        backend_label: &'static str,
    ) -> Self {
        let metadata = RunnerMetadata {
            format: RunnerFormat::BurnDirect,
            encoder_path: weights_origin.clone(),
            predictor_path: weights_origin,
            optimized: false,
            intra_op_threads: available_intra_op_threads(),
        };
        Self {
            model,
            device,
            metadata,
            backend_label,
        }
    }

    /// Return the configured action dimension.
    #[must_use]
    pub fn action_dim(&self) -> usize {
        self.model.config().action_encoder.input_dim
    }

    /// Return the configured latent dimension exposed by the encoder/projector.
    #[must_use]
    pub fn latent_dim(&self) -> usize {
        self.model.config().predictor.input_dim
    }

    /// Return the configured input image side length.
    #[must_use]
    pub fn image_size(&self) -> usize {
        self.model.config().encoder.image_size
    }

    /// Return the configured input channel count.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.model.config().encoder.num_channels
    }
}

impl<B: Backend> InferenceRunner for BurnJepaRunner<B>
where
    B::Device: Send,
{
    fn encode(&mut self, pixels: &[f32; IMAGE_ELEMENT_COUNT]) -> Result<Vec<f32>, RunnerError> {
        let expected_size = self.image_size();
        let expected_channels = self.channel_count();
        if expected_channels != 3 || expected_size != 224 {
            return Err(RunnerError::InvalidShape {
                reason: format!(
                    "BurnJepaRunner currently requires (3, 224, 224) inputs, but the model expects ({expected_channels}, {expected_size}, {expected_size})"
                ),
            });
        }
        let pixels = Tensor::<B, 5>::from_data(
            TensorData::new(pixels.to_vec(), [1, 1, 3, 224, 224]),
            &self.device,
        );
        let encoded = self
            .model
            .encode(pixels)
            .map_err(|error| core_to_runner_error(self.backend_label, "encode", &error))?;
        let latent_dim = self.latent_dim();
        let data = encoded
            .reshape([1, latent_dim])
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| RunnerError::Backend {
                context: format!("[burn-{}] reading encoder output", self.backend_label),
                source: format!("{error:?}"),
            })?;
        Ok(data)
    }

    fn predict(
        &mut self,
        history: &[f32],
        actions: &[f32],
        h: usize,
        a: usize,
    ) -> Result<Vec<f32>, RunnerError> {
        let latent_dim = validate_predict_shapes(history, actions, h, a)?;
        let expected_latent = self.latent_dim();
        if latent_dim != expected_latent {
            return Err(RunnerError::InvalidShape {
                reason: format!(
                    "latent dim mismatch: runner expects {expected_latent}, got {latent_dim}"
                ),
            });
        }
        let expected_action = self.action_dim();
        if a != expected_action {
            return Err(RunnerError::InvalidShape {
                reason: format!("action dim mismatch: runner expects {expected_action}, got {a}"),
            });
        }
        let history_tensor = Tensor::<B, 3>::from_data(
            TensorData::new(history.to_vec(), [1, h, latent_dim]),
            &self.device,
        );
        let actions_tensor =
            Tensor::<B, 3>::from_data(TensorData::new(actions.to_vec(), [1, h, a]), &self.device);
        let predicted = self
            .model
            .predict(history_tensor, actions_tensor)
            .map_err(|error| core_to_runner_error(self.backend_label, "predict", &error))?;
        let data = predicted
            .reshape([h * latent_dim])
            .to_data()
            .to_vec::<f32>()
            .map_err(|error| RunnerError::Backend {
                context: format!("[burn-{}] reading predictor output", self.backend_label),
                source: format!("{error:?}"),
            })?;
        Ok(data)
    }

    fn metadata(&self) -> RunnerMetadata {
        self.metadata.clone()
    }
}

fn import_to_runner_error(path: &Path, backend_label: &str, error: &ImportError) -> RunnerError {
    RunnerError::Backend {
        context: format!(
            "[burn-{backend_label}] loading Jepa weights from {}",
            path.display()
        ),
        source: error.to_string(),
    }
}

fn core_to_runner_error(
    backend_label: &str,
    operation: &str,
    error: &LewmCoreError,
) -> RunnerError {
    RunnerError::Backend {
        context: format!("[burn-{backend_label}] running Jepa {operation}"),
        source: error.to_string(),
    }
}

#[cfg(all(test, feature = "burn-cpu"))]
mod tests {
    use burn_ndarray::{NdArray, NdArrayDevice};

    use super::*;

    type CpuBackend = NdArray<f32>;

    fn tiny_config() -> JepaConfig {
        let mut config = JepaConfig::default();
        // Keep the default architecture (PushT-locked) so tests exercise the
        // real shape contract. We don't downsize here because the encoder shape
        // checks below rely on the 224×224 contract that defaults already use.
        config.horizon = config.history_size + 1;
        config
    }

    #[test]
    #[allow(clippy::large_stack_arrays)]
    fn burn_runner_encode_predict_round_trips() -> Result<(), Box<dyn std::error::Error>> {
        let device = NdArrayDevice::default();
        let config = tiny_config();
        let model = Jepa::<CpuBackend>::init(config.clone(), &device)?;
        let mut runner = BurnJepaRunner::<CpuBackend>::from_model(
            model,
            device,
            PathBuf::from("inline://burn-runner-test"),
            "cpu-test",
        );
        assert_eq!(runner.latent_dim(), config.predictor.input_dim);
        assert_eq!(runner.action_dim(), config.action_encoder.input_dim);

        let pixels = vec![0.0_f32; IMAGE_ELEMENT_COUNT];
        let pixels_array: Box<[f32; IMAGE_ELEMENT_COUNT]> = pixels
            .into_boxed_slice()
            .try_into()
            .map_err(|_| "zero pixels did not match shape")?;
        let latent = runner.encode(pixels_array.as_ref())?;
        assert_eq!(latent.len(), config.predictor.input_dim);

        let history_size = config.history_size;
        let mut history = Vec::with_capacity(history_size * runner.latent_dim());
        for _ in 0..history_size {
            history.extend_from_slice(&latent);
        }
        let actions = vec![0.0_f32; history_size * runner.action_dim()];
        let predicted = runner.predict(&history, &actions, history_size, runner.action_dim())?;
        assert_eq!(predicted.len(), history.len());
        Ok(())
    }
}
