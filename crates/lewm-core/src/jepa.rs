//! Top-level JEPA wrapper from RFC 0002.

use burn::module::Ignored;
use burn::tensor::{Tensor, backend::Backend};
use rand_chacha::ChaCha20Rng;

use crate::LewmCoreError;
use crate::config::JepaConfig;
use crate::embedder::Embedder;
use crate::init::{ModelInitRng, model_init_rng};
use crate::losses::{SigReg, prediction_loss};
use crate::mlp::Mlp;
use crate::predictor::ArPredictor;
use crate::vit::Vit;

/// Training-time JEPA loss scalars.
#[derive(Debug)]
pub struct JepaLosses<B: Backend> {
    /// Prediction MSE loss.
    pub pred: Tensor<B, 1>,
    /// `SIGReg` target-embedding regularizer.
    pub sigreg: Tensor<B, 1>,
    /// `pred + lambda_sigreg * sigreg`.
    pub total: Tensor<B, 1>,
}

/// Top-level `LeWM` JEPA module tying encoder, projectors, action encoder, and
/// autoregressive predictor together.
#[derive(burn::module::Module, Debug)]
pub struct Jepa<B: Backend> {
    encoder: Vit<B>,
    action_encoder: Embedder<B>,
    predictor: ArPredictor<B>,
    projector: Mlp<B>,
    pred_proj: Mlp<B>,
    sigreg: SigReg<B>,
    config: Ignored<JepaConfig>,
}

impl<B: Backend> Jepa<B> {
    /// Initialize a JEPA module with the deterministic default model-init seed.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::ConstructionFailed`] when the config violates
    /// the cross-module shape contract, or [`LewmCoreError::InvalidInit`] when
    /// deterministic parameter initialization fails.
    pub fn init(config: JepaConfig, device: &B::Device) -> Result<Self, LewmCoreError> {
        Self::init_with_seed(config, 0, device)
    }

    /// Initialize a JEPA module with an explicit global model-init seed.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::ConstructionFailed`] when the config violates
    /// the cross-module shape contract, or [`LewmCoreError::InvalidInit`] when
    /// deterministic parameter initialization fails.
    pub fn init_with_seed(
        config: JepaConfig,
        seed: u64,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        config
            .validate_shape_contract()
            .map_err(|errors| LewmCoreError::ConstructionFailed {
                reason: errors.join("; "),
            })?;
        let mut rng = model_init_rng(seed)?;
        Self::init_with_rng(config, &mut rng, device)
    }

    fn init_with_rng(
        config: JepaConfig,
        rng: &mut ModelInitRng,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        Ok(Self {
            encoder: Vit::init_with_rng(config.encoder.clone(), rng, device)?,
            action_encoder: Embedder::init_with_rng(&config.action_encoder, rng, device)?,
            predictor: ArPredictor::init_with_rng(config.predictor.clone(), rng, device)?,
            projector: Mlp::init_with_rng(&config.projector, rng, device)?,
            pred_proj: Mlp::init_with_rng(&config.pred_proj, rng, device)?,
            sigreg: SigReg::init(device)?,
            config: Ignored(config),
        })
    }

    /// Return the immutable model config.
    #[must_use]
    pub fn config(&self) -> &JepaConfig {
        &self.config.0
    }

    /// Encode a windowed image tensor to projected CLS embeddings.
    ///
    /// # Shape
    ///
    /// - input `pixels`: `(B, T, C, H, W)`
    /// - output: `(B, T, D)`
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidShape`] when pixel dimensions do not
    /// match the configured encoder.
    pub fn encode(&self, pixels: Tensor<B, 5>) -> Result<Tensor<B, 3>, LewmCoreError> {
        let [batch_size, steps, channels, height, width] = pixels.dims();
        self.validate_pixels(batch_size, steps, channels, height, width)?;
        let flat_batch = checked_mul(batch_size, steps, "JEPA encode batch*time overflowed usize")?;
        let hidden_dim = self.config.0.projector.output_dim;

        let encoder_output = self
            .encoder
            .forward(pixels.reshape([flat_batch, channels, height, width]));
        let cls = Vit::cls_from(&encoder_output);
        Ok(self
            .projector
            .forward(cls)
            .reshape([batch_size, steps, hidden_dim]))
    }

    /// Predict embeddings from projected context embeddings and raw actions.
    ///
    /// # Shape
    ///
    /// - input `context`: `(B, T_ctx, D)`
    /// - input `actions`: `(B, T_ctx, A)`
    /// - output: `(B, T_ctx, D)`
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidShape`] when dimensions do not match the
    /// configured predictor/action encoder, or [`LewmCoreError::SequenceTooLong`]
    /// when `T_ctx` exceeds the predictor positional embedding.
    pub fn predict(
        &self,
        context: Tensor<B, 3>,
        actions: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>, LewmCoreError> {
        self.validate_predict_shapes(&context, &actions)?;
        let action_emb = self.action_encoder.forward(actions);
        Ok(self
            .pred_proj
            .forward(self.predictor.forward(context, action_emb)?))
    }

    /// Autoregressively roll out embeddings with a sliding history window.
    ///
    /// # Shape
    ///
    /// - input `start_embeds`: `(B, history_size, D)`
    /// - input `actions`: `(B, horizon - history_size, A)`
    /// - output: `(B, horizon, D)`
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidShape`] when inputs do not match the
    /// configured rollout contract, or predictor errors from [`Self::predict`].
    pub fn rollout(
        &self,
        start_embeds: Tensor<B, 3>,
        actions: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>, LewmCoreError> {
        self.validate_rollout_shapes(&start_embeds, &actions)?;

        let [batch_size, history_size, hidden_dim] = start_embeds.dims();
        let [_, action_steps, action_dim] = actions.dims();
        let mut embeddings = start_embeds;

        for action_index in 0..action_steps {
            let current_len = embeddings.dims()[1];
            let next_action_index = action_index + 1;
            let ctx_z = embeddings.clone().slice([
                0..batch_size,
                (current_len - history_size)..current_len,
                0..hidden_dim,
            ]);
            let ctx_a = actions
                .clone()
                .slice([
                    0..batch_size,
                    action_index..next_action_index,
                    0..action_dim,
                ])
                .expand([batch_size, history_size, action_dim]);
            let pred = self.predict(ctx_z, ctx_a)?;
            let z_next = pred.slice([0..batch_size, history_size - 1..history_size, 0..hidden_dim]);
            embeddings = Tensor::cat(vec![embeddings, z_next], 1);
        }
        drop(actions);

        Ok(embeddings)
    }

    /// Compute training losses for one window.
    ///
    /// # Shape
    ///
    /// - input `pixels`: `(B, horizon, C, H, W)`
    /// - input `actions`: `(B, horizon, A)` or `(B, horizon - history_size, A)`
    /// - output: three scalar tensors shaped `(1,)`
    ///
    /// # Errors
    ///
    /// Returns shape/predictor/SIGReg errors when the window cannot be evaluated.
    pub fn criterion(
        &self,
        pixels: Tensor<B, 5>,
        actions: Tensor<B, 3>,
        lambda_sigreg: f64,
        rng: &mut ChaCha20Rng,
    ) -> Result<JepaLosses<B>, LewmCoreError> {
        if !lambda_sigreg.is_finite() {
            return Err(LewmCoreError::InvalidTensorOp {
                reason: "lambda_sigreg must be finite".to_owned(),
            });
        }

        let [batch_size, steps, channels, height, width] = pixels.dims();
        if steps != self.config.0.horizon {
            return Err(LewmCoreError::InvalidShape {
                expected: vec![batch_size, self.config.0.horizon, channels, height, width],
                found: vec![batch_size, steps, channels, height, width],
            });
        }

        let history_size = self.config.0.history_size;
        let embeddings = self.encode(pixels)?;
        let hidden_dim = embeddings.dims()[2];
        let start_embeds =
            embeddings
                .clone()
                .slice([0..batch_size, 0..history_size, 0..hidden_dim]);
        let target = embeddings.slice([0..batch_size, history_size..steps, 0..hidden_dim]);
        let rollout_actions = self.criterion_rollout_actions(actions, batch_size)?;
        let pred = self.rollout(start_embeds, rollout_actions)?.slice([
            0..batch_size,
            history_size..steps,
            0..hidden_dim,
        ]);

        let pred_loss = prediction_loss(pred, target.clone());
        let sigreg_loss = self.sigreg.forward(target, rng)?;
        let total = pred_loss.clone() + sigreg_loss.clone().mul_scalar(lambda_sigreg);

        Ok(JepaLosses {
            pred: pred_loss,
            sigreg: sigreg_loss,
            total,
        })
    }

    /// Compute per-batch planning cost against a goal embedding.
    ///
    /// # Shape
    ///
    /// - input `z_history`: `(B, history_size, D)`
    /// - input `actions`: `(B, horizon - history_size, A)`
    /// - input `z_goal`: `(B, D)`
    /// - output: `(B,)`
    ///
    /// # Errors
    ///
    /// Returns shape/predictor errors when the rollout cannot be evaluated.
    pub fn get_cost(
        &self,
        z_history: Tensor<B, 3>,
        actions: Tensor<B, 3>,
        z_goal: Tensor<B, 2>,
    ) -> Result<Tensor<B, 1>, LewmCoreError> {
        self.validate_rollout_shapes(&z_history, &actions)?;
        let [batch_size, _, hidden_dim] = z_history.dims();
        let [goal_batch, goal_dim] = z_goal.dims();
        if goal_batch != batch_size || goal_dim != hidden_dim {
            return Err(LewmCoreError::InvalidShape {
                expected: vec![batch_size, hidden_dim],
                found: vec![goal_batch, goal_dim],
            });
        }

        let z_full = self.rollout(z_history, actions)?;
        let z_final = z_full
            .slice([
                0..batch_size,
                self.config.0.horizon - 1..self.config.0.horizon,
                0..hidden_dim,
            ])
            .squeeze_dim::<2>(1);
        let diff = z_final - z_goal;
        Ok(diff.clone().mul(diff).mean_dim(1).squeeze_dim::<1>(1))
    }

    fn criterion_rollout_actions(
        &self,
        actions: Tensor<B, 3>,
        batch_size: usize,
    ) -> Result<Tensor<B, 3>, LewmCoreError> {
        let [action_batch, action_steps, action_dim] = actions.dims();
        let history_size = self.config.0.history_size;
        let horizon = self.config.0.horizon;
        let tail_steps = self.tail_steps()?;
        if action_batch != batch_size || action_dim != self.config.0.action_encoder.input_dim {
            return Err(LewmCoreError::InvalidShape {
                expected: vec![batch_size, horizon, self.config.0.action_encoder.input_dim],
                found: vec![action_batch, action_steps, action_dim],
            });
        }

        if action_steps == horizon {
            Ok(actions.slice([0..batch_size, history_size..horizon, 0..action_dim]))
        } else if action_steps == tail_steps {
            Ok(actions)
        } else {
            Err(LewmCoreError::InvalidShape {
                expected: vec![
                    batch_size,
                    tail_steps,
                    self.config.0.action_encoder.input_dim,
                ],
                found: vec![action_batch, action_steps, action_dim],
            })
        }
    }

    fn validate_pixels(
        &self,
        batch_size: usize,
        steps: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<(), LewmCoreError> {
        let config = &self.config.0.encoder;
        if batch_size == 0
            || steps == 0
            || channels != config.num_channels
            || height != config.image_size
            || width != config.image_size
        {
            return Err(LewmCoreError::InvalidShape {
                expected: vec![
                    batch_size.max(1),
                    steps.max(1),
                    config.num_channels,
                    config.image_size,
                    config.image_size,
                ],
                found: vec![batch_size, steps, channels, height, width],
            });
        }

        Ok(())
    }

    fn validate_predict_shapes(
        &self,
        context: &Tensor<B, 3>,
        actions: &Tensor<B, 3>,
    ) -> Result<(), LewmCoreError> {
        let [batch_size, steps, hidden_dim] = context.dims();
        let [action_batch, action_steps, action_dim] = actions.dims();
        if batch_size == 0 || steps == 0 || hidden_dim != self.config.0.predictor.input_dim {
            return Err(LewmCoreError::InvalidShape {
                expected: vec![
                    batch_size.max(1),
                    steps.max(1),
                    self.config.0.predictor.input_dim,
                ],
                found: vec![batch_size, steps, hidden_dim],
            });
        }

        if action_batch != batch_size
            || action_steps != steps
            || action_dim != self.config.0.action_encoder.input_dim
        {
            return Err(LewmCoreError::InvalidShape {
                expected: vec![batch_size, steps, self.config.0.action_encoder.input_dim],
                found: vec![action_batch, action_steps, action_dim],
            });
        }

        Ok(())
    }

    fn validate_rollout_shapes(
        &self,
        start_embeds: &Tensor<B, 3>,
        actions: &Tensor<B, 3>,
    ) -> Result<(), LewmCoreError> {
        let [batch_size, history_size, hidden_dim] = start_embeds.dims();
        let [action_batch, action_steps, action_dim] = actions.dims();
        let expected_tail = self.tail_steps()?;

        if batch_size == 0
            || history_size != self.config.0.history_size
            || hidden_dim != self.config.0.predictor.input_dim
        {
            return Err(LewmCoreError::InvalidShape {
                expected: vec![
                    batch_size.max(1),
                    self.config.0.history_size,
                    self.config.0.predictor.input_dim,
                ],
                found: vec![batch_size, history_size, hidden_dim],
            });
        }

        if action_batch != batch_size
            || action_steps != expected_tail
            || action_dim != self.config.0.action_encoder.input_dim
        {
            return Err(LewmCoreError::InvalidShape {
                expected: vec![
                    batch_size,
                    expected_tail,
                    self.config.0.action_encoder.input_dim,
                ],
                found: vec![action_batch, action_steps, action_dim],
            });
        }

        Ok(())
    }

    fn tail_steps(&self) -> Result<usize, LewmCoreError> {
        self.config
            .0
            .horizon
            .checked_sub(self.config.0.history_size)
            .filter(|steps| *steps > 0)
            .ok_or_else(|| LewmCoreError::InvalidTensorOp {
                reason: "horizon must be greater than history_size".to_owned(),
            })
    }
}

fn checked_mul(left: usize, right: usize, reason: &str) -> Result<usize, LewmCoreError> {
    left.checked_mul(right)
        .ok_or_else(|| LewmCoreError::InvalidTensorOp {
            reason: reason.to_owned(),
        })
}

#[cfg(test)]
mod tests {
    use burn::tensor::backend::Backend;
    use burn::tensor::{Tensor, TensorData};

    use crate::config::{
        EmbedderConfig, GeluVariant, JepaConfig, MlpConfig, NormVariant, PredictorConfig,
        VitConfig, VitSize,
    };
    use crate::rng::{SIGREG_SKETCH_STREAM, substream_rng};

    use super::*;

    type CpuBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn jepa_encode_predict_shape_contracts() -> Result<(), LewmCoreError> {
        let device = burn_ndarray::NdArrayDevice::default();
        let model = compact_model(device)?;
        let pixels = Tensor::<CpuBackend, 5>::zeros([2, 4, 3, 16, 16], &device);

        let z = model.encode(pixels)?;
        assert_eq!(z.dims(), [2, 4, 8]);

        let context = z.slice([0..2, 0..2, 0..8]);
        let actions = Tensor::<CpuBackend, 3>::ones([2, 2, 2], &device);
        let pred = model.predict(context, actions)?;

        assert_eq!(pred.dims(), [2, 2, 8]);
        Ok(())
    }

    #[test]
    fn jepa_rollout_uses_repeated_step_action_window() -> Result<(), LewmCoreError> {
        let device = burn_ndarray::NdArrayDevice::default();
        let model = compact_model(device)?;
        let start = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(
                (0_u16..32).map(|value| f32::from(value) / 32.0).collect(),
                [2, 2, 8],
            ),
            &device,
        );
        let actions = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(vec![0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4], [2, 2, 2]),
            &device,
        );

        let full = model.rollout(start.clone(), actions.clone())?;
        let first_action = actions.slice([0..2, 0..1, 0..2]).expand([2, 2, 2]);
        let manual = model.predict(start, first_action)?;

        assert_eq!(full.dims(), [2, 4, 8]);
        assert_tensors_close(
            &full.slice([0..2, 2..3, 0..8]),
            &manual.slice([0..2, 1..2, 0..8]),
        );
        Ok(())
    }

    #[test]
    fn jepa_criterion_and_cost_contracts() -> Result<(), LewmCoreError> {
        let device = burn_ndarray::NdArrayDevice::default();
        let model = compact_model(device)?;
        let pixels = Tensor::<CpuBackend, 5>::ones([2, 4, 3, 16, 16], &device);
        let full_actions = Tensor::<CpuBackend, 3>::ones([2, 4, 2], &device);
        let mut rng = substream_rng(0, SIGREG_SKETCH_STREAM)?;

        let losses = model.criterion(pixels.clone(), full_actions, 0.5, &mut rng)?;
        assert_eq!(losses.pred.dims(), [1]);
        assert_eq!(losses.sigreg.dims(), [1]);
        assert_eq!(losses.total.dims(), [1]);
        assert!(scalar(&losses.total).is_finite());
        assert_close(
            scalar(&losses.total),
            scalar(&losses.pred) + (0.5 * scalar(&losses.sigreg)),
            1.0e-5,
        );

        let z = model.encode(pixels)?;
        let history = z.clone().slice([0..2, 0..2, 0..8]);
        let goal = z.slice([0..2, 3..4, 0..8]).squeeze_dim::<2>(1);
        let tail_actions = Tensor::<CpuBackend, 3>::ones([2, 2, 2], &device);
        let costs = model.get_cost(history, tail_actions, goal)?;

        assert_eq!(costs.dims(), [2]);
        assert!(tensor_values(&costs).iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn jepa_rollout_rejects_wrong_action_horizon() -> Result<(), LewmCoreError> {
        let device = burn_ndarray::NdArrayDevice::default();
        let model = compact_model(device)?;
        let start = Tensor::<CpuBackend, 3>::zeros([2, 2, 8], &device);
        let actions = Tensor::<CpuBackend, 3>::zeros([2, 1, 2], &device);

        let err = model
            .rollout(start, actions)
            .expect_err("wrong rollout action length should be rejected");

        assert!(matches!(err, LewmCoreError::InvalidShape { .. }));
        Ok(())
    }

    fn compact_model(
        device: burn_ndarray::NdArrayDevice,
    ) -> Result<Jepa<CpuBackend>, LewmCoreError> {
        Jepa::<CpuBackend>::init_with_seed(compact_config(), 13, &device)
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
                layer_norm_eps: 1.0e-12,
                use_cls_token: true,
                interpolate_pos_encoding: false,
                use_mask_token: false,
                pretrained: false,
            },
            action_encoder: EmbedderConfig {
                input_dim: 2,
                smoothed_dim: 2,
                emb_dim: 8,
                mlp_scale: 2,
            },
            predictor: PredictorConfig {
                num_frames: 3,
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
                hidden_dim: 16,
                output_dim: 8,
                norm: NormVariant::None,
            },
            pred_proj: MlpConfig {
                input_dim: 8,
                hidden_dim: 16,
                output_dim: 8,
                norm: NormVariant::None,
            },
            history_size: 2,
            horizon: 4,
        }
    }

    fn scalar<B: Backend>(tensor: &Tensor<B, 1>) -> f32 {
        tensor_values(tensor)[0]
    }

    fn tensor_values<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> Vec<f32> {
        tensor
            .to_data()
            .to_vec::<f32>()
            .expect("test tensor should contain f32 values")
    }

    fn assert_tensors_close<B: Backend, const D: usize>(
        actual: &Tensor<B, D>,
        expected: &Tensor<B, D>,
    ) {
        let actual = tensor_values(actual);
        let expected = tensor_values(expected);
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_close(actual, expected, 1.0e-5);
        }
    }

    #[track_caller]
    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "expected {expected}, got {actual}, diff {diff}, tolerance {tolerance}"
        );
    }
}
