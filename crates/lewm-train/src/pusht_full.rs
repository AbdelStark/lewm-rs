//! Config-shaped host `PushT` `LeWM` core for bounded training runs.
//!
//! This module keeps the first full-module train path CPU-cheap while using the
//! real `JepaConfig` dimensions and component boundaries. It is not the final
//! Burn `ViT` implementation; it is the host-side module path that replaces the
//! previous 4-D scalar core for local/container/HF smoke-scale training.

use core::fmt;

use lewm_core::JepaConfig;

/// Run id stored in checkpoint sidecars for the full-module bounded path.
pub(crate) const PUSHT_FULL_LEWM_RUN_ID: &str = "pusht-full-module-lewm-v1";

/// Run id stored in checkpoint sidecars for the SO-100 full-module path.
pub(crate) const SO100_FULL_LEWM_RUN_ID: &str = "so100-full-module-lewm-v1";

const IMAGE_CHANNELS: usize = 3;
const INIT_NOISE_SCALE: f64 = 1.0 / 4_294_967_295.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ParameterSlice {
    start: usize,
    len: usize,
}

impl ParameterSlice {
    const fn end(self) -> usize {
        self.start + self.len
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PushtFullLewmLayout {
    encoder_bias: ParameterSlice,
    encoder_pixel_weight: ParameterSlice,
    encoder_energy_weight: ParameterSlice,
    encoder_channel_weight: ParameterSlice,
    encoder_time_weight: ParameterSlice,
    action_bias: ParameterSlice,
    action_weight: ParameterSlice,
    predictor_bias: ParameterSlice,
    predictor_history_weight: ParameterSlice,
    predictor_action_weight: ParameterSlice,
    projector_bias: ParameterSlice,
    projector_weight: ParameterSlice,
    pred_proj_bias: ParameterSlice,
    pred_proj_weight: ParameterSlice,
}

/// Stable checkpoint tensor metadata for one parameter group.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PushtFullLewmParameterSpec {
    /// Stable checkpoint tensor name.
    pub(crate) name: String,
    /// Row-major tensor shape.
    pub(crate) shape: Vec<usize>,
    slice: ParameterSlice,
    /// Whether decoupled `AdamW` weight decay applies.
    pub(crate) apply_weight_decay: bool,
}

/// Error returned by the bounded full-module core.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PushtFullLewmError {
    /// Config dimensions are not supported by the bounded host implementation.
    UnsupportedShape(String),
    /// An example did not satisfy the model shape contract.
    InvalidExample(String),
}

impl fmt::Display for PushtFullLewmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedShape(reason) | Self::InvalidExample(reason) => {
                formatter.write_str(reason)
            },
        }
    }
}

impl std::error::Error for PushtFullLewmError {}

/// Per-frame image summary consumed by the host encoder module.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PushtFullLewmImageFeatures {
    /// Mean normalized pixel value for the frame.
    pub(crate) pixel_mean: f64,
    /// Mean squared normalized pixel value for the frame.
    pub(crate) pixel_energy: f64,
    /// Mean normalized RGB channel values.
    pub(crate) channel_mean: [f64; IMAGE_CHANNELS],
    /// Deterministic temporal feature.
    pub(crate) time_fraction: f64,
}

/// One full-module `PushT` training example.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PushtFullLewmExample {
    /// Historical source frames.
    pub(crate) source: Vec<PushtFullLewmImageFeatures>,
    /// Future target frames.
    pub(crate) target: Vec<PushtFullLewmImageFeatures>,
    /// Packed actions aligned with each target frame.
    pub(crate) packed_actions: Vec<Vec<f64>>,
}

/// Loss values emitted by one full-module `PushT` example.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PushtFullLewmStepLoss {
    /// Total objective.
    pub(crate) total: f64,
    /// Latent prediction MSE.
    pub(crate) pred: f64,
    /// Moment proxy for the `SIGReg` target-latent regularizer.
    pub(crate) sigreg_proxy: f64,
}

/// Config-shaped host `LeWM` module stack.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PushtFullLewmCore {
    hidden_dim: usize,
    action_dim: usize,
    action_emb_dim: usize,
    history_size: usize,
    layout: PushtFullLewmLayout,
    specs: Vec<PushtFullLewmParameterSpec>,
    params: Vec<f64>,
}

impl PushtFullLewmCore {
    /// Build a deterministic host model from the loaded JEPA config.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn new(config: &JepaConfig, seed: u64) -> Result<Self, PushtFullLewmError> {
        validate_supported_shape(config)?;

        let hidden_dim = config.predictor.hidden_dim;
        let action_dim = config.action_encoder.input_dim;
        let action_emb_dim = config.action_encoder.emb_dim;
        let history_size = config.history_size;
        let mut specs = Vec::new();
        let mut params = Vec::new();

        let encoder_bias = push_zeros(
            &mut specs,
            &mut params,
            "encoder.bias",
            &[hidden_dim],
            false,
        );
        let encoder_pixel_weight = push_init(
            &mut specs,
            &mut params,
            "encoder.pixel_mean.weight",
            &[hidden_dim],
            true,
            seed,
            0x11,
            0.06,
        );
        let encoder_energy_weight = push_init(
            &mut specs,
            &mut params,
            "encoder.pixel_energy.weight",
            &[hidden_dim],
            true,
            seed,
            0x12,
            0.03,
        );
        let encoder_channel_weight = push_init(
            &mut specs,
            &mut params,
            "encoder.channel_mean.weight",
            &[hidden_dim, IMAGE_CHANNELS],
            true,
            seed,
            0x13,
            0.04,
        );
        let encoder_time_weight = push_init(
            &mut specs,
            &mut params,
            "encoder.time.weight",
            &[hidden_dim],
            true,
            seed,
            0x14,
            0.01,
        );
        let action_bias = push_zeros(
            &mut specs,
            &mut params,
            "action_encoder.bias",
            &[action_emb_dim],
            false,
        );
        let action_weight = push_init(
            &mut specs,
            &mut params,
            "action_encoder.input.weight",
            &[action_emb_dim, action_dim],
            true,
            seed,
            0x21,
            0.02,
        );
        let predictor_bias = push_zeros(
            &mut specs,
            &mut params,
            "predictor.bias",
            &[hidden_dim],
            false,
        );
        let predictor_history_weight =
            push_history_weights(&mut specs, &mut params, hidden_dim, history_size);
        let predictor_action_weight = push_init(
            &mut specs,
            &mut params,
            "predictor.action.weight",
            &[hidden_dim, action_emb_dim],
            true,
            seed,
            0x31,
            0.01,
        );
        let projector_bias = push_zeros(
            &mut specs,
            &mut params,
            "projector.bias",
            &[hidden_dim],
            false,
        );
        let projector_weight = push_ones(
            &mut specs,
            &mut params,
            "projector.weight",
            &[hidden_dim],
            true,
        );
        let pred_proj_bias = push_zeros(
            &mut specs,
            &mut params,
            "pred_proj.bias",
            &[hidden_dim],
            false,
        );
        let pred_proj_weight = push_ones(
            &mut specs,
            &mut params,
            "pred_proj.weight",
            &[hidden_dim],
            true,
        );

        Ok(Self {
            hidden_dim,
            action_dim,
            action_emb_dim,
            history_size,
            layout: PushtFullLewmLayout {
                encoder_bias,
                encoder_pixel_weight,
                encoder_energy_weight,
                encoder_channel_weight,
                encoder_time_weight,
                action_bias,
                action_weight,
                predictor_bias,
                predictor_history_weight,
                predictor_action_weight,
                projector_bias,
                projector_weight,
                pred_proj_bias,
                pred_proj_weight,
            },
            specs,
            params,
        })
    }

    /// Return the number of scalar parameters.
    pub(crate) fn parameter_count(&self) -> usize {
        self.params.len()
    }

    /// Return all named parameter specs.
    pub(crate) fn parameter_specs(&self) -> &[PushtFullLewmParameterSpec] {
        &self.specs
    }

    /// Return one named parameter tensor as flat values.
    pub(crate) fn parameter_values(&self, spec: &PushtFullLewmParameterSpec) -> &[f64] {
        &self.params[spec.slice.start..spec.slice.end()]
    }

    /// Return one scalar parameter by flat optimizer index.
    pub(crate) fn parameter(&self, flat_index: usize) -> f64 {
        self.params[flat_index]
    }

    /// Write one scalar parameter by flat optimizer index.
    pub(crate) fn set_parameter(&mut self, flat_index: usize, value: f64) {
        self.params[flat_index] = value;
    }

    /// Return all scalar parameters in optimizer/checkpoint order.
    pub(crate) fn flat_parameters(&self) -> &[f64] {
        &self.params
    }

    /// Return the parameter spec that owns a flat scalar index.
    pub(crate) fn parameter_spec_for_flat_index(
        &self,
        flat_index: usize,
    ) -> &PushtFullLewmParameterSpec {
        self.specs
            .iter()
            .find(|spec| flat_index >= spec.slice.start && flat_index < spec.slice.end())
            .unwrap_or_else(|| &self.specs[self.specs.len() - 1])
    }

    /// Evaluate one training example and return scalar gradients.
    pub(crate) fn loss_and_gradients(
        &self,
        example: &PushtFullLewmExample,
        sigreg_weight: f64,
    ) -> Result<(PushtFullLewmStepLoss, Vec<f64>), PushtFullLewmError> {
        self.validate_example(example)?;

        let source_encoded = example
            .source
            .iter()
            .map(|features| self.encode_frame(*features))
            .collect::<Vec<_>>();
        let source_projected = source_encoded
            .iter()
            .map(|latent| self.project(latent))
            .collect::<Vec<_>>();
        let target_encoded = self.encode_frame(example.target[0]);
        let target_projected = self.project(&target_encoded);
        let action_embedding = self.encode_action(&example.packed_actions[0]);
        let predictor_raw = self.predict_next(&source_projected, &action_embedding);
        let predicted_projected = self.pred_project(&predictor_raw);

        let hidden_scale = 1.0 / usize_to_f64(self.hidden_dim);
        let mut pred_loss = 0.0;
        let mut grad_pred_projected = vec![0.0; self.hidden_dim];
        let mut grad_target_projected = vec![0.0; self.hidden_dim];
        for dim in 0..self.hidden_dim {
            let residual = predicted_projected[dim] - target_projected[dim];
            pred_loss += residual * residual * hidden_scale;
            grad_pred_projected[dim] += 2.0 * residual * hidden_scale;
            grad_target_projected[dim] -= 2.0 * residual * hidden_scale;
        }

        let (sigreg_proxy, sigreg_grad) = sigreg_moment_proxy(&target_projected, hidden_scale);
        for dim in 0..self.hidden_dim {
            grad_target_projected[dim] += sigreg_weight * sigreg_grad[dim];
        }

        let mut gradients = vec![0.0; self.params.len()];
        let mut grad_predictor_raw = vec![0.0; self.hidden_dim];
        self.accumulate_pred_proj_gradients(
            &mut gradients,
            &grad_pred_projected,
            &predictor_raw,
            &mut grad_predictor_raw,
        );

        let mut grad_action_embedding = vec![0.0; self.action_emb_dim];
        let mut grad_source_projected = vec![vec![0.0; self.hidden_dim]; self.history_size];
        self.accumulate_predictor_gradients(
            &mut gradients,
            &grad_predictor_raw,
            &source_projected,
            &action_embedding,
            &mut grad_source_projected,
            &mut grad_action_embedding,
        );
        self.accumulate_action_gradients(
            &mut gradients,
            &grad_action_embedding,
            &example.packed_actions[0],
        );

        let mut grad_target_encoded = vec![0.0; self.hidden_dim];
        self.accumulate_projector_gradients(
            &mut gradients,
            &grad_target_projected,
            &target_encoded,
            &mut grad_target_encoded,
        );
        self.accumulate_encoder_gradients(&mut gradients, &grad_target_encoded, example.target[0]);

        for (source_index, features) in example.source.iter().copied().enumerate() {
            let mut grad_source_encoded = vec![0.0; self.hidden_dim];
            self.accumulate_projector_gradients(
                &mut gradients,
                &grad_source_projected[source_index],
                &source_encoded[source_index],
                &mut grad_source_encoded,
            );
            self.accumulate_encoder_gradients(&mut gradients, &grad_source_encoded, features);
        }

        Ok((
            PushtFullLewmStepLoss {
                total: pred_loss + (sigreg_weight * sigreg_proxy),
                pred: pred_loss,
                sigreg_proxy,
            },
            gradients,
        ))
    }

    fn validate_example(&self, example: &PushtFullLewmExample) -> Result<(), PushtFullLewmError> {
        if example.source.len() != self.history_size {
            return Err(PushtFullLewmError::InvalidExample(format!(
                "PushT full LeWM expects {} source frames, found {}",
                self.history_size,
                example.source.len()
            )));
        }
        if example.target.len() != 1 {
            return Err(PushtFullLewmError::InvalidExample(format!(
                "PushT full LeWM bounded path expects exactly 1 target frame, found {}",
                example.target.len()
            )));
        }
        if example.packed_actions.len() != 1 {
            return Err(PushtFullLewmError::InvalidExample(format!(
                "PushT full LeWM bounded path expects exactly 1 packed action, found {}",
                example.packed_actions.len()
            )));
        }
        if example.packed_actions[0].len() != self.action_dim {
            return Err(PushtFullLewmError::InvalidExample(format!(
                "PushT full LeWM expects packed action dim {}, found {}",
                self.action_dim,
                example.packed_actions[0].len()
            )));
        }
        Ok(())
    }

    fn encode_frame(&self, features: PushtFullLewmImageFeatures) -> Vec<f64> {
        let mut latent = vec![0.0; self.hidden_dim];
        for (dim, value) in latent.iter_mut().enumerate() {
            let channel_start = self.layout.encoder_channel_weight.start + (dim * IMAGE_CHANNELS);
            let channel_term = (0..IMAGE_CHANNELS)
                .map(|channel| {
                    self.params[channel_start + channel] * features.channel_mean[channel]
                })
                .sum::<f64>();
            *value = self.param_at(self.layout.encoder_bias, dim)
                + (self.param_at(self.layout.encoder_pixel_weight, dim) * features.pixel_mean)
                + (self.param_at(self.layout.encoder_energy_weight, dim) * features.pixel_energy)
                + (self.param_at(self.layout.encoder_time_weight, dim) * features.time_fraction)
                + channel_term;
        }
        latent
    }

    fn encode_action(&self, action: &[f64]) -> Vec<f64> {
        let mut embedding = vec![0.0; self.action_emb_dim];
        for (emb, value) in embedding.iter_mut().enumerate() {
            let weight_start = self.layout.action_weight.start + (emb * self.action_dim);
            let action_term = action
                .iter()
                .enumerate()
                .map(|(action_dim, action_value)| {
                    self.params[weight_start + action_dim] * action_value
                })
                .sum::<f64>();
            *value = self.param_at(self.layout.action_bias, emb) + action_term;
        }
        embedding
    }

    fn project(&self, latent: &[f64]) -> Vec<f64> {
        affine_diag(
            latent,
            self.slice(self.layout.projector_weight),
            self.slice(self.layout.projector_bias),
        )
    }

    fn pred_project(&self, latent: &[f64]) -> Vec<f64> {
        affine_diag(
            latent,
            self.slice(self.layout.pred_proj_weight),
            self.slice(self.layout.pred_proj_bias),
        )
    }

    fn predict_next(&self, source_projected: &[Vec<f64>], action_embedding: &[f64]) -> Vec<f64> {
        let action_scale = 1.0 / usize_to_f64(self.action_emb_dim).sqrt();
        let mut prediction = vec![0.0; self.hidden_dim];
        for (dim, value) in prediction.iter_mut().enumerate() {
            let mut total = self.param_at(self.layout.predictor_bias, dim);
            let history_start =
                self.layout.predictor_history_weight.start + (dim * self.history_size);
            for (history_index, source) in source_projected.iter().enumerate() {
                total += self.params[history_start + history_index] * source[dim];
            }
            let action_start =
                self.layout.predictor_action_weight.start + (dim * self.action_emb_dim);
            for (emb, action_value) in action_embedding.iter().enumerate() {
                total += self.params[action_start + emb] * action_value * action_scale;
            }
            *value = total;
        }
        prediction
    }

    fn accumulate_pred_proj_gradients(
        &self,
        gradients: &mut [f64],
        grad_output: &[f64],
        input: &[f64],
        grad_input: &mut [f64],
    ) {
        for dim in 0..self.hidden_dim {
            gradients[self.layout.pred_proj_bias.start + dim] += grad_output[dim];
            gradients[self.layout.pred_proj_weight.start + dim] += grad_output[dim] * input[dim];
            grad_input[dim] += grad_output[dim] * self.param_at(self.layout.pred_proj_weight, dim);
        }
    }

    fn accumulate_projector_gradients(
        &self,
        gradients: &mut [f64],
        grad_output: &[f64],
        input: &[f64],
        grad_input: &mut [f64],
    ) {
        for dim in 0..self.hidden_dim {
            gradients[self.layout.projector_bias.start + dim] += grad_output[dim];
            gradients[self.layout.projector_weight.start + dim] += grad_output[dim] * input[dim];
            grad_input[dim] += grad_output[dim] * self.param_at(self.layout.projector_weight, dim);
        }
    }

    fn accumulate_predictor_gradients(
        &self,
        gradients: &mut [f64],
        grad_output: &[f64],
        source_projected: &[Vec<f64>],
        action_embedding: &[f64],
        grad_source_projected: &mut [Vec<f64>],
        grad_action_embedding: &mut [f64],
    ) {
        let action_scale = 1.0 / usize_to_f64(self.action_emb_dim).sqrt();
        for dim in 0..self.hidden_dim {
            gradients[self.layout.predictor_bias.start + dim] += grad_output[dim];
            let history_start =
                self.layout.predictor_history_weight.start + (dim * self.history_size);
            for (history_index, source) in source_projected.iter().enumerate() {
                gradients[history_start + history_index] += grad_output[dim] * source[dim];
                grad_source_projected[history_index][dim] +=
                    grad_output[dim] * self.params[history_start + history_index];
            }
            let action_start =
                self.layout.predictor_action_weight.start + (dim * self.action_emb_dim);
            for (emb, action_value) in action_embedding.iter().enumerate() {
                gradients[action_start + emb] += grad_output[dim] * action_value * action_scale;
                grad_action_embedding[emb] +=
                    grad_output[dim] * self.params[action_start + emb] * action_scale;
            }
        }
    }

    fn accumulate_action_gradients(
        &self,
        gradients: &mut [f64],
        grad_embedding: &[f64],
        action: &[f64],
    ) {
        for (emb, grad) in grad_embedding.iter().copied().enumerate() {
            gradients[self.layout.action_bias.start + emb] += grad;
            let weight_start = self.layout.action_weight.start + (emb * self.action_dim);
            for (action_dim, action_value) in action.iter().enumerate() {
                gradients[weight_start + action_dim] += grad * action_value;
            }
        }
    }

    fn accumulate_encoder_gradients(
        &self,
        gradients: &mut [f64],
        grad_latent: &[f64],
        features: PushtFullLewmImageFeatures,
    ) {
        for (dim, grad) in grad_latent.iter().copied().enumerate() {
            gradients[self.layout.encoder_bias.start + dim] += grad;
            gradients[self.layout.encoder_pixel_weight.start + dim] += grad * features.pixel_mean;
            gradients[self.layout.encoder_energy_weight.start + dim] +=
                grad * features.pixel_energy;
            gradients[self.layout.encoder_time_weight.start + dim] += grad * features.time_fraction;
            let channel_start = self.layout.encoder_channel_weight.start + (dim * IMAGE_CHANNELS);
            for channel in 0..IMAGE_CHANNELS {
                gradients[channel_start + channel] += grad * features.channel_mean[channel];
            }
        }
    }

    fn param_at(&self, slice: ParameterSlice, index: usize) -> f64 {
        self.params[slice.start + index]
    }

    fn slice(&self, slice: ParameterSlice) -> &[f64] {
        &self.params[slice.start..slice.end()]
    }
}

fn validate_supported_shape(config: &JepaConfig) -> Result<(), PushtFullLewmError> {
    if let Err(errors) = config.validate_shape_contract() {
        return Err(PushtFullLewmError::UnsupportedShape(format!(
            "PushT full LeWM config shape contract failed: {}",
            errors.join("; ")
        )));
    }

    if config.encoder.hidden_size != config.predictor.hidden_dim
        || config.projector.input_dim != config.encoder.hidden_size
        || config.projector.output_dim != config.predictor.hidden_dim
        || config.pred_proj.input_dim != config.predictor.hidden_dim
        || config.pred_proj.output_dim != config.predictor.hidden_dim
    {
        return Err(PushtFullLewmError::UnsupportedShape(
            "PushT full LeWM bounded path currently requires equal encoder/projector/predictor dimensions".to_owned(),
        ));
    }

    if config.horizon <= config.history_size {
        return Err(PushtFullLewmError::UnsupportedShape(format!(
            "PushT full LeWM bounded path requires horizon > history_size, found {} <= {}",
            config.horizon, config.history_size
        )));
    }

    if config.horizon - config.history_size != 1 {
        return Err(PushtFullLewmError::UnsupportedShape(format!(
            "PushT full LeWM bounded path currently supports one-step targets, found {}",
            config.horizon - config.history_size
        )));
    }

    Ok(())
}

fn push_zeros(
    specs: &mut Vec<PushtFullLewmParameterSpec>,
    params: &mut Vec<f64>,
    name: &str,
    shape: &[usize],
    apply_weight_decay: bool,
) -> ParameterSlice {
    push_values(
        specs,
        params,
        name,
        shape,
        apply_weight_decay,
        vec![0.0; element_count(shape)],
    )
}

fn push_ones(
    specs: &mut Vec<PushtFullLewmParameterSpec>,
    params: &mut Vec<f64>,
    name: &str,
    shape: &[usize],
    apply_weight_decay: bool,
) -> ParameterSlice {
    push_values(
        specs,
        params,
        name,
        shape,
        apply_weight_decay,
        vec![1.0; element_count(shape)],
    )
}

#[allow(clippy::too_many_arguments)]
fn push_init(
    specs: &mut Vec<PushtFullLewmParameterSpec>,
    params: &mut Vec<f64>,
    name: &str,
    shape: &[usize],
    apply_weight_decay: bool,
    seed: u64,
    stream: u64,
    scale: f64,
) -> ParameterSlice {
    let len = element_count(shape);
    let values = (0..len)
        .map(|index| deterministic_weight(seed, stream, index, scale))
        .collect::<Vec<_>>();
    push_values(specs, params, name, shape, apply_weight_decay, values)
}

fn push_history_weights(
    specs: &mut Vec<PushtFullLewmParameterSpec>,
    params: &mut Vec<f64>,
    hidden_dim: usize,
    history_size: usize,
) -> ParameterSlice {
    let mut values = Vec::with_capacity(hidden_dim * history_size);
    for _dim in 0..hidden_dim {
        for history_index in 0..history_size {
            let is_latest = history_index + 1 == history_size;
            values.push(if is_latest {
                0.75
            } else {
                0.25 / usize_to_f64(history_size)
            });
        }
    }
    push_values(
        specs,
        params,
        "predictor.history.weight",
        &[hidden_dim, history_size],
        true,
        values,
    )
}

fn push_values(
    specs: &mut Vec<PushtFullLewmParameterSpec>,
    params: &mut Vec<f64>,
    name: &str,
    shape: &[usize],
    apply_weight_decay: bool,
    values: Vec<f64>,
) -> ParameterSlice {
    let slice = ParameterSlice {
        start: params.len(),
        len: values.len(),
    };
    params.extend(values);
    specs.push(PushtFullLewmParameterSpec {
        name: name.to_owned(),
        shape: shape.to_vec(),
        slice,
        apply_weight_decay,
    });
    slice
}

fn deterministic_weight(seed: u64, stream: u64, index: usize, scale: f64) -> f64 {
    let index = u64::try_from(index).unwrap_or(u64::MAX);
    let mut value = seed ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ index;
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    let high_bits = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let unit = f64::from(high_bits) * INIT_NOISE_SCALE;
    ((unit * 2.0) - 1.0) * scale
}

fn affine_diag(input: &[f64], weight: &[f64], bias: &[f64]) -> Vec<f64> {
    input
        .iter()
        .zip(weight)
        .zip(bias)
        .map(|((input, weight), bias)| bias + (weight * input))
        .collect()
}

fn sigreg_moment_proxy(target_projected: &[f64], hidden_scale: f64) -> (f64, Vec<f64>) {
    let mean_square = target_projected
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        * hidden_scale;
    let delta = mean_square - 1.0;
    let loss = 0.5 * delta * delta;
    let gradients = target_projected
        .iter()
        .map(|value| 2.0 * delta * value * hidden_scale)
        .collect::<Vec<_>>();
    (loss, gradients)
}

fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn usize_to_f64(value: usize) -> f64 {
    u32::try_from(value).map_or(f64::from(u32::MAX), f64::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_lewm_uses_config_dimensions() -> Result<(), Box<dyn std::error::Error>> {
        let model = PushtFullLewmCore::new(&JepaConfig::default(), 7)?;

        assert_eq!(model.hidden_dim, 192);
        assert_eq!(model.action_dim, 10);
        assert!(model.parameter_count() > 192);
        assert!(
            model
                .parameter_specs()
                .iter()
                .any(|spec| spec.name == "predictor.action.weight" && spec.shape == [192, 192])
        );
        Ok(())
    }

    #[test]
    fn full_lewm_loss_returns_finite_gradients() -> Result<(), Box<dyn std::error::Error>> {
        let model = PushtFullLewmCore::new(&JepaConfig::default(), 7)?;
        let example = PushtFullLewmExample {
            source: vec![features(0.0), features(0.1), features(0.2)],
            target: vec![features(0.3)],
            packed_actions: vec![vec![0.05; 10]],
        };

        let (loss, gradients) = model.loss_and_gradients(&example, 1.0)?;

        assert!(loss.total.is_finite());
        assert!(loss.pred.is_finite());
        assert!(loss.sigreg_proxy.is_finite());
        assert_eq!(gradients.len(), model.parameter_count());
        assert!(gradients.iter().all(|gradient| gradient.is_finite()));
        Ok(())
    }

    fn features(time_fraction: f64) -> PushtFullLewmImageFeatures {
        PushtFullLewmImageFeatures {
            pixel_mean: 0.2,
            pixel_energy: 0.3,
            channel_mean: [0.1, 0.2, 0.3],
            time_fraction,
        }
    }
}
