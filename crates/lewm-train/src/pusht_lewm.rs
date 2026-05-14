//! Minimal repo-native `PushT` `LeWM` core used by bounded train runs.
//!
//! This module is intentionally small: it gives the first real `PushT` train
//! path named encoder, action-encoder, predictor, projector, and prediction
//! projection components without claiming to be the full Burn `ViT` stack.

/// `PushT` action dimensionality.
pub(crate) const PUSHT_ACTION_DIM: usize = 2;

/// Latent width for the bounded minimal core.
pub(crate) const PUSHT_MINIMAL_LEWM_LATENT_DIM: usize = 4;

const PUSHT_MINIMAL_LEWM_PARAM_GROUPS: usize = 14;

/// Flat scalar parameter count across all component groups.
pub(crate) const PUSHT_MINIMAL_LEWM_PARAM_COUNT: usize =
    PUSHT_MINIMAL_LEWM_LATENT_DIM * PUSHT_MINIMAL_LEWM_PARAM_GROUPS;

type LatentVector = [f64; PUSHT_MINIMAL_LEWM_LATENT_DIM];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ParameterGroup {
    EncoderBias = 0,
    EncoderPixelWeight = 1,
    EncoderEnergyWeight = 2,
    EncoderTimeWeight = 3,
    ActionEncoderBias = 4,
    ActionEncoderXWeight = 5,
    ActionEncoderYWeight = 6,
    PredictorBias = 7,
    PredictorLatentWeight = 8,
    PredictorActionWeight = 9,
    ProjectorBias = 10,
    ProjectorWeight = 11,
    PredProjBias = 12,
    PredProjWeight = 13,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PushtMinimalLewmParameterSpec {
    /// Stable checkpoint tensor name.
    pub(crate) name: &'static str,
    group: ParameterGroup,
    /// Whether decoupled `AdamW` weight decay applies.
    pub(crate) apply_weight_decay: bool,
}

const fn parameter_spec(
    name: &'static str,
    group: ParameterGroup,
    apply_weight_decay: bool,
) -> PushtMinimalLewmParameterSpec {
    PushtMinimalLewmParameterSpec {
        name,
        group,
        apply_weight_decay,
    }
}

/// Named component parameter groups in checkpoint order.
pub(crate) const PUSHT_MINIMAL_LEWM_PARAMETER_SPECS: [PushtMinimalLewmParameterSpec;
    PUSHT_MINIMAL_LEWM_PARAM_GROUPS] = [
    parameter_spec("encoder.bias", ParameterGroup::EncoderBias, false),
    parameter_spec(
        "encoder.pixel.weight",
        ParameterGroup::EncoderPixelWeight,
        true,
    ),
    parameter_spec(
        "encoder.energy.weight",
        ParameterGroup::EncoderEnergyWeight,
        true,
    ),
    parameter_spec(
        "encoder.time.weight",
        ParameterGroup::EncoderTimeWeight,
        true,
    ),
    parameter_spec(
        "action_encoder.bias",
        ParameterGroup::ActionEncoderBias,
        false,
    ),
    parameter_spec(
        "action_encoder.x.weight",
        ParameterGroup::ActionEncoderXWeight,
        true,
    ),
    parameter_spec(
        "action_encoder.y.weight",
        ParameterGroup::ActionEncoderYWeight,
        true,
    ),
    parameter_spec("predictor.bias", ParameterGroup::PredictorBias, false),
    parameter_spec(
        "predictor.latent.weight",
        ParameterGroup::PredictorLatentWeight,
        true,
    ),
    parameter_spec(
        "predictor.action.weight",
        ParameterGroup::PredictorActionWeight,
        true,
    ),
    parameter_spec("projector.bias", ParameterGroup::ProjectorBias, false),
    parameter_spec("projector.weight", ParameterGroup::ProjectorWeight, true),
    parameter_spec("pred_proj.bias", ParameterGroup::PredProjBias, false),
    parameter_spec("pred_proj.weight", ParameterGroup::PredProjWeight, true),
];

/// Return the named parameter group for a flat scalar parameter index.
pub(crate) fn parameter_spec_for_flat_index(flat_index: usize) -> PushtMinimalLewmParameterSpec {
    PUSHT_MINIMAL_LEWM_PARAMETER_SPECS[flat_index / PUSHT_MINIMAL_LEWM_LATENT_DIM]
}

/// Low-dimensional image summary consumed by the minimal encoder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PushtMinimalLewmFeatures {
    /// Mean normalized pixel value over the selected time range.
    pub(crate) pixel_mean: f64,
    /// Mean squared normalized pixel value over the selected time range.
    pub(crate) pixel_energy: f64,
    /// Start-frame proxy used as a deterministic temporal feature.
    pub(crate) time_fraction: f64,
}

/// One training pair for the bounded `PushT` objective.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PushtMinimalLewmExample {
    /// History-window features.
    pub(crate) source: PushtMinimalLewmFeatures,
    /// Future-window features.
    pub(crate) target: PushtMinimalLewmFeatures,
    /// Mean action over the window.
    pub(crate) action_mean: [f64; PUSHT_ACTION_DIM],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PushtFeatureEncoder {
    bias: LatentVector,
    pixel_weight: LatentVector,
    energy_weight: LatentVector,
    time_weight: LatentVector,
}

impl PushtFeatureEncoder {
    const fn initial() -> Self {
        Self {
            bias: [0.0; PUSHT_MINIMAL_LEWM_LATENT_DIM],
            pixel_weight: [0.08, -0.09, 0.10, -0.11],
            energy_weight: [0.04, 0.03, 0.02, 0.01],
            time_weight: [0.015, -0.015, 0.015, -0.015],
        }
    }

    fn encode(&self, features: PushtMinimalLewmFeatures) -> LatentVector {
        let mut latent = [0.0; PUSHT_MINIMAL_LEWM_LATENT_DIM];
        for (dim, value) in latent.iter_mut().enumerate() {
            *value = self.bias[dim]
                + (self.pixel_weight[dim] * features.pixel_mean)
                + (self.energy_weight[dim] * features.pixel_energy)
                + (self.time_weight[dim] * features.time_fraction);
        }
        latent
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PushtActionEncoder {
    bias: LatentVector,
    x_weight: LatentVector,
    y_weight: LatentVector,
}

impl PushtActionEncoder {
    const fn initial() -> Self {
        Self {
            bias: [0.0; PUSHT_MINIMAL_LEWM_LATENT_DIM],
            x_weight: [0.05, -0.055, 0.06, -0.065],
            y_weight: [-0.04, 0.04, -0.04, 0.04],
        }
    }

    fn encode(&self, action_mean: [f64; PUSHT_ACTION_DIM]) -> LatentVector {
        let mut embedding = [0.0; PUSHT_MINIMAL_LEWM_LATENT_DIM];
        for (dim, value) in embedding.iter_mut().enumerate() {
            *value = self.bias[dim]
                + (self.x_weight[dim] * action_mean[0])
                + (self.y_weight[dim] * action_mean[1]);
        }
        embedding
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LatentActionPredictor {
    bias: LatentVector,
    latent_weight: LatentVector,
    action_weight: LatentVector,
}

impl LatentActionPredictor {
    const fn initial() -> Self {
        Self {
            bias: [0.0; PUSHT_MINIMAL_LEWM_LATENT_DIM],
            latent_weight: [0.65; PUSHT_MINIMAL_LEWM_LATENT_DIM],
            action_weight: [0.25; PUSHT_MINIMAL_LEWM_LATENT_DIM],
        }
    }

    fn predict(&self, source_latent: LatentVector, action_embedding: LatentVector) -> LatentVector {
        let mut prediction = [0.0; PUSHT_MINIMAL_LEWM_LATENT_DIM];
        for (dim, value) in prediction.iter_mut().enumerate() {
            *value = self.bias[dim]
                + (self.latent_weight[dim] * source_latent[dim])
                + (self.action_weight[dim] * action_embedding[dim]);
        }
        prediction
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LatentProjector {
    bias: LatentVector,
    weight: LatentVector,
}

impl LatentProjector {
    const fn identity() -> Self {
        Self {
            bias: [0.0; PUSHT_MINIMAL_LEWM_LATENT_DIM],
            weight: [1.0; PUSHT_MINIMAL_LEWM_LATENT_DIM],
        }
    }

    fn project(&self, latent: LatentVector) -> LatentVector {
        let mut projected = [0.0; PUSHT_MINIMAL_LEWM_LATENT_DIM];
        for (dim, value) in projected.iter_mut().enumerate() {
            *value = self.bias[dim] + (self.weight[dim] * latent[dim]);
        }
        projected
    }
}

/// Minimal componentized `LeWM` core for bounded `PushT` training.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PushtMinimalLewmCore {
    encoder: PushtFeatureEncoder,
    action_encoder: PushtActionEncoder,
    predictor: LatentActionPredictor,
    projector: LatentProjector,
    pred_proj: LatentProjector,
}

impl PushtMinimalLewmCore {
    /// Build the deterministic initialization used by smoke-scale runs.
    pub(crate) const fn initial() -> Self {
        Self {
            encoder: PushtFeatureEncoder::initial(),
            action_encoder: PushtActionEncoder::initial(),
            predictor: LatentActionPredictor::initial(),
            projector: LatentProjector::identity(),
            pred_proj: LatentProjector::identity(),
        }
    }

    fn encode(&self, features: PushtMinimalLewmFeatures) -> LatentVector {
        self.encoder.encode(features)
    }

    fn action_encode(&self, action_mean: [f64; PUSHT_ACTION_DIM]) -> LatentVector {
        self.action_encoder.encode(action_mean)
    }

    fn predict(&self, source_latent: LatentVector, action_embedding: LatentVector) -> LatentVector {
        self.predictor.predict(source_latent, action_embedding)
    }

    fn project_target(&self, target_latent: LatentVector) -> LatentVector {
        self.projector.project(target_latent)
    }

    fn project_prediction(&self, predicted_latent: LatentVector) -> LatentVector {
        self.pred_proj.project(predicted_latent)
    }

    /// Read one scalar parameter by flat optimizer index.
    pub(crate) fn parameter(&self, flat_index: usize) -> f64 {
        let group = parameter_spec_for_flat_index(flat_index).group;
        let dim = flat_index % PUSHT_MINIMAL_LEWM_LATENT_DIM;
        self.parameter_group(group)[dim]
    }

    /// Write one scalar parameter by flat optimizer index.
    pub(crate) fn set_parameter(&mut self, flat_index: usize, value: f64) {
        let group = parameter_spec_for_flat_index(flat_index).group;
        let dim = flat_index % PUSHT_MINIMAL_LEWM_LATENT_DIM;
        self.parameter_group_mut(group)[dim] = value;
    }

    /// Return one named parameter vector for checkpoint serialization.
    pub(crate) fn parameter_values(&self, spec: PushtMinimalLewmParameterSpec) -> LatentVector {
        self.parameter_group(spec.group)
    }

    /// Return all scalar parameters in optimizer/checkpoint order.
    pub(crate) fn flat_parameters(&self) -> [f64; PUSHT_MINIMAL_LEWM_PARAM_COUNT] {
        let mut params = [0.0; PUSHT_MINIMAL_LEWM_PARAM_COUNT];
        for (index, param) in params.iter_mut().enumerate() {
            *param = self.parameter(index);
        }
        params
    }

    fn parameter_group(&self, group: ParameterGroup) -> LatentVector {
        match group {
            ParameterGroup::EncoderBias => self.encoder.bias,
            ParameterGroup::EncoderPixelWeight => self.encoder.pixel_weight,
            ParameterGroup::EncoderEnergyWeight => self.encoder.energy_weight,
            ParameterGroup::EncoderTimeWeight => self.encoder.time_weight,
            ParameterGroup::ActionEncoderBias => self.action_encoder.bias,
            ParameterGroup::ActionEncoderXWeight => self.action_encoder.x_weight,
            ParameterGroup::ActionEncoderYWeight => self.action_encoder.y_weight,
            ParameterGroup::PredictorBias => self.predictor.bias,
            ParameterGroup::PredictorLatentWeight => self.predictor.latent_weight,
            ParameterGroup::PredictorActionWeight => self.predictor.action_weight,
            ParameterGroup::ProjectorBias => self.projector.bias,
            ParameterGroup::ProjectorWeight => self.projector.weight,
            ParameterGroup::PredProjBias => self.pred_proj.bias,
            ParameterGroup::PredProjWeight => self.pred_proj.weight,
        }
    }

    fn parameter_group_mut(&mut self, group: ParameterGroup) -> &mut LatentVector {
        match group {
            ParameterGroup::EncoderBias => &mut self.encoder.bias,
            ParameterGroup::EncoderPixelWeight => &mut self.encoder.pixel_weight,
            ParameterGroup::EncoderEnergyWeight => &mut self.encoder.energy_weight,
            ParameterGroup::EncoderTimeWeight => &mut self.encoder.time_weight,
            ParameterGroup::ActionEncoderBias => &mut self.action_encoder.bias,
            ParameterGroup::ActionEncoderXWeight => &mut self.action_encoder.x_weight,
            ParameterGroup::ActionEncoderYWeight => &mut self.action_encoder.y_weight,
            ParameterGroup::PredictorBias => &mut self.predictor.bias,
            ParameterGroup::PredictorLatentWeight => &mut self.predictor.latent_weight,
            ParameterGroup::PredictorActionWeight => &mut self.predictor.action_weight,
            ParameterGroup::ProjectorBias => &mut self.projector.bias,
            ParameterGroup::ProjectorWeight => &mut self.projector.weight,
            ParameterGroup::PredProjBias => &mut self.pred_proj.bias,
            ParameterGroup::PredProjWeight => &mut self.pred_proj.weight,
        }
    }
}

/// Loss values emitted by one minimal `PushT` `LeWM` example.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PushtMinimalLewmStepLoss {
    /// Total objective.
    pub(crate) total: f64,
    /// Latent prediction loss.
    pub(crate) pred: f64,
    /// Low-dimensional collapse-regularization proxy.
    pub(crate) sigreg_proxy: f64,
}

/// Evaluate the minimal `LeWM` objective and its scalar gradients.
pub(crate) fn loss_and_gradients(
    model: &PushtMinimalLewmCore,
    example: PushtMinimalLewmExample,
    sigreg_weight: f64,
) -> (
    PushtMinimalLewmStepLoss,
    [f64; PUSHT_MINIMAL_LEWM_PARAM_COUNT],
) {
    let source_latent = model.encode(example.source);
    let target_latent = model.encode(example.target);
    let action_embedding = model.action_encode(example.action_mean);
    let predicted_latent = model.predict(source_latent, action_embedding);
    let target_projected = model.project_target(target_latent);
    let predicted_projected = model.project_prediction(predicted_latent);

    let dim_scale = 0.25;
    let mut pred_loss = 0.0;
    let mut target_mean_square = 0.0;
    for dim in 0..PUSHT_MINIMAL_LEWM_LATENT_DIM {
        let residual = predicted_projected[dim] - target_projected[dim];
        pred_loss += 0.5 * residual * residual * dim_scale;
        target_mean_square += target_projected[dim] * target_projected[dim] * dim_scale;
    }
    let sigreg_delta = target_mean_square - 1.0;
    let sigreg_proxy_loss = 0.5 * sigreg_delta * sigreg_delta;
    let total_loss = pred_loss + (sigreg_weight * sigreg_proxy_loss);

    let mut gradients = [0.0; PUSHT_MINIMAL_LEWM_PARAM_COUNT];
    for dim in 0..PUSHT_MINIMAL_LEWM_LATENT_DIM {
        let residual = predicted_projected[dim] - target_projected[dim];
        let grad_pred_projected = residual * dim_scale;
        let grad_sigreg_target =
            sigreg_weight * sigreg_delta * 2.0 * target_projected[dim] * dim_scale;
        let grad_target_projected = (-residual * dim_scale) + grad_sigreg_target;

        gradients[param_index(ParameterGroup::PredProjBias, dim)] += grad_pred_projected;
        gradients[param_index(ParameterGroup::PredProjWeight, dim)] +=
            grad_pred_projected * predicted_latent[dim];
        let grad_predicted_latent = grad_pred_projected * model.pred_proj.weight[dim];

        gradients[param_index(ParameterGroup::PredictorBias, dim)] += grad_predicted_latent;
        gradients[param_index(ParameterGroup::PredictorLatentWeight, dim)] +=
            grad_predicted_latent * source_latent[dim];
        gradients[param_index(ParameterGroup::PredictorActionWeight, dim)] +=
            grad_predicted_latent * action_embedding[dim];
        let grad_source_latent = grad_predicted_latent * model.predictor.latent_weight[dim];
        let grad_action_embedding = grad_predicted_latent * model.predictor.action_weight[dim];

        gradients[param_index(ParameterGroup::ActionEncoderBias, dim)] += grad_action_embedding;
        gradients[param_index(ParameterGroup::ActionEncoderXWeight, dim)] +=
            grad_action_embedding * example.action_mean[0];
        gradients[param_index(ParameterGroup::ActionEncoderYWeight, dim)] +=
            grad_action_embedding * example.action_mean[1];

        gradients[param_index(ParameterGroup::ProjectorBias, dim)] += grad_target_projected;
        gradients[param_index(ParameterGroup::ProjectorWeight, dim)] +=
            grad_target_projected * target_latent[dim];
        let grad_target_latent = grad_target_projected * model.projector.weight[dim];

        accumulate_encoder_gradients(&mut gradients, dim, grad_source_latent, example.source);
        accumulate_encoder_gradients(&mut gradients, dim, grad_target_latent, example.target);
    }

    (
        PushtMinimalLewmStepLoss {
            total: total_loss,
            pred: pred_loss,
            sigreg_proxy: sigreg_proxy_loss,
        },
        gradients,
    )
}

const fn param_index(group: ParameterGroup, dim: usize) -> usize {
    ((group as usize) * PUSHT_MINIMAL_LEWM_LATENT_DIM) + dim
}

fn accumulate_encoder_gradients(
    gradients: &mut [f64; PUSHT_MINIMAL_LEWM_PARAM_COUNT],
    dim: usize,
    latent_gradient: f64,
    features: PushtMinimalLewmFeatures,
) {
    gradients[param_index(ParameterGroup::EncoderBias, dim)] += latent_gradient;
    gradients[param_index(ParameterGroup::EncoderPixelWeight, dim)] +=
        latent_gradient * features.pixel_mean;
    gradients[param_index(ParameterGroup::EncoderEnergyWeight, dim)] +=
        latent_gradient * features.pixel_energy;
    gradients[param_index(ParameterGroup::EncoderTimeWeight, dim)] +=
        latent_gradient * features.time_fraction;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_specs_cover_component_boundaries() {
        assert_eq!(
            PUSHT_MINIMAL_LEWM_PARAMETER_SPECS.len(),
            PUSHT_MINIMAL_LEWM_PARAM_GROUPS
        );
        assert!(
            PUSHT_MINIMAL_LEWM_PARAMETER_SPECS
                .iter()
                .any(|spec| spec.name.starts_with("encoder."))
        );
        assert!(
            PUSHT_MINIMAL_LEWM_PARAMETER_SPECS
                .iter()
                .any(|spec| spec.name.starts_with("action_encoder."))
        );
        assert!(
            PUSHT_MINIMAL_LEWM_PARAMETER_SPECS
                .iter()
                .any(|spec| spec.name.starts_with("predictor."))
        );
        assert!(
            PUSHT_MINIMAL_LEWM_PARAMETER_SPECS
                .iter()
                .any(|spec| spec.name.starts_with("projector."))
        );
        assert!(
            PUSHT_MINIMAL_LEWM_PARAMETER_SPECS
                .iter()
                .any(|spec| spec.name.starts_with("pred_proj."))
        );
        assert!(!PUSHT_MINIMAL_LEWM_PARAMETER_SPECS[0].apply_weight_decay);
        assert!(PUSHT_MINIMAL_LEWM_PARAMETER_SPECS[1].apply_weight_decay);
    }

    #[test]
    fn loss_and_gradients_are_finite() {
        let model = PushtMinimalLewmCore::initial();
        let example = PushtMinimalLewmExample {
            source: PushtMinimalLewmFeatures {
                pixel_mean: -0.1,
                pixel_energy: 0.4,
                time_fraction: 0.0,
            },
            target: PushtMinimalLewmFeatures {
                pixel_mean: 0.2,
                pixel_energy: 0.5,
                time_fraction: 0.1,
            },
            action_mean: [0.3, -0.2],
        };

        let (loss, gradients) = loss_and_gradients(&model, example, 1.0);

        assert!(loss.total.is_finite());
        assert!(loss.pred.is_finite());
        assert!(loss.sigreg_proxy.is_finite());
        assert_eq!(gradients.len(), PUSHT_MINIMAL_LEWM_PARAM_COUNT);
        assert!(gradients.iter().all(|gradient| gradient.is_finite()));
    }
}
