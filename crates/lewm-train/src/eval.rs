//! `Jepa<B>` → CEM adapter used by the planning evaluation harness.
//!
//! This module wires the parity-verified [`Jepa<B>`] model into
//! [`lewm_plan::Cem`]'s [`CemCostModel`] trait so the eval CLI in
//! `lewm-plan` (and any downstream consumer) can compute candidate-action
//! costs from a real checkpoint instead of the
//! [`StaticPushtPlanner::zeros`][static] stub.
//!
//! The adapter is intentionally minimal: it tensorises the flat CEM batch,
//! replicates the (per-episode) latent history and goal across the
//! candidate axis, and forwards the call to [`Jepa::get_cost`], which
//! already implements the MSE-to-goal cost contract from RFC 0006.
//!
//! ## Horizon contract
//!
//! CEM owns one number: [`CemCostRequest::horizon_plan`] — the count of
//! action steps it explores per candidate. The model owns a different
//! number: `horizon - history_size` (a.k.a. *tail steps*) — the count of
//! action steps [`Jepa::rollout`] consumes. The adapter rejects requests
//! where the two disagree; eval configs (`configs/pusht_eval.toml`,
//! `configs/so100_eval.toml`) must set
//! `cem.horizon_plan == jepa.horizon - jepa.history_size`, or the planner
//! must pre-process the candidates into a tail-step buffer before handing
//! them to the adapter.
//!
//! Keeping the contract strict avoids silently inventing a padding policy
//! that the reference paper does not specify; future work that needs a
//! different planning horizon should encode the padding strategy
//! explicitly in a new wrapper rather than fold it in here.
//!
//! ## Determinism
//!
//! The adapter is gradient-free: every tensor is constructed on the
//! device of the model with autodiff implicitly disabled by Burn's
//! `forward()` path. No RNG is consumed.
//!
//! [static]: lewm_plan::pusht_eval::StaticPushtPlanner::zeros

use std::sync::Mutex;

use burn_core::tensor::{Tensor, backend::Backend};
use lewm_core::Jepa;
use lewm_plan::{CemCostModel, CemCostRequest, LewmPlanError};

/// Adapter from a backend-generic [`Jepa<B>`] to the [`CemCostModel`] trait.
///
/// See the [module docs](self) for the horizon-plan and determinism
/// contracts.
pub struct JepaCemCostModel<'a, B: Backend> {
    model: Mutex<&'a Jepa<B>>,
    device: B::Device,
    expected_horizon_plan: usize,
    expected_action_dim: usize,
    expected_latent_dim: usize,
    expected_history_size: usize,
}

impl<B: Backend> std::fmt::Debug for JepaCemCostModel<'_, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JepaCemCostModel")
            .field("expected_horizon_plan", &self.expected_horizon_plan)
            .field("expected_action_dim", &self.expected_action_dim)
            .field("expected_latent_dim", &self.expected_latent_dim)
            .field("expected_history_size", &self.expected_history_size)
            .finish_non_exhaustive()
    }
}

impl<'a, B: Backend> JepaCemCostModel<'a, B> {
    /// Build a new adapter for `model` on `device`.
    ///
    /// `device` must match the device on which `model`'s tensors live; the
    /// adapter allocates fresh input tensors with this device when it
    /// dispatches the cost call.
    pub fn new(model: &'a Jepa<B>, device: B::Device) -> Self {
        let config = model.config();
        let tail_steps = config.horizon.saturating_sub(config.history_size);
        Self {
            model: Mutex::new(model),
            device,
            expected_horizon_plan: tail_steps,
            expected_action_dim: config.action_encoder.input_dim,
            expected_latent_dim: config.projector.output_dim,
            expected_history_size: config.history_size,
        }
    }

    /// The `horizon_plan` value this adapter expects in CEM requests.
    #[must_use]
    pub fn expected_horizon_plan(&self) -> usize {
        self.expected_horizon_plan
    }
}

impl<B: Backend> CemCostModel for JepaCemCostModel<'_, B> {
    fn get_cost(&self, request: CemCostRequest<'_>) -> Result<Vec<f32>, LewmPlanError> {
        let input = request.input;

        if request.horizon_plan != self.expected_horizon_plan {
            return Err(LewmPlanError::InvalidCemInput {
                reason: format!(
                    "JepaCemCostModel expects horizon_plan == jepa.horizon - history_size = {} \
                     (received {}). Configure cem.horizon_plan to match the model, or wrap the \
                     adapter with a tail-step padding policy.",
                    self.expected_horizon_plan, request.horizon_plan,
                ),
            });
        }
        if input.action_dim != self.expected_action_dim {
            return Err(LewmPlanError::InvalidCemInput {
                reason: format!(
                    "JepaCemCostModel expects action_dim == {} (received {})",
                    self.expected_action_dim, input.action_dim,
                ),
            });
        }
        if input.latent_dim != self.expected_latent_dim {
            return Err(LewmPlanError::InvalidCemInput {
                reason: format!(
                    "JepaCemCostModel expects latent_dim == {} (received {})",
                    self.expected_latent_dim, input.latent_dim,
                ),
            });
        }
        if input.history_len != self.expected_history_size {
            return Err(LewmPlanError::InvalidCemInput {
                reason: format!(
                    "JepaCemCostModel expects history_len == {} (received {})",
                    self.expected_history_size, input.history_len,
                ),
            });
        }
        if request.batch_size == 0 {
            return Ok(Vec::new());
        }

        // Per-candidate latent history: replicate (H, D) across the batch
        // dimension so every candidate sees the same context.
        let history_len = input.history_len;
        let latent_dim = input.latent_dim;
        let history_data: Vec<f32> = input
            .z_history
            .iter()
            .copied()
            .cycle()
            .take(request.batch_size * history_len * latent_dim)
            .collect();
        let history_tensor = Tensor::<B, 1>::from_floats(history_data.as_slice(), &self.device)
            .reshape([request.batch_size, history_len, latent_dim]);

        let goal_data: Vec<f32> = input
            .z_goal
            .iter()
            .copied()
            .cycle()
            .take(request.batch_size * latent_dim)
            .collect();
        let goal_tensor = Tensor::<B, 1>::from_floats(goal_data.as_slice(), &self.device)
            .reshape([request.batch_size, latent_dim]);

        let action_dim = input.action_dim;
        let horizon_plan = request.horizon_plan;
        let actions_tensor = Tensor::<B, 1>::from_floats(request.candidates, &self.device)
            .reshape([request.batch_size, horizon_plan, action_dim]);

        let model = self
            .model
            .lock()
            .map_err(|_| LewmPlanError::InvalidCemInput {
                reason: "JepaCemCostModel mutex poisoned".to_owned(),
            })?;
        let costs = model
            .get_cost(history_tensor, actions_tensor, goal_tensor)
            .map_err(|error| LewmPlanError::CostEvaluation {
                reason: format!("Jepa::get_cost: {error}"),
            })?;

        // Burn returns a (B,) tensor; pull it onto the host as f32s.
        let host_costs: Vec<f32> =
            costs
                .into_data()
                .to_vec::<f32>()
                .map_err(|error| LewmPlanError::InvalidCemCost {
                    reason: format!("Jepa::get_cost into_data: {error:?}"),
                })?;
        Ok(host_costs)
    }
}

#[cfg(test)]
mod tests {
    use burn_ndarray::NdArray;
    use lewm_core::{
        EmbedderConfig, GeluVariant, Jepa, JepaConfig, MlpConfig, NormVariant, PredictorConfig,
        VitConfig, VitSize,
    };
    use lewm_plan::{Cem, CemCostRequest, CemPlanInput};

    use super::*;

    type B = NdArray<f32>;

    fn compact_jepa_config() -> JepaConfig {
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

    fn make_jepa() -> (Jepa<B>, JepaConfig) {
        let config = compact_jepa_config();
        config.validate_shape_contract().expect("shape contract");
        let device = burn_ndarray::NdArrayDevice::default();
        let model = Jepa::<B>::init(config.clone(), &device).expect("init Jepa");
        (model, config)
    }

    fn cem_with_horizon(horizon_plan: usize) -> Cem {
        Cem {
            n_iter: 2,
            n_cand: 4,
            n_elite: 2,
            horizon_plan,
            sigma_init: 0.5,
            sigma_min: 0.05,
            chunk_size: lewm_plan::DEFAULT_CEM_CHUNK_SIZE,
            max_batch_bytes: lewm_plan::DEFAULT_CEM_MAX_BATCH_BYTES,
        }
    }

    #[test]
    fn round_trip_through_cem_plan() {
        let (model, config) = make_jepa();
        let device = <B as Backend>::Device::default();
        let adapter = JepaCemCostModel::new(&model, device);
        let tail_steps = config.horizon - config.history_size;
        assert_eq!(adapter.expected_horizon_plan(), tail_steps);

        let history_len = config.history_size;
        let latent_dim = config.projector.output_dim;
        let action_dim = config.action_encoder.input_dim;
        // Smooth, small inputs — bounded-magnitude f32s.
        let mut z_history: Vec<f32> = Vec::with_capacity(history_len * latent_dim);
        let mut accumulator = 0.0_f32;
        for _ in 0..(history_len * latent_dim) {
            z_history.push(accumulator);
            accumulator += 0.01;
        }
        let mut z_goal: Vec<f32> = Vec::with_capacity(latent_dim);
        let mut goal_acc = 0.0_f32;
        for _ in 0..latent_dim {
            z_goal.push(goal_acc);
            goal_acc += 0.02;
        }
        let cem = cem_with_horizon(tail_steps);
        let plan = cem
            .plan(
                &adapter,
                CemPlanInput {
                    z_history: &z_history,
                    history_len,
                    latent_dim,
                    z_goal: &z_goal,
                    action_dim,
                },
                42,
            )
            .expect("cem.plan runs end-to-end");
        assert_eq!(plan.horizon_plan, tail_steps);
        assert_eq!(plan.action_dim, action_dim);
        assert!(plan.best_cost.is_finite());
        assert_eq!(plan.best_actions.len(), tail_steps * action_dim);
    }

    #[test]
    fn rejects_horizon_mismatch() {
        let (model, config) = make_jepa();
        let device = <B as Backend>::Device::default();
        let adapter = JepaCemCostModel::new(&model, device);
        let wrong_horizon = adapter.expected_horizon_plan() + 1;
        let history_len = config.history_size;
        let latent_dim = config.projector.output_dim;
        let action_dim = config.action_encoder.input_dim;
        let z_history = vec![0.0_f32; history_len * latent_dim];
        let z_goal = vec![0.0_f32; latent_dim];
        let candidates = vec![0.0_f32; wrong_horizon * action_dim];
        let err = adapter
            .get_cost(CemCostRequest {
                input: CemPlanInput {
                    z_history: &z_history,
                    history_len,
                    latent_dim,
                    z_goal: &z_goal,
                    action_dim,
                },
                candidates: &candidates,
                batch_size: 1,
                batch_offset: 0,
                horizon_plan: wrong_horizon,
                no_grad: true,
            })
            .expect_err("wrong horizon should be rejected");
        assert!(format!("{err}").contains("horizon_plan"));
    }

    #[test]
    fn rejects_action_dim_mismatch() {
        let (model, _config) = make_jepa();
        let device = <B as Backend>::Device::default();
        let adapter = JepaCemCostModel::new(&model, device);
        let z_history = vec![0.0_f32; 2 * 8];
        let z_goal = vec![0.0_f32; 8];
        let candidates = vec![0.0_f32; 2 * 3];
        let err = adapter
            .get_cost(CemCostRequest {
                input: CemPlanInput {
                    z_history: &z_history,
                    history_len: 2,
                    latent_dim: 8,
                    z_goal: &z_goal,
                    action_dim: 3, // adapter expects 2
                },
                candidates: &candidates,
                batch_size: 1,
                batch_offset: 0,
                horizon_plan: 2,
                no_grad: true,
            })
            .expect_err("wrong action_dim should be rejected");
        assert!(format!("{err}").contains("action_dim"));
    }

    #[test]
    fn empty_batch_returns_empty_vec() {
        let (model, _config) = make_jepa();
        let device = <B as Backend>::Device::default();
        let adapter = JepaCemCostModel::new(&model, device);
        let z_history = vec![0.0_f32; 2 * 8];
        let z_goal = vec![0.0_f32; 8];
        let costs = adapter
            .get_cost(CemCostRequest {
                input: CemPlanInput {
                    z_history: &z_history,
                    history_len: 2,
                    latent_dim: 8,
                    z_goal: &z_goal,
                    action_dim: 2,
                },
                candidates: &[],
                batch_size: 0,
                batch_offset: 0,
                horizon_plan: 2,
                no_grad: true,
            })
            .expect("empty batch is a no-op");
        assert!(costs.is_empty());
    }
}
