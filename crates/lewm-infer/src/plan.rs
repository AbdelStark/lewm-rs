//! CPU-side Cross Entropy Method planning for inference runners.

use std::fmt;

use lewm_core::LewmCoreError;
use lewm_core::rng::substream_rng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Normal};

use crate::runner::{InferenceRunner, RunnerError};

/// RFC 0013 CEM RNG sub-stream name.
pub const CEM_RNG_STREAM: &str = "rng:cem";

/// Default number of CPU CEM iterations.
pub const DEFAULT_N_ITER: usize = 5;
/// Default number of CPU action candidates.
pub const DEFAULT_N_CAND: usize = 16;
/// Default number of elite candidates retained per iteration.
pub const DEFAULT_N_ELITE: usize = 4;
/// Default CPU planning horizon.
pub const DEFAULT_HORIZON_PLAN: usize = 5;
/// Default initial action proposal standard deviation.
pub const DEFAULT_SIGMA_INIT: f32 = 1.0;
/// Default action proposal standard deviation floor.
pub const DEFAULT_SIGMA_MIN: f32 = 0.05;

/// Create the RFC 0013 CEM RNG sub-stream for a global seed.
///
/// # Errors
///
/// Returns [`PlanError::Rng`] only if the public RFC 0013 stream registry stops
/// recognizing [`CEM_RNG_STREAM`].
pub fn cem_rng(global_seed: u64) -> Result<ChaCha20Rng, PlanError> {
    substream_rng(global_seed, CEM_RNG_STREAM).map_err(PlanError::Rng)
}

/// CPU inference CEM configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuCem {
    /// Number of CEM proposal-update iterations.
    pub n_iter: usize,
    /// Number of candidate action sequences per iteration.
    pub n_cand: usize,
    /// Number of lowest-cost candidates used to update the proposal.
    pub n_elite: usize,
    /// Number of future action steps to plan.
    pub horizon_plan: usize,
    /// Initial proposal standard deviation in normalized action space.
    pub sigma_init: f32,
    /// Proposal standard deviation floor in normalized action space.
    pub sigma_min: f32,
}

impl Default for CpuCem {
    fn default() -> Self {
        Self {
            n_iter: DEFAULT_N_ITER,
            n_cand: DEFAULT_N_CAND,
            n_elite: DEFAULT_N_ELITE,
            horizon_plan: DEFAULT_HORIZON_PLAN,
            sigma_init: DEFAULT_SIGMA_INIT,
            sigma_min: DEFAULT_SIGMA_MIN,
        }
    }
}

impl CpuCem {
    /// Run CPU CEM against an inference runner.
    ///
    /// `z_history` is a flat `(H * D,)` history, `z_goal` is `(D,)`, and each
    /// returned action sequence has shape `(horizon_plan * action_dim,)`.
    ///
    /// # Errors
    ///
    /// Returns [`PlanError`] when CEM config, input shapes, RNG construction,
    /// runner prediction, or runner output shape validation fails.
    pub fn plan<R: InferenceRunner + ?Sized>(
        &self,
        runner: &mut R,
        z_history: &[f32],
        z_goal: &[f32],
        rng: &mut ChaCha20Rng,
        action_dim: usize,
    ) -> Result<CpuPlanResult, PlanError> {
        let dimensions = self.validate_inputs(z_history, z_goal, action_dim)?;
        let normal = Normal::<f32>::new(0.0, 1.0).map_err(|error| PlanError::InvalidConfig {
            reason: format!("normal proposal distribution rejected std=1.0: {error}"),
        })?;
        let action_len =
            self.horizon_plan
                .checked_mul(action_dim)
                .ok_or_else(|| PlanError::InvalidConfig {
                    reason: "candidate action length overflowed usize".to_owned(),
                })?;
        let mut mu = vec![0.0_f32; action_len];
        let mut sigma = vec![self.sigma_init; action_len];
        let mut trace = Vec::with_capacity(self.n_iter);
        let mut final_candidates = Vec::new();
        let mut final_costs = Vec::new();

        for iteration in 0..self.n_iter {
            let candidates = self.sample_candidates(&mu, &sigma, normal, rng);
            let costs = candidates
                .iter()
                .map(|candidate| {
                    self.candidate_cost(runner, z_history, z_goal, candidate, dimensions)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elite_indices = elite_indices(&costs, self.n_elite);
            let best_index = elite_indices[0];

            trace.push(CpuCemIterTrace {
                iteration,
                best_cost: costs[best_index],
                mean_cost: mean(&costs),
                sigma_mean: mean(&sigma),
            });

            let (next_mu, next_sigma) =
                proposal_from_elites(&candidates, &elite_indices, self.sigma_min);
            mu = next_mu;
            sigma = next_sigma;
            final_candidates = candidates;
            final_costs = costs;
        }

        let best_index = elite_indices(&final_costs, 1)[0];
        Ok(CpuPlanResult {
            best_actions: final_candidates[best_index].clone(),
            best_cost: final_costs[best_index],
            trace,
        })
    }

    fn validate_inputs(
        &self,
        z_history: &[f32],
        z_goal: &[f32],
        action_dim: usize,
    ) -> Result<PlanDimensions, PlanError> {
        if self.n_iter == 0 {
            return Err(invalid_config("n_iter must be non-zero"));
        }
        if self.n_cand == 0 {
            return Err(invalid_config("n_cand must be non-zero"));
        }
        if self.n_elite == 0 || self.n_elite > self.n_cand {
            return Err(invalid_config("n_elite must be in 1..=n_cand"));
        }
        if self.horizon_plan == 0 {
            return Err(invalid_config("horizon_plan must be non-zero"));
        }
        if action_dim == 0 {
            return Err(invalid_input("action_dim must be non-zero"));
        }
        if !self.sigma_init.is_finite() || self.sigma_init <= 0.0 {
            return Err(invalid_config("sigma_init must be finite and positive"));
        }
        if !self.sigma_min.is_finite() || self.sigma_min <= 0.0 {
            return Err(invalid_config("sigma_min must be finite and positive"));
        }
        if z_goal.is_empty() {
            return Err(invalid_input("z_goal must be non-empty"));
        }
        if z_history.is_empty() {
            return Err(invalid_input("z_history must be non-empty"));
        }
        if !z_history.len().is_multiple_of(z_goal.len()) {
            return Err(PlanError::InvalidInput {
                reason: "z_history length must be divisible by z_goal latent dimension".to_owned(),
            });
        }
        validate_finite("z_history", z_history)?;
        validate_finite("z_goal", z_goal)?;

        Ok(PlanDimensions {
            history_steps: z_history.len() / z_goal.len(),
            latent_dim: z_goal.len(),
            action_dim,
        })
    }

    fn sample_candidates(
        &self,
        mu: &[f32],
        sigma: &[f32],
        normal: Normal<f32>,
        rng: &mut ChaCha20Rng,
    ) -> Vec<Vec<f32>> {
        (0..self.n_cand)
            .map(|_| {
                mu.iter()
                    .zip(sigma.iter())
                    .map(|(mean, std)| *mean + *std * normal.sample(rng))
                    .collect()
            })
            .collect()
    }

    fn candidate_cost<R: InferenceRunner + ?Sized>(
        &self,
        runner: &mut R,
        z_history: &[f32],
        z_goal: &[f32],
        candidate: &[f32],
        dimensions: PlanDimensions,
    ) -> Result<f32, PlanError> {
        let final_latent =
            rollout_final_latent(runner, z_history, candidate, self.horizon_plan, dimensions)?;
        mse(&final_latent, z_goal)
    }
}

/// CPU CEM planning output.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuPlanResult {
    /// Best candidate action sequence in normalized action space.
    pub best_actions: Vec<f32>,
    /// Latent-space MSE cost for the best candidate.
    pub best_cost: f32,
    /// Per-iteration planning trace.
    pub trace: Vec<CpuCemIterTrace>,
}

/// Per-iteration CEM trace.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuCemIterTrace {
    /// Zero-based CEM iteration.
    pub iteration: usize,
    /// Lowest candidate cost in the iteration.
    pub best_cost: f32,
    /// Mean candidate cost in the iteration.
    pub mean_cost: f32,
    /// Mean proposal standard deviation before the elite update.
    pub sigma_mean: f32,
}

/// CPU CEM planning error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// Invalid CEM configuration.
    InvalidConfig {
        /// Failure reason.
        reason: String,
    },
    /// Invalid user input shape or value.
    InvalidInput {
        /// Failure reason.
        reason: String,
    },
    /// RFC 0013 RNG sub-stream construction failed.
    Rng(LewmCoreError),
    /// Runner execution failed.
    Runner {
        /// Runner failure.
        source: RunnerError,
    },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { reason } => write!(f, "invalid CPU CEM config: {reason}"),
            Self::InvalidInput { reason } => write!(f, "invalid CPU CEM input: {reason}"),
            Self::Rng(source) => write!(f, "failed to build CPU CEM RNG: {source}"),
            Self::Runner { source } => write!(f, "CPU CEM runner failure: {source}"),
        }
    }
}

impl std::error::Error for PlanError {}

impl From<RunnerError> for PlanError {
    fn from(source: RunnerError) -> Self {
        Self::Runner { source }
    }
}

#[derive(Debug, Clone, Copy)]
struct PlanDimensions {
    history_steps: usize,
    latent_dim: usize,
    action_dim: usize,
}

fn rollout_final_latent<R: InferenceRunner + ?Sized>(
    runner: &mut R,
    z_history: &[f32],
    candidate: &[f32],
    horizon_plan: usize,
    dimensions: PlanDimensions,
) -> Result<Vec<f32>, PlanError> {
    let mut window = z_history.to_vec();
    let mut repeated_action = vec![0.0_f32; dimensions.history_steps * dimensions.action_dim];

    for step in 0..horizon_plan {
        let action_start = step * dimensions.action_dim;
        let action = &candidate[action_start..action_start + dimensions.action_dim];
        for chunk in repeated_action.chunks_exact_mut(dimensions.action_dim) {
            chunk.copy_from_slice(action);
        }

        let predicted = runner.predict(
            &window,
            &repeated_action,
            dimensions.history_steps,
            dimensions.action_dim,
        )?;
        let expected_len = dimensions.history_steps * dimensions.latent_dim;
        if predicted.len() != expected_len {
            return Err(PlanError::InvalidInput {
                reason: format!(
                    "runner predictor returned {} values, expected {expected_len}",
                    predicted.len()
                ),
            });
        }
        validate_finite("runner predictor output", &predicted)?;

        let next_latent_start = (dimensions.history_steps - 1) * dimensions.latent_dim;
        let next_latent = &predicted[next_latent_start..next_latent_start + dimensions.latent_dim];
        if dimensions.history_steps > 1 {
            window.copy_within(dimensions.latent_dim..expected_len, 0);
        }
        let window_tail = (dimensions.history_steps - 1) * dimensions.latent_dim;
        window[window_tail..window_tail + dimensions.latent_dim].copy_from_slice(next_latent);
    }

    let final_start = (dimensions.history_steps - 1) * dimensions.latent_dim;
    Ok(window[final_start..final_start + dimensions.latent_dim].to_vec())
}

fn proposal_from_elites(
    candidates: &[Vec<f32>],
    elite_indices: &[usize],
    sigma_min: f32,
) -> (Vec<f32>, Vec<f32>) {
    let action_len = candidates[0].len();
    let elite_count = count_as_f32(elite_indices.len());
    let mut mu = vec![0.0_f32; action_len];
    for index in elite_indices {
        for (sum, value) in mu.iter_mut().zip(candidates[*index].iter()) {
            *sum += *value;
        }
    }
    for value in &mut mu {
        *value /= elite_count;
    }

    let mut variance = vec![0.0_f32; action_len];
    for index in elite_indices {
        for ((sum_sq, value), mean) in variance
            .iter_mut()
            .zip(candidates[*index].iter())
            .zip(mu.iter())
        {
            let diff = *value - *mean;
            *sum_sq += diff * diff;
        }
    }
    let sigma = variance
        .into_iter()
        .map(|sum_sq| (sum_sq / elite_count).sqrt().max(sigma_min))
        .collect();
    (mu, sigma)
}

fn elite_indices(costs: &[f32], n_elite: usize) -> Vec<usize> {
    let mut ranked = costs
        .iter()
        .enumerate()
        .map(|(index, cost)| (index, *cost))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
    ranked
        .into_iter()
        .take(n_elite)
        .map(|(index, _)| index)
        .collect()
}

fn mse(left: &[f32], right: &[f32]) -> Result<f32, PlanError> {
    if left.len() != right.len() {
        return Err(PlanError::InvalidInput {
            reason: format!(
                "MSE inputs must have equal lengths, got {} and {}",
                left.len(),
                right.len()
            ),
        });
    }
    let mut sum = 0.0_f32;
    let mut count = 0.0_f32;
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        let diff = *left_value - *right_value;
        sum += diff * diff;
        count += 1.0;
    }
    Ok(sum / count)
}

fn mean(values: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let mut count = 0.0_f32;
    for value in values {
        sum += *value;
        count += 1.0;
    }
    sum / count
}

fn count_as_f32(count: usize) -> f32 {
    (0..count).fold(0.0_f32, |acc, _| acc + 1.0)
}

fn validate_finite(name: &str, values: &[f32]) -> Result<(), PlanError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(PlanError::InvalidInput {
            reason: format!("{name} must contain only finite values"),
        })
    }
}

fn invalid_config(reason: &str) -> PlanError {
    PlanError::InvalidConfig {
        reason: reason.to_owned(),
    }
}

fn invalid_input(reason: &str) -> PlanError {
    PlanError::InvalidInput {
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use lewm_core::rng::substream_rng;

    use super::*;
    use crate::runner::{RunnerFormat, RunnerMetadata};

    const HISTORY_STEPS: usize = 2;
    const LATENT_DIM: usize = 2;
    const ACTION_DIM: usize = 2;

    #[test]
    fn cpu_cem_defaults_match_rfc0007() {
        let planner = CpuCem::default();

        assert_eq!(planner.n_iter, 5);
        assert_eq!(planner.n_cand, 16);
        assert_eq!(planner.n_elite, 4);
        assert_eq!(planner.horizon_plan, 5);
        assert!((planner.sigma_init - 1.0).abs() <= f32::EPSILON);
        assert!((planner.sigma_min - 0.05).abs() <= f32::EPSILON);
    }

    #[test]
    fn cpu_cem_matches_gpu_seed_to_1e3() -> Result<(), Box<dyn std::error::Error>> {
        let planner = CpuCem {
            n_iter: 4,
            n_cand: 10,
            n_elite: 4,
            horizon_plan: 3,
            sigma_init: 0.8,
            sigma_min: 0.05,
        };
        let z_history = [0.0_f32, 0.0, 0.2, -0.1];
        let z_goal = [0.7_f32, -0.4];
        let mut runner = ToyRunner::default();
        let mut cpu_rng = cem_rng(0)?;
        let actual = planner.plan(&mut runner, &z_history, &z_goal, &mut cpu_rng, ACTION_DIM)?;

        let mut reference_rng = cem_rng(0)?;
        let expected = reference_plan(&planner, &z_history, &z_goal, &mut reference_rng)?;

        assert_close_slice(&actual.best_actions, &expected.best_actions, 1e-6);
        assert!((actual.best_cost - expected.best_cost).abs() <= 1e-6);
        assert_eq!(
            runner.predict_calls,
            planner.n_iter * planner.n_cand * planner.horizon_plan
        );
        assert_eq!(actual.trace.len(), planner.n_iter);
        assert!(actual.best_cost.is_finite());

        Ok(())
    }

    #[test]
    fn cpu_cem_uses_rfc0013_cem_substream() -> Result<(), Box<dyn std::error::Error>> {
        let mut left = cem_rng(7)?;
        let mut right = substream_rng(7, CEM_RNG_STREAM)?;
        let planner = CpuCem {
            n_iter: 2,
            n_cand: 6,
            n_elite: 2,
            horizon_plan: 2,
            sigma_init: 1.0,
            sigma_min: 0.05,
        };
        let z_history = [0.0_f32, 0.0, 0.0, 0.0];
        let z_goal = [0.25_f32, -0.5];
        let mut left_runner = ToyRunner::default();
        let mut right_runner = ToyRunner::default();

        let left_result =
            planner.plan(&mut left_runner, &z_history, &z_goal, &mut left, ACTION_DIM)?;
        let right_result = planner.plan(
            &mut right_runner,
            &z_history,
            &z_goal,
            &mut right,
            ACTION_DIM,
        )?;

        assert_eq!(left_result, right_result);
        Ok(())
    }

    #[test]
    fn cpu_cem_rejects_invalid_shapes() -> Result<(), Box<dyn std::error::Error>> {
        let planner = CpuCem::default();
        let mut runner = ToyRunner::default();
        let mut rng = cem_rng(0)?;

        let err = match planner.plan(&mut runner, &[1.0, 2.0, 3.0], &[1.0, 2.0], &mut rng, 2) {
            Ok(result) => {
                return Err(format!("invalid shape unexpectedly planned: {result:?}").into());
            },
            Err(err) => err,
        };

        assert!(matches!(err, PlanError::InvalidInput { .. }));
        Ok(())
    }

    #[derive(Debug, Default)]
    struct ToyRunner {
        predict_calls: usize,
    }

    impl InferenceRunner for ToyRunner {
        fn encode(
            &mut self,
            _pixels: &[f32; crate::runner::IMAGE_ELEMENT_COUNT],
        ) -> Result<Vec<f32>, RunnerError> {
            Ok(Vec::new())
        }

        fn predict(
            &mut self,
            history: &[f32],
            actions: &[f32],
            h: usize,
            a: usize,
        ) -> Result<Vec<f32>, RunnerError> {
            self.predict_calls += 1;
            assert_eq!(h, HISTORY_STEPS);
            assert_eq!(a, ACTION_DIM);
            assert_eq!(history.len(), HISTORY_STEPS * LATENT_DIM);
            assert_eq!(actions.len(), HISTORY_STEPS * ACTION_DIM);

            let mut output = history.to_vec();
            let previous = &history[(HISTORY_STEPS - 1) * LATENT_DIM..HISTORY_STEPS * LATENT_DIM];
            let action = &actions[0..ACTION_DIM];
            let next_start = (HISTORY_STEPS - 1) * LATENT_DIM;
            output[next_start] = previous[0] + action[0];
            output[next_start + 1] = previous[1] + action[1];
            Ok(output)
        }

        fn metadata(&self) -> RunnerMetadata {
            RunnerMetadata {
                format: RunnerFormat::BurnDirect,
                encoder_path: "toy-encoder".into(),
                predictor_path: "toy-predictor".into(),
                optimized: false,
                intra_op_threads: 1,
            }
        }
    }

    fn reference_plan(
        planner: &CpuCem,
        z_history: &[f32],
        z_goal: &[f32],
        rng: &mut ChaCha20Rng,
    ) -> Result<CpuPlanResult, Box<dyn std::error::Error>> {
        let normal = Normal::<f32>::new(0.0, 1.0)?;
        let action_len = planner.horizon_plan * ACTION_DIM;
        let mut mu = vec![0.0_f32; action_len];
        let mut sigma = vec![planner.sigma_init; action_len];
        let mut final_candidates = Vec::new();
        let mut final_costs = Vec::new();
        let mut trace = Vec::new();

        for iteration in 0..planner.n_iter {
            let candidates = (0..planner.n_cand)
                .map(|_| {
                    mu.iter()
                        .zip(sigma.iter())
                        .map(|(mean, std)| *mean + *std * normal.sample(rng))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let costs = candidates
                .iter()
                .map(|candidate| toy_rollout_cost(z_history, z_goal, candidate, planner))
                .collect::<Vec<_>>();
            let elite_indices = elite_indices(&costs, planner.n_elite);
            let best_index = elite_indices[0];
            trace.push(CpuCemIterTrace {
                iteration,
                best_cost: costs[best_index],
                mean_cost: mean(&costs),
                sigma_mean: mean(&sigma),
            });
            let (next_mu, next_sigma) =
                proposal_from_elites(&candidates, &elite_indices, planner.sigma_min);
            mu = next_mu;
            sigma = next_sigma;
            final_candidates = candidates;
            final_costs = costs;
        }

        let best_index = elite_indices(&final_costs, 1)[0];
        Ok(CpuPlanResult {
            best_actions: final_candidates[best_index].clone(),
            best_cost: final_costs[best_index],
            trace,
        })
    }

    fn toy_rollout_cost(
        z_history: &[f32],
        z_goal: &[f32],
        candidate: &[f32],
        planner: &CpuCem,
    ) -> f32 {
        let mut final_latent = z_history[(HISTORY_STEPS - 1) * LATENT_DIM..].to_vec();
        for step in 0..planner.horizon_plan {
            let action_start = step * ACTION_DIM;
            final_latent[0] += candidate[action_start];
            final_latent[1] += candidate[action_start + 1];
        }
        let diff_0 = final_latent[0] - z_goal[0];
        let diff_1 = final_latent[1] - z_goal[1];
        f32::midpoint(diff_0 * diff_0, diff_1 * diff_1)
    }

    fn assert_close_slice(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (left_value, right_value) in left.iter().zip(right.iter()) {
            assert!(
                (*left_value - *right_value).abs() <= tolerance,
                "{left_value} != {right_value}"
            );
        }
    }
}
