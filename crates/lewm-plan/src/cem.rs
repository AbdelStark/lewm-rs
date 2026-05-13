//! Cross Entropy Method action search.

use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::LewmPlanError;

/// RFC 0013 RNG sub-stream used for all CEM action proposal draws.
pub const CEM_RNG_STREAM: &str = "rng:cem";

/// Candidate chunk size used when the cost batch would exceed the memory budget.
pub const DEFAULT_CEM_CHUNK_SIZE: usize = 250;

/// RFC 0006 memory fallback threshold for CEM cost evaluation.
pub const DEFAULT_CEM_MAX_BATCH_BYTES: usize = 18 * 1024 * 1024 * 1024;

/// Cross Entropy Method hyperparameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Cem {
    /// Number of CEM refinement iterations.
    pub n_iter: usize,
    /// Number of action candidates sampled per iteration.
    pub n_cand: usize,
    /// Number of lowest-cost candidates used to update the proposal.
    pub n_elite: usize,
    /// Number of action steps in each candidate sequence.
    pub horizon_plan: usize,
    /// Initial proposal standard deviation in normalized action space.
    pub sigma_init: f32,
    /// Lower bound on proposal standard deviation.
    pub sigma_min: f32,
    /// Candidate batch size for the RFC 0006 memory fallback.
    pub chunk_size: usize,
    /// Maximum estimated cost batch footprint before chunking.
    pub max_batch_bytes: usize,
}

impl Default for Cem {
    fn default() -> Self {
        Self {
            n_iter: 5,
            n_cand: 1000,
            n_elite: 100,
            horizon_plan: 5,
            sigma_init: 1.0,
            sigma_min: 0.05,
            chunk_size: DEFAULT_CEM_CHUNK_SIZE,
            max_batch_bytes: DEFAULT_CEM_MAX_BATCH_BYTES,
        }
    }
}

impl Cem {
    /// Validate CEM hyperparameters.
    ///
    /// # Errors
    ///
    /// Returns [`LewmPlanError::InvalidCemConfig`] when a hyperparameter is
    /// zero, non-finite, or incompatible with another hyperparameter.
    pub fn validate(&self) -> Result<(), LewmPlanError> {
        if self.n_iter == 0 {
            return Err(LewmPlanError::invalid_config("n_iter must be non-zero"));
        }

        if self.n_cand == 0 {
            return Err(LewmPlanError::invalid_config("n_cand must be non-zero"));
        }

        if self.n_elite == 0 {
            return Err(LewmPlanError::invalid_config("n_elite must be non-zero"));
        }

        if self.n_elite > self.n_cand {
            return Err(LewmPlanError::invalid_config(
                "n_elite must be less than or equal to n_cand",
            ));
        }

        if self.horizon_plan == 0 {
            return Err(LewmPlanError::invalid_config(
                "horizon_plan must be non-zero",
            ));
        }

        if !self.sigma_init.is_finite() || self.sigma_init <= 0.0 {
            return Err(LewmPlanError::invalid_config(
                "sigma_init must be finite and positive",
            ));
        }

        if !self.sigma_min.is_finite() || self.sigma_min <= 0.0 {
            return Err(LewmPlanError::invalid_config(
                "sigma_min must be finite and positive",
            ));
        }

        if self.sigma_min > self.sigma_init {
            return Err(LewmPlanError::invalid_config(
                "sigma_min must be less than or equal to sigma_init",
            ));
        }

        if self.chunk_size == 0 {
            return Err(LewmPlanError::invalid_config("chunk_size must be non-zero"));
        }

        if self.max_batch_bytes == 0 {
            return Err(LewmPlanError::invalid_config(
                "max_batch_bytes must be non-zero",
            ));
        }

        Ok(())
    }

    /// Run CEM with an RFC 0013 `rng:cem` sub-stream derived from `global_seed`.
    ///
    /// # Errors
    ///
    /// Returns [`LewmPlanError`] when the planner configuration, input shapes,
    /// RNG setup, or cost-model output is invalid.
    pub fn plan<M: CemCostModel>(
        &self,
        model: &M,
        input: CemPlanInput<'_>,
        global_seed: u64,
    ) -> Result<CemResult, LewmPlanError> {
        let mut rng = lewm_core::substream_rng(global_seed, CEM_RNG_STREAM)
            .map_err(|err| LewmPlanError::rng(err.to_string()))?;
        self.plan_with_rng(model, input, &mut rng)
    }

    /// Run CEM with a caller-supplied `rng:cem` RNG state.
    ///
    /// Use [`Cem::plan`] when starting from a global seed. This method exists for
    /// resume-aware eval loops that restore the RFC 0013 CEM sub-stream state.
    ///
    /// # Errors
    ///
    /// Returns [`LewmPlanError`] when the planner configuration, input shapes, or
    /// cost-model output is invalid.
    pub fn plan_with_rng<M: CemCostModel>(
        &self,
        model: &M,
        input: CemPlanInput<'_>,
        rng: &mut ChaCha20Rng,
    ) -> Result<CemResult, LewmPlanError> {
        self.validate()?;
        input.validate()?;

        let action_len = self
            .horizon_plan
            .checked_mul(input.action_dim)
            .ok_or_else(|| {
                LewmPlanError::invalid_input("horizon_plan * action_dim overflowed usize")
            })?;

        let candidate_len = self.n_cand.checked_mul(action_len).ok_or_else(|| {
            LewmPlanError::invalid_input("n_cand * horizon_plan * action_dim overflowed usize")
        })?;

        let mut mu = vec![0.0; action_len];
        let mut sigma = vec![self.sigma_init; action_len];
        let mut final_candidates = Vec::new();
        let mut final_costs = Vec::new();
        let mut trace = Vec::with_capacity(self.n_iter);

        for iter in 0..self.n_iter {
            let candidates = sample_candidates(self.n_cand, action_len, &mu, &sigma, rng);
            let costs = self.evaluate_costs(model, input, &candidates)?;
            let best_idx = best_cost_index(&costs)?;
            let elite_indices = elite_indices(&costs, self.n_elite);
            let elites = gather_elites(&candidates, action_len, &elite_indices);
            let (next_mu, next_sigma) =
                update_proposal(&elites, self.n_elite, action_len, self.sigma_min)?;
            let cost_min = costs[best_idx];
            let cost_mean = mean(&costs);
            mu = next_mu;
            sigma = next_sigma;
            trace.push(CemIterTrace {
                iter,
                cost_min,
                cost_mean,
                sigma_mean: mean(&sigma),
                cost_evals: costs.len(),
                chunked: self.should_chunk(input),
            });
            final_candidates = candidates;
            final_costs = costs;
        }

        debug_assert_eq!(final_candidates.len(), candidate_len);
        let best_idx = best_cost_index(&final_costs)?;
        let start = best_idx
            .checked_mul(action_len)
            .ok_or_else(|| LewmPlanError::invalid_cost("best candidate offset overflowed usize"))?;
        let end = start + action_len;

        Ok(CemResult {
            best_actions: final_candidates[start..end].to_vec(),
            best_cost: final_costs[best_idx],
            final_mu: mu,
            final_sigma: sigma,
            trace,
            horizon_plan: self.horizon_plan,
            action_dim: input.action_dim,
        })
    }

    fn evaluate_costs<M: CemCostModel>(
        &self,
        model: &M,
        input: CemPlanInput<'_>,
        candidates: &[f32],
    ) -> Result<Vec<f32>, LewmPlanError> {
        let action_len = self.horizon_plan * input.action_dim;
        if self.should_chunk(input) && self.n_cand > self.chunk_size {
            let mut costs = Vec::with_capacity(self.n_cand);
            for start_candidate in (0..self.n_cand).step_by(self.chunk_size) {
                let end_candidate = (start_candidate + self.chunk_size).min(self.n_cand);
                let start = start_candidate * action_len;
                let end = end_candidate * action_len;
                let request = CemCostRequest {
                    input,
                    candidates: &candidates[start..end],
                    batch_size: end_candidate - start_candidate,
                    batch_offset: start_candidate,
                    horizon_plan: self.horizon_plan,
                    no_grad: true,
                };
                costs.extend(evaluate_chunk(model, request)?);
            }
            Ok(costs)
        } else {
            let request = CemCostRequest {
                input,
                candidates,
                batch_size: self.n_cand,
                batch_offset: 0,
                horizon_plan: self.horizon_plan,
                no_grad: true,
            };
            evaluate_chunk(model, request)
        }
    }

    fn should_chunk(&self, input: CemPlanInput<'_>) -> bool {
        estimate_cost_batch_bytes(
            self.n_cand,
            input.history_len,
            input.latent_dim,
            self.horizon_plan,
            input.action_dim,
        ) > self.max_batch_bytes
    }
}

/// Inputs shared by all CEM candidate cost evaluations.
#[derive(Debug, Clone, Copy)]
pub struct CemPlanInput<'a> {
    /// Flattened latent history with shape `(history_len, latent_dim)`.
    pub z_history: &'a [f32],
    /// Number of history steps.
    pub history_len: usize,
    /// Latent feature dimension.
    pub latent_dim: usize,
    /// Goal latent with shape `(latent_dim)`.
    pub z_goal: &'a [f32],
    /// Per-step action dimension.
    pub action_dim: usize,
}

impl CemPlanInput<'_> {
    /// Validate shape metadata against the flattened buffers.
    ///
    /// # Errors
    ///
    /// Returns [`LewmPlanError::InvalidCemInput`] when any dimension is zero,
    /// multiplication overflows, or a buffer length does not match its shape.
    pub fn validate(&self) -> Result<(), LewmPlanError> {
        if self.history_len == 0 {
            return Err(LewmPlanError::invalid_input("history_len must be non-zero"));
        }

        if self.latent_dim == 0 {
            return Err(LewmPlanError::invalid_input("latent_dim must be non-zero"));
        }

        if self.action_dim == 0 {
            return Err(LewmPlanError::invalid_input("action_dim must be non-zero"));
        }

        let expected_history = self
            .history_len
            .checked_mul(self.latent_dim)
            .ok_or_else(|| {
                LewmPlanError::invalid_input("history_len * latent_dim overflowed usize")
            })?;

        if self.z_history.len() != expected_history {
            return Err(LewmPlanError::invalid_input(format!(
                "z_history length must be {expected_history}, got {}",
                self.z_history.len()
            )));
        }

        if self.z_goal.len() != self.latent_dim {
            return Err(LewmPlanError::invalid_input(format!(
                "z_goal length must be {}, got {}",
                self.latent_dim,
                self.z_goal.len()
            )));
        }

        if self
            .z_history
            .iter()
            .chain(self.z_goal.iter())
            .any(|value| !value.is_finite())
        {
            return Err(LewmPlanError::invalid_input(
                "z_history and z_goal must be finite",
            ));
        }

        Ok(())
    }
}

/// Batched cost request passed to a JEPA-compatible cost model.
#[derive(Debug, Clone, Copy)]
pub struct CemCostRequest<'a> {
    /// Shared latent inputs for every candidate in this batch.
    pub input: CemPlanInput<'a>,
    /// Flattened action candidates with shape `(batch_size, horizon_plan, action_dim)`.
    pub candidates: &'a [f32],
    /// Number of candidate sequences in this request.
    pub batch_size: usize,
    /// Offset of this batch in the full candidate set.
    pub batch_offset: usize,
    /// Number of action steps in each candidate sequence.
    pub horizon_plan: usize,
    /// Always true for `Cem::plan`; adapters should disable gradient tracking.
    pub no_grad: bool,
}

/// Cost-model boundary used by the planner.
///
/// A concrete `Jepa` adapter should implement this by forwarding to
/// `Jepa::get_cost` with `z_history` and `z_goal` expanded across the candidate
/// batch.
pub trait CemCostModel {
    /// Compute one scalar cost per candidate; smaller is better.
    ///
    /// # Errors
    ///
    /// Returns [`LewmPlanError`] when the cost model cannot evaluate the batch.
    fn get_cost(&self, request: CemCostRequest<'_>) -> Result<Vec<f32>, LewmPlanError>;
}

/// Result returned by [`Cem::plan`].
#[derive(Debug, Clone, PartialEq)]
pub struct CemResult {
    /// Best candidate action sequence, flattened `(horizon_plan, action_dim)`.
    pub best_actions: Vec<f32>,
    /// Cost of the best candidate from the final CEM iteration.
    pub best_cost: f32,
    /// Final proposal mean, flattened `(horizon_plan, action_dim)`.
    pub final_mu: Vec<f32>,
    /// Final proposal standard deviation, flattened `(horizon_plan, action_dim)`.
    pub final_sigma: Vec<f32>,
    /// Per-iteration CEM trace.
    pub trace: Vec<CemIterTrace>,
    /// Number of action steps represented by `best_actions`.
    pub horizon_plan: usize,
    /// Per-step action dimension represented by `best_actions`.
    pub action_dim: usize,
}

impl CemResult {
    /// Return the best action at a planning step.
    pub fn action_at(&self, step: usize) -> Option<&[f32]> {
        if step >= self.horizon_plan {
            return None;
        }

        let start = step * self.action_dim;
        let end = start + self.action_dim;
        Some(&self.best_actions[start..end])
    }
}

/// Per-iteration diagnostic metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CemIterTrace {
    /// Iteration index.
    pub iter: usize,
    /// Lowest candidate cost in this iteration.
    pub cost_min: f32,
    /// Mean candidate cost in this iteration.
    pub cost_mean: f32,
    /// Mean proposal standard deviation after this iteration's update.
    pub sigma_mean: f32,
    /// Number of candidate costs evaluated in this iteration.
    pub cost_evals: usize,
    /// Whether this iteration used chunked cost evaluation.
    pub chunked: bool,
}

fn sample_candidates(
    n_cand: usize,
    action_len: usize,
    mu: &[f32],
    sigma: &[f32],
    rng: &mut ChaCha20Rng,
) -> Vec<f32> {
    let mut candidates = Vec::with_capacity(n_cand * action_len);
    for _ in 0..n_cand {
        for action_idx in 0..action_len {
            let eps: f32 = StandardNormal.sample(rng);
            candidates.push(mu[action_idx] + sigma[action_idx] * eps);
        }
    }
    candidates
}

fn evaluate_chunk<M: CemCostModel>(
    model: &M,
    request: CemCostRequest<'_>,
) -> Result<Vec<f32>, LewmPlanError> {
    let action_len = request
        .horizon_plan
        .checked_mul(request.input.action_dim)
        .ok_or_else(|| {
            LewmPlanError::invalid_input("horizon_plan * action_dim overflowed usize")
        })?;
    let expected_candidates = request.batch_size.checked_mul(action_len).ok_or_else(|| {
        LewmPlanError::invalid_input("batch_size * horizon_plan * action_dim overflowed usize")
    })?;
    if request.candidates.len() != expected_candidates {
        return Err(LewmPlanError::invalid_input(format!(
            "candidate batch length must be {expected_candidates}, got {}",
            request.candidates.len()
        )));
    }

    let costs = model.get_cost(request)?;
    if costs.len() != request.batch_size {
        return Err(LewmPlanError::invalid_cost(format!(
            "get_cost returned {} costs for {} candidates",
            costs.len(),
            request.batch_size
        )));
    }

    if costs.iter().any(|cost| !cost.is_finite()) {
        return Err(LewmPlanError::invalid_cost(
            "get_cost returned non-finite costs",
        ));
    }

    Ok(costs)
}

fn elite_indices(costs: &[f32], n_elite: usize) -> Vec<usize> {
    let mut indexed = costs.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|(left_idx, left_cost), (right_idx, right_cost)| {
        left_cost
            .total_cmp(right_cost)
            .then_with(|| left_idx.cmp(right_idx))
    });

    indexed
        .into_iter()
        .take(n_elite)
        .map(|(idx, _cost)| idx)
        .collect()
}

fn best_cost_index(costs: &[f32]) -> Result<usize, LewmPlanError> {
    costs
        .iter()
        .copied()
        .enumerate()
        .min_by(|(left_idx, left_cost), (right_idx, right_cost)| {
            left_cost
                .total_cmp(right_cost)
                .then_with(|| left_idx.cmp(right_idx))
        })
        .map(|(idx, _cost)| idx)
        .ok_or_else(|| LewmPlanError::invalid_cost("cost vector must not be empty"))
}

fn gather_elites(candidates: &[f32], action_len: usize, elite_indices: &[usize]) -> Vec<f32> {
    let mut elites = Vec::with_capacity(elite_indices.len() * action_len);
    for candidate_idx in elite_indices {
        let start = candidate_idx * action_len;
        let end = start + action_len;
        elites.extend_from_slice(&candidates[start..end]);
    }
    elites
}

#[allow(clippy::cast_precision_loss)]
fn update_proposal(
    elites: &[f32],
    n_elite: usize,
    action_len: usize,
    sigma_min: f32,
) -> Result<(Vec<f32>, Vec<f32>), LewmPlanError> {
    if elites.len() != n_elite * action_len {
        return Err(LewmPlanError::invalid_input(format!(
            "elite buffer length must be {}, got {}",
            n_elite * action_len,
            elites.len()
        )));
    }

    let denom = n_elite as f32;
    let mut mu = vec![0.0; action_len];
    for elite in elites.chunks_exact(action_len) {
        for (idx, value) in elite.iter().enumerate() {
            mu[idx] += *value;
        }
    }
    for value in &mut mu {
        *value /= denom;
    }

    let mut variance = vec![0.0; action_len];
    for elite in elites.chunks_exact(action_len) {
        for (idx, value) in elite.iter().enumerate() {
            let diff = *value - mu[idx];
            variance[idx] += diff * diff;
        }
    }
    let sigma = variance
        .into_iter()
        .map(|value| (value / denom).sqrt().max(sigma_min))
        .collect();

    Ok((mu, sigma))
}

#[allow(clippy::cast_precision_loss)]
fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

fn estimate_cost_batch_bytes(
    n_cand: usize,
    history_len: usize,
    latent_dim: usize,
    horizon_plan: usize,
    action_dim: usize,
) -> usize {
    let per_candidate_scalars = history_len
        .saturating_mul(latent_dim)
        .saturating_add(latent_dim)
        .saturating_add(horizon_plan.saturating_mul(action_dim));
    n_cand
        .saturating_mul(per_candidate_scalars)
        .saturating_mul(std::mem::size_of::<f32>())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    #[derive(Debug)]
    struct QuadraticCost {
        target: Vec<f32>,
    }

    impl CemCostModel for QuadraticCost {
        fn get_cost(&self, request: CemCostRequest<'_>) -> Result<Vec<f32>, LewmPlanError> {
            assert!(request.no_grad);
            let action_len = request.horizon_plan * request.input.action_dim;
            assert_eq!(self.target.len(), action_len);
            Ok(request
                .candidates
                .chunks_exact(action_len)
                .map(|candidate| {
                    candidate
                        .iter()
                        .zip(self.target.iter())
                        .map(|(actual, expected)| {
                            let diff = actual - expected;
                            diff * diff
                        })
                        .sum::<f32>()
                })
                .collect())
        }
    }

    #[derive(Debug, Default)]
    struct RecordingCost {
        batches: Mutex<Vec<usize>>,
    }

    impl CemCostModel for RecordingCost {
        fn get_cost(&self, request: CemCostRequest<'_>) -> Result<Vec<f32>, LewmPlanError> {
            self.batches
                .lock()
                .expect("test mutex should not be poisoned")
                .push(request.batch_size);
            Ok(vec![0.0; request.batch_size])
        }
    }

    fn input() -> CemPlanInput<'static> {
        CemPlanInput {
            z_history: &[0.0, 0.0, 0.0, 0.0],
            history_len: 2,
            latent_dim: 2,
            z_goal: &[0.0, 0.0],
            action_dim: 2,
        }
    }

    #[test]
    fn cem_proposal_update_correct() {
        let elites = vec![1.0, 3.0, 3.0, 7.0];
        let (mu, sigma) =
            update_proposal(&elites, 2, 2, 0.05).expect("valid elite buffer should update");

        assert_eq!(mu, vec![2.0, 5.0]);
        assert_eq!(sigma, vec![1.0, 2.0]);
    }

    #[test]
    fn cem_seed_determinism() {
        let cem = Cem {
            n_iter: 4,
            n_cand: 128,
            n_elite: 16,
            horizon_plan: 3,
            ..Cem::default()
        };
        let target = vec![0.2; cem.horizon_plan * input().action_dim];
        let cost = QuadraticCost { target };

        let left = cem.plan(&cost, input(), 0).expect("left run should pass");
        let right = cem.plan(&cost, input(), 0).expect("right run should pass");
        let different = cem
            .plan(&cost, input(), 1)
            .expect("different-seed run should pass");

        assert_eq!(left.best_actions, right.best_actions);
        assert_eq!(left.trace, right.trace);
        assert_ne!(left.best_actions, different.best_actions);
    }

    #[test]
    fn cem_converges_on_toy_quadratic() {
        let cem = Cem {
            n_iter: 8,
            n_cand: 512,
            n_elite: 64,
            horizon_plan: 5,
            sigma_min: 0.01,
            ..Cem::default()
        };
        let target = [0.5, -0.25]
            .into_iter()
            .cycle()
            .take(cem.horizon_plan * input().action_dim)
            .collect::<Vec<_>>();
        let cost = QuadraticCost {
            target: target.clone(),
        };
        let result = cem
            .plan(&cost, input(), 0)
            .expect("toy quadratic CEM should pass");

        assert!(result.best_cost < 0.05, "cost={}", result.best_cost);
        for (actual, expected) in result.best_actions.iter().zip(target.iter()) {
            assert!((actual - expected).abs() < 0.25);
        }
    }

    #[test]
    fn cem_chunks_cost_when_estimate_exceeds_memory_budget() {
        let cem = Cem {
            n_iter: 1,
            n_cand: 20,
            n_elite: 4,
            horizon_plan: 2,
            chunk_size: 7,
            max_batch_bytes: 1,
            ..Cem::default()
        };
        let cost = RecordingCost::default();

        let result = cem
            .plan(&cost, input(), 0)
            .expect("chunked planning should pass");

        assert!(result.trace[0].chunked);
        assert_eq!(
            *cost
                .batches
                .lock()
                .expect("test mutex should not be poisoned"),
            vec![7, 7, 6]
        );
    }
}
