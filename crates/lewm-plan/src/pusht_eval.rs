//! `PushT` evaluation loop and JSON-RPC simulator boundary.

use std::ffi::OsStr;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use base64::Engine as _;
use lewm_data::{ActionNormalizer, ImagePreprocessor};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::EvalError;

const DEFAULT_IMAGE_SIZE: usize = 224;
const RGB_CHANNELS: usize = 3;

/// Static `PushT` eval configuration loaded from `configs/pusht.toml`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PushtConfigFile {
    /// Evaluation-loop settings.
    pub eval: PushtEvalConfig,
    /// CEM defaults pinned next to the episode set for RFC 0006 traceability.
    pub cem: PushtCemConfig,
}

impl PushtConfigFile {
    /// Load a `PushT` eval config from TOML.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read, TOML parsing fails, or
    /// the decoded values violate the RFC 0006 eval contract.
    pub fn from_toml_path(path: impl AsRef<Path>) -> Result<Self, EvalError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| EvalError::io(path, source))?;
        let config: Self = toml::from_str(&text).map_err(|source| EvalError::toml(path, source))?;
        config.validate()?;
        Ok(config)
    }

    /// Validate cross-field invariants.
    ///
    /// # Errors
    ///
    /// Returns an error when the config has no episodes, invalid action stats,
    /// invalid CEM sizes, or a zero step budget.
    pub fn validate(&self) -> Result<(), EvalError> {
        self.eval.validate()?;
        self.cem.validate()
    }
}

/// CEM hyperparameters used by the `PushT` planner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PushtCemConfig {
    /// Number of CEM iterations.
    pub n_iter: usize,
    /// Number of candidate action sequences per iteration.
    pub n_cand: usize,
    /// Number of elite candidates used to update the proposal.
    pub n_elite: usize,
    /// Planning horizon in action steps.
    pub horizon_plan: usize,
    /// Initial normalized action standard deviation.
    pub sigma_init: f32,
    /// Minimum normalized action standard deviation.
    pub sigma_min: f32,
}

impl PushtCemConfig {
    /// Validate the RFC 0006 CEM defaults.
    ///
    /// # Errors
    ///
    /// Returns an error when the proposal sizes or sigma values are invalid.
    pub fn validate(&self) -> Result<(), EvalError> {
        if self.n_iter == 0 {
            return Err(EvalError::InvalidConfig(
                "cem.n_iter must be greater than zero".to_owned(),
            ));
        }
        if self.n_cand == 0 {
            return Err(EvalError::InvalidConfig(
                "cem.n_cand must be greater than zero".to_owned(),
            ));
        }
        if self.n_elite == 0 || self.n_elite > self.n_cand {
            return Err(EvalError::InvalidConfig(format!(
                "cem.n_elite must be in 1..={} but found {}",
                self.n_cand, self.n_elite
            )));
        }
        if self.horizon_plan == 0 {
            return Err(EvalError::InvalidConfig(
                "cem.horizon_plan must be greater than zero".to_owned(),
            ));
        }
        if !self.sigma_init.is_finite() || self.sigma_init <= 0.0 {
            return Err(EvalError::InvalidConfig(
                "cem.sigma_init must be finite and positive".to_owned(),
            ));
        }
        if !self.sigma_min.is_finite() || self.sigma_min <= 0.0 {
            return Err(EvalError::InvalidConfig(
                "cem.sigma_min must be finite and positive".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Runtime settings for the `PushT` eval loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PushtEvalConfig {
    /// Deterministic held-out episode identifiers.
    pub episode_ids: Vec<u32>,
    /// Maximum simulator steps per episode.
    pub max_steps_per_episode: u32,
    /// Global eval seed forwarded to the simulator reset and planner adapter.
    pub seed: u64,
    /// Latent/observation history length consumed by the planner.
    pub history_size: usize,
    /// Per-dimension raw-action mean used for inverse normalization.
    pub action_mean: Vec<f32>,
    /// Per-dimension raw-action standard deviation used for inverse normalization.
    pub action_std: Vec<f32>,
}

impl PushtEvalConfig {
    /// Validate the eval-loop contract.
    ///
    /// # Errors
    ///
    /// Returns an error when there are no episodes, duplicate episode IDs, no
    /// history, no step budget, or invalid action normalization statistics.
    pub fn validate(&self) -> Result<(), EvalError> {
        if self.episode_ids.is_empty() {
            return Err(EvalError::InvalidConfig(
                "eval.episode_ids must contain at least one episode".to_owned(),
            ));
        }
        let mut sorted = self.episode_ids.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(EvalError::InvalidConfig(
                "eval.episode_ids must not contain duplicates".to_owned(),
            ));
        }
        if self.max_steps_per_episode == 0 {
            return Err(EvalError::InvalidConfig(
                "eval.max_steps_per_episode must be greater than zero".to_owned(),
            ));
        }
        if self.history_size == 0 {
            return Err(EvalError::InvalidConfig(
                "eval.history_size must be greater than zero".to_owned(),
            ));
        }
        self.action_normalizer()?;
        Ok(())
    }

    /// Build an action normalizer from the configured action stats.
    ///
    /// # Errors
    ///
    /// Returns an error when the stats have invalid lengths or values.
    pub fn action_normalizer(&self) -> Result<ActionNormalizer, EvalError> {
        ActionNormalizer::new(self.action_mean.clone(), self.action_std.clone())
            .map_err(EvalError::from)
    }

    /// Number of action dimensions.
    pub fn action_dim(&self) -> usize {
        self.action_mean.len()
    }
}

/// Request passed from the evaluator into a planner implementation.
#[derive(Debug)]
pub struct PushtPlanRequest<'a> {
    /// Current held-out episode.
    pub episode_id: u32,
    /// Zero-based simulator step index.
    pub step_index: u32,
    /// Global eval seed.
    pub seed: u64,
    /// Most recent normalized CHW observations, oldest first.
    pub obs_history_chw: &'a [Vec<f32>],
    /// Simulator state vector returned by the sidecar.
    pub state: &'a [f32],
}

/// Planner output for one simulator step.
#[derive(Debug, Clone, PartialEq)]
pub struct PushtPlan {
    /// First normalized action to execute in the simulator.
    pub normalized_action: Vec<f32>,
    /// Planner cost associated with the chosen action sequence.
    pub cost: f32,
}

/// Model-backed or deterministic planner used by [`PushtEvaluator`].
pub trait PushtPlanner {
    /// Plan one normalized action from the current observation history.
    ///
    /// # Errors
    ///
    /// Returns an error when model inference, CEM, or planner output validation
    /// fails.
    fn plan(&mut self, request: PushtPlanRequest<'_>) -> Result<PushtPlan, EvalError>;
}

/// `PushT` simulator observation returned by reset and step calls.
#[derive(Debug, Clone, PartialEq)]
pub struct PushtObservation {
    /// Flat HWC RGB frame.
    pub frame_hwc_rgb: Vec<u8>,
    /// Frame shape as `[height, width, channels]`.
    pub frame_shape: [usize; 3],
    /// Simulator state vector.
    pub state: Vec<f32>,
    /// Reward returned by the simulator, if present.
    pub reward: Option<f32>,
    /// Whether the simulator episode is done.
    pub done: bool,
    /// Whether the task success criterion has fired.
    pub success: bool,
}

impl PushtObservation {
    /// Validate and construct an observation.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape is not RGB, overflows, or does not match
    /// the frame byte count.
    #[allow(clippy::fn_params_excessive_bools)]
    pub fn new(
        frame_hwc_rgb: Vec<u8>,
        frame_shape: [usize; 3],
        state: Vec<f32>,
        reward: Option<f32>,
        done: bool,
        success: bool,
    ) -> Result<Self, EvalError> {
        validate_frame(&frame_hwc_rgb, frame_shape)?;
        Ok(Self {
            frame_hwc_rgb,
            frame_shape,
            state,
            reward,
            done,
            success,
        })
    }
}

/// Long-lived `PushT` simulator boundary.
pub trait PushtRpc {
    /// Reset the simulator to a deterministic episode start.
    ///
    /// # Errors
    ///
    /// Returns an error when the sidecar fails or returns an invalid frame.
    fn reset(&mut self, episode_id: u32, seed: u64) -> Result<PushtObservation, EvalError>;

    /// Step the simulator with one raw action.
    ///
    /// # Errors
    ///
    /// Returns an error when the sidecar fails or returns an invalid frame.
    fn step(&mut self, action: &[f32]) -> Result<PushtObservation, EvalError>;

    /// Ask the simulator sidecar to close cleanly.
    ///
    /// # Errors
    ///
    /// Returns an error when the close request cannot be delivered.
    fn close(&mut self) -> Result<(), EvalError> {
        Ok(())
    }
}

/// `PushT` evaluation driver.
#[derive(Debug)]
pub struct PushtEvaluator<P, R> {
    planner: P,
    image_preproc: ImagePreprocessor,
    action_norm: ActionNormalizer,
    rpc: R,
    config: PushtEvalConfig,
}

impl<P, R> PushtEvaluator<P, R>
where
    P: PushtPlanner,
    R: PushtRpc,
{
    /// Build a `PushT` evaluator.
    ///
    /// # Errors
    ///
    /// Returns an error when the eval config or action normalizer is invalid.
    pub fn new(
        planner: P,
        image_preproc: ImagePreprocessor,
        action_norm: ActionNormalizer,
        rpc: R,
        config: PushtEvalConfig,
    ) -> Result<Self, EvalError> {
        config.validate()?;
        if action_norm.action_dim() != config.action_dim() {
            return Err(EvalError::InvalidConfig(format!(
                "action normalizer has {} dims but config has {} dims",
                action_norm.action_dim(),
                config.action_dim()
            )));
        }
        Ok(Self {
            planner,
            image_preproc,
            action_norm,
            rpc,
            config,
        })
    }

    /// Run the configured `PushT` episode set.
    ///
    /// # Errors
    ///
    /// Returns an error when simulator reset/step, image preprocessing, planner
    /// inference, or action inverse-normalization fails.
    pub fn run(&mut self) -> Result<PushtEvalReport, EvalError> {
        let start = Instant::now();
        let mut per_episode = Vec::with_capacity(self.config.episode_ids.len());
        let mut trajectories = Vec::new();
        let mut total_steps = 0_u32;

        for episode_id in self.config.episode_ids.clone() {
            let outcome = self.run_episode(episode_id, &mut trajectories)?;
            total_steps = total_steps
                .checked_add(outcome.steps_taken)
                .ok_or_else(|| {
                    EvalError::InvalidData("total PushT eval step count overflowed u32".to_owned())
                })?;
            per_episode.push(outcome);
        }

        self.rpc.close()?;
        let wins = per_episode.iter().filter(|outcome| outcome.success).count();
        let success_rate = ratio(wins, per_episode.len());

        Ok(PushtEvalReport {
            success_rate,
            per_episode,
            wall_time_s: start.elapsed().as_secs_f32(),
            total_steps,
            seed: self.config.seed,
            max_steps_per_episode: self.config.max_steps_per_episode,
            trajectories,
        })
    }

    fn run_episode(
        &mut self,
        episode_id: u32,
        trajectories: &mut Vec<TrajectoryStep>,
    ) -> Result<EpisodeOutcome, EvalError> {
        let initial_obs = self.rpc.reset(episode_id, self.config.seed)?;
        let initial_chw = self.preprocess(&initial_obs)?;
        let mut obs_history = vec![initial_chw; self.config.history_size];
        let mut state = initial_obs.state;
        let mut steps_taken = 0_u32;
        let mut final_cost = 0.0_f32;
        let mut success = initial_obs.success;

        if success {
            return Ok(EpisodeOutcome {
                episode_id,
                success,
                steps_taken,
                final_cost,
                trajectory_summary: TrajectorySummary {
                    first_state: state.clone(),
                    final_state: state,
                },
            });
        }

        let first_state = state.clone();
        for step_index in 0..self.config.max_steps_per_episode {
            let plan = self.planner.plan(PushtPlanRequest {
                episode_id,
                step_index,
                seed: self.config.seed,
                obs_history_chw: &obs_history,
                state: &state,
            })?;
            if plan.normalized_action.len() != self.action_norm.action_dim() {
                return Err(EvalError::InvalidData(format!(
                    "planner returned {} action dims but normalizer expects {}",
                    plan.normalized_action.len(),
                    self.action_norm.action_dim()
                )));
            }
            if !plan.cost.is_finite() {
                return Err(EvalError::InvalidData(
                    "planner cost must be finite".to_owned(),
                ));
            }
            let raw_action = self.action_norm.inverse(&plan.normalized_action)?;
            let obs = self.rpc.step(&raw_action)?;
            let next_chw = self.preprocess(&obs)?;
            final_cost = plan.cost;
            steps_taken = step_index + 1;
            success = obs.success;
            trajectories.push(TrajectoryStep {
                episode_id,
                step_index,
                action: raw_action,
                cost: plan.cost,
                reward: obs.reward,
                done: obs.done,
                success: obs.success,
            });
            state = obs.state;
            obs_history.remove(0);
            obs_history.push(next_chw);
            if success || obs.done {
                break;
            }
        }

        Ok(EpisodeOutcome {
            episode_id,
            success,
            steps_taken,
            final_cost,
            trajectory_summary: TrajectorySummary {
                first_state,
                final_state: state,
            },
        })
    }

    fn preprocess(&self, obs: &PushtObservation) -> Result<Vec<f32>, EvalError> {
        let [height, width, channels] = obs.frame_shape;
        if channels != RGB_CHANNELS {
            return Err(EvalError::InvalidData(format!(
                "expected RGB frame with {RGB_CHANNELS} channels, found {channels}"
            )));
        }
        let height = u32::try_from(height).map_err(|_| {
            EvalError::InvalidData(format!("frame height {height} does not fit u32"))
        })?;
        let width = u32::try_from(width)
            .map_err(|_| EvalError::InvalidData(format!("frame width {width} does not fit u32")))?;
        self.image_preproc
            .apply(&obs.frame_hwc_rgb, height, width)
            .map_err(EvalError::from)
    }
}

/// JSON-serializable `PushT` report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PushtEvalReport {
    /// `wins / episode_count`.
    pub success_rate: f32,
    /// Per-episode outcomes.
    pub per_episode: Vec<EpisodeOutcome>,
    /// Total eval wall-clock time in seconds.
    pub wall_time_s: f32,
    /// Total simulator steps executed.
    pub total_steps: u32,
    /// Global eval seed.
    pub seed: u64,
    /// Step cap applied to each episode.
    pub max_steps_per_episode: u32,
    /// Per-step trajectory records written to Parquet as the trace artifact.
    pub trajectories: Vec<TrajectoryStep>,
}

/// Outcome for one `PushT` episode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeOutcome {
    /// Held-out episode identifier.
    pub episode_id: u32,
    /// Whether the simulator success flag fired.
    pub success: bool,
    /// Number of simulator steps taken.
    pub steps_taken: u32,
    /// Last planner cost observed before termination.
    pub final_cost: f32,
    /// Compact start/end simulator-state summary.
    pub trajectory_summary: TrajectorySummary,
}

/// Compact trajectory summary embedded in `results.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectorySummary {
    /// Simulator state immediately after reset.
    pub first_state: Vec<f32>,
    /// Simulator state at episode termination.
    pub final_state: Vec<f32>,
}

/// Per-step trajectory row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryStep {
    /// Held-out episode identifier.
    pub episode_id: u32,
    /// Zero-based step index.
    pub step_index: u32,
    /// Raw simulator action after inverse normalization.
    pub action: Vec<f32>,
    /// Planner cost for the selected action sequence.
    pub cost: f32,
    /// Simulator reward, if present.
    pub reward: Option<f32>,
    /// Whether the simulator ended on this step.
    pub done: bool,
    /// Whether the task success flag fired on this step.
    pub success: bool,
}

/// Deterministic planner used by plumbing tests and mock CLI runs.
#[derive(Debug, Clone)]
pub struct StaticPushtPlanner {
    normalized_action: Vec<f32>,
    cost: f32,
}

impl StaticPushtPlanner {
    /// Build a zero-action planner for the requested action dimension.
    pub fn zeros(action_dim: usize) -> Self {
        Self {
            normalized_action: vec![0.0; action_dim],
            cost: 0.0,
        }
    }
}

impl PushtPlanner for StaticPushtPlanner {
    fn plan(&mut self, _request: PushtPlanRequest<'_>) -> Result<PushtPlan, EvalError> {
        Ok(PushtPlan {
            normalized_action: self.normalized_action.clone(),
            cost: self.cost,
        })
    }
}

/// Deterministic in-process `PushT` sidecar used by tests.
#[derive(Debug, Clone)]
pub struct MockPushtRpc {
    success_after_steps: u32,
    image_size: usize,
    step_count: u32,
    episode_id: u32,
}

impl MockPushtRpc {
    /// Build a mock sidecar that succeeds after `success_after_steps`.
    pub fn new(success_after_steps: u32) -> Self {
        Self {
            success_after_steps,
            image_size: DEFAULT_IMAGE_SIZE,
            step_count: 0,
            episode_id: 0,
        }
    }

    #[allow(clippy::fn_params_excessive_bools)]
    fn observation(
        &self,
        reward: Option<f32>,
        done: bool,
        success: bool,
    ) -> Result<PushtObservation, EvalError> {
        let frame_len = self
            .image_size
            .checked_mul(self.image_size)
            .and_then(|pixels| pixels.checked_mul(RGB_CHANNELS))
            .ok_or_else(|| EvalError::InvalidData("mock frame size overflowed".to_owned()))?;
        PushtObservation::new(
            vec![0; frame_len],
            [self.image_size, self.image_size, RGB_CHANNELS],
            vec![
                if success { 1.0 } else { 0.0 },
                if done { 1.0 } else { 0.0 },
            ],
            reward,
            done,
            success,
        )
    }
}

impl PushtRpc for MockPushtRpc {
    fn reset(&mut self, episode_id: u32, _seed: u64) -> Result<PushtObservation, EvalError> {
        self.episode_id = episode_id;
        self.step_count = 0;
        self.observation(None, false, false)
    }

    fn step(&mut self, action: &[f32]) -> Result<PushtObservation, EvalError> {
        if action.is_empty() {
            return Err(EvalError::InvalidData(
                "mock PushT step requires at least one action dimension".to_owned(),
            ));
        }
        self.step_count = self.step_count.checked_add(1).ok_or_else(|| {
            EvalError::InvalidData("mock PushT step count overflowed u32".to_owned())
        })?;
        let success = self.step_count >= self.success_after_steps;
        self.observation(Some(if success { 1.0 } else { 0.0 }), success, success)
    }
}

/// Long-lived subprocess JSON-RPC client for `python/pusht_runner.py`.
#[derive(Debug)]
pub struct SubprocessPushtRpc {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl SubprocessPushtRpc {
    /// Spawn a subprocess RPC sidecar.
    ///
    /// # Errors
    ///
    /// Returns an error when the process cannot be spawned or stdio pipes are
    /// unavailable.
    pub fn spawn(program: impl AsRef<OsStr>, args: &[String]) -> Result<Self, EvalError> {
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|source| EvalError::Rpc(format!("spawn sidecar: {source}")))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| EvalError::Rpc("sidecar stdin pipe unavailable".to_owned()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| EvalError::Rpc("sidecar stdout pipe unavailable".to_owned()))?;
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
        })
    }

    /// Spawn the repository's Python `PushT` sidecar.
    ///
    /// # Errors
    ///
    /// Returns an error when the Python process cannot be spawned.
    pub fn spawn_python_runner(
        python: impl AsRef<OsStr>,
        runner_path: impl AsRef<Path>,
        mock: bool,
    ) -> Result<Self, EvalError> {
        let mut args = vec!["-u".to_owned(), runner_path.as_ref().display().to_string()];
        if mock {
            args.push("--mock".to_owned());
        }
        Self::spawn(python, &args)
    }

    fn request(
        &mut self,
        method: &str,
        params: impl Serialize,
    ) -> Result<serde_json::Value, EvalError> {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| EvalError::Rpc("JSON-RPC request id overflowed".to_owned()))?;
        let request = json!({
            "id": id,
            "method": method,
            "params": params,
        });
        let line = serde_json::to_string(&request)
            .map_err(|source| EvalError::json("serializing JSON-RPC request", source))?;
        writeln!(self.stdin, "{line}")
            .map_err(|source| EvalError::Rpc(format!("write request to sidecar: {source}")))?;
        self.stdin
            .flush()
            .map_err(|source| EvalError::Rpc(format!("flush sidecar stdin: {source}")))?;

        let mut response_line = String::new();
        let bytes = self
            .stdout
            .read_line(&mut response_line)
            .map_err(|source| EvalError::Rpc(format!("read response from sidecar: {source}")))?;
        if bytes == 0 {
            return Err(EvalError::Rpc(
                "sidecar closed stdout before sending a response".to_owned(),
            ));
        }
        let response: serde_json::Value = serde_json::from_str(&response_line)
            .map_err(|source| EvalError::json("parsing JSON-RPC response", source))?;
        if response
            .get("id")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|response_id| response_id != id)
        {
            return Err(EvalError::Rpc(format!(
                "sidecar response id did not match request id {id}: {response}"
            )));
        }
        if response
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .is_some_and(|ok| !ok)
        {
            return Err(EvalError::Rpc(format!(
                "sidecar returned error: {}",
                response
                    .get("error")
                    .map_or_else(|| response.to_string(), serde_json::Value::to_string)
            )));
        }
        Ok(response.get("result").cloned().unwrap_or(response))
    }
}

impl PushtRpc for SubprocessPushtRpc {
    fn reset(&mut self, episode_id: u32, seed: u64) -> Result<PushtObservation, EvalError> {
        let value = self.request("reset", json!({ "episode": episode_id, "seed": seed }))?;
        decode_rpc_observation(value)
    }

    fn step(&mut self, action: &[f32]) -> Result<PushtObservation, EvalError> {
        let value = self.request("step", json!({ "action": action }))?;
        decode_rpc_observation(value)
    }

    fn close(&mut self) -> Result<(), EvalError> {
        let _value = self.request("close", json!({}))?;
        Ok(())
    }
}

impl Drop for SubprocessPushtRpc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Debug, Deserialize)]
struct RpcObservationPayload {
    #[serde(alias = "frame")]
    obs: String,
    #[serde(default, alias = "obs_shape", alias = "image_shape")]
    frame_shape: Option<[usize; 3]>,
    #[serde(default)]
    state: Vec<f32>,
    #[serde(default)]
    reward: Option<f32>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    success: bool,
}

fn decode_rpc_observation(value: serde_json::Value) -> Result<PushtObservation, EvalError> {
    let payload: RpcObservationPayload = serde_json::from_value(value)
        .map_err(|source| EvalError::json("decoding PushT RPC observation", source))?;
    let frame = base64::engine::general_purpose::STANDARD
        .decode(payload.obs)
        .map_err(|source| EvalError::Rpc(format!("decode base64 RGB frame: {source}")))?;
    PushtObservation::new(
        frame,
        payload
            .frame_shape
            .unwrap_or([DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, RGB_CHANNELS]),
        payload.state,
        payload.reward,
        payload.done,
        payload.success,
    )
}

fn validate_frame(frame: &[u8], shape: [usize; 3]) -> Result<(), EvalError> {
    let [height, width, channels] = shape;
    if channels != RGB_CHANNELS {
        return Err(EvalError::InvalidData(format!(
            "expected RGB frame with {RGB_CHANNELS} channels, found {channels}"
        )));
    }
    let expected_len = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| EvalError::InvalidData("RGB frame shape overflowed".to_owned()))?;
    if frame.len() != expected_len {
        return Err(EvalError::InvalidData(format!(
            "RGB frame length {} does not match shape {:?} length {expected_len}",
            frame.len(),
            shape
        )));
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn test_config(episode_ids: Vec<u32>) -> PushtEvalConfig {
        PushtEvalConfig {
            episode_ids,
            max_steps_per_episode: 10,
            seed: 0,
            history_size: 3,
            action_mean: vec![0.0, 0.0],
            action_std: vec![1.0, 1.0],
        }
    }

    #[test]
    fn pusht_eval_loop_terminates_correctly() -> Result<(), Box<dyn std::error::Error>> {
        let config = test_config(vec![17, 23]);
        let action_norm = config.action_normalizer()?;
        let mut evaluator = PushtEvaluator::new(
            StaticPushtPlanner::zeros(config.action_dim()),
            ImagePreprocessor::default(),
            action_norm,
            MockPushtRpc::new(5),
            config,
        )?;

        let report = evaluator.run()?;

        assert!((report.success_rate - 1.0).abs() <= f32::EPSILON);
        assert_eq!(report.total_steps, 10);
        assert_eq!(report.per_episode.len(), 2);
        assert!(report.per_episode.iter().all(|episode| episode.success));
        assert!(
            report
                .per_episode
                .iter()
                .all(|episode| episode.steps_taken == 5)
        );
        assert_eq!(report.trajectories.len(), 10);
        Ok(())
    }

    #[test]
    fn pusht_config_pins_default_episode_set() -> Result<(), Box<dyn std::error::Error>> {
        let config_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/pusht.toml");
        let config = PushtConfigFile::from_toml_path(config_path)?;

        assert_eq!(config.eval.episode_ids.len(), 50);
        assert_eq!(config.eval.episode_ids.first(), Some(&25));
        assert_eq!(config.eval.episode_ids.last(), Some(&1023));
        assert_eq!(config.eval.max_steps_per_episode, 100);
        assert_eq!(config.eval.seed, 0);
        assert_eq!(config.eval.history_size, 3);
        assert_eq!(config.cem.n_iter, 5);
        assert_eq!(config.cem.n_cand, 1000);
        assert_eq!(config.cem.n_elite, 100);
        assert_eq!(config.cem.horizon_plan, 5);
        Ok(())
    }
}
