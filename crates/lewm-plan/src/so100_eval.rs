//! SO-100 latent-rollout evaluation and report artifact generation.

use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fs::{self, File},
    path::Path,
    sync::Arc,
};

use arrow_array::{ArrayRef, Float64Array, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_writer::ArrowWriter;
use serde::{Deserialize, Serialize};

use crate::EvalError;

/// Latent embedding vector in model latent space.
pub type LatentVector = Vec<f64>;

/// Normalized action vector replayed through the predictor.
pub type ActionVector = Vec<f64>;

/// One SO-100 held-out episode represented after frame encoding.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct So100Episode {
    /// Dataset episode identifier.
    pub episode_id: u32,
    /// Encoded latent trajectory for all frames in episode order.
    pub target_latents: Vec<LatentVector>,
    /// Recorded expert actions in episode order.
    pub expert_actions: Vec<ActionVector>,
}

/// Configuration for the SO-100 latent evaluator.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct So100EvalConfig {
    /// Number of latent history tokens used to seed the predictor.
    pub history_size: usize,
    /// Spearman floor for the pass/partial/null policy.
    pub spearman_floor: f64,
}

impl Default for So100EvalConfig {
    fn default() -> Self {
        Self {
            history_size: 3,
            spearman_floor: 0.6,
        }
    }
}

/// Model adapter used by the SO-100 evaluator.
pub trait So100RolloutModel {
    /// Prepare model-side state before an episode is replayed.
    ///
    /// # Errors
    ///
    /// Returns an error when the adapter cannot initialize state for the
    /// requested episode.
    fn begin_episode(&mut self, episode_id: u32) -> Result<(), EvalError>;

    /// Predict the next latent from current history and the recorded expert action.
    ///
    /// # Errors
    ///
    /// Returns an error when the adapter cannot produce a finite latent with the
    /// configured model or recorded fixture.
    fn predict_next(
        &mut self,
        history: &[LatentVector],
        action: &[f64],
    ) -> Result<LatentVector, EvalError>;
}

/// Deterministic model adapter for tests and encoded-latent CLI runs.
#[derive(Debug, Clone)]
pub struct RecordedRolloutModel {
    predictions_by_episode: BTreeMap<u32, Vec<LatentVector>>,
    active_episode: Option<u32>,
    cursor: usize,
}

impl RecordedRolloutModel {
    /// Create a recorded rollout adapter keyed by episode id.
    #[must_use]
    pub fn new(predictions_by_episode: BTreeMap<u32, Vec<LatentVector>>) -> Self {
        Self {
            predictions_by_episode,
            active_episode: None,
            cursor: 0,
        }
    }
}

impl So100RolloutModel for RecordedRolloutModel {
    fn begin_episode(&mut self, episode_id: u32) -> Result<(), EvalError> {
        if !self.predictions_by_episode.contains_key(&episode_id) {
            return Err(EvalError::invalid_episode(
                episode_id,
                "missing recorded predictions for episode",
            ));
        }
        self.active_episode = Some(episode_id);
        self.cursor = 0;
        Ok(())
    }

    fn predict_next(
        &mut self,
        _history: &[LatentVector],
        _action: &[f64],
    ) -> Result<LatentVector, EvalError> {
        let episode_id = self
            .active_episode
            .ok_or_else(|| EvalError::InvalidInput("begin_episode was not called".to_owned()))?;
        let predictions = self
            .predictions_by_episode
            .get(&episode_id)
            .ok_or_else(|| EvalError::invalid_episode(episode_id, "missing predictions"))?;
        let predicted = predictions.get(self.cursor).ok_or_else(|| {
            EvalError::invalid_episode(episode_id, "recorded predictions ended before actions")
        })?;
        self.cursor += 1;
        Ok(predicted.clone())
    }
}

/// SO-100 latent evaluator parameterized by a rollout model.
#[derive(Debug)]
pub struct So100Evaluator<M> {
    model: M,
    config: So100EvalConfig,
}

impl<M> So100Evaluator<M>
where
    M: So100RolloutModel,
{
    /// Construct an evaluator.
    #[must_use]
    pub fn new(model: M, config: So100EvalConfig) -> Self {
        Self { model, config }
    }

    /// Run the SO-100 latent-rollout protocol across held-out episodes.
    ///
    /// # Errors
    ///
    /// Returns an error when input episodes are malformed, the rollout adapter
    /// fails, or metrics are degenerate.
    pub fn run(&mut self, episodes: &[So100Episode]) -> Result<So100EvalRun, EvalError> {
        if episodes.is_empty() {
            return Err(EvalError::InvalidInput(
                "SO-100 eval requires at least one episode".to_owned(),
            ));
        }
        if self.config.history_size == 0 {
            return Err(EvalError::InvalidInput(
                "history_size must be greater than zero".to_owned(),
            ));
        }
        if !(0.0..=1.0).contains(&self.config.spearman_floor) {
            return Err(EvalError::InvalidInput(
                "spearman_floor must be in [0, 1]".to_owned(),
            ));
        }

        let mut episode_reports = Vec::with_capacity(episodes.len());
        let mut traces = Vec::new();

        for episode in episodes {
            let (episode_report, episode_traces) = self.run_episode(episode)?;
            episode_reports.push(episode_report);
            traces.extend(episode_traces);
        }

        let latent_mse_mean = mean(
            episode_reports
                .iter()
                .map(|episode| episode.latent_mse)
                .collect::<Vec<_>>()
                .as_slice(),
        )?;
        let spearman_mean = mean(
            episode_reports
                .iter()
                .map(|episode| episode.spearman)
                .collect::<Vec<_>>()
                .as_slice(),
        )?;
        let acceptance = So100Acceptance::from_spearman(spearman_mean, self.config.spearman_floor);
        let report = So100EvalReport {
            episode_count: episode_reports.len(),
            history_size: self.config.history_size,
            latent_mse_mean,
            spearman_mean,
            acceptance,
            episodes: episode_reports,
        };

        Ok(So100EvalRun { report, traces })
    }

    fn run_episode(
        &mut self,
        episode: &So100Episode,
    ) -> Result<(So100EpisodeReport, Vec<So100LatentTraceRow>), EvalError> {
        validate_episode(episode, self.config.history_size)?;
        self.model.begin_episode(episode.episode_id)?;

        let rollout_len = episode.target_latents.len() - self.config.history_size;
        let latent_dim = episode.target_latents[0].len();
        let mut history = vec![episode.target_latents[0].clone(); self.config.history_size];
        let mut predicted = Vec::with_capacity(rollout_len);
        let mut targets = Vec::with_capacity(rollout_len);

        for step in 0..rollout_len {
            let next = self
                .model
                .predict_next(&history, &episode.expert_actions[step])?;
            validate_vector_dims(episode.episode_id, "predicted latent", &next, latent_dim)?;
            let target = episode.target_latents[step + self.config.history_size].clone();
            history.remove(0);
            history.push(next.clone());
            predicted.push(next);
            targets.push(target);
        }

        let mse_per_step = latent_mse_per_step(&predicted, &targets)?;
        let latent_mse = mean(&mse_per_step)?;
        let spearman = trajectory_spearman(&predicted, &targets)?;
        let traces = build_trace_rows(episode.episode_id, &predicted, &targets);
        let report = So100EpisodeReport {
            episode_id: episode.episode_id,
            rollout_steps: rollout_len,
            latent_dim,
            latent_mse,
            spearman,
            latent_mse_per_step: mse_per_step,
        };

        Ok((report, traces))
    }
}

/// Full result of an SO-100 evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct So100EvalRun {
    /// JSON/Markdown report payload.
    pub report: So100EvalReport,
    /// Per-latent-dimension trace rows written to Parquet.
    pub traces: Vec<So100LatentTraceRow>,
}

/// Serializable report for `results.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct So100EvalReport {
    /// Number of episodes evaluated.
    pub episode_count: usize,
    /// History size used for autoregressive rollout.
    pub history_size: usize,
    /// Mean latent MSE across episodes.
    pub latent_mse_mean: f64,
    /// Mean Spearman rank correlation across episodes.
    pub spearman_mean: f64,
    /// Acceptance classification.
    pub acceptance: So100Acceptance,
    /// Per-episode metrics.
    pub episodes: Vec<So100EpisodeReport>,
}

/// Per-episode SO-100 metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct So100EpisodeReport {
    /// Dataset episode identifier.
    pub episode_id: u32,
    /// Number of autoregressive rollout steps.
    pub rollout_steps: usize,
    /// Latent vector dimension.
    pub latent_dim: usize,
    /// Mean per-step latent MSE for this episode.
    pub latent_mse: f64,
    /// Spearman rank correlation for this episode.
    pub spearman: f64,
    /// Per-step latent MSE values.
    pub latent_mse_per_step: Vec<f64>,
}

/// Acceptance policy derived from RFC 0006.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct So100Acceptance {
    /// Outcome bucket.
    pub outcome: So100Outcome,
    /// Spearman floor used for pass.
    pub spearman_floor: f64,
    /// Null-result threshold from RFC 0006.
    pub null_threshold: f64,
}

impl So100Acceptance {
    fn from_spearman(spearman: f64, floor: f64) -> Self {
        let outcome = if spearman >= floor {
            So100Outcome::Pass
        } else if spearman < 0.4 {
            So100Outcome::Null
        } else {
            So100Outcome::Partial
        };
        Self {
            outcome,
            spearman_floor: floor,
            null_threshold: 0.4,
        }
    }
}

/// SO-100 acceptance bucket.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum So100Outcome {
    /// Meets the Spearman floor.
    Pass,
    /// Between null threshold and pass floor.
    Partial,
    /// Below the null threshold.
    Null,
}

/// Per-dimension latent trace row for `latent_traces.parquet`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct So100LatentTraceRow {
    /// Dataset episode identifier.
    pub episode_id: u32,
    /// Rollout step index.
    pub step: u32,
    /// Latent dimension index.
    pub latent_dim: u32,
    /// Predicted latent value.
    pub predicted: f64,
    /// Target latent value.
    pub target: f64,
}

/// Warm-start delta report.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct WarmStartDelta {
    /// Mean latent MSE for the scratch model.
    pub scratch_latent_mse_mean: f64,
    /// Mean latent MSE for the warm-started model.
    pub warm_latent_mse_mean: f64,
    /// `scratch_latent_mse_mean - warm_latent_mse_mean`.
    pub delta: f64,
}

impl WarmStartDelta {
    /// Returns true when the warm-started model has lower mean latent MSE.
    #[must_use]
    pub fn is_positive_transfer(self) -> bool {
        self.delta > 0.0
    }
}

/// Compute per-step latent MSE, matching the RFC's Python reference formula.
///
/// # Errors
///
/// Returns an error when trajectories have different lengths, are empty, have
/// mismatched latent dimensions, or contain non-finite values.
pub fn latent_mse_per_step(
    predicted: &[LatentVector],
    target: &[LatentVector],
) -> Result<Vec<f64>, EvalError> {
    if predicted.len() != target.len() {
        return Err(EvalError::Metric(
            "predicted and target trajectories must have the same length".to_owned(),
        ));
    }
    if predicted.is_empty() {
        return Err(EvalError::Metric(
            "latent MSE requires at least one rollout step".to_owned(),
        ));
    }

    predicted
        .iter()
        .zip(target)
        .map(|(pred, targ)| vector_mse(pred, targ))
        .collect()
}

/// Compute Spearman rank correlation over pairwise latent distances.
///
/// # Errors
///
/// Returns an error when trajectories are too short, have mismatched latent
/// dimensions, contain non-finite values, or produce degenerate ranks.
pub fn trajectory_spearman(
    predicted: &[LatentVector],
    target: &[LatentVector],
) -> Result<f64, EvalError> {
    let pred_dist = pairwise_distances(predicted)?;
    let targ_dist = pairwise_distances(target)?;
    spearman_rank_correlation(&pred_dist, &targ_dist)
}

/// Compute upper-triangle Euclidean pairwise distances for a trajectory.
///
/// # Errors
///
/// Returns an error when the trajectory has fewer than three vectors, has empty
/// or mismatched latent dimensions, or contains non-finite values.
pub fn pairwise_distances(vectors: &[LatentVector]) -> Result<Vec<f64>, EvalError> {
    if vectors.len() < 3 {
        return Err(EvalError::Metric(
            "Spearman requires at least three rollout latents".to_owned(),
        ));
    }
    let dim = vectors[0].len();
    if dim == 0 {
        return Err(EvalError::Metric(
            "pairwise distances require non-empty latents".to_owned(),
        ));
    }
    for vector in vectors {
        validate_metric_vector(vector, dim)?;
    }

    let mut distances = Vec::with_capacity((vectors.len() * (vectors.len() - 1)) / 2);
    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            distances.push(euclidean_distance(&vectors[i], &vectors[j])?);
        }
    }
    Ok(distances)
}

/// Compute Spearman rank correlation with scipy-compatible average ranks.
///
/// # Errors
///
/// Returns an error when the inputs have different lengths, have fewer than two
/// values, contain non-finite values, or are constant after ranking.
pub fn spearman_rank_correlation(left: &[f64], right: &[f64]) -> Result<f64, EvalError> {
    if left.len() != right.len() {
        return Err(EvalError::Metric(
            "Spearman inputs must have the same length".to_owned(),
        ));
    }
    if left.len() < 2 {
        return Err(EvalError::Metric(
            "Spearman requires at least two distance values".to_owned(),
        ));
    }
    let left_ranks = average_ranks(left)?;
    let right_ranks = average_ranks(right)?;
    pearson_correlation(&left_ranks, &right_ranks)
}

/// Compute scipy `rankdata(method = "average")` ranks.
///
/// # Errors
///
/// Returns an error when the input is empty or contains non-finite values.
pub fn average_ranks(values: &[f64]) -> Result<Vec<f64>, EvalError> {
    if values.is_empty() {
        return Err(EvalError::Metric(
            "average ranks require non-empty values".to_owned(),
        ));
    }
    for value in values {
        if !value.is_finite() {
            return Err(EvalError::Metric(
                "average ranks require finite values".to_owned(),
            ));
        }
    }

    let mut indexed = values
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f64)>>();
    indexed.sort_by(|left, right| left.1.total_cmp(&right.1));

    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < indexed.len() {
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].1.total_cmp(&indexed[start].1) == Ordering::Equal
        {
            end += 1;
        }
        let average_rank = usize_to_f64(start + 1 + end) / 2.0;
        for item in &indexed[start..end] {
            ranks[item.0] = average_rank;
        }
        start = end;
    }
    Ok(ranks)
}

/// Compute warm-start delta from two SO-100 reports.
#[must_use]
pub fn warm_start_delta(scratch: &So100EvalReport, warm: &So100EvalReport) -> WarmStartDelta {
    let scratch_latent_mse_mean = scratch.latent_mse_mean;
    let warm_latent_mse_mean = warm.latent_mse_mean;
    WarmStartDelta {
        scratch_latent_mse_mean,
        warm_latent_mse_mean,
        delta: scratch_latent_mse_mean - warm_latent_mse_mean,
    }
}

/// Render a Markdown report for a completed SO-100 evaluation.
#[must_use]
pub fn render_report_markdown(report: &So100EvalReport) -> String {
    let mut lines = vec![
        "# SO-100 latent rollout evaluation".to_owned(),
        String::new(),
        format!("- Episodes: {}", report.episode_count),
        format!("- History size: {}", report.history_size),
        format!("- Latent MSE mean: {:.6}", report.latent_mse_mean),
        format!("- Spearman mean: {:.6}", report.spearman_mean),
        format!(
            "- Outcome: {:?} (floor {:.3}, null below {:.3})",
            report.acceptance.outcome,
            report.acceptance.spearman_floor,
            report.acceptance.null_threshold
        ),
        String::new(),
        "| episode_id | rollout_steps | latent_dim | latent_mse | spearman |".to_owned(),
        "|---:|---:|---:|---:|---:|".to_owned(),
    ];

    for episode in &report.episodes {
        lines.push(format!(
            "| {} | {} | {} | {:.6} | {:.6} |",
            episode.episode_id,
            episode.rollout_steps,
            episode.latent_dim,
            episode.latent_mse,
            episode.spearman
        ));
    }
    lines.push(String::new());
    lines.join("\n")
}

/// Write `results.json`, `report.md`, and `latent_traces.parquet`.
///
/// # Errors
///
/// Returns an error when output directories or files cannot be written, JSON
/// rendering fails, or Parquet/Arrow encoding fails.
pub fn write_so100_outputs(output_dir: &Path, run: &So100EvalRun) -> Result<(), EvalError> {
    fs::create_dir_all(output_dir).map_err(|source| EvalError::io(output_dir, source))?;

    let results_path = output_dir.join("results.json");
    let json = serde_json::to_string_pretty(&run.report).map_err(EvalError::json_encode)?;
    fs::write(&results_path, json).map_err(|source| EvalError::io(&results_path, source))?;

    let report_path = output_dir.join("report.md");
    fs::write(&report_path, render_report_markdown(&run.report))
        .map_err(|source| EvalError::io(&report_path, source))?;

    let traces_path = output_dir.join("latent_traces.parquet");
    write_trace_parquet(&traces_path, &run.traces)
}

fn validate_episode(episode: &So100Episode, history_size: usize) -> Result<(), EvalError> {
    if episode.target_latents.len() <= history_size {
        return Err(EvalError::invalid_episode(
            episode.episode_id,
            "target_latents length must be greater than history_size",
        ));
    }
    let latent_dim = episode.target_latents[0].len();
    if latent_dim == 0 {
        return Err(EvalError::invalid_episode(
            episode.episode_id,
            "target_latents must have non-empty vectors",
        ));
    }
    for latent in &episode.target_latents {
        validate_vector_dims(episode.episode_id, "target latent", latent, latent_dim)?;
    }

    let rollout_len = episode.target_latents.len() - history_size;
    if episode.expert_actions.len() < rollout_len {
        return Err(EvalError::invalid_episode(
            episode.episode_id,
            format!(
                "expert_actions length {} is shorter than rollout length {rollout_len}",
                episode.expert_actions.len()
            ),
        ));
    }
    for action in &episode.expert_actions[..rollout_len] {
        if action.is_empty() {
            return Err(EvalError::invalid_episode(
                episode.episode_id,
                "expert actions must be non-empty",
            ));
        }
        if action.iter().any(|value| !value.is_finite()) {
            return Err(EvalError::invalid_episode(
                episode.episode_id,
                "expert actions must be finite",
            ));
        }
    }
    Ok(())
}

fn validate_vector_dims(
    episode_id: u32,
    label: &str,
    vector: &[f64],
    expected_dim: usize,
) -> Result<(), EvalError> {
    if vector.len() != expected_dim {
        return Err(EvalError::invalid_episode(
            episode_id,
            format!("{label} has dim {}, expected {expected_dim}", vector.len()),
        ));
    }
    validate_metric_vector(vector, expected_dim)
        .map_err(|error| EvalError::invalid_episode(episode_id, error.to_string()))
}

fn validate_metric_vector(vector: &[f64], expected_dim: usize) -> Result<(), EvalError> {
    if vector.len() != expected_dim {
        return Err(EvalError::Metric(format!(
            "latent dim {}, expected {expected_dim}",
            vector.len()
        )));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(EvalError::Metric(
            "latent vectors must contain finite values".to_owned(),
        ));
    }
    Ok(())
}

fn vector_mse(left: &[f64], right: &[f64]) -> Result<f64, EvalError> {
    if left.len() != right.len() {
        return Err(EvalError::Metric(
            "latent MSE vectors must have equal dimensions".to_owned(),
        ));
    }
    if left.is_empty() {
        return Err(EvalError::Metric(
            "latent MSE vectors must be non-empty".to_owned(),
        ));
    }
    if left.iter().chain(right).any(|value| !value.is_finite()) {
        return Err(EvalError::Metric(
            "latent MSE vectors must contain finite values".to_owned(),
        ));
    }

    let sum_sq = left
        .iter()
        .zip(right)
        .map(|(left_value, right_value)| {
            let diff = left_value - right_value;
            diff * diff
        })
        .sum::<f64>();
    Ok(sum_sq / usize_to_f64(left.len()))
}

fn euclidean_distance(left: &[f64], right: &[f64]) -> Result<f64, EvalError> {
    Ok(vector_mse(left, right)?
        .mul_add(usize_to_f64(left.len()), 0.0)
        .sqrt())
}

fn pearson_correlation(left: &[f64], right: &[f64]) -> Result<f64, EvalError> {
    if left.len() != right.len() {
        return Err(EvalError::Metric(
            "Pearson inputs must have the same length".to_owned(),
        ));
    }
    let left_mean = mean(left)?;
    let right_mean = mean(right)?;
    let mut numerator = 0.0;
    let mut left_var = 0.0;
    let mut right_var = 0.0;
    for (left_value, right_value) in left.iter().zip(right) {
        let left_delta = left_value - left_mean;
        let right_delta = right_value - right_mean;
        numerator += left_delta * right_delta;
        left_var += left_delta * left_delta;
        right_var += right_delta * right_delta;
    }
    let denom = (left_var * right_var).sqrt();
    if denom == 0.0 {
        return Err(EvalError::Metric(
            "rank correlation is undefined for a constant trajectory".to_owned(),
        ));
    }
    Ok(numerator / denom)
}

fn mean(values: &[f64]) -> Result<f64, EvalError> {
    if values.is_empty() {
        return Err(EvalError::Metric(
            "mean requires at least one value".to_owned(),
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(EvalError::Metric("mean requires finite values".to_owned()));
    }
    Ok(values.iter().sum::<f64>() / usize_to_f64(values.len()))
}

fn build_trace_rows(
    episode_id: u32,
    predicted: &[LatentVector],
    target: &[LatentVector],
) -> Vec<So100LatentTraceRow> {
    let total_rows = predicted
        .first()
        .map_or(0, |first| predicted.len() * first.len());
    let mut rows = Vec::with_capacity(total_rows);
    for (step, (predicted_latent, target_latent)) in predicted.iter().zip(target).enumerate() {
        for (latent_dim, (predicted_value, target_value)) in
            predicted_latent.iter().zip(target_latent).enumerate()
        {
            rows.push(So100LatentTraceRow {
                episode_id,
                step: usize_to_u32(step),
                latent_dim: usize_to_u32(latent_dim),
                predicted: *predicted_value,
                target: *target_value,
            });
        }
    }
    rows
}

fn write_trace_parquet(path: &Path, traces: &[So100LatentTraceRow]) -> Result<(), EvalError> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("episode_id", DataType::UInt32, false),
        Field::new("step", DataType::UInt32, false),
        Field::new("latent_dim", DataType::UInt32, false),
        Field::new("predicted", DataType::Float64, false),
        Field::new("target", DataType::Float64, false),
    ]));
    let episode_ids = traces.iter().map(|row| row.episode_id).collect::<Vec<_>>();
    let steps = traces.iter().map(|row| row.step).collect::<Vec<_>>();
    let latent_dims = traces.iter().map(|row| row.latent_dim).collect::<Vec<_>>();
    let predicted = traces.iter().map(|row| row.predicted).collect::<Vec<_>>();
    let target = traces.iter().map(|row| row.target).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(UInt32Array::from(episode_ids)) as ArrayRef,
            Arc::new(UInt32Array::from(steps)) as ArrayRef,
            Arc::new(UInt32Array::from(latent_dims)) as ArrayRef,
            Arc::new(Float64Array::from(predicted)) as ArrayRef,
            Arc::new(Float64Array::from(target)) as ArrayRef,
        ],
    )
    .map_err(|source| EvalError::arrow(path, source))?;

    let file = File::create(path).map_err(|source| EvalError::io(path, source))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|source| EvalError::parquet(path, source))?;
    writer
        .write(&batch)
        .map_err(|source| EvalError::parquet(path, source))?;
    writer
        .close()
        .map_err(|source| EvalError::parquet(path, source))?;
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

#[allow(clippy::cast_possible_truncation)]
fn usize_to_u32(value: usize) -> u32 {
    value as u32
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    const EPS: f64 = 1.0e-9;

    #[test]
    fn so100_latent_mse_matches_python() -> Result<(), Box<dyn std::error::Error>> {
        let predicted = vec![vec![2.0, 0.0], vec![3.0, 0.0], vec![4.0, 1.0]];
        let target = vec![vec![2.0, 0.0], vec![3.0, 0.0], vec![4.0, 3.0]];

        let mse = latent_mse_per_step(&predicted, &target)?;

        assert!((mse[0] - 0.0).abs() < EPS);
        assert!((mse[1] - 0.0).abs() < EPS);
        assert!((mse[2] - 2.0).abs() < EPS);
        Ok(())
    }

    #[test]
    fn so100_spearman_matches_scipy_average_ties() -> Result<(), Box<dyn std::error::Error>> {
        let left = vec![1.0, 2.0, 2.0, 4.0];
        let right = vec![4.0, 1.0, 1.0, 2.0];

        let rho = spearman_rank_correlation(&left, &right)?;

        assert!((rho + (1.0 / 3.0)).abs() < EPS);
        Ok(())
    }

    #[test]
    fn so100_eval_pipeline_end_to_end_on_synthetic() -> Result<(), Box<dyn std::error::Error>> {
        let episode = So100Episode {
            episode_id: 5,
            target_latents: vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![2.0, 0.0],
                vec![3.0, 0.0],
                vec![4.0, 0.0],
            ],
            expert_actions: vec![vec![0.0], vec![0.0], vec![0.0]],
        };
        let mut predictions = BTreeMap::new();
        predictions.insert(5, vec![vec![2.0, 0.0], vec![3.0, 0.0], vec![4.0, 0.0]]);
        let model = RecordedRolloutModel::new(predictions);
        let mut evaluator = So100Evaluator::new(
            model,
            So100EvalConfig {
                history_size: 2,
                spearman_floor: 0.6,
            },
        );

        let run = evaluator.run(&[episode])?;

        assert!((run.report.latent_mse_mean - 0.0).abs() < EPS);
        assert!((run.report.spearman_mean - 1.0).abs() < EPS);
        assert_eq!(run.report.acceptance.outcome, So100Outcome::Pass);
        assert_eq!(run.traces.len(), 6);

        let dir = tempfile::tempdir()?;
        write_so100_outputs(dir.path(), &run)?;
        assert!(dir.path().join("results.json").exists());
        assert!(dir.path().join("report.md").exists());
        let parquet = fs::read(dir.path().join("latent_traces.parquet"))?;
        assert!(parquet.starts_with(b"PAR1"));
        Ok(())
    }

    #[test]
    fn warm_start_delta_correct_sign() {
        let scratch = report_with_mse(2.5);
        let warm = report_with_mse(1.0);

        let delta = warm_start_delta(&scratch, &warm);

        assert!((delta.delta - 1.5).abs() < EPS);
        assert!(delta.is_positive_transfer());
    }

    fn report_with_mse(latent_mse_mean: f64) -> So100EvalReport {
        So100EvalReport {
            episode_count: 1,
            history_size: 2,
            latent_mse_mean,
            spearman_mean: 1.0,
            acceptance: So100Acceptance::from_spearman(1.0, 0.6),
            episodes: Vec::new(),
        }
    }
}
