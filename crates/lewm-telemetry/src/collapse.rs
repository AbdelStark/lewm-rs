//! Collapse detector state machine and artifact writer for RFC 0009.

use std::{fs, path::PathBuf};

use chrono::{SecondsFormat, Utc};
use lewm_core::{CollapseProbeResult, CollapseThresholds, run_collapse_probe_with_thresholds};
use serde_json::json;

use crate::{TelemetryContext, TelemetryError};

/// Fixed held-out collapse probe fixture path, relative to the repository root.
pub const COLLAPSE_PROBE_FIXTURE_PATH: &str = "tests/fixtures/collapse_probe.npz";

/// RFC 0009 held-out probe batch size.
pub const COLLAPSE_PROBE_BATCH_FRAMES: usize = 32;

/// Number of consecutive threshold trips required before writing an artifact.
pub const COLLAPSE_TRIPS_REQUIRED: u32 = 3;

/// Collapse detector configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct CollapseDetectorConfig {
    /// Run context attached to emitted artifacts and log events.
    pub context: TelemetryContext,
    /// Directory where `collapse_suspected_{step}.json` artifacts are written.
    pub artifact_dir: PathBuf,
    /// Thresholds used to classify one probe sample.
    pub thresholds: CollapseThresholds,
    /// Consecutive trip count required before the detector reports suspicion.
    pub trips_required: u32,
}

impl CollapseDetectorConfig {
    /// Build a detector config with RFC 0009 thresholds and three-trip policy.
    #[must_use]
    pub fn new(context: TelemetryContext, artifact_dir: impl Into<PathBuf>) -> Self {
        Self {
            context,
            artifact_dir: artifact_dir.into(),
            thresholds: CollapseThresholds::default(),
            trips_required: COLLAPSE_TRIPS_REQUIRED,
        }
    }

    /// Override collapse thresholds.
    #[must_use]
    pub fn with_thresholds(mut self, thresholds: CollapseThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Override the consecutive trip requirement.
    #[must_use]
    pub fn with_trips_required(mut self, trips_required: u32) -> Self {
        self.trips_required = trips_required;
        self
    }
}

/// Stateful collapse detector.
#[derive(Debug, Clone)]
pub struct CollapseDetector {
    config: CollapseDetectorConfig,
    trips_in_a_row: u32,
}

impl CollapseDetector {
    /// Create a collapse detector.
    ///
    /// # Errors
    ///
    /// Returns an error when run context fields are empty, the artifact
    /// directory is empty, or `trips_required` is zero.
    pub fn new(config: CollapseDetectorConfig) -> Result<Self, TelemetryError> {
        validate_config(&config)?;
        Ok(Self {
            config,
            trips_in_a_row: 0,
        })
    }

    /// Return the current consecutive trip count.
    #[must_use]
    pub fn trips_in_a_row(&self) -> u32 {
        self.trips_in_a_row
    }

    /// Evaluate one held-out CLS batch and update detector state.
    ///
    /// The CLS buffer must be row-major `(32, dim)` embeddings from the fixed
    /// held-out fixture after the trainer has run its encoder with no-grad
    /// semantics.
    ///
    /// # Errors
    ///
    /// Returns an error when the batch size is not 32, the core probe rejects
    /// the CLS tensor or thresholds, or an artifact cannot be written.
    pub fn observe_cls(
        &mut self,
        step: u64,
        epoch: u64,
        cls: &[f32],
        batch: usize,
        dim: usize,
    ) -> Result<CollapseDetectorDecision, TelemetryError> {
        if batch != COLLAPSE_PROBE_BATCH_FRAMES {
            return Err(TelemetryError::InvalidConfig(format!(
                "collapse probe batch must contain {COLLAPSE_PROBE_BATCH_FRAMES} frames, got {batch}"
            )));
        }

        let result = run_collapse_probe_with_thresholds(cls, batch, dim, self.config.thresholds)
            .map_err(|error| TelemetryError::Collapse(error.to_string()))?;

        if result.is_collapsed() {
            self.trips_in_a_row = self.trips_in_a_row.saturating_add(1);
        } else {
            self.trips_in_a_row = 0;
        }

        let artifact_path = if self.trips_in_a_row == self.config.trips_required {
            let path = self.write_artifact(step, epoch, &result)?;
            tracing::error!(
                severity = "CRITICAL",
                run_id = %self.config.context.run_id,
                phase = %self.config.context.phase,
                git_short_sha = %self.config.context.git_short_sha,
                step = step,
                epoch = epoch,
                trips_in_a_row = self.trips_in_a_row,
                mean_abs_cls = result.probe.mean_abs_cls,
                cls_variance_per_dim_mean = result.probe.cls_variance_per_dim_mean,
                mean_pairwise_cosine = result.probe.mean_pairwise_cosine,
                artifact_path = %path.display(),
                "collapse_suspected"
            );
            Some(path)
        } else {
            None
        };

        Ok(CollapseDetectorDecision {
            result,
            trips_in_a_row: self.trips_in_a_row,
            artifact_path,
        })
    }

    fn write_artifact(
        &self,
        step: u64,
        epoch: u64,
        result: &CollapseProbeResult,
    ) -> Result<PathBuf, TelemetryError> {
        fs::create_dir_all(&self.config.artifact_dir).map_err(|error| {
            TelemetryError::Collapse(format!(
                "failed to create collapse artifact directory {}: {error}",
                self.config.artifact_dir.display()
            ))
        })?;

        let path = self
            .config
            .artifact_dir
            .join(format!("collapse_suspected_{step:07}.json"));
        let artifact = json!({
            "schema_version": "1.0",
            "run_id": self.config.context.run_id,
            "step": step,
            "epoch": epoch,
            "probes": result.probe,
            "thresholds": result.thresholds,
            "trips_in_a_row": self.trips_in_a_row,
            "wall_time": Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        });
        let bytes = serde_json::to_vec_pretty(&artifact)
            .map_err(|error| TelemetryError::Collapse(error.to_string()))?;
        fs::write(&path, bytes).map_err(|error| {
            TelemetryError::Collapse(format!(
                "failed to write collapse artifact {}: {error}",
                path.display()
            ))
        })?;

        Ok(path)
    }
}

/// Decision returned after one collapse probe observation.
#[derive(Debug, Clone, PartialEq)]
pub struct CollapseDetectorDecision {
    /// Core probe metrics, thresholds, and per-threshold trip flags.
    pub result: CollapseProbeResult,
    /// Consecutive threshold trips after this observation.
    pub trips_in_a_row: u32,
    /// Artifact path written on the exact trip that reaches the configured threshold.
    pub artifact_path: Option<PathBuf>,
}

impl CollapseDetectorDecision {
    /// Return `true` when this observation tripped any collapse threshold.
    #[must_use]
    pub fn is_collapsed(&self) -> bool {
        self.result.is_collapsed()
    }

    /// Return `true` when this observation wrote a suspicion artifact.
    #[must_use]
    pub fn wrote_artifact(&self) -> bool {
        self.artifact_path.is_some()
    }
}

fn validate_config(config: &CollapseDetectorConfig) -> Result<(), TelemetryError> {
    validate_non_empty("run_id", &config.context.run_id)?;
    validate_non_empty("phase", &config.context.phase)?;
    validate_non_empty("git_short_sha", &config.context.git_short_sha)?;
    if config.artifact_dir.as_os_str().is_empty() {
        return Err(TelemetryError::InvalidConfig(
            "collapse artifact directory must be non-empty".to_owned(),
        ));
    }
    if config.trips_required == 0 {
        return Err(TelemetryError::InvalidConfig(
            "collapse trips_required must be greater than zero".to_owned(),
        ));
    }
    Ok(())
}

fn validate_non_empty(field: &str, value: &str) -> Result<(), TelemetryError> {
    if value.trim().is_empty() {
        return Err(TelemetryError::InvalidConfig(format!(
            "{field} must be non-empty"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        env, fs,
        path::Path,
        process,
        time::{SystemTime, UNIX_EPOCH},
    };

    use serde_json::Value;

    use super::*;

    #[test]
    fn collapse_probe_fixture_exists() -> Result<(), Box<dyn std::error::Error>> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let fixture_path = repo_root.join(COLLAPSE_PROBE_FIXTURE_PATH);
        let bytes = fs::read(fixture_path)?;

        assert!(
            bytes.starts_with(b"PK"),
            "collapse probe fixture must be an npz archive"
        );

        let meta: Value = serde_json::from_slice(&fs::read(
            repo_root.join("tests/fixtures/collapse_probe.meta.json"),
        )?)?;
        assert_eq!(meta["fixture_path"], COLLAPSE_PROBE_FIXTURE_PATH);
        assert_eq!(meta["source_split"], "eval");
        assert_eq!(
            meta["pixels"]["shape"],
            serde_json::json!([32, 3, 224, 224])
        );
        Ok(())
    }

    #[test]
    fn collapse_detector_three_in_row_trips() -> Result<(), Box<dyn std::error::Error>> {
        let artifact_dir = temp_artifact_dir("three_in_row");
        let mut detector =
            CollapseDetector::new(CollapseDetectorConfig::new(test_context(), &artifact_dir))?;
        let cls = collapsed_cls(4);

        let first = detector.observe_cls(100, 1, &cls, COLLAPSE_PROBE_BATCH_FRAMES, 4)?;
        assert!(first.is_collapsed());
        assert_eq!(first.trips_in_a_row, 1);
        assert!(!first.wrote_artifact());

        let second = detector.observe_cls(200, 1, &cls, COLLAPSE_PROBE_BATCH_FRAMES, 4)?;
        assert!(second.is_collapsed());
        assert_eq!(second.trips_in_a_row, 2);
        assert!(!second.wrote_artifact());

        let third = detector.observe_cls(300, 1, &cls, COLLAPSE_PROBE_BATCH_FRAMES, 4)?;
        assert!(third.is_collapsed());
        assert_eq!(third.trips_in_a_row, COLLAPSE_TRIPS_REQUIRED);
        let artifact_path = third
            .artifact_path
            .ok_or("third trip should write artifact")?;
        assert_eq!(
            artifact_path.file_name().and_then(|name| name.to_str()),
            Some("collapse_suspected_0000300.json")
        );

        let artifact: Value = serde_json::from_slice(&fs::read(&artifact_path)?)?;
        assert_eq!(artifact["schema_version"], "1.0");
        assert_eq!(artifact["run_id"], "run-collapse-001");
        assert_eq!(artifact["step"], 300);
        assert_eq!(artifact["epoch"], 1);
        assert_eq!(artifact["trips_in_a_row"], 3);
        assert_eq!(artifact["probes"]["cls_variance_per_dim_mean"], 0.0);
        assert_eq!(artifact["probes"]["mean_pairwise_cosine"], 1.0);
        let variance_floor = artifact["thresholds"]["cls_variance_per_dim_floor"]
            .as_f64()
            .ok_or("variance floor must be numeric")?;
        assert!((variance_floor - 0.05).abs() < 1e-6);

        let fourth = detector.observe_cls(400, 1, &cls, COLLAPSE_PROBE_BATCH_FRAMES, 4)?;
        assert_eq!(fourth.trips_in_a_row, 4);
        assert!(!fourth.wrote_artifact());

        let _ = fs::remove_dir_all(artifact_dir);
        Ok(())
    }

    #[test]
    fn collapse_detector_no_false_positive() -> Result<(), Box<dyn std::error::Error>> {
        let artifact_dir = temp_artifact_dir("healthy");
        let mut detector =
            CollapseDetector::new(CollapseDetectorConfig::new(test_context(), &artifact_dir))?;
        let cls = healthy_cls(4);

        for step in [100, 200, 300, 400] {
            let decision = detector.observe_cls(step, 1, &cls, COLLAPSE_PROBE_BATCH_FRAMES, 4)?;
            assert!(!decision.is_collapsed());
            assert_eq!(decision.trips_in_a_row, 0);
            assert!(!decision.wrote_artifact());
        }

        assert!(!artifact_dir.exists());
        let _ = fs::remove_dir_all(artifact_dir);
        Ok(())
    }

    fn test_context() -> TelemetryContext {
        TelemetryContext {
            run_id: "run-collapse-001".to_owned(),
            phase: "phase-2".to_owned(),
            git_short_sha: "abc1234".to_owned(),
        }
    }

    fn collapsed_cls(dim: usize) -> Vec<f32> {
        vec![1.0; COLLAPSE_PROBE_BATCH_FRAMES * dim]
    }

    fn healthy_cls(dim: usize) -> Vec<f32> {
        let mut cls = Vec::with_capacity(COLLAPSE_PROBE_BATCH_FRAMES * dim);
        for row in 0..COLLAPSE_PROBE_BATCH_FRAMES {
            for feature in 0..dim {
                let bit = (row >> feature) & 1;
                cls.push(if bit == 0 { -1.0 } else { 1.0 });
            }
        }
        cls
    }

    fn temp_artifact_dir(test_name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        env::temp_dir().join(format!(
            "lewm-rs-collapse-{test_name}-{}-{nonce}",
            process::id()
        ))
    }
}
