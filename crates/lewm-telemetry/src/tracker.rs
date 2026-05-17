//! Trackio-compatible local JSONL metric writer.

use std::{
    fmt,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::Mutex,
    time::{Duration, Instant},
};

use serde_json::json;

use crate::{MetricName, MetricSink, TelemetryContext, TelemetryError};

const DEFAULT_FLUSH_RECORDS: usize = 1_000;
const DEFAULT_FLUSH_INTERVAL: Duration = Duration::from_secs(5);

/// Local Trackio-format JSONL writer.
pub struct TrackioWriter {
    run_dir: PathBuf,
    metrics_path: PathBuf,
    flush_records: usize,
    flush_interval: Duration,
    state: Mutex<TrackioState>,
}

impl fmt::Debug for TrackioWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrackioWriter")
            .field("run_dir", &self.run_dir)
            .field("metrics_path", &self.metrics_path)
            .field("flush_records", &self.flush_records)
            .field("flush_interval", &self.flush_interval)
            .finish_non_exhaustive()
    }
}

struct TrackioState {
    sink: BufWriter<File>,
    records_since_flush: usize,
    last_flush: Instant,
}

impl TrackioWriter {
    /// Create a writer under `<root>/runs/<run_id>/metrics.jsonl`.
    ///
    /// # Errors
    ///
    /// Returns an error when the run directory or append-only metrics file cannot be opened.
    pub fn new(root: impl AsRef<Path>, run_id: &str) -> Result<Self, TelemetryError> {
        Self::with_flush_policy(root, run_id, DEFAULT_FLUSH_RECORDS, DEFAULT_FLUSH_INTERVAL)
    }

    /// Create a writer with an explicit buffered flush policy.
    ///
    /// # Errors
    ///
    /// Returns an error when the run directory or append-only metrics file cannot be opened.
    pub fn with_flush_policy(
        root: impl AsRef<Path>,
        run_id: &str,
        flush_records: usize,
        flush_interval: Duration,
    ) -> Result<Self, TelemetryError> {
        validate_flush_policy(flush_records, flush_interval)?;
        if run_id.trim().is_empty() {
            return Err(TelemetryError::InvalidConfig(
                "Trackio run_id must be non-empty".to_string(),
            ));
        }

        let run_dir = root.as_ref().join("runs").join(run_id);
        fs::create_dir_all(&run_dir).map_err(TelemetryError::sink)?;
        let metrics_path = run_dir.join("metrics.jsonl");
        let sink = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&metrics_path)
            .map_err(TelemetryError::sink)?;

        Ok(Self {
            run_dir,
            metrics_path,
            flush_records,
            flush_interval,
            state: Mutex::new(TrackioState {
                sink: BufWriter::new(sink),
                records_since_flush: 0,
                last_flush: Instant::now(), // determinism-lint: allow Instant::now telemetry flush cadence
            }),
        })
    }

    /// Directory for this Trackio run.
    #[must_use]
    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    /// Path to `metrics.jsonl`.
    #[must_use]
    pub fn metrics_path(&self) -> &Path {
        &self.metrics_path
    }

    fn write_json_line(&self, value: &serde_json::Value) -> Result<(), TelemetryError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        serde_json::to_writer(&mut state.sink, &value).map_err(TelemetryError::sink)?;
        state.sink.write_all(b"\n").map_err(TelemetryError::sink)?;
        self.maybe_flush_locked(&mut state)
    }

    fn maybe_flush_locked(&self, state: &mut TrackioState) -> Result<(), TelemetryError> {
        state.records_since_flush += 1;
        if state.records_since_flush >= self.flush_records
            || state.last_flush.elapsed() >= self.flush_interval
        {
            flush_locked(state)?;
        }
        Ok(())
    }
}

impl MetricSink for TrackioWriter {
    fn emit_scalar(
        &self,
        context: &TelemetryContext,
        name: MetricName,
        step: u64,
        value: f32,
    ) -> Result<(), TelemetryError> {
        self.write_json_line(&json!({
            "git_short_sha": context.git_short_sha,
            "name": name.as_str(),
            "phase": context.phase,
            "run_id": context.run_id,
            "step": step,
            "value": value,
        }))
    }

    fn emit_histogram(
        &self,
        context: &TelemetryContext,
        name: MetricName,
        step: u64,
        values: &[f32],
    ) -> Result<(), TelemetryError> {
        self.write_json_line(&json!({
            "git_short_sha": context.git_short_sha,
            "name": name.as_str(),
            "phase": context.phase,
            "run_id": context.run_id,
            "step": step,
            "values": values,
        }))
    }

    fn flush(&self) -> Result<(), TelemetryError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        flush_locked(&mut state)
    }
}

fn validate_flush_policy(
    flush_records: usize,
    flush_interval: Duration,
) -> Result<(), TelemetryError> {
    if flush_records == 0 {
        return Err(TelemetryError::InvalidConfig(
            "flush_records must be greater than zero".to_string(),
        ));
    }
    if flush_interval.is_zero() {
        return Err(TelemetryError::InvalidConfig(
            "flush_interval must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

fn flush_locked(state: &mut TrackioState) -> Result<(), TelemetryError> {
    state.sink.flush().map_err(TelemetryError::sink)?;
    state.records_since_flush = 0;
    state.last_flush = Instant::now(); // determinism-lint: allow Instant::now telemetry flush cadence
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use serde_json::Value;

    use super::*;

    #[test]
    fn trackio_writer_appends_jsonl() -> Result<(), Box<dyn std::error::Error>> {
        let root = temp_root("trackio_writer_appends_jsonl")?;
        let context = TelemetryContext {
            run_id: "run-001".to_string(),
            phase: "phase-2".to_string(),
            git_short_sha: "abc1234".to_string(),
        };

        let writer =
            TrackioWriter::with_flush_policy(&root, &context.run_id, 2, Duration::from_mins(1))?;
        writer.emit_scalar(&context, MetricName::LossTotal, 3, 1.5)?;
        writer.emit_scalar(&context, MetricName::OptimLr, 3, 0.001)?;
        writer.flush()?;

        let writer =
            TrackioWriter::with_flush_policy(&root, &context.run_id, 2, Duration::from_mins(1))?;
        writer.emit_scalar(&context, MetricName::DataQueueDepth, 4, 8.0)?;
        writer.flush()?;

        let lines = fs::read_to_string(writer.metrics_path())?
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert_eq!(lines.len(), 3);
        let first = serde_json::from_str::<Value>(&lines[0])?;
        assert_eq!(first["name"], "loss/total");
        assert_eq!(first["step"], 3);
        assert_eq!(first["value"], 1.5);
        assert_eq!(first["run_id"], "run-001");
        assert!(
            writer
                .metrics_path()
                .ends_with("runs/run-001/metrics.jsonl")
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    fn temp_root(name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let root = std::env::temp_dir().join(format!(
            "lewm-telemetry-{name}-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        let _ignored = fs::remove_dir_all(&root);
        fs::create_dir_all(&root)?;
        Ok(root)
    }
}
