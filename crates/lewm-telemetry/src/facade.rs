//! Public telemetry facade and sink boundary.

use std::{
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{MetricName, SpanName, TelemetryError};

/// Telemetry initialization settings.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TelemetryConfig {
    /// Stable run identifier attached to every metric, span, and log line.
    pub run_id: String,
    /// Run phase attached to every metric, span, and log line.
    pub phase: String,
    /// Short git SHA attached to every metric, span, and log line.
    pub git_short_sha: String,
}

impl TelemetryConfig {
    /// Build a telemetry config from required run attributes.
    #[must_use]
    pub fn new(
        run_id: impl Into<String>,
        phase: impl Into<String>,
        git_short_sha: impl Into<String>,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            phase: phase.into(),
            git_short_sha: git_short_sha.into(),
        }
    }

    fn validate(&self) -> Result<(), TelemetryError> {
        validate_non_empty("run_id", &self.run_id)?;
        validate_non_empty("phase", &self.phase)?;
        validate_non_empty("git_short_sha", &self.git_short_sha)?;
        Ok(())
    }
}

/// Context copied onto every telemetry record.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TelemetryContext {
    /// Stable run identifier.
    pub run_id: String,
    /// Run phase.
    pub phase: String,
    /// Short git SHA.
    pub git_short_sha: String,
}

impl From<TelemetryConfig> for TelemetryContext {
    fn from(value: TelemetryConfig) -> Self {
        Self {
            run_id: value.run_id,
            phase: value.phase,
            git_short_sha: value.git_short_sha,
        }
    }
}

/// Metric exporter boundary used by the facade.
pub trait MetricSink: Send + Sync {
    /// Accept one scalar metric record.
    ///
    /// # Errors
    ///
    /// Returns a sink-specific error when the record cannot be persisted.
    fn emit_scalar(
        &self,
        context: &TelemetryContext,
        name: MetricName,
        step: u64,
        value: f32,
    ) -> Result<(), TelemetryError>;

    /// Accept one histogram metric record.
    ///
    /// # Errors
    ///
    /// Returns a sink-specific error when the record cannot be persisted.
    fn emit_histogram(
        &self,
        context: &TelemetryContext,
        name: MetricName,
        step: u64,
        values: &[f32],
    ) -> Result<(), TelemetryError>;

    /// Flush all buffered records.
    ///
    /// # Errors
    ///
    /// Returns a sink-specific error when flushing fails.
    fn flush(&self) -> Result<(), TelemetryError>;
}

#[derive(Debug, Default)]
struct NoopMetricSink;

impl MetricSink for NoopMetricSink {
    fn emit_scalar(
        &self,
        _context: &TelemetryContext,
        _name: MetricName,
        _step: u64,
        _value: f32,
    ) -> Result<(), TelemetryError> {
        Ok(())
    }

    fn emit_histogram(
        &self,
        _context: &TelemetryContext,
        _name: MetricName,
        _step: u64,
        _values: &[f32],
    ) -> Result<(), TelemetryError> {
        Ok(())
    }

    fn flush(&self) -> Result<(), TelemetryError> {
        Ok(())
    }
}

/// Single public entry point for metrics, spans, and logs.
pub struct Telemetry {
    context: TelemetryContext,
    metric_sink: Arc<dyn MetricSink>,
}

impl fmt::Debug for Telemetry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Telemetry")
            .field("context", &self.context)
            .finish_non_exhaustive()
    }
}

impl Telemetry {
    /// Initialize telemetry with the default no-op metric sink.
    ///
    /// # Errors
    ///
    /// Returns an error when required run attributes are empty.
    pub fn init(config: TelemetryConfig) -> Result<Self, TelemetryError> {
        Self::with_metric_sink(config, Arc::new(NoopMetricSink))
    }

    /// Initialize telemetry with an explicit metric sink.
    ///
    /// # Errors
    ///
    /// Returns an error when required run attributes are empty.
    pub fn with_metric_sink(
        config: TelemetryConfig,
        metric_sink: Arc<dyn MetricSink>,
    ) -> Result<Self, TelemetryError> {
        config.validate()?;
        Ok(Self {
            context: config.into(),
            metric_sink,
        })
    }

    /// Return the run context attached to every emitted record.
    #[must_use]
    pub fn context(&self) -> &TelemetryContext {
        &self.context
    }

    /// Emit one scalar metric.
    ///
    /// # Errors
    ///
    /// Returns an error when the sink rejects the record.
    pub fn emit_scalar(
        &self,
        name: MetricName,
        step: u64,
        value: f32,
    ) -> Result<(), TelemetryError> {
        self.metric_sink
            .emit_scalar(&self.context, name, step, value)
    }

    /// Emit one scalar metric by stable string name.
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError::UnknownMetric`] when `name` is outside the closed registry,
    /// or a sink error when the record cannot be persisted.
    pub fn emit_scalar_named(
        &self,
        name: &str,
        step: u64,
        value: f32,
    ) -> Result<(), TelemetryError> {
        self.emit_scalar(MetricName::from_name(name)?, step, value)
    }

    /// Emit a batch of scalar metrics at one step.
    ///
    /// # Errors
    ///
    /// Returns the first sink error encountered while emitting the batch.
    pub fn emit_scalars(
        &self,
        step: u64,
        batch: &[(MetricName, f32)],
    ) -> Result<(), TelemetryError> {
        for (name, value) in batch {
            self.emit_scalar(*name, step, *value)?;
        }
        Ok(())
    }

    /// Emit one histogram metric.
    ///
    /// # Errors
    ///
    /// Returns an error when the sink rejects the record.
    pub fn emit_histogram(
        &self,
        name: MetricName,
        step: u64,
        values: &[f32],
    ) -> Result<(), TelemetryError> {
        self.metric_sink
            .emit_histogram(&self.context, name, step, values)
    }

    /// Start a named span with run-level attributes.
    #[must_use]
    pub fn start_span(&self, name: SpanName) -> SpanGuard<'_> {
        SpanGuard {
            context: &self.context,
            name,
            step: None,
            epoch: None,
            started_at: Instant::now(),
            _entered_span: tracing_span_for(&self.context, name, None, None).entered(),
        }
    }

    /// Start a named step-level span with required step and epoch attributes.
    #[must_use]
    pub fn start_step_span(&self, name: SpanName, step: u64, epoch: u64) -> SpanGuard<'_> {
        SpanGuard {
            context: &self.context,
            name,
            step: Some(step),
            epoch: Some(epoch),
            started_at: Instant::now(),
            _entered_span: tracing_span_for(&self.context, name, Some(step), Some(epoch)).entered(),
        }
    }

    /// Flush all exporters and consume the facade.
    ///
    /// # Errors
    ///
    /// Returns an error when any exporter fails to flush.
    pub fn shutdown(self) -> Result<(), TelemetryError> {
        self.metric_sink.flush()
    }
}

/// Active telemetry span guard.
#[derive(Debug)]
pub struct SpanGuard<'a> {
    context: &'a TelemetryContext,
    name: SpanName,
    step: Option<u64>,
    epoch: Option<u64>,
    started_at: Instant,
    _entered_span: tracing::span::EnteredSpan,
}

impl SpanGuard<'_> {
    /// Span name.
    #[must_use]
    pub fn name(&self) -> SpanName {
        self.name
    }

    /// Run context attached to this span.
    #[must_use]
    pub fn context(&self) -> &TelemetryContext {
        self.context
    }

    /// Optional step attribute for step-level spans.
    #[must_use]
    pub fn step(&self) -> Option<u64> {
        self.step
    }

    /// Optional epoch attribute for step-level spans.
    #[must_use]
    pub fn epoch(&self) -> Option<u64> {
        self.epoch
    }

    /// Current wall-clock duration of the active span.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }
}

fn tracing_span_for(
    context: &TelemetryContext,
    name: SpanName,
    step: Option<u64>,
    epoch: Option<u64>,
) -> tracing::Span {
    if step.is_some() || epoch.is_some() {
        return tracing_step_span_for(
            context,
            name,
            step.unwrap_or_default(),
            epoch.unwrap_or_default(),
        );
    }

    tracing_run_span_for(context, name)
}

fn tracing_run_span_for(context: &TelemetryContext, name: SpanName) -> tracing::Span {
    macro_rules! run_span {
        ($span_name:literal) => {
            tracing::info_span!(
                $span_name,
                run_id = %context.run_id,
                phase = %context.phase,
                git_short_sha = %context.git_short_sha
            )
        };
    }

    match name {
        n if n == SpanName::TRAINING_RUN => run_span!("training.run"),
        n if n == SpanName::TRAINING_EPOCH => run_span!("training.epoch"),
        n if n == SpanName::TRAINING_STEP => run_span!("training.step"),
        n if n == SpanName::TRAINING_FORWARD => run_span!("training.forward"),
        n if n == SpanName::TRAINING_BACKWARD => run_span!("training.backward"),
        n if n == SpanName::TRAINING_OPTIM_STEP => run_span!("training.optim_step"),
        n if n == SpanName::TRAINING_CHECKPOINT_SAVE => run_span!("training.checkpoint_save"),
        n if n == SpanName::TRAINING_PARITY_PROBE => run_span!("training.parity_probe"),
        n if n == SpanName::TRAINING_COLLAPSE_PROBE => run_span!("training.collapse_probe"),
        n if n == SpanName::TRAINING_EVAL => run_span!("training.eval"),
        n if n == SpanName::EVAL_EPISODE => run_span!("eval.episode"),
        n if n == SpanName::EVAL_CEM_ITER => run_span!("eval.cem_iter"),
        n if n == SpanName::EVAL_CEM_COST_EVAL => run_span!("eval.cem_cost_eval"),
        n if n == SpanName::EVAL_RPC_STEP => run_span!("eval.rpc_step"),
        n if n == SpanName::DATA_DATASET_OPEN => run_span!("data.dataset_open"),
        n if n == SpanName::DATA_GET_WINDOW => run_span!("data.get_window"),
        n if n == SpanName::DATA_COLLATE => run_span!("data.collate"),
        n if n == SpanName::DATA_PREFETCH_WORKER_LIFETIME => {
            run_span!("data.prefetch_worker.lifetime")
        },
        _ => run_span!("lewm.unknown"),
    }
}

fn tracing_step_span_for(
    context: &TelemetryContext,
    name: SpanName,
    step: u64,
    epoch: u64,
) -> tracing::Span {
    macro_rules! step_span {
        ($span_name:literal) => {
            tracing::info_span!(
                $span_name,
                run_id = %context.run_id,
                phase = %context.phase,
                git_short_sha = %context.git_short_sha,
                step = step,
                epoch = epoch
            )
        };
    }

    match name {
        n if n == SpanName::TRAINING_RUN => step_span!("training.run"),
        n if n == SpanName::TRAINING_EPOCH => step_span!("training.epoch"),
        n if n == SpanName::TRAINING_STEP => step_span!("training.step"),
        n if n == SpanName::TRAINING_FORWARD => step_span!("training.forward"),
        n if n == SpanName::TRAINING_BACKWARD => step_span!("training.backward"),
        n if n == SpanName::TRAINING_OPTIM_STEP => step_span!("training.optim_step"),
        n if n == SpanName::TRAINING_CHECKPOINT_SAVE => step_span!("training.checkpoint_save"),
        n if n == SpanName::TRAINING_PARITY_PROBE => step_span!("training.parity_probe"),
        n if n == SpanName::TRAINING_COLLAPSE_PROBE => step_span!("training.collapse_probe"),
        n if n == SpanName::TRAINING_EVAL => step_span!("training.eval"),
        n if n == SpanName::EVAL_EPISODE => step_span!("eval.episode"),
        n if n == SpanName::EVAL_CEM_ITER => step_span!("eval.cem_iter"),
        n if n == SpanName::EVAL_CEM_COST_EVAL => step_span!("eval.cem_cost_eval"),
        n if n == SpanName::EVAL_RPC_STEP => step_span!("eval.rpc_step"),
        n if n == SpanName::DATA_DATASET_OPEN => step_span!("data.dataset_open"),
        n if n == SpanName::DATA_GET_WINDOW => step_span!("data.get_window"),
        n if n == SpanName::DATA_COLLATE => step_span!("data.collate"),
        n if n == SpanName::DATA_PREFETCH_WORKER_LIFETIME => {
            step_span!("data.prefetch_worker.lifetime")
        },
        _ => step_span!("lewm.unknown"),
    }
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
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    #[derive(Debug, Default)]
    struct RecordingSink {
        records: Mutex<Vec<String>>,
        flush_count: AtomicUsize,
    }

    impl RecordingSink {
        fn records(&self) -> Vec<String> {
            self.records
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }

        fn flush_count(&self) -> usize {
            self.flush_count.load(Ordering::Acquire)
        }
    }

    impl MetricSink for RecordingSink {
        fn emit_scalar(
            &self,
            context: &TelemetryContext,
            name: MetricName,
            step: u64,
            value: f32,
        ) -> Result<(), TelemetryError> {
            self.records
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(format!(
                    "{}:{}:{}:{}:{value}",
                    context.run_id,
                    context.phase,
                    name.as_str(),
                    step
                ));
            Ok(())
        }

        fn emit_histogram(
            &self,
            context: &TelemetryContext,
            name: MetricName,
            step: u64,
            values: &[f32],
        ) -> Result<(), TelemetryError> {
            self.records
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(format!(
                    "{}:{}:{}:{}:{}",
                    context.run_id,
                    context.phase,
                    name.as_str(),
                    step,
                    values.len()
                ));
            Ok(())
        }

        fn flush(&self) -> Result<(), TelemetryError> {
            self.flush_count.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }
    }

    #[test]
    fn shutdown_flushes_all_exporters() -> Result<(), Box<dyn std::error::Error>> {
        let sink = Arc::new(RecordingSink::default());
        let telemetry = Telemetry::with_metric_sink(
            TelemetryConfig::new("run-001", "phase-2", "abc1234"),
            sink.clone(),
        )?;

        telemetry.emit_scalar(MetricName::LossTotal, 7, 1.5)?;
        telemetry.emit_scalars(
            7,
            &[
                (MetricName::OptimLr, 1e-4),
                (MetricName::DataQueueDepth, 3.0),
            ],
        )?;
        telemetry.emit_histogram(MetricName::ModelEncoderClsVar, 7, &[0.1, 0.2, 0.3])?;

        {
            let span = telemetry.start_step_span(SpanName::TRAINING_STEP, 7, 2);
            assert_eq!(span.name(), SpanName::TRAINING_STEP);
            assert_eq!(span.step(), Some(7));
            assert_eq!(span.epoch(), Some(2));
        }

        telemetry.shutdown()?;

        assert_eq!(sink.flush_count(), 1);
        assert_eq!(sink.records().len(), 4);
        assert!(sink.records()[0].contains("run-001:phase-2:loss/total:7:1.5"));
        Ok(())
    }

    #[test]
    fn unknown_metric_returns_error() -> Result<(), Box<dyn std::error::Error>> {
        let telemetry = Telemetry::init(TelemetryConfig::new("run-001", "phase-2", "abc1234"))?;
        let err = telemetry
            .emit_scalar_named("loss/not_registered", 7, 1.0)
            .err()
            .ok_or("unknown metric should fail")?;

        assert!(
            matches!(err, TelemetryError::UnknownMetric(name) if name == "loss/not_registered")
        );
        Ok(())
    }
}
