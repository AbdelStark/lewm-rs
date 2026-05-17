//! `OpenTelemetry` `OTLP` trace exporter.

use std::{env, fmt, sync::Arc, time::SystemTime};

use opentelemetry::{
    KeyValue,
    trace::{Span as _, Tracer as _, TracerProvider as _},
};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    Resource,
    trace::{
        self, BatchConfigBuilder, BatchSpanProcessor, SdkTracer as Tracer, SdkTracerProvider,
        SpanExporter,
    },
};

use crate::{SpanName, TelemetryContext, TelemetryError};

const SERVICE_NAME: &str = "lewm-rs";
const TRACER_NAME: &str = "lewm";
const OTLP_ENDPOINT_ENV: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";
const OTLP_MAX_EXPORT_BATCH_SIZE: usize = 512;
const OTLP_FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// `OTLP` trace exporter handle.
#[derive(Clone)]
pub struct OtlpTracer {
    provider: Arc<SdkTracerProvider>,
    tracer: Tracer,
}

impl fmt::Debug for OtlpTracer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OtlpTracer")
            .field("service_name", &SERVICE_NAME)
            .finish_non_exhaustive()
    }
}

impl OtlpTracer {
    /// Start an exported span with RFC 0009 run attributes.
    #[must_use]
    pub fn start_span(
        &self,
        context: &TelemetryContext,
        name: SpanName,
        step: Option<u64>,
        epoch: Option<u64>,
    ) -> OtlpSpanGuard {
        let span = self
            .tracer
            .span_builder(name.as_str())
            .with_start_time(SystemTime::now())
            .with_attributes(span_attributes(context, step, epoch))
            .start(&self.tracer);
        OtlpSpanGuard { span: Some(span) }
    }

    /// Clone the underlying SDK tracer for advanced integrations.
    #[must_use]
    pub fn tracer(&self) -> Tracer {
        self.tracer.clone()
    }

    /// Flush queued spans without consuming the exporter.
    ///
    /// # Errors
    ///
    /// Returns an error when any span processor fails to flush.
    pub fn force_flush(&self) -> Result<(), TelemetryError> {
        self.provider
            .force_flush()
            .map_err(|err| TelemetryError::TraceExporter(err.to_string()))
    }

    /// Flush and shut down the exporter.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider fails to shut down.
    pub fn shutdown(&self) -> Result<(), TelemetryError> {
        self.provider
            .shutdown()
            .map_err(|err| TelemetryError::TraceExporter(err.to_string()))
    }

    fn from_exporter(exporter: impl SpanExporter + 'static, context: &TelemetryContext) -> Self {
        let processor = BatchSpanProcessor::builder(exporter)
            .with_batch_config(otlp_batch_config())
            .build();
        let provider = SdkTracerProvider::builder()
            .with_span_processor(processor)
            .with_resource(resource_attributes(context))
            .build();
        let tracer = provider.tracer(TRACER_NAME);

        Self {
            provider: Arc::new(provider),
            tracer,
        }
    }
}

/// Active `OTLP` span guard.
pub struct OtlpSpanGuard {
    span: Option<opentelemetry_sdk::trace::Span>,
}

impl fmt::Debug for OtlpSpanGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OtlpSpanGuard")
            .field("active", &self.span.is_some())
            .finish()
    }
}

impl Drop for OtlpSpanGuard {
    fn drop(&mut self) {
        if let Some(mut span) = self.span.take() {
            span.end();
        }
    }
}

/// Initialize an `OTLP` tracer from an explicit endpoint.
///
/// The `git_short_sha` resource attribute is set to `unknown`; use
/// [`init_tracer_with_context`] or [`init_tracer_from_env`] when a full run
/// context is available.
/// Must be called from a Tokio runtime because the batch span processor uses
/// Tokio for flush scheduling.
///
/// # Errors
///
/// Returns an error when the `OTLP` exporter cannot be built.
pub fn init_tracer(endpoint: &str, run_id: &str) -> Result<OtlpTracer, TelemetryError> {
    let context = TelemetryContext {
        run_id: run_id.to_string(),
        phase: "unknown".to_string(),
        git_short_sha: "unknown".to_string(),
    };
    init_tracer_with_context(endpoint, &context)
}

/// Initialize an `OTLP` tracer from an explicit endpoint and full run context.
///
/// Must be called from a Tokio runtime because the batch span processor uses
/// Tokio for flush scheduling.
///
/// # Errors
///
/// Returns an error when the `OTLP` exporter cannot be built.
pub fn init_tracer_with_context(
    endpoint: &str,
    context: &TelemetryContext,
) -> Result<OtlpTracer, TelemetryError> {
    if endpoint.trim().is_empty() {
        return Err(TelemetryError::InvalidConfig(
            "OTLP endpoint must be non-empty".to_string(),
        ));
    }

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint.to_string())
        .build()
        .map_err(|err| TelemetryError::TraceExporter(err.to_string()))?;

    Ok(OtlpTracer::from_exporter(exporter, context))
}

/// Initialize an `OTLP` tracer from `OTEL_EXPORTER_OTLP_ENDPOINT`.
///
/// Must be called from a Tokio runtime when the endpoint is set. If the endpoint
/// is unset, this returns `Ok(None)` without requiring a runtime.
///
/// # Errors
///
/// Returns an error when the environment endpoint is present but the exporter cannot be built.
pub fn init_tracer_from_env(
    context: &TelemetryContext,
) -> Result<Option<OtlpTracer>, TelemetryError> {
    init_tracer_from_endpoint(env::var(OTLP_ENDPOINT_ENV).ok(), context)
}

fn init_tracer_from_endpoint(
    endpoint: Option<String>,
    context: &TelemetryContext,
) -> Result<Option<OtlpTracer>, TelemetryError> {
    match endpoint.map(|value| value.trim().to_string()) {
        Some(endpoint) if !endpoint.is_empty() => {
            init_tracer_with_context(&endpoint, context).map(Some)
        },
        _ => {
            tracing::warn!(
                target: "lewm_telemetry::otlp",
                "OTLP exporter disabled; OTEL_EXPORTER_OTLP_ENDPOINT is unset"
            );
            Ok(None)
        },
    }
}

fn resource_attributes(context: &TelemetryContext) -> Resource {
    Resource::builder_empty()
        .with_attributes([
            KeyValue::new("service.name", SERVICE_NAME),
            KeyValue::new("run.id", context.run_id.clone()),
            KeyValue::new("git_short_sha", context.git_short_sha.clone()),
        ])
        .build()
}

fn span_attributes(
    context: &TelemetryContext,
    step: Option<u64>,
    epoch: Option<u64>,
) -> Vec<KeyValue> {
    let mut attributes = vec![
        KeyValue::new("run_id", context.run_id.clone()),
        KeyValue::new("phase", context.phase.clone()),
        KeyValue::new("git_short_sha", context.git_short_sha.clone()),
    ];
    if let Some(step) = step {
        attributes.push(KeyValue::new("step", bounded_i64(step)));
    }
    if let Some(epoch) = epoch {
        attributes.push(KeyValue::new("epoch", bounded_i64(epoch)));
    }
    attributes
}

fn bounded_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn otlp_batch_config() -> trace::BatchConfig {
    BatchConfigBuilder::default()
        .with_max_export_batch_size(OTLP_MAX_EXPORT_BATCH_SIZE)
        .with_scheduled_delay(OTLP_FLUSH_INTERVAL)
        .build()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use opentelemetry::Key;
    use opentelemetry_sdk::error::OTelSdkResult;
    use opentelemetry_sdk::trace::SpanData;

    use super::*;

    #[derive(Clone, Debug, Default)]
    struct RecordingSpanExporter {
        spans: Arc<Mutex<Vec<SpanData>>>,
        resource: Arc<Mutex<Option<Resource>>>,
    }

    impl RecordingSpanExporter {
        fn spans(&self) -> Vec<SpanData> {
            self.spans
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }

        fn resource(&self) -> Option<Resource> {
            self.resource
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }
    }

    impl SpanExporter for RecordingSpanExporter {
        async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
            self.spans
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .extend(batch);
            Ok(())
        }

        fn set_resource(&mut self, resource: &Resource) {
            *self
                .resource
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(resource.clone());
        }
    }

    #[test]
    fn otlp_disabled_without_endpoint() -> Result<(), Box<dyn std::error::Error>> {
        let context = context();

        let tracer = init_tracer_from_endpoint(None, &context)?;

        assert!(tracer.is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn otlp_export_smoke_with_local_collector() -> Result<(), Box<dyn std::error::Error>> {
        let exporter = RecordingSpanExporter::default();
        let context = context();
        let tracer = OtlpTracer::from_exporter(exporter.clone(), &context);

        {
            let _span = tracer.start_span(&context, SpanName::TRAINING_STEP, Some(42), Some(7));
        }
        tracer.force_flush()?;

        let spans = exporter.spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "training.step");
        assert_attr(&spans[0].attributes, "run_id", "run-001");
        assert_attr(&spans[0].attributes, "phase", "phase-2");
        assert_attr(&spans[0].attributes, "git_short_sha", "abc1234");
        assert_attr(&spans[0].attributes, "step", "42");
        assert_attr(&spans[0].attributes, "epoch", "7");

        let resource = exporter.resource().ok_or("resource should be set")?;
        assert_resource(&resource, "service.name", SERVICE_NAME);
        assert_resource(&resource, "run.id", "run-001");
        assert_resource(&resource, "git_short_sha", "abc1234");
        Ok(())
    }

    fn context() -> TelemetryContext {
        TelemetryContext {
            run_id: "run-001".to_string(),
            phase: "phase-2".to_string(),
            git_short_sha: "abc1234".to_string(),
        }
    }

    fn assert_attr(attributes: &[KeyValue], key: &str, value: &str) {
        let actual = attributes
            .iter()
            .find(|attr| attr.key.as_str() == key)
            .map(|attr| attr.value.to_string());
        assert_eq!(actual.as_deref(), Some(value));
    }

    fn assert_resource(resource: &Resource, key: &'static str, value: &str) {
        let actual = resource
            .get(&Key::from_static_str(key))
            .map(|value| value.to_string());
        assert_eq!(actual.as_deref(), Some(value));
    }
}
