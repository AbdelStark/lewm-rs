//! Structured JSON logging and secret redaction for RFC 0009.

use std::{
    fmt,
    io::{self, Write},
    sync::{Arc, LazyLock},
    time::Instant,
};

use chrono::{SecondsFormat, Utc};
use opentelemetry::trace::Tracer as OtelTracer;
use regex::Regex;
use serde_json::{Map, Number, Value};
use tracing::{
    Event, Subscriber,
    field::{Field, Visit},
    span::{Attributes, Id, Record},
};
use tracing_subscriber::{
    EnvFilter, Layer, Registry,
    layer::{Context, SubscriberExt},
    registry::LookupSpan,
    util::SubscriberInitExt,
};

use crate::{TelemetryConfig, TelemetryContext, TelemetryError};

const DEFAULT_ENV_FILTER: &str = "info";
const REDACTED: &str = "[REDACTED]";

static SENSITIVE_FIELD_RE: LazyLock<Result<Regex, regex::Error>> =
    LazyLock::new(|| Regex::new(r"(?i)^(.*(token|secret|key|password).*)$"));

/// Initialize structured JSON logging with a no-op OpenTelemetry layer.
///
/// Use [`init_logging_with_tracer`] when an OTLP tracer is available. This
/// variant still installs the OpenTelemetry bridge layer with its no-op tracer
/// so `tracing` spans are shaped consistently even without a remote exporter.
///
/// # Errors
///
/// Returns an error when the context is invalid, the `RUST_LOG` filter cannot
/// be built, or a global tracing subscriber is already installed.
pub fn init_logging(context: TelemetryContext) -> Result<(), TelemetryError> {
    validate_context(&context)?;
    let subscriber = Registry::default()
        .with(env_filter()?)
        .with(tracing_opentelemetry::layer())
        .with(StructuredJsonLayer::stdout(context));

    subscriber
        .try_init()
        .map_err(|error| TelemetryError::Logger(error.to_string()))
}

/// Initialize structured JSON logging from a telemetry config.
///
/// # Errors
///
/// Returns an error under the same conditions as [`init_logging`].
pub fn init_logging_with_config(config: TelemetryConfig) -> Result<(), TelemetryError> {
    init_logging(config.into())
}

/// Initialize structured JSON logging and attach an OpenTelemetry tracer layer.
///
/// # Errors
///
/// Returns an error when the context is invalid, the `RUST_LOG` filter cannot
/// be built, or a global tracing subscriber is already installed.
pub fn init_logging_with_tracer<T>(
    context: TelemetryContext,
    tracer: T,
) -> Result<(), TelemetryError>
where
    T: OtelTracer + Send + Sync + 'static,
    T::Span: Send + Sync + 'static,
{
    validate_context(&context)?;
    let subscriber = Registry::default()
        .with(env_filter()?)
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .with(StructuredJsonLayer::stdout(context));

    subscriber
        .try_init()
        .map_err(|error| TelemetryError::Logger(error.to_string()))
}

#[derive(Clone)]
struct StructuredJsonLayer {
    context: TelemetryContext,
    sink: Arc<dyn LogSink>,
    started_at: Instant,
}

impl StructuredJsonLayer {
    fn stdout(context: TelemetryContext) -> Self {
        Self::with_sink(
            context,
            Arc::new(StdoutLogSink),
            Instant::now(), // determinism-lint: allow Instant::now telemetry wall time
        )
    }

    fn with_sink(context: TelemetryContext, sink: Arc<dyn LogSink>, started_at: Instant) -> Self {
        Self {
            context,
            sink,
            started_at,
        }
    }

    fn wall_time_ms(&self) -> Value {
        number_value(self.started_at.elapsed().as_secs_f64() * 1000.0)
    }
}

impl fmt::Debug for StructuredJsonLayer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StructuredJsonLayer")
            .field("context", &self.context)
            .field("started_at", &self.started_at)
            .finish_non_exhaustive()
    }
}

impl<S> Layer<S> for StructuredJsonLayer
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        attrs.record(&mut visitor);

        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(SpanFields {
                fields: visitor.fields,
            });
        }
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else {
            return;
        };

        let mut visitor = FieldVisitor::default();
        values.record(&mut visitor);
        let mut extensions = span.extensions_mut();

        if let Some(stored) = extensions.get_mut::<SpanFields>() {
            merge_fields(&mut stored.fields, visitor.fields);
        } else {
            extensions.insert(SpanFields {
                fields: visitor.fields,
            });
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let mut fields = Map::new();

        if let Some(scope) = ctx.event_scope(event) {
            for span in scope.from_root() {
                if let Some(span_fields) = span.extensions().get::<SpanFields>() {
                    merge_fields(&mut fields, span_fields.fields.clone());
                }
            }
        }

        let mut event_fields = FieldVisitor::default();
        event.record(&mut event_fields);
        merge_fields(&mut fields, event_fields.fields);

        fields.insert(
            "run_id".to_string(),
            Value::String(self.context.run_id.clone()),
        );
        fields.insert(
            "phase".to_string(),
            Value::String(self.context.phase.clone()),
        );
        fields.insert(
            "git_short_sha".to_string(),
            Value::String(self.context.git_short_sha.clone()),
        );
        fields.insert("wall_time_ms".to_string(), self.wall_time_ms());

        let metadata = event.metadata();
        let mut record = Map::new();
        record.insert(
            "timestamp".to_string(),
            Value::String(Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)),
        );
        record.insert(
            "level".to_string(),
            Value::String(metadata.level().as_str().to_string()),
        );
        record.insert(
            "target".to_string(),
            Value::String(metadata.target().to_string()),
        );
        record.insert(
            "message".to_string(),
            Value::String(
                event_fields
                    .message
                    .unwrap_or_else(|| metadata.name().to_string()),
            ),
        );
        record.insert("fields".to_string(), Value::Object(fields));

        if let Ok(line) = serde_json::to_string(&Value::Object(record)) {
            let _ = self.sink.write_line(&line);
        }
    }
}

#[derive(Debug, Clone)]
struct SpanFields {
    fields: Map<String, Value>,
}

#[derive(Debug, Default)]
struct FieldVisitor {
    fields: Map<String, Value>,
    message: Option<String>,
}

impl FieldVisitor {
    fn record_value(&mut self, field: &Field, value: Value) {
        let name = field.name();
        let value = if is_sensitive_field(name) {
            Value::String(REDACTED.to_string())
        } else {
            value
        };

        if name == "message" {
            self.message = Some(message_value(&value));
        } else {
            self.fields.insert(name.to_string(), value);
        }
    }
}

impl Visit for FieldVisitor {
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.record_value(field, Value::Bool(value));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.record_value(field, Value::Number(Number::from(value)));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.record_value(field, Value::Number(Number::from(value)));
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.record_value(field, number_value(value));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.record_value(field, Value::String(value.to_string()));
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.record_value(field, debug_value(value));
    }
}

trait LogSink: Send + Sync {
    fn write_line(&self, line: &str) -> io::Result<()>;
}

#[derive(Debug)]
struct StdoutLogSink;

impl LogSink for StdoutLogSink {
    fn write_line(&self, line: &str) -> io::Result<()> {
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        writeln!(handle, "{line}")
    }
}

fn env_filter() -> Result<EnvFilter, TelemetryError> {
    EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(DEFAULT_ENV_FILTER))
        .map_err(|error| TelemetryError::Logger(error.to_string()))
}

fn validate_context(context: &TelemetryContext) -> Result<(), TelemetryError> {
    validate_non_empty("run_id", &context.run_id)?;
    validate_non_empty("phase", &context.phase)?;
    validate_non_empty("git_short_sha", &context.git_short_sha)
}

fn validate_non_empty(field: &str, value: &str) -> Result<(), TelemetryError> {
    if value.trim().is_empty() {
        return Err(TelemetryError::InvalidConfig(format!(
            "{field} must be non-empty"
        )));
    }
    Ok(())
}

fn is_sensitive_field(name: &str) -> bool {
    match &*SENSITIVE_FIELD_RE {
        Ok(regex) => regex.is_match(name),
        Err(_) => is_sensitive_field_fallback(name),
    }
}

fn is_sensitive_field_fallback(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
    normalized.contains("token")
        || normalized.contains("secret")
        || normalized.contains("key")
        || normalized.contains("password")
}

fn merge_fields(target: &mut Map<String, Value>, source: Map<String, Value>) {
    for (key, value) in source {
        target.insert(key, value);
    }
}

fn number_value(value: f64) -> Value {
    Number::from_f64(value).map_or_else(|| Value::String(value.to_string()), Value::Number)
}

fn debug_value(value: &dyn fmt::Debug) -> Value {
    let rendered = format!("{value:?}");
    serde_json::from_str::<Value>(&rendered).unwrap_or(Value::String(rendered))
}

fn message_value(value: &Value) -> String {
    match value {
        Value::String(message) => message.clone(),
        _ => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use serde_json::json;

    use super::*;

    #[derive(Debug, Default)]
    struct MemoryLogSink {
        lines: Mutex<Vec<String>>,
    }

    impl MemoryLogSink {
        fn lines(&self) -> Vec<String> {
            self.lines
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }
    }

    impl LogSink for MemoryLogSink {
        fn write_line(&self, line: &str) -> io::Result<()> {
            self.lines
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(line.to_string());
            Ok(())
        }
    }

    fn test_context() -> TelemetryContext {
        TelemetryContext {
            run_id: "run-structured-001".to_string(),
            phase: "phase-2".to_string(),
            git_short_sha: "abc1234".to_string(),
        }
    }

    fn subscriber_with_sink(sink: Arc<MemoryLogSink>) -> impl Subscriber + Send + Sync {
        Registry::default().with(StructuredJsonLayer::with_sink(
            test_context(),
            sink,
            Instant::now(), // determinism-lint: allow Instant::now telemetry wall time
        ))
    }

    #[test]
    fn redactor_masks_token_field() -> Result<(), Box<dyn std::error::Error>> {
        let sink = Arc::new(MemoryLogSink::default());
        let subscriber = subscriber_with_sink(sink.clone());

        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(
                hf_token = "hf_live_value",
                api_secret = "secret-value",
                public_field = "visible",
                "loaded credentials"
            );
        });

        let lines = sink.lines();
        assert_eq!(lines.len(), 1);
        assert!(!lines[0].contains("hf_live_value"));
        assert!(!lines[0].contains("secret-value"));

        let record: Value = serde_json::from_str(&lines[0])?;
        assert_eq!(record["message"], json!("loaded credentials"));
        assert_eq!(record["fields"]["hf_token"], json!(REDACTED));
        assert_eq!(record["fields"]["api_secret"], json!(REDACTED));
        assert_eq!(record["fields"]["public_field"], json!("visible"));
        Ok(())
    }

    #[test]
    fn log_line_includes_run_context_wall_time_and_span_step()
    -> Result<(), Box<dyn std::error::Error>> {
        let sink = Arc::new(MemoryLogSink::default());
        let subscriber = subscriber_with_sink(sink.clone());

        tracing::subscriber::with_default(subscriber, || {
            tracing::info_span!("training.step", step = 42_u64, epoch = 7_u64)
                .in_scope(|| tracing::info!("finished step"));
        });

        let lines = sink.lines();
        assert_eq!(lines.len(), 1);

        let record: Value = serde_json::from_str(&lines[0])?;
        assert_eq!(record["level"], json!("INFO"));
        assert_eq!(record["message"], json!("finished step"));
        assert_eq!(record["fields"]["run_id"], json!("run-structured-001"));
        assert_eq!(record["fields"]["phase"], json!("phase-2"));
        assert_eq!(record["fields"]["git_short_sha"], json!("abc1234"));
        assert_eq!(record["fields"]["step"], json!(42));
        assert_eq!(record["fields"]["epoch"], json!(7));
        assert!(record["fields"]["wall_time_ms"].is_number());
        Ok(())
    }
}
