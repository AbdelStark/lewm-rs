//! Telemetry facade, structured logging, metric naming, collapse detection
//! signals, and export boundaries for `Trackio` and `OpenTelemetry`. This crate
//! keeps observability concerns out of the core model; see [RFC 0009].
//!
//! [RFC 0009]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0009-observability-and-mlops.md
//!
//! ## Module index
//!
//! - [`Telemetry`] is the public facade for metrics and spans.
//! - [`MetricName`] and [`SpanName`] are the closed RFC 0009 registries.

pub mod collapse;
pub mod errors;
pub mod facade;
pub mod logs;
pub mod metrics;
#[cfg(feature = "nvtx")]
pub mod nvtx_layer;
pub mod otlp;
pub mod spans;
pub mod system;
pub mod tensorboard;
pub mod tracker;

pub use crate::collapse::{
    COLLAPSE_PROBE_BATCH_FRAMES, COLLAPSE_PROBE_FIXTURE_PATH, COLLAPSE_TRIPS_REQUIRED,
    CollapseDetector, CollapseDetectorConfig, CollapseDetectorDecision,
};
pub use crate::errors::TelemetryError;
pub use crate::facade::{
    MetricFanout, MetricSink, SpanGuard, Telemetry, TelemetryConfig, TelemetryContext,
};
pub use crate::logs::{init_logging, init_logging_with_config, init_logging_with_tracer};
pub use crate::metrics::{MetricKind, MetricName};
#[cfg(feature = "nvtx")]
pub use crate::nvtx_layer::{NvtxLayer, nvtx_layer};
pub use crate::otlp::{
    OtlpSpanGuard, OtlpTracer, init_tracer, init_tracer_from_env, init_tracer_with_context,
};
pub use crate::spans::SpanName;
pub use crate::system::{
    GpuMetrics, SYSTEM_DISK_CADENCE, SYSTEM_FAST_CADENCE, SystemEmitReport, SystemMetricCadence,
    SystemMetrics, SystemSampleDue, SystemSampler,
};
pub use crate::tensorboard::TensorboardWriter;
pub use crate::tracker::TrackioWriter;
