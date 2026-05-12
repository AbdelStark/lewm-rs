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

pub mod errors;
pub mod facade;
pub mod logs;
pub mod metrics;
pub mod spans;

pub use crate::errors::TelemetryError;
pub use crate::facade::{MetricSink, SpanGuard, Telemetry, TelemetryConfig, TelemetryContext};
pub use crate::logs::{init_logging, init_logging_with_config, init_logging_with_tracer};
pub use crate::metrics::{MetricKind, MetricName};
pub use crate::spans::SpanName;
