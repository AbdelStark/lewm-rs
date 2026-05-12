//! Error types for telemetry initialization and emission.

/// Telemetry facade failures.
#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    /// The telemetry configuration is invalid.
    #[error("invalid telemetry configuration: {0}")]
    InvalidConfig(String),

    /// A caller attempted to emit a metric outside the RFC 0009 registry.
    #[error("unknown metric name: {0}")]
    UnknownMetric(String),

    /// A metric sink failed while accepting or flushing records.
    #[error("metric sink error: {0}")]
    Sink(String),
}
