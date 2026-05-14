//! Opt-in smoke test for exporting one span to a local OTLP collector.

use lewm_telemetry::{SpanName, Telemetry, TelemetryConfig};

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a local OTLP collector; use scripts/otel_smoke.py"]
async fn otlp_endpoint_accepts_span_from_facade() -> Result<(), Box<dyn std::error::Error>> {
    let endpoint = match std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
        Ok(value) if !value.trim().is_empty() => value,
        _ => {
            eprintln!("skipping OTLP smoke: OTEL_EXPORTER_OTLP_ENDPOINT is unset");
            return Ok(());
        },
    };
    eprintln!("exporting OTLP smoke span to {endpoint}");

    let telemetry = Telemetry::init(TelemetryConfig::new(
        "otel-smoke-run",
        "local-otel-smoke",
        "local",
    ))?;

    {
        let _span = telemetry.start_step_span(SpanName::TRAINING_STEP, 1, 0);
    }

    telemetry.shutdown()?;
    Ok(())
}
