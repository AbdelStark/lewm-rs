# `lewm-telemetry`

Structured-logging + OpenTelemetry exporter facade. Cost-zero when
no OTLP endpoint is configured.

## What it owns

- **Trait `Telemetry`** with a `no-op` impl for tests and an OTLP
  impl for production.
- **JSONL writer**: emits one line per step / event to stdout.
- **OTLP wiring**: initialises a tracer when
  `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
- **Schema**: stable v1.0.0 schema for `StepRecord` and
  `TransitionRecord`.

## Public API

```rust,ignore
pub trait Telemetry: Send + Sync {
    fn emit_step(&self, record: StepRecord);
    fn emit_transition(&self, from: State, to: State, gate_result: GateResult);
    fn emit_error(&self, kind: &str, message: &str, context: serde_json::Value);
    fn flush(&self);
}
```

## Cost-zero contract

`crates/lewm-telemetry/tests/no_endpoint_no_cost.rs` verifies that
when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset, `Telemetry::emit_step`
does no network I/O and no allocation beyond the JSONL line itself.

## Dependencies

- `opentelemetry`, `opentelemetry_sdk`, `opentelemetry-otlp` (all
  optional, gated behind feature `otlp`)
- `serde_json`
- `tracing`

## Source

[`crates/lewm-telemetry`](https://github.com/AbdelStark/lewm-rs/tree/main/crates/lewm-telemetry)
