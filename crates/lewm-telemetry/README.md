# `lewm-telemetry`

Telemetry facade, structured logging, metric naming, collapse detection
signals, and export boundaries for Trackio and OpenTelemetry. This crate
keeps observability concerns out of the core model so `lewm-core` stays
agnostic to where metrics land.

**Specs:** [RFC 0009 — observability and MLOps][rfc-0009].

**Depends on:** `lewm-core`.

## Module map

- `facade` — public `Telemetry` facade. All instrumentation goes through it.
- `metrics` — `MetricName` registry (closed enum, RFC 0009).
- `spans` — `SpanName` registry.
- `logs` — structured logging hooks (JSON via `tracing-subscriber`).
- `otlp` — OpenTelemetry OTLP/gRPC exporter wiring.
- `tracker` — Trackio compatibility helpers.
- `tensorboard` — TensorBoard event-file writer for offline curve plots.
- `collapse` — `CollapseDetector` (`COLLAPSE_TRIPS_REQUIRED`,
  `COLLAPSE_PROBE_BATCH_FRAMES`) for SIGReg-style representation collapse.
- `system` — CPU/GPU/memory sampling (gracefully degrades without `nvml`).
- `nvtx_layer` (feature `nvtx`) — NVIDIA Tools Extension layer for `nsys`.

## Exports

- **OTLP**: when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, traces, metrics, and
  logs are exported via gRPC. The `infra/otel/` directory ships a
  docker-compose stack (Tempo + Prometheus + Grafana) for local development.
- **Trackio**: enabled via `TRACKIO_PROJECT` / `TRACKIO_RUN`.
- **TensorBoard**: written under the run directory as `events.out.tfevents.*`.

CI and smoke runs do **not** require OTLP; the exporter no-ops when the
endpoint is unset.

## Collapse detection

The detector consumes activation statistics emitted by `lewm-core` and trips
after `COLLAPSE_TRIPS_REQUIRED = 3` consecutive low-variance batches, which
causes the trainer to halt with a non-zero exit code (`CollapseDetected`).

[rfc-0009]: ../../specs/rfcs/0009-observability-and-mlops.md
