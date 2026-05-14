# Self-hosted OpenTelemetry

This stack is the optional local telemetry backend for real training runs. It
runs an OpenTelemetry Collector, Tempo, Prometheus, and Grafana from checked-in
Docker Compose configuration.

Smoke training and CI do not require this stack. Leave
`OTEL_EXPORTER_OTLP_ENDPOINT` unset and the Rust OTLP exporter stays disabled.

## Start

```sh
docker compose -f infra/otel/docker-compose.yml --env-file infra/otel/env.example up -d
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317
```

Grafana is available at <http://127.0.0.1:3000>. The example admin password in
`infra/otel/env.example` is for local development only; override it for any
shared machine.

## Use With A Real Run

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317 \
  lewm-train train --config configs/pusht.toml --data-dir /data/pusht
```

For HF Jobs, pass `OTEL_ENDPOINT=http://<collector-host>:4317` only on runs that
should export traces. Smoke jobs can omit it.

## Stop

```sh
docker compose -f infra/otel/docker-compose.yml down
```

To remove local telemetry state:

```sh
docker compose -f infra/otel/docker-compose.yml down -v
```
