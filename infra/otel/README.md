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

Collector health is available at <http://127.0.0.1:13133>. Collector internal
metrics are available at <http://127.0.0.1:8888/metrics>; Prometheus scrapes
that endpoint for span ingest/export counters.

## Smoke

```sh
python3 scripts/otel_smoke.py
```

The smoke script starts the local stack, runs an ignored Rust integration test
that emits one `training.step` span through `OTEL_EXPORTER_OTLP_ENDPOINT`, and
polls the collector internal metrics until accepted/exported span counters
increase. Use `--down-after` when the stack should be stopped after the check.

No public Trackio dashboard URL is documented here yet. Add that URL only after
the tracking Space exists.

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
