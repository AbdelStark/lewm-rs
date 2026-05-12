---
rfc: "0009"
title: "lewm-telemetry — observability, metrics, traces, logs, MLOps"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§6 Observability and MLOps"]
depends_on: ["0001", "0003"]
related: ["0005", "0010", "0011", "0017"]
---

# RFC 0009 — `lewm-telemetry`: observability, metrics, traces, logs, MLOps

> **Status:** Accepted · **Version:** 1.0.0
>
> Production-grade ML training is "nothing about a run depends on a human watching it." Every metric is named, every trace is recorded, every log is structured. This RFC pins the metric and span names, the dashboards, the collapse-detection subsystem, and the export targets.

---

## 1. Introduction

### 1.1 Motivation

When an 8-hour A10G run silently underperforms because the GPU was 50 % idle on data, you need a span timeline, not a scalar curve. When the encoder collapses to a constant, you need a per-step variance probe, not next-day analysis. The observability layer is the system's nervous system; this RFC makes it explicit.

### 1.2 Goals

1. Define `lewm-telemetry` as the single facade for metrics, traces, and logs across the workspace.
2. Pin the metric **name registry** — every metric used anywhere in `lewm-rs` is listed here.
3. Pin the span **name registry**.
4. Specify Trackio, Tensorboard, and OTLP exporters with their config knobs.
5. Specify the collapse-detection subsystem.
6. Specify the dashboards delivered as artifacts.

### 1.3 Non-goals

- Profiling tooling (`flamegraph`, `perf`) — covered by [RFC 0014](0014-performance-engineering.md).
- Cost ledger details — covered by [RFC 0010 §6](0010-huggingface-hub-integration.md).

---

## 2. Conventions

- **Metric** — a named scalar (or scalar over time) reported each step or each epoch.
- **Span** — a named time interval; spans nest. Attached attributes describe the run.
- **Log line** — a structured JSON record at one timestamp.
- **Exporter** — a sink for metrics, spans, or logs.

Metric names are dot-separated with the namespace as the first segment: `loss/total`, `optim/lr`, `data/queue_depth`. The full registry lives in §5.

---

## 3. Crate layout

```
lewm-telemetry/
└── src/
    ├── lib.rs
    ├── facade.rs              # `Telemetry` struct: the single public entry point
    ├── tracker.rs              # Trackio bridge (writes to local Trackio dir; uploaded by sidecar)
    ├── tensorboard.rs          # Tensorboard event writer (pure Rust)
    ├── otlp.rs                 # OpenTelemetry OTLP exporter
    ├── logs.rs                 # tracing-subscriber bridge (structured JSON)
    ├── metrics.rs              # MetricRegistry, MetricName, MetricKind
    ├── spans.rs                # Span name constants
    ├── collapse.rs             # Collapse detector subsystem
    ├── system.rs               # GPU mem, util, CPU util, RSS samplers
    └── errors.rs
```

---

## 4. Public facade

```rust
pub struct Telemetry {
    metric_writer: MetricWriter,        // multiplexed Trackio + TB
    tracer:        Tracer,               // tracing-opentelemetry handle
    json_logger:   StructuredLogger,
    collapse:      CollapseSubsystem,
    system:        SystemSampler,
}

impl Telemetry {
    pub fn init(config: TelemetryConfig) -> Result<Self, TelemetryError> { /* … */ }

    pub fn emit_scalar(&self, name: MetricName, step: u64, value: f32);
    pub fn emit_scalars(&self, step: u64, batch: &[(MetricName, f32)]);
    pub fn emit_histogram(&self, name: MetricName, step: u64, values: &[f32]);
    pub fn start_span(&self, name: SpanName) -> SpanGuard<'_>;
    pub fn shutdown(self) -> Result<(), TelemetryError>;
}
```

**RFC0009-001 [MUST]** — `Telemetry::init` returns a `Result`; failure to initialize the exporter is fatal to the trainer. (Better to halt than to silently lose metrics.)

**RFC0009-002 [MUST]** — `Telemetry::shutdown` flushes all exporters and is called explicitly from the trainer's `Drop` path. Tests verify no metric loss on graceful shutdown.

**RFC0009-003 [MUST]** — Emitting an unregistered metric name **MUST** return `TelemetryError::UnknownMetric`. The registry (§5) is the closed set.

---

## 5. Metric registry

The registry is implemented as a `MetricName` enum with one variant per metric, plus methods that return the stable name string. CI rejects PRs that introduce a metric variant without adding a registry row here.

### 5.1 Loss metrics

| Name | Kind | When | Source |
|------|------|------|--------|
| `loss/total` | scalar | every step | `Jepa::criterion` |
| `loss/pred` | scalar | every step | `prediction_loss` |
| `loss/sigreg` | scalar | every step | `sigreg.forward` |
| `loss/sigreg_per_proj_min` | scalar | every step | aggregate over K |
| `loss/sigreg_per_proj_max` | scalar | every step | aggregate over K |

### 5.2 Optimizer metrics

| Name | Kind | When | Source |
|------|------|------|--------|
| `optim/lr` | scalar | every step | scheduler |
| `optim/grad_norm_pre` | scalar | every step | pre-clip global norm |
| `optim/grad_norm_post` | scalar | every step | post-clip |
| `optim/effective_step_norm` | scalar | every step | lr × grad_norm_post |
| `optim/momentum_norm` | scalar | every 100 steps | AdamW state |
| `optim/exp_avg_sq_norm` | scalar | every 100 steps | AdamW state |
| `optim/skipped_steps_total` | scalar | every 100 steps | NaN-skip counter |

### 5.3 Model probes

| Name | Kind | When | Source |
|------|------|------|--------|
| `model/encoder_cls_var` | scalar | every 100 steps | collapse probe |
| `model/encoder_cls_mean_abs` | scalar | every 100 steps | collapse probe |
| `model/cls_cosine_pair_mean` | scalar | every 100 steps | collapse probe |
| `model/predictor_output_var` | scalar | every 100 steps | predictor probe |

### 5.4 Throughput metrics

| Name | Kind | When | Source |
|------|------|------|--------|
| `throughput/samples_per_sec` | scalar | every 10 steps | rolling window |
| `throughput/tokens_per_sec` | scalar | every 10 steps | samples × seq_len |
| `throughput/batches_per_sec` | scalar | every 10 steps | step counter / wall |

### 5.5 Data plane metrics

| Name | Kind | When | Source |
|------|------|------|--------|
| `data/throughput_samples_per_sec` | scalar | every 10 steps | prefetcher |
| `data/throughput_bytes_per_sec` | scalar | every 10 steps | prefetcher |
| `data/queue_depth` | scalar | every step | channel |
| `data/io_wait_ms_p50` | scalar | every 100 steps | worker timing |
| `data/io_wait_ms_p99` | scalar | every 100 steps | worker timing |
| `data/error_count{kind}` | scalar | on event | error sites |

### 5.6 System metrics

| Name | Kind | When | Source |
|------|------|------|--------|
| `system/gpu_mem_used_gb` | scalar | every 30 s | nvml |
| `system/gpu_util_pct` | scalar | every 30 s | nvml |
| `system/cpu_util_pct` | scalar | every 30 s | sysinfo |
| `system/host_rss_gb` | scalar | every 30 s | sysinfo |
| `system/disk_used_gb` | scalar | every 5 min | df-equivalent |

### 5.7 State machine metrics

| Name | Kind | When | Source |
|------|------|------|--------|
| `state/<NAME>/wall_seconds` | scalar | on state exit | state machine |
| `state/transitions_total` | scalar | on every transition | counter |

### 5.8 Eval metrics

Per RFC 0006 §9.1: `eval/episode_success`, `eval/episode_steps`, `eval/episode_final_cost`, `eval/cem_iter_cost_min`, `eval/success_rate`, `eval/latent_mse_mean`, `eval/spearman_mean`, `eval/warm_start_delta`.

### 5.9 Checkpoint metrics

| Name | Kind | When | Source |
|------|------|------|--------|
| `checkpoint/written_count` | scalar | per save | trainer |
| `checkpoint/disk_usage_gb` | scalar | per save | trainer |
| `checkpoint/save_wall_ms` | scalar | per save | timer |

### 5.10 SIGReg sub-stream

`loss/sigreg_per_proj_*` already listed. Plus, on demand (`--debug-sigreg`):

| Name | Kind | When | Source |
|------|------|------|--------|
| `sigreg/cos_max` | scalar | every 100 steps | empirical CF max |
| `sigreg/sin_max` | scalar | every 100 steps | empirical CF max |

---

## 6. Span registry

```
training.run                          # spans the whole trainer process
  training.epoch[i]                    # one per epoch
    training.step[N]                    # one per optimizer step (sampled at 1/100)
      training.forward
      training.backward
      training.optim_step
      training.checkpoint_save
      training.parity_probe
    training.collapse_probe
    training.eval
      eval.episode[k]
        eval.cem_iter[j]
          eval.cem_cost_eval
        eval.rpc_step

data.dataset_open
data.get_window
data.collate
data.prefetch_worker.lifetime
```

**RFC0009-004 [MUST]** — Span names match the registry exactly. Adding a new span requires updating this section.

**RFC0009-005 [MUST]** — Every span carries the attributes `run_id`, `phase`, `git_short_sha`. Step-level spans additionally carry `step`, `epoch`.

---

## 7. Exporters

### 7.1 Trackio

Trackio is HF's experiment tracker; we use its Python SDK via the post-run sidecar. The Rust trainer **writes Trackio's local format** (a `runs/<run_id>/metrics.jsonl` directory) which the sidecar `python/upload_trackio.py` then uploads.

```rust
pub struct TrackioWriter {
    run_dir: PathBuf,
    sink:    std::fs::File,    // metrics.jsonl
}

impl TrackioWriter {
    fn append(&mut self, name: &str, step: u64, value: f32) -> Result<(), TelemetryError> {
        let line = serde_json::json!({"name": name, "step": step, "value": value});
        writeln!(self.sink, "{}", line)?;
        Ok(())
    }
}
```

**RFC0009-006 [MUST]** — Trackio writer file is opened with `O_APPEND` for crash safety.

### 7.2 Tensorboard

Pure-Rust `tensorboard::EventWriter`. Writes `events.out.tfevents.<timestamp>.<run_id>` to `tb/`. Format is the protobuf-encoded `Summary` per the Tensorboard spec.

**RFC0009-007 [MUST]** — Tensorboard files are buffered with `BufWriter` and flushed every 1000 records or 5 s, whichever first.

### 7.3 OTLP

```rust
pub fn init_tracer(endpoint: &str, run_id: &str) -> Tracer {
    let exporter = opentelemetry_otlp::new_exporter().tonic().with_endpoint(endpoint);
    let provider = opentelemetry_sdk::trace::TracerProvider::builder()
        .with_batch_exporter(exporter.build_span_exporter()?, opentelemetry_sdk::runtime::Tokio)
        .with_config(opentelemetry_sdk::trace::Config::default()
            .with_resource(opentelemetry_sdk::Resource::new(vec![
                opentelemetry::KeyValue::new("service.name", "lewm-rs"),
                opentelemetry::KeyValue::new("run.id", run_id.to_string()),
            ])))
        .build();
    provider.tracer("lewm")
}
```

**RFC0009-008 [MUST]** — OTLP endpoint configured via `OTEL_EXPORTER_OTLP_ENDPOINT` env var. If absent, the OTLP exporter is **silently disabled** (with a warning) — Trackio and TB remain active.

**RFC0009-009 [MUST]** — Span batching is enabled; max batch 512 spans; flush interval 5 s.

### 7.4 Structured logs

```
fn init_logging() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer().json())
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .init();
}
```

Every log line is JSON with at least:

```
{
  "timestamp": "2026-05-12T14:30:02.345Z",
  "level": "INFO",
  "target": "lewm_train::trainer",
  "message": "Starting epoch 5",
  "fields": {"run_id": "...", "step": 14400, "wall_time_ms": 1234.5}
}
```

**RFC0009-010 [MUST]** — Log level is filterable by env var (`RUST_LOG`) per `EnvFilter` semantics.

**RFC0009-011 [MUST]** — Secrets (HF tokens, etc.) **MUST NOT** appear in log lines. The `tracing` layer in `logs.rs` includes a redactor that masks fields whose name matches `^(.*(token|secret|key|password).*)$` (case-insensitive).

---

## 8. Collapse detection

### 8.1 Recipe

Every `eval_every_n_steps` (default `100`):

1. Sample a held-out 32-frame batch (deterministic; the same batch is used across the entire run).
2. Encode with `model.encoder(batch).cls()` → `(32, D)` matrix.
3. Compute:
   - `mean_abs_cls = mean over batch and dim of abs(cls)` — scalar.
   - `cls_variance_per_dim_mean = mean over dim of var(cls, dim=batch)` — scalar.
   - `mean_pairwise_cosine = mean over (i, j) pairs of cos(cls[i], cls[j])` — scalar.
4. Check thresholds (TOL-007/008/009).
5. If any threshold trips, increment a counter; if counter reaches 3, write the artifact and emit CRITICAL log line.

### 8.2 Held-out batch

A fixed 32-frame batch lives at `tests/fixtures/collapse_probe.npz`. It is drawn from the **eval split** to keep the train split untouched, and is the same batch across all runs of the project (so trends across runs are comparable).

### 8.3 Artifact

```json
// collapse_suspected_0014500.json
{
  "schema_version": "1.0",
  "run_id": "20260512-143002-9f3a-abcd",
  "step": 14500,
  "epoch": 5,
  "probes": {
    "mean_abs_cls": 5.42,
    "cls_variance_per_dim_mean": 0.043,
    "mean_pairwise_cosine": 0.87
  },
  "thresholds": {
    "mean_abs_cls_ceiling": 5.0,
    "cls_variance_per_dim_floor": 0.05,
    "mean_pairwise_cosine_ceiling": 0.85
  },
  "trips_in_a_row": 3,
  "wall_time": "2026-05-12T15:14:02Z"
}
```

The artifact is uploaded alongside the run's other artifacts on UPLOAD.

---

## 9. Dashboard contract

The Trackio dashboard for a `lewm-rs` run is laid out as follows:

```
┌── Overview ──────────────────────────────────────────────────────┐
│  ▸ loss/total, loss/pred, loss/sigreg (overlay)                  │
│  ▸ optim/lr                                                       │
│  ▸ throughput/samples_per_sec                                     │
│  ▸ data/queue_depth                                                │
└──────────────────────────────────────────────────────────────────┘
┌── Optimization ──────────────────────────────────────────────────┐
│  ▸ optim/grad_norm_pre vs grad_norm_post                          │
│  ▸ optim/effective_step_norm                                       │
│  ▸ optim/skipped_steps_total                                       │
└──────────────────────────────────────────────────────────────────┘
┌── Health probes ─────────────────────────────────────────────────┐
│  ▸ model/encoder_cls_var (with TOL-007 line)                       │
│  ▸ model/encoder_cls_mean_abs (with TOL-008 line)                  │
│  ▸ model/cls_cosine_pair_mean (with TOL-009 line)                  │
└──────────────────────────────────────────────────────────────────┘
┌── System ────────────────────────────────────────────────────────┐
│  ▸ system/gpu_util_pct                                              │
│  ▸ system/gpu_mem_used_gb                                            │
│  ▸ system/host_rss_gb                                                │
└──────────────────────────────────────────────────────────────────┘
┌── Eval ──────────────────────────────────────────────────────────┐
│  ▸ eval/success_rate (PushT) or eval/spearman_mean (SO-100)       │
└──────────────────────────────────────────────────────────────────┘
```

Dashboard config lives at `tracking/dashboards/lewm-rs.json` and is uploaded with the run.

---

## 10. Testing strategy

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0009-TRACK-001 | `trackio_writer_appends_jsonl` | unit | RFC0009-006 |
| TST-0009-TB-001 | `tensorboard_writer_event_format` | unit | RFC0009-007 |
| TST-0009-OTLP-001 | `otlp_export_smoke_with_local_collector` | integration | RFC0009-008 |
| TST-0009-COL-001 | `collapse_detector_three_in_row_trips` | unit | §8 |
| TST-0009-COL-002 | `collapse_detector_no_false_positive` | unit | §8 |
| TST-0009-LOG-001 | `redactor_masks_token_field` | unit | RFC0009-011 |
| TST-0009-REG-001 | `metric_registry_no_dup_no_typo` | unit | §5 |
| TST-0009-SHUTDOWN-001 | `shutdown_flushes_all_exporters` | integration | RFC0009-002 |

Fixtures:

- A local OTLP collector spun up in CI via `otel-collector-contrib` Docker.
- A synthetic collapsed-encoder fixture for `TST-0009-COL-001`.

---

## 11. Operational considerations

### 11.1 Runbook

- **"Trackio dashboard shows no metrics."** — verify the sidecar `python/upload_trackio.py` ran post-step. Often a missing HF token.
- **"OTLP collector receives zero spans."** — verify `OTEL_EXPORTER_OTLP_ENDPOINT` env var; check the collector's auth.
- **"Logs flooded with `data/error_count` increments."** — a malformed shard; see RFC 0004 §13.2.

### 11.2 Capacity

- Trackio JSONL ~ 10 MB per epoch (every step × N metrics).
- Tensorboard events ~ 5 MB per epoch.
- OTLP spans ~ 50 MB per epoch (sampled).
- Total per 10-epoch run: ~ 650 MB observability artifacts.

These are uploaded to a separate "observability" prefix in the run repo for cleanliness.

---

## 12. Performance considerations

- Metric emission is non-blocking: a `crossbeam::channel` between the trainer and the exporter threads.
- Channel saturation drops the oldest *system* metrics first; loss/optim/data metrics are never dropped (channel `Unbounded` for those).
- Trace sampling: step-level spans are sampled at 1/100; epoch spans always recorded; eval spans always recorded.

---

## 13. Security considerations

- See RFC0009-011 redactor.
- OTLP endpoint creds in env var only.

---

## 14. Alternatives considered

- **A1 — Native Trackio Rust SDK.** No such SDK exists at the pinned date. We use the local-file + Python-sidecar pattern.
- **A2 — Stack Driver / Datadog.** Out of scope; HF-native Trackio + OTLP-to-Honeycomb is sufficient.
- **A3 — Skip Tensorboard.** Rejected: PRD §6.1 requires the portability backstop.

---

## 15. Acceptance criteria

- [ ] All metrics in §5 emitted at the documented frequency.
- [ ] All spans in §6 produced; sampled appropriately.
- [ ] Trackio dashboard imports successfully on a fresh Space.
- [ ] Collapse detector produces a synthetic-collapse artifact in the unit test.
- [ ] `shutdown` test confirms no metric loss.

---

## 16. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | Trackio API breaks | M | M | We write local format; sidecar adapts |
| R-2 | OTLP collector unreliable | M | L | Exporter degrades silently; logs say so |
| R-3 | Tensorboard file format drift | L | L | We pin protobuf schema in `tensorboard.rs` |
| R-4 | Metric explosion (cardinality) | L | M | Closed registry; CI rejects new metrics without RFC update |

---

## 17. Open questions

OQ-2009-1 — Should we mirror Trackio metrics to W&B? Not required; could be added behind a feature flag.

---

## 18. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0009.*
