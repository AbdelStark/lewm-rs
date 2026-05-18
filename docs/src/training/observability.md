# Observability and OTLP telemetry

> **Motivation.** A training run that is not observed cannot be
> debugged. lewm-rs emits per-step metrics in two channels: structured
> JSONL on stdout, and (optionally) OpenTelemetry Protocol traces.
>
> **Position.** Ninth sub-page in [Part III](./pipeline.md).
>
> **What you should leave with.** What is logged, how to enable OTLP,
> and how to read the resulting dashboards.

## 1. Two channels

| Channel | Default | When |
|---------|---------|------|
| **JSONL on stdout** | Always on | Every `log_interval` steps (default 10) and at every state transition. |
| **OTLP traces** | Off unless `OTEL_EXPORTER_OTLP_ENDPOINT` is set | Per-step spans + state-transition events when an endpoint is configured. |

The JSONL channel is the source of truth for the post-run analysis
(`reports/`, model card stats). The OTLP channel is for *live*
dashboards during a run; it is decoupled and the JSONL channel works
without it.

## 2. The JSONL schema

Each line is one JSON object with a stable schema (v1.0.0):

```json
{
  "ts": "2026-05-15T10:09:12.345Z",
  "kind": "step",
  "step": 12500,
  "state": "STEADY",
  "loss": { "total": 6.09e-06, "pred": 1.13e-06, "sigreg": 4.96e-06 },
  "lr": 1.50e-04,
  "grad_norm_pre_clip": 4.97e-03,
  "samples_per_sec": 47.3,
  "rng_substreams": { "master": "<seed>", ... },
  "device": "cuda:0"
}
```

`kind` is one of: `step`, `state_transition`, `parity_result`,
`checkpoint_saved`, `collapse_probe`, `error`.

The trainer writes a single line per event; downstream tools
(`python/plot_curves.py`, the model card upload) read the file
line-by-line.

## 3. The OTLP wiring

When `OTEL_EXPORTER_OTLP_ENDPOINT` is set (e.g., to
`http://localhost:4317`), the trainer initialises an OpenTelemetry
tracer with:

- Service name: `lewm-train`
- Service version: the workspace's git SHA
- Resource attributes: `host.name`, `cloud.region`, `device.name`

Spans are emitted per step:

```text
training-run                                       (root span, whole run)
├── state-transition INIT → PARITY_CHECK
├── parity-check                                   (one span; 10 sub-events)
├── state-transition PARITY_CHECK → SMOKE
├── smoke-train                                    (50 sub-spans, one per step)
├── ...
├── training-step (step=12500)                     (one span per logged step)
│    ├── data-prefetch (12.4ms)
│    ├── forward       (18.6ms)
│    ├── backward      (24.1ms)
│    ├── optimizer-step (5.3ms)
│    └── attributes: loss.*, lr, grad_norm
└── ...
```

This drives the local Grafana / Tempo / Loki stack in
[`infra/otel/`](https://github.com/AbdelStark/lewm-rs/blob/main/infra/otel/README.md).

### 3.1 Cost contract

**RFC0009-001 [MUST]** — When `OTEL_EXPORTER_OTLP_ENDPOINT` is unset
(the CI and HF Jobs default), the OTLP exporter is **disabled** and
adds zero overhead. Training does not depend on a working telemetry
endpoint; the JSONL channel is sufficient on its own.

This was tested by running a 50-step smoke with and without
`OTEL_EXPORTER_OTLP_ENDPOINT`: wall-time difference was within
measurement noise.

## 4. The reports pipeline

After a run completes, `python/plot_curves.py` reads
`train_losses.jsonl` and produces:

- Loss curves (total, pred, sigreg) on log-y, linear-x axes.
- LR schedule curve.
- Gradient-norm trace with TOL-011 ceiling overlay.
- Samples/sec histogram (for throughput debugging).

The figures land in `paper/figures/` and feed both the paper writeup
and the model card.

The PushT bounded-core training report (`reports/pusht_training.md`)
and SO-100 training report (`reports/so100_training.md`) are written
by hand from these figures plus the sidecar metadata.

## 5. The `lewm-telemetry` crate

The telemetry layer lives in `crates/lewm-telemetry`:

```rust,ignore
// In lewm-train/src/step.rs:
telemetry.emit_step(StepRecord {
    step,
    state: state_machine.current(),
    loss_total, loss_pred, loss_sigreg,
    lr: scheduler.lr(),
    grad_norm: grad_norm_pre_clip,
    samples_per_sec,
});
```

The `Telemetry` trait has a single OTLP-aware impl and a no-op impl
for tests. The implementation guards against repeat work: if no OTLP
endpoint is configured, the OTLP impl is a thin shim that just writes
JSONL.

## 6. The optional dashboards

The opt-in self-hosted stack in
[`infra/otel/`](https://github.com/AbdelStark/lewm-rs/blob/main/infra/otel/README.md)
provides:

- **Tempo** for traces (per-step spans, latency distributions).
- **Loki** for JSONL logs (queryable via LogQL).
- **Grafana** dashboards that join the two and produce live loss
  curves, throughput, and grad-norm panels.

For a quick local check: `python3 scripts/otel_smoke.py` runs a minimal
collector and verifies the exporter can connect.

## 7. Source pointers

| Topic | Source |
|-------|--------|
| `lewm-telemetry` crate | `crates/lewm-telemetry/src/lib.rs` |
| Step emit | `crates/lewm-train/src/step.rs` |
| Plot script | `python/plot_curves.py` |
| OTLP infra | `infra/otel/` |
| Local smoke | `scripts/otel_smoke.py` |
| Cost-zero contract test | `crates/lewm-telemetry/tests/no_endpoint_no_cost.rs` |

[RFC 0009]: ../reference/rfcs.md
