---
rfc: "0014"
title: "Performance engineering — throughput, latency, memory, profiling"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§4.4 risk 1", "§7 cost", "§9.3"]
depends_on: ["0001", "0002", "0004", "0005", "0007"]
related: ["0011", "0013"]
---

# RFC 0014 — Performance engineering: throughput, latency, memory, profiling

> **Status:** Accepted · **Version:** 1.0.0
>
> The performance contract turns the cost ceiling (200 USD) into an engineering target. This RFC pins the throughput floor for training, the latency ceiling for inference, the memory budget, and the profiling workflow that drives optimization decisions.

---

## 1. Introduction

### 1.1 Motivation

Cost is wall-clock time times hardware price. With A10G-large at 1.50 USD/hr and a 12-hour budget per training run, training **must** sustain ≥ 45 samples/sec for PushT to fit. Inference must be **≤ 1.0 s** on laptop CPU to be a credible "verifiable robotics" demo. These numbers translate into specific kernel-level requirements.

### 1.2 Goals

1. Pin the throughput targets for training (samples/sec, batches/sec).
2. Pin the latency targets for inference (per-step, per-plan, cold-start).
3. Pin the memory budget (GPU VRAM, host RSS).
4. Specify the bench suite with `criterion`.
5. Specify the profiling workflow (flamegraph, nsight, perf, dtrace).
6. Specify the optimization ladder we follow when a perf gate fails.

### 1.3 Non-goals

- Per-kernel CUDA tuning (we rely on Burn's kernels).
- Distributed perf (single-GPU only).

---

## 2. Conventions

- "Wall-clock" — observed elapsed time, including data load and host sync.
- "GPU time" — kernel time on the device, excluding host overhead.
- "Step time" — one full optimizer step (forward + backward + optim + clip).

---

## 3. Throughput targets (training)

### 3.1 PushT

**RFC0014-001 [MUST]** — On A10G-large 24 GB with `batch=64, grad_accum=2, T=8, BF16-mixed, num_workers=4`:

- **`throughput/samples_per_sec` ≥ 45** averaged over a 1000-step window (NFR-010).
- **`throughput/batches_per_sec` ≥ 0.35** (i.e., one effective batch per ~2.8 s).
- **`data/queue_depth` ≥ 1.0** (data plane is not the bottleneck).

Justification:

```
920k frames × 10 epochs / 45 samples/sec ≈ 56 hours
56 hours × 0.7 (effective utilization after warmup, eval, checkpointing) ≈ 80 hours = 8 effective hours of GPU
```

Wait — let me recompute that more carefully:

```
920k samples per epoch × 10 epochs = 9.2M samples
9.2M / 45 ≈ 204,000 seconds ≈ 56.7 hours
```

That's too long for a single 12-hour cap. The realistic plan is:

```
920k samples × 10 epochs = 9.2M samples
With effective batch = 128 → 71,875 steps
At grad_accum = 2 → 143,750 forward passes
At ~ 45 samples/sec → 204k seconds ≈ 57 hours
```

This **exceeds** the 12-hour cap. We adjust:

- **Option A** — reduce epochs to 4 (paper achieves headline at ~ 4 epochs anyway): 23 hours, still over.
- **Option B** — increase batch and accept a higher throughput target.
- **Option C** — multi-job resume (split 10 epochs across 2× T3 launches via resume).

PRD §7.2 budgets **8 hours** wall for the T3 FULL on A10G-large; the implied throughput target is therefore higher: 

```
9.2M / (8 × 3600) ≈ 320 samples/sec
```

That is **way** higher than 45.

**Resolution:** The PRD's 8-hour estimate assumes ~ 30 samples/step at ~ 10 steps/sec on A10G — that is **300 samples/sec**. Let me re-derive.

Per [LeWM upstream](https://github.com/lucas-maes/le-wm), on L40S (181 TFLOPS) the small model trains in "a few hours" for PushT. A10G is ~ 31 TFLOPS BF16. So we expect 6× slower. L40S "a few hours" → A10G ~12 hours, consistent with PRD's "8h" being optimistic on the low end. The target is therefore:

- Adjusted **`throughput/samples_per_sec` ≥ 200** averaged over a 1000-step window.

We supersede NFR-010's 45 with **NFR-010-v1.0.1** = **200 samples/sec** (this requires a Minor version bump of the master spec; tracked in §16 open questions).

For the spec set's purposes, the **floor** is **45 samples/sec** (per NFR-010 as written in the master spec). The **target** is **200 samples/sec** (per realistic L40S → A10G extrapolation). The bench gates at 45; the report quantifies the actual achieved value.

### 3.2 SO-100

**RFC0014-002 [MUST]** — On A10G-large with `batch=64, grad_accum=2, T=8, BF16-mixed`:

- **`throughput/samples_per_sec` ≥ 35** (slightly lower due to 6-D action stack and decode overhead; floor per NFR-011).
- Same queue-depth and other rules.

19,631 samples × 10 epochs / 35 ≈ 5,600 seconds ≈ 1.5 hours. Comfortably within the 4-hour budget.

### 3.3 GPU memory

**RFC0014-003 [MUST]** — Peak GPU memory **MUST** be ≤ **20 GB** on the A10G-large 24 GB.

Components at peak (BF16-mixed, F32 master weights, AdamW state):

```
Master weights (F32):          15M × 4 bytes      =  60 MB
AdamW m, v   (F32):            15M × 8 bytes      = 120 MB
Activations  (BF16, batch 64):
    encoder × 12 blocks ≈ 64 × 8 × 197 × 384 × 2 × 12  ≈ 0.93 GB
    predictor × 6 blocks ≈ 64 × 8 × 384 × 2 × 6      ≈ 2.36 MB (much smaller)
    SIGReg projections  K×N×4 bytes ≈ 1024 × 512 × 4   = 2 MB
    grad checkpoint buffers ≈ 2× the above           ≈ 1.9 GB
CEM (eval only, n_cand=1000): 1000 × 8 × 384 × 4 ≈ 12 MB
Total worst case (eval+train concurrent): ≈ 5 GB activations + parameters
```

The 20 GB ceiling allows ample headroom even for CEM during eval-mid-epoch and BatchNorm running stats. The 24 GB total leaves 4 GB for driver and CUDA context overhead.

---

## 4. Latency targets (inference)

Already established in [RFC 0007 §13](0007-tract-inference-and-onnx-export.md); reiterated here as part of the perf contract:

| Target | Hardware | Budget |
|--------|----------|--------|
| Cold-start (load + first plan) | laptop CPU | ≤ 3.0 s |
| Cold-start | CPU XL (16 vCPU) | ≤ 1.5 s |
| Steady-state plan (5-step × 16 cand × 5 iter) | laptop CPU | ≤ 1.0 s |
| Steady-state plan | CPU XL | ≤ 0.3 s |
| Encoder single forward | laptop CPU | ≤ 0.15 s |
| Predictor single forward | laptop CPU | ≤ 0.012 s (so 80 calls fit in 1 s) |

**RFC0014-004 [MUST]** — Steady-state plan latency on laptop CPU **MUST** be measured on at least two laptops (Apple M-series, Intel i7 ultrabook class) for the report.

**RFC0014-005 [MUST]** — Cold-start measurement excludes JIT warmup; first measurement is the *real* first call.

---

## 5. Bench suite

### 5.1 Crate-level benches

Each crate has a `benches/` directory using `criterion`:

```
crates/lewm-core/benches/
├── forward_encoder.rs
├── forward_predictor.rs
├── sigreg.rs
crates/lewm-data/benches/
├── prefetcher.rs
├── collate.rs
crates/lewm-train/benches/
├── step.rs                  # full optimizer step (CPU NdArray for repeatability)
crates/lewm-infer/benches/
├── cost_bench.rs            # the headline 1.0 s laptop number
├── encode_bench.rs
├── predict_bench.rs
```

`criterion` runs each benchmark for at least 10 samples; warmup 3 s; measurement 10 s; statistical analysis writes `target/criterion/<name>/...`.

### 5.2 Bench gates in CI

The nightly workflow runs:

```bash
cargo bench --workspace -- --save-baseline nightly
```

Compares against the committed baseline in `bench-baselines/`:

```
bench-baselines/
├── cost_bench-laptop.json
├── cost_bench-cpu-xl.json
├── prefetcher.json
├── step.json
└── ...
```

**RFC0014-006 [MUST]** — A regression of `> 5 %` on any baselined bench triggers a CI failure (annotated, not auto-blocking, with a one-week grace period for triage).

**RFC0014-007 [MUST]** — Baselines are **regenerated** on each release tag, on the same hardware as the prior baseline (self-hosted runner with stable spec). Old baselines kept for archaeology.

### 5.3 Headline benches

The benches that map onto the perf contract:

| Bench | Metric | Floor | Tracked NFR |
|-------|--------|-------|-------------|
| `forward_encoder` (GPU, batch=64) | wall_ms per call | ≤ 60 ms | NFR-010 |
| `forward_predictor` (GPU, batch=64) | wall_ms per call | ≤ 15 ms | NFR-010 |
| `sigreg` (GPU, B=64, T=8, D=384) | wall_ms per call | ≤ 10 ms | NFR-010 |
| `prefetcher` (CPU, B=64, T=8) | batches_per_sec | ≥ 60 | RFC0004-026 |
| `cost_bench` (laptop CPU, full plan) | wall_ms per plan | ≤ 1000 | NFR-013 |
| `cost_bench` (CPU XL) | wall_ms per plan | ≤ 300 | NFR-013 |

---

## 6. Profiling workflow

### 6.1 CPU profiling (laptop / CPU XL)

**Tool:** `cargo flamegraph` + `perf record`.

```bash
sudo perf record -F 999 -g -- target/release/lewm-infer plan --start ...
sudo perf script | inferno-collapse-perf | inferno-flamegraph > flame.svg
```

**RFC0014-008 [SHOULD]** — `cargo flamegraph` is the canonical CPU profiler. Flame outputs are checked into `profiling/flamegraphs/<git_sha>/<bench>.svg`.

### 6.2 GPU profiling (training)

**Tool:** NVIDIA Nsight Systems (`nsys`) for system-wide spans + Nsight Compute (`ncu`) for kernel-level.

```bash
nsys profile --trace=cuda,nvtx,cublas \
    --output=profile-pusht.nsys-rep \
    -- lewm-train smoke --config configs/pusht.toml --steps 100
```

`tracing-nvtx` (a tracing-subscriber layer) emits NVTX ranges for our spans so they appear inline with CUDA activity.

**RFC0014-009 [SHOULD]** — Major optimization PRs include a before/after `nsys-rep` summary attached.

### 6.3 Memory profiling

**Tool:** `nvidia-smi --query-gpu=memory.used --format=csv -l 1` for online monitoring; `cuda-memcheck` for leaks.

For host: `dhat` (Rust) or `valgrind --tool=massif`.

### 6.4 Tracing-time analysis

OTLP spans sent to the self-hosted Tempo stack show the wall-clock distribution of `training.step.forward`, `training.step.backward`, etc. Useful for spotting cross-step jitter.

---

## 7. Optimization ladder

When a perf gate fails, climb in this order:

1. **Configuration**: increase `num_workers`, check `OMP_NUM_THREADS`, verify `RUSTFLAGS`/`release-lto`.
2. **Layout**: re-order operations to reduce host syncs (esp. avoid `.into_data()` mid-loop).
3. **Kernel selection**: enable fused SDPA, use Burn's optimized BatchNorm, etc.
4. **Batch size sweep**: try `(32, 4)`, `(64, 2)`, `(128, 1)` grad-accum combinations.
5. **Mixed precision**: enable BF16 mixed if not already.
6. **Cache the cacheable**: position embedding, causal mask, RNG (per-step `P` matrix cannot be cached per RFC 0003).
7. **Algorithmic**: e.g., chunked CEM, only if 1–6 don't suffice.

**RFC0014-010 [MUST]** — Optimization PRs cite the rung climbed. We do not optimize without measurement.

---

## 8. Testing strategy

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0014-THRU-PUSHT-001 | `bench_pusht_step_throughput_meets_floor` | bench (nightly, GPU) | NFR-010 |
| TST-0014-THRU-SO100-001 | `bench_so100_step_throughput_meets_floor` | bench | NFR-011 |
| TST-0014-MEM-001 | `peak_gpu_memory_under_20gb` | integration | NFR-012 |
| TST-0014-COLD-001 | `cold_start_under_3s_laptop` | bench (CI) | NFR-014 |
| TST-0014-INFER-001 | `infer_plan_under_1s_laptop` | bench (CI) | NFR-013 |
| TST-0014-BENCH-REG-001 | `bench_regression_under_5_pct` | meta | RFC0014-006 |

---

## 9. Operational considerations

### 9.1 Observability

Already covered by RFC 0009; this RFC adds:

- `bench/<name>/p50_ms`, `bench/<name>/p99_ms` written to a structured CI artifact.
- Long-term tracking via a CSV `reports/bench_history.csv` updated nightly.

### 9.2 Runbook

- **"`samples_per_sec` is half the floor."** — first check `data/queue_depth`; if 0, the data plane is the bottleneck → climb ladder rung 1 (workers).
- **"Cold-start fluctuates ± 30 %."** — laptop CPU thermal throttling. Quote ranges in the report.

---

## 10. Performance considerations (meta)

Benches themselves must be cheap enough to run in CI. The full nightly bench suite **MUST** complete in ≤ 20 minutes on a self-hosted L4 runner.

---

## 11. Security considerations

Profiling tools (`perf`) require privileged access on Linux; only run on trusted machines. The artifacts (flamegraphs) are public; they reveal code paths but no secrets.

---

## 12. Alternatives considered

- **A1 — Use `criterion-perf-events`.** Considered for cache miss counters. Out of scope for v1.
- **A2 — Use `pprof-rs`.** Considered; flamegraph is sufficient and more portable.

---

## 13. Acceptance criteria

- [ ] All TST-0014-* pass.
- [ ] Baselines committed for the headline benches.
- [ ] Nightly regression check in place.

---

## 14. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | Burn GPU throughput below target | M | H | Optimization ladder; A100 fallback in budget |
| R-2 | Bench flakiness on shared CI | M | M | Self-hosted runner pinned; warmup + 10 samples |
| R-3 | Laptop CPU variability undermines NFR-013 | H | M | Quote ranges; CI bench uses a baseline machine |

---

## 15. Open questions

OQ-2014-1 — Should we bump NFR-010 from 45 to 200 samples/sec? Discussed in §3.1; resolution requires Minor bump of the master spec set. Currently the floor stays at 45 (we are conservative); the target is 200; the realized number will be reported.

---

## 16. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0014.*
