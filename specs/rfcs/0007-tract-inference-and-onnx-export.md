---
rfc: "0007"
title: "lewm-infer — Tract CPU inference, ONNX/NNEF export"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§5.2", "§7", "§9.3", "§11"]
depends_on: ["0001", "0002", "0006"]
related: ["0014", "0015"]
---

# RFC 0007 — `lewm-infer`: Tract CPU inference, ONNX/NNEF export

> **Status:** Accepted · **Version:** 1.0.0
>
> The deployment story is "Rust binary on a laptop CPU, sub-second planning cost." This RFC fixes the export path (ONNX preferred, NNEF fallback, Burn-record-direct loader as last resort), the Tract runner, the inference CEM, and the Gradio Space integration.

---

## 1. Introduction

### 1.1 Motivation

JEPA inference on a CPU is exactly the kind of "verifiable robotics" target the project's cypherpunk-fit rationale (PRD §1) speaks to. To realize it we need:

- A deterministic export from the trained Burn model to a portable tensor graph.
- A pure-Rust runtime that loads and executes the graph on CPU.
- A CEM-style planner that fits in a sub-second wall-clock budget.

### 1.2 Goals

1. Specify the export pipeline from Burn → ONNX (primary) → Tract.
2. Specify the NNEF fallback path triggered if any required op is unsupported by Burn's ONNX exporter.
3. Specify the third-tier fallback: a Burn-record-direct Rust loader hand-rolled in `lewm-infer` if both export paths fail.
4. Specify the `lewm-infer plan` binary semantics.
5. Specify the CPU-side CEM implementation.
6. Specify the bench protocol that proves NFR-013 (≤ 1.0 s laptop, ≤ 0.3 s CPU XL).

### 1.3 Non-goals

- GPU inference in Rust (PRD §2 non-goal).
- Mobile / embedded deployment (revisit in v2).
- Quantization (v2).

---

## 2. Conventions

- "Encoder graph" = the ViT + projector, taking `(B=1, 3, 224, 224) F32` to `(1, D=192) F32`.
- "Predictor graph" = `Embedder + ArPredictor + pred_proj`, taking `(B, H, D) F32` and `(B, H, A) F32` to `(B, H, D) F32`.
- We export **two separate** graphs (encoder and predictor), not a fused JEPA graph. The reason: CEM batches the predictor `n_cand` times but encodes start/goal only once.

---

## 3. Crate layout

```
lewm-infer/
└── src/
    ├── lib.rs
    ├── bin/
    │   └── lewm-infer.rs
    ├── export/
    │   ├── mod.rs
    │   ├── onnx.rs               # ONNX exporter wrapper (calls Burn)
    │   ├── nnef.rs               # NNEF exporter wrapper
    │   └── verifier.rs           # output equivalence vs. Burn forward
    ├── runner/
    │   ├── mod.rs
    │   ├── tract_onnx_runner.rs
    │   ├── tract_nnef_runner.rs
    │   └── traits.rs             # InferenceRunner trait
    ├── plan.rs                   # CEM on CPU
    ├── preprocess.rs              # image preprocess (mirrors lewm-data)
    ├── space/                     # Gradio bridge
    │   ├── mod.rs
    │   ├── server.rs              # HTTP shim that the Python Space calls
    │   └── api.rs
    └── errors.rs
```

**Important:** `lewm-infer` **MUST NOT** depend on `burn-cuda`, `burn-autodiff`, `burn-train`, or `lewm-train` (INV-003). It depends only on `lewm-core` for shape and config types, `tract`, and `tract-onnx` / `tract-nnef`.

---

## 4. Export pipeline

### 4.1 ONNX export from Burn

Burn provides ONNX export via `burn-import`'s `onnx::ToOnnx` trait (or `burn::export::onnx::Exporter` depending on Burn version pinned at `=0.21.0`, per ADR 0003). The recipe:

```rust
let device = burn_ndarray::NdArrayDevice::default();
let jepa: Jepa<Backend> = load_burn_record("step_0014400.mpk", &device)?;
let dummy_pixels = Tensor::random([1, 3, 224, 224], Distribution::Default, &device);
let dummy_history = Tensor::random([16, 3, 192], Distribution::Default, &device);
let dummy_actions = Tensor::random([16, 3, 2], Distribution::Default, &device);

// Encoder graph
burn::export::onnx::export(
    "encoder.onnx",
    &|x: Tensor<_, 4>| jepa.encode(x.unsqueeze::<5>(1)).squeeze::<2>(1),
    (dummy_pixels,),
    OnnxConfig {
        opset_version: 18,
        dynamic_axes: btreemap!{ 0 => "batch" },
        ..Default::default()
    },
)?;

// Predictor graph
burn::export::onnx::export(
    "predictor.onnx",
    &|h: Tensor<_, 3>, a: Tensor<_, 3>| jepa.predict(h, a),
    (dummy_history, dummy_actions),
    OnnxConfig {
        opset_version: 18,
        dynamic_axes: btreemap!{ 0 => "batch", 1 => "history" },
        ..Default::default()
    },
)?;
```

**RFC0007-001 [MUST]** — Opset version is fixed at **18**. Reasoning: Tract 0.22.x covers ≥ 85 % of opset 18; opset 19+ has shaky coverage as of pinning.

**RFC0007-002 [MUST]** — Batch and history axes are dynamic. Output ONNX has named dims so Tract can specialize.

### 4.2 Export verification

After export, we compare:

```
y_burn  = jepa_burn.encode(fixed_input)           # Burn NdArray F32
y_tract = tract_onnx_runner.run("encoder.onnx", fixed_input)
assert L∞(y_burn - y_tract) < 1e-4
```

**RFC0007-003 [MUST]** — Every export verifies with the L∞ tolerance above. Failure of verification triggers either:

- **(a)** the NNEF fallback (§5),
- **(b)** the Burn-record-direct fallback (§6),
- **(c)** an explicit op-table update in Tract (file an upstream issue + ADR).

The fallback decision is encoded in `export::strategy::pick_export(jepa)`:

```
1. Try ONNX export. Verify. On success: return Onnx.
2. Try NNEF export. Verify. On success: return Nnef.
3. Return BurnDirect.
```

This ladder runs once per release and the resulting strategy is recorded in the model card.

### 4.3 Known op risk: AdaLN-zero

The AdaLN-zero modulation produces a sequence of ops: `Linear → split into 6 → broadcast multiply, add → residual gate`. All are bread-and-butter ONNX ops, but:

- `split` to 6 chunks along axis 2 maps to ONNX `Split` (opset 13+).
- The gated residual `x = x + g * y` is `Mul + Add`.
- No exotic ops.

**RFC0007-004 [SHOULD]** — Tract should handle this cleanly. The verifier test confirms.

### 4.4 Activation op coverage

- `gelu_tanh_approx` — Tract supports `Gelu` since 0.21; opset 20 added the variant. We work around by lowering to its explicit form (`0.5 * x * (1 + tanh(...))`) at export time when opset is `≤ 19`.
- `gelu_erf` — supported natively.
- `silu` (used in `Embedder` and `AdaLNZero`) — Tract supports as `Sigmoid * x`; lowered explicitly.

**RFC0007-005 [MUST]** — The exporter wraps non-trivial activations in their explicit op forms to avoid Tract op-coverage gaps. This is implemented in `export::lowering`.

---

## 5. NNEF fallback

NNEF is Tract's native format. We use `tract-nnef` to export from the Burn forward via a trace.

**RFC0007-006 [MUST]** — If ONNX export fails verification, the build pipeline automatically attempts NNEF. NNEF is preferred over the Burn-direct path because NNEF graphs can be inspected with `tract-cli dump` for debugging.

**RFC0007-007 [SHOULD]** — In practice, NNEF should succeed if ONNX failed for AdaLN — Tract's NNEF can express the same primitives with more flexible quantizers. This is a hypothesis; the exporter test answers definitively.

---

## 6. Burn-record-direct fallback (last resort)

If both ONNX and NNEF fail, we implement a Rust loader that reads the Burn `.mpk` directly and walks the model graph with hand-coded Tract primitives. This is significant work; we hope to avoid it.

**RFC0007-008 [MUST]** — The fallback is gated behind `--feature burn-direct` in `lewm-infer`. It is **not** the default. The decision to enable it requires an ADR.

**RFC0007-009 [SHOULD]** — If enabled, the fallback implements:

- A `BurnRecordReader` that parses the MPK and produces a parameter-name → tensor map.
- A `ManualGraph` builder that uses `tract-core` primitives to assemble the equivalent computation graph.
- An equivalence test vs. Burn NdArray forward.

The full design will be captured in an ADR if the fallback is ever needed.

---

## 7. Tract runner

### 7.1 Trait

```rust
pub trait InferenceRunner: Send {
    /// Encode one image and return the (D,) embedding.
    fn encode(&mut self, pixels: &[f32; 3 * 224 * 224]) -> Result<Vec<f32>, InferError>;

    /// Run the predictor: history (H, D) + actions (H, A) → next embeddings (H, D).
    fn predict(&mut self, history: &[f32], actions: &[f32], h: usize, a: usize) -> Result<Vec<f32>, InferError>;

    /// Return the runner metadata.
    fn metadata(&self) -> RunnerMetadata;
}
```

Implementations:

- `TractOnnxRunner` — wraps `tract_onnx::onnx().model_for_path(...)`.
- `TractNnefRunner` — wraps `tract_nnef::nnef().model_for_path(...)`.
- `BurnDirectRunner` — feature-gated.

### 7.2 Loading

```rust
pub fn load(checkpoint_dir: &Path) -> Result<Box<dyn InferenceRunner>, InferError> {
    if checkpoint_dir.join("encoder.onnx").exists() {
        TractOnnxRunner::new(checkpoint_dir).map(boxed)
    } else if checkpoint_dir.join("encoder.nnef").exists() {
        TractNnefRunner::new(checkpoint_dir).map(boxed)
    } else if cfg!(feature = "burn-direct") {
        BurnDirectRunner::new(checkpoint_dir).map(boxed)
    } else {
        Err(InferError::NoExportFound)
    }
}
```

### 7.3 Optimization

**RFC0007-010 [MUST]** — On load, runners call `model.into_optimized()?.into_runnable()?` per Tract's recommended pattern. This triggers constant folding, op fusion, and shape inference.

**RFC0007-011 [SHOULD]** — Runners enable Tract's `multithread` mode: `model.set_intra_op_thread_pool(rayon::current_num_threads())`. This is critical for the 1-second laptop target.

---

## 8. CPU CEM

`plan::cem_cpu` mirrors RFC 0006 §4 but with two specializations:

1. **Smaller `n_cand`** — default 16 (vs 1000 on GPU). The latency budget rules out 1000.
2. **No batched rollout** — the CPU runner serializes the `n_cand` rollouts. We rely on Tract's intra-op parallelism, not inter-op batching.

```rust
pub struct CpuCem {
    pub n_iter: usize,        // 5
    pub n_cand: usize,        // 16
    pub n_elite: usize,        // 4
    pub horizon_plan: usize,   // 5
    pub sigma_init: f32,
    pub sigma_min: f32,
}

impl CpuCem {
    pub fn plan<R: InferenceRunner>(
        &self,
        runner: &mut R,
        z_history: &[f32],     // (H * D,)
        z_goal: &[f32],         // (D,)
        rng: &mut ChaCha20Rng,
        action_dim: usize,
    ) -> Result<CpuPlanResult, InferError> { /* … */ }
}
```

**RFC0007-012 [MUST]** — `CpuCem` produces the **same** action sequence as the GPU CEM when both are seeded with the same `rng:cem` state and the same `(z_history, z_goal)`, modulo CPU/GPU float drift (< 1e-3 in the cost).

---

## 9. CLI

```text
lewm-infer <subcommand> [flags]

Subcommands:
  plan        Run a single planning cost computation; report cost and best action sequence.
  bench       Run the latency benchmark.
  serve       Start the HTTP shim used by the Gradio Space.
  verify      Compare ONNX/NNEF outputs to a reference Burn forward.

Global flags:
  --checkpoint-dir <PATH>     Directory containing encoder.{onnx|nnef} and predictor.{onnx|nnef}.
  --action-dim <INT>          2 (PushT) or 6 (SO-100). Inferred from metadata if absent.
  --threads <INT>             Rayon thread count (default = num_cpus).
```

### 9.1 `plan` flags

```text
--start <PATH>          Start image (JPEG/PNG, any size; resized internally).
--goal <PATH>           Goal image.
--horizon <INT>          Default 5.
--n-cand <INT>           Default 16.
--n-iter <INT>           Default 5.
--out <PATH>             JSON output (cost, best actions).
```

### 9.2 `bench` flags

```text
--episodes <INT>         Default 50.
--warmup-runs <INT>      Default 5.
--report <PATH>          JSON report path; also appends to `reports/inference.md`.
```

---

## 10. Gradio Space

The Space lives in the HF Hub repo `abdelstark/lewm-rs-demo`. It is a small Python Gradio app that wraps the Rust binary via subprocess.

### 10.1 Layout

```
space/
├── app.py              # Gradio interface
├── requirements.txt    # gradio + minimal helpers
├── README.md           # space README (markdown + YAML frontmatter)
└── assets/
    ├── pusht_start_example.png
    └── pusht_goal_example.png
```

### 10.2 Behaviour

The Gradio app:

1. Accepts two file uploads: start and goal images.
2. Validates inputs (size, channel count).
3. Invokes the bundled `lewm-infer plan` binary with `--start ... --goal ...`.
4. Parses the JSON output, displays the predicted cost and the action sequence (with a small ASCII visualization).
5. Logs the request, anonymized, to the Space's stdout (for cost-tracking only).

**RFC0007-013 [MUST]** — The Space **MUST** auto-pause after 15 minutes of inactivity to stay within the PRD §7.2 cost target.

**RFC0007-014 [MUST]** — The Space **MUST** include a "About this demo" link to the paper writeup.

### 10.3 Hardware

PRD §7.2 line P13 budgets the Space at T4 small × 50 % uptime. We **also** ship a `cpu-basic` variant (free tier) for graceful degradation; the Space's README states the CPU variant is slower (~ 3 s per plan instead of 1 s).

---

## 11. Testing strategy

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0007-EXPORT-001 | `onnx_export_encoder_round_trip` | integration | RFC0007-001/003 |
| TST-0007-EXPORT-002 | `onnx_export_predictor_round_trip` | integration | RFC0007-001/003 |
| TST-0007-RUN-001 | `tract_onnx_runner_load_and_encode` | integration | §7.1 |
| TST-0007-RUN-002 | `tract_onnx_runner_predict` | integration | §7.1 |
| TST-0007-RUN-003 | `cpu_cem_matches_gpu_seed_to_1e3` | integration | RFC0007-012 |
| TST-0007-BENCH-001 | `bench_laptop_under_1s` | bench (CI-gated) | NFR-013 |
| TST-0007-BENCH-002 | `bench_cpu_xl_under_300ms` | bench | NFR-013 |
| TST-0007-BENCH-COLD-001 | `cold_start_under_3s` | bench | NFR-014 |
| TST-0007-SPACE-001 | `space_app_unit` | python | Gradio app smoke |
| TST-0007-NNEF-001 | `nnef_export_round_trip` | integration | §5 |

---

## 12. Operational considerations

### 12.1 Observability

The inference binary emits metrics to a local JSON log (no OTLP — inference runs at the edge):

```
infer/cold_start_ms
infer/encode_ms
infer/predict_ms_per_call
infer/cem_total_ms
infer/peak_rss_mb
```

### 12.2 Runbook

- **"Export verification fails."** — inspect the failing layer with `tract-cli dump`. File issue to Tract if op coverage missing; fall back to NNEF.
- **"Bench misses 1s target."** — first check thread count (`--threads $(nproc)`); then check that release-lto build was used; then profile with `perf record` and look for SIMD opportunities.
- **"Space request times out."** — Gradio default is 60 s; we explicitly set 30 s, leaving ample headroom. Timeouts indicate a Rust binary deadlock.

### 12.3 Capacity

A `lewm-infer plan` call peaks at:

- ~ 250 MB peak RSS (model weights + Tract tensors).
- 4 CPU cores fully utilized for ~ 1 s on laptop / 0.3 s on CPU XL.

---

## 13. Performance considerations

The performance budget per laptop:

| Stage | Budget | Realistic |
|-------|--------|-----------|
| ONNX load + optimize | 1 s (one-time) | 1.5 s |
| Encoder forward (×2: start + goal) | 200 ms | 250 ms |
| Predictor forward (`n_cand=16, n_iter=5, horizon=5` = 80 calls) | 600 ms | 700 ms |
| Misc (RNG, sort, cost) | 50 ms | 50 ms |
| **Steady-state plan total** | 850 ms | ~1.0 s |

**Cold start** dominated by model load. We do not optimize cold start in v1; it is 2–3 s on laptop.

See [RFC 0014 §6](0014-performance-engineering.md) for the full perf budget and profiling plan.

---

## 14. Security considerations

- The inference binary accepts arbitrary image input from the Gradio Space; we validate `(width, height, channels)` and reject malformed JPEGs (the `image` crate does this automatically).
- No network I/O in inference (the Space's HTTP server is in Python, in a separate process).
- The HTTP shim (`serve` subcommand) is loopback-only by default; binding to a public interface requires `--bind 0.0.0.0` and a warning log.

---

## 15. Alternatives considered

- **A1 — ORT (Microsoft ONNX Runtime) Rust bindings.** Considered. Heavier dep (C++), faster than Tract by ~2x historically. Rejected: pure-Rust deployment is a goal; the latency target is met by Tract.
- **A2 — `candle` for inference.** Considered. Younger ecosystem; we choose `tract` for the production-grade CPU optimizer.
- **A3 — `wonnx` for WebAssembly path.** Out of scope for v1; revisit if Space wants a browser demo.

---

## 16. Acceptance criteria

- [ ] ONNX export of encoder and predictor verifies within 1e-4.
- [ ] Tract runners pass `TST-0007-RUN-001..003`.
- [ ] Bench laptop reports steady-state ≤ 1.0 s (TST-0007-BENCH-001).
- [ ] Space `lewm-rs-demo` is reachable and responds within 30 s.
- [ ] Cold-start ≤ 3 s.

---

## 17. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | Tract op coverage gap for AdaLN | M | H | NNEF fallback; Burn-direct fallback |
| R-2 | Laptop CPU varies wildly across machines | H | M | Two laptop classes benched; report ranges |
| R-3 | Burn ONNX export bug | M | H | NNEF fallback; export verifier catches |
| R-4 | Space cost overrun | M | M | Aggressive auto-pause; cpu-basic fallback |
| R-5 | Threading saturates the system | L | L | `--threads` configurable; default num_cpus - 1 |

---

## 18. Open questions

OQ-2007-1 — Whether to ship a single fused JEPA ONNX graph (one model) vs the two-graph (encoder + predictor) approach. Two-graph wins on CEM reuse (encoder runs once for start, once for goal; predictor batches over `n_cand`). v1 ships two-graph; v2 may evaluate fused.

---

## 19. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0007.*
