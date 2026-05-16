# Tract CPU runner

> **Motivation.** Tract is the pure-Rust ONNX runtime that powers the
> "single binary, CPU only, no Python" deployment story. This page
> documents how `lewm-infer` loads and drives the Tract-compat ONNX
> graphs.
>
> **Position.** Sub-page of [Part V](./onnx-export.md).
>
> **What you should leave with.** The Tract version, the runner's
> interface, and the bench protocol.

## 1. Why Tract

[Tract](https://github.com/sonos/tract) is a pure-Rust ONNX/NNEF
inference runtime by Sonos. We choose it for `lewm-rs` because:

- **Pure Rust.** No Python, no C++ runtime, no shared library. The
  built `lewm-infer` binary is statically self-contained.
- **CPU-first.** Tract is optimised for CPU inference on commodity
  hardware (x86_64 with AVX2/AVX-512, ARM with NEON). Apple M-series
  is well-supported.
- **Stable API.** Tract 0.22.1 is the pinned version. The API has
  been stable since 0.20.

## 2. The runner

`crates/lewm-infer/src/runner/tract_onnx_runner.rs`:

```rust,ignore
pub struct TractOnnxRunner {
    encoder: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    predictor: SimplePlan<...>,
    action_dim: usize,
    history_steps: usize,
}

impl TractOnnxRunner {
    pub fn load(checkpoint_dir: &Path) -> Result<Self, InferError> { /* … */ }
    pub fn encode(&self, pixels: &[f32]) -> Result<Vec<f32>, InferError> { /* … */ }
    pub fn predict(&self, history: &[f32], actions: &[f32]) -> Result<Vec<f32>, InferError> { /* … */ }
}
```

The runner exposes the same conceptual interface as `Jepa<B>` but in
plain `&[f32]` slices instead of typed tensors. This is the right
abstraction at the deployment boundary.

## 3. Loading

```rust,ignore
let encoder_onnx = checkpoint_dir.join("encoder.onnx");
let model = tract_onnx::onnx().model_for_path(&encoder_onnx)?;
let model = model
    .with_input_fact(0, f32::fact([1, 3, 224, 224]).into())?
    .into_optimized()?;
let encoder = SimplePlan::new(model)?;
```

Three steps:
1. **Parse**: read the ONNX graph.
2. **Fact-constrain**: tell Tract the exact shape and dtype of each
   input. This is the step that requires fixed batch axis — Tract uses
   the constraints to specialise the graph.
3. **Optimise + plan**: Tract performs constant folding, kernel fusion,
   and other AOT optimisations, producing a `SimplePlan` that owns the
   ready-to-run graph.

The optimisation pass is the most expensive part of loading
(~200–400 ms on Apple M-series). After loading, individual `encode` /
`predict` calls are fast.

## 4. The CEM driver

`crates/lewm-infer/src/plan.rs` implements the CEM loop against the
Tract runner. The hot loop is:

```rust,ignore
for iter in 0..n_iter {
    // 1. Sample candidates
    let candidates = sample_candidates(mu, sigma, n_cand, horizon, action_dim, &mut rng);

    // 2. Score (predictor rollout per candidate)
    let mut costs = Vec::with_capacity(n_cand);
    for cand_idx in 0..n_cand {
        let cost = rollout_and_score(&self.runner, &z_history, &z_goal, &candidates[cand_idx]);
        costs.push(cost);
    }

    // 3..5. Pick elites, update mu/sigma, track best
    ...
}
```

In the current implementation, the rollout loop over candidates is
**serial** (one predictor call per candidate per horizon step).
Parallelising over candidates is a future optimisation; the present
4.08 s/episode benchmark is with the serial implementation.

## 5. The bench protocol

```sh
lewm-infer bench \
    --checkpoint-dir abdelstark/lewm-rs-pusht/tract-compat/ \
    --history-steps 3 \
    --action-dim 10 \
    --cem-iter 5 --cem-cand 1024 \
    --episodes 10
```

The bench:
1. Loads the runner.
2. Constructs 10 synthetic `(observation, goal)` pairs (random pixels,
   for latency measurement; correctness is verified separately).
3. Runs CEM on each.
4. Reports per-episode wall time as min/p50/p95/max.

Current numbers (Apple M3, release build):

| Metric | Value |
|--------|------:|
| Median latency/episode | 4.08 s |
| p95 latency/episode    | 4.13 s |
| Episodes / run         | 10 |

See [Benchmarks](./benchmark.md) for the full table.

## 6. Where the cost lives

Profiling (with `cargo flamegraph`) shows:

- **~85 %** of wall time is in `SimplePlan::run` for the predictor —
  the matmul kernels in Tract's optimised attention/MLP path.
- **~10 %** is in candidate sampling and proposal updates (the rng:cem
  path).
- **~5 %** is in encode + cost computation.

Debug-vs-release build is essentially indistinguishable (within
measurement noise), because the hot path is Tract's pre-compiled
kernels, not lewm-infer's orchestration code. The Tract ARM backend
uses optimised matmul kernels regardless of the host crate's
optimisation level.

## 7. Source pointers

| Topic | Source |
|-------|--------|
| Tract runner | `crates/lewm-infer/src/runner/tract_onnx_runner.rs` |
| Trait | `crates/lewm-infer/src/runner/traits.rs` |
| CEM driver | `crates/lewm-infer/src/plan.rs` |
| Preprocess | `crates/lewm-infer/src/preprocess.rs` |
| CLI | `crates/lewm-infer/src/bin/lewm-infer.rs` |
| Tract version | `=0.22.1` in `Cargo.toml` |
