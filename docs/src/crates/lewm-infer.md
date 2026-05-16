# `lewm-infer`

The deployment crate. Loads ONNX graphs into Tract, runs CEM on CPU,
and provides a parity-evaluation CLI. Crucially, it has **no
dependency on `burn-autodiff`, `burn-cuda`, `burn-train`, or
`lewm-train`** (INV-003).

## What it owns

- **Runners**: a trait `InferenceRunner` with three impls:
  - `TractOnnxRunner` (default, CPU ONNX via Tract).
  - `BurnNdArrayRunner` (CPU parity reference).
  - `BurnCudaRunner` (GPU parity reference, feature-gated).
- **CEM**: a CPU-friendly CEM driver that calls the runner's
  `predict` in a loop.
- **Preprocess**: image preprocessing (matches `lewm-data` for
  decode + resize + normalize).
- **Eval**: per-stage activation comparison against the reference
  dumps.
- **Bench**: latency benchmark.
- **Demo bridge**: a Rust HTTP shim used by the Gradio Space (current
  Space is pure Python; this is for future deploys).

## Module layout

```text
lewm-infer/src/
├── lib.rs
├── bin/
│   └── lewm-infer.rs       # clap CLI: plan, bench, eval, export
├── export/                  # ONNX/NNEF export adapters
├── runner/
│   ├── mod.rs
│   ├── traits.rs            # InferenceRunner trait
│   ├── tract_onnx_runner.rs # Tract ONNX (CPU)
│   ├── burn_ndarray_runner.rs
│   └── burn_cuda_runner.rs  # feature = "burn-cuda"
├── plan.rs                  # CEM on CPU
├── eval.rs                  # parity eval driver
├── preprocess.rs            # image pre-processing
└── errors.rs
```

## CLI

```text
lewm-infer <subcommand> [flags]

Subcommands:
  plan        Run CEM for one (start, goal) pair.
  bench       Latency benchmark across N synthetic episodes.
  eval        Parity-eval vs official reference dumps.
  export      Wrap python/export_onnx.py for end-users.

Common flags:
  --backend <tract|burn-cpu|burn-cuda>
  --checkpoint-dir <DIR>         Directory holding encoder.onnx, predictor.onnx, stats.safetensors
  --history-steps <N>            Default 3
  --action-dim <N>               Inferred from predictor by default
  --cem-iter <N>, --cem-cand <N>, --horizon <N>
```

## Dependencies

- `lewm-core` (for shape and config types, *not* for runtime)
- `tract-onnx`, `tract-core` (= 0.22.1)
- `burn-ndarray` (CPU runner)
- `burn-cuda` (CUDA runner, feature-gated)
- `safetensors`
- (**no** dependency on `burn-autodiff`, `burn-train`, `lewm-train`)

## Source

[`crates/lewm-infer`](https://github.com/AbdelStark/lewm-rs/tree/main/crates/lewm-infer)
