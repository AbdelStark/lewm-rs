# Burn NdArray and CUDA runners

> **Motivation.** Tract is the deployment runner, but for parity
> evaluation and debugging we also run the model through Burn directly
> — on CPU (NdArray) and GPU (CUDA). These runners load Burn
> `.safetensors` weights into the in-Rust `Jepa<B>` module and run
> them in-process.
>
> **Position.** Sub-page of [Part V](./onnx-export.md).
>
> **What you should leave with.** What `--backend burn-cpu` and
> `--backend burn-cuda` do, what they're useful for, and which
> reference-parity bound they hit.

## 1. The two backends

The `lewm-infer` binary supports three runners, selected by
`--backend`:

| `--backend` | Backend | Use case |
|-------------|---------|----------|
| `tract` (default) | Tract 0.22.1 (CPU, ONNX) | Deployment, CPU planning, low-latency. |
| `burn-cpu` | Burn `NdArray<f32>` | Parity reference on CPU; no GPU required. |
| `burn-cuda` (feature `burn-cuda`) | Burn `Cuda<f32>` | GPU parity reference; A10G / consumer cards. |

All three runners conform to the same `InferenceRunner` trait
(`crates/lewm-infer/src/runner/traits.rs`) and produce the same
function-signature outputs. The difference is purely the underlying
compute.

## 2. The Burn `NdArray` runner

```rust,ignore
use burn::backend::NdArray;
use burn::record::DefaultRecorder;

type B = NdArray<f32>;

let device = burn_ndarray::NdArrayDevice::default();
let config: JepaConfig = JepaConfig::load("config.toml")?;
let mut jepa: Jepa<B> = config.init(&device);
let record = DefaultRecorder::new().load("step_0050000.mpk".into(), &device)?;
jepa = jepa.load_record(record);

let z = jepa.encode(pixels);                  // (B, T+1, 192)
let z_pred = jepa.predict(history, actions);  // (B, T, 192)
```

This is the "obvious" path: load the Burn `.mpk` record, run the
`Jepa<NdArray>` module directly. The advantage: it uses the same Rust
code paths as training; any parity issue with Tract isolates to the
ONNX boundary (encoder.onnx, predictor.onnx, Tract's optimiser) rather
than the model code.

## 3. The CUDA runner

`--backend burn-cuda` is gated behind the Cargo feature `burn-cuda`.
With it enabled, the same `Jepa<B>` module runs on a NVIDIA GPU:

```rust,ignore
type B = burn_cuda::Cuda<f32>;
let device = burn_cuda::CudaDevice::default();
let jepa: Jepa<B> = config.init(&device).load_record(record);
let z = jepa.encode(pixels);
```

This path is used for GPU parity evaluation in
`reports/gpu_inference.md` and is built + CI-checked but has not
been benchmarked on A10G yet (see status table on the
[Project status](../status.md) page).

## 4. The eval CLI

The unified parity-eval CLI exercises all three runners against the
official reference dumps:

```sh
lewm-infer eval --dumps-dir <path> --backend tract
lewm-infer eval --dumps-dir <path> --backend burn-cpu
lewm-infer eval --dumps-dir <path> --backend burn-cuda
```

Each invocation reads the locked input fixture from `<path>`, runs
the chosen backend, and writes a per-stage JSON of L∞ / RMSE against
the reference dump. The relevant tolerances are the [TOL-001..003]
encoder / predictor / sigreg tolerances and (for `burn-cuda`) the
TOL-010 mixed-precision bound when BF16 is enabled.

The output JSON looks like:

```json
{
  "schema_version": "1.0.0",
  "backend": "burn-cpu",
  "stages": {
    "encoder_cls":      { "l_inf": 1.92e-05, "rmse": 8.7e-06, "pass": true },
    "predictor":        { "l_inf": 4.31e-05, "rmse": 1.4e-05, "pass": true },
    "sigreg_scalar":    { "abs_diff": 2.7e-04,                 "pass": true }
  }
}
```

## 5. Why all three

| Tract | Burn CPU | Burn CUDA | Purpose |
|:-----:|:--------:|:---------:|---------|
| ✅ | | | Deployment runner. The one users actually call. |
| | ✅ | | CPU parity reference. Verifies the Tract graph is faithful. |
| | | ✅ | GPU parity reference. Closes the loop on the trained model. |

The first two together establish: *"the Tract graph matches what
Burn-CPU computes from the same `.mpk`"*, which is the contract the
demo Space depends on. The third closes the loop back to training:
*"the GPU-trained `.mpk` is numerically faithful to itself when run
on CPU and via Tract"*.

## 6. Source pointers

| Topic | Source |
|-------|--------|
| Trait | `crates/lewm-infer/src/runner/traits.rs` |
| Tract runner | `crates/lewm-infer/src/runner/tract_onnx_runner.rs` |
| Burn-CPU runner | `crates/lewm-infer/src/runner/burn_ndarray_runner.rs` |
| Burn-CUDA runner | `crates/lewm-infer/src/runner/burn_cuda_runner.rs` (feature-gated) |
| Eval driver | `crates/lewm-infer/src/eval.rs` |
| `lewm-gpu` crate | `crates/lewm-gpu/` — CUDA-specific helpers |

[TOL-001..003]: ../reference/tolerances.md
