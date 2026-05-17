# `lewm-gpu`

GPU inference glue for `LeWM`. This crate keeps the `burn-cuda` dependency
**outside** of `lewm-infer` per the RFC 0007 layering rule, which is enforced
by [`scripts/check_layers.py`][layers] (`burn-cuda` is in
`INFER_BANNED_DEPS`).

**Specs:** [RFC 0007 — Tract inference and ONNX export][rfc-0007],
[RFC 0014 — performance engineering][rfc-0014].

**Depends on:** `lewm-core`, `lewm-infer`.

## Why a separate crate?

`lewm-infer` must build on CUDA-less hosts (CI runners, Apple Silicon,
distroless containers). Adding `burn-cuda` there would pull in `cudarc`,
`nvcc`, and `nvml` and break the workspace build everywhere we cannot ship a
CUDA toolchain. Moving CUDA glue into a tiny terminal crate keeps the
inference library clean while still exposing a GPU path to downstream
binaries.

## Surface

```rust,ignore
use lewm_core::JepaConfig;
use lewm_gpu::{LewmGpuError, load_cuda_runner};

let runner = load_cuda_runner(
    std::path::Path::new("/path/to/weights.safetensors"),
    JepaConfig::default(),
)?;
# Ok::<_, LewmGpuError>(())
```

The crate re-uses `lewm-infer`'s backend-generic `BurnJepaRunner` and the
`lewm-core::import` Safetensors loader; the only CUDA-specific code is the
backend type selection.

## Features

- `burn-cuda` (default on): wire up the CUDA backend. Off-by-default
  consumers can disable it for a CPU-only build.

## Testing

CI verifies that `cargo clippy --workspace --all-targets` is warning-free
under the `default`, `cpu-only`, and `parity-fixtures` feature matrices.
Runtime CUDA execution is exercised manually against the published checkpoint
and reported in [`reports/gpu_inference.md`](../../reports/gpu_inference.md).

[rfc-0007]: ../../specs/rfcs/0007-tract-inference-and-onnx-export.md
[rfc-0014]: ../../specs/rfcs/0014-performance-engineering.md
[layers]: ../../scripts/check_layers.py
