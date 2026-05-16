# `lewm-gpu`

CUDA-specific helpers used by `lewm-infer` and `lewm-train` when the
`burn-cuda` feature is enabled. Kept in its own crate so the rest of
the workspace compiles cleanly on machines without a CUDA toolkit.

## What it owns

- **CUDA device discovery**: enumerate available devices, set the
  current device.
- **CUDA-specific BF16 / F32 helpers**: explicit casts that avoid
  Burn's "auto-cast" surprises on cuDNN-backed kernels.
- **Memory probes**: optional VRAM-usage probes for logging.

## Feature gate

The crate is built only when the workspace feature `cuda` is enabled
(which itself implies `burn-cuda`). On a CPU-only build, `lewm-gpu` is
not compiled, and downstream crates use a stubbed module shim.

## Source

[`crates/lewm-gpu`](https://github.com/AbdelStark/lewm-rs/tree/main/crates/lewm-gpu)
