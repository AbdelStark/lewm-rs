# `lewm-infer`

CPU inference and export verification for the deployment path. This crate
intentionally excludes CUDA and autodiff dependencies so it builds on
CUDA-less hosts (CI, Apple Silicon, etc.).

**Specs:** [RFC 0007 — Tract inference and ONNX export][rfc-0007],
[RFC 0008 — reference parity testing][rfc-0008].

**Depends on:** `lewm-core`, `lewm-telemetry`.
**Layering invariant:** `burn-cuda`, `burn-autodiff`, and `nvml-wrapper` are
**banned** here (enforced by `scripts/check_layers.py`); GPU glue lives in the
separate `lewm-gpu` crate.

## Module map

- `errors` — crate error type (`InferError`) + `InferResult`.
- `export` — RFC 0007 ONNX export graph contract + verifier fallback.
- `eval` — parity evaluation CLI helpers (`L∞` / `RMSE` per stage).
- `plan` — CPU-side CEM action search for inference.
- `preprocess` — RFC 0004-compatible image preprocessing.
- `runner` — backend-generic `InferenceRunner` trait, the Tract loader, and
  the `BurnJepaRunner<B>` for backend-generic Burn runs (CPU default).

## Binary

`lewm-infer` exposes:

- `lewm-infer infer …` — single-shot inference against a Tract ONNX file or a
  Burn Safetensors checkpoint.
- `lewm-infer eval --dumps-dir DIR --backend BACKEND --safetensors WEIGHTS`
  — compares any runner against the reference parity dumps and emits per-stage
  `L∞` / `RMSE` JSON.
- `lewm-infer bench` — CPU benchmark over the CEM planning hot path (Tract
  median: ≈4.1 s / episode on Apple M3 ARM, release build).

## ONNX export contract

The release exports two ONNX flavors:

- **opset 18** (`onnxruntime`): dynamo-exported, dynamic axes, used by the
  Gradio demo Space.
- **opset 17** (`tract-compat`): fixed-batch, causal-mask materialized as a
  buffer, used by the Rust Tract runner.

Both variants pass byte-for-byte verification via the
`export_verifier_release_smoke` integration test.

[rfc-0007]: ../../specs/rfcs/0007-tract-inference-and-onnx-export.md
[rfc-0008]: ../../specs/rfcs/0008-reference-parity-testing.md
