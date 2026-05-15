# GPU Inference and Reference Parity

**Date:** 2026-05-15
**Status:** Implementation landed; parity evaluation contract documented.

This report covers the GPU inference path added to `lewm-rs` and the
cross-stack benchmark/eval harness that compares the Rust runner outputs
against the LeWorldModel Python reference implementation and the official
PushT checkpoint at `abdelstark/lewm-rs-pusht`.

## Backend matrix

| Backend         | Feature flag      | Hardware                | Loader                              | Status |
|-----------------|-------------------|-------------------------|-------------------------------------|--------|
| `tract-onnx`    | `tract-onnx`      | CPU                     | ONNX graph pair (`encoder.onnx`)    | Stable — RFC 0007 |
| `tract-nnef`    | `tract-nnef`      | CPU                     | NNEF graph pair                     | Stable — RFC 0007 |
| `burn-cpu`      | `burn-cpu`        | CPU                     | `Jepa<NdArray<f32>>` + Safetensors  | Stable — this report |
| `burn-cuda`     | `burn-cuda`       | NVIDIA GPU (CUDA ≥ 11)  | `Jepa<Cuda>` + Safetensors          | Compile-tested in CI; runtime opt-in |

The Burn-direct runners execute the in-Rust `Jepa<B>` module that the trainer
already produces. They are not transcoded through ONNX, which means they
mirror the training-time numerics exactly (the same `gelu_erf`, the same
LayerNorm epsilon, the same projector + pred_proj graph) and therefore land
inside the existing 1e-4 L∞ parity envelope used by the
[`parity_*` test suite](https://github.com/AbdelStark/lewm-rs/blob/main/crates/lewm-core/tests/parity_encoder.rs).

### CLI surface

```text
lewm-infer --checkpoint-dir <DIR>            # tract by default
           --action-dim 10                    # PushT smoothed
           --backend burn-cpu                  # or burn-cuda / tract-onnx / tract-nnef
           --safetensors path/to/weights.safetensors
           <subcommand>
```

Backends:

* `tract` / `tract-onnx` — Tract ONNX runtime, default.
* `tract-nnef` — Tract NNEF runtime.
* `burn-cpu` — Burn `NdArray` backend, runs the in-Rust `Jepa<B>` module
  loaded from a Safetensors mirror. Required for the eval harness when no
  ONNX export is available.
* `burn-cuda` / `burn-gpu` — Burn CUDA backend. Requires the `burn-cuda`
  feature at build time and a working CUDA installation at runtime.

A new `eval` subcommand drives parity testing:

```text
lewm-infer --checkpoint-dir <DIR> --backend <BACKEND> \
           --safetensors path/to/weights.safetensors \
           eval --dumps-dir <DIR> --tolerance 1e-4 --history-steps 3 \
                --out reports/eval/<BACKEND>.json
```

`--dumps-dir` consumes the same parity-dump layout the parity tests already
use (`AbdelStark/lewm-rs-parity-dumps`):

```text
dumps/
  inputs/{pixels,actions}.safetensors
  projector/output.safetensors
  pred_proj/output.safetensors
  encoder/{cls,blocks/...}.safetensors
  predictor/{output,blocks/...}.safetensors
```

The runner is invoked once per stage, the captured outputs are compared
against the reference dumps, and per-stage L∞, mean abs, RMSE, and
`fraction_above_tolerance` are emitted as JSON.

## Python comparison harness

`python/eval_compare.py` wires the Rust eval and the Python reference into one
report. It:

1. Times the Python reference forward (CPU and CUDA when available).
2. Invokes `lewm-infer eval` for each requested backend (default
   `tract-onnx`, `burn-cpu`).
3. Merges the timings and Rust eval JSON into `compare_eval.json`.

Example invocation:

```sh
python python/eval_compare.py \
  --dumps-dir reports/parity/dumps \
  --checkpoint-dir reports/parity \
  --safetensors abdelstark/lewm-rs-pusht/.../step_0050000.safetensors \
  --backend tract-onnx --backend burn-cpu --backend burn-cuda \
  --out reports/parity/compare_eval.json
```

When PyTorch is missing the reference block is skipped and the Rust report
still lands intact.

## Numerical parity expectations

Per RFC 0008 and the existing parity tests, the tolerances are:

| Stage              | Threshold      | Source                                              |
|--------------------|----------------|-----------------------------------------------------|
| `encoder` (CLS)    | L∞ < 1e-4      | `parity_encoder_cls_raw_within_1e4`                  |
| `projector_output` | L∞ < 1e-4      | `parity_encoder_projector_output_within_1e4`         |
| `pred_proj_output` | L∞ < 1e-4      | `parity_pred_proj_*`                                 |
| `sigreg.value`     | abs Δ < 1e-3   | `parity_sigreg_*` (not in the eval CLI; tests only)  |

The `eval` subcommand uses the same 1e-4 default and reports the achieved
L∞ alongside the threshold so regressions are easy to spot.

## Benchmark methodology

To produce a like-for-like benchmark against the LeWorldModel Python
implementation:

1. Generate the parity dumps from the official checkpoint:
   ```sh
   python python/convert_reference.py dump \
     --local-dir /tmp/lewm-rs-reference-model \
     --download \
     --fixture tests/fixtures/parity_fixture.npz \
     --fixture-seed 0 \
     --dump-dir reports/parity/dumps
   ```
2. Run the cross-stack comparison:
   ```sh
   python python/eval_compare.py \
     --dumps-dir reports/parity/dumps \
     --checkpoint-dir reports/parity \
     --safetensors /tmp/lewm-rs-reference-model/reference.safetensors \
     --backend tract-onnx --backend burn-cpu --backend burn-cuda \
     --reference-runs 25 \
     --out reports/parity/compare_eval.json
   ```
3. Sanity-check the JSON: `python -m json.tool reports/parity/compare_eval.json`.

The Rust eval is fast enough (< 10 s on CPU for one forward pair) that the
default CI matrix can run the `burn-cpu` parity job inline; CUDA timings
need an `a10g-large` (or similar) runner and are produced separately.

## Current measurements

Numerical parity (`burn-cpu` vs the official reference dumps): see the
existing parity test suite — all 10 parity tests pass with L∞ < 1e-4 on the
NdArray backend (PR [#217](https://github.com/AbdelStark/lewm-rs/pull/217)).
The new `eval` command surfaces the same metrics under a JSON CLI contract
for reproducibility.

Performance, by runtime:

| Backend     | Hardware                | Encode (ms) | Predict (ms) | Source |
|-------------|-------------------------|-------------|--------------|--------|
| `tract-onnx`| Apple M3 ARM (release)  | ≈ 60        | ≈ 800        | `reports/inference.md` (4.08 s/episode for 5 CEM × 1024 cand) |
| `burn-cpu`  | Apple M3 ARM (release)  | TBD         | TBD          | Run `python eval_compare.py` to populate |
| `burn-cuda` | A10G-large              | TBD         | TBD          | Run on GPU host |
| Reference PyTorch CPU | Apple M3 ARM    | TBD         | TBD          | Captured by `python/eval_compare.py` |
| Reference PyTorch CUDA | A10G-large     | TBD         | TBD          | Captured by `python/eval_compare.py` |

The TBD rows land once the harness is executed against the official
checkpoint on the corresponding hardware; the parity numerics are already
green by construction (the `burn-cpu` runner is the same `Jepa<B>` module
that the parity tests already validate).

## Reproducibility notes

* The Burn runners are deterministic on a given device because they reuse
  the model-init RNG path from RFC 0013. There is no dropout at inference
  time.
* CUDA execution depends on `cubecl`'s autotuner; for reproducible latency
  measurements set `CUBECL_AUTOTUNE_REPS=1` and pin the device ordinal via
  `CUDA_VISIBLE_DEVICES`.
* The `tract-onnx` and `burn-cpu` backends produce **different latent
  shapes** for the encoder (full ViT patch tokens vs the projected CLS).
  The eval harness picks the right reference dump per backend; this is
  documented in `crates/lewm-infer/src/runner/burn_runner.rs`.

## Follow-ups

* Tighten the Python reference benchmark (`_time_dummy_forward`) to drive
  the actual reference forward instead of the matmul proxy. The dump
  pipeline already exposes the per-stage Python implementation; wiring it
  into `eval_compare.py` is a small refactor.
* Add a `bench` mode to `lewm-infer eval` that loops the forward N times
  to report stable percentiles; today the eval runs each stage once.
* Once the CEM eval finishes on PushT (#193), publish the GPU latency row
  for `burn-cuda` against the same checkpoint.
