# Running CPU inference

## 1. Get the artifacts

```sh
git clone https://huggingface.co/abdelstark/lewm-rs-pusht
cd lewm-rs-pusht/tract-compat
ls
# encoder.onnx  predictor.onnx  stats.safetensors  README.md
```

The `tract-compat/` directory contains the Tract-compatible ONNX
graphs (fixed batch, opset 17, embedded weights). For onnxruntime use,
the root-level `encoder.onnx` + `encoder.onnx.data` + `predictor.onnx` +
`predictor.onnx.data` files are the dynamo-exported versions.

## 2. Benchmark

```sh
target/release/lewm-infer bench \
    --checkpoint-dir lewm-rs-pusht/tract-compat/ \
    --history-steps 3 \
    --action-dim 10 \
    --cem-iter 5 --cem-cand 1024 --horizon 5 \
    --episodes 10
```

Expected output (Apple M-series, release build):

```text
{"kind":"bench","ep":0,"latency_ms":4123.4,...}
...
{"kind":"bench_summary","episodes":10,"p50_ms":4083.2,"p95_ms":4127.5,...}
```

## 3. Plan a single episode

```sh
target/release/lewm-infer plan \
    --checkpoint-dir lewm-rs-pusht/tract-compat/ \
    --start-image example_start.png \
    --goal-image example_goal.png \
    --cem-iter 5 --cem-cand 1024 \
    --horizon 5
```

Output is the planned action sequence (5 actions of dim 2 for PushT)
plus the per-iteration cost trace.

## 4. Parity-eval against the reference dumps

```sh
target/release/lewm-infer eval \
    --dumps-dir AbdelStark/lewm-rs-parity-dumps \
    --backend tract \
    --checkpoint-dir lewm-rs-pusht/tract-compat/
```

Reports per-stage $L_\infty$ / RMSE; should pass with $L_\infty <
10^{-4}$ on encoder and predictor.

## 5. Try the demo Space

The hosted demo at
[`abdelstark/lewm-rs-demo`](https://huggingface.co/spaces/abdelstark/lewm-rs-demo)
provides a Gradio UI for the same CEM planning flow. No local install
required.

## 6. Burn-CPU and Burn-CUDA backends

For parity reference:

```sh
target/release/lewm-infer eval \
    --dumps-dir AbdelStark/lewm-rs-parity-dumps \
    --backend burn-cpu \
    --checkpoint-dir <local clone with step_0050000.safetensors>

# With CUDA feature:
cargo build --release -p lewm-infer --features burn-cuda
target/release/lewm-infer eval \
    --dumps-dir AbdelStark/lewm-rs-parity-dumps \
    --backend burn-cuda \
    --checkpoint-dir <local clone with step_0050000.safetensors>
```

See [Burn runners](../inference/burn-runners.md).
