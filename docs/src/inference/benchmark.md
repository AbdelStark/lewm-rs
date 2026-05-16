# Latency benchmarks

> **Motivation.** "CPU planning on a laptop" is only meaningful if the
> latency is known. This page collects the bench results.
>
> **Position.** Sub-page of [Part V](./onnx-export.md).
>
> **What you should leave with.** Current numbers, the bench protocol,
> and where the cost goes.

## 1. Headline numbers

| Backend | Hardware | Build | p50 / episode | p95 / episode |
|---------|----------|------:|--------------:|--------------:|
| Tract (CPU ONNX) | Apple M3 (ARM) | debug | 4.10 s | 4.15 s |
| Tract (CPU ONNX) | Apple M3 (ARM) | release | 4.08 s | 4.13 s |
| Burn NdArray (CPU) | Apple M3 | release | pending | pending |
| Burn CUDA | A10G | release | pending | pending |

CEM configuration: `n_iter = 5`, `n_cand = 1024`, `horizon_plan = 5`,
`H_hist = 3`, `action_dim = 10` (smoothed). The benchmark workload is
10 synthetic episodes (random pixels, random goals).

Why debug ≈ release? Tract's hot path is its pre-compiled kernel
library, not lewm-infer's orchestration. The host crate's optimisation
level affects only a thin shim around Tract's `SimplePlan::run`.

## 2. The bench command

```sh
lewm-infer bench \
    --checkpoint-dir abdelstark/lewm-rs-pusht/tract-compat/ \
    --history-steps 3 \
    --action-dim 10 \
    --cem-iter 5 --cem-cand 1024 --horizon 5 \
    --episodes 10
```

Output goes to stdout as JSON-lines and to a summary at the end:

```text
{"kind":"bench","ep":0,"latency_ms":4123.4,"iter_ms":[820.4, 818.7, 821.1, 822.2, 820.0]}
{"kind":"bench","ep":1,"latency_ms":4099.8, ...}
...
{"kind":"bench_summary","episodes":10,"p50_ms":4083.2,"p95_ms":4127.5,...}
```

## 3. Where the cost goes

Profiling (with `cargo flamegraph`):

| Phase | Share of wall time |
|-------|-------------------:|
| Predictor forward (Tract `SimplePlan::run`) | ~85 % |
| Candidate sampling (RNG + reshape) | ~7 % |
| Elite selection / proposal update | ~3 % |
| Encode (encoder forward) | ~3 % |
| Misc (cost computation, logging) | ~2 % |

The 4-second budget is dominated by the predictor. To cut latency
significantly we would need:

- Parallel candidates (currently serial — the planner runs one
  candidate at a time through the predictor).
- Quantised ONNX (INT8) — a 2–4× speedup on Tract is plausible, with
  some accuracy cost.
- A smaller predictor (e.g. 4 blocks instead of 6).

All three are listed in [`ROADMAP.md`] as future work.

## 4. Latency vs CEM hyperparameters

Cutting CEM aggression shaves wall time linearly:

| Config | Predicted p50 (Apple M3, release) |
|--------|------------------------------------|
| `n_iter=5, n_cand=1024` (default) | 4.08 s |
| `n_iter=3, n_cand=1024` | ~2.45 s |
| `n_iter=5, n_cand=512`  | ~2.04 s |
| `n_iter=3, n_cand=512`  | ~1.22 s |

These are linear projections based on the (predictor calls ∝ `n_iter ·
n_cand · H`) cost model. The actual `n_iter=3, n_cand=512` benchmark is
pending; the linear projection is a guide, not a guarantee.

## 5. CPU throughput on x86_64

The current benchmark is Apple M3 (ARM). Tract has AVX2/AVX-512 paths
on x86_64. A back-of-the-envelope projection: on a modern x86_64 CPU
with AVX-512 (Ice Lake server, Sapphire Rapids workstation), the
predictor matmul should run ~2× faster than on Apple M3 (which has
extremely strong per-core but no AVX-512). A 1.5–2.5 s p50 is
plausible.

The "release-build x86_64 baseline" is listed as future work in
[`ROADMAP.md`].

## 6. Source pointers

| Topic | Source |
|-------|--------|
| Bench CLI | `crates/lewm-infer/src/bin/lewm-infer.rs` (`bench` subcommand) |
| CEM driver | `crates/lewm-infer/src/plan.rs` |
| Inference reports | `reports/inference.md`, `reports/gpu_inference.md` |
| Flamegraph scripts | `profiling/` |

[`ROADMAP.md`]: https://github.com/AbdelStark/lewm-rs/blob/main/ROADMAP.md
