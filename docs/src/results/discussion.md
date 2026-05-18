# Discussion and limitations

> **Motivation.** Honest engineering reports include their open
> questions. This page enumerates what is known to be unfinished, what
> the results do and do not establish, and what the natural next steps
> are.
>
> **Position.** Closing sub-page of [Part VII — Results](./pusht.md).
>
> **What you should leave with.** A clear-eyed view of where lewm-rs
> stands as a reproduction, a deployment, and a research baseline.

## 1. What the results establish

The training and parity work to date establishes:

1. **Numerical parity** with the upstream PyTorch reference, to
   $L_\infty < 10^{-4}$ on every checked activation. The implementation
   is a faithful port, not a re-interpretation.
2. **A working bounded-core training pipeline** in pure Rust,
   end-to-end, that converges on both PushT (50 k steps, 5.3 h on
   A10G) and SO-100 (5 k steps, 14 min on A10G), with zero gradient
   explosions and zero collapse-probe trips.
3. **A working CPU inference path** via reference ONNX export + Tract,
   at 4.08 s per planning episode on Apple M-series. No Python is
   required at inference time, but these are not yet the F1 trained
   `onnx-full/` artifacts.
4. **A reproducible engineering envelope**: pinned toolchain, locked
   `Cargo.lock`, deterministic seed handling, parity tests gated in
   CI, cost ledger under \$12 / \$200 cap.

## 2. What the results do not yet establish

Four things are pending evaluation or publication:

1. **Full Burn/Jepa PushT checkpoint.** The historical 50 k-step PushT
   checkpoint is a bounded-core artifact. F1 still needs an
   approval-gated full Burn/Jepa run that publishes
   `train/pusht-full-burn-jepa-*`.
2. **PushT planning success rate.** The CEM eval against the 50-episode
   PushT test set has not been run. The target is ≥ 87 % (matching the
   upstream paper). The bounded-core training loss curve looks like a
   successful model, but until the F1 export and eval run, this is
   unverified for the release checkpoint.
3. **SO-100 latent-MSE and Spearman.** The held-out eval split has
   been prepared but not evaluated.
4. **Warm-start ablation.** The from-PushT SO-100 training run has not
   yet been launched.

## 3. The "bounded model" gap

The most important open engineering item is the **bounded-model gap**:
the difference between the simplified core used for end-to-end training
and the full Burn `Jepa` used for parity and export. Concretely, the
50 k-step PushT training run uses `PushtFullLewmCore`, a simplified
$\sim 14$-tensor Rust core, not the full Burn `Jepa` (303 tensors,
$18\,042\,672$ parameters).

The full `Jepa<B>` module passes all 10 parity tests against the
upstream checkpoint — it is *correct*. The checked-in training job now
has an opt-in full Burn/Jepa path and a local contract smoke, but the
50 k-step approval-gated run has not been launched and published yet.
This is the primary release blocker tracked as F1.

Until that is done:

- The release `onnx-full/` artifacts do not exist.
- The current ONNX export uses converted PyTorch reference weights, not
  a natively Rust-trained ViT checkpoint.
- The CPU inference benchmarks measure the *upstream model's* CPU cost,
  not a Rust-trained release checkpoint.
- The PushT eval is blocked until F1 publishes a trained full-layout
  checkpoint and verified ONNX artifacts.

This is an important caveat. The project's claim of "Rust reproduction"
is currently true at the *architecture* and *parity* level, partially
true at the *training* level (PushT minimal core trained end-to-end),
and pending at the *full-stack* level (`Jepa<B>` trained end-to-end
and exported).

## 4. Cross-platform reproducibility

The determinism contracts are *same-hardware*: given the same source,
config, seed, and GPU, two runs produce bit-identical losses. They are
**not** cross-platform: a CPU run and a GPU run on the same seed will
differ; an A10G run and an H100 run on the same seed will differ. The
divergence is bounded by TOL-005 (rel. < 1e-2) at step 100 for
CPU-vs-GPU smokes, but is not bit-identical.

This is unavoidable at the F32 reduction-order level. Bit-identical
cross-platform reproducibility is not a project goal.

## 5. Performance ceiling

The Tract CPU inference benchmark at 4.08 s/episode is **with the
serial CEM loop**, on Apple M3 (ARM, no AVX). The roadmap items that
would meaningfully change this:

- **Parallel candidates.** Batch the predictor over all $n_{\text{cand}}$
  in a single Tract call. Expected speedup: 4–8×.
- **INT8 quantisation.** Quantise the ONNX graphs to INT8 and use
  Tract's INT8 kernels. Expected: 2× on top.
- **x86_64 AVX-512.** A modern x86_64 CPU should run the predictor
  ~2× faster than Apple M3. Expected with parallel candidates +
  AVX-512: sub-500 ms per episode.

These are listed in `ROADMAP.md` but are not part of the v1 deliverable.

## 6. Comparison to upstream

LeWM (Maes et al., 2026, arXiv:2502.16560) reports PushT success rates
around 87 %. `lewm-rs` aims to match this on the converted upstream
weights and to demonstrate that the Rust pipeline is faithful enough
that the *pipeline* can reproduce the result given the same compute.
This is exactly what the [PushT result](./pusht.md) plus the pending
eval are designed to show.

We have **not** improved over upstream. The algorithm, the
architecture, the hyperparameters, the seed, the optimizer — all
match. The contribution is reproducibility, deployability, and
language choice (Rust rather than Python at training time, Rust + Tract
at inference time).

## 7. What this project is good for

- **Reproducing the LeWM result** in a Rust-first stack.
- **Understanding the LeWM design** through the docs and specs.
- **Porting LeWM to a new dataset** — the SO-100 extension shows how.
- **Deploying LeWM-style world models on CPU** — the Tract path is
  the closest the field gets to "robot world model in a single binary".

## 8. What this project is not good for

- **Improving LeWM's algorithm.** The Rust port did not change any
  algorithmic choice. To improve, you need a different paper.
- **Production robotics.** The SO-100 work is research-grade
  reproduction; sim-to-real, safety guarantees, and real-time hardware
  integration are out of scope.
- **Pure CPU training at scale.** Burn's CPU backend works (NdArray)
  but is not competitive with GPU for 50 k-step runs. Training is a
  GPU task.

## 9. Next steps

In rough priority order, from [`ROADMAP.md`]:

1. Wire `lewm_core::Jepa` into the training loop and re-run PushT
   50 k-step end-to-end in Rust.
2. Run the PushT CEM planning eval.
3. Run the SO-100 eval and the warm-start ablation.
4. Add release-build x86_64 benchmark.
5. Parallel-candidate CEM on Tract.
6. Multi-camera SO-100 inputs.
7. INT8 ONNX quantisation.

If you are interested in contributing any of these, see
[Contributing](../community/contributing.md).

[`ROADMAP.md`]: https://github.com/AbdelStark/lewm-rs/blob/main/ROADMAP.md
