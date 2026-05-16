# Training pipeline

> **Motivation.** A reproducible training pipeline is the sum of many
> small contracts. This page surveys the whole pipeline at once so
> later pages can focus on individual contracts in depth.
>
> **Position.** Top of [Part III](../introduction.md).
>
> **What you should leave with.** A mental model of the training
> state machine, the role of each crate in one optimizer step, and a
> map of the rest of Part III.

## 1. The state machine

[RFC 0005] specifies an 8-state training pipeline:

```text
   INIT  →  PARITY_CHECK  →  SMOKE  →  WARMUP  →  STEADY  →  COOLDOWN  →  EVAL  →  UPLOAD  →  DONE
```

Each transition is gated:

| Transition | Gate |
|------------|------|
| `INIT → PARITY_CHECK` | Config loaded; model + optimizer + dataset constructed. |
| `PARITY_CHECK → SMOKE` | All 10 parity tests pass on the locked fixture. |
| `SMOKE → WARMUP` | 50-step CPU smoke completes; sidecar shapes match. |
| `WARMUP → STEADY` | Cosine LR has crossed peak. |
| `STEADY → COOLDOWN` | LR has dropped below cosine inflection. |
| `COOLDOWN → EVAL` | Final cosine-min reached; checkpoint saved. |
| `EVAL → UPLOAD` | Eval report written; success-rate / latent-MSE recorded. |
| `UPLOAD → DONE` | All artifacts uploaded; cost ledger updated. |

The state is persisted in the checkpoint sidecar (`step_{N}.json`) so a
crash-restart resumes at the correct state.

## 2. One optimizer step, end to end

One step of training touches every crate. Here is the dataflow:

```text
   ┌─────────────────┐
   │  HDF5 file(s)   │   PushT / SO-100 dataset
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  lewm-data      │   PushtDataset / So100Dataset
   │                 │   Window sampler, RGB→f32, action norm
   │  prefetch       │   Worker pool, channel
   └────────┬────────┘
            │ (Batch struct: pixels, actions, metadata)
            ▼
   ┌─────────────────┐
   │  lewm-train     │   step.rs: one optimizer step
   │                 │
   │  jepa.encode()  │
   │  jepa.predict() │   →  lewm-core
   │  losses         │   →  lewm-core::losses
   │                 │
   │  backward       │
   │  AdamW step     │
   │  schedule       │
   │  grad clip      │
   │                 │
   │  emit metrics   │   →  lewm-telemetry
   │  every N steps  │
   │                 │
   │  checkpoint     │
   │  every M steps  │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  lewm-hub       │   upload checkpoints (post-run)
   └─────────────────┘
```

Crate responsibilities at one glance:

| Crate | Role in one step |
|-------|------------------|
| `lewm-data` | Reads HDF5, samples windows, normalises images and actions. |
| `lewm-core` | Owns `Jepa<B>`; provides forward, losses, init, parity helpers. |
| `lewm-train` | Outer loop, optimizer, schedule, mixed precision, checkpoints, resume. |
| `lewm-telemetry` | Stdout JSONL + optional OTLP traces. |
| `lewm-hub` | Post-step (not per-step) artifact upload. |

## 3. A single step in pseudocode

The body of the inner loop in `crates/lewm-train/src/step.rs`:

```rust,ignore
// 1. Pull a micro-batch from the prefetch channel.
let batch = prefetcher.next()?;

// 2. Forward (with autograd).
let z      = jepa.encode(batch.pixels);                 // (B, T+1, D)
let z_proj = jepa.projector.forward(z);                  // (B, T+1, 1024)

let history_z   = z.narrow(1, 0, T);                     // (B, T, D)
let pred_z      = jepa.predict(history_z, batch.actions);// (B, T, D)
let pred_z_1024 = jepa.pred_proj.forward(pred_z);        // (B, T, 1024)

let target_z    = z_proj.narrow(1, 1, T);                // (B, T, 1024)

// 3. Compute losses (sigreg always in F32 — see Mixed precision).
let l_pred   = prediction_loss(pred_z_1024, target_z);
let l_sigreg = sigreg_loss(z_proj.reshape([-1, 1024]));
let l_total  = l_pred + lambda * l_sigreg;

// 4. Backward (BF16 if mixed; F32 islands enforced).
let grads = l_total.backward();

// 5. Pre-clip grad norm for monitoring; trip TOL-011 if > 1e3.
let grad_norm = grads.norm_l2();
assert!(grad_norm < grad_norm_ceiling);

// 6. Clip and step.
grads = grads.clip_l2(1.0);
optimizer.step(grads);
scheduler.advance();

// 7. Emit metrics (every N steps).
if step % log_interval == 0 {
    telemetry.emit_step(step, l_total, l_pred, l_sigreg, lr_now, grad_norm);
}

// 8. Checkpoint (every M steps).
if step % ckpt_interval == 0 {
    checkpoint.save(&jepa, &optimizer, &scheduler, step, rng_state);
}
```

With `grad_accum_steps = K`, the optimizer step happens only once every
$K$ micro-batches. The forward/backward are run on each micro-batch and
their gradients are summed before the step.

## 4. Map of the rest of Part III

Each step above corresponds to a sub-page:

- **[Data plane and window sampling](./data.md)** — what `Batch`
  contains, how it is produced.
- **[Loss functions](./losses.md)** — the F32-island SIGReg and the
  prediction MSE in detail.
- **[Gradient flow](./gradient-flow.md)** — which arms receive
  gradient, why no EMA.
- **[AdamW, decay groups, schedule](./optimizer.md)** — the
  decay/no-decay split, cosine+warmup math, $\beta_1, \beta_2$.
- **[Mixed precision](./mixed-precision.md)** — the BF16 envelope
  and F32 islands.
- **[State machine](./state-machine.md)** — the 8-state contract
  in detail, with the gate predicates.
- **[Determinism](./determinism.md)** — RNG sub-streams, seeding,
  the "bit-identical resume" property.
- **[Checkpoints](./checkpoints.md)** — the sidecar schema, the
  `.mpk`/`.safetensors` split, atomic writes.
- **[Observability](./observability.md)** — JSONL emission, OTLP
  traces, the dashboard.

## 5. Where this happens

| Topic | Source |
|-------|--------|
| State machine driver | `crates/lewm-train/src/trainer.rs` |
| Inner step | `crates/lewm-train/src/step.rs` |
| Optimizer wrapper | `crates/lewm-train/src/optim.rs` |
| Scheduler | `crates/lewm-train/src/schedule.rs` |
| Mixed precision | `crates/lewm-train/src/mixed_precision.rs` |
| Checkpoint save/load | `crates/lewm-train/src/checkpoint.rs` |
| Resume | `crates/lewm-train/src/resume.rs` |
| Warmstart | `crates/lewm-train/src/warmstart.rs` |
| CLI | `crates/lewm-train/src/bin/lewm-train.rs` |

[RFC 0005]: ../reference/rfcs.md
