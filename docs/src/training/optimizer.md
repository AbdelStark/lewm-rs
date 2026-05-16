# AdamW, decay groups, and the schedule

> **Motivation.** The optimizer and LR schedule are pinned to specific
> hyperparameters that match the upstream LeWM paper. Getting them
> wrong is a common reason for "trains, but to a different loss floor"
> failures.
>
> **Position.** Fourth sub-page in [Part III](./pipeline.md).
>
> **What you should leave with.** The exact AdamW configuration, the
> decay / no-decay split, the cosine + warmup math, and a pointer to
> the source.

## 1. AdamW

```text
AdamW with:
    β₁ = 0.9
    β₂ = 0.95
    ε  = 1e-8
    weight_decay = 0.05  (PushT)  / 0.01 (SO-100)
```

$\beta_2 = 0.95$ (not the default $0.999$) is the *transformer*
convention dating back to GPT-3: it tracks the second-moment with a
shorter half-life, which makes the optimizer more responsive to
non-stationary gradient statistics during warmup.

The update rule is the standard AdamW (Loshchilov & Hutter, 2019):

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\,g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\,g_t^2 \\
\hat m_t &= m_t / (1 - \beta_1^t) \\
\hat v_t &= v_t / (1 - \beta_2^t) \\
\theta_{t+1} &= \theta_t - \eta_t\,\Bigl(\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon} + w \cdot \theta_t\Bigr)
\end{aligned}
$$

with $w$ the weight decay coefficient and $\eta_t$ the current LR.

The implementation in `crates/lewm-train/src/optim.rs` wraps Burn's
`burn::optim::AdamWConfig` with parameter-group support.

## 2. The decay / no-decay split

Modern transformer training applies weight decay **only** to weight
matrices, not to biases, LayerNorm gains, or learned embeddings. The
split:

| Group | Decay weight | Examples |
|-------|--------------|----------|
| **Decay** | 0.05 (PushT) / 0.01 (SO-100) | All `Linear.weight`, `Conv1d.weight`, `Conv2d.weight`, `Attention.qkv.weight`, `Attention.proj.weight`, MLP weights. |
| **No decay** | 0.0 | All biases (`*.bias`), LayerNorm $\gamma$/$\beta$, `cls_token`, `position_embeddings`, predictor `pos_emb`, AdaLN modulation biases. |

The split is computed at trainer init time by walking the parameter
tree and dispatching by parameter name suffix. The implementation in
`crates/lewm-train/src/optim.rs`:

```rust,ignore
let (decay, no_decay): (Vec<_>, Vec<_>) = jepa.parameters_iter()
    .partition(|(name, _)| !matches!(
        name.suffix(),
        "bias" | "weight" if name.is_layer_norm()
              | "cls_token" | "position_embeddings" | "pos_emb"
    ));
```

This matches the upstream LeWM PyTorch optimizer setup and is the only
correct choice — applying decay to LayerNorm gains, in particular,
breaks parity and degrades convergence.

## 3. The schedule: linear warmup + cosine

```text
        ▲
   LR   │
        │           ╭──────────╮
   peak │          ╱            ╲
   3e-4 │         ╱              ╲
        │        ╱                ╲
        │       ╱                  ╲___
        │      ╱                       ╲___
   1e-5 │_____╱─ warmup ─                  ╲────  final
        │
        └─────────────────────────────────────────▶ step
                0       w               max_steps
```

The schedule is:

$$
\eta_t = \begin{cases}
\eta_{\max} \cdot \frac{t}{w}, & 0 \le t < w \quad \text{(linear warmup)} \\
\eta_{\min} + \frac{1}{2}\,(\eta_{\max} - \eta_{\min})\,\Bigl(1 + \cos\!\frac{(t - w)\pi}{T - w}\Bigr), & w \le t \le T \\
\eta_{\min}, & t > T
\end{cases}
$$

with:

| Symbol | PushT | SO-100 |
|--------|------:|-------:|
| $\eta_{\max}$ (peak) | 3.0e-4 | 3.0e-4 |
| $\eta_{\min}$ (final) | 1.0e-5 | 1.0e-5 |
| $w$ (warmup steps) | 1 000 | 500 |
| $T$ (total steps) | 50 000 | 5 000 |

The warmup ramps from 0 to $\eta_{\max}$ linearly over the first $w$
steps. The cosine then decays from $\eta_{\max}$ to $\eta_{\min}$ over
the remaining $T - w$ steps.

The implementation in `crates/lewm-train/src/schedule.rs`:

```rust,ignore
pub fn lr_at_step(&self, step: usize) -> f64 {
    if step < self.warmup {
        self.lr_max * (step as f64 / self.warmup as f64)
    } else if step <= self.total {
        let progress = (step - self.warmup) as f64 / (self.total - self.warmup) as f64;
        self.lr_min + 0.5 * (self.lr_max - self.lr_min)
            * (1.0 + (progress * std::f64::consts::PI).cos())
    } else {
        self.lr_min
    }
}
```

## 4. Gradient accumulation

PushT runs with `grad_accum_steps = 2` (effective batch = 64 × 2 =
128). SO-100 runs with `grad_accum_steps = 1`.

The trainer drives accumulation by zeroing the gradient buffer at the
beginning of each effective batch, accumulating $K$ micro-batch
gradients, and stepping the optimizer once. The `step` counter is
incremented once per **optimizer step**, not per micro-batch — so a
"50 000-step" PushT run is 100 000 micro-batches of forward + backward.

## 5. Gradient clipping

Every step, after accumulation:

```rust,ignore
let raw_norm = grads.l2_norm();          // pre-clip, for monitoring
if raw_norm > GRAD_NORM_CEILING {        // 1e3 — TOL-011
    abort_with_diagnostic(step, raw_norm);
}
grads = grads.clip_l2(1.0);              // standard transformer clip
```

The pre-clip norm is logged on every step; the clip is applied
unconditionally. In 50 000 + 5 000 training steps to date, TOL-011 has
not fired.

## 6. Putting it together: one step

```rust,ignore
// Inside step.rs, after backward:
optimizer.step(grads, &mut model);
scheduler.advance();                   // increment internal step
let lr_now = scheduler.lr();           // for telemetry
```

The optimizer wrapper handles the parameter-group split internally;
`scheduler.advance()` increments and `scheduler.lr()` returns the
current scalar LR for the next step.

## 7. Source pointers

| Topic | Source |
|-------|--------|
| Optimizer wrapper | `crates/lewm-train/src/optim.rs` |
| Scheduler | `crates/lewm-train/src/schedule.rs` |
| Decay-group split | `crates/lewm-train/src/optim.rs::partition_params` |
| Configs | `configs/pusht.toml`, `configs/so100.toml` |
| Burn AdamW | `burn::optim::AdamWConfig` |

## 8. Bibliography

- Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay
  Regularization* (AdamW). ICLR.
- Brown, T., et al. (2020). *Language Models are Few-Shot Learners*.
  NeurIPS — popularised the $\beta_2 = 0.95$ transformer convention.
