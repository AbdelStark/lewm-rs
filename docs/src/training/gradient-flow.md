# Gradient flow and end-to-end stability

> **Motivation.** "End-to-end training, no EMA, no stop-gradient" is
> the single most distinctive design choice in LeWM. It is also the
> most non-obvious in why it works. This page draws the gradient graph
> in detail.
>
> **Position.** Third sub-page in [Part III](./pipeline.md).
>
> **What you should leave with.** A clear picture of which modules
> receive which gradients, what would go wrong without SIGReg, and
> what the empirical record shows about stability.

## 1. The computational graph

For one micro-batch, the autograd graph that backpropagation traverses
is:

```text
   pixels (B, T+1, 3, 224, 224)
      │
      ▼
   encoder  ────────────────────────────────┐
      │                                     │  z[:, 0..T]                 z[:, 1..T+1]
      │                                     ▼                              │
      │                                projector                           │
      │                                     │                              │
      │                                     ▼  source-arm history          │
      │                                                                    │
   actions ──▶ action_enc ──▶ predictor (consumes history)                  │
                                  │                                        │
                                  ▼                                        │
                              pred_proj                                    │
                                  │                                        │
                                  ▼  source-arm prediction                  │
                                                                           ▼
                              ┌────────── L_pred = MSE ──────────┐ ◀── target-arm = projector(encoder(z[:, 1..T+1]))
                              │                                  │
                              └──── L_total = L_pred + λ·L_sigreg ◀── L_sigreg = SIGReg(z_proj[:, 0..T+1])
```

Every arrow in this graph carries gradient backwards. There are
**no stop-gradients** anywhere in the diagram.

## 2. Per-module gradient sources

| Module | Receives gradient from |
|--------|------------------------|
| `vit` (encoder) | $\mathcal L_{\text{pred}}$ via source arm; $\mathcal L_{\text{pred}}$ via target arm; $\mathcal L_{\text{sigreg}}$ via target arm. |
| `projector` | Same three paths as the encoder. |
| `action_enc` | $\mathcal L_{\text{pred}}$ via source arm. |
| `predictor` | $\mathcal L_{\text{pred}}$ via source arm. |
| `pred_proj` | $\mathcal L_{\text{pred}}$ via source arm. |

The encoder and `projector` are special — they get gradient from
**three** independent paths on every step. This is what justifies
"end-to-end".

## 3. Why this works (and earlier JEPAs didn't dare try it)

Earlier JEPAs (I-JEPA, V-JEPA) use an EMA-tracked target encoder. The
loss has the form

$$
\mathcal L = \big\lVert g_\phi\bigl(f_\theta(x), c\bigr) - f_{\bar\theta}(y) \big\rVert^2,
$$

where $\bar\theta$ is an exponential moving average of $\theta$ and the
gradient on the target arm is severed (`y_target = f_target.forward(y)
.detach()`). The reasons cited in the I-JEPA / V-JEPA papers:

1. **Symmetric trivial solution.** If $\bar\theta = \theta$ and both
   arms get gradient, the model can drive both sides toward the same
   collapsed point and minimise the loss to zero with $f_\theta \equiv$
   constant. The EMA + stop-grad asymmetry breaks this symmetry.
2. **Training instability.** Without the asymmetry, the loss can
   oscillate or diverge.

LeWM's contribution is observing that **SIGReg also breaks the symmetry
*and* prevents collapse** — and does so more directly. SIGReg pulls the
encoder's output distribution toward $\mathcal N(0, I_{1024})$, which is
*non-degenerate by definition*. There is no way for $f_\theta \equiv$
constant to coexist with $\mathcal L_{\text{sigreg}} \to 0$, because a
delta-function distribution has trivially nonzero distance from the
standard normal in characteristic-function space.

With collapse prevented by SIGReg, the EMA and stop-gradient become
unnecessary. The training is *symmetric*: both arms use the same
encoder, the gradient flows through both, and the parameter update is
consistent across the two paths.

## 4. What the empirical record shows

From the PushT 50 k-step training run:

- **Zero gradient explosions** across 50 000 steps. Pre-clip gradient
  norm stayed below the $10^3$ ceiling at every step.
- **Smooth loss trajectory.** No oscillation, no divergence, no spikes.
  The cosine LR + warmup schedule sees a clean monotonic-ish descent
  to the noise floor.
- **No collapse probe trips.** The encoder's per-dim variance stayed
  above 0.05; mean abs CLS below 5; pairwise cosine below 0.85
  throughout.

This is consistent with LeWM's design claim that *SIGReg is sufficient
to stabilise end-to-end training*.

## 5. The role of AdaLN-zero in stabilising early training

A second mechanism contributes to the smoothness of the trajectory:
**AdaLN-zero**.

At init, every `ConditionalBlock` in the predictor is the identity.
That means the predictor's output at step 0 is exactly:

```text
predictor(history) at init = output_proj(final_norm(pos_emb + input_proj(history)))
                           = LayerNorm(linear projection)
```

There is no random transformer output yet. The prediction loss at
step 0 is therefore well-behaved: pred is a deterministic, low-norm
function of history, and the gradient is *also* well-behaved. As the
AdaLN-zero modulation heads slowly depart from zero (over the first
~1 000 steps), the predictor "wakes up" and the prediction loss starts
to drop. The transition is gradual rather than sudden, which is
exactly what we want for stability.

In short: SIGReg prevents the long-term collapse mode; AdaLN-zero
prevents the short-term random-init mode. Together they make end-to-end
training of an 18 M-parameter transformer world model feasible without
the EMA + stop-gradient apparatus.

## 6. What goes wrong if you remove a piece

We have not run the ablations, but the design predicts:

- **Remove SIGReg, keep EMA + stop-gradient.** This is essentially
  V-JEPA. It works but needs the EMA half-life and the stop-grad
  placement to be right. More knobs, similar quality.
- **Remove SIGReg, keep end-to-end (no EMA).** Collapse to a constant
  in ~200 steps. The asymmetric stabiliser is gone *and* the
  distributional stabiliser is gone.
- **Keep SIGReg, add stop-gradient on target.** Works but loses one
  gradient signal into the encoder. Slower convergence, likely lower
  final loss floor.
- **Remove AdaLN-zero, use plain AdaLN.** Random predictor outputs at
  init, large prediction-loss gradients early, possible training
  instability or slower warmup.

These predictions are consistent with the LeWM paper's reported
ablations. lewm-rs does not re-run them; the design choices are taken
as given.

## 7. The grad-clipping safety net

Even with all the design choices above, the training loop applies
gradient clipping at $\lVert g \rVert_2 \le 1.0$ on every step. This
is a belt-and-braces safety mechanism; in 50 000 + 5 000 steps of
training, the clip was applied only when the natural gradient norm
already happened to be close to 1.0, never as a rescue from divergence.

The pre-clip norm is the *monitored* quantity. A pre-clip norm > $10^3$
trips TOL-011 and aborts the run with a diagnostic. This has never
fired.

## 8. Source pointers

| Topic | Source |
|-------|--------|
| Training step (where backward happens) | `crates/lewm-train/src/step.rs` |
| Grad clip implementation | `crates/lewm-train/src/optim.rs` |
| Grad-norm probe | `crates/lewm-train/src/step.rs` |
| Burn's autograd backend | `burn::backend::Autodiff<B>` |
