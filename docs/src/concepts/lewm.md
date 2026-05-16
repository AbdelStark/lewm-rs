# LeWorldModel: the specialization

> **Motivation.** "JEPA" is a principle. "LeWorldModel" is a precise
> instantiation. This page enumerates every commitment LeWM makes on top
> of the general JEPA recipe and explains why each commitment is there.
>
> **Position.** Everything from here down — architecture, training,
> planning, inference — implements the choices listed below. If you
> understand this page, you understand the *shape* of every later page.
>
> **What you should leave with.** A one-screen summary of LeWM and an
> appreciation of which knobs are pinned and which are free.

## 1. The thirty-second summary

LeWM is an action-conditioned visual world model with the following
fixed structure:

```text
                pixels (B, T, 3, 224, 224)
                          │
                          ▼
                ┌──────────────────┐
                │  ViT-Tiny encoder │  D=192, 12 layers, 3 heads
                │  (HF-compatible)  │
                └──────────────────┘
                          │
                z_target  ▼  z_source
                ┌──────────────────┐
                │     projector     │  Linear→GELU→Linear, MLP
                │  (proj_dim=1024)  │
                └──────────────────┘
                          │  z̃   (B, T, 1024)
                          │
                          │            actions (B, T-1, A)
                          │                 │
                          │                 ▼
                          │   ┌─────────────────────┐
                          │   │   Conv1d smoother    │  A → 10
                          │   │   + SiLU MLP (×4)    │  10 → 192
                          │   └─────────────────────┘
                          │                 │  a_emb
                          │                 ▼
                          │   ┌─────────────────────┐
                          └──▶│  ArPredictor         │  AdaLN-zero
                              │  6 blocks, 16 heads  │  (a_emb modulates)
                              │  MLP=2048            │  causal mask
                              └─────────────────────┘
                                         │
                                         ▼
                              ┌──────────────────┐
                              │   pred_proj       │  Linear→GELU→Linear
                              │   (D=192→1024)    │
                              └──────────────────┘
                                         │
                                         ▼   ẑ_next (B, T, 1024)
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                             ▼
                  L_pred = MSE(ẑ_next, z̃[:,1:])   L_sigreg = SIGReg(z̃)

                                Total: L = L_pred + λ · L_sigreg
```

Every block in this diagram is specified to byte-exact reproducibility in
[RFC 0002] and the [Architecture](../architecture/overview.md) section of
these docs.

## 2. The seven commitments

Going around the diagram in order, LeWM makes the following commitments
that distinguish it from "JEPA in general":

### 2.1 The encoder is HF-style ViT-Tiny.

- 224 × 224 RGB input, 14 × 14 patches → 256 patches + 1 learned CLS.
- 12 transformer blocks, 3 attention heads, hidden dim $D = 192$.
- Exact-erf GELU (not the fast tanh approximation).
- LayerNorm $\varepsilon = 10^{-12}$ (not the PyTorch default $10^{-5}$).
- No dropout at LeWM scales.
- The "embedding of a frame" is the CLS token at the output of the 12th
  block.

These choices match Hugging Face's `ViTConfig` defaults (other than the
LayerNorm epsilon and the GELU variant, which match the upstream LeWM
checkpoint specifically). Parity to upstream depends on getting **both**
the eps and the GELU right — see
[Implementation gotchas](../parity/gotchas.md).

### 2.2 Actions are smoothed before they enter the predictor.

Raw actions are low-dimensional (2-D for PushT, 6-D for SO-100). They
arrive at sample rate (10–30 Hz). LeWM's predictor consumes them at the
same step rate as the latent stream ($T = 3$ history frames), so the raw
actions must first be **collapsed across frameskip**:

```text
raw action (A,) ──Conv1d(kernel=5, stride=1)──▶ smoothed (10,)
                                                      │
                                                      ▼
                                         ┌─────────────────────┐
                                         │  Linear → SiLU       │  10 → 4·D = 768
                                         │  Linear              │  768 → D = 192
                                         └─────────────────────┘
                                                      │
                                                      ▼ action embedding (D,)
```

The Conv1d acts as a fixed-resampling-kernel smoother; the MLP lifts the
smoothed 10-D signal to the embedding dimension $D = 192$ so it can be
broadcast into the predictor's AdaLN modulation heads.

### 2.3 The predictor is an autoregressive transformer with AdaLN-zero.

- 6 transformer blocks, 16 attention heads, MLP inner dim 2048.
- Causal mask (upper-triangular boolean) pre-registered as a buffer.
- Each block uses **AdaLN-zero**: the action embedding $a_t$ is mapped
  through a zero-initialised linear head to produce per-block scale,
  shift, and gate parameters. At initialisation, every modulated block
  acts as the identity, so the predictor is initially well-behaved.
- Output dimension matches the encoder's: $(B, T, D)$.

See [The autoregressive predictor](../architecture/predictor.md) and
[AdaLN-zero conditioning](./adaln.md).

### 2.4 There is no EMA, no stop-gradient on the encoder.

This is the most important design departure from I-JEPA / V-JEPA. In
those works, the loss has the form

$$
\mathcal L = \big\lVert g_\phi(f_\theta(x), c) - f_{\bar\theta}(y) \big\rVert^2
$$

where $\bar\theta$ is an EMA copy of $\theta$ and the gradient on the
target arm is severed. In LeWM, $\bar\theta = \theta$, both arms share
the same encoder, and **gradient flows through both copies**. This is
made possible by SIGReg, which gives the encoder enough regularization
that the symmetric collapse mode ($f_\theta \equiv$ constant) is
explicitly penalised.

The consequence: every step of training updates the encoder using the
combined signal from the prediction loss (through both the source and
target arms) and the SIGReg loss. See
[Gradient flow](../training/gradient-flow.md).

### 2.5 The projector lifts to a wider space for SIGReg.

The encoder produces $(B, T, D=192)$. Before SIGReg, a projector MLP lifts
this to $(B, T, \text{proj\_dim} = 1024)$. This is where the SIGReg loss
lives. The 1024-D space gives SIGReg's $K = 1024$ random directions room
to be approximately orthogonal.

The same projector also produces the **target** of the prediction loss:
the loss compares `pred_proj(predictor(projector(...)))` (the source
arm's output, lifted by both projectors) to `projector(encoder(y))` (the
target arm's output, lifted by the source projector only). See
[Loss functions](../training/losses.md) for the exact loss equation.

### 2.6 The regularizer is SIGReg, not VICReg, not Barlow.

SIGReg — the *Sketch Isotropic Gaussian Regularizer* — measures how far
the empirical distribution of the projected latents is from the standard
normal $\mathcal N(0, I_d)$, using a windowed Epps–Pulley statistic on
$K = 1024$ random directions and $J = 17$ frequency knots. This is
specified in full in [RFC 0003] and discussed in
[SIGReg deep dive](./sigreg.md).

Algorithmically, SIGReg is one equation. Numerically, it is fiddly: it
must be computed in F32 (not BF16) to keep the high-frequency trigonometric
terms stable, and the random projection matrix must be re-sampled every
call from a named RNG sub-stream so resume reproduces the same loss.

### 2.7 Planning uses CEM on the predictor.

The world model serves a downstream controller. The controller is CEM —
Cross-Entropy Method — over the predictor's latent rollout, with the
cost given by

$$
\text{cost}(\mathbf{a}_{1:H}) \;=\; \big\lVert \hat{\mathbf z}_H - \mathbf z_{\text{goal}} \big\rVert^2_2,
$$

where $\hat{\mathbf z}_H$ is the predictor's rollout after $H$ steps from
the current latent under candidate action sequence $\mathbf{a}_{1:H}$, and
$\mathbf z_{\text{goal}}$ is the encoder's embedding of a goal image.
The simplicity of this cost — "predict and compare" — is the payoff of
all the JEPA setup. See [Planning with CEM](../planning/cem.md).

## 3. What is fixed vs. tunable

The architecture, the loss, and the determinism contract are **fixed**:
they are specified in RFC 0002, 0003, and 0013 respectively, and changing
them requires an ADR. The remaining knobs — listed below — are tunable
per dataset and run.

| Knob | Default | Where pinned |
|------|---------|--------------|
| $T$ (history frames) | 3 | RFC 0002 §4.7 |
| Batch size | 64 | `configs/pusht.toml`, `configs/so100.toml` |
| Grad accum | 2 (PushT), 1 (SO-100) | configs |
| Steps | 50 000 (PushT), 5 000 (SO-100) | configs |
| Peak LR / final LR | 3e-4 / 1e-5 | configs |
| Warmup steps | 1 000 (PushT), 500 (SO-100) | configs |
| Weight decay | 0.05 (PushT), 0.01 (SO-100) | configs |
| $\lambda$ (SIGReg weight) | 1.0 | RFC 0003 §4.2.3 |
| Seed | 0 | RFC 0013 |

## 4. The promise of this docsite

The rest of these documents is a guided tour through the seven commitments
above. By the end of [Part II](../architecture/overview.md) you should be
able to draw the diagram from §1 from memory. By the end of
[Part III](../training/pipeline.md) you should be able to step through
the training loop in your head. By the end of [Part IV](../planning/cem.md)
you should be able to plan a 5-step manipulation rollout on a CPU.

[RFC 0002]: ../reference/rfcs.md
[RFC 0003]: ../reference/rfcs.md
