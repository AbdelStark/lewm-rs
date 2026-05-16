# AdaLN-zero conditioning

> **Motivation.** Conditioning a transformer on an external signal (an
> action, a class label, a timestep) is a classic problem. Naïve choices
> (concatenation, addition of a learned embedding) are inflexible and
> can destabilize training. AdaLN-zero — *Adaptive Layer Normalization
> with zero initialization* — solves it elegantly: at initialization the
> conditioning is *exactly the identity*, so adding it to a working
> model cannot break it.
>
> **Position.** Conceptual prelude to [The predictor](../architecture/predictor.md).
>
> **What you should leave with.** A clear understanding of what AdaLN-zero
> does, why "zero" matters, and exactly which gates and biases the
> predictor's modulation heads produce.

## 1. The conditioning problem

We have a transformer block of the standard pre-norm form,

```text
x ← x + Attention( LayerNorm(x) )
x ← x + MLP(       LayerNorm(x) )
```

and we want each block's behaviour to depend on an external vector
$\mathbf c$ — in LeWM, $\mathbf c$ is the action embedding for the
current time step. The challenge is to mix $\mathbf c$ in without:

- adding so many extra parameters that the block doubles in size,
- destabilizing early training by injecting a strong, random signal,
- making it impossible to *ablate* the conditioning (e.g. for an
  unconditional-prediction probe).

AdaLN (Perez et al., 2018; Peebles & Xie, 2023) is one popular answer.
It replaces the unconditional LayerNorm with a *conditional* one whose
scale $\gamma$ and shift $\beta$ are produced by a small MLP that
consumes $\mathbf c$:

```text
γ, β = MLP(c)                              # (D,), (D,)
LayerNorm(x; γ, β) = γ · (x - μ)/σ + β     # element-wise
```

This works but, at initialisation, $\gamma$ and $\beta$ are random, so
the conditional block has a different forward pass than the
unconditional one even at step 0. AdaLN-zero, introduced for diffusion
transformers (DiT) by Peebles & Xie (2023), fixes this with one trick:
**initialise the conditioning MLP's final layer to zero**.

## 2. AdaLN-zero, precisely

For each transformer block in the predictor, define:

- $\mathbf c \in \mathbb R^{d_c}$ — the conditioning vector (the action
  embedding at the current time step).
- $W^{\text{mod}} \in \mathbb R^{d_c \times 6D}$, $\mathbf b^{\text{mod}}
  \in \mathbb R^{6D}$ — a single linear layer producing 6 modulation
  parameters per block.
- The modulation parameters are split as

$$
[\gamma_1, \beta_1, \alpha_1,\; \gamma_2, \beta_2, \alpha_2] \;\leftarrow\; W^{\text{mod}}\,\mathbf c + \mathbf b^{\text{mod}},
$$

where $\gamma_i, \beta_i, \alpha_i \in \mathbb R^D$ for $i \in \{1, 2\}$.

The block then computes:

```text
y_attn ← Attention( γ_1 · LayerNorm(x) + β_1 )      # scale-and-shift LN, then attn
x      ← x + α_1 · y_attn                            # gated residual

y_mlp  ← MLP(        γ_2 · LayerNorm(x) + β_2 )      # scale-and-shift LN, then MLP
x      ← x + α_2 · y_mlp                             # gated residual
```

That is, AdaLN-zero gives every block three knobs for each of its two
sub-layers:

1. **scale** $\gamma$ — multiplies the LayerNorm-ed input,
2. **shift** $\beta$ — adds to the LayerNorm-ed input,
3. **gate** $\alpha$ — multiplies the sub-layer output before the
   residual addition.

The gate is the critical addition over plain AdaLN. It controls how much
of the sub-layer's output mixes into the residual stream, on a per-block,
per-condition basis.

## 3. The "zero" — what gets zero-initialised

The trick is to initialise $W^{\text{mod}}$ and $\mathbf b^{\text{mod}}$
to **zero**. At initialisation, then:

$$
\gamma_1 = \gamma_2 = 0,\quad \beta_1 = \beta_2 = 0,\quad \alpha_1 = \alpha_2 = 0.
$$

Substituting into the block:

```text
y_attn ← Attention( 0 · LayerNorm(x) + 0 ) = Attention(0) = 0   (after softmax(0) etc.)
x      ← x + 0 · y_attn = x

y_mlp  ← MLP( 0 · LayerNorm(x) + 0 ) = MLP(0) = 0
x      ← x + 0 · y_mlp = x
```

In other words: at initialisation, **every modulated block is the
identity**. The whole stack of $L = 6$ predictor blocks reduces to a
single pass-through of the input. This is the AdaLN-zero invariant.

The practical consequence is huge. We can drop AdaLN-zero into any
working unconditional transformer architecture, and at step 0 it
contributes *nothing*. The model is guaranteed to start in the same
place as the unconditional baseline. Then gradient descent, slowly, lets
$W^{\text{mod}}$ depart from zero and the conditioning starts to do
something.

## 4. What this looks like in LeWM

In LeWM's predictor, $\mathbf c$ is the action embedding for the current
time step:

$$
\mathbf c_t \;=\; \text{Embedder}(\mathbf a_t) \;\in\; \mathbb R^{D=192}.
$$

Each of the 6 predictor blocks has its own $W^{\text{mod}} \in \mathbb
R^{192 \times 1152}$ (since $6 \cdot D = 1152$). The modulation
parameters are produced **per block, per time step**: for $T = 3$
history frames, each block emits 6 different modulation vectors, one
per step.

The forward pass through one block looks like (Burn pseudocode):

```rust,ignore
let mod_params = self.ada_ln_modulation.forward(c);      // (B, T, 6D)
let [g1, b1, a1, g2, b2, a2] = mod_params.split_at_last_dim(D);  // each (B, T, D)

let ln1 = self.norm1.forward(x);                          // (B, T, D)
let modulated1 = ln1 * (1.0 + g1) + b1;                   // FiLM-style modulation
let attn_out = self.attention.forward(modulated1, mask);  // (B, T, D)
let x = x + a1 * attn_out;                                // gated residual

let ln2 = self.norm2.forward(x);
let modulated2 = ln2 * (1.0 + g2) + b2;
let mlp_out = self.mlp.forward(modulated2);
let x = x + a2 * mlp_out;
```

A small but important detail: the scale is applied as `(1 + γ) · x`,
not `γ · x`. This makes $\gamma = 0$ correspond to a *unit* scale, not a
*zero* scale. The identity invariant still holds: at init $\gamma = 0$,
so the scale factor is $1$; the shift is $0$; and the gate $\alpha = 0$
zeros out the sub-layer contribution. Together the block is the
identity.

## 5. Comparison with other conditioning schemes

| Scheme | Where used | Identity at init? | Parameter cost |
|--------|-----------|-------------------|----------------|
| **Concatenation** ($x \mathbin\Vert \mathbf c$ at the input) | Many early conditional models | No (architecture is different from unconditional) | Small (one extra input slot) |
| **Additive embedding** ($x \leftarrow x + W \mathbf c$ once at input) | Encoder-decoder transformers | No (linear shift everywhere) | Tiny (one weight matrix) |
| **FiLM** (Perez et al., 2018) | Visual reasoning, vision-language | No (random $\gamma, \beta$) | One MLP per FiLM site |
| **AdaLN** (Peebles & Xie, 2023, base case) | Diffusion transformers | No | One MLP per block, producing $2D$ per block |
| **AdaLN-zero** | DiT (Peebles & Xie, 2023); LeWM | **Yes** | One MLP per block, producing $6D$ per block |

LeWM's predictor is small enough (6 blocks × 1024-d MLP) that the
parameter cost of producing $6D$ modulation values per block — about
$192 \cdot 1152 = 221\,184$ weights per block — is acceptable in
exchange for the initialisation guarantee.

## 6. The numerical contract

Two contracts in [RFC 0002] are about AdaLN-zero specifically:

1. **RFC0002-AdaLN-Init [MUST]**: the linear weight $W^{\text{mod}}$ and
   bias $\mathbf b^{\text{mod}}$ of the modulation head of every
   `ConditionalBlock` are initialised to the *all-zero matrix and the
   zero vector* at the end of `Module::init`.
2. **RFC0002-AdaLN-Order [MUST]**: the modulation vector is split into
   $[\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2]$ in that
   exact order along the last dim. This ordering matches the upstream
   `module.py::ConditionalBlock` source line-for-line; any other
   ordering breaks parity.

The parity test [`parity_predictor`] verifies the predictor's output
matches the PyTorch reference to $L_\infty < 10^{-4}$ across all three
time steps — which is only possible if both AdaLN-zero contracts are
satisfied.

## 7. Why this matters for the LeWM training story

The full LeWM model has 18 M parameters split across the encoder (5.5 M),
the predictor (10.5 M), the action encoder (0.2 M), and the projectors
(1.8 M). At init, the AdaLN-zero predictor degenerates to the identity,
which means the *prediction loss* at step 0 has the form

$$
\mathcal L_{\text{pred}}^{(0)} \;=\; \big\lVert \text{pred\_proj}(\mathbf z_{t-1}^{\text{source}}) - \mathbf z_t^{\text{target}} \big\rVert^2.
$$

The predictor is the identity; only the projectors (`projector` and
`pred_proj`) and the encoder shape this loss. This gives the encoder
clean SIGReg + a small "be predictable across nearby frames" signal in
the very first steps, **without** the gradient being dominated by random
predictor outputs. As training proceeds and $W^{\text{mod}}$ moves away
from zero, the predictor's contribution gradually takes over.

Empirically (see [PushT results](../results/pusht.md)), the prediction
loss stays close to its initial value $\sim 6\times 10^{-4}$ for the
first 500 steps while SIGReg falls by an order of magnitude. After that,
the predictor "wakes up" and both losses drive down together.

## 8. Bibliography

- Perez, E., Strub, F., De Vries, H., Dumoulin, V., Courville, A. (2018).
  *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI.
- Peebles, W., Xie, S. (2023). *Scalable Diffusion Models with
  Transformers* (DiT). ICCV. **The source of AdaLN-zero.**
- Karras, T., Aittala, M., Aila, T., Laine, S. (2022). *Elucidating the
  Design Space of Diffusion-Based Generative Models*. NeurIPS — analysis
  of why zero-initialised conditioning is well-behaved.

[RFC 0002]: ../reference/rfcs.md
[`parity_predictor`]: https://github.com/AbdelStark/lewm-rs/blob/main/crates/lewm-core/tests/parity_predictor.rs
