# Symbol conventions

The canonical math / tensor symbol table used across the docs, the
specs, and the source. Identical to
[`specs/glossary.md` §6](https://github.com/AbdelStark/lewm-rs/blob/main/specs/glossary.md#6-symbol-conventions).

## Tensor shape symbols

| Symbol | Meaning | Common value |
|:------:|---------|--------------|
| $B$ | Batch dimension | 64 (micro), 128 (effective) |
| $T$ | Temporal dimension (frames in a window) | 3 history + 1 next |
| $H, W$ | Image height and width in pixels | 224, 224 |
| $C$ | Channel dimension | 3 (RGB) |
| $D$ | Embedding dim (encoder hidden size) | 192 (PushT, locked) |
| $K$ | Number of random projections in SIGReg | 1024 |
| $J$ | Number of frequency knots in SIGReg | 17 |
| $t$ | Within-window time index | $t \in [0, T)$ |
| $\lambda$ (lambda) | SIGReg loss weight | 1.0 (default) |

## Action symbols

| Symbol | Meaning | Common value |
|:------:|---------|--------------|
| $A$ | Raw action dim | 2 (PushT) / 6 (SO-100) |
| $A_p$ | Packed action dim after frameskip Conv1d | 10 |
| $E_a$ | Action embedding dim after Embedder MLP | 192 (matches $D$) |

## Loss symbols

| Symbol | Meaning |
|:------:|---------|
| $\mathcal L_{\text{pred}}$ | Prediction loss (MSE) |
| $\mathcal L_{\text{sigreg}}$ | SIGReg loss |
| $\mathcal L$ | Total loss: $\mathcal L_{\text{pred}} + \lambda\,\mathcal L_{\text{sigreg}}$ |

## Optimizer symbols

| Symbol | Meaning | Value |
|:------:|---------|-------|
| $\eta_{\max}$ | Peak LR | $3 \times 10^{-4}$ |
| $\eta_{\min}$ | Final LR | $1 \times 10^{-5}$ |
| $\beta_1, \beta_2$ | AdamW EMA factors | $0.9, 0.95$ |
| $w$ | Warmup steps | 1 000 (PushT) / 500 (SO-100) |
| $T$ (schedule) | Total steps | 50 000 (PushT) / 5 000 (SO-100) |

## Rust type symbols

| Symbol | Meaning |
|:------:|---------|
| `B: Backend` | Burn backend type parameter |
| `B::Device` | The backend's device handle |
| `B::FloatElem` | The backend's default float element (F32 / BF16) |

## Notational conventions

- All tensor shapes are written `(B, T, D)`-style throughout, matching
  the PyTorch / Burn convention.
- All "norms" without subscript are L2 (Frobenius) norms.
- $\lVert \cdot \rVert_\infty$ is the elementwise max-abs (L∞) norm.
- $\mathbb E_b[\cdot]$ denotes empirical mean over the batch axis.
- $\mathcal N(\mu, \sigma^2)$ is the normal distribution; $\mathcal
  N(\mathbf 0, I_D)$ is the standard $D$-variate normal.
- $\Phi(x)$ is the standard-normal CDF; $\mathrm{erf}(x)$ is the error
  function; $\Phi(x) = (1 + \mathrm{erf}(x/\sqrt 2))/2$.
