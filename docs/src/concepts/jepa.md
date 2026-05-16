# Foundations of JEPA

> **Motivation.** Before we can describe LeWorldModel, we need to be precise
> about what *Joint-Embedding Predictive Architectures* are and what design
> pressures motivated them. JEPA is not a model — it is a **principle** for
> structuring self-supervised representation learners.
>
> **Position.** This is the conceptual root of the docsite. Everything that
> follows specializes JEPA in some direction.
>
> **What you should leave with.** A working definition of JEPA, the meaning
> of "predicting in latent space", a list of failure modes that JEPA
> deliberately rules out, and a map of the JEPA family tree up to LeWM.

## 1. The principle, stated once

Let $x$ be an observation (an image, a video clip, a sensor record). Let
$y$ be a *related* observation — a temporally adjacent frame, a masked
patch, a future state. Let $c$ be a *context* — an action taken between
$x$ and $y$, a positional offset, a viewpoint change. The JEPA principle
says:

> **Learn an encoder $f_\theta$ such that, for related $(x, y, c)$, the
> embedding $f_\theta(y)$ is predictable from $f_\theta(x)$ and $c$.**

Concretely, we train a predictor $g_\phi$ jointly with $f_\theta$ to
minimise a distance in the latent space:

$$
\min_{\theta, \phi}\; \mathbb{E}_{(x, y, c)\,\sim\,\mathcal D}\,\Big\lVert g_\phi\!\bigl(f_\theta(x),\, c\bigr) - f_\theta(y) \Big\rVert^2_2.
$$

The crucial design choice is the *space in which the loss lives*: the loss
is **not** on pixels, **not** on reconstructions, **not** on contrastive
samples. It is on the encoder's own outputs.

This single move buys a lot.

## 2. Why predict in latent space

Pixel-space prediction has a long and difficult history. Two problems
dominate:

1. **Aleatoric noise.** The pixels of "the next frame" are not a function
   of "the current frame and the action taken". They depend on lighting,
   sensor noise, motion blur, jpeg artefacts, the exact phase of any
   periodic motion, and a hundred other latent factors. A pixel-MSE
   objective spends most of its capacity modelling things that are
   strictly irrelevant for control.
2. **The blur problem.** Under quadratic loss, the optimal prediction in
   the face of multimodal aleatoric noise is the *mean* of the modes. The
   model therefore renders blurred futures, smudges of all possible
   outcomes superposed. This is empirically observed in pixel-prediction
   world models from PixelRNN through to early VQ-VAE based systems.

Latent prediction sidesteps both. The encoder is free to **discard** the
parts of $y$ that are not predictable from $(x, c)$. The predictor's job
is to model the predictable part of the latent — a much lower-dimensional,
much more deterministic signal.

The trade-off is that we now have to **prevent the encoder from
collapsing**. A constant encoder $f_\theta(\cdot) = \mathbf z_0$ trivially
satisfies the prediction objective ($g_\phi$ outputs $\mathbf z_0$ as
well). So a JEPA objective always pairs the prediction loss with some
form of regularization that forbids collapse.

## 3. The JEPA family tree

The JEPA principle is implemented several different ways. The differences
boil down to (a) **what counts as "related"** and (b) **how collapse is
prevented**.

| Method | Relation $(x, y, c)$ | Collapse prevention |
|--------|----------------------|---------------------|
| **I-JEPA** (Assran et al., 2023) | $x$ is a context patch grid of an image; $y$ is a held-out target patch grid; $c$ is the position offset. | Stop-gradient on the target encoder + EMA-tracked target weights. |
| **V-JEPA** (Bardes et al., 2024) | $x$ is a masked video clip; $y$ is the masked region; $c$ is implicit. | Same as I-JEPA: EMA target + stop-gradient. |
| **DINOv2 / iBOT family** (related, not strictly JEPA) | $x, y$ are two augmentations of the same image; $c$ is implicit. | Centring / sharpening + EMA. |
| **LeWorldModel** (Maes et al., 2026) | $x$ is a history of past frames; $y$ is the next frame; $c$ is the action taken. | **SIGReg**: a single regularizer pulling the latent distribution toward $\mathcal N(0, I_D)$. **No EMA, no stop-gradient**. |

LeWM's contribution to this family is its **simplification**. Earlier JEPAs
use an asymmetric architecture (online encoder + EMA target encoder) plus
a stop-gradient on the target arm. This works but is operationally heavy:
EMA half-life is yet another hyperparameter, the asymmetry can introduce
subtle bugs when checkpoints are saved or evaluated, and the stop-gradient
must be carefully placed so it does not accidentally cut off useful
signal. LeWM replaces this whole edifice with one regularizer, **SIGReg**,
which constrains the encoder's output distribution directly. The training
becomes **end-to-end** in the strict sense: every parameter receives a
gradient on every step, and the only stop-gradient in the system is the
one inside the SIGReg sketch (and even that is on the projection matrix,
not on a network).

LeWM's other contribution — and the reason it shows up in robotics venues
— is that $c$ is an **action**, not a positional embedding. The encoder is
forced to learn representations that are *consistent under action
conditioning*: it must preserve the information the predictor needs to
read off the next-step embedding, while compressing away anything the
action does not control.

## 4. What JEPA is not

It is useful to be explicit about what JEPA does **not** do, because
adjacent self-supervised paradigms make different commitments:

- **JEPA is not generative.** It does not learn $p(x)$ or $p(y \mid x)$.
  There is no decoder; the latent embedding need not contain enough
  information to reconstruct the input.
- **JEPA is not contrastive (in the InfoNCE sense).** There is no notion
  of "positives vs negatives" in a JEPA loss. The collapse mechanism is
  a regularizer (SIGReg, or EMA + stop-grad), not a contrastive view.
- **JEPA is not strictly self-supervised in the masked-prediction sense.**
  When $c$ is an action, JEPA is *action-conditioned representation
  learning*, which sits between SSL and model-based RL.

## 5. What you get when JEPA works

When the principle is realized correctly — encoder, predictor, and an
anti-collapse mechanism in balance — three useful properties emerge:

1. **Compact representations.** The encoder retains only what the
   predictor needs. For LeWM, that turns out to be a 192-D vector per
   frame, which is enough to plan a 5-step manipulation sequence over
   1024 action candidates on a CPU in seconds.
2. **Sample-efficient learning.** Latent-space prediction does not waste
   capacity on aleatoric pixel noise, so it converges faster than a pixel-
   prediction objective on the same data budget.
3. **Plannable rollouts.** The predictor is differentiable and cheap to
   call. Planning algorithms like CEM can sample thousands of action
   sequences, run them through the predictor, and score the resulting
   latent against a goal embedding — all without rendering a single
   pixel.

The rest of the [Concepts](./lewm.md) section unpacks each of these.
The next page, [LeWM specialization](./lewm.md), narrows from
"JEPA in general" to "the precise instantiation that lewm-rs reproduces".

## 6. Bibliography

- LeCun, Y. (2022). *A path towards autonomous machine intelligence*. OpenReview.
  The original argument for JEPA as a path to non-generative world models.
- Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M.,
  LeCun, Y., Ballas, N. (2023). *Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture*. CVPR.
- Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M.,
  Ballas, N. (2024). *V-JEPA: Latent Video Prediction for Visual Representation
  Learning*. arXiv:2404.08471.
- Maes, L., Le Lidec, Q., Scieur, D., Balestriero, R., LeCun, Y. (2026).
  *LeWorldModel: Learning World Models in Latent Space*. arXiv:2502.16560.

Continue to: [LeWorldModel: the specialization](./lewm.md).
