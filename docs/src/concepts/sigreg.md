# SIGReg: sketch-isotropic Gaussian regularization

> **Motivation.** SIGReg is the entire collapse-prevention strategy of
> LeWM. Get it wrong and the encoder collapses to a constant in 200
> steps. Get it numerically wrong (BF16 high-frequency trig drift) and
> the same thing happens, only slower. This is the chapter where we
> dwell on the math until it is comfortable.
>
> **Position.** Conceptual root for [Loss functions](../training/losses.md),
> [Gradient flow](../training/gradient-flow.md), and the SIGReg parity
> test in [Parity](../parity/tests.md).
>
> **What you should leave with.** A working understanding of what the
> Epps–Pulley statistic measures, why a sketched version is used, and
> what the trapezoid-with-Gaussian-window in $[0, 3]$ buys us.

## 1. The problem SIGReg solves

Recall the prediction loss:

$$
\mathcal L_{\text{pred}}(\theta, \phi) \;=\; \mathbb{E}_{(x, y, \mathbf a)} \big\lVert g_\phi(f_\theta(x), \mathbf a) - f_\theta(y) \big\rVert^2_2.
$$

This is solved trivially by

$$
f_\theta \equiv \mathbf z_0,\quad g_\phi \equiv \mathbf z_0
$$

(both arms learn the constant function). Such a model has minimum loss
and is useless. We need a side-constraint on the encoder that excludes
the constant function — and, more broadly, that pulls the encoder's
output distribution toward a *non-degenerate* shape.

Different JEPAs answer this differently. SIGReg's answer is:

> **The encoder's projected output distribution should look like a
> standard normal $\mathcal N(0, I_d)$.**

This is a *distributional* constraint, not a contrastive one. It says
nothing about which sample maps where; it only says that the marginal
distribution over the batch should be Gaussian-shaped.

## 2. The Epps–Pulley statistic

The classical goodness-of-fit test for normality used by SIGReg is the
**Epps–Pulley test** (Epps & Pulley, 1983). It compares the empirical
characteristic function of a sample to the standard-normal characteristic
function.

The characteristic function of a real random variable $X$ is
$\psi_X(t) = \mathbb E[e^{itX}]$. For $X \sim \mathcal N(0, 1)$, this is

$$
\psi_{\mathcal N}(t) \;=\; e^{-t^2/2}.
$$

The real and imaginary parts are $\Re\psi = e^{-t^2/2}$ and $\Im\psi = 0$.
Given a sample $\{x_1, \dots, x_N\}$ in $\mathbb R$, the empirical
characteristic function is

$$
\psi_N(t) \;=\; \frac{1}{N}\sum_{n=1}^N e^{itx_n}
\;=\; \underbrace{\frac{1}{N}\sum_n \cos(t x_n)}_{c(t)}
\;+\; i\underbrace{\frac{1}{N}\sum_n \sin(t x_n)}_{s(t)}.
$$

The Epps–Pulley statistic at a single frequency $t$ is the squared
distance between empirical and target characteristic functions in the
complex plane:

$$
r(t) \;=\; \bigl(c(t) - e^{-t^2/2}\bigr)^2 \;+\; s(t)^2.
$$

The full Epps–Pulley test integrates $r(t)$ against a Gaussian weight
$w(t) = e^{-t^2/2}$ over $t \in [0, \infty)$. The weight makes the
integral finite and emphasizes low-frequency information (large-scale
distributional shape) over high-frequency information (which is
dominated by noise on a finite sample).

SIGReg discretizes this integral on a finite grid $\{t_0, \dots, t_{J-1}\}$
in $[0, t_{\max}]$ and uses trapezoid quadrature.

## 3. From 1-D to $D$-D: the sketch

The Epps–Pulley test is *univariate*. In LeWM the encoder's projected
output lives in $\mathbb R^{1024}$. Estimating a multivariate empirical
characteristic function in 1024 dimensions from a batch of 64–128 samples
is statistically hopeless.

SIGReg's trick is to project the high-dimensional distribution onto $K =
1024$ random one-dimensional directions and apply the univariate test to
each:

$$
\mathbf p_k \;\sim\; \mathcal N(\mathbf 0, I_D/D),\qquad
\mathbf p_k \;\leftarrow\; \mathbf p_k\,/\,\lVert \mathbf p_k\rVert
\quad\text{(unit-norm rows)}.
$$

For sample $\mathbf z_n \in \mathbb R^D$, the projection along direction
$\mathbf p_k$ is the scalar $y_{k,n} = \langle \mathbf p_k, \mathbf z_n
\rangle$. The univariate Epps–Pulley statistic on the sample $\{y_{k,n}\}_n$
tests whether the marginal along $\mathbf p_k$ is standard-normal. We
average over $k$ to obtain the SIGReg loss.

The intuition is the **Cramér–Wold theorem**: the distribution of a random
vector is determined by all of its 1-D projections. We can't check all of
them — there are uncountably many — but we can check $K = 1024$ random
ones, and that turns out to be enough in practice.

The directions $\mathbf p_k$ are **resampled every call**. This gives
SIGReg the stochastic-projection flavour of the original sketch, makes
the gradient less collinear across batches, and prevents adversarial
solutions in any fixed subspace.

## 4. The exact equation

Putting it together, given a batch of projected latents $\mathbf Z \in
\mathbb R^{N \times D}$ (where $N = B \cdot T$ flattens the batch and time
axes):

$$
\boxed{\;
\mathcal L_{\text{sigreg}}(\mathbf Z) \;=\; \frac{1}{K}\sum_{k=1}^{K}\;\sum_{j=0}^{J-1} q_j\, w(t_j)\, \Bigl[\bigl(c_{k,j} - \phi(t_j)\bigr)^2 + s_{k,j}^2\Bigr]
\;}
$$

with

$$
c_{k,j} = \frac{1}{N}\sum_n \cos\!\bigl(t_j\,\mathbf p_k^\top \mathbf z_n\bigr),\qquad
s_{k,j} = \frac{1}{N}\sum_n \sin\!\bigl(t_j\,\mathbf p_k^\top \mathbf z_n\bigr),
$$

$$
\phi(t) = w(t) = e^{-t^2/2},\qquad t_j = \frac{j}{J-1}\, t_{\max},\quad j = 0,\dots,J-1.
$$

The trapezoid weights on a uniform grid with spacing $\Delta t = t_{\max}/(J-1)$
are $q_0 = q_{J-1} = \Delta t / 2$, $q_j = \Delta t$ otherwise.

LeWM's defaults: $K = 1024$, $J = 17$, $t_{\max} = 3$. These match the
upstream `module.py::SIGReg.__init__`.

## 5. Why $t_{\max} = 3$ and $J = 17$?

The Gaussian window $w(t) = e^{-t^2/2}$ decays to $e^{-4.5} \approx 0.011$
at $t = 3$, so the integrand has effectively zero weight beyond $t_{\max}
= 3$. Pushing the upper limit further would only add noise.

The knot count $J = 17$ is enough to integrate a smooth function of $t$
on $[0, 3]$ to better than $10^{-3}$ precision under trapezoid
quadrature, which is well below the noise floor introduced by the
finite-batch empirical characteristic function. Adding more knots is
expensive (more sin/cos calls) without measurable benefit.

## 6. The numerical contract

The mathematics above is precise; the implementation has to be precise
too. Three contracts are pinned in [RFC 0003]:

1. **F32 invariant (INV-005).** Steps 4–9 of the algorithm — projection,
   $\cos/\sin$, characteristic function comparison, trapezoid integration
   — **must run in F32**, even when the surrounding training loop is in
   BF16 mixed precision. The high-frequency trig terms drift in BF16.
   See [Mixed precision](../training/mixed-precision.md).
2. **Sketch RNG sub-stream.** The projection matrix is sampled from the
   named sub-stream `rng:sigreg_sketch` defined in [RFC 0013] §4. This
   makes the loss exactly reproducible from a given seed.
3. **No stop-gradient on the projection matrix.** $\mathbf P$ is a
   *random buffer*, not a parameter; it carries no gradient, but the
   gradient through $\mathbf Z$ flows freely.

## 7. What the gradient says to the encoder

The gradient of $\mathcal L_{\text{sigreg}}$ with respect to a single
projected latent $\mathbf z_n$ is:

$$
\frac{\partial \mathcal L_{\text{sigreg}}}{\partial \mathbf z_n}
= \frac{2}{KN}\sum_{k,j} q_j\,w(t_j)\;
\Big[(c_{k,j} - \phi(t_j))\, t_j\, \mathbf p_k\,\bigl(-\sin(t_j\,\mathbf p_k^\top \mathbf z_n)\bigr)
+ s_{k,j}\, t_j\, \mathbf p_k\,\cos(t_j\,\mathbf p_k^\top \mathbf z_n)\Big].
$$

This is dense but interpretable: the gradient is a sum of $\mathbf p_k$
directions, weighted by how far the empirical CF is from the target CF
along that direction, modulated by the frequency $t_j$ and the local
derivative of $\cos/\sin$ at $t_j\,\mathbf p_k^\top \mathbf z_n$.

In plain English: SIGReg's gradient *pushes each latent toward the value
that would make its random 1-D projections look more standard-normal*. It
is a many-direction, distributional gradient — not a per-sample
contrastive one.

## 8. Collapse detection probes

Even with SIGReg, training can in principle slip into a partial collapse
(some encoder dimensions become near-constant). RFC 0003 §5 specifies
three runtime probes that read off the same projected batch SIGReg sees:

| Probe | Quantity | Threshold |
|-------|----------|-----------|
| TOL-007 | Per-dim CLS variance $\min_d \mathrm{Var}_b(\mathbf z_b^{(d)})$ | $\geq 0.05$ |
| TOL-008 | Mean abs CLS $\max_d \lvert\mathbb E_b[\mathbf z_b^{(d)}]\rvert$ | $\leq 5.0$ |
| TOL-009 | Mean pairwise CLS cosine $\mathbb E_{b\neq b'}[\cos(\mathbf z_b, \mathbf z_{b'})]$ | $\leq 0.85$ |

A run that trips any of these emits a `collapse_suspected_{step}.json`
diagnostic file and aborts. None of the PushT or SO-100 runs in
[Results](../results/pusht.md) tripped any of the probes.

## 9. Numerical example

To make this concrete, here is what SIGReg computes for one batch.

- $B = 64$, $T = 3$, $D = 192$, $\text{proj\_dim} = 1024$, so $N = 192$
  samples in $\mathbb R^{1024}$.
- Sample $\mathbf P \in \mathbb R^{1024 \times 1024}$, normalize rows.
- Compute $\mathbf Y = \mathbf Z \mathbf P^\top \in \mathbb R^{192 \times
  1024}$, where $\mathbf Z$ is the batch's flattened, projected latents.
- For each of the $J = 17$ knots $t_j$ in $\{0, 0.1875, 0.375, \dots, 3\}$:
  - $\mathbf c_j = \frac{1}{192}\sum_n \cos(t_j \mathbf Y_{:,k})$ for all $k$;
  - $\mathbf s_j = \frac{1}{192}\sum_n \sin(t_j \mathbf Y_{:,k})$ for all $k$;
- For each $k, j$: $r_{k,j} = (c_{k,j} - e^{-t_j^2/2})^2 + s_{k,j}^2$.
- Weighted trapezoid integration: $I_k = \sum_j q_j\, e^{-t_j^2/2}\, r_{k,j}$.
- Final: $\mathcal L_{\text{sigreg}} = (1/1024)\sum_k I_k$.

At init (random encoder), $\mathcal L_{\text{sigreg}}$ is approximately
$0.49$ for the locked PushT ViT — close to the value of $\sim 0.5$ that
the Epps–Pulley statistic takes on an empirical distribution far from
standard normal. Over training, it falls into the $10^{-5}$ range (see
[PushT results](../results/pusht.md)).

## 10. Why this is enough

It is fair to ask: why is this enough? Why doesn't the encoder learn
some pathological distribution that happens to fool the windowed
Epps–Pulley test on $K = 1024$ random directions while still being
useless for prediction?

The short answer is: because **the prediction loss is also running**.
SIGReg forbids collapse and degenerate distributions; the prediction
loss forces the latent to retain enough information about the next
observation that the predictor can recover it under action conditioning.
The two losses pin the encoder from opposite sides — one forbidding
collapse, one forbidding uselessness — and the intersection is exactly
the space of representations that are both *non-trivial* and
*action-predictable*. That intersection is what we want.

In practice, the SIGReg loss drops by 5 orders of magnitude in the first
1 000 steps of PushT training (from 0.49 to ~10⁻⁵), after which both
losses live in the 10⁻⁵–10⁻⁶ band for the rest of training. See
[PushT training curves](../results/pusht.md).

## 11. Bibliography

- Epps, T. W. and Pulley, L. B. (1983). *A test for normality based on
  the empirical characteristic function*. Biometrika 70(3): 723–726.
- Cramér, H. and Wold, H. (1936). *Some theorems on distribution
  functions*. J. London Math. Soc. 11: 290–294.
- Maes, L. et al. (2026). *LeWorldModel*. arXiv:2502.16560 — §3.2
  introduces SIGReg in the LeWM context.
- The upstream Python source `module.py::SIGReg` (lines 13–39 of
  `lucas-maes/le-wm`) is the byte-level reference; our implementation
  in `crates/lewm-core/src/losses/` reproduces it line-for-line.

[RFC 0003]: ../reference/rfcs.md
[RFC 0013]: ../reference/rfcs.md
