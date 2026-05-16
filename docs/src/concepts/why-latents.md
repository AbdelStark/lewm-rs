# Why latent prediction works

> **Motivation.** It is one thing to assert that "predicting in latent
> space" is a good idea; it is another to understand *why* it works
> empirically, and what would have to be true for it to stop working.
> This page collects the conceptual arguments and the empirical
> evidence from `lewm-rs` itself.
>
> **Position.** Closing chapter of [Part I — Concepts](./jepa.md).
>
> **What you should leave with.** A few clear intuitions about why
> latent prediction outperforms pixel prediction, a vocabulary for
> distinguishing *predictable* from *aleatoric* signal, and an
> appreciation of what the SIGReg + prediction balance actually buys.

## 1. The two-source decomposition

For any predictable system, the next observation can be decomposed as

$$
\mathbf y \;=\; \underbrace{\mathcal F(\mathbf x, \mathbf a)}_{\text{predictable part}}\;+\;\underbrace{\boldsymbol\eta}_{\text{aleatoric noise}}
$$

where $\mathcal F$ is some (unknown) deterministic function and
$\boldsymbol\eta$ is irreducible randomness that does not depend on
$(\mathbf x, \mathbf a)$. The predictable part is the only thing that
matters for control — by definition, nothing the controller does can
affect $\boldsymbol\eta$.

A pixel-space MSE objective on $\mathbf y$ spends capacity on both
parts: it tries to render the predictable structure *and* match the
aleatoric noise (or, under quadratic loss, average it). A latent-space
JEPA objective, by contrast, is computed in $f_\theta(\mathbf y)$
space — and a sufficiently expressive encoder is free to **discard**
$\boldsymbol\eta$ in $f_\theta$.

This is the key insight. The encoder is not asked to be a faithful
descriptor of $\mathbf y$. It is asked to be a *predictable* descriptor.
If a feature of $\mathbf y$ is aleatoric — sensor noise, lighting
flicker, motion blur — the encoder can map all noisy realizations to
nearby points in latent space without paying any prediction-loss
penalty, because the predictor will then easily land "near" those
points.

The math is clean: under the JEPA objective, an encoder that ignores
$\boldsymbol\eta$ has the same loss as an encoder that doesn't, *as
long as the resulting latent is still distinguishable for non-aleatoric
configurations*. SIGReg prevents the degenerate solution (collapse), so
the encoder is squeezed toward maximally compact, maximally predictable
representations.

## 2. The "blur problem" revisited

The classic failure mode of pixel-space prediction is **blur**. When
the future is multimodal — e.g., a block being pushed could either
slide left or topple — the MSE-optimal prediction is the arithmetic
mean of the modes, which renders as a blurred superposition of
"sliding" and "toppling" pixels.

In latent space, this problem is largely defused. The latent encoding
is a learned function; it can map "sliding" and "toppling" to two
*separated* points in latent space, with the predictor's output being
the *posterior mean over modes*, which is a perfectly valid latent
(unlike its pixel-space preimage, which is the blurry mess).

More precisely: the latent space induces a metric under which the
quadratic loss is meaningful. If the encoder spaces the modes far
apart, the predictor's "mean" prediction does not lie at any single
mode — but the controller never asks the predictor for a pixel
rendering. It asks for the latent, and the latent is compared to a
*goal* latent. So the predictor can stay deliberately fuzzy on
unresolved modes without misleading the controller.

This is the closest LeWM comes to a "generative" guarantee: not that
it produces sharp images (it produces none), but that the latent
predictions are coherent under the planning cost function.

## 3. Why the encoder doesn't collapse

We have established that SIGReg prevents trivial collapse to a constant.
But why doesn't the encoder collapse to some *partial* solution — say,
encoding only one bit of information that happens to be enough to win
the prediction game on this batch?

Two forces argue against partial collapse:

1. **SIGReg measures the *distributional shape* in $\mathbb R^{D}$
   ($D = 192$).** A degenerate encoder whose projector output lives on
   a low-dimensional sub-manifold of $\mathbb R^{192}$ would fail the
   test against a standard normal — the characteristic function along
   directions orthogonal to the manifold would be far from the target.
   SIGReg's $K = 1024$ random projections sample enough directions of
   $\mathbb R^{192}$ — many more than the ambient dimension itself —
   that low-dimensional collapse is detected.
2. **The prediction loss penalises *any* loss of information that the
   action can recover.** If the encoder discards a feature of the
   observation that the predictor (under action conditioning) could
   have predicted, that's a non-zero prediction-loss penalty. The
   only features the encoder is free to discard are those that are
   *not* predictable from the action — which is exactly the aleatoric
   signal.

Together, these two losses pin the encoder from the "below" and the
"above": SIGReg prevents collapse, prediction prevents triviality. The
intersection — non-degenerate, action-predictable representations — is
what we want.

## 4. Empirical evidence from `lewm-rs`

The argument above is theoretical. The empirical record from this
project is consistent with it.

### 4.1 The training trajectory

From [PushT training](../results/pusht.md):

| Step | $\mathcal L$ | $\mathcal L_{\text{sigreg}}$ | $\mathcal L_{\text{pred}}$ |
|-----:|-------------:|-----------------------------:|---------------------------:|
| 1     | 4.91e-01 | 4.90e-01 | 6.82e-04 |
| 100   | 4.90e-01 | 4.89e-01 | 6.14e-04 |
| 500   | 4.38e-01 | 4.38e-01 | 2.27e-04 |
| 1 000 | 8.69e-02 | 8.69e-02 | 8.43e-07 |
| 5 000 | 6.09e-06 | 4.96e-06 | 1.13e-06 |
| 50 000 | 3.17e-06 | 3.00e-06 | 1.69e-07 |

Two things to note:

- **SIGReg dominates the early loss.** At init the projector's output
  distribution is far from $\mathcal N(\mathbf 0, I_D)$, so SIGReg is
  huge ($\sim 0.49$). The prediction loss is already small ($6.8 \times
  10^{-4}$) because the AdaLN-zero predictor is the identity at init
  and the projectors give an easy starting point.
- **The prediction loss drops below SIGReg around step 1 000.** This
  is the point where the predictor "wakes up" — $W^{\text{mod}}$ has
  moved meaningfully away from zero, the predictor starts using the
  action signal, and the prediction loss falls by three orders of
  magnitude in 500 steps.
- **After ~5 000 steps both losses live in the $10^{-5}$ band**, with
  minor oscillation from the cosine LR tail. The model has settled
  into the intersection.

This is the canonical successful JEPA training trajectory: SIGReg leads
early, prediction loss takes over once the predictor is functional, and
the two settle near the noise floor of the finite-batch empirical
characteristic function (which is the relevant lower bound, not 0).

### 4.2 No collapse detected

The three collapse probes from [SIGReg](./sigreg.md) §8 fired zero
times across the 50 000-step PushT run and zero times across the
5 000-step SO-100 run. The encoder's per-dimension variance stayed
comfortably above the 0.05 floor; mean abs CLS stayed below 5.0; mean
pairwise CLS cosine stayed below 0.85 throughout.

### 4.3 Zero gradient explosions

The gradient norm before clipping was monitored across both training
runs. There were 0 instances of pre-clip gradient norm exceeding the
$10^3$ ceiling specified in TOL-011. This is in part a property of the
optimizer ($\beta_2 = 0.95$, weight decay 0.05), but it is also a
property of AdaLN-zero: a predictor that starts as the identity
cannot produce large gradients in the early steps, which is when most
gradient explosions in transformer training occur.

## 5. When latent prediction would fail

It is useful to know the failure modes that JEPA training would
encounter, in principle:

1. **Encoder over-capacity for the data.** With too many parameters
   and not enough data, the encoder can find degenerate solutions
   that satisfy SIGReg's $K = 1024$ random-direction test by
   accident — i.e., look Gaussian along the sampled directions but
   not along all of them. We have not observed this at the LeWM scale
   (18 M parameters, ~2 M PushT windows), but it would be a real
   concern at larger ratios.
2. **Predictor under-capacity for the dynamics.** If the predictor
   cannot represent the dynamics adequately, the prediction loss
   stalls and the encoder gets only the SIGReg signal — which by
   itself is satisfied by *any* standard-normal-shaped distribution,
   including useless ones.
3. **Action-frame mismatch.** If the smoothed action stream is
   misaligned with the latent stream — e.g., off by one frame — the
   predictor cannot find $\mathcal F$ and the model degenerates.
4. **Precision failures.** SIGReg's high-frequency trig terms drift
   in BF16. We mitigate this with the F32 island in
   [Mixed precision](../training/mixed-precision.md).

None of these triggered in the two runs reported here, but they are
real, and parity tests + collapse probes are how we know.

## 6. The bigger picture

LeWM's combination of latent prediction + SIGReg + AdaLN-zero
predictor + CEM planner is one specific point on a rich design space.
Other points (Dreamer, TD-MPC, V-JEPA) make different trade-offs. The
fact that LeWM works as cleanly as it does — 18 M parameters, two
losses, end-to-end training, no EMA, ≥ 87 % PushT success in the
upstream paper — is genuinely surprising in retrospect and is the
strongest empirical evidence that *latent prediction is the right
high-level paradigm for short-horizon visual control*. The job of
`lewm-rs` is to make that result more accessible, more reproducible,
and more portable.

This concludes Part I. The remaining ten parts of the docsite take the
ideas you have just internalized and turn them into concrete code,
contracts, and numbers.

Continue to: [Architecture at a glance](../architecture/overview.md).
