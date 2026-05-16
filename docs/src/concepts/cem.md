# The Cross-Entropy Method

> **Motivation.** The world model is only half a controller. The other
> half is a *planner*: a procedure that, given the current latent state
> and a goal latent, picks an action sequence. LeWM uses the
> Cross-Entropy Method (CEM), a sample-based, derivative-free planner
> that is unreasonably effective for short-horizon manipulation. This
> page introduces it from first principles.
>
> **Position.** Conceptual root for [Planning with CEM](../planning/cem.md),
> which gives the exact algorithm pinned in [RFC 0006].
>
> **What you should leave with.** Why CEM is the right tool for this
> job, what the proposal distribution updates look like, and where the
> wall-clock cost lives.

## 1. The problem CEM solves

We have:

- a forward dynamics model $\mathbf z_{t+1} = W_\phi(\mathbf z_t, \mathbf a_t)$
  (in LeWM, this is the JEPA predictor),
- a cost function $J(\mathbf a_{1:H}) = \lVert \mathbf z_H - \mathbf
  z_{\text{goal}} \rVert^2$ over action sequences of length $H$,
- a current latent $\mathbf z_0$,
- and a goal latent $\mathbf z_{\text{goal}}$.

We want $\mathbf a^\star_{1:H} = \arg\min_{\mathbf a_{1:H}} J$. This is a
continuous, non-convex optimisation in $\mathbb R^{HM}$ where $M$ is the
action dimension.

Three families of solvers are commonly considered:

1. **Gradient-based MPC** (e.g. iLQR, gradient through a differentiable
   simulator). Requires differentiating through $W_\phi$, which works
   for JEPA in principle, but introduces second-order stability concerns
   and a planning-time autograd dependency we'd like to avoid for CPU
   deployment.
2. **Trajectory optimisation via continuous-time methods** (Pontryagin /
   DDP). Powerful, but overkill for $H \le 5$ and requires significant
   structural assumptions on $W_\phi$.
3. **Sample-based planners.** Sample many candidate action sequences,
   score them under $W_\phi$, pick the best. Embarrassingly parallel,
   makes no assumptions about $W_\phi$ beyond callability.

LeWM picks the third route, specifically **CEM**, because it is fast,
simple, derivative-free, and trivially parallel — properties that
together let CPU planning fit into a sub-second budget on a laptop.

## 2. CEM as iterative cross-entropy

The Cross-Entropy Method (de Boer et al., 2005) is, fundamentally, an
*iterative importance sampling* procedure for optimisation. The
procedure is:

```text
Initialise proposal distribution q_0 over action sequences (a 1:H).
For iter = 1 to n_iter:
    Sample n_cand candidates a_(1), …, a_(n_cand)  ~  q_{iter-1}.
    Score each candidate: J(a_(i)).
    Select the top n_elite candidates by lowest J.
    Update q_iter by minimising the cross-entropy KL( elites ‖ q_iter ),
      i.e. by fitting q_iter to the elite empirical distribution.
Return arg min of J among samples seen.
```

When the proposal $q$ is a diagonal Gaussian $\mathcal N(\boldsymbol\mu,
\boldsymbol\sigma^2)$ (which is the case for LeWM), the cross-entropy
update has a closed form: fit $\boldsymbol\mu$ and $\boldsymbol\sigma$
to the mean and standard deviation of the elites.

So each CEM iteration is:

1. **Sample.** $n_{\text{cand}}$ candidate sequences from
   $\mathcal N(\boldsymbol\mu_t, \boldsymbol\sigma_t^2)$ — broadcast
   across the per-step shape.
2. **Score.** Run the predictor $W_\phi$ over each candidate, in batch.
   Compute the latent-distance cost.
3. **Select.** Pick the top $n_{\text{elite}}$ candidates by lowest cost.
4. **Refit.** Set $\boldsymbol\mu_{t+1} = \text{mean}(\text{elites})$,
   $\boldsymbol\sigma_{t+1} = \text{std}(\text{elites})$, optionally
   smoothing with the previous step.

After $n_{\text{iter}}$ iterations, return the best candidate seen.

## 3. The wall-clock budget

In LeWM's deployment scenario, the dominant cost is **step 2** — the
batched predictor call. For each candidate sequence of $H$ actions, we
run the predictor $H$ times (or once with a batched temporal axis).
With $n_{\text{cand}} = 1024$ and $H = 5$, that is $5120$ predictor
calls per CEM iteration. On Tract CPU (Apple M-series), this currently
runs at $\sim 800$ ms per iteration. With $n_{\text{iter}} = 5$, total
planning time is $\sim 4$ s per episode — matching the
[Tract benchmark](../inference/benchmark.md).

The encoder is called only twice per planning decision: once on the
current observation to get $\mathbf z_0$, once on the goal image to get
$\mathbf z_{\text{goal}}$. Encoder cost is therefore amortised across
all CEM iterations and is not the bottleneck.

## 4. Hyperparameters and their effect

The CEM behaviour is shaped by five hyperparameters:

| Parameter | Symbol | LeWM default | Effect |
|-----------|--------|--------------|--------|
| Number of iterations | $n_{\text{iter}}$ | 5 | More = better optima at higher wall-clock cost. |
| Candidate sequences per iter | $n_{\text{cand}}$ | 1024 | More = better exploration; trivially parallel. |
| Elite count | $n_{\text{elite}}$ | $\lceil 0.1\, n_{\text{cand}}\rceil$ = 103 | Smaller = more aggressive refinement; larger = more exploration. |
| Initial standard deviation | $\sigma_0$ | $1.0$ in normalised action space | Larger = broader initial exploration. |
| Minimum standard deviation | $\sigma_{\min}$ | $0.05$ | Floor prevents the proposal from collapsing too fast. |
| Planning horizon | $H$ | 5 | Trade-off between predictor accuracy at the end of the rollout and the amount of control authority. |

All defaults are pinned in `configs/pusht_eval.toml` and visible in
`crates/lewm-plan/src/cem.rs`.

## 5. Why CEM survives despite being old

CEM dates from the 1990s. It has been outperformed in pure
sample-efficiency by gradient-based MPC and by learned policies in many
benchmarks. So why use it in LeWM?

Three reasons:

1. **It does not assume differentiability of $W_\phi$.** This matters
   because the *deployment* graph (ONNX → Tract) is not differentiable.
   We could keep the autograd graph for planning, but it would force
   Burn at deployment time, which conflicts with the CPU-only goal.
2. **It is trivially parallel across candidates.** The hot loop is one
   big batched predictor call, which both Burn and Tract handle well.
3. **It is robust to a small model.** A learned policy needs more
   capacity and more training data than LeWM has. CEM exploits any
   model that can roll out and score, without needing it to be a
   reliable global optimum.

For short horizons ($H \le 10$) on smooth, low-dimensional action
spaces, this combination is hard to beat with anything that is also
simple, derivative-free, and fast on CPU. LeWM stays with it.

## 6. CEM and replanning

In a robot loop, CEM is typically used in a **model-predictive control**
(MPC) loop: at each time step, the planner computes the best
$\mathbf a_{1:H}$, **executes only the first action $\mathbf a_1$**,
then observes the new state and replans. This receding-horizon strategy
is what makes CEM robust to model errors: any drift in $W_\phi$ over
$H$ steps is compensated by re-planning every step.

For PushT and SO-100 evaluation, the loop is exactly:

```text
while not goal_reached:
    z_t = encoder(observation_at_t)
    a_1:H = CEM(z_t, z_goal, predictor)
    execute a_1
    observation_at_{t+1} = env.step(a_1)
```

The encoder is called once per environment step (cheap), and CEM runs
$n_{\text{iter}}$ predictor batches per step (expensive). On the
PushT 50-episode test set, this yields the success-rate metric
discussed in [PushT eval](../planning/pusht-eval.md).

## 7. Bibliography

- de Boer, P.-T., Kroese, D. P., Mannor, S., Rubinstein, R. Y. (2005).
  *A Tutorial on the Cross-Entropy Method*. Annals of Operations
  Research, 134(1): 19–67.
- Wang, T., Ba, J. (2020). *Exploring Model-based Planning with Policy
  Networks*. ICLR — uses CEM with a learned dynamics model on continuous
  control.
- Hansen, N., Wang, X. (2022). *Temporal Difference Learning for Model
  Predictive Control* — TD-MPC, the closest cousin of LeWM's planner.

Continue to: [Why latent prediction works](./why-latents.md).
