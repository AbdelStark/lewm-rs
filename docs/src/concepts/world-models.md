# World models for robotics

> **Motivation.** Calling something a "world model" is a strong claim. In
> robotics, a world model is a *learned simulator* that a controller can
> roll out in its head. LeWM is a world model in a precise, restricted
> sense; this page makes that sense explicit and locates LeWM on the
> spectrum of robotic world models.
>
> **Position.** Conceptual context for [LeWM](./lewm.md), [Planning](../planning/cem.md),
> and the [SO-100 extension](../planning/so100-eval.md).
>
> **What you should leave with.** A working definition of "world model",
> the distinction between *pixel-space* and *latent-space* world models,
> and the place LeWM occupies.

## 1. What we mean by "world model"

Following Ha & Schmidhuber (2018), a **world model** is a learned function

$$
\hat{\mathbf s}_{t+1} \;=\; W_\phi(\mathbf s_t,\; \mathbf a_t)
$$

that, given a representation $\mathbf s_t$ of the world at time $t$ and an
action $\mathbf a_t$, predicts the representation at time $t+1$. The
representation $\mathbf s$ may be the raw sensor reading (pixels), a
hand-crafted state (joint angles + object poses), or a learned latent (a
vector emerging from an encoder). The downstream user of the world model
is a **controller** that selects actions to drive $\mathbf s$ toward a
target — most commonly a planner using model-predictive control (MPC) or
a policy trained via model-based reinforcement learning.

Three things must be true for a world model to be useful in robotics:

1. **Cheap to roll out.** A planner that calls the world model $10^3$ to
   $10^5$ times per decision needs $W$ to cost milliseconds, not seconds.
   This rules out pixel-rendering models on CPU.
2. **Accurate enough for the planner's horizon.** Errors compound under
   autoregressive rollout. A model that drifts by 5 % per step is
   useless after 10 steps.
3. **Action-conditioned in a meaningful way.** The model must distinguish
   between trajectories under different actions. A model whose predictions
   are barely modulated by $\mathbf a$ cannot drive a controller.

LeWM is built to satisfy all three for the particular case of latent
action-conditioned prediction on visual observations.

## 2. Pixel vs. latent: a brief history

| Class | Representative | Why it's hard |
|-------|----------------|---------------|
| **Pixel world models** | PixelRNN (van den Oord et al., 2016); World Models (Ha & Schmidhuber, 2018); Dreamer V1/V2/V3 (Hafner et al., 2020–2023). | Predicting pixels wastes capacity on aleatoric noise; under MSE the optimum is mean-blur; rolling out at planning rate is expensive because every step renders an image. |
| **Latent generative world models** | Dreamer's RSSM recurrent latent; TD-MPC2 (Hansen et al., 2023). | Reconstruction loss still anchors the latent to "decodable" features, which leaks aleatoric noise back in. RSSM works by carefully balancing prior, posterior, and KL terms — many knobs. |
| **JEPA world models** | LeWM (Maes et al., 2026); V-JEPA-based predictors. | No reconstruction. The latent is constrained only by predictability under action and by SIGReg's distribution prior. |

The trajectory of the field has been to move the loss further and further
away from pixels. LeWM is the current endpoint of that trajectory for
visual manipulation: the loss is entirely in the encoder's own output
space, and the only thing tethering the encoder to "useful features" is
the predictor's need to find them when it rolls out.

## 3. The control loop LeWM sits in

A typical use of the LeWM world model at deployment time looks like:

```text
   observation o_t (pixels)             goal observation o_g
            │                                    │
            ▼                                    ▼
       encoder f_θ                         encoder f_θ
            │                                    │
            ▼  z_t                               ▼  z_g
            │                                    │
            └────────────┐                ┌──────┘
                         ▼                ▼
                  ┌──────────────────────────┐
                  │            CEM            │  proposes a_1:H, scores
                  │   over horizon H = 5      │  via predictor + cost
                  └──────────────────────────┘
                              │
                              ▼  best a_1:H
                              │
                              ▼ execute first action a_1
                              │
                              ▼
                  environment step → new o_t
                  (repeat)
```

The predictor and encoder are exercised many times per decision (CEM
typically samples $n_{\text{cand}} \in [128, 2048]$ action sequences per
iteration over $n_{\text{iter}} \in [3, 10]$ iterations), but **no pixel
is ever rendered** during planning. All comparisons are between latent
vectors in $\mathbb R^{D = 192}$. This is what makes CPU planning viable
in practice.

## 4. The two evaluation regimes

LeWM is evaluated against two datasets with quite different planning
contracts:

### 4.1 PushT — simulator-grounded, success-rate metric.

PushT is a 2-D block-pushing task with a procedurally generated
simulator. Evaluation is binary success / failure over 50 held-out
initial conditions: did the planner's actions, when rolled out in the
simulator, drive the block into the target zone within a step budget?
The metric is a percentage. The target for `lewm-rs` is **≥ 87 %**,
matching the upstream LeWM paper.

### 4.2 SO-100 — real-robot data, latent-MSE metric.

The SO-100 pick-and-place dataset has no in-the-loop simulator: it is 50
teleoperated episodes recorded on a real SO-100 6-DOF arm. Evaluation
must therefore be done in **open loop**, comparing the predictor's
latent rollout to the encoder's embedding of the actual recorded frames.
The metric is the mean MSE between predicted and observed latents over a
held-out episode, optionally with the Spearman rank correlation of the
per-step error. A *warm-start* ablation — initialising SO-100 training
from a PushT checkpoint vs. from scratch — measures whether PushT-learned
features transfer to a different robot and task.

Both metrics are pinned in [RFC 0006] and discussed in detail in
[PushT eval](../planning/pusht-eval.md) and
[SO-100 eval](../planning/so100-eval.md).

## 5. What LeWM does not aspire to

LeWM is one rung on a long ladder. It is not, and does not try to be:

- A **general** world model. The domain is short-horizon (≤ 5 steps) visual
  manipulation. Long-horizon planning, multi-step credit assignment, and
  hierarchical control are out of scope.
- A **multi-camera, multi-sensor** model. The PushT and SO-100 pipelines
  consume a single 224 × 224 RGB camera feed. Adding wrist-camera or
  depth is future work (RFC 0012 §4.3).
- A **policy**. LeWM is a model, not a controller. The controller is CEM,
  which is a non-learned, sample-based planner. Drop-in replacement with
  a learned policy (DPO over the predictor, e.g.) is a separate project.

For an explicit list of project goals and non-goals, see the PRD §2 in the
repository.

[RFC 0006]: ../reference/rfcs.md

## 6. Bibliography

- Ha, D. and Schmidhuber, J. (2018). *World Models*. NeurIPS.
- Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T. (2023).
  *DreamerV3: Mastering diverse domains through world models*.
- Hansen, N., Su, H., Wang, X. (2023). *TD-MPC2: Scalable, Robust World
  Models for Continuous Control*. arXiv:2310.16828.
- van den Oord, A., Kalchbrenner, N., Vinyals, O., Espeholt, L., Graves,
  A., Kavukcuoglu, K. (2016). *Conditional Image Generation with PixelCNN
  Decoders*. NeurIPS.

Continue to: [SIGReg deep dive](./sigreg.md).
