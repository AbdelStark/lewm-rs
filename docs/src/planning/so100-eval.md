# SO-100 evaluation protocol

> **Motivation.** Unlike PushT, SO-100 has no in-the-loop simulator.
> Evaluation must be done in open loop, comparing the predictor's
> latent rollout to the encoder's embedding of the recorded frames.
>
> **Position.** Sub-page of [Part IV](./cem.md).
>
> **What you should leave with.** The two SO-100 metrics
> (latent-rollout MSE and Spearman rank correlation), and the warm-start
> ablation contract.

## 1. The challenge: no simulator

The SO-100 pick-and-place dataset is recorded on a real 6-DOF arm. The
data ships as 50 teleoperated episodes, with no simulator that we can
run the planner against. The only ground truth is the recorded
observations and the recorded actions.

This rules out the "success rate" metric used for PushT. Instead, we
evaluate the *world model itself*: how well does the predictor's
latent rollout match the encoder's embedding of the recorded frames?

## 2. Metric 1: latent-rollout MSE

For each held-out episode and each frame $t$ in the episode:

1. Encode the *recorded* observation: $\mathbf z_t^{\text{recorded}} =
   \text{encoder}(\mathbf o_t)$.
2. Construct the predictor history from the recorded encoder outputs
   $[\mathbf z_{t-H_{\text{hist}}}, \dots, \mathbf z_{t-1}]$.
3. Roll the predictor forward by $k$ steps using the *recorded* action
   sequence $[\mathbf a_{t-H_{\text{hist}}}, \dots, \mathbf a_{t-1+k}]$.
4. Compare the rollout's final latent $\hat{\mathbf z}_{t+k}$ to the
   recorded $\mathbf z_{t+k}^{\text{recorded}}$:

$$
\text{MSE}_k(t) = \big\lVert \hat{\mathbf z}_{t+k} - \mathbf z_{t+k}^{\text{recorded}} \big\rVert^2_2 / D.
$$

5. Average over $t$ and over episodes.

The metric is reported for $k \in \{1, 2, 3, 5\}$ — short-horizon (1-step),
mid-horizon (3-step), and the standard planning horizon (5-step).

## 3. Metric 2: Spearman rank correlation

A second, robust metric: for a held-out episode, rank the per-step
prediction errors and check whether they are *increasing* with $k$.
A model that has learned dynamics should have monotonically increasing
$k$-step error; a model that has collapsed will have flat error.

We compute Spearman's $\rho$ between the per-step horizon and the
per-step latent-MSE. Values close to $+1$ mean the model degrades
gracefully with horizon; values close to $0$ indicate something is
off.

## 4. The warm-start ablation

The SO-100 training pipeline supports two initialisation modes:

- **From scratch**: random init (standard truncated-normal).
- **From PushT**: load the PushT step-50000 checkpoint into the
  encoder and predictor before training begins.

The ablation compares the latent-rollout MSE of the two checkpoints
after the SO-100 training budget (5 000 steps). The contract pinned by
TOL-006:

> $\text{MSE}_{\text{warm}} \le \text{MSE}_{\text{scratch}}$.

(That is, warm-start should be at least as good as from-scratch; ideally
materially better.)

**Status:** <span class="lewm-badge lewm-badge--partial">Eval pending</span>.
The from-scratch SO-100 checkpoint exists; the warm-start training run
has not yet been launched.

## 5. The held-out split

The SO-100 dataset has 50 episodes total. The train / eval split is
fixed in `crates/lewm-data/src/so100.rs`:

- **Train**: episodes 0..45 (~5 920 frames).
- **Eval**: episodes 45..50 (~639 frames).

The held-out 5 episodes are *not* seen by the training loop and are
used only for the latent-rollout metrics.

## 6. The report

`lewm-eval so100` writes `eval_so100.json` and `eval_so100.md`:

```json
{
  "schema_version": "1.0.0",
  "checkpoint": "step_0005000",
  "config_hash": "...",
  "num_episodes": 5,
  "num_frames": 639,
  "latent_mse": {
    "k_1": 0.0042,
    "k_2": 0.0078,
    "k_3": 0.0119,
    "k_5": 0.0203
  },
  "spearman_rho": 0.91,
  "warm_start": null,
  "per_episode": [ {"id": 45, "mse_k1": ..., "mse_k5": ...}, ... ]
}
```

When the warm-start ablation is run, the `"warm_start"` field holds a
delta object comparing from-scratch and from-PushT checkpoints.

## 7. Reproducing

```sh
lewm-eval so100 \
    --checkpoint abdelstark/lewm-rs-so100/train/.../step_0005000.mpk \
    --eval-split 45..50 \
    --horizons 1,2,3,5 \
    --out reports/eval_so100.json
```

## 8. Source pointers

| Topic | Source |
|-------|--------|
| Eval driver | `crates/lewm-plan/src/so100_eval.rs` |
| Reports | `crates/lewm-plan/src/reports.rs` |
| Warm-start training mode | `crates/lewm-train/src/warmstart.rs` |
| Eval CLI | `crates/lewm-plan/src/bin/lewm-eval.rs` |
