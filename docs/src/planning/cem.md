# Planning with CEM

> **Motivation.** The trained world model is used at deployment time
> through the *Cross-Entropy Method* (CEM) planner. This page documents
> the algorithm at the level of pseudocode pinned in [RFC 0006].
>
> **Position.** Top of [Part IV — Planning and evaluation](./cem.md).
>
> **What you should leave with.** The CEM hyperparameters, the
> per-iteration update equations, and which crate implements which
> path.

## 1. CEM, restated

See [The Cross-Entropy Method](../concepts/cem.md) for the conceptual
introduction. Here we focus on the *exact* algorithm as implemented in
`crates/lewm-plan/src/cem.rs`.

## 2. The pinned hyperparameters

| Parameter | PushT default | SO-100 default |
|-----------|--------------:|---------------:|
| `n_iter` | 5 | 5 |
| `n_cand` | 1024 | 1024 |
| `n_elite` | 103 (= 10 % of `n_cand`) | 103 |
| `horizon_plan` ($H$) | 5 | 5 |
| `sigma_init` | 1.0 (normalised action space) | 1.0 |
| `sigma_min` | 0.05 | 0.05 |
| `momentum` ($\eta$) | 0.5 | 0.5 |

`momentum` is the EMA factor applied when updating $\mu$ and $\sigma$
from the elite statistics — see §3 step 4.

The defaults are pinned in `configs/pusht_eval.toml`. They match the
upstream LeWM paper.

## 3. The algorithm

Inputs to one planning decision:
- `z_history: (1, H_hist, D)` — encoder output for the current history.
- `z_goal:    (D,)` — encoder output for the goal image.
- The trained `Jepa` model.

```text
mu       = zeros (horizon_plan, M)
sigma    = sigma_init * ones (horizon_plan, M)
best_a   = None
best_cost = +inf

for iter in 0 .. n_iter:
    # 1. Sample candidates
    eps   = randn (n_cand, horizon_plan, M) from rng:cem
    cand  = mu + sigma * eps                   # (n_cand, horizon_plan, M)

    # 2. Score candidates (batched predictor rollout)
    z_hist_b = z_history.expand((n_cand, H_hist, D))
    z_goal_b = z_goal.expand((n_cand, D))
    z_final  = rollout(z_hist_b, cand)         # (n_cand, D)
    cost     = get_cost(z_final, z_goal_b)     # (n_cand,)

    # 3. Pick elites
    elite_idx = argsort(cost)[:n_elite]
    elite_a   = cand[elite_idx]                 # (n_elite, horizon_plan, M)

    # 4. Update proposal (with momentum)
    mu_new    = mean(elite_a, dim=0)
    sigma_new = std (elite_a, dim=0).clamp_min(sigma_min)
    mu        = momentum * mu    + (1 - momentum) * mu_new
    sigma     = momentum * sigma + (1 - momentum) * sigma_new

    # 5. Track best-seen
    if cost[elite_idx[0]] < best_cost:
        best_cost = cost[elite_idx[0]]
        best_a    = cand[elite_idx[0]]

return best_a
```

The output `best_a` is a sequence of `horizon_plan` action vectors. In
MPC mode, only `best_a[0]` is executed; the planner is rerun after the
new observation arrives.

## 4. The rollout

The `rollout` function inside CEM uses the same sliding-window pattern
documented in [Jepa wrapper](../architecture/jepa-wrapper.md) §4:

```text
history: (B, H_hist, D)   # B = n_cand
actions: (B, horizon_plan, M)

for step in 0..horizon_plan:
    a_step = actions.narrow(1, step, 1)        # (B, 1, M)
    a_emb  = action_enc(a_step)                # (B, 1, D)
    z_next = predictor(history, a_emb)         # (B, H_hist, D)
    z_pred = z_next.narrow(1, H_hist-1, 1)     # (B, 1, D)
    history = concat([history.narrow(1, 1, H_hist-1), z_pred], dim=1)

z_final = history.narrow(1, H_hist-1, 1).squeeze(1)
return z_final
```

This pattern keeps the predictor input shape constant. After `horizon_plan`
steps, `z_final` is the predictor's estimate of the latent at the end
of the action sequence; the cost is the squared L2 distance to the
goal latent.

## 5. Two implementations

There are two CEM runners with identical algorithmic content but
different compute backends:

### 5.1 `lewm-plan::cem` — Burn

Used by `lewm-eval` against a Burn checkpoint (`.mpk` or
`.safetensors`). The forward runs on whatever backend was used for
training (CUDA, NdArray CPU). This is the high-accuracy reference
path.

### 5.2 `lewm-infer::plan` — Tract CPU

Used by the deployed CPU runner and the Gradio Space. The encoder and
predictor are loaded as separate ONNX graphs and executed via Tract.
The CEM logic is the same, but it runs entirely outside Burn.

The two implementations are exercised by the parity-eval CLI:

```sh
lewm-infer eval --dumps-dir <path> --backend tract     # Tract CEM
lewm-infer eval --dumps-dir <path> --backend burn-cpu  # Burn CPU CEM
lewm-infer eval --dumps-dir <path> --backend burn-cuda # Burn CUDA CEM
```

Each backend writes a per-stage JSON of L∞ / RMSE against the official
reference dumps. See
[`reports/gpu_inference.md`](https://github.com/AbdelStark/lewm-rs/blob/main/reports/gpu_inference.md).

## 6. Numerics

CEM itself runs in F32 throughout. The predictor inside CEM uses
whatever precision its backend was configured with; the cost function
(`get_cost`) is F32. The proposal distribution $\mu, \sigma$ is F32.

The `rng:cem` substream is seeded deterministically from the master
seed and the substream name (`"cem"`), so the same checkpoint + seed
produces the same CEM trajectory.

## 7. Source pointers

| Topic | Source |
|-------|--------|
| Burn CEM | `crates/lewm-plan/src/cem.rs` |
| Tract CEM | `crates/lewm-infer/src/plan.rs` |
| Eval CLI | `crates/lewm-plan/src/bin/lewm-eval.rs` |
| Parity eval CLI | `crates/lewm-infer/src/eval.rs` |
| Reference Python CEM | `python/pusht_runner.py` (used to produce parity dumps) |

[RFC 0006]: ../reference/rfcs.md
