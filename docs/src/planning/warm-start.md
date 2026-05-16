# Warm-start ablation

> **Motivation.** A world model trained on one task should provide
> useful features for related tasks. The SO-100 warm-start ablation
> tests this hypothesis.
>
> **Position.** Sub-page of [Part IV](./cem.md).
>
> **What you should leave with.** What "warm-start" means in lewm-rs,
> the contract on the latent-MSE delta, and the current status.

## 1. The setup

Two SO-100 training runs are compared:

| Run | Initialisation |
|-----|----------------|
| **From scratch** | Truncated-normal init, `σ = 0.02`, for every parameter. |
| **From PushT** | Load the PushT step-50000 checkpoint's `vit`, `projector`, `predictor`, `pred_proj`; randomly init `action_enc` (its `smoother.weight` has a different input-channel dim, 2 → 6, so it must be re-init). |

Both runs use the same SO-100 config (`configs/so100.toml`), same seed
(0), same 5 000-step budget, and the same dataset split. The only
difference is the starting parameters.

After training, both checkpoints are evaluated against the SO-100
held-out split using the latent-rollout MSE metric from
[SO-100 eval](./so100-eval.md).

## 2. The hypothesis

If LeWM's visual encoder learned *task-generic* features on PushT (a 2-D
block-pushing task with a top-down camera), those features should
transfer to SO-100 (a 6-DOF arm pick-and-place task with a different
camera angle and object set). The transfer is partial — the action
spaces differ, so the action encoder must be retrained — but the
visual representation should carry over.

The empirical prediction: at the same training budget, the warm-start
checkpoint should have *lower* latent-MSE on held-out SO-100 episodes
than the from-scratch checkpoint.

## 3. The contract (TOL-006)

Pinned in the [glossary](../reference/glossary.md):

> $\text{MSE}_{\text{warm}}^{k=5} \le \text{MSE}_{\text{scratch}}^{k=5}$

That is: warm-start must be no worse than from-scratch on the 5-step
latent rollout. Ideally it should be materially better, but the
contract is the weaker non-regression form (`≤` not `<`).

## 4. The training launcher

The warm-start path is in `crates/lewm-train/src/warmstart.rs`. The
config:

```toml
# configs/so100_warmstart.toml
[init]
mode = "warm_start"
warm_start_checkpoint = "abdelstark/lewm-rs-pusht/.../step_0050000.safetensors"
warm_start_components = ["vit", "projector", "predictor", "pred_proj"]
warm_start_strict = true
```

`warm_start_strict = true` makes the loader assert that every requested
component's weights are present in the source checkpoint. Missing
parameters trigger an error rather than a silent random-init fallback
— this is critical for reproducibility.

The action encoder is *not* in `warm_start_components` because its
`smoother.weight` has a different shape (`(10, 6, 5)` vs `(10, 2, 5)`).
It is randomly initialised, like every other component would be in a
from-scratch run.

## 5. Status

| Item | Status |
|------|--------|
| From-scratch SO-100 checkpoint | <span class="lewm-badge lewm-badge--done">Done</span> (5 000 steps, loss 0.50 → 9.56e-05) |
| Warm-start SO-100 training | <span class="lewm-badge lewm-badge--todo">Planned</span> |
| Warm-start eval / comparison | <span class="lewm-badge lewm-badge--todo">Planned</span> |

The warm-start run is tracked in [`ROADMAP.md`] and will be the subject
of an entry in `reports/so100_warmstart.md` once complete.

## 6. Source pointers

| Topic | Source |
|-------|--------|
| Warm-start loader | `crates/lewm-train/src/warmstart.rs` |
| Warm-start config | `configs/so100_warmstart.toml` |
| Eval metric | `crates/lewm-plan/src/so100_eval.rs` |

[`ROADMAP.md`]: https://github.com/AbdelStark/lewm-rs/blob/main/ROADMAP.md
