# PushT evaluation protocol

> **Motivation.** PushT is the dataset where LeWM's planning success
> rate is measured. This page documents the simulator interface, the
> 50-episode test set, the success criterion, and the reporting
> format.
>
> **Position.** Sub-page of [Part IV](./cem.md).
>
> **What you should leave with.** What the metric is, how it is
> computed, and where it currently stands.

## 1. The metric

The PushT eval metric is **planning success rate**: the percentage of
test episodes in which the planner's actions, when rolled out in the
simulator, drive the block into the target zone within a fixed step
budget.

**Target:** ≥ 87 % on the 50-episode test set (matching the upstream
LeWM paper).

**Current:** <span class="lewm-badge lewm-badge--partial">Eval pending</span>
on the lewm-rs PushT 50 k-step checkpoint. The wiring is complete —
`lewm_train::eval::JepaCemCostModel` adapts the parity-verified
`Jepa<B>` to `lewm_plan::CemCostModel` with strict horizon-plan,
latent-dim, action-dim, and history-len validation, and a unit-tested
end-to-end CEM round-trip on a compact synthetic JEPA (4 tests in
`crates/lewm-train/src/eval.rs`). What remains is the **runtime**:
load the trained PushT checkpoint, spawn the `gym-pusht` subprocess,
and run the loop documented in §7 below.

## 2. The episode protocol

For each test episode $e$:

1. **Reset** the simulator. The initial state (block position,
   orientation) is determined by the episode's seed.
2. **Set goal**. The goal observation is the simulator's
   `goal_observation()` (a 224 × 224 RGB rendering of the target
   configuration).
3. **Encode goal once**: `z_goal = encoder(goal_observation)`.
4. **Loop until success or step budget**:
   - Encode current observation: `z_t = encoder(current_observation)`.
   - Build history `z_history` from the last $H_{\text{hist}} = 3$
     encoded frames (padding with $z_t$ if fewer than 3 frames are
     available).
   - Run CEM to obtain `a_1:H = CEM(z_history, z_goal)` with
     `H = horizon_plan = 5`.
   - Execute action `a_1` in the simulator.
   - Append new observation to the history buffer (drop oldest).
   - Step counter += 1.
   - If `simulator.success()` returns true, count episode as a success.
   - If step counter > `step_budget = 200`, count episode as a failure.

## 3. The simulator wrapper

The PushT simulator is `gym-pusht`, a Python `gymnasium` environment.
`lewm-rs` accesses it via a thin Python bridge in
`crates/lewm-plan/src/pusht_eval.rs` that uses `pyo3` to:

- Create the env.
- Reset to a given seed.
- Step it with an action and return the resulting `(observation, reward,
  done, info)` tuple.
- Read `info["is_success"]` for the success bit.

The wrapper is deliberately thin: all the planning logic lives in
Rust; the simulator is treated as a black-box environment.

## 4. The 50-episode test set

The test set is the first 50 episodes of `quentinll/lewm-pusht`'s
*test* split, identified by seed. The exact seed list is `seeds = 0..50`
(the half-open Rust range), i.e. integer seeds $0, 1, \dots, 49$.

Reusing this well-known seed range is a deliberate choice for direct
comparability with the LeWM paper.

## 5. The success criterion

`gym-pusht`'s success criterion is built into the env. Roughly: the
block's centre is within a tolerance of the target zone's centre, with
the orientation within a tolerance angle. The exact constants are part
of `gym-pusht` and not redefined here.

## 6. The report

The eval CLI writes `eval_pusht.json` and `eval_pusht.md` under the
checkpoint's output dir:

```json
{
  "schema_version": "1.0.0",
  "checkpoint": "step_0050000",
  "config_hash": "438eb30f4bb0",
  "num_episodes": 50,
  "num_success": 44,
  "success_rate": 0.88,
  "step_budgets": { "min": 47, "median": 89, "max": 200 },
  "cem_config": { "n_iter": 5, "n_cand": 1024, "n_elite": 103, "horizon": 5 },
  "per_episode": [ {"seed": 0, "success": true, "steps": 67}, ... ]
}
```

The markdown sibling is rendered into the model card.

## 7. Reproducing

```sh
lewm-eval pusht \
    --checkpoint abdelstark/lewm-rs-pusht/train/.../step_0050000.mpk \
    --num-episodes 50 \
    --cem-iter 5 --cem-cand 1024 \
    --out reports/eval_pusht_50ep.json
```

The eval respects the `rng:cem` substream so re-runs with the same
checkpoint and seed produce identical per-episode results.

## 8. Source pointers

| Topic | Source |
|-------|--------|
| Eval driver | `crates/lewm-plan/src/pusht_eval.rs` |
| CEM | `crates/lewm-plan/src/cem.rs` |
| Reports | `crates/lewm-plan/src/reports.rs` |
| Eval CLI | `crates/lewm-plan/src/bin/lewm-eval.rs` |
| JEPA ↔ CEM adapter | `crates/lewm-train/src/eval.rs` (`JepaCemCostModel`) |
| `gym-pusht` integration | Python via `pyo3`; see RFC 0006 §6 |
