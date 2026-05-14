---
rfc: "0006"
title: "lewm-plan — CEM planner and evaluation drivers"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§4.1 Rollout / planning", "§9 Evaluation"]
depends_on: ["0001", "0002", "0003", "0004"]
related: ["0005", "0007", "0012", "0013"]
---

# RFC 0006 — `lewm-plan`: CEM planner and evaluation drivers

> **Status:** Accepted · **Version:** 1.0.0
>
> The planner turns the trained world model into a controller via CEM. The PushT evaluator uses the same planner against the simulator; the SO-100 evaluator measures latent-space agreement against held-out expert trajectories. This RFC pins both algorithms exactly, including hyperparameters, seeds, and the cost function entry point.

---

## 1. Introduction

### 1.1 Motivation

The CEM planner is the second-largest source of evaluation variance after the training stochasticity itself. Specifying every hyperparameter and its seeding precisely is the only way to keep the planning success rate reproducible across runs of the same checkpoint.

### 1.2 Goals

1. Specify CEM (Cross Entropy Method) to a level of worked pseudocode.
2. Specify the PushT eval protocol: simulator wiring, 50-episode set, success criterion, reporting.
3. Specify the SO-100 eval protocol: latent-rollout MSE, Spearman rank correlation, warm-start delta.
4. Specify the `lewm-eval` binary CLI.
5. Specify the eval-report file produced for each model.

### 1.3 Non-goals

- The PushT simulator itself (we use the Python `gymnasium`/`gym-pusht`-equivalent via a Python adapter; v1 does not re-implement the simulator in Rust).
- A graphical eval visualizer (the rerun.io pipeline used by LeRobot is sufficient; we link to it from the model card).
- Real-robot deployment.

---

## 2. Conventions

`H = history_size`, `K = horizon`, `M = action_dim`. CEM hyperparameters:

- `n_iter` — number of CEM iterations.
- `n_cand` — number of action candidates sampled per iteration.
- `n_elite` — number of top candidates used to update the proposal distribution.
- `horizon_plan` — number of action steps to plan (typically 5 at eval time).

---

## 3. Crate layout

```
lewm-plan/
└── src/
    ├── lib.rs
    ├── bin/
    │   └── lewm-eval.rs
    ├── cem.rs                   # CEM implementation
    ├── cost.rs                  # cost-function adapter (calls Jepa::get_cost)
    ├── proposal.rs              # Gaussian proposal distribution & update
    ├── pusht_eval.rs            # PushT simulator wrapper + episode runner
    ├── so100_eval.rs            # SO-100 latent-rollout & Spearman
    ├── reports.rs               # eval_<dataset>.md generator
    └── errors.rs
```

---

## 4. CEM specification

### 4.1 Algorithm

```
Inputs:
    z_history: (1, H, D)       initial latent context
    z_goal:    (D,)             goal latent
    n_iter, n_cand, n_elite, horizon_plan, sigma_init, sigma_min
    action_dim M
    rng:cem RNG sub-stream

Initialize:
    mu    = zeros (horizon_plan, M)                # in normalized action space
    sigma = sigma_init * ones (horizon_plan, M)

For iter in 0 .. n_iter:
    # 1. Sample n_cand candidate sequences
    eps   = randn (n_cand, horizon_plan, M)         # using rng:cem
    cand  = mu + sigma * eps                        # broadcasting: (n_cand, horizon_plan, M)

    # 2. Compute cost for each candidate
    #    Expand z_history & z_goal across n_cand axis, run get_cost in a single batched call.
    z_hist_b = z_history.expand((n_cand, H, D))
    z_goal_b = z_goal.expand((n_cand, D))
    costs    = jepa.get_cost(z_hist_b, cand, z_goal_b)     # (n_cand,)

    # 3. Select top n_elite by lowest cost
    elite_idx = argsort(costs)[:n_elite]
    elites    = cand[elite_idx]                            # (n_elite, horizon_plan, M)

    # 4. Update proposal
    mu_new    = elites.mean(dim=0)                          # (horizon_plan, M)
    sigma_new = max(elites.std(dim=0), sigma_min)
    mu, sigma = mu_new, sigma_new

# Return the best candidate from the final iteration
best_idx = argmin(costs)
return cand[best_idx], costs[best_idx]
```

### 4.2 Hyperparameters

| Symbol | Default (PushT) | Default (SO-100) | Source |
|--------|------------------|------------------|--------|
| `n_iter` | 5 | 5 | PRD §9.1 |
| `n_cand` | 1000 | 16 (inference-time) / 1000 (offline) | PRD §9.1; for live planning we use 16 to fit the latency budget |
| `n_elite` | 100 | 4 (inference) / 100 (offline) | PRD §9.1 |
| `horizon_plan` | 5 | 5 | PRD §9.1 |
| `sigma_init` | 1.0 | 1.0 | upstream `stable-worldmodel` |
| `sigma_min` | 0.05 | 0.05 | sanity floor |

**RFC0006-001 [MUST]** — All CEM RNG draws **MUST** come from the `rng:cem` sub-stream defined in RFC 0013. Seeding the planner with the global seed `0` produces bit-identical action proposals across runs.

**RFC0006-002 [MUST]** — The cost is computed via `Jepa::get_cost`, which internally calls `rollout`. This means the same `Jepa` model used at training time is the operator at eval time; no separate forward path.

### 4.3 Numerical contract

**RFC0006-003 [MUST]** — Cost is in latent-space MSE units. Smaller is better. CEM minimizes.

**RFC0006-004 [MUST]** — In normalized action space, the proposal is centered at zero and the per-dim std starts at `1.0`. The output `mu`/`best` is in normalized action space; the eval driver inverse-normalizes before stepping the simulator (using `ActionNormalizer::inverse`).

### 4.4 Implementation

```rust
pub struct Cem<B: Backend> {
    pub n_iter: usize,
    pub n_cand: usize,
    pub n_elite: usize,
    pub horizon_plan: usize,
    pub sigma_init: f64,
    pub sigma_min: f64,
}

impl<B: Backend> Cem<B> {
    pub fn plan(
        &self,
        model: &Jepa<B>,
        z_history: Tensor<B, 3>,
        z_goal: Tensor<B, 2>,
        rng: &mut ChaCha20Rng,
        action_dim: usize,
        device: &B::Device,
    ) -> CemResult<B> {
        // see §4.1 pseudocode
    }
}

pub struct CemResult<B: Backend> {
    /// Best candidate action sequence, shape (horizon_plan, M), normalized space.
    pub best_actions: Tensor<B, 2>,
    /// Cost of the best candidate.
    pub best_cost: f32,
    /// Per-iteration trace for debugging.
    pub trace: Vec<CemIterTrace>,
}
```

**RFC0006-005 [SHOULD]** — `Cem::plan` runs in `no_grad` mode. There is no gradient backflow through the planner.

---

## 5. PushT evaluation

### 5.1 Protocol

Per PRD §9.1. 50 held-out episodes. For each:

1. Reset the simulator to the episode's start state.
2. Encode the start frame and the goal frame: `z_0 = encode(start), z_goal = encode(goal)`.
3. Build initial `z_history` by replicating `z_0` H times (history size 3 → `z_history = (1, 3, D)` with each row = `z_0`).
4. For up to `max_steps_per_episode = 100` simulator steps:
   - Run CEM with `(z_history, z_goal)` → `best_actions: (horizon_plan, M)`.
   - Take the first action `best_actions[0]`, inverse-normalize, step the simulator.
   - Encode the new observation. Append to `z_history`; drop the oldest.
   - If the simulator's `success` flag fires, count the episode as a win and break.
5. Record success/failure.

**Headline metric:** `planning_success_rate = wins / 50`.

### 5.2 Simulator wiring

The PushT simulator is invoked through a Python sidecar (`python/pusht_runner.py`), called from Rust via the `python-bridge` feature on `lewm-plan` (PyO3 dev-only) **or** via a subprocess JSON-RPC interface.

**RFC0006-006 [MUST]** — v1 uses the **subprocess JSON-RPC** path:

```
Rust -> child process python/pusht_runner.py (long-lived)
        sends: {"method": "reset", "params": {"episode": 17, "seed": 42}}
        recvs: {"obs": <base64 RGB 224x224x3 u8>, "state": [...]}
        sends: {"method": "step", "params": {"action": [0.42, -0.15]}}
        recvs: {"obs": <...>, "reward": 0.0, "done": false, "success": false}
```

This avoids the PyO3 build dep and is straightforward to mock for tests.

**RFC0006-007 [MUST]** — The child process is a single long-lived `python -u python/pusht_runner.py` per eval run. One process per eval, not per episode.

**RFC0006-008 [MUST]** — The 50 held-out episode IDs are deterministic per dataset; pinned in `configs/pusht.toml::eval.episode_ids`.

### 5.3 Eval driver

```rust
pub struct PushtEvaluator<B: Backend> {
    model: Jepa<B>,
    cem: Cem<B>,
    image_preproc: ImagePreprocessor,
    action_norm: ActionNormalizer,
    rpc: PushtRpc,
    config: PushtEvalConfig,
}

pub struct PushtEvalConfig {
    pub episode_ids: Vec<u32>,
    pub max_steps_per_episode: u32,
    pub seed: u64,
}

impl<B: Backend> PushtEvaluator<B> {
    pub fn run(&mut self) -> Result<PushtEvalReport, EvalError> { /* … */ }
}

pub struct PushtEvalReport {
    pub success_rate: f32,
    pub per_episode: Vec<EpisodeOutcome>,
    pub wall_time_s: f32,
    pub total_steps: u32,
}

pub struct EpisodeOutcome {
    pub episode_id: u32,
    pub success: bool,
    pub steps_taken: u32,
    pub final_cost: f32,
    pub trajectory_summary: TrajectorySummary,
}
```

### 5.4 Acceptance

**RFC0006-009 [MUST]** — `success_rate >= 0.87` on the 50-episode set is the FR-051 acceptance floor (PRD §9.1).

**RFC0006-010 [MUST]** — A run that misses the floor by less than 2 absolute points triggers one **re-run with a different seed** (still recorded in the report). A run that misses by more than 2 points is a **null result** to be documented honestly, with diagnostics (collapse-detector trace, training metrics) included in the report.

---

## 6. SO-100 evaluation

### 6.1 Protocol

Per PRD §9.2. 5 held-out episodes. For each:

1. Encode the start frame: `z_start`.
2. Encode the goal frame (last frame of the episode): `z_goal`.
3. Replay the *recorded* expert action sequence through the predictor:
   - Set `z = z_start.expand(H)` as initial history.
   - For `t in 0..(episode_len - H)`: `z = rollout(z, recorded_action[t])`.
4. Encode each frame in the actual recorded trajectory to get `target_z[t]`.
5. Compute:
   - **Latent MSE**: `mean over t of ‖pred_z[t] - target_z[t]‖²`.
   - **Spearman rank correlation** between the pairwise distance matrices of the predicted vs. target trajectories. See §6.2 for the exact formula.

### 6.2 Spearman protocol

Build the pairwise distance matrices:

```
D_pred[i, j] = ‖pred_z[i] - pred_z[j]‖   for all i, j
D_targ[i, j] = ‖target_z[i] - target_z[j]‖
```

Flatten the upper triangle (excluding diagonal): `d_pred ∈ ℝ^{n(n-1)/2}`, `d_targ ∈ ℝ^{n(n-1)/2}`.

Spearman rank correlation: `ρ_s = Pearson(rank(d_pred), rank(d_targ))`.

**RFC0006-011 [MUST]** — Ranks are computed via the average-rank method for ties (`scipy.stats.rankdata(method='average')` equivalent).

**Headline metric:** average `ρ_s` across the 5 episodes.

### 6.3 Warm-start delta

Two models are trained in Phase 5: one from scratch (`scratch`), one warm-started from the PushT epoch-10 encoder (`warm`). The warm-start delta is:

```
Δ = mean_episode latent_mse_scratch - mean_episode latent_mse_warm
```

A positive `Δ` is evidence that PushT pretraining transferred.

### 6.4 Acceptance

**RFC0006-012 [MUST]** — `ρ_s >= 0.6` is the FR-052 acceptance floor.

**RFC0006-013 [MUST]** — `ρ_s < 0.4` is declared a null result and documented honestly in `reports/so100_training.md` per PRD §9.2.

**RFC0006-014 [MUST]** — `0.4 ≤ ρ_s < 0.6` is reported as a **partial** result; the operator decides whether to publish with explicit caveat. (This case is unlikely but the policy is binding.)

---

## 7. CLI

```text
lewm-eval <subcommand> [flags]

Subcommands:
  pusht       Run the PushT 50-episode protocol.
  so100       Run the SO-100 5-episode protocol.
  report      Render an eval report from a JSON results file.

Global flags:
  --checkpoint <PATH>      Burn record (.mpk) of the model to evaluate.
  --output-dir <PATH>      Eval output dir (default ./out-eval/<run_id>).
  --seed <INT>             Override the global seed (default 0).
  --episodes <INT>         Override episode count (defaults: 50 / 5).
  --max-steps <INT>        Per-episode step cap (default 100 for PushT).
  --device <DEVICE>        cuda:0 | cpu (default cuda:0).
```

### 7.1 Output

For PushT:

- `out-eval/<run_id>/results.json` — full per-episode results.
- `out-eval/<run_id>/report.md` — human-readable summary.
- `out-eval/<run_id>/trajectories.parquet` — per-step (action, cost) traces.

For SO-100:

- `out-eval/<run_id>/results.json` — per-episode latent_mse, ρ_s.
- `out-eval/<run_id>/report.md`.
- `out-eval/<run_id>/latent_traces.parquet` — predicted vs. target embeddings.

---

## 8. Testing strategy

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0006-CEM-001 | `cem_proposal_update_correct` | unit | §4.1 step 4 |
| TST-0006-CEM-002 | `cem_seed_determinism` | unit | RFC0006-001 |
| TST-0006-CEM-003 | `cem_converges_on_toy_quadratic` | unit | smoke for the algorithm itself |
| TST-0006-EVAL-PUSHT-001 | `pusht_eval_loop_terminates_correctly` | integration | §5.3 |
| TST-0006-EVAL-PUSHT-002 | `pusht_rpc_mock_success_rate` | integration | Mocked simulator; verifies plumbing |
| TST-0006-EVAL-LAT-001 | `so100_latent_mse_matches_python` | integration | RFC0006-011 |
| TST-0006-EVAL-LAT-002 | `so100_spearman_matches_scipy` | integration | RFC0006-011 |
| TST-0006-EVAL-DELTA-001 | `warm_start_delta_correct_sign` | integration | §6.3 |
| TST-0006-REPORT-001 | `report_md_format_stable` | unit (insta snapshot) | report rendering |

**Fixtures:**

- A toy quadratic cost function for `TST-0006-CEM-003`.
- Synthetic latent trajectories with known `ρ_s` for `TST-0006-EVAL-LAT-002`.
- A mock `pusht_runner.py` that always reports `success` after 5 steps, for `TST-0006-EVAL-PUSHT-002`.

---

## 9. Operational considerations

### 9.1 Observability

Metrics per eval episode:

```
eval/episode_success           # 0 / 1
eval/episode_steps             # int
eval/episode_final_cost        # f32
eval/cem_iter_cost_min         # f32, last iter
eval/cem_iter_cost_mean        # f32
eval/cem_iter_sigma_mean       # f32, last iter
eval/wall_time_per_episode_s   # f32
```

Aggregates:

```
eval/success_rate              # f32 (PushT)
eval/latent_mse_mean           # f32 (SO-100)
eval/spearman_mean             # f32 (SO-100)
eval/warm_start_delta          # f32 (SO-100 paired runs)
```

Spans:

```
eval.episode
eval.cem_iter
eval.cem_cost_eval
eval.rpc_step
```

### 9.2 Runbook

- **"Eval hangs on episode 17."** — the Python sidecar process likely crashed; check `out-eval/<run_id>/rpc.log`. Restart with `--episodes 17`.
- **"Success rate is 0/50."** — almost certainly an action-normalization mismatch; verify `stats.safetensors` is the one used at training time.
- **"Spearman is NaN."** — degenerate trajectories (all-zero predictor output). Indicates the model collapsed during training; verify training-time collapse detector.

### 9.3 Capacity

- GPU memory: the CEM `n_cand × horizon_plan × M` action tensor is small; the cost dominates by the `rollout` forward, which is `n_cand` parallel rollouts. At `n_cand=1000`, batch size for the predictor is 1000; this is the highest GPU memory pressure of the entire project. The locked PushT checkpoint uses `D=192`, reducing this estimate from the earlier `D=384` draft.

**RFC0006-015 [MUST]** — If GPU memory exceeds 18 GB during eval, automatically **chunk** the `n_cand` candidates into chunks of 250 and concatenate the costs. This is a runtime fallback, transparent to the user.

---

## 10. Performance considerations

Per-step latency budget at eval:

| Step | Budget |
|------|--------|
| Encode obs | 5 ms |
| CEM, 5 iter × 1000 cand × 5-step rollout | 200 ms (GPU) |
| RPC roundtrip | 5 ms |
| Total per simulator step | 210 ms |
| 100 steps × 50 episodes | ~ 17 minutes |

Inference CPU path is much slower; see RFC 0007.

---

## 11. Security considerations

- Subprocess RPC is unauthenticated and localhost-only; no network exposure.
- The Python sidecar runs under the same UID as the trainer; we trust it.

---

## 12. Alternatives considered

- **A1 — MPPI instead of CEM.** Considered. CEM is the published method; MPPI is a future extension.
- **A2 — Synchronous PyO3 binding instead of subprocess.** Considered. Subprocess is simpler to build and mock; we revisit if RPC overhead dominates.
- **A3 — Native Rust PushT simulator.** Out of scope for v1; gym_pusht is well-tested.

---

## 13. Acceptance criteria

- [ ] All TST-0006-* pass.
- [ ] CEM hyperparameters in `configs/pusht.toml` match §4.2.
- [ ] Eval driver produces `report.md` matching the insta snapshot.
- [ ] Mock-simulator integration test passes deterministically.

---

## 14. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | gym_pusht version drift changes the task subtly | M | M | Pin version in `python/pyproject.toml` |
| R-2 | RPC sidecar deadlock | L | M | Sidecar uses line-buffered JSON; timeout per request 30s |
| R-3 | Memory blow-up at n_cand=1000 on smaller GPUs | M | M | RFC0006-015 chunking |
| R-4 | Spearman implementation drift | L | L | Reference against scipy on fixed input |

---

## 15. Open questions

OQ-2006-1 — Should eval be parallelized across episodes (one RPC sidecar per worker)? Not needed at 50 episodes × 200 ms/step; revisit in v2.

---

## 16. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0006.*
