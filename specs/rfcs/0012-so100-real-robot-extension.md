---
rfc: "0012"
title: "SO-100 real-robot extension — dataset prep, warm-start, eval"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§4.2", "§4.4 risk 5", "§8 Phase 4-5", "§9.2"]
depends_on: ["0001", "0002", "0004", "0005", "0006"]
related: ["0010", "0013"]
---

# RFC 0012 — SO-100 real-robot extension: dataset prep, warm-start, eval

> **Status:** Accepted · **Version:** 1.0.0
>
> The SO-100 extension is the project's novel contribution: a LeWM trained on real robot footage, not synthetic environments. This RFC pins the dataset preparation pipeline (MP4 → HDF5), the warm-start protocol (from the PushT epoch-10 encoder), and the eval method (latent rollout against expert trajectories).

---

## 1. Introduction

### 1.1 Motivation

PushT is a clean benchmark; SO-100 is a small real-robot dataset (19,631 frames over 50 episodes). The combination "train a JEPA on this with no simulator" is the first such public artifact and the headline novelty. The dataset is small for from-scratch JEPA training, so warm-starting from PushT is the operational lever that makes the result tractable.

### 1.2 Goals

1. Specify the SO-100 dataset preparation pipeline (`python/decode_so100_to_h5.py`).
2. Specify the action normalization for the 6-D action vector.
3. Specify the warm-start protocol: which parameters transfer, how the rest are initialized.
4. Specify the eval protocol with held-out episode selection.
5. Specify the warm-start delta computation and reporting.

### 1.3 Non-goals

- Real-robot deployment (out of scope).
- Multi-camera fusion (out of scope for v1).

---

## 2. Conventions

- `A = 6` — SO-100 action dim (6 joint positions in the LeRobot v2.1 schema).
- `fps_native = 30`, `fps_target = 10` — frame-rate downsampling factor 3.
- `H_native = 480, W_native = 640`, `H_target = W_target = 224`.

---

## 3. Background

`lerobot/svla_so100_pickplace` is a LeRobot v2.1 dataset with:

- 50 episodes (pick-and-place tasks on the SO-100 arm).
- Two cameras per frame: top and wrist.
- Native 30 Hz, 480×640 RGB.
- 6-D action (joint positions).
- ~393 frames per episode average (= 19,631 / 50).

LeRobot v2.1 layout:

```
svla_so100_pickplace/
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   └── ...
├── videos/
│   ├── chunk-000/
│   │   ├── observation.images.top/
│   │   │   ├── episode_000000.mp4
│   │   │   └── ...
│   │   └── observation.images.wrist/
│   │       └── ...
└── meta/
    ├── episodes.jsonl
    ├── tasks.jsonl
    └── info.json
```

Parquet files contain the per-step actions and metadata. MP4 files contain the video frames synchronized to the parquet via the `episode_index` and `frame_index` columns.

---

## 4. Dataset prep pipeline

### 4.1 `decode_so100_to_h5.py`

A one-shot script that produces the HDF5 the Rust loader consumes:

```text
USAGE
    python decode_so100_to_h5.py
        --src <hf-dataset-cache-dir>                # downloaded `svla_so100_pickplace`
        --out <hdf5-path>                            # destination HDF5
        --fps-target 10                              # downsampling target
        --size 224                                    # output H = W
        --interp bilinear                            # bilinear | bicubic
        --camera-views top wrist                     # one or both
        --workers 8                                   # ffmpeg + write parallelism
        --validate                                    # post-write sanity checks
```

Steps:

1. **Read the LeRobot metadata** (`meta/episodes.jsonl`, `info.json`).
2. **Iterate episodes** in parallel:
   a. Open the per-episode MP4 with `pyav` (PyAV).
   b. Decode frames at native FPS, resample to `fps_target` by **dropping** every `(fps_native // fps_target)`-th frame's complement, or via `av.filters.fps`.
   c. Resize each kept frame to `(H_target, W_target)` with bilinear interpolation (`PIL.Image.resize(..., resample=PIL.Image.BILINEAR)`).
   d. Convert to `uint8 HWC` and stack.
   e. Read the matching parquet rows; subsample to the same frame indices.
3. **Write** the HDF5 atomically:
   - `/episode_index`: int32, shape `(N,)`.
   - `/timestep`: int32, shape `(N,)`.
   - `/observation/pixels_top`: uint8, shape `(N, 224, 224, 3)`. Chunked `(64, 224, 224, 3)` for streaming.
   - `/observation/pixels_wrist`: uint8, same shape.
   - `/action`: float32, shape `(N, 6)`.
   - `/joint_pos`: float32, shape `(N, 6)`.
4. **Validate** (when `--validate`):
   - `N == sum(episode_lengths_after_resample)`.
   - Action shape matches expected.
   - Frame count matches manifest.

**RFC0012-001 [MUST]** — `decode_so100_to_h5.py` is **deterministic** given the same inputs and the same PyAV / PIL / numpy versions (pinned in `python/pyproject.toml`). Two runs produce the same HDF5 bytes modulo HDF5's library-internal padding (verified by content-hash, not byte-hash).

**RFC0012-002 [MUST]** — The downsampling **MUST** keep every third frame at native 30 Hz to reach 10 Hz. Time alignment with actions uses the LeRobot timestamp column.

**RFC0012-003 [MUST]** — Frames where the parquet action column is `NaN` (rare; happens at episode boundaries) are **dropped**.

### 4.2 Output schema

Reproduced from [RFC 0004 §6.2](0004-data-pipeline.md); pinned here for stability:

```
/episode_index            : (N,)             int32
/timestep                 : (N,)             int32
/observation/
  pixels_top              : (N, 224, 224, 3) uint8
  pixels_wrist            : (N, 224, 224, 3) uint8
/action                   : (N, 6)           float32
/joint_pos                : (N, 6)           float32
```

Plus root-group attributes:

```
attrs:
  schema_version          : str  "1.0"
  fps_native              : int  30
  fps_target              : int  10
  size                    : int  224
  interp                  : str  "bilinear"
  source_dataset          : str  "lerobot/svla_so100_pickplace"
  source_revision         : str  "<sha at download>"
  decode_tool_versions    : json {pyav: "...", pillow: "...", numpy: "..."}
  content_hash            : str  "<blake3 of array bytes>"
```

**RFC0012-004 [MUST]** — The output HDF5 is uploaded to `abdelstark/so100-pickplace-lewm-ready` per [RFC 0010 §7.4](0010-huggingface-hub-integration.md) with a manifest.

---

## 5. Action normalization for SO-100

Per RFC 0004 §7.2, action normalization is per-dim zero-mean unit-std. For SO-100:

- 6 joint position dimensions (radians or normalized [-1, 1] depending on the dataset's convention; LeRobot v2.1 uses normalized).
- Mean and std computed over the training split's 45 episodes (5 held out).

**RFC0012-005 [MUST]** — `python/compute_so100_stats.py` (a thin wrapper over `lewm-data compute_stats`) is invoked once during Phase 4 to compute and persist `stats.safetensors`.

**RFC0012-006 [MUST]** — If any std is below `1e-6`, it is clamped to `1.0` per RFC0004-017. This is unlikely for SO-100 but defensively encoded.

---

## 6. Held-out split

**RFC0012-007 [MUST]** — Held-out episode IDs are pinned: `[5, 14, 23, 31, 42]` (every 9th roughly). The choice is recorded in `configs/so100.toml::eval.episode_ids` and in the dataset manifest.

**RFC0012-008 [MUST]** — The 5 held-out episodes **MUST NOT** appear in any training batch under any seed. Verified by `TST-0004-SO100-004`.

---

## 7. Warm-start protocol

### 7.1 What transfers

```
PushT epoch-10 model:
  encoder (full ViT + final norm)        ───►  SO-100 encoder (full ViT + final norm)
  predictor (all blocks)                  ───►  SO-100 predictor (all blocks)
  projector (Mlp + BN)                    ───►  SO-100 projector (Mlp + BN)
  pred_proj (Mlp + BN)                    ───►  SO-100 pred_proj (Mlp + BN)
  action_encoder (Embedder)               ───►  re-initialized from scratch
                                                (PushT is 2-D, SO-100 is 6-D)
```

**RFC0012-009 [MUST]** — `Embedder` cannot transfer because the input dim differs. It is re-initialized from scratch per RFC 0002 §4.3 init recipe.

**RFC0012-010 [MUST]** — All other modules transfer their parameters verbatim. BatchNorm running statistics also transfer.

### 7.2 Loading

```rust
// lewm-train::warmstart::load
pub fn load_warmstart<B: Backend>(
    so100_config: &JepaConfig,
    pusht_checkpoint: &Path,
    device: &B::Device,
) -> Result<Jepa<B>, TrainError> {
    // 1. Build a default Jepa with SO-100 config (init all params).
    let mut so100 = Jepa::new(so100_config, device);
    // 2. Load the PushT model into a temporary Jepa.
    let pusht = load_burn_record::<B>(pusht_checkpoint, &VitConfig::default(), device)?;
    // 3. Copy parameters except action_encoder.
    so100.encoder        = pusht.encoder;
    so100.predictor      = pusht.predictor;
    so100.projector      = pusht.projector;
    so100.pred_proj      = pusht.pred_proj;
    // so100.action_encoder is untouched — keeps the default init.
    Ok(so100)
}
```

**RFC0012-011 [MUST]** — Warm-start is invoked by passing `--warmstart-from <path>` to `lewm-train train`. Absent the flag, training runs from scratch.

**RFC0012-012 [MUST]** — When `--warmstart-from` is set, the trainer's provenance preamble (RFC 0005 §4) includes the warm-start source SHA-256.

### 7.3 Optimizer state

**RFC0012-013 [MUST]** — Optimizer state is **not** transferred. SO-100 training starts with a fresh AdamW state (zero momentum, zero variance), even when the model parameters are warm-started.

### 7.4 LR schedule

**RFC0012-014 [MUST]** — Warm-start runs use **the same** cosine schedule as from-scratch (same `lr_peak`, `lr_min`, `warmup_steps`). Empirical evidence justifies a lower `lr_peak` for warm starts; we keep the same for parity, and revisit if SO-100 diverges.

---

## 8. Control: from-scratch arm

To measure the warm-start delta we run **two** SO-100 trainings:

| Run | Init | Config |
|-----|------|--------|
| `scratch` | random | `configs/so100.toml` |
| `warm` | PushT epoch-10 | `configs/so100_warmstart.toml` (same as `so100.toml` except `--warmstart-from`) |

Both run on T3 FULL A10G-large, 10 epochs, ~ 4 hours each. Total Phase 5 spend ~ 12 USD (per PRD §7.2 lines P9/P10, but with two runs not one rerun).

**RFC0012-015 [MUST]** — Both runs **MUST** use the same seed `0` for fair comparison.

---

## 9. Evaluation

Per [RFC 0006 §6](0006-planning-and-evaluation.md), specialized for SO-100:

### 9.1 Per-episode protocol

For each of the 5 held-out episodes:

1. Load the episode's frames (already preprocessed by the loader at runtime).
2. Identify the start frame (`timestep == 0`) and the goal frame (`timestep == max`).
3. Encode start: `z_start = model.encode(start_frame).squeeze(0)` → `(D,)`.
4. Encode goal: `z_goal = model.encode(goal_frame).squeeze(0)` → `(D,)`.
5. Build initial history: `z_hist = z_start.unsqueeze(0).repeat(H, 1)` → `(H, D)`. The conventional way to seed the predictor when no warm-up context is available.
6. Roll out the predictor with the **recorded** expert action sequence:
   - For `t = 0, …, K - 1` (where `K = episode_length - H`):
     - `z_next = model.predict(z_hist.unsqueeze(0), expert_action[t].unsqueeze(0))[-1]`
     - `z_hist = concat([z_hist[1:], z_next.unsqueeze(0)], dim=0)`
     - record `predicted_z[t] = z_next`
7. Compute target: `target_z[t] = model.encode(actual_frame[t + H])` for each `t`.
8. Compute:
   - `latent_mse_per_step = mean over D of (predicted_z - target_z)²` → vector of length `K`.
   - `latent_mse_episode = mean over K of latent_mse_per_step`.
   - `spearman_episode = spearman_rho(pairwise_distances(predicted_z), pairwise_distances(target_z))`.

### 9.2 Aggregate metrics

```
latent_mse_mean    = mean over 5 episodes
spearman_mean      = mean over 5 episodes
```

### 9.3 Warm-start delta

```
delta = latent_mse_mean(scratch) - latent_mse_mean(warm)
```

**RFC0012-016 [MUST]** — `delta > 0` (positive, i.e., warm-start strictly better) is the acceptance signal. `delta ≤ 0` is reported honestly as **null transfer**.

### 9.4 Optional analyses

- **Per-step error growth.** Plot `latent_mse_per_step[t]` averaged across episodes. Expectation: monotonic increase due to autoregressive drift.
- **Nearest-neighbor sanity.** For each `predicted_z[t]`, find the closest training-set frame by L2 in latent space; tabulate. Useful for the writeup but not part of acceptance.

---

## 10. Configs

### 10.1 `configs/so100.toml`

```toml
[dataset]
kind = "so100"
hdf5_path = "/data/so100/svla_so100_pickplace.h5"
camera_view = "top"
stats_path = "/data/so100/stats.safetensors"

[model.encoder]
# same as PushT defaults

[model.action_encoder]
input_dim = 6
smoothed_dim = 10
emb_dim = 192
mlp_scale = 4

[model.predictor]
# same as PushT defaults

[model.projector]
# same as PushT defaults

[model.pred_proj]
# same as PushT defaults

[loss]
lambda_sigreg = 1.0

[training]
history_size = 3
horizon = 4
batch_size = 64
grad_accum_steps = 2
optimizer = "adamw"
lr_peak = 3e-4
lr_min = 1e-5
warmup_steps = 500              # shorter due to smaller dataset
weight_decay = 0.05
betas = [0.9, 0.95]
epochs = 10
precision = "bf16_mixed"
seed = 0

[eval]
kind = "so100_latent_rollout"
episode_ids = [5, 14, 23, 31, 42]
```

### 10.2 `configs/so100_warmstart.toml`

Inherits `configs/so100.toml` and adds:

```toml
[training]
warmstart_from = "/checkpoints/lewm-rs-pusht/step_0014400.mpk"
```

---

## 11. Testing strategy

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0012-PREP-001 | `decode_so100_to_h5_deterministic` | python | RFC0012-001 |
| TST-0012-PREP-002 | `decode_so100_to_h5_schema_correct` | python | §4.2 |
| TST-0012-PREP-003 | `decode_so100_drops_nan_actions` | python | RFC0012-003 |
| TST-0012-STATS-001 | `so100_compute_stats_per_dim` | python | RFC0012-005 |
| TST-0012-HOLDOUT-001 | `so100_holdout_episode_ids_pinned` | unit | RFC0012-007 |
| TST-0012-WARMSTART-001 | `warmstart_copies_encoder` | integration | RFC0012-010 |
| TST-0012-WARMSTART-002 | `warmstart_reinits_action_encoder` | integration | RFC0012-009 |
| TST-0012-WARMSTART-003 | `warmstart_no_optim_state_transfer` | integration | RFC0012-013 |
| TST-0012-EVAL-001 | `so100_eval_pipeline_end_to_end_on_synthetic` | integration | §9.1 |
| TST-0012-EVAL-002 | `so100_eval_spearman_against_scipy` | integration | §9.2 |

Fixtures:

- `tests/fixtures/so100_synth.h5` — 2 synthetic episodes, 32 frames each.
- A miniature PushT checkpoint for the warm-start test (`tests/fixtures/tiny_pusht_ckpt.mpk`).

---

## 12. Operational considerations

### 12.1 Observability

Prep-time:

```
prep/episode_count
prep/frames_kept
prep/frames_dropped
prep/decode_wall_s
```

Train-time additionally to RFC 0009:

```
train/warmstart_source_sha256
```

Eval-time:

```
eval/so100/latent_mse_mean
eval/so100/spearman_mean
eval/so100/warm_start_delta
```

### 12.2 Runbook

- **"Decode is slow."** — bump `--workers`; ensure ffmpeg has hardware accel (NVDEC) when available.
- **"`latent_mse` is large in epoch 1."** — expected; SO-100 is small and noisy. Look at the trajectory plot in the report.
- **"Spearman is unstable across runs."** — verify seed; check that the held-out IDs are correct.

### 12.3 Capacity

- Prep: ~ 2 GB output HDF5; ~ 2 hours on CPU XL (`P6` in PRD §7.2).
- Train: ~ 2 GB peak GPU memory (smaller than PushT due to dataset cache); same A10G-large.

---

## 13. Performance considerations

The MP4 decode is the slowest step in prep. We optimize via parallel workers. If a future v2 wants in-loop decode (Rust ffmpeg-next), this RFC's HDF5 schema becomes a fast lane; the slow lane is added without breaking existing artifacts.

---

## 14. Security considerations

- Dataset is publicly available on HF; no privacy concern.
- Decoded frames stored unencrypted; same trust boundary as PushT.

---

## 15. Alternatives considered

- **A1 — Use both camera views as a 6-channel input.** Considered. Doubles encoder cost; rejected for v1.
- **A2 — Train SO-100 from scratch only, no warm-start.** Rejected: the project's headline novelty is **warm-start delta**.
- **A3 — Use joint positions as input rather than actions.** Considered. Out of scope for v1; the model architecture conditions on actions, not states.
- **A4 — `ffmpeg-next` Rust crate for decode in-loop.** Deferred to v2 per PRD §11 question 2.

---

## 16. Acceptance criteria

- [ ] `decode_so100_to_h5.py` exists and is deterministic.
- [ ] HDF5 schema matches §4.2.
- [ ] Stats safetensors uploaded.
- [ ] Warm-start mechanism implemented and unit-tested.
- [ ] Two trained models published: `so100-scratch` and `so100-warm`.
- [ ] Eval report produced.
- [ ] Warm-start delta computed and reported.

---

## 17. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | Null result (Spearman < 0.4) | M | M | Document honestly; the contribution is the methodology |
| R-2 | Warm-start delta negative | M | M | Document; report both as a valid scientific finding |
| R-3 | Frame/action sync drift | L | H | Validate timestamps during prep; reject episodes with drift > 1 frame |
| R-4 | LeRobot revision changes | L | M | Pin revision SHA in manifest |
| R-5 | Domain gap (PushT to SO-100) larger than expected | M | M | Documented in writeup; future work |

---

## 18. Open questions

OQ-2012-1 — Should we report eval on the train split too (for diagnostic comparison)? Likely yes; add to the report template.

OQ-2012-2 — Whether to publish the eval *trajectories* (latent traces parquet) is a privacy question, but the dataset is public, so they should be public too. Confirm with LeRobot team in Phase 4.

---

## 19. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0012.*
