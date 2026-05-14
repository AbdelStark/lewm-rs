---
rfc: "0005"
title: "lewm-train — training system, optimizer, schedule, checkpoints"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§5.2", "§5.4", "§5.5", "§6.3"]
depends_on: ["0001", "0002", "0003", "0004", "0009", "0013", "0018"]
related: ["0006", "0008", "0010", "0011", "0014", "0017"]
---

# RFC 0005 — `lewm-train`: training system, optimizer, schedule, checkpoints

> **Status:** Accepted · **Version:** 1.0.0
>
> The training crate is where the model from RFC 0002, the losses from RFC 0003, and the data pipeline from RFC 0004 meet on the GPU. This RFC pins the exact step semantics: how the optimizer, the schedule, the grad accumulation, the mixed precision, the checkpoint cadence, the state machine, and the resume protocol all combine. The deliverable is one binary, `lewm-train`, that runs reproducibly from clean start to publish-ready artifact.

---

## 1. Introduction

### 1.1 Motivation

A modern ML training loop has roughly forty knobs and a thousand things that can go wrong. The reproducibility goal (NFR-002, NFR-051) plus the parity goal (FR-021..023) plus the cost ceiling (NFR-060) leave little room for off-by-one error. We pin everything that can be pinned and validate the rest at run time.

### 1.2 Goals

1. Specify the `lewm-train` CLI verb set and arguments.
2. Specify the optimizer (AdamW), schedule (cosine + warmup), grad accumulation, grad clipping, and mixed precision implementations.
3. Specify the checkpoint contract: what is written, when, how it is named, how it is loaded.
4. Specify the crash-resume protocol such that an interrupted run resumes with bit-identical RNG and step state.
5. Specify the per-epoch parity probe, the eval trigger, and the upload hook.
6. Codify the training state machine of PRD §5.5 with transition semantics.

### 1.3 Non-goals

- Loss math (RFC 0003).
- Data pipeline (RFC 0004).
- Eval algorithms (RFC 0006).
- ONNX export (RFC 0007).
- HF Jobs orchestration YAML (RFC 0011).

---

## 2. Conventions

Same as master spec. Beyond glossary:

- **micro-batch** — one forward+backward sized at `batch_size`.
- **effective batch** — `grad_accum_steps × batch_size` micro-batches grouped into one optimizer step.
- **step** — one optimizer step (i.e., one effective batch).
- **epoch** — one full pass over the training dataset's window count.

---

## 3. Crate layout

```
lewm-train/
└── src/
    ├── lib.rs
    ├── bin/
    │   └── lewm-train.rs              # clap CLI
    ├── trainer.rs                      # outer loop, state machine
    ├── step.rs                          # one optimizer step (micro-batches + grad clip)
    ├── optim.rs                         # AdamW wrapper, decay/no-decay parameter split
    ├── schedule.rs                      # cosine with warmup
    ├── mixed_precision.rs               # BF16 cast helpers, F32 islands
    ├── checkpoint.rs                    # save/load, sidecar
    ├── resume.rs                        # crash detection, restoration
    ├── parity_probe.rs                  # per-epoch probe driver
    ├── eval.rs                          # eval hooks (delegates to lewm-plan)
    ├── upload.rs                        # post-run upload (delegates to lewm-hub)
    ├── state.rs                         # State enum, transitions
    ├── monitor.rs                       # metrics emission facade
    └── errors.rs
```

---

## 4. CLI

```text
lewm-train <subcommand> [flags]

Subcommands:
  train       Run the full pipeline from INIT through UPLOAD.
  smoke       Run a short local smoke (50 steps) on NdArray CPU; PRD stage L2.
  parity      Run the parity harness (RFC 0008). No training.
  eval        Run eval on a checkpoint, no training.
  convert     Convert HF/PyTorch reference weights → Burn record.

Global flags:
  --config <PATH>         Path to TOML config.
  --set KEY=VALUE         Override a single config key (repeatable).
  --output-dir <PATH>     Run output directory (default ./out/<run_id>).
  --resume-if-present     If the output-dir contains a `run_id.txt`, resume from latest checkpoint.
  --seed <INT>            Override the global seed (default from config).
  --device <DEVICE>       cuda:0 | cpu | metal:0 (default cuda:0 when feature enabled).
  --log-level <LEVEL>     RUST_LOG-equivalent, default info.
  --dry-run               Build everything but do not execute the step loop.
  --max-steps <INT>       Cap step count (debugging).
```

Subcommand-specific flags:

```text
train:
  --data-dir <PATH>       Directory holding the dataset's HDF5 files.
  --hf-token <ENV>        Env var holding the HF token for the post-run upload (optional).

smoke:
  --steps <INT>           Default 50.
  --batch-size <INT>      Default 4.

parity:
  --reference <PATH>      Path to the Safetensors weights.
  --dump-dir <PATH>       Path to the per-layer reference dumps.

eval:
  --checkpoint <PATH>     Path to the Burn record.
  --episodes <INT>        Default 50 (PushT) / 5 (SO-100).

convert:
  --pt <PATH>             PyTorch checkpoint (.pt).
  --out <PATH>            Output Burn record (.mpk).
  --intermediate <PATH>   Optional Safetensors intermediate.
```

**RFC0005-001 [MUST]** — `clap` derives the CLI from a `#[derive(Parser)]` struct in `bin/lewm-train.rs`. The struct **MUST** be `serde::Deserialize` so the CLI can be driven from a job YAML if needed (future).

**RFC0005-002 [MUST]** — Every subcommand prints a single-line provenance preamble:

```
lewm-train v0.1.0 (git: <short_sha>, build: <date>); seed=<seed>; device=<device>; config_hash=<blake3-hex-12>
```

This line is the first thing in every log; CI parses it to verify the build was reproducible per NFR-050.

---

## 5. The training step

### 5.1 Top-level loop

```
fn train(config, output_dir):
    state = State::Init
    while state != State::Done:
        match state:
            State::Init                  -> init_run(...)            -> state = ParityCheck
            State::ParityCheck           -> run_parity(...)          -> state = Smoke
            State::Smoke                 -> run_smoke(...)           -> state = Warmup
            State::Warmup                -> warmup_steps(...)        -> state = Steady
            State::Steady                -> run_epochs(...)          -> state = Cooldown
            State::Cooldown              -> finalize_lr(...)         -> state = Eval
            State::Eval                  -> run_eval(...)            -> state = Upload
            State::Upload                -> upload_artifacts(...)    -> state = Done
```

Each transition writes a `transition_<ts>.json` file and a checkpoint per PRD §5.5.

### 5.2 Inner step (one optimizer step)

```
fn step(model, optim, scheduler, prefetcher, mp_policy, lambda, step_i):
    grads = None
    for k in 0..grad_accum_steps:
        batch = prefetcher.next()?
        with mp_policy.autocast():
            losses = model.criterion(batch.pixels, batch.actions, lambda)
            scaled = losses.total / (grad_accum_steps as f32)
        grads_k = scaled.backward()
        grads   = accumulate(grads, grads_k)

    grad_norm_pre  = grads.norm()
    grads, clipped = clip_global_norm(grads, max_norm=1.0)
    grad_norm_post = grads.norm()

    optim.step(grads)
    scheduler.step(step_i)

    emit_metrics(losses, grad_norm_pre, grad_norm_post, lr=scheduler.lr())
    return losses
```

Notable rules:

**RFC0005-003 [MUST]** — Gradient accumulation **MUST** scale the per-micro-batch loss by `1 / grad_accum_steps` before backward. The optimizer thus sees the average, not the sum.

**RFC0005-004 [MUST]** — Gradient clipping uses **global** L2 norm across the entire gradient flat-vector (all params concatenated). Per-param clipping is forbidden.

**RFC0005-005 [MUST]** — Both pre- and post-clip norms are emitted as metrics. The pre-clip norm above `TOL-011 = 1e3` triggers an `ERROR` log line and a `grad_explosion_{step}.json` artifact. Training continues; the operator decides whether to abort. (Hard auto-abort is reserved for NaN/Inf detection in §5.5.)

**RFC0005-006 [MUST]** — NaN/Inf detection: after `losses.total.backward()`, the trainer checks `losses.total.is_finite()` and `grads.is_finite()` (a single elementwise scan across the flat gradient). On non-finite, the step is **skipped** (gradient zeroed, optimizer state unchanged) and a `nan_detected_{step}.json` artifact is written. Three skipped steps in a row → fatal abort.

### 5.3 Optimizer

```rust
// crates/lewm-train/src/optim.rs

pub struct LewmAdamW<B: Backend> {
    inner: burn::optim::AdamW<B>,
    decay_params: Vec<String>,
    no_decay_params: Vec<String>,
}

#[derive(burn::config::Config, Debug)]
pub struct OptimConfig {
    #[config(default = "0.9")]
    pub beta1: f64,
    #[config(default = "0.95")]
    pub beta2: f64,
    #[config(default = "1e-8")]
    pub epsilon: f64,
    #[config(default = "0.05")]
    pub weight_decay: f64,
}
```

**RFC0005-007 [MUST]** — Weight decay applies to `Linear.weight`, `Conv1d.weight`, `Conv2d.weight`, and `Embedder.smoother.weight`. It does **not** apply to:

- LayerNorm / BatchNorm scales and biases (`*.weight`, `*.bias` of any norm module).
- Any `*.bias` of a Linear/Conv (per the PyTorch/timm convention).
- `cls_token` and `pos_embed`.

The parameter-partitioning logic lives in `optim::partition_decay_no_decay` and is **unit-tested** (TST-0005-OPT-001) by enumerating the param names of a freshly built `Jepa`.

**RFC0005-008 [MUST]** — `betas = (0.9, 0.95)` matches the LeWM paper. `epsilon = 1e-8` matches Burn default and PyTorch default.

### 5.4 Learning-rate schedule

```rust
// crates/lewm-train/src/schedule.rs

pub struct CosineWarmup {
    lr_peak: f64,
    lr_min: f64,
    warmup_steps: u32,
    total_steps: u32,
}

impl CosineWarmup {
    pub fn lr(&self, step: u32) -> f64 {
        if step < self.warmup_steps {
            // linear ramp from 0 → lr_peak
            self.lr_peak * (step as f64) / (self.warmup_steps as f64)
        } else {
            let progress = (step - self.warmup_steps) as f64
                / ((self.total_steps - self.warmup_steps).max(1) as f64);
            let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.lr_min + (self.lr_peak - self.lr_min) * cosine
        }
    }
}
```

**RFC0005-009 [MUST]** — `total_steps = epochs × (len / effective_batch)` is computed at schedule construction; if `total_steps` is unknown (e.g., resume), it is read from the sidecar.

**RFC0005-010 [MUST]** — At `step = warmup_steps`, the schedule reaches exactly `lr_peak`. At `step = total_steps`, it reaches exactly `lr_min`. Beyond `total_steps`, it returns `lr_min` (clamped).

**RFC0005-011 [MUST]** — Defaults: `lr_peak = 3e-4`, `lr_min = 1e-5`, `warmup_steps = 1000`, `epochs = 10` (PRD §5.2).

### 5.5 Mixed precision

```rust
// crates/lewm-train/src/mixed_precision.rs

pub enum MixedPrecisionPolicy {
    F32,                  // everything F32
    Bf16Mixed,            // forward/backward BF16, optimizer F32 (master weights), SIGReg F32
}

impl MixedPrecisionPolicy {
    pub fn autocast<B: Backend, F, R>(&self, scope: F) -> R
        where F: FnOnce() -> R
    { /* enables BF16 within the scope */ }

    pub fn cast_sigreg_input<B: Backend>(&self, x: Tensor<B, 3>) -> Tensor<B, 3>
    { /* F32 cast on entry */ }
}
```

**RFC0005-012 [MUST]** — In `Bf16Mixed`:

- Model parameters are stored in **F32** ("master weights").
- Forward and backward inside the autocast scope run in BF16 via Burn's `autocast` (the `Cuda<BF16>` backend with autograd).
- The optimizer step uses F32 gradients. (Burn's AMP pattern.)
- SIGReg internals are F32 per INV-005 (RFC 0003 §4.2.5).

**RFC0005-013 [MUST]** — On a non-BF16-capable backend (NdArray CPU), `Bf16Mixed` is downgraded to `F32` with a warning. The smoke tier uses `F32` deliberately.

**RFC0005-014 [MUST]** — Loss-scaling is **not** required for BF16 (BF16 has F32-equivalent dynamic range). We do **not** use a `GradScaler`-equivalent. If we move to FP16 in the future, this changes.

### 5.6 Effective batch and accumulation

```
config:
  batch_size       = 64
  grad_accum_steps = 2
  effective_batch  = 128
```

**RFC0005-015 [MUST]** — `effective_batch = batch_size × grad_accum_steps` is computed eagerly at config validation. CI fails if the effective batch is below 32 or above 512 (sanity bounds; ADR required to exceed).

---

## 6. Checkpointing

### 6.1 Files written per epoch

At every epoch boundary, **four** files are written:

1. `step_{N:07d}.mpk` — Burn record (model + optimizer state).
2. `step_{N:07d}.safetensors` — Safetensors mirror of model parameters only (for portability).
3. `step_{N:07d}.json` — Sidecar metadata.
4. `step_{N:07d}.parity.json` — Per-epoch parity probe results.

`N` is the optimizer step at the end of the epoch.

**Sidecar JSON schema:**

```json
{
  "schema_version": "1.0",
  "run_id": "20260512-143002-9f3a-abcd",
  "step": 14400,
  "epoch": 5,
  "wall_time_s": 12345.6,
  "git_short_sha": "9f3a8e2",
  "config_hash": "blake3-12hex",
  "rng_state": {
    "global_seed": 0,
    "step_at_save": 14400,
    "data_shuffle": "<chacha-state>",
    "sigreg_sketch": "<chacha-state>",
    "dropout": "<chacha-state>",
    "cem": "<chacha-state>",
    "model_init": "<chacha-state>"
  },
  "metrics_last_step": {
    "loss/total": 0.0123,
    "loss/pred": 0.0099,
    "loss/sigreg": 0.0024,
    "optim/lr": 0.000235,
    "optim/grad_norm_pre": 1.42
  },
  "checkpoint_files": {
    "model_burn": "step_0014400.mpk",
    "model_safetensors": "step_0014400.safetensors",
    "parity": "step_0014400.parity.json"
  }
}
```

**RFC0005-016 [MUST]** — `config_hash` is `blake3(config_canonical_toml_bytes)`, where the canonical form is `serde_toml::to_string_pretty(...)` with sorted keys. Two configs that differ only by key order produce the same hash.

**RFC0005-017 [MUST]** — Files **MUST** be written via atomic rename: write to `step_*.mpk.tmp`, fsync, then rename. The sidecar is the **last** file written; its presence is the signal that the checkpoint is complete.

**RFC0005-018 [MUST]** — Last **three** epoch checkpoints are kept on disk; older ones are removed by `checkpoint::prune` on each new write. All checkpoints are kept on Hub (see RFC 0010 §6).

### 6.2 Saving model state

```rust
pub fn save_checkpoint<B: Backend>(
    model: &Jepa<B>,
    optim: &LewmAdamW<B>,
    schedule_state: &ScheduleState,
    rng_state: &RngState,
    step: u64,
    output_dir: &Path,
) -> Result<CheckpointPaths, TrainError> { /* … */ }
```

**RFC0005-019 [MUST]** — Saves via Burn's `NamedMpkFileRecorder` for the `.mpk` and a hand-rolled walker for the `.safetensors`. The Safetensors walker visits parameters in the deterministic order of `Module::visit_params`.

### 6.3 Per-epoch parity probe

`parity_probe::run` evaluates the current model on a fixed input (the parity fixture from RFC 0008) and dumps:

```json
{
  "encoder_cls_l_inf": 7.2e-05,
  "predictor_l_inf": 9.5e-05,
  "sigreg_value": 0.00731
}
```

This is **not** a parity *test* — it does not assert tolerances. It is a per-epoch *probe* that the report job later compares across epochs to spot training-time drift. The actual parity tests run only at PARITY_CHECK transition and in CI.

---

## 7. Resume protocol

### 7.1 Detection

On startup, the trainer scans `<output_dir>/run_id.txt`:

- **Present and `--resume-if-present`**: enter resume mode.
- **Absent or no `--resume-if-present`**: fresh mode.
- **Present but `--resume-if-present` not set**: error `RunDirOccupied` (refuse to clobber).

### 7.2 Restore

```
1. Read run_id.txt and locate latest step_{N}.json sidecar.
2. Verify sidecar SHA pinned to current git HEAD; warn if mismatch.
3. Load model from step_{N}.mpk (full module + optimizer state).
4. Restore RNG sub-streams from sidecar.
5. Restore scheduler state (step counter).
6. Resume from State::Steady (or the state recorded in transition_*.json if more advanced).
```

**RFC0005-020 [MUST]** — Resume **MUST** reproduce the next step's RNG state. Concretely: after resume from step `N`, the next batch sampled is the same one that would have been sampled in a non-crashing run.

**RFC0005-021 [MUST]** — Resume **MUST NOT** require any flag beyond `--resume-if-present`. The output dir is sufficient.

**RFC0005-022 [MUST]** — On signal `SIGTERM` or `SIGINT`, the trainer:

1. Finishes the current step (or aborts if mid-micro-batch).
2. Writes an emergency checkpoint at the current step (not the epoch boundary).
3. Updates the sidecar.
4. Exits with code 0.

The HF Jobs cancel path sends SIGTERM by default, so this is the expected behavior.

---

## 8. State machine

```
INIT -> PARITY_CHECK -> SMOKE -> WARMUP -> STEADY -> COOLDOWN -> EVAL -> UPLOAD -> DONE
                                         ^_________|
                                         resume cycle
```

### 8.1 INIT

- Load config, validate.
- Build model on device.
- Build prefetcher.
- Build optimizer (parameter partition).
- Build schedule.
- Compute `total_steps`.
- Initialize telemetry (trace exporter, Trackio).
- Write `state.json` with `state=INIT, step=0`.

### 8.2 PARITY_CHECK

- Load reference weights from `parity.reference_path` if config requests parity.
- Run parity probe on fixed input.
- If tolerances satisfy `glossary.md` §4, proceed; otherwise abort with `ParityFailed`.

**RFC0005-023 [MAY]** — Parity check **MAY** be skipped via `--no-parity-check` for ablation runs. Production runs **MUST NOT** skip.

### 8.3 SMOKE

- Run 50 steps on a small fixed subset.
- Verify loss decreases (slope test: best-fit linear slope over steps 10..50 must be negative).
- Verify per-step throughput meets a low bar (NFR-010 / 4 on smoke hardware).
- On failure: `SmokeFailed`.

### 8.4 WARMUP

- Run `warmup_steps` optimizer steps with the linear warmup schedule.
- Emit standard metrics.

### 8.5 STEADY

- Run epochs until `step >= total_steps` or `epochs` reached.
- Per-epoch: parity probe + checkpoint.
- Periodic: eval-mini (50 episodes) every `eval_every_n_epochs` (default 5).

### 8.6 COOLDOWN

- One full epoch with `lr` already at `lr_min` to allow stats normalization to stabilize.

### 8.7 EVAL

- Full eval (RFC 0006) on the held-out split.
- Write `reports/eval_<dataset>.md`.

### 8.8 UPLOAD

- Push checkpoints, training report, eval report to HF Hub (RFC 0010).
- Update cost ledger.

### 8.9 DONE

- Final summary line emitted to stdout.

### 8.10 ERROR states

Each non-DONE state has a corresponding error transition. The trainer writes `error_{state}.json` with the cause and exits non-zero. CI alerts on non-zero exit.

---

## 9. Runbook

### 9.1 Smoke

```bash
make smoke
# or
cargo run --release --bin lewm-train -- \
    smoke --config configs/pusht.toml --steps 50 --batch-size 4
```

Expected: green in 2–5 minutes on a laptop with `--device cpu`.

### 9.2 Cloud T1

```bash
hf jobs run \
  --namespace abdelstark \
  --flavor l4x1 \
  --timeout 30m \
  --image ghcr.io/abdelstark/lewm-rs:latest \
  -- bash -c "lewm-train smoke --config configs/pusht.toml --steps 200 --batch-size 16"
```

### 9.3 Cloud T2 SHORT

```bash
hf jobs run \
  --namespace abdelstark \
  --hardware a10g-large \
  --timeout 2h \
  --image ghcr.io/abdelstark/lewm-rs:latest \
  -- bash -c "lewm-train train --config configs/pusht.toml --max-steps 7500"
```

### 9.4 Cloud T3 FULL

```bash
hf jobs run \
  --namespace abdelstark \
  --hardware a10g-large \
  --timeout 12h \
  --image ghcr.io/abdelstark/lewm-rs:latest \
  -- bash -c "lewm-train train --config configs/pusht.toml --resume-if-present"
```

### 9.5 Resume after preemption

```bash
hf jobs run \
  --namespace abdelstark \
  --hardware a10g-large \
  --timeout 12h \
  --image ghcr.io/abdelstark/lewm-rs:latest \
  -- bash -c "lewm-train train --config configs/pusht.toml --output-dir /run/<run_id> --resume-if-present"
```

### 9.6 Common failures

- **`SmokeFailed: slope positive`** — usually a data pipeline mis-config (action stats mismatch, wrong normalization). Re-run `compute_stats`.
- **`grad_explosion at step N`** — typically a config bug (`lr_peak` too high, `betas[1]` wrong). Lower `lr_peak`.
- **`NanDetected at step N` three in a row** — almost certainly INV-005 violated (SIGReg in BF16). Check the F32-cast call site.
- **`ParityFailed at PARITY_CHECK`** — weight import bug. Re-run `lewm-train convert` and compare layer-by-layer dumps.

---

## 10. Testing strategy

### 10.1 Test inventory

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0005-CLI-001 | `cli_parse_train_default` | unit | RFC0005-001 |
| TST-0005-CLI-002 | `cli_parse_smoke_overrides` | unit | RFC0005-001 |
| TST-0005-CLI-003 | `cli_set_key_value_override` | unit | RFC0018 integration |
| TST-0005-CLI-004 | `cli_resume_if_present_detects_dir` | integration | RFC0005-020 |
| TST-0005-CLI-005 | `cli_provenance_preamble_format` | unit | RFC0005-002 |
| TST-0005-OPT-001 | `adamw_param_partition_decay_no_decay` | unit | RFC0005-007 |
| TST-0005-OPT-002 | `adamw_step_matches_pytorch_on_toy_problem` | integration | RFC0005-007/008 |
| TST-0005-OPT-003 | `cosine_schedule_endpoints_exact` | unit | RFC0005-009/010 |
| TST-0005-ACC-001 | `grad_accumulation_loss_equiv_to_single_pass` | integration | RFC0005-003/015 |
| TST-0005-CLIP-001 | `grad_clip_norm_global` | unit | RFC0005-004 |
| TST-0005-BF16-001 | `mixed_precision_master_weights_f32` | integration | RFC0005-012 |
| TST-0005-BF16-002 | `sigreg_under_bf16_outer_is_f32` | integration | INV-005 |
| TST-0005-CKPT-001 | `checkpoint_roundtrip_burn` | integration | RFC0005-019 |
| TST-0005-CKPT-002 | `checkpoint_safetensors_mirror` | integration | RFC0005-019 |
| TST-0005-CKPT-003 | `checkpoint_atomic_rename` | integration | RFC0005-017 |
| TST-0005-CKPT-004 | `checkpoint_prune_keeps_three` | unit | RFC0005-018 |
| TST-0005-RESUME-001 | `resume_rng_bitwise_identical` | integration | RFC0005-020 |
| TST-0005-RESUME-002 | `resume_via_sigterm_simulation` | integration | RFC0005-022 |
| TST-0005-PROBE-001 | `parity_probe_artifact_written` | integration | §6.3 |
| TST-0005-SMOKE-001 | `local_smoke_loss_decreases` | integration | §8.3 |

### 10.2 Fixtures

- A miniature config `tests/fixtures/tiny.toml` that builds a tiny Jepa (hidden=32, depth=2) on the synthetic PushT fixture.
- A pickled PyTorch optimizer state for AdamW toy-problem comparison.

### 10.3 Property tests

- *P-1: schedule monotonicity in cooldown.* `lr(step) ≤ lr(step-1)` for `step > warmup_steps`. Proptest over `(warmup_steps, total_steps)`.
- *P-2: grad accumulation = single-pass.* With identical seeds, accumulating 4 micro-batches of size 16 produces the same optimizer step as one micro-batch of size 64.

### 10.4 Negative tests

- `train_aborts_on_grad_explosion_after_logging`
- `train_aborts_on_three_consecutive_nans`
- `train_refuses_to_clobber_existing_run_dir_without_resume_flag`

---

## 11. Operational considerations

### 11.1 Observability

All metrics from RFC 0003 §4.5 plus:

```
optim/lr
optim/momentum_norm
optim/exp_avg_sq_norm
optim/effective_step_norm        # step_size × grad_norm_post; the L2 of the actual param update
optim/skipped_steps              # count of NaN-skipped steps
data/queue_depth
state/{INIT,PARITY,SMOKE,…}/wall_seconds
checkpoint/written_count
checkpoint/disk_usage_gb
```

Spans:

```
training.epoch          (one per epoch)
training.step           (one per optimizer step; sampled at 1/100 if hot loop)
training.forward
training.backward
training.optim_step
training.checkpoint_save
training.parity_probe
```

### 11.2 Runbook (additional)

See §9.

### 11.3 Capacity

- GPU memory: model + optimizer (2× model in F32 master) + activations. With `B=64, T=8, BF16-mixed`, peak is ≈ 17 GB on the A10G-large 24 GB.
- Disk: 4 × `(model + optim + sidecar) ≈ 4 × 250 MB = 1 GB` per run (last 3 epoch checkpoints + safetensors mirrors).

---

## 12. Performance considerations

Hot path is the GPU; the trainer minimizes per-step host overhead:

- Single-allocation device tensors reused per step via Burn's autograd graph reuse.
- Metrics are gathered to host every step but flushed every `N=100` steps (configurable) to avoid sync-points.
- `criterion`'s F32 cast in SIGReg adds ≈ 0.5 % step overhead; acceptable.

See [RFC 0014 §5](0014-performance-engineering.md) for the bench plan.

---

## 13. Security considerations

- HF token consumed via env var only; never logged.
- `hf-hub` operates with read-only scope for downloads; write scope only for the explicit upload step.
- Resume protocol verifies the git short SHA of the sidecar against current HEAD; mismatch is a warning, not an error (to allow legitimate code changes mid-run).

---

## 14. Alternatives considered

- **A1 — Use `burn-train`'s `Trainer`.** Considered. Rejected because `burn-train`'s state machine doesn't quite fit our 9-state pipeline (no native parity-check / smoke states). We use it for low-level checkpoint serialization but write our own outer loop.
- **A2 — DeepSpeed-style ZeRO.** Out of scope; single-GPU only per PRD.
- **A3 — Lion optimizer.** Default AdamW; Lion permitted in lambda sweep only.
- **A4 — Periodic eval mid-epoch.** Considered; we keep epoch-boundary eval for v1 since the run is short.

---

## 15. Acceptance criteria

- [ ] All TST-0005-* pass on `linux-x86_64` (CPU + CUDA-shimmed) and `aarch64-darwin` (CPU only).
- [ ] CLI passes `--help` golden tests.
- [ ] Smoke run on synthetic data completes in ≤ 2 minutes locally.
- [ ] Checkpoint roundtrip is bit-identical (Burn record).
- [ ] Resume from SIGTERM produces an identical next-batch hash compared to a non-crashing run.

---

## 16. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | Burn `AdamW` numerical drift vs PyTorch | M | M | TST-0005-OPT-002 toy comparison; tolerance 1e-6 |
| R-2 | BF16 autocast leaks F32 into hot path | L | M | RFC 0003 INV-005 tests; tracer log of dtypes in critical span |
| R-3 | Atomic-rename race on shared FS | L | L | We assume local NVMe; HF Jobs ephemeral disk is fine |
| R-4 | Resume RNG drift due to `rand_chacha` version | L | H | `rand_chacha` version pinned; unit-tested across versions |

---

## 17. Open questions

OQ-2005-1 — Should we expose `--ema-encoder` as an experimental flag? Out of scope for v1 (architectural change). v2 maybe.

---

## 18. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0005.*
