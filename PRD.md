# LeWM-rs — Product Requirements Document & Specification

**Project codename:** lewm-rs
**Author:** Abdel
**Status:** v0.1 draft, locked for execution
**Date:** May 12, 2026
**License:** MIT (matches upstream lucas-maes/le-wm)

---

## 0. TL;DR

Pure-Rust end-to-end reproduction and extension of LeWorldModel (Maes et al., 2026), the first stable single-loss JEPA world model from pixels. Phase 1 reproduces the published PushT result in Rust using Burn for training and Tract for CPU inference. Phase 2 ports the same Rust stack to real LeRobot SO-100 pick-and-place data and ships a trained checkpoint with the same planning evaluation protocol. Everything runs on Hugging Face Jobs on a single A10G-large GPU. Total budget ceiling 200 USD, target spend 60-90 USD. Everything trained, evaluated, packaged, and documented is published to the Hub: code repo, two model repos with multi-epoch checkpoints, two dataset mirrors with provenance, one Space hosting the Tract CPU planning demo, one paper-style writeup.

This is not a toy port. The deliverable is a reference Rust implementation of LeWM with parity tested against the published PyTorch weights, a production-grade MLOps pipeline, and a real-robot extension that did not previously exist.

---

## 1. Why

LeWM is the cleanest JEPA world model published to date. Two loss terms, one hyperparameter, 15M parameters, single-GPU trainable. That elegance makes it the right target for the first serious Rust port of a modern world model. Three reasons it matters to ship this:

**Engineering.** Rust + Burn + Tract gives a single-language path from data loader to training to deployment, no Python at runtime, statically linked binaries, deterministic builds, zero pip-resolver pain on the edge. Almost no production-grade Rust ML reference exists for vision-based RL world models. Filling that gap is a credible artifact.

**Research signal.** Reproducing PushT in Rust validates that Burn is good enough for ViT-class research, not just inference. Extending to SO-100 produces the first published LeWM result on real robot footage rather than synthetic environments. Both are defensible contributions.

**Cypherpunk fit.** Verifiable robotics control needs auditable, portable inference. CPU-only Tract running a JEPA on-device aligns with the same line of work as VeriFlow and the verifiable AI thesis. The world model becomes a candidate target for zero-knowledge proving of physical AI decisions downstream. Not in scope for v1, but the foundation is laid.

---

## 2. Goals and non-goals

### Goals

1. Faithful reproduction of LeWM PushT result in pure Rust, parity-tested against the published `quentinll/lewm-pusht` weights to numerical tolerance below 1e-3 on encoder CLS output.
2. End-to-end training of the same model from scratch on PushT, reaching at least 90 percent of the paper's reported 96 percent planning success rate on the standard eval protocol (target floor: 87 percent absolute).
3. End-to-end training of the same architecture on `lerobot/svla_so100_pickplace`, with a planning-style goal-recall evaluation since no simulator exists for the real robot data.
4. Both trained models published on the Hub with checkpoints at epochs 1, 2, 5, 10 (or equivalent fractional milestones), model cards, training reports.
5. Tract CPU inference demo achieving sub-second cost computation for a 5-step planning horizon on a standard laptop CPU.
6. Paper-style writeup of the engineering and results, suitable for an arXiv tech report or a long-form blog post on the Hub.
7. Total cloud spend at or below 200 USD across smoke tests, full runs, and any reruns.

### Non-goals

- Beating the paper's numbers. Reproduction within tolerance is the win.
- GPU inference in Rust. Tract is CPU only by design and that is the deployment story.
- Distributed training. Single GPU only.
- New JEPA architectures. The architecture is fixed by upstream.
- A graphical eval visualizer beyond the existing rerun.io workflow used by LeRobot.
- Production-hosted Inference Endpoint. Demo Space only.

---

## 3. Scope and deliverables

| ID | Deliverable | Format | Location |
|----|------------|--------|----------|
| D1 | `lewm-rs` source repo | Rust workspace, MIT | github.com/AbdelStark/lewm-rs |
| D2 | PushT trained checkpoints (epoch 1, 2, 5, 10) | Burn record `.mpk` + Safetensors mirror | `abdelstark/lewm-rs-pusht` model repo |
| D3 | SO-100 trained checkpoints (epoch 1, 2, 5, 10) | Burn record `.mpk` + Safetensors mirror | `abdelstark/lewm-rs-so100-pickplace` model repo |
| D4 | Dataset mirrors with provenance manifests | LeWM HDF5 untouched; SO-100 with Rust loader doc | `abdelstark/lewm-pusht-mirror`, `abdelstark/so100-pickplace-lewm-ready` |
| D5 | Training reports (parity, full PushT, full SO-100) | Trackio runs + Markdown summaries committed to repo | Repo `reports/` + Trackio public links |
| D6 | Inference report | Wall-clock, FLOPs, peak memory on CPU laptop class | Repo `reports/inference.md` |
| D7 | Tract CPU planning Space | Gradio Space wrapping the Rust binary via Python bridge | `abdelstark/lewm-rs-demo` Space |
| D8 | Paper-style writeup | Markdown + assets, also a PDF render | Repo `paper/` and a Hub blog post |
| D9 | Cost ledger | Per-phase HF Jobs spend with screenshots | Repo `reports/cost.md` |

---

## 4. Background, technical audit findings

### 4.1 Upstream LeWM architecture, verified from source

From `jepa.py` and `module.py` in lucas-maes/le-wm:

**JEPA top-level wrapper.** Holds an encoder, an action encoder, a predictor, a projector and a pred_proj. The encoder is an HF `transformers`-style ViT consumed via `output.last_hidden_state[:, 0]` (CLS token). Forward path: pixels at shape `(B, T, C, H, W)` are flattened to `(B*T, C, H, W)`, run through the ViT, CLS token taken, projector applied, reshaped back to `(B, T, D)`.

**ARPredictor.** Takes embeddings `(B, T, D)` and action embeddings `(B, T, A_emb)` and predicts next-step embeddings. Architecture: learned positional embedding, dropout, then a stack of `ConditionalBlock`s using AdaLN-zero conditioning on the action embedding.

**ConditionalBlock.** Standard pre-norm transformer block with self-attention (causal, scaled-dot-product) and a feedforward MLP, both modulated by AdaLN-zero shift/scale/gate from the action embedding. Final adaLN linear is zero-initialized so each block starts as identity.

**Embedder (action encoder).** `Conv1d(kernel=1) + Linear + SiLU + Linear`. The Conv1d-k1 is mathematically a Linear but we preserve the exact graph shape for weight loading.

**MLP heads.** Two-layer MLP, configurable norm function (BatchNorm1d in the upstream HF config) and GELU activation.

**SIGReg loss.** Sketch Isotropic Gaussian Regularizer. Sample 1024 random unit-norm projections of dimension `D`, project all latents onto them, compute the Epps-Pulley statistic against the standard normal characteristic function across 17 knots in [0, 3]. Loss is the mean across projections and time of `(cos.mean - phi_window).square + sin.mean.square` weighted by trapezoid-rule weights times a Gaussian window. This is the only regularizer.

**Training objective.** `L = L_pred + lambda * L_sigreg`, where `L_pred` is MSE between predicted and target next-step embeddings (target is the encoder's own forward pass on the next frame, no EMA, no stop-gradient). `lambda` is the single tunable hyperparameter.

**Rollout / planning.** Autoregressive with a sliding history window of 3 by default. Given a start image and an action sequence of length T, encode the start, then for t in 0..T-history, take the last `history_size` embeddings, run the predictor, concatenate, advance.

**Cost function for planning.** MSE between predicted final-step embedding and goal embedding. Cross Entropy Method (CEM) wraps this for action search.

### 4.2 Datasets, verified

| Dataset | Format | Size | Action dim | Image | Source |
|---------|--------|------|-----------|-------|--------|
| `quentinll/lewm-pusht` | `pusht_expert_train.h5.zst` | ~920k frames | 2 | 224x224 RGB | HF dataset |
| `lerobot/svla_so100_pickplace` | Parquet + MP4, LeRobot v2.1 | 19,631 frames across 50 episodes, two camera views | 6 | 480x640 RGB native | HF dataset |

PushT is ~50x larger and is the right vehicle for the parity and stack-validation phase. SO-100 needs a dedicated Rust LeRobot-v2.1 loader writing into the same tensor shapes the model already consumes (frames at 224x224, action vectors at 6-D).

### 4.3 Tooling, verified

**Burn 0.20.1.** CUDA backend with Fusion enabled by default, Autodiff wrapper, BF16/F32, gradient accumulation, Safetensors and PyTorch import. Has Linear, GELU, Dropout, LayerNorm, MultiHeadAttention, AdamW, gradient clipping, learning rate schedulers, TUI training dashboard.

**Tract 0.22.1.** CPU only, 85 percent of ONNX backends pass, ViT-class models proven to run. Slower than ORT by ~2x historically but adequate for sub-second cost computation on a 15M model.

**HF Jobs.** Per-minute billing, A10G-large 24 GB at 1.50 USD/hr, L4 24 GB at 0.80 USD/hr, A100 80 GB at 2.50 USD/hr. Free during build. Default 30 minute timeout, must be raised.

**ml-intern.** Apache-2.0 HF agent CLI with HF docs, HF Jobs launcher, GH code search, sandbox tools, MCP. Used for dataset scouting, hyperparameter sweeps, and the build-fix loop on the Rust toolchain.

### 4.4 Risks identified

1. **Burn ViT throughput.** Burn on CUDA has not been publicly benchmarked on ViT to the same level as PyTorch's torch.compile path. We may see 1.5x-3x slower throughput per step. Mitigation: budget allows 2x wall-clock vs paper, hardware sized accordingly.
2. **SIGReg numerical stability.** The Epps-Pulley test uses high-frequency trig ops on projections. Float32 is required, BF16 mixed precision must keep the SIGReg path in F32. Verified in the upstream code that `t`, `phi`, `weights` are F32 buffers.
3. **PyTorch weight import correctness.** The reference checkpoint uses HF `transformers` ViT internals (rotary positional embedding interpolation, specific layer-norm placements). Parity test must run on the actual loaded weights, not on randomly initialized ones.
4. **SO-100 small data, no simulator.** 19,631 frames is small for from-scratch JEPA training. We mitigate by: (a) warm-starting the encoder from the PushT-trained model, (b) using a recall-based eval that compares predicted latent trajectories to held-out expert demonstrations rather than requiring a simulator.
5. **MP4 decode in Rust.** SO-100 video frames are MP4-encoded per-episode. Need a Rust-callable decoder. `ffmpeg-next` crate is the proven path. Falling back to a Python pre-decode step is acceptable since the rules allow Python at the dataset prep edge.
6. **HF Jobs default timeout.** 30 minutes will silently kill long runs. The runbook explicitly sets `--timeout 12h` on every full training launch.

---

## 5. Architecture

### 5.1 Workspace layout

```
lewm-rs/
├── Cargo.toml                       # workspace manifest, resolver=2
├── README.md
├── LICENSE                          # MIT
├── crates/
│   ├── lewm-core/                   # model, losses, no I/O
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── vit.rs               # ViT encoder, interpolated pos enc
│   │   │   ├── predictor.rs         # ARPredictor, ConditionalBlock
│   │   │   ├── embedder.rs          # action Embedder
│   │   │   ├── mlp.rs               # projector, pred_proj
│   │   │   ├── jepa.rs              # top-level JEPA wrapper, encode/predict/rollout
│   │   │   ├── losses/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── prediction.rs    # MSE next-embedding loss
│   │   │   │   └── sigreg.rs        # SIGReg, F32 fixed
│   │   │   └── config.rs            # JepaConfig, mirrors HF config.json
│   │   └── tests/
│   │       ├── parity_encoder.rs    # vs reference weights
│   │       ├── parity_predictor.rs
│   │       └── parity_sigreg.rs
│   ├── lewm-data/                   # data loaders
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── pusht_hdf5.rs        # streaming HDF5 reader
│   │   │   ├── lerobot_v21.rs       # Parquet+MP4 loader for SO-100
│   │   │   ├── transform.rs         # resize, normalize, history windowing
│   │   │   └── batch.rs             # collate, action stacking
│   │   └── tests/
│   ├── lewm-train/                  # training binary
│   │   ├── src/
│   │   │   ├── main.rs              # CLI: train, smoke, parity
│   │   │   ├── trainer.rs           # epoch loop, grad accum, checkpoints
│   │   │   ├── optim.rs             # AdamW, cosine schedule, warmup
│   │   │   ├── monitor.rs           # metrics emission
│   │   │   └── checkpoint.rs        # save/load, HF upload hook
│   │   └── configs/
│   │       ├── pusht.toml
│   │       └── so100.toml
│   ├── lewm-plan/                   # CEM planner, eval driver
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── cem.rs
│   │   │   ├── eval_pusht.rs
│   │   │   └── eval_so100.rs
│   │   └── tests/
│   ├── lewm-infer/                  # Tract inference binary
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── onnx_runner.rs
│   │   │   └── plan.rs
│   │   └── benches/
│   │       └── cost_bench.rs
│   ├── lewm-telemetry/              # OTLP, Trackio bridge
│   │   └── src/lib.rs
│   └── lewm-hub/                    # huggingface_hub-rs wrappers
│       └── src/lib.rs
├── python/                          # Python only at the edges
│   ├── convert_reference.py         # HF weights.pt -> Safetensors -> Burn record
│   ├── decode_so100_to_h5.py        # MP4 -> H5 for fast Rust loading
│   ├── upload_checkpoints.py
│   └── plot_curves.py
├── jobs/                            # HF Jobs launch specs
│   ├── smoke_pusht.yaml
│   ├── train_pusht.yaml
│   ├── train_so100.yaml
│   └── eval.yaml
├── reports/
├── paper/
└── .ml-intern/                      # ml-intern config and saved sessions
    └── cli_agent_config.json
```

### 5.2 Model specification, locked

These values are reproduced verbatim from the upstream config.json convention. They are the contract the Rust implementation must match.

```toml
[encoder]                            # HF transformers ViT
size = "small"                       # patch=16, hidden=384, depth=12, heads=6
patch_size = 16
image_size = 224
use_mask_token = false
pretrained = false                    # train from scratch end-to-end

[action_encoder]                     # Embedder
input_dim = 2                        # 2 for PushT, 6 for SO-100
smoothed_dim = 16
emb_dim = 64
mlp_scale = 4

[predictor]                          # ARPredictor
num_frames = 16
depth = 6
heads = 6
mlp_dim = 1536
input_dim = 384
hidden_dim = 384
output_dim = 384
dim_head = 64
dropout = 0.0
emb_dropout = 0.0

[projector]                          # MLP, BatchNorm1d
input_dim = 384
hidden_dim = 1536
output_dim = 384

[pred_proj]                          # MLP, BatchNorm1d
input_dim = 384
hidden_dim = 1536
output_dim = 384

[loss]
lambda_sigreg = 1.0                  # tuned via ml-intern, expected near 1.0
sigreg_knots = 17
sigreg_num_proj = 1024

[training]
history_size = 3
horizon = 8                          # T per sample for rollout target
batch_size = 64
grad_accum_steps = 2                 # effective batch 128
optimizer = "adamw"
lr_peak = 3e-4
lr_min = 1e-5
warmup_steps = 1000
weight_decay = 0.05
betas = [0.9, 0.95]
epochs = 10
precision = "bf16_mixed"
seed = 0
```

Total parameter count target: 14.8M to 15.2M, matching the paper.

### 5.3 Burn implementation contract

Each module in `lewm-core` follows the same pattern:

```rust
#[derive(Module, Debug)]
pub struct Vit<B: Backend> {
    patch_embed: PatchEmbed<B>,
    cls_token: Param<Tensor<B, 3>>,
    pos_embed: Param<Tensor<B, 3>>,
    blocks: Vec<Block<B>>,
    norm: LayerNorm<B>,
}

#[derive(Config)]
pub struct VitConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub mlp_ratio: f32,
}

impl VitConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Vit<B> { ... }
}

impl<B: Backend> Vit<B> {
    pub fn forward(&self, pixels: Tensor<B, 4>, interpolate_pos: bool) -> Tensor<B, 3> { ... }
    pub fn cls(&self, hidden: Tensor<B, 3>) -> Tensor<B, 2> { ... }
}
```

The attention layer is implemented manually rather than using `burn::nn::attention::MultiHeadAttention` because the upstream uses pre-norm with `F.scaled_dot_product_attention(is_causal=True)` and we want byte-exact equivalence. Causal masking is built once and reused.

SIGReg is implemented in `losses/sigreg.rs` with all internal tensors forced to F32 via `B::FloatElem` casting, even when the outer training pass is BF16.

### 5.4 Data flow per training step

```
Stream HDF5 (or Parquet+MP4)
       │
       ▼  io thread pool, 4 workers
Decode + resize 224x224 + normalize
       │
       ▼  channel of (pixels, action)
Sample window of T=8 frames
       │
       ▼  Burn tensor on device
Encoder forward (no_grad on the target arm)
       │                            │
       │                            ▼ target embeddings (no grad)
       ▼
ActionEncoder + Predictor forward
       │
       ▼ predicted embeddings (with grad)
       │
       ├─► L_pred = MSE(pred, target)
       │
       └─► L_sigreg = SIGReg(projector(target).cast::<F32>())
              │
              ▼
        L = L_pred + lambda * L_sigreg
              │
              ▼
        backward(), AdamW step, grad clip
              │
              ▼
        emit metrics: step, loss, l_pred, l_sigreg, grad_norm,
                      lr, throughput, gpu_mem
```

The target arm is the encoder applied to the next frames in the window. There is no EMA, no stop-gradient. The encoder receives gradient from both the target arm (through `L_sigreg`) and the source arm (through `L_pred`). This is what "end-to-end stable" means in the paper.

### 5.5 Training pipeline state machine

```
states: INIT -> PARITY_CHECK -> SMOKE -> WARMUP -> STEADY -> COOLDOWN -> EVAL -> UPLOAD -> DONE
```

Each transition writes a checkpoint and a transition record to disk. A run can resume from any checkpoint. Crash inside STEADY triggers automatic resume from the latest checkpoint with the same RNG state.

---

## 6. Observability and MLOps

Production-grade means: nothing about a training run depends on a human watching it. Everything that matters is captured, queryable, and reproducible after the fact.

### 6.1 Metrics, where they go

**Trackio** is the HF-native experiment tracker. It is the primary dashboard and the public artifact for D5. Local Trackio runs are uploaded to a Trackio Space owned by the project. Metrics emitted per step:

- `loss/total`, `loss/pred`, `loss/sigreg`
- `optim/lr`, `optim/grad_norm`, `optim/grad_norm_clipped`
- `model/encoder_cls_var`, `model/encoder_cls_mean_abs`  (collapse detector)
- `model/predictor_output_var`
- `throughput/samples_per_sec`, `throughput/tokens_per_sec`
- `system/gpu_mem_used_gb`, `system/gpu_util_pct`, `system/cpu_util_pct`

Per epoch we also emit `eval/planning_success_rate` on a 50-episode subset, and `eval/latent_rollout_error` for SO-100.

**Tensorboard** is written in parallel as a portability backstop. Anyone without HF account can still inspect runs.

**OpenTelemetry traces** are emitted from the Rust trainer using the `tracing` + `tracing-opentelemetry` crates. Spans: `epoch`, `step`, `forward`, `backward`, `optim_step`, `data_load`, `checkpoint_save`. The default backend is the optional self-hosted stack under `infra/otel`; smoke training and CI leave the OTLP endpoint unset. This is what makes the system debuggable when something stalls, since a "loss is fine but throughput collapsed" issue is invisible in scalar logs but obvious in span timing.

**Structured logs** via `tracing` with JSON formatter to stdout. HF Jobs captures stdout to the job log. Each line carries a `run_id`, `phase`, `step`, `wall_time_ms` so reports can be reconstructed by `jq` post-hoc.

### 6.2 Collapse detection

The single highest-value runtime check is collapse detection. JEPA without SIGReg collapses fast, with SIGReg it should not but I want a real check on the wire. Every 100 steps we compute, on a held-out 32-frame batch:

- `mean_abs_cls` should stay below 5.0
- `cls_variance_per_dim_mean` should stay above 0.05
- `cosine_similarity_between_random_pairs` should stay below 0.85

If any of these fails for 3 consecutive checkpoints, the trainer prints a CRITICAL line to stdout, writes a `collapse_suspected.json` file, and continues. The post-run report flags the run.

### 6.3 Checkpointing

Every epoch boundary writes:
- `step_{N}.mpk` Burn record, full model + optimizer state
- `step_{N}.safetensors` for portability
- `step_{N}.json` metadata: config hash, RNG state, git SHA, wall time
- `step_{N}.parity.json` results of the per-epoch parity probe on a fixed input

Last 3 checkpoints kept on disk during a run, all 10+ kept on the Hub.

### 6.4 Local validation before any HF Jobs spend

Three local stages, all on Abdel's hardware (StarkWare workstation, assumed laptop or workstation class, no GPU strictly required for stages 1 and 2):

**Stage L0, unit tests.** `cargo test --workspace`. Pure logic, no GPU.

**Stage L1, parity probe.** Load reference `quentinll/lewm-pusht` weights, run on a fixed 1-batch input, assert per-layer outputs match PyTorch reference dump to 1e-4. This catches 95 percent of weight-loading and op-mismatch bugs at zero cost.

**Stage L2, CPU smoke train.** 50 steps of training on 16-sample subset using the NdArray backend. Verifies that loss decreases at all, the optimizer steps, and checkpointing roundtrips. Runs in 2 to 5 minutes on a laptop.

Only after L0, L1, L2 are all green does any HF Jobs launch happen.

### 6.5 HF Jobs launches, three tiers

```
T1 SMOKE   L4 24GB         $0.80/hr   30 min cap     verifies cloud env
T2 SHORT   A10G-large 24GB $1.50/hr   2  h  cap      first 1 epoch, validates throughput
T3 FULL    A10G-large 24GB $1.50/hr   12 h cap       the actual training run
```

Each tier has a yaml in `jobs/` and is launched via `hf jobs run --namespace abdelstark --timeout <cap> --flavor <flavor> ...`. T2 must succeed before T3 launches. T3 is the only run that costs real money.

### 6.6 ml-intern usage protocol

ml-intern is the agent that runs sweeps and the build-fix loop, but it gets a tight leash because it is the easiest thing in this stack to burn money on.

**Allowed:**
- Read HF docs, paper, repo, dataset cards
- Search GitHub for similar Burn implementations
- Submit a sandbox HF Space for a 50-step Burn build test
- Run a sweep over `lambda_sigreg` in {0.3, 0.5, 1.0, 2.0, 5.0} on the T2 SHORT tier
- Generate the first draft of the README, model card, training report

**Forbidden without explicit human approval:**
- Launching any T3 FULL run
- Launching any A100 or larger
- Launching jobs without a `--timeout`
- Modifying anything in `crates/lewm-core/src/losses/` (the loss math must be locked)
- Editing `jobs/*.yaml` to a hardware tier above A10G-large

These are encoded in `.ml-intern/cli_agent_config.json` and reinforced in the agent system prompt loaded at session start. All ml-intern sessions are auto-uploaded to a private HF dataset so the actions are auditable post-hoc.

---

## 7. Financial and resource plan

### 7.1 Hardware recommendation, locked

**Primary training hardware: 1x Nvidia A10G-large on HF Jobs, 1.50 USD/hr.**

Reasoning. The paper used L40S (181 TFLOPS) and reported "a few hours" for the small model. A10G is roughly 31 TFLOPS dense, A10G-large bundles more host RAM and CPU which helps the data loader pipeline. L4 (30 TFLOPS, similar compute, less host) would also work but the 24GB VRAM on A10G-large gives room for the larger batches in BF16 without aggressive grad accumulation. We are not budget-bound to L4 since the difference for a full run is roughly 5 USD.

A100 (2.50 USD/hr, 312 TFLOPS BF16) is faster and would finish PushT in 2 to 3 hours instead of 6 to 8. The cost difference for one run is small (about 7 USD savings on wall clock vs A10G total) but A10G is better practice for the "commodity hardware reproducibility" framing of the writeup. We default to A10G-large and document the A100 option.

L40S (1.80 USD/hr, the paper's hardware) is the closest like-for-like comparison and is reserved for the final headline run if A10G hits unexpected throughput issues.

### 7.2 Cost ledger

All numbers in USD. Conservative side of each estimate.

| Phase | Hardware | Wall clock | Subtotal | Cum. |
|-------|----------|-----------|----------|------|
| P0 setup, env, parity | local | n/a | 0 | 0 |
| P1 PushT T1 SMOKE (4 attempts incl. fails) | L4 | 4 x 30min | 1.60 | 1.60 |
| P2 PushT T2 SHORT (1 epoch, 3 attempts) | A10G-large | 3 x 1.5h | 6.75 | 8.35 |
| P3 PushT T3 FULL (10 epochs) | A10G-large | 8h | 12.00 | 20.35 |
| P4 PushT T3 FULL rerun (1 fail margin) | A10G-large | 8h | 12.00 | 32.35 |
| P5 ml-intern lambda sweep (5 configs, 1 epoch each) | A10G-large | 5 x 1.5h | 11.25 | 43.60 |
| P6 SO-100 dataset prep job (one-shot, MP4 decode) | CPU XL | 2h | 2.00 | 45.60 |
| P7 SO-100 T1 SMOKE (3 attempts) | L4 | 3 x 30min | 1.20 | 46.80 |
| P8 SO-100 T2 SHORT | A10G-large | 1.5h | 2.25 | 49.05 |
| P9 SO-100 T3 FULL (10 epochs, smaller dataset, faster) | A10G-large | 4h | 6.00 | 55.05 |
| P10 SO-100 T3 FULL rerun margin | A10G-large | 4h | 6.00 | 61.05 |
| P11 Eval jobs both models (CEM is GPU-friendly) | A10G-small | 2 x 2h | 4.00 | 65.05 |
| P12 Tract inference benchmarking | CPU XL | 1h | 1.00 | 66.05 |
| P13 Demo Space hosting (T4 small, monthly) | T4 small | 30 days idle ~ 50% | ~15.00 | 81.05 |
| P14 Contingency, 20 percent | | | ~16.00 | 97.05 |

**Target spend: 80-100 USD. Hard ceiling: 200 USD.**

Notes:
- HF Jobs is billed per minute, so any run that crashes early costs less than the line item.
- The 20 percent contingency covers Burn perf surprises that force one extra rerun.
- The demo Space is hosted on T4 small (0.40 USD/hr) with auto-pause; assuming 50 percent uptime over 30 days that is ~144 USD a month. We will set auto-pause aggressively to keep it under 15 USD a month, and document the option to run it on CPU Basic (free) at slower speed.

### 7.3 Cost controls

- Every HF Jobs invocation passes `--timeout <hard-cap>`. Default 30min is too short and silently kills.
- `hf jobs cancel <id>` is wired into the run-wrapper script as a SIGINT handler.
- HF Billing alert thresholds set at 50 USD and 100 USD.
- Weekly cron in ml-intern dumps the org-level `hf jobs` history into `reports/cost.md`.
- Smoke jobs cap at 30 min via `--timeout 30m` so a forgotten launch costs at most 0.40 USD on L4.

---

## 8. Phase plan and timeline

Timeline assumes ~10 to 15 hours of focused engineering per phase plus async ml-intern work and waits on HF Jobs. Calendar weeks shown for a focused part-time effort given Abdel's day job at StarkWare. Total elapsed: 6 to 8 weeks. Total active engineering: 60 to 90 hours.

### Phase 0 — Bootstrap, week 1

- Scaffold workspace, CI, formatting, basic burn-cuda hello tensor
- Pull and decompress `lewm-pusht`, document schema in `reports/pusht_schema.md`
- Pull reference HF checkpoint via `hf download quentinll/lewm-pusht`
- Set up Trackio Space, optional self-hosted OTel exporter, HF Jobs org billing
- Set up ml-intern config with the allowed/forbidden list from section 6.6
- **Exit gate:** `cargo build --workspace` green, reference weights on disk, Trackio dashboard reachable.

### Phase 1 — Parity, week 2

- Implement `lewm-core`: ViT, ARPredictor, Embedder, MLP, JEPA wrapper, SIGReg
- Implement `python/convert_reference.py`: weights.pt -> Safetensors -> Burn record via `burn-import`
- Write parity tests against the loaded reference weights on a fixed `(B=4, T=4, C=3, H=224, W=224)` input
- **Exit gate:** encoder CLS output matches PyTorch reference to 1e-4, predictor output to 1e-4, SIGReg statistic value matches to 1e-3 (some tolerance due to random projection RNG difference)
- This phase costs $0 because everything runs locally or on CPU.

### Phase 2 — PushT smoke + short, week 3

- Implement `lewm-data::pusht_hdf5` streaming loader, validate throughput on local CPU
- Implement `lewm-train::trainer` with monitor, checkpoint, OTel spans
- Local L2 smoke: 50 steps NdArray CPU, expect loss to decrease in the first 20 steps
- Cloud T1 SMOKE on L4: 200 steps, verify GPU utilization > 50 percent, throughput report
- Cloud T2 SHORT on A10G-large: 1 epoch, projected wall clock, collapse detector verified active
- **Exit gate:** T2 SHORT reports loss curve consistent with the paper's first-epoch shape, no collapse signal.

### Phase 3 — PushT full, week 4

- Cloud T3 FULL on A10G-large, 10 epochs, ~8 hours
- ml-intern runs the 5-point lambda_sigreg sweep in parallel on 1-epoch T2 SHORT jobs (read-only role: it can launch sweep jobs in a separate namespace, results merged into the final report)
- After main run completes: eval job on the standard 50-episode planning protocol
- **Exit gate:** planning success rate >= 87 percent (paper reports 96 percent), no collapse, all 4 checkpoints uploaded to `abdelstark/lewm-rs-pusht`, training report rendered.

### Phase 4 — SO-100 prep and short, week 5

- Implement `lewm-data::lerobot_v21` loader (Parquet for actions, video frames via either `ffmpeg-next` Rust crate OR a Python pre-decode to H5 if Rust path is fragile)
- Resample SO-100 videos to 224x224 at 10 Hz (downsample from 30 Hz) and dump to a single HDF5 archive uploaded to `abdelstark/so100-pickplace-lewm-ready`
- Init the SO-100 model from the PushT epoch-10 encoder (warm start) while keeping a from-scratch arm as a control
- Adapt config: `action_encoder.input_dim = 6`, action vector normalization parameters computed from the dataset
- T1 SMOKE on L4, T2 SHORT on A10G-large
- **Exit gate:** loss decreases, collapse detector clean, throughput acceptable.

### Phase 5 — SO-100 full + eval, week 6

- Cloud T3 FULL, 10 epochs on the 19,631-frame dataset, expected ~4 hours
- Eval: split off 5 episodes as held-out, for each held-out episode encode start and goal frames and roll the predictor out using the recorded expert action sequence, measure mean squared error in latent space and Spearman rank correlation between predicted and recorded latent trajectories
- Optional reach-style: nearest-neighbor search in the latent space to identify "closest training frame" for each predicted step, build a visual ribbon for the writeup
- **Exit gate:** latent rollout error below an absolute threshold determined by the from-scratch control + warm-start delta is positive, both checkpoints uploaded.

### Phase 6 — Tract inference, week 7

- Export PushT and SO-100 models to ONNX via Burn's ONNX export OR via tract-OPL via NNEF if ONNX export hits an unsupported op
- Build `lewm-infer` binary: loads ONNX, runs encoder + predictor for a single planning cost computation, reports wall clock
- Bench on commodity laptop CPU: target sub-second for a 5-step horizon with 16 action candidates (paper reports 48x speedup vs DINO-WM, our absolute time on laptop CPU should be 200ms to 800ms)
- Build a minimal Gradio Space that uploads start and goal images, calls the Rust binary, displays the predicted cost and the best action sequence
- **Exit gate:** Space live, inference report committed.

### Phase 7 — Writeup, week 8

- Draft the paper-style writeup in `paper/lewm-rs.md`: motivation, architecture as implemented, parity test results, PushT result vs paper, SO-100 extension, Tract benchmarks, ablations from the lambda sweep, lessons learned
- Render to PDF
- Cross-link from each model card and the repo README
- Publish a short Hub blog post pointing to all artifacts
- **Exit gate:** writeup committed, blog post live, cost ledger finalized.

---

## 9. Evaluation protocol

### 9.1 PushT evaluation

Identical to the paper. CEM planner with 5 iterations, 100 elites out of 1000 candidates, horizon 5, history 3. 50 episodes from the held-out split. Reported metric: percentage of episodes where the final state is within the configured success tolerance of the goal state. Paper reports 96 percent. Target floor: 87 percent (90 percent of paper).

A `--seed` is fixed for the action candidate distribution so the eval is bit-reproducible.

### 9.2 SO-100 evaluation

There is no simulator for the real SO-100 dataset, so a sim-style success rate cannot be measured. The eval protocol is:

1. Hold out 5 episodes (10 percent of the 50-episode dataset).
2. For each held-out episode: encode the start frame and the goal frame (last frame of the episode), then roll the predictor forward using the recorded expert action sequence.
3. Measure (a) mean MSE between predicted latent trajectory and target latent trajectory (from encoding actual frames), and (b) Spearman rank correlation between predicted-trajectory and target-trajectory pairwise distances. (b) is more interpretable since it is invariant to latent-space scale.

Headline metric: Spearman rank correlation >= 0.6 averaged over held-out episodes is the success bar. Below 0.4 we declare the result null and document it honestly.

We also report the warm-start delta: latent-MSE for the from-scratch SO-100 model minus latent-MSE for the PushT-warm-started SO-100 model. A positive delta is evidence that PushT pretraining transfers, which would be a small but real contribution.

### 9.3 Inference benchmark

Reported in `reports/inference.md`. Two machines: cloud CPU XL (16 vCPU, $1.00/hr) and a representative laptop spec (Apple Silicon M-series via Tract's Metal-less CPU path, or Intel i7 ultrabook class). For each machine, for each model:

- Cold-start time (load + first inference)
- Steady-state time per planning cost computation (encoder + predictor rollout for 5 steps, 16 action candidates)
- Peak RSS

Target: steady-state under 1.0 second on the laptop, under 0.3 seconds on CPU XL.

---

## 10. Acceptance criteria

The project ships when every box below is checked.

- [ ] All `cargo test --workspace` green on Linux x86-64 in CI
- [ ] `parity_encoder` test passes at 1e-4 tolerance against `quentinll/lewm-pusht` weights
- [ ] `parity_predictor` test passes at 1e-4 tolerance
- [ ] `parity_sigreg` test passes at 1e-3 tolerance
- [ ] PushT model published with 4 checkpoints, model card, training report
- [ ] PushT planning success rate >= 87 percent reported
- [ ] SO-100 model published with 4 checkpoints, model card, eval report
- [ ] SO-100 latent rollout Spearman >= 0.6, or null result honestly documented
- [ ] Tract CPU inference: encoder + 5-step planner cost under 1.0 second on laptop CPU
- [ ] Demo Space live and reachable
- [ ] Cost ledger committed, total spend <= 200 USD
- [ ] Writeup published, blog post live
- [ ] No data, weights, or code that prevents another person from reproducing this end-to-end on their own HF account

---

## 11. Open questions and decisions deferred

These are intentionally not decided yet. They will be decided during execution and recorded as ADRs in `docs/adr/`.

1. ONNX export from Burn vs Burn-record-direct serving via `lewm-infer`. Burn has an ONNX export crate but its op coverage may miss AdaLN. If export fails we ship a Rust-native loader that reads the Burn record directly into a hand-rolled Tract-equivalent CPU runtime, which is more work but more portable.
2. ffmpeg-next vs Python pre-decode for SO-100 video. Default to Python pre-decode (allowed under "Python at the edges" rule) for v1; revisit if a future v2 wants embedded device deployment.
3. AdamW vs Lion as the optimizer for SO-100 warm-start. Default AdamW for parity with PushT, allow ml-intern to test Lion in the lambda sweep.
4. Whether to expose a Python binding via PyO3 for the trained Rust model. Out of scope for v1 unless a clear user materializes.

---

## 12. Risks and mitigations, condensed

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Burn ViT throughput 2-3x slower than PyTorch | Medium | Wall clock 2x, cost 2x | Budget allows; ceiling 200 USD; A100 fallback |
| Burn ONNX export missing AdaLN op | Medium | Tract demo blocked | Ship Burn-record-direct loader as fallback |
| SIGReg numerical drift in BF16 | Low | Training silent fail | Force F32 path inside SIGReg, verified in code |
| Reference weight import bug | Medium | Parity test fails | Layer-by-layer parity, ml-intern build-fix loop |
| SO-100 dataset too small | High | Null result | Warm start from PushT, document honestly, fallback to recall metric only |
| HF Jobs spec changes | Low | Pricing change | Cost ledger updated weekly, alerts at 50/100 USD |
| ml-intern blows the budget | Medium | Budget overrun | Tier restrictions encoded in cli_agent_config, all sessions audited |

---

## 13. Provenance and licensing

- Code: MIT, same as upstream lucas-maes/le-wm.
- Trained checkpoints: Apache-2.0, with the dataset attribution in the model card.
- Datasets: PushT mirrored at MIT under `abdelstark/lewm-pusht-mirror` with the original license preserved; SO-100 derivative under the original LeRobot license, attributed to `lerobot/svla_so100_pickplace`.
- Writeup: CC-BY-4.0, with citation to `maes_lelidec2026lewm` mandatory.

All artifacts carry the upstream citation block in their cards.

---

## 14. Appendix A — Reference upstream pseudocode mapping

For each Rust module, the upstream Python file and line range it implements:

| Rust crate::module | Python source | Notes |
|-------------------|---------------|-------|
| `lewm-core::vit` | HF transformers `ViTModel` | Reimplemented; use `interpolate_pos_encoding=true` semantics |
| `lewm-core::predictor` | `module.py::ARPredictor` (lines 244-285) | Includes pos embedding, dropout, transformer with ConditionalBlock |
| `lewm-core::embedder` | `module.py::Embedder` (lines 199-225) | Conv1d-k1 preserved for weight loading |
| `lewm-core::mlp` | `module.py::MLP` (lines 227-242) | BatchNorm1d default |
| `lewm-core::jepa` | `jepa.py::JEPA` (full file) | encode, predict, rollout, criterion, get_cost |
| `lewm-core::losses::sigreg` | `module.py::SIGReg` (lines 13-39) | F32 forced, knots=17, num_proj=1024 |
| `lewm-plan::cem` | `stable-worldmodel` upstream | Reimplement based on cited Williams 2015 CEM |

---

## 15. Appendix B — Sample HF Jobs YAML

```yaml
# jobs/train_pusht.yaml
hardware: a10g-large
timeout: 12h
namespace: abdelstark
image: ghcr.io/abdelstark/lewm-rs:latest
env:
  RUST_LOG: lewm=info,burn=info
  HF_TOKEN: ${HF_TOKEN}
  HF_HOME: /tmp/hf
  TRACKIO_PROJECT: lewm-rs
  TRACKIO_RUN: pusht-full-${SLURM_JOB_ID:-local}
  OTEL_EXPORTER_OTLP_ENDPOINT: ${OTEL_ENDPOINT}
command: >
  bash -c "
    hf download quentinll/lewm-pusht pusht_expert_train.h5.zst --repo-type dataset --local-dir /tmp/data &&
    zstd -f -d /tmp/data/pusht_expert_train.h5.zst -o /tmp/data/pusht_expert_train.h5 &&
    lewm-train train
      --config configs/pusht.toml
      --data-dir /tmp/data
      --output-dir /tmp/out
      --resume-if-present &&
    python python/upload_checkpoints.py
      --src /tmp/out
      --dst abdelstark/lewm-rs-pusht
  "
```

---

## 16. Appendix C — Initial `Cargo.toml` skeleton

```toml
[workspace]
resolver = "2"
members = [
    "crates/lewm-core",
    "crates/lewm-data",
    "crates/lewm-train",
    "crates/lewm-plan",
    "crates/lewm-infer",
    "crates/lewm-telemetry",
    "crates/lewm-hub",
]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "MIT"
authors = ["Abdel <abdel@starkware.co>"]
repository = "https://github.com/AbdelStark/lewm-rs"

[workspace.dependencies]
# Numerics
burn = { version = "0.20.1", default-features = false }
burn-cuda = { version = "0.20.1" }
burn-ndarray = { version = "0.20.1" }
burn-import = { version = "0.20.1" }
tract = "0.22.1"
tract-onnx = "0.22.1"
tract-nnef = "0.22.1"

# Data
hdf5-metno = "0.10"
parquet = "56"
arrow = "56"
image = "0.25"

# Telemetry
tracing = "0.1"
tracing-subscriber = "0.3"
tracing-opentelemetry = "0.27"
opentelemetry = "0.27"
opentelemetry-otlp = "0.27"

# Utilities
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"
clap = { version = "4", features = ["derive"] }
anyhow = "1"
thiserror = "2"

# HF Hub
hf-hub = "0.4"
```

---

## 17. Appendix D — Sample model card outline

```yaml
---
library_name: burn
license: apache-2.0
tags:
  - jepa
  - world-model
  - robotics
  - rust
  - burn
  - lewm
datasets:
  - quentinll/lewm-pusht
metrics:
  - planning_success_rate
base_model: quentinll/lewm-pusht
---

# lewm-rs-pusht

Pure-Rust reproduction of LeWorldModel on PushT. Trained with Burn 0.20.1 on a
single Nvidia A10G GPU on Hugging Face Jobs.

## Result

Planning success rate: 89.4 percent (paper reports 96 percent).
Parity vs reference weights: encoder CLS within 7e-5 absolute, predictor within
9e-5 absolute on a fixed seeded input batch.

## How to use

For Rust inference on CPU:

    cargo install --git https://github.com/AbdelStark/lewm-rs lewm-infer
    hf download abdelstark/lewm-rs-pusht --local-dir ckpt
    lewm-infer plan --checkpoint ckpt/step_10.mpk --start start.png --goal goal.png

For Python loading via Safetensors mirror:

    from safetensors.torch import load_file
    weights = load_file("step_10.safetensors")

## Training details

See https://github.com/AbdelStark/lewm-rs/blob/main/reports/pusht_training.md.

## Citation

    @article{maes_lelidec2026lewm,
      title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
      author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
      journal={arXiv preprint},
      year={2026}
    }
```

---

End of PRD.
