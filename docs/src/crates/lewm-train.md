# `lewm-train`

The trainer crate: outer loop, optimizer, schedule, mixed precision,
checkpoint, resume, state machine, CLI.

## What it owns

- **Trainer**: the outer state machine driver.
- **Step**: one optimizer step (forward + backward + clip + step).
- **Optimizer wrapper**: AdamW with parameter-group split (decay /
  no-decay), pinned to `β₁ = 0.9, β₂ = 0.95, ε = 1e-8`.
- **Scheduler**: cosine + linear warmup.
- **Mixed precision**: BF16 envelope + F32 islands.
- **Checkpoint**: atomic write of the `.mpk` / `.safetensors` /
  sidecar quartet.
- **Resume**: discover last complete checkpoint, restore RNG /
  optimizer state.
- **Warmstart**: load specified components from a source checkpoint.
- **Bounded training mode**: `PushtFullLewmCore` for the current
  50 k-step PushT run.

## Module layout

```text
lewm-train/src/
├── lib.rs
├── bin/
│   └── lewm-train.rs       # clap CLI
├── trainer.rs              # outer loop, state machine
├── step.rs                 # one optimizer step
├── optim.rs                # AdamW + decay-group split
├── schedule.rs             # cosine + warmup
├── mixed_precision.rs      # cast helpers
├── checkpoint.rs           # save/load
├── resume.rs               # crash-resume
├── warmstart.rs            # selective component restore
├── config.rs               # training-specific config types
├── pusht_full.rs           # PushtFullLewmCore (bounded mode)
├── pusht_lewm.rs           # full Jepa training path (pending wire-up)
└── ...
```

## CLI

```text
lewm-train <subcommand> [flags]

Subcommands:
  train       Run the full pipeline from INIT through UPLOAD.
  smoke       Run a 50-step local smoke (NdArray CPU).
  parity      Run the parity harness.
  eval        Run eval on a checkpoint.
  convert     Convert HF/PyTorch reference weights → Burn record.

Common flags:
  --config <PATH>          Path to TOML config.
  --device <cpu|cuda>      Compute device.
  --output-dir <DIR>       Where to write checkpoints + logs.
  --max-steps <N>          Truncate total steps (for smoke / short runs).
  --resume-if-present      Restore from latest complete checkpoint.
  --set KEY=VALUE          Override a single config key (repeatable).
```

## Modes (training paths)

`lewm-train` historically supports several training paths to
accommodate the migration from the bounded model to the full Burn
ViT:

| Mode | Description |
|------|-------------|
| `pusht-minimal-lewm` | Legacy bounded `PushtFullLewmCore` label used by the current historical PushT 50 k-step Hub artifact. |
| `pusht-bounded-module-lewm` | Current checked-in `PushtFullLewmCore` bounded host-module train path; future PushT bounded artifacts use this label. |
| `so100-full-lewm` | Same `PushtFullLewmCore` adapted to 6-DOF action; SO-100 result. |
| `pusht-full-burn-jepa` | Full `lewm_core::Jepa` (303 params); wire-up pending. |

The current PushT `train` path selects the bounded host-module mode. The full
Burn/Jepa mode remains pending and must not share bounded artifact labels.

## Dependencies

- `lewm-core`, `lewm-data`, `lewm-telemetry`, `lewm-hub`
- `burn`, `burn-autodiff`, `burn-cuda` (feature-gated)
- `clap`, `toml`, `serde`
- (no dep on `lewm-infer`, `lewm-plan` for the training-only path)

## Source

[`crates/lewm-train`](https://github.com/AbdelStark/lewm-rs/tree/main/crates/lewm-train)
