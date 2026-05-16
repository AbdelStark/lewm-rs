# `lewm-plan`

The CEM planner and the eval drivers for PushT (simulator) and
SO-100 (latent-rollout MSE).

## What it owns

- **CEM**: the Cross-Entropy Method planner, in Burn.
- **PushT eval**: wrapper around `gym-pusht` via `pyo3`, episode loop,
  success-rate aggregation.
- **SO-100 eval**: latent-rollout MSE and Spearman computation.
- **Reports**: JSON + Markdown report generators.

## Module layout

```text
lewm-plan/src/
├── lib.rs
├── bin/
│   └── lewm-eval.rs        # clap CLI
├── cem.rs                  # CEM in Burn (parity reference)
├── pusht_eval.rs           # PushT simulator wrapper, episode loop
├── so100_eval.rs           # SO-100 latent-MSE / Spearman
├── reports.rs              # eval_<dataset>.{json,md}
└── errors.rs
```

## CLI

```text
lewm-eval <dataset> [flags]

Datasets:
  pusht       Run CEM planning eval against gym-pusht.
  so100       Run latent-rollout MSE eval against held-out SO-100 episodes.

Flags:
  --checkpoint <PATH>      Burn .mpk or .safetensors.
  --num-episodes <N>       Default 50 for PushT, 5 for SO-100.
  --cem-iter <N>           CEM iterations (default 5).
  --cem-cand <N>           CEM candidates (default 1024).
  --horizon <N>            Planning horizon (default 5).
  --out <PATH>             Output JSON path.
```

## Dependencies

- `lewm-core`
- `burn`, `burn-ndarray` (CPU) / `burn-cuda` (feature)
- `pyo3` (for the PushT simulator)
- `serde_json` (for reports)
- (no dep on `lewm-train`)

## Source

[`crates/lewm-plan`](https://github.com/AbdelStark/lewm-rs/tree/main/crates/lewm-plan)
