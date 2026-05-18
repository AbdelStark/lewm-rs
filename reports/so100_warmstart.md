# SO-100 Warm-Start Preflight

**Date:** 2026-05-18
**Issue:** F3 / #245
**Status:** Trainer wiring and fail-closed job spec complete; launch still blocked

## Objective

Launch the SO-100 warm-start ablation from
`jobs/train_so100_warmstart.yaml`, upload artifacts under
`abdelstark/lewm-rs-so100/train/so100-warmstart-*/`, and compute the
from-scratch vs. warm-start Spearman / loss delta.

## Preflight Evidence

The GitHub issue is still open and carries the `human-only` label. The issue
also states that launch requires human approval because it costs money.

The referenced job file is now present in the checkout:

```text
jobs/train_so100_warmstart.yaml
```

The agent safety leash does not list `train_so100_warmstart.yaml` in either
`jobs_allowed` or `jobs_human_approval_required`. `scripts/launch_hf_job.py`
therefore rejects the job name:

```text
python3 scripts/launch_hf_job.py jobs/train_so100_warmstart.yaml --dry-run
launch_hf_job.py: train_so100_warmstart.yaml is not listed in the leash config
```

The job spec also fails closed inside the shell command unless
`LEWM_PUSHT_WARMSTART_MPK` points at a compatible PushT `.mpk` path in the
source model repo. This prevents accidentally launching against the stale
`configs/so100_warmstart.toml` default.

The config exists:

```text
configs/so100_warmstart.toml
training.warmstart_from = "/checkpoints/lewm-rs-pusht/step_0014400.mpk"
```

Before the 2026-05-18 release pass, `training.warmstart_from` was parsed in
`crates/lewm-train/src/config.rs` but not consumed by the training loop.

The trainer now applies `training.warmstart_from` for fresh SO-100 full-module
training starts. The transfer copies only `encoder.*`, `predictor.*`,
`projector.*`, and `pred_proj.*` from a PushT full-module record, preserves the
fresh SO-100 `action_encoder.*` parameters, resets AdamW state, and records
source path, source SHA-256, copied parameter names, preserved action-encoder
names, and dropped optimizer-state count in the train report/checkpoint record.

The configured source checkpoint is also stale. No
`step_0014400.mpk` artifact exists in `abdelstark/lewm-rs-pusht`, and the
current 50k PushT checkpoint is the older 4-dimensional
`pusht-minimal-lewm` bounded-core artifact, not the current 192-dimensional
SO-100 bounded-core layout.

Focused validation:

```text
cargo test -p lewm-train so100_warmstart -- --nocapture
```

Result: 2 passed.

```text
python3 scripts/check_jobs.py
```

Result: `check_jobs: HF Jobs specs ok`.

## Required Resolution

F3 can be launched only after all of the following are true:

1. A valid warm-start source checkpoint exists and is compatible with the
   current SO-100 training layout.
2. `lewm-train` applies `training.warmstart_from` before SO-100 training starts
   and records warm-start provenance in the run report. **Done locally.**
3. `jobs/train_so100_warmstart.yaml` is added and validated against the real
   source checkpoint path. **Spec added locally; final source path still
   pending.**
4. The safety leash is updated by a human to list the job under
   `jobs_human_approval_required`.
5. A human explicitly approves the paid HF Job launch.
