# SO-100 Warm-Start Preflight

**Date:** 2026-05-18
**Issue:** F3 / #245
**Status:** Blocked before launch

## Objective

Launch the SO-100 warm-start ablation from
`jobs/train_so100_warmstart.yaml`, upload artifacts under
`abdelstark/lewm-rs-so100/train/so100-warmstart-*/`, and compute the
from-scratch vs. warm-start Spearman / loss delta.

## Preflight Evidence

The GitHub issue is still open and carries the `human-only` label. The issue
also states that launch requires human approval because it costs money.

The referenced job file is not present in the checkout:

```text
jobs/train_so100_warmstart.yaml: missing
```

The agent safety leash does not list `train_so100_warmstart.yaml` in either
`jobs_allowed` or `jobs_human_approval_required`. `scripts/launch_hf_job.py`
would reject the job name even if a YAML were added locally.

The config exists:

```text
configs/so100_warmstart.toml
training.warmstart_from = "/checkpoints/lewm-rs-pusht/step_0014400.mpk"
```

However, `training.warmstart_from` is currently parsed only in
`crates/lewm-train/src/config.rs`; it is not consumed by the training loop.
An HF job using this config would therefore not perform a real warm-start
without additional trainer integration.

The configured source checkpoint is also stale. No
`step_0014400.mpk` artifact exists in `abdelstark/lewm-rs-pusht`, and the
current 50k PushT checkpoint is the older 4-dimensional
`pusht-minimal-lewm` bounded-core artifact, not the current 192-dimensional
SO-100 bounded-core layout.

## Required Resolution

F3 can be launched only after all of the following are true:

1. A valid warm-start source checkpoint exists and is compatible with the
   current SO-100 training layout.
2. `lewm-train` applies `training.warmstart_from` before SO-100 training starts
   and records warm-start provenance in the run report.
3. `jobs/train_so100_warmstart.yaml` is added and validated.
4. The safety leash is updated by a human to list the job under
   `jobs_human_approval_required`.
5. A human explicitly approves the paid HF Job launch.
