# SO-100 Warm-Start Preflight

**Date:** 2026-05-18
**Issue:** F3 / #245
**Status:** Trainer wiring, fail-closed job spec, and safety-leash gating complete; launch still blocked

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

The agent safety leash now lists `train_so100_warmstart.yaml` under
`jobs_human_approval_required`, not `jobs_allowed`. `scripts/launch_hf_job.py`
therefore rejects the job unless the operator passes the explicit
approval-required flag and provides a source `.mpk` path:

```text
python3 scripts/launch_hf_job.py jobs/train_so100_warmstart.yaml --dry-run
launch_hf_job.py: train_so100_warmstart.yaml requires --allow-approval-required

python3 scripts/launch_hf_job.py jobs/train_so100_warmstart.yaml --dry-run --allow-approval-required
launch_hf_job.py: train_so100_warmstart.yaml requires LEWM_PUSHT_WARMSTART_MPK to name a compatible PushT .mpk source path
```

With both the explicit approval-required flag and a relative `.mpk` source
path, the launcher dry-run renders the `hf jobs run` command for `a10g-large`
and `6h` without submitting the job:

```text
LEWM_PUSHT_WARMSTART_MPK=train/pusht-bounded-module-lewm-source/step_0050000.mpk \
  python3 scripts/launch_hf_job.py jobs/train_so100_warmstart.yaml --dry-run --allow-approval-required
hf jobs run --namespace abdelstark --flavor a10g-large --timeout 6h ...
```

The job spec also fails closed inside the shell command unless
`LEWM_PUSHT_WARMSTART_MPK` points at a compatible PushT `.mpk` path in the
source model repo. The launcher now catches an empty, absolute,
parent-traversal, placeholder, globbed, or non-`.mpk` source path before even
rendering the HF command. This prevents accidentally launching against the
stale `configs/so100_warmstart.toml` default or a handoff placeholder.

The job now runs a local source-check before training:

```text
python3 scripts/check_warmstart_source.py \
  --path "$WARMSTART_LOCAL" \
  --config configs/pusht.toml
```

That verifier requires the current bounded PushT warm-start record contract:
`schema_version == "1.1.0"`, `kind ==
"lewm-rs-pusht-bounded-module-lewm-record"`, and the `41,856`-parameter layout
derived from `configs/pusht.toml`. This is intentionally the bounded-core
SO-100 trainer boundary. A full Burn/Jepa `NamedMpk` source from the F1 path is
not accepted by this job unless SO-100 warm-start is migrated to the full
Burn/Jepa trainer path with a separate contract update.

The current bounded PushT writer can produce that source contract locally. This
operator smoke uses the built-in PushT fixture, runs one bounded training step,
and checks the produced `.mpk` with the same launch verifier:

```text
scripts/pusht_warmstart_source_smoke.py \
  --report reports/pusht_warmstart_source_smoke.json

warm-start source ok: path=.../step_0000001.mpk step=1 params=41856
PushT warm-start source smoke ok: output_dir=...
```

The committed report records `schema_version = 1.1.0`, kind
`lewm-rs-pusht-bounded-module-lewm-record`, 41,856 model params, 41,856 AdamW
params, and the source-checker result. This proves the current writer/checker
contract locally; it does **not** replace the missing trained PushT source
artifact needed for the release warm-start ablation.

The currently published 50k PushT `.mpk` is rejected immediately:

```text
hf download abdelstark/lewm-rs-pusht \
  --include 'train/pusht-full-lewm-20260515T100908Z/step_0050000.mpk' \
  --local-dir /tmp/pusht-warmstart-source-check

python3 scripts/check_warmstart_source.py \
  --path /tmp/pusht-warmstart-source-check/train/pusht-full-lewm-20260515T100908Z/step_0050000.mpk \
  --config configs/pusht.toml

check_warmstart_source.py: .../step_0050000.mpk: schema_version must be '1.1.0', got '1.0.0'
```

The broader public Hub audit reaches the same conclusion for every current
PushT `.mpk` in `abdelstark/lewm-rs-pusht`:

```text
python3 scripts/audit_pusht_warmstart_sources.py \
  --report reports/pusht_warmstart_hub_audit.json

PushT warm-start Hub audit: candidates=6 compatible=0 status=blocked
```

`reports/pusht_warmstart_hub_audit.json` records all six public candidates and
their verifier failures. The current 50k `train/pusht-full-lewm-*` candidate is
a `lewm-rs-pusht-minimal-lewm-record` with 56 params, not the bounded-module
record with 41,856 params. Other public candidates are scalar-smoke,
action-probe, or tiny-JEPA records. None satisfy the `schema_version = 1.1.0`,
bounded record kind, and 41,856-param source contract required by
`scripts/check_warmstart_source.py`.

The config exists:

```text
configs/so100_warmstart.toml
training.warmstart_from = "/checkpoints/lewm-rs-pusht/step_0014400.mpk"
```

Before the 2026-05-18 release pass, `training.warmstart_from` was parsed in
`crates/lewm-train/src/config.rs` but not consumed by the training loop.

The trainer now applies `training.warmstart_from` for fresh SO-100 full-module
training starts. The transfer copies only `encoder.*`, `predictor.*`,
`projector.*`, and `pred_proj.*` from a PushT bounded-module record, preserves
the fresh SO-100 `action_encoder.*` parameters, resets AdamW state, and records
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

```text
uv run --project python pytest python/tests/test_check_warmstart_source.py
```

Result: 4 passed.

```text
uv run --project python --frozen pytest python/tests/test_launch_hf_job.py
```

Result: 23 passed, including the SO-100 warm-start approval-required and
source-path gates.

```text
python3 scripts/check_pusht_warmstart_source_smoke_report.py
```

Result: `PushT warm-start source smoke report ok: params=41856
mode=pusht-bounded-module-lewm`.

## Required Resolution

F3 can be launched only after all of the following are true:

1. A valid bounded-core warm-start source checkpoint exists and is compatible
   with the current SO-100 training layout, or the SO-100 warm-start path is
   migrated to full Burn/Jepa with an explicit contract update. **Local writer
   contract proven; release source artifact still pending.**
2. `lewm-train` applies `training.warmstart_from` before SO-100 training starts
   and records warm-start provenance in the run report. **Done locally.**
3. `jobs/train_so100_warmstart.yaml` is added and validated against the real
   source checkpoint path. **Spec added locally; final source path still
   pending.**
4. The safety leash lists the job under `jobs_human_approval_required`. **Done
   locally.**
5. A human explicitly approves the paid HF Job launch.
