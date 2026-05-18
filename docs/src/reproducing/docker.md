# Docker and HF Jobs

## 1. The image

`ghcr.io/abdelstark/lewm-rs:latest`. Built from the checked-in
[`Dockerfile`](https://github.com/AbdelStark/lewm-rs/blob/main/Dockerfile).

Contents:

- `lewm-train`, `lewm-eval`, `lewm-infer` (release binaries).
- All configs under `/app/configs/`.
- The HF Jobs spec files under `/app/jobs/`.
- Python helpers in `/app/python/`: `convert_reference.py`,
  `decode_so100_to_h5.py`, `compute_so100_stats.py`,
  `export_onnx.py`, `upload_checkpoints.py`.
- The `hf` CLI, `zstd`, and `bash`.

The image is rebuilt on every push to `main` by
`.github/workflows/ci.yml` and pushed to GHCR.

## 2. Building locally

```sh
docker build -t lewm-rs:dev .
```

The Dockerfile uses a multi-stage build: a Rust + CUDA toolchain stage
for compilation, a slim stage for runtime.

## 3. Launching HF Jobs

HF Jobs spec files live under `jobs/`:

| File | Purpose |
|------|---------|
| `jobs/smoke_pusht.yaml` | 50-step smoke train on PushT, CPU. |
| `jobs/short_pusht.yaml` | 10-step "real" train path on PushT. |
| `jobs/train_pusht.yaml` | Approval-gated 50 k-step PushT train on A10G-large. |
| `jobs/train_so100.yaml` | Approval-gated 5 k-step SO-100 train on A10G-large. |
| `jobs/train_so100_warmstart.yaml` | Approval-gated 5 k-step SO-100 warm-start from PushT; requires a compatible `.mpk` source path. |

Launch with the helper:

```sh
scripts/launch_hf_job.py jobs/train_pusht.yaml --allow-approval-required
```

The helper:

1. Validates the YAML against the schema in
   `scripts/check_jobs.py`.
2. Resolves `${HF_TOKEN}` and other env-var placeholders.
3. Calls `hf jobs run` to schedule the job.
4. Returns the job ID for monitoring.

## 4. Job lifecycle

- The job pulls the image, mounts a scratch disk, exports `HF_TOKEN`,
  and runs the command from the spec.
- Logs (stdout) are tailed to the HF Jobs UI live.
- Checkpoints are pushed to the Hub repo via the trainer's UPLOAD
  state.
- On crash, re-launching the same training job preserves the trainer's
  `--resume-if-present` behavior when the job spec includes it.

## 5. Cost

A10G-large is \$1.50 / hour (confirmed via
`python3 python/hf_pricing.py`). The cost ledger in
`reports/cost.md` is updated after every run; the cost-cap check
(`python3 python/cost_ledger.py check --cap-usd 200`) blocks new
launches that would exceed the project ceiling.

## 6. The job-config gate

`scripts/check_jobs.py` validates every job spec:

- Image is the expected GHCR runtime image, except source-building approval
  jobs that intentionally use `rust:1.95.0-bookworm`.
- Command uses only binaries / scripts present in the image.
- Hardware tier is on the approved list.
- Required env vars are declared.

Run it via:

```sh
python3 scripts/check_jobs.py
```

This is wired into `make check`.
