# Docker and HF Jobs

## 1. The image

`ghcr.io/abdelstark/lewm-rs:latest`. Built from the checked-in
[`Dockerfile`](https://github.com/AbdelStark/lewm-rs/blob/main/Dockerfile).
For release or production reruns, pin the image at launch time with
`scripts/launch_hf_job.py --image-tag <tag>` or `LEWM_IMAGE_TAG=<tag>`.

Contents:

- `lewm-train` (release binary) on `PATH`.
- All configs under `/workspace/configs/`.
- The HF Jobs spec files under `/workspace/jobs/`.
- Python helpers in `/workspace/python/`: `convert_reference.py`,
  `decode_so100_to_h5.py`, `compute_so100_stats.py`,
  `export_onnx.py`, `upload_checkpoints.py`.
- The `hf` CLI, `hdf5plugin`, `safetensors`, `zstd`, `tini`, and `bash`.

The release workflow builds and pushes `ghcr.io/abdelstark/lewm-rs:<tag>` and
`ghcr.io/abdelstark/lewm-rs:latest` when a `v*.*.*` tag is published. The
image-only `runtime-image` workflow publishes a concrete non-`latest` tag from
the selected git ref for pre-release paid jobs such as F1. The CI workflow
validates the code but does not publish the container.

```sh
image_tag="f1-runtime-$(git rev-parse --short HEAD)"
gh workflow run runtime-image.yml --ref main -f image_tag="${image_tag}"
python3 scripts/verify_runtime_image.py --image-tag "${image_tag}"
```

Run the workflow only after the selected ref points at the commit you intend to
launch; the verifier defaults to local `HEAD`.

When GHCR package permissions are unavailable, F1 can use
`jobs/train_pusht_source.yaml` as an approval-gated fallback. That job builds
from `LEWM_SOURCE_REVISION` inside HF Jobs and does not push or pull GHCR, but
it is still a paid production run and does not replace the final release
container requirement.

## 2. Building locally

```sh
docker build -t lewm-rs:dev .
```

The Dockerfile uses a multi-stage build: a Rust 1.95 builder stage for
`lewm-train`, then a slim Debian runtime stage for HF Jobs.

## 3. Launching HF Jobs

HF Jobs spec files live under `jobs/`:

| File | Purpose |
|------|---------|
| `jobs/smoke_pusht.yaml` | 50-step smoke train on PushT, CPU. |
| `jobs/short_pusht.yaml` | 10-step "real" train path on PushT. |
| `jobs/train_pusht.yaml` | Approval-gated 50 k-step PushT train on A10G-large. |
| `jobs/train_pusht_source.yaml` | Approval-gated 50 k-step PushT fallback that builds from `LEWM_SOURCE_REVISION` instead of GHCR. |
| `jobs/train_so100.yaml` | Approval-gated 5 k-step SO-100 train on A10G-large. |
| `jobs/train_so100_warmstart.yaml` | Approval-gated 5 k-step SO-100 warm-start from PushT; requires a compatible `.mpk` source path. |

Launch with the helper:

```sh
python3 scripts/verify_runtime_image.py \
  --image-tag REPLACE_WITH_RUNTIME_IMAGE_TAG

scripts/launch_hf_job.py jobs/train_pusht.yaml \
  --allow-approval-required \
  --image-tag REPLACE_WITH_RUNTIME_IMAGE_TAG
```

For a release-pinned job:

```sh
scripts/launch_hf_job.py jobs/train_pusht.yaml --allow-approval-required --image-tag vX.Y.Z
```

The helper:

1. Validates the YAML against the schema in
   `scripts/check_jobs.py`.
2. Verifies approval-required PushT image tags with
   `scripts/verify_runtime_image.py`.
3. Resolves `${HF_TOKEN}` and other env-var placeholders.
4. Rewrites the image tag when `--image-tag` or `LEWM_IMAGE_TAG` is set.
5. Calls `hf jobs run` to schedule the job.
6. Returns the job ID for monitoring.

## 4. Job lifecycle

- The job pulls the image, mounts a scratch disk, exports `HF_TOKEN`,
  and runs the command from the spec.
- Logs (stdout) are tailed to the HF Jobs UI live.
- Checkpoints are pushed to the Hub repo by `python/upload_checkpoints.py`
  after the job command's local validation steps pass.
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
