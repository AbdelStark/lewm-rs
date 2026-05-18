# Runtime Image Publication Attempt

**Updated:** 2026-05-18  
**Workflow:** `runtime-image.yml`  
**Run:** <https://github.com/AbdelStark/lewm-rs/actions/runs/26035809317>  
**Commit:** `97880d00508efb54cb5a4a1e4b200f59f93c4c47`  
**Image tag:** `f1-runtime-97880d0`  
**Status:** blocked

## Command

```bash
gh workflow run runtime-image.yml --ref main -f image_tag=f1-runtime-97880d0
gh run watch 26035809317 --exit-status
```

## Result

The workflow reached GHCR login and completed the Docker build, but failed while
pushing `ghcr.io/abdelstark/lewm-rs:f1-runtime-97880d0`:

```text
denied: permission_denied: write_package
```

No Hugging Face Job was launched and no Hub artifact was uploaded.

## Required Resolution

F11 remains open. Grant repository `AbdelStark/lewm-rs` **Write** access to the
`ghcr.io/abdelstark/lewm-rs` package under:

<https://github.com/users/abdelstark/packages/container/lewm-rs/settings>

After that user action, rerun:

```bash
gh workflow run runtime-image.yml --ref main -f image_tag=f1-runtime-97880d0
gh run watch
python3 scripts/verify_runtime_image.py --image-tag f1-runtime-97880d0
```

F1 must not launch `jobs/train_pusht.yaml` until the verifier passes for a
concrete non-`latest` runtime tag.

## Source-Build Fallback

The repository also carries `jobs/train_pusht_source.yaml` as a no-GHCR fallback
for F1. It uses `rust:1.95.0-bookworm`, requires the caller to provide a full
40-character `LEWM_SOURCE_REVISION`, fetches exactly that git commit, builds
`lewm-train`, runs the same full Burn/Jepa PushT command, checks the safetensors
ONNX export contract before upload, and remains listed under
`jobs_human_approval_required`.

Dry-run preflight:

```bash
LEWM_SOURCE_REVISION="$(git rev-parse HEAD)" \
  python3 scripts/launch_hf_job.py jobs/train_pusht_source.yaml \
    --dry-run \
    --allow-approval-required
```

The rendered command is still a paid 12h A10G-large job and must not be launched
without explicit human approval. This fallback does not resolve F11; the final
release container still requires GHCR package write access.
