---
rfc: "0010"
title: "lewm-hub — Hugging Face Hub integration, model cards, cost ledger"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§3 deliverables D2/D3/D4", "§7 cost", "§13 provenance"]
depends_on: ["0001", "0005", "0009"]
related: ["0011", "0015", "0016"]
---

# RFC 0010 — `lewm-hub`: HF Hub integration, model cards, cost ledger

> **Status:** Accepted · **Version:** 1.0.0
>
> Every artifact this project produces lives on the Hugging Face Hub. This RFC pins the upload pipeline, the model-card template, the dataset-mirror manifests, and the cost-ledger automation. Idempotency, attribution, and provenance are the recurring themes.

---

## 1. Introduction

### 1.1 Motivation

Hub publication is the *act* that turns "a trained model in `out/`" into a public, reproducible artifact. It has to be:

- **Idempotent** — failed uploads recover cleanly.
- **Provenanced** — every artifact carries enough metadata to be reproduced.
- **Attributed** — upstream LeWM and dataset authors are credited per PRD §13.

### 1.2 Goals

1. Specify the `lewm-hub` Rust API.
2. Specify the model-card template and its required fields (rendered from training metadata).
3. Specify the dataset-mirror manifest format.
4. Specify the upload pipeline including the Python fallback for operations the Rust client cannot yet do.
5. Specify the cost-ledger automation.

### 1.3 Non-goals

- Authentication itself (covered by [RFC 0016 §3](0016-security-and-supply-chain.md)).
- The Demo Space contents (covered by [RFC 0007 §10](0007-tract-inference-and-onnx-export.md) and [RFC 0015 §6](0015-documentation-paper-and-demo.md)).

---

## 2. Conventions

- **Model repo** — an HF repo of type `model` (e.g., `AbdelStark/lewm-rs-pusht`).
- **Dataset repo** — type `dataset` (e.g., `AbdelStark/lewm-pusht-mirror`).
- **Space repo** — type `space` (e.g., `AbdelStark/lewm-rs-demo`).
- **Revision** — a commit SHA on the repo.

---

## 3. Crate layout

```
lewm-hub/
└── src/
    ├── lib.rs
    ├── client.rs                 # auth, repo CRUD via hf-hub
    ├── upload.rs                 # idempotent upload with retry
    ├── download.rs               # downloads (datasets, references)
    ├── model_card.rs             # YAML frontmatter + Markdown body
    ├── dataset_card.rs
    ├── manifest.rs               # dataset mirror manifest schema
    ├── cost_ledger.rs            # `reports/cost.md` updater
    ├── python_bridge.rs          # PyO3 sidecar for ops unsupported by hf-hub-rs
    └── errors.rs
```

**Note:** `python_bridge.rs` is gated behind the `python-bridge` feature. It does **not** ship in default builds.

---

## 4. Hub client

### 4.1 Auth

```rust
pub struct HubClient {
    api:    hf_hub::api::tokio::Api,
    user:   String,
    namespace: String,        // typically "AbdelStark"
    token:  SecretString,
}

impl HubClient {
    pub fn from_env() -> Result<Self, HubError> {
        let token = std::env::var("HF_TOKEN")
            .map_err(|_| HubError::TokenMissing)?;
        let api = hf_hub::api::tokio::ApiBuilder::new()
            .with_token(Some(token.clone()))
            .build()?;
        let user = whoami_via_api(&api, &token)?;
        Ok(Self { api, user: user.clone(), namespace: "AbdelStark".to_string(), token: SecretString::new(token) })
    }
}
```

**RFC0010-001 [MUST]** — `from_env` reads the token from `HF_TOKEN`. **Never** from a file in-repo. Never logged.

**RFC0010-002 [MUST]** — `HubClient::whoami` is called eagerly to fail-fast on bad tokens.

### 4.2 Repo CRUD

```rust
impl HubClient {
    pub async fn ensure_repo(&self, name: &str, kind: RepoKind, private: bool)
        -> Result<RepoHandle, HubError>;

    pub async fn upload_file(&self, repo: &RepoHandle, local: &Path, remote: &str, commit_message: &str)
        -> Result<UploadResult, HubError>;

    pub async fn upload_folder(&self, repo: &RepoHandle, local_dir: &Path, remote_prefix: &str, commit_message: &str)
        -> Result<UploadResult, HubError>;

    pub async fn delete_file(&self, repo: &RepoHandle, remote: &str, commit_message: &str)
        -> Result<(), HubError>;
}
```

**RFC0010-003 [MUST]** — `ensure_repo` is idempotent: if the repo exists, returns its handle; otherwise creates it. Race-safe via HTTP 409 retry.

**RFC0010-004 [MUST]** — All uploads carry a **commit message** with the run id and a one-line action description (e.g., `"lewm-rs-pusht: upload step_0014400.mpk for run 20260512-143002-9f3a-abcd"`).

---

## 5. Upload pipeline

### 5.1 Idempotency

```
upload_artifact(local_path, remote_path):
  local_sha256 = sha256(local_path)
  try:
    remote_meta = hf_get_file_meta(repo, remote_path)
    if remote_meta.sha256 == local_sha256:
      log "already up to date"; return Skip
  except FileNotFound:
    pass
  upload_file(...)
  return Uploaded
```

**RFC0010-005 [MUST]** — File-level idempotency is via SHA-256 comparison. The remote SHA is read from the HF API metadata; if the API does not expose SHA for the file kind, we use the `lfs.sha256` field for LFS objects.

**RFC0010-006 [MUST]** — Folder uploads are idempotent at the file level — i.e., a partial folder upload that crashed halfway resumes by re-checking each file's SHA on retry.

### 5.2 Retry policy

```
- HTTP 5xx        : exponential backoff, 5 retries (1s, 2s, 4s, 8s, 16s)
- HTTP 429        : honor Retry-After header
- Network error   : 3 retries with 1s gap
- HTTP 4xx (non-429) : fail-fast
```

**RFC0010-007 [MUST]** — Retry envelope is in `upload::retry::with_backoff(closure)`. All public upload functions use it.

### 5.3 Order of operations

For a training run's UPLOAD state:

1. Generate model card (§7).
2. Generate cost-ledger update (§9).
3. Upload checkpoints `step_*.mpk` and `step_*.safetensors` (last 10).
4. Upload sidecars `step_*.json` (all).
5. Upload `parity_*.json`.
6. Upload `events.out.tfevents.*` (Tensorboard).
7. Upload `runs/<run_id>/metrics.jsonl` (Trackio local format).
8. Upload `reports/<training-report>.md`.
9. Upload `README.md` (rendered model card).
10. (Sidecar) `python/upload_checkpoints.py --trackio` uploads the Trackio dir to the Trackio Space.

All of the above run as a single Rust function with structured tracing.

**RFC0010-008 [MUST]** — On any failure, the function logs the partial progress and exits non-zero. A re-run resumes from the SHA-based dedupe; no manual cleanup is required.

---

## 6. Cost ledger

### 6.1 File location

`reports/cost.md`, with a Markdown table per row:

```markdown
# `lewm-rs` cost ledger

> Updated automatically by `lewm-hub::cost_ledger::append_entry` at every job termination.
> Manual entries are forbidden; use `cost_ledger::backfill --from <job_url>` to import.

| Date (UTC)          | Phase | Job ID            | Hardware     | Wall   | Cost (USD) | Cumulative (USD) | Notes |
|---------------------|-------|-------------------|--------------|--------|-----------:|----------------:|-------|
| 2026-05-12 14:30:02 | P1    | hfjob-2026-05-12-001 | l4 (24GB)    | 0:28:13 | 0.38 | 0.38 | smoke pusht |
```

### 6.2 Update protocol

```rust
pub fn append_entry(entry: CostEntry, ledger_path: &Path) -> Result<(), HubError> {
    let table = read_ledger(ledger_path)?;
    table.append(entry);
    write_ledger(ledger_path, &table)?;
    Ok(())
}
```

**RFC0010-009 [MUST]** — Cumulative sum is recomputed on every append from scratch; integrity is verified by a CI test that re-parses the ledger and checks the cumulative column.

**RFC0010-010 [MUST]** — A ledger row at any time **MUST NOT** show a cumulative greater than NFR-060's hard cap of 200 USD. CI fails if it does.

### 6.3 Source of cost data

For each HF Jobs run, the launcher (see [RFC 0011 §3](0011-ci-cd-and-release-engineering.md)) records:

```
job_id, hardware_flavor, started_at, ended_at, exit_code
```

The cost is computed as:

```
hours = (ended_at - started_at).as_seconds() / 3600.0
cost  = hours * HF_HARDWARE_PRICE_USD_PER_HOUR[hardware_flavor]
```

Hardware prices are stored in `python/hf_pricing.py` and refreshed manually when HF changes pricing (an ADR is filed). Currently:

```python
HF_HARDWARE_PRICE_USD_PER_HOUR = {
    "cpu-basic":   0.00,   # free
    "cpu-xl":      1.00,
    "l4":          0.80,
    "a10g-small":  1.00,
    "a10g-large":  1.50,
    "l40s":        1.80,
    "a100-large":  2.50,
    "h100":        8.00,
}
```

**RFC0010-011 [MUST]** — Cost calculation is **conservative**: round up to the nearest minute (HF's per-minute billing actually rounds, but we conservatively overstate).

### 6.4 Backfill

```
$ python python/cost_ledger.py backfill --since 2026-05-01
```

Reads `hf jobs list --org AbdelStark --since 2026-05-01` and appends any missing rows.

---

## 7. Model card

### 7.1 Template

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
  - {{dataset_tag}}                       # "pusht" or "so100-pickplace"
datasets:
  - {{primary_dataset}}                    # "quentinll/lewm-pusht" or "lerobot/svla_so100_pickplace"
metrics:
  - planning_success_rate                  # for pusht
  - latent_rollout_mse                     # for so100
  - spearman_rank_correlation              # for so100
base_model:
  - quentinll/lewm-pusht                    # if warm-started; else null
language:
  - en
pipeline_tag: robotics
model-index:
  - name: {{repo_name}}
    results:
      - task:
          type: world-model-planning
          name: PushT planning
        dataset:
          name: lewm-pusht
          type: quentinll/lewm-pusht
        metrics:
          - type: planning_success_rate
            value: {{success_rate}}
            name: success rate
          - type: latency_per_plan_step_ms
            value: {{latency_ms}}
            name: CPU plan latency
---

# {{repo_name}}

Pure-Rust reproduction of LeWorldModel ({{dataset_display}}). Trained with Burn
{{burn_version}} on a single Nvidia {{hardware}} GPU on Hugging Face Jobs.

## Result

- **Headline metric**: {{headline_metric}}: {{headline_value}}
- **Parity vs reference** (epoch 10 vs `quentinll/lewm-pusht`):
    - encoder CLS L∞ : {{parity_encoder}}
    - predictor L∞   : {{parity_predictor}}
- **CPU inference** (Tract, laptop): {{tract_laptop_ms}} ms per planning cost computation
- **Total training cost**: {{cost_usd}} USD

## How to use

For Rust inference on CPU:

```bash
cargo install --git https://github.com/AbdelStark/lewm-rs lewm-infer
hf download {{repo_name}} --local-dir ckpt
lewm-infer plan --checkpoint-dir ckpt --start start.png --goal goal.png
```

For Python loading via Safetensors mirror:

```python
from safetensors.torch import load_file
weights = load_file("step_0014400.safetensors")
```

## Training details

See training report: `{{report_url}}`.

## Citation

```bibtex
{{citation_lewm}}
{{citation_lewm_rs}}
```

## Provenance

| Field | Value |
|-------|-------|
| git SHA | {{git_sha}} |
| Burn version | {{burn_version}} |
| Rust toolchain | {{rust_version}} |
| Config hash | {{config_hash}} |
| Run id | {{run_id}} |
| Hardware | {{hardware}} |
| Wall time | {{wall_time}} |
| Cost | {{cost_usd}} USD |

## License

Apache-2.0. See [LICENSE](https://github.com/AbdelStark/lewm-rs/blob/main/LICENSE).
```

### 7.2 Generation

`model_card::render(metadata)` returns the rendered string. `metadata` is the merged content of `step_{N}.json`, the eval report, and the cost-ledger row for the run.

**RFC0010-012 [MUST]** — Every placeholder in the template **MUST** be filled. Missing fields are a render error.

**RFC0010-013 [MUST]** — The rendered card is committed to the model repo as `README.md`. The Hub renders it as the repo's landing page.

### 7.3 Dataset card

Dataset mirrors carry a card declaring upstream provenance and the transformation applied:

```yaml
---
license: cc-by-4.0
task_categories:
  - robotics
language:
  - en
size_categories:
  - 1M<n<10M
configs:
  - config_name: default
    data_files:
      - shards/pusht_000.h5
      - shards/pusht_001.h5
      - ...
source_datasets:
  - quentinll/lewm-pusht
---

# AbdelStark/lewm-pusht-mirror

Mirror of `quentinll/lewm-pusht` for the `lewm-rs` project. Re-uploaded with a
[provenance manifest](manifest.json) for byte-level reproducibility.

The mirror is byte-identical to the upstream. No transformations applied.
```

For `so100-pickplace-lewm-ready`, the description states the transformation (MP4 → 224×224 HDF5 at 10 Hz).

### 7.4 Manifest

`manifest.json` in each dataset repo:

```json
{
  "schema_version": "1.0",
  "source": {
    "repo": "quentinll/lewm-pusht",
    "revision": "<sha>",
    "fetched_at": "2026-05-12T14:30:02Z"
  },
  "files": [
    {
      "path": "shards/pusht_000.h5",
      "sha256": "...",
      "size_bytes": 12345678
    },
    ...
  ],
  "transformations": [
    {
      "kind": "byte_identity_mirror"
    }
  ]
}
```

For SO-100:

```json
{
  "transformations": [
    {
      "kind": "video_resample",
      "from_fps": 30,
      "to_fps": 10,
      "from_size": [480, 640],
      "to_size": [224, 224],
      "interp": "bilinear",
      "tool": "ffmpeg",
      "tool_version": "6.0"
    },
    {
      "kind": "container_change",
      "from": "mp4+parquet",
      "to": "hdf5"
    }
  ]
}
```

**RFC0010-014 [MUST]** — `manifest.json` SHA-256 is **the** integrity check. The cost ledger and eval report reference the SHA, not the revision SHA of the HF repo.

---

## 8. Python bridge

A small PyO3-gated module wraps `huggingface_hub` Python operations not covered by `hf-hub-rs`:

```python
# python/upload_checkpoints.py
import argparse, os, sys
from huggingface_hub import HfApi

def upload(src, dst, repo_type, commit_message):
    api = HfApi()
    api.upload_folder(folder_path=src, repo_id=dst, repo_type=repo_type, commit_message=commit_message)

if __name__ == "__main__":
    ...
```

**RFC0010-015 [MUST]** — The Python sidecar is invoked via subprocess from Rust (`python -m python.upload_checkpoints ...`), **not** via PyO3 in the default build. Reason: avoids requiring Python at the link site of the trainer binary.

---

## 9. Testing strategy

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0010-CLIENT-001 | `client_from_env_fails_without_token` | unit | RFC0010-001 |
| TST-0010-UPLOAD-001 | `upload_idempotent_on_same_file` | integration | RFC0010-005 |
| TST-0010-UPLOAD-002 | `upload_retry_on_5xx` | integration | RFC0010-007 |
| TST-0010-IDEMP-001 | `folder_upload_resume_after_crash` | integration | RFC0010-006 |
| TST-0010-CARD-001 | `model_card_renders_all_placeholders` | unit | RFC0010-012 |
| TST-0010-CARD-002 | `model_card_yaml_frontmatter_valid` | unit | RFC0010-013 |
| TST-0010-MANIFEST-001 | `manifest_sha256_matches_uploaded_files` | integration | RFC0010-014 |
| TST-0010-COST-001 | `cost_ledger_cumulative_correct` | unit | RFC0010-009 |
| TST-0010-COST-002 | `cost_ledger_under_200_usd` | unit (CI gate) | RFC0010-010 |
| TST-0010-ATTR-001 | `model_card_has_citation_block` | unit | NFR-041 |
| TST-0010-ART-PUSHT-001 | `pusht_repo_has_checkpoint_n4` | integration (release) | acceptance |
| TST-0010-ART-PUSHT-002 | `pusht_repo_has_safetensors_n4` | integration | acceptance |
| TST-0010-ART-PUSHT-003 | `pusht_repo_has_card_with_metrics` | integration | acceptance |
| TST-0010-ART-PUSHT-004 | `pusht_repo_has_training_report` | integration | acceptance |
| TST-0010-ART-SO100-001..004 | analogous SO-100 | integration | acceptance |

Fixtures:

- A mock HF API using `wiremock`-rs for unit tests.
- A real (small) staging repo `AbdelStark/lewm-rs-test-staging` for integration tests under a CI-only secret token.

---

## 10. Operational considerations

### 10.1 Observability

`upload/...` metrics emitted:

```
upload/bytes_total
upload/file_count
upload/skipped_count
upload/retry_count
upload/wall_seconds
```

### 10.2 Runbook

- **"401 Unauthorized."** — token invalid or scope insufficient. Regenerate via HF settings. Update GitHub secret.
- **"LFS quota exceeded."** — older runs need pruning; use `python/prune_old_runs.py --keep-last 5`.
- **"Folder upload stalled."** — re-run; idempotent.

### 10.3 Capacity

Per training run: ~ 2 GB on HF (model + safetensors + reports). 10 runs total ≈ 20 GB.

---

## 11. Performance considerations

Network-bound; we do not optimize.

---

## 12. Security considerations

- HF token is sole credential; see RFC 0016 §3.
- Public repos by default; private repos for parity dumps only.

---

## 13. Alternatives considered

- **A1 — Full Python upload pipeline.** Considered. Rejected: idempotency logic in Rust gives stronger guarantees; Python only for ops Rust can't yet do.
- **A2 — Direct git LFS push.** Possible but heavyweight; the HF API is the canonical path.

---

## 14. Acceptance criteria

- [ ] All TST-0010-* pass on CI.
- [ ] Staging repo round-trip works end-to-end.
- [ ] `cost.md` is updated on every job completion.
- [ ] Model card on the production repos has all PRD §13 attribution.

---

## 15. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | HF API rate limits | M | M | Retry with backoff; lower concurrency |
| R-2 | LFS quota | L | M | Prune script |
| R-3 | Card template drift vs HF schema | L | L | Validate frontmatter against HF schema in CI |
| R-4 | Cost ledger row miss | L | M | Backfill tool; CI re-checks integrity |

---

## 16. Open questions

OQ-2010-1 — Should we mirror to a Linear/Notion dashboard? Out of scope.

---

## 17. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0010.*
