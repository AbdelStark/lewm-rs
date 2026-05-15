# Release Checklist (Issue #197)

**Updated:** 2026-05-15
**Target release:** v0.4.0 (post-training-completion)

This document tracks all items needed before tagging a public release.

## Security

| Item | Status | Notes |
|------|--------|-------|
| `.env` excluded from git | ✅ Done | Listed in `.gitignore` as `.env` and `.env.*` |
| No hardcoded secrets in source | ✅ Verified | grep scan + gitleaks config in place |
| `HF_TOKEN` rotated before release | ⚠️ **USER ACTION REQUIRED** | Rotate at huggingface.co/settings/tokens |
| GHCR token (GITHUB_TOKEN) | ⚠️ **USER ACTION REQUIRED** | Add `AbdelStark/lewm-rs` repo to package settings at github.com/users/abdelstark/packages/container/lewm-rs/settings |
| `gitleaks` scan clean | ⚠️ Not installed | Install gitleaks and run `python3 scripts/check_secrets.py` before release |
| `.ml-intern/cli_agent_config.json` | ✅ Safe | Contains only billing limits and access control rules; no secrets |

## Code quality

| Item | Status | Notes |
|------|--------|-------|
| `make check` passes | ✅ Green | CI spec checks pass; local gate passes |
| All 10 parity tests pass | ✅ Verified | L∞ < 1e-4 across all components |
| `CARGO_INCREMENTAL=0 make check` passes | ✅ | Documented in ROADMAP |
| No `clippy` warnings | ✅ | CI enforces `--deny warnings` |
| Rustdoc builds cleanly | ✅ | `make docs` passes |

## Artifacts

| Item | Status | Notes |
|------|--------|-------|
| PushT training artifacts on Hub | ⏳ Pending | Job `6a06f0c43308d79117b90276` still running |
| PushT CEM planning eval | ⏳ Pending | Needs trained checkpoint |
| PushT model card with metrics | ⏳ Pending | Needs eval results |
| SO-100 training artifacts on Hub | ✅ Done | `abdelstark/lewm-rs-so100/train/so100-full-20260515T122820Z/` |
| SO-100 model card | ✅ Done | Uploaded via `scripts/upload_model_cards.py` |
| ONNX artifacts (onnxruntime) | ✅ Done | `encoder.onnx` + `.data` + `predictor.onnx` + `.data` at root of `abdelstark/lewm-rs-pusht` |
| ONNX artifacts (Tract-compat) | ✅ Done | `tract-compat/encoder.onnx` + `predictor.onnx` at `abdelstark/lewm-rs-pusht` |
| Demo Space live | ✅ Done | `abdelstark/lewm-rs-demo` |
| Parity dumps on Hub | ✅ Done | `AbdelStark/lewm-rs-parity-dumps` |
| SO-100 dataset on Hub | ✅ Done | `abdelstark/so100-pickplace-lewm-ready` |

## Documentation

| Item | Status | Notes |
|------|--------|-------|
| README updated | ✅ Done | Reflects current state |
| CHANGELOG updated | ✅ Done | Unreleased section current |
| ROADMAP updated | ✅ Done | Issues #195 marked Done |
| Paper draft | ✅ Done | `paper/lewm-rs.md` — TBD sections for PushT eval |
| SO-100 training report | ✅ Done | `reports/so100_training.md` |
| PushT training report | ⏳ Pending | Needs job completion |
| Model cards with eval metrics | ⏳ Pending | Needs eval runs |
| Cost ledger final | ⏳ Pending | Update when PushT job completes |

## Release mechanics

| Item | Status | Notes |
|------|--------|-------|
| `CHANGELOG.md` release section | ⏳ Pending | Move [Unreleased] → [0.4.0] when ready |
| Git tag `v0.4.0` | ⏳ Pending | After all required items complete |
| GitHub release draft | ⏳ Pending | Link to model cards, ONNX artifacts, demo |
| GHCR container image | ⚠️ Blocked | Needs user to grant repo GITHUB_TOKEN write access to package |
| `cargo publish` (if applicable) | ⏳ N/A | crates.io publish not planned for v0.4.0 |

## Billing guardrails

| Item | Status | Notes |
|------|--------|-------|
| `.ml-intern/cli_agent_config.json` hard cap | ✅ Set | $200 USD lifetime, $20/session |
| Soft cap | ✅ Set | $100 USD — triggers notification |
| Per-job default timeout | ✅ Set | 30 min default |
| A100/H100 hardware denied | ✅ Set | Only cpu/l4/a10g allowed |
| High-cost jobs require human approval | ✅ Set | `train_pusht.yaml`, `train_so100.yaml` |

## Required user actions before release

1. **Rotate HF_TOKEN**: Go to https://huggingface.co/settings/tokens, revoke
   the current token, create a new read+write token, update `.env` locally.
   Do NOT commit the new token.

2. **Fix GHCR permissions**: Visit
   https://github.com/users/abdelstark/packages/container/lewm-rs/settings
   and add repository `AbdelStark/lewm-rs` with Write role. This unblocks
   the `container` job in the release workflow.

3. **Run gitleaks scan**: `pip install gitleaks` then
   `python3 scripts/check_secrets.py` to get a clean report before tagging.

4. **Monitor HF billing dashboard**: Check actual billed amounts once PushT
   job completes and update `reports/cost.md` with real numbers.
