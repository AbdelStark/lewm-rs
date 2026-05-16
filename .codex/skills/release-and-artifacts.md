---
name: release-and-artifacts
description: Hugging Face Hub artifact discipline — checkpoints, model cards, ONNX exports, demo Space. Activate before any upload to `abdelstark/lewm-rs-*`, when running `python/upload_checkpoints.py` / `python/upload_model_cards.py`, when modifying model cards in `python/model_cards/`, when editing `reports/release_checklist.md`, or when planning a release tag. Public artifact paths are policy-gated: smoke/ subpaths are free to overwrite; non-smoke paths require explicit approval.
prerequisites: `HF_TOKEN` for actual uploads (use `HF_TOKEN=dummy … --dry-run` for plumbing checks)
---

# Releases & Hub Artifacts

<purpose>
The project's external deliverables live on the Hub: `abdelstark/lewm-rs-pusht` (PushT checkpoints + ONNX), `abdelstark/lewm-rs-so100` (SO-100 checkpoints), and `abdelstark/lewm-rs-demo` (Gradio Space). This skill encodes how to write artifacts safely, where they go, and how to validate them before/after upload.
</purpose>

<context>
- **Repositories**:
  - `abdelstark/lewm-rs-pusht` — PushT checkpoints (Safetensors + `.mpk`), ONNX (opset 18, `tract-compat/` opset 17), parity JSON, losses JSONL, run reports, model cards. Examples: `train/pusht-full-lewm-…/`, `smoke/local-…/`.
  - `abdelstark/lewm-rs-so100` — SO-100 equivalent. Example: `train/so100-full-20260515T122820Z/`.
  - `abdelstark/lewm-rs-demo` — HF Space (Gradio + onnxruntime CEM planning). `sdk_version` pinned in the Space settings (currently 5.33.0). Auto-detects `action_dim` from predictor input shape.
  - `abdelstark/lewm-rs-parity-dumps` — per-layer activation Safetensors used by CI parity job.
  - `abdelstark/so100-pickplace-lewm-ready` — processed SO-100 dataset.
- **Path policy** (per `.ml-intern/cli_agent_config.json` and project convention):
  - `smoke/**` and `**/smoke-…` subpaths: agent may write/overwrite.
  - `train/**` and other non-smoke paths: **gated** — public artifacts; require human approval AND a fresh timestamped path.
- **Helpers**:
  - `python/upload_checkpoints.py --src DIR --dst REPO --path-prefix PFX [--dry-run]`
  - `python/upload_model_cards.py` (and `scripts/upload_model_cards.py`)
  - `scripts/check_hub_artifacts.py` — release-side validator
  - `scripts/check_release_inventory.sh` — release-manifest cross-check
- **Artifact contract** (per RFC 0010 / RFC 0011): run-report `.json`, `losses.jsonl`, checkpoint sidecar JSON, `.mpk` (Burn record), `.safetensors`, parity JSON, model card (`README.md`) optional ONNX (`.onnx` + `.onnx.data`).
- **Reproducible builds**: `scripts/build_reproducible_release.sh` produces a deterministic artifact bundle; `scripts/verify_reproducible_release.sh` checks the same on a clean tree.
- **SBOM**: `scripts/sbom.py` generates SBOM input used at release tagging time.
</context>

<procedure>
**Pre-upload (every time):**

1. Identify destination repo and path-prefix.
2. Classify the prefix:
   - Starts with `smoke/` → OK (still: dry-run first).
   - Anything else → STOP. Ask the human for approval AND propose a timestamped path (`train/<run>-$(date -u +%Y%m%dT%H%M%SZ)`).
3. Always **dry-run first**:
   ```
   HF_TOKEN=dummy python3 python/upload_checkpoints.py \
     --src /tmp/out --dst abdelstark/lewm-rs-pusht \
     --path-prefix smoke/local-test --dry-run
   ```
4. Inspect the dry-run file list. Confirm `.mpk`, `.safetensors`, `run_report.json`, `losses.jsonl`, parity JSON, and (if applicable) ONNX are present.

**Upload:**

5. Run without `--dry-run` (real `HF_TOKEN`). State the destination URL in chat.
6. Verify on the Hub side: file count, sizes, and (for parity-relevant uploads) re-run `scripts/check_hub_artifacts.py`.

**Model cards:**

7. Cards live in `python/model_cards/`. Edit there, then upload via `python/upload_model_cards.py` or `scripts/upload_model_cards.py`. Both target the corresponding model repo's `README.md`.
8. Model cards must reflect actual training results — no aspirational metrics. Match `reports/*_training.md`.

**Release tag (rare):**

9. Update `reports/release_checklist.md` with verification evidence.
10. Run `make accept` — it gates on hub artifacts and release inventory.
11. Run `scripts/build_reproducible_release.sh` and confirm output is byte-stable via `verify_reproducible_release.sh`.
12. Tag via the maintainer-driven path; do not push tags from the agent without explicit approval.

**Demo Space:**

13. Touching `abdelstark/lewm-rs-demo` is gated; coordinate with the maintainer. Pin `sdk_version` and confirm `.onnx.data` external-weight files are uploaded alongside `.onnx`.
</procedure>

<patterns>
<do>
— Timestamp every non-smoke path: `train/<run>-$(date -u +%Y%m%dT%H%M%SZ)`. Don't overwrite published artifacts.
— Cross-link Hub paths in `reports/*.md` and the corresponding model card.
— Mirror artifact layout between PushT and SO-100 — symmetry helps `scripts/check_hub_artifacts.py`.
— Upload model cards as part of the same logical change as the checkpoints they describe.
</do>
<dont>
— Don't overwrite `train/<existing-timestamp>/`. Treat published paths as immutable.
— Don't upload partial run outputs. The artifact contract expects a complete set; partial uploads break the demo Space and parity gate.
— Don't bump `sdk_version` on the demo Space without testing locally first — past breakages traced to silent gradio upgrades.
— Don't push `onnx` without its `onnx.data` companion when external weights apply; consumers fail at runtime.
</dont>
</patterns>

<examples>
Allowed smoke upload (illustrative):
```
HF_TOKEN=dummy python3 python/upload_checkpoints.py \
  --src /tmp/lewm-smoke \
  --dst abdelstark/lewm-rs-pusht \
  --path-prefix smoke/local-$(date -u +%Y%m%dT%H%M%SZ) \
  --dry-run
# inspect file list → proceed without --dry-run
```

Gated non-smoke upload:
```
DEST=train/pusht-full-lewm-$(date -u +%Y%m%dT%H%M%SZ)
echo "Proposed dest: abdelstark/lewm-rs-pusht/$DEST"
# AskUserQuestion("Approve upload to $DEST? Includes <files> totaling <size>.")
# Only proceed if user confirms.
```
</examples>

<troubleshooting>
| Symptom                                                          | Cause                                                  | Fix                                                                              |
|------------------------------------------------------------------|--------------------------------------------------------|----------------------------------------------------------------------------------|
| `check_hub_artifacts.py` fails: missing parity JSON               | Run did not emit parity probe output                   | Confirm trainer ran with parity probe enabled; re-run the bounded train          |
| Demo Space 500s after upload                                      | New `.onnx` without `.onnx.data` companion             | Re-upload BOTH files atomically                                                  |
| `upload_checkpoints.py --dry-run` lists 0 files                   | `--src` empty or wrong                                 | Confirm trainer wrote to the directory; check `output_dir`                       |
| Tract Space build fails: "opset 18 unsupported"                   | Wrong ONNX file used                                   | Use the `tract-compat/` (opset 17 fixed-batch) variant                           |
| `verify_reproducible_release.sh` reports byte diff                | Nondeterminism leaked into build                       | See `determinism-rng.md`; profile to find the offending step                     |
| Model card metrics don't match `reports/*_training.md`            | Card edited in isolation                               | Sync card to the report; never report numbers not present in a committed report  |
</troubleshooting>

<references>
- `python/upload_checkpoints.py` — primary uploader
- `python/upload_model_cards.py`, `scripts/upload_model_cards.py` — model-card upload
- `python/model_cards/` — card source
- `scripts/check_hub_artifacts.py`, `scripts/check_release_inventory.sh`
- `scripts/build_reproducible_release.sh`, `scripts/verify_reproducible_release.sh`
- `reports/release_checklist.md`, `reports/cost.md`
- `specs/rfcs/0010-huggingface-hub-integration.md`, `specs/rfcs/0011-ci-cd-and-release-engineering.md`
</references>
