# Roadmap and Completion Backlog

Updated: 2026-05-15 (v0.3.0)

Canonical GitHub tracker: [#189](https://github.com/AbdelStark/lewm-rs/issues/189)

The PRD and accepted RFCs remain the product contract. This document is the
live execution backlog: it records what is proven, what is not yet claimed, and
the next vertical slices needed to finish the project.

## Current Verified State

| Area | Status | Evidence |
|------|--------|----------|
| Specs and workspace | Implemented enough for current local gates | `CARGO_INCREMENTAL=0 make check` passed during this refresh |
| GHCR training image | Published | `ghcr.io/abdelstark/lewm-rs:latest@sha256:831f685a733a801620bbfa3f7ea649a4795ed731934bcb230896d3a47428d3e9` |
| HF Jobs short PushT run | Completed | `https://huggingface.co/jobs/abdelstark/6a05cf0ee48bea4538b9ccd6` |
| HF artifact upload | Completed for earlier minimal short run | `abdelstark/lewm-rs-pusht/train/pusht-minimal-lewm-short-20260514T133423Z/` |
| Full PushT training job | Running | `https://huggingface.co/jobs/abdelstark/6a06f0c43308d79117b90276`; 50k steps on a10g-large |
| SO-100 training job | Running | `https://huggingface.co/jobs/abdelstark/6a070293e48bea4538b9e1fb`; 10 epochs on a10g-large; rust:1.89.0-bookworm + HDF5 compat symlink |
| Demo Space | Created | `https://huggingface.co/spaces/abdelstark/lewm-rs-demo`; Gradio app with CEM planning via ONNX; loads model from Hub when available |
| SO-100 processed dataset | Uploaded | `abdelstark/so100-pickplace-lewm-ready`; 1.9 GB HDF5 + stats.safetensors; 6,559 timesteps, 50 episodes at 10 fps |
| SO-100 training support | Implemented | `lewm-train` trainer dispatches on `DatasetConfig::So100`; `run_so100_full_lewm_training`; 6-DOF action packing; commit `6add7fd` |
| ONNX export pipeline | Implemented (pending trained checkpoint) | `python/export_onnx.py` inverts param_name_map and exports encoder + predictor to ONNX opset 18 for Tract runner |
| PushT train command | Bounded full-module host path exists | `lewm-train --config configs/pusht.toml --device cpu --output-dir /tmp/lewm-train-pusht --max-steps 10 train` |
| PushT reference architecture | Locked | `tests/fixtures/reference_model.meta.json`; [#190](https://github.com/AbdelStark/lewm-rs/issues/190) |
| Burn ViT encoder | Implemented | `lewm_core::vit`; RFC 0002 shape coverage; PR [#201](https://github.com/AbdelStark/lewm-rs/pull/201) |
| Burn action embedder | Implemented | `lewm_core::embedder`; Conv1d-k1 smoothing preserved; PR [#202](https://github.com/AbdelStark/lewm-rs/pull/202) |
| Burn MLP heads (projector/pred_proj) | Implemented | `lewm_core::mlp`; feature-axis BatchNorm1d; PR [#203](https://github.com/AbdelStark/lewm-rs/pull/203) |
| Burn AdaLN-zero conditioner | Implemented | `lewm_core::ada_ln::AdaLNZero`; zero-init modulation heads; PR [#204](https://github.com/AbdelStark/lewm-rs/pull/204) |
| Burn autoregressive predictor | Implemented | `lewm_core::predictor::{ConditionalBlock,ArPredictor}`; PR [#205](https://github.com/AbdelStark/lewm-rs/pull/205) |
| SIGReg loss | Implemented | `lewm_core::losses::SigReg`; RFC 0003 constants; PR [#206](https://github.com/AbdelStark/lewm-rs/pull/206) |
| Prediction loss | Implemented | `lewm_core::losses::prediction_loss`; MSE kernel with gradient coverage; PR [#207](https://github.com/AbdelStark/lewm-rs/pull/207) |
| JEPA top-level wrapper | Implemented | `lewm_core::Jepa`; encode/predict/rollout/criterion/cost; PR [#208](https://github.com/AbdelStark/lewm-rs/pull/208) |
| Parity init shape audit | Implemented | `crates/lewm-core/tests/parity_init.rs`; parameter shape and count match reference metadata; PR [#209](https://github.com/AbdelStark/lewm-rs/pull/209) |
| PushT reference conversion scripts | Implemented | `python/param_name_map.py` (303 source tensors), `python/convert_reference.py` (audit + convert commands), `python/verify_conversion.py`; PRs [#210](https://github.com/AbdelStark/lewm-rs/pull/210)–[#212](https://github.com/AbdelStark/lewm-rs/pull/212) |
| Core Safetensors export | Implemented | `lewm_core::export::to_safetensors` writes deterministic `Jepa` parameter mirrors with BatchNorm state; PR [#213](https://github.com/AbdelStark/lewm-rs/pull/213) |
| Python dump subcommand | Implemented | `python/convert_reference.py dump` runs locked PyTorch reference model on parity fixture and captures all per-layer activations as Safetensors; PR [#214](https://github.com/AbdelStark/lewm-rs/pull/214) |
| Rust parity test suite | Implemented | 10 parity tests (encoder, action_encoder, predictor, pred_proj, sigreg) gated behind `parity-fixtures` + `LEWM_PARITY_DUMPS`/`LEWM_REFERENCE_SAFETENSORS` env vars; skip gracefully without dumps; PR [#215](https://github.com/AbdelStark/lewm-rs/pull/215) |
| CI parity workflow | Implemented | `parity` job caches dumps keyed on fixture hash, downloads from `AbdelStark/lewm-rs-parity-dumps` when `HF_TOKEN` available, runs full numerical tests or falls back to shape-only; PR [#216](https://github.com/AbdelStark/lewm-rs/pull/216) |
| Numerical parity correctness | Verified | All 10 parity tests pass (L∞ &lt; 1e-4 encoder/action_encoder/predictor/pred_proj, \|Δ\| &lt; 1e-3 sigreg); LayerNorm eps=1e-12 and exact-erf GELU fixes; dumps uploaded to `AbdelStark/lewm-rs-parity-dumps`; PR [#217](https://github.com/AbdelStark/lewm-rs/pull/217) |
| Artifact contract | Implemented for smoke and bounded PushT train | run report, losses JSONL, checkpoint sidecar, `.mpk`, `.safetensors`, parity JSON |
| Optional observability | Implemented as optional infra | `infra/otel/`; CI and smoke runs do not require OTLP |
| SO-100 full training | Completed | v11a job `6a070e02e48bea4538b9e2a5` completed (864s, 5000 steps, A10G-large); artifacts at `abdelstark/lewm-rs-so100/train/so100-full-20260515T122820Z/` (safetensors, mpk, losses, report, parity JSON) |
| Inference/export | Partially implemented | Tract runner/export scaffolds and tests exist; real trained-checkpoint benchmark and Space validation are pending |

## Non-Claims

- The Burn ViT parity stack is now numerically validated: all 10 activation-level parity tests pass against the locked reference PyTorch checkpoint; dumps are live in `AbdelStark/lewm-rs-parity-dumps` and CI downloads them automatically when `HF_TOKEN` is set.
- `pusht-full-module-lewm` is a config-shaped host training path, not the final
  Burn ViT parity stack. It is a narrow real training path for validating data,
  module boundaries, training, checkpoint, resume, upload, and job mechanics.
- PushT planning success rate has not been measured for a trained Rust model.
- SO-100 full training and evaluation have not been run to a publishable report.
- Resume is implemented for the bounded `pusht-full-module-lewm` path, including
  sidecar, `.mpk`, `.safetensors`, config hash, seed, step, AdamW, and RNG
  validation before continuing from the next step.
- Tract inference has not yet been benchmarked from a real trained checkpoint,
  and the demo Space is not release-validated.
- Paper, blog, and release evidence are not complete.

## Definition of Full Completion

The project is complete when the acceptance criteria in `PRD.md` are satisfied
with linked evidence:

- Burn-backed LeWorldModel training in Rust with parity to the published PushT
  checkpoint, beyond the bounded host full-module path.
- Reference parity against the published PushT checkpoint, with fixtures small
  enough for CI.
- Full PushT training, CEM planning evaluation, model card, report, and Hub
  artifacts.
- SO-100 short/full training, warm-start evaluation, report, and Hub artifacts.
- Tract CPU export/runner benchmark from a real checkpoint and a reachable demo
  Space.
- Cost ledger, security controls, credential rotation, release notes, and paper
  artifact updated from actual runs.

## Current Backlog

| Priority | Issue | Work | Acceptance |
|----------|-------|------|------------|
| Done | [#190](https://github.com/AbdelStark/lewm-rs/issues/190) | Lock final LeWM architecture and parity source of truth | Final module dimensions and parity fixture contract are documented; RFC 0002 open question is resolved |
| Done | [#191](https://github.com/AbdelStark/lewm-rs/issues/191) | Replace minimal PushT core with bounded full-module LeWM training | Short CPU train can run `pusht-full-module-lewm` and preserve the artifact contract |
| Done | [#192](https://github.com/AbdelStark/lewm-rs/issues/192) | Implement robust checkpoint restore and resume | Bounded full-module training can resume with model, optimizer, scheduler target, RNG, config hash, seed, and step state validated |
| Done | [#26](https://github.com/AbdelStark/lewm-rs/issues/26)–[#34](https://github.com/AbdelStark/lewm-rs/issues/34), [#40](https://github.com/AbdelStark/lewm-rs/issues/40) | Implement Burn-backed ViT/predictor/SIGReg parity stack and Safetensors export | All `lewm-core` module issues closed; ViT, embedder, MLP, AdaLN-zero, predictor, SIGReg, prediction loss, JEPA wrapper, and Safetensors export implemented with shape and gradient coverage; parity init shape audit passes |
| Done | [#35](https://github.com/AbdelStark/lewm-rs/issues/35) | Implement python/convert_reference.py and param_name_map.py | Scripts implemented; 303 source tensors mapped; audit, convert, verify, and dump commands available |
| Done | [#37](https://github.com/AbdelStark/lewm-rs/issues/37) | Add dump subcommand to convert_reference.py | `python/convert_reference.py dump` captures all per-layer activations as Safetensors; PR [#214](https://github.com/AbdelStark/lewm-rs/pull/214); numerical fixes in PR [#217](https://github.com/AbdelStark/lewm-rs/pull/217) |
| Done | [#38](https://github.com/AbdelStark/lewm-rs/issues/38) | Implement Rust parity test suite | 10 tests for encoder/action_encoder/predictor/pred_proj/sigreg; graceful skip without dumps; PR [#215](https://github.com/AbdelStark/lewm-rs/pull/215) |
| Done | [#39](https://github.com/AbdelStark/lewm-rs/issues/39) | Wire CI parity workflow | Cache + HF download + numerical/shape conditional; PR [#216](https://github.com/AbdelStark/lewm-rs/pull/216) |
| In Progress | [#193](https://github.com/AbdelStark/lewm-rs/issues/193) | Run full PushT training, planning eval, and publish artifacts | Training job running: `6a06f0c43308d79117b90276` (50k steps, GHCR image, CUDA); pending: collect artifacts from `abdelstark/lewm-rs-pusht`, planning eval, model card |
| In Progress | [#194](https://github.com/AbdelStark/lewm-rs/issues/194) | Complete SO-100 short/full training and evaluation path | v11a COMPLETED: `6a070e02e48bea4538b9e2a5` (864s); artifacts at `abdelstark/lewm-rs-so100/train/so100-full-20260515T122820Z/`; v11b `6a070f393308d79117b902de` still running; pending: ONNX export, warm-start eval, model card upload |
| P1 | [#195](https://github.com/AbdelStark/lewm-rs/issues/195) | Finish Tract export, CPU benchmark, and demo Space validation | Export pipeline ready (`python/export_onnx.py`); Tract runner implemented; pending: run benchmark from trained checkpoint, create demo Space |
| P2 | [#196](https://github.com/AbdelStark/lewm-rs/issues/196) | Finish public reports, paper, and release evidence | CHANGELOG updated; ROADMAP updated; pending: README final pass, reports, paper PDF |
| P2 | [#197](https://github.com/AbdelStark/lewm-rs/issues/197) | Complete release operations and security/cost controls | Tokens are rotated, billing guardrails are documented, and no secret is committed |

## Blockers and Required Human Actions

- **GHCR container push**: GitHub Actions GITHUB_TOKEN cannot push to the user-owned
  `ghcr.io/abdelstark/lewm-rs` package without "Manage Actions Access" configured.
  User must visit `https://github.com/users/abdelstark/packages/container/lewm-rs/settings`
  and add `AbdelStark/lewm-rs` repository with Write role. This unblocks the `container`
  job in the release workflow.
- **Token rotation**: The `HF_TOKEN` in `.env` must be rotated before public release.
  Use env vars only; no live secrets in git.
- Three training jobs running on HF (as of 2026-05-15):
  - PushT full: `6a06f0c43308d79117b90276` (50k steps, A10G-large, GHCR image with auto-upload)
  - SO-100 v11a: `6a070e02e48bea4538b9e2a5` (5000 steps + upload to `abdelstark/lewm-rs-so100`)
  - SO-100 v11b: `6a070f393308d79117b902de` (5000 steps + upload, duplicate submission)
  After completion: collect artifacts, run eval, upload model cards.
- **SO-100 v10** (`6a0709973308d79117b902c2`) completed successfully but had no upload step; artifacts were lost.

## Issue Hygiene

The older phase issues remain useful implementation detail, but [#189](https://github.com/AbdelStark/lewm-rs/issues/189)
and this file are the current sequencing source of truth. Close older issues
only with evidence in the closing comment. When an old horizontal issue conflicts
with a current vertical slice, link it to the matching R0-R7 issue instead of
creating a second tracker.

## Next Logical Steps

Both PushT (#193) and SO-100 (#194) training jobs are running on HF. When they
complete:
1. Collect artifacts from `abdelstark/lewm-rs-pusht` and `abdelstark/lewm-rs-so100`.
2. Run `python/export_onnx.py` with the trained safetensors to produce ONNX files.
3. Run `lewm-infer bench --checkpoint-dir <onnx_dir>` to record CPU latency.
4. Run PushT planning eval to measure success rate.
5. Create model cards for both repos on HuggingFace Hub.
6. Fix GHCR permission (user action required) and tag a release.
7. Rotate HF token before release.
