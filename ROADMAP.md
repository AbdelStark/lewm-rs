# Roadmap and Completion Backlog

Updated: 2026-05-17 (v0.3.3)

Canonical GitHub tracker: [#189](https://github.com/AbdelStark/lewm-rs/issues/189)

The PRD and accepted RFCs remain the product contract. This document is the
live execution backlog: it records what is proven, what is not yet claimed, and
the next vertical slices needed to finish the project.

## Current Verified State

| Area | Status | Evidence |
|------|--------|----------|
| Specs and workspace | Implemented enough for current local gates | `CARGO_INCREMENTAL=0 make check` passed during this refresh |
| GHCR training image | Legacy image published; F1-verified tag blocked | `ghcr.io/abdelstark/lewm-rs:latest@sha256:831f685a733a801620bbfa3f7ea649a4795ed731934bcb230896d3a47428d3e9` exists, but F1 requires a concrete non-`latest` tag that passes `scripts/verify_runtime_image.py`; GHCR write permission is tracked in #253 |
| HF Jobs short PushT run | Completed | `https://huggingface.co/jobs/abdelstark/6a05cf0ee48bea4538b9ccd6` |
| HF artifact upload | Completed for earlier minimal short run | `abdelstark/lewm-rs-pusht/train/pusht-minimal-lewm-short-20260514T133423Z/` |
| Historical bounded-core PushT job | Completed | `https://huggingface.co/jobs/abdelstark/6a06f0c43308d79117b90276`; 50k steps on A10G-large; loss 0.4912→3.17e-06; wall 318 min; artifacts at `abdelstark/lewm-rs-pusht/train/pusht-full-lewm-20260515T100908Z/`; not a valid F1 full Burn/Jepa checkpoint |
| SO-100 training job | Completed | v11a `6a070e02e48bea4538b9e2a5`: 864s, 5000 steps, loss 0.5002→9.56e-05; artifacts at `abdelstark/lewm-rs-so100/train/so100-full-20260515T122820Z/` |
| Demo Space | Created | `https://huggingface.co/spaces/abdelstark/lewm-rs-demo`; Gradio app with CEM planning via ONNX; loads model from Hub when available |
| SO-100 processed dataset | Uploaded | `abdelstark/so100-pickplace-lewm-ready`; 1.9 GB HDF5 + stats.safetensors; 6,559 timesteps, 50 episodes at 10 fps |
| SO-100 training support | Implemented | `lewm-train` trainer dispatches on `DatasetConfig::So100`; `run_so100_full_lewm_training`; 6-DOF action packing; commit `6add7fd` |
| ONNX export pipeline | Implemented and validated on full-layout/reference checkpoints | `python/export_onnx.py` exports encoder + predictor to opset 18 (onnxruntime) and opset 17 (Tract-compat); current Hub root ONNX files are reference exports, while F1 `onnx-full/` trained-checkpoint artifacts remain blocked |
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
| Numerical parity correctness | Verified | All 10 parity tests pass (`L∞ < 1e-4` on encoder/action_encoder/predictor/pred_proj, `|Δ| < 1e-3` on sigreg); LayerNorm eps=1e-12 and exact-erf GELU fixes; dumps uploaded to `AbdelStark/lewm-rs-parity-dumps`; PR [#217](https://github.com/AbdelStark/lewm-rs/pull/217) |
| Artifact contract | Implemented for smoke and bounded PushT train | run report, losses JSONL, checkpoint sidecar, `.mpk`, `.safetensors`, parity JSON |
| Optional observability | Implemented as optional infra | `infra/otel/`; CI and smoke runs do not require OTLP |
| SO-100 full training | Completed | v11a job `6a070e02e48bea4538b9e2a5` completed (864s, 5000 steps, A10G-large); artifacts at `abdelstark/lewm-rs-so100/train/so100-full-20260515T122820Z/` (safetensors, mpk, losses, report, parity JSON) |
| Inference/export | ONNX export complete + Tract benchmark done | `python/export_onnx.py` validated end-to-end: reference safetensors → 303 keys recovered → onnxruntime ONNX (dynamo opset 18) + Tract-compat ONNX (legacy opset 17, fixed-batch) both uploaded to `abdelstark/lewm-rs-pusht`; onnxruntime inference verified; demo Space updated (sdk_version 5.33.0); Tract CPU benchmark: 4.08s median/episode (release build p50, Apple M3 ARM, 5 CEM iterations × 1024 candidates — debug and release identical, hot path is Tract); Tract-compat files at `tract-compat/` subfolder; `onnx_export.json` action_dim bug fixed (was recording raw 2-DOF instead of inferred smoothed 10-DOF) |
| GPU inference (Burn) | Implemented | `lewm_infer::runner::BurnJepaRunner<B>` is backend-generic and runs the Burn `Jepa<B>` module from a Safetensors file. CPU wiring (`burn-cpu`, default-on) ships in `lewm-infer` and is selectable via `lewm-infer --backend burn-cpu`. CUDA wiring lives in the new [`lewm-gpu`](crates/lewm-gpu/) crate per RFC 0007 (`scripts/check_layers.py` keeps `burn-cuda`/`burn-autodiff`/`nvml-wrapper` out of `lewm-infer`); call `lewm_gpu::load_cuda_runner` from a downstream binary. CI verifies `cargo clippy --workspace --all-targets` is warning-free under the default, `cpu-only`, and `parity-fixtures` feature matrices |
| Parity eval CLI | Implemented | `lewm-infer eval --dumps-dir DIR --backend BACKEND --safetensors WEIGHTS` compares any runner against the reference parity dumps and emits per-stage L∞/RMSE JSON; backed by `lewm_core::import` (backend-generic Safetensors → `Jepa<B>` loader); see `reports/gpu_inference.md` |
| Cross-stack benchmark harness | Implemented | `python/eval_compare.py` runs the PyTorch reference baseline (CPU + CUDA when available) and the Rust `lewm-infer eval` for each requested backend, then merges the JSON into a single side-by-side report |
| Reports and paper | Complete (eval TBDs remain) | `paper/lewm-rs.md` §6.1 training curves filled; `reports/pusht_training.md`, `reports/so100_training.md`, `reports/inference.md`, `reports/cost.md` ($11.70), `reports/release_checklist.md`; `python/plot_curves.py` + CSV in `paper/figures/` |
| Quality gate | Passing | `CARGO_INCREMENTAL=0 make check` passes: fmt, clippy, cargo check, Python lint (`make py-lint` via Ruff), specs, jobs, otel, SO-100 contract, nondet lint, cost ledger, deny, audit |
| Python lint baseline | Implemented | Ruff configured in `python/pyproject.toml` (E, F, W, B, UP, SIM, RUF, I); `make py-lint` from the root and `make check` inside `python/` enforce zero diagnostics across `python/` and `scripts/`; `python/Makefile` activates the optional `make accept` hook |
| Supply-chain attestation | Implemented | `release.yml` emits GitHub built-in build provenance for binaries, SBOM, and the container image; cosign signs the image by digest; CycloneDX SBOM generated deterministically (`scripts/sbom.py`); reproducible Linux+macOS builds verified by `verify-reproducible` |
| Dependency automation | Implemented | `.github/dependabot.yml` covers Cargo, GitHub Actions, Docker, and Python; Burn/Tract/HDF5 major upgrades frozen per ADR 0002 and RFC 0007 |
| HF Jobs cost guard | Implemented | `scripts/launch_hf_job.py --cost-cap-usd` performs pre-flight worst-case spend check (default $20, soft cap from CLAUDE.md); `--image-tag` pins the GHCR tag without editing YAML; `python/hf_pricing.py` covers l4x1 / l4x4 / cpu-upgrade / h100x8 |
| Container hygiene | Implemented | Dockerfile runs `lewm-train` under `tini` (PID 1), ships a `HEALTHCHECK`, populates OCI metadata (revision/created/version/base.name), accepts build args for the release workflow to stamp |
| Release runbook | Implemented | `RELEASE.md` documents the pre-flight checklist, tag-cut commands, audit-trail expectations, and rollback procedure |
| Per-crate docs | Implemented | Each workspace crate ships a `README.md` (layering, module map, public surface, feature flags) cross-linked to its RFC |
| Pre-commit hooks | Implemented | `.pre-commit-config.yaml` wires gitleaks, ruff, cargo fmt, and the cheap project validators into the standard `pre-commit` framework as a fast local pre-flight |

## Non-Claims

- The Burn ViT parity stack is now numerically validated: all 10 activation-level parity tests pass against the locked reference PyTorch checkpoint; dumps are live in `AbdelStark/lewm-rs-parity-dumps` and CI downloads them automatically when `HF_TOKEN` is set.
- `pusht-full-module-lewm` is a config-shaped host training path, not the final
  Burn ViT parity stack. It is a narrow real training path for validating data,
  module boundaries, training, checkpoint, resume, upload, and job mechanics.
- PushT planning success rate has not been measured for a trained Rust model.
- SO-100 full training completed (v11a: 5000 steps, 864s, loss 0.5002→9.56e-05); warm-start evaluation has not been run.
- Resume is implemented for the bounded `pusht-full-module-lewm` path, including
  sidecar, `.mpk`, `.safetensors`, config hash, seed, step, AdamW, and RNG
  validation before continuing from the next step.
- Tract CPU benchmark: 4.08s/episode median (release build, Apple M3 ARM); debug and release are identical because the hot path is Tract's pre-compiled ONNX engine.
- PushT CEM planning success rate has not been measured (eval pending).
- SO-100 warm-start ablation (init from PushT vs. random) has not been run.
- Blog post has not been written.
- Demo Space functional state pending verification after rebuild with sdk_version 5.33.0.

## Definition of Full Completion

The project is complete when the acceptance criteria in `PRD.md` are satisfied
with linked evidence:

- Burn-backed LeWorldModel training in Rust with parity to the published PushT
  checkpoint, beyond the bounded host full-module path.
- Reference parity against the published PushT checkpoint, with fixtures small
  enough for CI.
- Full Burn/Jepa PushT training, CEM planning evaluation, model card, report,
  and Hub artifacts.
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
| In Progress | [#193](https://github.com/AbdelStark/lewm-rs/issues/193) | Run full Burn/Jepa PushT training, planning eval, and publish artifacts | Historical bounded-core training completed: `6a06f0c43308d79117b90276` (50k steps, loss 0.4912→3.17e-06, 318 min); artifacts at `abdelstark/lewm-rs-pusht/train/pusht-full-lewm-20260515T100908Z/` are not valid F1 sources. Pending: approval-gated full Burn/Jepa run, ONNX export, CEM planning eval, model card |
| In Progress | [#194](https://github.com/AbdelStark/lewm-rs/issues/194) | Complete SO-100 short/full training and evaluation path | v11a COMPLETED: `6a070e02e48bea4538b9e2a5` (864s, 5000 steps); artifacts at `abdelstark/lewm-rs-so100/train/so100-full-20260515T122820Z/`; model card uploaded; ONNX export skipped (SO-100 checkpoint uses bounded model, not full ViT); pending: warm-start eval, model card final pass |
| Done | [#195](https://github.com/AbdelStark/lewm-rs/issues/195) | Finish Tract export, CPU benchmark, and demo Space validation | Tract-compat ONNX exported (opset 17, fixed-batch, causal-mask buffer); `lewm-infer bench` benchmark: ~4.1s median/episode (debug, M-series Mac); both onnxruntime and tract-compat variants uploaded to `abdelstark/lewm-rs-pusht`; demo Space `app.py` fixed (auto-detects action_dim, downloads .data files) |
| Done | [#196](https://github.com/AbdelStark/lewm-rs/issues/196) | Finish public reports, paper, and release evidence | All reports done: pusht_training.md, so100_training.md, inference.md, cost.md ($11.70), release_checklist.md; paper §6.1 training curves filled; README final pass; cross-links done; pandoc CI works; CEM eval §6.2 and SO-100 warm-start §7.3 remain TBD |
| P2 | [#197](https://github.com/AbdelStark/lewm-rs/issues/197) | Complete release operations and security/cost controls | Billing guardrails documented; pending: HF_TOKEN rotation (user action), GHCR package permissions (user action) |

## Blockers and Required Human Actions

- **GHCR container push**: GitHub Actions GITHUB_TOKEN cannot push to the user-owned
  `ghcr.io/abdelstark/lewm-rs` package without "Manage Actions Access" configured.
  User must visit `https://github.com/users/abdelstark/packages/container/lewm-rs/settings`
  and add `AbdelStark/lewm-rs` repository with Write role. This unblocks the `container`
  job in the release workflow.
- **Token rotation**: The `HF_TOKEN` in `.env` must be rotated before public release.
  Use env vars only; no live secrets in git.
- **F1 / CEM eval** (#193, #243, #244): the live
  `abdelstark/lewm-rs-pusht/train/pusht-full-lewm-20260515T100908Z/`
  checkpoint is bounded-core only and is rejected by the trained-checkpoint
  ONNX exporter. Need a human-approved full Burn/Jepa F1 run under
  `train/pusht-full-burn-jepa-*`, then export `onnx-full/`, run CEM planning
  eval, and measure success rate on 50 test episodes. Target ≥ 87%.
- **SO-100 warm-start** (#194): scratch checkpoint shipped; needs the warm-started run
  from PushT epoch-10 and Spearman delta to close the ablation.

## Issue Hygiene

The older phase issues remain useful implementation detail, but [#189](https://github.com/AbdelStark/lewm-rs/issues/189)
and this file are the current sequencing source of truth. Close older issues
only with evidence in the closing comment. When an old horizontal issue conflicts
with a current vertical slice, link it to the matching R0-R7 issue instead of
creating a second tracker.

## Next Logical Steps

Historical bounded-core PushT training (50k steps, 318 min, A10G-large) and
SO-100 v11a (5000 steps, 864s) are both complete; artifacts live on the Hub.
Quality gate (`make check`) is passing with the Python lint surface promoted to
Ruff (see `python/pyproject.toml`). The F1 full Burn/Jepa PushT release
checkpoint is still pending human-approved paid execution.

Immediate next actions:
1. Run the approved F1 full Burn/Jepa PushT job or source-build fallback after
   explicit human approval.
2. Export and verify `onnx-full/` from the resulting
   `train/pusht-full-burn-jepa-*` safetensors.
3. Run CEM planning eval with the exported ONNX and record the success rate
   (target ≥ 87 %) — closes the acceptance criterion for #193.
4. Upload the PushT model card with the measured eval metrics.
5. Launch the SO-100 warm-started training arm and compute the warm-start
   Spearman delta (closes the acceptance criterion for #194).

Remaining release steps (user actions required):
5. Fix GHCR package permissions (see Blockers above).
6. Rotate `HF_TOKEN` before pushing/releasing.
7. Tag release after the CEM eval, warm-start delta, and model cards are done.
