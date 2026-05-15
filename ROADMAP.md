# Roadmap and Completion Backlog

Updated: 2026-05-15

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
| SO-100 preparation | Partially implemented | decode/stats/config/job scaffolds exist; full hosted run evidence is pending |
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
| P1 | [#193](https://github.com/AbdelStark/lewm-rs/issues/193) | Run full PushT training, planning eval, and publish artifacts | HF run, planning success report, model card, uploaded checkpoints, and cost ledger are linked |
| P1 | [#194](https://github.com/AbdelStark/lewm-rs/issues/194) | Complete SO-100 short/full training and evaluation path | Prepared data, short/full runs, warm-start eval, report, and Hub artifacts are linked |
| P1 | [#195](https://github.com/AbdelStark/lewm-rs/issues/195) | Finish Tract export, CPU benchmark, and demo Space validation | Export from a real trained checkpoint works; CPU benchmark and Space smoke are recorded |
| P2 | [#196](https://github.com/AbdelStark/lewm-rs/issues/196) | Finish public reports, paper, and release evidence | README, reports, paper PDF, release checklist, and Hub/blog links match actual artifacts |
| P2 | [#197](https://github.com/AbdelStark/lewm-rs/issues/197) | Complete release operations and security/cost controls | Tokens are rotated, billing guardrails are documented, and no secret is committed |

## Blockers and Required Human Actions

- Numerical parity is now green (all 10 tests pass locally, dumps live on HF). The gate before full PushT training is cleared.
- Full PushT and SO-100 runs require HF quota and explicit cost control; they can now proceed since parity is confirmed.
- [#198](https://github.com/AbdelStark/lewm-rs/issues/198): resolved by moving
  the repo/toolchain contract to Rust 1.89 and adding a direct `lewm-core` Burn
  compile smoke. Burn-backed module structs can now proceed on that toolchain.
- The pasted HF token must be rotated before public release. The repo should
  continue using environment variables and must not commit live secrets.
- SO-100 raw Parquet/MP4 decode remains a Python edge-prep path for v1; Rust
  training consumes prepared HDF5/stat artifacts.

## Issue Hygiene

The older phase issues remain useful implementation detail, but [#189](https://github.com/AbdelStark/lewm-rs/issues/189)
and this file are the current sequencing source of truth. Close older issues
only with evidence in the closing comment. When an old horizontal issue conflicts
with a current vertical slice, link it to the matching R0-R7 issue instead of
creating a second tracker.

## Next Logical Step

Parity stack is fully green. All 10 numerical parity tests pass and dumps are
live on `AbdelStark/lewm-rs-parity-dumps`. CI downloads them automatically when
`HF_TOKEN` secret is set (now set). Proceed to full PushT training ([#193](https://github.com/AbdelStark/lewm-rs/issues/193)).
4. Upload `/tmp/lewm-parity-dumps` to `AbdelStark/lewm-rs-parity-dumps` (HF dataset).
5. Set `HF_TOKEN` in GitHub repo secrets → CI parity job will auto-download and
   run full numerical validation on every PR.

Once parity tests are green locally, proceed to full PushT training ([#193](https://github.com/AbdelStark/lewm-rs/issues/193)).
