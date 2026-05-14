# Roadmap and Completion Backlog

Updated: 2026-05-14

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
| Core prediction loss | Implemented as MSRV-compatible kernel | `lewm_core::prediction_loss`; first slice of [#32](https://github.com/AbdelStark/lewm-rs/issues/32) |
| Artifact contract | Implemented for smoke and bounded PushT train | run report, losses JSONL, checkpoint sidecar, `.mpk`, `.safetensors`, parity JSON |
| Optional observability | Implemented as optional infra | `infra/otel/`; CI and smoke runs do not require OTLP |
| SO-100 preparation | Partially implemented | decode/stats/config/job scaffolds exist; full hosted run evidence is pending |
| Inference/export | Partially implemented | Tract runner/export scaffolds and tests exist; real trained-checkpoint benchmark and Space validation are pending |

## Non-Claims

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
| P0 | [#26](https://github.com/AbdelStark/lewm-rs/issues/26)-[#33](https://github.com/AbdelStark/lewm-rs/issues/33) | Replace host full-module path with Burn-backed ViT/predictor/SIGReg parity stack | `lewm-core::Jepa` modules train with autodiff and pass reference parity fixtures; [#32](https://github.com/AbdelStark/lewm-rs/issues/32) has an MSRV-compatible prediction-loss kernel started |
| P1 | [#193](https://github.com/AbdelStark/lewm-rs/issues/193) | Run full PushT training, planning eval, and publish artifacts | HF run, planning success report, model card, uploaded checkpoints, and cost ledger are linked |
| P1 | [#194](https://github.com/AbdelStark/lewm-rs/issues/194) | Complete SO-100 short/full training and evaluation path | Prepared data, short/full runs, warm-start eval, report, and Hub artifacts are linked |
| P1 | [#195](https://github.com/AbdelStark/lewm-rs/issues/195) | Finish Tract export, CPU benchmark, and demo Space validation | Export from a real trained checkpoint works; CPU benchmark and Space smoke are recorded |
| P2 | [#196](https://github.com/AbdelStark/lewm-rs/issues/196) | Finish public reports, paper, and release evidence | README, reports, paper PDF, release checklist, and Hub/blog links match actual artifacts |
| P2 | [#197](https://github.com/AbdelStark/lewm-rs/issues/197) | Complete release operations and security/cost controls | Tokens are rotated, billing guardrails are documented, and no secret is committed |

## Blockers and Required Human Actions

- Reference weights or exact upstream dumps may require human-owned HF access.
  If the reference checkpoint cannot be used directly in CI, R0 must produce a
  small derived parity fixture.
- Full PushT and SO-100 runs require HF quota and explicit cost control. The
  long runs should not start until the Burn parity stack is green locally.
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

Resolve the [Burn/MSRV gate](https://github.com/AbdelStark/lewm-rs/issues/198), then continue [#26](https://github.com/AbdelStark/lewm-rs/issues/26)-[#33](https://github.com/AbdelStark/lewm-rs/issues/33):
replace the host full-module path with the Burn-backed ViT/predictor/SIGReg
parity stack. Until that decision is made, keep landing MSRV-compatible core
math contracts such as prediction loss and SIGReg constants. Hosted GPU time
should still wait until the Burn parity stack is green locally.
