# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Docker / HF Jobs reproduction docs now match the checked-in Dockerfile and
  release workflow: the runtime image ships `lewm-train` under `/workspace`,
  GHCR publication happens from the release workflow, and production HF Jobs
  should pin an image tag through `scripts/launch_hf_job.py --image-tag`.
- **Burn 0.20.1 → 0.21.0** (per ADR 0003). Updated all seven workspace
  Burn dependencies (`burn`, `burn-core`, `burn-cuda`, `burn-ndarray`,
  `burn-autodiff`, `burn-import`, `burn-train`). Migrated five
  deprecated `Ignored<T>` field wrappers in `lewm-core` (`vit`, `mlp`,
  `jepa`, `predictor`, `losses/sigreg`) to the recommended
  `#[module(skip)]` attribute, updated `B::ad_enabled()` callers to
  the new `(&device)` signature, switched `Shape::dims` (now private)
  to `Shape::as_slice()` at two call sites, and refreshed
  `<B as Backend>::Device` to the `Device<B>` type alias.
  Tract remains pinned at `=0.22.1` (latest stable; `0.23.0` is still
  only a `-dev` pre-release). MSRV documentation aligned to the
  already-pinned `1.95.0` toolchain. Workspace tests, clippy, layer /
  spec / jobs / nondet validators, and Python validators all pass.

### Added

- **Opt-in full Burn/Jepa PushT checkpoint path**: `lewm-train` now supports
  `experimental.pusht_train_mode = "full_burn_jepa"` for CPU Burn autodiff
  PushT runs that train `lewm_core::Jepa` directly and write a real Burn
  `NamedMpk` record plus full safetensors mirror. Checkpoint safetensors now
  preserve both F32 and I64 tensors, covering BatchNorm `num_batches_tracked`
  keys required by the ONNX exporter contract. The checked-in PushT jobs still
  default to the bounded trainer until the production 50k full-JEPA run is
  launched under the release leash.
- **PushT ONNX export gate**: `python/export_onnx.py` now has an explicit
  `--variant {both,onnxruntime,tract-compat}` path that writes
  `onnxruntime/` (opset 18, dynamic batch) and `tract-compat/` (opset 17,
  fixed batch) layouts with root/per-variant `onnx_export.json` sidecars.
  New `python/verify_onnx.py` verifies ONNX Runtime shape execution for both
  variants. `reports/pusht_onnx_export.md` records the F1 blocker: the current
  50k PushT artifact is a 14-tensor bounded host-core checkpoint, not the
  255-tensor full Burn/Jepa mirror required to recover the 303 PyTorch source
  keys for trained-checkpoint ONNX export. The exporter now fails that
  mismatch up front with a checkpoint contract diagnostic instead of surfacing
  a raw missing-key error, and that invalid-checkpoint preflight no longer
  requires `torch`. `python/upload_checkpoints.py --dry-run` now validates the
  final `onnx-full/` upload command without requiring `HF_TOKEN` or the `hf`
  CLI, while real uploads still require both. The F1 handoff wrapper now
  rejects legacy `train/pusht-full-lewm-*` prefixes before printing or running
  commands, so the known bounded-core 50k artifact cannot be mistaken for a
  full Burn/Jepa source run. Its ONNX verification stage now injects
  `onnxruntime` explicitly instead of assuming the `parity` Python extra
  provides it. `scripts/audit_pusht_full_safetensors.py` and
  `reports/pusht_full_safetensors_hub_audit.json` now audit public PushT
  `.safetensors` headers without downloading full checkpoint payloads, showing
  all six current public candidates fail the F1 full-run source preflight and
  that zero `train/pusht-full-burn-jepa-*` step-50000 candidates are ready for
  the export wrapper.
- **SO-100 warm-start wiring + preflight**: `lewm-train` now consumes
  `training.warmstart_from` for fresh SO-100 full-module training starts,
  transfers shared PushT modules through the RFC 0012 warm-start boundary,
  preserves the fresh SO-100 action encoder, resets AdamW state, and records
  warm-start provenance in reports/checkpoints. `reports/so100_warmstart.md`
  still blocks F3 launch because the compatible trained PushT source checkpoint
  is absent and paid launch requires human approval.
  Added `jobs/train_so100_warmstart.yaml` as a fail-closed HF Job spec: it
  refuses to run unless `LEWM_PUSHT_WARMSTART_MPK` names a compatible PushT
  `.mpk` source, downloads SO-100 data, runs the warm-start config with an
  explicit `training.warmstart_from` override, and uploads to
  `train/so100-warmstart-*`. The job now runs
  `scripts/check_warmstart_source.py` after downloading the PushT `.mpk`, so
  stale `schema_version = 1.0.0` / minimal records fail before paid training
  begins. The checker now makes the current boundary explicit: full Burn/Jepa
  `NamedMpk` records are not accepted by the bounded-core SO-100 warm-start
  path. `scripts/launch_hf_job.py` also refuses the warm-start job before
  dry-run or submit unless `LEWM_PUSHT_WARMSTART_MPK` is set to a relative
  `.mpk` source path. `scripts/pusht_warmstart_source_smoke.py` and
  `reports/pusht_warmstart_source_smoke.json` prove the current bounded PushT
  writer can emit a launch-compatible `schema_version = 1.1.0` source record
  with the expected 41,856-parameter layout. Reproduction/result docs now use
  the checked-in `train_*.yaml` job names, describe the PushT 50k artifact as
  a bounded 14-tensor host-core checkpoint rather than a trained 255-tensor
  Burn/Jepa mirror, and no longer present the legacy 2026-05-15 step-50000
  `.mpk` as a valid SO-100 warm-start source. The warm-start launcher now also
  rejects placeholder and globbed `LEWM_PUSHT_WARMSTART_MPK` values before
  rendering an HF command.
  `scripts/audit_pusht_warmstart_sources.py` and
  `reports/pusht_warmstart_hub_audit.json` now audit every public PushT `.mpk`
  candidate and show that all six current Hub candidates are rejected by the
  warm-start verifier. The audit now records observed record kind and parameter
  counts too, showing the 50k `pusht-full-lewm-*` `.mpk` is a 56-param
  `minimal-lewm` record rather than a migratable 41,856-param bounded-module
  source.
- **Release blocker gate**: new `conformance/release_blockers.json` and
  `scripts/check_release_blockers.py` keep `make check` schema-validating known
  blockers while making `make accept` fail until the F1 ONNX and F3 warm-start
  blockers are resolved. `reports/phase_a_handoff.json` and
  `scripts/check_phase_a_handoff.py` now pin the ordered F1/F3 handoff
  commands, human-approval gates, accepted/rejected PushT source prefixes, and
  warm-start source requirements in one machine-checked artifact. The handoff
  and approval validators now also require placeholder commands to declare the
  exact placeholder and replacement rule before they can pass. The handoff
  validator also cross-checks `conformance/release_blockers.json` so F1/F3
  cannot drift away from their blocked Phase A release status, and enforces
  that dry-run, execute, and upload commands stay in their intended stages.
  README, conformance, and quality-gate docs now describe the Phase A handoff
  gate and no longer present `jobs/train_pusht.yaml` as a reproduction command
  for the legacy bounded PushT artifact. F1 handoff docs now use a shell-safe
  `REPLACE_WITH_UTC_TIMESTAMP` token, and `scripts/f1_export_pusht_onnx.py`
  rejects placeholders or non-`YYYYMMDDTHHMMSSZ` Hub run suffixes before any
  download, export, or upload command runs. `reports/phase_a_approval.json`
  and `scripts/check_phase_a_approval.py` now pin the paid approval packet and
  make explicit that F1 ($18.00) plus F3 ($9.00) exceeds the $20.00 session cap
  if approved together.
- **Model-card accuracy**: local PushT and SO-100 Hub cards now label the
  current uploads as bounded trainer artifacts instead of full release
  checkpoints, and `scripts/upload_model_cards.py --dry-run` no longer requires
  `HF_TOKEN`.
- **F1 root-cause report**: `reports/full_burn_jepa_training_gap.md` records
  why the current published PushT artifact cannot satisfy the required
  exact 255 Burn destination / 303 PyTorch source ONNX export contract and
  lists the remaining gates before ONNX export can be marked complete.
- **PushT job selection**: the approval-gated production PushT job now selects
  CPU-backed `experimental.pusht_train_mode = "full_burn_jepa"`, reports
  `--device cpu`, defaults to 50k steps, and uploads to
  `train/pusht-full-burn-jepa-*` only after `python/export_onnx.py
  --check-contract-only` verifies the produced safetensors contains the exact
  255 Burn destination tensors and recovers the 303 PyTorch source keys. The
  training image now includes `safetensors` for that pre-upload gate. Bounded
  PushT smoke/short jobs keep
  `pusht-bounded-module-lewm` TrackIO/upload labels, future bounded PushT
  checkpoints use bounded run IDs / record kinds / train-report modes, and
  `scripts/check_jobs.py` rejects bounded PushT jobs that publish under
  `pusht-full*` paths.
- **ONNX export key contract**: the Rust `Jepa` safetensors exporter now skips
  generated `sigreg.consts.*` tensors, matching the import contract and the
  Python ONNX exporter map. A shared fixture locks the 255 Burn destination
  tensor names used to recover the 303 PyTorch source keys before F1 upload.
- **`lewm-train` eval adapter**: new `lewm_train::eval` module provides
  `JepaCemCostModel<B>`, a `lewm_plan::CemCostModel` adapter that
  tensorises CEM batches and forwards to the parity-verified
  `Jepa<B>::get_cost`. Strictly validates `horizon_plan ==
  jepa.horizon - jepa.history_size`, `action_dim`, `latent_dim`, and
  `history_len`, and short-circuits empty batches. Covered by four
  unit tests including an end-to-end `Cem::plan` round-trip on a
  compact synthetic JEPA. Closes the structural gap between the
  scaffolded `lewm-eval pusht` binary and a model-backed planner;
  what remains is loading a real checkpoint + running the gym-pusht
  loop.
- **Workspace lints**: `expect_used` is now `deny` (was `warn`).
  Production code has zero `.expect()` calls outside `#[cfg(test)]`
  modules, so the deny is a permanent guarantee rather than a
  current state.

### Added (release polish)

- **MLOps / supply chain**: `.github/dependabot.yml` watches Cargo, GitHub
  Actions, the Dockerfile, and the Python edge layer on a weekly cadence with
  Burn / Tract / HDF5 major-version freezes that match ADR 0002 and RFC 0007.
- **Release supply chain**: the `release` workflow now generates GitHub
  built-in build provenance attestations for the binaries, the SBOM, and the
  container image. The image is published with a stable digest, cosign-signed
  by that digest, and the provenance is pushed to the registry alongside it.
- **Release runbook**: new top-level `RELEASE.md` with the full pre-flight
  checklist, tag-cut commands, audit-trail expectations, and the rollback
  procedure.
- **Per-crate READMEs**: each of the eight workspace crates now ships a
  `README.md` with its layering rule, module map, public-surface contract,
  and feature gates. The READMEs cross-link to the RFCs they implement so the
  spec ↔ code traceability is one click in either direction.
- **Local pre-commit**: new `.pre-commit-config.yaml` wires gitleaks, Ruff
  (`check` + `format --check`), `cargo fmt --check`, and the four Python
  validators (`check_layers`, `check_specs`, `check_jobs`, `check_nondet`)
  into the standard `pre-commit` framework. Setup: `pipx install pre-commit
  && pre-commit install`. The CI gate stays authoritative.
- **Container hygiene**: the runtime image now ships `tini` as PID 1 for
  zombie reaping + signal forwarding, a `HEALTHCHECK` that validates the
  binary + Python edge layer, OCI metadata labels (`revision`, `created`,
  `version`, `base.name`), and parameterised build args (`BUILD_REVISION`,
  `BUILD_DATE`, `SOURCE_VERSION`) populated by the release workflow.
- **HF Jobs cost guard**: `scripts/launch_hf_job.py` now performs a
  pre-flight worst-case cost estimate (`hardware-rate * YAML-timeout`) and
  refuses to submit when the spend would exceed `--cost-cap-usd` (default
  `20.00`, matching the per-session soft cap in `CLAUDE.md`). It also accepts
  `--image-tag VERSION` (or `LEWM_IMAGE_TAG`) to pin the GHCR image to a
  release tag without editing the YAML.
- **HF pricing table**: `python/hf_pricing.py` adds `l4x1`, `l4x4`,
  `cpu-upgrade`, and `h100x8` flavours so the cost guard and the post-hoc
  ledger speak the same vocabulary.
- **CI ergonomics**: `concurrency: cancel-in-progress` for PR runs in `ci`
  and `nightly` workflows; `workflow_dispatch` triggers added so operators
  can replay a green build on demand; `CARGO_NET_RETRY` / `RUSTUP_MAX_RETRIES`
  surfaced to env to ride out transient network blips.
- **CODEOWNERS**: explicit ownership for `paper/`, `python/`, `jobs/`,
  `infra/`, `RELEASE.md`, `SECURITY.md`, `Dockerfile`, `.gitleaks.toml`, and
  `.pre-commit-config.yaml` so review routing matches what is touched.

### Changed

- `lewm-train` trainer: removed the legacy `pusht_lewm` (14-parameter "minimal
  LeWM") module and its orphaned 600-line dead-code chain inside
  `trainer.rs` (`run_pusht_minimal_lewm_training`,
  `write_pusht_minimal_lewm_checkpoint`, `apply_pusht_minimal_lewm_adamw`,
  `PushtMinimalLewm{AdamWState,Outcome,Record}`, the `_features` /
  `_example_from_sample` helpers, and `sequential_training_sample_index`).
  The replacement `pusht_full` path has subsumed it for the entire RFC 0005
  release line; nothing inside or outside the workspace referenced these
  symbols. Net: −1 463 lines from `lewm-train`.
- `lewm-train` trainer: unified the two near-identical PushT and SO-100 full
  LeWM training loops behind a single generic `run_full_lewm_training` driven
  by a new `TrainingSampleSource` trait and an `&dyn Fn` example builder.
  `write_pusht_full_lewm_checkpoint` / `write_so100_full_lewm_checkpoint`
  delegate to a shared `write_full_lewm_checkpoint`. No behaviour change; the
  per-step iteration order, RNG advancement, and on-disk artifacts are
  identical (validated by the existing 57-test trainer suite).

### Added

- Python lint baseline: Ruff configured in `python/pyproject.toml` (rule
  families `E`, `F`, `W`, `B`, `UP`, `SIM`, `RUF`, `I`) with `param_name_map`
  and sibling helpers declared as first-party so import ordering is stable
  whether Ruff is invoked from the repo root or `python/`. New `python/Makefile`
  exposes `make check` / `make lint` / `make fix` and activates the existing
  `make accept` hook in the root `Makefile`. The root `Makefile` gains a
  `py-lint` target that is now part of `make check`; it falls back to a
  `py_compile` sweep when `ruff` is not installed so minimal environments
  degrade gracefully.
- SO-100 full training completed: v11a job `6a070e02e48bea4538b9e2a5` (5000 steps,
  864s, A10G-large); artifacts at `abdelstark/lewm-rs-so100/train/so100-full-20260515T122820Z/`
  (safetensors, mpk, losses, report, parity JSON); model card uploaded.
- Tract-compatible ONNX export: `python/export_onnx.py` updated to opset 17 +
  `dynamo=False` (legacy TorchScript exporter) + fixed-batch mode (no dynamic_axes)
  for Tract 0.22.1 compatibility; pre-registers causal mask as `nn.Module` buffer
  to avoid dynamic `torch.ones(T,T)` in the ONNX graph; infers `action_dim`
  automatically from smoother Conv1d weight shape.
- Tract CPU benchmark: `lewm-infer bench` produces ~4.1s median per planning
  episode (debug build, M-series Mac, 5 CEM iterations × 1024 candidates,
  H=3 history steps, action dim=10); Tract-compat ONNX files uploaded to
  `abdelstark/lewm-rs-pusht/tract-compat/`.
- Demo Space live at `abdelstark/lewm-rs-demo` (Gradio, onnxruntime CEM planning,
  auto-detects action_dim from predictor input shape, downloads `.onnx.data`
  external weight files alongside `.onnx`).
- Model cards uploaded to `abdelstark/lewm-rs-pusht` and `abdelstark/lewm-rs-so100`.

### Changed

- Python helpers cleaned up against the new Ruff baseline: `zip(..., strict=True)`
  in `python/export_onnx.py`'s QKV inverse-transforms (closes a silent
  length-mismatch class), `raise SystemExit(130) from None` on
  `KeyboardInterrupt` in `python/decode_so100_to_h5.py`, `contextlib.suppress`
  in place of bare `try/except/pass` for AV seek failures, `[*a, b]`-style
  list construction over `+` concatenation, and unused unpacking renamed to
  `_config_path`. No behaviour change.
- `ROADMAP.md` and `reports/release_checklist.md` refreshed: PushT training
  is recorded as complete, stale "still running" / "4 commits ahead of main"
  notes removed, and the Python lint baseline added to the quality-gate row.
- `python/export_onnx.py`: switched ONNX opset 18 → 17 and `dynamo=True` →
  `dynamo=False` for Tract compatibility; removed `dynamic_axes`; causal mask
  is now a pre-registered buffer in `LeWMPredictorModule.__init__`.
- `lewm-infer` `bench` subcommand: default `--history-steps` changed from 2 to
  3 to match predictor's fixed `T=num_frames` from config (was causing a Tract
  shape clash at runtime).

- `python/convert_reference.py dump` subcommand for capturing per-layer
  activations from the locked PushT reference checkpoint as Safetensors (RFC
  0008 §4.2); supports `--skip-sha256` and `--fixture-seed` overrides.
- Rust parity test suite (`crates/lewm-core/tests/parity_*.rs`) with 10 tests
  (encoder, action_encoder, predictor, pred_proj, sigreg) gated behind
  `parity-fixtures` feature and `LEWM_PARITY_DUMPS` / `LEWM_REFERENCE_SAFETENSORS`
  env vars; gracefully skip without dumps.
- CI `parity` workflow caches dumps keyed on fixture hash, downloads from
  `AbdelStark/lewm-rs-parity-dumps` when `HF_TOKEN` is available, and runs
  full numerical tests or falls back to shape-only.
- Numerical parity verified: all 10 activation-level parity tests pass against
  the locked PushT reference checkpoint (L∞ < 1e-4 encoder/action_encoder/
  predictor/pred_proj; |Δ| < 1e-3 sigreg) with LayerNorm eps=1e-12 and
  exact-erf GELU fixes; dumps uploaded to `AbdelStark/lewm-rs-parity-dumps`.
- `lewm_core::export::to_safetensors` deterministic Safetensors export for
  `Jepa` parameters with BatchNorm running state and integer counters.
- SO-100 training support in `lewm-train`: `So100Dataset`, dimension-agnostic
  `PushtFullLewmCore` (handles 6-DOF SO-100 actions directly), full training
  loop, checkpoint/resume, and artifact upload via `SO100_FULL_LEWM_RUN_ID`.
- `python/decode_so100_to_h5.py`: decodes `lerobot/svla_so100_pickplace`
  Parquet+AV1 data into RFC 0012 HDF5 at 10 fps / 224×224.
- `python/compute_so100_stats.py`: wraps the `compute_stats` Rust binary to
  compute SO-100 action mean/std safetensors.
- `python/export_onnx.py`: exports a trained Burn safetensors checkpoint to
  ONNX opset 18 (encoder + predictor) for Tract CPU inference via inverse
  parameter-name-map transform.
- PushT full training job submitted to HuggingFace Jobs
  (`abdelstark/6a06f0c43308d79117b90276`; 50k steps on A10G-large).
- SO-100 training job submitted to HuggingFace Jobs
  (`abdelstark/6a06fe17e48bea4538b9e1cb`; 10 epochs on A10G-large).

### Changed

- Release workflow `build-linux-static` and `verify-reproducible` jobs: added
  `git` to apt-get installs (required before `git config` in Ubuntu container).
- Release workflow `release-notes` awk regex corrected from `\\[` to `\[` for
  literal bracket match in CHANGELOG section headers.
- Release workflow `container` job: added `packages: write` permission for
  GHCR image push via GITHUB_TOKEN.

- `lewm-infer` CPU inference runner for ONNX/NNEF graph pairs via Tract with
  plan, bench, serve, and verify subcommands.
- `lewm-infer` export verifier fallback ladder (ONNX → NNEF → Burn-direct)
  with RFC 0007 L∞ tolerance and model-card section renderer.
- `lewm-infer` CEM planner for CPU-side action search from exported graphs.
- SO-100 processed dataset uploaded to `abdelstark/so100-pickplace-lewm-ready`
  (1.9 GB HDF5, 6,559 timesteps, 50 episodes at 10 fps).



- RFC 0013 RNG sub-stream state serialization and a nondeterminism lint for
  Rust sources.
- RFC 0009 system metric samplers for CPU utilization, process RSS, disk usage,
  and optional NVML GPU utilization/memory telemetry.
- Initial Rust workspace, spec validation, quality gate configuration, and OSS
  scaffolding.
- Optional `lewm-telemetry/nvtx` profiling layer and local profiling artifact
  workflow for RFC 0014.
- Cost-ledger append, integrity-check, pricing, backfill, and CI cap gates.
- Hugging Face model-card renderer with YAML frontmatter, provenance, and
  attribution blocks.
- Hugging Face Hub client upload pipeline with SHA-256 idempotency and retry
  policy.
- PushT evaluation driver plumbing, pinned eval config, `lewm-eval pusht`
  CLI, and JSON/Markdown/Parquet eval artifacts.
- PushT JSON-RPC sidecar with a deterministic mock backend and pinned
  `gym-pusht` simulator extra for real eval runs.
- RFC 0006 CEM planner with deterministic `rng:cem` sampling, chunked cost
  evaluation fallback, and toy-quadratic convergence tests.
- Resume-aware PushT full-training HF Jobs spec and the ml-intern approval
  leash entry for human-gated full training launches.
- PushT smoke and short HF Jobs specs with local schema checks for hardware,
  timeout, environment passthrough, and checkpoint upload steps.
- Root TOML config loader with layered environment/CLI overrides, validation,
  canonical BLAKE3 hashing, and the `configs/pusht.toml` fixture.
- `lewm-train` clap CLI contract for train, smoke, parity, eval, and convert
  commands, including config overrides and provenance preamble formatting.
- Added `lewm-train` trainer state-machine, transition, parity-probe, smoke
  slope, and eval-cadence primitives.
- Added `lewm-train` resume detection, RNG restoration, and shutdown-signal
  checkpoint handler primitives.
- Added `lewm-train` checkpoint sidecar, atomic write, safetensors mirror, and
  pruning primitives.
- Added `lewm-train` inner-step accumulation, clipping, and non-finite guard
  primitives.
- Added `lewm-train` mixed-precision policy contracts.
- Added `lewm-train` cosine warmup learning-rate schedule.
- Added `lewm-train` AdamW RFC defaults and decay/no-decay parameter
  partitioning.
- Direct `lewm-core` Burn `0.20.1` dependency and compile smoke for the
  Rust `1.89.0` parity implementation path.
- Burn-backed `lewm-core::vit` encoder modules with RFC 0002 shape coverage.
- Burn-backed `lewm-core::embedder` action encoder with preserved Conv1d-k1
  smoothing and shape coverage.
- Burn-backed `lewm-core::mlp` projector heads with feature-axis normalization
  and RFC 0002 shape coverage.
- Burn-backed `lewm-core::ada_ln::AdaLNZero` with zero-initialized
  modulation heads and RFC 0002 shape coverage.
- Burn-backed `lewm-core::predictor::{ConditionalBlock, ArPredictor}` with
  AdaLN-zero conditioning, causal attention, and RFC 0002 shape coverage.
- Burn-backed `lewm-core::losses::SigReg` with RFC 0003 constants,
  deterministic sketch sampling, fixed-projection parity entry point, and
  gradient/RNG coverage.
- Burn tensor `lewm-core::losses::prediction_loss` with shape, value, and
  bidirectional-gradient coverage.
- Burn-backed `lewm-core::Jepa` top-level wrapper with encode, predict,
  rollout, criterion, and per-candidate cost contracts.
- RFC 0008 `lewm-core` parity initialization shape audit for the top-level
  JEPA wrapper.
- Python reference-checkpoint parameter-name map for the locked PushT source
  state dict and Burn record conversion preflight.
- `python/convert_reference.py audit` for validating the downloaded PushT
  reference checkpoint keys against the locked conversion map.
- `python/convert_reference.py convert`, `python/verify_conversion.py`, and
  `lewm-reference-record` for emitting a deterministic Safetensors mirror,
  load-checked Burn `NamedMpk` record, and Safetensors-vs-record drift check
  from the locked PushT reference checkpoint.
- `lewm_core::export` Safetensors export helpers for deterministic `Jepa`
  parameter mirrors, including BatchNorm running state and integer counters.

### Changed

- Expanded GitHub issue templates, the pull request traceability checklist, and
  CODEOWNERS mappings for crate-level review routing.
- Bumped the pinned Rust toolchain, CI toolchain, and training image builder to
  Rust `1.89.0` to satisfy the Burn `0.20.1` MSRV.
- Aligned the Burn predictor attention and feed-forward submodules with the
  upstream PushT checkpoint layout, including predictor qkv bias removal and
  affine pre-norm parameters.

### Deprecated

### Removed

### Fixed

### Security

- Added a gitleaks-backed secret scan wrapper and CI gate for
  `TST-0016-SECRETS-001`.
- Documented ADR 0002's date-bounded `cargo audit` waiver for Burn's transitive
  `RUSTSEC-2025-0141` `bincode` dependency.

## [0.1.0] - TBD

Initial public release target.
