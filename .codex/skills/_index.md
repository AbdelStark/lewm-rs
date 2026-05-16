# Skill Registry — lewm-rs

Last updated: 2026-05-16
Canonical location: `.codex/skills/`. Compatibility symlinks: `.claude/skills`, `.agents/skills`.

Load a skill by reading its file when entering the domain. Skills are project-specific — they reference real files, real RFC clauses, and real lint rules in this repository.

## Core skills

| Skill                          | File                          | Triggers                                                                 | Priority |
|--------------------------------|-------------------------------|--------------------------------------------------------------------------|----------|
| Crate layering                 | `crate-layering.md`           | Adding inter-crate dep, `scripts/check_layers.py` failure                | Core     |
| Reference parity testing       | `parity-testing.md`           | Edits to encoder/predictor/AdaLN/MLP/embedder/losses/jepa/tensor_ops     | Core     |
| Burn `Module` patterns         | `burn-modules.md`             | Implementing or modifying a `burn::module::Module`                       | Core     |
| Determinism & RNG              | `determinism-rng.md`          | RNG usage, seeds, `check_nondet.py` failure, checkpoint/resume           | Core     |
| HF Jobs cost discipline        | `hf-jobs-cost.md`             | Launching HF Jobs, editing `jobs/*.yaml`, updating `reports/cost.md`     | Core     |
| RFC & ADR process              | `rfc-adr-process.md`          | Editing/adding RFC/ADR; behaviour contradicts an Accepted clause          | Core     |
| Local quality gate             | `quality-gate.md`             | `make check` / `make accept` failures; reproducing CI locally            | Core     |

## Extend skills

| Skill                          | File                          | Triggers                                                                 | Priority |
|--------------------------------|-------------------------------|--------------------------------------------------------------------------|----------|
| Python helpers                 | `python-helpers.md`           | Anything under `python/` or `scripts/`; Ruff failure                     | Extend   |
| Releases & Hub artifacts       | `release-and-artifacts.md`    | Uploads to `abdelstark/lewm-rs-*`, model cards, release tags             | Extend   |

## Missing skills (recommendations)

These are candidates worth scaffolding when their usage frequency increases. Each one is currently absorbed by `CLAUDE.md` plus an RFC; promote to a dedicated skill if churn justifies it.

- [ ] `onnx-export.md` — Tract opset / fixed-batch constraints + `python/export_onnx.py` patterns. Covered today by RFC 0007 + skill `release-and-artifacts.md`.
- [ ] `cem-planning.md` — CEM planner tuning, episode-time profiling. Covered today by RFC 0006 + `crates/lewm-plan/src/`.
- [ ] `otel-telemetry.md` — Local OTel stack, NVTX/NVML feature gating. Covered today by RFC 0009 + `infra/otel/README.md`.
- [ ] `lerobot-dataset.md` — LeRobot v2.1 + SO-100 ingestion. Covered today by RFC 0004 + `python/decode_so100_to_h5.py`.

## Skill format

See any existing skill for the canonical structure. Skill files MUST:

1. Start with YAML frontmatter (`name`, `description`).
2. Use XML semantic sections (`<purpose>`, `<context>`, `<procedure>`, `<patterns>`, `<examples>`, `<troubleshooting>`, `<references>`).
3. Reference real file paths and real RFC clauses in this repo.
4. Stay under ~2000 tokens. Split if larger.
