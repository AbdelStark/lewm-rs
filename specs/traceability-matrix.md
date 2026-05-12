# Traceability Matrix — `lewm-rs`

**Status:** Accepted · **Version:** 1.0.0 · **Last updated:** 2026-05-12 · **CI-checked:** yes

This matrix is the single authoritative crosswalk between **PRD acceptance criteria**, **functional/non-functional requirements**, **RFCs**, and **test IDs**.

It is machine-checked by `scripts/check_specs.py --check-traceability` in the `specs.yml` CI workflow (see [RFC 0011 §4.3](rfcs/0011-ci-cd-and-release-engineering.md)). The script enforces:

1. Every PRD §10 box appears in the **Acceptance** table below.
2. Every FR-NNN appears in the **Requirements** table.
3. Every test ID appearing here also appears in the corresponding RFC's §6.1.
4. No retired ID is reused.

When you add or modify a requirement, RFC, or test, you **MUST** update this file in the same PR. CI will block otherwise.

---

## 1. Acceptance table (PRD §10 → spec set)

| PRD §10 line item | Owning FR(s) | Owning RFC(s) | Conformance test ID(s) |
|---|---|---|---|
| `cargo test --workspace` green on Linux x86-64 in CI | NFR-001, NFR-003 | 0001, 0011 | `TST-0011-CI-001` |
| `parity_encoder` ≤ 1e-4 vs `quentinll/lewm-pusht` | FR-001, FR-002, FR-021 | 0002, 0008 | `TST-0008-ENC-001`, `TST-0008-ENC-002` |
| `parity_predictor` ≤ 1e-4 | FR-003, FR-004, FR-022 | 0002, 0008 | `TST-0008-PRED-001`, `TST-0008-PRED-002` |
| `parity_sigreg` ≤ 1e-3 (same seed) | FR-008, FR-023 | 0003, 0008 | `TST-0008-SR-001`, `TST-0008-SR-002` |
| PushT model published — 4 checkpoints, card, report | FR-045, FR-070, FR-073 | 0005, 0009, 0010 | `TST-0010-ART-PUSHT-001`, `TST-0010-ART-PUSHT-002`, `TST-0010-ART-PUSHT-003`, `TST-0010-ART-PUSHT-004` |
| PushT planning success rate ≥ 87 % | FR-006, FR-050, FR-051 | 0006 | `TST-0006-EVAL-PUSHT-001` |
| SO-100 model published — 4 checkpoints, card, eval report | FR-045, FR-070, FR-073, FR-052 | 0005, 0010, 0012 | `TST-0010-ART-SO100-001..004` |
| SO-100 latent rollout Spearman ≥ 0.6 OR null documented | FR-052, FR-053 | 0006, 0012 | `TST-0012-EVAL-001`, `TST-0012-EVAL-002` |
| Tract CPU inference ≤ 1.0 s laptop / ≤ 0.3 s CPU XL | FR-060, FR-061, FR-062, NFR-013 | 0007, 0014 | `TST-0007-BENCH-001`, `TST-0007-BENCH-002` |
| Demo Space live and reachable | FR-070 | 0007, 0015 | `TST-0015-DEMO-001` |
| Cost ledger committed, total spend ≤ 200 USD | NFR-060, FR-074 | 0010 | `TST-0010-COST-001` |
| Writeup published, blog post live | FR-073 | 0015 | `TST-0015-PAPER-001`, `TST-0015-BLOG-001` |
| Reproducibility — third party can rebuild on own HF account | NFR-050, NFR-051 | 0011, 0013 | `TST-0011-REPRO-001`, `TST-0013-DET-001` |

---

## 2. Requirements table (FR/NFR → realization)

### 2.1 Functional requirements

| Req ID | Statement (short) | RFC | Tests |
|---|---|---|---|
| FR-001 | ViT-Small encoder | 0002 | `TST-0002-ENC-001` |
| FR-002 | CLS extraction | 0002 | `TST-0002-ENC-002` |
| FR-003 | Action `Embedder` | 0002 | `TST-0002-EMB-001` |
| FR-004 | `ArPredictor` AdaLN-zero | 0002 | `TST-0002-PRED-001..003` |
| FR-005 | Projector & pred_proj MLPs | 0002 | `TST-0002-MLP-001` |
| FR-006 | `Jepa` wrapper API | 0002 | `TST-0002-JEPA-001..003` |
| FR-007 | Prediction MSE | 0003 | `TST-0003-PRED-001` |
| FR-008 | SIGReg | 0003 | `TST-0003-SR-001..006` |
| FR-009 | Total loss combination | 0003 | `TST-0003-TOTAL-001` |
| FR-010 | Rollout sliding window | 0002 | `TST-0002-RO-001..003` |
| FR-011 | Cost function | 0002 | `TST-0002-COST-001` |
| FR-020 | Reference weight import | 0008 | `TST-0008-IMP-001..004` |
| FR-021 | Encoder parity ≤ 1e-4 | 0008 | `TST-0008-ENC-001..002` |
| FR-022 | Predictor parity ≤ 1e-4 | 0008 | `TST-0008-PRED-001..002` |
| FR-023 | SIGReg parity | 0008 | `TST-0008-SR-001..002` |
| FR-030 | PushT HDF5 loader | 0004 | `TST-0004-PUSHT-001..004` |
| FR-031 | SO-100 loader | 0004, 0012 | `TST-0004-SO100-001..004` |
| FR-032 | Image preprocess | 0004 | `TST-0004-XFORM-001..003` |
| FR-033 | Action normalize | 0004 | `TST-0004-XFORM-004..005` |
| FR-034 | Window sample | 0004 | `TST-0004-WIN-001..003` |
| FR-040 | Train binary subcommands | 0005 | `TST-0005-CLI-001..005` |
| FR-041 | AdamW + cosine schedule | 0005 | `TST-0005-OPT-001..003` |
| FR-042 | Grad accumulation | 0005 | `TST-0005-ACC-001` |
| FR-043 | Grad clipping | 0005 | `TST-0005-CLIP-001` |
| FR-044 | Mixed precision | 0005 | `TST-0005-BF16-001..002` |
| FR-045 | Checkpoints | 0005 | `TST-0005-CKPT-001..004` |
| FR-046 | Resume | 0005 | `TST-0005-RESUME-001..002` |
| FR-047 | Per-epoch parity probe | 0005 | `TST-0005-PROBE-001` |
| FR-050 | CEM | 0006 | `TST-0006-CEM-001..003` |
| FR-051 | PushT success rate eval | 0006 | `TST-0006-EVAL-PUSHT-001` |
| FR-052 | Latent rollout metrics | 0006 | `TST-0006-EVAL-LAT-001..002` |
| FR-053 | Warm-start delta | 0006 | `TST-0006-EVAL-DELTA-001` |
| FR-060 | ONNX export | 0007 | `TST-0007-EXPORT-001..002` |
| FR-061 | Tract runner | 0007 | `TST-0007-RUN-001..003` |
| FR-062 | Inference latency | 0007 | `TST-0007-BENCH-001..002` |
| FR-070 | Trackio metrics | 0009 | `TST-0009-TRACK-001` |
| FR-071 | Tensorboard mirror | 0009 | `TST-0009-TB-001` |
| FR-072 | OTLP traces | 0009 | `TST-0009-OTLP-001` |
| FR-073 | Model card generation | 0010 | `TST-0010-CARD-001..002` |
| FR-074 | Cost ledger | 0010 | `TST-0010-COST-001` |
| FR-080 | ml-intern tier restrictions | 0016 | `TST-0016-INTERN-001` |
| FR-081 | ml-intern session audit | 0016 | `TST-0016-INTERN-002` |

### 2.2 Non-functional requirements

| Req ID | Statement | RFC | Tests |
|---|---|---|---|
| NFR-001 | Parity-tested correctness | 0008 | `TST-0008-*` |
| NFR-002 | Determinism contract | 0013 | `TST-0013-DET-001..004` |
| NFR-003 | Static type safety | 0011, 0017 | `TST-0011-LINT-001` |
| NFR-010 | PushT throughput | 0014 | `TST-0014-THRU-PUSHT-001` |
| NFR-011 | SO-100 throughput | 0014 | `TST-0014-THRU-SO100-001` |
| NFR-012 | GPU memory ≤ 20 GB | 0014 | `TST-0014-MEM-001` |
| NFR-013 | Inference latency laptop | 0007, 0014 | `TST-0007-BENCH-001` |
| NFR-014 | Cold-start ≤ 3 s | 0007 | `TST-0007-BENCH-COLD-001` |
| NFR-020 | Crash-resume | 0005, 0013 | `TST-0005-RESUME-001..002` |
| NFR-021 | Idempotent uploads | 0010 | `TST-0010-IDEMP-001` |
| NFR-022 | No resource leaks | 0011 | `TST-0011-LEAK-001` |
| NFR-030 | One-command quickstart | 0001 | `TST-0001-QS-001` |
| NFR-031 | Error message style | 0017 | `TST-0017-MSG-001..002` |
| NFR-032 | `cargo doc` clean | 0011 | `TST-0011-DOC-001` |
| NFR-040 | License audit | 0016 | `TST-0016-DENY-001` |
| NFR-041 | Attribution present | 0010 | `TST-0010-ATTR-001` |
| NFR-050 | Reproducible build | 0011 | `TST-0011-REPRO-001` |
| NFR-051 | Reproducible training | 0013 | `TST-0013-REPRO-001` |
| NFR-060 | Cost ≤ 200 USD | 0010 | `TST-0010-COST-001` |

---

## 3. Invariants table

| INV-ID | Statement | Where enforced |
|---|---|---|
| INV-001 | `lewm-core` has no workspace deps | RFC 0001 §4.4; CI `check_layers.py` |
| INV-002 | `lewm-data` has no orchestration deps | RFC 0001 §4.4; CI |
| INV-003 | `lewm-infer` has no GPU/autodiff deps | RFC 0001 §4.4; CI |
| INV-004 | No PyO3 in v1 | RFC 0001 §4.4; ADR required to lift |
| INV-005 | SIGReg internal F32 regardless of outer precision | RFC 0003 §5 |
| INV-006 | AdaLN final linear init = 0 | RFC 0002 §4.6 |
| INV-007 | RNG seed tree as in RFC 0013 §4 | RFC 0013 |
| INV-008 | `step_{N}.{mpk,safetensors,json,parity.json}` always co-located | RFC 0005 §6.1 |
| INV-009 | No `unwrap` outside tests | RFC 0017 §3 |
| INV-010 | Config `deny_unknown_fields` | RFC 0018 §5 |
| INV-011 | Every metric flows to Trackio AND Tensorboard | RFC 0009 §4 |
| INV-012 | Every HF Jobs launch has `--timeout` | RFC 0011 §3; RFC 0016 §6 |

---

## 4. Retired IDs

*(empty; this section grows as the project evolves)*

When an ID is retired, list it here with the rationale and the PR that retired it. The number **MUST NOT** be reused.

---

## 5. Coverage gaps

*(empty if all green; CI fails if non-empty)*

If a coverage gap is identified during a review, add a row here with the gap description and an `OWNER` field. The release pipeline blocks until the gap is closed or explicitly waived by a published ADR.

---

*End of `traceability-matrix.md`.*
