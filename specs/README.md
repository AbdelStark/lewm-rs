# lewm-rs — Specification Set

**Project:** `lewm-rs` — pure-Rust reproduction & extension of LeWorldModel (Maes et al., 2026)
**Spec set version:** 1.0.0 (locked for execution)
**Spec set status:** Accepted
**Date:** 2026-05-12
**Custodian:** Abdel (`@AbdelStark`)
**License:** Specs distributed under CC-BY-4.0; reference implementation under MIT.

---

## 0. Purpose

This directory is the **single source of truth** for everything that must be built, tested, measured, and shipped in `lewm-rs`. The Product Requirements Document (`../PRD.md`) defines **what** and **why**; this spec set defines **how**, **to which tolerance**, and **with which proof**.

Once a section reaches `Status: Accepted`, the implementation **MUST** conform. Deviations are only allowed via a published ADR (`adr/`) that supersedes the relevant clause.

The spec set is intentionally redundant with the PRD on the small number of cross-cutting decisions where reading either document alone must be sufficient to act correctly. Where the spec set and the PRD disagree on a *contract* (numerical tolerance, API shape, file format), the **spec set wins** and the PRD is updated. Where they disagree on *intent or motivation*, the **PRD wins**.

---

## 1. Document inventory

### 1.1 Master documents

| Path | Status | Owner | Summary |
|------|--------|-------|---------|
| [`TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) | Accepted | Abdel | Cross-cutting architecture, contracts, conformance criteria. Reads as a complete system spec on its own. |
| [`glossary.md`](glossary.md) | Accepted | Abdel | Authoritative definitions for every domain term, acronym, and tolerance constant. |
| [`traceability-matrix.md`](traceability-matrix.md) | Accepted | Abdel | PRD requirement ↔ RFC clause ↔ Test ID mapping. CI consumes this. |

### 1.2 Requests for Comments (RFCs)

RFCs are numbered in execution-friendly order: read in sequence to learn the system bottom-up.

| # | Title | Status | Owner |
|---|-------|--------|-------|
| [0001](rfcs/0001-project-foundation-and-build-system.md) | Project foundation, workspace layout, build system | Accepted | Abdel |
| [0002](rfcs/0002-core-model-architecture.md) | `lewm-core` — ViT, ARPredictor, Embedder, MLP, JEPA wrapper | Accepted | Abdel |
| [0003](rfcs/0003-sigreg-and-loss-functions.md) | SIGReg loss, prediction loss, gradient contracts | Accepted | Abdel |
| [0004](rfcs/0004-data-pipeline.md) | `lewm-data` — PushT HDF5, LeRobot v2.1, batching | Accepted | Abdel |
| [0005](rfcs/0005-training-system.md) | `lewm-train` — trainer, optimizer, schedule, checkpoints | Accepted | Abdel |
| [0006](rfcs/0006-planning-and-evaluation.md) | `lewm-plan` — CEM planner, eval drivers | Accepted | Abdel |
| [0007](rfcs/0007-tract-inference-and-onnx-export.md) | `lewm-infer` — ONNX export, Tract runner, fallbacks | Accepted | Abdel |
| [0008](rfcs/0008-reference-parity-testing.md) | Weight import & parity test harness | Accepted | Abdel |
| [0009](rfcs/0009-observability-and-mlops.md) | Metrics, traces, logs, collapse detection, Trackio | Accepted | Abdel |
| [0010](rfcs/0010-huggingface-hub-integration.md) | `lewm-hub` — uploads, model cards, dataset mirrors | Accepted | Abdel |
| [0011](rfcs/0011-ci-cd-and-release-engineering.md) | CI matrix, release pipeline, container images | Accepted | Abdel |
| [0012](rfcs/0012-so100-real-robot-extension.md) | SO-100 dataset prep, warm-start, eval | Accepted | Abdel |
| [0013](rfcs/0013-determinism-and-reproducibility.md) | RNG architecture, bitwise contracts | Accepted | Abdel |
| [0014](rfcs/0014-performance-engineering.md) | Throughput targets, benchmarks, profiling | Accepted | Abdel |
| [0015](rfcs/0015-documentation-paper-and-demo.md) | Documentation system, paper, demo Space | Accepted | Abdel |
| [0016](rfcs/0016-security-and-supply-chain.md) | Supply chain, secrets, threat model | Accepted | Abdel |
| [0017](rfcs/0017-error-model-and-failure-handling.md) | Error taxonomy, recovery, panic policy | Accepted | Abdel |
| [0018](rfcs/0018-configuration-system.md) | Config layering, schema, validation | Accepted | Abdel |

### 1.3 Architectural Decision Records (ADRs)

ADRs capture **one** decision each, made during execution, with the context, the options considered, the choice, and the consequences. They are immutable once accepted; superseding decisions are new ADRs.

See [`adr/README.md`](adr/README.md) for the process and [`adr/0000-template.md`](adr/0000-template.md) for the template.

### 1.4 Diagrams

See [`diagrams/`](diagrams/) for system diagrams. All diagrams **MUST** have a textual ASCII or Mermaid source under version control; binary renders are convenience artifacts only.

---

## 2. Conventions used throughout this spec set

### 2.1 RFC 2119 normative keywords

The keywords **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** are to be interpreted as described in [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119) and [RFC 8174](https://datatracker.ietf.org/doc/html/rfc8174). They appear in **boldface** only when they carry their normative force; lowercase use is non-normative.

### 2.2 Requirement IDs

Every normative clause carries an identifier of the form `<scope>-<NNN>`:

- `FR-NNN` — functional requirement (master spec §2)
- `NFR-NNN` — non-functional requirement (master spec §3)
- `RFC<XXXX>-<NNN>` — requirement local to RFC `<XXXX>`
- `INV-NNN` — invariant that must hold at all times
- `TST-<XXXX>-<NNN>` — test specification

IDs are stable. When a requirement is dropped, its ID is **retired**, never reused.

### 2.3 Tolerance vocabulary

Numerical tolerances are stated as absolute (`abs`) or relative (`rel`) with the comparison operator and the metric:

```
TOL: |y_rust - y_torch|_∞       ≤ 1e-4   (encoder CLS, F32)
TOL: |y_rust - y_torch|_2 / |y_torch|_2  ≤ 1e-3   (predictor output, F32)
```

`|·|_∞` denotes infinity norm (max absolute), `|·|_2` denotes L2 norm. Defaults are listed in [`glossary.md`](glossary.md) §4.

### 2.4 Status lifecycle

```
   Draft  ───▶  Proposed  ───▶  Accepted  ───▶  Implemented  ───▶  Superseded
                                                                       │
                                                                       ▼
                                                                   Retired
```

- **Draft** — author writing; not for review.
- **Proposed** — under review; comments welcome.
- **Accepted** — implementation **MUST** conform; CI gates may rely on it.
- **Implemented** — the code actually exists and passes the conformance tests.
- **Superseded** — a later spec replaces this one; transitive references must be updated.
- **Retired** — no longer in force; preserved for archaeology.

### 2.5 File format conventions

- Markdown is GitHub Flavored Markdown (CommonMark + tables + fenced code).
- Mermaid diagrams use `mermaid` fences.
- Code samples are **runnable** when so labeled (`# runnable`); otherwise they are illustrative skeletons.
- All line endings are LF.
- All files are UTF-8 with no BOM.
- Maximum line length in narrative text is soft (no enforced wrap). Code examples follow each language's idiomatic line length (Rust 100, Python 99).

### 2.6 Versioning

The spec set follows [SemVer 2.0.0](https://semver.org/) at the set level (this `README.md`'s `Spec set version`):

- **Major** — a contract change that requires a code change in already-conforming implementations.
- **Minor** — a new RFC, or a backwards-compatible expansion of an existing one (e.g., tighter tolerance, additional optional metric).
- **Patch** — typos, clarifications, examples, non-normative additions.

Individual RFCs carry their own version in their frontmatter and bump independently.

---

## 3. Reading guides

### 3.1 If you are about to write code

1. Read [`TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) §1–§5 (system overview, requirements, architecture).
2. Read the RFC for the crate you are touching (one of 0002–0007, 0009, 0010).
3. Read [RFC 0013 (Determinism)](rfcs/0013-determinism-and-reproducibility.md) and [RFC 0017 (Errors)](rfcs/0017-error-model-and-failure-handling.md). These are cross-cutting and easy to get wrong.
4. Run the test suite locally before writing a single line: `cargo test --workspace`.

### 3.2 If you are running the ml-intern agent

1. Read [RFC 0001 §7 (Agent leash)](rfcs/0001-project-foundation-and-build-system.md) and [RFC 0016 (Security)](rfcs/0016-security-and-supply-chain.md).
2. Read PRD §6.6 (allowed/forbidden list).
3. Read [`adr/`](adr/) in full — these are the binding decisions you must not violate.

### 3.3 If you are reviewing a PR

1. Open [`traceability-matrix.md`](traceability-matrix.md). Verify the PR's tests trace to the modified requirements.
2. Check the touched RFC: is its `Status` `Accepted` and unchanged by the PR? If the PR changes a spec clause, it **MUST** also bump the RFC version and update this `README`.
3. Check [RFC 0011 (CI)](rfcs/0011-ci-cd-and-release-engineering.md) gates have all passed.

### 3.4 If you are reproducing the result

1. Read PRD §10 (acceptance criteria) and [`TECHNICAL_SPECIFICATION.md`](TECHNICAL_SPECIFICATION.md) §12 (conformance).
2. Read [RFC 0013 (Determinism)](rfcs/0013-determinism-and-reproducibility.md) to understand which numerical drift is expected and which is a bug.
3. Follow the runbooks in [RFC 0005 §9](rfcs/0005-training-system.md) (training) and [RFC 0007 §7](rfcs/0007-tract-inference-and-onnx-export.md) (inference).

---

## 4. How to propose a change

1. Open a draft RFC (new number, status `Draft`) or an ADR if the change is a single decision.
2. File a PR titled `spec: RFC-XXXX — <topic>` referencing the issue.
3. Tag at least one reviewer with `code-owner` over the touched crates.
4. CI runs the spec-lint job (`scripts/check_specs.py`) which validates frontmatter, link integrity, requirement-ID uniqueness, and traceability completeness.
5. Once reviewed, status moves to `Proposed`. The author advertises the proposal in the project channel for 72 hours minimum.
6. On approval (LGTM + 24h cooling period for non-trivial changes), status moves to `Accepted` and the spec set version is bumped per §2.6.

Use [`adr/0000-template.md`](adr/0000-template.md) for the first real ADR; worked examples
will live beside accepted ADRs once they exist.

---

## 5. Out of scope for this spec set

The following are out of scope and **MUST NOT** be relied upon by conforming implementations:

- Behaviour on operating systems other than Linux x86_64 and macOS arm64 (Tract inference path is portable; trainer is not).
- Behaviour with Burn versions other than the pinned version in `Cargo.toml`.
- Behaviour with CUDA driver versions outside the matrix in [RFC 0011 §4](rfcs/0011-ci-cd-and-release-engineering.md).
- Models, datasets, or evaluation protocols not enumerated in this set.

The PRD §2 "Non-goals" list is canonical and binding for scope discussions.

---

## 6. Citation

If you build on this spec set, please cite both the implementation and the upstream paper:

```
@software{lewm_rs_2026,
  author = {Abdel},
  title  = {lewm-rs: A Pure-Rust Reproduction of LeWorldModel},
  year   = {2026},
  url    = {https://github.com/AbdelStark/lewm-rs}
}

@article{maes_lelidec2026lewm,
  title  = {LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author = {Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal = {arXiv preprint},
  year   = {2026}
}
```

---

## 7. Contact

- Author: Abdel — `abdel@starkware.co`
- Issue tracker: `https://github.com/AbdelStark/lewm-rs/issues`
- Security disclosures: see [RFC 0016 §10](rfcs/0016-security-and-supply-chain.md).

---

*End of `specs/README.md`. The substantive content is in the linked documents.*
