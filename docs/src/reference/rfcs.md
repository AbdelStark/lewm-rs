# RFC index

The normative engineering specifications. Every claim made in these
docs is grounded in one of these RFCs (or in source). The RFCs live
under [`specs/rfcs/`](https://github.com/AbdelStark/lewm-rs/tree/main/specs/rfcs)
and are immutable once Accepted; later changes go to follow-up RFCs
or ADRs.

| # | Title | Status | Source |
|---|-------|--------|--------|
| 0001 | Project foundation and build system | Accepted | [0001-project-foundation-and-build-system.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0001-project-foundation-and-build-system.md) |
| 0002 | `lewm-core` — model architecture, modules, forward semantics | Accepted | [0002-core-model-architecture.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0002-core-model-architecture.md) |
| 0003 | SIGReg, prediction loss, gradient contracts | Accepted | [0003-sigreg-and-loss-functions.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0003-sigreg-and-loss-functions.md) |
| 0004 | `lewm-data` — datasets, transforms, batching | Accepted | [0004-data-pipeline.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0004-data-pipeline.md) |
| 0005 | `lewm-train` — training system, optimizer, schedule, checkpoints | Accepted | [0005-training-system.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0005-training-system.md) |
| 0006 | `lewm-plan` — CEM planner and evaluation drivers | Accepted | [0006-planning-and-evaluation.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0006-planning-and-evaluation.md) |
| 0007 | `lewm-infer` — Tract CPU inference, ONNX/NNEF export | Accepted | [0007-tract-inference-and-onnx-export.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0007-tract-inference-and-onnx-export.md) |
| 0008 | Reference parity testing | Accepted | [0008-reference-parity-testing.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0008-reference-parity-testing.md) |
| 0009 | Observability and MLOps | Accepted | [0009-observability-and-mlops.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0009-observability-and-mlops.md) |
| 0010 | Hugging Face Hub integration | Accepted | [0010-huggingface-hub-integration.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0010-huggingface-hub-integration.md) |
| 0011 | CI/CD and release engineering | Accepted | [0011-ci-cd-and-release-engineering.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0011-ci-cd-and-release-engineering.md) |
| 0012 | SO-100 real-robot extension | Accepted | [0012-so100-real-robot-extension.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0012-so100-real-robot-extension.md) |
| 0013 | Determinism and reproducibility | Accepted | [0013-determinism-and-reproducibility.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0013-determinism-and-reproducibility.md) |
| 0014 | Performance engineering | Accepted | [0014-performance-engineering.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0014-performance-engineering.md) |
| 0015 | Documentation, paper writeup, demo Space | Accepted | [0015-documentation-paper-and-demo.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0015-documentation-paper-and-demo.md) |
| 0016 | Security and supply chain | Accepted | [0016-security-and-supply-chain.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0016-security-and-supply-chain.md) |
| 0017 | Error model and failure handling | Accepted | [0017-error-model-and-failure-handling.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0017-error-model-and-failure-handling.md) |
| 0018 | Configuration system | Accepted | [0018-configuration-system.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0018-configuration-system.md) |

## How to read an RFC

Each RFC has a fixed structure:

1. **Front-matter** — `rfc:`, `title:`, `status:`, `version:`,
   `authors:`, `created:`, `updated:`, dependency graph, tracking.
2. **§1 Introduction** — motivation, goals, non-goals, stakeholders.
3. **§2 Conventions** — local notation overrides (rare).
4. **§3 Background** — context, prior art.
5. **§4 Detailed design** — the bulk of the document; sectioned by
   sub-component.
6. **§5+** — additional contracts (e.g. RFC 0003 §5 collapse probes,
   RFC 0005 §9 runbook).
7. **Cross-refs** — citations into other RFCs, the PRD, the glossary.

Every requirement in the design body carries an ID like
`RFC0003-001 [MUST]`. These IDs are referenced from the parity tests
and the source.

## The RFC governance contract

- Drafted in a feature branch, reviewed via PR, accepted by merge.
- Once Accepted, the document is immutable except for typographical
  and link fixes (which update the version's patch number).
- Material changes require either a new RFC superseding the old one
  (with `supersedes:` / `superseded_by:` fields filled in) or an ADR
  decision overriding a specific clause.

## Template

New RFCs start from
[`specs/rfcs/0000-template.md`](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0000-template.md).
