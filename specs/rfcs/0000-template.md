<!--
This is the canonical RFC template for the `lewm-rs` spec set.
Copy this file to `specs/rfcs/NNNN-short-kebab-title.md` and fill in each section.
Sections marked OPTIONAL may be omitted with a one-line justification in their place.
Sections marked REQUIRED must be present even if the content is "N/A — <reason>".
-->

---
rfc:   "0000"
title: "RFC template"
status: Draft           # Draft | Proposed | Accepted | Implemented | Superseded | Retired
version: 0.1.0           # SemVer; bump per `specs/README.md` §2.6
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []           # RFC numbers, e.g. ["0007"]
superseded_by: null
tracks_prd: ["§<n>"]     # PRD sections this RFC realizes
depends_on: []           # other RFC numbers
related: []
---

# RFC 0000 — RFC template

> **Status:** Draft · **Version:** 0.1.0 · **Authors:** Abdel · **Reviewers:** —
>
> *One-paragraph abstract. State what this RFC adds, changes, or removes, in plain prose. No more than 5 sentences.*

---

## 1. Introduction

### 1.1 Motivation [REQUIRED]

Why does this RFC exist? What hurts today, what will hurt tomorrow if we do nothing, what concrete demand does this satisfy?

### 1.2 Goals [REQUIRED]

Numbered, observable, falsifiable. Each goal becomes a measurable outcome in §11 Acceptance.

1. Goal 1 ...
2. Goal 2 ...

### 1.3 Non-goals [REQUIRED]

What this RFC explicitly does **not** cover, to prevent scope drift. Each line **MUST** justify the exclusion.

### 1.4 Stakeholders [OPTIONAL]

Roles that read or are affected: implementer, reviewer, operator, downstream RFCs, end user.

---

## 2. Conventions and definitions

This RFC follows the conventions of [`specs/README.md`](../README.md) §2 and the glossary in [`specs/glossary.md`](../glossary.md). Terms introduced here that are not yet in the glossary **MUST** be added to the glossary in the same PR.

### 2.1 RFC-local definitions [OPTIONAL]

Terms used only in this RFC; if any later RFC also uses them, promote to the glossary.

### 2.2 Normative language

The keywords MUST / SHOULD / MAY apply per RFC 2119 / RFC 8174.

---

## 3. Background [OPTIONAL but recommended]

What does the reader need to know — about the upstream codebase, the literature, prior decisions, current state of the project — to evaluate this RFC?

Cite sources with permanent URLs (DOI, commit SHAs, paper arXiv IDs). Do **not** cite branch tips.

---

## 4. Detailed design [REQUIRED]

This is the bulk of the RFC. It **MUST** be sufficient for an independent implementer to write the code without speaking to the author. Recommended subsections:

### 4.1 Overview

Two-paragraph summary of the design.

### 4.2 Module structure and public API

Rust type signatures with documentation comments. If a type is publicly exported, the full `pub` API surface goes here. Implementation details may be hidden behind `// elided` markers as long as semantics are described elsewhere in this RFC.

```rust
/// Brief.
///
/// # Invariants
/// - ...
///
/// # Errors
/// - returns `XxxError::Yyy` when ...
pub struct Foo<B: Backend> { /* ... */ }

impl<B: Backend> Foo<B> {
    /// Constructs a new `Foo`. Idempotent given the same `device`.
    pub fn new(device: &B::Device) -> Self { /* ... */ }

    /// ...
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> { /* ... */ }
}
```

### 4.3 Algorithms

Pseudocode and the **exact** numerical recipe for anything non-trivial. Use the symbol conventions of `glossary.md` §6.

### 4.4 Data formats and schemas

For anything that crosses a process boundary or is persisted (files, sockets, env-vars):

- Format identifier (e.g., "TOML 0.5", "MessagePack via `rmp-serde 1.x`")
- Schema with field types, units, ranges, default values
- Backward-compat policy

### 4.5 Configuration

Which fields belong to which config file or env-var. Validation rules.

### 4.6 State and lifecycle

Diagrams (Mermaid or ASCII) of state machines, lifecycles, and protocols.

### 4.7 Concurrency model

Threads, channels, shared state, locks, async runtime usage if any.

### 4.8 Error model

Which errors the module produces, when, and how the caller should react.

### 4.9 Memory and resource budget

Memory ceilings, file-handle counts, network connection counts, GPU memory footprint.

---

## 5. Reference implementation [REQUIRED]

A pointer to the canonical implementation **after** the RFC is implemented:

- Crate path
- Entry point (module / function)
- Test path

While the RFC is Draft/Proposed, this section sketches the skeleton.

---

## 6. Testing strategy [REQUIRED]

### 6.1 Test inventory

Table of every test that is part of the conformance suite for this RFC.

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-NNNN-001 | `name_of_test` | unit/integration/e2e/property/bench | one-line summary |

### 6.2 Test fixtures

Datasets, golden outputs, reference dumps. Where they live, who blesses them, how they are regenerated.

### 6.3 Property-based tests

Properties (in the QuickCheck / `proptest` sense) that **MUST** hold.

### 6.4 Negative tests

Inputs that **MUST** fail with the specified error.

---

## 7. Operational considerations [REQUIRED]

How the code behaves in production: logging, metrics, traces, runbooks, alerts, rollback.

### 7.1 Observability

Metrics emitted, log lines emitted, span names.

### 7.2 Runbook

What an operator does when something breaks.

### 7.3 Capacity planning

Compute, memory, bandwidth needs at each tier defined in PRD §6.5.

---

## 8. Performance considerations [REQUIRED]

Target throughput, latency, memory. Benchmarks committed under `benches/`. See [RFC 0014](0014-performance-engineering.md).

---

## 9. Security considerations [REQUIRED]

Threats this RFC introduces or addresses. Cross-link to [RFC 0016](0016-security-and-supply-chain.md). Do **not** write "N/A" — every RFC at least clarifies its trust boundary.

---

## 10. Alternatives considered [REQUIRED]

For each alternative: what it was, why it was rejected, what would change our mind.

- **A1** — ... (rejected because ...).
- **A2** — ... (rejected because ...).

---

## 11. Acceptance criteria [REQUIRED]

A bulleted, machine-checkable list. Each criterion maps to one or more tests in §6.1 or a measurable artifact elsewhere in the repo.

- [ ] Criterion 1 — `TST-NNNN-001` green in CI on `linux-x86_64`.
- [ ] ...

---

## 12. Risks [REQUIRED]

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| R-1 | ... | L/M/H | L/M/H | ... |

---

## 13. Open questions [OPTIONAL]

Numbered questions to resolve before the RFC moves to Implemented. Each resolution becomes an ADR or is folded back into this RFC.

---

## 14. Appendix [OPTIONAL]

Long-form derivations, reference dumps, worked examples, full configuration examples.

---

## 15. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 0.1.0 | 2026-05-12 | Abdel | Initial draft. |

---

*End of RFC 0000.*
