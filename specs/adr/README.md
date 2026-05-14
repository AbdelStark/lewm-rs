# Architectural Decision Records (ADRs) — `lewm-rs`

**Status:** Accepted · **Version:** 1.0.0 · **Last updated:** 2026-05-12

ADRs capture **one decision each** taken during the execution of `lewm-rs`. Each ADR is **immutable** once accepted; a later decision that changes course is a **new** ADR that **supersedes** the prior one. The history is the value.

---

## 1. What is an ADR

The spec set (RFCs, master technical spec) describes the system as it ought to be. ADRs describe the *moments* where two or more reasonable paths existed and we picked one. They are short, focused, and dated.

An ADR is **not** an RFC: an ADR records a choice, an RFC describes a system. An ADR is **not** a discussion thread: discussion happens on the PR; the ADR is the conclusion.

This style is adapted from Michael Nygard's classic ADR template, with light extensions for status lifecycle and traceability.

---

## 2. When to write an ADR

Write an ADR when **all** of the following hold:

1. The decision will be hard to reverse.
2. A reasonable future contributor could ask "why did we pick X over Y?" and a one-line code comment will not satisfy them.
3. The decision is **not** already pinned in a Accepted RFC.

Examples of decisions that warrant an ADR:

- Choosing ONNX export over a Burn-native Tract loader.
- Picking AdamW over Lion for SO-100.
- Setting `lambda_sigreg = 1.0` rather than the swept optimum.
- Forking a dependency.
- Disabling a CI gate temporarily for a known issue.

Examples that do **not** warrant an ADR:

- Renaming a variable.
- Choosing between two equivalent crate versions.
- Tweaking a log format.
- Anything already settled by an Accepted RFC. Update the RFC instead.

---

## 3. Numbering and naming

- ADRs are numbered consecutively from `0001`. `0000` is reserved for the template.
- Filenames follow `NNNN-short-kebab-title.md`.
- Numbers are **never** reused. A retired ADR keeps its slot.
- The title in the filename is the *issue*, not the *answer*. Good: `0007-action-encoder-conv1d-vs-linear.md`. Bad: `0007-we-use-conv1d.md`.

---

## 4. Lifecycle

```
Proposed → Accepted → Implemented → Superseded
                                    ↘
                                     Retired (if obsolete)
```

- **Proposed** — open PR; under review.
- **Accepted** — approved; binding.
- **Implemented** — the corresponding code change is merged.
- **Superseded** — replaced by a later ADR; the later one cites this one in its frontmatter.
- **Retired** — the decision is no longer relevant (e.g., the feature was removed); not actively replaced.

An ADR moves to **Implemented** automatically when the cited PR merges into `main`. The implementor updates the frontmatter in the *same* PR that implements the change.

---

## 5. Authoring process

1. Copy [`0000-template.md`](0000-template.md) to `NNNN-short-kebab-title.md`.
2. Fill in frontmatter and sections. Keep prose under ~600 words; ADRs reward brevity.
3. Open a PR titled `adr: NNNN — <short title>`. Tag the touched-area code owner and at least one peer reviewer.
4. CI runs `scripts/check_specs.py --check-adr` to validate frontmatter and link integrity.
5. After 24h (or 48h for cross-RFC consequences) and one LGTM, the author may mark `Accepted` and merge.
6. If the ADR's decision crosses into RFC territory (changes a public contract), the same PR **MUST** update the affected RFCs and bump their versions per [`specs/README.md`](../README.md) §2.6.

---

## 6. Cross-referencing

- Reference an ADR with `ADR-NNNN`.
- Reference an RFC with `RFC-XXXX`.
- Backlinks: when an ADR is superseded, its `superseded_by` frontmatter field gets the new ADR's number, and the new ADR's `supersedes` lists the old one.

---

## 7. Index of ADRs

*(This index is populated as ADRs are accepted. CI fails if an `0000`-numbered ADR file other than the template is committed, or if any accepted ADR is missing from this index.)*

| # | Title | Status | Tracks |
|---|-------|--------|--------|
| [0001](0001-pusht-reference-architecture.md) | PushT reference architecture source of truth | Implemented | RFC 0002, RFC 0008, RFC 0018 |

---

## 8. Reading guide

- Read ADRs in chronological order to understand the project's evolution.
- Read ADRs by topic (filter by `tracks_rfc` frontmatter) to understand a specific subsystem's history.
- When an ADR is `Superseded`, read the successor first, then jump back if the rationale matters.

---

*End of `adr/README.md`.*
