---
name: rfc-adr-process
description: Spec-first governance protocol. Activate when adding/modifying any file under `specs/`, when a code change appears to contradict an Accepted RFC clause, when adding a new ADR, or when `python3 scripts/check_specs.py` fails. RFCs are the contract; ADRs are the immutable decision log. The PRD is the product intent.
prerequisites: None — this skill describes the process
---

# RFC & ADR Process

<purpose>
`lewm-rs` is spec-first. Anything that changes behavior, contracts, dependency policy, artifact policy, or release semantics is governed by `PRD.md` + `specs/`. This skill is the map: how to read, how to amend, and how to add new specs without breaking traceability.
</purpose>

<context>
- **`PRD.md`** (root): product intent. Read but rarely edit. Defines *what* and *why*.
- **`specs/`** (binding contracts):
  - `TECHNICAL_SPECIFICATION.md` — cross-cutting architecture and conformance.
  - `glossary.md` — every domain term, every tolerance constant. Use the terms verbatim.
  - `traceability-matrix.md` — PRD requirement ↔ RFC clause ↔ Test ID. **CI consumes this.**
  - `rfcs/` — numbered RFCs 0001–0018, all currently Accepted v1.x. **Once Accepted, the implementation MUST conform.**
  - `adr/` — numbered decisions. **Immutable once Accepted**; supersede with a new ADR.
- Normative language follows RFC 2119/8174: **MUST**, **MUST NOT**, **SHOULD**, **MAY** — bold when normative, lowercase otherwise.
- Tolerance vocab: `|y_rust − y_torch|_∞ ≤ 1e-4` style; default constants in `glossary.md` §4.
- Requirement IDs: `FR-NNN`, `NFR-NNN`, `RFC<XXXX>-<NNN>`, `INV-NNN`, `TST-<XXXX>-<NNN>`. IDs are stable; retired, never reused.
- Spec set status: `Draft` → `Review` → `Accepted` → `Superseded`. Only `Accepted` is binding.
- CI: `.github/workflows/specs.yml` runs `scripts/check_specs.py --check-frontmatter --check-links --check-traceability` on every PR touching `PRD.md` or `specs/`. External-link check via Lychee.
</context>

<procedure>
**Reading specs (most common):**

1. Start from `specs/README.md` §1 — the document inventory.
2. For a domain, find the owning RFC via the table (e.g., training → 0005, inference → 0007, parity → 0008).
3. Cross-reference `traceability-matrix.md` to see which tests cover which clauses.

**Code change that touches an RFC clause:**

1. Locate the binding clause (search the relevant RFC by ID).
2. If your change CONFORMS — proceed; cite the clause ID in your PR's Traceability section.
3. If your change CONTRADICTS — STOP. Pick one of:
   - **Update the RFC in the same PR** and bump its version (e.g., 1.0.0 → 1.1.0). Follow `specs/README.md` §2.4.
   - **Open an ADR** that supersedes the specific clause. ADRs win over RFCs by definition.
   In both cases, run `python3 scripts/check_specs.py` before commit.

**Adding a new RFC:**

1. `cp specs/rfcs/0000-template.md specs/rfcs/NNNN-<kebab-title>.md` — next free number.
2. Fill frontmatter: `rfc`, `title`, `status: Draft`, `version: 0.1.0`, `authors`, `created`, `tracks_prd`, `depends_on`, `related`.
3. Add a row to `specs/README.md` §1.2.
4. Set `status: Review`, request review. After acceptance, set `status: Accepted` and `version: 1.0.0`, and update traceability.
5. Validate: `python3 scripts/check_specs.py --check-frontmatter --check-links --check-traceability`.

**Adding a new ADR:**

1. `cp specs/adr/0000-template.md specs/adr/NNNN-<kebab-decision>.md` — next free number.
2. Fill: Context · Decision · Consequences · Alternatives · Status. Be specific about which RFC clause (if any) it supersedes.
3. Once `Accepted`, the file is immutable. Future changes are NEW ADRs.

**Editing the traceability matrix:**

- Any new RFC clause that has a corresponding test or PRD line MUST appear here.
- Run `scripts/check_specs.py --check-traceability` — it asserts mappings exist for every Accepted RFC.

**Glossary entries:**

- Every term that appears in any spec doc with a defined technical meaning MUST be in `glossary.md`. Add before first use.
</procedure>

<patterns>
<do>
— Use exact glossary terminology in code, comments, RFCs, and PR bodies. (E.g., "CLS output" and "Rollout" have precise meanings.)
— Tie every PR back to (issue → RFC clause → test) via the PR template's Traceability section.
— Prefer raising an ADR for narrow execution decisions (e.g., dependency waivers, hardware leashes) — ADRs are cheap and immutable.
— Reserve RFC version bumps for actual contract changes (tolerances, shapes, public APIs, file formats).
</do>
<dont>
— Don't edit an Accepted RFC silently to "match the code." That inverts spec-first. Update via version bump or ADR.
— Don't reuse retired requirement IDs. They're permanently parked.
— Don't add an RFC without listing it in `specs/README.md` — `check_specs.py` will fail.
— Don't reference external links from RFCs without expecting Lychee to verify them.
</dont>
</patterns>

<examples>
Conformance citation pattern in a PR body:

```
## Traceability
- Closes #218
- RFC 0008 §5 (parity probe schedule) — preserved
- RFC 0003 §3 (SIGReg knot grid) — unchanged
- Tests: TST-0008-PARITY-007, TST-0008-PARITY-008 (both pass with L∞ < 1e-4)
```

ADR scaffolding for a hardware leash change:

```
specs/adr/0003-allow-cpu-xl-for-eval-jobs.md
Status: Proposed
Context: eval.yaml currently runs on l4x1 (~$0.80/hr); CPU-bound eval would be cheaper on cpu-xl.
Decision: Add cpu-xl to .ml-intern/cli_agent_config.json `hardware_allowed`.
Consequences: cpu-xl jobs become unilaterally launchable by the intern agent.
Alternatives: keep l4x1 (rejected: cost), add a100-large (rejected: leash forbids).
```
</examples>

<troubleshooting>
| Symptom                                                        | Cause                                                | Fix                                                                                  |
|----------------------------------------------------------------|------------------------------------------------------|--------------------------------------------------------------------------------------|
| `check_specs.py: frontmatter missing key X`                     | RFC/ADR header incomplete                            | Use the template; fill all `RFC_REQUIRED` keys                                       |
| `check_specs.py: broken local link to specs/…`                  | Renamed or removed a file                            | Update the link; or use the file's redirect note                                     |
| Lychee fails on an external URL                                 | Link rot                                             | Use Wayback or remove; do not silently broaden Lychee config                         |
| Code change rejected: "contradicts RFC 0007 §4"                 | Behavior contradicts an Accepted clause              | Decide: amend RFC (version bump) or open ADR that supersedes the clause              |
| Traceability lint: clause X has no test                         | New RFC text without test coverage                   | Add a test stub mapped to the clause ID before merging the spec change                |
</troubleshooting>

<references>
- `specs/README.md` — document inventory, conventions, lifecycle
- `specs/glossary.md` — terms and tolerances
- `specs/traceability-matrix.md` — PRD ↔ RFC ↔ test mapping
- `specs/rfcs/0000-template.md`, `specs/adr/0000-template.md`
- `scripts/check_specs.py` — frontmatter / links / traceability gate
- `CONTRIBUTING.md` — high-level spec-first reminder
</references>
