# ADR index

Architectural Decision Records. Each ADR is one immutable decision
document, used when a single specific choice needs to be recorded
without rewriting an RFC. ADRs live under
[`specs/adr/`](https://github.com/AbdelStark/lewm-rs/tree/main/specs/adr).

| # | Title | Status | Source |
|---|-------|--------|--------|
| 0001 | PushT reference architecture lock | Accepted | [0001-pusht-reference-architecture.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/adr/0001-pusht-reference-architecture.md) |
| 0002 | Burn MSRV and `cargo audit` waiver | Accepted | [0002-burn-msrv-and-audit-waiver.md](https://github.com/AbdelStark/lewm-rs/blob/main/specs/adr/0002-burn-msrv-and-audit-waiver.md) |

## When to write an ADR vs an RFC

| If the change is | Use |
|------------------|-----|
| A new component, a new contract surface, a new spec area | An RFC |
| A single, scoped decision (e.g. "lock the reference checkpoint to commit X") | An ADR |
| Override of one clause in an existing RFC | An ADR citing the RFC |
| A bug fix in an existing RFC | A patch-version update of the RFC, in-place |

The ADR governance contract is the same as RFCs': drafted on a
feature branch, reviewed by PR, accepted on merge, immutable after.

## Template

New ADRs start from
[`specs/adr/0000-template.md`](https://github.com/AbdelStark/lewm-rs/blob/main/specs/adr/0000-template.md).

## See also

The PRD (`PRD.md` in the repo root) is the higher-level product
requirements document. RFCs trace back to PRD sections via the
`tracks_prd:` front-matter field. The
[traceability matrix](https://github.com/AbdelStark/lewm-rs/blob/main/specs/traceability-matrix.md)
maps PRD § → RFC IDs → source modules → tests.
