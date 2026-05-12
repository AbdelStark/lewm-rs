# `lewm-rs` diagrams

**Status:** Accepted · **Version:** 1.0.0

This directory holds system diagrams that supplement the specification set. Diagrams **MUST** have a textual source under version control; rendered binary artifacts (`.svg`, `.png`, `.pdf`) are convenience-only.

---

## Diagram inventory

| Source | Render | Subject | Owning RFC |
|--------|--------|---------|------------|
| `system-overview.mmd` | `system-overview.svg` | Top-level dataflow (datasets → loader → trainer → planner → inference) | TECHNICAL_SPECIFICATION §1.4 |
| `crate-graph.mmd` | `crate-graph.svg` | Workspace crate dependency graph | TECHNICAL_SPECIFICATION §4.2 |
| `training-state-machine.mmd` | `training-state-machine.svg` | Trainer state machine | RFC 0005 §8 |
| `rng-tree.mmd` | `rng-tree.svg` | RNG sub-stream tree | RFC 0013 §4 |
| `sigreg-shape-walk.mmd` | `sigreg-shape-walk.svg` | SIGReg per-step shape transformations | RFC 0003 Appendix A |
| `export-pipeline.mmd` | `export-pipeline.svg` | Burn → ONNX/NNEF → Tract export ladder | RFC 0007 §4–§6 |
| `ci-pipeline.mmd` | `ci-pipeline.svg` | GitHub Actions workflow matrix and dependencies | RFC 0011 §3 |

---

## Conventions

- **Mermaid** (`.mmd`) is the default source format. Renders directly on GitHub.
- **PlantUML** (`.puml`) is used for sequence diagrams that exceed Mermaid's ergonomics.
- **Graphviz** (`.dot`) is used for arbitrary directed graphs.
- ASCII inline in the parent doc is acceptable for small diagrams; this directory is reserved for diagrams that would dominate the prose.

Every diagram file carries a header comment:

```
%% lewm-rs · system-overview · v1.0.0 · 2026-05-12
%% Owned by TECHNICAL_SPECIFICATION.md §1.4
```

---

## Rendering

Renders happen on demand:

```bash
mmdc -i diagrams/system-overview.mmd -o diagrams/system-overview.svg
```

The `docs.yml` CI workflow ([RFC 0011 §3.5](../rfcs/0011-ci-cd-and-release-engineering.md)) does this for all `.mmd` files and publishes alongside the rendered specs on GitHub Pages.

---

## Updating a diagram

1. Edit the `.mmd` (or other source).
2. Re-render locally for visual check.
3. Commit only the source; CI re-renders for publication.
4. Update the inventory above if a new diagram is added.

*End of `specs/diagrams/README.md`.*
