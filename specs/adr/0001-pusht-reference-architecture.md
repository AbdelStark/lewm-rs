---
adr: "0001"
title: "PushT reference architecture source of truth"
status: Implemented
date: 2026-05-14
authors: ["Abdel"]
tracks_rfc: ["0002", "0008", "0018"]
supersedes: []
superseded_by: null
pr: null
---

# ADR 0001 — PushT Reference Architecture Source of Truth

## Context

RFC 0002 originally carried an unresolved question about whether the published
PushT checkpoint used ViT-Small or ViT-Tiny. The repository now needs a locked
target before replacing `pusht-minimal-lewm` with the full trainable model.
Guessing this from the paper summary would risk implementing the wrong parameter
shapes.

## Decision

Use the public `quentinll/lewm-pusht` model repo at revision
`22b330c28c27ead4bfd1888615af1340e3fe9052` as the source of truth for the v1
PushT architecture and parity target.

The locked architecture is ViT-Tiny with `patch_size=14`, `hidden_size=192`,
`depth=12`, `heads=3`, and `intermediate_size=768`; an action encoder over
packed PushT actions with `raw_action_dim=2`, `frameskip=5`, and
`input_dim=10`; a predictor with `num_frames=3`, `depth=6`, `heads=16`,
`dim_head=64`, `attention_inner_dim=1024`, `mlp_dim=2048`, and dropout `0.1`;
and projector/pred_proj MLPs with `input=192`, `hidden=2048`, `output=192`.

## Alternatives considered

- **Keep the ViT-Small draft** — rejected because it contradicts the published
  checkpoint config and produces a different shape/parameter contract.
- **Infer only from the paper parameter count** — rejected because the checkpoint
  gives exact dimensions, hashes, and state-dict shapes.

## Consequences

### Positive

- The full Rust model path can target exact checkpoint shapes.
- Parity tests can fail early on source drift via committed metadata hashes.
- SO-100 warm-start can share the real PushT latent width.

### Negative

- Several accepted specs and draft cost estimates that assumed `D=384` need
  follow-up cleanup beyond this ADR.

### Neutral or to revisit

- The current `pusht-minimal-lewm` smoke mode remains intentionally separate and
  keeps using its 4-D scalar training core.

## Implementation

Implemented by the commit that adds `tests/fixtures/reference_model.meta.json`,
refreshes the parity fixture action shape, and updates the Rust config defaults.

## References

- `https://huggingface.co/quentinll/lewm-pusht`
- `https://github.com/lucas-maes/le-wm`
- RFC 0002
- RFC 0008

---

*End of ADR 0001.*
