# How to read these docs

These documents are designed to be **read linearly** as a course on
LeWorldModel reproduction, and **searched non-linearly** as a reference
manual. To help with both modes, every page follows a consistent layout.

## Page anatomy

Each chapter begins with three short framing paragraphs:

- **Motivation** — why the topic exists at all, in one or two sentences.
- **Position in the system** — where the topic sits in the pipeline, with a
  small ASCII diagram or table.
- **What you should leave with** — a bullet list of the concrete take-aways.

Long pages then split into:

1. **The story**, prose, with equations rendered in MathJax.
2. **The contract**, a bullet list of normative requirements (RFC IDs in
   brackets, e.g. `[RFC0003-001]`).
3. **The code**, Rust signatures or pseudocode showing how the contract is
   realized in `crates/`.
4. **The numerical evidence**, parity tests, loss curves, or benchmark
   tables, with links to the source dumps.
5. **Cross-references**, to other docs pages, RFCs, and source files.

## Notation

All notation is defined once in the [symbol conventions](./reference/notation.md)
page and used consistently across the rest of the site. The most common
symbols are:

| Symbol | Meaning |
|:------:|---------|
| $B$    | Batch dimension |
| $T$    | Temporal dimension (frames in a window) |
| $H, W$ | Image height and width in pixels |
| $C$    | Channels (3 for RGB) |
| $D$    | Embedding dim ($192$ for the locked PushT ViT-tiny) |
| $K$    | Number of random projections in SIGReg ($1024$) |
| $J$    | Number of frequency knots in SIGReg ($17$) |
| $\lambda$ | SIGReg loss weight ($1.0$ default) |
| $\mathcal L_{\text{pred}}$ | Prediction MSE loss |
| $\mathcal L_{\text{sigreg}}$ | SIGReg loss |
| $\mathcal L$ | Total loss, $\mathcal L_{\text{pred}} + \lambda\,\mathcal L_{\text{sigreg}}$ |

Tensor shapes are written `(B, T, D)`-style throughout, matching the
PyTorch / Burn convention. Equations are written in standard math notation
with no PyTorch- or Rust-specific shorthand.

## Status badges

Anywhere a feature, result, or contract has a meaningful status, one of the
following badges appears:

<span class="lewm-badge lewm-badge--done">Done</span>
The item is implemented, parity-checked, and merged.

<span class="lewm-badge lewm-badge--partial">Partial</span>
The item is implemented but missing one or more sub-checks (e.g. trained but
not yet evaluated).

<span class="lewm-badge lewm-badge--todo">Planned</span>
The item is specified but not yet implemented.

<span class="lewm-badge lewm-badge--note">Note</span>
A neutral annotation, not a status claim.

## Cross-reference policy

Wherever a non-trivial design choice is described, the docs cite the
authoritative source:

- **RFC** references look like [RFC 0003 §4.2.1](./reference/rfcs.md) and
  link to the in-repo accepted spec.
- **ADR** references look like [ADR 0001](./reference/adrs.md).
- **Source** references look like
  `crates/lewm-core/src/predictor.rs:120` and resolve to the live source on
  GitHub.
- **Report** references look like
  [`reports/pusht_training.md`](https://github.com/AbdelStark/lewm-rs/blob/main/reports/pusht_training.md).

If a docs page makes a claim that does not cite one of these, treat it as
"editorial commentary" — informative but not normative.

## What's missing

The docs are an explanatory layer on top of the specs and code. Where the
specs are more authoritative, the docs link out rather than duplicate. In
particular, the docs do **not** restate:

- The full RFC text. The [RFC index](./reference/rfcs.md) is your map.
- The CI configuration. See `.github/workflows/`.
- The training reports. See [`reports/`](https://github.com/AbdelStark/lewm-rs/tree/main/reports).
- The model cards. See the Hugging Face repos
  [`abdelstark/lewm-rs-pusht`](https://huggingface.co/abdelstark/lewm-rs-pusht)
  and [`abdelstark/lewm-rs-so100`](https://huggingface.co/abdelstark/lewm-rs-so100).

When in doubt, the source is the ground truth and the specs are the design
contract. Use this site to find your way in.
