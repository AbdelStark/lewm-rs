# lewm-rs documentation site

This directory holds the [mdBook](https://rust-lang.github.io/mdBook/)
documentation site for `lewm-rs`. Source files are in `src/`; the
configuration is `book.toml`; custom theme overrides live in `theme/`.

The built site is served at
[`https://abdelstark.github.io/lewm-rs/`](https://abdelstark.github.io/lewm-rs/).

## Local build

```sh
cargo install mdbook --version "=0.4.52" --locked
mdbook build docs       # produces docs/book/
mdbook serve docs       # interactive preview at http://localhost:3000
mdbook test  docs       # compile-check all Rust code blocks
```

Or from the repository root:

```sh
make docsite
```

## Layout

```text
docs/
├── book.toml             # mdBook configuration
├── theme/
│   ├── custom.css        # serif content typography, academic table styling
│   └── footnote.js       # right-aligns numeric table cells
└── src/
    ├── SUMMARY.md        # table of contents
    ├── introduction.md
    ├── reading-guide.md
    ├── status.md
    ├── concepts/         # Part I: JEPA, LeWM, SIGReg, ViT, AdaLN-zero, CEM
    ├── architecture/     # Part II: encoder, predictor, action encoder, projector, jepa
    ├── training/         # Part III: pipeline, data, losses, optimizer, mixed precision, …
    ├── planning/         # Part IV: CEM, PushT eval, SO-100 eval, warm-start
    ├── inference/        # Part V: ONNX export, Tract, Burn runners, benchmark, demo
    ├── parity/           # Part VI: parity tests, tolerances, gotchas
    ├── results/          # Part VII: PushT, SO-100, cost, discussion
    ├── crates/           # Part VIII: per-crate API tours
    ├── reproducing/      # Part IX: quickstart, training, inference, docker, gate
    ├── reference/        # Part X: glossary, notation, tolerances, RFC index, ADRs, bibliography
    └── community/        # contributing, code of conduct, security, license, acknowledgments
```

## Editing conventions

- **Headings**: H1 only at the top of a page; H2/H3 for sections.
- **Math**: MathJax is enabled. Inline with `$...$`, display with `$$...$$`.
- **Code fences**:
  - ` ```rust,ignore ` for illustrative Rust pseudocode.
  - ` ```text ` for ASCII diagrams, dataflow drawings, output examples.
  - ` ```sh ` / ` ```python ` for runnable commands or snippets in
    other languages.
- **RFC cross-references**: link to the in-repo path,
  e.g. `[RFC 0003 §4.2.1](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0003-sigreg-and-loss-functions.md#421-algorithm)`.
- **Status badges**: use the `lewm-badge` CSS classes for
  `done` / `partial` / `todo` / `note`.

## What this site is for

This site is the **explanatory layer** on top of `specs/` and the
source. Specs are normative; this site is pedagogical. Where a claim
in this site is non-obvious, it should cite an RFC, an ADR, a report,
or a source path. See [`docs/src/reading-guide.md`](src/reading-guide.md)
for the conventions.
