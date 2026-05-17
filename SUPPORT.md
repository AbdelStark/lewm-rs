# Getting help

This document tells you **where to ask** depending on the kind of question you
have. Routing things to the right surface is the single biggest force
multiplier for a single-maintainer project.

| You want to                                          | Surface                                                                                  |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Report a bug                                         | [GitHub issue](https://github.com/AbdelStark/lewm-rs/issues/new?template=bug.md)         |
| Request a feature                                    | [GitHub issue](https://github.com/AbdelStark/lewm-rs/issues/new?template=feature.md)     |
| Report a numerical parity regression                 | [GitHub issue](https://github.com/AbdelStark/lewm-rs/issues/new?template=parity.md)      |
| Report a security vulnerability                      | See [`SECURITY.md`](SECURITY.md) — do **not** open a public issue.                       |
| Ask "how do I do X" in lewm-rs                       | [GitHub Discussions](https://github.com/AbdelStark/lewm-rs/discussions)                  |
| Discuss the LeWorldModel paper or JEPA in general    | Upstream — see [arXiv:2502.16560](https://arxiv.org/abs/2502.16560)                      |
| Propose a non-trivial design / contract change       | RFC PR under `specs/rfcs/` (start from `0000-template.md`)                               |

## Before you ask

Most questions are already answered in:

1. The **docsite**: <https://abdelstark.github.io/lewm-rs/>. It has a
   per-section index — start at the [introduction](docs/src/introduction.md)
   if you do not know where to look.
2. The **RFCs** (`specs/rfcs/0001-…-0018-…`). These are the authoritative
   contracts; the docsite is a friendlier read on top of them.
3. The **glossary** (`specs/glossary.md`) covers project-specific terms
   (SIGReg, AdaLN-zero, pred-proj, etc.).
4. The **changelog** (`CHANGELOG.md`) records every behaviour change.

When in doubt, search the closed issues first; many "is this a bug?" questions
have been resolved with a docs link or a missed flag.

## What a good bug report looks like

For numerical bugs (training divergence, parity test failures), include:

* Exact command and config used (or a minimal reproducer).
* The git SHA, `rustc --version`, and `cargo --version`.
* The full output of `make check`.
* The relevant lines from `losses.jsonl` and `run_report.json`.

For runtime / build bugs, include:

* OS + arch (e.g. `Ubuntu 22.04 x86_64`, `macOS 14 aarch64`).
* The exact `cargo …` invocation.
* Whether `make check` passes on the same checkout.

## Response times

This is a single-maintainer single-author project. Expected response time on
GitHub issues is **best-effort within 7 days**. Security reports follow the
SLAs in [`SECURITY.md`](SECURITY.md).

## I want to contribute, not just ask

Wonderful. Start with [`CONTRIBUTING.md`](CONTRIBUTING.md) and the
[Contributing page on the docsite](docs/src/community/contributing.md).
