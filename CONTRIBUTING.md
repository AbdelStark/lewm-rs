# Contributing to lewm-rs

This repository is spec-first. Before changing behavior, read `PRD.md`,
`specs/README.md`, and the RFC that owns the subsystem you are touching.

## Development style

- Keep changes small and tied to one issue.
- Prefer the existing workspace patterns over new abstractions.
- Run formatting, linting, tests, spec checks, and layer checks before opening a PR.
- Document public APIs with the rustdoc shape, errors, invariants, and examples
  required by RFC 0015.
- Do not add dependencies casually. New dependencies must pass `cargo deny`,
  be MIT-compatible, and be listed in `CHANGELOG.md`.

## RFC and ADR process

- RFCs under `specs/rfcs/` are binding once accepted.
- Use `specs/adr/0000-template.md` for decisions that affect architecture,
  dependency policy, artifact policy, or release behavior.
- If a code change contradicts an accepted RFC, update the RFC and bump its
  version in the same PR, following `specs/README.md`.
- Keep traceability intact: user-facing behavior should map back to PRD,
  RFC, and test IDs.

## Commit and PR expectations

- Use clear commits that explain the change without tool-specific branding.
- Include validation in the PR body.
- Link the issue with `Closes #<number>` when the acceptance criteria are met.
- Keep generated artifacts out of commits unless the RFC explicitly requires
  them.

## Developer Certificate of Origin

Contributions use the Developer Certificate of Origin. By adding a
`Signed-off-by` trailer, you certify that you have the right to submit the
work under this repository's license:

```text
Signed-off-by: Your Name <you@example.com>
```

Use `git commit -s` to add the trailer automatically.
