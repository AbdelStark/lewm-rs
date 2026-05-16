# Security policy

Full policy in
[`SECURITY.md`](https://github.com/AbdelStark/lewm-rs/blob/main/SECURITY.md).
Highlights:

- **Reporting.** Report vulnerabilities by emailing the maintainer
  (address in `SECURITY.md`). Do **not** open a public issue for a
  security bug.
- **Scope.** The threat model is documented in
  [RFC 0016 — Security and supply chain](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0016-security-and-supply-chain.md).
- **Supply chain.** `cargo deny` and `cargo audit` gates run on every
  CI build. The set of allowed advisory waivers is enumerated in
  the `Makefile`'s `check` target.
- **Secret scanning.** Gitleaks runs on every PR. Configuration in
  `.gitleaks.toml`.
- **Container image.** `ghcr.io/abdelstark/lewm-rs:latest` is built
  from the checked-in `Dockerfile` only — no third-party base layers
  beyond Debian slim + the official Rust + CUDA images.

## What the threat model covers

- Compromised training data → model misbehaviour.
- Compromised checkpoint → unsafe deployment.
- Compromised build supply chain → injected code in the binary.

## What it does not cover

- Adversarial inputs at inference time (out of scope for v1).
- Side-channel attacks on the inference binary (out of scope).
