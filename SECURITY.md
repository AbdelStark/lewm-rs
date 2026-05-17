# Security policy

## Reporting a vulnerability

**Do not open a public GitHub issue for security findings.**

Report security vulnerabilities by either:

1. Emailing **abdel@starkware.co** with the subject line
   `[SECURITY] lewm-rs: <one-line summary>`.
2. Opening a private [security advisory](https://github.com/AbdelStark/lewm-rs/security/advisories/new)
   on GitHub.

Please include, at minimum:

- A description of the issue and its potential impact.
- A minimal reproducer (commit, branch, container tag, environment).
- Whether you have a proposed fix.
- Whether you intend to disclose publicly and on what timeline.

## What you can expect

| Phase                | Target SLA                                                  |
| -------------------- | ----------------------------------------------------------- |
| Initial acknowledgement | within 7 calendar days                                   |
| Triage decision      | within 14 calendar days                                     |
| Fix for critical      | within 30 calendar days from triage                         |
| Coordinated disclosure | 90 days from initial report, or sooner by mutual agreement |

We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_disclosure):
the fix lands first, then we publish an advisory, then we credit the reporter
(unless anonymity is requested).

## Supported versions

| Version line | Status              | Receives security fixes? |
| ------------ | ------------------- | ------------------------ |
| `0.x`        | Current             | Yes — the latest tag.    |
| Pre-release  | `vX.Y.Z-rcN` tags   | No — use the GA tag.     |

The release cadence is documented in [`RELEASE.md`](RELEASE.md). Hot-fix
patches for critical CVEs are tagged out-of-band when warranted.

## Threat model and scope

`lewm-rs` is a research / engineering codebase: training, planning, and
inference for the LeWorldModel architecture. The threat model is documented
in [RFC 0016 — security and supply chain][rfc-0016]. The short version:

**In scope** (please report):

- Memory safety in `unsafe` code paths (none exist today — the workspace is
  `unsafe_code = "deny"`; a regression is in scope).
- Command injection in `scripts/launch_hf_job.py`, `scripts/sbom.py`, or
  any other publication-bound script.
- Credential exfiltration: `HF_TOKEN`, OTLP credentials, GHCR PAT,
  Trackio API keys, etc.
- Container escape from the published `ghcr.io/abdelstark/lewm-rs:*` image.
- Supply-chain tampering: forged signatures, malicious dependencies, broken
  reproducibility, missing or invalid build attestations.
- Authentication / authorisation bypasses in any networked service the
  project ships (the demo Space, the OTLP exporter, etc.).

**Out of scope** (no SLA, but we still appreciate the report):

- Issues that require the attacker to already have write access to the
  repository or the Hub namespace.
- Theoretical attacks against PyTorch / Burn / Tract that we cannot mitigate
  from this side of the dependency boundary — please file with the upstream
  project; we will track and mirror their advisories.
- Denial-of-service in training paths (training jobs are intentionally
  resource-heavy).

## Supply-chain controls

Each published release ships with:

- Cosign-signed container image (`ghcr.io/abdelstark/lewm-rs:vX.Y.Z`) in the
  Sigstore transparency log (Rekor).
- GitHub built-in build provenance attestations for the binaries, the
  CycloneDX SBOM, and the container image, verifiable with
  `gh attestation verify <artifact> --owner AbdelStark`.
- A reproducible Linux musl build verified by the `verify-reproducible` job.
- A deterministic CycloneDX SBOM at `dist/sbom.cdx.json`.

These controls are documented in [RFC 0016][rfc-0016] and the release runbook
in [`RELEASE.md`](RELEASE.md).

## Credential hygiene

This repository **must not** contain live credentials. Secret scanning is
enforced by:

- `gitleaks` (CI + pre-commit — config in `.gitleaks.toml`).
- `scripts/check_secrets.py` (CI).
- GitHub native push-protection on the org.

If you discover a leaked secret, treat it as a security report: open a
private advisory or email, and we will rotate the secret immediately.

[rfc-0016]: specs/rfcs/0016-security-and-supply-chain.md
