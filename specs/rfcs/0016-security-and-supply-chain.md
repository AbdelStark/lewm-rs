---
rfc: "0016"
title: "Security, supply chain, secrets, threat model, ml-intern leash"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§6.6 ml-intern", "§7.3 cost controls", "§13 provenance"]
depends_on: ["0001", "0011"]
related: ["0010", "0017"]
---

# RFC 0016 — Security, supply chain, secrets, threat model, ml-intern leash

> **Status:** Accepted · **Version:** 1.0.0
>
> Public Rust crate plus public HF artifacts plus an LLM agent equals three new attack surfaces. This RFC pins how we manage dependencies, secrets, vulnerabilities, container scanning, and the agent's blast radius.

---

## 1. Introduction

### 1.1 Motivation

The blast radius of a compromised supply chain or a runaway agent is the entire budget (200 USD), the entire artifact set, and the reputational standing of the project. The cost of "we forgot to set `--timeout`" is borne by a real card. The investment in this RFC is bounded but the upside is the difference between a controlled project and a dumpster fire.

### 1.2 Goals

1. Define the trust boundary precisely.
2. Specify secret management (where they live, how they rotate).
3. Specify the supply chain controls (`cargo deny`, `cargo audit`, container scan).
4. Specify the threat model: what we defend against, what we accept as residual.
5. Specify the ml-intern leash in detail, building on PRD §6.6 and RFC 0001 §7.
6. Specify the disclosure policy and contact path.

### 1.3 Non-goals

- Cryptographic threat modeling of LeWM itself (out of scope).
- Hardware security (out of scope).

---

## 2. Trust boundary

```
            ┌──────────────────────────────────────────────────────┐
TRUSTED ─►  │ Developer laptop                                       │
            │ GitHub repo (HEAD)                                     │
            │ GitHub Actions runners (managed + self-hosted)         │
            │ HF org `AbdelStark`                                    │
            │ Rust toolchain (rustup)                                │
            └──────────────────────────────────────────────────────┘
                          │
                          │ enters trusted via signed/verified channel
                          ▼
UNTRUSTED   ┌──────────────────────────────────────────────────────┐
INPUTS  ─►  │ Datasets (PushT, SO-100) — hash-verified              │
            │ Reference checkpoints — hash-verified                  │
            │ Container base images (`nvidia/cuda:*`) — Trivy scanned │
            │ Third-party crates — cargo deny + audit                │
            │ ml-intern outputs — reviewed before merge              │
            │ Space user inputs — sanitized in `lewm-infer`           │
            └──────────────────────────────────────────────────────┘
```

**RFC0016-001 [MUST]** — Anything outside the trust boundary requires a verification step (hash, signature, scan, or human review) before being acted upon.

---

## 3. Secret management

### 3.1 Inventory

| Secret | Use | Storage | Rotation |
|--------|-----|---------|----------|
| `HF_TOKEN` | HF Hub auth (download + upload) | GitHub Actions secret, HF Spaces secret, local `~/.config/lewm-rs/secrets.toml` | every 90 days |
| `OTEL_EXPORTER_OTLP_ENDPOINT_AUTH` | Honeycomb/Grafana auth | GitHub Actions secret, optional in HF Spaces | per provider policy |
| `GHCR_TOKEN` | GHCR push (release) | GitHub Actions auto-provided via `${{ secrets.GITHUB_TOKEN }}` | per-job ephemeral |
| `INTERN_AUDIT_HF_TOKEN` | upload ml-intern session logs to private dataset | local + ml-intern session env | every 30 days |

**RFC0016-002 [MUST]** — Secrets **MUST NOT** appear in any committed file. CI's `scripts/check_secrets.py` (a thin wrapper over `gitleaks`) runs on every PR.

**RFC0016-003 [MUST]** — Secrets are scoped narrowly:

- `HF_TOKEN`: write access only to `AbdelStark/lewm-rs-*` repos; read access to public datasets.
- `INTERN_AUDIT_HF_TOKEN`: write access only to `AbdelStark/lewm-rs-intern-audit`.

**RFC0016-004 [MUST]** — `~/.config/lewm-rs/secrets.toml` has mode `0600`; the loader rejects readable-by-others files.

### 3.2 Local secrets file format

```toml
# ~/.config/lewm-rs/secrets.toml
# This file MUST be mode 0600.

[hf]
token = "hf_..."
namespace = "AbdelStark"

[intern_audit]
token = "hf_..."

[otel]
endpoint = "https://api.honeycomb.io/v1/traces"
auth_header = "x-honeycomb-team: <key>"
```

### 3.3 Loading at runtime

```rust
pub fn load_secrets() -> Result<Secrets, SecurityError> {
    // 1. env vars (highest priority — for CI)
    if let Ok(token) = std::env::var("HF_TOKEN") {
        return Ok(Secrets::from_env(token, ...));
    }
    // 2. local file
    let path = dirs::config_dir().unwrap().join("lewm-rs/secrets.toml");
    let mode = std::fs::metadata(&path)?.permissions().mode();
    if mode & 0o077 != 0 {
        return Err(SecurityError::SecretsFilePermissive { path });
    }
    let text = std::fs::read_to_string(&path)?;
    Ok(toml::from_str(&text)?)
}
```

**RFC0016-005 [MUST]** — Secrets in memory are wrapped in `secrecy::SecretString`; their `Display` impl is `"[REDACTED]"`.

---

## 4. Dependency controls

### 4.1 `cargo deny`

`deny.toml` declares:

```toml
[graph]
all-features = true

[advisories]
db-path = "~/.cargo/advisory-db"
db-urls = ["https://github.com/rustsec/advisory-db"]
vulnerability = "deny"
unmaintained = "warn"
yanked = "deny"
notice = "warn"

[licenses]
unlicensed = "deny"
copyleft = "deny"
allow-osi-fsf-free = "neither"
default = "deny"
confidence-threshold = 0.93
allow = [
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC",
    "Zlib", "Unlicense", "CC0-1.0", "MPL-2.0",
]

[bans]
multiple-versions = "warn"     # Burn ecosystem may pull dup versions of `serde`
wildcards = "deny"
deny = [
    { name = "openssl-sys" },   # we use rustls
]
```

**RFC0016-006 [MUST]** — `cargo deny check` passes on every PR.

**RFC0016-007 [MUST]** — Adding a new dependency requires a one-line entry in `CHANGELOG.md` under "Added" and an explicit license assertion in the PR description.

### 4.2 `cargo audit`

Runs on every PR and nightly:

```bash
cargo audit --deny warnings
```

**RFC0016-008 [MUST]** — Any RUSTSEC advisory affecting a workspace dep blocks merge until either:

- The dep is upgraded past the advisory.
- An ADR documents the waiver with a date by which the advisory will be resolved.

### 4.3 SBOM

`scripts/sbom.py` generates a CycloneDX SBOM for the workspace:

```bash
cargo cyclonedx --format json > sbom.json
```

**RFC0016-009 [MUST]** — Each release attaches the SBOM as a release asset.

### 4.4 Container scanning

Already specified in [RFC 0011 §5.2](0011-ci-cd-and-release-engineering.md). HIGH and CRITICAL Trivy findings block release.

---

## 5. Threat model

### 5.1 In-scope threats

| ID | Threat | Defense |
|----|--------|---------|
| T-01 | Malicious dependency | `cargo deny`; pinned versions; audited supply chain |
| T-02 | Compromised HF token | Scoped tokens; rotation; redaction in logs |
| T-03 | Container vulnerability | Trivy; pinned base image; cosign signing |
| T-04 | ml-intern runaway cost | JSON config leash; system prompt; per-job `--timeout` |
| T-05 | ml-intern weight tampering | Read-only glob for `losses/`; loss tests catch divergence |
| T-06 | Malformed dataset input | Schema validation in `lewm-data`; shape/dtype assertions |
| T-07 | Image upload in Space (DoS via huge file) | Gradio size limit; Rust binary timeout |
| T-08 | Prompt injection of ml-intern | System prompt reinforces; outputs reviewed; sessions audited |
| T-09 | Reproducibility regression | Pinned toolchain; reproducible build; CI verify |
| T-10 | Secret leak in log | Redactor in `tracing` layer |

### 5.2 Out-of-scope threats

- Nation-state actor with GitHub repo access (out of practical defense).
- Compromised CI runner (we use GitHub-hosted, trust GitHub).
- Side-channel attacks on the training cluster.

### 5.3 Residual risk

- A single contributor compromise of `AbdelStark` GitHub account would compromise everything. **Mitigation**: hardware MFA, branch protection, signed commits required for `main`.

---

## 6. ml-intern leash (detailed)

### 6.1 Configuration file

Reproduced from [RFC 0001 §7.1](0001-project-foundation-and-build-system.md) and specified in full here for finality:

```jsonc
{
  "schema_version": "1.0.0",
  "project": "lewm-rs",
  "namespace": "AbdelStark",
  "billing": {
    "hard_cap_usd": 200,
    "soft_cap_usd": 100,
    "per_job_default_timeout": "30m",
    "session_cap_usd": 20
  },
  "hardware_allowed":          ["cpu-basic", "cpu-xl", "l4", "a10g-large"],
  "hardware_denied":           ["a100-large", "a100-xl", "h100", "h100-xl"],
  "jobs_allowed":              ["smoke_pusht.yaml", "short_pusht.yaml", "smoke_so100.yaml", "short_so100.yaml", "eval.yaml"],
  "jobs_human_approval_required": ["train_pusht.yaml", "train_so100.yaml"],
  "files_readonly_glob": [
    "crates/lewm-core/src/losses/**",
    "specs/**",
    "PRD.md",
    "configs/pusht.toml",
    "configs/so100.toml",
    "rust-toolchain.toml",
    "Cargo.lock"
  ],
  "files_writable_glob": [
    "reports/**",
    "python/**",
    "jobs/**",
    "configs/overrides/**",
    ".ml-intern/sessions/**"
  ],
  "audit": {
    "session_log_repo": "AbdelStark/lewm-rs-intern-audit",
    "private": true,
    "upload_at_session_end": true,
    "redact_keys": ["HF_TOKEN", "OTEL_AUTH"]
  },
  "command_denylist": [
    "rm -rf",
    "rm -rf /",
    "git push --force",
    "git push -f",
    "git push --force-with-lease",
    "git checkout main",
    "git reset --hard",
    "cargo install --git",
    "curl .* | sh",
    "wget .* | sh",
    "hf jobs run .* --hardware a100.*",
    "hf jobs run .* --hardware h100.*",
    "hf jobs run .*(?!.*--timeout)"
  ],
  "command_require_confirmation": [
    "hf jobs run",
    "hf upload",
    "git push",
    "cargo publish",
    "cargo add",
    "uv add"
  ]
}
```

### 6.2 System prompt

```markdown
# ml-intern role for lewm-rs

You are an assistant operating inside the `lewm-rs` project. Adhere to the following at all times:

1. **PRD and spec set are binding.** `/PRD.md` and `/specs/` are the source of truth. If your action would contradict a clause, stop and ask the human.

2. **Cost discipline.** The project's hard cap is 200 USD. Your session cap is 20 USD. Any `hf jobs run` MUST carry `--timeout` and use one of the approved hardware tiers.

3. **Read-only files.** Do not modify any file under `crates/lewm-core/src/losses/`, `specs/`, `PRD.md`, `configs/pusht.toml`, `configs/so100.toml`. Other read-only globs in `cli_agent_config.json`.

4. **Approval-required jobs.** `train_pusht.yaml` and `train_so100.yaml` MUST NOT be launched without explicit human go-ahead. Smoke and short variants are pre-approved.

5. **Audit.** Every session's commands and outputs are uploaded to `AbdelStark/lewm-rs-intern-audit` (private). Assume everything you do is logged.

6. **Disagreement protocol.** If you believe a rule is wrong or blocks legitimate work, file an issue tagged `intern-policy-discussion`. Do not bypass the rule.

7. **Tool boundaries.** Use the tools provided. Do not invoke shells that bypass the leash (e.g., `bash -c "$(curl ...)"`).
```

**RFC0016-010 [MUST]** — The system prompt **MUST** be loaded by every ml-intern session at start. Verified by the session log: every uploaded log carries a header attesting the prompt was loaded.

### 6.3 Session log format

Each session log is a JSONL file in the private audit repo:

```json
{"ts": "2026-05-12T14:00:00Z", "kind": "session_start", "prompt_sha256": "...", "config_sha256": "..."}
{"ts": "2026-05-12T14:00:05Z", "kind": "command", "cmd": "hf jobs run --hardware l4 --timeout 30m -f jobs/smoke_pusht.yaml"}
{"ts": "2026-05-12T14:00:08Z", "kind": "command_output", "stdout_hash": "...", "exit_code": 0}
...
{"ts": "2026-05-12T15:30:00Z", "kind": "session_end", "duration_s": 5400, "cost_usd": 1.20}
```

### 6.4 Per-session ceiling

`session_cap_usd: 20` means: when the running cost of HF Jobs launched **in this session** crosses 20 USD, the intern emits a CRITICAL log line and refuses to launch further jobs until a human acknowledges via a freshly committed file under `.ml-intern/sessions/<id>/ack.json`.

---

## 7. Disclosure policy

### 7.1 SECURITY.md

```markdown
# Security policy for lewm-rs

If you find a security issue, please email abdel@starkware.co with the subject
"[SECURITY] lewm-rs". We will acknowledge within 7 days and aim to remediate
critical issues within 30 days.

Do not file public issues for vulnerabilities.

# Supported versions

Only the latest `0.x` release receives security updates.
```

### 7.2 GitHub Security Advisories

Enabled on the repo. We commit to publishing CVE-grade reports for any confirmed vulnerability that affects users (e.g., a deserialization issue in a checkpoint loader).

### 7.3 Coordinated disclosure

For dependency vulnerabilities, we coordinate with upstream (e.g., Burn) before public disclosure.

---

## 8. Testing strategy

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0016-DENY-001 | `cargo_deny_passes` | unit (CI) | RFC0016-006 |
| TST-0016-AUDIT-001 | `cargo_audit_passes` | unit (CI) | RFC0016-008 |
| TST-0016-SECRETS-001 | `gitleaks_clean` | unit (CI) | RFC0016-002 |
| TST-0016-PERM-001 | `secrets_file_mode_0600` | integration | RFC0016-004 |
| TST-0016-REDACT-001 | `secret_redaction_in_logs` | unit | RFC0016-005 |
| TST-0016-INTERN-001 | `intern_config_schema_valid` | unit | §6.1 |
| TST-0016-INTERN-002 | `intern_session_log_uploaded` | integration | §6.3 |
| TST-0016-SBOM-001 | `sbom_generation_on_release` | integration | RFC0016-009 |
| TST-0016-CONTAINER-001 | `trivy_no_high_critical` | unit (CI) | RFC0016-010 |

---

## 9. Operational considerations

### 9.1 Observability

```
security/deny_failures       # count of cargo deny denials triaged
security/audit_advisories    # count of RUSTSEC advisories active
security/intern_sessions     # count of intern sessions
security/intern_cost_usd     # cumulative cost attributed to intern
```

### 9.2 Runbook

- **"Trivy reports a critical vuln in the base image."** — bump base image; re-tag release; document in CHANGELOG.
- **"`cargo audit` reports a new advisory."** — open issue with `security` label; bump or wait per ADR.
- **"ml-intern blew past 20 USD."** — review the audit log; if a runaway, file an issue and tighten the prompt.
- **"User reports a security issue."** — follow §7.1 process.

### 9.3 Capacity

Security tooling overhead is small: `cargo deny` ~ 30 s; `cargo audit` ~ 10 s; Trivy ~ 30 s.

---

## 10. Performance considerations

None beyond §9.3.

---

## 11. Acceptance criteria

- [ ] All TST-0016-* pass.
- [ ] `deny.toml`, `SECURITY.md` exist.
- [ ] `cli_agent_config.json` matches §6.1.
- [ ] System prompt at `.ml-intern/prompts/system.md` matches §6.2.

---

## 12. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | A prompt-injection in a dataset leaks into ml-intern | L | M | Datasets read by Rust loaders, not exposed to intern; intern sees only diffs and reports |
| R-2 | Upstream RUSTSEC for a critical dep | M | M | Pin + monitor + replacement path documented |
| R-3 | HF token in a public log | L | H | Redactor; rotation policy |
| R-4 | Container base image yanked | L | L | Pin SHA; vendored mirror plan |

---

## 13. Open questions

OQ-2016-1 — Should we adopt `crev` for dependency review? Considered. v2 maybe; cost-benefit unclear for a small workspace.

OQ-2016-2 — Should we run a bug bounty? Probably not for v1; revisit.

---

## 14. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0016.*
