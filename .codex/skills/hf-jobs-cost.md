---
name: hf-jobs-cost
description: Cost-discipline and approval protocol for Hugging Face Jobs. Activate before launching ANY `hf jobs run`, editing files under `jobs/`, updating `reports/cost.md`, or interpreting `scripts/launch_hf_job.py` output. Caps are enforced by `.ml-intern/cli_agent_config.json` AND by `scripts/check_jobs.py` / `scripts/check_train_so100_job.py` / `python/cost_ledger.py`. Total project hard cap is $200 (current spend: $11.70 per `reports/cost.md`).
prerequisites: `HF_TOKEN` for actual launches (NOT required for dry-runs and `check_jobs.py`)
---

# HF Jobs Cost Discipline

<purpose>
Training runs cost real money on shared infrastructure. This project operates under a hard $200 budget. Every `hf jobs run` invocation must be policy-checked first. This skill encodes the policy, the launch protocol, and the cost-ledger update flow.
</purpose>

<context>
- Authoritative leash: `.ml-intern/cli_agent_config.json`.
  - `billing.hard_cap_usd = 200`, `soft_cap_usd = 100`, `session_cap_usd = 20`, default per-job timeout `30m`.
  - `hardware_allowed = ["cpu-basic", "cpu-xl", "l4x1", "a10g-large"]`.
  - `hardware_denied = ["a100-large", "a100-xl", "h100", "h100-xl"]`.
  - `jobs_allowed = ["smoke_pusht.yaml", "short_pusht.yaml", "smoke_so100.yaml", "short_so100.yaml", "eval.yaml"]` — agent may launch unilaterally after running through the protocol.
  - `jobs_human_approval_required = ["train_pusht.yaml", "train_so100.yaml"]` — agent MUST stop and ask before launching.
- `scripts/check_jobs.py` validates the shape of every `jobs/*.yaml` (hardware, timeout, env vars, image, command tokens). Runs inside `make check`.
- `scripts/check_train_so100_job.py` validates the SO-100 full-training job specifically.
- `python/cost_ledger.py check --path reports/cost.md --cap-usd 200` runs inside `make check` and is enforced.
- `python/hf_pricing.py` is the pricing source of truth (A10G-large was previously documented at $5/hr; the verified rate is **$1.50/hr**, per the May 2026 fix).
- All jobs share `image: ghcr.io/abdelstark/lewm-rs:latest` and `namespace: abdelstark`.
</context>

<procedure>
**Pre-flight (every launch):**

1. **Identify the job file.** `jobs/<name>.yaml`. Confirm it exists.
2. **Lookup classification** in `.ml-intern/cli_agent_config.json`:
   - In `jobs_allowed` → OK, continue.
   - In `jobs_human_approval_required` → STOP. Use `AskUserQuestion` to request approval. Do not proceed without an explicit "yes."
   - Not listed → **forbidden**. Do not launch; either add to the allowlist (which itself requires the maintainer) or pick a different job.
3. **Validate the YAML statically**: `python3 scripts/check_jobs.py`. Fix any diagnostics first.
4. **Read the YAML**:
   - `hardware:` MUST be in `hardware_allowed`. NEVER `a100*` / `h100*`.
   - `timeout:` MUST be present and finite (regex-banned otherwise).
   - `env.HF_TOKEN: ${HF_TOKEN}` MUST be present.
   - `env.OTEL_EXPORTER_OTLP_ENDPOINT: ${OTEL_ENDPOINT:-}` is the canonical opt-in pattern; do not hard-code endpoints.
5. **Cost-estimate**: from `python/hf_pricing.py`, multiply hardware $/hr × timeout. Compare to remaining budget = `200 - <total in reports/cost.md>`. If the worst-case run would breach the hard cap, STOP and ask.

**Launch:**

6. State out loud, in chat: job name, hardware, timeout, estimated max cost, remaining budget, and the upload destination prefix.
7. `scripts/launch_hf_job.py jobs/<name>.yaml`.
8. Record the returned `job_id`.

**Post-flight:**

9. Append a row to `reports/cost.md` (format: see the existing rows). Include: timestamp UTC, job id, hardware, wall time, actual cost, link.
10. Run `python3 python/cost_ledger.py check --path reports/cost.md --cap-usd 200` — it must pass.
11. Re-run `make check` to confirm the gate is still green.
12. If the job produced artifacts, follow `release-and-artifacts.md` to record the upload destination (do not silently overwrite production paths).

**Editing `jobs/*.yaml`:**

- New jobs must pass `check_jobs.py` first.
- To promote a new job into `jobs_allowed`, edit `.ml-intern/cli_agent_config.json` IN A SEPARATE PR with an explicit rationale. This is gated.
</procedure>

<patterns>
<do>
— Always launch `smoke_*` first when validating a new training change end-to-end.
— Use `dry-run` patterns (`HF_TOKEN=dummy python3 python/upload_checkpoints.py … --dry-run`) for upload flow checks without spending.
— Keep job command bodies idempotent and `set -euo pipefail`'d; check existing jobs for the pattern.
— Log every launch (allowed or pre-approved) to `reports/cost.md` even if the cost ends up tiny.
</do>
<dont>
— Don't add `a100*` / `h100*` to any `jobs/*.yaml`. The CLI agent leash denies these and the regex in `command_denylist` catches drift.
— Don't omit `--timeout` in any `hf jobs run` invocation (denylist enforces).
— Don't directly write to non-`smoke/` paths under `abdelstark/lewm-rs-*` without explicit approval — these are public artifact paths.
— Don't change `python/hf_pricing.py` without re-deriving the cost ledger; the ledger references the pricing constants.
</dont>
</patterns>

<examples>
Allowed-job launch (model agent flow):

```
1. job = jobs/smoke_pusht.yaml  (in jobs_allowed)
2. python3 scripts/check_jobs.py     → ok
3. hardware = l4x1, timeout = 30m
4. l4x1 @ ~$0.80/hr × 0.5 h ≈ $0.40 max; remaining budget = $200 - $11.70 = $188.30 → OK
5. State: "Launching smoke_pusht on l4x1 (30 min timeout, ~$0.40 max); ledger has $188.30 remaining."
6. scripts/launch_hf_job.py jobs/smoke_pusht.yaml
7. Append row to reports/cost.md: 2026-05-16T19:32Z, abdelstark/<job_id>, l4x1, …
8. python3 python/cost_ledger.py check …  →  ok
```

Gated-job launch (must stop):

```
job = jobs/train_pusht.yaml  → jobs_human_approval_required
ACTION: AskUserQuestion("train_pusht.yaml requires approval. Estimated cost: <calc>. Approve launch?")
Do NOT call scripts/launch_hf_job.py until the user answers "yes."
```
</examples>

<troubleshooting>
| Symptom                                                       | Cause                                                | Fix                                                                                  |
|---------------------------------------------------------------|------------------------------------------------------|--------------------------------------------------------------------------------------|
| `check_jobs.py: missing required env HF_TOKEN`                 | YAML lacks `env.HF_TOKEN: ${HF_TOKEN}`               | Add it; do not hard-code a token                                                     |
| `command_denylist` match on `hf jobs run`                      | Missing `--timeout` or banned hardware in command    | Use `scripts/launch_hf_job.py` (it sets `--timeout` from YAML); fix hardware         |
| `cost_ledger.py check`: "cap exceeded"                         | Total $ in `reports/cost.md` > 200                   | DO NOT bypass. Stop and escalate.                                                    |
| Job timeout fires mid-training                                 | YAML `timeout` too short                             | Bump timeout in a PR; re-validate `check_jobs.py`                                    |
| Pricing in ledger looks ~3.3× too high                         | Used the old $5/hr rate for A10G                     | A10G-large is $1.50/hr per the May 2026 correction. Recompute affected rows.         |
</troubleshooting>

<references>
- `.ml-intern/cli_agent_config.json` — the leash (authoritative)
- `scripts/launch_hf_job.py` — launcher
- `scripts/check_jobs.py` — shape validation
- `scripts/check_train_so100_job.py` — SO-100 full-training contract
- `python/hf_pricing.py`, `python/cost_ledger.py`
- `reports/cost.md` — running ledger
- `specs/rfcs/0011-ci-cd-and-release-engineering.md`, `specs/rfcs/0016-security-and-supply-chain.md`
</references>
