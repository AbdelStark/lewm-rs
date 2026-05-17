# `lewm-hub`

Hugging Face Hub integration: model uploads, model-card rendering, artifact
manifests, and cost-ledger enforcement. This crate owns publishing mechanics;
credential management and billing controls stay outside the crate boundary.

**Specs:** [RFC 0010 — Hugging Face Hub integration][rfc-0010],
[RFC 0016 — security and supply chain][rfc-0016].

**Depends on:** `lewm-core`.

## Module map

- `client` — `HubClient`, `HubTransport`, `EnvironmentHubTransport`, repo
  ensure/create/delete and idempotent upload APIs.
- `cost_ledger` — RFC 0010 cost-ledger parsing, append, and the cap-check
  pipeline. Mirrors the on-disk `reports/cost.md` table.
- `model_card` — model repository README rendering with stable section order.
- `upload` — SHA-256 idempotency layer and exponential-backoff retries.
- `errors` — `HubError`.

## Cost ledger

The hub crate enforces three caps in concert with
`python/cost_ledger.py check`:

| Cap          | Value | Purpose                                                |
| ------------ | ----- | ------------------------------------------------------ |
| Hard         | $200  | Workspace-wide budget over the project lifetime.       |
| Soft         | $100  | Warning threshold; surfaces as a CI annotation.        |
| Per-session  | $20   | Default `--cost-cap-usd` in `scripts/launch_hf_job.py`.|

Manual ledger edits are forbidden; the `append_entry` path is the only way to
add a row.

[rfc-0010]: ../../specs/rfcs/0010-huggingface-hub-integration.md
[rfc-0016]: ../../specs/rfcs/0016-security-and-supply-chain.md
