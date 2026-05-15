# `lewm-rs` cost ledger

> Updated automatically by `lewm-hub::cost_ledger::append_entry` at every job termination.
> Manual entries are forbidden; use `cost_ledger::backfill --from <job_url>` to import.

| Date (UTC)          | Phase | Job ID            | Hardware     | Wall   | Cost (USD) | Cumulative (USD) | Notes |
|---------------------|-------|-------------------|--------------|--------|-----------:|----------------:|-------|
| 2026-05-14 13:34    | smoke | 6a05cf0ee48bea4538b9ccd6 | a10g-large | ~10 min | ~$0.83 | ~$0.83 | Short PushT smoke run |
| 2026-05-14T17:xx    | train | 6a06ef5d3308d79117b9025b | a10g-large | ~50 min | ~$4.17 | ~$5.00 | Full PushT training attempt v1 |
| 2026-05-15T09:27    | train | 6a06f0c43308d79117b90276 | a10g-large | running | TBD | TBD | Full PushT 50k steps (in progress) |
| 2026-05-15T11:05    | train | 6a06fe17e48bea4538b9e1cb | a10g-large | ~1 min  | ~$0.08 | TBD | SO-100 v1 (failed: rustup path) |
| 2026-05-15T11:06    | train | 6a0700da3308d79117b9029c | a10g-large | ~2 min  | ~$0.17 | TBD | SO-100 v2 (failed: cargo not found) |
| 2026-05-15T11:06    | train | 6a0701143308d79117b9029e | a10g-large | ~2 min  | ~$0.17 | TBD | SO-100 v3 (failed: HDF5 path) |
| 2026-05-15T11:07    | train | 6a0701b0e48bea4538b9e1f5 | a10g-large | ~4 min  | ~$0.33 | TBD | SO-100 v4 (failed: TOML quoting) |
| 2026-05-15T11:32    | train | 6a070293e48bea4538b9e1fb | a10g-large | running | TBD | TBD | SO-100 v5 10 epochs (in progress) |

> Prices estimated at ~$5/hr for A10G-large based on HF Jobs published pricing.
> Exact figures will be updated when jobs complete and actual billed amounts are available.
> Training budget cap: monitor HF billing dashboard; expected ceiling ~$50 total.
