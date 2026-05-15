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
| 2026-05-15T11:32    | train | 6a070293e48bea4538b9e1fb | a10g-large | ~1 min  | ~$0.08 | TBD | SO-100 v5 (failed: precision fp32 invalid) |
| 2026-05-15T11:40    | train | 6a0703cf3308d79117b902aa | a10g-large | ~3 min  | ~$0.25 | TBD | SO-100 v6 (failed: --max-steps required guard) |
| 2026-05-15T11:42    | train | 6a0706a8e48bea4538b9e229 | a10g-large | ~1 min  | ~$0.08 | TBD | SO-100 v7 (failed: GHCR image stale, no SO-100 trainer) |
| 2026-05-15T11:45    | train | 6a0707653308d79117b902b4 | a10g-large | ~5 min  | ~$0.42 | TBD | SO-100 v8 (failed: cmake not installed, hdf5-metno-src fallback) |
| 2026-05-15T11:52    | train | 6a0708903308d79117b902bc | a10g-large | ~10 min | ~$0.83 | TBD | SO-100 v9 (failed: --data-dir before train subcommand, not valid for SO-100) |
| 2026-05-15T12:05    | train | 6a0709973308d79117b902c2 | a10g-large | ~14 min | ~$1.19 | TBD | SO-100 v10 COMPLETED (no upload step; artifacts lost) |
| 2026-05-15T12:14    | train | 6a070e02e48bea4538b9e2a5 | a10g-large | running | TBD | TBD | SO-100 v11a rust:bookworm + upload step (running) |
| 2026-05-15T12:19    | train | 6a070f393308d79117b902de | a10g-large | running | TBD | TBD | SO-100 v11b duplicate submission (hf jobs run blocking) |

> Prices estimated at ~$5/hr for A10G-large based on HF Jobs published pricing.
> Exact figures will be updated when jobs complete and actual billed amounts are available.
> Training budget cap: monitor HF billing dashboard; expected ceiling ~$50 total.
