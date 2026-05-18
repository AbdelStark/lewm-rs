# `lewm-rs` cost ledger

> Updated automatically by `lewm-hub::cost_ledger::append_entry` at every job termination.
> Manual entries are forbidden; use `cost_ledger::backfill --from <job_url>` to import.

| Date (UTC)          | Phase | Job ID            | Hardware     | Wall   | Cost (USD) | Cumulative (USD) | Notes |
|---------------------|-------|-------------------|--------------|--------|-----------:|----------------:|-------|
| 2026-05-14 11:42    | train | —                 | a10g-large   | 5m     | 0.13       | 0.13            | PushT short run (pusht-short-20260514T114211Z; job ID not recorded) |
| 2026-05-14 12:39    | train | —                 | a10g-large   | 5m     | 0.13       | 0.26            | PushT tiny-jepa short run (pusht-tiny-jepa-short-20260514T123936Z; job ID not recorded) |
| 2026-05-14 12:42    | train | —                 | a10g-large   | 5m     | 0.13       | 0.39            | PushT tiny-jepa short run (pusht-tiny-jepa-short-20260514T124237Z; job ID not recorded) |
| 2026-05-14 13:34    | smoke | 6a05cf0ee48bea4538b9ccd6 | a10g-large | 10m    | 0.25       | 0.64            | PushT minimal-lewm-short smoke (pusht-minimal-lewm-short-20260514T133423Z) |
| 2026-05-14 17:xx    | train | 6a06ef5d3308d79117b9025b | a10g-large | 50m    | 1.25       | 1.89            | PushT bounded-module training attempt v1 (aborted early) |
| 2026-05-15 11:05    | train | 6a06fe17e48bea4538b9e1cb | a10g-large | 1m     | 0.03       | 1.92            | SO-100 v1 (failed: rustup path) |
| 2026-05-15 11:06    | train | 6a0700da3308d79117b9029c | a10g-large | 2m     | 0.05       | 1.97            | SO-100 v2 (failed: cargo not found) |
| 2026-05-15 11:06    | train | 6a0701143308d79117b9029e | a10g-large | 2m     | 0.05       | 2.02            | SO-100 v3 (failed: HDF5 path) |
| 2026-05-15 11:07    | train | 6a0701b0e48bea4538b9e1f5 | a10g-large | 4m     | 0.10       | 2.12            | SO-100 v4 (failed: TOML quoting) |
| 2026-05-15 11:32    | train | 6a070293e48bea4538b9e1fb | a10g-large | 1m     | 0.03       | 2.15            | SO-100 v5 (failed: precision fp32 invalid) |
| 2026-05-15 11:40    | train | 6a0703cf3308d79117b902aa | a10g-large | 3m     | 0.08       | 2.23            | SO-100 v6 (failed: --max-steps required guard) |
| 2026-05-15 11:42    | train | 6a0706a8e48bea4538b9e229 | a10g-large | 1m     | 0.03       | 2.26            | SO-100 v7 (failed: GHCR image stale, no SO-100 trainer) |
| 2026-05-15 11:45    | train | 6a0707653308d79117b902b4 | a10g-large | 5m     | 0.13       | 2.39            | SO-100 v8 (failed: cmake not installed, hdf5-metno-src fallback) |
| 2026-05-15 11:52    | train | 6a0708903308d79117b902bc | a10g-large | 10m    | 0.25       | 2.64            | SO-100 v9 (failed: --data-dir before train subcommand) |
| 2026-05-15 12:05    | train | 6a0709973308d79117b902c2 | a10g-large | 14m    | 0.35       | 2.99            | SO-100 v10 COMPLETED (no upload step; artifacts lost) |
| 2026-05-15 12:14    | train | 6a070e02e48bea4538b9e2a5 | a10g-large | 15m    | 0.38       | 3.37            | SO-100 v11a COMPLETED; artifacts at abdelstark/lewm-rs-so100/train/so100-full-20260515T122820Z/ |
| 2026-05-15 12:19    | train | 6a070f393308d79117b902de | a10g-large | 15m    | 0.38       | 3.75            | SO-100 v11b COMPLETED; duplicate (same config as v11a) |
| 2026-05-15 10:09    | train | 6a06f0c43308d79117b90276 | a10g-large | 318m   | 7.95       | 11.70           | PushT bounded-module 50k steps COMPLETED; artifacts at abdelstark/lewm-rs-pusht/train/pusht-full-lewm-20260515T100908Z/ |

> Pricing: HuggingFace Jobs a10g-large = $1.50/hr (per published rate sheet as of 2026-05).
> Costs rounded up to the nearest minute, then to the nearest cent.
> Three pre-2026-05-15 PushT short runs lack job IDs (artifact timestamps used as proxy dates).
> PushT wall time estimated from job created_at (10:09:08 UTC) to artifact upload timestamp (15:26:43 UTC) = 318 min.
