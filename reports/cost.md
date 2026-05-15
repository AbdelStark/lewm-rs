# `lewm-rs` cost ledger

| Date (UTC)          | Phase | Job ID            | Hardware     | Wall     | Cost (USD) | Cumulative (USD) | Notes |
|---------------------|-------|-------------------|--------------|----------|-----------:|-----------------:|-------|
| 2026-05-14 11:42    | train | —                 | a10g-large   | ~5 min   | ~$0.13     | ~$0.13           | PushT short run (pusht-short-20260514T114211Z; job ID not recorded) |
| 2026-05-14 12:39    | train | —                 | a10g-large   | ~5 min   | ~$0.13     | ~$0.26           | PushT tiny-jepa short run (pusht-tiny-jepa-short-20260514T123936Z; job ID not recorded) |
| 2026-05-14 12:42    | train | —                 | a10g-large   | ~5 min   | ~$0.13     | ~$0.39           | PushT tiny-jepa short run (pusht-tiny-jepa-short-20260514T124237Z; job ID not recorded) |
| 2026-05-14 13:34    | smoke | 6a05cf0ee48bea4538b9ccd6 | a10g-large | ~10 min  | ~$0.25     | ~$0.64           | PushT minimal-lewm-short smoke (pusht-minimal-lewm-short-20260514T133423Z) |
| 2026-05-14T17:xx    | train | 6a06ef5d3308d79117b9025b | a10g-large | ~50 min  | ~$1.25     | ~$1.89           | Full PushT training attempt v1 (aborted early) |
| 2026-05-15T09:27    | train | 6a06f0c43308d79117b90276 | a10g-large | running (>4h) | TBD   | TBD              | Full PushT 50k steps (still running as of 2026-05-15T14:30 UTC) |
| 2026-05-15T11:05    | train | 6a06fe17e48bea4538b9e1cb | a10g-large | ~1 min   | ~$0.03     | TBD              | SO-100 v1 (failed: rustup path) |
| 2026-05-15T11:06    | train | 6a0700da3308d79117b9029c | a10g-large | ~2 min   | ~$0.05     | TBD              | SO-100 v2 (failed: cargo not found) |
| 2026-05-15T11:06    | train | 6a0701143308d79117b9029e | a10g-large | ~2 min   | ~$0.05     | TBD              | SO-100 v3 (failed: HDF5 path) |
| 2026-05-15T11:07    | train | 6a0701b0e48bea4538b9e1f5 | a10g-large | ~4 min   | ~$0.10     | TBD              | SO-100 v4 (failed: TOML quoting) |
| 2026-05-15T11:32    | train | 6a070293e48bea4538b9e1fb | a10g-large | ~1 min   | ~$0.03     | TBD              | SO-100 v5 (failed: precision fp32 invalid) |
| 2026-05-15T11:40    | train | 6a0703cf3308d79117b902aa | a10g-large | ~3 min   | ~$0.08     | TBD              | SO-100 v6 (failed: --max-steps required guard) |
| 2026-05-15T11:42    | train | 6a0706a8e48bea4538b9e229 | a10g-large | ~1 min   | ~$0.03     | TBD              | SO-100 v7 (failed: GHCR image stale, no SO-100 trainer) |
| 2026-05-15T11:45    | train | 6a0707653308d79117b902b4 | a10g-large | ~5 min   | ~$0.13     | TBD              | SO-100 v8 (failed: cmake not installed, hdf5-metno-src fallback) |
| 2026-05-15T11:52    | train | 6a0708903308d79117b902bc | a10g-large | ~10 min  | ~$0.25     | TBD              | SO-100 v9 (failed: --data-dir before train subcommand) |
| 2026-05-15T12:05    | train | 6a0709973308d79117b902c2 | a10g-large | ~14 min  | ~$0.35     | TBD              | SO-100 v10 COMPLETED (no upload step; artifacts lost) |
| 2026-05-15T12:14    | train | 6a070e02e48bea4538b9e2a5 | a10g-large | 864s     | ~$0.38     | TBD              | SO-100 v11a COMPLETED; artifacts at `abdelstark/lewm-rs-so100/train/so100-full-20260515T122820Z/` |
| 2026-05-15T12:19    | train | 6a070f393308d79117b902de | a10g-large | 860s     | ~$0.38     | TBD              | SO-100 v11b COMPLETED; duplicate (same config as v11a) |

**Known spend (excluding PushT full run):** ~$3.25  
**PushT full run (50k steps, A10G-large, 12h timeout):** TBD — estimated $6–$12 depending on wall time  
**Projected total:** ~$10–$15  
**Budget cap:** $50 (monitor HF billing dashboard)

> Pricing: HuggingFace Jobs a10g-large = $1.50/hr (per published rate sheet as of 2026-05).
> Costs rounded up to the nearest minute, then to the nearest cent.
> Three pre-2026-05-15 PushT short runs lack job IDs (artifact timestamps used as proxy dates).
