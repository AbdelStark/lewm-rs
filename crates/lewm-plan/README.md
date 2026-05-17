# `lewm-plan`

Planning and evaluation primitives, including `CEM` action search, `PushT`
planning evaluation, and `SO-100` latent trajectory metrics. This crate stays
separate from training and Tract inference runners.

**Specs:** [RFC 0006 — planning and evaluation][rfc-0006],
[RFC 0012 — SO-100 real-robot extension][rfc-0012].

**Depends on:** `lewm-core`, `lewm-data`, `lewm-telemetry`.

## Module map

- [`cem`] — Cross Entropy Method action search for normalized action
  sequences. The `Cem` planner is generic over a `CemCostModel`; the cost is
  computed in chunks to keep peak memory bounded.
- [`pusht_eval`] — `PushT` eval loop and the simulator RPC boundary
  (`PushtRpc`, `SubprocessPushtRpc`, `MockPushtRpc`). The CLI in
  `lewm-infer eval` drives this.
- [`reports`] — eval artifact rendering (JSON + Markdown) and persistence.
- [`so100_eval`] — RFC 0006 / 0012 SO-100 latent-rollout metric contract,
  used to score the warm-start ablation.

## CEM defaults

| Parameter         | Value                  | Notes                                |
| ----------------- | ---------------------- | ------------------------------------ |
| `chunk_size`      | 256                    | `DEFAULT_CEM_CHUNK_SIZE`             |
| `max_batch_bytes` | 1 GiB                  | `DEFAULT_CEM_MAX_BATCH_BYTES`        |
| RNG stream        | `CEM_RNG_STREAM`       | sub-stream key per RFC 0013          |

## Evaluation contract

`PushtEvaluator` rolls out 50 test episodes and records:
- success rate (target `>= 87 %` for the reference PushT checkpoint),
- mean episode reward,
- per-episode trajectory summaries,
- planning wall-clock and CEM iteration cost histograms.

Reports are emitted as both `eval.json` (machine-readable) and `eval.md`
(human-readable) under the run directory.

[rfc-0006]: ../../specs/rfcs/0006-planning-and-evaluation.md
[rfc-0012]: ../../specs/rfcs/0012-so100-real-robot-extension.md
