# Cost ledger

> **Motivation.** Reproducibility includes *economic* reproducibility:
> how much does it cost to re-run the project end to end? This page
> summarises the ledger.
>
> **Position.** Third sub-page in [Part VII — Results](./pusht.md).
>
> **What you should leave with.** Confirmed total spend, per-run
> breakdown, and the cost cap.

## 1. Confirmed total

**\$11.70 USD**, at \$1.50 / hour for A10G-large on Hugging Face Jobs.
Source of truth:
[`reports/cost.md`](https://github.com/AbdelStark/lewm-rs/blob/main/reports/cost.md).

## 2. Per-run breakdown

| Run | Wall time | Cost |
|-----|----------:|-----:|
| SO-100 attempts + pre-training (multiple short runs) | 2.50 h | \$3.75 |
| PushT 50 k-step bounded-core run | 5.30 h | \$7.95 |
| **Total** | **7.80 h** | **\$11.70** |

The historical PushT bounded-core run dominates: a single 5.3-hour
A10G-large session accounts for two-thirds of the total spend. It is
not the F1 full Burn/Jepa release checkpoint; that approval-gated run
is still pending.

## 3. The cost cap

The project ceiling is **\$200**. Enforced by:

```sh
python3 python/cost_ledger.py check --path reports/cost.md --cap-usd 200
```

This check runs as part of `make check` and the CI gate. A run that
would push the cumulative spend over \$200 fails the gate before it
launches. At \$11.70 of \$200, we are at **5.85 % utilization**.

## 4. The breakdown by phase

| Phase | Cost | Comment |
|-------|-----:|---------|
| Parity validation (CI runs the parity workflow on every push) | \$0 | Local + CI, no cloud compute |
| Bounded smoke train (PushT) | included | Hours-of-A10G needed: ~10 min for plumbing |
| SO-100 dataset prep | \$0 | Local decode via `ffmpeg` |
| SO-100 full training | ~\$0.36 | Inside the "pre-training" bucket |
| PushT bounded-core training | \$7.95 | Historical 50 k-step run; not the F1 full Burn/Jepa release checkpoint |
| ONNX export | \$0 | Local |
| Tract benchmark | \$0 | Local |
| Hub uploads | \$0 | Hub uploads are free |
| **Total** | **\$11.70** | |

## 5. The cost-zero items

The project deliberately keeps several items at zero cost:

- **CI**: parity, conformance, docs, and specs workflows run on free
  GitHub Actions minutes; no paid runners.
- **Hub upload**: model and dataset hosting on HF is free for public
  repos.
- **Demo Space**: free Spaces tier (CPU). User-facing latency is
  higher than on a local Apple M-series but still usable.
- **Telemetry**: OTLP / Grafana / Tempo / Loki run on a self-hosted
  Docker stack (in [`infra/otel/`](https://github.com/AbdelStark/lewm-rs/blob/main/infra/otel/)).
  No cloud telemetry services.

## 6. What would be next on the budget

Remaining budgeted activities (from [`ROADMAP.md`]):

| Item | Estimated cost |
|------|---------------:|
| F1 full Burn/Jepa source-build run (PushT 50 k steps) | up to \$18 |
| F3 warm-start SO-100 training | up to \$9 |
| Planning eval (PushT, 50 episodes × CEM) | ~\$2 |
| Multi-camera SO-100 experiment | ~\$2 |
| **Total remaining** | **~\$31** |

Combined with the \$11.70 already spent: roughly \$43 total against
the \$200 cap. The cap is roomy, but F1 and F3 cannot be approved in
the same intern session because their combined \$27 worst-case envelope
exceeds the \$20 session cap in `.ml-intern/cli_agent_config.json`.

## 7. Source pointers

| Topic | Source |
|-------|--------|
| Cost ledger | `reports/cost.md` |
| Ledger gate | `python/cost_ledger.py` |
| HF Jobs pricing | `python/hf_pricing.py` |
| Job specs | `jobs/*.yaml` |

[`ROADMAP.md`]: https://github.com/AbdelStark/lewm-rs/blob/main/ROADMAP.md
