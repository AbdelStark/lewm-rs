# The training state machine

> **Motivation.** Long training runs need to be observable, resumable,
> and gated. The state machine in [RFC 0005 §6] gives the run a coarse
> structure with explicit transitions, each one auditable.
>
> **Position.** Sixth sub-page in [Part III](./pipeline.md).
>
> **What you should leave with.** The 8 states, the transition gates,
> and which CLI subcommand drives each state.

## 1. The states

```text
   INIT  →  PARITY_CHECK  →  SMOKE  →  WARMUP  →  STEADY  →  COOLDOWN  →  EVAL  →  UPLOAD  →  DONE
                ▲                                                                                │
                └────────────────────────────────────────────────────────────────────────────────┘
                            crash-resume: state restored from sidecar
```

| State | What runs |
|-------|-----------|
| **INIT** | Config load, model + optimizer + scheduler construct, dataset open. |
| **PARITY_CHECK** | All 10 activation-level parity tests against the locked fixture. Aborts on first failure. |
| **SMOKE** | 50-step CPU smoke train; verifies shape and convergence direction. |
| **WARMUP** | Optimizer steps 1..`warmup`; LR rises linearly from 0 to peak. |
| **STEADY** | Steps `warmup`..`max_steps - cooldown`; main cosine descent. |
| **COOLDOWN** | Steps in the cosine tail; LR approaches `lr_min`. |
| **EVAL** | Run `lewm-eval` against the held-out test set; compute success rate / latent MSE. |
| **UPLOAD** | Push final artifacts to Hugging Face Hub; write cost ledger entry. |
| **DONE** | Terminal. |

## 2. The transitions

Every transition is **gated**. A gate failure aborts the run with a
structured diagnostic.

| Transition | Gate predicate |
|------------|----------------|
| `INIT → PARITY_CHECK` | Config loaded, all sub-modules constructed without panic. |
| `PARITY_CHECK → SMOKE` | All 10 parity tests pass on the locked fixture. |
| `SMOKE → WARMUP` | 50-step smoke completes; step-1 loss > step-50 loss (sanity). |
| `WARMUP → STEADY` | `scheduler.step >= warmup_steps`. |
| `STEADY → COOLDOWN` | `scheduler.step >= max_steps - cooldown_steps` (cooldown is the final 5 % of steps by default). |
| `COOLDOWN → EVAL` | `scheduler.step >= max_steps`; final checkpoint saved. |
| `EVAL → UPLOAD` | Eval report written, success-rate / latent-MSE recorded, no eval-time failures. |
| `UPLOAD → DONE` | All checkpoints, sidecars, JSONL, ONNX, and model card present in the Hub repo. |

## 3. The CLI mapping

`lewm-train` CLI subcommands drive the state machine in different
modes:

| Subcommand | Runs states |
|------------|-------------|
| `lewm-train smoke` | INIT → PARITY_CHECK → SMOKE → DONE |
| `lewm-train parity` | INIT → PARITY_CHECK → DONE |
| `lewm-train train` | INIT → PARITY_CHECK → SMOKE → WARMUP → STEADY → COOLDOWN → DONE |
| `lewm-train train --eval` | INIT → … → COOLDOWN → EVAL → DONE |
| `lewm-train train --upload` | INIT → … → EVAL → UPLOAD → DONE |
| `lewm-train eval` | EVAL only, on a given checkpoint. |
| `lewm-train convert` | One-shot HF/PyTorch → Burn record conversion. |

`--resume-if-present` flag restores from the latest complete checkpoint
in `--output-dir` and continues from the recorded state.

## 4. The sidecar schema

Every checkpoint writes a `step_{N}.json` sidecar. The schema (v1.0.0):

```json
{
  "schema_version": "1.0.0",
  "step": 12500,
  "state": "STEADY",
  "wall_time_s": 12345.6,
  "config_hash": "438eb30f4bb0",
  "git_sha": "<commit>",
  "seed": 0,
  "rng_state": {
    "master":         "<base64>",
    "dataset_sample": "<base64>",
    "sigreg_sketch":  "<base64>",
    "...":             "..."
  },
  "lr_now": 1.5e-4,
  "loss_window_recent_100": { "total": 6.09e-06, "pred": 1.13e-06, "sigreg": 4.96e-06 },
  "grad_norm_pre_clip": 4.97e-03,
  "epoch_progress": 0.25,
  "hardware": { "device": "cuda:0", "host": "ip-10-..." }
}
```

The sidecar is the **resume contract**: every field is read at resume
time and used to restore the run to bit-identical state.

## 5. Atomicity

Checkpoint writes are atomic. The pattern in
`crates/lewm-train/src/checkpoint.rs`:

```text
write step_{N}.mpk.tmp           # full Burn record
write step_{N}.safetensors.tmp   # parameter mirror
write step_{N}.json.tmp          # sidecar
fsync each
mv step_{N}.mpk.tmp        step_{N}.mpk           # atomic rename
mv step_{N}.safetensors.tmp step_{N}.safetensors
mv step_{N}.json.tmp       step_{N}.json
```

The sidecar is renamed *last*, so a resume that finds a sidecar can
safely assume the `.mpk` and `.safetensors` siblings are also complete.

## 6. The cost-cap predicate

The cost ledger ([`reports/cost.md`](https://github.com/AbdelStark/lewm-rs/blob/main/reports/cost.md))
caps total training spend at \$200. `make check` (and the CI `cost`
workflow) verifies this every run via `python/cost_ledger.py check
--path reports/cost.md --cap-usd 200`. A run that would push total
spend over \$200 fails the gate before it launches.

Current ledger: \$11.70 total. Cap is comfortably out of reach.

## 7. Source pointers

| Topic | Source |
|-------|--------|
| State enum, transitions | `crates/lewm-train/src/trainer.rs` (see also the conceptual `state.rs` description in RFC 0005 §3) |
| Sidecar schema | `crates/lewm-train/src/checkpoint.rs` |
| Resume protocol | `crates/lewm-train/src/resume.rs` |
| CLI | `crates/lewm-train/src/bin/lewm-train.rs` |
| Cost ledger | `python/cost_ledger.py` |

[RFC 0005 §6]: https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0005-training-system.md
