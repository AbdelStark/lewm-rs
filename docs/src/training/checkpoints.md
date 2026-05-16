# Checkpointing and crash-resume

> **Motivation.** A 5-hour PushT training run that loses its progress
> on the 4-hour mark is not a result — it is a re-run waiting to
> happen. This page documents the checkpoint format, the cadence, and
> the resume protocol.
>
> **Position.** Eighth sub-page in [Part III](./pipeline.md).
>
> **What you should leave with.** Which files are written when, how
> resume works, and how the `.mpk` vs `.safetensors` split is used.

## 1. The checkpoint quartet

Every checkpoint is **four files**, all in the run's output dir:

| File | Contents |
|------|----------|
| `step_{N}.mpk` | Full Burn record: model parameters + optimizer state (Adam moments) + scheduler step counter + RNG state. Binary, MessagePack-serialized. |
| `step_{N}.safetensors` | Parameter mirror: model weights only, in Safetensors format. Compatible with the upstream PyTorch loader. |
| `step_{N}.json` | Sidecar metadata: config hash, git SHA, step, seed, RNG state (mirror of `.mpk`), wall-time, loss window, hardware. |
| `step_{N}.parity.json` | Optional: per-checkpoint parity probe against a fixed fixture (used by the `train --parity` flag). |

Why both `.mpk` and `.safetensors`? `.mpk` is the resume-capable
format — it has optimizer state. `.safetensors` is the
upstream-compatible format — it is what the ONNX exporter and the HF
Hub model card consume. Saving both costs ~50 MB extra disk per
checkpoint and saves a conversion step.

## 2. Cadence

The default checkpoint cadence is **every 500 steps** (PushT) or
**every 250 steps** (SO-100), plus an unconditional save at the end of
the run. The exact value is configurable via the TOML:

```toml
[checkpoint]
every_steps = 500
keep_last = 3       # rolling window; older checkpoints are deleted
save_final = true
```

The `keep_last` knob retains only the most recent N checkpoints (plus
the final). This keeps disk usage bounded over long runs.

## 3. Atomic writes

Checkpoints are written atomically using the temp-and-rename pattern:

```rust,ignore
fn save_atomic(path: &Path, content: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, content)?;
    std::fs::File::open(&tmp)?.sync_all()?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}
```

The four files are written in order: `.mpk`, `.safetensors`, `.parity.json`
(if present), and **finally** `.json`. A crash mid-write leaves the
sidecar absent or stale, so a resume that inspects the sidecar can
detect incomplete writes and fall back to the previous checkpoint.

## 4. Resume

```sh
lewm-train --config configs/pusht.toml \
    --output-dir /scratch/pusht-run-1 \
    --resume-if-present \
    train
```

On `--resume-if-present`, the trainer:

1. Lists checkpoints in `--output-dir` matching `step_{N}.json`.
2. Picks the highest $N$ whose four-file quartet is complete and
   consistent (config hash matches, sidecar parseable).
3. Restores model + optimizer + scheduler + RNG from `.mpk`.
4. Verifies sidecar fields: config hash matches current config; seed
   matches; git SHA matches (warning if not).
5. Sets the run state to the one recorded in the sidecar and continues.

If no checkpoint is found, the run starts from step 0. If a checkpoint
is found but the config has changed (different hash), the run aborts
with a clear error.

## 5. Bit-identical resume

Pinned by [RFC 0013] §RFC0013-001 and verified by
`crates/lewm-train/tests/resume_parity.rs`:

> Resuming from a checkpoint at step $N$ produces bit-identical losses
> at step $N+1, N+2, \dots$ as a fresh run that reached step $N$
> without interruption.

This requires that the resume protocol restores **every** piece of
training state:

- Model parameters ✔ (`.mpk`)
- AdamW moments ✔ (`.mpk`)
- Scheduler step ✔ (`.mpk` + sidecar)
- RNG sub-streams: master, dataset_sample, dataset_worker.\*,
  sigreg_sketch, cem (eval), … ✔ (sidecar)
- Dataset prefetcher state: not strictly required since the worker
  RNGs are seeded by name + step; the prefetch order is implicitly
  reset to be consistent with the resumed step.

## 6. The `.mpk` format

Burn's `.mpk` is a MessagePack-serialized record. The schema is
hierarchical, mirroring the module tree:

```text
{
  "vit": { "embeddings": {...}, "blocks": [{...}, ...], "final_norm": {...} },
  "predictor": { "input_proj": {...}, "blocks": [...], ... },
  "action_enc": { "smoother": {...}, "fc1": {...}, "fc2": {...} },
  "projector": { "fc1": {...}, "fc2": {...} },
  "pred_proj": { "fc1": {...}, "fc2": {...} },
  "optimizer": {
    "state": [ {"m": [...], "v": [...], "step": ...}, ... ],   # per-param Adam state
    "config": { "beta1": 0.9, "beta2": 0.95, ... }
  },
  "scheduler": { "step": 12500, "warmup": 1000, "max": 50000, "lr_max": 3e-4, "lr_min": 1e-5 },
  "rng": { "master": "<base64>", "sigreg_sketch": "<base64>", ... }
}
```

The format is binary and not human-readable, but the sidecar JSON
duplicates the relevant scalar fields for inspection.

## 7. The `.safetensors` format

For Hub compatibility and for the ONNX exporter, the same model weights
are also written in Safetensors format. The Safetensors file uses the
parameter names from `python/param_name_map.py`, *not* the Burn-style
paths — i.e., they look like

```text
vit.embeddings.patch_embed.proj.weight
vit.embeddings.cls_token
vit.blocks.0.attention.qkv.weight
...
```

This makes the file load directly via `safetensors.torch.load_file`
into a PyTorch ViT for cross-checks (and into `python/export_onnx.py`
for ONNX export).

## 8. Source pointers

| Topic | Source |
|-------|--------|
| Save / load | `crates/lewm-train/src/checkpoint.rs` |
| Resume protocol | `crates/lewm-train/src/resume.rs` |
| Safetensors export | `crates/lewm-core/src/export/safetensors.rs` |
| Burn record (`.mpk`) | `burn::record::DefaultRecorder` |
| Bit-identical resume test | `crates/lewm-train/tests/resume_parity.rs` |

[RFC 0013]: ../reference/rfcs.md
