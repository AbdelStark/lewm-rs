# `lewm-train`

Training orchestration, checkpoint state, resume semantics, optimization, and
mixed-precision policy for `LeWM` experiments. This crate is the library
surface behind the `lewm-train` binary.

**Specs:** [RFC 0005 — training system][rfc-0005],
[RFC 0013 — determinism and reproducibility][rfc-0013],
[RFC 0017 — error model and failure handling][rfc-0017],
[RFC 0018 — configuration system][rfc-0018].

**Depends on:** `lewm-core`, `lewm-data`, `lewm-telemetry`, `lewm-hub`,
`lewm-plan`.

## Module map

- [`checkpoint`] — epoch checkpoint files, sidecars, atomic writes, pruning,
  and the Safetensors mirror.
- [`config`] — root training TOML schema, layered overrides, and the
  reproducibility hash.
- [`mixed_precision`] — precision policy, F32 islands, and bf16 autocast scope.
- [`optim`] — AdamW configuration and decay / no-decay partitioning.
- `pusht_full` — bounded config-shaped PushT `LeWM` training core.
- [`resume`] — run-directory resume detection, RNG restoration, SIGTERM
  handling, and the bit-identical resume contract.
- [`schedule`] — cosine decay with linear warmup.
- [`step`] — gradient accumulation, global-norm clipping, NaN guard, and the
  three-NaN abort path.
- [`trainer`] — outer-loop state machine and trainer artifacts.
- [`warmstart`] — SO-100 warm-start transfer policy and provenance.

## Binaries

- `lewm-train` — main entrypoint:
  - `lewm-train smoke …` — fixed-size synthetic data smoke run.
  - `lewm-train train …` — real-data training with the full artifact contract
    (run report, losses JSONL, checkpoint sidecars, `.mpk`, `.safetensors`,
    parity probe JSON).
- `lewm-reference-record` — converts a reference Safetensors checkpoint into
  the workspace's named MessagePack layout.

## Resume contract

Resume restores the model, optimizer state, scheduler target, RNG, config
hash, seed, and step. The `resume_rng_bitwise_identical` test guarantees that
a resume from any checkpoint produces the same RNG stream as an uninterrupted
run.

## Artifacts

Each run writes a deterministic artifact tree under `--output-dir`:

```
run_report.json        # provenance, config hash, peak memory, exit code
losses.jsonl           # one record per logged step
step_<N>.safetensors   # parameter snapshots
step_<N>.mpk           # Burn MessagePack module
step_<N>.sidecar.json  # optimizer + RNG state for resume
parity_probe.json      # per-step activation statistics (when configured)
```

`python/upload_checkpoints.py` mirrors this tree to the Hub.

[rfc-0005]: ../../specs/rfcs/0005-training-system.md
[rfc-0013]: ../../specs/rfcs/0013-determinism-and-reproducibility.md
[rfc-0017]: ../../specs/rfcs/0017-error-model-and-failure-handling.md
[rfc-0018]: ../../specs/rfcs/0018-configuration-system.md
