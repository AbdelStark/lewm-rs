# Quickstart

> **Motivation.** Get from `git clone` to "I have run a forward pass"
> in under five minutes.
>
> **Position.** Top of [Part IX — Reproducing the results](./quickstart.md).

## 1. Prerequisites

- **Rust toolchain.** Pinned to `1.95.0` in `rust-toolchain.toml`.
  `rustup` will install it automatically when you `cargo` anything in
  the repo.
- **Python 3.13** for the Python helpers and ONNX export. A `pyproject.toml`
  in `python/` declares the dependencies; install with
  `pip install -e python/`.
- **HDF5 + Blosc plugin** for the PushT dataset:
  `pip install hdf5plugin` and set
  `HDF5_PLUGIN_PATH=$(python -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")`.
- **A GPU is *not* required for parity tests or smoke runs.** Full
  training requires CUDA + Burn's `burn-cuda` feature.

## 2. Clone and inspect

```sh
git clone https://github.com/AbdelStark/lewm-rs.git
cd lewm-rs
rustup show active-toolchain
ls
```

## 3. Run the workspace gate

```sh
cargo check --workspace --locked
python3 scripts/check_specs.py
python3 scripts/check_layers.py
```

`cargo check` confirms the workspace compiles. The two Python scripts
verify the specs are consistent (RFCs cross-reference correctly, the
traceability matrix is up to date) and the layer-dependency map is
sane (INV-003 holds).

## 4. Build the release binaries

```sh
cargo build --release --workspace
```

This produces:

- `target/release/lewm-train`
- `target/release/lewm-eval`
- `target/release/lewm-infer`

Release builds are required for the inference benchmark to be
meaningful. Debug builds compile faster but Tract latency is
essentially the same (the hot path is pre-compiled).

## 5. Run a 50-step smoke train

```sh
cargo run --release -p lewm-train -- \
    --config configs/pusht.toml \
    --device cpu \
    --output-dir /tmp/lewm-smoke \
    smoke --steps 50 --batch-size 4
```

This exercises the data path, the forward, the backward, and the
checkpoint writer in 30–60 seconds on a modern laptop. Output:

```text
/tmp/lewm-smoke/
├── step_0000050.mpk
├── step_0000050.safetensors
├── step_0000050.json
└── train_losses.jsonl
```

Inspect `train_losses.jsonl` to see the per-step loss decrease.

## 6. Run a 10-step real training run

```sh
cargo run --release -p lewm-train -- \
    --config configs/pusht.toml \
    --device cpu \
    --output-dir /tmp/lewm-train-pusht \
    --max-steps 10 train
```

This runs the actual training path (not the bounded smoke), but
truncated to 10 steps so you can see the full pipeline complete
locally without GPU or HDF5 data. If `quentinll/lewm-pusht` is not
mirrored locally, the trainer falls back to a fixture-marked path so
plumbing checks still work.

## 7. Run the parity tests

```sh
make py-lint && make check
```

Or, focused on the parity crate:

```sh
HF_TOKEN=... python python/convert_reference.py --download --out refs/
HF_TOKEN=... python python/convert_reference.py dump --all --out dumps/

LEWM_REFERENCE_SAFETENSORS=$(pwd)/refs/pusht.safetensors \
LEWM_PARITY_DUMPS=$(pwd)/dumps \
cargo test -p lewm-core --features parity-fixtures
```

All 10 tests should pass with $L_\infty < 10^{-4}$.

## 8. Run the inference benchmark

```sh
target/release/lewm-infer bench \
    --checkpoint-dir <local clone of abdelstark/lewm-rs-pusht/tract-compat/> \
    --history-steps 3 \
    --action-dim 10 \
    --episodes 10
```

Median latency should land at ~4 s/episode on Apple M-series. See
[Benchmarks](../inference/benchmark.md) for the table.

## 9. Next steps

- For the full PushT or SO-100 training runs, see
  [Reproducing PushT training](./training-pusht.md) or
  [Reproducing SO-100 training](./training-so100.md).
- For container-based reproduction via Hugging Face Jobs, see
  [Docker and HF Jobs](./docker.md).
- For the local quality gate run by maintainers, see
  [Local quality gate](./quality-gate.md).
