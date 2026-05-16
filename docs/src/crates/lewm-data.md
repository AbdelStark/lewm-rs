# `lewm-data`

PushT and SO-100 dataset loaders, window sampling, image and action
transforms, and a worker-pool prefetcher.

## What it owns

- **Loaders**: `PushtDataset` and `So100Dataset`. Both consume the same
  unified HDF5 schema.
- **Transforms**: image (`f32 / 255`), action normalisation,
  window sampling.
- **Batch**: `Batch<B>` struct and collation.
- **Prefetch**: worker-pool prefetcher with bounded channel.
- **Stats**: tools to compute per-action mean/std over the train split.

## Module layout

```text
lewm-data/src/
├── lib.rs                # re-exports
├── errors.rs             # DataError
├── bin/                  # stats computation binaries
├── pusht.rs              # PushtDataset
├── so100.rs              # So100Dataset
├── transform/
│   ├── mod.rs
│   ├── image.rs          # resize / cast / normalize
│   ├── action.rs         # normalisation
│   └── window.rs         # window sampler
├── batch.rs              # collate, Batch
├── prefetch.rs           # worker pool
└── stats.rs              # mean/std over train split
```

## Public API

```rust,ignore
pub trait LewmDataset {
    type Item: Send;
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Result<Self::Item, DataError>;
}

pub struct PushtDataset<B: Backend>(...);
pub struct So100Dataset<B: Backend>(...);

pub struct Batch<B: Backend> { /* pixels, actions, metadata */ }
pub struct Prefetcher<B: Backend, D: LewmDataset> { /* ... */ }
```

## Schema

See [Data plane](../training/data.md) §1.1 for the HDF5 schema and §2
for the window-sampling contract.

## Throughput target

≥ 45 PushT samples/sec on A10G-large (NFR-010 in the PRD). The
worker-pool prefetcher with $N = 4$ workers and channel depth 8
meets this with margin.

## Dependencies

- `lewm-core` (for shape and config types)
- `hdf5-metno` (Blosc-aware HDF5 reader)
- `safetensors` (for stats files)
- `crossbeam-channel`, `rayon` (worker pool)

## Source

[`crates/lewm-data`](https://github.com/AbdelStark/lewm-rs/tree/main/crates/lewm-data)
