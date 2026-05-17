# `lewm-data`

Data loading, preprocessing, batching, and dataset streaming for `PushT` and
`SO-100` inputs. This crate owns dataset shape normalization and per-modality
statistics; Python-only preparation stays at the repository edge (see
`python/`).

**Specs:** [RFC 0004 — data pipeline][rfc-0004],
[RFC 0012 — SO-100 real-robot extension][rfc-0012],
[RFC 0013 — determinism and reproducibility][rfc-0013].

**Depends on:** `lewm-core`.

## Module map

| Module     | Responsibility                                                            |
| ---------- | ------------------------------------------------------------------------- |
| `batch`    | `Batch`, `BatchBackend`, `BatchTensor`, collation, and the host backend.  |
| `errors`   | Crate error type (`DataError`).                                           |
| `prefetch` | Backpressure-bounded prefetcher with the `data.queue.depth` metric.       |
| `pusht`    | `PushtDataset` — HDF5-backed PushT loader (zstd / blosc / lz4 codecs).    |
| `so100`    | `So100Dataset` — LeRobot v2.1 SO-100 loader with held-out episode pin.    |
| `stats`    | `compute_stats` and `DatasetStats` for per-modality normalization.        |
| `transform`| Pixel decoding, normalization, and frame stacking.                        |

## On-disk contracts

- **PushT HDF5** (`quentinll/lewm-pusht`): chunked by episode, zstd-compressed.
  Frames are 96×96×3 RGB; actions are 2-DOF velocity targets.
- **SO-100 HDF5** (`abdelstark/so100-pickplace-lewm-ready`): 6,559 timesteps /
  50 episodes at 10 fps, 6-DOF action packing, 320×240×3 wrist + front views.
  Episodes 47–49 are pinned as held-out (`SO100_HELD_OUT_EPISODES`).

## Reproducibility

Datasets are sampled via the `data.{prefetch,shuffle,frame}` RNG sub-streams
(see RFC 0013). No `thread_rng` is used; `scripts/check_nondet.py` enforces
this.

[rfc-0004]: ../../specs/rfcs/0004-data-pipeline.md
[rfc-0012]: ../../specs/rfcs/0012-so100-real-robot-extension.md
[rfc-0013]: ../../specs/rfcs/0013-determinism-and-reproducibility.md
