# Data plane and window sampling

> **Motivation.** Data pipelines are the second-most-common source of
> silent ML bugs, after loss math. This page documents the window
> sampling, normalisation, and batching contracts pinned by [RFC 0004].
>
> **Position.** First sub-page in [Part III](./pipeline.md).
>
> **What you should leave with.** Why HDF5, how windows are sampled
> across episode boundaries, and what one `Batch` looks like.

## 1. The two datasets

| Dataset | On-disk format | Frames | Action dim |
|---------|----------------|-------:|-----------:|
| PushT (`quentinll/lewm-pusht`) | HDF5 (Blosc-compressed pixels) | ~2 092 000 windows | 2 |
| SO-100 (`abdelstark/so100-pickplace-lewm-ready`) | HDF5 (raw 224×224 RGB) | ~6 559 frames / 50 episodes | 6 |

Both share the same internal Rust loader (`crates/lewm-data`) and the
same HDF5 layout, after the SO-100 raw Parquet+MP4 is pre-decoded by
`python/decode_so100_to_h5.py`. The unified loader has *one* schema and
*two* dataset façades.

### 1.1 HDF5 layout

```text
file.h5
├── /episodes/0/
│   ├── pixels      (T_0, 3, 224, 224) uint8
│   ├── actions     (T_0, A)            float32 (normalised)
│   ├── start_idx   scalar              int64   (global index of first frame)
│   └── end_idx     scalar              int64   (one past last)
├── /episodes/1/...
├── /stats/
│   ├── action_mean (A,)  float32
│   └── action_std  (A,)  float32
└── /meta/
    ├── num_episodes  int64
    ├── total_frames  int64
    └── fps           float32
```

PushT additionally stores pixels under the Blosc HDF5 filter; the
`HDF5_PLUGIN_PATH` env var must point at the Python `hdf5plugin`
package's `plugin_dir`. This is wired into the Dockerfile and the
Burn-side loader.

## 2. Window sampling

A *window* is a $T + 1$ consecutive frames from one episode (the
predictor's $T$-frame history plus the $T{+}1$-th frame that serves as
the next-step target), together with the $T_{\text{raw}}$ raw action
frames that pack into the $T$ packed-action vectors the encoder
consumes. With `frameskip = 5`, every packed action is a concatenation
of 5 consecutive raw action vectors, so $T_{\text{raw}} = T \cdot
\text{frameskip}$. For $T = 3$ (PushT and SO-100 default) the raw
action window has $T_{\text{raw}} = 15$ frames; after packing, the
encoder receives a $(T, A_p) = (3, 10)$ stream. The sampler:

1. Picks an episode index uniformly at random.
2. Picks a start frame uniformly at random in
   `[0, episode_len − max(T+1, T_raw)]`, guaranteeing the window fits
   inside the episode without crossing into the next.
3. Returns `(pixels, actions)` of shapes $(T+1, 3, 224, 224)$ and
   $(T_{\text{raw}}, A)$; the trainer packs the action stream to
   $(T, A_p = A \cdot \text{frameskip})$ before handing it to the
   encoder.

**RFC0004-001 [MUST]** — Windows never cross episode boundaries. The
"next frame" used as the prediction target must come from the same
episode as the history; otherwise the predictor is asked to predict
across a reset, which it cannot do and which would corrupt the
gradient.

**RFC0004-002 [MUST]** — Window sampling uses the named RNG sub-stream
`rng:dataset_sample`. This makes window choices reproducible across
runs with the same seed.

## 3. Image preprocessing

```text
raw uint8 (T+1, 3, 224, 224)
   │
   ▼
cast to f32                  → (T+1, 3, 224, 224)
   │
   ▼
divide by 255.0              → values in [0, 1]
```

No mean/std normalisation, no augmentation, no random crops. The PushT
upstream pipeline operates in $[0, 1]$ pixel space; lewm-rs matches.

## 4. Action normalisation

Raw actions are stored *normalised* in the HDF5 file. The normalisation
is

$$
\mathbf a_t^{\text{normed}} \;=\; \frac{\mathbf a_t^{\text{raw}} - \boldsymbol\mu_a}{\boldsymbol\sigma_a},
$$

with $\boldsymbol\mu_a, \boldsymbol\sigma_a$ computed once over the
training split by `python/compute_stats.py` (or
`compute_so100_stats.py`). The stats are stored alongside the dataset:

```text
/stats/action_mean  (A,) float32
/stats/action_std   (A,) float32
```

For inference, the same stats must be applied to raw actions before
they enter the encoder. The runner ships the stats file
(`stats.safetensors`) along with the ONNX graphs.

## 5. The `Batch` struct

After sampling, prefetching, and collation, what `lewm-train` consumes
is a `Batch`:

```rust,ignore
#[derive(Debug, Clone)]
pub struct Batch<B: Backend> {
    /// Per-window pixels, in float32 / value range [0, 1].
    /// Shape: (B, T+1, 3, 224, 224).
    pub pixels: Tensor<B, 5>,

    /// Per-window raw actions, normalised. The trainer packs
    /// `frameskip` consecutive raw actions into one packed action
    /// before handing the stream to the encoder.
    /// Shape: (B, T_raw, A) with T_raw = T * frameskip.
    pub actions: Tensor<B, 3>,

    /// Original episode index and start frame per sample, for debugging.
    pub episode_ids: Vec<i64>,
    pub start_frames: Vec<i64>,
}
```

The collate function in `crates/lewm-data/src/batch.rs` stacks per-sample
tensors into batch tensors along axis 0. There is no per-sample padding;
all windows are the same length $T + 1$.

## 6. The prefetch pipeline

To keep the GPU fed at the throughput target of ≥ 45 PushT samples/s on
A10G-large (NFR-010), the loader uses a worker-pool prefetcher in
`crates/lewm-data/src/prefetch.rs`:

```text
   ┌──────────┐    sample idx     ┌─────────────┐    Batch    ┌─────────┐
   │ sampler  │─────────────────▶│ worker × N  │────────────▶│ channel │─▶ trainer
   │ (1 thread)│                  │ (read HDF5, │             │ depth=8 │
   └──────────┘                   │  decode,    │             └─────────┘
                                  │  cast)      │
                                  └─────────────┘
```

- **N workers** parallelise the HDF5 read + cast pipeline. With Blosc
  decompression on the PushT critical path, $N = 4$ is the sweet spot
  on A10G-large.
- **Channel depth 8** smooths the variance in per-window read latency
  so the trainer never starves.
- **Workers seed independently from the named substream
  `rng:dataset_worker.{worker_id}`** so the global window sequence is
  reproducible (RFC 0013 §4).

## 7. The error taxonomy

`crates/lewm-data/src/errors.rs` defines `DataError` with variants:

- `HdfRead(io_error)` — HDF5 backend returned an error.
- `WindowOutOfBounds { episode, start, len }` — sampler attempted to
  read past the episode boundary. Never expected in a correct sampler.
- `ShapeMismatch { expected, got, where }` — dataset content does not
  match the schema.
- `StatsMismatch { expected_dim, got_dim }` — action stats file
  disagrees with the dataset's action dimension.

Each error type maps to a structured log line and (if OTLP is enabled)
a span event with the appropriate severity.

## 8. Source pointers

| Topic | Source |
|-------|--------|
| HDF5 schema | `crates/lewm-data/src/schema.rs` |
| PushT loader | `crates/lewm-data/src/pusht.rs` |
| SO-100 loader | `crates/lewm-data/src/so100.rs` |
| Window sampler | `crates/lewm-data/src/transform/window.rs` |
| Image transform | `crates/lewm-data/src/transform/image.rs` |
| Action transform | `crates/lewm-data/src/transform/action.rs` |
| Batch / collate | `crates/lewm-data/src/batch.rs` |
| Prefetcher | `crates/lewm-data/src/prefetch.rs` |
| Stats compute | `crates/lewm-data/src/stats.rs`, `python/compute_stats.py` |

[RFC 0004]: ../reference/rfcs.md
