---
rfc: "0004"
title: "lewm-data — datasets, transforms, batching"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§4.2", "§5.1", "§5.4", "§6.4"]
depends_on: ["0001", "0002"]
related: ["0005", "0012", "0013"]
---

# RFC 0004 — `lewm-data`: datasets, transforms, batching

> **Status:** Accepted · **Version:** 1.0.0
>
> The data pipeline is the second most common source of silent ML bugs after loss math. This RFC pins the exact contract for sampling, decoding, transforming, and batching the two datasets (PushT and SO-100). It also defines the streaming I/O architecture that keeps the GPU fed at the throughput targets in NFR-010/011.

---

## 1. Introduction

### 1.1 Motivation

The model has been pinned by RFC 0002 and the loss math by RFC 0003. To run training, we need a data plane that:

1. Streams PushT HDF5 (~30 GB unpacked) without loading it into RAM.
2. Loads SO-100 (Parquet for actions + MP4 per-episode video) either via Python pre-decode or via Rust ffmpeg bindings.
3. Resizes images to 224×224, normalizes channels.
4. Normalizes action vectors per-dim using training-split statistics.
5. Samples non-boundary-crossing temporal windows of length `T = horizon` with `history_size` warm-up frames.
6. Collates batches such that the resulting tensors match the shape contract in §6.1 of the master spec.

All of this must sustain ≥ 45 PushT samples/sec on A10G-large (NFR-010) without GPU idle gaps.

### 1.2 Goals

1. Define the public API of `lewm-data` precisely.
2. Pin the on-disk schemas the loaders expect.
3. Specify each transform with its exact parameters.
4. Specify the window-sampling distribution and seeding contract.
5. Specify the I/O architecture (worker threads, prefetch depth, channel sizes).
6. Specify the error taxonomy for data-time failures.

### 1.3 Non-goals

- Model code (RFC 0002).
- The training-loop integration (RFC 0005).
- The eval-time data wrangling (RFC 0006).
- Dataset *creation* — the SO-100 preprocessing pipeline lives in [RFC 0012](0012-so100-real-robot-extension.md). This RFC defines only the loader for the resulting file.

---

## 2. Conventions

Per master spec and glossary. Specific to this RFC:

- **Sample** = the unit returned by the dataset's `get(idx)` method: one window of `T` frames plus its action tensor and metadata.
- **Frame** = one image timestep, shape `(C, H, W)` after preprocessing.
- **Window** = `T` consecutive frames within a single episode, sampled per FR-034.

---

## 3. Background

The two datasets are heterogeneous:

| Dataset | Container | Per-frame access | Random-access cost |
|---------|-----------|------------------|--------------------|
| PushT | `pusht_expert_train.h5.zst` decompressed to HDF5 | constant (HDF5 chunk read) | low (~µs) |
| SO-100 (raw) | per-episode Parquet + MP4 | linear (decode from MP4) | high (~ms per frame) |
| SO-100 (preprocessed) | one HDF5 per dataset | constant | low |

Streaming throughput differs by 2–3 orders of magnitude. The pipeline therefore standardizes on HDF5 internally: PushT is HDF5 natively; SO-100 is converted to HDF5 in [RFC 0012](0012-so100-real-robot-extension.md). The Rust loader has **one** schema (HDF5) and **two** dataset façades that map onto it.

---

## 4. Detailed design — module layout

```
lewm-data/
└── src/
    ├── lib.rs                  # re-exports
    ├── errors.rs               # DataError, FrameError
    ├── schema.rs               # HDF5 schema constants & validators
    ├── pusht.rs                # PushtDataset
    ├── so100.rs                # So100Dataset
    ├── transform/
    │   ├── mod.rs
    │   ├── image.rs            # resize, normalize, channel ordering
    │   ├── action.rs           # action normalization & stats
    │   └── window.rs           # window sampling
    ├── batch.rs                # collate, prefetch channel, batch struct
    ├── prefetch.rs             # worker pool, channel, lifecycle
    ├── stats.rs                # compute training-split stats (run-once tool)
    └── io/
        ├── mod.rs
        ├── hdf5.rs             # streaming HDF5 reader
        └── safetensors_stats.rs  # action-stat persistence
```

**Public re-exports in `lib.rs`:**

```rust
pub use crate::batch::{Batch, BatchBuilder, collate};
pub use crate::errors::DataError;
pub use crate::pusht::PushtDataset;
pub use crate::so100::So100Dataset;
pub use crate::transform::{ActionNormalizer, ImagePreprocessor, WindowSampler};
pub use crate::prefetch::{Prefetcher, PrefetcherConfig};
pub use crate::stats::{DatasetStats, compute_stats};
```

---

## 5. PushT loader

### 5.1 On-disk schema (HDF5)

The upstream PushT archive (`quentinll/lewm-pusht`) decompresses to a directory of HDF5 shards:

```
data/
  pusht_000.h5
  pusht_001.h5
  …
  pusht_NNN.h5
```

Each shard has the following groups and datasets (verified against `lucas-maes/le-wm/data_loading/pusht_dataset.py`):

```
/episode_index            : (E,)             int32   — episode this row belongs to
/timestep                 : (N,)             int32   — within-episode time index
/observation              :
    /pixels               : (N, 224, 224, 3) uint8   — RGB, native size
    /state                : (N, 5)           float32 — environment state (unused by LeWM)
/action                   : (N, 2)           float32 — 2-D PushT action
/next                     :
    /observation/pixels   : (N, 224, 224, 3) uint8   — frame at t+1
    /reward               : (N,)             float32
    /done                 : (N,)             bool
```

`N` is the number of transitions in the shard. Episode boundaries are detected by `timestep == 0` (the start of an episode).

**RFC0004-001 [MUST]** — The loader **MUST** validate the schema at open time and emit `DataError::SchemaMismatch { expected, found }` on any discrepancy.

**RFC0004-002 [MUST]** — Frames are stored in `uint8` HWC order. Conversion to `float32` CHW is the loader's responsibility (§7).

### 5.2 Public API

```rust
pub struct PushtDataset {
    shards: Vec<HdfShard>,        // sorted by filename
    episodes: Vec<EpisodeIndex>,   // global episode list
    config: PushtConfig,
    stats: DatasetStats,
}

#[derive(burn::config::Config, Debug, Clone)]
pub struct PushtConfig {
    pub root_path: PathBuf,
    pub split: Split,                            // Train | Eval (held-out 5% by episode)
    pub horizon: usize,                          // T
    pub history_size: usize,                     // h
    #[config(default = "Some(0)")]
    pub seed: Option<u64>,
    #[config(default = "true")]
    pub validate_schema: bool,
    /// Path to action-stat safetensors; if absent, computed at first epoch.
    pub stats_path: Option<PathBuf>,
}

#[derive(burn::config::Config, Debug, Clone, Copy, Eq, PartialEq)]
pub enum Split { Train, Eval }

impl PushtDataset {
    pub fn new(config: PushtConfig) -> Result<Self, DataError> { /* … */ }

    /// Total number of *windows* — episodes × (frames_per_episode - T + 1) summed.
    pub fn len(&self) -> usize { /* … */ }

    /// Fetches the `idx`-th window. `idx` is interpreted modulo `len()` to allow
    /// virtual epochs (training shuffles via the data-shuffle RNG sub-stream).
    pub fn get(&self, idx: usize) -> Result<Sample, DataError> { /* … */ }

    /// Yields owned `Sample`s in a deterministic order given the dataset seed,
    /// with prefetch baked in. See §9.
    pub fn iter(&self) -> SampleIter<'_> { /* … */ }
}

#[derive(Debug, Clone)]
pub struct Sample {
    /// (T, 3, 224, 224)  F32  in [0, 1] before image preprocessor normalization
    pub frames_t: Vec<u8>,      // flat HWC buffer; normalized lazily in collate
    pub frame_shape: (usize, usize, usize, usize),    // (T, H, W, C) sanity
    /// (T, 2)  F32  raw action (un-normalized)
    pub actions: Vec<f32>,
    pub action_shape: (usize, usize),
    /// Episode and start frame index — for trace logs and debugging.
    pub meta: SampleMeta,
}

#[derive(Debug, Clone, Copy)]
pub struct SampleMeta { pub episode_id: u32, pub start_frame: u32, pub shard: u16 }
```

**RFC0004-003 [MUST]** — `Sample` stores raw `u8` pixel buffers and raw `f32` action buffers. Final tensor construction (resize, normalize, action-normalize, cast to BF16/F32) happens in `collate`. Reason: this keeps `Sample` cheap to clone across threads (no large tensor allocations) and centralizes the device/dtype concern.

**RFC0004-004 [MUST]** — `len()` returns the number of valid windows in the split. For the train split this is roughly 920k − 50 × 8 ≈ 920k (each episode loses `T − 1` windows due to boundary).

### 5.3 Train/eval split

**RFC0004-005 [MUST]** — Split is **by episode**, not by frame. We hash `episode_id` with a fixed key and assign to eval if `hash % 20 == 0` (5 %), else train. This deterministic mapping is recomputable and immune to shard reordering.

### 5.4 Window sampling

**RFC0004-006 [MUST]** — Window index `idx ∈ [0, len)` maps to `(episode, start_frame)` via a precomputed table built at open time:

```
for episode in episodes_in_split:
    L = episode.length
    if L < T:                 # too short for one window
        continue
    for k in 0 .. (L - T + 1):
        table.push((episode.id, k))
```

`get(idx)` then reads frames `[start_frame .. start_frame + T)` from the appropriate shard.

**RFC0004-007 [MUST]** — When `T > episode.length`, the episode is **skipped entirely**, not padded. Padding would bias the latent distribution and break SIGReg's assumption of i.i.d. samples.

### 5.5 Shard caching

**RFC0004-008 [MUST]** — Open HDF5 file handles are pooled across the worker thread set, one handle per shard. `parking_lot::Mutex<HdfShard>` guards each handle; HDF5's thread-safety guarantees are conservative and we serialize per-shard reads. Sharded parallelism across shards is unaffected.

---

## 6. SO-100 loader

### 6.1 Two intake formats

Two loader implementations live behind a common `So100Dataset` façade:

1. **`So100Dataset::from_hdf5(path)`** — reads the pre-decoded HDF5 produced by `python/decode_so100_to_h5.py`. **Primary path** for v1.
2. **`So100Dataset::from_raw(parquet_dir, mp4_dir)`** — reads Parquet + MP4 directly via `parquet` and `ffmpeg-next` Rust crates. **Optional fallback** for v2 work; not required for v1 acceptance.

The `from_hdf5` path **MUST** be used for all training runs that produce v1 acceptance artifacts.

### 6.2 HDF5 schema for SO-100

After `decode_so100_to_h5.py` (RFC 0012 §4):

```
/episode_index            : (E,)             int32
/timestep                 : (N,)             int32
/observation/
  pixels_top              : (N, 224, 224, 3) uint8   — top camera, resampled to 10 Hz, 224x224
  pixels_wrist            : (N, 224, 224, 3) uint8   — wrist camera
/action                   : (N, 6)           float32 — 6-D action
/joint_pos                : (N, 6)           float32 — joint state (unused for LeWM training, available for analysis)
```

**RFC0004-009 [MUST]** — Only one of `pixels_top` or `pixels_wrist` is consumed by training. The default is `pixels_top`. The choice is a config flag (`So100Config::camera_view`).

**RFC0004-010 [SHOULD]** — A future v2 may concatenate both views into a 6-channel input. v1 keeps 3-channel for parity with PushT.

### 6.3 Public API

```rust
pub struct So100Dataset { /* analogous to PushtDataset */ }

#[derive(burn::config::Config, Debug, Clone)]
pub struct So100Config {
    pub hdf5_path: PathBuf,
    pub split: Split,
    pub horizon: usize,
    pub history_size: usize,
    pub seed: Option<u64>,
    #[config(default = "CameraView::Top")]
    pub camera_view: CameraView,
    pub stats_path: Option<PathBuf>,
}

#[derive(burn::config::Config, Debug, Clone, Copy, Eq, PartialEq)]
pub enum CameraView { Top, Wrist }

impl So100Dataset {
    pub fn from_hdf5(config: So100Config) -> Result<Self, DataError> { /* … */ }
    pub fn len(&self) -> usize;
    pub fn get(&self, idx: usize) -> Result<Sample, DataError>;
    pub fn iter(&self) -> SampleIter<'_>;
}
```

### 6.4 Eval split

**RFC0004-011 [MUST]** — Five episodes are held out for eval (10 %). The specific episode IDs are pinned in [RFC 0012 §6](0012-so100-real-robot-extension.md) and stored in the dataset stats file so the split is recomputable.

---

## 7. Transforms

### 7.1 Image preprocessor

```rust
pub struct ImagePreprocessor {
    /// Target image size (square). LeWM default: 224.
    pub target_size: u32,
    /// Per-channel mean (HF ViT default).
    pub mean: [f32; 3],
    /// Per-channel std (HF ViT default).
    pub std:  [f32; 3],
    /// Bilinear (default) or bicubic.
    pub interp: InterpKind,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum InterpKind { Bilinear, Bicubic }

impl Default for ImagePreprocessor {
    fn default() -> Self {
        Self {
            target_size: 224,
            mean: [0.5, 0.5, 0.5],
            std:  [0.5, 0.5, 0.5],
            interp: InterpKind::Bilinear,
        }
    }
}

impl ImagePreprocessor {
    /// Convert a flat HWC `u8` buffer to a CHW `f32` Vec, resized to `target_size`,
    /// and normalized by `(x - mean) / std`.
    ///
    /// Input  `src`: `(H * W * 3,)` `u8`.
    /// Output: `(3 * target_size * target_size,)` `f32`.
    pub fn apply(&self, src: &[u8], src_h: u32, src_w: u32) -> Vec<f32> { /* … */ }
}
```

**RFC0004-012 [MUST]** — Default `mean=std=[0.5,0.5,0.5]` matches HF ViT defaults at the pinned `transformers` version. Verified by inspecting `preprocessor_config.json` of `quentinll/lewm-pusht` (Phase 0).

**RFC0004-013 [MUST]** — Resize: if `src_h == src_w == target_size`, the resize is a strict identity (just `u8 → f32 / 255`). PushT frames are already 224×224; SO-100 frames are 224×224 after the Python preprocessing in RFC 0012. So at training time, the resize is a no-op. We retain the code path so `lewm-infer` can accept arbitrary-size inputs.

**RFC0004-014 [MUST]** — Bilinear resize uses the `image` crate's `imageops::resize` with `FilterType::Triangle`. Bicubic uses `FilterType::CatmullRom`. Both are F32-exact across runs of the same `image` crate version (pinned in `Cargo.toml`).

**RFC0004-015 [MUST]** — Channel order: input HWC → output CHW. Permutation in Rust is a stride re-pack; no transpose op needed in the tensor sense.

### 7.2 Action normalizer

```rust
pub struct ActionNormalizer {
    pub mean: Vec<f32>,        // (A,)
    pub std:  Vec<f32>,        // (A,)
}

impl ActionNormalizer {
    /// Map raw action to normalized action: (a - mean) / std.
    pub fn apply(&self, src: &[f32]) -> Vec<f32> { /* … */ }

    /// Inverse: normalized → raw. Used at planning time.
    pub fn inverse(&self, src: &[f32]) -> Vec<f32> { /* … */ }
}
```

**RFC0004-016 [MUST]** — `mean` and `std` are **per dimension** and computed from the **training split only** (no leakage from eval).

**RFC0004-017 [MUST]** — `std[d]` is replaced by `1.0` whenever the empirical std is below `1e-6`. This avoids NaNs on degenerate dimensions (e.g., a constant action axis in a debug dataset).

**RFC0004-018 [MUST]** — Stats are persisted to `safetensors` at `stats_path` with the layout:

```
key                shape   dtype
action_mean        (A,)    f32
action_std         (A,)    f32
pixel_mean         (3,)    f32   (informational; overridden by ImagePreprocessor)
pixel_std          (3,)    f32
n_train_samples    ()      i64
content_hash       (32,)   u8    BLAKE3 of the underlying dataset bytes
```

The `content_hash` ensures stats are tied to a specific dataset version.

### 7.3 Window sampler

The window sampler is implicit in `PushtDataset::get` / `So100Dataset::get`: the index → window map is precomputed at dataset open (per §5.4). No additional sampler module is needed.

**RFC0004-019 [MUST]** — Iteration order is determined by an explicit shuffle of the window indices using the `rng:data_shuffle` sub-stream (RFC 0013 §4). The shuffle is **per epoch** (Fisher–Yates with the sub-stream).

---

## 8. Batching

### 8.1 `Batch` struct

```rust
#[derive(Debug)]
pub struct Batch<B: Backend> {
    /// (B, T, 3, 224, 224) F32 or BF16 depending on backend
    pub pixels: Tensor<B, 5>,
    /// (B, T, A) F32 (always — actions are small and not worth lowering)
    pub actions: Tensor<B, 3>,
    /// Per-sample metadata for traceability.
    pub meta: Vec<SampleMeta>,
}
```

**RFC0004-020 [MUST]** — `pixels` dtype matches the model dtype (BF16 for mixed runs, F32 otherwise). `actions` stays F32 because the action stream is small and the dtype cast introduces no perf benefit but does affect downstream `Embedder` precision.

### 8.2 `collate`

```rust
pub fn collate<B: Backend>(
    samples: &[Sample],
    image_preproc: &ImagePreprocessor,
    action_norm: &ActionNormalizer,
    device: &B::Device,
    dtype: BatchDtype,
) -> Result<Batch<B>, DataError> { /* … */ }
```

**Algorithm:**

```
for each sample:
    pixels_f32 = image_preproc.apply(sample.frames, h, w)            # (T*3*H*W,)
    actions_norm = action_norm.apply(sample.actions)                  # (T*A,)
    push to bulk buffers

pixels_tensor = Tensor::<B::F32>::from_data(bulk_pixels)
                  .reshape([B, T, 3, H, W])
if dtype == BF16: pixels_tensor = pixels_tensor.cast::<BF16>()

actions_tensor = Tensor::<B::F32>::from_data(bulk_actions)
                  .reshape([B, T, A])

return Batch { pixels: pixels_tensor, actions: actions_tensor, meta }
```

**RFC0004-021 [MUST]** — `collate` **MUST NOT** allocate intermediate per-sample tensors. The bulk buffer pattern keeps memory traffic linear in batch size.

**RFC0004-022 [MUST]** — `collate` **MUST** validate that all samples have identical `(T, H, W, C)` and identical `(T, A)`. Mismatch returns `DataError::InconsistentShapes`.

### 8.3 Batch dtype enum

```rust
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BatchDtype { F32, Bf16 }
```

Routed from training config; never inferred from the backend type at runtime.

---

## 9. I/O architecture and prefetch

### 9.1 Topology

```
disk (HDF5) ──┐
              ├─ blocking-pool worker (× N) ──► bounded channel ──► main thread ──► collate ──► device
disk (HDF5) ──┘
```

- `N = 4` workers by default (PushT-optimized; SO-100 with HDF5 path has identical workload).
- Channel depth `K = 4`: i.e., up to 4 ready samples buffered. Beyond this, workers block.
- Workers run on `tokio::task::spawn_blocking`-managed threads so I/O does not starve the runtime.

### 9.2 `Prefetcher`

```rust
pub struct PrefetcherConfig {
    pub num_workers: usize,        // default 4
    pub channel_capacity: usize,   // default 4
    pub batch_size: usize,
    pub epoch_seed: u64,
}

pub struct Prefetcher<B: Backend> {
    rx: crossbeam::channel::Receiver<Batch<B>>,
    /// Shutdown handle. Drop guarantees workers exit.
    _handle: PrefetchHandle,
}

impl<B: Backend> Prefetcher<B> {
    pub fn new<D: Dataset + Send + Sync + 'static>(
        dataset: Arc<D>,
        config: PrefetcherConfig,
        device: B::Device,
        image_preproc: ImagePreprocessor,
        action_norm: ActionNormalizer,
        dtype: BatchDtype,
    ) -> Result<Self, DataError> { /* … */ }

    pub fn next(&mut self) -> Option<Batch<B>> { self.rx.recv().ok() }
}
```

Where `Dataset` is the common trait:

```rust
pub trait Dataset {
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Result<Sample, DataError>;
}
```

**RFC0004-023 [MUST]** — `Prefetcher::next` returns `None` when the epoch is exhausted; the main loop then advances `epoch_seed` and calls `Prefetcher::new` again. (Alternative API: `start_new_epoch` to recycle workers; we keep the simpler restart pattern in v1.)

**RFC0004-024 [MUST]** — Worker shutdown is **graceful**: dropping `Prefetcher` closes the channel; workers detect EOF on the next sample emit and exit. Tested by `TST-0004-PREFETCH-001`.

**RFC0004-025 [SHOULD]** — A backpressure metric `data/queue_depth` is emitted every step. Persistent `queue_depth == 0` indicates the data plane is the bottleneck; persistent `queue_depth == K` indicates the GPU is.

### 9.3 Throughput contract

**RFC0004-026 [MUST]** — On the A10G-large with `B=64, T=8, num_workers=4, channel=4`, the data plane sustains **≥ 60 batches/sec** when feeding to a no-op consumer (i.e., the loader is not the bottleneck). Below this is a perf regression; see [RFC 0014 §5](0014-performance-engineering.md).

---

## 10. Error taxonomy

```rust
#[derive(thiserror::Error, Debug)]
pub enum DataError {
    #[error("HDF5 schema mismatch: expected {expected:?}, found {found:?}")]
    SchemaMismatch { expected: String, found: String },

    #[error("HDF5 read error at shard={shard}, dataset={dataset}: {source}")]
    HdfRead { shard: PathBuf, dataset: &'static str, #[source] source: hdf5_metno::Error },

    #[error("Window index out of range: idx={idx}, len={len}")]
    IndexOutOfRange { idx: usize, len: usize },

    #[error("Inconsistent shapes in collate: {detail}")]
    InconsistentShapes { detail: String },

    #[error("Action stats file missing or malformed: {path}")]
    StatsMissing { path: PathBuf },

    #[error("Image decode failed for sample (shard={shard}, frame={frame}): {source}")]
    ImageDecode { shard: u16, frame: u32, #[source] source: image::ImageError },

    #[error("Prefetch worker channel closed unexpectedly")]
    ChannelClosed,

    #[error("Validation failed: {0}")]
    Validation(String),
}
```

**RFC0004-027 [MUST]** — `DataError` is `Send + Sync + 'static` and integrates with `anyhow::Error` at the binary boundary.

**RFC0004-028 [MUST]** — On a recoverable error during prefetch (e.g., a single shard checksum mismatch), the worker logs a `WARN` and tries the next index; on **two consecutive** failures from the same shard, the worker emits a fatal `ChannelClosed` to the channel which propagates as a training-level fatal.

---

## 11. Stats computation tool

`lewm-data` exposes a one-shot binary `compute_stats` under `src/bin/compute_stats.rs`:

```
$ compute_stats --dataset pusht --root /data/pusht --out /data/pusht/stats.safetensors
```

It iterates the training split once, accumulates per-dim sums and sums-of-squares for `action`, computes mean & std, hashes the input bytes, and writes the safetensors file.

**RFC0004-029 [MUST]** — The tool **MUST** be deterministic: the same input dataset and `--seed` produce bit-identical stats files. (Used in CI.)

---

## 12. Testing strategy

### 12.1 Test inventory

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0004-PUSHT-001 | `pusht_open_and_len` | integration | RFC0004-001, RFC0004-005 |
| TST-0004-PUSHT-002 | `pusht_get_window_shapes` | integration | RFC0004-004 |
| TST-0004-PUSHT-003 | `pusht_no_episode_crossing` | integration | RFC0004-006 |
| TST-0004-PUSHT-004 | `pusht_short_episode_skipped` | integration | RFC0004-007 |
| TST-0004-SO100-001 | `so100_hdf5_open_and_len` | integration | analogous |
| TST-0004-SO100-002 | `so100_camera_view_select` | integration | RFC0004-009 |
| TST-0004-SO100-003 | `so100_get_window_shapes` | integration | shape contract |
| TST-0004-SO100-004 | `so100_holdout_episodes_disjoint` | integration | RFC0004-011 |
| TST-0004-XFORM-001 | `image_preprocess_identity_at_224` | unit | RFC0004-012/013 |
| TST-0004-XFORM-002 | `image_preprocess_normalize` | unit | RFC0004-012 |
| TST-0004-XFORM-003 | `image_preprocess_resize_192_to_224` | unit | RFC0004-014 |
| TST-0004-XFORM-004 | `action_normalize_roundtrip` | unit | RFC0004-016 |
| TST-0004-XFORM-005 | `action_normalize_zero_std_replace` | unit | RFC0004-017 |
| TST-0004-WIN-001 | `window_index_table_correct` | unit | RFC0004-006 |
| TST-0004-WIN-002 | `window_shuffle_deterministic_per_seed` | unit | RFC0004-019 |
| TST-0004-WIN-003 | `window_shuffle_differs_across_seeds` | unit | RFC0004-019 |
| TST-0004-PREFETCH-001 | `prefetcher_clean_shutdown` | integration | RFC0004-024 |
| TST-0004-PREFETCH-002 | `prefetcher_throughput_at_least_60_bps` | bench | RFC0004-026 |
| TST-0004-STATS-001 | `compute_stats_deterministic` | unit | RFC0004-029 |
| TST-0004-ERR-001 | `corrupted_shard_recoverable_then_fatal` | integration | RFC0004-028 |

### 12.2 Test fixtures

- A synthetic PushT-format HDF5 (`tests/fixtures/pusht_synth.h5`, ~ 1 MB, 2 episodes × 16 frames each). Committed via Git LFS.
- A synthetic SO-100-format HDF5 (`tests/fixtures/so100_synth.h5`). Same.
- A "corrupted" shard with a deliberately bad checksum for `TST-0004-ERR-001`.

### 12.3 Property tests

- *P-1: action normalizer round-trip.* `inverse(apply(a)) == a` to within `1e-6` for any `a`.
- *P-2: image preprocess deterministic.* Same input → same output bytes across runs (modulo the `image` crate version).
- *P-3: window index never crosses an episode boundary.* QuickCheck-style over generated episode lengths.

---

## 13. Operational considerations

### 13.1 Observability

Metrics emitted via `lewm-telemetry`:

- `data/throughput_samples_per_sec`
- `data/throughput_bytes_per_sec`
- `data/queue_depth`
- `data/io_wait_ms_p50`, `data/io_wait_ms_p99`
- `data/error_count` (per `DataError` variant, tagged)

Spans:

- `data.dataset_open` (per dataset)
- `data.get_window` (per sample; sampled at 1 in 1000)
- `data.collate` (per batch)
- `data.prefetch_worker.lifetime` (per worker)

### 13.2 Runbook

- **"`queue_depth == 0` persistently."** — increase `num_workers` (4 → 6/8); check disk IOPS via `iostat`; verify HDF5 shards are on NVMe not network FS.
- **"`SchemaMismatch` on open."** — the dataset version on disk does not match the loader's expected schema. Re-download with `hf download --revision <pinned-rev>`.
- **"`StatsMissing` on resume."** — pass `--stats-path` explicitly or run `compute_stats` before resume.

### 13.3 Capacity

- Memory budget: HDF5 file handles ~ 50 MB; bulk buffer in collate ~ `B·T·H·W·C·4 = 64·8·224·224·3·4 ≈ 308 MB` peak before device upload. Free after upload.
- File handles: one per shard × workers; bounded by `ulimit -n`. Default `1024` is sufficient.

---

## 14. Performance considerations

Hot loops in `image::imageops::resize` and HDF5 chunked reads. Both are SIMD-accelerated (image via `wide`, HDF5 via libhdf5 native). No additional SIMD work planned for v1; revisit only if benches regress.

For SO-100 raw-MP4 path (out of v1 scope), `ffmpeg-next` is the natural choice but adds a large native dep. v1 uses Python pre-decode (RFC 0012).

---

## 15. Security considerations

- HDF5 files are untrusted input — sanity-check shape and dtype at every open.
- No code injection surface: we never `eval` HDF5 string attrs; only known keys are read.
- Resource exhaustion: `len()` is bounded by file size; we enforce a configurable upper cap of `10^9` to detect malformed shards.

---

## 16. Alternatives considered

- **A1 — Use `tch-rs` (LibTorch) data loaders.** Rejected: pulls in a heavy C++ dep and defeats the "pure Rust" deliverable.
- **A2 — Lazy decode of MP4 in-loop.** Rejected for v1 throughput; revisit in v2.
- **A3 — Memory-mapped HDF5 via `hdf5-metno`.** Used implicitly. The crate uses libhdf5's mmap path when available.
- **A4 — Async I/O for HDF5.** Rejected: HDF5's threading model is awkward; `spawn_blocking` is the idiomatic Rust path.

---

## 17. Acceptance criteria

- [ ] All TST-0004-* pass on `linux-x86_64` and `aarch64-darwin`.
- [ ] `compute_stats` produces a bit-identical safetensors file on two independent runs with the same input.
- [ ] `PushtDataset::iter` yields windows in the deterministic shuffled order specified by RFC0004-019.
- [ ] Prefetcher hits the throughput target NFR-026 in the bench.
- [ ] Synthetic fixtures committed under `tests/fixtures/`.

---

## 18. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | HDF5 schema drift in upstream PushT dump | L | M | Validate at open; pin dataset revision in `configs/pusht.toml` |
| R-2 | SO-100 Python pre-decode binary drift | M | M | Hash input MP4 set; re-run if hash mismatches |
| R-3 | `image` crate version bump changes resize output | L | L | Pin `image = "0.25"` in workspace deps |
| R-4 | Prefetcher deadlock on shutdown | L | M | Channel close idiom + explicit `Drop` test |
| R-5 | Per-channel mean/std mismatch with HF preprocessor | M | H | Phase 0 verification of `preprocessor_config.json` from `quentinll/lewm-pusht` |

---

## 19. Open questions

OQ-2004-1 — Should SO-100 stats include `joint_pos` for future analysis even though we do not train on it? Likely yes for parity with `lerobot` conventions; decide in Phase 4.

---

## 20. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0004.*
