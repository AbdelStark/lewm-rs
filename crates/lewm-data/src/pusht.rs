//! `PushT` HDF5 streaming dataset loader.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use hdf5_metno as hdf5;
use ndarray::{Ix2, Ix4, s};
use parking_lot::Mutex;

use crate::DataError;

const RGB_CHANNELS: usize = 3;
const PUSHT_ACTION_DIM: usize = 2;
const EPISODE_DATASET_CANDIDATES: &[&str] = &["episode_index", "episode_idx"];
const TIMESTEP_DATASET_CANDIDATES: &[&str] = &["timestep", "step_idx"];
const PIXEL_DATASET_CANDIDATES: &[&str] = &["observation/pixels", "pixels"];
const ACTION_DATASET_CANDIDATES: &[&str] = &["action"];
const EVAL_SPLIT_BUCKETS: u64 = 20;
const EVAL_BUCKET: u64 = 0;
const SPLIT_KEY: [u8; 32] = [
    0x6c, 0x65, 0x77, 0x6d, 0x2d, 0x72, 0x73, 0x2d, 0x70, 0x75, 0x73, 0x68, 0x74, 0x2d, 0x73, 0x70,
    0x6c, 0x69, 0x74, 0x2d, 0x6b, 0x65, 0x79, 0x2d, 0x76, 0x31, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Dataset split selected at open time.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Split {
    /// Training split: all episodes not assigned to the deterministic eval bucket.
    Train,
    /// Evaluation split: deterministic 5 percent episode holdout.
    Eval,
}

/// `PushT` dataset loader configuration.
#[derive(Debug, Clone)]
pub struct PushtConfig {
    /// Path to either a single HDF5 shard or a directory containing `.h5`/`.hdf5` shards.
    pub root_path: PathBuf,
    /// Episode split to expose.
    pub split: Split,
    /// Number of frames/actions in each sampled window.
    pub horizon: usize,
    /// Number of historical warm-up frames expected by downstream training.
    pub history_size: usize,
    /// Optional deterministic seed reserved for iterator shuffling.
    pub seed: Option<u64>,
    /// Validate HDF5 schema at open time.
    pub validate_schema: bool,
    /// Optional path to persisted action statistics.
    pub stats_path: Option<PathBuf>,
}

impl PushtConfig {
    /// Create a `PushT` config with training split and schema validation enabled.
    #[must_use]
    pub fn new(root_path: impl Into<PathBuf>, horizon: usize) -> Self {
        Self {
            root_path: root_path.into(),
            split: Split::Train,
            horizon,
            history_size: 0,
            seed: Some(0),
            validate_schema: true,
            stats_path: None,
        }
    }
}

/// A raw `PushT` temporal window.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    /// Flat HWC pixel buffer for a `(T, H, W, C)` window.
    pub frames_t: Vec<u8>,
    /// Pixel shape `(T, H, W, C)`.
    pub frame_shape: (usize, usize, usize, usize),
    /// Flat action buffer for a `(T, 2)` window.
    pub actions: Vec<f32>,
    /// Action shape `(T, action_dim)`.
    pub action_shape: (usize, usize),
    /// Episode and shard metadata for traceability.
    pub meta: SampleMeta,
}

/// Metadata identifying the source of a sampled `PushT` window.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SampleMeta {
    /// Deterministic episode id.
    pub episode_id: u32,
    /// Timestep of the first frame in the window.
    pub start_frame: u32,
    /// Source shard ordinal in sorted filename order.
    pub shard: u16,
}

/// Streaming `PushT` dataset backed by HDF5 shards.
pub struct PushtDataset {
    shards: Vec<Mutex<HdfShard>>,
    windows: Vec<WindowIndex>,
    config: PushtConfig,
}

impl fmt::Debug for PushtDataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PushtDataset")
            .field("shards", &self.shards.len())
            .field("windows", &self.windows.len())
            .field("config", &self.config)
            .finish()
    }
}

impl PushtDataset {
    /// Open `PushT` shards, validate their schema, and precompute the window index table.
    ///
    /// The dataset stores only shard handles and row indexes. Pixel/action data are
    /// read from HDF5 on demand in [`Self::get`].
    ///
    /// # Errors
    ///
    /// Returns an error when configuration is invalid, shards cannot be discovered
    /// or opened, a shard schema is invalid, or the selected split has no valid
    /// windows for the configured horizon.
    pub fn new(config: PushtConfig) -> Result<Self, DataError> {
        validate_config(&config)?;

        let shard_paths = discover_shards(&config.root_path)?;
        let mut shards = Vec::with_capacity(shard_paths.len());
        let mut windows = Vec::new();

        for (shard_index, path) in shard_paths.iter().enumerate() {
            let shard_id = u16::try_from(shard_index).map_err(|_| {
                DataError::InvalidConfig("PushT supports at most 65535 shards".to_string())
            })?;
            let (shard, mut shard_windows) = HdfShard::open(path, shard_id, &config)?;
            windows.append(&mut shard_windows);
            shards.push(Mutex::new(shard));
        }

        if windows.is_empty() {
            return Err(DataError::EmptyDataset(format!(
                "no valid {:?} windows with horizon {} under {}",
                config.split,
                config.horizon,
                config.root_path.display()
            )));
        }

        Ok(Self {
            shards,
            windows,
            config,
        })
    }

    /// Number of valid windows in the configured split.
    #[must_use]
    pub fn len(&self) -> usize {
        self.windows.len()
    }

    /// Return whether the dataset has no valid windows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Fetch a raw `PushT` sample window.
    ///
    /// `idx` is interpreted modulo [`Self::len`] so virtual epochs can be longer
    /// than the physical window table.
    ///
    /// # Errors
    ///
    /// Returns an error if the dataset is empty, a window references a missing
    /// shard, or the underlying HDF5 slice read fails.
    pub fn get(&self, idx: usize) -> Result<Sample, DataError> {
        if self.windows.is_empty() {
            return Err(DataError::EmptyDataset(
                "cannot sample from an empty PushT dataset".to_string(),
            ));
        }

        let window = self.windows[idx % self.windows.len()];
        let shard = self
            .shards
            .get(usize::from(window.shard))
            .ok_or_else(|| DataError::EmptyDataset("window references missing shard".to_string()))?
            .lock();

        shard.read_window(window, self.config.horizon)
    }
}

#[derive(Debug)]
struct HdfShard {
    file: hdf5::File,
    shard_id: u16,
    pixel_path: String,
    action_path: String,
    pixel_shape: [usize; 4],
    action_shape: [usize; 2],
}

impl HdfShard {
    fn open(
        path: &Path,
        shard_id: u16,
        config: &PushtConfig,
    ) -> Result<(Self, Vec<WindowIndex>), DataError> {
        let file = hdf5::File::open(path)
            .map_err(|source| DataError::hdf5(format!("open {}", path.display()), source))?;

        let schema = ShardSchema::read(path, &file, config.validate_schema)?;
        let episodes = build_episodes(&schema)?;
        let windows = build_windows(&episodes, shard_id, config);
        let shard = Self {
            file,
            shard_id,
            pixel_path: schema.pixel_path,
            action_path: schema.action_path,
            pixel_shape: schema.pixel_shape,
            action_shape: schema.action_shape,
        };

        Ok((shard, windows))
    }

    fn read_window(&self, window: WindowIndex, horizon: usize) -> Result<Sample, DataError> {
        let start = window.row_start;
        let end = start + horizon;

        let pixels = self
            .file
            .dataset(&self.pixel_path)
            .map_err(|source| DataError::hdf5(format!("open {}", self.pixel_path), source))?
            .read_slice::<u8, _, Ix4>(s![start..end, .., .., ..])
            .map_err(|source| {
                DataError::hdf5(format!("read {} window", self.pixel_path), source)
            })?;
        let actions = self
            .file
            .dataset(&self.action_path)
            .map_err(|source| DataError::hdf5(format!("open {}", self.action_path), source))?
            .read_slice::<f32, _, Ix2>(s![start..end, ..])
            .map_err(|source| {
                DataError::hdf5(format!("read {} window", self.action_path), source)
            })?;

        Ok(Sample {
            frames_t: pixels.iter().copied().collect(),
            frame_shape: (
                horizon,
                self.pixel_shape[1],
                self.pixel_shape[2],
                self.pixel_shape[3],
            ),
            actions: actions.iter().copied().collect(),
            action_shape: (horizon, self.action_shape[1]),
            meta: SampleMeta {
                episode_id: window.episode_id,
                start_frame: window.start_frame,
                shard: self.shard_id,
            },
        })
    }
}

#[derive(Debug)]
struct ShardSchema {
    episode_index: Vec<i64>,
    timestep: Vec<i64>,
    pixel_path: String,
    action_path: String,
    pixel_shape: [usize; 4],
    action_shape: [usize; 2],
}

impl ShardSchema {
    fn read(path: &Path, file: &hdf5::File, validate_schema: bool) -> Result<Self, DataError> {
        let (episode_ds, episode_path) = dataset_any(file, EPISODE_DATASET_CANDIDATES)?;
        let (timestep_ds, timestep_path) = dataset_any(file, TIMESTEP_DATASET_CANDIDATES)?;
        let (pixels_ds, pixel_path) = dataset_any(file, PIXEL_DATASET_CANDIDATES)?;
        let (action_ds, action_path) = dataset_any(file, ACTION_DATASET_CANDIDATES)?;

        if validate_schema {
            require_index_dtype(&episode_ds, &episode_path)?;
            require_index_dtype(&timestep_ds, &timestep_path)?;
            require_dtype::<u8>(&pixels_ds, &pixel_path)?;
            require_dtype::<f32>(&action_ds, &action_path)?;
        }

        let timestep_shape = timestep_ds.shape();
        let episode_shape = episode_ds.shape();
        let pixel_shape = vec_to_array::<4>(pixels_ds.shape(), &pixel_path)?;
        let action_shape = vec_to_array::<2>(action_ds.shape(), &action_path)?;

        if timestep_shape.len() != 1 {
            return Err(DataError::schema(
                "timestep",
                "shape (N)",
                format!("shape {timestep_shape:?}"),
            ));
        }
        if episode_shape.len() != 1 {
            return Err(DataError::schema(
                "episode_index",
                "shape (N) or (E)",
                format!("shape {episode_shape:?}"),
            ));
        }
        if pixel_shape[3] != RGB_CHANNELS {
            return Err(DataError::schema(
                &pixel_path,
                "last dimension 3 RGB channels",
                format!("shape {pixel_shape:?}"),
            ));
        }
        if action_shape[1] != PUSHT_ACTION_DIM {
            return Err(DataError::schema(
                &action_path,
                "shape (N, 2)",
                format!("shape {action_shape:?}"),
            ));
        }
        if timestep_shape[0] != pixel_shape[0] || timestep_shape[0] != action_shape[0] {
            return Err(DataError::schema(
                path.display().to_string(),
                format!("matching N across {timestep_path}, {pixel_path}, and {action_path}"),
                format!(
                    "timestep={timestep_shape:?}, pixels={pixel_shape:?}, action={action_shape:?}"
                ),
            ));
        }

        Ok(Self {
            episode_index: read_index_dataset(&episode_ds, &episode_path)?,
            timestep: read_index_dataset(&timestep_ds, &timestep_path)?,
            pixel_path,
            action_path,
            pixel_shape,
            action_shape,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct Episode {
    id: u32,
    row_start: usize,
    len: usize,
    first_timestep: u32,
}

#[derive(Debug, Clone, Copy)]
struct WindowIndex {
    shard: u16,
    row_start: usize,
    episode_id: u32,
    start_frame: u32,
}

fn validate_config(config: &PushtConfig) -> Result<(), DataError> {
    if config.horizon == 0 {
        return Err(DataError::InvalidConfig(
            "horizon must be greater than zero".to_string(),
        ));
    }
    if config.history_size > config.horizon {
        return Err(DataError::InvalidConfig(format!(
            "history_size {} cannot exceed horizon {}",
            config.history_size, config.horizon
        )));
    }
    Ok(())
}

fn discover_shards(root_path: &Path) -> Result<Vec<PathBuf>, DataError> {
    let metadata = fs::metadata(root_path).map_err(|source| DataError::io(root_path, source))?;
    if metadata.is_file() {
        return Ok(vec![root_path.to_path_buf()]);
    }
    if !metadata.is_dir() {
        return Err(DataError::InvalidConfig(format!(
            "{} is neither a file nor directory",
            root_path.display()
        )));
    }

    let mut paths = Vec::new();
    for entry in fs::read_dir(root_path).map_err(|source| DataError::io(root_path, source))? {
        let entry = entry.map_err(|source| DataError::io(root_path, source))?;
        let path = entry.path();
        if path.is_file() && is_hdf5_path(&path) {
            paths.push(path);
        }
    }
    paths.sort();

    if paths.is_empty() {
        return Err(DataError::EmptyDataset(format!(
            "no .h5 or .hdf5 shards under {}",
            root_path.display()
        )));
    }

    Ok(paths)
}

fn is_hdf5_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| matches!(ext, "h5" | "hdf5"))
}

fn dataset_any(file: &hdf5::File, names: &[&str]) -> Result<(hdf5::Dataset, String), DataError> {
    for name in names {
        if let Ok(dataset) = file.dataset(name) {
            return Ok((dataset, (*name).to_owned()));
        }
    }

    Err(DataError::schema(
        names.join("|"),
        "one of the supported PushT dataset paths",
        "missing",
    ))
}

fn require_dtype<T: hdf5::H5Type>(dataset: &hdf5::Dataset, path: &str) -> Result<(), DataError> {
    let dtype = dataset
        .dtype()
        .map_err(|source| DataError::hdf5(format!("read dtype {path}"), source))?;
    if dtype.is::<T>() {
        Ok(())
    } else {
        Err(DataError::schema(
            path,
            std::any::type_name::<T>(),
            "different HDF5 dtype",
        ))
    }
}

fn require_index_dtype(dataset: &hdf5::Dataset, path: &str) -> Result<(), DataError> {
    let dtype = dataset
        .dtype()
        .map_err(|source| DataError::hdf5(format!("read dtype {path}"), source))?;
    if dtype.is::<i32>() || dtype.is::<i64>() {
        Ok(())
    } else {
        Err(DataError::schema(
            path,
            "int32 or int64",
            "different HDF5 dtype",
        ))
    }
}

fn read_index_dataset(dataset: &hdf5::Dataset, path: &str) -> Result<Vec<i64>, DataError> {
    let dtype = dataset
        .dtype()
        .map_err(|source| DataError::hdf5(format!("read dtype {path}"), source))?;
    if dtype.is::<i32>() {
        let values = dataset
            .read_raw::<i32>()
            .map_err(|source| DataError::hdf5(format!("read {path}"), source))?;
        return Ok(values.into_iter().map(i64::from).collect());
    }
    if dtype.is::<i64>() {
        return dataset
            .read_raw::<i64>()
            .map_err(|source| DataError::hdf5(format!("read {path}"), source));
    }

    Err(DataError::schema(
        path,
        "int32 or int64",
        "different HDF5 dtype",
    ))
}

fn vec_to_array<const N: usize>(shape: Vec<usize>, path: &str) -> Result<[usize; N], DataError> {
    shape.try_into().map_err(|shape: Vec<usize>| {
        DataError::schema(path, format!("{N}-D shape"), format!("shape {shape:?}"))
    })
}

fn build_episodes(schema: &ShardSchema) -> Result<Vec<Episode>, DataError> {
    if schema.timestep.is_empty() {
        return Err(DataError::EmptyDataset(
            "PushT shard has zero timesteps".to_string(),
        ));
    }
    if schema.episode_index.is_empty() {
        return Err(DataError::schema(
            "episode_index",
            "non-empty episode index",
            "empty dataset",
        ));
    }
    if schema.episode_index.len() != schema.timestep.len()
        && schema.episode_index.len() > schema.timestep.len()
    {
        return Err(DataError::schema(
            "episode_index",
            "shape (N) or shape (E <= N)",
            format!(
                "{} ids for {} rows",
                schema.episode_index.len(),
                schema.timestep.len()
            ),
        ));
    }

    let per_row_episode_ids = schema.episode_index.len() == schema.timestep.len();
    let mut episodes = Vec::new();
    let mut start = 0usize;
    let mut ordinal = 0usize;

    for row in 1..schema.timestep.len() {
        let episode_changed = per_row_episode_ids
            && schema.episode_index[row] != schema.episode_index[row.saturating_sub(1)];
        if schema.timestep[row] == 0 || episode_changed {
            push_episode(
                schema,
                per_row_episode_ids,
                start,
                row,
                ordinal,
                &mut episodes,
            )?;
            ordinal += 1;
            start = row;
        }
    }
    push_episode(
        schema,
        per_row_episode_ids,
        start,
        schema.timestep.len(),
        ordinal,
        &mut episodes,
    )?;

    Ok(episodes)
}

fn push_episode(
    schema: &ShardSchema,
    per_row_episode_ids: bool,
    start: usize,
    end: usize,
    ordinal: usize,
    episodes: &mut Vec<Episode>,
) -> Result<(), DataError> {
    let episode_id = if per_row_episode_ids {
        schema.episode_index[start]
    } else {
        *schema.episode_index.get(ordinal).ok_or_else(|| {
            DataError::schema(
                "episode_index",
                "one id per detected episode",
                format!("missing id for episode ordinal {ordinal}"),
            )
        })?
    };
    let episode_id = u32::try_from(episode_id).map_err(|_| {
        DataError::schema(
            "episode_index",
            "non-negative int32 episode id",
            format!("episode id {episode_id}"),
        )
    })?;
    let first_timestep = u32::try_from(schema.timestep[start]).map_err(|_| {
        DataError::schema(
            "timestep",
            "non-negative int32 timestep",
            format!("timestep {}", schema.timestep[start]),
        )
    })?;

    episodes.push(Episode {
        id: episode_id,
        row_start: start,
        len: end - start,
        first_timestep,
    });
    Ok(())
}

fn build_windows(episodes: &[Episode], shard_id: u16, config: &PushtConfig) -> Vec<WindowIndex> {
    let mut windows = Vec::new();
    for episode in episodes {
        if !split_contains(config.split, episode.id) || episode.len < config.horizon {
            continue;
        }
        for offset in 0..=(episode.len - config.horizon) {
            windows.push(WindowIndex {
                shard: shard_id,
                row_start: episode.row_start + offset,
                episode_id: episode.id,
                start_frame: episode.first_timestep + u32::try_from(offset).unwrap_or(u32::MAX),
            });
        }
    }
    windows
}

fn split_contains(split: Split, episode_id: u32) -> bool {
    let bucket = split_bucket(episode_id);
    match split {
        Split::Train => bucket != EVAL_BUCKET,
        Split::Eval => bucket == EVAL_BUCKET,
    }
}

fn split_bucket(episode_id: u32) -> u64 {
    let hash = blake3::keyed_hash(&SPLIT_KEY, &episode_id.to_le_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&hash.as_bytes()[..8]);
    u64::from_le_bytes(bytes) % EVAL_SPLIT_BUCKETS
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Array4};

    #[test]
    fn pusht_open_and_len() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let train_a = episode_for_split(Split::Train, 10);
        let train_b = episode_for_split(Split::Train, train_a + 1);
        let eval = episode_for_split(Split::Eval, train_b + 1);
        write_fixture(
            &dir.path().join("pusht_000.h5"),
            &[(train_a, 3), (eval, 4), (train_b, 2)],
        )?;

        let dataset = PushtDataset::new(PushtConfig::new(dir.path(), 2))?;

        assert_eq!(dataset.len(), 3);
        assert!(!dataset.is_empty());
        Ok(())
    }

    #[test]
    fn pusht_get_window_shapes() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let train = episode_for_split(Split::Train, 100);
        write_fixture(&dir.path().join("pusht_000.h5"), &[(train, 3)])?;

        let dataset = PushtDataset::new(PushtConfig::new(dir.path(), 2))?;
        let sample = dataset.get(0)?;

        assert_eq!(sample.frame_shape, (2, 224, 224, 3));
        assert_eq!(sample.frames_t.len(), 2 * 224 * 224 * 3);
        assert_eq!(sample.action_shape, (2, 2));
        assert_eq!(sample.actions, vec![0.0, 0.25, 1.0, 1.25]);
        assert_eq!(sample.meta.episode_id, train);
        assert_eq!(sample.meta.start_frame, 0);
        assert_eq!(sample.meta.shard, 0);
        Ok(())
    }

    #[test]
    fn pusht_public_lewm_schema_aliases_open() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let train = episode_for_split(Split::Train, 150);
        write_public_schema_fixture(&dir.path().join("pusht_000.h5"), &[(train, 3)])?;

        let dataset = PushtDataset::new(PushtConfig::new(dir.path(), 2))?;
        let sample = dataset.get(0)?;

        assert_eq!(dataset.len(), 2);
        assert_eq!(sample.frame_shape, (2, 4, 4, 3));
        assert_eq!(sample.frames_t.len(), 2 * 4 * 4 * 3);
        assert_eq!(sample.action_shape, (2, 2));
        assert_eq!(sample.actions, vec![0.0, 0.25, 1.0, 1.25]);
        assert_eq!(sample.meta.episode_id, train);
        assert_eq!(sample.meta.start_frame, 0);
        Ok(())
    }

    #[test]
    fn pusht_no_episode_crossing() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let train_a = episode_for_split(Split::Train, 200);
        let train_b = episode_for_split(Split::Train, train_a + 1);
        write_fixture(
            &dir.path().join("pusht_000.h5"),
            &[(train_a, 2), (train_b, 2)],
        )?;

        let dataset = PushtDataset::new(PushtConfig::new(dir.path(), 2))?;
        let first = dataset.get(0)?;
        let second = dataset.get(1)?;

        assert_eq!(dataset.len(), 2);
        assert_eq!(first.meta.episode_id, train_a);
        assert_eq!(first.actions, vec![0.0, 0.25, 1.0, 1.25]);
        assert_eq!(second.meta.episode_id, train_b);
        assert_eq!(second.actions, vec![2.0, 2.25, 3.0, 3.25]);
        Ok(())
    }

    #[test]
    fn pusht_short_episode_skipped() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let train_short = episode_for_split(Split::Train, 300);
        let train_long = episode_for_split(Split::Train, train_short + 1);
        write_fixture(
            &dir.path().join("pusht_000.h5"),
            &[(train_short, 1), (train_long, 3)],
        )?;

        let dataset = PushtDataset::new(PushtConfig::new(dir.path(), 2))?;

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.get(0)?.meta.episode_id, train_long);
        Ok(())
    }

    #[test]
    fn pusht_schema_mismatch_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("pusht_000.h5");
        let file = hdf5::File::create(&path)?;
        let observation = file.create_group("observation")?;
        file.new_dataset_builder()
            .with_data(&Array1::<i32>::from_vec(vec![1, 1]))
            .create("episode_index")?;
        file.new_dataset_builder()
            .with_data(&Array1::<i32>::from_vec(vec![0, 1]))
            .create("timestep")?;
        observation
            .new_dataset_builder()
            .with_data(&Array4::<u8>::zeros((2, 224, 224, 1)))
            .create("pixels")?;
        file.new_dataset_builder()
            .with_data(&Array2::<f32>::zeros((2, 2)))
            .create("action")?;

        let err = PushtDataset::new(PushtConfig::new(dir.path(), 2))
            .err()
            .ok_or("schema mismatch should fail")?;

        assert!(matches!(err, DataError::SchemaMismatch { .. }));
        Ok(())
    }

    fn episode_for_split(split: Split, start: u32) -> u32 {
        let mut episode_id = start;
        while !split_contains(split, episode_id) {
            episode_id = episode_id.saturating_add(1);
        }
        episode_id
    }

    fn write_fixture(
        path: &Path,
        episodes: &[(u32, usize)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let rows = episodes.iter().map(|(_, len)| *len).sum::<usize>();
        let file = hdf5::File::create(path)?;
        let observation = file.create_group("observation")?;

        let mut episode_index = Vec::with_capacity(rows);
        let mut timestep = Vec::with_capacity(rows);
        let mut row = 0usize;
        for (episode_id, len) in episodes {
            for step in 0..*len {
                episode_index.push(i32::try_from(*episode_id)?);
                timestep.push(i32::try_from(step)?);
                row += 1;
            }
        }
        debug_assert_eq!(row, rows);

        let pixels = Array4::from_shape_fn((rows, 224, 224, 3), |(r, y, x, c)| {
            u8::try_from((r + y + x + c) % 251).unwrap_or(0)
        });
        let actions = Array2::from_shape_fn((rows, 2), |(r, c)| {
            let row = u16::try_from(r).map(f32::from).unwrap_or(f32::INFINITY);
            let col = u16::try_from(c).map(f32::from).unwrap_or(f32::INFINITY);
            row + (col * 0.25)
        });

        file.new_dataset_builder()
            .with_data(&Array1::<i32>::from_vec(episode_index))
            .create("episode_index")?;
        file.new_dataset_builder()
            .with_data(&Array1::<i32>::from_vec(timestep))
            .create("timestep")?;
        observation
            .new_dataset_builder()
            .with_data(&pixels)
            .create("pixels")?;
        file.new_dataset_builder()
            .with_data(&actions)
            .create("action")?;

        Ok(())
    }

    fn write_public_schema_fixture(
        path: &Path,
        episodes: &[(u32, usize)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let rows = episodes.iter().map(|(_, len)| *len).sum::<usize>();
        let file = hdf5::File::create(path)?;

        let mut episode_idx = Vec::with_capacity(rows);
        let mut step_idx = Vec::with_capacity(rows);
        let mut row = 0usize;
        for (episode_id, len) in episodes {
            for step in 0..*len {
                episode_idx.push(i64::from(*episode_id));
                step_idx.push(i64::try_from(step)?);
                row += 1;
            }
        }
        debug_assert_eq!(row, rows);

        let pixels = Array4::from_shape_fn((rows, 4, 4, 3), |(r, y, x, c)| {
            u8::try_from((r + y + x + c) % 251).unwrap_or(0)
        });
        let actions = Array2::from_shape_fn((rows, 2), |(r, c)| {
            let row = u16::try_from(r).map(f32::from).unwrap_or(f32::INFINITY);
            let col = u16::try_from(c).map(f32::from).unwrap_or(f32::INFINITY);
            row + (col * 0.25)
        });

        file.new_dataset_builder()
            .with_data(&Array1::<i64>::from_vec(episode_idx))
            .create("episode_idx")?;
        file.new_dataset_builder()
            .with_data(&Array1::<i64>::from_vec(step_idx))
            .create("step_idx")?;
        file.new_dataset_builder()
            .with_data(&pixels)
            .create("pixels")?;
        file.new_dataset_builder()
            .with_data(&actions)
            .create("action")?;

        Ok(())
    }
}
