//! `SO-100` HDF5 streaming dataset loader.

use std::fmt;
use std::path::{Path, PathBuf};

use hdf5_metno as hdf5;
use ndarray::{Ix2, Ix4, s};
use parking_lot::Mutex;

use crate::{DataError, Sample, SampleMeta, Split};

const RGB_CHANNELS: usize = 3;
const SO100_ACTION_DIM: usize = 6;
const SHARD_ID: u16 = 0;

/// RFC 0012 held-out SO-100 episode IDs.
pub const SO100_HELD_OUT_EPISODES: [u32; 5] = [5, 14, 23, 31, 42];

/// Camera view selected from the decoded SO-100 HDF5 mirror.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum CameraView {
    /// Top fixed camera.
    #[default]
    Top,
    /// Wrist-mounted camera.
    Wrist,
}

impl CameraView {
    fn dataset_path(self) -> &'static str {
        match self {
            Self::Top => "observation/pixels_top",
            Self::Wrist => "observation/pixels_wrist",
        }
    }
}

/// `SO-100` HDF5 loader configuration.
#[derive(Debug, Clone)]
pub struct So100Config {
    /// Path to the pre-decoded RFC 0012 HDF5 file.
    pub hdf5_path: PathBuf,
    /// Episode split to expose.
    pub split: Split,
    /// Number of frames/actions in each sampled window.
    pub horizon: usize,
    /// Number of historical warm-up frames expected by downstream training.
    pub history_size: usize,
    /// Optional deterministic seed reserved for iterator shuffling.
    pub seed: Option<u64>,
    /// Camera view consumed by training.
    pub camera_view: CameraView,
    /// Optional path to persisted action statistics.
    pub stats_path: Option<PathBuf>,
}

impl So100Config {
    /// Create an `SO-100` config with training split, top camera, and schema
    /// validation through [`So100Dataset::from_hdf5`].
    #[must_use]
    pub fn new(hdf5_path: impl Into<PathBuf>, horizon: usize) -> Self {
        Self {
            hdf5_path: hdf5_path.into(),
            split: Split::Train,
            horizon,
            history_size: 0,
            seed: Some(0),
            camera_view: CameraView::Top,
            stats_path: None,
        }
    }
}

/// Streaming `SO-100` dataset backed by a pre-decoded HDF5 mirror.
pub struct So100Dataset {
    shard: Mutex<So100Hdf>,
    windows: Vec<WindowIndex>,
    config: So100Config,
}

impl fmt::Debug for So100Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("So100Dataset")
            .field("shard", &"<hdf5>")
            .field("windows", &self.windows.len())
            .field("config", &self.config)
            .finish()
    }
}

impl So100Dataset {
    /// Open a pre-decoded SO-100 HDF5 mirror and precompute sample windows.
    ///
    /// # Errors
    ///
    /// Returns an error when the config is invalid, the file cannot be opened,
    /// schema validation fails, or the selected split has no valid windows.
    pub fn from_hdf5(config: So100Config) -> Result<Self, DataError> {
        validate_config(&config)?;

        let (shard, windows) = So100Hdf::open(&config.hdf5_path, &config)?;
        if windows.is_empty() {
            return Err(DataError::EmptyDataset(format!(
                "no valid {:?} SO-100 windows with horizon {} in {}",
                config.split,
                config.horizon,
                config.hdf5_path.display()
            )));
        }

        Ok(Self {
            shard: Mutex::new(shard),
            windows,
            config,
        })
    }

    /// Raw Parquet + MP4 loading is intentionally out of scope for v1.
    ///
    /// # Errors
    ///
    /// Always returns an invalid-configuration error directing callers to the
    /// Python pre-decode pipeline.
    pub fn from_raw(
        _parquet_dir: impl AsRef<Path>,
        _mp4_dir: impl AsRef<Path>,
    ) -> Result<Self, DataError> {
        Err(DataError::InvalidConfig(
            "SO-100 raw Parquet/MP4 loading is out of scope for v1; run \
             python/decode_so100_to_h5.py and use So100Dataset::from_hdf5"
                .to_string(),
        ))
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

    /// Fetch a raw `SO-100` sample window.
    ///
    /// `idx` is interpreted modulo [`Self::len`] so virtual epochs can be
    /// longer than the physical window table.
    ///
    /// # Errors
    ///
    /// Returns an error if the dataset is empty, a window references invalid
    /// rows, or the underlying HDF5 read fails.
    pub fn get(&self, idx: usize) -> Result<Sample, DataError> {
        if self.windows.is_empty() {
            return Err(DataError::EmptyDataset(
                "cannot sample from an empty SO-100 dataset".to_string(),
            ));
        }

        let window = self.windows[idx % self.windows.len()];
        self.shard.lock().read_window(window, self.config.horizon)
    }
}

#[derive(Debug)]
struct So100Hdf {
    file: hdf5::File,
    pixel_dataset: &'static str,
    pixel_shape: [usize; 4],
    action_shape: [usize; 2],
}

impl So100Hdf {
    fn open(path: &Path, config: &So100Config) -> Result<(Self, Vec<WindowIndex>), DataError> {
        let file = hdf5::File::open(path)
            .map_err(|source| DataError::hdf5(format!("open {}", path.display()), source))?;
        let schema = HdfSchema::read(path, &file, config.camera_view)?;
        let episodes = build_episodes(&schema)?;
        let windows = build_windows(&episodes, config);
        let shard = Self {
            file,
            pixel_dataset: config.camera_view.dataset_path(),
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
            .dataset(self.pixel_dataset)
            .map_err(|source| DataError::hdf5(format!("open {}", self.pixel_dataset), source))?
            .read_slice::<u8, _, Ix4>(s![start..end, .., .., ..])
            .map_err(|source| DataError::hdf5("read SO-100 pixel window", source))?;
        let actions = self
            .file
            .dataset("action")
            .map_err(|source| DataError::hdf5("open action", source))?
            .read_slice::<f32, _, Ix2>(s![start..end, ..])
            .map_err(|source| DataError::hdf5("read SO-100 action window", source))?;

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
                shard: SHARD_ID,
            },
        })
    }
}

#[derive(Debug)]
struct HdfSchema {
    episode_index: Vec<i32>,
    timestep: Vec<i32>,
    pixel_shape: [usize; 4],
    action_shape: [usize; 2],
}

impl HdfSchema {
    fn read(path: &Path, file: &hdf5::File, camera_view: CameraView) -> Result<Self, DataError> {
        let episode_ds = dataset(file, "episode_index")?;
        let timestep_ds = dataset(file, "timestep")?;
        let pixels_ds = dataset(file, camera_view.dataset_path())?;
        let action_ds = dataset(file, "action")?;
        let joint_pos_ds = dataset(file, "joint_pos")?;

        require_dtype::<i32>(&episode_ds, "episode_index")?;
        require_dtype::<i32>(&timestep_ds, "timestep")?;
        require_dtype::<u8>(&pixels_ds, camera_view.dataset_path())?;
        require_dtype::<f32>(&action_ds, "action")?;
        require_dtype::<f32>(&joint_pos_ds, "joint_pos")?;

        let timestep_shape = timestep_ds.shape();
        let episode_shape = episode_ds.shape();
        let pixel_shape = vec_to_array::<4>(pixels_ds.shape(), camera_view.dataset_path())?;
        let action_shape = vec_to_array::<2>(action_ds.shape(), "action")?;
        let joint_pos_shape = vec_to_array::<2>(joint_pos_ds.shape(), "joint_pos")?;

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
                camera_view.dataset_path(),
                "last dimension 3 RGB channels",
                format!("shape {pixel_shape:?}"),
            ));
        }
        if action_shape[1] != SO100_ACTION_DIM {
            return Err(DataError::schema(
                "action",
                "shape (N, 6)",
                format!("shape {action_shape:?}"),
            ));
        }
        if joint_pos_shape[1] != SO100_ACTION_DIM {
            return Err(DataError::schema(
                "joint_pos",
                "shape (N, 6)",
                format!("shape {joint_pos_shape:?}"),
            ));
        }
        if timestep_shape[0] != pixel_shape[0]
            || timestep_shape[0] != action_shape[0]
            || timestep_shape[0] != joint_pos_shape[0]
        {
            return Err(DataError::schema(
                path.display().to_string(),
                "matching N across timestep, selected pixels, action, and joint_pos",
                format!(
                    "timestep={timestep_shape:?}, pixels={pixel_shape:?}, \
                     action={action_shape:?}, joint_pos={joint_pos_shape:?}"
                ),
            ));
        }

        Ok(Self {
            episode_index: episode_ds
                .read_raw()
                .map_err(|source| DataError::hdf5("read episode_index", source))?,
            timestep: timestep_ds
                .read_raw()
                .map_err(|source| DataError::hdf5("read timestep", source))?,
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
    row_start: usize,
    episode_id: u32,
    start_frame: u32,
}

fn validate_config(config: &So100Config) -> Result<(), DataError> {
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

fn dataset(file: &hdf5::File, name: &str) -> Result<hdf5::Dataset, DataError> {
    file.dataset(name)
        .map_err(|source| DataError::hdf5(format!("open {name}"), source))
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

fn vec_to_array<const N: usize>(shape: Vec<usize>, path: &str) -> Result<[usize; N], DataError> {
    shape.try_into().map_err(|shape: Vec<usize>| {
        DataError::schema(path, format!("{N}-D shape"), format!("shape {shape:?}"))
    })
}

fn build_episodes(schema: &HdfSchema) -> Result<Vec<Episode>, DataError> {
    if schema.timestep.is_empty() {
        return Err(DataError::EmptyDataset(
            "SO-100 HDF5 file has zero timesteps".to_string(),
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
    schema: &HdfSchema,
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

fn build_windows(episodes: &[Episode], config: &So100Config) -> Vec<WindowIndex> {
    let mut windows = Vec::new();
    for episode in episodes {
        if !so100_split_contains(config.split, episode.id) || episode.len < config.horizon {
            continue;
        }
        for offset in 0..=(episode.len - config.horizon) {
            windows.push(WindowIndex {
                row_start: episode.row_start + offset,
                episode_id: episode.id,
                start_frame: episode.first_timestep + u32::try_from(offset).unwrap_or(u32::MAX),
            });
        }
    }
    windows
}

fn so100_split_contains(split: Split, episode_id: u32) -> bool {
    let held_out = SO100_HELD_OUT_EPISODES.contains(&episode_id);
    match split {
        Split::Train => !held_out,
        Split::Eval => held_out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Array4};

    #[test]
    fn so100_hdf5_open_and_len() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("so100.h5");
        write_fixture(&path, &[(0, 3), (5, 3), (1, 2)])?;

        let dataset = So100Dataset::from_hdf5(So100Config::new(&path, 2))?;

        assert_eq!(dataset.len(), 3);
        assert!(!dataset.is_empty());
        Ok(())
    }

    #[test]
    fn so100_camera_view_select() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("so100.h5");
        write_fixture(&path, &[(0, 2)])?;
        let mut config = So100Config::new(&path, 2);
        config.camera_view = CameraView::Wrist;

        let dataset = So100Dataset::from_hdf5(config)?;
        let sample = dataset.get(0)?;

        assert_eq!(sample.frames_t[0], 100);
        assert_eq!(sample.frame_shape, (2, 224, 224, 3));
        Ok(())
    }

    #[test]
    fn so100_get_window_shapes() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("so100.h5");
        write_fixture(&path, &[(0, 3)])?;

        let dataset = So100Dataset::from_hdf5(So100Config::new(&path, 2))?;
        let sample = dataset.get(0)?;

        assert_eq!(sample.frame_shape, (2, 224, 224, 3));
        assert_eq!(sample.frames_t.len(), 2 * 224 * 224 * 3);
        assert_eq!(sample.action_shape, (2, 6));
        assert_eq!(
            sample.actions,
            vec![
                0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25
            ]
        );
        assert_eq!(sample.meta.episode_id, 0);
        assert_eq!(sample.meta.start_frame, 0);
        assert_eq!(sample.meta.shard, SHARD_ID);
        Ok(())
    }

    #[test]
    fn so100_holdout_episodes_disjoint() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("so100.h5");
        write_fixture(
            &path,
            &[(0, 1), (5, 1), (14, 1), (23, 1), (31, 1), (42, 1), (49, 1)],
        )?;

        let train = So100Dataset::from_hdf5(So100Config::new(&path, 1))?;
        let mut eval_config = So100Config::new(&path, 1);
        eval_config.split = Split::Eval;
        let eval = So100Dataset::from_hdf5(eval_config)?;

        let train_ids = sample_episode_ids(&train);
        let eval_ids = sample_episode_ids(&eval);

        assert_eq!(SO100_HELD_OUT_EPISODES, [5, 14, 23, 31, 42]);
        assert_eq!(train_ids, vec![0, 49]);
        assert_eq!(eval_ids, SO100_HELD_OUT_EPISODES.to_vec());
        assert!(
            train_ids
                .iter()
                .all(|id| !SO100_HELD_OUT_EPISODES.contains(id))
        );
        assert!(
            eval_ids
                .iter()
                .all(|id| SO100_HELD_OUT_EPISODES.contains(id))
        );
        Ok(())
    }

    #[test]
    fn so100_raw_loader_stub_is_explicit() {
        let err = So100Dataset::from_raw("data", "videos")
            .expect_err("raw loader is intentionally unsupported");
        assert!(
            matches!(err, DataError::InvalidConfig(message) if message.contains("out of scope"))
        );
    }

    #[test]
    fn so100_schema_mismatch_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("bad.h5");
        let file = hdf5::File::create(&path)?;
        let observation = file.create_group("observation")?;
        file.new_dataset_builder()
            .with_data(&Array1::<i32>::from_vec(vec![0, 0]))
            .create("episode_index")?;
        file.new_dataset_builder()
            .with_data(&Array1::<i32>::from_vec(vec![0, 1]))
            .create("timestep")?;
        observation
            .new_dataset_builder()
            .with_data(&Array4::<u8>::zeros((2, 224, 224, 1)))
            .create("pixels_top")?;
        file.new_dataset_builder()
            .with_data(&Array2::<f32>::zeros((2, 6)))
            .create("action")?;
        file.new_dataset_builder()
            .with_data(&Array2::<f32>::zeros((2, 6)))
            .create("joint_pos")?;

        let err = So100Dataset::from_hdf5(So100Config::new(&path, 2))
            .err()
            .ok_or("schema mismatch should fail")?;

        assert!(matches!(err, DataError::SchemaMismatch { .. }));
        Ok(())
    }

    fn sample_episode_ids(dataset: &So100Dataset) -> Vec<u32> {
        (0..dataset.len())
            .map(|idx| dataset.get(idx).expect("sample").meta.episode_id)
            .collect()
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
        for (episode_id, len) in episodes {
            for step in 0..*len {
                episode_index.push(i32::try_from(*episode_id)?);
                timestep.push(i32::try_from(step)?);
            }
        }

        let pixels_top = Array4::from_shape_fn((rows, 224, 224, 3), |(r, y, x, c)| {
            u8::try_from((r + y + x + c) % 251).unwrap_or(0)
        });
        let pixels_wrist = Array4::from_shape_fn((rows, 224, 224, 3), |(r, y, x, c)| {
            100u8.saturating_add(u8::try_from((r + y + x + c) % 101).unwrap_or(0))
        });
        let actions = Array2::from_shape_fn((rows, 6), |(r, c)| {
            let row = u16::try_from(r).map(f32::from).unwrap_or(f32::INFINITY);
            let col = u16::try_from(c).map(f32::from).unwrap_or(f32::INFINITY);
            row + (col * 0.25)
        });
        let joint_pos = Array2::from_shape_fn((rows, 6), |(r, c)| {
            let row = u16::try_from(r).map(f32::from).unwrap_or(f32::INFINITY);
            let col = u16::try_from(c).map(f32::from).unwrap_or(f32::INFINITY);
            row + (col * 0.5)
        });

        file.new_dataset_builder()
            .with_data(&Array1::<i32>::from_vec(episode_index))
            .create("episode_index")?;
        file.new_dataset_builder()
            .with_data(&Array1::<i32>::from_vec(timestep))
            .create("timestep")?;
        observation
            .new_dataset_builder()
            .with_data(&pixels_top)
            .create("pixels_top")?;
        observation
            .new_dataset_builder()
            .with_data(&pixels_wrist)
            .create("pixels_wrist")?;
        file.new_dataset_builder()
            .with_data(&actions)
            .create("action")?;
        file.new_dataset_builder()
            .with_data(&joint_pos)
            .create("joint_pos")?;

        Ok(())
    }
}
