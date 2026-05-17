//! Training-split action-stat computation for dataset normalization.

use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use crate::{
    DataError, PushtConfig, PushtDataset, So100Config, So100Dataset, Split, TransformStats,
};

const DEFAULT_STATS_HORIZON: usize = 1;
const HASH_BUFFER_BYTES: usize = 64 * 1024;
const PIXEL_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const PIXEL_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Dataset family supported by the run-once stats tool.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum StatsDataset {
    /// `PushT` HDF5 shards.
    Pusht,
    /// Pre-decoded `SO-100` HDF5 mirror.
    So100,
}

/// Configuration for deterministic training-split statistics.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ComputeStatsConfig {
    /// Dataset family to inspect.
    pub dataset: StatsDataset,
    /// Dataset root: either a single HDF5 shard or a directory of shards.
    /// For `SO-100`, this points to the pre-decoded HDF5 mirror file or to a
    /// directory containing exactly one mirror file.
    pub root_path: PathBuf,
    /// Deterministic seed recorded by the caller; current `PushT` stats read rows in sorted order.
    pub seed: u64,
    /// Temporal window length used when exposing the training split.
    pub horizon: usize,
    /// Validate the HDF5 schema before reading samples.
    pub validate_schema: bool,
}

impl ComputeStatsConfig {
    /// Build a `PushT` stats config with the deterministic defaults used by CI.
    pub fn pusht(root_path: impl Into<PathBuf>) -> Self {
        Self {
            dataset: StatsDataset::Pusht,
            root_path: root_path.into(),
            seed: 0,
            horizon: DEFAULT_STATS_HORIZON,
            validate_schema: true,
        }
    }

    /// Build an `SO-100` stats config with the deterministic defaults used by CI.
    pub fn so100(hdf5_path: impl Into<PathBuf>) -> Self {
        Self {
            dataset: StatsDataset::So100,
            root_path: hdf5_path.into(),
            seed: 0,
            horizon: DEFAULT_STATS_HORIZON,
            validate_schema: true,
        }
    }
}

/// Persisted dataset statistics.
pub type DatasetStats = TransformStats;

/// Compute deterministic action normalization stats for a dataset.
///
/// # Errors
///
/// Returns an error when the dataset cannot be opened, samples are malformed,
/// no training actions are available, or the underlying bytes cannot be hashed.
pub fn compute_stats(config: &ComputeStatsConfig) -> Result<DatasetStats, DataError> {
    match config.dataset {
        StatsDataset::Pusht => compute_pusht_stats(config),
        StatsDataset::So100 => compute_so100_stats(config),
    }
}

fn compute_pusht_stats(config: &ComputeStatsConfig) -> Result<DatasetStats, DataError> {
    validate_stats_config(config)?;

    let content_hash = hash_dataset_bytes(&config.root_path)?;
    let mut pusht_config = PushtConfig::new(config.root_path.clone(), config.horizon);
    pusht_config.split = Split::Train;
    pusht_config.seed = Some(config.seed);
    pusht_config.validate_schema = config.validate_schema;
    let dataset = PushtDataset::new(pusht_config)?;

    let mut accum = ActionAccumulator::default();
    for index in 0..dataset.len() {
        let sample = dataset.get(index)?;
        accum.push_actions(&sample.actions, sample.action_shape)?;
    }

    let (mean, std, n_train_samples) = accum.finish()?;
    TransformStats::new(
        mean,
        std,
        PIXEL_MEAN,
        PIXEL_STD,
        n_train_samples,
        content_hash,
    )
}

fn compute_so100_stats(config: &ComputeStatsConfig) -> Result<DatasetStats, DataError> {
    validate_stats_config(config)?;

    let hdf5_path = resolve_so100_hdf5_path(&config.root_path)?;
    let content_hash = hash_dataset_bytes(&hdf5_path)?;
    let mut so100_config = So100Config::new(hdf5_path, config.horizon);
    so100_config.split = Split::Train;
    so100_config.seed = Some(config.seed);
    let dataset = So100Dataset::from_hdf5(so100_config)?;

    let mut accum = ActionAccumulator::default();
    for index in 0..dataset.len() {
        let sample = dataset.get(index)?;
        accum.push_actions(&sample.actions, sample.action_shape)?;
    }

    let (mean, std, n_train_samples) = accum.finish()?;
    TransformStats::new(
        mean,
        std,
        PIXEL_MEAN,
        PIXEL_STD,
        n_train_samples,
        content_hash,
    )
}

fn validate_stats_config(config: &ComputeStatsConfig) -> Result<(), DataError> {
    if config.horizon == 0 {
        return Err(DataError::InvalidConfig(
            "stats horizon must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

fn resolve_so100_hdf5_path(root_path: &Path) -> Result<PathBuf, DataError> {
    let paths = data_files(root_path)?;
    match paths.as_slice() {
        [path] => Ok(path.clone()),
        _ => Err(DataError::InvalidConfig(format!(
            "SO-100 stats expects exactly one .h5 or .hdf5 mirror under {}, found {}",
            root_path.display(),
            paths.len()
        ))),
    }
}

#[derive(Debug, Default)]
struct ActionAccumulator {
    action_dim: Option<usize>,
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
    count: usize,
}

impl ActionAccumulator {
    fn push_actions(
        &mut self,
        actions: &[f32],
        action_shape: (usize, usize),
    ) -> Result<(), DataError> {
        let (time, action_dim) = action_shape;
        if action_dim == 0 {
            return Err(DataError::InvalidTransform(
                "sample action_dim must be greater than zero".to_string(),
            ));
        }
        let expected_len = time.checked_mul(action_dim).ok_or_else(|| {
            DataError::InvalidTransform(format!(
                "sample action shape {action_shape:?} overflows flat length"
            ))
        })?;
        if actions.len() != expected_len {
            return Err(DataError::InvalidTransform(format!(
                "sample action buffer has {} values but shape {action_shape:?} requires {expected_len}",
                actions.len()
            )));
        }
        self.ensure_action_dim(action_dim)?;

        for action in actions.chunks_exact(action_dim) {
            for (dim, value) in action.iter().copied().enumerate() {
                if !value.is_finite() {
                    return Err(DataError::InvalidTransform(format!(
                        "action value at dim {dim} must be finite"
                    )));
                }
                let value = f64::from(value);
                self.sum[dim] += value;
                self.sum_sq[dim] += value * value;
            }
            self.count += 1;
        }
        Ok(())
    }

    fn ensure_action_dim(&mut self, action_dim: usize) -> Result<(), DataError> {
        match self.action_dim {
            Some(existing) if existing != action_dim => Err(DataError::InvalidTransform(format!(
                "sample action_dim {action_dim} does not match previous action_dim {existing}"
            ))),
            Some(_) => Ok(()),
            None => {
                self.action_dim = Some(action_dim);
                self.sum = vec![0.0; action_dim];
                self.sum_sq = vec![0.0; action_dim];
                Ok(())
            },
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn finish(self) -> Result<(Vec<f32>, Vec<f32>, i64), DataError> {
        let action_dim = self.action_dim.ok_or_else(|| {
            DataError::EmptyDataset("no training actions available for stats".to_string())
        })?;
        if self.count == 0 {
            return Err(DataError::EmptyDataset(
                "no training actions available for stats".to_string(),
            ));
        }

        let count = self.count as f64;
        let mut mean = Vec::with_capacity(action_dim);
        let mut std = Vec::with_capacity(action_dim);
        for dim in 0..action_dim {
            let dim_mean = self.sum[dim] / count;
            let variance = (self.sum_sq[dim] / count) - (dim_mean * dim_mean);
            mean.push(dim_mean as f32);
            std.push(variance.max(0.0).sqrt() as f32);
        }
        let n_train_samples = i64::try_from(self.count).map_err(|_| {
            DataError::InvalidTransform(format!("n_train_samples {} exceeds i64::MAX", self.count))
        })?;

        Ok((mean, std, n_train_samples))
    }
}

fn hash_dataset_bytes(root_path: &Path) -> Result<[u8; 32], DataError> {
    let paths = data_files(root_path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];

    for path in paths {
        let mut file = fs::File::open(&path).map_err(|source| DataError::io(&path, source))?;
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|source| DataError::io(&path, source))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    }

    Ok(*hasher.finalize().as_bytes())
}

fn data_files(root_path: &Path) -> Result<Vec<PathBuf>, DataError> {
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

#[cfg(test)]
mod tests {
    use hdf5_metno as hdf5;
    use ndarray::{Array1, Array2, Array4};

    use super::*;

    #[test]
    fn compute_stats_deterministic() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let shard_path = dir.path().join("pusht_000.h5");
        write_fixture(&shard_path, 64)?;

        let mut config = ComputeStatsConfig::pusht(dir.path());
        config.seed = 123;
        let stats_a = compute_stats(&config)?;
        let stats_b = compute_stats(&config)?;
        let path_a = dir.path().join("stats_a.safetensors");
        let path_b = dir.path().join("stats_b.safetensors");

        stats_a.save_safetensors(&path_a)?;
        stats_b.save_safetensors(&path_b)?;

        assert_eq!(fs::read(&path_a)?, fs::read(&path_b)?);
        assert_eq!(stats_a, stats_b);
        assert_eq!(stats_a.content_hash, hash_dataset_bytes(dir.path())?);
        assert!(stats_a.n_train_samples > 0);

        let normalizer = stats_a.action_normalizer()?;
        let raw = [1.25, -2.0];
        let mapped = normalizer.apply(&raw)?;
        let restored = normalizer.inverse(&mapped)?;
        for (left, right) in restored.iter().zip(raw) {
            assert!((*left - right).abs() <= 1e-5);
        }

        Ok(())
    }

    #[test]
    fn so100_compute_stats_per_dim() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let hdf5_path = dir.path().join("so100.h5");
        write_so100_fixture(&hdf5_path)?;

        let stats = compute_stats(&ComputeStatsConfig::so100(dir.path()))?;
        let stats_path = dir.path().join("stats.safetensors");
        stats.save_safetensors(&stats_path)?;
        let loaded = TransformStats::load_safetensors(&stats_path)?;

        assert_eq!(stats, loaded);
        assert_eq!(stats.action_mean, vec![3.0, 14.0, 0.0, 7.0, 2.0, 11.0]);
        assert_close(stats.action_std[2], 1.0);
        assert_close(stats.action_std[4], 1.0);
        assert_close(stats.action_std[0], population_std(&[1.0, 3.0, 5.0]));
        assert_close(stats.action_std[1], population_std(&[10.0, 14.0, 18.0]));
        assert_close(stats.action_std[3], population_std(&[5.0, 7.0, 9.0]));
        assert_close(stats.action_std[5], population_std(&[7.0, 11.0, 15.0]));
        assert_eq!(stats.n_train_samples, 3);
        assert_eq!(stats.content_hash, hash_dataset_bytes(&hdf5_path)?);

        let normalizer = loaded.action_normalizer()?;
        assert_eq!(normalizer.action_dim(), 6);
        let restored = normalizer.inverse(&normalizer.apply(&[3.0, 14.0, 0.0, 7.0, 2.0, 11.0])?)?;
        for (left, right) in restored.iter().zip([3.0, 14.0, 0.0, 7.0, 2.0, 11.0]) {
            assert_close(*left, right);
        }

        Ok(())
    }

    fn write_fixture(path: &Path, rows: usize) -> Result<(), Box<dyn std::error::Error>> {
        let file = hdf5::File::create(path)?;
        let observation = file.create_group("observation")?;

        let episode_index = Array1::from_iter(
            (0..rows)
                .map(i32::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        );
        let timestep = Array1::<i32>::zeros(rows);
        let pixels = Array4::from_shape_fn((rows, 2, 2, 3), |(r, y, x, c)| {
            u8::try_from((r + y + x + c) % 251).unwrap_or(0)
        });
        let actions = Array2::from_shape_fn((rows, 2), |(r, c)| {
            let row = u16::try_from(r).map_or(f32::INFINITY, f32::from);
            let col = u16::try_from(c).map_or(f32::INFINITY, f32::from);
            row + (col * -0.5)
        });

        file.new_dataset_builder()
            .with_data(&episode_index)
            .create("episode_index")?;
        file.new_dataset_builder()
            .with_data(&timestep)
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

    fn write_so100_fixture(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let file = hdf5::File::create(path)?;
        let observation = file.create_group("observation")?;
        let rows = 5usize;

        let episode_index = Array1::<i32>::from_vec(vec![0, 0, 5, 5, 1]);
        let timestep = Array1::<i32>::from_vec(vec![0, 1, 0, 1, 0]);
        let pixels_top = Array4::from_shape_fn((rows, 2, 2, 3), |(r, y, x, c)| {
            u8::try_from((r + y + x + c) % 251).unwrap_or(0)
        });
        let pixels_wrist = Array4::from_shape_fn((rows, 2, 2, 3), |(r, y, x, c)| {
            100u8.saturating_add(u8::try_from((r + y + x + c) % 101).unwrap_or(0))
        });
        let actions = Array2::<f32>::from_shape_vec(
            (rows, 6),
            vec![
                1.0, 10.0, 0.0, 5.0, 2.0, 7.0, 3.0, 14.0, 0.0, 7.0, 2.0, 11.0, 999.0, 999.0, 999.0,
                999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 5.0, 18.0, 0.0, 9.0,
                2.0, 15.0,
            ],
        )?;
        let joint_pos = Array2::<f32>::zeros((rows, 6));

        file.new_dataset_builder()
            .with_data(&episode_index)
            .create("episode_index")?;
        file.new_dataset_builder()
            .with_data(&timestep)
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

    fn population_std(values: &[f64]) -> f64 {
        let len = u32::try_from(values.len())
            .map(f64::from)
            .expect("fixture length fits in u32");
        let mean = values.iter().copied().sum::<f64>() / len;
        let variance = values
            .iter()
            .copied()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / len;
        variance.sqrt()
    }

    fn assert_close(left: impl Into<f64>, right: impl Into<f64>) {
        let left = left.into();
        let right = right.into();
        assert!((left - right).abs() <= 1e-5, "left={left}, right={right}");
    }
}
