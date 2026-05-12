//! Training-split action-stat computation for dataset normalization.

use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use crate::{DataError, PushtConfig, PushtDataset, Split, TransformStats};

const DEFAULT_STATS_HORIZON: usize = 1;
const HASH_BUFFER_BYTES: usize = 64 * 1024;
const PIXEL_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const PIXEL_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Dataset family supported by the run-once stats tool.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum StatsDataset {
    /// `PushT` HDF5 shards.
    Pusht,
}

/// Configuration for deterministic training-split statistics.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ComputeStatsConfig {
    /// Dataset family to inspect.
    pub dataset: StatsDataset,
    /// Dataset root: either a single HDF5 shard or a directory of shards.
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
    }
}

fn compute_pusht_stats(config: &ComputeStatsConfig) -> Result<DatasetStats, DataError> {
    if config.horizon == 0 {
        return Err(DataError::InvalidConfig(
            "stats horizon must be greater than zero".to_string(),
        ));
    }

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
            let row = u16::try_from(r).map(f32::from).unwrap_or(f32::INFINITY);
            let col = u16::try_from(c).map(f32::from).unwrap_or(f32::INFINITY);
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
}
