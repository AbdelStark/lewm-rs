//! Multi-worker data prefetch for collated batches.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use crossbeam::channel::{Receiver, Sender, bounded};
use lewm_core::{DATA_SHUFFLE_STREAM, substream_seed};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha20Rng;
use tokio::task::JoinHandle;

use crate::{
    ActionNormalizer, Batch, BatchBackend, BatchDtype, DataError, HostBackend, HostDevice,
    ImagePreprocessor, PushtDataset, Sample, So100Dataset, collate,
};

/// Queue-depth metric emitted by [`Prefetcher::try_next`].
pub const DATA_QUEUE_DEPTH_METRIC: &str = "data/queue_depth";

const DEFAULT_NUM_WORKERS: usize = 4;
const DEFAULT_CHANNEL_CAPACITY: usize = 4;
const DEFAULT_BATCH_SIZE: usize = 64;
const WORKER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
const FATAL_CONSECUTIVE_ERRORS: usize = 2;

/// Common dataset interface consumed by the prefetcher.
pub trait Dataset: Send + Sync {
    /// Number of sample windows available for one physical epoch.
    fn len(&self) -> usize;

    /// Return whether the dataset has no sample windows.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fetch one raw sample by index.
    ///
    /// # Errors
    ///
    /// Returns a data error when the underlying loader cannot produce the
    /// requested sample.
    fn get(&self, idx: usize) -> Result<Sample, DataError>;
}

impl Dataset for PushtDataset {
    fn len(&self) -> usize {
        Self::len(self)
    }

    fn is_empty(&self) -> bool {
        Self::is_empty(self)
    }

    fn get(&self, idx: usize) -> Result<Sample, DataError> {
        Self::get(self, idx)
    }
}

impl Dataset for So100Dataset {
    fn len(&self) -> usize {
        Self::len(self)
    }

    fn is_empty(&self) -> bool {
        Self::is_empty(self)
    }

    fn get(&self, idx: usize) -> Result<Sample, DataError> {
        Self::get(self, idx)
    }
}

/// Prefetch worker and bounded-channel configuration.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PrefetcherConfig {
    /// Number of blocking-pool workers.
    pub num_workers: usize,
    /// Maximum number of collated batches buffered for the consumer.
    pub channel_capacity: usize,
    /// Number of samples per collated batch.
    pub batch_size: usize,
    /// Deterministic epoch shuffle seed.
    pub epoch_seed: u64,
}

impl PrefetcherConfig {
    /// Build a config with the RFC 0004 worker/channel defaults.
    #[must_use]
    pub fn new(batch_size: usize, epoch_seed: u64) -> Self {
        Self {
            batch_size,
            epoch_seed,
            ..Self::default()
        }
    }

    fn validate(&self) -> Result<(), DataError> {
        if self.num_workers == 0 {
            return Err(DataError::InvalidConfig(
                "prefetch num_workers must be greater than zero".to_string(),
            ));
        }
        if self.channel_capacity == 0 {
            return Err(DataError::InvalidConfig(
                "prefetch channel_capacity must be greater than zero".to_string(),
            ));
        }
        if self.batch_size == 0 {
            return Err(DataError::InvalidConfig(
                "prefetch batch_size must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for PrefetcherConfig {
    fn default() -> Self {
        Self {
            num_workers: DEFAULT_NUM_WORKERS,
            channel_capacity: DEFAULT_CHANNEL_CAPACITY,
            batch_size: DEFAULT_BATCH_SIZE,
            epoch_seed: 0,
        }
    }
}

/// Multi-worker prefetcher yielding collated batches.
#[derive(Debug)]
pub struct Prefetcher<B: BatchBackend> {
    rx: Receiver<PrefetchMessage<B>>,
    _handle: PrefetchHandle,
}

impl<B: BatchBackend> Prefetcher<B> {
    /// Start blocking-pool workers for a deterministic physical epoch.
    ///
    /// # Errors
    ///
    /// Returns an error when the config is invalid, the dataset is empty, the
    /// requested dtype is unsupported, or no Tokio runtime is active.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new<D: Dataset + 'static>(
        dataset: Arc<D>,
        config: PrefetcherConfig,
        device: B::Device,
        image_preproc: ImagePreprocessor,
        action_norm: ActionNormalizer,
        dtype: BatchDtype,
    ) -> Result<Self, DataError> {
        config.validate()?;
        if dataset.is_empty() {
            return Err(DataError::EmptyDataset(
                "cannot prefetch from an empty dataset".to_string(),
            ));
        }
        if !B::supports_dtype(&device, dtype) {
            return Err(DataError::InvalidConfig(format!(
                "backend {} on {:?} does not support requested pixel dtype {:?}",
                B::name(&device),
                device,
                dtype
            )));
        }

        let runtime = tokio::runtime::Handle::try_current().map_err(|source| {
            DataError::InvalidConfig(format!(
                "Prefetcher::new requires an active Tokio runtime: {source}"
            ))
        })?;
        let (tx, rx) = bounded(config.channel_capacity);
        let stop = Arc::new(AtomicBool::new(false));
        let indices = Arc::new(shuffled_indices(dataset.len(), config.epoch_seed));
        let cursor = Arc::new(AtomicUsize::new(0));
        let (done_tx, done_rx) = bounded(config.num_workers);
        let mut worker_count = 0usize;

        for worker_id in 0..config.num_workers {
            let _worker = spawn_worker(
                &runtime,
                WorkerArgs {
                    worker_id,
                    dataset: Arc::clone(&dataset),
                    indices: Arc::clone(&indices),
                    cursor: Arc::clone(&cursor),
                    stop: Arc::clone(&stop),
                    tx: tx.clone(),
                    done_tx: done_tx.clone(),
                    config: config.clone(),
                    device: device.clone(),
                    image_preproc: image_preproc.clone(),
                    action_norm: action_norm.clone(),
                    dtype,
                },
            );
            worker_count += 1;
        }
        drop(tx);
        drop(done_tx);

        Ok(Self {
            rx,
            _handle: PrefetchHandle {
                stop,
                done_rx,
                worker_count,
            },
        })
    }

    /// Receive the next batch while preserving worker fatal errors.
    ///
    /// # Errors
    ///
    /// Returns a worker data error if prefetching reached a fatal loader or
    /// collation error.
    pub fn try_next(&mut self) -> Result<Option<Batch<B>>, DataError> {
        self.emit_queue_depth();
        match self.rx.recv() {
            Ok(PrefetchMessage::Batch(batch)) => Ok(Some(batch)),
            Ok(PrefetchMessage::Fatal(err)) => Err(err),
            Err(_) => Ok(None),
        }
    }

    /// Return the current number of ready batches buffered for the consumer.
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        self.rx.len()
    }

    fn emit_queue_depth(&self) {
        tracing::debug!(
            metric = DATA_QUEUE_DEPTH_METRIC,
            queue_depth = self.queue_depth(),
            "prefetch queue depth"
        );
    }
}

impl<B: BatchBackend> Iterator for Prefetcher<B> {
    type Item = Batch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.try_next() {
            Ok(batch) => batch,
            Err(err) => {
                tracing::error!(error = %err, "prefetch worker fatal error");
                None
            },
        }
    }
}

#[derive(Debug)]
struct PrefetchHandle {
    stop: Arc<AtomicBool>,
    done_rx: Receiver<()>,
    worker_count: usize,
}

impl Drop for PrefetchHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        for _ in 0..self.worker_count {
            if self.done_rx.recv_timeout(WORKER_SHUTDOWN_TIMEOUT).is_err() {
                break;
            }
        }
    }
}

#[derive(Debug)]
enum PrefetchMessage<B: BatchBackend> {
    Batch(Batch<B>),
    Fatal(DataError),
}

struct WorkerArgs<B: BatchBackend, D: Dataset + 'static> {
    worker_id: usize,
    dataset: Arc<D>,
    indices: Arc<Vec<usize>>,
    cursor: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    tx: Sender<PrefetchMessage<B>>,
    done_tx: Sender<()>,
    config: PrefetcherConfig,
    device: B::Device,
    image_preproc: ImagePreprocessor,
    action_norm: ActionNormalizer,
    dtype: BatchDtype,
}

fn spawn_worker<B, D>(runtime: &tokio::runtime::Handle, args: WorkerArgs<B, D>) -> JoinHandle<()>
where
    B: BatchBackend,
    D: Dataset + 'static,
{
    runtime.spawn_blocking(move || run_worker(args))
}

fn run_worker<B, D>(args: WorkerArgs<B, D>)
where
    B: BatchBackend,
    D: Dataset + 'static,
{
    let WorkerArgs {
        worker_id,
        dataset,
        indices,
        cursor,
        stop,
        tx,
        done_tx,
        config,
        device,
        image_preproc,
        action_norm,
        dtype,
    } = args;
    let mut consecutive_errors = 0usize;

    while !stop.load(Ordering::Acquire) {
        let start = cursor.fetch_add(config.batch_size, Ordering::AcqRel);
        if start >= indices.len() {
            break;
        }
        let end = start.saturating_add(config.batch_size).min(indices.len());
        let mut samples = Vec::with_capacity(end - start);

        for index in &indices[start..end] {
            if stop.load(Ordering::Acquire) {
                break;
            }
            match dataset.get(*index) {
                Ok(sample) => {
                    consecutive_errors = 0;
                    samples.push(sample);
                },
                Err(err) => {
                    consecutive_errors += 1;
                    tracing::warn!(
                        worker_id,
                        index = *index,
                        error = %err,
                        "recoverable prefetch read error"
                    );
                    if consecutive_errors >= FATAL_CONSECUTIVE_ERRORS {
                        send_fatal(&tx, DataError::ChannelClosed);
                        let _ignored = done_tx.send(());
                        return;
                    }
                },
            }
        }

        if samples.is_empty() {
            continue;
        }
        match collate::<B>(&samples, &image_preproc, &action_norm, &device, dtype) {
            Ok(batch) => {
                if tx.send(PrefetchMessage::Batch(batch)).is_err() {
                    break;
                }
            },
            Err(err) => {
                send_fatal(&tx, err);
                break;
            },
        }
    }

    let _ignored = done_tx.send(());
}

fn send_fatal<B: BatchBackend>(tx: &Sender<PrefetchMessage<B>>, err: DataError) {
    let _ignored = tx.send(PrefetchMessage::Fatal(err));
}

fn shuffled_indices(len: usize, epoch_seed: u64) -> Vec<usize> {
    let mut indices = (0..len).collect::<Vec<_>>();
    let mut rng = ChaCha20Rng::from_seed(substream_seed(epoch_seed, DATA_SHUFFLE_STREAM));
    indices.shuffle(&mut rng);
    indices
}

/// Convenience alias for host-memory data-pipeline smoke tests.
pub type HostPrefetcher = Prefetcher<HostBackend>;

/// Convenience device alias for host-memory prefetching.
pub type HostPrefetchDevice = HostDevice;

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeSet,
        sync::atomic::{AtomicUsize, Ordering},
        time::Instant,
    };

    use super::*;

    #[derive(Debug)]
    struct SyntheticDataset {
        len: usize,
        failures: Vec<bool>,
        delay: Duration,
        calls: AtomicUsize,
    }

    impl SyntheticDataset {
        fn new(len: usize) -> Self {
            Self {
                len,
                failures: vec![false; len],
                delay: Duration::ZERO,
                calls: AtomicUsize::new(0),
            }
        }

        fn with_delay(mut self, delay: Duration) -> Self {
            self.delay = delay;
            self
        }

        fn with_failures(mut self, failures: Vec<bool>) -> Self {
            self.failures = failures;
            self
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Acquire)
        }
    }

    impl Dataset for SyntheticDataset {
        fn len(&self) -> usize {
            self.len
        }

        fn get(&self, idx: usize) -> Result<Sample, DataError> {
            self.calls.fetch_add(1, Ordering::AcqRel);
            if !self.delay.is_zero() {
                std::thread::sleep(self.delay);
            }
            if self.failures.get(idx).copied().unwrap_or(false) {
                return Err(DataError::InvalidConfig(format!(
                    "synthetic corrupted shard at index {idx}"
                )));
            }
            Ok(sample(idx))
        }
    }

    #[test]
    fn data_shuffle_deterministic() {
        let left = shuffled_indices(32, 7);
        let right = shuffled_indices(32, 7);
        let different_seed = shuffled_indices(32, 8);

        assert_eq!(left, right);
        assert_ne!(left, different_seed);
        assert_eq!(left.iter().copied().collect::<BTreeSet<_>>().len(), 32);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn prefetcher_clean_shutdown() -> Result<(), Box<dyn std::error::Error>> {
        let dataset = Arc::new(SyntheticDataset::new(10_000).with_delay(Duration::from_millis(2)));
        let config = PrefetcherConfig {
            num_workers: 4,
            channel_capacity: 1,
            batch_size: 1,
            epoch_seed: 42,
        };
        let mut prefetcher = host_prefetcher(Arc::clone(&dataset), config)?;

        let first = prefetcher.try_next()?;
        assert!(first.is_some());
        let started_calls = dataset.calls();
        let start = Instant::now(); // determinism-lint: allow Instant::now test timing
        drop(prefetcher);

        assert!(start.elapsed() < Duration::from_secs(1));
        assert!(dataset.calls() >= started_calls);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn prefetcher_throughput_at_least_60_bps() -> Result<(), Box<dyn std::error::Error>> {
        let len = 1_024;
        let batch_size = 8;
        let expected_batches = len / batch_size;
        let dataset = Arc::new(SyntheticDataset::new(len));
        let config = PrefetcherConfig {
            num_workers: 4,
            channel_capacity: 4,
            batch_size,
            epoch_seed: 7,
        };
        let mut prefetcher = host_prefetcher(dataset, config)?;

        let start = Instant::now(); // determinism-lint: allow Instant::now test timing
        let mut batches = 0usize;
        while prefetcher.try_next()?.is_some() {
            batches += 1;
        }

        assert_eq!(batches, expected_batches);
        let batches_f64 = f64::from(u32::try_from(batches).unwrap_or(u32::MAX));
        let batches_per_sec = batches_f64 / start.elapsed().as_secs_f64();
        assert!(
            batches_per_sec >= 60.0,
            "expected >=60 batches/sec, observed {batches_per_sec:.2}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn corrupted_shard_recoverable_then_fatal() -> Result<(), Box<dyn std::error::Error>> {
        let seed = 11;
        let order = shuffled_indices(4, seed);
        let mut failures = vec![false; 4];
        failures[order[0]] = true;
        failures[order[2]] = true;
        failures[order[3]] = true;

        let dataset = Arc::new(SyntheticDataset::new(4).with_failures(failures));
        let config = PrefetcherConfig {
            num_workers: 1,
            channel_capacity: 1,
            batch_size: 1,
            epoch_seed: seed,
        };
        let mut prefetcher = host_prefetcher(dataset, config)?;

        assert!(prefetcher.try_next()?.is_some());
        let err = prefetcher
            .try_next()
            .err()
            .ok_or("expected fatal prefetch error")?;
        assert!(matches!(err, DataError::ChannelClosed));
        Ok(())
    }

    fn host_prefetcher(
        dataset: Arc<SyntheticDataset>,
        config: PrefetcherConfig,
    ) -> Result<HostPrefetcher, DataError> {
        Prefetcher::<HostBackend>::new(
            dataset,
            config,
            HostDevice::Cpu,
            ImagePreprocessor {
                target_size: 1,
                ..ImagePreprocessor::default()
            },
            ActionNormalizer::new(vec![0.0], vec![1.0])?,
            BatchDtype::F32,
        )
    }

    fn sample(idx: usize) -> Sample {
        let pixel = u8::try_from(idx % 255).unwrap_or(0);
        Sample {
            frames_t: vec![pixel, pixel.saturating_add(1), pixel.saturating_add(2)],
            frame_shape: (1, 1, 1, 3),
            actions: vec![f32::from(pixel)],
            action_shape: (1, 1),
            meta: crate::SampleMeta {
                episode_id: u32::try_from(idx).unwrap_or(u32::MAX),
                start_frame: 0,
                shard: 0,
            },
        }
    }
}
