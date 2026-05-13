//! Minimal `TensorBoard` event writer for scalar summaries.

use std::{
    fmt,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::Mutex,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::{MetricName, MetricSink, TelemetryContext, TelemetryError};

const DEFAULT_FLUSH_RECORDS: usize = 1_000;
const DEFAULT_FLUSH_INTERVAL: Duration = Duration::from_secs(5);
const FILE_VERSION: &str = "brain.Event:2";
const CRC32C_POLY: u32 = 0x82f6_3b78;
const CRC_MASK_DELTA: u32 = 0xa282_ead8;

/// `TensorBoard` `TFRecord` event writer.
pub struct TensorboardWriter {
    tb_dir: PathBuf,
    event_path: PathBuf,
    flush_records: usize,
    flush_interval: Duration,
    state: Mutex<TensorboardState>,
}

impl fmt::Debug for TensorboardWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorboardWriter")
            .field("tb_dir", &self.tb_dir)
            .field("event_path", &self.event_path)
            .field("flush_records", &self.flush_records)
            .field("flush_interval", &self.flush_interval)
            .finish_non_exhaustive()
    }
}

struct TensorboardState {
    sink: BufWriter<File>,
    records_since_flush: usize,
    last_flush: Instant,
}

impl TensorboardWriter {
    /// Create a writer under `<root>/tb/events.out.tfevents.<timestamp>.<run_id>`.
    ///
    /// # Errors
    ///
    /// Returns an error when the `TensorBoard` directory or event file cannot be created.
    pub fn new(root: impl AsRef<Path>, run_id: &str) -> Result<Self, TelemetryError> {
        Self::with_flush_policy(root, run_id, DEFAULT_FLUSH_RECORDS, DEFAULT_FLUSH_INTERVAL)
    }

    /// Create a writer with an explicit buffered flush policy.
    ///
    /// # Errors
    ///
    /// Returns an error when the `TensorBoard` directory or event file cannot be created.
    pub fn with_flush_policy(
        root: impl AsRef<Path>,
        run_id: &str,
        flush_records: usize,
        flush_interval: Duration,
    ) -> Result<Self, TelemetryError> {
        validate_flush_policy(flush_records, flush_interval)?;
        if run_id.trim().is_empty() {
            return Err(TelemetryError::InvalidConfig(
                "TensorBoard run_id must be non-empty".to_string(),
            ));
        }

        let tb_dir = root.as_ref().join("tb");
        fs::create_dir_all(&tb_dir).map_err(TelemetryError::sink)?;
        let timestamp = unix_seconds();
        let event_path = tb_dir.join(format!("events.out.tfevents.{timestamp}.{run_id}"));
        let mut sink = BufWriter::new(File::create(&event_path).map_err(TelemetryError::sink)?);
        write_tfrecord(&mut sink, &event_file_version(now_wall_time()))?;
        sink.flush().map_err(TelemetryError::sink)?;

        Ok(Self {
            tb_dir,
            event_path,
            flush_records,
            flush_interval,
            state: Mutex::new(TensorboardState {
                sink,
                records_since_flush: 0,
                last_flush: Instant::now(), // determinism-lint: allow Instant::now telemetry flush cadence
            }),
        })
    }

    /// `TensorBoard` directory.
    #[must_use]
    pub fn tb_dir(&self) -> &Path {
        &self.tb_dir
    }

    /// Path to the event file.
    #[must_use]
    pub fn event_path(&self) -> &Path {
        &self.event_path
    }

    fn write_summary(&self, name: MetricName, step: u64, value: f32) -> Result<(), TelemetryError> {
        let event = event_scalar(now_wall_time(), step, name.as_str(), value);
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        write_tfrecord(&mut state.sink, &event)?;
        self.maybe_flush_locked(&mut state)
    }

    fn maybe_flush_locked(&self, state: &mut TensorboardState) -> Result<(), TelemetryError> {
        state.records_since_flush += 1;
        if state.records_since_flush >= self.flush_records
            || state.last_flush.elapsed() >= self.flush_interval
        {
            flush_locked(state)?;
        }
        Ok(())
    }
}

impl MetricSink for TensorboardWriter {
    fn emit_scalar(
        &self,
        _context: &TelemetryContext,
        name: MetricName,
        step: u64,
        value: f32,
    ) -> Result<(), TelemetryError> {
        self.write_summary(name, step, value)
    }

    fn emit_histogram(
        &self,
        _context: &TelemetryContext,
        name: MetricName,
        step: u64,
        values: &[f32],
    ) -> Result<(), TelemetryError> {
        if values.is_empty() {
            return Ok(());
        }
        let count = values.iter().fold(0.0_f32, |count, _| count + 1.0);
        let mean = values.iter().copied().sum::<f32>() / count;
        self.write_summary(name, step, mean)
    }

    fn flush(&self) -> Result<(), TelemetryError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        flush_locked(&mut state)
    }
}

fn validate_flush_policy(
    flush_records: usize,
    flush_interval: Duration,
) -> Result<(), TelemetryError> {
    if flush_records == 0 {
        return Err(TelemetryError::InvalidConfig(
            "flush_records must be greater than zero".to_string(),
        ));
    }
    if flush_interval.is_zero() {
        return Err(TelemetryError::InvalidConfig(
            "flush_interval must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

fn flush_locked(state: &mut TensorboardState) -> Result<(), TelemetryError> {
    state.sink.flush().map_err(TelemetryError::sink)?;
    state.records_since_flush = 0;
    state.last_flush = Instant::now(); // determinism-lint: allow Instant::now telemetry flush cadence
    Ok(())
}

fn write_tfrecord(writer: &mut impl Write, payload: &[u8]) -> Result<(), TelemetryError> {
    let len = u64::try_from(payload.len()).map_err(TelemetryError::sink)?;
    let len_bytes = len.to_le_bytes();
    writer.write_all(&len_bytes).map_err(TelemetryError::sink)?;
    writer
        .write_all(&masked_crc32c(&len_bytes).to_le_bytes())
        .map_err(TelemetryError::sink)?;
    writer.write_all(payload).map_err(TelemetryError::sink)?;
    writer
        .write_all(&masked_crc32c(payload).to_le_bytes())
        .map_err(TelemetryError::sink)?;
    Ok(())
}

fn event_file_version(wall_time: f64) -> Vec<u8> {
    let mut event = Vec::new();
    write_key(&mut event, 1, 1);
    event.extend_from_slice(&wall_time.to_le_bytes());
    write_string_field(&mut event, 3, FILE_VERSION);
    event
}

fn event_scalar(wall_time: f64, step: u64, tag: &str, value: f32) -> Vec<u8> {
    let mut value_msg = Vec::new();
    write_string_field(&mut value_msg, 1, tag);
    write_key(&mut value_msg, 2, 5);
    value_msg.extend_from_slice(&value.to_le_bytes());

    let mut summary = Vec::new();
    write_len_field(&mut summary, 1, &value_msg);

    let mut event = Vec::new();
    write_key(&mut event, 1, 1);
    event.extend_from_slice(&wall_time.to_le_bytes());
    write_key(&mut event, 2, 0);
    write_varint(&mut event, step);
    write_len_field(&mut event, 5, &summary);
    event
}

fn write_string_field(buffer: &mut Vec<u8>, field: u64, value: &str) {
    write_key(buffer, field, 2);
    write_varint(buffer, value.len() as u64);
    buffer.extend_from_slice(value.as_bytes());
}

fn write_len_field(buffer: &mut Vec<u8>, field: u64, value: &[u8]) {
    write_key(buffer, field, 2);
    write_varint(buffer, value.len() as u64);
    buffer.extend_from_slice(value);
}

fn write_key(buffer: &mut Vec<u8>, field: u64, wire_type: u64) {
    write_varint(buffer, (field << 3) | wire_type);
}

fn write_varint(buffer: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        let byte = value.to_le_bytes()[0] & 0x7f;
        buffer.push(byte | 0x80);
        value >>= 7;
    }
    buffer.push(value.to_le_bytes()[0]);
}

fn masked_crc32c(bytes: &[u8]) -> u32 {
    let crc = crc32c(bytes);
    crc.rotate_right(15).wrapping_add(CRC_MASK_DELTA)
}

fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ CRC32C_POLY;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

fn now_wall_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |duration| duration.as_secs_f64())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::Path,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use super::*;

    #[test]
    fn tensorboard_writer_event_format() -> Result<(), Box<dyn std::error::Error>> {
        let root = temp_root("tensorboard_writer_event_format")?;
        let context = TelemetryContext {
            run_id: "run-001".to_string(),
            phase: "phase-2".to_string(),
            git_short_sha: "abc1234".to_string(),
        };
        let writer = TensorboardWriter::with_flush_policy(
            &root,
            &context.run_id,
            1,
            Duration::from_secs(60),
        )?;

        writer.emit_scalar(&context, MetricName::LossTotal, 11, 2.5)?;
        writer.flush()?;

        assert!(writer.event_path().starts_with(writer.tb_dir()));
        let records = read_tfrecords(writer.event_path())?;
        assert_eq!(records.len(), 2);
        assert!(
            records[0]
                .windows(FILE_VERSION.len())
                .any(|chunk| chunk == FILE_VERSION.as_bytes())
        );
        assert!(
            records[1]
                .windows("loss/total".len())
                .any(|chunk| chunk == b"loss/total")
        );
        assert!(records[1].contains(&42));
        assert!(records[1].contains(&21));

        fs::remove_dir_all(root)?;
        Ok(())
    }

    fn read_tfrecords(path: &Path) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        let bytes = fs::read(path)?;
        let mut records = Vec::new();
        let mut offset = 0usize;
        while offset < bytes.len() {
            let len_bytes: [u8; 8] = bytes[offset..offset + 8].try_into()?;
            offset += 8;
            let expected_len_crc = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
            offset += 4;
            assert_eq!(expected_len_crc, masked_crc32c(&len_bytes));
            let len = usize::try_from(u64::from_le_bytes(len_bytes))?;
            let payload = bytes[offset..offset + len].to_vec();
            offset += len;
            let expected_payload_crc = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
            offset += 4;
            assert_eq!(expected_payload_crc, masked_crc32c(&payload));
            records.push(payload);
        }
        Ok(records)
    }

    fn temp_root(name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let root = std::env::temp_dir().join(format!(
            "lewm-telemetry-{name}-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        let _ignored = fs::remove_dir_all(&root);
        fs::create_dir_all(&root)?;
        Ok(root)
    }
}
