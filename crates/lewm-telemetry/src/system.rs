//! Periodic RFC 0009 system metric samplers.

use std::{
    fmt,
    time::{Duration, Instant},
};

use sysinfo::{
    CpuRefreshKind, DiskRefreshKind, Disks, ProcessRefreshKind, ProcessesToUpdate, RefreshKind,
    System, get_current_pid,
};

use crate::{MetricName, Telemetry, TelemetryError};

const BYTES_PER_GIB: f32 = 1024.0 * 1024.0 * 1024.0;

/// Cadence for CPU, RSS, and GPU system metrics.
pub const SYSTEM_FAST_CADENCE: Duration = Duration::from_secs(30);

/// Cadence for disk usage system metrics.
pub const SYSTEM_DISK_CADENCE: Duration = Duration::from_secs(5 * 60);

/// One NVML GPU metric sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuMetrics {
    /// GPU framebuffer memory currently used, in GiB.
    pub mem_used_gb: f32,
    /// GPU utilization percentage reported by NVML.
    pub util_pct: f32,
}

/// One fast RFC 0009 system metric sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SystemMetrics {
    /// GPU metrics, when the `nvml` feature is enabled and NVML is available.
    pub gpu: Option<GpuMetrics>,
    /// Host CPU utilization percentage.
    pub cpu_util_pct: f32,
    /// Current process resident set size, in GiB.
    pub host_rss_gb: f32,
}

impl SystemMetrics {
    /// Emit the fast system metrics through the telemetry facade.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured metric sink rejects a metric.
    pub fn emit_fast(&self, telemetry: &Telemetry, step: u64) -> Result<(), TelemetryError> {
        if let Some(gpu) = self.gpu {
            telemetry.emit_scalar(MetricName::SystemGpuMemUsedGb, step, gpu.mem_used_gb)?;
            telemetry.emit_scalar(MetricName::SystemGpuUtilPct, step, gpu.util_pct)?;
        }

        telemetry.emit_scalar(MetricName::SystemCpuUtilPct, step, self.cpu_util_pct)?;
        telemetry.emit_scalar(MetricName::SystemHostRssGb, step, self.host_rss_gb)?;
        Ok(())
    }
}

/// System metric cadence decision for one wall-clock instant.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct SystemSampleDue {
    /// CPU, RSS, and GPU metrics should be sampled.
    pub fast_metrics: bool,
    /// Disk usage should be sampled.
    pub disk_metric: bool,
}

impl SystemSampleDue {
    /// Return whether at least one metric family is due.
    #[must_use]
    pub const fn is_any_due(self) -> bool {
        self.fast_metrics || self.disk_metric
    }
}

/// Emission summary for one sampler tick.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct SystemEmitReport {
    /// CPU, RSS, and available GPU metrics were emitted.
    pub fast_metrics: bool,
    /// Disk usage was emitted.
    pub disk_metric: bool,
    /// NVML GPU metrics were emitted.
    pub gpu_metrics: bool,
}

impl SystemEmitReport {
    /// Return whether at least one metric family was emitted.
    #[must_use]
    pub const fn emitted_any(self) -> bool {
        self.fast_metrics || self.disk_metric
    }
}

/// RFC 0009 wall-clock cadence tracker for system metrics.
#[derive(Debug, Clone, Default)]
pub struct SystemMetricCadence {
    last_fast_emit: Option<Instant>,
    last_disk_emit: Option<Instant>,
}

impl SystemMetricCadence {
    /// Build a cadence tracker with no prior emissions.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            last_fast_emit: None,
            last_disk_emit: None,
        }
    }

    /// Return which metric families are due at `now`.
    #[must_use]
    pub fn due_at(&self, now: Instant) -> SystemSampleDue {
        SystemSampleDue {
            fast_metrics: is_due(self.last_fast_emit, now, SYSTEM_FAST_CADENCE),
            disk_metric: is_due(self.last_disk_emit, now, SYSTEM_DISK_CADENCE),
        }
    }

    /// Record that the selected metric families were emitted at `now`.
    pub fn record_emit(&mut self, due: SystemSampleDue, now: Instant) {
        if due.fast_metrics {
            self.last_fast_emit = Some(now);
        }
        if due.disk_metric {
            self.last_disk_emit = Some(now);
        }
    }
}

/// RFC 0009 system metric sampler.
#[derive(Debug)]
pub struct SystemSampler {
    system: System,
    disks: Disks,
    pid: sysinfo::Pid,
    gpu_sampler: GpuSampler,
    cadence: SystemMetricCadence,
}

impl SystemSampler {
    /// Create a sampler for the current process and host.
    ///
    /// # Errors
    ///
    /// Returns an error when the current process id cannot be resolved.
    pub fn new() -> Result<Self, TelemetryError> {
        let pid = get_current_pid().map_err(|err| TelemetryError::Sampler(err.to_string()))?;
        let mut system = System::new_with_specifics(
            RefreshKind::nothing()
                .with_cpu(CpuRefreshKind::nothing().with_cpu_usage())
                .with_processes(ProcessRefreshKind::nothing().with_memory().without_tasks()),
        );
        refresh_current_process(&mut system, pid);

        Ok(Self {
            system,
            disks: Disks::new_with_refreshed_list_specifics(
                DiskRefreshKind::nothing().with_storage(),
            ),
            pid,
            gpu_sampler: GpuSampler::new(),
            cadence: SystemMetricCadence::new(),
        })
    }

    /// Sample CPU, current process RSS, and available GPU metrics immediately.
    #[must_use]
    pub fn sample_fast(&mut self) -> SystemMetrics {
        self.system.refresh_cpu_usage();
        refresh_current_process(&mut self.system, self.pid);

        let host_rss_gb = self
            .system
            .process(self.pid)
            .map_or(0.0, |process| bytes_to_gib(process.memory()));

        SystemMetrics {
            gpu: self.gpu_sampler.sample(),
            cpu_util_pct: self.system.global_cpu_usage(),
            host_rss_gb,
        }
    }

    /// Sample total used disk space across mounted disks immediately.
    #[must_use]
    pub fn sample_disk_used_gb(&mut self) -> f32 {
        self.disks.refresh(true);
        bytes_to_gib(total_disk_used_bytes(&self.disks))
    }

    /// Emit every system metric family that is due at the current instant.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured metric sink rejects a metric.
    pub fn emit_due(
        &mut self,
        telemetry: &Telemetry,
        step: u64,
    ) -> Result<SystemEmitReport, TelemetryError> {
        self.emit_due_at(
            telemetry,
            step,
            Instant::now(), // determinism-lint: allow Instant::now system telemetry cadence
        )
    }

    /// Emit every system metric family that is due at `now`.
    ///
    /// This method is useful for deterministic tests and external schedulers that
    /// already hold a monotonic timestamp.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured metric sink rejects a metric.
    pub fn emit_due_at(
        &mut self,
        telemetry: &Telemetry,
        step: u64,
        now: Instant,
    ) -> Result<SystemEmitReport, TelemetryError> {
        let due = self.cadence.due_at(now);
        let report = self.emit_due_metrics(telemetry, step, due)?;
        self.cadence.record_emit(due, now);
        Ok(report)
    }

    fn emit_due_metrics(
        &mut self,
        telemetry: &Telemetry,
        step: u64,
        due: SystemSampleDue,
    ) -> Result<SystemEmitReport, TelemetryError> {
        let mut report = SystemEmitReport::default();

        if due.fast_metrics {
            let metrics = self.sample_fast();
            report.gpu_metrics = metrics.gpu.is_some();
            metrics.emit_fast(telemetry, step)?;
            report.fast_metrics = true;
        }

        if due.disk_metric {
            telemetry.emit_scalar(
                MetricName::SystemDiskUsedGb,
                step,
                self.sample_disk_used_gb(),
            )?;
            report.disk_metric = true;
        }

        Ok(report)
    }
}

fn refresh_current_process(system: &mut System, pid: sysinfo::Pid) {
    let pids = [pid];
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&pids),
        false,
        ProcessRefreshKind::nothing().with_memory().without_tasks(),
    );
}

fn is_due(last_emit: Option<Instant>, now: Instant, cadence: Duration) -> bool {
    last_emit.is_none_or(|last| {
        now.checked_duration_since(last)
            .is_none_or(|elapsed| elapsed >= cadence)
    })
}

fn total_disk_used_bytes(disks: &Disks) -> u64 {
    disks.list().iter().fold(0_u64, |acc, disk| {
        acc.saturating_add(disk_used_bytes(disk.total_space(), disk.available_space()))
    })
}

const fn disk_used_bytes(total: u64, available: u64) -> u64 {
    total.saturating_sub(available)
}

#[allow(clippy::cast_precision_loss)]
fn bytes_to_gib(bytes: u64) -> f32 {
    bytes as f32 / BYTES_PER_GIB
}

#[derive(Default)]
enum GpuSampler {
    #[cfg(feature = "nvml")]
    Nvml { nvml: Box<nvml_wrapper::Nvml> },
    #[default]
    Unavailable,
}

impl GpuSampler {
    fn new() -> Self {
        #[cfg(feature = "nvml")]
        {
            nvml_wrapper::Nvml::init()
                .map(|nvml| Self::Nvml {
                    nvml: Box::new(nvml),
                })
                .unwrap_or(Self::Unavailable)
        }

        #[cfg(not(feature = "nvml"))]
        {
            Self::Unavailable
        }
    }

    fn sample(&self) -> Option<GpuMetrics> {
        match self {
            #[cfg(feature = "nvml")]
            Self::Nvml { nvml } => sample_nvml_gpu(nvml),
            Self::Unavailable => None,
        }
    }
}

impl fmt::Debug for GpuSampler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "nvml")]
            Self::Nvml { .. } => formatter
                .debug_struct("GpuSampler::Nvml")
                .finish_non_exhaustive(),
            Self::Unavailable => formatter.write_str("GpuSampler::Unavailable"),
        }
    }
}

#[cfg(feature = "nvml")]
fn sample_nvml_gpu(nvml: &nvml_wrapper::Nvml) -> Option<GpuMetrics> {
    let device = nvml.device_by_index(0).ok()?;
    let memory = device.memory_info().ok()?;
    let utilization = device.utilization_rates().ok()?;
    Some(GpuMetrics {
        mem_used_gb: bytes_to_gib(memory.used),
        #[allow(clippy::cast_precision_loss)]
        util_pct: utilization.gpu as f32,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::{MetricSink, TelemetryConfig, TelemetryContext};

    #[derive(Debug, Default)]
    struct RecordingSink {
        records: Mutex<Vec<(MetricName, u64, f32)>>,
    }

    impl RecordingSink {
        fn records(&self) -> Vec<(MetricName, u64, f32)> {
            self.records
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }
    }

    impl MetricSink for RecordingSink {
        fn emit_scalar(
            &self,
            _context: &TelemetryContext,
            name: MetricName,
            step: u64,
            value: f32,
        ) -> Result<(), TelemetryError> {
            self.records
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push((name, step, value));
            Ok(())
        }

        fn emit_histogram(
            &self,
            _context: &TelemetryContext,
            _name: MetricName,
            _step: u64,
            _values: &[f32],
        ) -> Result<(), TelemetryError> {
            Ok(())
        }

        fn flush(&self) -> Result<(), TelemetryError> {
            Ok(())
        }
    }

    #[test]
    fn bytes_and_disk_usage_are_converted_safely() {
        assert!((bytes_to_gib(1_073_741_824) - 1.0).abs() < f32::EPSILON);
        assert_eq!(disk_used_bytes(500, 125), 375);
        assert_eq!(disk_used_bytes(125, 500), 0);
    }

    #[test]
    fn cadence_matches_rfc_0009_intervals() {
        let mut cadence = SystemMetricCadence::new();
        let start = Instant::now(); // determinism-lint: allow Instant::now test clock seed

        let first = cadence.due_at(start);
        assert_eq!(
            first,
            SystemSampleDue {
                fast_metrics: true,
                disk_metric: true
            }
        );
        cadence.record_emit(first, start);

        assert_eq!(
            cadence.due_at(start + Duration::from_secs(29)),
            SystemSampleDue::default()
        );
        assert_eq!(
            cadence.due_at(start + Duration::from_secs(30)),
            SystemSampleDue {
                fast_metrics: true,
                disk_metric: false
            }
        );

        let fast_due = cadence.due_at(start + Duration::from_secs(30));
        cadence.record_emit(fast_due, start + Duration::from_secs(30));
        assert_eq!(
            cadence.due_at(start + SYSTEM_DISK_CADENCE),
            SystemSampleDue {
                fast_metrics: true,
                disk_metric: true
            }
        );
    }

    #[test]
    fn fast_metrics_emit_registered_names() -> Result<(), Box<dyn std::error::Error>> {
        let sink = Arc::new(RecordingSink::default());
        let telemetry = Telemetry::with_metric_sink(
            TelemetryConfig::new("run-001", "train", "abc1234"),
            sink.clone(),
        )?;
        let metrics = SystemMetrics {
            gpu: Some(GpuMetrics {
                mem_used_gb: 8.0,
                util_pct: 42.0,
            }),
            cpu_util_pct: 17.5,
            host_rss_gb: 1.25,
        };

        metrics.emit_fast(&telemetry, 11)?;
        telemetry.emit_scalar(MetricName::SystemDiskUsedGb, 11, 256.0)?;

        let records = sink.records();
        let names: Vec<_> = records.iter().map(|(name, _, _)| name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "system/gpu_mem_used_gb",
                "system/gpu_util_pct",
                "system/cpu_util_pct",
                "system/host_rss_gb",
                "system/disk_used_gb",
            ]
        );
        assert!(records.iter().all(|(_, step, _)| *step == 11));
        Ok(())
    }

    #[test]
    #[cfg(not(feature = "nvml"))]
    fn sampler_emit_due_emits_system_metrics_without_gpu() -> Result<(), Box<dyn std::error::Error>>
    {
        let sink = Arc::new(RecordingSink::default());
        let telemetry = Telemetry::with_metric_sink(
            TelemetryConfig::new("run-001", "train", "abc1234"),
            sink.clone(),
        )?;
        let mut sampler = SystemSampler::new()?;
        let now = Instant::now(); // determinism-lint: allow Instant::now test clock seed

        let report = sampler.emit_due_at(&telemetry, 13, now)?;
        assert_eq!(
            report,
            SystemEmitReport {
                fast_metrics: true,
                disk_metric: true,
                gpu_metrics: false,
            }
        );

        let names: Vec<_> = sink
            .records()
            .iter()
            .map(|(name, _, _)| name.as_str())
            .collect();
        assert_eq!(
            names,
            vec![
                "system/cpu_util_pct",
                "system/host_rss_gb",
                "system/disk_used_gb",
            ]
        );

        let skipped = sampler.emit_due_at(&telemetry, 14, now + Duration::from_secs(1))?;
        assert_eq!(skipped, SystemEmitReport::default());
        assert_eq!(sink.records().len(), 3);
        Ok(())
    }

    #[test]
    #[cfg(not(feature = "nvml"))]
    fn gpu_metrics_are_absent_without_nvml_feature() {
        assert_eq!(GpuSampler::new().sample(), None);
    }
}
