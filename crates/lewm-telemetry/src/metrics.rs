//! Metric registry pinned by RFC 0009.

use crate::TelemetryError;

/// Metric data shape.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum MetricKind {
    /// Single scalar value at a step.
    Scalar,
    /// Distribution-like value at a step.
    Histogram,
}

/// Closed metric-name registry.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum MetricName {
    /// Metric `loss/total`.
    LossTotal,
    /// Metric `loss/pred`.
    LossPred,
    /// Metric `loss/sigreg`.
    LossSigreg,
    /// Metric `loss/sigreg_per_proj_min`.
    LossSigregPerProjMin,
    /// Metric `loss/sigreg_per_proj_max`.
    LossSigregPerProjMax,
    /// Metric `optim/lr`.
    OptimLr,
    /// Metric `optim/grad_norm_pre`.
    OptimGradNormPre,
    /// Metric `optim/grad_norm_post`.
    OptimGradNormPost,
    /// Metric `optim/effective_step_norm`.
    OptimEffectiveStepNorm,
    /// Metric `optim/momentum_norm`.
    OptimMomentumNorm,
    /// Metric `optim/exp_avg_sq_norm`.
    OptimExpAvgSqNorm,
    /// Metric `optim/skipped_steps_total`.
    OptimSkippedStepsTotal,
    /// Metric `model/encoder_cls_var`.
    ModelEncoderClsVar,
    /// Metric `model/encoder_cls_mean_abs`.
    ModelEncoderClsMeanAbs,
    /// Metric `model/cls_cosine_pair_mean`.
    ModelClsCosinePairMean,
    /// Metric `model/predictor_output_var`.
    ModelPredictorOutputVar,
    /// Metric `throughput/samples_per_sec`.
    ThroughputSamplesPerSec,
    /// Metric `throughput/tokens_per_sec`.
    ThroughputTokensPerSec,
    /// Metric `throughput/batches_per_sec`.
    ThroughputBatchesPerSec,
    /// Metric `data/throughput_samples_per_sec`.
    DataThroughputSamplesPerSec,
    /// Metric `data/throughput_bytes_per_sec`.
    DataThroughputBytesPerSec,
    /// Metric `data/queue_depth`.
    DataQueueDepth,
    /// Metric `data/io_wait_ms_p50`.
    DataIoWaitMsP50,
    /// Metric `data/io_wait_ms_p99`.
    DataIoWaitMsP99,
    /// Metric `data/error_count{kind}`.
    DataErrorCountKind,
    /// Metric `system/gpu_mem_used_gb`.
    SystemGpuMemUsedGb,
    /// Metric `system/gpu_util_pct`.
    SystemGpuUtilPct,
    /// Metric `system/cpu_util_pct`.
    SystemCpuUtilPct,
    /// Metric `system/host_rss_gb`.
    SystemHostRssGb,
    /// Metric `system/disk_used_gb`.
    SystemDiskUsedGb,
    /// Metric `state/<NAME>/wall_seconds`.
    StateWallSeconds,
    /// Metric `state/transitions_total`.
    StateTransitionsTotal,
    /// Metric `eval/episode_success`.
    EvalEpisodeSuccess,
    /// Metric `eval/episode_steps`.
    EvalEpisodeSteps,
    /// Metric `eval/episode_final_cost`.
    EvalEpisodeFinalCost,
    /// Metric `eval/cem_iter_cost_min`.
    EvalCemIterCostMin,
    /// Metric `eval/success_rate`.
    EvalSuccessRate,
    /// Metric `eval/latent_mse_mean`.
    EvalLatentMseMean,
    /// Metric `eval/spearman_mean`.
    EvalSpearmanMean,
    /// Metric `eval/warm_start_delta`.
    EvalWarmStartDelta,
    /// Metric `checkpoint/written_count`.
    CheckpointWrittenCount,
    /// Metric `checkpoint/disk_usage_gb`.
    CheckpointDiskUsageGb,
    /// Metric `checkpoint/save_wall_ms`.
    CheckpointSaveWallMs,
    /// Metric `sigreg/cos_max`.
    SigregCosMax,
    /// Metric `sigreg/sin_max`.
    SigregSinMax,
}

impl MetricName {
    /// Every registered metric variant, in RFC 0009 order.
    pub const ALL: &'static [Self] = &[
        Self::LossTotal,
        Self::LossPred,
        Self::LossSigreg,
        Self::LossSigregPerProjMin,
        Self::LossSigregPerProjMax,
        Self::OptimLr,
        Self::OptimGradNormPre,
        Self::OptimGradNormPost,
        Self::OptimEffectiveStepNorm,
        Self::OptimMomentumNorm,
        Self::OptimExpAvgSqNorm,
        Self::OptimSkippedStepsTotal,
        Self::ModelEncoderClsVar,
        Self::ModelEncoderClsMeanAbs,
        Self::ModelClsCosinePairMean,
        Self::ModelPredictorOutputVar,
        Self::ThroughputSamplesPerSec,
        Self::ThroughputTokensPerSec,
        Self::ThroughputBatchesPerSec,
        Self::DataThroughputSamplesPerSec,
        Self::DataThroughputBytesPerSec,
        Self::DataQueueDepth,
        Self::DataIoWaitMsP50,
        Self::DataIoWaitMsP99,
        Self::DataErrorCountKind,
        Self::SystemGpuMemUsedGb,
        Self::SystemGpuUtilPct,
        Self::SystemCpuUtilPct,
        Self::SystemHostRssGb,
        Self::SystemDiskUsedGb,
        Self::StateWallSeconds,
        Self::StateTransitionsTotal,
        Self::EvalEpisodeSuccess,
        Self::EvalEpisodeSteps,
        Self::EvalEpisodeFinalCost,
        Self::EvalCemIterCostMin,
        Self::EvalSuccessRate,
        Self::EvalLatentMseMean,
        Self::EvalSpearmanMean,
        Self::EvalWarmStartDelta,
        Self::CheckpointWrittenCount,
        Self::CheckpointDiskUsageGb,
        Self::CheckpointSaveWallMs,
        Self::SigregCosMax,
        Self::SigregSinMax,
    ];

    /// Stable RFC 0009 metric name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LossTotal => "loss/total",
            Self::LossPred => "loss/pred",
            Self::LossSigreg => "loss/sigreg",
            Self::LossSigregPerProjMin => "loss/sigreg_per_proj_min",
            Self::LossSigregPerProjMax => "loss/sigreg_per_proj_max",
            Self::OptimLr => "optim/lr",
            Self::OptimGradNormPre => "optim/grad_norm_pre",
            Self::OptimGradNormPost => "optim/grad_norm_post",
            Self::OptimEffectiveStepNorm => "optim/effective_step_norm",
            Self::OptimMomentumNorm => "optim/momentum_norm",
            Self::OptimExpAvgSqNorm => "optim/exp_avg_sq_norm",
            Self::OptimSkippedStepsTotal => "optim/skipped_steps_total",
            Self::ModelEncoderClsVar => "model/encoder_cls_var",
            Self::ModelEncoderClsMeanAbs => "model/encoder_cls_mean_abs",
            Self::ModelClsCosinePairMean => "model/cls_cosine_pair_mean",
            Self::ModelPredictorOutputVar => "model/predictor_output_var",
            Self::ThroughputSamplesPerSec => "throughput/samples_per_sec",
            Self::ThroughputTokensPerSec => "throughput/tokens_per_sec",
            Self::ThroughputBatchesPerSec => "throughput/batches_per_sec",
            Self::DataThroughputSamplesPerSec => "data/throughput_samples_per_sec",
            Self::DataThroughputBytesPerSec => "data/throughput_bytes_per_sec",
            Self::DataQueueDepth => "data/queue_depth",
            Self::DataIoWaitMsP50 => "data/io_wait_ms_p50",
            Self::DataIoWaitMsP99 => "data/io_wait_ms_p99",
            Self::DataErrorCountKind => "data/error_count{kind}",
            Self::SystemGpuMemUsedGb => "system/gpu_mem_used_gb",
            Self::SystemGpuUtilPct => "system/gpu_util_pct",
            Self::SystemCpuUtilPct => "system/cpu_util_pct",
            Self::SystemHostRssGb => "system/host_rss_gb",
            Self::SystemDiskUsedGb => "system/disk_used_gb",
            Self::StateWallSeconds => "state/<NAME>/wall_seconds",
            Self::StateTransitionsTotal => "state/transitions_total",
            Self::EvalEpisodeSuccess => "eval/episode_success",
            Self::EvalEpisodeSteps => "eval/episode_steps",
            Self::EvalEpisodeFinalCost => "eval/episode_final_cost",
            Self::EvalCemIterCostMin => "eval/cem_iter_cost_min",
            Self::EvalSuccessRate => "eval/success_rate",
            Self::EvalLatentMseMean => "eval/latent_mse_mean",
            Self::EvalSpearmanMean => "eval/spearman_mean",
            Self::EvalWarmStartDelta => "eval/warm_start_delta",
            Self::CheckpointWrittenCount => "checkpoint/written_count",
            Self::CheckpointDiskUsageGb => "checkpoint/disk_usage_gb",
            Self::CheckpointSaveWallMs => "checkpoint/save_wall_ms",
            Self::SigregCosMax => "sigreg/cos_max",
            Self::SigregSinMax => "sigreg/sin_max",
        }
    }

    /// Metric kind for the registry entry.
    #[must_use]
    #[allow(clippy::unused_self)]
    pub const fn kind(self) -> MetricKind {
        MetricKind::Scalar
    }

    /// Parse a stable metric name from RFC 0009.
    ///
    /// # Errors
    ///
    /// Returns [`TelemetryError::UnknownMetric`] when `name` is outside the closed registry.
    pub fn from_name(name: &str) -> Result<Self, TelemetryError> {
        Self::ALL
            .iter()
            .copied()
            .find(|metric| metric.as_str() == name)
            .ok_or_else(|| TelemetryError::UnknownMetric(name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    const EXPECTED_METRICS: &[&str] = &[
        "loss/total",
        "loss/pred",
        "loss/sigreg",
        "loss/sigreg_per_proj_min",
        "loss/sigreg_per_proj_max",
        "optim/lr",
        "optim/grad_norm_pre",
        "optim/grad_norm_post",
        "optim/effective_step_norm",
        "optim/momentum_norm",
        "optim/exp_avg_sq_norm",
        "optim/skipped_steps_total",
        "model/encoder_cls_var",
        "model/encoder_cls_mean_abs",
        "model/cls_cosine_pair_mean",
        "model/predictor_output_var",
        "throughput/samples_per_sec",
        "throughput/tokens_per_sec",
        "throughput/batches_per_sec",
        "data/throughput_samples_per_sec",
        "data/throughput_bytes_per_sec",
        "data/queue_depth",
        "data/io_wait_ms_p50",
        "data/io_wait_ms_p99",
        "data/error_count{kind}",
        "system/gpu_mem_used_gb",
        "system/gpu_util_pct",
        "system/cpu_util_pct",
        "system/host_rss_gb",
        "system/disk_used_gb",
        "state/<NAME>/wall_seconds",
        "state/transitions_total",
        "eval/episode_success",
        "eval/episode_steps",
        "eval/episode_final_cost",
        "eval/cem_iter_cost_min",
        "eval/success_rate",
        "eval/latent_mse_mean",
        "eval/spearman_mean",
        "eval/warm_start_delta",
        "checkpoint/written_count",
        "checkpoint/disk_usage_gb",
        "checkpoint/save_wall_ms",
        "sigreg/cos_max",
        "sigreg/sin_max",
    ];

    #[test]
    fn metric_registry_no_dup_no_typo() {
        let names = MetricName::ALL
            .iter()
            .map(|metric| metric.as_str())
            .collect::<Vec<_>>();
        let unique = names.iter().copied().collect::<HashSet<_>>();

        assert_eq!(names.len(), unique.len(), "metric registry has duplicates");
        assert_eq!(names, EXPECTED_METRICS);
        for name in EXPECTED_METRICS {
            assert_eq!(
                MetricName::from_name(name).ok().map(MetricName::as_str),
                Some(*name)
            );
        }
        assert!(matches!(
            MetricName::from_name("loss/not_registered"),
            Err(TelemetryError::UnknownMetric(metric)) if metric == "loss/not_registered"
        ));
    }
}
