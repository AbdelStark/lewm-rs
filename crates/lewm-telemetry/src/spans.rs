//! Span-name registry pinned by RFC 0009.

/// Closed span-name registry.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct SpanName(&'static str);

impl SpanName {
    /// Span `training.run`.
    pub const TRAINING_RUN: Self = Self("training.run");
    /// Span `training.epoch`.
    pub const TRAINING_EPOCH: Self = Self("training.epoch");
    /// Span `training.step`.
    pub const TRAINING_STEP: Self = Self("training.step");
    /// Span `training.forward`.
    pub const TRAINING_FORWARD: Self = Self("training.forward");
    /// Span `training.backward`.
    pub const TRAINING_BACKWARD: Self = Self("training.backward");
    /// Span `training.optim_step`.
    pub const TRAINING_OPTIM_STEP: Self = Self("training.optim_step");
    /// Span `training.checkpoint_save`.
    pub const TRAINING_CHECKPOINT_SAVE: Self = Self("training.checkpoint_save");
    /// Span `training.parity_probe`.
    pub const TRAINING_PARITY_PROBE: Self = Self("training.parity_probe");
    /// Span `training.collapse_probe`.
    pub const TRAINING_COLLAPSE_PROBE: Self = Self("training.collapse_probe");
    /// Span `training.eval`.
    pub const TRAINING_EVAL: Self = Self("training.eval");
    /// Span `eval.episode`.
    pub const EVAL_EPISODE: Self = Self("eval.episode");
    /// Span `eval.cem_iter`.
    pub const EVAL_CEM_ITER: Self = Self("eval.cem_iter");
    /// Span `eval.cem_cost_eval`.
    pub const EVAL_CEM_COST_EVAL: Self = Self("eval.cem_cost_eval");
    /// Span `eval.rpc_step`.
    pub const EVAL_RPC_STEP: Self = Self("eval.rpc_step");
    /// Span `data.dataset_open`.
    pub const DATA_DATASET_OPEN: Self = Self("data.dataset_open");
    /// Span `data.get_window`.
    pub const DATA_GET_WINDOW: Self = Self("data.get_window");
    /// Span `data.collate`.
    pub const DATA_COLLATE: Self = Self("data.collate");
    /// Span `data.prefetch_worker.lifetime`.
    pub const DATA_PREFETCH_WORKER_LIFETIME: Self = Self("data.prefetch_worker.lifetime");

    /// Every registered span name, in RFC 0009 order.
    pub const ALL: &'static [Self] = &[
        Self::TRAINING_RUN,
        Self::TRAINING_EPOCH,
        Self::TRAINING_STEP,
        Self::TRAINING_FORWARD,
        Self::TRAINING_BACKWARD,
        Self::TRAINING_OPTIM_STEP,
        Self::TRAINING_CHECKPOINT_SAVE,
        Self::TRAINING_PARITY_PROBE,
        Self::TRAINING_COLLAPSE_PROBE,
        Self::TRAINING_EVAL,
        Self::EVAL_EPISODE,
        Self::EVAL_CEM_ITER,
        Self::EVAL_CEM_COST_EVAL,
        Self::EVAL_RPC_STEP,
        Self::DATA_DATASET_OPEN,
        Self::DATA_GET_WINDOW,
        Self::DATA_COLLATE,
        Self::DATA_PREFETCH_WORKER_LIFETIME,
    ];

    /// Stable RFC 0009 span name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn span_registry_no_dup_no_typo() {
        let names = SpanName::ALL
            .iter()
            .map(|span| span.as_str())
            .collect::<Vec<_>>();
        let unique = names.iter().copied().collect::<HashSet<_>>();

        assert_eq!(names.len(), unique.len(), "span registry has duplicates");
        assert_eq!(
            names,
            [
                "training.run",
                "training.epoch",
                "training.step",
                "training.forward",
                "training.backward",
                "training.optim_step",
                "training.checkpoint_save",
                "training.parity_probe",
                "training.collapse_probe",
                "training.eval",
                "eval.episode",
                "eval.cem_iter",
                "eval.cem_cost_eval",
                "eval.rpc_step",
                "data.dataset_open",
                "data.get_window",
                "data.collate",
                "data.prefetch_worker.lifetime",
            ]
        );
    }
}
