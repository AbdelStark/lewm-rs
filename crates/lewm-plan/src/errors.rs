//! Error types for planning and evaluation APIs.

/// Errors surfaced by `lewm-plan` algorithms.
#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum LewmPlanError {
    /// A CEM hyperparameter or runtime input is invalid.
    #[error("invalid CEM configuration: {reason}")]
    InvalidCemConfig {
        /// Concrete validation failure.
        reason: String,
    },

    /// The caller passed tensors or buffers with incoherent shapes.
    #[error("invalid CEM input: {reason}")]
    InvalidCemInput {
        /// Concrete validation failure.
        reason: String,
    },

    /// The configured cost model returned an invalid response.
    #[error("invalid CEM cost output: {reason}")]
    InvalidCemCost {
        /// Concrete validation failure.
        reason: String,
    },

    /// The cost model failed while scoring candidate actions.
    #[error("CEM cost evaluation failed: {reason}")]
    CostEvaluation {
        /// Concrete cost-model failure.
        reason: String,
    },

    /// The required RFC 0013 RNG sub-stream was unavailable.
    #[error("CEM RNG setup failed: {reason}")]
    Rng {
        /// Concrete RNG setup failure.
        reason: String,
    },
}

impl LewmPlanError {
    pub(crate) fn invalid_config(reason: impl Into<String>) -> Self {
        Self::InvalidCemConfig {
            reason: reason.into(),
        }
    }

    pub(crate) fn invalid_input(reason: impl Into<String>) -> Self {
        Self::InvalidCemInput {
            reason: reason.into(),
        }
    }

    pub(crate) fn invalid_cost(reason: impl Into<String>) -> Self {
        Self::InvalidCemCost {
            reason: reason.into(),
        }
    }

    pub(crate) fn rng(reason: impl Into<String>) -> Self {
        Self::Rng {
            reason: reason.into(),
        }
    }
}
