//! Public error type for `lewm-core`.

/// Error type surfaced by `lewm-core` APIs.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum LewmCoreError {
    /// A tensor or flat buffer shape did not match the configured model contract.
    #[error(
        "invalid tensor shape: expected {expected:?}, found {found:?}; pass data with the configured model dimensions"
    )]
    InvalidShape {
        /// Expected shape.
        expected: Vec<usize>,
        /// Found shape.
        found: Vec<usize>,
    },

    /// A model component could not be constructed from its config.
    #[error("module construction failed: {reason}; verify the model config shape invariants")]
    ConstructionFailed {
        /// Concrete construction failure.
        reason: String,
    },

    /// A named parameter was missing from an imported record.
    #[error(
        "parameter '{name}' not found in record; verify the checkpoint was converted with the lewm-rs parameter map"
    )]
    ParamNotFound {
        /// Parameter name.
        name: String,
    },

    /// An initialization request was not well-formed.
    #[error(
        "invalid initialization request: {reason}; use a non-empty shape and finite init parameters"
    )]
    InvalidInit {
        /// Concrete initialization failure.
        reason: String,
    },

    /// A named RNG stream was not recognized by the public RFC 0013 tree.
    #[error("rng sub-stream error: {name}; use an RFC 0013 sub-stream name")]
    RngSubstream {
        /// Invalid sub-stream name.
        name: String,
    },

    /// A serialized RNG state could not be restored.
    #[error("rng state error: {reason}; restore from the 48-byte RFC 0013 state format")]
    RngState {
        /// Concrete state parsing failure.
        reason: String,
    },

    /// A predictor input sequence exceeded the learned positional embedding.
    #[error(
        "predictor sequence too long: got {got}, max {max}; slice to the configured num_frames"
    )]
    SequenceTooLong {
        /// Sequence length received by the predictor.
        got: usize,
        /// Maximum sequence length supported by the predictor.
        max: usize,
    },

    /// A tensor helper received invalid dimensions or options.
    #[error("invalid tensor operation: {reason}; pass shape-compatible finite F32 tensors")]
    InvalidTensorOp {
        /// Concrete tensor operation failure.
        reason: String,
    },

    /// Catch-all for unexpected internal failures.
    #[error("internal error: {0}; file a bug with the minimal reproducer")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_type_is_send_sync_static() {
        fn assert_traits<T: std::error::Error + Send + Sync + 'static>() {}

        assert_traits::<LewmCoreError>();
    }

    #[test]
    fn error_messages_have_context_and_fix() {
        let err = LewmCoreError::InvalidShape {
            expected: vec![2, 3],
            found: vec![3, 2],
        };
        let msg = err.to_string();

        assert!(msg.starts_with("invalid tensor shape:"));
        assert!(msg.contains("; pass data"));
    }
}
