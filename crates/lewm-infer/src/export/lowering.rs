//! Activation lowering contract for ONNX export.

use serde::{Deserialize, Serialize};

use crate::errors::{InferError, InferResult};

/// Explicit activation lowering used to keep Tract execution on primitive ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationLowering {
    /// Tanh-approximate GELU lowered to `Mul`, `Add`, `Tanh`, and `Pow`.
    GeluTanhApprox,
    /// `SiLU` lowered to `Sigmoid` followed by `Mul`.
    Silu,
}

impl ActivationLowering {
    /// Return the canonical activation name from the RFC.
    pub const fn activation_name(self) -> &'static str {
        match self {
            Self::GeluTanhApprox => "gelu_tanh_approx",
            Self::Silu => "silu",
        }
    }

    /// Return the primitive ONNX ops used by this lowering.
    pub const fn primitive_ops(self) -> &'static [&'static str] {
        match self {
            Self::GeluTanhApprox => &["Mul", "Add", "Pow", "Tanh"],
            Self::Silu => &["Sigmoid", "Mul"],
        }
    }
}

/// Set of activation lowerings required by RFC 0007.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivationLoweringSet {
    lowerings: Vec<ActivationLowering>,
}

impl ActivationLoweringSet {
    /// Return the RFC 0007 activation lowering set.
    pub fn rfc0007() -> Self {
        Self {
            lowerings: vec![ActivationLowering::GeluTanhApprox, ActivationLowering::Silu],
        }
    }

    /// Return all configured lowerings.
    pub fn as_slice(&self) -> &[ActivationLowering] {
        &self.lowerings
    }

    /// Check that all required RFC 0007 lowerings are present.
    ///
    /// # Errors
    ///
    /// Returns [`InferError::InvalidExportContract`] when an activation is not
    /// covered by the export lowering set.
    pub fn validate_rfc0007(&self) -> InferResult<()> {
        for required in [ActivationLowering::GeluTanhApprox, ActivationLowering::Silu] {
            if !self.lowerings.contains(&required) {
                return Err(InferError::invalid_export_contract(format!(
                    "missing explicit lowering for {}",
                    required.activation_name()
                )));
            }
        }

        Ok(())
    }
}

impl Default for ActivationLoweringSet {
    fn default() -> Self {
        Self::rfc0007()
    }
}
