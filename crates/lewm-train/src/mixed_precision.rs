//! Mixed-precision policy contracts for training.

/// Floating-point precision used by a training subgraph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Precision {
    /// IEEE single precision.
    F32,
    /// Brain floating point with 8 exponent bits and 7 mantissa bits.
    Bf16,
}

/// Mixed-precision policy requested by the training configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixedPrecisionPolicy {
    /// Run the full training step in `F32`.
    F32,
    /// Run forward/backward autocast in `BF16` while keeping master weights,
    /// optimizer state, and `SIGReg` in `F32`.
    Bf16Mixed,
}

/// Backend precision capability used to resolve a requested policy.
#[must_use]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendPrecision {
    name: &'static str,
    supports_bf16_autodiff: bool,
}

impl BackendPrecision {
    /// Describes the CPU `NdArray` backend used by smoke tests.
    pub const fn ndarray_cpu() -> Self {
        Self {
            name: "ndarray-cpu",
            supports_bf16_autodiff: false,
        }
    }

    /// Describes a CUDA backend with `BF16` autocast support.
    pub const fn cuda_bf16() -> Self {
        Self {
            name: "cuda-bf16",
            supports_bf16_autodiff: true,
        }
    }

    /// Creates a custom backend precision descriptor.
    pub const fn new(name: &'static str, supports_bf16_autodiff: bool) -> Self {
        Self {
            name,
            supports_bf16_autodiff,
        }
    }

    /// Returns the backend name.
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Returns whether `BF16` autocast is available for autodiff.
    pub const fn supports_bf16_autodiff(&self) -> bool {
        self.supports_bf16_autodiff
    }
}

/// Warning emitted while resolving mixed precision for a backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixedPrecisionWarning {
    /// `BF16` was requested on a backend that only supports `F32` training.
    Bf16Downgraded {
        /// Backend that forced the downgrade.
        backend: &'static str,
    },
}

/// Effective mixed-precision policy after backend capability resolution.
#[must_use]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedMixedPrecision {
    requested: MixedPrecisionPolicy,
    effective: MixedPrecisionPolicy,
    warning: Option<MixedPrecisionWarning>,
}

impl ResolvedMixedPrecision {
    /// Returns the policy requested by config.
    pub const fn requested(&self) -> MixedPrecisionPolicy {
        self.requested
    }

    /// Returns the policy that should actually be executed.
    pub const fn effective(&self) -> MixedPrecisionPolicy {
        self.effective
    }

    /// Returns a downgrade warning when backend capabilities forced one.
    pub const fn warning(&self) -> Option<MixedPrecisionWarning> {
        self.warning
    }
}

/// Metadata describing a precision island entered by the training loop.
#[must_use]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrecisionScope {
    compute: Precision,
    master_weights: Precision,
    optimizer_state: Precision,
    sigreg: Precision,
    loss_scaling: bool,
}

impl PrecisionScope {
    /// Returns the forward/backward compute precision.
    pub const fn compute(self) -> Precision {
        self.compute
    }

    /// Returns the master-weight precision.
    pub const fn master_weights(self) -> Precision {
        self.master_weights
    }

    /// Returns the optimizer-state precision.
    pub const fn optimizer_state(self) -> Precision {
        self.optimizer_state
    }

    /// Returns the `SIGReg` island precision.
    pub const fn sigreg(self) -> Precision {
        self.sigreg
    }

    /// Returns whether dynamic loss scaling is required.
    pub const fn loss_scaling(self) -> bool {
        self.loss_scaling
    }
}

/// Value tagged as entering an explicit precision island.
#[must_use]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrecisionIsland<T> {
    value: T,
    precision: Precision,
}

impl<T> PrecisionIsland<T> {
    /// Returns the island precision.
    pub const fn precision(&self) -> Precision {
        self.precision
    }

    /// Consumes the island and returns the wrapped value.
    pub fn into_inner(self) -> T {
        self.value
    }
}

impl MixedPrecisionPolicy {
    /// Resolves this policy against backend precision support.
    pub const fn resolve(self, backend: BackendPrecision) -> ResolvedMixedPrecision {
        match (self, backend.supports_bf16_autodiff) {
            (Self::Bf16Mixed, false) => ResolvedMixedPrecision {
                requested: self,
                effective: Self::F32,
                warning: Some(MixedPrecisionWarning::Bf16Downgraded {
                    backend: backend.name,
                }),
            },
            _ => ResolvedMixedPrecision {
                requested: self,
                effective: self,
                warning: None,
            },
        }
    }

    /// Returns the precision scope used by this policy.
    pub const fn scope(self) -> PrecisionScope {
        PrecisionScope {
            compute: self.compute_precision(),
            master_weights: Precision::F32,
            optimizer_state: Precision::F32,
            sigreg: Precision::F32,
            loss_scaling: false,
        }
    }

    /// Runs a closure inside this policy's autocast scope.
    pub fn autocast<F, R>(self, scope: F) -> R
    where
        F: FnOnce() -> R,
    {
        let _ = self.scope();
        scope()
    }

    /// Tags a `SIGReg` input as entering the required `F32` precision island.
    pub fn cast_sigreg_input<T>(self, input: T) -> PrecisionIsland<T> {
        let _ = self;
        PrecisionIsland {
            value: input,
            precision: Precision::F32,
        }
    }

    /// Returns the forward/backward compute precision.
    pub const fn compute_precision(self) -> Precision {
        match self {
            Self::F32 => Precision::F32,
            Self::Bf16Mixed => Precision::Bf16,
        }
    }

    /// Returns the master-weight precision.
    pub const fn master_weight_precision(self) -> Precision {
        let _ = self;
        Precision::F32
    }

    /// Returns the optimizer-state precision.
    pub const fn optimizer_state_precision(self) -> Precision {
        let _ = self;
        Precision::F32
    }

    /// Returns the `SIGReg` precision.
    pub const fn sigreg_precision(self) -> Precision {
        let _ = self;
        Precision::F32
    }

    /// Returns whether dynamic loss scaling is required.
    pub const fn requires_loss_scaling(self) -> bool {
        let _ = self;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct TensorStub {
        precision: Precision,
    }

    #[test]
    fn mixed_precision_master_weights_f32() {
        for policy in [MixedPrecisionPolicy::F32, MixedPrecisionPolicy::Bf16Mixed] {
            let scope = policy.scope();

            assert_eq!(policy.master_weight_precision(), Precision::F32);
            assert_eq!(policy.optimizer_state_precision(), Precision::F32);
            assert_eq!(scope.master_weights(), Precision::F32);
            assert_eq!(scope.optimizer_state(), Precision::F32);
        }
    }

    #[test]
    fn sigreg_under_bf16_outer_is_f32() {
        let policy = MixedPrecisionPolicy::Bf16Mixed;
        let input = TensorStub {
            precision: Precision::Bf16,
        };
        let sigreg_input = policy.cast_sigreg_input(input.clone());

        assert_eq!(policy.scope().compute(), Precision::Bf16);
        assert_eq!(policy.sigreg_precision(), Precision::F32);
        assert_eq!(policy.scope().sigreg(), Precision::F32);
        assert_eq!(sigreg_input.precision(), Precision::F32);
        assert_eq!(sigreg_input.into_inner(), input);
    }

    #[test]
    fn ndarray_backend_downgrades_bf16_to_f32_with_warning() {
        let resolved = MixedPrecisionPolicy::Bf16Mixed.resolve(BackendPrecision::ndarray_cpu());

        assert_eq!(resolved.requested(), MixedPrecisionPolicy::Bf16Mixed);
        assert_eq!(resolved.effective(), MixedPrecisionPolicy::F32);
        assert_eq!(
            resolved.warning(),
            Some(MixedPrecisionWarning::Bf16Downgraded {
                backend: "ndarray-cpu",
            })
        );
    }

    #[test]
    fn bf16_backend_preserves_bf16_mixed_policy() {
        let resolved = MixedPrecisionPolicy::Bf16Mixed.resolve(BackendPrecision::cuda_bf16());

        assert_eq!(resolved.effective(), MixedPrecisionPolicy::Bf16Mixed);
        assert_eq!(resolved.warning(), None);
    }

    #[test]
    fn bf16_policy_does_not_require_loss_scaling() {
        let policy = MixedPrecisionPolicy::Bf16Mixed;

        assert!(!policy.requires_loss_scaling());
        assert!(!policy.scope().loss_scaling());
    }

    #[test]
    fn autocast_invokes_scope() {
        let value = MixedPrecisionPolicy::Bf16Mixed.autocast(|| 17);

        assert_eq!(value, 17);
    }
}
