//! NVTX integration for NVIDIA Nsight timelines.
//!
//! This module exposes a small `tracing-subscriber` layer that maps entered
//! tracing spans to NVTX ranges. Enable it with the `nvtx` feature and compose it
//! with the process subscriber used by the training binary.

use tracing::{Id, Subscriber};
use tracing_subscriber::{Layer, layer::Context, registry::LookupSpan};

/// A `tracing-subscriber` layer that mirrors span enter/exit events to NVTX.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct NvtxLayer {
    enabled: bool,
}

impl Default for NvtxLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl NvtxLayer {
    /// Create an enabled NVTX layer.
    #[must_use]
    pub const fn new() -> Self {
        Self { enabled: true }
    }

    /// Create a disabled layer that can be left installed in non-profile runs.
    #[must_use]
    pub const fn disabled() -> Self {
        Self { enabled: false }
    }

    /// Return whether this layer emits NVTX ranges.
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        self.enabled
    }
}

/// Convenience constructor for the Nsight NVTX tracing layer.
#[must_use]
pub const fn nvtx_layer() -> NvtxLayer {
    NvtxLayer::new()
}

impl<S> Layer<S> for NvtxLayer
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        if !self.enabled {
            return;
        }

        if let Some(span) = ctx.span(id) {
            let metadata = span.metadata();
            let _ = ::nvtx::range_push!("{}:{}", metadata.target(), metadata.name());
        }
    }

    fn on_exit(&self, _id: &Id, _ctx: Context<'_, S>) {
        if self.enabled {
            let _ = ::nvtx::range_pop!();
        }
    }
}

#[cfg(test)]
mod tests {
    use tracing_subscriber::{layer::SubscriberExt, registry::Registry};

    use super::*;

    #[test]
    fn nvtx_layer_can_be_composed_with_tracing_registry() {
        let subscriber = Registry::default().with(NvtxLayer::new());

        tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!("training.step", step = 1_u64, epoch = 0_u64);
            let _entered = span.enter();
        });
    }

    #[test]
    fn disabled_layer_keeps_configuration_visible() {
        assert!(!NvtxLayer::disabled().is_enabled());
        assert!(nvtx_layer().is_enabled());
    }
}
