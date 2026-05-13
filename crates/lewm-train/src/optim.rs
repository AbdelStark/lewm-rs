//! `AdamW` optimizer configuration and parameter partitioning.

/// RFC 0005 `AdamW` first-moment coefficient.
pub const ADAMW_BETA1: f64 = 0.9;

/// RFC 0005 `AdamW` second-moment coefficient.
pub const ADAMW_BETA2: f64 = 0.95;

/// RFC 0005 `AdamW` numerical stability term.
pub const ADAMW_EPSILON: f64 = 1e-8;

/// RFC 0005 decoupled `AdamW` weight decay for decay-eligible parameters.
pub const ADAMW_WEIGHT_DECAY: f64 = 0.05;

/// RFC 0005 optimizer configuration.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimConfig {
    /// `AdamW` first-moment coefficient.
    pub beta1: f64,

    /// `AdamW` second-moment coefficient.
    pub beta2: f64,

    /// `AdamW` numerical stability term.
    pub epsilon: f64,

    /// Decoupled `AdamW` weight decay for decay-eligible parameters.
    pub weight_decay: f64,
}

impl Default for OptimConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimConfig {
    /// Creates a new config with RFC 0005 defaults.
    pub const fn new() -> Self {
        Self {
            beta1: ADAMW_BETA1,
            beta2: ADAMW_BETA2,
            epsilon: ADAMW_EPSILON,
            weight_decay: ADAMW_WEIGHT_DECAY,
        }
    }

    /// Overrides the first-moment coefficient.
    pub const fn with_beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Overrides the second-moment coefficient.
    pub const fn with_beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Overrides the numerical stability term.
    pub const fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Overrides the decoupled weight-decay coefficient.
    pub const fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Returns the same config with decoupled weight decay disabled.
    pub const fn without_weight_decay(self) -> Self {
        self.with_weight_decay(0.0)
    }
}

/// Deterministic decay/no-decay parameter split.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParamPartition {
    /// Parameters that should receive `AdamW` decoupled weight decay.
    pub decay_params: Vec<String>,

    /// Parameters that should not receive `AdamW` decoupled weight decay.
    pub no_decay_params: Vec<String>,
}

/// `LeWM` `AdamW` wrapper that carries RFC 0005 optimizer configuration and
/// decay/no-decay parameter groups.
#[must_use]
#[derive(Clone, Debug)]
pub struct LewmAdamW {
    inner: OptimConfig,
    decay_params: Vec<String>,
    no_decay_params: Vec<String>,
}

impl LewmAdamW {
    /// Creates an `AdamW` wrapper from RFC config and parameter names.
    pub fn new<I, S>(config: OptimConfig, param_names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let partition = partition_decay_no_decay(param_names);

        Self {
            inner: config,
            decay_params: partition.decay_params,
            no_decay_params: partition.no_decay_params,
        }
    }

    /// Returns the `AdamW` configuration for decay-eligible parameters.
    pub const fn decay_config(&self) -> OptimConfig {
        self.inner
    }

    /// Returns the `AdamW` configuration for parameters excluded from decay.
    pub const fn no_decay_config(&self) -> OptimConfig {
        self.inner.without_weight_decay()
    }

    /// Returns parameter names that should receive weight decay.
    #[must_use]
    pub fn decay_params(&self) -> &[String] {
        &self.decay_params
    }

    /// Returns parameter names that should not receive weight decay.
    #[must_use]
    pub fn no_decay_params(&self) -> &[String] {
        &self.no_decay_params
    }
}

/// Partitions parameter names into `AdamW` decay and no-decay groups.
///
/// The output preserves input order within each group.
#[must_use]
pub fn partition_decay_no_decay<I, S>(param_names: I) -> ParamPartition
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut partition = ParamPartition {
        decay_params: Vec::new(),
        no_decay_params: Vec::new(),
    };

    for name in param_names {
        let name = name.as_ref();
        if uses_weight_decay(name) {
            partition.decay_params.push(name.to_owned());
        } else {
            partition.no_decay_params.push(name.to_owned());
        }
    }

    partition
}

/// Returns whether RFC 0005 applies decoupled weight decay to a parameter.
#[must_use]
pub fn uses_weight_decay(param_name: &str) -> bool {
    let normalized = param_name.to_ascii_lowercase();
    let segments = normalized.split('.').collect::<Vec<_>>();

    if has_terminal_segment(&normalized, "bias")
        || segments.iter().any(|segment| *segment == "cls_token")
        || segments.iter().any(|segment| *segment == "pos_embed")
        || segments.iter().any(|segment| is_norm_segment(segment))
    {
        return false;
    }

    has_terminal_segment(&normalized, "weight")
}

fn has_terminal_segment(param_name: &str, terminal: &str) -> bool {
    param_name
        .rsplit('.')
        .next()
        .is_some_and(|segment| segment == terminal)
}

fn is_norm_segment(segment: &str) -> bool {
    segment == "bn"
        || segment.starts_with("bn")
        || segment.starts_with("norm")
        || segment.ends_with("norm")
        || segment.contains("layernorm")
        || segment.contains("batchnorm")
        || segment.contains("batch_norm")
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOY_LR: f64 = 0.01;

    #[derive(Clone, Copy, Debug, Default)]
    struct AdamWScalarState {
        step: i32,
        exp_avg: f64,
        exp_avg_sq: f64,
    }

    fn adamw_step_matches_pytorch_formula(
        param: f64,
        grad: f64,
        state: &mut AdamWScalarState,
        config: &OptimConfig,
        apply_weight_decay: bool,
    ) -> f64 {
        state.step += 1;
        state.exp_avg = (config.beta1 * state.exp_avg) + ((1.0 - config.beta1) * grad);
        state.exp_avg_sq = (config.beta2 * state.exp_avg_sq) + ((1.0 - config.beta2) * grad * grad);

        let decayed_param = if apply_weight_decay {
            param * (1.0 - (TOY_LR * config.weight_decay))
        } else {
            param
        };
        let corrected_avg = state.exp_avg / (1.0 - config.beta1.powi(state.step));
        let corrected_avg_sq = state.exp_avg_sq / (1.0 - config.beta2.powi(state.step));

        decayed_param - (TOY_LR * corrected_avg / (corrected_avg_sq.sqrt() + config.epsilon))
    }

    #[test]
    fn adamw_config_defaults_match_rfc_0005() {
        let config = OptimConfig::new();

        assert!((config.beta1 - ADAMW_BETA1).abs() < f64::EPSILON);
        assert!((config.beta2 - ADAMW_BETA2).abs() < f64::EPSILON);
        assert!((config.epsilon - ADAMW_EPSILON).abs() < f64::EPSILON);
        assert!((config.weight_decay - ADAMW_WEIGHT_DECAY).abs() < f64::EPSILON);
    }

    #[test]
    fn adamw_param_partition_decay_no_decay() {
        let partition = partition_decay_no_decay([
            "encoder.patch_embed.proj.weight",
            "encoder.cls_token",
            "encoder.pos_embed",
            "encoder.blocks.0.norm1.weight",
            "encoder.blocks.0.norm1.bias",
            "encoder.blocks.0.attn.qkv.bias",
            "embedder.smoother.weight",
            "embedder.fc1.bias",
            "predictor.blocks.0.mlp.fc1.weight",
            "predictor.action_embed.linear.weight",
            "data_batch_norm.weight",
        ]);

        assert_eq!(
            partition.decay_params,
            [
                "encoder.patch_embed.proj.weight",
                "embedder.smoother.weight",
                "predictor.blocks.0.mlp.fc1.weight",
                "predictor.action_embed.linear.weight",
            ]
        );
        assert_eq!(
            partition.no_decay_params,
            [
                "encoder.cls_token",
                "encoder.pos_embed",
                "encoder.blocks.0.norm1.weight",
                "encoder.blocks.0.norm1.bias",
                "encoder.blocks.0.attn.qkv.bias",
                "embedder.fc1.bias",
                "data_batch_norm.weight",
            ]
        );
    }

    #[test]
    fn lewm_adamw_keeps_decay_and_no_decay_configs_separate() {
        let optimizer = LewmAdamW::new(
            OptimConfig::new(),
            [
                "encoder.patch_embed.proj.weight",
                "encoder.blocks.0.norm1.weight",
            ],
        );

        assert_eq!(
            optimizer.decay_params(),
            &["encoder.patch_embed.proj.weight".to_owned()]
        );
        assert_eq!(
            optimizer.no_decay_params(),
            &["encoder.blocks.0.norm1.weight".to_owned()]
        );
        assert!((optimizer.decay_config().weight_decay - ADAMW_WEIGHT_DECAY).abs() < f64::EPSILON);
        assert!(optimizer.no_decay_config().weight_decay.abs() < f64::EPSILON);
    }

    #[test]
    fn adamw_step_matches_pytorch_on_toy_problem() {
        let config = OptimConfig::new();
        let mut states = [AdamWScalarState::default(); 3];
        let mut params = [1.0, -2.0, 0.5];

        for grads in [[0.1, -0.2, 0.05], [-0.3, 0.4, 0.2]] {
            for ((param, grad), state) in params.iter_mut().zip(grads).zip(states.iter_mut()) {
                *param = adamw_step_matches_pytorch_formula(*param, grad, state, &config, true);
            }
        }

        let expected = [0.993_898_208_458, -1.991_639_239_89, 0.480_757_809_223];
        for (actual, expected) in params.iter().zip(expected) {
            assert!((*actual - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn adamw_toy_problem_skips_decay_for_no_decay_group() {
        let config = OptimConfig::new();
        let mut state = AdamWScalarState::default();
        let param = adamw_step_matches_pytorch_formula(1.0, 0.1, &mut state, &config, false);

        assert!((param - 0.990_000_001).abs() <= 1e-6);
    }
}
