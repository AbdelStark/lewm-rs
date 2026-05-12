//! Configuration contracts for the core `LeWM` architecture.
//!
//! These structs are intentionally strict at the Serde boundary: missing fields
//! fall back to the PRD/RFC defaults, while unknown fields fail deserialization.

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Vision Transformer size family used by the upstream config file.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VitSize {
    /// ViT-Tiny, accepted only for parsing upstream metadata.
    Tiny,
    /// ViT-Small, the locked lewm-rs v1 model size.
    #[default]
    Small,
    /// ViT-Base, accepted only for parsing upstream metadata.
    Base,
}

/// GELU implementation variant used inside the `ViT` encoder.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GeluVariant {
    /// Error-function GELU.
    Erf,
    /// Tanh-approximate GELU, serialized as the canonical config value.
    #[serde(rename = "gelu_tanh", alias = "tanh_approx")]
    #[default]
    TanhApprox,
}

/// Normalization layer used by projector-style MLP heads.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NormVariant {
    /// `BatchNorm1d` over the feature dimension.
    #[serde(rename = "batch_norm_1d", alias = "batch_norm1d")]
    #[default]
    BatchNorm1d,
    /// `LayerNorm` over the feature dimension.
    LayerNorm,
    /// No normalization.
    None,
}

/// Vision Transformer configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate)]
#[serde(default, deny_unknown_fields)]
#[allow(clippy::struct_excessive_bools)]
pub struct VitConfig {
    /// Upstream size family. The locked v1 value is `small`.
    pub size: VitSize,
    /// Square image side length in pixels.
    #[validate(range(min = 16, max = 4096))]
    pub image_size: usize,
    /// Square patch side length in pixels.
    #[validate(range(min = 1, max = 512))]
    pub patch_size: usize,
    /// Input channel count.
    #[validate(range(min = 1, max = 16))]
    pub num_channels: usize,
    /// Embedding dimension `D`.
    #[serde(alias = "hidden_dim")]
    #[validate(range(min = 1, max = 8192))]
    pub hidden_size: usize,
    /// Number of transformer blocks.
    #[serde(alias = "depth")]
    #[validate(range(min = 1, max = 128))]
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    #[serde(alias = "heads")]
    #[validate(range(min = 1, max = 128))]
    pub num_attention_heads: usize,
    /// FFN inner dimension.
    #[validate(range(min = 1, max = 65_536))]
    pub intermediate_size: usize,
    /// Encoder MLP activation.
    pub hidden_act: GeluVariant,
    /// Attention probability dropout.
    #[validate(range(min = 0.0, max = 1.0))]
    pub attention_probs_dropout_prob: f64,
    /// Residual/FFN dropout.
    #[validate(range(min = 0.0, max = 1.0))]
    pub hidden_dropout_prob: f64,
    /// `LayerNorm` epsilon.
    #[validate(range(min = 1.0e-12, max = 1.0e-3))]
    pub layer_norm_eps: f64,
    /// Whether to include a learnable CLS token.
    pub use_cls_token: bool,
    /// Whether to interpolate position embeddings at forward time.
    pub interpolate_pos_encoding: bool,
    /// Whether upstream HF weights use a mask token.
    pub use_mask_token: bool,
    /// Whether upstream HF weights start from an external pretrained encoder.
    pub pretrained: bool,
}

impl Default for VitConfig {
    fn default() -> Self {
        Self {
            size: VitSize::Small,
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 384,
            num_hidden_layers: 12,
            num_attention_heads: 6,
            intermediate_size: 1536,
            hidden_act: GeluVariant::TanhApprox,
            attention_probs_dropout_prob: 0.0,
            hidden_dropout_prob: 0.0,
            layer_norm_eps: 1.0e-12,
            use_cls_token: true,
            interpolate_pos_encoding: false,
            use_mask_token: false,
            pretrained: false,
        }
    }
}

impl VitConfig {
    /// Create a config with the locked defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of patches per image side.
    pub fn grid_size(&self) -> Option<usize> {
        if self.patch_size == 0 || self.image_size % self.patch_size != 0 {
            return None;
        }

        Some(self.image_size / self.patch_size)
    }

    /// Total patch count, excluding the CLS token.
    pub fn num_patches(&self) -> Option<usize> {
        self.grid_size().map(|grid_size| grid_size * grid_size)
    }

    /// Attention head dimension implied by the encoder config.
    pub fn head_dim(&self) -> Option<usize> {
        if self.num_attention_heads == 0 || self.hidden_size % self.num_attention_heads != 0 {
            return None;
        }

        Some(self.hidden_size / self.num_attention_heads)
    }

    /// Cross-field shape validation that Serde/validator ranges cannot express.
    pub fn shape_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.patch_size == 0 {
            errors.push("encoder.patch_size must be non-zero".to_owned());
        } else if self.image_size % self.patch_size != 0 {
            errors.push("encoder.image_size must be divisible by encoder.patch_size".to_owned());
        }

        if self.num_attention_heads == 0 {
            errors.push("encoder.num_attention_heads must be non-zero".to_owned());
        } else if self.hidden_size % self.num_attention_heads != 0 {
            errors.push(
                "encoder.hidden_size must be divisible by encoder.num_attention_heads".to_owned(),
            );
        }

        errors
    }

    /// Return `Ok` when all cross-field shape invariants hold.
    ///
    /// # Errors
    ///
    /// Returns a list of invariant failures when image/patch or attention
    /// dimensions are incoherent.
    pub fn validate_shape_contract(&self) -> Result<(), Vec<String>> {
        let errors = self.shape_errors();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Action embedder configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct EmbedderConfig {
    /// Per-step action dimensionality. `PushT` defaults to 2.
    #[validate(range(min = 1, max = 1024))]
    pub input_dim: usize,
    /// Intermediate dim after the Conv1d-k1 smoother.
    #[validate(range(min = 1, max = 4096))]
    pub smoothed_dim: usize,
    /// Output embedding dim used by the predictor.
    #[validate(range(min = 1, max = 8192))]
    pub emb_dim: usize,
    /// Inner MLP scale.
    #[validate(range(min = 1, max = 64))]
    pub mlp_scale: usize,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            input_dim: 2,
            smoothed_dim: 16,
            emb_dim: 64,
            mlp_scale: 4,
        }
    }
}

impl EmbedderConfig {
    /// Create a config with the locked defaults.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Projector and prediction-projector MLP configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct MlpConfig {
    /// Input feature dimension.
    #[validate(range(min = 1, max = 8192))]
    pub input_dim: usize,
    /// Hidden feature dimension.
    #[validate(range(min = 1, max = 65_536))]
    pub hidden_dim: usize,
    /// Output feature dimension.
    #[validate(range(min = 1, max = 8192))]
    pub output_dim: usize,
    /// Normalization variant.
    pub norm: NormVariant,
}

impl Default for MlpConfig {
    fn default() -> Self {
        Self {
            input_dim: 384,
            hidden_dim: 1536,
            output_dim: 384,
            norm: NormVariant::BatchNorm1d,
        }
    }
}

impl MlpConfig {
    /// Create a config with the locked defaults.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Autoregressive predictor configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct PredictorConfig {
    /// Max sequence length supported by the learned positional embedding.
    #[validate(range(min = 1, max = 1024))]
    pub num_frames: usize,
    /// Number of conditional transformer blocks.
    #[validate(range(min = 1, max = 128))]
    pub depth: usize,
    /// Number of attention heads per block.
    #[validate(range(min = 1, max = 128))]
    pub heads: usize,
    /// FFN inner dimension in each conditional block.
    #[validate(range(min = 1, max = 65_536))]
    pub mlp_dim: usize,
    /// Per-head dimension.
    #[validate(range(min = 1, max = 1024))]
    pub dim_head: usize,
    /// Input token dimension accepted by the predictor.
    #[validate(range(min = 1, max = 8192))]
    pub input_dim: usize,
    /// Token dimension inside the predictor.
    #[validate(range(min = 1, max = 8192))]
    pub hidden_dim: usize,
    /// Output token dimension produced by the predictor.
    #[validate(range(min = 1, max = 8192))]
    pub output_dim: usize,
    /// Action embedding dimension consumed by AdaLN-zero.
    #[validate(range(min = 1, max = 8192))]
    pub action_emb_dim: usize,
    /// Sequence dropout.
    #[validate(range(min = 0.0, max = 1.0))]
    pub dropout: f64,
    /// Embedding dropout.
    #[validate(range(min = 0.0, max = 1.0))]
    pub emb_dropout: f64,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            num_frames: 16,
            depth: 6,
            heads: 6,
            mlp_dim: 1536,
            dim_head: 64,
            input_dim: 384,
            hidden_dim: 384,
            output_dim: 384,
            action_emb_dim: 64,
            dropout: 0.0,
            emb_dropout: 0.0,
        }
    }
}

impl PredictorConfig {
    /// Create a config with the locked defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cross-field shape validation that Serde/validator ranges cannot express.
    pub fn shape_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.heads.saturating_mul(self.dim_head) != self.hidden_dim {
            errors.push(
                "predictor.heads * predictor.dim_head must equal predictor.hidden_dim".to_owned(),
            );
        }

        if self.input_dim != self.hidden_dim {
            errors.push("predictor.input_dim must equal predictor.hidden_dim".to_owned());
        }

        if self.output_dim != self.hidden_dim {
            errors.push("predictor.output_dim must equal predictor.hidden_dim".to_owned());
        }

        errors
    }

    /// Return `Ok` when all cross-field shape invariants hold.
    ///
    /// # Errors
    ///
    /// Returns a list of invariant failures when attention factorization or
    /// token dimensions are incoherent.
    pub fn validate_shape_contract(&self) -> Result<(), Vec<String>> {
        let errors = self.shape_errors();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Top-level JEPA model configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate)]
#[serde(default, deny_unknown_fields)]
pub struct JepaConfig {
    /// `ViT` encoder config.
    #[validate(nested)]
    pub encoder: VitConfig,
    /// Action embedder config.
    #[validate(nested)]
    pub action_encoder: EmbedderConfig,
    /// Autoregressive predictor config.
    #[validate(nested)]
    pub predictor: PredictorConfig,
    /// Encoder projector config.
    #[validate(nested)]
    pub projector: MlpConfig,
    /// Predictor output projection config.
    #[validate(nested)]
    pub pred_proj: MlpConfig,
    /// Number of context steps fed to the predictor.
    #[validate(range(min = 1, max = 100))]
    pub history_size: usize,
    /// Maximum rollout horizon supported.
    #[validate(range(min = 1, max = 1024))]
    pub horizon: usize,
}

impl Default for JepaConfig {
    fn default() -> Self {
        Self {
            encoder: VitConfig::default(),
            action_encoder: EmbedderConfig::default(),
            predictor: PredictorConfig::default(),
            projector: MlpConfig::default(),
            pred_proj: MlpConfig::default(),
            history_size: 3,
            horizon: 8,
        }
    }
}

impl JepaConfig {
    /// Create a config with the locked defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cross-field shape validation that Serde/validator ranges cannot express.
    pub fn shape_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        errors.extend(self.encoder.shape_errors());
        errors.extend(self.predictor.shape_errors());

        if self.projector.input_dim != self.encoder.hidden_size {
            errors.push("projector.input_dim must equal encoder.hidden_size".to_owned());
        }

        if self.projector.output_dim != self.predictor.hidden_dim {
            errors.push("projector.output_dim must equal predictor.hidden_dim".to_owned());
        }

        if self.pred_proj.input_dim != self.predictor.hidden_dim {
            errors.push("pred_proj.input_dim must equal predictor.hidden_dim".to_owned());
        }

        if self.pred_proj.output_dim != self.predictor.hidden_dim {
            errors.push("pred_proj.output_dim must equal predictor.hidden_dim".to_owned());
        }

        if self.action_encoder.emb_dim != self.predictor.action_emb_dim {
            errors.push("action_encoder.emb_dim must equal predictor.action_emb_dim".to_owned());
        }

        if self.history_size > self.horizon {
            errors.push("history_size must be less than or equal to horizon".to_owned());
        }

        if self.horizon > self.predictor.num_frames {
            errors.push("horizon must be less than or equal to predictor.num_frames".to_owned());
        }

        errors
    }

    /// Return `Ok` when all cross-field shape invariants hold.
    ///
    /// # Errors
    ///
    /// Returns a list of invariant failures when nested model dimensions do not
    /// compose into the locked `JEPA` data flow.
    pub fn validate_shape_contract(&self) -> Result<(), Vec<String>> {
        let errors = self.shape_errors();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_locked_pusht_model_contract() {
        let config = JepaConfig::default();

        assert_eq!(config.encoder.size, VitSize::Small);
        assert_eq!(config.encoder.image_size, 224);
        assert_eq!(config.encoder.patch_size, 16);
        assert_eq!(config.encoder.hidden_size, 384);
        assert_eq!(config.encoder.num_hidden_layers, 12);
        assert_eq!(config.encoder.num_attention_heads, 6);
        assert_eq!(config.encoder.intermediate_size, 1536);
        assert_eq!(config.action_encoder.input_dim, 2);
        assert_eq!(config.action_encoder.emb_dim, 64);
        assert_eq!(config.predictor.hidden_dim, 384);
        assert_eq!(config.predictor.heads * config.predictor.dim_head, 384);
        assert_eq!(config.projector.norm, NormVariant::BatchNorm1d);
        assert_eq!(config.pred_proj.output_dim, 384);
        assert_eq!(config.history_size, 3);
        assert_eq!(config.horizon, 8);
        assert_eq!(config.encoder.num_patches(), Some(196));

        config.validate().expect("default ranges should validate");
        config
            .validate_shape_contract()
            .expect("default shape contract should validate");
    }

    #[test]
    fn model_config_round_trips_through_toml() {
        let fixture = r#"
history_size = 3
horizon = 8

[encoder]
size = "small"
image_size = 224
patch_size = 16
num_channels = 3
hidden_size = 384
num_hidden_layers = 12
num_attention_heads = 6
intermediate_size = 1536
hidden_act = "gelu_tanh"
attention_probs_dropout_prob = 0.0
hidden_dropout_prob = 0.0
layer_norm_eps = 1.0e-12
use_cls_token = true
interpolate_pos_encoding = false

[action_encoder]
input_dim = 2
smoothed_dim = 16
emb_dim = 64
mlp_scale = 4

[predictor]
num_frames = 16
depth = 6
heads = 6
mlp_dim = 1536
dim_head = 64
hidden_dim = 384
action_emb_dim = 64
dropout = 0.0
emb_dropout = 0.0

[projector]
input_dim = 384
hidden_dim = 1536
output_dim = 384
norm = "batch_norm_1d"

[pred_proj]
input_dim = 384
hidden_dim = 1536
output_dim = 384
norm = "batch_norm_1d"
"#;

        let parsed: JepaConfig = toml::from_str(fixture).expect("fixture should parse");
        parsed.validate().expect("fixture ranges should validate");
        parsed
            .validate_shape_contract()
            .expect("fixture shape contract should validate");

        let encoded = toml::to_string_pretty(&parsed).expect("fixture should serialize");
        let reparsed: JepaConfig =
            toml::from_str(&encoded).expect("serialized fixture should parse");
        assert_eq!(reparsed, parsed);
    }

    #[test]
    fn unknown_toml_keys_are_rejected() {
        let err = toml::from_str::<VitConfig>(
            r"
image_size = 224
patch_size = 16
wieght_decay = 0.05
",
        )
        .expect_err("unknown fields should fail deserialization");

        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn validator_ranges_and_shape_invariants_are_enforced() {
        let invalid_range = VitConfig {
            patch_size: 0,
            ..VitConfig::default()
        };
        assert!(invalid_range.validate().is_err());

        let mut invalid_shape = JepaConfig::default();
        invalid_shape.predictor.heads = 7;
        invalid_shape
            .validate()
            .expect("individual ranges remain valid");

        let errors = invalid_shape
            .validate_shape_contract()
            .expect_err("shape invariant should fail");
        assert!(
            errors
                .iter()
                .any(|error| error.contains("heads * predictor.dim_head"))
        );
    }

    #[test]
    fn upstream_aliases_parse_without_weakening_serialized_schema() {
        let parsed: VitConfig = toml::from_str(
            r"
hidden_dim = 384
depth = 12
heads = 6
",
        )
        .expect("upstream aliases should parse");

        assert_eq!(parsed.hidden_size, 384);
        assert_eq!(parsed.num_hidden_layers, 12);
        assert_eq!(parsed.num_attention_heads, 6);

        let serialized = toml::to_string_pretty(&parsed).expect("aliases should serialize");
        assert!(serialized.contains("hidden_size"));
        assert!(serialized.contains("num_hidden_layers"));
        assert!(serialized.contains("num_attention_heads"));
    }
}
