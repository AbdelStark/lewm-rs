//! Burn-backed Vision Transformer encoder from RFC 0002.

use burn::module::{Initializer, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::{Tensor, backend::Backend};

use crate::LewmCoreError;
use crate::config::{GeluVariant, VitConfig};
use crate::init::{ModelInitRng, model_init_rng, trunc_normal_param, zeros_param};

const GELU_TANH_CUBIC: f64 = 0.044_715;

/// Patch embedding layer implemented as a strided `Conv2d`.
#[derive(burn::module::Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    proj: Conv2d<B>,
    num_patches: usize,
    patch_size: usize,
}

impl<B: Backend> PatchEmbed<B> {
    /// Initialize a patch embedder from the encoder config.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::ConstructionFailed`] when the image/patch
    /// contract is incoherent, or [`LewmCoreError::InvalidInit`] when parameter
    /// initialization fails.
    pub fn init(
        config: &VitConfig,
        rng: &mut ModelInitRng,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        let num_patches =
            config
                .num_patches()
                .ok_or_else(|| LewmCoreError::ConstructionFailed {
                    reason: "encoder image_size must be divisible by patch_size".to_owned(),
                })?;
        let mut proj = Conv2dConfig::new(
            [config.num_channels, config.hidden_size],
            [config.patch_size, config.patch_size],
        )
        .with_stride([config.patch_size, config.patch_size])
        .with_initializer(Initializer::Zeros)
        .init(device);
        proj.weight = trunc_normal_param(
            [
                config.hidden_size,
                config.num_channels,
                config.patch_size,
                config.patch_size,
            ],
            rng,
            device,
        )?;
        proj.bias = Some(zeros_param([config.hidden_size], device)?);

        Ok(Self {
            proj,
            num_patches,
            patch_size: config.patch_size,
        })
    }

    /// Return the configured number of patch tokens.
    pub const fn num_patches(&self) -> usize {
        self.num_patches
    }

    /// Return the configured patch side length in pixels.
    pub const fn patch_size(&self) -> usize {
        self.patch_size
    }

    /// Convert images shaped `(B, C, H, W)` into patch tokens `(B, P, D)`.
    pub fn forward(&self, pixels: Tensor<B, 4>) -> Tensor<B, 3> {
        let encoded = self.proj.forward(pixels);
        let [batch_size, hidden_size, grid_h, grid_w] = encoded.dims();
        encoded
            .reshape([batch_size, hidden_size, grid_h * grid_w])
            .swap_dims(1, 2)
    }
}

/// `ViT` token embeddings: patch tokens, CLS token, position embeddings, dropout.
#[derive(burn::module::Module, Debug)]
pub struct ViTEmbeddings<B: Backend> {
    patch_embed: PatchEmbed<B>,
    cls_token: Param<Tensor<B, 3>>,
    pos_embed: Param<Tensor<B, 3>>,
    dropout: Dropout,
}

impl<B: Backend> ViTEmbeddings<B> {
    /// Initialize the embedding stack from the encoder config.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::ConstructionFailed`] when the config is
    /// unsupported, or [`LewmCoreError::InvalidInit`] when deterministic
    /// parameter initialization fails.
    pub fn init(
        config: &VitConfig,
        rng: &mut ModelInitRng,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        if !config.use_cls_token {
            return Err(LewmCoreError::ConstructionFailed {
                reason: "ViT encoder requires use_cls_token=true".to_owned(),
            });
        }
        let patch_embed = PatchEmbed::init(config, rng, device)?;
        let cls_token = trunc_normal_param([1, 1, config.hidden_size], rng, device)?;
        let pos_embed = trunc_normal_param(
            [1, patch_embed.num_patches() + 1, config.hidden_size],
            rng,
            device,
        )?;
        let dropout = DropoutConfig::new(config.hidden_dropout_prob).init();

        Ok(Self {
            patch_embed,
            cls_token,
            pos_embed,
            dropout,
        })
    }

    /// Embed a pixel batch into `(B, P+1, D)` tokens.
    pub fn forward(&self, pixels: Tensor<B, 4>, interpolate_pos_encoding: bool) -> Tensor<B, 3> {
        let patches = self.patch_embed.forward(pixels);
        let [batch_size, patch_count, hidden_size] = patches.dims();
        let cls = self.cls_token.val().expand([batch_size, 1, hidden_size]);
        let tokens = Tensor::cat(vec![cls, patches], 1);
        let pos = self.position_embedding(patch_count, hidden_size, interpolate_pos_encoding);

        self.dropout
            .forward(tokens + pos.expand([batch_size, patch_count + 1, hidden_size]))
    }

    fn position_embedding(
        &self,
        patch_count: usize,
        hidden_size: usize,
        interpolate_pos_encoding: bool,
    ) -> Tensor<B, 3> {
        let pos_embed = self.pos_embed.val();
        let [_, source_token_count, _] = pos_embed.dims();
        let source_patch_count = source_token_count - 1;

        if !interpolate_pos_encoding || source_patch_count == patch_count {
            return pos_embed;
        }

        interpolate_position_embedding(pos_embed, source_patch_count, patch_count, hidden_size)
    }
}

/// Multi-head self-attention used by the encoder.
#[derive(burn::module::Module, Debug)]
pub struct Attention<B: Backend> {
    qkv: Linear<B>,
    proj: Linear<B>,
    attn_drop: Dropout,
    proj_drop: Dropout,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl<B: Backend> Attention<B> {
    /// Initialize non-causal encoder attention from the encoder config.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::ConstructionFailed`] if attention dimensions are
    /// incoherent, or [`LewmCoreError::InvalidInit`] when parameter
    /// initialization fails.
    pub fn init(
        config: &VitConfig,
        rng: &mut ModelInitRng,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        let head_dim = config
            .head_dim()
            .ok_or_else(|| LewmCoreError::ConstructionFailed {
                reason: "encoder hidden_size must be divisible by num_attention_heads".to_owned(),
            })?;
        let head_dim_u32 =
            u32::try_from(head_dim).map_err(|err| LewmCoreError::ConstructionFailed {
                reason: format!("attention head_dim does not fit u32: {err}"),
            })?;
        let mut qkv = linear_zeros(config.hidden_size, 3 * config.hidden_size, device);
        qkv.weight = trunc_normal_param([config.hidden_size, 3 * config.hidden_size], rng, device)?;
        qkv.bias = Some(zeros_param([3 * config.hidden_size], device)?);

        let mut proj = linear_zeros(config.hidden_size, config.hidden_size, device);
        proj.weight = trunc_normal_param([config.hidden_size, config.hidden_size], rng, device)?;
        proj.bias = Some(zeros_param([config.hidden_size], device)?);

        Ok(Self {
            qkv,
            proj,
            attn_drop: DropoutConfig::new(config.attention_probs_dropout_prob).init(),
            proj_drop: DropoutConfig::new(config.hidden_dropout_prob).init(),
            num_heads: config.num_attention_heads,
            head_dim,
            scale: 1.0 / f64::from(head_dim_u32).sqrt(),
        })
    }

    /// Run bidirectional encoder attention over token sequences `(B, N, D)`.
    pub fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, token_count, hidden_size] = tokens.dims();
        let qkv = self
            .qkv
            .forward(tokens)
            .reshape([batch_size, token_count, 3, self.num_heads, self.head_dim])
            .permute([2, 0, 3, 1, 4]);
        let mut chunks = qkv.chunk(3, 0);
        let values = chunks.remove(2).squeeze_dim::<4>(0);
        let keys = chunks.remove(1).squeeze_dim::<4>(0);
        let queries = chunks.remove(0).squeeze_dim::<4>(0);

        let scores = queries.matmul(keys.swap_dims(2, 3)).mul_scalar(self.scale);
        let attention = self.attn_drop.forward(softmax(scores, 3));
        let attended = attention.matmul(values).swap_dims(1, 2).reshape([
            batch_size,
            token_count,
            hidden_size,
        ]);

        self.proj_drop.forward(self.proj.forward(attended))
    }
}

/// Feed-forward block inside a `ViT` encoder layer.
#[derive(burn::module::Module, Debug)]
pub struct MlpBlock<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    drop: Dropout,
    #[module(skip)]
    act: GeluVariant,
}

impl<B: Backend> MlpBlock<B> {
    /// Initialize a `ViT` encoder MLP block.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidInit`] when parameter initialization
    /// fails.
    pub fn init(
        config: &VitConfig,
        rng: &mut ModelInitRng,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        let mut fc1 = linear_zeros(config.hidden_size, config.intermediate_size, device);
        fc1.weight =
            trunc_normal_param([config.hidden_size, config.intermediate_size], rng, device)?;
        fc1.bias = Some(zeros_param([config.intermediate_size], device)?);

        let mut fc2 = linear_zeros(config.intermediate_size, config.hidden_size, device);
        fc2.weight =
            trunc_normal_param([config.intermediate_size, config.hidden_size], rng, device)?;
        fc2.bias = Some(zeros_param([config.hidden_size], device)?);

        Ok(Self {
            fc1,
            fc2,
            drop: DropoutConfig::new(config.hidden_dropout_prob).init(),
            act: config.hidden_act,
        })
    }

    /// Run `fc1 -> GELU -> dropout -> fc2`.
    pub fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let activated = match self.act {
            GeluVariant::Erf => gelu(self.fc1.forward(tokens)),
            GeluVariant::TanhApprox => gelu_tanh_tensor(self.fc1.forward(tokens)),
        };
        self.fc2.forward(self.drop.forward(activated))
    }
}

/// Pre-norm transformer encoder block.
#[derive(burn::module::Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    norm1: LayerNorm<B>,
    attn: Attention<B>,
    norm2: LayerNorm<B>,
    mlp: MlpBlock<B>,
}

impl<B: Backend> EncoderBlock<B> {
    /// Initialize one pre-norm `ViT` encoder block.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::ConstructionFailed`] if dimensions are
    /// incoherent, or [`LewmCoreError::InvalidInit`] when parameter
    /// initialization fails.
    pub fn init(
        config: &VitConfig,
        rng: &mut ModelInitRng,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        Ok(Self {
            norm1: layer_norm(config, device),
            attn: Attention::init(config, rng, device)?,
            norm2: layer_norm(config, device),
            mlp: MlpBlock::init(config, rng, device)?,
        })
    }

    /// Run the pre-norm attention and MLP residual branches.
    pub fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let tokens = tokens.clone() + self.attn.forward(self.norm1.forward(tokens));
        tokens.clone() + self.mlp.forward(self.norm2.forward(tokens))
    }
}

/// Vision Transformer output.
#[derive(Debug, Clone)]
pub struct ViTOutput<B: Backend> {
    /// All token outputs after the final `LayerNorm`. Shape `(B, P+1, D)`.
    pub last_hidden_state: Tensor<B, 3>,
}

/// Burn-backed Vision Transformer encoder.
#[derive(burn::module::Module, Debug)]
pub struct Vit<B: Backend> {
    embeddings: ViTEmbeddings<B>,
    blocks: Vec<EncoderBlock<B>>,
    norm: LayerNorm<B>,
    #[module(skip)]
    config: VitConfig,
}

impl<B: Backend> Vit<B> {
    /// Initialize a `ViT` with the deterministic default model-init seed.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::ConstructionFailed`] when the config violates
    /// shape invariants, or [`LewmCoreError::InvalidInit`] when deterministic
    /// parameter initialization fails.
    pub fn init(config: VitConfig, device: &B::Device) -> Result<Self, LewmCoreError> {
        Self::init_with_seed(config, 0, device)
    }

    /// Initialize a `ViT` with an explicit global model-init seed.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::ConstructionFailed`] when the config violates
    /// shape invariants, or [`LewmCoreError::InvalidInit`] when deterministic
    /// parameter initialization fails.
    pub fn init_with_seed(
        config: VitConfig,
        seed: u64,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        config
            .validate_shape_contract()
            .map_err(|errors| LewmCoreError::ConstructionFailed {
                reason: errors.join("; "),
            })?;
        let mut rng = model_init_rng(seed)?;
        Self::init_with_rng(config, &mut rng, device)
    }

    pub(crate) fn init_with_rng(
        config: VitConfig,
        rng: &mut ModelInitRng,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        let embeddings = ViTEmbeddings::init(&config, rng, device)?;
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            blocks.push(EncoderBlock::init(&config, rng, device)?);
        }
        let norm = layer_norm(&config, device);

        Ok(Self {
            embeddings,
            blocks,
            norm,
            config,
        })
    }

    /// Run the encoder and return all post-norm token states.
    pub fn forward(&self, pixels: Tensor<B, 4>) -> ViTOutput<B> {
        let mut tokens = self
            .embeddings
            .forward(pixels, self.config.interpolate_pos_encoding);
        for block in &self.blocks {
            tokens = block.forward(tokens);
        }

        ViTOutput {
            last_hidden_state: self.norm.forward(tokens),
        }
    }

    /// Extract the post-final-LayerNorm CLS row. Shape `(B, D)`.
    pub fn cls_from(output: &ViTOutput<B>) -> Tensor<B, 2> {
        let [batch_size, _, hidden_size] = output.last_hidden_state.dims();
        output
            .last_hidden_state
            .clone()
            .slice([0..batch_size, 0..1, 0..hidden_size])
            .squeeze_dim::<2>(1)
    }
}

fn layer_norm<B: Backend>(config: &VitConfig, device: &B::Device) -> LayerNorm<B> {
    LayerNormConfig::new(config.hidden_size)
        .with_epsilon(config.layer_norm_eps)
        .init(device)
}

fn linear_zeros<B: Backend>(d_input: usize, d_output: usize, device: &B::Device) -> Linear<B> {
    LinearConfig::new(d_input, d_output)
        .with_initializer(Initializer::Zeros)
        .init(device)
}

fn gelu_tanh_tensor<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    let cubic = tensor.clone().powi_scalar(3).mul_scalar(GELU_TANH_CUBIC);
    let inner = (tensor.clone() + cubic)
        .mul_scalar(std::f64::consts::FRAC_2_SQRT_PI / std::f64::consts::SQRT_2)
        .tanh();

    tensor.mul_scalar(0.5).mul(inner.add_scalar(1.0))
}

fn interpolate_position_embedding<B: Backend>(
    pos_embed: Tensor<B, 3>,
    source_patch_count: usize,
    target_patch_count: usize,
    hidden_size: usize,
) -> Tensor<B, 3> {
    let source_side = square_side(source_patch_count);
    let target_side = square_side(target_patch_count);
    let class_pos = pos_embed.clone().slice([0..1, 0..1, 0..hidden_size]);
    #[allow(clippy::range_plus_one)]
    let patch_pos = pos_embed
        .slice([0..1, 1..source_patch_count + 1, 0..hidden_size])
        .reshape([1, source_side, source_side, hidden_size])
        .permute([0, 3, 1, 2]);
    let resized = burn::nn::interpolate::Interpolate2dConfig::new()
        .with_output_size(Some([target_side, target_side]))
        .with_mode(burn::nn::interpolate::InterpolateMode::Cubic)
        .init()
        .forward(patch_pos)
        .permute([0, 2, 3, 1])
        .reshape([1, target_patch_count, hidden_size]);

    Tensor::cat(vec![class_pos, resized], 1)
}

fn square_side(patch_count: usize) -> usize {
    let mut side = 1usize;
    while side.saturating_mul(side) < patch_count {
        side += 1;
    }
    side
}
