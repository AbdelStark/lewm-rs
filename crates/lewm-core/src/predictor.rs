//! Burn-backed autoregressive predictor from RFC 0002.

use burn::module::{Initializer, Param};
use burn::nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::{Tensor, TensorData, backend::Backend};

use crate::LewmCoreError;
use crate::ada_ln::AdaLNZero;
use crate::config::PredictorConfig;
use crate::init::{ModelInitRng, model_init_rng, trunc_normal_param, zeros_param};
use crate::tensor_ops::{DeviceKey, build_causal_mask};

const DEFAULT_LAYER_NORM_EPSILON: f64 = 1.0e-5;
const AFFINE_FREE_LAYER_NORM_EPSILON: f64 = 1.0e-6;

/// Pre-norm transformer block with AdaLN-zero action conditioning.
#[derive(burn::module::Module, Debug)]
pub struct ConditionalBlock<B: Backend> {
    #[module(skip)]
    norm1: AffineFreeLayerNorm,
    attn: CausalSelfAttention<B>,
    #[module(skip)]
    norm2: AffineFreeLayerNorm,
    mlp: PredictorMlpBlock<B>,
    adaln: AdaLNZero<B>,
}

impl<B: Backend> ConditionalBlock<B> {
    /// Initialize a conditional predictor block.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::ConstructionFailed`] for incoherent attention
    /// dimensions, or [`LewmCoreError::InvalidInit`] for deterministic
    /// parameter initialization failures.
    pub fn init(
        config: &PredictorConfig,
        rng: &mut ModelInitRng,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        Ok(Self {
            norm1: AffineFreeLayerNorm::new(config.hidden_dim),
            attn: CausalSelfAttention::init(config, rng, device)?,
            norm2: AffineFreeLayerNorm::new(config.hidden_dim),
            mlp: PredictorMlpBlock::init(config, rng, device)?,
            adaln: AdaLNZero::init(config.hidden_dim, config.action_emb_dim, device)?,
        })
    }

    /// Run the AdaLN-zero modulated block on `(B, T, D)` tokens and actions.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::InvalidTensorOp`] if causal mask construction
    /// fails.
    pub fn forward(
        &self,
        tokens: Tensor<B, 3>,
        conditioning: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>, LewmCoreError> {
        let mods = self.adaln.forward(conditioning);

        let attn_input = modulate(
            self.norm1.forward(tokens.clone()),
            mods.shift_msa,
            mods.scale_msa,
        );
        let tokens = tokens + mods.gate_msa * self.attn.forward(attn_input)?;

        let mlp_input = modulate(
            self.norm2.forward(tokens.clone()),
            mods.shift_mlp,
            mods.scale_mlp,
        );

        Ok(tokens + mods.gate_mlp * self.mlp.forward(mlp_input))
    }
}

/// Autoregressive predictor over projected context embeddings.
#[derive(burn::module::Module, Debug)]
pub struct ArPredictor<B: Backend> {
    pos_embed: Param<Tensor<B, 3>>,
    dropout: Dropout,
    blocks: Vec<ConditionalBlock<B>>,
    norm: LayerNorm<B>,
    #[module(skip)]
    config: PredictorConfig,
}

impl<B: Backend> ArPredictor<B> {
    /// Initialize a predictor with the deterministic default model-init seed.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::ConstructionFailed`] when the config violates
    /// predictor shape invariants, or [`LewmCoreError::InvalidInit`] when
    /// deterministic parameter initialization fails.
    pub fn init(config: PredictorConfig, device: &B::Device) -> Result<Self, LewmCoreError> {
        Self::init_with_seed(config, 0, device)
    }

    /// Initialize a predictor with an explicit global model-init seed.
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::ConstructionFailed`] when the config violates
    /// predictor shape invariants, or [`LewmCoreError::InvalidInit`] when
    /// deterministic parameter initialization fails.
    pub fn init_with_seed(
        config: PredictorConfig,
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
        config: PredictorConfig,
        rng: &mut ModelInitRng,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        if config.depth == 0 {
            return Err(LewmCoreError::ConstructionFailed {
                reason: "predictor depth must be non-zero".to_owned(),
            });
        }

        let pos_embed = trunc_normal_param([1, config.num_frames, config.hidden_dim], rng, device)?;
        let mut blocks = Vec::with_capacity(config.depth);
        for _ in 0..config.depth {
            blocks.push(ConditionalBlock::init(&config, rng, device)?);
        }
        let norm = LayerNormConfig::new(config.hidden_dim)
            .with_epsilon(DEFAULT_LAYER_NORM_EPSILON)
            .init(device);
        let dropout = DropoutConfig::new(config.emb_dropout).init();

        Ok(Self {
            pos_embed,
            dropout,
            blocks,
            norm,
            config,
        })
    }

    /// Run the predictor.
    ///
    /// # Shape
    ///
    /// - `tokens`: `(B, T, D)`
    /// - `actions`: `(B, T, E_a)`
    /// - output: `(B, T, D)`
    ///
    /// # Errors
    ///
    /// Returns [`LewmCoreError::SequenceTooLong`] when `T > num_frames`, or
    /// [`LewmCoreError::InvalidShape`] when tensor dimensions do not match the
    /// predictor config.
    pub fn forward(
        &self,
        tokens: Tensor<B, 3>,
        actions: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>, LewmCoreError> {
        self.validate_forward_shapes(&tokens, &actions)?;

        let [batch_size, seq_len, hidden_dim] = tokens.dims();
        let pos = self
            .pos_embed
            .val()
            .slice([0..1, 0..seq_len, 0..hidden_dim])
            .expand([batch_size, seq_len, hidden_dim]);
        let mut tokens = self.dropout.forward(tokens + pos);

        for block in &self.blocks {
            tokens = block.forward(tokens, actions.clone())?;
        }
        drop(actions);

        Ok(self.norm.forward(tokens))
    }

    /// Return the configured maximum sequence length.
    pub fn num_frames(&self) -> usize {
        self.config.num_frames
    }

    fn validate_forward_shapes(
        &self,
        tokens: &Tensor<B, 3>,
        actions: &Tensor<B, 3>,
    ) -> Result<(), LewmCoreError> {
        let [batch_size, seq_len, hidden_dim] = tokens.dims();
        let [action_batch, action_seq_len, action_dim] = actions.dims();
        let config = &self.config;

        if seq_len == 0 {
            return Err(LewmCoreError::InvalidShape {
                expected: vec![batch_size, 1, config.hidden_dim],
                found: vec![batch_size, seq_len, hidden_dim],
            });
        }

        if seq_len > config.num_frames {
            return Err(LewmCoreError::SequenceTooLong {
                got: seq_len,
                max: config.num_frames,
            });
        }

        if hidden_dim != config.input_dim {
            return Err(LewmCoreError::InvalidShape {
                expected: vec![batch_size, seq_len, config.input_dim],
                found: vec![batch_size, seq_len, hidden_dim],
            });
        }

        if action_batch != batch_size
            || action_seq_len != seq_len
            || action_dim != config.action_emb_dim
        {
            return Err(LewmCoreError::InvalidShape {
                expected: vec![batch_size, seq_len, config.action_emb_dim],
                found: vec![action_batch, action_seq_len, action_dim],
            });
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct AffineFreeLayerNorm {
    epsilon: f64,
}

impl AffineFreeLayerNorm {
    fn new(_hidden_dim: usize) -> Self {
        Self {
            epsilon: AFFINE_FREE_LAYER_NORM_EPSILON,
        }
    }

    fn forward<B: Backend>(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let (var, mean) = input.clone().var_mean_bias(2);
        input.sub(mean).div(var.add_scalar(self.epsilon).sqrt())
    }
}

#[derive(burn::module::Module, Debug)]
struct CausalSelfAttention<B: Backend> {
    norm: LayerNorm<B>,
    qkv: Linear<B>,
    proj: Linear<B>,
    attn_drop: Dropout,
    proj_drop: Dropout,
    num_heads: usize,
    head_dim: usize,
    inner_dim: usize,
    scale: f64,
}

impl<B: Backend> CausalSelfAttention<B> {
    fn init(
        config: &PredictorConfig,
        rng: &mut ModelInitRng,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        let inner_dim =
            config
                .attention_inner_dim()
                .ok_or_else(|| LewmCoreError::ConstructionFailed {
                    reason: "predictor heads * dim_head overflowed usize".to_owned(),
                })?;
        let qkv_dim =
            inner_dim
                .checked_mul(3)
                .ok_or_else(|| LewmCoreError::ConstructionFailed {
                    reason: "predictor qkv inner dimension overflowed usize".to_owned(),
                })?;
        let head_dim_u32 =
            u32::try_from(config.dim_head).map_err(|err| LewmCoreError::ConstructionFailed {
                reason: format!("predictor dim_head does not fit u32: {err}"),
            })?;
        let mut qkv = linear_zeros(config.hidden_dim, qkv_dim, device);
        qkv.weight = trunc_normal_param([config.hidden_dim, qkv_dim], rng, device)?;
        qkv.bias = None;

        let mut proj = linear_zeros(inner_dim, config.hidden_dim, device);
        proj.weight = trunc_normal_param([inner_dim, config.hidden_dim], rng, device)?;
        proj.bias = Some(zeros_param([config.hidden_dim], device)?);

        Ok(Self {
            norm: layer_norm(config.hidden_dim, device),
            qkv,
            proj,
            attn_drop: DropoutConfig::new(config.dropout).init(),
            proj_drop: DropoutConfig::new(config.dropout).init(),
            num_heads: config.heads,
            head_dim: config.dim_head,
            inner_dim,
            scale: 1.0 / f64::from(head_dim_u32).sqrt(),
        })
    }

    fn forward(&self, tokens: Tensor<B, 3>) -> Result<Tensor<B, 3>, LewmCoreError> {
        let tokens = self.norm.forward(tokens);
        let device = tokens.device();
        let [batch_size, seq_len, _hidden_dim] = tokens.dims();
        let qkv = self
            .qkv
            .forward(tokens)
            .reshape([batch_size, seq_len, 3, self.num_heads, self.head_dim])
            .permute([2, 0, 3, 1, 4]);
        let mut chunks = qkv.chunk(3, 0);
        let values = chunks.remove(2).squeeze_dim::<4>(0);
        let keys = chunks.remove(1).squeeze_dim::<4>(0);
        let queries = chunks.remove(0).squeeze_dim::<4>(0);

        let scores = queries.matmul(keys.swap_dims(2, 3)).mul_scalar(self.scale)
            + causal_mask_tensor(seq_len, &device)?;
        let attention = self.attn_drop.forward(softmax(scores, 3));
        let attended =
            attention
                .matmul(values)
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, self.inner_dim]);

        Ok(self.proj_drop.forward(self.proj.forward(attended)))
    }
}

#[derive(burn::module::Module, Debug)]
struct PredictorMlpBlock<B: Backend> {
    norm: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    drop: Dropout,
}

impl<B: Backend> PredictorMlpBlock<B> {
    fn init(
        config: &PredictorConfig,
        rng: &mut ModelInitRng,
        device: &B::Device,
    ) -> Result<Self, LewmCoreError> {
        let mut fc1 = linear_zeros(config.hidden_dim, config.mlp_dim, device);
        fc1.weight = trunc_normal_param([config.hidden_dim, config.mlp_dim], rng, device)?;
        fc1.bias = Some(zeros_param([config.mlp_dim], device)?);

        let mut fc2 = linear_zeros(config.mlp_dim, config.hidden_dim, device);
        fc2.weight = trunc_normal_param([config.mlp_dim, config.hidden_dim], rng, device)?;
        fc2.bias = Some(zeros_param([config.hidden_dim], device)?);

        Ok(Self {
            norm: layer_norm(config.hidden_dim, device),
            fc1,
            fc2,
            drop: DropoutConfig::new(config.dropout).init(),
        })
    }

    fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        self.fc2.forward(
            self.drop
                .forward(gelu(self.fc1.forward(self.norm.forward(tokens)))),
        )
    }
}

fn modulate<B: Backend>(
    normalized: Tensor<B, 3>,
    shift: Tensor<B, 3>,
    scale: Tensor<B, 3>,
) -> Tensor<B, 3> {
    normalized * scale.add_scalar(1.0) + shift
}

fn causal_mask_tensor<B: Backend>(
    seq_len: usize,
    device: &B::Device,
) -> Result<Tensor<B, 4>, LewmCoreError> {
    let device_key = DeviceKey::new(format!("{device:?}"))?;
    let mask = build_causal_mask(seq_len, &device_key)?;
    let data = TensorData::new(mask.values().to_vec(), [seq_len, seq_len]);

    Ok(Tensor::<B, 2>::from_data(data, device).reshape([1, 1, seq_len, seq_len]))
}

fn linear_zeros<B: Backend>(d_input: usize, d_output: usize, device: &B::Device) -> Linear<B> {
    LinearConfig::new(d_input, d_output)
        .with_initializer(Initializer::Zeros)
        .init(device)
}

fn layer_norm<B: Backend>(hidden_dim: usize, device: &B::Device) -> LayerNorm<B> {
    LayerNormConfig::new(hidden_dim)
        .with_epsilon(DEFAULT_LAYER_NORM_EPSILON)
        .init(device)
}

#[cfg(test)]
mod tests {
    use burn::module::Param;
    use burn_ndarray::NdArrayDevice;

    use super::*;

    type CpuBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn conditional_block_adaln_zero_identity() {
        let device = NdArrayDevice::default();
        let config = compact_config(3);
        let mut rng = model_init_rng(7).expect("valid model init seed");
        let block = ConditionalBlock::<CpuBackend>::init(&config, &mut rng, &device)
            .expect("compact conditional block should initialize");
        let tokens = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.1, 0.2, 0.3, //
                    1.0, 1.1, 1.2, 1.3, //
                    2.0, 2.1, 2.2, 2.3,
                ],
                [1, 3, 4],
            ),
            &device,
        );
        let actions = Tensor::<CpuBackend, 3>::ones([1, 3, 5], &device);

        let output = block
            .forward(tokens.clone(), actions)
            .expect("identity forward should not fail");

        assert_close(&output, &tokens, 1.0e-6);
    }

    #[test]
    fn predictor_causal_mask_blocks_future() {
        let device = NdArrayDevice::default();
        let config = PredictorConfig {
            num_frames: 2,
            depth: 1,
            heads: 1,
            mlp_dim: 1,
            dim_head: 1,
            input_dim: 3,
            hidden_dim: 3,
            output_dim: 3,
            action_emb_dim: 1,
            dropout: 0.0,
            emb_dropout: 0.0,
        };
        let mut rng = model_init_rng(11).expect("valid model init seed");
        let mut attn = CausalSelfAttention::<CpuBackend>::init(&config, &mut rng, &device)
            .expect("compact attention should initialize");
        attn.qkv.weight = Param::from_data(
            TensorData::new(
                vec![
                    1.0, 1.0, 0.0, //
                    0.0, 0.0, 1.0, //
                    0.0, 0.0, 0.0,
                ],
                [3, 3],
            ),
            &device,
        );
        attn.proj.weight = Param::from_data(TensorData::new(vec![1.0, 0.0, 0.0], [1, 3]), &device);
        attn.proj.bias = Some(zeros_param([3], &device).expect("zero projection bias"));

        let baseline = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, 1.0, //
                    0.0, 1.0, 0.0,
                ],
                [1, 2, 3],
            ),
            &device,
        );
        let changed_future = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, 1.0, //
                    1.0, 0.0, 0.0,
                ],
                [1, 2, 3],
            ),
            &device,
        );

        let baseline = attn
            .forward(baseline)
            .expect("causal attention should run")
            .to_data()
            .to_vec::<f32>()
            .expect("f32 attention output");
        let changed_future = attn
            .forward(changed_future)
            .expect("causal attention should run")
            .to_data()
            .to_vec::<f32>()
            .expect("f32 attention output");

        assert!((baseline[0] - changed_future[0]).abs() <= 1.0e-6);
        assert!((baseline[1] - changed_future[1]).abs() <= 1.0e-6);
        assert!((baseline[2] - changed_future[2]).abs() <= 1.0e-6);
        assert!((baseline[3] - changed_future[3]).abs() > 1.0e-3);
    }

    #[test]
    fn predictor_errors_when_sequence_exceeds_position_embedding() {
        let device = NdArrayDevice::default();
        let predictor = ArPredictor::<CpuBackend>::init(compact_config(2), &device)
            .expect("compact predictor should initialize");
        let tokens = Tensor::<CpuBackend, 3>::zeros([1, 3, 4], &device);
        let actions = Tensor::<CpuBackend, 3>::zeros([1, 3, 5], &device);

        let err = predictor
            .forward(tokens, actions)
            .expect_err("sequence longer than num_frames should fail");

        assert_eq!(err, LewmCoreError::SequenceTooLong { got: 3, max: 2 });
    }

    fn compact_config(num_frames: usize) -> PredictorConfig {
        PredictorConfig {
            num_frames,
            depth: 2,
            heads: 2,
            mlp_dim: 8,
            dim_head: 2,
            input_dim: 4,
            hidden_dim: 4,
            output_dim: 4,
            action_emb_dim: 5,
            dropout: 0.0,
            emb_dropout: 0.0,
        }
    }

    fn assert_close(left: &Tensor<CpuBackend, 3>, right: &Tensor<CpuBackend, 3>, tolerance: f32) {
        let left = left
            .clone()
            .to_data()
            .to_vec::<f32>()
            .expect("f32 left tensor");
        let right = right
            .clone()
            .to_data()
            .to_vec::<f32>()
            .expect("f32 right tensor");
        assert_eq!(left.len(), right.len());
        for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
            assert!(
                (left - right).abs() <= tolerance,
                "tensor mismatch at {index}: left={left} right={right}"
            );
        }
    }
}
