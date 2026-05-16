# Bibliography

A single canonical list of every primary reference cited across the
docs. Each entry includes a short note on what the work contributes
to `lewm-rs`'s context.

## JEPA and LeWorldModel

- **Maes, L., Le Lidec, Q., Scieur, D., Balestriero, R., LeCun, Y.**
  (2026). *LeWorldModel: Learning World Models in Latent Space.*
  arXiv:2502.16560.
  *The paper this repository reproduces. The architecture, losses,
  and training procedure follow Maes et al. without modification.*

- **LeCun, Y.** (2022). *A Path Towards Autonomous Machine
  Intelligence.* OpenReview.
  *The original argument for JEPA as a path to non-generative world
  models. Sets up the conceptual frame that LeWM operationalises.*

- **Assran, M., Duval, Q., Misra, I., et al.** (2023). *Self-Supervised
  Learning from Images with a Joint-Embedding Predictive Architecture.*
  CVPR. — *I-JEPA, the image-only ancestor.*

- **Bardes, A., Garrido, Q., Ponce, J., et al.** (2024). *V-JEPA:
  Latent Video Prediction for Visual Representation Learning.*
  arXiv:2404.08471. — *V-JEPA, the video extension; closest cousin
  of LeWM in the literature.*

## Architectural primitives

- **Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al.** (2021).
  *An Image is Worth 16×16 Words: Transformers for Image Recognition at
  Scale.* ICLR. — *The ViT.*

- **Touvron, H., Cord, M., Douze, M., et al.** (2021). *Training
  data-efficient image transformers & distillation through attention.*
  ICML. — *DeiT-Tiny convention, which sets the ViT-Tiny size LeWM uses.*

- **Peebles, W., Xie, S.** (2023). *Scalable Diffusion Models with
  Transformers* (DiT). ICCV. — *The source of AdaLN-zero; the
  initialisation trick LeWM borrows for its action-conditioned
  predictor.*

- **Perez, E., Strub, F., De Vries, H., et al.** (2018). *FiLM:
  Visual Reasoning with a General Conditioning Layer.* AAAI. — *The
  ancestor of AdaLN: feature-wise linear modulation.*

- **Xiong, R., Yang, Y., He, D., et al.** (2020). *On Layer
  Normalization in the Transformer Architecture.* ICML. — *Pre-norm
  vs post-norm; LeWM's encoder is pre-norm.*

- **Hendrycks, D., Gimpel, K.** (2016). *Gaussian Error Linear Units
  (GELUs).* arXiv:1606.08415. — *The activation function; LeWM uses
  the exact erf form, not the tanh approximation.*

## Optimization

- **Loshchilov, I., Hutter, F.** (2019). *Decoupled Weight Decay
  Regularization* (AdamW). ICLR. — *The optimizer LeWM uses, with the
  decay / no-decay parameter split.*

- **Brown, T., et al.** (2020). *Language Models are Few-Shot Learners.*
  NeurIPS. — *Popularised the $\beta_2 = 0.95$ transformer convention
  inherited by LeWM.*

## Mixed precision and numerical engineering

- **Micikevicius, P., Narang, S., Alben, J., et al.** (2018).
  *Mixed Precision Training.* ICLR. — *The "AMP" recipe LeWM uses:
  BF16 for compute, F32 for normalisation, reduction, and the
  optimizer.*

## SIGReg

- **Epps, T. W., Pulley, L. B.** (1983). *A test for normality based
  on the empirical characteristic function.* Biometrika 70(3): 723–726.
  — *The univariate Gaussianity test SIGReg is built on.*

- **Cramér, H., Wold, H.** (1936). *Some theorems on distribution
  functions.* J. London Math. Soc. 11: 290–294. — *The
  Cramér–Wold theorem that motivates the random-projection sketch.*

## Planning

- **de Boer, P.-T., Kroese, D. P., Mannor, S., Rubinstein, R. Y.**
  (2005). *A Tutorial on the Cross-Entropy Method.* Annals of
  Operations Research, 134(1): 19–67. — *CEM, the planner LeWM uses.*

- **Wang, T., Ba, J.** (2020). *Exploring Model-based Planning with
  Policy Networks.* ICLR. — *CEM with a learned dynamics model in
  continuous control.*

- **Hansen, N., Wang, X.** (2022). *Temporal Difference Learning for
  Model Predictive Control* (TD-MPC). — *Closest cousin of LeWM's
  planning setup.*

- **Hansen, N., Su, H., Wang, X.** (2023). *TD-MPC2: Scalable, Robust
  World Models for Continuous Control.* arXiv:2310.16828.

## World models in robotics

- **Ha, D., Schmidhuber, J.** (2018). *World Models.* NeurIPS. — *The
  modern reference framing for world-model-based control.*

- **Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T.** (2023).
  *DreamerV3: Mastering diverse domains through world models.* — *The
  generative latent world-model alternative LeWM reads against.*

- **van den Oord, A., Kalchbrenner, N., Vinyals, O., et al.** (2016).
  *Conditional Image Generation with PixelCNN Decoders.* NeurIPS. —
  *Representative pixel-prediction world model; LeWM avoids this class.*

## Rust ML stack

- **Tracel.ai** (2024–2026). *Burn: A Deep Learning Framework in Rust.*
  <https://github.com/tracel-ai/burn>. Pinned at `= 0.20.1`.

- **Sonos** (2019–2026). *Tract: Practical Neural Network Inference
  in Rust.* <https://github.com/sonos/tract>. Pinned at `= 0.22.1`.

- **Hugging Face** (2024–2026). *lerobot: State-of-the-art Machine
  Learning for Real-World Robotics.* <https://github.com/huggingface/lerobot>.
  *Source of the SO-100 raw dataset.*

## Upstream reference implementation

- **Maes, L.** (2026). *le-wm.* <https://github.com/lucas-maes/le-wm>.
  *The PyTorch reference implementation lewm-rs reproduces, line
  for line where parity demanded.*
