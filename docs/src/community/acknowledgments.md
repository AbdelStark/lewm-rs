# Acknowledgments

This project builds on the work of many.

## The algorithm

- **Lucas Maes, Quentin Le Lidec, Damien Scieur, Randall Balestriero,
  and Yann LeCun** for the LeWorldModel paper and design.
- **Lucas Maes** for the upstream PyTorch reference implementation
  ([`lucas-maes/le-wm`](https://github.com/lucas-maes/le-wm)), which
  this project line-for-line reproduces in Rust where parity demands.

## The framework stack

- **Tracel.ai** for the [Burn](https://github.com/tracel-ai/burn)
  deep learning framework — the substrate on which everything in this
  repo runs.
- **Sonos** for [Tract](https://github.com/sonos/tract) — the
  pure-Rust ONNX runtime that makes the deployment story work.
- **Hugging Face** for the [`lerobot`](https://github.com/huggingface/lerobot)
  library and the SO-100 dataset, and for the Hub / Jobs / Spaces
  platform.

## The conceptual stack

- **Yann LeCun** for the original JEPA argument.
- **Mahmoud Assran et al.** for I-JEPA, the image-only ancestor.
- **William Peebles and Saining Xie** for AdaLN-zero, the
  initialisation trick that makes the predictor stable.

## The numerical primitives

- **T. W. Epps and L. B. Pulley** for the empirical-characteristic-function
  goodness-of-fit test on which SIGReg is built (1983).
- **Cramér and Wold** for the theorem that justifies the random-projection
  sketch (1936).
- **Reuven Y. Rubinstein** for the Cross-Entropy Method (1990s onward).

## The supply chain

- **The Rust foundation** for the language and toolchain.
- **The cargo ecosystem** for the dependency graph: `serde`, `tokio`,
  `safetensors`, `clap`, and several hundred crates whose authors
  do not appear individually here but whose work is essential.

## Mistakes are mine

Any error in the reproduction, the parity contracts, or the docs is
the author's. Corrections welcome — see
[Contributing](./contributing.md).
