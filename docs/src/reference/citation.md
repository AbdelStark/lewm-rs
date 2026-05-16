# How to cite

## Cite the upstream

If you use the model, the algorithm, or the architecture, cite Maes
et al.'s paper first. `lewm-rs` makes no algorithmic contribution
beyond the reproduction.

```bibtex
@misc{maes2026leworldmodel,
  title         = {Learning World Models in Latent Space},
  author        = {Maes, Lucas and Le Lidec, Quentin and Scieur, Damien
                   and Balestriero, Randall and LeCun, Yann},
  year          = {2026},
  eprint        = {2502.16560},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2502.16560}
}
```

## Cite this implementation

If you use the lewm-rs Rust port, the parity contracts, the
Tract inference path, the SO-100 reproduction, or these docs:

```bibtex
@software{lewm_rs_2026,
  title  = {lewm-rs: Rust reproduction and extension of LeWorldModel},
  author = {Abdel},
  year   = {2026},
  url    = {https://github.com/AbdelStark/lewm-rs}
}
```

## Cite the framework dependencies

Where appropriate, also cite the runtime / framework stack:

```bibtex
@software{burn_framework_2026,
  title  = {Burn: A Deep Learning Framework in Rust},
  author = {{Tracel AI}},
  year   = {2026},
  url    = {https://github.com/tracel-ai/burn}
}

@software{tract_2026,
  title  = {Tract: Practical Neural Network Inference in Rust},
  author = {{Sonos}},
  year   = {2026},
  url    = {https://github.com/sonos/tract}
}
```
