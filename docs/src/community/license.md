# License

| Artifact | License |
|----------|---------|
| Code | MIT (file: [`LICENSE`](https://github.com/AbdelStark/lewm-rs/blob/main/LICENSE)) |
| Trained checkpoints | Apache-2.0 (intended; see model cards on the Hub repos) |
| Paper writeup | CC-BY-4.0 (intended; `paper/lewm-rs.md`) |
| Documentation site (this site) | CC-BY-4.0 |

## Why this combination

- **MIT code.** Permissive, compatible with everything, low friction.
- **Apache-2.0 checkpoints.** Patent grant, which matters more for
  trained model weights than for source code.
- **CC-BY-4.0 prose.** The right license for written work: attribution
  required, derivatives allowed.

## Third-party

- **Burn** (Tracel.ai) — Apache-2.0 / MIT (dual).
- **Tract** (Sonos) — Apache-2.0 / MIT (dual).
- **safetensors** (HF) — Apache-2.0.
- **PyTorch** (Meta) — BSD-3.

See `cargo deny check` and the workspace `deny.toml` for the full
dependency license inventory.
