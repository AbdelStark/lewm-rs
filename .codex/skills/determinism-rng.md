---
name: determinism-rng
description: RNG, seed, and reproducibility discipline per RFC 0013 v1.0.0. Activate when adding RNG, modifying init order, dealing with `scripts/check_nondet.py` failures, working on checkpoint resume, or chasing nondeterministic test flakes. The project enforces a named sub-stream tree from a single global seed and bans OS randomness in source code.
prerequisites: Familiarity with `crates/lewm-core/src/rng.rs`
---

# Determinism & RNG

<purpose>
"Same seed, same numbers" is a contract this project takes seriously on CPU (bitwise) and statistically on GPU. Determinism is a stack — RNG sources, init order, summation, dataloader scheduling, cublas/cudnn flags, and the reproducible-build pipeline all participate. This skill is the working knowledge for staying on-contract.
</purpose>

<context>
- Global seed: a 64-bit integer surfaced through configs and config-validation. Seed propagation lives in `lewm_core::rng`.
- Sub-stream tree: deterministic derivation from the global seed, keyed by a stable string (e.g., `"init.vit.patch_embed"`, `"train.dataloader.shuffle"`). New consumers MUST register their key in code, not magic-number it.
- Allowed RNG: `rand_chacha::ChaCha20Rng` seeded from a sub-stream. Anything else is suspect.
- Banned: `rand::thread_rng`, `rand::random()`, OS-randomness, ad-hoc seeds inside library functions.
- `scripts/check_nondet.py` greps Rust sources for forbidden patterns. It runs inside `make check`. Inline opt-out is `// determinism-lint: allow <rationale>` and is reserved for non-numerical paths (e.g., a CLI banner).
- Determinism levels (RFC 0013 §3): Strict (bitwise on CPU/Burn-NdArray), Strong (bitwise within a fixed cublas workspace), Statistical (training loss curves within tolerance), Best-effort (multi-GPU — out of scope).
- Checkpoint contract: AdamW state, RNG state, step counter, config hash, and seed are all serialized; resume validates each before continuing.
</context>

<procedure>
1. **Need randomness?** Take a `&mut rand_chacha::ChaCha20Rng` parameter (or, at the entry point, a `lewm_core::rng::SubStreamKey`). Never seed inside the function.

2. **Adding a new RNG consumer**:
   - Pick a stable, descriptive key. Convention: dotted scope, e.g. `"data.pusht.augmentation"`.
   - Register the key in `lewm_core::rng` (see existing `SubStream` constants).
   - Document the key in the relevant RFC's "RNG keys" section.

3. **Init code**: route through `lewm_core::init` helpers. They consume a sub-stream RNG and apply the RFC-0013-specified distribution.

4. **Iteration order**: do NOT iterate `HashMap`s or `HashSet`s in code paths whose outputs are part of the parity / training contract. Use `BTreeMap` / `BTreeSet` or sort an `IndexMap`. (`check_nondet.py` flags the highest-risk cases; reviewers catch the rest.)

5. **Floats**: avoid intentional re-ordering of reductions. If a refactor reorders a sum, run `parity-testing.md`.

6. **Checkpoint changes**: if you alter the checkpoint payload, update both:
   - `crates/lewm-train/src/checkpoint.rs` (write path)
   - `crates/lewm-train/src/resume.rs` (read + validate path, including config-hash and RNG-state checks)
   And bump the checkpoint version constant.

7. **Verify**:
   ```
   cargo test -p lewm-train resume -- --nocapture
   python3 scripts/check_nondet.py
   ```
</procedure>

<patterns>
<do>
— `fn build_model<B: Backend>(cfg: &Config, rng: &mut ChaCha20Rng) -> Jepa<B>` — RNG passed in.
— Use `SubStream::derive("init.predictor.block.0")` at the entry point to get the sub-stream once, then thread it through.
— For dataloader shuffling, derive a per-epoch sub-stream: `epoch_rng = base.derive(&format!("epoch.{epoch}"))`.
— When adding a new key, also add a unit test that asserts repeatability (same key + global seed → same bytes).
</do>
<dont>
— `let mut rng = rand::thread_rng();` — banned by `check_nondet.py`.
— `let rng = ChaCha20Rng::from_seed([0; 32]);` inside library code — seeding belongs at the entry point.
— `for (k, v) in some_hashmap` where the iteration affects parameters' init values or checkpoint contents.
— `f32::EPSILON` substitutions for documented constants like LayerNorm's `1e-12` — the constants are part of the parity contract.
</dont>
</patterns>

<examples>
Good pattern, threaded RNG:

```rust
use rand_chacha::ChaCha20Rng;

pub fn init_predictor<B: Backend>(
    cfg: &PredictorConfig,
    rng: &mut ChaCha20Rng,
) -> ArPredictor<B> {
    let block_rng = lewm_core::rng::derive(rng, "init.predictor.block");
    ArPredictor {
        blocks: (0..cfg.num_layers)
            .map(|i| ConditionalBlock::init(
                cfg,
                &mut lewm_core::rng::derive(&block_rng, &format!("layer.{i}")),
            ))
            .collect(),
    }
}
```

Bad pattern, banned:

```rust
// determinism-lint would (and check_nondet.py will) flag this:
let mut rng = rand::thread_rng();
let init_value = rng.gen::<f32>();
```
</examples>

<troubleshooting>
| Symptom                                                          | Cause                                                       | Fix                                                                                |
|------------------------------------------------------------------|-------------------------------------------------------------|------------------------------------------------------------------------------------|
| `check_nondet.py: thread_rng — use an RFC 0013 ChaCha20Rng …`     | OS RNG call in source                                       | Replace with sub-stream `ChaCha20Rng`; do not silence the lint                     |
| Test passes locally, flakes in CI                                 | Hidden `HashMap` iteration or non-seeded random              | Switch to `BTreeMap` or seed via sub-stream; add a regression test                  |
| Resume validation fails: "config hash mismatch"                   | Config changed between checkpoint and resume                 | Either recheckpoint after change, or skip resume; do not bypass the check          |
| Resume succeeds but loss curve diverges from pre-restart curve    | RNG state not captured / restored at the right step          | Audit `resume.rs`: RNG-state read must happen before the first dataloader pull     |
| Encoder parity passes on CPU, drifts on CUDA                       | cublas non-deterministic algo / workspace size mismatch      | Set cublas deterministic flag + workspace per RFC 0013 §4; document in PR          |
</troubleshooting>

<references>
- `specs/rfcs/0013-determinism-and-reproducibility.md` — the contract
- `crates/lewm-core/src/rng.rs` — `SubStream`, key registry
- `crates/lewm-core/src/init.rs` — RNG-consuming init helpers
- `crates/lewm-train/src/checkpoint.rs` & `resume.rs` — serialization + validation
- `scripts/check_nondet.py` — enforcement
</references>
