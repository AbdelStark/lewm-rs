---
name: parity-testing
description: Reference parity is the single highest-value local check in this project. Activate whenever editing numerical code in `lewm-core` (encoder, predictor, AdaLN, MLP heads, action embedder, losses, JEPA wrapper, tensor ops), when running or interpreting parity tests, when L∞/RMSE tolerances appear in a diff, or when CI's `parity` job fails. The contract is RFC 0008 v1.1.0 — 10 activation-level tests must pass with L∞ < 1e-4 (RMSE < 1e-3 for SIGReg).
prerequisites: `HF_TOKEN` for downloading dumps from `AbdelStark/lewm-rs-parity-dumps` (CI sets this); otherwise tests skip gracefully
---

# Reference Parity Testing

<purpose>
Numerical drift in encoder/predictor/AdaLN/MLP/embedder/SIGReg silently breaks reproduction. This skill defines the parity discipline: when to run tests, how to interpret results, and what to do if tolerances regress.
</purpose>

<context>
- Reference: `quentinll/lewm-pusht` PyTorch checkpoint at the pinned SHA in `specs/rfcs/0008-…md`.
- Reference dumps: per-layer activation Safetensors at `AbdelStark/lewm-rs-parity-dumps` (HF Hub).
- 10 parity tests under `crates/lewm-core/tests/`:
  - `parity_encoder.rs`         (ViT-Tiny encoder, CLS output)
  - `parity_action_encoder.rs`  (action embedder MLP + Conv1d-k1 smoother)
  - `parity_predictor.rs`       (autoregressive AdaLN-zero predictor)
  - `parity_pred_proj.rs`       (prediction projector head)
  - `parity_sigreg.rs`          (SIGReg loss value)
  - plus `parity_init.rs`, `parity_fixture.rs`, and shape-only siblings (`*_shape.rs`)
- Tolerances per RFC 0008 / glossary §2.3:
  - `|y_rust − y_torch|_∞ ≤ 1e-4` for encoder, action_encoder, predictor, pred_proj (F32)
  - `|Δ| < 1e-3` for SIGReg loss value
- Activation: behind Cargo feature `parity-fixtures` AND env var `LEWM_PARITY_DUMPS` (path to dumps dir) and `LEWM_REFERENCE_SAFETENSORS` (path to converted weights). Tests skip cleanly when env vars are unset, so `make check` is green on minimal hosts.
- LayerNorm `eps = 1e-12`. GELU is the EXACT-erf variant (not tanh approximation). These two were the bugs in PR #217; do not "round" or substitute.
- CI: `.github/workflows/ci.yml` `parity` job downloads dumps when `HF_TOKEN` is available; falls back to shape-only otherwise.
</context>

<procedure>
1. **Decide if parity is required for this change.** It IS required if you touched:
   `crates/lewm-core/src/{vit,predictor,ada_ln,mlp,embedder,losses,jepa,tensor_ops,init,import}.rs`
   or anything that changes the order of float operations on the encoder/predictor forward path.

2. **Run shape tests first** (they don't need dumps):
   ```
   cargo test -p lewm-core --features parity-fixtures \
     vit_shape ada_ln_shape mlp_shape embedder_shape predictor_shape
   ```

3. **Acquire dumps**, then run full numerical parity:
   ```
   # If you have HF_TOKEN; otherwise CI does this for you on PR.
   hf download AbdelStark/lewm-rs-parity-dumps --local-dir /tmp/parity-dumps
   hf download quentinll/lewm-pusht --local-dir /tmp/pusht-ref
   export LEWM_PARITY_DUMPS=/tmp/parity-dumps
   export LEWM_REFERENCE_SAFETENSORS=/tmp/pusht-ref/model.safetensors
   cargo test -p lewm-core --features parity-fixtures parity_ -- --nocapture
   ```

4. **Interpret failures**: each test prints the per-stage L∞ and RMSE.
   - L∞ > 1e-4 → parity broken. STOP. Bisect with `git bisect` against the test name.
   - Small drift (1e-5 → 5e-5) is acceptable across builds; do not chase it.
   - If your change is intentional (e.g., new ADR superseding RFC 0008), the change is **gated** — pause and ask the human, then open an ADR before adjusting the constants.

5. **Never** loosen tolerances or `#[ignore]` a parity test to "make it pass." If a parity test is wrong, fix the test in a dedicated PR that documents why (with reference-side evidence).

6. **Regenerate dumps** (rare, ADR-territory only): see RFC 0008 §4.2 and `python/convert_reference.py dump`. New dumps must be uploaded to `AbdelStark/lewm-rs-parity-dumps` with a new fixture hash; the CI cache key is fixture-hash-based.
</procedure>

<patterns>
<do>
— Use `LayerNorm(eps = 1e-12)` everywhere — the reference uses this and tighter eps is required for L∞ < 1e-4.
— Use exact-erf GELU (`x * 0.5 * (1 + erf(x / sqrt(2)))`). The tanh approximation breaks parity.
— When importing reference weights, go through `lewm_core::import` (the parity-preserving Safetensors loader). It applies the name map in `python/param_name_map.py` (303 source tensors → Burn names).
— When exporting Burn modules, go through `lewm_core::export::to_safetensors` for deterministic byte-stable output.
</do>
<dont>
— Don't introduce `tanh`-approximation GELU or `eps = 1e-5` LayerNorm "for performance." That's a parity-breaking optimization.
— Don't change parameter init order in `crates/lewm-core/src/init.rs` — RFC 0013 pins it.
— Don't compose `Module::forward` in a way that reorders sums on the hot path; parity depends on summation order matching the reference (within F32 nondeterminism).
</dont>
</patterns>

<examples>
Example failure session:
```
$ cargo test -p lewm-core --features parity-fixtures parity_encoder -- --nocapture
…
encoder stage `post_layernorm`: L∞ = 2.3e-3  (limit 1e-4) — FAIL
```
This means the encoder's final-block LayerNorm drifted. Likely cause: someone changed `eps` or the activation order. `git log -p -- crates/lewm-core/src/vit.rs` since the last green run, look for `eps`, `gelu`, or `LayerNorm` edits. Revert / fix, then re-run.
</examples>

<troubleshooting>
| Symptom                                                       | Cause                                          | Fix                                                              |
|---------------------------------------------------------------|------------------------------------------------|------------------------------------------------------------------|
| All parity tests "ignored" with no error                       | `LEWM_PARITY_DUMPS` / `LEWM_REFERENCE_SAFETENSORS` unset | Set env vars after `hf download` (step 3)                       |
| `parity_init.rs` mismatches parameter count                    | Module surgery changed parameter shape          | Update RFC 0002 §X, regenerate dumps, open ADR                   |
| `parity_sigreg.rs` fails by ~1e-3 with rest passing           | SIGReg knot grid / RNG sub-stream drift          | Check `lewm-core::rng` substream key; do not change knot count  |
| Encoder parity passes but predictor fails by ~5e-4            | AdaLN-zero conditioner not zero-initialized      | Final adaLN linear weight must be all-zero at init               |
| CI parity job skipped on PR                                    | `HF_TOKEN` not available on fork                 | Expected; maintainer-driven CI run will pick it up                |
</troubleshooting>

<references>
- `specs/rfcs/0008-reference-parity-testing.md` — full contract
- `specs/rfcs/0002-core-model-architecture.md` — module shapes
- `specs/rfcs/0003-sigreg-and-loss-functions.md` — SIGReg constants
- `crates/lewm-core/tests/parity_*.rs` — actual harnesses
- `crates/lewm-core/src/import.rs` — Safetensors → `Jepa<B>` loader
- `python/param_name_map.py`, `python/convert_reference.py`, `python/verify_conversion.py`
</references>
