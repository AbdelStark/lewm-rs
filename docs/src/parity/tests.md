# The 10-test parity harness

> **Motivation.** This page is the exhaustive list of what's tested,
> against what tolerance, and where the test lives.
>
> **Position.** Sub-page of [Part VI](./why-parity.md).
>
> **What you should leave with.** The complete parity contract.

## 1. The test list

All ten parity tests pass against `quentinll/lewm-pusht@22b330c`. They
live under `crates/lewm-core/tests/` and are gated behind the
`parity-fixtures` Cargo feature plus the `LEWM_PARITY_DUMPS` /
`LEWM_REFERENCE_SAFETENSORS` env vars.

| # | Test | Tolerance | Component | Test file |
|---|------|-----------|-----------|-----------|
| 1 | encoder_cls            | $L_\infty < 10^{-4}$ (TOL-001)        | `Vit::forward(...)[:, 0, :]` | `parity_encoder_cls.rs` |
| 2 | encoder_all_tokens     | $L_\infty < 10^{-4}$                  | `Vit::forward(...)` full (B, 257, 192) | `parity_encoder_all.rs` |
| 3 | encoder_mixed_precision| rel. $< 2\!\times\!10^{-2}$ (TOL-010) | Encoder in BF16 vs F32 | `parity_encoder_mixed_precision.rs` |
| 4 | action_encoder         | $L_\infty < 10^{-4}$                  | `Embedder::forward(actions)` | `parity_action_encoder.rs` |
| 5 | predictor              | $L_\infty < 10^{-4}$ (TOL-002)        | `ArPredictor::forward(history, action_emb)` | `parity_predictor.rs` |
| 6 | predictor_mixed_precision | rel. $< 2\!\times\!10^{-2}$        | Predictor in BF16 vs F32 | `parity_predictor_mixed_precision.rs` |
| 7 | pred_proj              | $L_\infty < 10^{-4}$                  | `Mlp::forward(predictor_out)` | `parity_pred_proj.rs` |
| 8 | projector              | $L_\infty < 10^{-4}$                  | `Mlp::forward(encoder_out)` | `parity_projector.rs` |
| 9 | sigreg_seeded          | $\lvert\Delta\rvert < 10^{-3}$ (TOL-003) | `sigreg_loss` with identical RNG seed | `parity_sigreg.rs` |
| 10 | sigreg_seedfree       | rel. $< 5\!\times\!10^{-2}$ (TOL-004)  | `sigreg_loss` with different RNG seed | `parity_sigreg_seedfree.rs` |

## 2. The test pattern

Every parity test follows the same pattern:

```rust,ignore
#[cfg(feature = "parity-fixtures")]
#[test]
fn parity_encoder_cls() -> anyhow::Result<()> {
    // 1. Load the locked reference model into Burn
    let device = NdArrayDevice::default();
    let jepa: Jepa<NdArray<f32>> = load_reference_safetensors(
        env!("LEWM_REFERENCE_SAFETENSORS"),
        &device,
    )?;

    // 2. Load the parity fixture (pixels, actions)
    let (pixels, _) = load_fixture("tests/fixtures/parity_fixture.safetensors", &device)?;

    // 3. Run the Burn forward
    let z = jepa.encode(pixels);                              // (1, 1, 192) for T=1
    let cls = z.narrow(1, 0, 1).squeeze(1);                    // (1, 192)

    // 4. Load the expected (PyTorch) activation dump
    let dumps_dir = env!("LEWM_PARITY_DUMPS");
    let expected = load_safetensors_tensor(
        format!("{dumps_dir}/encoder_cls.safetensors"),
        "cls",
        &device,
    )?;

    // 5. Compare
    let l_inf = (cls - expected).abs().max();
    assert!(l_inf.into_scalar() < 1e-4, "L_inf = {}", l_inf);
    Ok(())
}
```

The same skeleton applies to every test, varying only the forward to
call and the expected dump.

## 3. Running the harness

### 3.1 With HF token

```sh
export HF_TOKEN=...
python python/convert_reference.py --download   # fetch reference into ./refs/
python python/convert_reference.py dump --all   # produce dumps into ./dumps/
LEWM_REFERENCE_SAFETENSORS=./refs/pusht.safetensors \
LEWM_PARITY_DUMPS=./dumps \
cargo test -p lewm-core --features parity-fixtures
```

### 3.2 Without HF token (shape-only fallback)

```sh
cargo test -p lewm-core
```

Without the feature flag, the tests are compiled but `#[ignore]`'d.
Running them prints `skipped: no parity dumps available`. The shape
check still runs via the `cargo check` gate.

## 4. CI integration

The `parity` GitHub Actions workflow:

1. Caches the dumps directory on the fixture hash.
2. Downloads the reference checkpoint and dumps from
   `AbdelStark/lewm-rs-parity-dumps` when `HF_TOKEN` is available.
3. Runs `cargo test -p lewm-core --features parity-fixtures` with the
   env vars set.
4. If no token: runs shape-only checks (model loads, forward runs,
   shape matches expected dump shape).

## 5. Reproducing the dumps

Should the dumps ever need to be regenerated (e.g., after a reference
checkpoint update):

```sh
python python/convert_reference.py dump \
    --reference quentinll/lewm-pusht \
    --reference-revision 22b330c \
    --fixture tests/fixtures/parity_fixture.safetensors \
    --out dumps/
```

The script downloads the reference, runs the upstream PyTorch forward
on the fixture, captures activations at every checked depth, and
writes Safetensors files in `dumps/`. The script also writes a
`dumps_meta.json` recording the reference SHA256 and the script's
git SHA at dump time, so any drift is detectable.

## 6. Source pointers

| Topic | Source |
|-------|--------|
| Parity tests | `crates/lewm-core/tests/parity_*.rs` |
| Fixture | `tests/fixtures/parity_fixture.safetensors` |
| Fixture metadata | `tests/fixtures/reference_model.meta.json` |
| Dump producer | `python/convert_reference.py dump` |
| Reference download | `python/convert_reference.py --download` |
| Parity CI workflow | `.github/workflows/ci.yml` (parity job) |
