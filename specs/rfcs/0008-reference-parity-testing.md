---
rfc: "0008"
title: "Reference parity testing and weight import"
status: Accepted
version: 1.1.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-14
supersedes: []
superseded_by: null
tracks_prd: ["§4.4 risk 3", "§6.4 Stage L1", "§10 acceptance"]
depends_on: ["0001", "0002", "0003"]
related: ["0005", "0007", "0011", "0013"]
---

# RFC 0008 — Reference parity testing and weight import

> **Status:** Accepted · **Version:** 1.1.0
>
> The single highest-value local check before any cloud spend is parity: load `quentinll/lewm-pusht` weights into `lewm-core`, run the forward, and assert the output matches the PyTorch reference to a fixed tolerance. This RFC pins the harness end-to-end: how weights are converted, what fixture inputs are used, what tolerances apply, and how parity probes are wired into the per-epoch training loop and into CI.

---

## 1. Introduction

### 1.1 Motivation

A model that compiles is not a model that runs the paper's experiment. The gap between "Rust code that produces a tensor of the right shape" and "Rust code that reproduces the PyTorch reference within 1e-4 absolute" is exactly where 95 % of port bugs hide: wrong init order, wrong activation variant, wrong LayerNorm placement, wrong AdaLN convention, off-by-one slicing. Parity testing is the single most cost-effective way to catch these.

### 1.2 Goals

1. Specify the weight import pipeline (PyTorch `.pt` → Safetensors intermediate → Burn record).
2. Specify the parity fixture (input shape, seed, content).
3. Specify the per-layer parity dump procedure on the PyTorch side.
4. Specify the parity test harness on the Rust side, with the binding tolerances.
5. Specify the per-epoch parity probe used by the trainer.
6. Specify the CI gate that runs parity on every PR.

### 1.3 Non-goals

- Re-implementing the PyTorch reference. We trust upstream's PyTorch as ground truth.
- Quantitative comparison of training trajectories. That is a separate concern (RFC 0013).

---

## 2. Conventions

- "PyTorch reference" = `quentinll/lewm-pusht` checkpoint at a pinned revision SHA.
- "Reference dump" = a directory of `.safetensors` files containing per-layer activations from running the PyTorch model on the parity fixture input.
- "Parity probe" = a non-asserting in-loop comparison emitted to the per-epoch sidecar.
- "Parity test" = an asserting test in `lewm-core::tests::parity_*`.

---

## 3. Weight import pipeline

### 3.1 Overview

```
HF Hub repo:  quentinll/lewm-pusht/pytorch_model.bin           # original
                          │
                          ▼  python/convert_reference.py
                pytorch_model.bin → reference.safetensors      # canonicalized
                          │
                          ▼  python/convert_reference.py
                reference.safetensors → reference.mpk           # Burn record via burn-import
                          │
                          ▼  lewm-core::load_burn_record
                Jepa<NdArray<F32>>
```

The pipeline runs once, in Phase 1, by `python/convert_reference.py`. Output artifacts are committed to the `abdelstark/lewm-rs-pusht` model repo for downstream reproducibility (under the `parity/` subfolder).

### 3.2 `convert_reference.py`

```text
USAGE
    python convert_reference.py
        --pt <path-to-pytorch-bin>
        --safetensors-out <path>
        --burn-record-out <path>
        --jepa-config <path-to-config.toml>
        --intermediate-checks                       # emit per-layer dumps
        --dump-dir <path>                            # destination for per-layer activations
        --fixture-seed <int>                         # default 0
```

The script:

1. **Loads** the PyTorch `.bin` via `torch.load(map_location="cpu", weights_only=True)`.
2. **Asserts** the parameter shapes match those expected by the JEPA config (per §4 below).
3. **Renames** parameters from the upstream naming convention to the `lewm-rs` naming convention. The mapping is in `python/param_name_map.py` and is tested for completeness.
4. **Writes** a Safetensors file with the renamed weights, F32 (no bf16 round-trip).
5. **Builds** an empty `Jepa<NdArray<F32>>` Rust process via subprocess (`lewm-train convert --intermediate <sf> --out <mpk>`), then loads the Safetensors into Burn via `burn::record::FullPrecisionSettings`.
6. **Optionally** runs the per-layer dump (§5) on the PyTorch model, writing each layer's output for the parity fixture.

**RFC0008-001 [MUST]** — Conversion is one-way: PyTorch → Burn. We never round-trip back through PyTorch.

**RFC0008-002 [MUST]** — Every parameter in the PyTorch dict **MUST** be mapped to exactly one parameter in the Burn module. Unmapped or duplicated parameters cause `convert_reference.py` to abort with an explicit list. `param_name_map.py` is the canonical mapping.

### 3.3 Parameter name mapping

A few examples from the mapping (full list in `python/param_name_map.py`):

```
encoder.embeddings.patch_embeddings.projection.weight    →  encoder.embeddings.patch_embed.proj.weight
encoder.embeddings.cls_token                              →  encoder.embeddings.cls_token
encoder.embeddings.position_embeddings                    →  encoder.embeddings.pos_embed
encoder.encoder.layer.{i}.layernorm_before.weight         →  encoder.blocks.{i}.norm1.weight
encoder.encoder.layer.{i}.attention.attention.query.weight → (combined with key, value into) encoder.blocks.{i}.attn.qkv.weight
encoder.layernorm.weight                                  →  encoder.norm.weight
projector.0.weight                                        →  projector.fc1.weight
projector.1.weight                                        →  projector.norm.weight        # BN
projector.1.bias                                          →  projector.norm.bias
projector.1.running_mean                                  →  projector.norm.running_mean
projector.1.running_var                                   →  projector.norm.running_var
projector.3.weight                                        →  projector.fc2.weight
…
predictor.blocks.{i}.adaln.linear.weight                   →  predictor.blocks.{i}.adaln.linear.weight
```

**RFC0008-003 [MUST]** — The combined QKV mapping reshapes three `(D, D)` matrices into a single `(3D, D)` matrix per RFC 0002 §4.2.6. The script asserts shape post-concat.

**RFC0008-004 [MUST]** — `projector.0` is a Linear (fc1), `projector.1` is a BatchNorm1d (norm), `projector.3` is a Linear (fc2). The 2 index is the GELU activation, which has no parameters. The mapping skips empty modules.

### 3.4 Round-trip equivalence

After Safetensors → Burn record conversion, we run an equivalence check:

```python
# python/verify_conversion.py
pt_model = load_pt(args.pt)
pt_y     = pt_model(fixture_input).detach().cpu().numpy()

subprocess.run([
    "lewm-train", "parity", "--reference", args.burn_record, "--fixture", "tests/fixtures/parity_fixture.npz"
], check=True)

# read back the Rust output
rs_y = np.load("tests/fixtures/parity_rust_output.npz")["encoder_cls"]
assert np.allclose(pt_y, rs_y, atol=1e-4)
```

**RFC0008-005 [MUST]** — `verify_conversion.py` is the equivalent of a CI smoke for the conversion script. It runs as part of `make accept` whenever the reference checkpoint is touched.

---

## 4. Parity fixture

### 4.1 Input

The parity fixture is a single batch of deterministic inputs, stored under `tests/fixtures/parity_fixture.npz`:

```
pixels:        (B=4, T=4, C=3, H=224, W=224) F32   in [0, 1] post-imagenet-norm
actions:       (B=4, T=4, A=10)               F32   packed PushT action dim
seed:          int32 = 0
git_short_sha: str   = "<sha at fixture generation>"
```

**Generation:**

```python
# python/build_parity_fixture.py
torch.manual_seed(0)
pixels = torch.rand(4, 4, 3, 224, 224)              # uniform in [0, 1]
pixels = (pixels - 0.5) / 0.5                       # ImageNet-ViT normalize
actions = torch.randn(4, 4, 10) * 0.5               # mild scale; raw 2-D actions packed by frameskip=5
np.savez("tests/fixtures/parity_fixture.npz",
    pixels=pixels.numpy(), actions=actions.numpy(), seed=0)
```

The fixture is committed via Git LFS (~ 24 MB).

**RFC0008-006 [MUST]** — Fixture **MUST NOT** be regenerated on a whim. Regeneration requires a new RFC version bump (Minor) because all per-layer dumps must be re-baselined.

### 4.2 PyTorch per-layer dump

Running `convert_reference.py --intermediate-checks` produces:

```
dump_dir/
├── inputs/
│   ├── pixels.safetensors          # echo of the input for self-check
│   └── actions.safetensors
├── encoder/
│   ├── after_patch_embed.safetensors    (B*T, P, D)
│   ├── after_cls_concat.safetensors     (B*T, P+1, D)
│   ├── after_pos_embed.safetensors      (B*T, P+1, D)
│   ├── blocks/
│   │   ├── 00_after_attn.safetensors
│   │   ├── 00_after_mlp.safetensors
│   │   ├── …
│   │   └── 11_after_mlp.safetensors
│   ├── after_final_norm.safetensors     (B*T, P+1, D)
│   └── cls.safetensors                  (B*T, D)
├── projector/
│   └── output.safetensors               (B*T, D) → reshape (B, T, D)
├── action_encoder/
│   └── output.safetensors               (B, T, E_a)
├── predictor/
│   ├── after_pos_add.safetensors        (B, T, D)
│   ├── blocks/
│   │   ├── 00_after_attn.safetensors
│   │   ├── 00_after_mlp.safetensors
│   │   ├── …
│   │   └── 05_after_mlp.safetensors
│   └── output.safetensors               (B, T, D)
├── pred_proj/
│   └── output.safetensors               (B, T, D)
├── sigreg/
│   ├── projection_seed_0.safetensors    (K, D)   the sampled P
│   ├── empirical_c_s.safetensors         (J, K)   c and s stats
│   └── value.safetensors                 ()       L_sigreg scalar
└── meta.json                              # config hash, weights hash, generation time
```

Total size ~ 100 MB; committed to a private HF dataset `abdelstark/lewm-rs-parity-dumps` (Git LFS in-repo would exceed budget).

**RFC0008-007 [MUST]** — `meta.json` includes:

- `weights_sha256` — SHA-256 of `pytorch_model.bin`.
- `torch_version`, `transformers_version`.
- `python_version`, `numpy_version`.
- `cuda_version` — even though dumps are CPU-only, recorded for completeness.
- `fixture_seed`, `fixture_hash`.

Tests verify the metadata matches the active workspace versions; mismatch is a CI warning.

---

## 5. Rust parity tests

### 5.1 Location

```
crates/lewm-core/tests/
├── parity_encoder.rs
├── parity_predictor.rs
├── parity_sigreg.rs
├── parity_action_encoder.rs
├── parity_projector.rs
├── parity_pred_proj.rs
├── parity_init.rs               # checks init-time parameter shapes & dtypes
└── support/
    ├── mod.rs
    └── load.rs                   # helper that loads burn-record + fixture
```

### 5.2 Test skeleton

```rust
#[test]
fn encoder_cls_within_1e_4() {
    let device = burn_ndarray::NdArrayDevice::default();
    let model = support::load_jepa(&device);                       // loads reference.mpk
    let (pixels, _actions) = support::load_fixture();
    let dumps = support::load_dumps();

    let out = model.encode(pixels);                                 // (B, T, D)
    let expected_cls = dumps.encoder_cls;                           // (B*T, D)

    let diff = (out.reshape([16, 192]) - expected_cls).abs().max().into_scalar();
    assert!(diff < 1e-4, "encoder CLS L_inf = {}", diff);
}
```

### 5.3 Tolerance table

| Test | What is compared | Tolerance | Constant |
|------|------------------|-----------|----------|
| `TST-0008-IMP-001 reference_checkpoint_downloads` | reference checkpoint artifact availability | exact | n/a |
| `TST-0008-IMP-002 param_name_map_complete` | PyTorch-to-Rust parameter-name map coverage | exact | n/a |
| `TST-0008-IMP-003 burn_record_roundtrip` | converted Burn record loads into `Jepa` | exact | n/a |
| `TST-0008-IMP-004 safetensors_mirror_matches_record` | Safetensors mirror vs Burn record tensors | `1e-7` L∞ | n/a |
| `TST-0008-ENC-001 parity_encoder` | `Jepa::encode` output | `1e-4` L∞ | TOL-001 |
| `TST-0008-ENC-002 parity_encoder_per_block` | per-block hidden state | `1e-4` L∞ | TOL-001 |
| `TST-0008-PRED-001 parity_predictor` | `Jepa::predict` output | `1e-4` L∞ | TOL-002 |
| `TST-0008-PRED-002 parity_predictor_per_block` | per-block hidden state | `1e-4` L∞ | TOL-002 |
| `TST-0008-SR-001 parity_sigreg_same_seed` | scalar `L_sigreg` with identical sketch | `1e-3` abs | TOL-003 |
| `TST-0008-SR-002 parity_sigreg_diff_seed` | scalar `L_sigreg` with different sketch | `5e-2` rel | TOL-004 |
| `TST-0008-AE-001 parity_action_encoder` | `Embedder` output | `1e-4` L∞ | TOL-001 |
| `TST-0008-PROJ-001 parity_projector` | `projector(cls)` output | `1e-4` L∞ | TOL-002 |
| `TST-0008-PRPROJ-001 parity_pred_proj` | `pred_proj(predictor)` output | `1e-4` L∞ | TOL-002 |
| `TST-0008-INIT-001 parity_init` | Module parameter shapes | exact | n/a |

**RFC0008-008 [MUST]** — Tolerance is **L∞ norm**, not Frobenius or mean. Per-element max abs diff. Reason: a single elementwise off-by-axis bug shows as a spike; an averaged metric would hide it.

**RFC0008-009 [MUST]** — `TST-0008-SR-001` requires that the Rust implementation read `dumps.encoder.sigreg.projection_seed_0.safetensors` as the projection matrix `P`, **not** sample its own. This isolates the comparison to the post-projection arithmetic.

### 5.4 Per-block tests

The per-block tests iterate over the 12 encoder blocks and 6 predictor blocks, asserting parity at the output of each. A block's failure is annotated with its index; the test reports the first failure, allowing bisection.

```rust
#[test]
fn encoder_per_block_within_1e_4() {
    let (mut activations, model) = support::load_with_recording();
    let (pixels, _) = support::load_fixture();

    let _ = model.encode(pixels);                                   // recorder collects intermediate

    let dumps = support::load_dumps();
    for i in 0..12 {
        let rust = activations.get(&format!("encoder.blocks.{}.after_mlp", i)).unwrap();
        let pt   = dumps.encoder_block(i).after_mlp.clone();
        let diff = (rust.clone() - pt).abs().max().into_scalar();
        assert!(diff < 1e-4, "block {} L_inf = {}", i, diff);
    }
}
```

The recorder is a thin instrumentation layer that taps each block's output. See §5.6.

### 5.5 Activation recorder

`lewm-core::tensor_ops::recorder` provides a `Recorder` that modules can plug into during forward. In production, the recorder is a no-op; in tests it captures tensors into a `HashMap<String, Tensor<B, _>>`.

**RFC0008-010 [MUST]** — The recorder **MUST NOT** be enabled in production binaries (gated by `cfg(test)` or the `parity-fixtures` feature). Recording every layer in production would 10× memory.

### 5.6 Fixture loading

`tests/support/load.rs` knows two paths:

1. **Vendored** — fixture and dumps live under `tests/fixtures/`. Default.
2. **HF dataset** — for CI under bandwidth constraints, fetch from `abdelstark/lewm-rs-parity-dumps` via `hf-hub` with a checksum check.

```rust
pub fn load_dumps() -> ParityDumps {
    let path = std::env::var("LEWM_PARITY_DUMPS").unwrap_or_else(|_| "tests/fixtures/dumps".to_string());
    let dumps = ParityDumps::load(&path).expect("parity dumps missing; run python/convert_reference.py");
    verify_metadata(&dumps);
    dumps
}
```

---

## 6. Per-epoch parity probe

The trainer (RFC 0005 §6.3) runs a per-epoch parity probe on the fixture input. The probe is **non-asserting**: it writes `step_{N}.parity.json` with the L∞ deviations of `encoder_cls`, `predictor_output`, and `sigreg_value` from the fixture's reference. The report job ([RFC 0010 §6](0010-huggingface-hub-integration.md)) plots these over training and includes them in the model card.

The probe answers: **how does the trained model drift from the reference as training progresses?** It is **not** a parity assertion (since the trained model is meant to drift; that is the whole point of training). It is a diagnostic.

**RFC0008-011 [MUST]** — The probe runs on the **fixed fixture input**, using the **current trained weights**, and reports L∞ deviations from the **reference dump on the same input** (which used the reference weights). It is therefore a strange beast: it expects a *non-zero, growing* deviation, not a zero one.

---

## 7. CI integration

### 7.1 PR-time

On every PR, the `parity.yml` workflow:

1. Fetches the parity fixture and dumps from cache (or pulls from HF if cache miss).
2. Runs `cargo test --workspace --features parity-fixtures --test 'parity_*'`.
3. Reports failures with the L∞ value in the PR comment.

**RFC0008-012 [MUST]** — A red parity test **MUST** block merge. The only override is a documented ADR explaining a deliberate tolerance bump.

### 7.2 Nightly

A nightly workflow re-runs the parity tests with the *latest* HF dataset revision, alerting if the dataset itself has shifted. Catches upstream-side regressions.

### 7.3 Release

The release workflow (RFC 0011 §7) packages the parity dumps into the release artifact and uploads them to HF for the model card.

---

## 8. Testing the parity harness itself

The harness has its own tests:

- `test_param_name_map_complete` — runs against the actual reference state dict; asserts every key is mapped.
- `test_recorder_no_overhead_in_release` — confirms `cfg(test)` gates work.
- `test_fixture_load_round_trip` — saves and reloads the fixture; expects bit-equality.

---

## 9. Operational considerations

### 9.1 Observability

The parity test driver emits, for each test:

- `parity/<module>_max_abs_diff` (the L∞ value).
- `parity/<module>_passed` (bool).

These are written to a JSON file per run and uploaded as a CI artifact.

### 9.2 Runbook

- **"`parity_encoder` fails at 1.2e-4."** — likely an activation variant mismatch (gelu_erf vs gelu_tanh) or a LayerNorm placement issue. Bisect with `parity_encoder_per_block` to find the first failing block; examine that block's parameters with `print_tensor_diff`.
- **"`parity_sigreg_same_seed` fails."** — almost certainly a broadcasting issue in SIGReg; consult Appendix A of RFC 0003.
- **"Param name map has unmapped keys."** — `convert_reference.py` lists them; add to `param_name_map.py`. If the upstream added a new submodule, this is an expected interrupt.

### 9.3 Capacity

Disk: dumps ~ 100 MB on the dev machine, on HF dataset otherwise. Fixture ~ 24 MB on disk.

---

## 10. Performance considerations

Parity tests run on **CPU NdArray F32**, not GPU, to remove cublas non-determinism from the equation. Per test ≤ 30 s. Total parity suite ≤ 5 minutes. CI budget: under 8 minutes including artifact upload.

---

## 11. Security considerations

The reference checkpoint is publicly hosted on HF and not sensitive. Dumps are not sensitive but private to limit accidental hot-link distribution.

---

## 12. Alternatives considered

- **A1 — Use HF's `transformers` Python in the test loop (round-trip).** Rejected: pulls Python into the Rust test loop. Pre-dumped per-layer activations is cleaner.
- **A2 — Tolerance per-layer rather than uniform.** Considered. Some layers (e.g., `attn_softmax`) are more sensitive; a uniform 1e-4 absolute is fine for F32 and we keep the contract simple.
- **A3 — Property-based parity (random inputs).** Considered. Adds noise; deterministic fixture is sufficient and easier to debug.

---

## 13. Acceptance criteria

- [ ] `python/convert_reference.py` and `python/verify_conversion.py` exist and pass.
- [ ] `tests/fixtures/parity_fixture.npz` and `tests/fixtures/reference_model.meta.json` are committed.
- [ ] `abdelstark/lewm-rs-parity-dumps` HF dataset created and populated.
- [ ] All `parity_*` tests pass at the tolerances in §5.3.
- [ ] CI workflow `parity.yml` exists and runs on every PR.

---

## 14. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | Upstream HF `transformers` semantics changes | L | H | Pinned `transformers` version in `python/pyproject.toml`; nightly job alerts |
| R-2 | LFS quota exceeded | L | L | 24 MB fixture is well within budget |
| R-3 | Dumps go stale relative to upstream weights | M | M | Nightly job re-runs comparison |
| R-4 | Tolerance too tight → CI flake | L | M | We baseline at 1e-4 with 2× headroom (typical observed: 5e-5) |
| R-5 | Tolerance too loose → real regression slips through | L | H | The 2× headroom is conservative; tighter than the noise floor |

---

## 15. Open questions

OQ-2008-1 — Should we also dump intermediate gradients for backward-parity testing? Out of scope for v1 — forward parity plus 1-step optimizer test (TST-0005-OPT-002) covers backward indirectly.

---

## 16. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.1.0 | 2026-05-14 | Abdel | Refreshed fixture action shape to packed PushT actions and added source-model metadata contract. |
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0008.*
