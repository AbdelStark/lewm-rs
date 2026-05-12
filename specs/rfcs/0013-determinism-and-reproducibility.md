---
rfc: "0013"
title: "Determinism, RNG architecture, reproducibility contracts"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§4.4 risks 2,3", "§6.4 stages"]
depends_on: ["0001", "0002", "0003"]
related: ["0005", "0008", "0011"]
---

# RFC 0013 — Determinism, RNG architecture, reproducibility contracts

> **Status:** Accepted · **Version:** 1.0.0
>
> Reproducibility is a stack: bitwise on CPU, statistical on GPU. This RFC pins the RNG architecture (a named sub-stream tree), the determinism levels and where each is required, the cublas/cudnn workspace configuration, and the reproducible-build contract.

---

## 1. Introduction

### 1.1 Motivation

"Same seed, same numbers" is a contract that hardware reality breaks at three points: cublas reduction order, BF16 round-off, and the OS scheduling of data-loader threads. The right response is not "give up on determinism" but "specify what is determinable and isolate the rest."

### 1.2 Goals

1. Specify the global seed and the named RNG sub-stream tree.
2. Specify four determinism levels (Strict, Strong, Statistical, Best-effort) and where each is required.
3. Specify the cublas/cudnn workspace and operation-deterministic flags.
4. Specify the reproducible-build contract (RFC 0011 §8 dependency).
5. Specify how RNG state is serialized into checkpoints and restored at resume.

### 1.3 Non-goals

- Multi-GPU determinism (out of scope; single-GPU only per PRD).
- Floating-point summation order tweaks at the kernel level (we don't touch Burn's kernels).

---

## 2. Conventions

- "Seed" — a 64-bit integer.
- "Sub-stream" — a named RNG seeded deterministically from the global seed.
- "Bitwise" — element-wise identical floating-point comparison.
- "Statistical" — within a defined tolerance.

---

## 3. RNG library

We use **`rand_chacha::ChaCha20Rng`** exclusively for all randomness inside `lewm-rs` Rust code. Reasons:

- Cryptographically strong (overkill but harmless).
- Counter-based — supports `set_word_pos` for resumption.
- Cross-platform deterministic across `rand_chacha` v0.3.x.
- Wide library support (`rand_distr` builds on `RngCore`).

**RFC0013-001 [MUST]** — All RNG draws inside `lewm-rs` Rust code use `ChaCha20Rng`. `thread_rng()` is **forbidden** outside of test helpers explicitly marked `non-deterministic-rng = "allow"` in `clippy.toml`.

**RFC0013-002 [MUST]** — Python edge scripts (`python/`) use `numpy.random.Generator(np.random.PCG64(seed))` and `torch.Generator()` consistently. Python-side RNG state is **not** versioned across runs; it is single-shot per script invocation.

---

## 4. RNG sub-stream tree

### 4.1 Tree

```
global_seed (default 0, configurable via [training.seed])
 ├── rng:data_shuffle      — Fisher–Yates per-epoch shuffle of window indices
 ├── rng:model_init        — parameter init (consumed once at model construction)
 ├── rng:sigreg_sketch     — projection matrix P resampled per step
 ├── rng:dropout           — dropout masks (consumed per step, if dropout > 0)
 ├── rng:cem               — CEM action proposal draws (per planning step)
 └── rng:misc              — miscellaneous (collapse-probe synthetic input, etc.)
```

Future sub-streams **MUST** be added to this tree only via RFC update; the names are part of the public API of `lewm-rs`.

### 4.2 Sub-stream derivation

Sub-stream seeds derive deterministically from `(global_seed, sub_stream_name)`:

```rust
pub fn substream_seed(global: u64, name: &str) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&global.to_le_bytes());
    hasher.update(b"::");
    hasher.update(name.as_bytes());
    hasher.finalize().into()
}

pub fn substream_rng(global: u64, name: &str) -> ChaCha20Rng {
    ChaCha20Rng::from_seed(substream_seed(global, name))
}
```

**RFC0013-003 [MUST]** — `substream_seed` is the **only** way to derive sub-stream seeds. No ad-hoc hashing.

**RFC0013-004 [MUST]** — `substream_rng` is invoked once per sub-stream per run (at run start, or at resume). The returned RNG is the single owner of randomness for that sub-stream's lifetime.

### 4.3 Sub-stream lifecycle

| Sub-stream | Owner | Lifetime | Resume-restore |
|------------|-------|----------|----------------|
| `data_shuffle` | trainer (per epoch) | run | from sidecar |
| `model_init` | trainer (at INIT) | initialization only | not restored on resume (model already exists) |
| `sigreg_sketch` | `Jepa::criterion` | run | from sidecar |
| `dropout` | `Jepa` modules | run | from sidecar |
| `cem` | `Cem::plan` | per eval call | from sidecar (if mid-eval) |
| `misc` | various | run | from sidecar |

---

## 5. Determinism levels

| Level | Backend | Guarantee | Required for |
|-------|---------|-----------|--------------|
| **Strict** | `NdArray<f32>` CPU | bitwise identical across runs and machines (same `rand_chacha` v0.3.x, same OS, same CPU arch) | unit tests (L0), parity (L1), CPU smoke (L2) |
| **Strong** | single-GPU `Cuda<f32>` | bitwise identical across runs on same hardware + driver (with cublas/cudnn flags in §6) | T1 SMOKE, T2 SHORT |
| **Statistical** | single-GPU `Cuda<bf16-mixed>` | identical to TOL-010 end-of-epoch | T3 FULL |
| **Best-effort** | different hardware (A100 vs A10G) | metrics within TOL-005 | cross-hardware comparison reporting |

**RFC0013-005 [MUST]** — CI enforces **Strict** at L0/L1/L2 (running on CPU runners).

**RFC0013-006 [MUST]** — A T3 run records the determinism level used in its `state.json` and in the model card.

**RFC0013-007 [MUST]** — Running `lewm-train smoke --device cpu` twice with the same seed produces bit-identical loss curves at every step (Strict).

**RFC0013-008 [SHOULD]** — Running `lewm-train train --device cuda:0 --precision f32 --max-steps 100` twice produces bit-identical loss curves (Strong). The "should" reflects the cublas non-determinism we cannot fully eliminate; the configuration in §6 minimizes the gap to bitwise.

---

## 6. cublas / cudnn deterministic configuration

When `--device cuda:0`, the trainer sets:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8                # the deterministic workspace size
CUDA_DETERMINISTIC_OPS=1                        # NVIDIA's "everything deterministic" guard
CUDNN_DETERMINISTIC=1                           # legacy cudnn flag
```

**RFC0013-009 [MUST]** — The trainer **emits a warning** if `CUBLAS_WORKSPACE_CONFIG` is unset and CUDA is selected. (Burn does not set it for us.)

**RFC0013-010 [SHOULD]** — Set these via the HF Jobs YAML environment block to lock the value at run launch.

**RFC0013-011 [MUST]** — Some operations remain non-deterministic at the kernel level (e.g., `atomicAdd` reductions). Burn's high-level API does not expose direct control over these; we accept the residual non-determinism and document it in the Strong level's small bitwise drift (typically ≤ 1e-7 per element on F32).

---

## 7. Deterministic operations checklist

A short pre-flight list ensuring no source of avoidable non-determinism slips in:

- [ ] File system enumeration: use `read_dir().sorted()` everywhere. `walkdir::WalkDir::sort_by_file_name()` enabled.
- [ ] HashMap iteration: never iterate `std::collections::HashMap`; use `BTreeMap` or sort.
- [ ] Time as input: only via `Clock` trait that tests can inject; no `Instant::now()` in core logic.
- [ ] OS-provided randomness: forbidden in core; only `ChaCha20Rng`.
- [ ] Threading order: in data-loader workers, the result is reassembled by `idx`, so worker scheduling does not affect output order.
- [ ] Mutex order: lock acquisition order is documented per module.

CI's `scripts/check_nondet.py` greps for `thread_rng`, `HashMap::iter`, and `Instant::now` outside allowlisted files.

---

## 8. RNG state serialization

### 8.1 In sidecar

Per RFC 0005 §6.1, the checkpoint sidecar includes:

```json
"rng_state": {
  "global_seed": 0,
  "step_at_save": 14400,
  "data_shuffle":    "<base64 ChaCha20 state>",
  "sigreg_sketch":   "<base64>",
  "dropout":         "<base64>",
  "cem":             "<base64>",
  "model_init":      "<base64>"
}
```

### 8.2 Serialization format

`ChaCha20Rng::get_word_pos() -> u128` + the 32-byte seed gives a full state. We serialize as `(seed: [u8;32], word_pos: u128)`:

```rust
pub fn serialize_rng(rng: &ChaCha20Rng) -> Vec<u8> {
    let mut buf = Vec::with_capacity(48);
    buf.extend_from_slice(&rng.get_seed());
    buf.extend_from_slice(&rng.get_word_pos().to_le_bytes());
    buf
}

pub fn deserialize_rng(bytes: &[u8]) -> Result<ChaCha20Rng, TrainError> {
    let seed: [u8; 32] = bytes[..32].try_into()?;
    let word_pos = u128::from_le_bytes(bytes[32..48].try_into()?);
    let mut rng = ChaCha20Rng::from_seed(seed);
    rng.set_word_pos(word_pos);
    Ok(rng)
}
```

**RFC0013-012 [MUST]** — Resume restores each sub-stream's RNG via `deserialize_rng`; the **next** draw after restore is exactly the draw that would have happened next in a non-crashing run.

### 8.3 Anti-drift

`model_init` is consumed once at the start and never again; its post-init state is recorded for **auditing** but not restored. If resume is from a step `N > 0`, the model is loaded from checkpoint (which captures the post-trained state); re-initializing the model would defeat the resume.

---

## 9. Reproducible build contract

Already covered in [RFC 0011 §8](0011-ci-cd-and-release-engineering.md). Tied here because reproducible build is a determinism property:

- Same git SHA + pinned toolchain → byte-identical binary.
- Binary is the artifact that, given the same dataset and seed, reproduces the loss curve.

---

## 10. Cross-version stability

Bumping a dep that touches RNG (e.g., `rand_chacha` minor version) is a **breaking** change to the reproducibility contract.

**RFC0013-013 [MUST]** — `rand_chacha` is pinned `=0.3.x` with a TBD x at lock time. Any bump that crosses a minor version requires:

- An ADR explaining the change.
- A side-by-side comparison of a parity probe before/after.
- Acceptance that "previous-version checkpoints continue to reproduce on the previous version" — we do not rewrite history.

---

## 11. Testing strategy

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0013-DET-001 | `cpu_strict_bitwise_two_runs` | integration | RFC0013-007 |
| TST-0013-DET-002 | `data_shuffle_deterministic` | unit | §4 |
| TST-0013-DET-003 | `substream_seed_distinct` | unit | RFC0013-003 |
| TST-0013-DET-004 | `rng_serialize_round_trip` | unit | §8.2 |
| TST-0013-REPRO-001 | `resume_loss_curves_continue_identical` | integration | RFC0013-012 |
| TST-0013-NONDET-001 | `check_nondet_lint_passes` | meta | §7 |

Fixtures:

- A `tests/fixtures/seed_0_pusht_first_100_steps.jsonl` file capturing the canonical loss curve for `seed=0, NdArray, F32`. CI re-runs the smoke and compares.

---

## 12. Operational considerations

### 12.1 Observability

- `rng/state_word_pos[<substream>]` emitted every checkpoint (for trace post-hoc).
- `rng/seed` emitted once at startup.

### 12.2 Runbook

- **"Two runs with the same seed differ at step 50."** — first suspect a `HashMap::iter` somewhere. Run `scripts/check_nondet.py --include-tests`. Second suspect a Burn op that has internal randomness; inspect `tracing` for unexpected `randn` calls.
- **"Resume produces a different next batch."** — verify the sidecar's `step_at_save` matches the model's step counter. Check the `data_shuffle` sub-stream's `word_pos`.

### 12.3 Capacity

RNG state is tiny (48 bytes per sub-stream); negligible.

---

## 13. Performance considerations

`ChaCha20` is fast enough at the per-step granularity. Sub-stream construction (BLAKE3) is one-shot per run. No optimization needed.

---

## 14. Security considerations

The RNG seeds are not secret; they are configuration. Cryptographic strength of `ChaCha20Rng` is incidental.

---

## 15. Alternatives considered

- **A1 — `rand::thread_rng`.** Rejected: non-deterministic, OS-dependent.
- **A2 — `xoshiro` family.** Considered. ChaCha20 chosen for resume word_pos support and library uniformity.
- **A3 — Per-step explicit RNG passing via function args.** Implemented internally. The sub-stream pattern is the public surface.

---

## 16. Acceptance criteria

- [ ] All TST-0013-* pass.
- [ ] CI's nondet linter passes.
- [ ] `seed_0_pusht_first_100_steps.jsonl` matches on two CI runs.
- [ ] Reproducible-build check (RFC 0011) passes on release.

---

## 17. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | A Burn op uses internal `thread_rng` | M | H | Tracing-level audit at first SMOKE; file Burn issue if found |
| R-2 | cublas non-determinism larger than expected | L | M | Strong level is "should"; Statistical is acceptable in T3 |
| R-3 | RNG sub-stream tree expands incoherently | L | M | RFC update required to add a sub-stream |
| R-4 | rand_chacha bump breaks existing checkpoints | L | M | Pin to `=0.3.x`; ADR-gated bump |

---

## 18. Open questions

OQ-2013-1 — Should we offer a `--strict` flag that aborts on any detected non-determinism (e.g., NaN that wasn't in the canonical run)? Considered. v1 instead relies on the canonical-curve regression test in CI.

---

## 19. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0013.*
