# Determinism and reproducibility

> **Motivation.** A reproducible training run is the difference between
> "the result is in the paper" and "the result is in the field". This
> page documents the determinism contracts pinned by [RFC 0013].
>
> **Position.** Seventh sub-page in [Part III](./pipeline.md).
>
> **What you should leave with.** What "deterministic" means here, the
> RNG sub-stream system, and the resume guarantee.

## 1. What "deterministic" means

A `lewm-rs` training run is deterministic in the following precise
sense:

> **Given the same source revision, the same TOML config, the same
> dataset checksum, and the same seed, two runs on the same hardware
> produce bit-identical losses at every step.**

This excludes:

- **Non-determinism in CUDA kernels.** Some CUDA matmul kernels make
  reduction-order choices that are not bit-identical between runs.
  This is unavoidable at the cuDNN / cuBLAS layer and is set aside.
  Burn picks deterministic kernels where available.
- **Cross-hardware reproducibility.** Float reductions on different
  GPU architectures are not bit-identical. lewm-rs's contract is
  *same-hardware* determinism.
- **CPU vs GPU.** A CPU smoke and a GPU full run will not produce
  bit-identical losses, but they should match to within the tolerance
  TOL-005 (rel. < 1e-2) at step 100.

The full contract is in [RFC 0013].

## 2. The named RNG sub-streams

`lewm-rs` does not pass around `rand::ThreadRng`. Every stochastic
operation in the system draws from a *named, deterministic sub-stream*
of a master RNG. The master RNG is seeded once from the run's seed
(default 0).

| Sub-stream | Used by |
|------------|---------|
| `rng:master` | Top of the tree; spawns all others. |
| `rng:weight_init` | Truncated-normal weight initialisation in `Module::init`. |
| `rng:dataset_sample` | Window sampling (which episode, which start frame). |
| `rng:dataset_worker.{worker_id}` | Per-worker prefetch order. |
| `rng:sigreg_sketch` | SIGReg's random projection matrix. |
| `rng:cem` | CEM proposal sampling at eval time. |
| `rng:dropout.{module_path}` | Future-proofed; LeWM has no dropout but the path exists. |

Each sub-stream is seeded by hashing `(master_seed, name)`. This makes
the sub-stream order independent of the order in which they are first
drawn from, so adding a new sub-stream does not perturb the values of
existing ones.

The implementation is `crates/lewm-core/src/rng.rs`:

```rust,ignore
pub struct RngTree {
    master_seed: u64,
    streams: HashMap<String, Xoshiro256PlusPlus>,
}

impl RngTree {
    pub fn substream(&mut self, name: &str) -> &mut Xoshiro256PlusPlus {
        self.streams.entry(name.to_string()).or_insert_with(|| {
            let mut hasher = SipHasher::new_with_keys(self.master_seed, 0);
            name.hash(&mut hasher);
            let sub_seed = hasher.finish();
            Xoshiro256PlusPlus::seed_from_u64(sub_seed)
        })
    }
}
```

Substream RNGs are persisted in the checkpoint sidecar
(`step_{N}.json::rng_state`) and restored on resume.

## 3. The resume guarantee

**RFC0013-001 [MUST]** — Given a checkpoint at step $N$, resuming from
it produces *bit-identical* losses at step $N+1, N+2, \dots$ as a fresh
run that reaches step $N$ without interruption.

This is verified by the resume parity test in
`crates/lewm-train/tests/resume_parity.rs`: it runs 100 steps in one
shot, then again from a step-50 checkpoint, and asserts that the
step-51..100 losses are identical to 6 decimal places.

The resume protocol restores:

- All Burn module parameters (from `step_{N}.mpk`).
- AdamW first and second moments.
- Scheduler step counter.
- All named RNG sub-streams.
- The training state (one of the 8 states in
  [State machine](./state-machine.md)).

## 4. Non-determinism gates

Beyond the seed-based determinism, three CI gates check for *accidental*
non-determinism:

1. **`scripts/check_nondet.py`** — scans the codebase for forbidden
   patterns: `rand::thread_rng()`, `SystemTime::now()` in non-telemetry
   paths, `std::collections::HashMap::iter()` in serialization code
   (HashMap iteration order is non-deterministic; BTreeMap is required
   for canonical orderings).
2. **Workspace lint** — `clippy::nondet_iter`, `clippy::nondet_clone`.
3. **The parity workflow** — implicitly catches non-determinism by
   comparing activations against fixed dumps. Any non-determinism would
   surface as flaky parity tests.

## 5. The seed envelope

| Knob | Value | Notes |
|------|-------|-------|
| Master seed | 0 | TOML override allowed. |
| Sub-stream derivation | SipHash(master_seed, name) | Deterministic function of master_seed and name. |
| RNG algorithm | Xoshiro256PlusPlus | Same seed → same sequence on any platform. |

The choice of Xoshiro256++ over PCG64 is purely pragmatic: it's the
default RNG in the Rust ecosystem and matches Burn's internal RNG
choice. Either would satisfy the contract.

## 6. Source pointers

| Topic | Source |
|-------|--------|
| RNG tree | `crates/lewm-core/src/rng.rs` |
| Sidecar RNG state | `crates/lewm-train/src/checkpoint.rs` |
| Resume protocol | `crates/lewm-train/src/resume.rs` |
| Non-det check | `scripts/check_nondet.py` |
| Resume parity test | `crates/lewm-train/tests/resume_parity.rs` |

[RFC 0013]: ../reference/rfcs.md
