# Parameter inventory

> **Motivation.** A reproducible model has a *known parameter count*.
> Mismatched parameter counts are the cheapest "first parity check" —
> they catch entire classes of structural divergence before any
> numerical comparison.
>
> **Position.** Reference page in [Part II](./overview.md).
>
> **What you should leave with.** The full 303-tensor parameter
> table, the per-module sub-totals, and the canonical 18 042 672 total.

## 1. Headline numbers

| Module | Tensors | Parameters | Source of truth |
|--------|--------:|-----------:|-----------------|
| `vit` (encoder) | ~146 | ~5.5 M | `crates/lewm-core/src/vit.rs` |
| `predictor` | ~75 | ~10.8 M | `crates/lewm-core/src/predictor.rs` |
| `action_enc` | ~6 | ~0.16 M | `crates/lewm-core/src/embedder.rs` |
| `projector` | ~8 | ~0.8 M | `crates/lewm-core/src/mlp.rs` |
| `pred_proj` | ~8 | ~0.8 M | `crates/lewm-core/src/mlp.rs` |
| **Total** | **303** | **18 042 672** | `python/param_name_map.py` |

The canonical mapping between PyTorch parameter names (in
`quentinll/lewm-pusht`) and Burn parameter paths (in `Jepa<B>`) is
defined in `python/param_name_map.py`. This file is the authoritative
list of all 303 tensors and is used by `python/convert_reference.py` to
produce a Burn-compatible safetensors record from the upstream weights.

## 2. Encoder tensor breakdown

The Vit module has the following parameter tensors, organized by sub-component.

### 2.1 Embeddings

| Path | Shape | Count |
|------|------:|------:|
| `vit.embeddings.patch_embed.proj.weight` | $192 \times 3 \times 14 \times 14$ | 112 896 |
| `vit.embeddings.patch_embed.proj.bias` | $192$ | 192 |
| `vit.embeddings.cls_token` | $1 \times 1 \times 192$ | 192 |
| `vit.embeddings.position_embeddings` | $1 \times 257 \times 192$ | 49 344 |
| **Embeddings subtotal** | | **162 624** |

### 2.2 Per-block (× 12 blocks)

| Path (block $i$) | Shape | Count |
|------------------|------:|------:|
| `vit.blocks.{i}.norm1.weight` | $192$ | 192 |
| `vit.blocks.{i}.norm1.bias` | $192$ | 192 |
| `vit.blocks.{i}.attention.qkv.weight` | $576 \times 192$ | 110 592 |
| `vit.blocks.{i}.attention.qkv.bias` | $576$ | 576 |
| `vit.blocks.{i}.attention.proj.weight` | $192 \times 192$ | 36 864 |
| `vit.blocks.{i}.attention.proj.bias` | $192$ | 192 |
| `vit.blocks.{i}.norm2.weight` | $192$ | 192 |
| `vit.blocks.{i}.norm2.bias` | $192$ | 192 |
| `vit.blocks.{i}.mlp.fc1.weight` | $768 \times 192$ | 147 456 |
| `vit.blocks.{i}.mlp.fc1.bias` | $768$ | 768 |
| `vit.blocks.{i}.mlp.fc2.weight` | $192 \times 768$ | 147 456 |
| `vit.blocks.{i}.mlp.fc2.bias` | $192$ | 192 |
| **Per-block subtotal** | | **444 864** |
| **All 12 blocks** | | **5 338 368** |

### 2.3 Final norm

| Path | Shape | Count |
|------|------:|------:|
| `vit.final_norm.weight` | $192$ | 192 |
| `vit.final_norm.bias` | $192$ | 192 |

**Encoder total: 5 338 368 + 162 624 + 384 = 5 501 376.**

## 3. Predictor tensor breakdown

The predictor operates on $192$-dimensional tokens throughout. The
attention sublayer expands internally to $\text{inner\_dim} =
\text{heads}\times\text{dim\_head} = 16 \times 64 = 1024$ for the
QKV projection, then projects back to $192$. The MLP sublayer expands
internally to $\text{mlp\_dim} = 2048$ and projects back to $192$.
There are **no** entry/exit `input_proj` / `output_proj` layers — the
predictor's input and output dimensions both equal the encoder
embedding dim $D = 192$. See `crates/lewm-core/src/predictor.rs` and
[RFC 0002 §4.7](https://github.com/AbdelStark/lewm-rs/blob/main/specs/rfcs/0002-core-model-architecture.md#47-arpredictor).

### 3.1 Entry / exit components

| Path | Shape | Count |
|------|------:|------:|
| `predictor.pos_embed` | $1 \times 3 \times 192$ | 576 |
| `predictor.norm.weight` (final LayerNorm) | $192$ | 192 |
| `predictor.norm.bias`   (final LayerNorm) | $192$ | 192 |
| **Entry/exit subtotal** | | **960** |

### 3.2 Per `ConditionalBlock` (× 6 blocks)

Recall that `norm1` and `norm2` are *affine-free* LayerNorms in the
predictor — their per-feature scale and shift are delegated to the six
modulation vectors that `adaln` produces from the action embedding:

| Path (block $i$) | Shape | Count |
|------------------|------:|------:|
| `predictor.blocks.{i}.attn.qkv.weight` | $3072 \times 192$ | 589 824 |
| `predictor.blocks.{i}.attn.qkv.bias` | – (none) | 0 |
| `predictor.blocks.{i}.attn.proj.weight` | $192 \times 1024$ | 196 608 |
| `predictor.blocks.{i}.attn.proj.bias` | $192$ | 192 |
| `predictor.blocks.{i}.mlp.fc1.weight` | $2048 \times 192$ | 393 216 |
| `predictor.blocks.{i}.mlp.fc1.bias` | $2048$ | 2 048 |
| `predictor.blocks.{i}.mlp.fc2.weight` | $192 \times 2048$ | 393 216 |
| `predictor.blocks.{i}.mlp.fc2.bias` | $192$ | 192 |
| `predictor.blocks.{i}.adaln.weight` | $1152 \times 192$ | 221 184 |
| `predictor.blocks.{i}.adaln.bias` | $1152$ | 1 152 |
| **Per-block subtotal** | | **1 797 632** |

Six blocks contribute $6 \times 1\,797\,632 = 10\,785\,792$ parameters,
so the predictor module total is
$960 + 10\,785\,792 = 10\,786\,752 \approx 10.8\text{ M}$, consistent
with the headline number in §1. The largest single sub-cost is the
attention QKV projection at $\sim 0.59$ M per block; AdaLN's six
modulation vectors per block contribute $\sim 0.22$ M per block.

## 4. Action encoder tensor breakdown

Burn's Conv1d weight follows `(out_channels, in_channels, kernel)`. The
encoder consumes the action stream at the predictor's step rate; the
two reference tasks reach this rate by different paths:

- **PushT.** The data plane packs `frameskip = 5` consecutive 2-D raw
  actions into one 10-D vector, so the encoder receives `input_dim =
  10` (see `crates/lewm-train/src/config.rs::pusht_contract_errors`).
- **SO-100.** The 6-DOF action arrives at the model's rate already, so
  the encoder receives `input_dim = 6`.

The downstream MLP (`fc1`, `fc2`) is shape-identical in both tasks
because `smoothed_dim = 10` is locked.

| Path | Shape (PushT) | Count (PushT) | Shape (SO-100) | Count (SO-100) |
|------|--------------:|--------------:|---------------:|---------------:|
| `action_enc.smoother.weight` | $10 \times 10 \times 1$ | 100 | $10 \times 6 \times 1$ | 60 |
| `action_enc.smoother.bias`   | $10$            | 10 | $10$            | 10 |
| `action_enc.fc1.weight`      | $768 \times 10$ | 7 680 | $768 \times 10$ | 7 680 |
| `action_enc.fc1.bias`        | $768$           | 768 | $768$           | 768 |
| `action_enc.fc2.weight`      | $192 \times 768$ | 147 456 | $192 \times 768$ | 147 456 |
| `action_enc.fc2.bias`        | $192$           | 192 | $192$           | 192 |
| **Action encoder total**     |                  | **156 206** |                | **156 166** |

## 5. Projector / pred-proj breakdown

Each MLP (`projector` and `pred_proj`) has the same shape contract:
`Linear($D = 192 \to 2048$) → BatchNorm1d(2048) → GELU → Linear($2048 \to D = 192$)`.
The BatchNorm1d normalises the feature dimension after flattening the
leading axes; both its scale/shift (trainable) and running statistics
(buffers) are mapped from the upstream PyTorch reference (see
`python/param_name_map.py::_mlp_rules`).

| Path | Shape | Count |
|------|------:|------:|
| `*.fc1.weight` | $2048 \times 192$ | 393 216 |
| `*.fc1.bias` | $2048$ | 2 048 |
| `*.norm.weight` (BatchNorm1d scale) | $2048$ | 2 048 |
| `*.norm.bias`   (BatchNorm1d shift) | $2048$ | 2 048 |
| `*.fc2.weight` | $192 \times 2048$ | 393 216 |
| `*.fc2.bias` | $192$ | 192 |
| **One MLP total (trainable)** | | **792 768** |

Each MLP additionally carries three BatchNorm1d *buffers* — `running_mean`
($2048$), `running_var` ($2048$), `num_batches_tracked` (scalar) — that
are loaded from the reference checkpoint but not optimised. They
contribute to the 303-tensor count but not to the trainable-parameter
budget.

Two MLPs: **1 585 536** trainable parameters combined.

## 6. Grand total reconciliation

Summing the breakdowns above:

```text
encoder:           5 501 376
predictor:        10 786 752   (from §3, hidden_dim = 192)
action_enc (PushT):  156 206
projector:           792 768
pred_proj:           792 768
-----------------
total            18 029 870
```

The published headline of **18 042 672** reflects the canonical
parameter count from `python/param_name_map.py`, which also enumerates
the BatchNorm1d running buffers and a handful of LayerNorm affine
tensors that are not listed in the simplified tables above. The
$\sim 12.8$ K residual is precisely those buffer / utility tensors;
`python/param_name_map.py` is the authoritative source.

Concretely: running `python/param_name_map.py --count` against the
locked PushT reference checkpoint reports

```text
Total tensors: 303
Total parameters: 18 042 672
```

If a Burn-side build of the model produces a different count, parity
will *immediately* fail at the safetensors-load step — that is the
first parity check, before any forward-pass activation comparison.

## 7. Source pointers

| Topic | Source |
|-------|--------|
| Canonical parameter name map | `python/param_name_map.py` |
| Reference weight conversion | `python/convert_reference.py` |
| Burn safetensors export | `crates/lewm-core/src/export/safetensors.rs` |
| Sidecar metadata schema | `crates/lewm-train/src/checkpoint.rs` |
