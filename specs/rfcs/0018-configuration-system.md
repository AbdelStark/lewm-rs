---
rfc: "0018"
title: "Configuration system, TOML schema, layered overrides"
status: Accepted
version: 1.0.0
authors: ["Abdel"]
reviewers: []
created: 2026-05-12
updated: 2026-05-12
supersedes: []
superseded_by: null
tracks_prd: ["§5.2"]
depends_on: ["0001"]
related: ["0002", "0005", "0017"]
---

# RFC 0018 — Configuration system, TOML schema, layered overrides

> **Status:** Accepted · **Version:** 1.0.0
>
> Configs are where ML projects most often die from typos. This RFC pins the configuration schema, the layered-override semantics, the validation rules, and the deny-unknown-fields invariant that makes typos hard to commit.

---

## 1. Introduction

### 1.1 Motivation

A 40-knob training config in a YAML file with no schema is the classic recipe for "wait, why is `wieght_decay: 0.05` being ignored?" We use TOML, serde with `deny_unknown_fields`, validator with explicit constraints, and a layered-merge that is explicit about precedence.

### 1.2 Goals

1. Define a single Rust struct hierarchy as the source of truth for configurable knobs.
2. Pin the TOML schema in canonical, human-readable form.
3. Pin the override precedence (defaults → file → CLI).
4. Pin the validation rules.
5. Pin the config-hash contract used for reproducibility.

### 1.3 Non-goals

- Runtime reconfiguration (configs are immutable per run).
- Distributed config (single-process only).

---

## 2. Conventions

- TOML 1.0.
- Keys are `snake_case`.
- All durations in seconds unless suffixed (`{value}s`, `{value}m`, `{value}h`).
- All sizes in bytes unless suffixed (`{value}KB`, `{value}MB`, `{value}GB`).
- All learning rates and loss weights are `f64` to allow precise scientific notation.

---

## 3. Crate placement

The config schema lives in `lewm-train::config` since the trainer owns the top-level binary. Module crates' configs are imported and composed:

```
lewm-train::config::RootConfig
 ├── dataset:        lewm-data::config::DatasetConfig (enum: Pusht | So100)
 ├── model:          lewm-core::config::JepaConfig
 ├── loss:           lewm-train::config::LossConfig
 ├── training:       lewm-train::config::TrainingConfig
 ├── eval:           lewm-plan::config::EvalConfig
 ├── infer:          lewm-infer::config::InferConfig (only for serve subcommand)
 ├── observability:  lewm-telemetry::config::TelemetryConfig
 └── hub:            lewm-hub::config::HubConfig (subcommand-dependent)
```

`lewm-eval` and `lewm-infer` binaries use the same root with subsets of sections activated.

---

## 4. Schema (Rust)

### 4.1 Root

```rust
#[derive(serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct RootConfig {
    pub schema_version: SemverString,         // e.g., "1.0.0"
    pub dataset:        DatasetConfig,
    pub model:          JepaConfig,
    pub loss:           LossConfig,
    pub training:       TrainingConfig,
    pub eval:           EvalConfig,

    #[serde(default)]
    pub observability:  TelemetryConfig,
    #[serde(default)]
    pub hub:            HubConfig,
    #[serde(default)]
    pub infer:          InferConfig,

    /// Reserved for ml-intern overlays.
    #[serde(default)]
    pub experimental:   ExperimentalConfig,
}

#[derive(serde::Deserialize, Debug, Clone)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DatasetConfig {
    Pusht(PushtConfig),
    So100(So100Config),
}
```

**RFC0018-001 [MUST]** — Every struct has `#[serde(deny_unknown_fields)]`. Typo'd keys produce an explicit error at load time.

**RFC0018-002 [MUST]** — `schema_version` is mandatory and is compared against the binary's compiled-in expectation. Mismatch is a load-time error suggesting the fix.

### 4.2 `TrainingConfig`

```rust
#[derive(serde::Deserialize, Debug, Clone, validator::Validate)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    #[validate(range(min = 1, max = 100))]
    pub history_size: usize,                 // default 3

    #[validate(range(min = 2, max = 64))]
    pub horizon: usize,                       // default 8

    #[validate(range(min = 1, max = 1024))]
    pub batch_size: usize,                    // default 64

    #[validate(range(min = 1, max = 32))]
    pub grad_accum_steps: usize,              // default 2

    #[serde(default = "default_optimizer")]
    pub optimizer: OptimizerKind,             // "adamw"

    #[validate(range(min = 1.0e-7, max = 1.0e-2))]
    pub lr_peak: f64,                         // default 3e-4

    #[validate(range(min = 1.0e-9, max = 1.0e-3))]
    pub lr_min: f64,                          // default 1e-5

    #[validate(range(min = 0, max = 100_000))]
    pub warmup_steps: u32,                    // default 1000

    #[validate(range(min = 0.0, max = 1.0))]
    pub weight_decay: f64,                    // default 0.05

    pub betas: (f64, f64),                    // default (0.9, 0.95)

    #[validate(range(min = 1, max = 1000))]
    pub epochs: u32,                          // default 10

    #[serde(default)]
    pub precision: PrecisionKind,             // default "bf16_mixed"

    #[serde(default)]
    pub seed: u64,                            // default 0

    #[serde(default = "default_grad_clip")]
    #[validate(range(min = 0.1, max = 100.0))]
    pub grad_clip_norm: f64,                  // default 1.0

    #[serde(default)]
    pub warmstart_from: Option<PathBuf>,

    #[serde(default = "default_eval_every")]
    #[validate(range(min = 1, max = 100))]
    pub eval_every_n_epochs: u32,             // default 5

    #[serde(default = "default_probe_every")]
    #[validate(range(min = 10, max = 10_000))]
    pub probe_every_n_steps: u32,             // default 100
}

#[derive(serde::Deserialize, Debug, Clone, Copy, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind { Adamw, Lion }

#[derive(serde::Deserialize, Debug, Clone, Copy, Eq, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionKind {
    F32,
    #[default]
    Bf16Mixed,
}
```

### 4.3 `LossConfig`

```rust
#[derive(serde::Deserialize, Debug, Clone, validator::Validate)]
#[serde(deny_unknown_fields)]
pub struct LossConfig {
    #[validate(range(min = 0.0, max = 100.0))]
    pub lambda_sigreg: f64,                  // default 1.0

    #[validate(range(min = 8, max = 64))]
    pub sigreg_knots: usize,                  // default 17

    #[validate(range(min = 64, max = 8192))]
    pub sigreg_num_proj: usize,               // default 1024

    #[validate(range(min = 1.0, max = 10.0))]
    pub sigreg_t_max: f64,                    // default 3.0
}
```

### 4.4 Other sections

Full schemas for `JepaConfig`, `PushtConfig`, `So100Config`, `EvalConfig`, `TelemetryConfig`, `HubConfig`, `InferConfig`, `ExperimentalConfig` are in the corresponding crate-specific RFCs and code; this RFC enumerates only their existence and required validation.

---

## 5. Canonical TOML

### 5.1 `configs/pusht.toml` (full)

```toml
schema_version = "1.0.0"

[dataset]
kind = "pusht"
root_path = "/data/lewm-pusht"
split = "train"
horizon = 8
history_size = 3
seed = 0

[model.encoder]
size = "small"
image_size = 224
patch_size = 16
num_channels = 3
hidden_size = 384
num_hidden_layers = 12
num_attention_heads = 6
intermediate_size = 1536
hidden_act = "gelu_tanh"
attention_probs_dropout_prob = 0.0
hidden_dropout_prob = 0.0
layer_norm_eps = 1.0e-12
use_cls_token = true
interpolate_pos_encoding = false

[model.action_encoder]
input_dim = 2
smoothed_dim = 16
emb_dim = 64
mlp_scale = 4

[model.predictor]
num_frames = 16
depth = 6
heads = 6
mlp_dim = 1536
dim_head = 64
hidden_dim = 384
action_emb_dim = 64
dropout = 0.0
emb_dropout = 0.0

[model.projector]
input_dim = 384
hidden_dim = 1536
output_dim = 384
norm = "batch_norm_1d"

[model.pred_proj]
input_dim = 384
hidden_dim = 1536
output_dim = 384
norm = "batch_norm_1d"

[model.history_size]
value = 3

[model.horizon]
value = 8

[loss]
lambda_sigreg = 1.0
sigreg_knots = 17
sigreg_num_proj = 1024
sigreg_t_max = 3.0

[training]
history_size = 3
horizon = 8
batch_size = 64
grad_accum_steps = 2
optimizer = "adamw"
lr_peak = 3.0e-4
lr_min = 1.0e-5
warmup_steps = 1000
weight_decay = 0.05
betas = [0.9, 0.95]
epochs = 10
precision = "bf16_mixed"
seed = 0
grad_clip_norm = 1.0
eval_every_n_epochs = 5
probe_every_n_steps = 100

[eval]
kind = "pusht_simulated"
episode_ids = [0, 7, 13, 21, 28, 34, 41, 48, 55, 62, 69, 76, 83, 90, 97,
               104, 111, 118, 125, 132, 139, 146, 153, 160, 167, 174, 181,
               188, 195, 202, 209, 216, 223, 230, 237, 244, 251, 258, 265,
               272, 279, 286, 293, 300, 307, 314, 321, 328, 335, 342]
max_steps_per_episode = 100
n_iter = 5
n_cand = 1000
n_elite = 100
horizon_plan = 5
sigma_init = 1.0
sigma_min = 0.05

[observability]
trackio_run_name_prefix = "lewm-rs-pusht"
otel_endpoint_env = "OTEL_EXPORTER_OTLP_ENDPOINT"
tensorboard_dir = "tb"

[hub]
model_repo = "AbdelStark/lewm-rs-pusht"
upload_at_end = true
upload_every_n_epochs = 0   # only at end; set >0 to upload mid-run

[experimental]
```

### 5.2 `configs/so100.toml`

Per RFC 0012 §10.1; identical structure, dataset = so100, action_encoder.input_dim = 6.

### 5.3 `configs/so100_warmstart.toml`

Inherits so100.toml; sets `[training].warmstart_from`.

---

## 6. Override mechanism

### 6.1 Order

```
defaults (in Rust structs)   < file (--config)   < CLI (--set key=value)
```

### 6.2 CLI override

```bash
lewm-train train --config configs/pusht.toml \
    --set training.lr_peak=1.0e-4 \
    --set training.batch_size=32
```

Each `--set` is parsed as `key=value`, the value parsed as TOML scalar. Repeated `--set` overrides accumulate. Setting an unknown key fails with the same `deny_unknown_fields` error.

**RFC0018-003 [MUST]** — Override syntax is `key.path=value` with dots delimiting nesting. Arrays are set wholesale: `--set training.betas=[0.9,0.99]`.

### 6.3 Env-var overrides

A small allowlist of env vars overrides specific fields:

```
LEWM_SEED                  → training.seed
LEWM_LR_PEAK               → training.lr_peak
LEWM_HF_TOKEN              → hub.token (loaded into Secrets, not the config struct)
LEWM_OTEL_ENDPOINT         → observability.otel_endpoint
LEWM_DEVICE                → device flag (not a config field)
```

**RFC0018-004 [MUST]** — Env vars come *after* file but *before* CLI `--set` in precedence.

### 6.4 Merge implementation

```rust
pub fn load_root(
    config_path: &Path,
    env: &EnvOverrides,
    cli_sets: &[(String, String)],
) -> Result<RootConfig, ConfigError> {
    let text = std::fs::read_to_string(config_path)?;
    let mut value: toml::Value = toml::from_str(&text)?;
    apply_env(&mut value, env)?;
    for (key, val) in cli_sets {
        apply_set(&mut value, key, val)?;
    }
    let root: RootConfig = value.try_into()?;
    root.validate()?;  // validator::Validate run on all annotated structs
    Ok(root)
}
```

---

## 7. Validation

Every numeric field uses `validator::Validate` with explicit ranges (§4). Some semantic checks are cross-field:

- `epochs * (dataset_len / (batch_size * grad_accum_steps)) > warmup_steps`.
- `eval.horizon_plan ≤ model.predictor.num_frames - training.history_size`.
- `loss.sigreg_num_proj` divisible by 32 (recommended for kernel alignment) — emits a warning, not an error.

These cross-field checks live in `RootConfig::validate_post(&self)` and are run after deserialization.

**RFC0018-005 [MUST]** — Cross-field checks **MUST** emit precise error messages following the style guide of RFC 0017 §5.

---

## 8. Config hash

The reproducibility contract (RFC 0013 §9) uses a `config_hash` to tie a checkpoint to its config:

```rust
pub fn canonical_hash(root: &RootConfig) -> String {
    let text = toml::to_string_pretty(root)?;          // serde_toml sorts keys
    let h    = blake3::hash(text.as_bytes());
    hex::encode(&h.as_bytes()[..6])                    // 12 hex chars
}
```

**RFC0018-006 [MUST]** — The hash is the canonical TOML representation, not the original file bytes. Reformatting the same config produces the same hash.

**RFC0018-007 [MUST]** — Two checkpoints with the same `config_hash` are guaranteed to have been trained with the same logical config (up to env-var-injected overrides, which are also folded in).

---

## 9. Testing strategy

| ID | Name | Type | What it covers |
|----|------|------|----------------|
| TST-0018-DENY-001 | `unknown_field_rejected` | unit | RFC0018-001 |
| TST-0018-VAL-001 | `numeric_out_of_range_rejected` | unit | §7 |
| TST-0018-VAL-002 | `cross_field_validation` | unit | RFC0018-005 |
| TST-0018-OVERRIDE-001 | `cli_set_overrides_file` | unit | §6.2 |
| TST-0018-OVERRIDE-002 | `env_overrides_file_but_not_cli` | unit | RFC0018-004 |
| TST-0018-HASH-001 | `config_hash_stable_under_reformat` | unit | RFC0018-006 |
| TST-0018-HASH-002 | `config_hash_differs_on_meaningful_change` | unit | RFC0018-007 |
| TST-0018-SCHEMA-001 | `schema_version_mismatch_explicit_error` | unit | RFC0018-002 |
| TST-0018-FIXTURE-001 | `pusht_toml_loads_and_validates` | unit | §5.1 |
| TST-0018-FIXTURE-002 | `so100_toml_loads_and_validates` | unit | §5.2 |

---

## 10. Operational considerations

### 10.1 Observability

Config is dumped to stdout (lossless canonical form) at the start of every run. Sensitive fields (none currently in the config; tokens live in Secrets) would be redacted.

Config hash is emitted in the provenance preamble (RFC 0005 §4) and in every metric/log.

### 10.2 Runbook

- **"My config doesn't load."** — the error message gives the field name and reason. Fix and retry.
- **"My override doesn't apply."** — verify the precedence (CLI `--set` wins). Check the dotted-path syntax.
- **"Config hash changed unexpectedly."** — likely a field default changed in a Burn/lewm-rs version bump. Inspect with `lewm-train train --dry-run --print-config`.

---

## 11. Performance considerations

None. Config parsing is one-shot.

---

## 12. Security considerations

- Tokens are **not** in the config TOML; they live in the Secrets file or env vars.
- Config files are not signed; we trust the file path.

---

## 13. Alternatives considered

- **A1 — YAML instead of TOML.** Rejected: TOML is simpler, less ambiguous, has explicit datetime/number support.
- **A2 — JSON.** Rejected: not human-friendly for editing.
- **A3 — Procedural macro to generate the schema.** `serde::Deserialize` + `validator::Validate` is sufficient.
- **A4 — JSON Schema generation for IDE autocomplete.** Considered. Out of scope for v1; would be nice.

---

## 14. Acceptance criteria

- [ ] All TST-0018-* pass.
- [ ] `configs/pusht.toml` and `configs/so100.toml` load without modification on a fresh build.
- [ ] Config hash dump on run start.

---

## 15. Risks

| ID | Risk | L | I | Mitigation |
|----|------|---|---|-----------|
| R-1 | Schema drift between minor versions | M | M | `schema_version` mandatory; migration guide in CHANGELOG |
| R-2 | Cross-field validation incomplete | M | L | Add as new bugs found |
| R-3 | TOML 1.1 ambiguities | L | L | Pin `toml = "0.8"` |

---

## 16. Open questions

OQ-2018-1 — Whether to generate a JSON Schema for editor integration. Defer to v2.

---

## 17. Change log

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0.0 | 2026-05-12 | Abdel | Initial accepted version. |

*End of RFC 0018.*
