//! Model repository README rendering.

use std::fmt::Write as _;

/// Upstream `LeWorldModel` citation block required by NFR-041.
pub const LEWM_CITATION_BIBTEX: &str = r"@article{maes_lelidec2026lewm,
  title  = {LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author = {Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal = {arXiv preprint},
  year   = {2026}
}";

/// `lewm-rs` implementation citation block.
pub const LEWM_RS_CITATION_BIBTEX: &str = r"@software{lewm_rs_2026,
  author = {Abdel},
  title  = {lewm-rs: A Pure-Rust Reproduction of LeWorldModel},
  year   = {2026},
  url    = {https://github.com/AbdelStark/lewm-rs}
}";

/// Metadata required to render an RFC 0010 model card.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelCardMetadata {
    /// Hub repository name, for example `abdelstark/lewm-rs-pusht`.
    pub repo_name: Option<String>,
    /// Dataset tag used in Hub frontmatter, for example `pusht`.
    pub dataset_tag: Option<String>,
    /// Primary source dataset id.
    pub primary_dataset: Option<String>,
    /// Human-readable dataset name used in prose.
    pub dataset_display: Option<String>,
    /// Optional warm-start base model id.
    pub base_model: Option<String>,
    /// Burn version used for training.
    pub burn_version: Option<String>,
    /// Hardware flavor used for training.
    pub hardware: Option<String>,
    /// Headline metric name shown in the result section.
    pub headline_metric: Option<String>,
    /// Headline metric value shown in the result section.
    pub headline_value: Option<String>,
    /// Planning success rate for Hub model-index metadata.
    pub success_rate: Option<f64>,
    /// CPU planning latency in milliseconds for Hub model-index metadata.
    pub latency_ms: Option<f64>,
    /// Encoder parity bound shown in the result section.
    pub parity_encoder: Option<String>,
    /// Predictor parity bound shown in the result section.
    pub parity_predictor: Option<String>,
    /// Laptop CPU inference latency in milliseconds.
    pub tract_laptop_ms: Option<f64>,
    /// Total training cost in USD.
    pub cost_usd: Option<f64>,
    /// Training report URL.
    pub report_url: Option<String>,
    /// Source git SHA for the training run.
    pub git_sha: Option<String>,
    /// Rust toolchain used for the training run.
    pub rust_version: Option<String>,
    /// Configuration hash for the training run.
    pub config_hash: Option<String>,
    /// Training run id.
    pub run_id: Option<String>,
    /// Training wall time.
    pub wall_time: Option<String>,
}

/// Model-card rendering failures.
#[derive(Debug, thiserror::Error, Eq, PartialEq)]
pub enum ModelCardError {
    /// A required metadata field was absent.
    #[error("model-card metadata missing required field: {0}")]
    MissingField(&'static str),

    /// A numeric field was NaN or infinite.
    #[error("model-card metadata field must be finite: {0}")]
    InvalidNumber(&'static str),

    /// A template placeholder survived rendering.
    #[error("model-card template left placeholder unrendered: {0}")]
    UnrenderedPlaceholder(String),
}

/// Render an RFC 0010 model card.
///
/// # Errors
///
/// Returns an error when any required metadata field is missing, when numeric
/// values are not finite, or when the template leaves an unfilled placeholder.
pub fn render(metadata: &ModelCardMetadata) -> Result<String, ModelCardError> {
    let fields = ModelCardFields::from_metadata(metadata)?;
    let mut card = String::new();

    fields.write_frontmatter(&mut card)?;
    fields.write_body(&mut card)?;
    ensure_no_placeholders(&card)?;
    Ok(card)
}

#[derive(Debug)]
struct ModelCardFields<'a> {
    repo_name: &'a str,
    dataset_tag: &'a str,
    primary_dataset: &'a str,
    dataset_display: &'a str,
    base_model: String,
    burn_version: &'a str,
    hardware: &'a str,
    headline_metric: &'a str,
    headline_value: &'a str,
    success_rate: String,
    latency_ms: String,
    parity_encoder: &'a str,
    parity_predictor: &'a str,
    tract_laptop_ms: String,
    cost_usd: String,
    report_url: &'a str,
    git_sha: &'a str,
    rust_version: &'a str,
    config_hash: &'a str,
    run_id: &'a str,
    wall_time: &'a str,
}

impl<'a> ModelCardFields<'a> {
    fn from_metadata(metadata: &'a ModelCardMetadata) -> Result<Self, ModelCardError> {
        Ok(Self {
            repo_name: required_str("repo_name", metadata.repo_name.as_deref())?,
            dataset_tag: required_str("dataset_tag", metadata.dataset_tag.as_deref())?,
            primary_dataset: required_str("primary_dataset", metadata.primary_dataset.as_deref())?,
            dataset_display: required_str("dataset_display", metadata.dataset_display.as_deref())?,
            base_model: render_base_model(metadata.base_model.as_deref()),
            burn_version: required_str("burn_version", metadata.burn_version.as_deref())?,
            hardware: required_str("hardware", metadata.hardware.as_deref())?,
            headline_metric: required_str("headline_metric", metadata.headline_metric.as_deref())?,
            headline_value: required_str("headline_value", metadata.headline_value.as_deref())?,
            success_rate: format_number(required_f64("success_rate", metadata.success_rate)?),
            latency_ms: format_number(required_f64("latency_ms", metadata.latency_ms)?),
            parity_encoder: required_str("parity_encoder", metadata.parity_encoder.as_deref())?,
            parity_predictor: required_str(
                "parity_predictor",
                metadata.parity_predictor.as_deref(),
            )?,
            tract_laptop_ms: format_number(required_f64(
                "tract_laptop_ms",
                metadata.tract_laptop_ms,
            )?),
            cost_usd: format_number(required_f64("cost_usd", metadata.cost_usd)?),
            report_url: required_str("report_url", metadata.report_url.as_deref())?,
            git_sha: required_str("git_sha", metadata.git_sha.as_deref())?,
            rust_version: required_str("rust_version", metadata.rust_version.as_deref())?,
            config_hash: required_str("config_hash", metadata.config_hash.as_deref())?,
            run_id: required_str("run_id", metadata.run_id.as_deref())?,
            wall_time: required_str("wall_time", metadata.wall_time.as_deref())?,
        })
    }

    fn write_frontmatter(&self, card: &mut String) -> Result<(), ModelCardError> {
        write!(
            card,
            r"---
library_name: burn
license: apache-2.0
tags:
  - jepa
  - world-model
  - robotics
  - rust
  - burn
  - lewm
  - {dataset_tag_yaml}
datasets:
  - {primary_dataset_yaml}
metrics:
  - planning_success_rate
  - latent_rollout_mse
  - spearman_rank_correlation
{base_model}language:
  - en
pipeline_tag: robotics
model-index:
  - name: {repo_name_yaml}
    results:
      - task:
          type: world-model-planning
          name: PushT planning
        dataset:
          name: lewm-pusht
          type: {primary_dataset_yaml}
        metrics:
          - type: planning_success_rate
            value: {success_rate}
            name: success rate
          - type: latency_per_plan_step_ms
            value: {latency_ms}
            name: CPU plan latency
---

",
            base_model = self.base_model,
            dataset_tag_yaml = yaml_double_quote(self.dataset_tag),
            latency_ms = self.latency_ms,
            primary_dataset_yaml = yaml_double_quote(self.primary_dataset),
            repo_name_yaml = yaml_double_quote(self.repo_name),
            success_rate = self.success_rate,
        )
        .map_err(|_| formatting_error())
    }

    fn write_body(&self, card: &mut String) -> Result<(), ModelCardError> {
        write!(
            card,
            r#"# {repo_name}

Pure-Rust reproduction of LeWorldModel ({dataset_display}). Trained with Burn
{burn_version} on a single NVIDIA {hardware} GPU on Hugging Face Jobs.

## Result

- **Headline metric**: {headline_metric}: {headline_value}
- **Parity vs reference** (epoch 10 vs `quentinll/lewm-pusht`):
    - encoder CLS L_inf : {parity_encoder}
    - predictor L_inf   : {parity_predictor}
- **CPU inference** (Tract, laptop): {tract_laptop_ms} ms per planning cost computation
- **Total training cost**: {cost_usd} USD

## How to use

For Rust inference on CPU:

```bash
cargo install --git https://github.com/AbdelStark/lewm-rs lewm-infer
hf download {repo_name} --local-dir ckpt
lewm-infer plan --checkpoint-dir ckpt --start start.png --goal goal.png
```

For Python loading via Safetensors mirror:

```python
from safetensors.torch import load_file
weights = load_file("step_0014400.safetensors")
```

## Training details

See training report: `{report_url}`.

## Citation

```bibtex
{citation_lewm}
{citation_lewm_rs}
```

## Provenance

| Field | Value |
|-------|-------|
| git SHA | {git_sha} |
| Burn version | {burn_version} |
| Rust toolchain | {rust_version} |
| Config hash | {config_hash} |
| Run id | {run_id} |
| Hardware | {hardware} |
| Wall time | {wall_time} |
| Cost | {cost_usd} USD |

## License

Apache-2.0. See [LICENSE](https://github.com/AbdelStark/lewm-rs/blob/main/LICENSE).
"#,
            burn_version = self.burn_version,
            citation_lewm = LEWM_CITATION_BIBTEX,
            citation_lewm_rs = LEWM_RS_CITATION_BIBTEX,
            config_hash = self.config_hash,
            cost_usd = self.cost_usd,
            dataset_display = self.dataset_display,
            git_sha = self.git_sha,
            hardware = self.hardware,
            headline_metric = self.headline_metric,
            headline_value = self.headline_value,
            parity_encoder = self.parity_encoder,
            parity_predictor = self.parity_predictor,
            repo_name = self.repo_name,
            report_url = self.report_url,
            run_id = self.run_id,
            rust_version = self.rust_version,
            tract_laptop_ms = self.tract_laptop_ms,
            wall_time = self.wall_time,
        )
        .map_err(|_| formatting_error())
    }
}

fn formatting_error() -> ModelCardError {
    ModelCardError::UnrenderedPlaceholder("formatting failed".to_owned())
}

fn required_str<'a>(
    field: &'static str,
    value: Option<&'a str>,
) -> Result<&'a str, ModelCardError> {
    value
        .filter(|value| !value.trim().is_empty())
        .ok_or(ModelCardError::MissingField(field))
}

fn required_f64(field: &'static str, value: Option<f64>) -> Result<f64, ModelCardError> {
    let value = value.ok_or(ModelCardError::MissingField(field))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ModelCardError::InvalidNumber(field))
    }
}

fn render_base_model(base_model: Option<&str>) -> String {
    match base_model.filter(|value| !value.trim().is_empty()) {
        Some(base_model) => format!("base_model:\n  - {}\n", yaml_double_quote(base_model)),
        None => "base_model: null\n".to_owned(),
    }
}

fn yaml_double_quote(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    format!("\"{escaped}\"")
}

fn format_number(value: f64) -> String {
    let mut rendered = format!("{value:.6}");
    while rendered.contains('.') && rendered.ends_with('0') {
        rendered.pop();
    }
    if rendered.ends_with('.') {
        rendered.pop();
    }
    rendered
}

fn ensure_no_placeholders(card: &str) -> Result<(), ModelCardError> {
    if let Some(index) = card.find("{{") {
        let end = card[index..]
            .find("}}")
            .map_or(index + 2, |offset| index + offset + 2);
        return Err(ModelCardError::UnrenderedPlaceholder(
            card[index..end].to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_card_renders_all_placeholders() -> Result<(), Box<dyn std::error::Error>> {
        let card = render(&sample_metadata())?;

        assert!(!card.contains("{{"));
        assert!(!card.contains("}}"));
        assert!(card.contains("# abdelstark/lewm-rs-pusht"));
        assert!(card.contains("planning_success_rate: 0.82"));
        assert!(card.contains("| Config hash | cfg-123 |"));
        Ok(())
    }

    #[test]
    fn model_card_yaml_frontmatter_valid() -> Result<(), Box<dyn std::error::Error>> {
        let card = render(&sample_metadata())?;
        let frontmatter = frontmatter(&card)?;
        let yaml: serde_yaml_ng::Value = serde_yaml_ng::from_str(frontmatter)?;

        assert_eq!(yaml["library_name"], serde_yaml_ng::Value::from("burn"));
        assert_eq!(yaml["license"], serde_yaml_ng::Value::from("apache-2.0"));
        assert_eq!(
            yaml["datasets"][0],
            serde_yaml_ng::Value::from("quentinll/lewm-pusht")
        );
        assert!(yaml["model-index"].as_sequence().is_some());
        Ok(())
    }

    #[test]
    fn model_card_has_citation_block() -> Result<(), Box<dyn std::error::Error>> {
        let card = render(&sample_metadata())?;

        assert!(card.contains("@article{maes_lelidec2026lewm"));
        assert!(card.contains("@software{lewm_rs_2026"));
        Ok(())
    }

    #[test]
    fn model_card_missing_field_errors() -> Result<(), Box<dyn std::error::Error>> {
        let mut metadata = sample_metadata();
        metadata.repo_name = None;

        let Err(error) = render(&metadata) else {
            return Err("missing field should fail".into());
        };

        assert_eq!(error, ModelCardError::MissingField("repo_name"));
        Ok(())
    }

    fn frontmatter(card: &str) -> Result<&str, &'static str> {
        let mut parts = card.splitn(3, "---");
        if parts.next() != Some("") {
            return Err("frontmatter must start at beginning");
        }
        parts.next().ok_or("missing frontmatter")
    }

    fn sample_metadata() -> ModelCardMetadata {
        ModelCardMetadata {
            repo_name: Some("abdelstark/lewm-rs-pusht".to_owned()),
            dataset_tag: Some("pusht".to_owned()),
            primary_dataset: Some("quentinll/lewm-pusht".to_owned()),
            dataset_display: Some("PushT".to_owned()),
            base_model: Some("quentinll/lewm-pusht".to_owned()),
            burn_version: Some("0.20.1".to_owned()),
            hardware: Some("A10G".to_owned()),
            headline_metric: Some("planning_success_rate".to_owned()),
            headline_value: Some("0.82".to_owned()),
            success_rate: Some(0.82),
            latency_ms: Some(12.5),
            parity_encoder: Some("1e-5".to_owned()),
            parity_predictor: Some("2e-5".to_owned()),
            tract_laptop_ms: Some(24.0),
            cost_usd: Some(17.25),
            report_url: Some(
                "https://huggingface.co/abdelstark/lewm-rs-pusht/blob/main/reports/training.md"
                    .to_owned(),
            ),
            git_sha: Some("abc123".to_owned()),
            rust_version: Some("rustc 1.89.0".to_owned()),
            config_hash: Some("cfg-123".to_owned()),
            run_id: Some("20260512-143002-9f3a-abcd".to_owned()),
            wall_time: Some("2:13:00".to_owned()),
        }
    }
}
