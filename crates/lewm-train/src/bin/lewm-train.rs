//! Command-line entry point for training and training-adjacent operations.
//!
//! The command surface is implemented by the RFC 0005 training issue set.

use std::{
    error::Error,
    fmt,
    io::{self, Write},
    path::PathBuf,
};

use clap::{Parser, Subcommand};
use serde::Deserialize;

#[cfg(feature = "cuda")]
const DEFAULT_DEVICE: &str = "cuda:0";
#[cfg(all(not(feature = "cuda"), feature = "metal"))]
const DEFAULT_DEVICE: &str = "metal:0";
#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
const DEFAULT_DEVICE: &str = "cpu";

const DEFAULT_LOG_LEVEL: &str = "info";
const DEFAULT_OUTPUT_DIR: &str = "./out/<run_id>";
const DEFAULT_SEED: u64 = 0;
const UNKNOWN_BUILD_VALUE: &str = "unknown";
const UNKNOWN_CONFIG_HASH: &str = "000000000000";

#[derive(Clone, Debug, Deserialize, Eq, Parser, PartialEq)]
#[command(
    name = "lewm-train",
    version,
    about = "Training and training-adjacent operations for LeWM.",
    propagate_version = true
)]
struct Cli {
    #[arg(long, global = true, value_name = "PATH")]
    config: Option<PathBuf>,

    #[arg(
        long = "set",
        global = true,
        value_name = "KEY=VALUE",
        value_parser = parse_config_override
    )]
    overrides: Vec<ConfigOverride>,

    #[arg(
        long,
        global = true,
        value_name = "PATH",
        default_value = DEFAULT_OUTPUT_DIR
    )]
    output_dir: PathBuf,

    #[arg(long, global = true)]
    resume_if_present: bool,

    #[arg(long, global = true, value_name = "INT")]
    seed: Option<u64>,

    #[arg(long, global = true, value_name = "DEVICE", default_value = DEFAULT_DEVICE)]
    device: String,

    #[arg(
        long,
        global = true,
        value_name = "LEVEL",
        default_value = DEFAULT_LOG_LEVEL
    )]
    log_level: String,

    #[arg(long, global = true)]
    dry_run: bool,

    #[arg(long, global = true, value_name = "INT")]
    max_steps: Option<u64>,

    #[command(subcommand)]
    command: Command,
}

impl Cli {
    fn provenance_preamble(
        &self,
        git_short_sha: &str,
        build_date: &str,
        config_hash: &str,
    ) -> String {
        format_provenance_preamble(
            env!("CARGO_PKG_VERSION"),
            git_short_sha,
            build_date,
            self.seed.unwrap_or(DEFAULT_SEED),
            &self.device,
            config_hash,
        )
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Subcommand)]
#[serde(rename_all = "kebab-case")]
enum Command {
    /// Run the full pipeline from initialization through upload.
    Train(TrainArgs),
    /// Run a short local smoke on `NdArray` CPU.
    Smoke(SmokeArgs),
    /// Run the reference parity harness without training.
    Parity(ParityArgs),
    /// Evaluate a checkpoint without training.
    Eval(EvalArgs),
    /// Convert `PyTorch` reference weights to a Burn record.
    Convert(ConvertArgs),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, clap::Args)]
struct TrainArgs {
    #[arg(long, value_name = "PATH")]
    data_dir: Option<PathBuf>,

    #[arg(long, value_name = "ENV")]
    hf_token: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, clap::Args)]
struct SmokeArgs {
    #[arg(long, default_value_t = 50, value_name = "INT")]
    steps: u64,

    #[arg(long, default_value_t = 4, value_name = "INT")]
    batch_size: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, clap::Args)]
struct ParityArgs {
    #[arg(long, value_name = "PATH")]
    reference: PathBuf,

    #[arg(long, value_name = "PATH")]
    dump_dir: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, clap::Args)]
struct EvalArgs {
    #[arg(long, value_name = "PATH")]
    checkpoint: PathBuf,

    #[arg(long, default_value_t = 50, value_name = "INT")]
    episodes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, clap::Args)]
struct ConvertArgs {
    #[arg(long, value_name = "PATH")]
    pt: PathBuf,

    #[arg(long, value_name = "PATH")]
    out: PathBuf,

    #[arg(long, value_name = "PATH")]
    intermediate: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct ConfigOverride {
    key: String,
    value: String,
}

#[derive(Debug, Eq, PartialEq)]
struct OverrideParseError(String);

impl fmt::Display for OverrideParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for OverrideParseError {}

fn main() -> Result<(), CliError> {
    let cli = Cli::parse();
    write_preamble(io::stdout(), &cli)?;
    Ok(())
}

fn write_preamble(mut writer: impl Write, cli: &Cli) -> Result<(), CliError> {
    let line = cli.provenance_preamble(
        option_env!("LEWM_GIT_SHA").unwrap_or(UNKNOWN_BUILD_VALUE),
        option_env!("LEWM_BUILD_DATE").unwrap_or(UNKNOWN_BUILD_VALUE),
        UNKNOWN_CONFIG_HASH,
    );
    writeln!(writer, "{line}")?;
    Ok(())
}

fn parse_config_override(raw: &str) -> Result<ConfigOverride, OverrideParseError> {
    let Some((key, value)) = raw.split_once('=') else {
        return Err(OverrideParseError(
            "override must use key.path=value syntax".to_string(),
        ));
    };

    validate_override_key(key)?;
    validate_override_value(value)?;

    Ok(ConfigOverride {
        key: key.to_string(),
        value: value.to_string(),
    })
}

fn validate_override_value(value: &str) -> Result<(), OverrideParseError> {
    if value.is_empty() {
        return Err(OverrideParseError(
            "override value must not be empty".to_string(),
        ));
    }

    let wrapped = format!("value = {value}");
    let parsed: OverrideValue = toml::from_str(&wrapped).map_err(|source| {
        OverrideParseError(format!("override value must be valid TOML: {source}"))
    })?;

    if matches!(parsed.value, toml::Value::Table(_)) {
        return Err(OverrideParseError(
            "override value must be a TOML scalar or array".to_string(),
        ));
    }

    Ok(())
}

fn validate_override_key(key: &str) -> Result<(), OverrideParseError> {
    if key.is_empty() {
        return Err(OverrideParseError(
            "override key must not be empty".to_string(),
        ));
    }

    if !key.contains('.') {
        return Err(OverrideParseError(
            "override key must be a dotted path".to_string(),
        ));
    }

    if key.split('.').any(str::is_empty) {
        return Err(OverrideParseError(
            "override key must not contain empty path segments".to_string(),
        ));
    }

    Ok(())
}

fn format_provenance_preamble(
    version: &str,
    git_short_sha: &str,
    build_date: &str,
    seed: u64,
    device: &str,
    config_hash: &str,
) -> String {
    format!(
        "lewm-train v{version} (git: {git_short_sha}, build: {build_date}); seed={seed}; device={device}; config_hash={config_hash}"
    )
}

#[derive(Debug, Deserialize)]
struct OverrideValue {
    value: toml::Value,
}

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CliError {}

impl From<io::Error> for CliError {
    fn from(source: io::Error) -> Self {
        Self(source.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn Error>>;

    #[test]
    fn cli_parse_train_default() -> TestResult {
        let cli = Cli::try_parse_from(["lewm-train", "train"])?;

        assert_eq!(cli.config, None);
        assert_eq!(cli.output_dir, PathBuf::from(DEFAULT_OUTPUT_DIR));
        assert_eq!(cli.seed, None);
        assert_eq!(cli.device, DEFAULT_DEVICE);
        assert_eq!(cli.log_level, DEFAULT_LOG_LEVEL);
        assert_eq!(cli.max_steps, None);
        assert!(!cli.resume_if_present);
        assert!(!cli.dry_run);
        assert_eq!(
            cli.command,
            Command::Train(TrainArgs {
                data_dir: None,
                hf_token: None
            })
        );

        Ok(())
    }

    #[test]
    fn cli_parse_smoke_overrides() -> TestResult {
        let cli = Cli::try_parse_from([
            "lewm-train",
            "--config",
            "configs/pusht.toml",
            "--output-dir",
            "out/smoke",
            "--device",
            "cpu",
            "--log-level",
            "debug",
            "--seed",
            "42",
            "--dry-run",
            "--max-steps",
            "10",
            "smoke",
            "--steps",
            "200",
            "--batch-size",
            "16",
        ])?;

        let Command::Smoke(args) = cli.command else {
            return Err("expected smoke subcommand".into());
        };

        assert_eq!(cli.config, Some(PathBuf::from("configs/pusht.toml")));
        assert_eq!(cli.output_dir, PathBuf::from("out/smoke"));
        assert_eq!(cli.device, "cpu");
        assert_eq!(cli.log_level, "debug");
        assert_eq!(cli.seed, Some(42));
        assert_eq!(cli.max_steps, Some(10));
        assert!(cli.dry_run);
        assert_eq!(args.steps, 200);
        assert_eq!(args.batch_size, 16);

        Ok(())
    }

    #[test]
    fn cli_set_key_value_override() -> TestResult {
        let cli = Cli::try_parse_from([
            "lewm-train",
            "train",
            "--set",
            "training.lr_peak=1.0e-4",
            "--set",
            "training.betas=[0.9,0.99]",
        ])?;

        assert_eq!(
            cli.overrides,
            vec![
                ConfigOverride {
                    key: "training.lr_peak".to_string(),
                    value: "1.0e-4".to_string()
                },
                ConfigOverride {
                    key: "training.betas".to_string(),
                    value: "[0.9,0.99]".to_string()
                }
            ]
        );

        Ok(())
    }

    #[test]
    fn cli_resume_if_present_detects_dir() -> TestResult {
        let cli = Cli::try_parse_from([
            "lewm-train",
            "--output-dir",
            "out/resume-run",
            "--resume-if-present",
            "train",
        ])?;

        assert!(cli.resume_if_present);
        assert_eq!(cli.output_dir, PathBuf::from("out/resume-run"));

        Ok(())
    }

    #[test]
    fn cli_provenance_preamble_format() {
        let preamble =
            format_provenance_preamble("0.1.0", "abc1234", "2026-05-12", 7, "cpu", "fedcba987654");

        assert_eq!(
            preamble,
            "lewm-train v0.1.0 (git: abc1234, build: 2026-05-12); seed=7; device=cpu; config_hash=fedcba987654"
        );
    }

    #[test]
    fn cli_rejects_invalid_override_syntax() {
        assert!(Cli::try_parse_from(["lewm-train", "train", "--set", "training.lr_peak"]).is_err());
        assert!(Cli::try_parse_from(["lewm-train", "train", "--set", "=1e-4"]).is_err());
        assert!(Cli::try_parse_from(["lewm-train", "train", "--set", "training=1e-4"]).is_err());
        assert!(
            Cli::try_parse_from(["lewm-train", "train", "--set", "training..lr=1e-4"]).is_err()
        );
        assert!(
            Cli::try_parse_from(["lewm-train", "train", "--set", "training.lr_peak="]).is_err()
        );
        assert!(
            Cli::try_parse_from([
                "lewm-train",
                "train",
                "--set",
                "training.lr_peak=not a scalar"
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from(["lewm-train", "train", "--set", "training.schedule={lr=1}"])
                .is_err()
        );
    }

    #[test]
    fn cli_accepts_toml_override_values() -> TestResult {
        let cli = Cli::try_parse_from([
            "lewm-train",
            "train",
            "--set",
            "training.note=\"a=b\"",
            "--set",
            "training.enabled=true",
        ])?;

        assert_eq!(
            cli.overrides,
            vec![
                ConfigOverride {
                    key: "training.note".to_string(),
                    value: "\"a=b\"".to_string()
                },
                ConfigOverride {
                    key: "training.enabled".to_string(),
                    value: "true".to_string()
                }
            ]
        );

        Ok(())
    }
}
