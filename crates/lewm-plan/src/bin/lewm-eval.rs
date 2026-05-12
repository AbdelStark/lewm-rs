//! Command-line entry point for `PushT` and `SO-100` evaluation.

use std::path::PathBuf;

use chrono::Utc;
use clap::{Args, Parser, Subcommand};
use lewm_data::ImagePreprocessor;
use lewm_plan::{
    EvalError, MockPushtRpc, PushtConfigFile, PushtEvalConfig, PushtEvalReport, PushtEvaluator,
    StaticPushtPlanner, SubprocessPushtRpc, render_pusht_report, write_pusht_artifacts,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()?;
    Ok(())
}

fn run() -> Result<(), EvalError> {
    let cli = Cli::parse();
    match &cli.command {
        Command::Pusht(args) => run_pusht(args, &cli),
        Command::Report(args) => run_report(args),
        Command::So100 => Err(EvalError::InvalidConfig(
            "SO-100 evaluation is tracked by the RFC 0006 SO-100 issue set".to_owned(),
        )),
    }
}

#[derive(Debug, Parser)]
#[command(name = "lewm-eval", about = "Run LeWM evaluation protocols")]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Burn record (.mpk) of the model to evaluate.
    #[arg(long, global = true)]
    checkpoint: Option<PathBuf>,

    /// Eval output directory.
    #[arg(long, global = true)]
    output_dir: Option<PathBuf>,

    /// Override the global seed.
    #[arg(long, global = true)]
    seed: Option<u64>,

    /// Override episode count.
    #[arg(long, global = true)]
    episodes: Option<usize>,

    /// Override per-episode step cap.
    #[arg(long, global = true)]
    max_steps: Option<u32>,

    /// Device selector reserved for the model-backed planner adapter.
    #[arg(long, global = true, default_value = "cuda:0")]
    device: String,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the `PushT` 50-episode protocol.
    Pusht(PushtArgs),
    /// Render an eval report from a JSON results file.
    Report(ReportArgs),
    /// Run the SO-100 5-episode protocol.
    So100,
}

#[derive(Debug, Args)]
struct PushtArgs {
    /// `PushT` eval config path.
    #[arg(long, default_value = "configs/pusht.toml")]
    config: PathBuf,

    /// Use the in-process deterministic simulator mock.
    #[arg(long)]
    mock_rpc: bool,

    /// Python executable for the JSON-RPC sidecar.
    #[arg(long, default_value = "python3")]
    python: String,

    /// Path to `python/pusht_runner.py`.
    #[arg(long, default_value = "python/pusht_runner.py")]
    runner: PathBuf,

    /// Ask the Python sidecar to run in mock mode.
    #[arg(long)]
    runner_mock: bool,

    /// Mock sidecar success step.
    #[arg(long, default_value_t = 5)]
    mock_success_after: u32,
}

#[derive(Debug, Args)]
struct ReportArgs {
    /// JSON results file produced by `lewm-eval pusht`.
    #[arg(long)]
    results: PathBuf,

    /// Markdown output path.
    #[arg(long)]
    output: PathBuf,
}

fn run_pusht(args: &PushtArgs, cli: &Cli) -> Result<(), EvalError> {
    validate_reserved_global_flags(cli)?;
    let config_file = PushtConfigFile::from_toml_path(&args.config)?;
    let config = override_config(config_file.eval, cli)?;
    let action_norm = config.action_normalizer()?;
    let output_dir = cli
        .output_dir
        .clone()
        .unwrap_or_else(|| default_output_dir("pusht"));

    if args.mock_rpc {
        let mut evaluator = PushtEvaluator::new(
            StaticPushtPlanner::zeros(config.action_dim()),
            ImagePreprocessor::default(),
            action_norm,
            MockPushtRpc::new(args.mock_success_after),
            config,
        )?;
        let report = evaluator.run()?;
        write_pusht_artifacts(&report, output_dir)?;
        return Ok(());
    }

    let mut evaluator = PushtEvaluator::new(
        StaticPushtPlanner::zeros(config.action_dim()),
        ImagePreprocessor::default(),
        action_norm,
        SubprocessPushtRpc::spawn_python_runner(&args.python, &args.runner, args.runner_mock)?,
        config,
    )?;
    let report = evaluator.run()?;
    write_pusht_artifacts(&report, output_dir)
}

fn validate_reserved_global_flags(cli: &Cli) -> Result<(), EvalError> {
    if cli.device.trim().is_empty() {
        return Err(EvalError::InvalidConfig(
            "--device must not be empty".to_owned(),
        ));
    }
    if let Some(checkpoint) = &cli.checkpoint {
        if checkpoint.as_os_str().is_empty() {
            return Err(EvalError::InvalidConfig(
                "--checkpoint must not be an empty path".to_owned(),
            ));
        }
    }
    Ok(())
}

fn override_config(mut config: PushtEvalConfig, cli: &Cli) -> Result<PushtEvalConfig, EvalError> {
    if let Some(seed) = cli.seed {
        config.seed = seed;
    }
    if let Some(max_steps) = cli.max_steps {
        config.max_steps_per_episode = max_steps;
    }
    if let Some(episodes) = cli.episodes {
        if episodes == 0 {
            return Err(EvalError::InvalidConfig(
                "--episodes must be greater than zero".to_owned(),
            ));
        }
        if episodes > config.episode_ids.len() {
            return Err(EvalError::InvalidConfig(format!(
                "--episodes requested {episodes} episodes but config only pins {}",
                config.episode_ids.len()
            )));
        }
        config.episode_ids.truncate(episodes);
    }
    config.validate()?;
    Ok(config)
}

fn run_report(args: &ReportArgs) -> Result<(), EvalError> {
    let bytes =
        std::fs::read(&args.results).map_err(|source| EvalError::io(&args.results, source))?;
    let report: PushtEvalReport = serde_json::from_slice(&bytes)
        .map_err(|source| EvalError::json("parsing PushT results JSON", source))?;
    let markdown = render_pusht_report(&report);
    std::fs::write(&args.output, markdown).map_err(|source| EvalError::io(&args.output, source))
}

fn default_output_dir(prefix: &str) -> PathBuf {
    PathBuf::from("out-eval").join(format!("{prefix}-{}", Utc::now().format("%Y%m%d-%H%M%S")))
}
