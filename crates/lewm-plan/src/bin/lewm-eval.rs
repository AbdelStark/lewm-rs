//! Command-line entry point for `PushT` and `SO-100` evaluation.

use std::{collections::BTreeMap, fs, path::PathBuf};

use chrono::Utc;
use clap::{Args, Parser, Subcommand};
use lewm_data::ImagePreprocessor;
use lewm_plan::{
    EvalError, LatentVector, MockPushtRpc, PushtConfigFile, PushtEvalConfig, PushtEvalReport,
    PushtEvaluator, RecordedRolloutModel, So100Episode, So100EvalConfig, So100EvalReport,
    So100Evaluator, StaticPushtPlanner, SubprocessPushtRpc, render_pusht_report,
    render_report_markdown, write_pusht_artifacts, write_so100_outputs,
};
use serde::Deserialize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()?;
    Ok(())
}

fn run() -> Result<(), EvalError> {
    let cli = Cli::parse();
    match &cli.command {
        Command::Pusht(args) => run_pusht(args, &cli),
        Command::So100(args) => run_so100(args, &cli),
        Command::Report(args) => run_report(args),
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

    /// Device selector reserved for model-backed adapters.
    #[arg(long, global = true, default_value = "cuda:0")]
    device: String,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the `PushT` 50-episode protocol.
    Pusht(PushtArgs),
    /// Run the SO-100 latent-rollout protocol.
    So100(So100Args),
    /// Render an eval report from a JSON results file.
    Report(ReportArgs),
}

#[derive(Debug, Args)]
struct PushtArgs {
    /// `PushT` eval config path.
    #[arg(long, default_value = "configs/pusht_eval.toml")]
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
struct So100Args {
    /// Encoded latent fixture with target and predicted latent trajectories.
    #[arg(long)]
    encoded_episodes: PathBuf,

    /// Number of latent history entries used to seed the rollout.
    #[arg(long, default_value_t = 3)]
    history_size: usize,

    /// Spearman floor for pass/partial/null classification.
    #[arg(long, default_value_t = 0.6)]
    spearman_floor: f64,
}

#[derive(Debug, Args)]
struct ReportArgs {
    /// JSON results file produced by `lewm-eval pusht`.
    #[arg(long)]
    results: Option<PathBuf>,

    /// JSON results file produced by `lewm-eval so100`.
    #[arg(long)]
    so100_results: Option<PathBuf>,

    /// Backward-compatible alias for `--so100-results`.
    #[arg(long)]
    results_json: Option<PathBuf>,

    /// Markdown output path.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct EncodedInput {
    episodes: Vec<EncodedEpisode>,
}

#[derive(Debug, Deserialize)]
struct EncodedEpisode {
    episode_id: u32,
    target_latents: Vec<LatentVector>,
    expert_actions: Vec<Vec<f64>>,
    predicted_latents: Vec<LatentVector>,
}

fn run_pusht(args: &PushtArgs, cli: &Cli) -> Result<(), EvalError> {
    validate_reserved_global_flags(cli)?;
    let config_file = PushtConfigFile::from_toml_path(&args.config)?;
    let config = override_pusht_config(config_file.eval, cli)?;
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

fn run_so100(args: &So100Args, cli: &Cli) -> Result<(), EvalError> {
    validate_reserved_global_flags(cli)?;
    if cli.episodes.is_some() {
        return Err(EvalError::InvalidConfig(
            "--episodes is reserved for model-backed SO-100 eval".to_owned(),
        ));
    }
    if cli.max_steps.is_some() {
        return Err(EvalError::InvalidConfig(
            "--max-steps is reserved for model-backed SO-100 eval".to_owned(),
        ));
    }

    let input = read_encoded_input(&args.encoded_episodes)?;
    let episodes = input
        .episodes
        .iter()
        .map(|episode| So100Episode {
            episode_id: episode.episode_id,
            target_latents: episode.target_latents.clone(),
            expert_actions: episode.expert_actions.clone(),
        })
        .collect::<Vec<_>>();
    let predictions = input
        .episodes
        .iter()
        .map(|episode| (episode.episode_id, episode.predicted_latents.clone()))
        .collect::<BTreeMap<_, _>>();
    let model = RecordedRolloutModel::new(predictions);
    let config = So100EvalConfig {
        history_size: args.history_size,
        spearman_floor: args.spearman_floor,
    };
    let output_dir = cli
        .output_dir
        .clone()
        .unwrap_or_else(|| default_output_dir("so100"));
    let mut evaluator = So100Evaluator::new(model, config);
    let run = evaluator.run(&episodes)?;
    write_so100_outputs(&output_dir, &run)
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

fn override_pusht_config(
    mut config: PushtEvalConfig,
    cli: &Cli,
) -> Result<PushtEvalConfig, EvalError> {
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
    let selected = [
        args.results.as_ref().map(|path| ("pusht", path)),
        args.so100_results.as_ref().map(|path| ("so100", path)),
        args.results_json.as_ref().map(|path| ("so100", path)),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();

    if selected.len() != 1 {
        return Err(EvalError::InvalidConfig(
            "report requires exactly one of --results, --so100-results, or --results-json"
                .to_owned(),
        ));
    }

    let (kind, path) = selected[0];
    let markdown = match kind {
        "pusht" => {
            let bytes = fs::read(path).map_err(|source| EvalError::io(path, source))?;
            let report: PushtEvalReport = serde_json::from_slice(&bytes)
                .map_err(|source| EvalError::json("parsing PushT results JSON", source))?;
            render_pusht_report(&report)
        },
        "so100" => {
            let text = fs::read_to_string(path).map_err(|source| EvalError::io(path, source))?;
            let report = serde_json::from_str::<So100EvalReport>(&text)
                .map_err(|source| EvalError::json_decode(path, source))?;
            render_report_markdown(&report)
        },
        _ => unreachable!("validated report kind"),
    };
    fs::write(&args.output, markdown).map_err(|source| EvalError::io(&args.output, source))
}

fn read_encoded_input(path: &PathBuf) -> Result<EncodedInput, EvalError> {
    let text = fs::read_to_string(path).map_err(|source| EvalError::io(path, source))?;
    serde_json::from_str(&text).map_err(|source| EvalError::json_decode(path, source))
}

fn default_output_dir(prefix: &str) -> PathBuf {
    PathBuf::from("out-eval").join(format!("{prefix}-{}", Utc::now().format("%Y%m%d-%H%M%S")))
}
