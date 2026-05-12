//! Command-line entry point for `PushT` and `SO-100` evaluation.

use std::{collections::BTreeMap, fs, path::PathBuf};

use clap::{Args, Parser, Subcommand};
use lewm_plan::{
    EvalError, LatentVector, RecordedRolloutModel, So100Episode, So100EvalConfig, So100EvalReport,
    So100Evaluator, render_report_markdown, write_so100_outputs,
};
use serde::Deserialize;

#[derive(Debug, Parser)]
#[command(name = "lewm-eval")]
#[command(about = "Run lewm-rs evaluation protocols")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the SO-100 latent-rollout protocol.
    So100(So100Args),
    /// Run the `PushT` planning protocol.
    Pusht(UnsupportedArgs),
    /// Render an eval report from a JSON results file.
    Report(ReportArgs),
}

#[derive(Debug, Args)]
struct So100Args {
    /// Encoded latent fixture with target and predicted latent trajectories.
    #[arg(long)]
    encoded_episodes: PathBuf,
    /// Output directory for `results.json`, `report.md`, and `latent_traces.parquet`.
    #[arg(long, default_value = "./out-eval/so100")]
    output_dir: PathBuf,
    /// Number of latent history entries used to seed the rollout.
    #[arg(long, default_value_t = 3)]
    history_size: usize,
    /// Spearman floor for pass/partial/null classification.
    #[arg(long, default_value_t = 0.6)]
    spearman_floor: f64,
}

#[derive(Debug, Args)]
struct ReportArgs {
    /// Existing SO-100 results.json file.
    #[arg(long)]
    results_json: PathBuf,
    /// Destination Markdown report path.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Debug, Args)]
struct UnsupportedArgs {
    /// Checkpoint path reserved for the future model-backed evaluator.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
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

fn main() -> Result<(), EvalError> {
    let cli = Cli::parse();
    match cli.command {
        Command::So100(args) => run_so100(&args),
        Command::Pusht(_) => Err(EvalError::InvalidInput(
            "PushT eval is not implemented in this milestone".to_owned(),
        )),
        Command::Report(args) => render_report(&args),
    }
}

fn run_so100(args: &So100Args) -> Result<(), EvalError> {
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
    let mut evaluator = So100Evaluator::new(model, config);
    let run = evaluator.run(&episodes)?;
    write_so100_outputs(&args.output_dir, &run)
}

fn render_report(args: &ReportArgs) -> Result<(), EvalError> {
    let text = fs::read_to_string(&args.results_json)
        .map_err(|source| EvalError::io(&args.results_json, source))?;
    let report = serde_json::from_str::<So100EvalReport>(&text)
        .map_err(|source| EvalError::json_decode(&args.results_json, source))?;
    let markdown = render_report_markdown(&report);
    fs::write(&args.output, markdown).map_err(|source| EvalError::io(&args.output, source))
}

fn read_encoded_input(path: &PathBuf) -> Result<EncodedInput, EvalError> {
    let text = fs::read_to_string(path).map_err(|source| EvalError::io(path, source))?;
    serde_json::from_str(&text).map_err(|source| EvalError::json_decode(path, source))
}
