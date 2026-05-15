//! Command-line entry point for CPU inference and planning demos.
//!
//! The command surface follows RFC 0007's `lewm-infer` plan, bench, serve, and
//! verify contract.

use std::fs::File;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Instant as StdInstant;

use clap::{Args, Parser, Subcommand};
use lewm_infer::eval::{
    DEFAULT_TOLERANCE, EvalError, EvalReport, ParityEvalInputs, run_parity_eval,
};
use lewm_infer::plan::{
    CpuCem, DEFAULT_HORIZON_PLAN, DEFAULT_N_CAND, DEFAULT_N_ELITE, DEFAULT_N_ITER, PlanError,
    cem_rng,
};
use lewm_infer::preprocess::{PreprocessError, preprocess_path};
#[cfg(any(feature = "burn-cpu", feature = "burn-cuda"))]
use lewm_infer::runner::{BackendKind, load_with_backend};
use lewm_infer::runner::{IMAGE_ELEMENT_COUNT, InferenceRunner, RunnerError, load};
use serde::Serialize;

fn main() {
    if let Err(error) = run() {
        let mut stderr = std::io::stderr().lock();
        let _ = writeln!(stderr, "error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), CliError> {
    let cli = Cli::parse();
    let checkpoint_dir = validate_global(&cli)?.to_path_buf();
    let action_dim = cli.action_dim;
    let backend = cli.backend.as_deref();
    let safetensors = cli.safetensors.clone();
    let clock = WallClock;
    configure_threads(cli.threads)?;

    match cli.command {
        Commands::Plan(args) => run_plan(
            &checkpoint_dir,
            action_dim,
            backend,
            safetensors.as_deref(),
            &args,
            &clock,
        ),
        Commands::Bench(args) => run_bench(
            &checkpoint_dir,
            action_dim,
            backend,
            safetensors.as_deref(),
            &args,
            &clock,
        ),
        Commands::Serve(args) => run_serve(&checkpoint_dir, backend, safetensors.as_deref(), &args),
        Commands::Verify(args) => run_verify(
            &checkpoint_dir,
            backend,
            safetensors.as_deref(),
            &args,
            &clock,
        ),
        Commands::Eval(args) => run_eval(
            &checkpoint_dir,
            action_dim,
            backend,
            safetensors.as_deref(),
            &args,
        ),
    }
}

fn build_runner(
    checkpoint_dir: &Path,
    backend: Option<&str>,
    safetensors: Option<&Path>,
) -> Result<Box<dyn InferenceRunner>, CliError> {
    let backend_name = backend.unwrap_or("tract-onnx");
    if backend_name == "tract" || backend_name == "tract-onnx" || backend_name == "tract-nnef" {
        if safetensors.is_some() {
            return Err(CliError::InvalidInput(format!(
                "--safetensors is not supported by backend '{backend_name}' (Tract loads ONNX/NNEF graphs only)"
            )));
        }
        return load(checkpoint_dir).map_err(CliError::Runner);
    }
    build_burn_runner(backend_name, checkpoint_dir, safetensors)
}

#[cfg(any(feature = "burn-cpu", feature = "burn-cuda"))]
fn build_burn_runner(
    backend_name: &str,
    checkpoint_dir: &Path,
    safetensors: Option<&Path>,
) -> Result<Box<dyn InferenceRunner>, CliError> {
    let backend_kind = BackendKind::parse_cli(backend_name).map_err(CliError::InvalidInput)?;
    load_with_backend(backend_kind, checkpoint_dir, safetensors, None).map_err(CliError::Runner)
}

#[cfg(not(any(feature = "burn-cpu", feature = "burn-cuda")))]
fn build_burn_runner(
    backend_name: &str,
    _checkpoint_dir: &Path,
    _safetensors: Option<&Path>,
) -> Result<Box<dyn InferenceRunner>, CliError> {
    Err(CliError::InvalidInput(format!(
        "backend '{backend_name}' requires building with feature `burn-cpu` or `burn-cuda`"
    )))
}

#[derive(Debug, Parser)]
#[command(
    name = "lewm-infer",
    version,
    about = "Run CPU inference and planning with exported LeWM graphs.",
    long_about = None,
    term_width = 100,
    max_term_width = 100
)]
struct Cli {
    #[arg(
        long,
        global = true,
        value_name = "PATH",
        help = "Directory containing encoder/predictor ONNX or NNEF graph files."
    )]
    checkpoint_dir: Option<PathBuf>,

    #[arg(
        long,
        global = true,
        default_value_t = 2,
        value_name = "INT",
        help = "Action dimension, for example 2 for PushT or 6 for SO-100."
    )]
    action_dim: usize,

    #[arg(
        long,
        global = true,
        value_name = "INT",
        help = "Rayon worker thread count used by Tract-backed execution."
    )]
    threads: Option<usize>,

    #[arg(
        long,
        global = true,
        value_name = "NAME",
        help = "Inference backend: tract|tract-onnx|tract-nnef|burn-cpu|burn-cuda."
    )]
    backend: Option<String>,

    #[arg(
        long,
        global = true,
        value_name = "PATH",
        help = "Safetensors weights for Burn backends. Defaults to weights.safetensors or the latest step_*.safetensors in --checkpoint-dir."
    )]
    safetensors: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run one planning request and emit cost, actions, and latency JSON.
    Plan(PlanArgs),
    /// Run repeated planning requests and emit latency summary JSON.
    Bench(BenchArgs),
    /// Start the loopback HTTP shim used by the demo Space.
    Serve(ServeArgs),
    /// Load a checkpoint and optionally run an image encode smoke.
    Verify(VerifyArgs),
    /// Compare runner outputs against Python reference dumps and emit a parity report.
    Eval(EvalArgs),
}

#[derive(Debug, Args)]
struct PlanArgs {
    #[arg(long, value_name = "PATH", help = "Start image, JPEG or PNG.")]
    start: PathBuf,
    #[arg(long, value_name = "PATH", help = "Goal image, JPEG or PNG.")]
    goal: PathBuf,
    #[arg(long, default_value_t = DEFAULT_HORIZON_PLAN, value_name = "INT")]
    horizon: usize,
    #[arg(long, default_value_t = DEFAULT_N_CAND, value_name = "INT")]
    n_cand: usize,
    #[arg(long, default_value_t = DEFAULT_N_ITER, value_name = "INT")]
    n_iter: usize,
    #[arg(long, value_name = "PATH", help = "Optional JSON output path.")]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = 0, hide = true)]
    seed: u64,
    #[arg(long, hide = true)]
    n_elite: Option<usize>,
    #[arg(long, default_value_t = 2, hide = true)]
    history_steps: usize,
}

#[derive(Debug, Args)]
struct BenchArgs {
    #[arg(long, default_value_t = 50, value_name = "INT")]
    episodes: usize,
    #[arg(long, default_value_t = 5, value_name = "INT")]
    warmup_runs: usize,
    #[arg(long, value_name = "PATH", help = "Optional JSON report path.")]
    report: Option<PathBuf>,
    #[arg(long, default_value_t = 0, hide = true)]
    seed: u64,
    #[arg(long, default_value_t = 3, hide = true)]
    history_steps: usize,
}

#[derive(Debug, Args)]
struct ServeArgs {
    #[arg(
        long,
        default_value = "127.0.0.1:7861",
        value_name = "ADDR",
        help = "Address for the loopback HTTP shim."
    )]
    bind: SocketAddr,
}

#[derive(Debug, Args)]
struct VerifyArgs {
    #[arg(long, value_name = "PATH", help = "Optional image encode smoke input.")]
    image: Option<PathBuf>,
    #[arg(long, value_name = "PATH", help = "Optional JSON output path.")]
    out: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct EvalArgs {
    #[arg(
        long,
        value_name = "PATH",
        help = "Directory of parity dumps in the AbdelStark/lewm-rs-parity-dumps layout."
    )]
    dumps_dir: PathBuf,
    #[arg(
        long,
        value_name = "PATH",
        help = "Optional reference image to use for the encoder smoke. Defaults to a zero pixel buffer."
    )]
    image: Option<PathBuf>,
    #[arg(
        long,
        default_value_t = DEFAULT_TOLERANCE,
        value_name = "FLOAT",
        help = "Pass threshold for the per-stage L∞ comparison."
    )]
    tolerance: f32,
    #[arg(
        long,
        default_value_t = 3,
        value_name = "INT",
        help = "History context length used to assemble the predictor input."
    )]
    history_steps: usize,
    #[arg(long, value_name = "PATH", help = "Optional JSON output path.")]
    out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct PlanJson {
    cost: f32,
    best_actions: Vec<Vec<f32>>,
    latency_ms: f64,
    checkpoint_format: String,
    trace: Vec<TraceJson>,
}

#[derive(Debug, Serialize)]
struct TraceJson {
    iteration: usize,
    best_cost: f32,
    mean_cost: f32,
    sigma_mean: f32,
}

#[derive(Debug, Serialize)]
struct BenchJson {
    episodes: usize,
    warmup_runs: usize,
    latency_ms: f64,
    min_latency_ms: f64,
    p50_latency_ms: f64,
    p95_latency_ms: f64,
    max_latency_ms: f64,
}

#[derive(Debug, Serialize)]
struct VerifyJson {
    ok: bool,
    checkpoint_format: String,
    optimized: bool,
    intra_op_threads: usize,
    encoded_len: Option<usize>,
    latency_ms: f64,
}

#[derive(Debug, Serialize)]
struct HealthJson {
    ok: bool,
    checkpoint_format: String,
}

trait Clock {
    type Instant: Copy;

    fn now(&self) -> Self::Instant;
    fn elapsed_ms(&self, start: Self::Instant) -> f64;
}

#[derive(Debug, Clone, Copy)]
struct WallClock;

impl Clock for WallClock {
    type Instant = StdInstant;

    fn now(&self) -> Self::Instant {
        StdInstant::now()
    }

    fn elapsed_ms(&self, start: Self::Instant) -> f64 {
        start.elapsed().as_secs_f64() * 1000.0
    }
}

fn run_plan<C: Clock>(
    checkpoint_dir: &Path,
    action_dim: usize,
    backend: Option<&str>,
    safetensors: Option<&Path>,
    args: &PlanArgs,
    clock: &C,
) -> Result<(), CliError> {
    validate_positive("horizon", args.horizon)?;
    validate_positive("n-cand", args.n_cand)?;
    validate_positive("n-iter", args.n_iter)?;
    validate_positive("history-steps", args.history_steps)?;

    let start_time = clock.now();
    let start_pixels = preprocess_path(&args.start)?;
    let goal_pixels = preprocess_path(&args.goal)?;
    let mut runner = build_runner(checkpoint_dir, backend, safetensors)?;
    let start_latent = runner.encode(start_pixels.as_ref())?;
    let goal_latent = runner.encode(goal_pixels.as_ref())?;
    let z_history = repeat_history(&start_latent, args.history_steps);
    let planner = CpuCem {
        n_iter: args.n_iter,
        n_cand: args.n_cand,
        n_elite: args.n_elite.unwrap_or(DEFAULT_N_ELITE.min(args.n_cand)),
        horizon_plan: args.horizon,
        ..CpuCem::default()
    };
    let mut rng = cem_rng(args.seed)?;
    let result = planner.plan(&mut *runner, &z_history, &goal_latent, &mut rng, action_dim)?;
    let metadata = runner.metadata();
    let payload = PlanJson {
        cost: result.best_cost,
        best_actions: action_rows(&result.best_actions, action_dim),
        latency_ms: clock.elapsed_ms(start_time),
        checkpoint_format: metadata.format.to_string(),
        trace: result
            .trace
            .into_iter()
            .map(|trace| TraceJson {
                iteration: trace.iteration,
                best_cost: trace.best_cost,
                mean_cost: trace.mean_cost,
                sigma_mean: trace.sigma_mean,
            })
            .collect(),
    };

    write_json(args.out.as_deref(), &payload)
}

fn run_bench<C: Clock>(
    checkpoint_dir: &Path,
    action_dim: usize,
    backend: Option<&str>,
    safetensors: Option<&Path>,
    args: &BenchArgs,
    clock: &C,
) -> Result<(), CliError> {
    validate_positive("episodes", args.episodes)?;
    validate_positive("history-steps", args.history_steps)?;

    let mut runner = build_runner(checkpoint_dir, backend, safetensors)?;
    let pixels = zero_pixels()?;
    let planner = CpuCem::default();
    let total_runs = args
        .warmup_runs
        .checked_add(args.episodes)
        .ok_or_else(|| CliError::InvalidInput("warmup-runs + episodes overflowed usize".into()))?;
    let mut latencies = Vec::with_capacity(args.episodes);

    for run_index in 0..total_runs {
        let start_time = clock.now();
        let start_latent = runner.encode(pixels.as_ref())?;
        let goal_latent = runner.encode(pixels.as_ref())?;
        let z_history = repeat_history(&start_latent, args.history_steps);
        let mut rng = cem_rng(args.seed.saturating_add(run_index as u64))?;
        let _result = planner.plan(&mut *runner, &z_history, &goal_latent, &mut rng, action_dim)?;
        let latency = clock.elapsed_ms(start_time);
        if run_index >= args.warmup_runs {
            latencies.push(latency);
        }
    }

    latencies.sort_by(f64::total_cmp);
    let payload = BenchJson {
        episodes: args.episodes,
        warmup_runs: args.warmup_runs,
        latency_ms: mean_latency(&latencies),
        min_latency_ms: latencies[0],
        p50_latency_ms: percentile(&latencies, 50),
        p95_latency_ms: percentile(&latencies, 95),
        max_latency_ms: latencies[latencies.len() - 1],
    };

    write_json(args.report.as_deref(), &payload)
}

fn run_serve(
    checkpoint_dir: &Path,
    backend: Option<&str>,
    safetensors: Option<&Path>,
    args: &ServeArgs,
) -> Result<(), CliError> {
    let runner = build_runner(checkpoint_dir, backend, safetensors)?;
    let health = HealthJson {
        ok: true,
        checkpoint_format: runner.metadata().format.to_string(),
    };
    let health_body = serde_json::to_vec(&health)?;
    let listener = TcpListener::bind(args.bind)?;

    for stream in listener.incoming() {
        let mut stream = stream?;
        handle_connection(&mut stream, &health_body)?;
    }

    Ok(())
}

fn run_verify<C: Clock>(
    checkpoint_dir: &Path,
    backend: Option<&str>,
    safetensors: Option<&Path>,
    args: &VerifyArgs,
    clock: &C,
) -> Result<(), CliError> {
    let start_time = clock.now();
    let mut runner = build_runner(checkpoint_dir, backend, safetensors)?;
    let encoded_len = if let Some(image) = &args.image {
        let pixels = preprocess_path(image)?;
        Some(runner.encode(pixels.as_ref())?.len())
    } else {
        None
    };
    let metadata = runner.metadata();
    let payload = VerifyJson {
        ok: true,
        checkpoint_format: metadata.format.to_string(),
        optimized: metadata.optimized,
        intra_op_threads: metadata.intra_op_threads,
        encoded_len,
        latency_ms: clock.elapsed_ms(start_time),
    };

    write_json(args.out.as_deref(), &payload)
}

fn run_eval(
    checkpoint_dir: &Path,
    action_dim: usize,
    backend: Option<&str>,
    safetensors: Option<&Path>,
    args: &EvalArgs,
) -> Result<(), CliError> {
    validate_positive("history-steps", args.history_steps)?;
    if !(args.tolerance.is_finite() && args.tolerance >= 0.0) {
        return Err(CliError::InvalidInput(
            "--tolerance must be a finite non-negative float".to_owned(),
        ));
    }

    let pixels = if let Some(image) = &args.image {
        preprocess_path(image)?
    } else {
        zero_pixels()?
    };

    let mut runner = build_runner(checkpoint_dir, backend, safetensors)?;
    let backend_label = backend.map_or_else(
        || runner.metadata().format.to_string(),
        std::string::ToString::to_string,
    );

    // Build a synthetic predictor input by encoding the (possibly all-zero)
    // image once and repeating the latent as the history window. The matching
    // reference dumps were captured against the same fixture in the parity
    // pipeline, so this is consistent for an end-to-end smoke; for full-fidelity
    // comparisons the caller can pre-supply matching history/action arrays via
    // a follow-up flag (left for a future iteration).
    let start_latent = runner.encode(pixels.as_ref()).map_err(CliError::Runner)?;
    let history = repeat_history(&start_latent, args.history_steps);
    let actions = vec![0.0_f32; args.history_steps * action_dim];

    let inputs = ParityEvalInputs {
        pixels: pixels.as_ref(),
        history: &history,
        actions: &actions,
        history_steps: args.history_steps,
        action_dim,
        dumps_dir: &args.dumps_dir,
        tolerance: args.tolerance,
        backend: &backend_label,
    };
    let report = run_parity_eval(&mut *runner, inputs).map_err(CliError::Eval)?;

    write_json(args.out.as_deref(), &report)?;
    if !report.pass {
        return Err(CliError::EvalFailed(report));
    }
    Ok(())
}

fn handle_connection(stream: &mut TcpStream, health_body: &[u8]) -> Result<(), CliError> {
    let mut buffer = [0_u8; 1024];
    let read_len = stream.read(&mut buffer)?;
    let request = String::from_utf8_lossy(&buffer[..read_len]);
    if request.starts_with("GET /healthz ") || request.starts_with("GET / ") {
        write_http_response(stream, "200 OK", "application/json", health_body)?;
    } else {
        write_http_response(
            stream,
            "404 Not Found",
            "application/json",
            br#"{"ok":false,"error":"not found"}"#,
        )?;
    }
    Ok(())
}

fn write_http_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> Result<(), CliError> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    Ok(())
}

fn validate_global(cli: &Cli) -> Result<&Path, CliError> {
    validate_positive("action-dim", cli.action_dim)?;
    if let Some(threads) = cli.threads {
        validate_positive("threads", threads)?;
    }
    cli.checkpoint_dir
        .as_deref()
        .ok_or_else(|| CliError::InvalidInput("--checkpoint-dir is required".into()))
}

fn validate_positive(name: &str, value: usize) -> Result<(), CliError> {
    if value == 0 {
        Err(CliError::InvalidInput(format!("{name} must be non-zero")))
    } else {
        Ok(())
    }
}

fn configure_threads(threads: Option<usize>) -> Result<(), CliError> {
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|source| CliError::ThreadPool(source.to_string()))?;
    }
    Ok(())
}

fn repeat_history(latent: &[f32], history_steps: usize) -> Vec<f32> {
    let mut history = Vec::with_capacity(latent.len() * history_steps);
    for _ in 0..history_steps {
        history.extend_from_slice(latent);
    }
    history
}

fn action_rows(actions: &[f32], action_dim: usize) -> Vec<Vec<f32>> {
    actions
        .chunks(action_dim)
        .map(<[f32]>::to_vec)
        .collect::<Vec<_>>()
}

fn zero_pixels() -> Result<Box<[f32; IMAGE_ELEMENT_COUNT]>, CliError> {
    vec![0.0_f32; IMAGE_ELEMENT_COUNT]
        .into_boxed_slice()
        .try_into()
        .map_err(|_| CliError::InvalidInput("zero image did not match encoder input shape".into()))
}

fn write_json<T: Serialize>(path: Option<&Path>, value: &T) -> Result<(), CliError> {
    if let Some(path) = path {
        let writer = File::create(path)?;
        serde_json::to_writer_pretty(writer, value)?;
    } else {
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        serde_json::to_writer_pretty(&mut handle, value)?;
        handle.write_all(b"\n")?;
    }
    Ok(())
}

fn mean_latency(values: &[f64]) -> f64 {
    let (sum, count) = values
        .iter()
        .fold((0.0_f64, 0.0_f64), |(sum, count), value| {
            (sum + *value, count + 1.0)
        });
    sum / count
}

fn percentile(values: &[f64], percentile: usize) -> f64 {
    let index = (values.len() - 1) * percentile / 100;
    values[index]
}

#[derive(Debug)]
enum CliError {
    InvalidInput(String),
    ThreadPool(String),
    Io(std::io::Error),
    Json(serde_json::Error),
    Preprocess(PreprocessError),
    Runner(RunnerError),
    Plan(PlanError),
    Eval(EvalError),
    EvalFailed(EvalReport),
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(reason) => write!(f, "invalid input: {reason}"),
            Self::ThreadPool(source) => write!(f, "failed to configure thread pool: {source}"),
            Self::Io(source) => write!(f, "{source}"),
            Self::Json(source) => write!(f, "{source}"),
            Self::Preprocess(source) => write!(f, "{source}"),
            Self::Runner(source) => write!(f, "{source}"),
            Self::Plan(source) => write!(f, "{source}"),
            Self::Eval(source) => write!(f, "{source}"),
            Self::EvalFailed(report) => {
                use std::fmt::Write as _;
                let mut summary = String::new();
                for stage in &report.stages {
                    write!(
                        &mut summary,
                        "\n  {stage}: linf={linf:.3e} tol={tol:.3e} pass={pass}",
                        stage = stage.stage,
                        linf = stage.linf,
                        tol = stage.tolerance,
                        pass = stage.pass,
                    )?;
                }
                write!(
                    f,
                    "parity eval failed for backend {backend}; tolerance {tol:.3e}{summary}",
                    backend = report.backend,
                    tol = report.tolerance
                )
            },
        }
    }
}

impl std::error::Error for CliError {}

impl From<std::io::Error> for CliError {
    fn from(source: std::io::Error) -> Self {
        Self::Io(source)
    }
}

impl From<serde_json::Error> for CliError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json(source)
    }
}

impl From<PreprocessError> for CliError {
    fn from(source: PreprocessError) -> Self {
        Self::Preprocess(source)
    }
}

impl From<RunnerError> for CliError {
    fn from(source: RunnerError) -> Self {
        Self::Runner(source)
    }
}

impl From<PlanError> for CliError {
    fn from(source: PlanError) -> Self {
        Self::Plan(source)
    }
}

use std::fmt;

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::Cli;

    #[test]
    fn help_lists_all_subcommands() {
        let help = Cli::command().render_help().to_string();

        // Spot-check the help surface rather than pinning the full formatting,
        // which is sensitive to clap's wrapping and column widths.
        assert!(help.contains("plan"), "help missing plan command: {help}");
        assert!(help.contains("bench"), "help missing bench command: {help}");
        assert!(help.contains("serve"), "help missing serve command: {help}");
        assert!(
            help.contains("verify"),
            "help missing verify command: {help}"
        );
        assert!(help.contains("eval"), "help missing eval command: {help}");
        assert!(
            help.contains("--backend"),
            "help missing --backend flag: {help}"
        );
        assert!(
            help.contains("--safetensors"),
            "help missing --safetensors flag: {help}"
        );
    }
}
