//! Command-line entry point for deterministic dataset stats computation.

use std::{
    env,
    error::Error,
    fmt,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
};

use lewm_data::{ComputeStatsConfig, StatsDataset, compute_stats};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            let _ignored = writeln!(io::stderr(), "error: {err}");
            ExitCode::FAILURE
        },
    }
}

fn run() -> Result<(), CliError> {
    let outcome = Args::parse(env::args().skip(1))?;
    let Some(args) = outcome else {
        write_usage(io::stdout())?;
        return Ok(());
    };

    let config = ComputeStatsConfig {
        dataset: args.dataset,
        root_path: args.root,
        seed: args.seed,
        horizon: args.horizon,
        validate_schema: args.validate_schema,
    };
    let stats = compute_stats(&config)?;
    stats.save_safetensors(&args.out)?;
    Ok(())
}

#[derive(Debug)]
struct Args {
    dataset: StatsDataset,
    root: PathBuf,
    out: PathBuf,
    seed: u64,
    horizon: usize,
    validate_schema: bool,
}

impl Args {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Self>, CliError> {
        let mut args = args.into_iter();
        let mut dataset = None;
        let mut root = None;
        let mut out = None;
        let mut seed = 0;
        let mut horizon = 1;
        let mut validate_schema = true;

        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--help" | "-h" => return Ok(None),
                "--dataset" => dataset = Some(parse_dataset(&next_value(&mut args, &flag)?)?),
                "--root" => root = Some(PathBuf::from(next_value(&mut args, &flag)?)),
                "--out" => out = Some(PathBuf::from(next_value(&mut args, &flag)?)),
                "--seed" => seed = parse_u64("--seed", &next_value(&mut args, &flag)?)?,
                "--horizon" => {
                    horizon = parse_usize("--horizon", &next_value(&mut args, &flag)?)?;
                },
                "--no-schema-validate" => validate_schema = false,
                unknown => {
                    return Err(CliError(format!(
                        "unknown argument {unknown:?}; pass --help for usage"
                    )));
                },
            }
        }

        Ok(Some(Self {
            dataset: dataset.ok_or_else(|| CliError("missing --dataset".to_string()))?,
            root: root.ok_or_else(|| CliError("missing --root".to_string()))?,
            out: out.ok_or_else(|| CliError("missing --out".to_string()))?,
            seed,
            horizon,
            validate_schema,
        }))
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, CliError> {
    args.next()
        .ok_or_else(|| CliError(format!("{flag} requires a value")))
}

fn parse_dataset(value: &str) -> Result<StatsDataset, CliError> {
    match value {
        "pusht" => Ok(StatsDataset::Pusht),
        _ => Err(CliError(format!(
            "unsupported dataset {value:?}; expected \"pusht\""
        ))),
    }
}

fn parse_u64(flag: &str, value: &str) -> Result<u64, CliError> {
    value
        .parse()
        .map_err(|source| CliError(format!("invalid {flag} value {value:?}: {source}")))
}

fn parse_usize(flag: &str, value: &str) -> Result<usize, CliError> {
    value
        .parse()
        .map_err(|source| CliError(format!("invalid {flag} value {value:?}: {source}")))
}

fn write_usage(mut writer: impl Write) -> Result<(), CliError> {
    writer.write_all(
        b"Usage: compute_stats --dataset pusht --root <path> --out <stats.safetensors> [--seed <u64>] [--horizon <n>] [--no-schema-validate]\n",
    )?;
    Ok(())
}

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CliError {}

impl From<lewm_data::DataError> for CliError {
    fn from(value: lewm_data::DataError) -> Self {
        Self(value.to_string())
    }
}

impl From<io::Error> for CliError {
    fn from(value: io::Error) -> Self {
        Self(value.to_string())
    }
}
