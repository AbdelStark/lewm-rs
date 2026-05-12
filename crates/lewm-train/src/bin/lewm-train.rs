//! Command-line entry point for training and training-adjacent operations.
//!
//! The command surface is implemented by the RFC 0005 training issue set.

use std::{
    env,
    ffi::OsString,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
};

use lewm_train::{ConfigError, load_root_config, to_pretty_toml};

#[derive(Debug)]
struct TrainArgs {
    config: PathBuf,
    dry_run: bool,
    print_config: bool,
}

#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error("missing value for --config")]
    MissingConfigValue,
    #[error("argument is not valid UTF-8")]
    NonUtf8Argument,
    #[error("unknown argument {0}")]
    UnknownArgument(String),
    #[error("lewm-train currently supports --dry-run only; the training loop is not implemented")]
    TrainingLoopUnavailable,
    #[error("could not write output: {0}")]
    Write(#[from] io::Error),
}

fn main() -> ExitCode {
    match run(env::args_os().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ = writeln!(io::stderr(), "{error}");
            ExitCode::FAILURE
        },
    }
}

fn run(args: impl Iterator<Item = OsString>) -> Result<(), CliError> {
    let args = parse_args(args)?;
    let config = load_root_config(&args.config)?;

    if args.print_config {
        let text = to_pretty_toml(&config)?;
        io::stdout().write_all(text.as_bytes())?;
    }

    if !args.dry_run {
        return Err(CliError::TrainingLoopUnavailable);
    }

    Ok(())
}

fn parse_args(args: impl Iterator<Item = OsString>) -> Result<TrainArgs, CliError> {
    let mut config = PathBuf::from("configs/so100.toml");
    let mut dry_run = false;
    let mut print_config = false;
    let mut iter = args.peekable();

    if matches!(iter.peek().and_then(|arg| arg.to_str()), Some("train")) {
        let _ = iter.next();
    }

    while let Some(arg) = iter.next() {
        let arg = arg.to_str().ok_or(CliError::NonUtf8Argument)?;
        if arg == "--dry-run" {
            dry_run = true;
        } else if arg == "--print-config" {
            print_config = true;
        } else if arg == "--config" {
            config = iter.next().ok_or(CliError::MissingConfigValue)?.into();
        } else if let Some(value) = arg.strip_prefix("--config=") {
            config = PathBuf::from(value);
        } else {
            return Err(CliError::UnknownArgument(arg.to_owned()));
        }
    }

    Ok(TrainArgs {
        config,
        dry_run,
        print_config,
    })
}
