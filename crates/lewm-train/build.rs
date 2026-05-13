#![allow(clippy::print_stdout, missing_docs)]

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs/heads");
    println!("cargo:rustc-env=LEWM_GIT_SHA={}", git_short_sha());
    println!("cargo:rustc-env=LEWM_BUILD_DATE={}", build_date());
}

fn git_short_sha() -> String {
    command_stdout("git", &["rev-parse", "--short=12", "HEAD"])
        .unwrap_or_else(|| "unknown".to_string())
}

fn build_date() -> String {
    command_stdout("date", &["-u", "+%Y-%m-%d"]).unwrap_or_else(|| "unknown".to_string())
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();

    if trimmed.is_empty() {
        return None;
    }

    Some(trimmed.to_string())
}
