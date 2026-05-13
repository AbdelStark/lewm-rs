#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Build release binaries with the reproducible-build environment.

Environment:
  REPRO_TARGET       Rust target triple. Defaults to x86_64-unknown-linux-musl.
  REPRO_PROFILE      Cargo profile. Defaults to release-lto.
  REPRO_TARGET_DIR   Cargo target directory. Defaults to target/reproducible.
  REPRO_OUTPUT_DIR   Optional directory to copy lewm-* binaries into.
  SOURCE_DATE_EPOCH  Release timestamp. Defaults to the current git commit time.
  RUSTFLAGS          Extra rustc flags appended after the reproducibility flags.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)"
cd "$repo_root"

target="${REPRO_TARGET:-x86_64-unknown-linux-musl}"
profile="${REPRO_PROFILE:-release-lto}"
target_dir="${REPRO_TARGET_DIR:-target/reproducible}"
output_dir="${REPRO_OUTPUT_DIR:-}"

if [[ -z "${SOURCE_DATE_EPOCH:-}" ]]; then
  SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)"
  export SOURCE_DATE_EPOCH
fi

source_root="$(pwd -P)"
mkdir -p "$target_dir"
target_dir_abs="$(cd "$target_dir" && pwd -P)"
repro_flags="-C strip=symbols --remap-path-prefix=${source_root}=/src --remap-path-prefix=${target_dir_abs}=/src/target"
if [[ "$target" == *-apple-darwin ]]; then
  repro_flags="${repro_flags} -C link-arg=-Wl,-no_uuid"
fi
export RUSTFLAGS="${repro_flags}${RUSTFLAGS:+ ${RUSTFLAGS}}"
export CARGO_TARGET_DIR="$target_dir"

cargo build --workspace --bins --profile "$profile" --target "$target" --locked

if [[ -n "$output_dir" ]]; then
  mkdir -p "$output_dir"
  build_dir="${target_dir}/${target}/${profile}"
  found=0
  for binary in "${build_dir}"/lewm-*; do
    if [[ -f "$binary" && -x "$binary" ]]; then
      cp "$binary" "$output_dir/"
      found=1
    fi
  done
  if [[ "$found" -eq 0 ]]; then
    printf 'no lewm-* binaries found in %s\n' "$build_dir" >&2
    exit 1
  fi
fi
