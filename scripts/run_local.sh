#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/run_local.sh flamegraph <name> -- <cargo-flamegraph args...>
  scripts/run_local.sh nsys <name> -- <command...>
  scripts/run_local.sh ncu <name> -- <command...>

Examples:
  scripts/run_local.sh flamegraph infer-plan --bin lewm-infer -- plan --help
  scripts/run_local.sh nsys pusht-smoke -- lewm-train smoke --config configs/pusht.toml --steps 100

Artifacts default to profiling/{flamegraphs,nsys,ncu}/<git_sha>/.
Override with LEWM_PROFILE_DIR=/path/to/output.
USAGE
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'missing required command: %s\n' "$1" >&2
    exit 127
  fi
}

git_sha() {
  git rev-parse --short HEAD
}

mode="${1:-}"
name="${2:-}"

if [ -z "$mode" ] || [ -z "$name" ]; then
  usage
  exit 64
fi

shift 2
if [ "${1:-}" = "--" ]; then
  shift
fi

if [ "$#" -eq 0 ]; then
  usage
  exit 64
fi

case "$mode" in
  flamegraph)
    require_cmd cargo
    if ! cargo flamegraph --help >/dev/null 2>&1; then
      printf '%s\n' 'missing cargo-flamegraph; install with: cargo install flamegraph' >&2
      exit 127
    fi
    out_dir="${LEWM_PROFILE_DIR:-profiling/flamegraphs/$(git_sha)}"
    mkdir -p "$out_dir"
    output="${out_dir}/${name}.svg"
    RUSTFLAGS="${RUSTFLAGS:--C force-frame-pointers=yes}" cargo flamegraph --output "$output" "$@"
    printf 'wrote %s\n' "$output"
    ;;
  nsys)
    require_cmd nsys
    out_dir="${LEWM_PROFILE_DIR:-profiling/nsys/$(git_sha)}"
    mkdir -p "$out_dir"
    output="${out_dir}/${name}"
    nsys profile --trace=cuda,nvtx,cublas --output="$output" -- "$@"
    printf 'wrote %s.nsys-rep\n' "$output"
    ;;
  ncu)
    require_cmd ncu
    out_dir="${LEWM_PROFILE_DIR:-profiling/ncu/$(git_sha)}"
    mkdir -p "$out_dir"
    output="${out_dir}/${name}"
    ncu --set full --export "$output" -- "$@"
    printf 'wrote %s.ncu-rep\n' "$output"
    ;;
  *)
    usage
    exit 64
    ;;
esac
