#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Verify release binaries by rebuilding and comparing bytes.

Usage:
  scripts/verify_reproducible_release.sh <published-dir> [rebuilt-dir]

If rebuilt-dir is omitted, the script rebuilds the workspace with
scripts/build_reproducible_release.sh into a temporary directory.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

published_dir="${1:-}"
rebuilt_dir="${2:-}"

if [[ -z "$published_dir" ]]; then
  usage >&2
  exit 64
fi

if [[ ! -d "$published_dir" ]]; then
  printf 'published binary directory not found: %s\n' "$published_dir" >&2
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd -P)"
cd "$repo_root"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/lewm-repro.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

if [[ -z "$rebuilt_dir" ]]; then
  rebuilt_dir="${tmp_dir}/rebuilt"
  REPRO_TARGET_DIR="${REPRO_TARGET_DIR:-${tmp_dir}/target}" \
    REPRO_OUTPUT_DIR="$rebuilt_dir" \
    scripts/build_reproducible_release.sh
fi

if [[ ! -d "$rebuilt_dir" ]]; then
  printf 'rebuilt binary directory not found: %s\n' "$rebuilt_dir" >&2
  exit 1
fi

normalize_dir() {
  local source_dir="$1"
  local normalized_dir="$2"
  local found=0

  mkdir -p "$normalized_dir"
  for binary in "${source_dir}"/lewm-*; do
    if [[ -f "$binary" && -x "$binary" ]]; then
      local output="${normalized_dir}/$(basename "$binary")"
      cp "$binary" "$output"
      chmod u+w "$output"
      if command -v objcopy >/dev/null 2>&1; then
        objcopy --remove-section=.note.gnu.build-id "$output" 2>/dev/null || true
      fi
      found=1
    fi
  done

  if [[ "$found" -eq 0 ]]; then
    printf 'no lewm-* binaries found in %s\n' "$source_dir" >&2
    exit 1
  fi
}

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

published_normalized="${tmp_dir}/published-normalized"
rebuilt_normalized="${tmp_dir}/rebuilt-normalized"
normalize_dir "$published_dir" "$published_normalized"
normalize_dir "$rebuilt_dir" "$rebuilt_normalized"

for expected in "${published_normalized}"/lewm-*; do
  binary_name="$(basename "$expected")"
  actual="${rebuilt_normalized}/${binary_name}"
  if [[ ! -f "$actual" ]]; then
    printf 'rebuilt binary missing: %s\n' "$binary_name" >&2
    exit 1
  fi

  if ! cmp -s "$expected" "$actual"; then
    printf 'reproducible build mismatch: %s\n' "$binary_name" >&2
    printf 'published sha256: %s\n' "$(sha256 "$expected")" >&2
    printf 'rebuilt   sha256: %s\n' "$(sha256 "$actual")" >&2
    exit 1
  fi

  printf 'verified reproducible binary: %s\n' "$binary_name"
done
