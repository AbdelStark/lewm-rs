#!/usr/bin/env python3
"""Prepare or run the F1 PushT ONNX export and Hub upload workflow."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = "abdelstark/lewm-rs-pusht"
DEFAULT_STEP = 50_000
DEFAULT_ACTION_DIM = 10
DEFAULT_WORK_DIR = Path("/tmp/lewm-f1-pusht-onnx")
DEFAULT_META = Path("tests/fixtures/reference_model.meta.json")
REQUIRED_RUN_PREFIX = "train/pusht-full-burn-jepa-"
LEGACY_BOUNDED_RUN_PREFIX = "train/pusht-full-lewm-"
ONNXRUNTIME_DEP = "onnxruntime>=1.22,<2"
RUN_SUFFIX_RE = re.compile(r"^\d{8}T\d{6}Z$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-prefix",
        help="Hub path containing step_NNNNNNN.safetensors, for example "
        "train/pusht-full-burn-jepa-20260518T120000Z",
    )
    source.add_argument(
        "--safetensors",
        type=Path,
        help="Local full PushT Burn/Jepa safetensors checkpoint. Skips Hub download.",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Hub repo id ({DEFAULT_REPO})")
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP,
        help=f"Checkpoint step to export ({DEFAULT_STEP})",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help=f"Working directory for downloads and ONNX output ({DEFAULT_WORK_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="ONNX output directory. Defaults to <work-dir>/onnx-full.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=DEFAULT_META,
        help=f"Locked architecture metadata ({DEFAULT_META})",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=DEFAULT_ACTION_DIM,
        help=f"Packed PushT action dimension ({DEFAULT_ACTION_DIM})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the workflow. Default is dry-run command printing.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Actually upload ONNX artifacts. Without this, upload_checkpoints.py runs in dry-run mode.",
    )
    return parser.parse_args(argv)


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def step_file_name(step: int) -> str:
    if step <= 0:
        raise ValueError("--step must be positive")
    return f"step_{step:07d}.safetensors"


def download_dir(work_dir: Path) -> Path:
    return work_dir / "hub"


def default_output_dir(work_dir: Path) -> Path:
    return work_dir / "onnx-full"


def downloaded_safetensors(work_dir: Path, run_prefix: str, step: int) -> Path:
    return download_dir(work_dir) / run_prefix / step_file_name(step)


def validate_run_prefix(run_prefix: str) -> None:
    if any(char in run_prefix for char in "*?["):
        raise ValueError("--run-prefix must be a literal Hub directory, not a glob")
    if run_prefix.endswith("/"):
        raise ValueError("--run-prefix must not end with '/'")
    if run_prefix.startswith(LEGACY_BOUNDED_RUN_PREFIX):
        raise ValueError(
            "--run-prefix points at the legacy bounded PushT artifact family; "
            f"F1 requires a completed {REQUIRED_RUN_PREFIX}YYYYMMDDTHHMMSSZ run"
        )
    if not run_prefix.startswith(REQUIRED_RUN_PREFIX):
        raise ValueError(
            f"--run-prefix must start with {REQUIRED_RUN_PREFIX!r} for the F1 full Burn/Jepa handoff"
        )
    suffix = run_prefix.removeprefix(REQUIRED_RUN_PREFIX)
    if RUN_SUFFIX_RE.fullmatch(suffix) is None:
        raise ValueError(
            f"--run-prefix must be a completed Hub directory like "
            f"{REQUIRED_RUN_PREFIX}YYYYMMDDTHHMMSSZ"
        )


def download_command(repo: str, run_prefix: str, work_dir: Path) -> list[str]:
    return [
        "hf",
        "download",
        repo,
        "--include",
        f"{run_prefix}/*",
        "--local-dir",
        str(download_dir(work_dir)),
    ]


def uv_python_command(script: str, *args: str) -> list[str]:
    return [
        "uv",
        "run",
        "--project",
        "python",
        "--frozen",
        "--extra",
        "parity",
        "python",
        script,
        *args,
    ]


def uv_python_with_command(script: str, *args: str, package: str) -> list[str]:
    return [
        "uv",
        "run",
        "--project",
        "python",
        "--frozen",
        "--extra",
        "parity",
        "--with",
        package,
        "python",
        script,
        *args,
    ]


def contract_command(safetensors: Path) -> list[str]:
    return uv_python_command(
        "python/export_onnx.py",
        "--safetensors",
        str(safetensors),
        "--check-contract-only",
    )


def export_command(
    safetensors: Path,
    meta: Path,
    output_dir: Path,
    action_dim: int,
) -> list[str]:
    return uv_python_command(
        "python/export_onnx.py",
        "--safetensors",
        str(safetensors),
        "--meta",
        str(meta),
        "--output-dir",
        str(output_dir),
        "--variant",
        "both",
        "--action-dim",
        str(action_dim),
    )


def verify_command(output_dir: Path) -> list[str]:
    return uv_python_with_command(
        "python/verify_onnx.py",
        "--dir",
        str(output_dir),
        package=ONNXRUNTIME_DEP,
    )


def upload_command(output_dir: Path, repo: str, *, upload: bool) -> list[str]:
    command = [
        "uv",
        "run",
        "--project",
        "python",
        "--frozen",
        "python",
        "python/upload_checkpoints.py",
        "--src",
        str(output_dir),
        "--dst",
        repo,
        "--path-prefix",
        "onnx-full/",
    ]
    if not upload:
        command.append("--dry-run")
    return command


def workflow_commands(args: argparse.Namespace) -> list[list[str]]:
    work_dir = resolve_repo_path(args.work_dir)
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else default_output_dir(work_dir)
    meta = resolve_repo_path(args.meta)

    commands: list[list[str]] = []
    if args.run_prefix:
        validate_run_prefix(args.run_prefix)
        commands.append(download_command(args.repo, args.run_prefix, work_dir))
        safetensors = downloaded_safetensors(work_dir, args.run_prefix, args.step)
    else:
        safetensors = resolve_repo_path(args.safetensors)

    commands.append(contract_command(safetensors))
    commands.append(export_command(safetensors, meta, output_dir, args.action_dim))
    commands.append(verify_command(output_dir))
    commands.append(upload_command(output_dir, args.repo, upload=args.upload))
    return commands


def run_commands(commands: list[list[str]]) -> int:
    for command in commands:
        print(f"+ {shlex.join(command)}", flush=True)
        result = subprocess.run(command, cwd=repo_root(), check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


def print_commands(commands: list[list[str]]) -> None:
    for command in commands:
        print(shlex.join(command))


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        commands = workflow_commands(args)
    except ValueError as exc:
        print(f"f1_export_pusht_onnx.py: {exc}", file=sys.stderr)
        return 64

    if not args.execute:
        print_commands(commands)
        return 0
    return run_commands(commands)


if __name__ == "__main__":
    raise SystemExit(main())
