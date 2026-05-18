#!/usr/bin/env python3
"""Run a local full PushT Burn/Jepa checkpoint contract smoke."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_CONFIG = Path("configs/pusht.toml")
DEFAULT_STEPS = 1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"PushT config to smoke ({DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to a new /tmp/lewm-pusht-full-contract-* directory.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Training steps to run before checking the safetensors contract ({DEFAULT_STEPS})",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def default_output_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="lewm-pusht-full-contract-"))


def checkpoint_path(output_dir: Path, steps: int) -> Path:
    return output_dir / f"step_{steps:07d}.safetensors"


def train_command(config: Path, output_dir: Path, steps: int) -> list[str]:
    return [
        "cargo",
        "run",
        "-p",
        "lewm-train",
        "--bin",
        "lewm-train",
        "--",
        "--config",
        str(config),
        "--set",
        'experimental.pusht_train_mode="full_burn_jepa"',
        "--device",
        "cpu",
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(steps),
        "train",
    ]


def contract_command(checkpoint: Path) -> list[str]:
    return [
        "uv",
        "run",
        "--project",
        "python",
        "--frozen",
        "python",
        "python/export_onnx.py",
        "--safetensors",
        str(checkpoint),
        "--check-contract-only",
    ]


def run_command(command: list[str]) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=repo_root(), check=True)


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        print("full_pusht_contract_smoke.py: --steps must be positive", file=sys.stderr)
        return 64

    config = resolve_repo_path(args.config)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir()
    output_dir = resolve_repo_path(output_dir)
    checkpoint = checkpoint_path(output_dir, args.steps)

    try:
        run_command(train_command(config, output_dir, args.steps))
        run_command(contract_command(checkpoint))
    except subprocess.CalledProcessError as exc:
        return exc.returncode

    print(f"full PushT Burn/Jepa contract smoke ok: output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
