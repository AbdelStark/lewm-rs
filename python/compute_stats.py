#!/usr/bin/env python3
"""Convenience wrapper for the lewm-data compute_stats binary."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute deterministic lewm-data action stats."
    )
    parser.add_argument("--dataset", choices=("pusht", "so100"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--no-schema-validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    binary = os.environ.get("LEWM_COMPUTE_STATS_BIN")
    if binary:
        command = [binary]
    else:
        command = ["cargo", "run", "--quiet", "-p", "lewm-data", "--bin", "compute_stats", "--"]

    command.extend(
        [
            "--dataset",
            args.dataset,
            "--root",
            str(args.root),
            "--out",
            str(args.out),
            "--seed",
            str(args.seed),
            "--horizon",
            str(args.horizon),
        ]
    )
    if args.no_schema_validate:
        command.append("--no-schema-validate")

    return subprocess.run(command, check=False, cwd=repo_root).returncode


if __name__ == "__main__":
    sys.exit(main())
