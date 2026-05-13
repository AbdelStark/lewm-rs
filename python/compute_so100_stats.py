#!/usr/bin/env python3
"""Thin SO-100 wrapper for the lewm-data compute_stats binary."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute deterministic SO-100 action stats."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script = Path(__file__).with_name("compute_stats.py")
    command = [
        sys.executable,
        str(script),
        "--dataset",
        "so100",
        "--root",
        str(args.root),
        "--out",
        str(args.out),
        "--seed",
        str(args.seed),
        "--horizon",
        str(args.horizon),
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
