#!/usr/bin/env python3
"""Upload a training output directory to a Hugging Face Hub repository."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_COMMIT_MESSAGE = "Upload lewm-rs training artifacts"
DEFAULT_REPO_TYPE = "model"
VALID_REPO_TYPES = {"model", "dataset", "space"}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="Local file or directory to upload.")
    parser.add_argument("--dst", required=True, help="Destination Hub repo id, for example abdelstark/lewm-rs-pusht.")
    parser.add_argument(
        "--repo-type",
        default=DEFAULT_REPO_TYPE,
        choices=sorted(VALID_REPO_TYPES),
        help=f"Hub repository type. Default: {DEFAULT_REPO_TYPE}.",
    )
    parser.add_argument(
        "--path-prefix",
        default=".",
        help="Path inside the repo. Use '.' to upload the directory contents at repo root.",
    )
    parser.add_argument(
        "--commit-message",
        default=DEFAULT_COMMIT_MESSAGE,
        help="Hub commit message.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Return success when --src is an empty directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the hf upload command without executing it.",
    )
    return parser.parse_args(argv)


def upload_command(args: argparse.Namespace) -> list[str]:
    command = [
        "hf",
        "upload",
        "--repo-type",
        args.repo_type,
        "--commit-message",
        args.commit_message,
        "--quiet",
        args.dst,
        str(args.src),
    ]
    if args.path_prefix:
        command.append(args.path_prefix)
    return command


def has_payload(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 0
    if not path.is_dir():
        return False
    return any(child.is_file() for child in path.rglob("*"))


def validate(args: argparse.Namespace) -> str | None:
    if shutil.which("hf") is None:
        return "hf CLI is required in PATH"
    if not args.src.exists():
        return f"--src does not exist: {args.src}"
    if not args.dst or "/" not in args.dst:
        return "--dst must be a Hub repo id in namespace/name form"
    if not os.environ.get("HF_TOKEN"):
        return "HF_TOKEN is required for Hub upload"
    if not args.allow_empty and not has_payload(args.src):
        return f"--src contains no uploadable files: {args.src}"
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    failure = validate(args)
    if failure is not None:
        print(f"upload_checkpoints.py: {failure}", file=sys.stderr)
        return 2

    command = upload_command(args)
    if args.dry_run:
        print(" ".join(command))
        return 0

    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"upload_checkpoints.py: hf upload failed with exit code {result.returncode}", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
