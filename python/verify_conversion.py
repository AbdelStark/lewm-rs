#!/usr/bin/env python3
"""Verify converted PushT reference artifacts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from convert_reference import sha256_file

MAX_ABS_RE = re.compile(r"max_abs_diff=([0-9.eE+-]+)")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safetensors-in", type=Path, required=True)
    parser.add_argument("--burn-record-in", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument(
        "--max-abs-diff",
        type=float,
        default=1.0e-7,
        help="Maximum allowed Safetensors-vs-Burn-record tensor drift.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    meta = load_json(args.meta)
    verify_hashes(args, meta)
    max_abs_diff = run_record_verify(args.safetensors_in, args.burn_record_in)
    if max_abs_diff > args.max_abs_diff:
        raise SystemExit(
            f"conversion verification failed: max_abs_diff={max_abs_diff:.8e} "
            f"> {args.max_abs_diff:.8e}"
        )
    print(
        "conversion verification: "
        f"ok=true max_abs_diff={max_abs_diff:.8e} "
        f"safetensors={args.safetensors_in} burn_record={args.burn_record_in}"
    )
    return 0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hashes(args: argparse.Namespace, meta: dict[str, Any]) -> None:
    artifacts = meta.get("artifacts", {})
    expected_safetensors = artifacts.get("safetensors_sha256")
    expected_burn_record = artifacts.get("burn_record_sha256")
    if expected_safetensors is None or expected_burn_record is None:
        raise SystemExit("conversion metadata is missing artifact SHA-256 values")

    actual_safetensors = sha256_file(args.safetensors_in)
    if actual_safetensors != expected_safetensors:
        raise SystemExit(
            f"safetensors sha256 mismatch: got {actual_safetensors}, "
            f"expected {expected_safetensors}"
        )

    actual_burn_record = sha256_file(args.burn_record_in)
    if actual_burn_record != expected_burn_record:
        raise SystemExit(
            f"burn record sha256 mismatch: got {actual_burn_record}, "
            f"expected {expected_burn_record}"
        )


def run_record_verify(safetensors_in: Path, burn_record_in: Path) -> float:
    command = [
        "cargo",
        "run",
        "--locked",
        "-p",
        "lewm-train",
        "--bin",
        "lewm-reference-record",
        "--",
        "--safetensors-in",
        str(safetensors_in),
        "--burn-record-in",
        str(burn_record_in),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    match = MAX_ABS_RE.search(completed.stdout)
    if match is None:
        raise SystemExit(
            "could not parse max_abs_diff from lewm-reference-record output:\n"
            + completed.stdout
        )
    return float(match.group(1))


if __name__ == "__main__":
    raise SystemExit(main())
