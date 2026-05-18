#!/usr/bin/env python3
"""Run a local full PushT Burn/Jepa checkpoint contract smoke."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_CONFIG = Path("configs/pusht.toml")
DEFAULT_STEPS = 1
RECOVERED_RE = re.compile(r"recovered (?P<recovered>\d+) of (?P<expected>\d+) expected")
BURN_TENSORS_RE = re.compile(r"Burn destination tensors: (?P<count>\d+)")
SHA256_RE = re.compile(r"Safetensors SHA-256: (?P<sha256>[0-9a-f]{64})")


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
    parser.add_argument(
        "--report",
        type=Path,
        help="Write a JSON evidence report after a successful smoke.",
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


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"+ {shlex.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()
    return result


def parse_contract_output(stdout: str) -> dict[str, object]:
    recovered = RECOVERED_RE.search(stdout)
    burn_tensors = BURN_TENSORS_RE.search(stdout)
    sha256 = SHA256_RE.search(stdout)
    if recovered is None or burn_tensors is None or sha256 is None:
        raise ValueError("contract output did not include recovered count, Burn tensor count, and SHA-256")
    return {
        "recovered_pytorch_keys": int(recovered.group("recovered")),
        "expected_pytorch_keys": int(recovered.group("expected")),
        "burn_destination_tensors": int(burn_tensors.group("count")),
        "safetensors_sha256": sha256.group("sha256"),
    }


def write_report(
    path: Path,
    *,
    config: Path,
    output_dir: Path,
    steps: int,
    checkpoint: Path,
    train: subprocess.CompletedProcess[str],
    contract: subprocess.CompletedProcess[str],
) -> None:
    contract_summary = parse_contract_output(contract.stdout)
    payload = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "config": str(config),
        "output_dir": str(output_dir),
        "steps": steps,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "train_command": train.args,
        "contract_command": contract.args,
        "contract": contract_summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        print("full_pusht_contract_smoke.py: --steps must be positive", file=sys.stderr)
        return 64

    config = resolve_repo_path(args.config)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir()
    output_dir = resolve_repo_path(output_dir)
    report = resolve_repo_path(args.report) if args.report is not None else None
    checkpoint = checkpoint_path(output_dir, args.steps)

    try:
        train = run_command(train_command(config, output_dir, args.steps))
        contract = run_command(contract_command(checkpoint))
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    except ValueError as exc:
        print(f"full_pusht_contract_smoke.py: {exc}", file=sys.stderr)
        return 1

    if report is not None:
        try:
            write_report(
                report,
                config=config,
                output_dir=output_dir,
                steps=args.steps,
                checkpoint=checkpoint,
                train=train,
                contract=contract,
            )
        except (OSError, ValueError) as exc:
            print(f"full_pusht_contract_smoke.py: failed to write report: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {report}")

    print(f"full PushT Burn/Jepa contract smoke ok: output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
