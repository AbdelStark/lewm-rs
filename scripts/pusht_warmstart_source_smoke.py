#!/usr/bin/env python3
"""Run a local bounded PushT warm-start source checkpoint smoke."""

from __future__ import annotations

import argparse
import hashlib
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
SOURCE_CHECK_RE = re.compile(
    r"warm-start source ok: path=(?P<path>.+) step=(?P<step>\d+) params=(?P<params>\d+)"
)
TRAIN_SUMMARY_RE = re.compile(
    r"mode=(?P<mode>[^;]+); data_source=(?P<data_source>[^;]+); "
    r"final_loss=(?P<final_loss>[0-9.eE+-]+); checkpoint_step=(?P<checkpoint_step>\d+); "
    r"checkpoint_complete=(?P<checkpoint_complete>true|false)"
)


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
        help="Output directory. Defaults to a new /tmp/lewm-pusht-warmstart-source-* directory.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Training steps to run before checking the .mpk source ({DEFAULT_STEPS})",
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
    return Path(tempfile.mkdtemp(prefix="lewm-pusht-warmstart-source-"))


def checkpoint_path(output_dir: Path, steps: int) -> Path:
    return output_dir / f"step_{steps:07d}.mpk"


def report_path(path: Path) -> str:
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def report_command(command: object) -> object:
    root_prefix = f"{repo_root()}/"
    if not isinstance(command, list):
        return command
    normalized = []
    for item in command:
        if isinstance(item, str) and item.startswith(root_prefix):
            normalized.append(item.removeprefix(root_prefix))
        else:
            normalized.append(item)
    return normalized


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
        "--device",
        "cpu",
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(steps),
        "train",
    ]


def source_check_command(checkpoint: Path, config: Path) -> list[str]:
    return [
        "python3",
        "scripts/check_warmstart_source.py",
        "--path",
        str(checkpoint),
        "--config",
        str(config),
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


def parse_train_output(stdout: str) -> dict[str, object]:
    match = TRAIN_SUMMARY_RE.search(stdout)
    if match is None:
        raise ValueError("train output did not include mode/data/final-loss checkpoint summary")
    return {
        "mode": match.group("mode"),
        "data_source": match.group("data_source"),
        "final_loss": float(match.group("final_loss")),
        "checkpoint_step": int(match.group("checkpoint_step")),
        "checkpoint_complete": match.group("checkpoint_complete") == "true",
    }


def parse_source_check_output(stdout: str) -> dict[str, object]:
    match = SOURCE_CHECK_RE.search(stdout)
    if match is None:
        raise ValueError("source-check output did not include path, step, and params")
    return {
        "path": match.group("path"),
        "step": int(match.group("step")),
        "params": int(match.group("params")),
    }


def load_record_summary(checkpoint: Path) -> dict[str, object]:
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{checkpoint}: record root must be an object")
    params = payload.get("params")
    adamw_params = payload.get("adamw_params", [])
    if not isinstance(params, list):
        raise ValueError(f"{checkpoint}: params must be a list")
    if not isinstance(adamw_params, list):
        raise ValueError(f"{checkpoint}: adamw_params must be a list")
    return {
        "schema_version": payload.get("schema_version"),
        "kind": payload.get("kind"),
        "step": payload.get("step"),
        "params": len(params),
        "adamw_params": len(adamw_params),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(
    path: Path,
    *,
    config: Path,
    output_dir: Path,
    steps: int,
    checkpoint: Path,
    train: subprocess.CompletedProcess[str],
    source_check: subprocess.CompletedProcess[str],
) -> None:
    payload = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "config": report_path(config),
        "output_dir": str(output_dir),
        "steps": steps,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_command": report_command(train.args),
        "source_check_command": report_command(source_check.args),
        "train": parse_train_output(train.stdout),
        "source_check": parse_source_check_output(source_check.stdout),
        "record": load_record_summary(checkpoint),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        print("pusht_warmstart_source_smoke.py: --steps must be positive", file=sys.stderr)
        return 64

    config = resolve_repo_path(args.config)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir()
    output_dir = resolve_repo_path(output_dir)
    report = resolve_repo_path(args.report) if args.report is not None else None
    checkpoint = checkpoint_path(output_dir, args.steps)

    try:
        train = run_command(train_command(config, output_dir, args.steps))
        source_check = run_command(source_check_command(checkpoint, config))
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"pusht_warmstart_source_smoke.py: {exc}", file=sys.stderr)
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
                source_check=source_check,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"pusht_warmstart_source_smoke.py: failed to write report: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {report}")

    print(f"PushT warm-start source smoke ok: output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
