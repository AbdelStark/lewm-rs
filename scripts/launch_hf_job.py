#!/usr/bin/env python3
"""Launch a checked-in Hugging Face Job spec safely."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LEASH_PATH = ROOT / ".ml-intern" / "cli_agent_config.json"
ENV_EXPR_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*))?\}$")


class LaunchError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path, help="path to jobs/*.yaml")
    parser.add_argument("--dry-run", action="store_true", help="print the command without running it")
    parser.add_argument("--detach", action="store_true", help="submit and return the HF Job id")
    parser.add_argument(
        "--allow-approval-required",
        action="store_true",
        help="allow jobs listed as human-approval-required in the leash config",
    )
    args = parser.parse_args()

    try:
        job_path = resolve_job_path(args.job)
        job = parse_job(job_path)
        leash = load_leash()
        validate_job(job_path, job, leash, args.allow_approval_required)
        command = render_command(job, detach=args.detach)
    except LaunchError as error:
        print(f"launch_hf_job.py: {error}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(shlex.join(command))
        return 0

    if shutil.which("hf") is None:
        print("launch_hf_job.py: missing required command: hf", file=sys.stderr)
        return 1

    return subprocess.run(command, check=False).returncode


def resolve_job_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else ROOT / path
    if not candidate.is_file():
        raise LaunchError(f"missing job spec: {path}")
    try:
        candidate.relative_to(ROOT / "jobs")
    except ValueError as error:
        raise LaunchError("job spec must live under jobs/") from error
    return candidate


def parse_job(path: Path) -> dict[str, object]:
    top: dict[str, object] = {}
    env: dict[str, str] = {}
    command_lines: list[str] = []
    section: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        if raw_line.startswith("  ") and section == "env":
            key, value = split_key_value(raw_line.strip(), path)
            env[key] = value
            continue

        if raw_line.startswith("  ") and section == "command":
            command_lines.append(raw_line.strip())
            continue

        section = None
        key, value = split_key_value(raw_line, path)
        if key == "env":
            section = "env"
            top["env"] = env
        elif key == "command":
            section = "command"
            top["command"] = ""
        else:
            top[key] = value

    if command_lines:
        top["command"] = "\n".join(command_lines)
    return top


def split_key_value(line: str, path: Path) -> tuple[str, str]:
    if ":" not in line:
        raise LaunchError(f"{path.relative_to(ROOT)}: invalid YAML line: {line!r}")
    key, value = line.split(":", 1)
    key = key.strip()
    value = value.strip()
    if value in {">", "|"}:
        value = ""
    if not key:
        raise LaunchError(f"{path.relative_to(ROOT)}: empty YAML key in line: {line!r}")
    return key, value


def load_leash() -> dict[str, object]:
    try:
        payload = json.loads(LEASH_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise LaunchError(f"missing leash config: {LEASH_PATH.relative_to(ROOT)}") from error
    except json.JSONDecodeError as error:
        raise LaunchError(f"{LEASH_PATH.relative_to(ROOT)} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise LaunchError("leash config must be a JSON object")
    return payload


def validate_job(
    path: Path,
    job: dict[str, object],
    leash: dict[str, object],
    allow_approval_required: bool,
) -> None:
    required = ("hardware", "timeout", "namespace", "image", "env", "command")
    missing = [key for key in required if key not in job]
    if missing:
        raise LaunchError(f"{path.relative_to(ROOT)} missing keys: {', '.join(missing)}")

    namespace = str(job["namespace"])
    if namespace != leash.get("namespace"):
        raise LaunchError(f"{path.relative_to(ROOT)} namespace {namespace!r} does not match leash")

    hardware = str(job["hardware"])
    allowed_hardware = as_string_set(leash.get("hardware_allowed"))
    if hardware not in allowed_hardware:
        raise LaunchError(f"{path.relative_to(ROOT)} hardware {hardware!r} is not allowed")

    for denied in as_string_set(leash.get("hardware_denied")):
        if denied and denied in hardware:
            raise LaunchError(f"{path.relative_to(ROOT)} hardware {hardware!r} is denied")

    job_name = path.name
    approval_required = as_string_set(leash.get("jobs_human_approval_required"))
    allowed_jobs = as_string_set(leash.get("jobs_allowed"))
    if job_name in approval_required and not allow_approval_required:
        raise LaunchError(f"{job_name} requires --allow-approval-required")
    if job_name not in allowed_jobs and job_name not in approval_required:
        raise LaunchError(f"{job_name} is not listed in the leash config")

    if not str(job["timeout"]):
        raise LaunchError(f"{path.relative_to(ROOT)} must set timeout")

    try:
        shlex.split(str(job["command"]))
    except ValueError as error:
        raise LaunchError(f"{path.relative_to(ROOT)} command is not shell-parseable: {error}") from error


def render_command(job: dict[str, object], *, detach: bool) -> list[str]:
    command = [
        "hf",
        "jobs",
        "run",
        "--namespace",
        str(job["namespace"]),
        "--flavor",
        str(job["hardware"]),
        "--timeout",
        str(job["timeout"]),
    ]
    if detach:
        command.append("--detach")

    env = job.get("env", {})
    if not isinstance(env, dict):
        raise LaunchError("job env must be a mapping")
    for key, raw_value in sorted(env.items()):
        if key == "HF_TOKEN" and raw_value == "${HF_TOKEN}":
            command.extend(["--secrets", "HF_TOKEN"])
            continue
        command.extend(["--env", f"{key}={resolve_env_value(str(raw_value))}"])

    command.append(str(job["image"]))
    command.append("--")
    command.extend(shlex.split(str(job["command"])))
    return command


def resolve_env_value(value: str) -> str:
    match = ENV_EXPR_RE.fullmatch(value)
    if not match:
        return value

    name, default = match.groups()
    return os.environ.get(name, default or "")


def as_string_set(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value}


if __name__ == "__main__":
    raise SystemExit(main())
