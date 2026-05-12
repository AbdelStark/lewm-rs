#!/usr/bin/env python3
"""Validate HF Jobs YAML contracts without requiring a YAML dependency."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "jobs"

ALLOWED_HARDWARE = {"cpu-basic", "cpu-xl", "l4", "a10g-large"}
REQUIRED_TOP_LEVEL = {"name", "hardware", "timeout", "namespace", "image", "env", "command"}
TIMEOUT_RE = re.compile(r"^[1-9][0-9]*[mh]$")

EXPECTED_JOBS = {
    "smoke_so100.yaml": {
        "hardware": "l4",
        "timeout": "30m",
        "fragments": [
            "hf download AbdelStark/so100-pickplace-lewm-ready",
            "--repo-type dataset",
            "lewm-train smoke",
            "--config configs/so100.toml",
            "--max-steps 200",
        ],
    },
    "short_so100.yaml": {
        "hardware": "a10g-large",
        "timeout": "2h",
        "fragments": [
            "hf download AbdelStark/so100-pickplace-lewm-ready",
            "--repo-type dataset",
            "lewm-train train",
            "--config configs/so100.toml",
            "--set training.epochs=1",
            "--set experimental.subset_name=so100-short",
        ],
    },
}


class JobSpecError(Exception):
    """Raised when a job spec violates the local schema contract."""


def main() -> int:
    try:
        check_jobs()
    except JobSpecError as error:
        print(f"check_jobs: {error}", file=sys.stderr)
        return 1
    print("check_jobs: jobs ok")
    return 0


def check_jobs() -> None:
    if not JOBS_DIR.is_dir():
        raise JobSpecError("jobs/ directory is missing")

    paths = sorted(JOBS_DIR.glob("*.yaml"))
    if not paths:
        raise JobSpecError("jobs/ contains no .yaml files")

    seen = {path.name for path in paths}
    missing = sorted(set(EXPECTED_JOBS) - seen)
    if missing:
        raise JobSpecError(f"missing expected job specs: {', '.join(missing)}")

    for path in paths:
        spec = parse_simple_yaml(path)
        validate_common(path, spec)
        if path.name in EXPECTED_JOBS:
            validate_expected(path, spec, EXPECTED_JOBS[path.name])


def parse_simple_yaml(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    data: dict[str, object] = {}
    index = 0

    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        if raw.startswith(" "):
            index += 1
            continue

        match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):(?:\s*(.*))?$", raw)
        if match is None:
            raise JobSpecError(f"{path}: could not parse line {index + 1}: {raw!r}")

        key, value = match.groups()
        value = value or ""
        if key == "command" and value in {">", "|"}:
            block, index = collect_indented_block(lines, index + 1)
            data[key] = " ".join(block.split())
            continue
        if key == "env" and value == "":
            env, index = collect_env(lines, index + 1)
            data[key] = env
            continue

        data[key] = strip_quotes(value)
        index += 1

    return data


def collect_indented_block(lines: list[str], start: int) -> tuple[str, int]:
    block: list[str] = []
    index = start
    while index < len(lines):
        raw = lines[index]
        if raw and not raw.startswith(" "):
            break
        block.append(raw.strip())
        index += 1
    return "\n".join(block), index


def collect_env(lines: list[str], start: int) -> tuple[dict[str, str], int]:
    env: dict[str, str] = {}
    index = start
    while index < len(lines):
        raw = lines[index]
        if raw and not raw.startswith(" "):
            break
        stripped = raw.strip()
        if stripped:
            match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", stripped)
            if match is None:
                raise JobSpecError(f"invalid env line: {raw!r}")
            key, value = match.groups()
            env[key] = strip_quotes(value)
        index += 1
    return env, index


def strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def validate_common(path: Path, spec: dict[str, object]) -> None:
    missing = sorted(REQUIRED_TOP_LEVEL - set(spec))
    if missing:
        raise JobSpecError(f"{path}: missing required keys: {', '.join(missing)}")

    hardware = expect_string(path, spec, "hardware")
    if hardware not in ALLOWED_HARDWARE:
        raise JobSpecError(f"{path}: hardware {hardware!r} is not allowed")

    timeout = expect_string(path, spec, "timeout")
    if not TIMEOUT_RE.fullmatch(timeout):
        raise JobSpecError(f"{path}: timeout {timeout!r} must look like 30m or 2h")

    namespace = expect_string(path, spec, "namespace")
    if namespace != "AbdelStark":
        raise JobSpecError(f"{path}: namespace must be AbdelStark")

    command = expect_string(path, spec, "command")
    if not command:
        raise JobSpecError(f"{path}: command is empty")

    env = spec["env"]
    if not isinstance(env, dict):
        raise JobSpecError(f"{path}: env must be a mapping")
    for key in ("RUST_LOG", "HF_HOME", "TRACKIO_PROJECT"):
        if key not in env:
            raise JobSpecError(f"{path}: env.{key} is required")


def validate_expected(path: Path, spec: dict[str, object], expected: dict[str, object]) -> None:
    hardware = expect_string(path, spec, "hardware")
    timeout = expect_string(path, spec, "timeout")
    command = expect_string(path, spec, "command")

    if hardware != expected["hardware"]:
        raise JobSpecError(f"{path}: expected hardware {expected['hardware']!r}, got {hardware!r}")
    if timeout != expected["timeout"]:
        raise JobSpecError(f"{path}: expected timeout {expected['timeout']!r}, got {timeout!r}")

    for fragment in expected["fragments"]:
        if fragment not in command:
            raise JobSpecError(f"{path}: command missing fragment {fragment!r}")


def expect_string(path: Path, spec: dict[str, object], key: str) -> str:
    value = spec[key]
    if not isinstance(value, str):
        raise JobSpecError(f"{path}: {key} must be a string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
