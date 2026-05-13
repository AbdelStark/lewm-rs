#!/usr/bin/env python3
"""Validate checked-in HF Jobs specs without requiring HF credentials."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ENV = {
    "RUST_LOG",
    "HF_TOKEN",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
}

JOB_SPECS = {
    "smoke_pusht.yaml": {
        "hardware": "l4",
        "timeout": "30m",
        "command_tokens": [
            "lewm-train smoke",
            "--config configs/pusht.toml",
            "--steps 200",
            "--batch-size 16",
            "--max-steps 200",
            "python python/upload_checkpoints.py",
        ],
    },
    "short_pusht.yaml": {
        "hardware": "a10g-large",
        "timeout": "2h",
        "command_tokens": [
            "lewm-train train",
            "--config configs/pusht.toml",
            "--data-dir /tmp/data",
            "--max-steps 7500",
            "python python/upload_checkpoints.py",
        ],
    },
    "train_pusht.yaml": {
        "hardware": "a10g-large",
        "timeout": "12h",
        "command_tokens": [
            "lewm-train train",
            "--config configs/pusht.toml",
            "--data-dir /tmp/data",
            "--output-dir /tmp/out",
            "--resume-if-present",
            "python python/upload_checkpoints.py",
        ],
    },
    "smoke_so100.yaml": {
        "hardware": "l4",
        "timeout": "30m",
        "requires_upload": False,
        "command_tokens": [
            "hf download AbdelStark/so100-pickplace-lewm-ready",
            "--repo-type dataset",
            "lewm-train smoke",
            "--config configs/so100.toml",
            "--data-dir /tmp/data/so100",
            "--max-steps 200",
        ],
    },
    "short_so100.yaml": {
        "hardware": "a10g-large",
        "timeout": "2h",
        "requires_upload": False,
        "command_tokens": [
            "hf download AbdelStark/so100-pickplace-lewm-ready",
            "--repo-type dataset",
            "lewm-train train",
            "--config configs/so100.toml",
            "--data-dir /tmp/data/so100",
            "--set training.epochs=1",
            "--set experimental.subset_name=so100-short",
        ],
    },
}


def main() -> int:
    failures: list[str] = []

    for name, expected in JOB_SPECS.items():
        path = ROOT / "jobs" / name
        if not path.exists():
            failures.append(f"{path}: missing job spec")
            continue

        text = path.read_text(encoding="utf-8")
        fields = parse_top_level_fields(text)
        env_keys = parse_env_keys(text)
        command = normalize_command(fields.get("command", ""))

        for field in ("hardware", "timeout", "namespace", "image", "command"):
            if field not in fields:
                failures.append(f"{path}: missing top-level {field!r}")

        if fields.get("hardware") != expected["hardware"]:
            failures.append(
                f"{path}: hardware must be {expected['hardware']!r}, got {fields.get('hardware')!r}"
            )

        if fields.get("timeout") != expected["timeout"]:
            failures.append(
                f"{path}: timeout must be {expected['timeout']!r}, got {fields.get('timeout')!r}"
            )

        missing_env = sorted(REQUIRED_ENV - env_keys)
        if missing_env:
            failures.append(f"{path}: missing env passthrough keys {missing_env}")

        for token in expected["command_tokens"]:
            if token not in command:
                failures.append(f"{path}: command missing {token!r}")

        if expected.get("requires_upload", True):
            upload_pos = command.rfind("python python/upload_checkpoints.py")
            train_pos = max(command.rfind("lewm-train smoke"), command.rfind("lewm-train train"))
            if upload_pos <= train_pos:
                failures.append(f"{path}: upload_checkpoints.py must run after lewm-train")

    validate_intern_config(failures)

    if failures:
        for failure in failures:
            print(f"check_jobs: {failure}", file=sys.stderr)
        return 1

    print("check_jobs: HF Jobs specs ok")
    return 0


def parse_top_level_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    lines = text.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index]
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$", line)
        if not match:
            index += 1
            continue

        key, value = match.groups()
        if value == ">":
            block: list[str] = []
            index += 1
            while index < len(lines) and (not lines[index] or lines[index].startswith(" ")):
                block.append(lines[index])
                index += 1
            fields[key] = "\n".join(block)
            continue

        fields[key] = value
        index += 1

    return fields


def parse_env_keys(text: str) -> set[str]:
    keys: set[str] = set()
    lines = text.splitlines()
    in_env = False

    for line in lines:
        if line == "env:":
            in_env = True
            continue
        if in_env and line and not line.startswith(" "):
            break
        if in_env:
            match = re.match(r"^\s{2}([A-Za-z_][A-Za-z0-9_]*):", line)
            if match:
                keys.add(match.group(1))

    return keys


def normalize_command(command: str) -> str:
    return " ".join(command.split())


def validate_intern_config(failures: list[str]) -> None:
    path = ROOT / ".ml-intern" / "cli_agent_config.json"
    if not path.exists():
        failures.append(f"{path}: missing ml-intern leash config")
        return

    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        failures.append(f"{path}: invalid JSON: {error}")
        return

    approval_required = set(config.get("jobs_human_approval_required", []))
    jobs_allowed = set(config.get("jobs_allowed", []))

    if "train_pusht.yaml" not in approval_required:
        failures.append(f"{path}: train_pusht.yaml must require human approval")
    if "train_pusht.yaml" in jobs_allowed:
        failures.append(f"{path}: train_pusht.yaml must not be pre-approved")


if __name__ == "__main__":
    raise SystemExit(main())
