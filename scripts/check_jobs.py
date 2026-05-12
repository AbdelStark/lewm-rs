#!/usr/bin/env python3
"""Validate checked-in HF Jobs specs without requiring HF credentials."""

from __future__ import annotations

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

        upload_pos = command.rfind("python python/upload_checkpoints.py")
        train_pos = max(command.rfind("lewm-train smoke"), command.rfind("lewm-train train"))
        if upload_pos <= train_pos:
            failures.append(f"{path}: upload_checkpoints.py must run after lewm-train")

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


if __name__ == "__main__":
    raise SystemExit(main())
