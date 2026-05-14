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
EXPECTED_NAMESPACE = "abdelstark"
EXPECTED_IMAGE = "ghcr.io/abdelstark/lewm-rs:latest"
OPTIONAL_OTEL_ENDPOINT_VALUE = "${OTEL_ENDPOINT:-}"

JOB_SPECS = {
    "smoke_pusht.yaml": {
        "hardware": "l4x1",
        "timeout": "30m",
        "command_tokens": [
            "lewm-train smoke",
            "--config configs/pusht.toml",
            "--device cpu",
            "--steps 50",
            "--batch-size 4",
            "python python/upload_checkpoints.py",
            "--path-prefix smoke/pusht-$(date -u +%Y%m%dT%H%M%SZ)",
        ],
    },
    "short_pusht.yaml": {
        "hardware": "cpu-xl",
        "timeout": "45m",
        "command_tokens": [
            "hf download quentinll/lewm-pusht pusht_expert_train.h5.zst",
            "zstd -f -d /tmp/data/pusht_expert_train.h5.zst -o /tmp/data/pusht_expert_train.h5",
            "export HDF5_PLUGIN_PATH=$(python -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)')",
            "lewm-train train",
            "--config configs/pusht.toml",
            "--device cpu",
            "--data-dir /tmp/data",
            "--max-steps 10",
            "python python/upload_checkpoints.py",
            "--path-prefix train/pusht-full-module-lewm-short-$(date -u +%Y%m%dT%H%M%SZ)",
        ],
    },
    "train_pusht.yaml": {
        "hardware": "a10g-large",
        "timeout": "12h",
        "command_tokens": [
            "hf download quentinll/lewm-pusht pusht_expert_train.h5.zst",
            "zstd -f -d /tmp/data/pusht_expert_train.h5.zst -o /tmp/data/pusht_expert_train.h5",
            "export HDF5_PLUGIN_PATH=$(python -c 'import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)')",
            "lewm-train train",
            "--config configs/pusht.toml",
            "--data-dir /tmp/data",
            "--output-dir /tmp/out",
            "--resume-if-present",
            "--max-steps ${LEWM_MAX_STEPS:-1000}",
            "python python/upload_checkpoints.py",
            "--path-prefix train/pusht-full-module-lewm-$(date -u +%Y%m%dT%H%M%SZ)",
        ],
    },
    "smoke_so100.yaml": {
        "hardware": "l4x1",
        "timeout": "30m",
        "requires_upload": False,
        "command_tokens": [
            "hf download abdelstark/so100-pickplace-lewm-ready",
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
            "hf download abdelstark/so100-pickplace-lewm-ready",
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
    validate_image_contract(failures)

    for name, expected in JOB_SPECS.items():
        path = ROOT / "jobs" / name
        if not path.exists():
            failures.append(f"{path}: missing job spec")
            continue

        text = path.read_text(encoding="utf-8")
        fields = parse_top_level_fields(text)
        env_keys = parse_env_keys(text)
        env_values = parse_env_values(text)
        command = normalize_command(fields.get("command", ""))
        validate_shell_continuations(fields.get("command", ""), path, failures)

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

        if fields.get("namespace") != EXPECTED_NAMESPACE:
            failures.append(
                f"{path}: namespace must be {EXPECTED_NAMESPACE!r}, got {fields.get('namespace')!r}"
            )

        if fields.get("image") != EXPECTED_IMAGE:
            failures.append(f"{path}: image must be {EXPECTED_IMAGE!r}, got {fields.get('image')!r}")

        missing_env = sorted(REQUIRED_ENV - env_keys)
        if missing_env:
            failures.append(f"{path}: missing env passthrough keys {missing_env}")
        elif env_values.get("OTEL_EXPORTER_OTLP_ENDPOINT") != OPTIONAL_OTEL_ENDPOINT_VALUE:
            failures.append(
                f"{path}: OTEL_EXPORTER_OTLP_ENDPOINT must be {OPTIONAL_OTEL_ENDPOINT_VALUE!r}"
            )

        for token in expected["command_tokens"]:
            if token not in command:
                failures.append(f"{path}: command missing {token!r}")
        if "archive.tar.zst" in command:
            failures.append(f"{path}: command references removed PushT archive.tar.zst path")

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
        if value in {">", "|"}:
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


def validate_image_contract(failures: list[str]) -> None:
    dockerfile = ROOT / "Dockerfile"
    upload_script = ROOT / "python" / "upload_checkpoints.py"
    launcher_script = ROOT / "scripts" / "launch_hf_job.py"

    if not dockerfile.is_file():
        failures.append(f"{dockerfile}: missing training image Dockerfile")
    else:
        text = dockerfile.read_text(encoding="utf-8")
        for token in (
            "FROM rust:1.89.0-bookworm AS builder",
            "cargo build --locked --release -p lewm-train",
            "huggingface_hub==",
            'org.opencontainers.image.source="https://github.com/AbdelStark/lewm-rs"',
            "COPY python ./python",
            "CMD [\"lewm-train\", \"--help\"]",
        ):
            if token not in text:
                failures.append(f"{dockerfile}: missing image contract token {token!r}")

    if not upload_script.is_file():
        failures.append(f"{upload_script}: missing checkpoint upload helper")

    if not launcher_script.is_file():
        failures.append(f"{launcher_script}: missing HF Jobs launcher")
    else:
        launcher = launcher_script.read_text(encoding="utf-8")
        for token in ("hf", "jobs", "run", '"--"', "shlex.split"):
            if token not in launcher:
                failures.append(f"{launcher_script}: missing launcher contract token {token!r}")


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


def parse_env_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    lines = text.splitlines()
    in_env = False

    for line in lines:
        if line == "env:":
            in_env = True
            continue
        if in_env and line and not line.startswith(" "):
            break
        if in_env:
            match = re.match(r"^\s{2}([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
            if match:
                key, value = match.groups()
                values[key] = value.strip()

    return values


def normalize_command(command: str) -> str:
    return " ".join(command.split())


def validate_shell_continuations(command: str, path: Path, failures: list[str]) -> None:
    previous: str | None = None
    for raw_line in command.splitlines():
        line = raw_line.strip()
        if not line or line in {'"', 'bash -lc "', 'bash -c "'}:
            continue
        if line.startswith("--") and not (previous and previous.endswith("\\")):
            failures.append(f"{path}: shell flag line {line!r} must follow a backslash continuation")
        previous = line


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
