#!/usr/bin/env python3
"""Validate the approval-gated SO-100 full-training job contract."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB_PATH = ROOT / "jobs" / "train_so100.yaml"
LEASH_PATH = ROOT / ".ml-intern" / "cli_agent_config.json"


class CheckError(RuntimeError):
    pass


def _parse_job(path: Path) -> dict[str, object]:
    if not path.exists():
        raise CheckError(f"missing job spec: {path.relative_to(ROOT)}")

    top: dict[str, object] = {}
    env: dict[str, str] = {}
    command_lines: list[str] = []
    section: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        if raw_line.startswith("  ") and section == "env":
            key, value = _split_key_value(raw_line.strip(), path)
            env[key] = value
            continue

        if raw_line.startswith("  ") and section == "command":
            command_lines.append(raw_line.strip())
            continue

        section = None
        key, value = _split_key_value(raw_line, path)
        if key == "env":
            section = "env"
            top["env"] = env
        elif key == "command":
            section = "command"
            top["command"] = " ".join(command_lines)
        else:
            top[key] = value

    if command_lines:
        top["command"] = "\n".join(command_lines)

    return top


def _split_key_value(line: str, path: Path) -> tuple[str, str]:
    if ":" not in line:
        raise CheckError(f"{path.relative_to(ROOT)}: invalid YAML line: {line!r}")

    key, value = line.split(":", 1)
    key = key.strip()
    value = value.strip()
    if value in {">", "|"}:
        value = ""
    if not key:
        raise CheckError(f"{path.relative_to(ROOT)}: empty YAML key in line: {line!r}")
    return key, value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def _expect_contains(haystack: str, needles: list[str], field: str) -> None:
    compact = re.sub(r"\s+", " ", haystack)
    for needle in needles:
        _require(needle in compact, f"{field} missing required fragment: {needle}")


def _load_leash(path: Path) -> dict[str, object]:
    if not path.exists():
        raise CheckError(f"missing intern leash: {path.relative_to(ROOT)}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckError(f"{path.relative_to(ROOT)} is not valid JSON: {exc}") from exc
    _require(isinstance(payload, dict), "intern leash must be a JSON object")
    return payload


def _validate_job(job: dict[str, object]) -> None:
    expected_top = {
        "name",
        "hardware",
        "timeout",
        "namespace",
        "image",
        "env",
        "command",
    }
    missing = sorted(expected_top.difference(job))
    _require(not missing, f"train_so100.yaml missing required keys: {', '.join(missing)}")

    _require(job["name"] == "so100-full-train", "job name must be so100-full-train")
    _require(job["hardware"] == "a10g-large", "train_so100.yaml must use a10g-large")
    _require(job["timeout"] == "6h", "train_so100.yaml must use timeout 6h")
    _require(job["namespace"] == "abdelstark", "train_so100.yaml namespace must be abdelstark")
    _require(
        job["image"] == "ghcr.io/abdelstark/lewm-rs:latest",
        "train_so100.yaml image must be ghcr.io/abdelstark/lewm-rs:latest",
    )

    env = job["env"]
    _require(isinstance(env, dict), "train_so100.yaml env must be a mapping")
    for key in (
        "RUST_LOG",
        "HF_TOKEN",
        "HF_HOME",
        "TRACKIO_PROJECT",
        "TRACKIO_RUN",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "SO100_VARIANT",
    ):
        _require(key in env, f"train_so100.yaml env missing {key}")
    _require(
        env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "${OTEL_ENDPOINT:-}",
        "OTEL_EXPORTER_OTLP_ENDPOINT must default to empty for optional OTEL",
    )
    _require(env["SO100_VARIANT"] == "so100", "SO100_VARIANT default must be so100")

    command = str(job["command"])
    _expect_contains(
        command,
        [
            "hf download abdelstark/so100-pickplace-lewm-ready",
            "--repo-type dataset",
            "--local-dir /tmp/data/so100",
            "lewm-train train",
            "--config configs/${SO100_VARIANT}.toml",
            "--data-dir /tmp/data/so100",
            "--output-dir /tmp/out/so100-${SO100_VARIANT}",
            "--resume-if-present",
        ],
        "train_so100.yaml command",
    )


def _validate_leash(leash: dict[str, object]) -> None:
    _require(leash.get("schema_version") == "1.0.0", "intern leash schema_version must be 1.0.0")
    _require(leash.get("namespace") == "abdelstark", "intern leash namespace must be abdelstark")

    allowed = leash.get("hardware_allowed")
    denied = leash.get("hardware_denied")
    approval_required = leash.get("jobs_human_approval_required")
    jobs_allowed = leash.get("jobs_allowed")

    _require(isinstance(allowed, list), "intern leash hardware_allowed must be a list")
    _require("l4x1" in allowed, "intern leash must allow l4x1 hardware")
    _require("a10g-large" in allowed, "intern leash must allow a10g-large hardware")
    _require(isinstance(denied, list), "intern leash hardware_denied must be a list")
    _require(any("a100" in str(item) for item in denied), "intern leash must deny a100 tiers")
    _require(any("h100" in str(item) for item in denied), "intern leash must deny h100 tiers")
    _require(
        isinstance(approval_required, list) and "train_so100.yaml" in approval_required,
        "train_so100.yaml must be listed under jobs_human_approval_required",
    )
    _require(
        isinstance(jobs_allowed, list) and "train_so100.yaml" not in jobs_allowed,
        "train_so100.yaml must not be listed under jobs_allowed",
    )


def _render_launch(job: dict[str, object]) -> str:
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
            str(job["image"]),
        ]
    command.append("--")
    command.extend(shlex.split(str(job["command"])))
    return shlex.join(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-render",
        action="store_true",
        help="print the hf jobs command that would be launched after human approval",
    )
    args = parser.parse_args()

    try:
        job = _parse_job(JOB_PATH)
        leash = _load_leash(LEASH_PATH)
        _validate_job(job)
        _validate_leash(leash)
    except CheckError as exc:
        print(f"check_train_so100_job.py: {exc}", file=sys.stderr)
        return 1

    if args.dry_render:
        print(_render_launch(job))
    else:
        print("train_so100 job contract ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
