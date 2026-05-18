#!/usr/bin/env python3
"""Validate the local full PushT Burn/Jepa contract smoke evidence report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_REPORT = Path("reports/full_pusht_contract_smoke.json")
EXPECTED_PYTORCH_KEYS = 303
EXPECTED_BURN_TENSORS = 255
MIN_FULL_CHECKPOINT_BYTES = 10_000_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SmokeReportError(RuntimeError):
    """Raised when the smoke report is missing or malformed."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"smoke report path ({DEFAULT_REPORT})",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def load_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SmokeReportError(f"missing full PushT contract smoke report: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SmokeReportError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SmokeReportError(f"{path}: report root must be an object")
    return payload


def require_str(payload: dict[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SmokeReportError(f"{path}: {key} must be a non-empty string")
    return value


def require_int(payload: dict[str, Any], key: str, path: Path) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise SmokeReportError(f"{path}: {key} must be an integer")
    return value


def require_command(payload: dict[str, Any], key: str, path: Path) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise SmokeReportError(f"{path}: {key} must be a non-empty string list")
    if not all(isinstance(item, str) and item for item in value):
        raise SmokeReportError(f"{path}: {key} must contain only non-empty strings")
    return value


def require_contract(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    value = payload.get("contract")
    if not isinstance(value, dict):
        raise SmokeReportError(f"{path}: contract must be an object")
    return value


def validate_report(payload: dict[str, Any], path: Path) -> None:
    if payload.get("schema_version") != "1.0.0":
        raise SmokeReportError(f"{path}: schema_version must be '1.0.0'")
    require_str(payload, "generated_at", path)
    config = require_str(payload, "config", path)
    if config != "configs/pusht.toml":
        raise SmokeReportError(f"{path}: config must be 'configs/pusht.toml', got {config!r}")

    steps = require_int(payload, "steps", path)
    if steps <= 0:
        raise SmokeReportError(f"{path}: steps must be positive")
    checkpoint_size = require_int(payload, "checkpoint_size_bytes", path)
    if checkpoint_size < MIN_FULL_CHECKPOINT_BYTES:
        raise SmokeReportError(
            f"{path}: checkpoint_size_bytes {checkpoint_size} is too small for full Burn/Jepa"
        )
    require_str(payload, "checkpoint", path)
    require_str(payload, "output_dir", path)

    contract = require_contract(payload, path)
    recovered = require_int(contract, "recovered_pytorch_keys", path)
    expected = require_int(contract, "expected_pytorch_keys", path)
    burn_tensors = require_int(contract, "burn_destination_tensors", path)
    if recovered != EXPECTED_PYTORCH_KEYS or expected != EXPECTED_PYTORCH_KEYS:
        raise SmokeReportError(
            f"{path}: expected {EXPECTED_PYTORCH_KEYS}/{EXPECTED_PYTORCH_KEYS} "
            f"recovered PyTorch keys, got {recovered}/{expected}"
        )
    if burn_tensors != EXPECTED_BURN_TENSORS:
        raise SmokeReportError(
            f"{path}: expected {EXPECTED_BURN_TENSORS} Burn destination tensors, got {burn_tensors}"
        )
    sha256 = require_str(contract, "safetensors_sha256", path)
    if SHA256_RE.fullmatch(sha256) is None:
        raise SmokeReportError(f"{path}: safetensors_sha256 must be lowercase hex SHA-256")

    train_command = require_command(payload, "train_command", path)
    contract_command = require_command(payload, "contract_command", path)
    train_text = " ".join(train_command)
    contract_text = " ".join(contract_command)
    for token in (
        "lewm-train",
        "configs/pusht.toml",
        'experimental.pusht_train_mode="full_burn_jepa"',
        "--device cpu",
    ):
        if token not in train_text:
            raise SmokeReportError(f"{path}: train_command missing {token!r}")
    if "--max-steps" not in train_command or str(steps) not in train_command:
        raise SmokeReportError(f"{path}: train_command must include --max-steps {steps}")
    for token in ("python/export_onnx.py", "--check-contract-only", "--frozen"):
        if token not in contract_text:
            raise SmokeReportError(f"{path}: contract_command missing {token!r}")


def main() -> int:
    path = resolve_path(parse_args().path)
    try:
        payload = load_report(path)
        validate_report(payload, path)
    except SmokeReportError as exc:
        print(f"check_full_pusht_contract_smoke_report.py: {exc}", file=sys.stderr)
        return 1

    print(
        "full PushT contract smoke report ok: "
        f"recovered={EXPECTED_PYTORCH_KEYS}/{EXPECTED_PYTORCH_KEYS} "
        f"burn_tensors={EXPECTED_BURN_TENSORS}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
