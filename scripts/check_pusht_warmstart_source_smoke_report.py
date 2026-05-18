#!/usr/bin/env python3
"""Validate the local bounded PushT warm-start source smoke evidence report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_REPORT = Path("reports/pusht_warmstart_source_smoke.json")
EXPECTED_PARAM_COUNT = 41_856
EXPECTED_RECORD_KIND = "lewm-rs-pusht-bounded-module-lewm-record"
EXPECTED_RECORD_SCHEMA = "1.1.0"
EXPECTED_TRAIN_MODE = "pusht-bounded-module-lewm"
MIN_SOURCE_BYTES = 1_000_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class WarmstartSourceSmokeReportError(RuntimeError):
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
        raise WarmstartSourceSmokeReportError(
            f"missing PushT warm-start source smoke report: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise WarmstartSourceSmokeReportError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise WarmstartSourceSmokeReportError(f"{path}: report root must be an object")
    return payload


def require_str(payload: dict[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise WarmstartSourceSmokeReportError(f"{path}: {key} must be a non-empty string")
    return value


def require_int(payload: dict[str, Any], key: str, path: Path) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise WarmstartSourceSmokeReportError(f"{path}: {key} must be an integer")
    return value


def require_float(payload: dict[str, Any], key: str, path: Path) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise WarmstartSourceSmokeReportError(f"{path}: {key} must be numeric")
    return float(value)


def require_command(payload: dict[str, Any], key: str, path: Path) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise WarmstartSourceSmokeReportError(f"{path}: {key} must be a non-empty string list")
    if not all(isinstance(item, str) and item for item in value):
        raise WarmstartSourceSmokeReportError(f"{path}: {key} must contain only non-empty strings")
    return value


def require_object(payload: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise WarmstartSourceSmokeReportError(f"{path}: {key} must be an object")
    return value


def validate_report(payload: dict[str, Any], path: Path) -> None:
    if payload.get("schema_version") != "1.0.0":
        raise WarmstartSourceSmokeReportError(f"{path}: schema_version must be '1.0.0'")
    require_str(payload, "generated_at", path)
    config = require_str(payload, "config", path)
    if config != "configs/pusht.toml":
        raise WarmstartSourceSmokeReportError(
            f"{path}: config must be 'configs/pusht.toml', got {config!r}"
        )
    steps = require_int(payload, "steps", path)
    if steps <= 0:
        raise WarmstartSourceSmokeReportError(f"{path}: steps must be positive")
    checkpoint_size = require_int(payload, "checkpoint_size_bytes", path)
    if checkpoint_size < MIN_SOURCE_BYTES:
        raise WarmstartSourceSmokeReportError(
            f"{path}: checkpoint_size_bytes {checkpoint_size} is too small for bounded source"
        )
    require_str(payload, "checkpoint", path)
    require_str(payload, "output_dir", path)
    checkpoint_sha256 = require_str(payload, "checkpoint_sha256", path)
    if SHA256_RE.fullmatch(checkpoint_sha256) is None:
        raise WarmstartSourceSmokeReportError(f"{path}: checkpoint_sha256 must be SHA-256 hex")

    train = require_object(payload, "train", path)
    mode = require_str(train, "mode", path)
    if mode != EXPECTED_TRAIN_MODE:
        raise WarmstartSourceSmokeReportError(
            f"{path}: train.mode must be {EXPECTED_TRAIN_MODE!r}, got {mode!r}"
        )
    if "pusht-compatible-fixture" not in require_str(train, "data_source", path):
        raise WarmstartSourceSmokeReportError(
            f"{path}: train.data_source must identify the local PushT fixture"
        )
    final_loss = require_float(train, "final_loss", path)
    if final_loss < 0:
        raise WarmstartSourceSmokeReportError(f"{path}: train.final_loss must be non-negative")
    if require_int(train, "checkpoint_step", path) != steps:
        raise WarmstartSourceSmokeReportError(f"{path}: train.checkpoint_step must equal steps")
    if train.get("checkpoint_complete") is not True:
        raise WarmstartSourceSmokeReportError(f"{path}: train.checkpoint_complete must be true")

    record = require_object(payload, "record", path)
    if require_str(record, "schema_version", path) != EXPECTED_RECORD_SCHEMA:
        raise WarmstartSourceSmokeReportError(
            f"{path}: record.schema_version must be {EXPECTED_RECORD_SCHEMA!r}"
        )
    if require_str(record, "kind", path) != EXPECTED_RECORD_KIND:
        raise WarmstartSourceSmokeReportError(
            f"{path}: record.kind must be {EXPECTED_RECORD_KIND!r}"
        )
    if require_int(record, "step", path) != steps:
        raise WarmstartSourceSmokeReportError(f"{path}: record.step must equal steps")
    if require_int(record, "params", path) != EXPECTED_PARAM_COUNT:
        raise WarmstartSourceSmokeReportError(
            f"{path}: record.params must be {EXPECTED_PARAM_COUNT}"
        )
    if require_int(record, "adamw_params", path) != EXPECTED_PARAM_COUNT:
        raise WarmstartSourceSmokeReportError(
            f"{path}: record.adamw_params must be {EXPECTED_PARAM_COUNT}"
        )

    source_check = require_object(payload, "source_check", path)
    if require_int(source_check, "step", path) != steps:
        raise WarmstartSourceSmokeReportError(f"{path}: source_check.step must equal steps")
    if require_int(source_check, "params", path) != EXPECTED_PARAM_COUNT:
        raise WarmstartSourceSmokeReportError(
            f"{path}: source_check.params must be {EXPECTED_PARAM_COUNT}"
        )

    train_command = require_command(payload, "train_command", path)
    source_check_command = require_command(payload, "source_check_command", path)
    train_text = " ".join(train_command)
    source_check_text = " ".join(source_check_command)
    for token in ("lewm-train", "configs/pusht.toml", "--device cpu"):
        if token not in train_text:
            raise WarmstartSourceSmokeReportError(f"{path}: train_command missing {token!r}")
    if "full_burn_jepa" in train_text:
        raise WarmstartSourceSmokeReportError(
            f"{path}: train_command must exercise bounded PushT mode, not full Burn/Jepa"
        )
    if "--max-steps" not in train_command or str(steps) not in train_command:
        raise WarmstartSourceSmokeReportError(
            f"{path}: train_command must include --max-steps {steps}"
        )
    for token in ("scripts/check_warmstart_source.py", "--config", "configs/pusht.toml"):
        if token not in source_check_text:
            raise WarmstartSourceSmokeReportError(
                f"{path}: source_check_command missing {token!r}"
            )


def main() -> int:
    path = resolve_path(parse_args().path)
    try:
        payload = load_report(path)
        validate_report(payload, path)
    except WarmstartSourceSmokeReportError as exc:
        print(f"check_pusht_warmstart_source_smoke_report.py: {exc}", file=sys.stderr)
        return 1

    print(
        "PushT warm-start source smoke report ok: "
        f"params={EXPECTED_PARAM_COUNT} mode={EXPECTED_TRAIN_MODE}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
