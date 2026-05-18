#!/usr/bin/env python3
"""Validate the committed PushT warm-start Hub audit report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_REPORT = Path("reports/pusht_warmstart_hub_audit.json")
EXPECTED_REPO = "abdelstark/lewm-rs-pusht"
EXPECTED_VERIFIER = "scripts/check_warmstart_source.py"
EXPECTED_SCHEMA_VERSION = "1.1.0"
EXPECTED_KIND = "lewm-rs-pusht-bounded-module-lewm-record"
EXPECTED_PARAM_COUNT = 41_856


class HubAuditReportError(RuntimeError):
    """Raised when the committed Hub audit report is malformed."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Hub audit report path ({DEFAULT_REPORT})",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def require_str(payload: dict[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise HubAuditReportError(f"{path}: {key} must be a non-empty string")
    return value


def require_int(payload: dict[str, Any], key: str, path: Path) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value < 0:
        raise HubAuditReportError(f"{path}: {key} must be a non-negative integer")
    return value


def validate_candidate(candidate: Any, path: Path, index: int) -> str:
    if not isinstance(candidate, dict):
        raise HubAuditReportError(f"{path}: candidates[{index}] must be an object")
    candidate_path = require_str(candidate, "path", path)
    if not candidate_path.endswith(".mpk"):
        raise HubAuditReportError(f"{path}: candidates[{index}].path must end in .mpk")
    require_str(candidate, "download_url", path)

    size = candidate.get("size")
    if not isinstance(size, int) or size <= 0:
        raise HubAuditReportError(f"{path}: candidates[{index}].size must be a positive integer")

    status = require_str(candidate, "status", path)
    if status not in {"compatible", "rejected"}:
        raise HubAuditReportError(
            f"{path}: candidates[{index}].status must be compatible or rejected"
        )
    reason = require_str(candidate, "reason", path)
    if status == "rejected" and candidate_path not in reason:
        raise HubAuditReportError(
            f"{path}: candidates[{index}].reason must name the rejected path"
        )
    return status


def validate_report(payload: dict[str, Any], path: Path) -> None:
    if payload.get("schema_version") != "1.0.0":
        raise HubAuditReportError(f"{path}: schema_version must be '1.0.0'")
    require_str(payload, "updated", path)
    if payload.get("repo") != EXPECTED_REPO:
        raise HubAuditReportError(f"{path}: repo must be {EXPECTED_REPO!r}")
    require_str(payload, "revision", path)
    require_str(payload, "source", path)
    if payload.get("source_verifier") != EXPECTED_VERIFIER:
        raise HubAuditReportError(f"{path}: source_verifier must be {EXPECTED_VERIFIER!r}")

    expected = payload.get("expected")
    if not isinstance(expected, dict):
        raise HubAuditReportError(f"{path}: expected must be an object")
    if expected.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise HubAuditReportError(
            f"{path}: expected.schema_version must be {EXPECTED_SCHEMA_VERSION!r}"
        )
    if expected.get("kind") != EXPECTED_KIND:
        raise HubAuditReportError(f"{path}: expected.kind must be {EXPECTED_KIND!r}")
    if expected.get("param_count") != EXPECTED_PARAM_COUNT:
        raise HubAuditReportError(f"{path}: expected.param_count must be {EXPECTED_PARAM_COUNT}")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise HubAuditReportError(f"{path}: candidates must be a list")
    candidate_count = require_int(payload, "candidate_count", path)
    compatible_count = require_int(payload, "compatible_count", path)
    if candidate_count != len(candidates):
        raise HubAuditReportError(f"{path}: candidate_count does not match candidates length")

    statuses = [validate_candidate(candidate, path, index) for index, candidate in enumerate(candidates)]
    actual_compatible = statuses.count("compatible")
    if compatible_count != actual_compatible:
        raise HubAuditReportError(f"{path}: compatible_count does not match candidate statuses")

    status = require_str(payload, "status", path)
    if compatible_count == 0 and status != "blocked":
        raise HubAuditReportError(f"{path}: status must be 'blocked' when compatible_count is 0")
    if compatible_count > 0 and status != "ready":
        raise HubAuditReportError(f"{path}: status must be 'ready' when compatible_count is positive")
    require_str(payload, "summary", path)


def main() -> int:
    args = parse_args()
    path = resolve_path(args.path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise HubAuditReportError(f"{path}: root must be an object")
        validate_report(payload, path)
    except FileNotFoundError:
        print(f"check_pusht_warmstart_hub_audit_report.py: missing report: {path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"check_pusht_warmstart_hub_audit_report.py: {path}: invalid JSON: {exc}", file=sys.stderr)
        return 1
    except HubAuditReportError as exc:
        print(f"check_pusht_warmstart_hub_audit_report.py: {exc}", file=sys.stderr)
        return 1

    print(
        "PushT warm-start Hub audit report ok: "
        f"candidates={payload['candidate_count']} compatible={payload['compatible_count']} "
        f"status={payload['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
