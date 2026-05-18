#!/usr/bin/env python3
"""Validate the committed PushT full-safetensors Hub audit report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_REPORT = Path("reports/pusht_full_safetensors_hub_audit.json")
EXPECTED_REPO = "abdelstark/lewm-rs-pusht"
EXPECTED_VERIFIER = "scripts/f1_export_pusht_onnx.py --execute"
EXPECTED_DESTINATION_TENSOR_COUNT = 255
EXPECTED_SOURCE_KEY_COUNT = 303
REQUIRED_PATH_RE = re.compile(
    r"^train/pusht-full-burn-jepa-\d{8}T\d{6}Z/step_0050000\.safetensors$"
)


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


def require_str_list(value: Any, path: Path, context: str) -> list[str]:
    if not isinstance(value, list):
        raise HubAuditReportError(f"{path}: {context} must be a list")
    if not all(isinstance(item, str) and item for item in value):
        raise HubAuditReportError(f"{path}: {context} must contain non-empty strings")
    return value


def validate_observed(observed: Any, path: Path, index: int) -> dict[str, Any]:
    if not isinstance(observed, dict):
        raise HubAuditReportError(f"{path}: candidates[{index}].observed must be an object")
    observed_format = require_str(observed, "format", path)
    if observed_format == "safetensors":
        tensor_count = observed.get("tensor_count")
        if not isinstance(tensor_count, int) or tensor_count < 0:
            raise HubAuditReportError(
                f"{path}: candidates[{index}].observed.tensor_count must be a non-negative int"
            )
        require_str_list(observed.get("first_tensors"), path, f"candidates[{index}].first_tensors")
        require_str_list(observed.get("dtypes"), path, f"candidates[{index}].dtypes")
        require_str_list(observed.get("metadata_keys"), path, f"candidates[{index}].metadata_keys")
        require_str(observed, "inferred_family", path)
        return observed
    if observed_format == "unsupported":
        require_str(observed, "error", path)
        return observed
    raise HubAuditReportError(
        f"{path}: candidates[{index}].observed.format must be safetensors or unsupported"
    )


def validate_violations(violations: Any, path: Path, index: int, status: str) -> None:
    require_str_list(violations, path, f"candidates[{index}].violations")
    if status == "ready_for_contract_check" and violations:
        raise HubAuditReportError(
            f"{path}: candidates[{index}].violations must be empty for ready candidates"
        )
    if status == "rejected" and not violations:
        raise HubAuditReportError(
            f"{path}: candidates[{index}].violations must explain rejected candidates"
        )


def validate_candidate(candidate: Any, path: Path, index: int) -> str:
    if not isinstance(candidate, dict):
        raise HubAuditReportError(f"{path}: candidates[{index}] must be an object")
    candidate_path = require_str(candidate, "path", path)
    if not candidate_path.endswith(".safetensors"):
        raise HubAuditReportError(f"{path}: candidates[{index}].path must end in .safetensors")
    require_str(candidate, "download_url", path)
    size = candidate.get("size")
    if size is not None and (not isinstance(size, int) or size <= 0):
        raise HubAuditReportError(f"{path}: candidates[{index}].size must be positive or null")

    status = require_str(candidate, "status", path)
    if status not in {"ready_for_contract_check", "rejected"}:
        raise HubAuditReportError(
            f"{path}: candidates[{index}].status must be ready_for_contract_check or rejected"
        )
    reason = require_str(candidate, "reason", path)
    if status == "rejected" and candidate_path not in reason:
        raise HubAuditReportError(
            f"{path}: candidates[{index}].reason must name the rejected path"
        )
    observed = validate_observed(candidate.get("observed"), path, index)
    validate_violations(candidate.get("violations"), path, index, status)

    if status == "ready_for_contract_check":
        if REQUIRED_PATH_RE.fullmatch(candidate_path) is None:
            raise HubAuditReportError(
                f"{path}: candidates[{index}].path is not an F1 full-run path"
            )
        if observed.get("tensor_count") != EXPECTED_DESTINATION_TENSOR_COUNT:
            raise HubAuditReportError(
                f"{path}: candidates[{index}].observed.tensor_count must be "
                f"{EXPECTED_DESTINATION_TENSOR_COUNT}"
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
    require_str(expected, "required_path_pattern", path)
    if expected.get("destination_tensor_count") != EXPECTED_DESTINATION_TENSOR_COUNT:
        raise HubAuditReportError(
            f"{path}: expected.destination_tensor_count must be "
            f"{EXPECTED_DESTINATION_TENSOR_COUNT}"
        )
    if expected.get("source_key_count_after_contract_check") != EXPECTED_SOURCE_KEY_COUNT:
        raise HubAuditReportError(
            f"{path}: expected.source_key_count_after_contract_check must be "
            f"{EXPECTED_SOURCE_KEY_COUNT}"
        )

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise HubAuditReportError(f"{path}: candidates must be a list")
    candidate_count = require_int(payload, "candidate_count", path)
    ready_count = require_int(payload, "ready_count", path)
    if candidate_count != len(candidates):
        raise HubAuditReportError(f"{path}: candidate_count does not match candidates length")

    statuses = [validate_candidate(candidate, path, index) for index, candidate in enumerate(candidates)]
    actual_ready = statuses.count("ready_for_contract_check")
    if ready_count != actual_ready:
        raise HubAuditReportError(f"{path}: ready_count does not match candidate statuses")

    status = require_str(payload, "status", path)
    if ready_count == 0 and status != "blocked":
        raise HubAuditReportError(f"{path}: status must be 'blocked' when ready_count is 0")
    if ready_count > 0 and status != "ready_for_contract_check":
        raise HubAuditReportError(
            f"{path}: status must be 'ready_for_contract_check' when ready_count is positive"
        )
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
        print(f"check_pusht_full_safetensors_hub_audit_report.py: missing report: {path}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(
            f"check_pusht_full_safetensors_hub_audit_report.py: {path}: invalid JSON: {exc}",
            file=sys.stderr,
        )
        return 1
    except HubAuditReportError as exc:
        print(f"check_pusht_full_safetensors_hub_audit_report.py: {exc}", file=sys.stderr)
        return 1

    print(
        "PushT full safetensors Hub audit report ok: "
        f"candidates={payload['candidate_count']} ready={payload['ready_count']} "
        f"status={payload['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
