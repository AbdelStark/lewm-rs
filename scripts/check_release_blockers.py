#!/usr/bin/env python3
"""Validate release blockers and fail release acceptance while any remain open."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_BLOCKERS = Path("conformance/release_blockers.json")
OPEN_STATUSES = {"blocked", "pending", "open"}
RESOLVED_STATUS = "resolved"
EXPECTED_BACKLOG_ISSUES = {f"F{index}": 242 + index for index in range(1, 14)}
RELEASE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "F2": ("F1", "F3"),
    "F4": ("F2",),
    "F5": ("F3",),
    "F6": ("F1", "F3"),
    "F7": ("F2",),
    "F8": ("F3",),
    "F9": ("F7", "F8"),
    "F10": ("F7", "F8"),
    "F13": tuple(f"F{index}" for index in range(1, 13)),
}
REQUIRED_EVIDENCE_BY_ID: dict[str, tuple[str, ...]] = {
    "F1": (
        "reports/full_burn_jepa_training_gap.md",
        "reports/full_pusht_contract_smoke.json",
        "reports/pusht_full_safetensors_hub_audit.json",
        "scripts/f1_export_pusht_onnx.py",
        "scripts/audit_pusht_full_safetensors.py",
        "scripts/check_pusht_full_safetensors_hub_audit_report.py",
        "scripts/verify_runtime_image.py",
        ".github/workflows/runtime-image.yml",
        "reports/phase_a_handoff.json",
        "reports/phase_a_approval.json",
        "scripts/check_phase_a_approval.py",
    ),
    "F3": (
        "jobs/train_so100_warmstart.yaml",
        ".ml-intern/cli_agent_config.json",
        "reports/phase_a_approval.json",
        "reports/pusht_warmstart_source_smoke.json",
        "reports/pusht_warmstart_hub_audit.json",
        "scripts/check_phase_a_approval.py",
        "scripts/pusht_warmstart_source_smoke.py",
        "scripts/check_pusht_warmstart_source_smoke_report.py",
        "scripts/audit_pusht_warmstart_sources.py",
        "scripts/check_pusht_warmstart_hub_audit_report.py",
        "reports/phase_a_handoff.json",
    ),
    "F13": (
        "conformance/release_blockers.json",
    ),
}
REQUIRED_RESOLUTION_SUBSTRINGS_BY_ID: dict[str, tuple[str, ...]] = {
    "F1": (
        "GHCR runtime image tag",
        "scripts/verify_runtime_image.py",
        "255-tensor Burn/Jepa safetensors layout",
        "onnx-full/",
    ),
    "F11": (
        "ghcr.io/abdelstark/lewm-rs",
        "container job passes",
        "latest GHCR image",
    ),
}


class BlockerError(RuntimeError):
    """Raised when the blocker file is malformed."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_BLOCKERS,
        help=f"release blocker file, relative to repo root by default ({DEFAULT_BLOCKERS})",
    )
    parser.add_argument(
        "--allow-open",
        action="store_true",
        help="validate structure but do not fail when blockers remain open",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def require_str(entry: dict[str, Any], key: str, context: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise BlockerError(f"{context}.{key} must be a non-empty string")
    return value


def require_str_list(entry: dict[str, Any], key: str, context: str) -> list[str]:
    value = entry.get(key)
    if not isinstance(value, list) or not value:
        raise BlockerError(f"{context}.{key} must be a non-empty string list")
    if not all(isinstance(item, str) and item for item in value):
        raise BlockerError(f"{context}.{key} must contain only non-empty strings")
    return value


def validate_blocker(entry: Any, index: int) -> dict[str, Any]:
    context = f"blockers[{index}]"
    if not isinstance(entry, dict):
        raise BlockerError(f"{context} must be an object")

    require_str(entry, "id", context)
    require_str(entry, "phase", context)
    require_str(entry, "title", context)
    status = require_str(entry, "status", context)
    require_str_list(entry, "evidence", context)
    require_str_list(entry, "required_resolution", context)

    issue = entry.get("issue")
    if not isinstance(issue, int) or issue <= 0:
        raise BlockerError(f"{context}.issue must be a positive integer")
    if status not in OPEN_STATUSES | {RESOLVED_STATUS}:
        raise BlockerError(
            f"{context}.status must be one of {sorted(OPEN_STATUSES | {RESOLVED_STATUS})}"
        )

    return entry


def validate_evidence_paths(blockers: list[dict[str, Any]], path: Path) -> None:
    """Require each evidence entry to point at an existing repo-local file."""
    root = repo_root()
    for entry in blockers:
        for evidence in entry["evidence"]:
            evidence_path = Path(evidence)
            context = f"{entry['id']}.evidence {evidence!r}"
            if evidence_path.is_absolute():
                raise BlockerError(f"{path}: {context} must be repo-relative")
            candidate = (root / evidence_path).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise BlockerError(f"{path}: {context} must stay within the repo") from exc
            if not candidate.exists():
                raise BlockerError(f"{path}: {context} does not exist")


def validate_dependency_order(blockers: list[dict[str, Any]], path: Path) -> None:
    """Require resolved blockers to have resolved release prerequisites."""
    status_by_id = {entry["id"]: entry["status"] for entry in blockers}
    for entry in blockers:
        if entry["status"] != RESOLVED_STATUS:
            continue
        for dependency in RELEASE_DEPENDENCIES.get(entry["id"], ()):
            dependency_status = status_by_id.get(dependency)
            if dependency_status is None:
                raise BlockerError(f"{path}: {entry['id']} depends on missing blocker {dependency}")
            if dependency_status != RESOLVED_STATUS:
                raise BlockerError(
                    f"{path}: {entry['id']} cannot be resolved while "
                    f"{dependency} is {dependency_status}"
                )


def validate_required_evidence(blockers: list[dict[str, Any]], path: Path) -> None:
    """Require release-critical blockers to retain their current gate evidence."""
    evidence_by_id = {entry["id"]: set(entry["evidence"]) for entry in blockers}
    for blocker_id, required_paths in REQUIRED_EVIDENCE_BY_ID.items():
        evidence = evidence_by_id.get(blocker_id)
        if evidence is None:
            raise BlockerError(f"{path}: missing blocker {blocker_id}")
        missing = [required for required in required_paths if required not in evidence]
        if missing:
            raise BlockerError(
                f"{path}: {blocker_id} evidence missing required path(s): {', '.join(missing)}"
            )


def load_blockers(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BlockerError(f"missing release blocker file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BlockerError(f"{path}: invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise BlockerError(f"{path}: root must be a JSON object")
    if payload.get("schema_version") != "1.0.0":
        raise BlockerError(f"{path}: schema_version must be '1.0.0'")
    require_str(payload, "updated", "root")

    raw_blockers = payload.get("blockers")
    if not isinstance(raw_blockers, list):
        raise BlockerError(f"{path}: blockers must be a list")

    blockers = [validate_blocker(entry, index) for index, entry in enumerate(raw_blockers)]
    validate_backlog_contract(blockers, path)
    validate_dependency_order(blockers, path)
    validate_evidence_paths(blockers, path)
    validate_required_evidence(blockers, path)
    validate_required_resolutions(blockers, path)
    return blockers


def validate_backlog_contract(blockers: list[dict[str, Any]], path: Path) -> None:
    """Validate that the release gate tracks the complete F1-F13 backlog."""
    seen: dict[str, int] = {}
    for entry in blockers:
        blocker_id = entry["id"]
        if blocker_id in seen:
            raise BlockerError(f"{path}: duplicate blocker id {blocker_id!r}")
        seen[blocker_id] = entry["issue"]

    expected_ids = set(EXPECTED_BACKLOG_ISSUES)
    actual_ids = set(seen)
    missing = sorted(expected_ids - actual_ids, key=lambda item: int(item[1:]))
    unexpected = sorted(actual_ids - expected_ids)
    if missing:
        raise BlockerError(f"{path}: missing release backlog blocker(s): {', '.join(missing)}")
    if unexpected:
        raise BlockerError(f"{path}: unexpected release backlog blocker(s): {', '.join(unexpected)}")

    for blocker_id, expected_issue in EXPECTED_BACKLOG_ISSUES.items():
        issue = seen[blocker_id]
        if issue != expected_issue:
            raise BlockerError(
                f"{path}: {blocker_id} must map to issue #{expected_issue}, got #{issue}"
            )


def validate_required_resolutions(blockers: list[dict[str, Any]], path: Path) -> None:
    """Require high-risk blockers to retain specific resolution criteria."""
    resolution_by_id = {
        entry["id"]: "\n".join(entry["required_resolution"])
        for entry in blockers
    }
    for blocker_id, substrings in REQUIRED_RESOLUTION_SUBSTRINGS_BY_ID.items():
        resolution = resolution_by_id.get(blocker_id)
        if resolution is None:
            raise BlockerError(f"{path}: missing blocker {blocker_id}")
        missing = [substring for substring in substrings if substring not in resolution]
        if missing:
            raise BlockerError(
                f"{path}: {blocker_id} required_resolution missing required text: "
                f"{', '.join(missing)}"
            )


def main() -> int:
    args = parse_args()
    path = resolve_path(args.path)
    try:
        blockers = load_blockers(path)
    except BlockerError as exc:
        print(f"check_release_blockers.py: {exc}", file=sys.stderr)
        return 1

    open_blockers = [entry for entry in blockers if entry["status"] in OPEN_STATUSES]
    if open_blockers and not args.allow_open:
        print("release acceptance blocked:", file=sys.stderr)
        for entry in open_blockers:
            print(
                f"- {entry['id']} (#{entry['issue']}): {entry['title']} [{entry['status']}]",
                file=sys.stderr,
            )
        print(
            f"resolve these blockers or run with --allow-open for structure-only validation: {path}",
            file=sys.stderr,
        )
        return 1

    suffix = "open allowed" if args.allow_open else "all resolved"
    print(f"release blocker check ok: {len(blockers)} blocker(s), {len(open_blockers)} open ({suffix})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
