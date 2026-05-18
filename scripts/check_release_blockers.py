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
    validate_evidence_paths(blockers, path)
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
