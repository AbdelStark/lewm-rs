#!/usr/bin/env python3
"""Validate the F1 no-GHCR source-build dry-run evidence report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

import launch_hf_job

DEFAULT_REPORT = Path("reports/f1_source_build_dry_run.json")
EXPECTED_TASK = "F1"
EXPECTED_ISSUE = 243
EXPECTED_JOB = "jobs/train_pusht_source.yaml"
EXPECTED_STATUS = "preflight_passed_pending_human_approval"
EXPECTED_COST_CAP_USD = Decimal("20.00")
EXPECTED_RENDERED_TOKENS = (
    "hf jobs run",
    "--namespace abdelstark",
    "--flavor a10g-large",
    "--timeout 12h",
    "rust:1.95.0-bookworm",
    "TRACKIO_RUN=pusht-full-burn-jepa-source",
    'experimental.pusht_train_mode="full_burn_jepa"',
    "python/export_onnx.py",
    "--check-contract-only",
    "python/upload_checkpoints.py",
    "--path-prefix train/pusht-full-burn-jepa-",
)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class F1SourceBuildDryRunReportError(RuntimeError):
    """Raised when the F1 source-build dry-run report is malformed."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"dry-run report path ({DEFAULT_REPORT})",
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
        raise F1SourceBuildDryRunReportError(
            f"missing F1 source-build dry-run report: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise F1SourceBuildDryRunReportError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise F1SourceBuildDryRunReportError(f"{path}: report root must be an object")
    return payload


def require_str(payload: dict[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise F1SourceBuildDryRunReportError(f"{path}: {key} must be a non-empty string")
    return value


def require_bool(payload: dict[str, Any], key: str, path: Path) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise F1SourceBuildDryRunReportError(f"{path}: {key} must be a boolean")
    return value


def require_command(payload: dict[str, Any], key: str, path: Path) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise F1SourceBuildDryRunReportError(f"{path}: {key} must be a non-empty list")
    if not all(isinstance(item, str) and item for item in value):
        raise F1SourceBuildDryRunReportError(f"{path}: {key} must contain strings")
    return value


def require_str_list(payload: dict[str, Any], key: str, path: Path) -> list[str]:
    return require_command(payload, key, path)


def parse_money(raw: Any, key: str, path: Path) -> Decimal:
    if not isinstance(raw, str):
        raise F1SourceBuildDryRunReportError(f"{path}: {key} must be a USD string")
    try:
        return Decimal(raw).quantize(Decimal("0.01"))
    except Exception as exc:
        raise F1SourceBuildDryRunReportError(f"{path}: {key} must be decimal USD") from exc


def require_literal(payload: dict[str, Any], key: str, expected: object, path: Path) -> None:
    value = payload.get(key)
    if value != expected:
        raise F1SourceBuildDryRunReportError(f"{path}: {key} must be {expected!r}")


def validate_command(
    command: list[str],
    *,
    revision: str,
    dry_run: bool,
    path: Path,
    key: str,
) -> None:
    source_env = f"LEWM_SOURCE_REVISION={revision}"
    expected_tokens = (
        source_env,
        "scripts/launch_hf_job.py",
        EXPECTED_JOB,
        "--allow-approval-required",
    )
    for token in expected_tokens:
        if token not in command:
            raise F1SourceBuildDryRunReportError(f"{path}: {key} missing {token!r}")
    has_dry_run = "--dry-run" in command
    if dry_run and not has_dry_run:
        raise F1SourceBuildDryRunReportError(f"{path}: {key} must include --dry-run")
    if not dry_run and has_dry_run:
        raise F1SourceBuildDryRunReportError(f"{path}: {key} must not include --dry-run")
    for token in command:
        if "REPLACE_WITH_" in token or "<" in token or ">" in token:
            raise F1SourceBuildDryRunReportError(f"{path}: {key} contains placeholder {token!r}")


def validate_job_cost(payload: dict[str, Any], path: Path) -> None:
    job = launch_hf_job.parse_job(repo_root() / EXPECTED_JOB)
    hardware = str(job["hardware"])
    timeout = str(job["timeout"])
    price = launch_hf_job.lookup_price(hardware)
    if price is None:
        raise F1SourceBuildDryRunReportError(f"{path}: unknown hardware {hardware!r}")
    hours = launch_hf_job.parse_timeout_hours(timeout)
    worst_case = (price * hours).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    if worst_case != parse_money(payload.get("worst_case_usd"), "worst_case_usd", path):
        raise F1SourceBuildDryRunReportError(
            f"{path}: worst_case_usd must match {EXPECTED_JOB} cost {worst_case}"
        )
    cost_cap = parse_money(payload.get("cost_cap_usd"), "cost_cap_usd", path)
    if cost_cap != EXPECTED_COST_CAP_USD:
        raise F1SourceBuildDryRunReportError(f"{path}: cost_cap_usd must be 20.00")
    if worst_case > cost_cap:
        raise F1SourceBuildDryRunReportError(
            f"{path}: worst_case_usd {worst_case} exceeds cost_cap_usd {cost_cap}"
        )


def validate_rendered_checks(payload: dict[str, Any], revision: str, path: Path) -> None:
    rendered = require_str_list(payload, "rendered_command_checks", path)
    required = (*EXPECTED_RENDERED_TOKENS, f"--env LEWM_SOURCE_REVISION={revision}")
    for token in required:
        if token not in rendered:
            raise F1SourceBuildDryRunReportError(
                f"{path}: rendered_command_checks missing {token!r}"
            )
    for token in rendered:
        if "REPLACE_WITH_" in token:
            raise F1SourceBuildDryRunReportError(
                f"{path}: rendered_command_checks contains placeholder {token!r}"
            )


def validate_blockers(payload: dict[str, Any], path: Path) -> None:
    blocked_on = require_str_list(payload, "blocked_on", path)
    joined = " ".join(blocked_on).lower()
    for token in ("human approval", "train/pusht-full-burn-jepa-", "onnx-full/"):
        if token not in joined:
            raise F1SourceBuildDryRunReportError(f"{path}: blocked_on missing {token!r}")


def validate_report(payload: dict[str, Any], path: Path) -> None:
    require_literal(payload, "schema_version", "1.0.0", path)
    require_str(payload, "updated", path)
    require_literal(payload, "task", EXPECTED_TASK, path)
    require_literal(payload, "issue", EXPECTED_ISSUE, path)
    require_literal(payload, "status", EXPECTED_STATUS, path)
    require_literal(payload, "job", EXPECTED_JOB, path)
    require_literal(payload, "result", "passed", path)
    if require_bool(payload, "launched_paid_job", path):
        raise F1SourceBuildDryRunReportError(f"{path}: launched_paid_job must be false")
    if require_bool(payload, "uploaded_artifacts", path):
        raise F1SourceBuildDryRunReportError(f"{path}: uploaded_artifacts must be false")

    revision = require_str(payload, "source_revision", path)
    if SHA_RE.fullmatch(revision) is None:
        raise F1SourceBuildDryRunReportError(
            f"{path}: source_revision must be a full lowercase git SHA"
        )
    if "REPLACE_WITH_" in require_str(payload, "source_revision_note", path):
        raise F1SourceBuildDryRunReportError(f"{path}: source_revision_note has placeholder text")

    validate_command(
        require_command(payload, "dry_run_command", path),
        revision=revision,
        dry_run=True,
        path=path,
        key="dry_run_command",
    )
    validate_command(
        require_command(payload, "approval_command", path),
        revision=revision,
        dry_run=False,
        path=path,
        key="approval_command",
    )
    validate_job_cost(payload, path)
    validate_rendered_checks(payload, revision, path)
    validate_blockers(payload, path)


def main() -> int:
    path = resolve_path(parse_args().path)
    try:
        payload = load_report(path)
        validate_report(payload, path)
    except (F1SourceBuildDryRunReportError, launch_hf_job.LaunchError) as exc:
        print(f"check_f1_source_build_dry_run_report.py: {exc}", file=sys.stderr)
        return 1

    print("F1 source-build dry-run report ok: job=jobs/train_pusht_source.yaml cost=$18.00")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
