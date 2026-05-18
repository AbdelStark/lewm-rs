#!/usr/bin/env python3
"""Validate the Phase A paid-job approval packet."""

from __future__ import annotations

import argparse
import json
import sys
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import launch_hf_job  # noqa: E402

DEFAULT_REPORT = Path("reports/phase_a_approval.json")
DEFAULT_LEASH = Path(".ml-intern/cli_agent_config.json")
EXPECTED_TASKS = {
    "F1": {
        "issue": 243,
        "job": "jobs/train_pusht.yaml",
        "dry_run_tokens": (
            "scripts/launch_hf_job.py",
            "jobs/train_pusht.yaml",
            "--dry-run",
            "--allow-approval-required",
        ),
        "approval_tokens": (
            "scripts/launch_hf_job.py",
            "jobs/train_pusht.yaml",
            "--allow-approval-required",
        ),
    },
    "F3": {
        "issue": 245,
        "job": "jobs/train_so100_warmstart.yaml",
        "env_prefix": "LEWM_PUSHT_WARMSTART_MPK=train/",
        "template_placeholder": "REPLACE_WITH_COMPATIBLE_BOUNDED_RUN",
        "dry_run_tokens": (
            "LEWM_PUSHT_WARMSTART_MPK=train/REPLACE_WITH_COMPATIBLE_BOUNDED_RUN/step_0050000.mpk",
            "scripts/launch_hf_job.py",
            "jobs/train_so100_warmstart.yaml",
            "--dry-run",
            "--allow-approval-required",
        ),
        "approval_tokens": (
            "LEWM_PUSHT_WARMSTART_MPK=train/REPLACE_WITH_COMPATIBLE_BOUNDED_RUN/step_0050000.mpk",
            "scripts/launch_hf_job.py",
            "jobs/train_so100_warmstart.yaml",
            "--allow-approval-required",
        ),
    },
}


class ApprovalError(RuntimeError):
    """Raised when the Phase A approval packet is malformed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"approval packet JSON ({DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--leash",
        type=Path,
        default=DEFAULT_LEASH,
        help=f"agent leash JSON ({DEFAULT_LEASH})",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ApprovalError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ApprovalError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ApprovalError(f"{path}: root must be an object")
    return payload


def require_str(payload: dict[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ApprovalError(f"{path}: {key} must be a non-empty string")
    return value


def require_bool(payload: dict[str, Any], key: str, path: Path) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ApprovalError(f"{path}: {key} must be a boolean")
    return value


def require_command(payload: dict[str, Any], key: str, path: Path) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ApprovalError(f"{path}: {key} must be a non-empty command list")
    if not all(isinstance(item, str) and item for item in value):
        raise ApprovalError(f"{path}: {key} must contain only non-empty strings")
    return value


def require_str_list(payload: dict[str, Any], key: str, path: Path) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ApprovalError(f"{path}: {key} must be a non-empty string list")
    if not all(isinstance(item, str) and item for item in value):
        raise ApprovalError(f"{path}: {key} must contain only non-empty strings")
    return value


def parse_money(value: Any, context: str) -> Decimal:
    if not isinstance(value, str):
        raise ApprovalError(f"{context} must be a USD string")
    try:
        return Decimal(value).quantize(Decimal("0.01"))
    except Exception as exc:
        raise ApprovalError(f"{context} must be a decimal USD string") from exc


def money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01")))


def validate_evidence(paths: list[str], context: str, report_path: Path) -> None:
    for evidence in paths:
        evidence_path = Path(evidence)
        if evidence_path.is_absolute():
            raise ApprovalError(f"{report_path}: {context}.evidence {evidence!r} must be relative")
        candidate = (ROOT / evidence_path).resolve()
        try:
            candidate.relative_to(ROOT)
        except ValueError as exc:
            raise ApprovalError(
                f"{report_path}: {context}.evidence {evidence!r} must stay in repo"
            ) from exc
        if not candidate.exists():
            raise ApprovalError(
                f"{report_path}: {context}.evidence {evidence!r} does not exist"
            )


def validate_command_tokens(
    command: list[str],
    required: tuple[str, ...],
    *,
    context: str,
    report_path: Path,
) -> None:
    for token in command:
        if "<" in token or ">" in token:
            raise ApprovalError(f"{report_path}: {context} contains shell-unsafe placeholder {token!r}")
    for token in required:
        if token not in command:
            raise ApprovalError(f"{report_path}: {context} missing {token!r}")
    if "--cost-cap-usd" in command and "0" in command:
        raise ApprovalError(f"{report_path}: {context} must not disable cost cap")


def validate_job_cost(task: dict[str, Any], expected: dict[str, Any], report_path: Path) -> Decimal:
    job_path = str(expected["job"])
    job = launch_hf_job.parse_job(ROOT / job_path)
    hardware = str(job["hardware"])
    timeout = str(job["timeout"])
    hours = launch_hf_job.parse_timeout_hours(timeout)
    price = launch_hf_job.lookup_price(hardware)
    if price is None:
        raise ApprovalError(f"{report_path}: {job_path} has unknown hardware {hardware!r}")
    worst_case = (price * hours).quantize(Decimal("0.01"), rounding=ROUND_CEILING)

    if task.get("hardware") != hardware:
        raise ApprovalError(f"{report_path}: {task['id']}.hardware must be {hardware!r}")
    if task.get("timeout") != timeout:
        raise ApprovalError(f"{report_path}: {task['id']}.timeout must be {timeout!r}")
    if parse_money(task.get("price_usd_per_hour"), f"{task['id']}.price_usd_per_hour") != price:
        raise ApprovalError(f"{report_path}: {task['id']}.price_usd_per_hour must be {money(price)}")
    if parse_money(task.get("worst_case_usd"), f"{task['id']}.worst_case_usd") != worst_case:
        raise ApprovalError(f"{report_path}: {task['id']}.worst_case_usd must be {money(worst_case)}")
    return worst_case


def validate_task(
    task: Any,
    leash: dict[str, Any],
    report_path: Path,
) -> Decimal:
    if not isinstance(task, dict):
        raise ApprovalError(f"{report_path}: tasks entries must be objects")
    task_id = require_str(task, "id", report_path)
    expected = EXPECTED_TASKS.get(task_id)
    if expected is None:
        raise ApprovalError(f"{report_path}: unexpected Phase A task {task_id!r}")
    if task.get("issue") != expected["issue"]:
        raise ApprovalError(f"{report_path}: {task_id}.issue must be #{expected['issue']}")
    if task.get("job") != expected["job"]:
        raise ApprovalError(f"{report_path}: {task_id}.job must be {expected['job']!r}")
    if not require_bool(task, "requires_human_approval", report_path):
        raise ApprovalError(f"{report_path}: {task_id} must require human approval")

    approval_required = set(leash.get("jobs_human_approval_required", []))
    allowed = set(leash.get("jobs_allowed", []))
    job_name = Path(str(expected["job"])).name
    if job_name not in approval_required:
        raise ApprovalError(f"{report_path}: {job_name} must be approval-required in leash")
    if job_name in allowed:
        raise ApprovalError(f"{report_path}: {job_name} must not be pre-approved in leash")

    dry_run = require_command(task, "dry_run_command", report_path)
    approval = require_command(task, "approval_command", report_path)
    validate_command_tokens(
        dry_run,
        expected["dry_run_tokens"],
        context=f"{task_id}.dry_run_command",
        report_path=report_path,
    )
    validate_command_tokens(
        approval,
        expected["approval_tokens"],
        context=f"{task_id}.approval_command",
        report_path=report_path,
    )
    if "--dry-run" in approval:
        raise ApprovalError(f"{report_path}: {task_id}.approval_command must not dry-run")
    require_str(task, "title", report_path)
    require_str_list(task, "blocked_on", report_path)
    validate_template_declaration(task, expected, report_path)
    validate_evidence(require_str_list(task, "evidence", report_path), task_id, report_path)
    return validate_job_cost(task, expected, report_path)


def validate_template_declaration(
    task: dict[str, Any],
    expected: dict[str, Any],
    report_path: Path,
) -> None:
    placeholder = expected.get("template_placeholder")
    if not isinstance(placeholder, str):
        return

    raw_placeholders = task.get("template_placeholders")
    if not isinstance(raw_placeholders, list) or not raw_placeholders:
        raise ApprovalError(
            f"{report_path}: {task['id']}.template_placeholders must be a non-empty string list"
        )
    if not all(isinstance(item, str) and item for item in raw_placeholders):
        raise ApprovalError(
            f"{report_path}: {task['id']}.template_placeholders must contain only strings"
        )
    placeholders = raw_placeholders
    if placeholder not in placeholders:
        raise ApprovalError(
            f"{report_path}: {task['id']}.template_placeholders must include {placeholder!r}"
        )
    resolution = require_str(task, "template_resolution", report_path)
    if placeholder not in resolution:
        raise ApprovalError(
            f"{report_path}: {task['id']}.template_resolution must mention {placeholder!r}"
        )
    if "replace" not in resolution.lower():
        raise ApprovalError(
            f"{report_path}: {task['id']}.template_resolution must say the placeholder is replaced"
        )


def validate_approval(payload: dict[str, Any], leash: dict[str, Any], report_path: Path) -> None:
    if payload.get("schema_version") != "1.0.0":
        raise ApprovalError(f"{report_path}: schema_version must be '1.0.0'")
    require_str(payload, "updated", report_path)
    if payload.get("phase") != "A":
        raise ApprovalError(f"{report_path}: phase must be 'A'")
    if payload.get("status") != "blocked_pending_human_approval":
        raise ApprovalError(f"{report_path}: status must be blocked_pending_human_approval")

    session_cap = parse_money(payload.get("session_cap_usd"), "session_cap_usd")
    leash_cap = Decimal(str(leash.get("billing", {}).get("session_cap_usd"))).quantize(Decimal("0.01"))
    if session_cap != leash_cap:
        raise ApprovalError(f"{report_path}: session_cap_usd must match leash cap {money(leash_cap)}")

    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(EXPECTED_TASKS):
        raise ApprovalError(f"{report_path}: tasks must list F1 and F3")
    seen: set[str] = set()
    costs: list[Decimal] = []
    for task in tasks:
        if isinstance(task, dict):
            seen.add(str(task.get("id")))
        costs.append(validate_task(task, leash, report_path))
    if seen != set(EXPECTED_TASKS):
        raise ApprovalError(f"{report_path}: tasks must be exactly F1 and F3")

    total = sum(costs, Decimal("0.00")).quantize(Decimal("0.01"))
    if parse_money(payload.get("total_worst_case_usd"), "total_worst_case_usd") != total:
        raise ApprovalError(f"{report_path}: total_worst_case_usd must be {money(total)}")
    separate = require_bool(payload, "combined_requires_separate_approval_sessions", report_path)
    if total > session_cap and not separate:
        raise ApprovalError(
            f"{report_path}: combined worst-case ${money(total)} exceeds session cap "
            f"${money(session_cap)} and must require separate approval sessions"
        )
    require_str(payload, "summary", report_path)


def main() -> int:
    args = parse_args()
    report_path = resolve_path(args.path)
    leash_path = resolve_path(args.leash)
    try:
        payload = load_json(report_path, "Phase A approval packet")
        leash = load_json(leash_path, "agent leash")
        validate_approval(payload, leash, report_path)
    except ApprovalError as exc:
        print(f"check_phase_a_approval.py: {exc}", file=sys.stderr)
        return 1

    print("Phase A approval packet ok: F1=$18.00 F3=$9.00 combined=$27.00")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
