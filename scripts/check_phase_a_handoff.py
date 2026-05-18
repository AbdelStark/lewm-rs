#!/usr/bin/env python3
"""Validate the Phase A release handoff contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_HANDOFF = Path("reports/phase_a_handoff.json")
DEFAULT_BLOCKERS = Path("conformance/release_blockers.json")
EXPECTED_TASKS = (
    {
        "id": "F1",
        "issue": 243,
        "source_prefix": "train/pusht-full-burn-jepa-",
        "rejected_source_prefix": "train/pusht-full-lewm-",
        "required_tokens": (
            "jobs/train_pusht.yaml",
            "--allow-approval-required",
            "scripts/f1_export_pusht_onnx.py",
            "--run-prefix",
            "train/pusht-full-burn-jepa-REPLACE_WITH_UTC_TIMESTAMP",
            "--execute",
            "--upload",
        ),
    },
    {
        "id": "F3",
        "issue": 245,
        "source_env": "LEWM_PUSHT_WARMSTART_MPK",
        "source_verifier": "scripts/check_warmstart_source.py",
        "required_tokens": (
            "LEWM_PUSHT_WARMSTART_MPK=train/REPLACE_WITH_COMPATIBLE_BOUNDED_RUN/step_0050000.mpk",
            "jobs/train_so100_warmstart.yaml",
            "--allow-approval-required",
            "scripts/check_pusht_warmstart_source_smoke_report.py",
            "scripts/check_pusht_warmstart_hub_audit_report.py",
        ),
    },
)


class HandoffError(RuntimeError):
    """Raised when the Phase A handoff file is malformed."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_HANDOFF,
        help=f"Phase A handoff file ({DEFAULT_HANDOFF})",
    )
    parser.add_argument(
        "--blockers",
        type=Path,
        default=DEFAULT_BLOCKERS,
        help=f"release blocker file used for cross-checking ({DEFAULT_BLOCKERS})",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def require_str(payload: dict[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise HandoffError(f"{path}: {key} must be a non-empty string")
    return value


def require_bool(payload: dict[str, Any], key: str, path: Path) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise HandoffError(f"{path}: {key} must be a boolean")
    return value


def require_str_list(payload: dict[str, Any], key: str, path: Path) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise HandoffError(f"{path}: {key} must be a non-empty string list")
    if not all(isinstance(item, str) and item for item in value):
        raise HandoffError(f"{path}: {key} must contain only non-empty strings")
    return value


def require_commands(payload: dict[str, Any], path: Path) -> dict[str, list[list[str]]]:
    value = payload.get("commands")
    if not isinstance(value, dict) or not value:
        raise HandoffError(f"{path}: commands must be a non-empty object")
    commands: dict[str, list[list[str]]] = {}
    for name, raw_group in value.items():
        if not isinstance(name, str) or not name:
            raise HandoffError(f"{path}: command group names must be non-empty strings")
        if not isinstance(raw_group, list) or not raw_group:
            raise HandoffError(f"{path}: commands.{name} must be a non-empty command list")
        group: list[list[str]] = []
        for index, raw_command in enumerate(raw_group):
            if not isinstance(raw_command, list) or not raw_command:
                raise HandoffError(f"{path}: commands.{name}[{index}] must be a non-empty list")
            if not all(isinstance(item, str) and item for item in raw_command):
                raise HandoffError(f"{path}: commands.{name}[{index}] must contain strings")
            group.append(raw_command)
        commands[name] = group
    return commands


def flatten_commands(commands: dict[str, list[list[str]]]) -> list[str]:
    return [token for group in commands.values() for command in group for token in command]


def require_command_group(
    commands: dict[str, list[list[str]]],
    name: str,
    task_id: str,
    path: Path,
) -> list[list[str]]:
    group = commands.get(name)
    if group is None:
        raise HandoffError(f"{path}: {task_id}.commands missing {name!r}")
    return group


def command_has(command: list[str], *tokens: str) -> bool:
    return all(token in command for token in tokens)


def require_no_token(
    commands: list[list[str]],
    token: str,
    context: str,
    path: Path,
) -> None:
    for command in commands:
        if token in command:
            raise HandoffError(f"{path}: {context} must not contain {token!r}")


def require_any_command(
    commands: list[list[str]],
    context: str,
    path: Path,
    *tokens: str,
) -> None:
    if not any(command_has(command, *tokens) for command in commands):
        joined = ", ".join(repr(token) for token in tokens)
        raise HandoffError(f"{path}: {context} must include a command with {joined}")


def validate_f1_command_stages(commands: dict[str, list[list[str]]], path: Path) -> None:
    preflight = require_command_group(commands, "preflight", "F1", path)
    require_no_token(preflight, "--execute", "F1.preflight", path)
    require_no_token(preflight, "--upload", "F1.preflight", path)
    require_any_command(preflight, "F1.preflight", path, "scripts/check_full_pusht_contract_smoke_report.py")
    require_any_command(
        preflight,
        "F1.preflight",
        path,
        "scripts/launch_hf_job.py",
        "jobs/train_pusht.yaml",
        "--dry-run",
        "--allow-approval-required",
    )

    approval = require_command_group(commands, "after_human_approval", "F1", path)
    require_no_token(approval, "--dry-run", "F1.after_human_approval", path)
    require_any_command(
        approval,
        "F1.after_human_approval",
        path,
        "scripts/launch_hf_job.py",
        "jobs/train_pusht.yaml",
        "--allow-approval-required",
    )

    export = require_command_group(commands, "after_full_checkpoint_exists", "F1", path)
    if len(export) != 3:
        raise HandoffError(f"{path}: F1.after_full_checkpoint_exists must have 3 commands")
    for index, command in enumerate(export):
        if not command_has(
            command,
            "scripts/f1_export_pusht_onnx.py",
            "--run-prefix",
            "train/pusht-full-burn-jepa-REPLACE_WITH_UTC_TIMESTAMP",
        ):
            raise HandoffError(f"{path}: F1.after_full_checkpoint_exists[{index}] is malformed")
    if "--execute" in export[0] or "--upload" in export[0]:
        raise HandoffError(f"{path}: F1 export dry-run command must not execute or upload")
    if "--execute" not in export[1] or "--upload" in export[1]:
        raise HandoffError(f"{path}: F1 export execute command must execute without upload")
    if "--execute" not in export[2] or "--upload" not in export[2]:
        raise HandoffError(f"{path}: F1 final upload command must include --execute and --upload")


def validate_f3_command_stages(commands: dict[str, list[list[str]]], path: Path) -> None:
    preflight = require_command_group(commands, "preflight", "F3", path)
    require_any_command(
        preflight,
        "F3.preflight",
        path,
        "scripts/check_pusht_warmstart_source_smoke_report.py",
    )
    require_any_command(
        preflight,
        "F3.preflight",
        path,
        "scripts/check_pusht_warmstart_hub_audit_report.py",
    )
    require_any_command(
        preflight,
        "F3.preflight",
        path,
        "LEWM_PUSHT_WARMSTART_MPK=train/REPLACE_WITH_COMPATIBLE_BOUNDED_RUN/step_0050000.mpk",
        "scripts/launch_hf_job.py",
        "jobs/train_so100_warmstart.yaml",
        "--dry-run",
        "--allow-approval-required",
    )

    approval = require_command_group(commands, "after_human_approval", "F3", path)
    require_no_token(approval, "--dry-run", "F3.after_human_approval", path)
    require_any_command(
        approval,
        "F3.after_human_approval",
        path,
        "LEWM_PUSHT_WARMSTART_MPK=train/REPLACE_WITH_COMPATIBLE_BOUNDED_RUN/step_0050000.mpk",
        "scripts/launch_hf_job.py",
        "jobs/train_so100_warmstart.yaml",
        "--allow-approval-required",
    )


def validate_command_stages(
    task_id: str,
    commands: dict[str, list[list[str]]],
    path: Path,
) -> None:
    if task_id == "F1":
        validate_f1_command_stages(commands, path)
    elif task_id == "F3":
        validate_f3_command_stages(commands, path)


def validate_evidence_paths(paths: list[str], context: str, handoff_path: Path) -> None:
    root = repo_root()
    for evidence in paths:
        evidence_path = Path(evidence)
        if evidence_path.is_absolute():
            raise HandoffError(f"{handoff_path}: {context} evidence {evidence!r} must be relative")
        candidate = (root / evidence_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise HandoffError(
                f"{handoff_path}: {context} evidence {evidence!r} must stay in repo"
            ) from exc
        if not candidate.exists():
            raise HandoffError(
                f"{handoff_path}: {context} evidence {evidence!r} does not exist"
            )


def validate_task(task: Any, expected: dict[str, Any], handoff_path: Path) -> None:
    if not isinstance(task, dict):
        raise HandoffError(f"{handoff_path}: task must be an object")

    task_id = require_str(task, "id", handoff_path)
    if task_id != expected["id"]:
        raise HandoffError(f"{handoff_path}: expected task {expected['id']}, got {task_id}")
    issue = task.get("issue")
    if issue != expected["issue"]:
        raise HandoffError(f"{handoff_path}: {task_id}.issue must be #{expected['issue']}")
    if require_str(task, "status", handoff_path) != "blocked":
        raise HandoffError(f"{handoff_path}: {task_id}.status must stay blocked")
    if not require_bool(task, "requires_human_approval", handoff_path):
        raise HandoffError(f"{handoff_path}: {task_id} must require human approval")

    validate_evidence_paths(require_str_list(task, "evidence", handoff_path), task_id, handoff_path)
    require_str_list(task, "blocked_on", handoff_path)
    require_str_list(task, "acceptance", handoff_path)
    commands = require_commands(task, handoff_path)
    validate_command_stages(task_id, commands, handoff_path)
    tokens = flatten_commands(commands)

    for token in expected["required_tokens"]:
        if token not in tokens:
            raise HandoffError(f"{handoff_path}: {task_id} commands missing {token!r}")
    if "source_prefix" in expected:
        if task.get("source_prefix") != expected["source_prefix"]:
            raise HandoffError(
                f"{handoff_path}: {task_id}.source_prefix must be {expected['source_prefix']!r}"
            )
        rejected = require_str_list(task, "rejected_source_prefixes", handoff_path)
        if expected["rejected_source_prefix"] not in rejected:
            raise HandoffError(
                f"{handoff_path}: {task_id} must reject {expected['rejected_source_prefix']!r}"
            )
    if "source_env" in expected:
        if task.get("source_env") != expected["source_env"]:
            raise HandoffError(
                f"{handoff_path}: {task_id}.source_env must be {expected['source_env']!r}"
            )
        if task.get("source_verifier") != expected["source_verifier"]:
            raise HandoffError(
                f"{handoff_path}: {task_id}.source_verifier must be "
                f"{expected['source_verifier']!r}"
            )


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HandoffError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HandoffError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HandoffError(f"{path}: root must be an object")
    return payload


def repo_relative(path: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(repo_root()))
    except ValueError:
        return None


def validate_blocker_alignment(
    tasks: list[Any],
    blockers_payload: dict[str, Any],
    handoff_path: Path,
    blockers_path: Path,
) -> None:
    blockers = blockers_payload.get("blockers")
    if not isinstance(blockers, list):
        raise HandoffError(f"{blockers_path}: blockers must be a list")
    by_id = {
        entry.get("id"): entry
        for entry in blockers
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    handoff_evidence = repo_relative(handoff_path)
    for task in tasks:
        if not isinstance(task, dict):
            raise HandoffError(f"{handoff_path}: task must be an object")
        task_id = task.get("id")
        blocker = by_id.get(task_id)
        if not isinstance(blocker, dict):
            raise HandoffError(f"{blockers_path}: missing blocker {task_id}")
        if blocker.get("issue") != task.get("issue"):
            raise HandoffError(f"{blockers_path}: {task_id}.issue does not match handoff")
        if blocker.get("phase") != "A":
            raise HandoffError(f"{blockers_path}: {task_id}.phase must be 'A'")
        if blocker.get("status") != "blocked":
            raise HandoffError(f"{blockers_path}: {task_id}.status must stay blocked")
        evidence = blocker.get("evidence")
        if not isinstance(evidence, list):
            raise HandoffError(f"{blockers_path}: {task_id}.evidence must be a list")
        if handoff_evidence is not None and handoff_evidence not in evidence:
            raise HandoffError(
                f"{blockers_path}: {task_id}.evidence must include {handoff_evidence!r}"
            )


def validate_handoff(
    payload: dict[str, Any],
    *,
    path: Path,
    blockers_payload: dict[str, Any],
    blockers_path: Path,
) -> None:
    if payload.get("schema_version") != "1.0.0":
        raise HandoffError(f"{path}: schema_version must be '1.0.0'")
    require_str(payload, "updated", path)
    if payload.get("phase") != "A":
        raise HandoffError(f"{path}: phase must be 'A'")
    if payload.get("status") != "blocked":
        raise HandoffError(f"{path}: status must be 'blocked'")
    require_str(payload, "summary", path)

    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise HandoffError(f"{path}: tasks must be a list")
    if len(tasks) != len(EXPECTED_TASKS):
        raise HandoffError(f"{path}: expected {len(EXPECTED_TASKS)} Phase A task(s)")
    for task, expected in zip(tasks, EXPECTED_TASKS, strict=True):
        validate_task(task, expected, path)
    validate_blocker_alignment(tasks, blockers_payload, path, blockers_path)


def main() -> int:
    args = parse_args()
    path = resolve_path(args.path)
    blockers_path = resolve_path(args.blockers)
    try:
        payload = load_json(path, label="Phase A handoff file")
        blockers_payload = load_json(blockers_path, label="release blocker file")
        validate_handoff(
            payload,
            path=path,
            blockers_payload=blockers_payload,
            blockers_path=blockers_path,
        )
    except HandoffError as exc:
        print(f"check_phase_a_handoff.py: {exc}", file=sys.stderr)
        return 1

    print("Phase A handoff ok: F1 and F3 remain blocked behind explicit human gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
