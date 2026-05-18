from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_phase_a_approval.py"


def approval_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "updated": "2026-05-18",
        "phase": "A",
        "status": "blocked_pending_human_approval",
        "session_cap_usd": "20.00",
        "total_worst_case_usd": "27.00",
        "combined_requires_separate_approval_sessions": True,
        "summary": "approval required",
        "tasks": [
            {
                "id": "F1",
                "issue": 243,
                "title": "Export trained full PushT ONNX artifacts",
                "job": "jobs/train_pusht.yaml",
                "hardware": "a10g-large",
                "timeout": "12h",
                "price_usd_per_hour": "1.50",
                "worst_case_usd": "18.00",
                "requires_human_approval": True,
                "dry_run_command": [
                    "python3",
                    "scripts/launch_hf_job.py",
                    "jobs/train_pusht.yaml",
                    "--dry-run",
                    "--allow-approval-required",
                ],
                "approval_command": [
                    "scripts/launch_hf_job.py",
                    "jobs/train_pusht.yaml",
                    "--allow-approval-required",
                ],
                "blocked_on": ["approval"],
                "evidence": [
                    "jobs/train_pusht.yaml",
                    ".ml-intern/cli_agent_config.json",
                    "reports/full_pusht_contract_smoke.json",
                    "reports/full_burn_jepa_training_gap.md",
                    "scripts/f1_export_pusht_onnx.py",
                ],
            },
            {
                "id": "F3",
                "issue": 245,
                "title": "Launch SO-100 warm-start ablation",
                "job": "jobs/train_so100_warmstart.yaml",
                "hardware": "a10g-large",
                "timeout": "6h",
                "price_usd_per_hour": "1.50",
                "worst_case_usd": "9.00",
                "requires_human_approval": True,
                "template_placeholders": ["REPLACE_WITH_COMPATIBLE_BOUNDED_RUN"],
                "template_resolution": (
                    "Replace REPLACE_WITH_COMPATIBLE_BOUNDED_RUN with a compatible Hub source."
                ),
                "dry_run_command": [
                    "LEWM_PUSHT_WARMSTART_MPK=train/REPLACE_WITH_COMPATIBLE_BOUNDED_RUN/step_0050000.mpk",
                    "python3",
                    "scripts/launch_hf_job.py",
                    "jobs/train_so100_warmstart.yaml",
                    "--dry-run",
                    "--allow-approval-required",
                ],
                "approval_command": [
                    "LEWM_PUSHT_WARMSTART_MPK=train/REPLACE_WITH_COMPATIBLE_BOUNDED_RUN/step_0050000.mpk",
                    "scripts/launch_hf_job.py",
                    "jobs/train_so100_warmstart.yaml",
                    "--allow-approval-required",
                ],
                "blocked_on": ["approval"],
                "evidence": [
                    "jobs/train_so100_warmstart.yaml",
                    ".ml-intern/cli_agent_config.json",
                    "reports/pusht_warmstart_source_smoke.json",
                    "reports/pusht_warmstart_hub_audit.json",
                    "scripts/check_warmstart_source.py",
                ],
            },
        ],
    }
    payload.update(updates)
    return payload


def run_check(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--path", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_valid_phase_a_approval_packet_passes(tmp_path: Path) -> None:
    path = tmp_path / "phase_a_approval.json"
    path.write_text(json.dumps(approval_payload()), encoding="utf-8")

    result = run_check(path)

    assert result.returncode == 0
    assert "Phase A approval packet ok" in result.stdout


def test_rejects_wrong_total_cost(tmp_path: Path) -> None:
    path = tmp_path / "phase_a_approval.json"
    path.write_text(json.dumps(approval_payload(total_worst_case_usd="20.00")), encoding="utf-8")

    result = run_check(path)

    assert result.returncode == 1
    assert "total_worst_case_usd must be 27.00" in result.stderr


def test_rejects_combined_without_separate_sessions(tmp_path: Path) -> None:
    path = tmp_path / "phase_a_approval.json"
    path.write_text(
        json.dumps(approval_payload(combined_requires_separate_approval_sessions=False)),
        encoding="utf-8",
    )

    result = run_check(path)

    assert result.returncode == 1
    assert "must require separate approval sessions" in result.stderr


def test_rejects_shell_unsafe_placeholder(tmp_path: Path) -> None:
    path = tmp_path / "phase_a_approval.json"
    payload = approval_payload()
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    f3 = tasks[1]
    assert isinstance(f3, dict)
    command = f3["dry_run_command"]
    assert isinstance(command, list)
    command[0] = "LEWM_PUSHT_WARMSTART_MPK=train/<compatible>/step_0050000.mpk"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(path)

    assert result.returncode == 1
    assert "shell-unsafe placeholder" in result.stderr


def test_rejects_f3_without_template_resolution(tmp_path: Path) -> None:
    path = tmp_path / "phase_a_approval.json"
    payload = approval_payload()
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    f3 = tasks[1]
    assert isinstance(f3, dict)
    f3["template_placeholders"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(path)

    assert result.returncode == 1
    assert "F3.template_placeholders must be a non-empty string list" in result.stderr


def test_rejects_approval_dry_run(tmp_path: Path) -> None:
    path = tmp_path / "phase_a_approval.json"
    payload = approval_payload()
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    f1 = tasks[0]
    assert isinstance(f1, dict)
    command = f1["approval_command"]
    assert isinstance(command, list)
    command.append("--dry-run")
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(path)

    assert result.returncode == 1
    assert "approval_command must not dry-run" in result.stderr
