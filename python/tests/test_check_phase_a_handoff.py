from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_phase_a_handoff.py"


def handoff_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "updated": "2026-05-18",
        "phase": "A",
        "status": "blocked",
        "summary": "Phase A requires human-approved jobs.",
        "tasks": [
            {
                "id": "F1",
                "issue": 243,
                "title": "Export trained full PushT ONNX artifacts",
                "status": "blocked",
                "requires_human_approval": True,
                "source_prefix": "train/pusht-full-burn-jepa-",
                "rejected_source_prefixes": ["train/pusht-full-lewm-"],
                "evidence": [
                    "reports/pusht_onnx_export.md",
                    "reports/full_burn_jepa_training_gap.md",
                    "reports/full_pusht_contract_smoke.json",
                    "scripts/f1_export_pusht_onnx.py",
                    "jobs/train_pusht.yaml",
                    ".ml-intern/cli_agent_config.json",
                ],
                "blocked_on": ["human-approved full PushT job"],
                "commands": {
                    "preflight": [
                        ["python3", "scripts/check_full_pusht_contract_smoke_report.py"],
                        [
                            "python3",
                            "scripts/launch_hf_job.py",
                            "jobs/train_pusht.yaml",
                            "--dry-run",
                            "--allow-approval-required",
                        ],
                    ],
                    "after_human_approval": [
                        [
                            "scripts/launch_hf_job.py",
                            "jobs/train_pusht.yaml",
                            "--allow-approval-required",
                        ]
                    ],
                    "after_full_checkpoint_exists": [
                        [
                            "scripts/f1_export_pusht_onnx.py",
                            "--run-prefix",
                            "train/pusht-full-burn-jepa-REPLACE_WITH_UTC_TIMESTAMP",
                        ],
                        [
                            "scripts/f1_export_pusht_onnx.py",
                            "--run-prefix",
                            "train/pusht-full-burn-jepa-REPLACE_WITH_UTC_TIMESTAMP",
                            "--execute",
                        ],
                        [
                            "scripts/f1_export_pusht_onnx.py",
                            "--run-prefix",
                            "train/pusht-full-burn-jepa-REPLACE_WITH_UTC_TIMESTAMP",
                            "--execute",
                            "--upload",
                        ],
                    ],
                },
                "acceptance": ["verified onnx-full upload"],
            },
            {
                "id": "F3",
                "issue": 245,
                "title": "Launch SO-100 warm-start ablation",
                "status": "blocked",
                "requires_human_approval": True,
                "source_env": "LEWM_PUSHT_WARMSTART_MPK",
                "source_verifier": "scripts/check_warmstart_source.py",
                "evidence": [
                    "reports/so100_warmstart.md",
                    "jobs/train_so100_warmstart.yaml",
                    ".ml-intern/cli_agent_config.json",
                    "reports/pusht_warmstart_source_smoke.json",
                    "reports/pusht_warmstart_hub_audit.json",
                    "scripts/pusht_warmstart_source_smoke.py",
                    "scripts/check_pusht_warmstart_source_smoke_report.py",
                    "scripts/audit_pusht_warmstart_sources.py",
                    "scripts/check_pusht_warmstart_hub_audit_report.py",
                    "scripts/check_warmstart_source.py",
                ],
                "blocked_on": ["compatible PushT source and human approval"],
                "commands": {
                    "preflight": [
                        ["python3", "scripts/check_pusht_warmstart_source_smoke_report.py"],
                        ["python3", "scripts/check_pusht_warmstart_hub_audit_report.py"],
                        [
                            "LEWM_PUSHT_WARMSTART_MPK=train/REPLACE_WITH_COMPATIBLE_BOUNDED_RUN/step_0050000.mpk",
                            "python3",
                            "scripts/launch_hf_job.py",
                            "jobs/train_so100_warmstart.yaml",
                            "--dry-run",
                            "--allow-approval-required",
                        ],
                    ],
                    "after_human_approval": [
                        [
                            "LEWM_PUSHT_WARMSTART_MPK=train/REPLACE_WITH_COMPATIBLE_BOUNDED_RUN/step_0050000.mpk",
                            "scripts/launch_hf_job.py",
                            "jobs/train_so100_warmstart.yaml",
                            "--allow-approval-required",
                        ]
                    ],
                },
                "acceptance": ["warm-start artifacts and delta uploaded"],
            },
        ],
    }


def blocker_payload(statuses: dict[str, str] | None = None) -> dict[str, object]:
    statuses = statuses or {}
    return {
        "schema_version": "1.0.0",
        "updated": "2026-05-18",
        "blockers": [
            {
                "id": "F1",
                "issue": 243,
                "phase": "A",
                "status": statuses.get("F1", "blocked"),
                "evidence": ["reports/phase_a_handoff.json"],
                "required_resolution": ["resolve"],
            },
            {
                "id": "F3",
                "issue": 245,
                "phase": "A",
                "status": statuses.get("F3", "blocked"),
                "evidence": ["reports/phase_a_handoff.json"],
                "required_resolution": ["resolve"],
            },
        ],
    }


def run_check(path: Path, blockers: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT), "--path", str(path)]
    if blockers is not None:
        command.extend(["--blockers", str(blockers)])
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_valid_phase_a_handoff_passes(tmp_path: Path) -> None:
    handoff = tmp_path / "phase_a_handoff.json"
    handoff.write_text(json.dumps(handoff_payload()), encoding="utf-8")

    result = run_check(handoff)

    assert result.returncode == 0
    assert "Phase A handoff ok" in result.stdout


def test_rejects_f1_without_legacy_rejection(tmp_path: Path) -> None:
    handoff = tmp_path / "phase_a_handoff.json"
    payload = handoff_payload()
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    f1 = tasks[0]
    assert isinstance(f1, dict)
    f1["rejected_source_prefixes"] = ["train/other-prefix-"]
    handoff.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(handoff)

    assert result.returncode == 1
    assert "must reject 'train/pusht-full-lewm-'" in result.stderr


def test_rejects_f3_without_warmstart_env(tmp_path: Path) -> None:
    handoff = tmp_path / "phase_a_handoff.json"
    payload = handoff_payload()
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    f3 = tasks[1]
    assert isinstance(f3, dict)
    f3["source_env"] = "PUSHT_SOURCE"
    handoff.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(handoff)

    assert result.returncode == 1
    assert "F3.source_env must be 'LEWM_PUSHT_WARMSTART_MPK'" in result.stderr


def test_rejects_f1_upload_in_preflight(tmp_path: Path) -> None:
    handoff = tmp_path / "phase_a_handoff.json"
    payload = handoff_payload()
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    f1 = tasks[0]
    assert isinstance(f1, dict)
    commands = f1["commands"]
    assert isinstance(commands, dict)
    preflight = commands["preflight"]
    assert isinstance(preflight, list)
    launch = preflight[1]
    assert isinstance(launch, list)
    launch.append("--upload")
    handoff.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(handoff)

    assert result.returncode == 1
    assert "F1.preflight must not contain '--upload'" in result.stderr


def test_rejects_f3_dry_run_after_human_approval(tmp_path: Path) -> None:
    handoff = tmp_path / "phase_a_handoff.json"
    payload = handoff_payload()
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    f3 = tasks[1]
    assert isinstance(f3, dict)
    commands = f3["commands"]
    assert isinstance(commands, dict)
    approval = commands["after_human_approval"]
    assert isinstance(approval, list)
    launch = approval[0]
    assert isinstance(launch, list)
    launch.append("--dry-run")
    handoff.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(handoff)

    assert result.returncode == 1
    assert "F3.after_human_approval must not contain '--dry-run'" in result.stderr


def test_rejects_resolved_phase_a_blocker(tmp_path: Path) -> None:
    handoff = tmp_path / "phase_a_handoff.json"
    blockers = tmp_path / "release_blockers.json"
    handoff.write_text(json.dumps(handoff_payload()), encoding="utf-8")
    blockers.write_text(json.dumps(blocker_payload(statuses={"F1": "resolved"})), encoding="utf-8")

    result = run_check(handoff, blockers)

    assert result.returncode == 1
    assert "F1.status must stay blocked" in result.stderr
