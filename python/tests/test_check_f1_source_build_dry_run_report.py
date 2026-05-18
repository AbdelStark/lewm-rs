from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_f1_source_build_dry_run_report.py"
REVISION = "a" * 40


def report_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "updated": "2026-05-18",
        "task": "F1",
        "issue": 243,
        "status": "preflight_passed_pending_human_approval",
        "job": "jobs/train_pusht_source.yaml",
        "source_revision": REVISION,
        "source_revision_note": "Dry-run source revision selected from pushed main.",
        "dry_run_command": [
            f"LEWM_SOURCE_REVISION={REVISION}",
            "python3",
            "scripts/launch_hf_job.py",
            "jobs/train_pusht_source.yaml",
            "--dry-run",
            "--allow-approval-required",
        ],
        "approval_command": [
            f"LEWM_SOURCE_REVISION={REVISION}",
            "scripts/launch_hf_job.py",
            "jobs/train_pusht_source.yaml",
            "--allow-approval-required",
        ],
        "result": "passed",
        "launched_paid_job": False,
        "uploaded_artifacts": False,
        "cost_cap_usd": "20.00",
        "worst_case_usd": "18.00",
        "rendered_command_checks": [
            "hf jobs run",
            "--namespace abdelstark",
            "--flavor a10g-large",
            "--timeout 12h",
            f"--env LEWM_SOURCE_REVISION={REVISION}",
            "rust:1.95.0-bookworm",
            "TRACKIO_RUN=pusht-full-burn-jepa-source",
            "numpy==2.4.4",
            'experimental.pusht_train_mode="full_burn_jepa"',
            "python/export_onnx.py",
            "--check-contract-only",
            "python/upload_checkpoints.py",
            "--path-prefix train/pusht-full-burn-jepa-",
        ],
        "blocked_on": [
            "Explicit human approval for the paid 12h A10G-large source-build launch.",
            "Successful upload under train/pusht-full-burn-jepa-YYYYMMDDTHHMMSSZ/.",
            "Post-run F1 ONNX export, verification, and upload under onnx-full/.",
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


def test_valid_f1_source_build_dry_run_report_passes(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(report_payload()), encoding="utf-8")

    result = run_check(report)

    assert result.returncode == 0
    assert "F1 source-build dry-run report ok" in result.stdout


def test_rejects_placeholder_revision(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    payload = report_payload(source_revision="REPLACE_WITH_SOURCE_REVISION")
    payload["dry_run_command"] = [
        "LEWM_SOURCE_REVISION=REPLACE_WITH_SOURCE_REVISION",
        "python3",
        "scripts/launch_hf_job.py",
        "jobs/train_pusht_source.yaml",
        "--dry-run",
        "--allow-approval-required",
    ]
    report.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(report)

    assert result.returncode == 1
    assert "source_revision must be a full lowercase git SHA" in result.stderr


def test_rejects_approval_command_with_dry_run(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    payload = report_payload()
    approval = payload["approval_command"]
    assert isinstance(approval, list)
    payload["approval_command"] = [*approval, "--dry-run"]
    report.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(report)

    assert result.returncode == 1
    assert "approval_command must not include --dry-run" in result.stderr


def test_rejects_uploaded_artifact_claim(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(report_payload(uploaded_artifacts=True)), encoding="utf-8")

    result = run_check(report)

    assert result.returncode == 1
    assert "uploaded_artifacts must be false" in result.stderr


def test_rejects_wrong_cost(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(report_payload(worst_case_usd="9.00")), encoding="utf-8")

    result = run_check(report)

    assert result.returncode == 1
    assert "worst_case_usd must match jobs/train_pusht_source.yaml cost 18.00" in (
        result.stderr
    )


def test_rejects_missing_numpy_runtime_dependency(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    payload = report_payload()
    rendered = payload["rendered_command_checks"]
    assert isinstance(rendered, list)
    payload["rendered_command_checks"] = [item for item in rendered if item != "numpy==2.4.4"]
    report.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(report)

    assert result.returncode == 1
    assert "rendered_command_checks missing 'numpy==2.4.4'" in result.stderr


def test_rejects_rendered_token_missing_from_actual_dry_run(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    payload = report_payload()
    rendered = payload["rendered_command_checks"]
    assert isinstance(rendered, list)
    payload["rendered_command_checks"] = [*rendered, "not-present-in-dry-run"]
    report.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(report)

    assert result.returncode == 1
    assert "dry_run_command output missing rendered token 'not-present-in-dry-run'" in result.stderr
