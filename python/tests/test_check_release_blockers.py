from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_release_blockers.py"
REQUIRED_EVIDENCE_BY_ID = {
    "F1": [
        "reports/full_burn_jepa_training_gap.md",
        "reports/full_pusht_contract_smoke.json",
        "scripts/f1_export_pusht_onnx.py",
    ],
    "F3": [
        "jobs/train_so100_warmstart.yaml",
        ".ml-intern/cli_agent_config.json",
        "reports/pusht_warmstart_source_smoke.json",
        "scripts/pusht_warmstart_source_smoke.py",
        "scripts/check_pusht_warmstart_source_smoke_report.py",
    ],
    "F13": [
        "conformance/release_blockers.json",
    ],
}


def blocker_manifest(
    evidence: str = "README.md",
    statuses: dict[str, str] | None = None,
    include_required_evidence: bool = True,
) -> dict[str, object]:
    statuses = statuses or {}
    return {
        "schema_version": "1.0.0",
        "updated": "2026-05-18",
        "blockers": [
            {
                "id": f"F{index}",
                "issue": 242 + index,
                "phase": "test",
                "title": f"Blocker {index}",
                "status": statuses.get(f"F{index}", "blocked"),
                "evidence": [
                    evidence,
                    *(
                        REQUIRED_EVIDENCE_BY_ID.get(f"F{index}", [])
                        if include_required_evidence
                        else []
                    ),
                ],
                "required_resolution": ["resolve"],
            }
            for index in range(1, 14)
        ],
    }


def run_check(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--path", str(path), "--allow-open"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_release_blocker_manifest_requires_existing_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "release_blockers.json"
    manifest.write_text(json.dumps(blocker_manifest("missing/evidence.md")), encoding="utf-8")

    result = run_check(manifest)

    assert result.returncode == 1
    assert "F1.evidence 'missing/evidence.md' does not exist" in result.stderr


def test_release_blocker_manifest_accepts_repo_relative_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "release_blockers.json"
    manifest.write_text(json.dumps(blocker_manifest("README.md")), encoding="utf-8")

    result = run_check(manifest)

    assert result.returncode == 0
    assert "release blocker check ok: 13 blocker(s), 13 open" in result.stdout


def test_release_blocker_manifest_rejects_resolved_blocker_with_open_dependency(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "release_blockers.json"
    manifest.write_text(
        json.dumps(
            blocker_manifest(statuses={"F1": "blocked", "F2": "resolved", "F3": "resolved"})
        ),
        encoding="utf-8",
    )

    result = run_check(manifest)

    assert result.returncode == 1
    assert "F2 cannot be resolved while F1 is blocked" in result.stderr


def test_release_blocker_manifest_accepts_resolved_blocker_after_dependencies(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "release_blockers.json"
    manifest.write_text(
        json.dumps(
            blocker_manifest(statuses={"F1": "resolved", "F2": "resolved", "F3": "resolved"})
        ),
        encoding="utf-8",
    )

    result = run_check(manifest)

    assert result.returncode == 0
    assert "release blocker check ok: 13 blocker(s), 10 open" in result.stdout


def test_release_blocker_manifest_rejects_release_tag_before_prior_blockers(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "release_blockers.json"
    statuses = {f"F{index}": "resolved" for index in range(1, 14)}
    statuses["F12"] = "pending"
    manifest.write_text(json.dumps(blocker_manifest(statuses=statuses)), encoding="utf-8")

    result = run_check(manifest)

    assert result.returncode == 1
    assert "F13 cannot be resolved while F12 is pending" in result.stderr


def test_release_blocker_manifest_requires_phase_a_gate_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "release_blockers.json"
    manifest.write_text(
        json.dumps(blocker_manifest(include_required_evidence=False)),
        encoding="utf-8",
    )

    result = run_check(manifest)

    assert result.returncode == 1
    assert "F1 evidence missing required path(s)" in result.stderr
    assert "scripts/f1_export_pusht_onnx.py" in result.stderr
