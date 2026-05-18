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
        "reports/pusht_full_safetensors_hub_audit.json",
        "reports/f1_source_build_dry_run.json",
        "scripts/check_f1_source_build_dry_run_report.py",
        "jobs/train_pusht_source.yaml",
        "scripts/f1_export_pusht_onnx.py",
        "scripts/audit_pusht_full_safetensors.py",
        "scripts/check_pusht_full_safetensors_hub_audit_report.py",
        "scripts/verify_runtime_image.py",
        ".github/workflows/runtime-image.yml",
        "reports/phase_a_handoff.json",
        "reports/phase_a_approval.json",
        "scripts/check_phase_a_approval.py",
        "reports/runtime_image_publish.md",
    ],
    "F3": [
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
    ],
    "F13": [
        "conformance/release_blockers.json",
    ],
    "F11": [
        ".github/workflows/runtime-image.yml",
        "reports/runtime_image_publish.md",
    ],
}
REQUIRED_RESOLUTION_BY_ID = {
    "F1": [
        "Publish a concrete non-latest GHCR runtime image tag from the intended git commit.",
        "Verify the runtime image tag with scripts/verify_runtime_image.py before paid HF Job launch, or use the approval-gated source-build fallback with a concrete LEWM_SOURCE_REVISION.",
        "Produce and upload a PushT checkpoint with the exact 255-tensor Burn/Jepa safetensors layout expected by python/export_onnx.py.",
        "Export both onnxruntime and tract-compat variants under onnx-full/.",
    ],
    "F11": [
        "Grant write access to the ghcr.io/abdelstark/lewm-rs package settings.",
        "Trigger release.yml and verify the container job passes.",
        "Verify the latest GHCR image is published and signed.",
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
                "required_resolution": [
                    "resolve",
                    *REQUIRED_RESOLUTION_BY_ID.get(f"F{index}", []),
                ],
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


def test_release_blocker_manifest_requires_f1_runtime_image_resolution(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "release_blockers.json"
    payload = blocker_manifest()
    blockers = payload["blockers"]
    assert isinstance(blockers, list)
    f1 = blockers[0]
    assert isinstance(f1, dict)
    f1["required_resolution"] = ["resolve"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(manifest)

    assert result.returncode == 1
    assert "F1 required_resolution missing required text" in result.stderr
    assert "scripts/verify_runtime_image.py" in result.stderr
