from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_pusht_warmstart_hub_audit_report.py"


def report_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "updated": "2026-05-18",
        "repo": "abdelstark/lewm-rs-pusht",
        "revision": "main",
        "source": "https://huggingface.co/api/models/abdelstark/lewm-rs-pusht/tree/main?recursive=1",
        "source_verifier": "scripts/check_warmstart_source.py",
        "expected": {
            "schema_version": "1.1.0",
            "kind": "lewm-rs-pusht-bounded-module-lewm-record",
            "param_count": 41_856,
        },
        "candidate_count": 1,
        "compatible_count": 0,
        "status": "blocked",
        "summary": "No public source is compatible.",
        "candidates": [
            {
                "path": "train/run/step_0050000.mpk",
                "size": 1266,
                "download_url": "https://huggingface.co/example/resolve/main/train/run/step_0050000.mpk",
                "status": "rejected",
                "reason": "train/run/step_0050000.mpk: schema_version must be '1.1.0'",
            }
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


def test_valid_hub_audit_report_passes(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report_payload()), encoding="utf-8")

    result = run_check(path)

    assert result.returncode == 0
    assert "PushT warm-start Hub audit report ok" in result.stdout


def test_rejects_candidate_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report_payload(candidate_count=2)), encoding="utf-8")

    result = run_check(path)

    assert result.returncode == 1
    assert "candidate_count does not match" in result.stderr


def test_rejects_blocked_status_with_compatible_candidate(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    payload = report_payload(compatible_count=1, status="blocked")
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    candidate = candidates[0]
    assert isinstance(candidate, dict)
    candidate["status"] = "compatible"
    candidate["reason"] = "accepted by scripts/check_warmstart_source.py"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(path)

    assert result.returncode == 1
    assert "status must be 'ready'" in result.stderr


def test_rejects_rejection_reason_without_path(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    payload = report_payload()
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    candidate = candidates[0]
    assert isinstance(candidate, dict)
    candidate["reason"] = "schema_version must be '1.1.0'"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(path)

    assert result.returncode == 1
    assert "reason must name the rejected path" in result.stderr
