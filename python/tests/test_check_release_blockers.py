from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_release_blockers.py"


def blocker_manifest(evidence: str = "README.md") -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "updated": "2026-05-18",
        "blockers": [
            {
                "id": f"F{index}",
                "issue": 242 + index,
                "phase": "test",
                "title": f"Blocker {index}",
                "status": "blocked",
                "evidence": [evidence],
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
