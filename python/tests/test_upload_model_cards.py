from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "upload_model_cards.py"


def run_upload_cards(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_does_not_require_hf_token() -> None:
    result = run_upload_cards("all", "--dry-run")

    assert result.returncode == 0
    assert "[pusht] DRY RUN" in result.stdout
    assert "[so100] DRY RUN" in result.stdout
    assert "HF_TOKEN" not in result.stderr


def test_upload_requires_hf_token() -> None:
    result = run_upload_cards("pusht")

    assert result.returncode == 1
    assert "HF_TOKEN environment variable required" in result.stderr
