from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "python" / "upload_checkpoints.py"


def run_upload(*args: str, path: str = "") -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env["PATH"] = path
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_validates_onnx_full_upload_without_hf_token_or_cli(tmp_path: Path) -> None:
    export_dir = tmp_path / "onnx-full"
    export_dir.mkdir()
    (export_dir / "onnx_export.json").write_text("{}", encoding="utf-8")

    result = run_upload(
        "--src",
        str(export_dir),
        "--dst",
        "abdelstark/lewm-rs-pusht",
        "--path-prefix",
        "onnx-full/",
        "--dry-run",
    )

    assert result.returncode == 0
    assert "hf upload" in result.stdout
    assert "abdelstark/lewm-rs-pusht" in result.stdout
    assert "onnx-full/" in result.stdout
    assert "HF_TOKEN" not in result.stderr
    assert "hf CLI" not in result.stderr


def test_real_upload_still_requires_hf_cli(tmp_path: Path) -> None:
    export_dir = tmp_path / "onnx-full"
    export_dir.mkdir()
    (export_dir / "onnx_export.json").write_text("{}", encoding="utf-8")

    result = run_upload(
        "--src",
        str(export_dir),
        "--dst",
        "abdelstark/lewm-rs-pusht",
        "--path-prefix",
        "onnx-full/",
    )

    assert result.returncode == 2
    assert "hf CLI is required in PATH" in result.stderr


def test_dry_run_rejects_empty_directory(tmp_path: Path) -> None:
    result = run_upload(
        "--src",
        str(tmp_path),
        "--dst",
        "abdelstark/lewm-rs-pusht",
        "--path-prefix",
        "onnx-full/",
        "--dry-run",
    )

    assert result.returncode == 2
    assert "contains no uploadable files" in result.stderr
