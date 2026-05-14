from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import verify_conversion as verify  # noqa: E402


def test_verify_conversion_checks_hashes_and_parses_helper_output(
    tmp_path: Path, monkeypatch
) -> None:
    safetensors = tmp_path / "reference.safetensors"
    burn_record = tmp_path / "reference.mpk"
    meta = tmp_path / "meta.json"
    safetensors.write_bytes(b"safe")
    burn_record.write_bytes(b"mpk")
    meta.write_text(
        json.dumps(
            {
                "artifacts": {
                    "safetensors_sha256": hashlib.sha256(b"safe").hexdigest(),
                    "burn_record_sha256": hashlib.sha256(b"mpk").hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_run(command, check, text, capture_output):
        assert check is True
        assert text is True
        assert capture_output is True
        assert "--burn-record-in" in command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="reference burn record verify: tensors=255 max_abs_diff=0.00000000e0\n",
            stderr="",
        )

    monkeypatch.setattr(verify.subprocess, "run", fake_run)

    assert (
        verify.main(
            [
                "--safetensors-in",
                str(safetensors),
                "--burn-record-in",
                str(burn_record),
                "--meta",
                str(meta),
            ]
        )
        == 0
    )


def test_verify_conversion_rejects_hash_mismatch(tmp_path: Path) -> None:
    safetensors = tmp_path / "reference.safetensors"
    burn_record = tmp_path / "reference.mpk"
    meta = tmp_path / "meta.json"
    safetensors.write_bytes(b"safe")
    burn_record.write_bytes(b"mpk")
    meta.write_text(
        json.dumps(
            {
                "artifacts": {
                    "safetensors_sha256": "wrong",
                    "burn_record_sha256": hashlib.sha256(b"mpk").hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        verify.main(
            [
                "--safetensors-in",
                str(safetensors),
                "--burn-record-in",
                str(burn_record),
                "--meta",
                str(meta),
            ]
        )
    except SystemExit as exc:
        assert "safetensors sha256 mismatch" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
