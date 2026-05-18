from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_full_pusht_contract_smoke_report.py"


def report_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "generated_at": "2026-05-18T11:11:53+00:00",
        "config": "configs/pusht.toml",
        "output_dir": "/tmp/lewm-pusht-full-contract",
        "steps": 1,
        "checkpoint": "/tmp/lewm-pusht-full-contract/step_0000001.safetensors",
        "checkpoint_size_bytes": 72_195_968,
        "train_command": [
            "cargo",
            "run",
            "-p",
            "lewm-train",
            "--bin",
            "lewm-train",
            "--",
            "--config",
            "configs/pusht.toml",
            "--set",
            'experimental.pusht_train_mode="full_burn_jepa"',
            "--device",
            "cpu",
            "--output-dir",
            "/tmp/lewm-pusht-full-contract",
            "--max-steps",
            "1",
            "train",
        ],
        "contract_command": [
            "uv",
            "run",
            "--project",
            "python",
            "--frozen",
            "python",
            "python/export_onnx.py",
            "--safetensors",
            "/tmp/lewm-pusht-full-contract/step_0000001.safetensors",
            "--check-contract-only",
        ],
        "contract": {
            "recovered_pytorch_keys": 303,
            "expected_pytorch_keys": 303,
            "burn_destination_tensors": 255,
            "safetensors_sha256": "a" * 64,
        },
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


def test_valid_full_pusht_contract_smoke_report_passes(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(report_payload()), encoding="utf-8")

    result = run_check(report)

    assert result.returncode == 0
    assert "full PushT contract smoke report ok" in result.stdout


def test_rejects_bounded_sized_smoke_report(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(report_payload(checkpoint_size_bytes=1_264)),
        encoding="utf-8",
    )

    result = run_check(report)

    assert result.returncode == 1
    assert "too small for full Burn/Jepa" in result.stderr


def test_rejects_incomplete_contract_counts(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    payload = report_payload()
    payload["contract"] = {
        "recovered_pytorch_keys": 0,
        "expected_pytorch_keys": 303,
        "burn_destination_tensors": 14,
        "safetensors_sha256": "a" * 64,
    }
    report.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(report)

    assert result.returncode == 1
    assert "expected 303/303 recovered PyTorch keys" in result.stderr
