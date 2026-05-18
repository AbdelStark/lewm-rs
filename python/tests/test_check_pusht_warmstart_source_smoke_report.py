from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_pusht_warmstart_source_smoke_report.py"
EXPECTED_PARAM_COUNT = 41_856


def report_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "generated_at": "2026-05-18T11:30:00+00:00",
        "config": "configs/pusht.toml",
        "output_dir": "/tmp/lewm-pusht-warmstart-source",
        "steps": 1,
        "checkpoint": "/tmp/lewm-pusht-warmstart-source/step_0000001.mpk",
        "checkpoint_size_bytes": 4_181_822,
        "checkpoint_sha256": "a" * 64,
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
            "--device",
            "cpu",
            "--output-dir",
            "/tmp/lewm-pusht-warmstart-source",
            "--max-steps",
            "1",
            "train",
        ],
        "source_check_command": [
            "python3",
            "scripts/check_warmstart_source.py",
            "--path",
            "/tmp/lewm-pusht-warmstart-source/step_0000001.mpk",
            "--config",
            "configs/pusht.toml",
        ],
        "train": {
            "mode": "pusht-bounded-module-lewm",
            "data_source": "pusht-compatible-fixture:128-samples:16x16",
            "final_loss": 0.5,
            "checkpoint_step": 1,
            "checkpoint_complete": True,
        },
        "source_check": {
            "path": "/tmp/lewm-pusht-warmstart-source/step_0000001.mpk",
            "step": 1,
            "params": EXPECTED_PARAM_COUNT,
        },
        "record": {
            "schema_version": "1.1.0",
            "kind": "lewm-rs-pusht-bounded-module-lewm-record",
            "step": 1,
            "params": EXPECTED_PARAM_COUNT,
            "adamw_params": EXPECTED_PARAM_COUNT,
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


def test_valid_pusht_warmstart_source_smoke_report_passes(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(report_payload()), encoding="utf-8")

    result = run_check(report)

    assert result.returncode == 0
    assert "PushT warm-start source smoke report ok" in result.stdout


def test_rejects_full_burn_jepa_train_command(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    payload = report_payload()
    train_command = payload["train_command"]
    assert isinstance(train_command, list)
    payload["train_command"] = [*train_command, 'experimental.pusht_train_mode="full_burn_jepa"']
    report.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(report)

    assert result.returncode == 1
    assert "must exercise bounded PushT mode" in result.stderr


def test_rejects_wrong_record_param_count(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    payload = report_payload()
    payload["record"] = {
        "schema_version": "1.1.0",
        "kind": "lewm-rs-pusht-bounded-module-lewm-record",
        "step": 1,
        "params": 14,
        "adamw_params": 14,
    }
    report.write_text(json.dumps(payload), encoding="utf-8")

    result = run_check(report)

    assert result.returncode == 1
    assert "record.params must be 41856" in result.stderr
