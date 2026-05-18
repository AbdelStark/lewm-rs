from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_warmstart_source.py"
EXPECTED_PARAM_COUNT = 41_856


def write_record(path: Path, **updates: object) -> None:
    payload: dict[str, object] = {
        "schema_version": "1.1.0",
        "kind": "lewm-rs-pusht-bounded-module-lewm-record",
        "step": 50_000,
        "params": [0.0] * EXPECTED_PARAM_COUNT,
        "adamw_params": [{}] * EXPECTED_PARAM_COUNT,
    }
    payload.update(updates)
    path.write_text(json.dumps(payload), encoding="utf-8")


def run_check(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--path", str(path), "--config", "configs/pusht.toml"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_valid_bounded_pusht_source_passes(tmp_path: Path) -> None:
    source = tmp_path / "step_0050000.mpk"
    write_record(source)

    result = run_check(source)

    assert result.returncode == 0
    assert "warm-start source ok" in result.stdout
    assert f"params={EXPECTED_PARAM_COUNT}" in result.stdout


def test_rejects_stale_minimal_record(tmp_path: Path) -> None:
    source = tmp_path / "step_0050000.mpk"
    write_record(
        source,
        kind="lewm-rs-pusht-minimal-lewm-record",
        schema_version="1.0.0",
        params=[0.0] * 14,
    )

    result = run_check(source)

    assert result.returncode == 1
    assert "schema_version must be '1.1.0'" in result.stderr


def test_rejects_wrong_parameter_count(tmp_path: Path) -> None:
    source = tmp_path / "step_0050000.mpk"
    write_record(source, params=[0.0] * 14, adamw_params=[])

    result = run_check(source)

    assert result.returncode == 1
    assert "does not match expected bounded-core parameter count" in result.stderr
