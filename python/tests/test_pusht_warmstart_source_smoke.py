from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "pusht_warmstart_source_smoke.py"
EXPECTED_PARAM_COUNT = 41_856

spec = importlib.util.spec_from_file_location("pusht_warmstart_source_smoke", SCRIPT)
assert spec is not None
smoke = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(smoke)


def write_checkpoint(path: Path) -> None:
    payload = {
        "schema_version": "1.1.0",
        "kind": "lewm-rs-pusht-bounded-module-lewm-record",
        "step": 1,
        "params": [0.0] * EXPECTED_PARAM_COUNT,
        "adamw_params": [{}] * EXPECTED_PARAM_COUNT,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_train_command_uses_bounded_pusht_mode(tmp_path: Path) -> None:
    command = smoke.train_command(ROOT / "configs/pusht.toml", tmp_path, 1)

    assert command[:7] == ["cargo", "run", "-p", "lewm-train", "--bin", "lewm-train", "--"]
    assert command[command.index("--config") + 1] == str(ROOT / "configs/pusht.toml")
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--max-steps") + 1] == "1"
    assert "full_burn_jepa" not in " ".join(command)
    assert command[-1] == "train"


def test_source_check_command_uses_warmstart_verifier(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_0000001.mpk"
    command = smoke.source_check_command(checkpoint, ROOT / "configs/pusht.toml")

    assert command == [
        "python3",
        "scripts/check_warmstart_source.py",
        "--path",
        str(checkpoint),
        "--config",
        str(ROOT / "configs/pusht.toml"),
    ]


def test_parse_train_output_extracts_bounded_summary() -> None:
    summary = smoke.parse_train_output(
        "train artifacts written to /tmp/out; "
        "mode=pusht-bounded-module-lewm; "
        "data_source=pusht-compatible-fixture:128-samples:16x16; "
        "final_loss=0.49996439; "
        "checkpoint_step=1; "
        "checkpoint_complete=true"
    )

    assert summary == {
        "mode": "pusht-bounded-module-lewm",
        "data_source": "pusht-compatible-fixture:128-samples:16x16",
        "final_loss": 0.49996439,
        "checkpoint_step": 1,
        "checkpoint_complete": True,
    }


def test_parse_source_check_output_extracts_counts() -> None:
    summary = smoke.parse_source_check_output(
        "warm-start source ok: path=/tmp/out/step_0000001.mpk step=1 params=41856"
    )

    assert summary == {
        "path": "/tmp/out/step_0000001.mpk",
        "step": 1,
        "params": EXPECTED_PARAM_COUNT,
    }


def test_write_report_records_source_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_0000001.mpk"
    write_checkpoint(checkpoint)
    report = tmp_path / "reports" / "smoke.json"
    train = subprocess.CompletedProcess(
        args=["cargo", "run"],
        returncode=0,
        stdout=(
            "train artifacts written to /tmp/out; "
            "mode=pusht-bounded-module-lewm; "
            "data_source=pusht-compatible-fixture:128-samples:16x16; "
            "final_loss=0.5; "
            "checkpoint_step=1; "
            "checkpoint_complete=true"
        ),
        stderr="",
    )
    source_check = subprocess.CompletedProcess(
        args=["python3", "scripts/check_warmstart_source.py"],
        returncode=0,
        stdout="warm-start source ok: path=/tmp/out/step_0000001.mpk step=1 params=41856\n",
        stderr="",
    )

    smoke.write_report(
        report,
        config=ROOT / "configs/pusht.toml",
        output_dir=tmp_path,
        steps=1,
        checkpoint=checkpoint,
        train=train,
        source_check=source_check,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0.0"
    assert payload["steps"] == 1
    assert payload["record"]["schema_version"] == "1.1.0"
    assert payload["record"]["kind"] == "lewm-rs-pusht-bounded-module-lewm-record"
    assert payload["record"]["params"] == EXPECTED_PARAM_COUNT
    assert payload["source_check"]["params"] == EXPECTED_PARAM_COUNT
