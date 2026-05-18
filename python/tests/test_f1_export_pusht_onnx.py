from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "f1_export_pusht_onnx.py"

spec = importlib.util.spec_from_file_location("f1_export_pusht_onnx", SCRIPT)
assert spec is not None
f1 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(f1)


def parse(*args: str):
    return f1.parse_args(list(args))


def test_hub_run_workflow_orders_download_contract_export_verify_upload() -> None:
    args = parse(
        "--run-prefix",
        "train/pusht-full-burn-jepa-20260518T120000Z",
        "--work-dir",
        "/tmp/f1",
    )

    commands = f1.workflow_commands(args)

    assert commands[0][:4] == ["hf", "download", "abdelstark/lewm-rs-pusht", "--include"]
    assert commands[0][4] == "train/pusht-full-burn-jepa-20260518T120000Z/*"
    assert commands[1][0:8] == [
        "uv",
        "run",
        "--project",
        "python",
        "--frozen",
        "--extra",
        "parity",
        "python",
    ]
    assert commands[1][-1] == "--check-contract-only"
    assert commands[1][commands[1].index("--safetensors") + 1] == (
        "/tmp/f1/hub/train/pusht-full-burn-jepa-20260518T120000Z/step_0050000.safetensors"
    )
    assert commands[2][0:8] == commands[1][0:8]
    assert commands[2][commands[2].index("--variant") + 1] == "both"
    assert commands[2][commands[2].index("--action-dim") + 1] == "10"
    assert commands[2][commands[2].index("--meta") + 1].endswith(
        "tests/fixtures/reference_model.meta.json"
    )
    assert commands[2][commands[2].index("--output-dir") + 1] == "/tmp/f1/onnx-full"
    assert commands[3][-2:] == ["--dir", "/tmp/f1/onnx-full"]
    assert commands[4][0:6] == ["uv", "run", "--project", "python", "--frozen", "python"]
    assert commands[4][-1] == "--dry-run"
    assert "onnx-full/" in commands[4]


def test_local_safetensors_workflow_skips_download_and_can_upload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_0050000.safetensors"
    args = parse(
        "--safetensors",
        str(checkpoint),
        "--output-dir",
        str(tmp_path / "onnx-full"),
        "--upload",
    )

    commands = f1.workflow_commands(args)

    assert commands[0][8] == "python/export_onnx.py"
    assert commands[0][commands[0].index("--safetensors") + 1] == str(checkpoint)
    assert commands[-1][6] == "python/upload_checkpoints.py"
    assert "--dry-run" not in commands[-1]


def test_hub_run_rejects_legacy_bounded_prefix() -> None:
    args = parse("--run-prefix", "train/pusht-full-lewm-20260515T100908Z")

    try:
        f1.workflow_commands(args)
    except ValueError as exc:
        assert "legacy bounded PushT artifact family" in str(exc)
    else:
        raise AssertionError("expected legacy bounded PushT run prefix to fail")


def test_hub_run_rejects_glob_prefix() -> None:
    args = parse("--run-prefix", "train/pusht-full-burn-jepa-*")

    try:
        f1.workflow_commands(args)
    except ValueError as exc:
        assert "literal Hub directory" in str(exc)
    else:
        raise AssertionError("expected globbed PushT run prefix to fail")


def test_step_file_name_uses_release_width() -> None:
    assert f1.step_file_name(50_000) == "step_0050000.safetensors"


def test_step_file_name_rejects_non_positive_step() -> None:
    try:
        f1.step_file_name(0)
    except ValueError as exc:
        assert "--step must be positive" in str(exc)
    else:
        raise AssertionError("expected non-positive step to fail")
