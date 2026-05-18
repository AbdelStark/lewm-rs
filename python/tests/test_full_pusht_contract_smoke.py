from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "full_pusht_contract_smoke.py"

spec = importlib.util.spec_from_file_location("full_pusht_contract_smoke", SCRIPT)
assert spec is not None
smoke = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(smoke)


def test_train_command_selects_release_full_burn_jepa_contract(tmp_path: Path) -> None:
    command = smoke.train_command(ROOT / "configs/pusht.toml", tmp_path, 1)

    assert command[:7] == ["cargo", "run", "-p", "lewm-train", "--bin", "lewm-train", "--"]
    assert command[command.index("--config") + 1] == str(ROOT / "configs/pusht.toml")
    assert command[command.index("--set") + 1] == 'experimental.pusht_train_mode="full_burn_jepa"'
    assert command[command.index("--device") + 1] == "cpu"
    assert command[command.index("--max-steps") + 1] == "1"
    assert command[-1] == "train"


def test_contract_command_uses_frozen_python_exporter(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_0000001.safetensors"
    command = smoke.contract_command(checkpoint)

    assert command == [
        "uv",
        "run",
        "--project",
        "python",
        "--frozen",
        "python",
        "python/export_onnx.py",
        "--safetensors",
        str(checkpoint),
        "--check-contract-only",
    ]


def test_checkpoint_path_uses_train_step_width(tmp_path: Path) -> None:
    assert smoke.checkpoint_path(tmp_path, 50_000) == tmp_path / "step_0050000.safetensors"
