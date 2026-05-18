from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import export_onnx as export  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def test_destination_key_fixture_matches_export_map() -> None:
    fixture = ROOT / "tests" / "fixtures" / "onnx_export_destination_keys.txt"
    expected = [line for line in fixture.read_text(encoding="utf-8").splitlines() if line]

    assert len(expected) == export.pnm.REFERENCE_DESTINATION_TENSOR_COUNT
    assert expected == list(export.pnm.expected_destination_keys())


def test_selected_variants_expands_both() -> None:
    assert export.selected_variants("both") == ("onnxruntime", "tract-compat")
    assert export.selected_variants("tract-compat") == ("tract-compat",)


def test_parse_step_count_from_safetensors_name() -> None:
    assert export.parse_step_count(Path("step_0050000.safetensors")) == 50_000
    assert export.parse_step_count(Path("weights.safetensors")) is None


def test_metadata_records_variant_layout_and_core_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_0050000.safetensors"
    checkpoint.write_bytes(b"weights")
    arch = {
        "encoder": {"image_size": 224},
        "predictor": {"num_frames": 3, "input_dim": 192},
    }

    metadata = export.build_metadata(
        safetensors=checkpoint,
        output_dir=tmp_path / "onnx",
        arch=arch,
        action_dim=10,
        variants=("onnxruntime", "tract-compat"),
        export_timestamp="2026-05-18T00:00:00Z",
    )

    assert metadata["step_count"] == 50_000
    assert metadata["config"]["action_dim"] == 10
    assert metadata["variants"]["onnxruntime"] == {
        "opset_version": 18,
        "dynamic_batch": True,
        "encoder": "onnxruntime/encoder.onnx",
        "predictor": "onnxruntime/predictor.onnx",
    }
    assert metadata["variants"]["tract-compat"] == {
        "opset_version": 17,
        "dynamic_batch": False,
        "encoder": "tract-compat/encoder.onnx",
        "predictor": "tract-compat/predictor.onnx",
    }


def test_checkpoint_contract_accepts_complete_recovered_key_set() -> None:
    expected = set(export.pnm.expected_source_keys())

    assert export.checkpoint_contract_error(set(), expected) is None


def test_checkpoint_contract_reports_bounded_core_artifact() -> None:
    burn_keys = {
        "action_encoder.bias",
        "action_encoder.x.weight",
        "action_encoder.y.weight",
        "encoder.bias",
        "encoder.energy.weight",
        "encoder.pixel.weight",
        "encoder.time.weight",
        "pred_proj.bias",
        "pred_proj.weight",
        "predictor.action.weight",
        "predictor.bias",
        "predictor.latent.weight",
        "projector.bias",
        "projector.weight",
    }

    message = export.checkpoint_contract_error(burn_keys, set())

    assert message is not None
    assert "recovered 0 of" in message
    assert "bounded PushtFullLewmCore training artifact" in message
    assert "full 303-tensor lewm_core::Jepa checkpoint" in message


def test_recover_pytorch_numpy_reports_bounded_core_before_torch(tmp_path: Path) -> None:
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    checkpoint = tmp_path / "step_0050000.safetensors"
    safetensors_numpy.save_file(
        {
            key: np.array([0.0], dtype=np.float32)
            for key in {
                "action_encoder.bias",
                "action_encoder.x.weight",
                "action_encoder.y.weight",
                "encoder.bias",
                "encoder.energy.weight",
                "encoder.pixel.weight",
                "encoder.time.weight",
                "pred_proj.bias",
                "pred_proj.weight",
                "predictor.action.weight",
                "predictor.bias",
                "predictor.latent.weight",
                "projector.bias",
                "projector.weight",
            }
        },
        checkpoint,
    )

    with pytest.raises(export.CheckpointContractError) as exc_info:
        export.recover_pytorch_numpy_from_burn(checkpoint)

    message = str(exc_info.value)
    assert "source safetensors tensor count: 14" in message
    assert "bounded PushtFullLewmCore training artifact" in message


def test_check_contract_only_accepts_full_layout_without_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    safetensors_numpy = pytest.importorskip("safetensors.numpy")
    checkpoint = tmp_path / "step_0050000.safetensors"
    tensors = {}
    for rule in export.pnm.parameter_rules():
        if rule.transform == export.pnm.Transform.QKV_LINEAR_CONCAT_TRANSPOSE:
            tensors[rule.destination] = np.zeros((1, len(rule.sources)), dtype=np.float32)
        elif rule.transform == export.pnm.Transform.QKV_BIAS_CONCAT:
            tensors[rule.destination] = np.zeros((len(rule.sources),), dtype=np.float32)
        elif rule.transform == export.pnm.Transform.LINEAR_TRANSPOSE:
            tensors[rule.destination] = np.zeros((1, 1), dtype=np.float32)
        else:
            tensors[rule.destination] = np.zeros((1,), dtype=np.float32)
    safetensors_numpy.save_file(tensors, checkpoint)
    monkeypatch.setattr(export, "_TORCH_OK", False)
    monkeypatch.setattr(export, "torch", None)

    result = export.main(["--safetensors", str(checkpoint), "--check-contract-only"])

    assert result == 0
    captured = capsys.readouterr()
    assert "Checkpoint contract ok: recovered 303 of 303 expected PyTorch keys" in captured.out
    assert "Safetensors SHA-256:" in captured.out


def test_export_requires_output_dir_unless_check_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    checkpoint = tmp_path / "step_0050000.safetensors"
    checkpoint.write_bytes(b"not used")

    result = export.main(["--safetensors", str(checkpoint)])

    assert result == 1
    assert "--output-dir is required unless --check-contract-only is set" in capsys.readouterr().err
