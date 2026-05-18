from __future__ import annotations

import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import export_onnx as export  # noqa: E402


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
